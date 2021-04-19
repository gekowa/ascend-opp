# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
fake_quant_with_min_max_vars_gradient
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_conf import api_check_support
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce
from te.utils.op_utils import *

# value of default num_bits
NUM_BITS_MIN = 2
# value of max num_bits
NUM_BITS_MAX = 16
# value of min num_bits
NUM_BITS_DEFAULT = 8
# 0.5
FLOAT_HALF_VALUE = 0.5
# data type
D_TYPE = "float32"


def _less_compare_float32(data_x, data_y):
    """
    Compare data_x and data_y to determine whether data_x is less than data_y.
    If the element in data_x is less than in data_y, then return 1,
    else return 0.

    max num of float32 is 2**126
    but cce can only support 2**62, so use 62/62/2 to adaptor 126
    (2**(-126))*(2**(62))*(2**(62))*(2**(2)) = 1
    so min_value*max_value*max_value*factor_value = 1
    """
    shape_inputs = te.lang.cce.util.shape_to_list(data_x.shape)
    min_value = tvm.const(2 ** (-126), dtype=D_TYPE)
    max_value = tvm.const(2 ** 62, dtype=D_TYPE)
    factor_value = tvm.const(2 ** 2, dtype=D_TYPE)

    if api_check_support("te.lang.cce.vmaxs", data_x.dtype):
        res_sub = te.lang.cce.vsub(data_y, data_x)
        res_min = te.lang.cce.vmins(res_sub, min_value)
        res_max = te.lang.cce.vmaxs(res_min, tvm.const(0, dtype=D_TYPE))
    else:
        data_zero = te.lang.cce.vmuls(data_x, 0)
        min_value_tensor = te.lang.cce.vadds(data_zero, min_value)

        res_sub = te.lang.cce.vsub(data_y, data_x)
        res_min = te.lang.cce.vmin(res_sub, min_value_tensor)
        res_max = te.lang.cce.vmax(res_min, data_zero)

    res_max_mul = te.lang.cce.vmuls(res_max, max_value)
    res_max_mul_max = te.lang.cce.vmuls(res_max_mul, max_value)
    res = te.lang.cce.vmuls(res_max_mul_max, factor_value)

    return res


def _bool_negate(input_bool):
    """
    The value of the input tensor is 0 or 1, Negate every value then output
    """
    shape = te.lang.cce.util.shape_to_list(input_bool.shape)
    dtype = input_bool.dtype
    # broadcast 1 to a tensor of shape
    tensor_one = te.lang.cce.broadcast(1, shape, dtype)
    output_bool = te.lang.cce.vsub(tensor_one, input_bool)
    return output_bool


# pylint: disable=locally-disabled,too-many-statements,too-many-locals
def _nudged_min_max_compute(min_broadcast, max_broadcast, num_bits,
                            narrow_range):
    """
    Compute gradients for a FakeQuantWithMinMaxVars operation.
    quant_max = 2^num_bits - 1
    """
    dtype = min_broadcast.dtype
    if narrow_range is False:
        quant_min = 0
    else:
        quant_min = 1
    quant_max = 2 ** num_bits - 1

    tensor_zero = te.lang.cce.vmuls(min_broadcast, tvm.const(0, dtype))
    quant_min_float = te.lang.cce.vadds(tensor_zero,
                                        tvm.const(quant_min, dtype))
    quant_max_float = te.lang.cce.vadds(tensor_zero,
                                        tvm.const(quant_max, dtype))
    max_sub_min = te.lang.cce.vsub(max_broadcast, min_broadcast)
    quant_max_sub_quant_min = te.lang.cce.vsub(quant_max_float, quant_min_float)
    scale = te.lang.cce.vdiv(max_sub_min, quant_max_sub_quant_min)
    min_div_scale = te.lang.cce.vdiv(min_broadcast, scale)

    zero_point_from_min = te.lang.cce.vsub(quant_min_float, min_div_scale)
    bool_less_quant_min_float = _less_compare_float32(zero_point_from_min,
                                                      quant_min_float)
    bool_more_quant_max_float = _less_compare_float32(quant_max_float,
                                                      zero_point_from_min)
    less_quant_min_float = te.lang.cce.vmul(quant_min_float,
                                            bool_less_quant_min_float)
    more_quant_max_float = te.lang.cce.vmul(quant_max_float,
                                            bool_more_quant_max_float)
    bool_not_less_quant_min_float = _bool_negate(bool_less_quant_min_float)
    bool_not_more_quant_max_float = _bool_negate(bool_more_quant_max_float)

    bool_between_min_max = te.lang.cce.vmul(bool_not_less_quant_min_float,
                                            bool_not_more_quant_max_float)
    between_min_max_float = te.lang.cce.vmul(zero_point_from_min,
                                             bool_between_min_max)
    # use DSL floor(x+0.5) to implement the round(x) function of the tf
    between_min_max_add_half_one = te.lang.cce.vadds(between_min_max_float,
                                                     tvm.const(FLOAT_HALF_VALUE,
                                                               dtype))
    between_min_max_round = te.lang.cce.floor(between_min_max_add_half_one)
    nudged_zero_point_tensor = te.lang.cce.vadd(less_quant_min_float,
                                                more_quant_max_float)
    nudged_zero_point = te.lang.cce.vadd(nudged_zero_point_tensor,
                                         between_min_max_round)

    nudged_min_tensor = te.lang.cce.vsub(quant_min_float, nudged_zero_point)
    nudged_min = te.lang.cce.vmul(nudged_min_tensor, scale)

    tensor_zero_second = te.lang.cce.vmuls(min_broadcast, tvm.const(0, dtype))
    quant_min_float_second = te.lang.cce.vadds(tensor_zero_second,
                                               tvm.const(quant_min, dtype))
    quant_max_float_second = te.lang.cce.vadds(tensor_zero_second,
                                               tvm.const(quant_max, dtype))
    max_sub_min_second = te.lang.cce.vsub(max_broadcast, min_broadcast)
    quant_max_sub_quant_min_second = te.lang.cce.vsub(quant_max_float_second,
                                                      quant_min_float_second)
    scale_second = te.lang.cce.vdiv(max_sub_min_second,
                                    quant_max_sub_quant_min_second)
    min_div_scale_second = te.lang.cce.vdiv(min_broadcast, scale_second)

    zero_point_from_min_second = te.lang.cce.vsub(quant_min_float_second,
                                                  min_div_scale_second)
    bool_less_quant_min_second = _less_compare_float32(
        zero_point_from_min_second,
        quant_min_float_second)
    bool_more_quant_max_second = \
        _less_compare_float32(quant_max_float_second,
                              zero_point_from_min_second)
    less_quant_min_float_second = te.lang.cce.vmul(quant_min_float_second,
                                                   bool_less_quant_min_second)
    more_quant_max_float_second = te.lang.cce.vmul(quant_max_float_second,
                                                   bool_more_quant_max_second)
    bool_not_less_quant_min_second = _bool_negate(bool_less_quant_min_second)
    bool_not_more_quant_max_second = _bool_negate(bool_more_quant_max_second)
    bool_between_min_max_second = te.lang.cce.vmul(
        bool_not_less_quant_min_second,
        bool_not_more_quant_max_second)

    between_min_max_float_second = te.lang.cce.vmul(zero_point_from_min_second,
                                                    bool_between_min_max_second)
    min_max_add_half_one_second = te.lang.cce.vadds(
        between_min_max_float_second,
        tvm.const(FLOAT_HALF_VALUE, dtype))
    between_min_max_round_second = te.lang.cce.floor(
        min_max_add_half_one_second)
    nudged_zero_point_tensor_second = te.lang.cce.vadd(
        less_quant_min_float_second,
        more_quant_max_float_second)
    nudged_zero_point_second = te.lang.cce.vadd(nudged_zero_point_tensor_second,
                                                between_min_max_round_second)

    nudged_max_tensor = te.lang.cce.vsub(quant_max_float_second,
                                         nudged_zero_point_second)
    nudged_max = te.lang.cce.vmul(nudged_max_tensor, scale_second)
    res = nudged_min, nudged_max

    return res


# pylint: disable=locally-disabled,too-many-locals,invalid-name
def _between_nudged_min_max_compute(x, nudged_min, nudged_max):
    """
    Compare x with nudged_min and nudged_max.
    If the element in x is greater than nudged_min and less than nudged_max,
    then return 1, else return 0.

    max num of float32 is 2**126
    but cce can only support 2**62, so use 62/62/2 to adaptor 126
    (2**(-126))*(2**(62))*(2**(62))*(2**(2)) = 1
    """
    shape_inputs = te.lang.cce.util.shape_to_list(x.shape)
    min_value = tvm.const(2 ** (-126), dtype=D_TYPE)
    max_value = tvm.const(2 ** 62, dtype=D_TYPE)
    factor_value = tvm.const(2 ** 2, dtype=D_TYPE)

    if api_check_support("te.lang.cce.vmaxs", x.dtype):
        sub_tensor_min = te.lang.cce.vsub(x, nudged_min)
        sub_min = te.lang.cce.vadds(sub_tensor_min, min_value)
        more_nudged_min_tensor = te.lang.cce.vmaxs(sub_min, tvm.const(0, dtype=D_TYPE))

        sub_tensor_max = te.lang.cce.vsub(nudged_max, x)
        sub_max = te.lang.cce.vadds(sub_tensor_max, min_value)
        less_nudged_max_tensor = te.lang.cce.vmaxs(sub_max, tvm.const(0, dtype=D_TYPE))

        between_nudged_tensor = te.lang.cce.vmul(more_nudged_min_tensor,
                                                 less_nudged_max_tensor)
        between_nudged_element = te.lang.cce.vmins(between_nudged_tensor, min_value)
    else:
        data_zero = te.lang.cce.vmuls(x, 0)
        min_value_tensor = te.lang.cce.vadds(data_zero, min_value)

        sub_tensor_min = te.lang.cce.vsub(x, nudged_min)
        sub_min = te.lang.cce.vadds(sub_tensor_min, min_value)
        more_nudged_min_tensor = te.lang.cce.vmax(sub_min, data_zero)

        sub_tensor_max = te.lang.cce.vsub(nudged_max, x)
        sub_max = te.lang.cce.vadds(sub_tensor_max, min_value)
        less_nudged_max_tensor = te.lang.cce.vmax(sub_max, data_zero)

        between_nudged_tensor = te.lang.cce.vmul(more_nudged_min_tensor,
                                                 less_nudged_max_tensor)
        between_nudged_element = te.lang.cce.vmin(between_nudged_tensor,
                                                  min_value_tensor)

    vmul_max_value = te.lang.cce.vmuls(between_nudged_element, max_value)
    vmul_factor_value = te.lang.cce.vmuls(vmul_max_value, max_value)
    between_nudged = te.lang.cce.vmuls(vmul_factor_value, factor_value)

    return between_nudged


def _check_parameters(gradients, x, input_min, input_max, kernel_name):
    """
    Check the validity of input parameters.
    """
    # get dtype and shape attributes
    dtype_gradient = gradients.get("dtype").lower()
    shape_gradient = gradients.get("shape")
    dtype_data = x.get("dtype").lower()
    shape_data = x.get("shape")
    dtype_min = input_min.get("dtype").lower()
    shape_min = input_min.get("shape")
    dtype_max = input_max.get("dtype").lower()
    shape_max = input_max.get("shape")

    shape_min = util.scalar2tensor_one(shape_min)
    shape_max = util.scalar2tensor_one(shape_max)

    # check kernel name and shape
    check_shape(shape_gradient, param_name="gradients")
    check_shape(shape_data, param_name="x")
    check_shape(shape_min, min_rank=1, max_rank=1, param_name="min")
    check_shape(shape_max, min_rank=1, max_rank=1, param_name="max")

    # check data type of input tensor
    check_list = (D_TYPE,)
    check_dtype(dtype_gradient, check_list, param_name="gradients")
    check_dtype(dtype_data, check_list, param_name="x")
    check_dtype(dtype_min, check_list, param_name="min")
    check_dtype(dtype_max, check_list, param_name="max")

    # check whether the shape of gradients and x are the same
    if list(shape_gradient) != list(shape_data):
        raise RuntimeError("dimensions in both shapes must be equal")


def _both_min_max_zero(input_min, input_max, input_shape, dtype):
    """
    Check whether the values of min and max are both 0.
    If both value is 0, the result of the corresponding element is 0. Otherwise,
    the result is 1.
    """
    tensor_zero = te.lang.cce.broadcast(0, input_shape, dtype)
    min_broad = te.lang.cce.broadcast(input_min, input_shape, dtype)
    max_broad = te.lang.cce.broadcast(input_max, input_shape, dtype)
    min_abs = te.lang.cce.vabs(min_broad)
    max_abs = te.lang.cce.vabs(max_broad)
    min_max_add = te.lang.cce.vadd(min_abs, max_abs)

    bool_min_max_less_zero = _less_compare_float32(min_max_add, tensor_zero)
    bool_min_max_more_zero = _less_compare_float32(tensor_zero, min_max_add)
    bool_both_no_zero = te.lang.cce.vadd(bool_min_max_less_zero,
                                         bool_min_max_more_zero)

    return bool_both_no_zero


# pylint: disable=locally-disabled,unused-argument,too-many-locals,too-many-arguments
# pylint: disable=locally-disabled,redefined-builtin
@fusion_manager.register("fake_quant_with_min_max_vars_gradient")
def fake_quant_with_min_max_vars_gradient_compute(gradients, x, min,
                                                  max, backprops_wrt_x,
                                                  backprop_wrt_min,
                                                  backprop_wrt_max,
                                                  num_bits, narrow_range,
                                                  kernel_name="fake_quant_"
                                                              "with_min_max"
                                                              "_vars_gradient"):
    """
    Compute gradients for a FakeQuantWithMinMaxVars operation.

    Parameters
    ----------
    gradients: tvm.tensor
        input tensor has shape and dtype attributes
    x: tvm.tensor
        input tensor has shape and dtype attributes
    min: tvm.tensor
    max: tvm.tensor
    backprops_wrt_x: tvm.tensor
        output tensor has shape and dtype attributes
    backprop_wrt_min: tvm.tensor
        output tensor has shape and dtype attributes
    backprop_wrt_max: TVM tensor
        output tensor has shape and dtype attributes
    num_bits: int
        the bitwidth of the quantization, between 2 and 16
    narrow_range: bool
        whether to quantize into 2^num_bits - 1 distinct values
    kernel_name: str
        cce kernel name, default value is "fake_quant_with_min_max_vars_gradient"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """
    input_shape = te.lang.cce.util.shape_to_list(x.shape)
    dtype = x.dtype

    min_broadcast = te.lang.cce.broadcast(min, input_shape, dtype)
    max_broadcast = te.lang.cce.broadcast(max, input_shape, dtype)
    nudged_min, nudged_max = _nudged_min_max_compute(min_broadcast,
                                                     max_broadcast, num_bits,
                                                     narrow_range)
    nudged_min_backup = te.lang.cce.vadds(nudged_min, tvm.const(0, D_TYPE))
    nudged_max_backup = te.lang.cce.vadds(nudged_max, tvm.const(0, D_TYPE))

    between_nudged_min_max = _between_nudged_min_max_compute(x, nudged_min,
                                                             nudged_max)
    wrt_input_tensor = te.lang.cce.vmul(between_nudged_min_max, gradients)
    shape_list = []
    for i, _ in enumerate(input_shape):
        shape_list.append(i)

    bool_below_min = _less_compare_float32(x, nudged_min_backup)
    below_min_data = te.lang.cce.vmul(bool_below_min, gradients)

    bool_below_max = _less_compare_float32(nudged_max_backup, x)
    below_max_data = te.lang.cce.vmul(bool_below_max, gradients)

    # process min and max are both zero
    tensor_one = te.lang.cce.broadcast(1, input_shape, dtype)
    bool_both_no_zero = _both_min_max_zero(min, max, input_shape, dtype)

    bool_both_no_zero_reverse = te.lang.cce.vsub(tensor_one, bool_both_no_zero)
    bool_both_no_zero_broad = te.lang.cce.broadcast(bool_both_no_zero,
                                                    input_shape, dtype)
    bool_both_no_zero_reverse = te.lang.cce.broadcast(bool_both_no_zero_reverse,
                                                      input_shape, dtype)

    wrt_input_weight = te.lang.cce.vmul(wrt_input_tensor,
                                        bool_both_no_zero_broad)
    gradients_weight = te.lang.cce.vmul(gradients, bool_both_no_zero_reverse)
    backprops_wrt_x = te.lang.cce.vadd(wrt_input_weight, gradients_weight)

    # cloud version: optimize to eliminating workspace by reducing atomic
    if util.get_product_version() == util.VERSION_CLOUD:
        # insert temp node to make vadd_last node as mid_outputTensor for eliminating workspace
        temp_insert_node_mul = te.lang.cce.vmuls(backprops_wrt_x,
                                                 tvm.const(0, D_TYPE))
        temp_insert_node_add = te.lang.cce.vadd(temp_insert_node_mul,
                                                below_min_data)
        below_min_data_tensor = te.lang.cce.vmul(temp_insert_node_add,
                                                 bool_both_no_zero)
        below_max_data_tensor = te.lang.cce.vmul(below_max_data,
                                                 bool_both_no_zero)
        backprop_wrt_min_max_list = te.lang.cce.tuple_sum(
            [below_min_data_tensor,
             below_max_data_tensor],
            axis=shape_list)
        output_list = [backprops_wrt_x] + list(backprop_wrt_min_max_list)

    else:
        below_min_data_tensor = te.lang.cce.vmul(below_min_data,
                                                 bool_both_no_zero)
        below_max_data_tensor = te.lang.cce.vmul(below_max_data,
                                                 bool_both_no_zero)
        backprop_wrt_min = te.lang.cce.sum(below_min_data_tensor,
                                           axis=shape_list)
        backprop_wrt_max = te.lang.cce.sum(below_max_data_tensor,
                                           axis=shape_list)
        output_list = [backprops_wrt_x, backprop_wrt_min, backprop_wrt_max]

    return output_list


# pylint: disable=locally-disabled,redefined-builtin,invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT, OPTION_ATTR_INT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def fake_quant_with_min_max_vars_gradient(gradients, x, min, max,
                                          backprops_wrt_x, backprops_wrt_min,
                                          backprops_wrt_max,
                                          num_bits=NUM_BITS_DEFAULT,
                                          narrow_range=False,
                                          kernel_name="fake_quant_"
                                                      "with_min_max_vars_"
                                                      "gradient"):
    """
    Compute gradients for a FakeQuantWithMinMaxVars operation use
    fake_quant_with_min_max_vars_gradient_compute.

    Parameters
    ----------
    gradients: dict
        dict{"shape": tuple or list, "dtype": float32}
        shape and data type of gradients, only support float32
    x: dict
        dict{"shape": tuple or list, "dtype": float32}
        shape and data type of gradients, only support float32
    min: dict
        dict{"shape": tuple or list, "dtype": float32}
        shape and data type of min, only support float32
    max: dict
        dict{"shape": tuple or list, "dtype": float32}
        shape and data type of max, only support float32
    backprops_wrt_x: dict
        gradients of input data
    backprops_wrt_min: dict
        reverse of input min
    backprops_wrt_max: dict
        reverse of input max
    num_bits: int
        the bitwidth of the quantization, between 2 and 16, defaults value is 8
    narrow_range: bool
        whether to quantize into 2^num_bits - 1 distinct values, defaults
        value is False
    kernel_name: str
        cce kernel name, default value is
        "fake_quant_with_min_max_vars_gradient"

    Returns
    ------
    None
    """
    _check_parameters(gradients, x, min, max, kernel_name)

    if num_bits < NUM_BITS_MIN or num_bits > NUM_BITS_MAX:
        raise RuntimeError("num_bits must be between 2 and 16")

    dtype_gradient = gradients.get("dtype").lower()
    shape_gradient = gradients.get("shape")
    dtype_min = min.get("dtype").lower()
    shape_min = min.get("shape")
    shape_max = max.get("shape")
    check_shape(shape_min, param_name="min")
    check_shape(shape_max, param_name="max")
    shape_backprops_wrt_x = backprops_wrt_x.get("shape")
    shape_backprops_wrt_min = backprops_wrt_min.get("shape")
    shape_backprops_wrt_max = backprops_wrt_max.get("shape")
    shape_backprops_wrt_min = util.scalar2tensor_one(shape_backprops_wrt_min)
    shape_backprops_wrt_max = util.scalar2tensor_one(shape_backprops_wrt_max)
    check_shape(shape_backprops_wrt_x, param_name="backprops_wrt_x")
    check_shape(shape_backprops_wrt_min, param_name="backprops_wrt_min")
    check_shape(shape_backprops_wrt_max, param_name="backprops_wrt_max")
    shape_min = util.scalar2tensor_one(shape_min)
    shape_gradient = (functools_reduce(lambda x, y: x * y, shape_gradient[:]),)
    _, min_new_shape, _ = broadcast_shapes(shape_gradient, shape_min,
                                              param_name_input1="gradients",
                                              param_name_input2="min")
    gradient_data = tvm.placeholder(shape_gradient, name="gradient_data",
                                    dtype=dtype_gradient)
    x = tvm.placeholder(shape_gradient, name="x", dtype=dtype_gradient)
    min_data = tvm.placeholder(min_new_shape, name="min_data", dtype=dtype_min)
    max_data = tvm.placeholder(min_new_shape, name="max_data", dtype=dtype_min)

    res_list = fake_quant_with_min_max_vars_gradient_compute(gradient_data, x,
                                                             min_data,
                                                             max_data,
                                                             backprops_wrt_x,
                                                             backprops_wrt_min,
                                                             backprops_wrt_max,
                                                             num_bits,
                                                             narrow_range,
                                                             kernel_name)

    input_placeholders = (gradient_data, x, min_data, max_data)
    with tvm.target.cce():
        sch = generic.auto_schedule(res_list)

    config = {"name": kernel_name,
              "tensor_list": list(input_placeholders) + list(res_list)}
    te.lang.cce.cce_build_code(sch, config)
