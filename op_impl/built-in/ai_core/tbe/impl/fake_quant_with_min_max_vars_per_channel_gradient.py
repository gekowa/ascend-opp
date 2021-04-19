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
Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_conf import api_check_support
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

# define a scalar for add
HALF_ONE = 0.5
# define zero and one for broadcast
ZERO_VALUE = 0
ONE_VALUE = 1


def _less_compare_float32(data_x, data_y):
    """
    if x is less than y, then return 1, else return 0.

    Parameters:
    ----------
    data_x : TVM tensor
        tensor x
    data_y : TVM tensor
        tensor y

    Returns
    -------
    the compare result
    """
    shape_inputs = te.lang.cce.util.shape_to_list(data_x.shape)
    # minimun num of float32 2**(-126)
    min_value = tvm.const(2 ** (-126), dtype="float32")

    if api_check_support("te.lang.cce.vmaxs", data_x.dtype):
        res_sub = te.lang.cce.vsub(data_y, data_x)
        res_min = te.lang.cce.vmins(res_sub, min_value)
        res_max = te.lang.cce.vmaxs(res_min, tvm.const(0, dtype="float32"))
    else:
        data_zero = te.lang.cce.vmuls(data_x, 0)
        data_min = te.lang.cce.vadds(data_zero, min_value)
        res_sub = te.lang.cce.vsub(data_y, data_x)
        res_min = te.lang.cce.vmin(res_sub, data_min)
        res_max = te.lang.cce.vmax(res_min, data_zero)

    # max num of float32 is 2**126
    # but cce can only support 2**62, so use 50/50/26 to adaptor 126
    res_mul_first = te.lang.cce.vmuls(res_max,
                                      tvm.const(2 ** 50, dtype="float32"))
    res_mul_second = te.lang.cce.vmuls(res_mul_first,
                                       tvm.const(2 ** 50, dtype="float32"))
    res = te.lang.cce.vmuls(res_mul_second, tvm.const(2 ** 26, dtype="float32"))

    return res


# pylint: disable=locally-disabled,too-many-locals
def _less_equal_compare_float32(data_x, data_y):
    """
    if x is less than y or equal y, then return 1, else return 0.

    Parameters:
    ----------
    data_x : TVM tensor
        tensor x
    data_y : TVM tensor
        tensor y

    Returns
    -------
    the compare result
    """
    scalar_min_fp32 = tvm.const(2 ** (-126), dtype="float32")
    scalar_mul_fp32_first = tvm.const(2 ** (50), dtype="float32")
    scalar_mul_fp32_second = tvm.const(2 ** (26), dtype="float32")
    scalar_one_fp32 = tvm.const(1.0, dtype="float32")
    scalar_one_fp32_neg = scalar_one_fp32 * tvm.const(-1.0, dtype="float32")

    if api_check_support("te.lang.cce.vmaxs", data_x.dtype):
        data_max = te.lang.cce.vmax(data_x, data_y)
        data_sub = te.lang.cce.vsub(data_y, data_max)
        data_abs = te.lang.cce.vabs(data_sub)
        data_min = te.lang.cce.vmins(data_abs, scalar_min_fp32)

        data_mul = te.lang.cce.vmuls(data_min, scalar_mul_fp32_first)
        data_mul_first = te.lang.cce.vmuls(data_mul, scalar_mul_fp32_first)
        data_mul_second = te.lang.cce.vmuls(data_mul_first, scalar_mul_fp32_second)

        data_sub_first = te.lang.cce.vadds(data_mul_second, scalar_one_fp32_neg)
        data_out = te.lang.cce.vabs(data_sub_first)
    else:
        tensor_zero = te.lang.cce.vmuls(data_x, 0)
        tensor_min_fp32 = te.lang.cce.vadds(tensor_zero, scalar_min_fp32)

        data_max = te.lang.cce.vmax(data_x, data_y)
        data_sub = te.lang.cce.vsub(data_y, data_max)
        data_abs = te.lang.cce.vabs(data_sub)
        data_min = te.lang.cce.vmin(data_abs, tensor_min_fp32)

        data_mul = te.lang.cce.vmuls(data_min, scalar_mul_fp32_first)
        data_mul_first = te.lang.cce.vmuls(data_mul, scalar_mul_fp32_first)
        data_mul_second = te.lang.cce.vmuls(data_mul_first, scalar_mul_fp32_second)

        data_sub_first = te.lang.cce.vadds(data_mul_second, scalar_one_fp32_neg)
        data_out = te.lang.cce.vabs(data_sub_first)

    return data_out


# pylint: disable=locally-disabled,too-many-statements,invalid-name
def _nudged_min_max_compute(min_broadcast, max_broadcast, num_bits,
                            narrow_range):
    """
    compute nudged_min and nudged_max by input parameters

    Parameters:
    ----------
    min : TVM tensor
        tensor min has broadcast to input_data shape
    max : TVM tensor
        tensor max has broadcast to input_data shape
    num_bits: int
        An optional int,defaults to 8,shoud be range [2,16]
    narrow_range: bool
        is narrow_range or not,defaults to False

    Returns
    -------
    res: TVM tensor
        if is_nudged_min is True,return nudged_min,else return nudged_max
    """
    dtype = min_broadcast.dtype
    if narrow_range is False:
        quant_min = 0
    else:
        quant_min = 1
    quant_max = 2 ** num_bits - 1
    tensor_zero = te.lang.cce.vmuls(min_broadcast, tvm.const(ZERO_VALUE, dtype))
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
    between_min_max_add_half_one = te.lang.cce.vadds(between_min_max_float,
                                                     tvm.const(HALF_ONE, dtype))
    between_min_max_round = te.lang.cce.floor(between_min_max_add_half_one)
    nudged_zero_point_tmp = te.lang.cce.vadd(less_quant_min_float,
                                             more_quant_max_float)
    nudged_zero_point = te.lang.cce.vadd(nudged_zero_point_tmp,
                                         between_min_max_round)

    nudged_min_tmp = te.lang.cce.vsub(quant_min_float, nudged_zero_point)
    nudged_min = te.lang.cce.vmul(nudged_min_tmp, scale)

    tensor_zero_second = te.lang.cce.vmuls(min_broadcast,
                                           tvm.const(ZERO_VALUE, dtype))
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
    bool_more_quant_max_second = _less_compare_float32(quant_max_float_second,
                                                       zero_point_from_min_second)
    less_quant_min_second = te.lang.cce.vmul(quant_min_float_second,
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
        tvm.const(HALF_ONE, dtype))
    between_min_max_round_second = te.lang.cce.floor(
        min_max_add_half_one_second)
    nudged_zero_point_tmp_second = te.lang.cce.vadd(less_quant_min_second,
                                                    more_quant_max_float_second)
    nudged_zero_point_second = te.lang.cce.vadd(nudged_zero_point_tmp_second,
                                                between_min_max_round_second)

    nudged_max_tmp = te.lang.cce.vsub(quant_max_float_second,
                                      nudged_zero_point_second)
    nudged_max = te.lang.cce.vmul(nudged_max_tmp, scale_second)
    res = nudged_min, nudged_max

    return res


def _bool_both_zero_compute(juduged_min, juduged_max):
    """
    if input min and max are both zero then output_date will be all zero
    so need a juduge compute tensor

    Parameters:
    ----------
    min : TVM tensor
        tensor min
    max : TVM tensor
        tensor max

    Returns
    -------
    res : TVM tensor
        a tensor for juduge compute
    """
    dtype = juduged_min.dtype
    tensor_zero = te.lang.cce.vmuls(juduged_min, tvm.const(ZERO_VALUE, dtype))
    min_abs = te.lang.cce.vabs(juduged_min)
    max_abs = te.lang.cce.vabs(juduged_max)
    min_max_replace = te.lang.cce.vadd(min_abs, max_abs)
    bool_min_max_product_less_zero = _less_compare_float32(min_max_replace,
                                                           tensor_zero)
    bool_min_max_product_more_zero = _less_compare_float32(tensor_zero,
                                                           min_max_replace)
    bool_both_zero = te.lang.cce.vadd(bool_min_max_product_less_zero,
                                      bool_min_max_product_more_zero)
    res = bool_both_zero

    return res


def _bool_negate(input_bool):
    """
    The value of the input tensor is 0 or 1,Negate every value then output

    Parameters:
    ----------
    input_bool : TVM tensor
        tensor min

    Returns
    -------
    output_bool : TVM tensor
    """
    shape = te.lang.cce.util.shape_to_list(input_bool.shape)
    dtype = input_bool.dtype
    tensor_one = te.lang.cce.broadcast(ONE_VALUE, shape, dtype)
    output_bool = te.lang.cce.vsub(tensor_one, input_bool)

    return output_bool


# pylint: disable=locally-disabled,unused-argument,too-many-locals,too-many-arguments
# pylint: disable=locally-disabled,redefined-builtin
@fusion_manager.register("fake_quant_with_min_max_vars_per_channel_gradient")
def fake_quant_with_min_max_vars_per_channel_gradient_compute(gradients, x,
                                                              min,
                                                              max,
                                                              backprops_wrt_x,
                                                              backprop_wrt_min,
                                                              backprop_wrt_max,
                                                              num_bits=8,
                                                              narrow_range=False,
                                                              kernel_name="fake_quant_with_min_max"
                                                                          "_vars_per_channel_gradient"):
    """
    Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.

    Parameters
    ----------
    gradients: TVM tensor
        input tensor has shape and dtype attributes
        shape, gradients_shape equals input_data_shape,
        dtype, gradients_dtype equals input_data_dtype, only support fp32
    x: TVM tensor
        input tensor has shape and dtype attributes
        shape, input_data_shape equals output_data_shape,
        dtype, input_data_dtype equals output_data_dtype, only support fp32
    min: TVM tensor
        input tensor has shape and dtype attributes
        shape of min,min shape equals to max shape
        The shapes of min,max and shape_inputs last one dimension shoud be same
        the min data type,only support fp32
    max: TVM tensor
        input tensor has shape and dtype attributes
        shape of max,min shape equals to max shape
        The shapes of min,max and shape_inputs last one dimension shoud be same
        the max data type,only support fp32
    num_bits: int
        An optional int,defaults to 8,shoud be range [2,16]
    narrow_range: bool
        is narrow_range or not,defaults to False
    backprops_wrt_x: dict
        dict with keys(shape and dtype) of backprops output
    backprop_wrt_min : dict
        dict with keys(shape and dtype) of backprop min
    backprop_wrt_max : dict
        dict with keys(shape and dtype) of backprop max
    kernel_name: str
        cce kernel name, default value is
        "fake_quant_with_min_max_vars_per_channel"

    Returns
    ------
    res: tensor list
        [backprops_wrt_input, backprop_wrt_min, backprop_wrt_max]
        output tensor has shape and dtype attributes
        shape, output_data_shape equals input_data_shape
        dtype, output_data_dtype equals input_data_dtype, only support fp32
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    sum_axis = []
    shape_len = len(shape)
    for i in range(0, shape_len - 1):
        sum_axis.append(i)
    dtype = x.dtype
    min_broadcast = te.lang.cce.broadcast(min, shape, dtype)
    max_broadcast = te.lang.cce.broadcast(max, shape, dtype)

    nudged_min, nudged_max = _nudged_min_max_compute(min_broadcast,
                                                     max_broadcast, num_bits,
                                                     narrow_range)

    bool_both_zero_value = _bool_both_zero_compute(min_broadcast, max_broadcast)
    bool_both_zero_negate = _bool_negate(bool_both_zero_value)

    bool_less_equal_nudged_max = _less_equal_compare_float32(x, nudged_max)
    bool_more_equal_nudged_min = _less_equal_compare_float32(nudged_min, x)
    boolbetween_nudged_min_max = te.lang.cce.vmul(bool_less_equal_nudged_max,
                                                  bool_more_equal_nudged_min)
    backprops_wrt_input_tmp = te.lang.cce.vmul(boolbetween_nudged_min_max,
                                               gradients)
    backprops_wrt_bool_both_zero = te.lang.cce.vmul(backprops_wrt_input_tmp,
                                                    bool_both_zero_value)
    gradients_both_zero = te.lang.cce.vmul(gradients, bool_both_zero_negate)
    backprops_wrt_input = te.lang.cce.vadd(backprops_wrt_bool_both_zero,
                                           gradients_both_zero)

    bool_less_nudged_min = _bool_negate(bool_more_equal_nudged_min)
    backprop_wrt_min_tmp = te.lang.cce.vmul(bool_less_nudged_min, gradients)
    backprop_wrt_min_bool = te.lang.cce.vmul(backprop_wrt_min_tmp,
                                             bool_both_zero_value)
    backprop_wrt_min = te.lang.cce.sum(backprop_wrt_min_bool, sum_axis, True)

    bool_more_nudged_max = _bool_negate(bool_less_equal_nudged_max)
    backprop_wrt_max_tmp = te.lang.cce.vmul(bool_more_nudged_max, gradients)
    backprop_wrt_max_bool = te.lang.cce.vmul(backprop_wrt_max_tmp,
                                             bool_both_zero_value)
    backprop_wrt_max = te.lang.cce.sum(backprop_wrt_max_bool, sum_axis, True)
    res = [backprops_wrt_input, backprop_wrt_min, backprop_wrt_max]

    return res


# pylint: disable=locally-disabled,redefined-builtin,invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT, OPTION_ATTR_INT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def fake_quant_with_min_max_vars_per_channel_gradient(gradients, x,
                                                      min, max,
                                                      backprops_wrt_x,
                                                      backprops_wrt_min,
                                                      backprops_wrt_max,
                                                      num_bits=8,
                                                      narrow_range=False,
                                                      kernel_name="fake_quant_"
                                                                  "with_min_max"
                                                                  "_vars_per_"
                                                                  "channel_"
                                                                  "gradient"):
    """
    Generate fake_quant_with_min_max_vars_per_channel_gradient cce operator use
    fake_quant_with_min_max_vars_per_channel_gradient_compute

    Parameters
    ----------
    gradients: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data, assume gradients_shape equals input_data_shape,
        the data type, src_dtype equals dst_dtype, support fp32
    x: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data, assume input_data_shape equals output_data_shape,
        the data type, src_dtype equals dst_dtype, support fp32
    min: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of min,min shape equals to max shape and only 1 rank
        The shapes of min,max and shape_inputs last one dimension shoud be same
        the min data type,only support fp32
    max: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of max,min shape equals to max shape and only 1 rank
        The shapes of min,max and shape_inputs last one dimension shoud be same
        the max data type,only support fp32
    num_bits: int
        An optional int,defaults to 8,shoud be range [2,16]
    narrow_range: bool
        is narrow_range or not,defaults to False
    backprops_wrt_x: dict
        dict with keys(shape and dtype) of backprops output
    backprops_wrt_min : dict
        dict with keys(shape and dtype) of backprop min
    backprops_wrt_max : dict
        dict with keys(shape and dtype) of backprop max
    kernel_name: str
        cce kernel name, default value is
        "fake_quant_with_min_max_vars_per_channel_gradient"

    Returns
    ------
    None
    """
    # get dtype and shape attributes
    shape_gradients = gradients.get("shape")
    dtype_gradients = gradients.get("dtype")
    shape_inputs = x.get("shape")
    dtype_inputs = x.get("dtype")
    shape_min = min.get("shape")
    dtype_min = min.get("dtype")
    shape_max = max.get("shape")
    dtype_max = max.get("dtype")
    shape_backprops_wrt_x = backprops_wrt_x.get("shape")
    shape_backprops_wrt_min = backprops_wrt_min.get("shape")
    shape_backprops_wrt_max = backprops_wrt_max.get("shape")

    # check_kernel_name & shape
    dtype_inputs = dtype_inputs.lower()
    dtype_gradients = dtype_gradients.lower()
    dtype_min = dtype_min.lower()
    dtype_max = dtype_max.lower()
    check_shape(shape_backprops_wrt_x, param_name="backprops_wrt_x")
    check_shape(shape_backprops_wrt_min, param_name="backprops_wrt_min")
    check_shape(shape_backprops_wrt_max, param_name="backprops_wrt_max")
    check_shape(shape_inputs, param_name="x")
    check_shape(shape_gradients, param_name="gradients")
    check_shape(shape_min, min_rank=1, max_rank=1, param_name="min")
    check_shape(shape_max, min_rank=1, max_rank=1, param_name="max")
    # check input tensor data_type
    check_dtype(dtype_inputs, "float32", param_name="x")
    check_dtype(dtype_gradients, "float32", param_name="gradients")
    check_dtype(dtype_min, "float32", param_name="min")
    check_dtype(dtype_max, "float32", param_name="max")
    # check shape_min & shape_max,shape_gradients & shape_inputs
    if list(shape_gradients) != list(shape_inputs):
        raise RuntimeError("The shapes of gradients and inputs shoud be same")
    if list(shape_min) != list(shape_max):
        raise RuntimeError("The shapes of min and max shoud be same")
    if shape_min[0] != shape_inputs[-1]:
        raise RuntimeError(
            "The shapes of min,max and shape_inputs last one dimension"
            "shoud be same")
    # check num_bits range
    if num_bits > 16 or num_bits < 2:
        raise RuntimeError("numbits should be range[2,16]")

    # produce shape_min and shape_max for palceholder
    shape_min_broadcast, _, _ = broadcast_shapes(shape_min, shape_inputs, param_name_input1="min", param_name_input2="x")

    # definition of four input placeholders
    data_gradients = tvm.placeholder(shape_gradients, name="data_gradients",
                                     dtype=dtype_gradients)
    data_inputs = tvm.placeholder(shape_inputs, name="data_inputs",
                                  dtype=dtype_inputs)
    min_inputs = tvm.placeholder(shape_min_broadcast, name="min_inputs",
                                 dtype=dtype_min)
    max_inputs = tvm.placeholder(shape_min_broadcast, name="max_inputs",
                                 dtype=dtype_max)

    # get output by fake_quant_with_min_max_vars_per_channel_gradient_compute function
    res = fake_quant_with_min_max_vars_per_channel_gradient_compute(
        data_gradients, data_inputs,
        min_inputs, max_inputs,
        backprops_wrt_x,
        backprops_wrt_min,
        backprops_wrt_max,
        num_bits,
        narrow_range, kernel_name)

    input_placeholders = (data_gradients, data_inputs, min_inputs, max_inputs)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": list(input_placeholders) + list(res)}
    te.lang.cce.cce_build_code(sch, config)
