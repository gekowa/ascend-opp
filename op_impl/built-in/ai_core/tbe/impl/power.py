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
power
"""
# pylint: disable=redefined-outer-name
import math
from functools import reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from te.utils.op_utils import *
from topi import generic
from topi.cce import util


def positive_compute(base, power, version, input_dtype):
    """
    calculate power for positive elements of base tensor

    Parameters
    ----------
    base: the base tensor
    power: attr power
    version: the product version
    input_dtype: dtype of input

    Returns
    ----------
    res: the result tensor
    """

    base_cast = base

    if input_dtype == "float16" and \
       tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32") and \
       tbe_platform.cce_conf.api_check_support("te.lang.cce.vlog", "float32"):
        base_cast = te.lang.cce.cast_to(base, "float32")

    log_val = te.lang.cce.vlog(base_cast)
    mul_val = te.lang.cce.vmuls(log_val, power)
    exp_val = te.lang.cce.vexp(mul_val)

    if exp_val.dtype.lower() != input_dtype:
        exp_val = te.lang.cce.cast_to(exp_val, input_dtype)

    return exp_val


def negtive_compute(base, power, nan_values, version, input_dtype):
    """
    calculate power for negative elements of base tensor

    Parameters
    ----------
    base: the base tensor
    power: attr power
    nan_values: a tensor with nan values
    version: the product version
    input_dtype: dtype of input

    Returns
    ----------
    res: the result tensor
    """

    if float(power).is_integer():
        base_cast = base

        if input_dtype == "float16" and \
           tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32") and \
           tbe_platform.cce_conf.api_check_support("te.lang.cce.vlog", "float32"):
            base_cast = te.lang.cce.cast_to(base, "float32")

        sign_value = math.pow(-1, power)
        abs_base_value = te.lang.cce.vabs(base_cast)
        log_value = te.lang.cce.vlog(abs_base_value)
        mul_value = te.lang.cce.vmuls(log_value, power)
        exp_value = te.lang.cce.vexp(mul_value)
        res = te.lang.cce.vmuls(exp_value, sign_value)

        if res.dtype.lower() != input_dtype:
            res = te.lang.cce.cast_to(res, input_dtype)

        return res

    return nan_values


def zero_compute(power, nan_values, zero_values):
    """
    calculate power for zero elements of base tensor

    Parameters
    ----------
    power: attr power
    nan_values: a tensor with nan values
    zero_values: a tensor with zero values

    Returns
    ----------
    res: the result tensor
    """

    if power > 0.0:
        return zero_values

    return nan_values


def power_scalar(input_x, base, power):
    """
    calculate power when attr scale is 0.0 and attr power is not

    Parameters
    ----------
    input_x: placeholder of input
    base: the base value, equals attr shift
    power: attr power

    Returns
    ----------
    res: the result when attr scale is 0.0 and attr power is not
    """

    tmp_zero = te.lang.cce.vmuls(input_x, 0)
    ones = te.lang.cce.vadds(tmp_zero, 1)
    zeros = tmp_zero


    if base > 0.0:
        res = te.lang.cce.vmuls(ones, math.pow(base, power))
        return res
    if base < 0.0:
        if float(power).is_integer():
            res = te.lang.cce.vmuls(ones, math.pow(base, power))
            return res

        # return abnormal value
        res = te.lang.cce.vrec(zeros)
        return res

    if power > 0:
        return zeros

    # return abnormal value
    res = te.lang.cce.vrec(zeros)

    return res


def zero_diff_scale_compute(input_x, shift, power):
    """
    calculate power when power*scale is 0.0

    Parameters
    ----------
    input_x: placeholder of input
    shift: attr shift
    power: attr power

    Returns
    ----------
    res: the result when power*scale is 0.0
    """

    if power == 0.0:
        tmp_zero = te.lang.cce.vmuls(input_x, 0)
        res = te.lang.cce.vadds(tmp_zero, 1)
        return res

    res = power_scalar(input_x, shift, power)

    return res

# pylint: disable=locally-disabled,unused-argument,too-many-arguments
@fusion_manager.register("power")
def power_compute(input_x, output_y, power=1.0, scale=1.0,
          shift=0.0, kernel_name="power"):
    """
    calculate power according to different cases

    Parameters
    ----------
    input_x: placeholder of input
    power: attr power
    scale: attr scale
    shift: attr shift

    Returns
    ----------
    res: result of power
    """

    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    input_dtype = input_x.dtype.lower()

    diff_scale = power * scale

    if diff_scale == 0.0:
        res = zero_diff_scale_compute(input_x, shift, power)
        return res

    shift_scaled_x = te.lang.cce.vmuls(input_x, scale)
    shift_scaled_x = te.lang.cce.vadds(shift_scaled_x, shift)

    tmp_zero = te.lang.cce.vmuls(input_x, 0)
    zeros = tmp_zero

    nan_value = te.lang.cce.vrec(zeros)

    if power == 1.0:
        res = shift_scaled_x
        return res
    if power == 2.0:
        res = te.lang.cce.vmul(shift_scaled_x, shift_scaled_x)
        return res
    if power == 3.0:
        res = te.lang.cce.vmul(shift_scaled_x, shift_scaled_x)
        res = te.lang.cce.vmul(res, shift_scaled_x)
        return res

    positive_pow_val = \
        positive_compute(shift_scaled_x, power, cce_product, input_dtype)
    negative_pow_val = \
        negtive_compute(shift_scaled_x, power,
                        nan_value, cce_product, input_dtype)
    zero_pow_val = zero_compute(power, nan_value, zeros)

    res = te.lang.cce.vcmpsel(shift_scaled_x, zeros,
                              'gt', positive_pow_val, negative_pow_val)
    res = te.lang.cce.vcmpsel(shift_scaled_x, zeros,
                              'eq', zero_pow_val, res)

    return res


# pylint: disable=redefined-outer-name, too-many-arguments, unused-variable
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_FLOAT,
                 OPTION_ATTR_FLOAT, OPTION_ATTR_FLOAT, KERNEL_NAME)
def power(input_x, output_y, power=1.0, scale=1.0,
          shift=0.0, kernel_name="power"):
    """
    calculate power of input tensor according to
    y = (x * scale + shift) ** power

    Parameters
    ----------
    input_x: dict of input, include shape and
    dtype, dtype support float16, float32
    output_y: dict of output, include shape and
    dtype, dtype support float16, float32
    power: attr power, default value is 1.0
    scale: attr scale, default value is 1.0
    shift: attr shift, default value is 0.0
    kernel_name: cce kernel name, default value is "power"

    Returns
    ----------
    None
    """

    shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    check_shape(shape, param_name="x")
    type_tuple = ("float16", "float32")
    check_dtype(input_dtype, type_tuple, param_name="x")


    fuseshape = [1]
    fuseshape[0] = reduce(lambda x, y: x*y, shape)

    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)

    cur_cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if cur_cce_product in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS"):
        if input_dtype == "float32":
            error_info = {}
            error_info['errCode'] = 'E80008'
            error_info['param_name'] = 'input_x'
            error_info['op_name'] = 'power'
            error_info['expect_value'] = "float16"
            error_info['real_value'] = input_dtype
            raise RuntimeError(error_info, "In op[%s], the parameter[%s]'s dtype " 
                                           "should be [%s], but actually is [%s]."
                               % (error_info['op_name'], error_info['param_name'],
                                  error_info['expect_value'], error_info['real_value']))

        res = power_compute(data_input, output_y, power, scale, shift, kernel_name)
    else:
        res = power_compute(data_input, output_y, power, scale, shift, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res],
              "print_ir": True}

    te.lang.cce.cce_build_code(sch, config)
