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
gelu_grad
"""
from __future__ import absolute_import

import operator

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as reduceIns
from te import platform as tbe_platform
from te.utils.op_utils import *

# CSVALUE equals 0.044715
CSVALUE = tvm.const(0.044715, "float32")
# SQURT equals np.sqrt(2 / np.pi)
SQURT = tvm.const(0.7978846, "float32")
# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648

#CSVALUE_4 equals 0.5*np.sqrt(2 / np.pi)*3*CSVALUE
CSVALUE_4 = tvm.const(0.0535161122, "float32")
#CSVALUE_5 equals 0.5*np.sqrt(2 / np.pi)
CSVALUE_5 = tvm.const(0.3989422804, "float32")

# min float32 value
MIN_FP32 = 2**(-126)


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
def tanh_compute(input_x, output_y, kernel_name="tanh"):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is tanh

    Returns
    -------
    res : tvm.tensor
        the result of tanh
    """
    input_dtype = input_x.dtype

    has_improve_precision = False
    if input_dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                    "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        has_improve_precision = True

    input_abs = te.lang.cce.vabs(input_x)
    power_val = te.lang.cce.vmuls(input_abs, tvm.const(-2, "float32"))
    exp_val = te.lang.cce.vexp(power_val)

    up_val_tmp = te.lang.cce.vmul(exp_val, input_x)
    up_val = te.lang.cce.vsub(input_x, up_val_tmp)

    input_x_tmp = te.lang.cce.vadds(input_abs, MIN_FP32)
    down_val_tmp = te.lang.cce.vadds(exp_val, tvm.const(1, "float32"))
    down_val = te.lang.cce.vmul(down_val_tmp, input_x_tmp)

    res = te.lang.cce.vdiv(up_val, down_val)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


def _math_four_compute(placeholders):
    """
    placeholders: data_x
    return: math_four
    math_four equals (np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))
    """
    data_x = placeholders
    datax_pow = te.lang.cce.vmul(data_x, data_x)
    datax_pow1 = te.lang.cce.vmul(datax_pow, data_x)
    datax_muls_c = te.lang.cce.vmuls(datax_pow1, CSVALUE)
    datax_addx = te.lang.cce.vadd(datax_muls_c, data_x)
    datax_muls_s = te.lang.cce.vmuls(datax_addx, SQURT)

    return datax_muls_s



def _result2_compute(placeholders):
    """
    placeholders: data_x
    return: result
    result equals np.sqrt(2 / np.pi) (1 + 3*0.044715*x2)
    """
    data_x = placeholders
    val1 = CSVALUE_5
    data_x_sqr = te.lang.cce.vmul(data_x, data_x)
    data_x_sqr_vmul = te.lang.cce.vmuls(data_x_sqr, CSVALUE_4)
    data_x_sqr_vmul_add1 = te.lang.cce.vadds(data_x_sqr_vmul, val1)

    return data_x_sqr_vmul_add1


def _result3_compute(placeholders):
    """
    placeholders: data_x
    return: result3
    result3 equals x*0.5*(1 - tanh(math_four)*tanh(math_four))
    """
    data_x = placeholders
    val1 = tvm.const(1.0, "float32")
    math_four = _math_four_compute(data_x)
    tanh_math_four = tanh_compute(math_four, placeholders[1])
    tanh_math_four_squ = te.lang.cce.vmul(tanh_math_four, tanh_math_four)
    val3 = tvm.const(-1.0, "float32")
    math_four_squ_n = te.lang.cce.vmuls(tanh_math_four_squ, val3)
    add_compute = te.lang.cce.vadds(math_four_squ_n, val1)
    result3 = te.lang.cce.vmul(add_compute, data_x)

    return result3, tanh_math_four


def _result_grad_compute(placeholders):
    """
    placeholders: data_x, data_gelu
    return: res_grad
    res_grad = res/x +
       x*0.5*(1 - tanh(math_four)*tanh(math_four))*
       np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)
    """
    data_x = placeholders[0]

    result2 = _result2_compute(data_x)
    result3, tanh_math_four_result = _result3_compute(data_x)
    mul_result2_3 = te.lang.cce.vmul(result2, result3)

    # compute res1 = res/x = f1 = x*(0.5*(1+tanh_math_four_result))
    mul_compute_1 =  te.lang.cce.vadds(tanh_math_four_result, 1)
    mul_compute_2 = te.lang.cce.vmuls(mul_compute_1, 0.5)

    res_grad = te.lang.cce.vadd(mul_compute_2, mul_result2_3)

    return res_grad

# pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
@fusion_manager.register("gelu_grad")
def gelu_grad_compute(input_dy, input_x, input_y,
                      output_z, kernel_name="gelu_grad"):
    """
    algorithm: gelu_grad
    calculating: dy*res'
    res' = res/x +
           x*0.5*(1 - tanh(math_four)*tanh(math_four))*
           np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)
    math_four = (np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))

    Parameters
    ----------
    placeholders: TVM tensor.
        input placeholder tensors data
    shape_dy: list or tuple.
        shape of dy
    shape_x: list or tuple.
        shape of x
    shape_y: list or tuple.
        shape of gelu
    dtype: str
        the data type, assume src_dtype equals dst_dtype,
         only support float16, float32,
    kernel_name: str
        cce kernel name, default value is "cce_gelu_grad"
    need_build: str
        if need to build CCEC kernel, default value is False
    need_print: str
        if need to print the ir, default value is False

    Returns:
    -------
    A TVM tensor same as input placeholders.
    """
    input_dtype = input_dy.dtype.lower()

    has_improve_precision = False
    if input_dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                    "float32"):
        input_dy = te.lang.cce.cast_to(input_dy, "float32")
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")
        has_improve_precision = True
    # compute res'
    result5 = _result_grad_compute([input_x, output_z])
    # compute dy*res'
    result_temp1 = te.lang.cce.vmul(input_dy, result5)

    # input_y must be involved in order to keep it
    input_y_temp_1 = te.lang.cce.vmuls(input_y, 0)

    result = te.lang.cce.vadd(result_temp1, input_y_temp_1)
    if has_improve_precision:
        result = te.lang.cce.cast_to(result, "float16")

    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 KERNEL_NAME)
def gelu_grad(input_dy, input_x, input_y, output_z, kernel_name="gelu_grad"):
    """
    algorithm: gelu_grad
    calculating: dy*res'
    res' = res/x +
           x*0.5*(1 - tanh(math_four)*tanh(math_four))*
           np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)
    math_four = (np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))

    Parameters
    ----------
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    input_y : dict
        shape and dtype of y input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is gelu_grad

    Returns:
    -------
    none.
    """
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")

    check_shape(shape_dy, param_name="input_dy")
    check_shape(shape_x, param_name="input_x")
    check_shape(shape_y, param_name="input_y")
    input_dtype = input_dy.get("dtype").lower()
    check_list = ("float16", "float32")
    check_dtype(input_dtype, check_list, param_name="input_dy")
    shape_dy = list(shape_dy)
    shape_x = list(shape_x)
    shape_y = list(shape_y)
    if not (operator.eq(shape_dy, shape_x) and operator.eq(shape_dy, shape_y)):
        raise RuntimeError("all input shape must be equal")

    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape_dy)
    data_dy = tvm.placeholder(fuseshape, name="data_dy", dtype=input_dtype)
    data_x = tvm.placeholder(fuseshape, name="data_x", dtype=input_dtype)
    data_gelu = tvm.placeholder(fuseshape, name="data_gelu", dtype=input_dtype)
    res = gelu_grad_compute(data_dy, data_x, data_gelu, output_z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_dy, data_x, data_gelu, res]}

    te.lang.cce.cce_build_code(sch, config)
