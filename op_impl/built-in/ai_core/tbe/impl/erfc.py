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
erfc
"""
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce
from te.utils.op_utils import *

# define a scaler, value = 1
SCALER_ONE = 1
# define a scaler, value = -1
SCALER_NEGATIVE_ONE = -1
# define a scaler, value = -0.47047, only used in compute of erfc and erf
SCALER_P = 0.47047
# define a scaler, value = 0.3480242, only used in compute of erfc and erf
SCALER_A = 0.3480242
# define a scaler, value = -0.0958798, only used in compute of erfc and erf
SCALER_B = -0.0958798
# define a scaler, value = 0.7478556, only used in compute of erfc and erf
SCALER_C = 0.7478556
# define a scaler, value = 32768
SCALER_FP16_MAX = 32768
# define a scaler, value = 2**(-15)
SCALER_FP16_MIN = 2**(-15)


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("erfc")
def erfc_compute(input_x, output_y, kernel_name="erfc"):
    """
    compute erfc

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        he dict of output_data, include keys(shape and dtype)
    kernel_name: str
        kernel name, default value is "erfc"

    Returns
    -------
    erfc_result: TVM tensor
        the =result of compute
    """

    dtype = input_x.dtype
    shape = te.lang.cce.util.shape_to_list(input_x.shape)

    const_one = tvm.const(SCALER_ONE, dtype="float32")
    const_negative_one = tvm.const(SCALER_NEGATIVE_ONE, dtype="float32")
    const_p = tvm.const(SCALER_P, dtype="float32")
    const_a = tvm.const(SCALER_A, dtype="float32")
    const_b = tvm.const(SCALER_B, dtype="float32")
    const_c = tvm.const(SCALER_C, dtype="float32")
    fp16_max = tvm.const(SCALER_FP16_MAX, dtype=dtype)
    fp16_min = tvm.const(SCALER_FP16_MIN, dtype=dtype)

    if dtype == "float16":
        input_x = te.lang.cce.cast_to(input_x, "float32")

    data_sign_vmuls = te.lang.cce.vmuls(input_x, fp16_max)
    data_sign_abs = te.lang.cce.vabs(data_sign_vmuls)
    data_vadds = te.lang.cce.vadds(data_sign_abs, fp16_min)
    data_sign_div = te.lang.cce.vdiv(data_sign_vmuls, data_vadds)
    data_round = te.lang.cce.round(data_sign_div)
    tensor_sign = te.lang.cce.cast_to(data_round, dtype)

    tensor_one = te.lang.cce.broadcast(const_one, shape, "float32")
    tensor_abs = te.lang.cce.vabs(input_x)
    erfc_t_vmuls = te.lang.cce.vmuls(tensor_abs, const_p)
    erfc_t_vadds = te.lang.cce.vadds(erfc_t_vmuls, const_one)
    erfc_data_t = te.lang.cce.vdiv(tensor_one, erfc_t_vadds)

    erfc_abs_square = te.lang.cce.vmul(tensor_abs, tensor_abs)
    erfc_data_vmuls = te.lang.cce.vmuls(erfc_abs_square, const_negative_one)
    erfc_data_exp = te.lang.cce.vexp(erfc_data_vmuls)

    erfc_data_t_square = te.lang.cce.vmul(erfc_data_t, erfc_data_t)
    erfc_data_t_cube = te.lang.cce.vmul(erfc_data_t, erfc_data_t_square)

    erfc_t_vmuls = te.lang.cce.vmuls(erfc_data_t, const_a)
    erfc_t_square_vmuls = te.lang.cce.vmuls(erfc_data_t_square, const_b)
    erfc_t_cube_vmuls = te.lang.cce.vmuls(erfc_data_t_cube, const_c)

    erfc_square_vadd = te.lang.cce.vadd(erfc_t_vmuls, erfc_t_square_vmuls)
    erfc_cube_vadd_ = te.lang.cce.vadd(erfc_square_vadd, erfc_t_cube_vmuls)
    erfc_cube_vmuls = te.lang.cce.vmuls(erfc_cube_vadd_, const_negative_one)
    erfc_exp_vmul = te.lang.cce.vmul(erfc_cube_vmuls, erfc_data_exp)
    erfc_exp_vadds = te.lang.cce.vadds(erfc_exp_vmul, const_one)
    erfc_sign_vmul = te.lang.cce.vmul(tensor_sign, erfc_exp_vadds)
    erfc_sign_vmuls = te.lang.cce.vmuls(erfc_sign_vmul, const_negative_one)
    erfc_result = te.lang.cce.vadds(erfc_sign_vmuls, const_one)

    if dtype == "float16":
        erfc_result = te.lang.cce.cast_to(erfc_result, dtype)
    return erfc_result


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def erfc(input_x, output_y, kernel_name="erfc"):
    """
    algorithm: erfc
    Computes the Gauss error function of `x` element-wise

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "erfc"

    Returns
    -------
    None
    """
    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype")

    check_shape(shape_input, param_name="input_x")

    dtype_input = dtype_input.lower()
    check_list = ("float16", "float32")
    check_dtype(dtype_input, check_list, param_name="input_x")

    shape_input = util.shape_refine(shape_input)
    reshape_input = (functools_reduce(lambda x, y: x * y, shape_input[:]),)
    data_input = tvm.placeholder(reshape_input, name="data_input",
                                 dtype=dtype_input)

    erfc_result = erfc_compute(data_input, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(erfc_result)

    config = {"name": kernel_name,
              "tensor_list": [data_input, erfc_result]}

    te.lang.cce.cce_build_code(sch, config)
