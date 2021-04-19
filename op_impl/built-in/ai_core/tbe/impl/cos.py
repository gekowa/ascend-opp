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
cos
"""
from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

# 2pi, the cycle of cosin
TWO_PI = 2*3.14159265358979

# pylint: disable=locally-disabled, unused-argument
@fusion_manager.register("cos")
def cos_compute(input_x, output_y, kernel_name="cos"):
    """
    algorithm: cos
    calculating data's cos x = 1 - x^2/2! + x^4/4! + ... + (-1)^k*x^2k/(2k)!

    Parameters
    ----------
    input_x : TVM tensor
              data of input
    output_y: dict
              shape and dtype of output, should be same shape and type as input
    kernel_name: str
              kernel name, default value is "cos"

    Returns
    -------
    res : TVM tensor
          the result of cos
    """

    dtype = input_x.dtype
    shape = te.lang.cce.util.shape_to_list(input_x.shape)

    # cast to type float32 when type is float16
    has_improve_precision = False
    if dtype.lower() == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul",
                                                    "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    # round the input
    round_fp16 = te.lang.cce.round(te.lang.cce.vmuls(input_x, 1.0/TWO_PI))
    round_fp32 = te.lang.cce.cast_to(round_fp16, dtype)
    input_x_round = te.lang.cce.vsub(input_x,
                                     te.lang.cce.vmuls(round_fp32, TWO_PI))

    # the initial value one
    const_res = tvm.const(1.0, dtype=dtype)
    res = te.lang.cce.broadcast(const_res, shape)
    # compute the rank 2
    input_x_power = te.lang.cce.vmul(input_x_round, input_x_round)
    iter_value = te.lang.cce.vmuls(input_x_power, -1.0/2.0)
    res = te.lang.cce.vadd(res, iter_value)
    # compute the rank 4~14
    iter_list = (4, 6, 8, 10, 12, 14)
    for i in iter_list:
        iter_value = te.lang.cce.vmuls(te.lang.cce.vmul(input_x_power,
                                                        iter_value),
                                       -1.0/(i*(i-1)))
        res = te.lang.cce.vadd(res, iter_value)

    # cast the dtype to float16
    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def cos(input_x, output_y, kernel_name="cos"):
    """
    algorithm: cos
    calculating data's cos x = 1 - x^2/2! + x^4/4! + ... + (-1)^k*x^2k/(2k)!

    Parameters
    ----------
    input_x : dict
              shape and dtype of input, only support float16, float32
    output_y: dict
              shape and dtype of output, should be same shape and type as input
    kernel_name : str
              kernel name, default value is "cos"

    Returns
    -------
    None
    """
    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype").lower()

    check_shape(shape_input, param_name="input_x")
    check_list = ("float16", "float32")
    check_dtype(dtype_input, check_list, param_name="input_x")

    reshape_input = (functools_reduce(lambda x, y: x * y, shape_input[:]),)
    data_input = tvm.placeholder(reshape_input,
                                 name="data_input", dtype=dtype_input)
    res = cos_compute(data_input, output_y, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
