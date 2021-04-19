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
cosh
"""
import te.lang.cce
from te import tvm
from functools import reduce as functools_reduce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import *

# define a scaler , value = -1
SCALER_NEGATIVE_ONE = -1
# define a scaler , value = 0.5
SCALER_ZERO_POINT_FIVE = 0.5
# define a scaler , value = 2
SCALAR_TWO = 2

# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("cosh")
def cosh_compute(input_x, output_cosh, kernel_name="cosh"):
    """
    algorithm: cosh
    calculating data's cosh, y = (e^(x)+e^(-x))/2

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_cosh: TVM tensor
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "cosh"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype
    shape = input_x.shape
    has_improve_precision = False
    if dtype != "float32" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                    "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    data_mul = te.lang.cce.vmuls(input_x,
                                 tvm.const(SCALER_NEGATIVE_ONE, dtype))
    data_exp = te.lang.cce.vexp(data_mul)
    data_exp_x = te.lang.cce.vmuls(data_exp,
                                   tvm.const(SCALER_ZERO_POINT_FIVE,
                                             dtype))

    tensor_two = te.lang.cce.broadcast(tvm.const(SCALAR_TWO, dtype), shape)
    data_ln2 = te.lang.cce.vlog(tensor_two)
    data_neg_ln2 = te.lang.cce.vmuls(data_ln2,
                                     tvm.const(SCALER_NEGATIVE_ONE, dtype))
    data_x = te.lang.cce.vadd(input_x, data_neg_ln2)
    data_exp_data = te.lang.cce.vexp(data_x)

    res = te.lang.cce.vadd(data_exp_x, data_exp_data)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def cosh(input_x, output_cosh, kernel_name="cosh"):
    """
    algorithm: cosh
    calculating data's cosh, y = (e^(2x)+e^(-x))/2

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32
    output_cosh: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "cosh"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    check_shape(shape, param_name="input_x")
    check_list = ("float16", "float32")
    input_dtype = dtype.lower()
    check_dtype(input_dtype, check_list, param_name="input_x")
    reshape_input = (functools_reduce(lambda x, y: x * y, shape[:]),)
    data_input = tvm.placeholder(reshape_input,
                                 name="data_input", dtype=input_dtype)
    res = cosh_compute(data_input, output_cosh, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
