# Copyright 2020 Huawei Technologies Co., Ltd
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
abs
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from functools import reduce as reduceIns
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

SHAPE_SIZE_LIMIT = 2147483648  # shape limit


# pylint: disable=invalid-name,unused-argument
@fusion_manager.register("abs")
def abs_compute(x, y, kernel_name="abs"):
    """
    algorithm: abs

    Parameters
    ----------
    x: TVM tensor
        the placeholder of x
    y: dict
        dict info of y
    kernel_name: str
        kernel name, default value is "abs"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    inp_dtype = x.dtype

    res = te.lang.cce.vabs(x)
    if inp_dtype == "int32":
        res = te.lang.cce.round(res)
    return res


# pylint: disable=redefined-builtin
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def abs(x, y, kernel_name="abs"):
    """
    algorithm: abs

    calculating data's abs,y= |x|

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support float16, float32, int32
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is abs

    Returns
    -------
    None
    """
    shape = x.get("shape")
    check_shape(shape, param_name="x")

    check_list = ["float16", "float32", "int32"]
    inp_dtype = x.get("dtype").lower()
    check_dtype(inp_dtype, check_list, param_name="x")

    shape = util.shape_refine(shape)
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=inp_dtype)

    res = abs_compute(data, y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)
