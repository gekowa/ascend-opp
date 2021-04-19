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
adds
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


@fusion_manager.register("adds")
def adds_compute(x, scalar, kernel_name="adds"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    scalar : a number of float or int
    kernel_name : str
        kernel name, default value is "adds"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    res = te.lang.cce.vadds(x, scalar)
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_FLOAT, KERNEL_NAME)
def adds(x, y, value, kernel_name="adds"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    value: a number of float
    kernel_name : str
        kernel name, default value is "adds"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    check_list = ("float16", "float32", "int32")
    check_dtype(dtype, check_list, param_name="x")

    scalar = tvm.const(value, dtype=dtype)
    data_input = tvm.placeholder(shape, name="data_input", dtype=dtype)
    res = adds_compute(data_input, scalar)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": [data_input, res]
    }
    te.lang.cce.cce_build_code(sch, config)
