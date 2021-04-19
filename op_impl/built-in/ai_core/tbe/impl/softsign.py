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
softsign
"""
from __future__ import absolute_import

from te import tvm
from te import platform as tbe_platform
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce
from te.utils.op_utils import *

# define a scalar, value = 1
SCALAR_ONE = 1


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("softsign")
def softsign_compute(input_x, y, kernel_name="softsign"):
    """
    Computes for softsign.
    The compute: "x / (abs(x) + 1)".

    Parameters
    ----------
    input_x: TVM tensor
        data of input.
        source data type, support "float16", "float32".
    y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "softsign".

    Returns
    -------
    res: TVM tensor
        output tensor. has the same type as `input_x`.
    """
    dtype = input_x.dtype
    if dtype == "float16" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul",
                                                "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")

    data_abs = te.lang.cce.vabs(input_x)
    data_add = te.lang.cce.vadds(data_abs, SCALAR_ONE)
    data_rec = te.lang.cce.vrec(data_add)
    res = te.lang.cce.vmul(input_x, data_rec)

    if dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def softsign(x, y, kernel_name="softsign"):
    """
    Computes for softsign.

    Parameters
    ----------
    x: dict
        data of input.
        source data type, support "float16", "float32".
    y: dict
        data of output.
    kernel_name : str
        kernel name, default value is "softsign".

    Returns
    -------
    None
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32")
    check_dtype(dtype_input.lower(), check_list, param_name="x")

    shape = util.shape_refine(shape_input)
    shape_x = (functools_reduce(lambda x, y: x*y, shape[:]),)
    input_dtype = dtype_input.lower()
    data = tvm.placeholder(shape_x, name="data", dtype=input_dtype)

    res = softsign_compute(data, y, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)
