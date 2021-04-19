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
invert
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("invert")
def invert_compute(input_x, output_y, kernel_name="invert"):
    """Flips all bits elementwise.

    Parameters
    ----------
    input_x: TVM tensor
        input tensor.
    output_y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "invert".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    res = te.lang.cce.vnot(input_x)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def invert(input_x, output_y, kernel_name="invert"):
    """Flips all bits elementwise.

    Parameters
    ----------
    input_x: dict
        the dict of input tensor.
        Must be one of the following types: `int16`, `uint16`.
    output_y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "invert".

    Returns
    -------
    None.
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    dtype_x_lower = dtype_x.lower()
    check_list = ("int16", "uint16")

    check_shape(shape_x, param_name="input_x")
    check_dtype(dtype_x_lower, check_list, param_name="input_x")

    shape_x = (functools_reduce(lambda x, y: x * y, shape_x[:]),)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x_lower)
    res = invert_compute(data_x, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_x, res]}
    te.lang.cce.cce_build_code(sch, config)
