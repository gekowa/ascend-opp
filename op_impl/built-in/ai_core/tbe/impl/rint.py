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
rint
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from functools import reduce as reduceIns
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("rint")
def rint_compute(input_x, output_y, kernel_name="rint"):
    """
    rint compute
    calculating rint(x):
    returns the integer nearest to x by element-wise
    If the result is between two representable values,
     the even number should be used.
    For example:
    x :    [0.9, 2.5, 2.3, 1.5, -4.5]
    res : [ 1.0, 2.0, 2.0, 2.0, -4.0 ]

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    output_y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "rint"

    Returns
    -------
    res: TVM tensor
        the result of rint compute
    """
    res = te.lang.cce.round(input_x)
    res = te.lang.cce.cast_to(res, input_x.dtype)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def rint(input_x, output_y, kernel_name="rint"):
    """
    algorithm: rint
    calculating rint(x):
    returns the integer nearest to x by element-wise
    If the result is between two representable values,
     the even number should be used.
    For example:
    x :    [0.9, 2.5, 2.3, 1.5, -4.5]
    res : [ 1.0, 2.0, 2.0, 2.0, -4.0 ]

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    output_y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "rint"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype = input_x.get("dtype")

    check_shape(shape_x, param_name="input_x")

    check_list = ("float16", "float32")
    check_dtype(dtype.lower(), check_list, param_name="input_x")
    fuseshape = [1]
    fuseshape[0] = reduceIns(lambda x, y: x*y, shape_x)
    data_x = tvm.placeholder(fuseshape, dtype=dtype.lower(), name="data")
    res = rint_compute(data_x, output_y, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    te.lang.cce.cce_build_code(sch, config)
