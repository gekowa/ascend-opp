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
div
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import refine_shapes_for_broadcast
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable=locally-disabled,too-many-locals,unused-argument
@fusion_manager.register("div")
def div_compute(input_x, input_y, output_div, kernel_name="div"):
    """
    div compute
    calculating data's div, res =x / y

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "div"

    Returns
    -------
    res: TVM tensor
        the result of div compute
    """
    input_data1 = te.lang.cce.util.shape_to_list(input_x.shape)
    input_data2 = te.lang.cce.util.shape_to_list(input_y.shape)
    shape_list = broadcast_shapes(input_data1, input_data2,
                                  param_name_input1="input_x",
                                  param_name_input2="input_y")
    dtype = input_x.dtype
    int_list = ("int8", "uint8", "int32")
    int_flag = dtype in int_list
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")
    data_x_broad = te.lang.cce.broadcast(input_x, shape_list[2])
    data_y_broad = te.lang.cce.broadcast(input_y, shape_list[2])
    res = te.lang.cce.vdiv(data_x_broad, data_y_broad)

    if int_flag:
        res = te.lang.cce.floor(res)

    res = te.lang.cce.cast_to(res, dtype)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def div(input_x, input_y, output_div, kernel_name="div"):
    """
    algorithm: div
    calculating data's div, res =x / y

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    input_y: dict
        dict with keys(shape and dtype) of input_y
    output_div: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "div"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype = input_x.get("dtype")
    shape_y = input_y.get("shape")

    check_shape(shape_x, param_name="input_x")
    check_shape(shape_y, param_name="input_y")
    shape_list = broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                  param_name_input2="input_y")
    input_dtype = dtype.lower()
    check_list = ("float16", "float32", "int8", "uint8", "int32")
    check_dtype(input_dtype, check_list, param_name="input_x")

    reshape_x, reshape_y = refine_shapes_for_broadcast(shape_list[0],
                                                       shape_list[1])
    data_x = tvm.placeholder(reshape_x, dtype=input_dtype, name="data_x")
    data_y = tvm.placeholder(reshape_y, dtype=input_dtype, name="data_y")

    res = div_compute(data_x, data_y, output_div, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    te.lang.cce.cce_build_code(sch, config)
