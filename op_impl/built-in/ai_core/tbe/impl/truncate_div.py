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
truncate_div
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
@fusion_manager.register("truncate_div")
def truncate_div_compute(input_x, input_y, output_x,
                         kernel_name="truncate_div"):
    """
    compute truncate_div
    calculating data's truncate_div, res = floor(x / y) if x/y>0 else ceil(x/y)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data x
    input_y: TVM tensor
        the placeholder of input data y
    output_x: dict
        not used yet
    kernel_name: str
        kernel name

    Returns
    -------
    res: TVM tensor
        the result of truncate_div_compute
    """
    shape_list = broadcast_shapes(
        te.lang.cce.util.shape_to_list(input_x.shape),
        te.lang.cce.util.shape_to_list(input_y.shape),
        param_name_input1="input_x", param_name_input2="input_y")
    int_list = ("int32", "int8", "uint8")
    input_dtype = input_x.dtype

    if input_dtype in int_list:
        data_zero = te.lang.cce.broadcast(tvm.const(0, 'float32'),
                                          shape_list[2], 'float32')
        data_x_broad = te.lang.cce.cast_to(input_x, 'float32')
        data_y_broad = te.lang.cce.cast_to(input_y, 'float32')
        data_x_broad = te.lang.cce.broadcast(data_x_broad, shape_list[2])
        data_y_broad = te.lang.cce.broadcast(data_y_broad, shape_list[2])
        res_div = te.lang.cce.vdiv(data_x_broad, data_y_broad)
        res_min_int = te.lang.cce.ceil(te.lang.cce.vmin(res_div, data_zero))
        res_max_int = te.lang.cce.floor(te.lang.cce.vmax(res_div, data_zero))
        res_trunc = te.lang.cce.vadd(res_min_int, res_max_int)
    else:
        if tbe_platform.cce_conf.api_check_support("te.lang.cce.vlog",
                                                   "float32"):
            input_x = te.lang.cce.cast_to(input_x, 'float32')
            input_y = te.lang.cce.cast_to(input_y, 'float32')
        data_x_broad = te.lang.cce.broadcast(input_x, shape_list[2])
        data_y_broad = te.lang.cce.broadcast(input_y, shape_list[2])
        res_trunc = te.lang.cce.vdiv(data_x_broad, data_y_broad)

    res = te.lang.cce.cast_to(res_trunc, input_dtype)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def truncate_div(input_x, input_y, output_x, kernel_name="truncate_div"):
    """
    algorithm: truncate_div
    calculating data's truncate_div, res = floor(x / y) if x/y>0 else ceil(x/y)

    Parameters
    ----------
    input_x: dict with keys(shape and dtype)
        only support {float16, float32, int8, uint8(on mini)},
        {float16, float32(on cloud)}
    input_y: dict with keys(shape and dtype)
        dict info of input_y
    output_x: dict with keys(shape and dtype)
        dict info of output_x
    kernel_name: str
        kernel name, default value is "truncate_div"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    dtype = input_x.get("dtype")

    check_shape(shape_x, param_name="input_x")
    check_shape(shape_y, param_name="input_y")

    input_dtype = dtype.lower()
    check_list = ("float16", "float32", "int32", "int8", "uint8")
    check_dtype(input_dtype, check_list, param_name="input_x")

    shape_list = broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                  param_name_input2="input_y")
    reshape_x, reshape_y = refine_shapes_for_broadcast(shape_list[0],
                                                       shape_list[1])
    data1 = tvm.placeholder(reshape_x, dtype=input_dtype, name="data1")
    data2 = tvm.placeholder(reshape_y, dtype=input_dtype, name="data2")
    res = truncate_div_compute(data1, data2, output_x, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data1, data2, res]}
    te.lang.cce.cce_build_code(sch, config)
