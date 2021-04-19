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
mod
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from te.utils.op_utils import refine_shapes_for_broadcast
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("mod")
def mod_compute(input_x, input_y, output_z, kernel_name="mod"):
    """
    Returns element-wise remainder of division.
    the result here is consistent with a truncating divide.
    'truncate_mod(x, y) = x - truncate_div(x, y) * y'.

    Parameters
    ----------
    input_x: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32", "int8", "uint8", "int32".
    input_y: TVM tensor
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_x'.
    output_z: dict
        data of output.
        Must have the same type as 'input_x'.
    kernel_name: str
        kernel name, default value is "mod"

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "input_x".
    """
    shape_x = te.lang.cce.util.shape_to_list(input_x.shape)
    shape_y = te.lang.cce.util.shape_to_list(input_y.shape)
    dtype = input_x.dtype.lower()

    has_improve_precision = False
    if dtype != "float32" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_y = te.lang.cce.cast_to(input_y, "float32")
        has_improve_precision = True

    if list(shape_x) != list(shape_y):
        shape_x, shape_y, shape_broadcast = broadcast_shapes(shape_x, shape_y,
                                                             param_name_input1="input_x",
                                                             param_name_input2="input_y")
        input_x = te.lang.cce.broadcast(input_x, shape_broadcast, "float32")
        input_y = te.lang.cce.broadcast(input_y, shape_broadcast, "float32")
    else:
        shape_broadcast = shape_x

    data_div = te.lang.cce.vdiv(input_x, input_y)
    data_zero = te.lang.cce.broadcast(tvm.const(0, "float32"), shape_broadcast,
                                      "float32")
    data_div_min = te.lang.cce.vmin(data_div, data_zero)
    data_div_max = te.lang.cce.vmax(data_div, data_zero)
    data_div_max_floor = te.lang.cce.floor(data_div_max)
    data_div_min_ceil = te.lang.cce.ceil(data_div_min)

    if dtype != "int32" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul", "float32"):
        data_div_max_floor = te.lang.cce.cast_to(data_div_max_floor, "float32")
        data_div_min_ceil = te.lang.cce.cast_to(data_div_min_ceil, "float32")

    data_div_res = te.lang.cce.vadd(data_div_max_floor, data_div_min_ceil)
    data_mul = te.lang.cce.vmul(data_div_res, input_y)
    res = te.lang.cce.vsub(input_x, data_mul)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, dtype)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def mod(input_x, input_y, output_z, kernel_name="mod"):
    """
    Returns element-wise remainder of division.

    Parameters
    ----------
    input_x: dict
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32", "int8", "uint8", "int32".
    input_y: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_x'.
    output_z: dict
        data of output.
        Must have the same type as 'input_x'.
    kernel_name: str
        kernel name, default value is "mod"

    Returns:
    None
    """
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")

    util.compare_tensor_dict_key(input_x, input_y, "dtype")
    check_shape(shape_x, param_name="input_x")
    check_shape(shape_y, param_name="input_y")

    check_list = ("float16", "float32", "int8", "uint8", "int32")
    input_dtype = input_x.get("dtype").lower()
    check_dtype(input_dtype, check_list, param_name="input_x")
    shape_x, shape_y, shape_broadcast = broadcast_shapes(shape_x, shape_y, param_name_input1="input_x", param_name_input2="input_y")


    reshape_x, reshape_y = refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(reshape_x, dtype=input_dtype, name="data_x")
    data_y = tvm.placeholder(reshape_y, dtype=input_dtype, name="data_y")
    res = mod_compute(data_x, data_y, output_z, kernel_name="mod")

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}

    te.lang.cce.cce_build_code(sch, config)
