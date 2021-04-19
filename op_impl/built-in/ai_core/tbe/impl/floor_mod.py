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
floor_mod
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast
from te import platform as tbe_platform
from te.utils.op_utils import *

# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
@fusion_manager.register("floor_mod")
def floor_mod_compute(x1, x2, y, kernel_name="floor_mod"):
    """
    Compute remainder of division
    res = x1 -floor(input_data_x / input_data_y)* input_data_y

    Parameters
    ----------
    x1: TVM tensor
        input tensor has shape and dtype attributes
    x2: TVM tensor
        input tensor has shape and dtype attributes
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        cce kernel name, default value is "floor_mod"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """
    # calculate result, using float32 for better precision
    dtype = x1.dtype
    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    shape_x, shape_y, shape = broadcast_shapes(shape_x, shape_y, param_name_input1="x1", param_name_input2="x2")

    has_improve_precision = False
    input_x_fp32 = x1
    input_y_fp32 = x2
    if tbe_platform.cce_conf.api_check_support("te.lang.cce.vlog", "float32"):
        input_x_fp32 = te.lang.cce.cast_to(x1, "float32")
        input_y_fp32 = te.lang.cce.cast_to(x2, "float32")
        has_improve_precision = True

    input_x_fp32 = te.lang.cce.broadcast(input_x_fp32, shape)
    input_y_fp32 = te.lang.cce.broadcast(input_y_fp32, shape)

    res = te.lang.cce.vdiv(input_x_fp32, input_y_fp32)

    res = te.lang.cce.floor(res)

    if dtype != "int32":
        if has_improve_precision:
            res = te.lang.cce.cast_to(res, "float32")
        else:
            res = te.lang.cce.cast_to(res, "float16")
        res = te.lang.cce.vmul(res, input_y_fp32)
        res = te.lang.cce.vsub(input_x_fp32, res)
        if has_improve_precision:
            res = te.lang.cce.cast_to(res, dtype)
    else:
        x2_broad = te.lang.cce.broadcast(x2, shape)
        x1_broad = te.lang.cce.broadcast(x1, shape)
        res = te.lang.cce.vmul(res, x2_broad)
        res = te.lang.cce.vsub(x1_broad, res)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def floor_mod(x1, x2, y, kernel_name="floor_mod"):
    """
    calculate the remainder of division, support fp16,fp32,int32
    res = x1 -floor(input_data_x / input_data_y)* input_data_y

    Parameters
    ----------
    x1: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32
    x2: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32
    y: dict, reserved field
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "floor_mod"

    Returns
    ------
    None
    """
    # get dtype and shape attributes
    dtype_x = x1.get("dtype").lower()
    shape_x = x1.get("shape")
    dtype_y = x2.get("dtype").lower()
    shape_y = x2.get("shape")

    # check_kernel_name & shape
    check_shape(shape_x, param_name="x1")
    check_shape(shape_y, param_name="x2")

    # check input tensor data_type
    check_list = ("float16", "float32", "int32")
    check_dtype(dtype_x, check_list, param_name="x1")
    check_dtype(dtype_y, check_list, param_name="x2")

    if dtype_x != dtype_y:
        raise RuntimeError("the type of dtype in two dict is not the same")

    shape_x, shape_y, shape_max = broadcast_shapes(shape_x, shape_y, param_name_input1="x1", param_name_input2="x2")
    shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)

    input_data_x = tvm.placeholder(shape_x, name="input_data_x", dtype=dtype_x)
    input_data_y = tvm.placeholder(shape_y, name="input_data_y", dtype=dtype_y)
    res = floor_mod_compute(input_data_x, input_data_y, y, kernel_name)
    with tvm.target.cce():
        auto_sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data_x, input_data_y, res]}
    te.lang.cce.cce_build_code(auto_sch, config)
