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
dynamic floor_mod
"""
import te.lang.dynamic
from te import platform as tbe_platform
from te import tvm
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_elewise_shape_range
from te.utils.op_utils import variable_shape
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import OP_ERROR_CODE_018
from topi import generic


# pylint: disable=locally-disabled,unused-argument,invalid-name
def floor_mod_compute(x1, x2, y, kernel_name="floor_mod"):
    """
    Compute remainder of division
    res = x1 - floor(input_data_x / input_data_y) * input_data_y

    Parameters
    ----------
    x1: TVM tensor
        input tensor has shape, dtype and range attributes
    x2: TVM tensor
        input tensor has shape, dtype and range attributes
    y: dict
        dict with keys(shape, dtype and range) of output
    kernel_name : str
        cce kernel name, default value is "floor_mod"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """

    dtype = x1.dtype
    shape_x = te.lang.dynamic.shape_to_list(x1.shape)
    shape_y = te.lang.dynamic.shape_to_list(x2.shape)

    shape_x, shape_y, shape = broadcast_shapes(shape_x, shape_y,
                                               param_name_input1="x1",
                                               param_name_input2="x2")

    # calculate result, using float32 for better precision
    has_improve_precision = False
    input_x_fp32 = x1
    input_y_fp32 = x2
    if tbe_platform.cce_conf.api_check_support("te.lang.dynamic.vdiv",
                                               "float32"):
        input_x_fp32 = te.lang.dynamic.cast_to(x1, "float32")
        input_y_fp32 = te.lang.dynamic.cast_to(x2, "float32")
        has_improve_precision = True

    input_x_fp32 = te.lang.dynamic.broadcast(input_x_fp32, shape)
    input_y_fp32 = te.lang.dynamic.broadcast(input_y_fp32, shape)

    res = te.lang.dynamic.vdiv(input_x_fp32, input_y_fp32)

    if tbe_platform.cce_conf.api_check_support("te.lang.dynamic.floor",
                                               res.dtype):
        res = te.lang.dynamic.floor(res)
    else:
        res = te.lang.dynamic.cast_to(res, "float16")
        res = te.lang.dynamic.floor(res)

    if dtype != "int32":
        if has_improve_precision:
            res = te.lang.dynamic.cast_to(res, "float32")
        else:
            res = te.lang.dynamic.cast_to(res, "float16")
        res = te.lang.dynamic.vmul(res, input_y_fp32)
        res = te.lang.dynamic.vsub(input_x_fp32, res)
        if has_improve_precision:
            res = te.lang.dynamic.cast_to(res, dtype)
    else:
        x2_broad = te.lang.dynamic.broadcast(x2, shape)
        x1_broad = te.lang.dynamic.broadcast(x1, shape)
        res = te.lang.dynamic.vmul(res, x2_broad)
        res = te.lang.dynamic.vsub(x1_broad, res)

    return res


@te.op.register_operator("FloorMod")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def floor_mod(x1, x2, y, kernel_name="floor_mod"):
    """
    calculate the remainder of division, support fp16,fp32,int32
    res = x1 -floor(input_data_x / input_data_y)* input_data_y

    Parameters
    ----------
    x1: dict
        dict{"shape":tuple or list,"dtype":str, "range": tuple or list}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32
    x2: dict
        dict{"shape":tuple or list,"dtype":str, "range": tuple or list}
        shape of data
        the data type, src_dtype equals  of dst_dtype, support fp16,fp32,int32
    y: dict, reserved field
        dict with keys(shape, dtype and range) of output
    kernel_name: str
        cce kernel name, default value is "floor_mod"

    Returns
    ------
    None
    """

    # check input tensor data_type
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
    check_dtype(dtype_x, check_list, param_name="x1")
    check_dtype(dtype_y, check_list, param_name="x2")
    check_elewise_shape_range([x1, x2], support_broadcast=True)

    if dtype_x != dtype_y:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_018
        error_info['op_name'] = 'floor_mod'
        error_info['param_name1'] = 'dtype_x'
        error_info['param_name2'] = 'dtype_y'
        error_info['param1_dtype'] = str(dtype_x)
        error_info['param2_dtype'] = str(dtype_y)
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s][%s] are not equal in "
                           "dtype with dtype[%s][%s]." % (
                               error_info['op_name'],
                               error_info['param_name1'],
                               error_info['param_name2'],
                               error_info['param1_dtype'],
                               error_info['param2_dtype']))

    ins = classify([x1, x2], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with te.op.compute():
            shape_x, shape_y = variable_shape([x1, x2], support_broadcast=True)
            shape_x, shape_y = refine_shapes_for_broadcast(shape_x, shape_y)
            input_data_x = tvm.placeholder(shape_x, name="input_data_x",
                                           dtype=dtype_x)
            input_data_y = tvm.placeholder(shape_y, name="input_data_y",
                                           dtype=dtype_y)
            res = floor_mod_compute(input_data_x, input_data_y, y, kernel_name)

            tensors.append([input_data_x, input_data_y, res])
        with tvm.target.cce():
            auto_sch = generic.auto_schedule(res)
        schedules.append(auto_sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    te.lang.dynamic.build(schedules, config)
