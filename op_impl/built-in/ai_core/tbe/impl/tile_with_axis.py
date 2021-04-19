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
tile_with_axis
"""
# pylint: disable=import-error
import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils import op_utils
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


@fusion_manager.register("tile_with_axis")
def tile_with_axis_compute(data, shape_y):
    """TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    shape_y: tuple or list.
        The shape of output.

    Returns
    -------
    res the compute results
    """
    res = te.lang.cce.broadcast(data, shape_y)

    return res


# pylint: disable=unused-argument
def op_select_format(input_x, output_y, tiles, axis=1, kernel_name="tile_with_axis"):
    """
    select format dynamically
    """
    ori_format = input_x.get("ori_format")
    ori_shape = input_x.get("ori_shape")

    if ori_shape is not None:
        axis = util.axis_check(len(ori_shape), axis)

    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    # for 5hd, axis is only valid for n,h,w
    if ((ori_format == "NHWC" and axis != 3) or (ori_format == "NCHW" and axis != 1)) and \
            len(ori_shape) == 4:
        # NC1HWC0+ND
        if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
            # fp16
            input0 = gen_param(
                classify="input0", name="x",
                datatype="float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64,"
                         "float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,"
                       "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
            output0 = gen_param(
                classify="output0", name="y",
                datatype="float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64,"
                         "float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,"
                       "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
        else:
            # fp16/fp32
            input0 = gen_param(
                classify="input0", name="x",
                datatype="float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64,"
                         "float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,NC1HWC0,"
                       "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
            output0 = gen_param(
                classify="output0", name="y",
                datatype="float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64,"
                         "float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,NC1HWC0,"
                       "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
    else:
        # ND
        if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS"):
            # fp16
            input0 = gen_param(
                classify="input0", name="x",
                datatype="float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND")
            output0 = gen_param(
                classify="output0", name="y",
                datatype="float16,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND")
        else:
            # fp16/fp32
            input0 = gen_param(
                classify="input0", name="x",
                datatype="float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,ND")
            output0 = gen_param(
                classify="output0", name="y",
                datatype="float16,float32,int8,int16,int32,int64,uint8,uint16,uint32,uint64",
                format="ND,ND,ND,ND,ND,ND,ND,ND,ND,ND")

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_OUTPUT,
                          op_utils.REQUIRED_ATTR_INT, op_utils.OPTION_ATTR_INT,
                          op_utils.KERNEL_NAME)
def tile_with_axis(input_x, output_y, tiles, axis=1, kernel_name="tile_with_axis"):
    """
    algorithm: tile.
    Expanding the input tensor according to a specified dimension,
    and the expansion multiple is specified by the tiles param.
    For example, tiling [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11,
    12]]], which shape is (2, 3, 2), by axis:1 and tiles:2 produces
    [[[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]],
    [[7, 8], [9, 10], [11, 12], [7, 8], [9, 10], [11, 12]]]
    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same type as input
    axis: int
         The index of the axis to tile
    tiles: int
        The number of copies (tiles) of the blob to output.
    kernel_name : str
        kernel name, default value is "tile_with_axis"

    Returns
    -------
    tik_instance
    """

    axis, shape_x, shape_y, dtype_x = check_param(input_x, output_y, tiles,
                                                  axis, kernel_name)

    input_data = tvm.placeholder(shape_x, name="input_data", dtype=dtype_x)

    if tiles > 1:
        res = tile_with_axis_compute(input_data, shape_y)
    else:
        zero_data = tvm.const(0, dtype=dtype_x)
        res = te.lang.cce.vadds(input_data, zero_data)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [input_data, res]}

    te.lang.cce.cce_build_code(sch, config)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def check_param_range(param_name, min_value, max_value, real_value, op_name='tile_with_axis'):

    error_info = {}
    error_info['errCode'] = 'E80002'
    error_info['opname'] = op_name
    error_info['param_name'] = param_name
    error_info['min_value'] = str(min_value)
    error_info['max_value'] = str(max_value)
    error_info['real_value'] = str(real_value)
    raise RuntimeError(error_info, "In op[%s], the parameter[%s] should be in the range of [%s, %s], but actually is [%s]."
                       % (error_info['opname'], error_info['param_name'], error_info['min_value'],
                          error_info['max_value'], error_info['real_value']))
    

# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def check_param(input_x, output_y, tiles, axis, kernel_name):
    """
    Check the input parameter

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same type as input
    axis: int
         The index of the axis to tile
    tiles: int
        The number of copies (tiles) of the blob to output.
    kernel_name : str
        kernel name, default value is "tile_with_axis"

    Returns
    ----------
    axis: int
         The index of the axis to tile which is adjusted to positive
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype").lower()
    shape_y = output_y.get("shape")
    dtype_y = output_y.get("dtype").lower()
    
    op_utils.check_shape(shape_x, param_name="input_x")
    op_utils.check_shape(shape_y, param_name="input_y")

    check_list = ["int8", "int16", "int32", "int64", "uint8", "uint16",
                  "uint32", "uint64", "float16", "float32"]
    
    op_utils.check_dtype(dtype_x, check_list, param_name="input_x")
    op_utils.check_dtype(dtype_y, check_list, param_name="input_x")

    if dtype_x != dtype_y:
        error_info = {}
        error_info['errCode'] = 'E80019'
        error_info['op_name'] = 'tile_with_axis'
        error_info['input1_name'] = 'x'
        error_info['input2_name'] = 'y'
        error_info['input1_dtype'] = str(dtype_x)
        error_info['input2_dtype'] = str(dtype_y)
        raise RuntimeError(
            "In op[%s], the shape of input[%s] and input[%s] should be the same, but actually are [%s] and [%s]."
            % (error_info[op_name], error_info['input1_name'], error_info['input2_name'],
               error_info['input1_dtype'], error_info['input2_dtype']))

    if tiles <= 0:
        check_param_range('tiles', 1, 'inf', tiles)

    shape_x_len = len(shape_x)

    # check for 5HD
    input_format = input_x.get("format")
    if input_format == "NC1HWC0":
        shape_x_ori = input_x.get("ori_shape")
        ori_format = input_x.get("ori_format")
        length_x_ori = len(shape_x_ori)

        if ori_format not in ("NCHW", "NHWC"):
            raise RuntimeError("input_x's ori_format is invalid for 5D Tensor")
        if shape_x_len != 5:
            raise RuntimeError("input_x's shape is invalid for 5D Tensor")
        if length_x_ori != 4:
            raise RuntimeError("input_x's ori_shape is invalid for 5D Tensor")
        axis = util.axis_check(length_x_ori, axis)
        axis = util.axis_transfrom_5d(axis, ori_format)
        if axis in (1, 4):
            raise RuntimeError("axis is invalid for 5D Tensor")
    else:
        if axis >= shape_x_len or axis < -shape_x_len:
            check_param_range('axis', -shape_x_len, shape_x_len-1, axis)

        if axis < 0:
            axis += shape_x_len

    shape_y_expected = [0] * shape_x_len
    shape_y_expected[0:shape_x_len] = shape_x[0:shape_x_len]
    shape_y_expected[axis] *= tiles

    if not check_same_shape(shape_y, shape_y_expected):
        error_info = {}
        error_info['errCode'] = 'E80017'
        error_info['op_name'] = 'tile_with_axis'
        error_info['attr_name'] = 'shape_y'
        error_info['expect_value'] = str(shape_y_expected)
        error_info['real_value'] = str(shape_y)
        raise RuntimeError("In op[%s], the parameter[%s] should be [%s], but actually is [%s]."
                           % (error_info['op_name'], error_info['attr_name'], error_info['expect_value'], error_info['real_value']))

    shape_x_adapt = []
    shape_y_adapt = []
    for i in range(shape_x_len):
        if i == axis:
            shape_x_adapt.append(1)
            shape_y_adapt.append(tiles)
            if shape_x[i] == 1:
                continue
        shape_x_adapt.append(shape_x[i])
        shape_y_adapt.append(shape_x[i])

    return axis, shape_x_adapt, shape_y_adapt, dtype_x


def check_same_shape(shape_x, shape_y):
    """
    check shape_x is the same shape as shape_y

    Parameters
    ----------
    shape_x: a tuple or list
    shape_y: a tuple or list

    Returns
    -------
    boolean: True, if the same shape otherwise False
    """
    shape_x_len = len(shape_x)
    shape_y_len = len(shape_y)

    if shape_x_len != shape_y_len:
        return False

    for i in range(shape_x_len):
        if shape_x[i] != shape_y[i]:
            return False

    return True
