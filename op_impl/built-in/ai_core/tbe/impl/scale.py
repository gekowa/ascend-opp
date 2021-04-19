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
scale
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
from te.utils import op_utils
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

NONETYPE = type(None)


def check_param_range(param_name, min_value, max_value, real_value, op_name='ssd_detection_output'):
    
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
                         

# pylint: disable=too-many-arguments,unused-argument,invalid-name,redefined-outer-name
# pylint: disable=too-many-boolean-expressions,too-many-locals,unused-variable
def op_select_format(x, scale, bias, y, axis=1, num_axes=1, scale_from_blob=True,
                     kernel_name="scale"):
    """
    select format dynamically
    """
    shape_x_ori = x.get("ori_shape")
    shape_x = x.get("shape")
    shape_scale_ori = scale.get("ori_shape")
    shape_scale = scale.get("shape")

    length_x_ori = len(shape_x_ori)
    length_x = len(shape_x)
    length_scale_ori = len(shape_scale_ori)
    length_scale = len(shape_scale)

    if length_scale == 1 and shape_scale[0] == 1:
        format_scale = "ND,ND,ND,ND"
        format_bias = "ND,ND,ND,ND"
        format_scale_hisi = "ND,ND"
        format_bias_hisi = "ND,ND"
    else:
        format_scale = "NC1HWC0,NC1HWC0,ND,ND"
        format_bias = "NC1HWC0,NC1HWC0,ND,ND"
        format_scale_hisi = "NC1HWC0,ND"
        format_bias_hisi = "NC1HWC0,ND"

    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if length_x_ori == 4:
        # NC1HWC0+ND
        if product_version in ("Hi3796CV300ES", "Hi3796CV300CS"):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float16",
                               format="NC1HWC0,ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16,float16",
                               format=format_scale_hisi)
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16,float16",
                               format=format_bias_hisi)
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16",
                                format="NC1HWC0,ND")
        else:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float,float16,float",
                               format="NC1HWC0,NC1HWC0,ND,ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16,float,float16,float",
                               format=format_scale)
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16,float,float16,float",
                               format=format_bias)
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float,float16,float",
                                format="NC1HWC0,NC1HWC0,ND,ND")
    else:
        # ND+ND
        if product_version in ("Hi3796CV300ES", "Hi3796CV300CS"):
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16",
                               format="ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16",
                               format="ND")
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16",
                               format="ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16",
                                format="ND")
        else:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16,float",
                               format="ND,ND")
            input1 = gen_param(classify="input1", name="scale",
                               datatype="float16,float",
                               format="ND,ND")
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16,float",
                               format="ND,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float",
                                format="ND,ND")

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def param_scale_check(shape_x, shape_scale):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_scale : list or tuple.
        shape of scale.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if not(length_scale == 1 and shape_scale[0] == 1):
        if length_x != length_scale:
            
            error_info = {}
            error_info['errCode'] = 'E81014'
            error_info['real_x_dims'] = str(length_x)
            error_info['real_scale_dims'] = str(length_scale)
            raise RuntimeError(error_info, 
                "In op[scale], the dims of input tensor x and tensor scale should be equal, but actually are [%s] and [%s]."
                % (error_info['real_x_dims'], error_info['real_scale_dims']))
        
        for i in range(length_scale):
            if shape_scale[i] != shape_x[i] and shape_scale[i] != 1:
            
                error_info = {}
                error_info['errCode'] = 'E80013'
                error_info['opname'] = 'scale'
                error_info['input1_name'] = 'x'
                error_info['input2_name'] = 'scale'
                error_info['input1_shape'] = str(shape_x)
                error_info['input2_shape'] = str(shape_scale)
                raise RuntimeError(error_info, 
                    "In op[%s], the inputs[%s][%s] could not be broadcast together with shapes[%s][%s]."
                    % (error_info['opname'], error_info['input1_name'], error_info['input2_name'],
                       error_info['input1_shape'], error_info['input2_shape']))


def get_param_scale_shape(shape_x, shape_scale):
    """
    Function to calculate the shape of scale.

    Parameters
    ----------
    shape_x : list or tuple.
        shape of x.
    shape_scale : list or tuple.
        shape of scale.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if length_scale == 1 and shape_scale[0] == 1:
        shape = [1] * length_x
    else:
        shape = list(shape_scale)

    return shape


def _check_dtype(input_dtype, name):
    """
    Function to check dtype of input data.

    Parameters
    ----------

    input_dtype: str
        dtype of input data
    Returns
    -------
    None
    """

    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    if product_version in ("Hi3796CV300ES", "Hi3796CV300CS"):
        if input_dtype == "float32":
            raise RuntimeError("float32 is not support in ES")
        op_utils.check_dtype(input_dtype, ["float16"], param_name=name)
    else:
        op_utils.check_dtype(input_dtype, ["float16", "float32"], param_name=name)


# pylint: disable=too-many-branches
def _check_scale_shape_axis(shape_x, shape_scale, axis, num_axes, scale_from_blob):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        scale's data shape
    axis : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes:
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.

    Returns
    -------
    None
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)

    if (axis >= length_x) or (axis < (-length_x)):
        error_info['errCode'] = 'E80002'
        error_info['opname'] = 'scale'
        error_info['param_name'] = 'axis'
        error_info['min_value'] = str(-length_x)
        error_info['max_value'] = str(length_x - 1)
        error_info['real_value'] = str(axis)
        raise RuntimeError(error_info, "In op[%s], the parameter[%s] should be in the range of [%s, %s], but actually is [%s]."
                           % (error_info['opname'], error_info['param_name'], error_info['min_value'],
                              error_info['max_value'], error_info['real_value']))

    if num_axes < -1:
        error_info['errCode'] = 'E81015'
        error_info['opname'] = 'scale'
        error_info['param_name'] = 'num_axes'
        error_info['real_value'] = str(num_axes)
        raise RuntimeError(error_info, "In op[scale], the parameter[%s] should be be non-negative or -1, but actually is [%s]."
                           % (error_info['param_name'], error_info['real_value']))

    if axis < 0:
        axis_ = length_x + axis
    else:
        axis_ = axis

    # from blob
    if scale_from_blob:
        if num_axes == -1:
            scale_num = length_x - axis_
            if length_scale != scale_num:
                raise RuntimeError(
                    "length_scale and scale_num must be equal")
            for i in range(scale_num):
                if shape_x[axis_ + i] != shape_scale[i]:
                    raise RuntimeError(
                        "Dimensions shape_x and shape_scale must be equal")
        if num_axes == 0:
            if length_scale != 1 or shape_scale[0] != 1:
                raise RuntimeError("scale must be a scalar ")
        if num_axes > 0:
            num_axis = axis_ + num_axes
            if num_axis > length_x:
                raise RuntimeError(
                    "scale shape extends x shape when applied")
            if length_scale != num_axes:
                raise RuntimeError(
                    "length_scale and num_axes must be equal")
            for i in range(num_axes):
                if shape_x[axis_ + i] != shape_scale[i]:
                    raise RuntimeError(
                        "dimensions shape_x and shape_scale must be equal")

    # from bottom
    if not scale_from_blob:
        if not(length_scale == 1 and shape_scale[0] == 1):
            scale_num = axis_ + length_scale
            if scale_num > length_x:
                raise RuntimeError(
                    "scale shape extends x shape when applied")
            for i in range(length_scale):
                if shape_x[axis_ + i] != shape_scale[i]:
                    raise RuntimeError(
                        "Dimensions shape_x and shape_scale must be equal")


def get_scale_shape(shape_x, shape_scale, axis_, num_axes, scale_from_blob):
    """
    Function to calculate shape of scale.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape_scale: list or tuple
        scale's data shape
    axis_ : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes:
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.

    Returns
    -------
    shape: list or tuple
        the shape of scale
    """

    length_x = len(shape_x)
    length_scale = len(shape_scale)
    if scale_from_blob:
        if num_axes == -1:
            shape_left = [1] * axis_
            shape = shape_left + list(shape_scale)
        elif num_axes == 0:
            shape = [1] * length_x
        else:
            left_length = length_x - num_axes - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_scale) + shape_right
    else:
        if length_scale == 1 and shape_scale[0] == 1:
            shape = [1] * length_x
        else:
            left_length = length_x - length_scale - axis_
            shape_left = [1] * axis_
            shape_right = [1] * left_length
            shape = shape_left + list(shape_scale) + shape_right

    return shape


def get_fusion_params(x_tensor, scale_tensor, bias_tensor, y):
    """
    Get L1 fusion_params
    Parameters
    ----------
    x_tensor : tensor of input data
    y : dict of output data
    x_tensor_num: input tensor num
    Returns
    -------
    fusion_params
    """
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    in_l1_flag_list = []
    in_valid_shape_list = []
    in_slice_offset_list = []
    in_select_read_flag_list = []
    is_l1_depth_fusion = False

    input_tensor = [x_tensor, scale_tensor, bias_tensor]
    for x_tensor in input_tensor:
        if x_tensor is not None:
            l1_fusion_type = x_tensor.op.attrs["L1_fusion_type"].value \
                if "L1_fusion_type" in x_tensor.op.attrs else -1
            if l1_fusion_type == 1:
                raise RuntimeError("Scale does not support l1 width fusion")
            is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
            in_l1_flag = x_tensor.op.attrs["addr_type"].value == 1 \
                if "addr_type" in x_tensor.op.attrs else False
            in_l1_flag_list.append(in_l1_flag)
            in_valid_shape = x_tensor.op.attrs["valid_shape"] \
                if "valid_shape" in x_tensor.op.attrs else []
            in_valid_shape_list.append(in_valid_shape)
            in_slice_offset = x_tensor.op.attrs["slice_offset"] \
                if "slice_offset" in x_tensor.op.attrs else []
            in_slice_offset_list.append(in_slice_offset)
            in_select_read_flag = x_tensor.op.tag == "read_select_5d"
            in_select_read_flag_list.append(in_select_read_flag)

    l1_fusion_type = 0 if is_l1_depth_fusion is True else -1
    out_l1_flag = y.get("addr_type", 0) == 1
    out_valid_shape = y.get("valid_shape", [])
    out_slice_offset = y.get("slice_offset", [])
    out_select_write_flag = bool(out_valid_shape)

    fusion_params = {"is_l1fusion": is_l1_depth_fusion,
                     "l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag_list,
                     "in_select_read_flag": in_select_read_flag_list,
                     "in_valid_shape": in_valid_shape_list,
                     "in_slice_offset": in_slice_offset_list,
                     "out_l1_flag": out_l1_flag,
                     "out_select_write_flag": out_select_write_flag,
                     "out_valid_shape": out_valid_shape,
                     "out_slice_offset": out_slice_offset}
    return fusion_params


# pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_compute(x, scale):
    """
    algorithm: Scale
    y = scale*x

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data

    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """

    dtype_x = x.dtype
    dtype_scale = scale.dtype

    is_cast = False
    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS"):
        if dtype_x == "float16":
            is_cast = True
            x = te.lang.cce.cast_to(x, 'float32')
        if dtype_scale == "float16":
            scale = te.lang.cce.cast_to(scale, 'float32')

    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    scale_broad = te.lang.cce.broadcast(scale, shape_x)

    res = te.lang.cce.vmul(x, scale_broad)

    if is_cast:
        res = te.lang.cce.cast_to(res, dtype_x)

    return res


# pylint: disable=invalid-name,redefined-outer-name
def _fused_scale_bias_compute(x, scale, bias):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data
    bias : TVM tensor
        contains bias data
    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """

    dtype_x = x.dtype
    dtype_scale = scale.dtype
    dtype_bias = bias.dtype

    is_cast = False
    product_version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if product_version not in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS"):
        if dtype_x == "float16":
            is_cast = True
            x = te.lang.cce.cast_to(x, 'float32')
        if dtype_scale == "float16":
            scale = te.lang.cce.cast_to(scale, 'float32')
        if dtype_bias == "float16":
            bias = te.lang.cce.cast_to(bias, 'float32')

    shape_x = te.lang.cce.util.shape_to_list(x.shape)

    scale_broad = te.lang.cce.broadcast(scale, shape_x)
    bias_broad = te.lang.cce.broadcast(bias, shape_x)

    res_tmp = te.lang.cce.vmul(x, scale_broad)
    res = te.lang.cce.vadd(res_tmp, bias_broad)

    if is_cast:
        res = te.lang.cce.cast_to(res, dtype_x)

    return res


# pylint: disable=too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("scale")
def scale_compute(x, scale, bias, y, axis, num_axes, scale_from_blob,
                  kernel_name="scale"):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : TVM tensor
        contains x data
    scale : TVM tensor
        contains scale data
    bias : TVM tensor
        contains bias data
    y : dict
        dict of output,
        A Tensor for output, should be same shape and type as x.
    axis : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes: int
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.
    kernel_name : str
        kernel name, default value is "scale"

    Returns
    -------
    res: TVM tensor list
        the result of scale compute
    """
    tmp_y = {}
    tmp_y["addr_type"] = 0
    tmp_y["valid_shape"] = []
    tmp_y["slice_offset"] = []
    fuse_y = tmp_y if y is None else y
    fusion_params = get_fusion_params(x, scale, bias, fuse_y)

    res = None
    if bias is not None:
        res = _fused_scale_bias_compute(x, scale, bias)
    else:
        res = _fused_scale_compute(x, scale)

    res.op.attrs["ele_fusion_params"] = fusion_params

    return res


# pylint: disable=too-many-locals,no-member,invalid-name,too-many-statements,line-too-long
@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT, op_utils.OPTION_INPUT, op_utils.REQUIRED_OUTPUT,
                          op_utils.OPTION_ATTR_INT, op_utils.OPTION_ATTR_INT, op_utils.OPTION_ATTR_BOOL, op_utils.KERNEL_NAME)
def scale(x, scale, bias, y, axis=1, num_axes=1, scale_from_blob=True,
          kernel_name="scale"):
    """
    algorithm: Scale
    y = scale*x + bias

    Parameters
    ----------
    x : dict
        dict of input, A Tensor for input data.
    scale : dict
        dict of scale,
        A Tensor for scaling factor, to scale the input data.
    bias : dict
        dict of bias,
        A Tensor for bias, to shift to the input data.
    y : dict
        dict of output,
        A Tensor for y, should be same shape and type as x.
    axis : int
        A int num indicates shape of scale when scale is from bottom.
    num_axes: int
        A int num indicates shape of scale when scale is from blob.
    scale_from_blob:
        A bool value indicates scale is from blob or bottom.
    kernel_name : str
        kernel name, default value is "scale"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    op_utils.check_shape(shape_x, param_name="input_x")
    _check_dtype(dtype_x.lower(), "input_x")

    shape_scale = scale.get("shape")
    dtype_scale = scale.get("dtype")
    op_utils.check_shape(shape_scale, param_name="input_scale")
    _check_dtype(dtype_scale.lower(), "input_scale")

    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        dtype_bias = bias.get("dtype")
        op_utils.check_shape(shape_bias, param_name="input_bias")
        _check_dtype(dtype_bias.lower(), "input_bias")

    shape_x_ori = x.get("ori_shape")
    length_x_ori = len(shape_x_ori)

    shape_scale_new = []
    shape_bias_new = []

    if length_x_ori == 4:
        param_scale_check(shape_x, shape_scale)
        shape_scale_new = get_param_scale_shape(shape_x, shape_scale)
        if len(shape_bias) > 0:
            shape_bias_new = shape_scale_new
    else:
        _check_scale_shape_axis(shape_x, shape_scale, axis, num_axes, scale_from_blob)

        length_x = len(shape_x)
        if axis < 0:
            axis_ = length_x + axis
        else:
            axis_ = axis

        shape_scale_new = get_scale_shape(shape_x, shape_scale, axis_, num_axes, scale_from_blob)
        if len(shape_bias) > 0:
            shape_bias_new = shape_scale_new

    input_list = [x, scale, bias]
    input_shape_list = [shape_x, shape_scale_new, shape_bias_new]
    name_list = ["x", "scale", "bias"]
    input_tensor_list = []
    is_l1_depth_fusion = False
    for input_, input_shape, name_ in \
        zip(input_list, input_shape_list, name_list):
            if len(input_shape) > 0:
                l1_fusion_type = input_.get("L1_fusion_type", -1)
                if l1_fusion_type == 1:
                    raise RuntimeError("scale does not support l1 width fusion")
                is_l1_depth_fusion = (l1_fusion_type == 0) or is_l1_depth_fusion
                dtype = input_.get("dtype")
                addr_type = input_.get("addr_type", 0)
                valid_shape = input_.get("valid_shape", [])
                slice_offset = input_.get("slice_offset", [])
                attr_x = {"addr_type": addr_type,
                          "valid_shape": valid_shape,
                          "slice_offset": slice_offset,
                          "L1_fusion_type": l1_fusion_type}
                input_tensor = tvm.placeholder(input_shape, name=name_,
                                               dtype=dtype, attrs=attr_x)
                input_tensor_list.append(input_tensor)

    if len(shape_bias) == 0:
        input_tensor_list.append(None)

    x_input, scale_input, bias_input = input_tensor_list
    res = scale_compute(x_input, scale_input, bias_input, y,
                        axis, num_axes, scale_from_blob, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = (x_input, scale_input, res)
    if len(shape_bias) > 0:
        tensor_list = (x_input, scale_input, bias_input, res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list,
              "l1_fusion_option": is_l1_depth_fusion}
    te.lang.cce.cce_build_code(sch, config)
