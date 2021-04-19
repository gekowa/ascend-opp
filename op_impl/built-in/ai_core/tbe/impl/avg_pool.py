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
avg_pool
"""
import te.lang.cce
from te import tvm
from te.utils.op_utils import *
from topi import generic
from te.platform.cce_policy import get_L1_info
from te.utils.error_manager import error_manager_util as err_mana


def get_fusion_params(input_data, output_data, is_fused_compute=True):
    """
    :param input_data: tensor of input_data
    :param output_data: dict of output_data
    :return: dict fusion_params
    """
    # l1 fusion params assign
    # 0: L1 depth fusion, 1: L1 width fusion, -1: no L1 fusion
    l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value \
        if "L1_fusion_type" in input_data.op.attrs else -1
    in_l1_flag = input_data.op.attrs["addr_type"].value == 1 \
        if "addr_type" in input_data.op.attrs else False
    in_valid_shape = input_data.op.attrs["valid_shape"] \
        if "valid_shape" in input_data.op.attrs else []
    in_slice_offset = input_data.op.attrs["slice_offset"] \
        if "slice_offset" in input_data.op.attrs else []
    in_select_read_flag = bool(in_valid_shape)
    in_split_index = input_data.op.attrs["split_index"].value \
        if "split_index" in input_data.op.attrs else 0
    out_l1_flag = output_data.get("addr_type") == 1
    fusion_params = {"is_fused_compute": is_fused_compute,
                     "l1_fusion_type": l1_fusion_type,
                     "in_l1_flag": in_l1_flag,
                     "out_l1_flag": out_l1_flag,
                     "in_select_read_flag": in_select_read_flag,
                     "in_split_index": in_split_index,
                     "in_slice_offset": in_slice_offset}

    return fusion_params


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    if isinstance(shape, (list, tuple)):
        return shape
    tmp = []
    if shape == "":
        return ()
    for i in shape:
        tmp.append(i.value)
    return tmp


def avgpool_conv2d_fusion_para(inputs, outputs):
    """
    get L1 fusion para for depthwise_conv2d
    """
    input_memory_type = inputs.op.attrs["addr_type"] \
        if "addr_type" in inputs.op.attrs else 0
    output_memory_type = outputs["addr_type"] \
        if "addr_type" in outputs else 0
    valid_shape = inputs.op.attrs["valid_shape"] \
        if "valid_shape" in inputs.op.attrs else ()
    slice_offset = inputs.op.attrs["slice_offset"] \
        if "slice_offset" in inputs.op.attrs else ()
    l1_fusion_type = inputs.op.attrs["L1_fusion_type"] \
        if "L1_fusion_type" in inputs.op.attrs else -1

    fmap_l1_addr_flag = inputs.op.attrs["L1_addr_flag"] \
        if "L1_addr_flag" in inputs.op.attrs else -1
    fmap_l1_valid_size = inputs.op.attrs["L1_valid_size"] \
        if "L1_valid_size" in inputs.op.attrs else -1

    l1_fusion_enable_flag = get_L1_info("L1_fusion_enabled")
    if not l1_fusion_enable_flag:
        l1_fusion_type = -1

    valid_shape = shape_to_list(valid_shape)
    slice_offset = shape_to_list(slice_offset)

    if not l1_fusion_enable_flag:
        input_memory_type = 0
        output_memory_type = 0
        valid_shape = []
        slice_offset = []
        l1_fusion_type = -1

    if int(input_memory_type) not in (0, 1, 2):
        err_man.raise_err_input_mem_type("depthwise_conv2d",
                                         input_memory_type)
    if int(output_memory_type) not in (0, 1, 2):
        err_man.raise_err_output_mem_type("depthwise_conv2d",
                                          output_memory_type)
    if valid_shape and not slice_offset:
        err_man.raise_err_specific_user(
            "depthwise_conv2d",
            "if valid_shape exists slice_offset can not be []")

    fusion_para = {"input_memory_type": input_memory_type,
                   "output_memory_type": output_memory_type,
                   "valid_shape": valid_shape,
                   "slice_offset": slice_offset,
                   "l1_fusion_type": l1_fusion_type,
                   "fmap_l1_addr_flag": fmap_l1_addr_flag,
                   "fmap_l1_valid_size": fmap_l1_valid_size}

    return fusion_para


# pylint: disable=locally-disabled,too-many-arguments,too-many-statements
# pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
def check_window_rule(ksize, strides, data_format):
    """
    check ksize and strides of window in pooling
    """
    if data_format in ("NHWC",):
        if len(ksize) != 4:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_012
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = 'ksize'
            errorInfo['min_value'] = '4'
            errorInfo['max_value'] = '4'
            errorInfo['real_value'] = len(ksize)
            raise RuntimeError(errorInfo,
                               "In op[%s], the num of dimensions of input[%s]"
                               "should be in the range of [%s, %s],"
                               "but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['min_value'], errorInfo['max_value'],
                                errorInfo['real_value']))

        elif ksize[0] != 1 or ksize[3] != 1:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_000
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = ",".join(("ksize[1]", "ksize[3]"))
            errorInfo['expected_value'] = '1'
            errorInfo['real_value'] = ",".join((ksize[1], ksize[3]))
            raise RuntimeError("In op[%s], the parameter[%s] should be [%s], "
                               "but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['expected_value'],
                                errorInfo['real_value']))
        if len(strides) != 4:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_012
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = 'strides'
            errorInfo['min_value'] = '4'
            errorInfo['max_value'] = '4'
            errorInfo['real_value'] = len(strides)
            raise RuntimeError(errorInfo,
                               "In op[%s], the num of dimensions of input[%s]"
                               "should be in the range of [%s, %s],"
                               "but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['min_value'], errorInfo['max_value'],
                                errorInfo['real_value']))
        elif strides[0] != 1 or strides[3] != 1:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_000
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = ",".join(("strides[1]", "strodes[3]"))
            errorInfo['expected_value'] = '1'
            errorInfo['real_value'] = ",".join((strides[1], strides[3]))
            raise RuntimeError("In op[%s], the parameter[%s] should be [%s],"
                               " but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['expected_value'],
                                errorInfo['real_value']))
    elif data_format in ("NC1HWC0", "NCHW"):
        if len(ksize) != 4:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_012
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = 'ksize'
            errorInfo['min_value'] = '4'
            errorInfo['max_value'] = '4'
            errorInfo['real_value'] = len(ksize)
            raise RuntimeError(errorInfo,
                               "In op[%s], the num of dimensions of input[%s]"
                               "should be in the range of [%s, %s],"
                               "but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['min_value'], errorInfo['max_value'],
                                errorInfo['real_value']))
        elif ksize[0] != 1 or ksize[1] != 1:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_000
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = ",".join(("ksize[0]", "ksize[1]"))
            errorInfo['expected_value'] = '1'
            errorInfo['real_value'] = ",".join((ksize[0], ksize[1]))
            raise RuntimeError("In op[%s], the parameter[%s] should be [%s],"
                               " but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['expected_value'],
                                errorInfo['real_value']))
        if len(strides) != 4:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_012
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = 'strides'
            errorInfo['min_value'] = '4'
            errorInfo['max_value'] = '4'
            errorInfo['real_value'] = len(strides)
            raise RuntimeError(errorInfo,
                               "In op[%s], the num of dimensions of input[%s]"
                               "should be in the range of [%s, %s], but"
                               "actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['min_value'], errorInfo['max_value'],
                                errorInfo['real_value']))
        elif strides[0] != 1 or strides[1] != 1:
            errorInfo = {}
            errorInfo['errCode'] = OP_ERROR_CODE_000
            errorInfo['op_name'] = 'avg_pool'
            errorInfo['param_name'] = ",".join(("strides[0]", "strodes[1]"))
            errorInfo['expected_value'] = '1'
            errorInfo['real_value'] = ",".join((strides[1], strides[1]))
            raise RuntimeError("In op[%s], the parameter[%s] should be [%s],"
                               " but actually is [%s]." %
                               (errorInfo['op_name'], errorInfo['param_name'],
                                errorInfo['expected_value'],
                                errorInfo['real_value']))
    else:
        errorInfo = {}
        errorInfo['errCode'] = OP_ERROR_CODE_015
        errorInfo['op_name'] = 'avg_pool'
        errorInfo['param_name'] = 'x'
        errorInfo['excepted_format_list'] = ",".join(("NC1HWC0",
                                                      "NCHW", "NHWC"))
        errorInfo['format'] = data_format
        raise RuntimeError(errorInfo, "In op[%s], the format[%s] of input"
                                      "should be one of [%s],"
                                      "but actually is [%s]."
                           % (errorInfo['op_name'], errorInfo['param_name'],
                              errorInfo['excepted_format_list'],
                              errorInfo['format']))


def get_corrected_pad(input_pad):
    """
    algorithm:
    get corrected pad value

    Parameters
    ----------
    input_pad: the value of pad
    Returns
    -------
    output_pad: the value of pad
    """
    if input_pad < 0:
        output_pad = 0
    else:
        output_pad = input_pad
    return output_pad


def avg_pool_check_rule(input_shape, input_dtype,
                        output_dtype, input_format, ksize, strides,
                        data_format, kernel_name):
    """
    :param input_shape: shape of input_data
    :param input_dtype: dtype of input_data
    :param output_dtype: dtype of output_data
    :param ksize: the window of avgpooling
    :param strides: the stride of avgpooling window
    :param data_format: NHWC default
    :param kernel_name: cce kernel name
    :return: None

    """
    # check input and output
    check_shape(input_shape)
    check_dtype(input_dtype, ["float16", "int8"])
    check_dtype(output_dtype, ["float16", "int8", "int32"])

    check_window_rule(ksize, strides, data_format)


def avg_pool_compute1(x, y, ksize, strides,
                      padding="VALID", data_format="NHWC",
                      is_fused_compute=True,
                      kernel_name="avg_pool"):
    """
    describe compute
    return: tensor
    """
    # create window and stride for pooling2d
    if data_format in ("NHWC",):
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
    else:
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]

    window = list(window)
    stride = list(stride)

    # l1 fusion and l2 fusion
    l1_fusion_type = x.op.attrs["L1_fusion_type"].value \
        if "L1_fusion_type" in x.op.attrs else -1
    fusion_params = get_fusion_params(x, y, is_fused_compute)
    in_select_read_flag = fusion_params.get("in_select_read_flag")
    in_valid_shape = fusion_params.get("in_valid_shape")
    in_slice_offset = fusion_params.get("in_slice_offset")

    if in_select_read_flag:
        select_tensor_in = tvm.compute(in_valid_shape,
                                       lambda n, c1, h, w, c0:
                                       x(n, c1, h + in_slice_offset[2], w, c0),
                                       name="tensor_read_select",
                                       attrs=x.op.attrs)
        res = te.lang.cce.pooling2d(select_tensor_in, window, stride, "AVG",
                                    padding, fusion_params=fusion_params)
    elif l1_fusion_type == 1:
        x.op.attrs["addr_type"].value = 1
        in_l1_flag = True
        fusion_params["in_l1_flag"] = in_l1_flag

        l1_width_fusion_in = tvm.compute(x.shape,
                                         lambda n, c1, h, w, c0:
                                         x(n, c1, h, w, c0),
                                         name="l1_width_fusion_tensor_in",
                                         attrs=x.op.attrs)
        res = te.lang.cce.pooling2d(l1_width_fusion_in, window, stride,
                                    "AVG", padding,
                                    fusion_params=fusion_params)
    else:
        res = te.lang.cce.pooling2d(x, window, stride, "AVG", padding,
                                    fusion_params=fusion_params)

    return res


# pylint: disable=unnecessary-lambda,redefined-builtin,too-many-locals
# pylint: disable=unnecessary-lambda,too-many-statements
@fusion_manager.register("avg_pool")
def avg_pool_compute(x, filter, bias, y, ksize, strides, padding="VALID",
                     data_format="NHWC", offset_x=0, kernel_name="avg_pool"):
    """
    algorithm: avg_pool
    calculating the average pooling

    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16
    filter : dict, shape and dtype of input_data, only support float16
    y : dict, shape and dtype of output_data, only support float16
    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W
    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W
    padding : str, the mode of padding, support padding and not padding
    data_format : str, default = "NHWC"
    kernel_name : kernel name, default value is "avg_pool"

    Returns
    -------
    None
    """
    out_dtype = y.get("dtype")
    # create window and stride for pooling2d
    if data_format in ("NHWC",):
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
    else:
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]

    shape_x = x.shape
    input_h = shape_x[2]
    input_w = shape_x[3]
    dilations = (1, 1)

    dsl_flag = True

    if padding == "SAME":
        output_h = (input_h + stride[0] - 1) // stride[0]
        output_w = (input_w + stride[1] - 1) // stride[1]
        pad_row = (output_h - 1) * stride[0] + \
                  ((window[0] - 1) * dilations[0] + 1) - input_h
        pad_col = (output_w - 1) * stride[1] + \
                  ((window[1] - 1) * dilations[1] + 1) - input_w
        pad_top = pad_row // 2
        pad_bottom = pad_row - pad_top
        pad_left = pad_col // 2
        pad_right = pad_col - pad_left
        pad_top = get_corrected_pad(int(pad_top))
        pad_bottom = get_corrected_pad(int(pad_bottom))
        pad_left = get_corrected_pad(int(pad_left))
        pad_right = get_corrected_pad(int(pad_right))
        pad = (pad_top, pad_bottom, pad_left, pad_right)
    else:
        pad = (0, 0, 0, 0)
    if int(input_h) == int(window[0]) and int(input_h) == int(window[1]):
        res = avg_pool_compute1(x, y, ksize, strides, padding, data_format,
                                is_fused_compute=True, kernel_name=kernel_name)
    else:
        l1_fusion_para = avgpool_conv2d_fusion_para(x, y)
        res = te.lang.cce.te_compute.depthwise_conv2d_compute(
            x, filter, out_dtype.lower(), stride, pad, dilations, {
                "bias_tensor": bias, "dsl_flag": dsl_flag,
                "offset_x": offset_x}, l1_fusion_para, kernel_name)

    return res


@check_op_params(REQUIRED_INPUT, OPTION_INPUT, OPTION_INPUT, REQUIRED_OUTPUT,
                 REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_LIST_INT,
                 REQUIRED_ATTR_STR, OPTION_ATTR_STR, OPTION_ATTR_INT,
                 KERNEL_NAME)
def avg_pool(x, filter, bias, y, ksize, strides,
             padding="VALID", data_format="NHWC", offset_x=0,
             kernel_name="avg_pool"):
    """
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, shape is 4
        dims, format is NCHW

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of avgpooling, only support avgpooling
            in H or W

    strides : list or tuple, the stride of avgpooling window, only support
              avgpooling in H or W

    padding : str, the mode of padding, support padding and not padding

    data_format : str, default = "NHWC"

    kernel_name : cce kernel name, default value is "avg_pool_cce"

    Returns
    -------
    None
    """
    # get shape&dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()
    input_format = x.get("format")

    # check others parameter
    avg_pool_check_rule(input_shape, input_dtype,
                        output_dtype, input_format, ksize, strides,
                        data_format, kernel_name)

    # set tensor attrs, during L1 fusion these attrs will assign by te_fusion
    addr_type = x.get("addr_type", 0)
    valid_shape = x.get("valid_shape", [])
    slice_offset = x.get("slice_offset", [])
    split_index = x.get("split_index", 0)
    l1_fusion_type = x.get("L1_fusion_type", -1)
    attr = {"addr_type": addr_type,
            "valid_shape": valid_shape,
            "slice_offset": slice_offset,
            "split_index": split_index,
            "L1_fusion_type": l1_fusion_type}
    is_l1fusion = l1_fusion_type in (0, 1)

    if data_format in ("NHWC",):
        ksizeH = ksize[1]
        ksizeW = ksize[2]
        hw = ksizeH * ksizeW
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
    else:
        ksizeH = ksize[2]
        ksizeW = ksize[3]
        hw = ksizeH * ksizeW
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]


    # compute
    # create tensor_in
    tensor_in = tvm.placeholder(input_shape, name="tensor_in",
                                dtype=input_dtype, attrs=attr)

    if filter is not None:
        filter_shape = filter.get("shape")
        filter_dtype = filter.get("dtype").lower()
        filter_c1 = filter_shape[0] / hw
        if filter_dtype in("float16", "float32"):
            filter_shape_5d = filter_c1, ksizeH, ksizeW, filter_shape[2], \
                              filter_shape[3]
        else:

            filter_shape_5d = filter_shape[0], ksizeH, ksizeW, 32, \
                              32
        filter_in = tvm.placeholder(filter_shape_5d, name="filter_in",
                                    dtype=filter_dtype, attrs=attr)
        bias_tensor = None
        if bias is not None and bias != {}:
            bias_shape = bias.get("shape")
            bias_tensor = tvm.placeholder(bias_shape,
                                          name='bias_tensor',
                                          dtype=output_dtype.lower())

        out_dtype = y.get("dtype")

        shape_x = input_shape
        input_h = shape_x[2]
        input_w = shape_x[3]
        dilations = (1, 1)
        dsl_flag = False

        if padding == "SAME":
            output_h = (input_h + stride[0] - 1) // stride[0]
            output_w = (input_w + stride[1] - 1) // stride[1]
            pad_row = (output_h - 1) * stride[0] + \
                      ((window[0] - 1) * dilations[0] + 1) - input_h
            pad_col = (output_w - 1) * stride[1] + \
                      ((window[1] - 1) * dilations[1] + 1) - input_w
            pad_top = pad_row // 2
            pad_bottom = pad_row - pad_top
            pad_left = pad_col // 2
            pad_right = pad_col - pad_left
            pad_top = get_corrected_pad(int(pad_top))
            pad_bottom = get_corrected_pad(int(pad_bottom))
            pad_left = get_corrected_pad(int(pad_left))
            pad_right = get_corrected_pad(int(pad_right))
            pad = (pad_top, pad_bottom, pad_left, pad_right)
        else:
            pad = (0, 0, 0, 0)
        res = te.lang.cce.te_compute.depthwise_conv2d_compute(
            tensor_in, filter_in, out_dtype.lower(), stride, pad, dilations, {
                "bias_tensor": bias_tensor, "dsl_flag": dsl_flag,
                "offset_x": offset_x}, None, kernel_name)


        tensor_list = [tensor_in, filter_in, res]
        if bias_tensor is not None:
            tensor_list = [tensor_in, filter_in, bias_tensor, res]
    else:
        res = avg_pool_compute1(tensor_in, y, ksize, strides, padding,
                                data_format, False, kernel_name)

        tensor_list = [tensor_in, res]
    # schedule
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    # build
    config = {"print_ir": False,
              "need_build": False,
              "name": kernel_name,
              "tensor_list": tensor_list,
              "l1_fusion_option": is_l1fusion}

    te.lang.cce.cce_build_code(sch, config)

