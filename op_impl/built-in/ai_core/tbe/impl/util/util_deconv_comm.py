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
deconv_comm
provide common function used by conv2d_backprop_input and deconvlution
"""
from __future__ import absolute_import

from te.tvm import api as tvm
from te.platform import get_soc_spec
from te.platform import cce_params
from te.utils.error_manager import error_manager_util as err_man
from te.lang.cce.te_compute.cube_util import shape_to_list
from te.utils import check_para


# the dim of shape in conv_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4
# the dim of strides in conv_backprop must be 2
STRIDES_SHAPE_DIM = 2
# the dim of pads in conv_backprop must be 4
PADDING_SHAPE_DIM = 4

# fmapH, fmapW must be in [2,4096]
FMAP_HW_MIN = 2
FMAP_HW_MAX = 4096

# DeDy H,W must be in [2,4096]
DEDY_HW_MIN = 2
DEDY_HW_MAX = 4096

# filterH, filterW must be in [1,255]
FILTER_HW_MIN = 1
FILTER_HW_MAX = 255

# conv1d situation support w not larger than 2^31-1
CONV1D_W_MAX = 2147483647

# stride must be in [1,63] and h*w not lagger than 256
STRIDE_HW_MIN = 1
STRIDE_HW_MAX = 63
STRIDE_SIZE_MAX = 256

# pad must be in [0,255]
PAD_MIN = 0
PAD_MAX = 255

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255

# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 1000000

# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
# same as (2**63-1)
DATA_SIZE_MAX = 9223372036854775807

# If pads is string , only support "SAME" or "VALID"
PADDING_SUPPORT = ('SAME', 'VALID')
# pads valid mode to be [0, 0, 0, 0]
PADDING_VAILD = [0, 0, 0, 0]


def get_filter_shape(ori_format_filters, ori_shape_filters):
    """
    Get filter shape of NCHW from original shape
    :param ori_format_filters:
    :param ori_shape_filters:
    :return: filter shape of NCHW
    """
    if ori_format_filters == "NCHW":
        shape_filters = ori_shape_filters
    elif ori_format_filters == "NHWC":
        shape_filters = (ori_shape_filters[0],
                         ori_shape_filters[3],
                         ori_shape_filters[1],
                         ori_shape_filters[2])
    elif ori_format_filters == "HWCN":
        shape_filters = (ori_shape_filters[3],
                         ori_shape_filters[2],
                         ori_shape_filters[0],
                         ori_shape_filters[1])
    else:
        args_dict = {
            "errCode": "E60004",
            "param_name": "filter",
            "expected_format_list": "[NCHW,NHWC,HWCN]",
            "format": ori_format_filters
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return shape_filters


def exchange_filter_nc_axis(ori_format_filters, ori_shape_filters):
    """
    Get filter shape of NCHW from original shape
    :param ori_format_filters:
    :param ori_shape_filters:
    :return: filter shape of exchange filter nc axis
    """
    if ori_format_filters == "NCHW":
        shape_filters = (ori_shape_filters[1],
                         ori_shape_filters[0],
                         ori_shape_filters[2],
                         ori_shape_filters[3])
    elif ori_format_filters == "NHWC":
        shape_filters = (ori_shape_filters[3],
                         ori_shape_filters[1],
                         ori_shape_filters[2],
                         ori_shape_filters[0])
    elif ori_format_filters == "HWCN":
        shape_filters = (ori_shape_filters[0],
                         ori_shape_filters[1],
                         ori_shape_filters[3],
                         ori_shape_filters[2])
    else:
        args_dict = {
            "errCode": "E60004",
            "param_name": "filter",
            "expected_format_list": "[NCHW,NHWC,HWCN]",
            "format": ori_format_filters
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return shape_filters


def align(x_1, x_2):
    """
    Get minimum y: y >= x_1 and y % x_2 == 0
    :param x_1:
    :param x_2:
    :return: minimum y: y >= x_1 and y % x_2 == 0
    """
    if x_2 == 0:
        args_dict = {
            "errCode": "E60114",
            "reason": "Division by zero",
            "value": "x_1 = {}, x_2 = {}".format(x_1, x_2)
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return (x_1 + x_2 - 1) // x_2 * x_2


# pylint: disable=too-many-locals,
def get_padlist(pads, shape_res, strides, shape_filters, dilations):
    """
    Get pad list of int
    :param pads: "SAME" or "VALID" or list of int
    :param shape_res: shape of dx
    :param strides:
    :param shape_filters:
    :param dilations:
    :return: pad list of int
    """
    fmap_h, fmap_w = shape_res[2], shape_res[3]
    _, _, filter_h, filter_w = shape_filters
    stride_h, stride_w = strides
    _, _, dilation_h, dilation_w = dilations

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    if pads == 'SAME':
        pad_h = \
            align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
        pad_h = tvm.max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = \
            align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
        pad_w = tvm.max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pads = [pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = PADDING_VAILD
    pads = shape_to_list(pads)
    return pads


def get_shape_out_backprop(ori_format_out_backprop, ori_shape_out_backprop):
    """
    Get out_backpro shape of NCHW from original shape
    :param ori_format_out_backprop:
    :param ori_shape_out_backprop:
    :return: out_backpro shape of NCHW
    """
    if ori_format_out_backprop == "NCHW":
        shape_out_backprop = ori_shape_out_backprop
    elif ori_format_out_backprop == "NHWC":
        shape_out_backprop = (ori_shape_out_backprop[0],
                              ori_shape_out_backprop[3],
                              ori_shape_out_backprop[1],
                              ori_shape_out_backprop[2])
    else:
        args_dict = {
            "errCode": "E60004",
            "param_name": "out_backprop",
            "expected_format_list": "[NCHW,NHWC]",
            "format": ori_format_out_backprop
        }
        raise RuntimeError(args_dict,
                           err_man.get_error_message(args_dict))
    return shape_out_backprop


def get_shape_dilation(ori_out_backprop_format, dilations):
    """
    Get result shape of NCHW from original shape
    :param ori_out_backprop_format:
    :param dilations:
    :return: result shape of NCHW
    """
    if ori_out_backprop_format == "NCHW":
        shape_dilations = dilations
    elif ori_out_backprop_format == "NHWC":
        shape_dilations = (dilations[0],
                           dilations[3],
                           dilations[1],
                           dilations[2])
    else:
        args_dict = {
            "errCode": "E60004",
            "param_name": "out_backprop",
            "expected_format_list": "[NCHW,NHWC]",
            "format": ori_out_backprop_format
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return shape_dilations


def get_shape_res(ori_format_res, ori_shape_res):
    """
    Get result shape of NCHW from original shape
    :param ori_format_res:
    :param ori_shape_res:
    :return: result shape of NCHW
    """
    if ori_format_res == "NCHW":
        shape_res = ori_shape_res
    elif ori_format_res == "NHWC":
        shape_res = (
            ori_shape_res[0], ori_shape_res[3],
            ori_shape_res[1], ori_shape_res[2])
    else:
        args_dict = {
            "errCode": "E60004",
            "param_name": "y",
            "expected_format_list": "[NCHW,NHWC]",
            "format": ori_format_res
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return shape_res


# pylint: disable=too-many-arguments, too-many-locals, too-many-statements
@check_para.check_input_type(
    (list, tuple), (list, tuple), (list, tuple),
    (list, tuple), (str, list, tuple), (list, tuple),
    str,
    str,
    str,
    str,
    (check_para.NONE_TYPE, dict),
    (str, check_para.NONE_TYPE))
def check_conv2dbp_input_params(shape_filter, shape_out_backprop, input_sizes,
                                strides, pads, dilations,
                                filter_dtype,
                                out_backprop_dtype,
                                res_dtype,
                                kernel_name,
                                fusion_para=None,
                                dynamic_mode=None):
    """
    The params check function of conv2d backprop input and deconvolution

    Parameters:
    ----------
    shape_filter : The shape of filter.
                   4-D with shape [batch, channels, height, weight].

    shape_out_backprop : The shape of gradients.
                         4-D with shape [batch, channels, height, weight].

    input_sizes : The shape of feature map.
                  4-D with shape [batch, channels, height, weight].

    strides : A list of ints. The stride of the sliding window.

    pads : "SAME"or"VALID" indicating the type of pads algorithm to use,
           or list.

    dilations : An optional list of ints. Default value is [1, 1, 1, 1].

    filter_dtype : The dtype of filter data. Default value is float16.

    out_backprop_dtype : The dtype of gradients data. Default value is float16.

    res_dtype : The dtype of result(De/Dx) data. Default value is float16.

    kernel_name : Cce kernel name. Default value is "conv2d_backprop_input_cce"

    dynamic_mode : dynamic type, "dynamic_hw" or "dynamic_batch"

    Returns : All transformed params.
    ----------
    """

    def _check_attr_range(attr_name, attr_value, attr_min=None, attr_max=None):
        if not attr_min and not attr_max:
            return
        if not attr_min:
            if attr_value > attr_max:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "{} exceed max_value."
                              " max_value={}.".format(attr_name, attr_max),
                    "value": "attr_value = {}".format(attr_value)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))
        elif not attr_max:
            if attr_value < attr_min:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "{} less than min_value. "
                              "min_value={}.".format(attr_name, attr_min),
                    "value": "attr_value = {}".format(attr_value)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))
        elif attr_value < attr_min or attr_value > attr_max:
            args_dict = {
                "errCode": "E60011",
                "range": "[{},{}]".format(attr_min, attr_max),
                "attr_name": attr_name,
                "value": attr_value
            }
            raise RuntimeError(args_dict,
                               err_man.get_error_message(args_dict))

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype is None:
            bit_ratio = BIT_RATIO_DICT.get("float16")
        else:
            bit_ratio = BIT_RATIO_DICT.get(dtype)
        if attr_value * bit_ratio > DATA_SIZE_MAX:
            args_dict = {
                "errCode": "E60020",
                "attr_name": attr_name
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    def _check_shape_relation():
        if fmap_channel != filter_channel:
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape",
                "param1_name": "Fmap'C",
                "param1_value": fmap_channel,
                "param2_name": "Filter'C",
                "param2_value": filter_channel
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if dedy_channel != filter_batch:
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape",
                "param1_name": "Dedy's C",
                "param1_value": dedy_channel,
                "param2_name": "Filter'N",
                "param2_value": filter_batch
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if fmap_batch != dedy_batch:
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape",
                "param1_name": "Fmap's N",
                "param1_value": fmap_batch,
                "param2_name": "Dedy'N",
                "param2_value": dedy_batch
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if filter_h_dilation > fmap_h_padding and h_match_rule:
            args_dict = {
                "errCode": "E60014",
                "h_of_x": fmap_h_padding,
                "h_of_filter": filter_h_dilation
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if filter_w_dilation > fmap_w_padding:
            args_dict = {
                "errCode": "E60014",
                "h_of_x": fmap_w_padding,
                "h_of_filter": filter_w_dilation
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

        if pad_up >= filter_h_dilation or pad_down >= filter_h_dilation:
            args_dict = {
                "errCode": "E60016",
                "h_of_filter": filter_h_dilation,
                "h_of_pad": pad_up if pad_up > pad_down else pad_down
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if pad_left >= filter_w_dilation or pad_right >= filter_w_dilation:
            args_dict = {
                "errCode": "E60017",
                "w_of_filter": filter_w_dilation,
                "w_of_pad": pad_left if pad_left > pad_right else pad_right
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    def _check_l1_size_limit():
        def _l1fusion_size_limit(l1_size):
            l1fusion_l1_size = 0
            if pads != [0, 0, 0, 0] or [filter_h, filter_w] != [1, 1]:
                if stride_h > 1 or stride_w > 1:
                    l1fusion_l1_size = l1_size
            return l1fusion_l1_size

        c0_size = cce_params.C0_SIZE
        c0_size_k = cce_params.CUBE_MKN[filter_dtype]['mac'][1]
        w_value = dedy_w * stride_w

        if fmap_w > c0_size:
            h_value_max = filter_h_dilation + 1
        elif c0_size % fmap_w == 0:
            h_value_max = filter_h_dilation + c0_size // fmap_w - 1
        else:
            h_value_max = filter_h_dilation + c0_size // fmap_w + 1

        a_l1_size = h_value_max * w_value *\
                    c0_size_k * BIT_RATIO_DICT.get(out_backprop_dtype)
        if _is_conv1d_situation():
            load3d_stride = 1
            a_l1_m_length = (c0_size - 1) * load3d_stride + filter_w_dilation
            a_l1_size = a_l1_m_length *\
                        c0_size_k * BIT_RATIO_DICT.get(out_backprop_dtype)
        b_l1_size = filter_h_dilation * filter_w_dilation *\
                    c0_size * c0_size_k * BIT_RATIO_DICT.get(filter_dtype)
        if fusion_para.get("l1_fusion_type") != -1:
            a_l1_size = _l1fusion_size_limit(a_l1_size)
        l1_size = get_soc_spec("L1_SIZE")
        if (a_l1_size + b_l1_size) > l1_size:
            args_dict = {
                "errCode": "E60022",
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    def _check_pads():
        if isinstance(pads, (tuple, list)) \
                and len(pads) != CONV_BACKPROP_SHAPE_DIM:
            args_dict = {
                "errCode": "E60107",
                "param_name": "pads"
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

        if isinstance(pads, str) and pads not in PADDING_SUPPORT:
            args_dict = {
                "errCode": "E60021",
                "expected_pad_mode": PADDING_SUPPORT,
                "actual_pad_mode": pads
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    def _is_conv1d_situation():
        if fmap_h_padding == 1 and filter_h_dilation == 1 and stride_h == 1:
            return True
        return False

    def _is_load3d_special():
        # limitation by chip:
        # Ascend910
        # load3d not support when only fmap w after padding equals to filter w
        if get_soc_spec("SOC_VERSION") == 'Ascend910' \
            and fmap_h_padding != filter_h \
            and fmap_w_padding == filter_w:
            return False
        # limitation by chip:
        # if kernel h,w in [1,11]
        # and fmap h/w after padding equals to filter h/w
        # load3d support h,w is 1
        if (1 <= filter_h <= 11) and (1 <= filter_w <= 11) \
            and (fmap_h_padding == filter_h or fmap_w_padding == filter_w):
            return True
        return False

    def fusion_para_check(fusion_para, shape_out_backprop):
        c0_k = cce_params.CUBE_MKN[filter_dtype]["mac"][1]

        def handle_valid_shape():
            if not slice_offset:
                reason = "valid shape exists, slice shape cannot be []"
                args_dict = {
                    "errCode": "E60108",
                    "reason": reason
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))
            if not output_offset:
                reason = "valid shape exists,output offset cannot be []"
                args_dict = {
                    "errCode": "E60108",
                    "reason": reason
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))

            if slice_offset[2] >= shape_out_backprop[2] \
                    or slice_offset[2] < 0:
                reason = "invalid slice_offset"
                args_dict = {
                    "errCode": "E60114",
                    "reason": reason,
                    "value": str(slice_offset)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))
            if output_offset[2] < 0:
                reason = "invalid outputoffset"
                args_dict = {
                    "errCode": "E60114",
                    "reason": reason,
                    "value": str(output_offset)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))
            if slice_offset[0] != 0 or slice_offset[1] != 0 \
                    or slice_offset[3] != 0 or slice_offset[4] != 0:
                reason = "invalid slice_offset"
                args_dict = {
                    "errCode": "E60114",
                    "reason": reason,
                    "value": str(slice_offset)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))
            if output_offset[0] != 0 or output_offset[1] != 0 \
                    or output_offset[3] != 0 or output_offset[4] != 0:
                reason = "invalid output_offset"
                args_dict = {
                    "errCode": "E60114",
                    "reason": reason,
                    "value": str(output_offset)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))
            if valid_shape[2] + slice_offset[2] > shape_out_backprop[2] \
                    or valid_shape[2] + slice_offset[2] < 0:
                reason = "invalid valid_shape"
                args_dict = {
                    "errCode": "E60114",
                    "reason": reason,
                    "value": str(valid_shape)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))
            if valid_shape[0] != shape_out_backprop[0] \
                    or valid_shape[3] != shape_out_backprop[3] \
                    or valid_shape[1]*valid_shape[4] \
                    != (shape_out_backprop[1] + c0_k - 1) // c0_k * c0_k:
                reason = "invalid valid_shape"
                args_dict = {
                    "errCode": "E60114",
                    "reason": reason,
                    "value": str(valid_shape)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))

        valid_shape = fusion_para.get("valid_shape")
        slice_offset = fusion_para.get("slice_offset")
        output_offset = fusion_para.get("output_offset")
        l1_fusion_type = fusion_para.get("l1_fusion_type")
        input_memory_type = fusion_para.get("input_memory_type")
        output_memory_type = fusion_para.get("output_memory_type")

        if l1_fusion_type == -1:
            if input_memory_type == 1:
                args_dict = {
                    "errCode": "E60109",
                    "input_memory_type": str(input_memory_type)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))
            if output_memory_type == 1:
                args_dict = {
                    "errCode": "E60110",
                    "output_memory_type": str(output_memory_type)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))

        if valid_shape:
            handle_valid_shape()

    def modify_fusion_para(fusion_para, input_h,
                           input_hw_mini, output_hw_mini):
        h_match_rule = True
        if fusion_para is None:
            fusion_para = {"input_memory_type": 0,
                           "output_memory_type": 0,
                           "valid_shape": (),
                           "slice_offset": (),
                           "output_offset": (),
                           "l1_fusion_type": -1,
                           "fmap_l1_addr_flag": False,
                           "fmap_l1_valid_size": 0}
        if fusion_para.get("valid_shape"):
            input_h = fusion_para.get("valid_shape")[2]
            h_match_rule = False
            input_hw_mini = 1
            output_hw_mini = 1
        return fusion_para, input_h, h_match_rule,\
               input_hw_mini, output_hw_mini

    # First : Base check, Mainly required by interface appearance
    # util check
    check_para.check_kernel_name(kernel_name)
    check_para.check_shape_rule(shape_filter, CONV_BACKPROP_SHAPE_DIM,
                                CONV_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)
    if dynamic_mode is None:
        check_para.check_shape_rule(shape_out_backprop,
                            CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                            DEFAULT_MAX_SHAPE_NUM)
        check_para.check_shape_rule(input_sizes,
                            CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                            DEFAULT_MAX_SHAPE_NUM)
    check_para.check_shape_rule(dilations, CONV_BACKPROP_SHAPE_DIM,
                                CONV_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)
    check_para.check_shape_rule(strides, STRIDES_SHAPE_DIM, STRIDES_SHAPE_DIM,
                                DEFAULT_MAX_SHAPE_NUM)
    _check_pads()

    # dilations check
    dilation_n, dilation_c, dilation_h, dilation_w = dilations
    if dilation_n != 1 or dilation_c != 1:
        args_dict = {
            "errCode": "E60023",
            "dilation_n": dilation_n,
            "dilation_c": dilation_c
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    # dtype check
    valid_filter_dtype = ("float16", "int8")
    valid_dedy_dtype = ("float16", "int8")
    valid_res_dtype = ("float16", "int32")

    filter_dtype = filter_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    check_para.check_dtype_rule(filter_dtype, valid_filter_dtype)
    check_para.check_dtype_rule(out_backprop_dtype, valid_dedy_dtype)
    check_para.check_dtype_rule(res_dtype, valid_res_dtype)

    # Second : Furture Check, Mainly required by SRS
    # the relation limits between shape
    shape_filter = list(shape_filter)
    shape_out_backprop = list(shape_out_backprop)
    input_sizes = list(input_sizes)
    strides = list(strides)
    fmap_batch, fmap_channel, fmap_h, fmap_w = input_sizes
    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_out_backprop
    filter_batch, filter_channel, filter_h, filter_w = shape_filter
    stride_h, stride_w = strides

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    pads = get_padlist(pads, input_sizes, strides, shape_filter, dilations)
    pad_up, pad_down, pad_left, pad_right = pads

    fmap_h_padding = fmap_h + pad_up + pad_down
    fmap_w_padding = fmap_w + pad_left + pad_right

    # special cases
    dedy_hw_min, fmap_hw_min = DEDY_HW_MIN, FMAP_HW_MIN
    dedy_hw_max, fmap_hw_max = DEDY_HW_MAX, FMAP_HW_MAX

    fusion_para, dedy_h, h_match_rule, dedy_hw_min, fmap_hw_min \
        = modify_fusion_para(fusion_para, dedy_h, dedy_hw_min, fmap_hw_min)

    # exchange h and w will not change date in memmory
    if fmap_w_padding == 1 and filter_w == 1 and dedy_w == 1:
        input_sizes = (fmap_batch, fmap_channel, fmap_w, fmap_h)
        shape_out_backprop = (dedy_batch, dedy_channel, dedy_w, dedy_h)
        shape_filter = (filter_batch, filter_channel, filter_w, filter_h)
        strides = stride_w, stride_h
        dilations = dilation_n, dilation_c, dilation_w, dilation_h
        fmap_h_padding, fmap_w_padding = fmap_w_padding, fmap_h_padding
        dedy_h, dedy_w = dedy_w, dedy_h
        fmap_h, fmap_w = fmap_w, fmap_h
        filter_h, filter_w = filter_w, filter_h
        filter_h_dilation, filter_w_dilation = filter_w_dilation,\
                                               filter_h_dilation

    if _is_load3d_special():
        dedy_hw_min = 1
        fmap_hw_min = 1

    # if conv1d situation, make sure w is in [1,CONV1D_W_MAX]
    if _is_conv1d_situation():
        dedy_hw_min = 1
        fmap_hw_min = 1
        dedy_hw_max = CONV1D_W_MAX
        fmap_hw_max = CONV1D_W_MAX

    if dynamic_mode is None:
        _check_shape_relation()

        # Dedy value limit
        _check_attr_range("Dedy's H after expands", dedy_h * stride_h,
                        dedy_hw_min, dedy_hw_max)
        if filter_h == 1 and filter_w == 1:
            _check_attr_range("Dedy's W after expands",
                            dedy_w * stride_w * stride_h,
                            dedy_hw_min, dedy_hw_max)
        else:
            _check_attr_range("Dedy's W after expands", dedy_w * stride_w,
                            dedy_hw_min, dedy_hw_max)

    # filter value limit
    _check_attr_range("filter's H", filter_h, FILTER_HW_MIN, FILTER_HW_MAX)
    _check_attr_range("filter's W", filter_w, FILTER_HW_MIN, FILTER_HW_MAX)

    if dynamic_mode is None:
        # Fmap value limit
        _check_attr_range("Fmap's H", fmap_h, fmap_hw_min, fmap_hw_max)
        _check_attr_range("Fmap's W", fmap_w, fmap_hw_min, fmap_hw_max)

    # stride value limit
    _check_attr_range("stride's H", stride_h, STRIDE_HW_MIN, STRIDE_HW_MAX)
    _check_attr_range("stride's W", stride_w, STRIDE_HW_MIN, STRIDE_HW_MAX)
    _check_attr_range("stride size", stride_h * stride_w,
                      attr_max=STRIDE_SIZE_MAX)

    # dilation value limit
    _check_attr_range("dilations's H", dilation_h, DILATION_MIN, DILATION_MAX)
    _check_attr_range("dilations's W", dilation_w, DILATION_MIN, DILATION_MAX)

    # Third : value check, Mainly required by the convolution rule
    if dynamic_mode != "dynamic_hw":
        if ((fmap_h - filter_h_dilation + pad_up + pad_down) // stride_h + 1) \
                != dedy_h and h_match_rule:
            args_dict = {
                "errCode": "E60024",
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if ((fmap_w - filter_w_dilation + pad_left + pad_right) // stride_w + 1) \
                != dedy_w:
            args_dict = {
                "errCode": "E60025",
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    # Forth : L1 limitation, Mainly required by chip
    if dynamic_mode != "dynamic_hw":
        _check_l1_size_limit()

    # Fifth : check shape size, 64 bits limitation
    c0_size = cce_params.C0_SIZE
    fmap_size = fmap_batch * align(fmap_channel, c0_size) * fmap_h * fmap_w
    dedy_size = dedy_batch * align(dedy_channel, c0_size) * dedy_h * dedy_w
    filter_size = \
        align(filter_batch, c0_size) * align(filter_channel, c0_size) * \
        filter_h * filter_w
    if not dynamic_mode:
        _check_64bits_limitation("fmap_size", fmap_size, dtype=res_dtype)
        _check_64bits_limitation("dedy_size", dedy_size, dtype=out_backprop_dtype)
        _check_64bits_limitation("filter_size", filter_size, dtype=filter_dtype)

    result = (
        shape_filter, shape_out_backprop, input_sizes, strides, pads,
        dilations, filter_dtype, out_backprop_dtype, res_dtype, kernel_name)

    fusion_para_check(fusion_para, shape_out_backprop)

    return result
