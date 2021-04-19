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
conv3d_backprop_filter_d
"""
from __future__ import absolute_import
from functools import reduce as func_reduce
import te.lang.cce
from te import tvm
from te.platform import cce_params
from te.platform.cce_conf import get_soc_spec
from te.utils.error_manager import error_manager_util as err_mana
from topi import generic
from topi.cce import util

# the dim of shape in CONV3D_BACKPROP must be 5
CONV3D_BACKPROP_SHAPE_DIM = 5
# the dim of strides in CONV3D_BACKPROP must be 3
STRIDES_SHAPE_DIM = 3
# the dim of pads in CONV3D_BACKPROP must be 6
PADDING_SHAPE_DIM = 6
# the min x or y dim for cube mul
C0 = 16
# fmapH, fmapW must be in [1,4096]
FMAP_HW_MAX = 4096
FMAP_HW_MIN = 1

# DeDy H,W must be in [2,4096]
DEDY_HW_MAX = 4096
DEDY_HW_MIN = 2

# filterH, filterW must be in [1,255]
FILTER_HW_MAX = 255
FILTER_HW_MIN = 1

# stride must be in [1,63]
STRIDE_HW_MAX = 63
STRIDE_HW_MIN = 1

# pad must be in [0,255]
PAD_MAX = 255
PAD_MIN = 0

# dilation must be in [1,255]
DILATION_MIN = 1
DILATION_MAX = 255

# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 1000000

# the max size is 2**63-1
DATA_SIZE_MAX = 9223372036854775807

# the bytes length of several dtype
BIT_RATIO_DICT = {
    "int32": 4,
    "float32": 4,
    "float16": 2,
    "uint8": 1,
    "int8": 1,
    "uint4": 0.5,
    "int4": 0.5
}

# pads valid mode to be [0, 0, 0, 0]
PADDING_VAILD = [0, 0, 0, 0, 0, 0]
# If pads is string , only support "SAME" or "VALID"
PADDING_SUPPORT = ('SAME', 'VALID')


def align(input_x, input_y):
    if input_y == 0:
        args_dict = {
            'errCode': 'E62502',
            'first_operand': str(input_x),
            'second_operand': str(input_y)
        }
        raise RuntimeError(args_dict, err_mana.get_error_message(args_dict))
    return (input_x + input_y - 1) // input_y * input_y


@util.check_input_type(dict, dict, dict, (tuple, list), (tuple, list),
                       (str, tuple, list), (tuple, list), int, str, str)
def conv3d_backprop_filter_d(x_dict,
                             out_backprop,
                             y_dict,
                             filter_size,
                             strides,
                             pads,
                             dilations=(1, 1, 1, 1, 1),
                             groups=1,
                             data_format='NDHWC',
                             kernel_name="conv3d_backprop_filter"):
    """
    algorithm: conv3d_backprop_filter

    Parameters
    ----------
    x_dict: dict with keys(shape and dtype)
       input feature map tensor

    out_backprop: dict with keys(shape and dtype)
                  input weight tensor

    y_dict: dict with keys(shape and dtype)
       output tensor, dtype must be assigned

    filter_size: The shape of filter.
                  5-D with shape [batch, depth, channels, height, weight].

    strides: tuple/list of 3 integers
             filter move stride

    pads: string of "SAME" or "VAILD"
             [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 5 integers
               filter expand size of dilated conv3d_backprop_filter

    data_format: str
            An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
            Specify the data format of the input and output data.

    kernel_name: str
                 kernel name, default value is "conv3d_backprop_filter"

    Returns
    -------
    None
    """
    def _check_inputs_rules():
        if (not isinstance(ori_shape_out_backprop, (tuple, list))) \
                or len(ori_shape_out_backprop) != 5:
            args_dict = {
                'errCode': 'E62002',
                'param_name': 'out_backprop_shape',
                'expected_type': '[{}, {}]'.format('tuple', 'list'),
                'expected_length': '5',
                'type': str(type(ori_shape_out_backprop)),
                'length': str(len(ori_shape_out_backprop))
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

        if (not isinstance(ori_shape_x, (tuple, list))) or \
                len(ori_shape_x) != 5:
            args_dict = {
                'errCode': 'E62002',
                'param_name': 'input_shape',
                'expected_type': '[{}, {}]'.format('tuple', 'list'),
                'expected_length': '5',
                'type': str(type(ori_shape_x)),
                'length': str(len(ori_shape_x))
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

        if (not isinstance(ori_shape_res, (tuple, list))) \
                or len(ori_shape_res) != 5:
            args_dict = {
                'errCode': 'E62002',
                'param_name': 'res_shape',
                'expected_type': '[{}, {}]'.format('tuple', 'list'),
                'expected_length': '5',
                'type': str(type(ori_shape_res)),
                'length': str(len(ori_shape_res))
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

        if len(strides) != 3:
            args_dict = {
                'errCode': 'E60006',
                'param_name': 'strides',
                'expected_length': '3',
                'length': str(len(strides))
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

        if len(filter_size) != 5:
            args_dict = {
                'errCode': 'E60006',
                'param_name': 'filter_size',
                'expected_length': '5',
                'length': str(len(filter_size))
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

        if len(dilations) != 5:
            args_dict = {
                'errCode': 'E60006',
                'param_name': 'dilations',
                'expected_length': '5',
                'length': str(len(dilations))
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

        if isinstance(pads, str) and pads not in PADDING_SUPPORT:
            args_dict = {
                'errCode': 'E60021',
                'expected_pad_mode': '[{}, {}]'.format('SAME', 'VALID'),
                'actual_pad_mode': str(pads)
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

        if isinstance(pads, (tuple, list)) and len(pads) != 6:
            args_dict = {'errCode': 'E62501', 'param_name': 'pads'}
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

    def _normalize_shape_ndchw(ori_shape, ori_format, format_list,
                               param_name='input_param'):
        """
        normalizing the shape to NDCHW
        """
        if ori_format not in format_list:
            args_dict = {
                'errCode': 'E60008',
                'param_name': param_name,
                'expected_format_list': ','.join(format_list),
                'format': ori_format
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

        n_index = ori_format.find('N')
        d_index = ori_format.find('D')
        c_index = ori_format.find('C')
        h_index = ori_format.find('H')
        w_index = ori_format.find('W')

        new_shape = [
            ori_shape[n_index], ori_shape[d_index],
            ori_shape[c_index], ori_shape[h_index],
            ori_shape[w_index]
        ]

        return new_shape

    ori_shape_x = x_dict.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = y_dict.get("ori_shape")

    x_dtype = x_dict.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_dict.get("dtype")

    ori_format_x = x_dict.get("ori_format")
    ori_format_out_backprop = out_backprop.get("ori_format")
    ori_format_res = y_dict.get("ori_format")

    if len(strides) == 5:
        d_index = data_format.find('D')
        h_index = data_format.find('H')
        w_index = data_format.find('W')
        strides = [strides[d_index], strides[h_index], strides[w_index]]

    _check_inputs_rules()

    input_format_list = ("NDHWC", "NCDHW")
    shape_x = _normalize_shape_ndchw(ori_shape_x,
                                     ori_format_x,
                                     input_format_list,
                                     'x')
    shape_out_backprop = _normalize_shape_ndchw(
                            ori_shape_out_backprop,
                            ori_format_out_backprop,
                            input_format_list,
                            'out_backprop')
    dilations = _normalize_shape_ndchw(dilations,
                                       ori_format_out_backprop,
                                       input_format_list,
                                       'dilations')

    res_format_list = ("NDHWC", "NCDHW", "DHWCN")
    shape_res = _normalize_shape_ndchw(ori_shape_res,
                                       ori_format_res,
                                       res_format_list,
                                       'y')

    conv3d_backprop_filter_cce(shape_x, shape_out_backprop, shape_res,
                                   strides, pads, dilations, x_dtype,
                                   out_backprop_dtype, res_dtype, kernel_name)


@util.check_input_type((list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (str, list, tuple), (list, tuple), str,
                       str, str, str)
def check_conv3dbp_filter_params(shape_x, shape_out_backprop, filter_sizes,
                                 strides, pads, dilations, x_dtype,
                                 out_backprop_dtype, res_dtype, kernel_name):
    """
    The params check function of conv3d_backprop_filter

    Parameters:
    ----------
    shape_x : The shape of feature map,
              which is 5-D [batch, depth, channels, height, weight].

    shape_out_backprop : The shape of gradients,
                         which is 5-D [batch, depth,channels, height, weight].

    filter_sizes : The shape of filter.
                   which is 5-D [batch, depth, channels, height, weight].

    strides : The stride of the sliding window. A list of ints.

    pads : "SAME"or"VALID",
           indicating the type of pads algorithm to use, or list.

    dilations : An optional list of ints. Default value is [1, 1, 1, 1].

    x_dtype : Fmeature map  data dtype. Default value is float16.

    out_backprop_dtype : Gradients data dtype. Default value is float16.

    res_dtype : Result(De/Dw) data dtype. Default value is float32.

    kernel_name : Kernel name of cce.
                  Default value is "conv3d_backprop_filter_cce"

    Returns : All transformed params.
    ----------
    """
    def _check_attr_range_dw(name, value, attr_min=None, attr_max=None):
        if not attr_min and not attr_max:
            return
        if not attr_min:
            if value > attr_max:
                args_dict = {
                    'errCode': 'E60011',
                    'range': '(,{}]'.format(attr_max),
                    'attr_name': name,
                    'value': str(value)
                }
                raise RuntimeError(args_dict,
                                   err_mana.get_error_message(args_dict))
        elif not attr_max:
            if value < attr_min:
                args_dict = {
                    'errCode': 'E60011',
                    'range': '[{},)'.format(attr_min),
                    'attr_name': name,
                    'value': str(value)
                }
                raise RuntimeError(args_dict,
                                   err_mana.get_error_message(args_dict))
        elif value > attr_max or value < attr_min:
            args_dict = {
                'errCode': 'E60011',
                'range': '[{},{}]'.format(attr_min, attr_max),
                'attr_name': name,
                'value': str(value)
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype:
            bit_ratio = BIT_RATIO_DICT.get(dtype)
        else:
            bit_ratio = BIT_RATIO_DICT.get("float16")
        if attr_value * bit_ratio > DATA_SIZE_MAX:
            args_dict = {'errCode': 'E60020', 'attr_name': attr_name}
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

    # First : Base check, Mainly required by interface appearance
    # ===========================================================
    # util check
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x, CONV3D_BACKPROP_SHAPE_DIM,
                          CONV3D_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(shape_out_backprop, CONV3D_BACKPROP_SHAPE_DIM,
                          CONV3D_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(filter_sizes, CONV3D_BACKPROP_SHAPE_DIM,
                          CONV3D_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(strides, STRIDES_SHAPE_DIM, STRIDES_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)

    def _check_attr_pads():
        # pads check
        if isinstance(pads, (tuple, list)) and \
                len(pads) != PADDING_SHAPE_DIM:
            args_dict = {'errCode': 'E62501', 'param_name': 'pads'}
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

        if isinstance(pads, str) and pads not in PADDING_SUPPORT:
            args_dict = {
                'errCode': 'E60021',
                'expected_pad_mode': '[{}]'.format(PADDING_SUPPORT),
                'actual_pad_mode': str(pads)
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

    _check_attr_pads()

    # dilations check
    util.check_shape_rule(dilations, CONV3D_BACKPROP_SHAPE_DIM,
                          CONV3D_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)
    dilation_n, dilation_d, dilation_c, dilation_h, dilation_w = dilations
    _check_attr_range_dw("dilations's H", dilation_h, DILATION_MIN,
                         DILATION_MAX)
    _check_attr_range_dw("dilations's W", dilation_w, DILATION_MIN,
                         DILATION_MAX)

    if dilation_n != 1 or dilation_c != 1:
        args_dict = {
            'errCode': 'E60023',
            'dilation_n': str(dilation_n),
            'dilation_c': str(dilation_c)
        }
        raise RuntimeError(args_dict, err_mana.get_error_message(args_dict))

    # detype check
    x_dtype = x_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    util.check_dtype_rule(x_dtype, ['float16'])
    util.check_dtype_rule(out_backprop_dtype, ['float16'])
    util.check_dtype_rule(res_dtype, ['float32', 'float16'])

    # Second : Furture Check, Mainly required by SRS
    # ===========================================================
    # the relation limits between shape
    shape_x = list(shape_x)
    shape_out_backprop = list(shape_out_backprop)
    filter_sizes = list(filter_sizes)
    strides = list(strides)
    fmap_batch, fmap_d, fmap_channel, fmap_h, fmap_w = shape_x
    dedy_batch, dedy_d, dedy_channel, dedy_h, dedy_w = shape_out_backprop
    filter_batch, filter_d, filter_channel, filter_h, filter_w = filter_sizes
    stride_d, stride_h, stride_w = strides

    filter_d_dilation = (filter_d - 1) * dilation_d + 1
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    # pads compute
    if pads == 'SAME':
        pad_d = \
            align(fmap_d, stride_d) - stride_d + filter_d_dilation - fmap_d
        pad_d = max(pad_d, 0)
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        pad_w = \
            align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
        pad_w = max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_h = \
            align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pads = [pad_front, pad_back, pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = PADDING_VAILD
    pads = list(pads)
    pad_front, pad_back, pad_up, pad_down, pad_left, pad_right = pads
    if pad_front >= filter_d_dilation or pad_back >= filter_d_dilation:
        args_dict = {
            'errCode': 'E60013',
            'depth_of_pad': '{}, {}'.format(pad_front, pad_back),
            'depth_of_filter': '{}'.format(filter_d_dilation)
        }
        raise RuntimeError(args_dict, err_mana.get_error_message(args_dict))
    if pad_up >= filter_h_dilation or pad_down >= filter_h_dilation:
        args_dict = {
            'errCode': 'E60016',
            'h_of_filter': '{}'.format(filter_h_dilation),
            'h_of_pad': '{}, {}'.format(pad_up, pad_down)
        }
        raise RuntimeError(args_dict, err_mana.get_error_message(args_dict))
    if pad_left >= filter_w_dilation or pad_right >= filter_w_dilation:
        args_dict = {
            'errCode': 'E60017',
            'w_of_filter': '{}'.format(filter_w_dilation),
            'w_of_pad': '{}, {}'.format(pad_left, pad_right)
        }
        raise RuntimeError(args_dict, err_mana.get_error_message(args_dict))

    fmap_w_padding = fmap_w + pad_left + pad_right
    fmap_h_padding = fmap_h + pad_up + pad_down

    # special cases
    fmap_hw_min, dey_hw_min = FMAP_HW_MIN, DEDY_HW_MIN
    # limitation by chip:
    # if kernel h,w in [1,11] and fmap h/w after padding equals to filter h/w
    # load3d support h,w is 1
    if (1 <= filter_w <= 11) and (1 <= filter_h <= 11) and (1 <= filter_d <= 11)\
            and (fmap_w_padding == filter_w or fmap_h_padding == filter_h):
        fmap_hw_min = 1
        dey_hw_min = 1

    # Dedy value limit
    _check_attr_range_dw("Dedy's H", dedy_h, dey_hw_min, DEDY_HW_MAX)
    _check_attr_range_dw("Dedy's W", dedy_w, dey_hw_min, DEDY_HW_MAX)

    # filter value limit
    _check_attr_range_dw("filter's H", filter_h, FILTER_HW_MIN, FILTER_HW_MAX)
    _check_attr_range_dw("filter's W", filter_w, FILTER_HW_MIN, FILTER_HW_MAX)

    # Fmap value limit
    _check_attr_range_dw("Fmap's H", fmap_h, fmap_hw_min, FMAP_HW_MAX)
    _check_attr_range_dw("Fmap's W", fmap_w, fmap_hw_min, FMAP_HW_MAX)

    # stride value limit
    _check_attr_range_dw("stride's H", stride_h, STRIDE_HW_MIN, STRIDE_HW_MAX)
    _check_attr_range_dw("stride's W", stride_w, STRIDE_HW_MIN, STRIDE_HW_MAX)

    def _check_axis_hw():
        if fmap_batch != dedy_batch:
            args_dict = {
                'errCode': 'E62503',
                'backprop_N': str(dedy_batch),
                'forward_shape': str(fmap_batch)
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))
        if dedy_channel != filter_batch:
            args_dict = {
                'errCode': 'E62504',
                'backprop_C': str(dedy_channel),
                'forward_shape': str(filter_batch)
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))
        if fmap_channel != filter_channel:
            args_dict = {
                'errCode': 'E60010',
                'channel_of_x': str(fmap_channel),
                'channel_of_filter': str(filter_channel)
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))
        if filter_w_dilation > fmap_w_padding:
            args_dict = {
                'errCode': 'E60015',
                'w_of_x': str(fmap_w_padding),
                'w_of_filter': str(filter_w_dilation)
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))
        if filter_h_dilation > fmap_h_padding:
            args_dict = {
                'errCode': 'E60014',
                'h_of_x': str(fmap_h_padding),
                'h_of_filter': str(filter_h_dilation)
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

        # Third : value check, Mainly required by the convolution rule
        if ((fmap_w - filter_w_dilation + pad_left + pad_right) // stride_w +
                1) != dedy_w:
            args_dict = {'errCode': 'E60025'}
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))
        if ((fmap_h - filter_h_dilation + pad_up + pad_down) // stride_h +
                1) != dedy_h:
            args_dict = {'errCode': 'E60024'}
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

    _check_axis_hw()

    def _min_l1_byte():
        # Forth : L1 limitation, Mainly required by chip
        al1_min_byte = C0 * C0 * 2

        if dedy_w % C0 == 0:
            bl1_min_byte = filter_h_dilation * fmap_w * C0 * 2
        else:
            bl1_min_byte = (filter_h_dilation + stride_h) * fmap_w * C0 * 2

        l1_size = get_soc_spec("L1_SIZE")  # L1 size
        if (al1_min_byte + bl1_min_byte) > l1_size:
            args_dict = {'errCode': 'E60022'}
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))

    _min_l1_byte()
    # Fifth : check shape size, 64 bits limitation
    c0_size = cce_params.C0_SIZE
    fmap_size = fmap_batch * fmap_d * align(fmap_channel,
                                            c0_size) * fmap_h * fmap_w
    dedy_size = dedy_batch * dedy_d * align(dedy_channel,
                                            c0_size) * dedy_h * dedy_w
    filter_size = \
        align(filter_batch, c0_size) * filter_d * align(filter_channel, c0_size) \
        * filter_h * filter_w
    _check_64bits_limitation("fmap_size", fmap_size, dtype=x_dtype)
    _check_64bits_limitation("dedy_size", dedy_size, dtype=out_backprop_dtype)
    _check_64bits_limitation("filter_size", filter_size, dtype=res_dtype)

    result = (shape_x, shape_out_backprop, filter_sizes, strides, pads,
              dilations, x_dtype, out_backprop_dtype, res_dtype, kernel_name)
    return result


@util.check_input_type((list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (str, list, tuple), (list, tuple), str,
                       str, str, str)
def conv3d_backprop_filter_cce(shape_x,
                               shape_out_backprop,
                               filter_sizes,
                               strides,
                               pads,
                               dilations=(1, 1, 1, 1),
                               x_dtype='float16',
                               out_backprop_dtype='float16',
                               res_dtype='float32',
                               kernel_name="conv3d_backprop_filter_cce"):
    """
    Topi interface of conv3d backprop filter

    Parameters:
    ----------
    shape_x : The shape of feature map.
              5-D with shape [batch, depth, channels, height, weight].

    shape_out_backprop : The shape of gradients.
                         5-D with shape [batch, depth, channels, height, weight].

    filter_sizes : The shape of filter.
                   5-D with shape [batch, depth, channels, height, weight].

    strides : A list of ints. The stride of the sliding window.

    pads : "SAME"or"VALID",
           indicating the type of pads algorithm to use, or list.

    dilations : An optional list of ints. Default value is [1, 1, 1, 1].

    x_dtype : The dtype of feature map data. Default value is float16.

    out_backprop_dtype : The dtype of gradients data.
                         Default value is float16.

    res_dtype : The dtype of result(De/Dw) data. Default value is float32.

    kernel_name : Cce kernel name.
                  Default value is "conv3d_backprop_filter_cce"

    need_build : If need to build CCEC kernel. Default value is False.

    Returns : None
    ----------
    """
    def _ceil(x_1, x_2):
        if x_2 == 0:
            args_dict = {
                'errCode': 'E62502',
                'first_operand': str(x_1),
                'second_operand': str(x_2)
            }
            raise RuntimeError(args_dict,
                               err_mana.get_error_message(args_dict))
        return (x_1 + x_2 - 1) // x_2

    if get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES",
                                       "Hi3796CV300CS"):
        res_dtype = "float16"

    res = check_conv3dbp_filter_params(shape_x, shape_out_backprop,
                                       filter_sizes, strides, pads, dilations,
                                       x_dtype, out_backprop_dtype, res_dtype,
                                       kernel_name)
    shape_x, shape_out_backprop, filter_sizes, strides, pads, dilations, \
    x_dtype, out_backprop_dtype, res_dtype, kernel_name = res
    fmap_batch, fmap_depth, fmap_channel, fmap_h, fmap_w = shape_x
    dedy_batch, dedy_d, dedy_channel, dedy_h, dedy_w = shape_out_backprop

    c0_size = cce_params.C0_SIZE  # Channel axis should be align with 16
    shape_dedy = (dedy_batch, dedy_d, \
                  _ceil(dedy_channel, c0_size), dedy_h, dedy_w, c0_size)
    shape_fmap = (fmap_batch, fmap_depth, \
                  _ceil(fmap_channel, c0_size), fmap_h, fmap_w, c0_size)
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
    fmap = tvm.placeholder(shape_fmap, name="fmap", dtype=x_dtype)

    dedw = te.lang.cce.conv3d_backprop_filter_compute(
        input_x=fmap,
        out_backprop=dedy,
        filter_sizes=filter_sizes,
        strides=strides,
        padding=pads,
        dilations=dilations,
        res_dtype=res_dtype,
        kernel_name=kernel_name)

    tensor_list_input = [fmap, dedy]
    with tvm.target.cce():
        sch = generic.auto_schedule(dedw)

    real_outs = sch.cce_special["real_out_tensor"]
    tensor_list = tensor_list_input + real_outs

    config = {"name": kernel_name, "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
