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
conv3d_transpose_d
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform import cce_params
from te.utils.error_manager import error_manager_util as err_mana
from topi import generic
from topi.cce import util
from .conv3d_backprop_input_d import check_conv3dbp_input_params


Nonetype = type(None)


def ceil(x_1, x_2):
    """
    do ceiling division

    Parameters
    ----------
    x_1: int
    x_2: int
    Returns
    -------
    result
    """
    if x_2 == 0:
        dict_args = {
            'errCode': 'E62502',
            'first_operand': str(x_1),
            'second_operand': str(x_2),
        }
        raise RuntimeError(dict_args,
                           err_mana.get_error_message(dict_args))
    return (x_1 + x_2 - 1) // x_2


def align(x_1, x_2):
    """
    align x_1 with x_2

    Parameters
    ----------
    x_1: int
    x_2: int
    Returns
    -------
    result
    """
    if x_2 == 0:
        dict_args = {
            'errCode': 'E62502',
            'first_operand': str(x_1),
            'second_operand': str(x_2),
        }
        raise RuntimeError(dict_args,
                           err_mana.get_error_message(dict_args))
    return ((x_1 + x_2 - 1) // x_2) * x_2


@util.check_input_type(dict, dict, (dict, Nonetype), (dict, Nonetype), dict,
                       (tuple, list), (tuple, list), (list, tuple, str),
                       (tuple, list), int, str, (list, tuple), int, str)
def conv3d_transpose_d(out_backprop, filters, # pylint: disable=R0913,R0914
                       bias, offset_w, y_input, input_sizes,
                       strides, pads, dilations=(1, 1, 1, 1, 1), groups=1,
                       data_format="NDHWC",
                       output_padding=[0, 0, 0, 0, 0],
                       offset_x=0, kernel_name="conv3d_transpose"):
    """
    algorithm: conv3d_transpose

    Parameters
    ----------
    out_backprop: dict with keys(shape and dtype)
                  The shape of gradients.

    filters: dict with keys(shape and dtype)
            input weight tensor

    bias: dict with keys(shape and dtype) or None
        input bias tensor

    offset_w: dict with keys(shape and dtype) or None
        input offset_w tensor

    y_input: dict with keys(shape and dtype)
       conv3d_transpose output tensor, dtype must be assigned

    input_sizes: The shape of feature map.
                 5-D with shape [batch, depth, height, weight, channels].

    strides: tuple/list of 5 integers
             filter move stride

    pads: tuple/list of 6 integers
             [pad_front, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 5 integers
           filter expand size of dilated conv3d_transpose

    groups: int of blocked connections from input channels to output channels

    data_format: The data format of the input and output data. With the
        default format "NDHWC"

    output_padding: The size will be added in the output shape.

    offset_x: int
        input offset_x value

    kernel_name: str
                 kernel name, default value is "conv3d_transpose"

    Returns
    -------
    None
    :param data_format:
    """
    def _ncdhw2ndhwc(shape1):
        shape2 = (shape1[0], shape1[2], shape1[3], shape1[4], shape1[1])
        return shape2

    ori_shape_filters = filters.get("ori_shape")
    ori_shape_out_backprop = out_backprop.get("ori_shape")
    ori_shape_res = input_sizes
    ori_shape_strides = strides
    ori_shape_dialtions = dilations

    filters_dtype = filters.get("dtype")
    out_backprop_dtype = out_backprop.get("dtype")
    res_dtype = y_input.get("dtype")

    ori_format_filters = filters.get("ori_format")
    ori_format_out_backprop = data_format
    ori_format_res = data_format

    if ori_format_filters == "DHWCN":
        shape_filters = ori_shape_filters
    elif ori_format_filters == "NDHWC":
        shape_filters = (ori_shape_filters[1],
                         ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[0],
                        )
    elif ori_format_filters == "NCDHW":
        shape_filters = (ori_shape_filters[2],
                         ori_shape_filters[3],
                         ori_shape_filters[4],
                         ori_shape_filters[1],
                         ori_shape_filters[0],
                        )
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'filter',
            'expected_format_list': '[{}, {}, {}]'
                                    .format('DHWCN', 'NDHWC', 'NCDHW'),
            'format': ori_format_filters
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    if ori_format_out_backprop == "NDHWC":
        shape_out_backprop = ori_shape_out_backprop
        shape_strides = ori_shape_strides
        shape_dilations = ori_shape_dialtions
    elif ori_format_out_backprop == "NCDHW":
        shape_out_backprop = _ncdhw2ndhwc(ori_shape_out_backprop)
        shape_strides = _ncdhw2ndhwc(ori_shape_strides)
        shape_dilations = _ncdhw2ndhwc(ori_shape_dialtions)
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y_backprop',
            'expected_format_list': '[{}, {}]'.format('NDHWC', 'NCDHW'),
            'format': ori_format_out_backprop
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    if ori_format_res == "NDHWC":
        shape_res = ori_shape_res
    elif ori_format_res == "NCDHW":
        shape_res = _ncdhw2ndhwc(ori_shape_res)
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'y',
            'expected_format_list': '[{}, {}]'.format('NDHWC', 'NCDHW'),
            'format': ori_format_res
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    conv3d_transpose_cce(shape_filters,
                         shape_out_backprop,
                         shape_res,
                         shape_strides,
                         pads,
                         shape_dilations,
                         filters_dtype,
                         out_backprop_dtype,
                         res_dtype,
                         kernel_name)


@util.check_input_type((list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (str, list, tuple), (list, tuple),
                       str,
                       str,
                       str,
                       str)
def conv3d_transpose_cce(shape_filter, # pylint: disable=R0913,R0914
                         shape_out_backprop, input_sizes,
                         strides, pads, dilations=(1, 1, 1, 1, 1),
                         filter_dtype='float16',
                         out_backprop_dtype='float16',
                         res_dtype='float16',
                         kernel_name="conv3d_transpose_cce"):
    """
    Topi interface of conv3d transpose

    Parameters:
    ----------
    shape_filter : The shape of filter.
                   5-D with shape [ depth, height, weight, batch, channels].

    shape_out_backprop : The shape of gradients.
                         5-D with shape [batch,
                                         depth, height, weight, channels].

    input_sizes : The shape of feature map.
                  5-D with shape [batch, depth, height, weight, channels].

    strides : A list of ints. The stride of the sliding window.

    pads : A list of ints.

    dilations : An optional list of ints. Only support [1, 1, 1, 1, 1] now.

    filter_dtype : The dtype of filter data. Default value is float16.

    out_backprop_dtype : The dtype of gradients data. Default value is float16.

    res_dtype : The dtype of result(De/Dx) data. Default value is float16.

    kernel_name : Cce kernel name. Default value is "conv3d_transpose_cce"

    Returns : None
    ----------
    """


    def _conv3d_transpose_achieve_with_tvm():
        dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=out_backprop_dtype)
        shape_filter_ncdhw = [filter_batch, filter_channel, filter_depth,
                              filter_h, filter_w]

        filters = tvm.placeholder(shape_filter_frac,
                                  name="filter", dtype=filter_dtype)

        dedx = te.lang.cce.conv3d_backprop_input_compute(
            filters=filters,
            out_backprop=dedy,
            filter_sizes=shape_filter_ncdhw,
            input_sizes=input_sizes,
            strides=strides,
            padding=padding,
            dilations=dilations,
            res_dtype=res_dtype,
            kernel_name=kernel_name
        )
        tensor_list = [dedy, filters, dedx]

        with tvm.target.cce():
            sch = generic.auto_schedule(dedx)

        config = {
            "name": kernel_name,
            "tensor_list": tensor_list
        }
        te.lang.cce.cce_build_code(sch, config)

    res = check_conv3dbp_input_params(shape_filter, shape_out_backprop,
                                      input_sizes, strides, pads, dilations,
                                      filter_dtype, out_backprop_dtype,
                                      res_dtype, kernel_name)
    shape_filter, shape_out_backprop, input_sizes, strides, padding, dilations,\
    filter_dtype, out_backprop_dtype, res_dtype, kernel_name = res

    dedy_batch, dedy_deep, dedy_h, dedy_w, dedy_channel = shape_out_backprop
    filter_depth, filter_h, filter_w,\
    filter_channel, filter_batch = shape_filter

    # Channel axis should be align with 16
    c0_size = cce_params.C0_SIZE
    shape_dedy = (dedy_batch,
                  dedy_deep,
                  ceil(dedy_channel, c0_size), dedy_h, dedy_w, c0_size)

    shape_filter_frac = (filter_depth,
                         ceil(filter_channel, c0_size) * filter_h * filter_w,
                         ceil(filter_batch, c0_size), c0_size, c0_size)
    _conv3d_transpose_achieve_with_tvm()
