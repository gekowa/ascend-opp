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
conv3d
"""
# pylint: disable=too-many-arguments, too-many-locals, too-many-statements, too-many-lines
from __future__ import absolute_import
from functools import reduce as func_reduce
from te.platform import CUBE_MKN
from te.platform.cce_conf import get_soc_spec
import te.lang.cce
from te.lang.cce.te_compute import conv3d_compute
from te.utils.error_manager import error_manager_util as err_mana
from te import tvm
from topi import generic
from topi.cce import util

Nonetype = type(None)

BIAS_LENGTH = 1
# [strides_batch, strides_depth, strides_height,
#  strides_width, strides_channel]
STRIDE_LENGTH = 5

DILATION_LENGTH = 5
# [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]
PADS_LENGTH = 6
# NDHWC or NCDHW
SHAPE_DIMS = 5


def _get_mad_dtype(w_dtype):
    """
    algorithm: get the dtype of mad

    Parameters
    ----------
    w_dtype: the dtype of filter

    Returns
    -------
    mad dtype
    """
    mad_dtype = "float32"
    if w_dtype == 'int8':
        mad_dtype = "int32"
    elif get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES",
                                         "Hi3796CV300CS"):
        mad_dtype = "float16"

    return mad_dtype


def _conv3d_compute(shape_fm,
                    shape_filter,
                    bias,
                    stride_dhw,
                    pads,
                    fmp_dtype,
                    w_dtype,
                    res_dtype,
                    kernel_name='conv3d'):
    """
    algorithm: compute conv3d

    Parameters
    ----------
    shape_fm: the shape of feature,
        a list/tuple of 'int' that has length `== 5`

    shape_filter: the shape of filter, a list of 'int' that has length `== 5`

    bias: dict with keys(shape and dtype) or None
        input bias tensor

    stride_dhw: A list of `ints` that has length `== 3`.

    pads: tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    fmp_dtype: the dtype of feature

    w_dtype: the dtype of filter

    res_dtype: the dtype of output

    Returns
    -------
    list of tensor
    """
    batch, cin, fmp_d, fmp_h, fmp_w = shape_fm
    fmp_block_k = CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fmp_ndc1hwc0 = (batch, fmp_d, cin // fmp_block_k, fmp_h, fmp_w,
                          fmp_block_k)

    cout, cin, w_d, w_h, w_w = shape_filter
    w_block_k = CUBE_MKN[w_dtype]['mac'][1]
    w_block_n = CUBE_MKN[w_dtype]['mac'][2]
    shape_w_frac_z = (w_d * cin * w_h * w_w // w_block_k, cout // w_block_n,
                      w_block_n, w_block_k)

    mad_dtype = _get_mad_dtype(w_dtype)

    data = tvm.placeholder(shape_fmp_ndc1hwc0, name='Fmap', dtype=fmp_dtype)
    weight = tvm.placeholder(shape_w_frac_z, name='Filter', dtype=w_dtype)
    bias_tensor = None
    if bias is not None:
        bias_tensor = tvm.placeholder((cout, ),
                                      name='bias_tensor',
                                      dtype=res_dtype)
    conv3d_dict = {
        "bias_tensor": bias_tensor,
        "pads": pads,
        "shape_filter_ncdhw": shape_filter,
        "stride_dhw": stride_dhw,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "kernel_name": kernel_name
    }
    conv_res = te.lang.cce.te_compute.conv3d(data, weight, conv3d_dict)
    if bias is not None:
        tensor_list = [data, weight, bias_tensor, conv_res]
    else:
        tensor_list = [data, weight, conv_res]

    return tensor_list


def check_conv3d_dtype(fmp_dtype, w_dtype, res_dtype):
    """
    algorithm: check the input params of conv3d

    Parameters
    ----------

    fmp_dtype: the dtype of feature

    w_dtype: the dtype of filter

    res_dtype: the dtype of output

    Returns
    -------
    None
    """

    util.check_dtype_rule(fmp_dtype, ('float16', ))
    util.check_dtype_rule(w_dtype, ('float16', ))
    util.check_dtype_rule(res_dtype, ('float16', ))


def format_normalize(fmp_format, w_format, fmp_shape, w_shape, strides,
                     dilations):
    """
    algorithm: unified format

    Parameters
    ----------
    fmp_format: The data format of the input feature.

    w_format: The data format of the input filter.

    fmp_shape: the shape of feature,
        a list/tuple of 'int' that has length `== 5`

    w_shape: the shape of filter, a list of 'int' that has length `== 5`

    strides: A list of `ints` that has length `== 5`.

    dilations: tuple/list of 5 integers.
        dilation on D/H/W, format sensitive,
        Dilations in the batch and depth dimensions must be 1.

    Returns
    -------
    shape_fm, shape_filter, stride_dhw, dilation_hw
    """
    if fmp_format == "NCDHW":
        shape_fm = list(fmp_shape)
        stride_dhw = strides[2:]
        dilation_hw = dilations[3:]
    elif fmp_format == "NDHWC":
        shape_fm = [
            fmp_shape[0], fmp_shape[4], fmp_shape[1], fmp_shape[2],
            fmp_shape[3]
        ]
        stride_dhw = strides[1:4]
        dilation_hw = dilations[2:4]
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'input',
            'expected_format_list': '[{}, {}]'.format('NCDHW', 'NDHWC'),
            'format': fmp_format
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    if w_format == "NCDHW":
        shape_filter = list(w_shape)
    elif w_format == "NDHWC":
        shape_filter = [
            w_shape[0], w_shape[4], w_shape[1], w_shape[2], w_shape[3]
        ]
    elif w_format == "DHWCN":
        shape_filter = [
            w_shape[4], w_shape[3], w_shape[0], w_shape[1], w_shape[2]
        ]
    else:
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'weight',
            'expected_format_list': '[{}, {}, {}]'
                                    .format('NCDHW', 'NDHWC', 'DHWCN'),
            'format': w_format
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))
    return shape_fm, shape_filter, stride_dhw, dilation_hw


def check_input_param(fmp_shape, w_shape, fmp_dtype, w_dtype, res_dtype,
                      fmp_format, w_format, bias, strides, pads, dilations):
    """
    algorithm: check the input params of conv3d

    Parameters
    ----------
    fmp_shape: the shape of feature,
        a list/tuple of 'int' that has length `== 5`

    w_shape: the shape of filter, a list of 'int' that has length `== 5`

    fmp_dtype: the dtype of feature

    w_dtype: the dtype of filter

    res_dtype: the dtype of output

    fmp_format: The data format of the input feature.

    w_format: The data format of the input filter.

    bias: dict with keys(shape and dtype) or None
        input bias tensor

    strides: A list of `ints` that has length `== 5`.

    pads: tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 5 integers.
        dilation on D/H/W, format sensitive,
        Dilations in the batch and depth dimensions must be 1.

    Returns
    -------
    None
    """
    if bias is not None:
        bias_dtype = bias.get("dtype")
        util.check_dtype_rule(bias_dtype, ('float16', ))
        bias_shape = bias.get("ori_shape")
        if len(bias_shape) != BIAS_LENGTH:
            dict_args = {
                'errCode': 'E60006',
                'param_name': 'bias',
                'expected_length': '1',
                'length': '{}'.format(len(bias_shape))
            }
            raise RuntimeError(dict_args,
                               err_mana.get_error_message(dict_args))
    if len(strides) != STRIDE_LENGTH:
        dict_args = {
            'errCode': 'E60006',
            'param_name': 'strides',
            'expected_length': '5',
            'length': '{}'.format(len(strides))
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))
    if len(dilations) != DILATION_LENGTH:
        dict_args = {
            'errCode': 'E60006',
            'param_name': 'dilations',
            'expected_length': '5',
            'length': '{}'.format(len(dilations))
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))
    # check dilations for it1
    if len(set(dilations)) != 1 or dilations[2] != 1:
        dict_args = {
            'errCode': 'E62001',
            'dilation_h': str(dilations[2]),
            'dilation_w': str(dilations[3]),
            'dilation_d': str(dilations[1])
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    if len(pads) != PADS_LENGTH:
        dict_args = {
            'errCode': 'E62501',
            'param_name': 'pads',
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    util.check_shape_rule(fmp_shape, min_dim=SHAPE_DIMS, max_dim=SHAPE_DIMS)
    util.check_shape_rule(w_shape, min_dim=SHAPE_DIMS, max_dim=SHAPE_DIMS)

    # normalized format as NCDHW
    shape_fm, shape_filter, stride_dhw, dilation_hw = format_normalize(
        fmp_format, w_format, fmp_shape, w_shape, strides, dilations)

    check_conv3d_dtype(fmp_dtype, w_dtype, res_dtype)

    te.lang.cce.te_compute.check_conv3d_shape(shape_fm, shape_filter, pads,
                                              stride_dhw, fmp_dtype, w_dtype)

    return shape_fm, shape_filter, stride_dhw, dilation_hw


@util.check_input_type(dict, dict, (dict, Nonetype), (dict, Nonetype), dict,
                       (tuple, list), (tuple, list), (tuple, list), int,
                       str, int, str)
def conv3d(fmap,
           weight,
           bias,
           offset_w,
           output,
           strides,
           pads,
           dilations=(1, 1, 1, 1, 1),
           groups=1,
           data_format="NDHWC",
           offset_x=0,
           kernel_name="conv3d"):
    """
    algorithm: conv3d

    Parameters
    ----------
    fmap: dict with keys(shape and dtype)
        input 5d feature map tensor

    weight: dict with keys(shape and dtype)
        input 5d weight tensor

    bias: dict with keys(shape and dtype) or None
        input bias tensor

    offset_w: dict with keys(shape and dtype) or None
        input offset_w tensor

    output: dict with keys(shape and dtype)
        output tensor, dtype must be assigned

    strides: tuple/list of 5 integers, format sensitive
        [strides_batch, strides_depth, strides_height,
         strides_width, strides_channel]

    pads: tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 5 integers.
        dilation on D/H/W, format sensitive,
        Dilations in the batch and depth dimensions must be 1.

    groups: int of blocked connections from input channels to output channels
        default value 1

    data_format: The data format of the input and output data. With the
        default format "NDHWC",

    offset_x: int
        input offset_x value

    kernel_name: str
        kernel name, default value is "conv3d"

    Returns
    -------
    None
    """
    def _conv3d_achieve_with_tvm():
        tensor_list = _conv3d_compute(shape_fm,
                                      shape_filter,
                                      bias,
                                      stride_dhw,
                                      pads,
                                      fmp_dtype,
                                      w_dtype,
                                      res_dtype,
                                      kernel_name=kernel_name)

        with tvm.target.cce():
            sch = generic.auto_schedule(tensor_list[-1])

        config = {"name": kernel_name, "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    fmp_shape = fmap.get("ori_shape")
    fmp_dtype = fmap.get("dtype")
    fmp_format = data_format
    w_shape = weight.get("ori_shape")
    w_dtype = weight.get("dtype")
    w_format = weight.get("ori_format")
    res_dtype = output.get("dtype")

    fmp_dtype = fmp_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()

    # normalized format as NCDHW
    shape_fm, shape_filter, stride_dhw, _ = \
        check_input_param(fmp_shape, w_shape, fmp_dtype, w_dtype,
                          res_dtype, fmp_format, w_format, bias, strides,
                          pads, dilations)

    pads = list(pads)
    stride_dhw = list(stride_dhw)

    # C and Cout align 16
    shape_fm = list(shape_fm)
    fmp_block_k = CUBE_MKN[fmp_dtype]['mac'][1]
    shape_fm[1] = (
        (shape_fm[1] + fmp_block_k - 1) // fmp_block_k) * fmp_block_k
    w_block_k = CUBE_MKN[w_dtype]['mac'][1]
    shape_filter = list(shape_filter)
    shape_filter[1] = (
        (shape_filter[1] + w_block_k - 1) // w_block_k) * w_block_k
    w_block_n = CUBE_MKN[w_dtype]['mac'][2]
    shape_filter[0] = (
        (shape_filter[0] + w_block_n - 1) // w_block_n) * w_block_n

    _conv3d_achieve_with_tvm()
