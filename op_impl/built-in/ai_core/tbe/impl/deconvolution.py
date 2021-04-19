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
deconvolution
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform import CUBE_MKN
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_policy import get_L1_info
from te.utils.error_manager import error_manager_util as err_man
from topi import generic
from topi.cce import util
import impl.util.util_deconv_comm as comm

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

NoneType = type(None)


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


@util.check_input_type(dict, dict, (dict, NoneType), (dict, NoneType), dict,
                       (tuple, list), (str, tuple, list), (tuple, list),
                       int, str, int, str)
def deconvolution(x, # pylint: disable=invalid-name,R0913,R0914,W0613
                  weight, bias, offset_w, y, strides,
                  pads, dilations=(1, 1, 1, 1),
                  groups=None, data_format="NCHW", offset_x=0,
                  kernel_name="deconvolution"):
    """
    algorithm: deconvolution

    Parameters
    ----------
    x: dict with keys(shape and dtype)
                  The shape of gradients.

    weight: dict with keys(shape and dtype)
            input weight tensor

    offset_w: the offset for weight

    bias: dict with keys(shape and dtype)
        The shape of bias.

    y: dict with keys(shape and dtype)
       deconvolution output tensor, dtype must be assigned

    strides: tuple/list of 2 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated deconvolution
    groups: int
            param for group deconvolution

    data_format: str
            An optional string from: "NCHW". Defaults to "NCHW".
            Specify the data format of the input and output data.

    kernel_name: str
                 kernel name, default value is "deconvolution"

    offset_x: the offset for x

    Returns
    -------
    None
    """

    ori_shape_x = x.get("ori_shape")
    ori_shape_filters = weight.get("ori_shape")
    ori_shape_res = y.get("ori_shape")

    x_dtype = x.get("dtype")
    filters_dtype = weight.get("dtype")
    res_dtype = y.get("dtype")

    ori_format_x = x.get("ori_format")
    ori_format_filters = weight.get("ori_format")
    ori_format_res = y.get("ori_format")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(ori_shape_filters,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(ori_shape_x,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(ori_shape_res,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)
    util.check_shape_rule(dilations,
                          CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                          DEFAULT_MAX_SHAPE_NUM)

    if len(strides) == 4:
        h_index = data_format.find('H')
        w_index = data_format.find('W')
        strides = [strides[h_index], strides[w_index]]

    shape_filters = comm.get_filter_shape(ori_format_filters, ori_shape_filters)

    shape_x = comm.get_shape_out_backprop(
        ori_format_x, ori_shape_x)

    shape_res = comm.get_shape_res(ori_format_res, ori_shape_res)

    # get fusion para in L1 fusion
    fusion_para = deconvolution_fusion_para(x, y)
    valid_shape = fusion_para.get("valid_shape")
    if valid_shape and valid_shape[2] == shape_x[2]:
        fusion_para["valid_shape"] = ()
        fusion_para["slice_offset"] = ()
        fusion_para["output_offset"] = ()

    dilations = comm.get_shape_dilation(ori_format_x, dilations)

    bias_flag = bias is not None

    deconvolution_cce(shape_filters, shape_x, shape_res, strides,
                      pads, dilations=dilations, filter_dtype=filters_dtype,
                      x_dtype=x_dtype, res_dtype=res_dtype, bias=bias_flag,
                      offset_x=offset_x, fusion_para=fusion_para,
                      kernel_name=kernel_name)


def deconvolution_fusion_para(x, y):
    """
    get fusion para for L1 fusion
    """
    input_memory_type = x.get("addr_type") \
        if "addr_type" in x else 0
    output_memory_type = y.get("addr_type") \
        if "addr_type" in y else 0
    valid_shape = x.get("valid_shape") \
        if "valid_shape" in x else ()
    slice_offset = x.get("slice_offset") \
        if "slice_offset" in x else ()
    output_offset = y.get("slice_offset") \
        if "slice_offset" in y else ()
    l1_fusion_type = x.get("L1_fusion_type") \
        if "L1_fusion_type" in x else -1
    fmap_l1_addr_flag = x.get("L1_addr_flag", False)
    fmap_l1_valid_size = x.get("L1_valid_size", 0)



    l1_fusion_enable_flag = get_L1_info("L1_fusion_enabled")

    if input_memory_type not in (0, 1, 2):
        args_dict = {
            "errCode": "E65008",
            "input_memory_type_range": "(0, 1, 2)",
            "input_memory_type": str(input_memory_type)
        }
        raise RuntimeError(args_dict,
                           err_man.get_error_message(args_dict))
    if output_memory_type not in (0, 1, 2):
        args_dict = {
            "errCode": "E65009",
            "output_memory_type_range": "(0, 1, 2)",
            "output_memory_type": str(output_memory_type)
        }
        raise RuntimeError(args_dict,
                           err_man.get_error_message(args_dict))
    if valid_shape and not slice_offset:
        reason = "valid shape exists, slice shape cannot be []"
        args_dict = {
            "errCode": "E60108",
            "reason": reason
        }
        raise RuntimeError(args_dict,
                           err_man.get_error_message(args_dict))
    if valid_shape and not output_offset:
        reason = "valid shape exists, output offset cannot be []"
        args_dict = {
            "errCode": "E60108",
            "reason": reason
        }
        raise RuntimeError(args_dict,
                           err_man.get_error_message(args_dict))

    valid_shape = shape_to_list(valid_shape)
    slice_offset = shape_to_list(slice_offset)
    output_offset = shape_to_list(output_offset)

    if not l1_fusion_enable_flag:
        input_memory_type = 0
        output_memory_type = 0
        valid_shape = []
        slice_offset = []
        output_offset = []
        l1_fusion_type = -1
        fmap_l1_addr_flag = False
        fmap_l1_valid_size = 0

    fusion_para = {"input_memory_type": input_memory_type,
                   "output_memory_type": output_memory_type,
                   "valid_shape": valid_shape,
                   "slice_offset": slice_offset,
                   "output_offset": output_offset,
                   "l1_fusion_type": l1_fusion_type,
                   "fmap_l1_addr_flag": fmap_l1_addr_flag,
                   "fmap_l1_valid_size": fmap_l1_valid_size}

    return fusion_para


def deconv_compute_fusion_para(x, y):
    """
    get fusion para for L1 fusion
    """
    input_memory_type = x.op.attrs["addr_type"].value \
        if "addr_type" in x.op.attrs else 0
    valid_shape = x.op.attrs["valid_shape"] \
        if "valid_shape" in x.op.attrs else ()
    slice_offset = x.op.attrs["slice_offset"] \
        if "slice_offset" in x.op.attrs else ()
    output_offset = y.get("slice_offset") \
        if "slice_offset" in y else ()
    l1_fusion_type = x.op.attrs["L1_fusion_type"].value \
        if "L1_fusion_type" in x.op.attrs else -1
    fmap_l1_addr_flag = x.op.attrs["L1_addr_flag"].value \
        if "L1_addr_flag" in x.op.attrs else False
    fmap_l1_valid_size = x.op.attrs["L1_valid_size"].value\
        if "L1_valid_size" in x.op.attrs else 0

    l1_fusion_enable_flag = get_L1_info("L1_fusion_enabled")

    if input_memory_type not in (0, 1, 2):
        args_dict = {
            "errCode": "E65008",
            "input_memory_type_range": "(0, 1, 2)",
            "input_memory_type": str(input_memory_type)
        }
        raise RuntimeError(args_dict,
                           err_man.get_error_message(args_dict))

    if valid_shape and not slice_offset:
        reason = "valid shape exists, slice shape cannot be []"
        args_dict = {
            "errCode": "E60108",
            "reason": reason
        }
        raise RuntimeError(args_dict,
                           err_man.get_error_message(args_dict))
    if valid_shape and not output_offset:
        reason = "valid shape exists, output offset cannot be []"
        args_dict = {
            "errCode": "E60108",
            "reason": reason
        }
        raise RuntimeError(args_dict,
                           err_man.get_error_message(args_dict))

    valid_shape = shape_to_list(valid_shape)
    slice_offset = shape_to_list(slice_offset)
    output_offset = shape_to_list(output_offset)

    if not l1_fusion_enable_flag:
        input_memory_type = 0
        valid_shape = []
        slice_offset = []
        output_offset = []
        l1_fusion_type = -1
        fmap_l1_addr_flag = False
        fmap_l1_valid_size = 0

    fusion_para = {"input_memory_type": input_memory_type,
                   "output_memory_type": "fuse_flag",
                   "valid_shape": valid_shape,
                   "slice_offset": slice_offset,
                   "output_offset": output_offset,
                   "l1_fusion_type": l1_fusion_type,
                   "fmap_l1_addr_flag": fmap_l1_addr_flag,
                   "fmap_l1_valid_size": fmap_l1_valid_size}

    return fusion_para


@fusion_manager.register("deconvolution")
def deconvolution_compute(x, weight, bias, offset_w, y, strides,
                          pads, dilations=(1, 1, 1, 1),
                          groups=None, data_format="NCHW", offset_x=0,
                          kernel_name="deconvolution"):
    """
    used for fusion
    Parameters
    ----------
    x: dict with keys(shape and dtype)
                  The shape of gradients.

    weight: dict with keys(shape and dtype)
            input weight tensor

    offset_w: the offset for weight

    bias: dict with keys(shape and dtype)
        The shape of bias.

    y: dict with keys(shape and dtype)
       deconvolution output tensor, dtype must be assigned

    strides: tuple/list of 2 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated deconvolution
    groups: int
            param for group deconvolution

    data_format: str
            An optional string from: "NCHW". Defaults to "NCHW".
            Specify the data format of the input and output data.

    kernel_name: str
                 kernel name, default value is "deconvolution"

    offset_x: the offset for x

    Returns
    -------
    None
    """

    ori_shape_weight = [i.value for i in weight.op.attrs["ori_shape"]]
    ori_shape_x = \
        [i.value for i in x.op.attrs["ori_shape"]]
    ori_shape_res = [i for i in y["ori_shape"]]

    weight_dtype = weight.dtype
    x_dtype = x.dtype
    res_dtype = y["dtype"]

    ori_format_weight = weight.op.attrs["ori_format"]
    ori_format_x = x.op.attrs["ori_format"]
    ori_format_res = y["ori_format"]

    fusion_para = deconv_compute_fusion_para(x, y)
    if len(strides) == 4:
        h_index = data_format.find('H')
        w_index = data_format.find('W')
        strides = [strides[h_index], strides[w_index]]

    if weight_dtype == 'int8':
        ori_shape_weight[0], ori_shape_weight[1] = ori_shape_weight[1],\
                                                   ori_shape_weight[0]
    shape_weight = comm.get_filter_shape(
        ori_format_weight, ori_shape_weight)
    shape_x = comm.get_shape_out_backprop(
        ori_format_x, ori_shape_x)
    shape_res = comm.get_shape_res(ori_format_res, ori_shape_res)
    dilations = comm.get_shape_dilation(ori_format_x, dilations)

    valid_shape = fusion_para.get("valid_shape")
    if valid_shape and valid_shape[2] == shape_x[2]:
        fusion_para["valid_shape"] = ()
        fusion_para["slice_offset"] = ()
        fusion_para["output_offset"] = ()

    comm.check_conv2dbp_input_params(shape_weight, shape_x,
                                     shape_res, strides, pads, dilations,
                                     weight_dtype, x_dtype,
                                     res_dtype, kernel_name, fusion_para)

    pads = comm.get_padlist(pads, shape_res, strides, shape_weight, dilations)

    res = te.lang.cce.conv2d_backprop_input_compute(weight,
                                                    x,
                                                    shape_weight,
                                                    shape_res,
                                                    strides,
                                                    pads,
                                                    dilations,
                                                    res_dtype=res_dtype,
                                                    tensor_bias=bias,
                                                    offset_x=offset_x,
                                                    fusion_para=fusion_para,
                                                    kernel_name=kernel_name)

    return res


@util.check_input_type((list, tuple), (list, tuple), (list, tuple),
                       (list, tuple), (str, list, tuple), (list, tuple),
                       str,
                       str,
                       str,
                       bool,
                       int,
                       (dict, NoneType),
                       str)
def deconvolution_cce(shape_filter, shape_x, # pylint: disable=R0913, R0914
                      input_sizes, strides, pads, dilations=(1, 1, 1, 1),
                      filter_dtype='float16', x_dtype='float16',
                      res_dtype='float16', bias=False, offset_x=0,
                      fusion_para=None,
                      kernel_name="deconvolution_cce"):
    """
    Topi interface of deconvolution

    Parameters:
    ----------
    shape_filter : The shape of filter.
                   4-D with shape [batch, channels, height, width].

    shape_x : The shape of gradients.
              4-D with shape [batch, channels, height, width].

    input_sizes : The shape of feature map.
                  4-D with shape [batch, channels, height, width].

    strides : A list of ints. The stride of the sliding window.

    pads : "SAME"or"VALID" indicating the type of pads algorithm to use,
           or list.

    dilations : An optional list of ints. Default value is [1, 1, 1, 1].

    filter_dtype : The dtype of filter data. Default value is float16.

    x_dtype : The dtype of gradients data. Default value is float16.

    res_dtype : The dtype of result(De/Dx) data. Default value is float16.

    bias: False: no bias, True: have bias

    fusion_para: the L1 fusion para

    kernel_name : Cce kernel name. Default value is "deconvolution_cce"

    Returns : None
    ----------
    """

    def _ceil(x_1, x_2):
        if x_2 == 0:
            raise RuntimeError("Division by zero")
        return (x_1 + x_2 - 1) // x_2
    if fusion_para is None:
        fusion_para = {"input_memory_type": 0,
                       "output_memory_type": 0,
                       "valid_shape": (),
                       "slice_offset": (),
                       "output_offset": (),
                       "l1_fusion_type": -1,
                       "fmap_l1_addr_flag": False,
                       "fmap_l1_valid_size": 0}

    if filter_dtype == "int8" and x_dtype == "int8":
        shape_filter = [shape_filter[1], shape_filter[0],
                        shape_filter[2], shape_filter[3]]
    res = comm.check_conv2dbp_input_params(shape_filter, shape_x, input_sizes,
                                           strides, pads, dilations,
                                           filter_dtype, x_dtype,
                                           res_dtype, kernel_name, fusion_para)

    shape_filter, shape_x, input_sizes, strides, pads, dilations, \
    filter_dtype, x_dtype, res_dtype, kernel_name = res

    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_x
    filter_batch, filter_channel, filter_h, filter_w = shape_filter

    _, dy_k0, _ = CUBE_MKN[x_dtype]['mac']
    _, w_k0, w_n0 = CUBE_MKN[filter_dtype]['mac']
    shape_dedy = (dedy_batch,
                  _ceil(dedy_channel, dy_k0), dedy_h, dedy_w, dy_k0)
    filter_channel = comm.align(filter_channel, w_n0)
    if filter_dtype == "int8" and x_dtype == "int8":
        shape_filter_frac = (
            _ceil(filter_batch, w_k0)*filter_h*filter_w,
            _ceil(filter_channel, w_n0), w_n0, w_k0)
    else:
        shape_filter_frac = (
            _ceil(filter_channel, w_n0)*filter_h*filter_w,
            _ceil(filter_batch, w_k0), w_k0, w_n0)
    tensor_dedy = tvm.placeholder(shape_dedy, name="dedy", dtype=x_dtype)

    tensor_filter_frac = tvm.placeholder(shape_filter_frac,
                                         name="filter", dtype=filter_dtype)

    if bias:
        tensor_bias = tvm.placeholder(
            (filter_channel,), name='tensor_bias', dtype=res_dtype
        )
    else:
        tensor_bias = None

    dedx = te.lang.cce.conv2d_backprop_input_compute(
        filters=tensor_filter_frac,
        out_backprop=tensor_dedy,
        filter_sizes=shape_filter,
        input_sizes=input_sizes,
        strides=strides,
        padding=pads,
        dilations=dilations,
        res_dtype=res_dtype,
        tensor_bias=tensor_bias,
        offset_x=offset_x,
        fusion_para=fusion_para,
        kernel_name=kernel_name
    )
    if bias:
        tensor_list = [tensor_dedy, tensor_filter_frac, tensor_bias, dedx]
    else:
        tensor_list = [tensor_dedy, tensor_filter_frac, dedx]

    with tvm.target.cce():
        sch = generic.auto_schedule(dedx)

    config = {
        "name": kernel_name,
        "tensor_list": tensor_list
    }

    te.lang.cce.cce_build_code(sch, config)
