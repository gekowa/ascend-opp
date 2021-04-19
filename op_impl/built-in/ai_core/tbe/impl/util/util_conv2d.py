#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

conv2d common
provide common function used by conv2d
"""

from te.platform import CUBE_MKN
from te.platform import cce_conf
from te.platform.cce_policy import get_L1_info
from te.utils.check_para import check_input_type
from te.utils.check_para import check_kernel_name
from te.utils.check_para import check_dtype_rule
from te.utils.error_manager import error_manager_conv2d as err_man
from te.lang.cce.te_compute.conv_compute import check_conv_shape

PAD_SHAPE_DIM = 2


def calc_para_from_tensor(inputs, weights, bias, offset_w, strides, pads,
                          dilations, offset_x, kernel_name,
                          data_format="NCHW"):

    shape_w = []
    for i in weights.op.attrs['ori_shape']:
        shape_w.append(i.value)
    shape_fm = []
    for i in inputs.shape:
        shape_fm.append(i.value)

    input_h = shape_fm[2]
    input_w = shape_fm[3]

    format_w = weights.op.attrs['ori_format'].value
    all_fmt = ["NCHW", "NHWC", "HWCN"]
    if format_w not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", \
        "weights", ["NCHW", "NHWC", "HWCN"], format_w)

    pos_c = format_w.find('C')
    pos_h = format_w.find('H')
    pos_w = format_w.find('W')
    weight_h = shape_w[pos_h]
    weight_w = shape_w[pos_w]
    shape_c = shape_w[pos_c]

    if len(strides) != 4:
        err_man.raise_err_should_be_4d("conv2d", "strides")
    if len(dilations) != 4:
        err_man.raise_err_should_be_4d("conv2d", "directions")

    format_x = inputs.op.attrs['ori_format'].value

    all_fmt = ["NCHW", "NHWC"]
    if format_x not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", \
        "input", ["NCHW", "NHWC"], format_x)
    pos_h = data_format.find('H')
    pos_w = data_format.find('W')
    strideh = strides[pos_h]
    stridew = strides[pos_w]
    dlt_h = dilations[pos_h]
    dlt_w = dilations[pos_w]

    if len(pads) == 4:
        padh = [pads[0], pads[1]]
        padw = [pads[2], pads[3]]
    else:
        err_man.raise_err_should_be_4d("conv2d", "pads shape")

    fusion_para = _conv2d_compute_fusion_para(inputs)

    valid_shape = fusion_para.get("valid_shape")
    if valid_shape and valid_shape[2] == shape_fm[2]:
        valid_shape = ()
        fusion_para["valid_shape"] = ()
        fusion_para["slice_offset"] = ()
    if valid_shape:
        input_h = valid_shape[2]
        input_w = valid_shape[3]

    strideh = _trans_stride(input_h, weight_h, strideh, padh, dlt_h)
    stridew = _trans_stride(input_w, weight_w, stridew, padw, dlt_w)

    para_dict = {"pad_h": padh, "pad_w": padw, "stride_h": strideh,
                 "stride_w": stridew, "dilate_h": dlt_h, "dilate_w": dlt_w,
                 "offset_x": offset_x, "filter_h": weight_h,
                 "filter_w": weight_w, "bias_tensor": bias,
                 "offset_w_tensor": offset_w,
                 "fusion_para": fusion_para,
                 "kernel_name": kernel_name}

    if cce_conf.get_soc_spec("SOC_VERSION") in \
    ("Hi3796CV300ES", "Hi3796CV300CS"):
        para_dict["mad_dtype"] = "float16"
        if weights.dtype != "float16":
            para_dict["mad_dtype"] = "int32"
    else:
        if cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend310",) \
        and weights.dtype == "int8":
            para_dict["mad_dtype"] = "int32"

    c0_optim_flg = False
    if shape_c <= 4 and ("format" in weights.op.attrs and
                         weights.op.attrs['format'].value == "FRACTAL_Z_C04"):
        c0_optim_flg = True
        if (weight_h == 1) and (weight_w == 1):
            err_man.raise_err_specific_user("conv2d", "weight shape does "\
                + "not support that H and W are both equal to 1 when C0=4.")

        if fusion_para["input_memory_type"] == 1:
            err_man.raise_err_specific_input_shape("conv2d", "c0 optim not "\
                + "support fmap from L1 directly (instead of DDR)")

    optim_dict = {"c0_optim_flg": c0_optim_flg}

    return para_dict, optim_dict


def calc_para_from_dict(inputs, weights, strides, pads,
                        dilations, outputs, data_format="NCHW"):
    shape_x = inputs.get("ori_shape")
    shape_w = weights.get("ori_shape")

    if len(strides) != 4:
        err_man.raise_err_should_be_4d("conv2d", "strides")
    if len(dilations) != 4:
        err_man.raise_err_should_be_4d("conv2d", "dilations")

    if len(pads) == 4:
        padh = [pads[0], pads[1]]
        padw = [pads[2], pads[3]]
    else:
        err_man.raise_err_should_be_4d("conv2d", "pads shape")

    if (not isinstance(shape_x, (tuple, list))) or len(shape_x) != 4:
        err_man.raise_err_should_be_4d("conv2d", "inputs")


    if (not isinstance(shape_w, (tuple, list))) or len(shape_w) != 4:
        err_man.raise_err_should_be_4d("conv2d", "weights")

    format_x = inputs.get("ori_format")
    all_fmt = ["NCHW", "NHWC"]
    if format_x not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", \
            "inputs", ["NCHW", "NHWC"], format_x)
    pos_n = format_x.find('N')
    pos_c = format_x.find('C')
    pos_h = format_x.find('H')
    pos_w = format_x.find('W')
    shape_fm = [shape_x[pos_n], shape_x[pos_c], shape_x[pos_h], shape_x[pos_w]]

    pos_attr_h = data_format.find('H')
    pos_attr_w = data_format.find('W')
    strideh = strides[pos_attr_h]
    stridew = strides[pos_attr_w]
    dlt_h = dilations[pos_attr_h]
    dlt_w = dilations[pos_attr_w]

    format_w = weights.get("ori_format")
    all_fmt = ["NCHW", "NHWC", "HWCN"]
    if format_w not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", \
            "weights", ["NCHW", "NHWC", "HWCN"], format_w)
    pos_n = format_w.find('N')
    pos_c = format_w.find('C')
    pos_h = format_w.find('H')
    pos_w = format_w.find('W')
    shape_filter = [shape_w[pos_n], shape_w[pos_c], \
                    shape_w[pos_h], shape_w[pos_w]]

    fusion_para = _conv2d_fusion_para(inputs, outputs)

    valid_shape = fusion_para.get("valid_shape")
    if valid_shape and valid_shape[2] == shape_fm[2]:
        valid_shape = ()
        fusion_para["valid_shape"] = ()
        fusion_para["slice_offset"] = ()

    if valid_shape:
        input_h = valid_shape[2]
        input_w = valid_shape[3]
    else:
        input_h = shape_fm[2]
        input_w = shape_fm[3]

    strideh = _trans_stride(input_h, shape_filter[2], strideh, padh, dlt_h)
    stridew = _trans_stride(input_w, shape_filter[3], stridew, padw, dlt_w)

    c0_optim_flg = False
    if shape_w[pos_c] <= 4 and weights.get("format") == "FRACTAL_Z_C04":
        c0_optim_flg = True
        if (shape_w[pos_h] == 1) and (shape_w[pos_w] == 1):
            err_man.raise_err_specific_user("conv2d", "weight shape "\
                + "does not support that H and W are both "\
                + "equal to 1 when C0=4.")
        if fusion_para["input_memory_type"] == 1:
            err_man.raise_err_specific_input_shape("conv2d", \
                "c0 optim not support fmap "\
                + "from L1 directly (instead of DDR)")
    optim_dict = {"c0_optim_flg": c0_optim_flg}

    return shape_fm, shape_filter, padh, padw, strideh, stridew, \
           dlt_h, dlt_w, optim_dict, fusion_para


@check_input_type((list, tuple), (list, tuple), (list, int), (list, int),
                       int, int, str, str, str, str,
                       bool, str, int, int, dict, dict)
def conv_layer_cce_para_check(shape_in, shape_w, padh, padw, strideh, stridew,
                              in_dtype, w_dtype, res_dtype, offset_w_dtype,
                              bias, kernel_name, dilateh=1, dilatew=1,
                              optim_dict=None, fusion_para=None):
    """

    Parameters
    ----------
    shape_in: shape of feature map

    shape_w: shape of weight

    padh: H direction padding

    padw: W direction padding

    strideh: H direction stride

    stridew: W direction stride

    in_dtype: the feature map data type

    w_dtype: the weight data type

    res_dtype: the result data type

    offset_w_dtype: weight offset data type, default 'int32'

    bias: the tag for bias or not

    fusion_para: the config for L1 or L2 Fusion

    kernel_name: cce kernel name

    dilateh: H direction spacing between kernel

    dilatew: W direction spacing between kernel

    optim_dict: optimize feature dict

    Returns
    -------
    None

    """
    check_kernel_name(kernel_name)
    check_dtype_rule(offset_w_dtype, ['int32'])
    if cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend310", "Hi3796CV300ES", \
        "Ascend710", "Ascend615", "Ascend610", "Hi3796CV300CS"):
        check_dtype_rule(in_dtype, ('int8', "float16"))
        check_dtype_rule(w_dtype, ('int8', "float16"))
        check_dtype_rule(res_dtype, ('int32', "float16"))
    else:
        check_dtype_rule(in_dtype, ['float16'])
        check_dtype_rule(w_dtype, ['float16'])
        check_dtype_rule(res_dtype, ['float16'])

    if isinstance(padh, list):
        if len(padh) != PAD_SHAPE_DIM:
            err_man.raise_err_specific_user("conv2d", "Dimension must be "\
                                            + str(PAD_SHAPE_DIM) + \
                                            " when padh is a list.")
        pad_top = padh[0]
        pad_bottom = padh[1]
    else:
        pad_top = padh
        pad_bottom = padh

    if isinstance(padw, list):
        if len(padw) != PAD_SHAPE_DIM:
            err_man.raise_err_specific_user("conv2d", "Dimension must be "\
                                            + str(PAD_SHAPE_DIM) + \
                                            " when padw is a list.")
        pad_left = padw[0]
        pad_right = padw[1]
    else:
        pad_left = padw
        pad_right = padw
    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False}
    optim_off = shape_in[1] > 4 or shape_w[1] > 4 or \
                (shape_w[2] == 1 and shape_w[3] == 1)
    if optim_dict.get("c0_optim_flg") is True:
        if optim_off:
            err_man.raise_err_specific_user("conv2d", "Invalid "\
                + "config for c0=4 optimize feature.")

    if fusion_para is None:
        fusion_para = {"input_memory_type": 0, "output_memory_type": 0,
                       "valid_shape": (), "slice_offset": (), \
                       "l1_fusion_type": -1, \
                       "fmap_l1_addr_flag": 0, \
                       "fmap_l1_valid_size": -1}

    dilation_not_pass = (dilateh > 1 or dilatew > 1) and w_dtype == 'int8'
    if dilation_not_pass:
        err_man.raise_err_specific_user("conv2d", "Quant conv does not "\
            + "support dilate > 1.")

    shape_in, shape_w = check_conv_shape(shape_in, shape_w,
                                         pad_top, pad_bottom,
                                         pad_left, pad_right, strideh, stridew,
                                         in_dtype, w_dtype, fusion_para,
                                         optim_dict, dilateh, dilatew)

    return shape_in, shape_w


def conv_layer_cce_shape_calc(shape_in, shape_w, in_dtype, \
    w_dtype, optim_dict):
    """

    Parameters
    ----------
    shape_in: shape of feature map

    shape_w: shape of weight

    in_dtype: the feature map data type

    w_dtype: the weight data type

    optim_dict: optimize feature dict

    Returns
    -------
    None

    """
    block_size_k = CUBE_MKN[in_dtype]['mac'][1]
    if optim_dict["c0_optim_flg"] and cce_conf.get_soc_spec("SOC_VERSION") in \
    ("Ascend710", "Ascend615", "Ascend610", "Hi3796CV300CS"):
        block_size_k = 4
    fmap_shape_nc1hwc0 = (shape_in[0], (shape_in[1] + block_size_k - 1) \
                          // block_size_k,
                          shape_in[2], shape_in[3], block_size_k)

    out_channel, in_channel_weight, filter_h, filter_w = shape_w
    block_size_k = CUBE_MKN[w_dtype]['mac'][1]
    block_size_n = CUBE_MKN[w_dtype]['mac'][2]
    if optim_dict["c0_optim_flg"]:
        filter_shape_frac_z = ((4 * filter_h * filter_w + block_size_k - 1) \
                               // block_size_k,
                               out_channel // block_size_n, block_size_n,
                               block_size_k)
    else:
        filter_shape_frac_z = (in_channel_weight * filter_h * filter_w \
                               // block_size_k,
                               out_channel // block_size_n, block_size_n,
                               block_size_k)
    return fmap_shape_nc1hwc0, filter_shape_frac_z


def _conv2d_compute_fusion_para(inputs):
    """
    get L2 fusion para for conv2d_compute
    """
    input_memory_type = inputs.op.attrs["addr_type"].value \
        if "addr_type" in inputs.op.attrs else 0
    valid_shape = inputs.op.attrs["valid_shape"] \
        if "valid_shape" in inputs.op.attrs else ()
    slice_offset = inputs.op.attrs["slice_offset"] \
        if "slice_offset" in inputs.op.attrs else ()
    l1_fusion_type = inputs.op.attrs["L1_fusion_type"].value \
    if "L1_fusion_type" in inputs.op.attrs else -1

    fmap_l1_addr_flag = inputs.op.attrs["L1_addr_flag"].value \
        if "L1_addr_flag" in inputs.op.attrs else "nothing"
    fmap_l1_valid_size = inputs.op.attrs["L1_valid_size"].value \
        if "L1_valid_size" in inputs.op.attrs else -1

    l1_fusion_enable_flag = get_L1_info("L1_fusion_enabled")
    if not l1_fusion_enable_flag:
        l1_fusion_type = -1

    valid_shape = _shape_to_list(valid_shape)
    slice_offset = _shape_to_list(slice_offset)

    l2_fusion_enable_flag = get_L1_info("L2_fusion_enabled")
    l1_fusion_enable_flag = get_L1_info("L1_fusion_enabled")

    if (not l2_fusion_enable_flag) and (not l1_fusion_enable_flag):
        input_memory_type = 0
        valid_shape = []
        slice_offset = []
        l1_fusion_type = -1

    if input_memory_type not in (0, 1, 2):
        err_man.raise_err_input_mem_type("conv2d", input_memory_type)
    if valid_shape and not slice_offset:
        err_man.raise_err_specific_user("conv2d", "if valid_shape exists "\
            + "slice_offset can not be []")

    fusion_para = {"input_memory_type": input_memory_type,
                   "output_memory_type": "fuse_flag",
                   "valid_shape": valid_shape, "slice_offset": slice_offset,
                   "l1_fusion_type": l1_fusion_type, \
                   "fmap_l1_addr_flag": fmap_l1_addr_flag, \
                   "fmap_l1_valid_size": fmap_l1_valid_size}

    return fusion_para


def _trans_stride(input_size, kernel, stride, pad, dlt):
    """
    transform stride

    Notice
    ------
    adapt stride value to hardware request

    Parameters
    ----------
    input_size: int
        feature map H/W size
    kernel: int
        kernel H/W size
    pad: 2D list of int
        pad on H/W side
    strides: int
        stride on H/W
    dlt: int
        dilation on H/W
    Returns
    -------
    new stride
    """
    return 1 if input_size + pad[0] + pad[1] == \
                    (kernel - 1)*dlt + 1 else stride


def _conv2d_fusion_para(inputs, outputs):
    """
    get L2 fusion para for conv2d
    """
    input_memory_type = inputs.get("addr_type") \
        if "addr_type" in inputs else 0
    output_memory_type = outputs.get("addr_type") \
        if "addr_type" in outputs else 0
    valid_shape = inputs.get("valid_shape") \
        if "valid_shape" in inputs else ()
    slice_offset = inputs.get("slice_offset") \
        if "slice_offset" in inputs else ()
    l1_fusion_type = inputs.get("L1_fusion_type") \
        if "L1_fusion_type" in inputs else -1

    fmap_l1_addr_flag = inputs.get("L1_addr_flag", "nothing")
    fmap_l1_valid_size = inputs.get("L1_valid_size", -1)

    l1_fusion_enable_flag = get_L1_info("L1_fusion_enabled")
    if not l1_fusion_enable_flag:
        l1_fusion_type = -1

    valid_shape = _shape_to_list(valid_shape)
    slice_offset = _shape_to_list(slice_offset)

    l2_fusion_enable_flag = get_L1_info("L2_fusion_enabled")

    if not l2_fusion_enable_flag and (not l1_fusion_enable_flag):
        input_memory_type = 0
        output_memory_type = 0
        valid_shape = []
        slice_offset = []
        l1_fusion_type = -1

    if input_memory_type not in (0, 1, 2):
        err_man.raise_err_input_mem_type("conv2d", input_memory_type)
    if output_memory_type not in (0, 1, 2):
        err_man.raise_err_output_mem_type("conv2d", output_memory_type)
    if valid_shape and not slice_offset:
        err_man.raise_err_specific_user("conv2d", "if valid_shape exists "\
           + "slice_offset can not be []")

    fusion_para = {"input_memory_type": input_memory_type,
                   "output_memory_type": output_memory_type,
                   "valid_shape": valid_shape, "slice_offset": slice_offset, \
                   "l1_fusion_type": l1_fusion_type, \
                   "fmap_l1_addr_flag": fmap_l1_addr_flag, \
                   "fmap_l1_valid_size": fmap_l1_valid_size}

    return fusion_para


def _shape_to_list(shape):
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