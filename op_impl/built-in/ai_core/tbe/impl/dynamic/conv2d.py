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
dynamic conv2d
"""
from __future__ import absolute_import

from collections import OrderedDict

import te.lang.cce
import te.lang.dynamic
from te import tvm
from te.platform import CUBE_MKN
from te.platform import cce_conf
from te.platform import operation
from te.platform.fusion_manager import get_fusion_build_cfg
from topi import generic
from topi.cce import util
from impl.util import fusion_util
from te.utils.error_manager import error_manager_conv2d as err_man

PAD_SHAPE_DIM = 2
NONETYPE = type(None)


# pylint: disable=too-many-locals,invalid-name,unused-argument,too-many-arguments
def dynamic_shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """

    tmp = []

    for i in shape:
        if not isinstance(i, tvm.expr.IntImm) and len(i) > 1:
            inner_tmp = []
            for j in i:
                if isinstance(j, tvm.expr.IntImm):
                    inner_tmp.append(int(j))
                else:
                    inner_tmp.append(j)
            tmp.append(inner_tmp)
        else:
            if isinstance(i, tvm.expr.IntImm):
                tmp.append(int(i))
            else:
                tmp.append(i)

    return tmp


def pos_from_format(ele_format):
    """
    get value from ele_format
    """

    pos_n = ele_format.find('N')
    pos_c = ele_format.find('C')
    pos_h = ele_format.find('H')
    pos_w = ele_format.find('W')
    return pos_n, pos_c, pos_h, pos_w


def set_default_para():
    """
    set default parameter value
    """

    optim_dict = {"c0_optim_flg": False}
    fusion_para = {"input_memory_type": 0, "output_memory_type": 0,
                   "valid_shape": (), "slice_offset": (),
                   "l1_fusion_type": -1}
    return optim_dict, fusion_para


def config_dynamic_mode(in_shape, w_shape):
    """
    config dynamic mode
    """

    dynamic_mode = None
    if in_shape[2] == in_shape[3] == -1 and in_shape[0] != -1 \
            and in_shape[1] != -1 and -1 not in w_shape:
        dynamic_mode = "dynamic_hw"
    elif in_shape[0] == -1 and in_shape[1] != -1 and in_shape[2] != -1 \
            and in_shape[3] != -1 and -1 not in w_shape:
        dynamic_mode = "dynamic_batch"
    return dynamic_mode


def check_and_config_para(inputs, weights, bias, offset_w, outputs, strides,
                          pads, dilations, data_format, offset_x, kernel_name):
    """
    check and config dynamic mode
    """

    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    if soc_version in ("Hi3796CV300ES", "Hi3796CV300CS"):
        err_man.raise_err_specific_user("conv2d", \
            "Hi3796CV300ES and Hi3796CV300CS don't support dynamic shape")

    in_shape = list(inputs.get("ori_shape"))
    w_shape = list(weights.get("ori_shape"))
    in_dtype = inputs.get("dtype")
    w_dtype = weights.get("dtype")
    all_fmt = ["NCHW", "NHWC"]
    in_format = inputs.get("ori_format")
    w_format = weights.get("ori_format")
    fmap_range = inputs.get("range")
    w_range = weights.get("range")
    optim_dict, fusion_para = set_default_para()

    util.check_kernel_name(kernel_name)
    # util.check_dtype_rule(offset_w.dtype, ['int32'])
    util.check_dtype_rule(in_dtype, ['float16'])
    util.check_dtype_rule(w_dtype, ['float16'])
    util.check_dtype_rule(outputs.get("dtype"), ['float16'])

    if (not isinstance(in_shape, (tuple, list))) or len(in_shape) != 4:
        err_man.raise_err_should_be_4d("conv2d", "in_shape")

    if (not isinstance(w_shape, (tuple, list))) or len(w_shape) != 4:
        err_man.raise_err_should_be_4d("conv2d", "weights")

    if len(strides) != 4:
        err_man.raise_err_should_be_4d("conv2d", "strides")
    if len(dilations) != 4:
        err_man.raise_err_should_be_4d("conv2d", "dilations")
    if len(pads) != 4:
        err_man.raise_err_should_be_4d("conv2d", "pads")
    if data_format not in all_fmt:
        err_man.raise_err_input_format_invalid("conv2d", \
            "input", ["NCHW", "NHWC"], data_format)
    if in_format != data_format:
        err_man.raise_err_specific_user("conv2d", "in_format != data_format")
    if w_format not in all_fmt and w_format != "HWCN":
        err_man.raise_err_input_format_invalid("conv2d", \
            "input", ["NCHW", "NHWC"], data_format)

    pos_n, pos_c, pos_h, pos_w = pos_from_format(data_format)
    w_pos_n, w_pos_c, w_pos_h, w_pos_w = pos_from_format(w_format)
    in_shape = [in_shape[pos_n], in_shape[pos_c], in_shape[pos_h], in_shape[pos_w]]
    w_shape = [w_shape[w_pos_n], w_shape[w_pos_c], w_shape[w_pos_h], w_shape[w_pos_w]]
    strides = [strides[pos_n], strides[pos_c], strides[pos_h], strides[pos_w]]
    dilations = [dilations[pos_n], dilations[pos_c], dilations[pos_h], dilations[pos_w]]
    if len(fmap_range) == 5:
        fmap_range = [fmap_range[pos_n], (in_shape[pos_c], in_shape[pos_c]),
                      fmap_range[2], fmap_range[3]]
    else:
        fmap_range = [fmap_range[pos_n], fmap_range[pos_c],
                      fmap_range[pos_h], fmap_range[pos_w]]

    dynamic_mode = config_dynamic_mode(in_shape, w_shape)
    if dynamic_mode not in ("dynamic_hw", "dynamic_batch"):
        err_man.raise_err_specific_user("conv2d", \
            "Only dynamic_hw and dynamic_batch are supported currently")
    dynamic_para = {
        "dynamic_mode": dynamic_mode,
        "fmap_range": fmap_range
    }

    in_shape, w_shape = \
        te.lang.cce.check_conv_shape(in_shape, w_shape, *pads, *strides[2:],
                                     in_dtype, w_dtype, fusion_para,
                                     optim_dict, *dilations[2:], dynamic_para)

    return in_shape, w_shape, pads, strides, dilations, in_dtype, w_dtype, \
        optim_dict, fmap_range, w_range, dynamic_mode


def calc_shape(shape_in, shape_w, in_dtype, w_dtype, optim_dict):
    """
    calculate shape
    """

    batch_size, in_channel, feature_map_h, feature_map_w = shape_in
    block_size_k = CUBE_MKN[in_dtype]['mac'][1]
    fmap_shape_nc1hwc0 = [batch_size,
                          (in_channel + block_size_k - 1) // block_size_k,
                          feature_map_h, feature_map_w, block_size_k]

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


@te.op.register_fusion_compute("Conv2D")
def conv2d_fusion_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                          groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                          dsl_flag=True):
    fusion_util.check_fusion_input([inputs])
    fusion_util.check_fusion_input([weights])

    # set fusion build config
    build_cfg = get_fusion_build_cfg()
    build_cfg['constant_realize_extent_in_infer_bound'] = False

    return conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                          groups, data_format, offset_x, kernel_name, dsl_flag)


def conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                   groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                   dsl_flag=True):

    """
    conv2d compute

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset for fmap

    Returns
    -------
    tvm compute
    """

    res_dtype = "float16"
    mad_dtype = 'float32'
    if isinstance(inputs, dict) and isinstance(weights, dict):
        shape_fm, shape_filter, pads, strides, dilations, in_dtype, w_dtype, \
        optim_dict, fmap_range, _, dynamic_mode = check_and_config_para(
            inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
            data_format, offset_x, kernel_name)

        fmap_shape_nc1hwc0, filter_shape_frac_z = calc_shape(
            shape_fm, shape_filter, in_dtype, w_dtype, optim_dict)
        if dynamic_mode == "dynamic_hw":
            fmap_shape_nc1hwc0[2] = operation.var("fmap_h", tuple(fmap_range[2]))
            fmap_shape_nc1hwc0[3] = operation.var("fmap_w", tuple(fmap_range[3]))
            operation.add_exclude_bound_var(fmap_shape_nc1hwc0[2])
            operation.add_exclude_bound_var(fmap_shape_nc1hwc0[3])
        elif dynamic_mode == "dynamic_batch":
            fmap_shape_nc1hwc0[0] = operation.var("batch_n", tuple(fmap_range[0]))
            operation.add_exclude_bound_var(fmap_shape_nc1hwc0[0])

        fmap = tvm.placeholder(fmap_shape_nc1hwc0, name='Fmap', dtype=in_dtype)
        weight = tvm.placeholder(filter_shape_frac_z, name='Filter', dtype=w_dtype)

        fmap_shape_ori = shape_fm
        tgt_batch, tgt_cin, tgt_h, tgt_w = fmap_shape_ori
    else:
        err_man.raise_err_specific_user("conv2d",\
            "In op[inputs], [weights] must be list or tuple")
    filter_shape_ori = shape_filter
    filter_h, filter_w = filter_shape_ori[2:]
    tgt_cout, _, _, _ = filter_shape_ori
    pad_t, pad_b, pad_l, pad_r = pads
    stride_h, stride_w = strides[2:]
    dilate_h, dilate_w = dilations[2:]
    optim_dict, fusion_para = set_default_para()
    shape_ori = None

    _, _, _ = CUBE_MKN[in_dtype]['mac']

    var_map = OrderedDict()
    if dynamic_mode == "dynamic_hw":
        var_map['fmap_h'] = fmap_shape_nc1hwc0[2]
        var_map['fmap_w'] = fmap_shape_nc1hwc0[3]
        var_map['ho'] = operation.var('ho', tuple(fmap_range[2]))
        var_map['wo'] = operation.var('wo', tuple(fmap_range[3]))
        operation.add_exclude_bound_var(var_map['ho'])
        operation.add_exclude_bound_var(var_map['wo'])
    elif dynamic_mode == "dynamic_batch":
        var_map['batch_n'] = fmap_shape_nc1hwc0[0]
    else:
        err_man.raise_err_specific_user("conv2d",\
            "dynamic_only support dynamic_hw or dynamic_batch")

    op_res = te.lang.cce.conv(fmap, weight,
                              {"bias_tensor": bias,
                               "offset_w_tensor": offset_w,
                               "pad_h": [pad_t, pad_b], "pad_w": [pad_l, pad_r],
                               "stride_h": stride_h, "stride_w": stride_w,
                               "dilate_h": dilate_h, "dilate_w": dilate_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               "offset_x": offset_x,
                               "res_dtype": res_dtype, "mad_dtype": mad_dtype,
                               "fusion_para": fusion_para,
                               "var_map": var_map, "shape_ori": shape_ori,
                               "dynamic_mode": dynamic_mode,
                               "fmap_range": fmap_range,
                               "dynamic_tgt": [tgt_batch, tgt_cin, tgt_h, tgt_w, tgt_cout],
                               "kernel_name": kernel_name},
                              optim_dict=optim_dict,
                              dsl_flag=dsl_flag)

    return {"op_placeholder": (fmap, weight), "op_res": [op_res]}


@te.op.register_operator("Conv2D")
@util.check_input_type(dict, dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                       (tuple, list), (tuple, list), (tuple, list),
                       int, str, int, str, str)
def conv2d(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
           groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d"):
    """
    algorithm: conv2d

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset of fmap
    kernel_name: str
        kernel name, default value is "conv2d"

    Returns
    -------
    None
    """

    bias = None
    offset_w = None

    with te.op.compute():
        res = conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                             groups, data_format, offset_x, kernel_name, dsl_flag=False)

    with tvm.target.cce():
        sch = generic.auto_schedule(res.get("op_res"))
    tensor_list = [res.get("op_placeholder")[0], res.get("op_placeholder")[1], res.get("op_res")[0]]

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False}
    }

    te.lang.dynamic.build(sch, config)
