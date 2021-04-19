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
conv2d
"""
from __future__ import absolute_import
from te import tvm
from te.lang.cce.te_schedule import cce_build_code
from te.lang.cce.te_compute.conv_compute import conv
from te.platform import cce_conf
from te.platform.fusion_manager import fusion_manager
from te.utils.error_manager import error_manager_conv2d as err_man
from te.utils.check_para import check_input_type
from te.utils.check_para import NONE_TYPE
from te.utils.cce import auto_schedule
from te.utils.operate_shape import scalar2tensor_one
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_conv2d import calc_para_from_tensor
from impl.util.util_conv2d import conv_layer_cce_para_check
from impl.util.util_conv2d import conv_layer_cce_shape_calc
from impl.util.util_conv2d import calc_para_from_dict


def op_select_format(inputs, weights, bias, offset_w, outputs, strides,
                     pads, dilations, groups=1, data_format='NHWC',
                     offset_x=0, kernel_name="conv2d"):
    """
    select format dynamically
    """
    def _select_format(params):
        inputs = params[0]
        weights = params[1]
        c0_optim_flg = False
        shape_x = inputs.get("ori_shape")
        shape_x = scalar2tensor_one(shape_x)
        format_fm = inputs.get("ori_format")
        if format_fm == "NCHW":
            shape_fm = shape_x
        elif format_fm == "NHWC":
            shape_fm = [shape_x[0], shape_x[3], shape_x[1], shape_x[2]]
        else:
            err_man.raise_err_input_format_invalid("conv2d", "inputs", \
                ["NCHW", "NHWC"], format_fm)

        shape_w = weights.get("ori_shape")
        if (not isinstance(shape_w, (tuple, list))) or len(shape_w) != 4:
            err_man.raise_err_should_be_4d("conv2d", "weights")
        format_w = weights.get("ori_format")
        if format_w == "NCHW":
            shape_filter = shape_w
        elif format_w == "NHWC":
            shape_filter = [shape_w[0], shape_w[3], shape_w[1], shape_w[2]]
        elif format_w == "HWCN":
            shape_filter = [shape_w[3], shape_w[2], shape_w[0], shape_w[1]]
        else:
            err_man.raise_err_input_format_invalid("conv2d", "weights", \
                ["NCHW", "NHWC", "HWCN"], format_w)
        if shape_fm[1] <= 4:
            c0_optim_flg = True
        if (shape_filter[2] == 1) and (shape_filter[3] == 1):
            c0_optim_flg = False
        # format NC1HWC0_C04 can only be used at first conv layer
        # for those soc using NC1HWC0_C04, ensure is_first_layer == 1
        if inputs.get("is_first_layer") != 1 and \
            cce_conf.get_soc_spec("SOC_VERSION") \
            in ("Ascend710", "Ascend615", "Ascend610", "Hi3796CV300CS"):
            c0_optim_flg = False
        if c0_optim_flg:
            if cce_conf.get_soc_spec("SOC_VERSION") in \
            ("Ascend710", "Ascend615", "Ascend610", "Hi3796CV300CS"):
                input0 = gen_param(classify="input0", name="x",
                                   datatype="float16,float16,int8,int8",
                                   format="NC1HWC0_C04,NC1HWC0,"
                                          "NC1HWC0_C04,NC1HWC0")
            else:
                input0 = gen_param(classify="input0", name="x",
                                   datatype="float16,float16,int8,int8",
                                   format="NC1HWC0,NC1HWC0,"
                                          "NC1HWC0,NC1HWC0")
            input1 = gen_param(classify="input1", name="filter",
                               datatype="float16,float16,int8,int8",
                               format="FRACTAL_Z_C04,FRACTAL_Z,"
                                      "FRACTAL_Z_C04,FRACTAL_Z")
            input2 = gen_param(classify="input2", name="bias",
                               datatype="float16,float16,int32,int32",
                               format="ND,ND,ND,ND")
            input3 = gen_param(classify="input3", name="offset_w",
                               datatype="int8,int8,int8,int8",
                               format="ND,ND,ND,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,float16,int32,int32",
                                format="NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
        else:
            # only dynamic_hw or dynamic_batch is supported by dynamic conv2d
            if (shape_fm[0] == -1 and -1 not in shape_fm[1:]) or \
                (shape_fm[2] == -1 and shape_fm[3] == -1 and -1 not in shape_fm[:2]):
                input0 = gen_param(classify="input0", name="x",
                                   datatype="float16",
                                   format="NC1HWC0",
                                   unknownshape_format="NC1HWC0")
                input1 = gen_param(classify="input1", name="filter",
                                   datatype="float16",
                                   format="FRACTAL_Z",
                                   unknownshape_format="FRACTAL_Z")
                input2 = gen_param(classify="input2", name="bias",
                                   datatype="float16",
                                   format="ND")
                input3 = gen_param(classify="input3", name="offset_w",
                                   datatype="int8",
                                   format="ND")
                output0 = gen_param(classify="output0", name="y",
                                    datatype="float16",
                                    format="NC1HWC0",
                                    unknownshape_format="NC1HWC0")
            else:
                input0 = gen_param(classify="input0", name="x",
                                   datatype="float16,int8",
                                   format="NC1HWC0,NC1HWC0")
                input1 = gen_param(classify="input1", name="filter",
                                   datatype="float16,int8",
                                   format="FRACTAL_Z,FRACTAL_Z")
                input2 = gen_param(classify="input2", name="bias",
                                   datatype="float16,int32",
                                   format="ND,ND")
                input3 = gen_param(classify="input3", name="offset_w",
                                   datatype="int8,int8",
                                   format="ND,ND")
                output0 = gen_param(classify="output0", name="y",
                                    datatype="float16,int32",
                                    format="NC1HWC0,NC1HWC0")
        return [input0, input1, input2, input3, output0]

    params = [inputs, weights, bias, offset_w, outputs, strides,
              pads, dilations, groups, data_format, offset_x,
              kernel_name]
    param_list = _select_format(params)
    return get_dynamic_param_in_json(param_list)


@fusion_manager.register("conv2d")
def conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads,
                   dilations, groups=1, data_format='NCHW', offset_x=0,
                   kernel_name="conv2d"):
    """
    conv2d compute

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: tvm placeholder
        input 5hd feature map tensor
    weights: tvm placeholder
        input frac_z weight tensor
    outputs: tvm placeholder
        output tensor, dtype must be assigned
    bias: tvm placeholder or None
        input 1d bias tensor
    offset_w: tvm placeholder or None
        offset_w bias tensor
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
    para_dict, optim_dict = calc_para_from_tensor(
        inputs, weights, bias, offset_w, strides, \
        pads, dilations, offset_x, kernel_name, data_format)

    res = conv(inputs, weights, para_dict, optim_dict)

    return res


@check_input_type(dict, dict, (dict, NONE_TYPE), (dict, NONE_TYPE), dict,
                       (tuple, list), (tuple, list), (tuple, list), int,
                       str, int, str)
def conv2d(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
           groups=1, data_format='NCHW', offset_x=0, kernel_name="conv2d"):
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
    in_dtype = inputs.get("dtype")
    w_dtype = weights.get("dtype")
    res_dtype = outputs.get("dtype")

    shape_fm, shape_filter, padh, padw, strideh, stridew, \
    dlt_h, dlt_w, optim_dict, fusion_para = calc_para_from_dict(
        inputs, weights, strides, pads, dilations, outputs, data_format)

    use_bias = True
    if bias is None:
        use_bias = False
    use_offset_w = True
    if offset_w is None:
        use_offset_w = False


    _conv_layer_cce(shape_fm, shape_filter, in_dtype, w_dtype, res_dtype,
                   padh, padw, strideh, stridew, dlt_h, dlt_w,
                   offset_x, groups=groups, offset_w=use_offset_w,
                   bias=use_bias, optim_dict=optim_dict,
                   fusion_para=fusion_para,
                   kernel_name=kernel_name, need_build=True,
                   need_print=False)


@check_input_type((list, tuple), (list, tuple), str, str, str, \
    (list, int), (list, int), int, int, (int, NONE_TYPE), (int, NONE_TYPE), \
    int, int, str, \
    bool, bool,
    dict, (dict, NONE_TYPE), str, \
    bool, bool)
def _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                   padh, padw, strideh, stridew, dilateh=1, dilatew=1,
                   offset_x=0, groups=1, offset_w_dtype='int32',
                   offset_w=False, bias=False,
                   optim_dict=None, fusion_para=None, kernel_name="cce_conv",
                   need_build=False, need_print=False):
    """

    Parameters
    ----------
    shape_in: shape of feature map

    shape_w: shape of weight

    in_dtype: the feature map data type

    w_dtype: the weight data type

    res_dtype: the result data type

    padh: H direction padding

    padw: W direction padding

    strideh: H direction stride

    stridew: W direction stride

    dilateh: H direction spacing between kernel

    dilatew: W direction spacing between kernel

    offset_x: the offset for fmap

    offset_w_dtype: weight offset data type, default 'int32'

    offset_w: the tag for offset_w or not

    bias: the tag for bias or not

    fusion_para: the config for L2 Fusion
                input_memory_type: feature map from L2/GM, 0 for GM, 2 for L2
                output_memory_type: calculation results are outputs to L2/GM
                valid_shape: valid shape in L1 buffer, NC1HWC0
                slice_offset: the offset of each dimension
                              between valid shape and shape in

    kernel_name: cce kernel name, default value is "cce_conv"

    need_build: if need to build CCEC kernel, default value is False

    need_print: if need to print the ir, default value is False

    Returns
    -------
    wrapped_tensor

    """
    # for pylint, otherwise "Dangerous default value [] as argument"
    if optim_dict is None:
        optim_dict = {"c0_optim_flg": False}

    if fusion_para is None:
        fusion_para = {"input_memory_type": 0, "output_memory_type": 0,
                       "valid_shape": (), "slice_offset": (), \
                       "l1_fusion_type": -1, \
                       "fmap_l1_addr_flag": 0, \
                       "fmap_l1_valid_size": -1}

    in_dtype = in_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()
    offset_w_dtype = offset_w_dtype.lower()

    mad_dtype = 'float32'
    if w_dtype == 'int8':
        mad_dtype = 'int32'

    shape_in = list(shape_in)
    shape_w = list(shape_w)

    shape_in, shape_w = \
            conv_layer_cce_para_check(shape_in, shape_w, padh, padw,
                                      strideh, stridew, in_dtype, w_dtype,
                                      res_dtype, offset_w_dtype, bias,
                                      kernel_name, dilateh, dilatew,
                                      optim_dict, fusion_para)

    out_channel, in_channel_weight, filter_h, filter_w = shape_w

    fmap_shape_nc1hwc0, filter_shape_frac_z = conv_layer_cce_shape_calc(
        shape_in, shape_w, in_dtype, w_dtype, optim_dict)

    tensor_list = []
    with tvm.target.cce():
        data = tvm.placeholder(
            fmap_shape_nc1hwc0, name='Fmap', dtype=in_dtype)
        tensor_list.append(data)
        weight = tvm.placeholder(
            filter_shape_frac_z, name='Filter', dtype=w_dtype)
        tensor_list.append(weight)
        bias_tensor = None
        offset_w_tensor = None

        if bias:
            bias_tensor = tvm.placeholder((out_channel,), name='bias_tensor',
                                          dtype=res_dtype)
            tensor_list.append(bias_tensor)
        conv_res = conv(
            data, weight, para_dict={"bias_tensor": bias_tensor,
                           "offset_w_tensor": offset_w_tensor,
                           "pad_h": padh, "pad_w": padw,
                           "stride_h": strideh, "stride_w": stridew,
                           "dilate_h": dilateh, "dilate_w": dilatew,
                           "filter_h": filter_h, "filter_w": filter_w,
                           "offset_x": offset_x, "groups": groups,
                           "res_dtype": res_dtype,
                           "fusion_para": fusion_para,
                           "kernel_name": kernel_name},
            optim_dict=optim_dict,
            dsl_flag=False)
        tensor_list.append(conv_res)
        sch = auto_schedule(conv_res)

    config = {
        "print_ir": need_print,
        "need_build": need_build,
        "name": kernel_name,
        "tensor_list": tensor_list
    }

    cce_build_code(sch, config)