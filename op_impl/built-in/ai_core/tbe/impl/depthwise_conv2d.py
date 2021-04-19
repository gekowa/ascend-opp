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
depthwise_conv2d
"""
import te.lang.cce
from te import platform as tbe_platform
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils
from te.utils.error_manager import error_manager_conv2d as err_man
from te.utils.error_manager import error_manager_util as err_mana
from topi import generic
from te.platform.cce_policy import get_L1_info

# shape's dim of input and output must be 4
FEATURE_MAP_DIM = 5

# shape's dim of filter must be 4
FILTER_DIM = 6

# shape's dim of strides must be 2
STRIDES_DIM = 4

NONETYPE = type(None)


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


def depthwise_conv2d_fusion_para(inputs, outputs):
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
        err_man.raise_err_input_mem_type("depthwise_conv2d", input_memory_type)
    if int(output_memory_type) not in (0, 1, 2):
        err_man.raise_err_output_mem_type("depthwise_conv2d",
                                          output_memory_type)
    if valid_shape and not slice_offset:
        err_man.raise_err_specific_user(
            "depthwise_conv2d",
            "if valid_shape exists slice_offset can not be []")

    fusion_para = {
        "input_memory_type": input_memory_type,
        "output_memory_type": output_memory_type,
        "valid_shape": valid_shape,
        "slice_offset": slice_offset,
        "l1_fusion_type": l1_fusion_type,
        "fmap_l1_addr_flag": fmap_l1_addr_flag,
        "fmap_l1_valid_size": fmap_l1_valid_size
    }

    return fusion_para


# pylint: disable=locally-disabled, too-many-locals, too-many-arguments,
# pylint: disable=unused-argument
# pylint: disable=locally-disabled, bad-continuation, import-error
# pylint: disable=too-many-statements, redefined-builtin, invalid-name
@fusion_manager.register("depthwise_conv2d")
def depthwise_compute(fmap, filter, bias, offset_w, out,
                      strides, dilations, pads, \
                      data_format='NHWC', offset_x=0, dsl_flag=True, \
                      kernel_name="depthwise_conv2d"):
    """
    algorithm: depthwise conv2d compute
    calculating  depthwise compute
    Parameters
    ----------
    fmap : a tensor of featureMap
    filter : a tensor of filter
    bias : a tensor of bias
    offset_w : a tensor of filter offset
    out : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.
    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]
    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]
    pads : padding added to each dimension of the input
    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]
    offset_x : offset of the input
    Returns
    -------
    None
    """
    out_dtype = out.get("dtype")
    l1_fusion_para = depthwise_conv2d_fusion_para(fmap, out)
    DIM_H, DIM_W = 2, 3
    if data_format == 'NHWC':
        DIM_H, DIM_W = 1, 2

    strides_2d = strides[DIM_H], strides[DIM_W]
    dilations_2d = dilations[DIM_H], dilations[DIM_W]

    out = te.lang.cce.te_compute.depthwise_conv2d_compute(
        fmap, filter, out_dtype.lower(), strides_2d, pads, dilations_2d, {
            "bias_tensor": bias,
            "dsl_flag": dsl_flag,
            "offset_x": offset_x
        }, l1_fusion_para, kernel_name)
    return out


@op_utils.check_op_params(
    op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT, op_utils.OPTION_INPUT,
    op_utils.OPTION_INPUT, op_utils.REQUIRED_OUTPUT,
    op_utils.REQUIRED_ATTR_LIST_INT, op_utils.OPTION_ATTR_LIST_INT,
    op_utils.REQUIRED_ATTR_LIST_INT, op_utils.OPTION_ATTR_STR,
    op_utils.REQUIRED_ATTR_INT, op_utils.KERNEL_NAME)
def depthwise_conv2d(
        x,
        filter,
        bias,
        offset_w,
        y,
        strides,
        dilations=(1, 1, 1, 1),
        pads=(0, 0, 0, 0),
        data_format='NHWC',
        offset_x=0,
        kernel_name="depthwise_conv2d",
):
    """
    algorithm: depthwise conv2d

    calculating  depthwise convolution

    Parameters
    ----------
    x : a dict of featureMap
        {"shape", "dtype", "format"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    filter : a dict of filter
        {"shape", "dtype"}
        shape of filter tensor [C1, H, W, K, Co, C0],
        K is depthwise_multiplier, support int.

    bias : a dict of bias
        {"shape", "dtype"}
        shape of bias tensor [C1*C0,]
        support int8.

    offset_w : a dict of filter offset
        {"shape", "dtype"}
        shape of offset tensor [C1, H, W, K, Co, C0]
        support float16.

    y : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]

    pads : padding added to each dimension of the input

    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    offset_x : offset of the input

    kernel_name : str
       cce kernel name

    Returns
    -------
    None

    """
    shape_w = filter.get("shape")
    shape_in = x.get("shape")
    output_dtype = y.get("dtype")
    in_dtype = x.get("dtype")
    w_dtype = filter.get("dtype")
    fmap_data_format = x.get("format")

    op_utils.check_dtype(in_dtype, ('float16', 'int8'), param_name="x")
    op_utils.check_dtype(w_dtype, ('float16', 'int8'), param_name="filter")
    op_utils.check_dtype(output_dtype, ('float16', 'int32'), param_name="y")

    op_utils.check_shape(shape_in,
                         min_rank=FEATURE_MAP_DIM,
                         max_rank=FEATURE_MAP_DIM,
                         param_name="x")
    op_utils.check_shape(shape_w,
                         min_rank=FILTER_DIM,
                         max_rank=FILTER_DIM,
                         param_name="filter")
    op_utils.check_shape(strides,
                         min_rank=STRIDES_DIM,
                         max_rank=STRIDES_DIM,
                         param_name="strides")

    if fmap_data_format != "NC1HWC0":
        dict_args = {
            'errCode': 'E60008',
            'op_name': 'depthwise_conv2d',
            'param_name': 'featuremap',
            'expected_format_list': '[{}]'.format('NC1HWC0'),
            'format': fmap_data_format
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    def _check_shape(fmap_shape, filter_shape):
        """check input shape"""
        _, in_c1, _, _, _ = fmap_shape
        filter_c1, _, _, filter_k, _, _ = filter_shape

        # check feature map API feature map  shape is 5hd
        # The shape of feature map and filter must be 5HD
        if len(fmap_shape) != FEATURE_MAP_DIM:
            dict_args = {
                'errCode': 'E60008',
                'op_name': 'depthwise_conv2d',
                'param_name': 'featuremap',
                'expected_format_list': '[{}]'.format('NC1HWC0'),
                'format': fmap_data_format
            }
            raise RuntimeError(dict_args,
                               err_mana.get_error_message(dict_args))

        # check feature map shape of c, equal filter of c
        if in_c1 != filter_c1:
            dict_args = {
                'errCode': 'E60002',
                'op_name': 'depthwise_conv2d',
                'attr_name': 'channel',
                'param1_name': 'fmap',
                'param2_name': 'filter',
                'param1_value': str(in_c1),
                'param2_value': str(filter_c1)
            }
            raise RuntimeError(dict_args,
                               err_mana.get_error_message(dict_args))

        # check multiplier equal 1
        if filter_k != 1:
            dict_args = {
                'errCode': 'E60000',
                'op_name': 'depthwise_conv2d',
                'param_name': 'filter_k',
                'expected_value': '1',
                'input_value': str(filter_k)
            }
            raise RuntimeError(dict_args,
                               err_mana.get_error_message(dict_args))

    # fmap shape reshape, c ceil 16, 6d shape;
    # c must be 16x, if data not 16x, framework reshape c 16x
    in_n, in_c1, in_h, in_w, in_c0 = shape_in
    fmap_shape_5d = in_n, in_c1, in_h, in_w, in_c0
    shape_w_5d = shape_w[0], shape_w[1], shape_w[2], shape_w[4], shape_w[5]

    #filter shape: C1HWNCoC0
    filter_c1, filter_h, filter_w, _, _, _ = shape_w

    if data_format != 'NCHW' and data_format != 'NHWC':
        dict_args = {
            'errCode': 'E50002',
            'op_name': 'depthwise_conv2d',
            'param': 'featuremap',
            'expected_format_list': '[{}, {}]'.format('NCHW', 'NHWC'),
            'format': data_format
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    _check_shape(shape_in, shape_w)

    DIM_N, DIM_C, DIM_H, DIM_W = 0, 1, 2, 3  # NCHW
    if data_format == 'NHWC':
        DIM_N, DIM_H, DIM_W, DIM_C = 0, 1, 2, 3

    # check strides is list, strides[0] ==shape_in[1]
    # strides list, and h w value equal
    if not isinstance(strides, (list, tuple)) and len(strides) == 4:
        dict_args = {
            'errCode': 'E60107',
            'op_name': 'depthwise_conv2d',
            'param_name': 'strides'
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    if strides[DIM_N] != 1 or strides[DIM_C] != 1:
        err_man.raise_err_specific_user("depthwise_conv2d",\
            "stride only support 1 in N axis and C axis.")
    if strides[DIM_H] != strides[DIM_W]:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d',
            'attr_name': 'stride value',
            'param1_name': 'strides[DIM_H]',
            'param2_name': 'strides[DIM_W]',
            'param1_value': str(strides[DIM_H]),
            'param2_value': str(strides[DIM_W])
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))
    if dilations[DIM_N] != 1 or dilations[DIM_C] != 1:
        dict_args = {
            'errCode': 'E60023',
            'op_name': 'depthwise_conv2d',
            'dilation_n': str(dilations[DIM_N]),
            'dilation_c': str(dilations[DIM_C])
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))
    if dilations[DIM_H] != dilations[DIM_W]:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d',
            'attr_name': 'dilations value',
            'param1_name': 'dilations[DIM_H]',
            'param2_name': 'dilations[DIM_W]',
            'param1_value': str(dilations[DIM_H]),
            'param2_value': str(dilations[DIM_W])
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    # check pad parameter
    if len(pads) != 4:
        dict_args = {
            'errCode': 'E50001',
            'param': 'pads',
            'op_name': 'depthwise_conv2d',
            'expected_length': "4",
            'length': str(len(pads))
        }
        raise RuntimeError(dict_args, err_mana.get_error_message(dict_args))

    strides_2d = strides[DIM_H], strides[DIM_W]
    dilations_2d = dilations[DIM_H], dilations[DIM_W]
    bias_tensor = None
    if bias is not None and bias != {}:
        bias_tensor = tvm.placeholder((filter_c1 * 16, ),
                                      name='bias_tensor',
                                      dtype=output_dtype.lower())
    fmap_placeholder = tvm.placeholder(fmap_shape_5d,
                                       dtype=in_dtype.lower(),
                                       name='fmap')
    filter_placeholder = tvm.placeholder(shape_w_5d,
                                         dtype=w_dtype.lower(),
                                         name='filter')
    dsl_flag = False
    out = te.lang.cce.te_compute.depthwise_conv2d_compute(
        fmap_placeholder, filter_placeholder, output_dtype.lower(), strides_2d,
        pads, dilations_2d, {
            "bias_tensor": bias_tensor,
            "dsl_flag": dsl_flag,
            "offset_x": offset_x
        }, None, kernel_name)

    tensor_list = [fmap_placeholder, filter_placeholder, out]
    if bias_tensor is not None:
        tensor_list = [fmap_placeholder, filter_placeholder, bias_tensor, out]

    with tvm.target.cce():
        sch = generic.auto_schedule(out)

    with tbe_platform.build_config:
        tvm.build(sch, tensor_list, "cce", name=kernel_name)
