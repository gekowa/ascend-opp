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
max_pool
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


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


# pylint: disable=unnecessary-lambda
@fusion_manager.register("max_pool")
def max_pool_fuse_compute(input_data, output_data, ksize, strides, padding=None,
                          data_format="NC1HWC0", kernel_name="max_pool_fuse"):
    """
    Performs max pooling on the input.

    Parameters
    ----------
    input_data: TVM tensor
        A `Tensor`. Must be one of the following types: `float16`.
        4-D input to pool over.
    output_data: dict
        dict of output_data, include keys(shape and dtype).
    ksize: list or tuple
        A list of `ints` that has length 2.
        The size of the window for H, W dimension of the input tensor.
    strides: list or tuple
        A list of `ints` that has length 2.
        The stride of the sliding window for H, W .
    padding: str
         A `string` from: `"SAME", "VALID"`.The type of padding.
    data_format: str
        A `string` from: "NC1HWC0"
    kernel_name: str
        kernel name, default value is 'max_pool'

    Returns:
    -------
    res: TVM tensor
        output tensor. Has the same type as `input_data`.
    """
    if data_format in ("NHWC",):
        window_h, window_w = ksize[1], ksize[2]
        stride_h, stride_w = strides[1], strides[2]
    elif data_format in ("NC1HWC0", "NCHW"):
        window_h, window_w = ksize[2], ksize[3]
        stride_h, stride_w = strides[2], strides[3]
    else:
        raise RuntimeError("don't support this format")

    conv_pooling_flag = False
    temp_tensor = input_data
    while temp_tensor.op.input_tensors:
        if temp_tensor.op.tag == "convolution_C":
            conv_pooling_flag = True
            break
        temp_tensor = temp_tensor.op.input_tensors[0]
    if conv_pooling_flag:
        res = te.lang.cce.max_pool_compute(input_data, (window_h, window_w),
                                           (stride_h, stride_w), padding,
                                           data_mode=1)
    else:
        # l1 fusion and l2 fusion
        l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value \
            if "L1_fusion_type" in input_data.op.attrs else -1

        # l1 fusion params assign
        fusion_params = get_fusion_params(input_data, output_data, True)
        in_select_read_flag = fusion_params.get("in_select_read_flag")
        in_valid_shape = fusion_params.get("in_valid_shape")
        in_slice_offset = fusion_params.get("in_slice_offset")

        if in_select_read_flag:
            select_tensor_in = \
                tvm.compute(in_valid_shape,
                            lambda n, c1, h, w, c0:
                            input_data(n, c1, h + in_slice_offset[2],
                                       w, c0),
                            name="tensor_read_select",
                            attrs=input_data.op.attrs)
            res = te.lang.cce.pooling2d(select_tensor_in,
                                        (window_h, window_w),
                                        (stride_h, stride_w),
                                        "MAX", padding, pad=(0, 0, 0, 0),
                                        fusion_params=fusion_params)
        elif l1_fusion_type == 1:
            input_data.op.attrs["addr_type"].value = 1
            in_l1_flag = True
            fusion_params["in_l1_flag"] = in_l1_flag

            l1_width_fusion_in = \
                tvm.compute(input_data.shape,
                            lambda n, c1, h, w, c0:
                            input_data(n, c1, h, w, c0),
                            name="l1_width_fusion_tensor_in",
                            attrs=input_data.op.attrs)
            res = te.lang.cce.pooling2d(l1_width_fusion_in,
                                        (window_h, window_w),
                                        (stride_h, stride_w),
                                        "MAX", padding, pad=(0, 0, 0, 0),
                                        fusion_params=fusion_params)
        else:
            res = te.lang.cce.pooling2d(input_data, (window_h, window_w),
                                        (stride_h, stride_w),
                                        "MAX", padding, pad=(0, 0, 0, 0),
                                        fusion_params=fusion_params)
    return res


# pylint: disable=unnecessary-lambda,too-many-locals
def max_pool_compute(input_data, output_data, ksize, strides, padding,
                     data_format="NC1HWC0", kernel_name="max_pool"):
    """
    Performs max pooling on the input.

    Parameters
    ----------
    input_data: TVM tensor
        A `Tensor`. Must be one of the following types: `float16`, `uint8`, `int8`.
        5-D input to pool over.
    output_data: dict
        dict of output_data, include keys(shape and dtype).
    ksize: list or tuple
        A list of `ints` that has length 4.
        The size of the window for each dimension of the input tensor.
    strides: list or tuple
        A list of `ints` that has length 4.
        The stride of the sliding window for each dimension of the input tensor.
    padding: str
         A `string` from: `"SAME", "VALID"`.The type of padding algorithm to use.
    data_format: str
        A `string` from: `"NC1HWC0", "NHWC", "NCHW"`.
    kernel_name: str
        kernel name, default value is 'max_pool'

    Returns:
    -------
    res: TVM tensor
        output tensor. Has the same type as `input_data`.
    """
    if data_format in ("NHWC",):
        window_h, window_w = ksize[1], ksize[2]
        stride_h, stride_w = strides[1], strides[2]
    else:
        window_h, window_w = ksize[2], ksize[3]
        stride_h, stride_w = strides[2], strides[3]

    # l1 fusion and l2 fusion
    l1_fusion_type = input_data.op.attrs["L1_fusion_type"].value \
        if "L1_fusion_type" in input_data.op.attrs else -1
    fusion_params = get_fusion_params(input_data, output_data, False)
    in_select_read_flag = fusion_params.get("in_select_read_flag")
    in_valid_shape = fusion_params.get("in_valid_shape")
    in_slice_offset = fusion_params.get("in_slice_offset")

    if in_select_read_flag:
        select_tensor_in = tvm.compute(in_valid_shape,
                                       lambda n, c1, h, w, c0:
                                       input_data(n, c1, h +
                                                  in_slice_offset[2], w, c0),
                                       name="tensor_read_select",
                                       attrs=input_data.op.attrs)
        res = te.lang.cce.pooling2d(select_tensor_in, (window_h, window_w),
                                    (stride_h, stride_w),
                                    "MAX", padding, pad=(0, 0, 0, 0),
                                    fusion_params=fusion_params)
    elif l1_fusion_type == 1:
        input_data.op.attrs["addr_type"].value = 1
        in_l1_flag = True
        fusion_params["in_l1_flag"] = in_l1_flag

        l1_width_fusion_in = tvm.compute(input_data.shape,
                                         lambda n, c1, h, w, c0:
                                         input_data(n, c1, h, w, c0),
                                         name="l1_width_fusion_tensor_in",
                                         attrs=input_data.op.attrs)
        res = te.lang.cce.pooling2d(l1_width_fusion_in, (window_h, window_w),
                                    (stride_h, stride_w), "MAX", padding,
                                    pad=(0, 0, 0, 0),
                                    fusion_params=fusion_params)
    else:
        res = te.lang.cce.pooling2d(input_data, (window_h, window_w),
                                    (stride_h, stride_w),
                                    "MAX", padding, pad=(0, 0, 0, 0),
                                    fusion_params=fusion_params)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_LIST_INT,
                 REQUIRED_ATTR_STR, OPTION_ATTR_STR, KERNEL_NAME)
# pylint: disable=too-many-locals
def max_pool(input_data, output_data, ksize, strides, padding,
             data_format="NC1HWC0", kernel_name="max_pool"):
    """
    Performs max pooling on the input.

    Parameters
    ----------
    input_data: dict
        dict of input_data, include keys(shape and dtype).
    output_data: dict
        dict of output_data, include keys(shape and dtype).
    ksize: list or tuple
        A list of `ints` that has length 4.
        The size of the window for each dimension of the input tensor.
    strides: list or tuple
        A list of `ints` that has length 4.
        The stride of the sliding window for each dimension of the input tensor.
    padding: str
        A `string` from: `"SAME", "VALID"`.The type of padding algorithm to use.
    data_format: str
        A `string` from: `"NC1HWC0", "NHWC", "NCHW"`.
    kernel_name: str
        kernel name, default value is 'max_pool'

    Returns:
    -------
    None
    """
    shape_input = input_data.get("shape")
    dtype_input = input_data.get("dtype").lower()

    check_shape(shape_input, param_name="input_data")

    check_list = ("float16", "int8", "uint8")
    check_dtype(dtype_input, check_list, param_name="input_data")

    if data_format in ("NHWC",):
        if len(ksize) != 4:
            raise RuntimeError("Invalid ksize params, ksize dim must be 4.")
        if ksize[0] != 1 or ksize[3] != 1:
            raise RuntimeError(
                "MaxPool only supports pooling across width/height,"
                "and other ksize dimension should be one")
        if len(strides) != 4:
            raise RuntimeError("Invalid strides params, strides dim must be 4.")
        if strides[0] != 1 or strides[3] != 1:
            raise RuntimeError(
                "MaxPool only supports pooling across width/height,"
                "and other strides dimension should be one")
    elif data_format in ("NC1HWC0", "NCHW"):
        if len(ksize) != 4:
            raise RuntimeError("Invalid ksize params, ksize dim must be 4.")
        if ksize[0] != 1 or ksize[1] != 1:
            raise RuntimeError(
                "MaxPool only supports pooling across width/height,"
                "and other ksize dimension should be one")
        if len(strides) != 4:
            raise RuntimeError("Invalid strides params, strides dim must be 4.")
        if strides[0] != 1 or strides[1] != 1:
            raise RuntimeError(
                "MaxPool only supports pooling across width/height,"
                "and other strides dimension should be one")
    else:
        raise RuntimeError("The data_format is not supported")

    if padding not in ("SAME", "VALID"):
        raise RuntimeError(
            "MaxPool can only support SAME or VALID padding mode.")

    # set tensor attrs, during L1 fusion these attrs will assign by te_fusion
    addr_type = input_data.get("addr_type", 0)
    valid_shape = input_data.get("valid_shape", [])
    slice_offset = input_data.get("slice_offset", [])
    split_index = input_data.get("split_index", 0)
    l1_fusion_type = input_data.get("L1_fusion_type", -1)
    attr = {"addr_type": addr_type,
            "valid_shape": valid_shape,
            "slice_offset": slice_offset,
            "split_index": split_index,
            "L1_fusion_type": l1_fusion_type}
    is_l1fusion = l1_fusion_type in (0, 1)

    data_input = tvm.placeholder(shape_input, name="data_input",
                                 dtype=dtype_input, attrs=attr)

    res = max_pool_compute(data_input, output_data, ksize, strides, padding,
                           data_format=data_format, kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [data_input, res],
        "l1_fusion_option": is_l1fusion}
    te.lang.cce.cce_build_code(sch, config)
