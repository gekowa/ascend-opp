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
max_pool3d_grad_grad_d
"""
from __future__ import absolute_import

# pylint: disable=E0401
import te.lang.cce
import time
from te import tvm
from topi import generic

# shape limit
# int32's max value
SHAPE_SIZE_LIMIT = 2 ** 31 - 1
C0SIZE = 16
NoneType = type(None)
CAFFE_DATA_MODE = 0
TENSORFLOW_DATA_MODE = 1
MAX_BUILD_ROUND_FOR_RECALC_UB = 8


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=too-many-locals,too-many-boolean-expressions
def max_pool3d_grad_grad_d(orig_input, orig_output, grad_grad, assist, output,
                           ksize, strides, pads=(0, 0, 0, 0, 0, 0),
                           data_format="NDHWC",
                           kernel_name="max_pool3d_grad_grad_d"):
    """
    Parameters
    ----------
    orig_input : dict, shape and dtype of input_data, format is NDC1HWC0

    orig_output : dict, result of max_pool3d(orig_input, ksize, ...),format is NDC1HWC0

    grad_grad : dict, gradients of gradients, format is NDC1HWC0

    output: dict, shape and dtype of output_data,format is NDC1HWC0

    ksize : list or tuple, the window of max_pool3d_grad_grad_d,
            only support max_pool3d_grad_grad_d in D or H or W

    strides : list or tuple, the stride of max_pool3d window,
              only support max_pool3d_grad_grad_d in D or H or W

    pads : reserved.

    data_format : str, default = "NDHWC"

    kernel_name : cce kernel name, default value is "max_pool3d_grad_grad_d"

    Returns
    -------
    None
    """
    if (pads[0] == 0 and pads[1] == 0 and pads[2] == 0 and\
        pads[3] == 0 and pads[4] == 0 and pads[5] == 0):
        padding = "VALID"
    else:
        padding = "SAME"

    orig_input_shape = orig_input.get("shape")
    orig_output_shape = orig_output.get("shape")
    grad_grad_shape = grad_grad.get("shape")
    assist_shape = assist.get("shape")

    input_dtype = orig_input.get("dtype")
    output_dtype = orig_output.get("dtype")
    input_dtype = input_dtype.lower()
    output_dtype = output_dtype.lower()

    window_d, window_h, window_w = _get_ksize(ksize, data_format)
    stride_d, stride_h, stride_w = _get_stride(strides, data_format)
    orig_input_tensor = tvm.placeholder(orig_input_shape,
                                        name="orig_data_input",
                                        dtype=input_dtype)
    orig_output_tensor = tvm.placeholder(orig_output_shape,
                                         name="orig_data_output",
                                         dtype=output_dtype)
    grad_grad_tensor = tvm.placeholder(grad_grad_shape,
                                       name="grad_grad",
                                       dtype=input_dtype)
    assist_tensor = tvm.placeholder(assist_shape,
                                    name="assist",
                                    dtype=input_dtype)

    #UB size can not be calculated accurately, so retry 8 times at most
    build_count = 0
    while build_count <= MAX_BUILD_ROUND_FOR_RECALC_UB:
        res = te.lang.cce.pooling3d_max_grad_grad(orig_input_tensor,
                                              orig_output_tensor,
                                              grad_grad_tensor,
                                              assist_tensor,
                                              (window_d, window_h, window_w),
                                              (stride_d, stride_h, stride_w),
                                              pads, data_format, padding)
        try:
            with tvm.target.cce():
                #because of attr could be assigned only once, so use attr name in schedule to judge tiling round.
                res.op.attrs["recalc_ub_round_"+str(build_count)] = build_count
                build_count = build_count + 1
                sch = generic.auto_schedule(res)
                config = {
                    "name": kernel_name,
                    "dummy_placeholder": True,
                    "tensor_list": [orig_input_tensor, orig_output_tensor,
                                    grad_grad_tensor, assist_tensor, res]}
                te.lang.cce.cce_build_code(sch, config)
                break
        except tvm.TVMError as e:
            if str(e).find("VMError: Allocation exceed bound of memory tag:local.UB") != -1:
                print("shenmin, find: ", str(e).find("VMError: Allocation exceed bound of memory tag:local.UB"))
                continue
            raise
            break


def _get_ksize(ksize, data_format):
    if len(ksize) == 1:
        return ksize[0], ksize[0], ksize[0]
    if len(ksize) == 3:
        return ksize[0], ksize[1], ksize[2]
    if data_format == "NDHWC" and len(ksize) == 5:
        return ksize[1], ksize[2], ksize[3]
    if data_format == "NCDHW" and len(ksize) == 5:
        return ksize[2], ksize[3], ksize[4]
    raise RuntimeError("Invalid ksize")


def _get_stride(strides, data_format):
    if len(strides) == 1:
        return strides[0], strides[0], strides[0]
    if len(strides) == 3:
        return strides[0], strides[1], strides[2]
    if data_format == "NDHWC" and len(strides) == 5:
        return strides[1], strides[2], strides[3]
    if data_format == "NCDHW" and len(strides) == 5:
        return strides[2], strides[3], strides[4]
    raise RuntimeError("Invalid strides")
