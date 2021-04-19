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
clip_by_value
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("clip_by_value")
def clip_by_value_compute(input_t, clip_value_min, clip_value_max, output_t,
                          kernel_name="clip_by_value"):
    """
    algorithm: clip_by_value
    data = max  if data > max
    data = min  if data < min

    Parameters
    ----------
    input_t: TVM tensor
        the placeholders of input data
    clip_value_min: TVM tensor
        the placeholders of clip_value_min
    clip_value_max: TVM tensor
        the placeholders of data_max
    output_t: dict
        shape and dtype of output
    kernel_name: str
        kernel name, default value is "clip_by_value"

    Returns
    -------
    res: TVM tensor
        result of compute
    """
    input_dtype = input_t.dtype
    input_shape = te.lang.cce.util.shape_to_list(input_t.shape)
    shape_min_org = te.lang.cce.util.shape_to_list(clip_value_min.shape)
    shape_max_org = te.lang.cce.util.shape_to_list(clip_value_max.shape)
    if list(shape_min_org) != list(input_shape):
        clip_value_min = te.lang.cce.broadcast(clip_value_min, input_shape)
    if list(shape_max_org) != list(input_shape):
        clip_value_max = te.lang.cce.broadcast(clip_value_max, input_shape)
    res_min = te.lang.cce.vmin(input_t, clip_value_max)
    res_max = te.lang.cce.vmax(res_min, clip_value_min)
    res = te.lang.cce.cast_to(res_max, input_dtype)
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def clip_by_value(input_t, clip_value_min, clip_value_max,
                  output_t, kernel_name="clip_by_value"):
    """
    algorithm: clip_by_value
    Clips tensor values to a specified min and max.
    Given a tensor t, this operation returns a tensor of
    the same type and shape as t
    with its values clipped to clip_value_min and clip_value_max.
    Any values less than clip_value_min are set to clip_value_min.
    Any values greater than clip_value_max are set to clip_value_max.

    Parameters
    ----------
    input_t: dict with keys(shape and dtype)
           input tensor
    clip_value_min: dict with keys(shape and dtype) or scaler
           The minimum value to clip by.
    clip_value_max: dict with keys(shape and dtype) or scaler
           The minimum value to clip by.
    output_t: dict
           info of output tensor with the same shape as input.
    kernel_name: str
           kernel name, default value is "clip_by_value"

    Returns
    -------
    None
    """
    shape_x = input_t.get("shape")
    dtype = input_t.get("dtype")
    shape_min = clip_value_min.get("shape")
    shape_max = clip_value_max.get("shape")
    input_dtype = dtype.lower()
    check_dtype(input_dtype, ("float16", "float32", "int32"), param_name="input_t")
    if (shape_min != 0 and shape_max != 0):
        if (len(shape_min) > 1 and list(shape_min) != list(shape_x)):
            for i in range(0, len(shape_x)):
                if shape_min[i] != shape_x[i] and shape_min[i] != 1:
                    raise RuntimeError(
                        "min/max: A 0-D (scalar) Tensor, "
                        "or a Tensor with the same shape as t, "
                        "or a Tensor broadcast to shape as t.")
        if (len(shape_max) > 1 and list(shape_max) != list(shape_x)):
            for i in range(0, len(shape_x)):
                if shape_max[i] != shape_x[i] and shape_max[i] != 1:
                    raise RuntimeError(
                        "min/max: A 0-D (scalar) Tensor, "
                        "or a Tensor with the same shape as t, "
                        "or a Tensor broadcast to shape as t.")
    check_shape(shape_x, param_name="input_t")
    shape_x = util.shape_refine(shape_x)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=input_dtype)

    data_value = {}
    check_shape(shape_min, param_name="clip_value_min")
    shape_min = util.shape_refine(shape_min)
    if len(shape_min) != len(shape_x) and len(shape_min) == 1:
        list_min = [1]*(len(shape_x) - 1)
        shape_min = shape_min + list_min
    data_value["min"] = tvm.placeholder(shape_min, name="data_min",
                                        dtype=input_dtype)

    check_shape(shape_max, param_name="clip_value_max")
    shape_max = util.shape_refine(shape_max)
    if len(shape_max) != len(shape_x) and len(shape_max) == 1:
        list_max = [1]*(len(shape_x) - 1)
        shape_max = shape_max + list_max
    data_value["max"] = tvm.placeholder(shape_max, name="data_max",
                                        dtype=input_dtype)

    res = clip_by_value_compute(data_x, data_value["min"], data_value["max"],
                                output_t, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_value["min"],
                              data_value["max"], res]}
    te.lang.cce.cce_build_code(sch, config)
