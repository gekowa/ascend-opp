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
reduce_all_d
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648

# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("reduce_all_d")
def reduce_all_d_compute(input_data, output_data, axes,
                         keepdims, kernel_name="reduce_all_d"):
    """ TVM calculation process, used for fusion operation

    Parameters
    ----------
    input_data: TVM tensor
        the placeholder of input data
    output_data: dict
        shape and dtype of output, should be same shape and type as input
    axes: int, list ,tuple or None.
        the first axes to reduce, may be negative to index from the end
        (e.g., -1 for the last axes).
        axes may be int or list(e.g. [1,2])
    keepdims : bool or None .
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name: str
        cce kernel name, default value is "all_cce"

    Returns
    -------
    result: TVM tensor.
    """
    shape = te.lang.cce.util.shape_to_list(input_data.shape)
    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)

    dtype = input_data.dtype
    data_fp16 = te.lang.cce.cast_to(input_data, "float16")
    data_abs = te.lang.cce.vabs(data_fp16)
    result_tmp = te.lang.cce.reduce_min(data_abs, axes, keepdims=False)
    result = te.lang.cce.cast_to(result_tmp, dtype, True)

    return result

@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, (REQUIRED_ATTR_INT, REQUIRED_ATTR_LIST_INT),
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def reduce_all_d(input_data, output_data, axes,
                 keepdims=None, kernel_name="reduce_all_d"):
    """
    Reduce a tensor on a certain axes based on min

    Parameters:
    ----------
    input_data: dict
        shape and dtype of input_data, only support int8
    output_data: dict
        source data type, only support int8
    axes : int, list ,tuple or None.
        the first axes to reduce, may be negative to index from the end
        (e.g., -1 for the last axes).
        axes may be int or list(e.g. [1,2])
    keepdims : bool or None .
        if true, retains reduced dimensions with length 1,
        default value is None
    kernel_name : str
        cce kernel name, default value is "cce_all"

    Returns
    -------
    None
    """
    input_shape = input_data.get("shape")
    input_dtype = input_data.get("dtype").lower()
    if input_dtype == "bool":
        input_dtype = "int8"
    check_shape(input_shape, param_name="input_data")
    check_dtype(input_dtype, ("int8"), param_name="input_data")

    shape_len = len(input_shape)
    if not axes:
        axes = range(shape_len)

    if hasattr(axes, 'index'):
        axes = list(axes)
    axes = util.axis_check(shape_len, axes)

    if not isinstance(axes, int):
        for i in axes:
            if i >= len(input_shape):
                raise RuntimeError("axes should be less than dimension")
    else:
        if axes >= len(input_shape):
            raise RuntimeError("axes should be less than dimension")

    # 5HD Special param for 5hd schedule
    is_5hdc = util.check_and_init_5hdc_reduce_support(input_data, axes)
    if not is_5hdc:
        input_shape, axes = util.shape_refine(list(input_shape), axes)
        input_shape, axes = util.simplify_axis_shape(input_shape, axes)

    data_input = tvm.placeholder(input_shape, name="data_input_" + kernel_name,
                                 dtype=input_dtype)
    result = reduce_all_d_compute(data_input, output_data, axes,
                                  keepdims, kernel_name)
    if is_5hdc:
        result.ori_shape = input_data["ori_shape"]
        result.ori_format = input_data["ori_format"]

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_input, result]}
    te.lang.cce.cce_build_code(sch, config)
