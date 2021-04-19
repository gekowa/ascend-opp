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
euclidean_norm_d
"""
import te.lang.cce
from te import tvm
from te.utils.op_utils import *
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.platform import get_soc_spec


@fusion_manager.register("euclidean_norm_d")
def euclidean_norm_d_compute(x,
                             y,
                             axes,
                             keepdims,
                             kernel_name="euclidean_norm_d"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input_x
    y : dict
        dict of output_y, include keys(shape and dtype)
    axes: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name : str
        kernel name, default value is "euclidean_norm_d"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    dtype = x.dtype.lower()
    shape = x.shape
    product = get_soc_spec("SOC_VERSION")
    if product != "Ascend310" and dtype != "float32":
        x = te.lang.cce.cast_to(x, "float32")
    one_flag = []
    axis = list(axes)
    for i in axis:
        one_flag.append(int(shape[i]))

    if int(len(set(one_flag))) == 1 and int(one_flag[0]) == 1:
        res = te.lang.cce.vmuls(x, 1)
    else:
        res_mul = te.lang.cce.vmul(x, x)
        res_sum = te.lang.cce.sum(res_mul, axes, keepdims)
        res = te.lang.cce.vsqrt(res_sum, 1)

    if res.dtype != dtype:
        res = te.lang.cce.cast_to(res, dtype)
    return res


def refine_shape_axes_custom(shape, axes):
    wrapped_axes = wrap_axes_to_positive(axes, len(shape))
    wrapped_axes = sorted(wrapped_axes)
    refined_axes = []
    reduce_flag = -1
    refined_shape = []
    for idx, dim in enumerate(shape):
        tmp_flag = 1 if idx in wrapped_axes else 0
        if reduce_flag == 1 and tmp_flag == 1:
            # continues reduce
            refined_shape[-1] *= dim
        elif reduce_flag == 0 and tmp_flag == 0:
            # continues no reduce
            refined_shape[-1] *= dim
        else:
            refined_shape.append(dim)
            if tmp_flag == 1:
                refined_axes.append(len(refined_shape) - 1)
            reduce_flag = tmp_flag

    return refined_shape, refined_axes


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_LIST_INT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def euclidean_norm_d(input_data,
                     output_data,
                     axes=None,
                     keepdims=False,
                     kernel_name="euclidean_norm_d"):
    """
    calculating data

    Parameters
    ----------
    input_data : dict
        shape and dtype of input
    output_data : dict
        shape and dtype of output, should be same format and type as input
    axes : int, list ,tuple or None.
        the first axes to reduce, may be negative to index from the end
    keepdims : bool or None .
        if true, retains reduced dimensions with length 1,
        default value is None

    Returns
    -------
    None
    """
    shape = input_data.get("shape")
    dtype = input_data.get("dtype")
    input_dtype = dtype.lower()

    check_list = ["float16", "float32", "int32"]
    check_dtype(input_dtype, check_list)

    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)

    axes = wrap_axes_to_positive(axes, len(shape))

    check_shape(axes, min_dim=-shape_len, max_dim=shape_len - 1)
    axis = list(axes)
    one_flag = []
    for i in axis:
        one_flag.append(int(shape[i]))

    if int(len(set(one_flag))) == 1 and int(one_flag[0]) == 1:
        refined_shape, refined_axes = shape, axes
    else:
        refined_shape, refined_axes = refine_shape_axes_custom(shape, axes)
    data_input = tvm.placeholder(refined_shape, name="data_input", dtype=input_dtype)
    res = euclidean_norm_d_compute(data_input, output_data,
                                   refined_axes, keepdims, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(schedule, config)

