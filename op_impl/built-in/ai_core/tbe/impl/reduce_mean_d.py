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
reduce_mean_d
"""
from collections import Iterable

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te import platform as tbe_platform
from topi import generic
from topi.cce import util
from te.utils.op_utils import *

SHAPE_SIZE_LIMIT = 100000000  # shape limit for tf_reduce_mean

NoneType = type(None)
ori_shape = [[0], [0]]
ori_format = ["NCHW", "NCHW"]


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("reduce_mean_d")
def reduce_mean_d_compute(x,
                          y,
                          axes,
                          keepdims,
                          kernel_name="reduce_mean_d",
                          impl_mode="high_performance",
                          is_5hdc=False):
    """reduce_mean_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axes: int, list, tuple or NoneType
        the axes for reduce.
    keepdims: bool or NoneType
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_mean_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)

    shape_len = len(shape)
    if not axes:
        axes = range(shape_len)
    if hasattr(axes, 'index'):
        axes = list(axes)
    axes = util.axis_check(shape_len, axes)

    reduce_elts = 1.0
    if isinstance(axes, Iterable):
        for i in axes:
            reduce_elts *= shape[i]
    else:
        reduce_elts = shape[axes]
    cof = reduce_elts ** (-1)

    if ori_format[0] == 'NHWC' and ori_format[1] == 'NC1HWC0' and len(axes) == 2 \
            and axes == [1, 4] and len(ori_shape[0]) == 4:
        cof = ori_shape[0][-1] ** (-1)

    dtype = x.dtype
    data_input_tmp = x

    has_improve_precision = False
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if cce_product not in ("Ascend310",) and dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support(
                "te.lang.cce.sum", "float32") and not is_5hdc:
        data_input_tmp = te.lang.cce.cast_to(data_input_tmp, "float32")
        has_improve_precision = True
    elif cce_product in ("Ascend310",) and dtype == "float16" \
            and tbe_platform.cce_conf.api_check_support("te.lang.cce.sum",
                                                        "float32") \
            and not is_5hdc and impl_mode != "high_performance":
        data_input_tmp = te.lang.cce.cast_to(data_input_tmp, "float32")
        has_improve_precision = True

    data_input_tmp = te.lang.cce.vmuls(data_input_tmp, cof)
    res = te.lang.cce.sum(data_input_tmp, axis=axes, keepdims=keepdims)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, dtype)

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT,
                 (REQUIRED_ATTR_INT, REQUIRED_ATTR_LIST_INT),
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def reduce_mean_d(input_x, output_y, axes,
                  keepdims=None, kernel_name="reduce_mean_d",
                  impl_mode="high_performance"):
    """
    Reduce a tensor on a certa in axes based on mean.

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input
    output_y: dict
        shape and dtype of output
    axes : int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keepdims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    kernel_name : str
        cce kernel name, default value is reduce_mean_d

    Returns
    -------
    None
    """
    global ori_shape
    global ori_format
    shape = input_x.get("shape")
    check_shape(shape, param_name="input_x")
    check_list = ["float16", "float32"]
    shape_len = len(shape)

    if not axes:
        axes = range(shape_len)

    if hasattr(axes, 'index'):
        axes = list(axes)

    inp_dtype = input_x.get("dtype").lower()
    check_dtype(inp_dtype, check_list, param_name="input_x")

    axes = util.axis_check(shape_len, axes)

    # Shape should not be modified in 5HD mode
    # 5HD Special param for 5hd schedule
    is_5hdc = util.check_and_init_5hdc_reduce_support(input_x, axes)
    if not is_5hdc:
        shape, axes = util.shape_refine(list(shape), axes)
        shape, axes = util.simplify_axis_shape(shape, axes)

    ori_shape = [input_x["ori_shape"], input_x["shape"]]
    ori_format = [input_x["ori_format"], input_x["format"]]
    data_input = tvm.placeholder(shape, name="data_input", dtype=inp_dtype)
    res = reduce_mean_d_compute(data_input, output_y, axes, keepdims,
                                impl_mode=impl_mode, is_5hdc=is_5hdc)
    if is_5hdc:
        res.ori_shape = input_x["ori_shape"]
        res.ori_format = input_x["ori_format"]

    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
