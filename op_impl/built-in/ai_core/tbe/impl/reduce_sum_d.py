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
reduce_sum_d
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import *

# define the type of None
NONETYPE = type(None)
# define the limit of shape dim
MAX_SHAPE_NUM = 10000000


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("reduce_sum_d")
def reduce_sum_d_compute(x,
                         y,
                         axis,
                         keepdims,
                         kernel_name="reduce_sum_d",
                         is_5hdc=False):
    """redusce_sum_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    shape_len = len(shape)
    if not axis:
        axis = range(shape_len)
    if hasattr(axis, 'index'):
        axis = list(axis)

    dtype = x.dtype
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if cce_product not in ("Ascend310",) and dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support(
                "te.lang.cce.sum", "float32") and not is_5hdc:
        x = te.lang.cce.cast_to(x, "float32")
    res_sum = te.lang.cce.sum(x, axis=axis, keepdims=keepdims)
    res = te.lang.cce.cast_to(res_sum, dtype)

    return res


# pylint: disable=locally-disabled,too-many-locals
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_INT, OPTION_ATTR_BOOL, KERNEL_NAME)
def reduce_sum_d(x, y, axis, keepdims=None, kernel_name="reduce_sum_d"):
    """reduce a tensor on a certain axis based on sum.

    Parameters:
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32")

    check_shape(shape, param_name="x")
    check_dtype(dtype_lower, check_list, param_name="x")

    axis_d = []
    shape_len = len(shape)
    if not axis:
        for i, _ in enumerate(shape):
            axis_d.append(i)
    else:
        axis_d = list(axis)
    axis_d = util.axis_check(shape_len, axis_d)
    # 5HD Special param for 5hd schedule
    is_5hdc = util.check_and_init_5hdc_reduce_support(x, axis)

    if not keepdims and not is_5hdc:
        shape, axis_d = util.shape_refine(list(shape), axis_d, keepdims)
        shape, axis_d = util.simplify_axis_shape(shape, axis_d)

    data_input = tvm.placeholder(shape, name="data_input_" + kernel_name,
                                 dtype=dtype_lower)
    res = reduce_sum_d_compute(data_input, y, axis_d, keepdims,
                               is_5hdc=is_5hdc)
    if is_5hdc:
        res.ori_shape = x["ori_shape"]
        res.ori_format = x["ori_format"]

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}
    te.lang.cce.cce_build_code(sch, config)
