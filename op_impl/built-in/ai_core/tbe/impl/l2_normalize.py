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
l2_normalize
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.lang.cce.te_compute.util import shape_to_list
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import *


# pylint: disable=unused-argument
@fusion_manager.register("l2_normalize")
def l2_normalize_compute(input_x,
                         output_y,
                         axis,
                         epsilon,
                         kernel_name="l2_normalize"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    axis : list
        the axis which to be computed
    epsilon : float
        the minimum value, in case the denominator is zero
    kernel_name : str
        kernel name, default value is "l2_normalize"

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmul", "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
    x_square = te.lang.cce.vmul(input_x, input_x)
    x_square_sum = te.lang.cce.sum(x_square, axis, keepdims=True)
    const_epsilon = tvm.const(epsilon, "float32")
    x_l2norm = te.lang.cce.vmaxs(x_square_sum, const_epsilon)
    x_l2norm_sqrt = te.lang.cce.vsqrt(x_l2norm)
    x_l2norm_sqrt = te.lang.cce.broadcast(x_l2norm_sqrt,
                                          shape_to_list(input_x.shape))

    result = te.lang.cce.vdiv(input_x, x_l2norm_sqrt)

    if dtype == "float16":
        result = te.lang.cce.cast_to(result, "float16")
    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_FLOAT,
                 KERNEL_NAME)
def l2_normalize(input_x, output_y, axis, epsilon, kernel_name="l2_normalize"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    axis : list
        the axis which to be computed
    epsilon : float
        the minimum value, in case the denominator is zero
    kernel_name : str
        kernel name, default value is "l2_normalize"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()

    check_shape(shape, param_name="input_x")
    check_dtype(input_dtype, ("float16", "float32"), param_name="input_x")

    for i in axis:
        if not isinstance(i, int):
            raise RuntimeError("the axis element must be int")
        if i >= len(shape) or i < -len(shape):
            raise RuntimeError("the axis is invalid")

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = l2_normalize_compute(data_input, output_y,
                               axis, epsilon, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}

    te.lang.cce.cce_build_code(sch, config)

