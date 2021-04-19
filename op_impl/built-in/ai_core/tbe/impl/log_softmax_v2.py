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
log_softmax_v2
"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import *

# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648

# pylint: disable = locally-disabled,unused-argument
@fusion_manager.register("log_softmax_v2")
def log_softmax_v2_compute(input_x, output_y, axis=-1, kernel_name="log_softmax_v2", impl_mode="high_performance"):
    """
    process of calculating data's log_softmax, x - log(sum(exp(x)))
    this x is x - xmax

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output: dict
        shape and dtype of output, should be same shape and type as input
    axis: int, list or tuple
        the data's axis, range is [-d, d-1]
    kernel_name : str
        cce kernel name, default value is log_softmax_v2

    Returns
    -------
    result: TVM tensor.
    """
    inp_dtype = input_x.dtype
    shape = te.lang.cce.util.shape_to_list(input_x.shape)

    if impl_mode == "high_precision":
        data_max = te.lang.cce.reduce_max(input_x, axis=axis, keepdims=True, priority_flag=True)
    else:
        data_max = te.lang.cce.reduce_max(input_x, axis=axis, keepdims=True)
    data_max_broadcast = te.lang.cce.broadcast(data_max, shape)
    data_sub = te.lang.cce.vsub(input_x, data_max_broadcast)

    # increase accuracy
    has_improve_precision = False
    if inp_dtype == "float16" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                "float32"):
        data_sub = te.lang.cce.cast_to(data_sub, "float32")
        has_improve_precision = True

    data_exp = te.lang.cce.vexp(data_sub)
    data_sum = te.lang.cce.sum(data_exp, axis=axis, keepdims=True)
    data_log = te.lang.cce.vlog(data_sum)
    data_log_broadcast = te.lang.cce.broadcast(data_log, shape)
    res = te.lang.cce.vsub(data_sub, data_log_broadcast)

    # cast output type same as input type
    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, (OPTION_ATTR_INT, OPTION_ATTR_LIST_INT), KERNEL_NAME)
def log_softmax_v2(input_x, output_y, axis=-1, kernel_name="log_softmax_v2", impl_mode="high_performance"):
    """
    algorithm: log_softmax
    calculating data's log_softmax, x - log(sum(exp(x)))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis: int, list or tuple
        the data's axis, range is [-d, d-1]
    kernel_name : str
        cce kernel name, default value is log_softmax_v2

    Returns
    -------
    None
    """
    check_list = ("float16", "float32")
    shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    shape_len = len(shape)
    shape_list = list(shape)

    if not isinstance(axis, int):
        axis = list(axis)

    check_shape(shape, param_name="input_x")
    check_dtype(input_dtype, check_list, param_name="input_x")

    axis = util.axis_check(shape_len, axis)

    if not isinstance(axis, int):
        for i in axis:
            if shape_list[i] == 1:
                raise RuntimeError("Cannot reduce on an axis with dimension 1")
    else:
        if shape_list[axis] == 1:
            raise RuntimeError("Cannot reduce on an axis with dimension 1")

    shape, axis = util.shape_refine(list(shape), axis)
    shape, axis = util.simplify_axis_shape(shape, axis)

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    result = log_softmax_v2_compute(data_input, output_y, axis=axis,
                                    kernel_name=kernel_name, impl_mode=impl_mode)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_input, result]}

    te.lang.cce.cce_build_code(sch, config)
