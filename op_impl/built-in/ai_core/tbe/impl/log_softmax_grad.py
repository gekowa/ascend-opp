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
log_softmax_grad
"""
from __future__ import absolute_import
import operator

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import *

# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648

# pylint: disable = locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("log_softmax_grad")
def log_softmax_grad_compute(input_dy, input_x, output_z, axis,
                             kernel_name="log_softmax_grad"):
    """
    TVM calculation process, used for fusion operation.
        dy - (exp(x) * sum(dy))

    Parameters
    ----------
    input_dy: TVM tensor
        the placeholder of input grad data
    input_x: TVM tensor
        the placeholder of input data
    output_z: dict
        shape and dtype of output, should be the same shape and type as input
    axis: int, list or tuple .
        the first axis to reduce, may be negative to index from the end
        (e.g., -1 for the last axis).
        axis may be int or list(e.g. [1,2])
        if true, retains reduced dimensions with length 1,
        default value is -1
    kernel_name: str
        cce kernel name, default value is log_softmax_grad

    Returns
    -------
    result: TVM tensor.
    """
    dtype = input_dy.dtype
    shape1 = te.lang.cce.util.shape_to_list(input_dy.shape)    
    has_improve_precision = False
    if dtype == "float16" and \
        tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp",
                                                "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        input_dy = te.lang.cce.cast_to(input_dy, "float32")
        has_improve_precision = True

    data_exp = te.lang.cce.vexp(input_x)
    data_sum = te.lang.cce.sum(input_dy, axis, True)
    data_sum_broadcast = te.lang.cce.broadcast(data_sum, shape1)
    data_softmax = te.lang.cce.vmul(data_exp, data_sum_broadcast)

    result = te.lang.cce.vsub(input_dy, data_softmax)
    if has_improve_precision:
        result = te.lang.cce.cast_to(result, "float16")

    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 (OPTION_ATTR_INT, OPTION_ATTR_LIST_INT), KERNEL_NAME)
def log_softmax_grad(input_dy, input_x, output_z, axis=-1,
                     kernel_name="log_softmax_grad"):
    """
    algorithm: log_softmax_grad
    calculating: gradient of log_softmax

    Parameters
    ----------
    input_dy : dict
        shape and dtype of grad input, only support float16, float32
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be the same shape and type as input
    axis: int, list or tuple .
        the first axis to reduce, may be negative to index from the end
        (e.g., -1 for the last axis).
        axis may be int or list(e.g. [1,2])
        if true, retains reduced dimensions with length 1,
        default value is -1
    kernel_name: str
        cce kernel name, default value is log_softmax_grad

    Returns
    -------
    None
    """
    check_list = ("float16", "float32")
    input_dtype = input_dy.get("dtype").lower()

    if not isinstance(axis, int):
        axis = list(axis)

    shape1 = input_dy.get("shape")
    shape2 = input_x.get("shape")
    check_shape(shape1, param_name="input_dy")
    check_shape(shape2, param_name="input_x")
    check_dtype(input_dtype, check_list, param_name="input_dy")

    axis = util.axis_check(len(shape1), axis)

    if not isinstance(axis, int):
        for i in axis:
            if list(shape1)[i] == 1:
                raise RuntimeError("Cannot reduce on an axis with dimension 1")
    else:
        if list(shape1)[axis] == 1:
            raise RuntimeError("Cannot reduce on an axis with dimension 1")

    if not operator.eq(list(shape1), list(shape2)):
        raise RuntimeError("all input shape must be equal")

    shape1, axis = util.shape_refine(list(shape1), axis)
    shape2 = shape1

    data1 = tvm.placeholder(shape1, dtype=input_dtype, name="data1")
    data2 = tvm.placeholder(shape2, dtype=input_dtype, name="data2")
    result = log_softmax_grad_compute(data1, data2, output_z, axis, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data1, data2, result]}
    te.lang.cce.cce_build_code(sch, config)
