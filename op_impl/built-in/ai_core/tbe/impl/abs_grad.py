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
abs_grad

  Op_description :
    Computes gradients for abs operation

    # abs_grad(
    #   y,
    #   dy,
    #   z,
    #   kernel_name="cce_abs_grad")

  Supportive_dtype_format :
    ['float16', 'float32']
    ['ALL']

  Constraint :
    [1] All : 'y' and 'dy' must have the same type and shape.
    [2] All : shape size limit is 2147483648.
"""
import operator

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import *
from topi import generic
from topi.cce import util

SHAPE_SIZE_LIMIT = 2147483648

# pylint: disable=unused-argument,too-many-locals,invalid-name
@fusion_manager.register("abs_grad")
def abs_grad_compute(y, dy, z, kernel_name="abs_grad"):
    """
    do abs_grad compute
    Parameters:
    ----------------
    y: input tensor y
    dy: input tensor dy
    z: output dict
    kernel_name: cce kernel name, default value is "abs_grad"
    return: data_dy * sign(data_y)
    ----------------
    """

    dtype = dy.dtype

    if dtype == "float16":
        fp_max = tvm.const(2 ** 15, dtype)
        fp_min = tvm.const(2 ** (-15), dtype)
    else:
        fp_max = tvm.const(2 ** 62, dtype)
        fp_min = tvm.const(2 ** (-127), dtype)
    new_data = te.lang.cce.vmuls(y, fp_max)
    abs_data = te.lang.cce.vabs(new_data)
    denominator = te.lang.cce.vadds(abs_data, fp_min)
    res = te.lang.cce.vdiv(new_data, denominator)
    res = te.lang.cce.round(res)
    data1_res = te.lang.cce.vmul(res, dy)
    return data1_res

# pylint: disable=invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def abs_grad(y, dy, z, kernel_name="abs_grad"):
    """
    do element-wise abs_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32

    dy : dict of dy, include shape and dtype, dtype support float16, float32

    z : dict of z, include shape and dtype, dtype support float16, float32

    kernel_name : cce kernel name, default value is "abs_grad"
    -------
    """

    # get the shape and dtype for input_1,input_2
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype_y = y.get("dtype")
    dtype_dy = dy.get("dtype")

    check_shape(shape_y, param_name="y")
    check_shape(shape_dy, param_name="dy")
    shape_y, _ = refine_shape_axes(shape_y, [])
    shape_dy, _ = refine_shape_axes(shape_dy, [])

    check_list = ("float16", "float32")
    check_dtype(dtype_y, check_list, param_name="y")
    check_dtype(dtype_dy, check_list, param_name="dy")
    dtype_y = dtype_y.lower()
    dtype_dy = dtype_dy.lower()
    if not operator.eq(shape_y, shape_dy):
        raise RuntimeError(
            "abs_grad only support input shape while input_shape1 equals to input_shape2")
    if dtype_y != dtype_dy:
        raise RuntimeError(
            "abs_grad only support dtype while input_dtype1 equals to input_dtype2")
    shape_y, _ = refine_shape_axes(shape_y, [])
    shape_dy, _ = refine_shape_axes(shape_dy, [])

    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data1")
    data_dy = tvm.placeholder(shape_dy, dtype=dtype_dy, name="data2")
    res = abs_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_y, data_dy, res]}
    te.lang.cce.cce_build_code(sch, config)
