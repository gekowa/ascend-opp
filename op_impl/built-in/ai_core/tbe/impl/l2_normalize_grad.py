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
l2_normalize_grad
"""
import te.lang.cce
from te.lang.cce.te_compute.util import shape_to_list
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import *


# shape limit for aicore equals 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument,too-many-locals
@fusion_manager.register("l2_normalize_grad")
def l2_normalize_grad_compute(input_data_x,
                              input_data_y,
                              input_data_dy,
                              output_data_dx,
                              dim=1,
                              eps=1e-12,
                              kernel_name="l2_normalize_grad"):
    """
    calculating l2_normalize_grad

    Parameters
    ----------
    input_data_x : TVM tensor
        the placeholder of foward input
    input_data_y : TVM tensor
        the placeholder of foward output
    input_data_dy : TVM tensor
        the placeholder of backward input
    output_data_dx: dict
        the placeholder of backward output, only support float16, float32
    dim : int
        the dimemsion to normalize. Default : 1.
        only support dim=1(for NC/NCHW) and
        dim=[1,4](for NC1HWC0) in this version
    eps : float
        small value to avoid division by zero. Default: 1e-12
    kernel_name : str
        cce kernel name, default value is l2_normalize_grad

    Returns
    -------
    result: TVM tensor
        the result of l2_normalize_grad
    """
    dtype = input_data_x.dtype
    # max(||x||, eps)
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        input_data_x = te.lang.cce.cast_to(input_data_x, "float32")

    x_square = te.lang.cce.vmul(input_data_x, input_data_x)
    x_l2norm = te.lang.cce.sum(x_square, dim, keepdims=True)
    x_l2norm = te.lang.cce.vsqrt(x_l2norm)
    x_l2norm = te.lang.cce.vmaxs(x_l2norm, tvm.const(eps, dtype))
    x_l2norm_broadcast = te.lang.cce.broadcast(x_l2norm,
                                               shape_to_list(
                                                   input_data_x.shape))

    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        input_data_y = te.lang.cce.cast_to(input_data_y, "float32")
        input_data_dy = te.lang.cce.cast_to(input_data_dy, "float32")

    y_mul_dy = te.lang.cce.vmul(input_data_y, input_data_dy)
    sum_y_mul_dy = te.lang.cce.sum(y_mul_dy, dim, keepdims=True)
    sum_y_mul_dy_broadcast = te.lang.cce.broadcast(sum_y_mul_dy,
                                                   shape_to_list(
                                                       input_data_x.shape))

    # dx = (dy - y * sum(dy*y)) / max(||x||, eps)
    numerator = te.lang.cce.vsub(input_data_dy,
                                 te.lang.cce.vmul(input_data_y,
                                                  sum_y_mul_dy_broadcast))
    result = te.lang.cce.vdiv(numerator, x_l2norm_broadcast)

    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        result = te.lang.cce.cast_to(result, "float16")

    return result


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 (REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_INT), REQUIRED_ATTR_FLOAT, KERNEL_NAME)
def l2_normalize_grad(input_x,
                      input_y,
                      input_dy,
                      output_dx,
                      dim=1,
                      eps=1e-12,
                      kernel_name="l2_normalize_grad"):
    """
    algorithm: l2_normalize_grad
    calculating dx = (dy - y * sum(dy*y)) / max(||x||, eps)

    Parameters
    ----------
    input_x : dict
        shape and dtype of foward input,
        only support float16, float32
    input_y : dict
        shape and dtype of foward output,
        should be same shape and type as input_x
    input_dy : dict
        shape and dtype of backward input,
        should be same shape and type as input_x
    output_dx: dict
        shape and dtype of backward output,
        should be same shape and type as input_x
    dim : int
        the dimemsion to normalize. Default : 1.
        only support dim=1(for NC/NCHW) and
        dim=[1,4](for NC1HWC0) in this version
    eps : float16/float32
        small value to avoid division by zero.
        Default: 1e-12(float32), 1e-4(float16)
    kernel_name : str
        cce kernel name, default value is l2_normalize_grad

    Returns
    -------
    None
    """
    input_shape_x = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    check_shape(input_shape_x, param_name="input_x")
    check_dtype(input_dtype, ("float16", "float32"), param_name="input_x")
    input_data_x = tvm.placeholder(
        input_shape_x, name="input_data_x", dtype=input_dtype)

    input_shape_y = input_y.get("shape")
    input_dtype_y = input_y.get("dtype").lower()
    check_shape(input_shape_y, param_name="input_y")
    check_dtype(input_dtype_y, input_dtype, param_name="input_y")
    input_data_y = tvm.placeholder(
        input_shape_y, name="input_data_y", dtype=input_dtype)

    input_shape_dy = input_dy.get("shape")
    input_dtype_dy = input_dy.get("dtype").lower()
    check_shape(input_shape_dy, param_name="input_dy")
    check_dtype(input_dtype_dy, input_dtype, param_name="input_dy")
    input_data_dy = tvm.placeholder(
        input_shape_dy, name="input_data_dy", dtype=input_dtype)

    assert (input_shape_x == input_shape_y and input_shape_x == input_shape_dy
            ), "all input/output should have the same shape"

    result = l2_normalize_grad_compute(input_data_x, input_data_y,
                                       input_data_dy, output_dx, dim, eps,
                                       kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(result)

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": [input_data_x, input_data_y, input_data_dy, result]
    }

    te.lang.cce.cce_build_code(sch, config)

