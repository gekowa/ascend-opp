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
softmax_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import *
# pylint: disable=locally-disabled,unused-argument
@fusion_manager.register("softmax_grad")
def softmax_grad_compute(softmax, grad_softmax, grad_x,
                         kernel_name="softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

    Parameters
    ----------
    softmax: TVM tensor
        the placeholder of first input data
    grad_softmax: TVM tensor
        the placeholder of second input data
    grad_x: dict
        the dict of output data
    kernel_name: str
        cce kernel name, default value is "softmax_grad"

    Returns
    -------
    res: TVM tensor
        the result of softmax_grad_compute
    """
    dtype = softmax.dtype
    shape_input1 = te.lang.cce.util.shape_to_list(softmax.shape)
    shape_input2 = te.lang.cce.util.shape_to_list(grad_softmax.shape)
    has_improve_precision = False
    if list(shape_input1) != list(shape_input2):
        shape_input1, shape_input2, shape = broadcast_shapes(shape_input1, shape_input2,
                                                             param_name_input1="softmax",
                                                             param_name_input2="grad_softmax")
        softmax = te.lang.cce.broadcast(softmax, shape, dtype)
        grad_softmax = te.lang.cce.broadcast(grad_softmax, shape, dtype)

    data_vmul = te.lang.cce.vmul(softmax, grad_softmax)
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        data_vmul = te.lang.cce.cast_to(data_vmul, "float32")
        has_improve_precision = True
    data_sum = te.lang.cce.sum(data_vmul, axis=-1, keepdims=True)
    if list(shape_input1) != list(shape_input2):
        data_sum_tmp = te.lang.cce.broadcast(data_sum, shape)
    else:
        data_sum_tmp = te.lang.cce.broadcast(data_sum, shape_input2)
    data_sub = te.lang.cce.vsub(grad_softmax, data_sum_tmp)
    res = te.lang.cce.vmul(softmax, data_sub)
    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def softmax_grad(softmax, grad_softmax, grad_x, kernel_name="softmax_grad"):
    """
    Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

    Parameters
    ----------
    softmax: dict
        shape and dtype of first input, only support float16, float32
    grad_softmax: dict
        shape and dtype of second input, only support float16, float32
    grad_x: dict
        shape and dtype of output data, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "softmax_grad"

    Returns
    -------
    None
    """
    shape_softmax = softmax.get("shape")
    shape_grad_softmax = grad_softmax.get("shape")
    dtype_softmax = softmax.get("dtype")

    util.compare_tensor_dict_key(softmax, grad_softmax, "dtype")
    check_shape(shape_softmax, param_name="softmax")
    check_shape(shape_grad_softmax, param_name="grad_softmax")

    check_list = ("float16", "float32")
    input_dtype = dtype_softmax.lower()

    check_dtype(input_dtype, check_list, param_name="softmax")
    if list(shape_softmax) != list(shape_grad_softmax):
        shape_softmax, shape_grad_softmax, shape_max = \
            broadcast_shapes(shape_softmax, shape_grad_softmax, param_name_input1="softmax", param_name_input2="grad_softmax")

    softmax = tvm.placeholder(shape_softmax, name="softmax", dtype=input_dtype)
    grad_softmaxgrad = tvm.placeholder(shape_grad_softmax,
                                       name="grad_softmaxgrad",
                                       dtype=input_dtype)

    res = softmax_grad_compute(softmax, grad_softmaxgrad, grad_x,
                               kernel_name=kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [softmax, grad_softmaxgrad, res]}
    te.lang.cce.cce_build_code(sch, config)
