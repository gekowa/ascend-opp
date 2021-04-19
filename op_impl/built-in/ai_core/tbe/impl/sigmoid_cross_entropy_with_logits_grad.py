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
sigmoid_cross_entropy_with_logits_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from functools import reduce as functools_reduce
from te.utils.op_utils import *

# define a scalar, value = 1
SCALAR_ONE = 1
# define a scalar, value = -1
SCALAR_NEGTIVE_ONE = -1


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("sigmoid_cross_entropy_with_logits_grad")
def sigmoid_cross_entropy_with_logits_grad_compute(
        predict,
        target,
        dout,
        gradient,
        kernel_name):
    """
    calculating sigmoid_cross_entropy_with_logits_grad_compute

    Parameters
    ----------
    predict : TVM tensor
        the output of previous layer
    target : TVM tensor
        label
    dout : TVM tensor
        last gradient
    gradient : TVM tensor
        result after compute
    Returns
    -------
    output tensor
    """
    dtype = predict.dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmul", "float32"):
        predict = te.lang.cce.cast_to(predict, "float32")
        target = te.lang.cce.cast_to(target, "float32")
        dout = te.lang.cce.cast_to(dout, "float32")

    # e^x
    val1 = te.lang.cce.vexp(predict)
    # 1 + e^x
    val2 = te.lang.cce.vadds(val1, tvm.const(SCALAR_ONE, dtype="float32"))

    val3 = te.lang.cce.vdiv(val1, val2)
    # -target
    val4 = te.lang.cce.vmuls(target,
                             tvm.const(SCALAR_NEGTIVE_ONE, dtype="float32"))

    val5 = te.lang.cce.vadd(val3, val4)

    result = te.lang.cce.vmul(val5, dout)

    if dtype == "float16":
        result = te.lang.cce.cast_to(result, dtype)
    return result


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def sigmoid_cross_entropy_with_logits_grad(
        predict,
        target,
        dout,
        gradient,
        kernel_name="sigmoid_cross_entropy_with_logits_grad"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        the output of previous layer
    target : dict
        label
    dout : dict
        last gradient
    gradient : dict
        result after compute
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits_grad"

    Returns
    -------
    None
    """
    check_list = ("float16", "float32")
    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype")
    gradient_dtype = gradient.get("dtype").lower()
    predict_dtype_lower = predict_dtype.lower()
    check_dtype(gradient_dtype, check_list, param_name="gradient")
    check_dtype(predict_dtype_lower, check_list, param_name="predict")

    check_shape(predict_shape, param_name="predict")

    target_shape = target.get("shape")
    target_dtype = target.get("dtype")
    target_dtype_lower = target_dtype.lower()
    check_dtype(target_dtype_lower, check_list, param_name="target")

    check_shape(target_shape, param_name="target")

    dout_shape = dout.get("shape")
    dout_dtype = dout.get("dtype")
    dout_dtype_lower = dout_dtype.lower()
    check_dtype(dout_dtype_lower, check_list, param_name="dout")

    check_shape(dout_shape, param_name="dout")
    util.compare_tensor_dict_key(predict, target, "shape")
    util.compare_tensor_dict_key(predict, dout, "shape")
    shape = (functools_reduce(lambda x, y: x * y, predict_shape[:]),)
    predict_data_input = tvm.placeholder(
        shape, name="predict_data_input", dtype=predict_dtype_lower)
    target_data_input = tvm.placeholder(
        shape, name="target_data_input", dtype=target_dtype_lower)
    dout_data_input = tvm.placeholder(
        shape, name="dout_data_input", dtype=dout_dtype_lower)

    res = sigmoid_cross_entropy_with_logits_grad_compute(
        predict_data_input, target_data_input, dout_data_input, gradient,
        kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name":
            kernel_name,
        "tensor_list": [
            predict_data_input, target_data_input, dout_data_input, res
        ]
    }

    te.lang.cce.cce_build_code(sch, config)
