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
binary_cross_entropy_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import *


# define a scalar, value = 1
SCALAR_ONE = 1
# define a scalar, value = -1
SCALAR_NEGTIVE_ONE = -1
# define a scalar, value = -1
SCALAR_EPS = 1e-12
NoneType = type(None)


# pylint: disable=locally-disabled,unused-argument
# pylint: disable=too-many-arguments,invalid-name,too-many-locals
@fusion_manager.register("binary_cross_entropy_grad")
def binary_cross_entropy_grad_compute(
        x,
        y,
        grad_output,
        weight,
        output,
        reduction,
        kernel_name):
    """
    calculating binary_cross_entropy_grad_compute

    Parameters
    ----------
    x : TVM tensor
        the output of previous layer
    y : TVM tensor
        label
    grad_output : TVM tensor
        last gradient
    weight : None or TVM tensor
        weight for bce
    output : TVM tensor
        result after compute
    reduction : string
        reduce type of bceloss
    kernel_name : str
        kernel name

    Returns
    -------
    output tensor
    """
    shape = te.lang.cce.util.shape_to_list(x.shape)
    dtype = x.dtype
    support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "float32")
    if dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul",
                                                    "float32"):
        x = te.lang.cce.cast_to(x, "float32")
        y = te.lang.cce.cast_to(y, "float32")
        grad_output = te.lang.cce.cast_to(grad_output, "float32")
        if weight is not None:
            weight = te.lang.cce.cast_to(weight, "float32")

    # if grad_output is scaler will boradcast to predict tensor
    # else not changed
    grad_output = te.lang.cce.broadcast(grad_output, shape)

    val1 = te.lang.cce.vsub(x, y)
    if support is True:
        minus_predict = te.lang.cce.vmuls(
            x, tvm.const(SCALAR_NEGTIVE_ONE, dtype="float32"))

        val2_tmp = te.lang.cce.vadds(
            minus_predict, tvm.const(SCALAR_ONE, dtype="float32"))
        val2 = te.lang.cce.vmul(x, val2_tmp)
        val2 = te.lang.cce.vmaxs(val2, tvm.const(SCALAR_EPS, dtype="float32"))
    else:
        minus_predict = te.lang.cce.vmuls(
            x, tvm.const(SCALAR_NEGTIVE_ONE, dtype="float16"))

        val2_tmp = te.lang.cce.vadds(
            minus_predict, tvm.const(SCALAR_ONE, dtype="float16"))
        val2 = te.lang.cce.vmul(x, val2_tmp)
        val2 = te.lang.cce.vmaxs(val2, tvm.const(SCALAR_EPS, dtype="float16"))
    result = te.lang.cce.vdiv(val1, val2)
    if weight is not None:
        result = te.lang.cce.vmul(weight, result)
    result = te.lang.cce.vmul(grad_output, result)

    if reduction == "mean":
        reduce_elts = 1.0
        for i in shape:
            reduce_elts *= i
        result = te.lang.cce.vmuls(result, reduce_elts**(-1))

    if dtype == "float16":
        result = te.lang.cce.cast_to(result, dtype)

    return result


# pylint: disable=invalid-name,too-many-locals,too-many-statements
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, OPTION_INPUT,
                 REQUIRED_OUTPUT, OPTION_ATTR_STR, KERNEL_NAME)
def binary_cross_entropy_grad(
        x,
        y,
        grad_output,
        weight,
        output,
        reduction="mean",
        kernel_name="binary_cross_entropy_grad"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        the predict of previous layer shape and dtype
    y : dict
        target label
    grad_output : dict
        last gradient/dout, if scalar, reshape first
    weight : None or TVM tensor
        weight for bce
    output : dict
        result gradient after compute
    reduction : string
        reduce type of bceloss, must be "none", "sum" or "mean"
    kernel_name : str
        kernel name, default value is "binary_cross_entropy_grad"

    Returns
    -------
    None
    """
    predict_shape = x.get("shape")
    predict_dtype = x.get("dtype")
    predict_dtype_lower = predict_dtype.lower()
    check_shape(predict_shape, param_name="x")
    shape_size = util.check_tensor_shape_size(predict_shape)
    predict_data_input = tvm.placeholder(
        [shape_size], name="predict_data_input", dtype=predict_dtype_lower)

    target_shape = y.get("shape")
    target_dtype = y.get("dtype")
    target_dtype_lower = target_dtype.lower()
    check_shape(target_shape, param_name="y")
    shape_size = util.check_tensor_shape_size(target_shape)
    target_data_input = tvm.placeholder(
        [shape_size], name="target_data_input", dtype=target_dtype_lower)

    dout_shape = grad_output.get("shape")
    dout_dtype = grad_output.get("dtype")
    dout_dtype_lower = dout_dtype.lower()
    check_shape(dout_shape, param_name="grad_output")

    # if dout is scaler get the boardcast shape, else not chenged
    _, dout_shape, _ = broadcast_shapes(target_shape, dout_shape, param_name_input1="y", param_name_input2="grad_output")
    shape_size = util.check_tensor_shape_size(dout_shape)
    dout_data_input = tvm.placeholder(
        [shape_size], name="dout_data_input", dtype=dout_dtype_lower)

    check_list = ("float16", "float32")
    check_dtype(predict_dtype_lower, check_list, param_name="x")

    if predict_shape != target_shape:
        raise RuntimeError("predictx(x) and target(y)"
                           " should have the same shape")

    if not (predict_dtype_lower == target_dtype_lower and
            predict_dtype_lower == dout_dtype_lower):
        raise RuntimeError("all input should have the same dtype")

    weight_data_input = None
    if weight is not None:
        weight_shape = weight.get("shape")
        weight_dtype = weight.get("dtype")
        weight_dtype_lower = weight_dtype.lower()
        check_shape(weight_shape, param_name="weight")
        shape_size = util.check_tensor_shape_size(weight_shape)
        weight_data_input = tvm.placeholder(
            [shape_size], name="weight_data_input", dtype=weight_dtype_lower)

        if predict_shape != weight_shape:
            raise RuntimeError("predictx(x) and weight"
                               " should have the same shape")
        if predict_dtype_lower != weight_dtype_lower:
            raise RuntimeError("predictx(x) and weight"
                               " should have the same dtype")
    if reduction not in ("mean", "sum", "none"):
        raise RuntimeError("reduction type should in mean/sum/none")

    res = binary_cross_entropy_grad_compute(
        predict_data_input, target_data_input,
        dout_data_input, weight_data_input, output,
        reduction, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    if weight is not None:
        config = {
            "print_ir": False,
            "name":
                kernel_name,
            "tensor_list": [
                predict_data_input, target_data_input,
                dout_data_input, weight_data_input, res
            ]
        }
    else:
        config = {
            "print_ir": False,
            "name":
                kernel_name,
            "tensor_list": [
                predict_data_input, target_data_input, dout_data_input, res
            ]
        }

    te.lang.cce.cce_build_code(sch, config)

