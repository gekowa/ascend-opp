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
sigmoid_cross_entropy_with_logits
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as tbe_platform
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import *

# define a scalar, value = 1
SCALAR_ONE = 1
# define a scalar, value = 0
SCALAR_ZREO = 0


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("sigmoid_cross_entropy_with_logits")
def sigmoid_cross_entropy_with_logits_compute(predict,
                                              target,
                                              loss,
                                              kernel_name):
    """
    calculating data

    Parameters
    ----------
    predict : TVM tensor
        the placeholder of predict
    target : TVM tensor
        the placeholder of target
    loss : dict
        dict of loss, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits"

    Returns
    -------
    output tensor
    """
    predict_dtype = predict.dtype
    target_dtype = target.dtype
    if predict_dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vsub", "float32"):
        predict = te.lang.cce.cast_to(predict, "float32")
    if target_dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmul", "float32"):
        target = te.lang.cce.cast_to(target, "float32")

    dtype_predict = predict.dtype
    shape_predict = te.lang.cce.util.shape_to_list(predict.shape)

    const_zero = tvm.const(SCALAR_ZREO, dtype=dtype_predict)
    max_predict_zero = te.lang.cce.vmaxs(predict, const_zero)

    abs_predict = te.lang.cce.vabs(predict)
    const_zero_broadcast = te.lang.cce.broadcast(const_zero, shape_predict)
    reverse_abs_predict = te.lang.cce.vsub(const_zero_broadcast, abs_predict)
    vexp_predict = te.lang.cce.vexp(reverse_abs_predict)
    const_one = tvm.const(SCALAR_ONE, dtype=dtype_predict)
    vadds_res = te.lang.cce.vadds(vexp_predict, const_one)
    vlog_res = te.lang.cce.vlog(vadds_res, priority_flag=1)
    vmul_res = te.lang.cce.vmul(predict, target)
    res = te.lang.cce.vsub(vlog_res, vmul_res)
    loss = te.lang.cce.vadd(res, max_predict_zero)

    if predict_dtype == "float16":
        loss = te.lang.cce.cast_to(loss, "float16")

    return loss


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def sigmoid_cross_entropy_with_logits(
        predict, target, loss,
        kernel_name="sigmoid_cross_entropy_with_logits"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of predict
    target : dict
        shape and dtype of target
    loss : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits"

    Returns
    -------
    None
    """
    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype")
    input_dtype_predict = dtype_predict.lower()
    check_shape(shape_predict, param_name="predict")

    shape_target = target.get("shape")
    dtype_target = target.get("dtype")
    input_dtype_target = dtype_target.lower()
    check_shape(shape_target, param_name="target")


    check_list = ("float16", "float32")
    check_dtype(input_dtype_predict, check_list, param_name="predict")
    check_dtype(input_dtype_target, check_list, param_name="target")
    shape_predict, shape_target = \
        refine_shapes_for_broadcast(shape_predict, shape_target)
    data_predict = tvm.placeholder(shape_predict,
                                   name="data_predict",
                                   dtype=input_dtype_predict)
    data_target = tvm.placeholder(shape_target,
                                  name="data_target",
                                  dtype=input_dtype_target)
    loss = sigmoid_cross_entropy_with_logits_compute(data_predict,
                                                     data_target,
                                                     loss,
                                                     kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(loss)

    config = {"name": kernel_name,
              "tensor_list": [data_predict, data_target, loss]}

    te.lang.cce.cce_build_code(sch, config)
