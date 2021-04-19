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
smooth_l1_loss
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import *


# pylint: disable=locally-disabled,unused-argument,too-many-locals
@fusion_manager.register("smooth_l1_loss")
def smooth_l1_loss_compute(input_predict,
                           input_label,
                           output_loss,
                           sigma,
                           kernel_name="smooth_l1_loss"):
    """
    calculating data

    Parameters
    ----------
    input_predict : TVM tensor
        the placeholder of input_predict
    input_label : TVM tensor
        the placeholder of input_label
    output_loss : dict
        dict of output_loss, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "smooth_l1_loss"

    Returns
    -------
    output tensor
    """

    input_dtype = input_predict.dtype
    half_const = tvm.const(0.5, dtype=input_dtype)
    half_const_tensor = te.lang.cce.broadcast(half_const, input_predict.shape)
    one_const = tvm.const(1.0, dtype=input_dtype)
    one_const_tensor = te.lang.cce.broadcast(one_const, input_predict.shape)

    sigma_scalar = tvm.const(sigma, dtype=input_dtype)

    input_sub_res = te.lang.cce.vsub(input_predict, input_label)

    method_one_res = te.lang.cce.vmul(
        te.lang.cce.vmuls(input_sub_res, half_const), input_sub_res)
    method_one_res = te.lang.cce.vmuls(method_one_res, 1 / sigma_scalar)
    predict_label_sub_abs = te.lang.cce.vabs(input_sub_res)
    method_two_res = te.lang.cce.vsub(predict_label_sub_abs,
                                      te.lang.cce.vmuls(half_const_tensor,
                                                        sigma_scalar))

    is_method_one_res = te.lang.cce.vcmpsel(predict_label_sub_abs,
                                            sigma_scalar,
                                            'lt', 1.0, 0.0)
    is_method_two_res = te.lang.cce.vsub(one_const_tensor, is_method_one_res)
    method_one_get_res = te.lang.cce.vmul(method_one_res, is_method_one_res)
    method_two_get_res = te.lang.cce.vmul(method_two_res, is_method_two_res)
    res = te.lang.cce.vadd(method_one_get_res, method_two_get_res)
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_FLOAT, KERNEL_NAME)
def smooth_l1_loss(predict,
                   label,
                   loss,
                   sigma=1.0,
                   kernel_name="smooth_l1_loss"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of input
    loss : dict
        shape and dtype of output,
        should be same shape and type as input
    sigma: float
        sigma,default value is 1
    kernel_name : str
        kernel name, default value is "smooth_l1_loss"

    Returns
    -------
    None
    """

    check_list = ("float16", "float32")
    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype")
    input_predict_dtype = dtype_predict.lower()
    check_dtype(input_predict_dtype, check_list, param_name="predict")

    shape_label = label.get("shape")
    dtype_label = label.get("dtype")
    input_label_dtype = dtype_label.lower()
    dtype_loss = loss.get("dtype").lower()
    check_dtype(input_label_dtype, check_list, param_name="label")
    check_dtype(dtype_loss, check_list, param_name="loss")

    util.compare_tensor_dict_key(predict, label, "shape")
    check_shape(shape_predict, param_name="predict")
    check_shape(shape_label, param_name="label")
    check_list = ("float16", "float32")
    check_dtype(input_predict_dtype, check_list, param_name="predict")
    shape_predict, shape_label = \
        refine_shapes_for_broadcast(shape_predict, shape_label)
    input_predict = tvm.placeholder(
        shape_predict, name="predict", dtype=input_predict_dtype)
    input_label = tvm.placeholder(
        shape_label, name="label", dtype=input_label_dtype)
    res = smooth_l1_loss_compute(input_predict, input_label, loss, sigma,
                                 kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [input_predict, input_label, res]
    }

    te.lang.cce.cce_build_code(sch, config)
