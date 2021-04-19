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
smooth_l1_loss_grad
"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from functools import reduce as functools_reduce
from te.utils.op_utils import *


# pylint: disable=unused-argument,too-many-arguments
@fusion_manager.register("smooth_l1_loss_grad")
def smooth_l1_loss_grad_compute(predict, label, dout, gradient, sigma,
                                kernel_name):
    """
    calculating data

    Parameters
    ----------
    predict : TVM tensor
        the placeholder of predict
    label : TVM tensor
        the placeholder of label
    dout : TVM tensor
        the placeholder of dout
    gradient : dict
        dict of gradient, include keys(shape and dtype)
    sigma : float
        sigma
    kernel_name : str
        kernel name, default value is "smooth_l1_loss_grad"

    Returns
    -------
    output tensor
    """
    dtype = predict.dtype
    shape_input_predict = te.lang.cce.util.shape_to_list(predict.shape)
    shape_input_label = te.lang.cce.util.shape_to_list(label.shape)

    if list(shape_input_predict) != list(shape_input_label):
        shape_input_predict, shape_input_label, shape = \
            broadcast_shapes(shape_input_predict, shape_input_label,
                             param_name_input1="predict",
                             param_name_input2="label")
        predict = te.lang.cce.broadcast(predict, shape, dtype)
        label = te.lang.cce.broadcast(label, shape, dtype)
    out_sub = te.lang.cce.vsub(predict, label)
    # out = sigma if out_sub > sigma
    out_sub_one = te.lang.cce.vmins(out_sub, sigma)
    # out = -sigma if out_sub < -sigma
    out_sub_one_neg_one = te.lang.cce.vmaxs(out_sub_one, -sigma)
    out_sub_one_neg_one_sigma = te.lang.cce.vmuls(out_sub_one_neg_one,
                                                  1 / float(sigma))
    res = te.lang.cce.vmul(out_sub_one_neg_one_sigma, dout)

    return res


# pylint: disable=too-many-arguments,too-many-locals
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 REQUIRED_ATTR_FLOAT, KERNEL_NAME)
def smooth_l1_loss_grad(predict,
                        label,
                        dout,
                        gradient,
                        sigma=1.0,
                        kernel_name="smooth_l1_loss_grad"):
    """
    calculating data
    smooth = x/sigma        if -sigma < x < sigma
             1              if x > sigma
             -1             if x < -sigma
    out = smooth * dout

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of output, should be same shape and type as predict
    gradient : dict
        shape and dtype of output, should be same shape and type as predict
    dout : dict
        shape and dtype of output, should be same shape and type as predict
    sigma : float
        sigma
    kernel_name : str
        kernel name, default value is "smooth_l1_loss_grad"

    Returns
    -------
    None
    """

    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype")
    label_shape = label.get("shape")
    dout_shape = dout.get("shape")
    input_dtype = predict_dtype.lower()
    label_dtype = label.get("dtype").lower()
    dout_dtype = dout.get("dtype").lower()

    util.compare_tensor_dict_key(predict, label, "shape")
    util.compare_tensor_dict_key(predict, dout, "shape")
    util.compare_tensor_dict_key(predict, label, "dtype")
    util.compare_tensor_dict_key(predict, dout, "dtype")
    check_list = ("float16", "float32")
    check_dtype(input_dtype, check_list, param_name="predict")
    check_dtype(label_dtype, check_list, param_name="label")
    check_dtype(dout_dtype, check_list, param_name="dout")

    check_shape(predict_shape, param_name="predict")
    check_shape(label_shape, param_name="label")
    check_shape(dout_shape, param_name="dout")
    shape = (functools_reduce(lambda x, y: x * y, predict_shape[:]),)
    predict_input = tvm.placeholder(
        shape, name="predict_input", dtype=input_dtype)
    label_input = tvm.placeholder(
        shape, name="label_input", dtype=input_dtype)
    dout_input = tvm.placeholder(
        shape, name="dout_input", dtype=input_dtype)
    res = smooth_l1_loss_grad_compute(predict_input, label_input, dout_input,
                                      gradient, sigma, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [predict_input, label_input, dout_input, res]
    }

    te.lang.cce.cce_build_code(sch, config)
