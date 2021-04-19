"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

mse_loss_grad
"""

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils
from topi import generic
from topi.cce import util
from functools import reduce


@fusion_manager.register("mse_loss_grad")
def mse_loss_grad_compute(predict, label, dout, grad, reduction="mean", kernel_name="mse_loss_grad"):
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
    grad : dict
        dict of gradient, include keys(shape and dtype)
    reduction : str
        reduce mode, can be 'mean','sum' or 'none'
    kernel_name : str
        kernel name, default value is "mse_loss_grad"

    Returns
    -------
    output tensor
    """
    shape_input_predict = te.lang.cce.util.shape_to_list(predict.shape)

    num = reduce(lambda x, y: x * y, shape_input_predict)
    norm = 2.0 / num if reduction == "mean" else 2.0

    sub_res = te.lang.cce.vsub(predict, label)
    norm_grad = te.lang.cce.vmuls(sub_res, norm)
    grad_res = te.lang.cce.vmul(norm_grad, dout)

    return grad_res


@op_utils.check_op_params(dict, dict, dict, dict, str, str)
def mse_loss_grad(predict, label, dout, grad, reduction="mean", kernel_name="mse_loss_grad"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        shape and dtype of input
    label : dict
        shape and dtype of output, should be same shape and type as predict
    dout : dict
        shape and dtype of output, should be same shape and type as predict
    grad : dict
        shape and dtype of output, should be same shape and type as predict
    reduction : str
        reduce mode,can be 'mean','sum' or 'none'
    kernel_name : str
        kernel name, default value is "mse_loss_grad"

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
    op_utils.check_dtype(input_dtype, check_list)
    op_utils.check_dtype(label_dtype, check_list)
    op_utils.check_dtype(dout_dtype, check_list)

    op_utils.check_shape(predict_shape)
    op_utils.check_shape(label_shape)
    op_utils.check_shape(dout_shape)

    util.check_kernel_name(kernel_name)

    predict_input = tvm.placeholder(predict_shape, name="predict_input", dtype=input_dtype)
    label_input = tvm.placeholder(label_shape, name="label_input", dtype=input_dtype)
    dout_input = tvm.placeholder(dout_shape, name="dout_input", dtype=input_dtype)

    res = mse_loss_grad_compute(predict_input, label_input, dout_input, grad, reduction, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [predict_input, label_input, dout_input, res]}

    te.lang.cce.cce_build_code(schedule, config)
