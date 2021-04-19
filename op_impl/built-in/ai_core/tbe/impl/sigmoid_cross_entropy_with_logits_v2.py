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

sigmoid_cross_entropy_with_logitsv2
"""

import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util
from te.utils import op_utils
from functools import reduce


def _shape_check(shape):
    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)


def _5hd_input_C_axis_limit(ori_shape_input_1, ori_shape_input_2, index):
    if ori_shape_input_1[index] != ori_shape_input_2[index]:
        raise RuntimeError("if current format is NC1HWC0, dim C of input should be same")


def _5hd_input_ori_shape_check(input_1, input_2):
    ori_shape_input_1 = input_1.get("ori_shape")
    ori_shape_input_2 = input_2.get("ori_shape")
    if len(ori_shape_input_1) != len(ori_shape_input_2):
        raise RuntimeError("if format is NC1HWC0, dims of weight or pos_weight should be same as predict")

    ori_format_input_1 = input_1.get("ori_format")
    if ori_format_input_1 == 'NCHW':
        if len(ori_shape_input_1) <= 3:
            _5hd_input_C_axis_limit(ori_shape_input_1, ori_shape_input_2, 0)
        elif len(ori_shape_input_1) == 4:
            _5hd_input_C_axis_limit(ori_shape_input_1, ori_shape_input_2, 1)

    if ori_format_input_1 == 'NHWC':
        if len(ori_shape_input_1) == 3:
            _5hd_input_C_axis_limit(ori_shape_input_1, ori_shape_input_2, 2)
        elif len(ori_shape_input_1) == 4:
            _5hd_input_C_axis_limit(ori_shape_input_1, ori_shape_input_2, 3)


def _broadcast_shape_check(input_shape, target_shape):
    try:
        util.produce_shapes(input_shape, target_shape)
    except RuntimeError:
        raise RuntimeError("input_shape can't be broadcast to target_shape")


def _set_reduce_axis(reduce_tensor):
    shape_reduce = te.lang.cce.util.shape_to_list(reduce_tensor.shape)
    axis_d = []
    for i, _ in enumerate(shape_reduce):
        axis_d.append(i)
    axis_d = util.axis_check(len(shape_reduce), axis_d)
    return axis_d


def _optional_input_tensor_apply(input_dict, tensor_name, predict):
    data_input = None
    if input_dict is not None:
        shape_input = input_dict.get("shape")
        dtype_input = input_dict.get("dtype").lower()
        shape_predict = predict.get("shape")
        format_predict = predict.get("format")
        _shape_check(shape_input)
        if format_predict == "NC1HWC0":
            _5hd_input_ori_shape_check(predict, input_dict)
        _broadcast_shape_check(shape_input, shape_predict)
        dis_len = len(shape_predict) - len(shape_input)
        shape_weight_append1 = tuple([1 for i in range(dis_len)]) + shape_input
        data_input = tvm.placeholder(shape_weight_append1, name=tensor_name, dtype=dtype_input)
    return data_input


def _pos_weight_compute(pos_weight, dtype_input, target, shape_target, ln_2, same_shape,
                        is_last_dim_shape_same):
    # log_weight is equal to (pos_weight - 1) * y + 1
    pos_weight = te.lang.cce.cast_to(pos_weight, dtype_input)
    shape_pos_weight = te.lang.cce.util.shape_to_list(pos_weight.shape)
    if shape_target != shape_pos_weight:
        same_shape = False
        is_last_dim_shape_same = False if shape_target[-1] != shape_pos_weight[-1] else True
        pos_weight_broadcast = te.lang.cce.broadcast(pos_weight, shape_target)
        log_weight = te.lang.cce.vadds(pos_weight_broadcast, tvm.const(-1, dtype=dtype_input))
    else:
        log_weight = te.lang.cce.vadds(pos_weight, tvm.const(-1, dtype=dtype_input))
    log_weight = te.lang.cce.vmul(log_weight, target)
    log_weight = te.lang.cce.vadds(log_weight, tvm.const(1, dtype=dtype_input))
    ln_2 = te.lang.cce.vmul(log_weight, ln_2)
    return ln_2, same_shape, is_last_dim_shape_same


def _weight_compute(weight, dtype_input, shape_target, ln, same_shape, is_last_dim_shape_same):
    # ln is equal to: ln * weight
    weight = te.lang.cce.cast_to(weight, dtype_input)
    shape_weight = te.lang.cce.util.shape_to_list(weight.shape)
    if shape_target != shape_weight:
        same_shape = False
        is_last_dim_shape_same = False if shape_target[-1] != shape_weight[-1] else True
        weight_broadcast = te.lang.cce.broadcast(weight, shape_target)
        ln = te.lang.cce.vmul(ln, weight_broadcast)
    else:
        ln = te.lang.cce.vmul(ln, weight)
    return ln, same_shape, is_last_dim_shape_same


def _reduce_operation_spec(ln, reduction, ori_shape):
    dtype_ln = ln.dtype
    if reduction == 'mean':
        element_num = reduce(lambda x, y: x * y, ori_shape)
        res = te.lang.cce.vmuls(ln, tvm.const(element_num ** -1, dtype=dtype_ln))
        res = te.lang.cce.sum(res, axis=-1, keepdims=False)
        axis_d = _set_reduce_axis(res)
        res = te.lang.cce.sum(res, axis=axis_d, keepdims=False)
        return res
    elif reduction == 'sum':
        res = te.lang.cce.sum(ln, axis=-1, keepdims=False)
        axis_d = _set_reduce_axis(res)
        res = te.lang.cce.sum(ln, axis=axis_d, keepdims=False)
        return res
    else:
        return ln


def _reduce_operation(ln, reduction, ori_shape):
    dtype_ln = ln.dtype
    axis_d = _set_reduce_axis(ln)
    if reduction == 'mean':
        element_num = reduce(lambda x, y: x * y, ori_shape)
        res = te.lang.cce.vmuls(ln, tvm.const(element_num ** -1, dtype=dtype_ln))
        res = te.lang.cce.sum(res, axis=axis_d, keepdims=False)
        return res
    elif reduction == 'sum':
        res = te.lang.cce.sum(ln, axis=axis_d, keepdims=False)
        return res
    else:
        return ln


def _get_tensor_list(pos_weight, weight, data_predict, data_target, data_weight, data_pos_weight, res):
    if pos_weight is None and weight is None:
        tensor_list = [data_predict, data_target, res]
    elif pos_weight is None and weight is not None:
        tensor_list = [data_predict, data_target, data_weight, res]
    elif pos_weight is not None and weight is None:
        tensor_list = [data_predict, data_target, data_pos_weight, res]
    else:
        tensor_list = [data_predict, data_target, data_weight, data_pos_weight, res]
    return tensor_list


def sigmoid_cross_entropy_with_logitsv2_compute(predict, target, weight, pos_weight, reduction, ori_shape):
    """
    Parameters
    ----------
        predict : TVM tensor, the placeholder of predict
        target: TVM tensor, the placeholder of target
        weight: TVM tensor, the placeholder of weight
        pos_weight: TVM tensor, the placeholder of pos_weight
        reduction : str, specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        ori_shape: it is the ori shape of predict, used to count the number of elements
    Returns
    -------
        res: result of loss value
    """

    dtype_input = "float32"
    if predict.dtype.lower() == "float16":
        predict = te.lang.cce.cast_to(predict, dtype_input)
        target = te.lang.cce.cast_to(target, dtype_input)

    reverse_predict = te.lang.cce.vmuls(predict, tvm.const(-1, dtype=dtype_input))      # -x
    max_val = te.lang.cce.vmaxs(reverse_predict, tvm.const(0, dtype=dtype_input))       # max(-x, 0)
    reverse_max_val = te.lang.cce.vmuls(max_val, tvm.const(-1, dtype=dtype_input))      # -max_val
    reverse_target = te.lang.cce.vmuls(target, tvm.const(-1, dtype=dtype_input))        # -y
    ln_0 = te.lang.cce.vadds(reverse_target, tvm.const(1, dtype=dtype_input))           # (1-y)
    ln_1 = te.lang.cce.vmul(ln_0, predict)                                              # (1 - y) * x

    add_x_vmax = te.lang.cce.vadd(reverse_predict, reverse_max_val)                     # -x-max_val
    exp_1 = te.lang.cce.vexp(reverse_max_val)                                           # e^-max_val
    exp_2 = te.lang.cce.vexp(add_x_vmax)                                                # e^(-x-max_val)
    exp_add = te.lang.cce.vadd(exp_1, exp_2)                                            # e^-max_val + e^(-x-max_val)
    log_ = te.lang.cce.vlog(exp_add)                                                    # log(e^ + e^)
    ln_2 = te.lang.cce.vadd(log_, max_val)                                              # log(e^ + e^) + max_val

    same_shape = True
    is_last_dim_shape_same = True
    shape_target = te.lang.cce.util.shape_to_list(target.shape)
    cur_element_num = reduce(lambda x, y: x * y, shape_target)
    if pos_weight is not None:
        ln_2, same_shape, is_last_dim_shape_same = _pos_weight_compute(pos_weight, dtype_input, target, shape_target,
                                                                       ln_2, same_shape, is_last_dim_shape_same)
    ln = te.lang.cce.vadd(ln_1, ln_2)

    if weight is not None:
        ln, same_shape, is_last_dim_shape_same = _weight_compute(weight, dtype_input, shape_target, ln, same_shape,
                                                                 is_last_dim_shape_same)

    # when shape is not same (mean that shape need to be broadcast) and last dim is same
    # and shape is not align to 32b, it should process specially.
    if not same_shape and is_last_dim_shape_same and (cur_element_num % 16 != 0):
        res = _reduce_operation_spec(ln, reduction, ori_shape)
    else:
        res = _reduce_operation(ln, reduction, ori_shape)

    return res


@op_utils.check_op_params(dict, dict, dict, dict, dict, str, str)
def sigmoid_cross_entropy_with_logits_v2(predict, target, weight, pos_weight, loss, reduction="mean",
                                         kernel_name="sigmoid_cross_entropy_with_logitsv2"):
    """
    Function: it measures Binary Cross Entropy between target and output logits.
              this loss combines a Sigmoid layer and the BCELoss in one single class.
    Parameters
    ----------
        predict: dict, shape_predict and dtype of input, required
        target: dict, shape_target and dtype of input, required
                a manual rescaling weight given to the loss of each batch element
        weight: dict, shape_weight and dtype of input, optional
        pos_weight: dict, shape_pos_weight and dtype of input, optional
        loss: dict, shape_loss and dtype of output, should be same shape_predict
              and type as input
        reduction: str, specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        kernel_name : str, kernel name
                      default value is "sigmoid_cross_entropy_with_logitsv2"
    Returns
    -------
        None
    """

    util.check_kernel_name(kernel_name)
    shape_predict = predict.get("shape")
    dtype_input = predict.get("dtype").lower()
    shape_target = target.get("shape")
    dtype_target = target.get("dtype").lower()
    _shape_check(shape_predict)
    _shape_check(shape_target)

    if shape_predict != shape_target:
        raise RuntimeError("target size must be the same as predict size")

    format_predict = predict.get("format")
    predict_ori_shape = predict.get("ori_shape")
    if format_predict == "NC1HWC0":
        if len(predict_ori_shape) > 4:
            raise RuntimeError("Dim of shape bigger than 4 is not support when format is NC1HWC0")
        if len(shape_predict) != 5:
            raise RuntimeError("Dim of shape should be 5 when format is NC1HWC0")

    data_predict = tvm.placeholder(shape_predict, name="data_predict", dtype=dtype_input)
    data_target = tvm.placeholder(shape_target, name="data_target", dtype=dtype_target)

    data_weight = _optional_input_tensor_apply(weight, "data_weight", predict)
    data_pos_weight = _optional_input_tensor_apply(pos_weight, "data_pos_weight", predict)

    check_reduct = {"mean", "sum", "none"}
    if reduction not in check_reduct:
        raise ValueError("reduction should be one of the ['mean', 'sum', 'none'], \
                         {} is not a valid value for reduction".format(reduction))

    res = sigmoid_cross_entropy_with_logitsv2_compute(data_predict, data_target, data_weight, data_pos_weight,
                                                      reduction, predict_ori_shape)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    tensor_list = _get_tensor_list(pos_weight, weight, data_predict, data_target, data_weight, data_pos_weight, res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(schedule, config)
