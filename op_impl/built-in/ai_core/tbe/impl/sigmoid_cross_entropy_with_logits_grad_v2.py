import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils import op_utils
from topi.cce import util
from functools import reduce


def _broadcast_shape_check(input_shape, target_shape):
    try:
        util.produce_shapes(input_shape, target_shape)
    except RuntimeError:
        raise RuntimeError("input_shape can't be broadcast to target_shape")


@fusion_manager.register("sigmoid_cross_entropy_with_logits_grad_v2")
def sigmoid_cross_entropy_with_logits_grad_v2_compute(predict, target, dout, weight, pos_weight, reduction="mean"):
    """
    :param predict: TVM tensor, the placeholder of predict
    :param target: TVM tensor, the placeholder of target
    :param dout: TVM tensor, the placeholder of dout
    :param weight: TVM tensor, the placeholder of weight
    :param pos_weight: TVM tensor, the placeholder of pos_weight
    :param reduction: str, specifies the reduction mode :'none' | 'mean' | 'sum'
    :return: TVM tensor
    """
    predict_shape = te.lang.cce.util.shape_to_list(predict.shape)
    predict_dtype = predict.dtype

    precision_dtype = "float32"

    if predict.dtype.lower() == "float16":
        predict = te.lang.cce.cast_to(predict, precision_dtype)
        target = te.lang.cce.cast_to(target, precision_dtype)

    # calculate sigmoid(predict)
    exp_predict = te.lang.cce.vexp(predict)
    exp_add1 = te.lang.cce.vadds(exp_predict, tvm.const(1, precision_dtype))
    sigmoid_tmp = te.lang.cce.vdiv(exp_predict, exp_add1)
    sigmoid_res = te.lang.cce.cast_to(sigmoid_tmp, precision_dtype)

    # calculate the result of gradient = ((log_weight + 1 - target) * sigmoid(predict) - log_weight) * dout
    if pos_weight is not None:
        pos_weight_shape = te.lang.cce.util.shape_to_list(pos_weight.shape)
        if pos_weight_shape != predict_shape:
            _, _, broadcast_pos_shape = util.produce_shapes(pos_weight_shape, predict_shape)
            pos_weight = te.lang.cce.broadcast(pos_weight, broadcast_pos_shape, precision_dtype)

        log_weight = te.lang.cce.vmul(pos_weight, target)
        weight_tmp = te.lang.cce.vadds(log_weight, tvm.const(1, precision_dtype))
        weight_sub = te.lang.cce.vsub(weight_tmp, target)
        grad_tmp = te.lang.cce.vmul(weight_sub, sigmoid_res)
        grad_cur = te.lang.cce.vsub(grad_tmp, log_weight)
        grad_output = te.lang.cce.vmul(grad_cur, dout)
    else:
        grad_cur = te.lang.cce.vsub(sigmoid_res, target)
        grad_output = te.lang.cce.vmul(grad_cur, dout)

    # calculate the result of gradient = gradient * weight
    if weight is not None:
        weight_shape = te.lang.cce.util.shape_to_list(weight.shape)
        if weight_shape != predict_shape:
            _, _, broadcast_weight_shape = util.produce_shapes(weight_shape, predict_shape)
            weight = te.lang.cce.broadcast(weight, broadcast_weight_shape, precision_dtype)

        grad_output = te.lang.cce.vmul(grad_output, weight)

    # calculate the result of gradient = gradient / num
    if reduction == "mean":
        num = reduce(lambda x, y: x * y, predict_shape)
        norm = 1.0 / num
        grad_output = te.lang.cce.vmuls(grad_output, norm)

    grad_output = te.lang.cce.cast_to(grad_output, predict_dtype)
    return grad_output


def optional_weight(tensor_list, predict_shape, dtype_list, weight, pos_weight):
    weight_data = None
    pos_weight_data = None
    if weight is not None:
        weight_shape = weight.get("shape")
        weight_dtype = weight.get("dtype").lower()
        op_utils.check_dtype(weight_dtype, dtype_list)
        _broadcast_shape_check(weight_shape, predict_shape)

        weight_shape = tuple([1] * (len(predict_shape) - len(weight_shape))) + tuple(weight_shape)
        weight_data = tvm.placeholder(weight_shape, weight_dtype, name="weight_data")
        tensor_list.append(weight_data)

    if pos_weight is not None:
        pos_weight_shape = pos_weight.get("shape")
        pos_weight_dtype = pos_weight.get("dtype").lower()

        op_utils.check_dtype(pos_weight_dtype, dtype_list)
        _broadcast_shape_check(pos_weight_shape, predict_shape)

        pos_weight_shape = tuple([1] * (len(predict_shape) - len(pos_weight_shape))) + tuple(pos_weight_shape)
        pos_weight_data = tvm.placeholder(pos_weight_shape, pos_weight_dtype,
                                          name="pos_weight_data")
        tensor_list.append(pos_weight_data)

    return weight_data, pos_weight_data


@op_utils.check_op_params(dict, dict, dict, dict, dict, dict, str, str)
def sigmoid_cross_entropy_with_logits_grad_v2(predict, target, dout, weight, pos_weight, gradient,
                                              reduction="mean",
                                              kernel_name="sigmoid_cross_entropy_with_logits_grad_v2"):
    """
    Function: it measures the gradient of Binary Cross Entropy With Logits.
    -----------
    :param predict: dict, shape and dtype of input, required
    :param target: dict,shape and dtype of target, should be same shape and type as predict, required
    :param dout: dict,shape and dtype of dout, should be same shape and type as predict, required
    :param weight: dict,shape and dtype of weight, should be same shape and type as predict, optional
    :param pos_weight: dict,shape and dtype of pos_weight, should be same shape and type as predict, optional
    :param gradient: dict,shape and dtype of target, should be same shape and type as predict, required
    :param reduction: str, specifies the reduction mode: 'none' | 'mean' | 'sum', default to 'mean'
    :param kernel_name: str, kernel name, default to 'sigmoid_cross_entropy_with_logits_grad_v2'
    :return: None
    """
    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype").lower()
    target_shape = target.get("shape")
    target_dtype = target.get("dtype").lower()
    dout_shape = dout.get("shape")
    dout_dtype = dout.get("dtype").lower()

    util.compare_tensor_dict_key(predict, target, "shape")
    util.compare_tensor_dict_key(predict, dout, "shape")
    util.compare_tensor_dict_key(predict, target, "dtype")
    util.compare_tensor_dict_key(predict, dout, "dtype")

    dtype_list = ["float16", "float32"]
    op_utils.check_dtype(predict_dtype, dtype_list)
    op_utils.check_shape(predict_shape)

    reduction_list = ["none", "mean", "sum"]
    if reduction not in reduction_list:
        raise RuntimeError("reduction should be one of ['none','mean','sum']")

    util.check_kernel_name(kernel_name)

    tensor_list = []

    predict_data = tvm.placeholder(predict_shape, predict_dtype, name="predict_data")
    target_data = tvm.placeholder(target_shape, target_dtype, name="target_data")
    dout_data = tvm.placeholder(dout_shape, dout_dtype, name="dout_data")

    tensor_list.append(predict_data)
    tensor_list.append(target_data)
    tensor_list.append(dout_data)

    weight_data, pos_weight_data = optional_weight(tensor_list, predict_shape, dtype_list, weight, pos_weight)

    res = sigmoid_cross_entropy_with_logits_grad_v2_compute(predict_data, target_data, dout_data, weight_data,
                                                            pos_weight_data, reduction)

    tensor_list.append(res)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(schedule, config)
