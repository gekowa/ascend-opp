import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils import op_utils
from functools import reduce


@fusion_manager.register("l1_loss_grad")
def l1_loss_grad_compute(grad_out, predict, target, y, reduction="mean", kernel_name="l1_loss_grad"):
    predict_dtype = predict.dtype.lower()
    zero_tensor = te.lang.cce.vmuls(predict, tvm.const(0, dtype=predict_dtype))
    one_tensor = te.lang.cce.vadds(zero_tensor, tvm.const(1, dtype=predict_dtype))
    neg_one_tensor = te.lang.cce.vadds(zero_tensor, tvm.const(-1, dtype=predict_dtype))
    # if predict is equal or bigger than target, the sign will be given 1; else -1
    sign = te.lang.cce.vcmpsel(predict, target, "gt", one_tensor, neg_one_tensor)
    # rectify sign to 0 when predict equal to target
    sign = te.lang.cce.vcmpsel(predict, target, "eq", zero_tensor, sign)
    grad_shape = te.lang.cce.util.shape_to_list(grad_out.shape)
    n = reduce(lambda x, y: x * y, grad_shape)
    norm = grad_out
    # if choose "mean", grad_out should divide over n
    if reduction == "mean":
        norm = te.lang.cce.vmuls(norm, tvm.const(1 / n, dtype=predict_dtype))
    # chain multiplication to get the gradient of L1 with respect to weights(grad_out)
    res = te.lang.cce.vmul(sign, norm)
    return res


@op_utils.check_op_params(dict, dict, dict, dict, str, str)
def l1_loss_grad(grads, predict, label, y, reduction="mean", kernel_name="l1_loss_grad"):
    """
    Parameters
    ----------
    grads : dict
        shape and dtype of grad_out as input
    predict : dict
        shape and dtype of predict as input, should be same shape and type as grads
    label : dict
        shape and dtype of label as input, should be same shape and type as grads
    y : dict
        shape and dtype of output, should be same shape and type as grads
    reduction: string
        reduction name, default value is "mean"
    kernel_name : str
        kernel name, default value is "l1_loss_grad"

    Returns
    -------
    None
    """
    dtype_list = ["float16", "float32"]
    reduction_list = ["none", "mean", "sum"]
    grads_data_type = grads.get("dtype").lower()
    grads_shape = grads.get("shape")
    predict_data_type = predict.get("dtype").lower()
    predict_shape = predict.get("shape")
    label_data_type = label.get("dtype").lower()
    label_shape = label.get("shape")

    op_utils.check_dtype(grads_data_type, dtype_list)
    op_utils.check_dtype(predict_data_type, dtype_list)
    op_utils.check_dtype(label_data_type, dtype_list)

    op_utils.check_shape(grads_shape)
    op_utils.check_shape(predict_shape)
    op_utils.check_shape(label_shape)

    util.compare_tensor_dict_key(grads, predict, "shape")
    util.compare_tensor_dict_key(grads, label, "shape")
    util.compare_tensor_dict_key(grads, predict, "dtype")
    util.compare_tensor_dict_key(grads, label, "dtype")

    if reduction not in reduction_list:
        raise RuntimeError("reduction should be one of ['none','mean','sum']")
    grads = tvm.placeholder(grads_shape, dtype=grads_data_type, name="grads")
    predict = tvm.placeholder(predict_shape, dtype=predict_data_type, name="predict")
    label = tvm.placeholder(label_shape, dtype=label_data_type, name="label")

    res = l1_loss_grad_compute(grads, predict, label, y, reduction=reduction, kernel_name="l1_loss_grad")

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [grads, predict, label, res]}
    te.lang.cce.cce_build_code(schedule, config)
