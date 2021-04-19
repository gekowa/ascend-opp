import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils import op_utils
from functools import reduce


def l1_loss_compute(predict, label, reduction):
    """
    :param predict: TVM tensor
        the placeholder of predict
    :param label: TVM tensor
        the placeholder of label
    :param reduction: str
        reduce mode, can be 'mean','sum' or 'none'
    :return: output tensor
    """
    predict_shape = te.lang.cce.util.shape_to_list(predict.shape)

    # float16 cast to float32
    precision_dtype = "float32"
    predict_dtype = predict.dtype.lower()
    if predict_dtype == "float16":
        predict = te.lang.cce.cast_to(predict, precision_dtype)
        label = te.lang.cce.cast_to(label, precision_dtype)

    # calculate the result of loss = |predict-label|
    loss = te.lang.cce.vabs(te.lang.cce.vsub(predict, label))

    # calculate the result of sum(loss)
    if reduction == "sum":
        dims = list(range(len(predict_shape)))
        loss = te.lang.cce.sum(loss, dims)

    # calculate the result of mean(loss)
    if reduction == "mean":
        dims = list(range(len(predict_shape)))
        sum_loss = te.lang.cce.sum(loss, dims)
        num = reduce(lambda x, y: x * y, predict_shape)
        norm = 1.0 / num
        loss = te.lang.cce.vmuls(sum_loss, norm)

    loss = te.lang.cce.cast_to(loss, predict_dtype)
    return loss


@fusion_manager.register("lp_loss")
def lp_loss_compute(predict, label, p, reduction, kernel_name="lp_loss"):
    """
    :param predict: TVM tensor
        the placeholder of predict
    :param label: TVM tensor
        the placeholder of label
    :param p: int
        decides which loss to compute, now the p only can be 1 to compute l1_loss
    :param reduction: str
        reduce mode,can be 'mean','sum' or 'none'
    :param kernel_name: ernel name, default value is "lp_loss"
    :return: output tensor
    """
    res = l1_loss_compute(predict, label, reduction)
    return res


@op_utils.check_op_params(dict, dict, dict, int, str, str)
def lp_loss(predict, label, y, p, reduction="mean", kernel_name="lp_loss"):
    """
    :param predict: dict
        shape and dtype of input
    :param label: dict
        shape and dtype of label, should be same shape and type as predict
    :param y: dict
        shape and dtype of y, should be same shape and type as predict
    :param p: int
        decides which loss to compute, now the p only can be 1 to compute l1_loss
    :param reduction: str
        reduce mode,can be 'mean','sum' or 'none'
    :param kernel_name: kernel name, default value is "lp_loss"
    :return:
        None
    """
    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype").lower()
    label_shape = label.get("shape")
    label_dtype = label.get("dtype").lower()

    dtype_list = ["float16", "float32"]
    reduction_list = ["none", "mean", "sum"]

    op_utils.check_dtype(predict_dtype, dtype_list)
    op_utils.check_dtype(label_dtype, dtype_list)
    op_utils.check_shape(predict_shape)
    op_utils.check_shape(label_shape)

    util.compare_tensor_dict_key(predict, label, "shape")
    util.compare_tensor_dict_key(predict, label, "dtype")

    if p != 1:
        raise RuntimeError("lp_loss only supports l1_loss")

    if reduction not in reduction_list:
        raise RuntimeError("reduction should be one of ['none','mean','sum']")

    predict_data = tvm.placeholder(predict_shape, dtype=predict_dtype, name="predict_data")
    label_data = tvm.placeholder(label_shape, dtype=label_dtype, name="label_data")

    res = lp_loss_compute(predict_data, label_data, p, reduction, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [predict_data, label_data, res]}
    te.lang.cce.cce_build_code(schedule, config)
