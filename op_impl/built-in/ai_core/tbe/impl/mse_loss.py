import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te.utils import op_utils
from te import platform as tbe_platform


@fusion_manager.register("mse_loss")
def mse_loss_compute(predict, label, reduction='mean', kernel_name="mse_loss"):
    '''
    calculating mse_loss
    :param predict: TVM tensor
                   the output of previous layer
    :param label: TVM tensor
                label
    :param reduction: str
                    reduce configuration parameter: mean/sum/none. Default: mean
    :param kernel_name: str
                    kernel name, default value is "mse_loss"
    :return:y
            when reduction=none:TVM tensor, output tensor
            when reduction=sum/mean, A Scalar
    '''
    ori_dtype = predict.dtype
    shape = te.lang.cce.util.shape_to_list(predict.shape)

    if ori_dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmul", "float32"):
        predict = te.lang.cce.cast_to(predict, "float32")
        label = te.lang.cce.cast_to(label, "float32")

    # get total number of tensor
    reduce_elts = 1.0
    for i in shape:
        reduce_elts *= i
    cof = reduce_elts**(-1)

    # get total axis for reduce
    axis_d = []
    for i, _ in enumerate(shape):
        axis_d.append(i)
    axis_d = util.axis_check(len(shape), axis_d)

    # calcu value:(predict_n - label_n)^2
    res = te.lang.cce.vsub(predict, label)
    res_sqr = te.lang.cce.vmul(res, res)

    y = 0.0

    if reduction == 'mean':
        # calcu mean
        y = te.lang.cce.sum(res_sqr, axis=axis_d, keepdims=False)
        y = te.lang.cce.vmuls(y, cof)
    elif reduction == 'sum':
        # calcu sum
        y = te.lang.cce.sum(res_sqr, axis=axis_d, keepdims=False)
    elif reduction == 'none':
        y = res_sqr

    if ori_dtype == "float16":
        y = te.lang.cce.cast_to(y, "float16")

    return y


@op_utils.check_op_params(dict, dict, dict, str, str)
def mse_loss(predict, label, y, reduction='mean', kernel_name="mse_loss"):
    '''
    calculating data
    sum = (predict_n - label_n)^2
    if  reduction == sum: res = sum output a scalal
    if reduction == mean: res == sum/total_number_of_tensor output a scalar
    if reduction == none: res == (predict_n - label_n)^2  output a tensor

    :param predict: dict
                    shape and dtype of tensor predict
    :param label: dict
                    shape and dtype of tensor real label,
                    should be same shape and dtype as predict
    :param y: dict
              shape and dtype of output, loss result after compute
    :param reduction: str
                      Specifies the reduction to apply to the output:'none' | 'mean' | 'sum'
                      Default: 'mean'
                      'none': no reduction will be applied,
                      'mean': the sum of the output will be divided by the number
                            of elements in the output
                      'sum': the output will be summed. Note: size_average and reduce
                           are in the process of being deprecated
                           and in the meantime, specifying either of those
                           two args will override reduction.
    :param kernel_name: str
                      kernel name, default value is "mse_loss"
    :return: none
    '''

    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype")
    predict_dtype_lower = predict_dtype.lower()

    label_shape = label.get("shape")
    label_dtype = label.get("dtype")
    label_dtype_lower = label_dtype.lower()

    # check dtype
    dtype_list = ("float16", "float32")
    op_utils.check_dtype(predict_dtype, dtype_list)
    op_utils.check_dtype(predict_dtype, dtype_list)

    # check shape
    op_utils.check_shape(predict_shape)
    op_utils.check_shape(label_shape)

    # check kernel_name
    util.check_kernel_name(kernel_name)

    predict_size, _ = op_utils.refine_shape_axes(predict_shape, [])
    data_predict = tvm.placeholder(predict_size, dtype=predict_dtype_lower, name="data_predict")

    label_size, _ = op_utils.refine_shape_axes(label_shape, [])
    data_label = tvm.placeholder(label_size, dtype=label_dtype_lower, name="data_label")

    if predict_size != label_size:
        raise RuntimeError("predict tensor size don't match label tensor")
    if reduction not in ("mean", "sum", "none"):
        raise RuntimeError("reduction type should in mean/sum/none")

    res = mse_loss_compute(data_predict, data_label, reduction, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_predict, data_label, res]}
    te.lang.cce.cce_build_code(schedule, config)
