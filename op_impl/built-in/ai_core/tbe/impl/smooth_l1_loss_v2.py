import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util
from functools import reduce
from te.platform.fusion_manager import fusion_manager


@fusion_manager.register("smooth_l1_loss_v2")
def smooth_l1_loss_v2_compute(input_predict,
                              input_label,
                              sigma,
                              reduction):
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
    reduction: str
       type of result, default value is "mean"
    kernel_name : str
       kernel name, default value is "smooth_l1_loss_v2"

    Returns
    -------
    output tensor
    """
    ori_dtype = input_predict.dtype
    shape = input_predict.shape
    input_dtype = "float32"

    half_const_tensor = te.lang.cce.broadcast(tvm.const(0.5, dtype=input_dtype), input_predict.shape)
    one_const_tensor = te.lang.cce.broadcast(tvm.const(1.0, dtype=input_dtype), input_predict.shape)

    if ori_dtype == "float16":
        input_predict = te.lang.cce.cast_to(input_predict, input_dtype)
        input_label = te.lang.cce.cast_to(input_label, input_dtype)

    sigma_scalar = tvm.const(sigma, dtype=input_dtype)
    input_sub_res = te.lang.cce.vsub(input_predict, input_label)
    method_one_res = te.lang.cce.vmul(te.lang.cce.vmul(input_sub_res, half_const_tensor), input_sub_res)
    predict_label_sub_abs = te.lang.cce.vabs(input_sub_res)
    method_two_res = te.lang.cce.vsub(te.lang.cce.vmuls(predict_label_sub_abs, sigma_scalar),
                                      te.lang.cce.vmuls(half_const_tensor, sigma_scalar * sigma_scalar))

    is_method_one_res = te.lang.cce.vcmpsel(predict_label_sub_abs, sigma_scalar, 'lt', 1.0, 0.0)
    is_method_two_res = te.lang.cce.vsub(one_const_tensor, is_method_one_res)
    method_one_get_res = te.lang.cce.vmul(method_one_res, is_method_one_res)
    method_two_get_res = te.lang.cce.vmul(method_two_res, is_method_two_res)
    res = te.lang.cce.vadd(method_one_get_res, method_two_get_res)

    list = []
    if reduction == "sum":
        for i in range(len(shape)):
            list.append(i)
        res = te.lang.cce.sum(res, axis=list)
    elif reduction == "mean":
        for i in range(len(shape)):
            list.append(i)
        res = te.lang.cce.sum(res, axis=list)

        shape_val = reduce(lambda x, y: x * y, shape)
        scalar = tvm.const(int(shape_val), dtype=input_dtype)
        res = te.lang.cce.vmuls(res, 1 / scalar)

    if ori_dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, dict, float, str, str)
def smooth_l1_loss_v2(predict,
                      label,
                      loss,
                      sigma=1.0,
                      reduction="mean",
                      kernel_name="smooth_l1_loss_v2"):
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
        sigma, default value is 1
    reduction: str
        type of result, default value is "mean"
    kernel_name : str
        kernel name, default value is "smooth_l1_lossV2"

    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)
    check_list = ("float16", "float32")

    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype").lower()
    util.check_dtype_rule(dtype_predict, check_list)

    shape_label = label.get("shape")
    dtype_label = label.get("dtype").lower()
    util.check_dtype_rule(dtype_label, check_list)

    shape_loss = label.get("shape")
    dtype_loss = loss.get("dtype").lower()
    util.check_dtype_rule(dtype_loss, check_list)

    util.check_shape_rule(shape_predict)
    util.check_shape_rule(shape_label)
    util.check_shape_rule(shape_loss)

    util.compare_tensor_dict_key(predict, label, "shape")

    check_list_reduction = ("none", "mean", "sum")
    reduction_type = reduction.lower()

    util.check_dtype_rule(reduction_type, check_list_reduction)

    input_predict = tvm.placeholder(
        shape_predict, name="predict", dtype=dtype_predict)
    input_label = tvm.placeholder(
        shape_label, name="label", dtype=dtype_label)

    res = smooth_l1_loss_v2_compute(input_predict, input_label, sigma,
                                    reduction_type)

    # TODO:auto schedule
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    # TODO:operator build
    config = {
        "name": kernel_name,
        "tensor_list": [input_predict, input_label, res]
    }

    te.lang.cce.cce_build_code(sch, config)
