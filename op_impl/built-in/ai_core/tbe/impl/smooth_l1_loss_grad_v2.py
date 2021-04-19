import te.lang.cce
from te import tvm
from topi import generic
from topi.cce import util
from functools import reduce
from te.platform.fusion_manager import fusion_manager


@fusion_manager.register("smooth_l1_loss_grad_v2")
def smooth_l1_loss_grad_v2_compute(input_predict, input_label, input_dout, sigma, reduction):

    ori_dtype = input_predict.dtype
    all_shape = input_predict.shape
    all_dtype = "float32"

    if ori_dtype == "float16":
        input_predict = te.lang.cce.cast_to(input_predict, all_dtype)
        input_label = te.lang.cce.cast_to(input_label, all_dtype)
        input_dout = te.lang.cce.cast_to(input_dout, all_dtype)

    # calculate input_predict-input_label
    x = te.lang.cce.vsub(input_predict, input_label)

    # calculate |input_predict-input_label|
    x_abs = te.lang.cce.vabs(x)

    # create sigma_tensor and negative_sigma_tensor
    sigma_const = tvm.const(sigma, dtype=all_dtype)
    negative_sigma_const = tvm.const(-sigma, dtype=all_dtype)
    sigma_tensor = te.lang.cce.broadcast(sigma_const, all_shape)
    negative_sigma_tensor = te.lang.cce.broadcast(negative_sigma_const, all_shape)

    # calculate smooth
    temp = te.lang.cce.vdiv(x, sigma_tensor)
    smooth1 = te.lang.cce.vcmpsel(x, negative_sigma_tensor, 'le', -1.0, 0.0)
    smooth2 = te.lang.cce.vcmpsel(x, sigma_tensor, 'ge', 1.0, 0.0)
    smooth3_temp = te.lang.cce.vcmpsel(x_abs, sigma, 'lt', 1.0, 0.0)
    smooth3 = te.lang.cce.vmul(temp, smooth3_temp)
    smooth1_2 = te.lang.cce.vadd(smooth1, smooth2)
    smooth = te.lang.cce.vadd(smooth1_2, smooth3)

    # calculate the res value and return
    res = te.lang.cce.vmul(smooth, input_dout)

    # judge reduction
    list = []
    if reduction == "sum":
        for i in range(len(all_shape)):
            list.append(i)
        res = te.lang.cce.sum(res, axis=list)
    elif reduction == "mean":
        for i in range(len(all_shape)):
            list.append(i)
        res = te.lang.cce.sum(res, axis=list)

        shape_val = reduce(lambda x, y: x * y, all_shape)
        scalar = tvm.const(int(shape_val), dtype=all_dtype)
        res = te.lang.cce.vmuls(res, 1/scalar)

    # choose dtype
    if ori_dtype == "float16":
        res = te.lang.cce.cast_to(res, "float16")

    return res


@util.check_input_type(dict, dict, dict, dict, float, str, str)
def smooth_l1_loss_grad_v2(predict, label, dout, gradient, sigma=1.0, reduction='mean',
                           kernel_name="smooth_l1_loss_grad_v2"):

    # check input: predict label dout
    check_list = ("float16", "float32")

    shape_predict = predict.get("shape")
    dtype_predict = predict.get("dtype").lower()
    util.check_dtype_rule(dtype_predict, check_list)

    shape_label = label.get("shape")
    dtype_label = label.get("dtype").lower()
    util.check_dtype_rule(dtype_label, check_list)

    shape_dout = dout.get("shape")
    dtype_dout = dout.get("dtype").lower()
    util.check_dtype_rule(dtype_dout, check_list)

    util.check_shape_rule(shape_predict)
    util.check_shape_rule(shape_label)
    util.check_shape_rule(shape_dout)

    util.compare_tensor_dict_key(predict, label, "shape")
    util.compare_tensor_dict_key(predict, dout, "shape")

    # check reduction
    check_list_reduction = ("none", "mean", "sum")
    reduction_type = reduction.lower()

    util.check_dtype_rule(reduction_type, check_list_reduction)

    input_predict = tvm.placeholder(
        shape_predict, name="predict", dtype=dtype_predict)
    input_label = tvm.placeholder(
        shape_label, name="label", dtype=dtype_label)
    input_dout = tvm.placeholder(
        shape_dout, name="dout", dtype=dtype_dout)

    res = smooth_l1_loss_grad_v2_compute(input_predict, input_label, input_dout, sigma, reduction_type)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [input_predict, input_label, input_dout, res]
    }

    te.lang.cce.cce_build_code(sch, config)
