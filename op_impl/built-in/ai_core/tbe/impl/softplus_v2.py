import te.lang.cce
from te import tvm
from topi.cce import util
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_dtype, check_shape
from topi import generic

NEG_LN_2 = - 0.69314718055994530941723212145818


@fusion_manager.register("softplus_v2")
def softplus_v2_compute(input_features, beta, threshold, kernel_name="softplus_v2"):

    input_dtype = input_features.dtype
    input_shape = input_features.shape

    if input_dtype != "float32":
        input_features = te.lang.cce.cast_to(input_features, "float32")

    beta_const_tensor = te.lang.cce.broadcast(tvm.const(beta, dtype="float32"), input_shape)
    one_const_tensor = te.lang.cce.broadcast(tvm.const(1.0, dtype="float32"), input_shape)
    # calculate log(1+exp(beta*x))/beta
    beta_mul_x = te.lang.cce.vmul(input_features, beta_const_tensor)
    positive_part = te.lang.cce.vmaxs(beta_mul_x, tvm.const(0, dtype="float32"))
    negative_part = te.lang.cce.vmins(beta_mul_x, tvm.const(0, dtype="float32"))
    # calculate negative part softplus
    neg_to_pos = te.lang.cce.vmuls(negative_part, tvm.const(-1, dtype="float32"))
    exp_neg = te.lang.cce.vexp(neg_to_pos)
    exp_add_one = te.lang.cce.vadd(exp_neg, one_const_tensor)
    log_neg = te.lang.cce.vlog(exp_add_one)
    res_neg = te.lang.cce.vadd(log_neg, negative_part)
    # calculate positive part softplus
    exp_pos = te.lang.cce.vexp(positive_part)
    add_one = te.lang.cce.vadd(exp_pos, one_const_tensor)
    res_pos = te.lang.cce.vlog(add_one)

    log_res = te.lang.cce.vadd(res_neg, res_pos)
    res_tmp = te.lang.cce.vadds(log_res, NEG_LN_2)
    method_one_res = te.lang.cce.vdiv(res_tmp, beta_const_tensor)

    # combine two results
    cmp = threshold / beta
    is_method_one = te.lang.cce.vcmpsel(input_features, tvm.const(cmp, dtype="float32"), 'le', 1.0, 0.0)
    is_method_two = te.lang.cce.vsub(one_const_tensor, is_method_one)

    method_one_get_res = te.lang.cce.vmul(method_one_res, is_method_one)
    method_two_get_res = te.lang.cce.vmul(input_features, is_method_two)

    res_tmp = te.lang.cce.vadd(method_one_get_res, method_two_get_res)
    if input_dtype == "float16":
        res = te.lang.cce.cast_to(res_tmp, "float16")
    else:
        res = res_tmp

    return res


@util.check_input_type(dict, dict, float, float, str)
def softplus_v2(x, y, beta=1.0, threshold=20.0, kernel_name="softplus_v2"):
    """
    Computes softplus operation with attribute beta and threshold.
    The output: log(1+exp(beta*x))/beta if x/beta <= threshold else x.

    Parameters
    ----------
    x: dict
        The input_features passed as input to the corresponding softplus operation.
        source data type support "float16", "float32".
    y: dict
        data of output.
    beta: float16/float32, option, default:1.0
    threshold: float16/float32, option, default:20.0

    kernel_name: str
        kernel name, default value is "softplus_v2".
    Returns
    -------
    None
    """
    shape_feature = x.get("shape")
    dtype_feature = x.get("dtype")
    dtype_output = y.get("dtype")
    # check dtype and shape
    check_list = ("float16", "float32")
    check_dtype(dtype_feature, check_list, param_name="x")
    check_dtype(dtype_output, check_list, param_name="y")
    check_shape(shape_feature, param_name="x")

    if beta == 0.0:
        raise ZeroDivisionError("the value of beta must be non-zero")

    data_features = tvm.placeholder(shape_feature, dtype=dtype_feature, name="data_features")

    res = softplus_v2_compute(data_features, beta, threshold, kernel_name)

    # TODO:auto schedule
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    # TODO:operator build
    config = {"name": kernel_name,
              "tensor_list": [data_features, res]}
    te.lang.cce.cce_build_code(schedule, config)
