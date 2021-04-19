import te.lang.cce
from te import tvm
from topi.cce import util
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import broadcast_shapes, refine_shapes_for_broadcast, check_dtype, check_shape
from topi import generic


@fusion_manager.register("softplus_v2_grad")
def softplus_v2_grad_compute(input_gradients, input_features, beta, threshold):
    # TODO:Please refer to the TE DSL Manual, And code here with TE DSL.
    input_dtype = input_gradients.dtype
    shape_grad = te.lang.cce.util.shape_to_list(input_gradients.shape)
    shape_feature = te.lang.cce.util.shape_to_list(input_features.shape)
    if list(shape_grad) != list(shape_feature):
        shape_grad, shape_feature, shape_max = broadcast_shapes(shape_grad, shape_feature,
                                                                param_name_input1="input_gradients",
                                                                param_name_input2="input_features")
        input_gradients = te.lang.cce.broadcast(input_gradients, shape_max, input_dtype)
        input_features = te.lang.cce.broadcast(input_features, shape_max, input_dtype)
    if input_dtype != "float32":
        input_gradients = te.lang.cce.cast_to(input_gradients, "float32")
        input_features = te.lang.cce.cast_to(input_features, "float32")

    one_const_tensor = te.lang.cce.broadcast(tvm.const(1.0, dtype="float32"), input_features.shape)
    beta_const_tensor = te.lang.cce.broadcast(tvm.const(beta, dtype="float32"), input_features.shape)
    # calculate exp(beta*x)/ (1 + exp(beta*x))
    beta_mul_x = te.lang.cce.vmul(input_features, beta_const_tensor)
    exp_res = te.lang.cce.vexp(beta_mul_x)
    exp_add_one_res = te.lang.cce.vadd(exp_res, one_const_tensor)
    method_one_res = te.lang.cce.vdiv(exp_res, exp_add_one_res)

    # combine method one and two's result
    cmp = threshold / beta
    is_method_one = te.lang.cce.vcmpsel(input_features, tvm.const(cmp, dtype="float32"), 'le', 1, 0)
    is_method_two = te.lang.cce.vsub(one_const_tensor, is_method_one)

    method_one_get_res = te.lang.cce.vmul(method_one_res, is_method_one)
    method_two_get_res = is_method_two
    grad_out = te.lang.cce.vadd(method_one_get_res, method_two_get_res)

    res_tmp = te.lang.cce.vmul(grad_out, input_gradients)
    if input_dtype == "float16":
        res = te.lang.cce.cast_to(res_tmp, "float16")
    else:
        res = res_tmp

    return res


@util.check_input_type(dict, dict, dict, float, float, str)
def softplus_v2_grad(input_gradients, input_features, output_backprops,
                     beta=1.0, threshold=20.0, kernel_name="softplus_v2_grad"):
    """
    Computes softplus gradients for a softplus operation.
    The gradients: "dy * exp(x)/ (1 + exp(x))" if x/beta <= threshold else dy.

    Parameters
    ----------
    input_gradients: dict
        The backpropagated gradients to the corresponding softplus operation.
    input_features: dict
        The input_features passed as input to the corresponding softplus operation.
        source data type support "float16", "float32".
    output_backprops: dict
        data of output.
    beta: float16/float32, option, default:1.0
    threshold: float16/float32, option, default:20.0

    kernel_name: str
        kernel name, default value is "softplus_grad_v2".
    Returns
    -------
    None
    """
    # TODO:Please refer to the TE DSL Manual, And code here with TE DSL.
    shape_grad = input_gradients.get("shape")
    shape_feature = input_features.get("shape")
    dtype_grad = input_gradients.get("dtype")
    dtype_feature = input_features.get("dtype")
    # check dtype and shape
    if dtype_grad.lower() != dtype_feature.lower():
        raise RuntimeError(
            "type of grads and type of feature must be same, \
             while the types are different")
    input_dtype = dtype_grad.lower()

    check_list = ("float16", "float32")
    check_dtype(input_dtype, check_list, param_name="input_gradients")

    check_shape(shape_grad, param_name="input_gradients")
    check_shape(shape_feature, param_name="input_features")
    if beta == 0.0:
        raise ZeroDivisionError("the value of beta must be non-zero")
    # broadcast grad and feature
    if len(list(shape_grad)) != len(list(shape_feature)):
        raise RuntimeError(
            "shape of grads and shape of feature \
             must have the same length")
    shape_grad, shape_feature, shape_max = broadcast_shapes(shape_grad, shape_feature,
                                                            param_name_input1="input_gradients",
                                                            param_name_input2="input_features")
    reshape_grad, reshape_feature = refine_shapes_for_broadcast(shape_grad, shape_feature)

    data_grads = tvm.placeholder(reshape_grad, dtype=input_dtype, name="data_grads")
    data_features = tvm.placeholder(reshape_feature, dtype=input_dtype, name="data_features")

    res = softplus_v2_grad_compute(data_grads, data_features, beta, threshold)

    # TODO:auto schedule
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    # TODO:operator build
    config = {"name": kernel_name,
              "tensor_list": [data_grads, data_features, res]}

    te.lang.cce.cce_build_code(schedule, config)
