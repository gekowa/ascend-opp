# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
in_training_update_v2
"""
from __future__ import division

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils.op_utils import *
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

NONETYPE = type(None)


# pylint: disable=locally-disabled,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin
def op_select_format(x, sum, square_sum, gamma, beta, mean, variance,
                     y, batch_mean, batch_variance,
                     momentum, epsilon,
                     kernel_name="in_training_update_v2"):
    """
    select format dynamically
    """
    input0 = gen_param(classify="input0", name="x",
                       datatype="float16,float",
                       format="NC1HWC0,NC1HWC0")
    input1 = gen_param(classify="input1", name="sum",
                       datatype="float,float",
                       format="NC1HWC0,NC1HWC0")
    input2 = gen_param(classify="input2", name="square_sum",
                       datatype="float,float",
                       format="NC1HWC0,NC1HWC0")
    input3 = gen_param(classify="input3", name="gamma",
                       datatype="float,float",
                       format="NC1HWC0,NC1HWC0")
    input4 = gen_param(classify="input4", name="beta",
                       datatype="float,float",
                       format="NC1HWC0,NC1HWC0")
    input5 = gen_param(classify="input5", name="mean",
                       datatype="float,float",
                       format="NC1HWC0,NC1HWC0")
    input6 = gen_param(classify="input6", name="variance",
                       datatype="float,float",
                       format="NC1HWC0,NC1HWC0")
    output0 = gen_param(classify="output0", name="y",
                        datatype="float16,float",
                        format="NC1HWC0,NC1HWC0")
    output1 = gen_param(classify="output1", name="batch_mean",
                        datatype="float,float",
                        format="NC1HWC0,NC1HWC0")
    output2 = gen_param(classify="output2", name="batch_variance",
                        datatype="float,float",
                        format="NC1HWC0,NC1HWC0")

    param_list = [input0, input1, input2, input3, input4, input5, input6,
                  output0, output1, output2]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_shape(shape, param_name=""):
    """
     Function to check input tensors dims.

     Parameters
     ----------
     shape: list or tuple
         data shape of test input
     Returns
     -------
     None
     """
    check_shape(shape, min_rank=5, max_rank=5, param_name=param_name)


def check_rule(data, rule_desc, param_name=PARAM_NAME):
    """
    The special check rule for tensor
    """
    if data is None or rule_desc is None:
        return
    error_info = {}
    error_info['errCode'] = OP_ERROR_CODE_009
    error_info['op_name'] = OP_NAME
    error_info['param_name'] = param_name
    error_info['rule_desc'] = rule_desc
    error_info['param_value'] = data
    raise RuntimeError(error_info,
                       "Op[%s] has rule: %s, but [%s] is [%s]."\
                        % (error_info['op_name'],
                        error_info['rule_desc'],
                        error_info['param_name'],
                        error_info['param_value']))


def _check_dims_equal(shape_x, shape):
    """
    Function to check the dimension C to be equal.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    shape: list or tuple
        data shape of test input

    Returns
    -------
    None
    """
    if shape_x[0] != shape[0] or shape_x[1] != shape[1] or shape_x[4] != \
            shape[4]:
        check_rule("{} and {}".format(shape_x, shape),
                   "The dimensions N, C1, C0 of shape_x"\
                   "and shape must be equal",
                   "shape_x and shape")
    if shape[2] != 1 or shape[3] != 1:
        check_rule("{} and {}".format(shape[2], shape[3]),
                   "Dimension H,W must be 1",
                   "H,W")


@fusion_manager.register("in_training_update_v2")
def in_training_update_compute(x, sum, square_sum,
                               gamma, beta, mean, variance,
                               data_format,
                               y, mean_out, variance_out,
                               momentum, epsilon,
                               kernel_name="in_training_update_v2"):
    """
    algorithm: instance_norm_v2
    instance normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    gamma: TVM tensor
        contains scale data
    beta: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    data_format: str
        data format
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    mean_out: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    momentum: float
        A ratio to calculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "in_training_update_v2"

    Returns
    -------
    res: TVM tensor list
        the result of in_training_update_v2 compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)

    # compute the instance normalization of x
    is_cast = False
    if x.dtype == "float16":
        is_cast = True
        x = te.lang.cce.cast_to(x, "float32")

    num = shape_x[2] * shape_x[3]
    num_rec = 1.0 / num
    # compute the saved mean of x
    compute_mean = te.lang.cce.vmuls(sum, num_rec)
    mean_boardcast = te.lang.cce.broadcast(compute_mean, shape_x)

    # compute the saved variance of x
    variance_div = te.lang.cce.vmuls(square_sum, num_rec)
    variance_square = te.lang.cce.vmul(compute_mean, compute_mean)
    compute_var = te.lang.cce.vsub(variance_div, variance_square)

    # (x - mean) / sqrt(var + eps)
    # x_mean = x - mean
    # multiplier_add = var + eps
    # multiplier_sqrt = sqrt(var + eps)

    x_mean = te.lang.cce.vsub(x, mean_boardcast)
    multiplier_add = te.lang.cce.vadds(compute_var, epsilon)
    multiplier_sqrt = te.lang.cce.vsqrt(multiplier_add)
    sqrt_boardcast = te.lang.cce.broadcast(multiplier_sqrt, shape_x)
    mean_wo_scale = te.lang.cce.vdiv(x_mean, sqrt_boardcast)
    result = mean_wo_scale
    if gamma is not None and beta is not None:
        gamma = te.lang.cce.broadcast(gamma, shape_x)
        beta = te.lang.cce.broadcast(beta, shape_x)
        gamma_scale = te.lang.cce.vmul(result, gamma)
        result = te.lang.cce.vadd(gamma_scale, beta)

    if is_cast:
        result = te.lang.cce.cast_to(result, "float16")

    if num == 1:
        batch_var_scalar = 0.0
    else:
        batch_var_scalar = float(num) / (num - 1)

    result_mean = compute_mean
    result_variance = te.lang.cce.vmuls(compute_var, batch_var_scalar)

    # if input mean and var, use input values and momentum to update
    # else, output compute values
    if mean is not None and variance is not None:
        factor_reverse = 1.0 - momentum
        mean_mul = te.lang.cce.vmuls(compute_mean, momentum)
        mean_mul_rev = te.lang.cce.vmuls(mean, factor_reverse)
        result_mean = te.lang.cce.vadd(mean_mul, mean_mul_rev)

        var_mul = te.lang.cce.vmuls(result_variance, momentum)
        var_mul_rev = te.lang.cce.vmuls(variance, factor_reverse)
        result_variance = te.lang.cce.vadd(var_mul, var_mul_rev)
    res = [result, result_mean, result_variance]
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 OPTION_INPUT, OPTION_INPUT,
                 OPTION_INPUT, OPTION_INPUT,
                 REQUIRED_OUTPUT, OPTION_OUTPUT, OPTION_OUTPUT,
                 OPTION_ATTR_FLOAT, OPTION_ATTR_FLOAT, KERNEL_NAME)
def in_training_update_v2(x, sum, square_sum,
                          gamma, beta, mean, variance,
                          y, batch_mean, batch_variance,
                          momentum=0.1, epsilon=0.00001,
                          kernel_name="in_training_update_v2"):
    """
    algorithm: instance_norm_v2
    instance normalization.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    sum: dict
        dict of sum, A Tensor for sum.
        The output of instance_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A Tensor for square_sum.
        The output of instance_normalization_forward_training_reduce.
    gamma: dict
        dict of scale, A Tensor for mean.
    beta: dict
        dict of offset, A Tensor for variance.
    mean: dict
        dict of mean, A Tensor for mean.
    variance: dict
        dict of variance, A Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    batch_variance: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    momentum: float
        A ratio to calculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "in_training_update_v2"

    Returns
    -------
    None
    """
    data_format = x.get("format")
    # Process x, sum, square_sum
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    check_format(data_format, ("NC1HWC0", ), param_name="x")

    _check_shape(shape_x, param_name="x")
    check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    shape_sum = sum.get("shape")
    shape_square_sum = square_sum.get("shape")
    _check_shape(shape_sum, param_name="sum")
    _check_dims_equal(shape_x, shape_sum)
    _check_shape(shape_square_sum, param_name="square_sum")
    _check_dims_equal(shape_x, shape_square_sum)

    dtype_sum = sum.get("dtype")
    dtype_square_sum = square_sum.get("dtype")
    check_dtype(dtype_sum.lower(), ("float32",), param_name="sum")
    check_dtype(dtype_square_sum.lower(), ("float32",),
                param_name="square_sum")

    x_input = tvm.placeholder(shape_x, name="x_input",
                              dtype=dtype_x.lower())
    sum_input = tvm.placeholder(shape_sum, name="sum_input",
                                dtype=dtype_sum.lower())
    square_sum_input = tvm.placeholder(shape_square_sum,
                                       name="square_sum_input",
                                       dtype=dtype_square_sum.lower())
    gamma_input, beta_input, mean_input, var_input = None, None, None, None

    # Process gamma and beta

    scale = False
    if gamma is not None and beta is not None:
        scale = True
        shape_gamma = gamma.get("shape")
        dtype_gamma = gamma.get("dtype")
        _check_shape(shape_gamma, param_name="gamma")
        _check_dims_equal(shape_x, shape_gamma)

        shape_beta = beta.get("shape")
        dtype_beta = beta.get("dtype")
        _check_shape(shape_beta, param_name="beta")
        _check_dims_equal(shape_x, shape_beta)

        check_dtype(dtype_gamma.lower(), ("float32",), param_name="gamma")
        check_dtype(dtype_beta.lower(), ("float32",), param_name="beta")

        gamma_input = tvm.placeholder(shape_gamma, name="gamma_input",
                                      dtype=dtype_gamma.lower())
        beta_input = tvm.placeholder(shape_beta, name="beta_input",
                                     dtype=dtype_beta.lower())

    # Process mean and var
    use_mean = False
    if mean is not None and variance is not None:
        use_mean = True
        shape_mean = mean.get("shape")
        dtype_mean = mean.get("dtype")
        _check_shape(shape_mean, param_name="mean")
        _check_dims_equal(shape_x, shape_mean)

        shape_var = variance.get("shape")
        dtype_var = variance.get("dtype")
        _check_shape(shape_var, param_name="variance")
        _check_dims_equal(shape_x, shape_var)

        check_dtype(dtype_mean.lower(), ("float32",), param_name="mean")
        check_dtype(dtype_var.lower(), ("float32",), param_name="variance")

        mean_input = tvm.placeholder(shape_mean, name="mean_input",
                                     dtype=dtype_mean.lower())
        var_input = tvm.placeholder(shape_var, name="variance_input",
                                    dtype=dtype_var.lower())

    res = in_training_update_compute(x_input, sum_input, square_sum_input,
                                     gamma_input, beta_input, mean_input,
                                     var_input,
                                     data_format,
                                     y, batch_mean, batch_variance,
                                     momentum, epsilon,
                                     kernel_name=kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    if use_mean:
        if scale:
            tensor_list = [x_input, sum_input, square_sum_input,
                           gamma_input, beta_input, mean_input,
                           var_input] + list(res)
        else:
            tensor_list = [x_input, sum_input, square_sum_input,
                           mean_input, var_input] + list(res)
    else:
        if scale:
            tensor_list = [x_input, sum_input, square_sum_input,
                           gamma_input, beta_input] + list(res)
        else:
            tensor_list = [x_input, sum_input, square_sum_input] + list(
                res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
