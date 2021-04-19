# Copyright 2020 Huawei Technologies Co., Ltd
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
in_infer_v2d
"""
from __future__ import absolute_import
from __future__ import division

import te.lang.cce
from te import tvm
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils.op_utils import *

NONETYPE = type(None)


def _check_shape(shape, data_format="NC1HWC0", param_name="x"):
    """
     Function to check input tensors dims.

     Parameters
     ----------
     shape: list or tuple
         data shape of test input
     data_format: str
         format of input data
     Returns
     -------
     None
     """
    check_shape(shape, min_rank=5, max_rank=5,
                param_name=param_name)
    check_format(data_format.upper(), ("NC1HWC0",),
                 param_name=param_name)


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
                       "Op[%s] has rule: %s, but [%s] is [%s]." \
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
    data_format: str
        format of input data

    Returns
    -------
    None
    """
    if shape_x[0] != shape[0] or \
            shape_x[1] != shape[1] or shape_x[4] != shape[4]:
        check_rule("{} and {}".format(shape_x, shape),
                   "The dimensions N, C1, C0 of shape_x" \
                   "and shape must be equal",
                   "shape_x and shape")
    if shape[2] != 1 or shape[3] != 1:
        check_rule("{} and {}".format(shape[2], shape[3]),
                   "Dimension H,W must be 1",
                   "H,W")


# pylint: disable=locally-disabled,too-many-arguments,invalid-name
def _output_data_y_compute(x, mean, scale, offset, y_sqrt):
    """
    Function to calculate the y, which is a public function

    Parameters
    ----------
    x: TVM tensor
        contains x data
    mean: TVM tensor
        contains mean data.
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    y_sqrt: float
        sqrt A small float number added to the variance of x.

    Returns
    -------
    res: TVM tensor
        the y of instance_norm_v2 compute
    """
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    var_sub = te.lang.cce.vsub(x, mean)

    if scale is not None and offset is not None:

        scale_sqrt = te.lang.cce.vdiv(scale, y_sqrt)
        scale_broad = te.lang.cce.broadcast(scale_sqrt, shape_x)
        offset_broad = te.lang.cce.broadcast(offset, shape_x)
        res = te.lang.cce.vadd(te.lang.cce.vmul(scale_broad, var_sub),
                               offset_broad)
    else:
        sqrt_broadcast = te.lang.cce.broadcast(y_sqrt, shape_x)
        res = te.lang.cce.vdiv(var_sub, sqrt_broadcast)
    return res


# pylint: disable=locally-disabled,too-many-locals
@fusion_manager.register("in_infer_v2d")
def instance_norm_inf_compute(x, scale, offset, mean, variance,
                              y_sqrt, kernel_name="in_infer_v2"):
    """
    Function to calculate the output of instance_norm_v2 for inferance.

    Description of calculating process with TE api,
    the computational formula is as follows.
    x = (x - mean)/(var + epsilon)**0.5
    y = scale*x + offset

    Parameters
    ----------
    x: TVM tensor
        contains x data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data. Used for inference only.
    variance: TVM tensor
        contains variance data. Used for inference only.
    y_sqrt: Tensor
        sqrt A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "instance_norm_v2"

    Returns
    -------
    res: TVM tensor list
        the result of instance_norm_ext2 inference compute
    """

    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    is_cast = False

    if x.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        is_cast = True
        x = te.lang.cce.cast_to(x, "float32")

    mean_broadcast = te.lang.cce.broadcast(mean, shape_x)
    var_broadcast = te.lang.cce.broadcast(variance, shape_x)
    res_y = _output_data_y_compute(x, mean_broadcast,
                                   scale, offset, y_sqrt)
    if is_cast:
        res_y = te.lang.cce.cast_to(res_y, "float16")

    scalar_zero = 0.0
    res_batch_mean = te.lang.cce.vadds(mean, scalar_zero)
    res_batch_var = te.lang.cce.vadds(variance, scalar_zero)
    res = [res_y, res_batch_mean, res_batch_var]

    return res


@check_op_params(REQUIRED_INPUT, OPTION_INPUT, OPTION_INPUT,
                 OPTION_INPUT, OPTION_INPUT, OPTION_INPUT,
                 REQUIRED_OUTPUT, OPTION_OUTPUT, OPTION_OUTPUT,
                 KERNEL_NAME)
def in_infer_v2d(x, gamma, beta, mean, variance, variance_sqrt,
                 y, batch_mean, batch_variance, kernel_name="in_infer_v2"):
    """
    algorithm: in_infer_v2
    instance normalization for inference.
    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.

    gamma: dict
        dict of scale,
        A Tensor for scaling factor, to scale the normalized x.
    beta: dict
        dict of offset, A Tensor for offset, to shift to the normalized x.
    mean: dict
        dict of mean, A Tensor for population mean.
        if not empty in training, update the running_mean with momentum
    variance: dict
        dict of variance, A Tensor for population variance.
        if not empty in training, update the running_mean with momentum
    y: dict
        dict of output, A `Tensor`. Has the same type as `input_x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`. Has the same type as `input_mean`.
    batch_variance: dict
        dict of batch_var, A `Tensor`. Has the same type as `input_variance`.
    variance_sqrt: float
        sqrt A small float number added to the variance of x
    kernel_name: str
        kernel name, default value is "instance_norm_v2"

    Returns
    -------
    None
    """

    affine = False
    use_exist_mean = False

    shape_input = x.get("shape")
    dtype_x = x.get("dtype")
    data_format = x.get("format")
    _check_shape(shape_input, data_format, "x")
    check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    x_input = tvm.placeholder(shape_input, name="x_input",
                              dtype=dtype_x.lower())

    gamma_input, beta_input, mean_input, variance_input, \
    variance_sqrt_input = \
        None, None, None, None, None
    if gamma is not None and beta is not None:
        affine = True
        shape_gamma = gamma.get("shape")
        shape_beta = beta.get("shape")
        dtype_gamma = gamma.get("dtype")
        dtype_beta = beta.get("dtype")
        check_dtype(dtype_gamma.lower(), ("float32",), param_name="gamma")
        check_dtype(dtype_beta.lower(), ("float32",), param_name="beta")
        _check_shape(shape_gamma, data_format, "gamma")
        _check_shape(shape_beta, data_format, "beta")
        _check_dims_equal(shape_input, shape_gamma)
        _check_dims_equal(shape_input, shape_beta)
        gamma_input = tvm.placeholder(shape_gamma, name="gamma_input",
                                      dtype=dtype_gamma.lower())
        beta_input = tvm.placeholder(shape_beta, name="beta_input",
                                     dtype=dtype_beta.lower())

    if mean is not None and variance is not None:
        use_exist_mean = True

        dtype_mean = mean.get("dtype")
        dtype_variance = variance.get("dtype")
        check_dtype(dtype_mean.lower(), ("float32",), param_name="mean")
        check_dtype(dtype_variance.lower(), ("float32",), param_name="variance")

        dtype_variance_sqrt = variance_sqrt.get("dtype")
        check_dtype(dtype_variance_sqrt.lower(), ("float32",),
                    param_name="variance_sqrt")

        shape_mean = mean.get("shape")
        shape_variance = variance.get("shape")
        shape_variance_sqrt = variance_sqrt.get("shape")

        _check_shape(shape_mean, data_format, "mean")
        _check_shape(shape_variance, data_format, "variance")
        _check_shape(shape_variance_sqrt, data_format, "variance_sqrt")
        _check_dims_equal(shape_input, shape_mean)
        _check_dims_equal(shape_input, shape_variance)
        _check_dims_equal(shape_input, shape_variance_sqrt)

        mean_input = tvm.placeholder(shape_mean, name="mean_input",
                                     dtype=dtype_mean.lower())
        variance_input = tvm.placeholder(shape_variance,
                                         name="variance_input",
                                         dtype=dtype_variance.lower())

        variance_sqrt_input = tvm.placeholder(
            shape_variance_sqrt,
            name="variance_sqrt_input",
            dtype=dtype_variance_sqrt.lower())

        res = instance_norm_inf_compute(x_input, gamma_input, beta_input,
                                        mean_input, variance_input,
                                        variance_sqrt_input, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    if affine:
        if use_exist_mean:
            tensor_list = [x_input, gamma_input, beta_input,
                           mean_input, variance_input,
                           variance_sqrt_input] + list(res)
        else:
            tensor_list = [x_input, gamma_input, beta_input] + list(res)
    else:
        if use_exist_mean:
            tensor_list = [x_input, mean_input, variance_input,
                           variance_sqrt_input] + list(res)
        else:
            tensor_list = [x_input] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
