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
lars_v2_update
"""
import operator
from functools import reduce as functools_reduce

import te.lang.cce
from te import platform as tbe_platform
from te import tvm
from te.platform.cce_build import build_config
from te.platform.cce_build import build_config_update
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils
from topi import generic


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@fusion_manager.register("lars_v2_update")
def lars_v2_update_compute(inputs_data,
                           hyperparam,
                           epsilon,
                           use_clip,
                           out,
                           kernel_name="lars"):
    """
    lars_update compute

    Parameters:
    ----------
    inputs_data: list
        the placeholders of input data
    hyperparam: float
        default value is 0.001
    epsilon: float
        default value is 1e-5
    use_clip: bool
        default value is "False".
    out: dict
        output contains shape and dtype attributes.
    kernel_name : str
        kernel name, default value is "lars_update"

    Returns:
    None
    """
    weight, grad, weight_s, grad_s, weight_decay, learning_rate = inputs_data

    weight_norm = te.lang.cce.vsqrt(weight_s)
    grad_norm = te.lang.cce.vsqrt(grad_s)

    coeff_weight_norm = te.lang.cce.vmuls(weight_norm, hyperparam)
    weight_norm_decay = te.lang.cce.vmul(weight_norm, weight_decay)
    weight_grad_norm = te.lang.cce.vadd(weight_norm_decay, grad_norm)
    norm_res = te.lang.cce.vadds(weight_grad_norm, epsilon)
    coeff = te.lang.cce.vdiv(coeff_weight_norm, norm_res)

    if use_clip:
        coeff_clip = te.lang.cce.vdiv(coeff, learning_rate)
        coff_max = te.lang.cce.vmins(coeff_clip,
                                     tvm.const(1, dtype=weight.dtype))
        clip_coff = te.lang.cce.vmaxs(coff_max, tvm.const(0,
                                                          dtype=weight.dtype))
        coeff_broadcast = te.lang.cce.broadcast(clip_coff, weight.shape)
    else:
        coeff_broadcast = te.lang.cce.broadcast(coeff, weight.shape)

    weight_decay_broadcast = te.lang.cce.broadcast(weight_decay, weight.shape)
    weight_weight_decay = te.lang.cce.vmul(weight, weight_decay_broadcast)
    weight_grad = te.lang.cce.vadd(weight_weight_decay, grad)

    out = te.lang.cce.vmul(weight_grad, coeff_broadcast)

    return out


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_OUTPUT, op_utils.OPTION_ATTR_FLOAT,
                          op_utils.OPTION_ATTR_FLOAT,
                          op_utils.OPTION_ATTR_BOOL, op_utils.KERNEL_NAME)
def lars_v2_update(weight,
                   grad,
                   weight_s,
                   grad_s,
                   weight_decay,
                   learning_rate,
                   out,
                   hyperparam=0.001,
                   epsilon=1e-5,
                   use_clip=False,
                   kernel_name="lars_update"):
    """
    the opreator's compute
    hyper_weight_norm = hyperparam * sqrt(weight_s)
    grad_weight_norm = sqrt(grad_s) + weight_decay*sqrt(weight_s) + epsilon
    grad_weight = grad + weight_decay * weight

    if use_clip == True:
        coeff = hyper_weight_norm / grad_weight_norm
        coeff = min(coeff / learning_rate, 1)
        coeff = max(coeff, 0)
    else:
        coeff = hyper_weight_norm / grad_weight_norm

    grad_new = coeff * grad_weight
    Parameters:
    ----------
    weight: dict
        input tensor contains shape and dtype attributes.
        only support float32.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype and shape as 'weight'.
    weight_s: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    grad_s: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    weight_decay: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    learning_rate: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    out: dict
        output tensor contains shape and dtype attributes.
        Must have the same dtype and shape  as 'weight'.
    hyperparam: float
        default value is 0.001
    epsilon: float
        default value is 1e-5
    use_clip: bool
        default value is "False".
    kernel_name : str
        kernel name, default value is "lars_update"

    Returns:
    None
    """

    check_list = ("float16", "float32")
    inputs = [weight, grad, weight_s, grad_s, weight_decay, learning_rate]

    weight_shape = weight.get("shape")
    grad_shape = grad.get("shape")
    weight_dtype = weight.get("dtype")
    grad_dtype = grad.get("dtype")
    if list(weight_shape) != list(grad_shape):
        raise RuntimeError("weight and grad must be the same shape")

    if grad_dtype != weight_dtype:
        raise RuntimeError("wight and grad must be the same dtype")

    vdiv_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vdiv", "float32")
    if weight_dtype == "float32" and not vdiv_support:
        raise RuntimeError(
            "Input dtype is float32, but do not support on the platform")

    input_place_holders = []
    for i, input_val in enumerate(inputs):
        input_dtype = input_val.get("dtype").lower()
        input_shape = input_val.get("shape")
        op_utils.check_shape(input_shape)
        op_utils.check_dtype(input_dtype, check_list)
        shape_one_dim = (functools_reduce(operator.mul, input_shape), )
        input_place_holders.append(
            tvm.placeholder(shape_one_dim,
                            name="input_data_%d" % i,
                            dtype=input_dtype))

    res = lars_v2_update_compute(input_place_holders, hyperparam, epsilon,
                                 use_clip, out, kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    data = input_place_holders
    data.append(res)

    new_config = build_config_update(build_config, "dummy_placeholder", True)
    with new_config:
        tvm.build(schedule, data, "cce", name=kernel_name)
