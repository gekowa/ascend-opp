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
apply_adagradv2_d
"""
import operator
from functools import reduce as functools_reduce

import te.lang.cce
from te import platform as tbe_platform
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils
from topi import generic


# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("apply_adagradv2_d")
def apply_adagradv2_d_compute(var,
                              accum,
                              lr,
                              grad,
                              out_var,
                              out_accum,
                              epsilon,
                              update_slots,
                              kernel_name="apply_adagradv2_d"):
    """
    ApplyAdagradv2 algorithm:

    if update_slots
        accum += grad ** 2
    var -= lr * grad / (accum.sqrt() + epsilon)

    Parameters:
    ----------
    var: placeholder, input tensor with dtype float32

    accum: placeholder, has same shape and dtype as var

    lr: placeholder, has same dtype as var

    grad: placeholder, has same shape and dtype as var

    out_var: output var, has same shape and dtype as var

    out_accum: output accum, has same shape and dtype as var

    epsilon: scalar, has same dtype as var

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagradv2_d".

    Returns:
    -------
    out_var, out_accum
    """
    input_dtype = var.dtype
    vmul_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "float32")
    if input_dtype == "float32" and not vmul_support:
        raise RuntimeError(
            "Input dtype is float32, but do not support on the platform")

    if update_slots is True:
        grad_square = te.lang.cce.vmul(grad, grad)
        out_accum = te.lang.cce.vadd(accum, grad_square)
    else:
        out_accum = te.lang.cce.vadds(accum, tvm.const(0.0, input_dtype))

    lr_brc = te.lang.cce.broadcast(lr, grad.shape)
    lr_grad = te.lang.cce.vmul(grad, lr_brc)
    sqrt_accum = te.lang.cce.vsqrt(out_accum)
    sqrt_accum_epsilon = te.lang.cce.vadds(sqrt_accum,
                                           tvm.const(epsilon, input_dtype))
    update = te.lang.cce.vdiv(lr_grad, sqrt_accum_epsilon)
    out_var = te.lang.cce.vsub(var, update)

    return out_var, out_accum


def _get_placeholder(dict_list, name_list):
    list_placeholder = []
    var_shape = []
    for var, name in zip(dict_list, name_list):
        shape = var.get('shape')
        dtype = var.get('dtype').lower()
        if name == 'var':
            var_shape = list(shape)
        if name != 'lr' and var_shape != list(shape):
            raise RuntimeError("The shape of var, accum, grad must be equal.")
        if name == 'lr' and shape[0] != 1:
            raise RuntimeError("The shape of lr must be scalar.")

        op_utils.check_dtype(dtype, ('float32', ), param_name="var")
        op_utils.check_shape(shape)
        shape_refine = (functools_reduce(operator.mul, shape), )
        list_placeholder.append(
            tvm.placeholder(shape=shape_refine, name=name, dtype=dtype))
    return list_placeholder


# pylint: disable=unbalanced-tuple-unpacking,invalid-name
@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_OUTPUT, op_utils.REQUIRED_OUTPUT,
                          op_utils.REQUIRED_ATTR_FLOAT,
                          op_utils.OPTION_ATTR_BOOL, op_utils.KERNEL_NAME)
def apply_adagradv2_d(var,
                      accum,
                      lr,
                      grad,
                      out_var,
                      out_accum,
                      epsilon,
                      update_slots=True,
                      kernel_name="apply_adagradv2_d"):
    """
    Update '*var' according to the Adagrad algorithm.

    if update_slots
        accum += grad ** 2
    var -= lr * grad / (accum.sqrt() + epsilon)

    Parameters:
    ----------
    var: the dict of var, only support float32

    accum: the dict of accum, only support float32

    lr: the dict of lr, only support float32

    grad: the dict of grad, only support float32

    out_var: the dict of output, only support float32

    out_accum: the dict of output, only support float32

    epsilon: scalar, only support float32

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagradv2_d".

    Returns
    -------
    None
    """
    input_name_list = ['var', 'accum', 'lr', 'grad']
    var, accum, lr, grad = _get_placeholder([var, accum, lr, grad],
                                            input_name_list)

    out_var, out_accum = apply_adagradv2_d_compute(var, accum, lr, grad,
                                                   out_var, out_accum, epsilon,
                                                   update_slots)
    outs = [out_var, out_accum]
    build_list = [var, accum, lr, grad, out_var, out_accum]
    with tvm.target.cce():
        sch = generic.auto_schedule(outs)
    config = {"name": kernel_name, "tensor_list": build_list}

    te.lang.cce.cce_build_code(sch, config)
