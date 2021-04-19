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
apply_keras_momentum_d
"""
import te.lang.cce
from impl.util.util_apply_op_schedule import ApplyOpConfig
from impl.util.util_apply_op_schedule import common_apply_op_process
from te import platform as tbe_platform
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils


# pylint: disable=too-many-arguments,invalid-name,too-many-locals
# pylint: disable=unused-argument
@fusion_manager.register("apply_keras_momentum_d")
def apply_keras_momentum_d_compute(var,
                                   accum,
                                   lr,
                                   grad,
                                   momentum,
                                   out_var,
                                   out_accum,
                                   use_nesterov,
                                   kernel_name="apply_keras_momentum_d"):
    """
    the operator's compute
    :param var: weight, placeholder
    :param accum: accum, placeholder
    :param lr: learning rate, placeholder
    :param grad: gradient, placeholder
    :param momentum: nesterov momentum, placeholder
    :param out_var: updated of var
    :param out_accum: updated of accum
    :param use_nesterov: bool
    :return: out_var, out_accum
    """
    inp_dtype = var.dtype
    # check the instruction supports or not
    vmul_support = tbe_platform.cce_conf.api_check_support(
        "te.lang.cce.vmul", "float32")
    if inp_dtype == "float32" and not vmul_support:
        raise RuntimeError(
            "Input dtype is float32, but do not support on the platform")

    # update var and accum according to the momentum scheme
    # accum = accum * momentum - grad * lr
    accum_momen = tvm.compute(accum.shape,
                              lambda *indices: accum(*indices) * momentum[0],
                              tag='elewise_single_VS_mul')
    grad_lr = tvm.compute(grad.shape,
                          lambda *indices: grad(*indices) * lr[0],
                          tag='elewise_single_VS_mul')
    out_accum = te.lang.cce.vsub(accum_momen, grad_lr)

    # var = var + accum * momentum - grad * lr
    if use_nesterov is True:
        accum_momen2 = tvm.compute(
            accum.shape,
            lambda *indices: out_accum(*indices) * momentum[0],
            tag='elewise_single_VS_mul')
        add_var_am = te.lang.cce.vadd(var, accum_momen2)
        out_var = te.lang.cce.vsub(add_var_am, grad_lr)
    # var = var + accum
    else:
        out_var = te.lang.cce.vadd(var, out_accum)

    def _compute(*index):
        return out_var(*index), out_accum(*index)

    return tvm.compute(var.shape, _compute, name='outputs')


# pylint: disable=too-many-arguments,unused-argument,unbalanced-tuple-unpacking
@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_OUTPUT,
                          op_utils.REQUIRED_OUTPUT, op_utils.OPTION_ATTR_BOOL,
                          op_utils.OPTION_ATTR_BOOL, op_utils.KERNEL_NAME)
def apply_keras_momentum_d(var,
                           accum,
                           lr,
                           grad,
                           momentum,
                           out_var,
                           out_accum,
                           use_locking=False,
                           use_nesterov=False,
                           kernel_name="apply_keras_momentum_d"):
    """
    Update '*var' according to the momentum scheme.

    accum = accum * momentum - grad * lr
    if use_nesterov is True:
        var = var + accum * momentum - grad * lr
    else:
        var = var + accum

    Parameters
    ----------
    var : dict of tensor var, include shape and dtype.

    accum : dict of tensor accum, include shape and dtype.

    lr: dict of scalar lr(learning rate), include shape and dtype.

    grad: dict of tensor grad, include shape and dtype.

    momentum: dict of scala, include shape and dtype.

    out_var: dict of updated var.

    out_accum: dict of updated accum.

    use_locking: bool, default value is "False",
                 if "True", var will be updated by using Nesterov momentum.

    use_nesterov: bool, default value is "False".

    kernel_name :  kernel name, default value is "apply_keras_momentum_d"

    Returns
    -------
    None
    """

    input_dict = (var, accum, lr, grad, momentum)

    args = ApplyOpConfig.TensorArgs(
        input_dict,
        apply_keras_momentum_d_compute,
        [out_var, out_accum],
        6 if use_nesterov else 5,
    )
    name = ApplyOpConfig.TensorName(all=('var', 'accum', 'lr', 'grad',
                                         'momentum'),
                                    scalar=('lr', 'momentum'),
                                    reuse=())
    options = ApplyOpConfig.TensorOptions(attrs=use_nesterov)

    common_apply_op_process(ApplyOpConfig(args, name, options), kernel_name)
