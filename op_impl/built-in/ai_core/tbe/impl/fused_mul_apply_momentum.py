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
fused_mul_apply_momentum
"""
import operator
from functools import reduce as functools_reduce

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils
from topi import generic


# pylint: disable=unused-argument,invalid-name
@fusion_manager.register("fused_mul_apply_momentum")
def _fused_mul_apply_momentum_compute(var,
                                      accum,
                                      lr,
                                      x1,
                                      momentum,
                                      x2,
                                      out_var,
                                      out_accum,
                                      use_nesterov,
                                      kernel_name='fused_mul_apply_momentum'):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + x1 * x2
    if use_nesterov is True:
        var -= x1 * x2 * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : mutable tensor var.

    accum: mutable tensor accum.

    lr : scalar lr.

    x1 : tensor x1.

    momentum : scalar momentum.

    x2 : scalar x2.

    out_var : the var output.

    out_accum : the accum output

    use_nesterov: bool. If true, use nesterov computing grad,
                  default value is False.

    kernel_name : cce kernel name, default value is
                 "cce_fused_mul_apply_momentum" (optional).

    Returns:
    -------
    out_var, out_accum
    """

    # cast to float32 for higher accuracy
    dtype = var.dtype

    if dtype == "float16":
        var = te.lang.cce.cast_to(var, "float32")
        accum = te.lang.cce.cast_to(accum, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        x1 = te.lang.cce.cast_to(x1, "float32")
        x2 = te.lang.cce.cast_to(x2, "float32")
        momentum = te.lang.cce.cast_to(momentum, "float32")

    # calc grad
    x2_brc = te.lang.cce.broadcast(x2, x1.shape)
    grad = te.lang.cce.vmul(x1, x2_brc)
    # update accum
    momentum_brc = te.lang.cce.broadcast(momentum, accum.shape)
    accum_delta = te.lang.cce.vmul(accum, momentum_brc)
    accum_t = te.lang.cce.vadd(accum_delta, grad)

    # update var
    lr_brc = te.lang.cce.broadcast(lr, accum.shape)
    if use_nesterov:
        var_delta = te.lang.cce.vmul(grad, lr_brc)
        var_delta_2 = te.lang.cce.vmul(accum_t, momentum_brc)
        var_delta_2 = te.lang.cce.vmul(var_delta_2, lr_brc)
        var_delta = te.lang.cce.vadd(var_delta, var_delta_2)
        var_t = te.lang.cce.vsub(var, var_delta)
    else:
        var_delta = te.lang.cce.vmul(accum_t, lr_brc)
        var_t = te.lang.cce.vsub(var, var_delta)

    if dtype == "float16":
        var_t = te.lang.cce.cast_to(var_t, "float16")
        accum_t = te.lang.cce.cast_to(accum_t, "float16")

    return var_t, accum_t


def _get_placeholder(dict_list, name_list):
    list_placeholder = []
    var_shape = []
    for var, name in zip(dict_list, name_list):
        shape = var.get('shape')
        dtype = var.get('dtype').lower()
        if name == 'var':
            var_shape = list(shape)
        if name != 'lr' and name != 'momentum' and name != 'x2' \
            and var_shape != list(shape):
            raise RuntimeError(
                "The shapes of var, accum and x1 must be equal.")
        if (name == 'lr' or name == 'momentum'
                or name == 'x2') and shape[0] != 1:
            raise RuntimeError(
                "The shapes of lr, momentum and x2 must be scalar.")

        op_utils.check_dtype(dtype, ('float32', 'float16'), param_name="var")
        op_utils.check_shape(shape, param_name="var")
        shape_refine = (functools_reduce(operator.mul, shape), )
        list_placeholder.append(
            tvm.placeholder(shape=shape_refine, name=name, dtype=dtype))
    return list_placeholder


# pylint: disable=unbalanced-tuple-unpacking
@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_OUTPUT, op_utils.REQUIRED_OUTPUT,
                          op_utils.OPTION_ATTR_BOOL, op_utils.KERNEL_NAME)
def fused_mul_apply_momentum(var,
                             accum,
                             lr,
                             x1,
                             momentum,
                             x2,
                             out_var,
                             out_accum,
                             use_nesterov=False,
                             kernel_name="fused_mul_apply_momentum"):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + x1 * x2
    if use_nesterov is True:
        var -= gard * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : the dict of mutable tensor var, only support float16, float32.

    accum: the dict of mutable tensor accum. Must have the same dtype as `var`.

    lr : the dict of scalar lr. Must have the same dtype as `var`.

    x1 : the dict of tensor grad. Must have the same dtype as `var`.

    momentum : the dict of scalar momentum. Must have the same dtype as `var`.

    x2 : the dict of scalar grad. Must have the same dtype as `var`.

    out_var : the dict of var output.

    out_accum : the dict of accum output

    use_nesterov: bool. If true, use nesterov computing grad,
                 default value is False.

    kernel_name : cce kernel name, default value is "fused_mul_apply_momentum".

    Returns
    -------
    None
    """

    input_name_list = ['var', 'accum', 'lr', 'x1', 'momentum', 'x2']
    var, accum, lr, x1, momentum, x2 = _get_placeholder(
        [var, accum, lr, x1, momentum, x2], input_name_list)
    out_var, out_accum = _fused_mul_apply_momentum_compute(
        var, accum, lr, x1, momentum, x2, out_var, out_accum, use_nesterov)
    outs = [out_var, out_accum]
    build_list = [var, accum, lr, x1, momentum, x2, out_var, out_accum]

    with tvm.target.cce():
        sch = generic.auto_schedule(outs)
    config = {"name": kernel_name, "tensor_list": build_list}
    te.lang.cce.cce_build_code(sch, config)
