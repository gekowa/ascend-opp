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
fused_mul_apply_momentum_extern

  Op_description :
    Update '*var' according to the ApplyMomentum algorithm.
"""
import te.lang.cce
from impl.util.util_apply_op_schedule import ApplyOpConfig
from impl.util.util_apply_op_schedule import common_apply_op_process
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import op_utils


# pylint: disable=too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("fused_mul_apply_momentum_extern")
def _fused_mul_apply_momentum_extern_compute(var,
                                             accum,
                                             lr,
                                             x1,
                                             momentum,
                                             x2,
                                             var_copy,
                                             out_fp32,
                                             out_fp16,
                                             out_accum,
                                             use_nesterov,
                                             kernel_name="fused_mul_apply_"
                                             "momentum_extern"):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + x1 * x2
    if use_nesterov is True:
        var -= x1 * x2 * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : mutable tensor var. Dtype is float32

    accum: mutable tensor accum.

    lr : scalar lr.

    x1: tensor x.

    momentum : scalar momentum.

    x2: tensor y.

    var_copy : mutable tensor var. Dtype is float16

    out_fp32 : the dict of output. Dtype is float32.

    out_fp16 : the dict of output. Dtype is float16.

    out_accum : the dict of output. Dtype is same as input accum

    use_nesterov: bool. If true, use nesterov computing grad,
                  default value is False.

    kernel_name : cce kernel name, default value is
                 "fused_mul_apply_momentum_extern" (optional).

    Returns:
    -------
    None
    """
    # cast to float32 for higher accuracy
    dtype = accum.dtype
    if dtype == "float16":
        accum = te.lang.cce.cast_to(accum, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        x1 = te.lang.cce.cast_to(x1, "float32")
        x2 = te.lang.cce.cast_to(x2, "float32")
        momentum = te.lang.cce.cast_to(momentum, "float32")

    # calc grad
    grad = tvm.compute(x1.shape,
                       lambda *indice: x1(*indice) * x2[0],
                       tag='elewise_single_VS_mul')
    # update accum
    accum_delta = tvm.compute(accum.shape,
                              lambda *indice: accum(*indice) * momentum[0],
                              tag='elewise_single_VS_mul')
    accum_t = te.lang.cce.vadd(accum_delta, grad)

    # update var
    if use_nesterov:
        var_delta = tvm.compute(grad.shape,
                                lambda *indice: grad(*indice) * lr[0],
                                tag='elewise_single_VS_mul')
        var_delta_2 = tvm.compute(
            accum_t.shape,
            lambda *indice: accum_t(*indice) * momentum[0],
            tag='elewise_single_VS_mul')
        var_delta_2 = tvm.compute(var_delta_2.shape,
                                  lambda *indice: var_delta_2(*indice) * lr[0],
                                  tag='elewise_single_VS_mul')
        var_delta = te.lang.cce.vadd(var_delta, var_delta_2)
        var_t_fp32 = te.lang.cce.vsub(var, var_delta)

        var_delta_fp16 = te.lang.cce.cast_to(var_delta, "float16")
        var_t_fp16 = te.lang.cce.vsub(var_copy, var_delta_fp16)
    else:
        var_delta = tvm.compute(accum_t.shape,
                                lambda *indice: accum_t(*indice) * lr[0],
                                tag='elewise_single_VS_mul')
        var_t_fp32 = te.lang.cce.vsub(var, var_delta)
        var_delta_fp16 = te.lang.cce.cast_to(var_delta, "float16")
        var_t_fp16 = te.lang.cce.vsub(var_copy, var_delta_fp16)

    if dtype == "float16":
        accum_t = te.lang.cce.cast_to(accum_t, "float16")
    var_out_fp32 = te.lang.cce.vadds(var_t_fp32,
                                     tvm.const(0.0, var_t_fp32.dtype))
    var_out_fp16 = te.lang.cce.vadds(var_t_fp16,
                                     tvm.const(0.0, var_t_fp16.dtype))
    var_out_accum = te.lang.cce.vadds(accum_t, tvm.const(0.0, accum_t.dtype))

    def _compute(*index):
        return accum_t(*index), var_t_fp32(*index), var_t_fp16(
            *index), var_out_fp32(*index), var_out_fp16(*index), var_out_accum(
                *index)

    return tvm.compute(accum.shape, _compute, name="outputs")


# pylint: disable=too-many-arguments,too-many-locals
@op_utils.check_op_params(op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_INPUT,
                          op_utils.REQUIRED_INPUT, op_utils.REQUIRED_OUTPUT,
                          op_utils.REQUIRED_OUTPUT, op_utils.REQUIRED_OUTPUT,
                          op_utils.OPTION_ATTR_BOOL, op_utils.KERNEL_NAME)
def fused_mul_apply_momentum_extern(var,
                                    accum,
                                    lr,
                                    x1,
                                    momentum,
                                    x2,
                                    var_copy,
                                    out_fp32,
                                    out_fp16,
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
    var : the dict of mutable tensor var, Dtype is float32.

    accum: the dict of mutable tensor accum.

    lr : the dict of scalar lr.

    x1 : the dict of tensor grad.

    momentum : the dict of scalar momentum.

    x2 : the dict of tensor grad.

    var_copy : the dict of mutable tensor var, Dtype is float16.

    out_fp32 : the dict of output. Dtype is float32.

    out_fp16 : the dict of output. Dtype is float16.

    out_accum : the dict of output. Dtype is same as input accum.

    use_nesterov: bool. If true, use nesterov computing grad,
                 default value is False.

    kernel_name : cce kernel name, default value is "fused_mul_apply_momentum".

    Returns
    -------
    None
    """
    var_dtype = var.get("dtype")
    op_utils.check_dtype(var_dtype, ("float32", ), param_name="var")
    var_copy_dtype = var_copy.get("dtype")
    op_utils.check_dtype(var_copy_dtype, ("float16", ), param_name="var_copy")
    input_dict = (var, accum, lr, x1, momentum, x2, var_copy)
    outputs = [out_fp32, out_fp16, out_accum]

    args = ApplyOpConfig.TensorArgs(
        input_dict,
        _fused_mul_apply_momentum_extern_compute,
        outputs,
        10 if use_nesterov else 8,
    )
    name = ApplyOpConfig.TensorName(all=('var', 'accum', 'lr', 'x1',
                                         'momentum', 'x2', 'var_copy'),
                                    scalar=('lr', 'momentum', 'x2'),
                                    reuse=('accum', 'var', 'var_copy'))
    options = ApplyOpConfig.TensorOptions(attrs=use_nesterov)

    common_apply_op_process(ApplyOpConfig(args, name, options),
                            kernel_name,
                            same_flag=False)
