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
this file achieved the apply_adagrad_d which is a optimizer operator
to update weight, this file contains compute and schedule.

apply_adagrad_d

  Op_description :
    Update '*var' according to the Adagrad algorithm.

    # apply_adagrad_d(var,
    #   accum,
    #   lr,
    #   grad,
    #   var_out,
    #   accum_out,
    #   update_slots,
    #   kernel_name='apply_adagrad_d')

  Supportive_dtype_format :
    ['int32', 'int8', 'uint8', 'float32', 'float16']
    ['ND', 'NCHW', 'NHWC', 'NC1HWC0']

  Constraint :
    [1] All : the input tensors must have the same shape and type.
    [2] All : shape size limit is 2147483648.
"""
import te.lang.cce
from topi.cce import util
from te.platform.cce_conf import api_check_support
from te.utils.op_utils import *
from topi import generic
NUM_ZERO = 0.0

# pylint: disable=locally-disabled, too-many-arguments, unused-argument
# pylint: disable=too-many-locals, invalid-name
@fusion_manager.register("apply_adagrad_d")
def apply_adagrad_d_compute(var,
                            accum,
                            lr,
                            grad,
                            var_out,
                            accum_out,
                            update_slots,
                            kernel_name="apply_adagrad_d"):
    """
    Update '*var' according to the Adagrad algorithm.

    accum += grad ** 2
    var -= lr * grad / accum.sqrt()

    Parameters:
    ----------
    var: the dict of var, only support float16, float32

    accum: the dict of accum, only support float16, float32

    lr: the dict of lr, only support float16, float32

    grad: the dict of grad, only support float16, float32

    var_out: the dict of var output, only support float16, float32

    accum_out: the dict of accum output, only support float16, float32

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagrad".

    Returns
    -------
    None
    """
    shape_list = broadcast_shapes(
        te.lang.cce.util.shape_to_list(var.shape),
        te.lang.cce.util.shape_to_list(lr.shape),
        param_name_input1="input_var", param_name_input2="input_lr")

    input_dtype = var.dtype

    if input_dtype == "float16" and api_check_support("te.lang.cce.vadd",
                                                      "float32"):
        var = te.lang.cce.cast_to(var, "float32")
        accum = te.lang.cce.cast_to(accum, "float32")
        lr = te.lang.cce.cast_to(lr, "float32")
        grad = te.lang.cce.cast_to(grad, "float32")
    lr = te.lang.cce.broadcast(lr, shape_list[2])

    if update_slots is True:
        grad_square = te.lang.cce.vmul(grad, grad)
        accum = te.lang.cce.vadd(accum, grad_square)
    else:
        accum = te.lang.cce.vadds(accum, tvm.const(NUM_ZERO, "float32"))
    sqrtdata = te.lang.cce.vsqrt(accum)
    update = te.lang.cce.vdiv(grad, sqrtdata)
    lr_update = te.lang.cce.vmul(lr, update)
    output_var = te.lang.cce.vsub(var, lr_update)

    if input_dtype == "float16":
        output_var = te.lang.cce.cast_to(output_var, "float16")
        accum = te.lang.cce.cast_to(accum, "float16")

    return [output_var, accum]


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, REQUIRED_OUTPUT, OPTION_ATTR_BOOL, KERNEL_NAME)
def apply_adagrad_d(var,
                    accum,
                    lr,
                    grad,
                    var_out,
                    accum_out,
                    update_slots=True,
                    kernel_name="apply_adagrad_d"):
    """
    Update '*var' according to the Adagrad algorithm.

    accum += grad ** 2
    var -= lr * grad / accum.sqrt()

    Parameters:
    ----------
    var: the dict of var, only support float16, float32

    accum: the dict of accum, only support float16, float32

    lr: the dict of lr, only support float16, float32

    grad: the dict of grad, only support float16, float32

    var_out: the dict of var output, only support float16, float32

    accum_out: the dict of accum output, only support float16, float32

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagrad".

    Returns
    -------
    None
    """

    check_list = ('float16', 'float32')
    dtype = var.get('dtype')
    check_dtype(dtype, check_list, param_name="var")
    var_shape = var.get("shape")
    lr_shape = lr.get("shape")
    var_dtype = var.get("dtype")
    if len(lr_shape) != 1 or int(lr_shape[0]) != 1:
        raise RuntimeError("lr shape must be 1")

    util.compare_tensor_dict_key(var, accum, "shape")
    util.compare_tensor_dict_key(accum, grad, "shape")
    util.compare_tensor_dict_key(var, lr, "dtype")
    util.compare_tensor_dict_key(accum, grad, "dtype")

    shape_list = broadcast_shapes(var_shape, lr_shape, param_name_input1="input_x",
                                  param_name_input2="input_y")
    reshape_x, reshape_y = refine_shapes_for_broadcast(shape_list[0],
                                                       shape_list[1])

    var_data = tvm.placeholder(reshape_x, dtype=var_dtype, name="var")
    accum_data = tvm.placeholder(reshape_x, dtype=var_dtype, name="accum_data")
    lr_data = tvm.placeholder(reshape_y, dtype=var_dtype, name="lr_data")
    grad_data = tvm.placeholder(reshape_x, dtype=var_dtype, name="grad_data")
    res = apply_adagrad_d_compute(var_data, accum_data, lr_data,
                                  grad_data, var_out, accum_out, update_slots, kernel_name)
    inputlist = [var_data, accum_data, lr_data, grad_data]
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res)}

    te.lang.cce.cce_build_code(sch, config)

