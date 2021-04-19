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
tanh_compute
"""
# pylint: disable=too-many-lines
from te import tvm
import te.platform.cce_params as cce
import topi


# pylint: disable=too-many-statements
def tanh_compute(shape, input_x, symbol, impl_mode="high_performance"):
    """
    tanh compute
    shape : tensor shape
    input_x : tensor
    symbol : tensor symbol
    return: res, operations, scope
    """
    if impl_mode == "high_performance":
        return tanh_compute_high_performance(shape, input_x, symbol)
    else:
        return tanh_compute_high_precision(shape, input_x, symbol)


# pylint: disable=too-many-statements
def tanh_compute_high_precision(shape, input_x, symbol):
    """
    the function of tanh
    series for Hyperbolic Functions
        tanh x = x - x^3/3 + 2*x^5/15 -17x^7/315 ...
    Parameters
    ----------
    shape : tensor shape
    input_x : tensor
    symbol : tensor symbol
    Returns res, operations, scope
    -------
    """
    res = {}
    operation = {}
    scope = {}
    dtype_x = input_x.dtype

    const_1 = tvm.const(-1/3, dtype=dtype_x)
    const_2 = tvm.const(2/15, dtype=dtype_x)
    const_3 = tvm.const(-17/315, dtype=dtype_x)

    split_res, split_operation, split_scop = \
        tanh_split_input_by_val(shape, input_x, symbol)
    input_lt = split_res["input_lt_" + symbol]
    input_gt = split_res["input_gt_" + symbol]
    res.update(split_res)
    operation.update(split_operation)
    scope.update(split_scop)

    # x^2
    key = "x_sqr_" + symbol
    x_sqr = tvm.compute(shape, lambda *i: input_lt(*i) * input_lt(*i), name=key)
    res[key] = x_sqr
    operation[key] = "vector_mul"
    scope[key] = cce.scope_ubuf

    # x^3
    key = "val_1_" + symbol
    val_1 = tvm.compute(shape, lambda *i: input_lt(*i) * x_sqr(*i), name=key)
    res[key] = val_1
    operation[key] = "vector_mul"
    scope[key] = cce.scope_ubuf

    # -1/3 x^3
    key = "val_2_" + symbol
    val_2 = tvm.compute(shape, lambda *i: const_1 * val_1(*i), name=key)
    res[key] = val_2
    operation[key] = "vector_muls"
    scope[key] = cce.scope_ubuf

    # x-1/3x^3
    key = "res_1_" + symbol
    res_1 = tvm.compute(shape, lambda *i: input_lt(*i) + val_2(*i), name=key)
    res[key] = res_1
    operation[key] = "vector_add"
    scope[key] = cce.scope_ubuf

    # x^5
    key = "val_3_" + symbol
    val_3 = tvm.compute(shape, lambda *i: x_sqr(*i) * val_1(*i), name=key)
    res[key] = val_3
    operation[key] = "vector_mul"
    scope[key] = cce.scope_ubuf

    # 2/15 x^5
    key = "val_4_" + symbol
    val_4 = tvm.compute(shape, lambda *i: const_2 * val_3(*i), name=key)
    res[key] = val_4
    operation[key] = "vector_muls"
    scope[key] = cce.scope_ubuf

    # x-1/3x^3 + 2/15x^5
    key = "res_2_" + symbol
    res_2 = tvm.compute(shape, lambda *i: res_1(*i) + val_4(*i), name=key)
    res[key] = res_2
    operation[key] = "vector_add"
    scope[key] = cce.scope_ubuf

    # x^7
    key = "val_5_" + symbol
    val_5 = tvm.compute(shape, lambda *i: x_sqr(*i) * val_3(*i), name=key)
    res[key] = val_5
    operation[key] = "vector_mul"
    scope[key] = cce.scope_ubuf

    # -17/315x^7
    key = "val_6_" + symbol
    val_6 = tvm.compute(shape, lambda *i: const_3 * val_5(*i), name=key)
    res[key] = val_6
    operation[key] = "vector_mul"
    scope[key] = cce.scope_ubuf

    key = "res_3_" + symbol
    res_3 = tvm.compute(shape, lambda *i: res_2(*i) + val_6(*i), name=key)
    res[key] = res_3
    operation[key] = "vector_add"
    scope[key] = cce.scope_ubuf

    res_gt, operation_gt, scope_gt = tanh_compute_high_performance(shape,
                                                                   input_gt,
                                                                   symbol+"_gt")
    res.update(res_gt)
    operation.update(operation_gt)
    scope.update(scope_gt)

    key = "ub_tanh_" + symbol
    ub_tanh = tvm.compute(shape,
                          lambda *i: res_3(*i) +
                                     res_gt["ub_tanh_"+symbol+"_gt"](*i),
                          name=key)
    res[key] = ub_tanh
    operation[key] = "vector_add"
    scope[key] = cce.scope_ubuf

    return res, operation, scope


# pylint: disable=too-many-statements
def tanh_compute_high_performance(shape, input_x, symbol):
    """
    the function of tanh
    Parameters
    ----------
    shape : tensor shape
    input_x : tensor
    symbol : tensor symbol
    Returns
    -------
    """
    res = {}
    operation = {}
    scope = {}

    dtype_x = input_x.dtype
    const_one = tvm.const(1, dtype=dtype_x)
    const_neg_two = tvm.const(-2, dtype=dtype_x)
    const_fp32_min = tvm.const(2 ** (-126), dtype=dtype_x)

    key = "input_abs_" + symbol
    input_abs = tvm.compute(
        shape, lambda *i: tvm.abs(input_x(*i)), name=key)
    res[key] = input_abs
    operation[key] = "vector_abs"
    scope[key] = cce.scope_ubuf

    key = "power_val_" + symbol
    power_val = tvm.compute(
        shape, lambda *i: input_abs(*i) * const_neg_two, name=key)
    res[key] = power_val
    operation[key] = "vector_muls"
    scope[key] = cce.scope_ubuf

    if dtype_x == "float32":
        key = "exp_val_fp16_" + symbol
        exp_val_fp16 = tvm.compute(
            shape, lambda *i: topi.cast(power_val(*i), "float16"), name=key)
        res[key] = exp_val_fp16
        operation[key] = "vector_conv"
        scope[key] = cce.scope_ubuf

        key = "exp_val_" + symbol
        exp_val = tvm.compute(
            shape, lambda *i: tvm.exp(exp_val_fp16(*i)), name=key)
        res[key] = exp_val
        operation[key] = "vector_exp"
        scope[key] = cce.scope_ubuf

        key = "exp_val_fp32_" + symbol
        exp_val_fp32 = tvm.compute(
            shape, lambda *i: topi.cast(exp_val(*i), "float32"), name=key)
        res[key] = exp_val_fp32
        operation[key] = "vector_conv"
        scope[key] = cce.scope_ubuf

        exp_val_true = exp_val_fp32
    else:
        key = "exp_val_" + symbol
        exp_val = tvm.compute(
            shape, lambda *i: tvm.exp(power_val(*i)), name=key)
        res[key] = exp_val
        operation[key] = "vector_exp"
        scope[key] = cce.scope_ubuf
        exp_val_true = exp_val

    key = "up_val_tmp_" + symbol
    up_val_tmp = tvm.compute(
        shape, lambda *i: exp_val_true(*i) * input_x(*i), name=key)
    res[key] = up_val_tmp
    operation[key] = "vector_mul"
    scope[key] = cce.scope_ubuf

    key = "up_val_" + symbol
    up_val = tvm.compute(
        shape, lambda *i: input_x(*i) - up_val_tmp(*i), name=key)
    res[key] = up_val
    operation[key] = "vector_sub"
    scope[key] = cce.scope_ubuf

    key = "input_tmp_" + symbol
    input_tmp = tvm.compute(
        shape, lambda *i: input_abs(*i) + const_fp32_min, name=key)
    res[key] = input_tmp
    operation[key] = "vector_adds"
    scope[key] = cce.scope_ubuf

    key = "down_val_tmp_" + symbol
    down_val_tmp = tvm.compute(
        shape, lambda *i: exp_val_true(*i) + const_one, name=key)
    res[key] = down_val_tmp
    operation[key] = "vector_adds"
    scope[key] = cce.scope_ubuf

    key = "down_val_" + symbol
    down_val = tvm.compute(
        shape, lambda *i: down_val_tmp(*i) * input_tmp(*i), name=key)
    res[key] = down_val
    operation[key] = "vector_mul"
    scope[key] = cce.scope_ubuf

    ub_rec = tvm.compute(
        shape, lambda *i: const_one / down_val(*i), name="ub_rec_" + symbol)
    res["ub_rec_" + symbol] = ub_rec
    operation["ub_rec_" + symbol] = "vector_rec"
    scope["ub_rec_" + symbol] = cce.scope_ubuf

    iter_num = 1
    tensor_list, scope_list, emit_list = newton_iteration(
        shape, ub_rec, down_val, symbol, iter_num)
    res.update(tensor_list)
    operation.update(emit_list)
    scope.update(scope_list)

    newton_res = tensor_list["tmp_" + symbol + str(iter_num - 1)]

    ub_tanh = tvm.compute(
        shape,
        lambda *i: up_val(*i) * newton_res(*i),
        name="ub_tanh_" + symbol)
    res["ub_tanh_" + symbol] = ub_tanh
    operation["ub_tanh_" + symbol] = "vector_mul"
    scope["ub_tanh_" + symbol] = cce.scope_ubuf

    return res, operation, scope


# pylint: disable=too-many-statements
def tanh_split_input_by_val(shape, input_x, symbol):
    """
    split input into two tensor by 0.5
    shape : tensor shape
    input_x : tensor
    symbol : tensor symbol
    return: res, operations, scope
    """
    res = {}
    operation = {}
    scope = {}
    dtype_x = input_x.dtype
    const_zero = tvm.const(0.0, dtype="float16")
    const_0 = tvm.const(0.5, dtype="float16")

    key = "input_abs_" + symbol
    input_abs = tvm.compute(shape, lambda *i: tvm.abs(input_x(*i)), name=key)
    res[key] = input_abs
    operation[key] = "vector_abs"
    scope[key] = cce.scope_ubuf

    # vcmp only support fp16
    if dtype_x == "float32":
        key = "cmp_val_fp16_" + symbol
        cmp_val_fp16 = tvm.compute(
            shape, lambda *i: topi.cast(input_abs(*i), "float16"), name=key)
        res[key] = cmp_val_fp16
        operation[key] = "vector_conv"
        scope[key] = cce.scope_ubuf

        key = "input_val_fp16_" + symbol
        input_val_fp16 = tvm.compute(
            shape, lambda *i: topi.cast(input_x(*i), "float16"), name=key)
        res[key] = input_val_fp16
        operation[key] = "vector_conv"
        scope[key] = cce.scope_ubuf

        key = "input_gt_fp16_" + symbol
        input_gt_fp16 = \
            tvm.compute(shape,
                        lambda *i: tvm.select(cmp_val_fp16(*i) > const_0,
                                              input_val_fp16(*i), const_zero),
                        name=key)
        res[key] = input_gt_fp16
        operation[key] = "vector_select_gt"
        scope[key] = cce.scope_ubuf

        key = "input_lt_fp16_" + symbol
        input_lt_fp16 = \
            tvm.compute(shape,
                        lambda *i: tvm.select(cmp_val_fp16(*i) <= const_0,
                                              input_val_fp16(*i), const_zero),
                        name=key)
        res[key] = input_lt_fp16
        operation[key] = "vector_select_le"
        scope[key] = cce.scope_ubuf

        key = "input_gt_" + symbol
        input_gt = tvm.compute(
            shape, lambda *i: topi.cast(input_gt_fp16(*i), "float32"), name=key)
        res[key] = input_gt
        operation[key] = "vector_conv"
        scope[key] = cce.scope_ubuf

        key = "input_lt_" + symbol
        input_lt = tvm.compute(
            shape, lambda *i: topi.cast(input_lt_fp16(*i), "float32"), name=key)
        res[key] = input_lt
        operation[key] = "vector_conv"
        scope[key] = cce.scope_ubuf
    else:
        key = "input_gt_" + symbol
        input_gt = tvm.compute(shape,
                               lambda *i: tvm.select(input_abs(*i) > const_0,
                                                     input_x(*i), const_zero),
                               name=key)
        res[key] = input_gt
        operation[key] = "vector_select_gt"
        scope[key] = cce.scope_ubuf

        key = "input_lt_" + symbol
        input_lt = tvm.compute(shape,
                               lambda *i: tvm.select(input_abs(*i) <= const_0,
                                                     input_x(*i), const_zero),
                               name=key)
        res[key] = input_lt
        operation[key] = "vector_select_le"
        scope[key] = cce.scope_ubuf

    return res, operation, scope


# pylint: disable=too-many-locals
def newton_iteration(shape, tensor_x_rec, tensor_x, symbol, iter_num):
    """
    the function of newton_iteration
    Parameters
    ----------
    shape: tensor shape
    tensor_x_rec: tensor
    tensor_x: tensor
    symbol: tensor symbol

    Returns
    -------
    tensor_list: dict
    scope_list: dict
    emit_list: dict
    """
    dtype_c = tensor_x_rec.dtype
    num_two = tvm.const(2, dtype=dtype_c)
    neg_one = tvm.const(-1, dtype=dtype_c)
    tmp = tensor_x_rec

    tensor_list = {}
    scope_list = {}
    emit_list = {}
    tmp_mul = None
    tmp_neg = None
    tmp_add = None
    for index in range(0, iter_num):
        key = "tmp_mul_" + symbol + str(index)
        tmp_mul = tvm.compute(
            shape, lambda *i: tensor_x(*i) * tmp(*i), name=key)
        tensor_list[key] = tmp_mul
        scope_list[key] = cce.scope_ubuf
        emit_list[key] = "vector_mul"

        key = "tmp_neg_" + symbol + str(index)
        tmp_neg = tvm.compute(
            shape, lambda *i: tmp_mul(*i) * neg_one, name=key)
        tensor_list[key] = tmp_neg
        scope_list[key] = cce.scope_ubuf
        emit_list[key] = "vector_muls"

        key = "tmp_add_" + symbol + str(index)
        tmp_add = tvm.compute(
            shape, lambda *i: tmp_neg(*i) + num_two, name=key)
        tensor_list[key] = tmp_add
        scope_list[key] = cce.scope_ubuf
        emit_list[key] = "vector_adds"

        key = "tmp_" + symbol + str(index)
        tmp = tvm.compute(shape, lambda *i: tmp_add(*i) * tmp(*i), name=key)
        tensor_list[key] = tmp
        scope_list[key] = cce.scope_ubuf
        emit_list[key] = "vector_mul"

    return tensor_list, scope_list, emit_list
