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
dynamic_lstm
"""
import os
import json
import stat
from functools import reduce as functools_reduce
import re

import te.lang.cce
from topi.cce import util
from te import tvm
from te import platform as cce
from te.platform.cce_build import build_config
from te.utils.error_manager import error_manager_vector


def sigmoid_compute(input_x):
    """
    calculating sigmoid
    """
    data_input = input_x
    dtype = input_x.dtype
    exp_support = cce.cce_conf.api_check_support(
        "te.lang.cce.vexp", "float32")
    mul_support = cce.cce_conf.api_check_support(
        "te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_specific_reson("DynamicLSTM",
                                                      "Input dtype only support float16 while input dtype is float32")

    const_num_neg_one = tvm.const(-1, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    tmp_negative = te.lang.cce.vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative = te.lang.cce.cast_to(tmp_negative, "float16")
    tmp_exp = te.lang.cce.vexp(tmp_negative)
    if dtype == "float32" and not exp_support:
        tmp_exp = te.lang.cce.cast_to(tmp_exp, "float32")
    tmp_sum = te.lang.cce.vadds(tmp_exp, const_num_one)
    if dtype == "float32":
        inp_shape = tmp_sum.shape
        tensor_one = te.lang.cce.broadcast(tvm.const(1, dtype), inp_shape)
        res = te.lang.cce.vdiv(tensor_one, tmp_sum)
    else:
        res = te.lang.cce.vrec(tmp_sum)

    return res


def tanh_compute(input_x):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)
    """
    input_dtype = input_x.dtype
    # positive min float32 value
    min_fp_data = 2 ** (-126)
    const_dtype = input_dtype
    # positive min float16 value
    if input_dtype == "float16":
        min_fp_data = 2 ** (-14)

    has_improve_precision = False

    if input_dtype == "float16" and \
            cce.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        input_x = te.lang.cce.cast_to(input_x, "float32")
        has_improve_precision = True
        const_dtype = "float32"

    input_abs = te.lang.cce.vabs(input_x)
    power_val = te.lang.cce.vmuls(input_abs, tvm.const(-2, const_dtype))
    exp_val = te.lang.cce.vexp(power_val)

    up_val_tmp = te.lang.cce.vmul(exp_val, input_x)
    up_val = te.lang.cce.vsub(input_x, up_val_tmp)

    input_x_tmp = te.lang.cce.vadds(input_abs, min_fp_data)
    down_val_tmp = te.lang.cce.vadds(exp_val, tvm.const(1, const_dtype))
    down_val = te.lang.cce.vmul(down_val_tmp, input_x_tmp)

    res = te.lang.cce.vdiv(up_val, down_val)

    if has_improve_precision:
        res = te.lang.cce.cast_to(res, "float16")

    return res


def get_emit_insn_map(tensor):
    """
    get tensor's emit_insn key
    """
    insn_map = {"elewise_single_cast": "vector_conv",
                "elewise_single_VS_max": "vector_maxs",
                "elewise_single_VS_min": "vector_mins",
                "elewise_single_log": "vector_ln",
                "elewise_single_exp": "vector_exp",
                "elewise_single_rec": "vector_rec",
                "elewise_single_relu": "vector_relu",
                "elewise_single_abs": "vector_abs",
                "elewise_single_not": "vector_not",
                "elewise_single_sqrt": "vector_sqrt",
                "elewise_single_rsqrt": "vector_rsqrt",
                "elewise_binary_mul": "vector_mul",
                "elewise_single_VS_mul": "vector_muls",
                "elewise_binary_div": "vector_div",
                "elewise_binary_add": "vector_add",
                "elewise_single_VS_add": "vector_adds",
                "elewise_binary_min": "vector_min",
                "elewise_binary_max": "vector_max",
                "elewise_binary_vcmpv_gt": "vector_gt",
                "elewise_binary_vcmpv_ge": "vector_ge",
                "elewise_binary_vcmpv_lt": "vector_lt",
                "elewise_binary_vcmpv_le": "vector_le",
                "elewise_binary_vcmpv_eq": "vector_eq",
                "elewise_binary_vcmpv_ne": "vector_ne",
                "elewise_binary_or": "vector_or",
                "elewise_binary_and": "vector_and",
                "elewise_multiple_mla": "vector_multiple",
                "elewise_multiple_madd": "vector_multiple",
                "elewise_multiple_maddrelu": "vector_multiple",
                "broadcast_for_tensor": "broadcast_for_tensor",
                "elewise_binary_sub": "vector_sub",
                "broadcast": "broadcast"}
    if tensor.op.tag.find("|") != -1:
        str_list = tensor.op.tag.split("|")
        insn = insn_map.get(str_list[0])
    else:
        insn = insn_map.get(tensor.op.tag)
    return insn


lstm_tiling_map = {
    "32_512_256": (2, 8, 8, 2, 8, 8),
    "16_1536_2048": (1, 32, 12, 1, 32, 12),
}


def _get_lstm_tiling(m_size, k_size, n_size):
    """
    get lstm tiling
    :return:
    """
    key = "_".join(str(i) for i in (m_size*16, k_size*16, n_size*16))

    if key not in lstm_tiling_map:
        error_manager_vector.raise_err_specific_reson("DynamicLSTM", "Unsupported lstm shape tiling!")
    return lstm_tiling_map[key]


def check_dtype(input_x, weight, bias, output_h):
    """
    check parameters dtype
    :return:
    """
    if input_x["dtype"] != "float32" or weight["dtype"] != "float32" \
       or weight["dtype"] != "float32" or output_h["dtype"] != "float32":
        error_manager_vector.raise_err_specific_reson("DynamicLSTM", "x, w, b, output_h supports dtype float32 only!")
    return


def check(shape_x_input, shape_w_input, shape_b_input, shape_output):
    """
    check parameters
    :return:
    """
    if shape_x_input[2] != shape_output[2]:
        error_manager_vector.raise_err_specific_reson("DynamicLSTM", "x, output_h shape is wrong, please check!")
    if shape_w_input[0] != shape_x_input[1] + shape_output[1]:
        error_manager_vector.raise_err_specific_reson("DynamicLSTM", "x, w, output_h shape is wrong, please check!")
    if shape_w_input[1] != 4 * shape_output[1]:
        error_manager_vector.raise_err_specific_reson("DynamicLSTM", "w, output_h shape is wrong, please check!")
    if (shape_b_input[0] + 15) // 16 != shape_w_input[1]:
        error_manager_vector.raise_err_specific_reson("DynamicLSTM", "w, b shape is wrong, please check!")
    return


# pylint: disable=too-many-arguments,too-many-locals,invalid-name
@util.check_input_type(dict, dict, dict, dict, str)
def dynamic_lstm(input_x, weight, bias,
                 output_h, kernel_name="dynamic_lstm"):
    """
    x : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float32,
        the format can be [FRACTAL_NZ]
    w : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float32,
        the format can be [FRACTAL_ZN_LSTM]
    b : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float32,
        the format can be [ND]
    output_h : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float32,
        the format can be [FRACTAL_NZ]
    """

    check_dtype(input_x, weight, bias, output_h)

    shape_x_input = input_x.get("shape")
    shape_w_input = weight.get("shape")
    shape_b_input = bias.get("shape")
    shape_output = output_h.get("shape")

    check(shape_x_input, shape_w_input, shape_b_input, shape_output)

    scan_one_num = 1
    t_size = shape_x_input[0] + scan_one_num
    m_size = shape_x_input[2]
    k_size = shape_w_input[0]
    n_size = shape_w_input[1]
    hidden_size = shape_output[1]
    block_size = n_size // hidden_size
    in_x = k_size - hidden_size

    shape_b = (1, k_size, block_size, hidden_size, 16, 16)
    shape_c = (1, block_size, hidden_size, m_size, 16, 16)
    shape_bias = (1, block_size, hidden_size, 1, 1, 16)
    shape_x = (t_size, in_x, m_size, 16, 16)
    shape_h = (1, k_size - in_x, m_size, 16, 16)
    shape_i = (1, hidden_size, m_size, 16, 16)
    shape_i_t = (t_size, hidden_size, m_size, 16, 16)

    core_num = cce.get_soc_spec("CORE_NUM")
    # one core use 4 int64 that is 32B align
    shape_sync = (4 * core_num,)

    k0_size = 16

    input_dtype = input_x.get("dtype")
    data_dtype = 'float16'
    sync_dtype = 'int64'

    # define placeholder
    input_x = tvm.placeholder(shape_x, dtype=input_dtype, name='input_x')
    weight = tvm.placeholder(shape_b, dtype=input_dtype, name='weight')
    bias = tvm.placeholder(shape_bias, name='bias', dtype=input_dtype)
    s_state_h = tvm.placeholder(shape_h, dtype=input_dtype, name='state_h')
    s_state_c = tvm.placeholder(shape_i, dtype=input_dtype, name='state_c')

    sync0 = tvm.placeholder(shape_sync, name="sync0", dtype='int64')

    # compute

    # weight need first to ub and cast to float16
    weight_ub = \
        tvm.compute(
            shape_b,
            lambda *indices: weight(*indices),
            name="weight_ub")

    weight_fp16 = \
        tvm.compute(shape_b,
                    lambda *indices: weight_ub(*indices).astype(data_dtype),
                    name="weight_fp16")

    # input and s_state_h need first to ub and cast to float16
    shape_a_z_bigz = (t_size, m_size, k_size, 16, 16)

    # input and s_start_h is Nz, need trans to zZ
    # so change axis 1 and 2
    a_ub = tvm.compute(shape_a_z_bigz,
                       lambda *indice:
                       tvm.select(indice[2] < in_x,
                                  input_x[indice[0],
                                          indice[2],
                                          indice[1],
                                          indice[3],
                                          indice[4]],
                                  s_state_h[0,
                                            indice[2] - in_x,
                                            indice[1],
                                            indice[3],
                                            indice[4]]
                                  ),
                       name="a_ub", tag="concat")

    shape_a_z_bigz_1 = (1, m_size, k_size, 16, 16)

    a_ub_fp16 = \
        tvm.compute(shape_a_z_bigz_1,
                    lambda *indices: a_ub(*indices).astype(data_dtype),
                    name="a_ub_fp16")

    a_l1 = tvm.compute(shape_a_z_bigz_1,
                       lambda *indices: a_ub_fp16(*indices),
                       name='a_l1')
    b_l1 = tvm.compute(shape_b,
                       lambda *indices: weight_fp16(*indices),
                       name='b_l1')

    # shape_a_z_bigz_1 = (1, m_size, k_size, 16, 16)
    a_l0a = tvm.compute(shape_a_z_bigz, lambda *indices: a_l1(*indices), name="a_l0a")
    b_l0b = tvm.compute(shape_b, lambda *indices: b_l1(*indices), name="b_l0b")

    k1 = tvm.reduce_axis((0, k_size), name='k1')
    k0 = tvm.reduce_axis((0, k0_size), name='k0')

    c_l0c = tvm.compute(shape_c,
                        lambda t, nb_0, nb_1, mb, mp, np:
                        tvm.sum((a_l0a[t, mb, k1, mp, k0] * \
                                b_l0b[t, k1, nb_0, nb_1, np, k0]) \
                                .astype('float32'),
                                axis=[k1, k0]),
                        name='c_l0c')

    c_ub = tvm.compute(shape_c, lambda *indices: c_l0c(*indices), name="c_ub")

    bias_ub = tvm.compute(shape_bias,
                          lambda *indices: bias(*indices),
                          name='bias_ub')

    bias_bc_ub = te.lang.cce.broadcast(bias_ub, shape_c)
    c_ub_bias = te.lang.cce.vadd(c_ub, bias_bc_ub)

    # split matmul res
    i_t_index = 0
    j_t_index = 1
    f_t_index = 2
    o_t_index = 3
    i_t = \
        tvm.compute(shape_i,
                    lambda t, i, j, k, l: c_ub_bias(t, i_t_index, i, j, k, l),
                    name="i_t")
    j_t = \
        tvm.compute(shape_i,
                    lambda t, i, j, k, l: c_ub_bias(t, j_t_index, i, j, k, l),
                    name="j_t")
    f_t = \
        tvm.compute(shape_i,
                    lambda t, i, j, k, l: c_ub_bias(t, f_t_index, i, j, k, l),
                    name="f_t")
    o_t = \
        tvm.compute(shape_i,
                    lambda t, i, j, k, l: c_ub_bias(t, o_t_index, i, j, k, l),
                    name="o_t")

    f_t_sigmoid = sigmoid_compute(f_t)
    i_t_sigmoid = sigmoid_compute(i_t)
    o_t_sigmoid = sigmoid_compute(o_t)
    j_t_tanh = tanh_compute(j_t)

    c_t_tmp1 = te.lang.cce.vmul(s_state_c, f_t_sigmoid)
    c_t_tmp2 = te.lang.cce.vmul(j_t_tanh, i_t_sigmoid)
    update_c = te.lang.cce.vadd(c_t_tmp1, c_t_tmp2)

    update_c_gm = tvm.compute(shape_i_t,
                              lambda t, i, j, k, l: update_c(0, i, j, k, l),
                              name="update_c_gm")

    c_t_tanh = tanh_compute(update_c)

    update_h = te.lang.cce.vmul(c_t_tanh, o_t_sigmoid)
    update_h_gm = tvm.compute(shape_i_t,
                              lambda t, i, j, k, l: update_h(0, i, j, k, l),
                              name="update_h_gm")

    update_hc_vn = \
        tvm.compute(
            shape_i_t,
            lambda t, i, j, k, l: update_c_gm(0, i, j, k, l) +\
                                  update_h_gm(t, i, j, k, l),
            name="update_hc_vn")

    update_c_gm_vn = \
        tvm.compute(
            shape_i_t,
            lambda t, i, j, k, l: update_hc_vn(0, i, j, k, l),
            name="update_c_gm_vn")

    update_h_gm_vn = \
        tvm.compute(
            shape_i_t,
            lambda t, i, j, k, l: update_hc_vn(0, i, j, k, l),
            name="update_h_gm_vn")

    update_c_ub = \
        tvm.compute(
            shape_i,
            lambda t, i, j, k, l: update_c_gm_vn(t, i, j, k, l),
            name="update_c_ub")

    update_c_gm_2 = \
        tvm.compute(shape_i_t,
                    lambda t, i, j, k, l: update_c_ub(0, i, j, k, l),
                    name="update_c_gm_2")
    update_h_ub = \
        tvm.compute(
            shape_i,
            lambda t, i, j, k, l: update_h_gm_vn(t, i, j, k, l),
            name="update_h_ub")

    update_h_gm_2 = \
        tvm.compute(
            shape_i_t,
            lambda t, i, j, k, l: update_h_ub(0, i, j, k, l) +\
                                  update_c_gm_2(t, i, j, k, l),
            name="update_h_gm_2")

    update_h_gm_2_dummy = \
        tvm.compute(shape_i_t,
                    lambda t, i, j, k, l: update_h_gm_2(t, i, j, k, l),
                    name="update_h_gm_2_dummy")

    # state init
    init_shape = (1, hidden_size, m_size, 16, 16)

    s_state_h_ub = \
        tvm.compute(shape_h,
                    lambda *indices: tvm.const(0.0, dtype=input_dtype),
                    name='s_state_h_ub')
    s_state_c_ub = \
        tvm.compute(shape_i,
                    lambda *indices: tvm.const(0.0, dtype=input_dtype),
                    name='s_state_c_ub')

    s_init_h = \
        tvm.compute(
            init_shape,
            lambda _, i, j, k, l: s_state_h_ub[0, i, j, k, l],
            name="s_init_h")

    s_init_c = \
        tvm.compute(
            init_shape,
            lambda _, i, j, k, l: s_state_c_ub[0, i, j, k, l],
            name="s_init_c")

    # scan
    scan_h, scan_c = tvm.scan(
        [s_init_h, s_init_c],
        [update_h_ub, update_c_ub],
        [s_state_h, s_state_c],
        scan_update=[update_h_gm_2, update_h_gm_2_dummy],
        name="lstm_scan")

    # end compute

    # schedule
    s = tvm.create_schedule([scan_h.op, scan_c.op])

    new_build_list = [input_x, weight, bias, update_h_gm, update_c_gm,
                      sync0, update_h_gm_vn, update_c_gm_vn]

    def gen_reversed_subgraph_list(out_tensor, tensor_list):
        """
        traverse tensors by Depth-First-Search
        """
        if out_tensor is None:
            return
        stack = [out_tensor]
        visited_list = []
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor not in visited_list:
                    stack.append(in_tensor)
                    if "elewise" in in_tensor.op.tag or \
                            "broadcast" == in_tensor.op.tag:
                        if in_tensor not in tensor_list:
                            tensor_list.append(in_tensor)

    elewise_tensors = []
    gen_reversed_subgraph_list(update_h_gm, elewise_tensors)

    barrier_tensor = c_ub_bias
    elewise_before_barrier_tensors = [bias_bc_ub]

    # set scope
    s[a_l1].set_scope(cce.scope_cbuf)
    s[b_l1].set_scope(cce.scope_cbuf)
    s[a_l0a].set_scope(cce.scope_ca)
    s[b_l0b].set_scope(cce.scope_cb)
    s[c_l0c].set_scope(cce.scope_cc)
    s[c_ub].set_scope(cce.scope_ubuf)
    s[s_init_h].set_scope(cce.scope_ubuf)
    s[bias_ub].set_scope(cce.scope_ubuf)
    s[bias_bc_ub].set_scope(cce.scope_ubuf)
    s[scan_h].set_scope(cce.scope_ubuf)
    s[scan_c].set_scope(cce.scope_ubuf)
    s[update_h_ub].set_scope(cce.scope_ubuf)
    s[update_c_ub].set_scope(cce.scope_ubuf)
    s[s_state_h_ub].set_scope(cce.scope_ubuf)
    s[s_state_c_ub].set_scope(cce.scope_ubuf)

    s[weight_ub].set_scope(cce.scope_ubuf)
    s[weight_fp16].set_scope(cce.scope_ubuf)
    s[a_ub].set_scope(cce.scope_ubuf)
    s[a_ub_fp16].set_scope(cce.scope_ubuf)

    for tensor in elewise_tensors:
        s[tensor].set_scope(cce.scope_ubuf)

    # compute inline
    compute_inline_tensors = [i_t, j_t, f_t, o_t]
    for tensor in compute_inline_tensors:
        s[tensor].compute_inline()

    # matmul tiling
    factor_l1_m, factor_l1_n, factor_l1_k, \
    factor_l0_m, factor_l0_n, factor_l0_k = \
        _get_lstm_tiling(m_size, k_size, n_size)
    l1_n_outer, l1_n_inner = \
        s[c_l0c].split(c_l0c.op.axis[2],
                       factor=factor_l1_n // block_size)

    l1_m_outer, l1_m_inner = \
        s[c_l0c].split(c_l0c.op.axis[3],
                       factor=factor_l1_m)
    l1_k_outer, l1_k_inner = \
        s[c_l0c].split(c_l0c.op.reduce_axis[0],
                       factor=factor_l1_k)

    l0_n_outer, l0_n_inner = s[c_l0c].split(l1_n_inner,
                                            factor=factor_l0_n)
    l0_m_outer, l0_m_inner = s[c_l0c].split(l1_m_inner,
                                            factor=factor_l0_m)
    l0_k_outer, l0_k_inner = s[c_l0c].split(l1_k_inner,
                                            factor=factor_l0_k)

    s[c_l0c].reorder(l1_n_outer, c_l0c.op.axis[1],
                     l1_m_outer, l1_k_outer,
                     l0_n_outer, l0_m_outer, l0_k_outer,
                     l0_n_inner, l0_m_inner, c_l0c.op.axis[3 + 1],
                     c_l0c.op.axis[4 + 1], l0_k_inner,
                     c_l0c.op.reduce_axis[1])

    s[weight_ub].compute_at(s[c_l0c], l1_k_outer)
    s[weight_fp16].compute_at(s[c_l0c], l1_k_outer)
    s[a_ub].compute_at(s[c_l0c], l1_k_outer)
    s[a_ub_fp16].compute_at(s[c_l0c], l1_k_outer)

    s[a_l0a].compute_at(s[c_l0c], l0_k_outer)
    s[b_l0b].compute_at(s[c_l0c], l0_k_outer)
    s[a_l1].compute_at(s[c_l0c], l1_k_outer)
    s[b_l1].compute_at(s[c_l0c], l1_k_outer)

    ub_n_outer, ub_n_inner = \
        s[c_ub].split(c_ub.op.axis[2],
                      factor=factor_l1_n // block_size)

    ub_m_outer, ub_m_inner = s[c_ub].split(c_ub.op.axis[3],
                                           factor=factor_l1_m)
    s[c_ub].reorder(ub_n_outer, c_ub.op.axis[1], ub_m_outer,
                    ub_n_inner, ub_m_inner, c_ub.op.axis[4],
                    c_ub.op.axis[5])

    s[c_l0c].compute_at(s[c_ub], ub_n_outer)

    # elewise compute_at
    barrier_outer, barrier_inner = \
        s[barrier_tensor].split(barrier_tensor.op.axis[2],
                                factor=factor_l1_n // block_size)

    s[barrier_tensor].reorder(
        barrier_tensor.op.axis[0], barrier_outer,
        barrier_tensor.op.axis[1], barrier_inner,
        barrier_tensor.op.axis[3],
        barrier_tensor.op.axis[4],
        barrier_tensor.op.axis[5])

    s[c_ub].compute_at(s[barrier_tensor], barrier_outer)
    s[bias_ub].compute_at(s[barrier_tensor], barrier_outer)

    for tensor in elewise_before_barrier_tensors:
        s[tensor].compute_at(s[barrier_tensor], barrier_outer)

    vn_outer, vn_inner = \
        s[update_hc_vn].split(update_hc_vn.op.axis[0 + 1],
                              factor=factor_l1_n // block_size)

    second_split_factor = \
        (hidden_size // (factor_l1_n // block_size)) // core_num

    vn_o_outer, vn_o_inner = \
        s[update_hc_vn].split(vn_outer,
                              factor=second_split_factor)

    s[barrier_tensor].compute_at(s[update_hc_vn], vn_o_inner)

    for tensor in elewise_tensors:
        if tensor not in elewise_before_barrier_tensors:
            s[tensor].compute_at(s[update_hc_vn], vn_o_inner)

    s[update_c_gm].compute_at(s[update_hc_vn], vn_o_inner)
    s[update_h_gm].compute_at(s[update_hc_vn], vn_o_inner)

    second_split_factor = hidden_size // core_num

    res_h_outer, res_h_inner = \
        s[update_h_gm_2].split(update_h_gm_2.op.axis[1],
                               factor=hidden_size)

    s[update_hc_vn].compute_at(s[update_h_gm_2], update_h_gm_2.op.axis[0])

    s[update_c_gm_vn].compute_at(s[update_h_gm_2], res_h_outer)
    s[update_h_gm_vn].compute_at(s[update_h_gm_2], res_h_outer)
    s[update_c_ub].compute_at(s[update_h_gm_2], res_h_outer)
    s[update_c_gm_2].compute_at(s[update_h_gm_2], res_h_outer)
    s[update_h_ub].compute_at(s[update_h_gm_2], res_h_outer)

    s[update_h_gm_vn].bind_buffer(
        update_h_gm_vn.op.axis[0], 0,
        scan_h.op.scan_axis + res_h_outer)
    s[update_c_gm_vn].bind_buffer(
        update_c_gm_vn.op.axis[0], 0,
        scan_h.op.scan_axis + res_h_outer)

    # bind
    s[update_hc_vn].bind(vn_o_outer, tvm.thread_axis("blockIdx.x"))

    # multi core sync
    s[update_hc_vn].pragma(update_hc_vn.op.axis[0],
                          pragma_type="multicore_sync_wait_after",
                          pragma_value=sync0[0])
    s[update_hc_vn].pragma(update_hc_vn.op.axis[0],
                          pragma_type="multicore_sync_set_after",
                          pragma_value=sync0[0])

    # modify for extend
    s[input_x].bind_buffer(0, 0, scan_h.op.scan_axis)

    s[update_h_gm].buffer_tile((scan_h.op.scan_axis*1, 1),
                               (None, None), (None, None),
                               (None, None), (None, None))

    s[update_c_gm].buffer_tile((scan_h.op.scan_axis*1, 1),
                               (None, None), (None, None),
                               (None, None), (None, None))

    s[update_h_gm_2].buffer_tile((0, 1), (None, None), (None, None),
                                 (None, None), (None, None))
    s[update_c_gm_2].buffer_tile((0, 1), (None, None), (None, None),
                                 (None, None), (None, None))

    # buffer reuse
    s[update_h_gm].reused_by(update_h_gm_vn)
    s[update_c_gm].reused_by(update_c_gm_vn)

    # emit_insn
    s[a_l1].emit_insn(a_l1.op.axis[0], 'dma_copy')
    s[b_l1].emit_insn(b_l1.op.axis[0], 'dma_copy')
    s[a_l0a].emit_insn(a_l0a.op.axis[0], 'dma_copy')
    s[b_l0b].emit_insn(b_l0b.op.axis[0], 'dma_copy')

    s[weight_ub].emit_insn(weight_ub.op.axis[0], 'dma_copy')
    s[weight_fp16].emit_insn(weight_fp16.op.axis[0], 'vector_conv')

    s[a_ub].emit_insn(a_ub.op.axis[0], 'dma_copy')
    s[a_ub_fp16].emit_insn(a_ub_fp16.op.axis[0], 'vector_conv')

    mad_dict = {"mad_pattern": 0, "k_outer": [l1_k_outer, l0_k_outer]}
    s[c_l0c].emit_insn(l0_n_inner, 'mad', mad_dict)
    s[c_ub].emit_insn(ub_n_inner, 'dma_copy')

    s[s_init_h].emit_insn(s_init_h.op.axis[0], 'dma_copy')
    s[s_init_c].emit_insn(s_init_c.op.axis[0], 'dma_copy')
    s[bias_bc_ub].emit_insn(bias_bc_ub.op.axis[0], 'unified_broadcast')

    s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'broadcast')
    s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'broadcast')

    s[barrier_tensor].emit_insn(barrier_tensor.op.axis[1], 'vector_add')

    for tensor in elewise_tensors:
        if tensor != barrier_tensor:
            insn = get_emit_insn_map(tensor)
            s[tensor].emit_insn(tensor.op.axis[0], insn)

    s[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')

    s[update_c_gm].emit_insn(s[update_c_gm].op.axis[1], 'dma_copy')
    s[update_h_gm].emit_insn(s[update_h_gm].op.axis[1], 'dma_copy')

    s[update_c_ub].emit_insn(update_c_ub.op.axis[1], 'dma_copy')
    s[update_h_ub].emit_insn(update_h_ub.op.axis[1], 'dma_copy')

    s[update_hc_vn].emit_insn(vn_inner, 'phony_insn')
    s[update_c_gm_vn].emit_insn(s[update_c_gm_vn].op.axis[0], 'phony_insn')
    s[update_h_gm_vn].emit_insn(s[update_h_gm_vn].op.axis[0], 'phony_insn')
    s[update_h_gm_2].emit_insn(res_h_inner, 'phony_insn')
    s[update_c_gm_2].emit_insn(s[update_c_gm_2].op.axis[0], 'phony_insn')
    s[update_h_gm_2_dummy].emit_insn(
        update_h_gm_2_dummy.op.axis[0], 'phony_insn')

    def _write_workspace_info(shape_list, dtype_list, sync_num, kernel_name):
        """
        modify json after build
        """
        def _write_code(wkspace_dict, fname):
            fname = os.path.realpath(fname)
            if fname.startswith(os.getcwd()):
                if os.path.exists(fname):
                    with open(fname, "r") as f:
                        load_dict = json.load(f)

                    load_dict.update(wkspace_dict)
                    with open(fname, "w") as f:
                        json.dump(load_dict, f,
                                  sort_keys=True, indent=4,
                                  separators=(',', ':'))

        def _get_data_width(ele):
            """
            get data width
            """
            m_sea = re.search(r'\d+', ele)
            if m_sea:
                return int(m_sea.group(0)) // 8
            return 0

        if not os.path.exists("kernel_meta"):
            os.mkdir("kernel_meta")
            os.chmod("kernel_meta", stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)

        num = len(shape_list)
        wkspace_dict = {}
        if num:
            total_size = [functools_reduce(lambda x, y: x * y, list_i) for
                          list_i in shape_list]

            addr_type_list = []
            for i, element in enumerate(dtype_list):
                total_size[i] = total_size[i] * _get_data_width(element)
                addr_type_list.append(0)

            if not os.path.exists("kernel_meta"):
                os.mkdir("kernel_meta")
                os.chmod("kernel_meta",
                         stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)

            wkspace_dict["workspace"] = {"num": num,
                                         "size": total_size,
                                         "type": addr_type_list}

        if sync_num:
            parameters_list = \
                (len(new_build_list) - 2 - sync_num) * [0, ] + sync_num * [1, ]
            wkspace_dict["parameters"] = parameters_list

        if wkspace_dict:
            _write_code(wkspace_dict, "kernel_meta/" + kernel_name + ".json")

    with build_config:
        tvm.build(s, new_build_list, "cce", name=kernel_name)
        _write_workspace_info(
            [shape_i_t, shape_sync],
            [input_dtype, sync_dtype],
            1, kernel_name)
