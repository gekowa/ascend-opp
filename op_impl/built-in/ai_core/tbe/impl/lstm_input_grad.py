"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

lstm_grad
"""

# pylint: disable=locally-disabled,import-error,unused-import,ungrouped-imports
from te.lang.cce import vmul
from te.lang.cce import vadd
from te.lang.cce import matmul
from te.lang.cce import broadcast
from te.lang.cce import cast_to
from te.lang.cce import concat
from te.lang.cce import vsub
from te.tvm import api as tvm
from te.platform import insn_cmd
from te.platform.fusion_manager import fusion_manager
from te.utils.cce import auto_schedule
from te.platform.cce_build import build_config
from te.platform import scope_ubuf
from te import tvm
from te import platform as cceconf


NONETYPE = type(None)
C0 = 16


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def matmul_compute(tensor_d_gate, tensor_w, tensor_list, emit_list, scope_list, cast_type):
    """
    matmul_compute for compute dx dh
    """
    if cast_type:
        tensor_d_gate_fp16 = cast_to(tensor_d_gate, "float16")
        tensor_list["tensor_d_gate_fp16"] = tensor_d_gate_fp16
        scope_list["tensor_d_gate_fp16"] = scope_ubuf
        emit_list["tensor_d_gate_fp16"] = insn_cmd.CAST

        tensor_w_fp16 = cast_to(tensor_w, "float16")
        tensor_list["tensor_w_fp16"] = tensor_w_fp16
        scope_list["tensor_w_fp16"] = scope_ubuf
        emit_list["tensor_w_fp16"] = insn_cmd.CAST

        dst_type = "float32"
    else:
        tensor_d_gate_fp16 = tensor_d_gate
        tensor_w_fp16 = tensor_w
        dst_type = "float16"
    matmul_res_gm = matmul(tensor_a=tensor_d_gate_fp16, tensor_b=tensor_w_fp16, trans_a=True,
                                       trans_b=False,
                                       format_a="FRACTAL_NZ", format_b="FRACTAL_NZ",
                                       alpha_num=1.0, beta_num=0.0,
                                       dst_dtype=dst_type, tensor_bias=None)

    matmul_res = matmul_res_gm.op.input_tensors[0]

    tensor_list["matmul_res"] = matmul_res
    scope_list["matmul_res"] = scope_ubuf

    # matmul_result_shape [n, input_size + output_size] ==> [in+out // 16, n//16, 16, 16]
    # dx_shape [n, input_size]
    # dh_shape [n, output_size]
    n = tensor_d_gate.shape[2].value
    output_size = tensor_d_gate.shape[1].value // 4
    input_size = tensor_w.shape[2].value - output_size

    dx_shape = [1, input_size, n, 16, 16]
    tensor_dx = tvm.compute(dx_shape, lambda t, i, j, k, l: matmul_res[0, i, j, k, l],
                            name="tensor_dx", tag="tensor_dx")
    tensor_list["tensor_dx"] = tensor_dx
    emit_list["tensor_dx"] = insn_cmd.DMA_COPY

    dh_shape = [1, output_size, n, 16, 16]

    tensor_dh = tvm.compute(dh_shape, lambda t, i, j, k, l: matmul_res[0, i + input_size, j, k, l],
                            name="tensor_dh", tag="tensor_dh")
    tensor_list["tensor_dh"] = tensor_dh
    emit_list["tensor_dh"] = insn_cmd.DMA_COPY

    tensor_add_fake = vadd(tensor_dh, tensor_list["tensor_d_ct1_gm_ele"])
    tensor_list["tensor_add_fake"] = tensor_add_fake

    # fake_concat_shape = [input_size + output_size]
    fake_tensors = [tensor_dx, tensor_add_fake]
    tensor_concat_fake = concat(fake_tensors, 1)
    tensor_list["tensor_concat_fake"] = tensor_concat_fake

    return tensor_dx, tensor_dh, tensor_list, emit_list, scope_list


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def elewise_compute_for_gate(tensor_c, tensor_d_ht, tensor_d_ht_1, tensor_d_ht_1_ct,
                             tensor_d_ct, tensor_it, tensor_it_dup, tensor_jt,
                             tensor_ft, tensor_ot, tensor_tanh_ct, tensor_d_ht_ct, tensor_ot_ct,
                             tensor_tanh_ct_fake, tensor_d_ct_ct, tensor_ft_ct,
                             tensor_jt_dc_list, tensor_it_dc_list, tensor_ft_dc_list, tensor_jt_it_fake,
                             tensor_list, emit_list, scope_list):
    """
    elewise_compute for compute dgate result
    """

    if tensor_tanh_ct is None:
        tensor_tanh_ct = sigmod_compute(tensor_c, tensor_list, emit_list)

    shape_list = [1, tensor_ot.shape[1].value, tensor_ot.shape[2].value,
                  tensor_ot.shape[3].value, tensor_ot.shape[4].value]

    # compute for dct
    data_one_ct = broadcast(tvm.const(1, 'float16'), shape_list, 'float16')
    tensor_list["data_one_ct_ele"] = data_one_ct
    emit_list["data_one_ct_ele"] = insn_cmd.DUP
    scope_list["data_one_ct_ele"] = scope_ubuf

    tensor_d_ht_ct_add = vadd(tensor_d_ht_ct, tensor_d_ht_1_ct)
    tensor_list["tensor_d_ht_add_ct_ele"] = tensor_d_ht_ct_add
    emit_list["tensor_d_ht_add_ct_ele"] = insn_cmd.ADD
    scope_list["tensor_d_ht_add_ct_ele"] = scope_ubuf

    tensor_ot_mul_ct = vmul(tensor_d_ht_ct_add, tensor_ot_ct)
    tensor_list["tensor_ot_mul_ct_ele"] = tensor_ot_mul_ct
    emit_list["tensor_ot_mul_ct_ele"] = insn_cmd.MUL
    scope_list["tensor_ot_mul_ct_ele"] = scope_ubuf

    tensor_tanct_pow_ct = vmul(tensor_tanh_ct_fake, tensor_tanh_ct_fake)
    tensor_list["tensor_tanct_pow_ct_ele"] = tensor_tanct_pow_ct
    emit_list["tensor_tanct_pow_ct_ele"] = insn_cmd.MUL
    scope_list["tensor_tanct_pow_ct_ele"] = scope_ubuf

    tensor_one_sub_tanct_ct = vsub(data_one_ct, tensor_tanct_pow_ct)
    tensor_list["tensor_one_sub_tanct_ct_ele"] = tensor_one_sub_tanct_ct
    emit_list["tensor_one_sub_tanct_ct_ele"] = insn_cmd.SUB
    scope_list["tensor_one_sub_tanct_ct_ele"] = scope_ubuf

    tensor_ot_ct_mul_ct = vmul(tensor_ot_mul_ct, tensor_one_sub_tanct_ct)
    tensor_list["tensor_ot_ct_mul_ct_ele"] = tensor_ot_ct_mul_ct
    emit_list["tensor_ot_ct_mul_ct_ele"] = insn_cmd.MUL
    scope_list["tensor_ot_ct_mul_ct_ele"] = scope_ubuf

    tensor_d_c_ct = vadd(tensor_ot_ct_mul_ct, tensor_d_ct_ct)
    tensor_list["tensor_d_c_ct_ele"] = tensor_d_c_ct
    emit_list["tensor_d_c_ct_ele"] = insn_cmd.ADD
    scope_list["tensor_d_c_ct_ele"] = scope_ubuf

    # compute for d_ot
    data_one = broadcast(tvm.const(1, 'float16'), shape_list, 'float16')
    tensor_list["data_one_ele"] = data_one
    emit_list["data_one_ele"] = insn_cmd.DUP
    scope_list["data_one_ele"] = scope_ubuf

    tensor_ot_sub = vsub(data_one, tensor_ot)
    tensor_list["tensor_ot_sub_ele"] = tensor_ot_sub
    emit_list["tensor_ot_sub_ele"] = insn_cmd.SUB
    scope_list["tensor_ot_sub_ele"] = scope_ubuf

    # tensor_ot_mul  tensor_tanh_ct tensor_jt
    tensor_d_ht_add = vadd(tensor_d_ht, tensor_d_ht_1)
    tensor_list["tensor_d_ht_add_ele"] = tensor_d_ht_add
    emit_list["tensor_d_ht_add_ele"] = insn_cmd.ADD
    scope_list["tensor_d_ht_add_ele"] = scope_ubuf

    tensor_ot_mul = vmul(tensor_d_ht_add, tensor_ot)
    tensor_list["tensor_ot_mul_ele"] = tensor_ot_mul
    emit_list["tensor_ot_mul_ele"] = insn_cmd.MUL
    scope_list["tensor_ot_mul_ele"] = scope_ubuf

    tensor_ot_hc_mul = vmul(tensor_ot_mul, tensor_tanh_ct)
    tensor_list["tensor_ot_hc_mul_ele"] = tensor_ot_hc_mul
    emit_list["tensor_ot_hc_mul_ele"] = insn_cmd.MUL
    scope_list["tensor_ot_hc_mul_ele"] = scope_ubuf

    tensor_d_ot = vmul(tensor_ot_sub, tensor_ot_hc_mul)
    tensor_list["tensor_d_ot_ele"] = tensor_d_ot
    emit_list["tensor_d_ot_ele"] = insn_cmd.MUL
    scope_list["tensor_d_ot_ele"] = scope_ubuf

    def get_d_c(data_one_fake, tensor_tanh_ct_fake, tensor_d_ht_fake,
                tensor_d_ht_1_fake, tensor_ot_fake, tensor_d_ct_fake, name):
        """
        return dc for xt
        """
        key_0 = "tensor_d_ht_add_" + name + "_ele"
        tensor_d_ht_add = vadd(tensor_d_ht_fake, tensor_d_ht_1_fake)
        tensor_list[key_0] = tensor_d_ht_add
        emit_list[key_0] = insn_cmd.ADD
        scope_list[key_0] = scope_ubuf

        key_1 = "tensor_ot_mul_" + name + "_ele"
        tensor_ot_mul = vmul(tensor_d_ht_add, tensor_ot_fake)
        tensor_list[key_1] = tensor_ot_mul
        emit_list[key_1] = insn_cmd.MUL
        scope_list[key_1] = scope_ubuf

        key_2 = "tensor_tanct_pow_" + name + "_ele"
        tensor_tanct_pow = vmul(tensor_tanh_ct_fake, tensor_tanh_ct_fake)
        tensor_list[key_2] = tensor_tanct_pow
        emit_list[key_2] = insn_cmd.MUL
        scope_list[key_2] = scope_ubuf

        key_3 = "tensor_one_sub_tanct_" + name + "_ele"
        tensor_one_sub_tanct = vsub(data_one_fake, tensor_tanct_pow)
        tensor_list[key_3] = tensor_one_sub_tanct
        emit_list[key_3] = insn_cmd.SUB
        scope_list[key_3] = scope_ubuf

        key_4 = "tensor_ot_ct_mul_" + name + "_ele"
        tensor_ot_ct_mul = vmul(tensor_ot_mul, tensor_one_sub_tanct)
        tensor_list[key_4] = tensor_ot_ct_mul
        emit_list[key_4] = insn_cmd.MUL
        scope_list[key_4] = scope_ubuf

        key_5 = "tensor_d_c_" + name + "_ele"
        tensor_d_c = vadd(tensor_ot_ct_mul, tensor_d_ct_fake)
        tensor_list[key_5] = tensor_d_c
        emit_list[key_5] = insn_cmd.ADD
        scope_list[key_5] = scope_ubuf

        return tensor_d_c

    # compute for d_jt
    tensor_jt_pow = vmul(tensor_jt, tensor_jt)
    tensor_list["tensor_jt_pow_ele"] = tensor_jt_pow
    emit_list["tensor_jt_pow_ele"] = insn_cmd.MUL
    scope_list["tensor_jt_pow_ele"] = scope_ubuf

    data_one_jt_fake = broadcast(tvm.const(1, 'float16'), shape_list, 'float16')
    tensor_list["data_one_jt_fake_ele"] = data_one_jt_fake
    emit_list["data_one_jt_fake_ele"] = "vector_dup"
    scope_list["data_one_jt_fake_ele"] = scope_ubuf

    tensor_one_sub_jt = vsub(data_one_jt_fake, tensor_jt_pow)
    tensor_list["tensor_one_sub_jt_ele"] = tensor_one_sub_jt
    emit_list["tensor_one_sub_jt_ele"] = insn_cmd.SUB
    scope_list["tensor_one_sub_jt_ele"] = scope_ubuf

    tensor_jt_dc = get_d_c(data_one_jt_fake, tensor_jt_dc_list[0],
                           tensor_jt_dc_list[1],
                           tensor_jt_dc_list[2],
                           tensor_jt_dc_list[3],
                           tensor_jt_dc_list[4], "jt_fake")

    tensor_dc_it = vmul(tensor_jt_dc, tensor_it)
    tensor_list["tensor_dc_it_ele"] = tensor_dc_it
    emit_list["tensor_dc_it_ele"] = insn_cmd.MUL
    scope_list["tensor_dc_it_ele"] = scope_ubuf

    tensor_d_jt = vmul(tensor_one_sub_jt, tensor_dc_it)
    tensor_list["tensor_d_jt_ele"] = tensor_d_jt
    emit_list["tensor_d_jt_ele"] = insn_cmd.MUL
    scope_list["tensor_d_jt_ele"] = scope_ubuf

    # compute for d_it
    data_one_it_fake = broadcast(tvm.const(1, 'float16'), shape_list, 'float16')
    tensor_list["data_one_it_fake_ele"] = data_one_it_fake
    emit_list["data_one_it_fake_ele"] = "vector_dup"
    scope_list["data_one_it_fake_ele"] = scope_ubuf

    tensor_it_sub = vsub(data_one_it_fake, tensor_it_dup)
    tensor_list["tensor_it_sub_ele"] = tensor_it_sub
    emit_list["tensor_it_sub_ele"] = insn_cmd.SUB
    scope_list["tensor_it_sub_ele"] = scope_ubuf

    tensor_it_dc = get_d_c(data_one_it_fake, tensor_it_dc_list[0],
                           tensor_it_dc_list[1],
                           tensor_it_dc_list[2],
                           tensor_it_dc_list[3],
                           tensor_it_dc_list[4], "it_fake")

    tensor_dc_jt = vmul(tensor_it_dc, tensor_jt_it_fake)
    tensor_list["tensor_dc_jt_ele"] = tensor_dc_jt
    emit_list["tensor_dc_jt_ele"] = insn_cmd.MUL
    scope_list["tensor_dc_jt_ele"] = scope_ubuf

    tensor_dc_jt_it = vmul(tensor_it_dup, tensor_dc_jt)
    tensor_list["tensor_dc_jt_it_ele"] = tensor_dc_jt_it
    emit_list["tensor_dc_jt_it_ele"] = insn_cmd.MUL
    scope_list["tensor_dc_jt_it_ele"] = scope_ubuf

    tensor_d_it = vmul(tensor_dc_jt_it, tensor_it_sub)
    tensor_list["tensor_d_it_ele"] = tensor_d_it
    emit_list["tensor_d_it_ele"] = insn_cmd.MUL
    scope_list["tensor_d_it_ele"] = scope_ubuf

    # compute for d_ft
    data_one_ft_fake = broadcast(tvm.const(1, 'float16'), shape_list, 'float16')
    tensor_list["data_one_ft_fake_ele"] = data_one_ft_fake
    emit_list["data_one_ft_fake_ele"] = "vector_dup"
    scope_list["data_one_ft_fake_ele"] = scope_ubuf

    tensor_ft_sub = vsub(data_one_ft_fake, tensor_ft)
    tensor_list["tensor_ft_sub_ele"] = tensor_ft_sub
    emit_list["tensor_ft_sub_ele"] = insn_cmd.SUB
    scope_list["tensor_ft_sub_ele"] = scope_ubuf

    tensor_ft_dc = get_d_c(data_one_ft_fake, tensor_ft_dc_list[0],
                           tensor_ft_dc_list[1],
                           tensor_ft_dc_list[2],
                           tensor_ft_dc_list[3],
                           tensor_ft_dc_list[4], "ft_fake")

    tensor_dc_c = vmul(tensor_ft_dc, tensor_c)
    tensor_list["tensor_dc_c_ele"] = tensor_dc_c
    emit_list["tensor_dc_c_ele"] = insn_cmd.MUL
    scope_list["tensor_dc_c_ele"] = scope_ubuf

    tensor_dc_c_ft = vmul(tensor_dc_c, tensor_ft)
    tensor_list["tensor_dc_c_ft_ele"] = tensor_dc_c_ft
    emit_list["tensor_dc_c_ft_ele"] = insn_cmd.MUL
    scope_list["tensor_dc_c_ft_ele"] = scope_ubuf

    tensor_d_ft = vmul(tensor_dc_c_ft, tensor_ft_sub)
    tensor_list["tensor_d_ft_ele"] = tensor_d_ft
    emit_list["tensor_d_ft_ele"] = insn_cmd.MUL
    scope_list["tensor_d_ft_ele"] = scope_ubuf

    # compute for dct-1
    tensor_d_ct1 = vmul(tensor_d_c_ct, tensor_ft_ct)
    tensor_list["tensor_d_ct1_ele"] = tensor_d_ct1
    emit_list["tensor_d_ct1_ele"] = insn_cmd.MUL
    scope_list["tensor_d_ct1_ele"] = scope_ubuf

    # dma_copy for dct-1
    tensor_d_ct1_gm = tvm.compute(shape_list, lambda t, i, j, k, l: tensor_d_ct1[t, i, j, k, l],
                                  name="tensor_d_ct1_gm", tag="tensor_d_ct1_gm")

    tensor_list["tensor_d_ct1_gm_ele"] = tensor_d_ct1_gm
    emit_list["tensor_d_ct1_gm_ele"] = insn_cmd.DMA_COPY

    dgate_tensors = [tensor_d_it, tensor_d_jt, tensor_d_ft, tensor_d_ot]
    tensor_d_gate = concat(dgate_tensors, 1)

    tensor_list["tensor_d_gate"] = tensor_d_gate
    emit_list["tensor_d_gate"] = insn_cmd.DMA_COPY
    result_tensor = [tensor_d_gate, tensor_d_ct1_gm]

    return result_tensor, scope_list, tensor_list, emit_list


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def copy_input_to_ub_compute(is_last_time, tensor_c, tensor_d_ht, tensor_d_ht_1,
                             tensor_d_ct, tensor_it, tensor_jt,
                             tensor_ft, tensor_ot, tensor_tanh_ct,
                             tensor_list, emit_list, scope_list):
    """
    return the compute object for copy the data to ub
    """
    tensor_ele_shape = [1, tensor_c.shape[1].value, tensor_c.shape[2].value,
                        tensor_c.shape[3].value, tensor_c.shape[4].value]
    tensor_c_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_c[0, i, j, k, l],
                              name="tensor_c_ub", tag="tensor_c_ub")
    tensor_list["tensor_c_ub_ele"] = tensor_c_ub
    emit_list["tensor_c_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_c_ub_ele"] = scope_ubuf

    tensor_d_ht_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ht[0, i, j, k, l],
                                 name="tensor_d_ht_ub", tag="tensor_d_ht_ub")
    tensor_list["tensor_d_ht_ub_ele"] = tensor_d_ht_ub
    emit_list["tensor_d_ht_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ht_ub_ele"] = scope_ubuf

    tensor_d_ht_1_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ht_1[0, i, j, k, l],
                                   name="tensor_d_ht_1_ub", tag="tensor_d_ht_1_ub")
    tensor_list["tensor_d_ht_1_ub_ele"] = tensor_d_ht_1_ub
    emit_list["tensor_d_ht_1_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ht_1_ub_ele"] = scope_ubuf

    tensor_it_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_it[0, i, j, k, l],
                               name="tensor_it_ub", tag="tensor_it_ub")
    tensor_list["tensor_it_ub_ele"] = tensor_it_ub
    emit_list["tensor_it_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_it_ub_ele"] = scope_ubuf

    tensor_it_ub_dup = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_it[0, i, j, k, l],
                                   name="tensor_it_ub_dup", tag="tensor_it_ub_dup")
    tensor_list["tensor_it_ub_dup_ele"] = tensor_it_ub_dup
    emit_list["tensor_it_ub_dup_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_it_ub_dup_ele"] = scope_ubuf

    tensor_jt_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_jt[0, i, j, k, l],
                               name="tensor_jt_ub", tag="tensor_jt_ub")
    tensor_list["tensor_jt_ub_ele"] = tensor_jt_ub
    emit_list["tensor_jt_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_jt_ub_ele"] = scope_ubuf

    # tensor_jt_it_fake
    tensor_jt_it_fake = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_jt[0, i, j, k, l],
                                    name="tensor_jt_it_ub", tag="tensor_jt_it_ub")
    tensor_list["tensor_jt_it_ub_ele"] = tensor_jt_it_fake
    emit_list["tensor_jt_it_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_jt_it_ub_ele"] = scope_ubuf

    tensor_ft_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_ft[0, i, j, k, l],
                               name="tensor_ft_ub", tag="tensor_ft_ub")
    tensor_list["tensor_ft_ub_ele"] = tensor_ft_ub
    emit_list["tensor_ft_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_ft_ub_ele"] = scope_ubuf

    tensor_ot_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_ot[0, i, j, k, l],
                               name="tensor_ot_ub", tag="tensor_ot_ub")
    tensor_list["tensor_ot_ub_ele"] = tensor_ot_ub
    emit_list["tensor_ot_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_ot_ub_ele"] = scope_ubuf

    tensor_tanh_ct_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_tanh_ct[0, i, j, k, l],
                                    name="tensor_tanh_ct_ub", tag="tensor_tanh_ct_ub")
    tensor_list["tensor_tanh_ct_ub_ele"] = tensor_tanh_ct_ub
    emit_list["tensor_tanh_ct_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_tanh_ct_ub_ele"] = scope_ubuf

    tensor_d_ht_ub_ct = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ht[0, i, j, k, l],
                                    name="tensor_d_ht_ub_ct", tag="tensor_d_ht_ub_ct")
    tensor_list["tensor_d_ht_ub_ct_ele"] = tensor_d_ht_ub_ct
    emit_list["tensor_d_ht_ub_ct_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ht_ub_ct_ele"] = scope_ubuf

    tensor_d_ht_1_ub_ct = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ht_1[0, i, j, k, l],
                                      name="tensor_d_ht_1_ub_ct", tag="tensor_d_ht_1_ub_ct")
    tensor_list["tensor_d_ht_1_ub_ct_ele"] = tensor_d_ht_1_ub_ct
    emit_list["tensor_d_ht_1_ub_ct_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ht_1_ub_ct_ele"] = scope_ubuf

    tensor_ot_ub_ct = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_ot[0, i, j, k, l],
                                  name="tensor_ot_ub_ct", tag="tensor_ot_ub_ct")
    tensor_list["tensor_ot_ub_ct_ele"] = tensor_ot_ub_ct
    emit_list["tensor_ot_ub_ct_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_ot_ub_ct_ele"] = scope_ubuf

    tensor_tanh_ct_ub_ct = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_tanh_ct[0, i, j, k, l],
                                       name="tensor_tanh_ct_ub_ct", tag="tensor_tanh_ct_ub_ct")
    tensor_list["tensor_tanh_ct_ub_ct_ele"] = tensor_tanh_ct_ub_ct
    emit_list["tensor_tanh_ct_ub_ct_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_tanh_ct_ub_ct_ele"] = scope_ubuf

    tensor_d_ct_ub_ct = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ct[0, i, j, k, l],
                                    name="tensor_d_ct_ub_ct", tag="tensor_d_ct_ub_ct")
    tensor_list["tensor_d_ct_ub_ct_ele"] = tensor_d_ct_ub_ct
    emit_list["tensor_d_ct_ub_ct_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ct_ub_ct_ele"] = scope_ubuf

    tensor_ft_ub_ct = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_ft[0, i, j, k, l],
                                  name="tensor_ft_ub_ct", tag="tensor_ft_ub_ct")
    tensor_list["tensor_ft_ub_ct_ele"] = tensor_ft_ub_ct
    emit_list["tensor_ft_ub_ct_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_ft_ub_ct_ele"] = scope_ubuf

    # tensor_tanh_ct, tensor_d_ht, tensor_ot

    tensor_tanh_ct_jt_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_tanh_ct[0, i, j, k, l],
                                          name="tensor_tanh_ct_jt_dc_ub", tag="tensor_tanh_ct_jt_dc_ub")
    tensor_list["tensor_tanh_ct_jt_dc_ub_ele"] = tensor_tanh_ct_jt_dc_ub
    emit_list["tensor_tanh_ct_jt_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_tanh_ct_jt_dc_ub_ele"] = scope_ubuf

    tensor_tanh_ct_it_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_tanh_ct[0, i, j, k, l],
                                          name="tensor_tanh_ct_it_dc_ub", tag="tensor_tanh_ct_it_dc_ub")
    tensor_list["tensor_tanh_ct_it_dc_ub_ele"] = tensor_tanh_ct_it_dc_ub
    emit_list["tensor_tanh_ct_it_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_tanh_ct_it_dc_ub_ele"] = scope_ubuf

    tensor_tanh_ct_ft_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_tanh_ct[0, i, j, k, l],
                                          name="tensor_tanh_ct_ft_dc_ub", tag="tensor_tanh_ct_ft_dc_ub")
    tensor_list["tensor_tanh_ct_ft_dc_ub_ele"] = tensor_tanh_ct_ft_dc_ub
    emit_list["tensor_tanh_ct_ft_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_tanh_ct_ft_dc_ub_ele"] = scope_ubuf

    tensor_d_ht_jt_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ht[0, i, j, k, l],
                                       name="tensor_d_ht_jt_dc_ub", tag="tensor_d_ht_jt_dc_ub")
    tensor_list["tensor_d_ht_jt_dc_ub_ele"] = tensor_d_ht_jt_dc_ub
    emit_list["tensor_d_ht_jt_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ht_jt_dc_ub_ele"] = scope_ubuf

    tensor_d_ht_it_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ht[0, i, j, k, l],
                                       name="tensor_d_ht_it_dc_ub", tag="tensor_d_ht_it_dc_ub")
    tensor_list["tensor_d_ht_it_dc_ub_ele"] = tensor_d_ht_it_dc_ub
    emit_list["tensor_d_ht_it_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ht_it_dc_ub_ele"] = scope_ubuf

    tensor_d_ht_ft_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ht[0, i, j, k, l],
                                       name="tensor_d_ht_ft_dc_ub", tag="tensor_d_ht_ft_dc_ub")
    tensor_list["tensor_d_ht_ft_dc_ub_ele"] = tensor_d_ht_ft_dc_ub
    emit_list["tensor_d_ht_ft_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ht_ft_dc_ub_ele"] = scope_ubuf

    tensor_d_ht_1_jt_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ht_1[0, i, j, k, l],
                                         name="tensor_d_ht_1_jt_dc_ub", tag="tensor_d_ht_1_jt_dc_ub")
    tensor_list["tensor_d_ht_1_jt_dc_ub_ele"] = tensor_d_ht_1_jt_dc_ub
    emit_list["tensor_d_ht_1_jt_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ht_1_jt_dc_ub_ele"] = scope_ubuf

    tensor_d_ht_1_it_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ht_1[0, i, j, k, l],
                                         name="tensor_d_ht_1_it_dc_ub", tag="tensor_d_ht_1_it_dc_ub")
    tensor_list["tensor_d_ht_1_it_dc_ub_ele"] = tensor_d_ht_1_it_dc_ub
    emit_list["tensor_d_ht_1_it_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ht_1_it_dc_ub_ele"] = scope_ubuf

    tensor_d_ht_1_ft_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ht_1[0, i, j, k, l],
                                         name="tensor_d_ht_1_ft_dc_ub", tag="tensor_d_ht_1_ft_dc_ub")
    tensor_list["tensor_d_ht_1_ft_dc_ub_ele"] = tensor_d_ht_1_ft_dc_ub
    emit_list["tensor_d_ht_1_ft_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ht_1_ft_dc_ub_ele"] = scope_ubuf

    tensor_ot_jt_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_ot[0, i, j, k, l],
                                     name="tensor_ot_jt_dc_ub", tag="tensor_ot_jt_dc_ub")
    tensor_list["tensor_ot_jt_dc_ub_ele"] = tensor_ot_jt_dc_ub
    emit_list["tensor_ot_jt_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_ot_jt_dc_ub_ele"] = scope_ubuf

    tensor_ot_it_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_ot[0, i, j, k, l],
                                     name="tensor_ot_it_dc_ub", tag="tensor_ot_it_dc_ub")
    tensor_list["tensor_ot_it_dc_ub_ele"] = tensor_ot_it_dc_ub
    emit_list["tensor_ot_it_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_ot_it_dc_ub_ele"] = scope_ubuf

    tensor_ot_ft_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_ot[0, i, j, k, l],
                                     name="tensor_ot_ft_dc_ub", tag="tensor_ot_ft_dc_ub")
    tensor_list["tensor_ot_ft_dc_ub_ele"] = tensor_ot_ft_dc_ub
    emit_list["tensor_ot_ft_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_ot_ft_dc_ub_ele"] = scope_ubuf

    tensor_d_ct_jt_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ct[0, i, j, k, l],
                                       name="tensor_d_ct_jt_dc_ub", tag="tensor_d_ct_jt_dc_ub")
    tensor_list["tensor_d_ct_jt_dc_ub_ele"] = tensor_d_ct_jt_dc_ub
    emit_list["tensor_d_ct_jt_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ct_jt_dc_ub_ele"] = scope_ubuf

    tensor_d_ct_it_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ct[0, i, j, k, l],
                                       name="tensor_d_ct_it_dc_ub", tag="tensor_d_ct_it_dc_ub")
    tensor_list["tensor_d_ct_it_dc_ub_ele"] = tensor_d_ct_it_dc_ub
    emit_list["tensor_d_ct_it_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ct_it_dc_ub_ele"] = scope_ubuf

    tensor_d_ct_ft_dc_ub = tvm.compute(tensor_ele_shape, lambda t, i, j, k, l: tensor_d_ct[0, i, j, k, l],
                                       name="tensor_d_ct_ft_dc_ub", tag="tensor_d_ct_ft_dc_ub")
    tensor_list["tensor_d_ct_ft_dc_ub_ele"] = tensor_d_ct_ft_dc_ub
    emit_list["tensor_d_ct_ft_dc_ub_ele"] = insn_cmd.DMA_COPY
    scope_list["tensor_d_ct_ft_dc_ub_ele"] = scope_ubuf

    tensor_jt_dc_list = [tensor_tanh_ct_jt_dc_ub, tensor_d_ht_jt_dc_ub, tensor_d_ht_1_jt_dc_ub,
                         tensor_ot_jt_dc_ub, tensor_d_ct_jt_dc_ub]
    tensor_it_dc_list = [tensor_tanh_ct_it_dc_ub, tensor_d_ht_it_dc_ub, tensor_d_ht_1_it_dc_ub,
                         tensor_ot_it_dc_ub, tensor_d_ct_it_dc_ub]
    tensor_ft_dc_list = [tensor_tanh_ct_ft_dc_ub, tensor_d_ht_ft_dc_ub, tensor_d_ht_1_ft_dc_ub,
                         tensor_ot_ft_dc_ub, tensor_d_ct_ft_dc_ub]

    return tensor_list, emit_list, scope_list, tensor_jt_dc_list, tensor_it_dc_list, tensor_ft_dc_list


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def basic_lstm_cell_compute(tensor_w, tensor_c, tensor_d_ht, tensor_d_ht_1,
                            tensor_d_ct, tensor_it, tensor_jt,
                            tensor_ft, tensor_ot, tensor_tanh_ct, is_last_time):
    """
    the total compute process for lstm
    """
    tensor_input_list = [tensor_w, tensor_c, tensor_d_ht, tensor_d_ht_1,
                         tensor_d_ct, tensor_it, tensor_jt,
                         tensor_ft, tensor_ot, tensor_tanh_ct]
    # elewise compute for dgate and dc
    tensor_list = {}
    emit_list = {}
    scope_list = {}

    tensor_list, \
    emit_list, \
    scope_list, tensor_jt_dc_list, \
    tensor_it_dc_list, \
    tensor_ft_dc_list = copy_input_to_ub_compute(is_last_time, tensor_c, tensor_d_ht, tensor_d_ht_1,
                                                 tensor_d_ct, tensor_it, tensor_jt,
                                                 tensor_ft, tensor_ot,
                                                 tensor_tanh_ct, tensor_list,
                                                 emit_list, scope_list)

    ele_result_list, scope_list, \
    tensor_list, emit_list = elewise_compute_for_gate(tensor_list["tensor_c_ub_ele"],
                                                      tensor_list["tensor_d_ht_ub_ele"],
                                                      tensor_list["tensor_d_ht_1_ub_ele"],
                                                      tensor_list["tensor_d_ht_1_ub_ct_ele"],
                                                      tensor_list["tensor_c_ub_ele"],
                                                      tensor_list["tensor_it_ub_ele"],
                                                      tensor_list["tensor_it_ub_dup_ele"],
                                                      tensor_list["tensor_jt_ub_ele"],
                                                      tensor_list["tensor_ft_ub_ele"],
                                                      tensor_list["tensor_ot_ub_ele"],
                                                      tensor_list["tensor_tanh_ct_ub_ele"],
                                                      tensor_list["tensor_d_ht_ub_ct_ele"],
                                                      tensor_list["tensor_ot_ub_ct_ele"],
                                                      tensor_list["tensor_tanh_ct_ub_ct_ele"],
                                                      tensor_list["tensor_d_ct_ub_ct_ele"],
                                                      tensor_list["tensor_ft_ub_ct_ele"],
                                                      tensor_jt_dc_list, tensor_it_dc_list, tensor_ft_dc_list,
                                                      tensor_list["tensor_jt_it_ub_ele"],
                                                      tensor_list, emit_list, scope_list)

    tensor_d_gate = ele_result_list[0]
    tensor_d_ct1 = ele_result_list[1]

    tensor_dx, tensor_dh, \
    tensor_list, emit_list, scope_list = matmul_compute(tensor_d_gate, tensor_w,
                                                        tensor_list, emit_list, scope_list, is_last_time)

    tensor_output_list = [tensor_dx, tensor_dh, tensor_d_ct1, tensor_d_gate]
    build_list = [tensor_w, tensor_c, tensor_d_ht, tensor_d_ht_1, tensor_d_ct, tensor_it,
                  tensor_jt, tensor_ft, tensor_ot, tensor_tanh_ct,
                  tensor_d_gate, tensor_dx, tensor_dh, tensor_d_ct1]

    return build_list, tensor_input_list, tensor_output_list, tensor_list, emit_list, scope_list


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def matmul_schedule(sch, matmul_res, tensor_list, tiling_info, cast_type):
    """
    :param sch:
    :param matmul_res:
    :return:
    """

    tensor_c = matmul_res.op.input_tensors[0]
    tensor_a_l0 = tensor_c.op.input_tensors[0]
    tensor_b_l0 = tensor_c.op.input_tensors[1]
    tensor_a_l1 = tensor_a_l0.op.input_tensors[0]
    tensor_b_l1 = tensor_b_l0.op.input_tensors[0]
    from te.platform import scope_cbuf
    from te.platform import scope_ca
    from te.platform import scope_cb
    from te.platform import scope_cc
    sch[tensor_a_l1].set_scope(scope_cbuf)
    sch[tensor_b_l1].set_scope(scope_cbuf)
    sch[tensor_a_l0].set_scope(scope_ca)
    sch[tensor_b_l0].set_scope(scope_cb)
    sch[tensor_c].set_scope(scope_cc)
    sch[matmul_res].set_scope(scope_ubuf)

    # compute_at and tiling  [6, 1, 16, 16]
    n_outer, n_inner = sch[tensor_c].split(tensor_c.op.axis[1],
                                           factor=1)

    m_outer, m_inner = sch[tensor_c].split(tensor_c.op.axis[2],
                                           factor=tiling_info["factor_n"])

    kb_outer, kb_inner = sch[tensor_c].split(tensor_c.op.reduce_axis[0],
                                             factor=tiling_info["k_factor"])

    n_inner_outer, n_inner_inner = sch[tensor_c].split(n_inner,
                                                       factor=2 * tiling_info["factor_output"])
    m_inner_outer, m_inner_inner = sch[tensor_c].split(m_inner,
                                                       factor=tiling_info["factor_n"])

    kb_inner_outer, kb_inner_inner = sch[tensor_c].split(kb_inner,
                                                         factor=tiling_info["k_factor"])

    sch[tensor_c].reorder(tensor_c.op.axis[0], n_outer, m_outer, kb_outer,
                          n_inner_outer, m_inner_outer, kb_inner_outer,
                          n_inner_inner, m_inner_inner, tensor_c.op.axis[3],
                          tensor_c.op.axis[4], kb_inner_inner,
                          tensor_c.op.reduce_axis[1])

    compute_axis = kb_inner_outer
    sch[tensor_a_l1].compute_at(sch[tensor_c], compute_axis)
    sch[tensor_b_l1].compute_at(sch[tensor_c], compute_axis)
    sch[tensor_a_l0].compute_at(sch[tensor_c], compute_axis)
    sch[tensor_b_l0].compute_at(sch[tensor_c], compute_axis)
    if cast_type:
        sch[tensor_list["tensor_d_gate_fp16"]].compute_at(sch[tensor_c], compute_axis)
        sch[tensor_list["tensor_w_fp16"]].compute_at(sch[tensor_c], compute_axis)
        sch[tensor_list["tensor_w_fp16"]].set_scope(scope_ubuf)
        sch[tensor_list["tensor_d_gate_fp16"]].set_scope(scope_ubuf)

        sch[tensor_list["tensor_d_gate_fp16"]].emit_insn(sch[tensor_list["tensor_d_gate_fp16"]].op.axis[1],
                                                         emit_list["tensor_d_gate_fp16"])
        sch[tensor_list["tensor_w_fp16"]].emit_insn(sch[tensor_list["tensor_w_fp16"]].op.axis[1],
                                                    emit_list["tensor_w_fp16"])
    sch[tensor_list["tensor_d_gate"]].compute_at(sch[tensor_c], compute_axis)

    # tensor_d_gate: ele_res[ub -> gm]  tensor_a_l1: [res_gm -> L1]   [res_ub -> L1]

    # split fake node
    tensor_concat_fake = tensor_list["tensor_concat_fake"]
    fake_n_outer, fake_n_inner = sch[tensor_concat_fake].split(tensor_concat_fake.op.axis[1],
                                                               factor=2 * tiling_info["factor_output"])
    fake_m_outer, fake_m_inner = sch[tensor_concat_fake].split(tensor_concat_fake.op.axis[2],
                                                               factor=tiling_info["factor_n"])
    # in 8  out 16   24 = 6*4

    fake_n_outer_outer, fake_n_outer_inner = sch[tensor_concat_fake].split(fake_n_inner,
                                                                           factor=2 * tiling_info["factor_output"])

    fake_m_outer_outer, fake_m_outer_inner = sch[tensor_concat_fake].split(fake_m_inner,
                                                                           factor=tiling_info["factor_n"])

    sch[tensor_concat_fake].reorder(tensor_concat_fake.op.axis[0],
                                    fake_n_outer, fake_m_outer, fake_n_outer_outer,
                                    fake_m_outer_outer, fake_n_outer_inner, fake_m_outer_inner,
                                    tensor_concat_fake.op.axis[3], tensor_concat_fake.op.axis[4])

    sch[tensor_list["tensor_dx"]].compute_at(sch[tensor_concat_fake], fake_m_outer)
    sch[tensor_list["tensor_dh"]].compute_at(sch[tensor_concat_fake], fake_m_outer)
    sch[matmul_res].compute_at(sch[tensor_concat_fake], fake_m_outer)
    sch[tensor_c].compute_at(sch[tensor_concat_fake], fake_m_outer)
    sch[tensor_list["tensor_add_fake"]].compute_at(sch[tensor_concat_fake], fake_m_outer)
    sch[tensor_list["tensor_d_ct1_gm_ele"]].compute_at(sch[tensor_concat_fake], fake_m_outer)

    sch[tensor_a_l1].emit_insn(sch[tensor_a_l1].op.axis[1], insn_cmd.DMA_COPY)
    sch[tensor_b_l1].emit_insn(sch[tensor_b_l1].op.axis[1], insn_cmd.DMA_COPY)

    sch[tensor_a_l0].emit_insn(sch[tensor_a_l0].op.axis[1], insn_cmd.DMA_COPY)
    sch[tensor_b_l0].emit_insn(sch[tensor_b_l0].op.axis[1], insn_cmd.DMA_COPY)

    sch[matmul_res].emit_insn(sch[matmul_res].op.axis[1], insn_cmd.DMA_COPY)

    # emit_insn for dht

    sch[tensor_list["tensor_add_fake"]].emit_insn(sch[tensor_list["tensor_add_fake"]].op.axis[1], "phony_insn")
    sch[tensor_list["tensor_concat_fake"]].emit_insn(fake_n_outer_inner, "phony_insn")
    from te.platform.cce_params import GEMM_MODE
    mad_pattern = GEMM_MODE
    mad_dict = {"mad_pattern": mad_pattern,
                "k_outer": [kb_outer, kb_inner_outer],
                }
    sch[tensor_c].emit_insn(n_inner_inner, 'mad', mad_dict)

    return sch


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def schedule_for_cell(tensor_input_list, tensor_list, tensor_output_list, emit_list, scope_list, build_list, cast_type):
    """
    return the schedule object for the compute process
    """
    schedule_list = [tensor_list["tensor_concat_fake"].op]
    sch = tvm.create_schedule(schedule_list)

    # set scope
    for key in tensor_list.keys():
        if scope_list.__contains__(key):
            sch[tensor_list[key]].set_scope(scope_list[key])
        if emit_list.__contains__(key):
            if key != "tensor_d_gate":
                sch[tensor_list[key]].emit_insn(sch[tensor_list[key]].op.axis[1], emit_list[key])

    # compute_at
    tensor_d_gate = tensor_list["tensor_d_gate"]

    input_size = tensor_list["tensor_concat_fake"].shape[1].value - (tensor_d_gate.shape[1].value // 4)
    tiling_info = get_tiling(input_size, tensor_d_gate.shape[1].value // 4, tensor_d_gate.shape[2].value)
    o_outer, o_inner = sch[tensor_d_gate].split(tensor_d_gate.op.axis[1], factor=tiling_info["factor_output"] * 4)
    n_outer, n_inner = sch[tensor_d_gate].split(tensor_d_gate.op.axis[2], factor=tiling_info["factor_n"])
    sch[tensor_d_gate].reorder(tensor_d_gate.op.axis[0],
                               o_outer, n_outer, o_inner, n_inner,
                               tensor_d_gate.op.axis[3],
                               tensor_d_gate.op.axis[4])
    sch[tensor_d_gate].emit_insn(o_inner, insn_cmd.DMA_COPY)

    # wanglinmu

    concat_compute_axis = n_outer
    no_attach_compute = ["tensor_d_ct1_ele", "tensor_d_ct1_gm_ele"]

    tensor_ct_list = {}
    for key in tensor_list.keys():
        if "ele" in key and key not in no_attach_compute:
            if "_ct_ele" not in key:
                sch[tensor_list[key]].compute_at(sch[tensor_d_gate], concat_compute_axis)
            else:
                tensor_ct_list[key] = tensor_list[key]

    tensor_d_ct1_gm = tensor_list["tensor_d_ct1_gm_ele"]
    # tensor_c_output [output, n, 16, 16]
    ct_gm_o_outer, ct_gm_o_inner = sch[tensor_d_ct1_gm].split(tensor_d_ct1_gm.op.axis[1],
                                                              factor=tiling_info["factor_output"])
    ct_gm_n_outer, ct_gm_n_inner = sch[tensor_d_ct1_gm].split(tensor_d_ct1_gm.op.axis[2],
                                                              factor=tiling_info["factor_n"])
    sch[tensor_d_ct1_gm].reorder(tensor_d_ct1_gm.op.axis[0],
                                 ct_gm_o_outer, ct_gm_n_outer, ct_gm_o_inner,
                                 ct_gm_n_inner, tensor_d_ct1_gm.op.axis[3],
                                 tensor_d_ct1_gm.op.axis[4])
    sch[tensor_d_ct1_gm].emit_insn(ct_gm_o_inner, insn_cmd.DMA_COPY)

    ct_gm_compute_axis = ct_gm_n_outer
    sch[tensor_list["tensor_d_ct1_ele"]].compute_at(sch[tensor_d_ct1_gm], ct_gm_compute_axis)
    for key in tensor_ct_list:
        sch[tensor_list[key]].compute_at(sch[tensor_d_ct1_gm], ct_gm_compute_axis)

    # tiling

    # emit_insn
    matmul_res = tensor_list["matmul_res"]
    sch = matmul_schedule(sch, matmul_res, tensor_list, tiling_info, cast_type)

    return sch


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def get_tiling(input_size, output_size, n):
    """
    return the tiling result
    """
    # tiling_info = {"factor_n": factor_n, "factor_output": factor_output}
    k_shape = 4 * output_size

    k_factors = [64, 32, 16]
    k_factor = -1
    for cur_k in k_factors:
        if k_shape % cur_k == 0:
            k_factor = cur_k

    if k_factor == -1:
        if k_shape > 64:
            k_factor = 64
        else:
            k_factor = k_shape

    def get_factor_list(input_size, output_size):
        res = []
        min_num = input_size
        if input_size > output_size:
            min_num = output_size

        for i in range(1, min_num):
            if input_size % i == 0 and output_size % i == 0:
                res.append(i)

        return res

    output_list = get_factor_list(input_size, output_size)

    def get_ub_size(n, output_size):
        broadcast = output_size * n * 16 * 16

        ub_size = 11 * broadcast + 6 * broadcast

        return ub_size

    ub_size = cceconf.CceProductParams().getParams("Unified_Buffer")

    total_ub_size = ub_size // 4
    factor_n = n

    def _decrement_out_factor(temp_out_factor, block_out):
        div = (block_out // temp_out_factor) + 1
        while block_out % div != 0 and div < block_out:
            div = div + 1
        res = block_out // div

        return res

    while get_ub_size(factor_n, output_size) >= total_ub_size and factor_n > 1:
        factor_n = _decrement_out_factor(factor_n, n)

    output_list.sort(reverse=True)
    factor_output = -1

    if get_ub_size(factor_n, output_size) >= total_ub_size:
        for cur_output in output_list:
            if get_ub_size(factor_n, cur_output) < total_ub_size:
                factor_output = cur_output
                break;

    tiling_info = {"factor_n": factor_n, "factor_output": factor_output, "k_factor": k_factor}

    if factor_output == -1:
        tiling_info = {'factor_n': 1, 'factor_output': 2, 'k_factor': 16}

    return tiling_info


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def do_cell_process(input_list, isExtentbuffer):
    """
    The cell op for LSTM
    """

    # there compute for cell. And then, use tik to do the compute
    tensor_w = input_list[0]
    tensor_c = input_list[1]
    tensor_d_ht = input_list[2]
    tensor_d_ht_1 = input_list[3]
    tensor_d_ct = input_list[4]
    tensor_it = input_list[5]
    tensor_jt = input_list[6]

    tensor_ft = input_list[7]
    tensor_ot = input_list[8]
    tensor_tanh_ct = input_list[9]

    is_last_time = isExtentbuffer[0]

    build_list, tensor_input_list, \
    tensor_output_list, tensor_list, \
    emit_list, scope_list = basic_lstm_cell_compute(tensor_w, tensor_c, tensor_d_ht, tensor_d_ht_1,
                                                    tensor_d_ct, tensor_it, tensor_jt,
                                                    tensor_ft, tensor_ot, tensor_tanh_ct, is_last_time)

    sch = schedule_for_cell(tensor_input_list, tensor_list,
                            tensor_output_list, emit_list,
                            scope_list, build_list, is_last_time)

    return tensor_output_list, sch


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda,too-many-locals,invalid-name,too-many-arguments
def lstm_input_grad(w, init_c, c, dy, dh, dc, i, j,
                    f, o, tanhct, dx,
                    dh_prev, dc_prev, dgate,
                    kernel_name="lstm_input_grad"):
    """
    Parameters
    ----------
    w : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    init_c : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        float32, the format can be [ND]
    c : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dy : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dh : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dc : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    i : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [ND]
    j : dict
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [ND]
    f:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    o:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    tanhct:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dx:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dh_prev:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dc_prev:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    dgate:
        A dict object, contains a Tensor 's type and
        shape and format, the type can be float16,
        the format can be [FRACTAL_NZ]
    kernel_name : str
        cce kernel name, default value == "lstm_input_grad"
    Returns
    -------
    None
    """
    dct = dc
    it = i
    jt = j
    ft = f
    ot = o
    dxt = dx
    dh_t = dh_prev
    from te.tik import scope_gm
    from te.tik import Tik
    from te.tik import Dprofile
    tik_instance = Tik(Dprofile())
    inp_src_dtype = w["dtype"].lower()

    # add 1
    w_shape = [1, w["shape"][0], w["shape"][1], w["shape"][2], w["shape"][3]]
    tensor_w = tik_instance.Tensor(inp_src_dtype, w_shape, scope_gm, "tensor_w")

    init_c_shape = [1, init_c["shape"][0], init_c["shape"][1], init_c["shape"][2], init_c["shape"][3]]
    tensor_init_c = tik_instance.Tensor(inp_src_dtype, init_c_shape, scope_gm, "tensor_init_c")

    dh_shape = [1, dh["shape"][0], dh["shape"][1], dh["shape"][2], dh["shape"][3]]
    tensor_dh = tik_instance.Tensor(inp_src_dtype, dh_shape, scope_gm, "tensor_dh")

    tensor_c = tik_instance.Tensor(inp_src_dtype, c["shape"], scope_gm, "tensor_c")
    tensor_d_ht = tik_instance.Tensor(inp_src_dtype, dy["shape"], scope_gm, "tensor_d_ht")
    # add 1
    dct_shape = [1, dct["shape"][0], dct["shape"][1], dct["shape"][2], dct["shape"][3]]
    tensor_d_ct = tik_instance.Tensor(inp_src_dtype, dct_shape, scope_gm, "tensor_d_ct")
    tensor_it = tik_instance.Tensor(inp_src_dtype, it["shape"], scope_gm, "tensor_it")
    tensor_jt = tik_instance.Tensor(inp_src_dtype, jt["shape"], scope_gm, "tensor_jt")
    tensor_ft = tik_instance.Tensor(inp_src_dtype, ft["shape"], scope_gm, "tensor_ft")
    tensor_ot = tik_instance.Tensor(inp_src_dtype, ot["shape"], scope_gm, "tensor_ot")
    tensor_tanh_ct = tik_instance.Tensor(inp_src_dtype, tanhct["shape"], scope_gm, "tensor_tanh_ct")

    tensor_output_dxt = tik_instance.Tensor(inp_src_dtype, dxt["shape"], scope_gm, "tensor_output_dxt")
    # add 1
    dh_t_shape = [1, dh_t["shape"][0], dh_t["shape"][1], dh_t["shape"][2], dh_t["shape"][3]]
    tensor_output_dht = tik_instance.Tensor(inp_src_dtype, dh_t_shape, scope_gm, "tensor_output_dht")
    tensor_output_dgate = tik_instance.Tensor(inp_src_dtype, dgate["shape"], scope_gm, "tensor_output_dgate")
    # add 1
    tensor_output_dct = tik_instance.Tensor(inp_src_dtype, dh_t_shape, scope_gm, "tensor_output_dct")

    temp_output_dct = tik_instance.Tensor(inp_src_dtype, dgate["shape"], scope_gm,
                                          "temp_output_dct", is_workspace=True)
    temp_output_dht = tik_instance.Tensor(inp_src_dtype, dgate["shape"], scope_gm,
                                          "temp_output_dht", is_workspace=True)

    t_num = c["shape"][0]

    n_num = c["shape"][2]

    start_num = 0
    stop_num = t_num

    with tik_instance.for_range(start_num, stop_num) as cur_t:
        with tik_instance.for_range(0, n_num, block_num=n_num) as batch:
            i = t_num - cur_t - 1
            tensor_w_cur = tensor_w
            tensor_c_cur = tensor_c[i - 1: i, :, batch:batch + 1, :, :]
            tensor_d_ht_cur = tensor_d_ht[i: i + 1, :, batch:batch + 1, :, :]

            # [i*j + j, :, :, :]
            tensor_it_cur = tensor_it[i: i + 1, :, batch:batch + 1, :, :]
            tensor_jt_cur = tensor_jt[i: i + 1, :, batch:batch + 1, :, :]
            tensor_ft_cur = tensor_ft[i: i + 1, :, batch:batch + 1, :, :]
            tensor_ot_cur = tensor_ot[i: i + 1, :, batch:batch + 1, :, :]
            tensor_tanh_ct_cur = tensor_tanh_ct[i: i + 1, :, batch:batch + 1, :, :]

            tensor_output_dxt_cur = tensor_output_dxt[i: i + 1, :, batch:batch + 1, :, :]

            tensor_output_dgate_cur = tensor_output_dgate[i: i + 1, :, batch:batch + 1, :, :]
            tensor_output_dct_cur = tensor_output_dct[0, :, batch:batch + 1, :, :]

            tensor_output_dht_cur = tensor_output_dht[0, :, batch:batch + 1, :, :]

            temp_output_dct_cur = temp_output_dct[cur_t + 1: cur_t + 2, :, batch:batch + 1, :, :]
            temp_output_dht_cur = temp_output_dht[cur_t + 1: cur_t + 2, :, batch:batch + 1, :, :]

            temp_input_dct_cur = temp_output_dct[cur_t: cur_t + 1, :, batch:batch + 1, :, :]
            temp_input_dht_cur = temp_output_dht[cur_t: cur_t + 1, :, batch:batch + 1, :, :]

            has_flag = False
            if inp_src_dtype == "float32":
                has_flag = True
            with tik_instance.if_scope(cur_t == 0):
                with tik_instance.if_scope(cur_t == t_num - 1):
                    tik_instance.call_module(
                        do_cell_process,
                        [tensor_w_cur, tensor_init_c[0, :, batch:batch + 1, :, :],
                         tensor_dh[0, :, batch:batch + 1, :, :], tensor_d_ht_cur,
                         tensor_d_ct[0, :, batch:batch + 1, :, :], tensor_it_cur, tensor_jt_cur, tensor_ft_cur,
                         tensor_ot_cur, tensor_tanh_ct_cur],
                        [tensor_output_dxt_cur, tensor_output_dht_cur,
                         tensor_output_dct_cur, tensor_output_dgate_cur],
                        [has_flag])
                with tik_instance.else_scope():
                    tik_instance.call_module(
                        do_cell_process,
                        [tensor_w_cur, tensor_c_cur, tensor_dh[0, :, batch:batch + 1, :, :],
                         tensor_d_ht_cur,
                         tensor_d_ct[0, :, batch:batch + 1, :, :], tensor_it_cur, tensor_jt_cur, tensor_ft_cur,
                         tensor_ot_cur, tensor_tanh_ct_cur],
                        [tensor_output_dxt_cur, temp_output_dht_cur,
                         temp_output_dct_cur, tensor_output_dgate_cur],
                        [has_flag])
            with tik_instance.else_scope():
                with tik_instance.if_scope(cur_t == t_num - 1):
                    tik_instance.call_module(
                        do_cell_process,
                        [tensor_w_cur, tensor_init_c[0, :, batch:batch + 1, :, :],
                         temp_input_dht_cur,
                         tensor_d_ht_cur,
                         temp_input_dct_cur,
                         tensor_it_cur, tensor_jt_cur, tensor_ft_cur,
                         tensor_ot_cur, tensor_tanh_ct_cur],
                        [tensor_output_dxt_cur, tensor_output_dht_cur,
                         tensor_output_dct_cur, tensor_output_dgate_cur],
                        [has_flag])
                with tik_instance.else_scope():
                    tik_instance.call_module(
                        do_cell_process,
                        [tensor_w_cur, tensor_c_cur,
                         temp_input_dht_cur,
                         tensor_d_ht_cur,
                         temp_input_dct_cur,
                         tensor_it_cur, tensor_jt_cur, tensor_ft_cur,
                         tensor_ot_cur, tensor_tanh_ct_cur],
                        [tensor_output_dxt_cur, temp_output_dht_cur,
                         temp_output_dct_cur, tensor_output_dgate_cur],
                        [has_flag])

    config_map = {
        "dump_cce_code": False,
    }

    input_list = [tensor_w, tensor_init_c, tensor_c, tensor_d_ht, tensor_dh, tensor_d_ct,
                  tensor_it, tensor_jt, tensor_ft, tensor_ot, tensor_tanh_ct]
    output_list = [tensor_output_dxt, tensor_output_dht, tensor_output_dct, tensor_output_dgate]
    tik_instance.BuildCCE(kernel_name,
                          input_list,
                          output_list,
                          config=config_map)
