#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

PadD: Not Align
"""
from te import tik
# vector_repeat
MAX_REPEAT = 255
# block_size
BLOCK_SIZE = 32


def set_vector_dup(obj, num_data, number):
    """
    Re:
    Func supports that num_data == N*mask(less than buf_size)
    """
    tik_instance = obj.tik_instance
    unit = MAX_REPEAT * obj.mask
    repeat_merchant = num_data // unit
    repeat_remainder = num_data % unit
    dst_blk_stride = 1
    dst_rep_stride = 8

    with tik_instance.for_range(0, repeat_merchant) as i:
        tik_instance.vector_dup(obj.mask,
                                obj.buf[i*unit],
                                number,
                                MAX_REPEAT,
                                dst_blk_stride,
                                dst_rep_stride)

    with tik_instance.if_scope(repeat_remainder != 0):
        repeats = repeat_remainder / obj.mask
        with tik_instance.if_scope(repeats != 0):
            tik_instance.vector_dup(obj.mask,
                                    obj.buf[repeat_merchant*unit],
                                    number,
                                    repeats,
                                    dst_blk_stride,
                                    dst_rep_stride)


def copy_buf2gm_circulation(obj, ac_num, vir_num, dst_idx, pattern=None):
    """
    ac_num: vol of actual_data in UB.
    vir_num: ultimate value for ac_num that must be 32B align.
    pattern: "top" and "bottom" in recursion will has different ways.
    Re: Func has three kinds to move_out.
    1. ac_num is too lager to save in UB.
    2. ac_num >= 32B that includes part of align and not.
    3. ac_num < 32B that only support "Single Core".
    """
    tik_instance = obj.tik_instance
    num_bit = obj.num_bit
    dst = obj.output_gm
    src = obj.buf

    tail = ac_num // vir_num
    tail_block = ac_num % vir_num
    block_num = BLOCK_SIZE // num_bit

    def _copy_ub2gm(begin_idx, data_len, idx):
        idx += begin_idx
        n_burst = 1
        burst_len = data_len * num_bit // BLOCK_SIZE
        src_stride = 0
        dst_stride = 0
        tik_instance.data_move(dst[idx],
                               src[0],
                               0,
                               n_burst,
                               burst_len,
                               src_stride,
                               dst_stride)

    # kind_0
    with tik_instance.if_scope(tail != 0):
        with tik_instance.for_range(0, tail) as serial:
            _copy_ub2gm(serial*vir_num, vir_num, dst_idx)

    with tik_instance.if_scope(tail_block != 0):
        align_vol = tail_block / block_num * block_num
        not_align_vol = tail_block % block_num
        offset = block_num - not_align_vol

        # kind_1
        with tik_instance.if_scope(align_vol != 0):
            _copy_ub2gm(tail*vir_num, align_vol, dst_idx)

        # kind_2
        with tik_instance.if_scope(not_align_vol != 0):
            address = tail * vir_num + align_vol - offset
            if pattern == "bottom":
                _copy_ub2gm(address, block_num, dst_idx)
            else:
                with tik_instance.if_scope(address > 0):
                    _copy_ub2gm(address, block_num, dst_idx)
                with tik_instance.else_scope():
                    _copy_ub2gm(0, block_num, dst_idx)


def _do_vec_dup(pattern, obj, max_num, blk_idx, mark, axis):
    """
    Params:
    top_address: start address for top padding.
    top_div_core: dividing line between two types of cores in top padding.
    top_total_core: physical cores for top padding.
    top_core_vol_x: volume of data processed by each core(type_x) for top padding.
    top_core_gap_x: gap between different cores(type_x) for top padding.

    Solution: MAX_CORE = 32
    in_shape is [34,16,16,16,...],func will work in [0, ] only.
    in_shape is [16,16,16,16,...],func will work in [0, 1].
    """
    if pattern == "top":
        begin_index = obj.top_address[axis]
        division_core = obj.top_div_core[axis]
        total_core = obj.top_total_core[axis]
        core_data_0 = obj.top_core_vol_0[axis]
        core_data_1 = obj.top_core_vol_1[axis]
        core_gap_0 = obj.top_core_gap_0[axis]
        core_gap_1 = obj.top_core_gap_1[axis]
        pad_data = obj.top_vol[axis]
    else:
        begin_index = obj.bottom_address[axis]
        division_core = obj.bottom_div_core[axis]
        total_core = obj.bottom_total_core[axis]
        core_data_0 = obj.bottom_core_vol_0[axis]
        core_data_1 = obj.bottom_core_vol_1[axis]
        core_gap_0 = obj.bottom_core_gap_0[axis]
        core_gap_1 = obj.bottom_core_gap_1[axis]
        pad_data = obj.bottom_vol[axis]

    # discriminate first layer or not.
    offset = obj.tik_instance.Scalar("int64", name="cir_offset_")
    offset_value = pad_data - core_data_0 * (division_core + 1) \
                   - core_data_1 * (total_core - division_core - 1)
    offset.set_as(offset_value)
    with obj.tik_instance.if_scope(pad_data - core_data_0 == 0):
        # not the first layer
        offset.set_as(0)

    vir_num, block_index = max_num, blk_idx

    # vector_dup: all physical cores.
    with obj.tik_instance.if_scope(mark != 1):
        set_vector_dup(obj, vir_num, 0)

    # data_move
    with obj.tik_instance.if_scope(block_index < division_core):
        dst_idx = begin_index + block_index * core_gap_0
        copy_buf2gm_circulation(obj, core_data_0, vir_num, dst_idx)

    with obj.tik_instance.if_scope(block_index == division_core):
        dst_idx = begin_index + division_core * core_gap_0
        copy_buf2gm_circulation(obj, core_data_0+offset, vir_num, dst_idx)

    with obj.tik_instance.if_scope(
            tik.all(block_index > division_core,
                    block_index < total_core)):
        begin_index += core_gap_0 * (division_core + 1) + offset
        block_index = block_index - (division_core + 1)
        dst_idx = begin_index + block_index * core_gap_1
        copy_buf2gm_circulation(obj, core_data_1, vir_num, dst_idx)


def _copy_gm2buf(obj, in_num, src_ub, src_gm):
    # ub must can be save all_data
    obj.tik_instance.data_move(obj.buf[src_ub],
                               obj.input_gm[src_gm],
                               0, 1,
                               in_num * obj.num_bit // BLOCK_SIZE,
                               0, 0)


def _copy_buf2buf(obj, n_burst, burst_len, src_stride, dst_stride, src_ub, dst_ub):
    obj.tik_instance.data_move(obj.buf[dst_ub],
                               obj.buf[src_ub],
                               0, n_burst, burst_len,
                               src_stride, dst_stride)


def _copy_buf2gm(obj, in_num, dst_gm, max_num):
    """
    Re:
    in_num: data that can be any value.
    Func requires in_num <= buf_size.
    """
    tik_instance = obj.tik_instance
    block_num = BLOCK_SIZE // obj.num_bit

    align_vol = in_num / block_num * block_num
    not_align_vol = in_num % block_num
    offset = block_num - not_align_vol

    def _move_out(begin_idx, data_len, dst_idx, buf):
        dst_idx += begin_idx
        n_burst = 1
        burst_len = data_len * obj.num_bit // BLOCK_SIZE
        src_stride = 0
        dst_stride = 0
        tik_instance.data_move(obj.output_gm[dst_idx],
                               buf[0],
                               0,
                               n_burst,
                               burst_len,
                               src_stride,
                               dst_stride)

    # Maybe not align.
    tik_align(obj, in_num, max_num, block_num)
    with tik_instance.if_scope(align_vol == 0):
        _move_out(0, block_num, dst_gm, obj.buf)

    with tik_instance.else_scope():
        _move_out(0, align_vol, dst_gm, obj.buf)
        # Move out not align
        with tik_instance.if_scope(not_align_vol != 0):
            index = align_vol-offset
            with tik_instance.for_range(0, block_num) as i:
                obj.help_buf[i] = obj.buf[index+i]
            _move_out(index, block_num, dst_gm, obj.help_buf)


def _data_move_last_dim(obj, in_num, src_gm, dst_gm, max_num):
    """
    in_num: actual input data(not padding).
    Re:
    Func requires in_num must >= 32(tiling.cpp)
    """
    tik_instance = obj.tik_instance
    block_num = BLOCK_SIZE // obj.num_bit
    vir_num = obj.buf_size

    # move align of in_num
    tail = in_num // vir_num
    tail_block = in_num % vir_num

    def _move_in(begin_idx, data_len, src_idx, buf):
        src_idx += begin_idx
        n_burst = 1
        burst_len = data_len * obj.num_bit // BLOCK_SIZE
        src_stride = 0
        dst_stride = 0
        tik_instance.data_move(buf[0],
                               obj.input_gm[src_idx],
                               0,
                               n_burst,
                               burst_len,
                               src_stride,
                               dst_stride)

    def _move_out(begin_idx, data_len, dst_idx, buf):
        dst_idx += begin_idx
        n_burst = 1
        burst_len = data_len * obj.num_bit // BLOCK_SIZE
        src_stride = 0
        dst_stride = 0
        tik_instance.data_move(obj.output_gm[dst_idx],
                               buf[0],
                               0,
                               n_burst,
                               burst_len,
                               src_stride,
                               dst_stride)

    # Must align: buf_size is N * mask.
    with tik_instance.if_scope(tail != 0):
        with tik_instance.for_range(0, tail) as serial:
            _move_in(serial*vir_num, vir_num, src_gm, obj.buf)
            _move_out(serial*vir_num, vir_num, dst_gm, obj.buf)

    # Maybe not align.
    with tik_instance.if_scope(tail_block != 0):
        align_vol = tail_block / block_num * block_num
        not_align_vol = tail_block % block_num
        offset = block_num - not_align_vol

        # Move in
        tik_align(obj, tail_block, max_num, block_num)
        with tik_instance.if_scope(align_vol == 0):
            _move_in(tail*vir_num-offset, block_num, src_gm, obj.buf)
            _move_out(tail*vir_num-offset, block_num, dst_gm, obj.buf)

        with tik_instance.else_scope():
            _move_in(tail*vir_num, max_num, src_gm, obj.buf)
            _move_out(tail*vir_num, align_vol, dst_gm, obj.buf)
            # Move out not align
            with tik_instance.if_scope(not_align_vol != 0):
                index = align_vol-offset
                with tik_instance.for_range(0, block_num) as i:
                    obj.help_buf[i] = obj.buf[index+i]
                _move_out(tail*vir_num+index, block_num, dst_gm, obj.help_buf)


def tik_max(obj, top, bottom, max_num):

    max_num.set_as(bottom)
    with obj.tik_instance.if_scope(top > bottom):
        max_num.set_as(top)
    tik_align(obj, max_num, max_num, obj.mask)


def tik_align(obj, in_num, max_num, align_vol):
    """
    in_num: vol of data
    max_num: scalar to save result of func.
    align_vol: standard of align: (BLOCK_SIZE/num_bit) or mask.
    buf_size: must be N*(BLOCK_SIZE/num_bit) or M*mask.
    Re:
    In module of "not_align", some vars must be align in computation.
    """
    max_num.set_as(in_num)
    with obj.tik_instance.if_scope(in_num % align_vol != 0):
        max_num.set_as((in_num / align_vol + 1) * align_vol)
    with obj.tik_instance.if_scope(max_num >= obj.buf_size):
        max_num.set_as(obj.buf_size)


def _circulation(obj, blk_idx, mark, axis):
    """
    eg: input: [16,16,22] output: [18,18,24]
        padding:[[1,1],[1,1],[1,1]]
        depth: 2

        input: [16,16,4] output: [18,18,6]
        padding:[[1,1],[1,1],[1,1]]
        depth: 1
        ps: input[1] can't satisfy multi core

    top_vol[0]: 1*18*24;
    bottom_vol[1]: 1*24;
    """
    # vol of padding.
    max_num = obj.tik_instance.Scalar("int32", name="max_num_")
    tik_max(obj, obj.top_vol[axis], obj.bottom_vol[axis], max_num)

    # do padding
    with obj.tik_instance.if_scope(obj.top_vol[axis] > 0):
        _do_vec_dup("top", obj, max_num, blk_idx, mark, axis)
        mark.set_as(1)

    with obj.tik_instance.if_scope(obj.bottom_vol[axis] > 0):
        _do_vec_dup("bottom", obj, max_num, blk_idx, mark, axis)
        mark.set_as(1)


def _recursion(obj, axis, dst_gm, src_gm, src_ub, dst_ub, max_num, mark):
    """
    recur_model: model include "Sort" and "MoveIn". "Sort" mean that shape[axis:] can be sorted in UB, "MoveIn" not.
    recur_dup_mk: mark of vector dup or not (One-Time-Triggered).
    prod_new_out: axis-by-axis multiplication base on new out_shape.
    prod_new_in: axis-by-axis multiplication base on new in_shape.
    recur_gm2buf_mk: mark of GM_2_BUF (One-Time-Triggered).
    new_padding_top: top of new_padding in recursion.
    new_in_shape: new in_shape in recursion
    """
    if axis == obj.axis_amount:
        return

    # ==================================
    # Only axis >= depth, tik will work.
    # ==================================
    # Status in different layers: Sort or MoveIn
    model = obj.recur_model[axis]
    buf_src = obj.tik_instance.Scalar("int32", name="buf_src_"+str(axis)+"_")
    buf_dst = obj.tik_instance.Scalar("int32", name="buf_dst_"+str(axis)+"_")
    buf_src.set_as(src_ub)
    buf_dst.set_as(dst_ub)

    # ===============================
    # Step1: Condition: "Sort"
    # Requirement: in_shape[-1] < 32
    # ===============================
    with obj.tik_instance.if_scope(model == 1):
        # Vector_Dup (One-Time-Triggered)
        # mark_dup: vec_dup or not.
        mark_dup = obj.recur_dup_mk[axis]
        with obj.tik_instance.if_scope(mark_dup == 1):
            tik_align(obj, obj.prod_new_out[axis], max_num, obj.mask)
            set_vector_dup(obj, max_num, 0)

        # GM_2_BUF (One-Time-Triggered)
        # mark_gm2buf: dma data from gm to ub or not.
        mark_gm2buf = obj.recur_gm2buf_mk[axis]
        with obj.tik_instance.if_scope(mark_gm2buf == 1):
            # init_align buf_src and num of moveIn.
            # requirement: align(output) + align(input) <= buf_size
            tik_align(obj, obj.prod_new_out[axis], buf_src, BLOCK_SIZE/obj.num_bit)
            tik_align(obj, obj.prod_new_in[axis], max_num, BLOCK_SIZE/obj.num_bit)
            _copy_gm2buf(obj, max_num, buf_src, src_gm)

        # Go to next level until the last dim
        top = obj.new_padding_top[axis] * obj.prod_new_out[axis+1]
        if axis <= obj.axis_amount - 2:
            loop = obj.new_in_shape[axis]
            with obj.tik_instance.for_range(0, loop) as i:
                dst_ub = buf_dst + top + obj.prod_new_out[axis+1] * i
                src_ub = buf_src + obj.prod_new_in[axis+1] * i
                _recursion(obj, axis+1, dst_gm, src_gm, src_ub, dst_ub, max_num, True)

        # the last dim
        # require total_num_ub < 32
        else:
            total_scalar = obj.prod_new_in[axis]
            with obj.tik_instance.for_range(0, total_scalar) as i:
                obj.buf[buf_dst + top + i] = obj.buf[buf_src + i]

        # BUF_2_GM (One-Time-Triggered)
        # Only happened in the layer which GM_2_BUF had worked.
        with obj.tik_instance.if_scope(mark_gm2buf == 1):
            in_num = obj.prod_new_out[axis]
            _copy_buf2gm(obj, in_num, dst_gm, max_num)

    # ================================
    # Step0: Condition: "MoveIn"
    # Requirement: in_shape[-1] >= 32
    # ================================
    if not mark:
        with obj.tik_instance.if_scope(model == 0):
            in_num_top = obj.new_padding_top[axis] * obj.prod_new_out[axis+1]
            in_num_bottom = obj.new_padding_bottom[axis] * obj.prod_new_out[axis+1]
            tik_max(obj, in_num_top, in_num_bottom, max_num)

            # vec_dup or not
            with obj.tik_instance.if_scope(max_num > 0):
                set_vector_dup(obj, max_num, 0)

            # axis in [0: last_dim), in_num_X must >= 32 or 0.
            # axis is last_dim, in_num_X can be any.
            with obj.tik_instance.if_scope(in_num_top > 0):
                copy_buf2gm_circulation(obj, in_num_top, max_num, dst_gm, "top")

            with obj.tik_instance.if_scope(in_num_bottom > 0):
                dst_gm_bottom = dst_gm + obj.new_in_shape[axis] * \
                                obj.prod_new_out[axis+1] + in_num_top
                copy_buf2gm_circulation(obj, in_num_bottom, max_num, dst_gm_bottom, "bottom")

            dst_gm += in_num_top

            if axis <= obj.axis_amount - 2:
                with obj.tik_instance.for_range(0, obj.new_in_shape[axis]) as i:
                    dst_gm += obj.prod_new_out[axis+1] * i
                    src_gm += obj.prod_new_in[axis+1] * i
                    _recursion(obj, axis+1, dst_gm, src_gm, buf_src, buf_dst, max_num, False)
            else:
                # copy_buf2gm until model is "MoveIn" in the last axis.
                _data_move_last_dim(obj, obj.prod_new_in[axis], src_gm, dst_gm, max_num)


def _circulation_compute(obj, blk_idx):
    """
    Supposing all axis should be traversed until axis exceed "depth".
    depth: depth from tiling.cpp.
    mark: status register to avoid invalid vector_dup in circulation
    """
    tik_instance = obj.tik_instance
    mark = obj.tik_instance.Scalar("int32", name="mark", init_value=0)
    for axis, _ in enumerate(range(obj.axis_amount)):
        with tik_instance.if_scope(axis < obj.depth[0]):
            _circulation(obj, blk_idx, mark, axis)


def _recursion_compute(obj, blk_idx):
    """
    recur_cond: condition that torch off stride between different cores.
    recur_gap_x: gap_x between in diff cores.
    recur_loop_x: work times by each core(type_x).
    recur_in_vol: volume of input_data by each core do once.
    recur_div_core: dividing line between two types of core.
    recur_total_core: physical cores in recursion.
    recur_start_address: start address in recursion
    """
    tik_instance = obj.tik_instance
    cond, gap0, gap1 = obj.recur_cond[0], obj.recur_gap_0[0], obj.recur_gap_1[0]
    loop0, loop1, in_vol = obj.recur_loop_0[0], obj.recur_loop_1[0], obj.recur_in_vol[0]
    max_num = obj.tik_instance.Scalar("int32", name="max_num_")

    def _main(processed, loop, block_index):
        src_ub = 0
        dst_ub = 0
        dst_gm = obj.recur_start_address[0]
        src_gm = 0
        axis = 0
        with tik_instance.for_range(0, loop) as idx:
            sum_core = processed + block_index * loop + idx
            dst_gm += sum_core / cond * gap0 + sum_core % cond * gap1
            src_gm += sum_core * in_vol
            _recursion(obj, axis, dst_gm, src_gm, src_ub, dst_ub, max_num, False)

    with tik_instance.if_scope(blk_idx <= obj.recur_div_core[0]):
        pro = 0
        _main(pro, loop0, blk_idx)

    with tik_instance.if_scope(tik.all(blk_idx > obj.recur_div_core[0],
                                       blk_idx < obj.recur_total_core[0])):

        pro = (obj.recur_div_core[0] + 1) * loop0
        blk_idx = blk_idx - obj.recur_div_core[0] - 1
        _main(pro, loop1, blk_idx)


def not_align_compute(obj, blk_idx):

    # =================
    # circulation layer
    # =================
    _circulation_compute(obj, blk_idx)

    # =================
    # recursion layer
    # =================
    _recursion_compute(obj, blk_idx)
