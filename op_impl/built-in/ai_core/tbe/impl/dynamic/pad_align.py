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

PadD: Align
"""
from te import tik
# vector_repeat
MAX_REPEAT = 255
# block_size
BLOCK_SIZE = 32


def set_vector_dup(obj, num_data, number):
    """
    num_data: volume of vec_dup.
    number: number will be filled in tensor.
    Re:
    Func supports any value of num_data(less than buf_size).
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
        repeats = repeat_remainder // obj.mask
        remainder = repeat_remainder % obj.mask

        with tik_instance.if_scope(repeats != 0):
            tik_instance.vector_dup(obj.mask,
                                    obj.buf[repeat_merchant*unit],
                                    number,
                                    repeats,
                                    dst_blk_stride,
                                    dst_rep_stride)

        with tik_instance.if_scope(remainder != 0):
            tik_instance.vector_dup(remainder,
                                    obj.buf
                                    [repeat_merchant*unit+repeats*obj.mask],
                                    number,
                                    1,
                                    dst_blk_stride,
                                    dst_rep_stride)


def copy_buf2gm_circulation(tik_instance, num_bit, ac_num,
                            vir_num, src, dst, dst_idx):
    """
    ac_num: actual data
    vir_num: ultimate value of ac_num(maybe buf_size)
    Re:
    Func requires ac_num and vir_num are 32B align.
    """
    tail = ac_num // vir_num
    tail_block = ac_num % vir_num

    def _copy_ub2gm(factor, data_len, idx):
        idx += factor * vir_num
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

    with tik_instance.if_scope(tail != 0):
        with tik_instance.for_range(0, tail) as serial:
            _copy_ub2gm(serial, vir_num, dst_idx)

    with tik_instance.if_scope(tail_block != 0):
        _copy_ub2gm(tail, tail_block, dst_idx)


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
    else:
        begin_index = obj.bottom_address[axis]
        division_core = obj.bottom_div_core[axis]
        total_core = obj.bottom_total_core[axis]
        core_data_0 = obj.bottom_core_vol_0[axis]
        core_data_1 = obj.bottom_core_vol_1[axis]
        core_gap_0 = obj.bottom_core_gap_0[axis]
        core_gap_1 = obj.bottom_core_gap_1[axis]

    vir_num, block_index = max_num, blk_idx

    # vector_dup: all physical cores.
    with obj.tik_instance.if_scope(mark != 1):
        set_vector_dup(obj, vir_num, 0)

    # data_move: part of physical cores.
    with obj.tik_instance.if_scope(block_index <= division_core):
        dst_idx = begin_index + block_index * core_gap_0

        copy_buf2gm_circulation(obj.tik_instance, obj.num_bit, core_data_0,
                                vir_num, obj.buf, obj.output_gm,
                                dst_idx)

    with obj.tik_instance.if_scope(
            tik.all(block_index > division_core,
                    block_index < total_core)):
        begin_index += core_gap_0 * (division_core + 1)
        block_index = block_index - (division_core + 1)
        dst_idx = begin_index + block_index * core_gap_1

        copy_buf2gm_circulation(obj.tik_instance, obj.num_bit, core_data_1,
                                vir_num, obj.buf, obj.output_gm,
                                dst_idx)


def _copy_gm2buf(obj, in_num, src_ub, src_gm):
    """
    Re:
    Func requires: in_num is 32B align and it must be less than buf_size.
    """
    obj.tik_instance.data_move(obj.buf[src_ub],
                               obj.input_gm[src_gm],
                               0, 1,
                               in_num * obj.num_bit // BLOCK_SIZE,
                               0, 0)


def _copy_buf2buf(obj, n_burst, burst_len, src_stride, dst_stride, src_ub, dst_ub):
    """
    Re:
    dst_ub and src_ub is 32B align.
    """
    obj.tik_instance.data_move(obj.buf[dst_ub],
                               obj.buf[src_ub],
                               0, n_burst, burst_len,
                               src_stride, dst_stride)


def _copy_buf2gm(obj, in_num, src_ub, dst_gm):
    """
    Re:
    Func requires: in_num is 32B align and it must be less than buf_size.
    """
    obj.tik_instance.data_move(obj.output_gm[dst_gm],
                               obj.buf[src_ub],
                               0, 1,
                               in_num * obj.num_bit // BLOCK_SIZE,
                               0, 0)


def _data_move_last_dim(obj, in_num, src_gm, dst_gm):
    """
    Re:
    Func requires: in_num is 32B align.
    MTE2 -> MTE3
    """

    tik_instance = obj.tik_instance
    tail = in_num // obj.buf_size
    tail_block = in_num % obj.buf_size

    def _main(serial, data_len, src_idx, dst_idx):
        src_idx += serial * obj.buf_size
        dst_idx += serial * obj.buf_size
        n_burst = 1
        burst_len = data_len * obj.num_bit // BLOCK_SIZE
        src_stride = 0
        dst_stride = 0

        tik_instance.data_move(obj.buf[0],
                               obj.input_gm[src_idx],
                               0,
                               n_burst,
                               burst_len,
                               src_stride,
                               dst_stride)

        tik_instance.data_move(obj.output_gm[dst_idx],
                               obj.buf[0],
                               0,
                               n_burst,
                               burst_len,
                               src_stride,
                               dst_stride)

    with tik_instance.if_scope(tail != 0):
        with tik_instance.for_range(0, tail) as serial:
            _main(serial, obj.buf_size, src_gm, dst_gm)

    with tik_instance.if_scope(tail_block != 0):
        _main(tail, tail_block, src_gm, dst_gm)


def tik_max(obj, top, bottom, max_num):

    with obj.tik_instance.if_scope(top >= bottom):
        max_num.set_as(top)
    with obj.tik_instance.else_scope():
        max_num.set_as(bottom)
    with obj.tik_instance.if_scope(max_num > obj.buf_size):
        max_num.set_as(obj.buf_size)


def _circulation(obj, blk_idx, mark, axis):
    """
    eg: input: [16,16,16], output:[18,18,48],
         padding:[[1,1],[1,1],[16,16]].
    top_vol[0]: 1*18*48;
    bottom_vol[1]: 1*48;
    """
    # compute max(top, bottom)
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
    # Status in different layers: Reorder or MoveIn
    model = obj.recur_model[axis]
    buf_src = obj.tik_instance.Scalar("int32", name="buf_src_"+str(axis)+"_")
    buf_dst = obj.tik_instance.Scalar("int32", name="buf_dst_"+str(axis)+"_")
    buf_src.set_as(src_ub)
    buf_dst.set_as(dst_ub)

    # ///////////////////////////
    # Step1: Condition: "Sort"///
    # ///////////////////////////
    with obj.tik_instance.if_scope(model == 1):
        # compute Vector_Dup (One-Time-Triggered)
        # mark_dup: vec_dup or not.
        mark_dup = obj.recur_dup_mk[axis]
        with obj.tik_instance.if_scope(mark_dup == 1):
            set_vector_dup(obj, obj.prod_new_out[axis], 0)

        # compute GM_2_BUF (One-Time-Triggered)
        # mark_gm2buf: dma data from gm to ub or not.
        mark_gm2buf = obj.recur_gm2buf_mk[axis]
        with obj.tik_instance.if_scope(mark_gm2buf == 1):
            buf_src.set_as(obj.prod_new_out[axis])
            in_num = obj.prod_new_in[axis]
            _copy_gm2buf(obj, in_num, buf_src, src_gm)

        # Go to next level until the last dim
        top = obj.new_padding_top[axis] * obj.prod_new_out[axis+1]
        if axis <= obj.axis_amount - 2:
            loop = obj.new_in_shape[axis]
            with obj.tik_instance.for_range(0, loop) as i:
                dst_ub = buf_dst + top + obj.prod_new_out[axis+1] * i
                src_ub = buf_src + obj.prod_new_in[axis+1] * i
                _recursion(obj, axis+1, dst_gm, src_gm, src_ub, dst_ub, max_num, True)

        # the last dim
        else:
            total_num_ub = obj.prod_new_in[axis]
            n_burst = 1
            burst_len = total_num_ub * obj.num_bit // BLOCK_SIZE
            src_stride = 0
            dst_stride = 0
            _copy_buf2buf(obj, n_burst, burst_len,
                          src_stride, dst_stride, buf_src, buf_dst+top)

        # compute BUF_2_GM (One-Time-Triggered)
        # Only happened in the layer which GM_2_BUF had worked.
        with obj.tik_instance.if_scope(mark_gm2buf == 1):
            in_num = obj.prod_new_out[axis]
            _copy_buf2gm(obj, in_num, 0, dst_gm)

    # ///////////////////////////////
    # Step0: Condition: "MoveIn"////
    # /////////////////////////////
    if not mark:
        with obj.tik_instance.if_scope(model == 0):
            in_num_top = obj.new_padding_top[axis] * obj.prod_new_out[axis+1]
            in_num_bottom = obj.new_padding_bottom[axis] * obj.prod_new_out[axis+1]
            tik_max(obj, in_num_top, in_num_bottom, max_num)

            # vec_dup or not
            with obj.tik_instance.if_scope(max_num > 0):
                set_vector_dup(obj, max_num, 0)

            with obj.tik_instance.if_scope(in_num_top > 0):
                copy_buf2gm_circulation(obj.tik_instance, obj.num_bit, in_num_top,
                                        max_num, obj.buf, obj.output_gm, dst_gm)

            with obj.tik_instance.if_scope(in_num_bottom > 0):
                dst_gm_bottom = dst_gm + obj.new_in_shape[axis] * \
                                obj.prod_new_out[axis+1] + in_num_top
                copy_buf2gm_circulation(obj.tik_instance, obj.num_bit, in_num_bottom,
                                        max_num, obj.buf, obj.output_gm, dst_gm_bottom)
            dst_gm += in_num_top

            if axis <= obj.axis_amount - 2:
                with obj.tik_instance.for_range(0, obj.new_in_shape[axis]) as i:
                    dst_gm += obj.prod_new_out[axis+1] * i
                    src_gm += obj.prod_new_in[axis+1] * i
                    _recursion(obj, axis+1, dst_gm, src_gm, buf_src, buf_dst, max_num, False)
            else:
                # copy_buf2gm until model is "MoveIn" in the last axis.
                _data_move_last_dim(obj, obj.prod_new_in[axis], src_gm, dst_gm)


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


def align_compute(obj, blk_idx):
    # =================
    # circulation layer
    # =================
    _circulation_compute(obj, blk_idx)

    # =================
    # recursion layer
    # =================
    # params: Only ONE fixed value.
    _recursion_compute(obj, blk_idx)
