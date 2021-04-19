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
dhwcn_2_fractal_z_3d
"""
from functools import reduce as functools_reduce

from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *
import te.platform.cce_params as cce_params


# UB size in byte
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# AICORE count
CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# C0 length
C0_LEN = 16
# repeat up limit
REPEAT_LIMIT = 255
# mask value
MASK_128 = 128


# pylint: disable=locally-disabled,too-many-lines
def _ceil_div(value_x, value_y):
    """
    do ceil division
    """
    return (value_x + value_y - 1) // value_y


def _ceil_fill(value, block):
    """
    fill the input value by block

    """
    return _ceil_div(value, block)*block


def _get_vnchwconv_ub_size(col_size, dtype):
    """
    get the ubuf size for vnchwconv branch
    """

    if dtype.lower() == "float16":
        byte_cnt = 2
    elif dtype.lower() == "float32":
        byte_cnt = 4

    # 16 lines, the unit is byte
    need_ub_size = _ceil_div(col_size, C0_LEN) * C0_LEN * C0_LEN * byte_cnt
    # the UB will be split into two parts, the unit is byte
    ub_half_248k_size = 248 * 1024 // 2
    ub_upper_limit = UB_SIZE // 2
    if ub_upper_limit > ub_half_248k_size:
        ub_upper_limit = ub_half_248k_size

    if need_ub_size >= ub_upper_limit:
        ub_size = ub_upper_limit // byte_cnt // C0_LEN * C0_LEN
    else:
        ub_size = need_ub_size // byte_cnt

    return ub_size


def _clean_ubuf(tik_inst, src, src_offset, dup_len):
    """
    clean ubuf to zero
    """

    if src.dtype.lower() == "float16":
        dtype_factor = 2
    elif src.dtype.lower() == "float32":
        dtype_factor = 1
    batch_size = 64

    if dup_len > 0:
        repeat = dup_len // (batch_size * dtype_factor)
        left_elem = dup_len % (batch_size * dtype_factor)
        repeat_loop = repeat // REPEAT_LIMIT
        repeat_left = repeat % REPEAT_LIMIT
        dup_value = float(0)

        if repeat_loop > 0:
            with tik_inst.for_range(0, repeat_loop) as rpt_idx:
                tik_inst.vector_dup(MASK_128,
                                    src[src_offset + rpt_idx *
                                        REPEAT_LIMIT *
                                        batch_size * dtype_factor],
                                    dup_value, REPEAT_LIMIT, 1, 8)

        if repeat_left > 0:
            tik_inst.vector_dup(MASK_128,
                                src[src_offset + repeat_loop *
                                    REPEAT_LIMIT *
                                    batch_size * dtype_factor],
                                dup_value, repeat_left, 1, 8)

        if left_elem > 0:
            tik_inst.vector_dup(left_elem,
                                src[src_offset + repeat *
                                    batch_size * dtype_factor],
                                dup_value, 1, 1, 8)


# pylint: disable=too-many-locals,too-many-statements
def _multi_core_on_c(tik_inst, data_in, data_out, shape_in):
    """
    process of multiple core on axis c
    """

    axis_d, axis_h, axis_w, axis_c, axis_n = shape_in
    multi_c_loop_cnt = (axis_c // C0_LEN) // CORE_NUM
    multi_c_loop_left = (axis_c // C0_LEN) % CORE_NUM
    axis_c_left = axis_c % C0_LEN
    vnchw_ub_size = _get_vnchwconv_ub_size(axis_n, data_in.dtype)
    # vnchwconv process 16 lines each time
    vnchw_col_size = vnchw_ub_size // C0_LEN

    in_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                             name="out_ub", scope=tik.scope_ubuf)

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
        with tik_inst.for_range(0, axis_d) as d_idx:
            with tik_inst.for_range(0, axis_h * axis_w) as hw_idx:

                def _inner_process(c_lp_idx, c_idx, c_cnt):
                    """
                    real transfer process for multiple core on axis c
                    """

                    ceil_axis_n = _ceil_div(axis_n, C0_LEN) * C0_LEN
                    n_16_size = ceil_axis_n * C0_LEN
                    idx_list = [0, 1, 2, 3, 4, 5, 6, 7,
                                8, 9, 10, 11, 12, 13, 14, 15]

                    def _inner_dhwcn_2_3d(n_lp_id, col_size):
                        """
                        do transfer from dhwcn to 3d
                        """

                        if axis_n == vnchw_col_size:
                            tik_inst.data_move(
                                in_ub,
                                data_in[n_lp_id * vnchw_col_size +
                                        (block_idx +
                                         c_lp_idx * CORE_NUM + c_idx) *
                                        C0_LEN * axis_n +
                                        (hw_idx + d_idx * axis_h * axis_w) *
                                        axis_c * axis_n],
                                0, 1,
                                c_cnt * _ceil_div(col_size, C0_LEN), 0, 0)
                        else:
                            with tik_inst.for_range(0, c_cnt) as c_index:
                                tik_inst.data_move(
                                    in_ub[c_index * vnchw_col_size],
                                    data_in[c_index * axis_n +
                                            n_lp_id * vnchw_col_size +
                                            (block_idx +
                                             c_lp_idx * CORE_NUM + c_idx) *
                                            C0_LEN * axis_n +
                                            (hw_idx + d_idx * axis_h * axis_w)
                                            * axis_c * axis_n],
                                    0, 1,
                                    _ceil_div(col_size, C0_LEN), 0, 0)

                        src_addr_list = [in_ub[vnchw_col_size * i] for i in
                                         idx_list]
                        dst_addr_list = [out_ub[C0_LEN * i] for i in idx_list]
                        repeat_cnt = _ceil_div(col_size, C0_LEN)
                        src_stride = 0 if repeat_cnt == 1 else 1
                        dst_stride = 0 if repeat_cnt == 1 else 16

                        tik_inst.vnchwconv(False, False, dst_addr_list,
                                           src_addr_list,
                                           repeat_cnt, dst_stride, src_stride)

                        # set n left to zero
                        if col_size % C0_LEN:
                            _clean_ubuf(tik_inst, out_ub,
                                        col_size * C0_LEN,
                                        (C0_LEN - col_size % C0_LEN) * C0_LEN)

                        ni_c0_size = C0_LEN * C0_LEN
                        no_ni_c0_size = _ceil_div(axis_n, C0_LEN) * ni_c0_size
                        tik_inst.data_move(
                            data_out[
                                n_lp_id * _ceil_div(vnchw_col_size, C0_LEN) *
                                ni_c0_size +
                                (hw_idx + (c_lp_idx * CORE_NUM + c_idx +
                                           block_idx + d_idx *
                                           _ceil_div(axis_c, C0_LEN)) *
                                 axis_h * axis_w) * no_ni_c0_size],
                            out_ub,
                            0, 1, repeat_cnt * C0_LEN, 0, 0)

                    if n_16_size > vnchw_ub_size:
                        axis_n_loop = axis_n // vnchw_col_size
                        axis_n_left = axis_n % vnchw_col_size

                        if axis_n_loop > 0:
                            with tik_inst.for_range(0,
                                                    axis_n_loop) as n_lp_idx:
                                _inner_dhwcn_2_3d(n_lp_idx, vnchw_col_size)

                        if axis_n_left > 0:
                            _inner_dhwcn_2_3d(axis_n_loop, axis_n_left)

                    else:
                        _inner_dhwcn_2_3d(0, axis_n)

                if multi_c_loop_cnt > 0:
                    with tik_inst.for_range(0, multi_c_loop_cnt) as c_loop_idx:
                        _inner_process(c_loop_idx, 0, C0_LEN)

                if multi_c_loop_left > 0:
                    with tik_inst.if_scope(block_idx < multi_c_loop_left):
                        _inner_process(multi_c_loop_cnt, 0, C0_LEN)

                if axis_c_left > 0:
                    _clean_ubuf(tik_inst,
                                in_ub,
                                axis_c_left * vnchw_col_size,
                                (C0_LEN - axis_c_left) * vnchw_col_size)

                    with tik_inst.if_scope(block_idx < 1):
                        _inner_process(0, axis_c // C0_LEN, axis_c_left)


def _multi_core_on_dhw(tik_inst, data_in, data_out, shape_in):
    """
    process of multiple core on axis d, h, w
    """

    axis_d, axis_h, axis_w, axis_c, _ = shape_in
    # move 16 * 256 elements each time
    out_ub_size = C0_LEN * C0_LEN * C0_LEN
    out_ub = tik_inst.Tensor(data_in.dtype, (out_ub_size,),
                             name="out_ub", scope=tik.scope_ubuf)
    # in order to keep ub size 16 align
    in_ub_size = (UB_SIZE // 2 - out_ub_size) // C0_LEN * C0_LEN
    in_ub = tik_inst.Tensor(data_in.dtype, (in_ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    c_list = [i for i in range(axis_c)]
    reg_list = [tik_inst.Scalar(data_in.dtype) for i in c_list]

    dhw_size = axis_d * axis_h * axis_w
    core_data_size = dhw_size // CORE_NUM * axis_c
    core_data_left = dhw_size % CORE_NUM * axis_c
    ni_c0_size = C0_LEN * C0_LEN

    # set out_ub to zero
    _clean_ubuf(tik_inst, out_ub, 0, out_ub_size)

    def _inner_process_dhw(block_index, slice_size, in_offset, out_offset):
        """
        real transfer process for multiple core on axis d, h, w
        """

        # to keep the axis_c align
        ub_align_c_size = in_ub_size // axis_c * axis_c

        def _inner_dhwcn_2_3d_dhw(lp_idx, col_size):
            """
            do transfer from dhwcn to 3d
            """

            tik_inst.data_move(in_ub,
                               data_in[lp_idx * ub_align_c_size +
                                       block_index * slice_size +
                                       in_offset],
                               0, 1, _ceil_div(col_size, C0_LEN), 0, 0)

            c_count = col_size // axis_c
            mv_loop = c_count // C0_LEN
            mv_left = c_count % C0_LEN

            def _move_elements(lp_index, mv_len):
                """
                move elements for output
                """

                if axis_c == C0_LEN:
                    tik_inst.data_move(out_ub,
                                       in_ub[lp_index * C0_LEN * C0_LEN],
                                       0, mv_len, 1, 0, 15)
                else:
                    with tik_inst.for_range(0, mv_len) as len_idx:
                        # mv_len * axis_c
                        for idx in c_list:
                            reg_list[idx].set_as(
                                in_ub[idx + len_idx * axis_c +
                                      lp_index * axis_c * C0_LEN])

                        for idx in c_list:
                            out_ub[len_idx * ni_c0_size + idx].set_as(
                                reg_list[idx])

                tik_inst.data_move(
                    data_out[(lp_index * C0_LEN +
                              lp_idx * (ub_align_c_size // axis_c) +
                              block_index * (slice_size // axis_c)) *
                             ni_c0_size + out_offset],
                    out_ub,
                    0, 1, mv_len * C0_LEN, 0, 0)

            if mv_loop > 0:
                with tik_inst.for_range(0, mv_loop) as mv_lp_idx:
                    _move_elements(mv_lp_idx, C0_LEN)

            if mv_left > 0:
                _move_elements(mv_loop, mv_left)

        if slice_size > in_ub_size:

            slice_loop = slice_size // ub_align_c_size
            slice_left = slice_size % ub_align_c_size

            if slice_loop > 0:
                with tik_inst.for_range(0, slice_loop) as slice_lp_idx:
                    _inner_dhwcn_2_3d_dhw(slice_lp_idx, ub_align_c_size)

            if slice_left > 0:
                _inner_dhwcn_2_3d_dhw(slice_loop, slice_left)

        else:
            _inner_dhwcn_2_3d_dhw(0, slice_size)

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
        if core_data_size > 0:
            _inner_process_dhw(block_idx, core_data_size, 0, 0)

        if core_data_left > 0:
            with tik_inst.if_scope(block_idx < (dhw_size % CORE_NUM)):
                _inner_process_dhw(
                    block_idx, axis_c,
                    core_data_size * CORE_NUM,
                    dhw_size // CORE_NUM * ni_c0_size * CORE_NUM)


# pylint: disable=too-many-statements, too-many-branches
def _multi_core_on_hw(tik_inst, data_in, data_out, shape_in):
    """
    process of multiple core on axis h, w
    """

    axis_d, axis_h, axis_w, axis_c, axis_n = shape_in

    if axis_n == 1:
        # move 16 * 256 elements each time
        out_ub_size = C0_LEN * C0_LEN * C0_LEN
        out_ub = tik_inst.Tensor(data_in.dtype, (out_ub_size,),
                                 name="out_ub", scope=tik.scope_ubuf)

        # in order to keep ub size 16 align
        in_ub_size = (UB_SIZE // 2 - out_ub_size) // C0_LEN * C0_LEN
        in_ub = tik_inst.Tensor(data_in.dtype, (in_ub_size,),
                                name="in_ub", scope=tik.scope_ubuf)

        # set out_ub to zero
        _clean_ubuf(tik_inst, out_ub, 0, out_ub_size)

        if axis_c % C0_LEN:
            c_list = [i for i in range(axis_c % C0_LEN)]
            reg_list = [tik_inst.Scalar(data_in.dtype) for i in c_list]

    else:
        vnchw_ub_size = _get_vnchwconv_ub_size(axis_n, data_in.dtype)
        # vnchwconv process 16 lines each time
        vnchw_col_size = vnchw_ub_size // C0_LEN
        in_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                                name="in_ub", scope=tik.scope_ubuf)
        out_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                                 name="out_ub", scope=tik.scope_ubuf)

    hw_size = axis_h * axis_w
    multi_hw_loop_cnt = hw_size // CORE_NUM
    multi_hw_left_cnt = hw_size % CORE_NUM
    c_loop_cnt = axis_c // C0_LEN
    c_left_cnt = axis_c % C0_LEN
    ni_c0_size = C0_LEN * C0_LEN
    c1hw_size = _ceil_div(axis_c, C0_LEN) * hw_size

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
        with tik_inst.for_range(0, axis_d) as d_idx:

            def _inner_process_hw(hw_lp_index):
                """
                real transfer process for multiple core on axis h, w
                """

                def _inner_dhwcn_2_3d_hw_vnchw():
                    """
                    do transfer from dhwcn to 3d by vnchwconv
                    """

                    ceil_axis_n = _ceil_div(axis_n, C0_LEN) * C0_LEN
                    n_16_size = ceil_axis_n * C0_LEN
                    idx_list = [0, 1, 2, 3, 4, 5, 6, 7,
                                8, 9, 10, 11, 12, 13, 14, 15]

                    def _c_loop_process(c_lp_index, c_cnt):
                        """
                        c loop process
                        """

                        def _vnchwconv_transfer_hw(n_lp_index, col_size):
                            """
                            vnchwconv transfer
                            """

                            if axis_n == vnchw_col_size:
                                tik_inst.data_move(
                                    in_ub,
                                    data_in[n_lp_index * vnchw_col_size +
                                            c_lp_index * C0_LEN * axis_n +
                                            (block_idx +
                                             hw_lp_index * CORE_NUM +
                                             d_idx * hw_size) *
                                            axis_c * axis_n],
                                    0, 1,
                                    c_cnt * _ceil_div(col_size, C0_LEN), 0, 0)
                            else:
                                with tik_inst.for_range(
                                        0, c_cnt) as c_line_idx:
                                    tik_inst.data_move(
                                        in_ub[c_line_idx * vnchw_col_size],
                                        data_in[c_line_idx * axis_n +
                                                n_lp_index * vnchw_col_size +
                                                c_lp_index * C0_LEN * axis_n +
                                                (block_idx +
                                                 hw_lp_index * CORE_NUM +
                                                 d_idx * hw_size) *
                                                axis_c * axis_n],
                                        0, 1,
                                        _ceil_div(col_size, C0_LEN), 0, 0)

                            src_addr_list = [in_ub[vnchw_col_size * i]
                                             for i in idx_list]
                            dst_addr_list = [out_ub[C0_LEN * i]
                                             for i in idx_list]
                            repeat_cnt = _ceil_div(col_size, C0_LEN)
                            src_stride = 0 if repeat_cnt == 1 else 1
                            dst_stride = 0 if repeat_cnt == 1 else 16

                            tik_inst.vnchwconv(False, False,
                                               dst_addr_list,
                                               src_addr_list,
                                               repeat_cnt,
                                               dst_stride, src_stride)
                            # set n left to zero
                            if col_size % C0_LEN:
                                _clean_ubuf(
                                    tik_inst, out_ub,
                                    col_size * C0_LEN,
                                    (C0_LEN - col_size % C0_LEN) * C0_LEN)

                            no_ni_c0_size = \
                                _ceil_div(axis_n, C0_LEN) * ni_c0_size
                            tik_inst.data_move(
                                data_out[n_lp_index *
                                         _ceil_div(vnchw_col_size, C0_LEN) *
                                         ni_c0_size +
                                         (c_lp_index * hw_size + block_idx +
                                          hw_lp_index * CORE_NUM +
                                          d_idx * c1hw_size) * no_ni_c0_size],
                                out_ub,
                                0, 1, repeat_cnt * C0_LEN, 0, 0)

                        if n_16_size > vnchw_ub_size:
                            n_loop = axis_n // vnchw_col_size
                            n_left = axis_n % vnchw_col_size

                            if n_loop > 0:
                                with tik_inst.for_range(0, n_loop) as n_lp_idx:
                                    _vnchwconv_transfer_hw(n_lp_idx,
                                                           vnchw_col_size)

                            if n_left > 0:
                                _vnchwconv_transfer_hw(n_loop, n_left)

                        else:
                            _vnchwconv_transfer_hw(0, axis_n)

                    if c_loop_cnt:
                        with tik_inst.for_range(0, c_loop_cnt) as c_lp_idx:
                            _c_loop_process(c_lp_idx, C0_LEN)

                    if c_left_cnt:
                        # clean the un-used lines of 16 lines to zero
                        _clean_ubuf(tik_inst, in_ub,
                                    c_left_cnt * vnchw_col_size,
                                    (C0_LEN - c_left_cnt) * vnchw_col_size)
                        _c_loop_process(c_loop_cnt, c_left_cnt)
                _inner_dhwcn_2_3d_hw_vnchw()

            def _inner_process_hwc(hw_len, in_offset, out_offset):
                """
                real transfer process for multiple core on axis h, w with n==1
                and c <= in_ub_size
                """

                def _dhwcn_2_3d_hw_hwc(hwc_index, col_size, c0_line):
                    """
                    dhwcn transfer to 3d by c with c <= in_ub_size
                    """

                    if not col_size % C0_LEN:
                        tik_inst.data_move(
                            in_ub,
                            data_in[(hwc_index * C0_LEN + block_idx * hw_len +
                                     d_idx * hw_size) * axis_c + in_offset],
                            0, c0_line, _ceil_div(col_size, C0_LEN), 0,
                            (hwc_col_size - col_size) // C0_LEN)
                    else:
                        with tik_inst.for_range(0, c0_line) as c0_idx:
                            tik_inst.data_move(
                                in_ub[c0_idx * hwc_col_size],
                                data_in[(hwc_index * C0_LEN +
                                         block_idx * hw_len +
                                         d_idx * hw_size) * axis_c +
                                        in_offset + c0_idx * col_size],
                                0, 1, _ceil_div(col_size, C0_LEN), 0, 0)

                    src_addr_list = [in_ub[hwc_col_size * i] for i in idx_list]
                    dst_addr_list = [in_ub[hwc_ub_size + C0_LEN * i]
                                     for i in idx_list]
                    repeat_cnt1 = _ceil_div(col_size, C0_LEN)
                    src_stride = 0 if repeat_cnt1 == 1 else 1
                    dst_stride = 0 if repeat_cnt1 == 1 else 16
                    # first vnchwconv
                    tik_inst.vnchwconv(False, False,
                                       dst_addr_list,
                                       src_addr_list,
                                       repeat_cnt1,
                                       dst_stride, src_stride)
                    # move ub_2_ub to padding c
                    tik_inst.data_move(
                        in_ub[hwc_ub_size * 2],
                        in_ub[hwc_ub_size],
                        0, col_size // axis_c, axis_c,
                        0, _ceil_div(axis_c, C0_LEN) * C0_LEN - axis_c)

                    repeat_cnt2 = 1
                    src_stride2 = 0
                    dst_stride2 = 0
                    c_cnt = col_size // axis_c
                    c0_cnt = _ceil_div(axis_c, C0_LEN)

                    with tik_inst.for_range(0, c0_cnt) as c0_index:
                        with tik_inst.for_range(0, c_cnt) as c_index:
                            src_tmp_list = [
                                in_ub[hwc_ub_size * 2 +
                                      (c0_index + c_index * c0_cnt) *
                                      ni_c0_size + C0_LEN * i]
                                for i in idx_list]
                            dst_tmp_list = [out_ub[ni_c0_size * i]
                                            for i in idx_list]

                            # mv 16 c0 to make 16 C0_LEN * C0_LEN
                            tik_inst.vnchwconv(False, False,
                                               dst_tmp_list,
                                               src_tmp_list,
                                               repeat_cnt2,
                                               dst_stride2, src_stride2)

                            tik_inst.data_move(
                                data_out[(c_index + c0_index * hw_size +
                                          hwc_index * C0_LEN +
                                          block_idx * hw_len +
                                          d_idx * c0_cnt * hw_size) *
                                         ni_c0_size + out_offset],
                                out_ub,
                                0, c0_line, C0_LEN,
                                0, (c_cnt - 1) * C0_LEN)

                c_count_in_col = \
                    hwc_col_size // (_ceil_div(axis_c, C0_LEN) * C0_LEN)
                actual_c_count_in_col = hw_len // C0_LEN
                c_count_left = hw_len % C0_LEN
                idx_list = [0, 1, 2, 3, 4, 5, 6, 7,
                            8, 9, 10, 11, 12, 13, 14, 15]

                if actual_c_count_in_col:
                    hwc_ub_loop = actual_c_count_in_col // c_count_in_col
                    hwc_ub_left = actual_c_count_in_col % c_count_in_col

                    if hwc_ub_loop:
                        with tik_inst.for_range(0, hwc_ub_loop) as hwc_ub_idx:
                            _dhwcn_2_3d_hw_hwc(
                                hwc_ub_idx * c_count_in_col,
                                c_count_in_col * axis_c, C0_LEN)

                    if hwc_ub_left:
                        _dhwcn_2_3d_hw_hwc(
                            hwc_ub_loop * c_count_in_col,
                            hwc_ub_left * axis_c, C0_LEN)

                if c_count_left:
                    _dhwcn_2_3d_hw_hwc(
                        actual_c_count_in_col,
                        axis_c, c_count_left)

            def _inner_process_c(hw_lp_index):
                """
                real transfer process for multiple core on axis h, w with n==1
                and c > in_ub_size
                """

                def _dhwcn_2_3d_hw_c(c_ub_index, col_size):
                    """
                    dhwcn transfer to 3d by c with c > in_ub_size
                    """

                    tik_inst.data_move(
                        in_ub,
                        data_in[c_ub_index * in_ub_size +
                                (block_idx + hw_lp_index * CORE_NUM +
                                 d_idx * hw_size) * axis_c],
                        0, 1, _ceil_div(col_size, C0_LEN), 0, 0)

                    c0_cnt = col_size // C0_LEN
                    col_left = col_size % C0_LEN
                    if c0_cnt:
                        c0_loop = c0_cnt // C0_LEN
                        c0_loop_left = c0_cnt % C0_LEN
                        with tik_inst.for_range(0, c0_loop) as c0_lp_idx:
                            tik_inst.data_move(
                                out_ub,
                                in_ub[c0_lp_idx * ni_c0_size],
                                0, C0_LEN, 1, 0, 15)

                            tik_inst.data_move(
                                data_out[(block_idx + hw_lp_index * CORE_NUM +
                                          (c0_lp_idx * C0_LEN + c_ub_index *
                                           in_ub_size // C0_LEN +
                                           d_idx * _ceil_div(axis_c, C0_LEN)) *
                                          hw_size) * ni_c0_size],
                                out_ub,
                                0, C0_LEN, C0_LEN, 0, (hw_size - 1) * C0_LEN)

                        if c0_loop_left:
                            tik_inst.data_move(
                                out_ub,
                                in_ub[c0_loop * ni_c0_size],
                                0, c0_loop_left, 1, 0, 15)

                            tik_inst.data_move(
                                data_out[(block_idx + hw_lp_index * CORE_NUM +
                                          (c0_loop * C0_LEN + c_ub_index *
                                           in_ub_size // C0_LEN +
                                           d_idx * _ceil_div(axis_c, C0_LEN)) *
                                          hw_size) * ni_c0_size],
                                out_ub,
                                0, c0_loop_left, C0_LEN,
                                0, (hw_size - 1) * C0_LEN)

                    if col_left:
                        # the left cnt is less than one block,
                        # so clean one block
                        _clean_ubuf(tik_inst, out_ub, 0, C0_LEN)
                        for idx in c_list:
                            reg_list[idx].set_as(in_ub[c0_cnt * C0_LEN + idx])
                        for idx in c_list:
                            out_ub[idx].set_as(reg_list[idx])

                        tik_inst.data_move(
                            data_out[(block_idx + hw_lp_index * CORE_NUM +
                                      (c0_cnt + c_ub_index *
                                       in_ub_size // C0_LEN +
                                       d_idx * _ceil_div(axis_c, C0_LEN)) *
                                      hw_size) * ni_c0_size],
                            out_ub,
                            0, 1, C0_LEN, 0, (hw_size - 1) * C0_LEN)

                c_ub_loop = axis_c // in_ub_size
                c_ub_left = axis_c % in_ub_size

                with tik_inst.for_range(0, c_ub_loop) as c_ub_idx:
                    _dhwcn_2_3d_hw_c(c_ub_idx, in_ub_size)

                if c_ub_left:
                    _dhwcn_2_3d_hw_c(c_ub_loop, c_ub_left)

            if axis_n > 1:
                if multi_hw_loop_cnt:
                    with tik_inst.for_range(0, multi_hw_loop_cnt) as hw_lp_idx:
                        _inner_process_hw(hw_lp_idx)

                if multi_hw_left_cnt:
                    with tik_inst.if_scope(block_idx < multi_hw_left_cnt):
                        _inner_process_hw(multi_hw_loop_cnt)
            else:
                col_threshold = 128
                if axis_c > col_threshold or multi_hw_loop_cnt == 1:
                    if multi_hw_loop_cnt:
                        with tik_inst.for_range(
                                0, multi_hw_loop_cnt) as hw_lp_idx:
                            _inner_process_c(hw_lp_idx)

                    if multi_hw_left_cnt:
                        with tik_inst.if_scope(block_idx < multi_hw_left_cnt):
                            _inner_process_c(multi_hw_loop_cnt)

                else:
                    # to split the in_ub_size into 3 parts and align with 16
                    hwc_col_size = in_ub_size // 3 // C0_LEN // C0_LEN * C0_LEN
                    hwc_ub_size = hwc_col_size * C0_LEN

                    # clean dst address of ub_to_ub one time
                    with tik_inst.if_scope(d_idx == 0):
                        with tik_inst.if_scope(axis_c % C0_LEN):
                            _clean_ubuf(tik_inst, in_ub,
                                        hwc_ub_size * 2, hwc_ub_size)

                    # process multi_hw_loop_cnt * axis_c each core
                    if multi_hw_loop_cnt:
                        _inner_process_hwc(multi_hw_loop_cnt, 0, 0)

                    # the left process axis_c each core
                    if multi_hw_left_cnt:
                        with tik_inst.if_scope(block_idx < multi_hw_left_cnt):
                            _inner_process_hwc(
                                1,
                                multi_hw_loop_cnt * CORE_NUM * axis_c,
                                multi_hw_loop_cnt * CORE_NUM * ni_c0_size)


def _multi_core_on_d(tik_inst, data_in, data_out, shape_in):
    """
    process of multiple core on axis d
    """

    axis_d, axis_h, axis_w, axis_c, axis_n = shape_in

    multi_d_loop_cnt = axis_d // CORE_NUM
    multi_d_left_cnt = axis_d % CORE_NUM
    hw_size = axis_h * axis_w
    idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    vnchw_ub_size = _get_vnchwconv_ub_size(axis_n, data_in.dtype)
    # vnchwconv process 16 lines each time
    vnchw_col_size = vnchw_ub_size // C0_LEN
    in_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(data_in.dtype, (vnchw_ub_size,),
                             name="out_ub", scope=tik.scope_ubuf)

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:

        def _inner_process_d(d_lp_index):
            """
            real transfer process for multiple core on axis h, w
            """

            c_loop_cnt = axis_c // C0_LEN
            c_left = axis_c % C0_LEN

            with tik_inst.for_range(0, hw_size) as hw_idx:

                def _dhwcn_2_3d_d(c_lp_index, c_cnt):
                    """
                    do transfer from dhwcn to 3d by vnchwconv
                    """

                    def _vnchwconv_process_d(n_lp_index, col_size):
                        """
                        vnchwconv transfer process
                        """

                        if axis_n == vnchw_col_size:
                            tik_inst.data_move(
                                in_ub,
                                data_in[n_lp_index * vnchw_col_size +
                                        c_lp_index * C0_LEN * axis_n +
                                        hw_idx * axis_c * axis_n +
                                        (block_idx + d_lp_index * CORE_NUM) *
                                        hw_size * axis_c * axis_n],
                                0, 1,
                                c_cnt * _ceil_div(col_size, C0_LEN), 0, 0)
                        else:
                            with tik_inst.for_range(0, c_cnt) as c_index:
                                tik_inst.data_move(
                                    in_ub[c_index * vnchw_col_size],
                                    data_in[n_lp_index * vnchw_col_size +
                                            c_index * axis_n +
                                            c_lp_index * C0_LEN * axis_n +
                                            hw_idx * axis_c * axis_n +
                                            (block_idx + d_lp_index * CORE_NUM)
                                            * hw_size * axis_c * axis_n],
                                    0, 1, _ceil_div(col_size, C0_LEN), 0, 0)

                        src_addr_list = [in_ub[vnchw_col_size * i]
                                         for i in idx_list]
                        dst_addr_list = [out_ub[C0_LEN * i] for i in idx_list]
                        repeat_cnt = _ceil_div(col_size, C0_LEN)
                        src_stride = 0 if repeat_cnt == 1 else 1
                        dst_stride = 0 if repeat_cnt == 1 else 16

                        tik_inst.vnchwconv(False, False,
                                           dst_addr_list,
                                           src_addr_list,
                                           repeat_cnt,
                                           dst_stride, src_stride)
                        # set n left to zero
                        if col_size % C0_LEN:
                            _clean_ubuf(
                                tik_inst, out_ub,
                                col_size * C0_LEN,
                                (C0_LEN - col_size % C0_LEN) * C0_LEN)

                        ni_c0_size = C0_LEN * C0_LEN
                        no_ni_c0_size = _ceil_div(axis_n, C0_LEN) * ni_c0_size
                        c1_cnt = _ceil_div(axis_c, C0_LEN)
                        c1hw_size = c1_cnt * hw_size
                        tik_inst.data_move(
                            data_out[n_lp_index *
                                     _ceil_div(vnchw_col_size, C0_LEN) *
                                     ni_c0_size +
                                     (c_lp_index * hw_size + hw_idx +
                                      (d_lp_index * CORE_NUM + block_idx) *
                                      c1hw_size) * no_ni_c0_size],
                            out_ub,
                            0, 1, repeat_cnt * C0_LEN, 0, 0)

                    if axis_n > vnchw_col_size:
                        n_loop = axis_n // vnchw_col_size
                        n_left = axis_n % vnchw_col_size

                        if n_loop:
                            with tik_inst.for_range(0, n_loop) as n_lp_idx:
                                _vnchwconv_process_d(n_lp_idx,
                                                     vnchw_col_size)

                        if n_left:
                            _vnchwconv_process_d(n_loop, n_left)

                    else:
                        _vnchwconv_process_d(0, axis_n)

                if c_loop_cnt:
                    with tik_inst.for_range(0, c_loop_cnt) as c_lp_idx:
                        _dhwcn_2_3d_d(c_lp_idx, C0_LEN)

                if c_left:
                    # set the lines will not be used to zero
                    _clean_ubuf(tik_inst, in_ub, c_left * vnchw_col_size,
                                (C0_LEN - c_left) * vnchw_col_size)
                    _dhwcn_2_3d_d(c_loop_cnt, c_left)

        if multi_d_loop_cnt:
            with tik_inst.for_range(0, multi_d_loop_cnt) as d_lp_idx:
                _inner_process_d(d_lp_idx)

        if multi_d_left_cnt:
            with tik_inst.if_scope(block_idx < multi_d_left_cnt):
                _inner_process_d(multi_d_loop_cnt)


def dhwcn_2_fractal_z_3d_compute(tik_inst, data_in, data_out, shape_in):
    """
    do dhwcn to fractal_z_3d transfer
    """

    axis_d, _, _, axis_c, axis_n = shape_in
    if axis_c <= C0_LEN and axis_n == 1:
        _multi_core_on_dhw(tik_inst, data_in, data_out, shape_in)
    elif axis_c // C0_LEN // CORE_NUM > 0 and axis_n > 1:
        _multi_core_on_c(tik_inst, data_in, data_out, shape_in)
    elif axis_d // CORE_NUM > 0 and axis_n > 1:
        _multi_core_on_d(tik_inst, data_in, data_out, shape_in)
    else:
        _multi_core_on_hw(tik_inst, data_in, data_out, shape_in)


def _set_core_num(origin_num):
    """
    function of set core num
    """
    if origin_num < CORE_NUM:
        return origin_num
    return CORE_NUM


def _set_loop(tik_instance, num_core, max_core, total_dim):
    """
    function of set loop
    """
    core_loop = tik_instance.Scalar("uint64")

    with tik_instance.if_scope(num_core < total_dim % CORE_NUM):
        core_loop.set_as(_ceil_div(total_dim, max_core))
    with tik_instance.else_scope():
        core_loop.set_as(total_dim // max_core)

    return core_loop


# pylint: disable=locally-disabled,too-many-instance-attributes
# pylint: disable=locally-disabled,old-style-class
class Dhwcn2Fz3dFp32Compute:
    """
    Rearranges data from DHWCN format to FRACTAL_Z_3D format fp32 scene

    Returns
    -------
    None
    """
    def __init__(self, src_shape, dst_shape, dtype, kernel_name):
        """
        initialize some properties
        """
        self.src_shape = list(src_shape)
        self.dst_shape = list(dst_shape)
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.float_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
        self.cp_align_len = cce_params.BLOCK_REDUCE_INT8 // self.float_size
        self.ub_ele = ((UB_SIZE - 64) // self.float_size // 2
                       // 256) * 256
        self.c_0 = self.dst_shape[3]
        self.n_i = self.dst_shape[2]
        self.c_1 = self.calc_c1()
        self.src_gm = None
        self.dst_gm = None

    def func_c0na_core(self, args):
        """
        function of moving data for c0na_core scene
        """
        tik_instance, ub_ori, ub_trans, d_index, \
        c1_index, hw_index, c_before, c_now = args

        _, h_d, w_d, c_d, n_d = self.src_shape
        hw_d = h_d * w_d
        n_o = self.dst_shape[1]

        in_ele = c_now * n_d
        src_offset = (d_index * hw_d + hw_index) * c_d * n_d + c_before * n_d
        burst_len = _ceil_div(in_ele, self.cp_align_len)
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        dim_ele = in_ele * 2
        dim_zu = _ceil_div(dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        _clean_ubuf(tik_instance, ub_ori, 0, self.ub_ele * 2)

        with tik_instance.for_range(0, c_now) as num_c:
            src_offset = num_c * n_d * 2 * 16
            dst_offset = num_c * 2 * 16
            n_burst = n_d
            burst_len = 2
            src_stride = 0
            dst_stride = (self.c_0 - 1) * 2
            tik_instance.data_move(ub_ori[dst_offset],
                                   ub_trans[src_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        mid_ele = self.c_0 * n_d * 2
        mid_zu = _ceil_div(mid_ele, 16)

        if mid_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, mid_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   mid_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        if n_d < n_o * self.n_i:
            dup_len = (n_o * self.n_i - n_d) * self.c_0 * 2
            dup_offset = n_d * self.c_0 * 2
            _clean_ubuf(tik_instance, ub_trans, dup_offset, dup_len)

        dst_offset = (d_index * self.c_1 * hw_d + c1_index * hw_d + hw_index)\
                     * n_o * self.n_i * self.c_0
        burst_len = n_o * self.n_i * self.c_0 // self.cp_align_len
        tik_instance.data_move(self.dst_gm[dst_offset],
                               ub_trans,
                               0, 1, burst_len, 0, 0)

    def c0na_core(self, tik_instance):
        """
        n_o * n_i * c_0 <= ub_ele
        """
        d_d, h_d, w_d, c_d, _ = self.src_shape
        hw_d = h_d * w_d

        all_core = d_d * self.c_1 * hw_d
        ac_num = _set_core_num(all_core)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor("float16",
                                         (self.ub_ele * 2,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (self.ub_ele * 2,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                c1hw_d = self.c_1 * hw_d
                d_index = core_index // c1hw_d
                c1hw_index = core_index % c1hw_d
                c1_index = c1hw_index // hw_d
                hw_index = c1hw_index % hw_d

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c_before = c1_index * self.c_0
                    c_now = self.c_0
                    args = tik_instance, ub_ori, ub_trans, d_index, \
                           c1_index, hw_index, c_before, c_now
                    self.func_c0na_core(args)

                with tik_instance.else_scope():
                    c_before = (self.c_1 - 1) * self.c_0
                    c_now = c_d - c_before
                    args = tik_instance, ub_ori, ub_trans, d_index, \
                           c1_index, hw_index, c_before, c_now
                    self.func_c0na_core(args)

        return tik_instance

    def func_c0n_core(self, args):
        """
        function of moving data for c0n_core scene
        """
        tik_instance, ub_ori, ub_trans, d_index, \
        c1_index, hw_index, c_before, c_now = args

        _, h_d, w_d, c_d, n_d = self.src_shape
        hw_d = h_d * w_d
        n_o = self.dst_shape[1]

        in_ele = c_now * n_d
        src_offset = (d_index * hw_d + hw_index) * c_d * n_d + c_before * n_d
        burst_len = _ceil_div(in_ele, self.cp_align_len)
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        dim_ele = in_ele * 2
        dim_zu = _ceil_div(dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        _clean_ubuf(tik_instance, ub_ori, 0, self.ub_ele * 2)

        with tik_instance.for_range(0, c_now) as num_c:
            src_offset = num_c * n_d * 2 * 16
            dst_offset = num_c * 2 * 16
            n_burst = n_d
            burst_len = 2
            src_stride = 0
            dst_stride = (self.c_0 - 1) * 2
            tik_instance.data_move(ub_ori[dst_offset],
                                   ub_trans[src_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        mid_ele = self.c_0 * n_d * 2
        mid_zu = _ceil_div(mid_ele, 16)

        if mid_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, mid_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   mid_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        dst_offset = (d_index * self.c_1 * hw_d + c1_index * hw_d + hw_index)\
                     * n_o * self.n_i * self.c_0
        burst_len = n_d * self.c_0 // self.cp_align_len
        tik_instance.data_move(self.dst_gm[dst_offset],
                               ub_trans,
                               0, 1, burst_len, 0, 0)

        if n_d < n_o * self.n_i:
            zero_row = n_o * self.n_i - n_d
            zero_ele = zero_row * self.c_0
            _clean_ubuf(tik_instance, ub_ori, 0, zero_ele * 2)

            dst_offset = (d_index * self.c_1 * hw_d
                          + c1_index * hw_d + hw_index)\
                         * n_o * self.n_i * self.c_0 + n_d * self.c_0
            burst_len = zero_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_ori,
                                   0, 1, burst_len, 0, 0)

    def c0n_core(self, tik_instance):
        """
        n_o * n_i * c_0 <= ub_ele
        """
        d_d, h_d, w_d, c_d, _ = self.src_shape
        hw_d = h_d * w_d

        all_core = d_d * self.c_1 * hw_d
        ac_num = _set_core_num(all_core)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor("float16",
                                         (self.ub_ele * 2,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (self.ub_ele * 2,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                c1hw_d = self.c_1 * hw_d
                d_index = core_index // c1hw_d
                c1hw_index = core_index % c1hw_d
                c1_index = c1hw_index // hw_d
                hw_index = c1hw_index % hw_d

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c_before = c1_index * self.c_0
                    c_now = self.c_0
                    args = tik_instance, ub_ori, ub_trans, d_index, \
                           c1_index, hw_index, c_before, c_now
                    self.func_c0n_core(args)

                with tik_instance.else_scope():
                    c_before = (self.c_1 - 1) * self.c_0
                    c_now = c_d - c_before
                    args = tik_instance, ub_ori, ub_trans, d_index, \
                           c1_index, hw_index, c_before, c_now
                    self.func_c0n_core(args)

        return tik_instance

    def func_split_n(self, args):
        """
        function of moving data for split_n scene
        """
        tik_instance, ub_ori, ub_trans, \
        d_index, c1_index, hw_index, c_before, c_now, \
        n_before, n_now = args

        _, h_d, w_d, c_d, n_d = self.src_shape
        hw_d = h_d * w_d

        with tik_instance.for_range(0, c_now) as num_c:
            src_offset = (d_index * hw_d + hw_index) * c_d * n_d \
                         + ((c_before + num_c) * n_d) + n_before
            ub_offset = num_c * n_now * 2
            burst_len = n_now // self.cp_align_len
            tik_instance.data_move(ub_ori[ub_offset],
                                   self.src_gm[src_offset],
                                   0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        in_ele = c_now * n_now
        dim_ele = in_ele * 2
        dim_zu = _ceil_div(dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        _clean_ubuf(tik_instance, ub_ori, 0, self.ub_ele * 2)

        with tik_instance.for_range(0, c_now) as num_c:
            src_offset = num_c * n_now * 2 * 16
            dst_offset = num_c * 2 * 16
            n_burst = n_now
            burst_len = 2
            src_stride = 0
            dst_stride = (self.c_0 - 1) * 2
            tik_instance.data_move(ub_ori[dst_offset],
                                   ub_trans[src_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        mid_ele = self.c_0 * n_now * 2
        mid_zu = _ceil_div(mid_ele, 16)

        if mid_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, mid_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   mid_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        n_o = self.dst_shape[1]
        dst_offset = (d_index * self.c_1 * hw_d + c1_index * hw_d + hw_index) \
                     * n_o * self.n_i * self.c_0 + n_before * self.c_0
        burst_len = n_now * self.c_0 // self.cp_align_len
        tik_instance.data_move(self.dst_gm[dst_offset],
                               ub_trans,
                               0, 1, burst_len, 0, 0)

        if n_d < n_o * self.n_i:
            with tik_instance.if_scope(n_before * n_now >= n_d):
                zero_row = n_o * self.n_i - n_d
                zero_ele = zero_row * self.c_0
                _clean_ubuf(tik_instance, ub_ori, 0, zero_ele * 2)

                dst_offset = (d_index * self.c_1 * hw_d
                              + c1_index * hw_d + hw_index) \
                             * n_o * self.n_i * self.c_0 + n_d * self.c_0
                burst_len = zero_ele // self.cp_align_len
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_ori,
                                       0, 1, burst_len, 0, 0)

    def split_n(self, tik_instance):
        """
        n_o * n_i * c_0 > ub_ele
        """
        d_d, h_d, w_d, c_d, n_d = self.src_shape
        hw_d = h_d * w_d
        n_ub = self.ub_ele // 16 // self.c_0\
               // self.cp_align_len * self.cp_align_len
        n_zu = _ceil_div(n_d, n_ub)

        all_core = d_d * self.c_1 * hw_d * n_zu
        ac_num = _set_core_num(all_core)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor("float16",
                                         (self.ub_ele * 2,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (self.ub_ele * 2,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                c1hwn_d = self.c_1 * hw_d * n_zu
                d_index = core_index // c1hwn_d
                c1hwn_index = core_index % c1hwn_d
                hwn_d = hw_d * n_zu
                c1_index = c1hwn_index // hwn_d
                hwn_index = c1hwn_index % hwn_d
                hw_index = hwn_index // n_zu
                nzu_index = hwn_index % n_zu

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c_before = c1_index * self.c_0
                    c_now = self.c_0

                    with tik_instance.if_scope(nzu_index < n_zu - 1):
                        n_before = nzu_index * n_ub
                        n_now = n_ub
                        args = tik_instance, ub_ori, ub_trans, \
                               d_index, c1_index, hw_index, c_before, c_now, \
                               n_before, n_now
                        self.func_split_n(args)

                    with tik_instance.else_scope():
                        n_now_temp = n_d - (n_zu - 1) * n_ub
                        n_now = _ceil_fill(n_now_temp, self.cp_align_len)
                        n_before = n_d - n_now
                        args = tik_instance, ub_ori, ub_trans, \
                               d_index, c1_index, hw_index, c_before, c_now, \
                               n_before, n_now
                        self.func_split_n(args)

                with tik_instance.else_scope():
                    c_before = (self.c_1 - 1) * self.c_0
                    c_now = c_d - c_before

                    with tik_instance.if_scope(nzu_index < n_zu - 1):
                        n_before = nzu_index * n_ub
                        n_now = n_ub
                        args = tik_instance, ub_ori, ub_trans, \
                               d_index, c1_index, hw_index, c_before, c_now, \
                               n_before, n_now
                        self.func_split_n(args)

                    with tik_instance.else_scope():
                        n_now_temp = n_d - (n_zu - 1) * n_ub
                        n_now = _ceil_fill(n_now_temp, self.cp_align_len)
                        n_before = n_d - n_now
                        args = tik_instance, ub_ori, ub_trans, \
                               d_index, c1_index, hw_index, c_before, c_now, \
                               n_before, n_now
                        self.func_split_n(args)

        return tik_instance

    def calc_c1(self):
        """
        function of calculating c_1
        """
        dc1hw_s = self.dst_shape[0]
        d_d, h_d, w_d, _, _ = self.src_shape
        c_1 = dc1hw_s // (d_d * h_d * w_d)
        return c_1

    def check_branch(self):
        """
        check which branch of dhwcn_2_fractal_z_3d fp32 compute
        """
        n_d = self.src_shape[4]
        n_o = self.dst_shape[1]

        c0na_ele = n_o * self.n_i * self.c_0 * 2 * 8
        c0n_ele = n_d * self.c_0 * 2 * 8

        if c0na_ele <= self.ub_ele:
            return "c0na_core"
        elif c0n_ele <= self.ub_ele:
            return "c0n_core"
        else:
            return "split_n"

    def dhwcn_2_fz3d_fp32_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        branch = self.check_branch()

        if branch == "c0na_core":
            tik_instance = self.c0na_core(tik_instance)
        elif branch == "c0n_core":
            tik_instance = self.c0n_core(tik_instance)
        elif branch == "split_n":
            tik_instance = self.split_n(tik_instance)

        return tik_instance

    def set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        src_element_number = functools_reduce(lambda x, y: x * y,
                                              self.src_shape[:])
        dst_element_number = functools_reduce(lambda x, y: x * y,
                                              self.dst_shape[:])
        self.src_gm = tik_instance.Tensor(self.dtype,
                                          (src_element_number,),
                                          name="src_gm",
                                          scope=tik.scope_gm)
        self.dst_gm = tik_instance.Tensor(self.dtype,
                                          (dst_element_number,),
                                          name="dst_gm",
                                          scope=tik.scope_gm)

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_instance = tik.Tik()
        self.set_src_dst_tensor(tik_instance)

        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.dhwcn_2_fz3d_fp32_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.src_gm],
                              outputs=[self.dst_gm])

        return tik_instance


def _error_log(param, value, reason):
    error_info = {
        'ErrCode': 'E10001',
        'parameter': param,
        'value': value,
        'reason': reason
    }
    raise RuntimeError(error_info,
                       "Invalid value for {parameter}[{value}], "
                       "{reason}.".format(**error_info))


def _check_parameters_fp32(src, dst, src_format, dst_format):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "dhwcn":
        reason = "src_format must be DHWCN !"
        _error_log("src_format", src_format, reason)

    if dst_format.lower() != "fractal_z_3d":
        reason = "dst_format must be FRACTAL_Z_3D !"
        _error_log("dst_format", dst_format, reason)

    check_list = ("float32",)
    check_dtype(dtype, check_list)
    if dtype != dtype_dst:
        reason = "dtype of src and dst are different !"
        _error_log("dst_dtype", dtype_dst, reason)

    check_shape(src_shape, min_rank=5, max_rank=5)
    check_shape(dst_shape, min_rank=4, max_rank=4)

    if dst_shape[2] != 16:
        reason = "the 3rd dimension of dst_shape is not 16, Ni must be 16 !"
        _error_log("Ni", dst_shape[2], reason)

    if dst_shape[3] != 16:
        reason = "the 4th dimension of dst_shape is not 16, C0 must be 16 !"
        _error_log("C0", dst_shape[3], reason)

    d_d, h_d, w_d, c_d, n_d = src_shape

    n_i = 16
    n_s = n_i - 1
    n_o = (n_d + n_s) // n_i

    if dst_shape[1] != n_o:
        reason = "the 2nd dimension of dst_shape is wrong, " \
                 "No must be (N + 15)//16 !"
        _error_log("No", dst_shape[1], reason)

    c_0 = 16
    c_s = c_0 - 1
    c_1 = (c_d + c_s) // c_0
    one_dim = d_d * c_1 * h_d * w_d

    if dst_shape[0] != one_dim:
        reason = "the 1st dimension of dst_shape is wrong, " \
                 "it must be D*C1*H*W !"
        _error_log("DC1HW", dst_shape[0], reason)


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR,
                 REQUIRED_ATTR_STR, KERNEL_NAME)
def dhwcn_2_fractal_z_3d(src, dst, src_format, dst_format,
                         kernel_name="dhwcn_2_fractal_z_3d"):
    """
    used to transfer dhwcn to fractal_z_3d

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        shape and dtype of output, should be same shape and type as input
    src_format : str
        input format, the value should be "DHWCN"
    dst_format : str
         output format, the value should be "FRACTAL_Z_3D"
    kernel_name : str
        kernel name, default value is "dhwcn_2_fractal_z_3d"

    Returns
    -------
    None
    """
    dtype = src.get("dtype").lower()

    if dtype == "float32":
        _check_parameters_fp32(src, dst, src_format, dst_format)
        src_shape = src.get("shape")
        dst_shape = dst.get("shape")
        dtype = src.get("dtype").lower()

        template_fp32 = Dhwcn2Fz3dFp32Compute(src_shape, dst_shape, dtype,
                                              kernel_name)
        return template_fp32.get_tik_instance()

    else:
        shape_in = src.get("shape")
        dtype = src.get("dtype")
        input_dtype = dtype.lower()
        dst_dtype = dst.get("dtype").lower()

        check_list = ("float16",)
        check_dtype(input_dtype, check_list)
        check_shape(shape_in, min_rank=5, max_rank=5)
        shape_out = (shape_in[0], _ceil_div(shape_in[3], C0_LEN),
                     shape_in[1] * shape_in[2],
                     _ceil_div(shape_in[4], C0_LEN),
                     C0_LEN, C0_LEN)
        check_shape(shape_out)

        if input_dtype != dst_dtype:
            raise RuntimeError("The input and output dtype should be same!")

        if src_format.upper() != "DHWCN" or dst_format.upper() != "FRACTAL_Z_3D":
            raise RuntimeError("The src_format must be DHWCN and"
                               " dst_format must be FRACTAL_Z_3D!")

        # initial Tik
        tik_inst = tik.Tik()
        # define input and output tensors
        data_in = tik_inst.Tensor(input_dtype, shape_in,
                                  tik.scope_gm, "data_in")
        data_out = tik_inst.Tensor(input_dtype, shape_out,
                                   tik.scope_gm, "data_out")

        # do transfer
        dhwcn_2_fractal_z_3d_compute(tik_inst, data_in, data_out, shape_in)

        # build cce
        tik_inst.BuildCCE(kernel_name=kernel_name,
                          inputs=[data_in], outputs=[data_out])
