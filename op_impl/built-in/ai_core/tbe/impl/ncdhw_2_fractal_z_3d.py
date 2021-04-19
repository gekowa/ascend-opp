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
ncdhw_2_fractal_z_3d
"""
from functools import reduce as func_reduce
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *


# UB size in byte
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# AICORE count
CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# C0 length
C0_LEN = 16
# bytes in one block
BLOCK_BYTE_SIZE = 32
# repeat up limit for vector
REPEAT_LIMIT = 255
# mask value for float32
MASK_64 = 64
# float16/32 type list
TYPE_FLOAT_LIST = ("float16", "float32")
# used for reg
REG_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7)
# used for vnchwconv
ADDR_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def _ceil_div(value_x, value_y):
    """
    do ceil division
    """
    return (value_x + value_y - 1) // value_y


def _get_dtype_factor(dtype):
    """
    get dtype length in byte
    """

    if dtype.lower() == "float16":
        dtype_factor = 2
    elif dtype.lower() in ("float32", "int32"):
        dtype_factor = 4
    else:
        dtype_factor = 1

    return dtype_factor


def _clean_ubuf(tik_inst, src, src_offset, dup_len):
    """
    clean ubuf to zero
    """

    if src.dtype.lower() == "float16":
        dtype_factor = 2
    elif src.dtype.lower() == "float32":
        dtype_factor = 1
    batch_size = MASK_64

    if dup_len > 0:
        repeat = dup_len // (batch_size * dtype_factor)
        left_elem = dup_len % (batch_size * dtype_factor)
        repeat_loop = repeat // REPEAT_LIMIT
        repeat_left = repeat % REPEAT_LIMIT
        dup_value = float(0)

        if repeat_loop > 0:
            with tik_inst.for_range(0, repeat_loop) as rpt_idx:
                tik_inst.vector_dup(MASK_64 * dtype_factor,
                                    src[src_offset + rpt_idx *
                                        REPEAT_LIMIT *
                                        batch_size * dtype_factor],
                                    dup_value, REPEAT_LIMIT, 1, 8)

        if repeat_left > 0:
            tik_inst.vector_dup(MASK_64 * dtype_factor,
                                src[src_offset + repeat_loop *
                                    REPEAT_LIMIT *
                                    batch_size * dtype_factor],
                                dup_value, repeat_left, 1, 8)

        if left_elem > 0:
            tik_inst.vector_dup(left_elem,
                                src[src_offset + repeat *
                                    batch_size * dtype_factor],
                                dup_value, 1, 1, 8)


def _get_vnchwconv_ub_size(cube_size, parts):
    """
    count needed ub size of the vnchwconv command
    """

    is_parts_cube_less_ub = parts * cube_size <= UB_SIZE
    if is_parts_cube_less_ub:
        allocated_ub = cube_size
    else:
        allocated_ub = UB_SIZE // parts

    # the max repeat time of vnchwconv is 255, for simple deduct 1024 from UB_SIZE
    if allocated_ub > (248 * 1024) // 2:
        allocated_ub = (248 * 1024) // 2

    return allocated_ub


# pylint: disable=too-many-arguments,too-many-locals
def _scalar_conv(tik_inst, dst_ub, src_ub, axis_params, reg_list):
    """
    do cdhw to c1dhwc0 transfer by scalar
    """

    sub_n_len, c_len, dhw_len, dhw_pad_len, c0_len = axis_params
    reg_cnt = len(reg_list)
    c1_cnt = c_len // c0_len
    c_left = c_len % c0_len

    def _c_bigger_eight_process(sub_n_index, c1_index, c0_size):
        """
        the convert process for c count bigger than reg count
        """

        reg_loop = c0_size // reg_cnt
        c0_left = c0_size % reg_cnt
        with tik_inst.for_range(0, dhw_len) as dhw_idx:
            if reg_loop:
                with tik_inst.for_range(0, reg_loop) as reg_lp_idx:
                    for idx in REG_IDX_LIST:
                        src_offset = (sub_n_index * c_len + c1_index * c0_len +
                                      reg_lp_idx * reg_cnt + idx) * dhw_pad_len + dhw_idx
                        reg_list[idx].set_as(src_ub[src_offset])

                    for idx in REG_IDX_LIST:
                        dst_offset = ((c1_index * dhw_len + dhw_idx) * sub_n_len +
                                      sub_n_index) * c0_len + reg_lp_idx * reg_cnt + idx
                        dst_ub[dst_offset].set_as(reg_list[idx])

            if c0_left:
                for idx in REG_IDX_LIST[:c0_left]:
                    src_offset = (sub_n_index * c_len + c1_index * c0_len +
                                  reg_loop * reg_cnt + idx) * dhw_pad_len + dhw_idx
                    reg_list[idx].set_as(src_ub[src_offset])

                for idx in REG_IDX_LIST[:c0_left]:
                    dst_offset = ((c1_index * dhw_len + dhw_idx) * sub_n_len +
                                  sub_n_index) * c0_len + reg_loop * reg_cnt + idx
                    dst_ub[dst_offset].set_as(reg_list[idx])

    def _c_less_equal_eight_process(sub_n_index, c1_count, c0_size):
        """
        the convert process for c count less than equal to reg count
        """

        reg_factor = reg_cnt // c0_size
        reg_cnt_new = reg_factor * c0_size
        dhw_len_new = dhw_len // reg_factor
        dhw_len_left = dhw_len % reg_factor
        with tik_inst.for_range(0, dhw_len_new) as dhw_idx:
            for idx in REG_IDX_LIST[:reg_cnt_new]:
                src_offset = ((idx % c0_size + c1_count * 16 + sub_n_index * c_len) * dhw_pad_len +
                              dhw_idx * reg_factor + idx // c0_size)
                reg_list[idx].set_as(src_ub[src_offset])

            for idx in REG_IDX_LIST[:reg_cnt_new]:
                dst_offset = ((dhw_idx * reg_factor + c1_count * dhw_len) * sub_n_len +
                              sub_n_index + idx // c0_size) * c0_len + idx % c0_size
                dst_ub[dst_offset].set_as(reg_list[idx])

        if dhw_len_left:
            with tik_inst.for_range(0, dhw_len_left) as dhw_left_idx:
                for idx in REG_IDX_LIST[:c0_size]:
                    src_offset = ((idx + c1_count * 16 + sub_n_index * c_len) * dhw_pad_len +
                                  dhw_len_new * reg_factor + dhw_left_idx)
                    reg_list[idx].set_as(src_ub[src_offset])

                for idx in REG_IDX_LIST[:reg_cnt_new]:
                    dst_offset = ((dhw_len_new * reg_factor + c1_count * dhw_len) * sub_n_len +
                                  dhw_left_idx + sub_n_index) * c0_len + idx
                    dst_ub[dst_offset].set_as(reg_list[idx])

    with tik_inst.for_range(0, sub_n_len) as sub_n_idx:
        if c_len > reg_cnt:
            if c1_cnt:
                with tik_inst.for_range(0, c1_cnt) as c1_idx:
                    _c_bigger_eight_process(sub_n_idx, c1_idx, c0_len)

            if 0 < c_left <= reg_cnt:
                _c_less_equal_eight_process(sub_n_idx, c1_cnt, c_left)
            elif c_left > reg_cnt:
                _c_bigger_eight_process(sub_n_idx, c1_cnt, c_left)
        else:
            _c_less_equal_eight_process(sub_n_idx, 0, c_len)


def _check_input_params(input_params):
    """
    to the check whether the input parameters is valid or not
    """

    in_shape, dst_shape, in_dtype, dst_dtype, src_format, dst_format = input_params
    check_list = TYPE_FLOAT_LIST

    if in_dtype != dst_dtype:
        raise RuntimeError("The input and output dtype should be same!")
    if not (src_format.upper() == "NCDHW" and
            dst_format.upper() == "FRACTAL_Z_3D"):
        raise RuntimeError("The src_format must be NCDHW and"
                           " dst_format must be FRACTAL_Z_3D!")

    check_dtype(in_dtype, check_list)
    check_shape(in_shape, min_rank=5, max_rank=5)
    check_shape(dst_shape)


# pylint: disable=too-many-locals,too-many-statements
def _multi_core_on_n(tik_inst, data_in, data_out, shape_in):
    """
    do ncdhw to fractal_z_3d transfer by multiple core on axis n
    """

    axis_n, axis_c, axis_d, axis_h, axis_w = shape_in
    hw_size = axis_h * axis_w
    ni_no_size = _ceil_div(axis_n, C0_LEN) * C0_LEN
    cdhw_size = func_reduce(lambda x, y: x * y, shape_in[1:])
    axis_c1 = _ceil_div(axis_c, C0_LEN)
    dtype_factor = _get_dtype_factor(data_in.dtype)

    # each core process certain cdhw lines
    core_num = _ceil_div(axis_n, _ceil_div(axis_n, CORE_NUM))
    per_core_n_cnt = _ceil_div(axis_n, core_num)
    left_n_cnt = axis_n - per_core_n_cnt * (core_num - 1)
    # to count the padding lines
    zero_loop_cnt = (_ceil_div(axis_n, C0_LEN) * C0_LEN - axis_n) // core_num
    zero_line_left = (_ceil_div(axis_n, C0_LEN) * C0_LEN - axis_n) % core_num

    # to check whether the half UB can hold cdhw or not
    dtype_factor = _get_dtype_factor(data_in.dtype)
    out_size = func_reduce(lambda x, y: x * y,
                           (axis_c1 * C0_LEN, axis_d, axis_h, axis_w, dtype_factor))
    # to avoid the repeat times of MTE bigger than 4095
    is_chwd_less_half_ub = out_size <= ((UB_SIZE - 4 * BLOCK_BYTE_SIZE) // 2)
    # used to scalar conv
    reg_list = [tik_inst.Scalar(data_in.dtype) for i in REG_IDX_LIST]

    if is_chwd_less_half_ub:
        # split the UB into two parts, and to load cdhw each time
        ub_size = (UB_SIZE - 4 * BLOCK_BYTE_SIZE) // 2 // dtype_factor
    else:
        # to adapt the vnchwconv command
        hw_align_c0_mul_16 = _ceil_div(hw_size, C0_LEN) * C0_LEN * 16
        # split the UB into two parts, and to make sure the ub_size is align with C0_LEN
        ub_size = _get_vnchwconv_ub_size(hw_align_c0_mul_16 * dtype_factor,
                                         2) // dtype_factor // C0_LEN * C0_LEN
        ub_col_size = ub_size // 16 // C0_LEN * C0_LEN
        if ub_col_size == 0:
            raise RuntimeError("The UB is too small!")

    # alloc input and output ub
    in_ub = tik_inst.Tensor(data_in.dtype, (ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(data_in.dtype, (ub_size,),
                             name="out_ub", scope=tik.scope_ubuf)

    with tik_inst.for_range(0, core_num, block_num=core_num) as block_idx:

        # pylint: disable=too-many-locals,too-many-statements
        def _n_transfer_process(n_len):
            """
            process of hw transfer
            """

            c0_count_in_c = axis_c // C0_LEN
            c_left = axis_c % C0_LEN

            if is_chwd_less_half_ub:
                dhw_align_size = _ceil_div(axis_d * hw_size, C0_LEN) * C0_LEN
                dhw_size = axis_d * hw_size

                def _get_mv_in_para():
                    """
                    to count how many cdhw can be loaded in one time
                    """

                    nc1dhwc0_cnt = ub_size // (axis_c1 * axis_d * hw_size * C0_LEN)
                    if n_len <= nc1dhwc0_cnt:
                        mv_in_lp = 1
                        len_new = n_len
                        len_left = 0
                    else:
                        mv_in_lp = n_len // nc1dhwc0_cnt
                        len_new = nc1dhwc0_cnt
                        len_left = n_len % nc1dhwc0_cnt

                    return mv_in_lp, len_left, len_new

                def _cdhw_less_half_ub_process(mv_in_lp_index, sub_n_len):
                    """
                    process of cdhw less than half UB
                    """

                    input_offset = (block_idx * per_core_n_cnt +
                                    mv_in_lp_index * n_len_new) * cdhw_size
                    # move in xcdhw size each time
                    tik_inst.data_move(
                        in_ub,
                        data_in[input_offset],
                        0, 1,
                        _ceil_div(sub_n_len * cdhw_size, BLOCK_BYTE_SIZE // dtype_factor),
                        0, 0)

                    if axis_c % C0_LEN:
                        # set the dst ub to zero to avoid dirty data
                        with tik_inst.if_scope(mv_in_lp_index == 0):
                            _clean_ubuf(tik_inst, out_ub, 0, ub_size)
                    # do transpose from xcdhw to c1dhwxc0
                    axis_param_1 = (sub_n_len, axis_c, axis_d * hw_size, axis_d * hw_size, C0_LEN)
                    _scalar_conv(tik_inst, out_ub, in_ub, axis_param_1, reg_list)

                    with tik_inst.for_range(0, axis_d) as d_idx:
                        with tik_inst.for_range(0, axis_c1) as c1_idx:
                            output_offset = (block_idx * per_core_n_cnt +
                                             mv_in_lp_index * n_len_new +
                                             (d_idx * axis_c1 + c1_idx) *
                                             hw_size * ni_no_size) * C0_LEN
                            mid_offset = (c1_idx * axis_d + d_idx) * hw_size * sub_n_len * C0_LEN
                            # move out hwc0 each time
                            tik_inst.data_move(data_out[output_offset],
                                               out_ub[mid_offset],
                                               0, hw_size, sub_n_len * dtype_factor // 2,
                                               0, (ni_no_size - sub_n_len) * dtype_factor // 2)

                def _cdhw_less_half_ub_process_fp16(n_index):
                    """
                    process of cdhw less than half UB for fp16
                    """

                    def _inner_process_fp16(c0_index, c_lines):
                        """
                        vnchwconv for c0dhw
                        """

                        with tik_inst.for_range(0, c_lines) as c_idx:
                            input_offset = ((block_idx * per_core_n_cnt + n_index) * cdhw_size +
                                            (c0_index * C0_LEN + c_idx) * dhw_size)
                            tik_inst.data_move(in_ub[c_idx * dhw_align_size],
                                               data_in[input_offset],
                                               0, 1, _ceil_div(dhw_size, C0_LEN), 0, 0)
                        # do vnchwconv
                        src_addr_list = [in_ub[dhw_align_size * i] for i in ADDR_IDX_LIST]
                        dst_addr_list = [out_ub[C0_LEN * i] for i in ADDR_IDX_LIST]
                        repeat_cnt = _ceil_div(dhw_size, C0_LEN)
                        src_stride = 0 if repeat_cnt == 1 else 1
                        dst_stride = 0 if repeat_cnt == 1 else 16
                        tik_inst.vnchwconv(False, False,
                                           dst_addr_list, src_addr_list,
                                           repeat_cnt, dst_stride, src_stride)

                        # move data out in d times
                        with tik_inst.for_range(0, axis_d) as d2_idx:
                            output_offset = (block_idx * per_core_n_cnt + n_index +
                                             (d2_idx * axis_c1 + c0_index) *
                                             hw_size * ni_no_size) * C0_LEN
                            tik_inst.data_move(data_out[output_offset],
                                               out_ub[d2_idx * hw_size * C0_LEN],
                                               0, hw_size, 1, 0, ni_no_size - 1)

                    if c0_count_in_c:
                        with tik_inst.for_range(0, c0_count_in_c) as c0_idx:
                            _inner_process_fp16(c0_idx, C0_LEN)
                        if c_left:
                            _clean_ubuf(tik_inst, in_ub,
                                        c_left * dhw_align_size, (16 - c_left) * dhw_align_size)
                            _inner_process_fp16(c0_count_in_c, c_left)
                    else:
                        with tik_inst.if_scope(n_index == 0):
                            _clean_ubuf(tik_inst, in_ub,
                                        c_left * dhw_align_size, (16 - c_left) * dhw_align_size)
                        _inner_process_fp16(0, c_left)

                if (data_in.dtype.lower() == "float16" and
                        dhw_align_size * C0_LEN <= ub_size and dhw_size >= C0_LEN):
                    with tik_inst.for_range(0, n_len) as n_idx_1:
                        _cdhw_less_half_ub_process_fp16(n_idx_1)
                else:
                    n_mv_in_lp, n_len_left, n_len_new = _get_mv_in_para()
                    with tik_inst.for_range(0, n_mv_in_lp) as mv_in_lp_idx:
                        _cdhw_less_half_ub_process(mv_in_lp_idx, n_len_new)
                    if n_len_left:
                        _cdhw_less_half_ub_process(n_mv_in_lp, n_len_left)

            else:
                with tik_inst.for_range(0, n_len) as n_idx:

                    def _cdhw_bigger_half_ub_process():
                        """
                        process of cdhw bigger than half UB
                        """

                        def _c0hw_hwc0_transfer(c0_index, sub_c_count):
                            """
                            do transpose from c0hw to hwc0
                            """

                            with tik_inst.for_range(0, axis_d) as d1_idx:

                                def _inner_process(loop_index, hw_len):
                                    """
                                    inner process of the transpose
                                    """

                                    # move in hw_len block in 16 times
                                    input_offset_1 = ((block_idx * per_core_n_cnt + n_idx) *
                                                      cdhw_size + c0_index * 16 * axis_d * hw_size
                                                      + d1_idx * hw_size +
                                                      loop_index * ub_col_size)
                                    with tik_inst.for_range(0, sub_c_count) as sub_c_idx:
                                        tik_inst.data_move(
                                            in_ub[sub_c_idx * ub_col_size],
                                            data_in[input_offset_1 + sub_c_idx * axis_d * hw_size],
                                            0, 1,
                                            _ceil_div(hw_len, BLOCK_BYTE_SIZE // dtype_factor),
                                            0, 0)

                                    if data_in.dtype.lower() == "float16":
                                        # do vnchwconv transfer
                                        src_addr_list = [in_ub[ub_col_size * i]
                                                         for i in ADDR_IDX_LIST]
                                        dst_addr_list = [out_ub[C0_LEN * i]
                                                         for i in ADDR_IDX_LIST]
                                        repeat_cnt = _ceil_div(hw_len, C0_LEN)
                                        src_stride = 0 if repeat_cnt == 1 else 1
                                        dst_stride = 0 if repeat_cnt == 1 else 16
                                        tik_inst.vnchwconv(False, False,
                                                           dst_addr_list, src_addr_list,
                                                           repeat_cnt, dst_stride, src_stride)
                                    else:
                                        axis_param = (1, sub_c_count, hw_len, ub_col_size, C0_LEN)
                                        _scalar_conv(tik_inst, out_ub, in_ub, axis_param, reg_list)

                                    # move out hw_len block each time
                                    output_offset_1 = (block_idx * per_core_n_cnt + n_idx +
                                                       ((d1_idx * axis_c1 + c0_index) * hw_size +
                                                        loop_index * ub_col_size) *
                                                       ni_no_size) * C0_LEN
                                    tik_inst.data_move(data_out[output_offset_1],
                                                       out_ub,
                                                       0, hw_len, dtype_factor // 2,
                                                       0, (ni_no_size - 1) * dtype_factor // 2)

                                is_hw_less_ub_col_size = hw_size <= ub_col_size
                                if is_hw_less_ub_col_size:
                                    _inner_process(0, hw_size)
                                else:
                                    buf_loop = hw_size // ub_col_size
                                    hw_left = hw_size % ub_col_size

                                    if buf_loop:
                                        with tik_inst.for_range(0, buf_loop) as buf_lp_idx:
                                            _inner_process(buf_lp_idx, ub_col_size)
                                        if hw_left:
                                            _inner_process(buf_loop, hw_left)

                        if c0_count_in_c:
                            with tik_inst.for_range(0, c0_count_in_c) as c0_idx:
                                _c0hw_hwc0_transfer(c0_idx, C0_LEN)
                            if c_left:
                                # to avoid dirty data
                                if data_in.dtype.lower() == "float16":
                                    _clean_ubuf(tik_inst, in_ub, c_left * ub_col_size,
                                                (C0_LEN - c_left) * ub_col_size)
                                else:
                                    _clean_ubuf(tik_inst, out_ub, 0, ub_size)
                                _c0hw_hwc0_transfer(c0_count_in_c, c_left)
                        else:
                            # only need to clean buf once
                            with tik_inst.if_scope(n_idx == 0):
                                # to avoid dirty data
                                if data_in.dtype.lower() == "float16":
                                    _clean_ubuf(tik_inst, in_ub, c_left * ub_col_size,
                                                (C0_LEN - c_left) * ub_col_size)
                                else:
                                    _clean_ubuf(tik_inst, out_ub, 0, ub_size)
                            _c0hw_hwc0_transfer(0, c_left)
                    _cdhw_bigger_half_ub_process()

        with tik_inst.if_scope(block_idx == core_num - 1):
            _n_transfer_process(left_n_cnt)
        with tik_inst.else_scope():
            _n_transfer_process(per_core_n_cnt)

        if axis_n % C0_LEN:
            if is_chwd_less_half_ub:
                _clean_ubuf(tik_inst, out_ub, 0, hw_size * C0_LEN)
                buf_loop_cnt = 0
                hw_left_size = 0
            else:
                _clean_ubuf(tik_inst, out_ub, 0, ub_size)
                buf_loop_cnt = hw_size // ub_col_size
                hw_left_size = hw_size % ub_col_size

            def _padding_ni_no_cube(output_offset_z):
                """
                set the left size in one ninoc0 cube to zero
                """
                if buf_loop_cnt == 0 or (hw_left_size == 0 and buf_loop_cnt == 1):
                    tik_inst.data_move(
                        data_out[output_offset_z],
                        out_ub,
                        0, hw_size,
                        dtype_factor // 2,
                        0, (ni_no_size - 1) * dtype_factor // 2)
                else:
                    with tik_inst.for_range(0, buf_loop_cnt) as lp_idx:
                        tik_inst.data_move(
                            data_out[output_offset_z + lp_idx * ub_col_size * ni_no_size * C0_LEN],
                            out_ub,
                            0, ub_col_size,
                            dtype_factor // 2,
                            0, (ni_no_size - 1) * dtype_factor // 2)
                    if hw_left_size:
                        tik_inst.data_move(
                            data_out[output_offset_z +
                                     buf_loop_cnt * ub_col_size * ni_no_size * C0_LEN],
                            out_ub,
                            0, hw_left_size,
                            dtype_factor // 2,
                            0, (ni_no_size - 1) * dtype_factor // 2)

            if zero_loop_cnt:
                with tik_inst.for_range(0, zero_loop_cnt) as z_lp_idx:
                    with tik_inst.for_range(0, axis_d) as d_idx:
                        with tik_inst.for_range(0, axis_c1) as c1_idx:
                            output_offset_zero = (axis_n + block_idx * zero_loop_cnt +
                                                  z_lp_idx + (d_idx * axis_c1 + c1_idx) *
                                                  hw_size * ni_no_size) * C0_LEN
                            _padding_ni_no_cube(output_offset_zero)

            if zero_line_left:
                with tik_inst.if_scope(block_idx < zero_line_left):
                    with tik_inst.for_range(0, axis_d) as d_idx:
                        with tik_inst.for_range(0, axis_c1) as c1_idx:
                            output_offset_zero = (axis_n + core_num * zero_loop_cnt +
                                                  block_idx + (d_idx * axis_c1 + c1_idx) *
                                                  hw_size * ni_no_size) * C0_LEN
                            _padding_ni_no_cube(output_offset_zero)


def ncdhw_2_fractal_z_3d_compute(tik_inst, data_in, data_out):
    """
    do ncdhw to fractal_z_3d transfer
    """

    shape_in = [int(x) for x in data_in.shape[:]]
    _multi_core_on_n(tik_inst, data_in, data_out, shape_in)


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR,
                 REQUIRED_ATTR_STR, KERNEL_NAME)
def ncdhw_2_fractal_z_3d(src, dst, src_format, dst_format,
                         kernel_name="ncdhw_2_fractal_z_3d"):
    """
    used to transfer ncdhw to fractal_z_3d

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        shape and dtype of output, should be same shape and type as input
    src_format : str
        input format, the value should be "NCDHW"
    dst_format : str
         output format, the value should be "FRACTAL_Z_3D"
    kernel_name : str
        kernel name, default value is "ncdhw_2_fractal_z_3d"

    Returns
    -------
    None
    """

    in_shape = src.get("shape")
    dst_shape = (in_shape[2], _ceil_div(in_shape[1], C0_LEN), in_shape[3],
                 in_shape[4], _ceil_div(in_shape[0], C0_LEN), C0_LEN, C0_LEN)
    in_dtype = src.get("dtype").lower()
    dst_dtype = dst.get("dtype").lower()

    # check input parameters valid or not
    input_params = (in_shape, dst_shape, in_dtype, dst_dtype, src_format, dst_format)
    _check_input_params(input_params)

    # initial Tik
    tik_inst = tik.Tik()
    # define input and output tensors
    data_in = tik_inst.Tensor(in_dtype, in_shape, tik.scope_gm, "data_in")
    data_out = tik_inst.Tensor(in_dtype, dst_shape, tik.scope_gm, "data_out")

    # do transfer
    ncdhw_2_fractal_z_3d_compute(tik_inst, data_in, data_out)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[data_in], outputs=[data_out])
