"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

transpose
"""


import te.lang.dynamic
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_LIST_INT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import check_op_params
from te.utils.error_manager import error_manager_vector

# pylint: disable=too-many-lines
# UB size in byte
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# AICORE count
CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# byte count one block
BLOCK_BYTE_COUNT = 32
# repeat up limit for vector
REPEAT_LIMIT = 255
# repeat up limit for mte
REPEAT_LIMIT_MTE = 4095
# threshold for mte
MTE_THRESHOLD = 3968
# stride up limit for mte
MTE_STRIDES = 65535
# mask value
MASK_128 = 128
# float16/32 type list
TYPE_FLOAT_LIST = ("float32",)
# int type list
TYPE_INT_LIST = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
# max int64 value
MAX_INT64_VALUE = 2 ** 64 - 1
# permute shape size
PERM_SHAPE_LIMIT = 32
# parameters for moving tiling data
TILING_CTRL_PARAM = ("int64", 64, 4)
# used for vnchwconv
ADDR_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def _ceil_div(value_x, value_y):
    """
    do ceil division
    """
    return (value_x + value_y - 1) // value_y


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


def _get_dtype_len(in_dtype):
    """
    get the byte count in certain dtype
    """

    temp_dtype = in_dtype.lower()

    if temp_dtype in ("int8", "uint8"):
        byte_len = 1
    elif temp_dtype in ("float16", "int16", "uint16"):
        byte_len = 2
    elif temp_dtype in ("float32", "int32", "uint32"):
        byte_len = 4
    elif temp_dtype in ("int64", "uint64"):
        byte_len = 8

    return byte_len


def _get_elment_cnt_one_block(in_dtype):
    """
    get element count in a block
    """

    byte_len = _get_dtype_len(in_dtype)
    element_cnt = BLOCK_BYTE_COUNT // byte_len

    return element_cnt


def _get_max_element_in_ub(in_dtype, ub_part):
    """
    get the up limit elements in UB
    """

    byte_len = _get_dtype_len(in_dtype)

    # the unit is Byte
    ub_upper_limit = (UB_SIZE - 8 * 1024) // ub_part
    element_size = ub_upper_limit // byte_len

    return element_size


def _check_input_params(input_params):
    """
    to the check whether the input parameters is valid or not
    """

    in_dtype, dst_dtype, perm = input_params

    if in_dtype != dst_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("transpose_d", "in_dtype",
                                                              "dst_dtype", in_dtype, dst_dtype)

    if len(perm) < 2 or len(perm) > 3:
        perm_rule = "the perm length should be in range (2,3)"
        error_manager_vector.raise_err_check_params_rules("transpose_d", perm_rule,
                                                          "perm", len(perm))

    if in_dtype not in TYPE_FLOAT_LIST:
        error_detail = "only support dtype float32 now"
        error_manager_vector.raise_err_two_input_dtype_invalid("transpose_d", "in_dtype",
                                                               "dst_dtype", error_detail)


def _get_positive_perm(old_perm):
    """
    get positive perm
    """

    perm_len = len(old_perm)
    new_perm = [x if x >= 0 else x + perm_len for x in old_perm]

    if max(new_perm) >= perm_len:
        perm_rule = "the dim value in perm is invalid"
        error_manager_vector.raise_err_check_params_rules("transpose_d", perm_rule,
                                                          "new_perm", new_perm)

    return new_perm


# pylint: disable=too-many-locals
def _data_move_in_mc_on_w(tik_inst, dst, src, data_pos_info):
    """
    do move in data from out to ub for multiple core along w
    """

    sub_h_size, sub_w_size, h_size, w_size, w_offset = data_pos_info
    data_cnt_one_block = _get_elment_cnt_one_block(src.dtype)
    sub_w_block = _ceil_div(sub_w_size, data_cnt_one_block)
    sub_h_align_block_size = sub_h_size // data_cnt_one_block * data_cnt_one_block
    sub_h_left = sub_h_size % data_cnt_one_block
    is_not_w_block_align = w_size % data_cnt_one_block > 0
    is_h_size_smaller_one_block = h_size < data_cnt_one_block

    def _move_in_one_more_block():
        """
        move in one more block of h when h > sub_h and sub_h is not block align
        """
        with tik_inst.for_range(0, sub_h_align_block_size) as sub_h_idx:
            tik_inst.data_move(dst[sub_w_block * data_cnt_one_block * sub_h_idx],
                               src[w_offset + w_size * sub_h_idx], 0, 1, sub_w_block, 0, 0)
            # in order to avoid dirty data when multiple core
        with tik_inst.for_range(0, data_cnt_one_block) as sub_h_idx_1:
            tik_inst.data_move(dst[sub_w_block * data_cnt_one_block *
                                   (sub_h_align_block_size + sub_h_idx_1)],
                               src[w_offset +
                                   w_size * (sub_h_size - data_cnt_one_block + sub_h_idx_1)],
                               0, 1, sub_w_block, 0, 0)

    with tik_inst.if_scope(is_not_w_block_align):
        # sub_h is block align or h is not enough one block
        with tik_inst.if_scope(tik.any(sub_h_left == 0, is_h_size_smaller_one_block)):
            with tik_inst.for_range(0, sub_h_size) as sub_h_idx:
                tik_inst.data_move(dst[sub_w_block * data_cnt_one_block * sub_h_idx],
                                   src[w_offset + w_size * sub_h_idx], 0, 1, sub_w_block, 0, 0)
        with tik_inst.else_scope():
            _move_in_one_more_block()

    with tik_inst.else_scope():
        with tik_inst.if_scope(tik.any(sub_h_left == 0, is_h_size_smaller_one_block)):
            src_strides = w_size // data_cnt_one_block - sub_w_block
            # mte max strides value is 65535
            with tik_inst.if_scope(src_strides > MTE_STRIDES):
                with tik_inst.for_range(0, sub_h_size) as sub_h_idx_2:
                    tik_inst.data_move(dst[sub_w_size * sub_h_idx_2],
                                       src[w_offset + w_size * sub_h_idx_2],
                                       0, 1, sub_w_block, 0, 0)
            with tik_inst.else_scope():
                tik_inst.data_move(dst, src[w_offset], 0, sub_h_size, sub_w_block, src_strides, 0)
        with tik_inst.else_scope():
            _move_in_one_more_block()


def _data_move_in_mc_on_h(tik_inst, dst, src, data_pos_info):
    """
    do move in data from out to ub for multiple core along h
    """

    sub_h_size, sub_w_size, h_size, w_size, in_offset = data_pos_info
    data_cnt_one_block = _get_elment_cnt_one_block(src.dtype)
    sub_w_block = _ceil_div(sub_w_size, data_cnt_one_block)
    sub_h_align_block_size = sub_h_size // data_cnt_one_block * data_cnt_one_block
    sub_h_left = sub_h_size % data_cnt_one_block
    sub_hw_align_block = _ceil_div(sub_h_size * w_size, data_cnt_one_block)
    is_not_w_block_align = w_size % data_cnt_one_block > 0
    is_h_size_smaller_one_block = h_size < data_cnt_one_block
    is_subw_equal_w = sub_w_size == w_size

    def _move_in_one_more_block():
        """
        move in one more block of h when h > sub_h and sub_h is not block align
        """
        with tik_inst.for_range(0, sub_h_align_block_size) as sub_h_idx_0:
            tik_inst.data_move(dst[sub_w_block * data_cnt_one_block * sub_h_idx_0],
                               src[in_offset + sub_h_idx_0 * w_size],
                               0, 1, sub_w_block, 0, 0)
            # move in one more block of h
        with tik_inst.for_range(0, data_cnt_one_block) as sub_h_idx_1:
            tik_inst.data_move(
                dst[sub_w_block * data_cnt_one_block * (sub_h_align_block_size + sub_h_idx_1)],
                src[in_offset + (sub_h_idx_1 + sub_h_size - data_cnt_one_block) * w_size],
                0, 1, sub_w_block, 0, 0)

    with tik_inst.if_scope(is_subw_equal_w):
        # no need to move in one more block
        with tik_inst.if_scope(tik.any(sub_h_left == 0, is_h_size_smaller_one_block)):
            tik_inst.data_move(dst, src[in_offset], 0, 1, sub_hw_align_block, 0, 0)
        with tik_inst.else_scope():
            _move_in_one_more_block()

    with tik_inst.else_scope():
        # no need move in one more block of h
        with tik_inst.if_scope(tik.any(sub_h_left == 0, is_h_size_smaller_one_block)):
            src_strides = w_size // data_cnt_one_block - sub_w_block
            # mte max strides value is 65535
            with tik_inst.if_scope(tik.any(src_strides > MTE_STRIDES, is_not_w_block_align)):
                with tik_inst.for_range(0, sub_h_size) as sub_h_idx:
                    tik_inst.data_move(dst[sub_w_block * data_cnt_one_block * sub_h_idx],
                                       src[in_offset + w_size * sub_h_idx], 0, 1, sub_w_block, 0, 0)
            with tik_inst.else_scope():
                tik_inst.data_move(dst, src[in_offset], 0, sub_h_size, sub_w_block, src_strides, 0)
        with tik_inst.else_scope():
            _move_in_one_more_block()


# pylint: disable=unused-variable
def _data_move_out_mc_on_w(tik_inst, dst, src, data_pos_info):
    """
    do move out data from ub to out for multiple core along w
    """

    # sub_h_size is the original value without any change
    sub_h_size, sub_w_size, h_size, w_size, out_offset = data_pos_info
    data_size_one_block = _get_elment_cnt_one_block(src.dtype)

    def _sub_h_not_block_align_bigger_one_block():
        """
        sub_h_size is not block align, sub_h_size is bigger than one block
        """

        sub_h_block = sub_h_size // data_size_one_block
        with tik_inst.for_range(0, sub_w_size) as sub_w_idx_2:
            with tik_inst.if_scope(sub_h_block > 0):
                tik_inst.data_move(
                    dst[out_offset + sub_w_idx_2 * h_size],
                    src[sub_w_idx_2 * (sub_h_block + 1) * data_size_one_block],
                    0, 1, sub_h_block, 0, 0)
            # move in one more block for this case
            tik_inst.data_move(
                dst[out_offset + sub_w_idx_2 * h_size + sub_h_size - data_size_one_block],
                src[sub_w_idx_2 * (sub_h_block + 1) * data_size_one_block +
                    sub_h_block * data_size_one_block],
                0, 1, 1, 0, 0)

    with tik_inst.if_scope(sub_h_size == h_size):
        # the data order in ub is the expected order
        sub_hw_size = sub_h_size * sub_w_size
        with tik_inst.if_scope(h_size % data_size_one_block == 0):
            tik_inst.data_move(dst[out_offset],
                               src,
                               0, 1, sub_hw_size // data_size_one_block, 0, 0)
        with tik_inst.else_scope():
            # sub_h_size is smaller than one block
            with tik_inst.if_scope(h_size < data_size_one_block):
                # the data_move will move 1 block at least
                with tik_inst.if_scope(sub_hw_size < data_size_one_block):
                    tik_inst.data_move(dst[out_offset],
                                       src,
                                       0, 1, 1, 0, 0)
                with tik_inst.else_scope():
                    sub_hw_block = sub_hw_size // data_size_one_block
                    tik_inst.data_move(dst[out_offset],
                                       src,
                                       0, 1, sub_hw_block, 0, 0)
                    # in order to avoid dirty data
                    with tik_inst.new_stmt_scope():
                        temp_reg = [tik_inst.Scalar(src.dtype)
                                    for i in ADDR_IDX_LIST[:data_size_one_block]]
                        for idx in ADDR_IDX_LIST[:data_size_one_block]:
                            temp_reg[idx].set_as(src[sub_hw_size - data_size_one_block + idx])
                        for idx in ADDR_IDX_LIST[:data_size_one_block]:
                            src[idx].set_as(temp_reg[idx])
                        tik_inst.data_move(dst[out_offset + sub_hw_size - data_size_one_block],
                                           src, 0, 1, 1, 0, 0)
            with tik_inst.else_scope():
                # sub_h_size is not block align, sub_h_size is bigger than one block
                _sub_h_not_block_align_bigger_one_block()

    with tik_inst.else_scope():
        # h_size > sub_h_size, h_size is block align
        stride_cnt = (h_size - sub_h_size) // data_size_one_block
        with tik_inst.if_scope(tik.all(h_size % data_size_one_block == 0,
                                       stride_cnt <= MTE_STRIDES)):
            tik_inst.data_move(dst[out_offset],
                               src,
                               0, sub_w_size, sub_h_size // data_size_one_block, 0, stride_cnt)
        with tik_inst.else_scope():
            # h_size is not block align, sub_h_size is block align
            with tik_inst.if_scope(sub_h_size % data_size_one_block == 0):
                with tik_inst.for_range(0, sub_w_size) as sub_w_idx:
                    tik_inst.data_move(dst[out_offset + sub_w_idx * h_size],
                                       src[sub_w_idx * sub_h_size],
                                       0, 1, sub_h_size // data_size_one_block, 0, 0)
            with tik_inst.else_scope():
                _sub_h_not_block_align_bigger_one_block()


def _data_move_in_last_dim_be_one_block(tik_inst, dst, src, data_pos_info):
    """
    move data in for transpose by mte when last dim bigger than or equal to one block
    """

    sub_h_size, sub_w_size, h_size, w_size, in_offset = data_pos_info
    data_size_one_block = _get_elment_cnt_one_block(src.dtype)
    is_w_block_align = w_size % data_size_one_block == 0
    sub_w_block = sub_w_size // data_size_one_block
    sub_w_left = sub_w_size % data_size_one_block
    w_block = w_size // data_size_one_block
    stride_len = w_block - sub_w_block

    with tik_inst.if_scope(is_w_block_align):
        with tik_inst.if_scope(tik.any(stride_len > MTE_STRIDES, sub_h_size > REPEAT_LIMIT_MTE)):
            with tik_inst.for_range(0, sub_h_size) as sub_h_idx:
                tik_inst.data_move(dst[sub_h_idx * sub_w_size],
                                   src[sub_h_idx * w_size + in_offset],
                                   0, 1, sub_w_block, 0, 0)
        with tik_inst.else_scope():
            with tik_inst.if_scope(stride_len > 0):
                tik_inst.data_move(dst, src[in_offset], 0, sub_h_size, sub_w_block, stride_len, 0)
            with tik_inst.else_scope():
                tik_inst.data_move(dst, src[in_offset], 0, 1, sub_h_size * sub_w_block, 0, 0)

    with tik_inst.else_scope():
        # suppose w_size > data_size_one_block
        with tik_inst.for_range(0, sub_h_size) as sub_h_idx:
            with tik_inst.if_scope(sub_w_block > 0):
                tik_inst.data_move(dst[sub_h_idx * (sub_w_block + 1) * data_size_one_block],
                                   src[sub_h_idx * w_size + in_offset],
                                   0, 1, sub_w_block, 0, 0)
            with tik_inst.if_scope(sub_w_left > 0):
                tik_inst.data_move(dst[(sub_h_idx * (sub_w_block + 1) + sub_w_block) *
                                       data_size_one_block],
                                   src[sub_h_idx * w_size + in_offset + sub_w_size -
                                       data_size_one_block],
                                   0, 1, 1, 0, 0)


# pylint: disable=unused-variable
def _data_move_out_last_dim_be_one_block(tik_inst, dst, src, data_pos_info):
    """
    move data out for transpose by mte when last dim bigger than or equal to one block
    """

    sub_h_size, sub_w_size, c_size, h_size, w_size, out_offset = data_pos_info
    data_size_one_block = _get_elment_cnt_one_block(src.dtype)
    is_w_block_align = w_size % data_size_one_block == 0
    sub_w_block = sub_w_size // data_size_one_block
    sub_w_left = sub_w_size % data_size_one_block
    cw_block = c_size * w_size // data_size_one_block
    stride_len = cw_block - sub_w_block

    with tik_inst.if_scope(is_w_block_align):
        with tik_inst.if_scope(tik.any(stride_len > MTE_STRIDES, sub_h_size > REPEAT_LIMIT_MTE)):
            with tik_inst.for_range(0, sub_h_size) as sub_h_idx:
                tik_inst.data_move(dst[sub_h_idx * c_size * w_size + out_offset],
                                   src[sub_h_idx * sub_w_size],
                                   0, 1, sub_w_block, 0, 0)
        with tik_inst.else_scope():
            tik_inst.data_move(dst[out_offset], src, 0, sub_h_size, sub_w_block, 0, stride_len)

    with tik_inst.else_scope():
        # suppose w_size > data_size_one_block
        with tik_inst.for_range(0, sub_h_size) as sub_h_idx:
            with tik_inst.if_scope(sub_w_block > 0):
                tik_inst.data_move(dst[sub_h_idx * c_size * w_size + out_offset],
                                   src[sub_h_idx * (sub_w_block + 1) * data_size_one_block],
                                   0, 1, sub_w_block, 0, 0)
            with tik_inst.if_scope(sub_w_left > 0):
                tik_inst.data_move(dst[sub_h_idx * c_size * w_size + out_offset +
                                       sub_w_size - data_size_one_block],
                                   src[(sub_h_idx * (sub_w_block + 1) + sub_w_block) *
                                       data_size_one_block],
                                   0, 1, 1, 0, 0)


def _data_move_in_last_dim_lt_one_block(tik_inst, dst, src, data_pos_info):
    """
    move data in for transpose by mte when last dim less than one block
    """
    sub_axis_1, sub_axis_0, axis_0, axis_1, axis_2, in_offset = data_pos_info
    data_size_one_block = _get_elment_cnt_one_block(src.dtype)

    with tik_inst.if_scope(sub_axis_1 == 1):
        with tik_inst.if_scope(tik.any(sub_axis_0 * axis_2 >= data_size_one_block,
                                       axis_0 * axis_1 * axis_2 <= data_size_one_block)):
            with tik_inst.for_range(0, sub_axis_0) as sub_axis_0_idx:
                tik_inst.data_move(
                    dst[sub_axis_0_idx * data_size_one_block],
                    src[sub_axis_0_idx * axis_1 * axis_2 + in_offset],
                    0, 1, 1, 0, 0)
        with tik_inst.else_scope():
            # to make sure move in data size is not less than one block
            m_sub_axis_0 = _ceil_div(data_size_one_block, axis_2)
            with tik_inst.for_range(0, m_sub_axis_0) as m_sub_axis_0_idx:
                tik_inst.data_move(
                    dst[m_sub_axis_0_idx * data_size_one_block],
                    src[(m_sub_axis_0_idx + sub_axis_0 - m_sub_axis_0) * axis_1 * axis_2 +
                        in_offset],
                    0, 1, 1, 0, 0)

    with tik_inst.else_scope():
        with tik_inst.if_scope(tik.any(sub_axis_1 * axis_0 * axis_2 >= data_size_one_block,
                                       axis_0 * axis_1 * axis_2 <= data_size_one_block)):
            with tik_inst.for_range(0, sub_axis_1) as sub_axis_1_idx:
                with tik_inst.for_range(0, sub_axis_0) as sub_axis_0_idx:
                    tik_inst.data_move(
                        dst[(sub_axis_0_idx + sub_axis_1_idx * sub_axis_0) * data_size_one_block],
                        src[(sub_axis_0_idx * axis_1 + sub_axis_1_idx) * axis_2 + in_offset],
                        0, 1, 1, 0, 0)
        with tik_inst.else_scope():
            # to make sure move in data size is not less than one block
            m_sub_axis_1 = _ceil_div(data_size_one_block, axis_0 * axis_2)
            with tik_inst.for_range(0, m_sub_axis_1) as m_sub_axis_1_idx:
                with tik_inst.for_range(0, sub_axis_0) as sub_axis_0_idx:
                    tik_inst.data_move(
                        dst[(sub_axis_0_idx + m_sub_axis_1_idx * sub_axis_0) *
                            data_size_one_block],
                        src[(sub_axis_0_idx * axis_1 + sub_axis_1 - m_sub_axis_1 +
                             m_sub_axis_1_idx) * axis_2 + in_offset],
                        0, 1, 1, 0, 0)


# pylint: disable=too-many-statements
def _data_move_out_last_dim_lt_one_block(tik_inst, dst, src, data_pos_info):
    """
    move data out for transpose by mte when last dim less than one block
    """
    sub_axis_1, sub_axis_0, axis_0, axis_1, axis_2, out_offset = data_pos_info
    data_size_one_block = _get_elment_cnt_one_block(src.dtype)

    with tik_inst.if_scope(sub_axis_1 == 1):
        with tik_inst.if_scope(sub_axis_0 * axis_2 > data_size_one_block):
            sub_axis_0_2 = sub_axis_0 * axis_2
            sub_axis_0_block_align = sub_axis_0_2 // data_size_one_block
            left_data = sub_axis_0_2 % data_size_one_block
            tik_inst.data_move(dst[out_offset], src, 0, 1, sub_axis_0_block_align, 0, 0)
            with tik_inst.if_scope(left_data > 0):
                with tik_inst.new_stmt_scope():
                    reg_temp = [tik_inst.Scalar(src.dtype)
                                for i in ADDR_IDX_LIST[:data_size_one_block]]
                    for idx in ADDR_IDX_LIST[:data_size_one_block]:
                        reg_temp[idx].set_as(src[sub_axis_0_2 - data_size_one_block + idx])
                    for idx in ADDR_IDX_LIST[:data_size_one_block]:
                        src[idx].set_as(reg_temp[idx])
                    tik_inst.data_move(dst[out_offset + sub_axis_0_2 - data_size_one_block],
                                       src, 0, 1, 1, 0, 0)
        with tik_inst.else_scope():
            # for case sub_axis_0 * axis_2 < data_size_one_block
            with tik_inst.if_scope(axis_0 * axis_1 * axis_2 > data_size_one_block):
                m_sub_axis_0 = _ceil_div(data_size_one_block, axis_2)
                sub_axis_0_2 = m_sub_axis_0 * axis_2
                sub_axis_0_block_align = sub_axis_0_2 // data_size_one_block
                left_data = sub_axis_0_2 % data_size_one_block
                tik_inst.data_move(dst[out_offset + (sub_axis_0 - m_sub_axis_0) * axis_2],
                                   src, 0, 1, sub_axis_0_block_align, 0, 0)
                with tik_inst.if_scope(left_data > 0):
                    with tik_inst.new_stmt_scope():
                        reg_temp = [tik_inst.Scalar(src.dtype)
                                    for i in ADDR_IDX_LIST[:data_size_one_block]]
                        for idx in ADDR_IDX_LIST[:data_size_one_block]:
                            reg_temp[idx].set_as(src[sub_axis_0_2 - data_size_one_block + idx])
                        for idx in ADDR_IDX_LIST[:data_size_one_block]:
                            src[idx].set_as(reg_temp[idx])
                        tik_inst.data_move(dst[out_offset + sub_axis_0 * axis_2 -
                                               data_size_one_block],
                                           src, 0, 1, 1, 0, 0)
            with tik_inst.else_scope():
                # the shape size is not bigger than one block size
                tik_inst.data_move(dst[out_offset], src, 0, 1,
                                   _ceil_div(axis_0 * axis_1 * axis_2, data_size_one_block), 0, 0)

    with tik_inst.else_scope():
        with tik_inst.if_scope(sub_axis_1 * axis_0 * axis_2 > data_size_one_block):
            sub_axis_1_0_2 = sub_axis_1 * axis_0 * axis_2
            sub_axis_1_0_2_block_align = sub_axis_1_0_2 // data_size_one_block
            left_data = sub_axis_1_0_2 % data_size_one_block
            tik_inst.data_move(dst[out_offset], src, 0, 1, sub_axis_1_0_2_block_align, 0, 0)
            with tik_inst.if_scope(left_data > 0):
                with tik_inst.new_stmt_scope():
                    reg_temp = [tik_inst.Scalar(src.dtype)
                                for i in ADDR_IDX_LIST[:data_size_one_block]]
                    for idx in ADDR_IDX_LIST[:data_size_one_block]:
                        reg_temp[idx].set_as(src[sub_axis_1_0_2 - data_size_one_block + idx])
                    for idx in ADDR_IDX_LIST[:data_size_one_block]:
                        src[idx].set_as(reg_temp[idx])
                    tik_inst.data_move(dst[out_offset + sub_axis_1_0_2 - data_size_one_block],
                                       src, 0, 1, 1, 0, 0)
        with tik_inst.else_scope():
            # to make sure move in data size is not less than one block
            with tik_inst.if_scope(axis_0 * axis_1 * axis_2 > data_size_one_block):
                m_sub_axis_1 = _ceil_div(data_size_one_block, axis_0 * axis_2)
                sub_axis_1_0_2 = m_sub_axis_1 * axis_0 * axis_2
                sub_axis_0_block_align = sub_axis_1_0_2 // data_size_one_block
                left_data = sub_axis_1_0_2 % data_size_one_block
                tik_inst.data_move(dst[out_offset + (sub_axis_1 - m_sub_axis_1) * axis_0 * axis_2],
                                   src, 0, 1, sub_axis_0_block_align, 0, 0)
                with tik_inst.if_scope(left_data > 0):
                    with tik_inst.new_stmt_scope():
                        reg_temp = [tik_inst.Scalar(src.dtype)
                                    for i in ADDR_IDX_LIST[:data_size_one_block]]
                        for idx in ADDR_IDX_LIST[:data_size_one_block]:
                            reg_temp[idx].set_as(src[sub_axis_1_0_2 - data_size_one_block + idx])
                        for idx in ADDR_IDX_LIST[:data_size_one_block]:
                            src[idx].set_as(reg_temp[idx])
                        tik_inst.data_move(dst[out_offset + sub_axis_1 * axis_0 * axis_2 -
                                               data_size_one_block],
                                           src, 0, 1, 1, 0, 0)
            with tik_inst.else_scope():
                # the shape size is not bigger than one block size
                tik_inst.data_move(dst[out_offset], src, 0, 1,
                                   _ceil_div(axis_0 * axis_1 * axis_2, data_size_one_block), 0, 0)


# pylint: disable=unnecessary-pass
def _data_move_out_mc_on_h():
    """
    do move out data from ub to out for multiple core along h
    """

    pass


def _data_move_in_no_transpose(tik_inst, dst, src, data_pos_info):
    """
    do move in data from out to ub for transpose nothing
    """

    data_size, in_offset, h_size, w_size = data_pos_info
    data_one_block = _get_elment_cnt_one_block(src.dtype)
    data_align_block = data_size // data_one_block
    is_data_block_algin = data_size % data_one_block == 0
    is_all_data_small_one_block = h_size * w_size < data_one_block

    with tik_inst.if_scope(is_data_block_algin):
        tik_inst.data_move(dst, src[in_offset], 0, 1, data_align_block, 0, 0)
    with tik_inst.else_scope():
        with tik_inst.if_scope(is_all_data_small_one_block):
            tik_inst.data_move(dst, src[in_offset], 0, 1, 1, 0, 0)
        with tik_inst.else_scope():
            tik_inst.data_move(dst, src[in_offset], 0, 1, data_align_block, 0, 0)
            tik_inst.data_move(dst[data_align_block * data_one_block],
                               src[in_offset + data_size - data_one_block],
                               0, 1, 1, 0, 0)


def _data_move_out_no_transpose(tik_inst, dst, src, data_pos_info):
    """
    do move out data from ub to out for transpose nothing
    """
    data_size, out_offset, h_size, w_size = data_pos_info
    data_one_block = _get_elment_cnt_one_block(src.dtype)
    data_align_block = data_size // data_one_block
    is_data_block_algin = data_size % data_one_block == 0
    is_all_data_small_one_block = h_size * w_size < data_one_block

    with tik_inst.if_scope(is_data_block_algin):
        tik_inst.data_move(dst[out_offset], src, 0, 1, data_align_block, 0, 0)
    with tik_inst.else_scope():
        with tik_inst.if_scope(is_all_data_small_one_block):
            tik_inst.data_move(dst[out_offset], src, 0, 1, 1, 0, 0)
        with tik_inst.else_scope():
            tik_inst.data_move(dst[out_offset], src, 0, 1, data_align_block, 0, 0)
            tik_inst.data_move(dst[out_offset + data_size - data_one_block],
                               src[data_align_block * data_one_block],
                               0, 1, 1, 0, 0)


def _transpose_by_2_vnchwconv(tik_inst, dst, src, sub_hw_size):
    """
    do transpose by two times vnchwconv
    """

    # whether the sub_h_size is block align or not should be decided before transferring in
    sub_h_size, sub_w_size = sub_hw_size
    data_size_one_block = _get_elment_cnt_one_block(src.dtype)
    w_block_cnt = _ceil_div(sub_w_size, data_size_one_block)
    fp16_src = src.reinterpret_cast_to("float16")
    fp16_dst = dst.reinterpret_cast_to("float16")
    fp16_data_one_block = _get_elment_cnt_one_block("float16")
    # vnchwconv get two bytes per time
    if src.dtype.lower() in ("float32", "int32", "uint32"):
        vnc_one_line_len = w_block_cnt * data_size_one_block * sub_h_size * 2
    elif src.dtype.lower() in ("float16", "int16", "uint16"):
        vnc_one_line_len = w_block_cnt * data_size_one_block * sub_h_size
    else:
        error_detail = "not support the dtype"
        error_manager_vector.raise_err_two_input_dtype_invalid("transpose_d", "in_dtype",
                                                               "dst_dtype", error_detail)

    # do 16hc to hc16 transfer
    src_addr_list = [fp16_src[vnc_one_line_len * i] for i in ADDR_IDX_LIST]
    dst_addr_list = [fp16_dst[fp16_data_one_block * i] for i in ADDR_IDX_LIST]
    repeat_cnt = _ceil_div(vnc_one_line_len, fp16_data_one_block)
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar("int64")
        dst_stride = tik_inst.Scalar("int64")
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(1)
            dst_stride.set_as(16)
        tik_inst.vnchwconv(False, False,
                           dst_addr_list, src_addr_list,
                           repeat_cnt, dst_stride, src_stride)

    # do hc16 to ch16 transfer
    with tik_inst.if_scope(sub_h_size > sub_w_size):
        with tik_inst.for_range(0, sub_w_size) as w_size_idx:
            tik_inst.data_move(
                fp16_src[w_size_idx * sub_h_size * fp16_data_one_block * 2],
                fp16_dst[w_size_idx * fp16_data_one_block * 2],
                0, sub_h_size, 2, (w_block_cnt * data_size_one_block - 1) * 2, 0)
    with tik_inst.else_scope():
        with tik_inst.for_range(0, sub_h_size) as h_size_idx:
            tik_inst.data_move(
                fp16_src[h_size_idx * fp16_data_one_block * 2],
                fp16_dst[h_size_idx * w_block_cnt * data_size_one_block * fp16_data_one_block * 2],
                0, sub_w_size, 2, 0, (sub_h_size - 1) * 2)

    # do ch16 to 16ch transfer
    src_addr_list = [fp16_src[fp16_data_one_block * i] for i in ADDR_IDX_LIST]
    dst_addr_list = [fp16_dst[vnc_one_line_len * i] for i in ADDR_IDX_LIST]
    repeat_cnt = _ceil_div(vnc_one_line_len, fp16_data_one_block)
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar("int64")
        dst_stride = tik_inst.Scalar("int64")
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(16)
            dst_stride.set_as(1)
        tik_inst.vnchwconv(False, False,
                           dst_addr_list, src_addr_list,
                           repeat_cnt, dst_stride, src_stride)


def _transpose_by_2_vnchwconv_not_last_dim(tik_inst, dst, src, sub_dim_size):
    """
    do transpose by two times vnchwconv
    """

    # whether the sub_h_size is block align or not should be decided before transferring in
    sub_axis_1, sub_axis_0, axis_2 = sub_dim_size
    data_size_one_block = _get_elment_cnt_one_block(src.dtype)
    axis_2_block_cnt = _ceil_div(axis_2, data_size_one_block)
    fp16_src = src.reinterpret_cast_to("float16")
    fp16_dst = dst.reinterpret_cast_to("float16")
    fp16_data_one_block = _get_elment_cnt_one_block("float16")
    # vnchwconv get two bytes per time
    if src.dtype.lower() in ("float32", "int32", "uint32"):
        vnc_one_line_len = axis_2_block_cnt * data_size_one_block * sub_axis_1 * sub_axis_0 * 2
    elif src.dtype.lower() in ("float16", "int16", "uint16"):
        vnc_one_line_len = axis_2_block_cnt * data_size_one_block * sub_axis_1 * sub_axis_0
    else:
        error_detail = "not support the dtype"
        error_manager_vector.raise_err_two_input_dtype_invalid("transpose_d", "in_dtype",
                                                               "dst_dtype", error_detail)

    # do 16hc to hc16 transfer
    src_addr_list = [fp16_src[vnc_one_line_len * i] for i in ADDR_IDX_LIST]
    dst_addr_list = [fp16_dst[fp16_data_one_block * i] for i in ADDR_IDX_LIST]
    repeat_cnt = _ceil_div(vnc_one_line_len, fp16_data_one_block)
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar("int64")
        dst_stride = tik_inst.Scalar("int64")
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(1)
            dst_stride.set_as(16)
        tik_inst.vnchwconv(False, False,
                           dst_addr_list, src_addr_list,
                           repeat_cnt, dst_stride, src_stride)

    # do sub_axis_1*sub_axis_0*16 to sub_axis_1*sub_axis_0*axis_2 transfer
    with tik_inst.for_range(0, sub_axis_1) as sub_axis_1_idx:
        tik_inst.data_move(
            fp16_src[sub_axis_1_idx * sub_axis_0 * axis_2 * fp16_data_one_block * 2],
            fp16_dst[sub_axis_1_idx * sub_axis_0 * fp16_data_one_block * fp16_data_one_block],
            0, sub_axis_0, 2 * axis_2, fp16_data_one_block - 2 * axis_2, 0)

    # do ch16 to 16ch transfer
    src_addr_list = [fp16_src[fp16_data_one_block * i] for i in ADDR_IDX_LIST]
    dst_addr_list = [fp16_dst[vnc_one_line_len * i] for i in ADDR_IDX_LIST]
    repeat_cnt = _ceil_div(vnc_one_line_len, fp16_data_one_block)
    with tik_inst.new_stmt_scope():
        src_stride = tik_inst.Scalar("int64")
        dst_stride = tik_inst.Scalar("int64")
        with tik_inst.if_scope(repeat_cnt == 1):
            src_stride.set_as(0)
            dst_stride.set_as(0)
        with tik_inst.else_scope():
            src_stride.set_as(16)
            dst_stride.set_as(1)
        tik_inst.vnchwconv(False, False,
                           dst_addr_list, src_addr_list,
                           repeat_cnt, dst_stride, src_stride)


# pylint: disable=unnecessary-pass
def _transpose_by_1_vnchwconv():
    """
    do transpose by one time vnchwconv
    """

    pass


def _get_tiling_params(tiling_reg_list, ub_tiling):
    """
    get tiling parameters
    """

    # get select key
    tiling_reg_list[0].set_as(ub_tiling[0])
    # get needed core num
    tiling_reg_list[1].set_as(ub_tiling[1])
    # get ub offset
    tiling_reg_list[2].set_as(ub_tiling[2])
    # get axis 0 max size
    tiling_reg_list[3].set_as(ub_tiling[3])
    # get axis 1 max size
    tiling_reg_list[4].set_as(ub_tiling[4])
    # get per core process size
    tiling_reg_list[5].set_as(ub_tiling[5])
    # get per core loop count, the ub can hold data in one loop
    tiling_reg_list[6].set_as(ub_tiling[6])
    # get per core per left data
    tiling_reg_list[7].set_as(ub_tiling[7])
    # get last core process size
    tiling_reg_list[8].set_as(ub_tiling[8])
    # get last core loop count
    tiling_reg_list[9].set_as(ub_tiling[9])
    # get last core left data
    tiling_reg_list[10].set_as(ub_tiling[10])
    # get axis 0 loop count
    tiling_reg_list[11].set_as(ub_tiling[11])
    # get axis 0 left
    tiling_reg_list[12].set_as(ub_tiling[12])
    # get axis 0
    tiling_reg_list[13].set_as(ub_tiling[13])
    # get axis 1
    tiling_reg_list[14].set_as(ub_tiling[14])
    # get axis 2
    tiling_reg_list[15].set_as(ub_tiling[15])


def fp32_1_0_transpose(tik_inst, block_idx, trans_params):
    """
    transpose process for permute (1, 0)
    """

    data_in, data_out, ub_input, ub_tiling, tiling_reg_list = trans_params

    _get_tiling_params(tiling_reg_list, ub_tiling)
    # rename tiling parameters
    need_core_num = tiling_reg_list[1]
    ub_offset = tiling_reg_list[2]
    max_sub_h_size = tiling_reg_list[3]
    max_sub_w_size = tiling_reg_list[4]
    per_core_col_size = tiling_reg_list[5]
    per_core_loop_cnt = tiling_reg_list[6]
    per_core_left_data = tiling_reg_list[7]
    last_core_loop_cnt = tiling_reg_list[9]
    last_core_left_data = tiling_reg_list[10]
    h_loop_cnt = tiling_reg_list[11]
    h_left = tiling_reg_list[12]
    axis_0 = tiling_reg_list[13]
    axis_1 = tiling_reg_list[14]
    axis_2 = tiling_reg_list[15]
    # check whether axis_1 is block align or not
    data_size_one_block = _get_elment_cnt_one_block(data_in.dtype)

    with tik_inst.if_scope(block_idx < need_core_num):

        def _fp32_1_0_t_mc_on_1(loop_cnt, left_size):
            """
            detail process for permute (1, 0)
            """

            def _fp32_vnchwconv_process(axis_0_index, h_loop_idx, h_size):
                """
                do transpose by vnchwconv
                """

                def _fp32_inner_vnchwconv(col_lp_idx, col_size):
                    """
                    inner vnchwconv
                    """

                    # move data in
                    in_offset = (block_idx * per_core_col_size + col_lp_idx * max_sub_w_size +
                                 h_loop_idx * max_sub_h_size * axis_2 +
                                 axis_0_index * axis_1 * axis_2)
                    data_in_info = (h_size, col_size, axis_1, axis_2, in_offset)
                    _data_move_in_mc_on_w(tik_inst, ub_input, data_in, data_in_info)

                    # for this case, data_move will move in one more block
                    with tik_inst.new_stmt_scope():
                        h_size_temp = tik_inst.Scalar("int64")
                        with tik_inst.if_scope(tik.all(axis_1 > data_size_one_block,
                                                       h_size % data_size_one_block > 0)):
                            h_size_temp.set_as(_ceil_div(h_size, data_size_one_block) *
                                               data_size_one_block)
                        with tik_inst.else_scope():
                            h_size_temp.set_as(h_size)
                        # transpose by vnchwconv
                        sub_hw_size = (h_size_temp, col_size)
                        _transpose_by_2_vnchwconv(tik_inst, ub_input[ub_offset],
                                                  ub_input, sub_hw_size)

                    # move data out
                    out_offset = ((block_idx * per_core_col_size + col_lp_idx * max_sub_w_size) *
                                  axis_1 + h_loop_idx * max_sub_h_size +
                                  axis_0_index * axis_1 * axis_2)
                    data_out_info = (h_size, col_size, axis_1, axis_2, out_offset)
                    _data_move_out_mc_on_w(tik_inst, data_out, ub_input[ub_offset], data_out_info)

                with tik_inst.for_range(0, loop_cnt) as lp_idx:
                    _fp32_inner_vnchwconv(lp_idx, max_sub_w_size)
                with tik_inst.if_scope(left_size > 0):
                    _fp32_inner_vnchwconv(loop_cnt, left_size)

            with tik_inst.for_range(0, axis_0) as axis_0_idx:
                with tik_inst.for_range(0, h_loop_cnt) as h_lp_idx:
                    _fp32_vnchwconv_process(axis_0_idx, h_lp_idx, max_sub_h_size)
                with tik_inst.if_scope(h_left > 0):
                    _fp32_vnchwconv_process(axis_0_idx, h_loop_cnt, h_left)

        with tik_inst.if_scope(block_idx == need_core_num - 1):
            _fp32_1_0_t_mc_on_1(last_core_loop_cnt, last_core_left_data)
        with tik_inst.else_scope():
            _fp32_1_0_t_mc_on_1(per_core_loop_cnt, per_core_left_data)


def fp32_0_1_transpose(tik_inst, block_idx, trans_params):
    """
    transpose process for permute (0, 1)
    """

    data_in, data_out, ub_input, ub_tiling, tiling_reg_list = trans_params

    _get_tiling_params(tiling_reg_list, ub_tiling)
    need_core_num = tiling_reg_list[1]
    max_sub_w_size = tiling_reg_list[4]
    per_core_col_size = tiling_reg_list[5]
    per_core_loop_cnt = tiling_reg_list[6]
    per_core_left_data = tiling_reg_list[7]
    last_core_loop_cnt = tiling_reg_list[9]
    last_core_left_data = tiling_reg_list[10]
    axis_0 = tiling_reg_list[14]
    axis_1 = tiling_reg_list[15]

    with tik_inst.if_scope(block_idx < need_core_num):

        def _fp32_0_1(core_loop_cnt, core_left_data):
            """
            detail process for permute (0, 1)
            """

            def _fp32_transpose_none(c_loop_index, data_len):
                """
                do transpose nothing
                """
                in_offset = block_idx * per_core_col_size + c_loop_index * max_sub_w_size
                data_pos_info = (data_len, in_offset, axis_0, axis_1)
                _data_move_in_no_transpose(tik_inst, ub_input, data_in, data_pos_info)
                _data_move_out_no_transpose(tik_inst, data_out, ub_input, data_pos_info)

            with tik_inst.for_range(0, core_loop_cnt) as c_lp_idx:
                _fp32_transpose_none(c_lp_idx, max_sub_w_size)
            with tik_inst.if_scope(core_left_data > 0):
                _fp32_transpose_none(core_loop_cnt, core_left_data)

        with tik_inst.if_scope(block_idx == need_core_num - 1):
            _fp32_0_1(last_core_loop_cnt, last_core_left_data)
        with tik_inst.else_scope():
            _fp32_0_1(per_core_loop_cnt, per_core_left_data)


# pylint: disable=too-many-statements
def fp32_not_last_dim_transpose(tik_inst, block_idx, trans_params):
    """
    transpose for not last dim
    """

    data_in, data_out, ub_input, ub_tiling, tiling_reg_list = trans_params
    _get_tiling_params(tiling_reg_list, ub_tiling)
    # rename tiling parameters
    select_key = tiling_reg_list[0]
    need_core_num = tiling_reg_list[1]
    ub_offset = tiling_reg_list[2]
    max_no_core_axis_size = tiling_reg_list[3]
    max_core_axis_size = tiling_reg_list[4]
    per_core_col_size = tiling_reg_list[5]
    per_core_loop_cnt = tiling_reg_list[6]
    per_core_left_data = tiling_reg_list[7]
    last_core_loop_cnt = tiling_reg_list[9]
    last_core_left_data = tiling_reg_list[10]
    no_core_loop_cnt = tiling_reg_list[11]
    no_core_left = tiling_reg_list[12]
    axis_0 = tiling_reg_list[13]
    axis_1 = tiling_reg_list[14]
    axis_2 = tiling_reg_list[15]

    with tik_inst.if_scope(block_idx < need_core_num):

        def _fp32_1_0_2_mc_on_2(loop_cnt, left_data):
            """
            detail process for permute (1, 0, 2) under multiple core on axis 2
            """

            def _fp32_mte_process(axis_0_index, h_lp_index, sub_h_size):
                """
                do transpose by mte for not last dim
                """

                def _fp32_inner_mte(w_lp_index, sub_w_size):
                    """
                    inner mte
                    """
                    # move data in
                    in_offset = (block_idx * per_core_col_size + axis_0_index * axis_1 * axis_2 +
                                 h_lp_index * max_no_core_axis_size * axis_2 +
                                 w_lp_index * max_core_axis_size)
                    data_in_inf = (sub_h_size, sub_w_size, axis_1, axis_2, in_offset)
                    _data_move_in_last_dim_be_one_block(tik_inst, ub_input, data_in, data_in_inf)

                    # move data out
                    out_offset = (block_idx * per_core_col_size + axis_0_index * axis_2 +
                                  h_lp_index * max_no_core_axis_size * axis_0 * axis_2 +
                                  w_lp_index * max_core_axis_size)
                    data_out_inf = (sub_h_size, sub_w_size, axis_0, axis_1, axis_2, out_offset)
                    _data_move_out_last_dim_be_one_block(tik_inst, data_out, ub_input, data_out_inf)

                with tik_inst.for_range(0, loop_cnt) as w_lp_idx:
                    _fp32_inner_mte(w_lp_idx, max_core_axis_size)
                with tik_inst.if_scope(left_data > 0):
                    _fp32_inner_mte(loop_cnt, left_data)

            with tik_inst.for_range(0, axis_0) as axis_0_idx:
                with tik_inst.for_range(0, no_core_loop_cnt) as h_lp_idx:
                    _fp32_mte_process(axis_0_idx, h_lp_idx, max_no_core_axis_size)
                with tik_inst.if_scope(no_core_left > 0):
                    _fp32_mte_process(axis_0_idx, no_core_loop_cnt, no_core_left)

        def _fp32_1_0_2_mc_on_1(loop_cnt, left_data):
            """
            detail process for permute (1, 0, 2) under multiple core on axis 1
            """

            def _fp32_mte_process_1(axis_0_index, w_lp_index, sub_w_size):
                """
                do transpose by mte for not last dim under multiple core on axis 1
                """

                def _fp32_inner_mte_1(h_lp_index, sub_h_size):
                    """
                    inner mte
                    """
                    # move data in
                    in_offset = ((block_idx * per_core_col_size + axis_0_index * axis_1 +
                                  h_lp_index * max_core_axis_size) * axis_2 +
                                 w_lp_index * max_no_core_axis_size)
                    data_in_inf = (sub_h_size, sub_w_size, axis_1, axis_2, in_offset)
                    _data_move_in_last_dim_be_one_block(tik_inst, ub_input, data_in, data_in_inf)

                    # move data out
                    out_offset = ((block_idx * per_core_col_size * axis_0 + axis_0_index +
                                   h_lp_index * max_core_axis_size * axis_0) * axis_2 +
                                  w_lp_index * max_no_core_axis_size)
                    data_out_inf = (sub_h_size, sub_w_size, axis_0, axis_1, axis_2, out_offset)
                    _data_move_out_last_dim_be_one_block(tik_inst, data_out, ub_input, data_out_inf)

                with tik_inst.for_range(0, loop_cnt) as h_lp_idx:
                    _fp32_inner_mte_1(h_lp_idx, max_core_axis_size)
                with tik_inst.if_scope(left_data > 0):
                    _fp32_inner_mte_1(loop_cnt, left_data)

            with tik_inst.for_range(0, axis_0) as axis_0_idx:
                with tik_inst.for_range(0, no_core_loop_cnt) as w_lp_idx:
                    _fp32_mte_process_1(axis_0_idx, w_lp_idx, max_no_core_axis_size)
                with tik_inst.if_scope(no_core_left > 0):
                    _fp32_mte_process_1(axis_0_idx, no_core_loop_cnt, no_core_left)

        def _fp32_1_0_2_mc_on_1_last_dim_lt_one_block(loop_cnt, left_data):
            """
            detail process for permute (1, 0, 2) under last dim less than one block
            """

            # two case:
            # 1. axis_0 * axis_2 > alloc_ub_size // 2, max_core_axis_size = 1
            # 2. axis_0 * axis_2 <= alloc_ub_size // 2, axis_0_loop_cnt = 0, max_core_axis_size != 1

            def _fp32_mte_process_lt_one_block(axis_1_lp_index, sub_axis_1):
                """
                do transpose for last dim less than one block
                """

                def _fp32_inner_last_dim_lt_one_block(axis_0_lp_index, sub_axis_0):
                    """
                    inner process of last dim less than one block
                    """

                    # move data in
                    in_offset = (block_idx * per_core_col_size +
                                 axis_1_lp_index * max_core_axis_size +
                                 axis_0_lp_index * max_no_core_axis_size * axis_1) * axis_2
                    data_pos_info = (sub_axis_1, sub_axis_0, axis_0, axis_1, axis_2, in_offset)
                    _data_move_in_last_dim_lt_one_block(tik_inst, ub_input, data_in, data_pos_info)

                    # do transpose
                    with tik_inst.new_stmt_scope():
                        temp_sub_axis_1 = tik_inst.Scalar("int64")
                        temp_sub_axis_0 = tik_inst.Scalar("int64")
                        data_size_one_block = _get_elment_cnt_one_block(data_in.dtype)
                        axis_1_0_2_size = axis_0 * axis_1 * axis_2
                        sub_axis_1_0_2_size = sub_axis_1 * sub_axis_0 * axis_2

                        # to avoid multiple core dirty data
                        with tik_inst.if_scope(tik.all(sub_axis_1_0_2_size < data_size_one_block,
                                                       axis_1_0_2_size > data_size_one_block)):
                            with tik_inst.if_scope(sub_axis_1 == 1):
                                temp_sub_axis_0.set_as(_ceil_div(data_size_one_block, axis_2))
                                temp_sub_axis_1.set_as(sub_axis_1)
                            with tik_inst.else_scope():
                                temp_sub_axis_0.set_as(sub_axis_0)
                                temp_sub_axis_1.set_as(_ceil_div(data_size_one_block,
                                                                 axis_0 * axis_2))
                        with tik_inst.else_scope():
                            temp_sub_axis_1.set_as(sub_axis_1)
                            temp_sub_axis_0.set_as(sub_axis_0)

                        sub_dim_size = (temp_sub_axis_1, temp_sub_axis_0, axis_2)
                        _transpose_by_2_vnchwconv_not_last_dim(tik_inst, ub_input[ub_offset],
                                                               ub_input, sub_dim_size)

                    # move data out
                    out_offset = ((block_idx * per_core_col_size +
                                   axis_1_lp_index * max_core_axis_size) * axis_0 +
                                  axis_0_lp_index * max_no_core_axis_size) * axis_2
                    data_pos_info = (sub_axis_1, sub_axis_0, axis_0, axis_1, axis_2, out_offset)
                    _data_move_out_last_dim_lt_one_block(tik_inst, data_out, ub_input[ub_offset],
                                                         data_pos_info)

                with tik_inst.for_range(0, no_core_loop_cnt) as axis_0_lp_idx:
                    _fp32_inner_last_dim_lt_one_block(axis_0_lp_idx, max_no_core_axis_size)
                with tik_inst.if_scope(no_core_left > 0):
                    _fp32_inner_last_dim_lt_one_block(no_core_loop_cnt, no_core_left)

            with tik_inst.for_range(0, loop_cnt) as axis_1_lp_idx:
                _fp32_mte_process_lt_one_block(axis_1_lp_idx, max_core_axis_size)
            with tik_inst.if_scope(left_data > 0):
                _fp32_mte_process_lt_one_block(loop_cnt, left_data)

        with tik_inst.if_scope(block_idx == need_core_num - 1):
            with tik_inst.if_scope(select_key == 1021):
                _fp32_1_0_2_mc_on_1(last_core_loop_cnt, last_core_left_data)
            with tik_inst.if_scope(select_key == 1022):
                _fp32_1_0_2_mc_on_2(last_core_loop_cnt, last_core_left_data)
            with tik_inst.if_scope(select_key == 1023):
                _fp32_1_0_2_mc_on_1_last_dim_lt_one_block(last_core_loop_cnt, last_core_left_data)
        with tik_inst.else_scope():
            with tik_inst.if_scope(select_key == 1021):
                _fp32_1_0_2_mc_on_1(per_core_loop_cnt, per_core_left_data)
            with tik_inst.if_scope(select_key == 1022):
                _fp32_1_0_2_mc_on_2(per_core_loop_cnt, per_core_left_data)
            with tik_inst.if_scope(select_key == 1023):
                _fp32_1_0_2_mc_on_1_last_dim_lt_one_block(per_core_loop_cnt, per_core_left_data)


def fp32_reverse_transpose(tik_inst, block_idx, trans_params):
    """
    transpose for reverse
    """

    data_in, data_out, ub_input, ub_tiling, tiling_reg_list = trans_params
    _get_tiling_params(tiling_reg_list, ub_tiling)
    # rename tiling parameters
    need_core_num = tiling_reg_list[1]
    max_no_core_axis_size = tiling_reg_list[3]
    no_core_loop_cnt = tiling_reg_list[11]
    no_core_left = tiling_reg_list[12]
    axis_0 = tiling_reg_list[13]
    axis_1 = tiling_reg_list[14]
    axis_2 = tiling_reg_list[15]
    data_size_one_block = _get_elment_cnt_one_block(data_in.dtype)

    def _fp32_2_1_0_reverse(axis_2_index, axis_1_index):
        """
        detail process for permute (2, 1, 0)
        """

        def _fp32_2_1_0_inner(axis_0_lp_index, sub_axis_0_size):
            """
            inner reverse process
            """
            in_offset = (axis_0_lp_index * max_no_core_axis_size * axis_1 * axis_2 +
                         axis_1_index * axis_2 + axis_2_index)
            repeat_stride = axis_1 * axis_2 // data_size_one_block - 1
            with tik_inst.if_scope(tik.all(axis_1 * axis_2 % data_size_one_block == 0,
                                           repeat_stride <= MTE_STRIDES)):
                # the repeat time of mte should be less than 4095
                rp_lp_cnt = sub_axis_0_size // REPEAT_LIMIT_MTE
                sub_axis_0_left = sub_axis_0_size % REPEAT_LIMIT_MTE
                with tik_inst.for_range(0, rp_lp_cnt) as rp_lp_idx:
                    with tik_inst.if_scope(repeat_stride == 0):
                        tik_inst.data_move(
                            ub_input[rp_lp_idx * REPEAT_LIMIT_MTE * data_size_one_block],
                            data_in[in_offset + rp_lp_idx * REPEAT_LIMIT_MTE * axis_1 * axis_2],
                            0, 1, REPEAT_LIMIT_MTE, 0, 0)
                    with tik_inst.else_scope():
                        tik_inst.data_move(
                            ub_input[rp_lp_idx * REPEAT_LIMIT_MTE * data_size_one_block],
                            data_in[in_offset + rp_lp_idx * REPEAT_LIMIT_MTE * axis_1 * axis_2],
                            0, REPEAT_LIMIT_MTE, 1, repeat_stride, 0)
                with tik_inst.if_scope(sub_axis_0_left > 0):
                    with tik_inst.if_scope(repeat_stride == 0):
                        tik_inst.data_move(
                            ub_input[rp_lp_cnt * REPEAT_LIMIT_MTE * data_size_one_block],
                            data_in[in_offset + rp_lp_cnt * REPEAT_LIMIT_MTE * axis_1 * axis_2],
                            0, 1, sub_axis_0_left, 0, 0)
                    with tik_inst.else_scope():
                        tik_inst.data_move(
                            ub_input[rp_lp_cnt * REPEAT_LIMIT_MTE * data_size_one_block],
                            data_in[in_offset + rp_lp_cnt * REPEAT_LIMIT_MTE * axis_1 * axis_2],
                            0, sub_axis_0_left, 1, repeat_stride, 0)
            with tik_inst.else_scope():
                with tik_inst.for_range(0, sub_axis_0_size) as sub_axis_0_idx:
                    tik_inst.data_move(
                        ub_input[sub_axis_0_idx * data_size_one_block],
                        data_in[in_offset + sub_axis_0_idx * axis_1 * axis_2],
                        0, 1, 1, 0, 0)

            # move data out
            out_offset = (axis_2_index * axis_1 * axis_0 + axis_1_index * axis_0 +
                          axis_0_lp_index * max_no_core_axis_size)
            with tik_inst.for_range(0, sub_axis_0_size) as sub_axis_0_idx_1:
                tik_inst.data_move(data_out[out_offset + sub_axis_0_idx_1],
                                   ub_input[sub_axis_0_idx_1 * data_size_one_block],
                                   0, 1, 1, 0, 0)

        with tik_inst.for_range(0, no_core_loop_cnt) as axis_0_lp_idx:
            _fp32_2_1_0_inner(axis_0_lp_idx, max_no_core_axis_size)
        with tik_inst.if_scope(no_core_left > 0):
            _fp32_2_1_0_inner(no_core_loop_cnt, no_core_left)

    with tik_inst.if_scope(block_idx < need_core_num):
        with tik_inst.for_range(0, axis_2) as axis_2_idx:
            with tik_inst.for_range(0, axis_1) as axis_1_idx:
                _fp32_2_1_0_reverse(axis_2_idx, axis_1_idx)


def transpose_compute(tik_inst, tensor_list, pos_perm):
    """
    do transpose
    """

    data_in, data_tiling, data_out = tensor_list
    pos_perm = tuple(pos_perm)

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
        ub_tiling = tik_inst.Tensor(TILING_CTRL_PARAM[0], (TILING_CTRL_PARAM[1],),
                                    tik.scope_ubuf, "ub_tiling")
        ub_size = _get_max_element_in_ub(data_in.dtype, 1)
        ub_input = tik_inst.Tensor(data_in.dtype, (ub_size,), tik.scope_ubuf, "ub_input")

        # used to store tiling parameters
        tiling_reg_list = [tik_inst.Scalar(TILING_CTRL_PARAM[0])
                           for i in range(TILING_CTRL_PARAM[1])]

        # move tiling data to ub
        tik_inst.data_move(ub_tiling, data_tiling,
                           0, 1, TILING_CTRL_PARAM[1] // TILING_CTRL_PARAM[2], 0, 0)

        trans_params = [data_in, data_out, ub_input, ub_tiling, tiling_reg_list]
        if pos_perm in [(0, 1), (0, 1, 2)]:
            fp32_0_1_transpose(tik_inst, block_idx, trans_params)
        elif pos_perm in [(1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)]:
            fp32_1_0_transpose(tik_inst, block_idx, trans_params)
        elif pos_perm == (1, 0, 2):
            fp32_not_last_dim_transpose(tik_inst, block_idx, trans_params)
        elif pos_perm == (2, 1, 0):
            fp32_reverse_transpose(tik_inst, block_idx, trans_params)


@te.op.register_operator("TransposeD")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_INT, KERNEL_NAME)
def transpose_d(x, y, perm, kernel_name="transpose_d"):
    """
    do transpose by perm attribute

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, the dtype should be same as input
    perm : list or tuple
        permutation of the dimension of tensor
    kernel_name : str
        kernel name, default value is "transpose_d"

    Returns
    -------
    compile info
    """

    in_dtype = x.get("dtype").lower()
    dst_dtype = y.get("dtype").lower()
    pos_perm = _get_positive_perm(perm)

    # check input parameters valid or not
    input_params = (in_dtype, dst_dtype, pos_perm)
    _check_input_params(input_params)

    # initial Tik
    tik_inst = tik.Tik()
    # define input and output tensors
    data_in = tik_inst.Tensor(in_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_in")
    data_tiling = tik_inst.Tensor(TILING_CTRL_PARAM[0], (TILING_CTRL_PARAM[1],),
                                  tik.scope_gm, "data_tiling")
    data_out = tik_inst.Tensor(in_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "data_out")

    # do transfer
    tensor_list = [data_in, data_tiling, data_out]
    transpose_compute(tik_inst, tensor_list, pos_perm)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[data_in], outputs=[data_out], flowtable=[data_tiling])

    # send compile information to tiling module
    ub_size = _get_max_element_in_ub(data_in.dtype, 1)
    te.op.add_compile_info("vars",
                           {"ub_size": ub_size, "core_num": CORE_NUM,
                            "perm": pos_perm, "dtype": in_dtype})
    return {"compile_info": te.op.get_compile_info()}
