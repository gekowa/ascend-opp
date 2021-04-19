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
nc1hwc0_2_nz
"""
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *


# UB size in byte
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# AICORE count
CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# C0 length
C0_LEN = 16
# repeat up limit for vector
REPEAT_LIMIT = 255
# repeat up limit for mte
MTE_REPEAT_LIMIT = 4095
# stride limit for mte
STRIDE_LIMIT = 65535
# mask value
MASK_128 = 128
# float16/32 type list
TYPE_FLOAT_LIST = ("float16",)
# int/uint8 type list
TYPE_CHAR_LIST = ("int8",)


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


def _fp16_2_char_conv(tik_inst, dst, src, dup_len, dst_offset=0):
    """
    do float16 to int8/uint8 transfer
    """

    par_count = 128

    if dup_len > 0:
        repeat = dup_len // par_count
        left_elem = dup_len % par_count
        repeat_loop = repeat // REPEAT_LIMIT
        repeat_left = repeat % REPEAT_LIMIT

        if repeat_loop > 0:
            with tik_inst.for_range(0, repeat_loop) as rpt_idx:
                tik_inst.vconv(MASK_128, "",
                               dst[rpt_idx * REPEAT_LIMIT * par_count +
                                   dst_offset],
                               src,
                               REPEAT_LIMIT, 1, 1, 4, 0)

        if repeat_left > 0:
            tik_inst.vconv(MASK_128, "",
                           dst[repeat_loop * REPEAT_LIMIT * par_count +
                               dst_offset],
                           src,
                           repeat_left, 1, 1, 4, 0)

        if left_elem > 0:
            tik_inst.vconv(left_elem, "",
                           dst[repeat * par_count + dst_offset],
                           src,
                           1, 1, 1, 4, 0)


def _get_max_element_in_ub(col_size, in_dtype):
    """
    get the up limit elements in UB
    """

    if in_dtype.lower() in TYPE_FLOAT_LIST:
        up_limit_size = UB_SIZE // 2
    elif in_dtype.lower() in TYPE_CHAR_LIST:
        # save 256 Byte for vector_dup zero
        up_limit_size = UB_SIZE - C0_LEN * C0_LEN

    if col_size < up_limit_size:
        up_limit_size = col_size

    return up_limit_size


def _get_max_element_in_ub_c1hw(col_size, block_size, in_dtype):
    """
    get the up limit elements in UB for multiple core on c1hw
    """

    if in_dtype.lower() in TYPE_FLOAT_LIST:
        up_limit_size = UB_SIZE // 2 // 2 // block_size
    else:
        # save 256 Byte to set zero for int8/uint8
        up_limit_size = (UB_SIZE - C0_LEN * C0_LEN) // 2 // block_size

    loop_cnt = col_size // up_limit_size
    left_col = col_size % up_limit_size

    if loop_cnt:
        in_ub_size = up_limit_size * block_size
        per_loop_col = up_limit_size
    else:
        in_ub_size = col_size * block_size
        per_loop_col = col_size
        left_col = 0

    return loop_cnt, left_col, per_loop_col, in_ub_size


# pylint: disable=too-many-locals,too-many-statements
def _multi_core_on_n(tik_inst, data_in, data_out, shape_in):
    """
    do nc1hwc0 to nz transfer by multiple core on axis n
    """

    axis_n, axis_c1, axis_h, axis_w, axis_c0 = shape_in

    multi_n_loop_cnt = axis_n // CORE_NUM
    multi_n_left_cnt = axis_n % CORE_NUM
    col_size = axis_c1 * axis_h * axis_w * axis_c0
    zero_loop_cnt = (_ceil_div(axis_n, C0_LEN) * C0_LEN - axis_n) // CORE_NUM
    zero_line_left = (_ceil_div(axis_n, C0_LEN) * C0_LEN - axis_n) % CORE_NUM

    in_ub_size = \
        _get_max_element_in_ub(col_size, data_in.dtype) // axis_c0 * axis_c0

    in_ub = tik_inst.Tensor(data_in.dtype, (in_ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    # tensor for setting in_ub to zero
    if data_in.dtype.lower() in TYPE_CHAR_LIST:
        zero_ub_size = C0_LEN * C0_LEN // 2
        zero_ub = tik_inst.Tensor("float16", (zero_ub_size,),
                                  name="zero_ub", scope=tik.scope_ubuf)
        _clean_ubuf(tik_inst, zero_ub, 0, zero_ub_size)

    with tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:

        def _n_transfer_process(n_line, n_loop_count):
            """
            process of n transfer
            """

            c1hwc0_size = axis_c1 * axis_h * axis_w * axis_c0
            is_large_in_ub = c1hwc0_size > in_ub_size

            def _inner_transfer_process_mte(ub_loop_index, data_size):
                """
                do nc1hwc0 to nz by mte
                """

                if not n_line:
                    input_offset = \
                        ub_loop_index * in_ub_size + \
                        (block_idx + n_loop_count * CORE_NUM) * c1hwc0_size
                    tik_inst.data_move(in_ub,
                                       data_in[input_offset],
                                       0, 1, data_size // axis_c0, 0, 0)

                repeat_cnt = data_size // axis_c0
                dst_strides = _ceil_div(axis_n, C0_LEN) * C0_LEN

                def _move_data_out(rp_lp_index, rp_cnt):
                    """
                    move data from ub to gm
                    """

                    output_offset = \
                        (ub_loop_index * (in_ub_size // axis_c0 * dst_strides)
                         + block_idx + n_loop_count * CORE_NUM +
                         rp_lp_index * dst_strides * MTE_REPEAT_LIMIT +
                         n_line) * axis_c0
                    ub_offset = rp_lp_index * MTE_REPEAT_LIMIT * axis_c0

                    # the strides limit is [1, 65535]
                    if (dst_strides - 1) <= STRIDE_LIMIT:
                        tik_inst.data_move(data_out[output_offset],
                                           in_ub[ub_offset],
                                           0, rp_cnt, 1, 0, dst_strides - 1)
                    else:
                        with tik_inst.for_range(0, rp_cnt) as idx:
                            tik_inst.data_move(
                                data_out[output_offset +
                                         idx * dst_strides * axis_c0],
                                in_ub[ub_offset + idx * axis_c0],
                                0, 1, 1, 0, 0)

                if repeat_cnt > MTE_REPEAT_LIMIT:
                    rp_lp_cnt = repeat_cnt // MTE_REPEAT_LIMIT
                    rp_left = repeat_cnt % MTE_REPEAT_LIMIT

                    with tik_inst.for_range(0, rp_lp_cnt) as rp_lp_idx:
                        _move_data_out(rp_lp_idx, MTE_REPEAT_LIMIT)
                    if rp_left:
                        _move_data_out(rp_lp_cnt, rp_left)

                else:
                    _move_data_out(0, repeat_cnt)

            if is_large_in_ub:
                ub_loop = c1hwc0_size // in_ub_size
                c1hwc0_left_size = c1hwc0_size % in_ub_size

                with tik_inst.for_range(0, ub_loop) as ub_lp_idx:
                    _inner_transfer_process_mte(ub_lp_idx, in_ub_size)

                if c1hwc0_left_size:
                    _inner_transfer_process_mte(ub_loop, c1hwc0_left_size)
            else:
                _inner_transfer_process_mte(0, c1hwc0_size)

        if multi_n_loop_cnt:
            with tik_inst.for_range(0, multi_n_loop_cnt) as n_lp_cnt:
                _n_transfer_process(0, n_lp_cnt)

        if multi_n_left_cnt:
            with tik_inst.if_scope(block_idx < multi_n_left_cnt):
                _n_transfer_process(0, multi_n_loop_cnt)

        if axis_n % C0_LEN:
            if data_in.dtype.lower() in TYPE_FLOAT_LIST:
                _clean_ubuf(tik_inst, in_ub, 0, in_ub_size)
            else:
                _fp16_2_char_conv(tik_inst, in_ub, zero_ub, in_ub_size)

            if zero_loop_cnt:
                with tik_inst.for_range(0, zero_loop_cnt) as z_lp_idx:
                    _n_transfer_process(axis_n, z_lp_idx)
            if zero_line_left:
                with tik_inst.if_scope(block_idx < zero_line_left):
                    _n_transfer_process(axis_n, zero_loop_cnt)


def _multi_core_on_c1hw(tik_inst, data_in, data_out, shape_in):
    """
    do nc1hwc0 to nz transfer by multiple core on axis c1, h and w
    """

    axis_n, axis_c1, axis_h, axis_w, axis_c0 = shape_in
    c1hw_size = axis_c1 * axis_h * axis_w

    # each core process certain c1hw lines
    core_num = _ceil_div(c1hw_size, _ceil_div(c1hw_size, CORE_NUM))
    per_core_c1hw_cnt = _ceil_div(c1hw_size, core_num)
    left_c1hw_cnt = c1hw_size - per_core_c1hw_cnt * (core_num - 1)
    n_16_align = _ceil_div(axis_n, C0_LEN) * C0_LEN
    nc0_align_size = n_16_align * axis_c0

    # to load xxx axis_n*axis_c0 each time
    c1hw_loop_cnt, left_c1hw_size, per_loop_cnt, ub_size = \
        _get_max_element_in_ub_c1hw(per_core_c1hw_cnt,
                                    nc0_align_size, data_in.dtype)

    # alloc input and output ub
    in_ub = tik_inst.Tensor(data_in.dtype, (ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(data_in.dtype, (ub_size,),
                             name="out_ub", scope=tik.scope_ubuf)
    # set out_ub to zero
    if data_in.dtype.lower() in TYPE_CHAR_LIST:
        zero_ub_size = C0_LEN * C0_LEN // 2
        zero_ub = tik_inst.Tensor("float16", (zero_ub_size,),
                                  name="zero_ub", scope=tik.scope_ubuf)
        _clean_ubuf(tik_inst, zero_ub, 0, zero_ub_size)
        _fp16_2_char_conv(tik_inst, out_ub, zero_ub, ub_size)
    else:
        _clean_ubuf(tik_inst, out_ub, 0, ub_size)

    with tik_inst.for_range(0, core_num, block_num=core_num) as block_idx:

        def _c1hw_transfer_process(c1hw_loop_count,
                                   c1hw_left_count, per_loop_size):
            """
            process of c1hw transfer
            """

            def _inner_transfer_mte_c1hw(c1hw_lp_index, col_size):
                """
                do transfer by mte under multiple core on c1hw
                """

                input_offset = (block_idx * per_core_c1hw_cnt +
                                c1hw_lp_index * per_loop_size) * axis_c0
                if (c1hw_size - col_size) <= STRIDE_LIMIT:
                    tik_inst.data_move(in_ub,
                                       data_in[input_offset],
                                       0,
                                       axis_n, col_size,
                                       c1hw_size - col_size, 0)
                else:
                    with tik_inst.for_range(0, axis_n) as idx:
                        tik_inst.data_move(
                            in_ub[idx * col_size * axis_c0],
                            data_in[input_offset + idx * c1hw_size * axis_c0],
                            0, 1, col_size, 0, 0)

                if col_size < axis_n:
                    with tik_inst.for_range(0, col_size) as col_idx:
                        if (col_size - 1) == 0:
                            tik_inst.data_move(
                                out_ub[col_idx * nc0_align_size],
                                in_ub[col_idx * axis_c0],
                                0, 1, axis_n, 0, 0)
                        elif (col_size - 1) <= STRIDE_LIMIT:
                            tik_inst.data_move(
                                out_ub[col_idx * nc0_align_size],
                                in_ub[col_idx * axis_c0],
                                0, axis_n, 1, col_size - 1, 0)
                        else:
                            with tik_inst.for_range(0, axis_n) as n_idx:
                                tik_inst.data_move(
                                    out_ub[col_idx * nc0_align_size +
                                           n_idx * axis_c0],
                                    in_ub[col_idx * axis_c0 +
                                          n_idx * col_size * axis_c0],
                                    0, 1, 1, 0, 0)
                else:
                    with tik_inst.for_range(0, axis_n) as col_idx:
                        if (n_16_align - 1) <= STRIDE_LIMIT:
                            tik_inst.data_move(
                                out_ub[col_idx * axis_c0],
                                in_ub[col_idx * col_size * axis_c0],
                                0, col_size, 1, 0, n_16_align - 1)
                        else:
                            with tik_inst.for_range(0, col_size) as n_idx:
                                tik_inst.data_move(
                                    out_ub[col_idx * axis_c0 +
                                           n_idx * n_16_align * axis_c0],
                                    in_ub[col_idx * col_size * axis_c0 +
                                          n_idx * axis_c0],
                                    0, 1, 1, 0, 0)

                output_offset = \
                    (block_idx * per_core_c1hw_cnt +
                     c1hw_lp_index * per_loop_size) * nc0_align_size
                tik_inst.data_move(data_out[output_offset],
                                   out_ub,
                                   0, 1, col_size * n_16_align, 0, 0)

            if c1hw_loop_count:
                with tik_inst.for_range(0, c1hw_loop_count) as c1hw_lp_idx:
                    _inner_transfer_mte_c1hw(c1hw_lp_idx, per_loop_size)
                if c1hw_left_count:
                    _inner_transfer_mte_c1hw(c1hw_loop_count, c1hw_left_count)
            else:
                _inner_transfer_mte_c1hw(0, per_loop_size)

        with tik_inst.if_scope(block_idx == core_num - 1):
            if left_c1hw_cnt:
                c1hw_loop_cnt_t, left_c1hw_size_t, per_loop_cnt_t, _ = \
                    _get_max_element_in_ub_c1hw(left_c1hw_cnt,
                                                nc0_align_size, data_in.dtype)
                _c1hw_transfer_process(c1hw_loop_cnt_t,
                                       left_c1hw_size_t, per_loop_cnt_t)
            else:
                _c1hw_transfer_process(c1hw_loop_cnt,
                                       left_c1hw_size, per_loop_cnt)
        with tik_inst.else_scope():
            _c1hw_transfer_process(c1hw_loop_cnt, left_c1hw_size, per_loop_cnt)


def _multi_core_on_nc1hw(tik_inst, data_in, data_out, shape_in):
    """
    do nc1hwc0 to nz transfer by multiple core on axis nc1hw
    """

    axis_n, _, _, _, axis_c0 = shape_in

    # each core process certain n lines
    n_16_align = _ceil_div(axis_n, C0_LEN) * C0_LEN
    core_num = _ceil_div(n_16_align, _ceil_div(n_16_align, CORE_NUM))
    per_core_n_cnt = _ceil_div(n_16_align, core_num)
    left_n_cnt = n_16_align - per_core_n_cnt * (core_num - 1)
    in_ub_size = \
        _get_max_element_in_ub(per_core_n_cnt * axis_c0,
                               data_in.dtype) // axis_c0 * axis_c0
    in_ub = tik_inst.Tensor(data_in.dtype, (in_ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    if data_in.dtype.lower() in TYPE_CHAR_LIST:
        zero_ub_size = C0_LEN * C0_LEN // 2
        zero_ub = tik_inst.Tensor("float16", (zero_ub_size,),
                                  name="zero_ub", scope=tik.scope_ubuf)
        _clean_ubuf(tik_inst, zero_ub, 0, zero_ub_size)

    with tik_inst.for_range(0, core_num, block_num=core_num) as block_idx:

        def _nc1hw_transfer_process(n_lines):
            """
            nc1hwc0 to nz transfer by multiple core on nc1hw
            """

            def _inner_transfer_mte_nc1hw(buf_lp_index, nc0_size):
                """
                do transfer by mte under multiple core on nc1hw
                """

                cur_nc0_size = (buf_lp_index + 1) * in_ub_size + \
                               block_idx * per_core_n_cnt * axis_c0
                input_offset = buf_lp_index * in_ub_size + \
                               block_idx * per_core_n_cnt * axis_c0
                with tik_inst.if_scope(cur_nc0_size > axis_n * axis_c0):
                    nc0_size_1 = axis_n * axis_c0 - input_offset
                    if data_in.dtype.lower() in TYPE_CHAR_LIST:
                        _fp16_2_char_conv(tik_inst, in_ub, zero_ub, nc0_size)
                    else:
                        _clean_ubuf(tik_inst, in_ub, 0, nc0_size)
                    with tik_inst.if_scope(nc0_size_1 >= axis_c0):
                        tik_inst.data_move(in_ub,
                                           data_in[input_offset],
                                           0, 1, nc0_size_1 // axis_c0, 0, 0)
                with tik_inst.else_scope():
                    tik_inst.data_move(in_ub,
                                       data_in[input_offset],
                                       0, 1, nc0_size // axis_c0, 0, 0)

                output_offset = buf_lp_index * in_ub_size + \
                                block_idx * per_core_n_cnt * axis_c0
                tik_inst.data_move(data_out[output_offset],
                                   in_ub,
                                   0, 1, nc0_size // axis_c0, 0, 0)

            is_larger_alloc_ub = n_lines * axis_c0 > in_ub_size
            if is_larger_alloc_ub:
                buf_loop = n_lines * axis_c0 // in_ub_size
                nc0_left = n_lines * axis_c0 % in_ub_size

                with tik_inst.for_range(0, buf_loop) as buf_lp_idx:
                    _inner_transfer_mte_nc1hw(buf_lp_idx, in_ub_size)
                if nc0_left:
                    _inner_transfer_mte_nc1hw(buf_loop, nc0_left)

            else:
                _inner_transfer_mte_nc1hw(0, n_lines * axis_c0)

        with tik_inst.if_scope(block_idx != core_num - 1):
            _nc1hw_transfer_process(per_core_n_cnt)
        with tik_inst.else_scope():
            _nc1hw_transfer_process(left_n_cnt)


def nc1hwc0_2_nz_compute(tik_inst, data_in, data_out):
    """
    do nc1hwc0 to nz transfer
    """
    shape_in = [int(x) for x in data_in.shape[:]]
    axis_n, axis_c1, axis_h, axis_w, _ = shape_in

    if axis_c1 * axis_h * axis_w == 1:
        _multi_core_on_nc1hw(tik_inst, data_in, data_out, shape_in)
    elif axis_n <= 3040:
        _multi_core_on_c1hw(tik_inst, data_in, data_out, shape_in)
    else:
        _multi_core_on_n(tik_inst, data_in, data_out, shape_in)


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR,
                 REQUIRED_ATTR_STR, KERNEL_NAME)
def nc1hwc0_2_nz(src, dst, src_format, dst_format, kernel_name="nc1hwc0_2_nz"):
    """
    used to transfer nc1hwc0 to fractal_z

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        shape and dtype of output, should be same shape and type as input
    src_format : str
        input format, the value should be "NC1HWC0"
    dst_format : str
         output format, the value should be "FRACTAL_Z"
    kernel_name : str
        kernel name, default value is "nc1hwc0_2_nz"

    Returns
    -------
    None
    """

    shape_in = src.get("shape")
    input_dtype = src.get("dtype").lower()
    dst_dtype = dst.get("dtype").lower()
    shape_out = (shape_in[1], shape_in[2], shape_in[3],
                 _ceil_div(shape_in[0], C0_LEN), C0_LEN, shape_in[-1])
    check_list = ("float16", "int8")

    if input_dtype != dst_dtype:
        raise RuntimeError("The input and output dtype should be same!")
    if not (src_format.upper() == "NC1HWC0" and
            dst_format.upper() == "FRACTAL_Z"):
        raise RuntimeError("The src_format must be NC1HWC0 and"
                           " dst_format must be FRACTAL_Z!")
    if not ((input_dtype == "int8" and shape_in[-1] == 2 * C0_LEN) or
            (input_dtype == "float16" and shape_in[-1] == C0_LEN)):
        raise RuntimeError("The last dim length of input tensor should "
                           "align with input dtype!")

    check_dtype(input_dtype, check_list)
    check_shape(shape_in, min_rank=5, max_rank=5)
    check_shape(shape_out)

    # initial Tik
    tik_inst = tik.Tik()
    # define input and output tensors
    data_in = tik_inst.Tensor(input_dtype, shape_in,
                              tik.scope_gm, "data_in")
    data_out = tik_inst.Tensor(input_dtype, shape_out,
                               tik.scope_gm, "data_out")

    # do transfer
    nc1hwc0_2_nz_compute(tik_inst, data_in, data_out)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[data_in], outputs=[data_out])
