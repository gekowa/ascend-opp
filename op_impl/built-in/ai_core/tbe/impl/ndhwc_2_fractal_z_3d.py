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
ndhwc_2_fractal_z_3d
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


# pylint: disable=too-many-arguments
def _padding_short_c(tik_inst, dst_ub, src_ub, c_len, dhw_len, reg_list):
    """
    do dhwc to c1dhwc0 transfer by scalar
    """

    c_align_len = _ceil_div(c_len, C0_LEN) * C0_LEN
    reg_cnt = len(reg_list)

    def _inner_conv_process():
        """
        the convert process
        """

        reg_loop = c_len // reg_cnt
        c_left = c_len % reg_cnt

        with tik_inst.for_range(0, dhw_len) as dhw_idx:
            if reg_loop:
                with tik_inst.for_range(0, reg_loop) as reg_lp_idx:
                    for idx in REG_IDX_LIST:
                        src_offset = reg_lp_idx * reg_cnt + idx + dhw_idx * c_len
                        reg_list[idx].set_as(src_ub[src_offset])

                    for idx in REG_IDX_LIST:
                        dst_offset = reg_lp_idx * reg_cnt + idx + dhw_idx * c_align_len
                        dst_ub[dst_offset].set_as(reg_list[idx])

            if c_left:
                for idx in REG_IDX_LIST[:c_left]:
                    src_offset = reg_loop * reg_cnt + idx + dhw_idx * c_len
                    reg_list[idx].set_as(src_ub[src_offset])

                for idx in REG_IDX_LIST[:c_left]:
                    dst_offset = reg_loop * reg_cnt + idx + dhw_idx * c_align_len
                    dst_ub[dst_offset].set_as(reg_list[idx])

    _inner_conv_process()


# pylint: disable=too-many-arguments
def _padding_long_c(tik_inst, dst_ub, src_ub, c_len, reg_list):
    """
    do c to c1c0 transfer
    """

    c0_cnt_in_c = c_len // C0_LEN
    c_left = c_len % C0_LEN
    reg_cnt = len(reg_list)

    if src_ub.dtype.lower() == "float16":
        block_factor = 1
    elif src_ub.dtype.lower() == "float32":
        block_factor = 2

    if c0_cnt_in_c:
        tik_inst.data_move(dst_ub, src_ub, 0, 1, c0_cnt_in_c * block_factor, 0, 0)

    if c_left:
        reg_loop = c_left // reg_cnt
        left_cnt = c_left % reg_cnt

        if reg_loop:
            with tik_inst.for_range(0, reg_loop) as reg_lp_idx:
                for idx in REG_IDX_LIST:
                    src_offset = reg_lp_idx * reg_cnt + idx + c0_cnt_in_c * C0_LEN
                    reg_list[idx].set_as(src_ub[src_offset])

                for idx in REG_IDX_LIST:
                    dst_offset = reg_lp_idx * reg_cnt + idx + c0_cnt_in_c * C0_LEN
                    dst_ub[dst_offset].set_as(reg_list[idx])

        if left_cnt:
            for idx in REG_IDX_LIST[:c_left]:
                src_offset = reg_loop * reg_cnt + idx + c0_cnt_in_c * C0_LEN
                reg_list[idx].set_as(src_ub[src_offset])

            for idx in REG_IDX_LIST[:c_left]:
                dst_offset = reg_loop * reg_cnt + idx + c0_cnt_in_c * C0_LEN
                dst_ub[dst_offset].set_as(reg_list[idx])


def _check_input_params(input_params):
    """
    to the check whether the input parameters is valid or not
    """

    in_shape, dst_shape, in_dtype, dst_dtype, src_format, dst_format = input_params
    check_list = TYPE_FLOAT_LIST

    if in_dtype != dst_dtype:
        raise RuntimeError("The input and output dtype should be same!")
    if not (src_format.upper() == "NDHWC" and
            dst_format.upper() == "FRACTAL_Z_3D"):
        raise RuntimeError("The src_format must be NDHWC and"
                           " dst_format must be FRACTAL_Z_3D!")

    check_dtype(in_dtype, check_list)
    check_shape(in_shape, min_rank=5, max_rank=5)
    check_shape(dst_shape)


# pylint: disable=too-many-locals,too-many-statements
def _multi_core_on_n(tik_inst, data_in, data_out, shape_in):
    """
    do ndhwc to fractal_z_3d transfer by multiple core on axis n
    """

    axis_n, axis_d, axis_h, axis_w, axis_c = shape_in
    hw_size = axis_h * axis_w
    ni_no_size = _ceil_div(axis_n, C0_LEN) * C0_LEN
    dhwc_size = func_reduce(lambda x, y: x * y, shape_in[1:])
    axis_c1 = _ceil_div(axis_c, C0_LEN)
    dtype_factor = _get_dtype_factor(data_in.dtype)
    dhwc_align_size = func_reduce(lambda x, y: x * y, (axis_d, hw_size, axis_c1, C0_LEN))
    # deduct 32 Bytes to avoid the repeat times of MTE bigger than 4095
    half_ub_size = (UB_SIZE - 4 * BLOCK_BYTE_SIZE) // 2 // dtype_factor
    is_c1dhwc0_bigger_half_ub = dhwc_align_size > half_ub_size

    # each core process certain dhwc lines
    core_num = _ceil_div(axis_n, _ceil_div(axis_n, CORE_NUM))
    per_core_n_cnt = _ceil_div(axis_n, core_num)
    left_n_cnt = axis_n - per_core_n_cnt * (core_num - 1)
    # count how many n lines need to pad zero
    zero_loop_cnt = (_ceil_div(axis_n, C0_LEN) * C0_LEN - axis_n) // core_num
    zero_line_left = (_ceil_div(axis_n, C0_LEN) * C0_LEN - axis_n) % core_num

    if axis_c % C0_LEN:
        ub_size = half_ub_size
        # this case will do data moving in ub
        out_ub = tik_inst.Tensor(data_in.dtype, (ub_size,), name="out_ub", scope=tik.scope_ubuf)
    else:
        ub_size = UB_SIZE // dtype_factor

    # alloc input and output ub
    in_ub = tik_inst.Tensor(data_in.dtype, (ub_size,), name="in_ub", scope=tik.scope_ubuf)
    # used for scalar operation
    reg_list = [tik_inst.Scalar(data_in.dtype) for i in REG_IDX_LIST]

    with tik_inst.for_range(0, core_num, block_num=core_num) as block_idx:

        # pylint: disable=too-many-locals,too-many-statements
        def _n_transfer_process(n_len):
            """
            process of hw transfer
            """

            with tik_inst.for_range(0, n_len) as n_idx:

                def _transfer_4_dhwc_less_half_ub():
                    """
                    the transfer process for dhwc less than half ub
                    """

                    input_offset = (block_idx * per_core_n_cnt + n_idx) * dhwc_size
                    tik_inst.data_move(in_ub,
                                       data_in[input_offset],
                                       0, 1, _ceil_div(dhwc_size, BLOCK_BYTE_SIZE // dtype_factor),
                                       0, 0)

                    if axis_c % C0_LEN:
                        with tik_inst.if_scope(n_idx == 0):
                            _clean_ubuf(tik_inst, out_ub, 0, dhwc_align_size)
                        _padding_short_c(tik_inst, out_ub, in_ub,
                                         axis_c, axis_d * hw_size, reg_list)

                    with tik_inst.for_range(0, axis_d) as d_idx:
                        with tik_inst.for_range(0, axis_c1) as c1_idx:
                            output_offset = (block_idx * per_core_n_cnt + n_idx +
                                             (d_idx * axis_c1 + c1_idx) *
                                             hw_size * ni_no_size) * C0_LEN
                            mid_offset = (d_idx * hw_size * axis_c1 + c1_idx) * C0_LEN

                            if axis_c % C0_LEN:
                                tik_inst.data_move(data_out[output_offset],
                                                   out_ub[mid_offset],
                                                   0, hw_size, dtype_factor // 2,
                                                   (axis_c1 - 1) * dtype_factor // 2,
                                                   (ni_no_size - 1) * dtype_factor // 2)
                            else:
                                tik_inst.data_move(data_out[output_offset],
                                                   in_ub[mid_offset],
                                                   0, hw_size, dtype_factor // 2,
                                                   (axis_c1 - 1) * dtype_factor // 2,
                                                   (ni_no_size - 1) * dtype_factor // 2)

                def _transfer_4_dhwc_larger_half_ub():
                    """
                    the transfer process for dhwc larger than half ub
                    """

                    with tik_inst.for_range(0, axis_d) as d_idx_1:
                        with tik_inst.for_range(0, hw_size) as hw_idx_1:

                            def _inner_c_transfer(sub_c, in_offset, out_offset):
                                """
                                the transfer for axis c
                                """

                                tik_inst.data_move(
                                    in_ub,
                                    data_in[in_offset],
                                    0, 1, _ceil_div(sub_c, BLOCK_BYTE_SIZE // dtype_factor), 0, 0)

                                c0_cnt_in_sub_c = _ceil_div(sub_c, C0_LEN)
                                repeat_stride = hw_size * ni_no_size * dtype_factor // 2
                                hwninoc0_size = hw_size * ni_no_size * C0_LEN
                                if sub_c % C0_LEN:
                                    with tik_inst.if_scope(n_idx == 0):
                                        _clean_ubuf(tik_inst, out_ub,
                                                    0, c0_cnt_in_sub_c * C0_LEN)
                                    _padding_long_c(tik_inst, out_ub, in_ub, sub_c, reg_list)
                                    temp_ub = out_ub
                                else:
                                    temp_ub = in_ub

                                if c0_cnt_in_sub_c <= 4095 and repeat_stride <= 65536:
                                    tik_inst.data_move(data_out[out_offset],
                                                       temp_ub,
                                                       0, c0_cnt_in_sub_c, dtype_factor // 2,
                                                       0, repeat_stride - dtype_factor // 2)
                                else:
                                    with tik_inst.for_range(0, c0_cnt_in_sub_c) as c0_idx:
                                        tik_inst.data_move(
                                            data_out[out_offset + c0_idx * hwninoc0_size],
                                            temp_ub[c0_idx * C0_LEN],
                                            0, 1, dtype_factor // 2, 0, 0)

                            if axis_c < ub_size:
                                input_offset_1 = ((block_idx * per_core_n_cnt + n_idx) * dhwc_size
                                                  + (d_idx_1 * hw_size + hw_idx_1) * axis_c)
                                output_offset_1 = (block_idx * per_core_n_cnt + n_idx +
                                                   (d_idx_1 * axis_c1 * hw_size + hw_idx_1) *
                                                   ni_no_size) * C0_LEN
                                _inner_c_transfer(axis_c, input_offset_1, output_offset_1)

                            else:
                                c_lp_cnt = axis_c // ub_size
                                c_left = axis_c % ub_size
                                with tik_inst.for_range(0, c_lp_cnt) as c_lp_idx:
                                    input_offset_1 = ((block_idx * per_core_n_cnt + n_idx) *
                                                      dhwc_size + (d_idx_1 * hw_size + hw_idx_1) *
                                                      axis_c + c_lp_idx * ub_size)
                                    output_offset_1 = (block_idx * per_core_n_cnt + n_idx +
                                                       (d_idx_1 * axis_c1 * hw_size + hw_idx_1) *
                                                       ni_no_size + c_lp_idx * (ub_size // C0_LEN)
                                                       * hw_size * ni_no_size) * C0_LEN
                                    _inner_c_transfer(ub_size, input_offset_1, output_offset_1)
                                if c_left:
                                    input_offset_1 = ((block_idx * per_core_n_cnt + n_idx) *
                                                      dhwc_size + (d_idx_1 * hw_size + hw_idx_1) *
                                                      axis_c + c_lp_cnt * ub_size)
                                    output_offset_1 = (block_idx * per_core_n_cnt + n_idx +
                                                       (d_idx_1 * axis_c1 * hw_size + hw_idx_1) *
                                                       ni_no_size + c_lp_cnt * (ub_size // C0_LEN)
                                                       * hw_size * ni_no_size) * C0_LEN
                                    _inner_c_transfer(c_left, input_offset_1, output_offset_1)

                if is_c1dhwc0_bigger_half_ub:
                    _transfer_4_dhwc_larger_half_ub()
                else:
                    _transfer_4_dhwc_less_half_ub()

        with tik_inst.if_scope(block_idx == core_num - 1):
            _n_transfer_process(left_n_cnt)
        with tik_inst.else_scope():
            _n_transfer_process(per_core_n_cnt)

        if axis_n % C0_LEN:

            def _padding_ni_no_process(z_lp_cnt, z_lp_index):
                """
                set the left size in one ninoc0 cube to zero
                """
                hwc0_size = hw_size * C0_LEN
                if hwc0_size > ub_size:
                    _clean_ubuf(tik_inst, in_ub, 0, ub_size)
                    hw_lp_cnt = hwc0_size // ub_size
                    repeat_time = ub_size // C0_LEN
                    hw_left_size = hwc0_size % ub_size // C0_LEN
                else:
                    _clean_ubuf(tik_inst, in_ub, 0, hw_size * C0_LEN)
                    hw_lp_cnt = 1
                    repeat_time = hw_size
                    hw_left_size = 0

                def _padding_ni_no_cube(out_offset):
                    """
                    set the left size in one ninoc0 cube to zero
                    """
                    with tik_inst.for_range(0, hw_lp_cnt) as hw_lp_idx:
                        tik_inst.data_move(data_out[out_offset + hw_lp_idx * ni_no_size * C0_LEN],
                                           in_ub,
                                           0, repeat_time, dtype_factor // 2,
                                           0, (ni_no_size - 1) * dtype_factor // 2)
                    if hw_left_size:
                        tik_inst.data_move(data_out[out_offset + hw_lp_cnt * ni_no_size * C0_LEN],
                                           in_ub,
                                           0, hw_left_size, dtype_factor // 2,
                                           0, (ni_no_size - 1) * dtype_factor // 2)

                with tik_inst.for_range(0, axis_d) as d_idx:
                    with tik_inst.for_range(0, axis_c1) as c1_idx:
                        output_offset = \
                            (axis_n + block_idx * z_lp_cnt + z_lp_index +
                             (d_idx * axis_c1 + c1_idx) * hw_size * ni_no_size) * C0_LEN
                        _padding_ni_no_cube(output_offset)

            if zero_loop_cnt:
                with tik_inst.for_range(0, zero_loop_cnt) as z_lp_idx:
                    _padding_ni_no_process(zero_loop_cnt, z_lp_idx)

            if zero_line_left:
                with tik_inst.if_scope(block_idx < zero_line_left):
                    _padding_ni_no_process(1, core_num * zero_loop_cnt)


def ndhwc_2_fractal_z_3d_compute(tik_inst, data_in, data_out):
    """
    do ndhwc to fractal_z_3d transfer
    """

    shape_in = [int(x) for x in data_in.shape[:]]
    _multi_core_on_n(tik_inst, data_in, data_out, shape_in)


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR,
                 REQUIRED_ATTR_STR, KERNEL_NAME)
def ndhwc_2_fractal_z_3d(src, dst, src_format, dst_format,
                         kernel_name="ndhwc_2_fractal_z_3d"):
    """
    used to transfer ndhwc to fractal_z_3d

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        shape and dtype of output, should be same shape and type as input
    src_format : str
        input format, the value should be "NDHWC"
    dst_format : str
         output format, the value should be "FRACTAL_Z_3D"
    kernel_name : str
        kernel name, default value is "ndhwc_2_fractal_z_3d"

    Returns
    -------
    None
    """

    in_shape = src.get("shape")
    dst_shape = (in_shape[1], _ceil_div(in_shape[4], C0_LEN), in_shape[2],
                 in_shape[3], _ceil_div(in_shape[0], C0_LEN), C0_LEN, C0_LEN)
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
    ndhwc_2_fractal_z_3d_compute(tik_inst, data_in, data_out)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[data_in], outputs=[data_out])
