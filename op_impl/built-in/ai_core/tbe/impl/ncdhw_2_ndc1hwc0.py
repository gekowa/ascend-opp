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
ncdhw_2_ndc1hwc0
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
# threshold for mte
MTE_THRESHOLD = 3968
# mask value
MASK_128 = 128
# float16/32 type list
TYPE_FLOAT_LIST = ("float16",)
# used for vnchwconv
VNC_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


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


def _get_max_element_in_ub(col_size, in_dtype):
    """
    get the up limit elements in UB
    """

    if in_dtype.lower() == "float16":
        byte_len = 2
    elif in_dtype.lower() == "float32":
        byte_len = 4

    # the unit is Byte
    in_ub_size = ((col_size - 1) // 16 + 1) * 16 * 16 * byte_len
    ub_upper_limit = UB_SIZE // 2
    if ub_upper_limit > (248 * 1024 // 2):
        ub_upper_limit = 248 * 1024 // 2

    if in_ub_size > ub_upper_limit:
        element_size = (ub_upper_limit // byte_len)
    else:
        element_size = (in_ub_size // byte_len)

    return element_size


def _check_input_params(input_params):
    """
    to the check whether the input parameters is valid or not
    """

    in_shape, dst_shape, in_dtype, dst_dtype, src_format, dst_format = input_params
    check_list = TYPE_FLOAT_LIST

    if in_dtype != dst_dtype:
        raise RuntimeError("The input and output dtype should be same!")
    if not (src_format.upper() == "NCDHW" and
            dst_format.upper() == "NDC1HWC0"):
        raise RuntimeError("The src_format must be NCDHW and"
                           " dst_format must be NDC1HWC0!")

    check_dtype(in_dtype, check_list)
    check_shape(in_shape, min_rank=5, max_rank=5)
    check_shape(dst_shape)


# pylint: disable=too-many-locals,too-many-statements
def _multi_core_on_hw(tik_inst, data_in, data_out, shape_in):
    """
    do ncdhw to ndc1hwc0 transfer by multiple core on axis h and w
    """

    axis_n, axis_c, axis_d, axis_h, axis_w = shape_in
    hw_size = axis_h * axis_w
    dhw_size = axis_d * hw_size

    # each core process certain hw lines
    core_num = _ceil_div(hw_size, _ceil_div(hw_size, CORE_NUM))
    per_core_hw_cnt = _ceil_div(hw_size, core_num) // C0_LEN * C0_LEN
    # if per_core_hw_cnt is equal to 0, set core_num as 1
    if per_core_hw_cnt == 0:
        per_core_hw_cnt = 1
        core_num = 1
    left_hw_cnt = hw_size - per_core_hw_cnt * (core_num - 1)
    axis_c1 = _ceil_div(axis_c, C0_LEN)
    c_left = axis_c % C0_LEN

    # split the UB into two parts, and to load 16 axis_h*axis_w each time
    ub_size = _get_max_element_in_ub(hw_size, data_in.dtype)
    ub_col_size = ub_size // C0_LEN // C0_LEN * C0_LEN
    if ub_col_size == 0:
        raise RuntimeError("The UB is too small!")

    # alloc input and output ub
    in_ub = tik_inst.Tensor(data_in.dtype, (ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(data_in.dtype, (ub_size,),
                             name="out_ub", scope=tik.scope_ubuf)

    with tik_inst.for_range(0, core_num, block_num=core_num) as block_idx:
        with tik_inst.for_range(0, axis_n) as n_idx:
            with tik_inst.for_range(0, axis_d) as d_idx:

                # pylint: disable=too-many-locals,too-many-statements
                def _hw_transfer_process(hw_len):
                    """
                    process of hw transfer
                    """

                    def _inner_transfer_vnchwconv_hw(c1_index, c0_line):
                        """
                        do transfer by vnchwconv under multiple core on hw
                        """

                        def _vnchwconv_process(in_offset, out_offset,
                                               in_block, out_block):
                            """
                            the vnchwconv process
                            """

                            with tik_inst.for_range(0, c0_line) as c0_idx:
                                tik_inst.data_move(
                                    in_ub[c0_idx * ub_col_size],
                                    data_in[in_offset + c0_idx * dhw_size],
                                    0, 1, in_block, 0, 0)

                            src_addr_list = [in_ub[ub_col_size * i] for i in VNC_IDX_LIST]
                            dst_addr_list = [out_ub[C0_LEN * i] for i in VNC_IDX_LIST]
                            repeat_cnt = in_block
                            src_stride = 0 if repeat_cnt == 1 else 1
                            dst_stride = 0 if repeat_cnt == 1 else 16

                            tik_inst.vnchwconv(False, False, dst_addr_list,
                                               src_addr_list,
                                               repeat_cnt,
                                               dst_stride, src_stride)

                            tik_inst.data_move(data_out[out_offset], out_ub, 0, 1, out_block, 0, 0)

                        if hw_len > ub_col_size:
                            buf_loop = hw_len // ub_col_size
                            hw_len_left = hw_len % ub_col_size

                            with tik_inst.for_range(0, buf_loop) as buf_lp_idx:
                                burst_len = _ceil_div(ub_col_size, C0_LEN)
                                input_offset = (block_idx * per_core_hw_cnt +
                                                ((c1_index * C0_LEN + n_idx * axis_c) * axis_d +
                                                 d_idx) * hw_size + buf_lp_idx * ub_col_size)
                                output_offset = (block_idx * per_core_hw_cnt +
                                                 ((n_idx * axis_d + d_idx) * axis_c1 +
                                                  c1_index) * hw_size +
                                                 buf_lp_idx * ub_col_size) * C0_LEN

                                _vnchwconv_process(input_offset, output_offset,
                                                   burst_len, ub_col_size)
                            if hw_len_left:
                                burst_len = _ceil_div(hw_len_left, C0_LEN)
                                input_offset = (block_idx * per_core_hw_cnt +
                                                ((c1_index * C0_LEN + n_idx * axis_c) * axis_d +
                                                 d_idx) * hw_size + buf_loop * ub_col_size)
                                output_offset = (block_idx * per_core_hw_cnt +
                                                 ((n_idx * axis_d + d_idx) * axis_c1 + c1_index) *
                                                 hw_size + buf_loop * ub_col_size) * C0_LEN

                                _vnchwconv_process(input_offset, output_offset,
                                                   burst_len, hw_len_left)
                        else:
                            burst_len = _ceil_div(hw_len, C0_LEN)
                            input_offset = (block_idx * per_core_hw_cnt +
                                            ((c1_index * C0_LEN + n_idx * axis_c) *
                                             axis_d + d_idx) * hw_size)
                            output_offset = (block_idx * per_core_hw_cnt +
                                             ((n_idx * axis_d + d_idx) * axis_c1 +
                                              c1_index) * hw_size) * C0_LEN

                            _vnchwconv_process(input_offset, output_offset,
                                               burst_len, hw_len)

                    if axis_c1 > 1:
                        # if c is not times of 16, then c1-1
                        if c_left:
                            axis_c1_new = axis_c1 - 1
                        else:
                            axis_c1_new = axis_c1
                        with tik_inst.for_range(0, axis_c1_new) as c1_idx:
                            _inner_transfer_vnchwconv_hw(c1_idx, C0_LEN)
                        if c_left:
                            # set the 16-c_left lines to zero
                            _clean_ubuf(tik_inst, in_ub,
                                        ub_col_size * c_left,
                                        ub_col_size * (C0_LEN - c_left))
                            _inner_transfer_vnchwconv_hw(axis_c1_new, c_left)
                    else:
                        # set the 16-c_left lines to zero, only once
                        with tik_inst.if_scope(n_idx == 0):
                            _clean_ubuf(tik_inst, in_ub,
                                        ub_col_size * axis_c,
                                        ub_col_size * (C0_LEN - axis_c))
                        _inner_transfer_vnchwconv_hw(0, axis_c)

                with tik_inst.if_scope(block_idx == core_num - 1):
                    _hw_transfer_process(left_hw_cnt)
                with tik_inst.else_scope():
                    _hw_transfer_process(per_core_hw_cnt)


# pylint: disable=too-many-locals,too-many-statements
def _multi_core_on_d(tik_inst, data_in, data_out, shape_in):
    """
    do ncdhw to ndc1hwc0 transfer by multiple core on axis h and w
    """

    axis_n, axis_c, axis_d, axis_h, axis_w = shape_in
    hw_size = axis_h * axis_w
    dhw_size = axis_d * hw_size

    # each core process certain hw lines
    core_num = _ceil_div(axis_d, _ceil_div(axis_d, CORE_NUM))
    per_core_d_cnt = _ceil_div(axis_d, core_num)
    left_d_cnt = axis_d - per_core_d_cnt * (core_num - 1)
    axis_c1 = _ceil_div(axis_c, C0_LEN)
    c_left = axis_c % C0_LEN

    # split the UB into two parts, and to load 16 axis_h*axis_w each time
    ub_size = _get_max_element_in_ub(hw_size, data_in.dtype)
    ub_col_size = ub_size // C0_LEN // C0_LEN * C0_LEN

    # alloc input and output ub
    in_ub = tik_inst.Tensor(data_in.dtype, (ub_size,),
                            name="in_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(data_in.dtype, (ub_size,),
                             name="out_ub", scope=tik.scope_ubuf)

    with tik_inst.for_range(0, core_num, block_num=core_num) as block_idx:
        with tik_inst.for_range(0, axis_n) as n_idx:

            # pylint: disable=too-many-locals,too-many-statements
            def _d_transfer_process(d_len):
                """
                process of hw transfer
                """

                def _inner_transfer_vnchwconv_d(c1_index, c0_line):
                    """
                    do transfer by vnchwconv under multiple core on hw
                    """

                    def _vnchwconv_process(in_offset, out_offset,
                                           in_block, out_block):
                        """
                        the vnchwconv process
                        """

                        with tik_inst.for_range(0, c0_line) as c0_idx:
                            tik_inst.data_move(
                                in_ub[c0_idx * ub_col_size],
                                data_in[in_offset + c0_idx * dhw_size],
                                0, 1, in_block, 0, 0)

                        src_addr_list = [in_ub[ub_col_size * i] for i in
                                         VNC_IDX_LIST]
                        dst_addr_list = [out_ub[C0_LEN * i] for i in
                                         VNC_IDX_LIST]
                        repeat_cnt = in_block
                        src_stride = 0 if repeat_cnt == 1 else 1
                        dst_stride = 0 if repeat_cnt == 1 else 16

                        tik_inst.vnchwconv(False, False, dst_addr_list,
                                           src_addr_list,
                                           repeat_cnt,
                                           dst_stride, src_stride)

                        tik_inst.data_move(
                            data_out[out_offset],
                            out_ub,
                            0, 1, out_block, 0, 0)

                    if hw_size > ub_col_size:
                        buf_loop = hw_size // ub_col_size
                        hw_len_left = hw_size % ub_col_size

                        with tik_inst.for_range(0, buf_loop) as buf_lp_idx:
                            burst_len = _ceil_div(ub_col_size, C0_LEN)
                            input_offset = (((c1_index * C0_LEN + n_idx * axis_c) * axis_d + d_idx
                                             + block_idx * per_core_d_cnt) * hw_size +
                                            buf_lp_idx * ub_col_size)
                            output_offset = (((n_idx * axis_d + d_idx + block_idx * per_core_d_cnt)
                                              * axis_c1 + c1_index) * hw_size +
                                             buf_lp_idx * ub_col_size) * C0_LEN

                            _vnchwconv_process(input_offset, output_offset,
                                               burst_len, ub_col_size)
                        if hw_len_left:
                            burst_len = _ceil_div(hw_len_left, C0_LEN)
                            input_offset = (((c1_index * C0_LEN + n_idx * axis_c) * axis_d +
                                             d_idx + block_idx * per_core_d_cnt) * hw_size +
                                            buf_loop * ub_col_size)
                            output_offset = (((n_idx * axis_d + d_idx +
                                               block_idx * per_core_d_cnt) * axis_c1 + c1_index) *
                                             hw_size + buf_loop * ub_col_size) * C0_LEN

                            _vnchwconv_process(input_offset, output_offset,
                                               burst_len, hw_len_left)
                    else:
                        burst_len = _ceil_div(hw_size, C0_LEN)
                        input_offset = ((c1_index * C0_LEN + n_idx * axis_c) * axis_d +
                                        d_idx + block_idx * per_core_d_cnt) * hw_size
                        output_offset = ((n_idx * axis_d + d_idx + block_idx * per_core_d_cnt) *
                                         axis_c1 + c1_index) * hw_size * C0_LEN

                        _vnchwconv_process(input_offset, output_offset,
                                           burst_len, hw_size)

                with tik_inst.for_range(0, d_len) as d_idx:
                    if axis_c1 > 1:
                        # if c is not times of 16, then c1-1
                        if c_left:
                            axis_c1_new = axis_c1 - 1
                        else:
                            axis_c1_new = axis_c1
                        with tik_inst.for_range(0, axis_c1_new) as c1_idx:
                            _inner_transfer_vnchwconv_d(c1_idx, C0_LEN)
                        if c_left:
                            # set the 16-c_left lines to zero
                            _clean_ubuf(tik_inst, in_ub,
                                        ub_col_size * c_left,
                                        ub_col_size * (C0_LEN - c_left))
                            _inner_transfer_vnchwconv_d(axis_c1_new, c_left)
                    else:
                        # set the 16-c_left lines to zero, only once
                        with tik_inst.if_scope(n_idx == 0):
                            _clean_ubuf(tik_inst, in_ub,
                                        ub_col_size * axis_c,
                                        ub_col_size * (C0_LEN - axis_c))
                        _inner_transfer_vnchwconv_d(0, axis_c)

            with tik_inst.if_scope(block_idx == core_num - 1):
                _d_transfer_process(left_d_cnt)
            with tik_inst.else_scope():
                _d_transfer_process(per_core_d_cnt)


def ncdhw_2_ndc1hwc0_compute(tik_inst, data_in, data_out):
    """
    do ncdhw to ndc1hwc0 transfer
    """
    shape_in = [int(x) for x in data_in.shape[:]]
    _, _, axis_d, axis_h, axis_w = shape_in

    # avoid moving in data in few blocks
    if axis_h * axis_w > MTE_THRESHOLD and axis_d < CORE_NUM:
        _multi_core_on_hw(tik_inst, data_in, data_out, shape_in)
    else:
        _multi_core_on_d(tik_inst, data_in, data_out, shape_in)


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR,
                 REQUIRED_ATTR_STR, KERNEL_NAME)
def ncdhw_2_ndc1hwc0(src, dst, src_format, dst_format,
                     kernel_name="ncdhw_2_ndc1hwc0"):
    """
    used to transfer ncdhw to ndc1hwc0

    Parameters
    ----------
    src : dict
        shape and dtype of input
    dst : dict
        shape and dtype of output, should be same shape and type as input
    src_format : str
        input format, the value should be "NCDHW"
    dst_format : str
         output format, the value should be "NDC1HWC0"
    kernel_name : str
        kernel name, default value is "ncdhw_2_ndc1hwc0"

    Returns
    -------
    None
    """

    in_shape = src.get("shape")
    dst_shape = (in_shape[0], in_shape[2], _ceil_div(in_shape[1], C0_LEN),
                 in_shape[3], in_shape[4], C0_LEN)
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
    ncdhw_2_ndc1hwc0_compute(tik_inst, data_in, data_out)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[data_in], outputs=[data_out])
