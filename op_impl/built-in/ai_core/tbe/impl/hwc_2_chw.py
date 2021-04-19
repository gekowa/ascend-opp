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

hwc_2_chw
"""


from te import tik
from te import platform as tbe_platform


# UB size in byte
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# AICORE count
CORE_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# C0 length
VNC_LEN = 16
# bytes in one block
BLOCK_BYTE_SIZE = 32
# repeat up limit for vector
REPEAT_LIMIT = 255
# mask value for float32
MASK_64 = 64
# float16/32 type list
TYPE_FLOAT_LIST = ("float32",)
# used for vnchwconv
VNC_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def _ceil_div(value_x, value_y):
    """
    do ceil division
    """
    return (value_x + value_y - 1) // value_y


def _floor_trunc(value_x, value_y):
    """
    do floor truncate
    """
    return value_x // value_y * value_y


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


# pylint: disable=too-many-locals,too-many-statements
def _multi_core_on_hw(tik_inst, data_in, data_out, shape_in):
    """
    do hwc to chw transfer by multiple core on axis hw
    """

    axis_hw, axis_c = shape_in
    hw_size = axis_hw
    dtype_len = _get_dtype_factor(data_in.dtype)
    ele_count_per_block = BLOCK_BYTE_SIZE // dtype_len
    # save 8kb to avoid repeat time of vnchwconv is larger than 255
    mod_ub_size = UB_SIZE
    if mod_ub_size > 248 * 1024:
        mod_ub_size = 248 * 1024
    need_ub_size = _floor_trunc(mod_ub_size // 2 // dtype_len, ele_count_per_block)

    # each core process certain hw'c lines
    core_num = _ceil_div(hw_size, _ceil_div(hw_size, CORE_NUM))
    # to make sure every core process hw is block align except last core
    sub_hw_per_loop = _floor_trunc(need_ub_size // VNC_LEN // axis_c, ele_count_per_block) * VNC_LEN
    per_core_hw_cnt = _floor_trunc(_ceil_div(hw_size, core_num), ele_count_per_block)
    last_core_hw_cnt = hw_size - per_core_hw_cnt * (core_num - 1)

    # alloc input and output ub
    in_ub = tik_inst.Tensor(data_in.dtype, (need_ub_size,), name="in_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(data_in.dtype, (need_ub_size,), name="out_ub", scope=tik.scope_ubuf)

    with tik_inst.for_range(0, core_num, block_num=core_num) as block_idx:

        # pylint: disable=too-many-locals,too-many-statements
        def _hw_transfer_process(sub_hw_len):
            """
            process of hw transfer
            """

            def _inner_hw_transfer(sub_hw_lp_idx, inner_hw_len):
                """
                inner hw transfer process
                """

                # move data to ubuf
                inner_hw_block_cnt = inner_hw_len // ele_count_per_block
                inner_hw_left = inner_hw_len % ele_count_per_block
                back_len = inner_hw_left - ele_count_per_block

                in_offset = (sub_hw_lp_idx * sub_hw_per_loop + block_idx * per_core_hw_cnt) * axis_c
                if inner_hw_block_cnt:
                    tik_inst.data_move(in_ub, data_in[in_offset],
                                       0, 1, inner_hw_block_cnt * axis_c, 0, 0)
                if inner_hw_left:
                    tik_inst.data_move(in_ub[inner_hw_block_cnt * ele_count_per_block * axis_c],
                                       data_in[in_offset +
                                               (inner_hw_block_cnt * ele_count_per_block + back_len)
                                               * axis_c],
                                       0, 1, axis_c, 0, 0)

                # do hwc to chw transfer
                inner_hw_len_1 = sub_hw_per_loop // VNC_LEN
                fp16_inner_hwc_len = inner_hw_len_1 * axis_c * 2
                fp16_in_ub = in_ub.reinterpret_cast_to("float16")
                fp16_out_ub = out_ub.reinterpret_cast_to("float16")
                # first vnchwconv
                src_addr_list = [fp16_in_ub[fp16_inner_hwc_len * i] for i in VNC_IDX_LIST]
                dst_addr_list = [fp16_out_ub[VNC_LEN * i] for i in VNC_IDX_LIST]
                repeat_cnt = _ceil_div(fp16_inner_hwc_len, VNC_LEN)
                src_stride = 0 if repeat_cnt == 1 else 1
                dst_stride = 0 if repeat_cnt == 1 else 16
                tik_inst.vnchwconv(False, False,
                                   dst_addr_list, src_addr_list,
                                   repeat_cnt, dst_stride, src_stride)
                # do hwc to chw transfer
                with tik_inst.for_range(0, inner_hw_len_1) as inner_hw_1_idx:
                    tik_inst.data_move(fp16_in_ub[inner_hw_1_idx * 2 * VNC_LEN],
                                       fp16_out_ub[inner_hw_1_idx * axis_c * 2 * VNC_LEN],
                                       0, axis_c, 2, 0, (inner_hw_len_1 - 1) * 2)
                # second vnchwconv
                src_addr_list = [fp16_in_ub[VNC_LEN * i] for i in VNC_IDX_LIST]
                dst_addr_list = [fp16_out_ub[fp16_inner_hwc_len * i] for i in VNC_IDX_LIST]
                repeat_cnt = _ceil_div(fp16_inner_hwc_len, VNC_LEN)
                src_stride = 0 if repeat_cnt == 1 else 16
                dst_stride = 0 if repeat_cnt == 1 else 1
                tik_inst.vnchwconv(False, False,
                                   dst_addr_list, src_addr_list,
                                   repeat_cnt, dst_stride, src_stride)

                # move hw in together
                with tik_inst.for_range(0, axis_c) as axis_c_idx:
                    with tik_inst.for_range(0, 2) as add_idx:
                        tik_inst.vadds(64, in_ub[axis_c_idx * 128 + add_idx * 64],
                                       out_ub[axis_c_idx * 8 + add_idx * axis_c * 8 * 8],
                                       0, 1, 1, axis_c, 8, 8)

                # move data to gm
                out_offset = sub_hw_lp_idx * sub_hw_per_loop + block_idx * per_core_hw_cnt
                if inner_hw_len % ele_count_per_block > 0:
                    if inner_hw_block_cnt > 0:
                        with tik_inst.for_range(0, axis_c) as c_idx:
                            tik_inst.data_move(data_out[out_offset + c_idx * hw_size],
                                               in_ub[c_idx * 128], 0, 1,
                                               inner_hw_len // ele_count_per_block,
                                               0, 0)
                    with tik_inst.for_range(0, axis_c) as c_idx1:
                        tik_inst.data_move(data_out[out_offset + c_idx1 * hw_size + back_len +
                                                    inner_hw_len // ele_count_per_block *
                                                    ele_count_per_block],
                                           in_ub[c_idx1 * 128 +
                                                 inner_hw_len // ele_count_per_block *
                                                 ele_count_per_block],
                                           0, 1, 1, 0, 0)

                else:
                    with tik_inst.for_range(0, axis_c) as c_idx:
                        tik_inst.data_move(data_out[out_offset + c_idx * hw_size],
                                           in_ub[c_idx * 128], 0, 1,
                                           _ceil_div(inner_hw_len, ele_count_per_block),
                                           0, 0)

            sub_hw_lp_cnt = sub_hw_len // sub_hw_per_loop
            sub_hw_left = sub_hw_len % sub_hw_per_loop

            with tik_inst.for_range(0, sub_hw_lp_cnt) as sub_hw_lp_idx:
                _inner_hw_transfer(sub_hw_lp_idx, sub_hw_per_loop)
            if sub_hw_left:
                _inner_hw_transfer(sub_hw_lp_cnt, sub_hw_left)

        with tik_inst.if_scope(block_idx == core_num - 1):
            _hw_transfer_process(last_core_hw_cnt)
        with tik_inst.else_scope():
            _hw_transfer_process(per_core_hw_cnt)


def hwc_2_chw_compute(tik_inst, data_in, data_out):
    """
    do hwc to chw transfer
    """

    shape_in = [int(x) for x in data_in.shape[:]]
    _multi_core_on_hw(tik_inst, data_in, data_out, shape_in)


def hwc_2_chw(in_shape, in_dtype, kernel_name="hwc_2_chw"):
    """
    used to transfer hwc to chw

    Parameters
    ----------
    src : dict
        shape and dtype of input
    perm : list
        permute of output dimension

    Returns
    -------
    None
    """

    dst_shape = (in_shape[1], in_shape[0])
    # initial Tik
    tik_inst = tik.Tik()
    # define input and output tensors
    data_in = tik_inst.Tensor(in_dtype, in_shape, tik.scope_gm, "data_in")
    data_out = tik_inst.Tensor(in_dtype, dst_shape, tik.scope_gm, "data_out")

    # do transfer
    hwc_2_chw_compute(tik_inst, data_in, data_out)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[data_in], outputs=[data_out])
