#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

padD
"""
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *
from te.utils.error_manager import error_manager_vector
import te.lang.dynamic
from impl.dynamic import pad_align
from impl.dynamic import pad_not_align
from impl.dynamic import pad_common

# maximum of gm
MAX_INT32 = 2**31 - 1
# byte of int32
INT32_BYTE = 4
# numbers in the block
INT32_BLOCK = 8


def pad_compute(obj):
    """
    obtain tik instance
    """
    obj.set_tik_instance()

    with obj.tik_instance.for_range(0, obj.max_core, block_num=obj.max_core) as blk_idx:
        # =====================
        # init tiling_params
        # =====================
        obj.tik_instance.data_move(obj.tiling_buf, obj.tiling_gm, 0, 1,
                                   obj.tiling_buf_size//INT32_BLOCK, 0, 0)
        pad_common.init_params(obj)

        # ======================
        # computation of main
        # ======================
        with obj.tik_instance.if_scope(obj.branch[0] == 1):
            pad_align.align_compute(obj, blk_idx)
        with obj.tik_instance.if_scope(obj.branch[0] == 0):
            pad_not_align.not_align_compute(obj, blk_idx)

    opt_config = {"out_of_bound_sync_check": True}
    obj.tik_instance.BuildCCE(kernel_name=obj.kernel_name,
                              inputs=[obj.input_gm],
                              outputs=[obj.output_gm],
                              flowtable=[obj.tiling_gm],
                              config=opt_config)
    te.op.add_compile_info("vars", {"ub_size": obj.buf_size, "core_num": obj.max_core,
                                    "padding": obj.ori_padding})

    return {"compile_info": te.op.get_compile_info()}


@te.op.register_operator("PadD")
def pad_d(input_x, output_x, paddings, kernel_name="pad_d"):
    """ calculating pad tensor by paddings parameters

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        shape and dtype of output
    paddings: list or tuple.
        For each dimension D of input, paddings[D, 0] indicates how many
        values to add
        before the contents of tensor in that dimension, and paddings[D, 1]
        indicates
        how many values to add after the contents of tensor in that dimension.
    kernel_name : str
        cce kernel name, default value is "pad_d"

    Returns
    -------
    None.
    """
    in_shape = list(input_x.get("shape"))
    pads = []
    for i in paddings:
        pads.append(list(i))
    src_dtype = input_x.get("dtype").lower()
    dst_dtype = output_x.get("dtype").lower()

    if len(in_shape) != len(pads):
        error_detail = "Length of input must be as same as paddings"
        error_manager_vector.raise_err_two_input_shpae_invalid("PadD", "input_x", "paddings",
                                                               error_detail)

    if src_dtype != dst_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("PadD", "src_dtype", "dst_dtype",
                                                              src_dtype, dst_dtype)

    if src_dtype not in ["float32", "float16", "int32"]:
        error_detail = "Only support float, float16 and int32"
        error_manager_vector.raise_err_two_input_dtype_invalid("PadD", "src_dtype", "dst_dtype",
                                                               error_detail)

    tik_obj = tik.Tik()
    pad = pad_common.PadInit(pads, src_dtype, kernel_name, tik_obj, True)
    return pad_compute(pad)

