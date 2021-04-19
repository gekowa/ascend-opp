#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

strided slice
"""

from __future__ import absolute_import
import te.lang.dynamic
from topi.cce import util
from impl import common_util
from te.utils.op_utils import *
from .strided_slice import StridedSlice


# pylint: disable=locally-disabled,too-many-arguments,
# pylint: unused-argument,too-many-locals
@te.op.register_operator("Slice")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def slice(x, offsets, size, y, kernel_name="slice"):
    """
    algorithm: slice
    calculating: this operation extracts a slice of size size
                 from a tensor input
                 starting at the location specified by begin.

    Parameters
    ----------
    x: dict
        contains shape and dtype information of input tensor
    y: dict
        contains shape and dtype information of output tensor
    offsets: dict
        represents the index of the first value to select
    size: dict
        represents the shape of output tensor
    kernel_name: str
        cce kernel name, default value is "slice".

    Returns
    -------
    tik instance
    """
    # dynamic slice does not use offsets, end params.
    strided_slice_instance = StridedSlice(x, None, 0, 0, 0, 0, 0, kernel_name)
    strided_slice_instance.strided_slice()
    inst = strided_slice_instance.tik_instance
    opt_config = {"out_of_bound_sync_check": True}
    inst.BuildCCE(kernel_name=strided_slice_instance.kernel_name,
                  inputs=(strided_slice_instance.input_gm,
                          strided_slice_instance.begin_gm,
                          strided_slice_instance.end_gm),
                  outputs=(strided_slice_instance.output_gm,),
                  flowtable=[strided_slice_instance.tiling_param.tiling_gm],
                  config=opt_config,
                  enable_l2=False)

    te.op.add_compile_info("vars", {"block_dim": strided_slice_instance.aicore_num})
    return inst
