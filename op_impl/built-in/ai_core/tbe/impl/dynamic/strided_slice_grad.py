#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

strided_slice_grad
"""
import te
from te import platform as tbe_platform
from te.utils.op_utils import *
from te import tik
from impl.dynamic import pad_align
from impl.dynamic import pad_not_align
from impl.dynamic import pad_common

# number of cores
INT32_BLOCK = 8


# pylint: disable=invalid-name, too-many-instance-attributes
# pylint: disable=too-many-arguments, useless-object-inheritance
# pylint: disable=too-many-locals, too-many-statements
# pylint: disable=attribute-defined-outside-init, unused-argument
# pylint: disable=attribute-defined-outside-init, chained-comparison


def _check_mask(input_mask, is_shrink=False):
    """ Check whether the value of the input mask is 0.

    Parameters
    ----------
    input_mask: int.
        value of the input mask.

    Returns
    -------
    None.
    """
    if is_shrink:
        if input_mask != 0 and input_mask != 2:
            raise RuntimeError("shrink_axis_mask only support 0/2 currently")
    elif input_mask != 0:
        raise RuntimeError("ellipsis_mask,new_axis_mask"
                           " only support 0 currently")


def grad_compute(obj, mask_list):
    """
    obtain tik instance
    """
    obj.set_tik_instance()

    with obj.tik_instance.for_range(0, obj.max_core, block_num=obj.max_core) as blk_idx:
        # =====================
        # init tiling_params
        # =====================
        obj.tik_instance.data_move(obj.tiling_buf, obj.tiling_gm, 0, 1,
                                   obj.tiling_buf_size // INT32_BLOCK, 0, 0)
        pad_common.init_params(obj)

        # ======================
        # computation of main
        # ======================
        with obj.tik_instance.if_scope(obj.branch[0] == 1):
            pad_align.align_compute(obj, blk_idx)
        with obj.tik_instance.if_scope(obj.branch[0] == 0):
            pad_not_align.not_align_compute(obj, blk_idx)

    opt_config = {"out_of_bound_sync_check": True}
    shape_size = 128
    shape_gm = obj.tik_instance.Tensor("int32", (shape_size,),
                                       name="shape_gm", scope=tik.scope_gm)
    begin_gm = obj.tik_instance.Tensor("int32", (shape_size,),
                                       name="begin_gm", scope=tik.scope_gm)
    end_gm = obj.tik_instance.Tensor("int32", (shape_size,),
                                     name="end_gm", scope=tik.scope_gm)
    strides_gm = obj.tik_instance.Tensor("int32", (shape_size,),
                                         name="strides_gm", scope=tik.scope_gm)

    obj.tik_instance.BuildCCE(kernel_name=obj.kernel_name,
                              inputs=[shape_gm, begin_gm, end_gm, strides_gm, obj.input_gm],
                              outputs=[obj.output_gm],
                              flowtable=[obj.tiling_gm],
                              config=opt_config)
    te.op.add_compile_info("vars", {"ubSize": obj.buf_size, "maxCore": obj.max_core,
                                    "begin_mask": mask_list[0], "end_mask": mask_list[1],
                                    "ellipsis_mask": mask_list[2], "new_axis_mask": mask_list[3],
                                    "shrink_axis_mask": mask_list[4]})

    return {"compile_info": te.op.get_compile_info()}


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@te.op.register_operator("StridedSliceGrad")
def strided_slice_grad(shape, begin, end, strides, dy, output, begin_mask=0,
                       end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                       kernel_name="strided_slice_grad"):
    """ Since `StridedSlice` cuts out pieces of its `input` which is size`shape_dy`, its gradient
    will have the same shape (which is passed here as `shape_x`). The gradient will be zero in any
    element that the slice does not select.

    Parameters
    ----------
    dy : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of out
    shape : list or tuple.
        shape of input
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification should shrink
        the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice_grad"

    Returns
    -------
    None.
    """
    dtype = dy.get("dtype").lower()
    check_dtype(dtype, ("float16", "float32", "int32"), param_name="dy")
    _check_mask(new_axis_mask)
    _check_mask(shrink_axis_mask, True)

    mask_list = (begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    tik_obj = tik.Tik()
    pads = []
    for i, _ in enumerate(range(len(dy.get("shape")))):
        pads.append([0, 0])
    grad = pad_common.PadInit(pads, dtype, kernel_name, tik_obj, False)
    return grad_compute(grad, mask_list)
