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
from te import tik
from impl import constant_util as constant

MAX_SIZE = 2 ** 31 - 1


def ceil_32bytes_align_count(count, dtype):
    type_size = common_util.get_data_size(dtype)
    block_count = math.ceil(count * type_size / constant.BLOCK_SIZE)
    return block_count * constant.BLOCK_SIZE // type_size


def _data_move(tik_instance: tik.Tik, dest: tik.Tensor, src: tik.Tensor, count):
    dtype_size = common_util.get_data_size(src.dtype)
    burst = math.ceil(count * dtype_size / constant.BLOCK_SIZE)
    tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)


# pylint: too-many-locals, too-many-statements, too-many-instance-attributes
class StridedSlice:
    # pylint: too-many-locals, too-many-statements, too-many-instance-attributes
    class TilingParam:
        def __init__(self, input_x_shape, inst: tik.Tik):
            """
            tiling param
            :param input_x_shape: input shape
            :param inst: tik instance
            """
            self.tik_instance = inst
            dtype = "int64"
            self.dtype = dtype
            # input_shape, output_shape, begin, end, stride
            tiling_gm_size = len(input_x_shape) * 5
            self.tiling_gm = inst.Tensor(dtype, (tiling_gm_size,), name="tiling_gm", scope=tik.scope_gm)

            def gen_shape(name, index):
                name += str(index)
                return inst.Scalar(dtype, name=name)

            self.input_shape = tuple(map(lambda x: gen_shape("input_dim", x[0]), enumerate(input_x_shape)))
            self.begin = tuple(map(lambda x: gen_shape("begin_", x[0]), enumerate(input_x_shape)))
            self.end = tuple(map(lambda x: gen_shape("end_", x[0]), enumerate(input_x_shape)))
            self.stride = tuple(map(lambda x: gen_shape("stride_", x[0]), enumerate(input_x_shape)))
            self.output_shape = tuple(map(lambda x: gen_shape("out_dim_", x[0]), enumerate(input_x_shape)))
            self.out_dim = inst.Scalar(dtype, name="out_dim")

        def init(self):
            with self.tik_instance.new_stmt_scope():
                need_ub_size = ceil_32bytes_align_count(self.tiling_gm.shape[0], self.dtype)
                ub = self.tik_instance.Tensor(self.dtype, (need_ub_size,), name="tiling_ub", scope=tik.scope_ubuf)
                _data_move(self.tik_instance, ub, self.tiling_gm, need_ub_size)
                items = (self.input_shape, self.output_shape, self.begin, self.end, self.stride)
                index = 0
                for item in items:
                    for value in item:
                        value.set_as(ub[index])
                        index += 1

            self.out_dim.set_as(1)
            for index, dim in enumerate(self.output_shape):
                dim.set_as(self.end[index] - self.begin[index])
                if index != len(self.output_shape) - 1:
                    self.out_dim.set_as(self.out_dim * dim)

    # pylint: disable=locally-disabled,too-many-arguments,
    # pylint: unused-argument,too-many-locals
    def __init__(self, input_x, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
                 kernel_name="strided_slice"):
        self.strides = strides
        self.begin_mask = begin_mask
        self.end_mask = end_mask
        self.ellipsis_mask = ellipsis_mask
        self.new_axis_mask = new_axis_mask
        self.shrink_axis_mask = shrink_axis_mask
        self.kernel_name = kernel_name

        inst = tik.Tik()
        self.tik_instance = inst
        self.tik_profiling = tik.Dprofile()
        self.tiling_param = self.TilingParam(input_x.get("shape"), inst)
        self.dtype = input_x.get("dtype").lower()
        self.dtype_size = common_util.get_data_size(self.dtype)
        self.input_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="input_gm", scope=tik.scope_gm)
        self.begin_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="begin_gm", scope=tik.scope_gm)
        self.end_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="end_gm", scope=tik.scope_gm)
        self.strides_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="strides_gm", scope=tik.scope_gm)
        self.output_gm = inst.Tensor(self.dtype, (MAX_SIZE,), name="output_gm", scope=tik.scope_gm)
        self.aicore_num = self.tik_profiling.get_aicore_num()
        self.block_element = constant.BLOCK_SIZE // self.dtype_size
        self.reserve_ub_size = 0
        self.ub_size = (self.tik_profiling.get_unified_buffer_size() // self.dtype_size // self.block_element *
                        self.block_element) - self.reserve_ub_size
        self.max_gap = 65535 * self.block_element
        self.max_last_dim = (self.max_gap + self.ub_size) // self.block_element

    def _ceil_div(self, int1: tik.Scalar, int2):
        """
        get ceil for (int1 / int2)
        """
        result = self.tik_instance.Scalar("int64")
        with self.tik_instance.if_scope(int1 == 0):
            result.set_as(1)
        with self.tik_instance.else_scope():
            result.set_as(int1 // int2)
        with self.tik_instance.if_scope(int1 % int2 != 0):
            result.set_as(result + 1)

        return result

    def _ceil_32bytes_count(self, count: tik.Scalar):
        ceil_num = self._ceil_div(count, self.block_element)
        return ceil_num * self.block_element

    def _get_input_gm_addr(self, cur_index: tik.Scalar):
        reverse_part_output_shape = self.tiling_param.output_shape[::-1][1:]
        reverse_input_shape = self.tiling_param.input_shape[::-1]
        reverse_begin = self.tiling_param.begin[::-1][1:]
        inst = self.tik_instance

        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        addr = inst.Scalar(self.tiling_param.dtype, name="input_addr")
        addr.set_as(self.tiling_param.begin[-1])
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")
        step = inst.Scalar(self.tiling_param.dtype, name="step")
        step.set_as(1)

        for idx, dim in enumerate(reverse_part_output_shape):
            step.set_as(step * reverse_input_shape[idx])
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * (tmp + reverse_begin[idx]))

            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _get_output_gm_addr(self, cur_index: tik.Scalar):
        reverse_part_output_shape = self.tiling_param.output_shape[::-1][1:]
        reverse_output_shape = self.tiling_param.output_shape[::-1]
        inst = self.tik_instance

        tmp_cur_index = inst.Scalar(self.tiling_param.dtype, name="tmp_cur_index")
        tmp_cur_index.set_as(cur_index)
        addr = inst.Scalar(self.tiling_param.dtype, name="output_addr")
        addr.set_as(0)
        tmp = inst.Scalar(self.tiling_param.dtype, name="dim")
        step = inst.Scalar(self.tiling_param.dtype, name="step")
        step.set_as(1)

        for idx, dim in enumerate(reverse_part_output_shape):
            step.set_as(step * reverse_output_shape[idx])
            tmp.set_as(tmp_cur_index % dim)
            addr.set_as(addr + step * tmp)

            tmp_cur_index.set_as(tmp_cur_index / dim)
        return addr

    def _data_move(self, dest: tik.Tensor, src: tik.Tensor, count: tik.Scalar):
        dtype_size = common_util.get_data_size(src.dtype)
        burst = self._ceil_div(count * dtype_size, constant.BLOCK_SIZE)
        self.tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)

    def strided_slice(self):
        inst = self.tik_instance
        core_num = self.aicore_num
        output_shape = self.tiling_param.output_shape
        with inst.for_range(0, core_num, block_num=core_num, name="core_idx") as i:
            self.tiling_param.init()
            with inst.if_scope(output_shape[-1] >= self.block_element):
                self._do_large_last_dim(i)
            with inst.else_scope():
                self._do_small_last_dim(i)

    def _do_large_last_dim(self, core_idx):
        self._do_large_last_dim_normal(core_idx)

    def _do_large_last_dim_normal(self, core_idx):
        inst = self.tik_instance
        core_num = self.aicore_num
        output_shape = self.tiling_param.output_shape
        inner_loops = self._ceil_div(output_shape[-1], self.ub_size)
        out_loops = self._ceil_div(self.tiling_param.out_dim, core_num)
        with inst.for_range(0, out_loops, name="out_loop") as loop_idx:
            idx = core_idx * out_loops + loop_idx
            with inst.if_scope(idx < self.tiling_param.out_dim):
                input_gm_addr = self._get_input_gm_addr(idx)
                output_gm_addr = self._get_output_gm_addr(idx)
                with inst.for_range(0, inner_loops, name="inner_loop") as inner_loop_idx:
                    with inst.if_scope(output_shape[-1] % self.block_element == 0):
                        self._do_large_last_dim_align(input_gm_addr, output_gm_addr, inner_loop_idx)
                    with inst.else_scope():
                        self._do_large_last_dim_not_align(input_gm_addr, output_gm_addr, inner_loop_idx)

    def _do_small_last_dim(self, core_idx):
        inst = self.tik_instance
        core_num = self.aicore_num
        output_shape = self.tiling_param.output_shape
        inner_dim = output_shape[-1]
        out_dim = self.tiling_param.out_dim
        out_loops = self._ceil_div(out_dim, core_num)
        tmp_ub_size = self.block_element
        ub_size = self.ub_size - self.block_element
        ub_data_count = inst.Scalar("int32", name="out_ub_data_count")
        ub_data_count.set_as(0)
        input_gm = self.input_gm
        output_gm = self.output_gm
        need_update_out_addr = inst.Scalar("int32", name="need_update_out_addr")
        need_update_out_addr.set_as(1)
        output_gm_addr = inst.Scalar(self.tiling_param.dtype,
                                     name="output_addr")
        with inst.new_stmt_scope():
            tmp_ub = inst.Tensor(self.dtype, (tmp_ub_size,), scope=tik.scope_ubuf, name="tmp_ub")
            ub = inst.Tensor(self.dtype, (ub_size,), scope=tik.scope_ubuf, name="out_ub")
            with inst.for_range(0, out_loops, name="out_loop") as loop_idx:
                idx = core_idx * out_loops + loop_idx
                with inst.if_scope(idx < self.tiling_param.out_dim):
                    input_gm_addr = self._get_input_gm_addr(idx)
                    with inst.if_scope(need_update_out_addr == 1):
                        need_update_out_addr.set_as(0)
                        output_gm_addr.set_as(self._get_output_gm_addr(idx))

                    with inst.if_scope(ub_data_count + inner_dim > ub_size):
                        self._data_move(output_gm[output_gm_addr], ub, ub_data_count)
                        ub_data_count.set_as(0)
                        output_gm_addr.set_as(self._get_output_gm_addr(idx))
                    self._data_move(tmp_ub, input_gm[input_gm_addr], self.block_element)

                    with inst.for_range(0, inner_dim) as index:
                        ub[ub_data_count + index] = tmp_ub[index]
                    ub_data_count.set_as(ub_data_count + inner_dim)

                    with inst.if_scope(loop_idx == out_loops - 1):
                        self._add_tail(ub, tmp_ub, idx, ub_data_count)
            with inst.if_scope(ub_data_count != 0):
                self._data_move(output_gm[output_gm_addr], ub, ub_data_count)

    def _do_large_last_dim_align(self, input_gm_addr, output_gm_addr, inner_loop_idx):
        inst = self.tik_instance
        total = self.tiling_param.output_shape[-1]
        input_gm = self.input_gm
        output_gm = self.output_gm
        count = inst.Scalar("int32", name="remain")
        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="out_ub")
            count.set_as(total - self.ub_size * inner_loop_idx)
            with inst.if_scope(count > self.ub_size):
                count.set_as(self.ub_size)

            self._data_move(ub, input_gm[input_gm_addr + inner_loop_idx * self.ub_size], count)
            self._data_move(output_gm[output_gm_addr + inner_loop_idx * self.ub_size], ub, count)

    # pylint: too-many-locals
    def _do_large_last_dim_not_align(self, input_gm_addr, output_gm_addr, inner_loop_idx):
        inst = self.tik_instance
        total = self.tiling_param.output_shape[-1]
        input_gm = self.input_gm
        output_gm = self.output_gm
        count = inst.Scalar("int32", name="remain")
        with inst.new_stmt_scope():
            ub = inst.Tensor(self.dtype, (self.ub_size,), scope=tik.scope_ubuf, name="out_ub")
            count.set_as(total - self.ub_size * inner_loop_idx)
            with inst.if_scope(count >= self.ub_size):
                self._data_move(ub, input_gm[input_gm_addr + inner_loop_idx * self.ub_size], self.ub_size)
                self._data_move(output_gm[output_gm_addr + inner_loop_idx * self.ub_size], ub, self.ub_size)
            with inst.else_scope():
                with inst.if_scope(inner_loop_idx > 0):
                    align_count = self._ceil_32bytes_count(count)
                    redundant_count = align_count - count
                    new_in_start_index = (input_gm_addr + inner_loop_idx * self.ub_size - redundant_count)
                    new_out_start_index = (output_gm_addr + inner_loop_idx * self.ub_size - redundant_count)
                    self._data_move(ub, input_gm[new_in_start_index:], align_count)
                    self._data_move(output_gm[new_out_start_index:], ub, align_count)
                with inst.else_scope():
                    in_start_index = (input_gm_addr + inner_loop_idx * self.ub_size)
                    out_start_index = (output_gm_addr + inner_loop_idx * self.ub_size)
                    self._data_move(ub, input_gm[in_start_index:], self.block_element)
                    self._data_move(output_gm[out_start_index:], ub, self.block_element)

                    in_start_index += self.block_element
                    out_start_index += self.block_element
                    align_count = self._ceil_32bytes_count(count - self.block_element)
                    redundant_count = align_count - count + self.block_element
                    new_in_start_index = in_start_index - redundant_count
                    new_out_start_index = out_start_index - redundant_count
                    self._data_move(ub, input_gm[new_in_start_index:], align_count)
                    self._data_move(output_gm[new_out_start_index:], ub, align_count)

    def _add_tail(self, ub, tmp_ub, idx, ub_data_count):
        inst = self.tik_instance
        inner_dim = self.tiling_param.output_shape[-1]
        out_dim = self.tiling_param.out_dim
        align_count = self._ceil_32bytes_count(ub_data_count)
        overlap_count = align_count - ub_data_count
        ext_rows = self._ceil_div(overlap_count, inner_dim)
        input_gm = self.input_gm
        with inst.for_range(1, ext_rows + 1, name="ext_row") as row_idx:
            with inst.if_scope(idx + row_idx < out_dim):
                input_addr = self._get_input_gm_addr(idx + row_idx)
                self._data_move(tmp_ub, input_gm[input_addr], self.block_element)
                with inst.for_range(0, inner_dim) as index:
                    with inst.if_scope(ub_data_count < align_count):
                        ub[ub_data_count] = tmp_ub[index]
                        ub_data_count.set_as(ub_data_count + 1)


# pylint: disable=locally-disabled,too-many-arguments,
# pylint: unused-argument,too-many-locals
@te.op.register_operator("StridedSlice")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, OPTION_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_INT,
                 OPTION_ATTR_INT, OPTION_ATTR_INT, OPTION_ATTR_INT, OPTION_ATTR_INT, KERNEL_NAME)
def strided_slice(input_x, begin, end, strides=None, output_x=None, begin_mask=0, end_mask=0,
                  ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, kernel_name="strided_slice"):
    """
    Extracts a strided slice of a tensor (generalized python array indexing).
    Roughly speaking, this op extracts a slice of size (end-begin)/stride
    from the given input_ tensor.
    Starting at the location specified by begin the slice continues
     by adding stride to the index
    until all dimensions are not less than end. Note that a stride
    can be negative, which causes a reverse slice.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of out
    begin: list.
        represents the index of the first value to select.
    end: list.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin
        value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position
        is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification
        should shrink the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice_d"

    Returns
    -------
    tik_instance
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "uint8", "bool", "int8")
    check_dtype(input_dtype, check_list, param_name="input_x")
    strided_slice_instance = StridedSlice(input_x, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                                          shrink_axis_mask, kernel_name)
    strided_slice_instance.strided_slice()
    inst = strided_slice_instance.tik_instance
    opt_config = {"out_of_bound_sync_check": True}
    inst.BuildCCE(kernel_name=strided_slice_instance.kernel_name,
                  inputs=(strided_slice_instance.input_gm,
                          strided_slice_instance.begin_gm,
                          strided_slice_instance.end_gm,
                          strided_slice_instance.strides_gm),
                  outputs=(strided_slice_instance.output_gm,),
                  flowtable=[strided_slice_instance.tiling_param.tiling_gm],
                  config=opt_config,
                  enable_l2=False)

    te.op.add_compile_info("vars", {"block_dim": strided_slice_instance.aicore_num,
                                    "begin_mask": strided_slice_instance.begin_mask,
                                    "end_mask": strided_slice_instance.end_mask,
                                    "ellipsis_mask": strided_slice_instance.ellipsis_mask,
                                    "new_axis_mask": strided_slice_instance.new_axis_mask,
                                    "shrink_axis_mask": strided_slice_instance.shrink_axis_mask})
    return inst
