#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

concat_v2_d: Concatenates tensors along one dimension.
            The number of dimensions of input tensors must match,
            and all dimensions except 'axis' must be equal.
            tf ConcactV2 op

"""
from __future__ import absolute_import
import te.lang.dynamic
from topi.cce import util
from impl import common_util
from te.utils.op_utils import *
from te import tik
from impl import constant_util as constant
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from te.utils.error_manager import error_manager_vector as error_manager

MAX_SIZE = 2 ** 31 - 1


# pylint: disable=locally-disabled,unused-argument,too-many-branches
# pylint: disable=too-many-locals,too-many-statements,unused-variable
# pylint: disable=too-many-boolean-expressions
def op_select_format(input_values, output_data, axis, kernel_name="concat_v2_d"):
    """
    select format dynamically
    """
    data_list = []

    datatype_4d = "float16,float,int32,int8,int16,int64,uint8,uint16," \
                  "uint32,uint64,bool"
    format_4d = "ND,ND,ND,ND,ND,ND,ND,ND,ND,ND,ND"

    # ND
    input0 = gen_param(classify="input0", name="input_values", datatype=datatype_4d, format=format_4d)
    output0 = gen_param(classify="output0", name="output_data", datatype=datatype_4d, format=format_4d)

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def ceil_32bytes_align_count(count, dtype):
    type_size = common_util.get_data_size(dtype)
    block_count = math.ceil(count * type_size / constant.BLOCK_SIZE)
    return block_count * constant.BLOCK_SIZE // type_size


def _gm2ub(tik_instance: tik.Tik, dest: tik.Tensor, src: tik.Tensor, count):
    dtype_size = common_util.get_data_size(src.dtype)
    burst = math.ceil(count * dtype_size / constant.BLOCK_SIZE)
    tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)


class ConcatV2:
    class TilingParam:
        def __init__(self, input_values, inst: tik.Tik):
            self.tik_instance = inst
            dtype = "int64"

            # data in tiling_gm likes:
            # 0---- 1----    2----          3----
            # axis, out_dim, max_inner_dim, min_inner_dim,
            # 4----                5----
            # output_inner_length, input_count
            # 6----    7----
            # reserve, reserve
            # 8----             9----
            # first_inner_dims, first_output_idx,
            # second_inner_dims, second_output_idx
            # ...
            self.dtype = dtype
            self.input_values = input_values
            self.axis = inst.Scalar(dtype, name="axis")
            self.out_dim = inst.Scalar(dtype, name="out_dim")
            self.max_inner_dim = inst.Scalar(dtype, name="max_inner_dim")
            self.min_inner_dim = inst.Scalar(dtype, name="min_inner_dim")
            self.output_inner_length = inst.Scalar(dtype,
                                                   name="output_inner_length")

            tiling_ub_size = max(len(input_values) * 2, 8)
            tiling_gm_size = 8 + tiling_ub_size
            tiling_gm_size = ceil_32bytes_align_count(tiling_gm_size, dtype)
            tiling_ub_size = ceil_32bytes_align_count(tiling_ub_size, dtype)
            self.tiling_ub_size = tiling_ub_size
            self.tiling_gm = inst.Tensor(dtype, (tiling_gm_size,),
                                         name="tiling_gm",
                                         scope=tik.scope_gm)

            self._need_ub_size = (self.tiling_ub_size *
                                  common_util.get_data_size(dtype))
            self._tiling_ub = None
            self._out_dim = None
            self._inner_dim = None

        def init(self):
            inst = self.tik_instance
            dtype = self.dtype
            self._tiling_ub = inst.Tensor(dtype, (self.tiling_ub_size,),
                                          name="tiling_ub",
                                          scope=tik.scope_ubuf)

            head_count = 8
            _gm2ub(inst, self._tiling_ub, self.tiling_gm, head_count)
            self.axis.set_as(self._tiling_ub[0])
            self.out_dim.set_as(self._tiling_ub[1])
            self.max_inner_dim.set_as(self._tiling_ub[2])
            self.min_inner_dim.set_as(self._tiling_ub[3])
            self.output_inner_length.set_as(self._tiling_ub[4])

            _gm2ub(inst, self._tiling_ub, self.tiling_gm[head_count:],
                   self.tiling_ub_size)

            self._out_dim = inst.Scalar(dtype, name="out_dim")
            self._inner_dim = inst.Scalar(dtype, name="inner_dim")

        def get_dims(self, input_index):
            """
            :param input_index: index of input tensors
            :return: inner dims, output_index of each row
            """
            index = input_index * 2
            self._out_dim.set_as(self._tiling_ub[index])
            self._inner_dim.set_as(self._tiling_ub[index + 1])

            return self._out_dim, self._inner_dim

        def need_ub_size(self):
            return self._need_ub_size

    def __init__(self, input_values, axis, kernel_name):
        self.tik_instance = tik.Tik()
        self.tik_profiling = tik.Dprofile()
        self.tiling_param = self.TilingParam(input_values, self.tik_instance)
        self.aicore_num = self.tik_profiling.get_aicore_num()
        self.kernel_name = kernel_name
        self.axis = axis

        self.dtype = input_values[0].get("dtype").lower()
        self.output_shape = (MAX_SIZE,)
        self.input_shape = (MAX_SIZE,)

        self.input_tensors, self.output_tensor = self._init_gm_tensor(self.input_shape, self.output_shape,
                                                                      len(input_values),
                                                                      self.dtype)

        dtype_bytes_size = common_util.get_data_size(self.dtype)
        self.ele_each_block = constant.BLOCK_SIZE // dtype_bytes_size
        valid_ub_size = self.tik_profiling.get_unified_buffer_size()
        valid_ub_size -= self.tiling_param.need_ub_size()
        self.ub_buffer_length = valid_ub_size

        # reserve one block size for not 32 bytes align
        self.ub_buffer_length -= constant.BLOCK_SIZE

        # make ub_buffer_length 32 bytes align
        self.ub_buffer_length //= constant.BLOCK_SIZE
        self.ub_buffer_length *= constant.BLOCK_SIZE

        self.ub_buffer_length //= dtype_bytes_size

    def _init_gm_tensor(self, input_shape, output_shape, input_count, dtype):
        """
        init gm tensor

        Parameters
        ----------
        input_shape: list
            shape of input tensor
        output_shape: list
            shape of output tensor
        dtype: str
            data type

        Returns
        -------
        input_tensors: tik tensor
            input gm tensor
        output_tensor: tik tensor
            output gm tensor
        """
        input_tensors = []
        for _, index in enumerate(range(input_count)):
            tensor_name = "gm_input_" + str(index)
            gm_tensor = self.tik_instance.Tensor(dtype, input_shape, name=tensor_name, scope=tik.scope_gm)
            input_tensors.append(gm_tensor)

        output_tensor = self.tik_instance.Tensor(dtype, output_shape, name="gm_output", scope=tik.scope_gm)

        return input_tensors, output_tensor

    def _data_move(self, dest: tik.Tensor, src: tik.Tensor, count: tik.Scalar):
        dtype_size = common_util.get_data_size(src.dtype)
        burst = self._ceil_div(count * dtype_size, constant.BLOCK_SIZE)
        self.tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)

    def concat_compute(self):
        """
        build concat op

        Returns
        -------
        None
        """
        inst = self.tik_instance
        aicore_num = self.aicore_num
        with inst.for_range(0, aicore_num, name="core_idx", block_num=aicore_num) as i:
            self.tiling_param.init()
            min_inner_dim = self.tiling_param.min_inner_dim
            with inst.if_scope(min_inner_dim < self.ele_each_block):
                self._concat_small_inner(i)
            with inst.else_scope():
                self._concat_large_inner(i)

        opt_config = {"out_of_bound_sync_check": True}
        inst.BuildCCE(kernel_name=self.kernel_name, inputs=self.input_tensors, outputs=(self.output_tensor,),
                      flowtable=[self.tiling_param.tiling_gm],
                      config=opt_config,
                      enable_l2=False)

        te.op.add_compile_info("vars", {"input_size": len(self.input_tensors),
                                        "concat_dim": self.axis,
                                        "block_dim": self.aicore_num
                                        })
        return inst

    def _ceil_div(self, int1: tik.Scalar, int2):
        """
        ceil for (int1 / int2)
        """
        result = self.tik_instance.Scalar("int64")
        with self.tik_instance.if_scope(int1 == 0):
            result.set_as(1)
        with self.tik_instance.else_scope():
            result.set_as(int1 // int2)
        with self.tik_instance.if_scope(int1 % int2 != 0):
            result.set_as(result + 1)

        return result

    def _get_ceil_32bytes_count(self, count: tik.Scalar):
        ceil_num = self._ceil_div(count, self.ele_each_block)
        return ceil_num * self.ele_each_block

    def _concat_inner_dim_each_split(self, out_dim_idx, inner_dim_split_idx):
        for index, _ in enumerate(self.input_tensors):
            self._concat_compute_tensor_inner_dim(out_dim_idx, inner_dim_split_idx, index)

    def _concat_compute_tensor_inner_dim(self, out_dim_idx, inner_dim_split_idx, tensor_index):
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        with self.tik_instance.if_scope(inner_dims % self.ele_each_block == 0):
            self._concat_tensor_align_inner_dim(out_dim_idx, inner_dim_split_idx, tensor_index)
        with self.tik_instance.else_scope():
            self._concat_tensor_not_align_inner_dim(out_dim_idx, inner_dim_split_idx, tensor_index)

    def _concat_tensor_align_inner_dim(self, out_dim_idx, inner_dim_split_idx, tensor_index):
        inst = self.tik_instance
        factor = self.ub_buffer_length
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        input_gm = self.input_tensors[tensor_index]
        output_gm = self.output_tensor
        with inst.new_stmt_scope():
            ub_length = self.ub_buffer_length
            ub = inst.Tensor(input_gm.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            in_start_index = inner_dim_split_idx * factor + inner_dims * out_dim_idx

            output_dim = self.tiling_param.output_inner_length
            out_start_index = output_idx + inner_dim_split_idx * factor + output_dim * out_dim_idx
            with inst.if_scope(in_start_index < inner_dims * (1 + out_dim_idx)):
                count = inst.Scalar("int64", name="count")
                count.set_as(inner_dims * (1 + out_dim_idx) - in_start_index)
                with inst.if_scope(count > ub_length):
                    count.set_as(ub_length)

                self._data_move(ub, input_gm[in_start_index:], count)
                self._data_move(output_gm[out_start_index:], ub, count)

    def _concat_tensor_not_align_inner_dim(self, out_dim_idx, inner_dim_split_idx, tensor_index):
        inst = self.tik_instance
        factor = self.ub_buffer_length
        inner_dims, output_idx = self.tiling_param.get_dims(tensor_index)
        input_gm = self.input_tensors[tensor_index]
        output_gm = self.output_tensor

        with inst.new_stmt_scope():
            ub_length = self.ub_buffer_length
            ub = inst.Tensor(input_gm.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            in_start_index = inner_dim_split_idx * factor + inner_dims * out_dim_idx

            output_dim = self.tiling_param.output_inner_length
            out_start_index = output_idx + inner_dim_split_idx * factor + output_dim * out_dim_idx
            with inst.if_scope(in_start_index < inner_dims * (1 + out_dim_idx)):
                count = inner_dims * (1 + out_dim_idx) - in_start_index
                with inst.if_scope(count > ub_length):
                    self._data_move(ub, input_gm[in_start_index:], ub_length)
                    self._data_move(output_gm[out_start_index:], ub, ub_length)
                with inst.else_scope():
                    with inst.if_scope(inner_dim_split_idx > 0):
                        align_count = self._get_ceil_32bytes_count(count)
                        redundant_count = align_count - count
                        new_in_start_index = in_start_index - redundant_count
                        new_out_start_index = out_start_index - redundant_count
                        self._data_move(ub, input_gm[new_in_start_index:], align_count)
                        self._data_move(output_gm[new_out_start_index:], ub, align_count)
                    with inst.else_scope():
                        self._data_move(ub, input_gm[in_start_index:], self.ele_each_block)
                        self._data_move(output_gm[out_start_index:], ub, self.ele_each_block)

                        in_start_index += self.ele_each_block
                        out_start_index += self.ele_each_block
                        align_count = self._get_ceil_32bytes_count(count - self.ele_each_block)
                        redundant_count = align_count - count + self.ele_each_block
                        new_in_start_index = in_start_index - redundant_count
                        new_out_start_index = out_start_index - redundant_count
                        self._data_move(ub, input_gm[new_in_start_index:], align_count)
                        self._data_move(output_gm[new_out_start_index:], ub, align_count)

    def _concat_large_inner(self, core_idx):
        """
        tiling with out_dims and split of inner_dims
        """
        inst = self.tik_instance
        aicore_num = self.aicore_num
        out_dims = self.tiling_param.out_dim
        max_inner_dim = self.tiling_param.max_inner_dim
        inner_dims_loops = self._ceil_div(max_inner_dim, self.ub_buffer_length)
        max_loops = out_dims * inner_dims_loops

        out_loops = self._ceil_div(max_loops, aicore_num)
        with inst.for_range(0, out_loops, name="out_loops_idx") as i:
            loop_idx = i + out_loops * core_idx
            with inst.if_scope(loop_idx < max_loops):
                out_dim_idx = loop_idx / inner_dims_loops
                inner_dim_split_idx = loop_idx % inner_dims_loops
                self._concat_inner_dim_each_split(out_dim_idx, inner_dim_split_idx)

    def _concat_small_inner(self, core_idx):
        """
        tiling with out_dims
        """
        inst = self.tik_instance
        aicore_num = self.aicore_num
        out_dims = self.tiling_param.out_dim
        count_each_core = self._ceil_div(out_dims, aicore_num)
        self._concat_small_inner_each_core(core_idx, out_dims, count_each_core)

    def _concat_small_inner_each_core(self, core_idx, out_dims, count_each_core):
        inst = self.tik_instance
        with inst.for_range(0, count_each_core, name="inner_loop") as j:
            row_idx = j + count_each_core * core_idx
            with inst.if_scope(row_idx < out_dims):
                with inst.if_scope(j != count_each_core - 1):
                    self._concat_small_inner_each_core_not_last_row(row_idx)
                with inst.else_scope():
                    self._concat_small_inner_each_core_last_row(row_idx)

    def _concat_small_inner_each_core_not_last_row(self, row_idx):
        self._concat_small_inner_each_core_without_treat_overlap(row_idx, self.input_tensors)

    def _concat_small_inner_each_core_last_row(self, row_idx):
        self._concat_small_inner_each_core_without_treat_overlap(row_idx,
                                                                 self.input_tensors[0:len(self.input_tensors) - 1])
        self._concat_small_inner_each_core_last_row_last_tensor(row_idx)

    def _concat_small_inner_each_core_without_treat_overlap(self, row_idx, tensors):
        inst = self.tik_instance
        output_tensor = self.output_tensor
        output_inner_len = self.tiling_param.output_inner_length
        ub_length = self.ub_buffer_length
        with inst.new_stmt_scope():
            out_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            ub_data_count = inst.Scalar("int32", name="ub_data_count")
            ub_data_count.set_as(0)
            tmp_ub = inst.Tensor(self.dtype, (self.ele_each_block,), scope=tik.scope_ubuf, name="tmp_ub")

            out_row_start_idx = output_inner_len * row_idx
            out_start_idx = inst.Scalar("int64", name="ub_data_count")
            out_start_idx.set_as(out_row_start_idx)
            for index, input_tensor in enumerate(tensors):
                inner_dim, output_idx = self.tiling_param.get_dims(index)
                in_start_idx = inner_dim * row_idx
                with inst.if_scope(ub_data_count >= self.ele_each_block):
                    self._data_move(output_tensor[out_start_idx:], out_ub, ub_data_count)
                    ub_data_count.set_as(0)

                with inst.if_scope(ub_data_count == 0):
                    out_start_idx.set_as(out_row_start_idx + output_idx)

                with inst.if_scope(inner_dim < self.ele_each_block):
                    self._data_move(tmp_ub, input_tensor[in_start_idx:], inner_dim)
                    with inst.for_range(0, inner_dim) as scalar_idx:
                        out_ub[ub_data_count] = tmp_ub[scalar_idx]
                        ub_data_count.set_as(ub_data_count + 1)

                with inst.else_scope():
                    with inst.if_scope(ub_data_count > 0):
                        self._data_move(output_tensor[out_start_idx:], out_ub, ub_data_count)
                        ub_data_count.set_as(0)
                        out_start_idx.set_as(out_row_start_idx + output_idx)

                    loops = self._ceil_div(inner_dim, ub_length)
                    with inst.for_range(0, loops, name="inner_loop") as idx:
                        in_start_idx = ub_length * idx + inner_dim * row_idx
                        out_start_idx.set_as(ub_length * idx + out_row_start_idx + output_idx)
                        count = inst.Scalar("int64", name="count")
                        count.set_as(inner_dim * (1 + row_idx) - in_start_idx)
                        with inst.if_scope(count > ub_length):
                            count.set_as(ub_length)

                        self._data_move(out_ub, input_tensor[in_start_idx:], count)
                        self._data_move(output_tensor[out_start_idx:], out_ub, count)

            with inst.if_scope(ub_data_count > 0):
                self._data_move(output_tensor[out_start_idx:], out_ub, ub_data_count)

    def _concat_small_inner_each_core_last_row_last_tensor(self, row_idx):
        inst = self.tik_instance
        inst = self.tik_instance
        ub_length = self.ub_buffer_length
        output_inner_len = self.tiling_param.output_inner_length
        out_dims = self.tiling_param.out_dim
        with inst.new_stmt_scope():
            out_ub = inst.Tensor(self.dtype, (ub_length,), scope=tik.scope_ubuf, name="out_ub")
            output_tensor = self.output_tensor
            last_idx = len(self.input_tensors) - 1
            input_tensor = self.input_tensors[last_idx]
            inner_dim, output_idx = self.tiling_param.get_dims(last_idx)
            out_start_idx = inst.Scalar("int64", name="ub_data_count")
            ub_data_count = inst.Scalar("int32", name="ub_data_count")
            tmp_ub = inst.Tensor(self.dtype, (self.ele_each_block,), scope=tik.scope_ubuf, name="tmp_ub")
            out_start_idx.set_as(row_idx * output_inner_len + output_idx)
            with inst.if_scope(inner_dim < self.ele_each_block):
                self._data_move(out_ub, input_tensor[inner_dim * row_idx], inner_dim)
                ub_data_count.set_as(inner_dim)
                pad_count = inst.Scalar("int32", name="pad_count")
                pad_count.set_as(self.ele_each_block - inner_dim)
                loops = self._ceil_div(pad_count, output_inner_len)
                with inst.for_range(0, loops) as loop:
                    new_out_dim_idx = row_idx + loop
                    with inst.if_scope(new_out_dim_idx < out_dims):
                        for idx, tmp_tensor in enumerate(self.input_tensors):
                            temp_inner_dims, _ = self.tiling_param.get_dims(idx)
                            with inst.if_scope(ub_data_count < self.ele_each_block):
                                self._data_move(tmp_ub, tmp_tensor[(row_idx + loop + 1) * temp_inner_dims],
                                                self.ele_each_block)
                                with inst.for_range(0, temp_inner_dims) as scalar_idx:
                                    with inst.if_scope(ub_data_count < self.ele_each_block):
                                        out_ub[ub_data_count] = tmp_ub[scalar_idx]
                                        ub_data_count.set_as(ub_data_count + 1)

                self._data_move(output_tensor[out_start_idx:], out_ub, inner_dim)
            with inst.else_scope():
                loops = self._ceil_div(inner_dim, ub_length)
                with inst.for_range(0, loops, name="inner_loop") as idx:
                    in_start_idx = (ub_length * idx + inner_dim * row_idx)
                    out_start_idx.set_as(ub_length * idx + output_inner_len * row_idx + output_idx)
                    count = inner_dim * (row_idx + 1) - in_start_idx
                    with inst.if_scope(count > ub_length):
                        self._data_move(out_ub, input_tensor[in_start_idx:], ub_length)
                        self._data_move(output_tensor[out_start_idx:], out_ub, ub_length)
                    with inst.else_scope():
                        with inst.if_scope(idx > 0):
                            align_count = self._get_ceil_32bytes_count(count)
                            redundant_cnt = (align_count - count)
                            new_in_start_index = in_start_idx - redundant_cnt
                            new_out_start_index = out_start_idx - redundant_cnt
                            self._data_move(out_ub, input_tensor[new_in_start_index:], count)
                            self._data_move(output_tensor[new_out_start_index:], out_ub, count)
                        with inst.else_scope():
                            self._data_move(out_ub, input_tensor[in_start_idx:], self.ele_each_block)
                            self._data_move(output_tensor[out_start_idx:], out_ub, self.ele_each_block)
                            in_start_idx += self.ele_each_block
                            out_start_idx += self.ele_each_block
                            align_count = self._get_ceil_32bytes_count(count - self.ele_each_block)
                            redundant_cnt = align_count - count + self.ele_each_block
                            new_in_start_index = in_start_idx - redundant_cnt
                            new_out_start_index = out_start_idx - redundant_cnt
                            self._data_move(out_ub, input_tensor[new_in_start_index:], align_count)
                            self._data_move(output_tensor[new_out_start_index:], out_ub, align_count)


def _check_shape(input_values, shape_name):
    # check the length of input shape must be equal
    dim_num = len(input_values[0].get(shape_name))
    for _, tensor_dict in enumerate(input_values):
        shape_input = tensor_dict.get(shape_name)
        if len(shape_input) != dim_num:
            error_manager.raise_err_check_params_rules("concat", "The length of each shape must be equal",
                                                       "input_values",
                                                       [i.get(shape_name) for i in input_values])


def __check_params(input_values, axis):
    _check_shape(input_values, "shape")
    _check_shape(input_values, "ori_shape")

    dim_num = len(input_values[0].get("ori_shape"))

    if axis >= dim_num or axis < -dim_num:
        error_manager.raise_err_input_value_invalid("concat",
                                                    "concat_dim",
                                                    "between " + str(min(-dim_num, dim_num - 1)) + " and " +
                                                    str(max(-dim_num, dim_num - 1)),
                                                    axis)

    shape_value = []
    for _, tensor_dict in enumerate(input_values):
        shape_value.append(tensor_dict.get("ori_shape"))
    first_input_shape = input_values[0].get("ori_shape")

    # dims must equal except merge axis
    axis_new = axis % dim_num
    for j, _ in enumerate(first_input_shape):
        if j == axis_new:
            continue

        dim_values = set()
        for _, element_shape in enumerate(shape_value):
            dim_values.add(element_shape[j])

        if -1 in dim_values:
            dim_values.remove(-1)

        if len(dim_values) > 1:
            error_manager.raise_err_check_params_rules("concat",
                                                       "Dims must be equal except merge concat axis[%s]" % axis,
                                                       "input_values",
                                                       shape_value)

    dtype_lists = []
    for input_value in input_values:
        input_format = input_value.get("format")
        dtype_lists.append(input_value.get("dtype"))
        supported_formats = {"ND", "NHWC", "NCHW"}
        if input_format not in supported_formats:
            error_manager.raise_err_input_format_invalid('concat',
                                                         'input_values',
                                                         ','.join(supported_formats),
                                                         input_format)

    dtype = dtype_lists[0]
    for index, dtype_ in enumerate(dtype_lists):
        if dtype != dtype_:
            error_manager.raise_err_inputs_dtype_not_equal("concat",
                                                           "input_values[0]",
                                                           "input_values[%s]" % index,
                                                           dtype,
                                                           dtype_)


@te.op.register_operator("ConcatV2D")
@check_op_params(DYNAMIC_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_INT, KERNEL_NAME)
def concat_v2_d(input_values, output_data, axis, kernel_name="concat_v2_d"):
    """
    algorithm: concat_v2_d

    Parameters
    ----------
    input_values : A list of dict objects.
                 dict with keys(shape and dtype) of tensor
                 dtype only support float32, int8, int16, int32, int64, uint8,
                 uint16, uint32, uint64, float16
    output_data : A dict resulting from concatenation of the input tensors
    axis : scalar,in the range [-rank(values), rank(values))
    kernel_name : cce kernel name, default value is "concat_v2_d"

    Returns
    -------
    tik instance
    """
    __check_params(input_values, axis)
    concat_instance = ConcatV2(input_values, axis, kernel_name)
    return concat_instance.concat_compute()
