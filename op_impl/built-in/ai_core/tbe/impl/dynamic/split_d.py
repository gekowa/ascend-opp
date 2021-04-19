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

split_d
"""
import math
from te import tik
from te import platform as tbe_platform
import te.lang.dynamic
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import DYNAMIC_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_INT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.error_manager import error_manager_vector

from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

# max int64
MAX_INT64 = 2**64 - 1
# tiling param num
TILING_ARG_NUM = 18
# reserved ub size
RESERVED_UB_SIZE = 9 * 1024
# 8 bit
EIGHT_BIT = 8
# bytes of one block
BLOCK_BYTES = 32


# pylint: disable=unused-argument,invalid-name
def op_select_format(x, y, split_dim, num_split, kernel_name="split_d"):
    """
    select format dynamically
    """
    dtype = "float16, float, int32, int8, int16, int64, uint8, uint16, uint32, uint64"
    input_format = "ND, ND, ND, ND, ND, ND, ND, ND, ND, ND"

    # ND
    input0 = gen_param(
        classify="input0",
        name="x",
        datatype=dtype,
        format=input_format)
    output0 = gen_param(
        classify="output0",
        name="y",
        datatype=dtype,
        format=input_format)

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# pylint: disable=too-many-instance-attributes,too-many-statements,too-many-locals,too-many-arguments,unused-variable
class SplitD:
    """
    Split a tensor into `num_split` tensors along one dimension.
    """
    def __init__(self, x, y, split_dim, num_split, kernel_name):
        """
        Init split_d parameters

        Parameters
        ----------
        x: dict
            the dict of input tensor.
        y: list or tuple
            the list of output tensor.
        split_dim: int
            the dimension along which to split_d.
        num_split: int
            an integer indicating the number of split_d along `split_dim`.
        kernel_name: str
            cce kernel name, default value is "split_d".

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.split_dim = split_dim
        self.num_split = num_split
        self.kernel_name = kernel_name
        self.input_dtype = x.get("dtype").lower()
        self.output_dtype = y[0].get("dtype").lower()
        self.input_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(self.input_dtype) // EIGHT_BIT
        self.input_data_each_block = BLOCK_BYTES // self.input_dtype_bytes_size
        self.core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) - RESERVED_UB_SIZE
        self.ub_number = self.ub_size // self.input_dtype_bytes_size
        self.ub_number = (self.ub_number // self.input_data_each_block) * self.input_data_each_block
        self.tiling_gm, self.input_gm, self.outs_gm = self.init_gm_tensor()
        self.check_input_params()

        self.ub_number_new = None
        self.input_ub = None
        self.temp_ub = None
        self.tiling_ub = None
        self.select_mode = None
        self.input_size_split = None
        self.output_size_split = None
        self.act_core_num = None
        self.loop_each_core = None
        self.loop_last_core = None
        self.data_each_core = None
        self.data_last_core = None
        self.loop_num = None
        self.last_num = None
        self.loop_num_last_core = None
        self.last_num_last_core = None
        self.input_num = None
        self.loop_each = None
        self.loop_last = None
        self.loop_each_last_core = None
        self.loop_last_last_core = None
        self.loop_burst_len = None

    def init_gm_tensor(self):
        """
        init gm tensor

        Parameters
        ----------
        None

        Returns
        -------
        gm tensors
        """
        tiling_gm = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        input_gm = self.tik_instance.Tensor(self.input_dtype, (MAX_INT64,), name="input_gm", scope=tik.scope_gm)
        outputs_gm = []
        for i in range(self.num_split):
            tensor_name = "gm_output_" + str(i)
            gm_tensor = self.tik_instance.Tensor(self.input_dtype, (MAX_INT64,), name=tensor_name, scope=tik.scope_gm)
            outputs_gm.append(gm_tensor)

        return tiling_gm, input_gm, outputs_gm

    def init_ub_tensor(self):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.input_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_number, ), name="input_ub",
                                                 scope=tik.scope_ubuf)
        self.temp_ub = self.tik_instance.Tensor(self.input_dtype, (self.input_data_each_block, ), name="temp_ub",
                                                scope=tik.scope_ubuf)

    def tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.select_mode = self.tik_instance.Scalar("int64")
        self.input_size_split = self.tik_instance.Scalar("int64")
        self.output_size_split = self.tik_instance.Scalar("int64")
        self.act_core_num = self.tik_instance.Scalar("int64")
        self.loop_each_core = self.tik_instance.Scalar("int64")
        self.loop_last_core = self.tik_instance.Scalar("int64")
        self.data_each_core = self.tik_instance.Scalar("int64")
        self.data_last_core = self.tik_instance.Scalar("int64")
        self.loop_num = self.tik_instance.Scalar("int64")
        self.last_num = self.tik_instance.Scalar("int64")
        self.loop_num_last_core = self.tik_instance.Scalar("int64")
        self.last_num_last_core = self.tik_instance.Scalar("int64")
        self.input_num = self.tik_instance.Scalar("int64")
        self.ub_number_new = self.tik_instance.Scalar("int64")
        self.loop_each = self.tik_instance.Scalar("int64")
        self.loop_last = self.tik_instance.Scalar("int64")
        self.loop_each_last_core = self.tik_instance.Scalar("int64")
        self.loop_last_last_core = self.tik_instance.Scalar("int64")
        self.loop_burst_len = self.tik_instance.Scalar("int64")

        self.select_mode.set_as(self.tiling_ub[0])
        self.input_size_split.set_as(self.tiling_ub[1])
        self.output_size_split.set_as(self.tiling_ub[2])
        self.act_core_num.set_as(self.tiling_ub[3])
        self.loop_each_core.set_as(self.tiling_ub[4])
        self.loop_last_core.set_as(self.tiling_ub[5])
        self.data_each_core.set_as(self.tiling_ub[6])
        self.data_last_core.set_as(self.tiling_ub[7])
        self.loop_num.set_as(self.tiling_ub[8])
        self.last_num.set_as(self.tiling_ub[9])
        self.loop_num_last_core.set_as(self.tiling_ub[10])
        self.last_num_last_core.set_as(self.tiling_ub[11])
        self.ub_number_new.set_as(self.tiling_ub[12])
        self.input_num.set_as(self.tiling_ub[13])
        self.loop_each.set_as(self.tiling_ub[14])
        self.loop_last.set_as(self.tiling_ub[15])
        self.loop_each_last_core.set_as(self.tiling_ub[16])
        self.loop_last_last_core.set_as(self.tiling_ub[17])
        self.loop_burst_len.set_as(0)

    def ceil_div(self, value_x, value_y):
        """
        do ceil division
        """
        return (value_x + value_y - 1) // value_y

    def floor_div(self, value_x, value_y):
        """
        do ceil division
        """
        return value_x // value_y

    def check_input_params(self):
        """
        to the check whether the input parameters is valid or not
        """
        if self.input_dtype != self.output_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal("split_d", "self.input_dtype",
                                                                  "self.output_dtype", self.input_dtype,
                                                                  self.output_dtype)

        dtype_list = ("float16, float32, int32, int8, int16, int64, uint8, uint16, uint32, uint64")
        check_dtype(self.input_dtype, dtype_list, param_name="x")

    def compute_move_copy(self, core_index, loop_num, last_num):
        """
        move copy

        Parameters
        ----------
        core_index: core index
        loop_num: loop num
        last_num: last num

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, loop_num) as loop_index:
            self.move_copy_process(core_index, loop_index, self.ub_number_new)
        with self.tik_instance.if_scope(last_num > 0):
            self.move_copy_process(core_index, loop_num, last_num)

    def move_copy_process(self, core_index, loop_index, ub_number):
        """
        move process

        Parameters
        ----------
        core_index: core index
        loop_index: loop index
        ub_number: ub number

        Returns
        -------
        None
        """
        self.loop_burst_len.set_as(self.ceil_div(ub_number, self.input_data_each_block))
        move_offset = (core_index * self.data_each_core + loop_index * self.ub_number_new)
        with self.tik_instance.if_scope(self.loop_burst_len > 1):
            with self.tik_instance.if_scope(ub_number % self.input_data_each_block != 0):
                align_offset = ub_number - self.input_data_each_block
                self.tik_instance.data_move(self.temp_ub, self.input_gm[move_offset + align_offset], 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.outs_gm[0][move_offset + align_offset], self.temp_ub,
                                            0, 1, 1, 0, 0)
                self.loop_burst_len.set_as(self.loop_burst_len - 1)
        self.tik_instance.data_move(self.input_ub, self.input_gm[move_offset], 0, 1, self.loop_burst_len, 0, 0)
        self.tik_instance.data_move(self.outs_gm[0][move_offset], self.input_ub, 0, 1, self.loop_burst_len,
                                    0, 0)

    def compute_first_dim_for_core(self, core_index, loop_each_core):
        """
        compute first dim for core

        Parameters
        ----------
        core_index: core index
        loop_each_core: loop each core

        Returns
        -------
        None
        """
        for out_index, output in enumerate(self.outs_gm):
            with self.tik_instance.for_range(0, loop_each_core) as out_loop_index:
                with self.tik_instance.for_range(0, self.loop_num) as loop_index:
                    self.first_dim_move_process(core_index, out_index, out_loop_index, loop_index, self.ub_number_new)
                with self.tik_instance.if_scope(self.last_num > 0):
                    self.first_dim_move_process(core_index, out_index, out_loop_index, self.loop_num, self.last_num)

    def first_dim_move_process(self, core_index, out_index, out_loop_index, loop_index, ub_number):
        """
        move process

        Parameters
        ----------
        core_index: core index
        out_index: out index
        out_loop_index: out loop index
        loop_index: loop index
        ub_number: ub number

        Returns
        -------
        None
        """
        self.loop_burst_len.set_as(self.ceil_div(ub_number, self.input_data_each_block))
        move_in_offset = (core_index * self.loop_each_core * self.input_size_split + out_index *
                          self.output_size_split + out_loop_index * self.input_size_split + loop_index *
                          self.ub_number_new)
        move_out_offset = (core_index * self.loop_each_core * self.output_size_split + out_loop_index *
                           self.output_size_split + loop_index * self.ub_number_new)
        with self.tik_instance.if_scope(self.loop_burst_len > 1):
            with self.tik_instance.if_scope(ub_number % self.input_data_each_block != 0):
                align_offset = ub_number - self.input_data_each_block
                self.tik_instance.data_move(self.temp_ub, self.input_gm[move_in_offset + align_offset], 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.outs_gm[out_index][move_out_offset + align_offset], self.temp_ub,
                                            0, 1, 1, 0, 0)
                self.loop_burst_len.set_as(self.loop_burst_len - 1)
        self.tik_instance.data_move(self.input_ub, self.input_gm[move_in_offset], 0, 1, self.loop_burst_len, 0, 0)
        self.tik_instance.data_move(self.outs_gm[out_index][move_out_offset], self.input_ub, 0, 1,
                                    self.loop_burst_len, 0, 0)

    def compute_last_dim_for_core(self, core_index, loop_num, last_num):
        """
        compute last dim for core

        Parameters
        ----------
        core_index: core index
        loop_num: loop num
        last_num: last num

        Returns
        -------
        None
        """
        for out_index, output in enumerate(self.outs_gm):
            with self.tik_instance.for_range(0, self.loop_each_core) as out_loop_index:
                with self.tik_instance.for_range(0, loop_num) as loop_index:
                    self.last_dim_move_process(core_index, out_index, out_loop_index, loop_index,
                                               self.ub_number_new)
                with self.tik_instance.if_scope(last_num > 0):
                    self.last_dim_move_process(core_index, out_index, out_loop_index, loop_num, last_num)

    def last_dim_move_process(self, core_index, out_index, out_loop_index, loop_index, ub_number):
        """
        move process

        Parameters
        ----------
        core_index: core index
        out_index: out index
        out_loop_index: out loop index
        loop_index: loop index
        ub_number: ub_number

        Returns
        -------
        None
        """
        self.loop_burst_len.set_as(self.ceil_div(ub_number, self.input_data_each_block))
        move_in_offset = (core_index * self.data_each_core + out_index * self.output_size_split + out_loop_index *
                          self.input_size_split + loop_index * self.ub_number_new)
        move_out_offset = (core_index * self.data_each_core + out_loop_index * self.output_size_split + loop_index *
                           self.ub_number_new)
        with self.tik_instance.if_scope(self.loop_burst_len > 1):
            with self.tik_instance.if_scope(ub_number % self.input_data_each_block != 0):
                align_offset = ub_number - self.input_data_each_block
                self.tik_instance.data_move(self.temp_ub, self.input_gm[move_in_offset + align_offset], 0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.outs_gm[out_index][move_out_offset + align_offset], self.temp_ub,
                                            0, 1, 1, 0, 0)
                self.loop_burst_len.set_as(self.loop_burst_len - 1)
        self.tik_instance.data_move(self.input_ub, self.input_gm[move_in_offset], 0, 1, self.loop_burst_len, 0, 0)
        self.tik_instance.data_move(self.outs_gm[out_index][move_out_offset], self.input_ub, 0, 1,
                                    self.loop_burst_len, 0, 0)

    def compute_one_core(self):
        """
        one core compute

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        loop_burst_len = self.ceil_div(self.output_size_split, self.input_data_each_block)
        for out_index, output in enumerate(self.outs_gm):
            with self.tik_instance.for_range(0, self.loop_each_core) as out_loop_index:
                move_in_offset = out_index * self.output_size_split + out_loop_index * self.input_size_split
                move_out_offset = out_loop_index * self.output_size_split
                self.tik_instance.data_move(self.input_ub, self.input_gm[move_in_offset], 0, 1, loop_burst_len, 0, 0)
                self.tik_instance.data_move(self.outs_gm[out_index][move_out_offset], self.input_ub, 0, 1,
                                            loop_burst_len, 0, 0)

    def compute_for_scalar(self, core_index, loop_num, last_num, loop_each, loop_last):
        """
        scalar operation

        Parameters
        ----------
        core_index: core index
        loop_num: loop num
        last_num: last num
        loop_each: loop num each time
        loop_last: loop num of last time

        Returns:
        ----------
        None
        """
        with self.tik_instance.for_range(0, loop_num) as out_loop_index:
            self.move_for_scalar(core_index, out_loop_index, loop_each, loop_each, self.ub_number_new)
        with self.tik_instance.if_scope(last_num > 0):
            self.move_for_scalar(core_index, loop_num, loop_each, loop_last, last_num)

    def move_for_scalar(self, core_index, out_loop_index, loop_each, loop_each_num, ub_number):
        """
        scalar operation

        Parameters
        ----------
        core_index: core index
        out_loop_index: out loop index
        loop_each: loop num
        loop_each_num: loop num each num
        ub_number: data num each time

        Returns:
        ----------
        None
        """
        move_in_offset = core_index * self.data_each_core + out_loop_index * self.ub_number_new
        loop_in_burst_len = self.ceil_div(ub_number, self.input_data_each_block)
        self.tik_instance.data_move(self.input_ub, self.input_gm[move_in_offset], 0, 1, loop_in_burst_len, 0, 0)
        align_ub_offset = (self.ceil_div(self.ub_number_new, self.input_data_each_block) *
                           self.input_data_each_block)
        for out_index, output in enumerate(self.outs_gm):
            with self.tik_instance.for_range(0, loop_each_num) as loop_index:
                with self.tik_instance.for_range(0, self.output_size_split) as less_index:
                    offset_in = (loop_index * self.input_size_split + out_index *
                                 self.output_size_split + less_index)
                    offset_out = align_ub_offset + loop_index * self.output_size_split + less_index
                    self.input_ub[offset_out] = self.input_ub[offset_in]
            out_num = self.floor_div(ub_number, self.num_split)
            loop_out_burst_len = self.ceil_div(out_num, self.input_data_each_block)
            move_out_offset = (core_index * self.loop_each_core * self.output_size_split + out_loop_index *
                               loop_each * self.output_size_split)
            self.loop_burst_len.set_as(loop_out_burst_len)
            with self.tik_instance.if_scope(self.loop_burst_len > 1):
                with self.tik_instance.if_scope(out_num % self.input_data_each_block != 0):
                    align_offset = out_num - self.input_data_each_block
                    with self.tik_instance.for_range(0, self.input_data_each_block) as block_index:
                        self.temp_ub[block_index] = self.input_ub[align_ub_offset + align_offset + block_index]
                    self.tik_instance.data_move(self.outs_gm[out_index][move_out_offset + align_offset],
                                                self.temp_ub, 0, 1, 1, 0, 0)
                    self.loop_burst_len.set_as(self.loop_burst_len - 1)
            self.tik_instance.data_move(self.outs_gm[out_index][move_out_offset], self.input_ub[align_ub_offset],
                                        0, 1, self.loop_burst_len, 0, 0)

    def split_d_compute_tiling(self):
        """
        split_d operation

        Parameters
        ----------
        None

        Returns:
        ----------
        compile info
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_index:
            self.init_ub_tensor()
            self.tiling_ub = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,), name="tiling_ub",
                                                      scope=tik.scope_ubuf)
            tiling_burst_len = self.ceil_div(TILING_ARG_NUM, 4)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, tiling_burst_len, 0, 0)
            self.tiling_args()
            with self.tik_instance.if_scope(self.select_mode == 0):
                self.compute_move_copy(core_index, self.loop_num, self.last_num)
            with self.tik_instance.if_scope(self.select_mode == 1):
                with self.tik_instance.if_scope(core_index < self.act_core_num - 1):
                    self.compute_first_dim_for_core(core_index, self.loop_each_core)
                with self.tik_instance.else_scope():
                    self.compute_first_dim_for_core(core_index, self.loop_last_core)
            with self.tik_instance.if_scope(self.select_mode == 2):
                with self.tik_instance.if_scope(core_index < self.act_core_num - 1):
                    self.compute_for_scalar(core_index, self.loop_num, self.last_num, self.loop_each,
                                            self.loop_last)
                with self.tik_instance.else_scope():
                    self.compute_for_scalar(core_index, self.loop_num_last_core, self.last_num_last_core,
                                            self.loop_each_last_core, self.loop_last_last_core)
            with self.tik_instance.if_scope(self.select_mode == 3):
                self.compute_one_core()
            with self.tik_instance.if_scope(self.select_mode == 4):
                with self.tik_instance.if_scope(core_index < self.act_core_num - 1):
                    self.compute_last_dim_for_core(core_index, self.loop_num, self.last_num)
                with self.tik_instance.else_scope():
                    self.compute_last_dim_for_core(core_index, self.loop_num_last_core, self.last_num_last_core)

    def split_d_operator(self):
        """
        split_d operation

        Parameters
        ----------
        None

        Returns:
        ----------
        compile info
        """
        self.split_d_compute_tiling()
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=self.input_gm,
            outputs=self.outs_gm,
            flowtable=[self.tiling_gm])

        te.op.add_compile_info("vars", {"ub_size": self.ub_number, "core_num": self.core_num,
                                        "split_dim": self.split_dim, "num_split": self.num_split})

        return self.tik_instance


@te.op.register_operator("SplitD")
@check_op_params(REQUIRED_INPUT, DYNAMIC_OUTPUT, REQUIRED_ATTR_INT, REQUIRED_ATTR_INT, KERNEL_NAME)
def split_d(x, y, split_dim, num_split, kernel_name="split_d"):
    """
    Split a tensor into `num_split` tensors along one dimension.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: list or tuple
        the list of output tensor.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        an integer indicating the number of split_d along `split_dim`.
    kernel_name: str
        cce kernel name, default value is "split_d".

    Returns
    -------
    compile info
    """
    obj = SplitD(x, y, split_dim, num_split, kernel_name)
    return obj.split_d_operator()
