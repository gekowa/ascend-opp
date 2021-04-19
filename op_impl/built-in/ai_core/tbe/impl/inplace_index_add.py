#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

inplace_index_add
"""
import math
from functools import reduce as functools_reduce

from te import tik
from te import platform as tbe_platform
from topi.cce import util
from te.utils import op_utils

# neg two
NEG_TWO = -2

# neg one
NEG_ONE = -1

# UB Reserve size
UB_RESERVE_SIZE = 8192

# Some basic global params
ONE_BLOCK_SIZE = 32
ONE_VECTOR_CALC_SIZE = 256
TYPE_BYTES_MAP = {"float16": 2, "float32": 4, "int8": 2, "uint8": 2, "int16": 2, "int32": 4}
BITS_OF_ONE_BYTE = 8
MAX_REPEAT_TIMES = 255
MAX_UBSIZE_USE_RATE = 0.9


class Scatter():
    """
       Function: use to store scatter base parameters
       Modify : 2019-10-28
    """

    def __init__(self, var, indices, updates, var_out, nd_flag, axis, kernel_name,
                 compute_type):
        """
        Init scatter base parameters
        :param var: dict
            data of input
            datatype suports float32,float16,int32,int8,uint8
        :param indices: dict
            data of indices
            datatype supports int32
        :param updates: dict
            data of updates
            datatype supports float32,float16,int32,int8,uint8
        :param var_out: dict
            data of input
        :param nd_flag: str
            if this op is nd operator
        :param axis: int
            which axis to compute index add
        :param kernel_name:
        :param compute_type:
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.nd_flag = nd_flag
        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()
        self.indices_shape = indices.get("shape")
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_shape = updates.get("shape")
        self.updates_dtype = updates.get("dtype").lower()
        self.var_out = var_out
        self.axis = axis
        self.indices_num = functools_reduce(lambda x, y: x * y,
                                            self.indices_shape)
        self.updates_num = functools_reduce(lambda x, y: x * y,
                                            self.updates_shape[axis:])
        self.kernel_name = kernel_name
        self.ai_core_num = tik.Dprofile().get_aicore_num()
        self.vconv_dst_dtype = "float16"
        self.compute_type = compute_type
        self.nd_flag = nd_flag

        if self.indices_shape == (1,) and \
                len(self.var_shape[axis:]) - len(self.updates_shape[axis:]) == 1:
            if not nd_flag:
                self.updates_shape = self.updates_shape[:axis] + (1,) + self.updates_shape[axis:]

        # check input attr params
        self.check_param()

        # get outer loop num
        self.get_outer_loop_and_block_num()

        # get data num
        self.get_data_num()

        # get some basic information
        self.get_some_basic_info()

        # decide the mask of computation
        self.max_num_one_repeat = ONE_VECTOR_CALC_SIZE // TYPE_BYTES_MAP[self.var_dtype]

        if self.update_data_num < self.var_data_each_block:
            self.block_num = 1

        # init some variable
        self.init_variable()

        # init gm of input
        self.create_gm_tensors()

        # calc the tiling parameters
        self.init_ub_tensor_para()

    def get_some_basic_info(self):
        """
        get some basic info, e.x. ub_size_bytes
        :return:
        """
        self.ub_size_bytes = (
                tbe_platform.cce_conf.get_soc_spec(
                    tbe_platform.cce_conf.UB_SIZE) - UB_RESERVE_SIZE)
        self.var_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.var_dtype) // BITS_OF_ONE_BYTE
        self.indices_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.indices_dtype) // BITS_OF_ONE_BYTE
        self.var_data_each_block = ONE_BLOCK_SIZE // self.var_dtype_bytes_size
        self.indices_data_each_block = ONE_BLOCK_SIZE // self.indices_dtype_bytes_size
        self.indices_ub_number = 0
        self.updates_ub_number = 0
        self.index_loop_num = 0
        self.vconv_ub_number = None

    def get_data_num(self):
        self.axis_and_after_data_num_of_updates = functools_reduce(lambda x, y: x * y, self.updates_shape[self.axis:])
        self.axis_and_after_data_num_of_var = functools_reduce(lambda x, y: x * y, self.var_shape[self.axis:])

        # get update_data_num and index_dims
        if self.nd_flag:
            if self.indices_shape[-1] == len(self.var_shape[self.axis:]):
                self.update_data_num = 1
            else:
                self.update_data_num = functools_reduce(
                    lambda x, y: x * y, self.var_shape[self.axis + self.indices_shape[-1]:])
            self.max_indices = functools_reduce(
                lambda x, y: x * y, self.var_shape[self.axis:self.indices_shape[-1]])
            self.index_dims = self.indices_shape[-1]
        else:
            if len(self.var_shape[self.axis:]) > 1:
                self.update_data_num = functools_reduce(lambda x, y: x * y,
                                                        self.var_shape[self.axis + 1:])
            else:
                self.update_data_num = 1
            self.max_indices = self.var_shape[self.axis]
            self.index_dims = 1

    def init_variable(self):
        """
        init Variable
        :return:
        """
        # state some ub tensor
        self.var_vconv_ub = None
        self.updates_vconv_ub = None
        self.var_tail_vconv_ub = None
        self.updates_tail_vconv_ub = None

        self.var_ub = None
        self.updates_ub = None
        self.indices_ub = None
        self.var_tail_ub = None
        self.updates_tail_ub = None

        self.var_read_index = None
        self.updates_read_index = None
        self.indices_loop_index = None
        self.indices_tmp = None

        self.outer_loop_start_index_every_block = None
        self.outer_loops_ub_per_block = None
        self.outer_loop_start_index_of_var = None
        self.outer_loop_start_index_of_updates = None

    def create_gm_tensors(self):
        self.var_gm = self.tik_instance.Tensor(
            self.var_dtype, self.var_shape, name="var_gm", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(
            self.indices_dtype,
            self.indices_shape,
            name="indices_gm",
            scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(
            self.updates_dtype,
            self.updates_shape,
            name="updates_gm",
            scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(
            self.var_dtype, self.var_shape, name="out_gm", scope=tik.scope_gm)

    def get_outer_loop_and_block_num(self):
        self.outer_loop = None
        self.outer_loops_per_block = None

        # get outer loop and block num
        if self.axis == 0:
            self.outer_loop = 1
        else:
            self.outer_loop = functools_reduce(lambda x, y: x * y, self.var_shape[0:self.axis])

        if self.outer_loop == 1:
            self.block_num = 1
            self.outer_loops_per_block = 0
        else:
            self.outer_loops_per_block = math.ceil(self.outer_loop / self.ai_core_num)
            self.block_num = math.ceil(self.outer_loop / self.outer_loops_per_block)

    def init_ub_tensor_para(self):
        """
        calc the tiling parameters
        :return:
        """
        updates_size_bytes = self.var_dtype_bytes_size * self.update_data_num
        indices_size_bytes = self.indices_dtype_bytes_size * self.indices_num

        # int8 and uint8 situation, calc the tiling
        need_vconv_dtype = ("int8", "uint8")
        if self.var_dtype in need_vconv_dtype:
            self.init_ub_tensor_para_of_int8_uint8(indices_size_bytes, updates_size_bytes)
        else:
            # if updates size * 2 is smaller than 0.9 ub size
            if updates_size_bytes * 2 < self.ub_size_bytes * MAX_UBSIZE_USE_RATE:
                self.updates_ub_number = math.ceil(
                    self.update_data_num /
                    self.var_data_each_block) * self.var_data_each_block
                self.indices_ub_number = (self.ub_size_bytes - updates_size_bytes * 2) // self.indices_dtype_bytes_size
                self.indices_ub_number = math.ceil(
                    self.indices_ub_number /
                    self.indices_data_each_block) * self.indices_data_each_block
                if self.indices_num < self.indices_ub_number:
                    self.indices_ub_number = math.ceil(
                        self.indices_num / self.indices_data_each_block
                    ) * self.indices_data_each_block
            # if indices size is smaller than 0.9 ub size
            elif indices_size_bytes < self.ub_size_bytes * MAX_UBSIZE_USE_RATE:
                self.indices_ub_number = math.ceil(
                    self.indices_num /
                    self.indices_data_each_block) * self.indices_data_each_block

                self.updates_ub_number = (self.ub_size_bytes - indices_size_bytes) // 2 // self.var_dtype_bytes_size

                self.updates_ub_number = math.ceil(
                    self.updates_ub_number /
                    self.var_data_each_block) * self.var_data_each_block
            else:
                self.indices_ub_number = (
                        self.ub_size_bytes // self.indices_dtype_bytes_size // 2 //
                        self.indices_data_each_block * self.indices_data_each_block)
                self.updates_ub_number = (
                        self.indices_ub_number // 2 // self.var_data_each_block *
                        self.var_data_each_block)

        # get last num, if last num size is smaller than 32B
        last_num = self.update_data_num % self.updates_ub_number
        if (last_num < self.var_data_each_block and
                self.update_data_num > self.updates_ub_number):
            self.updates_ub_number -= self.var_data_each_block

    def init_ub_tensor_para_of_int8_uint8(self, indices_size_bytes, updates_size_bytes):
        vconv_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.vconv_dst_dtype)
        vconv_data_each_block = ONE_BLOCK_SIZE // vconv_dtype_bytes_size
        vconv_size_bytes = (
                updates_size_bytes // self.var_dtype_bytes_size *
                vconv_dtype_bytes_size)
        if (updates_size_bytes + vconv_size_bytes) * 2 < (
                self.ub_size_bytes * MAX_UBSIZE_USE_RATE):
            self.updates_ub_number = math.ceil(
                self.update_data_num /
                self.var_data_each_block) * self.var_data_each_block

            self.vconv_ub_number = math.ceil(
                self.update_data_num /
                vconv_data_each_block) * vconv_data_each_block

            self.indices_ub_number = (self.ub_size_bytes - updates_size_bytes * 2 -
                                      vconv_size_bytes * 2) // self.indices_dtype_bytes_size

            self.indices_ub_number = math.ceil(
                self.indices_ub_number /
                self.indices_data_each_block) * self.indices_data_each_block

        elif indices_size_bytes < (self.ub_size_bytes * MAX_UBSIZE_USE_RATE):
            self.indices_ub_number = math.ceil(
                self.indices_num /
                self.indices_data_each_block) * self.indices_data_each_block
            self.updates_ub_number = (self.ub_size_bytes - indices_size_bytes) // self.var_dtype_bytes_size // 6

            self.updates_ub_number = math.ceil(
                self.updates_ub_number /
                self.var_data_each_block) * self.var_data_each_block

            self.vconv_ub_number = math.ceil(
                self.updates_ub_number /
                vconv_data_each_block) * vconv_data_each_block

        else:
            self.updates_ub_number = (
                    self.ub_size_bytes // 2 //
                    (vconv_dtype_bytes_size + self.var_dtype_bytes_size) // 2 //
                    self.var_data_each_block * self.var_data_each_block)
            self.indices_ub_number = (
                    self.ub_size_bytes // self.indices_dtype_bytes_size // 2 //
                    self.var_data_each_block * self.var_data_each_block)
            self.vconv_ub_number = self.updates_ub_number

    def init_ub_tensor(self):
        """
        Init the ub tensor
        :return:
        """
        # init int8 or uint8 ub tensors
        self.init_ub_tensor_of_int8_uint8()

        # var/indices/updates ub create
        self.init_ub_tensor_of_non_int8_uint8()

        # init ub tensor of read index
        self.init_ub_tensor_of_read_index()

    def init_ub_tensor_of_int8_uint8(self):
        """
        init ub tensor if var dtype is in (int8, uint8)
        :return:
        """
        need_vconv_dtype = ("int8", "uint8")
        if self.var_dtype in need_vconv_dtype:
            self.var_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.vconv_ub_number,),
                name="var_vconv_ub",
                scope=tik.scope_ubuf)
            self.updates_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.vconv_ub_number,),
                name="updates_vconv_ub",
                scope=tik.scope_ubuf)

            self.var_tail_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.var_data_each_block,),
                name="var_tail_vconv_ub",
                scope=tik.scope_ubuf)
            self.updates_tail_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.var_data_each_block,),
                name="updates_tail_vconv_ub",
                scope=tik.scope_ubuf)

    def init_ub_tensor_of_non_int8_uint8(self):
        """
        init ub tensor of non int8 or uint8
        :return:
        """
        self.var_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.updates_ub_number,),
            name="var_ub",
            scope=tik.scope_ubuf)
        self.updates_ub = self.tik_instance.Tensor(
            self.updates_dtype, (self.updates_ub_number,),
            name="updates_ub",
            scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(
            self.indices_dtype, (self.indices_ub_number,),
            name="indices_ub",
            scope=tik.scope_ubuf)

        self.var_tail_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="var_tail_ub",
            scope=tik.scope_ubuf)
        self.updates_tail_ub = self.tik_instance.Tensor(
            self.updates_dtype, (self.var_data_each_block,),
            name="updates_tail_ub",
            scope=tik.scope_ubuf)

    def init_ub_tensor_of_read_index(self):
        """
        Init ub tensor of loop/index
        :return:
        """
        self.var_read_index = self.tik_instance.Scalar("int32")
        self.var_read_index.set_as(0)

        self.updates_read_index = self.tik_instance.Scalar("int32")
        self.updates_read_index.set_as(0)

        self.indices_loop_index = self.tik_instance.Scalar("int32")
        self.indices_loop_index.set_as(0)

        self.indices_tmp = self.tik_instance.Scalar("int32")
        self.indices_tmp.set_as(0)

        self.outer_loop_start_index_every_block = self.tik_instance.Scalar("int32")
        self.outer_loop_start_index_every_block.set_as(0)

        self.outer_loops_ub_per_block = self.tik_instance.Scalar("int32")
        self.outer_loops_ub_per_block.set_as(self.outer_loop)

        self.outer_loop_start_index_of_var = self.tik_instance.Scalar("int32")
        self.outer_loop_start_index_of_var.set_as(0)

        self.outer_loop_start_index_of_updates = self.tik_instance.Scalar("int32")
        self.outer_loop_start_index_of_updates.set_as(0)

    def get_var_read_index(self, indices_ub_index):
        """
        Get var absolute index according to indices index
        :param indices_ub_index:
        :return:
        """
        if not self.nd_flag:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
        else:
            self.get_var_read_index_of_nd_flag()

    def get_var_read_index_of_nd_flag(self, indices_ub_index):
        """
        Get var read index when nd_flag is True
        :param indices_ub_index:
        :return:
        """
        indices_ub_index = indices_ub_index * self.indices_shape[-1]
        self.var_read_index.set_as(0)
        if self.indices_shape[-1] == 1:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
        else:
            for i in range(0, self.indices_shape[-1]):
                self.indices_tmp.set_as(self.indices_ub[indices_ub_index +
                                                        i])
                if i + 1 < self.indices_shape[-1]:
                    self.var_read_index.set_as(
                        self.var_read_index +
                        self.indices_tmp * functools_reduce(
                            lambda x, y: x * y,
                            self.var_shape[i + 1:self.indices_shape[-1]]))
                else:
                    self.var_read_index.set_as(self.var_read_index +
                                               self.indices_tmp)

    def get_updates_read_index(self, indices_ub_index):
        """
        Get absolute updates index according to indices index
        :param indices_ub_index:
        :return:
        """
        read_index = self.outer_loop_start_index_of_updates + indices_ub_index * self.update_data_num
        self.updates_read_index.set_as(read_index)

    def updates_the_var(self, indices_in_index, indice_num):
        """
        Update the update fragment corresponding to the index
        :param indices_in_index: start indices index
        :param indice_num: indices_num this time to update
        :return:
        """
        indices_burst_len = math.ceil(indice_num / self.indices_data_each_block)
        if self.indices_num == 1:
            self.tik_instance.data_move(self.indices_ub, self.indices_gm, 0, 1,
                                        indices_burst_len, 0, 0)
        else:
            self.tik_instance.data_move(self.indices_ub,
                                        self.indices_gm[indices_in_index], 0, 1,
                                        indices_burst_len, 0, 0)
        if self.nd_flag:
            indice_loop_num = indice_num // self.indices_shape[-1]
        else:
            indice_loop_num = indice_num

        # for loop start
        with self.tik_instance.for_range(0,
                                         indice_loop_num) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)

            if self.nd_flag:
                indices_in_index = indices_in_index // self.indices_shape[-1]
            self.get_updates_read_index(indices_ub_index + indices_in_index)
            self.var_read_index.set_as(self.var_read_index *
                                       self.update_data_num + self.outer_loop_start_index_of_var)
            self.calc_updates()

    def calc_updates(self):
        """
        Calculate updates fragment
        :return:
        """
        # calc the loop times of update data once to ub
        updates_loop = self.update_data_num // self.updates_ub_number
        if updates_loop > 0:
            with self.tik_instance.for_range(0, updates_loop) as loop_index:
                self.calc_updates_small(loop_index * self.updates_ub_number,
                                        self.updates_ub_number)

        # deal with tail num
        last_num = self.update_data_num % self.updates_ub_number
        if last_num > 0:
            self.calc_updates_small(updates_loop * self.updates_ub_number,
                                    last_num)

    def calc_updates_small(self, read_index_offset, element_num):
        """
        Move corresponding updates/var to UB and calculate
        :param read_index_offset: offset index of inner ub for loop
        :param element_num: element number once ub
        :return:
        """
        updates_burst_len = math.ceil(element_num / self.var_data_each_block)
        self.tik_instance.data_move(
            self.var_ub, self.var_gm[self.var_read_index + read_index_offset],
            0, 1, updates_burst_len, 0, 0)

        self.tik_instance.data_move(
            self.updates_ub,
            self.updates_gm[self.updates_read_index + read_index_offset], 0, 1,
            updates_burst_len, 0, 0)

        tail_ele_num = element_num % self.var_data_each_block
        align_offset = 0
        if (tail_ele_num != 0 and
                self.update_data_num > self.var_data_each_block):
            align_ele_num = (
                    element_num // self.var_data_each_block *
                    self.var_data_each_block)
            align_offset = (
                    read_index_offset + align_ele_num -
                    (self.var_data_each_block - tail_ele_num))
            self.tik_instance.data_move(
                self.var_tail_ub,
                self.var_gm[self.var_read_index + align_offset], 0, 1, 1, 0, 0)

            self.tik_instance.data_move(
                self.updates_tail_ub,
                self.updates_gm[self.updates_read_index + align_offset], 0, 1,
                1, 0, 0)

        # start calc repeat loop
        self.start_calc_repeat_loop(element_num)

        # compute the mask
        self.compute_mask(read_index_offset, element_num, tail_ele_num, updates_burst_len, align_offset)

    def start_calc_repeat_loop(self, element_num):
        """
        start calc repeat loop
        :param element_num:
        :return:
        """
        compute_loop = element_num // self.max_num_one_repeat // MAX_REPEAT_TIMES

        if compute_loop > 0:
            with self.tik_instance.for_range(0, compute_loop) as index:
                index_offset = index * self.max_num_one_repeat * MAX_REPEAT_TIMES
                self.calc_process(self.max_num_one_repeat, index_offset,
                                  index_offset, index_offset, MAX_REPEAT_TIMES, False)
        last_loop = element_num % (self.max_num_one_repeat *
                                   MAX_REPEAT_TIMES) // self.max_num_one_repeat

        if last_loop > 0:
            index_offset = compute_loop * self.max_num_one_repeat * MAX_REPEAT_TIMES
            self.calc_process(self.max_num_one_repeat, index_offset,
                              index_offset, index_offset, last_loop, False)

    def compute_mask(self, read_index_offset, element_num, tail_ele_num, updates_burst_len, align_offset):
        """
        compute the var and update data according every repeat
        :param read_index_offset:
        :param element_num:
        :param tail_ele_num:
        :param updates_burst_len:
        :param align_offset:
        :return:
        """
        compute_mask = element_num % self.max_num_one_repeat
        if compute_mask > 0:
            index_offset = (
                    element_num // self.max_num_one_repeat *
                    self.max_num_one_repeat)
            if (tail_ele_num == 0 or
                    self.update_data_num < self.var_data_each_block):
                self.calc_process(compute_mask, index_offset, index_offset,
                                  index_offset, 1, False)

                self.tik_instance.data_move(
                    self.out_gm[self.var_read_index + read_index_offset],
                    self.var_ub, 0, 1, updates_burst_len, 0, 0)
            else:
                self.calc_process(self.var_data_each_block, 0, 0, 0, 1, True)
                self.tik_instance.data_move(
                    self.out_gm[self.var_read_index + align_offset],
                    self.var_tail_ub, 0, 1, 1, 0, 0)
                self.calc_process(compute_mask, index_offset, index_offset,
                                  index_offset, 1, False)
                self.tik_instance.data_move(
                    self.out_gm[self.var_read_index + read_index_offset],
                    self.var_ub, 0, 1, updates_burst_len - 1, 0, 0)
        else:
            self.tik_instance.data_move(
                self.out_gm[self.var_read_index + read_index_offset],
                self.var_ub, 0, 1, updates_burst_len, 0, 0)

    def calc_process(self, mask, dest_addr, src_addr1, src_addr2, repeat_times,
                     is_tail):
        """
        Execute the corresponding calculation instruction
        :param mask: calc mask
        :param dest_addr: dst
        :param src_addr1: src1
        :param src_addr2: src2
        :param repeat_times: repeat times
        :param is_tail: bool, is tail
        :return:
        """
        need_vconv_dtype = ("int8", "uint8")
        mask, dst_ub, src1_ub, src2_ub, compute_repeat_stride = self.compute_paras(
            mask, dest_addr, src_addr1, src_addr2, repeat_times, is_tail)

        if self.compute_type == "vadd":
            self.tik_instance.vadd(mask, dst_ub, src1_ub, src2_ub, repeat_times,
                                   1, 1, 1, compute_repeat_stride,
                                   compute_repeat_stride, compute_repeat_stride)
        else:
            raise RuntimeError("the operater [%s] is not supported" %
                               self.compute_type)
        if self.var_dtype in need_vconv_dtype:
            if is_tail:
                self.tik_instance.vconv(mask, "", self.var_tail_ub,
                                        self.var_tail_vconv_ub, repeat_times, 1,
                                        1, 4, 8)
            else:
                self.tik_instance.vconv(mask, "", self.var_ub[src_addr1],
                                        self.var_vconv_ub[dest_addr],
                                        repeat_times, 1, 1, 4, 8)

    def compute_paras(self, mask, dest_addr, src_addr1, src_addr2, repeat_times,
                      is_tail):
        """
        compute the computation paras
        :param mask: calc mask
        :param dest_addr: dest_addr
        :param src_addr1: src_addr1
        :param src_addr2: src_addr2
        :param repeat_times: repeat_times
        :param is_tail: is tail or not
        :return:
        """
        need_vconv_dtype = ("int8", "uint8")
        if self.var_dtype in need_vconv_dtype:
            if is_tail:
                self.tik_instance.vconv(mask, "",
                                        self.var_tail_vconv_ub[dest_addr],
                                        self.var_tail_ub[src_addr1],
                                        repeat_times, 1, 1, 8, 4)
                self.tik_instance.vconv(mask, "",
                                        self.updates_tail_vconv_ub[dest_addr],
                                        self.updates_tail_ub[src_addr2],
                                        repeat_times, 1, 1, 8, 4)
                compute_repeat_stride = 8
                src1_ub = self.var_tail_vconv_ub
                src2_ub = self.updates_tail_vconv_ub
                dst_ub = self.var_tail_vconv_ub
                mask = self.var_data_each_block
            else:
                self.tik_instance.vconv(mask, "", self.var_vconv_ub[dest_addr],
                                        self.var_ub[src_addr1], repeat_times, 1,
                                        1, 8, 4)
                self.tik_instance.vconv(mask, "",
                                        self.updates_vconv_ub[dest_addr],
                                        self.updates_ub[src_addr2],
                                        repeat_times, 1, 1, 8, 4)
                compute_repeat_stride = 8
                src1_ub = self.var_vconv_ub[src_addr1]
                src2_ub = self.updates_vconv_ub[src_addr2]
                dst_ub = self.var_vconv_ub[dest_addr]

        else:
            if is_tail:
                compute_repeat_stride = (
                        self.max_num_one_repeat // self.var_data_each_block)
                src1_ub = self.var_tail_ub
                src2_ub = self.updates_tail_ub
                dst_ub = self.var_tail_ub
                mask = self.var_data_each_block
            else:
                compute_repeat_stride = (
                        self.max_num_one_repeat // self.var_data_each_block)
                src1_ub = self.var_ub[src_addr1]
                src2_ub = self.updates_ub[src_addr2]
                dst_ub = self.var_ub[dest_addr]

        return mask, dst_ub, src1_ub, src2_ub, compute_repeat_stride

    def traversing_indices(self):
        """
        Traversing the indices and update the var
        :return:
        """
        max_ub_idx_num = (
                self.indices_ub_number // self.index_dims * self.index_dims)
        indices_loop_num = self.indices_num // max_ub_idx_num

        if indices_loop_num > 0:
            with self.tik_instance.for_range(
                    0, indices_loop_num) as indices_loop_index:
                self.updates_the_var(indices_loop_index * max_ub_idx_num,
                                     max_ub_idx_num)

        indices_last_num = self.indices_num % max_ub_idx_num
        if indices_last_num > 0:
            self.updates_the_var(indices_loop_num * max_ub_idx_num,
                                 indices_last_num)

    def check_param(self):
        """
        check the parameters
        :param var_out:
        :return:
        """
        var_out_shape = self.var_out.get("shape")
        var_out_dtype = self.var_out.get("dtype").lower()
        if var_out_dtype == "bool":
            var_out_dtype = "int8"
        util.check_kernel_name(self.kernel_name)
        util.check_shape_rule(self.var_shape)
        util.check_shape_rule(self.indices_shape)
        util.check_shape_rule(self.updates_shape)
        util.check_shape_rule(var_out_shape)

        util.check_tensor_shape_size(self.var_shape)
        util.check_tensor_shape_size(self.indices_shape)
        util.check_tensor_shape_size(self.updates_shape)
        util.check_tensor_shape_size(var_out_shape)

        check_list_var = ("float16", "float32", "int32", "int8", "uint8")
        check_list_indices = "int32"
        util.check_dtype_rule(self.var_dtype, check_list_var)
        util.check_dtype_rule(self.indices_dtype, check_list_indices)
        util.check_dtype_rule(self.updates_dtype, check_list_var)
        util.check_dtype_rule(var_out_dtype, check_list_var)

        if var_out_shape != self.var_shape:
            raise RuntimeError(
                "var_out's shape must be the same as var's shape")

        if (self.updates_dtype != self.var_dtype or
                var_out_dtype != self.var_dtype):
            raise RuntimeError(
                "updates's datatype and var_out's datatype must be the"
                " same as var's datatype")

        if self.nd_flag:
            if len(self.indices_shape) < 2:
                raise RuntimeError(
                    "the lenth of indices_shape must be large than 2")
            k = self.indices_shape[-1]
            updates_len = len(self.indices_shape) - 1 + len(self.var_shape) - k
            if k > len(self.var_shape):
                raise RuntimeError(
                    "indices_shape[-1] can not be large than var's rank")
            if len(self.updates_shape) != updates_len:
                raise RuntimeError("the lenth of update must be len(indices_"
                                   "shape)-1+len(var_shape)-indices_shape[-1]")
            updates_true_shape = self.indices_shape[:-1] + self.var_shape[k:]
        else:
            updates_true_shape = self.var_shape[:self.axis] + self.indices_shape + self.var_shape[self.axis + 1:]

        if self.updates_shape != updates_true_shape:
            raise RuntimeError("updates's shape is illegal")

    def get_outer_loop_index_of_updates(self, outer_loop_num):
        """
        Get absolute outer loop start index of update
        :param outer_loop_num: which outer loop it belongs to
        :return: None
        """
        real_index = outer_loop_num * self.axis_and_after_data_num_of_updates
        self.outer_loop_start_index_of_updates.set_as(real_index)

    def get_outer_loop_index_of_var(self, outer_loop_num):
        """
        get absolute outer loop start index of var
        :param outer_loop_num: which outer loop it belongs to
        :return: None
        """
        real_index = outer_loop_num * self.axis_and_after_data_num_of_var
        self.outer_loop_start_index_of_var.set_as(real_index)

    def traversing_outer_loop_per_block(self):
        """
        traversing outer loop per block
        :return: None
        """
        with self.tik_instance.for_range(0, self.outer_loops_ub_per_block) as outer_i:
            self.get_outer_loop_index_of_var((self.outer_loop_start_index_every_block + outer_i))
            self.get_outer_loop_index_of_updates((self.outer_loop_start_index_every_block + outer_i))
            self.traversing_indices()

    def scatter_operator(self):
        """
        return the inplace index add tik instance
        :return:
        """
        if self.block_num > 1:
            with self.tik_instance.for_range(
                    0, self.block_num,
                    block_num=self.block_num) as block_i:
                self.init_ub_tensor()
                self.outer_loop_start_index_every_block.set_as(block_i * self.outer_loops_per_block)
                self.outer_loops_ub_per_block.set_as(self.outer_loops_per_block)
                with self.tik_instance.if_scope(block_i == self.block_num - 1):
                    self.outer_loops_ub_per_block.set_as(self.outer_loop - self.outer_loop_start_index_every_block)
                self.traversing_outer_loop_per_block()
        else:
            self.init_ub_tensor()
            self.traversing_outer_loop_per_block()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.var_gm, self.indices_gm, self.updates_gm),
            outputs=self.out_gm,
            enable_l2=False)

        return self.tik_instance


@op_utils.check_op_params(dict, dict, dict, dict, int, str)
def inplace_index_add(var, axis_indices, updates, var_out, axis, kernel_name="index_add"):
    """
    inplace_index_add interface
    :param var: input var data
    :param axis_indices: input indices
    :param updates: update data
    :param var_out: output
    :param axis: axis to update
    :param kernel_name:
    :return: inplace index add result will return
    """
    scatter = Scatter(var, axis_indices, updates, var_out, False, axis, kernel_name, "vadd")
    scatter.scatter_operator()
