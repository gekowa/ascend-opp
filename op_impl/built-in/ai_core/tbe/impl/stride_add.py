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

stride_add
"""


import math
from functools import reduce

from te import tik

RESERVE_SIZE = 16 * 1024
BLOCK_SIZE = 32
DTYPE_BYTES = {"float16": 2, "float32": 4}


def stride_add(x1, x2, y, x1_c1_offset, x2_c1_offset, c1_len,
               kernel_name="stride_add"):
    """
    the external interfaces of op stride_add

    Parameters
    ----------
    x1: dict including shape, format and dtype
        dtype supports float16, float32; format only support NC1HWC0
    x2: dict including shape, format and dtype
        dtype supports float16, float32; format only support NC1HWC0
    y: dict including shape, format and dtype
        dtype supports float16, float32; format only support NC1HWC0
    x1_c1_offset: offset of C1 dim of tensor x1
    x2_c1_offset: offset of C1 dim of tensor x2
    c1_len: the extract len of C1 dim to add
    kernel_name: cce kernel name

    Returns
    -------
    tik_instance: tik_instance
    """
    input_dict = {
        "x1": x1,
        "x2": x2,
        "y": y,
        "x1_c1_offset": x1_c1_offset,
        "x2_c1_offset": x2_c1_offset,
        "c1_len": c1_len,
        "kernel_name": kernel_name
    }
    check_param(input_dict)
    stride_add_process = StrideAdd(input_dict)
    stride_add_process.compute_stride_add()
    stride_add_process.tik_instance.BuildCCE(
        kernel_name=kernel_name,
        inputs=[stride_add_process.x1_gm, stride_add_process.x2_gm],
        outputs=[stride_add_process.y_gm])

    return stride_add_process.tik_instance


def check_param(input_dict):
    """
    check the parameters in the input_dict

    Parameters
    ----------
    input_dict: the dict including the input info

    Returns
    -------
    None
    """

    c0 = input_dict["x1"]["shape"][-1]
    if c0 != 16:
        raise RuntimeError(
            "In op[stride_add], the dim of c0[%d] must be 16" % c0)
    return


class StrideAdd():
    """
    the main class of op stride_add
    """
    def __init__(self, input_dict):
        """
        the constructor function of class StrideAdd

        Parameters
        ----------
        input_dict: the dict including the input info

        Returns
        -------
        None
        """
        self.input_dict = input_dict
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.dtype = input_dict["x1"]["dtype"].lower()
        self.dsize = DTYPE_BYTES[self.dtype]
        self.data_each_block = BLOCK_SIZE // self.dsize
        self.vector_mask_max = 8 * self.data_each_block
        available_ub_size = (tik.Dprofile().get_unified_buffer_size()
                             - RESERVE_SIZE)

        # maximum elements of each tensor on the UB
        self.ub_max_num = (available_ub_size // self.dsize // 3
                           // self.data_each_block * self.data_each_block)

        self.batch_size = input_dict["x1"]["shape"][0]
        self.y_shape = self.compute_ouput_shape()  # (N, C1_len, H, W, C0)

        self.x1_gm = self.tik_instance.Tensor(self.dtype,
                                              input_dict["x1"]["shape"],
                                              name="x1_gm",
                                              scope=tik.scope_gm)

        self.x2_gm = self.tik_instance.Tensor(self.dtype,
                                              input_dict["x2"]["shape"],
                                              name="x2_gm",
                                              scope=tik.scope_gm)

        self.y_gm = self.tik_instance.Tensor(self.dtype,
                                             self.y_shape,
                                             name="y_gm",
                                             scope=tik.scope_gm)
        return

    def compute_ouput_shape(self):
        """
        compute the output shape

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        y_shape = list(self.input_dict["x1"]["shape"])
        y_shape[1] = self.input_dict["c1_len"]  # (N, C1_len, H, W, C0)
        return tuple(y_shape)

    def split_aicore(self):
        """
        the aicore split scheme

        Parameters
        ----------
        None

        Returns
        -------
        used_aicore_num: used aicore num
        batch_num_per_aicore_process:
            batch num per aicore process
        batch_tail:
            the tail batch num after uniformly divided
        """
        available_aicore_num = tik.Dprofile().get_aicore_num()
        if self.batch_size < available_aicore_num:
            used_aicore_num = self.batch_size
            batch_num_per_aicore_process = 1
            batch_tail = 0
        else:
            used_aicore_num = available_aicore_num
            batch_num_per_aicore_process = self.batch_size // used_aicore_num
            batch_tail = self.batch_size % used_aicore_num

        return (used_aicore_num,
                batch_num_per_aicore_process,
                batch_tail)

    def compute_stride_add(self):
        """
        the main function of computing stride add

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        (used_aicore_num,
         batch_num_per_aicore_process,
         batch_tail) = self.split_aicore()

        with self.tik_instance.for_range(
                0, used_aicore_num, block_num=used_aicore_num) as aicore_id:
            self.x1_ub = self.tik_instance.Tensor(self.dtype,
                                                  (self.ub_max_num,),
                                                  name="x1_ub",
                                                  scope=tik.scope_ubuf)

            self.x2_ub = self.tik_instance.Tensor(self.dtype,
                                                  (self.ub_max_num,),
                                                  name="x2_ub",
                                                  scope=tik.scope_ubuf)

            self.y_ub = self.tik_instance.Tensor(self.dtype,
                                                 (self.ub_max_num,),
                                                 name="y_ub",
                                                 scope=tik.scope_ubuf)

            batch_id = self.tik_instance.Scalar("int32")
            self.compute_uniformly_divided_batches(
                aicore_id, batch_id,
                batch_tail, batch_num_per_aicore_process)
            self.compute_tail_batches(aicore_id, batch_id, batch_tail)

        return

    def compute_uniformly_divided_batches(self, aicore_id,
                                          batch_id, batch_tail,
                                          batch_num_per_aicore_process):
        """
        compute the uniformly divided batches on each aicore

        Parameters
        ----------
        aicore_id: the aicore index
        batch_id: the batch index
        batch_tail: the tail batch num after uniformly divided
        batch_num_per_aicore_process:
            batch num per aicore process

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(
                0, batch_num_per_aicore_process) as inner_cycle:

            batch_id.set_as(
                aicore_id * batch_num_per_aicore_process + inner_cycle)

            with self.tik_instance.if_scope(batch_tail > 0):

                with self.tik_instance.if_scope(aicore_id < batch_tail):
                    batch_id.set_as(batch_id + aicore_id)

                with self.tik_instance.else_scope():
                    batch_id.set_as(batch_id + batch_tail)
            self.compute_each_batch(batch_id)

        return

    def compute_tail_batches(self, aicore_id, batch_id, batch_tail):
        """
        compute the tail batches on each aicore

        Parameters
        ----------
        aicore_id: the aicore index
        batch_id: the batch index
        batch_tail: the tail batch num after uniformly divided

        Returns
        -------
        None
        """
        if batch_tail > 0:
            with self.tik_instance.if_scope(aicore_id < batch_tail):
                batch_id.set_as(batch_id + 1)
                self.compute_each_batch(batch_id)

        return

    def compute_each_batch(self, batch_id):
        """
        compute stride add on each batch

        Parameters
        ----------
        batch_id: batch index

        Returns
        -------
        None
        """
        # C1_len * H * W * C0
        compute_num = reduce(lambda x, y: x * y, self.y_shape[1:])

        # the base offset of each batch
        x1_base_offset = self.compute_offset(batch_id, "x1")
        x2_base_offset = self.compute_offset(batch_id, "x2")
        y_base_offset = batch_id * compute_num

        x1_base_offset_init = x1_base_offset
        x2_base_offset_init = x2_base_offset
        y_base_offset_init = y_base_offset

        loop_time = compute_num // self.ub_max_num

        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_index:

                x1_base_offset += loop_index * self.ub_max_num
                x2_base_offset += loop_index * self.ub_max_num
                y_base_offset += loop_index * self.ub_max_num
                self.compute_each_loop(x1_base_offset,
                                       x2_base_offset,
                                       y_base_offset,
                                       self.ub_max_num)

            x1_base_offset = x1_base_offset_init + loop_time * self.ub_max_num
            x2_base_offset = x2_base_offset_init + loop_time * self.ub_max_num
            y_base_offset = y_base_offset_init + loop_time * self.ub_max_num

        last_num_data = compute_num % self.ub_max_num
        if last_num_data > 0:
            self.compute_each_loop(x1_base_offset,
                                   x2_base_offset,
                                   y_base_offset,
                                   last_num_data)

        return

    def compute_each_loop(self, x1_offset, x2_offset,
                          y_offset, compute_num):
        """
        compute stride add on each loop

        Parameters
        ----------
        x1_offset: offset of x1
        x2_offset: offset of x2
        y_offset: offset of y
        compute_num: number of computing in this loop

        Returns
        -------
        None
        """
        burst_len = math.ceil(compute_num / self.data_each_block)
        self.tik_instance.data_move(self.x1_ub,
                                    self.x1_gm[x1_offset], 0, 1,
                                    burst_len, 0, 0)

        self.tik_instance.data_move(self.x2_ub,
                                    self.x2_gm[x2_offset], 0, 1,
                                    burst_len, 0, 0)

        add_loop = compute_num // (self.vector_mask_max * 255)

        add_offset = 0
        if add_loop > 0:
            with self.tik_instance.for_range(0, add_loop) as index:
                add_offset = index * self.vector_mask_max * 255
                self.tik_instance.vec_add(self.vector_mask_max,
                                          self.y_ub[add_offset],
                                          self.x1_ub[add_offset],
                                          self.x2_ub[add_offset],
                                          255, 8, 8, 8)

        repeat_time = (compute_num % (self.vector_mask_max * 255)
                       // self.vector_mask_max)

        if repeat_time > 0:
            add_offset = self.vector_mask_max * 255 * add_loop
            self.tik_instance.vec_add(self.vector_mask_max,
                                      self.y_ub[add_offset],
                                      self.x1_ub[add_offset],
                                      self.x2_ub[add_offset],
                                      repeat_time, 8, 8, 8)

        left_num = compute_num % self.vector_mask_max
        if left_num > 0:
            add_offset = (compute_num // self.vector_mask_max
                          * self.vector_mask_max)
            self.tik_instance.vec_add(left_num,
                                      self.y_ub[add_offset],
                                      self.x1_ub[add_offset],
                                      self.x2_ub[add_offset],
                                      1, 8, 8, 8)

        self.tik_instance.data_move(self.y_gm[y_offset],
                                    self.y_ub, 0, 1, burst_len, 0, 0)

        return

    def compute_offset(self, batch_id, x_id):
        """
        compute the offset of input and output tensors

        Parameters
        ----------
        batch_id: batch index
        x_id: indicates which input tensor; can be "x1" or "x2"

        Returns
        -------
        None
        """
        # C0
        c1_dim_size = self.input_dict[x_id]["shape"][1]
        # H * W * C0
        hwc0 = reduce(lambda x, y: x * y, self.input_dict[x_id]["shape"][2:])
        c1_offset = self.input_dict[x_id + "_c1_offset"]

        offset = self.tik_instance.Scalar("int32")
        offset.set_as(c1_offset*hwc0 + batch_id*c1_dim_size*hwc0)
        return offset
