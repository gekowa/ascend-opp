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

index_add
"""

import math
from functools import reduce as functools_reduce
from te import tik
from te import platform as tbe_platform
from topi.cce import util
from te.utils import op_utils
from te.platform.fusion_manager import fusion_manager

# neg two
NEG_TWO = -2
# neg one
NEG_ONE = -1
# ub reserver size
UB_RESERVE_SIZE = 8 * 1024


class ScatterAxis():
    """
       Function: scatter axis
       Modify : 2020-8
    """

    def __init__(self, var, indices, updates, var_out, axis, kernel_name, compute_type):
        """
        Init scatter axis parameters

        Parameters
        ----------
        var: dict
            data of input
            datatype suports float32,float16,int32,int8,uint8
        indices: dict
            data of indices
            datatype supports int32
        updates: dict
            data of updates
            datatype supports float32,float16,int32,int8,uint8
        var_out: dict
            data of input
        axis: bool
            axis
        kernel_name: str
            the name of the operator
        compute_type: str
            the compute type of scatter
        Returns
        -------
            example: var(2, 6, 8, 8) axis=1
            process uint is var[axis:] (6,8,8) slice shape
            small slice shape is var[axis+1:] (8,8)
            slice num is 2 and divide in each core to proc
            each proc of slice data(6,8,8)
            updates_date proc by indices info to copy
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.var_shape = var.get("shape")
        self.var_dtype = var.get("dtype").lower()
        self.indices_shape = indices.get("shape")
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_shape = updates.get("shape")
        self.updates_dtype = updates.get("dtype").lower()
        self.var_ele_num = functools_reduce(lambda x, y: x * y, self.var_shape)
        self.indices_num = functools_reduce(lambda x, y: x * y, self.indices_shape)
        self.updates_num = functools_reduce(lambda x, y: x * y, self.updates_shape)
        self.axis = axis
        self.kernel_name = kernel_name
        self.compute_type = compute_type

        self.ub_size_bytes = (tik.Dprofile().get_unified_buffer_size() - UB_RESERVE_SIZE)
        self.ai_core_num = tik.Dprofile().get_aicore_num()

        self.var_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(self.var_dtype) // 8
        self.indices_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(self.indices_dtype) // 8
        self.var_data_each_block = 32 // self.var_dtype_bytes_size
        self.indices_data_each_block = 32 // self.indices_dtype_bytes_size

        self.check_param(var_out)

        # indices buf size in ub
        self.indices_ub_number = 0
        # var and updates buf size in ub
        self.updates_ub_number = 0

        # slice is var[axis:],  one uint of process
        if axis == 0:
            self.slice_num = 1
        else:
            self.slice_num = functools_reduce(lambda x, y: x * y, self.var_shape[0:axis])
        self.slice_shape = self.var_shape[axis:]
        self.slice_data_num = functools_reduce(lambda x, y: x * y, self.var_shape[axis:])
        self.small_elem_num = self.slice_data_num // self.var_shape[axis]
        self.slice_size = self.slice_data_num * self.var_dtype_bytes_size

        self.max_num_one_repeat = 128
        if self.var_dtype in ("float32", "int32"):
            self.max_num_one_repeat = 64

        # decide block num
        if self.slice_num == 1:
            self.block_num = 1
            self.slice_step = 0
        else:
            self.slice_step = math.ceil(self.slice_num / self.ai_core_num)
            self.block_num = math.ceil(self.slice_num / self.slice_step)

        # each loop data buf now is one slice data var[axis:] date
        self.update_data_num = self.slice_data_num
        self.vconv_dst_dtype = "float16"

        self.init_gm_tensor()
        self.init_ub_tensor_para()
        self.init_scalar_val()

    def init_gm_tensor(self):
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, self.var_shape, name="var_gm", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype, self.indices_shape, name='indices_gm',
                                                   scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.updates_dtype, self.updates_shape, name="updates_gm",
                                                   scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.var_dtype, self.var_shape, name="out_gm", scope=tik.scope_gm)

    def init_scalar_val(self):
        self.var_vconv_ub = None
        self.updates_vconv_ub = None
        self.var_tile_vconv_ub = None
        self.updates_tile_vconv_ub = None

        self.var_ub = None
        self.updates_ub = None
        self.indices_ub = None
        self.var_tile_ub = None
        self.updates_tile_ub = None

        self.updates_move_tile_ub = None

        self.var_read_index = None
        self.updates_read_index = None

        self.elem_tailpart_start = None
        self.ori_index = None
        self.conv_index = None
        self.elem_data_offset = None
        self.elem_total_len = None
        self.elem_end = None
        self.burst_align_offset = None
        self.mid_elem_count = None
        self.elem_tailpart_len = None
        self.elem_burstpart_start = None
        self.elem_burstpart_len = None
        self.elem_burstpart_burstlen = None
        self.elem_tailpart_burstlen = None
        self.write_buf_offset = None
        self.head_elem_len = None
        self.end_elem_len = None

    def init_ub_tensor_para(self):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        indices_size_bytes = self.indices_dtype_bytes_size * self.indices_num
        updates_size_bytes = self.var_dtype_bytes_size * self.update_data_num

        self.indices_ub_number = math.ceil(
            self.indices_num / self.indices_data_each_block) * self.indices_data_each_block

        remail_ub_size_bytes = self.ub_size_bytes - indices_size_bytes

        need_vconv_dtype = ("int8", "uint8")
        if self.var_dtype in need_vconv_dtype:
            vconv_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(self.vconv_dst_dtype)
            vconv_data_each_block = 32 // vconv_dtype_bytes_size
            vconv_size_bytes = (updates_size_bytes // self.var_dtype_bytes_size * vconv_dtype_bytes_size)

            if (updates_size_bytes + vconv_size_bytes) * 2 < remail_ub_size_bytes:
                self.updates_ub_number = math.ceil(
                    self.update_data_num / self.var_data_each_block) * self.var_data_each_block
                self.vconv_ub_number = math.ceil(self.update_data_num / vconv_data_each_block) * vconv_data_each_block
            else:
                self.updates_ub_number = (remail_ub_size_bytes // (vconv_dtype_bytes_size + self.var_dtype_bytes_size)
                                          // 2 // self.var_data_each_block * self.var_data_each_block)
                self.vconv_ub_number = self.updates_ub_number
        else:
            if updates_size_bytes * 2 < remail_ub_size_bytes:
                self.updates_ub_number = math.ceil(
                    self.update_data_num / self.var_data_each_block) * self.var_data_each_block
            else:
                self.updates_ub_number = (remail_ub_size_bytes // self.var_dtype_bytes_size // 2
                                          // self.var_data_each_block * self.var_data_each_block)

    def init_elem_move_info(self):
        # elem move info
        self.elem_tailpart_start = self.tik_instance.Scalar("int32")
        self.elem_tailpart_start.set_as(0)

        self.ori_index = self.tik_instance.Scalar("int32")
        self.ori_index.set_as(0)

        self.conv_index = self.tik_instance.Scalar("int32")
        self.conv_index.set_as(0)

        self.elem_data_offset = self.tik_instance.Scalar("int32")
        self.elem_data_offset.set_as(0)

        self.elem_total_len = self.tik_instance.Scalar("int32")
        self.elem_total_len.set_as(0)

        self.elem_end = self.tik_instance.Scalar("int32")
        self.elem_end.set_as(0)

        self.burst_align_offset = self.tik_instance.Scalar("int32")
        self.burst_align_offset.set_as(0)

        self.mid_elem_count = self.tik_instance.Scalar("int32")
        self.mid_elem_count.set_as(0)

        self.elem_tailpart_len = self.tik_instance.Scalar("int32")
        self.elem_tailpart_len.set_as(0)

        self.elem_burstpart_start = self.tik_instance.Scalar("int32")
        self.elem_burstpart_start.set_as(0)

        self.elem_burstpart_len = self.tik_instance.Scalar("int32")
        self.elem_burstpart_len.set_as(0)

        self.elem_burstpart_burstlen = self.tik_instance.Scalar("int32")
        self.elem_burstpart_burstlen.set_as(0)

        self.elem_tailpart_burstlen = self.tik_instance.Scalar("int32")
        self.elem_tailpart_burstlen.set_as(0)

        self.write_buf_offset = self.tik_instance.Scalar("int32")
        self.write_buf_offset.set_as(0)

        self.head_elem_len = self.tik_instance.Scalar("int32")
        self.head_elem_len.set_as(0)

        self.end_elem_len = self.tik_instance.Scalar("int32")
        self.end_elem_len.set_as(0)

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
        need_vconv_dtype = ("int8", "uint8")
        if self.var_dtype in need_vconv_dtype:
            self.var_vconv_ub = self.tik_instance.Tensor(self.vconv_dst_dtype, (self.vconv_ub_number,),
                                                         name="var_vconv_ub", scope=tik.scope_ubuf)
            self.updates_vconv_ub = self.tik_instance.Tensor(self.vconv_dst_dtype, (self.vconv_ub_number,),
                                                             name="updates_vconv_ub", scope=tik.scope_ubuf)

            self.var_tile_vconv_ub = self.tik_instance.Tensor(self.vconv_dst_dtype, (self.var_data_each_block,),
                                                              name="var_tile_vconv_ub", scope=tik.scope_ubuf)
            self.updates_tile_vconv_ub = self.tik_instance.Tensor(self.vconv_dst_dtype, (self.var_data_each_block,),
                                                                  name="updates_tile_vconv_ub", scope=tik.scope_ubuf)

        self.var_ub = self.tik_instance.Tensor(self.var_dtype, (self.updates_ub_number,), name="var_ub",
                                               scope=tik.scope_ubuf)
        self.updates_ub = self.tik_instance.Tensor(self.updates_dtype, (self.updates_ub_number,), name="updates_ub",
                                                   scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (self.indices_ub_number,), name="indices_ub",
                                                   scope=tik.scope_ubuf)

        self.var_tile_ub = self.tik_instance.Tensor(self.var_dtype, (self.var_data_each_block,), name="var_tile_ub",
                                                    scope=tik.scope_ubuf)
        self.updates_tile_ub = self.tik_instance.Tensor(self.updates_dtype, (self.var_data_each_block,),
                                                        name="updates_tile_ub", scope=tik.scope_ubuf)

        self.updates_move_tile_ub = self.tik_instance.Tensor(self.updates_dtype, (self.var_data_each_block * 3,),
                                                             name="updates_move_tile_ub", scope=tik.scope_ubuf)

        self.var_read_index = self.tik_instance.Scalar("int32")
        self.var_read_index.set_as(0)

        self.updates_read_index = self.tik_instance.Scalar("int32")
        self.updates_read_index.set_as(0)

        self.slice_loop_index = self.tik_instance.Scalar("int32")
        self.slice_loop_index.set_as(0)

        self.slice_loop_count = self.tik_instance.Scalar("int32")
        self.slice_loop_count.set_as(self.slice_num)

        # init elem move info of updates
        self.init_elem_move_info()

        # copy indices to ub of each core
        indices_burst_len = math.ceil(self.indices_ub_number / self.indices_data_each_block)
        self.tik_instance.data_move(self.indices_ub, self.indices_gm, 0, 1, indices_burst_len, 0, 0)

    @fusion_manager.register("index_add")
    def check_param(self, var_out):
        """
        Check parameter

        Parameters
        ----------
        var_out: dict
            data of input
            datatype suports float32,float16,int32,int8,uint8
        Returns
        -------
        None
        """
        var_out_shape = var_out.get("shape")
        var_out_dtype = var_out.get("dtype").lower()
        if var_out_dtype == "bool":
            var_out_dtype = "int8"
        util.check_kernel_name(self.kernel_name)
        util.check_shape_rule(self.var_shape)
        util.check_shape_rule(self.indices_shape)
        util.check_shape_rule(self.updates_shape)

        util.check_tensor_shape_size(self.var_shape)
        util.check_tensor_shape_size(self.indices_shape)
        util.check_tensor_shape_size(self.updates_shape)
        util.check_tensor_shape_size(var_out_shape)

        check_list_indices = ("int32")
        util.check_dtype_rule(self.indices_dtype, check_list_indices)
        check_list_var = ("float16", "float32", "int32", "int8", "uint8")
        util.check_dtype_rule(self.var_dtype, check_list_var)
        util.check_dtype_rule(self.updates_dtype, check_list_var)
        util.check_dtype_rule(var_out_dtype, check_list_var)

        if (self.updates_dtype != self.var_dtype or var_out_dtype != self.var_dtype):
            raise RuntimeError(
                "dtype updates:{} var_out:{} must same as var{}".format(self.updates_dtype, var_out_dtype,
                                                                        self.var_dtype))

        if var_out_shape != self.var_shape:
            raise RuntimeError(
                "var_out's shape:{} must be the same as var's shape:{}".format(var_out_shape, self.var_shape))

        # updates is not support broadcast to var current
        if self.var_shape != self.updates_shape:
            raise RuntimeError(
                "var's shape:{} must same as updates's shape:{}".format(self.updates_shape, self.var_shape))

        if self.axis >= len(self.updates_shape):
            raise RuntimeError("axis:{} must in range updates shapes:{} len:{}".format(self.axis, self.updates_shape,
                                                                                       len(self.updates_shape)))

        # not support indecis is null
        if len(self.indices_shape) != 1:
            raise RuntimeError("indices_shape:{} len:{} must be l".format(self.indices_shape, len(self.indices_shape)))

        if self.indices_shape[0] != self.updates_shape[self.axis]:
            raise RuntimeError("indices:{} != updates.size(axis({})):{}".format(len(self.indices_shape), self.axis,
                                                                                self.updates_shape[self.axis]))

        # indicis now support cut slice to ub
        if (self.indices_dtype_bytes_size * self.indices_num) > (self.ub_size_bytes * 8 // 10):
            raise RuntimeError("indices num:{} large than ub size:{}".format(self.indices_num, self.ub_size_bytes))

    def get_updates_read_index(self, slice_index):
        """
        Calculate the index of the read updates
        Parameters
        ----------
        indices_ub_index:int32
            the index of the currently traversed indices in UB
        Returns
        -------
        None
        """
        read_index = slice_index * self.slice_data_num
        self.updates_read_index.set_as(read_index)

    def get_var_read_index(self, slice_index):
        """
        Calculate the index of the read updates
        Parameters
        ----------
        indices_ub_index:int32
            the index of the currently traversed var in UB
        Returns
        -------
        None
        """
        read_index = slice_index * self.slice_data_num
        self.var_read_index.set_as(read_index)

    def traversing_slices(self):
        """
        Traversing the slice in the slices
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.slice_loop_count) as slice_index:
            self.get_var_read_index((self.slice_loop_index + slice_index))
            self.get_updates_read_index((self.slice_loop_index + slice_index))
            self.calc_updates()

    def calc_updates(self):
        """
        Calculate updates fragment
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        updates_loop = self.update_data_num // self.updates_ub_number
        if updates_loop > 0:
            with self.tik_instance.for_range(0, updates_loop) as loop_index:
                self.calc_updates_small(loop_index * self.updates_ub_number, self.updates_ub_number)

        last_num = self.update_data_num % self.updates_ub_number
        if last_num > 0:
            self.calc_updates_small(updates_loop * self.updates_ub_number, last_num)

    def proc_elem_data_move(self, elem_start, elem_len, ub_offset):
        """
        proc updates small elem date move
        Parameters
        ----------
        elem_start: start addr
        elem_len: copy buf len
        ub_offset: ub offset of write
        Returns
        -------
            small elem is var[axis+1:] data
            small elem = tailpart(begin date, begin addr not align 32) +
            burstpart(else data, begin addr align 32 so data_move)
            updates_ub = updates[elem_data_offset+addr]
            elem_data_offset is calc by offset of indices_ub data
        """
        # set elem start addr
        self.elem_tailpart_start.set_as(elem_start)

        # get elem data offset by indices convert
        self.ori_index.set_as(self.elem_tailpart_start % self.slice_data_num // self.small_elem_num)
        self.conv_index.set_as(self.indices_ub[self.ori_index])
        self.elem_data_offset.set_as((self.conv_index - self.ori_index) * self.small_elem_num)

        self.elem_total_len.set_as(elem_len)
        # get elem end
        self.elem_end.set_as(self.elem_tailpart_start + self.elem_total_len)

        # get burst_align_offset in elem area
        with self.tik_instance.if_scope(elem_start % self.var_data_each_block == 0):
            self.burst_align_offset = elem_start
        with self.tik_instance.else_scope():
            self.burst_align_offset = ((elem_start - 1) // self.var_data_each_block + 1) * self.var_data_each_block

        # calc tailpart burstpart info
        with self.tik_instance.if_scope(self.elem_end < self.burst_align_offset):
            # tailpart all
            self.elem_tailpart_len.set_as(self.elem_total_len)
            # burstpart null
            self.elem_burstpart_start.set_as(self.elem_tailpart_start + self.burst_align_offset)
            self.elem_burstpart_len.set_as(self.elem_total_len - self.elem_tailpart_len)
        with self.tik_instance.if_scope(self.elem_tailpart_start <= self.burst_align_offset):
            self.elem_tailpart_len.set_as(self.burst_align_offset - self.elem_tailpart_start)
            self.elem_burstpart_start.set_as(self.burst_align_offset)
            self.elem_burstpart_len.set_as(self.elem_total_len - self.elem_tailpart_len)
        with self.tik_instance.else_scope():
            self.tik_instance.tikdb.debug_print('"error info please check"+str(self.burst_align_offset)')
            self.elem_burstpart_len.set_as(0)  # elem burst no need

        # get taillen align
        with self.tik_instance.if_scope(self.elem_tailpart_len % self.var_data_each_block == 0):
            self.elem_tailpart_burstlen.set_as(self.elem_tailpart_len // self.var_data_each_block)
        with self.tik_instance.else_scope():
            self.elem_tailpart_burstlen.set_as((self.elem_tailpart_len - 1) // self.var_data_each_block + 1)

        # get burstlen align
        with self.tik_instance.if_scope(self.elem_burstpart_len % self.var_data_each_block == 0):
            self.elem_burstpart_burstlen.set_as(self.elem_burstpart_len // self.var_data_each_block)
        with self.tik_instance.else_scope():
            self.elem_burstpart_burstlen.set_as((self.elem_burstpart_len - 1) // self.var_data_each_block + 1)

        # elem tailpart part not align so copy tail_ub and set_as
        with self.tik_instance.if_scope(self.elem_tailpart_burstlen > 0):
            self.tik_instance.data_move(self.updates_move_tile_ub,
                                        self.updates_gm[self.elem_tailpart_start + self.elem_data_offset], 0, 1,
                                        self.elem_tailpart_burstlen, 0, 0)
            with self.tik_instance.for_range(0, self.elem_tailpart_len) as i:
                self.updates_ub[ub_offset + i].set_as(self.updates_move_tile_ub[i])

        # Elem burstpart proc
        with self.tik_instance.if_scope(self.elem_burstpart_len > 0):
            self.tik_instance.data_move(self.updates_ub[ub_offset + self.elem_tailpart_len],
                                        self.updates_gm[self.elem_burstpart_start + self.elem_data_offset], 0, 1,
                                        self.elem_burstpart_burstlen, 0, 0)

    def move_updates_datas(self, dstbufaddr, dstbuflen):
        """
        move updates dats by indices data

        Parameters
        ----------
        dstbufaddr: gm start addr
        dstbuflen: gm copy len
        ub start addr is 0
        Returns
        -------
         dstbufaddr             dstbuflen
        ----------------------------------
        headElem       middleElem         endElem
        """
        # get headElem len in one elem
        self.head_elem_len.set_as(self.small_elem_num - dstbufaddr % self.slice_data_num % self.small_elem_num)
        with self.tik_instance.if_scope(self.head_elem_len > dstbuflen):
            self.head_elem_len.set_as(dstbuflen)
            self.mid_elem_count.set_as(0)
            self.end_elem_len.set_as(0)
        with self.tik_instance.else_scope():
            # get remail elem loop count
            self.mid_elem_count.set_as((dstbuflen - self.head_elem_len) // self.small_elem_num)
            self.end_elem_len.set_as((dstbuflen - self.head_elem_len) % self.small_elem_num)

        # proc headElem data move
        self.proc_elem_data_move(dstbufaddr, self.head_elem_len, 0)

        # proc middle Elem data move
        with self.tik_instance.if_scope(self.mid_elem_count > 0):
            with self.tik_instance.for_range(0, self.mid_elem_count) as i:
                self.proc_elem_data_move(dstbufaddr + self.head_elem_len + i * self.small_elem_num, self.small_elem_num,
                                         self.head_elem_len + i * self.small_elem_num)

        # proc end Elem data move
        with self.tik_instance.if_scope(self.end_elem_len > 0):
            self.proc_elem_data_move((dstbufaddr + self.head_elem_len + self.mid_elem_count * self.small_elem_num),
                                     self.end_elem_len,
                                     (self.head_elem_len + self.mid_elem_count * self.small_elem_num))

    def calc_updates_small(self, read_index_offset, element_num):
        """
        Transfer update to UB and calculate
        Parameters
        ----------
        read_index_offset: int32
            the offset used to read the updates fragment
        element_num:
            the number of elements in the slice of updates
        Returns
        -------
        None
        """
        updates_burst_len = math.ceil(element_num / self.var_data_each_block)
        self.tik_instance.data_move(self.var_ub, self.var_gm[self.var_read_index + read_index_offset], 0, 1,
                                    updates_burst_len, 0, 0)

        # replace this word by move updates accordding indices
        # func self.tik_instance.data_move(self.updates_ub,
        # func self.updates_gm[self.updates_read_index + read_index_offset],
        # func  0, 1, updates_burst_len, 0, 0)
        self.move_updates_datas(self.updates_read_index + read_index_offset,
                                updates_burst_len * self.var_data_each_block)

        tile_ele_num = element_num % self.var_data_each_block
        align_offset = 0
        if (tile_ele_num != 0 and self.update_data_num > self.var_data_each_block):
            align_ele_num = (element_num // self.var_data_each_block * self.var_data_each_block)
            align_offset = (read_index_offset + align_ele_num - (self.var_data_each_block - tile_ele_num))
            self.tik_instance.data_move(self.var_tile_ub, self.var_gm[self.var_read_index + align_offset], 0, 1, 1, 0,
                                        0)
            self.tik_instance.data_move(self.updates_tile_ub, self.updates_gm[self.updates_read_index + align_offset],
                                        0, 1, 1, 0, 0)

        compute_loop = element_num // self.max_num_one_repeat // 255

        if compute_loop > 0:
            with self.tik_instance.for_range(0, compute_loop) as index:
                index_offset = index * self.max_num_one_repeat * 255
                self.calc_process(self.max_num_one_repeat, index_offset, index_offset, index_offset, 255, False)
        last_loop = element_num % (self.max_num_one_repeat * 255) // self.max_num_one_repeat
        if last_loop > 0:
            index_offset = compute_loop * self.max_num_one_repeat * 255
            self.calc_process(self.max_num_one_repeat, index_offset, index_offset, index_offset, last_loop, False)

        compute_mask = element_num % self.max_num_one_repeat
        if compute_mask > 0:
            index_offset = (element_num // self.max_num_one_repeat * self.max_num_one_repeat)
            if (tile_ele_num == 0 or self.update_data_num < self.var_data_each_block):
                self.calc_process(compute_mask, index_offset, index_offset, index_offset, 1, False)
                self.tik_instance.data_move(self.out_gm[self.var_read_index + read_index_offset], self.var_ub, 0, 1,
                                            updates_burst_len, 0, 0)
            else:
                self.calc_process(self.var_data_each_block, 0, 0, 0, 1, True)
                self.tik_instance.data_move(self.out_gm[self.var_read_index + align_offset], self.var_tile_ub, 0, 1, 1,
                                            0, 0)
                self.calc_process(compute_mask, index_offset, index_offset, index_offset, 1, False)
                self.tik_instance.data_move(self.out_gm[self.var_read_index + read_index_offset], self.var_ub, 0, 1,
                                            updates_burst_len - 1, 0, 0)
        else:
            self.tik_instance.data_move(self.out_gm[self.var_read_index + read_index_offset], self.var_ub, 0, 1,
                                        updates_burst_len, 0, 0)

    def compute_function(self, mask, dst_ub, src1_ub, src2_ub, repeat_times, compute_repeat_strid):
        if self.compute_type == "vadd":
            self.tik_instance.vadd(mask, dst_ub, src1_ub, src2_ub, repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)
        elif self.compute_type == "vsub":
            self.tik_instance.vsub(mask, dst_ub, src1_ub, src2_ub, repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)
        elif self.compute_type == "vdiv":
            if tbe_platform.cce_conf.api_check_support("tik.vdiv", "float32"):
                self.tik_instance.vdiv(mask, dst_ub, src1_ub, src2_ub, repeat_times, 1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)
            else:
                tmp_tensor = self.tik_instance.Tensor(src2_ub.dtype, (mask * repeat_times,), scope=tik.scope_ubuf,
                                                      name="tmp_tensor")
                self.tik_instance.vrec(mask, tmp_tensor, src2_ub, repeat_times, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid)
                self.tik_instance.vmul(mask, src2_ub, src2_ub, tmp_tensor, repeat_times, 1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)
                self.tik_instance.vadds(mask, src2_ub, src2_ub, NEG_TWO, repeat_times, 1, 1, compute_repeat_strid,
                                        compute_repeat_strid)
                self.tik_instance.vmul(mask, src2_ub, src2_ub, tmp_tensor, repeat_times, 1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)
                self.tik_instance.vmuls(mask, src2_ub, src2_ub, NEG_ONE, repeat_times, 1, 1, compute_repeat_strid,
                                        compute_repeat_strid)
                # mul src1_ub * (1/src2_ub)
                self.tik_instance.vmul(mask, dst_ub, src1_ub, src2_ub, repeat_times, 1, 1, 1, compute_repeat_strid,
                                       compute_repeat_strid, compute_repeat_strid)
        elif self.compute_type == "vmax":
            self.tik_instance.vmax(mask, dst_ub, src1_ub, src2_ub, repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)
        elif self.compute_type == "vmin":
            self.tik_instance.vmin(mask, dst_ub, src1_ub, src2_ub, repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)
        elif self.compute_type == "vmul":
            self.tik_instance.vmul(mask, dst_ub, src1_ub, src2_ub, repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)
        elif self.compute_type == "update":
            self.tik_instance.vmuls(mask, dst_ub, src1_ub, 0, repeat_times, 1, 1, compute_repeat_strid,
                                    compute_repeat_strid)
            self.tik_instance.vadd(mask, dst_ub, src1_ub, src2_ub, repeat_times, 1, 1, 1, compute_repeat_strid,
                                   compute_repeat_strid, compute_repeat_strid)
        else:
            raise RuntimeError("the operater [%s] is not supported" %
                               self.compute_type)

    def calc_process(self, mask, dest_addr, src_addr1, src_addr2, repeat_times, is_tile):
        """
        Execute the corresponding calculation instruction
        Parameters
        ----------
        mask: int
            the mask of instruction
        dest_addr: int
            testination address offset
        src_addr1: int
            src1 address offset
        src_addr2: int
            src2 address offset
        repeat_times: int
            the repeat times of instruction
        is_tile: bool
            determine whether the currently calculated data is the tail of var
            and updates
        Returns
        -------
        None
        """
        need_vconv_dtype = ("int8", "uint8")
        if self.var_dtype in need_vconv_dtype:
            if is_tile:
                self.tik_instance.vconv(mask, "", self.var_tile_vconv_ub[dest_addr], self.var_tile_ub[src_addr1],
                                        repeat_times, 1, 1, 8, 4)
                self.tik_instance.vconv(mask, "", self.updates_tile_vconv_ub[dest_addr],
                                        self.updates_tile_ub[src_addr2], repeat_times, 1, 1, 8, 4)
                compute_repeat_strid = 8
                src1_ub = self.var_tile_vconv_ub
                src2_ub = self.updates_tile_vconv_ub
                dst_ub = self.var_tile_vconv_ub
                mask = self.var_data_each_block
            else:
                self.tik_instance.vconv(mask, "", self.var_vconv_ub[dest_addr], self.var_ub[src_addr1], repeat_times, 1,
                                        1, 8, 4)
                self.tik_instance.vconv(mask, "", self.updates_vconv_ub[dest_addr], self.updates_ub[src_addr2],
                                        repeat_times, 1, 1, 8, 4)
                compute_repeat_strid = 8
                src1_ub = self.var_vconv_ub[src_addr1]
                src2_ub = self.updates_vconv_ub[src_addr2]
                dst_ub = self.var_vconv_ub[dest_addr]
        else:
            if is_tile:
                compute_repeat_strid = (self.max_num_one_repeat // self.var_data_each_block)
                src1_ub = self.var_tile_ub
                src2_ub = self.updates_tile_ub
                dst_ub = self.var_tile_ub
                mask = self.var_data_each_block
            else:
                compute_repeat_strid = (self.max_num_one_repeat // self.var_data_each_block)
                src1_ub = self.var_ub[src_addr1]
                src2_ub = self.updates_ub[src_addr2]
                dst_ub = self.var_ub[dest_addr]

        # call compute operation accordding compute_type
        self.compute_function(mask, dst_ub, src1_ub, src2_ub, repeat_times, compute_repeat_strid)

        if self.var_dtype in need_vconv_dtype:
            if is_tile:
                self.tik_instance.vconv(mask, "", self.var_tile_ub, self.var_tile_vconv_ub, repeat_times, 1, 1, 4, 8)
            else:
                self.tik_instance.vconv(mask, "", self.var_ub[src_addr1], self.var_vconv_ub[dest_addr], repeat_times, 1,
                                        1, 4, 8)

    def scatter_axis_operator(self):
        """
        Scatter operation
        Parameters
        ----------
        None
        Returns:
        ----------
        tik_instance: tik instance
        """
        if self.block_num > 1:
            with self.tik_instance.for_range(0, self.block_num, block_num=self.block_num) as index:
                self.init_ub_tensor()
                self.slice_loop_index.set_as(index * self.slice_step)
                self.slice_loop_count.set_as(self.slice_step)
                with self.tik_instance.if_scope(index == self.block_num - 1):
                    self.slice_loop_count.set_as(self.slice_num - self.slice_loop_index)
                self.traversing_slices()
        else:
            self.init_ub_tensor()
            self.traversing_slices()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var_gm, self.indices_gm, self.updates_gm),
                                   outputs=(self.out_gm), enable_l2=False)

        return self.tik_instance


@op_utils.check_op_params(dict, dict, dict, dict, int, str)
def index_add(var, axis_indices, updates, var_out, axis=0, kernel_name="index_add"):
    '''
        index add
    :param var: dict
                input var data
    :param axis_indices: dict
                input indices data
    :param updates: dict
                input updates data
    :param var_out: dict
                output var_out data
    :param axis: int
                axis
    :param kernel_name: str
                  kernel name, default value is "index_add"
    :return: none
    '''
    scatter = ScatterAxis(var, axis_indices, updates, var_out, axis, kernel_name, "vadd")
    scatter.scatter_axis_operator()
