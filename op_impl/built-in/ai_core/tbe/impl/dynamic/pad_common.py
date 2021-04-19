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

pad_common
"""
from te import tik
from te import platform as tbe_platform
import math

# maximum of gm
MAX_INT32 = 2**31 - 1
# byte of int32
INT32_BYTE = 4
# numbers in the block
INT32_BLOCK = 8


def _prod(values):
    """
    Prod the input values by multiply.
    """
    res = 1
    for value in values:
        res *= value

    return res


def malloc_tiling_scalar(tik_instance, dtype, name, num):
    """
    return scalar_list which represents shape.
    eg: in_shape is [1,2,3]
        scalar_list is [Scalar0, Scalar1, Scalar2]
        in_shape is [16, 16, 16, 16]
        scalar_list is [Scalar0, Scalar1, Scalar2, Scalar3]
    """
    scalar_list = []
    for i, _ in enumerate(range(num)):
        scalar_list.append(tik_instance.Scalar(dtype, name=name + str(i) + "_"))

    return scalar_list


def _init(buf, scalar_list, begin_idx, length):

    for i, _ in enumerate(range(length)):
        scalar_list[i].set_as(buf[begin_idx+i])


def init_params(obj):
    in_dict = obj.tiling_arg_idx
    _init(obj.tiling_buf, obj.branch, in_dict.get("branch")[0], in_dict.get("branch")[1])
    _init(obj.tiling_buf, obj.depth, in_dict.get("depth")[0], in_dict.get("depth")[1])

    # circulation
    _init(obj.tiling_buf, obj.top_vol, in_dict.get("top_vol")[0], in_dict.get("top_vol")[1])
    _init(obj.tiling_buf, obj.top_address, in_dict.get("top_address")[0], in_dict.get("top_address")[1])
    _init(obj.tiling_buf, obj.top_div_core, in_dict.get("top_div_core")[0], in_dict.get("top_div_core")[1])
    _init(obj.tiling_buf, obj.top_total_core, in_dict.get("top_total_core")[0], in_dict.get("top_total_core")[1])
    _init(obj.tiling_buf, obj.top_core_vol_0, in_dict.get("top_core_vol_0")[0], in_dict.get("top_core_vol_0")[1])
    _init(obj.tiling_buf, obj.top_core_vol_1, in_dict.get("top_core_vol_1")[0], in_dict.get("top_core_vol_1")[1])
    _init(obj.tiling_buf, obj.top_core_gap_0, in_dict.get("top_core_gap_0")[0], in_dict.get("top_core_gap_0")[1])
    _init(obj.tiling_buf, obj.top_core_gap_1, in_dict.get("top_core_gap_1")[0], in_dict.get("top_core_gap_1")[1])

    _init(obj.tiling_buf, obj.bottom_vol, in_dict.get("bottom_vol")[0], in_dict.get("bottom_vol")[1])
    _init(obj.tiling_buf, obj.bottom_address, in_dict.get("bottom_address")[0], in_dict.get("bottom_address")[1])
    _init(obj.tiling_buf, obj.bottom_div_core, in_dict.get("bottom_div_core")[0], in_dict.get("bottom_div_core")[1])
    _init(obj.tiling_buf, obj.bottom_total_core,
          in_dict.get("bottom_total_core")[0], in_dict.get("bottom_total_core")[1])
    _init(obj.tiling_buf, obj.bottom_core_vol_0,
          in_dict.get("bottom_core_vol_0")[0], in_dict.get("bottom_core_vol_0")[1])
    _init(obj.tiling_buf, obj.bottom_core_vol_1,
          in_dict.get("bottom_core_vol_1")[0], in_dict.get("bottom_core_vol_1")[1])
    _init(obj.tiling_buf, obj.bottom_core_gap_0,
          in_dict.get("bottom_core_gap_0")[0], in_dict.get("bottom_core_gap_0")[1])
    _init(obj.tiling_buf, obj.bottom_core_gap_1,
          in_dict.get("bottom_core_gap_1")[0], in_dict.get("bottom_core_gap_1")[1])

    # recursion
    _init(obj.tiling_buf, obj.recur_total_core, in_dict.get("recur_total_core")[0], in_dict.get("recur_total_core")[1])
    _init(obj.tiling_buf, obj.recur_div_core, in_dict.get("recur_div_core")[0], in_dict.get("recur_div_core")[1])
    _init(obj.tiling_buf, obj.recur_in_vol, in_dict.get("recur_in_vol")[0], in_dict.get("recur_in_vol")[1])
    _init(obj.tiling_buf, obj.recur_loop_0, in_dict.get("recur_loop_0")[0], in_dict.get("recur_loop_0")[1])
    _init(obj.tiling_buf, obj.recur_loop_1, in_dict.get("recur_loop_1")[0], in_dict.get("recur_loop_1")[1])
    _init(obj.tiling_buf, obj.recur_gap_0, in_dict.get("recur_gap_0")[0], in_dict.get("recur_gap_0")[1])
    _init(obj.tiling_buf, obj.recur_gap_1, in_dict.get("recur_gap_1")[0], in_dict.get("recur_gap_1")[1])
    _init(obj.tiling_buf, obj.recur_cond, in_dict.get("recur_cond")[0], in_dict.get("recur_cond")[1])
    _init(obj.tiling_buf, obj.recur_start_address,
          in_dict.get("recur_start_address")[0], in_dict.get("recur_start_address")[1])

    _init(obj.tiling_buf, obj.recur_model, in_dict.get("recur_model")[0], in_dict.get("recur_model")[1])
    _init(obj.tiling_buf, obj.recur_dup_mk, in_dict.get("recur_dup_mk")[0], in_dict.get("recur_dup_mk")[1])
    _init(obj.tiling_buf, obj.prod_new_out, in_dict.get("prod_new_out")[0], in_dict.get("prod_new_out")[1])
    _init(obj.tiling_buf, obj.prod_new_in, in_dict.get("prod_new_in")[0], in_dict.get("prod_new_in")[1])
    _init(obj.tiling_buf, obj.recur_gm2buf_mk, in_dict.get("recur_gm2buf_mk")[0], in_dict.get("recur_gm2buf_mk")[1])
    _init(obj.tiling_buf, obj.new_in_shape, in_dict.get("new_in_shape")[0], in_dict.get("new_in_shape")[1])
    _init(obj.tiling_buf, obj.new_out_shape, in_dict.get("new_out_shape")[0], in_dict.get("new_out_shape")[1])
    _init(obj.tiling_buf, obj.new_padding_top, in_dict.get("new_padding_top")[0], in_dict.get("new_padding_top")[1])
    _init(obj.tiling_buf, obj.new_padding_bottom,
          in_dict.get("new_padding_bottom")[0], in_dict.get("new_padding_bottom")[1])


def calc_axis_amount(pads, mark):
    """
    return real length of pads after fused.
    regulation of fused:
    mark: True -> make fused, False -> don't fused
    1.[[x1,x2], [0,0]] => [[y1, y2]]
    2.[[0,0], [0,0]] => [[0, 0]]
    """
    # pads is list, not tuple
    # assume input is [1,1,..1,1]
    num = len(pads)
    if not mark:
        new_pad = pads.copy()
    else:
        index0 = num - 1
        index1 = num - 2
        new_pad = []

        while index1 >= 0:
            if [pads[index1], pads[index0]] == [[0, 0], [0, 0]]:
                num -= 1
                index0 -= 1
                index1 -= 1

            elif pads[index1] != [0, 0] and pads[index0] == [0, 0]:
                new_pad.append(pads[index1])
                num -= 1
                index0 -= 2
                index1 -= 2

            else:
                new_pad.append(pads[index0])
                index0 -= 1
                index1 -= 1

        for i, _ in enumerate(range(num - len(new_pad))):
            if i == (num - len(new_pad) - 1):
                new_pad.append(pads[0])
            else:
                new_pad.append([0, 0])

        new_pad.reverse()

    return num, new_pad


def make_dict(num0, num1, list0, list1):
    in_dict = {}
    for key, value in enumerate(list0):
        in_dict.setdefault(value, [key*num0, num0])

    length = len(list0)
    for key, value in enumerate(list1):
        in_dict.setdefault(value, [length+key*num1, num1])

    return in_dict


class PadInit(object):

    def __init__(self, padding, dtype, kernel_name, tik_obj, fuse_mark):
        """
        Function: store pad_d's parameters of compilation
        """
        self.dtype = dtype.lower()
        self.ori_padding = padding.copy()
        self.padding = padding.copy()
        self.kernel_name = kernel_name
        self.num_bit = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        self.fuse_mark = fuse_mark

        self.mask = 128
        if self.num_bit == 4:
            self.mask = 64
        self.max_ub_size = tik.Dprofile().get_unified_buffer_size() - 1024
        self.max_core = tik.Dprofile().get_aicore_num()

        self.tiling_gm = None
        self.input_gm = None
        self.output_gm = None

        self.tiling_buf = None
        self.tiling_buf_size = None
        self.buf = None
        self.buf_size = None
        self.help_buf = None
        self.tik_instance = tik_obj

        # circulation
        self.axis_amount = None
        self.branch = None
        self.depth = None

        self.top_vol = None
        self.top_address = None
        self.top_div_core = None
        self.top_total_core = None
        self.top_core_vol_0 = None
        self.top_core_vol_1 = None
        self.top_core_gap_0 = None
        self.top_core_gap_1 = None

        self.bottom_vol = None
        self.bottom_address = None
        self.bottom_div_core = None
        self.bottom_total_core = None
        self.bottom_core_vol_0 = None
        self.bottom_core_vol_1 = None
        self.bottom_core_gap_0 = None
        self.bottom_core_gap_1 = None

        # recursion
        self.recur_total_core = None
        self.recur_div_core = None
        self.recur_in_vol = None
        self.recur_loop_0 = None
        self.recur_loop_1 = None
        self.recur_gap_0 = None
        self.recur_gap_1 = None
        self.recur_cond = None
        self.recur_start_address = None

        self.new_in_shape = None
        self.new_out_shape = None
        self.new_padding_top = None
        self.new_padding_bottom = None
        self.recur_model = None
        self.recur_dup_mk = None
        self.recur_gm2buf_mk = None
        self.prod_new_in = None
        self.prod_new_out = None

        self.tiling_arg_kind = None
        self.tiling_arg_num = None
        self.tiling_arg_idx = None

    def set_tik_instance(self):
        """
        set tik_instance:
        1. set_tiling_args: malloc scalar and assure tiling_arg_num
        2. malloc ub_tensor: tiling_buf and buf
        3. set_src_dst_gm: input_gm, tiling_gm and output_gm
        """
        self.set_tiling_args(self.tik_instance)
        self.set_ub_tensor(self.tik_instance)
        self.set_src_dst_gm(self.tik_instance)

    def set_tiling_args(self, tik_instance):
        """
        set runtime container of params:
        1. kinds of params
        2. numbers of params
        3. begin index of each param in tiling_buf
        4. length of different param in tiling_buf
        """
        # Params In Circulation Layer:
        self.axis_amount, self.padding = calc_axis_amount(self.padding, self.fuse_mark)
        self.branch = malloc_tiling_scalar(tik_instance, "int32", "branch_", 1)
        self.depth = malloc_tiling_scalar(tik_instance, "int32", "depth_", 1)
        self.top_vol = malloc_tiling_scalar(tik_instance, "int32", "top_vol_", self.axis_amount)
        self.top_address = malloc_tiling_scalar(tik_instance, "int32", "top_address_", self.axis_amount)
        self.top_div_core = malloc_tiling_scalar(tik_instance, "int32", "top_div_core_", self.axis_amount)
        self.top_total_core = malloc_tiling_scalar(tik_instance, "int32", "top_total_core_", self.axis_amount)
        self.top_core_vol_0 = malloc_tiling_scalar(tik_instance, "int32", "top_core_vol_0_", self.axis_amount)
        self.top_core_vol_1 = malloc_tiling_scalar(tik_instance, "int32", "top_core_vol_1_", self.axis_amount)
        self.top_core_gap_0 = malloc_tiling_scalar(tik_instance, "int32", "top_core_gap_0_", self.axis_amount)
        self.top_core_gap_1 = malloc_tiling_scalar(tik_instance, "int32", "top_core_gap_1_", self.axis_amount)

        self.bottom_vol = malloc_tiling_scalar(tik_instance, "int32", "bottom_vol_", self.axis_amount)
        self.bottom_address = malloc_tiling_scalar(tik_instance, "int32", "bottom_address_", self.axis_amount)
        self.bottom_div_core = malloc_tiling_scalar(tik_instance, "int32", "bottom_div_core_", self.axis_amount)
        self.bottom_total_core = malloc_tiling_scalar(tik_instance, "int32", "bottom_total_core_", self.axis_amount)
        self.bottom_core_vol_0 = malloc_tiling_scalar(tik_instance, "int32", "bottom_core_vol_0_", self.axis_amount)
        self.bottom_core_vol_1 = malloc_tiling_scalar(tik_instance, "int32", "bottom_core_vol_1_", self.axis_amount)
        self.bottom_core_gap_0 = malloc_tiling_scalar(tik_instance, "int32", "bottom_core_gap_0_", self.axis_amount)
        self.bottom_core_gap_1 = malloc_tiling_scalar(tik_instance, "int32", "bottom_core_gap_1_", self.axis_amount)

        # Params In Recursion Layer:
        self.recur_total_core = malloc_tiling_scalar(tik_instance, "int32", "recur_total_core_", 1)
        self.recur_div_core = malloc_tiling_scalar(tik_instance, "int32", "recur_div_core_", 1)
        self.recur_in_vol = malloc_tiling_scalar(tik_instance, "int32", "recur_in_vol_", 1)
        self.recur_loop_0 = malloc_tiling_scalar(tik_instance, "int32", "recur_loop_0_", 1)
        self.recur_loop_1 = malloc_tiling_scalar(tik_instance, "int32", "recur_loop_1_", 1)
        self.recur_gap_0 = malloc_tiling_scalar(tik_instance, "int32", "recur_gap_0_", 1)
        self.recur_gap_1 = malloc_tiling_scalar(tik_instance, "int32", "recur_gap_1_", 1)
        self.recur_cond = malloc_tiling_scalar(tik_instance, "int32", "recur_cond_", 1)
        self.recur_start_address = malloc_tiling_scalar(tik_instance, "int32", "recur_start_address_", 1)

        self.new_in_shape = malloc_tiling_scalar(tik_instance, "int32", "new_in_shape_", self.axis_amount)
        self.new_out_shape = malloc_tiling_scalar(tik_instance, "int32", "new_out_shape", self.axis_amount)
        self.new_padding_top = malloc_tiling_scalar(tik_instance, "int32", "new_padding_top_", self.axis_amount)
        self.new_padding_bottom = malloc_tiling_scalar(tik_instance, "int32", "new_padding_bottom_", self.axis_amount)
        self.recur_model = malloc_tiling_scalar(tik_instance, "int32", "recur_model_", self.axis_amount)
        self.recur_dup_mk = malloc_tiling_scalar(tik_instance, "int32", "recur_dup_mk_", self.axis_amount)
        self.recur_gm2buf_mk = malloc_tiling_scalar(tik_instance, "int32", "recur_gm2buf_mk_", self.axis_amount)
        self.prod_new_in = malloc_tiling_scalar(tik_instance, "int32", "prod_new_in_", self.axis_amount)
        self.prod_new_out = malloc_tiling_scalar(tik_instance, "int32", "prod_new_out_", self.axis_amount)

        # avoid exceed axis: prod_value = 1
        self.prod_new_out.append(1)
        self.prod_new_in.append(1)

        # last param
        self.tiling_arg_kind = 25 + 11
        self.tiling_arg_num = self.axis_amount * 25 + 11
        self.tiling_buf_size = math.ceil(self.tiling_arg_num/INT32_BLOCK) * INT32_BLOCK

        # "Param":list represent begin idx of the param and length in tiling_buf
        num0, num1 = 1, self.axis_amount

        list0 = ["branch", "depth", "recur_total_core", "recur_div_core", "recur_in_vol", "recur_loop_0",
                 "recur_loop_1", "recur_gap_0", "recur_gap_1", "recur_cond", "recur_start_address"]

        list1 = ["top_vol", "top_address", "top_div_core", "top_total_core", "top_core_vol_0",
                 "top_core_vol_1", "top_core_gap_0", "top_core_gap_1",
                 "bottom_vol", "bottom_address", "bottom_div_core", "bottom_total_core",
                 "bottom_core_vol_0", "bottom_core_vol_1", "bottom_core_gap_0", "bottom_core_gap_1",
                 "recur_model", "recur_dup_mk", "recur_gm2buf_mk", "prod_new_out", "prod_new_in",
                 "new_in_shape", "new_out_shape", "new_padding_top", "new_padding_bottom"]

        self.tiling_arg_idx = make_dict(num0, num1, list0, list1)

    def set_ub_tensor(self, tik_instance):
        """
        set buf tensor(UB)
        """
        self.tiling_buf = tik_instance.Tensor("int32",
                                              (self.tiling_buf_size, ),
                                              name="tiling_buf",
                                              scope=tik.scope_ubuf)

        tiling_args_byte = self.tiling_buf_size * INT32_BYTE
        ub_byte = self.max_ub_size - tiling_args_byte
        self.buf_size = ub_byte // self.num_bit // self.mask * self.mask
        self.buf = tik_instance.Tensor(self.dtype,
                                       (self.buf_size, ),
                                       name="buf",
                                       scope=tik.scope_ubuf)
        self.help_buf = tik_instance.Tensor(self.dtype,
                                            (16, ), name="help_buf",
                                            scope=tik.scope_ubuf)

    def set_src_dst_gm(self, tik_instance):
        """
        set tiling, input, output tensor(gm)
        """
        self.tiling_gm = tik_instance.Tensor("int32", (self.tiling_buf_size, ),
                                             name="tiling_gm", scope=tik.scope_gm)

        self.input_gm = tik_instance.Tensor(self.dtype, (MAX_INT32, ),
                                            name="input_gm", scope=tik.scope_gm)

        self.output_gm = tik_instance.Tensor(self.dtype, (MAX_INT32, ),
                                             name="output_gm", scope=tik.scope_gm)
