# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
fractal_z_3d_2_ncdhw
"""
from functools import reduce as functools_reduce
from te import platform as cce
import te.platform.cce_params as cce_params
from te import tik
from te.utils.op_utils import *

# available ub size
UB_SIZE_B = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)
# available number of cores
AICORE_NUM = cce.cce_conf.get_soc_spec(cce.cce_conf.CORE_NUM)


# pylint: disable=locally-disabled,too-many-lines,too-many-locals
def _ceil_div(value, block):
    """
    integrate the input value by block

    """
    return (value + block - 1) // block


def _ceil_fill(value, block):
    """
    fill the input value by block

    """
    return _ceil_div(value, block)*block


def _gm_to_ub_one(args):
    """
    move data from GM to UB one scene

    """
    tik_instance, data, data_ub, data_offset, ub_offset, ori_nburst, \
    burst_len, src_stride, dst_stride, cp_align_len = args

    if src_stride <= 65535:
        if ori_nburst <= 4095:
            tik_instance.data_move(data_ub[ub_offset],
                                   data[data_offset],
                                   0, ori_nburst,
                                   burst_len,
                                   src_stride, dst_stride)

        else:
            n_burst = 4095
            c_cycle = ori_nburst // n_burst
            c_mod = ori_nburst % n_burst
            for num_cy in range(c_cycle):
                data_cur = data_offset + (burst_len + src_stride)\
                           * cp_align_len * n_burst * num_cy
                ub_cur = ub_offset + (burst_len + dst_stride)\
                         * cp_align_len * n_burst * num_cy
                tik_instance.data_move(
                    data_ub[ub_cur],
                    data[data_cur],
                    0, n_burst,
                    burst_len,
                    src_stride, dst_stride)

            if c_mod > 0:
                data_cur = data_offset + (burst_len + src_stride)\
                           * cp_align_len * n_burst * c_cycle
                ub_cur = ub_offset + (burst_len + dst_stride)\
                         * cp_align_len * n_burst * c_cycle
                tik_instance.data_move(
                    data_ub[ub_cur],
                    data[data_cur],
                    0, c_mod,
                    burst_len,
                    src_stride, dst_stride)

    else:
        for num_nb in range(ori_nburst):
            data_cur = data_offset + (burst_len + src_stride)\
                       * cp_align_len * num_nb
            ub_cur = ub_offset + (burst_len + dst_stride)\
                     * cp_align_len * num_nb
            tik_instance.data_move(
                data_ub[ub_cur],
                data[data_cur],
                0, 1,
                burst_len,
                0, 0)


def _ub_to_gm_one(args):
    """
    function of moving data from ub to gm

    """
    tik_instance, dst, data_res, dst_offset, res_offset, ori_nburst, \
    burst_len, src_stride, dst_stride, cp_align_len = args

    if dst_stride <= 65535:
        if ori_nburst <= 4095:
            tik_instance.data_move(
                dst[dst_offset],
                data_res[res_offset],
                0, ori_nburst, burst_len,
                src_stride, dst_stride)

        else:
            n_burst = 4095
            c_cycle = ori_nburst // n_burst
            c_mod = ori_nburst % n_burst

            for num_cy in range(c_cycle):
                dst_cur = dst_offset + (burst_len + dst_stride)\
                          * cp_align_len * n_burst * num_cy
                res_cur = res_offset + (burst_len + src_stride)\
                          * cp_align_len * n_burst * num_cy

                tik_instance.data_move(
                    dst[dst_cur],
                    data_res[res_cur],
                    0, n_burst, burst_len,
                    src_stride, dst_stride)

            if c_mod > 0:
                dst_cur = dst_offset + (burst_len + dst_stride)\
                          * cp_align_len * n_burst * c_cycle
                res_cur = res_offset + (burst_len + src_stride)\
                          * cp_align_len * n_burst * c_cycle

                tik_instance.data_move(
                    dst[dst_cur],
                    data_res[res_cur],
                    0, c_mod, burst_len,
                    src_stride, dst_stride)

    else:
        for num_nb in range(ori_nburst):
            dst_cur = dst_offset + (burst_len + dst_stride)\
                      * cp_align_len * num_nb
            res_cur = res_offset + (burst_len + src_stride)\
                      * cp_align_len * num_nb

            tik_instance.data_move(
                dst[dst_cur],
                data_res[res_cur],
                0, 1, burst_len,
                0, 0)


def _set_core_num(origin_num):
    """
    function of set core num
    """
    if origin_num < AICORE_NUM:
        return origin_num
    return AICORE_NUM


def _set_loop(tik_instance, num_core, max_core, total_dim):
    """
    function of set loop
    """
    core_loop = tik_instance.Scalar("uint64")

    with tik_instance.if_scope(num_core < total_dim % AICORE_NUM):
        core_loop.set_as(_ceil_div(total_dim, max_core))
    with tik_instance.else_scope():
        core_loop.set_as(total_dim // max_core)

    return core_loop


# pylint: disable=locally-disabled,too-many-instance-attributes
# pylint: disable=locally-disabled,old-style-class,too-many-statements
class Fz3d2NcdhwCompute:
    """
    Rearranges data from FRACTAL_Z_3D format to NCDHW format

    Returns
    -------
    None
    """
    def __init__(self, src_shape, dst_shape, dtype, kernel_name):
        """
        initialize some properties
        """
        self.src_shape = list(src_shape)
        self.dst_shape = list(dst_shape)
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.float_size = cce.cce_intrin.get_bit_len(dtype) // 8
        self.cp_align_len = cce_params.BLOCK_REDUCE_INT8 // self.float_size
        self.ub_ele = ((UB_SIZE_B - 64) // self.float_size // 2
                       // self.cp_align_len) * self.cp_align_len
        self.n_o = self.src_shape[1]
        self.n_i = self.src_shape[2]
        self.c_0 = self.src_shape[3]
        self.c_1 = self.calc_c1()
        self.src_gm = None
        self.dst_gm = None

    def calc_c1(self):
        """
        function of calculating c_1
        """
        dc1hw_s = self.src_shape[0]
        _, _, d_d, h_d, w_d = self.dst_shape
        c_1 = dc1hw_s // (d_d * h_d * w_d)
        return c_1

    def func_not_align_split_nc_fp16(self, args):
        """
        function of moving data for dhw_not_align_split_nc_fp16 scene
        """
        tik_instance, ub_ori, ub_trans, ub_tail, n_before, n_len = args

        _, c_d, d_d, h_d, w_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        hw_d = h_d * w_d

        data_offset = n_before * self.c_0
        ub_offset = 0
        ori_nburst = dhw_d * self.c_1
        burst_len = n_len * self.c_0 // self.cp_align_len
        src_stride = (self.n_o * self.n_i - n_len)\
                     * self.c_0 // self.cp_align_len
        dst_stride = 0
        args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset,\
               ori_nburst, burst_len, src_stride, dst_stride, self.cp_align_len
        _gm_to_ub_one(args)

        with tik_instance.for_range(0, d_d) as num_d:
            with tik_instance.for_range(0, hw_d) as num_hw:
                with tik_instance.for_range(0, n_len) as num_n:
                    ori_begin = num_d * self.c_1 * hw_d * n_len * self.c_0 \
                                + num_hw * n_len * self.c_0 + num_n * self.c_0
                    trans_begin = num_d * hw_d * n_len\
                                  * c_d * self.cp_align_len\
                                  + num_hw * n_len * c_d * self.cp_align_len\
                                  + num_n * c_d * self.cp_align_len
                    src_list = [ub_ori[ori_begin + self.c_0 * i]
                                for i in range(16)]
                    dst_list = [ub_trans[trans_begin + self.cp_align_len * i]
                                for i in range(16)]
                    repeat = self.c_1

                    if repeat == 1:
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list, repeat, 0, 0)
                    else:
                        src_rep_stride = hw_d * n_len\
                                         * self.c_0 // self.cp_align_len
                        dst_rep_stride = 16
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list,
                                               repeat,
                                               dst_rep_stride,
                                               src_rep_stride)

        with tik_instance.for_range(0, dhw_d) as num_dhw:
            src_offset = num_dhw * n_len * c_d * self.cp_align_len
            dst_offset = num_dhw * self.cp_align_len
            n_burst = n_len * c_d
            burst_len = 1
            src_stride = 0
            dst_stride = dhw_d - 1
            tik_instance.data_move(ub_ori[dst_offset],
                                   ub_trans[src_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        dim_ele = dhw_d * n_len * c_d
        dim_zu = _ceil_div(dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        dst_offset = n_before * c_d * dhw_d
        if dim_ele % self.cp_align_len > 0:
            if dim_ele > self.cp_align_len:
                sub_ele = dim_ele - self.cp_align_len
                burst_len = _ceil_div(sub_ele, self.cp_align_len)
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, burst_len, 0, 0)
                for k in range(self.cp_align_len):
                    ub_tail[k] = ub_trans[sub_ele + k]

                tik_instance.data_move(self.dst_gm[dst_offset + sub_ele],
                                       ub_tail,
                                       0, 1, 1, 0, 0)
            else:
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, 1, 0, 0)

        else:
            burst_len = dim_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def dhw_not_align_split_nc_fp16(self, tik_instance):
        """
        DC1HW,No,Ni,C0 > ub_ele and C1C0 < ub_ele
        """
        n_d, _, d_d, h_d, w_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        nc_one = self.ub_ele // dhw_d
        c_align = self.c_1 * self.c_0
        n_ub = nc_one // self.cp_align_len // c_align

        all_core = _ceil_div(n_d, n_ub)
        ac_num = _set_core_num(all_core)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (self.ub_ele,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (self.ub_ele,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            ub_tail = tik_instance.Tensor(self.dtype,
                                          (self.cp_align_len,),
                                          name="ub_tail",
                                          scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core

                with tik_instance.if_scope(core_index < all_core - 1):
                    n_len = n_ub
                    n_before = n_ub * core_index
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           n_before, n_len
                    self.func_not_align_split_nc_fp16(args)

                with tik_instance.else_scope():
                    n_before = (all_core - 1) * n_ub
                    n_len = n_d - n_before
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           n_before, n_len
                    self.func_not_align_split_nc_fp16(args)

        return tik_instance

    def func_dhw_not_align_split_c_fp16(self, args):
        """
        function of moving data for dhw_not_align_split_c_fp16 scene
        """
        tik_instance, ub_ori, ub_trans, ub_tail,\
        n_index, c1_before, c1_len, c_before, c_len = args

        _, c_d, d_d, h_d, w_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        hw_d = h_d * w_d

        with tik_instance.for_range(0, d_d) as num_d:
            data_offset = num_d * self.c_1 * hw_d * self.n_o\
                          * self.n_i * self.c_0 \
                          + c1_before * hw_d * self.n_o\
                          * self.n_i * self.c_0 \
                          + n_index * self.c_0
            ub_offset = num_d * c1_len * hw_d * self.c_0
            ori_nburst = hw_d * c1_len
            burst_len = self.c_0 // self.cp_align_len
            src_stride = (self.n_o * self.n_i - 1)\
                         * self.c_0 // self.cp_align_len
            dst_stride = 0
            args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset,\
                   ori_nburst, burst_len, src_stride, dst_stride,\
                   self.cp_align_len
            _gm_to_ub_one(args)

        with tik_instance.for_range(0, d_d) as num_d:
            with tik_instance.for_range(0, hw_d) as num_hw:
                ori_begin = num_d * c1_len * hw_d * self.c_0 \
                             + num_hw * self.c_0
                trans_begin = num_d * hw_d * c_len * self.cp_align_len \
                               + num_hw * c_len * self.cp_align_len
                src_list = [ub_ori[ori_begin + 16 * i]
                            for i in range(16)]
                dst_list = [ub_trans[trans_begin + 16 * i]
                            for i in range(16)]
                repeat = c1_len

                if repeat == 1:
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list, repeat, 0, 0)
                else:
                    src_rep_stride = hw_d * self.c_0 // self.cp_align_len
                    dst_rep_stride = 16
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list,
                                           repeat,
                                           dst_rep_stride,
                                           src_rep_stride)

        with tik_instance.for_range(0, dhw_d) as num_dhw:
            src_offset = num_dhw * c_len * self.cp_align_len
            dst_offset = num_dhw * self.cp_align_len
            n_burst = c_len
            burst_len = 1
            src_stride = 0
            dst_stride = dhw_d - 1
            tik_instance.data_move(ub_ori[dst_offset],
                                   ub_trans[src_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        dim_ele = dhw_d * c_len
        dim_zu = _ceil_div(dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        dst_offset = n_index * c_d * dhw_d + c_before * dhw_d
        if dim_ele % self.cp_align_len > 0:
            sub_ele = dim_ele - self.cp_align_len
            burst_len = _ceil_div(sub_ele, self.cp_align_len)
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)
            for k in range(16):
                ub_tail[k] = ub_trans[sub_ele + k]

            tik_instance.data_move(self.dst_gm[dst_offset + sub_ele],
                                   ub_tail,
                                   0, 1, 1, 0, 0)
        else:
            burst_len = dim_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def dhw_not_align_split_c_fp16(self, tik_instance):
        """
        DC1HW,No,Ni,C0 > ub_ele and C1C0 < ub_ele
        """
        n_d, c_d, d_d, h_d, w_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        nc_one = self.ub_ele // dhw_d
        c1_ub = nc_one // self.cp_align_len // self.c_0

        c1_zu = _ceil_div(self.c_1, c1_ub)
        all_core = n_d * c1_zu
        ac_num = _set_core_num(all_core)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor("float16",
                                         (self.ub_ele,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (self.ub_ele,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            ub_tail = tik_instance.Tensor("float16",
                                          (16,),
                                          name="ub_tail",
                                          scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                n_index = core_index // c1_zu
                c1_index = core_index % c1_zu

                with tik_instance.if_scope(c1_index < c1_zu - 1):
                    c1_len = c1_ub
                    c1_before = c1_ub * c1_index
                    c_before = c1_before * self.c_0
                    c_len = c1_ub * self.c_0
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           n_index, c1_before, c1_len, c_before, c_len
                    self.func_dhw_not_align_split_c_fp16(args)

                with tik_instance.else_scope():
                    c1_before = (c1_zu - 1) * c1_ub
                    c1_len = self.c_1 - c1_before
                    c_before = c1_before * self.c_0
                    c_len = c_d - c_before
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           n_index, c1_before, c1_len, c_before, c_len
                    self.func_dhw_not_align_split_c_fp16(args)

        return tik_instance

    def func_not_align_split_nc_fp32(self, args):
        """
        function of moving data for dhw_not_align_split_nc_fp32 scene
        """
        tik_instance, ub_ori, ub_trans, ub_tail, n_before, n_len = args

        _, c_d, d_d, h_d, w_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        hw_d = h_d * w_d

        data_offset = n_before * self.c_0
        ub_offset = 0
        ori_nburst = dhw_d * self.c_1
        burst_len = n_len * self.c_0 // self.cp_align_len
        src_stride = (self.n_o * self.n_i - n_len)\
                     * self.c_0 // self.cp_align_len
        dst_stride = 0
        args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset,\
               ori_nburst, burst_len, src_stride, dst_stride, self.cp_align_len
        _gm_to_ub_one(args)

        with tik_instance.for_range(0, d_d) as num_d:
            with tik_instance.for_range(0, hw_d) as num_hw:
                with tik_instance.for_range(0, n_len) as num_n:
                    for num_c0zu in range(2):
                        ori_begin = (num_d * self.c_1 * hw_d
                                     * n_len * self.c_0
                                     + num_hw * n_len * self.c_0
                                     + num_n * self.c_0 + num_c0zu * 8) * 2
                        trans_begin = (num_d * hw_d * n_len
                                       * c_d * 2 * self.cp_align_len
                                       + num_hw * n_len * c_d
                                       * 2 * self.cp_align_len
                                       + num_n * c_d * 2 * self.cp_align_len
                                       + num_c0zu * 16 * self.cp_align_len) * 2
                        src_list = [ub_ori[ori_begin + 16 * i]
                                    for i in range(16)]
                        dst_list = [
                            ub_trans[trans_begin + 16 * i]
                            for i in range(16)]
                        repeat = self.c_1

                        if repeat == 1:
                            tik_instance.vnchwconv(False, False, dst_list,
                                                   src_list, repeat, 0, 0)
                        else:
                            src_rep_stride = hw_d * n_len\
                                             * self.c_0 // self.cp_align_len
                            dst_rep_stride = 32
                            tik_instance.vnchwconv(False, False, dst_list,
                                                   src_list,
                                                   repeat,
                                                   dst_rep_stride,
                                                   src_rep_stride)

        with tik_instance.for_range(0, dhw_d) as num_dhw:
            src_offset = num_dhw * n_len * c_d * 2 * self.cp_align_len * 2
            dst_offset = num_dhw * 2 * self.cp_align_len * 2
            n_burst = n_len * c_d
            burst_len = 2
            src_stride = 0
            dst_stride = (dhw_d - 1) * 2
            tik_instance.data_move(ub_ori[dst_offset],
                                   ub_trans[src_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        dim_ele = dhw_d * n_len * c_d
        d_dim_ele = dim_ele * 2
        dim_zu = _ceil_div(d_dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        dst_offset = n_before * c_d * dhw_d
        if dim_ele % self.cp_align_len > 0:
            if dim_ele > self.cp_align_len:
                sub_ele = dim_ele - self.cp_align_len
                burst_len = _ceil_div(sub_ele, self.cp_align_len)
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, burst_len, 0, 0)
                for k in range(16):
                    ub_tail[k] = ub_trans[sub_ele * 2 + k]
                tik_instance.data_move(self.dst_gm[dst_offset + sub_ele],
                                       ub_tail,
                                       0, 1, 1, 0, 0)
            else:
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, 1, 0, 0)

        else:
            burst_len = dim_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def dhw_not_align_split_nc_fp32(self, tik_instance):
        """
        DC1HW,No,Ni,C0 > ub_ele and C1C0 < ub_ele
        """
        n_d, _, d_d, h_d, w_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        nc_one = self.ub_ele // dhw_d
        c_align = self.c_1 * self.c_0
        n_ub = nc_one // 2 // self.cp_align_len // c_align

        all_core = _ceil_div(n_d, n_ub)
        ac_num = _set_core_num(all_core)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor("float16",
                                         (self.ub_ele * 2,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (self.ub_ele * 2,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            ub_tail = tik_instance.Tensor("float16",
                                          (16,),
                                          name="ub_tail",
                                          scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core

                with tik_instance.if_scope(core_index < all_core - 1):
                    n_len = n_ub
                    n_before = n_ub * core_index
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           n_before, n_len
                    self.func_not_align_split_nc_fp32(args)

                with tik_instance.else_scope():
                    n_before = (all_core - 1) * n_ub
                    n_len = n_d - n_before
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           n_before, n_len
                    self.func_not_align_split_nc_fp32(args)

        return tik_instance

    def func_dhw_not_align_split_c_fp32(self, args):
        """
        function of moving data for dhw_not_align_split_c_fp32 scene
        """
        tik_instance, ub_ori, ub_trans, ub_tail,\
        n_index, c1_before, c1_len, c_before, c_len = args

        _, c_d, d_d, h_d, w_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        hw_d = h_d * w_d

        with tik_instance.for_range(0, d_d) as num_d:
            data_offset = num_d * self.c_1 * hw_d * self.n_o\
                          * self.n_i * self.c_0 \
                          + c1_before * hw_d * self.n_o\
                          * self.n_i * self.c_0 \
                          + n_index * self.c_0
            ub_offset = num_d * c1_len * hw_d * self.c_0 * 2
            ori_nburst = hw_d * c1_len
            burst_len = self.c_0 // self.cp_align_len
            src_stride = (self.n_o * self.n_i - 1)\
                         * self.c_0 // self.cp_align_len
            dst_stride = 0
            args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset,\
                   ori_nburst, burst_len, src_stride, dst_stride,\
                   self.cp_align_len
            _gm_to_ub_one(args)

        with tik_instance.for_range(0, d_d) as num_d:
            with tik_instance.for_range(0, hw_d) as num_hw:
                for num_c0zu in range(2):
                    ori_begin = (num_d * c1_len * hw_d * self.c_0
                                 + num_hw * self.c_0 + num_c0zu * 8) * 2
                    trans_begin = (num_d * hw_d * c_len * 2 * self.cp_align_len
                                   + num_hw * c_len * 2 * self.cp_align_len
                                   + num_c0zu * 16 * self.cp_align_len) * 2
                    src_list = [ub_ori[ori_begin + 16 * i]
                                for i in range(16)]
                    dst_list = [ub_trans[trans_begin + 16 * i]
                                for i in range(16)]
                    repeat = c1_len

                    if repeat == 1:
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list, repeat, 0, 0)
                    else:
                        src_rep_stride = hw_d * self.c_0 // self.cp_align_len
                        dst_rep_stride = 32
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list,
                                               repeat,
                                               dst_rep_stride,
                                               src_rep_stride)

        with tik_instance.for_range(0, dhw_d) as num_dhw:
            src_offset = num_dhw * c_len * 2 * self.cp_align_len * 2
            dst_offset = num_dhw * 2 * self.cp_align_len * 2
            n_burst = c_len
            burst_len = 2
            src_stride = 0
            dst_stride = (dhw_d - 1) * 2
            tik_instance.data_move(ub_ori[dst_offset],
                                   ub_trans[src_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        dim_ele = dhw_d * c_len
        d_dim_ele = dim_ele * 2
        dim_zu = _ceil_div(d_dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        dst_offset = n_index * c_d * dhw_d + c_before * dhw_d
        if dim_ele % self.cp_align_len > 0:
            sub_ele = dim_ele - self.cp_align_len
            burst_len = _ceil_div(sub_ele, self.cp_align_len)
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)
            for k in range(16):
                ub_tail[k] = ub_trans[sub_ele * 2 + k]

            tik_instance.data_move(self.dst_gm[dst_offset + sub_ele],
                                   ub_tail,
                                   0, 1, 1, 0, 0)
        else:
            burst_len = dim_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def dhw_not_align_split_c_fp32(self, tik_instance):
        """
        DC1HW,No,Ni,C0 > ub_ele and C1C0 < ub_ele
        """
        n_d, c_d, d_d, h_d, w_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        nc_one = self.ub_ele // dhw_d
        c1_ub = nc_one // 2 // self.cp_align_len // self.c_0

        c1_zu = _ceil_div(self.c_1, c1_ub)
        all_core = n_d * c1_zu
        ac_num = _set_core_num(all_core)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor("float16",
                                         (self.ub_ele * 2,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor("float16",
                                           (self.ub_ele * 2,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            ub_tail = tik_instance.Tensor("float16",
                                          (16,),
                                          name="ub_tail",
                                          scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                n_index = core_index // c1_zu
                c1_index = core_index % c1_zu

                with tik_instance.if_scope(c1_index < c1_zu - 1):
                    c1_len = c1_ub
                    c1_before = c1_ub * c1_index
                    c_before = c1_before * self.c_0
                    c_len = c1_ub * self.c_0
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           n_index, c1_before, c1_len, c_before, c_len
                    self.func_dhw_not_align_split_c_fp32(args)

                with tik_instance.else_scope():
                    c1_before = (c1_zu - 1) * c1_ub
                    c1_len = self.c_1 - c1_before
                    c_before = c1_before * self.c_0
                    c_len = c_d - c_before
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           n_index, c1_before, c1_len, c_before, c_len
                    self.func_dhw_not_align_split_c_fp32(args)

        return tik_instance

    def check_branch(self):
        """
        check which branch of fz3d_2_ncdhw compute
        """
        _, _, d_d, h_d, w_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        nc_one = self.ub_ele // dhw_d
        c_align = self.c_1 * self.c_0

        if self.c_0 * 16 * dhw_d > self.ub_ele:
            return "not_support"

        if self.dtype == "float16":
            n_ub = nc_one // self.cp_align_len // c_align
            if n_ub >= 1:
                return "dhw_not_align_split_nc_fp16"
            else:
                c1_ub = nc_one // self.cp_align_len // self.c_0
                if c1_ub >= 1 and dhw_d >= self.cp_align_len:
                    return "dhw_not_align_split_c_fp16"

        elif self.dtype == "float32":
            n_ub = nc_one // 2 // self.cp_align_len // c_align
            if n_ub >= 1:
                return "dhw_not_align_split_nc_fp32"
            else:
                c1_ub = nc_one // 2 // self.cp_align_len // self.c_0
                if c1_ub >= 1 and dhw_d >= self.cp_align_len:
                    return "dhw_not_align_split_c_fp32"

    def fz3d_2_ncdhw_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        branch = self.check_branch()

        if branch == "dhw_not_align_split_nc_fp16":
            tik_instance = self.dhw_not_align_split_nc_fp16(tik_instance)
        elif branch == "dhw_not_align_split_c_fp16":
            tik_instance = self.dhw_not_align_split_c_fp16(tik_instance)
        elif branch == "dhw_not_align_split_nc_fp32":
            tik_instance = self.dhw_not_align_split_nc_fp32(tik_instance)
        elif branch == "dhw_not_align_split_c_fp32":
            tik_instance = self.dhw_not_align_split_c_fp32(tik_instance)

        return tik_instance

    def set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        src_element_number = functools_reduce(lambda x, y: x * y,
                                              self.src_shape[:])
        dst_element_number = functools_reduce(lambda x, y: x * y,
                                              self.dst_shape[:])
        self.src_gm = tik_instance.Tensor(self.dtype,
                                          (src_element_number,),
                                          name="src_gm",
                                          scope=tik.scope_gm)
        self.dst_gm = tik_instance.Tensor(self.dtype,
                                          (dst_element_number,),
                                          name="dst_gm",
                                          scope=tik.scope_gm)

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_instance = tik.Tik()
        self.set_src_dst_tensor(tik_instance)

        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.fz3d_2_ncdhw_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.src_gm],
                              outputs=[self.dst_gm])

        return tik_instance


def _check_parameters(src, dst, src_format, dst_format):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "fractal_z_3d":
        raise RuntimeError("src_format must be FRACTAL_Z_3D !")

    if dst_format.lower() != "ncdhw":
        raise RuntimeError("dst_format must be NCDHW!")

    check_list = ("float16", "float32")
    check_dtype(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    check_shape(src_shape, min_rank=4, max_rank=4)
    check_shape(dst_shape, min_rank=5, max_rank=5)

    if src_shape[2] != 16:
        raise RuntimeError(
            "the 3rd dimension of src_shape is not 16, Ni must be 16 !")

    if src_shape[3] != 16:
        raise RuntimeError(
            "the 4th dimension of src_shape is not 16, C0 must be 16 !")

    n_d, c_d, d_d, h_d, w_d = dst_shape

    n_i = 16
    n_s = n_i - 1
    n_o = (n_d + n_s) // n_i

    if src_shape[1] != n_o:
        raise RuntimeError(
            "the 2nd dimension of src_shape is wrong, "
            "No must be (N + 15)//16 !")

    c_0 = 16
    c_s = c_0 - 1
    c_1 = (c_d + c_s) // c_0
    one_dim = d_d * c_1 * h_d * w_d

    if src_shape[0] != one_dim:
        raise RuntimeError(
            "the 1st dimension of src_shape is wrong, "
            "it must be D*C1*H*W !")


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR, REQUIRED_ATTR_STR, KERNEL_NAME)
def fractal_z_3d_2_ncdhw(src, dst, src_format, dst_format,
                         kernel_name="fractal_z_3d_2_ncdhw"):
    """
    algorithm: fractal_z_3d_2_ncdhw
    calculating: change data format from FRACTAL_Z_3D to NCDHW

    Parameters
    ----------
    src: dict
        dict with keys(shape, dtype) of src
    dst: dict
        dict with keys(shape, dtype) of dst
    src_format: str
        data format of src, only support "FRACTAL_Z_3D"
    dst_format: str
        data format of dst, only support "NCDHW"
    kernel_name: str
        kernel name, default value is "fractal_z_3d_2_ncdhw"

    Returns
    -------
    tik_instance: tik_instance
    """
    _check_parameters(src, dst, src_format, dst_format)
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype").lower()

    template_fp16 = Fz3d2NcdhwCompute(src_shape, dst_shape, dtype, kernel_name)
    if template_fp16.check_branch() != "not_support":
        return template_fp16.get_tik_instance()
    else:
        raise RuntimeError("not support this kind of transfer")
