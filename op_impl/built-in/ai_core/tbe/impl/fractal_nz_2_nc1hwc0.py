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
fractal_nz_2_nc1hwc0
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
# maximum repeat number
MAX_REPEATS = 255
# maximum burst number
MAX_BURST_NUMBER = 4095
# maximum rep stride
MAX_STRIDE_REP = 255
# maximum blk stride
MAX_STRIDE_BLK = 65535
# maximum mask
MAX_MASK = 128


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
            with tik_instance.for_range(0, c_cycle) as num_cy:
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
        with tik_instance.for_range(0, ori_nburst) as num_nb:
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

            with tik_instance.for_range(0, c_cycle) as num_cy:
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
        with tik_instance.for_range(0, ori_nburst) as num_nb:
            dst_cur = dst_offset + (burst_len + dst_stride)\
                      * cp_align_len * num_nb
            res_cur = res_offset + (burst_len + src_stride)\
                      * cp_align_len * num_nb

            tik_instance.data_move(
                dst[dst_cur],
                data_res[res_cur],
                0, 1, burst_len,
                0, 0)


def _func_vadds(args):
    """
    function of moving data with vadds function

    """
    tik_instance, data_ub, data_res, dtype, c_0, ub_offset,\
    res_offset, repeat, srcm0, dstm0, srcm1, dstm1, mask = args
    max_r = 255

    scalar_zero = tik_instance.Scalar(dtype=dtype, init_value=0.0)

    if repeat <= max_r:
        if repeat == 1:
            tik_instance.vadds(mask, data_res[res_offset], data_ub[ub_offset],
                               scalar_zero, repeat, dstm0, srcm0, 0, 0)

        else:
            tik_instance.vadds(mask, data_res[res_offset], data_ub[ub_offset],
                               scalar_zero, repeat, dstm0, srcm0,
                               dstm1, srcm1)

    else:
        zu_repeat = repeat // max_r
        mod_repeat = repeat % max_r
        with tik_instance.for_range(0, zu_repeat) as num_zr:
            ub_offset_cur = ub_offset + num_zr*max_r*c_0*srcm1
            res_offset_cur = res_offset + num_zr*max_r*c_0*dstm1
            tik_instance.vadds(mask, data_res[res_offset_cur],
                               data_ub[ub_offset_cur],
                               scalar_zero, max_r, dstm0, srcm0,
                               dstm1, srcm1)

        if mod_repeat > 0:
            ub_offset_cur = ub_offset + zu_repeat*max_r*c_0*srcm1
            res_offset_cur = res_offset + zu_repeat*max_r*c_0*dstm1
            if mod_repeat == 1:
                tik_instance.vadds(mask, data_res[res_offset_cur],
                                   data_ub[ub_offset_cur],
                                   scalar_zero, mod_repeat, dstm0, srcm0,
                                   0, 0)

            else:
                tik_instance.vadds(mask, data_res[res_offset_cur],
                                   data_ub[ub_offset_cur],
                                   scalar_zero, mod_repeat, dstm0, srcm0,
                                   dstm1, srcm1)


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
# pylint: disable=locally-disabled,old-style-class,too-many-return-statements
# pylint: disable=locally-disabled,too-many-statements, too-many-branches
# pylint: disable=locally-disabled,too-many-public-methods
class Fnz2Nc1hwc0Compute:
    """
    Rearranges data from FRACTAL_NZ format to NC1HWC0 format

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
        self.n_true = self.dst_shape[0]
        self.c_1 = self.dst_shape[1]
        self.h_i = self.dst_shape[2]
        self.w_i = self.dst_shape[3]
        self.c_0 = self.dst_shape[4]
        self.no_f = self.src_shape[1]
        self.ni_f = self.src_shape[2]
        self.ni_right = 16
        self.src_gm = None
        self.dst_gm = None

    def ub_to_ub(self, args):
        """
        function of moving data from UB to UB
        """
        tik_instance, ub_ori, ub_trans, one_i, n_i, n_str = args

        with tik_instance.if_scope(one_i < n_i):
            with tik_instance.for_range(0, one_i) as num_o:
                ori_offset = num_o * n_str * self.c_0
                trans_offset = num_o * self.c_0
                n_burst = n_i
                burst_len = self.c_0 // self.cp_align_len
                src_stride = 0
                dst_stride = (one_i - 1) * self.c_0 \
                             // self.cp_align_len
                tik_instance.data_move(ub_trans[trans_offset],
                                       ub_ori[ori_offset],
                                       0, n_burst, burst_len,
                                       src_stride, dst_stride)
        with tik_instance.else_scope():
            with tik_instance.for_range(0, n_i) as num_n:
                ori_offset = num_n * self.c_0
                trans_offset = num_n * one_i * self.c_0
                n_burst = one_i
                burst_len = self.c_0 // self.cp_align_len
                src_stride = (n_str - 1) * self.c_0 \
                             // self.cp_align_len
                dst_stride = 0
                tik_instance.data_move(ub_trans[trans_offset],
                                       ub_ori[ori_offset],
                                       0, n_burst, burst_len,
                                       src_stride, dst_stride)

    def vadds_n_small(self, args):
        """
        function of vadds for n_i <= one_i
        """
        tik_instance, ub_ori, ub_trans, one_i, n_i, n_str = args

        with tik_instance.for_range(0, n_i) as num_n:
            all_repeat = one_i
            repeat = all_repeat // 8
            repeat_mod = all_repeat % 8
            if repeat > 0:
                ub_offset = num_n * self.c_0
                res_offset = num_n * one_i * self.c_0
                srcm0 = n_str
                dstm0 = 1
                srcm1 = 8 * n_str
                dstm1 = 8
                mask = 128
                args = tik_instance, ub_ori, ub_trans, self.dtype, \
                       self.c_0, ub_offset, res_offset, repeat, \
                       srcm0, dstm0, srcm1, dstm1, mask
                _func_vadds(args)
            if repeat_mod > 0:
                ub_offset = n_str * self.c_0 * 8 * repeat \
                            + num_n * self.c_0
                res_offset = self.c_0 * 8 * repeat \
                             + num_n * one_i * self.c_0
                srcm0 = n_str
                dstm0 = 1
                srcm1 = 0
                dstm1 = 0
                mask = repeat_mod * 16
                args = tik_instance, ub_ori, ub_trans, self.dtype, \
                       self.c_0, ub_offset, res_offset, 1, \
                       srcm0, dstm0, srcm1, dstm1, mask
                _func_vadds(args)

    def vadds_o_small(self, args):
        """
        function of vadds for n_i > one_i
        """
        tik_instance, ub_ori, ub_trans, one_i, n_i, n_str = args

        with tik_instance.for_range(0, one_i) as num_o:
            all_repeat = n_i
            repeat = all_repeat // 8
            repeat_mod = all_repeat % 8
            if repeat > 0:
                ub_offset = num_o * self.c_0 * n_str
                res_offset = num_o * self.c_0
                srcm0 = 1
                dstm0 = one_i
                srcm1 = 8
                dstm1 = 8 * one_i
                mask = 128
                args = tik_instance, ub_ori, ub_trans, self.dtype, \
                       self.c_0, ub_offset, res_offset, repeat, \
                       srcm0, dstm0, srcm1, dstm1, mask
                _func_vadds(args)
            if repeat_mod > 0:
                ub_offset = self.c_0 * 8 * repeat \
                            + num_o * self.c_0 * n_str
                res_offset = one_i * self.c_0 * 8 * repeat \
                             + num_o * self.c_0
                srcm0 = 1
                dstm0 = one_i
                srcm1 = 0
                dstm1 = 0
                mask = repeat_mod * 16
                args = tik_instance, ub_ori, ub_trans, self.dtype, \
                       self.c_0, ub_offset, res_offset, 1, \
                       srcm0, dstm0, srcm1, dstm1, mask
                _func_vadds(args)

    def ub_to_ub_o_small(self, args):
        """
        function of ub_to_ub for n_i > one_i
        """
        tik_instance, ub_ori, ub_trans, one_i, n_i, n_str = args

        with tik_instance.for_range(0, one_i) as num_o:
            ori_offset = num_o * n_str * self.c_0
            trans_offset = num_o * self.c_0
            n_burst = n_i
            burst_len = self.c_0 // self.cp_align_len
            src_stride = 0
            dst_stride = (one_i - 1) * self.c_0 \
                         // self.cp_align_len
            tik_instance.data_move(ub_trans[trans_offset],
                                   ub_ori[ori_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

    def ub_to_ub_n_small(self, args):
        """
        function of ub_to_ub for n_i <= one_i
        """
        tik_instance, ub_ori, ub_trans, one_i, n_i, n_str = args

        with tik_instance.for_range(0, n_i) as num_n:
            ori_offset = num_n * self.c_0
            trans_offset = num_n * one_i * self.c_0
            n_burst = one_i
            burst_len = self.c_0 // self.cp_align_len
            src_stride = (n_str - 1) * self.c_0 \
                         // self.cp_align_len
            dst_stride = 0
            tik_instance.data_move(ub_trans[trans_offset],
                                   ub_ori[ori_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

    def ub_to_ub_new(self, args):
        """
        function of moving data from ub to ub with vadds or ub_to_ub
        """
        tik_instance, ub_ori, ub_trans, one_i, n_i, n_str = args

        with tik_instance.if_scope(one_i < n_i):
            dstm1 = 8 * one_i

            if dstm1 <= 255:
                args = tik_instance, ub_ori, ub_trans, one_i, n_i, n_str
                self.vadds_o_small(args)
            else:
                args = tik_instance, ub_ori, ub_trans, one_i, n_i, n_str
                self.ub_to_ub_o_small(args)

        with tik_instance.else_scope():
            srcm1 = 8 * n_str

            if srcm1 <= 255:
                args = tik_instance, ub_ori, ub_trans, one_i, n_i, n_str
                self.vadds_n_small(args)
            else:
                args = tik_instance, ub_ori, ub_trans, one_i, n_i, n_str
                self.ub_to_ub_n_small(args)

    def small_ub(self, tik_instance):
        """
        nc1hwc0 <= ub_ele
        """
        big_src_ele = self.c_1 * self.h_i * self.w_i \
                      * self.no_f * self.ni_f * self.c_0
        big_dst_ele = self.n_true * self.c_1 * self.h_i * self.w_i * self.c_0
        n_align = self.no_f * self.ni_f
        one_d = self.c_1 * self.h_i * self.w_i

        ub_ori = tik_instance.Tensor(self.dtype,
                                     (self.ub_ele,),
                                     name="ub_ori",
                                     scope=tik.scope_ubuf)
        ub_trans = tik_instance.Tensor(self.dtype,
                                       (self.ub_ele,),
                                       name="ub_trans",
                                       scope=tik.scope_ubuf)

        src_offset = 0
        burst_len = big_src_ele // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        args = tik_instance, ub_ori, ub_trans, \
               one_d, self.n_true, n_align
        if self.dtype == "float16":
            self.ub_to_ub_new(args)
        else:
            self.ub_to_ub(args)

        dst_offset = 0
        burst_len = big_dst_ele // self.cp_align_len
        tik_instance.data_move(self.dst_gm[dst_offset],
                               ub_trans,
                               0, 1, burst_len, 0, 0)

        return tik_instance

    def func_small_one_splitn(self, args):
        """
        function of moving data for small one splitn
        """
        tik_instance, ub_ori, ub_trans, n_before, n_cur = args

        one_d = self.c_1 * self.h_i * self.w_i

        data_offset = n_before * self.c_0
        ub_offset = 0
        ori_nburst = one_d
        burst_len = n_cur * self.c_0 // self.cp_align_len
        src_stride = (self.no_f * self.ni_f - n_cur) * self.c_0 // self.cp_align_len
        dst_stride = 0
        args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset,\
               ori_nburst, burst_len, src_stride, dst_stride,\
               self.cp_align_len
        _gm_to_ub_one(args)

        args = tik_instance, ub_ori, ub_trans, \
               one_d, n_cur, n_cur
        if self.dtype == "float16":
            self.ub_to_ub_new(args)
        else:
            self.ub_to_ub(args)

        dst_offset = n_before * one_d * self.c_0
        burst_len = n_cur * one_d * self.c_0 // self.cp_align_len
        tik_instance.data_move(self.dst_gm[dst_offset],
                               ub_trans,
                               0, 1, burst_len, 0, 0)

    def small_one_splitn(self, tik_instance):
        """
        c1hw <= n_true
        c1hw * c_0 <= ub_ele
        """
        c1hwc0 = self.c_1 * self.h_i * self.w_i * self.c_0
        n_ub = self.ub_ele // c1hwc0
        all_core = _ceil_div(self.n_true, n_ub)
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

            ub_loop = _set_loop(tik_instance, num_core, ac_num, all_core)

            with tik_instance.for_range(0, ub_loop)\
                    as num_u:
                core_index = num_u * ac_num + num_core
                n_before = core_index * n_ub

                if self.n_true > n_ub:
                    with tik_instance.if_scope(core_index < all_core - 1):
                        n_cur = n_ub
                        args = tik_instance, ub_ori, ub_trans, n_before, n_cur
                        self.func_small_one_splitn(args)
                    with tik_instance.else_scope():
                        n_cur = self.n_true - ((all_core - 1) * n_ub)
                        args = tik_instance, ub_ori, ub_trans, n_before, n_cur
                        self.func_small_one_splitn(args)
                else:
                    n_cur = self.n_true - ((all_core - 1) * n_ub)
                    args = tik_instance, ub_ori, ub_trans, n_before, n_cur
                    self.func_small_one_splitn(args)

        return tik_instance

    def func_small_n_splito(self, args):
        """
        function of moving data for small n splito
        """
        tik_instance, ub_ori, ub_trans, one_before, one_cur = args

        n_align = self.no_f * self.ni_f
        one_d = self.c_1 * self.h_i * self.w_i

        src_offset = one_before * n_align * self.c_0
        burst_len = one_cur * n_align * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        args = tik_instance, ub_ori, ub_trans, \
               one_cur, self.n_true, n_align
        if self.dtype == "float16":
            self.ub_to_ub_new(args)
        else:
            self.ub_to_ub(args)

        dst_offset = one_before * self.c_0
        res_offset = 0
        ori_nburst = self.n_true
        burst_len = one_cur * self.c_0 // self.cp_align_len
        src_stride = 0
        dst_stride = (one_d - one_cur) * self.c_0 // self.cp_align_len
        args = tik_instance, self.dst_gm, ub_trans, dst_offset, res_offset,\
               ori_nburst, burst_len, src_stride, dst_stride,\
               self.cp_align_len
        _ub_to_gm_one(args)

    def small_n_splito(self, tik_instance):
        """
        one_d > n_true
        no_f * ni_f * c_0 <= ub_ele
        """
        c0n_align = self.no_f * self.ni_f * self.c_0
        one_ub = self.ub_ele // c0n_align
        one_d = self.c_1 * self.h_i * self.w_i

        all_core = _ceil_div(one_d, one_ub)
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

            ub_loop = _set_loop(tik_instance, num_core, ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                one_before = core_index * one_ub

                with tik_instance.if_scope(core_index < all_core - 1):
                    one_cur = one_ub
                    args = tik_instance, ub_ori, ub_trans, one_before, one_cur
                    self.func_small_n_splito(args)
                with tik_instance.else_scope():
                    one_cur = one_d - ((all_core - 1) * one_ub)
                    args = tik_instance, ub_ori, ub_trans, one_before, one_cur
                    self.func_small_n_splito(args)

        return tik_instance

    def func_small_ntrue_splito(self, args):
        """
        function of moving data for small ntrue splito
        """
        tik_instance, ub_ori, ub_trans, one_before, one_cur = args

        n_align = self.no_f * self.ni_f
        one_d = self.c_1 * self.h_i * self.w_i

        data_offset = one_before * n_align * self.c_0
        ub_offset = 0
        ori_nburst = one_cur
        burst_len = self.n_true * self.c_0 // self.cp_align_len
        src_stride = (n_align - self.n_true) * self.c_0 // self.cp_align_len
        dst_stride = 0
        args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset, \
               ori_nburst, burst_len, src_stride, dst_stride, \
               self.cp_align_len
        _gm_to_ub_one(args)

        args = tik_instance, ub_ori, ub_trans, \
               one_cur, self.n_true, self.n_true
        if self.dtype == "float16":
            self.ub_to_ub_new(args)
        else:
            self.ub_to_ub(args)

        dst_offset = one_before * self.c_0
        res_offset = 0
        ori_nburst = self.n_true
        burst_len = one_cur * self.c_0 // self.cp_align_len
        src_stride = 0
        dst_stride = (one_d - one_cur) * self.c_0 // self.cp_align_len
        args = tik_instance, self.dst_gm, ub_trans, dst_offset, res_offset,\
               ori_nburst, burst_len, src_stride, dst_stride,\
               self.cp_align_len
        _ub_to_gm_one(args)

    def small_ntrue_splito(self, tik_instance):
        """
        one_d > n_true
        n_true * c_0 <= ub_ele
        """
        c0n_true = self.n_true * self.c_0
        one_ub = self.ub_ele // c0n_true
        one_d = self.c_1 * self.h_i * self.w_i

        all_core = _ceil_div(one_d, one_ub)
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

            ub_loop = _set_loop(tik_instance, num_core, ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                one_before = core_index * one_ub

                with tik_instance.if_scope(core_index < all_core - 1):
                    one_cur = one_ub
                    args = tik_instance, ub_ori, ub_trans, one_before, one_cur
                    self.func_small_ntrue_splito(args)
                with tik_instance.else_scope():
                    one_cur = one_d - ((all_core - 1) * one_ub)
                    args = tik_instance, ub_ori, ub_trans, one_before, one_cur
                    self.func_small_ntrue_splito(args)

        return tik_instance

    def func_big_split_one(self, args):
        """
        function of moving data for big split one
        """
        tik_instance, ub_ori, ub_trans, n_index, one_before, one_cur = args

        n_align = self.no_f * self.ni_f
        one_d = self.c_1 * self.h_i * self.w_i

        data_offset = one_before * n_align * self.c_0 + n_index * self.c_0
        ub_offset = 0
        ori_nburst = one_cur
        burst_len = self.c_0 // self.cp_align_len
        src_stride = (n_align - 1) * self.c_0 // self.cp_align_len
        dst_stride = 0
        args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset, \
               ori_nburst, burst_len, src_stride, dst_stride, \
               self.cp_align_len
        _gm_to_ub_one(args)

        args = tik_instance, ub_ori, ub_trans, \
               one_cur, 1, 1
        if self.dtype == "float16":
            self.ub_to_ub_new(args)
        else:
            self.ub_to_ub(args)

        dst_offset = n_index * one_d * self.c_0 + one_before * self.c_0
        burst_len = one_cur * self.c_0 // self.cp_align_len
        tik_instance.data_move(self.dst_gm[dst_offset],
                               ub_trans,
                               0, 1, burst_len, 0, 0)

    def big_split_one(self, tik_instance):
        """
        function of big split one
        """
        one_d = self.c_1 * self.h_i * self.w_i
        one_ub = self.ub_ele // self.c_0
        fen_n = _ceil_div(one_d, one_ub)
        all_core = self.n_true * fen_n

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

            ub_loop = _set_loop(tik_instance, num_core, ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core

                n_index = core_index // fen_n
                fen_index = core_index % fen_n

                one_before = fen_index * one_ub
                with tik_instance.if_scope(fen_index < (fen_n - 1)):
                    one_cur = one_ub
                    args = tik_instance, ub_ori, ub_trans, n_index,\
                           one_before, one_cur
                    self.func_big_split_one(args)
                with tik_instance.else_scope():
                    one_cur = one_d - ((fen_n - 1) * one_ub)
                    args = tik_instance, ub_ori, ub_trans, n_index, \
                           one_before, one_cur
                    self.func_big_split_one(args)

        return tik_instance

    def check_branch(self):
        """
        check which branch of fnz_3_dhwcn compute
        """
        big_src_ele = self.c_1 * self.h_i * self.w_i\
                      * self.no_f * self.ni_f * self.c_0
        one_d = self.c_1 * self.h_i * self.w_i

        if big_src_ele <= self.ub_ele:
            return "small_ub"
        elif one_d <= self.n_true and one_d * self.c_0 <= self.ub_ele:
            return "small_one_splitn"
        elif one_d > self.n_true\
                and self.no_f * self.ni_f * self.c_0 <= self.ub_ele:
            return "small_n_splito"
        elif one_d > self.n_true \
                and self.n_true * self.c_0 <= self.ub_ele:
            return "small_ntrue_splito"
        else:
            return "big_split_one"

    def fnz_2_nc1hwc0_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        branch = self.check_branch()

        if branch == "small_ub":
            tik_instance = self.small_ub(tik_instance)
        elif branch == "small_one_splitn":
            tik_instance = self.small_one_splitn(tik_instance)
        elif branch == "small_n_splito":
            tik_instance = self.small_n_splito(tik_instance)
        elif branch == "small_ntrue_splito":
            tik_instance = self.small_ntrue_splito(tik_instance)
        elif branch == "big_split_one":
            tik_instance = self.big_split_one(tik_instance)

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
        tik_instance = self.fnz_2_nc1hwc0_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.src_gm],
                              outputs=[self.dst_gm])

        return tik_instance


def _check_parameters(src, dst, src_format, dst_format):
    """
    check the parameters including src_shape, dst_shape,
    src_format, dst_format, dtype and kernel_name
    [C1HW, No, Ni, C0]  [N, C1, H, W, C0]

    """
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype")
    dtype_dst = dst.get("dtype")

    if src_format.lower() != "fractal_nz":
        raise RuntimeError("src_format must be FRACTAL_NZ !")

    if dst_format.lower() != "nc1hwc0":
        raise RuntimeError("dst_format must be NC1HWC0!")

    check_list = ("float16", "int8")
    check_dtype(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    check_shape(src_shape, min_rank=4, max_rank=4)
    check_shape(dst_shape, min_rank=5, max_rank=5)

    float_size = cce.cce_intrin.get_bit_len(dtype) // 8
    cp_align_len = cce_params.BLOCK_REDUCE_INT8 // float_size

    c1hw, no_f, ni_f, c_f = src_shape

    if c_f != cp_align_len:
        raise RuntimeError(
            "the 4th dimension of src_shape is wrong, "
            "when dtype is float16, C0 must be 16, "
            "when dtype is int8, C0 must be 32 !")

    if ni_f != 16:
        raise RuntimeError(
            "the 3th dimension of src_shape is wrong, Ni must be 16 !")

    n_i, c_1, h_i, w_i, c_0 = dst_shape

    if c_0 != cp_align_len:
        raise RuntimeError(
            "the 5th dimension of dst_shape is wrong, "
            "when dtype is float16, C0 must be 16, "
            "when dtype is int8, C0 must be 32 !")

    if no_f != (n_i + 15) // 16:
        raise RuntimeError(
            "the 2rd dimension of src_shape is wrong, "
            "No must be (N + 15)//16 !")

    if c_1 * h_i * w_i != c1hw:
        raise RuntimeError(
            "the 1st dimension of src_shape is wrong, "
            "it must be C1*H*W !")


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR, REQUIRED_ATTR_STR, KERNEL_NAME)
def fractal_nz_2_nc1hwc0(src, dst, src_format, dst_format,
                         kernel_name="fractal_nz_2_nc1hwc0"):
    """
    algorithm: fractal_nz_2_nc1hwc0
    calculating: change data format from FRACTAL_NZ to NC1HWC0

    Parameters
    ----------
    src: dict
        dict with keys(shape, dtype) of src
    dst: dict
        dict with keys(shape, dtype) of dst
    src_format: str
        data format of src, only support "FRACTAL_NZ"
    dst_format: str
        data format of dst, only support "NC1HWC0"
    kernel_name: str
        kernel name, default value is "fractal_nz_2_nc1hwc0"

    Returns
    -------
    tik_instance: tik_instance
    """
    _check_parameters(src, dst, src_format, dst_format)
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype").lower()

    template_fp16 = Fnz2Nc1hwc0Compute(src_shape, dst_shape,
                                       dtype, kernel_name)
    return template_fp16.get_tik_instance()
