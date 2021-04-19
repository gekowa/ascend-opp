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
ndc1hwc0_2_ncdhw
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


# pylint: disable=locally-disabled,too-many-lines
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


# pylint: disable=locally-disabled,too-many-locals
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


def _gm_to_ub_one_tik(args):
    """
    move data from GM to UB one scene

    """
    tik_instance, data, data_ub, data_offset, ub_offset, ori_nburst, \
    burst_len, src_stride, dst_stride, cp_align_len = args

    with tik_instance.if_scope(src_stride <= 65535):
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

    with tik_instance.else_scope():
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


def _ub_to_gm_one_tik(args):
    """
    function of moving data from ub to gm

    """
    tik_instance, dst, data_res, dst_offset, res_offset, ori_nburst, \
    burst_len, src_stride, dst_stride, cp_align_len = args

    with tik_instance.if_scope(dst_stride <= 65535):
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

    with tik_instance.else_scope():
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


def _ub_to_gm_one_tik2(args):
    """
    function of moving data from ub to gm

    """
    tik_instance, dst, data_res, dst_offset, res_offset, ori_nburst, \
    burst_len, src_stride, dst_stride, cp_align_len = args

    if dst_stride <= 65535:
        with tik_instance.if_scope(ori_nburst <= 4095):
            tik_instance.data_move(
                dst[dst_offset],
                data_res[res_offset],
                0, ori_nburst, burst_len,
                src_stride, dst_stride)

        with tik_instance.else_scope():
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

            with tik_instance.if_scope(c_mod > 0):
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
class Ndc1hwc02NcdhwCompute:
    """
    Rearranges data from NDC1HWC0 format to NCDHW format

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
                       // 256) * 256
        self.n_i = self.src_shape[0]
        self.d_i = self.src_shape[1]
        self.c_1 = self.src_shape[2]
        self.h_i = self.src_shape[3]
        self.w_i = self.src_shape[4]
        self.c_0 = self.src_shape[5]
        self.c_i = self.dst_shape[1]
        self.src_gm = None
        self.dst_gm = None

    def func_split_hw(self, args):
        """
        function of moving data for split hw scene

        """
        tik_instance, ub_ori, ub_trans, nd_index, n_index,\
        d_index, hw_before, hw_len = args

        hw_i = self.h_i * self.w_i
        c_align = self.c_1 * self.c_0

        data_offset = nd_index * self.c_1 * hw_i * self.c_0\
                      + hw_before * self.c_0
        ub_offset = 0
        ori_nburst = self.c_1
        burst_len = hw_len * self.c_0 // self.cp_align_len
        src_stride = (hw_i - hw_len) * self.c_0 // self.cp_align_len
        dst_stride = 0
        args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset,\
               ori_nburst, burst_len, src_stride, dst_stride, self.cp_align_len
        if isinstance(hw_len, int):
            _gm_to_ub_one(args)
        else:
            _gm_to_ub_one_tik(args)

        with tik_instance.for_range(0, self.c_1) as num_c1:
            ori_offset = num_c1 * hw_len * self.c_0
            trans_offset = num_c1 * self.c_0
            n_burst = hw_len
            burst_len = self.c_0 // self.cp_align_len
            src_stride = 0
            dst_stride = (self.c_1 - 1) * self.c_0 // self.cp_align_len
            tik_instance.data_move(ub_trans[trans_offset],
                                   ub_ori[ori_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        with tik_instance.for_range(0, self.c_1) as num_c2:
            ori_begin = num_c2 * self.c_0
            trans_begin = num_c2 * hw_len * self.c_0
            src_list = [ub_trans[ori_begin + c_align * i]
                        for i in range(16)]
            dst_list = [ub_ori[trans_begin + hw_len * i]
                        for i in range(16)]
            repeat = hw_len // 16
            if isinstance(hw_len, int):
                if repeat == 1:
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list, repeat, 0, 0)
                else:
                    src_rep_stride = 16 * self.c_1
                    dst_rep_stride = 1
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list,
                                           repeat,
                                           dst_rep_stride,
                                           src_rep_stride)
            else:
                with tik_instance.if_scope(repeat == 1):
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list, repeat, 0, 0)
                with tik_instance.else_scope():
                    src_rep_stride = 16 * self.c_1
                    dst_rep_stride = 1
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list,
                                           repeat,
                                           dst_rep_stride,
                                           src_rep_stride)

        if hw_i % self.cp_align_len > 0:
            with tik_instance.for_range(0, self.c_i) as num_ci:
                dst_cur = n_index * self.c_i * self.d_i * hw_i\
                          + num_ci * self.d_i * hw_i\
                          + d_index * hw_i + hw_before
                res_cur = num_ci * hw_len
                burst_len = hw_len // self.cp_align_len
                tik_instance.data_move(
                    self.dst_gm[dst_cur],
                    ub_ori[res_cur],
                    0, 1, burst_len, 0, 0)
        else:
            dst_offset = n_index * self.c_i * self.d_i * hw_i\
                         + d_index * hw_i + hw_before
            res_offset = 0
            ori_nburst = self.c_i
            burst_len = hw_len // self.cp_align_len
            src_stride = 0
            dst_stride = ((self.d_i - 1) * hw_i + (hw_i - hw_len))\
                         // self.cp_align_len
            args = tik_instance, self.dst_gm, ub_ori, dst_offset,\
                   res_offset, ori_nburst, burst_len, src_stride,\
                   dst_stride, self.cp_align_len
            if isinstance(hw_len, int):
                _ub_to_gm_one(args)
            else:
                _ub_to_gm_one_tik(args)

    def split_hw(self, tik_instance):
        """
        hw_i >= cp_align_len
        16 <= ub_ele // c_align < hw_i
        """
        c_align = self.c_1 * self.c_0
        hw_ub = (self.ub_ele // c_align // self.cp_align_len)\
                * self.cp_align_len
        hw_i = self.h_i * self.w_i
        hw_zu = _ceil_div(hw_i, hw_ub)
        all_core = self.n_i * self.d_i * hw_zu
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
                nd_index = core_index // hw_zu
                dhw_zu = self.d_i * hw_zu
                n_index = core_index // dhw_zu
                dhw_index = core_index % dhw_zu
                d_index = dhw_index // hw_zu
                hw_index = dhw_index % hw_zu

                with tik_instance.if_scope(hw_index < hw_zu - 1):
                    hw_len = hw_ub
                    hw_before = hw_index * hw_ub
                    args = tik_instance, ub_ori, ub_trans, nd_index,\
                           n_index, d_index, hw_before, hw_len
                    self.func_split_hw(args)
                with tik_instance.else_scope():
                    hw_temp = hw_i - (hw_index * hw_ub)
                    # hw_temp = hw_i - ((hw_zu - 1) * hw_ub)
                    hw_len = _ceil_fill(hw_temp, 16)
                    hw_before = hw_i - hw_len
                    args = tik_instance, ub_ori, ub_trans, nd_index,\
                           n_index, d_index, hw_before, hw_len
                    self.func_split_hw(args)

        return tik_instance

    def func_hw_align_chw_core(self, args):
        """
        function of moving data for hw align chw core scene

        """
        tik_instance, ub_ori, ub_trans, ub_tail, nd_index, \
        n_index, d_index = args

        hw_i = self.h_i * self.w_i
        c_align = self.c_1 * self.c_0
        src_offset = nd_index * self.c_1 * hw_i * self.c_0
        burst_len = self.c_1 * hw_i * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        with tik_instance.for_range(0, self.c_1) as num_c1:
            ori_offset = num_c1 * hw_i * self.c_0
            trans_offset = num_c1 * self.c_0
            n_burst = hw_i
            burst_len = self.c_0 // self.cp_align_len
            src_stride = 0
            dst_stride = (self.c_1 - 1) * self.c_0 // self.cp_align_len
            tik_instance.data_move(ub_trans[trans_offset],
                                   ub_ori[ori_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        hw_align = _ceil_fill(hw_i, self.cp_align_len)
        with tik_instance.for_range(0, self.c_1) as num_c2:
            ori_begin = num_c2 * self.c_0
            trans_begin = num_c2 * hw_align * self.c_0
            src_list = [ub_trans[ori_begin + c_align * i]
                        for i in range(16)]
            dst_list = [ub_ori[trans_begin + hw_align * i]
                        for i in range(16)]
            repeat = hw_align // 16
            if repeat == 1:
                tik_instance.vnchwconv(False, False, dst_list,
                                       src_list, repeat, 0, 0)
            else:
                src_rep_stride = 16 * self.c_1
                dst_rep_stride = 1
                tik_instance.vnchwconv(False, False, dst_list,
                                       src_list,
                                       repeat,
                                       dst_rep_stride,
                                       src_rep_stride)

        if hw_i % self.cp_align_len > 0:
            with tik_instance.for_range(0, self.c_i) as num_ci:
                dst_cur = n_index * self.c_i * self.d_i * hw_i\
                          + num_ci * self.d_i * hw_i\
                          + d_index * hw_i
                res_cur = num_ci * hw_align
                sub_ele = hw_i - self.cp_align_len
                if sub_ele > 0:
                    burst_len = _ceil_div(sub_ele, self.cp_align_len)
                    tik_instance.data_move(
                        self.dst_gm[dst_cur],
                        ub_ori[res_cur],
                        0, 1, burst_len, 0, 0)
                for k in range(self.cp_align_len):
                    ub_tail[k] = ub_ori[res_cur + sub_ele + k]
                tik_instance.data_move(
                    self.dst_gm[dst_cur + sub_ele],
                    ub_tail,
                    0, 1, 1, 0, 0)
        else:
            dst_offset = n_index * self.c_i * self.d_i * hw_i\
                         + d_index * hw_i
            res_offset = 0
            ori_nburst = self.c_i
            burst_len = hw_i // self.cp_align_len
            src_stride = 0
            dst_stride = ((self.d_i - 1) * hw_i) // self.cp_align_len
            args = tik_instance, self.dst_gm, ub_ori, dst_offset,\
                   res_offset, ori_nburst, burst_len, src_stride,\
                   dst_stride, self.cp_align_len
            _ub_to_gm_one(args)

    def hw_align_chw_core(self, tik_instance):
        """
        hw_i >= cp_align_len
        c_align * hw_i <= ub_ele
        """
        all_core = self.n_i * self.d_i
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
                nd_index = core_index
                n_index = core_index // self.d_i
                d_index = core_index % self.d_i

                args = tik_instance, ub_ori, ub_trans, ub_tail, nd_index, \
                       n_index, d_index
                self.func_hw_align_chw_core(args)

        return tik_instance

    def func_split_c(self, args):
        """
        function of moving data for split c scene

        """
        tik_instance, ub_ori, ub_trans, ub_tail, nd_index, n_index,\
        d_index, c1_before, c1_len, c_before, c_len = args

        hw_i = self.h_i * self.w_i
        hw_align = _ceil_fill(hw_i, self.cp_align_len)

        data_offset = nd_index * self.c_1 * hw_i * self.c_0 \
                      + c1_before * hw_i * self.c_0
        burst_len = c1_len * hw_i * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[data_offset],
                               0, 1, burst_len, 0, 0)

        with tik_instance.for_range(0, c1_len) as num_c1:
            ori_offset = num_c1 * hw_i * self.c_0
            trans_offset = num_c1 * self.c_0
            n_burst = hw_i
            burst_len = self.c_0 // self.cp_align_len
            src_stride = 0
            dst_stride = (c1_len - 1) * self.c_0 // self.cp_align_len
            tik_instance.data_move(ub_trans[trans_offset],
                                   ub_ori[ori_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        clen_align = c1_len * self.c_0
        with tik_instance.for_range(0, c1_len) as num_c2:
            ori_begin = num_c2 * self.c_0
            trans_begin = num_c2 * hw_align * self.c_0
            src_list = [ub_trans[ori_begin + clen_align * i]
                        for i in range(16)]
            dst_list = [ub_ori[trans_begin + hw_align * i]
                        for i in range(16)]
            repeat = hw_align // 16

            if repeat == 1:
                tik_instance.vnchwconv(False, False, dst_list,
                                       src_list, repeat, 0, 0)
            else:
                src_rep_stride = 16 * c1_len
                dst_rep_stride = 1
                tik_instance.vnchwconv(False, False, dst_list,
                                       src_list,
                                       repeat,
                                       dst_rep_stride,
                                       src_rep_stride)

        if hw_i % self.cp_align_len > 0:
            with tik_instance.for_range(0, c_len) as num_ci:
                dst_cur = n_index * self.c_i * self.d_i * hw_i\
                          + (c_before + num_ci) * self.d_i * hw_i\
                          + d_index * hw_i
                res_cur = num_ci * hw_align
                sub_ele = hw_i - self.cp_align_len
                if sub_ele > 0:
                    burst_len = _ceil_div(sub_ele, self.cp_align_len)
                    tik_instance.data_move(
                        self.dst_gm[dst_cur],
                        ub_ori[res_cur],
                        0, 1, burst_len, 0, 0)
                for k in range(self.cp_align_len):
                    ub_tail[k] = ub_ori[res_cur + sub_ele + k]
                tik_instance.data_move(
                    self.dst_gm[dst_cur + sub_ele],
                    ub_tail,
                    0, 1, 1, 0, 0)
        else:
            dst_offset = n_index * self.c_i * self.d_i * hw_i\
                         + c_before * self.d_i * hw_i\
                         + d_index * hw_i
            res_offset = 0
            ori_nburst = c_len
            burst_len = hw_i // self.cp_align_len
            src_stride = 0
            dst_stride = ((self.d_i - 1) * hw_i) // self.cp_align_len
            args = tik_instance, self.dst_gm, ub_ori, dst_offset, \
                   res_offset, ori_nburst, burst_len, src_stride, \
                   dst_stride, self.cp_align_len
            if isinstance(c_len, int):
                _ub_to_gm_one(args)
            else:
                _ub_to_gm_one_tik2(args)

    def split_c(self, tik_instance):
        """
        hw_i >= cp_align_len
        c_align > ub_ele
        hw_i * c_0 <= ub_ele
        """
        c_d = self.dst_shape[1]
        hw_i = self.h_i * self.w_i
        hwc0 = hw_i * self.c_0
        c1_ub = self.ub_ele // hwc0
        c1_zu = _ceil_div(self.c_1, c1_ub)
        all_core = self.n_i * self.d_i * c1_zu
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
                nd_index = core_index // c1_zu
                dc1_zu = self.d_i * c1_zu
                n_index = core_index // dc1_zu
                dc1_index = core_index % dc1_zu
                d_index = dc1_index // c1_zu
                c1_index = dc1_index % c1_zu

                with tik_instance.if_scope(c1_index < c1_zu - 1):
                    c1_before = c1_index * c1_ub
                    c1_len = c1_ub
                    c_before = c1_before * self.c_0
                    c_len = c1_len * self.c_0
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           nd_index, n_index, d_index, c1_before, c1_len,\
                           c_before, c_len
                    self.func_split_c(args)
                with tik_instance.else_scope():
                    c1_before = c1_index * c1_ub
                    c1_len = self.c_1 - (c1_index * c1_ub)
                    c_before = c1_before * self.c_0
                    c_len = c_d - c_before
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           nd_index, n_index, d_index, c1_before, c1_len, \
                           c_before, c_len
                    self.func_split_c(args)

        return tik_instance

    def func_split_big(self, args):
        """
        function of moving data for split big scene

        """
        tik_instance, ub_ori, ub_trans, nd_index, n_index, d_index, c1_index,\
        hw_before, hw_len, c_before, c_len = args

        hw_i = self.h_i * self.w_i
        data_offset = nd_index * self.c_1 * hw_i * self.c_0\
                      + c1_index * hw_i * self.c_0\
                      + hw_before * self.c_0
        burst_len = hw_len * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[data_offset],
                               0, 1, burst_len, 0, 0)

        hw_align = _ceil_fill(hw_len, self.cp_align_len)
        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + self.c_0 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + hw_align * i]
                    for i in range(16)]
        repeat = hw_align // 16
        if isinstance(hw_len, int):
            if repeat == 1:
                tik_instance.vnchwconv(False, False, dst_list,
                                       src_list, repeat, 0, 0)
            else:
                src_rep_stride = 16
                dst_rep_stride = 1
                tik_instance.vnchwconv(False, False, dst_list,
                                       src_list,
                                       repeat,
                                       dst_rep_stride,
                                       src_rep_stride)
        else:
            with tik_instance.if_scope(repeat == 1):
                tik_instance.vnchwconv(False, False, dst_list,
                                       src_list, repeat, 0, 0)
            with tik_instance.else_scope():
                src_rep_stride = 16
                dst_rep_stride = 1
                tik_instance.vnchwconv(False, False, dst_list,
                                       src_list,
                                       repeat,
                                       dst_rep_stride,
                                       src_rep_stride)

        if hw_i % self.cp_align_len > 0:
            with tik_instance.for_range(0, c_len) as num_ci:
                dst_cur = n_index * self.c_i * self.d_i * hw_i\
                          + (c_before + num_ci) * self.d_i * hw_i\
                          + d_index * hw_i + hw_before
                res_cur = num_ci * hw_len
                burst_len = hw_len // self.cp_align_len
                tik_instance.data_move(
                    self.dst_gm[dst_cur],
                    ub_trans[res_cur],
                    0, 1, burst_len, 0, 0)
        else:
            dst_offset = n_index * self.c_i * self.d_i * hw_i\
                          + c_before * self.d_i * hw_i\
                          + d_index * hw_i + hw_before
            res_offset = 0
            ori_nburst = c_len
            burst_len = hw_len // self.cp_align_len
            src_stride = 0
            dst_stride = (self.d_i * hw_i - hw_len) // self.cp_align_len
            args = tik_instance, self.dst_gm, ub_trans, dst_offset,\
                   res_offset, ori_nburst, burst_len, src_stride,\
                   dst_stride, self.cp_align_len
            if isinstance(hw_len, int):
                _ub_to_gm_one(args)
            else:
                _ub_to_gm_one_tik(args)

    def split_big(self, tik_instance):
        """
        hw_i >= cp_align_len
        hw_i * c_i > ub_ele
        """
        hw_i = self.h_i * self.w_i
        hw_ub = (self.ub_ele // self.c_0 // self.cp_align_len)\
                * self.cp_align_len
        hw_zu = _ceil_div(hw_i, hw_ub)
        all_core = self.n_i * self.d_i * self.c_1 * hw_zu
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
                dc1hw_zu = self.d_i * self.c_1 * hw_zu
                c1hw_zu = self.c_1 * hw_zu
                n_index = core_index // dc1hw_zu
                dc1hw_index = core_index % dc1hw_zu
                d_index = dc1hw_index // c1hw_zu
                nd_index = core_index // c1hw_zu
                c1hw_index = core_index % c1hw_zu
                c1_index = c1hw_index // hw_zu
                hw_index = c1hw_index % hw_zu

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c_before = c1_index * self.c_0
                    c_len = self.c_0
                    with tik_instance.if_scope(hw_index < hw_zu - 1):
                        hw_len = hw_ub
                        hw_before = hw_index * hw_ub
                        args = tik_instance, ub_ori, ub_trans, nd_index,\
                               n_index, d_index, c1_index, hw_before, hw_len,\
                               c_before, c_len
                        self.func_split_big(args)

                    with tik_instance.else_scope():
                        hw_temp = hw_i - ((hw_zu - 1) * hw_ub)
                        hw_len = _ceil_fill(hw_temp, self.cp_align_len)
                        hw_before = hw_i - hw_len
                        args = tik_instance, ub_ori, ub_trans, nd_index,\
                               n_index, d_index, c1_index, hw_before, hw_len,\
                               c_before, c_len
                        self.func_split_big(args)

                with tik_instance.else_scope():
                    c_before = (self.c_1 - 1) * self.c_0
                    c_len = self.c_i - c_before
                    with tik_instance.if_scope(hw_index < hw_zu - 1):
                        hw_len = hw_ub
                        hw_before = hw_index * hw_ub
                        args = tik_instance, ub_ori, ub_trans, nd_index,\
                               n_index, d_index, c1_index, hw_before, hw_len,\
                               c_before, c_len
                        self.func_split_big(args)

                    with tik_instance.else_scope():
                        hw_temp = hw_i - ((hw_zu - 1) * hw_ub)
                        hw_len = _ceil_fill(hw_temp, self.cp_align_len)
                        hw_before = hw_i - hw_len
                        args = tik_instance, ub_ori, ub_trans, nd_index,\
                               n_index, d_index, c1_index, hw_before, hw_len,\
                               c_before, c_len
                        self.func_split_big(args)

        return tik_instance

    def func_hwc0_core(self, args):
        """
        function of moving data for hwc0 core scene

        """
        tik_instance, ub_ori, ub_trans, ub_tail, nd_index, n_index, \
        d_index, c1_index, c_before, c_len = args

        hw_i = self.h_i * self.w_i
        hw_align = _ceil_fill(hw_i, self.cp_align_len)

        data_offset = nd_index * self.c_1 * hw_i * self.c_0 \
                      + c1_index * hw_i * self.c_0
        burst_len = hw_i * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[data_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + self.c_0 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + hw_align * i]
                    for i in range(16)]
        repeat = hw_align // 16
        if repeat == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, repeat, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   repeat,
                                   dst_rep_stride,
                                   src_rep_stride)

        if hw_i % self.cp_align_len > 0:
            with tik_instance.for_range(0, c_len) as num_ci:
                dst_cur = n_index * self.c_i * self.d_i * hw_i \
                          + (c_before + num_ci) * self.d_i * hw_i \
                          + d_index * hw_i
                res_cur = num_ci * hw_align
                sub_ele = hw_i - self.cp_align_len
                if sub_ele > 0:
                    burst_len = _ceil_div(sub_ele, self.cp_align_len)
                    tik_instance.data_move(
                        self.dst_gm[dst_cur],
                        ub_trans[res_cur],
                        0, 1, burst_len, 0, 0)
                for k in range(self.cp_align_len):
                    ub_tail[k] = ub_trans[res_cur + sub_ele + k]
                tik_instance.data_move(
                    self.dst_gm[dst_cur + sub_ele],
                    ub_tail,
                    0, 1, 1, 0, 0)
        else:
            dst_offset = n_index * self.c_i * self.d_i * hw_i \
                         + c_before * self.d_i * hw_i \
                         + d_index * hw_i
            res_offset = 0
            ori_nburst = c_len
            burst_len = hw_i // self.cp_align_len
            src_stride = 0
            dst_stride = ((self.d_i - 1) * hw_i) // self.cp_align_len
            args = tik_instance, self.dst_gm, ub_trans, dst_offset, \
                   res_offset, ori_nburst, burst_len, src_stride, \
                   dst_stride, self.cp_align_len
            if isinstance(c_len, int):
                _ub_to_gm_one(args)
            else:
                _ub_to_gm_one_tik2(args)

    def hwc0_core(self, tik_instance):
        """
        hw_i >= cp_align_len
        hw_i * c_0 <= ub_ele
        """
        c_d = self.dst_shape[1]
        all_core = self.n_i * self.d_i * self.c_1
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
                nd_index = core_index // self.c_1
                c1_index = core_index % self.c_1
                dc1_zu = self.d_i * self.c_1
                n_index = core_index // dc1_zu
                dc1_index = core_index % dc1_zu
                d_index = dc1_index // self.c_1

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c_before = c1_index * self.c_0
                    c_len = self.c_0
                    args = tik_instance, ub_ori, ub_trans, ub_tail, nd_index,\
                           n_index, d_index, c1_index, c_before, c_len
                    self.func_hwc0_core(args)
                with tik_instance.else_scope():
                    c_before = c1_index * self.c_0
                    c_len = c_d - c_before
                    args = tik_instance, ub_ori, ub_trans, ub_tail, nd_index,\
                           n_index, d_index, c1_index, c_before, c_len
                    self.func_hwc0_core(args)

        return tik_instance

    def check_branch(self):
        """
        check which branch of ndc1hwc0_2_ncdhw compute
        """
        hw_i = self.h_i * self.w_i
        c_align = self.c_1 * self.c_0

        if hw_i >= self.cp_align_len:
            if c_align * hw_i <= self.ub_ele:
                return "hw_align_chw_core"
            elif c_align < self.ub_ele and self.ub_ele // c_align >= 16 \
                    and self.ub_ele // c_align < hw_i:
                return "split_hw"
            elif c_align > self.ub_ele and hw_i * self.c_0 <= self.ub_ele:
                return "split_c"
            elif hw_i * self.c_0 > self.ub_ele:
                return "split_big"
            elif hw_i * self.c_0 <= self.ub_ele:
                return "hwc0_core"
        else:
            return "not support"

    def ndc1hwc0_2_ncdhw_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        branch = self.check_branch()

        if branch == "split_hw":
            tik_instance = self.split_hw(tik_instance)
        elif branch == "hw_align_chw_core":
            tik_instance = self.hw_align_chw_core(tik_instance)
        elif branch == "split_c":
            tik_instance = self.split_c(tik_instance)
        elif branch == "split_big":
            tik_instance = self.split_big(tik_instance)
        elif branch == "hwc0_core":
            tik_instance = self.hwc0_core(tik_instance)

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
        tik_instance = self.ndc1hwc0_2_ncdhw_compute()
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

    if src_format.lower() != "ndc1hwc0":
        raise RuntimeError("src_format must be NDC1HWC0 !")

    if dst_format.lower() != "ncdhw":
        raise RuntimeError("dst_format must be NCDHW!")

    check_list = ("float16",)
    check_dtype(dtype, check_list)
    if dtype != dtype_dst:
        raise RuntimeError("dtype of src and dst are different !")

    check_shape(src_shape, min_rank=6, max_rank=6)
    check_shape(dst_shape, min_rank=5, max_rank=5)

    if src_shape[5] != 16:
        raise RuntimeError(
            "the last dimension of src_shape is not 16, c0 must be 16 !")

    if src_shape[0] != dst_shape[0] or src_shape[1] != dst_shape[2]\
            or src_shape[3] != dst_shape[3] or src_shape[4] != dst_shape[4]:
        raise RuntimeError("the shape of src and dst not match, "
                           "the 1st,2nd,4th,5th dimension of src_shape and "
                           "the 1st,3rd,4th,5th dimension of dst_shape "
                           "must be the same !")
    c_dst = dst_shape[1]

    c_1 = src_shape[2]
    c_0 = src_shape[5]
    if not ((c_dst <= c_1*c_0) and (c_dst > (c_1 - 1)*c_0)):
        raise RuntimeError("c must be less than or equal to c1*c0,"
                           "and greater than ((c1 - 1)*c0 )!")


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR, REQUIRED_ATTR_STR, KERNEL_NAME)
def ndc1hwc0_2_ncdhw(src, dst, src_format, dst_format,
                     kernel_name='ndc1hwc0_2_ncdhw'):
    """
    algorithm: ndc1hwc0_2_ncdhw
    calculating: change data format from NDC1HWC0 to NCDHW

    Parameters
    ----------
    src: dict
        contains shape and dtype information of input tensor
    dst: dict
        contains shape and dtype information of output tensor
    src_format: str
        represents the format of input tensor, only support "NDC1HWC0"
    dst_format: str
        represents the format of output tensor, only support "NCDHW"
    kernel_name: str
        cce kernel name, default value is "ndc1hwc0_2_ncdhw"

    Returns
    -------
    None
    """
    _check_parameters(src, dst, src_format, dst_format)

    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype").lower()

    template_fp16 = Ndc1hwc02NcdhwCompute(src_shape, dst_shape,
                                          dtype, kernel_name)
    if template_fp16.check_branch() != "not support":
        return template_fp16.get_tik_instance()
    else:
        raise RuntimeError("not support this kind of transfer")
