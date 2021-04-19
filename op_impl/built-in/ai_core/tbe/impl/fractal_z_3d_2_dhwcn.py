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
fractal_z_3d_2_dhwcn
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
# pylint: disable=locally-disabled,old-style-class,too-many-return-statements
# pylint: disable=locally-disabled,too-many-statements, too-many-branches
# pylint: disable=locally-disabled,too-many-public-methods
class Fz3d2DhwcnCompute:
    """
    Rearranges data from FRACTAL_Z_3D format to DHWCN format

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
        self.c_0 = self.src_shape[3]
        self.n_i = self.src_shape[2]
        self.c_1 = self.calc_c1()
        self.src_gm = None
        self.dst_gm = None

    def check_branch(self):
        """
        check which branch of fz3d_2_dhwcn compute
        """
        d_d, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        big_dim_ele = h_d * w_d * c_d * n_d
        mid_dim_ele = c_d * n_d
        small_dim_ele = self.c_0 * n_d

        if n_d == n_o * self.n_i:
            big_in_ele = h_d * w_d * self.c_1 * self.c_0 * n_o * self.n_i
            mid_in_ele = self.c_1 * self.c_0 * n_o * self.n_i

            if big_in_ele <= self.ub_ele and d_d >= AICORE_NUM:
                num_dim_ub = self.ub_ele // big_in_ele
                num_dim_group = num_dim_ub * AICORE_NUM
                if num_dim_group <= d_d and num_dim_ub > 1:
                    return "n_align_multi_small"
                else:
                    return "n_align_small"
            elif mid_in_ele <= self.ub_ele:
                num_dim_ub = self.ub_ele // mid_in_ele
                num_dim_group = num_dim_ub * AICORE_NUM
                dhw_d = d_d * h_d * w_d
                if num_dim_group <= dhw_d and num_dim_ub > 1:
                    return "n_align_multi_mid"
                else:
                    return "n_align_mid"
            elif small_dim_ele <= self.ub_ele:
                return "n_align_big"
            else:
                return "n_align_splitn"

        else:
            big_src_ele = self.c_1 * h_d * w_d * n_o * self.n_i * self.c_0
            big_src_ele_plus = h_d * w_d * self.c_1 * n_d\
                            * self.c_0 * self.cp_align_len

            mid_src_ele = self.c_1 * self.c_0 * n_o * self.n_i
            mid_src_ele_plus = self.c_1 * self.c_0 * n_d * self.cp_align_len

            small_src_ele = self.c_0 * n_o * self.n_i
            small_src_ele_plus = self.c_0 * n_d * self.cp_align_len

            little_ele_plus = h_d * w_d * n_o * self.n_i\
                              * self.c_1 * self.c_0 * 16
            little_mid_ele_plus = self.n_i * self.c_0 * 16

            if big_dim_ele < self.cp_align_len\
                    and little_ele_plus <= self.ub_ele:

                num_big_unit_ub = self.ub_ele // little_ele_plus
                ele_one_core = num_big_unit_ub * big_dim_ele
                if ele_one_core >= self.cp_align_len:
                    return "not_align_little"
                else:
                    true_ele_plus = h_d * w_d * n_d * self.c_0 * 16
                    if true_ele_plus <= self.ub_ele:
                        num_true_ub = self.ub_ele // true_ele_plus
                        ele_true_core = num_true_ub * big_dim_ele
                        if ele_true_core >= self.cp_align_len:
                            return "not_align_mm"

            elif big_src_ele <= self.ub_ele\
                    and big_src_ele_plus <= self.ub_ele\
                    and big_dim_ele >= self.cp_align_len:

                big_unit_ele = h_d * w_d * n_o * self.n_i\
                               * self.c_1 * self.c_0 * 16
                num_big_unit_ub = self.ub_ele // big_unit_ele
                num_big_unit_group = num_big_unit_ub * AICORE_NUM
                if num_big_unit_group <= d_d and num_big_unit_ub > 1\
                        and num_big_unit_ub * big_dim_ele >= self.cp_align_len\
                        and little_ele_plus <= self.ub_ele:
                    return "not_align_little"
                else:
                    return "not_align_small"

            elif mid_dim_ele < self.cp_align_len\
                    and little_mid_ele_plus <= self.ub_ele:
                num_mid_ub = self.ub_ele // little_mid_ele_plus
                ele_core = num_mid_ub * mid_dim_ele
                if ele_core >= self.cp_align_len:
                    return "not_align_little_mid"
                else:
                    true_mid_plus = n_d * self.c_0 * 16
                    if true_mid_plus <= self.ub_ele:
                        num_truem_ub = self.ub_ele // true_mid_plus
                        ele_truem_ele = num_truem_ub * mid_dim_ele
                        if ele_truem_ele >= self.cp_align_len:
                            return "not_align_mm_mid"

            elif mid_src_ele <= self.ub_ele\
                    and mid_src_ele_plus <= self.ub_ele\
                    and mid_dim_ele >= self.cp_align_len:
                return "not_align_mid"
            elif small_src_ele <= self.ub_ele\
                    and small_src_ele_plus <= self.ub_ele\
                    and mid_dim_ele >= self.cp_align_len:
                return "not_align_big"

            dhwc1_d = d_d * h_d * w_d * self.c_1

            if dhwc1_d < AICORE_NUM:
                return "not_align_splitn_fencore"
            else:
                return "not_align_splitn"

    def check_branch_fp32(self):
        """
        check which branch of fz3d_2_dhwcn fp32 compute
        """
        c_d = self.dst_shape[3]
        n_d = self.dst_shape[4]
        c0n_ele = n_d * self.c_0 * 2 * 8

        if n_d * c_d < self.cp_align_len:
            n_o = self.src_shape[1]
            n_align = n_o * self.n_i
            dim_align_ele = n_align * self.c_0 * 16
            num_dim_one_core = self.ub_ele // dim_align_ele
            if num_dim_one_core * c_d * n_d < self.cp_align_len:
                return "little_mm_fp32"
            else:
                return "little_align_fp32"

        elif c0n_ele <= self.ub_ele:
            return "c0n_ele_fp32"
        else:
            return "split_n_fp32"

    def calc_c1(self):
        """
        function of calculating c_1
        """
        dc1hw_s = self.src_shape[0]
        d_d, h_d, w_d, _, _ = self.dst_shape
        c_1 = dc1hw_s // (d_d * h_d * w_d)
        return c_1

    def n_align_small(self, tik_instance):
        """
        n_d == n_o * n_i
        c_d < c_1 * c_0
        h_d * w_d * c_1 * c_0 * n_o * n_i <= ub_ele
        """
        d_d, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        ac_num = _set_core_num(d_d)
        hw_d = h_d * w_d

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (self.ub_ele,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (self.ub_ele,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, d_d)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                big_in_ele = h_d * w_d * self.c_1 * self.c_0 * n_o * self.n_i
                big_dim_ele = h_d * w_d * c_d * n_d
                small_dim_ele = n_d * self.c_0

                src_offset = core_index * big_in_ele
                burst_len = big_in_ele // self.cp_align_len
                tik_instance.data_move(ub_ori,
                                       self.src_gm[src_offset],
                                       0, 1, burst_len, 0, 0)
                with tik_instance.for_range(0, hw_d) as num_hw:
                    with tik_instance.for_range(0, self.c_1) as num_c:
                        ori_begin = num_c * hw_d * small_dim_ele \
                                    + num_hw * small_dim_ele
                        trans_begin = num_hw * c_d * n_d \
                                      + num_c * small_dim_ele
                        src_list = [ub_ori[ori_begin + self.c_0 * i]
                                    for i in range(16)]
                        dst_list = [ub_trans[trans_begin + n_d * i]
                                    for i in range(16)]
                        if n_o == 1:
                            tik_instance.vnchwconv(False, False, dst_list,
                                                   src_list, n_o, 0, 0)
                        else:
                            src_rep_stride = 16
                            dst_rep_stride = 1
                            tik_instance.vnchwconv(False, False, dst_list,
                                                   src_list,
                                                   n_o,
                                                   dst_rep_stride,
                                                   src_rep_stride)

                dst_offset = core_index * big_dim_ele
                burst_len = big_dim_ele // self.cp_align_len
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, burst_len, 0, 0)

        return tik_instance

    def func_n_align_multi_small(self, args):
        """
        moving data for n_align_multi_small scene
        """
        tik_instance, ub_ori, ub_trans, num_dim_before, num_dim_cur = args

        _, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        hw_d = h_d * w_d
        big_in_ele = h_d * w_d * self.c_1 * self.c_0 * n_o * self.n_i
        big_dim_ele = h_d * w_d * c_d * n_d
        small_dim_ele = n_d * self.c_0

        src_offset = num_dim_before * big_in_ele
        burst_len = num_dim_cur * big_in_ele // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        with tik_instance.for_range(0, num_dim_cur) as num_d:
            with tik_instance.for_range(0, hw_d) as num_hw:
                with tik_instance.for_range(0, self.c_1) as num_c:
                    ori_begin = num_d * big_in_ele\
                                + num_c * hw_d * small_dim_ele \
                                + num_hw * small_dim_ele
                    trans_begin = num_d * big_dim_ele\
                                  + num_hw * c_d * n_d \
                                  + num_c * small_dim_ele
                    src_list = [ub_ori[ori_begin + self.c_0 * i]
                                for i in range(16)]
                    dst_list = [ub_trans[trans_begin + n_d * i]
                                for i in range(16)]
                    if n_o == 1:
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list, n_o, 0, 0)
                    else:
                        src_rep_stride = 16
                        dst_rep_stride = 1
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list,
                                               n_o,
                                               dst_rep_stride,
                                               src_rep_stride)

        dst_offset = num_dim_before * big_dim_ele
        burst_len = num_dim_cur * big_dim_ele // self.cp_align_len
        tik_instance.data_move(self.dst_gm[dst_offset],
                               ub_trans,
                               0, 1, burst_len, 0, 0)

    def n_align_multi_small(self, tik_instance):
        """
        n_d == n_o * n_i
        c_d < c_1 * c_0
        h_d * w_d * c_1 * c_0 * n_o * n_i <= ub_ele
        """
        d_d, h_d, w_d, _, _ = self.dst_shape
        n_o = self.src_shape[1]
        big_in_ele = h_d * w_d * self.c_1 * self.c_0 * n_o * self.n_i
        num_dim_ub = self.ub_ele // big_in_ele
        core_true = _ceil_div(d_d, num_dim_ub)
        ac_num = _set_core_num(core_true)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (self.ub_ele,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (self.ub_ele,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, core_true)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                with tik_instance.if_scope(core_index < (core_true - 1)):
                    num_dim_cur = num_dim_ub
                    num_dim_before = core_index * num_dim_ub
                    args = tik_instance, ub_ori, ub_trans,\
                           num_dim_before, num_dim_cur
                    self.func_n_align_multi_small(args)
                with tik_instance.else_scope():
                    num_dim_cur = d_d - (num_dim_ub * (core_true - 1))
                    num_dim_before = core_index * num_dim_ub
                    args = tik_instance, ub_ori, ub_trans, \
                           num_dim_before, num_dim_cur
                    self.func_n_align_multi_small(args)

        return tik_instance

    def n_align_mid(self, tik_instance):
        """
        n_d == n_o * n_i
        c_d < c_1 * c_0
        h_d * w_d * c_1 * c_0 * n_o * n_i > ub_ele
        c_1 * c_0 * n_o * n_i <= ub_ele
        """
        d_d, h_d, w_d, c_d, n_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        n_o = self.src_shape[1]
        ac_num = _set_core_num(dhw_d)
        hw_d = h_d * w_d

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (self.ub_ele,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (self.ub_ele,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, dhw_d)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                d_index = core_index // hw_d
                hw_index = core_index % hw_d

                big_in_ele = h_d * w_d * self.c_1 * self.c_0 * n_o * self.n_i
                small_dim_ele = n_d * self.c_0

                src_offset = d_index * big_in_ele + hw_index * small_dim_ele
                n_burst = self.c_1
                burst_len = small_dim_ele // self.cp_align_len
                src_stride = (hw_d - 1) * small_dim_ele // self.cp_align_len

                args = tik_instance, self.src_gm, ub_ori, src_offset, 0,\
                       n_burst, burst_len, src_stride, 0, self.cp_align_len
                _gm_to_ub_one(args)

                with tik_instance.for_range(0, self.c_1) as num_c:
                    ori_begin = num_c * small_dim_ele
                    trans_begin = num_c * small_dim_ele
                    src_list = [ub_ori[ori_begin + self.c_0 * i]
                                for i in range(16)]
                    dst_list = [ub_trans[trans_begin + n_d * i]
                                for i in range(16)]
                    if n_o == 1:
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list, n_o, 0, 0)
                    else:
                        src_rep_stride = 16
                        dst_rep_stride = 1
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list,
                                               n_o,
                                               dst_rep_stride,
                                               src_rep_stride)

                dst_offset = core_index * c_d * n_d
                burst_len = c_d * n_d // self.cp_align_len
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, burst_len, 0, 0)

        return tik_instance

    def func_n_align_multi_mid(self, args):
        """
        moving data for n_align_multi_mid scene
        """
        tik_instance, ub_ori, ub_trans, num_dim_before, num_dim_cur = args
        _, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        hw_d = h_d * w_d
        in_ele = self.c_1 * self.c_0 * n_o * self.n_i

        big_in_ele = h_d * w_d * self.c_1 * self.c_0 * n_o * self.n_i
        small_dim_ele = n_d * self.c_0

        with tik_instance.for_range(0, num_dim_cur) as num_d1:
            dim_index = num_dim_before + num_d1
            d_index = dim_index // hw_d
            hw_index = dim_index % hw_d

            src_offset = d_index * big_in_ele + hw_index * small_dim_ele
            ub_offset = num_d1 * in_ele
            n_burst = self.c_1
            burst_len = small_dim_ele // self.cp_align_len
            src_stride = (hw_d - 1) * small_dim_ele // self.cp_align_len

            args = tik_instance, self.src_gm, ub_ori, src_offset, ub_offset, \
                   n_burst, burst_len, src_stride, 0, self.cp_align_len
            _gm_to_ub_one(args)

        with tik_instance.for_range(0, num_dim_cur) as num_d2:
            with tik_instance.for_range(0, self.c_1) as num_c:
                ori_begin = num_d2 * in_ele + num_c * small_dim_ele
                trans_begin = num_d2 * c_d * n_d + num_c * small_dim_ele
                src_list = [ub_ori[ori_begin + self.c_0 * i]
                            for i in range(16)]
                dst_list = [ub_trans[trans_begin + n_d * i]
                            for i in range(16)]
                if n_o == 1:
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list, n_o, 0, 0)
                else:
                    src_rep_stride = 16
                    dst_rep_stride = 1
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list,
                                           n_o,
                                           dst_rep_stride,
                                           src_rep_stride)

        dst_offset = num_dim_before * c_d * n_d
        burst_len = num_dim_cur * c_d * n_d // self.cp_align_len
        tik_instance.data_move(self.dst_gm[dst_offset],
                               ub_trans,
                               0, 1, burst_len, 0, 0)

    def n_align_multi_mid(self, tik_instance):
        """
        n_d == n_o * n_i
        c_d < c_1 * c_0
        h_d * w_d * c_1 * c_0 * n_o * n_i > ub_ele
        c_1 * c_0 * n_o * n_i <= ub_ele
        """
        d_d, h_d, w_d, _, _ = self.dst_shape
        dhw_d = d_d * h_d * w_d
        n_o = self.src_shape[1]
        in_ele = self.c_1 * self.c_0 * n_o * self.n_i
        num_dim_ub = self.ub_ele // in_ele
        core_true = _ceil_div(dhw_d, num_dim_ub)
        ac_num = _set_core_num(core_true)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (self.ub_ele,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (self.ub_ele,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, core_true)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                with tik_instance.if_scope(core_index < (core_true - 1)):
                    num_dim_cur = num_dim_ub
                    num_dim_before = core_index * num_dim_ub
                    args = tik_instance, ub_ori, ub_trans,\
                           num_dim_before, num_dim_cur
                    self.func_n_align_multi_mid(args)
                with tik_instance.else_scope():
                    num_dim_cur = dhw_d - (num_dim_ub * (core_true - 1))
                    num_dim_before = core_index * num_dim_ub
                    args = tik_instance, ub_ori, ub_trans, \
                           num_dim_before, num_dim_cur
                    self.func_n_align_multi_mid(args)

        return tik_instance

    def func_n_align_big(self, args):
        """
        function of n align big
        """
        tik_instance, ub_ori, ub_trans, d_index, hw_index,\
        c1_index, c0_len = args

        _, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        hw_d = h_d * w_d

        big_dim_ele = h_d * w_d * c_d * n_d
        big_in_ele = h_d * w_d * self.c_1 * self.c_0 * n_o * self.n_i
        small_dim_ele = n_d * self.c_0
        src_offset = d_index * big_in_ele \
                     + c1_index * hw_d * small_dim_ele \
                     + hw_index * small_dim_ele
        burst_len = small_dim_ele // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + self.c_0 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + n_d * i]
                    for i in range(16)]
        if n_o == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, n_o, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   n_o,
                                   dst_rep_stride,
                                   src_rep_stride)

        dst_offset = d_index * big_dim_ele \
                     + hw_index * c_d * n_d \
                     + c1_index * small_dim_ele
        burst_len = c0_len * n_d // self.cp_align_len
        tik_instance.data_move(self.dst_gm[dst_offset],
                               ub_trans,
                               0, 1, burst_len, 0, 0)

    def n_align_big(self, tik_instance):
        """
        n_d == n_o * n_i
        c_d < c_1 * c_0
        c_1 * c_0 * n_o * n_i > ub_ele && c_0 * n_d <= ub_ele
        """
        d_d, h_d, w_d, c_d, _ = self.dst_shape
        dhwc1_d = d_d * h_d * w_d * self.c_1
        ac_num = _set_core_num(dhwc1_d)
        hwc1_d = h_d * w_d * self.c_1

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (self.ub_ele,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (self.ub_ele,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, dhwc1_d)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                d_index = core_index // hwc1_d
                mod_index = core_index % hwc1_d
                hw_index = mod_index // self.c_1
                c1_index = mod_index % self.c_1

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c0_len = self.c_0
                    args = tik_instance, ub_ori, ub_trans, d_index,\
                           hw_index, c1_index, c0_len
                    self.func_n_align_big(args)

                with tik_instance.else_scope():
                    c0_len = c_d - (c1_index * self.c_0)
                    args = tik_instance, ub_ori, ub_trans, d_index, \
                           hw_index, c1_index, c0_len
                    self.func_n_align_big(args)

        return tik_instance

    def move_for_n_align_splitn(self, args):
        """
        function of moving data for n align splitn
        """
        tik_instance, ub_ori, ub_trans, fen_n, core_index,\
        fen_n_index, n_ub, n_len_now = args

        _, h_d, w_d, c_d, n_d = self.dst_shape
        hw_d = h_d * w_d
        hwc1fen_d = h_d * w_d * self.c_1 * fen_n
        c1fen_d = self.c_1 * fen_n
        d_index = core_index // hwc1fen_d
        one_index = core_index % hwc1fen_d
        hw_index = one_index // c1fen_d
        two_index = one_index % c1fen_d
        c1_index = two_index // fen_n

        big_in_ele = h_d * w_d * self.c_1 * self.c_0 * n_d
        big_dim_ele = h_d * w_d * c_d * n_d
        small_dim_ele = n_d * self.c_0

        src_offset = d_index * big_in_ele + c1_index * hw_d * small_dim_ele\
                     + hw_index * small_dim_ele\
                     + fen_n_index * n_ub * self.c_0
        burst_len = n_len_now * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + self.c_0 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + n_len_now * i]
                    for i in range(16)]
        n_g = n_len_now // self.n_i
        with tik_instance.if_scope(n_g == 1):
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, n_g, 0, 0)
        with tik_instance.else_scope():
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   n_g,
                                   dst_rep_stride,
                                   src_rep_stride)

        dst_offset = d_index * big_dim_ele + hw_index * c_d * n_d\
                     + c1_index * small_dim_ele + fen_n_index * n_ub
        burst_len = n_len_now // self.cp_align_len
        dst_stride = (n_d - n_len_now) // self.cp_align_len

        with tik_instance.if_scope(c1_index < self.c_1 - 1):
            n_burst = self.c_0
            args = tik_instance, self.dst_gm, ub_trans, dst_offset, 0,\
                   n_burst, burst_len, 0, dst_stride, self.cp_align_len
            _ub_to_gm_one(args)
        with tik_instance.else_scope():
            n_burst = c_d - ((self.c_1 - 1) * self.c_0)
            args = tik_instance, self.dst_gm, ub_trans, dst_offset, 0,\
                   n_burst, burst_len, 0, dst_stride, self.cp_align_len
            _ub_to_gm_one(args)

    def func_n_align_splitn_mod(self, args):
        """
        function of n align splitn mod
        """
        tik_instance, n_d, true_core_num, fen_n, n_ub = args

        ac_num = _set_core_num(true_core_num)
        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (self.ub_ele,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (self.ub_ele,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, true_core_num)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                fen_n_index = core_index % fen_n
                with tik_instance.if_scope(fen_n_index < fen_n - 1):
                    n_len_now = n_ub
                    args = tik_instance, ub_ori, ub_trans, fen_n, core_index,\
                           fen_n_index, n_ub, n_len_now
                    self.move_for_n_align_splitn(args)
                with tik_instance.else_scope():
                    n_len_now = n_d - ((fen_n - 1) * n_ub)
                    args = tik_instance, ub_ori, ub_trans, fen_n, core_index,\
                           fen_n_index, n_ub, n_len_now
                    self.move_for_n_align_splitn(args)

    def n_align_splitn(self, tik_instance):
        """
        n_d == n_o * n_i
        c_d < c_1 * c_0
        c_0 * n_d > ub_ele && 16 * 16 <= ub_ele
        split n_d
        """
        d_d, h_d, w_d, _, n_d = self.dst_shape
        dhwc1_d = d_d * h_d * w_d * self.c_1
        true_ele = (self.ub_ele // 256) * 256
        cn_ele = n_d * self.c_0
        n_ub = true_ele // self.c_0

        fen_n = _ceil_div(cn_ele, true_ele)
        true_core_num = dhwc1_d * fen_n
        args = tik_instance, n_d, true_core_num, fen_n, n_ub
        self.func_n_align_splitn_mod(args)

        return tik_instance

    def func_not_align_mm(self, args):
        """
        function of not align mm
        """
        tik_instance, ub_ori, ub_trans, ub_tail,\
        num_true_before, num_true_cur = args

        _, h_d, w_d, c_d, n_d = self.dst_shape

        true_unit_ele = h_d * w_d * self.n_i * self.c_0
        data_offset = num_true_before * true_unit_ele
        ub_offset = 0
        ori_nburst = num_true_cur * h_d * w_d
        burst_len = n_d
        src_stride = self.n_i - n_d
        dst_stride = 0
        args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset,\
               ori_nburst, burst_len, src_stride, dst_stride,\
               self.cp_align_len
        _gm_to_ub_one(args)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + self.c_0 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        repeat_vconv = num_true_cur * h_d * w_d * n_d
        if repeat_vconv == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, repeat_vconv, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   repeat_vconv,
                                   dst_rep_stride,
                                   src_rep_stride)

        hw_d = h_d * w_d
        with tik_instance.for_range(0, num_true_cur) as num_t:
            with tik_instance.for_range(0, hw_d) as num_hw:
                with tik_instance.for_range(0, c_d) as num_c:
                    trans_offset = num_t * hw_d * n_d * self.c_0 * 16\
                                   + num_hw * n_d * self.c_0 * 16\
                                   + num_c * 16
                    ori_offset = num_t * hw_d * n_d * c_d * 16\
                                 + num_hw * n_d * c_d * 16\
                                 + num_c * n_d * 16
                    n_burst = n_d
                    burst_len = 1
                    src_stride = self.c_0 - 1
                    tik_instance.data_move(ub_ori[ori_offset],
                                           ub_trans[trans_offset],
                                           0, n_burst, burst_len,
                                           src_stride, 0)

        all_ele = num_true_cur * h_d * w_d * c_d * n_d
        all_ele_div = _ceil_div(all_ele, 16)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        if all_ele_div == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, all_ele_div, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   all_ele_div,
                                   dst_rep_stride,
                                   src_rep_stride)

        if all_ele % self.cp_align_len > 0:
            if all_ele > self.cp_align_len:
                move_len = all_ele - self.cp_align_len
                dst_offset = num_true_before * h_d * w_d * c_d * n_d
                burst_len = _ceil_div(move_len, self.cp_align_len)
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, burst_len, 0, 0)
                for k in range(self.cp_align_len):
                    ub_tail[k] = ub_trans[move_len + k]
                tik_instance.data_move(self.dst_gm[dst_offset + move_len],
                                       ub_tail,
                                       0, 1, 1, 0, 0)
            else:
                dst_offset = num_true_before * h_d * w_d * c_d * n_d
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, 1, 0, 0)
        else:
            dst_offset = num_true_before * h_d * w_d * c_d * n_d
            burst_len = all_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def not_align_mm(self, tik_instance):
        """
        n_d < n_o * n_i
        c_d < c_1 * c_0
        h_d * w_d * c_d * n_d < cp_align_len
        and ub_ele // (h_d * w_d * n_o * n_i * c_1 * c_0) <= 1
        --> c < 16  n < 16
        """
        d_d, h_d, w_d, _, n_d = self.dst_shape
        hw_d = h_d * w_d
        true_ele = hw_d * n_d * self.c_0 * 16
        num_true_ub = self.ub_ele // true_ele
        all_core_num = _ceil_div(d_d, num_true_ub)
        ac_num = _set_core_num(all_core_num)

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

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core_num)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                num_true_before = core_index * num_true_ub
                with tik_instance.if_scope(core_index < all_core_num - 1):
                    num_true_cur = num_true_ub
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           num_true_before, num_true_cur
                    self.func_not_align_mm(args)
                with tik_instance.else_scope():
                    num_true_cur = d_d - ((all_core_num - 1) * num_true_ub)
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           num_true_before, num_true_cur
                    self.func_not_align_mm(args)

        return tik_instance

    def func_not_align_little(self, args):
        """
        function of not align little
        """
        tik_instance, ub_ori, ub_trans, ub_tail, hw_d,\
        num_unit_before, num_unit_cur = args

        _, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        big_src_ele = self.c_1 * h_d * w_d * n_o * self.n_i * self.c_0

        src_offset = num_unit_before * big_src_ele
        burst_len = num_unit_cur * big_src_ele // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + self.c_0 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        repeat_vconv = num_unit_cur * n_o * self.n_i * hw_d * self.c_1
        if repeat_vconv == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, repeat_vconv, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   repeat_vconv,
                                   dst_rep_stride,
                                   src_rep_stride)

        with tik_instance.for_range(0, num_unit_cur) as num_uc:
            with tik_instance.for_range(0, hw_d) as num_hw:
                with tik_instance.for_range(0, c_d) as num_cd:
                    num_c1 = num_cd // self.c_0
                    num_c0 = num_cd % self.c_0
                    trans_offset = num_uc * hw_d * n_o * self.n_i\
                                   * self.c_1 * self.c_0 * 16\
                                   + num_c1 * hw_d * n_o * self.n_i\
                                   * self.c_0 * 16\
                                   + num_hw * n_o * self.n_i * self.c_0 * 16\
                                   + num_c0 * 16
                    ori_offset = num_uc * hw_d * c_d * n_d * 16\
                                 + num_hw * c_d * n_d * 16\
                                 + num_cd * n_d * 16
                    n_burst = n_d
                    burst_len = 1
                    src_stride = self.c_0 - 1
                    tik_instance.data_move(ub_ori[ori_offset],
                                           ub_trans[trans_offset],
                                           0, n_burst, burst_len,
                                           src_stride, 0)

        all_ele = num_unit_cur * hw_d * c_d * n_d
        all_ele_div = _ceil_div(all_ele, 16)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        if all_ele_div == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, all_ele_div, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   all_ele_div,
                                   dst_rep_stride,
                                   src_rep_stride)

        if all_ele % self.cp_align_len > 0:
            if all_ele > self.cp_align_len:
                move_len = all_ele - self.cp_align_len
                dst_offset = num_unit_before * h_d * w_d * c_d * n_d
                burst_len = _ceil_div(move_len, self.cp_align_len)
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, burst_len, 0, 0)
                for k in range(self.cp_align_len):
                    ub_tail[k] = ub_trans[move_len + k]
                tik_instance.data_move(self.dst_gm[dst_offset + move_len],
                                       ub_tail,
                                       0, 1, 1, 0, 0)
            else:
                dst_offset = num_unit_before * h_d * w_d * c_d * n_d
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, 1, 0, 0)
        else:
            dst_offset = num_unit_before * h_d * w_d * c_d * n_d
            burst_len = all_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def not_align_little(self, tik_instance):
        """
        n_d < n_o * n_i
        c_d < c_1 * c_0
        multi h_d * w_d * c_d * n_d in one core
        """
        d_d, h_d, w_d, _, _ = self.dst_shape
        hw_d = h_d * w_d
        n_o = self.src_shape[1]
        big_unit_ele = hw_d * n_o * self.n_i * self.c_1 * self.c_0 * 16
        num_big_unit_ub = self.ub_ele // big_unit_ele
        all_core_num = _ceil_div(d_d, num_big_unit_ub)
        ac_num = _set_core_num(all_core_num)

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

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core_num)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                num_unit_before = core_index * num_big_unit_ub
                with tik_instance.if_scope(core_index < all_core_num - 1):
                    num_unit_cur = num_big_unit_ub
                    args = tik_instance, ub_ori, ub_trans, ub_tail, hw_d, \
                           num_unit_before, num_unit_cur
                    self.func_not_align_little(args)

                with tik_instance.else_scope():
                    num_unit_cur = d_d - ((all_core_num - 1) * num_big_unit_ub)
                    args = tik_instance, ub_ori, ub_trans, ub_tail, hw_d, \
                           num_unit_before, num_unit_cur
                    self.func_not_align_little(args)

        return tik_instance

    def ub_to_ub_not_align_small(self, args):
        """
        function of moving data from ub to ub for not align small scene
        """
        tik_instance, ub_ori, ub_trans, c_d, n_d, c0_len, num_h, num_c1 = args

        with tik_instance.for_range(0, c0_len) as num_clen:
            ori_offset = num_h * c_d * n_d * 16 \
                         + num_c1 * self.c_0 * n_d * 16 \
                         + num_clen * n_d * 16
            trans_offset = num_h * self.c_1 * n_d \
                           * self.c_0 * 16 \
                           + num_c1 * n_d * self.c_0 * 16 \
                           + num_clen * 16
            n_burst = n_d
            burst_len = 16 // self.cp_align_len
            src_stride = self.c_0 - 1
            dst_stride = 0
            tik_instance.data_move(ub_ori[ori_offset],
                                   ub_trans[trans_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

    def not_align_small(self, tik_instance):
        """
        n_d < n_o * n_i
        c_d < c_1 * c_0
        c_1 * h_d * w_d * n_o * n_i * c_0 <= ub_ele
        h_d * w_d * c_1 * c_0 * n_d * 16 <= ub_ele
        h_d * w_d * c_d * n_d >= cp_align_len
        """
        d_d, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        ac_num = _set_core_num(d_d)
        hw_d = h_d * w_d

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

            ub_loop = _set_loop(tik_instance, num_core, ac_num, d_d)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                big_src_ele = self.c_1 * h_d * w_d * n_o * self.n_i * self.c_0
                small_src_ele = n_o * self.n_i * self.c_0
                src_offset = core_index * big_src_ele
                burst_len = big_src_ele // self.cp_align_len
                tik_instance.data_move(ub_ori,
                                       self.src_gm[src_offset],
                                       0, 1, burst_len, 0, 0)

                with tik_instance.for_range(0, hw_d) as num_hw:
                    with tik_instance.for_range(0, self.c_1) as num_c:

                        ori_begin = num_c * hw_d * small_src_ele \
                                    + num_hw * small_src_ele
                        trans_begin = num_hw * self.c_1 * n_d * self.c_0 * 16\
                                      + num_c * n_d * self.c_0 * 16
                        src_list = [ub_ori[ori_begin + self.c_0 * i]
                                    for i in range(16)]
                        dst_list = [ub_trans[trans_begin + 16 * i]
                                    for i in range(16)]
                        if n_d == 1:
                            tik_instance.vnchwconv(False, False, dst_list,
                                                   src_list, n_d, 0, 0)
                        else:
                            src_rep_stride = 1
                            dst_rep_stride = 16
                            tik_instance.vnchwconv(False, False, dst_list,
                                                   src_list,
                                                   n_d,
                                                   dst_rep_stride,
                                                   src_rep_stride)

                with tik_instance.for_range(0, hw_d) as num_h:
                    with tik_instance.for_range(0, self.c_1) as num_c1:
                        with tik_instance.if_scope(num_c1 < self.c_1 - 1):
                            c0_len = self.c_0
                            args = tik_instance, ub_ori, ub_trans, c_d, n_d,\
                                   c0_len, num_h, num_c1
                            self.ub_to_ub_not_align_small(args)

                        with tik_instance.else_scope():
                            c0_len = c_d - ((self.c_1 - 1) * self.c_0)
                            args = tik_instance, ub_ori, ub_trans, c_d, n_d, \
                                   c0_len, num_h, num_c1
                            self.ub_to_ub_not_align_small(args)

                hwcn_d = h_d * w_d * c_d * n_d
                hwcn_d_align = _ceil_fill(hwcn_d, self.cp_align_len)
                hwcn_d_div = hwcn_d_align // self.cp_align_len

                ori_begin = 0
                trans_begin = 0
                src_list = [ub_ori[ori_begin + 16 * i]
                            for i in range(16)]
                dst_list = [ub_trans[trans_begin + 16 * i]
                            for i in range(16)]
                if hwcn_d_div == 1:
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list, hwcn_d_div, 0, 0)
                else:
                    src_rep_stride = 16
                    dst_rep_stride = 1
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list,
                                           hwcn_d_div,
                                           dst_rep_stride,
                                           src_rep_stride)

                if hwcn_d % self.cp_align_len > 0:
                    move_len = hwcn_d - self.cp_align_len
                    dst_offset = core_index * hwcn_d
                    burst_len = _ceil_div(move_len, self.cp_align_len)
                    tik_instance.data_move(self.dst_gm[dst_offset],
                                           ub_trans,
                                           0, 1, burst_len, 0, 0)
                    for k in range(self.cp_align_len):
                        ub_tail[k] = ub_trans[move_len + k]

                    tik_instance.data_move(self.dst_gm[dst_offset + move_len],
                                           ub_tail,
                                           0, 1, 1, 0, 0)
                else:
                    dst_offset = core_index * hwcn_d
                    burst_len = hwcn_d // self.cp_align_len
                    tik_instance.data_move(self.dst_gm[dst_offset],
                                           ub_trans,
                                           0, 1, burst_len, 0, 0)

        return tik_instance

    def func_not_align_mm_mid(self, args):
        """
        function of not align little mid
        """
        tik_instance, ub_ori, ub_trans, ub_tail,\
        num_unit_before, num_unit_cur = args

        _, _, _, c_d, n_d = self.dst_shape
        small_ele = n_d * self.c_0
        cn_d = c_d * n_d

        data_offset = num_unit_before * self.n_i * self.c_0
        ub_offset = 0
        ori_nburst = num_unit_cur
        burst_len = n_d
        src_stride = self.n_i - n_d
        dst_stride = 0
        args = tik_instance, self.src_gm, ub_ori, data_offset, ub_offset,\
               ori_nburst, burst_len, src_stride, dst_stride,\
               self.cp_align_len
        _gm_to_ub_one(args)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + self.c_0 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        repeat_vconv = num_unit_cur * n_d
        if repeat_vconv == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, repeat_vconv, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   repeat_vconv,
                                   dst_rep_stride,
                                   src_rep_stride)

        with tik_instance.for_range(0, num_unit_cur) as num_uc:
            with tik_instance.for_range(0, c_d) as num_cd:
                trans_offset = num_uc * small_ele * 16 + num_cd * 16
                ori_offset = num_uc * cn_d * 16 + num_cd * n_d * 16
                n_burst = n_d
                burst_len = 1
                src_stride = self.c_0 - 1
                tik_instance.data_move(ub_ori[ori_offset],
                                       ub_trans[trans_offset],
                                       0, n_burst, burst_len,
                                       src_stride, 0)

        all_ele = num_unit_cur * c_d * n_d
        all_ele_div = _ceil_div(all_ele, 16)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        if all_ele_div == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, all_ele_div, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   all_ele_div,
                                   dst_rep_stride,
                                   src_rep_stride)

        if all_ele % self.cp_align_len > 0:
            if all_ele > self.cp_align_len:
                move_len = all_ele - self.cp_align_len
                dst_offset = num_unit_before * c_d * n_d
                burst_len = _ceil_div(move_len, self.cp_align_len)
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, burst_len, 0, 0)
                for k in range(self.cp_align_len):
                    ub_tail[k] = ub_trans[move_len + k]
                tik_instance.data_move(self.dst_gm[dst_offset + move_len],
                                       ub_tail,
                                       0, 1, 1, 0, 0)
            else:
                dst_offset = num_unit_before * c_d * n_d
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, 1, 0, 0)
        else:
            dst_offset = num_unit_before * c_d * n_d
            burst_len = all_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def not_align_mm_mid(self, tik_instance):
        """
        n_d < n_o * n_i
        c_d < c_1 * c_0
        n_d < 16 && c_d < 16
        c_d * n_d < cp_align_len
        and num_cn * c_d * n_d < cp_align_len
        """
        d_d, h_d, w_d, _, n_d = self.dst_shape
        dhw_d = d_d * h_d * w_d
        unit_ele = n_d * self.c_0 * 16
        num_unit_ub = self.ub_ele // unit_ele
        all_core_num = _ceil_div(dhw_d, num_unit_ub)
        ac_num = _set_core_num(all_core_num)

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

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core_num)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                num_unit_before = core_index * num_unit_ub

                with tik_instance.if_scope(core_index < all_core_num - 1):
                    num_unit_cur = num_unit_ub
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           num_unit_before, num_unit_cur
                    self.func_not_align_mm_mid(args)
                with tik_instance.else_scope():
                    num_unit_cur = dhw_d - ((all_core_num - 1) * num_unit_ub)
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           num_unit_before, num_unit_cur
                    self.func_not_align_mm_mid(args)

        return tik_instance

    def func_not_align_little_mid(self, args):
        """
        function of not align little mid
        """
        tik_instance, ub_ori, ub_trans, ub_tail,\
        num_unit_before, num_unit_cur = args

        small_ele = self.n_i * self.c_0
        _, _, _, c_d, n_d = self.dst_shape
        cn_d = c_d * n_d

        src_offset = num_unit_before * small_ele
        burst_len = num_unit_cur * small_ele // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + self.c_0 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        repeat_vconv = num_unit_cur * self.n_i
        if repeat_vconv == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, repeat_vconv, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   repeat_vconv,
                                   dst_rep_stride,
                                   src_rep_stride)

        with tik_instance.for_range(0, num_unit_cur) as num_uc:
            with tik_instance.for_range(0, c_d) as num_cd:
                trans_offset = num_uc * small_ele * 16 + num_cd * 16
                ori_offset = num_uc * cn_d * 16 + num_cd * n_d * 16
                n_burst = n_d
                burst_len = 1
                src_stride = self.c_0 - 1
                tik_instance.data_move(ub_ori[ori_offset],
                                       ub_trans[trans_offset],
                                       0, n_burst, burst_len,
                                       src_stride, 0)

        all_ele = num_unit_cur * c_d * n_d
        all_ele_div = _ceil_div(all_ele, 16)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        if all_ele_div == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, all_ele_div, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   all_ele_div,
                                   dst_rep_stride,
                                   src_rep_stride)

        if all_ele % self.cp_align_len > 0:
            if all_ele > self.cp_align_len:
                move_len = all_ele - self.cp_align_len
                dst_offset = num_unit_before * c_d * n_d
                burst_len = _ceil_div(move_len, self.cp_align_len)
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, burst_len, 0, 0)
                for k in range(self.cp_align_len):
                    ub_tail[k] = ub_trans[move_len + k]
                tik_instance.data_move(self.dst_gm[dst_offset + move_len],
                                       ub_tail,
                                       0, 1, 1, 0, 0)
            else:
                dst_offset = num_unit_before * c_d * n_d
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, 1, 0, 0)
        else:
            dst_offset = num_unit_before * c_d * n_d
            burst_len = all_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def not_align_little_mid(self, tik_instance):
        """
        n_d < n_o * n_i
        c_d < c_1 * c_0
        n_d < 16 && c_d < 16
        """
        d_d, h_d, w_d, _, _ = self.dst_shape
        dhw_d = d_d * h_d * w_d
        unit_ele = self.n_i * self.c_0 * 16
        num_unit_ub = self.ub_ele // unit_ele
        all_core_num = _ceil_div(dhw_d, num_unit_ub)
        ac_num = _set_core_num(all_core_num)

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

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core_num)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                num_unit_before = core_index * num_unit_ub

                with tik_instance.if_scope(core_index < all_core_num - 1):
                    num_unit_cur = num_unit_ub
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           num_unit_before, num_unit_cur
                    self.func_not_align_little_mid(args)
                with tik_instance.else_scope():
                    num_unit_cur = dhw_d - ((all_core_num - 1) * num_unit_ub)
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           num_unit_before, num_unit_cur
                    self.func_not_align_little_mid(args)

        return tik_instance

    def ub_to_ub_not_align_mid(self, args):
        """
        function of moving data from ub to ub for not align mid scene
        """
        tik_instance, ub_ori, ub_trans, n_d, c0_len, num_c1 = args

        with tik_instance.for_range(0, c0_len) as num_clen:
            ori_offset = num_c1 * self.c_0 * n_d * 16 \
                         + num_clen * n_d * 16
            trans_offset = num_c1 * n_d * self.c_0 * 16 \
                           + num_clen * 16
            n_burst = n_d
            burst_len = 16 // self.cp_align_len
            src_stride = self.c_0 - 1
            dst_stride = 0
            tik_instance.data_move(ub_ori[ori_offset],
                                   ub_trans[trans_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

    def not_align_mid(self, tik_instance):
        """
        n_d < n_o * n_i
        c_d < c_1 * c_0
        c_1 * c_0 * n_o * n_i <= ub_ele
        c_1 * c_0 * n_d * 16 <= ub_ele
        c_d * n_d >= cp_align_len
        """
        d_d, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        dhw_d = d_d * h_d * w_d
        ac_num = _set_core_num(dhw_d)
        hw_d = h_d * w_d

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

            ub_loop = _set_loop(tik_instance, num_core, ac_num, dhw_d)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                d_index = core_index // hw_d
                hw_index = core_index % hw_d
                big_src_ele = self.c_1 * h_d * w_d * n_o * self.n_i * self.c_0
                small_src_ele = n_o * self.n_i * self.c_0
                src_offset = d_index * big_src_ele + hw_index * small_src_ele
                n_burst = self.c_1
                burst_len = small_src_ele // self.cp_align_len
                src_stride = (hw_d - 1) * small_src_ele // self.cp_align_len
                args = tik_instance, self.src_gm, ub_ori, src_offset, 0,\
                       n_burst, burst_len, src_stride, 0, self.cp_align_len
                _gm_to_ub_one(args)

                with tik_instance.for_range(0, self.c_1) as num_c:
                    ori_begin = num_c * small_src_ele
                    trans_begin = num_c * n_d * self.c_0 * 16
                    src_list = [ub_ori[ori_begin + self.c_0 * i]
                                for i in range(16)]
                    dst_list = [ub_trans[trans_begin + 16 * i]
                                for i in range(16)]
                    if n_d == 1:
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list, n_d, 0, 0)
                    else:
                        src_rep_stride = 1
                        dst_rep_stride = 16
                        tik_instance.vnchwconv(False, False, dst_list,
                                               src_list,
                                               n_d,
                                               dst_rep_stride,
                                               src_rep_stride)

                with tik_instance.for_range(0, self.c_1) as num_c1:
                    with tik_instance.if_scope(num_c1 < self.c_1 - 1):
                        c0_len = self.c_0
                        args = tik_instance, ub_ori, ub_trans,\
                               n_d, c0_len, num_c1
                        self.ub_to_ub_not_align_mid(args)

                    with tik_instance.else_scope():
                        c0_len = c_d - ((self.c_1 - 1) * self.c_0)
                        args = tik_instance, ub_ori, ub_trans,\
                               n_d, c0_len, num_c1
                        self.ub_to_ub_not_align_mid(args)

                cn_d = c_d * n_d
                cn_d_align = _ceil_fill(cn_d, self.cp_align_len)
                cn_d_div = cn_d_align // self.cp_align_len

                ori_begin = 0
                trans_begin = 0
                src_list = [ub_ori[ori_begin + 16 * i]
                            for i in range(16)]
                dst_list = [ub_trans[trans_begin + 16 * i]
                            for i in range(16)]
                if cn_d_div == 1:
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list, cn_d_div, 0, 0)
                else:
                    src_rep_stride = 16
                    dst_rep_stride = 1
                    tik_instance.vnchwconv(False, False, dst_list,
                                           src_list,
                                           cn_d_div,
                                           dst_rep_stride,
                                           src_rep_stride)

                if cn_d % self.cp_align_len > 0:
                    move_len = cn_d - self.cp_align_len
                    dst_offset = core_index * cn_d
                    burst_len = _ceil_div(move_len, self.cp_align_len)
                    tik_instance.data_move(self.dst_gm[dst_offset],
                                           ub_trans,
                                           0, 1, burst_len, 0, 0)
                    for k in range(self.cp_align_len):
                        ub_tail[k] = ub_trans[move_len + k]

                    tik_instance.data_move(self.dst_gm[dst_offset + move_len],
                                           ub_tail,
                                           0, 1, 1, 0, 0)
                else:
                    dst_offset = core_index * cn_d
                    burst_len = cn_d // self.cp_align_len
                    tik_instance.data_move(self.dst_gm[dst_offset],
                                           ub_trans,
                                           0, 1, burst_len, 0, 0)

        return tik_instance

    def ub_to_ub_not_align_big(self, args):
        """
        function of moving data from ub to ub for not align big scene
        """
        tik_instance, ub_ori, ub_trans, ori_begin, trans_begin,\
        n_d, c0_len = args

        with tik_instance.for_range(0, c0_len) as num_clen:
            ori_offset = ori_begin + num_clen * n_d * 16
            trans_offset = trans_begin + num_clen * 16
            n_burst = n_d
            burst_len = 16 // self.cp_align_len
            src_stride = self.c_0 - 1
            dst_stride = 0
            tik_instance.data_move(ub_ori[ori_offset],
                                   ub_trans[trans_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

    def func_not_align_big_one(self, args):
        """
        function of not align big one
        """
        tik_instance, ub_ori, ub_trans, ub_tail, hw_d, big_src_ele,\
        small_src_ele, d_index, hw_index, c1_index, c0_len = args

        _, h_d, w_d, c_d, n_d = self.dst_shape
        big_dim_ele = h_d * w_d * c_d * n_d
        cn_d = c_d * n_d

        src_offset = d_index * big_src_ele \
                     + c1_index * hw_d * small_src_ele \
                     + hw_index * small_src_ele
        burst_len = small_src_ele // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + self.c_0 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        if n_d == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, n_d, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   n_d,
                                   dst_rep_stride,
                                   src_rep_stride)

        args = tik_instance, ub_ori, ub_trans, 0, 0, n_d, c0_len
        self.ub_to_ub_not_align_big(args)

        c0n_d = c0_len * n_d
        c0n_d_align = _ceil_fill(c0n_d, self.cp_align_len)
        c0n_d_div = c0n_d_align // self.cp_align_len

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        if c0n_d_div == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, c0n_d_div, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   c0n_d_div,
                                   dst_rep_stride,
                                   src_rep_stride)

        if c0n_d % self.cp_align_len > 0:
            move_len = c0n_d - self.cp_align_len
            dst_offset = d_index * big_dim_ele + hw_index * cn_d \
                         + c1_index * self.c_0 * n_d
            burst_len = _ceil_div(move_len, self.cp_align_len)
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

            for k in range(self.cp_align_len):
                ub_tail[k] = ub_trans[move_len + k]

            tik_instance.data_move(self.dst_gm[dst_offset + move_len],
                                   ub_tail,
                                   0, 1, 1, 0, 0)
        else:
            dst_offset = d_index * big_dim_ele + hw_index * cn_d \
                         + c1_index * self.c_0 * n_d
            burst_len = c0n_d // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def func_not_align_big_two(self, args):
        """
        function of not align big two
        """
        tik_instance, ub_ori, ub_trans, ub_tail, hw_d, big_src_ele, \
        small_src_ele, d_index, hw_index, c1_index, c0_len = args

        _, h_d, w_d, c_d, n_d = self.dst_shape
        big_dim_ele = h_d * w_d * c_d * n_d
        cn_d = c_d * n_d
        c1_num = 2

        src_offset = d_index * big_src_ele \
                     + (c1_index - 1) * hw_d * small_src_ele \
                     + hw_index * small_src_ele
        n_burst = c1_num
        burst_len = small_src_ele // self.cp_align_len
        src_stride = (hw_d - 1) * small_src_ele // self.cp_align_len
        args = tik_instance, self.src_gm, ub_ori, src_offset, 0, n_burst, \
               burst_len, src_stride, 0, self.cp_align_len
        _gm_to_ub_one(args)

        with tik_instance.for_range(0, c1_num) as num_c:
            ori_begin = num_c * small_src_ele
            trans_begin = num_c * n_d * self.c_0 * 16
            src_list = [ub_ori[ori_begin + self.c_0 * i]
                        for i in range(16)]
            dst_list = [ub_trans[trans_begin + 16 * i]
                        for i in range(16)]
            if n_d == 1:
                tik_instance.vnchwconv(False, False, dst_list,
                                       src_list, n_d, 0, 0)
            else:
                src_rep_stride = 1
                dst_rep_stride = 16
                tik_instance.vnchwconv(False, False, dst_list,
                                       src_list,
                                       n_d,
                                       dst_rep_stride,
                                       src_rep_stride)

        args = tik_instance, ub_ori, ub_trans, 0, 0, n_d, self.c_0
        self.ub_to_ub_not_align_big(args)

        ori_begin = self.c_0 * n_d * 16
        trans_begin = self.c_0 * n_d * 16
        args = tik_instance, ub_ori, ub_trans, ori_begin, trans_begin,\
               n_d, c0_len
        self.ub_to_ub_not_align_big(args)

        move_len = (self.c_0 + c0_len) * n_d
        move_len_align = _ceil_fill(move_len, self.cp_align_len)
        move_len_div = move_len_align // self.cp_align_len

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        if move_len_div == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, move_len_div, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   move_len_div,
                                   dst_rep_stride,
                                   src_rep_stride)

        if move_len % self.cp_align_len > 0:
            move_len_sub = move_len - self.cp_align_len
            dst_offset = d_index * big_dim_ele + hw_index * cn_d \
                         + (c1_index - 1) * self.c_0 * n_d
            burst_len = _ceil_div(move_len_sub, self.cp_align_len)
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

            for k in range(self.cp_align_len):
                ub_tail[k] = ub_trans[move_len_sub + k]

            tik_instance.data_move(self.dst_gm[dst_offset + move_len_sub],
                                   ub_tail,
                                   0, 1, 1, 0, 0)
        else:
            dst_offset = d_index * big_dim_ele + hw_index * cn_d \
                         + (c1_index - 1) * self.c_0 * n_d
            burst_len = move_len // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def not_align_big(self, tik_instance):
        """
        n_d < n_o * n_i
        c_d < c_1 * c_0
        c_0 * n_o * n_i <= ub_ele
        c_0 * n_d * 16 <= ub_ele
        c_d * n_d >= cp_align_len
        """
        d_d, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        dhwc1_d = d_d * h_d * w_d * self.c_1
        ac_num = _set_core_num(dhwc1_d)
        hwc1_d = h_d * w_d * self.c_1
        hw_d = h_d * w_d

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

            ub_loop = _set_loop(tik_instance, num_core, ac_num, dhwc1_d)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                d_index = core_index // hwc1_d
                one_index = core_index % hwc1_d
                hw_index = one_index // self.c_1
                c1_index = one_index % self.c_1
                big_src_ele = self.c_1 * h_d * w_d * n_o * self.n_i * self.c_0
                small_src_ele = n_o * self.n_i * self.c_0

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c0_len = self.c_0
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           hw_d, big_src_ele, small_src_ele, d_index,\
                           hw_index, c1_index, c0_len
                    self.func_not_align_big_one(args)

                with tik_instance.else_scope():
                    c0_len = c_d - ((self.c_1 - 1) * self.c_0)
                    nclen = n_d * c0_len
                    if nclen >= self.cp_align_len:
                        args = tik_instance, ub_ori, ub_trans, ub_tail, \
                               hw_d, big_src_ele, small_src_ele, d_index, \
                               hw_index, c1_index, c0_len
                        self.func_not_align_big_one(args)
                    else:
                        args = tik_instance, ub_ori, ub_trans, ub_tail,\
                               hw_d, big_src_ele, small_src_ele, d_index,\
                               hw_index, c1_index, c0_len
                        self.func_not_align_big_two(args)

        return tik_instance

    def move_for_not_align_splitn(self, args):
        """
        function of moving data for not align splitn
        """
        tik_instance, ub_ori, ub_trans, hw_d, big_src_ele, \
        small_src_ele, d_index, hw_index, c1_index, \
        c0_len, n_begin, n_len = args

        _, h_d, w_d, c_d, n_d = self.dst_shape
        big_dim_ele = h_d * w_d * c_d * n_d
        cn_d = c_d * n_d

        src_offset = d_index * big_src_ele \
                     + c1_index * hw_d * small_src_ele \
                     + hw_index * small_src_ele \
                     + n_begin * self.c_0
        burst_len = n_len * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + self.c_0 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]

        with tik_instance.if_scope(n_len == 1):
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, n_len, 0, 0)
        with tik_instance.else_scope():
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   n_len,
                                   dst_rep_stride,
                                   src_rep_stride)

        args = tik_instance, ub_ori, ub_trans, 0, 0, n_len, c0_len
        self.ub_to_ub_not_align_big(args)

        move_len = c0_len * n_len
        move_len_align = _ceil_fill(move_len, self.cp_align_len)
        move_len_div = move_len_align // self.cp_align_len

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]

        with tik_instance.if_scope(move_len_div == 1):
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, move_len_div, 0, 0)
        with tik_instance.else_scope():
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   move_len_div,
                                   dst_rep_stride,
                                   src_rep_stride)

        with tik_instance.for_range(0, c0_len) as num_clen:
            dst_offset = d_index * big_dim_ele + hw_index * cn_d \
                         + (c1_index * self.c_0 + num_clen) * n_d \
                         + n_begin
            trans_offset = num_clen * n_len
            burst_len = n_len // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans[trans_offset],
                                   0, 1, burst_len, 0, 0)

    def func_not_align_splitn(self, args):
        """
        function of not align splitn scene
        """
        tik_instance, ub_ori, ub_trans, hw_d, big_src_ele,\
        small_src_ele, fen_n, d_index, hw_index, c1_index,\
        c0_len, n_ub = args

        n_d = self.dst_shape[4]

        with tik_instance.for_range(0, fen_n) as num_f:
            with tik_instance.if_scope(num_f < fen_n - 1):
                n_begin = num_f * n_ub
                n_len = n_ub
                args = tik_instance, ub_ori, ub_trans, hw_d,\
                       big_src_ele, small_src_ele, d_index, hw_index,\
                       c1_index, c0_len, n_begin, n_len
                self.move_for_not_align_splitn(args)
            with tik_instance.else_scope():
                n_tail = n_d - (num_f * n_ub)
                n_tail_align = _ceil_fill(n_tail, self.cp_align_len)
                n_len = n_tail_align
                n_begin = n_d - n_len
                args = tik_instance, ub_ori, ub_trans, hw_d, \
                       big_src_ele, small_src_ele, d_index, hw_index, \
                       c1_index, c0_len, n_begin, n_len
                self.move_for_not_align_splitn(args)

    def not_align_splitn(self, tik_instance):
        """
        n_d < n_o * n_i
        c_d < c_1 * c_0
        c_0 * n_o * n_i > ub_ele || c_0 * n_d * 16 > ub_ele
        c_d * n_d >= cp_align_len
        """
        d_d, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        unit_ele = 16 * 16 * 16
        true_ele = (self.ub_ele // unit_ele) * unit_ele
        mid_tmp_ele = n_d * self.c_0 * 16
        n_ub = (self.ub_ele // unit_ele) * 16

        fen_n = _ceil_div(mid_tmp_ele, true_ele)
        dhwc1_d = d_d * h_d * w_d * self.c_1
        hwc1_d = h_d * w_d * self.c_1
        hw_d = h_d * w_d
        ac_num = _set_core_num(dhwc1_d)

        with tik_instance.for_range(0, ac_num, block_num=ac_num) as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (self.ub_ele,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (self.ub_ele,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)

            ub_loop = _set_loop(tik_instance, num_core, ac_num, dhwc1_d)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                d_index = core_index // hwc1_d
                one_index = core_index % hwc1_d
                hw_index = one_index // self.c_1
                c1_index = one_index % self.c_1

                big_src_ele = self.c_1 * h_d * w_d * n_o * self.n_i * self.c_0
                small_src_ele = n_o * self.n_i * self.c_0

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c0_len = self.c_0
                    args = tik_instance, ub_ori, ub_trans,\
                           hw_d, big_src_ele, small_src_ele, fen_n, d_index,\
                           hw_index, c1_index, c0_len, n_ub
                    self.func_not_align_splitn(args)

                with tik_instance.else_scope():
                    c0_len = c_d - ((self.c_1 - 1) * self.c_0)
                    args = tik_instance, ub_ori, ub_trans,\
                           hw_d, big_src_ele, small_src_ele, fen_n, d_index, \
                           hw_index, c1_index, c0_len, n_ub
                    self.func_not_align_splitn(args)

        return tik_instance

    def not_align_splitn_fencore(self, tik_instance):
        """
        n_d < n_o * n_i
        c_d < c_1 * c_0
        c_0 * n_o * n_i > ub_ele || c_0 * n_d * 16 > ub_ele
        c_d * n_d >= cp_align_len
        """
        d_d, h_d, w_d, c_d, n_d = self.dst_shape
        n_o = self.src_shape[1]
        unit_ele = 16 * 16 * 16
        true_ele = (self.ub_ele // unit_ele) * unit_ele
        mid_tmp_ele = n_d * self.c_0 * 16
        n_ub = (self.ub_ele // unit_ele) * 16

        fen_n = _ceil_div(mid_tmp_ele, true_ele)
        dhwc1_d = d_d * h_d * w_d * self.c_1
        hwc1_fen_d = h_d * w_d * self.c_1 * fen_n
        hw_d = h_d * w_d
        c1_fen_d = self.c_1 * fen_n
        all_core = dhwc1_d * fen_n
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
                d_index = core_index // hwc1_fen_d
                one_index = core_index % hwc1_fen_d
                hw_index = one_index // c1_fen_d
                two_index = one_index % c1_fen_d
                c1_index = two_index // fen_n
                fen_index = two_index % fen_n

                big_src_ele = self.c_1 * h_d * w_d * n_o * self.n_i * self.c_0
                small_src_ele = n_o * self.n_i * self.c_0

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c0_len = self.c_0
                    with tik_instance.if_scope(fen_index < fen_n - 1):
                        n_begin = fen_index * n_ub
                        n_len = n_ub
                        args = tik_instance, ub_ori, ub_trans, hw_d, \
                               big_src_ele, small_src_ele, d_index, hw_index, \
                               c1_index, c0_len, n_begin, n_len
                        self.move_for_not_align_splitn(args)
                    with tik_instance.else_scope():
                        n_tail = n_d - (fen_index * n_ub)
                        n_tail_align = _ceil_fill(n_tail, self.cp_align_len)
                        n_len = n_tail_align
                        n_begin = n_d - n_len
                        args = tik_instance, ub_ori, ub_trans, hw_d, \
                               big_src_ele, small_src_ele, d_index, hw_index, \
                               c1_index, c0_len, n_begin, n_len
                        self.move_for_not_align_splitn(args)

                with tik_instance.else_scope():
                    c0_len = c_d - ((self.c_1 - 1) * self.c_0)
                    with tik_instance.if_scope(fen_index < fen_n - 1):
                        n_begin = fen_index * n_ub
                        n_len = n_ub
                        args = tik_instance, ub_ori, ub_trans, hw_d, \
                               big_src_ele, small_src_ele, d_index, hw_index, \
                               c1_index, c0_len, n_begin, n_len
                        self.move_for_not_align_splitn(args)
                    with tik_instance.else_scope():
                        n_tail = n_d - (fen_index * n_ub)
                        n_tail_align = _ceil_fill(n_tail, self.cp_align_len)
                        n_len = n_tail_align
                        n_begin = n_d - n_len
                        args = tik_instance, ub_ori, ub_trans, hw_d, \
                               big_src_ele, small_src_ele, d_index, hw_index, \
                               c1_index, c0_len, n_begin, n_len
                        self.move_for_not_align_splitn(args)

        return tik_instance

    def func_little_align_fp32(self, args):
        """
        function of moving data for little align fp32 scene
        """
        tik_instance, ub_ori, ub_trans, ub_tail, dim_before, dim_cur = args

        c_d = self.dst_shape[3]
        n_d = self.dst_shape[4]
        n_o = self.src_shape[1]
        n_align = n_o * self.n_i

        src_offset = dim_before * n_align * self.c_0
        burst_len = dim_cur * n_align * self.c_0
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        in_ele = dim_cur * n_align * self.c_0
        dim_ele = in_ele * 2
        dim_zu = _ceil_div(dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        with tik_instance.for_range(0, c_d) as num_c:
            with tik_instance.for_range(0, n_d) as num_n:
                src_offset = (num_n * self.c_0 + num_c) * 2 * 16
                dst_offset = (num_c * n_d + num_n) * 2 * 16
                n_burst = dim_cur
                burst_len = 2
                src_stride = (n_align * self.c_0 - 1) * 2
                dst_stride = (n_d * c_d - 1) * 2
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
        mid_ele = dim_cur * c_d * n_d * 2
        mid_zu = _ceil_div(mid_ele, 16)

        if mid_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, mid_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   mid_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        out_ele = dim_cur * c_d * n_d
        if out_ele % self.cp_align_len > 0:
            with tik_instance.if_scope(out_ele > self.cp_align_len):
                sub_ele = out_ele - self.cp_align_len
                if sub_ele > 0:
                    dst_offset = dim_before * c_d * n_d
                    burst_len = _ceil_div(sub_ele, self.cp_align_len)
                    tik_instance.data_move(self.dst_gm[dst_offset],
                                           ub_trans,
                                           0, 1, burst_len, 0, 0)

                    for k in range(16):
                        ub_tail[k] = ub_trans[sub_ele * 2 + k]

                    tik_instance.data_move(self.dst_gm[dst_offset + sub_ele],
                                           ub_tail,
                                           0, 1, 1, 0, 0)
            with tik_instance.else_scope():
                dst_offset = dim_before * c_d * n_d

                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, 1, 0, 0)

        else:
            dst_offset = dim_before * c_d * n_d
            burst_len = out_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def little_align_fp32(self, tik_instance):
        """
        n_o * n_i * c_0 * 16 <= ub_ele
        n_d * c_d < cp_align_len
        """
        d_d, h_d, w_d, _, _ = self.dst_shape

        n_o = self.src_shape[1]
        n_align = n_o * self.n_i
        dim_ele = n_align * self.c_0 * 16
        num_dim_one_core = self.ub_ele // dim_ele
        dhw_d = d_d * h_d * w_d
        all_core = _ceil_div(dhw_d, num_dim_one_core)
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

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                with tik_instance.if_scope(core_index < all_core - 1):
                    dim_before = core_index * num_dim_one_core
                    dim_cur = num_dim_one_core
                    args = tik_instance, ub_ori, ub_trans, ub_tail,\
                           dim_before, dim_cur
                    self.func_little_align_fp32(args)

                with tik_instance.else_scope():
                    dim_before = (all_core - 1) * num_dim_one_core
                    dim_cur = dhw_d - dim_before
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           dim_before, dim_cur
                    self.func_little_align_fp32(args)

        return tik_instance

    def func_little_mm_fp32(self, args):
        """
        function of moving data for little mm fp32 scene
        """
        tik_instance, ub_ori, ub_trans, ub_tail, dim_before, dim_cur = args

        c_d = self.dst_shape[3]
        n_d = self.dst_shape[4]
        n_o = self.src_shape[1]
        n_align = n_o * self.n_i

        src_offset = dim_before * n_align * self.c_0
        n_burst = dim_cur
        burst_len = n_d * self.c_0 // self.cp_align_len
        src_stride = (n_align - n_d) * self.c_0 // self.cp_align_len
        dst_stride = 0
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, n_burst, burst_len, src_stride, dst_stride)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        in_ele = dim_cur * n_d * self.c_0
        dim_ele = in_ele * 2
        dim_zu = _ceil_div(dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        with tik_instance.for_range(0, c_d) as num_c:
            with tik_instance.for_range(0, n_d) as num_n:
                src_offset = (num_n * self.c_0 + num_c) * 2 * 16
                dst_offset = (num_c * n_d + num_n) * 2 * 16
                n_burst = dim_cur
                burst_len = 2
                src_stride = (n_d * self.c_0 - 1) * 2
                dst_stride = (n_d * c_d - 1) * 2
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
        mid_ele = dim_cur * c_d * n_d * 2
        mid_zu = _ceil_div(mid_ele, 16)

        if mid_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, mid_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   mid_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        out_ele = dim_cur * c_d * n_d
        if out_ele % self.cp_align_len > 0:
            with tik_instance.if_scope(out_ele > self.cp_align_len):
                sub_ele = out_ele - self.cp_align_len
                if sub_ele > 0:
                    dst_offset = dim_before * c_d * n_d
                    burst_len = _ceil_div(sub_ele, self.cp_align_len)
                    tik_instance.data_move(self.dst_gm[dst_offset],
                                           ub_trans,
                                           0, 1, burst_len, 0, 0)

                    for k in range(16):
                        ub_tail[k] = ub_trans[sub_ele * 2 + k]

                    tik_instance.data_move(self.dst_gm[dst_offset + sub_ele],
                                           ub_tail,
                                           0, 1, 1, 0, 0)
            with tik_instance.else_scope():
                dst_offset = dim_before * c_d * n_d

                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, 1, 0, 0)

        else:
            dst_offset = dim_before * c_d * n_d
            burst_len = out_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def little_mm_fp32(self, tik_instance):
        """
        n_d * c_0 * 16 <= ub_ele
        n_d * c_d < cp_align_len
        """
        d_d, h_d, w_d, _, n_d = self.dst_shape
        dim_ele = n_d * self.c_0 * 16
        num_dim_one_core = self.ub_ele // dim_ele
        dhw_d = d_d * h_d * w_d
        all_core = _ceil_div(dhw_d, num_dim_one_core)
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

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                with tik_instance.if_scope(core_index < all_core - 1):
                    dim_before = core_index * num_dim_one_core
                    dim_cur = num_dim_one_core
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           dim_before, dim_cur
                    self.func_little_mm_fp32(args)

                with tik_instance.else_scope():
                    dim_before = (all_core - 1) * num_dim_one_core
                    dim_cur = dhw_d - dim_before
                    args = tik_instance, ub_ori, ub_trans, ub_tail, \
                           dim_before, dim_cur
                    self.func_little_mm_fp32(args)

        return tik_instance

    def func_c0n_core(self, args):
        """
        function of moving data for c0n_core scene
        """
        tik_instance, ub_ori, ub_trans, ub_tail, d_index, \
        c1_index, hw_index, c_before, c_now = args

        _, h_d, w_d, c_d, n_d = self.dst_shape
        hw_d = h_d * w_d
        n_o = self.src_shape[1]

        src_offset = (d_index * self.c_1 * hw_d + c1_index * hw_d + hw_index) \
                     * n_o * self.n_i * self.c_0
        burst_len = n_d * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        in_ele = n_d * self.c_0
        dim_ele = in_ele * 2
        dim_zu = _ceil_div(dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        with tik_instance.for_range(0, c_now) as num_c:
            src_offset = num_c * 2 * 16
            dst_offset = num_c * n_d * 2 * 16
            n_burst = n_d
            burst_len = 2
            src_stride = (self.c_0 - 1) * 2
            dst_stride = 0
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
        mid_ele = c_now * n_d * 2
        mid_zu = _ceil_div(mid_ele, 16)

        if mid_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, mid_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   mid_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        out_ele = c_now * n_d
        with tik_instance.if_scope(out_ele % self.cp_align_len > 0):
            sub_ele = out_ele - self.cp_align_len
            if sub_ele > 0:
                dst_offset = (d_index * hw_d + hw_index) * c_d * n_d\
                             + c_before * n_d
                burst_len = _ceil_div(sub_ele, self.cp_align_len)
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, burst_len, 0, 0)

                for k in range(16):
                    ub_tail[k] = ub_trans[sub_ele * 2 + k]

                tik_instance.data_move(self.dst_gm[dst_offset + sub_ele],
                                       ub_tail,
                                       0, 1, 1, 0, 0)

        with tik_instance.else_scope():
            dst_offset = (d_index * hw_d + hw_index) * c_d * n_d\
                         + c_before * n_d
            burst_len = out_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def func_c0n_core_tail(self, args):
        """
        function of moving data for tail of c0n_core scene
        """
        tik_instance, ub_ori, ub_trans, ub_tail, d_index, \
        c1_index, hw_index, c_before, c_now = args

        _, h_d, w_d, c_d, n_d = self.dst_shape
        hw_d = h_d * w_d
        n_o = self.src_shape[1]

        src_offset = (d_index * self.c_1 * hw_d + c1_index * hw_d + hw_index)\
                     * n_o * self.n_i * self.c_0
        burst_len = n_d * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        c1_before = c1_index + 1
        src_offset = (d_index * self.c_1 * hw_d + c1_before * hw_d + hw_index)\
                     * n_o * self.n_i * self.c_0
        ub_offset = n_d * self.c_0 * 2
        burst_len = n_d * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori[ub_offset],
                               self.src_gm[src_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        in_ele = n_d * self.c_0 * 2
        dim_ele = in_ele * 2
        dim_zu = _ceil_div(dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        c_one = self.c_0
        c_two = c_now - c_one
        with tik_instance.for_range(0, c_one) as num_c:
            src_offset = num_c * 2 * 16
            dst_offset = num_c * n_d * 2 * 16
            n_burst = n_d
            burst_len = 2
            src_stride = (self.c_0 - 1) * 2
            dst_stride = 0
            tik_instance.data_move(ub_ori[dst_offset],
                                   ub_trans[src_offset],
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

        with tik_instance.for_range(0, c_two) as num_c:
            src_offset = (c_one * n_d + num_c) * 2 * 16
            dst_offset = (c_one * n_d + num_c * n_d) * 2 * 16
            n_burst = n_d
            burst_len = 2
            src_stride = (self.c_0 - 1) * 2
            dst_stride = 0
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
        mid_ele = c_now * n_d * 2
        mid_zu = _ceil_div(mid_ele, 16)

        if mid_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, mid_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   mid_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        out_ele = c_now * n_d
        with tik_instance.if_scope(out_ele % self.cp_align_len > 0):
            sub_ele = out_ele - self.cp_align_len
            if sub_ele > 0:
                dst_offset = (d_index * hw_d + hw_index) * c_d * n_d\
                             + c_before * n_d
                burst_len = _ceil_div(sub_ele, self.cp_align_len)
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans,
                                       0, 1, burst_len, 0, 0)

                for k in range(16):
                    ub_tail[k] = ub_trans[sub_ele * 2 + k]

                tik_instance.data_move(self.dst_gm[dst_offset + sub_ele],
                                       ub_tail,
                                       0, 1, 1, 0, 0)

        with tik_instance.else_scope():
            dst_offset = (d_index * hw_d + hw_index) * c_d * n_d\
                         + c_before * n_d
            burst_len = out_ele // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, 1, burst_len, 0, 0)

    def c0n_ele_fp32(self, tik_instance):
        """
        n_d * self.c_0 * 16 <= ub_ele
        """
        d_d, h_d, w_d, c_d, n_d = self.dst_shape
        hw_d = h_d * w_d

        all_core = d_d * self.c_1 * hw_d
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

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                c1hw_d = self.c_1 * hw_d
                d_index = core_index // c1hw_d
                c1hw_index = core_index % c1hw_d
                c1_index = c1hw_index // hw_d
                hw_index = c1hw_index % hw_d

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c_before = c1_index * self.c_0
                    c_now = self.c_0
                    args = tik_instance, ub_ori, ub_trans, ub_tail, d_index, \
                           c1_index, hw_index, c_before, c_now
                    self.func_c0n_core(args)

                with tik_instance.else_scope():
                    c_before = (self.c_1 - 1) * self.c_0
                    c_now = c_d - c_before
                    if c_now * n_d >= self.cp_align_len:
                        args = tik_instance, ub_ori, ub_trans, ub_tail, \
                               d_index, c1_index, hw_index, c_before, c_now
                        self.func_c0n_core(args)
                    else:
                        c1_index = self.c_1 - 2
                        c_before = (self.c_1 - 2) * self.c_0
                        c_now = c_d - c_before
                        args = tik_instance, ub_ori, ub_trans, ub_tail, \
                               d_index, c1_index, hw_index, c_before, c_now
                        self.func_c0n_core_tail(args)

        return tik_instance

    def func_split_n(self, args):
        """
        function of moving data for split_n scene
        """
        tik_instance, ub_ori, ub_trans, \
        d_index, c1_index, hw_index, c_before, c_now, \
        n_before, n_now = args

        _, h_d, w_d, c_d, n_d = self.dst_shape
        hw_d = h_d * w_d
        n_o = self.src_shape[1]
        data_offset = (d_index * self.c_1 * hw_d + c1_index * hw_d + hw_index)\
                      * n_o * self.n_i * self.c_0 + n_before * self.c_0
        burst_len = n_now * self.c_0 // self.cp_align_len
        tik_instance.data_move(ub_ori,
                               self.src_gm[data_offset],
                               0, 1, burst_len, 0, 0)

        ori_begin = 0
        trans_begin = 0
        src_list = [ub_ori[ori_begin + 16 * i]
                    for i in range(16)]
        dst_list = [ub_trans[trans_begin + 16 * i]
                    for i in range(16)]
        in_ele = n_now * self.c_0
        dim_ele = in_ele * 2
        dim_zu = _ceil_div(dim_ele, 16)

        if dim_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, dim_zu, 0, 0)
        else:
            src_rep_stride = 1
            dst_rep_stride = 16
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   dim_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        with tik_instance.for_range(0, c_now) as num_c:
            src_offset = num_c * 2 * 16
            dst_offset = num_c * n_now * 2 * 16
            n_burst = n_now
            burst_len = 2
            src_stride = (self.c_0 - 1) * 2
            dst_stride = 0
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
        mid_ele = c_now * n_now * 2
        mid_zu = _ceil_div(mid_ele, 16)

        if mid_zu == 1:
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list, mid_zu, 0, 0)
        else:
            src_rep_stride = 16
            dst_rep_stride = 1
            tik_instance.vnchwconv(False, False, dst_list,
                                   src_list,
                                   mid_zu,
                                   dst_rep_stride,
                                   src_rep_stride)

        with tik_instance.if_scope(n_d % self.cp_align_len > 0):
            with tik_instance.for_range(0, c_now) as num_c:
                dst_offset = (d_index * hw_d + hw_index) * c_d * n_d \
                             + (c_before + num_c) * n_d + n_before
                ub_offset = num_c * n_now * 2
                burst_len = n_now // self.cp_align_len
                tik_instance.data_move(self.dst_gm[dst_offset],
                                       ub_trans[ub_offset],
                                       0, 1, burst_len, 0, 0)

        with tik_instance.else_scope():
            dst_offset = (d_index * hw_d + hw_index) * c_d * n_d\
                         + c_before * n_d + n_before
            n_burst = c_now
            burst_len = n_now // self.cp_align_len
            src_stride = 0
            dst_stride = (n_d - n_now) // self.cp_align_len
            tik_instance.data_move(self.dst_gm[dst_offset],
                                   ub_trans,
                                   0, n_burst, burst_len,
                                   src_stride, dst_stride)

    def split_n_fp32(self, tik_instance):
        """
        n_d * self.c_0 * 16 > ub_ele
        """
        d_d, h_d, w_d, c_d, n_d = self.dst_shape
        hw_d = h_d * w_d
        n_ub = self.ub_ele // 16 // self.c_0 \
               // self.cp_align_len * self.cp_align_len
        n_zu = _ceil_div(n_d, n_ub)

        all_core = d_d * self.c_1 * hw_d * n_zu
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

            ub_loop = _set_loop(tik_instance, num_core,
                                ac_num, all_core)

            with tik_instance.for_range(0, ub_loop) as num_u:
                core_index = num_u * ac_num + num_core
                c1hwn_d = self.c_1 * hw_d * n_zu
                d_index = core_index // c1hwn_d
                c1hwn_index = core_index % c1hwn_d
                hwn_d = hw_d * n_zu
                c1_index = c1hwn_index // hwn_d
                hwn_index = c1hwn_index % hwn_d
                hw_index = hwn_index // n_zu
                nzu_index = hwn_index % n_zu

                with tik_instance.if_scope(c1_index < self.c_1 - 1):
                    c_before = c1_index * self.c_0
                    c_now = self.c_0

                    with tik_instance.if_scope(nzu_index < n_zu - 1):
                        n_before = nzu_index * n_ub
                        n_now = n_ub
                        args = tik_instance, ub_ori, ub_trans, \
                               d_index, c1_index, hw_index, c_before, c_now, \
                               n_before, n_now
                        self.func_split_n(args)

                    with tik_instance.else_scope():
                        n_now_temp = n_d - (n_zu - 1) * n_ub
                        n_now = _ceil_fill(n_now_temp, self.cp_align_len)
                        n_before = n_d - n_now
                        args = tik_instance, ub_ori, ub_trans, \
                               d_index, c1_index, hw_index, c_before, c_now, \
                               n_before, n_now
                        self.func_split_n(args)

                with tik_instance.else_scope():
                    c_before = (self.c_1 - 1) * self.c_0
                    c_now = c_d - c_before

                    with tik_instance.if_scope(nzu_index < n_zu - 1):
                        n_before = nzu_index * n_ub
                        n_now = n_ub
                        args = tik_instance, ub_ori, ub_trans, \
                               d_index, c1_index, hw_index, c_before, c_now, \
                               n_before, n_now
                        self.func_split_n(args)

                    with tik_instance.else_scope():
                        n_now_temp = n_d - (n_zu - 1) * n_ub
                        n_now = _ceil_fill(n_now_temp, self.cp_align_len)
                        n_before = n_d - n_now
                        args = tik_instance, ub_ori, ub_trans, \
                               d_index, c1_index, hw_index, c_before, c_now, \
                               n_before, n_now
                        self.func_split_n(args)

        return tik_instance

    def fz3d_2_dhwcn_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        if self.dtype == "float32":
            branch = self.check_branch_fp32()
        else:
            branch = self.check_branch()

        if branch == "n_align_small":
            tik_instance = self.n_align_small(tik_instance)
        elif branch == "n_align_multi_small":
            tik_instance = self.n_align_multi_small(tik_instance)
        elif branch == "n_align_multi_mid":
            tik_instance = self.n_align_multi_mid(tik_instance)
        elif branch == "n_align_mid":
            tik_instance = self.n_align_mid(tik_instance)
        elif branch == "n_align_big":
            tik_instance = self.n_align_big(tik_instance)
        elif branch == "n_align_splitn":
            tik_instance = self.n_align_splitn(tik_instance)
        elif branch == "not_align_mm":
            tik_instance = self.not_align_mm(tik_instance)
        elif branch == "not_align_little":
            tik_instance = self.not_align_little(tik_instance)
        elif branch == "not_align_small":
            tik_instance = self.not_align_small(tik_instance)
        elif branch == "not_align_mm_mid":
            tik_instance = self.not_align_mm_mid(tik_instance)
        elif branch == "not_align_little_mid":
            tik_instance = self.not_align_little_mid(tik_instance)
        elif branch == "not_align_mid":
            tik_instance = self.not_align_mid(tik_instance)
        elif branch == "not_align_big":
            tik_instance = self.not_align_big(tik_instance)
        elif branch == "not_align_splitn_fencore":
            tik_instance = self.not_align_splitn_fencore(tik_instance)
        elif branch == "not_align_splitn":
            tik_instance = self.not_align_splitn(tik_instance)
        elif branch == "little_mm_fp32":
            tik_instance = self.little_mm_fp32(tik_instance)
        elif branch == "little_align_fp32":
            tik_instance = self.little_align_fp32(tik_instance)
        elif branch == "c0n_ele_fp32":
            tik_instance = self.c0n_ele_fp32(tik_instance)
        elif branch == "split_n_fp32":
            tik_instance = self.split_n_fp32(tik_instance)

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
        tik_instance = self.fz3d_2_dhwcn_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.src_gm],
                              outputs=[self.dst_gm])

        return tik_instance


def _error_log(param, value, reason):
    error_info = {
        'ErrCode': 'E10001',
        'parameter': param,
        'value': value,
        'reason': reason
    }
    raise RuntimeError(error_info,
                       "Invalid value for {parameter}[{value}], "
                       "{reason}.".format(**error_info))


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
        reason = "src_format must be FRACTAL_Z_3D !"
        _error_log("src_format", src_format, reason)

    if dst_format.lower() != "dhwcn":
        reason = "dst_format must be DHWCN !"
        _error_log("dst_format", dst_format, reason)

    check_list = ("float16", "float32")
    check_dtype(dtype, check_list)
    if dtype != dtype_dst:
        reason = "dtype of src and dst are different !"
        _error_log("dst_dtype", dtype_dst, reason)

    check_shape(src_shape, min_rank=4, max_rank=4)
    check_shape(dst_shape, min_rank=5, max_rank=5)

    if src_shape[2] != 16:
        reason = "the 3rd dimension of src_shape is not 16, Ni must be 16 !"
        _error_log("Ni", src_shape[2], reason)

    if src_shape[3] != 16:
        reason = "the 4th dimension of src_shape is not 16, C0 must be 16 !"
        _error_log("C0", src_shape[3], reason)

    d_d, h_d, w_d, c_d, n_d = dst_shape

    n_i = 16
    n_s = n_i - 1
    n_o = (n_d + n_s) // n_i

    if src_shape[1] != n_o:
        reason = "the 2nd dimension of src_shape is wrong, " \
                 "No must be (N + 15)//16 !"
        _error_log("No", src_shape[1], reason)

    c_0 = 16
    c_s = c_0 - 1
    c_1 = (c_d + c_s) // c_0
    one_dim = d_d * c_1 * h_d * w_d

    if src_shape[0] != one_dim:
        reason = "the 1st dimension of src_shape is wrong, " \
                 "it must be D*C1*H*W !"
        _error_log("DC1HW", src_shape[0], reason)


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR, REQUIRED_ATTR_STR, KERNEL_NAME)
def fractal_z_3d_2_dhwcn(src, dst, src_format, dst_format,
                         kernel_name="fractal_z_3d_2_dhwcn"):
    """
    algorithm: fractal_z_3d_2_dhwcn
    calculating: change data format from FRACTAL_Z_3D to DHWCN

    Parameters
    ----------
    src: dict
        dict with keys(shape, dtype) of src
    dst: dict
        dict with keys(shape, dtype) of dst
    src_format: str
        data format of src, only support "FRACTAL_Z_3D"
    dst_format: str
        data format of dst, only support "DHWCN"
    kernel_name: str
        kernel name, default value is "fractal_z_3d_2_dhwcn"

    Returns
    -------
    tik_instance: tik_instance
    """
    _check_parameters(src, dst, src_format, dst_format)
    src_shape = src.get("shape")
    dst_shape = dst.get("shape")
    dtype = src.get("dtype").lower()

    template_fp16 = Fz3d2DhwcnCompute(src_shape, dst_shape, dtype, kernel_name)
    return template_fp16.get_tik_instance()
