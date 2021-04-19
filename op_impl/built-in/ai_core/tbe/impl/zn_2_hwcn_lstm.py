# Copyright 2020 Huawei Technologies Co., Ltd
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
zn_2_hwcn_lstm
"""
import math
from functools import reduce as functools_reduce

from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *
from topi.cce import util

# available ub size
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# available number of cores
MAX_CORE_NUM = tbe_platform.cce_conf.get_soc_spec(
    tbe_platform.cce_conf.CORE_NUM)


# pylint: disable=too-many-instance-attributes,too-many-statements
# pylint: disable=too-many-arguments
class ZN2HWCNLSTM():

    """
    Rearranges data from HWCN format into FRACTAL_NZ_LSTM format
    """
    def __init__(self, src, dst, src_format, dst_format, kernel_name):

        """
        Init zn_2_hwcn_lstm parameters

        Parameters
        ----------
        src : dict, shape and dtype of input.
        dst: dict, shape and dtype of input.
        src_format: str, source data format, can be fractal_zn.
        dst_format: str, target data format, can be hwcn.
        kernel_name: str, kernel name, default value is "zn_2_hwcn_lstm".
        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.src_format = src_format
        self.dst_format = dst_format
        self.kernel_name = kernel_name
        self.src_shape = src.get("shape")
        self.src_dtype = src.get("dtype").lower()
        self.dst_shape = dst.get("shape")
        self.dst_dtype = dst.get("dtype").lower()
        self.h = self.dst_shape[3] // 4
        self.i = self.dst_shape[2] - self.h
        self.h_align = math.ceil(self.h / 16) * 16
        self.i_align = math.ceil(self.i / 16) * 16
        self.src_data_num = functools_reduce(lambda x, y: x * y,
                                             self.src_shape[:])
        self.ub_size_bytes = UB_SIZE - 9216
        self.core_num = MAX_CORE_NUM
        self.src_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(
            self.src_dtype) // 8
        self.src_data_each_block = 32 // self.src_dtype_bytes_size
        self.src_gm = self.tik_instance.Tensor(
            self.src_dtype, self.src_shape, name="src_gm", scope=tik.scope_gm)
        self.dst_gm = self.tik_instance.Tensor(
            self.dst_dtype, self.dst_shape, name="dst_gm",
            scope=tik.scope_gm)
        self.src_burst_len = 0
        self.burst_len = 0
        self.dst_burst_len = 0
        self.src_ub_number = 0
        self.temp_ub_number = 0
        self.half_ub_number = 0
        self.each_data_num = 0
        self.each_core_data = 0
        self.last_core_data = 0
        self.c_num = 0
        self.i_flag = 0
        self.n0_ni_c0 = (self.src_shape[1] * self.src_shape[2] *
                         self.src_shape[3])
        self.ni_c0 = self.src_shape[2] * self.src_shape[3]
        self.temp_c = None
        self.before_c = None
        self.core_loop_index = None
        self.src_ub = None
        self.temp_ub = None
        self.tile_ub = None
        self.temp_burst_len = None
        self.remain_data = None
        self.temp_data = None
        ai_core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.c_each_core = math.ceil(self.src_shape[0] / ai_core_num)
        self.core_num = math.ceil(self.src_shape[0] / self.c_each_core)
        self.each_core_data = self.c_each_core * self.n0_ni_c0
        self.c_last_core = (self.src_shape[0] - self.c_each_core *
                            (self.core_num - 1))
        self.last_core_data = self.c_last_core * self.n0_ni_c0
        data_num = (self.i % (self.c_each_core * 16)) * self.dst_shape[3]
        if (data_num < self.src_data_each_block) and data_num > 0:
            self.core_num = self.change_core_num()
            self.each_core_data = self.c_each_core * self.n0_ni_c0
            self.c_last_core = (self.src_shape[0] - self.c_each_core *
                                (self.core_num - 1))
            self.last_core_data = self.c_last_core * self.n0_ni_c0
        self.check_param()


    def change_core_num(self):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.c_each_core = self.c_each_core + 1
        core_num = math.ceil(self.src_shape[0] / self.c_each_core)
        num = (self.i % (self.c_each_core * 16)) * self.dst_shape[3]
        if (num < self.src_data_each_block) and (core_num > 1) and (num > 0):
            self.change_core_num()
        return core_num


    def init_ub_tensor(self, each_core_data, c_each_core):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        each_core_data: each core data num
        c_each_core: each core c num

        Returns
        -------
        None
        """
        self.half_ub_number = ((self.ub_size_bytes // 2 //
                                self.src_dtype_bytes_size //
                                self.src_data_each_block) *
                               self.src_data_each_block)
        if self.half_ub_number > each_core_data:
            self.src_ub_number = math.ceil(
                each_core_data /
                self.src_data_each_block) * self.src_data_each_block
            self.each_data_num = each_core_data
            self.c_num = c_each_core * self.src_shape[3]
        else:
            self.each_data_num = (self.half_ub_number // self.n0_ni_c0 *
                                  self.n0_ni_c0)
            self.src_ub_number = (math.ceil(self.each_data_num /
                                            self.src_data_each_block) *
                                  self.src_data_each_block)
            self.c_num = (self.half_ub_number // self.n0_ni_c0 *
                          self.src_shape[3])

        self.temp_ub_number = self.src_ub_number
        self.src_ub = self.tik_instance.Tensor(
            self.src_dtype, (self.src_ub_number,),
            name="src_ub", scope=tik.scope_ubuf)
        self.temp_ub = self.tik_instance.Tensor(
            self.src_dtype, (self.temp_ub_number,),
            name="temp_ub", scope=tik.scope_ubuf)
        self.tile_ub = self.tik_instance.Tensor(
            self.src_dtype, (self.src_data_each_block,),
            name="tile_ub", scope=tik.scope_ubuf)

        self.temp_c = self.tik_instance.Scalar("int32")
        self.temp_c.set_as(0)

        self.before_c = self.tik_instance.Scalar("int32")
        self.before_c.set_as(0)

        self.core_loop_index = self.tik_instance.Scalar("int32")
        self.core_loop_index.set_as(0)

        self.temp_burst_len = self.tik_instance.Scalar("int32")
        self.temp_burst_len.set_as(0)

        self.remain_data = self.tik_instance.Scalar("int32")
        self.remain_data.set_as(0)

        self.temp_data = self.tik_instance.Scalar("int32")
        self.temp_data.set_as(0)




    def cal_loop(self, each_core_data, c_each_core):
        """
        cal loop num

        Parameters
        ----------
        each_core_data: each core data num
        c_each_core: each core c num

        Returns
        -------
        None
        """
        loop = each_core_data // self.each_data_num
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as loop_index:
                self.move_to_ub(self.src_ub_number, loop_index)
                scalar_flag = self.scalar_or_not()
                if scalar_flag:
                    self.trans_format_scalar()
                else:
                    self.trans_format(self.each_core_data, self.c_each_core,
                                      self.each_data_num)
                self.move_to_gm(loop_index, each_core_data, c_each_core)
        last_num = each_core_data % self.each_data_num
        if last_num > 0:
            if loop > 0:
                self.c_num = ((c_each_core %
                               (self.half_ub_number // self.n0_ni_c0)) *
                              self.src_shape[3])
            self.move_to_ub(last_num, loop)
            scalar_flag = self.scalar_or_not()
            if scalar_flag:
                self.trans_format_scalar()
            else:
                self.trans_format(self.last_core_data, self.c_last_core,
                                  last_num)
            self.move_to_gm(loop, each_core_data, c_each_core)


    def move_to_ub(self, element_num, loop_index):
        """
        move data to UB

        Parameters
        ----------
        element_num: the data to move
        loop_index: loop num

        Returns
        -------
        None
        """

        self.src_burst_len = math.ceil(element_num /
                                       self.src_data_each_block)
        self.tik_instance.data_move(
            self.src_ub, self.src_gm[self.core_loop_index *
                                     self.each_core_data + loop_index *
                                     self.each_data_num], 0, 1,
            self.src_burst_len, 0, 0)


    def scalar_or_not(self):
        """
        use scalar or not

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        c_flag = self.dst_shape[2] % 16
        n_flag = self.dst_shape[3] % 16
        if (not c_flag) and (not n_flag) and (self.src_dtype == "float16"):
            return False
        return True


    def trans_format(self, each_core_data, c_each_core, data_num):
        """
        trans format

        Parameters
        ----------
        each_core_data: each core data num
        c_each_core: each core c num
        data_num: data num

        Returns
        -------
        None
        """
        if self.half_ub_number > each_core_data:
            repeat_num = c_each_core
        else:
            repeat_num = data_num // self.n0_ni_c0
        with self.tik_instance.for_range(0, repeat_num) as repeat_index:
            src_rep_stride = 16
            dst_rep_stride = 1
            src_list = [self.src_ub[repeat_index * self.n0_ni_c0 + 16 * i]
                        for i in range(16)]
            dst_list = [self.temp_ub[repeat_index * self.dst_shape[3] * 16 +
                                     self.dst_shape[3] * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                        self.src_shape[1], dst_rep_stride,
                                        src_rep_stride)


    def trans_format_scalar(self):
        """
        trans format

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.c_num) as c_index:
            with self.tik_instance.for_range(0, 4) as n_index:
                with self.tik_instance.for_range(0, self.h) as h_index:
                    num_c1 = c_index // 16
                    num_c0 = c_index % 16
                    temp_offset = (c_index * self.dst_shape[3] + n_index *
                                   self.h + h_index)
                    src_offset = (num_c1 * self.n0_ni_c0 + n_index *
                                  self.h_align * 16 + 16 * h_index + num_c0)
                    self.temp_ub[temp_offset] = self.src_ub[src_offset]


    def move_to_gm(self, loop_index, each_core_data, c_each_core):
        """
        move data to gm

        Parameters
        ----------
        loop_index:loop num
        each_core_data: each core data num
        c_each_core: each core c num

        Returns
        -------
        None
        """
        self.temp_c.set_as((self.core_loop_index * self.c_each_core *
                            self.src_shape[3] +
                            (self.each_data_num // self.n0_ni_c0 *
                             self.src_shape[3]) * (loop_index + 1)))
        self.before_c.set_as(self.core_loop_index * self.c_each_core *
                             self.src_shape[3] +
                             (self.each_data_num // self.n0_ni_c0 *
                              self.src_shape[3]) * loop_index)
        with self.tik_instance.if_scope(loop_index ==
                                        (each_core_data // self.each_data_num)):
            self.temp_c.set_as((self.core_loop_index * self.c_each_core +
                                c_each_core) * self.src_shape[3])
        with self.tik_instance.if_scope(self.temp_c <= self.i):
            temp_data = self.c_num * self.dst_shape[3]
            offset = self.before_c * self.dst_shape[3]
            self.burst_len = math.ceil(temp_data / self.src_data_each_block)
            self.tik_instance.data_move(
                self.dst_gm[offset], self.temp_ub, 0, 1, self.burst_len, 0, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.before_c <= self.i):
                self.i_flag = self.temp_c
                self.temp_data.set_as((self.i - self.before_c) *
                                      self.dst_shape[3])
                offset = self.before_c * self.dst_shape[3]
                remain_num = self.temp_data % self.src_data_each_block
                with self.tik_instance.if_scope(self.temp_data //
                                                self.src_data_each_block != 0):
                    with self.tik_instance.if_scope(remain_num == 0):
                        self.temp_burst_len.set_as(
                            self.temp_data // self.src_data_each_block)
                        self.tik_instance.data_move(
                            self.dst_gm[offset], self.temp_ub, 0, 1,
                            self.temp_burst_len, 0, 0)
                    with self.tik_instance.else_scope():
                        self.temp_burst_len.set_as(
                            self.temp_data // self.src_data_each_block + 1)
                        align_num = (self.temp_data //
                                     self.src_data_each_block *
                                     self.src_data_each_block)
                        align_offset = (offset + align_num -
                                        (self.src_data_each_block - remain_num))
                        with self.tik_instance.for_range(
                                0, self.src_data_each_block) as index:
                            temp_offset = (align_num -
                                           (self.src_data_each_block -
                                            remain_num) + index)
                            self.tile_ub[index] = self.temp_ub[temp_offset]
                        self.tik_instance.data_move(
                            self.dst_gm[align_offset], self.tile_ub, 0, 1,
                            1, 0, 0)
                        self.tik_instance.data_move(
                            self.dst_gm[offset], self.temp_ub, 0, 1,
                            self.temp_burst_len - 1, 0, 0)
                self.remain_data.set_as((self.temp_c - self.i_align) *
                                        self.dst_shape[3])
                offset = offset + self.temp_data
                offset_ub = ((self.i_align - self.before_c) *
                             self.dst_shape[3])
                self.temp_burst_len.set_as((self.remain_data +
                                            self.src_data_each_block - 1) //
                                           self.src_data_each_block)
                with self.tik_instance.if_scope(self.temp_burst_len != 0):
                    self.tik_instance.data_move(
                        self.dst_gm[offset], self.temp_ub[offset_ub], 0, 1,
                        self.temp_burst_len, 0, 0)
            with self.tik_instance.else_scope():
                temp_data = self.c_num * self.dst_shape[3]
                offset = ((self.i_flag - self.i_align + self.i) *
                          self.dst_shape[3] +
                          (self.before_c - self.i_flag) * self.dst_shape[3])
                self.burst_len = math.ceil(temp_data / self.src_data_each_block)
                self.tik_instance.data_move(
                    self.dst_gm[offset], self.temp_ub, 0, 1,
                    self.burst_len, 0, 0)


    def check_param(self):
        """
        Check parameter

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.src_format != "FRACTAL_ZN_LSTM":
            raise RuntimeError("src format must be FRACTAL_ZN_LSTM")

        if self.dst_format != "HWCN":
            raise RuntimeError("dst format must be HWCN")

        if self.src_dtype not in ("float16", "float32"):
            raise RuntimeError("dtype must be float16 and float32")

        if self.src_dtype != self.dst_dtype:
            raise RuntimeError("dtype of src and dst should be same !")

        if self.src_shape[2] != 16 or self.src_shape[3] != 16:
            raise RuntimeError("the third and forth dimension must be 16")

        if self.dst_shape[0] != 1 or self.dst_shape[1] != 1:
            raise RuntimeError("the dst shape's H and W must be 1")

        if (self.dst_shape[3] % 4) != 0:
            raise RuntimeError("the dst shape's N must be divide by 4")

        if len(self.src_shape) != 4 or len(self.dst_shape) != 4:
            raise RuntimeError("the src and dst must be 4 dimension")

        h_value = self.dst_shape[3] // 4
        i_value = self.dst_shape[2] - h_value
        h_align = math.ceil(h_value / 16) * 16
        i_align = math.ceil(i_value / 16) * 16
        if i_value <= 0:
            raise RuntimeError("the dst shape's C is wrong")

        if ((i_align + h_align) != self.src_shape[0] * self.src_shape[3]) or \
                (4 * h_align != self.src_shape[1] * self.src_shape[2]):
            raise RuntimeError("the src or dst shape is wrong")


    def zn_2_hwcn_lstm_operator(self):
        """
        zn_2_hwcn_lstm operation

        Parameters
        ----------
        None

        Returns:
        ----------
        tik_instance: tik instance
        """
        with self.tik_instance.for_range(
                0, self.core_num,
                block_num=self.core_num) as core_loop_index:
            with self.tik_instance.if_scope(
                    core_loop_index != self.core_num - 1):
                self.init_ub_tensor(self.each_core_data, self.c_each_core)
                self.core_loop_index.set_as(core_loop_index)
                self.cal_loop(self.each_core_data, self.c_each_core)
            with self.tik_instance.else_scope():
                self.init_ub_tensor(self.last_core_data, self.c_last_core)
                self.core_loop_index.set_as(core_loop_index)
                self.cal_loop(self.last_core_data, self.c_last_core)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=self.src_gm,
            outputs=self.dst_gm)

        return self.tik_instance


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR,
                 REQUIRED_ATTR_STR, KERNEL_NAME)
def zn_2_hwcn_lstm(src, dst, src_format, dst_format,
                   kernel_name="zn_2_hwcn_lstm"):
    """
    algorithm: zn_2_hwcn_lstm

    Parameters
    ----------
    src : dict, shape and dtype of input.
    dst: dict, shape and dtype of input.
    src_format: str, source data format, can be fractal_zn.
    dst_format: str, target data format, can be hwcn.
    kernel_name: str, kernel name, default value is "zn_2_hwcn_lstm".

    Returns
    -------
    None
    """
    zn_2_hwcn_lstm = ZN2HWCNLSTM(src, dst, src_format,
                                 dst_format, kernel_name)
    zn_2_hwcn_lstm.zn_2_hwcn_lstm_operator()
