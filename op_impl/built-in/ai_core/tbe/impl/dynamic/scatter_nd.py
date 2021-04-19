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
scatter nd
"""
from te import tvm
import te.lang.dynamic
from functools import reduce as reduceIns
from te import tik
from te import platform as cce
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.error_manager import error_manager_vector

MAX_INPUT_SIZE = 2**31 - 1
MAX_SHAPE = 2**31 - 1

MAX_CLOUD_UPDATES_UB = 474 * 128
MAX_UB_CORE_INDICES = 474
MAX_ALIGN_NUMBER = 128
MAX_INDICES_BURST_LEN = 60

MAX_FP32_INT32_MINI_INDICES_UB = 50 * 128
MAX_FP32_INT32_MINI_UPDATES_UB = 220 * 128 // 2
MAX_FP32_INT32_MINI_SHAPE_UB = 220 * 128


class ScatterNd():
    def __init__(self, indices, x, shape, y, kernel_name):
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_dtype = x.get("dtype").lower()
        self.shape_dtype = shape.get("dtype").lower()
        self.y_dtype = y.get("dtype").lower()
        indices_support_dtype_list = ("int32", )
        check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        updates_support_dtype_list = ("float32", )
        check_dtype(self.updates_dtype, updates_support_dtype_list, param_name="updates")
        shape_support_dtype_list = ("int32", )
        check_dtype(self.shape_dtype, shape_support_dtype_list, param_name="shape")
        if self.y_dtype != self.updates_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "y", "x",
                                                                  self.y_dtype, self.updates_dtype)
        self.tiling_dtype = "int32"
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.core_start = self.tik_instance.Scalar("int32")
        self.core_end = self.tik_instance.Scalar("int32")
        self.var_read_index = self.tik_instance.Scalar("int32")
        self.updates_read_index = self.tik_instance.Scalar("int32")
        self.indices_var = self.tik_instance.Scalar("int32")
        self.block_idx = self.tik_instance.Scalar("int32")
        self.zero_var = self.tik_instance.Scalar(self.updates_dtype)
        self.zero_var.set_as(0)
        self.var_ub = None
        self.indices_ub = None
        self.updates_ub = None
        self.shape_ub = None
        self.updates_ub_one = None
        self.indices_ub_one = None
        self.cur_var = self.tik_instance.Scalar(dtype=self.updates_dtype)
        self.cur_update = self.tik_instance.Scalar(dtype=self.updates_dtype)
        self.acc_var = self.tik_instance.Scalar(dtype=self.updates_dtype)
        self.updates_var = self.tik_instance.Scalar(dtype=self.updates_dtype)
        self.aicore_num = self._tik_get_core_num()
        self.ub_size = self._tik_get_ub_size()
        self.tbe_product = self._tik_get_platform()
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (32,), name="tiling_gm", scope=tik.scope_gm)
        self.input_indices = self.tik_instance.Tensor(self.indices_dtype, (MAX_INPUT_SIZE, ), name="input_indices",
                                                      scope=tik.scope_gm)
        self.input_updates = self.tik_instance.Tensor(self.updates_dtype, (MAX_INPUT_SIZE, ), name="input_updates",
                                                      scope=tik.scope_gm)
        self.input_shape = self.tik_instance.Tensor(self.indices_dtype, (MAX_SHAPE, ), name="input_shape",
                                                    scope=tik.scope_gm)
        #check platform
        if self.updates_dtype == "float32" and self.tbe_product in ("Ascend910", "Ascend610"):
            self.output_var = self.tik_instance.Tensor(self.updates_dtype, (MAX_SHAPE, ), name="output_var",
                                                       scope=tik.scope_gm, is_atomic_add=True)
        else:
            self.output_var = self.tik_instance.Tensor(self.updates_dtype, (MAX_SHAPE, ), name="output_var",
                                                       scope=tik.scope_gm)

    def _tik_get_core_num(self):
        """
        get core num
        Parameters
        ----------
        Returns
        ----------
        aicore num
        """
        return tik.Dprofile().get_aicore_num()

    def _tik_get_ub_size(self):
        """
        get up size byte
        Parameters
        ----------
        Returns
        ----------
        ub_size
        """
        ub_size = tik.Dprofile().get_unified_buffer_size()
        return ub_size

    def _tik_get_platform(self):
        """
        get paltform
        Parameters
        ----------
        Returns
        ----------
        tbe_product
        """
        tbe_product = cce.cce_conf.get_soc_spec("SOC_VERSION")
        return tbe_product

    def scatter_nd_compute(self):
        self.scatter_nd_compute_tiling_mode()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_indices, self.input_updates, self.input_shape],
                                   outputs=[self.output_var], flowtable=[self.tiling_gm])

    def init_ub_tensor(self):
        self.select_mode = self.tik_instance.Scalar(dtype="int32", name="select_mode")
        self.select_params = self.tik_instance.Scalar(dtype="int32", name="select_params")
        self.core_num = self.tik_instance.Scalar(dtype="int32", name="core_num")
        self.one_core_data = self.tik_instance.Scalar(dtype="int32", name="one_core_data")
        self.last_core_data_num = self.tik_instance.Scalar(dtype="int32", name="last_core_data_num")
        self.block_number = self.tik_instance.Scalar(dtype="int32", name="block_number")
        self.indices_num_one_burst_len = self.tik_instance.Scalar(dtype="int32", name="indices_num_one_burst_len")
        self.updates_num_one_burst_len = self.tik_instance.Scalar(dtype="int32", name="updates_num_one_burst_len")
        self.updates_data_num = self.tik_instance.Scalar(dtype="int32", name="updates_data_num")
        self.updates_burst_fact_len = self.tik_instance.Scalar(dtype="int32", name="updates_burst_fact_len")
        self.indices_num = self.tik_instance.Scalar(dtype="int32", name="indices_num")
        self.tail_indices_burst_len = self.tik_instance.Scalar(dtype="int32", name="tail_indices_burst_len")
        self.tail_updates_burst_len = self.tik_instance.Scalar(dtype="int32", name="tail_updates_burst_len")
        self.tail_updates_can_div = self.tik_instance.Scalar(dtype="int32", name="tail_updates_can_div")
        self.tail_indices_num_burst_len = self.tik_instance.Scalar(dtype="int32", name="tail_indices_num_burst_len")
        self.tail_indices_more_than_burst_len = self.tik_instance.Scalar(dtype="int32", name="tail_indices_more_than_burst_len")
        self.select_align_params = self.tik_instance.Scalar(dtype="int32", name="select_align_params")
        self.max_align_updates_data_num = self.tik_instance.Scalar(dtype="int32", name="max_align_updates_data_num")

        self.select_mode.set_as(self.tiling_ub[0])
        self.select_params.set_as(self.tiling_ub[1])
        self.indices_num.set_as(self.tiling_ub[2])
        self.core_num.set_as(self.tiling_ub[3])
        self.one_core_data.set_as(self.tiling_ub[4])
        self.last_core_data_num.set_as(self.tiling_ub[5])
        self.block_number.set_as(self.tiling_ub[6])
        self.indices_num_one_burst_len.set_as(self.tiling_ub[7])
        self.updates_num_one_burst_len.set_as(self.tiling_ub[8])
        self.updates_data_num.set_as(self.tiling_ub[9])
        self.updates_burst_fact_len.set_as(self.tiling_ub[10])
        self.tail_indices_burst_len.set_as(self.tiling_ub[11])
        self.tail_updates_burst_len.set_as(self.tiling_ub[12])
        self.tail_updates_can_div.set_as(self.tiling_ub[13])
        self.tail_indices_num_burst_len.set_as(self.tiling_ub[14])
        self.tail_indices_more_than_burst_len.set_as(self.tiling_ub[15])
        self.select_align_params.set_as(self.tiling_ub[16])
        self.max_align_updates_data_num.set_as(self.tiling_ub[17])


    def zero_var_ub(self, data, updates_ub):
        dup_len = 128
        if self.updates_dtype in ("float32", "int32"):
            dup_len = 64
        elif self.updates_dtype in ("int8", "uint8"):
            dup_len = 256
        repeat_one = data // dup_len
        repeat = self.tik_instance.Scalar("int32")
        repeat.set_as(repeat_one)
        remain_one = data % dup_len
        remain = self.tik_instance.Scalar("int32")
        remain.set_as(remain_one)
        with self.tik_instance.if_scope(repeat > 255):
            loop_index = repeat // 255
            repeat_remain = data - loop_index * 255 * dup_len
            with self.tik_instance.for_range(0, loop_index) as loop:
                self.tik_instance.vector_dup(dup_len,
                                             updates_ub[loop * 255 * dup_len],
                                             self.zero_var, 255, 1, 8, 0)
            with self.tik_instance.if_scope(repeat_remain != 0):
                repeat_time = repeat_remain // dup_len
                repeat_tail = repeat_remain % dup_len
                with self.tik_instance.if_scope(repeat_time > 0):
                    self.tik_instance.vector_dup(dup_len,
                                                 updates_ub[loop_index * 255 * dup_len],
                                                 self.zero_var, repeat_time,
                                                 1, 8, 0)
                with self.tik_instance.if_scope(repeat_tail > 0):
                    self.tik_instance.vector_dup(repeat_tail,
                                                 updates_ub[loop_index * 255 * dup_len + repeat_time * dup_len],
                                                 self.zero_var, 1,
                                                 1, 8, 0)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(repeat > 0):
                self.tik_instance.vector_dup(dup_len, updates_ub, self.zero_var,
                                             repeat, 1, 8, 0)
            with self.tik_instance.if_scope(remain > 0):
                self.tik_instance.vector_dup(remain, updates_ub[repeat * dup_len],
                                             self.zero_var, 1, 1, 8, 0)

    def align_compute_output(self, data):
        with self.tik_instance.for_range(0, data) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            self.var_read_index.set_as(self.var_read_index * self.updates_data_num)
            self.updates_read_index.set_as(indices_ub_index * self.updates_data_num)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(self.output_var[self.var_read_index], self.updates_ub[self.updates_read_index],
                                        0, 1, self.updates_burst_fact_len, 0, 0)
            self.tik_instance.set_atomic_add(0)

    def align_compute_more_than_output(self, data, loop_index):
        with self.tik_instance.for_range(0, data) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            self.var_read_index.set_as(self.var_read_index * self.updates_data_num)
            self.updates_read_index.set_as(self.block_idx * self.one_core_data * self.updates_data_num +
                                           loop_index * MAX_UB_CORE_INDICES * self.updates_data_num +
                                           indices_ub_index * self.updates_data_num)
            with self.tik_instance.if_scope(self.updates_data_num <= MAX_CLOUD_UPDATES_UB):
                self.tik_instance.data_move(self.updates_ub, self.input_updates[self.updates_read_index],
                                            0, 1, self.updates_burst_fact_len, 0, 0)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.output_var[self.var_read_index], self.updates_ub,
                                            0, 1, self.updates_burst_fact_len, 0, 0)
                self.tik_instance.set_atomic_add(0)
            with self.tik_instance.if_scope(self.updates_data_num > MAX_CLOUD_UPDATES_UB):
                loop_index_one = self.updates_data_num // MAX_CLOUD_UPDATES_UB
                tail_loop_num = self.updates_data_num % MAX_CLOUD_UPDATES_UB
                with self.tik_instance.for_range(0, loop_index_one) as idx:
                    self.tik_instance.data_move(self.updates_ub, self.input_updates[self.updates_read_index + idx *
                                                                                    MAX_CLOUD_UPDATES_UB],
                                                0, 1, MAX_CLOUD_UPDATES_UB // self.block_number, 0, 0)
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move(self.output_var[self.var_read_index + idx * MAX_CLOUD_UPDATES_UB],
                                                self.updates_ub, 0, 1, MAX_CLOUD_UPDATES_UB // self.block_number, 0, 0)
                    self.tik_instance.set_atomic_add(0)
                with self.tik_instance.if_scope(tail_loop_num != 0):
                    self.tik_instance.data_move(self.updates_ub, self.input_updates[self.updates_read_index +
                                                                                    loop_index_one *
                                                                                    MAX_CLOUD_UPDATES_UB],
                                                0, 1, tail_loop_num // self.block_number, 0, 0)
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move(self.output_var[self.var_read_index + loop_index_one *
                                                                MAX_CLOUD_UPDATES_UB], self.updates_ub,
                                                0, 1, tail_loop_num // self.block_number, 0, 0)
                    self.tik_instance.set_atomic_add(0)

    def no_align_less_than_compute_output(self, one_core_data, loop_index):
        mid_compute = self.updates_data_num - self.max_align_updates_data_num
        with self.tik_instance.for_range(0, one_core_data) as indices_ub_index:
            self.updates_read_index.set_as(indices_ub_index * self.updates_data_num + self.block_idx *
                                           self.one_core_data * self.updates_data_num + loop_index *
                                           MAX_UB_CORE_INDICES * self.updates_data_num)
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            self.var_read_index.set_as(self.var_read_index * self.updates_data_num)
            with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                self.tik_instance.data_move(self.updates_ub, self.input_updates[self.updates_read_index],
                                            0, 1, 1, 0, 0)
                self.tik_instance.vadd(self.updates_data_num, self.updates_ub_two, self.updates_ub_one,
                                       self.updates_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.output_var[self.var_read_index], self.updates_ub_two, 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)
            with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                with self.tik_instance.if_scope(self.updates_data_num < MAX_CLOUD_UPDATES_UB):
                    self.tik_instance.data_move(self.updates_ub, self.input_updates[self.updates_read_index],
                                                0, 1, self.updates_burst_fact_len, 0, 0)
                    self.tik_instance.vadd(mid_compute, self.updates_ub_two, self.updates_ub_one,
                                           self.updates_ub[self.max_align_updates_data_num], 1, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move(self.output_var[self.var_read_index], self.updates_ub,
                                                0, 1, self.max_align_updates_data_num // self.block_number, 0, 0)
                    self.tik_instance.data_move(self.output_var[self.var_read_index + self.max_align_updates_data_num],
                                                self.updates_ub_two, 0, 1, MAX_ALIGN_NUMBER // self.block_number, 0, 0)
                    self.tik_instance.set_atomic_add(0)
                with self.tik_instance.if_scope(self.updates_data_num > MAX_CLOUD_UPDATES_UB):
                    updates_loop_index = self.updates_data_num // MAX_CLOUD_UPDATES_UB
                    tail_loop_updates_num = self.updates_data_num % MAX_CLOUD_UPDATES_UB
                    with self.tik_instance.for_range(0, updates_loop_index) as idx:
                        self.tik_instance.data_move(self.updates_ub, self.input_updates[self.updates_read_index + idx
                                                                                        * MAX_CLOUD_UPDATES_UB],
                                                    0, 1, MAX_CLOUD_UPDATES_UB // self.block_number, 0, 0)
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(self.output_var[self.var_read_index + idx * MAX_CLOUD_UPDATES_UB],
                                                    self.updates_ub, 0, 1,
                                                    MAX_CLOUD_UPDATES_UB // self.block_number, 0, 0)
                        self.tik_instance.set_atomic_add(0)
                    with self.tik_instance.if_scope(tail_loop_updates_num != 0):
                        vadd_number = tail_loop_updates_num - self.tail_updates_can_div
                        self.tik_instance.data_move(self.updates_ub, self.input_updates[self.updates_read_index +
                                                                                        updates_loop_index *
                                                                                        MAX_CLOUD_UPDATES_UB],
                                                    0, 1, self.tail_updates_burst_len, 0, 0)
                        self.tik_instance.vadd(vadd_number, self.updates_ub_two, self.updates_ub_one,
                                               self.updates_ub[self.tail_updates_can_div], 1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(self.output_var[self.var_read_index + updates_loop_index *
                                                                    MAX_CLOUD_UPDATES_UB], self.updates_ub,
                                                    0, 1, self.tail_updates_can_div // self.block_number, 0, 0)
                        self.tik_instance.data_move(self.output_var[self.var_read_index + updates_loop_index *
                                                                    MAX_CLOUD_UPDATES_UB + self.tail_updates_can_div],
                                                    self.updates_ub_two,
                                                    0, 1, MAX_ALIGN_NUMBER // self.block_number, 0, 0)
                        self.tik_instance.set_atomic_add(0)

    def process_indices_less_than_thirty_two_byte(self):
        updates_burst_len_32_byte = self.indices_num * self.updates_data_num
        self.tik_instance.data_move(self.indices_ub, self.input_indices,
                                    0, 1, self.indices_num_one_burst_len, 0, 0)
        with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
            self.no_align_less_than_compute_output(self.indices_num, 0)
        with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
            with self.tik_instance.if_scope(self.select_align_params == 0):
                with self.tik_instance.if_scope(self.updates_data_num * self.indices_num <= MAX_CLOUD_UPDATES_UB):
                    self.tik_instance.data_move(self.updates_ub, self.input_updates,
                                                0, 1, updates_burst_len_32_byte // self.block_number, 0, 0)
                    self.align_compute_output(self.indices_num)
                with self.tik_instance.else_scope():
                    self.align_compute_more_than_output(self.indices_num, 0)
            with self.tik_instance.if_scope(self.select_align_params == 1):
                self.no_align_less_than_compute_output(self.indices_num, 0)

    def process_shape_pre_core_less_than_data(self):
        self.tik_instance.data_move(self.indices_ub, self.input_indices[self.block_idx * self.one_core_data],
                                    0, 1, self.indices_num_one_burst_len, 0, 0)
        with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
            self.no_align_less_than_compute_output(self.one_core_data, 0)
        with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
            with self.tik_instance.if_scope(self.select_align_params == 0):
                with self.tik_instance.if_scope(self.updates_data_num * self.one_core_data <= MAX_CLOUD_UPDATES_UB):
                    self.tik_instance.data_move(self.updates_ub, self.input_updates[self.block_idx * self.one_core_data
                                                                                    * self.updates_data_num],
                                                0, 1, self.updates_num_one_burst_len, 0, 0)
                    self.align_compute_output(self.one_core_data)
                with self.tik_instance.else_scope():
                    self.align_compute_more_than_output(self.one_core_data, 0)
            with self.tik_instance.if_scope(self.select_align_params == 1):
                self.no_align_less_than_compute_output(self.one_core_data, 0)

    def process_shape_last_core_less_than_data(self):
        one_core_data_num = self.one_core_data
        with self.tik_instance.if_scope(self.last_core_data_num <= MAX_UB_CORE_INDICES):
            self.tik_instance.data_move(self.indices_ub, self.input_indices[self.block_idx * one_core_data_num], 0, 1,
                                        self.tail_indices_burst_len, 0, 0)
            with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                self.no_align_less_than_compute_output(self.last_core_data_num, 0)
            with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                with self.tik_instance.if_scope(self.select_align_params == 0):
                    with self.tik_instance.if_scope(self.updates_data_num *
                                                    self.last_core_data_num <= MAX_CLOUD_UPDATES_UB):
                        self.tik_instance.data_move(self.updates_ub, self.input_updates[self.block_idx *
                                                                                        one_core_data_num *
                                                                                        self.updates_data_num],
                                                    0, 1, self.tail_updates_burst_len, 0, 0)
                        self.align_compute_output(self.last_core_data_num)
                    with self.tik_instance.else_scope():
                        self.align_compute_more_than_output(self.last_core_data_num, 0)
                with self.tik_instance.if_scope(self.select_align_params == 1):
                    self.no_align_less_than_compute_output(self.last_core_data_num, 0)
        with self.tik_instance.if_scope(self.last_core_data_num > MAX_UB_CORE_INDICES):
            loop_index = self.last_core_data_num // MAX_UB_CORE_INDICES
            tail_indices_num = self.last_core_data_num % MAX_UB_CORE_INDICES
            with self.tik_instance.for_range(0, loop_index) as loop_ub_index:
                self.tik_instance.data_move(self.indices_ub, self.input_indices[self.block_idx * one_core_data_num +
                                                                                loop_ub_index * MAX_UB_CORE_INDICES],
                                            0, 1, MAX_INDICES_BURST_LEN, 0, 0)
                with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                    self.no_align_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
                with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                    with self.tik_instance.if_scope(self.select_align_params == 0):
                        with self.tik_instance.if_scope(self.updates_data_num * MAX_UB_CORE_INDICES <=
                                                        MAX_CLOUD_UPDATES_UB):
                            self.tik_instance.data_move(self.updates_ub,
                                                        self.input_updates[self.block_idx *
                                                                           one_core_data_num * self.updates_data_num
                                                                           + loop_ub_index * MAX_UB_CORE_INDICES
                                                                           * self.updates_data_num],
                                                        0, 1, MAX_UB_CORE_INDICES * self.updates_data_num / self.block_number, 0, 0)
                            self.align_compute_output(MAX_UB_CORE_INDICES)
                        with self.tik_instance.else_scope():
                            self.align_compute_more_than_output(MAX_UB_CORE_INDICES, loop_ub_index)
                    with self.tik_instance.if_scope(self.select_align_params == 1):
                        self.no_align_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
            with self.tik_instance.if_scope(tail_indices_num != 0):
                self.tik_instance.data_move(self.indices_ub, self.input_indices[self.block_idx *
                                                                                one_core_data_num +
                                                                                loop_index * MAX_UB_CORE_INDICES],
                                            0, 1, self.tail_indices_num_burst_len, 0, 0)
                with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                    self.no_align_less_than_compute_output(tail_indices_num, loop_index)
                with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                    with self.tik_instance.if_scope(self.select_align_params == 0):
                        with self.tik_instance.if_scope(self.updates_data_num * tail_indices_num <=
                                                        MAX_CLOUD_UPDATES_UB):
                            self.tik_instance.data_move(self.updates_ub,
                                                        self.input_updates[self.block_idx * one_core_data_num *
                                                                           self.updates_data_num + loop_index *
                                                                           MAX_UB_CORE_INDICES * self.updates_data_num],
                                                        0, 1, tail_indices_num * self.updates_data_num // self.block_number, 0, 0)
                            self.align_compute_output(tail_indices_num)
                        with self.tik_instance.else_scope():
                            self.align_compute_more_than_output(tail_indices_num, loop_index)
                    with self.tik_instance.if_scope(self.select_align_params == 1):
                        self.no_align_less_than_compute_output(tail_indices_num, loop_index)

    def traversing_shape_more_than_pre_core(self):
        loop_index = self.one_core_data // MAX_UB_CORE_INDICES
        tail_indices_number = self.one_core_data % MAX_UB_CORE_INDICES
        with self.tik_instance.for_range(0, loop_index) as loop_ub_index:
            self.tik_instance.data_move(self.indices_ub, self.input_indices[self.block_idx * self.one_core_data
                                                                            + loop_ub_index * MAX_UB_CORE_INDICES],
                                        0, 1, MAX_INDICES_BURST_LEN, 0, 0)
            with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                self.no_align_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)

            with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                with self.tik_instance.if_scope(self.select_align_params == 0):
                    with self.tik_instance.if_scope(self.updates_data_num <= MAX_ALIGN_NUMBER):
                        self.tik_instance.data_move(self.updates_ub,
                                                    self.input_updates[self.block_idx * self.one_core_data *
                                                                       self.updates_data_num + loop_ub_index *
                                                                       MAX_UB_CORE_INDICES * self.updates_data_num],
                                                    0, 1, MAX_UB_CORE_INDICES * self.updates_data_num // self.block_number, 0, 0)
                        self.align_compute_output(MAX_UB_CORE_INDICES)
                    with self.tik_instance.if_scope(self.updates_data_num > MAX_ALIGN_NUMBER):
                        self.align_compute_more_than_output(MAX_UB_CORE_INDICES, loop_ub_index)
                with self.tik_instance.if_scope(self.select_align_params == 1):
                    self.no_align_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
        with self.tik_instance.if_scope(tail_indices_number != 0):
            self.tik_instance.data_move(self.indices_ub, self.input_indices[self.block_idx * self.one_core_data
                                                                            + loop_index * MAX_UB_CORE_INDICES],
                                        0, 1, self.tail_indices_burst_len, 0, 0)
            with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                self.no_align_less_than_compute_output(tail_indices_number, loop_index)
            with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                with self.tik_instance.if_scope(self.select_align_params == 0):
                    with self.tik_instance.if_scope(self.updates_data_num * tail_indices_number <=
                                                    MAX_CLOUD_UPDATES_UB):
                        self.tik_instance.data_move(self.updates_ub,
                                                    self.input_updates[self.block_idx * self.one_core_data *
                                                                       self.updates_data_num + loop_index *
                                                                       MAX_UB_CORE_INDICES * self.updates_data_num],
                                                    0, 1, tail_indices_number * self.updates_data_num // self.block_number, 0, 0)
                        self.align_compute_output(tail_indices_number)
                    with self.tik_instance.else_scope():
                        self.align_compute_more_than_output(tail_indices_number, loop_index)
                with self.tik_instance.if_scope(self.select_align_params == 1):
                    self.no_align_less_than_compute_output(tail_indices_number, loop_index)

    def traversing_shape_more_than_last_core(self):
        standard_number = MAX_CLOUD_UPDATES_UB // MAX_UB_CORE_INDICES
        with self.tik_instance.if_scope(self.last_core_data_num <= MAX_UB_CORE_INDICES):
            self.tik_instance.data_move(self.indices_ub, self.input_indices[self.block_idx * self.one_core_data],
                                        0, 1, self.tail_indices_burst_len, 0, 0)
            with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                self.no_align_less_than_compute_output(self.last_core_data_num, 0)
            with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                with self.tik_instance.if_scope(self.select_align_params == 0):
                    with self.tik_instance.if_scope(self.updates_data_num *
                                                    self.last_core_data_num <= MAX_CLOUD_UPDATES_UB):
                        self.tik_instance.data_move(self.updates_ub,
                                                    self.input_updates[self.block_idx * self.one_core_data
                                                                       * self.updates_data_num],
                                                    0, 1, self.tail_updates_burst_len, 0, 0)
                        self.align_compute_output(self.last_core_data_num)
                    with self.tik_instance.else_scope():
                        self.align_compute_more_than_output(self.last_core_data_num, 0)
                with self.tik_instance.if_scope(self.select_align_params == 1):
                    self.no_align_less_than_compute_output(self.last_core_data_num, 0)
        with self.tik_instance.if_scope(self.last_core_data_num > MAX_UB_CORE_INDICES):
            last_core_data_num = self.last_core_data_num
            loop_index = last_core_data_num // MAX_UB_CORE_INDICES
            tail_updates_number = last_core_data_num % MAX_UB_CORE_INDICES
            with self.tik_instance.for_range(0, loop_index) as loop_ub_index:
                self.tik_instance.data_move(self.indices_ub, self.input_indices[self.block_idx * self.one_core_data
                                                                                + loop_ub_index * MAX_UB_CORE_INDICES],
                                            0, 1, MAX_INDICES_BURST_LEN, 0, 0)
                with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                    self.no_align_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
                with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                    with self.tik_instance.if_scope(self.select_align_params == 0):
                        with self.tik_instance.if_scope(self.updates_data_num <= standard_number):
                            self.tik_instance.data_move(self.updates_ub,
                                                        self.input_updates[self.block_idx * self.one_core_data *
                                                                           self.updates_data_num + loop_ub_index *
                                                                           MAX_UB_CORE_INDICES * self.updates_data_num],
                                                        0, 1, MAX_UB_CORE_INDICES * self.updates_data_num // self.block_number, 0, 0)
                            self.align_compute_output(MAX_UB_CORE_INDICES)
                        with self.tik_instance.if_scope(self.updates_data_num > standard_number):
                            self.align_compute_more_than_output(MAX_UB_CORE_INDICES, loop_ub_index)
                    with self.tik_instance.if_scope(self.select_align_params == 1):
                        self.no_align_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
            with self.tik_instance.if_scope(tail_updates_number != 0):
                self.tik_instance.data_move(self.indices_ub, self.input_indices[self.block_idx * self.one_core_data
                                                                                + loop_index * MAX_UB_CORE_INDICES],
                                            0, 1, self.tail_indices_more_than_burst_len, 0, 0)
                with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                    self.no_align_less_than_compute_output(tail_updates_number, loop_index)
                with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                    with self.tik_instance.if_scope(self.select_align_params == 0):
                        with self.tik_instance.if_scope(self.updates_data_num *
                                                        tail_updates_number <= MAX_CLOUD_UPDATES_UB):
                            self.tik_instance.data_move(self.updates_ub,
                                                        self.input_updates[self.block_idx * self.one_core_data *
                                                                           self.updates_data_num + loop_index *
                                                                           MAX_UB_CORE_INDICES * self.updates_data_num],
                                                        0, 1, tail_updates_number * self.updates_data_num // self.block_number, 0, 0)
                            self.align_compute_output(tail_updates_number)
                        with self.tik_instance.else_scope():
                            self.align_compute_more_than_output(tail_updates_number, loop_index)
                    with self.tik_instance.if_scope(self.select_align_params == 1):
                        self.no_align_less_than_compute_output(tail_updates_number, loop_index)

    def scatter_nd_compute_tiling_mode(self):
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as block_idx:
            self.tiling_ub = self.tik_instance.Tensor("int32", (32,), name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
            self.init_ub_tensor()
            with self.tik_instance.if_scope(self.select_mode == 1):
                self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (MAX_UB_CORE_INDICES, ), name="indices_ub",
                                                          scope=tik.scope_ubuf)
                self.updates_ub = self.tik_instance.Tensor(self.updates_dtype, (MAX_CLOUD_UPDATES_UB, ), name="updates_ub",
                                                           scope=tik.scope_ubuf)
                self.updates_ub_one = self.tik_instance.Tensor(self.updates_dtype, (128, ), name="updates_ub_one",
                                                               scope=tik.scope_ubuf)
                self.updates_ub_two = self.tik_instance.Tensor(self.updates_dtype, (128, ), name="updates_ub_two",
                                                               scope=tik.scope_ubuf)
                self.zero_var_ub(128, self.updates_ub_one)
                self.zero_var_ub(128, self.updates_ub_two)
                self.block_idx.set_as(block_idx)
                with self.tik_instance.if_scope(self.select_params == 0):
                    with self.tik_instance.if_scope(self.core_num > 1):
                        with self.tik_instance.if_scope(block_idx < self.core_num - 1):
                            self.process_shape_pre_core_less_than_data()
                        with self.tik_instance.if_scope(block_idx == self.core_num - 1):
                            self.process_shape_last_core_less_than_data()
                    with self.tik_instance.if_scope(self.core_num == 1):
                        with self.tik_instance.if_scope(block_idx == self.core_num - 1):
                            self.process_indices_less_than_thirty_two_byte()
                with self.tik_instance.if_scope(self.select_params == 1):
                    with self.tik_instance.if_scope(self.core_num > 1):
                        with self.tik_instance.if_scope(block_idx < self.core_num - 1):
                            self.traversing_shape_more_than_pre_core()
                        with self.tik_instance.if_scope(block_idx == self.core_num - 1):
                            self.traversing_shape_more_than_last_core()


# pylint: disable=unused-argument,invalid-name,too-many-locals
@te.op.register_operator("ScatterNd")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def scatter_nd(indices, x, shape, y, kernel_name="ScatterNd"):
    """
        scatter_nd interface

        Parameters
        ----------
        indices_dict: input indices shape, dtype and range
        x_dict: input updates shape, dtype and range
        shape dict: input shape shape, dtype and range
        y_dict: output shape, dtype and range
        kernel_name: kernel name of scatter_nd op

        Returns
        -------
        compile info
        """
    obj = ScatterNd(indices, x, shape, y, kernel_name)
    obj.scatter_nd_compute()
    te.op.add_compile_info("vars", {"ub_size": obj.ub_size,
                                    "core_num": obj.aicore_num})



