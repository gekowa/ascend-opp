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
scatter add
"""
from te import tik
from te import platform as cce
import sys
import te.lang.dynamic
from te import tvm
from topi import generic
from functools import reduce as reduceIns
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.error_manager import error_manager_vector

MAX_ZERO_DIM_INDICE = 2**31 - 1
MAX_ZERO_DIM_VAR = 2**31 - 1

MAX_UB_INDICES = 474
MAX_UB_UPDATES = 474 * 128
MAX_UB_CORE_INDICES = 474
MAX_ALIGN_NUMBER = 128
MAX_INDICES_BURST_LEN = 60


class Scatter():
    def __init__(self, var, indices, updates, var_out, use_locking, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.indicesdtype = indices.get("dtype").lower()
        self.updatesdtype = updates.get("dtype").lower()
        self.vardtype = var.get("dtype").lower()
        self.var_out_dtype = var_out.get("dtype").lower()
        indices_support_dtype_list = ("int32", )
        check_dtype(self.indicesdtype, indices_support_dtype_list, param_name="indices")
        updates_support_dtype_list = ("float32",)
        check_dtype(self.updatesdtype, updates_support_dtype_list, param_name="updates")
        self.tiling_dtype = "int32"
        if self.updatesdtype != self.vardtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "updates", "var",
                                                                  self.updatesdtype, self.vardtype)
        if self.vardtype != self.var_out_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "var_out", "var",
                                                                  self.var_out_dtype, self.vardtype)
        self.kernel_name = kernel_name
        self.var_read_index = self.tik_instance.Scalar("int32")
        self.updates_read_index = self.tik_instance.Scalar("int32")
        self.indices_loop_index = self.tik_instance.Scalar("int32")
        self.zero_var = self.tik_instance.Scalar(dtype=self.updatesdtype, name="zero_var")
        self.zero_var.set_as(0)
        self.indices_ub = None
        self.updates_ub = None
        self.core_num = self._tik_get_core_num()
        self.ub_size = self._tik_get_ub_size()

        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (32, ), name="tiling_gm", scope=tik.scope_gm)
        self.input_var = self.tik_instance.Tensor(self.updatesdtype, (MAX_ZERO_DIM_VAR, ), name="input_var",
                                                  scope=tik.scope_gm)
        self.input_indices = self.tik_instance.Tensor(self.indicesdtype, (MAX_ZERO_DIM_INDICE, ), name="input_indices",
                                                      scope=tik.scope_gm)
        self.input_updates = self.tik_instance.Tensor(self.updatesdtype, (MAX_ZERO_DIM_INDICE, ), name="input_updates",
                                                      scope=tik.scope_gm)
        self.output_var = self.tik_instance.Tensor(self.updatesdtype, (MAX_ZERO_DIM_VAR, ), name="output_var",
                                                   scope=tik.scope_gm)

    def _tik_get_core_num(self):
        """
        get core num
        Returns
        -------
        aicore num
        """
        return tik.Dprofile().get_aicore_num()

    def _tik_get_ub_size(self):
        """
        get up size byte
        Returns
        -------
        ub_size
        """
        ub_size = tik.Dprofile().get_unified_buffer_size()
        return ub_size

    def _zero_updates_ub(self, updates_ub_number, updates_ub):
        dup_len = 128
        if self.updatesdtype in ("float32", "int32"):
            dup_len = 64
        elif self.updatesdtype in ("int8", "uint8"):
            dup_len = 256
        repeat_one = updates_ub_number // dup_len
        repeat = self.tik_instance.Scalar("int32")
        repeat.set_as(repeat_one)
        remain_one = updates_ub_number % dup_len
        remain = self.tik_instance.Scalar("int32")
        remain.set_as(remain_one)
        with self.tik_instance.if_scope(repeat > 0):
            self.tik_instance.vector_dup(dup_len, updates_ub, self.zero_var, repeat, 1, 8, 0)
        with self.tik_instance.if_scope(remain > 0):
            self.tik_instance.vector_dup(remain, updates_ub, self.zero_var, 1, 1, 8, 0)

    def _scatter_compute(self):
        self._scatter_compute_tiling_ub()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_var, self.input_indices, self.input_updates],
                                   outputs=[self.output_var], flowtable=[self.tiling_gm])

        te.op.add_compile_info("vars", {"ub_size": self.ub_size, "core_num": self.core_num})


    def _init_ub_tensor(self):
        self.indices_ub = self.tik_instance.Tensor(self.indicesdtype, (MAX_UB_INDICES, ), name="indices_ub",
                                                   scope=tik.scope_ubuf)
        self.updates_ub = self.tik_instance.Tensor(self.updatesdtype, (MAX_UB_UPDATES, ), name="updates_ub",
                                                   scope=tik.scope_ubuf)
        self.updates_ub_one = self.tik_instance.Tensor(self.updatesdtype, (128, ), name="updates_ub_one",
                                                       scope=tik.scope_ubuf)
        self.updates_ub_two = self.tik_instance.Tensor(self.updatesdtype, (128, ), name="updates_ub_two",
                                                       scope=tik.scope_ubuf)

    def _tiling_args(self):
        self.select_mode = self.tik_instance.Scalar("int32")
        self.select_params = self.tik_instance.Scalar("int32")
        self.final_one_core_data_num = self.tik_instance.Scalar("int32")
        self.last_core_data_num = self.tik_instance.Scalar("int32")
        self.indices_ub_number = self.tik_instance.Scalar("int32")
        self.core_num_one = self.tik_instance.Scalar("int32")
        self.updates_data_num = self.tik_instance.Scalar("int32")
        self.updates_burst_fact_len = self.tik_instance.Scalar("int32")
        self.indices_burst_len = self.tik_instance.Scalar("int32")
        self.updates_burst_len = self.tik_instance.Scalar("int32")
        self.block_number = self.tik_instance.Scalar("int32")
        self.tail_indices_burst_len = self.tik_instance.Scalar("int32")
        self.tail_indices_num_burst_len = self.tik_instance.Scalar("int32")
        self.tail_updates_burst_len = self.tik_instance.Scalar("int32")
        self.tail_indices_more_than_burst_len = self.tik_instance.Scalar("int32")
        self.tail_updates_can_div = self.tik_instance.Scalar("int32")
        self.select_align_params = self.tik_instance.Scalar("int32")
        self.max_align_updates_data_num = self.tik_instance.Scalar("int32")

        self.select_mode.set_as(self.tiling_ub[0])
        self.select_params.set_as(self.tiling_ub[1])
        self.final_one_core_data_num.set_as(self.tiling_ub[2])
        self.last_core_data_num.set_as(self.tiling_ub[3])
        self.indices_ub_number.set_as(self.tiling_ub[4])
        self.core_num_one.set_as(self.tiling_ub[5])
        self.updates_data_num.set_as(self.tiling_ub[6])
        self.updates_burst_fact_len.set_as(self.tiling_ub[7])
        self.indices_burst_len.set_as(self.tiling_ub[8])
        self.updates_burst_len.set_as(self.tiling_ub[9])
        self.block_number.set_as(self.tiling_ub[10])
        self.tail_indices_burst_len.set_as(self.tiling_ub[11])
        self.tail_indices_num_burst_len.set_as(self.tiling_ub[12])
        self.tail_updates_burst_len.set_as(self.tiling_ub[13])
        self.tail_indices_more_than_burst_len.set_as(self.tiling_ub[14])
        self.tail_updates_can_div.set_as(self.tiling_ub[15])
        self.select_align_params.set_as(self.tiling_ub[16])
        self.max_align_updates_data_num.set_as(self.tiling_ub[17])

    def _align_less_than_compute_output(self, one_core_data):
        with self.tik_instance.for_range(0, one_core_data) as indices_index:
            self.var_read_index.set_as(self.indices_ub[indices_index])
            self.var_read_index.set_as(self.var_read_index * self.updates_data_num)
            self.updates_read_index.set_as(indices_index * self.updates_data_num)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(self.input_var[self.var_read_index], self.updates_ub[self.updates_read_index],
                                        0, 1, self.updates_burst_fact_len, 0, 0)
            self.tik_instance.set_atomic_add(0)

    def _unalign_less_than_compute_output(self, one_core_data, loop_index):
        mid_compute = self.updates_data_num - self.max_align_updates_data_num
        with self.tik_instance.for_range(0, one_core_data) as indices_index:
            self.updates_read_index.set_as(indices_index * self.updates_data_num + self.indices_loop_index *
                                           self.final_one_core_data_num * self.updates_data_num + loop_index *
                                           MAX_UB_CORE_INDICES * self.updates_data_num)
            self.var_read_index.set_as(self.indices_ub[indices_index])
            self.var_read_index.set_as(self.var_read_index * self.updates_data_num)
            with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                self.tik_instance.data_move(self.updates_ub, self.input_updates[self.updates_read_index],
                                            0, 1, 1, 0, 0)
                self.tik_instance.vadd(self.updates_data_num, self.updates_ub_two, self.updates_ub_one, self.updates_ub,
                                       1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.input_var[self.var_read_index], self.updates_ub_two, 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)

            with self.tik_instance.if_scope(self.updates_data_num > self.block_number):
                with self.tik_instance.if_scope(self.updates_data_num < MAX_UB_UPDATES):
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
                with self.tik_instance.if_scope(self.updates_data_num > MAX_UB_UPDATES):
                    updates_loop_index = self.updates_data_num // MAX_UB_UPDATES
                    tail_loop_updates_num = self.updates_data_num % MAX_UB_UPDATES
                    with self.tik_instance.for_range(0, updates_loop_index) as idx:
                        self.tik_instance.data_move(self.updates_ub,
                                                    self.input_updates[self.updates_read_index + idx * MAX_UB_UPDATES],
                                                    0, 1, MAX_UB_UPDATES // self.block_number, 0, 0)
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(self.output_var[self.var_read_index + idx * MAX_UB_UPDATES],
                                                    self.updates_ub, 0, 1, MAX_UB_UPDATES // self.block_number, 0, 0)
                        self.tik_instance.set_atomic_add(0)
                    with self.tik_instance.if_scope(tail_loop_updates_num != 0):
                        vadd_number = tail_loop_updates_num - self.tail_updates_can_div
                        self.tik_instance.data_move(self.updates_ub,
                                                    self.input_updates[self.updates_read_index + updates_loop_index *
                                                                       MAX_UB_UPDATES],
                                                    0, 1, self.tail_updates_burst_len, 0, 0)
                        self.tik_instance.vadd(vadd_number, self.updates_ub_two, self.updates_ub_one,
                                               self.updates_ub[self.tail_updates_can_div], 1, 1, 1, 1, 8, 8, 8)
                        self.tik_instance.set_atomic_add(1)
                        self.tik_instance.data_move(self.output_var[self.var_read_index + updates_loop_index *
                                                                    MAX_UB_UPDATES], self.updates_ub,
                                                    0, 1, self.tail_updates_can_div // self.block_number, 0, 0)
                        self.tik_instance.data_move(self.output_var[self.var_read_index + updates_loop_index *
                                                                    MAX_UB_UPDATES + self.tail_updates_can_div],
                                                    self.updates_ub_two,
                                                    0, 1, MAX_ALIGN_NUMBER // self.block_number, 0, 0)
                        self.tik_instance.set_atomic_add(0)

    def _align_more_than_compute_output(self, one_core_data, loop_index):
        with self.tik_instance.for_range(0, one_core_data) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            self.var_read_index.set_as(self.var_read_index * self.updates_data_num)
            self.updates_read_index.set_as(self.indices_loop_index * self.final_one_core_data_num *
                                           self.updates_data_num + indices_ub_index * self.updates_data_num +
                                           loop_index * MAX_UB_CORE_INDICES * self.updates_data_num)
            with self.tik_instance.if_scope(self.updates_data_num <= MAX_UB_UPDATES):
                self.tik_instance.data_move(self.updates_ub, self.input_updates[self.updates_read_index],
                                            0, 1, self.updates_burst_fact_len, 0, 0)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.output_var[self.var_read_index], self.updates_ub,
                                            0, 1, self.updates_burst_fact_len, 0, 0)
                self.tik_instance.set_atomic_add(0)
            with self.tik_instance.if_scope(self.updates_data_num > MAX_UB_UPDATES):
                loop_index_one = self.updates_data_num // MAX_UB_UPDATES
                tail_loop_num = self.updates_data_num % MAX_UB_UPDATES
                with self.tik_instance.for_range(0, loop_index_one) as idx:
                    self.tik_instance.data_move(self.updates_ub,
                                                self.input_updates[self.updates_read_index + idx * MAX_UB_UPDATES],
                                                0, 1, MAX_UB_UPDATES // self.block_number, 0, 0)
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move(self.output_var[self.var_read_index + idx * MAX_UB_UPDATES],
                                                self.updates_ub, 0, 1, MAX_UB_UPDATES // self.block_number, 0, 0)
                    self.tik_instance.set_atomic_add(0)
                with self.tik_instance.if_scope(tail_loop_num != 0):
                    self.tik_instance.data_move(self.updates_ub, self.input_updates[self.updates_read_index +
                                                                                    loop_index_one * MAX_UB_UPDATES],
                                                0, 1, tail_loop_num // self.block_number, 0, 0)
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move(self.output_var[self.var_read_index + loop_index_one * MAX_UB_UPDATES],
                                                self.updates_ub, 0, 1, tail_loop_num // self.block_number, 0, 0)
                    self.tik_instance.set_atomic_add(0)

    def _process_indices_less_than_byte(self):
        updates_burst_len_32_byte = self.indices_ub_number * self.updates_data_num
        self.tik_instance.data_move(self.indices_ub, self.input_indices,
                                    0, 1, self.indices_burst_len, 0, 0)
        with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
            self._zero_updates_ub(128, self.updates_ub_one)
            self._zero_updates_ub(128, self.updates_ub_two)
            self._unalign_less_than_compute_output(self.indices_ub_number, 0)
        with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
            with self.tik_instance.if_scope(self.select_align_params == 0):
                with self.tik_instance.if_scope(self.updates_data_num * self.indices_ub_number <= MAX_UB_UPDATES):
                    self.tik_instance.data_move(self.updates_ub, self.input_updates,
                                                0, 1, updates_burst_len_32_byte // self.block_number, 0, 0)
                    self._align_less_than_compute_output(self.indices_ub_number)
                with self.tik_instance.else_scope():
                    self._align_more_than_compute_output(self.indices_ub_number, 0)
            with self.tik_instance.if_scope(self.select_align_params == 1):
                self._zero_updates_ub(128, self.updates_ub_one)
                self._zero_updates_ub(128, self.updates_ub_two)
                self._unalign_less_than_compute_output(self.indices_ub_number, 0)

    def _process_indices_pre_core_less_than_data(self):
        self.tik_instance.data_move(self.indices_ub,
                                    self.input_indices[self.indices_loop_index * self.final_one_core_data_num],
                                    0, 1, self.indices_burst_len, 0, 0)
        with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
            self._zero_updates_ub(128, self.updates_ub_one)
            self._zero_updates_ub(128, self.updates_ub_two)
            self._unalign_less_than_compute_output(self.final_one_core_data_num, 0)
        with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
            with self.tik_instance.if_scope(self.select_align_params == 0):
                with self.tik_instance.if_scope(self.updates_data_num * self.final_one_core_data_num <= MAX_UB_UPDATES):
                    self.tik_instance.data_move(self.updates_ub,
                                                self.input_updates[self.indices_loop_index *
                                                                   self.final_one_core_data_num * self.updates_data_num],
                                                0, 1, self.updates_burst_len, 0, 0)
                    self._align_less_than_compute_output(self.final_one_core_data_num)
                with self.tik_instance.else_scope():
                    self._align_more_than_compute_output(self.final_one_core_data_num, 0)
            with self.tik_instance.if_scope(self.select_align_params == 1):
                self._zero_updates_ub(128, self.updates_ub_one)
                self._zero_updates_ub(128, self.updates_ub_two)
                self._unalign_less_than_compute_output(self.final_one_core_data_num, 0)

    def _process_indices_last_core_less_than_data(self):
        with self.tik_instance.if_scope(self.last_core_data_num <= MAX_UB_CORE_INDICES):
            self.tik_instance.data_move(self.indices_ub,
                                        self.input_indices[self.indices_loop_index * self.final_one_core_data_num],
                                        0, 1, self.tail_indices_burst_len, 0, 0)
            with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                self._zero_updates_ub(128, self.updates_ub_one)
                self._zero_updates_ub(128, self.updates_ub_two)
                self._unalign_less_than_compute_output(self.last_core_data_num, 0)
            with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                with self.tik_instance.if_scope(self.select_align_params == 0):
                    with self.tik_instance.if_scope(self.updates_data_num * self.last_core_data_num <= MAX_UB_UPDATES):
                        self.tik_instance.data_move(self.updates_ub,
                                                    self.input_updates[self.indices_loop_index *
                                                                       self.final_one_core_data_num *
                                                                       self.updates_data_num],
                                                    0, 1, self.tail_updates_burst_len, 0, 0)
                        self._align_less_than_compute_output(self.last_core_data_num)
                    with self.tik_instance.else_scope():
                        self._align_more_than_compute_output(self.last_core_data_num, 0)
                with self.tik_instance.if_scope(self.select_align_params == 1):
                    self._zero_updates_ub(128, self.updates_ub_one)
                    self._zero_updates_ub(128, self.updates_ub_two)
                    self._unalign_less_than_compute_output(self.last_core_data_num, 0)
        with self.tik_instance.if_scope(self.last_core_data_num > MAX_UB_CORE_INDICES):
            loop_index = self.last_core_data_num // MAX_UB_CORE_INDICES
            tail_indices_num = self.last_core_data_num % MAX_UB_CORE_INDICES
            with self.tik_instance.for_range(0, loop_index) as loop_ub_index:
                self.tik_instance.data_move(self.indices_ub,
                                            self.input_indices[self.indices_loop_index * self.final_one_core_data_num
                                                               + loop_ub_index * MAX_UB_CORE_INDICES],
                                            0, 1, MAX_INDICES_BURST_LEN, 0, 0)
                with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                    self._zero_updates_ub(128, self.updates_ub_one)
                    self._zero_updates_ub(128, self.updates_ub_two)
                    self._unalign_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
                with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                    with self.tik_instance.if_scope(self.select_align_params == 0):
                        with self.tik_instance.if_scope(self.updates_data_num * MAX_UB_CORE_INDICES <= MAX_UB_UPDATES):
                            self.tik_instance.data_move(self.updates_ub,
                                                        self.input_updates[self.indices_loop_index *
                                                                           self.final_one_core_data_num *
                                                                           self.updates_data_num + loop_ub_index *
                                                                           MAX_UB_CORE_INDICES * self.updates_data_num],
                                                        0, 1, MAX_UB_CORE_INDICES * self.updates_data_num // self.block_number, 0, 0)
                            self._align_less_than_compute_output(MAX_UB_CORE_INDICES)
                        with self.tik_instance.else_scope():
                            self._align_more_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
                    with self.tik_instance.if_scope(self.select_align_params == 1):
                        self._zero_updates_ub(128, self.updates_ub_one)
                        self._zero_updates_ub(128, self.updates_ub_two)
                        self._unalign_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
            with self.tik_instance.if_scope(tail_indices_num != 0):
                self.tik_instance.data_move(self.indices_ub,
                                            self.input_indices[self.indices_loop_index * self.final_one_core_data_num
                                                               + loop_index * MAX_UB_CORE_INDICES],
                                            0, 1, self.tail_indices_num_burst_len, 0, 0)
                with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                    self._zero_updates_ub(128, self.updates_ub_one)
                    self._zero_updates_ub(128, self.updates_ub_two)
                    self._unalign_less_than_compute_output(tail_indices_num, loop_index)
                with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                    with self.tik_instance.if_scope(self.select_align_params == 0):
                        with self.tik_instance.if_scope(self.updates_data_num * tail_indices_num <= MAX_UB_UPDATES):
                            self.tik_instance.data_move(self.updates_ub,
                                                        self.input_updates[self.indices_loop_index *
                                                                           self.final_one_core_data_num *
                                                                           self.updates_data_num + loop_index *
                                                                           MAX_UB_CORE_INDICES * self.updates_data_num],
                                                        0, 1, tail_indices_num * self.updates_data_num // self.block_number, 0, 0)
                            self._align_less_than_compute_output(tail_indices_num)
                        with self.tik_instance.else_scope():
                            self._align_more_than_compute_output(tail_indices_num, loop_index)
                    with self.tik_instance.if_scope(self.select_align_params == 1):
                        self._zero_updates_ub(128, self.updates_ub_one)
                        self._zero_updates_ub(128, self.updates_ub_two)
                        self._unalign_less_than_compute_output(tail_indices_num, loop_index)

    def _traversing_indices_more_than_pre_core(self):
        loop_index = self.final_one_core_data_num // MAX_UB_CORE_INDICES
        tail_updates_number = self.final_one_core_data_num % MAX_UB_CORE_INDICES
        with self.tik_instance.for_range(0, loop_index) as loop_ub_index:
            self.tik_instance.data_move(self.indices_ub,
                                        self.input_indices[self.indices_loop_index * self.final_one_core_data_num
                                                           + loop_ub_index * MAX_UB_CORE_INDICES],
                                        0, 1, MAX_INDICES_BURST_LEN, 0, 0)
            #updates_data_num less than block_number
            with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                self._zero_updates_ub(128, self.updates_ub_one)
                self._zero_updates_ub(128, self.updates_ub_two)
                self._unalign_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
            #updates_data_num more than block_number
            with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                with self.tik_instance.if_scope(self.select_align_params == 0):
                    with self.tik_instance.if_scope(self.updates_data_num <= MAX_ALIGN_NUMBER):
                        self.tik_instance.data_move(self.updates_ub,
                                                    self.input_updates[self.indices_loop_index * self.final_one_core_data_num *
                                                                       self.updates_data_num + loop_ub_index *
                                                                       MAX_UB_CORE_INDICES * self.updates_data_num],
                                                    0, 1, MAX_UB_CORE_INDICES * self.updates_data_num // self.block_number, 0, 0)
                        self._align_less_than_compute_output(MAX_UB_CORE_INDICES)
                    with self.tik_instance.if_scope(self.updates_data_num > MAX_ALIGN_NUMBER):
                        self._align_more_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
                with self.tik_instance.if_scope(self.select_align_params == 1):
                    self._zero_updates_ub(128, self.updates_ub_one)
                    self._zero_updates_ub(128, self.updates_ub_two)
                    self._unalign_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
        with self.tik_instance.if_scope(tail_updates_number != 0):
            self.tik_instance.data_move(self.indices_ub,
                                        self.input_indices[self.indices_loop_index * self.final_one_core_data_num + loop_index *
                                                           MAX_UB_CORE_INDICES],
                                        0, 1, self.tail_indices_num_burst_len, 0, 0)
            with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                self._zero_updates_ub(128, self.updates_ub_one)
                self._zero_updates_ub(128, self.updates_ub_two)
                self._unalign_less_than_compute_output(tail_updates_number, loop_index)
            with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                with self.tik_instance.if_scope(self.select_align_params == 0):
                    with self.tik_instance.if_scope(self.updates_data_num * tail_updates_number <= MAX_UB_UPDATES):
                        self.tik_instance.data_move(self.updates_ub,
                                                    self.input_updates[self.indices_loop_index * self.final_one_core_data_num *
                                                                       self.updates_data_num + loop_index *
                                                                       MAX_UB_CORE_INDICES * self.updates_data_num],
                                                    0, 1, tail_updates_number * self.updates_data_num // self.block_number, 0, 0)
                        self._align_less_than_compute_output(tail_updates_number)
                    with self.tik_instance.else_scope():
                        self._align_more_than_compute_output(tail_updates_number, loop_index)
                with self.tik_instance.if_scope(self.select_align_params == 1):
                    self._zero_updates_ub(128, self.updates_ub_one)
                    self._zero_updates_ub(128, self.updates_ub_two)
                    self._unalign_less_than_compute_output(tail_updates_number, loop_index)

    def _traversing_indices_more_than_last_core(self):
        last_core_data_num = self.last_core_data_num
        with self.tik_instance.if_scope(self.last_core_data_num <= MAX_UB_CORE_INDICES):
            self.tik_instance.data_move(self.indices_ub,
                                        self.input_indices[self.indices_loop_index * self.final_one_core_data_num],
                                        0, 1, self.tail_indices_burst_len, 0, 0)
            with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                self._zero_updates_ub(128, self.updates_ub_one)
                self._zero_updates_ub(128, self.updates_ub_two)
                self._unalign_less_than_compute_output(self.last_core_data_num, 0)
            with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                with self.tik_instance.if_scope(self.select_align_params == 0):
                    with self.tik_instance.if_scope(self.updates_data_num * self.last_core_data_num <= MAX_UB_UPDATES):
                        self.tik_instance.data_move(self.updates_ub,
                                                    self.input_updates[self.indices_loop_index *
                                                                       self.final_one_core_data_num *
                                                                       self.updates_data_num],
                                                    0, 1, self.tail_updates_burst_len, 0, 0)
                        self._align_less_than_compute_output(self.last_core_data_num)
                    with self.tik_instance.else_scope():
                        self._align_more_than_compute_output(self.last_core_data_num, 0)
                with self.tik_instance.if_scope(self.select_align_params == 1):
                    self._zero_updates_ub(128, self.updates_ub_one)
                    self._zero_updates_ub(128, self.updates_ub_two)
                    self._unalign_less_than_compute_output(self.last_core_data_num, 0)

        with self.tik_instance.if_scope(self.last_core_data_num > MAX_UB_CORE_INDICES):
            loop_index = last_core_data_num // MAX_UB_CORE_INDICES
            tail_updates_number = last_core_data_num % MAX_UB_CORE_INDICES
            with self.tik_instance.for_range(0, loop_index) as loop_ub_index:
                self.tik_instance.data_move(self.indices_ub,
                                            self.input_indices[self.indices_loop_index * self.final_one_core_data_num
                                                               + loop_ub_index * MAX_UB_CORE_INDICES],
                                            0, 1, MAX_INDICES_BURST_LEN, 0, 0)
                with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                    self._unalign_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
                with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                    with self.tik_instance.if_scope(self.select_align_params == 0):
                        with self.tik_instance.if_scope(self.updates_data_num <= MAX_ALIGN_NUMBER):
                            self.tik_instance.data_move(self.updates_ub,
                                                        self.input_updates[self.indices_loop_index *
                                                                           self.final_one_core_data_num *
                                                                           self.updates_data_num + loop_ub_index *
                                                                           MAX_UB_CORE_INDICES * self.updates_data_num],
                                                        0, 1, MAX_UB_CORE_INDICES * self.updates_data_num // self.block_number, 0, 0)
                            self._align_less_than_compute_output(MAX_UB_CORE_INDICES)
                        with self.tik_instance.if_scope(self.updates_data_num > MAX_ALIGN_NUMBER):
                            self._align_more_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
                    with self.tik_instance.if_scope(self.select_align_params == 1):
                        self._unalign_less_than_compute_output(MAX_UB_CORE_INDICES, loop_ub_index)
            with self.tik_instance.if_scope(tail_updates_number != 0):
                self.tik_instance.data_move(self.indices_ub,
                                            self.input_indices[self.indices_loop_index * self.final_one_core_data_num
                                                               + loop_index * MAX_UB_CORE_INDICES],
                                            0, 1, self.tail_indices_more_than_burst_len, 0, 0)
                with self.tik_instance.if_scope(self.updates_data_num < self.block_number):
                    self._unalign_less_than_compute_output(tail_updates_number, loop_index)
                with self.tik_instance.if_scope(self.updates_data_num >= self.block_number):
                    with self.tik_instance.if_scope(self.select_align_params == 0):
                        with self.tik_instance.if_scope(self.updates_data_num * tail_updates_number <= MAX_UB_UPDATES):
                            self.tik_instance.data_move(self.updates_ub,
                                                        self.input_updates[self.indices_loop_index *
                                                                           self.final_one_core_data_num *
                                                                           self.updates_data_num + loop_index *
                                                                           MAX_UB_CORE_INDICES * self.updates_data_num],
                                                        0, 1, tail_updates_number * self.updates_data_num // self.block_number, 0, 0)
                            self._align_less_than_compute_output(tail_updates_number)
                        with self.tik_instance.else_scope():
                            self._align_more_than_compute_output(tail_updates_number, loop_index)
                    with self.tik_instance.if_scope(self.select_align_params == 1):
                        self._unalign_less_than_compute_output(tail_updates_number, loop_index)

    def _scatter_compute_tiling_ub(self):
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as indices_loop_index:
            self._init_ub_tensor()
            self.tiling_ub = self.tik_instance.Tensor("int32", (32,), name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
            self._tiling_args()
            self.indices_loop_index.set_as(indices_loop_index)
            with self.tik_instance.if_scope(self.select_mode == 1):
                with self.tik_instance.if_scope(self.select_params == 0):
                    with self.tik_instance.if_scope(self.core_num_one > 1):
                        with self.tik_instance.if_scope(indices_loop_index < self.core_num_one - 1):
                            self._process_indices_pre_core_less_than_data()
                        with self.tik_instance.if_scope(indices_loop_index == self.core_num_one - 1):
                            self._process_indices_last_core_less_than_data()
                    with self.tik_instance.if_scope(self.core_num_one == 1):
                        with self.tik_instance.if_scope(indices_loop_index == self.core_num_one - 1):
                            self._process_indices_less_than_byte()
                with self.tik_instance.if_scope(self.select_params == 1):
                    with self.tik_instance.if_scope(self.core_num_one > 1):
                        with self.tik_instance.if_scope(indices_loop_index < self.core_num_one - 1):
                            self._traversing_indices_more_than_pre_core()
                        with self.tik_instance.if_scope(indices_loop_index == self.core_num_one - 1):
                            self._traversing_indices_more_than_last_core()

# pylint: disable=unused-argument,invalid-name,too-many-locals
@te.op.register_operator("ScatterAdd")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def scatter_add(var, indices, updates, var_out, use_locking=False, kernel_name="ScatterAdd"):
    """
        scatter_add interface

        Parameters
        ----------
        var_dict: input var shape, dtype and range
        indices_dict: input indices shape, dtype and range
        updates_dict: input updates shape, dtype and range
        var_out_dict: output shape, dtype and range
        kernel_name: kernel name of scatter_add op

        Returns
        -------
        compile info
    """
    obj = Scatter(var, indices, updates, var_out, use_locking, kernel_name)
    return obj._scatter_compute()

