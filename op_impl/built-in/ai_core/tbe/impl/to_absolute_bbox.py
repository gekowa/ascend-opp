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
to_absolute_bbox
"""
import math
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *

# available ub size
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
REPEAT_TIMES_MAX = 255
SHAPE_VALUE_FOUR = 4
BLOCK_BYTES = 32
NEG_ONE = -1
POS_TWO = 2


# pylint: disable=too-many-instance-attributes,too-many-statements
class ToAbsoluteBBox:
    """
    Absolute the box that passes through the NormalizeBBox in Prediction
    """
    def __init__(self, boxes, shape_hw, reversed_box, kernel_name):
        """
        Init to_absolute_bbox parameters

        Parameters
        ----------
        boxes: dict
            shape and dtype of input
        shape_hw: dict
            shape and dtype of input, should be same shape and type as input
        reversed_box: bool
            default value is False
        kernel_name: kernel_name

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.boxes = boxes
        self.shape_hw = shape_hw
        self.reversed_box = reversed_box
        self.kernel_name = kernel_name
        self.boxes_shape = self.boxes.get("shape")
        self.shape_hw_shape = self.shape_hw.get("shape")
        self.boxes_dtype = self.boxes.get("dtype").lower()
        self.shape_hw_dtype = self.shape_hw.get("dtype").lower()
        self.ub_size_bytes = UB_SIZE - 9216
        self.core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.data_num = self.boxes_shape[1] * self.boxes_shape[2]
        self.boxes_dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(self.boxes_dtype) // 8
        self.boxes_data_each_block = BLOCK_BYTES // self.boxes_dtype_bytes_size
        if self.reversed_box is True:
            self.n_value = self.boxes_shape[2]
        else:
            self.n_value = self.boxes_shape[1]
        self.n_value_block = math.ceil(self.n_value / self.boxes_data_each_block) * self.boxes_data_each_block
        self.mask = 16
        if self.boxes_dtype in "float32":
            self.mask = 8
        self.max_num_one_repeat = 128
        if self.boxes_dtype in "float32":
            self.max_num_one_repeat = 64
        self.batch_each_core = 0
        self.batch_last_core = 0
        self.n_each_core = 0
        self.n_last_core = 0
        self.each_core_data = 0
        self.last_core_data = 0
        self.n_each_core_block = 0
        self.n_last_core_block = 0
        self.batch_each_core_data = 0
        self.core_data = 0
        self.batch_flag = False
        self.batch_or_num()
        if self.batch_flag:
            self.cal_batch_core_num()
        else:
            self.cal_core_num()
        self.ub_number = 0
        self.boxes_ub_number = 0
        self.shape_hw_ub_number = 0
        self.each_data_num = 0
        self.n_num_each = 0
        self.boxes_burst_len = 0
        self.core_loop_index = 0
        self.batch_index = 0
        self.boxes_ub = None
        self.shape_hw_ub_one = None
        self.shape_hw_ub_two = None
        self.temp_ub = None
        self.tile_ub = None
        self.hw_ub = None
        self.hw_ub_float = None
        self.hw_ub_temp = None
        self.height = None
        self.width = None

        self.boxes_gm = self.tik_instance.Tensor(self.boxes_dtype, self.boxes_shape, name="boxes_gm",
                                                 scope=tik.scope_gm)
        self.shape_hw_gm = self.tik_instance.Tensor(self.shape_hw_dtype, self.shape_hw_shape, name="shape_hw_gm",
                                                    scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.boxes_dtype, self.boxes_shape, name="out_gm", scope=tik.scope_gm)
        self.check_param()

    def batch_or_num(self):
        """
        how to segment core

        Parameters
        ----------
        None

        Returns
        -------
        core_num: the num of core
        """
        if self.reversed_box:
            if self.boxes_shape[0] > 1 and (self.n_value % self.boxes_data_each_block == 0):
                self.batch_flag = True
        else:
            if self.boxes_shape[0] > 1:
                self.batch_flag = True

    def cal_core_num(self):
        """
        update_core_num

        Parameters
        ----------
        None

        Returns
        -------
        core_num: the num of core
        """
        self.n_each_core = math.ceil(self.n_value / self.core_num)
        self.n_each_core_block = math.ceil(self.n_each_core / self.boxes_data_each_block) * self.boxes_data_each_block
        self.core_num = math.ceil(self.n_value / self.n_each_core)
        self.each_core_data = self.n_each_core * SHAPE_VALUE_FOUR
        self.n_last_core = self.n_value - (self.core_num - 1) * self.n_each_core
        self.n_last_core_block = math.ceil(self.n_last_core / self.boxes_data_each_block) * self.boxes_data_each_block
        self.last_core_data = self.n_last_core * SHAPE_VALUE_FOUR
        if (self.n_each_core % self.boxes_data_each_block != 0) or (self.n_last_core < self.boxes_data_each_block):
            self.core_num = self.update_core_num(self.core_num)
            self.n_each_core_block = (math.ceil(self.n_each_core / self.boxes_data_each_block) *
                                      self.boxes_data_each_block)
            self.each_core_data = self.n_each_core * SHAPE_VALUE_FOUR
            self.n_last_core = self.n_value - (self.core_num - 1) * self.n_each_core
            self.n_last_core_block = (math.ceil(self.n_last_core / self.boxes_data_each_block) *
                                      self.boxes_data_each_block)
            self.last_core_data = self.n_last_core * SHAPE_VALUE_FOUR

    def update_core_num(self, core_num):
        """
        update_core_num

        Parameters
        ----------
        core_num: the num of core

        Returns
        -------
        core_num: the num of core
        """
        if (self.n_each_core % self.boxes_data_each_block == 0) and (self.n_last_core < self.boxes_data_each_block):
            core_num = core_num - 1
            self.n_last_core = self.n_each_core + self.n_last_core
            if core_num == 1:
                self.n_each_core = self.n_last_core
                self.n_last_core = 0
            return core_num
        elif self.n_each_core % self.boxes_data_each_block != 0:
            self.n_each_core = self.n_each_core + 1
            core_num = math.ceil(self.n_value / self.n_each_core)
            if core_num == 1:
                return core_num
            self.n_last_core = self.n_value - (core_num - 1) * self.n_each_core
            if ((self.n_each_core % self.boxes_data_each_block == 0) and
                    (self.n_last_core >= self.boxes_data_each_block)):
                return core_num
        return self.update_core_num(core_num)

    def cal_batch_core_num(self):
        """
        update_core_num

        Parameters
        ----------
        None

        Returns
        -------
        core_num: the num of core
        """
        self.batch_each_core = math.ceil(self.boxes_shape[0] / self.core_num)
        self.core_num = math.ceil(self.boxes_shape[0] / self.batch_each_core)
        self.batch_last_core = self.boxes_shape[0] - (self.core_num - 1) * self.batch_each_core
        if not self.reversed_box:
            if self.data_num < self.boxes_data_each_block:
                self.core_num = 1
                self.batch_each_core = self.boxes_shape[0]
                self.batch_last_core = self.boxes_shape[0]
        self.n_each_core = self.n_value
        self.n_each_core_block = self.n_value_block
        self.each_core_data = self.data_num
        self.n_last_core = self.n_value
        self.n_last_core_block = self.n_value_block
        self.last_core_data = self.data_num
        self.batch_each_core_data = self.batch_each_core * self.data_num

    def init_ub_tensor(self, n_each_core_block, each_core_data):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        n_each_core_block: each core n block num
        each_core_data: each core data

        Returns
        -------
        None
        """
        self.ub_number = ((self.ub_size_bytes // self.boxes_dtype_bytes_size // self.boxes_data_each_block) *
                          self.boxes_data_each_block)
        if self.reversed_box:
            if self.ub_number > n_each_core_block * SHAPE_VALUE_FOUR:
                self.boxes_ub_number = n_each_core_block * SHAPE_VALUE_FOUR
                self.each_data_num = self.boxes_ub_number
                self.n_num_each = n_each_core_block
            else:
                self.boxes_ub_number = self.ub_number // self.boxes_data_each_block * self.boxes_data_each_block
                self.n_num_each = (self.boxes_ub_number // SHAPE_VALUE_FOUR // self.boxes_data_each_block *
                                   self.boxes_data_each_block)
                self.boxes_ub_number = self.n_num_each * SHAPE_VALUE_FOUR
                self.each_data_num = self.boxes_ub_number
            self.core_data = self.n_each_core
            if self.batch_flag:
                self.core_data = self.batch_each_core_data
        else:
            self.boxes_ub_number = self.ub_number // self.boxes_data_each_block * self.boxes_data_each_block
            if self.boxes_ub_number > each_core_data:
                self.boxes_ub_number = (math.ceil(each_core_data / self.boxes_data_each_block) *
                                        self.boxes_data_each_block)
            self.each_data_num = self.boxes_ub_number
            if not self.batch_flag:
                self.batch_each_core_data = self.each_core_data

        self.shape_hw_ub_number = self.boxes_data_each_block

        self.boxes_ub = self.tik_instance.Tensor(self.boxes_dtype, (self.boxes_ub_number,), name="boxes_ub",
                                                 scope=tik.scope_ubuf)
        self.shape_hw_ub_one = self.tik_instance.Tensor(self.boxes_dtype, (self.shape_hw_ub_number,),
                                                        name="shape_hw_ub_one", scope=tik.scope_ubuf)
        self.shape_hw_ub_two = self.tik_instance.Tensor(self.boxes_dtype, (self.shape_hw_ub_number,),
                                                        name="shape_hw_ub_two", scope=tik.scope_ubuf)
        self.hw_ub = self.tik_instance.Tensor(self.shape_hw_dtype, (self.shape_hw_ub_number,),
                                              name="hw_ub", scope=tik.scope_ubuf)
        self.hw_ub_float = self.tik_instance.Tensor(self.boxes_dtype, (self.shape_hw_ub_number,),
                                                    name="hw_ub_float", scope=tik.scope_ubuf)
        self.hw_ub_temp = self.tik_instance.Tensor("float16", (16,), name="hw_ub_temp", scope=tik.scope_ubuf)
        self.tile_ub = self.tik_instance.Tensor(self.boxes_dtype, (self.shape_hw_ub_number,),
                                                name="tile_ub", scope=tik.scope_ubuf)
        self.temp_ub = self.tik_instance.Tensor(self.boxes_dtype, (self.shape_hw_ub_number,),
                                                name="temp_ub", scope=tik.scope_ubuf)

        self.height = self.tik_instance.Scalar(self.boxes_dtype)
        self.height.set_as(0)
        self.width = self.tik_instance.Scalar(self.boxes_dtype)
        self.width.set_as(0)

    def get_hw_value(self):
        """
        get the value of height and width

        Returns
        -------
        None
        """
        batch_offset = 0
        h_value_index = 1
        w_value_index = 2
        self.tik_instance.data_move(self.hw_ub, self.shape_hw_gm[batch_offset], 0, 1, 1, 0, 0)
        if self.boxes_dtype == "float16":
            self.tik_instance.vconv(4, '', self.hw_ub_float, self.hw_ub, 1, 1, 1, 0, 0, deqscale=1.0)
        elif self.boxes_dtype == "float32":
            self.tik_instance.vconv(4, '', self.hw_ub_temp, self.hw_ub, 1, 1, 1, 0, 0, deqscale=1.0)
            self.tik_instance.vconv(4, '', self.hw_ub_float, self.hw_ub_temp, 1, 1, 1, 0, 0)
        else:
            error_info = {
                'errCode': OP_ERROR_CODE_008,
                'op_name': 'to_absolute_bbox',
                'param_name': 'normalized_boxes',
                'excepted_dtype_list': ["float16", "float32"],
                'dtype': self.boxes_dtype
            }
            raise RuntimeError(
                error_info,
                "In op[{op_name}], the parameter[{param_name}]'s dtype should "
                "be one of [{excepted_dtype_list}], but actually is [{dtype}]."
                .format(**error_info))

        self.height.set_as(self.hw_ub_float[h_value_index])
        self.width.set_as(self.hw_ub_float[w_value_index])

        if self.reversed_box:
            self.tik_instance.vector_dup(self.mask, self.shape_hw_ub_one, self.height, 1, 1, 8, 0)
            self.tik_instance.vector_dup(self.mask, self.shape_hw_ub_two, self.width, 1, 1, 8, 0)

        else:
            with self.tik_instance.for_range(0, self.boxes_data_each_block // 2) as block_index_one:
                self.shape_hw_ub_one[block_index_one * 2] = self.height
            with self.tik_instance.for_range(0, self.boxes_data_each_block // 2) as block_index_two:
                self.shape_hw_ub_one[1 + block_index_two * 2] = self.width

    def cal_loop_four_num(self, each_core_data):
        """
        cal loop num

        Parameters
        ----------
        each_core_data: each core data num

        Returns
        -------
        None
        """
        loop = each_core_data // self.each_data_num
        last_n = each_core_data % self.each_data_num // SHAPE_VALUE_FOUR
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as loop_index:
                self.move_to_ub_four_num(self.n_num_each, loop_index)
                self.absolute_process(self.n_num_each, loop_index)
        if last_n > 0:
            self.move_to_ub_four_num(last_n, loop)
            self.absolute_process(last_n, loop)

    def move_to_ub_four_num(self, n_num_each, loop_index):
        """
        move data to UB

        Parameters
        ----------
        n_num_each: the n value of each core
        loop_index: loop num

        Returns
        -------
        None
        """
        self.boxes_burst_len = math.ceil(n_num_each / self.boxes_data_each_block)
        for index in range(SHAPE_VALUE_FOUR):
            self.tik_instance.data_move(self.boxes_ub[index * self.n_num_each],
                                        self.boxes_gm[self.batch_index * self.boxes_shape[1] * self.boxes_shape[2] +
                                                      self.core_loop_index * self.core_data + loop_index *
                                                      self.n_num_each + index * self.n_value],
                                        0, 1, self.boxes_burst_len, 0, 0)

    def absolute_process(self, n_num_each, loop_index):
        """
        normalize process

        Parameters
        ----------
        n_num_each: the n value of each core
        loop_index: loop num

        Returns
        -------
        None
        """
        cal_loop = n_num_each // self.max_num_one_repeat // REPEAT_TIMES_MAX
        with self.tik_instance.for_range(0, cal_loop) as cal_index:
            index_offset = cal_index * self.max_num_one_repeat * REPEAT_TIMES_MAX
            self.cal_process_four_num(self.max_num_one_repeat, index_offset, REPEAT_TIMES_MAX)
        last_loop = n_num_each % (self.max_num_one_repeat * REPEAT_TIMES_MAX) // self.max_num_one_repeat
        if last_loop > 0:
            index_offset = cal_loop * self.max_num_one_repeat * REPEAT_TIMES_MAX
            self.cal_process_four_num(self.max_num_one_repeat, index_offset, last_loop)
        compute_mask = n_num_each % self.max_num_one_repeat
        tile_num = n_num_each % self.boxes_data_each_block
        if compute_mask > 0:
            index_offset = n_num_each // self.max_num_one_repeat * self.max_num_one_repeat
            self.cal_process_four_num(compute_mask, index_offset, 1)
            if tile_num == 0 or self.boxes_burst_len == 1:
                self.move_to_gm_four_num(loop_index)
            else:
                align_num = n_num_each // self.boxes_data_each_block * self.boxes_data_each_block
                align_offset = align_num - (self.boxes_data_each_block - tile_num)
                for tile_index in range(SHAPE_VALUE_FOUR):
                    with self.tik_instance.for_range(0, self.boxes_data_each_block) as index:
                        temp_offset = align_offset + index
                        self.tile_ub[index] = self.boxes_ub[tile_index * self.n_num_each + temp_offset]
                    self.tik_instance.data_move(self.out_gm[self.batch_index * self.boxes_shape[1] *
                                                            self.boxes_shape[2] + self.core_loop_index *
                                                            self.core_data + loop_index * self.n_num_each +
                                                            tile_index * self.n_value + align_offset],
                                                self.tile_ub, 0, 1, 1, 0, 0)
                self.move_to_gm_four_num(loop_index, align_offset)
        else:
            self.move_to_gm_four_num(loop_index)

    def cal_process_four_num(self, compute_mask, index_offset, repeat_time):
        """
        cal process

        Parameters
        ----------
        compute_mask: mask value
        index_offset: cal offset
        repeat_time: repeat time

        Returns
        -------
        None
        """
        self.tik_instance.vmul(compute_mask, self.boxes_ub[index_offset + self.n_num_each * 0],
                               self.boxes_ub[index_offset + self.n_num_each * 0],
                               self.shape_hw_ub_one, repeat_time, 1, 1, 0, 8, 8, 0)
        self.tik_instance.vmul(compute_mask, self.boxes_ub[index_offset + self.n_num_each * 1],
                               self.boxes_ub[index_offset + self.n_num_each * 1],
                               self.shape_hw_ub_two, repeat_time, 1, 1, 0, 8, 8, 0)
        self.tik_instance.vmul(compute_mask, self.boxes_ub[index_offset + self.n_num_each * 2],
                               self.boxes_ub[index_offset + self.n_num_each * 2],
                               self.shape_hw_ub_one, repeat_time, 1, 1, 0, 8, 8, 0)
        self.tik_instance.vmul(compute_mask, self.boxes_ub[index_offset + self.n_num_each * 3],
                               self.boxes_ub[index_offset + self.n_num_each * 3],
                               self.shape_hw_ub_two, repeat_time, 1, 1, 0, 8, 8, 0)

    def move_to_gm_four_num(self, loop_index, offset=0):
        """
        move data to UB

        Parameters
        ----------
        loop_index: loop num
        offset: offset

        Returns
        -------
        None
        """
        if offset:
            self.boxes_burst_len = self.boxes_burst_len - 1
        for index in range(SHAPE_VALUE_FOUR):
            self.tik_instance.data_move(self.out_gm[self.batch_index * self.boxes_shape[1] * self.boxes_shape[2] +
                                                    self.core_loop_index * self.core_data + loop_index *
                                                    self.n_num_each + index * self.n_value],
                                        self.boxes_ub[index * self.n_num_each], 0, 1, self.boxes_burst_len, 0, 0)

    def cal_loop_num_four(self, each_core_data):
        """
        cal loop num

        Parameters
        ----------
        each_core_data: each core data num

        Returns
        -------
        None
        """
        loop = each_core_data // self.each_data_num
        last_num = each_core_data % self.each_data_num
        if loop > 0:
            with self.tik_instance.for_range(0, loop) as loop_index:
                self.move_to_ub_num_four(self.boxes_ub_number, loop_index)
                self.to_absolute_process_num_four(self.boxes_ub_number, loop_index)
        if last_num > 0:
            self.move_to_ub_num_four(last_num, loop)
            self.to_absolute_process_num_four(last_num, loop)

    def move_to_ub_num_four(self, element_num, loop_index):
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
        self.boxes_burst_len = math.ceil(element_num / self.boxes_data_each_block)
        self.tik_instance.data_move(self.boxes_ub, self.boxes_gm[
            self.batch_index * self.boxes_shape[1] * self.boxes_shape[2] + self.core_loop_index *
            self.batch_each_core_data + loop_index * self.each_data_num], 0, 1, self.boxes_burst_len, 0, 0)

    def to_absolute_process_num_four(self, element_num, loop_index):
        """
        absolute process

        Parameters
        ----------
        element_num: the element num of a loop
        loop_index: loop num

        Returns
        -------
        None
        """
        cal_loop = element_num // self.max_num_one_repeat // REPEAT_TIMES_MAX
        if cal_loop > 0:
            with self.tik_instance.for_range(0, cal_loop) as cal_index:
                index_offset = cal_index * self.max_num_one_repeat * REPEAT_TIMES_MAX
                self.cal_process_num_four(self.max_num_one_repeat, index_offset, REPEAT_TIMES_MAX)
        last_loop = element_num % (self.max_num_one_repeat * REPEAT_TIMES_MAX) // self.max_num_one_repeat
        if last_loop > 0:
            index_offset = cal_loop * self.max_num_one_repeat * REPEAT_TIMES_MAX
            self.cal_process_num_four(self.max_num_one_repeat, index_offset, last_loop)
        compute_mask = element_num % self.max_num_one_repeat
        tile_num = element_num % self.boxes_data_each_block
        if compute_mask > 0:
            index_offset = element_num // self.max_num_one_repeat * self.max_num_one_repeat
            self.cal_process_num_four(compute_mask, index_offset, 1)
            if tile_num == 0 or self.boxes_burst_len == 1:
                self.move_to_gm_num_four(loop_index)
            else:
                align_num = element_num // self.boxes_data_each_block * self.boxes_data_each_block
                align_offset = align_num - (self.boxes_data_each_block - tile_num)
                with self.tik_instance.for_range(0, self.boxes_data_each_block) as index:
                    temp_offset = align_offset + index
                    self.tile_ub[index] = self.boxes_ub[temp_offset]
                self.tik_instance.data_move(self.out_gm[self.batch_index * self.boxes_shape[1] * self.boxes_shape[2] +
                                                        self.core_loop_index * self.batch_each_core_data +
                                                        loop_index * self.each_data_num + align_offset], self.tile_ub,
                                            0, 1, 1, 0, 0)
                self.move_to_gm_num_four(loop_index, align_offset)
        else:
            self.move_to_gm_num_four(loop_index)

    def cal_process_num_four(self, compute_mask, index_offset, repeat_time):
        """
        cal process

        Parameters
        ----------
        compute_mask: mask value
        index_offset: cal offset
        repeat_time: repeat time

        Returns
        -------
        None
        """
        self.tik_instance.vmul(compute_mask, self.boxes_ub[index_offset], self.boxes_ub[index_offset],
                               self.shape_hw_ub_one, repeat_time, 1, 1, 0, 8, 8, 0)

    def move_to_gm_num_four(self, loop_index, offset=0):
        """
        move data to UB

        Parameters
        ----------
        loop_index: loop num
        offset: offset

        Returns
        -------
        None
        """
        if offset:
            self.boxes_burst_len = self.boxes_burst_len - 1
        self.tik_instance.data_move(self.out_gm[self.batch_index * self.boxes_shape[1] * self.boxes_shape[2] +
                                                self.core_loop_index * self.batch_each_core_data + loop_index *
                                                self.each_data_num], self.boxes_ub, 0, 1, self.boxes_burst_len, 0, 0)

    def check_param(self):
        """
        check parameters

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        error_info = {
            "errCode": OP_ERROR_CODE_009,
            "op_name": "to_absolute_bbox",
            "rule_desc": "the boxes' shape must be 3 dimension",
            "param_name": "normalized_boxes",
            "param_value": self.boxes_shape
        }

        if len(self.boxes_shape) != 3:
            error_info["rule_desc"] = "the boxes' shape must be 3 dimension"
            raise RuntimeError(error_info,
                               "Op[{op_name}] has rule: {rule_desc}, "
                               "but [{param_name}] is [{param_value}]."
                               .format(**error_info))

        if self.reversed_box:
            if self.boxes_shape[1] != 4:
                error_info["rule_desc"] = "the boxes' second dimension must be 4"
                raise RuntimeError(error_info,
                                   "Op[{op_name}] has rule: {rule_desc}, "
                                   "but [{param_name}] is [{param_value}]."
                                   .format(**error_info))
        else:
            if self.boxes_shape[2] != 4:
                error_info["rule_desc"] = "the boxes' third dimension must be 4"
                raise RuntimeError(error_info,
                                   "Op[{op_name}] has rule: {rule_desc}, "
                                   "but [{param_name}] is [{param_value}]."
                                   .format(**error_info))

        if len(self.shape_hw_shape) != 1:
            error_info["rule_desc"] = "the shape_hw's shape must be 1 dimension"
            error_info["param_name"] = "shape_hw"
            raise RuntimeError(error_info,
                               "Op[{op_name}] has rule: {rule_desc}, "
                               "but [{param_name}] is [{param_value}]."
                               .format(**error_info))

        if self.shape_hw_shape[-1] != 4:
            error_info["rule_desc"] = "the shape_hw's first dimension must be 4"
            error_info["param_name"] = "shape_hw"
            raise RuntimeError(error_info,
                               "Op[{op_name}] has rule: {rule_desc}, "
                               "but [{param_name}] is [{param_value}]."
                               .format(**error_info))

    def to_absolute_bbox_operator(self):
        """
        to_absolute_bbox operation

        Parameters
        ----------
        None

        Returns:
        ----------
        self.tik_instance: tensor
        """
        if self.batch_flag:
            if self.reversed_box:
                with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_loop_index:
                    with self.tik_instance.if_scope(core_loop_index != self.core_num - 1):
                        with self.tik_instance.for_range(0, self.batch_each_core) as batch_index:
                            self.init_ub_tensor(self.n_value_block, self.data_num)
                            self.get_hw_value()
                            self.core_loop_index = core_loop_index
                            self.batch_index = batch_index
                            self.cal_loop_four_num(self.each_core_data)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(0, self.batch_last_core) as batch_index:
                            self.init_ub_tensor(self.n_value_block, self.data_num)
                            self.get_hw_value()
                            self.core_loop_index = core_loop_index
                            self.batch_index = batch_index
                            self.cal_loop_four_num(self.last_core_data)
            else:
                with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_loop_index:
                    with self.tik_instance.if_scope(core_loop_index != self.core_num - 1):
                        with self.tik_instance.for_range(0, self.batch_each_core) as batch_index:
                            self.init_ub_tensor(self.n_each_core_block, self.each_core_data)
                            self.get_hw_value()
                            self.core_loop_index = core_loop_index
                            self.batch_index = batch_index
                            self.cal_loop_num_four(self.each_core_data)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(0, self.batch_last_core) as batch_index:
                            self.init_ub_tensor(self.n_last_core_block, self.last_core_data)
                            self.get_hw_value()
                            self.core_loop_index = core_loop_index
                            self.batch_index = batch_index
                            self.cal_loop_num_four(self.last_core_data)
        else:
            with self.tik_instance.for_range(0, self.boxes_shape[0]) as batch_index:
                if self.reversed_box is True:
                    with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_loop_index:
                        with self.tik_instance.if_scope(core_loop_index != self.core_num - 1):
                            self.init_ub_tensor(self.n_each_core_block, self.each_core_data)
                            self.get_hw_value()
                            self.core_loop_index = core_loop_index
                            self.batch_index = batch_index
                            self.cal_loop_four_num(self.each_core_data)
                        with self.tik_instance.else_scope():
                            self.init_ub_tensor(self.n_last_core_block, self.last_core_data)
                            self.get_hw_value()
                            self.core_loop_index = core_loop_index
                            self.batch_index = batch_index
                            self.cal_loop_four_num(self.last_core_data)
                else:
                    with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as core_loop_index:
                        with self.tik_instance.if_scope(core_loop_index != self.core_num - 1):
                            self.init_ub_tensor(self.n_each_core_block, self.each_core_data)
                            self.get_hw_value()
                            self.core_loop_index = core_loop_index
                            self.batch_index = batch_index
                            self.cal_loop_num_four(self.each_core_data)
                        with self.tik_instance.else_scope():
                            self.init_ub_tensor(self.n_last_core_block, self.last_core_data)
                            self.get_hw_value()
                            self.core_loop_index = core_loop_index
                            self.batch_index = batch_index
                            self.cal_loop_num_four(self.last_core_data)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.boxes_gm, self.shape_hw_gm),
            outputs=self.out_gm)

        return self.tik_instance


# pylint: disable=unused-argument,invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_BOOL, KERNEL_NAME)
def to_absolute_bbox(normalized_boxes, shape_hw, y, reversed_box=False, kernel_name="to_absolute_bbox"):
    """
    Normalize the pre-selected box that passes through the NMS in Prediction

    Parameters
    ----------
    normalized_boxes : dict
        shape and dtype of input
    shape_hw : dict
        shape and dtype of input, should be same shape and type as input
    y : dict
        shape and dtype of output, should be same shape and type as normalized_boxes
    reversed_box : bool
        default value is False
    kernel_name : str
        kernel name, default value is "to_absolute_bbox"

    Returns
    -------
    None
    """
    to_absolute_bbox_result = ToAbsoluteBBox(normalized_boxes, shape_hw, reversed_box, kernel_name)
    return to_absolute_bbox_result.to_absolute_bbox_operator()
