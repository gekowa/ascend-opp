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
stn_compute
"""
import math
from topi.cce import util
from te import tik
from te import platform as cce
from functools import reduce
from te.utils.op_utils import *


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_LIST_INT, OPTION_ATTR_BOOL, KERNEL_NAME)
def stn_compute(input_x, input_theta, input_offset, output_y, size=(-1, -1, -1, -1), align_corners=False,
                kernel_name="stn_compute"):
    """
    spatial transformer by theta

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    input_theta: dict
        auxiliary_coefficients
    input_offset: dict
        auxiliary_offset
    size: tuple
        output_size
    align_corners: bool
        false
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "stn_compute"

    Returns
    -------
    None
    """

    shape = input_x.get("shape")

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_kernel_name(kernel_name)

    stn_instance = SpatialTransformer(input_x, input_theta, input_offset, kernel_name)
    stn_instance.spatial_transformer_compute()
    return stn_instance


# input_theta type same as input_x
class SpatialTransformer:
    def __init__(self, input_x, auxiliary_coefficients, auxiliary_offset, kernel_name='stn_compute'):

        self.d_type_x = input_x.get('dtype')
        self.theta_dtype = auxiliary_coefficients.get('dtype')
        self.position_dtype = auxiliary_offset.get('dtype')
        self.shape = input_x.get('shape')

        self.kernel_name = kernel_name

        # product_name = tik_get_soc_name.get_soc_name()
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.ai_core_num = tik.Dprofile().get_aicore_num()

        ub_size_bytes = cce.get_soc_spec(cce.cce_conf.UB_SIZE) // 2  # double buffer
        self.d_type_bytes_size = cce.cce_intrin.get_bit_len(self.d_type_x) // 8
        self.theta_type_bytes_size = cce.cce_intrin.get_bit_len(auxiliary_coefficients.get('dtype')) // 8
        self.offset_type_bytes_size = cce.cce_intrin.get_bit_len(auxiliary_offset.get('dtype')) // 8
        self.vec_compute_size = 256

        # theta size output_h * output_w * 4 * n * c1
        self.theta_size = auxiliary_coefficients.get('shape')[0] * auxiliary_coefficients.get('shape')[1] * \
                          auxiliary_coefficients.get('shape')[2]

        # output_h * output_w
        self.output_hw = self.theta_size // 4 // self.shape[0] // self.shape[1]

        # tiling policy
        self.total_c1 = self.shape[0] * self.shape[1]
        self.ub_tensor_size = 16 if self.shape[1] * self.shape[4] * self.d_type_bytes_size > ub_size_bytes * 0.4 else \
            self.shape[1] * self.shape[4]
        self.input_stride = (self.shape[2] * self.shape[3] * self.shape[4] - self.shape[4]) \
                            * self.d_type_bytes_size // 32
        self.output_stride = (self.output_hw * self.shape[4] - self.shape[4]) * self.d_type_bytes_size // 32
        self.if_skip_read_ceof = self.ub_tensor_size != 16

        # nc1hwc0 c0 = 16 theta type same as input_x
        self.ub_tensor_len = self.ub_tensor_size * self.d_type_bytes_size // 32

        self.input_hw = self.shape[2] * self.shape[3]

        # ub theta size must be a multiple of 4 and 32
        ub_theta_offset_can_use = (ub_size_bytes - self.ub_tensor_size * self.d_type_bytes_size * 2) \
                                  // (self.theta_type_bytes_size + self.offset_type_bytes_size)
        self.ub_theta_offset_size = ub_theta_offset_can_use - ub_theta_offset_can_use % 4

        theta_burst_len = self.ub_theta_offset_size * self.theta_type_bytes_size // 32
        offset_burst_len = self.ub_theta_offset_size * self.offset_type_bytes_size // 32
        self.ub_theta_offset_size = min(theta_burst_len * 32 // self.theta_type_bytes_size,
                                        offset_burst_len * 32 // self.offset_type_bytes_size)

        self.input_num = reduce(lambda x, y: x * y, input_x.get('shape'))
        # self.input_num = self.theta_size * 4

        # input data
        self.input_x_gm = self.tik_instance.Tensor(
            self.d_type_x, (self.input_num,), name='input_x_gm', scope=tik.scope_gm
        )
        # theta matrix
        self.input_theta_gm = self.tik_instance.Tensor(
            auxiliary_coefficients.get('dtype'), (self.theta_size,), name='input_theta_gm', scope=tik.scope_gm
        )
        # position offset matrix
        self.input_position_gm = self.tik_instance.Tensor(
            auxiliary_offset.get('dtype'), (self.theta_size,), name='input_position_gm', scope=tik.scope_gm
        )

        # output data
        self.output_y_gm = self.tik_instance.Tensor(
            self.d_type_x, (self.output_hw * self.shape[0] * self.shape[1] * self.shape[4],),
            name='output_y_gm', scope=tik.scope_gm
        )

    def spatial_transformer_compute(self):
        # handle one C1 data
        if self.if_skip_read_ceof:
            # self.ub_tensor_size = c1 * c0
            # ceof and offset size = n * h * w * 4
            ceof_and_offset_size = self.theta_size // self.shape[1]
            use_core_count = self._calc_loop_count(ceof_and_offset_size)
            each_core_need_process_ceof = ceof_and_offset_size // 4 // use_core_count * 4
            each_core_loop_count = each_core_need_process_ceof // self.ub_theta_offset_size
            if each_core_need_process_ceof > 0:
                with self.tik_instance.for_range(0, use_core_count, block_num=use_core_count) as core_index:
                    self.process_on_core(each_core_loop_count, each_core_need_process_ceof,
                                         core_index * each_core_need_process_ceof)
            remain_theta = ceof_and_offset_size // 4 % use_core_count * 4
            if remain_theta > 0:
                # one core handle remain theta
                need_loop_count = remain_theta // self.ub_theta_offset_size
                self.process_on_core(need_loop_count, remain_theta,
                                     each_core_need_process_ceof * use_core_count)

        else:
            use_core_count = self._calc_loop_count(self.theta_size)
            each_core_need_process_theta = self.theta_size // 4 // use_core_count * 4
            each_core_loop_count = each_core_need_process_theta // self.ub_theta_offset_size
            if each_core_need_process_theta > 0:
                with self.tik_instance.for_range(0, use_core_count, block_num=use_core_count) as core_index:
                    self.process_on_core(each_core_loop_count, each_core_need_process_theta,
                                         core_index * each_core_need_process_theta)
            remain_theta = self.theta_size // 4 % use_core_count * 4
            if remain_theta > 0:
                # one core handle remain theta
                need_loop_count = remain_theta // self.ub_theta_offset_size
                self.process_on_core(need_loop_count, remain_theta,
                                     each_core_need_process_theta * use_core_count)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_x_gm, self.input_theta_gm, self.input_position_gm],
            outputs=[self.output_y_gm])
        return self.tik_instance

    def process_on_core(self, each_core_loop_count, each_core_need_process_theta, processed_theta):
        if each_core_loop_count > 0:
            with self.tik_instance.for_range(0, each_core_loop_count) as move_theta_ub:
                already_process_ceof = processed_theta + move_theta_ub * self.ub_theta_offset_size
                input_theta_ub = self.tik_instance.Tensor(
                    self.d_type_x, (self.ub_theta_offset_size,), name='input_theta_ub', scope=tik.scope_ubuf
                )
                input_position_ub = self.tik_instance.Tensor(
                    self.position_dtype, (self.ub_theta_offset_size,), name='input_position_ub', scope=tik.scope_ubuf
                )
                if self.if_skip_read_ceof:
                    self.move_ceof_and_offset_to_ub_2(already_process_ceof, input_position_ub, input_theta_ub,
                                                      self.ub_theta_offset_size)
                    self.process_batch_hw(input_position_ub, input_theta_ub, already_process_ceof,
                                          self.ub_theta_offset_size)
                else:
                    theta_burst_len = self.ub_theta_offset_size * self.theta_type_bytes_size // 32
                    offset_burst_len = self.ub_theta_offset_size * self.offset_type_bytes_size // 32
                    self.tik_instance.data_move(
                        dst=input_theta_ub,
                        src=self.input_theta_gm[already_process_ceof],
                        sid=0, nburst=1, burst=theta_burst_len, src_stride=0, dst_stride=0
                    )
                    self.tik_instance.data_move(
                        dst=input_position_ub,
                        src=self.input_position_gm[already_process_ceof],
                        sid=0, nburst=1, burst=offset_burst_len, src_stride=0, dst_stride=0
                    )
                    # calc result
                    self.process_c1_hw(input_position_ub, input_theta_ub, already_process_ceof,
                                       self.ub_theta_offset_size)
        last_num = each_core_need_process_theta % self.ub_theta_offset_size
        if last_num > 0:
            already_process_ceof = processed_theta + each_core_need_process_theta - last_num
            input_theta_ub = self.tik_instance.Tensor(
                self.d_type_x, (self.ub_theta_offset_size,), name='input_theta_ub', scope=tik.scope_ubuf
            )
            input_position_ub = self.tik_instance.Tensor(
                self.position_dtype, (self.ub_theta_offset_size,), name='input_position_ub', scope=tik.scope_ubuf
            )

            # calc result
            if self.if_skip_read_ceof:
                self.move_ceof_and_offset_to_ub_2(already_process_ceof, input_position_ub, input_theta_ub, last_num)
                self.process_batch_hw(input_position_ub, input_theta_ub, already_process_ceof, last_num)
            else:
                theta_burst_len = math.ceil(last_num / (32 // self.theta_type_bytes_size))
                offset_burst_len = math.ceil(last_num / (32 // self.offset_type_bytes_size))
                self.tik_instance.data_move(
                    dst=input_theta_ub,
                    src=self.input_theta_gm[already_process_ceof],
                    sid=0, nburst=1, burst=theta_burst_len, src_stride=0, dst_stride=0
                )
                self.tik_instance.data_move(
                    dst=input_position_ub,
                    src=self.input_position_gm[already_process_ceof],
                    sid=0, nburst=1, burst=offset_burst_len, src_stride=0, dst_stride=0
                )
                self.process_c1_hw(input_position_ub, input_theta_ub, already_process_ceof, last_num)

    def move_ceof_and_offset_to_ub_2(self, already_process_ceof, input_position_ub, input_theta_ub, need_hand_count):
        move_size = need_hand_count if need_hand_count < self.ub_theta_offset_size else self.ub_theta_offset_size

        self.tik_instance.data_move(
            dst=input_theta_ub,
            src=self.input_theta_gm[already_process_ceof],
            sid=0, nburst=1,
            burst=self.ceil_value(move_size, 32 // self.theta_type_bytes_size),
            src_stride=0, dst_stride=0
        )
        self.tik_instance.data_move(
            dst=input_position_ub,
            src=self.input_position_gm[already_process_ceof],
            sid=0, nburst=1,
            burst=self.ceil_value(move_size, 32 // self.offset_type_bytes_size),
            src_stride=0, dst_stride=0
        )

    def process_batch_hw(self, input_position_ub, input_ceof_ub, already_process_ceof, hande_ceof_size):
        if self.ub_tensor_size > (self.vec_compute_size // self.d_type_bytes_size):
            repeats = self.ub_tensor_size * self.d_type_bytes_size // self.vec_compute_size
            repeats_nums = self.vec_compute_size // self.d_type_bytes_size
            last_num = (self.ub_tensor_size * self.d_type_bytes_size % self.vec_compute_size) // self.d_type_bytes_size
        else:
            repeats = 1
            repeats_nums = self.ub_tensor_size
            last_num = 0
        need_hand_last = last_num != 0
        thread_num = 2 if hande_ceof_size // 4 >= 2 else 1
        with self.tik_instance.for_range(0, hande_ceof_size // 4, thread_num=thread_num) as i:
            tmp_res_ub = self.tik_instance.Tensor(
                self.d_type_x, (self.ub_tensor_size,), name='tmp_res_ub', scope=tik.scope_ubuf
            )
            input_x_ub = self.tik_instance.Tensor(
                self.d_type_x, (self.ub_tensor_size,), name='input_x_ub', scope=tik.scope_ubuf
            )
            with self.tik_instance.for_range(0, 4) as index:
                # position
                move_input_times = 4 * i + index
                index_reg = self.tik_instance.Scalar(dtype="int32")
                index_reg.set_as(input_position_ub[move_input_times])
                theta_reg = self.tik_instance.Scalar(dtype=self.d_type_x)
                theta_reg.set_as(input_ceof_ub[move_input_times])

                batch_id = (already_process_ceof + move_input_times) // 4 // self.output_hw
                src_start_index = batch_id * self.input_hw * self.shape[1] * self.shape[4] + index_reg

                # first loop handle
                with self.tik_instance.if_scope(index == 0):
                    # move to tmp_res vec_mul
                    self.move_input_from_gm_2_ub(tmp_res_ub, src_start_index)
                    self.input_muls(need_hand_last, repeats, repeats_nums, theta_reg, tmp_res_ub, last_num)
                with self.tik_instance.else_scope():
                    # move input_x_ub
                    self.move_input_from_gm_2_ub(input_x_ub, src_start_index)
                    self.input_muls(need_hand_last, repeats, repeats_nums, theta_reg, input_x_ub, last_num)
                    self.input_add(need_hand_last, repeats, repeats_nums, tmp_res_ub, input_x_ub, tmp_res_ub, last_num)
                with self.tik_instance.if_scope(index == 3):
                    # move res to gm
                    output_start_index = batch_id * self.output_hw * self.shape[1] * self.shape[4] + \
                                         ((move_input_times + already_process_ceof) // 4 - batch_id * self.output_hw) * 16
                    if self.output_stride < 65535:
                        self.tik_instance.data_move(
                            dst=self.output_y_gm[output_start_index], src=tmp_res_ub, sid=0,
                            nburst=self.shape[1], burst=self.shape[4] * self.d_type_bytes_size // 32, src_stride=0,
                            dst_stride=self.output_stride
                        )
                    else:
                        with self.tik_instance.for_range(0, self.shape[1]) as c1_index:
                            self.tik_instance.data_move(
                                dst=self.output_y_gm[output_start_index + c1_index * self.output_hw * self.shape[4]],
                                src=tmp_res_ub[c1_index * self.shape[4]], sid=0,
                                nburst=1, burst=self.shape[4] * self.d_type_bytes_size // 32, src_stride=0,
                                dst_stride=0
                            )

    def input_add(self, need_hand_last, repeats, repeats_nums, src1_ub, src2_ub, dst_ub, last_num):
        self.tik_instance.vec_add(repeats_nums, dst_ub, src1_ub, src2_ub, repeats,
                                  repeats_nums * self.d_type_bytes_size // 32,
                                  repeats_nums * self.d_type_bytes_size // 32,
                                  repeats_nums * self.d_type_bytes_size // 32, )
        if need_hand_last:
            self.tik_instance.vec_add(last_num, dst_ub[repeats_nums * repeats], src1_ub[repeats_nums * repeats],
                                      src2_ub[repeats_nums * repeats], 1, 0, 0, 0)

    def input_muls(self, need_hand_last, repeats, repeats_nums, theta_reg, tmp_res_ub, last_num):
        self.tik_instance.vec_muls(repeats_nums, tmp_res_ub, tmp_res_ub, theta_reg,
                                   repeats, repeats_nums * self.d_type_bytes_size // 32,
                                   repeats_nums * self.d_type_bytes_size // 32)
        if need_hand_last:
            self.tik_instance.vec_muls(last_num, tmp_res_ub[repeats_nums * repeats],
                                       tmp_res_ub[repeats_nums * repeats], theta_reg, 1, 0, 0)

    def move_input_from_gm_2_ub(self, ub_input, start_index):
        if self.input_stride < 65535:
            self.tik_instance.data_move(
                dst=ub_input, src=self.input_x_gm[start_index], sid=0,
                nburst=self.shape[1], burst=self.shape[4] * self.d_type_bytes_size // 32,
                src_stride=self.input_stride, dst_stride=0

            )
        else:
            with self.tik_instance.for_range(0, self.shape[1]) as c1_index:
                self.tik_instance.data_move(
                    dst=ub_input[c1_index * self.shape[4]],
                    src=self.input_x_gm[start_index + self.shape[2] * self.shape[3] * self.shape[4] * c1_index],
                    sid=0,
                    nburst=1, burst=self.shape[4] * self.d_type_bytes_size // 32,
                    src_stride=0, dst_stride=0
                )

    def process_c1_hw(self, input_position_ub, input_theta_ub, already_process_theta, handle_theta_size):
        """
        Multiply-add calculation based on theta and position offset
        """
        # processed_c1_count = input_move_offset
        # processed_c1_count = already_process_theta * self.shape[0] * self.shape[1] // self.theta_size * 16
        # input ub
        input_move_offset = already_process_theta // 4 * self.ub_tensor_size
        tmp_res_ub = self.tik_instance.Tensor(
            self.d_type_x, (self.ub_tensor_size,), name='tmp_res_ub', scope=tik.scope_ubuf
        )
        input_x_ub = self.tik_instance.Tensor(
            self.d_type_x, (self.ub_tensor_size,), name='input_x_ub', scope=tik.scope_ubuf
        )
        with self.tik_instance.for_range(0, handle_theta_size, thread_num=1) as move_input_times:
            # Sequence number of the C1 that has been processed * each c1 data count
            processed_c1_count = (already_process_theta + move_input_times) // (self.output_hw * 4) * self.input_hw * 16
            # position
            index_reg = self.tik_instance.Scalar(dtype="int32")
            index_reg.set_as(input_position_ub[move_input_times])
            theta_reg = self.tik_instance.Scalar(dtype=self.d_type_x)
            theta_reg.set_as(input_theta_ub[move_input_times])

            # first loop handle
            with self.tik_instance.if_scope(move_input_times % 4 == 0):
                # move to tmp_res vec_mul
                self.tik_instance.data_move(
                    dst=tmp_res_ub, src=self.input_x_gm[processed_c1_count + index_reg],
                    sid=0, nburst=1, burst=self.ub_tensor_len, src_stride=0, dst_stride=0
                )
                self.tik_instance.vec_muls(16, tmp_res_ub, tmp_res_ub, theta_reg, 1, 0, 0)
            with self.tik_instance.else_scope():
                # move input_x_ub
                self.tik_instance.data_move(
                    dst=input_x_ub, src=self.input_x_gm[processed_c1_count + index_reg],
                    sid=0, nburst=1, burst=self.ub_tensor_len, src_stride=0, dst_stride=0
                )
                self.tik_instance.vec_muls(16, input_x_ub, input_x_ub, theta_reg, 1, 0, 0)
                self.tik_instance.vec_add(16, tmp_res_ub, tmp_res_ub, input_x_ub, 1, 0, 0, 0)
            with self.tik_instance.if_scope((move_input_times + 1) % 4 == 0):
                # move res to gm

                self.tik_instance.data_move(
                    dst=self.output_y_gm[input_move_offset + (move_input_times // 4) * 16], src=tmp_res_ub, sid=0,
                    nburst=1, burst=self.ub_tensor_len, src_stride=0, dst_stride=0
                )

    # calc first loop count
    def _calc_loop_count(self, ceof_size):
        loop = ceof_size // 4 // self.ai_core_num
        return ceof_size // 4 % self.ai_core_num if loop < 1 else self.ai_core_num

    def ceil_value(self, value, factor):
        """
        if not divide exactly then plus 1

        Parameters
        ----------
        value:  input number
        factor: factor

        Returns
        -------
        ceil value
        """
        return (value + factor - 1) // factor
