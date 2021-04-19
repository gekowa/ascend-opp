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
stn_pre
"""
import math
from topi.cce import util
from te import tik
from te import platform as cce
from te import platform as tbe_platform
from te.utils.op_utils import check_op_params
from te.utils.op_utils import OPTION_INPUT
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import OPTION_ATTR_LIST_INT
from te.utils.op_utils import OPTION_ATTR_LIST_FLOAT
from te.utils.op_utils import OPTION_ATTR_LIST_BOOL
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_OUTPUT


@check_op_params(OPTION_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_LIST_INT, OPTION_ATTR_LIST_FLOAT,
                 OPTION_ATTR_LIST_BOOL, OPTION_ATTR_BOOL, KERNEL_NAME)
def stn_pre(theta, w_index, h_index, pos_coef, pos_offset,
            size=(-1, -1, -1, -1), default_theta=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            use_default_theta=(False, False, False, False, False, False),
            align_corners=False, kernel_name='stn_pre'):
    """
    spatial transformer pre

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    w_index: dict
        w index matrix
    h_index: dict
        h index matrix
    pos_coef: dict
        output ceof matrix
    pos_offset: dict
        output offset matrix
    size: tuple
        output size
    default_theta: tuple
        default theta
    use_default_theta: list
        use default theta
    align_corners:
        align corners
    kernel_name : str
        kernel name, default value is "stn_pre"

    Returns
    -------
    None
    """

    shape = theta.get("shape")

    util.check_shape_rule(shape)
    util.check_tensor_shape_size(shape)
    util.check_kernel_name(kernel_name)

    stn_instance = SpatialTransformer(theta, w_index, h_index, pos_coef, pos_offset, size,
                                      default_theta, use_default_theta, kernel_name)
    stn_instance.stn_pre_compute()
    return stn_instance


class SpatialTransformer:
    def __init__(self, theta, w_index, h_index, pos_coef,
                 pos_offset, size, default_theta, use_default_theta, kernel_name='stn_pre'):

        self.kernel_name = kernel_name
        self.default_theta = default_theta
        self.use_default_theta = use_default_theta

        self.tik_instance = tik.Tik(tik.Dprofile())
        self.calc_by_fp16 = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS")
        self.ai_core_num = tik.Dprofile().get_aicore_num()

        ub_size_bytes = cce.get_soc_spec(cce.cce_conf.UB_SIZE) // 2
        self.theta_type_bytes_size = cce.cce_intrin.get_bit_len(theta.get('dtype')) // 8
        self.h_w_index_type_byte_size = cce.cce_intrin.get_bit_len(w_index.get('dtype')) // 8
        self.batch = pos_coef.get('shape')[0]
        self.total_c1 = pos_coef.get('shape')[1]
        self.if_skip_read_ceof = self.total_c1 * 16 * self.theta_type_bytes_size <= ub_size_bytes * 0.4
        self.output_hw_size = size[2] * size[3]
        self.pos_coef = pos_coef

        self.w_index = w_index
        self.w_index_ub_size = 128  # Tentative 128
        self.h_index = h_index
        self.size = size
        self.theta = theta

        # input theta gm
        self.theta_gm = self.tik_instance.Tensor(
            theta.get('dtype'), (theta.get('shape')[0] * theta.get('shape')[1] + 28,),
            name='theta_gm', scope=tik.scope_gm
        )
        # input w_index gm
        self.w_index_gm = self.tik_instance.Tensor(
            w_index.get('dtype'), (size[2] * size[3] + 512,), name='w_index_gm', scope=tik.scope_gm
        )
        # input w_index gm
        self.h_index_gm = self.tik_instance.Tensor(
            h_index.get('dtype'), (size[2] * size[3] + 512,), name='h_index_gm', scope=tik.scope_gm
        )

        # output pos_coef
        self.output_pos_coef_gm = self.tik_instance.Tensor(
            pos_coef.get('dtype'),
            (pos_coef.get('shape')[0] * pos_coef.get('shape')[1] * pos_coef.get('shape')[2] + 512,),
            name='output_pos_coef_gm', scope=tik.scope_gm
        )
        # output pos_offset
        self.output_pos_offset_gm = self.tik_instance.Tensor(
            pos_offset.get('dtype'),
            (pos_offset.get('shape')[0] * pos_offset.get('shape')[1] * pos_offset.get('shape')[2] + 512,),
            name='output_pos_offset_gm', scope=tik.scope_gm
        )

    def stn_pre_compute(self):
        # tiling by batch
        if self.batch > self.ai_core_num:
            block_num = self.ai_core_num
            each_core_loop_count = self.batch // self.ai_core_num
            each_core_last_batch = self.batch % self.ai_core_num
        else:
            block_num = self.batch
            each_core_loop_count = 0
            each_core_last_batch = self.batch
        each_batch_loop_count = self.output_hw_size // self.w_index_ub_size
        with self.tik_instance.for_range(0, block_num, block_num=block_num) as core_index:
            # each core process
            with self.tik_instance.for_range(0, each_core_loop_count) as loop_index:
                batch_id = block_num * loop_index + core_index
                self.process_on_each_core(batch_id, each_batch_loop_count)

            if each_core_last_batch > 0:
                batch_id = each_core_loop_count * block_num + core_index
                with self.tik_instance.if_scope(core_index < each_core_last_batch):
                    self.process_on_each_core(batch_id, each_batch_loop_count)
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.theta_gm, self.w_index_gm, self.h_index_gm],
            outputs=[self.output_pos_coef_gm, self.output_pos_offset_gm])

        return self.tik_instance

    def process_on_each_core(self, batch_id, each_batch_loop_count):
        theta1_1, theta1_2, theta1_3, theta2_1, theta2_2, theta2_3 = self.calc_theta(batch_id, "float16") if \
            self.calc_by_fp16 else self.calc_theta(batch_id, "float32")

        thread_num = 2 if each_batch_loop_count >= 2 else 1
        with self.tik_instance.for_range(0, each_batch_loop_count, thread_num=thread_num) as hw_index_handle_count:
            # one loop one batch
            # w index tensor
            w_index_ub = self.tik_instance.Tensor(
                self.w_index.get('dtype'), (self.w_index_ub_size,), name='w_index_ub', scope=tik.scope_ubuf
            )
            self.tik_instance.data_move(
                dst=w_index_ub,
                src=self.w_index_gm[hw_index_handle_count * self.w_index_ub_size],
                sid=0, nburst=1, burst=self.w_index_ub_size * self.h_w_index_type_byte_size // 32,
                src_stride=0, dst_stride=0
            )
            # h index tensor
            h_index_ub = self.tik_instance.Tensor(
                self.h_index.get('dtype'), (self.w_index_ub_size,), name='h_index_ub', scope=tik.scope_ubuf
            )
            self.tik_instance.data_move(
                dst=h_index_ub,
                src=self.h_index_gm[hw_index_handle_count * self.w_index_ub_size],
                sid=0, nburst=1, burst=self.w_index_ub_size * self.h_w_index_type_byte_size // 32,
                src_stride=0, dst_stride=0
            )
            if self.calc_by_fp16 or self.w_index.get('dtype') == 'float32':
                ch_cw_res_ceof_ub, ch_cw_res_offset_ub, ch_fw_res_ceof_ub, ch_fw_res_offset_ub, fh_cw_res_ceof_ub, \
                fh_cw_res_offset_ub, fh_fw_res_ceof_ub, fh_fw_res_offset_ub = self.calc_ceof_and_offset(
                    h_index_ub, theta1_1, theta1_2, theta1_3, theta2_1, theta2_2, theta2_3, w_index_ub)
            else:
                h_index_ub_fp32 = self.tik_instance.Tensor(
                    'float32', (self.w_index_ub_size,), name='h_index_ub_fp32', scope=tik.scope_ubuf
                )
                w_index_ub_fp32 = self.tik_instance.Tensor(
                    'float32', (self.w_index_ub_size,), name='w_index_ub_fp32', scope=tik.scope_ubuf
                )
                self.tik_instance.vec_conv(64, '', h_index_ub_fp32, h_index_ub, 2, 8, 4)
                self.tik_instance.vec_conv(64, '', w_index_ub_fp32, w_index_ub, 2, 8, 4)
                ch_cw_res_ceof_ub, ch_cw_res_offset_ub, ch_fw_res_ceof_ub, ch_fw_res_offset_ub, fh_cw_res_ceof_ub, \
                fh_cw_res_offset_ub, fh_fw_res_ceof_ub, fh_fw_res_offset_ub = self.calc_ceof_and_offset(
                    h_index_ub_fp32, theta1_1, theta1_2, theta1_3, theta2_1, theta2_2, theta2_3, w_index_ub_fp32)

            # change 4 * 128 to 128 * 4
            with self.tik_instance.new_stmt_scope():
                ceof_res_ub, offset_res_ub = self.merge_res(ch_cw_res_ceof_ub, ch_cw_res_offset_ub, ch_fw_res_ceof_ub,
                                                            ch_fw_res_offset_ub, fh_cw_res_ceof_ub, fh_cw_res_offset_ub,
                                                            fh_fw_res_ceof_ub, fh_fw_res_offset_ub)
                if self.if_skip_read_ceof:
                    output_index = batch_id * self.output_hw_size * 4 + hw_index_handle_count * 512
                    # move ceof to gm
                    self.tik_instance.data_move(
                        dst=self.output_pos_coef_gm[output_index], src=ceof_res_ub, sid=0,
                        nburst=1, burst=4 * 128 * self.h_w_index_type_byte_size // 32, src_stride=0, dst_stride=0
                    )
                    # move offset to gm
                    self.tik_instance.data_move(
                        dst=self.output_pos_offset_gm[output_index], src=offset_res_ub, sid=0,
                        nburst=1, burst=4 * 128 * 4 // 32, src_stride=0, dst_stride=0
                    )
                else:
                    with self.tik_instance.for_range(0, self.total_c1) as c1_count:
                        output_index = batch_id * self.total_c1 * self.output_hw_size * 4 + \
                                       c1_count * self.output_hw_size * 4 + hw_index_handle_count * 512
                        # move ceof to gm
                        self.tik_instance.data_move(
                            dst=self.output_pos_coef_gm[output_index], src=ceof_res_ub, sid=0,
                            nburst=1, burst=4 * 128 * self.h_w_index_type_byte_size // 32, src_stride=0, dst_stride=0
                        )
                        # move offset to gm
                        self.tik_instance.data_move(
                            dst=self.output_pos_offset_gm[output_index], src=offset_res_ub, sid=0,
                            nburst=1, burst=4 * 128 * 4 // 32, src_stride=0, dst_stride=0
                        )
        each_batch_loop_last_num = self.output_hw_size % self.w_index_ub_size
        if each_batch_loop_last_num > 0:
            # w index tensor
            hw_index_handle_count = self.output_hw_size - each_batch_loop_last_num
            w_index_ub = self.tik_instance.Tensor(
                self.w_index.get('dtype'), (self.w_index_ub_size,), name='w_index_ub', scope=tik.scope_ubuf
            )
            burst = self.w_index_ub_size * self.h_w_index_type_byte_size // 32
            self.tik_instance.data_move(
                dst=w_index_ub,
                src=self.w_index_gm[hw_index_handle_count],
                sid=0, nburst=1, burst=burst, src_stride=0, dst_stride=0
            )
            # h index tensor
            h_index_ub = self.tik_instance.Tensor(
                self.h_index.get('dtype'), (self.w_index_ub_size,), name='h_index_ub', scope=tik.scope_ubuf
            )
            self.tik_instance.data_move(
                dst=h_index_ub,
                src=self.h_index_gm[hw_index_handle_count],
                sid=0, nburst=1, burst=burst, src_stride=0, dst_stride=0
            )
            if self.calc_by_fp16 or self.w_index.get('dtype') == 'float32':
                ch_cw_res_ceof_ub, ch_cw_res_offset_ub, ch_fw_res_ceof_ub, ch_fw_res_offset_ub, fh_cw_res_ceof_ub, \
                fh_cw_res_offset_ub, fh_fw_res_ceof_ub, fh_fw_res_offset_ub = self.calc_ceof_and_offset(
                    h_index_ub, theta1_1, theta1_2, theta1_3, theta2_1, theta2_2, theta2_3, w_index_ub)
            else:
                h_index_ub_fp32 = self.tik_instance.Tensor(
                    'float32', (self.w_index_ub_size,), name='h_index_ub_fp32', scope=tik.scope_ubuf
                )
                w_index_ub_fp32 = self.tik_instance.Tensor(
                    'float32', (self.w_index_ub_size,), name='w_index_ub_fp32', scope=tik.scope_ubuf
                )
                self.tik_instance.vec_conv(64, '', h_index_ub_fp32, h_index_ub, 2, 8, 4)
                self.tik_instance.vec_conv(64, '', w_index_ub_fp32, w_index_ub, 2, 8, 4)
                ch_cw_res_ceof_ub, ch_cw_res_offset_ub, ch_fw_res_ceof_ub, ch_fw_res_offset_ub, fh_cw_res_ceof_ub, \
                fh_cw_res_offset_ub, fh_fw_res_ceof_ub, fh_fw_res_offset_ub = self.calc_ceof_and_offset(
                    h_index_ub_fp32, theta1_1, theta1_2, theta1_3, theta2_1, theta2_2, theta2_3, w_index_ub_fp32)

            # change 4 * 128 to 128 * 4
            with self.tik_instance.new_stmt_scope():
                ceof_res_ub, offset_res_ub = self.merge_res(ch_cw_res_ceof_ub, ch_cw_res_offset_ub, ch_fw_res_ceof_ub,
                                                            ch_fw_res_offset_ub, fh_cw_res_ceof_ub, fh_cw_res_offset_ub,
                                                            fh_fw_res_ceof_ub, fh_fw_res_offset_ub)
                if self.if_skip_read_ceof:
                    output_index = batch_id * self.output_hw_size * 4 + hw_index_handle_count * 4
                    # move ceof to gm
                    last_num_size = 4 * each_batch_loop_last_num
                    tail_num = last_num_size * self.h_w_index_type_byte_size % 32
                    burst_len = math.ceil(last_num_size * self.h_w_index_type_byte_size / 32)
                    if burst_len > 1 and tail_num > 0:
                        block_num_size = 32 // self.h_w_index_type_byte_size
                        block_ub = self.tik_instance.Tensor(
                            self.pos_coef.get('dtype'), (block_num_size,), name='block_ub', scope=tik.scope_ubuf
                        )
                        with self.tik_instance.for_range(0, block_num_size) as index:
                            block_ub[index].set_as(ceof_res_ub[last_num_size - block_num_size + index])
                        self.tik_instance.data_move(
                            dst=self.output_pos_coef_gm[output_index], src=ceof_res_ub, sid=0,
                            nburst=1, burst=burst_len - 1, src_stride=0, dst_stride=0
                        )

                        self.tik_instance.data_move(
                            dst=self.output_pos_coef_gm[output_index + last_num_size - block_num_size], src=block_ub,
                            sid=0,
                            nburst=1, burst=1, src_stride=0, dst_stride=0
                        )
                    else:
                        self.tik_instance.data_move(
                            dst=self.output_pos_coef_gm[output_index], src=ceof_res_ub, sid=0,
                            nburst=1, burst=burst_len, src_stride=0, dst_stride=0
                        )
                    # move offset to gm
                    tail_num = last_num_size % 8
                    burst_len = math.ceil(last_num_size / 8)
                    if burst_len > 1 and tail_num > 0:
                        block_int32_ub = self.tik_instance.Tensor(
                            'int32', (8,), name='block_int32_ub', scope=tik.scope_ubuf
                        )
                        with self.tik_instance.for_range(0, 8) as index:
                            block_int32_ub[index].set_as(offset_res_ub[last_num_size - 8 + index])
                        self.tik_instance.data_move(
                            dst=self.output_pos_offset_gm[output_index], src=offset_res_ub, sid=0,
                            nburst=1, burst=burst_len - 1, src_stride=0, dst_stride=0
                        )
                        self.tik_instance.data_move(
                            dst=self.output_pos_offset_gm[output_index + last_num_size - 8], src=block_int32_ub, sid=0,
                            nburst=1, burst=1, src_stride=0, dst_stride=0
                        )
                    else:
                        self.tik_instance.data_move(
                            dst=self.output_pos_offset_gm[output_index], src=offset_res_ub, sid=0,
                            nburst=1, burst=burst_len, src_stride=0, dst_stride=0
                        )
                else:
                    with self.tik_instance.for_range(0, self.total_c1) as c1_count:
                        output_index = (batch_id * self.total_c1 + c1_count) * self.output_hw_size * 4 \
                                       + hw_index_handle_count * 4
                        # move ceof to gm
                        last_num_size = 4 * each_batch_loop_last_num
                        tail_num = last_num_size * self.h_w_index_type_byte_size % 32
                        burst_len = math.ceil(last_num_size * self.h_w_index_type_byte_size / 32)
                        if burst_len > 1 and tail_num > 0:
                            block_num_size = 32 // self.h_w_index_type_byte_size
                            block_ub = self.tik_instance.Tensor(
                                self.pos_coef.get('dtype'), (block_num_size,), name='block_ub', scope=tik.scope_ubuf
                            )
                            with self.tik_instance.for_range(0, block_num_size) as index:
                                block_ub[index].set_as(ceof_res_ub[last_num_size - block_num_size + index])
                            self.tik_instance.data_move(
                                dst=self.output_pos_coef_gm[output_index], src=ceof_res_ub, sid=0,
                                nburst=1, burst=burst_len - 1, src_stride=0, dst_stride=0
                            )

                            self.tik_instance.data_move(
                                dst=self.output_pos_coef_gm[output_index + last_num_size - block_num_size],
                                src=block_ub,
                                sid=0,
                                nburst=1, burst=1, src_stride=0, dst_stride=0
                            )
                        else:
                            self.tik_instance.data_move(
                                dst=self.output_pos_coef_gm[output_index], src=ceof_res_ub, sid=0,
                                nburst=1, burst=burst_len, src_stride=0, dst_stride=0
                            )
                        # move offset to gm
                        tail_num = last_num_size % 8
                        burst_len = math.ceil(last_num_size / 8)
                        if burst_len > 1 and tail_num > 0:
                            block_int32_ub = self.tik_instance.Tensor(
                                'int32', (8,), name='block_int32_ub', scope=tik.scope_ubuf
                            )
                            with self.tik_instance.for_range(0, 8) as index:
                                block_int32_ub[index].set_as(offset_res_ub[last_num_size - 8 + index])
                            self.tik_instance.data_move(
                                dst=self.output_pos_offset_gm[output_index], src=offset_res_ub, sid=0,
                                nburst=1, burst=burst_len - 1, src_stride=0, dst_stride=0
                            )
                            self.tik_instance.data_move(
                                dst=self.output_pos_offset_gm[output_index + last_num_size - 8], src=block_int32_ub,
                                sid=0,
                                nburst=1, burst=1, src_stride=0, dst_stride=0
                            )
                        else:
                            self.tik_instance.data_move(
                                dst=self.output_pos_offset_gm[output_index], src=offset_res_ub, sid=0,
                                nburst=1, burst=burst_len, src_stride=0, dst_stride=0
                            )

    def merge_res(self, ch_cw_res_ceof_ub, ch_cw_res_offset_ub, ch_fw_res_ceof_ub, ch_fw_res_offset_ub,
                  fh_cw_res_ceof_ub, fh_cw_res_offset_ub, fh_fw_res_ceof_ub, fh_fw_res_offset_ub):
        offset_res_ub = self.tik_instance.Tensor(
            'int32', (128 * 4,), name='offset_res_ub', scope=tik.scope_ubuf
        )
        ceof_res_ub = self.tik_instance.Tensor(
            self.pos_coef.get('dtype'), (128 * 4,), name='ceof_res_ub', scope=tik.scope_ubuf
        )
        if self.pos_coef.get('dtype') == 'float16':
            with self.tik_instance.for_range(0, 128, thread_num=2) as i:
                index = 4 * i
                ceof_res_ub[index].set_as(fh_fw_res_ceof_ub[i])
                ceof_res_ub[index + 1].set_as(fh_cw_res_ceof_ub[i])
                ceof_res_ub[index + 2].set_as(ch_fw_res_ceof_ub[i])
                ceof_res_ub[index + 3].set_as(ch_cw_res_ceof_ub[i])
                offset_res_ub[index].set_as(fh_fw_res_offset_ub[i])
                offset_res_ub[index + 1].set_as(fh_cw_res_offset_ub[i])
                offset_res_ub[index + 2].set_as(ch_fw_res_offset_ub[i])
                offset_res_ub[index + 3].set_as(ch_cw_res_offset_ub[i])
        else:
            fh_fw_res_ceof_ub32 = self.tik_instance.Tensor(
                'float32', (128,), name='fh_fw_res_ceof_ub32', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_conv(64, '', fh_fw_res_ceof_ub32, fh_fw_res_ceof_ub, 2, 8, 4)
            fh_cw_res_ceof_ub32 = self.tik_instance.Tensor(
                'float32', (128,), name='fh_fw_res_ceof_ub32', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_conv(64, '', fh_cw_res_ceof_ub32, fh_cw_res_ceof_ub, 2, 8, 4)
            ch_fw_res_ceof_ub32 = self.tik_instance.Tensor(
                'float32', (128,), name='ch_fw_res_ceof_ub32', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_conv(64, '', ch_fw_res_ceof_ub32, ch_fw_res_ceof_ub, 2, 8, 4)
            ch_cw_res_ceof_ub32 = self.tik_instance.Tensor(
                'float32', (128,), name='ch_fw_res_ceof_ub32', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_conv(64, '', ch_cw_res_ceof_ub32, ch_cw_res_ceof_ub, 2, 8, 4)
            with self.tik_instance.for_range(0, 128, thread_num=2) as i:
                index = 4 * i
                ceof_res_ub[index].set_as(fh_fw_res_ceof_ub32[i])
                ceof_res_ub[index + 1].set_as(fh_cw_res_ceof_ub32[i])
                ceof_res_ub[index + 2].set_as(ch_fw_res_ceof_ub32[i])
                ceof_res_ub[index + 3].set_as(ch_cw_res_ceof_ub32[i])
                offset_res_ub[index].set_as(fh_fw_res_offset_ub[i])
                offset_res_ub[index + 1].set_as(fh_cw_res_offset_ub[i])
                offset_res_ub[index + 2].set_as(ch_fw_res_offset_ub[i])
                offset_res_ub[index + 3].set_as(ch_cw_res_offset_ub[i])
        return ceof_res_ub, offset_res_ub

    def calc_ceof_and_offset(self, h_index_ub, theta1_1, theta1_2, theta1_3, theta2_1, theta2_2, theta2_3, w_index_ub):
        # (theta1-1,theta1-2,theta1-3) dot ([h], [w], [1])
        origin_h_or_res_ub, origin_w_ub = self.calc_origin_h(h_index_ub, theta1_1, theta1_2, theta1_3, w_index_ub)
        # (theta2-1,theta1-2,theta1-3) dot ([h], [w], [1])
        origin_w_or_res_ub = self.calc_origin_w(h_index_ub, origin_w_ub, theta2_1, theta2_2, theta2_3, w_index_ub)
        if not self.calc_by_fp16:
            origin_w_or_res_ub16 = self.tik_instance.Tensor(
                "float16", (self.w_index_ub_size,), name='origin_w_or_res_ub16', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_conv(64, '', origin_w_or_res_ub16, origin_w_or_res_ub, 2, 4, 8)
            origin_h_res_ub16 = self.tik_instance.Tensor(
                "float16", (self.w_index_ub_size,), name='origin_h_res_ub16', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_conv(64, '', origin_h_res_ub16, origin_h_or_res_ub, 2, 4, 8)

            return self.mian_calc_ceof_offset(origin_h_or_res_ub, origin_w_or_res_ub, origin_h_res_ub16,
                                              origin_w_or_res_ub16)
        else:
            return self.mian_calc_ceof_offset(origin_h_or_res_ub, origin_w_or_res_ub, origin_h_or_res_ub,
                                              origin_w_or_res_ub)

    def mian_calc_ceof_offset(self, origin_h_or_res_ub, origin_w_or_res_ub, origin_h_res_ub16, origin_w_res_ub16):
        # floor origin h
        floor_h_fp16_ub, floor_h_ub_int32 = self.calc_floor_h(origin_h_res_ub16)
        # floor origin w
        floor_w_ub, floor_w_ub_int32 = self.calc_floor_w(origin_w_res_ub16)
        # ceil origin h
        ceil_h_ub, ceil_h_ub_int32 = self.calc_ceil_h(origin_h_res_ub16)
        # ceil origin w
        ceil_w_ub, ceil_w_ub_int32 = self.calc_ceil_w(origin_w_res_ub16)
        # calc (floor(h), floor(w)): (1 - abs(origin_h - floor(h))) * (1 - abs(origin_w - floor(w)))
        fh_fw_res_ceof_ub = self.tik_instance.Tensor(
            'float16', (128,), name='tmp_res_ceof_ub', scope=tik.scope_ubuf
        )
        if self.calc_by_fp16:
            self.calc_ceof(floor_w_ub, origin_h_res_ub16, origin_w_res_ub16, floor_h_fp16_ub, fh_fw_res_ceof_ub)
        else:
            self.calc_ceof_by_fp32(floor_w_ub, origin_h_or_res_ub, origin_w_or_res_ub, floor_h_fp16_ub, fh_fw_res_ceof_ub)
        # calc offset (floor(h), floor(w)): floor(h) * W * 16 + floor(w) * 16
        fh_fw_res_offset_ub = self.tik_instance.Tensor(
            'int32', (128,), name='fh_fw_res_offset_ub', scope=tik.scope_ubuf
        )
        int32_16_ub = self.tik_instance.Tensor(
            'int32', (128,), name='int32_16_ub', scope=tik.scope_ubuf
        )
        with self.tik_instance.new_stmt_scope():
            tmp_fh_fw_res_offset_ub = self.tik_instance.Tensor(
                'int32', (128,), name='tmp_fh_fw_res_offset_ub', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_dup(64, int32_16_ub, 16 * self.size[1], 2, 8)
            self.tik_instance.vec_mul(64, fh_fw_res_offset_ub, floor_h_ub_int32, int32_16_ub, 2, 8, 8, 8)
            self.tik_instance.vec_dup(64, int32_16_ub, 16, 2, 8)
            self.tik_instance.vec_mul(64, tmp_fh_fw_res_offset_ub, floor_w_ub_int32, int32_16_ub, 2, 8, 8, 8)
            self.tik_instance.vec_add(64, fh_fw_res_offset_ub, fh_fw_res_offset_ub, tmp_fh_fw_res_offset_ub, 2, 8, 8, 8)
        # filter ceof and offset
        float_filter_ub = self.tik_instance.Tensor(
            "float16", (128,), name='float_filter_ub', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_dup(128, float_filter_ub, 0.0, 1, 1)
        filter_index = self.tik_instance.Tensor(
            'uint16', (8,), name='filter_index', scope=tik.scope_ubuf
        )
        self.filter_coef(fh_fw_res_ceof_ub, float_filter_ub, floor_h_fp16_ub, floor_w_ub, filter_index,
                         fh_fw_res_offset_ub)
        # calc (fh, cw)
        fh_cw_res_ceof_ub = self.tik_instance.Tensor(
            'float16', (128,), name='fh_cw_res_ceof_ub', scope=tik.scope_ubuf
        )
        if self.calc_by_fp16:
            self.calc_ceof(ceil_w_ub, origin_h_res_ub16, origin_w_res_ub16, floor_h_fp16_ub, fh_cw_res_ceof_ub)
        else:
            self.calc_ceof_by_fp32(ceil_w_ub, origin_h_or_res_ub, origin_w_or_res_ub, floor_h_fp16_ub, fh_cw_res_ceof_ub)
        # calc offset (floor(h), ceil(w)): floor(h) * W * 16 + ceil(w) * 16
        fh_cw_res_offset_ub = self.tik_instance.Tensor(
            'int32', (128,), name='fh_cw_res_offset_ub', scope=tik.scope_ubuf
        )
        with self.tik_instance.new_stmt_scope():
            tmp_fh_cw_res_offset_ub = self.tik_instance.Tensor(
                'int32', (128,), name='tmp_fh_cw_res_offset_ub', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_mul(64, tmp_fh_cw_res_offset_ub, ceil_w_ub_int32, int32_16_ub, 2, 8, 8, 8)
            self.tik_instance.vec_dup(64, int32_16_ub, 16 * self.size[1], 2, 8)
            self.tik_instance.vec_mul(64, fh_cw_res_offset_ub, floor_h_ub_int32, int32_16_ub, 2, 8, 8, 8)
            self.tik_instance.vec_add(64, fh_cw_res_offset_ub, fh_cw_res_offset_ub, tmp_fh_cw_res_offset_ub, 2, 8, 8, 8)
        self.tik_instance.vec_dup(128, float_filter_ub, 0.0, 1, 1)
        self.filter_coef(fh_cw_res_ceof_ub, float_filter_ub, floor_h_fp16_ub, ceil_w_ub, filter_index,
                         fh_cw_res_offset_ub)
        # calc (ch, fw)
        ch_fw_res_ceof_ub = self.tik_instance.Tensor(
            'float16', (128,), name='ch_fw_res_ceof_ub', scope=tik.scope_ubuf
        )
        if self.calc_by_fp16:
            self.calc_ceof(floor_w_ub, origin_h_res_ub16, origin_w_res_ub16, ceil_h_ub, ch_fw_res_ceof_ub)
        else:
            self.calc_ceof_by_fp32(floor_w_ub, origin_h_or_res_ub, origin_w_or_res_ub, ceil_h_ub, ch_fw_res_ceof_ub)
        # calc offset (ceil(h), floor(w)): ceil(h) * W * 16 + floor(w) * 16
        ch_fw_res_offset_ub = self.tik_instance.Tensor(
            'int32', (128,), name='ch_fw_res_offset_ub', scope=tik.scope_ubuf
        )
        with self.tik_instance.new_stmt_scope():
            tmp_ch_fw_res_offset_ub = self.tik_instance.Tensor(
                'int32', (128,), name='tmp_ch_fw_res_offset_ub', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_mul(64, ch_fw_res_offset_ub, ceil_h_ub_int32, int32_16_ub, 2, 8, 8, 8)
            self.tik_instance.vec_dup(64, int32_16_ub, 16, 2, 8)
            self.tik_instance.vec_mul(64, tmp_ch_fw_res_offset_ub, floor_w_ub_int32, int32_16_ub, 2, 8, 8, 8)
            self.tik_instance.vec_add(64, ch_fw_res_offset_ub, ch_fw_res_offset_ub, tmp_ch_fw_res_offset_ub, 2, 8, 8, 8)
        self.filter_coef(ch_fw_res_ceof_ub, float_filter_ub, ceil_h_ub, floor_w_ub, filter_index, ch_fw_res_offset_ub)
        # calc (ch, cw)
        ch_cw_res_ceof_ub = self.tik_instance.Tensor(
            'float16', (128,), name='ch_cw_res_ceof_ub', scope=tik.scope_ubuf
        )
        if self.calc_by_fp16:
            self.calc_ceof(ceil_w_ub, origin_h_res_ub16, origin_w_res_ub16, ceil_h_ub, ch_cw_res_ceof_ub)
        else:
            self.calc_ceof_by_fp32(ceil_w_ub, origin_h_or_res_ub, origin_w_or_res_ub, ceil_h_ub, ch_cw_res_ceof_ub)
        # calc offset (ceil(h), floor(w)): ceil(h) * W * 16 + ceil(w) * 16
        ch_cw_res_offset_ub = self.tik_instance.Tensor(
            'int32', (128,), name='ch_cw_res_offset_ub', scope=tik.scope_ubuf
        )
        with self.tik_instance.new_stmt_scope():
            tmp_ch_cw_res_offset_ub = self.tik_instance.Tensor(
                'int32', (128,), name='tmp_ch_cw_res_offset_ub', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_mul(64, tmp_ch_cw_res_offset_ub, ceil_w_ub_int32, int32_16_ub, 2, 8, 8, 8)
            self.tik_instance.vec_dup(64, int32_16_ub, 16 * self.size[1], 2, 8)
            self.tik_instance.vec_mul(64, ch_cw_res_offset_ub, ceil_h_ub_int32, int32_16_ub, 2, 8, 8, 8)
            self.tik_instance.vec_add(64, ch_cw_res_offset_ub, ch_cw_res_offset_ub, tmp_ch_cw_res_offset_ub, 2, 8, 8, 8)
        self.filter_coef(ch_cw_res_ceof_ub, float_filter_ub, ceil_h_ub, ceil_w_ub, filter_index, ch_cw_res_offset_ub)
        # if floor h == ceil h set ch_fw, ch_cw as 0.0
        self.tik_instance.vec_cmpv_eq(filter_index, floor_h_fp16_ub, ceil_h_ub, 1, 8, 8)
        self.tik_instance.vec_dup(128, float_filter_ub, 0.0, 1, 1)
        ch_fw_ceof_ub = self.tik_instance.Tensor(
            'float16', (128,), name='ch_fw_ceof_ub', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_sel(128, 0, ch_fw_ceof_ub, filter_index, float_filter_ub, ch_fw_res_ceof_ub, 1, 0,
                                  0, 0)
        ch_cw_ceof_1_ub = self.tik_instance.Tensor(
            'float16', (128,), name='ch_cw_ceof_1_ub', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_sel(128, 0, ch_cw_ceof_1_ub, filter_index, float_filter_ub, ch_cw_res_ceof_ub, 1, 0,
                                  0, 0)
        # if floor w == ceil w  set fh_cw, ch_cw as 0.0
        filter_index_tmp = self.tik_instance.Tensor(
            'uint16', (8,), name='filter_index', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_cmpv_eq(filter_index_tmp, floor_w_ub, ceil_w_ub, 1, 8, 8)
        self.tik_instance.vec_dup(128, float_filter_ub, 0.0, 1, 1)
        fh_cw_ceof_ub = self.tik_instance.Tensor(
            'float16', (128,), name='fh_cw_ceof_ub', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_sel(128, 0, fh_cw_ceof_ub, filter_index_tmp, float_filter_ub, fh_cw_res_ceof_ub, 1, 8,
                                  8, 8)
        self.tik_instance.vec_sel(128, 0, ch_cw_res_ceof_ub, filter_index_tmp, float_filter_ub, ch_cw_ceof_1_ub, 1, 0,
                                  0, 0)
        return ch_cw_res_ceof_ub, ch_cw_res_offset_ub, ch_fw_ceof_ub, ch_fw_res_offset_ub, fh_cw_ceof_ub, \
               fh_cw_res_offset_ub, fh_fw_res_ceof_ub, fh_fw_res_offset_ub

    def calc_ceil_w(self, origin_w_or_res_ub):
        ceil_w_ub_int32 = self.tik_instance.Tensor(
            'int32', (128,), name='ceil_w_ub_int32', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_conv(64, 'ceil', ceil_w_ub_int32, origin_w_or_res_ub, 2, 8, 4)
        ceil_w_ub = self.tik_instance.Tensor(
            'float16', (128,), name='ceil_w_ub', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_conv(64, '', ceil_w_ub, ceil_w_ub_int32, 2, 4, 8, 1.0)
        return ceil_w_ub, ceil_w_ub_int32

    def calc_ceil_h(self, origin_h_or_res_ub):
        ceil_h_ub_int32 = self.tik_instance.Tensor(
            'int32', (128,), name='ceil_h_ub_int32', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_conv(64, 'ceil', ceil_h_ub_int32, origin_h_or_res_ub, 2, 8, 4)
        ceil_h_ub = self.tik_instance.Tensor(
            'float16', (128,), name='ceil_h_ub', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_conv(64, '', ceil_h_ub, ceil_h_ub_int32, 2, 4, 8, 1.0)
        return ceil_h_ub, ceil_h_ub_int32

    def calc_floor_w(self, origin_w_or_res_ub):
        floor_w_ub_int32 = self.tik_instance.Tensor(
            'int32', (128,), name='floor_w_ub_int32', scope=tik.scope_ubuf
        )
        floor_w_ub = self.tik_instance.Tensor(
            'float16', (128,), name='floor_w_ub', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_conv(64, 'floor', floor_w_ub_int32, origin_w_or_res_ub, 2, 8, 4)
        self.tik_instance.vec_conv(64, '', floor_w_ub, floor_w_ub_int32, 2, 4, 8, 1.0)
        return floor_w_ub, floor_w_ub_int32

    def calc_floor_h(self, origin_h_or_res_ub):
        floor_h_ub_int32 = self.tik_instance.Tensor(
            'int32', (128,), name='floor_h_ub_int32', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_conv(64, 'floor', floor_h_ub_int32, origin_h_or_res_ub, 2, 8, 4)
        floor_h_fp16_ub = self.tik_instance.Tensor(
            'float16', (128,), name='floor_h_fp16_ub', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_conv(64, '', floor_h_fp16_ub, floor_h_ub_int32, 2, 4, 8, 1.0)
        return floor_h_fp16_ub, floor_h_ub_int32

    def calc_origin_w(self, h_index_ub, origin_w_ub, theta2_1, theta2_2, theta2_3, w_index_ub):
        if self.calc_by_fp16:
            origin_w_or_res_ub = self.tik_instance.Tensor(
                "float16", (self.w_index_ub_size,), name='origin_w_or_res_ub', scope=tik.scope_ubuf
            )
            repeat = 1
            mask = 128
        else:
            origin_w_or_res_ub = self.tik_instance.Tensor(
                "float32", (self.w_index_ub_size,), name='origin_w_or_res_ub', scope=tik.scope_ubuf
            )
            repeat = 2
            mask = 64
        self.tik_instance.vec_muls(mask, origin_w_or_res_ub, h_index_ub, theta2_1, repeat, 8, 8)
        self.tik_instance.vec_muls(mask, origin_w_ub, w_index_ub, theta2_2, repeat, 8, 8)
        self.tik_instance.vec_add(mask, origin_w_or_res_ub, origin_w_or_res_ub, origin_w_ub, repeat, 8, 8, 8)
        self.tik_instance.vec_adds(mask, origin_w_or_res_ub, origin_w_or_res_ub, theta2_3, repeat, 8, 8)
        self.tik_instance.vec_adds(mask, origin_w_or_res_ub, origin_w_or_res_ub, 1.0, repeat, 8, 8)
        self.tik_instance.vec_muls(mask, origin_w_or_res_ub, origin_w_or_res_ub, 0.5, repeat, 8, 8)
        self.tik_instance.vec_muls(mask, origin_w_or_res_ub, origin_w_or_res_ub, self.size[1], repeat, 8, 8)
        return origin_w_or_res_ub

    def calc_origin_h(self, h_index_ub, theta1_1, theta1_2, theta1_3, w_index_ub):
        if self.calc_by_fp16:
            origin_h_or_res_ub = self.tik_instance.Tensor(
                'float16', (self.w_index_ub_size,), name='origin_h_or_res_ub',
                scope=tik.scope_ubuf
            )
            origin_w_ub = self.tik_instance.Tensor(
                'float16', (self.w_index_ub_size,), name='origin_w_ub', scope=tik.scope_ubuf
            )
            repeat = 1
            mask = 128
        else:
            origin_h_or_res_ub = self.tik_instance.Tensor(
                "float32", (self.w_index_ub_size,), name='origin_h_or_res_ub', scope=tik.scope_ubuf
            )
            origin_w_ub = self.tik_instance.Tensor(
                "float32", (self.w_index_ub_size,), name='origin_w_ub', scope=tik.scope_ubuf
            )
            repeat = 2
            mask = 64

        self.tik_instance.vec_muls(mask, origin_h_or_res_ub, h_index_ub, theta1_1, repeat, 8, 8)
        self.tik_instance.vec_muls(mask, origin_w_ub, w_index_ub, theta1_2, repeat, 8, 8)
        self.tik_instance.vec_add(mask, origin_h_or_res_ub, origin_h_or_res_ub, origin_w_ub, repeat, 8, 8, 8)
        self.tik_instance.vec_adds(mask, origin_h_or_res_ub, origin_h_or_res_ub, theta1_3, repeat, 8, 8)
        self.tik_instance.vec_adds(mask, origin_h_or_res_ub, origin_h_or_res_ub, 1.0, repeat, 8, 8)
        self.tik_instance.vec_muls(mask, origin_h_or_res_ub, origin_h_or_res_ub, 0.5, repeat, 8, 8)
        self.tik_instance.vec_muls(mask, origin_h_or_res_ub, origin_h_or_res_ub, self.size[0], repeat, 8, 8)
        return origin_h_or_res_ub, origin_w_ub

    def calc_theta(self, batch_id, d_type):
        # get theta
        if self.use_default_theta.count(True) != 6:
            input_theta_ub = self.tik_instance.Tensor(
                self.theta.get('dtype'), (128,), name='input_theta_ub',
                scope=tik.scope_ubuf
            )
            self.tik_instance.data_move(
                dst=input_theta_ub,
                src=self.theta_gm[batch_id * self.theta.get('shape')[1]],
                sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0
            )
            if not self.calc_by_fp16 and self.theta.get('dtype') == 'float16':
                input_theta_ub_fp32 = self.tik_instance.Tensor(
                    "float32", (128,), name='input_theta_ub_fp32',
                    scope=tik.scope_ubuf
                )
                self.tik_instance.vec_conv(64, '', input_theta_ub_fp32, input_theta_ub, 2, 8, 4)
                # get theta
                theta1_1, theta1_2, theta1_3, theta2_1, theta2_2, theta2_3 = self.get_theta_by_mix(d_type,
                                                                                                   input_theta_ub_fp32)
            else:
                theta1_1, theta1_2, theta1_3, theta2_1, theta2_2, theta2_3 = self.get_theta_by_mix(d_type,
                                                                                                   input_theta_ub)
        else:
            theta1_1 = self.default_theta[0]
            theta1_2 = self.default_theta[1]
            theta1_3 = self.default_theta[2]
            theta2_1 = self.default_theta[3]
            theta2_2 = self.default_theta[4]
            theta2_3 = self.default_theta[5]
        return theta1_1, theta1_2, theta1_3, theta2_1, theta2_2, theta2_3

    def get_theta_by_mix(self, d_type, input_theta_ub):
        use_default_theta_index = 0
        use_input_theta_index = 0
        if self.use_default_theta[0]:
            theta1_1 = self.default_theta[use_default_theta_index]
            use_default_theta_index += 1
        else:
            theta1_1 = self.tik_instance.Scalar(dtype=d_type)
            theta1_1.set_as(input_theta_ub[use_input_theta_index])
            use_input_theta_index += 1
        if self.use_default_theta[1]:
            theta1_2 = self.default_theta[use_default_theta_index]
            use_default_theta_index += 1
        else:
            theta1_2 = self.tik_instance.Scalar(dtype=d_type)
            theta1_2.set_as(input_theta_ub[use_input_theta_index])
            use_input_theta_index += 1
        if self.use_default_theta[2]:
            theta1_3 = self.default_theta[use_default_theta_index]
            use_default_theta_index += 1
        else:
            theta1_3 = self.tik_instance.Scalar(dtype=d_type)
            theta1_3.set_as(input_theta_ub[use_input_theta_index])
            use_input_theta_index += 1
        if self.use_default_theta[3]:
            theta2_1 = self.default_theta[use_default_theta_index]
            use_default_theta_index += 1
        else:
            theta2_1 = self.tik_instance.Scalar(dtype=d_type)
            theta2_1.set_as(input_theta_ub[use_input_theta_index])
            use_input_theta_index += 1
        if self.use_default_theta[4]:
            theta2_2 = self.default_theta[use_default_theta_index]
            use_default_theta_index += 1
        else:
            theta2_2 = self.tik_instance.Scalar(dtype=d_type)
            theta2_2.set_as(input_theta_ub[use_input_theta_index])
            use_input_theta_index += 1
        if self.use_default_theta[5]:
            theta2_3 = self.default_theta[use_default_theta_index]
            use_default_theta_index += 1
        else:
            theta2_3 = self.tik_instance.Scalar(dtype=d_type)
            theta2_3.set_as(input_theta_ub[use_input_theta_index])
            use_input_theta_index += 1
        return theta1_1, theta1_2, theta1_3, theta2_1, theta2_2, theta2_3

    def filter_coef(self, tmp_res_ceof_ub, float_filter_ub, floor_h_ub, floor_w_ub, filter_index, res_offset):
        # filter lt 0
        with self.tik_instance.new_stmt_scope():
            # filter ceof h < 0
            self.tik_instance.vec_cmpv_lt(filter_index, floor_h_ub, float_filter_ub, 1, 8, 8)
            tmp_res_ceof_1_ub = self.tik_instance.Tensor(
                'float16', (128,), name='tmp_res_ceof_1_ub', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_sel(128, 0, tmp_res_ceof_1_ub, filter_index, float_filter_ub, tmp_res_ceof_ub, 1, 0,
                                      0, 0)
            # build 1 tensor
            tmp_1_ub = self.tik_instance.Tensor(
                'float16', (128,), name='tmp_1_ub', scope=tik.scope_ubuf
            )
            offset_mul = self.tik_instance.Tensor(
                'float16', (128,), name='offset_mul', scope=tik.scope_ubuf
            )
            offset_mul_int32 = self.tik_instance.Tensor(
                'int32', (128,), name='offset_mul_int32', scope=tik.scope_ubuf
            )

            # filter offset
            self.tik_instance.vec_dup(128, tmp_1_ub, 1, 1, 1)
            self.tik_instance.vec_sel(128, 0, offset_mul, filter_index, float_filter_ub, tmp_1_ub, 1, 0, 0, 0)
            self.tik_instance.vec_conv(64, 'round', offset_mul_int32, offset_mul, 2, 8, 4)
            self.tik_instance.vec_mul(64, res_offset, res_offset, offset_mul_int32, 2, 8, 8, 8)

            # filter ceof w < 0
            self.tik_instance.vec_cmpv_lt(filter_index, floor_w_ub, float_filter_ub, 1, 8, 8)
            self.tik_instance.vec_sel(128, 0, tmp_res_ceof_ub, filter_index, float_filter_ub, tmp_res_ceof_1_ub, 1, 0,
                                      0, 0)

            # filter offset w < 0
            self.tik_instance.vec_sel(128, 0, offset_mul, filter_index, float_filter_ub, tmp_1_ub, 1, 0, 0, 0)
            self.tik_instance.vec_conv(64, 'round', offset_mul_int32, offset_mul, 2, 8, 4)
            self.tik_instance.vec_mul(64, res_offset, res_offset, offset_mul_int32, 2, 8, 8, 8)

            # filter ge h

            self.tik_instance.vec_dup(128, float_filter_ub, self.size[0], 1, 1)
            self.tik_instance.vec_cmpv_ge(filter_index, floor_h_ub, float_filter_ub, 1, 8, 8)
            self.tik_instance.vec_dup(128, float_filter_ub, 0.0, 1, 1)
            self.tik_instance.vec_sel(128, 0, tmp_res_ceof_1_ub, filter_index, float_filter_ub, tmp_res_ceof_ub, 1, 0,
                                      0, 0)
            self.tik_instance.vec_sel(128, 0, offset_mul, filter_index, float_filter_ub, tmp_1_ub, 1, 0, 0, 0)
            self.tik_instance.vec_conv(64, 'round', offset_mul_int32, offset_mul, 2, 8, 4)
            self.tik_instance.vec_mul(64, res_offset, res_offset, offset_mul_int32, 2, 8, 8, 8)

            # filter ge w
            self.tik_instance.vec_dup(128, float_filter_ub, self.size[1], 1, 1)
            self.tik_instance.vec_cmpv_ge(filter_index, floor_w_ub, float_filter_ub, 1, 8, 8)
            self.tik_instance.vec_dup(128, float_filter_ub, 0.0, 1, 1)
            self.tik_instance.vec_sel(128, 0, tmp_res_ceof_ub, filter_index, float_filter_ub, tmp_res_ceof_1_ub, 1, 0,
                                      0, 0)
            self.tik_instance.vec_sel(128, 0, offset_mul, filter_index, float_filter_ub, tmp_1_ub, 1, 0, 0, 0)
            self.tik_instance.vec_conv(64, 'round', offset_mul_int32, offset_mul, 2, 8, 4)
            self.tik_instance.vec_mul(64, res_offset, res_offset, offset_mul_int32, 2, 8, 8, 8)

    def calc_ceof_by_fp32(self, w_ub, origin_h_or_res_ub, origin_w_or_res_ub, h_ub, tmp_res_ceof_ub):
        # calc (h, w): (1 - abs(origin_h - h)) * (1 - abs(origin_w - w))
        mask = 64
        repeat = 2
        with self.tik_instance.new_stmt_scope():
            res = self.tik_instance.Tensor(
                'float32', (128,), name='res', scope=tik.scope_ubuf
            )
            w_ub_fp32 = self.tik_instance.Tensor(
                'float32', (128,), name='w_ub_fp32', scope=tik.scope_ubuf
            )
            h_ub_fp32 = self.tik_instance.Tensor(
                'float32', (128,), name='h_ub_fp32', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_conv(mask, '', w_ub_fp32, w_ub, repeat, 8, 4)
            self.tik_instance.vec_conv(mask, '', h_ub_fp32, h_ub, repeat, 8, 4)
            self.tik_instance.vec_muls(mask, res, h_ub_fp32, -1.0, repeat, 8, 8)
            self.tik_instance.vec_add(mask, res, origin_h_or_res_ub, res, repeat, 8, 8, 8)
            self.tik_instance.vec_abs(mask, res, res, repeat, 8, 8)
            self.tik_instance.vec_muls(mask, res, res, -1.0, repeat, 8, 8)
            self.tik_instance.vec_adds(mask, res, res, 1.0, repeat, 8, 8)
            tmp_w_ceof_ub = self.tik_instance.Tensor(
                'float32', (128,), name='tmp_w_ceof_ub', scope=tik.scope_ubuf
            )
            self.tik_instance.vec_muls(mask, tmp_w_ceof_ub, w_ub_fp32, -1.0, repeat, 8, 8)
            self.tik_instance.vec_add(mask, tmp_w_ceof_ub, origin_w_or_res_ub, tmp_w_ceof_ub, repeat, 8, 8, 8)
            self.tik_instance.vec_abs(mask, tmp_w_ceof_ub, tmp_w_ceof_ub, repeat, 8, 8)
            self.tik_instance.vec_muls(mask, tmp_w_ceof_ub, tmp_w_ceof_ub, -1.0, repeat, 8, 8)
            self.tik_instance.vec_adds(mask, tmp_w_ceof_ub, tmp_w_ceof_ub, 1.0, repeat, 8, 8)
            self.tik_instance.vec_mul(mask, res, res, tmp_w_ceof_ub, repeat, 8, 8, 8)
            self.tik_instance.vec_conv(64, '', tmp_res_ceof_ub, res, 2, 4, 8)

    def calc_ceof(self, w_ub, origin_h_or_res_ub, origin_w_or_res_ub, h_ub, tmp_res_ceof_ub):
        # calc (h, w): (1 - abs(origin_h - h)) * (1 - abs(origin_w - w))
        self.tik_instance.vec_muls(128, tmp_res_ceof_ub, h_ub, -1.0, 1, 0, 0)
        self.tik_instance.vec_add(128, tmp_res_ceof_ub, origin_h_or_res_ub, tmp_res_ceof_ub, 1, 0, 0, 0)
        self.tik_instance.vec_abs(128, tmp_res_ceof_ub, tmp_res_ceof_ub, 1, 0, 0)
        self.tik_instance.vec_muls(128, tmp_res_ceof_ub, tmp_res_ceof_ub, -1.0, 1, 0, 0)
        self.tik_instance.vec_adds(128, tmp_res_ceof_ub, tmp_res_ceof_ub, 1.0, 1, 0, 0)
        tmp_w_ceof_ub = self.tik_instance.Tensor(
            'float16', (128,), name='tmp_w_ceof_ub', scope=tik.scope_ubuf
        )
        self.tik_instance.vec_muls(128, tmp_w_ceof_ub, w_ub, -1.0, 1, 0, 0)
        self.tik_instance.vec_add(128, tmp_w_ceof_ub, origin_w_or_res_ub, tmp_w_ceof_ub, 1, 0, 0, 0)
        self.tik_instance.vec_abs(128, tmp_w_ceof_ub, tmp_w_ceof_ub, 1, 0, 0)
        self.tik_instance.vec_muls(128, tmp_w_ceof_ub, tmp_w_ceof_ub, -1.0, 1, 0, 0)
        self.tik_instance.vec_adds(128, tmp_w_ceof_ub, tmp_w_ceof_ub, 1.0, 1, 0, 0)
        self.tik_instance.vec_mul(128, tmp_res_ceof_ub, tmp_res_ceof_ub, tmp_w_ceof_ub, 1, 0, 0, 0)
