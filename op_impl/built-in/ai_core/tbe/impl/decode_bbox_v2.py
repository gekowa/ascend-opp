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
decode_bbox_v2
"""
import math
from functools import reduce as functools_reduce

from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *

SHAPE_SIZE_ONE = 1
SHAPE_SIZE_FOUR = 4
RESERVED_UB = 20480
# MAX ELIMENT NUM OF FP16 IN 1BLOCK
FP16_ELIMENTS_BLOCK = 16
# MAX ELIMENT NUM OF FP32 IN 1BLOCK
FP32_ELIMENTS_BLOCK = 8
FP16_VECTOR_MASK_MAX = 128
FP32_VECTOR_MASK_MAX = 64
# vnchwconv instr compute min blocks
TRANS_MIN_BLKS = 16
# do transpose need split ub to trisection
TRISECTION_UB = 3
BLOCK_BYTES = 32
CONFIG_TWO = 2
REPEAT_TIMES_MAX = 255
SMMU_ID = 0
DATA_MOV_STRIDE = 0
DATA_MOV_NBURST = 1
REMAIN_REPEAT = 1
DEQ_SCALE = 1.0
SCALAR_HALF = 0.5


class DecodeBboxV2():
    """
       Function: use to store  base parameters
       Modify : 2020-07-27
    """

    # pylint: disable=too-many-statements
    def __init__(self, boxes, anchors, scales, decode_clip, reversed_box,
                 kernel_name):
        """
        Init scatter base parameters

        Parameters
        ----------
        boxes: dict
            data of input boxes
            datatype suports float32,float16
        anchors: dict
            data of input anchors
            datatype supports float32,float16
        scales: ListFloat
            data of boxes scalse
        decode_clip: float
            data of input clip
        reversed_box: bool
            if input need do transpose
        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.boxes_shape = boxes.get("shape")
        self.boxes_dtype = boxes.get("dtype").lower()
        self.anchors_shape = anchors.get("shape")
        self.anchors_dtype = anchors.get("dtype").lower()
        self.scale_list = scales
        self.decode_clip = decode_clip
        self.reversed_box = reversed_box
        self.kernel_name = kernel_name
        # input param check
        self.check_param()
        self.scale_y = scales[0]
        self.scale_x = scales[1]
        self.scale_h = scales[2]
        self.scale_w = scales[3]
        self.total_ub = tik.Dprofile().get_unified_buffer_size()
        self.available_ub_size = self.total_ub // CONFIG_TWO - RESERVED_UB
        self.aicore_num = tik.Dprofile().get_aicore_num()
        self.ele_block_dict = {"float32": FP32_ELIMENTS_BLOCK,
                               "float16": FP16_ELIMENTS_BLOCK}
        self.vector_mask_max_dict = {"float32": FP32_VECTOR_MASK_MAX,
                                     "float16": FP16_VECTOR_MASK_MAX}
        self.boxes_ele_per_block = self.ele_block_dict.get(self.boxes_dtype)
        self.ele_reversed_dict = {True: SHAPE_SIZE_FOUR,
                                  False: SHAPE_SIZE_ONE}
        self.ele_multiple = self.ele_reversed_dict.get(self.reversed_box)
        self.compute_blks_multiple = self.ele_multiple / SHAPE_SIZE_FOUR
        self.element_num = int(functools_reduce(lambda x, y: x * y,
                                                self.boxes_shape) /
                               self.ele_multiple)
        self.boxes_blocks = math.ceil(self.element_num /
                                      self.boxes_ele_per_block)
        if self.reversed_box is True:
            self.block_nums_per_loop_max = self.available_ub_size // \
                                           (BLOCK_BYTES * self.ele_multiple *
                                            CONFIG_TWO)
        else:
            self.block_nums_per_loop_max = self.available_ub_size // \
                                           (BLOCK_BYTES * TRISECTION_UB) // \
                                           TRANS_MIN_BLKS * TRANS_MIN_BLKS
        # Calculate the blocks that need to be calculated for each core
        self.cal_blocks_per_core = math.ceil(self.boxes_blocks /
                                             self.aicore_num)
        # The calculation needs to use the number of cores
        self.cal_core_num = math.ceil(self.boxes_blocks /
                                      self.cal_blocks_per_core)
        # The blocks that need to be calculated for the last core
        self.cal_blocks_last_core = self.boxes_blocks-(self.cal_core_num-1) * \
                                    self.cal_blocks_per_core
        # Define input and output gm
        self.boxes_gm = self.tik_instance.Tensor(
            self.boxes_dtype, self.boxes_shape, name="boxes_gm",
            scope=tik.scope_gm)
        self.anchors_gm = self.tik_instance.Tensor(
            self.anchors_dtype, self.anchors_shape, name="anchors_gm",
            scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(
            self.boxes_dtype, self.boxes_shape, name="y_gm",
            scope=tik.scope_gm)

        # The number of times that each core data cycle is moved in
        self.loop_times = 0
        # The blocks that need to be calculated in each cycle
        self.cal_blocks_per_loop = 0
        # The blocks to be calculated in the last loop
        self.cal_blocks_last_loop = 0
        # The number of elements to be calculated in each loop
        self.ele_num_per_loop = 0
        self.thread_nums = 0
        self.vector_mask_max = 0
        self.blk_stride = 0
        self.dst_rep_stride = 0
        self.src_rep_stride = 0
        self.ty_ub = 0
        self.tx_ub = 0
        self.th_ub = 0
        self.tw_ub = 0
        self.anchor_ymin_ub = 0
        self.anchor_xmin_ub = 0
        self.anchor_ymax_ub = 0
        self.anchor_xmax_ub = 0

    def ub_init(self, move_blocks):
        if self.reversed_box is True:
            self.boxes_ub = self.tik_instance.Tensor(
                self.boxes_dtype, [move_blocks * self.boxes_ele_per_block *
                                   self.ele_multiple],
                name="boxes_ub", scope=tik.scope_ubuf)
            self.anchors_ub = self.tik_instance.Tensor(
                self.anchors_dtype, [move_blocks * self.boxes_ele_per_block *
                                     self.ele_multiple],
                name="anchors_ub", scope=tik.scope_ubuf)
        else:
            self.boxes_ub = self.tik_instance.Tensor(
                self.boxes_dtype, [math.ceil(move_blocks / TRANS_MIN_BLKS) *
                                   TRANS_MIN_BLKS * self.boxes_ele_per_block],
                name="boxes_ub", scope=tik.scope_ubuf)
            self.anchors_ub = self.tik_instance.Tensor(
                self.anchors_dtype, [math.ceil(move_blocks / TRANS_MIN_BLKS) *
                                    TRANS_MIN_BLKS * self.boxes_ele_per_block],
                name="anchors_ub", scope=tik.scope_ubuf)
            self.anchors_trans_ub = self.tik_instance.Tensor(
                self.anchors_dtype, [math.ceil(move_blocks / TRANS_MIN_BLKS) *
                                    TRANS_MIN_BLKS * self.boxes_ele_per_block],
                name="anchors_trans_ub", scope=tik.scope_ubuf)

    def input_unpack(self, compute_blocks):
        if self.reversed_box is True:
            self.ty_ub = self.boxes_ub[0 * compute_blocks *
                                       self.boxes_ele_per_block:]
            self.tx_ub = self.boxes_ub[1 * compute_blocks *
                                       self.boxes_ele_per_block:]
            self.th_ub = self.boxes_ub[2 * compute_blocks *
                                       self.boxes_ele_per_block:]
            self.tw_ub = self.boxes_ub[3 * compute_blocks *
                                       self.boxes_ele_per_block:]
            self.anchor_ymin_ub = self.anchors_ub[0 * compute_blocks *
                                                  self.boxes_ele_per_block:]
            self.anchor_xmin_ub = self.anchors_ub[1 * compute_blocks *
                                                  self.boxes_ele_per_block:]
            self.anchor_ymax_ub = self.anchors_ub[2 * compute_blocks *
                                                  self.boxes_ele_per_block:]
            self.anchor_xmax_ub = self.anchors_ub[3 * compute_blocks *
                                                  self.boxes_ele_per_block:]
        else:
            self.ty_ub = self.anchors_ub[0 * self.boxes_ele_per_block:]
            self.tx_ub = self.anchors_ub[1 * self.boxes_ele_per_block:]
            self.th_ub = self.anchors_ub[2 * self.boxes_ele_per_block:]
            self.tw_ub = self.anchors_ub[3 * self.boxes_ele_per_block:]
            self.anchor_ymin_ub = self.anchors_trans_ub[0 *
                                  self.boxes_ele_per_block:]
            self.anchor_xmin_ub = self.anchors_trans_ub[1 *
                                  self.boxes_ele_per_block:]
            self.anchor_ymax_ub = self.anchors_trans_ub[2 *
                                  self.boxes_ele_per_block:]
            self.anchor_xmax_ub = self.anchors_trans_ub[3 *
                                  self.boxes_ele_per_block:]

    def transpose_n4_to_4n(self, trans_blocks, src_ub, dst_ub):
        repeat_time = math.ceil(trans_blocks / TRANS_MIN_BLKS)
        dst_rep_stride = TRANS_MIN_BLKS
        src_rep_stride = TRANS_MIN_BLKS
        if repeat_time == 1:
            dst_rep_stride = 0
            src_rep_stride = 0
        src_addr_list = [src_ub[i*self.boxes_ele_per_block]
                         for i in range(0, TRANS_MIN_BLKS)]
        dst_addr_list = [dst_ub[i*self.boxes_ele_per_block]
                         for i in range(0, TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False, dst_addr_list, src_addr_list,
                                    repeat_time,
                                    dst_rep_stride, src_rep_stride)

    def cal_data_mov_loops(self, cal_blocks_this_core):
        # Calculate the number of cycles required
        self.loop_times = math.ceil(cal_blocks_this_core /
                                    self.block_nums_per_loop_max)
        self.cal_blocks_per_loop = math.ceil(cal_blocks_this_core /
                                             self.loop_times)
        self.cal_blocks_last_loop = cal_blocks_this_core-(self.loop_times-1) * \
                                    self.cal_blocks_per_loop
        # Multiple threads cannot be started in one loop
        if self.loop_times == 1:
            self.thread_nums = 1
        else:
            self.thread_nums = 2

    def decode_bboxv2_compute(self):
        with self.tik_instance.for_range(
                0, self.cal_core_num, block_num=self.cal_core_num) as index:
            move_offset_core = index * (self.cal_blocks_per_core *
                                        self.boxes_ele_per_block)
            with self.tik_instance.if_scope(index < (self.cal_core_num-1)):
                self.cal_data_mov_loops(self.cal_blocks_per_core)
                self.decode_bboxv2_compute_each_core(move_offset_core)
            with self.tik_instance.else_scope():
                move_offset_core = max((self.element_num -
                                        self.cal_blocks_last_core *
                                        self.boxes_ele_per_block), 0)
                self.cal_data_mov_loops(self.cal_blocks_last_core)
                self.decode_bboxv2_compute_each_core(move_offset_core)
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.boxes_gm, self.anchors_gm],
            outputs=[self.y_gm])

        return self.tik_instance

    def decode_bboxv2_compute_each_core(self, move_offset_core):
        with self.tik_instance.for_range(
                0, self.loop_times, thread_num=self.thread_nums) as i:
            self.ub_init(self.cal_blocks_per_loop)
            move_offset_loop = move_offset_core + i * \
                               self.cal_blocks_per_loop * \
                               self.boxes_ele_per_block
            with self.tik_instance.if_scope(i < (self.loop_times-1)):
                self.data_move_in_each_loop(move_offset_loop,
                                            self.cal_blocks_per_loop)
            with self.tik_instance.else_scope():
                self.data_move_in_each_loop(move_offset_loop,
                                            self.cal_blocks_last_loop)

    def data_move_in_each_loop(self, move_offset_loop, compute_blocks):
        if self.reversed_box is True:
            for j in range(0, SHAPE_SIZE_FOUR):
                self.tik_instance.data_move(
                    self.boxes_ub[j*compute_blocks*self.boxes_ele_per_block],
                    self.boxes_gm[move_offset_loop+j*self.element_num],
                    SMMU_ID, DATA_MOV_NBURST, compute_blocks,
                    DATA_MOV_STRIDE, DATA_MOV_STRIDE)
                self.tik_instance.data_move(
                    self.anchors_ub[j*compute_blocks*self.boxes_ele_per_block],
                    self.anchors_gm[move_offset_loop+j*self.element_num],
                    SMMU_ID, DATA_MOV_NBURST, compute_blocks,
                    DATA_MOV_STRIDE, DATA_MOV_STRIDE)
        else:
            self.tik_instance.data_move(
                self.boxes_ub,
                self.boxes_gm[move_offset_loop],
                SMMU_ID, DATA_MOV_NBURST, compute_blocks,
                DATA_MOV_STRIDE, DATA_MOV_STRIDE)
            self.tik_instance.data_move(
                self.anchors_ub,
                self.anchors_gm[move_offset_loop],
                SMMU_ID, DATA_MOV_NBURST, compute_blocks,
                DATA_MOV_STRIDE, DATA_MOV_STRIDE)
            self.transpose_n4_to_4n(compute_blocks, self.anchors_ub,
                                    self.anchors_trans_ub)
            self.transpose_n4_to_4n(compute_blocks, self.boxes_ub,
                                    self.anchors_ub)

        self.input_unpack(compute_blocks)

        # anchor_h = anchor_ymax - anchor_ymin
        self.vector_compute(self.anchor_ymax_ub, self.anchor_ymax_ub,
                            "vsub", compute_blocks, src_ub1=self.anchor_ymin_ub)
        self.anchor_h = self.anchor_ymax_ub
        # anchor_w = anchor_xmax - anchor_xmin
        self.vector_compute(self.anchor_xmax_ub, self.anchor_xmax_ub,
                            "vsub", compute_blocks, src_ub1=self.anchor_xmin_ub)
        self.anchor_w = self.anchor_xmax_ub

        # scaled_ty = ty / y_scale
        if not self.isclose(self.scale_y, DEQ_SCALE):
            self.vector_compute(self.ty_ub, self.ty_ub, "vmuls",
                                compute_blocks, scalar=1/self.scale_y)
        # scaled_tx = tx / x_scale
        if not self.isclose(self.scale_x, DEQ_SCALE):
            self.vector_compute(self.tx_ub, self.tx_ub, "vmuls",
                                compute_blocks, scalar=1/self.scale_x)
        # scaled_th = th / h_scale
        if not self.isclose(self.scale_h, DEQ_SCALE):
            self.vector_compute(self.th_ub, self.th_ub, "vmuls",
                                compute_blocks, scalar=1/self.scale_h)
        # scaled_tw = tw / w_scale
        if not self.isclose(self.scale_w, DEQ_SCALE):
            self.vector_compute(self.tw_ub, self.tw_ub, "vmuls",
                                compute_blocks, scalar=1/self.scale_w)
        # scaled_ty * anchor_h
        self.vector_compute(self.ty_ub, self.ty_ub, "vmul",
                            compute_blocks, src_ub1=self.anchor_h)
        # scaled_ty * anchor_h + anchor_ymin
        self.vector_compute(self.ty_ub, self.ty_ub, "vadd",
                            compute_blocks, src_ub1=self.anchor_ymin_ub)
        # scaled_tx * anchor_w
        self.vector_compute(self.tx_ub, self.tx_ub, "vmul",
                            compute_blocks, src_ub1=self.anchor_w)
        # scaled_tx * anchor_w + anchor_xmin
        self.vector_compute(self.tx_ub, self.tx_ub, "vadd",
                            compute_blocks, src_ub1=self.anchor_xmin_ub)

        if not self.isclose(self.decode_clip, 0):
            if tbe_platform.cce_conf.api_check_support("tik.vmins",
                                                       self.boxes_dtype):
                # min(scaled_tw,clip_scalar)
                self.vector_compute(self.tw_ub, self.tw_ub, "vmins",
                                    compute_blocks, scalar=self.decode_clip)
                # min(scaled_th,clip_scalar)
                self.vector_compute(self.th_ub, self.th_ub, "vmins",
                                    compute_blocks, scalar=self.decode_clip)
            else:
                # decode_clip vector_dump to anchor_x
                self.vector_compute(self.anchor_ymin_ub, self.anchor_ymin_ub,
                                    "vector_dup",
                                    compute_blocks, scalar=self.decode_clip)
                self.clip_vector = self.anchor_ymin_ub
                self.vector_compute(self.tw_ub, self.tw_ub, "vmin",
                                    compute_blocks, src_ub1=self.clip_vector)
                self.vector_compute(self.th_ub, self.th_ub, "vmin",
                                    compute_blocks, src_ub1=self.clip_vector)
        # w = exp(min(scaled_tw,clip_scalar)) * anchor_w
        self.vector_compute(self.tw_ub, self.tw_ub, "vexp",
                            compute_blocks)
        self.vector_compute(self.tw_ub, self.tw_ub, "vmul",
                            compute_blocks, src_ub1=self.anchor_w)
        # h = exp(min(scaled_th,cip_scalar)) * anchor_h
        self.vector_compute(self.th_ub, self.th_ub, "vexp",
                            compute_blocks)
        self.vector_compute(self.th_ub, self.th_ub, "vmul",
                            compute_blocks, src_ub1=self.anchor_h)
        # anchor_h / 2
        self.vector_compute(self.anchor_h, self.anchor_h,
                            "vmuls", compute_blocks, scalar=SCALAR_HALF)
        # anchor_w / 2
        self.vector_compute(self.anchor_w, self.anchor_w,
                            "vmuls", compute_blocks, scalar=SCALAR_HALF)
        # ycenter = scaled_ty * anchor_h + anchor_ymin + anchor_h / 2
        self.vector_compute(self.ty_ub, self.ty_ub,
                            "vadd", compute_blocks, src_ub1=self.anchor_h)
        # xcenter = scaled_tx * anchor_w + anchor_xmin + anchor_w / 2
        self.vector_compute(self.tx_ub, self.tx_ub,
                            "vadd", compute_blocks, src_ub1=self.anchor_w)
        # h / 2
        self.vector_compute(self.th_ub, self.th_ub, "vmuls",
                            compute_blocks, scalar=SCALAR_HALF)
        # w / 2
        self.vector_compute(self.tw_ub, self.tw_ub, "vmuls",
                            compute_blocks, scalar=SCALAR_HALF)
        # ymin = ycenter - h / 2
        self.vector_compute(self.anchor_ymin_ub, self.ty_ub, "vsub",
                            compute_blocks, src_ub1=self.th_ub)
        # ymax = ycenter + h / 2
        self.vector_compute(self.anchor_ymax_ub, self.ty_ub, "vadd",
                            compute_blocks, src_ub1=self.th_ub)
        # xmin = xcenter - w / 2
        self.vector_compute(self.anchor_xmin_ub, self.tx_ub, "vsub",
                            compute_blocks, src_ub1=self.tw_ub)
        # xmax = xcenter + w / 2
        self.vector_compute(self.anchor_xmax_ub, self.tx_ub, "vadd",
                            compute_blocks, src_ub1=self.tw_ub)
        if self.reversed_box is True:
            for j in range(0, SHAPE_SIZE_FOUR):
                self.tik_instance.data_move(
                    self.y_gm[move_offset_loop+j*self.element_num],
                    self.anchors_ub[j * compute_blocks *
                                    self.boxes_ele_per_block],
                    SMMU_ID, DATA_MOV_NBURST, compute_blocks,
                    DATA_MOV_STRIDE, DATA_MOV_STRIDE)
        else:
            self.transpose_n4_to_4n(compute_blocks, self.anchors_trans_ub,
                                    self.anchors_ub)
            self.tik_instance.data_move(
                self.y_gm[move_offset_loop],
                self.anchors_ub,
                SMMU_ID, DATA_MOV_NBURST, compute_blocks,
                DATA_MOV_STRIDE, DATA_MOV_STRIDE)

    def vector_compute(self, dst_ub, src_ub0, compute_type, compute_blocks,
                       src_ub1=None, scalar=None):
        self.get_vector_mask_max(dst_ub, src_ub0)
        compute_ele_nums = int(self.ele_block_dict.get(src_ub0.dtype) * \
                               compute_blocks * self.compute_blks_multiple)
        compute_instr_loops = compute_ele_nums // (self.vector_mask_max *
                                                   REPEAT_TIMES_MAX)
        compute_offset = 0
        if compute_instr_loops > 0:
            with self.tik_instance.for_range(0, compute_instr_loops) as \
                    instr_loops_index:
                compute_offset = instr_loops_index * \
                                 self.vector_mask_max * REPEAT_TIMES_MAX * \
                                 self.blk_stride
                if compute_type == "vconv":
                    self.vconv_instr_gen(self.vector_mask_max, compute_offset,
                                         dst_ub, src_ub0, REPEAT_TIMES_MAX)
                if compute_type in ["vsub", "vadd", "vmul", "vmin"]:
                    self.double_in_instr_gen(self.vector_mask_max,
                                             compute_offset,
                                             dst_ub, src_ub0, src_ub1,
                                             REPEAT_TIMES_MAX, compute_type)
                if compute_type in ["vmuls", "vmins", "vector_dup", "vexp"]:
                    self.tensor_scalar_instr_gen(self.vector_mask_max,
                                                 compute_offset,
                                                 dst_ub, src_ub0,
                                                 REPEAT_TIMES_MAX,
                                                 scalar, compute_type)
            compute_offset = compute_instr_loops * self.vector_mask_max * \
                             REPEAT_TIMES_MAX * self.blk_stride
        repeat_time = (compute_ele_nums %
                       (self.vector_mask_max * REPEAT_TIMES_MAX) //
                       self.vector_mask_max)
        if repeat_time > 0:
            if compute_type == "vconv":
                self.vconv_instr_gen(self.vector_mask_max, compute_offset,
                                     dst_ub, src_ub0, repeat_time)
            if compute_type in ["vsub", "vadd", "vmul", "vmin"]:
                self.double_in_instr_gen(self.vector_mask_max, compute_offset,
                                         dst_ub, src_ub0, src_ub1,
                                         repeat_time, compute_type)
            if compute_type in ["vmuls", "vmins", "vector_dup", "vexp"]:
                self.tensor_scalar_instr_gen(self.vector_mask_max,
                                             compute_offset,
                                             dst_ub, src_ub0, repeat_time,
                                             scalar, compute_type)
            compute_offset = compute_offset + repeat_time * \
                             self.vector_mask_max * self.blk_stride
        last_num = compute_ele_nums % self.vector_mask_max
        if self.reversed_box is False:
            if last_num > 0:
                if last_num > self.vector_mask_max/CONFIG_TWO:
                    last_num = int(self.vector_mask_max)
                else:
                    last_num = int(self.vector_mask_max/CONFIG_TWO)
        if last_num > 0:
            if compute_type == "vconv":
                self.vconv_instr_gen(last_num, compute_offset,
                                     dst_ub, src_ub0, REMAIN_REPEAT)
            if compute_type in ["vsub", "vadd", "vmul", "vmin"]:
                self.double_in_instr_gen(last_num, compute_offset,
                                         dst_ub, src_ub0, src_ub1,
                                         REMAIN_REPEAT, compute_type)
            if compute_type in ["vmuls", "vmins", "vector_dup", "vexp"]:
                self.tensor_scalar_instr_gen(last_num, compute_offset,
                                             dst_ub, src_ub0, REMAIN_REPEAT,
                                             scalar, compute_type)

    def double_in_instr_gen(self, mask, offset, dst_ub,
                            src_ub0, src_ub1, repeat_times, compute_type):
        tik_fun = None
        if compute_type == "vsub":
            tik_fun = self.tik_instance.vsub
        if compute_type == "vadd":
            tik_fun = self.tik_instance.vadd
        if compute_type == "vmul":
            tik_fun = self.tik_instance.vmul
        if compute_type == "vmin":
            tik_fun = self.tik_instance.vmin
        return tik_fun(mask, dst_ub[offset], src_ub0[offset], src_ub1[offset],
                       repeat_times, self.blk_stride, self.blk_stride,
                       self.blk_stride,
                       self.dst_rep_stride, self.src_rep_stride,
                       self.src_rep_stride,)

    def tensor_scalar_instr_gen(self, mask, offset, dst_ub,
                                src_ub, repeat_times, scalar, compute_type):
        tik_fun = None
        if compute_type == "vmuls":
            tik_fun = self.tik_instance.vmuls
        if compute_type == "vmins":
            tik_fun = self.tik_instance.vmins
        if compute_type == "vector_dup":
            return self.tik_instance.vector_dup(mask, dst_ub[offset],
                                                scalar, repeat_times,
                                                self.blk_stride,
                                                self.dst_rep_stride)
        if compute_type == "vexp":
            return self.tik_instance.vexp(mask, dst_ub[offset],
                                          src_ub[offset],
                                          repeat_times, self.blk_stride,
                                          self.blk_stride,
                                          self.dst_rep_stride,
                                          self.src_rep_stride)

        return tik_fun(mask, dst_ub[offset], src_ub[offset], scalar,
                       repeat_times, self.blk_stride, self.blk_stride,
                       self.dst_rep_stride, self.src_rep_stride)

    def get_vector_mask_max(self, dst_ub, src_ub):
        self.vector_mask_max = min(
            self.vector_mask_max_dict.get(dst_ub.dtype),
            self.vector_mask_max_dict.get(src_ub.dtype))
        if self.reversed_box is True:
            self.blk_stride = SHAPE_SIZE_ONE
        else:
            self.blk_stride = SHAPE_SIZE_FOUR
        self.dst_rep_stride = self.vector_mask_max // \
                              self.ele_block_dict.get(dst_ub.dtype) * \
                              self.blk_stride
        self.src_rep_stride = self.vector_mask_max // \
                              self.ele_block_dict.get(src_ub.dtype) * \
                              self.blk_stride

    def check_param(self):
        check_shape(self.boxes_shape, min_rank=2, max_rank=2,
                    param_name="boxes")
        check_shape(self.anchors_shape, min_rank=2, max_rank=2,
                    param_name="anchors")
        if self.boxes_shape != self.anchors_shape:
            error_info = {}
            error_info['errCode'] = 'E80017'
            error_info['op_name'] = 'decode_bbox_v2'
            error_info['param_name1'] = "boxes_shape"
            error_info['param_name2'] = "anchors_shape"
            error_info['param1_shape'] = self.boxes_shape
            error_info['param2_shape'] = self.anchors_shape
            raise RuntimeError(error_info,
            "In op[{op_name}], the parameter[{param_name1}][{param_name2}] "
            "is not match with the parameter[{param1_shape}]"
            "[{param1_shape}].".format(**error_info))
        if self.reversed_box is False:
            if self.boxes_shape[-1] != SHAPE_SIZE_FOUR:
                error_info = {}
                error_info['errCode'] = 'E80000'
                error_info['op_name'] = 'decode_bbox_v2'
                error_info['param_name'] = "boxes_shape's last dim"
                error_info['excepted_value'] = SHAPE_SIZE_FOUR
                error_info['real_value'] = self.boxes_shape[-1]
                raise RuntimeError(error_info,
                "In op[{op_name}], the parameter[{param_name}] "
                "should be [{excepted_value}], "
                "but actually is [{real_value}].".format(**error_info))
        if self.reversed_box is True:
            if self.boxes_shape[0] != SHAPE_SIZE_FOUR:
                error_info = {}
                error_info['errCode'] = 'E80000'
                error_info['op_name'] = 'decode_bbox_v2'
                error_info['param_name'] = "boxes_shape's first dim"
                error_info['excepted_value'] = SHAPE_SIZE_FOUR
                error_info['real_value'] = self.boxes_shape[0]
                raise RuntimeError(error_info,
                "In op[{op_name}], the parameter[{param_name}] "
                "should be [{excepted_value}], "
                "but actually is [{real_value}].".format(**error_info))
        check_list_boxes = ["float16", "float32"]
        check_list_anchors = ["float16", "float32"]
        check_dtype(self.boxes_dtype, check_list_boxes, param_name="boxes")
        check_dtype(self.anchors_dtype, check_list_anchors,
                    param_name="anchors")
        if not tbe_platform.cce_conf.api_check_support(
                "tik.vnchwconv", dtype=self.boxes_dtype):
            check_dtype(self.boxes_dtype, ["float16"], param_name="boxes")
        if self.boxes_dtype != self.anchors_dtype:
            error_info = {}
            error_info['errCode'] = 'E80018'
            error_info['op_name'] = 'decode_bbox_v2'
            error_info['param_name1'] = "boxes_dtype"
            error_info['param_name2'] = "anchors_dtype"
            error_info['param1_dtype'] = self.boxes_dtype
            error_info['param2_dtype'] = self.anchors_dtype
            raise RuntimeError(error_info,
            "In op[{op_name}], the parameter[{param_name1}][{param_name2}] "
            "are not equal in dtype with dtype[{param1_dtype}]"
            "[{param2_dtype}].".format(**error_info))
        if self.decode_clip < 0 or self.decode_clip > 10:
            error_info = {}
            error_info['errCode'] = 'E80002'
            error_info['op_name'] = 'decode_bbox_v2'
            error_info['param_name'] = "decode_clip"
            error_info['min_value'] = 0
            error_info['max_value'] = 10
            error_info['value'] = self.decode_clip
            raise RuntimeError(error_info,
                               "In op[{op_name}], the parameter[{param_name}] "
                               "should be in the range of "
                               "[{min_value}, {max_value}], but actually "
                               "is [{value}].".format(**error_info))
        if len(self.scale_list) != SHAPE_SIZE_FOUR:
            error_info = {}
            error_info['errCode'] = 'E80000'
            error_info['op_name'] = 'decode_bbox_v2'
            error_info['param_name'] = "length of scale_list"
            error_info['excepted_value'] = SHAPE_SIZE_FOUR
            error_info['real_value'] = len(self.scale_list)
            raise RuntimeError(error_info,
                               "In op[{op_name}], the parameter[{param_name}] "
                               "should be [{excepted_value}], but actually "
                               "is [{real_value}].".format(**error_info))

    def isclose(self, valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
        """
        determines whether the values of two floating-point numbers
        are close or equal
        """

        return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_LIST_FLOAT, OPTION_ATTR_FLOAT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def decode_bbox_v2(boxes, anchors, y, scales=[1.0, 1.0, 1.0, 1.0],
                   decode_clip=0.0, reversed_box=False,
                   kernel_name="decode_bbox_v2"):
    """
    calculating data

    Parameters
    ----------
    boxes : dict
        shape and dtype of input boxes
    anchors : dict
        shape and dtype of input anchors, should be same shape and type
        as boxes
    y : dict
        shape and dtype of output, should be same shape and type as input
    scales : ListFloat
        scales list has 4 float value
    decode_clip : float
         decode_clip value, default value is 0
    reversed_box : bool
         if True, boxes an anchors shape is [4,N],else is [N,4]
    kernel_name : str
        kernel name, default value is "decodeb_boxv2"

    Returns
    -------
    None
    """

    decode_bbox = DecodeBboxV2(boxes, anchors, scales, decode_clip,
                                 reversed_box, kernel_name)
    decode_bbox.decode_bboxv2_compute()

