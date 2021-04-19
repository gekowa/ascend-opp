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
batch_multi_class_non_max_suppression
"""
from functools import reduce
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *
from impl.batch_multi_class_nms_topk import sort_within_ub
from impl.batch_multi_class_nms_topk import sort_with_ub


# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements
# pylint: disable=too-many-arguments,unused-argument,too-many-locals,too-many-branches
# scaling factor
DOWN_FACTOR = 0.10
# process 128 proposals at a time
BURST_PROPOSAL_NUM = 128
# RPN compute 16 proposals per iteration
RPN_PROPOSAL_NUM = 16
# define the positive min value in fp16
MIN_SCALAR_FP16 = 2**(-24)
# define a fp16 value = 2**12
TMP_SCALAR_FP16 = 2**12


class BatchMultiClassNonMaxSuppression:
    """
    Function: use to store BatchMultiClassNonMaxSuppression base parameters
    Modify : 2020-7-8
    """
    def __init__(self,
                 boxes,
                 scores,
                 num_valid_boxes,
                 clip_window,
                 score_thresh,
                 iou_thresh,
                 max_size_per_class,
                 max_total_size,
                 change_coordinate_frame,
                 impl_mode):
        """
        Init BatchMultiClassNonMaxSuppression base parameters

        Returns
        -------
        None
        """
        boxes_shape = list(boxes.get("shape"))
        self.boxes_type = boxes.get("dtype")
        scores_shape = list(scores.get("shape"))
        # when input have no class dim, will extend 1 for input shape
        if len(scores_shape) == 2 and len(boxes_shape) == 3:
            self.boxes_shape = [boxes_shape[0], 1, boxes_shape[1], boxes_shape[2]]
            self.scores_shape = [scores_shape[0], 1, scores_shape[1]]
        else:
            self.boxes_shape = boxes_shape
            self.scores_shape = scores_shape

        if clip_window is None:
            self.need_clip_window = False
            self.clip_window_shape = None
        else:
            self.need_clip_window = True
            self.clip_window_shape = clip_window.get("shape")

        if num_valid_boxes is None:
            self.need_valid_num = False
            self.valid_num_shape = None
        else:
            self.need_valid_num = True
            self.valid_num_shape = num_valid_boxes.get("shape")

        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh / (1 + iou_thresh)
        self.max_size_per_class = max_size_per_class
        self.max_total_size = max_total_size
        self.change_coordinate_frame = change_coordinate_frame

        check_shape(self.boxes_shape, min_rank=4, max_rank=4, param_name="boxes")
        check_shape(self.scores_shape, min_rank=3, max_rank=3, param_name="scores")
        # parsing input
        _, self.boxes_classes, _, _ = self.boxes_shape
        self.batch, self.classes, self.boxes_num = self.scores_shape
        self.check_par()
        # whether down the boxes to avoid fp16 overflow
        self.down_flag = True
        self.is_second_nms = False
        if impl_mode == "high_precision":
            self.is_second_nms = True

        self.tik_instance = tik.Tik()
        self.aicore_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.ub_size = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        self.input_gm_list = []
        self.output_gm_list = []

        # calcu output shape
        self.nmsed_boxes_shape = [self.batch, 4, self.max_total_size]
        self.nmsed_scores_shape = [self.batch, self.max_total_size]
        self.nmsed_classes_shape = [self.batch, self.max_total_size]
        self.nmsed_num_shape = [self.batch, 8]

        # for topk
        self.ub_max_topk = None
        self.l1_nms_result = None
        self.l1_nms_result_zero = None
        self.workspace_proposal_gm = None
        self.workspace_second_nms_gm = None
        self.l1_score_valid = None
        self.l1_nms_area = None
        self.l1_nms_sup = None
        self.proposal_topk_k = self.ub_size // 4 // 16
        self.proposal_topk_k = min(self.proposal_topk_k, 255*16)
        self.topk_loop_time = 0
        self.topk_loop_tail = 0
        self.single_loop = True
        if self.boxes_num > self.proposal_topk_k:
            self.single_loop = False
            self.topk_loop_time = self.boxes_num // self.proposal_topk_k
            self.topk_loop_tail = self.boxes_num % self.proposal_topk_k
        self.topk_loop_time_reg = self.tik_instance.Scalar(dtype="int32")
        self.topk_loop_time_reg.set_as(self.topk_loop_time)
        self.topk_loop_time_tail = self.tik_instance.Scalar(dtype="int32")
        self.topk_loop_time_tail.set_as(self.topk_loop_tail)

        # whether user set_rpn_offset, mini do not support it
        self.is_need_rpn_offset = False

        # for nms function param calc
        self.max_selected_nms_num_in_ub = \
            ceil_div(max_size_per_class, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM
        # record the output nms num for one class
        self.selected_proposals_cnt = self.tik_instance.Scalar(dtype="uint16")
        # record the proposal burst num for one loop, value = 128 or self.proposal_topk_k % 128
        self.handling_proposals_cnt = self.tik_instance.Scalar(dtype="uint16")
        # init a scalar value = 0
        self.zero_scalar = self.tik_instance.Scalar(dtype="uint16")
        self.zero_scalar.set_as(0)
        # init a scalar value = 1
        self.one_scalar = self.tik_instance.Scalar(dtype="uint16")
        self.one_scalar.set_as(1)
        # init a fp16 scalar for output class
        self.nms_class_idx = self.tik_instance.Scalar(dtype="float16")
        self.nms_class_idx.set_as(0)
        # init 4 clip to windows scalar
        if self.need_clip_window:
            if self.change_coordinate_frame:
                self.down_flag = False
                self.clip_window_value_list = [self.tik_instance.Scalar(dtype="float16") for _ in range(6)]
            else:
                self.clip_window_value_list = [self.tik_instance.Scalar(dtype="float16") for _ in range(4)]
        else:
            self.clip_window_value_list = None
        # init 1 valid num scalar
        self.valid_num_value = self.tik_instance.Scalar(dtype="int32")

        self.down_scalar_list = None
        # init down scalar
        if self.down_flag:
            self.down_scalar_list = [self.tik_instance.Scalar(dtype="float16") for _ in range(2)]
            self.down_scalar_list[0].set_as(DOWN_FACTOR)
            self.down_scalar_list[1].set_as(1 / DOWN_FACTOR)

    def check_par(self):
        """check_par
        """
        def error_code_002_check(op_name, param_name, value_range, value):
            if value < value_range[0] or value > value_range[1]:
                error_info = {
                    'errCode': OP_ERROR_CODE_002,
                    'op_name': op_name,
                    'param_name': param_name,
                    'min_value': value_range[0],
                    'max_value': value_range[1],
                    'real_value': value
                }
                raise RuntimeError(error_info,
                                   "In op[{op_name}], the parameter[{param_name}] should be in"
                                   " the range of [{min_value}, {max_value}],"
                                   " but actually is [{real_value}].".format(**error_info))

        def error_code_000_check(op_name, param_name, excepted_value, value):
            if excepted_value != value:
                error_info = {
                    'errCode': OP_ERROR_CODE_000,
                    'op_name': op_name,
                    'param_name': param_name,
                    'excepted_value': excepted_value,
                    'real_value': value
                }
                raise RuntimeError(error_info,
                                   "In op[{op_name}], the parameter[{param_name}] should be [{excepted_value}],"
                                   " but actually is [{real_value}].".format(**error_info))

        error_code_002_check("BatchMultiClassNonMaxSuppression", "max_size_per_class",
                             [1, 400], self.max_size_per_class)
        error_code_002_check("BatchMultiClassNonMaxSuppression", "max_total_size",
                             [1, 400], self.max_total_size)
        error_code_002_check("BatchMultiClassNonMaxSuppression", "classes num from input scores shape",
                             [1, 200], self.classes)

        if not self.need_clip_window:
            error_code_000_check("BatchMultiClassNonMaxSuppression",
                                 "change_coordinate_frame(when don't do clip_window)",
                                 False,
                                 self.change_coordinate_frame)

        if self.need_valid_num:
            error_code_002_check("BatchMultiClassNonMaxSuppression", "input boxes num(when need valid_num)",
                                 [1, 1024], self.scores_shape[2])

        check_dtype(self.boxes_type, ("float16",), param_name="boxes")

    def get_tik_instance(self):
        """get_tik_instance
        """
        return self.tik_instance

    def build_tik_instance(self, kernel_name_value):
        """build_tik_instance
        """
        self.tik_instance.BuildCCE(kernel_name=kernel_name_value,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   output_files_path=None,
                                   enable_l2=False)

        return self.tik_instance

    def init_tik_mem(self):
        """init tik gm mem
        """
        # init gm input
        boxes_gm = self.tik_instance.Tensor("float16", self.boxes_shape, name="boxes_gm", scope=tik.scope_gm)
        scores_gm = self.tik_instance.Tensor("float16", self.scores_shape, name="scores_gm", scope=tik.scope_gm)

        clip_window_gm = None
        valid_num_gm = None
        if self.need_clip_window:
            clip_window_gm = self.tik_instance.Tensor("float16", self.clip_window_shape,
                                                      name="clip_window_gm", scope=tik.scope_gm)
        if self.need_valid_num:
            valid_num_gm = self.tik_instance.Tensor("int32", self.valid_num_shape,
                                                    name="valid_num_gm", scope=tik.scope_gm)
        if self.need_valid_num and self.need_clip_window:
            self.input_gm_list = [boxes_gm, scores_gm, clip_window_gm, valid_num_gm]
        elif self.need_clip_window:
            self.input_gm_list = [boxes_gm, scores_gm, clip_window_gm]
        elif self.need_valid_num:
            self.input_gm_list = [boxes_gm, scores_gm, valid_num_gm]
        else:
            self.input_gm_list = [boxes_gm, scores_gm]

        # init gm output
        nmsed_boxes_gm = self.tik_instance.Tensor("float16", self.nmsed_boxes_shape,
                                                  name="nmsed_boxes_gm", scope=tik.scope_gm)
        nmsed_scores_gm = self.tik_instance.Tensor("float16", self.nmsed_scores_shape,
                                                   name="nmsed_scores_gm", scope=tik.scope_gm)
        nmsed_classes_gm = self.tik_instance.Tensor("float16", self.nmsed_classes_shape,
                                                    name="nmsed_classes_gm", scope=tik.scope_gm)
        nmsed_num_gm = self.tik_instance.Tensor("int32", self.nmsed_num_shape,
                                                name="nmsed_num_gm", scope=tik.scope_gm)
        self.output_gm_list = [nmsed_boxes_gm, nmsed_scores_gm, nmsed_classes_gm, nmsed_num_gm]

        # init l1 buff for save multi class nms result, size = [classes, self.max_selected_nms_num_in_ub, 8]
        self.l1_nms_result = self.tik_instance.Tensor("float16", (self.classes, self.max_selected_nms_num_in_ub, 8),
                                                      name="l1_nms_result", scope=tik.scope_cbuf)

        if self.is_second_nms:
            # init l1 buff for save multi class nms area, size = [self.max_selected_nms_num_in_ub]
            self.l1_nms_area = self.tik_instance.Tensor("float16", (self.max_selected_nms_num_in_ub,),
                                                        name="l1_nms_area_tmp", scope=tik.scope_cbuf)
            # init l1 buff for save multi class nms sup, size = [self.max_selected_nms_num_in_ub]
            self.l1_nms_sup = self.tik_instance.Tensor("uint16", (self.max_selected_nms_num_in_ub,),
                                                       name="l1_nms_sup_tmp", scope=tik.scope_cbuf)

        # zero data in l1
        self.l1_nms_result_zero = \
            self.tik_instance.Tensor("float16", (self.max_selected_nms_num_in_ub, 8),
                                     name="l1_nms_result_zero", scope=tik.scope_cbuf)
        with self.tik_instance.new_stmt_scope():
            ub_nms_result = self.tik_instance.Tensor("float16", (self.max_selected_nms_num_in_ub, 8),
                                                     name="ub_nms_result", scope=tik.scope_ubuf)
            tik_func_vector(self.tik_instance, ub_nms_result, 0, self.max_selected_nms_num_in_ub*8)
            loop_burst_len = (self.max_selected_nms_num_in_ub*8) // 16
            self.tik_instance.data_move(self.l1_nms_result_zero,
                                        ub_nms_result, 0, 1, loop_burst_len, 0, 0)
        # workspace
        self.workspace_proposal_gm = self.tik_instance.Tensor("float16",
                                                              [self.aicore_num,
                                                               total_num(self.l1_nms_result.shape) + 128],
                                                              name="workspace_proposal_gm",
                                                              scope=tik.scope_gm, is_workspace=True)
        # workspace for second nms
        if self.is_second_nms:
            self.workspace_second_nms_gm = self.tik_instance.Tensor("float16",
                                                                    [self.aicore_num,
                                                                     self.boxes_num*8],
                                                                    name="workspace_second_nms_gm",
                                                                    scope=tik.scope_gm, is_workspace=True)
        if self.need_valid_num:
            self.l1_score_valid = self.tik_instance.Tensor("float16", (ceil_div(self.boxes_num, 16)*16,),
                                                           name="l1_score_valid", scope=tik.scope_cbuf)

    def init_tik_ub_mem_for_nms(self):
        """init_tik_ub_mem_for_nms
        """
        ub_selected_proposals = self.tik_instance.Tensor("float16", [self.max_selected_nms_num_in_ub, 8],
                                                         name="ub_selected_proposals", scope=tik.scope_ubuf)
        ub_selected_area = self.tik_instance.Tensor("float16", [self.max_selected_nms_num_in_ub],
                                                    name="ub_selected_area", scope=tik.scope_ubuf)
        ub_sup_vec = self.tik_instance.Tensor("uint16", [self.max_selected_nms_num_in_ub], name="ub_sup_vec",
                                              scope=tik.scope_ubuf)

        # when is_need_rpn_offset set rpn offset for vaadd and viou
        # else x2/y2 will do vadds -1 before nms and do vadds 1 after nms
        if self.is_need_rpn_offset:
            self.tik_instance.set_rpn_offset(0.0)

        topk_out_num = self.proposal_topk_k
        if self.boxes_num < self.proposal_topk_k:
            topk_out_num = self.boxes_num
        nms_var_dict = {
            # topk_out_info mean : nms input info
            "topk_out_ub": self.ub_max_topk,
            "topk_out_num": topk_out_num,
            # selected proposal info
            "selected_proposal_ub": ub_selected_proposals,
            "selected_area_ub": ub_selected_area,
            "sup_vec_ub": ub_sup_vec,
            # scalar reg info
            "zero_scalar": self.zero_scalar,
            "one_scalar": self.one_scalar,
            "selected_proposals_cnt": self.selected_proposals_cnt,
            "handling_proposals_cnt": self.handling_proposals_cnt,
            # nms output info
            "output_num": self.max_size_per_class
        }

        return nms_var_dict

    def init_tik_ub_mem_for_topk(self):
        """init_tik_ub_mem_for_topk
        """
        # init one ub for topk output
        self.ub_max_topk = self.tik_instance.Tensor("float16", (self.proposal_topk_k, 8),
                                                    name="ub_max_topk", scope=tik.scope_ubuf)

    def get_core_schedule(self):
        """get_core_schedule
        """
        if self.max_total_size < 16:
            self.aicore_num = 1
        batch_per_core = ceil_div(self.batch, self.aicore_num)
        core_used = ceil_div(self.batch, batch_per_core)
        batch_last_core = self.batch - (core_used - 1) * batch_per_core
        self.aicore_num = core_used

        return core_used, batch_per_core, batch_last_core


def total_num(shape):
    """return total_num"""
    shape_total_num = reduce(lambda a, b: a*b, shape)
    return shape_total_num


def read_valid_num_compute(tik_instance, input_window_gm, offset, scalar):
    """read_valid_num_compute
    """
    with tik_instance.new_stmt_scope():
        input_window_ub = tik_instance.Tensor(
            input_window_gm.dtype,
            (8,),
            name="input_window_ub",
            scope=tik.scope_ubuf)
        tik_instance.data_move(input_window_ub,
                               input_window_gm[offset],
                               0, 1, 1, 0, 0)
        scalar.set_as(input_window_ub[0])


def gen_valid_num_compute(tik_instance, l1_output, input_len, scalar):
    """gen_valid_num_compute
    """
    with tik_instance.new_stmt_scope():
        input_window_ub = tik_instance.Tensor(
            l1_output.dtype,
            (ceil_div(input_len, 16) * 16,),
            name="input_window_ub",
            scope=tik.scope_ubuf)
        tik_func_vector(tik_instance, input_window_ub, 0.0, ceil_div(input_len, 16) * 16)
        with tik_instance.if_scope(scalar // 128 > 0):
            tik_instance.vector_dup(128, input_window_ub, 1.0, scalar // 128, 1, 8)
        with tik_instance.if_scope(scalar % 128 > 0):
            tik_instance.vector_dup(scalar % 128, input_window_ub[(scalar // 128) * 128], 1.0, 1, 1, 8)
        tik_instance.data_move(l1_output, input_window_ub,
                               0, 1, ceil_div(input_len, 16), 0, 0)


def valid_num_compute(tik_instance, l1_valid_mask, ub_tmp_score, copy_num):
    """valid_num_compute
    """
    if l1_valid_mask is None:
        return
    input_window_ub = tik_instance.Tensor(
        l1_valid_mask.dtype,
        l1_valid_mask.shape,
        name="input_window_ub",
        scope=tik.scope_ubuf)
    tik_instance.data_move(input_window_ub,
                           l1_valid_mask,
                           0, 1, l1_valid_mask.shape[0] // 16, 0, 0)
    tik_func_vcomple(tik_instance, "vmul", ub_tmp_score, input_window_ub, ub_tmp_score, ceil_div(copy_num, 16) * 16)


def change_coordinate_frame_compute(tik_instance, clip_window_value_list, ub_tmp_boxes, do_num):
    """change_coordinate_frame_compute
    """
    if clip_window_value_list is None:
        # no need to do clip_window and change_coordinate_frame
        return
    is_need_change_coordinate_frame = False
    if len(clip_window_value_list) == 6:
        is_need_change_coordinate_frame = True

    if is_need_change_coordinate_frame:
        h_scale_scale = clip_window_value_list[4]
        w_scale_scale = clip_window_value_list[5]
        tik_func_vmuls(tik_instance, ub_tmp_boxes[0, :], ub_tmp_boxes[0, :], h_scale_scale, do_num)
        tik_func_vmuls(tik_instance, ub_tmp_boxes[1, :], ub_tmp_boxes[1, :], w_scale_scale, do_num)
        tik_func_vmuls(tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], h_scale_scale, do_num)
        tik_func_vmuls(tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], w_scale_scale, do_num)


def clip_window_compute(tik_instance, input_gm_list, input_ub_list, gm_offset, scalar_window,
                        copy_num, data_each_block=16):
    """clip_window_compute
    """
    input_num_boxes = input_ub_list[0].shape[1]

    dtype = input_ub_list[0].dtype
    win_y_min = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="win_y_min",
        scope=tik.scope_ubuf)

    win_x_min = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="win_x_min",
        scope=tik.scope_ubuf)

    win_y_max = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="win_y_man",
        scope=tik.scope_ubuf)

    win_x_max = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="win_x_max",
        scope=tik.scope_ubuf)

    zero_tensor = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="zero",
        scope=tik.scope_ubuf)
    # min float16 value

    min_fp16 = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="min_fp16",
        scope=tik.scope_ubuf)

    concst24 = tik_instance.Tensor(
        dtype,
        (data_each_block,),
        name="concst24",
        scope=tik.scope_ubuf)

    sub1 = tik_instance.Tensor(
        dtype,
        (input_num_boxes,),
        name="sub1",
        scope=tik.scope_ubuf)

    sub2 = tik_instance.Tensor(
        dtype,
        (input_num_boxes,),
        name="sub2",
        scope=tik.scope_ubuf)

    nburst = 1

    input_boxes_gm = input_gm_list[0]
    input_sorces_gm = input_gm_list[1]
    input_boxes_ub = input_ub_list[0]
    input_scorces_ub = input_ub_list[1]

    boxes_class = input_boxes_gm.shape[1]
    burse_len = math.ceil(copy_num / data_each_block)
    for i in range(4):
        if boxes_class == 1:
            class_offset = 0
        else:
            class_offset = gm_offset[1]
        tik_instance.data_move(input_boxes_ub[i, 0],
                               input_boxes_gm[gm_offset[0], class_offset, i, gm_offset[3]],
                               0, nburst,
                               burse_len, 0, 0)

    tik_instance.data_move(input_scorces_ub,
                           input_sorces_gm[gm_offset[0], gm_offset[1], gm_offset[3]],
                           0, nburst,
                           burse_len, 0, 0)
    if scalar_window is None:
        # no need to do clip to window, return directly
        return
    index_win_y_min = scalar_window[0]
    index_win_x_min = scalar_window[1]
    index_win_y_max = scalar_window[2]
    index_win_x_max = scalar_window[3]
    tik_instance.vector_dup(data_each_block,
                            win_y_min,
                            index_win_y_min,
                            1, 1, 1)
    tik_instance.vector_dup(data_each_block,
                            win_x_min,
                            index_win_x_min,
                            1, 1, 1)
    tik_instance.vector_dup(data_each_block,
                            win_y_max,
                            index_win_y_max,
                            1, 1, 1)
    tik_instance.vector_dup(data_each_block,
                            win_x_max,
                            index_win_x_max,
                            1, 1, 1)

    y_min_input = input_boxes_ub[0*input_num_boxes:]
    x_min_input = input_boxes_ub[1*input_num_boxes:]
    y_max_input = input_boxes_ub[2*input_num_boxes:]
    x_max_input = input_boxes_ub[3*input_num_boxes:]
    y_min_out = input_boxes_ub[0*input_num_boxes:]
    x_min_out = input_boxes_ub[1*input_num_boxes:]
    y_max_out = input_boxes_ub[2*input_num_boxes:]
    x_max_out = input_boxes_ub[3*input_num_boxes:]

    def tik_func_vmin_vmax(tik_instance, vmin_or_max,
                           out_dst, src0, src1, copy_num,
                           dst_blk, src0_blk, src1_blk,
                           dst_rep, src0_rep, src1_rep):
        repeat_time = copy_num // 128
        repeat_tail = copy_num % 128
        tik_fun = None
        if vmin_or_max == "vmin":
            tik_fun = tik_instance.vmin

        if vmin_or_max == "vmax":
            tik_fun = tik_instance.vmax

        if vmin_or_max == "vsub":
            tik_fun = tik_instance.vsub

        if vmin_or_max == "vmul":
            tik_fun = tik_instance.vmul

        if repeat_time > 0:
            tik_fun(128, out_dst, src0[0], src1[0],
                    repeat_time,
                    dst_blk, src0_blk, src1_blk,
                    dst_rep, src0_rep, src1_rep)

        if repeat_tail > 0:
            offset = repeat_time * 128
            tik_fun(repeat_tail, out_dst[offset], src0[offset], src1[0],
                    1,
                    dst_blk, src0_blk, src1_blk,
                    dst_rep, src0_rep, src1_rep)

    tik_func_vmin_vmax(tik_instance, "vmin", y_min_out, y_min_input, win_y_max,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmin", y_max_out, y_max_input, win_y_max,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmin", x_min_out, x_min_input, win_x_max,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmin", x_max_out, x_max_input, win_x_max,
                       copy_num, 1, 1, 0, 8, 8, 0)

    tik_func_vmin_vmax(tik_instance, "vmax", y_min_out, y_min_out, win_y_min,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmax", y_max_out, y_max_out, win_y_min,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmax", x_min_out, x_min_out, win_x_min,
                       copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vmin_vmax(tik_instance, "vmax", x_max_out, x_max_out, win_x_min,
                       copy_num, 1, 1, 0, 8, 8, 0)

    # get (y_max_clipped - y_min_clipped) * (x_max_clipped - x_min_clipped)
    tik_func_vmin_vmax(tik_instance, "vsub", sub1, y_max_out, y_min_out,
                       copy_num, 1, 1, 1, 8, 8, 8)
    tik_func_vmin_vmax(tik_instance, "vsub", sub2, x_max_out, x_min_out,
                       copy_num, 1, 1, 1, 8, 8, 8)
    tik_func_vmin_vmax(tik_instance, "vmul", sub1, sub1, sub2,
                       copy_num, 1, 1, 1, 8, 8, 8)

    tik_func_vector(tik_instance, zero_tensor, 0.0, 16)
    tik_func_vector(tik_instance, min_fp16, MIN_SCALAR_FP16, 16)
    tik_func_vector(tik_instance, concst24, TMP_SCALAR_FP16, 16)

    tik_func_vcomple(tik_instance, "vmin", sub1, sub1, min_fp16,
                     copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vcomple(tik_instance, "vmax", sub1, sub1, zero_tensor,
                     copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vcomple(tik_instance, "vmul", sub1, sub1, concst24,
                     copy_num, 1, 1, 0, 8, 8, 0)
    tik_func_vcomple(tik_instance, "vmul", sub1, sub1, concst24,
                     copy_num, 1, 1, 0, 8, 8, 0)

    # modify score = 0 when area <= 0
    tik_func_vmin_vmax(tik_instance, "vmul", input_scorces_ub,
                       input_scorces_ub, sub1,
                       copy_num, 1, 1, 1, 8, 8, 8)


def read_window_compute(tik_instance, input_window_gm, offset, scalar_list, down_scalar_list,
                        change_coordinate_frame=False):
    """read_window_compute
    """
    with tik_instance.new_stmt_scope():
        input_window_ub = tik_instance.Tensor(
            input_window_gm.dtype,
            (16*2,),
            name="input_window_ub",
            scope=tik.scope_ubuf)
        tik_instance.data_move(input_window_ub,
                               input_window_gm[offset],
                               0, 1, 1, 0, 0)

        [index_win_y_min, index_win_x_min, index_win_y_max, index_win_x_max] = \
            [scalar_list[0], scalar_list[1], scalar_list[2], scalar_list[3]]
        index_win_y_min.set_as(input_window_ub[0])
        index_win_x_min.set_as(input_window_ub[1])
        index_win_y_max.set_as(input_window_ub[2])
        index_win_x_max.set_as(input_window_ub[3])

        if down_scalar_list is not None:
            input_window_ub_int32 = tik_instance.Tensor(
                "int32",
                (16,),
                name="input_window_ub_int32",
                scope=tik.scope_ubuf)
            tik_instance.vconv(4, "round", input_window_ub_int32, input_window_ub,
                               1, 1, 1, 8, 4)
            max_h = tik_instance.Scalar(dtype="int32")
            max_w = tik_instance.Scalar(dtype="int32")
            max_h.set_as(input_window_ub_int32[2])
            max_w.set_as(input_window_ub_int32[3])
            with tik_instance.if_scope(max_h*max_w < 200*200):
                down_scalar_list[0].set_as(1.0)
                down_scalar_list[1].set_as(1.0)

        if change_coordinate_frame:
            [_, _, _, _, scale_h, scale_w] = scalar_list
            last_offset = offset[-1] + 2
            offset[-1] = last_offset
            tik_instance.data_move(input_window_ub[16],
                                   input_window_gm[offset],
                                   0, 1, 1, 0, 0)
            tik_func_vcomple(tik_instance, "vsub", input_window_ub, input_window_ub[16], input_window_ub, 16)
            # do rec in mini
            tik_instance.vrec(2, input_window_ub[16], input_window_ub, 1, 1, 1, 8, 8)
            tik_fuc_vrec_newton(tik_instance, input_window_ub[16], input_window_ub, 2)
            scale_h.set_as(input_window_ub[16])
            scale_w.set_as(input_window_ub[17])


def ceil_div(value, factor):
    """Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


def ub_offset(input_ub):
    """get ub offset
    when ub.shape is 1D tensor offset = 0
    when ub.shape is not 1D tensor change offset = 1D
    ex:
       ub.shape = [2,2,2]
       ub1 = ub[1,:,:]
       ub_offset(ub1) = 2*2 = 4 for ub
    """
    ub_shape = input_ub.shape
    if len(ub_shape) in (0, 1):
        return 0

    return input_ub.offset


def tik_fuc_vrec_newton(tik_instance, vrec_ub, origin_ub, do_len, newton_iteration=2, block_num=16):
    """tik_fuc_vrec_newton
    """
    with tik_instance.new_stmt_scope():
        vrec_newton_1 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_1", scope=tik.scope_ubuf)
        vrec_newton_2 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_2", scope=tik.scope_ubuf)

        def _one_newton():
            tik_instance.vmul(2, vrec_newton_1, vrec_ub, origin_ub, 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmuls(2, vrec_newton_2, vrec_newton_1, -1, 1, 1, 1, 8, 8)
            tik_instance.vadds(2, vrec_newton_1, vrec_newton_2, 2, 1, 1, 1, 8, 8)
            tik_instance.vmul(2, vrec_ub, vrec_newton_1, vrec_ub, 1, 1, 1, 1, 8, 8, 8)

        for _ in range(newton_iteration):
            _one_newton()


def tik_func_vcomple(tik_instance, function, out_dst, src0, src1, copy_num,
                     dst_blk=1, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8,
                     src1_rep=8):
    """tik_func_vcomple
    """
    do_dtype = out_dst.dtype
    if do_dtype in ("float16",):
        block_num = 16
    else:
        block_num = 8
    vector_num = block_num*8
    repeat_time = copy_num // vector_num
    repeat_tail = copy_num % vector_num
    tik_fun = None
    ori_offset_dst = ub_offset(out_dst)
    ori_offset_src0 = ub_offset(src0)
    ori_offset_src1 = ub_offset(src1)
    if function == "vmin":
        tik_fun = tik_instance.vmin
    elif function == "vmax":
        tik_fun = tik_instance.vmax
    elif function == "vmul":
        tik_fun = tik_instance.vmul
    elif function == "vadd":
        tik_fun = tik_instance.vadd
    elif function == "vsub":
        tik_fun = tik_instance.vsub

    while repeat_time > 255:
        tik_fun(vector_num,
                out_dst[ori_offset_dst],
                src0[ori_offset_src0],
                src1[ori_offset_src1],
                255,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)
        repeat_time = repeat_time - 255
        ori_offset_dst = ori_offset_dst + 255 * block_num * dst_rep
        ori_offset_src0 = ori_offset_src0 + 255 * block_num * src0_rep
        ori_offset_src1 = ori_offset_src1 + 255 * block_num * src1_rep

    if repeat_time > 0:
        tik_fun(vector_num,
                out_dst[ori_offset_dst],
                src0[ori_offset_src0],
                src1[ori_offset_src1],
                repeat_time,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)
        ori_offset_dst = ori_offset_dst + repeat_time * block_num * dst_rep
        ori_offset_src0 = ori_offset_src0 + repeat_time * block_num * src0_rep
        ori_offset_src1 = ori_offset_src1 + repeat_time * block_num * src1_rep

    if repeat_tail > 0:
        tik_fun(repeat_tail,
                out_dst[ori_offset_dst],
                src0[ori_offset_src0],
                src1[ori_offset_src1],
                1,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)


def do_nms_compute(tik_instance, nms_var_dict, thresh):
    """Compute output boxes after non-maximum suppression.

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    nms_var_dict: dict
        the input par for nms, keys as follows:
            topk_out_num: the num proposal to do nms
            output_num: total output nms proposals
            topk_out_ub: the sorted proposals ub
            selected_proposal_ub: the selected_proposal_ub, save selected proposal
            selected_area_ub: the selected_area_ub, save selected proposal area
            sup_vec_ub: sup_vec_ub
            handling_proposals_cnt: a uint16 scalar
            selected_proposals_cnt: a uint16 scalar, specifying the selected proposal num
            zero_scalar : a uint16 scalar, value = 0
    thresh: float
        iou thresh for nms
    """
    total_input_proposal_num = nms_var_dict.get("topk_out_num")
    total_output_proposal_num = nms_var_dict.get("output_num")
    ub_max_topk = nms_var_dict.get("topk_out_ub")
    ub_selected_proposals = nms_var_dict.get("selected_proposal_ub")
    ub_selected_area = nms_var_dict.get("selected_area_ub")
    ub_sup_vec = nms_var_dict.get("sup_vec_ub")
    handling_proposals_cnt = nms_var_dict.get("handling_proposals_cnt")
    selected_proposals_cnt = nms_var_dict.get("selected_proposals_cnt")
    zero_scalar = nms_var_dict.get("zero_scalar")
    # variables
    left_proposal_cnt = tik_instance.Scalar(dtype="uint16")
    left_proposal_cnt.set_as(total_input_proposal_num)
    # store the whole
    # change with burst
    temp_proposals_ub = tik_instance.Tensor("float16", [BURST_PROPOSAL_NUM, 8],
                                            name="temp_proposals_ub", scope=tik.scope_ubuf)
    tik_instance.vector_dup(128, temp_proposals_ub[0], 0, 8, 1, 8)
    temp_area_ub = tik_instance.Tensor("float16", [BURST_PROPOSAL_NUM],
                                       name="temp_area_ub", scope=tik.scope_ubuf)
    temp_iou_ub = \
        tik_instance.Tensor("float16", [ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM + 128,
                                        16],
                            name="temp_iou_ub", scope=tik.scope_ubuf)
    temp_join_ub = \
        tik_instance.Tensor("float16", [ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM + 128,
                                        16],
                            name="temp_join_ub", scope=tik.scope_ubuf)
    temp_sup_matrix_ub = \
        tik_instance.Tensor("uint16", [ceil_div(total_output_proposal_num, RPN_PROPOSAL_NUM) * RPN_PROPOSAL_NUM + 128],
                            name="temp_sup_matrix_ub", scope=tik.scope_ubuf)
    temp_sup_vec_ub = tik_instance.Tensor("uint16", [BURST_PROPOSAL_NUM],
                                          name="temp_sup_vec_ub", scope=tik.scope_ubuf)
    # main body
    nms_flag = tik_instance.Scalar(dtype="uint16")
    nms_flag.set_as(0)
    with tik_instance.for_range(0, ceil_div(total_input_proposal_num, BURST_PROPOSAL_NUM)) as burst_index:
        # update counter
        with tik_instance.if_scope(left_proposal_cnt < BURST_PROPOSAL_NUM):
            handling_proposals_cnt.set_as(left_proposal_cnt)
        with tik_instance.else_scope():
            handling_proposals_cnt.set_as(BURST_PROPOSAL_NUM)

        handling_ceil = tik_instance.Scalar(dtype="uint16")
        handling_ceil.set_as(ceil_div(handling_proposals_cnt, 16))
        selected_ceil = tik_instance.Scalar(dtype="uint16")
        selected_ceil.set_as(ceil_div(selected_proposals_cnt, 16))
        # clear temp_sup_vec_ub
        tik_instance.vector_dup(128, temp_sup_vec_ub[0], 1, temp_sup_vec_ub.shape[0] // BURST_PROPOSAL_NUM, 1, 8)
        temp_proposals_ub = ub_max_topk[burst_index*BURST_PROPOSAL_NUM*8]
        # calculate the area of reduced-proposal
        tik_instance.vrpac(temp_area_ub[0], temp_proposals_ub[0], handling_ceil)
        # start to update iou and or area from the first 16 proposal
        # and get suppression vector 16 by 16 proposal
        length = tik_instance.Scalar(dtype="uint16")
        length.set_as(selected_ceil * 16)
        # length.set_as(selected_proposals_cnt)
        with tik_instance.if_scope(selected_proposals_cnt < total_output_proposal_num):
            with tik_instance.new_stmt_scope():
                with tik_instance.for_range(0, handling_ceil) as i:
                    length.set_as(length + 16)
                    # calculate intersection of tempReducedProposals
                    # and selReducedProposals
                    tik_instance.viou(temp_iou_ub[0, 0], ub_selected_proposals[0],
                                      temp_proposals_ub[i*16*8], selected_ceil)
                    # calculate intersection of tempReducedProposals and
                    # tempReducedProposals(include itself)
                    tik_instance.viou(temp_iou_ub[selected_ceil * 16, 0],
                                      temp_proposals_ub[0], temp_proposals_ub[i*16*8], i + 1)
                    # calculate join of tempReducedProposals
                    # and selReducedProposals
                    tik_instance.vaadd(temp_join_ub[0, 0], ub_selected_area[0], temp_area_ub[i * 16],
                                       selected_ceil)
                    # calculate intersection of tempReducedProposals and
                    # tempReducedProposals(include itself)
                    tik_instance.vaadd(temp_join_ub[selected_ceil * 16, 0],
                                       temp_area_ub, temp_area_ub[i * 16], i + 1)
                    # calculate join*(thresh/(1+thresh))
                    tik_instance.vmuls(128, temp_join_ub[0, 0], temp_join_ub[0, 0], thresh,
                                       ceil_div(length, 8), 1, 1, 8, 8)
                    # compare and generate suppression matrix
                    tik_instance.vcmpv_gt(temp_sup_matrix_ub[0], temp_iou_ub[0, 0], temp_join_ub[0, 0],
                                          ceil_div(length, 8), 1, 1, 8, 8)
                    # generate suppression vector
                    # clear rpn_cor_ir
                    rpn_cor_ir = tik_instance.set_rpn_cor_ir(0)
                    # non-diagonal
                    rpn_cor_ir = tik_instance.rpn_cor(temp_sup_matrix_ub[0], ub_sup_vec[0], 1, 1,
                                                      selected_ceil)
                    with tik_instance.if_scope(i > 0):
                        rpn_cor_ir = \
                            tik_instance.rpn_cor(temp_sup_matrix_ub[selected_ceil * 16],
                                                 temp_sup_vec_ub[0], 1, 1, i)
                    # diagonal
                    tik_instance.rpn_cor_diag(temp_sup_vec_ub[i * 16], temp_sup_matrix_ub[length - 16], rpn_cor_ir)

                # find & mov unsuppressed proposals
                with tik_instance.for_range(0, handling_proposals_cnt) as i:
                    with tik_instance.if_scope(selected_proposals_cnt < total_output_proposal_num):
                        nms_flag.set_as(temp_sup_vec_ub[i])
                        with tik_instance.if_scope(nms_flag == 0):
                            ub_selected_proposals_uint64 = ub_selected_proposals.reinterpret_cast_to("uint64")
                            temp_proposals_ub_uint64 = temp_proposals_ub.reinterpret_cast_to("uint64")
                            ub_selected_proposals_uint64[selected_proposals_cnt*2 + 0] = \
                                temp_proposals_ub_uint64[i*2 + 0]
                            ub_selected_proposals_uint64[selected_proposals_cnt*2 + 1] = \
                                temp_proposals_ub_uint64[i*2 + 1]

                            ub_selected_area[selected_proposals_cnt] = temp_area_ub[i]
                            # update sup_vec_ub
                            ub_sup_vec[selected_proposals_cnt].set_as(zero_scalar)
                            # update counter
                            selected_proposals_cnt.set_as(selected_proposals_cnt + 1)
            left_proposal_cnt.set_as(left_proposal_cnt - handling_proposals_cnt)


def tik_func_vconcat(tik_instance, proposals_ub, _ub, trans_repeat, mode):
    """tik_func_vconcat
    """
    tik_instance.vconcat(proposals_ub, _ub, trans_repeat, mode)


def tik_func_vextract(tik_instance, proposals_ub, _ub, trans_repeat, mode):
    """tik_func_vextract
    """
    tik_instance.vextract(_ub, proposals_ub, trans_repeat, mode)


def tik_func_vadds(tik_instance, dst_ub, src_ub, value, do_len):
    """tik_func_vadds
    """
    repeat = do_len // 128
    repeat_tail = do_len % 128
    offset = ub_offset(src_ub)
    while repeat > 255:
        tik_instance.vadds(128, dst_ub[offset], src_ub[offset], value,
                           255, 1, 1, 8, 8)
        repeat = repeat - 255
        offset = offset + 128*255
    if repeat > 0:
        tik_instance.vadds(128, dst_ub[offset], src_ub[offset], value,
                           repeat, 1, 1, 8, 8)
        offset = offset + 128*repeat
    if repeat_tail > 0:
        tik_instance.vadds(repeat_tail, dst_ub[offset], src_ub[offset], value,
                           1, 1, 1, 8, 8)


def tik_func_vmuls(tik_instance, dst_ub, src_ub, value, do_len):
    """tik_func_vmuls
    """
    repeat = do_len // 128
    repeat_tail = do_len % 128
    offset = dst_ub.offset
    while repeat > 255:
        tik_instance.vmuls(128, dst_ub[offset], src_ub[offset], value,
                           255, 1, 1, 8, 8)
        repeat = repeat - 255
        offset = offset + 128*255
    if repeat > 0:
        tik_instance.vmuls(128, dst_ub[offset], src_ub[offset], value,
                           repeat, 1, 1, 8, 8)
        offset = offset + 128*repeat
    if repeat_tail > 0:
        tik_instance.vmuls(repeat_tail, dst_ub[offset], src_ub[offset], value,
                           1, 1, 1, 8, 8)


def tik_func_vector(tik_instance, _ub, value, dup_len):
    """tik_func_vector

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    _ub: ub
        vcetor ub
    value: value
        vcetor value
    dup_len: int
        vcetor data len

    Returns
    -------
    None
    """
    repeat = dup_len // 128
    repeat_tail = dup_len % 128
    offset = 0
    while repeat > 255:
        tik_instance.vector_dup(128, _ub[offset], value, 255, 1, 8)
        repeat = repeat - 255
        offset = offset + 128*255
    if repeat > 0:
        tik_instance.vector_dup(128, _ub[offset], value, repeat, 1, 8)
        offset = offset + 128*repeat
    if repeat_tail > 0:
        tik_instance.vector_dup(repeat_tail, _ub[offset], value, 1, 1, 8)


def tik_func_trans_to_proposals(tik_instance, proposals_ub, boxes_ub_list, score_ub, proposal_num):
    """tik_func_trans_to_proposals
    """
    x1_ub, y1_ub, x2_ub, y2_ub = boxes_ub_list
    trans_repeat = ceil_div(proposal_num, 16)
    # concat x1 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, x1_ub, trans_repeat, 0)
    # concat y1 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, y1_ub, trans_repeat, 1)
    # concat x2 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, x2_ub, trans_repeat, 2)
    # concat y2 to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, y2_ub, trans_repeat, 3)
    # concat scores to proposals_ub
    tik_func_vconcat(tik_instance, proposals_ub, score_ub, trans_repeat, 4)


def get_sorted_proposal_compute(tik_instance, output_ub, input_gm_list, gm_offset, copy_num,
                                sorted_num, clip_window_value_list, l1_valid_mask, reduce_scalar=None,
                                rpn_enble=False):
    """get_sorted_proposal_compute
    main function do copy boxes/scores, clip_window, change_coordinate, trans_to_proposals and sort

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    output_ub : ub
        output ub, save the sorted proposal list
    input_gm_list : list
        input gm list
    gm_offset : list
        input gm offset for score
    copy_num: int
        the process boxes num one time for copy
    sorted_num: int
        the sort boxes num one time for sort
    clip_window_value_list: list
        the window scalar list
    l1_valid_mask: cbuf
        num valid mask
    reduce_flag: bool
        whether reduce all box to avoid iou/vaadd overflows
    rpn_enble: bool
        whether support rpn

    Returns
    -------
    None
    """
    with tik_instance.new_stmt_scope():
        # apply ub for boxes copy_gm_to_ub
        ub_tmp_boxes = tik_instance.Tensor("float16", [4, sorted_num],
                                           name="copy_ub_tmp_boxes", scope=tik.scope_ubuf)
        # apply ub for score copy_gm_to_ub
        ub_tmp_score = tik_instance.Tensor("float16", [1, sorted_num],
                                           name="copy_ub_tmp_score", scope=tik.scope_ubuf)

        # step 1- copy boxes to ub with copy_num
        # step 2- copy scores to ub with copy_num
        # step 3- clip boxes and update scores
        input_ub_list = [ub_tmp_boxes, ub_tmp_score]
        with tik_instance.new_stmt_scope():
            clip_window_compute(tik_instance, input_gm_list, input_ub_list, gm_offset, clip_window_value_list,
                                copy_num)
        # DOWN_FACTOR
        if reduce_scalar is not None:
            tik_func_vmuls(tik_instance, ub_tmp_boxes[0, :], ub_tmp_boxes[0, :], reduce_scalar[0], copy_num)
            tik_func_vmuls(tik_instance, ub_tmp_boxes[1, :], ub_tmp_boxes[1, :], reduce_scalar[0], copy_num)
            tik_func_vmuls(tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], reduce_scalar[0], copy_num)
            tik_func_vmuls(tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], reduce_scalar[0], copy_num)

        # step 4- filter valid num
        with tik_instance.new_stmt_scope():
            valid_num_compute(tik_instance, l1_valid_mask, ub_tmp_score, copy_num)

        # step 5- change_coordinate_frame if len(clip_window_value_list) == 6. will do change_coordinate_frame
        with tik_instance.new_stmt_scope():
            change_coordinate_frame_compute(tik_instance, clip_window_value_list, ub_tmp_boxes, copy_num)

        if not rpn_enble:
            # x2  y2 sub 1 for iou RPN_offset
            tik_func_vadds(tik_instance, ub_tmp_boxes[2, :], ub_tmp_boxes[2, :], -1.0, copy_num)
            tik_func_vadds(tik_instance, ub_tmp_boxes[3, :], ub_tmp_boxes[3, :], -1.0, copy_num)

        # step 6- trans to proposal
        boxes_list = [ub_tmp_boxes[0, 0], ub_tmp_boxes[1, 0], ub_tmp_boxes[2, 0], ub_tmp_boxes[3, 0]]

        # vecter_dup the tail score = 0
        if copy_num % 16 != 0:
            dup_mask = int("0" * 48 + "1" * (16 - (copy_num % 16)) + "0" * (copy_num % 16), 2)
            tik_instance.vector_dup([0, dup_mask], ub_tmp_score[(copy_num // 16) * 16], 0.0, 1, 1, 8)

        tik_func_trans_to_proposals(tik_instance, output_ub, boxes_list, ub_tmp_score, copy_num)

    # step 5- sort within ub to output_ub with sorted_num
    sort_within_ub(tik_instance, output_ub, ceil_div(copy_num, 16) * 16)
    if ceil_div(copy_num, 16) * 16 != sorted_num:
        dup_len = (sorted_num - ceil_div(copy_num, 16) * 16)
        offset = ceil_div(copy_num, 16) * 16 * 8
        tik_func_vector(tik_instance, output_ub[offset:], 0.0, dup_len*8)


def tik_func_sort_with_ub(tik_instance, src_ub_list, dst_ub_list, sorted_num, whether_save_proposal=None):
    """sort two sorted proposals list:
        get the top sorted_num proposals from src_ub_list
        and copy top sorted_num to output_ub
        and if need, copy low sorted_num to l1

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    src_ub_list : list
        the proposal list, each list have been sorted
    dst_ub_list : list
        result ub, copy top sorted_num to output_ub
    sorted_num : int
        the proposal num of proposal list
    whether_save_proposal: gm
        whether copy low sorted_num to l1

    Returns
    -------
    None
    """
    list_len = len(src_ub_list)
    with tik_instance.new_stmt_scope():
        # apply second top k proposal ub
        ub_dst_sort_with_ub = tik_instance.Tensor("float16", [list_len*sorted_num*8],
                                                  name="ub_dst_sort_with_ub", scope=tik.scope_ubuf)
        sort_with_ub(tik_instance, src_ub_list, ub_dst_sort_with_ub, sorted_num)
        loop_burst_len = (sorted_num * 8) // 16
        tik_instance.data_move(dst_ub_list[0], ub_dst_sort_with_ub,
                               0, 1, loop_burst_len, 0, 0)
        if whether_save_proposal is not None:
            tik_instance.data_move(whether_save_proposal, ub_dst_sort_with_ub[sorted_num*8:],
                                   0, 1, loop_burst_len, 0, 0)


def filter_score_compute(tik_instance, score_ub, score_valid_num_ub, scores_valid_mask, score_num, score_thresh):
    """filter_score_compute, is score is less score_thresh, change score = 0
    """
    with tik_instance.new_stmt_scope():
        tmp_ub_for_vmax = tik_instance.Tensor(score_ub.dtype, [16],
                                              name="tmp_ub_for_vmax", scope=tik.scope_ubuf)
        tmp_ub_for_vmin = tik_instance.Tensor(score_ub.dtype, [16],
                                              name="tmp_ub_for_vmin", scope=tik.scope_ubuf)
        score_ub_mask = tik_instance.Tensor(score_ub.dtype, score_ub.shape,
                                            name="score_ub_mask", scope=tik.scope_ubuf)
        tik_func_vadds(tik_instance, scores_valid_mask, score_ub, score_thresh*(-1.0), score_num)
        tik_func_vector(tik_instance, tmp_ub_for_vmax, 0.0, 16)
        tik_func_vector(tik_instance, tmp_ub_for_vmin, MIN_SCALAR_FP16, 16)
        tik_func_vcomple(tik_instance, "vmin", scores_valid_mask, scores_valid_mask, tmp_ub_for_vmin,
                         score_num, 1, 1, 0, 8, 8, 0)
        tik_func_vcomple(tik_instance, "vmax", scores_valid_mask, scores_valid_mask, tmp_ub_for_vmax,
                         score_num, 1, 1, 0, 8, 8, 0)
        tik_func_vmuls(tik_instance, scores_valid_mask, scores_valid_mask, TMP_SCALAR_FP16, score_num)
        tik_func_vmuls(tik_instance, scores_valid_mask, scores_valid_mask, TMP_SCALAR_FP16, score_num)
        tik_func_vcomple(tik_instance, "vmul", score_ub, scores_valid_mask, score_ub,
                         score_num, 1, 1, 1, 8, 8, 8)
        tik_func_vmuls(tik_instance, score_ub_mask, scores_valid_mask, 1, score_num)
        repeat_loop = score_num // 128
        repeat_tail = score_num % 128
        if repeat_loop > 1:
            tik_func_vcomple(tik_instance, "vadd", score_ub_mask, score_ub_mask[128:], score_ub_mask,
                             (repeat_loop - 1) * 128, 1, 1, 1, 0, 8, 0)
        if repeat_tail != 1 and repeat_loop > 0:
            tik_func_vcomple(tik_instance, "vadd", score_ub_mask, score_ub_mask[128*repeat_loop:], score_ub_mask,
                             repeat_tail, 1, 1, 1, 8, 8, 8)
        if repeat_loop > 0:
            tik_instance.vcadd(128, score_ub_mask, score_ub_mask, 1, 1, 1, 8)
        else:
            tik_instance.vcadd(repeat_tail, score_ub_mask, score_ub_mask, 1, 1, 1, 8)

        tik_instance.vconv(8, "round", score_valid_num_ub, score_ub_mask, 1, 1, 1, 8, 4)


def nms_for_single_class(batch_idx, class_idx, nms, core_idx):
    """main func to get nms for each class,
    and copy result to l1 to concat
    """
    # get tik instance
    tik_instance = nms.get_tik_instance()
    # get first top_k proposals to ub_max_topk
    nms.init_tik_ub_mem_for_topk()
    topk_out_ub = nms.ub_max_topk
    clip_window_value_list = nms.clip_window_value_list
    gm_offset = [batch_idx, class_idx, 0, 0]
    sorted_k = nms.proposal_topk_k

    # valid num info
    l1_valid = nms.l1_score_valid

    # get first top 4096 high score boxes and do nms
    if nms.single_loop:
        get_sorted_proposal_compute(tik_instance, topk_out_ub, nms.input_gm_list, gm_offset, nms.boxes_num,
                                    ceil_div(nms.boxes_num, 16)*16, clip_window_value_list,
                                    l1_valid, reduce_scalar=nms.down_scalar_list)
    else:
        get_sorted_proposal_compute(tik_instance, topk_out_ub, nms.input_gm_list, gm_offset,
                                    sorted_k, sorted_k, clip_window_value_list, l1_valid,
                                    reduce_scalar=nms.down_scalar_list)

        # do topk k proposal loop to get final top proposal_topk_k proposals to ub_max_topk
        with tik_instance.new_stmt_scope():
            # apply second top k proposal ub
            ub_tmp_topk = tik_instance.Tensor("float16", topk_out_ub.shape,
                                              name="ub_tmp_topk", scope=tik.scope_ubuf)
            if nms.topk_loop_time > 1:
                with tik_instance.for_range(1, nms.topk_loop_time) as _top_k_idx:
                    gm_offset = [batch_idx, class_idx, 0, _top_k_idx*nms.proposal_topk_k]
                    if nms.is_second_nms:
                        workspace_offset = (_top_k_idx - 1)*sorted_k*8 + core_idx*(nms.boxes_num*8)
                        workspace_for_save_proposal = nms.workspace_second_nms_gm[workspace_offset]
                    else:
                        workspace_for_save_proposal = None
                    # get tmp sorted proposal to ub_tmp_topk
                    get_sorted_proposal_compute(tik_instance, ub_tmp_topk, nms.input_gm_list,
                                                gm_offset, sorted_k, sorted_k, clip_window_value_list, l1_valid,
                                                reduce_scalar=nms.down_scalar_list)
                    # sorted two proposals to one proposal list output the top sorted_k
                    tik_func_sort_with_ub(tik_instance, [ub_tmp_topk, topk_out_ub],
                                          [topk_out_ub, ub_tmp_topk], sorted_k,
                                          workspace_for_save_proposal)

            if nms.topk_loop_tail != 0:
                gm_offset = [batch_idx, class_idx, 0, nms.topk_loop_time*nms.proposal_topk_k]
                if nms.is_second_nms:
                    workspace_offset = (nms.topk_loop_time - 1)*sorted_k*8 + core_idx*nms.boxes_num*8
                    workspace_for_save_proposal = nms.workspace_second_nms_gm[workspace_offset]
                else:
                    workspace_for_save_proposal = None
                # get tmp sorted proposal to ub_tmp_topk
                get_sorted_proposal_compute(tik_instance, ub_tmp_topk, nms.input_gm_list,
                                            gm_offset, nms.topk_loop_tail, sorted_k, clip_window_value_list, l1_valid,
                                            reduce_scalar=nms.down_scalar_list)
                # sorted two proposals to one proposal list output the top sorted_k
                tik_func_sort_with_ub(tik_instance, [ub_tmp_topk, topk_out_ub],
                                      [topk_out_ub, ub_tmp_topk], sorted_k,
                                      workspace_for_save_proposal)

    # do nms use topk output to get nms proposals per class
    # and move result to l1
    with tik_instance.new_stmt_scope():
        nms_var = nms.init_tik_ub_mem_for_nms()
        nmsed_result_ub = nms_var.get("selected_proposal_ub")
        nmsed_result_area = nms_var.get("selected_area_ub")
        nmsed_result_sup = nms_var.get("sup_vec_ub")
        # init all sup_vec to 1, mean: no select proposal
        tik_func_vector(tik_instance, nmsed_result_sup, 1, nms.max_selected_nms_num_in_ub)
        # init select nms proposal = 0
        l1_buffer = nms.l1_nms_result_zero
        loop_burst_len = (nms.max_selected_nms_num_in_ub*8) // 16
        tik_instance.data_move(nmsed_result_ub, l1_buffer,
                               0, 1, loop_burst_len, 0, 0)
        # init select nms area = 0
        loop_burst_len = nms.max_selected_nms_num_in_ub // 16
        tik_instance.data_move(nmsed_result_area, l1_buffer,
                               0, 1, loop_burst_len, 0, 0)
        with tik_instance.new_stmt_scope():
            do_nms_compute(tik_instance, nms_var, nms.iou_thresh)
        # copy one class nms result to l1
        l1_buffer = nms.l1_nms_result
        l1_offset = [class_idx, 0, 0]
        loop_burst_len = (nms.max_selected_nms_num_in_ub*8) // 16
        tik_instance.data_move(l1_buffer[l1_offset], nmsed_result_ub,
                               0, 1, loop_burst_len, 0, 0)
        if nms.is_second_nms:
            loop_burst_len = nms.max_selected_nms_num_in_ub // 16
            tik_instance.data_move(nms.l1_nms_area, nmsed_result_area,
                                   0, 1, loop_burst_len, 0, 0)
            tik_instance.data_move(nms.l1_nms_sup, nms_var.get("sup_vec_ub"),
                                   0, 1, loop_burst_len, 0, 0)

    # if the select nms output num of the first top 4096 highest score boxes is less the output need
    # and the impl_mode is high_precision
    # will do nms again from the tail boxes 4096 boxes by 4096 boxes
    tool_loop = nms.topk_loop_time if nms.topk_loop_tail == 0 else (nms.topk_loop_time + 1)
    if nms.is_second_nms and tool_loop >= 3:
        # if not to output num
        with tik_instance.for_range(1, tool_loop - 1) as _top_n_idx:
            top_n_num_tail = tool_loop - _top_n_idx - 1
            with tik_instance.if_scope(nms.selected_proposals_cnt < nms.max_total_size):
                # copy a sorted proposals to topk_out_ub
                loop_burst_len = ceil_div(sorted_k*8, 16)
                tik_instance.data_move(topk_out_ub, nms.workspace_second_nms_gm[core_idx*nms.boxes_num*8],
                                       0, 1, loop_burst_len, 0, 0)
                # apply second top k proposal ub
                with tik_instance.new_stmt_scope():
                    ub_tmp_topk = tik_instance.Tensor("float16", topk_out_ub.shape,
                                                      name="ub_tmp_topk", scope=tik.scope_ubuf)
                    with tik_instance.for_range(0, top_n_num_tail) as _top_n_tail_idx:
                        workspace_proposal_offset = sorted_k*8 + _top_n_tail_idx*sorted_k*8 + core_idx*nms.boxes_num*8
                        tik_instance.data_move(ub_tmp_topk, nms.workspace_second_nms_gm[workspace_proposal_offset],
                                               0, 1, loop_burst_len, 0, 0)
                        workspace_offset = _top_n_tail_idx*sorted_k*8 + core_idx*nms.boxes_num*8
                        workspace_for_save_proposal = nms.workspace_second_nms_gm[workspace_offset]
                        # sorted two proposals to one proposal list output the top sorted_k
                        tik_func_sort_with_ub(tik_instance, [ub_tmp_topk, topk_out_ub],
                                              [topk_out_ub, ub_tmp_topk], sorted_k,
                                              workspace_for_save_proposal)
                # do nms use topk output to get nms proposals per class
                # and move result to l1
                with tik_instance.new_stmt_scope():
                    nms_var = nms.init_tik_ub_mem_for_nms()
                    nmsed_result_ub = nms_var.get("selected_proposal_ub")
                    nmsed_result_area = nms_var.get("selected_area_ub")
                    nmsed_result_sup = nms_var.get("sup_vec_ub")

                    # copy l1 tmp data to ub
                    l1_buffer = nms.l1_nms_result
                    l1_offset = [class_idx, 0, 0]
                    loop_burst_len = (nms.max_selected_nms_num_in_ub*8) // 16
                    # copy the selected proposal/area/sup_ub from L1 to UB
                    tik_instance.data_move(nmsed_result_ub, l1_buffer[l1_offset],
                                           0, 1, loop_burst_len, 0, 0)
                    loop_burst_len = nms.max_selected_nms_num_in_ub // 16
                    tik_instance.data_move(nmsed_result_area, nms.l1_nms_area,
                                           0, 1, loop_burst_len, 0, 0)
                    tik_instance.data_move(nmsed_result_sup, nms.l1_nms_sup,
                                           0, 1, loop_burst_len, 0, 0)

                    with tik_instance.new_stmt_scope():
                        do_nms_compute(tik_instance, nms_var, nms.iou_thresh)
                    # copy one class nms result to l1
                    l1_buffer = nms.l1_nms_result
                    l1_offset = [class_idx, 0, 0]
                    loop_burst_len = (nms.max_selected_nms_num_in_ub*8) // 16
                    tik_instance.data_move(l1_buffer[l1_offset], nmsed_result_ub,
                                           0, 1, loop_burst_len, 0, 0)
                    loop_burst_len = nms.max_selected_nms_num_in_ub // 16
                    tik_instance.data_move(nms.l1_nms_area, nmsed_result_area,
                                           0, 1, loop_burst_len, 0, 0)
                    tik_instance.data_move(nms.l1_nms_sup, nmsed_result_sup,
                                           0, 1, loop_burst_len, 0, 0)


def get_class_tensor(tik_instance, class_ub, class_num, len_per_class, start_class=0.0):
    """get class tensor
    """
    tik_func_vector(tik_instance, class_ub, start_class, len_per_class)
    with tik_instance.for_range(1, class_num) as _class_idx:
        dst_offset = _class_idx * len_per_class
        # get ub_class_all[n] = ub_class_all[n-1] + 1
        src_offset = (_class_idx - 1) * len_per_class
        _repeat_time = len_per_class // 128
        _repeat_tail = len_per_class % 128
        if _repeat_time != 0:
            tik_instance.vadds(128, class_ub[dst_offset], class_ub[src_offset], 1.0,
                               _repeat_time, 1, 1, 8, 8)
            dst_offset = 128*_repeat_time + dst_offset
            src_offset = 128*_repeat_time + src_offset
        if _repeat_tail != 0:
            tik_instance.vadds(_repeat_tail, class_ub[dst_offset], class_ub[src_offset], 1.0,
                               1, 1, 1, 8, 8)


def copy_tail_data(tik_instance, gm_dst_info, ub_src_info, gm_workspace_info, copy_len):
    """copy_tail_data when output is not align, will use workspace to align force
    """
    gm_dst, gm_dst_offset = gm_dst_info
    ub_src, ub_src_offset = ub_src_info
    gm_workspace, gm_workspace_offset = gm_workspace_info
    data_type = ub_src.dtype
    if data_type in ("float32", "int32"):
        block_num = 8
    else:
        block_num = 16
    copy_nbust_len = copy_len // block_num
    copy_tail_offset = copy_len % block_num
    tik_instance.data_move(gm_dst[gm_dst_offset], ub_src[ub_src_offset], 0, 1, copy_nbust_len, 0, 0)
    tik_instance.data_move(gm_workspace[gm_workspace_offset], ub_src[ub_src_offset + (copy_nbust_len - 1)*block_num],
                           0, 1, 2, 0, 0)
    tik_instance.data_move(ub_src[ub_src_offset], gm_workspace[gm_workspace_offset + copy_tail_offset],
                           0, 1, 1, 0, 0)
    tik_instance.data_move(gm_dst[gm_dst_offset + copy_tail_offset + (copy_nbust_len - 1)*block_num],
                           ub_src[ub_src_offset], 0, 1, 1, 0, 0)


def batch_multi_class_nms_copy_out(tik_instance, nms, ub_result_boxes, ub_result_boxes_class,
                                   output_batch_offset, workspace_core_offset):
    """batch_multi_class_nms_copy_out
    """
    core_used = nms.aicore_num
    workspace_flag = False
    if (core_used > 1) and (nms.max_total_size % 16 != 0):
        workspace_flag = True

    workspace = nms.workspace_proposal_gm
    down_scalar = None
    if nms.down_flag:
        down_scalar = nms.down_scalar_list[1]
    loop_burst_len = ceil_div(nms.max_total_size, 16)
    apply_men_len = ceil_div(nms.max_total_size, 16)
    less_flag = False
    if nms.max_selected_nms_num_in_ub * nms.classes < nms.max_total_size:
        less_flag = True
        loop_burst_len = ceil_div(nms.max_selected_nms_num_in_ub * nms.classes, 16)
    score_thresh = nms.score_thresh
    _batch = output_batch_offset // nms.max_total_size
    ub_scores_valid_mask = tik_instance.Tensor("float16", [apply_men_len*16],
                                               name="ub_scores_valid_mask", scope=tik.scope_ubuf)
    # process scores
    with tik_instance.new_stmt_scope():
        # scores
        ub_out_scores = tik_instance.Tensor("float16", [apply_men_len*16],
                                            name="ub_out_scores", scope=tik.scope_ubuf)
        ub_out_scores_valid = tik_instance.Tensor("int32", [16], name="ub_out_scores_valid",
                                                  scope=tik.scope_ubuf)
        if less_flag:
            tik_func_vector(tik_instance, ub_out_scores, 0, apply_men_len*16)
        tik_func_vextract(tik_instance, ub_result_boxes_class, ub_out_scores, loop_burst_len, 3)
        filter_score_compute(tik_instance, ub_out_scores, ub_out_scores_valid, ub_scores_valid_mask,
                             nms.max_total_size, score_thresh)
        if not workspace_flag:
            tik_instance.data_move(nms.output_gm_list[1][output_batch_offset], ub_out_scores,
                                   0, 1, apply_men_len, 0, 0)
        else:
            copy_tail_data(tik_instance,
                           [nms.output_gm_list[1], output_batch_offset],
                           [ub_out_scores, 0],
                           [workspace, workspace_core_offset],
                           nms.max_total_size)

        tik_instance.data_move(nms.output_gm_list[3][_batch*8], ub_out_scores_valid,
                               0, 1, 1, 0, 0)
        # x1
        ub_out_box_x1 = tik_instance.Tensor("float16", [apply_men_len*16],
                                            name="ub_out_box_x1", scope=tik.scope_ubuf)
        tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_x1, loop_burst_len, 0)
        tik_func_vcomple(tik_instance, "vmul", ub_out_box_x1, ub_scores_valid_mask, ub_out_box_x1,
                         apply_men_len*16)
        if nms.down_flag:
            tik_func_vmuls(tik_instance, ub_out_box_x1, ub_out_box_x1, down_scalar, nms.max_total_size)
        # y1
        ub_out_box_y1 = tik_instance.Tensor("float16", [apply_men_len*16],
                                            name="ub_out_box_y1", scope=tik.scope_ubuf)
        tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_y1, loop_burst_len, 1)
        tik_func_vcomple(tik_instance, "vmul", ub_out_box_y1, ub_scores_valid_mask, ub_out_box_y1,
                         apply_men_len*16)
        # DOWN_FACTOR
        if nms.down_flag:
            tik_func_vmuls(tik_instance, ub_out_box_y1, ub_out_box_y1, down_scalar, nms.max_total_size)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset*4], ub_out_box_x1,
                               0, 1, apply_men_len, 0, 0)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset*4 + nms.max_total_size],
                               ub_out_box_y1, 0, 1, apply_men_len, 0, 0)

        # x2
        ub_out_box_x2 = tik_instance.Tensor("float16", [apply_men_len*16],
                                            name="ub_out_box_x2", scope=tik.scope_ubuf)
        tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_x2, loop_burst_len, 2)

        if not nms.is_need_rpn_offset:
            tik_func_vadds(tik_instance, ub_out_box_x2, ub_out_box_x2, 1.0, nms.max_total_size)

        if nms.down_flag:
            tik_func_vmuls(tik_instance, ub_out_box_x2, ub_out_box_x2, down_scalar, nms.max_total_size)
        tik_func_vcomple(tik_instance, "vmul", ub_out_box_x2, ub_scores_valid_mask, ub_out_box_x2,
                         apply_men_len*16)
        tik_instance.data_move(nms.output_gm_list[0][output_batch_offset*4 + nms.max_total_size*2],
                               ub_out_box_x2, 0, 1, apply_men_len, 0, 0)

        # y2
        ub_out_box_y2 = tik_instance.Tensor("float16", [apply_men_len*16],
                                            name="ub_out_box_y2", scope=tik.scope_ubuf)
        tik_func_vextract(tik_instance, ub_result_boxes, ub_out_box_y2, loop_burst_len, 3)

        if not nms.is_need_rpn_offset:
            tik_func_vadds(tik_instance, ub_out_box_y2, ub_out_box_y2, 1.0, nms.max_total_size)

        if nms.down_flag:
            tik_func_vmuls(tik_instance, ub_out_box_y2, ub_out_box_y2, down_scalar, nms.max_total_size)
        tik_func_vcomple(tik_instance, "vmul", ub_out_box_y2, ub_scores_valid_mask, ub_out_box_y2,
                         apply_men_len*16)
        if not workspace_flag:
            tik_instance.data_move(nms.output_gm_list[0][output_batch_offset*4 + nms.max_total_size*3],
                                   ub_out_box_y2, 0, 1, apply_men_len, 0, 0)
        else:
            copy_tail_data(tik_instance,
                           [nms.output_gm_list[0], output_batch_offset*4 + nms.max_total_size*3],
                           [ub_out_box_y2, 0],
                           [workspace, workspace_core_offset],
                           nms.max_total_size)
        # class
        ub_out_class = tik_instance.Tensor("float16", [apply_men_len*16],
                                           name="ub_out_class", scope=tik.scope_ubuf)
        tik_func_vextract(tik_instance, ub_result_boxes_class, ub_out_class, loop_burst_len, 0)
        if not workspace_flag:
            tik_instance.data_move(nms.output_gm_list[2][output_batch_offset], ub_out_class,
                                   0, 1, apply_men_len, 0, 0)
        else:
            copy_tail_data(tik_instance,
                           [nms.output_gm_list[2], output_batch_offset],
                           [ub_out_class, 0],
                           [workspace, workspace_core_offset],
                           nms.max_total_size)


def batch_multi_class_nms_output(tik_instance, core_idx, _batch_idx, nms):
    """do batch_multi_class_nms_output

    Parameters:
    ----------
    tik_instance : tik_instance.
    _batch_idx : int.
        the process batch
    nms : class.
        all par for nms

    Returns
    -------
    None
    """
    result_total = total_num(nms.l1_nms_result.shape)
    class_num = nms.classes
    # get score batch offset
    output_batch_offset = _batch_idx * nms.max_total_size
    workspace = nms.workspace_proposal_gm
    workspace_offset = core_idx*nms.workspace_proposal_gm.shape[-1]
    if nms.classes * nms.max_selected_nms_num_in_ub < nms.proposal_topk_k:
        # when all output is less nms.proposal_topk_k
        # only use topk with ub for output proposal
        with tik_instance.new_stmt_scope():
            ub_result_boxes = tik_instance.Tensor("float16", [result_total // 8, 8],
                                                  name="ub_result_boxes", scope=tik.scope_ubuf)
            ub_result_boxes_class = tik_instance.Tensor("float16", [result_total // 8, 8],
                                                        name="ub_result_boxes_class", scope=tik.scope_ubuf)
            l1_buffer = nms.l1_nms_result
            l1_offset = [0, 0, 0]
            loop_burst_len = result_total // 16
            tik_instance.data_move(ub_result_boxes, l1_buffer[l1_offset],
                                   0, 1, loop_burst_len, 0, 0)
            tik_instance.data_move(ub_result_boxes_class, l1_buffer[l1_offset],
                                   0, 1, loop_burst_len, 0, 0)
            with tik_instance.new_stmt_scope():
                ub_class_all = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub*nms.classes],
                                                   name="ub_class_all", scope=tik.scope_ubuf)
                get_class_tensor(tik_instance, ub_class_all, class_num, nms.max_selected_nms_num_in_ub)

                trans_repeat = ceil_div(nms.max_selected_nms_num_in_ub*nms.classes, 16)
                tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_all, trans_repeat, 1)
                tik_instance.data_move(workspace[workspace_offset], ub_result_boxes_class,
                                       0, 1, loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, workspace[workspace_offset + 1],
                                       0, 1, loop_burst_len, 0, 0)
                tik_func_vextract(tik_instance, ub_result_boxes_class, ub_class_all, trans_repeat, 3)
                tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_all, trans_repeat, 4)

            if nms.classes != 1:
                sort_within_ub(tik_instance, ub_result_boxes_class, result_total // 8)
                sort_within_ub(tik_instance, ub_result_boxes, result_total // 8)

            with tik_instance.new_stmt_scope():
                batch_multi_class_nms_copy_out(tik_instance, nms, ub_result_boxes,
                                               ub_result_boxes_class, output_batch_offset, workspace_offset)
    else:
        l1_buffer = nms.l1_nms_result
        copy_classes_num = nms.proposal_topk_k // nms.max_selected_nms_num_in_ub // 2
        copy_loop = nms.classes // copy_classes_num
        copy_tail = nms.classes % copy_classes_num
        tmp_output_proposal_num = ceil_div(nms.max_total_size, 16) * 16
        ub_out_result = tik_instance.Tensor("float16", [tmp_output_proposal_num, 8],
                                            name="ub_out_result", scope=tik.scope_ubuf)
        ub_out_result_class = tik_instance.Tensor("float16", [tmp_output_proposal_num, 8],
                                                  name="ub_out_result_class", scope=tik.scope_ubuf)
        tik_func_vector(tik_instance, ub_out_result, 0.0, tmp_output_proposal_num * 8)
        tik_func_vector(tik_instance, ub_out_result_class, 0.0, tmp_output_proposal_num * 8)
        with tik_instance.new_stmt_scope():
            ub_result_boxes = tik_instance.Tensor("float16", [copy_classes_num*nms.max_selected_nms_num_in_ub, 8],
                                                  name="ub_result_boxes", scope=tik.scope_ubuf)
            ub_result_boxes_class = tik_instance.Tensor("float16", [copy_classes_num*nms.max_selected_nms_num_in_ub,
                                                                    8],
                                                        name="ub_result_boxes_class", scope=tik.scope_ubuf)
            ub_class_all = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub*copy_classes_num],
                                               name="ub_class_all", scope=tik.scope_ubuf)
            get_class_tensor(tik_instance, ub_class_all, copy_classes_num,
                             nms.max_selected_nms_num_in_ub, copy_classes_num*-1)

            def _do_copy_and_vconcat_class(_l1_offset, _loop_burst_len, ):
                tik_instance.data_move(ub_result_boxes, l1_buffer[_l1_offset],
                                       0, 1, _loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, l1_buffer[_l1_offset],
                                       0, 1, _loop_burst_len, 0, 0)
                # get copy_classes_num sort
                tik_func_vadds(tik_instance, ub_class_all, ub_class_all, copy_classes_num*1.0,
                               nms.max_selected_nms_num_in_ub*copy_classes_num)
                _trans_repeat = ceil_div(nms.max_selected_nms_num_in_ub*copy_classes_num, 16)
                tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_all, _trans_repeat, 1)
                tik_instance.data_move(workspace[workspace_offset], ub_result_boxes_class,
                                       0, 1, _loop_burst_len, 0, 0)
                tik_instance.data_move(ub_result_boxes_class, workspace[workspace_offset + 1],
                                       0, 1, _loop_burst_len, 0, 0)
                with tik_instance.new_stmt_scope():
                    ub_class_tmp = tik_instance.Tensor("float16", [nms.max_selected_nms_num_in_ub*copy_classes_num],
                                                       name="ub_class_tmp", scope=tik.scope_ubuf)
                    tik_func_vextract(tik_instance, ub_result_boxes_class, ub_class_tmp, _trans_repeat, 3)
                    tik_func_vconcat(tik_instance, ub_result_boxes_class, ub_class_tmp, _trans_repeat, 4)

            with tik_instance.for_range(0, copy_loop) as _class_idx:
                l1_offset = [_class_idx*copy_classes_num, 0, 0]
                loop_burst_len = copy_classes_num*nms.max_selected_nms_num_in_ub*8 // 16
                _do_copy_and_vconcat_class(l1_offset, loop_burst_len)
                sort_within_ub(tik_instance, ub_result_boxes, copy_classes_num*nms.max_selected_nms_num_in_ub)
                sort_within_ub(tik_instance, ub_result_boxes_class, copy_classes_num*nms.max_selected_nms_num_in_ub)
                tik_func_sort_with_ub(tik_instance, [ub_out_result, ub_result_boxes],
                                      [ub_out_result, ub_result_boxes], tmp_output_proposal_num)
                tik_func_sort_with_ub(tik_instance, [ub_out_result_class, ub_result_boxes_class],
                                      [ub_out_result_class, ub_result_boxes_class], tmp_output_proposal_num)

            if copy_tail != 0:
                l1_offset = [copy_loop*copy_classes_num, 0, 0]
                loop_burst_len = copy_tail*nms.max_selected_nms_num_in_ub*8 // 16
                _do_copy_and_vconcat_class(l1_offset, loop_burst_len)
                sort_within_ub(tik_instance, ub_result_boxes, copy_tail*nms.max_selected_nms_num_in_ub)
                sort_within_ub(tik_instance, ub_result_boxes_class, copy_tail*nms.max_selected_nms_num_in_ub)
                if copy_tail*nms.max_selected_nms_num_in_ub < tmp_output_proposal_num:
                    dup_len = tmp_output_proposal_num - copy_tail*nms.max_selected_nms_num_in_ub
                    dup_offset = copy_tail*nms.max_selected_nms_num_in_ub
                    tik_func_vector(tik_instance, ub_result_boxes[dup_offset:], 0.0, dup_len * 8)
                    tik_func_vector(tik_instance, ub_result_boxes_class[dup_offset:], 0.0, dup_len * 8)
                tik_func_sort_with_ub(tik_instance, [ub_out_result, ub_result_boxes],
                                      [ub_out_result, ub_result_boxes], tmp_output_proposal_num)
                tik_func_sort_with_ub(tik_instance, [ub_out_result_class, ub_result_boxes_class],
                                      [ub_out_result_class, ub_result_boxes_class], tmp_output_proposal_num)
        with tik_instance.new_stmt_scope():
            batch_multi_class_nms_copy_out(tik_instance, nms, ub_out_result, ub_out_result_class,
                                           output_batch_offset, workspace_offset)


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, OPTION_INPUT, OPTION_INPUT,
                 REQUIRED_OUTPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT,
                 REQUIRED_ATTR_FLOAT, REQUIRED_ATTR_FLOAT, REQUIRED_ATTR_INT, REQUIRED_ATTR_INT,
                 REQUIRED_ATTR_BOOL, REQUIRED_ATTR_BOOL, KERNEL_NAME)
def batch_multi_class_non_max_suppression(boxes, scores, clip_window, num_valid_boxes,
                                          nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num,
                                          score_threshold, iou_threshold, max_size_per_class,
                                          max_total_size, change_coordinate_frame, transpose_box,
                                          kernel_name="batch_multi_class_non_max_suppression",
                                          impl_mode="high_performance"):
    """
    do non_max_suppression for multi batch and multi class
    step 1- clip boxes use clip_window, when the area of boxes after clip, change the score = 0
    step 2- filter score, when the score is less score_threshold, change the score = 0
    step 3- filter valid num use num_valid_boxes
    step 4- trans the box and score to proposal
    step 5- sort the input proposals and get 4094 proposals
    step 6- do nms for each class in each batch use top 4094 proposals
    step 7- concat all class nms result in each batch
    step 8- sort the proposals and output the max_total_size box/class/score

    Parameters:
    ----------
    boxes : dict.
        shape, dtype of boxes, a 4D Tensor of type float16 with shape (batch, num_anchors, num_classes, 4).
        "batch" indicates the batch size of image,
        and "num_anchors" indicates num of boxes, and "num_classes" indicates classes of detect.
        and the value "4" refers to "x0", "x1", "y0", and "y1".
    scores : dict.
        shape, dtype of scores
        a 3D Tensor of type float16 with shape (batch, num_anchors, num_classes).
    clip_window : dict.
        shape, dtype of scores
        a 2D Tensor of type float16 with shape (batch, 4).
        4" refers to "anchor_x0", "anchor_x1", "anchor_y0", and "anchor_y1".
    num_valid_boxes : dict.
        A 1D Tensor of type int32 with shape (batch,).
        specifying valid boxes number for each batch
    nmsed_boxes : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size, 4).
        specifying the output nms boxes per batch
    nmsed_scores : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size).
        specifying the output nms score per batch
    nmsed_classes : dict.
        A 3D Tensor of type float16 with shape (batch, max_total_size).
        specifying the output nms class per batch
    nmsed_num : dict.
        A 1D Tensor of type int32 with shape (batch,),
        specifying the valid num of nmsed_boxes
    score_threshold : float.
        A required attribute of type float32, specifying the score filter iou iou_threshold.
    iou_threshold : float.
        A required attribute of type float32, specifying the nms iou iou_threshold
    max_size_per_class : int.
        A required attribute of type int, specifying the nms output num per class.
    max_total_size : int.
        A required attribute of type int, specifying the the nms output num per batch.
    change_coordinate_frame : bool.
        A required attribute of type bool, whether to normalize coordinates after clipping.
    transpose_box : bool.
        A required attribute of type bool, whether inserted transpose before this op
    kernel_name : str.
        cce kernel name, default value is "batch_multi_class_non_max_suppression"
    impl_mode: str.
        high_precision or high_performance for inference, default value is "high_performance".
        no need to add into ops_info file.

    Returns
    -------
    tik_instance
    """
    nms = BatchMultiClassNonMaxSuppression(boxes, scores, num_valid_boxes, clip_window,
                                           score_threshold, iou_threshold, max_size_per_class,
                                           max_total_size, change_coordinate_frame, impl_mode)
    # init ub
    core_used, batch_per_core, batch_last_core = nms.get_core_schedule()
    class_num = nms.classes
    nms.init_tik_mem()
    tik_instance = nms.get_tik_instance()

    def _run_one_core(_real_batch_idx, _real_core_idx):
        # get clip_to_window input data to scalar reg clip_window_value_list
        # and if need nms.change_coordinate_frame, will get scale scalar at the same time
        if nms.need_clip_window:
            read_window_compute(tik_instance, nms.input_gm_list[2], [_real_batch_idx, 0],
                                nms.clip_window_value_list, nms.down_scalar_list,
                                nms.change_coordinate_frame)

        if nms.need_valid_num:
            read_valid_num_compute(tik_instance, nms.input_gm_list[-1], [_real_batch_idx], nms.valid_num_value)
            gen_valid_num_compute(tik_instance, nms.l1_score_valid, nms.boxes_num, nms.valid_num_value)

        with tik_instance.for_range(0, class_num) as _class_idx:
            # for each class, init selected_proposals_cnt = 0
            nms.selected_proposals_cnt.set_as(0)
            with tik_instance.new_stmt_scope():
                nms_for_single_class(_real_batch_idx, _class_idx, nms, _real_core_idx)

        # process all class output result is in l1_nms_result, will process output
        # step 1 sort all select proposal with boxes
        # step 2 sort all select proposal with classes score
        with tik_instance.new_stmt_scope():
            batch_multi_class_nms_output(tik_instance, _real_core_idx, _real_batch_idx, nms)

    # do nms with multi cores
    with tik_instance.for_range(0, core_used, block_num=core_used) as _core_idx:
        if batch_per_core == batch_last_core or core_used == 1:
            with tik_instance.for_range(0, batch_per_core) as _batch_idx:
                real_batch_idx = _core_idx*batch_per_core + _batch_idx
                _run_one_core(real_batch_idx, _core_idx)
        else:
            with tik_instance.if_scope(_core_idx < core_used - 1):
                with tik_instance.for_range(0, batch_per_core) as _batch_idx:
                    real_batch_idx = _core_idx*batch_per_core + _batch_idx
                    _run_one_core(real_batch_idx, _core_idx)
            with tik_instance.else_scope():
                with tik_instance.for_range(0, batch_last_core) as _batch_idx:
                    real_batch_idx = _core_idx*batch_per_core + _batch_idx
                    _run_one_core(real_batch_idx, _core_idx)

    return nms.build_tik_instance(kernel_name)

