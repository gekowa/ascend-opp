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
proposal_d
"""
# pylint: disable=R0902
# pylint: disable=R0903
# pylint: disable=R0913
# pylint: disable=R0914
# pylint: disable=W0613
# pylint: disable=too-many-branches
from topi.cce import util
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *
from impl import decoded_bbox
from impl import nms
from impl import topk



def get_dtype_size(input_dtype):
    """
    :param input_dtype:
    :return:
    """
    if input_dtype == "float16":
        size = 2
    else:
        size = 4

    return size


def get_ub_size():
    """
    :return:
    """
    ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    return ub_size


def filte_device_core(batch):
    """
    :param batch:
    :return:
    """
    device_core_num = \
        tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    if batch >= device_core_num:
        batch_factor = batch//device_core_num
        batch_factor_tail = batch - batch_factor*device_core_num
    else:
        batch_factor = batch
        batch_factor_tail = 0
        device_core_num = 1

    return device_core_num, batch_factor, batch_factor_tail


def call_topk_sort(tik_instance, input_data, output):
    """
    :param tik_instance:
    :param input_data:
    :param output:
    :return:
    """

    score_threshold = 0
    k = input_data[0]
    regions_orig = input_data[1]
    mem_swap = input_data[2]
    proposal_num = input_data[3]

    batch_id = output[0]
    regions_sorted = output[1]
    proposal_actual_num = output[2]

    topk_input = {
        "proposal_num": proposal_num,
        "k": k,
        "score_threshold": score_threshold,
        "regions_orig": regions_orig,
        "mem_swap": mem_swap,
    }

    topk_out = {
        "batch_id": batch_id,
        "regions_sorted": regions_sorted,
        "proposal_actual_num": proposal_actual_num,
    }

    topk.tik_topk(tik_instance, topk_input, topk_out)


class InitProposalProcess:
    """
    Init Proposal Process
    """

    def __init__(self, input_data):
        """
        :param input_data:
        """
        self.tik_instance = input_data[0]
        feature_dic = input_data[1]
        self.im_info_dic = input_data[2]
        self.min_box_size = input_data[3]
        self.pre_nms_topn = input_data[4]
        self.post_nms_topn = input_data[5]
        self.nms_threshold = input_data[6]
        self.output_actual_rois_num = input_data[7]

        self.input_shape = feature_dic.get('shape')
        input_dtype = feature_dic.get('dtype')
        self.input_dtype = input_dtype

        channel = self.input_shape[1]
        height = self.input_shape[2]
        width = self.input_shape[3]
        num_anchor = channel//4
        num = (num_anchor*height*width + 127) // 128
        self.num = num


class ProposalProcess(InitProposalProcess):
    """
    Proposal Process
    """

    def __init__(self, input_data):
        """
        :param input_data:
        """
        super(ProposalProcess, self).__init__(input_data)
        feature_dic = input_data[1]
        input_dtype = feature_dic.get('dtype')
        input_shape = feature_dic.get('shape')
        batch, channel, height, width = input_shape
        num_anchor = channel//4
        num = (num_anchor*height*width + 127) // 128

        self.cls_prob = self.tik_instance.Tensor(input_dtype,
                                                 (batch, channel//2,
                                                  height, width),
                                                 name="cls_prob",
                                                 scope=tik.scope_gm)
        self.bbox_delta = self.tik_instance.Tensor(input_dtype,
                                                   self.input_shape,
                                                   name="bbox_delta",
                                                   scope=tik.scope_gm)
        self.rpn_bbox = self.tik_instance.Tensor(input_dtype,
                                                 self.input_shape,
                                                 name="rpn_bbox",
                                                 scope=tik.scope_gm)

        self.im_info = self.tik_instance.Tensor(input_dtype,
                                                self.im_info_dic.get('shape'),
                                                name="im_info",
                                                scope=tik.scope_gm)

        size = get_dtype_size(self.input_dtype)
        burst = ((num*128 - num_anchor*height*width)*8*size + 31)//32
        tail = burst*32//(8*size) - (num*128 - num_anchor*height*width)
        self.output_region_proposal = \
            self.tik_instance.Tensor(input_dtype,
                                     (batch, num*128+tail, 8),
                                     name="output_region_proposal",
                                     is_workspace=True,
                                     scope=tik.scope_gm)

        self.mem_swap = self.tik_instance.Tensor(input_dtype,
                                                 (batch, num*128+tail, 8),
                                                 name="mem_swap",
                                                 is_workspace=True,
                                                 scope=tik.scope_gm)

        self.topk_output_proposal = \
            self.tik_instance.Tensor(input_dtype,
                                     (batch,
                                      ((self.pre_nms_topn+15)//16)*16 + 4, 8),
                                     name="topk_output_proposal",
                                     is_workspace=True,
                                     scope=tik.scope_gm)

        self.temp_proposal_out = \
            self.tik_instance.Tensor(input_dtype,
                                     (batch, ((self.post_nms_topn + 15) // 16)*16, 8),
                                     name="temp_proposal_out",
                                     is_workspace=True,
                                     scope=tik.scope_gm)

        self.rois = self.tik_instance.Tensor(input_dtype,
                                             (batch, 5, ((self.post_nms_topn + 15) // 16)*16),
                                             name="rois",
                                             scope=tik.scope_gm)
        if self.output_actual_rois_num == 1:
            self.actual_rois_num = self.tik_instance.Tensor("int32", (batch, 8),
                                                            name="actual_rois_num",
                                                            scope=tik.scope_gm)
        else:
            self.actual_rois_num = self.tik_instance.Tensor("int32", (batch, 8),
                                                            name="actual_rois_num",
                                                            is_workspace=True,
                                                            scope=tik.scope_gm)

    def init_tail_zero(self, batch_id, size):
        """
        :param batch_id:
        :param size:
        :return:
        """
        if self.input_dtype == "float16":
            ratio = 1
        else:
            ratio = 2

        channel = self.input_shape[1]
        height = self.input_shape[2]
        width = self.input_shape[3]

        num_anchor = channel//4
        num = (num_anchor*height*width + 127) // 128

        tik_instance = self.tik_instance

        if num*128 > num_anchor*height*width:
            with tik_instance.if_scope(True):
                burst = ((num*128 - num_anchor*height*width)*8*size + 31)//32

                tmp_ub = tik_instance.Tensor(self.input_dtype, (128, 8),
                                             name="tmp_ub",
                                             scope=tik.scope_ubuf)
                tik_instance.vector_dup(128//ratio, tmp_ub, 0, 8*ratio, 1, 8)
                tik_instance.data_move(
                    self.output_region_proposal[batch_id,
                                                num_anchor*height*width, 0],
                    tmp_ub, 0, 1, burst, 0, 0)

    def cce_proposal(self, kernel_name="proposal"):
        """
        :param kernel_name:
        :return:
        """

        device_core_num, batch_factor, batch_factor_tail = \
            filte_device_core(self.input_shape[0])

        size = get_dtype_size(self.input_dtype)

        with self.tik_instance.for_range(
                0, device_core_num, block_num=device_core_num) as block_id:

            ub_size = get_ub_size()
            one_core_process_object = \
                decoded_bbox.OneCoreProcess((self.tik_instance,
                                             self.min_box_size,
                                             self.input_dtype, size,
                                             self.input_shape,
                                             device_core_num, batch_factor,
                                             ub_size))

            with self.tik_instance.for_range(0, batch_factor) as batch_index:
                batch_id = block_id*batch_factor + batch_index

                one_core_process_object.one_core_process_decode_bbox(
                    batch_id, self.cls_prob, self.bbox_delta, self.rpn_bbox, self.im_info,
                    self.output_region_proposal)

                self.init_tail_zero(batch_id, size)

                topk_output_actual_proposal_num = \
                    self.tik_instance.Scalar(dtype="int32")

                call_topk_sort(self.tik_instance,
                               (self.pre_nms_topn, self.output_region_proposal,
                                self.mem_swap, self.num*128),
                               (batch_id, self.topk_output_proposal,
                                topk_output_actual_proposal_num))

                input_offset = batch_id*(((self.pre_nms_topn+15)//16)*16 + 4)*8
                nms.cce_nms((self.input_dtype, ub_size,
                             self.nms_threshold, batch_id,
                             self.pre_nms_topn, self.post_nms_topn,
                             input_offset, self.im_info,
                             self.tik_instance),
                            self.temp_proposal_out,
                            self.topk_output_proposal,
                            topk_output_actual_proposal_num,
                            self.actual_rois_num, self.rois)

            with self.tik_instance.if_scope(block_id < batch_factor_tail):
                batch_id = batch_factor*device_core_num + block_id

                one_core_process_object.one_core_process_decode_bbox(
                    batch_id, self.cls_prob, self.bbox_delta, self.rpn_bbox, self.im_info,
                    self.output_region_proposal)

                self.init_tail_zero(batch_id, size)

                topk_output_actual_proposal_num = \
                    self.tik_instance.Scalar(dtype="int32")

                call_topk_sort(self.tik_instance,
                               (self.pre_nms_topn, self.output_region_proposal,
                                self.mem_swap, self.num*128),
                               (batch_id, self.topk_output_proposal,
                                topk_output_actual_proposal_num))

                input_offset = batch_id*(((self.pre_nms_topn+15)//16)*16 + 4)*8
                nms.cce_nms((self.input_dtype, ub_size,
                             self.nms_threshold, batch_id,
                             self.pre_nms_topn, self.post_nms_topn,
                             input_offset, self.im_info,
                             self.tik_instance),
                            self.temp_proposal_out,
                            self.topk_output_proposal,
                            topk_output_actual_proposal_num,
                            self.actual_rois_num, self.rois)
        if self.output_actual_rois_num == 1:
            self.tik_instance.BuildCCE(
                kernel_name,
                inputs=[self.cls_prob, self.bbox_delta, self.im_info, self.rpn_bbox],
                outputs=[self.rois, self.actual_rois_num])
        else:
            self.tik_instance.BuildCCE(
                kernel_name,
                inputs=[self.cls_prob, self.bbox_delta, self.im_info, self.rpn_bbox],
                outputs=[self.rois])

        return self.tik_instance


def check_datatype(tik_name, dtype):
    """
    :param tik_name:
    :param dtype:
    :return:
    """
    if not tbe_platform.cce_conf.api_check_support("tik.vrelu", "float32"):
        check_dtype(dtype.lower(), ["float16"], param_name="rpn_bbox_dic")
    else:
        check_dtype(dtype.lower(), ["float16", "float32"], param_name="rpn_bbox_dic")


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, OPTION_OUTPUT, OPTION_ATTR_FLOAT, OPTION_ATTR_FLOAT,
                 OPTION_ATTR_FLOAT, OPTION_ATTR_LIST_FLOAT, OPTION_ATTR_LIST_FLOAT,
                 OPTION_ATTR_INT, OPTION_ATTR_INT, OPTION_ATTR_FLOAT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def proposal_d(cls_prob_dic, bbox_delta_dic, im_info_dic, rpn_bbox_dic,
               rois_dic, actual_rois_num_dic,
               feat_stride, base_size, min_size,
               ratio, scale, pre_nms_topn,
               post_nms_topn, iou_threshold, output_actual_rois_num, kernel_name="cce_proposal"):
    """
    :param feature_dic:
    :param im_info:
    :param min_box_size:
    :param pre_nms_topn:
    :param post_nms_topn:
    :param iou_threshold:
    :param kernel_name:
    :return:
    """
    input_dtype = rpn_bbox_dic.get('dtype')
    input_shape = rpn_bbox_dic.get('shape')
    channel = input_shape[1]

    feature_dic = {"shape": input_shape, "dtype": input_dtype}

    tik_instance = tik.Tik(tik.Dprofile())
    tik_name = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
    check_datatype(tik_name, input_dtype)

    if min_size <= 0:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_002
        error_info['op_name'] = 'proposal_d'
        error_info['param_name'] = 'min_size'
        error_info['min_value'] = '0'
        error_info['max_value'] = 'inf'
        error_info['real_value'] = min_size
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s] should be in the range"
                           " of [%s, %s], but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['min_value'], error_info['max_value'],
                            error_info['real_value']))


    min_box_size = [min_size, min_size]

    for ratio_value in ratio:
        if ratio_value <= 0:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_002
            error_info['op_name'] = 'proposal_d'
            error_info['param_name'] = 'ratio_value'
            error_info['min_value'] = '0'
            error_info['max_value'] = 'inf'
            error_info['real_value'] = ratio_value
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s] should be in the range "
                               "of [%s, %s], but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['min_value'], error_info['max_value'],
                                error_info['real_value']))

    for scale_value in scale:
        if scale_value <= 0:
            error_info = {}
            error_info['errCode'] = OP_ERROR_CODE_002
            error_info['op_name'] = 'proposal_d'
            error_info['param_name'] = 'scale_value'
            error_info['min_value'] = '0'
            error_info['max_value'] = 'inf'
            error_info['real_value'] = scale_value
            raise RuntimeError(error_info,
                               "In op[%s], the parameter[%s] should be in the range"
                               " of [%s, %s], but actually is [%s]." %
                               (error_info['op_name'], error_info['param_name'],
                                error_info['min_value'], error_info['max_value'],
                                error_info['real_value']))

    if feat_stride <= 0 or base_size <= 0 or \
            pre_nms_topn <= 0 or post_nms_topn <= 0:
        raise RuntimeError("feat_stride, base_size, pre_nms_topn "
                           "and post_nms_topn must be greater than 0")

    if pre_nms_topn > 6000 or post_nms_topn > 6000:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_002
        error_info['op_name'] = 'proposal_d'
        error_info['param_name'] = 'pre_nms_topn or post_nms_topn'
        error_info['min_value'] = '0'
        error_info['max_value'] = '6000'
        error_info['real_value'] = ','.join((str(pre_nms_topn), str(post_nms_topn)))
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s] should be in the range"
                           " of [%s, %s], but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['min_value'], error_info['max_value'],
                            error_info['real_value']))

    if tik_name in ("Hi3796CV300ES", "Hi3796CV300CS") and \
            (pre_nms_topn > 3000 or post_nms_topn > 3000):
        raise RuntimeError("pre_nms_topn and post_nms_topn "
                           "must be <=3000 on hisi!")

    if channel % 4 != 0:
        raise RuntimeError("the channel must be multiples of 16!")

    if iou_threshold <= 0 or iou_threshold >= 1:
        error_info = {}
        error_info['errCode'] = OP_ERROR_CODE_002
        error_info['op_name'] = 'proposal_d'
        error_info['param_name'] = 'iou_threshold'
        error_info['min_value'] = '0'
        error_info['max_value'] = '1'
        error_info['real_value'] = str(iou_threshold)
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s] should be in the range "
                           "of [%s, %s], but actually is [%s]." %
                           (error_info['op_name'], error_info['param_name'],
                            error_info['min_value'], error_info['max_value'],
                            error_info['real_value']))

    proposal_result = ProposalProcess((tik_instance, feature_dic, im_info_dic,
                                       min_box_size,
                                       pre_nms_topn,
                                       post_nms_topn, iou_threshold, output_actual_rois_num))

    return proposal_result.cce_proposal(kernel_name)
