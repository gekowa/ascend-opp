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
yolo_v2_cls_prob
"""
# pylint: disable=import-error,too-many-locals,too-many-statements
from te import tik
from topi.cce import util
from te.utils import op_utils
from impl.yolo_v2_correct_box import CorrectBoxComputer
from impl.constant_util import BLOCK_SIZE
from impl.constant_util import VECTOR_BYTE_SIZE
from impl.constant_util import STRIDE_ONE
from impl.constant_util import CLOUD
from impl.constant_util import HISI_ES
from impl.constant_util import MINI
import impl.constant_util as constant
from impl import common_util as common
from te import platform as tbe_platform

# value zero
VALUE_ZERO = 0

# value two
VALUE_TWO = 2

# value three
VALUE_THREE = 3

# neg two
NEG_TWO = -2

# neg one
NEG_ONE = -1

# value half
VALUE_HALF = 0.5

# repeat time one
REPEAT_ONE = 1

# value one
VALUE_ONE = 1

# param for nms compute
PRE_NMS_TOPN = 1024


def check_param_range(param_name, min_value, max_value, real_value, op_name='yolo_v2_detection_output_d'):
    
    error_info = {}
    error_info['errCode'] = 'E80002'
    error_info['opname'] = op_name
    error_info['param_name'] = param_name
    error_info['min_value'] = str(min_value)
    error_info['max_value'] = str(max_value)
    error_info['real_value'] = str(real_value)
    raise RuntimeError(error_info, "In op[%s], the parameter[%s] should be in the range of [%s, %s], but actually is [%s]."
                       % (error_info['opname'], error_info['param_name'], error_info['min_value'],
                          error_info['max_value'], error_info['real_value']))


class ClsProbComputer(CorrectBoxComputer):
    """
     the class for cls prob compute
    """
    def __init__(self, input_dict):
        super(ClsProbComputer, self).__init__(input_dict)

        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ( \
            "Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS"):
            adj_hw = self.get_adj_hw(self.boxes * self.height * self.width)
            self.obj_prob_v200 = self.instance.Tensor(self.dtype,
                                                      (self.batch, adj_hw),
                                                      name="obj_prob_v200",
                                                      is_workspace=True,
                                                      scope=tik.scope_gm)

    def init_param(self, batch, param):
        """
          init some parameters

          Parameters
          ----------
          batch: the sequence of the photo
          param: param is an empty dict

          Returns
          -------
          None
          """
        index_ub = None
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
            "Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS"):
            index_ub = self.instance.Tensor("int32", (PRE_NMS_TOPN,),
                                            name="index_ub",
                                            scope=tik.scope_ubuf)
        index_offset = self.instance.Scalar("int32")
        index_offset.set_as(0)
        count = self.instance.Scalar("int32")
        count.set_as(0)
        param["index_offset"] = index_offset
        param["count"] = count
        param["index_ub"] = index_ub
        obj_gm_offset = self.instance.Scalar("int32")
        obj_gm_offset.set_as(batch * self.obj_num)
        param['obj_gm_offset'] = obj_gm_offset

    def init_small_clsprob_param(self, param):
        """
          init small cls prob parameters

          Parameters
          ----------
          param: param is a dict, the keys as follow:
                 ub_a: a tensor,store clsprob object data
                 ub_b: a tensor,store clsprob object threshold data
                 zero_tensor: a tensor, init with zero value
                 adj_len: the number of elements of each boxes with 32 alignment
                 burlen: data move nurst
                 num: process the number of elements with each repeat
                 repeat: vector repeat

          Returns
          -------
          None
          """
        param['ub_a'] = self.instance.Tensor(self.dtype,
                                             (self.one_max_size // self.dsize,),
                                             scope=tik.scope_ubuf, name="ub_a")
        param['ub_b'] = self.instance.Tensor(self.dtype,
                                             (self.one_max_size // self.dsize,),
                                             scope=tik.scope_ubuf, name="ub_b")
        param['zero_tensor'] = self.instance.Tensor(self.dtype,
                                                    (VECTOR_BYTE_SIZE,),
                                                    scope=tik.scope_ubuf,
                                                    name="zero_tensor")
        param['adj_len'] = self.get_adj_hw(self.boxes*self.height*self.width)
        param['burlen'] = self.instance.Scalar()
        param['burlen'].set_as(self.get_burlen(self.boxes*self.width*self.height))
        param['num'] = VECTOR_BYTE_SIZE // self.dsize
        param['repeat'] = self.get_repeat(self.boxes*self.width*self.height)

    def init_bigcls_param(self, loop, param):
        """
          init big cls prob parameters

          Parameters
          ----------
          param: param is a dict, the keys as follow:
                 ub_a: a tensor, store object data
                 ub_b: a tensor, store object threshold data
                 zero_tensor: a tensor, init with zero value
                 last_32b: a tensor, store last32b data
                 burlen: data move nurst
                 repeat: vector repeat times
                 num: process the number of elements with each repeat
          Returns
          -------
          None
          """
        param['ub_a'] = self.instance.Tensor(self.dtype,
                                             (self.one_max_size // self.dsize,),
                                             scope=tik.scope_ubuf, name="ub_a")

        param['ub_b'] = self.instance.Tensor(self.dtype,
                                             (self.one_max_size // self.dsize,),
                                             scope=tik.scope_ubuf, name="ub_b")
        param['zero_tensor'] = self.instance.Tensor(self.dtype,
                                                    (VECTOR_BYTE_SIZE,),
                                                    scope=tik.scope_ubuf,
                                                    name="zero_tensor")
        param['last_32b'] = self.instance.Tensor(self.dtype,
                                                 (BLOCK_SIZE,),
                                                 scope=tik.scope_ubuf,
                                                 name="last_32b")

        param['burlen'] = self.instance.Scalar(name="burlen")
        param['repeat'] = self.instance.Scalar(name="repeat")
        param['num'] = VECTOR_BYTE_SIZE // self.dsize
        with self.instance.if_scope(loop == param['mov_loop'] - VALUE_ONE):
            param['burlen'].set_as(self.get_burlen(param['last_len']))
            param['repeat'].set_as(self.get_repeat(param['last_len']))
        with self.instance.else_scope():
            param['burlen'].set_as(self.get_burlen(param['mov_len']))
            param['repeat'].set_as(self.get_repeat(param['mov_len']))

    def set_index_ub(self, param, length):
        """
          set object index after filtered by object threshold

          Parameters
          ----------
          param: param is a dict, the keys as follow:
                 index_offset: index_offset of objects
                 reduce_mask_ub: a tensor store reduce mask
                 index_ub: a tensor, store index
                 index_offset: a scalar,store index_offset
                 count: a scalar,store the number of index
          length: the number of element

          Returns
          -------
          None
          """
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
            "Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS"):
            sum_mask_ub = self.instance.Tensor(self.dtype, (16,),
                                               name="sum_mask_ub",
                                               scope=tik.scope_ubuf)
            work_tensor_ub = self.instance.Tensor(self.dtype, (16,),
                                                  name="work_tensor_ub",
                                                  scope=tik.scope_ubuf)
            self.instance.vec_reduce_add(self.mask, sum_mask_ub, param['reduce_mask_ub'], work_tensor_ub, 1, 8)

            mask_scalar = self.instance.Scalar("uint16", name="mask_scalar")
            mask_scalar.set_as(sum_mask_ub[0])
            with self.instance.if_scope(mask_scalar != 0):
                with self.instance.if_scope(param['count'] < PRE_NMS_TOPN):
                    with self.instance.for_range(0, length) as mask_index:
                        param['index_offset'].set_as(param['index_offset'] + 1)
                        with self.instance.if_scope(param['count'] < PRE_NMS_TOPN):
                            mask_scalar.set_as(param['reduce_mask_ub'][mask_index])

                            # 1 fp16 == 15360 uint16
                            with self.instance.if_scope(mask_scalar == 15360):
                                param['index_ub'][param['count']].set_as(
                                    param['index_offset'])
                                param['count'].set_as(param['count'] + 1)
            with self.instance.else_scope():
                param['index_offset'].set_as(param['index_offset'] + length)



    def cls_prob(self, batch, param):
        """
          compute cls pro

          Parameters
          ----------
          batch: the photo number
          param: param is a dict, the keys as follow:
                 index_offset: index_offset of objects
                 count: the number of filtered boxes before doing Iou
                 index_ub: is a tensor,which used to store filted obj index
                 obj_gm_offset: a scalar,store obj_gm_offset

          Returns
          -------
          None
          """
        with self.instance.new_stmt_scope():
            if self.boxes * self.hw_len * self.dsize < self.one_max_size // 2:
                self.small_clsprob(batch, param)
            else:
                self.big_clsprob(batch, param)

    def small_clsprob(self, batch, param):
        """

        Parameters
        ----------
        batch: the batch of data
        param: some param for compute

        Returns
        -------

        """
        self.init_small_clsprob_param(param)

        self.t_vmuls(param['zero_tensor'], param['zero_tensor'], VALUE_ZERO,
                     REPEAT_ONE)

        # The obj of small can be moved in at a time.
        self.t_data_move(param['ub_a'], self.obj_prob[batch, 0], param['burlen'])

        # The supplemented part is deleted for v200 reduce
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ( \
            "Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS"):
            self.t_data_move(
                self.obj_prob_v200[param['obj_gm_offset']], param['ub_a'],
                param['burlen'])

            param['obj_gm_offset'].set_as(param['obj_gm_offset'] + \
                                          self.boxes * self.hw_len)

        # if obj_prob < obj_threshold
        self.t_vector_dup(param['ub_b'], self.obj_threshold, param['repeat'])


        ones_ub = self.instance.Tensor(self.dtype, (128,), name="ones_ub",
                                       scope=tik.scope_ubuf)
        self.t_vector_dup(ones_ub, 1, 1)

        zeros_ub = self.instance.Tensor(self.dtype, (128,), name="zeros_ub",
                                        scope=tik.scope_ubuf)
        self.t_vector_dup(zeros_ub, 0, 1)


        reduce_mask_ub = self.instance.Tensor(self.dtype, (128,),
                                              name="reduce_mask_ub",
                                              scope=tik.scope_ubuf)
        index_len = self.instance.Scalar("int32")
        index_len.set_as(VECTOR_BYTE_SIZE // self.dsize)
        last_index_len = self.instance.Scalar("int32")
        last_index_len.set_as(self.obj_num % index_len)
        with self.instance.if_scope(last_index_len == 0):
            last_index_len.set_as(index_len)

        with self.instance.for_range(VALUE_ZERO, param['repeat']) as cycle:
            sel = self.instance.Tensor("uint16", (8, ),
                                       name="sel",
                                       scope=tik.scope_ubuf)
            self.instance.vec_dup(8, sel, 0, 1, 8)
            self.instance.vec_cmpv_gt(sel,
                                      param['ub_a'][
                                        param['num'] * cycle],
                                      param['ub_b'][
                                        param['num'] * cycle], 1, 8, 8)

            self.instance.vec_sel(self.mask, VALUE_ZERO,
                               param['ub_a'][param['num'] * cycle],
                               sel,
                               param['ub_a'][param['num'] * cycle],
                               param['zero_tensor'], REPEAT_ONE)
            self.instance.vec_sel(self.mask, 0, reduce_mask_ub[0], sel,
                               ones_ub[0], zeros_ub[0], REPEAT_ONE,
                               STRIDE_ONE)
            param['reduce_mask_ub'] = reduce_mask_ub
            with self.instance.if_scope(cycle == param['repeat'] - 1):
                index_len.set_as(last_index_len)
            self.set_index_ub(param, index_len)

        # Calculate the tilting parameter.
        param['faces_in_loop'], param['last_loop'], param['loop'] = \
            self.get_faces_params(param['adj_len'], self.classes)

        # Computing score
        with self.instance.for_range(VALUE_ZERO, param['loop']) as loop_idx:

            ub_c = self.instance.Tensor(self.dtype,
                                        (self.one_max_size // self.dsize,),
                                        scope=tik.scope_ubuf, name="ub_c")
            last_32b = self.instance.Tensor(self.dtype,
                                            (BLOCK_SIZE,),
                                            scope=tik.scope_ubuf,
                                            name="last_32b")
            faces = self.instance.Scalar("int32")
            with self.instance.if_scope(loop_idx != param['loop'] - VALUE_ONE):
                faces.set_as(param['faces_in_loop'])
            with self.instance.else_scope():
                faces.set_as(param['last_loop'])

            param['burlen'].set_as(
                (faces * param['adj_len'] * self.dsize) // BLOCK_SIZE)

            self.t_data_move(
                ub_c,
                self.classes_prob[batch, param['faces_in_loop']*loop_idx, 0],
                param['burlen'])

            param['burlen'].set_as(
                self.get_burlen(self.boxes*self.width*self.height))

            # a face = h*w*box
            with self.instance.for_range(VALUE_ZERO, faces) as loop:
                param['ub_d'] =\
                    self.instance.Tensor(self.dtype,
                                         (self.one_max_size // self.dsize,),
                                         scope=tik.scope_ubuf,
                                         name="ub_d")
                start_idx = self.instance.Scalar("int32")
                start_idx.set_as(loop * param['adj_len'])
                co_id = self.instance.Scalar("int32")
                co_id.set_as(param['faces_in_loop'] * loop_idx + loop)

                self.t_vmul(param['ub_d'], ub_c[start_idx], param['ub_a'],
                            param['repeat'])

                if (self.obj_num*self.dsize) % BLOCK_SIZE != 0 and \
                        self.block_num > 1:
                    with self.instance.if_scope(co_id == self.classes - 1):
                        param['burlen'].set_as(param['burlen'] - 1)
                        self.t_data_move(
                            self.inter_classes[batch, co_id, 0],
                            param['ub_d'], param['burlen'])
                        param['burlen'].set_as(param['burlen'] + 1)

                        tail_idx = self.instance.Scalar("int32")
                        tail_idx.set_as(self.obj_num - self.len_32b)
                        self.t_data_move(
                            last_32b,
                            self.inter_classes[batch, co_id, tail_idx], 1)

                        with self.instance.for_range(0, self.hwtail_len) as cycle:
                            tmp_scalar = self.instance.Scalar(self.dtype)
                            tmp_scalar.set_as(
                                param['ub_d'][self.obj_num - self.hwtail_len + cycle])
                            last_32b[self.len_32b - self.hwtail_len + cycle].set_as(
                                tmp_scalar)

                        self.t_data_move(
                            self.inter_classes[batch, co_id, tail_idx],
                            last_32b, 1)

                    with self.instance.else_scope():
                        self.t_data_move(
                            self.inter_classes[batch, co_id, 0],
                            param['ub_d'], param['burlen'])
                else:
                    self.t_data_move(
                        self.inter_classes[batch, co_id, 0],
                        param['ub_d'], param['burlen'])

    def big_clsprob(self, batch, param):
        """
        compute big data

        Parameters
        ----------
        batch: the batch of data
        param: some param for compute

        Returns
        -------

        """
        param['mov_len'], param['mov_loop'], param[
            'last_len'] = self.get_tiling_param(self.boxes * self.height,
                                                self.width)
        each_len = self.instance.Scalar("int32")
        each_len.set_as(param['mov_len'])
        with self.instance.for_range(VALUE_ZERO, param['mov_loop']) as loop:
            self.init_bigcls_param(loop, param)

            self.t_vmuls(param['zero_tensor'], param['zero_tensor'],
                         VALUE_ZERO, REPEAT_ONE)

            # move obj data to ub a
            self.t_data_move(param['ub_a'],
                             self.obj_prob[batch, param['mov_len'] * loop],
                             param['burlen'])

            # if obj_prob < obj_threshold
            self.t_vector_dup(param['ub_b'], self.obj_threshold,
                              param['repeat'])

            reduce_mask_ub = self.instance.Tensor(self.dtype, (128,),
                                                  name="reduce_mask_ub",
                                                  scope=tik.scope_ubuf)

            ones_ub = self.instance.Tensor(self.dtype, (128,), name="ones_ub",
                                           scope=tik.scope_ubuf)
            self.t_vector_dup(ones_ub, 1, 1)

            zeros_ub = self.instance.Tensor(self.dtype, (128,), name="zeros_ub",
                                            scope=tik.scope_ubuf)
            self.t_vector_dup(zeros_ub, 0, 1)


            with self.instance.if_scope(loop == param['mov_loop'] - 1):
                each_len.set_as(param['last_len'])
            index_len = self.instance.Scalar("int32")
            index_len.set_as(param['num'])
            last_index_len = self.instance.Scalar("int32")
            last_index_len.set_as(each_len % index_len)
            with self.instance.if_scope(last_index_len == 0):
                last_index_len.set_as(index_len)

            if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") not in ( \
                "Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS"):
                self.t_data_move(
                    self.obj_prob_v200[param['obj_gm_offset']],
                    param['ub_a'], param['burlen'])

                with self.instance.if_scope(loop == param['mov_loop'] - 1):
                    param['obj_gm_offset'].set_as(param['obj_gm_offset'] + \
                                                  param['last_len'])
                with self.instance.else_scope():
                    param['obj_gm_offset'].set_as(param['obj_gm_offset'] + \
                                                  param['mov_len'])

            with self.instance.for_range(VALUE_ZERO, param['repeat']) as cycle:
                sel = self.instance.Tensor("uint16", (8, ),
                                           name="sel",
                                           scope=tik.scope_ubuf)
                self.instance.vec_dup(8, sel, 0, 1, 8)
                self.instance.vec_cmpv_gt(sel,
                                          param['ub_a'][
                                              param['num'] * cycle], 
                                          param['ub_b'][
                                              param['num'] * cycle],
                                          1, 8, 8)

                self.instance.vec_sel(self.mask, VALUE_ZERO, param['ub_a'][
                    param['num'] * cycle], sel,
                                   param['ub_a'][
                                       param['num'] * cycle],
                                   param['zero_tensor'], REPEAT_ONE)

                self.instance.vec_sel(self.mask, 0, reduce_mask_ub[0], sel,
                                   ones_ub[0], zeros_ub[0], 1, 1)
                param['reduce_mask_ub'] = reduce_mask_ub
                with self.instance.if_scope(cycle == param['repeat'] - 1):
                    index_len.set_as(last_index_len)
                self.set_index_ub(param, index_len)
            shape = self.one_max_size // self.dsize
            thread_num = 1 if self.classes == 1 else 2
            with self.instance.for_range(VALUE_ZERO, self.classes,
                                         thread_num=thread_num) as co_id:

                param['ub_c'] = self.instance.Tensor(self.dtype,
                                                     (shape,),
                                                     scope=tik.scope_ubuf,
                                                     name="ub_c")

                # move classes data to ub c
                self.t_data_move(
                    param['ub_c'],
                    self.classes_prob[batch, co_id, param['mov_len'] * loop],
                    param['burlen'])

                self.t_vmul(param['ub_c'], param['ub_a'], param['ub_c'],
                            param['repeat'])

                if (self.obj_num*self.dsize) % BLOCK_SIZE != 0 and self.block_num > 1:
                    with self.instance.if_scope(
                            tik.all(co_id == self.classes - 1,
                                    loop == param['mov_loop'] - 1)):
                        param['burlen'].set_as(param['burlen'] - 1)
                        with self.instance.if_scope(param['burlen'] > 0):
                            self.t_data_move(
                                self.inter_classes[batch, co_id, param['mov_len'] * loop],
                                param['ub_c'], param['burlen'])
                        param['burlen'].set_as(param['burlen'] + 1)

                        tail_idx = self.instance.Scalar("int32")
                        tail_idx.set_as(param['last_len'] - self.len_32b)

                        self.t_data_move(
                            param['last_32b'],
                            self.inter_classes[batch, co_id, param['mov_len'] * loop + tail_idx],
                            1)

                        with self.instance.for_range(0, self.hwtail_len) as cycle:
                            scalar = self.instance.Scalar(self.dtype)
                            scalar.set_as(param['ub_c'][param['last_len'] \
                                                        - self.hwtail_len + cycle])
                            param['last_32b'][self.len_32b - \
                                              self.hwtail_len + cycle].set_as(scalar)
                        offset = param['mov_len'] * loop + tail_idx
                        dest = self.inter_classes[batch, co_id, offset]
                        self.t_data_move(dest, param['last_32b'], 1)

                    with self.instance.else_scope():
                        self.t_data_move(
                            self.inter_classes[batch, co_id, param['mov_len'] * loop],
                            param['ub_c'], param['burlen'])

                else:
                    self.t_data_move(
                        self.inter_classes[batch, co_id, param['mov_len'] * loop],
                        param['ub_c'], param['burlen'])


def get_loop_param(length, max_ub_num):
    """
    get loop parameters

    Parameters
    ----------
    length: total number
    max_ub_num: max of ub num

    Returns
    -------
    loop_cycle: loop cycle
    last_ub_num: the last data needs ub num
    """
    loop_cycle = length // max_ub_num
    last_ub_num = length % max_ub_num
    ub_num = max_ub_num
    if loop_cycle == 0:
        ub_num = length
    if last_ub_num != 0:
        loop_cycle = loop_cycle + 1
    else:
        last_ub_num = max_ub_num

    return loop_cycle, ub_num, last_ub_num


def check_param(input_dict):
    """
      check parameters

      Parameters
      ----------
      input_dict: input_dict is a dict, the keys as follow:
                  box1_info,box2_info,box3_info,biases1,biases2,biases3,
                  coords,boxes,classes,relative,obj_threshold,post_nms_topn,
                  post_nms_topn,iou_threshold,pre_nms_topn,
                  max_box_number_per_batch,kernel_name, for more details,
                  please check the yolov3_detection_output function

      Returns
      -------
      None
      """
    pre_nms_topn = input_dict.get("pre_nms_topn")
    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ( \
            "Ascend310", "Ascend910", "Hi3796CV300ES", "Hi3796CV300CS"):
        op_utils.check_dtype(input_dict.get("dtype"), ["float16"], param_name="windex")
    else:
        op_utils.check_dtype(input_dict.get("dtype"), ["float16", "float32"], param_name="windex")
    util.check_kernel_name(input_dict.get("kernel_name"))
    coords = input_dict.get("coords")
    post_nms_topn = input_dict.get("post_nms_topn")
    if coords != 4:
        error_info = {}
        error_info['errCode'] = 'E80017'
        error_info['opname'] = 'yolo_v2_detection_output_d'
        error_info['param_name'] = 'coords'
        error_info['expect_value'] = '4'
        error_info['real_value'] = str(coords)
        raise RuntimeError(error_info,
            "In op[%s], the parameter[%s] should be [%s], but actually is [%s]."
            % (error_info['opname'], error_info['param_name'], error_info['expect_value'],
               error_info['real_value']))

    max_box_number_per_batch = input_dict.get("max_box_number_per_batch")
    dtype = input_dict.get("dtype")
    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in (
            "Hi3796CV300ES", "Hi3796CV300CS") \
            or dtype == constant.DATA_TYPE_FP32:
        if pre_nms_topn > PRE_NMS_TOPN // 2 or pre_nms_topn <= 0:
            check_param_range("pre_nms_topn", 1, PRE_NMS_TOPN // 2,
                              pre_nms_topn)
    else:
        if pre_nms_topn > PRE_NMS_TOPN or pre_nms_topn <= 0:
            check_param_range("pre_nms_topn", 1, PRE_NMS_TOPN, pre_nms_topn)
    if max_box_number_per_batch > PRE_NMS_TOPN or max_box_number_per_batch <= 0:
        check_param_range("max_box_number_per_batch", 1, PRE_NMS_TOPN,
                          max_box_number_per_batch)

    dsize = common.get_data_size(input_dict.get("dtype"))
    height = input_dict.get("height")
    width = input_dict.get("width")
    if height * width * dsize < constant.BLOCK_SIZE:
        raise RuntimeError(
            "height[%d] multi with width[%d]'s size \
            must bigger than 32b" % (height, width))
