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
crop_and_resize
"""
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *
from impl import common_util
from impl.util.util_tik_comm_func import tik_func_vector
from impl.util.util_tik_comm_func import tik_func_vcomple
from impl.util.util_tik_comm_func import tik_func_vmuls
from impl.util.util_tik_comm_func import tik_func_vconv


# pylint: disable=locally-disabled,invalid-name,too-many-arguments,too-many-locals
# pylint: disable=unused-argument
def check_supported(x, boxes, box_index, y, crop_size, extrapolation_value,
                    method, kernel_name="crop_and_resize"):
    """To check whether the AICORE operator can support the length of w/h or not
    """
    input_shape = x.get("ori_shape")
    input_type = x.get("dtype")
    input_format = x.get("ori_format")
    output_h, output_w = crop_size
    boxes_shape = boxes.get("ori_shape")
    boxes_num = boxes_shape[0]

    if boxes_num <= 50 or boxes_num > 4000:
        # boxes_num is more, the performance is better than aicpu
        return False

    if input_type in ("float32", "float16",) and method in ("bilinear",) and len(input_shape) == 4:
        # shape must be [N, H, W, C] or [N, C, H, W]
        # method only support bilinear
        # c0 // num in one block
        copy_block = 2
    else:
        return False

    # format must be ("NHWC", "NCHW")
    if input_format in ("NHWC",):
        input_c = input_shape[3]
        input_h = input_shape[1]
        input_w = input_shape[2]
    elif input_format in ("NCHW",):
        input_c = input_shape[1]
        input_h = input_shape[2]
        input_w = input_shape[3]
    else:
        # format must be NHWC or NCHW
        return False

    if input_c > 2048 or input_c < 256 or max(output_h, output_w) > 16:
        # tmp limit for fasterrcnn
        return False

    if input_h * input_w * copy_block > 30000 or output_h * output_w * copy_block > 30000:
        return False

    return True


# pylint: disable=too-many-instance-attributes
class CropAndResize:
    """
    Function: use to store CropAndResize base parameters
    Modify : 2020-8-4
    """
    def __init__(self,
                 x,
                 boxes,
                 box_index,
                 crop_size,
                 y,
                 extrapolation_value,
                 method):
        """
        Init CropAndResize base parameters

        Returns
        -------
        None
        """
        self.image_shape = x.get("shape")
        self.image_type = x.get("dtype")
        self.boxes_shape = boxes.get("shape")
        self.boxes_type = boxes.get("dtype")
        self.boxes_index_shape = box_index.get("shape")
        self.boxes_index_type = box_index.get("dtype")
        self.crop_size = crop_size
        self.extrapolation_value = extrapolation_value
        self.method = method
        self.output_shape = y.get("shape")
        self.output_type = y.get("dtype")

        # init tik_instance
        self.tik_instance = tik.Tik()
        self.aicore_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.input_gm_list = []
        self.output_gm_list = []

        # parsing input
        self.crop_height, self.crop_width = crop_size
        self.batch_size, self.image_c1, self.image_height, self.image_width, self.image_c0 = self.image_shape
        self.num_boxes, _ = self.boxes_shape
        byte_num_one = common_util.get_data_size(self.image_type)
        self.image_block_num = 32 // byte_num_one
        self.image_vector_num = self.image_block_num*8
        byte_num_one = common_util.get_data_size(self.boxes_type)
        self.boxes_block_num = 32 // byte_num_one
        self.boxes_vector_num = self.boxes_block_num*8
        self.block_num = self.boxes_block_num
        self.vector_num = self.boxes_vector_num

        self.index_ub = None
        self.height_mask_list = None
        self.width_mask_list = None

    def get_tik_instance(self):
        """get_tik_instance
        """
        return self.tik_instance

    def apply_mem(self, shape, name, mem_type=tik.scope_ubuf, dtype=None):
        """tik.scope_ubuf / tik.scope_cbuf / tik.scope_gm
        """
        if dtype is None:
            dtype = self.image_type

        return self.tik_instance.Tensor(dtype, shape, name=name, scope=mem_type)

    def init_gm_mem(self):
        """init tik gm mem
        """
        # init gm input
        image_gm = self.apply_mem(self.image_shape, "image_gm", tik.scope_gm)
        boxes_gm = self.apply_mem(self.boxes_shape, "boxes_gm", tik.scope_gm, self.boxes_type)
        boxes_index_gm = self.apply_mem(self.boxes_index_shape, "boxes_index_gm", tik.scope_gm, self.boxes_index_type)

        self.input_gm_list = [image_gm, boxes_gm, boxes_index_gm]

        # init gm output
        y_gm = self.apply_mem(self.output_shape, "y_gm", tik.scope_gm, self.output_type)
        self.output_gm_list = [y_gm]

    def init_ub_scalar(self):
        """gen two uint64 scalar for vector mask
        one is 0001000100010001000100010001000100010001000100010001000100010001
        another is 0010001000100010001000100010001000100010001000100010001000100010
        """
        # for vector mask
        height_mask1_scalar = self.tik_instance.Scalar(dtype="uint64")
        width_mask1_scalar = self.tik_instance.Scalar(dtype="uint64")
        height_scale_mask = int("0001"*16, 2)
        width_scale_mask = int("0010"*16, 2)
        if self.block_num == 8:
            zero_scalar = self.tik_instance.Scalar(dtype="uint64")
            zero_scalar.set_as(0)
            height_mask1_scalar.set_as(height_scale_mask)
            width_mask1_scalar.set_as(width_scale_mask)
            self.height_mask_list = [zero_scalar, height_mask1_scalar]
            self.width_mask_list = [zero_scalar, width_mask1_scalar]
        else:
            height_mask1_scalar.set_as(height_scale_mask)
            width_mask1_scalar.set_as(width_scale_mask)
            self.height_mask_list = [height_mask1_scalar, height_mask1_scalar]
            self.width_mask_list = [width_mask1_scalar, width_mask1_scalar]

    def build_tik_instance(self, kernel_name_value):
        """build tik instance
        """
        self.tik_instance.BuildCCE(kernel_name=kernel_name_value,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   output_files_path=None,
                                   enable_l2=False)

        return self.tik_instance

    def get_core_cut(self):
        """calc the core used num and boxes num for per core
        """
        boxes_num_per_core = get_ceil_int(self.num_boxes, self.aicore_num)
        core_used = get_ceil_int(self.num_boxes, boxes_num_per_core)
        boxes_num_last_core = self.num_boxes - boxes_num_per_core * (core_used - 1)

        return core_used, boxes_num_per_core, boxes_num_last_core


def get_ceil_int(int1, int2):
    """get cel for input1 and input2
    """
    if int1 == 0:
        return 1
    _result = int1 // int2
    if int1 % int2 == 0:
        return _result

    return _result + 1


def fill_index_in_ub(tik_instance, idx_ub, idx_num):
    """fill 0,1,2  .... (idx_num -1) in idx_ub
    when the idx_num is less than 16, fill it one by one
    when the type is not int32, will fill in int32 ub and cast to idx_ub dtype
    when the type is int32, will fill in int32 one by one
    """
    # when the idx_num is less than 16, fill it one by one
    idx_ub_type = idx_ub.dtype
    if idx_num <= 16:
        _idx_scalar = tik_instance.Scalar(dtype=idx_ub.dtype)
        for _idx in range(idx_num):
            _idx_scalar.set_as(_idx)
            idx_ub[_idx].set_as(_idx_scalar)
    # when the type is not int32, will fill in int32 ub and cast to idx_ub dtype
    elif idx_ub_type not in ("int32",):
        with tik_instance.new_stmt_scope():
            idx_ub_int32 = tik_instance.Tensor("int32", (idx_num, ),
                                               name="idx_ub_int32", scope=tik.scope_ubuf)
            with tik_instance.for_range(0, idx_num) as _idx:
                idx_ub_int32[_idx].set_as(_idx)
            # cast to idx_ub
            tik_func_vconv(tik_instance, idx_ub, idx_ub_int32, idx_num)
    else:
        with tik_instance.for_range(0, idx_num) as _idx:
            idx_ub[_idx].set_as(_idx)


# pylint: disable=too-many-statements
def do_crop_and_resize_compute_one_core(box_num_sigment, obj, box_num_offset):
    """do crop and resize in one core
        step 1 read boxes from boxes and calc h_top_index/h_bottom_index/h_lerp/w_left_index/w_right_index/w_lerp
        step 2 read input_batch_num from box_index
        step 3 copy 4 data(Total C(C1*C0)) in ub
               use use input_batch_num/h_top_index/h_bottom_index/w_left_index/w_right_index
        step 4 calcu the out
               top = top_left + (top_right - top_left) * x_lerp
               bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
               out = top + (bottom - top) * y_lerp;

    Parameters:
    ----------
    box_num_sigment : int.
        the crop boxes num for one core
    obj : class.
        crop_and_resize par object
    box_num_offset: int
        copy boxes offset

    Returns
    -------
    None
    """
    tik_instance = obj.get_tik_instance()
    # get float32 index ub
    index_ub = obj.index_ub
    men_len = get_ceil_int(box_num_sigment*4, obj.boxes_vector_num) * obj.boxes_vector_num

    # apply ub mem for index
    boxes_ub_small = obj.apply_mem((men_len,), "boxes_ub_h1", tik.scope_ubuf, obj.boxes_type)
    boxes_ub_big = obj.apply_mem((men_len,), "boxes_ub_h2", tik.scope_ubuf, obj.boxes_type)
    boxes_ub_scale = obj.apply_mem((men_len,), "boxes_ub_scale", tik.scope_ubuf, obj.boxes_type)
    copy_burst_len = get_ceil_int(box_num_sigment*4, obj.boxes_block_num)

    # init ub for input offset
    batch_offset_ub = obj.apply_mem((obj.boxes_vector_num,), "batch_offset_ub", tik.scope_ubuf, "int32")
    height_offset_ub = obj.apply_mem((obj.boxes_vector_num,), "height_offset_ub", tik.scope_ubuf, "int32")
    width_offset_ub = obj.apply_mem((obj.boxes_vector_num,), "width_offset_ub", tik.scope_ubuf, "int32")
    tik_instance.vector_dup(obj.boxes_vector_num, batch_offset_ub,
                            obj.image_c1*obj.image_c0*obj.image_height*obj.image_width, 1, 1, 8)
    tik_instance.vector_dup(obj.boxes_vector_num, height_offset_ub,
                            obj.image_c0*obj.image_width, 1, 1, 8)
    tik_instance.vector_dup(obj.boxes_vector_num, width_offset_ub,
                            obj.image_c0, 1, 1, 8)

    # copy boxes in boxes_ub_small
    tik_instance.data_move(boxes_ub_small, obj.input_gm_list[1][box_num_offset*4],
                           0, 1, copy_burst_len, 0, 0)
    copy_burst_len = get_ceil_int(box_num_sigment*4 - 2, obj.boxes_block_num)
    # copy boxes[2] in boxes_ub_small
    tik_instance.data_move(boxes_ub_big, obj.input_gm_list[1][box_num_offset*4 + 2],
                           0, 1, copy_burst_len, 0, 0)
    # calc boxes[2] - boxes  means y2 - y1 and x2 - x1
    tik_func_vcomple(tik_instance, "vsub", boxes_ub_scale, boxes_ub_big, boxes_ub_small, men_len)
    if obj.crop_height <= 1 or obj.crop_width <= 1:
        tik_func_vcomple(tik_instance, "vadd", boxes_ub_big, boxes_ub_big, boxes_ub_small, men_len)

    # calc resize scale for h and w
    repeat_time = get_ceil_int(box_num_sigment*4, obj.boxes_vector_num)
    if obj.crop_height > 1:
        # to get scale_h: scale * (image_height - 1) / (crop_height - 1)
        tik_instance.vmuls([obj.height_mask_list[0], obj.height_mask_list[1]],
                           boxes_ub_scale, boxes_ub_scale, (obj.image_height - 1) / (obj.crop_height - 1),
                           repeat_time, 1, 1, 8, 8)
    if obj.crop_width > 1:
        # to get scale_w:  scale * (image_width - 1) / (crop_width - 1)
        tik_instance.vmuls(obj.width_mask_list,
                           boxes_ub_scale, boxes_ub_scale, (obj.image_width - 1) / (obj.crop_width - 1),
                           repeat_time, 1, 1, 8, 8)
    # to get h_small: h_small * (image_height - 1)
    if obj.crop_height > 1:
        # to get h_small: h_small * (image_height - 1)
        tik_instance.vmuls(obj.height_mask_list,
                           boxes_ub_small, boxes_ub_small, obj.image_height - 1,
                           repeat_time, 1, 1, 8, 8)
    else:
        # to get h_small: (h_small + h_big) * (image_height - 1) * 0.5
        tik_instance.vmuls(obj.height_mask_list,
                           boxes_ub_small, boxes_ub_big, 0.5,
                           repeat_time, 1, 1, 8, 8)
        tik_instance.vmuls(obj.height_mask_list,
                           boxes_ub_small, boxes_ub_small, obj.image_height - 1,
                           repeat_time, 1, 1, 8, 8)

    if obj.crop_width > 1:
        # to get w_small: w_small * (image_width - 1)
        tik_instance.vmuls(obj.width_mask_list,
                           boxes_ub_small, boxes_ub_small, obj.image_width - 1,
                           repeat_time, 1, 1, 8, 8)
    else:
        # to get w_small: (w_small + w_big) * (image_width - 1) * 0.5
        tik_instance.vmuls(obj.width_mask_list,
                           boxes_ub_small, boxes_ub_big, 0.5,
                           repeat_time, 1, 1, 8, 8)
        tik_instance.vmuls(obj.width_mask_list,
                           boxes_ub_small, boxes_ub_small, obj.image_width - 1,
                           repeat_time, 1, 1, 8, 8)

    # box_index process for one sigment
    box_index_ub = obj.apply_mem((get_ceil_int(box_num_sigment, obj.boxes_block_num)*obj.boxes_block_num,),
                                 "box_index_ub", tik.scope_ubuf, "int32")
    copy_burst_len = get_ceil_int(box_num_sigment, obj.boxes_block_num)
    tik_instance.data_move(box_index_ub, obj.input_gm_list[2][box_num_offset],
                           0, 1, copy_burst_len, 0, 0)
    tik_func_vcomple(tik_instance, "vmul", box_index_ub, box_index_ub, batch_offset_ub, box_num_sigment, src1_rep=0)

    with tik_instance.for_range(0, box_num_sigment) as _box_idx:
        _out_batch_idx = _box_idx + box_num_offset
        scaler_h_small = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        scaler_w_small = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        scaler_h_scale = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        scaler_w_scale = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        # read scale for h and w
        scaler_h_small.set_as(boxes_ub_small[_box_idx*4])
        scaler_w_small.set_as(boxes_ub_small[_box_idx*4 + 1])
        scaler_h_scale.set_as(boxes_ub_scale[_box_idx*4])
        scaler_w_scale.set_as(boxes_ub_scale[_box_idx*4 + 1])

        input_boxes_in_h = obj.apply_mem((get_ceil_int(obj.crop_height, obj.boxes_block_num) * obj.boxes_block_num,),
                                         "input_boxes_in_h", tik.scope_ubuf, obj.boxes_type)
        tik_instance.vmuls(obj.crop_height,
                           input_boxes_in_h, index_ub, scaler_h_scale,
                           1, 1, 1, 8, 8)
        input_boxes_in_w = obj.apply_mem((get_ceil_int(obj.crop_width, obj.boxes_block_num) * obj.boxes_block_num,),
                                         "input_boxes_in_w", tik.scope_ubuf, obj.boxes_type)
        tik_instance.vmuls(obj.crop_width,
                           input_boxes_in_w, index_ub, scaler_w_scale,
                           1, 1, 1, 8, 8)
        tik_instance.vadds(obj.crop_height,
                           input_boxes_in_h, input_boxes_in_h, scaler_h_small,
                           1, 1, 1, 8, 8)
        tik_instance.vadds(obj.crop_width,
                           input_boxes_in_w, input_boxes_in_w, scaler_w_small,
                           1, 1, 1, 8, 8)

        h_top_index = \
            obj.apply_mem((get_ceil_int(obj.crop_height, obj.boxes_block_num) * obj.boxes_block_num,),
                          "h_top_index", tik.scope_ubuf, "int32")
        w_left_index = \
            obj.apply_mem((get_ceil_int(obj.crop_width, obj.boxes_block_num) * obj.boxes_block_num,),
                          "w_left_index", tik.scope_ubuf, "int32")
        h_index_post = \
            obj.apply_mem((get_ceil_int(obj.crop_height, obj.boxes_block_num) * obj.boxes_block_num,),
                          "h_index_post", tik.scope_ubuf, "int32")
        w_index_post = \
            obj.apply_mem((get_ceil_int(obj.crop_width, obj.boxes_block_num) * obj.boxes_block_num,),
                          "w_index_post", tik.scope_ubuf, "int32")

        cast_flag = tbe_platform.cce_conf.api_check_support("tik.vconv", "f322s32r")
        with tik_instance.new_stmt_scope():
            tmp_float_ub_0 = obj.apply_mem((get_ceil_int(obj.crop_height,
                                                         obj.boxes_block_num)
                                            * obj.boxes_block_num,),
                                           "tmp_float_ub_0", tik.scope_ubuf, obj.boxes_type)
            if not cast_flag:
                tik_func_vconv(tik_instance, h_top_index, input_boxes_in_h, obj.crop_height,
                               mode="floor", mini_mid_ub=tmp_float_ub_0)
            else:
                tik_func_vconv(tik_instance, h_top_index, input_boxes_in_h, obj.crop_height,
                               mode="floor")
                # h_top_index vconv from int32 to float32
                tik_func_vconv(tik_instance, tmp_float_ub_0, h_top_index, obj.crop_height)
            tik_func_vcomple(tik_instance, "vmul", h_top_index, h_top_index, height_offset_ub, obj.crop_height,
                             src1_rep=0)

            # do: h_lerp = input_boxes_in_h - tmp_float_ub
            tik_func_vcomple(tik_instance, "vsub", input_boxes_in_h,
                             input_boxes_in_h, tmp_float_ub_0, obj.crop_height)
            tik_func_vconv(tik_instance, h_index_post, input_boxes_in_h, obj.crop_height,
                           mode="ceil")
            tmp_float_ub_1 = obj.apply_mem((get_ceil_int(obj.crop_width,
                                                         obj.boxes_block_num)
                                            * obj.boxes_block_num,),
                                           "tmp_float_ub_1", tik.scope_ubuf, obj.boxes_type)
            if not cast_flag:
                tik_func_vconv(tik_instance, w_left_index, input_boxes_in_w, obj.crop_width,
                               mode="floor", mini_mid_ub=tmp_float_ub_1)
            else:
                tik_func_vconv(tik_instance, w_left_index, input_boxes_in_w, obj.crop_width,
                               mode="floor")
                # h_top_index vconv from int32 to float32
                tik_func_vconv(tik_instance, tmp_float_ub_1, w_left_index, obj.crop_width)
            tik_func_vcomple(tik_instance, "vmul", w_left_index, w_left_index, width_offset_ub, obj.crop_width,
                             src1_rep=0)
            # do: w_lerp = input_boxes_in_h - tmp_float_ub
            tik_func_vcomple(tik_instance, "vsub", input_boxes_in_w,
                             input_boxes_in_w, tmp_float_ub_1, obj.crop_width)
            tik_func_vconv(tik_instance, w_index_post, input_boxes_in_w, obj.crop_width,
                           mode="ceil")

        # read input batch index and calc input offset
        input_batch_offset = tik_instance.Scalar(dtype="int32")
        input_batch_offset.set_as(box_index_ub[_box_idx])
        input_h_offset = tik_instance.Scalar(dtype="int32")
        input_h_post = tik_instance.Scalar(dtype="int32")
        h_lerp = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
        c0_block_num = obj.image_c0 // obj.block_num
        image_gm = obj.input_gm_list[0]
        output_gm = obj.output_gm_list[0]
        with tik_instance.for_range(0, obj.crop_height) as _crop_height_idx:
            input_h_offset.set_as(h_top_index[_crop_height_idx])
            input_h_post.set_as(h_index_post[_crop_height_idx])
            real_h_offset = input_h_offset + input_h_post
            with tik_instance.if_scope(
                    tik.all(input_h_offset >= 0,
                            real_h_offset <= (obj.image_height - 1)*obj.image_c0*obj.image_width)):
                h_lerp.set_as(input_boxes_in_h[_crop_height_idx])
                thread_num = 2
                if obj.crop_width <= 1:
                    thread_num = 1
                with tik_instance.for_range(0, obj.crop_width, thread_num=thread_num) as _crop_width_idx:
                    input_w_offset = tik_instance.Scalar(dtype="int32")
                    input_w_post = tik_instance.Scalar(dtype="int32")
                    w_lerp = tik_instance.Scalar(dtype=boxes_ub_small.dtype)
                    input_w_offset.set_as(w_left_index[_crop_width_idx])
                    input_w_post.set_as(w_index_post[_crop_width_idx])
                    real_w_offset = input_w_offset + input_w_post
                    with tik_instance.if_scope(tik.all(input_w_offset >= 0,
                                                       real_w_offset <= (obj.image_width - 1) * obj.image_c0)):
                        w_lerp.set_as(input_boxes_in_w[_crop_width_idx])
                        # copy all C data in ub
                        with tik_instance.new_stmt_scope():
                            h0_w_ub = obj.apply_mem((obj.image_c1*obj.image_c0*2,),
                                                    "h0_w_ub", tik.scope_ubuf, "float32")
                            h1_w_ub = obj.apply_mem((obj.image_c1*obj.image_c0*2,),
                                                    "h1_w_ub", tik.scope_ubuf, "float32")

                            if obj.image_block_num == obj.block_num:
                                # when input is fp32, just copy
                                if obj.image_width > 1:
                                    tik_instance.data_move(
                                        h0_w_ub, image_gm[input_batch_offset + input_h_offset + input_w_offset],
                                        0, obj.image_c1, c0_block_num*2,
                                        obj.image_height*obj.image_width*c0_block_num - c0_block_num*2, 0)
                                    if obj.image_height > 1:
                                        tik_instance.data_move(
                                            h1_w_ub,
                                            image_gm[input_batch_offset + input_h_offset + input_w_offset
                                                     + obj.image_width * obj.image_c0],
                                            0, obj.image_c1, c0_block_num*2,
                                            obj.image_height*obj.image_width*c0_block_num - c0_block_num*2, 0)
                                    else:
                                        tik_func_vector(tik_instance, h1_w_ub, 0, obj.image_c1*obj.image_c0*2)
                                else:
                                    tik_instance.data_move(
                                        h0_w_ub, image_gm[input_batch_offset + input_h_offset + input_w_offset],
                                        0, obj.image_c1, c0_block_num,
                                        obj.image_height*obj.image_width*c0_block_num - c0_block_num, c0_block_num)
                                    tik_instance.data_move(
                                        h0_w_ub[obj.image_c0],
                                        image_gm[input_batch_offset + input_h_offset + input_w_offset],
                                        0, obj.image_c1, c0_block_num,
                                        obj.image_height*obj.image_width*c0_block_num - c0_block_num, c0_block_num)
                                    if obj.image_height > 1:
                                        tik_instance.data_move(
                                            h1_w_ub,
                                            image_gm[input_batch_offset + input_h_offset + input_w_offset
                                                     + obj.image_width * obj.image_c0],
                                            0, obj.image_c1, c0_block_num,
                                            obj.image_height*obj.image_width*c0_block_num - c0_block_num, c0_block_num)
                                        tik_instance.data_move(
                                            h1_w_ub[obj.image_c0],
                                            image_gm[input_batch_offset + input_h_offset + input_w_offset
                                                     + obj.image_width * obj.image_c0],
                                            0, obj.image_c1, c0_block_num,
                                            obj.image_height*obj.image_width*c0_block_num - c0_block_num, c0_block_num)
                                    else:
                                        tik_func_vector(tik_instance, h1_w_ub, 0, obj.image_c1*obj.image_c0*2)
                            else:
                                # when input is fp16, will copy and cast to fp32
                                with tik_instance.new_stmt_scope():
                                    h0_w_ub_fp16 = obj.apply_mem((obj.image_c1*obj.image_c0*2,),
                                                                 "h0_w_ub_fp16", tik.scope_ubuf)
                                    h1_w_ub_fp16 = obj.apply_mem((obj.image_c1*obj.image_c0*2,),
                                                                 "h1_w_ub_fp16", tik.scope_ubuf)
                                    c0_block_fp16 = 1
                                    if obj.image_width > 1:
                                        tik_instance.data_move(
                                            h0_w_ub_fp16,
                                            image_gm[input_batch_offset + input_h_offset + input_w_offset],
                                            0, obj.image_c1, c0_block_fp16*2,
                                            obj.image_height*obj.image_width*c0_block_fp16 - c0_block_fp16*2, 0)
                                        tik_func_vconv(tik_instance, h0_w_ub, h0_w_ub_fp16,
                                                       obj.image_c1*obj.image_c0*2)
                                        if obj.image_height > 1:
                                            tik_instance.data_move(
                                                h1_w_ub_fp16,
                                                image_gm[input_batch_offset + input_h_offset + input_w_offset
                                                         + obj.image_width * obj.image_c0],
                                                0, obj.image_c1, c0_block_fp16*2,
                                                obj.image_height*obj.image_width*c0_block_fp16 - c0_block_fp16*2, 0)
                                            tik_func_vconv(tik_instance, h1_w_ub, h1_w_ub_fp16,
                                                           obj.image_c1*obj.image_c0*2)
                                        else:
                                            tik_func_vector(tik_instance, h1_w_ub, 0, obj.image_c1*obj.image_c0*2)
                                    else:
                                        tik_instance.data_move(
                                            h0_w_ub_fp16,
                                            image_gm[input_batch_offset + input_h_offset + input_w_offset],
                                            0, obj.image_c1, c0_block_fp16,
                                            obj.image_height*obj.image_width*c0_block_fp16 - c0_block_fp16,
                                            c0_block_fp16)
                                        tik_instance.data_move(
                                            h0_w_ub_fp16[c0_block_fp16*obj.image_c0],
                                            image_gm[input_batch_offset + input_h_offset + input_w_offset],
                                            0, obj.image_c1, c0_block_fp16,
                                            obj.image_height*obj.image_width*c0_block_fp16 - c0_block_fp16,
                                            c0_block_fp16)
                                        tik_func_vconv(tik_instance, h0_w_ub, h0_w_ub_fp16,
                                                       obj.image_c1*obj.image_c0*2)
                                        if obj.image_height > 1:
                                            tik_instance.data_move(
                                                h1_w_ub_fp16,
                                                image_gm[input_batch_offset + input_h_offset + input_w_offset
                                                         + obj.image_width * obj.image_c0],
                                                0, obj.image_c1, c0_block_fp16,
                                                obj.image_height*obj.image_width*c0_block_fp16 - c0_block_fp16,
                                                c0_block_fp16)
                                            tik_instance.data_move(
                                                h1_w_ub_fp16[c0_block_fp16*obj.image_c0],
                                                image_gm[input_batch_offset + input_h_offset + input_w_offset
                                                         + obj.image_width * obj.image_c0],
                                                0, obj.image_c1, c0_block_fp16,
                                                obj.image_height*obj.image_width*c0_block_fp16 - c0_block_fp16,
                                                c0_block_fp16)
                                            tik_func_vconv(tik_instance, h1_w_ub, h1_w_ub_fp16,
                                                           obj.image_c1*obj.image_c0*2)
                                        else:
                                            tik_func_vector(tik_instance, h1_w_ub, 0, obj.image_c1*obj.image_c0*2)

                            tik_func_vcomple(tik_instance, "vsub", h1_w_ub,
                                             h1_w_ub, h0_w_ub,
                                             obj.image_c1*obj.image_c0*2)

                            tik_func_vmuls(tik_instance, h1_w_ub,
                                           h1_w_ub, h_lerp, obj.image_c1*obj.image_c0*2)

                            tik_func_vcomple(tik_instance, "vadd", h0_w_ub,
                                             h1_w_ub, h0_w_ub,
                                             obj.image_c1*obj.image_c0*2)

                            tik_fun = tik_instance.vsub
                            tik_fun(obj.image_c0, h1_w_ub, h0_w_ub[16],
                                    h0_w_ub, obj.image_c1, 1, 1, 1, 2, 4, 4)
                            tik_func_vmuls(tik_instance, h1_w_ub,
                                           h1_w_ub, w_lerp, obj.image_c1*obj.image_c0)

                            tik_fun = tik_instance.vadd
                            tik_fun(obj.image_c0, h1_w_ub[obj.image_c1*obj.image_c0:],
                                    h1_w_ub, h0_w_ub, obj.image_c1,
                                    1, 1, 1, 2, 2, 4)
                            output_offset = \
                                _out_batch_idx*obj.image_c1*obj.crop_width*obj.crop_height*obj.image_c0 \
                                + _crop_height_idx*obj.crop_width*obj.image_c0 + _crop_width_idx*obj.image_c0
                            tik_instance.data_move(output_gm[output_offset],
                                                   h1_w_ub[obj.image_c1*obj.image_c0:], 0, obj.image_c1, c0_block_num,
                                                   0, obj.crop_height*obj.crop_width*c0_block_num - c0_block_num)
                    with tik_instance.else_scope():
                        with tik_instance.new_stmt_scope():
                            h1_w_ub = obj.apply_mem((get_ceil_int(obj.image_c1*obj.image_c0, obj.vector_num)
                                                     * obj.vector_num,),
                                                    "h1_w_ub", tik.scope_ubuf, "float32")
                            tik_func_vector(tik_instance, h1_w_ub, obj.extrapolation_value,
                                            obj.image_c1*obj.image_c0)

                            output_offset = \
                                _out_batch_idx*obj.image_c1*obj.crop_width*obj.crop_height*obj.image_c0 \
                                + _crop_height_idx*obj.crop_width*obj.image_c0 + _crop_width_idx*obj.image_c0
                            tik_instance.data_move(output_gm[output_offset],
                                                   h1_w_ub, 0, obj.image_c1, c0_block_num,
                                                   0, obj.crop_height*obj.crop_width*c0_block_num - c0_block_num)
            with tik_instance.else_scope():
                with tik_instance.new_stmt_scope():
                    h0_w_ub = obj.apply_mem((get_ceil_int(obj.image_c1*obj.crop_width*obj.image_c0,
                                                          obj.vector_num)
                                             * obj.vector_num,),
                                            "h0_w_ub", tik.scope_ubuf, "float32")
                    tik_func_vector(tik_instance, h0_w_ub, obj.extrapolation_value,
                                    obj.image_c1*obj.image_c0*obj.crop_width)
                    output_offset = \
                        _out_batch_idx*obj.image_c1*obj.crop_width*obj.crop_height*obj.image_c0 \
                        + _crop_height_idx*obj.crop_width*obj.image_c0
                    tik_instance.data_move(output_gm[output_offset],
                                           h0_w_ub, 0, obj.image_c1, c0_block_num*obj.crop_width, 0,
                                           obj.crop_height*obj.crop_width*c0_block_num - c0_block_num*obj.crop_width)


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_FLOAT,
                 REQUIRED_ATTR_STR, KERNEL_NAME)
def crop_and_resize(x, boxes, box_index, y, crop_size, extrapolation_value,
                    method, kernel_name="crop_and_resize"):
    """
    do crop_and_resize

    Parameters:
    ----------
    x : dict.
        dict info of images value, must include the keys(shape and dtype).
        and shape will be 5HD
    boxes : dict.
        dict info of boxes, a 2D Tensor of type float32 with shape (num_anchors, 4).
        "num_anchors" indicates num of boxes.
        the shape value "4" refers to "y1", "x1", "y2", and "x2".
    box_index : dict.
        dict info of box_index, a 1D Tensor of type int32 with shape (num_anchors).
        "num_anchors" indicates num of boxes. the value indicates the image index for each boxes
    y : dict.
        A 5HD Tensor with shape (batch,).
        specifying output crop image
    crop_size : list.
        A required attribute of type list int, specifying the output image size.
    extrapolation_value : float.
        A required attribute of type float32, specifying the extrapolation_value
    method : string.
        A required attribute of type string, specifying the resize type.
    kernel_name : str.
        cce kernel name, default value is "crop_and_resize"

    Returns
    -------
    tik_instance
    """
    # init object for crop_and_resize
    crop_and_resize_obj = CropAndResize(x, boxes, box_index, crop_size, y, extrapolation_value, method)
    # init gm
    tik_instance = crop_and_resize_obj.get_tik_instance()
    core_used, num_per_core, num_last_core = crop_and_resize_obj.get_core_cut()
    crop_and_resize_obj.init_gm_mem()
    with tik_instance.for_range(0, core_used, block_num=core_used) as _core_idx:
        crop_and_resize_obj.init_ub_scalar()
        max_idx = max(crop_and_resize_obj.crop_height, crop_and_resize_obj.crop_width)
        max_idx = get_ceil_int(max_idx, crop_and_resize_obj.boxes_block_num) * crop_and_resize_obj.boxes_block_num
        crop_and_resize_obj.index_ub = crop_and_resize_obj.apply_mem([max_idx], "tmp_index_ub",
                                                                     tik.scope_ubuf, crop_and_resize_obj.boxes_type)
        fill_index_in_ub(tik_instance, crop_and_resize_obj.index_ub, max_idx)
        if num_per_core == num_last_core or core_used == 1:
            do_crop_and_resize_compute_one_core(num_per_core, crop_and_resize_obj, num_per_core*_core_idx)
        else:
            with tik_instance.if_scope(_core_idx < core_used - 1):
                do_crop_and_resize_compute_one_core(num_per_core, crop_and_resize_obj, num_per_core*_core_idx)
            with tik_instance.else_scope():
                do_crop_and_resize_compute_one_core(num_last_core, crop_and_resize_obj, num_per_core*_core_idx)

    crop_and_resize_obj.build_tik_instance(kernel_name)

    return tik_instance

