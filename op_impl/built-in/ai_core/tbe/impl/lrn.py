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
lrn
"""
import math
from functools import reduce as functools_reduce
from topi.cce import util
from te import tik
from te import platform as tbe_platform
import impl.constant_util as constant
from impl import common_util
from te.utils.op_utils import *


DEPTH_RADIUS_SIZE_LIMIT = 48
MAX_CORE_NUMBER = 32
MAX_REPEAT_NUM = 255
PAR_COUNT_FP16 = 128
PAR_COUNT_FP32 = 64
MAX_HW_NUM = 65536
CUT_C_HW_SIZE = 128
C0_SIZE = 16


def _lrn_parameter_check(input_data, depth_radius, norm_region, kernel_name):
    shape_input = input_data.get("shape")
    dtype_input = input_data.get("dtype").lower()
    check_shape(shape_input, param_name="x")
    check_dtype(dtype_input, ("float16", "float32"), param_name="x")

    check_shape(shape_input, min_rank=4, max_rank=5, param_name="x")

    if depth_radius > DEPTH_RADIUS_SIZE_LIMIT:
        error_info = {}
        error_info['errCode'] = 'E81000'
        error_info['param_name'] = 'depth_radius'
        error_info['op_name'] = 'lrn'
        error_info['expect_value'] = "less than "+str(DEPTH_RADIUS_SIZE_LIMIT)
        error_info['real_value'] = "too large to calculate"
        raise ValueError(error_info, "In op[%s], the parameter [%s] is not right, it should be [%s],"
                                     "but actually is [%s]."
                         % (error_info['op_name'], error_info['param_name'],
                            error_info['expect_value'], error_info['real_value']))

    if norm_region != "ACROSS_CHANNELS":
        error_info = {}
        error_info['errCode'] = 'E81001'
        error_info['param_name'] = 'norm_region'
        error_info['op_name'] = 'lrn'
        error_info['expect_value'] = "ACROSS_CHANNELS"
        error_info['real_value'] = norm_region
        raise ValueError(error_info, "In op[%s], the parameter [%s] only support [%s] mode, "
                                     "but actually is [%s]."
                         % (error_info['op_name'], error_info['param_name'],
                            error_info['expect_value'], error_info['real_value']))

    if depth_radius < 0:
        error_info = {}
        error_info['errCode'] = 'E81000'
        error_info['param_name'] = 'depth_radius'
        error_info['op_name'] = 'lrn'
        error_info['expect_value'] = "greater equal than 0"
        error_info['real_value'] = depth_radius
        raise ValueError(error_info, "In op[%s], the parameter [%s] is not right, it should be [%s],"
                                     "but actually is [%s]."
                         % (error_info['op_name'], error_info['param_name'],
                            error_info['expect_value'], error_info['real_value']))


# pylint: disable=locally-disabled,too-many-locals,too-many-arguments,unused-argument, invalid-name
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_INT, OPTION_ATTR_FLOAT,
                 OPTION_ATTR_FLOAT, OPTION_ATTR_FLOAT, OPTION_ATTR_STR, KERNEL_NAME,
                 OPTION_ATTR_STR)
def lrn(x, y, depth_radius=5, bias=1, alpha=1, beta=0.5,
        norm_region="ACROSS_CHANNELS", kernel_name="lrn", impl_mode="high_performance"):
    """ Local Response Normalization.

    The 4-D `x` tensor is treated as a 3-D array of 1-D vectors (along the last
    dimension), and each vector is normalized independently.  Within a given
    vector, each component is divided by the weighted, squared sum of inputs
    within `depth_radius`.  In detail,

        sqr_sum[a, b, c, d] =
        sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
        output = input / (bias + alpha * sqr_sum) ** beta

    Parameters
    ----------
    x: dict.
        dict with keys(shape and dtype) of input tensor, and the dtype only
        support float16, float32. And the input format should be NCHW
    y: dict.
        dict with keys(shape and dtype) of y.
    depth_radius: int
        half-width of the 1-D normalization window,  defaults to 5.
    bias: float
        an offset, defaults to 1.
    alpha: float
        a scale factor, defaults to 1.
    beta: float
        an exponent, defaults to 0.5.
    kernel_name: str.
        cce kernel name, default value is "lrn".
    impl_mode: str.
        high_precision or high_performance for inference, default value is "high_performance".
        no need to add into ops_info file.

    Returns
    -------
    None.
    """
    _lrn_parameter_check(x, depth_radius, norm_region, kernel_name)
    if x.get("format") == "NCHW":
        lrn_d_obj = LRNBase4HD(x, depth_radius, bias, alpha, beta, kernel_name,
                               impl_mode)
        if lrn_d_obj.dtype_real_in_out == lrn_d_obj.input_dtype:
            # cast is not needed
            lrn_d_obj.alignment_standards = lrn_d_obj.one_block_ele_num_input

        if lrn_d_obj.one_column_size % lrn_d_obj.alignment_standards != 0:
            return lrn_d_obj.tik_instance_function_not_align()
        return lrn_d_obj.tik_instance_function()
    elif x.get("format") == "NC1HWC0":
        # only support float16
        lrn_d_obj = LRNBase5HD(x, depth_radius, bias, alpha, beta,
                               kernel_name, impl_mode)
        return lrn_d_obj.tik_instance_function()


class LRNBase(object):
    def __init__(self, tik_instance):
        self.tik_instance = tik_instance

    def _get_shape_size(self, data_shape):
        data_size = int(functools_reduce(lambda i, j: i * j, data_shape))
        return data_size

    def _get_mask_and_repeat(self, data_shape, data_type):
        data_size = int(functools_reduce(lambda i, j: i * j, data_shape))
        data_byte_num = common_util.get_data_size(data_type)
        one_block_num = constant.BLOCK_SIZE // data_byte_num

        front_mask = constant.REPEAT_STRIDE_EIGHT*one_block_num
        if data_size <= front_mask:
            front_mask = data_size
            last_mask = data_size
            repeat_times = constant.REPEAT_TIME_ONCE

            return front_mask, last_mask, repeat_times

        # in this case, repeat is greater than 1
        repeat_times = data_size // front_mask
        last_mask = front_mask
        if data_size % front_mask != 0:
            last_mask = data_size - repeat_times*front_mask
            repeat_times = repeat_times + 1

        return front_mask, last_mask, repeat_times

    def _double_vector_func(self, func_name, dest, src0, src1, compute_shape,
                            dest_offset=0, src0_offset=0, src1_offset=0):
        front_mask, last_mask, repeat_times = \
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        if repeat_times == 1:
            func_name(front_mask, dest[dest_offset],
                      src0[src0_offset], src1[src1_offset],
                      constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        elif repeat_times <= MAX_REPEAT_NUM:
            if front_mask == last_mask:
                func_name(front_mask, dest[dest_offset],
                          src0[src0_offset], src1[src1_offset],
                          repeat_times,
                          constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
            else:
                func_name(front_mask, dest[dest_offset],
                          src0[src0_offset], src1[src1_offset],
                          repeat_times - 1,
                          constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
                vector_offset = (repeat_times - 1)*front_mask
                func_name(last_mask, dest[dest_offset + vector_offset],
                          src0[src0_offset + vector_offset],
                          src1[src1_offset + vector_offset],
                          constant.REPEAT_TIME_ONCE,
                          constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)

        else:
            rest_repeat_num = repeat_times
            count = 0
            while rest_repeat_num > MAX_REPEAT_NUM:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src0[vector_offset],
                          src1[vector_offset], MAX_REPEAT_NUM, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
                count = count + 1
                rest_repeat_num = rest_repeat_num - MAX_REPEAT_NUM
            if rest_repeat_num != 1:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src0[vector_offset],
                          src1[vector_offset], rest_repeat_num - 1,
                          constant.STRIDE_ONE, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], src0[vector_offset],
                      src1[vector_offset], constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)

    def _vector_scalar_func(self, func_name, dest, src0, scalar_val,
                            compute_shape):
        front_mask, last_mask, repeat_times =\
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        if repeat_times == 1:
            func_name(front_mask, dest, src0, scalar_val,
                      constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        elif repeat_times <= MAX_REPEAT_NUM:
            func_name(front_mask, dest, src0, scalar_val, repeat_times - 1,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], src0[vector_offset],
                      scalar_val, constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        else:
            rest_repeat_num = repeat_times
            count = 0
            while rest_repeat_num > MAX_REPEAT_NUM:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src0[vector_offset],
                          scalar_val, MAX_REPEAT_NUM, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
                count = count + 1
                rest_repeat_num = rest_repeat_num - MAX_REPEAT_NUM
            if rest_repeat_num != 1:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src0[vector_offset],
                          scalar_val, rest_repeat_num - 1, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], src0[vector_offset],
                      scalar_val, constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)

    def _vector_dup_func(self, dest, scalar_val, compute_shape):
        func_name = self.tik_instance.vector_dup
        front_mask, last_mask, repeat_times =\
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        if repeat_times == 1:
            func_name(front_mask, dest, scalar_val, constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
        elif repeat_times <= MAX_REPEAT_NUM:
            func_name(front_mask, dest, scalar_val, repeat_times - 1,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], scalar_val,
                      constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
        else:
            rest_repeat_num = repeat_times
            count = 0
            while rest_repeat_num > MAX_REPEAT_NUM:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], scalar_val,
                          MAX_REPEAT_NUM, constant.STRIDE_ONE,
                          constant.REPEAT_STRIDE_EIGHT)
                count = count + 1
                rest_repeat_num = rest_repeat_num - MAX_REPEAT_NUM
            if rest_repeat_num != 1:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], scalar_val,
                          rest_repeat_num - 1,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], scalar_val,
                      constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT)

    def _single_vector_func(self, func_name, dest, src, compute_shape):
        front_mask, last_mask, repeat_times =\
            self._get_mask_and_repeat(compute_shape, dest.dtype)
        if repeat_times == 1:
            func_name(front_mask, dest, src, constant.REPEAT_TIME_ONCE,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        elif repeat_times <= MAX_REPEAT_NUM:
            func_name(front_mask, dest, src, repeat_times - 1,
                      constant.STRIDE_ONE, constant.STRIDE_ONE,
                      constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], src[vector_offset],
                      constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)
        else:
            rest_repeat_num = repeat_times
            count = 0
            while rest_repeat_num > MAX_REPEAT_NUM:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src[vector_offset],
                          MAX_REPEAT_NUM, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
                count = count + 1
                rest_repeat_num = rest_repeat_num - MAX_REPEAT_NUM
            if rest_repeat_num != 1:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                func_name(front_mask, dest[vector_offset], src[vector_offset],
                          rest_repeat_num - 1, constant.STRIDE_ONE,
                          constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                          constant.REPEAT_STRIDE_EIGHT)
            vector_offset = (repeat_times - 1)*front_mask
            func_name(last_mask, dest[vector_offset], src[vector_offset],
                      constant.REPEAT_TIME_ONCE, constant.STRIDE_ONE,
                      constant.STRIDE_ONE, constant.REPEAT_STRIDE_EIGHT,
                      constant.REPEAT_STRIDE_EIGHT)

    def _vconv_func(self, dest, src, compute_shape):
        front_mask, last_mask, repeat_times =\
            self._get_mask_and_repeat(compute_shape, constant.DATA_TYPE_FP32)
        if dest.dtype == constant.DATA_TYPE_FP32:
            dst_rep_stride = constant.REPEAT_STRIDE_EIGHT
        else:
            dst_rep_stride = constant.REPEAT_STRIDE_FOUR

        if src.dtype == constant.DATA_TYPE_FP32:
            src_rep_stride = constant.REPEAT_STRIDE_EIGHT
        else:
            src_rep_stride = constant.REPEAT_STRIDE_FOUR

        if repeat_times == 1:
            self.tik_instance.vconv(front_mask, "", dest, src,
                                    constant.REPEAT_TIME_ONCE,
                                    constant.STRIDE_ONE,
                                    constant.STRIDE_ONE, dst_rep_stride,
                                    src_rep_stride)
        elif repeat_times <= MAX_REPEAT_NUM:
            self.tik_instance.vconv(front_mask, "", dest, src, repeat_times - 1,
                                    constant.STRIDE_ONE, constant.STRIDE_ONE,
                                    dst_rep_stride, src_rep_stride)
            vector_offset = (repeat_times - 1)*front_mask
            self.tik_instance.vconv(last_mask, "", dest[vector_offset],
                                    src[vector_offset],
                                    constant.REPEAT_TIME_ONCE,
                                    constant.STRIDE_ONE,
                                    constant.STRIDE_ONE,
                                    dst_rep_stride, src_rep_stride)
        else:
            rest_repeat_num = repeat_times
            count = 0
            while rest_repeat_num > MAX_REPEAT_NUM:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                self.tik_instance.vconv(front_mask, "", dest[vector_offset],
                                        src[vector_offset], MAX_REPEAT_NUM,
                                        constant.STRIDE_ONE,
                                        constant.STRIDE_ONE,
                                        dst_rep_stride, src_rep_stride)
                count = count + 1
                rest_repeat_num = rest_repeat_num - MAX_REPEAT_NUM
            if rest_repeat_num != 1:
                vector_offset = count*MAX_REPEAT_NUM*front_mask
                self.tik_instance.vconv(front_mask, "", dest[vector_offset],
                                        src[vector_offset], rest_repeat_num - 1,
                                        constant.STRIDE_ONE,
                                        constant.STRIDE_ONE,
                                        dst_rep_stride, src_rep_stride)
            vector_offset = (repeat_times - 1)*front_mask
            self.tik_instance.vconv(last_mask, "", dest[vector_offset],
                                    src[vector_offset],
                                    constant.REPEAT_TIME_ONCE,
                                    constant.STRIDE_ONE,
                                    constant.STRIDE_ONE,
                                    dst_rep_stride, src_rep_stride)

    def _vmul_func(self, dest, src0, src1, compute_shape, dest_offset=0,
                   src0_offset=0, src1_offset=0):
        self._double_vector_func(self.tik_instance.vmul, dest, src0, src1,
                                 compute_shape, dest_offset, src0_offset,
                                 src1_offset)

    def _vadd_func(self, dest, src0, src1, compute_shape, dest_offset=0,
                   src0_offset=0, src1_offset=0):
        self._double_vector_func(self.tik_instance.vadd, dest, src0, src1,
                                 compute_shape, dest_offset, src0_offset,
                                 src1_offset)

    def _vsub_func(self, dest, src0, src1, compute_shape):
        self._double_vector_func(self.tik_instance.vsub, dest, src0, src1,
                                 compute_shape)

    def _vmuls_func(self, dest, src0, scalar_val, compute_shape):
        self._vector_scalar_func(self.tik_instance.vmuls, dest, src0,
                                 scalar_val, compute_shape)

    def _vadds_func(self, dest, src0, scalar_val, compute_shape):
        self._vector_scalar_func(self.tik_instance.vadds, dest, src0,
                                 scalar_val, compute_shape)

    def _vln_func(self, dest, src, compute_shape):
        self._single_vector_func(self.tik_instance.vln, dest, src,
                                 compute_shape)

    def _vexp_func(self, dest, src, compute_shape):
        self._single_vector_func(self.tik_instance.vexp, dest, src,
                                 compute_shape)


# pylint: disable=locally-disabled,too-many-lines,too-many-instance-attributes
# pylint: disable=locally-disabled,simplifiable-if-statement,no-self-use
class LRNBase5HD(LRNBase):
    """
    Function: use to store LRN compute parameters
    """
    def __init__(self, x, depth_radius, bias, alpha, beta, kernel_name,
                 impl_mode):
        self.tik_instance = tik.Tik()
        super(LRNBase5HD, self).__init__(self.tik_instance)
        self.input_shape = x.get("shape")
        self.input_dtype = x.get("dtype").lower()
        self.impl_mode = impl_mode
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.kernel_name = kernel_name
        self.alpha_sqrt = math.sqrt(self.alpha)
        self.n_size = self.input_shape[0]
        self.c1_size = self.input_shape[1]
        self.c_size = self.input_shape[1] * self.input_shape[4]
        self.hw_size = self.input_shape[2] * self.input_shape[3]
        self.input_data_size = common_util.get_data_size(self.input_dtype)
        self.one_block_ele_num_input = \
            constant.BLOCK_SIZE // self.input_data_size
        # NC1HWC0
        self.hw_align_size = math.ceil(self.hw_size / 16) * 16
        self.depth_radius_align = math.ceil(self.depth_radius / 16) * 16
        self.alignment_standards = 16
        self.one_batch_size = self.c_size * self.hw_size
        self.ub_byte_size = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        self.ub_max_ele_num = self.ub_byte_size // self.input_data_size
        self.is_cloud = False
        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend910",):
            self.is_cloud = True
        self.alpha_sqrt_flag = False
        if (0 < self.alpha < 0.001) and (not self.is_cloud):
            self.alpha_sqrt_flag = True
        self.device_aicore_num = self.tik_instance.d_profiling.get_aicore_num()
        self.input_gm = self.tik_instance.Tensor(
            self.input_dtype,
            (self._get_shape_size(self.input_shape),),
            name="input_gm", scope=tik.scope_gm)
        # the output has the same dtype and shape with input
        self.output_gm = self.tik_instance.Tensor(
            self.input_dtype,
            (self._get_shape_size(self.input_shape),),
            name="output_gm", scope=tik.scope_gm)
        # five2four need align 256 ele
        self.ub_split_size = ((self.ub_max_ele_num / 3 / 2) // 256) * 256
        self.data_ub1 = []
        self.data_ub2 = []
        self.data_ub3 = []
        for i in range(2):
            self.data_ub1.append(self.tik_instance.Tensor(
                self.input_dtype, (self.ub_split_size,),
                name="data_input_ub", scope=tik.scope_ubuf))
            self.data_ub2.append(self.tik_instance.Tensor(
                self.input_dtype, (self.ub_split_size,),
                name="data_square_ub", scope=tik.scope_ubuf))
            self.data_ub3.append(self.tik_instance.Tensor(
                self.input_dtype, (self.ub_split_size,),
                name="data_output_ub", scope=tik.scope_ubuf))

    def do_tiling(self):
        """
        generate tiling

        Return
        ----------
        dict
            tiling info
        """
        tiling = {}
        tiling["batch_once"] = self.n_size // self.device_aicore_num \
            if self.n_size > self.device_aicore_num else 1
        tiling["batch_tail"] = self.n_size % tiling["batch_once"] \
            if self.n_size % tiling["batch_once"] != 0 else \
            tiling["batch_once"]
        tiling["batch_loop"] = math.ceil(self.n_size / tiling["batch_once"])
        if self.c_size * self.hw_align_size <= self.ub_split_size:
            # one batch each time
            tiling["type"] = "cut_n"
            tiling["hw_once"] = self.hw_size
            tiling["hw_tail"] = self.hw_size
            tiling["hw_loop"] = 1
            tiling["c_once"] = self.c_size
            tiling["c_tail"] = self.c_size
            tiling["c_loop"] = 1
        elif self.c_size * CUT_C_HW_SIZE <= self.ub_split_size:
            # cut hw
            tiling["type"] = "cut_hw"
            hw_once = self.ub_split_size // self.c_size
            hw_once = hw_once if hw_once < self.hw_size else self.hw_size
            tiling["hw_once"] = (hw_once // CUT_C_HW_SIZE) * CUT_C_HW_SIZE
            tiling["hw_tail"] = self.hw_size % tiling["hw_once"] \
                if self.hw_size % tiling["hw_once"] != 0 else tiling["hw_once"]
            tiling["hw_loop"] = math.ceil(self.hw_size / tiling["hw_once"])
            tiling["c_once"] = self.c_size
            tiling["c_tail"] = self.c_size
            tiling["c_loop"] = 1
        else:
            # cut c, get max c while hw == 128
            tiling["type"] = "cut_c"
            if self.hw_size > CUT_C_HW_SIZE:
                tiling["hw_once"] = CUT_C_HW_SIZE
                tiling["hw_tail"] = self.hw_size % CUT_C_HW_SIZE \
                    if self.hw_size % CUT_C_HW_SIZE != 0 else CUT_C_HW_SIZE
                tiling["hw_loop"] = math.ceil(self.hw_size / CUT_C_HW_SIZE)
                max_c_with_overlap = self.ub_split_size // CUT_C_HW_SIZE
            else:
                tiling["hw_once"] = self.hw_size
                tiling["hw_tail"] = self.hw_size
                tiling["hw_loop"] = 1
                max_c_with_overlap = self.ub_split_size // self.hw_align_size
            max_c_with_overlap = \
                (max_c_with_overlap // self.alignment_standards) * \
                self.alignment_standards
            tiling["c_once"] = max_c_with_overlap - self.depth_radius_align * 2
            tiling["c_tail"] = self.c_size % tiling["c_once"] \
                if self.c_size % tiling["c_once"] != 0 else tiling["c_once"]
            tiling["c_loop"] = math.ceil(self.c_size / tiling["c_once"])
        return tiling

    def do_cut_n(self, tiling):
        """
        cut c tiling
        """
        core_num = tiling["batch_loop"]
        with self.tik_instance.for_range(
                0, core_num,
                block_num=core_num) as block_idx:
            in_gm_offset = \
                block_idx * tiling["batch_once"] * self.one_batch_size
            out_gm_offset = \
                block_idx * tiling["batch_once"] * self.one_batch_size
            batch_loop_in_core = tiling["batch_once"]
            with self.tik_instance.if_scope(
                    block_idx == tiling["batch_loop"] - 1):
                batch_loop_in_core = tiling["batch_tail"]
            # gm -> data_ub1
            self.move_data(self.data_ub1[0],
                           self.input_gm[in_gm_offset],
                           self.input_dtype, self.one_batch_size)
            in_gm_offset += self.one_batch_size
            for batch_idx in range(batch_loop_in_core):
                # gm -> data_ub1
                if batch_idx < batch_loop_in_core - 1:
                    self.move_data(self.data_ub1[(batch_idx + 1) % 2],
                                   self.input_gm[in_gm_offset],
                                   self.input_dtype, self.one_batch_size)
                    in_gm_offset += self.one_batch_size
                out_ub = self.do_operation(self.data_ub1[batch_idx % 2],
                                           self.data_ub2[batch_idx % 2],
                                           self.data_ub3[batch_idx % 2],
                                           tiling["hw_once"],
                                           tiling["c_once"])
                # data_ub2 -> gm
                self.move_data(self.output_gm[out_gm_offset], out_ub,
                               self.input_dtype, self.one_batch_size)
                out_gm_offset += self.one_batch_size

    def do_cut_hw(self, tiling, batch_gm_offset):
        buffer_idx = 0
        in_gm_offset = batch_gm_offset
        out_gm_offset = batch_gm_offset
        self.move_data_stride_in(self.data_ub1[buffer_idx % 2],
                                 self.input_gm,
                                 tiling["hw_once"], self.c_size,
                                 in_gm_offset)
        in_gm_offset += tiling["hw_once"] * C0_SIZE
        for hw_idx in range(tiling["hw_loop"]):
            if hw_idx < tiling["hw_loop"] - 1:
                hw_inner = tiling["hw_once"] \
                    if hw_idx < tiling["hw_loop"] - 2 \
                    else tiling["hw_tail"]
                self.move_data_stride_in(
                    self.data_ub1[(buffer_idx+1) % 2],
                    self.input_gm,
                    hw_inner, self.c_size,
                    in_gm_offset)
                in_gm_offset += hw_inner * C0_SIZE
            hw_inner = tiling["hw_once"] \
                if hw_idx < tiling["hw_loop"] - 1 \
                else tiling["hw_tail"]
            out_ub = self.do_operation(self.data_ub1[buffer_idx % 2],
                                       self.data_ub2[buffer_idx % 2],
                                       self.data_ub3[buffer_idx % 2],
                                       hw_inner, self.c_size)
            self.move_data_stride_out(self.output_gm, out_ub, hw_inner,
                                      self.c_size, out_gm_offset)
            out_gm_offset += hw_inner * C0_SIZE
            buffer_idx += 1

    def do_cut_c(self, tiling, batch_gm_offset):
        buffer_idx = 0
        hw_gm_offset = batch_gm_offset
        for hw_idx in range(tiling["hw_loop"]):
            hw_inner = tiling["hw_once"] \
                if hw_idx < tiling["hw_loop"] - 1 else tiling["hw_tail"]
            in_gm_offset = hw_gm_offset
            out_gm_offset = hw_gm_offset
            c_top = 0
            c_bottom = 0
            self.move_data_stride_in(self.data_ub1[buffer_idx % 2],
                                     self.input_gm,
                                     hw_inner,
                                     tiling["c_once"] +
                                     self.depth_radius_align,
                                     in_gm_offset)
            in_gm_offset += \
                (tiling["c_once"] - self.depth_radius_align) * self.hw_size

            c_bottom = self.depth_radius_align
            for c_idx in range(tiling["c_loop"]):
                if c_idx < tiling["c_loop"] - 1:
                    c_inner = tiling["c_once"] + 2 * self.depth_radius_align
                    if c_idx == tiling["c_loop"] - 2:
                        c_inner = tiling["c_tail"] + self.depth_radius_align
                    self.move_data_stride_in(
                        self.data_ub1[(buffer_idx+1) % 2],
                        self.input_gm,
                        hw_inner, c_inner, in_gm_offset)
                    in_gm_offset += tiling["c_once"] * self.hw_size
                else:
                    c_bottom = 0
                c_real = tiling["c_once"] \
                    if c_idx < tiling["c_loop"] - 1 else tiling["c_tail"]
                out_ub = self.do_operation(self.data_ub1[buffer_idx % 2],
                                           self.data_ub2[buffer_idx % 2],
                                           self.data_ub3[buffer_idx % 2],
                                           hw_inner, c_real, c_top, c_bottom)
                self.move_data_stride_out(self.output_gm, out_ub, hw_inner,
                                          c_real, out_gm_offset)
                out_gm_offset += c_real * self.hw_size
                buffer_idx += 1
                c_top = self.depth_radius_align
            hw_gm_offset += tiling["hw_once"] * C0_SIZE
        return buffer_idx

    def move_data_stride_in(self, dest, src, hw_size, c_size, in_gm_offset):
        nburst = int(c_size // 16)
        burst = int(hw_size)
        src_stride = int(self.hw_size - hw_size)
        if src_stride < MAX_HW_NUM:
            self.tik_instance.data_move(dest, src[in_gm_offset], constant.SID,
                                        nburst, burst,
                                        src_stride, constant.STRIDE_ZERO)
        else:
            with self.tik_instance.for_range(0, nburst) as idx:
                self.tik_instance.data_move(
                    dest[idx * burst * C0_SIZE],
                    src[in_gm_offset +
                        idx * self.hw_size * C0_SIZE],
                    constant.SID, constant.DEFAULT_NBURST, burst,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def move_data_stride_out(self, dest, src, hw_size, c_size, out_gm_offset):
        nburst = int(c_size // 16)
        burst = int(hw_size)
        dst_stride = int(self.hw_size - hw_size)
        if dst_stride < MAX_HW_NUM:
            self.tik_instance.data_move(dest[out_gm_offset], src, constant.SID,
                                        nburst, burst,
                                        constant.STRIDE_ZERO, dst_stride)
        else:
            with self.tik_instance.for_range(0, nburst) as idx:
                self.tik_instance.data_move(
                    dest[out_gm_offset +
                         idx * self.hw_size * C0_SIZE],
                    src[idx * burst * C0_SIZE],
                    constant.SID, constant.DEFAULT_NBURST, burst,
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def do_operation(self, data_ub1, data_ub2, data_ub3, hw_size, c_size,
                     c_top=0, c_bottom=0):
        hw_align_size = math.ceil(hw_size / 16) * 16
        compute_shape = (hw_align_size, c_size + c_top + c_bottom)

        # nc1hwc0 -> nchw
        self.five2four(data_ub2, data_ub1, hw_size, c_size + c_top + c_bottom)

        # x^2
        self.do_squared(data_ub2, compute_shape)

        # sum cross c-axis
        self.do_depth_operation(data_ub3, data_ub2, int(hw_align_size),
                                int(c_size), c_top, c_bottom)

        compute_shape = (hw_align_size, c_size)
        # *alpha
        if not self.alpha_sqrt_flag:
            self._vmuls_func(data_ub3, data_ub3, self.alpha, compute_shape)

        # + k
        self._vadds_func(data_ub3, data_ub3, self.bias, compute_shape)

        # ln
        self._vln_func(data_ub3, data_ub3, compute_shape)

        # *beta
        self._vmuls_func(data_ub3, data_ub3, self.beta*(-1), compute_shape)

        # exp
        self._vexp_func(data_ub3, data_ub3, compute_shape)

        # nchw -> nc1hwc0
        self.four2five(data_ub2, data_ub3, hw_size, c_size)

        # x * res
        self._vmul_func(data_ub2, data_ub1, data_ub2, compute_shape,
                        src0_offset=int(c_top * hw_size))

        return data_ub2

    def do_depth_operation(self, dest, src, hw_size, c_size, c_top, c_bottom):
        zero_scalar = self.tik_instance.Scalar(dtype=self.input_dtype,
                                               name="zero_scalar",
                                               init_value=0.0)
        self._vector_dup_func(dest, zero_scalar, dest.shape)
        if c_top == 0 and c_bottom == 0:
            self.do_depth_operation_all(dest, src, int(hw_size), int(c_size))
        elif c_top == 0 and c_bottom != 0:
            self.do_depth_operation_top(dest, src, int(hw_size), int(c_size))
        elif c_top != 0 and c_bottom != 0:
            self.do_depth_operation_mid(dest, src, int(hw_size), int(c_size),
                                        c_top)
        elif c_top != 0 and c_bottom == 0:
            self.do_depth_operation_bottom(dest, src, int(hw_size),
                                           int(c_size), c_top)

    def do_depth_operation_bottom(self, dest, src, hw_size, c_size, c_top):
        # add next
        first_flag = True
        top_offset = c_top * hw_size
        src_last_offset = hw_size * (c_size + c_top - 1)
        dst_last_offset = hw_size * (c_size - 1)
        src_ub_offset = top_offset + hw_size
        src1_ub_offset = top_offset - hw_size
        down_compute_size = hw_size * (c_size - 1)
        up_compute_size = hw_size * c_size
        for radius_idx in range(self.depth_radius):
            if first_flag:
                self._vadd_func(dest, src, src, (down_compute_size,),
                                src0_offset=top_offset,
                                src1_offset=src_ub_offset)
                # fill last line
                self._vadd_func(dest, dest, src, (hw_size,), dst_last_offset,
                                dst_last_offset, src_last_offset)
                first_flag = False
            else:
                self._vadd_func(dest, dest, src, (down_compute_size,),
                                src1_offset=src_ub_offset)
            # add pre c
            self._vadd_func(dest, dest, src, (up_compute_size,),
                            src1_offset=src1_ub_offset)
            down_compute_size -= hw_size
            src_ub_offset += hw_size
            src1_ub_offset -= hw_size

    def do_depth_operation_mid(self, dest, src, hw_size, c_size, c_top):
        first_flag = True
        top_offset = c_top * hw_size
        src_ub_offset = hw_size + top_offset
        src1_ub_offset = top_offset - hw_size
        compute_size = hw_size * c_size
        for radius_idx in range(self.depth_radius):
            if first_flag:
                self._vadd_func(dest, src, src, (compute_size,),
                                src0_offset=top_offset,
                                src1_offset=src_ub_offset)
                first_flag = False
            else:
                self._vadd_func(dest, dest, src, (compute_size,),
                                src1_offset=src_ub_offset)
            # add pre c
            self._vadd_func(dest, dest, src, (compute_size,),
                            src1_offset=src1_ub_offset)
            src_ub_offset += hw_size
            src1_ub_offset -= hw_size

    def do_depth_operation_top(self, dest, src, hw_size, c_size):
        first_flag = True
        src_ub_offset = hw_size
        dst_ub_offset = hw_size

        down_compute_size = hw_size * c_size
        up_compute_size = hw_size * (c_size - 1)
        for radius_idx in range(self.depth_radius):
            # add down c
            if first_flag:
                self._vadd_func(dest, src, src, (down_compute_size,),
                                src1_offset=src_ub_offset)
                first_flag = False
            else:
                self._vadd_func(dest, dest, src, (down_compute_size,),
                                src1_offset=src_ub_offset)
            # add up c
            self._vadd_func(dest, dest, src, (up_compute_size,),
                            dest_offset=dst_ub_offset,
                            src0_offset=dst_ub_offset)

            src_ub_offset += hw_size
            dst_ub_offset += hw_size
            up_compute_size -= hw_size

    def do_depth_operation_all(self, dest, src, hw_size, c_size):
        # add next
        src_ub_offset = hw_size
        first_flag = True
        last_offset = hw_size * (c_size - 1)
        dst_ub_offset = hw_size
        compute_size = hw_size * (c_size - 1)
        for radius_idx in range(self.depth_radius):
            if first_flag:
                self._vadd_func(dest, src, src,
                                (compute_size,), src1_offset=src_ub_offset)
                # fill last line
                self._vadd_func(dest, dest, src, (hw_size,),
                                last_offset, last_offset, last_offset)
                first_flag = False
            else:
                self._vadd_func(dest, dest, src, (compute_size,),
                                src1_offset=src_ub_offset)
            # add pre c
            self._vadd_func(dest, dest, src, (compute_size,),
                            dst_ub_offset, dst_ub_offset)
            src_ub_offset += hw_size
            dst_ub_offset += hw_size
            compute_size -= hw_size

    def do_squared(self, data_ub, compute_shape):
        if self.alpha_sqrt_flag:
            # do square operation
            self._vmuls_func(data_ub, data_ub, self.alpha_sqrt, compute_shape)
            self._vmul_func(data_ub, data_ub, data_ub, compute_shape)
        else:
            self._vmul_func(data_ub, data_ub, data_ub, compute_shape)

    def five2four(self, dest, src, hw_size, c_size):
        hw_size = int(hw_size)
        c_loop = int(c_size // 16)
        repeat = math.ceil(hw_size / 16)
        hw_align_size = int(math.ceil(hw_size / 16) * 16)
        # repeat_times 1
        #     real_src[0]/dst[0] address = src/dst_list + src/dst_rep_strie
        # repeat_times > 1
        #     real_src[0]/dst[0] address = src/dst_list
        src_stride = 16 if repeat > 1 else 0
        dst_stride = 1 if repeat > 1 else 0

        with self.tik_instance.for_range(0, c_loop) as c_idx:
            src_list = [src[16 * hw_size * c_idx + 16 * i] for i in range(16)]
            dst_list = \
                [dest[16 * hw_align_size * c_idx + hw_align_size * i]
                 for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                        repeat, dst_stride, src_stride)

    def four2five(self, dest, src, hw_size, c_size):
        hw_size = int(hw_size)
        c_loop = int(c_size // 16)
        repeat = math.ceil(hw_size / 16)
        hw_align_size = int(math.ceil(hw_size / 16) * 16)
        src_stride = 1 if repeat > 1 else 0
        dst_stride = 16 if repeat > 1 else 0
        with self.tik_instance.for_range(0, c_loop) as c_idx:
            src_list = \
                [src[16 * hw_align_size * c_idx + hw_align_size * i]
                 for i in range(16)]
            dst_list = [dest[16 * hw_size * c_idx + 16 * i] for i in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list,
                                        repeat, dst_stride, src_stride)

    def move_data(self, dest, src, data_type, copy_size):
        byte_num_one = common_util.get_data_size(data_type)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = copy_size // one_block_ele_num
        self.tik_instance.data_move(dest, src, constant.SID,
                                    constant.DEFAULT_NBURST, block_num,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def tik_instance_function(self):
        """
        do the LRN operation when H*W is 32B align

        Parameters
        ----------

        Returns
        -------
        None
        """
        tiling = self.do_tiling()
        if tiling["type"] == "cut_n":
            self.do_cut_n(tiling)
        else:
            with self.tik_instance.for_range(
                    0, self.n_size, block_num=self.n_size) as block_idx:
                batch_gm_offset = block_idx * self.one_batch_size
                if tiling["type"] == "cut_hw":
                    self.do_cut_hw(tiling, batch_gm_offset)
                else:
                    self.do_cut_c(tiling, batch_gm_offset)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.output_gm])
        return self.tik_instance


# pylint: disable=locally-disabled,too-many-lines,too-many-instance-attributes
# pylint: disable=locally-disabled,simplifiable-if-statement,no-self-use
class LRNBase4HD(LRNBase):
    """
    Function: use to store LRN compute parameters
    """
    def __init__(self, x, depth_radius, bias, alpha, beta, kernel_name,
                 impl_mode):
        self.tik_instance = tik.Tik()
        super(LRNBase4HD, self).__init__(self.tik_instance)
        self.input_shape = x.get("shape")
        self.dtype_real_in_out = x.get("dtype").lower()
        self.impl_mode = impl_mode
        self.input_dtype = self._get_compute_dtype()
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.kernel_name = kernel_name
        self.alpha_sqrt = math.sqrt(self.alpha)
        self.input_data_size = common_util.get_data_size(self.input_dtype)
        self.one_block_ele_num_input =\
            constant.BLOCK_SIZE // self.input_data_size
        self.one_batch_size =\
            self.input_shape[1]*self.input_shape[2]*self.input_shape[3]
        self.one_column_size = self.input_shape[2]*self.input_shape[3]
        self.N = self.input_shape[0]
        self.C = self.input_shape[1]
        self.H = self.input_shape[2]
        self.W = self.input_shape[3]
        self.core_num = self._get_target_core_num()
        self.batch_num_each_core, self.threshold_multi_core,\
        self.batch_num_front_core = self._get_multi_cores_param()
        self.ub_byte_size =\
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
        self.ub_max_ele_num = self.ub_byte_size // self.input_data_size
        self.pre_square_sum_ub = None
        self.right_square_sum_ub = None
        self.data_cast_ub = None
        self.data_output_ub = None
        self.data_square_ub = None
        self.data_input_ub = None
        self.alignment_standards = 16
        self.is_cloud = False
        self.one_repeat_ele_num =\
            constant.REPEAT_STRIDE_EIGHT*self.one_block_ele_num_input
        self.cast_ub_data_size =\
            common_util.get_data_size(self.dtype_real_in_out)
        self.one_block_ele_num_cast =\
            constant.BLOCK_SIZE // self.cast_ub_data_size

        if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend910",):
            self.is_cloud = True

        self.is_alpha_sqrt_flag = self._check_alpha_sqrt()
        self.device_aicore_num = self.tik_instance.d_profiling.get_aicore_num()

        self.input_gm =\
            self.tik_instance.Tensor(self.dtype_real_in_out,
                                     (self._get_shape_size(self.input_shape),),
                                     name="input_gm", scope=tik.scope_gm)
        # the output has the same dtype and shape with input
        self.output_gm =\
            self.tik_instance.Tensor(self.dtype_real_in_out,
                                     (self._get_shape_size(self.input_shape),),
                                     name="output_gm", scope=tik.scope_gm)

    def _get_compute_dtype(self):
        # check whether the platform is mini or not
        is_mini_flag = \
            (not tbe_platform.cce_conf.api_check_support("tik.vln",
                                                         "float32") or
             tbe_platform.cce_conf.api_check_support("tik.vbi", "float16"))
        # only when mini platform and input dtype is float16
        # support impl_mode parameter
        if is_mini_flag and self.dtype_real_in_out == constant.DATA_TYPE_FP16:
            vln_support_fp32_flag = \
                (tbe_platform.cce_conf.api_check_support("tik.vln", "float32")
                 and self.impl_mode == "high_precision")
        else:
            vln_support_fp32_flag = \
                tbe_platform.cce_conf.api_check_support("tik.vln", "float32")

        compute_dtype = self.dtype_real_in_out
        if vln_support_fp32_flag and\
                (self.dtype_real_in_out == constant.DATA_TYPE_FP16):
            compute_dtype = constant.DATA_TYPE_FP32
        if (not vln_support_fp32_flag) and\
                (self.dtype_real_in_out == constant.DATA_TYPE_FP32):
            compute_dtype = constant.DATA_TYPE_FP16

        return compute_dtype

    def _check_alpha_sqrt(self):
        alpha_sqrt_flag = False
        if (0 < self.alpha < 0.001) and (not self.is_cloud):
            alpha_sqrt_flag = True

        return alpha_sqrt_flag

    def _get_target_core_num(self):
        if self.N < MAX_CORE_NUMBER:
            return self.N

        return MAX_CORE_NUMBER

    def _get_multi_cores_param(self):
        if self.core_num < MAX_CORE_NUMBER:
            batch_num_each_core = 1
            threshold_multi_core = 0
            batch_num_front_core = 1
        else:
            batch_num_each_core = self.N // MAX_CORE_NUMBER
            threshold_multi_core = self.N % MAX_CORE_NUMBER
            batch_num_front_core = batch_num_each_core + 1

        return batch_num_each_core, threshold_multi_core, batch_num_front_core

    def _allocate_cast_ub(self):
        if self.dtype_real_in_out == constant.DATA_TYPE_FP16 and\
                self.input_dtype == constant.DATA_TYPE_FP32:
            byte_size_cast_ub = self.ub_byte_size // 7
        else:
            byte_size_cast_ub = (self.ub_byte_size // 5)*2

        ele_cast_ub = byte_size_cast_ub // self.cast_ub_data_size
        ele_cast_ub = self._get_align_size(ele_cast_ub)
        byte_size_remaining_ub =\
            self.ub_byte_size - ele_cast_ub*self.cast_ub_data_size
        self.ub_max_ele_num = byte_size_remaining_ub // self.input_data_size

        self.data_cast_ub =\
            self.tik_instance.Tensor(self.dtype_real_in_out, (ele_cast_ub,),
                                     name="data_cast_ub", scope=tik.scope_ubuf)

    def _allocate_ub_buffer(self):
        self.data_input_ub =\
            self.tik_instance.Tensor(self.input_dtype,
                                     (self._get_shape_size(self.input_shape),),
                                     name="data_input_ub",
                                     scope=tik.scope_ubuf)
        self.data_square_ub =\
            self.tik_instance.Tensor(self.input_dtype,
                                     (self._get_shape_size(self.input_shape),),
                                     name="data_square_ub",
                                     scope=tik.scope_ubuf)
        self.data_output_ub =\
            self.tik_instance.Tensor(self.input_dtype,
                                     (self._get_shape_size(self.input_shape),),
                                     name="data_output_ub",
                                     scope=tik.scope_ubuf)

    def _do_eltwise_operation_pre(self):
        if self.is_alpha_sqrt_flag:
            self._vmuls_func(self.data_square_ub, self.data_input_ub,
                             self.alpha_sqrt, self.data_input_ub.shape)

            # do square operation
            self._vmul_func(self.data_square_ub, self.data_square_ub,
                            self.data_square_ub, self.data_square_ub.shape)
        else:
            # do square operation
            self._vmul_func(self.data_square_ub, self.data_input_ub,
                            self.data_input_ub, self.data_input_ub.shape)

    def _do_eltwise_operation(self):
        if not self.is_alpha_sqrt_flag:
            # do vmuls operation
            self._vmuls_func(self.data_output_ub, self.data_output_ub, self.alpha,
                             self.data_output_ub.shape)

        # do vadds operation, get the tmp value
        self._vadds_func(self.data_output_ub, self.data_output_ub, self.bias,
                         self.data_output_ub.shape)

        # do the log operation
        self._vln_func(self.data_output_ub, self.data_output_ub,
                       self.data_output_ub.shape)

        # vmuls negative beta
        self._vmuls_func(self.data_output_ub, self.data_output_ub,
                         self.beta*(-1), self.data_output_ub.shape)

        # do the exp operation
        self._vexp_func(self.data_output_ub, self.data_output_ub,
                        self.data_output_ub.shape)

        # vmul input with output
        self._vmul_func(self.data_output_ub, self.data_output_ub,
                        self.data_input_ub, self.data_output_ub.shape)

    def _cut_batch_axis(self, ub_buffer_one_batch):
        # check how many batches UB can store once
        batch_num_ub_once = self.ub_max_ele_num // ub_buffer_one_batch
        loop_num = self.N // batch_num_ub_once
        batch_num_last = batch_num_ub_once
        if self.N % batch_num_ub_once != 0:
            batch_num_last = self.N - loop_num*batch_num_ub_once
            loop_num = loop_num + 1

        if loop_num == 1:
            batch_num_ub_once = self.N

        return loop_num, batch_num_ub_once, batch_num_last

    def _cut_h_w_axis(self):
        buffer_single_max = self.ub_max_ele_num // 3
        buffer_single_max =\
            (buffer_single_max // self.alignment_standards) *\
            self.alignment_standards

        h_w_len = buffer_single_max // self.C
        h_w_len = (h_w_len // self.alignment_standards) *\
                  self.alignment_standards

        h_w_size_each_loop = h_w_len
        loop_num = (self.input_shape[2]*self.input_shape[3]) // h_w_size_each_loop
        h_w_size_last_loop = h_w_size_each_loop
        if (self.input_shape[2]*self.input_shape[3]) % h_w_size_each_loop != 0:
            h_w_size_last_loop = self.input_shape[2]*self.input_shape[3] - \
                                 loop_num*h_w_size_each_loop
            loop_num = loop_num + 1

        if loop_num == 1:
            h_w_size_each_loop = h_w_size_last_loop

        return loop_num, h_w_size_each_loop, h_w_size_last_loop

    def _cut_hw_c_axis_hw(self):
        # each input_ub, square_ub, output_ub has depth_radius+1 column
        # pre_square_sum_ub, right_square_sum_ub has one column
        column_num = (self.depth_radius + 1)*3 + 2
        h_w_len = self.ub_max_ele_num // column_num
        h_w_len = (h_w_len // self.alignment_standards) *\
                  self.alignment_standards

        h_w_size_each_loop = h_w_len
        loop_num = self.one_column_size // h_w_size_each_loop
        h_w_size_last_loop = h_w_size_each_loop
        if self.one_column_size % h_w_size_each_loop != 0:
            h_w_size_last_loop = self.one_column_size - \
                                 loop_num*h_w_size_each_loop
            loop_num = loop_num + 1

        return loop_num, h_w_size_each_loop, h_w_size_last_loop

    def _cut_hw_c_axis_hw_not_align(self):
        column_num = (self.depth_radius + 1)*3 + 2
        h_w_len = self.ub_max_ele_num // column_num
        h_w_len = (h_w_len // self.alignment_standards) *\
                  self.alignment_standards

        h_w_size_each_loop = h_w_len
        if self.one_column_size < h_w_size_each_loop:
            h_w_size_each_loop =\
                (self.one_column_size // self.alignment_standards) *\
                self.alignment_standards
            h_w_size_first_loop = self.one_column_size - h_w_size_each_loop
            loop_num = 2
        else:
            loop_num = self.one_column_size // h_w_size_each_loop
            h_w_size_first_loop = h_w_size_each_loop

            if self.one_column_size % h_w_size_each_loop != 0:
                h_w_size_first_loop = self.one_column_size - \
                                      loop_num*h_w_size_each_loop
                loop_num = loop_num + 1

        return loop_num, h_w_size_each_loop, h_w_size_first_loop

    def _cut_hw_c_axis_c(self):
        c_size_each_loop = self.depth_radius + 1
        loop_num = self.C // c_size_each_loop
        c_size_last_loop = c_size_each_loop
        if self.C % c_size_each_loop != 0:
            c_size_last_loop = self.C - loop_num*c_size_each_loop
            loop_num = loop_num + 1

        return loop_num, c_size_each_loop, c_size_last_loop

    def _cut_h_w_axis_not_align(self):
        buffer_single_max = self.ub_max_ele_num // 3
        buffer_single_max = \
            (buffer_single_max // self.alignment_standards) *\
            self.alignment_standards

        h_w_len = buffer_single_max // self.C
        h_w_len = (h_w_len // self.alignment_standards) *\
                  self.alignment_standards

        h_w_size_each_loop = h_w_len
        loop_num = (self.H*self.W) // h_w_size_each_loop
        h_w_size_first_loop = h_w_size_each_loop

        if (self.H*self.W) % h_w_size_each_loop != 0:
            h_w_size_first_loop = self.H*self.W - \
                                  loop_num*h_w_size_each_loop
            loop_num = loop_num + 1

        return loop_num, h_w_size_each_loop, h_w_size_first_loop

    def _cut_c_axis(self):
        buffer_single_max =\
            (self.ub_max_ele_num -
             self._get_align_size(self.one_column_size)*2) // 3
        buffer_single_max = \
            (buffer_single_max // self.alignment_standards) *\
            self.alignment_standards

        c_len = buffer_single_max // self._get_align_size(self.one_column_size)
        c_size_each_loop = c_len
        loop_num = self.C // c_size_each_loop
        c_size_last_loop = c_size_each_loop
        if self.C % c_size_each_loop != 0:
            c_size_last_loop = self.C - loop_num*c_size_each_loop
            loop_num = loop_num + 1

        return loop_num, c_size_each_loop, c_size_last_loop

    def _get_align_size(self, orig_size):
        if orig_size % self.alignment_standards == 0:
            align_size = orig_size
        else:
            align_size = (orig_size // self.alignment_standards + 1) *\
                         self.alignment_standards

        return align_size

    def _do_tiling(self):
        n_cut_flag = False
        c_cut_flag = False
        hw_cut_flag = False
        hw_c_all_cut_flag = False

        # cut batch axis
        one_buffer_single_batch = self._get_align_size(self.one_batch_size)
        total_ub_single_batch = 3*one_buffer_single_batch

        if total_ub_single_batch < self.ub_max_ele_num:
            # only cut batch axis
            n_cut_flag = True
        else:
            one_buffer_c_axis = self.alignment_standards*self.C
            total_ub_c_axis = 3*one_buffer_c_axis
            if total_ub_c_axis < self.ub_max_ele_num:
                # cut hw axis
                hw_cut_flag = True
            else:
                one_buffer_single_hw = self._get_align_size(self.H*self.W)
                total_ub_single_hw = 8*one_buffer_single_hw
                if total_ub_single_hw < self.ub_max_ele_num:
                    # only cut c axis
                    c_cut_flag = True
                else:
                    # hw axis and c axis all need cut
                    hw_c_all_cut_flag = True

        return n_cut_flag, c_cut_flag, hw_cut_flag, hw_c_all_cut_flag

    def _do_tiling_not_align(self):
        n_cut_flag = False
        h_w_cut_flag = False
        hw_c_all_cut_flag = False

        one_column_align = self._get_align_size(self.one_column_size)
        one_buffer_single_batch_align = self.C*one_column_align
        total_ub_single_batch_align = 3*one_buffer_single_batch_align

        if total_ub_single_batch_align < self.ub_max_ele_num:
            n_cut_flag = True
        else:
            one_buffer_c_axis = self.alignment_standards*self.C
            total_ub_c_axis = 3*one_buffer_c_axis
            if total_ub_c_axis >= self.ub_max_ele_num:
                hw_c_all_cut_flag = True
            else:
                # cut h_w axis
                h_w_cut_flag = True

        return n_cut_flag, h_w_cut_flag, hw_c_all_cut_flag

    def _do_tiling_not_align_hw_little(self):
        n_cut_flag = False
        c_cut_flag = False

        one_column_align = self._get_align_size(self.one_column_size)
        one_buffer_single_batch_align = self.C*one_column_align
        total_ub_single_batch_align = 3*one_buffer_single_batch_align

        if total_ub_single_batch_align < self.ub_max_ele_num:
            n_cut_flag = True
        else:
            c_cut_flag = True

        return n_cut_flag, c_cut_flag

    def _data_move_default(self, dest, src, data_type, copy_shape):
        mte2_num = self._get_shape_size(copy_shape)
        mte2_num = self._get_align_size(mte2_num)
        byte_num_one = common_util.get_data_size(data_type)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = mte2_num // one_block_ele_num

        self.tik_instance.data_move(dest, src, constant.SID,
                                    constant.DEFAULT_NBURST, block_num,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    def _gm_2_ub_cut_h_w(self, dest, data_type, mte2_num, offset_gm):
        byte_num_one = common_util.get_data_size(data_type)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = mte2_num // one_block_ele_num
        hw_size = self.H * self.W

        if self.C > MAX_REPEAT_NUM or \
                hw_size % self.alignment_standards or \
                hw_size > MAX_HW_NUM:
            with self.tik_instance.for_range(0, self.C) as c_idx:
                self.tik_instance.data_move(
                    dest[c_idx*mte2_num],
                    self.input_gm[offset_gm + c_idx*self.H*self.W],
                    constant.SID,
                    constant.DEFAULT_NBURST, block_num,
                    constant.STRIDE_ZERO,
                    constant.STRIDE_ZERO)
        else:
            self.tik_instance.data_move(
                dest,
                self.input_gm[offset_gm],
                constant.SID,
                self.C,
                block_num,
                hw_size // self.alignment_standards - block_num,
                constant.STRIDE_ZERO)

    def _ub_2_gm_cut_h_w(self, src, data_type, mte2_num, offset_gm):
        byte_num_one = common_util.get_data_size(data_type)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = mte2_num // one_block_ele_num
        hw_size = self.H * self.W

        if self.C > MAX_REPEAT_NUM or \
                hw_size % self.alignment_standards or \
                hw_size > MAX_HW_NUM:
            with self.tik_instance.for_range(0, self.C) as c_idx:
                self.tik_instance.data_move(
                    self.output_gm[offset_gm + c_idx*self.H*self.W],
                    src[c_idx*mte2_num],
                    constant.SID, constant.DEFAULT_NBURST,
                    block_num, constant.STRIDE_ZERO,
                    constant.STRIDE_ZERO)
        else:
            self.tik_instance.data_move(
                self.output_gm[offset_gm],
                src,
                constant.SID,
                self.C,
                block_num,
                constant.STRIDE_ZERO,
                hw_size // self.alignment_standards - block_num)

    def _gm_2_ub_cut_batch_not_align(self, dest, data_type,
                                     one_column_size_aligned,
                                     one_batch_size_aligned, offset_gm):
        mte2_num = one_column_size_aligned
        byte_num_one = common_util.get_data_size(data_type)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = mte2_num // one_block_ele_num

        with self.tik_instance.for_range(0, self.N) as batch_idx:
            with self.tik_instance.for_range(0, self.C) as c_idx:
                self.tik_instance.data_move(dest[batch_idx *
                                                 one_batch_size_aligned +
                                                 c_idx*one_column_size_aligned],
                                            self.input_gm[offset_gm +
                                                          batch_idx *
                                                          self.one_batch_size +
                                                          c_idx *
                                                          self.one_column_size],
                                            constant.SID,
                                            constant.DEFAULT_NBURST, block_num,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)

    def _ub_2_gm_cut_batch_not_align(self, src, data_type,
                                     one_column_size_aligned,
                                     one_batch_size_aligned, offset_gm):
        mte2_num = one_column_size_aligned
        byte_num_one = common_util.get_data_size(data_type)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = mte2_num // one_block_ele_num

        with self.tik_instance.for_range(0, self.N) as batch_idx:
            with self.tik_instance.for_range(0, self.C) as c_idx:
                self.tik_instance.data_move(self.output_gm[offset_gm +
                                                           batch_idx *
                                                           self.one_batch_size +
                                                           c_idx *
                                                           self.one_column_size],
                                            src[batch_idx *
                                                one_batch_size_aligned +
                                                c_idx*one_column_size_aligned],
                                            constant.SID,
                                            constant.DEFAULT_NBURST, block_num,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)

    def _gm_2_ub_hw_c_all_cut_c(self, dest, data_type, mte2_num, offset_gm,
                                c_size_current_loop):
        byte_num_one = common_util.get_data_size(data_type)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = mte2_num // one_block_ele_num

        with self.tik_instance.for_range(0, c_size_current_loop) as c_idx:
            self.tik_instance.data_move(dest[c_idx*mte2_num],
                                        self.input_gm[offset_gm +
                                                      c_idx*self.H*self.W],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, block_num,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)

    def _ub_2_gm_hw_c_all_cut_c(self, src, data_type, mte2_num, offset_gm,
                                c_size_current_loop):
        byte_num_one = common_util.get_data_size(data_type)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = mte2_num // one_block_ele_num

        with self.tik_instance.for_range(0, c_size_current_loop) as c_idx:
            self.tik_instance.data_move(self.output_gm[offset_gm +
                                                       c_idx*self.H*self.W],
                                        src[c_idx*mte2_num], constant.SID,
                                        constant.DEFAULT_NBURST, block_num,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)

    def _gm_2_ub_hw_c_all_cut_hw(self, dest, data_type, mte2_num, offset_gm):
        mte2_num = self._get_align_size(mte2_num)
        byte_num_one = common_util.get_data_size(data_type)
        one_block_ele_num = constant.BLOCK_SIZE // byte_num_one
        block_num = mte2_num // one_block_ele_num

        with self.tik_instance.for_range(0, self.depth_radius+1) as c_idx:
            self.tik_instance.data_move(dest[c_idx*mte2_num],
                                        self.input_gm[offset_gm +
                                                      c_idx*self.H*self.W],
                                        constant.SID,
                                        constant.DEFAULT_NBURST, block_num,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)

    def _gm_2_ub_c_cut_branch_hw_little_not_align(self, dest, data_type,
                                                  mte2_num, offset_gm):
        self._gm_2_ub_hw_c_all_cut_hw(dest, data_type, mte2_num, offset_gm)

    def _do_operation_each_loop_cut_batch(self, offset_gm):
        if self.dtype_real_in_out != self.input_dtype:
            # copy gm to cast_ub
            self._data_move_default(self.data_cast_ub, self.input_gm[offset_gm],
                                    self.dtype_real_in_out, self.input_shape)
            mte2_num = self._get_shape_size(self.input_shape)
            mte2_num = self._get_align_size(mte2_num)

            # vconv from cast_ub to input_ub
            self._vconv_func(self.data_input_ub, self.data_cast_ub,
                             (mte2_num,))
        else:
            # copy gm to input_ub
            self._data_move_default(self.data_input_ub,
                                    self.input_gm[offset_gm], self.input_dtype,
                                    self.input_shape)

        self._do_eltwise_operation_pre()

        # do dup zero for output_ub
        zero_scalar = self.tik_instance.Scalar(dtype=self.input_dtype,
                                               name="zero_scalar",
                                               init_value=0.0)
        self._vector_dup_func(self.data_output_ub, zero_scalar,
                              self.data_output_ub.shape)

        # do add operation in  col
        with self.tik_instance.for_range(0, self.N) as batch_idx:
            # each batch to (C, H*W)
            with self.tik_instance.for_range(0, self.C) as c_idx:
                left_val = \
                    self.tik_instance.Scalar(dtype="int64", name="left_val",
                                             init_value=c_idx -
                                             self.depth_radius)
                scalar_zero_int =\
                    self.tik_instance.Scalar(dtype="int64",
                                             name="scalar_zero_int",
                                             init_value=0)
                self.tik_instance.scalar_max(left_val, left_val,
                                             scalar_zero_int)

                right_val = \
                    self.tik_instance.Scalar(dtype="int64", name="right_val",
                                             init_value=c_idx +
                                             self.depth_radius)
                scalar_c_max_int = \
                    self.tik_instance.Scalar(dtype="int64",
                                             name="scalar_c_max_int",
                                             init_value=self.input_shape[1]-1)
                self.tik_instance.scalar_min(right_val, right_val,
                                             scalar_c_max_int)

                with self.tik_instance.for_range(left_val, right_val+1) as \
                        add_idx:
                    dest_ub_offset = batch_idx*self.one_batch_size +\
                                     c_idx*self.one_column_size
                    src_ub_offset = batch_idx*self.one_batch_size +\
                                    add_idx*self.one_column_size

                    self._vadd_func(self.data_output_ub[dest_ub_offset],
                                    self.data_output_ub[dest_ub_offset],
                                    self.data_square_ub[src_ub_offset],
                                    (self.one_column_size,))

        self._do_eltwise_operation()

        if self.dtype_real_in_out != self.input_dtype:
            # vconv from output_ub to cast_ub
            self._vconv_func(self.data_cast_ub, self.data_output_ub,
                             self.input_shape)
            # copy gm to cast_ub
            self._data_move_default(self.output_gm[offset_gm],
                                    self.data_cast_ub,
                                    self.dtype_real_in_out, self.input_shape)
        else:
            # copy output_ub to gm
            self._data_move_default(self.output_gm[offset_gm],
                                    self.data_output_ub,
                                    self.input_dtype, self.input_shape)

    def _do_operation_each_loop_cut_h_w(self, offset_gm):
        mte2_num = self._get_align_size(self.input_shape[2]*self.input_shape[3])
        if self.dtype_real_in_out != self.input_dtype:
            # copy gm to cast_ub
            self._gm_2_ub_cut_h_w(self.data_cast_ub, self.dtype_real_in_out,
                                  mte2_num, offset_gm)
            # vconv from cast_ub to input_ub
            self._vconv_func(self.data_input_ub, self.data_cast_ub,
                             (self.C, mte2_num))
        else:
            # copy gm to input_ub
            self._gm_2_ub_cut_h_w(self.data_input_ub, self.input_dtype,
                                  mte2_num, offset_gm)

        self._do_eltwise_operation_pre()
        # do dup zero for output_ub
        zero_scalar = self.tik_instance.Scalar(dtype=self.input_dtype,
                                               name="zero_scalar",
                                               init_value=0.0)
        self._vector_dup_func(self.data_output_ub, zero_scalar,
                              self.data_output_ub.shape)

        # do add operation in  col
        # each col size is input_shape[2]
        one_column_size_cut_h_w = mte2_num
        if self.input_dtype == "float16":
            col_lp_cnt = one_column_size_cut_h_w // PAR_COUNT_FP16
            col_left_size = one_column_size_cut_h_w % PAR_COUNT_FP16
            par_count = PAR_COUNT_FP16
            block_data_cnt = 16
        else:
            col_lp_cnt = one_column_size_cut_h_w // PAR_COUNT_FP32
            col_left_size = one_column_size_cut_h_w % PAR_COUNT_FP32
            par_count = PAR_COUNT_FP32
            block_data_cnt = 8

        win_size = self.depth_radius * 2 + 1
        # special process for lrn inference
        is_not_align_with_16 = \
            (one_column_size_cut_h_w % self.alignment_standards) or \
            col_lp_cnt > (self.depth_radius * 2 + 1) or \
            one_column_size_cut_h_w > MAX_HW_NUM or \
            self.C < 2 * self.depth_radius

        scalar_zero_int = \
            self.tik_instance.Scalar(dtype="int64", name="scalar_zero_int",
                                     init_value=0)
        scalar_c_max_int \
            = self.tik_instance.Scalar(dtype="int64",
                                       name="scalar_c_max_int",
                                       init_value=self.input_shape[1]-1)
        with self.tik_instance.for_range(0, self.C) as c_idx:
            if is_not_align_with_16:
                left_val = self.tik_instance.Scalar(
                    dtype="int64", name="left_val",
                    init_value=c_idx - self.depth_radius)
                with self.tik_instance.if_scope(left_val < scalar_zero_int):
                    left_val.set_as(scalar_zero_int)
                right_val = \
                    self.tik_instance.Scalar(
                        dtype="int64", name="right_val",
                        init_value=c_idx + self.depth_radius)
                with self.tik_instance.if_scope(right_val > scalar_c_max_int):
                    right_val.set_as(scalar_c_max_int)

                with self.tik_instance.for_range(
                        left_val, right_val+1) as add_idx:
                    dest_ub_offset = c_idx * one_column_size_cut_h_w
                    src_ub_offset = add_idx * one_column_size_cut_h_w
                    self._vadd_func(self.data_output_ub[dest_ub_offset],
                                    self.data_output_ub[dest_ub_offset],
                                    self.data_square_ub[src_ub_offset],
                                    (one_column_size_cut_h_w,))
            else:
                def _inner_process():
                    left_val = self.tik_instance.Scalar(
                        dtype="int64", name="left_val",
                        init_value=c_idx - self.depth_radius)
                    with self.tik_instance.if_scope(
                            left_val < scalar_zero_int):
                        left_val.set_as(scalar_zero_int)
                    right_val = \
                        self.tik_instance.Scalar(
                            dtype="int64", name="right_val",
                            init_value=c_idx + self.depth_radius)
                    with self.tik_instance.if_scope(
                            right_val > scalar_c_max_int):
                        right_val.set_as(scalar_c_max_int)

                    dest_ub_offset = c_idx * one_column_size_cut_h_w
                    src_ub_offset = left_val * one_column_size_cut_h_w
                    repeat_stride = one_column_size_cut_h_w // block_data_cnt

                    if col_lp_cnt:
                        with self.tik_instance.for_range(
                                0, col_lp_cnt) as lp_idx:
                            self.tik_instance.vadd(
                                par_count,
                                self.data_output_ub[dest_ub_offset +
                                                    lp_idx * par_count],
                                self.data_square_ub[src_ub_offset +
                                                    lp_idx * par_count],
                                self.data_output_ub[dest_ub_offset +
                                                    lp_idx * par_count],
                                (right_val + 1 - left_val),
                                1, 1, 1, 0, repeat_stride, 0)
                    if col_left_size:
                        offset_cnt = col_lp_cnt * par_count
                        self.tik_instance.vadd(
                            col_left_size,
                            self.data_output_ub[dest_ub_offset + offset_cnt],
                            self.data_square_ub[src_ub_offset + offset_cnt],
                            self.data_output_ub[dest_ub_offset + offset_cnt],
                            (right_val + 1 - left_val),
                            1, 1, 1, 0, repeat_stride, 0)
                with self.tik_instance.if_scope(c_idx < self.depth_radius):
                    _inner_process()
                with self.tik_instance.if_scope(
                        c_idx > scalar_c_max_int - self.depth_radius):
                    _inner_process()

        if not is_not_align_with_16:
            repeat_stride = one_column_size_cut_h_w // block_data_cnt
            dest_ub_offset = self.depth_radius * one_column_size_cut_h_w
            repeat_loop = \
                (self.C - 2 * self.depth_radius) // MAX_REPEAT_NUM
            left_repeat = \
                (self.C - 2 * self.depth_radius) % MAX_REPEAT_NUM
            with self.tik_instance.for_range(0, win_size) as win_idx:
                src_ub_offset = win_idx * one_column_size_cut_h_w

                def _inner_vadd_align_c(rp_index, repeat_num):
                    if col_lp_cnt:
                        with self.tik_instance.for_range(
                                0, col_lp_cnt) as lp_idx:
                            self.tik_instance.vadd(
                                par_count,
                                self.data_output_ub[
                                    dest_ub_offset +
                                    lp_idx * par_count +
                                    rp_index * MAX_REPEAT_NUM *
                                    one_column_size_cut_h_w],
                                self.data_square_ub[
                                    src_ub_offset +
                                    lp_idx * par_count +
                                    rp_index * MAX_REPEAT_NUM *
                                    one_column_size_cut_h_w],
                                self.data_output_ub[
                                    dest_ub_offset +
                                    lp_idx * par_count +
                                    rp_index * MAX_REPEAT_NUM *
                                    one_column_size_cut_h_w],
                                repeat_num,
                                1, 1, 1,
                                repeat_stride, repeat_stride, repeat_stride)
                    if col_left_size:
                        offset_cnt = col_lp_cnt * par_count
                        self.tik_instance.vadd(
                            col_left_size,
                            self.data_output_ub[
                                dest_ub_offset + offset_cnt +
                                rp_index * MAX_REPEAT_NUM *
                                one_column_size_cut_h_w],
                            self.data_square_ub[
                                src_ub_offset + offset_cnt +
                                rp_index * MAX_REPEAT_NUM *
                                one_column_size_cut_h_w],
                            self.data_output_ub[
                                dest_ub_offset + offset_cnt +
                                rp_index * MAX_REPEAT_NUM *
                                one_column_size_cut_h_w],
                            repeat_num,
                            1, 1, 1,
                            repeat_stride, repeat_stride, repeat_stride)

                if repeat_loop:
                    with self.tik_instance.for_range(0, repeat_loop) as rp_idx:
                        _inner_vadd_align_c(rp_idx, MAX_REPEAT_NUM)
                if left_repeat:
                    _inner_vadd_align_c(repeat_loop, left_repeat)

        self._do_eltwise_operation()

        if self.dtype_real_in_out != self.input_dtype:
            # vconv from output_ub to cast_ub
            self._vconv_func(self.data_cast_ub, self.data_output_ub,
                             (self.C, mte2_num))
            # copy gm to cast_ub
            self._ub_2_gm_cut_h_w(self.data_cast_ub, self.dtype_real_in_out,
                                  mte2_num, offset_gm)
        else:
            # copy output_ub to gm
            self._ub_2_gm_cut_h_w(self.data_output_ub, self.input_dtype,
                                  mte2_num, offset_gm)

    def _do_operation_each_loop_cut_batch_not_align(self, offset_gm):
        one_column_size_aligned = self._get_align_size(self.one_column_size)
        one_batch_size_aligned = self.C*one_column_size_aligned
        if self.dtype_real_in_out != self.input_dtype:
            # copy gm to cast_ub
            self._gm_2_ub_cut_batch_not_align(self.data_cast_ub,
                                              self.dtype_real_in_out,
                                              one_column_size_aligned,
                                              one_batch_size_aligned, offset_gm)
            # vconv from cast_ub to input_ub
            self._vconv_func(self.data_input_ub, self.data_cast_ub,
                             (self.N, one_batch_size_aligned))
        else:
            # copy gm to input_ub
            self._gm_2_ub_cut_batch_not_align(self.data_input_ub,
                                              self.input_dtype,
                                              one_column_size_aligned,
                                              one_batch_size_aligned, offset_gm)

        self._do_eltwise_operation_pre()
        # do dup zero for output_ub
        zero_scalar = self.tik_instance.Scalar(dtype=self.input_dtype,
                                               name="zero_scalar",
                                               init_value=0.0)
        self._vector_dup_func(self.data_output_ub, zero_scalar,
                              self.data_output_ub.shape)

        # do add operation in  col
        with self.tik_instance.for_range(0, self.N) as batch_idx:
            # each batch to (C, H*W)
            with self.tik_instance.for_range(0, self.C) as c_idx:
                left_val =\
                    self.tik_instance.Scalar(dtype="int64", name="left_val",
                                             init_value=c_idx -
                                             self.depth_radius)
                scalar_zero_int =\
                    self.tik_instance.Scalar(dtype="int64",
                                             name="scalar_zero_int",
                                             init_value=0)
                self.tik_instance.scalar_max(left_val, left_val,
                                             scalar_zero_int)

                right_val =\
                    self.tik_instance.Scalar(dtype="int64", name="right_val",
                                             init_value=c_idx +
                                             self.depth_radius)
                scalar_c_max_int =\
                    self.tik_instance.Scalar(dtype="int64",
                                             name="scalar_c_max_int",
                                             init_value=self.input_shape[1]-1)
                self.tik_instance.scalar_min(right_val, right_val,
                                             scalar_c_max_int)

                with self.tik_instance.for_range(left_val, right_val+1) as\
                        add_idx:
                    dest_ub_offset = batch_idx*one_batch_size_aligned + \
                                     c_idx*one_column_size_aligned
                    src_ub_offset = batch_idx*one_batch_size_aligned + \
                                    add_idx*one_column_size_aligned

                    self._vadd_func(self.data_output_ub[dest_ub_offset],
                                    self.data_output_ub[dest_ub_offset],
                                    self.data_square_ub[src_ub_offset],
                                    (one_column_size_aligned,))

        self._do_eltwise_operation()

        if self.dtype_real_in_out != self.input_dtype:
            # vconv from output_ub to cast_ub
            self._vconv_func(self.data_cast_ub, self.data_output_ub,
                             (self.N, one_batch_size_aligned))
            # copy gm to cast_ub
            self._ub_2_gm_cut_batch_not_align(self.data_cast_ub,
                                              self.dtype_real_in_out,
                                              one_column_size_aligned,
                                              one_batch_size_aligned, offset_gm)
        else:
            # copy output_ub to gm
            self._ub_2_gm_cut_batch_not_align(self.data_output_ub,
                                              self.input_dtype,
                                              one_column_size_aligned,
                                              one_batch_size_aligned, offset_gm)

    def _do_operation_each_core_n_cut_branch(self, offset_gm):
        one_buffer_single_batch = self._get_align_size(self.one_batch_size)
        total_ub_single_batch = 3*one_buffer_single_batch
        loop_num, batch_num_ub_once, batch_num_last =\
            self._cut_batch_axis(total_ub_single_batch)

        self.input_shape = (batch_num_ub_once, self.C, self.H, self.W)
        self.N = self.input_shape[0]
        self._allocate_ub_buffer()

        if loop_num == 1:
            self._do_operation_each_loop_cut_batch(offset_gm)
        elif (loop_num != 1) and (batch_num_ub_once == batch_num_last):
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                offset_cut_batch =\
                    offset_gm + loop_idx*batch_num_ub_once*self.one_batch_size
                self._do_operation_each_loop_cut_batch(offset_cut_batch)
        else:
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                offset_cut_batch =\
                    offset_gm + loop_idx*batch_num_ub_once*self.one_batch_size
                with self.tik_instance.if_scope(loop_idx < loop_num - 1):
                    self._do_operation_each_loop_cut_batch(offset_cut_batch)
                with self.tik_instance.else_scope():
                    self.input_shape = (batch_num_last, self.C, self.H, self.W)
                    self.N = self.input_shape[0]
                    self._do_operation_each_loop_cut_batch(offset_cut_batch)

    def _do_the_last_batch_not_align(self, offset_gm):
        # in this case, the last batch is to be two fragment,
        # first is not align while the second is 32B align
        h_w_size_second_loop =\
            (self.H * self.W // self.alignment_standards) *\
            self.alignment_standards
        h_w_size_first_loop = self.H*self.W - h_w_size_second_loop

        # do the first loop
        self.input_shape = (1, self.C, h_w_size_first_loop, 1)
        self._do_operation_each_loop_cut_h_w(offset_gm)

        # do the second loop
        offset_cut_h_w = offset_gm + h_w_size_first_loop
        self.input_shape = (1, self.C, h_w_size_second_loop, 1)
        self._do_operation_each_loop_cut_h_w(offset_cut_h_w)

    def _do_operation_each_core_n_cut_branch_not_align(self, offset_gm):
        one_column_align = self._get_align_size(self.one_column_size)
        one_buffer_single_batch_align = self.C*one_column_align
        total_ub_single_batch_align = 3*one_buffer_single_batch_align

        if self.N == 1:
            self.input_shape = (1, self.C, one_column_align, 1)
            self._allocate_ub_buffer()
            self._do_the_last_batch_not_align(offset_gm)
        else:
            batch_num_current_core = self.N
            self.N = self.N - 1
            loop_num, batch_num_ub_once, batch_num_last =\
                self._cut_batch_axis(total_ub_single_batch_align)
            self.input_shape = (batch_num_ub_once, self.C, one_column_align, 1)
            self.N = self.input_shape[0]
            self._allocate_ub_buffer()

            if loop_num == 1:
                self._do_operation_each_loop_cut_batch_not_align(offset_gm)
            elif (loop_num != 1) and (batch_num_ub_once == batch_num_last):
                with self.tik_instance.for_range(0, loop_num) as loop_idx:
                    offset_cut_batch =\
                        offset_gm +\
                        loop_idx*batch_num_ub_once*self.one_batch_size
                    self._do_operation_each_loop_cut_batch_not_align(
                        offset_cut_batch)
            else:
                with self.tik_instance.for_range(0, loop_num) as loop_idx:
                    offset_cut_batch =\
                        offset_gm +\
                        loop_idx*batch_num_ub_once*self.one_batch_size
                    with self.tik_instance.if_scope(loop_idx < loop_num - 1):
                        self._do_operation_each_loop_cut_batch_not_align(
                            offset_cut_batch)
                    with self.tik_instance.else_scope():
                        self.input_shape = (batch_num_last, self.C,
                                            one_column_align, 1)
                        self.N = self.input_shape[0]
                        self._do_operation_each_loop_cut_batch_not_align(
                            offset_cut_batch)

            offset_cut_batch =\
                offset_gm + (batch_num_current_core - 1)*self.one_batch_size
            self._do_the_last_batch_not_align(offset_cut_batch)

    def _do_operation_each_core_n_cut_branch_hw_little_not_align(self,
                                                                 offset_gm):
        one_column_align = self._get_align_size(self.one_column_size)
        one_buffer_single_batch_align = self.C*one_column_align
        total_ub_single_batch_align = 3*one_buffer_single_batch_align

        loop_num, batch_num_ub_once, batch_num_last =\
            self._cut_batch_axis(total_ub_single_batch_align)
        self.input_shape = (batch_num_ub_once, self.C, one_column_align, 1)
        self.N = self.input_shape[0]
        self._allocate_ub_buffer()

        if loop_num == 1:
            self._do_operation_each_loop_cut_batch_not_align(offset_gm)
        elif (loop_num != 1) and (batch_num_ub_once == batch_num_last):
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                offset_cut_batch =\
                    offset_gm + loop_idx*batch_num_ub_once*self.one_batch_size
                self._do_operation_each_loop_cut_batch_not_align(
                    offset_cut_batch)
        else:
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                offset_cut_batch =\
                    offset_gm + loop_idx*batch_num_ub_once*self.one_batch_size
                with self.tik_instance.if_scope(loop_idx < loop_num - 1):
                    self._do_operation_each_loop_cut_batch_not_align(
                        offset_cut_batch)
                with self.tik_instance.else_scope():
                    self.input_shape = (batch_num_last, self.C,
                                        one_column_align, 1)
                    self.N = self.input_shape[0]
                    self._do_operation_each_loop_cut_batch_not_align(
                        offset_cut_batch)

    def _do_operation_each_core_c_cut_branch_hw_little_not_align(self,
                                                                 offset_gm):
        batch_num_current_core = self.N

        column_size_align = self._get_align_size(self.one_column_size)
        self.input_shape = (1, self.depth_radius + 1, column_size_align, 1)
        self.N = self.input_shape[0]
        self._allocate_ub_buffer()
        self.pre_square_sum_ub =\
            self.tik_instance.Tensor(self.input_dtype, (column_size_align,),
                                     name="pre_square_sum_ub",
                                     scope=tik.scope_ubuf)
        self.right_square_sum_ub =\
            self.tik_instance.Tensor(self.input_dtype, (column_size_align,),
                                     name="right_square_sum_ub",
                                     scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, batch_num_current_core) as\
                batch_idx:
            base_offset = offset_gm + batch_idx*self.one_batch_size

            # do dup zero for output_ub
            zero_scalar =\
                self.tik_instance.Scalar(dtype=self.input_dtype,
                                         name="zero_scalar", init_value=0.0)
            self._vector_dup_func(self.data_output_ub, zero_scalar,
                                  self.data_output_ub.shape)

            # compute the first column
            mte2_num = column_size_align
            if self.dtype_real_in_out != self.input_dtype:
                # copy gm to cast_ub
                self._gm_2_ub_c_cut_branch_hw_little_not_align(
                    self.data_cast_ub, self.dtype_real_in_out, mte2_num,
                    base_offset)
                # vconv from cast_ub to input_ub
                self._vconv_func(self.data_input_ub, self.data_cast_ub,
                                 (self.depth_radius+1, mte2_num))
            else:
                # copy gm to input_ub
                self._gm_2_ub_c_cut_branch_hw_little_not_align(
                    self.data_input_ub, self.input_dtype, mte2_num, base_offset)

            self._do_eltwise_operation_pre()
            with self.tik_instance.for_range(0, self.depth_radius+1) as c_idx:
                src_ub_offset = c_idx*mte2_num
                self._vadd_func(self.data_output_ub, self.data_output_ub,
                                self.data_square_ub[src_ub_offset], (mte2_num,))

            self.tik_instance.data_move(self.pre_square_sum_ub,
                                        self.data_output_ub, constant.SID,
                                        constant.DEFAULT_NBURST,
                                        constant.DEFAULT_BURST_LEN,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)

            loop_num, c_size_each_loop, c_size_last_loop =\
                self._cut_hw_c_axis_c()
            if c_size_each_loop == c_size_last_loop:
                with self.tik_instance.for_range(0, loop_num) as loop_idx:
                    # each loop deal with c_size_each_loop column data
                    offset_cut_c =\
                        base_offset +\
                        loop_idx*c_size_each_loop*self.one_column_size
                    self._do_operation_each_loop_c_cut_hw_little_not_align(
                        c_size_each_loop, c_size_each_loop, loop_idx,
                        offset_cut_c, column_size_align)
            else:
                with self.tik_instance.for_range(0, loop_num) as loop_idx:
                    offset_cut_c =\
                        base_offset +\
                        loop_idx*c_size_each_loop*self.one_column_size
                    with self.tik_instance.if_scope(loop_idx < loop_num - 1):
                        self._do_operation_each_loop_c_cut_hw_little_not_align(
                            c_size_each_loop, c_size_each_loop, loop_idx,
                            offset_cut_c, column_size_align)
                    with self.tik_instance.else_scope():
                        self._do_operation_each_loop_c_cut_hw_little_not_align(
                            c_size_each_loop, c_size_last_loop, loop_idx,
                            offset_cut_c, column_size_align)

    def _do_operation_each_core_h_w_cut_branch(self, offset_gm):
        batch_num_current_core = self.N

        loop_num, h_w_size_each_loop, h_w_size_last_loop = self._cut_h_w_axis()
        self.input_shape = (1, self.C, h_w_size_each_loop, 1)
        self.N = self.input_shape[0]
        self._allocate_ub_buffer()

        with self.tik_instance.for_range(0, batch_num_current_core) as\
                batch_idx:
            if loop_num == 1:
                offset_cut_h_w = offset_gm + batch_idx*self.one_batch_size
                self._do_operation_each_loop_cut_h_w(offset_cut_h_w)
            elif (loop_num != 1) and (h_w_size_each_loop == h_w_size_last_loop):
                with self.tik_instance.for_range(0, loop_num) as loop_idx:
                    offset_cut_h_w =\
                        offset_gm + batch_idx*self.one_batch_size +\
                        h_w_size_each_loop*loop_idx
                    self._do_operation_each_loop_cut_h_w(offset_cut_h_w)
            else:
                with self.tik_instance.for_range(0, loop_num) as loop_idx:
                    offset_cut_h_w =\
                        offset_gm + batch_idx*self.one_batch_size +\
                        h_w_size_each_loop*loop_idx
                    with self.tik_instance.if_scope(loop_idx < loop_num - 1):
                        self._do_operation_each_loop_cut_h_w(offset_cut_h_w)
                    with self.tik_instance.else_scope():
                        self.input_shape = (1, self.C, h_w_size_last_loop, 1)
                        self._do_operation_each_loop_cut_h_w(offset_cut_h_w)

    def _do_operation_each_core_h_w_cut_branch_not_align(self, offset_gm):
        # keep the first segment is not 32B align, the rest all 32B align
        batch_num_current_core = self.N

        loop_num, h_w_size_each_loop, h_w_size_first_loop =\
            self._cut_h_w_axis_not_align()
        self.input_shape = (1, self.C, h_w_size_each_loop, 1)
        self.N = self.input_shape[0]
        self._allocate_ub_buffer()

        with self.tik_instance.for_range(0, batch_num_current_core) as\
                batch_idx:
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                with self.tik_instance.if_scope(loop_idx == 0):
                    # in this case, the fragment is not align
                    offset_cut_h_w = offset_gm + batch_idx*self.one_batch_size
                    self.input_shape = (1, self.C, h_w_size_first_loop, 1)
                    self._do_operation_each_loop_cut_h_w(offset_cut_h_w)
                with self.tik_instance.else_scope():
                    # the remaining loop are 32B align
                    offset_cut_h_w =\
                        offset_gm + batch_idx*self.one_batch_size +\
                        h_w_size_first_loop + (loop_idx - 1)*h_w_size_each_loop
                    self.input_shape = (1, self.C, h_w_size_each_loop, 1)
                    self._do_operation_each_loop_cut_h_w(offset_cut_h_w)

    def _compute_first_column_square_sum(self, copy_c_size, offset_gm):
        mte2_num = self._get_align_size(self.one_column_size)*copy_c_size
        if self.dtype_real_in_out != self.input_dtype:
            # copy gm to cast_ub
            self._data_move_default(self.data_cast_ub, self.input_gm[offset_gm],
                                    self.dtype_real_in_out, (mte2_num,))
            # vconv from cast_ub to input_ub
            self._vconv_func(self.data_input_ub, self.data_cast_ub, (mte2_num,))
        else:
            # copy gm to input_ub
            self._data_move_default(self.data_input_ub,
                                    self.input_gm[offset_gm],
                                    self.input_dtype, (mte2_num,))

        self._do_eltwise_operation_pre()
        with self.tik_instance.for_range(0, copy_c_size) as c_idx:
            src_ub_offset = c_idx*self._get_align_size(self.one_column_size)
            self._vadd_func(self.data_output_ub, self.data_output_ub,
                            self.data_square_ub[src_ub_offset],
                            (self._get_align_size(self.one_column_size),))

    def _do_operation_each_loop_cut_c(self, c_size_pre_loop,
                                      c_size_current_loop, loop_idx, offset_gm):
        # copy c_size_current_loop to input_ub for the last vmul
        mte2_num = c_size_current_loop*self._get_align_size(self.one_column_size)
        if self.dtype_real_in_out != self.input_dtype:
            # copy gm to cast_ub
            self._data_move_default(self.data_cast_ub, self.input_gm[offset_gm],
                                    self.dtype_real_in_out, (mte2_num,))
            # vconv from cast_ub to input_ub
            self._vconv_func(self.data_input_ub, self.data_cast_ub, (mte2_num,))
        else:
            # copy gm to input_ub
            self._data_move_default(self.data_input_ub,
                                    self.input_gm[offset_gm], self.input_dtype,
                                    (mte2_num,))

        # do dup zero for output_ub
        zero_scalar = self.tik_instance.Scalar(dtype=self.input_dtype,
                                               name="zero_scalar",
                                               init_value=0.0)
        self._vector_dup_func(self.data_output_ub, zero_scalar,
                              self.data_output_ub.shape)

        batch_offset = offset_gm - loop_idx*c_size_pre_loop*self.one_column_size
        with self.tik_instance.for_range(0, c_size_current_loop) as c_idx:
            with self.tik_instance.if_scope((c_idx + loop_idx) == 0):
                self.tik_instance.data_move(self.data_output_ub,
                                            self.pre_square_sum_ub,
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            self._get_align_size(self.one_column_size) //
                                            self.one_block_ele_num_input,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
            with self.tik_instance.else_scope():
                actual_idx_current_batch = loop_idx*c_size_pre_loop + c_idx
                left_sub_val =\
                    self.tik_instance.Scalar(dtype="int64",
                                             name="left_sub_val",
                                             init_value=actual_idx_current_batch -
                                             self.depth_radius - 1)
                right_add_val =\
                    self.tik_instance.Scalar(dtype="int64",
                                             name="right_add_val",
                                             init_value=actual_idx_current_batch +
                                             self.depth_radius)

                scalar_zero_int =\
                    self.tik_instance.Scalar(dtype="int64",
                                             name="scalar_zero_int",
                                             init_value=0)
                scalar_c_max_int =\
                    self.tik_instance.Scalar(dtype="int64",
                                             name="scalar_c_max_int",
                                             init_value=self.C-1)

                self.tik_instance.data_move(self.data_output_ub[
                    c_idx*self._get_align_size(self.one_column_size)],
                                            self.pre_square_sum_ub,
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            self._get_align_size(self.one_column_size) //
                                            self.one_block_ele_num_input,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                with self.tik_instance.if_scope(left_sub_val >=
                                                scalar_zero_int):
                    # pre_square_sum_ub sub left_sub_val colum
                    if self.dtype_real_in_out != self.input_dtype:
                        # copy gm to cast_ub
                        self._data_move_default(self.data_cast_ub,
                                                self.input_gm[
                                                    batch_offset +
                                                    left_sub_val *
                                                    self.one_column_size],
                                                self.dtype_real_in_out,
                                                (self.one_column_size,))
                        # vconv from cast_ub to input_ub
                        self._vconv_func(self.data_square_ub, self.data_cast_ub,
                                         (self._get_align_size(self.one_column_size),))
                    else:
                        # copy gm to input_ub
                        self._data_move_default(self.data_square_ub,
                                                self.input_gm[
                                                    batch_offset +
                                                    left_sub_val *
                                                    self.one_column_size],
                                                self.input_dtype,
                                                (self.one_column_size,))

                    if self.is_alpha_sqrt_flag:
                        self._vmuls_func(self.data_square_ub, self.data_square_ub, self.alpha_sqrt,
                                         (self._get_align_size(self.one_column_size),))

                    self._vmul_func(self.data_square_ub,
                                    self.data_square_ub,
                                    self.data_square_ub,
                                    (self._get_align_size(self.one_column_size),))
                    self._vsub_func(self.data_output_ub[
                        c_idx*self._get_align_size(self.one_column_size)],
                                    self.data_output_ub[
                                        c_idx*self._get_align_size(self.one_column_size)],
                                    self.data_square_ub,
                                    (self._get_align_size(self.one_column_size),))

                with self.tik_instance.if_scope(right_add_val <=
                                                scalar_c_max_int):
                    # pre_square_sum_ub add right_add_val
                    if self.dtype_real_in_out != self.input_dtype:
                        # copy gm to cast_ub
                        self._data_move_default(self.data_cast_ub,
                                                self.input_gm[
                                                    batch_offset +
                                                    right_add_val *
                                                    self.one_column_size],
                                                self.dtype_real_in_out,
                                                (self.one_column_size,))
                        # vconv from cast_ub to input_ub
                        self._vconv_func(self.right_square_sum_ub,
                                         self.data_cast_ub,
                                         (self._get_align_size(self.one_column_size),))
                    else:
                        # copy gm to input_ub
                        self._data_move_default(self.right_square_sum_ub,
                                                self.input_gm[
                                                    batch_offset +
                                                    right_add_val *
                                                    self.one_column_size],
                                                self.input_dtype,
                                                (self.one_column_size,))

                    if self.is_alpha_sqrt_flag:
                        self._vmuls_func(self.right_square_sum_ub,
                                         self.right_square_sum_ub, self.alpha_sqrt,
                                         (self._get_align_size(self.one_column_size),))
                    self._vmul_func(self.right_square_sum_ub,
                                    self.right_square_sum_ub,
                                    self.right_square_sum_ub,
                                    (self._get_align_size(self.one_column_size),))
                    self._vadd_func(self.data_output_ub[
                        c_idx*self._get_align_size(self.one_column_size)],
                                    self.data_output_ub[c_idx *
                                                        self._get_align_size(self.one_column_size)],
                                    self.right_square_sum_ub,
                                    (self._get_align_size(self.one_column_size),))

                # refresh pre_square_sum_ub
                self.tik_instance.data_move(self.pre_square_sum_ub,
                                            self.data_output_ub[
                                                c_idx*self._get_align_size(self.one_column_size)],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            self._get_align_size(self.one_column_size) //
                                            self.one_block_ele_num_input,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)

        # do the remaining eltwise operation
        self._do_eltwise_operation()

        # copy data to gm
        if self.dtype_real_in_out != self.input_dtype:
            # vconv from output_ub to cast_ub
            self._vconv_func(self.data_cast_ub, self.data_output_ub,
                             (mte2_num,))
            # copy gm to cast_ub
            self._data_move_default(self.output_gm[offset_gm],
                                    self.data_cast_ub, self.dtype_real_in_out,
                                    (mte2_num,))
        else:
            # copy output_ub to gm
            self._data_move_default(self.output_gm[offset_gm],
                                    self.data_output_ub, self.input_dtype,
                                    (mte2_num,))

    def _do_operation_each_core_c_cut_branch(self, offset_gm):
        batch_num_current_core = self.N

        loop_num, c_size_each_loop, c_size_last_loop = self._cut_c_axis()
        self.input_shape = (1, c_size_each_loop, self._get_align_size(self.one_column_size), 1)
        self.N = self.input_shape[0]
        self._allocate_ub_buffer()
        self.pre_square_sum_ub =\
            self.tik_instance.Tensor(self.input_dtype,
                                     (self._get_align_size(
                                         self.one_column_size),),
                                     name="pre_square_sum_ub",
                                     scope=tik.scope_ubuf)
        self.right_square_sum_ub =\
            self.tik_instance.Tensor(self.input_dtype,
                                     (self._get_align_size(
                                         self.one_column_size),),
                                     name="right_square_sum_ub",
                                     scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, batch_num_current_core) as\
                batch_idx:
            # do dup zero for output_ub
            zero_scalar =\
                self.tik_instance.Scalar(dtype=self.input_dtype,
                                         name="zero_scalar", init_value=0.0)
            self._vector_dup_func(self.data_output_ub, zero_scalar,
                                  self.data_output_ub.shape)
            # compute the first column of output, need (r+1) input
            if c_size_each_loop >= self.depth_radius + 1:
                # the r+1 can be copy to UB once
                self._compute_first_column_square_sum(self.depth_radius + 1,
                                                      offset_gm + batch_idx *
                                                      self.one_batch_size)
            else:
                # the r+1 need be copied by many times
                copy_times = (self.depth_radius + 1) // c_size_each_loop
                copy_c_size_last = c_size_each_loop
                if (self.depth_radius + 1) % c_size_each_loop != 0:
                    copy_c_size_last =\
                        (self.depth_radius + 1) - copy_times*c_size_each_loop
                    copy_times = copy_times + 1

                if c_size_each_loop == copy_c_size_last:
                    with self.tik_instance.for_range(0, copy_times) as copy_idx:
                        offset_cut_c =\
                            offset_gm + batch_idx*self.one_batch_size +\
                            copy_idx*c_size_each_loop*self.one_column_size
                        self._compute_first_column_square_sum(c_size_each_loop,
                                                              offset_cut_c)
                else:
                    with self.tik_instance.for_range(0, copy_times) as copy_idx:
                        offset_cut_c =\
                            offset_gm + batch_idx*self.one_batch_size +\
                            copy_idx*c_size_each_loop*self.one_column_size
                        with self.tik_instance.if_scope(copy_idx <
                                                        copy_times - 1):
                            self._compute_first_column_square_sum(
                                c_size_each_loop, offset_cut_c)
                        with self.tik_instance.else_scope():
                            self._compute_first_column_square_sum(
                                copy_c_size_last, offset_cut_c)

            self.tik_instance.data_move(self.pre_square_sum_ub,
                                        self.data_output_ub, constant.SID,
                                        constant.DEFAULT_NBURST,
                                        self._get_align_size(self.one_column_size) //
                                        self.one_block_ele_num_input,
                                        constant.STRIDE_ZERO,
                                        constant.STRIDE_ZERO)

            if (loop_num != 1) and (c_size_each_loop == c_size_last_loop):
                with self.tik_instance.for_range(0, loop_num) as loop_idx:
                    # each loop deal with c_size_each_loop column data
                    offset_cut_c =\
                        offset_gm + batch_idx*self.one_batch_size +\
                        loop_idx*c_size_each_loop*self.one_column_size
                    self._do_operation_each_loop_cut_c(c_size_each_loop,
                                                       c_size_each_loop,
                                                       loop_idx, offset_cut_c)
            else:
                with self.tik_instance.for_range(0, loop_num) as loop_idx:
                    offset_cut_c =\
                        offset_gm + batch_idx*self.one_batch_size +\
                        loop_idx*c_size_each_loop*self.one_column_size
                    with self.tik_instance.if_scope(loop_idx < loop_num - 1):
                        self._do_operation_each_loop_cut_c(c_size_each_loop,
                                                           c_size_each_loop,
                                                           loop_idx,
                                                           offset_cut_c)
                    with self.tik_instance.else_scope():
                        self._do_operation_each_loop_cut_c(c_size_each_loop,
                                                           c_size_last_loop,
                                                           loop_idx,
                                                           offset_cut_c)

    def _do_operation_each_loop_hw_c_all_cut_c(self, c_size_pre_loop,
                                               c_size_current_loop,
                                               loop_idx, offset_gm,
                                               current_hw_size):
        current_hw_size = self._get_align_size(current_hw_size)
        mte2_num = current_hw_size
        if self.dtype_real_in_out != self.input_dtype:
            # copy gm to cast_ub
            self._gm_2_ub_hw_c_all_cut_c(self.data_cast_ub,
                                         self.dtype_real_in_out, mte2_num,
                                         offset_gm, c_size_current_loop)
            # vconv from cast_ub to input_ub
            self._vconv_func(self.data_input_ub, self.data_cast_ub,
                             (c_size_current_loop, current_hw_size))
        else:
            # copy gm to input_ub
            self._gm_2_ub_hw_c_all_cut_c(self.data_input_ub, self.input_dtype,
                                         mte2_num, offset_gm,
                                         c_size_current_loop)

        # do dup zero for output_ub
        zero_scalar =\
            self.tik_instance.Scalar(dtype=self.input_dtype, name="zero_scalar",
                                     init_value=0.0)
        self._vector_dup_func(self.data_output_ub, zero_scalar,
                              self.data_output_ub.shape)

        base_offset = offset_gm - loop_idx*c_size_pre_loop*self.one_column_size
        with self.tik_instance.for_range(0, c_size_current_loop) as c_idx:
            with self.tik_instance.if_scope((c_idx + loop_idx) == 0):
                self.tik_instance.data_move(self.data_output_ub,
                                            self.pre_square_sum_ub,
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            current_hw_size //
                                            self.one_block_ele_num_input,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
            with self.tik_instance.else_scope():
                actual_idx_current_batch = loop_idx*c_size_pre_loop + c_idx
                left_sub_val =\
                    self.tik_instance.Scalar(dtype="int64", name="left_sub_val",
                                             init_value=actual_idx_current_batch -
                                             self.depth_radius - 1)
                right_add_val =\
                    self.tik_instance.Scalar(dtype="int64",
                                             name="right_add_val",
                                             init_value=actual_idx_current_batch +
                                             self.depth_radius)

                scalar_zero_int =\
                    self.tik_instance.Scalar(dtype="int64",
                                             name="scalar_zero_int",
                                             init_value=0)
                scalar_c_max_int =\
                    self.tik_instance.Scalar(dtype="int64",
                                             name="scalar_c_max_int",
                                             init_value=self.C-1)

                self.tik_instance.data_move(self.data_output_ub[
                    c_idx*current_hw_size],
                                            self.pre_square_sum_ub,
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            current_hw_size //
                                            self.one_block_ele_num_input,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)
                with self.tik_instance.if_scope(left_sub_val >=
                                                scalar_zero_int):
                    # pre_square_sum_ub sub left_sub_val colum
                    if self.dtype_real_in_out != self.input_dtype:
                        # copy gm to cast_ub
                        self._data_move_default(self.data_cast_ub,
                                                self.input_gm[
                                                    base_offset +
                                                    left_sub_val *
                                                    self.one_column_size],
                                                self.dtype_real_in_out,
                                                (current_hw_size,))
                        # vconv from cast_ub to input_ub
                        self._vconv_func(self.data_square_ub, self.data_cast_ub,
                                         (current_hw_size,))
                    else:
                        # copy gm to input_ub
                        self._data_move_default(self.data_square_ub,
                                                self.input_gm[
                                                    base_offset +
                                                    left_sub_val *
                                                    self.one_column_size],
                                                self.input_dtype,
                                                (current_hw_size,))

                    if self.is_alpha_sqrt_flag:
                        self._vmuls_func(self.data_square_ub, self.data_square_ub,
                                         self.alpha_sqrt, (current_hw_size,))

                    self._vmul_func(self.data_square_ub, self.data_square_ub,
                                    self.data_square_ub, (current_hw_size,))
                    self._vsub_func(self.data_output_ub[c_idx*current_hw_size],
                                    self.data_output_ub[c_idx*current_hw_size],
                                    self.data_square_ub, (current_hw_size,))

                with self.tik_instance.if_scope(right_add_val <=
                                                scalar_c_max_int):
                    # pre_square_sum_ub add right_add_val
                    if self.dtype_real_in_out != self.input_dtype:
                        # copy gm to cast_ub
                        self._data_move_default(self.data_cast_ub,
                                                self.input_gm[
                                                    base_offset + right_add_val *
                                                    self.one_column_size],
                                                self.dtype_real_in_out,
                                                (current_hw_size,))
                        # vconv from cast_ub to input_ub
                        self._vconv_func(self.right_square_sum_ub,
                                         self.data_cast_ub, (current_hw_size,))
                    else:
                        # copy gm to input_ub
                        self._data_move_default(self.right_square_sum_ub,
                                                self.input_gm[
                                                    base_offset +
                                                    right_add_val *
                                                    self.one_column_size],
                                                self.input_dtype,
                                                (current_hw_size,))
                    if self.is_alpha_sqrt_flag:
                        self._vmuls_func(self.right_square_sum_ub, self.right_square_sum_ub,
                                         self.alpha_sqrt, (current_hw_size,))
                    self._vmul_func(self.right_square_sum_ub,
                                    self.right_square_sum_ub,
                                    self.right_square_sum_ub,
                                    (current_hw_size,))
                    self._vadd_func(self.data_output_ub[c_idx*current_hw_size],
                                    self.data_output_ub[c_idx*current_hw_size],
                                    self.right_square_sum_ub,
                                    (current_hw_size,))

                # refresh pre_square_sum_ub
                self.tik_instance.data_move(self.pre_square_sum_ub,
                                            self.data_output_ub[
                                                c_idx*current_hw_size],
                                            constant.SID,
                                            constant.DEFAULT_NBURST,
                                            current_hw_size //
                                            self.one_block_ele_num_input,
                                            constant.STRIDE_ZERO,
                                            constant.STRIDE_ZERO)

        # do the remaining eltwise operation
        self._do_eltwise_operation()

        # copy data to gm
        if self.dtype_real_in_out != self.input_dtype:
            # vconv from output_ub to cast_ub
            self._vconv_func(self.data_cast_ub, self.data_output_ub,
                             (c_size_current_loop, current_hw_size))
            # copy gm to cast_ub
            self._ub_2_gm_hw_c_all_cut_c(self.data_cast_ub,
                                         self.dtype_real_in_out,
                                         mte2_num, offset_gm,
                                         c_size_current_loop)
        else:
            # copy output_ub to gm
            self._ub_2_gm_hw_c_all_cut_c(self.data_output_ub, self.input_dtype,
                                         mte2_num, offset_gm,
                                         c_size_current_loop)

    def _do_operation_each_loop_c_cut_hw_little_not_align(
            self, c_size_pre_loop, c_size_current_loop, loop_idx, offset_gm,
            current_hw_size):
        self._do_operation_each_loop_hw_c_all_cut_c(c_size_pre_loop,
                                                    c_size_current_loop,
                                                    loop_idx, offset_gm,
                                                    current_hw_size)

    def _do_operation_each_loop_hw_c_all_cut_hw(self, current_hw_size,
                                                offset_gm):
        # do dup zero for output_ub
        zero_scalar =\
            self.tik_instance.Scalar(dtype=self.input_dtype,
                                     name="zero_scalar", init_value=0.0)
        self._vector_dup_func(self.data_output_ub, zero_scalar,
                              self.data_output_ub.shape)

        # compute the first column
        mte2_num = self._get_align_size(current_hw_size)
        if self.dtype_real_in_out != self.input_dtype:
            # copy gm to cast_ub
            self._gm_2_ub_hw_c_all_cut_hw(self.data_cast_ub,
                                          self.dtype_real_in_out, mte2_num,
                                          offset_gm)
            # vconv from cast_ub to input_ub
            self._vconv_func(self.data_input_ub, self.data_cast_ub,
                             (self.depth_radius+1, mte2_num))
        else:
            # copy gm to input_ub
            self._gm_2_ub_hw_c_all_cut_hw(self.data_input_ub, self.input_dtype,
                                          mte2_num, offset_gm)

        self._do_eltwise_operation_pre()
        with self.tik_instance.for_range(0, self.depth_radius+1) as c_idx:
            src_ub_offset = c_idx*mte2_num
            self._vadd_func(self.data_output_ub, self.data_output_ub,
                            self.data_square_ub[src_ub_offset], (mte2_num,))

        self.tik_instance.data_move(self.pre_square_sum_ub, self.data_output_ub,
                                    constant.SID, constant.DEFAULT_NBURST,
                                    mte2_num//self.one_block_ele_num_input,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

        loop_num, c_size_each_loop, c_size_last_loop = self._cut_hw_c_axis_c()
        if c_size_each_loop == c_size_last_loop:
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                # each loop deal with c_size_each_loop column data
                offset_cut_c =\
                    offset_gm + loop_idx*c_size_each_loop*self.one_column_size
                self._do_operation_each_loop_hw_c_all_cut_c(c_size_each_loop,
                                                            c_size_each_loop,
                                                            loop_idx,
                                                            offset_cut_c,
                                                            mte2_num)
        else:
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                offset_cut_c =\
                    offset_gm + loop_idx*c_size_each_loop*self.one_column_size
                with self.tik_instance.if_scope(loop_idx < loop_num - 1):
                    self._do_operation_each_loop_hw_c_all_cut_c(c_size_each_loop,
                                                                c_size_each_loop,
                                                                loop_idx,
                                                                offset_cut_c,
                                                                mte2_num)
                with self.tik_instance.else_scope():
                    self._do_operation_each_loop_hw_c_all_cut_c(c_size_each_loop,
                                                                c_size_last_loop,
                                                                loop_idx,
                                                                offset_cut_c,
                                                                mte2_num)

    def _do_operation_each_core_hw_c_all_cut_branch(self, offset_gm):
        batch_num_current_core = self.N

        loop_num, h_w_size_each_loop, h_w_size_last_loop =\
            self._cut_hw_c_axis_hw()
        self.input_shape = (1, self.depth_radius + 1, h_w_size_each_loop, 1)
        self.N = self.input_shape[0]
        self._allocate_ub_buffer()
        self.pre_square_sum_ub =\
            self.tik_instance.Tensor(self.input_dtype, (h_w_size_each_loop,),
                                     name="pre_square_sum_ub",
                                     scope=tik.scope_ubuf)
        self.right_square_sum_ub =\
            self.tik_instance.Tensor(self.input_dtype, (h_w_size_each_loop,),
                                     name="right_square_sum_ub",
                                     scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, batch_num_current_core) as\
                batch_idx:
            if h_w_size_each_loop == h_w_size_last_loop:
                with self.tik_instance.for_range(0, loop_num) as loop_idx:
                    offset_cut_hw_c =\
                        offset_gm + batch_idx*self.one_batch_size +\
                        h_w_size_each_loop*loop_idx
                    self._do_operation_each_loop_hw_c_all_cut_hw(h_w_size_each_loop,
                                                                 offset_cut_hw_c)
            else:
                with self.tik_instance.for_range(0, loop_num) as loop_idx:
                    offset_cut_hw_c =\
                        offset_gm + batch_idx*self.one_batch_size +\
                        h_w_size_each_loop*loop_idx
                    with self.tik_instance.if_scope(loop_idx < loop_num - 1):
                        self._do_operation_each_loop_hw_c_all_cut_hw(h_w_size_each_loop,
                                                                     offset_cut_hw_c)
                    with self.tik_instance.else_scope():
                        self.input_shape =\
                            (1, self.depth_radius + 1, h_w_size_last_loop, 1)
                        self._do_operation_each_loop_hw_c_all_cut_hw(h_w_size_last_loop,
                                                                     offset_cut_hw_c)

    def _do_operation_each_core_hw_c_cut_branch_not_align(self, offset_gm):
        batch_num_current_core = self.N

        loop_num, h_w_size_each_loop, h_w_size_first_loop =\
            self._cut_hw_c_axis_hw_not_align()
        self.input_shape = (1, self.depth_radius + 1, h_w_size_each_loop, 1)
        self.N = self.input_shape[0]
        self._allocate_ub_buffer()
        self.pre_square_sum_ub = \
            self.tik_instance.Tensor(self.input_dtype, (h_w_size_each_loop,),
                                     name="pre_square_sum_ub",
                                     scope=tik.scope_ubuf)
        self.right_square_sum_ub =\
            self.tik_instance.Tensor(self.input_dtype, (h_w_size_each_loop,),
                                     name="right_square_sum_ub",
                                     scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, batch_num_current_core) as\
                batch_idx:
            with self.tik_instance.for_range(0, loop_num) as loop_idx:
                with self.tik_instance.if_scope(loop_idx == 0):
                    offset_cut_hw_c = offset_gm + batch_idx*self.one_batch_size
                    self.input_shape =\
                        (1, self.depth_radius + 1, self._get_align_size(h_w_size_first_loop), 1)
                    self._do_operation_each_loop_hw_c_all_cut_hw(h_w_size_first_loop,
                                                                 offset_cut_hw_c)
                with self.tik_instance.else_scope():
                    offset_cut_hw_c =\
                        offset_gm + batch_idx*self.one_batch_size +\
                        h_w_size_first_loop + (loop_idx - 1)*h_w_size_each_loop
                    self.input_shape = (1, self.depth_radius + 1, h_w_size_each_loop, 1)
                    self._do_operation_each_loop_hw_c_all_cut_hw(h_w_size_each_loop,
                                                                 offset_cut_hw_c)

    def _do_operation_each_core(self, offset_gm):
        n_cut_flag, c_cut_flag, hw_cut_flag, hw_c_all_cut_flag = self._do_tiling()

        if n_cut_flag is True:
            self._do_operation_each_core_n_cut_branch(offset_gm)
        elif c_cut_flag is True:
            # ub can store one h*w column
            self._do_operation_each_core_c_cut_branch(offset_gm)
        elif hw_cut_flag is True:
            self._do_operation_each_core_h_w_cut_branch(offset_gm)
        elif hw_c_all_cut_flag is True:
            self._do_operation_each_core_hw_c_all_cut_branch(offset_gm)

    def _check_c_axis_too_large(self):
        n_cut_flag, c_cut_flag, hw_cut_flag, hw_c_all_cut_flag = self._do_tiling()
        if c_cut_flag or hw_c_all_cut_flag:
            return True

        return False

    def _get_target_core_num_batch_one(self):
        segment_num = self.one_column_size // self.alignment_standards
        if self.one_column_size % self.alignment_standards != 0:
            segment_num = segment_num + 1

        if segment_num <= self.device_aicore_num:
            core_num = segment_num
            hw_num_each_core = self.alignment_standards
        else:
            core_num = self.device_aicore_num
            hw_num_each_core = (segment_num // core_num)*self.alignment_standards
        hw_num_first_core = self.one_column_size - (core_num - 1)*hw_num_each_core

        return core_num, hw_num_each_core, hw_num_first_core

    def tik_instance_function(self):
        """
        do the LRN operation when H*W is 32B align

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.core_num == 1:
            # check if mutlti core can be used
            if (self.one_column_size <= self.alignment_standards) or \
                    self._check_c_axis_too_large():
                if self.input_dtype != self.dtype_real_in_out:
                    self._allocate_cast_ub()
                # multi core can not be used
                self.input_shape = (1, self.C, self.H, self.W)
                self.N = self.input_shape[0]
                offset_gm = 0
                self._do_operation_each_core(offset_gm)
            else:
                # multi core can be used
                core_num, hw_num_each_core, hw_num_first_core =\
                    self._get_target_core_num_batch_one()
                with self.tik_instance.for_range(0, core_num, block_num=core_num) as block_idx:
                    if self.input_dtype != self.dtype_real_in_out:
                        self._allocate_cast_ub()
                    with self.tik_instance.if_scope(block_idx == 0):
                        self.input_shape = (1, self.C, hw_num_first_core, 1)
                        self.N = self.input_shape[0]
                        offset_gm = 0
                        self._do_operation_each_core_h_w_cut_branch(offset_gm)
                    with self.tik_instance.else_scope():
                        self.input_shape = (1, self.C, hw_num_each_core, 1)
                        self.N = self.input_shape[0]
                        offset_gm = hw_num_first_core + (block_idx - 1)*hw_num_each_core
                        self._do_operation_each_core_h_w_cut_branch(offset_gm)

        elif (self.core_num > 1) and (self.core_num < MAX_CORE_NUMBER):
            # each core handle only one batch
            with self.tik_instance.for_range(0, self.core_num,
                                             block_num=self.core_num) as\
                    block_idx:
                if self.input_dtype != self.dtype_real_in_out:
                    self._allocate_cast_ub()
                self.input_shape = (1, self.C, self.H, self.W)
                self.N = self.input_shape[0]
                offset_gm = block_idx*self.one_batch_size
                self._do_operation_each_core(offset_gm)
        else:
            with self.tik_instance.for_range(0, self.core_num,
                                             block_num=self.core_num) as\
                    block_idx:
                if self.input_dtype != self.dtype_real_in_out:
                    self._allocate_cast_ub()
                with self.tik_instance.if_scope(block_idx <
                                                self.threshold_multi_core):
                    self.input_shape = (self.batch_num_front_core, self.C,
                                        self.H, self.W)
                    self.N = self.input_shape[0]
                    offset_gm =\
                        block_idx*self.batch_num_front_core*self.one_batch_size
                    self._do_operation_each_core(offset_gm)
                with self.tik_instance.else_scope():
                    self.input_shape = (self.batch_num_each_core, self.C,
                                        self.H, self.W)
                    self.N = self.input_shape[0]
                    offset_gm =\
                        self.threshold_multi_core * self.batch_num_front_core *\
                        self.one_batch_size + \
                        (block_idx - self.threshold_multi_core) *\
                        self.batch_num_each_core*self.one_batch_size
                    self._do_operation_each_core(offset_gm)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.output_gm])

        return self.tik_instance

    def _do_operation_each_core_not_align(self, offset_gm):
        n_cut_flag, h_w_cut_flag, hw_c_all_cut_flag =\
            self._do_tiling_not_align()

        if n_cut_flag is True:
            self._do_operation_each_core_n_cut_branch_not_align(offset_gm)
        elif h_w_cut_flag is True:
            self._do_operation_each_core_h_w_cut_branch_not_align(offset_gm)
        elif hw_c_all_cut_flag is True:
            self._do_operation_each_core_hw_c_cut_branch_not_align(offset_gm)

    def _do_operation_each_core_not_align_hw_little(self, offset_gm):
        # cut batch or cut c axis
        n_cut_flag, _ = self._do_tiling_not_align_hw_little()

        if n_cut_flag is True:
            self._do_operation_each_core_n_cut_branch_hw_little_not_align(offset_gm)
        else:
            self._do_operation_each_core_c_cut_branch_hw_little_not_align(offset_gm)

    def tik_instance_function_not_align(self):
        """
        do the LRN operation when H*W is not 32B align

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.one_column_size < self.alignment_standards:
            # if H*W is less than 32B, then only one core can be used
            if self.input_dtype != self.dtype_real_in_out:
                self._allocate_cast_ub()
            self._do_operation_each_core_not_align_hw_little(offset_gm=0)
        elif self.core_num == 1:
            # check if mutlti core can be used
            if self._check_c_axis_too_large():
                # multi core can not be used
                if self.input_dtype != self.dtype_real_in_out:
                    self._allocate_cast_ub()
                self.input_shape = (1, self.C, self.H, self.W)
                self.N = self.input_shape[0]
                offset_gm = 0
                self._do_operation_each_core_not_align(offset_gm)
            else:
                # multi core can be used
                core_num, hw_num_each_core, hw_num_first_core = \
                    self._get_target_core_num_batch_one()

                if (hw_num_first_core % self.alignment_standards) != 0:
                    real_first_core =\
                        (hw_num_first_core // self.alignment_standards + 1) * \
                        self.alignment_standards

                with self.tik_instance.for_range(0, core_num, block_num=core_num) as block_idx:
                    if self.input_dtype != self.dtype_real_in_out:
                        self._allocate_cast_ub()
                    with self.tik_instance.if_scope(block_idx == 0):
                        self.input_shape = (1, self.C, real_first_core, 1)
                        self.N = self.input_shape[0]
                        offset_gm = 0
                        self._do_operation_each_core_h_w_cut_branch(offset_gm)
                    with self.tik_instance.else_scope():
                        self.input_shape = (1, self.C, hw_num_each_core, 1)
                        self.N = self.input_shape[0]
                        offset_gm = hw_num_first_core + (block_idx - 1)*hw_num_each_core
                        self._do_operation_each_core_h_w_cut_branch(offset_gm)
        elif (self.core_num > 1) and (self.core_num < MAX_CORE_NUMBER):
            # each core handle only one batch
            with self.tik_instance.for_range(0, self.core_num,
                                             block_num=self.core_num) as\
                    block_idx:
                if self.input_dtype != self.dtype_real_in_out:
                    self._allocate_cast_ub()
                self.N = 1
                offset_gm = block_idx*self.one_batch_size
                self._do_operation_each_core_not_align(offset_gm)
        else:
            with self.tik_instance.for_range(0, self.core_num,
                                             block_num=self.core_num) as\
                    block_idx:
                if self.input_dtype != self.dtype_real_in_out:
                    self._allocate_cast_ub()
                with self.tik_instance.if_scope(block_idx <
                                                self.threshold_multi_core):
                    self.input_shape = (self.batch_num_front_core, self.C,
                                        self.H, self.W)
                    self.N = self.input_shape[0]
                    offset_gm =\
                        block_idx*self.batch_num_front_core*self.one_batch_size
                    self._do_operation_each_core_not_align(offset_gm)
                with self.tik_instance.else_scope():
                    self.input_shape = (self.batch_num_each_core, self.C,
                                        self.H, self.W)
                    self.N = self.input_shape[0]
                    offset_gm = self.threshold_multi_core *\
                                self.batch_num_front_core*self.one_batch_size +\
                                (block_idx - self.threshold_multi_core) *\
                                self.batch_num_each_core*self.one_batch_size
                    self._do_operation_each_core_not_align(offset_gm)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm],
                                   outputs=[self.output_gm])

        return self.tik_instance
