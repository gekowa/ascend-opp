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
util_tik_comm_func
"""
from te import tik
from te import platform as tbe_platform
from impl import common_util
from impl import constant_util


# define a scalar, value = 2**(-126), minimun num of float32 2**(-126)
SCALAR_MIN_FP32 = 2**(-126)
# define a scalar, value = 2**(50)
SCALAR_MUL_FP32 = 2**50
# define a scalar, value = 2**(26)
SCALAR_MUL2_FP32 = 2**26
# repeat max num
MAX_REPEAT_NUM = 255


def ub_offset(input_ub):
    """
    get ub offset
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


# pylint: disable=too-many-branches,too-many-statements,too-many-locals
# pylint: disable=too-many-arguments
def tik_func_vector(tik_instance, _ub, value, dup_len):
    """
    tik_func_vector

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    _ub: ub
        vector ub
    value: value
        vector value
    dup_len: int
        vector data len

    Returns
    -------
    None
    """
    do_dtype = _ub.dtype
    byte_num_one = common_util.get_data_size(do_dtype)
    block_num = constant_util.BLOCK_SIZE // byte_num_one
    vector_num = block_num*constant_util.REPEAT_STRIDE_EIGHT
    repeat = dup_len // vector_num
    repeat_tail = dup_len % vector_num
    offset = 0
    while repeat > MAX_REPEAT_NUM:
        tik_instance.vector_dup(vector_num, _ub[offset], value, MAX_REPEAT_NUM, 1, 8)
        repeat = repeat - MAX_REPEAT_NUM
        offset = offset + vector_num*MAX_REPEAT_NUM
    if repeat > 0:
        tik_instance.vector_dup(vector_num, _ub[offset], value, repeat, 1, 8)
        offset = offset + vector_num*repeat
    if repeat_tail > 0:
        tik_instance.vector_dup(repeat_tail, _ub[offset], value, 1, 1, 8)


def tik_func_vcomple(tik_instance, function, out_dst, src0, src1, copy_num,
                     dst_blk=1, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8,
                     src1_rep=8):
    """
    tik_func_vcomple
    """
    do_dtype = out_dst.dtype
    byte_num_one = common_util.get_data_size(do_dtype)
    block_num = constant_util.BLOCK_SIZE // byte_num_one
    vector_num = block_num*constant_util.REPEAT_STRIDE_EIGHT
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

    while repeat_time > MAX_REPEAT_NUM:
        tik_fun(vector_num,
                out_dst[ori_offset_dst],
                src0[ori_offset_src0],
                src1[ori_offset_src1],
                255,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)
        repeat_time = repeat_time - MAX_REPEAT_NUM
        ori_offset_dst = ori_offset_dst + MAX_REPEAT_NUM * block_num * dst_rep
        ori_offset_src0 = ori_offset_src0 + MAX_REPEAT_NUM * block_num * src0_rep
        ori_offset_src1 = ori_offset_src1 + MAX_REPEAT_NUM * block_num * src1_rep

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


def tik_func_vmuls(tik_instance, dst_ub, src_ub, value, do_len):
    """
    tik_func_vmuls
    """
    vmuls_type = dst_ub.dtype
    byte_num_one = common_util.get_data_size(vmuls_type)
    block_num = constant_util.BLOCK_SIZE // byte_num_one
    vector_num = block_num*constant_util.REPEAT_STRIDE_EIGHT
    repeat = do_len // vector_num
    repeat_tail = do_len % vector_num
    dst_offset = ub_offset(dst_ub)
    src_offset = ub_offset(src_ub)
    while repeat > MAX_REPEAT_NUM:
        tik_instance.vmuls(vector_num, dst_ub[dst_offset], src_ub[src_offset], value,
                           MAX_REPEAT_NUM, 1, 1, 8, 8)
        repeat = repeat - MAX_REPEAT_NUM
        dst_offset = dst_offset + vector_num * MAX_REPEAT_NUM
        src_offset = src_offset + vector_num * MAX_REPEAT_NUM
    if repeat > 0:
        tik_instance.vmuls(vector_num, dst_ub[dst_offset], src_ub[src_offset], value,
                           repeat, 1, 1, 8, 8)
        dst_offset = dst_offset + vector_num * repeat
        src_offset = src_offset + vector_num * repeat
    if repeat_tail > 0:
        tik_instance.vmuls(repeat_tail, dst_ub[dst_offset], src_ub[src_offset], value,
                           1, 1, 1, 8, 8)


def tik_func_vconv(tik_instance, dst_ub, src_ub, do_len, mode="", mini_mid_ub=None):
    """
    tik_func_vconv
    """
    src_dtype = src_ub.dtype
    dst_dtype = dst_ub.dtype

    def do_vconv(dst_repeat_stride, src_repeat_stride, deq_scale=None, block_num=64):
        ori_dst_offset = ub_offset(dst_ub)
        ori_src_offset = ub_offset(src_ub)
        repeat = do_len // block_num
        repeat_tail = do_len % block_num
        while repeat > MAX_REPEAT_NUM:
            tik_instance.vconv(block_num, mode, dst_ub[ori_dst_offset], src_ub[ori_src_offset],
                               MAX_REPEAT_NUM, 1, 1, dst_repeat_stride, src_repeat_stride, deqscale=deq_scale)
            repeat = repeat - MAX_REPEAT_NUM
            ori_dst_offset = ori_dst_offset + block_num*MAX_REPEAT_NUM
            ori_src_offset = ori_src_offset + block_num*MAX_REPEAT_NUM
        if repeat > 0:
            tik_instance.vconv(block_num, mode, dst_ub[ori_dst_offset], src_ub[ori_src_offset],
                               repeat, 1, 1, dst_repeat_stride, src_repeat_stride, deqscale=deq_scale)
            ori_dst_offset = ori_dst_offset + block_num*repeat
            ori_src_offset = ori_src_offset + block_num*repeat
        if repeat_tail > 0:
            tik_instance.vconv(repeat_tail, mode, dst_ub[ori_dst_offset], src_ub[ori_src_offset],
                               1, 1, 1, dst_repeat_stride, src_repeat_stride, deqscale=deq_scale)

    if src_dtype in ("float32",) and dst_dtype in ("int32",):
        cast_flag = tbe_platform.cce_conf.api_check_support("tik.vconv", "f322s32r")
        if not cast_flag:
            with tik_instance.new_stmt_scope():
                tmp_fp16_ub = tik_instance.Tensor(
                    "float16", (((do_len + 15) // 16) * 16,),
                    name="tmp_fp16_ub", scope=tik.scope_ubuf)
                tik_func_vconv(tik_instance, tmp_fp16_ub, src_ub, do_len)
                tik_func_vconv(tik_instance, dst_ub, tmp_fp16_ub, do_len, mode)
                if mode == "floor":
                    # when the product not support f322s32, will cast to fp16 and to int32, will get error
                    # ex: f32 value is 1.99998, cast int32 is 2, this step will reduce the error
                    # step 1 int32 cast to fp32_new   2.0
                    # step 2 int32_sub_fp32_value = f32_old - fp32_new
                    # step 3 int32_sub_fp32_value = 0 when int32_sub_fp32_value >= 0
                    #        int32_sub_fp32_value = 1 when int32_sub_fp32_value < 0
                    # step 4 int32 - int32_sub_fp32_value
                    if mini_mid_ub is None:
                        tmp_fp32_ub = tik_instance.Tensor(
                            "float32", (((do_len + 15) // 16) * 16,),
                            name="tmp_fp32_ub", scope=tik.scope_ubuf)
                    else:
                        tmp_fp32_ub = mini_mid_ub
                    tmp_fp32_ub_error = tik_instance.Tensor(
                        "float32", (((do_len + 15) // 16) * 16,),
                        name="tmp_fp32_ub_error", scope=tik.scope_ubuf)
                    tik_func_vconv(tik_instance, tmp_fp16_ub, dst_ub, do_len)
                    tik_func_vconv(tik_instance, tmp_fp32_ub, tmp_fp16_ub, do_len)
                    tik_func_vcomple(tik_instance, "vsub", tmp_fp32_ub_error,
                                     tmp_fp32_ub, src_ub, do_len)
                    tmp_zero = tik_instance.Tensor("float32", (8,), name="tmp_zero", scope=tik.scope_ubuf)
                    tmp_min_fp32 = tik_instance.Tensor("float32", (8,), name="tmp_minest_fp32", scope=tik.scope_ubuf)
                    tik_instance.vmuls(8, tmp_zero, tmp_zero, 0.0, 1, 1, 1, 8, 8)
                    tik_instance.vector_dup(8, tmp_min_fp32, SCALAR_MIN_FP32, 1, 1, 1)
                    tik_func_vcomple(tik_instance, "vmax", tmp_fp32_ub_error,
                                     tmp_zero, tmp_fp32_ub_error, do_len, src0_rep=0, src0_blk=0)
                    tik_func_vcomple(tik_instance, "vmin", tmp_fp32_ub_error,
                                     tmp_min_fp32, tmp_fp32_ub_error, do_len, src0_rep=0, src0_blk=0)
                    tik_func_vmuls(tik_instance, tmp_fp32_ub_error,
                                   tmp_fp32_ub_error, SCALAR_MUL_FP32, do_len)
                    tik_func_vmuls(tik_instance, tmp_fp32_ub_error,
                                   tmp_fp32_ub_error, SCALAR_MUL_FP32, do_len)
                    tik_func_vmuls(tik_instance, tmp_fp32_ub_error,
                                   tmp_fp32_ub_error, SCALAR_MUL2_FP32, do_len)
                    tik_func_vcomple(tik_instance, "vsub", tmp_fp32_ub,
                                     tmp_fp32_ub, tmp_fp32_ub_error, do_len)
                    tik_func_vconv(tik_instance, tmp_fp16_ub, tmp_fp32_ub, do_len)
                    tik_func_vconv(tik_instance, dst_ub, tmp_fp16_ub, do_len, "round")
        else:
            do_vconv(8, 8)

    elif src_dtype in ("float32",) and dst_dtype in ("float16",):
        do_vconv(4, 8)

    elif src_dtype in ("float16",) and dst_dtype in ("int32",):
        do_vconv(8, 4)

    elif src_dtype in ("int32",) and dst_dtype in ("float16",):
        do_vconv(4, 8, 1.0)

    elif src_dtype in ("float16",) and dst_dtype in ("float32",):
        do_vconv(8, 4)

    elif src_dtype in ("int32",) and dst_dtype in ("float32",):
        cast_flag = tbe_platform.cce_conf.api_check_support("tik.vconv", "s322f32")
        if not cast_flag:
            with tik_instance.new_stmt_scope():
                tmp_fp16_ub = tik_instance.Tensor(
                    "float16", (((do_len + 15) // 16) * 16,),
                    name="tmp_fp16_ub", scope=tik.scope_ubuf)
                tik_func_vconv(tik_instance, tmp_fp16_ub, src_ub, do_len)
                tik_func_vconv(tik_instance, dst_ub, tmp_fp16_ub, do_len)
        else:
            do_vconv(8, 8)

