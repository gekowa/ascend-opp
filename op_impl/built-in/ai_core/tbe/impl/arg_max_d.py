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
arg_max_d
"""
# pylint: disable=too-many-lines
from te import tik
from te import platform as tbe_platform
from te.utils.op_utils import *


# define a scalar, value = -(2**16 - 1)
SCALAR_MIN_FP16 = -(2 ** 16 - 1)
# define a scalar, value = -(2**32 - 1)
SCALAR_MIN_FP32 = -(2 ** 31 - 1)
# max set_mask_int64 value
MAX_MASK_INT64 = 2 ** 64 - 1
# max segment len
MAX_SEGMENT_LEN = 2048 * 4
# int32 num in 8*block
OUT_MASK = 64
# 0101 mask value
MASK_0_1 = 6148914691236517205
# max int32 output num
OUT_MAX_NUM = 2048 * 4


def _get_ceil_int(int1, int2):
    """Get Ceil Int

    Parameters
    ----------
    int1: int
        input int 1
    int2: int
        input int 2

    Returns
    -------
    ceil_int: int
    """
    _result = int1 // int2
    if int1 % int2 == 0:
        ceil_int = _result
    else:
        ceil_int = _result + 1

    return ceil_int


# pylint: disable=unused-argument,invalid-name,useless-object-inheritance
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT,
                 REQUIRED_ATTR_INT, KERNEL_NAME)
def arg_max_d(x, y, dimension, kernel_name="arg_max_d"):
    """
    Generate arg_max_d operator use arg_max_d

    Parameters
    ----------
    x: dict
        data of input.
        source data type, support "float16", "float32"
    y: dict
        index of output.
    dimension: int
        the axis value for reverse
    kernel_name: str
        kernel name, default value is "arg_max_d"

    Returns
    -------
    tik_instance
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    axis_list = dimension

    _param_check(shape_x, dtype_x, axis_list, kernel_name)
    max_index = Argmax(shape_x, dtype_x, axis_list, kernel_name)

    tik_instance = max_index.argmax_compute()

    return tik_instance


def _param_check(shape_x, dtype_x, axis, kernel_name):
    """check param

    Parameters
    ----------
    shape_x: list
        input shape
    dtype_x: str
        input dtype
    axis: int
        axis int num
    kernel_name: str
        kernel_name string

    Returns
    -------
    None
    """
    dim_num = len(shape_x)
    if axis < -dim_num or axis >= dim_num:
        raise RuntimeError("Axis value out of range,"
                           " must be in -len(shape) and len(shape)")

    check_shape(shape_x, param_name="x")
    check_list = ("float16", "float32")
    check_dtype(dtype_x.lower(), check_list, param_name="x")


class ArgmaxBase(object):
    """
       Function: use to store argmax base parameters
    """

    def __init__(self, shape_x, dtype_x, axis, kernel_name):
        """
        init argmax base parameters

        Parameters
        ----------
        shape_x: list
            shape of input x
        dtype_x: str
            dtype_x of input x
        axis: int
            process axis
        kernel_name: str
            kernel_name

        Returns
        -------
        None
        """
        self.tik_instance = None
        self.product_core_num = 0
        self.shape_x = list(shape_x)
        self.dtype_x = dtype_x
        self.axis = axis
        self.kernel_name = kernel_name
        self.set_tik_product()

    def get_instance(self):
        """
        init argmax  parameters

        Parameters
        ----------
        None
        Returns
        -------
        tik_instance: tik_instance
        """
        return self.tik_instance

    def set_tik_product(self):
        """
        init argmax parameters

        Parameters
        ----------
        None
        Returns
        -------
        tik_instance: tik_instance
        """
        self.product_core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.tik_instance = tik.Tik()


# pylint: disable=too-many-instance-attributes
class Argmax(ArgmaxBase):
    """
       Function: use to store argmax schedule parameters
    """

    def __init__(self, shape_x, dtype_x, axis, kernel_name):
        """
        init Argmax base parameters

        Parameters
        ----------
        shape_x:
        dtype_x:
        axis:
        Returns
        -------
        None
        """
        self.result_gm = None
        self.ub_result_int32 = None
        self.result_int32 = None
        self.result_float32 = None
        self.result_out_scalar = None
        super(Argmax, self).__init__(shape_x, dtype_x, axis, kernel_name)
        self.dtype_x = dtype_x
        dtype_bytes_size = 2 if dtype_x == "float16" else 4
        self.data_each_block = 32 // dtype_bytes_size
        self.data_each_vector = self.data_each_block * 8
        shape_len = len(shape_x)
        axis = axis % shape_len
        # To initialize the data.
        self.argmax_axis = axis
        self.first_dim_size = 1
        self.last_dim_size = 1
        self.axis_size = 1
        self.gm_result_size = 0
        self.full_mask = self.data_each_vector

        self.segment = MAX_SEGMENT_LEN
        self.out_mask = OUT_MASK

        self.c_align_ubsize = shape_x[-1]
        if axis < len(self.shape_x) - 1:
            i = 0
            while i < axis:
                self.first_dim_size = self.first_dim_size * shape_x[i]
                i = i + 1
            self.axis_size = shape_x[axis]
            i = axis + 1
            while i < len(shape_x):
                self.last_dim_size = self.last_dim_size * shape_x[i]
                i = i + 1
            self.gm_result_size = self.first_dim_size * self.last_dim_size
            self.repeat_times = \
                (self.last_dim_size * dtype_bytes_size + 255) // 256

        else:
            i = 0
            while i < len(shape_x) - 1:
                self.first_dim_size = self.first_dim_size * shape_x[i]
                i = i + 1
            self.axis_size = shape_x[axis]
            self.repeat_times = \
                (self.axis_size * dtype_bytes_size + 255) // 256
            self.gm_result_size = \
                self.first_dim_size + 2 * self.repeat_times + 15

        self.thread_num = 1
        if self.first_dim_size != 1:
            self.thread_num = 2

        self.data_gm = self.tik_instance.Tensor(
            self.dtype_x,
            (self.first_dim_size * self.axis_size * self.last_dim_size,),
            name="data_gm",
            scope=tik.scope_gm)

    def argmax_compute(self):
        """
        argmax_compute

        Parameters
        ----------

        Returns
        -------
        result : tik_instance
            self.tik_instance
        """
        # if not need split
        self.result_gm = self.tik_instance.Tensor(
            "int32", (self.gm_result_size,),
            name="result_gm",
            scope=tik.scope_gm)

        if self.argmax_axis < len(self.shape_x) - 1:
            self.argmax_not_last_axis()
        else:
            self.argmax_last_axis()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.data_gm,),
            outputs=(self.result_gm,))
        return self.tik_instance

    def get_cut_info_by_lastdims(self, dim_size, core_num):
        """
        get_cut_info_by_lastdims

        Parameters
        ----------
        dim_size : int
            dim_size
        core_num : int
            core_num

        Returns
        -------
        result : list
            [core_used, core_seg]
        """
        core_seg = _get_ceil_int(dim_size, core_num)
        core_seg = _get_ceil_int(core_seg, self.data_each_vector)
        core_seg = self.data_each_vector * core_seg
        core_used = _get_ceil_int(dim_size, core_seg)

        return core_used, core_seg

    # pylint: disable=too-many-locals
    def argmax_not_last_axis(self):
        """
        scedule for argmax_not_last_axis

        Parameters
        ----------

        Returns
        -------
        None
        """
        core_number = self.product_core_num
        # core size 1
        core_one, core_one_seg = \
            self.get_cut_info_by_lastdims(self.last_dim_size,
                                          core_number)
        offset = 0
        if self.first_dim_size >= core_one:
            segment_loop = self.last_dim_size // self.segment
            segment_tail = self.last_dim_size % self.segment
            segment_tail_data = segment_tail
            # calcu tail
            if segment_tail % self.data_each_block != 0 and segment_loop != 0:
                segment_tail_data = \
                    (segment_tail // self.data_each_block) * \
                    self.data_each_block + \
                    (self.data_each_block
                     if segment_tail % self.data_each_block != 0 else 0)
                offset = 0 + segment_tail - segment_tail_data

            if segment_tail != 0 and segment_tail < self.data_each_block and \
                    segment_loop == 0:
                core_number = 1
            core_number_all = self.first_dim_size
            core_loop = core_number_all // core_number
            core_over = core_number_all - (core_loop * core_number)

            with self.tik_instance.for_range(
                    0, core_number, block_num=core_number) as num_core_i:
                with self.tik_instance.for_range(0, core_loop) as num_core_j:
                    first_i = core_loop * num_core_i + num_core_j
                    self.compute_argmax_not_last_axis_cut_by_first_dim(
                        first_i, segment_loop, offset,
                        segment_tail, segment_tail_data)
                if core_over != 0:
                    with self.tik_instance.if_scope(num_core_i < core_over):
                        first_i = core_loop * core_number + num_core_i
                        self.compute_argmax_not_last_axis_cut_by_first_dim(
                            first_i, segment_loop, offset,
                            segment_tail, segment_tail_data)
        else:
            core_tail = core_one_seg * core_one - self.last_dim_size
            with self.tik_instance.for_range(0, self.first_dim_size) as num_i:
                if core_tail == 0:
                    with self.tik_instance.for_range(
                            0, core_one, block_num=core_one) as core_id:
                        offset_in = \
                            num_i * self.axis_size * self.last_dim_size + \
                            core_id * core_one_seg
                        offset_out = num_i * self.last_dim_size + \
                                     core_id * core_one_seg
                        self.compute_argmax_not_last_axis_cut_by_last_dim(
                            core_one_seg, offset_in, offset_out)
                else:
                    with self.tik_instance.for_range(
                            0, core_one, block_num=core_one) as core_id:
                        offset_in = \
                            num_i * self.axis_size * self.last_dim_size + \
                            core_id * core_one_seg
                        offset_out = num_i * self.last_dim_size + \
                                     core_id * core_one_seg
                        with self.tik_instance.if_scope(
                                core_id < core_one - 1):
                            self.compute_argmax_not_last_axis_cut_by_last_dim(
                                core_one_seg, offset_in, offset_out)
                        with self.tik_instance.else_scope():
                            tail_data = \
                                self.last_dim_size - \
                                core_one_seg * (core_one - 1)
                            self.compute_argmax_not_last_axis_cut_by_last_dim(
                                tail_data, offset_in, offset_out)

    def compute_argmax_not_last_axis_cut_by_last_dim(self,
                                                     data_segment,
                                                     in_offset,
                                                     out_offset):
        """
        compute for last_axis

        Parameters
        ----------
        data_segment : int
            data len for process
        in_offset : int
            gm addr begin offset
        out_offset : int
            gm addr end offset

        Returns
        -------
        None
        """
        # charge function
        not_last_axis_fuc = self.get_do_not_last_function(True)

        segment_loop = data_segment // self.segment
        segment_tail = data_segment % self.segment
        with self.tik_instance.for_range(0, segment_loop) as segm_i:
            gm_in_offset = in_offset + self.segment * segm_i
            gm_out_offset = out_offset + self.segment * segm_i
            not_last_axis_fuc(self.segment, gm_in_offset, gm_out_offset)
        if segment_tail != 0:
            segment_tail_data = \
                _get_ceil_int(segment_tail, self.data_each_vector) * \
                self.data_each_vector
            offset = segment_tail_data - segment_tail
            gm_in_offset = in_offset + self.segment * segment_loop - offset
            gm_out_offset = out_offset + self.segment * segment_loop - offset
            not_last_axis_fuc(segment_tail_data, gm_in_offset, gm_out_offset)

    # pylint: disable=too-many-arguments
    def compute_argmax_not_last_axis_cut_by_first_dim(
            self, first_i, segment_loop, offset,
            segment_tail, segment_tail_data):
        """
        compute when cut by first_dim

        Parameters
        ----------
        first_i : int
            data len for process
        segment_loop : int
            gm addr begin offset
        offset : int
            gm addr end offset
        segment_tail : int
            segment_tail
        segment_tail_data :int
            segment_tail_data

        Returns
        -------
        None
        """
        # charge function
        not_last_axis_fuc = self.get_do_not_last_function()

        if segment_loop != 0:
            with self.tik_instance.for_range(0, segment_loop) as segm_i:
                gm_in_offset = first_i * self.axis_size * self.last_dim_size \
                               + segm_i * self.segment
                gm_out_offset = first_i * self.last_dim_size + \
                                segm_i * self.segment
                not_last_axis_fuc(self.segment, gm_in_offset, gm_out_offset)
        if segment_tail != 0 and segment_tail_data % 8 == 0:
            gm_in_offset = first_i * self.axis_size * \
                           self.last_dim_size + segment_loop * \
                           self.segment + offset
            gm_out_offset = first_i * self.last_dim_size + \
                            segment_loop * self.segment + offset
            not_last_axis_fuc(segment_tail_data, gm_in_offset, gm_out_offset)

        elif segment_tail != 0 and segment_tail_data > 8:
            # last_axis < segment and not 8 alagn
            pro_len = _get_ceil_int(segment_tail_data, 2)
            pro_len = _get_ceil_int(pro_len, 8) * 8
            offset = segment_tail_data - pro_len
            gm_in_offset = first_i * self.axis_size * self.last_dim_size \
                           + segment_loop * self.segment
            gm_out_offset = first_i * self.last_dim_size + \
                            segment_loop * self.segment
            not_last_axis_fuc(pro_len, gm_in_offset, gm_out_offset)
            gm_in_offset = first_i * self.axis_size * self.last_dim_size + \
                           segment_loop * self.segment + offset
            gm_out_offset = first_i * self.last_dim_size + \
                            segment_loop * self.segment + offset
            not_last_axis_fuc(pro_len, gm_in_offset, gm_out_offset)

        elif segment_tail != 0:
            # one core if last_axis < 8
            gm_in_offset = first_i * self.axis_size * self.last_dim_size + \
                           segment_loop * self.segment
            gm_out_offset = first_i * self.last_dim_size + \
                            segment_loop * self.segment
            not_last_axis_fuc(segment_tail_data, gm_in_offset, gm_out_offset)

    def get_do_not_last_function(self, last_dim_cut=False):
        """get_do_not_last_function
        """
        # charge function
        not_last_axis_fuc = self.do_not_last
        self.c_align_ubsize = self.last_dim_size
        if self.dtype_x == "float16":
            not_last_axis_fuc = self.do_not_last_fp16_default
            block_align_num = self.last_dim_size // self.data_each_block
            if self.last_dim_size % self.data_each_block == 0 and \
                    block_align_num <= 4*8 and not last_dim_cut:
                not_last_axis_fuc = self.do_not_last_fp16_align

        return not_last_axis_fuc

    def do_not_last(self, segment, gm_in_offset, gm_out_offset):
        """
        process for a segment when arg not last dim

        Parameters
        ----------
        segment : int
            data len for process
        gm_in_offset : int
            gm addr begin offset
        gm_out_offset : int
            gm addr end offset

        Returns
        -------
        None
        """
        ub_a = self.tik_instance.Tensor(
            self.dtype_x, (self.segment,), name="ub_a", scope=tik.scope_ubuf)
        ub_c = self.tik_instance.Tensor(
            "int32", (self.segment,), name="ub_c", scope=tik.scope_ubuf)
        data_segment = segment
        nbust_len = _get_ceil_int(data_segment, self.data_each_block)
        self.tik_instance.data_move(ub_a, self.data_gm[gm_in_offset], 0, 1,
                                    nbust_len, 0, 0)
        # Init out
        repeat = _get_ceil_int(data_segment, self.data_each_block * 8)
        self.tik_instance.vector_dup(OUT_MASK, ub_c, 0, _get_ceil_int(
            data_segment, OUT_MASK), 1, 8)
        thread_num = 2 if self.axis_size > 2 else 1
        with self.tik_instance.for_range(1, self.axis_size,
                                         thread_num=thread_num) as axis_i:
            ub_mask = self.tik_instance.Tensor(
                "uint64", (self.segment // OUT_MASK,),
                name="ub_mask",
                scope=tik.scope_ubuf)
            ub_b = self.tik_instance.Tensor(
                self.dtype_x, (self.segment,), name="ub_b",
                scope=tik.scope_ubuf)
            self.tik_instance.data_move(
                ub_b, self.data_gm[gm_in_offset + axis_i * self.last_dim_size],
                0, 1, nbust_len, 0, 0)

            self.tik_instance.vcmpv_lt(ub_mask, ub_a, ub_b, repeat, 1, 1, 8, 8)
            int64_num = _get_ceil_int(data_segment, OUT_MASK)
            with self.tik_instance.for_range(0, int64_num) as i:
                mask_l = self.tik_instance.Scalar("uint64")
                mask_l.set_as(ub_mask[i])
                with self.tik_instance.if_scope(mask_l != 0):
                    self.tik_instance.vector_dup([mask_l, mask_l],
                                                 ub_c[i * OUT_MASK],
                                                 axis_i, 1, 1, 8)

            self.tik_instance.vmax(self.full_mask, ub_a, ub_a, ub_b, repeat, 1,
                                   1, 1, 8, 8, 8)
        nbust_len_out = _get_ceil_int(data_segment, 8)
        self.tik_instance.data_move(self.result_gm[gm_out_offset], ub_c, 0, 1,
                                    nbust_len_out, 0, 0)

    def do_not_last_fp16_default(self, segment, gm_in_offset, gm_out_offset):
        """
        process for a segment when arg not last dim

        Parameters
        ----------
        segment : int
            data len for process
        gm_in_offset : int
            gm addr begin offset
        gm_out_offset : int
            gm addr end offset

        Returns
        -------
        None
        """
        ub_index = self.tik_instance.Tensor(
            self.dtype_x, (self.data_each_vector,), name="ub_index",
            scope=tik.scope_ubuf)

        ub_out_fp16 = self.tik_instance.Tensor(
            "float16", (self.segment,), name="ub_out_fp16",
            scope=tik.scope_ubuf)

        ub_a = self.tik_instance.Tensor(
            self.dtype_x, (self.segment,), name="ub_a", scope=tik.scope_ubuf)
        ub_c = self.tik_instance.Tensor(
            "int32", (self.segment,), name="ub_c", scope=tik.scope_ubuf)
        data_segment = segment
        nbust_len = _get_ceil_int(data_segment, self.data_each_block)
        self.tik_instance.data_move(ub_a, self.data_gm[gm_in_offset], 0, 1,
                                    nbust_len, 0, 0)
        self.tik_instance.vector_dup(self.data_each_vector, ub_out_fp16, 0,
                                     _get_ceil_int(data_segment,
                                                   self.data_each_vector),
                                     1, 8)
        # Init out
        repeat = _get_ceil_int(data_segment, self.data_each_vector)
        self.tik_instance.vector_dup(OUT_MASK, ub_c, 0, _get_ceil_int(
            data_segment, OUT_MASK), 1, 8)
        thread_num = 2 if self.axis_size > 2 else 1
        self.tik_instance.vector_dup(self.data_each_vector, ub_index,
                                     0, 1, 1, 8)
        with self.tik_instance.for_range(1, self.axis_size,
                                         thread_num=thread_num) as axis_i:
            ub_mask = self.tik_instance.Tensor(
                "uint64", (self.segment // OUT_MASK,),
                name="ub_mask",
                scope=tik.scope_ubuf)
            ub_b = self.tik_instance.Tensor(
                self.dtype_x, (self.segment,), name="ub_b",
                scope=tik.scope_ubuf)
            self.tik_instance.data_move(
                ub_b, self.data_gm[gm_in_offset + axis_i * self.last_dim_size],
                0, 1, nbust_len, 0, 0)

            self.tik_instance.vadds(self.data_each_vector, ub_index, ub_index,
                                    1, 1, 1, 1, 8, 8)

            with self.tik_instance.for_range(0, repeat) as i:
                offset = i * self.data_each_vector
                self.tik_instance.vcmpv_lt(ub_mask, ub_a[offset],
                                           ub_b[offset],
                                           1, 1, 1, 8, 8)
                self.tik_instance.vec_sel(self.data_each_vector, 0,
                                          ub_out_fp16[offset],
                                          ub_mask, ub_index,
                                          ub_out_fp16[offset],
                                          1, 8, 0, 0)

            self.tik_instance.vmax(self.full_mask, ub_a, ub_a,
                                   ub_b, repeat, 1,
                                   1, 1, 8, 8, 8)
        vec_conv_repeat = _get_ceil_int(data_segment,
                                        OUT_MASK)
        self.tik_instance.vec_conv(64, "round", ub_c,
                                   ub_out_fp16,
                                   vec_conv_repeat, 8, 4)

        nbust_len_out = _get_ceil_int(data_segment, 8)
        self.tik_instance.data_move(self.result_gm[gm_out_offset], ub_c, 0, 1,
                                    nbust_len_out, 0, 0)

    def do_not_last_fp16_align(self, segment, gm_in_offset, gm_out_offset):
        """
        process for a segment when arg not last dim

        Parameters
        ----------
        segment : int
            data len for process
        gm_in_offset : int
            gm addr begin offset
        gm_out_offset : int
            gm addr end offset

        Returns
        -------
        None
        """
        ub_index = self.tik_instance.Tensor(
            self.dtype_x, (self.data_each_vector,), name="ub_index",
            scope=tik.scope_ubuf)

        ub_c = self.tik_instance.Tensor(
            "int32", (self.segment,), name="ub_c", scope=tik.scope_ubuf)

        ub_out_fp16 = self.tik_instance.Tensor(
            "float16", (self.segment,), name="ub_out_fp16",
            scope=tik.scope_ubuf)

        data_segment = segment

        # Init out
        self.tik_instance.vector_dup(OUT_MASK, ub_c, 0,
                                     _get_ceil_int(data_segment, OUT_MASK),
                                     1, 8)
        self.tik_instance.vector_dup(self.data_each_vector, ub_out_fp16, 0,
                                     _get_ceil_int(data_segment,
                                                   self.data_each_vector),
                                     1, 8)
        axis_ub_offset = \
            _get_ceil_int(data_segment, self.data_each_vector) * \
            self.data_each_vector
        repeat = _get_ceil_int(data_segment, self.data_each_vector)
        ub_max = self.tik_instance.Tensor(
            self.dtype_x, (axis_ub_offset,),
            name="ub_max", scope=tik.scope_ubuf)

        self.tik_instance.vector_dup(self.data_each_vector, ub_max,
                                     SCALAR_MIN_FP16, repeat, 1, 8)

        self.tik_instance.vector_dup(self.data_each_vector, ub_index,
                                     -1, 1, 1, 8)

        # calcu para
        max_axis_len = self.segment // axis_ub_offset
        axis_loop = self.axis_size // max_axis_len
        axis_tail = self.axis_size % max_axis_len
        nbust_len = \
            _get_ceil_int(data_segment, self.data_each_block)
        des_copy_stride = \
            _get_ceil_int(data_segment, self.data_each_vector) * \
            self.data_each_vector \
            - data_segment
        des_copy_stride = des_copy_stride // self.data_each_block

        def _run_one_sigment(axis_idx, axis_len):
            ub_a = self.tik_instance.Tensor(
                self.dtype_x, (self.segment,), name="ub_a",
                scope=tik.scope_ubuf)
            # copy all in
            self.tik_instance.data_move(
                ub_a, self.data_gm[gm_in_offset + axis_idx*data_segment],
                0, axis_len, nbust_len, 0, des_copy_stride)
            with self.tik_instance.for_range(0, axis_len) as axis_i:
                self.tik_instance.vadds(self.data_each_vector, ub_index,
                                        ub_index, 1, 1, 1, 1, 8, 8)
                ub_mask = self.tik_instance.Tensor(
                    "uint64", (self.segment // OUT_MASK,),
                    name="ub_mask",
                    scope=tik.scope_ubuf)
                _axis_offset = axis_ub_offset*axis_i
                with self.tik_instance.for_range(0, repeat) as i:
                    offset = i * self.data_each_vector
                    self.tik_instance.vcmpv_lt(ub_mask, ub_max[offset],
                                               ub_a[offset + _axis_offset],
                                               1, 1, 1, 8, 8)
                    self.tik_instance.vec_sel(self.data_each_vector, 0,
                                              ub_out_fp16[offset],
                                              ub_mask, ub_index,
                                              ub_out_fp16[offset],
                                              1, 8, 0, 0)

                self.tik_instance.vmax(self.full_mask, ub_max, ub_max,
                                       ub_a[_axis_offset], repeat, 1,
                                       1, 1, 8, 8, 8)

        if axis_loop > 1:
            thread_num = 2
        else:
            thread_num = 1
        if axis_loop != 0:
            with self.tik_instance.for_range(
                    0, axis_loop, thread_num=thread_num) as _axis_loop:
                input_axis_offset = _axis_loop*max_axis_len
                _run_one_sigment(input_axis_offset, max_axis_len)
        if axis_tail != 0:
            _run_one_sigment(axis_loop*max_axis_len, axis_tail)

        vec_conv_repeat = _get_ceil_int(data_segment,
                                        OUT_MASK)
        self.tik_instance.vec_conv(64, "round", ub_c,
                                   ub_out_fp16,
                                   vec_conv_repeat, 8, 4)
        nbust_len_out = _get_ceil_int(data_segment, 8)
        self.tik_instance.data_move(self.result_gm[gm_out_offset], ub_c, 0, 1,
                                    nbust_len_out, 0, 0)

    def get_tiling_info(self):
        """
        get_tiling_info when arg with last dim

        Parameters
        ----------
        None

        Returns
        -------
        result : list
            buf_size, loop_times, over_size, align_flag
        """
        if self.dtype_x == "float16":
            self.segment = MAX_SEGMENT_LEN * 3
        segment_size = self.segment
        align_flag = ((self.c_align_ubsize % segment_size) != 0)
        if segment_size <= self.c_align_ubsize:
            buf_size = segment_size
            loop_times = self.c_align_ubsize // segment_size
            over_size = self.c_align_ubsize - (loop_times * segment_size)
        else:
            loop_times = 0
            buf_size = self.c_align_ubsize
            over_size = buf_size

        return buf_size, loop_times, over_size, align_flag

    def argmax_last_axis(self):
        """
        scedule then do last axis

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # charge use compute function
        compute_fuction = self.compute_argmax_last_axis
        if self.dtype_x == "float16" and self.c_align_ubsize < MAX_SEGMENT_LEN:
            compute_fuction = self.compute_argmax_last_axis_copy_one_time
            if self.c_align_ubsize <= self.data_each_vector*2:
                compute_fuction = self.compute_argmax_last_axis_fp16_more_dims

        # calcu core policy
        core_number = self.product_core_num
        core_number_all = self.first_dim_size
        if core_number_all < 8:
            core_number = 1
        core_segment = core_number_all // core_number
        if core_segment == 0:
            core_segment = 1
        core_segment = _get_ceil_int(core_segment, 8) * 8
        core_num_used = _get_ceil_int(core_number_all, core_segment)
        core_segment_tail = core_number_all % core_segment
        with self.tik_instance.for_range(
                0, core_num_used, block_num=core_num_used) as n_i:
            if core_segment_tail == 0:
                compute_fuction(n_i, core_segment, core_segment)

            if core_segment_tail != 0:
                with self.tik_instance.if_scope(n_i < (core_num_used - 1)):
                    compute_fuction(n_i, core_segment, core_segment)
                with self.tik_instance.else_scope():
                    compute_fuction(n_i, core_segment_tail, core_segment)

    # pylint: disable=too-many-locals,too-many-statements
    def compute_argmax_last_axis(self, n_i, core_segment, segment_core):
        """
        compute arg when do last axis

        Parameters
        ----------
        n_i : int
            the first loop index
        core_segment : int
            segment process len
        segment_core : int
            the total segment index

        Returns
        -------
        None
        """
        ub_buf_size, loop_times, over_size, align_flag = self.get_tiling_info()
        self.ub_result_int32 = self.tik_instance.Tensor(
            "int32", (MAX_SEGMENT_LEN,),
            name="ub_result_int32",
            scope=tik.scope_ubuf)

        def _run(segment_len, segment_index):
            with self.tik_instance.for_range(0, segment_len) as core_i:
                index = core_i + MAX_SEGMENT_LEN * segment_index
                offset = n_i * segment_core + index
                self.result_int32 = self.tik_instance.Scalar("int32")
                self.result_int32.set_as(0)
                self.result_float32 = self.tik_instance.Scalar("float32")
                self.result_float32.set_as(SCALAR_MIN_FP32)
                self.result_out_scalar = self.tik_instance.Scalar(self.dtype_x)
                if self.dtype_x == "float16":
                    argmax_func = self.do_argmax_last_axis_fp16_default
                    self.result_out_scalar.set_as(SCALAR_MIN_FP16)
                else:
                    argmax_func = self.do_argmax_last_axis_fp32
                    self.result_out_scalar.set_as(SCALAR_MIN_FP32)
                if loop_times != 0:
                    thread_num = 1
                    if loop_times > 2:
                        thread_num = 2
                    with self.tik_instance.for_range(
                            0, loop_times, thread_num=thread_num) as loop:
                        argmax_func(ub_buf_size, loop, offset)
                if align_flag:
                    argmax_func(over_size, loop_times, offset)
                self.ub_result_int32[core_i] = self.result_int32
            gm_out_offset = n_i * segment_core + \
                            MAX_SEGMENT_LEN * segment_index
            out_nbust = _get_ceil_int(segment_len, 8)
            self.tik_instance.data_move(self.result_gm[gm_out_offset],
                                        self.ub_result_int32, 0, 1,
                                        out_nbust, 0, 0)

        _loop_segment = core_segment // MAX_SEGMENT_LEN
        _loop_segment_tail = core_segment % MAX_SEGMENT_LEN
        with self.tik_instance.for_range(
                0, _loop_segment) as _loop:
            _run(MAX_SEGMENT_LEN, _loop)
        if _loop_segment_tail != 0:
            _run(_loop_segment_tail, _loop_segment)

    def do_argmax_last_axis_fp16_default(self, ub_buf_size, loop, n_i):
        """
        do arg in one segment fo float16

        Parameters
        ----------
        ub_buf_size : int
            process len
        loop : int
            segment index in one core
        n_i : int
            the first loop index

        Returns
        -------
        None
        """
        ub_result = self.tik_instance.Tensor(
            self.dtype_x, (_get_ceil_int(ub_buf_size, self.data_each_vector) *
                           self.data_each_vector,),
            name="ub_result",
            scope=tik.scope_ubuf)
        ub_result_int32 = self.tik_instance.Tensor(
            "int32", (16,), name="ub_result_int32", scope=tik.scope_ubuf)
        ub_data = self.tik_instance.Tensor(
            self.dtype_x, (_get_ceil_int(ub_buf_size, self.data_each_vector) *
                           self.data_each_vector,),
            name="ub_data",
            scope=tik.scope_ubuf)
        offset = loop * self.segment + n_i * self.axis_size
        self.tik_instance.data_move(ub_data, self.data_gm[offset], 0, 1,
                                    _get_ceil_int(ub_buf_size,
                                                  self.data_each_block), 0, 0)

        def _calu_mask_by_one_zero(_len):
            _mask_h, _mask_l = 0, 0
            if _len > 32:
                _mask_l = MASK_0_1
                for i, _ in enumerate(range(_len - 32)):
                    _mask_h = _mask_h + 2 ** (2 * i)
            else:
                _mask_h = 0
                for i, _ in enumerate(range(_len)):
                    _mask_l = _mask_l + 2 ** (2 * i)
            return _mask_h, _mask_l

        def _get_tail_mask(tail_len):
            if tail_len <= OUT_MASK:
                mask = 2 ** tail_len - 1
                _mask_h = MAX_MASK_INT64
                _mask_l = MAX_MASK_INT64 - mask
            else:
                _mask_l = 0
                mask = 2 ** (tail_len - OUT_MASK) - 1
                _mask_h = MAX_MASK_INT64 - mask
            return _mask_h, _mask_l

        tail = ub_buf_size % self.data_each_vector
        if tail != 0:
            mask_h, mask_l = _get_tail_mask(tail)
            _offset = ub_buf_size // (self.data_each_vector)
            self.tik_instance.vector_dup(
                [mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                SCALAR_MIN_FP16, 1, 1, 8)

        repeat_times = _get_ceil_int(ub_buf_size, self.data_each_vector)
        self.tik_instance.vcmax(self.data_each_vector, ub_result, ub_data,
                                repeat_times, 1, 1, 8)

        if repeat_times > 64:
            _repeat_times = _get_ceil_int(repeat_times, 64)
            _repeat_tail = (repeat_times * 2) % self.data_each_vector
            if _repeat_tail != 0:
                mask_h, mask_l = _get_tail_mask(_repeat_tail)
                _offset = repeat_times * 2 // self.data_each_vector
                self.tik_instance.vector_dup(
                    [mask_h, mask_l],
                    ub_result[_offset * self.data_each_vector],
                    SCALAR_MIN_FP16, 1, 1, 8)
            repeat_times = _get_ceil_int(repeat_times, 64)
            ub_second_result = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(repeat_times, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_second_result",
                scope=tik.scope_ubuf)
            self.tik_instance.vcmax([MASK_0_1,
                                     MASK_0_1],
                                    ub_second_result, ub_result,
                                    _repeat_times, 1, 1, 8)

            ub_third_result = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(_repeat_times, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_third_result",
                scope=tik.scope_ubuf)

            _mask = _calu_mask_by_one_zero(repeat_times % 64)
            self.tik_instance.vcmax(_mask,
                                    ub_third_result, ub_second_result,
                                    1, 1, 1, 8)
            third_max_index = self.tik_instance.Scalar("uint16")
            third_max_index.set_as(ub_third_result[1])
            second_max_index = self.tik_instance.Scalar("uint16")
            second_max_index.set_as(ub_second_result[third_max_index + 1])
            last_max_index = self.tik_instance.Scalar("uint16")
            last_max_index.set_as(
                ub_result[third_max_index * 64 + second_max_index + 1])
            max_index = self.tik_instance.Scalar("uint16")
            max_index.set_as(
                third_max_index * 64 * 64 + second_max_index * 64 + \
                last_max_index)

        elif repeat_times > 1:
            _repeat_tail = repeat_times % 64
            _mask = _calu_mask_by_one_zero(_repeat_tail)
            if _repeat_tail == 0:
                _mask = [MASK_0_1, MASK_0_1]
            ub_second_result = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(repeat_times, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_second_result",
                scope=tik.scope_ubuf)
            self.tik_instance.vcmax(_mask,
                                    ub_second_result, ub_result,
                                    1, 1, 1, 8)
            second_max_index = self.tik_instance.Scalar("uint16")
            second_max_index.set_as(ub_second_result[1])
            last_max_index = self.tik_instance.Scalar("uint16")
            last_max_index.set_as(ub_result[second_max_index + 1])
            max_index = self.tik_instance.Scalar("uint16")
            max_index.set_as(second_max_index * 64 + last_max_index)
        else:
            max_index = self.tik_instance.Scalar("uint16")
            max_index.set_as(ub_result[1])

        max_index_int32 = self.tik_instance.Scalar("int32")
        max_index_int32.set_as(max_index)
        ub_result_cmp = self.tik_instance.Tensor(
            self.dtype_x, (self.data_each_vector,),
            name="ub_result_cmp",
            scope=tik.scope_ubuf)
        ub_result_cmp[0].set_as(self.result_out_scalar)
        ub_result_cmp[1].set_as(ub_data[max_index_int32])
        ub_result_int32[0].set_as(self.result_int32)
        ub_result_int32[1].set_as(max_index_int32 + loop * self.segment)
        self.tik_instance.vcmax(2, ub_result_cmp, ub_result_cmp,
                                1, 1, 1, 8)
        max_index1 = self.tik_instance.Scalar("uint16")
        max_index1.set_as(ub_result_cmp[1])
        self.result_int32.set_as(ub_result_int32[max_index1])
        self.result_out_scalar.set_as(ub_result_cmp[0])

    def compute_argmax_last_axis_fp16_more_dims(self, n_i, core_segment,
                                                segment_core):
        """
        compute arg when do last axis

        Parameters
        ----------
        n_i : int
            the first loop index
        core_segment : int
            segment process len
        segment_core : int
            the total segment index

        Returns
        -------
        None
        """
        self.get_tiling_info()
        self.ub_result_int32 = self.tik_instance.Tensor(
            "int32", (MAX_SEGMENT_LEN,),
            name="ub_result_int32",
            scope=tik.scope_ubuf)

        def _run(segment_len, segment_index, axis_size_sigment):
            offset = n_i * segment_core + axis_size_sigment * segment_index
            if self.c_align_ubsize >= self.data_each_vector:
                argmax_func = self.do_argmax_last_axis_fp16_more_vector
            else:
                argmax_func = self.do_argmax_last_axis_fp16_less_vector

            argmax_func(offset, segment_len)
            gm_out_offset = \
                n_i * segment_core + axis_size_sigment * segment_index
            out_nbust = _get_ceil_int(segment_len, 8)
            self.tik_instance.data_move(self.result_gm[gm_out_offset],
                                        self.ub_result_int32, 0, 1,
                                        out_nbust, 0, 0)

        align_num = get_align_num(self.c_align_ubsize, self.data_each_block)
        # calcu axis size one time
        if self.c_align_ubsize >= self.data_each_vector:
            vector_size_repeat = _get_ceil_int(self.c_align_ubsize,
                                               self.data_each_vector)
            vector_size = vector_size_repeat * self.data_each_vector
            input_total_size = self.segment
            axis_size_one_time = input_total_size // vector_size
            axis_size_one_time = axis_size_one_time * align_num
        else:
            vector_size = _get_ceil_int(self.c_align_ubsize,
                                        self.data_each_block)
            vector_size = vector_size * self.data_each_block
            input_total_size = self.segment
            axis_size_one_time = input_total_size // vector_size
            axis_size_one_time = min(axis_size_one_time, (255 // 8) * 8)
            axis_size_one_time = axis_size_one_time * align_num

        _loop_segment = core_segment // axis_size_one_time
        _loop_segment_tail = core_segment % axis_size_one_time

        if _loop_segment != 0:
            if _loop_segment >= 2 and align_num == 1:
                thread_num = 2
            else:
                thread_num = 1
            with self.tik_instance.for_range(
                    0, _loop_segment, thread_num=thread_num) as _loop:
                _run(axis_size_one_time, _loop, axis_size_one_time)
        if _loop_segment_tail != 0:
            _run(_loop_segment_tail, _loop_segment, axis_size_one_time)

    def do_argmax_last_axis_fp16_less_vector(self, offset, segment_len):
        """do arg in one segment fo float16
        """
        align_loop = get_align_num(self.c_align_ubsize,
                                   self.data_each_block)
        ub_data = self.tik_instance.Tensor(
            self.dtype_x, (self.segment,),
            name="ub_data",
            scope=tik.scope_ubuf)
        if align_loop != 1:
            ub_data_1 = self.tik_instance.Tensor(
                self.dtype_x, (self.segment,),
                name="ub_data_1",
                scope=tik.scope_ubuf)
            ping_pang_ub = [ub_data, ub_data_1]
        else:
            ping_pang_ub = [ub_data, ub_data]

        result_int32 = self.tik_instance.Scalar("int32")
        segment_align_num = segment_len // align_loop
        segment_align_tail = segment_len % align_loop

        for align_idx, _ in enumerate(range(align_loop)):
            copy_ub = ping_pang_ub[align_idx % 2]
            nburst_len = _get_ceil_int(self.c_align_ubsize,
                                       self.data_each_block)
            if align_idx < segment_align_tail:
                nburst = segment_align_num + 1
            else:
                nburst = segment_align_num

            if nburst < 1:
                break

            src_nburst_stride = \
                _get_ceil_int(self.c_align_ubsize*align_loop,
                              self.data_each_block) - nburst_len
            repeat_vcmax = nburst
            des_nburst_stride = 0
            repeat_block = nburst_len
            if align_loop == 1:
                nburst_len = nburst_len*nburst
                nburst = 1
                repeat_vcmax = segment_len

            input_gm_offset = (offset + align_idx)*self.c_align_ubsize
            self.tik_instance.data_move(copy_ub,
                                        self.data_gm[input_gm_offset],
                                        0, nburst, nburst_len,
                                        src_nburst_stride, des_nburst_stride)
            # get one axis vcmax
            self.tik_instance.vcmax(self.c_align_ubsize, copy_ub, copy_ub,
                                    repeat_vcmax, 1, 1,
                                    repeat_block)
            with self.tik_instance.for_range(0, repeat_vcmax) as _vcmax_idx:
                last_max_index = self.tik_instance.Scalar("uint16")
                last_max_index.set_as(copy_ub[_vcmax_idx*2 + 1])
                result_int32.set_as(last_max_index)
                self.ub_result_int32[_vcmax_idx*align_loop
                                     + align_idx].set_as(result_int32)

    def do_argmax_last_axis_fp16_more_vector(self, offset, segment_len):
        """do arg in one segment fo float16
        """
        align_loop = get_align_num(self.c_align_ubsize,
                                   self.data_each_block)
        ub_data = self.tik_instance.Tensor(
            self.dtype_x, (self.segment,),
            name="ub_data",
            scope=tik.scope_ubuf)
        ub_second_result = self.tik_instance.Tensor(
            self.dtype_x,
            (_get_ceil_int(segment_len*2, self.data_each_vector) *
             self.data_each_vector,),
            name="ub_second_result",
            scope=tik.scope_ubuf)

        if align_loop != 1:
            ub_data_1 = self.tik_instance.Tensor(
                self.dtype_x, (self.segment,),
                name="ub_data_1",
                scope=tik.scope_ubuf)
            ub_second_result_1 = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(segment_len*2, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_second_result_1",
                scope=tik.scope_ubuf)
            ping_pang_ub = [ub_data, ub_data_1]
            ping_pang_ub_second = [ub_second_result, ub_second_result_1]
        else:
            ping_pang_ub = [ub_data, ub_data]
            ping_pang_ub_second = [ub_second_result, ub_second_result]

        vector_size_repeat = _get_ceil_int(self.c_align_ubsize,
                                           self.data_each_vector)
        vector_size = vector_size_repeat * self.data_each_vector

        result_int32 = self.tik_instance.Scalar("int32")

        segment_align_num = segment_len // align_loop
        segment_align_tail = segment_len % align_loop

        def _calu_mask_by_repeat_times(_len):
            _str = "0000000000000000"*(8 - _len) + "0000000000000001"*_len
            _mask_h = int(_str[0:64], 2)
            _mask_l = int(_str[64:128], 2)
            return _mask_h, _mask_l

        def _get_tail_mask(tail_len):
            if tail_len <= OUT_MASK:
                mask = 2 ** tail_len - 1
                mask_h = MAX_MASK_INT64
                mask_l = MAX_MASK_INT64 - mask
            else:
                mask_l = 0
                mask = 2 ** (tail_len - OUT_MASK) - 1
                mask_h = MAX_MASK_INT64 - mask
            return mask_h, mask_l

        for align_idx, _ in enumerate(range(align_loop)):
            copy_ub = ping_pang_ub[align_idx % 2]
            second_ub = ping_pang_ub_second[align_idx % 2]
            nburst_len = _get_ceil_int(self.c_align_ubsize,
                                       self.data_each_block)
            if align_idx < segment_align_tail:
                nburst = segment_align_num + 1
            else:
                nburst = segment_align_num

            if nburst < 1:
                break
            src_nburst_stride = \
                _get_ceil_int(self.c_align_ubsize*align_loop,
                              self.data_each_block) - nburst_len
            des_nburst_stride = vector_size_repeat*8 - nburst_len
            input_gm_offset = (offset + align_idx)*self.c_align_ubsize
            self.tik_instance.data_move(copy_ub,
                                        self.data_gm[input_gm_offset],
                                        0, nburst, nburst_len,
                                        src_nburst_stride, des_nburst_stride)

            tail = self.c_align_ubsize % self.data_each_vector
            if tail != 0:
                mask_h, mask_l = _get_tail_mask(tail)
                _offset = self.c_align_ubsize // self.data_each_vector
                self.tik_instance.vector_dup(
                    [mask_h, mask_l], copy_ub[_offset * self.data_each_vector],
                    SCALAR_MIN_FP16, nburst,
                    1, vector_size_repeat*8)
            repeat_times = _get_ceil_int(self.c_align_ubsize,
                                         self.data_each_vector)
            self.tik_instance.vcmax(self.data_each_vector, copy_ub, copy_ub,
                                    vector_size_repeat*nburst, 8*8, 1, 8)
            _mask = _calu_mask_by_repeat_times(repeat_times)
            self.tik_instance.vcmax(_mask,
                                    second_ub, copy_ub,
                                    nburst, 1, 8, repeat_times*8)
            with self.tik_instance.for_range(0, nburst) as out_idx:
                second_max_index = self.tik_instance.Scalar("uint16")
                second_max_index.set_as(second_ub[out_idx*2 + 1])
                last_max_index = self.tik_instance.Scalar("uint16")
                last_max_index.set_as(copy_ub[vector_size*out_idx +
                                              second_max_index*8 + 1])
                result_int32.set_as(second_max_index * 8 + last_max_index)
                self.ub_result_int32[out_idx*align_loop
                                     + align_idx].set_as(result_int32)

    def compute_argmax_last_axis_copy_one_time(self, n_i, core_segment,
                                               segment_core):
        """
        compute arg when do last axis

        Parameters
        ----------
        n_i : int
            the first loop index
        core_segment : int
            segment process len
        segment_core : int
            the total segment index

        Returns
        -------
        None
        """
        _, loop_times, over_size, align_flag = self.get_tiling_info()
        self.ub_result_int32 = self.tik_instance.Tensor(
            "int32", (MAX_SEGMENT_LEN,),
            name="ub_result_int32",
            scope=tik.scope_ubuf)

        def _run(segment_len, segment_index):
            with self.tik_instance.for_range(0, segment_len) as core_i:
                index = core_i + MAX_SEGMENT_LEN * segment_index
                offset = n_i * segment_core + index
                self.result_int32 = self.tik_instance.Scalar("int32")
                if self.dtype_x == "float16":
                    argmax_func = self.do_argmax_last_axis_fp16_copy_one_time

                if align_flag:
                    argmax_func(over_size, loop_times, offset)
                self.ub_result_int32[core_i] = self.result_int32
            gm_out_offset = \
                n_i * segment_core + MAX_SEGMENT_LEN * segment_index
            out_nbust = _get_ceil_int(segment_len, 8)
            self.tik_instance.data_move(self.result_gm[gm_out_offset],
                                        self.ub_result_int32, 0, 1,
                                        out_nbust, 0, 0)

        _loop_segment = core_segment // MAX_SEGMENT_LEN
        _loop_segment_tail = core_segment % MAX_SEGMENT_LEN
        with self.tik_instance.for_range(
                0, _loop_segment) as _loop:
            _run(MAX_SEGMENT_LEN, _loop)

        if _loop_segment_tail != 0:
            _run(_loop_segment_tail, _loop_segment)

    def do_argmax_last_axis_fp16_copy_one_time(self, ub_buf_size, loop, n_i):
        """
        do arg in one segment fo float16

        Parameters
        ----------
        ub_buf_size : int
            process len
        loop : int
            segment index in one core
        n_i : int
            the first loop index

        Returns
        -------
        None
        """
        ub_result = self.tik_instance.Tensor(
            self.dtype_x, (_get_ceil_int(ub_buf_size, self.data_each_vector) *
                           self.data_each_vector,),
            name="ub_result",
            scope=tik.scope_ubuf)

        ub_data = self.tik_instance.Tensor(
            self.dtype_x, (_get_ceil_int(ub_buf_size, self.data_each_vector) *
                           self.data_each_vector,),
            name="ub_data",
            scope=tik.scope_ubuf)
        offset = loop * self.segment + n_i * self.axis_size
        self.tik_instance.data_move(ub_data, self.data_gm[offset], 0, 1,
                                    _get_ceil_int(ub_buf_size,
                                                  self.data_each_block), 0, 0)

        def _calu_mask_by_one_zero(_len):
            _mask_h, _mask_l = 0, 0
            if _len > 32:
                _mask_l = MASK_0_1
                for i, _ in enumerate(range(_len - 32)):
                    _mask_h = _mask_h + 2 ** (2 * i)
            else:
                _mask_h = 0
                for i, _ in enumerate(range(_len)):
                    _mask_l = _mask_l + 2 ** (2 * i)
            return _mask_h, _mask_l

        def _get_tail_mask(tail_len):
            if tail_len <= OUT_MASK:
                mask = 2 ** tail_len - 1
                _mask_h = MAX_MASK_INT64
                _mask_l = MAX_MASK_INT64 - mask
            else:
                _mask_l = 0
                mask = 2 ** (tail_len - OUT_MASK) - 1
                _mask_h = MAX_MASK_INT64 - mask
            return _mask_h, _mask_l

        tail = ub_buf_size % self.data_each_vector
        if tail != 0:
            mask_h, mask_l = _get_tail_mask(tail)
            _offset = ub_buf_size // self.data_each_vector
            self.tik_instance.vector_dup(
                [mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                SCALAR_MIN_FP16, 1, 1, 8)

        repeat_times = _get_ceil_int(ub_buf_size, self.data_each_vector)
        self.tik_instance.vcmax(self.data_each_vector, ub_result, ub_data,
                                repeat_times, 1, 1, 8)

        if repeat_times > 64:
            _repeat_times = _get_ceil_int(repeat_times, 64)
            _repeat_tail = (repeat_times * 2) % self.data_each_vector
            if _repeat_tail != 0:
                mask_h, mask_l = _get_tail_mask(_repeat_tail)
                _offset = repeat_times * 2 // self.data_each_vector
                self.tik_instance.vector_dup(
                    [mask_h, mask_l],
                    ub_result[_offset * self.data_each_vector],
                    SCALAR_MIN_FP16, 1, 1, 8)
            repeat_times = _get_ceil_int(repeat_times, 64)
            ub_second_result = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(repeat_times, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_second_result",
                scope=tik.scope_ubuf)
            self.tik_instance.vcmax([MASK_0_1,
                                     MASK_0_1],
                                    ub_second_result, ub_result,
                                    _repeat_times, 1, 1, 8)

            ub_third_result = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(_repeat_times, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_third_result",
                scope=tik.scope_ubuf)

            _mask = _calu_mask_by_one_zero(repeat_times % 64)
            self.tik_instance.vcmax(_mask,
                                    ub_third_result, ub_second_result,
                                    1, 1, 1, 8)
            third_max_index = self.tik_instance.Scalar("uint16")
            third_max_index.set_as(ub_third_result[1])
            second_max_index = self.tik_instance.Scalar("uint16")
            second_max_index.set_as(ub_second_result[third_max_index + 1])
            last_max_index = self.tik_instance.Scalar("uint16")
            last_max_index.set_as(
                ub_result[third_max_index * 64 + second_max_index + 1])
            self.result_int32.set_as(
                third_max_index * 64 * 64 + second_max_index * 64 + \
                last_max_index)

        elif repeat_times > 1:
            _repeat_tail = repeat_times % 64
            _mask = _calu_mask_by_one_zero(_repeat_tail)
            if _repeat_tail == 0:
                _mask = [MASK_0_1, MASK_0_1]
            ub_second_result = self.tik_instance.Tensor(
                self.dtype_x,
                (_get_ceil_int(repeat_times, self.data_each_vector) *
                 self.data_each_vector,),
                name="ub_second_result",
                scope=tik.scope_ubuf)
            self.tik_instance.vcmax(_mask,
                                    ub_second_result, ub_result,
                                    1, 1, 1, 8)
            second_max_index = self.tik_instance.Scalar("uint16")
            second_max_index.set_as(ub_second_result[1])
            last_max_index = self.tik_instance.Scalar("uint16")
            last_max_index.set_as(ub_result[second_max_index + 1])
            self.result_int32.set_as(second_max_index * 64 + last_max_index)
        else:
            second_max_index = self.tik_instance.Scalar("uint16")
            second_max_index.set_as(ub_result[1])
            self.result_int32.set_as(second_max_index)

    # pylint: disable=too-many-locals
    def do_argmax_last_axis_fp32(self, ub_buf_size, loop, n_i):
        """
        do arg in one segment fo float32

        Parameters
        ----------
        ub_buf_size : int
            process len
        loop : int
            segment index in one core
        n_i : int
            the first loop index

        Returns
        -------
        None
        """
        segment = ub_buf_size
        _ub_size = [
            self.data_each_block * _get_ceil_int(self.segment,
                                                 self.data_each_block)
        ]
        ub_index_int32 = self.tik_instance.Tensor(
            "int32", _ub_size, name="ub_index_int32", scope=tik.scope_ubuf)
        ub_data = self.tik_instance.Tensor(
            self.dtype_x, _ub_size, name="ub_data", scope=tik.scope_ubuf)
        ub_max_64 = self.tik_instance.Tensor(
            self.dtype_x, [self.data_each_vector],
            name="ub_max_64",
            scope=tik.scope_ubuf)
        cmp_mask_ub = self.tik_instance.Tensor(
            "uint64", [
                _get_ceil_int(
                    _get_ceil_int(self.segment, self.data_each_vector),
                    self.data_each_vector) * self.data_each_vector
            ],
            name="cmp_mask_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.data_each_vector, ub_max_64,
                                     SCALAR_MIN_FP32, 1, 1, 8)
        nbust = _get_ceil_int(segment, self.data_each_block)
        offset = loop * self.segment + n_i * self.axis_size
        repeat = _get_ceil_int(segment, self.data_each_vector)
        self.tik_instance.data_move(ub_data, self.data_gm[offset], 0, 1, nbust,
                                    0, 0)
        tail = ub_buf_size % self.data_each_vector
        if tail != 0:
            mask_h = 0
            mask = 2 ** tail - 1
            mask_l = MAX_MASK_INT64 - mask
            _offset = ub_buf_size // self.data_each_vector
            self.tik_instance.vector_dup(
                [mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                SCALAR_MIN_FP32, 1, 1, 8)
        self.tik_instance.vmax(self.data_each_vector, ub_max_64, ub_data,
                               ub_max_64, repeat, 1, 1, 1, 0, 8, 0)
        self.tik_instance.vcmpv_eq(cmp_mask_ub, ub_max_64, ub_data, repeat, 1,
                                   1, 0, 8)
        self.tik_instance.vector_dup(self.data_each_vector, ub_index_int32, 0,
                                     1, 1, 8)
        with self.tik_instance.for_range(0, repeat) as i:
            index = repeat - 1 - i
            mask_l = self.tik_instance.Scalar("uint64")
            mask_l.set_as(cmp_mask_ub[index])
            with self.tik_instance.if_scope(mask_l != 0):
                self.tik_instance.vector_dup([mask_l, mask_l], ub_index_int32,
                                             index * self.data_each_vector, 1,
                                             1, 8)
        # get one value from 64
        max_value = self.tik_instance.Scalar(self.dtype_x)
        max_index = self.tik_instance.Scalar("int32")
        max_value.set_as(ub_max_64[0])
        max_index.set_as(ub_index_int32[0])
        scalar_valid = self.data_each_vector \
            if segment > self.data_each_vector else segment
        with self.tik_instance.for_range(1, scalar_valid) as i:
            max_cmp_value = self.tik_instance.Scalar(self.dtype_x)
            max_cmp_index = self.tik_instance.Scalar("int32")
            max_cmp_value.set_as(ub_max_64[i])
            max_cmp_index.set_as(ub_index_int32[i])
            with self.tik_instance.if_scope(max_cmp_value > max_value):
                max_value.set_as(ub_max_64[i])
                max_index.set_as(max_cmp_index + i)
        with self.tik_instance.if_scope(max_value > self.result_out_scalar):
            self.result_out_scalar.set_as(max_value)
            self.result_int32.set_as(max_index + loop * self.segment)


def get_align_num(dim_size, align_size):
    """get_align_num
    """
    align_num = align_size
    for i, _ in enumerate(range(align_size)):
        align_num = i + 1
        if dim_size*align_num % align_size == 0:
            break

    return align_num

