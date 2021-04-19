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
reduce_max_d_tik
"""
from te import tik
from te import platform as tbe_platform
from topi.cce import util


# define a scalar, value = -(2**16 - 1)
SCALAR_MIN_FP16 = -(2**16 - 1)
# define a scalar, value = -(2**32 - 1)
SCALAR_MIN_FP32 = -(2**31 - 1)
# max set_mask_int64 value
MAX_MASK_INT64 = 2**64 - 1
# max segment len
MAX_SEGMENT_LEN = 2048*7
# int32 num in 8*block
OUT_MASK = 64


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
@util.check_input_type(dict, dict, int, str)
def reduce_max_d_tik(x, index, axis, kernel_name="reduce_max_d_tik"):
    """
    Calculate the last axis of fp32 of max_d operator

    Parameters
    ----------
    x: dict
        data of input.
        source data type, support "float16", "float32"
    index: dict
        index of output.
    axis: int
        the axis value for reverse
    kernel_name: str
        kernel name

    Returns
    -------
    tik_instance
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    axis_list = axis

    _param_check(shape_x, dtype_x, axis_list, kernel_name)
    max_index = Argmax(shape_x, dtype_x, axis_list, kernel_name)

    tik_instance = max_index.max_compute()

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
    util.check_shape_rule(shape_x, max_dim=8)
    util.check_tensor_shape_size(shape_x)
    check_list = ("int32", "float32")
    util.check_dtype_rule(dtype_x.lower(), check_list)
    util.check_kernel_name(kernel_name)


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
        self.tik_instance = tik.Tik()
        self.product_core_num = \
            tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)


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
        self.ub_result_64 = None
        self.ub_result_value = None
        self.result_gm = None
        self.result_out_scalar = None
        self.result_gm_value = None
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

    def max_compute(self):
        """
        max_compute

        Parameters
        ----------

        Returns
        -------
        result : tik_instance
            self.tik_instance
        """
        # if not need split
        self.result_gm_value = self.tik_instance.Tensor(
            self.dtype_x, (self.gm_result_size,),
            name="result_gm_value",
            scope=tik.scope_gm)

        self.max_last_axis()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=(self.data_gm,),
            outputs=(self.result_gm_value,))
        return self.tik_instance

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

    def max_last_axis(self):
        """
        scedule then do last axis

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
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
                self.compute_max_last_axis(n_i, core_segment, core_segment)

            if core_segment_tail != 0:
                with self.tik_instance.if_scope(n_i < (core_num_used - 1)):
                    self.compute_max_last_axis(n_i, core_segment,
                                               core_segment)
                with self.tik_instance.else_scope():
                    self.compute_max_last_axis(n_i, core_segment_tail,
                                               core_segment)

    def compute_max_last_axis(self, n_i, core_segment, segment_core):
        """
        compute max when do last axis

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
        self.ub_result_64 = self.tik_instance.Tensor(
            self.dtype_x, (64,),
            name="ub_result_8",
            scope=tik.scope_ubuf)
        self.ub_result_value = self.tik_instance.Tensor(
            self.dtype_x, (MAX_SEGMENT_LEN,),
            name="ub_result_value",
            scope=tik.scope_ubuf)

        def _run(segment_len, segment_index):
            with self.tik_instance.for_range(0, segment_len) as core_i:
                index = core_i + MAX_SEGMENT_LEN * segment_index
                offset = n_i * segment_core + index
                self.result_out_scalar = self.tik_instance.Scalar(self.dtype_x)

                argmax_func = self.do_max_last_axis_fp32
                self.result_out_scalar.set_as(SCALAR_MIN_FP32)
                self.tik_instance.vector_dup(self.data_each_vector,
                                             self.ub_result_64,
                                             SCALAR_MIN_FP32, 1, 1, 8)
                if loop_times != 0:
                    thread_num = 1
                    if loop_times > 2:
                        thread_num = 2
                    with self.tik_instance.for_range(
                            0, loop_times, thread_num=thread_num) as loop:
                        argmax_func(ub_buf_size, loop, offset)
                if align_flag:
                    argmax_func(over_size, loop_times, offset)
                self.get_one_from_64()
                self.result_out_scalar.set_as(self.ub_result_64[0])
                self.ub_result_value[core_i] = self.result_out_scalar
            gm_out_offset = n_i * segment_core + \
                            MAX_SEGMENT_LEN * segment_index
            out_nbust = _get_ceil_int(segment_len, self.data_each_block)
            self.tik_instance.data_move(self.result_gm_value[gm_out_offset],
                                        self.ub_result_value, 0, 1,
                                        out_nbust, 0, 0)

        _loop_segment = core_segment // MAX_SEGMENT_LEN
        _loop_segment_tail = core_segment % MAX_SEGMENT_LEN
        with self.tik_instance.for_range(
                0, _loop_segment) as _loop:
            _run(MAX_SEGMENT_LEN, _loop)
        if _loop_segment_tail != 0:
            _run(_loop_segment_tail, _loop_segment)

    def get_half_max(self, tik_instance, vmax_ub, vector_num):
        if vector_num == 1:
            return 1
        output_num = vector_num // 2
        ub2_offset = output_num * 64

        tik_instance.vmax(64, vmax_ub,
                          vmax_ub, vmax_ub[ub2_offset],
                          output_num, 1, 1, 1, 8, 8, 8)

        if vector_num % 2 == 1:
            tik_instance.vmax(64, vmax_ub,
                          vmax_ub, vmax_ub[ub2_offset*2],
                          1, 1, 1, 1, 8, 8, 8)
        return output_num

    # pylint: disable=too-many-locals
    def do_max_last_axis_fp32(self, ub_buf_size, loop, n_i):
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
        ub_data = self.tik_instance.Tensor(
            self.dtype_x, _ub_size, name="ub_data", scope=tik.scope_ubuf)
        nbust = _get_ceil_int(segment, self.data_each_block)
        offset = loop * self.segment + n_i * self.axis_size
        repeat = _get_ceil_int(segment, self.data_each_vector)
        self.tik_instance.data_move(ub_data, self.data_gm[offset], 0, 1, nbust,
                                    0, 0)
        tail = ub_buf_size % self.data_each_vector
        if tail != 0:
            mask_h = 0
            mask = 2**tail - 1
            mask_l = MAX_MASK_INT64 - mask
            _offset = ub_buf_size // self.data_each_vector
            self.tik_instance.vector_dup(
                [mask_h, mask_l], ub_data[_offset * self.data_each_vector],
                SCALAR_MIN_FP32, 1, 1, 8)
        total_block = repeat
        while total_block > 1:
            total_block = self.get_half_max(self.tik_instance,
                                            ub_data,
                                            total_block)

        self.tik_instance.vmax(self.data_each_vector, self.ub_result_64,
                               ub_data, self.ub_result_64,
                               1, 1, 1, 1, 0, 8, 0)

    def get_one_from_64(self, mode=None):
        """
        get_one_from_64

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ub_block_0 = self.ub_result_64[8*0]
        ub_block_1 = self.ub_result_64[8*1]
        ub_block_2 = self.ub_result_64[8*2]
        ub_block_3 = self.ub_result_64[8*3]
        ub_block_4 = self.ub_result_64[8*4]
        ub_block_5 = self.ub_result_64[8*5]
        ub_block_6 = self.ub_result_64[8*6]
        ub_block_7 = self.ub_result_64[8*7]
        self.tik_instance.vmax(32, ub_block_0,
                               ub_block_0, ub_block_4,
                               1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(16, ub_block_4,
                               ub_block_0, ub_block_2,
                               1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(8, ub_block_0,
                               ub_block_4, ub_block_5,
                               1, 1, 1, 1, 8, 8, 8)
        index_reg = [
            self.tik_instance.Scalar(dtype=self.dtype_x) for _ in range(8)
        ]
        for i in range(7):
            index_reg[i+1].set_as(self.ub_result_64[i+1])
        ub_block_1.set_as(index_reg[1])
        ub_block_2.set_as(index_reg[2])
        ub_block_3.set_as(index_reg[3])
        ub_block_4.set_as(index_reg[4])
        ub_block_5.set_as(index_reg[5])
        ub_block_6.set_as(index_reg[6])
        ub_block_7.set_as(index_reg[7])
        self.tik_instance.vmax(1, ub_block_0,
                               ub_block_0, ub_block_1,
                               1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(1, ub_block_2,
                               ub_block_2, ub_block_3,
                               1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(1, ub_block_4,
                               ub_block_4, ub_block_5,
                               1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(1, ub_block_6,
                               ub_block_6, ub_block_7,
                               1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(1, ub_block_0,
                               ub_block_0, ub_block_2,
                               1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(1, ub_block_4,
                               ub_block_6, ub_block_4,
                               1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(1, ub_block_0,
                               ub_block_0, ub_block_4,
                               1, 1, 1, 1, 8, 8, 8)
