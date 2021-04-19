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
gather_v2d
"""
from te import tvm
import te.lang.dynamic
from te import tik
from te import platform as tbe_platform
from functools import reduce as functools_reduce
from topi.cce import util
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_op_params
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import check_dtype
from te.utils.error_manager import error_manager_vector


PARAMS_SIZE = 2**31 - 1
INDICES_NUM = 2**31 - 1
TILING_ARG_NUM = 32
# TILING_ARG_NUM = 19

# data type of int32
INT32 = "int32"
# data type of int64
INT64 = "int64"
# one block size takes up 32b
BLOCK_SIZE = 32

TILING_MODE_1 = 1
TILING_MODE_2 = 2
TILING_MODE_3 = 3

TYPE_LEN_DICT = {"float16": 2, "float32": 4, "int8": 1, "uint8": 1,
                 "int16": 2, "uint16": 2, "int32": 4, "uint32": 4,
                 "int64": 8, "uint64": 8}


def ceil_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    ceil value
    """
    return (value + factor - 1) // factor


def align_value(value, factor):
    """
    Alignment based on factor.

    Parameters
    ----------
    value: input number
    factor: alignment base

    Returns
    -------
    res:
    """
    return (value + factor - 1) // factor*factor


class GatherV2():
    def __init__(self, params_dict, indices_dict, axis_dict, y_dict, kernel_name):
        """
        constructor of GatherV2

        Parameters
        ----------
        params_dict: dict
            shape and dtype of input params
        indices_dict: dict
            shape and dtype of input indices
        axis_dict: dict
            shape and dtype of input axis
        y_dict: dict
            shape and dtype of output, should be same dtype as input
        kernel_name: str
            kernel name, default value is "GatherV2"

        Returns
        -------
        None
        """
        self.params_dtype = params_dict.get("dtype").lower()
        self.indices_dtype = indices_dict.get("dtype").lower()
        self.axis_dtype = axis_dict.get("dtype").lower()
        self.y_dtype = y_dict.get("dtype").lower()
        self.tiling_dtype = INT32
        dtype_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                      "uint32", "uint64", "float16", "float32")
        indices_support_dtype_list = ("int32", "int64")
        check_dtype(self.params_dtype, dtype_list, param_name="x")
        check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        check_dtype(self.axis_dtype, (INT32,), param_name="axis")
        if self.y_dtype != self.params_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "y", "x",
                                                                  self.y_dtype, self.params_dtype)

        profile = tik.Dprofile()
        self.ub_size = profile.get_unified_buffer_size()
        self.l1_size = profile.get_l1_buffer_size()
        self.core_num = profile.get_aicore_num()
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.kernel_name = kernel_name

        self.axis_shape = (1,)
        self.x_shape = (PARAMS_SIZE,)
        self.indices_shape = (INDICES_NUM,)
        self.y_shape = (PARAMS_SIZE,)

        self.params_dsize = TYPE_LEN_DICT.get(self.params_dtype)
        self.indices_dsize = TYPE_LEN_DICT.get(self.indices_dtype)
        self.block_elem = BLOCK_SIZE // self.params_dsize

        self.x = None
        self.indices = None
        self.axis = None
        self.tiling_gm = None
        self.y = None

        self.params_pre = None
        self.params_axis = None
        self.params_row = None
        self.indices_num = None

        self.cache_params = None
        self.need_core_num = None
        self.tail_process_core = None
        self.indices_num_each_core = None
        self.indices_num_remaining = None
        self.indices_loop_num = None
        self.indices_row_num_once = None
        self.indices_row_num_last = None

        self.row_num_once_ub = None
        self.row_num_once_tail_ub = None
        self.inner_loop_num = None
        self.row_num_last_ub = None
        self.row_num_last_tail_ub = None
        self.inner_loop_num_last = None

    def get_tiling_args(self, tiling_ub):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from gather_nd tiling

        Returns
        -------
        None
        """
        self.params_pre = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_pre")
        self.params_pre.set_as(tiling_ub[1])
        self.params_axis = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_axis")
        self.params_axis.set_as(tiling_ub[2])
        self.params_row = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_row")
        self.params_row.set_as(tiling_ub[3])
        self.indices_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num")
        self.indices_num.set_as(tiling_ub[4])

        self.cache_params = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="cache_params")
        self.cache_params.set_as(tiling_ub[5])
        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.need_core_num.set_as(tiling_ub[6])
        self.tail_process_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tail_process_core")
        self.tail_process_core.set_as(tiling_ub[7])
        self.indices_num_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_each_core")
        self.indices_num_each_core.set_as(tiling_ub[8])
        self.indices_num_remaining = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_remaining")
        self.indices_num_remaining.set_as(tiling_ub[9])
        self.indices_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_loop_num")
        self.indices_loop_num.set_as(tiling_ub[10])
        self.indices_row_num_once = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_row_num_once")
        self.indices_row_num_once.set_as(tiling_ub[11])
        self.indices_row_num_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_row_num_last")
        self.indices_row_num_last.set_as(tiling_ub[12])

        self.row_num_once_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_once_ub")
        self.row_num_once_ub.set_as(tiling_ub[13])
        self.row_num_once_tail_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_once_tail_ub")
        self.row_num_once_tail_ub.set_as(tiling_ub[14])
        self.inner_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="inner_loop_num")
        self.inner_loop_num.set_as(tiling_ub[15])
        self.row_num_last_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_last_ub")
        self.row_num_last_ub.set_as(tiling_ub[16])
        self.row_num_last_tail_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_last_tail_ub")
        self.row_num_last_tail_ub.set_as(tiling_ub[17])
        self.inner_loop_num_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="inner_loop_num_last")
        self.inner_loop_num_last.set_as(tiling_ub[18])

    def gather_v2_compute_tiling(self):
        """
        Main process of gather_v2

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        half_ub_size = (self.ub_size - 2*1024) // 2
        tik_instance = self.tik_instance

        with tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            # get tiling data
            tiling_ub = tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,), name="tiling_ub",
                                            scope=tik.scope_ubuf)
            tik_instance.data_move(tiling_ub, self.tiling_gm, 0,
                                   1, ceil_value(TILING_ARG_NUM * TYPE_LEN_DICT.get(self.tiling_dtype), BLOCK_SIZE),
                                   0, 0)

            tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tiling_mode")
            tiling_mode.set_as(tiling_ub[0])

            # get run info
            self.get_tiling_args(tiling_ub)

            with tik_instance.if_scope(tiling_mode == TILING_MODE_1):
                with tik_instance.new_stmt_scope():
                    self.compute_mode_1(half_ub_size, block_id)
            with tik_instance.else_scope():
                with tik_instance.if_scope(tiling_mode == TILING_MODE_2):
                    with tik_instance.new_stmt_scope():
                        # compute_mode_2
                        self.compute_mode_1(half_ub_size, block_id)
                with tik_instance.else_scope():
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_3(half_ub_size, block_id)

    def compute_mode_1(self, half_ub_size, block_id):
        """
        compute for tiling mode 1

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_dsize = self.indices_dsize
        params_dsize = self.params_dsize

        with tik_instance.if_scope(block_id < self.need_core_num):
            indices_ub = tik_instance.Tensor(self.indices_dtype, ((half_ub_size + 256) // indices_dsize,),
                                             name="indices_ub", scope=tik.scope_ubuf)
            res_ub = tik_instance.Tensor(self.params_dtype, ((half_ub_size + BLOCK_SIZE) // params_dsize,),
                                         name="res_ub", scope=tik.scope_ubuf)

            burst_len_row = ceil_value(self.params_row * params_dsize, BLOCK_SIZE)

            with tik_instance.for_range(0, self.params_pre) as pre_i:
                gm_offset_base = pre_i * self.params_axis

                # indices_num_each_core = indices_row_num_once * indices_loop_num + indices_row_num_last
                with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
                    indices_offset = block_id * self.indices_num_each_core + \
                                     indices_loop_i * self.indices_row_num_once
                    # copy indices data to ub from gm
                    tik_instance.data_move(indices_ub, self.indices[indices_offset], 0, 1,
                                           ceil_value(self.indices_row_num_once * indices_dsize, BLOCK_SIZE), 0, 0)

                    # indices_row_num_once = row_num_once_ub * inner_loop_num + row_num_once_tail_ub
                    # a1. row_num_once_ub * inner_loop_num
                    burst_len_res = ceil_value(self.row_num_once_ub * self.params_row * params_dsize, BLOCK_SIZE)

                    with tik_instance.for_range(0, self.inner_loop_num) as inner_loop_i:
                        inner_indices_offset = inner_loop_i * self.row_num_once_ub
                        output_offset = (pre_i * self.indices_num +
                                         block_id * self.indices_num_each_core +
                                         indices_loop_i * self.indices_row_num_once +
                                         inner_loop_i * self.row_num_once_ub) * self.params_row

                        self.indices_inner_gather_1(indices_ub, res_ub, self.row_num_once_ub,
                                                    inner_indices_offset, gm_offset_base, output_offset,
                                                    burst_len_row, burst_len_res)

                    # a2. row_num_once_tail_ub
                    with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                        burst_len_res = ceil_value(self.row_num_once_tail_ub * self.params_row * params_dsize,
                                                   BLOCK_SIZE)
                        inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                        output_offset = (pre_i * self.indices_num +
                                         block_id * self.indices_num_each_core +
                                         indices_loop_i * self.indices_row_num_once +
                                         self.inner_loop_num * self.row_num_once_ub) * self.params_row

                        self.indices_inner_gather_last_1(indices_ub, res_ub, self.row_num_once_tail_ub,
                                                         inner_indices_offset, gm_offset_base, output_offset,
                                                         burst_len_row, burst_len_res)

                with tik_instance.if_scope(self.indices_row_num_last > 0):
                    burst_len_res = ceil_value(self.row_num_last_ub * self.params_row * params_dsize, BLOCK_SIZE)
                    indices_offset = block_id * self.indices_num_each_core + \
                                     self.indices_loop_num * self.indices_row_num_once
                    # copy indices data to ub from gm
                    tik_instance.data_move(indices_ub, self.indices[indices_offset], 0, 1,
                                           ceil_value(self.indices_row_num_last * indices_dsize, BLOCK_SIZE), 0, 0)

                    with tik_instance.for_range(0, self.inner_loop_num_last) as inner_loop_i:
                        inner_indices_offset  = inner_loop_i * self.row_num_last_ub
                        output_offset = (pre_i * self.indices_num +
                                         block_id * self.indices_num_each_core +
                                         self.indices_loop_num * self.indices_row_num_once +
                                         inner_loop_i * self.row_num_last_ub) * self.params_row

                        self.indices_inner_gather_1(indices_ub, res_ub, self.row_num_last_ub,
                                                    inner_indices_offset, gm_offset_base, output_offset,
                                                    burst_len_row, burst_len_res)

                    with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                        burst_len_res = ceil_value(self.row_num_last_tail_ub * self.params_row * params_dsize,
                                                   BLOCK_SIZE)
                        inner_indices_offset = self.inner_loop_num_last * self.row_num_last_ub
                        output_offset = (pre_i * self.indices_num +
                                         block_id * self.indices_num_each_core +
                                         self.indices_loop_num * self.indices_row_num_once +
                                         self.inner_loop_num_last * self.row_num_last_ub) * self.params_row

                        self.indices_inner_gather_last_1(indices_ub, res_ub, self.row_num_last_tail_ub,
                                                         inner_indices_offset, gm_offset_base, output_offset,
                                                         burst_len_row, burst_len_res)

                with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0,
                                                   block_id == self.tail_process_core)):
                    indices_offset = self.need_core_num * self.indices_num_each_core
                    # copy indices data to ub from gm
                    tik_instance.data_move(indices_ub, self.indices[indices_offset], 0, 1,
                                           ceil_value(self.indices_num_remaining * indices_dsize, BLOCK_SIZE), 0, 0)

                    output_offset = (pre_i * self.indices_num +
                                     self.need_core_num * self.indices_num_each_core) * self.params_row
                    burst_len_res_tail = ceil_value(self.indices_num_remaining * self.params_row * params_dsize,
                                                    BLOCK_SIZE)

                    self.indices_inner_gather_1(indices_ub, res_ub, self.indices_num_remaining,
                                                0, gm_offset_base, output_offset, burst_len_row, burst_len_res_tail)

    def indices_inner_gather_last_1(self, indices_ub, res_ub, row_num_last, inner_indices_offset, gm_offset_base,
                                    output_offset, burst_len_row, burst_len_res):
        """
        process last indices for tiling mode 1

        Parameters
        ----------
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        row_num_last: the last indices num
        inner_indices_offset: inner indices num offset
        gm_offset_base: base of gm offset
        output_offset: output offset
        burst_len_row: burst length of one params row
        burst_len_res: burst length of result

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub", scope=tik.scope_ubuf)

        with tik_instance.for_range(0, row_num_last, thread_num=1) as row_i:
            indices_i_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_i_value")
            indices_i_value.set_as(indices_ub[inner_indices_offset + row_i])

            gm_offset_i = (gm_offset_base + indices_i_value) * self.params_row

            # copy params row to block_ub from gm
            tik_instance.data_move(block_ub, self.x[gm_offset_i],
                                   0, 1, burst_len_row, 0, 0)

            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(block_ub[i])

        # copy result data to gm from ub
        tail_elem = (row_num_last * self.params_row) % self.block_elem
        with tik_instance.if_scope(tik.all(tail_elem != 0, burst_len_res > 1)):
            with tik_instance.for_range(0, self.block_elem) as num_i:
                block_ub[num_i].set_as(res_ub[row_num_last * self.params_row - self.block_elem + num_i])
            tik_instance.data_move(self.y[output_offset], res_ub, 0,
                                   1, burst_len_res - 1, 0, 0)
            tik_instance.data_move(self.y[output_offset + (row_num_last * self.params_row - self.block_elem)],
                                   block_ub, 0, 1, 1, 0, 0)
        with tik_instance.else_scope():
            tik_instance.data_move(self.y[output_offset], res_ub,
                                   0, 1, burst_len_res, 0, 0)

    def indices_inner_gather_1(self, indices_ub, res_ub, row_num_once_ub,
                               inner_indices_offset, gm_offset_base, output_offset, burst_len_row, burst_len_res):
        """
        process row_num_once_ub indices for tiling mode 1

        Parameters
        ----------
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        row_num_once_ub: store rows of params in half UB
        inner_indices_offset: inner indices num offset
        gm_offset_base: base of gm offset
        output_offset: output offset
        burst_len_row: burst length of one params row
        burst_len_res: burst length of result

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        with tik_instance.for_range(0, row_num_once_ub, thread_num=1) as row_i:
            indices_i_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_i_value")
            indices_i_value.set_as(indices_ub[inner_indices_offset + row_i])

            gm_offset_i = (gm_offset_base + indices_i_value) * self.params_row

            block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                           scope=tik.scope_ubuf)
            # copy params row to block_ub from gm
            tik_instance.data_move(block_ub, self.x[gm_offset_i],
                                   0, 1, burst_len_row, 0, 0)

            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(block_ub[i])

        # copy result data to gm from ub
        tik_instance.data_move(self.y[output_offset], res_ub,
                               0, 1, burst_len_res, 0, 0)

    def compute_mode_3(self, half_ub_size, block_id):
        """
        compute for tiling mode 3

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_dsize = self.indices_dsize
        params_dsize = self.params_dsize

        with tik_instance.if_scope(block_id < self.need_core_num):
            indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // indices_dsize,),
                                             name="indices_ub", scope=tik.scope_ubuf)
            res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // params_dsize,),
                                         name="res_ub", scope=tik.scope_ubuf)

            burst_len_row = self.params_row * params_dsize // BLOCK_SIZE

            with tik_instance.for_range(0, self.params_pre) as pre_i:
                gm_offset_base = pre_i * self.params_axis

                indices_offset = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_offset")

                with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
                    indices_offset.set_as(block_id * self.indices_num_each_core +
                                          indices_loop_i * self.indices_row_num_once)
                    # copy indices data to ub from gm
                    tik_instance.data_move(indices_ub, self.indices[indices_offset], 0, 1,
                                           ceil_value(self.indices_row_num_once * indices_dsize, BLOCK_SIZE), 0, 0)

                    burst_len_res = self.row_num_once_ub * self.params_row * params_dsize // BLOCK_SIZE
                    inner_indices_offset = tik_instance.Scalar(dtype=self.indices_dtype, name="inner_indices_offset")
                    output_offset = tik_instance.Scalar(dtype=self.indices_dtype, name="output_offset")
                    with tik_instance.for_range(0, self.inner_loop_num) as inner_loop_i:
                        inner_indices_offset.set_as(inner_loop_i * self.row_num_once_ub)
                        output_offset.set_as((pre_i * self.indices_num +
                                              block_id * self.indices_num_each_core +
                                              indices_loop_i * self.indices_row_num_once +
                                              inner_loop_i * self.row_num_once_ub)
                                             * self.params_row)

                        self.indices_inner_gather(indices_ub, res_ub, self.row_num_once_ub,
                                                  inner_indices_offset, gm_offset_base, output_offset,
                                                  burst_len_row, burst_len_res)

                    with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                        burst_len_res = self.row_num_once_tail_ub * self.params_row * params_dsize // BLOCK_SIZE
                        inner_indices_offset.set_as(self.inner_loop_num * self.row_num_once_ub)
                        output_offset.set_as((pre_i * self.indices_num +
                                              block_id * self.indices_num_each_core +
                                              indices_loop_i * self.indices_row_num_once +
                                              self.inner_loop_num * self.row_num_once_ub)
                                             * self.params_row)

                        self.indices_inner_gather(indices_ub, res_ub, self.row_num_once_tail_ub,
                                                  inner_indices_offset, gm_offset_base, output_offset,
                                                  burst_len_row, burst_len_res)

                with tik_instance.if_scope(self.indices_row_num_last > 0):
                    burst_len_res = self.row_num_last_ub * self.params_row * params_dsize // BLOCK_SIZE
                    indices_offset.set_as(block_id * self.indices_num_each_core +
                                          self.indices_loop_num * self.indices_row_num_once)
                    # copy indices data to ub from gm
                    tik_instance.data_move(indices_ub, self.indices[indices_offset], 0, 1,
                                           ceil_value(self.indices_row_num_last * indices_dsize, BLOCK_SIZE), 0, 0)

                    inner_indices_offset = tik_instance.Scalar(dtype=self.indices_dtype, name="inner_indices_offset")
                    output_offset = tik_instance.Scalar(dtype=self.indices_dtype, name="output_offset")
                    with tik_instance.for_range(0, self.inner_loop_num_last) as inner_loop_i:
                        inner_indices_offset.set_as(inner_loop_i * self.row_num_last_ub)
                        output_offset.set_as((pre_i * self.indices_num +
                                              block_id * self.indices_num_each_core +
                                              self.indices_loop_num * self.indices_row_num_once +
                                              inner_loop_i * self.row_num_last_ub)
                                             * self.params_row)

                        self.indices_inner_gather(indices_ub, res_ub, self.row_num_last_ub,
                                                  inner_indices_offset, gm_offset_base, output_offset,
                                                  burst_len_row, burst_len_res)

                    with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                        burst_len_res = self.row_num_last_tail_ub * self.params_row * params_dsize // BLOCK_SIZE
                        inner_indices_offset.set_as(self.inner_loop_num_last * self.row_num_last_ub)
                        output_offset.set_as((pre_i * self.indices_num +
                                              block_id * self.indices_num_each_core +
                                              self.indices_loop_num * self.indices_row_num_once +
                                              self.inner_loop_num_last * self.row_num_last_ub)
                                             * self.params_row)

                        self.indices_inner_gather(indices_ub, res_ub, self.row_num_last_tail_ub,
                                                  inner_indices_offset, gm_offset_base, output_offset,
                                                  burst_len_row, burst_len_res)

                with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0,
                                                   block_id == self.tail_process_core)):
                    indices_offset.set_as(self.need_core_num * self.indices_num_each_core)
                    # copy indices data to ub from gm
                    tik_instance.data_move(indices_ub, self.indices[indices_offset], 0, 1,
                                           ceil_value(self.indices_num_remaining * indices_dsize, BLOCK_SIZE), 0, 0)

                    output_offset = tik_instance.Scalar(dtype=self.indices_dtype, name="output_offset")
                    output_offset.set_as((pre_i * self.indices_num +
                                          self.need_core_num * self.indices_num_each_core)
                                         * self.params_row)
                    burst_len_res_tail = self.indices_num_remaining * self.params_row * params_dsize // BLOCK_SIZE

                    self.indices_inner_gather(indices_ub, res_ub, self.indices_num_remaining,
                                              0, gm_offset_base, output_offset, burst_len_row, burst_len_res_tail)

    def indices_inner_gather(self, indices_ub, res_ub, row_num_once_ub, inner_indices_offset,
                             gm_offset_base, output_offset, burst_len_row, burst_len_res):
        """
        process row_num_once_ub indices for tiling mode 3

        Parameters
        ----------
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        row_num_once_ub: store rows of params in half UB
        inner_indices_offset: inner indices num offset
        gm_offset_base: base of gm offset
        output_offset: output offset
        burst_len_row: burst length of one params row
        burst_len_res: burst length of result

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        with tik_instance.for_range(0, row_num_once_ub, thread_num=1) as row_i:
            indices_i_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_i_value")
            indices_i_value.set_as(indices_ub[inner_indices_offset + row_i])

            gm_offset_i = (gm_offset_base + indices_i_value) * self.params_row

            # copy params data to ub from gm at the offset position specified by every indice value
            tik_instance.data_move(res_ub[row_i * self.params_row], self.x[gm_offset_i],
                                   0, 1, burst_len_row, 0, 0)

        # copy result data to gm from ub
        tik_instance.data_move(self.y[output_offset], res_ub,
                               0, 1, burst_len_res, 0, 0)

    def gather_v2_compute(self):
        """
        compute of gather_v2

        Parameters
        ----------
        None

        Returns
        -------
        compile info
        """
        self.x = self.tik_instance.Tensor(self.params_dtype, self.x_shape,
                                          name="x", scope=tik.scope_gm)
        self.indices = self.tik_instance.Tensor(self.indices_dtype, self.indices_shape,
                                                name="indices", scope=tik.scope_gm)
        self.axis = self.tik_instance.Tensor(self.axis_dtype, self.axis_shape,
                                             name="axis", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,),
                                                  name="ddr_arg", scope=tik.scope_gm)
        self.y = self.tik_instance.Tensor(self.y_dtype, shape=self.y_shape,
                                          name="y", scope=tik.scope_gm)

        self.gather_v2_compute_tiling()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x, self.indices, self.axis),
                                   outputs=(self.y,),
                                   flowtable=(self.tiling_gm,), enable_l2=True)

        # add compile info
        te.op.add_compile_info("vars", {"core_num": self.core_num,
                                        "ub_size": self.ub_size,
                                        "l1_size": self.l1_size,
                                        "params_dsize": self.params_dsize,
                                        "indices_dsize": self.indices_dsize
                                        })


@te.op.register_operator("GatherV2")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def gather_v2(x_dict, indices_dict, axis_dict, y_dict, kernel_name="GatherV2"):
    """
    gather_v2 interface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    indices_dict: input indices shape, dtype and range
    axis_dict: input axis shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of gather_v2 op

    Returns
    -------
    compile info
    """
    obj = GatherV2(x_dict, indices_dict, axis_dict, y_dict, kernel_name)
    return obj.gather_v2_compute()
