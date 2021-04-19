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
gather_nd
"""
from te import tvm
import te.lang.dynamic
from topi.cce import util
from te import tik
from te import platform as tbe_platform
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import check_op_params
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import check_dtype
from te.utils.error_manager import error_manager_vector

# data type of int32
INT32 = "int32"
# data type of int64
INT64 = "int64"
# one block size takes up 32b
BLOCK_SIZE = 32
# 100K, caches input params data
CACHE_UB_SIZE = 100 * 1024
# reserved ub size
RESERVED_UB_SIZE = 6 * 1024
UB_2K_SIZE = 2 * 1024
UB_1K_SIZE = 1024

# the max size of SHAPE is 2^31
MAX_INT32 = 2**31 - 1
MAX_SHAPE_SIZE = MAX_INT32
INDICES_NUM = MAX_INT32

TILING_ARG_NUM = 32

# paramsRowSize < 32
# params is not cache in UB
TILING_MODE_1 = 1
# params is cache in UB
TILING_MODE_2 = 2

# paramsRowSize >= 32
# paramsRow is 32B aligned, params is cache in L1
TILING_MODE_3 = 3
# paramsRow is 32B aligned, params is in gm
TILING_MODE_4 = 4
# paramsRow is 32B aligned, params is cache in UB (100M)
TILING_MODE_5 = 5
# paramsRow is not 32B aligned
TILING_MODE_6 = 6
# one paramsRow of data can not store in half UB
TILING_MODE_7 = 7
# complete params data needs to be moved for one indice
TILING_MODE_8 = 8


TYPE_LEN_DICT = {"float16": 2, "float32": 4, "int8": 1, "uint8": 1,
                 "int32": 4, "int64": 8, }


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


# pylint: disable=invalid-name, too-many-locals, too-many-arguments, too-many-public-methods
# pylint: disable=too-many-instance-attributes, too-many-lines
class GatherNd():
    """
    Function: class that execute gather_nd
    """
    def __init__(self, params_dict, indices_dict, y_dict, kernel_name):
        """
        constructor of GatherNd

        Parameters
        ----------
        params_dict: dict
            shape and dtype of input params
        indices_dict: dict
            shape and dtype of input indices
        y_dict: dict
            shape and dtype of output, should be same dtype as input
        kernel_name: str
            kernel name, default value is "GatherNd"

        Returns
        -------
        None
        """
        self.params_dtype = params_dict.get("dtype").lower()
        self.indices_dtype = indices_dict.get("dtype").lower()
        self.y_dtype = y_dict.get("dtype").lower()
        self.tiling_dtype = INT32
        params_support_dtype_list = ("float16", "float32", "int32", "int8", "uint8")
        indices_support_dtype_list = ("int32", "int64")
        check_dtype(self.params_dtype, params_support_dtype_list, param_name="params")
        check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        if self.y_dtype != self.params_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "y", "x",
                                                                  self.y_dtype, self.params_dtype)

        profile = tik.Dprofile()
        self.ub_size = profile.get_unified_buffer_size()
        self.l1_size = profile.get_l1_buffer_size()
        self.core_num = profile.get_aicore_num()
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.kernel_name = kernel_name
        self.available_size = self.ub_size - RESERVED_UB_SIZE

        self.x_shape = (MAX_SHAPE_SIZE,)
        self.indices_shape = (INDICES_NUM,)
        self.y_shape = (MAX_SHAPE_SIZE,)
        self.tiling_shape = (TILING_ARG_NUM,)

        self.params_dsize = TYPE_LEN_DICT.get(self.params_dtype)
        self.indices_dsize = TYPE_LEN_DICT.get(self.indices_dtype)
        self.block_elem = BLOCK_SIZE // self.params_dsize

        self.x = None
        self.indices = None
        self.y = None
        self.tiling_gm = None
        self.tiling_ub = None

        # tiling data
        self.tiling_mode = None
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
        self.row_num_last_tail_ub = None
        self.inner_loop_num_last = None

        self.params_total = None
        self.params_row = None
        self.indices_last_dim = None
        self.params_total = None
        self.one_row_loop = None
        self.one_row_tail = None
        self.suffix_list_index = None

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
        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.need_core_num.set_as(tiling_ub[1])
        self.tail_process_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tail_process_core")
        self.tail_process_core.set_as(tiling_ub[2])
        self.indices_num_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_each_core")
        self.indices_num_each_core.set_as(tiling_ub[3])
        self.indices_num_remaining = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_remaining")
        self.indices_num_remaining.set_as(tiling_ub[4])
        self.indices_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_loop_num")
        self.indices_loop_num.set_as(tiling_ub[5])
        self.indices_row_num_once = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_row_num_once")
        self.indices_row_num_once.set_as(tiling_ub[6])
        self.indices_row_num_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_row_num_last")
        self.indices_row_num_last.set_as(tiling_ub[7])

        self.row_num_once_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_once_ub")
        self.row_num_once_ub.set_as(tiling_ub[8])
        self.row_num_once_tail_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_once_tail_ub")
        self.row_num_once_tail_ub.set_as(tiling_ub[9])
        self.inner_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="inner_loop_num")
        self.inner_loop_num.set_as(tiling_ub[10])
        self.row_num_last_tail_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_last_tail_ub")
        self.row_num_last_tail_ub.set_as(tiling_ub[11])
        self.inner_loop_num_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="inner_loop_num_last")
        self.inner_loop_num_last.set_as(tiling_ub[12])

        self.params_row = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="paramsRow")
        self.params_row.set_as(tiling_ub[13])
        self.indices_last_dim = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indicesLastDim")
        self.indices_last_dim.set_as(tiling_ub[14])
        self.params_total = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_total")
        self.params_total.set_as(tiling_ub[15])

        self.one_row_loop = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="one_row_loop")
        self.one_row_loop.set_as(tiling_ub[16])
        self.one_row_tail = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="one_row_tail")
        self.one_row_tail.set_as(tiling_ub[17])

        self.suffix_list_index = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_list_index")
        self.suffix_list_index.set_as(tiling_ub[18])

    def process_loop_mode_1(self, loop_num, indices_num_offset, indices_ub, res_ub):
        """
        previous loop_num times process for tiling mode 1

        Parameters
        ----------
        loop_num: loop times
        indices_num_offset: indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(self.row_num_once_ub * self.params_row * self.params_dsize, BLOCK_SIZE)

        with tik_instance.for_range(0, loop_num) as inner_loop_i:
            inner_indices_offset = inner_loop_i * self.row_num_once_ub
            output_offset = (indices_num_offset + inner_indices_offset) * self.params_row

            with tik_instance.for_range(0, self.row_num_once_ub, thread_num=2) as row_i:
                block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,),
                                               name="block_ub", scope=tik.scope_ubuf)

                # compute gm offset of x
                gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
                index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
                suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
                with tik_instance.for_range(0, self.indices_last_dim) as index_i:
                    index_value.set_as(indices_ub[(inner_indices_offset + row_i) * self.indices_last_dim + index_i])
                    suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
                    gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)
                # copy params row to block_ub from gm
                tik_instance.data_move(block_ub, self.x[gm_offset_i], 0,
                                       1, 1, 0, 0)

                res_ub_offset = row_i * self.params_row
                with tik_instance.for_range(0, self.params_row) as i:
                    res_ub[res_ub_offset + i].set_as(block_ub[i])

            # copy result data to gm from ub
            tik_instance.data_move(self.y[output_offset], res_ub,
                                   0, 1, burst_len_res, 0, 0)

    def process_last_mode_1(self, row_num_last, indices_num_offset, inner_indices_offset, indices_ub, res_ub):
        """
        process row_num_last indices for tiling mode 1

        Parameters
        ----------
        row_num_last: the last indices num
        indices_num_offset: indices num offset
        inner_indices_offset: inner indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(row_num_last * self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (indices_num_offset + inner_indices_offset) * self.params_row

        with tik_instance.for_range(0, row_num_last, thread_num=2) as row_i:
            block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,),
                                           name="block_ub", scope=tik.scope_ubuf)

            gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
            index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
            suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
            with tik_instance.for_range(0, self.indices_last_dim) as index_i:
                index_value.set_as(indices_ub[(inner_indices_offset + row_i) * self.indices_last_dim + index_i])
                suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
                gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)
            # move params row data to block_ub from gm
            tik_instance.data_move(block_ub, self.x[gm_offset_i], 0,
                                   1, 1, 0, 0)

            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(block_ub[i])

        # move result data to gm from ub
        tail_elem = (row_num_last * self.params_row) % self.block_elem
        with tik_instance.if_scope(tik.all(tail_elem != 0, burst_len_res > 1)):
            block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                           scope=tik.scope_ubuf)
            with tik_instance.for_range(0, self.block_elem) as num_i:
                block_ub[num_i].set_as(res_ub[row_num_last * self.params_row - self.block_elem + num_i])

            tik_instance.data_move(self.y[output_offset], res_ub, 0,
                                   1, burst_len_res - 1, 0, 0)
            tik_instance.data_move(self.y[output_offset + (row_num_last * self.params_row - self.block_elem)],
                                   block_ub, 0, 1, 1, 0, 0)
        with tik_instance.else_scope():
            tik_instance.data_move(self.y[output_offset], res_ub, 0,
                                   1, burst_len_res, 0, 0)

    def process_loop_mode_2(self, loop_num, indices_num_offset, indices_ub, res_ub, x_ub):
        """
        previous loop_num times process for tiling mode 2

        Parameters
        ----------
        loop_num: loop times
        indices_num_offset: indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_ub: source params that cache in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(self.row_num_once_ub * self.params_row * self.params_dsize, BLOCK_SIZE)

        with tik_instance.for_range(0, loop_num) as inner_loop_i:
            inner_indices_offset = inner_loop_i * self.row_num_once_ub
            output_offset = (indices_num_offset + inner_indices_offset) * self.params_row

            with tik_instance.for_range(0, self.row_num_once_ub, thread_num=2) as row_i:
                # compute gm offset of x
                gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
                index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
                suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
                with tik_instance.for_range(0, self.indices_last_dim) as index_i:
                    index_value.set_as(indices_ub[(inner_indices_offset + row_i) * self.indices_last_dim + index_i])
                    suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
                    gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

                # set result to res_ub
                res_ub_offset = row_i * self.params_row
                with tik_instance.for_range(0, self.params_row) as i:
                    res_ub[res_ub_offset + i].set_as(x_ub[gm_offset_i + i])

            # copy result data to gm from ub
            tik_instance.data_move(self.y[output_offset], res_ub,
                                   0, 1, burst_len_res, 0, 0)

    def process_last_mode_2(self, row_num_last, indices_num_offset, inner_indices_offset, indices_ub, res_ub, x_ub):
        """
        process row_num_last indices for tiling mode 2

        Parameters
        ----------
        row_num_last: the last indices num
        indices_num_offset: indices num offset
        inner_indices_offset: inner indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_ub: source params that cache in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(row_num_last * self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (indices_num_offset + inner_indices_offset) * self.params_row

        with tik_instance.for_range(0, row_num_last, thread_num=2) as row_i:
            gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
            index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
            suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
            with tik_instance.for_range(0, self.indices_last_dim) as index_i:
                index_value.set_as(indices_ub[(inner_indices_offset + row_i) * self.indices_last_dim + index_i])
                suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
                gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

            # set result to res_ub
            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(x_ub[gm_offset_i + i])

        # move result data to gm from ub
        tail_elem = (row_num_last * self.params_row) % self.block_elem
        with tik_instance.if_scope(tik.all(tail_elem != 0, burst_len_res > 1)):
            block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                           scope=tik.scope_ubuf)
            with tik_instance.for_range(0, self.block_elem) as num_i:
                block_ub[num_i].set_as(res_ub[row_num_last * self.params_row - self.block_elem + num_i])

            tik_instance.data_move(self.y[output_offset], res_ub, 0,
                                   1, burst_len_res - 1, 0, 0)
            tik_instance.data_move(self.y[output_offset + (row_num_last * self.params_row - self.block_elem)],
                                   block_ub, 0, 1, 1, 0, 0)
        with tik_instance.else_scope():
            tik_instance.data_move(self.y[output_offset], res_ub, 0,
                                   1, burst_len_res, 0, 0)

    def process_remaining_tail_mode_2(self, indices_ub, res_ub, x_ub):
        """
        process remaining tail indices in core 0 for tiling mode 2

        Parameters
        ----------
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_ub: source params that cache in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_num_offset = self.need_core_num * self.indices_num_each_core
        # move indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                               ceil_value(self.indices_num_remaining * self.indices_last_dim * self.indices_dsize,
                                          BLOCK_SIZE), 0, 0)

        burst_len_res = ceil_value(self.indices_num_remaining * self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = indices_num_offset * self.params_row

        with tik_instance.for_range(0, self.indices_num_remaining, thread_num=1) as row_i:
            gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
            index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
            suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
            with tik_instance.for_range(0, self.indices_last_dim) as index_i:
                index_value.set_as(indices_ub[row_i * self.indices_last_dim + index_i])
                suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
                gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(x_ub[gm_offset_i + i])

        # copy result data to gm from ub
        tik_instance.data_move(self.y[output_offset], res_ub, 0,
                               1, burst_len_res, 0, 0)

    def compute_mode_2(self, half_ub_size, block_id):
        """
        compute for tiling mode 2, params row < 32b, and params data cached in UB

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, ((half_ub_size + UB_2K_SIZE) // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, ((half_ub_size + BLOCK_SIZE) // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)
        x_ub = tik_instance.Tensor(self.params_dtype, (CACHE_UB_SIZE // self.params_dsize,),
                                   name="x_ub", scope=tik.scope_ubuf)
        # cache params data in UB from gm
        tik_instance.data_move(x_ub, self.x, 0, 1, ceil_value(self.params_total, self.block_elem), 0, 0)

        # 1. indices_num_each_core: indices_row_num_once * indices_loop_num + indices_row_num_last
        # a. process indices_row_num_once * indices_loop_num
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
            # move indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                                   ceil_value(self.indices_row_num_once * self.indices_last_dim * self.indices_dsize,
                                              BLOCK_SIZE), 0, 0)

            # indices_row_num_once: row_num_once_ub * inner_loop_num + row_num_once_tail_ub
            # a1. process row_num_once_ub * inner_loop_num
            with tik_instance.if_scope(self.inner_loop_num > 0):
                self.process_loop_mode_2(self.inner_loop_num, indices_num_offset, indices_ub, res_ub, x_ub)

            # a2. process row_num_once_tail_ub
            with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                self.process_last_mode_2(self.row_num_once_tail_ub, indices_num_offset, inner_indices_offset,
                                         indices_ub, res_ub, x_ub)

        # b. indices_row_num_last: row_num_once_ub * inner_loop_num_last + row_num_last_tail_ub
        with tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + \
                                 self.indices_loop_num * self.indices_row_num_once
            # copy indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                                   ceil_value(self.indices_row_num_last * self.indices_last_dim * self.indices_dsize,
                                              BLOCK_SIZE), 0, 0)

            # b1. process row_num_once_ub * inner_loop_num_last
            with tik_instance.if_scope(self.inner_loop_num_last > 0):
                self.process_loop_mode_2(self.inner_loop_num_last, indices_num_offset, indices_ub, res_ub, x_ub)

            # b2. process row_num_last_tail_ub
            with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num_last * self.row_num_once_ub
                self.process_last_mode_2(self.row_num_last_tail_ub, indices_num_offset, inner_indices_offset,
                                         indices_ub, res_ub, x_ub)

        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id == self.tail_process_core)):
            self.process_remaining_tail_mode_2(indices_ub, res_ub, x_ub)

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
        indices_ub = tik_instance.Tensor(self.indices_dtype, ((half_ub_size + UB_2K_SIZE) // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, ((half_ub_size + BLOCK_SIZE) // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)

        # 1. indices_num_each_core: indices_row_num_once * indices_loop_num + indices_row_num_last
        # a. indices_row_num_once * indices_loop_num
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
            # move indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                                   ceil_value(self.indices_row_num_once * self.indices_last_dim * self.indices_dsize,
                                              BLOCK_SIZE), 0, 0)

            # indices_row_num_once = row_num_once_ub * inner_loop_num + row_num_once_tail_ub
            # a1. process row_num_once_ub * inner_loop_num
            with tik_instance.if_scope(self.inner_loop_num > 0):
                self.process_loop_mode_1(self.inner_loop_num, indices_num_offset, indices_ub, res_ub)

            # a2. process row_num_once_tail_ub
            with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                self.process_last_mode_1(self.row_num_once_tail_ub, indices_num_offset, inner_indices_offset,
                                         indices_ub, res_ub)

        # b. indices_row_num_last: row_num_once_ub * inner_loop_num_last + row_num_last_tail_ub
        with tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + \
                                 self.indices_loop_num * self.indices_row_num_once
            # copy indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                                   ceil_value(self.indices_row_num_last * self.indices_last_dim * self.indices_dsize,
                                              BLOCK_SIZE), 0, 0)

            with tik_instance.if_scope(self.inner_loop_num_last > 0):
                # b1. process row_num_once_ub * inner_loop_num_last
                self.process_loop_mode_1(self.inner_loop_num_last, indices_num_offset, indices_ub, res_ub)

            # b2. process row_num_last_tail_ub
            with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num_last * self.row_num_once_ub
                self.process_last_mode_1(self.row_num_last_tail_ub, indices_num_offset, inner_indices_offset,
                                         indices_ub, res_ub)

        # process indices_num_remaining
        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id == self.tail_process_core)):
            self.process_remaining_tail_mode_1(indices_ub, res_ub)

    def process_remaining_tail_mode_1(self, indices_ub, res_ub):
        """
        process remaining tail indices in core 0 for tiling mode 1

        Parameters
        ----------
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_num_offset = self.need_core_num * self.indices_num_each_core
        # move indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                               ceil_value(self.indices_num_remaining * self.indices_last_dim * self.indices_dsize,
                                          BLOCK_SIZE), 0, 0)

        burst_len_res = ceil_value(self.indices_num_remaining * self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = indices_num_offset * self.params_row

        with tik_instance.for_range(0, self.indices_num_remaining, thread_num=1) as row_i:
            block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                           scope=tik.scope_ubuf)

            gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
            index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
            suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
            with tik_instance.for_range(0, self.indices_last_dim) as index_i:
                index_value.set_as(indices_ub[row_i * self.indices_last_dim + index_i])
                suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
                gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)
            # copy params row to block_ub from gm
            tik_instance.data_move(block_ub, self.x[gm_offset_i], 0, 1, 1, 0, 0)

            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(block_ub[i])

        # copy result data to gm from ub
        tik_instance.data_move(self.y[output_offset], res_ub, 0,
                               1, burst_len_res, 0, 0)

    def compute_mode_8(self, block_id):
        """
        compute for tiling mode 8

        Parameters
        ----------
        block_id: id of ai core

        Returns
        -------
        None
        """
        params_elem_per_loop = (self.ub_size - UB_2K_SIZE) // self.params_dsize
        ub_size = self.ub_size - UB_1K_SIZE
        tik_instance = self.tik_instance
        res_ub = tik_instance.Tensor(self.params_dtype, (ub_size // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)

        with tik_instance.for_range(0, self.indices_num_each_core) as indices_i:
            indices_num_offset = self.indices_num_each_core * block_id + indices_i
            output_offset_base = indices_num_offset * self.params_total

            self.process_one_indice_mode_8(params_elem_per_loop, output_offset_base, res_ub)

        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.indices_num_remaining)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id
            output_offset_base = indices_num_offset * self.params_total

            self.process_one_indice_mode_8(params_elem_per_loop, output_offset_base, res_ub)

    def process_one_indice_mode_8(self, params_elem_per_loop, output_offset_base, res_ub):
        """
        process one indice for tiling mode 8

        Parameters
        ----------
        params_elem_per_loop: number of param elements that can be stored in UB space
        output_offset_base: base of output offset
        res_ub: cache result data in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len = ceil_value(params_elem_per_loop, self.block_elem)
        burst_len_last = ceil_value(self.row_num_last_tail_ub, self.block_elem)

        # params total: inner_loop_num*params_elem_per_loop + row_num_last_tail_ub
        with tik_instance.for_range(0, self.inner_loop_num) as inner_loop_i:
            # move params_elem_per_loop data to res_ub from gm
            tik_instance.data_move(res_ub, self.x[inner_loop_i*params_elem_per_loop], 0,
                                   1, burst_len, 0, 0)
            # move result data to gm from ub
            tik_instance.data_move(self.y[output_offset_base + inner_loop_i*params_elem_per_loop], res_ub, 0,
                                   1, burst_len, 0, 0)

        with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
            # move row_num_last_tail_ub data to res_ub from gm
            tik_instance.data_move(res_ub, self.x[self.inner_loop_num*params_elem_per_loop], 0,
                                   1, burst_len_last, 0, 0)

            # move result data to gm from ub
            output_offset = output_offset_base + self.inner_loop_num*params_elem_per_loop
            with tik_instance.if_scope(self.row_num_last_tail_ub % self.block_elem > 0):
                # set tail 32B of result to block_ub
                block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                               scope=tik.scope_ubuf)
                with tik_instance.for_range(0, self.block_elem) as num_i:
                    block_ub[num_i].set_as(res_ub[self.params_row - self.block_elem + num_i])

                tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_last - 1, 0, 0)
                tik_instance.data_move(self.y[output_offset + (self.params_row - self.block_elem)], block_ub, 0,
                                       1, 1, 0, 0)
            with tik_instance.else_scope():
                tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_last, 0, 0)

    def compute_mode_7(self, half_ub_size, block_id):
        """
        compute for tiling mode 7

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, ((half_ub_size + BLOCK_SIZE) // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)
        half_ub_params_elem = half_ub_size // self.params_dsize

        # 1. indices_num_each_core = indices_row_num_once * indices_loop_num + indices_row_num_last
        # a. process indices_row_num_once * indices_loop_num
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + \
                                 indices_loop_i * self.indices_row_num_once
            # move indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                                   ceil_value(self.indices_row_num_once * self.indices_last_dim * self.indices_dsize,
                                              BLOCK_SIZE), 0, 0)

            # process one indices_row_num_once
            self.process_loop_mode_7(self.indices_row_num_once, indices_num_offset, indices_ub, res_ub,
                                     half_ub_params_elem)

        # b. process indices_row_num_last
        with tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + \
                                 self.indices_loop_num * self.indices_row_num_once
            # copy indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                                   ceil_value(self.indices_row_num_last * self.indices_last_dim * self.indices_dsize,
                                              BLOCK_SIZE), 0, 0)

            self.process_loop_mode_7(self.indices_row_num_last, indices_num_offset, indices_ub, res_ub,
                                     half_ub_params_elem)

        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.indices_num_remaining)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id
            self.process_remaining_tail_mode_7(indices_num_offset, indices_ub, res_ub, half_ub_params_elem)

    def process_remaining_tail_mode_7(self, indices_num_offset, indices_ub, res_ub, half_ub_params_elem):
        """
        process tail indices in previous indices_num_remaining core for tiling mode 7

        Parameters
        ----------
        indices_num_offset: indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        half_ub_params_elem: number of params element that can be stored in half UB space

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        # move one indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                               ceil_value(1 * self.indices_last_dim * self.indices_dsize, BLOCK_SIZE), 0, 0)

        burst_len_sub_row = ceil_value(half_ub_params_elem, self.block_elem)
        burst_len_sub_row_last = ceil_value(self.one_row_tail, self.block_elem)
        output_offset = indices_num_offset * self.params_row

        gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
        index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
        suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
        with tik_instance.for_range(0, self.indices_last_dim) as index_i:
            index_value.set_as(indices_ub[index_i])
            suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
            gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

        # process the front part of one params_row: one_row_loop * half_ub_params_elem
        with tik_instance.for_range(0, self.one_row_loop) as row_inner_i:
            # move half_ub_params_elem data of one row to res_ub from gm
            tik_instance.data_move(res_ub, self.x[gm_offset_i + row_inner_i*half_ub_params_elem], 0,
                                   1, burst_len_sub_row, 0, 0)
            # copy result data to gm from ub
            tik_instance.data_move(self.y[output_offset + row_inner_i*half_ub_params_elem], res_ub,
                                   0, 1, burst_len_sub_row, 0, 0)

        # process of one the tail part of params_row: one_row_tail
        with tik_instance.if_scope(self.one_row_tail > 0):
            # move one_row_tail data to res_ub from gm
            tik_instance.data_move(res_ub, self.x[gm_offset_i + (self.params_row - self.one_row_tail)], 0,
                                   1, burst_len_sub_row_last, 0, 0)
            # copy result data to gm from ub
            with tik_instance.if_scope(self.one_row_tail % self.block_elem != 0):
                block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                               scope=tik.scope_ubuf)
                with tik_instance.for_range(0, self.block_elem) as num_i:
                    block_ub[num_i].set_as(res_ub[self.one_row_tail - self.block_elem + num_i])

                tik_instance.data_move(self.y[output_offset + (self.params_row - self.one_row_tail)], res_ub,
                                       0, 1, burst_len_sub_row_last - 1, 0, 0)
                tik_instance.data_move(self.y[output_offset + (self.params_row - self.block_elem)], block_ub, 0,
                                       1, 1, 0, 0)
            with tik_instance.else_scope():
                tik_instance.data_move(self.y[output_offset + (self.params_row - self.one_row_tail)], res_ub,
                                       0, 1, burst_len_sub_row_last, 0, 0)

    def process_loop_mode_7(self, loop_num, indices_num_offset, indices_ub, res_ub, half_ub_params_elem):
        """
        previous loop_num times process for tiling mode 7

        Parameters
        ----------
        loop_num: loop times
        indices_num_offset: indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        half_ub_params_elem: number of params element that can be stored in half UB space

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub", scope=tik.scope_ubuf)
        burst_len_sub_row = ceil_value(half_ub_params_elem, self.block_elem)
        burst_len_sub_row_last = ceil_value(self.one_row_tail, self.block_elem)

        # indices_row_num_once
        with tik_instance.for_range(0, loop_num) as row_i:
            output_offset = (indices_num_offset + row_i) * self.params_row

            # compute gm offset in x tensor
            gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
            index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
            suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
            with tik_instance.for_range(0, self.indices_last_dim) as index_i:
                index_value.set_as(indices_ub[row_i * self.indices_last_dim + index_i])
                suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
                gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

            # process the front part of one params_row: one_row_loop * half_ub_params_elem
            with tik_instance.for_range(0, self.one_row_loop) as row_inner_i:
                # move half_ub_params_elem data of one row to res_ub from gm
                tik_instance.data_move(res_ub, self.x[gm_offset_i + row_inner_i*half_ub_params_elem], 0,
                                       1, burst_len_sub_row, 0, 0)
                # copy result data to gm from ub
                tik_instance.data_move(self.y[output_offset + row_inner_i*half_ub_params_elem], res_ub,
                                       0, 1, burst_len_sub_row, 0, 0)

            # process of one the tail part of params_row: one_row_tail
            with tik_instance.if_scope(self.one_row_tail > 0):
                # move one_row_tail data to res_ub from gm
                tik_instance.data_move(res_ub, self.x[gm_offset_i + (self.params_row - self.one_row_tail)], 0,
                                       1, burst_len_sub_row_last, 0, 0)

                # copy result data to gm from ub
                with tik_instance.if_scope(tik.all(self.one_row_tail % self.block_elem != 0, loop_num - 1 == row_i)):
                    with tik_instance.for_range(0, self.block_elem) as num_i:
                        block_ub[num_i].set_as(res_ub[self.one_row_tail - self.block_elem + num_i])

                    tik_instance.data_move(self.y[output_offset + (self.params_row - self.one_row_tail)], res_ub,
                                           0, 1, burst_len_sub_row_last - 1, 0, 0)
                    tik_instance.data_move(self.y[output_offset + (self.params_row - self.block_elem)], block_ub, 0,
                                           1, 1, 0, 0)
                with tik_instance.else_scope():
                    tik_instance.data_move(self.y[output_offset + (self.params_row - self.one_row_tail)], res_ub,
                                           0, 1, burst_len_sub_row_last, 0, 0)

    def compute_mode_6(self, half_ub_size, block_id):
        """
        compute for tiling mode 6

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)

        # 1. indices_num_each_core: indices_row_num_once * indices_loop_num + indices_row_num_last
        # a. process indices_row_num_once * indices_loop_num
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
            # move indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                                   ceil_value(self.indices_row_num_once * self.indices_last_dim * self.indices_dsize,
                                              BLOCK_SIZE), 0, 0)

            # process first (indices_loop_num - 1) loop: indices_row_num_once * (indices_loop_num - 1)
            self.process_loop_mode_6(self.indices_row_num_once, indices_num_offset, indices_ub, res_ub)
            # process last one loop in indices_loop_num: indices_row_num_once * 1
            self.process_last_mode_6(self.indices_row_num_once, indices_num_offset, indices_ub, res_ub)

        # b. process indices_row_num_last
        with tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + \
                                 self.indices_loop_num * self.indices_row_num_once
            # move indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                                   ceil_value(self.indices_row_num_last * self.indices_last_dim * self.indices_dsize,
                                              BLOCK_SIZE), 0, 0)

            # process first (indices_row_num_last - 1) loop
            self.process_loop_mode_6(self.indices_row_num_last, indices_num_offset, indices_ub, res_ub)
            # process last one loop
            self.process_last_mode_6(self.indices_row_num_last, indices_num_offset, indices_ub, res_ub)

        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.indices_num_remaining)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id
            self.process_remaining_tail_mode_6(indices_num_offset, indices_ub, res_ub)

    def process_remaining_tail_mode_6(self, indices_num_offset, indices_ub, res_ub):
        """
        process tail indices in previous indices_num_remaining core for tiling mode 6

        Parameters
        ----------
        indices_num_offset: indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        # move one indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                               ceil_value(1 * self.indices_last_dim * self.indices_dsize, BLOCK_SIZE), 0, 0)

        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = indices_num_offset * self.params_row

        gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
        index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
        suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
        with tik_instance.for_range(0, self.indices_last_dim) as index_i:
            index_value.set_as(indices_ub[index_i])
            suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
            gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

        # copy one params_row to res_ub from gm
        tik_instance.data_move(res_ub, self.x[gm_offset_i], 0,
                               1, burst_len_row, 0, 0)

        # copy result data to gm from ub
        block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, self.block_elem) as num_i:
            block_ub[num_i].set_as(res_ub[self.params_row - self.block_elem + num_i])

        tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_row - 1, 0, 0)
        tik_instance.data_move(self.y[output_offset + (self.params_row - self.block_elem)], block_ub, 0,
                               1, 1, 0, 0)

    def process_last_mode_6(self, loop_num, indices_num_offset, indices_ub, res_ub):
        """
        process the last loop_num indices for tiling mode 6

        Parameters
        ----------
        loop_num: the last indices num
        indices_num_offset: indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (indices_num_offset + (loop_num - 1)) * self.params_row

        # compute gm offset in x tensor
        gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
        index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
        suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
        with tik_instance.for_range(0, self.indices_last_dim) as index_i:
            index_value.set_as(indices_ub[(loop_num - 1) * self.indices_last_dim + index_i])
            suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
            gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

        # move params_row to res_ub from gm
        tik_instance.data_move(res_ub, self.x[gm_offset_i], 0,
                               1, burst_len_row, 0, 0)

        # set tail 32B of result to block_ub
        block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, self.block_elem) as num_i:
            block_ub[num_i].set_as(res_ub[self.params_row - self.block_elem + num_i])

        # copy result data to gm from ub
        tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_row - 1, 0, 0)
        tik_instance.data_move(self.y[output_offset + (self.params_row - self.block_elem)], block_ub, 0,
                               1, 1, 0, 0)

    def process_loop_mode_6(self, loop_num, indices_num_offset, indices_ub, res_ub):
        """
        previous loop_num times process for tiling mode 6

        Parameters
        ----------
        loop_num: loop times
        indices_num_offset: indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)

        with tik_instance.for_range(0, loop_num - 1) as row_i:
            output_offset = (indices_num_offset + row_i) * self.params_row

            # compute gm offset in x tensor
            gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
            index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
            suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
            with tik_instance.for_range(0, self.indices_last_dim) as index_i:
                index_value.set_as(indices_ub[row_i * self.indices_last_dim + index_i])
                suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
                gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

            # move params_row to res_ub from gm
            tik_instance.data_move(res_ub, self.x[gm_offset_i], 0,
                                   1, burst_len_row, 0, 0)

            # copy result data to gm from ub
            tik_instance.data_move(self.y[output_offset], res_ub,
                                   0, 1, burst_len_row, 0, 0)

    def compute_mode_5(self, remain_half_ub_size, block_id):
        """
        compute for tiling mode 5

        Parameters
        ----------
        remain_half_ub_size: bytes of remaining half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        x_ub = tik_instance.Tensor(self.params_dtype, (CACHE_UB_SIZE // self.params_dsize,),
                                   name="x_ub", scope=tik.scope_cbuf)
        # cache params data in UB from gm
        tik_instance.data_move(x_ub, self.x, 0, 1,
                               ceil_value(self.params_total, self.block_elem), 0, 0)

        self.compute_mode_32b_aligned(remain_half_ub_size, block_id, x_ub)

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
        x_cbuf = tik_instance.Tensor(self.params_dtype, (self.l1_size // self.params_dsize,),
                                     name="x_l1", scope=tik.scope_cbuf)
        # cache params data in L1 from gm
        tik_instance.data_move(x_cbuf, self.x, 0, 1,
                               ceil_value(self.params_total, self.block_elem), 0, 0)

        self.compute_mode_32b_aligned(half_ub_size, block_id, x_cbuf)

    def compute_mode_4(self, half_ub_size, block_id):
        """
        compute for tiling mode 4

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        self.compute_mode_32b_aligned(half_ub_size, block_id, self.x)

    def compute_mode_32b_aligned(self, half_ub_size, block_id, x_src):
        """
        compute for tiling mode of 32B aligned

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)

        # 1. indices_num_each_core: indices_row_num_once * indices_loop_num + indices_row_num_last
        # a. process indices_row_num_once * indices_loop_num
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
            # move indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                                   ceil_value(self.indices_row_num_once * self.indices_last_dim * self.indices_dsize,
                                              BLOCK_SIZE), 0, 0)

            # indices_row_num_once: row_num_once_ub * inner_loop_num + row_num_once_tail_ub
            # a1. process row_num_once_ub * inner_loop_num
            with tik_instance.if_scope(self.inner_loop_num > 0):
                self.process_loop_32b_aligned(self.inner_loop_num, indices_num_offset, indices_ub, res_ub, x_src)

            # a2. process row_num_once_tail_ub
            with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                self.process_last_32b_aligned(self.row_num_once_tail_ub, indices_num_offset, inner_indices_offset,
                                              indices_ub, res_ub, x_src)

        # b. indices_row_num_last: row_num_once_ub * inner_loop_num_last + row_num_last_tail_ub
        with tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + \
                                 self.indices_loop_num * self.indices_row_num_once
            # copy indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                                   ceil_value(self.indices_row_num_last * self.indices_last_dim * self.indices_dsize,
                                              BLOCK_SIZE), 0, 0)

            # b1. process row_num_once_ub * inner_loop_num_last
            with tik_instance.if_scope(self.inner_loop_num_last > 0):
                self.process_loop_32b_aligned(self.inner_loop_num_last, indices_num_offset, indices_ub, res_ub, x_src)

            # b2. process row_num_last_tail_ub
            with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num_last * self.row_num_once_ub
                self.process_last_32b_aligned(self.row_num_last_tail_ub, indices_num_offset, inner_indices_offset,
                                              indices_ub, res_ub, x_src)

        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.indices_num_remaining)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id
            self.process_remaining_tail_32b_aligned(indices_num_offset, indices_ub, res_ub, x_src)

    def process_remaining_tail_32b_aligned(self, indices_num_offset, indices_ub, res_ub, x_src):
        """
        process tail indices in previous indices_num_remaining core for tiling mode of 32B aligned

        Parameters
        ----------
        indices_num_offset: indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        # move one indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset * self.indices_last_dim], 0, 1,
                               ceil_value(1 * self.indices_last_dim * self.indices_dsize, BLOCK_SIZE), 0, 0)

        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = indices_num_offset * self.params_row

        gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
        index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
        suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
        with tik_instance.for_range(0, self.indices_last_dim) as index_i:
            index_value.set_as(indices_ub[index_i])
            suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
            gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

        # copy one params_row to res_ub from gm or UB or L1
        tik_instance.data_move(res_ub, x_src[gm_offset_i], 0,
                               1, burst_len_row, 0, 0)

        # copy result data to gm from ub
        tik_instance.data_move(self.y[output_offset], res_ub, 0,
                               1, burst_len_row, 0, 0)

    def process_last_32b_aligned(self, row_num_last, indices_num_offset, inner_indices_offset,
                                 indices_ub, res_ub, x_src):
        """
        process last row_num_last indices for tiling mode of 32B aligned

        Parameters
        ----------
        row_num_last: the last indices num
        indices_num_offset: indices num offset
        inner_indices_offset: inner indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(row_num_last * self.params_row * self.params_dsize, BLOCK_SIZE)
        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (indices_num_offset + inner_indices_offset) * self.params_row

        with tik_instance.for_range(0, row_num_last, thread_num=2) as row_i:
            # compute gm offset in x tensor
            gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
            index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
            suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
            with tik_instance.for_range(0, self.indices_last_dim) as index_i:
                index_value.set_as(indices_ub[(inner_indices_offset + row_i) * self.indices_last_dim + index_i])
                suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
                gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

            # move params_row data to res_ub from gm or UB or L1
            tik_instance.data_move(res_ub[row_i * self.params_row], x_src[gm_offset_i], 0,
                                   1, burst_len_row, 0, 0)

        # move result data to gm from ub
        tik_instance.data_move(self.y[output_offset], res_ub, 0,
                               1, burst_len_res, 0, 0)

    def process_loop_32b_aligned(self, loop_num, indices_num_offset, indices_ub, res_ub, x_src):
        """
        previous loop_num times process for tiling mode of 32B aligned

        Parameters
        ----------
        loop_num: loop times
        indices_num_offset: indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(self.row_num_once_ub * self.params_row * self.params_dsize, BLOCK_SIZE)
        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)

        with tik_instance.for_range(0, loop_num) as inner_loop_i:
            inner_indices_offset = inner_loop_i * self.row_num_once_ub
            output_offset = (indices_num_offset + inner_indices_offset) * self.params_row

            with tik_instance.for_range(0, self.row_num_once_ub, thread_num=2) as row_i:
                # compute gm offset in x tensor
                gm_offset_i = tik_instance.Scalar(dtype=self.tiling_dtype, name="gm_offset_i", init_value=0)
                index_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="index_value")
                suffix_value = tik_instance.Scalar(dtype=self.tiling_dtype, name="suffix_value")
                with tik_instance.for_range(0, self.indices_last_dim) as index_i:
                    index_value.set_as(indices_ub[(inner_indices_offset + row_i) * self.indices_last_dim + index_i])
                    suffix_value.set_as(self.tiling_ub[self.suffix_list_index + index_i])
                    gm_offset_i.set_as(gm_offset_i + index_value * suffix_value)

                # move params_row to res_ub from gm or UB or L1
                tik_instance.data_move(res_ub[row_i * self.params_row], x_src[gm_offset_i], 0,
                                       1, burst_len_row, 0, 0)

            # copy result data to gm from ub
            tik_instance.data_move(self.y[output_offset], res_ub,
                                   0, 1, burst_len_res, 0, 0)

    def gather_nd_compute_tiling(self):
        """
        Main process of gather_nd

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        half_ub_size = self.available_size // 2
        tik_instance = self.tik_instance

        with tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            # get run tiling data
            self.tiling_ub = tik_instance.Tensor(self.tiling_dtype, self.tiling_shape,
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0,
                                   1, ceil_value(TILING_ARG_NUM * TYPE_LEN_DICT.get(self.tiling_dtype), BLOCK_SIZE),
                                   0, 0)
            # get run tiling mode
            tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tiling_mode")
            tiling_mode.set_as(self.tiling_ub[0])

            self.get_tiling_args(self.tiling_ub)

            with tik_instance.if_scope(block_id < self.need_core_num):
                # 1. params_row size < 32
                with tik_instance.if_scope(tik.any(tiling_mode == TILING_MODE_1, tiling_mode == TILING_MODE_2)):
                    # TILING_MODE_1, params not cache
                    with tik_instance.if_scope(tiling_mode == TILING_MODE_1):
                        with tik_instance.new_stmt_scope():
                            self.compute_mode_1(half_ub_size, block_id)
                    # TILING_MODE_2, params cached in UB
                    with tik_instance.else_scope():
                        with tik_instance.new_stmt_scope():
                            self.compute_mode_2((self.available_size - CACHE_UB_SIZE) // 2, block_id)
                with tik_instance.else_scope():
                    # params_row is 32B aligned
                    with tik_instance.if_scope(tik.any(tiling_mode == TILING_MODE_3, tiling_mode == TILING_MODE_4,
                                                       tiling_mode == TILING_MODE_5)):
                        with tik_instance.if_scope(tiling_mode == TILING_MODE_5):
                            with tik_instance.new_stmt_scope():
                                # TILING_MODE_5, params cached in UB
                                self.compute_mode_5((self.available_size - CACHE_UB_SIZE) // 2, block_id)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(tiling_mode == TILING_MODE_3):
                                with tik_instance.new_stmt_scope():
                                    # TILING_MODE_3, params cached in L1
                                    self.compute_mode_3(half_ub_size, block_id)
                            with tik_instance.else_scope():
                                with tik_instance.new_stmt_scope():
                                    # TILING_MODE_4, params not cache
                                    self.compute_mode_4(half_ub_size, block_id)
                    with tik_instance.else_scope():
                        with tik_instance.if_scope(tiling_mode == TILING_MODE_6):
                            with tik_instance.new_stmt_scope():
                                # params_row is not 32B aligned
                                self.compute_mode_6(half_ub_size, block_id)
                        with tik_instance.else_scope():
                            with tik_instance.if_scope(tiling_mode == TILING_MODE_7):
                                with tik_instance.new_stmt_scope():
                                    # one params_row can not store in half UB
                                    self.compute_mode_7(half_ub_size, block_id)
                            with tik_instance.else_scope():
                                with tik_instance.new_stmt_scope():
                                    # TILING_MODE_8, complete params data needs to be moved for one indice
                                    self.compute_mode_8(block_id)

    def gather_nd_compute(self):
        """
        compute of gather_nd

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
        self.y = self.tik_instance.Tensor(self.y_dtype, shape=self.y_shape,
                                          name="y", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, self.tiling_shape,
                                                  name="ddr_arg", scope=tik.scope_gm)

        self.gather_nd_compute_tiling()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x, self.indices),
                                   outputs=(self.y,),
                                   flowtable=(self.tiling_gm,), enable_l2=True)

        # add compile info
        te.op.add_compile_info("vars", {"core_num": self.core_num,
                                        "ub_size": self.ub_size,
                                        "l1_size": self.l1_size,
                                        "params_dsize": self.params_dsize,
                                        "indices_dsize": self.indices_dsize
                                        })


@te.op.register_operator("GatherNd")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def gather_nd(x_dict, indices_dict, y_dict, kernel_name="GatherNd"):
    """
    gather_nd interface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    indices_dict: input indices shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of gather_nd op

    Returns
    -------
    compile info
    """
    obj = GatherNd(x_dict, indices_dict, y_dict, kernel_name)
    return obj.gather_nd_compute()

