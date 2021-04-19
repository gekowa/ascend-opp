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
unsorted_segment_sum
"""
import sys
import math
import json

import numpy as np

import te.lang.dynamic
from te import tvm
from te import tik
from te import platform
from topi import generic
from functools import reduce as reduceIns
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.error_manager import error_manager_vector

# fp32 select key
SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN = 1
SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE = 2
SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN = 4
SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E = 5
SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_BIG_E = 6
SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MODIFY = 7
SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MULTI = 8

# int32 select key
SELECT_KEY_MODE_INT32_SMALL_E = 11
SELECT_KEY_MODE_INT32_BIG_E = 12

DTYPE_FP32 = "float32"
DTYPE_INT32 = "int32"
TILING_PARAM_DTYPE = DTYPE_INT32

# max_int32
MAX_INT32 = 2 ** 31 - 1

# fp32 byte
BYTE_FP32 = 4

# int32 byte
BYTE_INT32 = 4

# full mask for fp32
MASK_FP32 = 64

# full mask for int32
MASK_INT32 = 64

# byte of one block
BYTE_BLOCK = 32

# byte of one repeat block
BYTE_REPEAT_BLOCK = 256

# max repeat time of vector instruction
MAX_REPEAT_TIME = 255

# min ids nums in data move
MIN_IDS_NUMS = BYTE_BLOCK // BYTE_FP32

# cloud block num
CLOUD_CORE_NUM = 32

# min_tensor_ele_num
MIN_TENSOR_ELE_NUM = 32

# tiling params num
TILING_PARAMS_NUM = 64

# fp32 ele num one ub block
ELE_NUM_ONE_BLOCK_FP32 = BYTE_BLOCK // BYTE_FP32

# modify last axis one, multi times
MULTI = 4

# num fp32
ONE = 1.0
NEG_ONE = -1.0
ZERO = 0.0


def _ceil_div(val, block):
    """
    compute ceil div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    ceil value
    """
    return (val + block - 1) // block


def _floor(val, block):
    """
    compute floor div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    floor value
    """
    return val // block * block


def _div(val, block):
    """
    compute front part and last part in ceil div

    Parameters
    ----------
    val: num
    block: factor

    Returns
    -------
    front_part_num: front part in ceil div
    last_part: last part in ceil div
    """
    front_part_num = val // block
    last_part = val - front_part_num * block
    return front_part_num, last_part


class UnsortedSegmentSum():
    def __init__(self, x_dict, segment_ids_dict, num_segments_dict, y_dict,
                 kernel_name):
        """
        constructor of class UnsortedSegmentSum

        Parameters
        ----------
        x_dict: dict
            shape and dtype of x
        segment_ids_dict: dict
            shape and dtype of segment_ids
        num_segments_dict: dict
            shape and dtype of num_segments
        y_dict: dict
            shape and dtype of y
        kernel_name: str
            kernel_name, default value is "UnsortedSegmentSum"

        Returns
        -------
        None
        """
        # get dtype
        self.input_dtype = x_dict.get("dtype", None)
        self.input_dtype = self.input_dtype.lower()
        self.ids_dtype = segment_ids_dict.get("dtype", None)
        self.ids_dtype = self.ids_dtype.lower()
        self.num_segments_dtype = num_segments_dict.get("dtype", None)
        self.num_segments_dtype = self.num_segments_dtype.lower()
        self.output_dtype = self.input_dtype
        self.fp32_ele_num_one_block = ELE_NUM_ONE_BLOCK_FP32
        self.is_double_buffer = True
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik()
        self.ub_size = _tik_get_ub_size(self.is_double_buffer)
        self.core_num = _tik_get_core_num()
        if (self.input_dtype == DTYPE_FP32):
            self.ub_tensor_num = 3

        class GmTensor():
            def __init__(self, tik_instance, input_dtype, ids_dtype,
                         num_segments_dtype):
                """
                constructor of class GmTensor

                Parameters
                ----------
                tik_instance: tik_instance
                input_dtype: x dtype
                ids_dtype: ids dtype
                num_segments_dtype: num_segments dtype

                Returns
                -------
                None
                """
                self.input_gm = tik_instance.Tensor(input_dtype, (MAX_INT32,),
                                                    name="input_gm",
                                                    scope=tik.scope_gm)
                self.ids_gm = tik_instance.Tensor(ids_dtype, (MAX_INT32,),
                                                  name="ids_gm",
                                                  scope=tik.scope_gm)
                self.num_segments_gm = tik_instance.Tensor(num_segments_dtype, (
                    MIN_TENSOR_ELE_NUM,), name="num_segments_gm",
                                                           scope=tik.scope_gm)
                if (input_dtype == DTYPE_FP32):
                    self.output_gm = tik_instance.Tensor(input_dtype,
                                                         (MAX_INT32,),
                                                         name="output_gm",
                                                         scope=tik.scope_gm,
                                                         is_atomic_add=True)
                else:
                    self.output_gm = tik_instance.Tensor(input_dtype,
                                                         (MAX_INT32,),
                                                         name="output_gm",
                                                         scope=tik.scope_gm)
                self.tiling_gm = tik_instance.Tensor(TILING_PARAM_DTYPE,
                                                     (TILING_PARAMS_NUM,),
                                                     name="tiling_gm",
                                                     scope=tik.scope_gm)

        class UbTensor():
            def __init__(self):
                """
                constructor of class UbTensor

                Parameters
                ----------
                None

                Returns
                -------
                None
                """
                self.input_ub = None
                self.ids_ub = None
                self.output_ub = None
                self.num_segments_ub = None

        # scalar of tiling params
        class CommonScalar():
            def __init__(self, tik_instance, num_segments_dtype, ids_dtype):
                """
                constructor of class CommonScalar

                Parameters
                ----------
                tik_instance: tik_instance
                num_segments_dtype: num_segments dtype
                ids_dtype: ids dtype

                Returns
                -------
                None
                """
                self.num_segments_scalar = tik_instance.Scalar(
                    dtype=num_segments_dtype, name="num_segments_scalar")
                self.id_val_scalar = tik_instance.Scalar(dtype=ids_dtype,
                                                         name="id_val_scalar")
                self.select_key = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE, name="select_key")
                self.need_core_num = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE, name="need_core_num")
                self.num_segments_front_core = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="num_segments_front_core")
                self.num_segments_last_core = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="num_segments_last_core")

        class Fp32InputDataInputScalar():
            def __init__(self, tik_instance):
                """
                constructor of class Fp32InputDataInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                # front core
                self.ele_num_front_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE, \
                    name="ele_num_front_core")
                # front part front core
                self.mov_times_gm2ub_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="mov_times_gm2ub_front_part_front_core")
                self.front_burst_len_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_burst_len_front_part_front_core")
                self.last_burst_len_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_burst_len_front_part_front_core")
                self.front_ele_num_ub_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_ele_num_ub_front_part_front_core")
                self.last_ele_num_ub_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_ele_num_ub_front_part_front_core")
                self.front_rows_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_rows_front_part_front_core")
                self.last_rows_front_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_rows_front_part_front_core")
                # last part front core
                self.mov_times_gm2ub_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="mov_times_gm2ub_last_part_front_core")
                self.front_burst_len_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_burst_len_last_part_front_core")
                self.last_burst_len_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_burst_len_last_part_front_core")
                self.front_ele_num_ub_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_ele_num_ub_last_part_front_core")
                self.last_ele_num_ub_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_ele_num_ub_last_part_front_core")
                self.front_rows_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_rows_last_part_front_core")
                self.last_rows_last_part_front_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_rows_last_part_front_core")

                # last core
                self.ele_num_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="ele_num_last_core")
                # front part last core
                self.mov_times_gm2ub_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="mov_times_gm2ub_front_part_last_core")
                self.front_burst_len_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_burst_len_front_part_last_core")
                self.last_burst_len_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_burst_len_front_part_last_core")
                self.front_ele_num_ub_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_ele_num_ub_front_part_last_core")
                self.last_ele_num_ub_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_ele_num_ub_front_part_last_core")
                self.front_rows_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_rows_front_part_last_core")
                self.last_rows_front_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_rows_front_part_last_core")
                # last part last core
                self.mov_times_gm2ub_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="mov_times_gm2ub_last_part_last_core")
                self.front_burst_len_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_burst_len_last_part_last_core")
                self.last_burst_len_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_burst_len_last_part_last_core")
                self.front_ele_num_ub_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_ele_num_ub_last_part_last_core")
                self.last_ele_num_ub_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_ele_num_ub_last_part_last_core")
                self.front_rows_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="front_rows_last_part_last_core")
                self.last_rows_last_part_last_core = \
                    tik_instance.Scalar(
                        dtype=TILING_PARAM_DTYPE,
                        name="last_rows_last_part_last_core")

        class Fp32ENumInputScalar():
            def __init__(self, tik_instance):
                """
                constructor of class Fp32ENumInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.e_num = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE, name="e_num")
                self.e_mov_times_gm2ub = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="e_mov_times_gm2ub")
                self.e_ub2gm_front_burst_len = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="e_ub2gm_front_burst_len")
                self.e_num_front_part = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="e_num_front_part")
                self.e_ub2gm_last_burst_len = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="e_ub2gm_last_burst_len")
                self.e_num_last_part = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="e_num_last_part")

        class Fp32IdsInputScalar():
            def __init__(self, tik_instance):
                """
                constructor of class Fp32IdsInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.size = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE, name="size")
                self.ele_num_front_core = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="ele_num_front_core")
                self.mov_times_gm2ub_front_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="mov_times_gm2ub_front_core")
                self.front_burst_len_front_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="front_burst_len_front_core")
                self.last_burst_len_front_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="last_burst_len_front_core")
                self.ele_num_ub_front_part_front_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="ele_num_ub_front_part_front_core")
                self.ele_num_ub_last_part_front_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="ele_num_ub_last_part_front_core")
                self.ele_num_last_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="ele_num_last_core")
                self.mov_times_gm2ub_last_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="mov_times_gm2ub_last_core")
                self.front_burst_len_last_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="front_burst_len_last_core")
                self.last_burst_len_last_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="last_burst_len_last_core")
                self.ele_num_ub_front_part_last_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="ele_num_ub_front_part_last_core")
                self.ele_num_ub_last_part_last_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="ele_num_ub_last_part_last_core")

        class Fp32OutputInitInputScalar():
            def __init__(self, tik_instance):
                """
                constructor of class Fp32OutputInitInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.\
        last_repeat_time_front_part_front_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="last_repeat_time_front_part_front_core")
                self.init_times_front_part_front_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="init_times_front_part_front_core")
                self.\
        last_repeat_time_last_part_front_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="last_repeat_time_last_part_front_core")
                self.init_times_last_part_front_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="init_times_last_part_front_core")
                self.\
        last_repeat_time_front_part_last_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="last_repeat_time_front_part_last_core")
                self.init_times_front_part_last_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="init_times_front_part_last_core")
                self.\
        last_repeat_time_last_part_last_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="last_repeat_time_last_part_last_core")
                self.init_times_last_part_last_core = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="init_times_last_part_last_core")
                self.last_axis_align_front_part = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="last_axis_align_front_part")
                self.last_axis_align_floor = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="last_axis_align_floor")
                self.last_part_vadd_mask = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="last_part_vadd_mask")

        class Int32IdsScalar():
            def __init__(self, tik_instance):
                """
                constructor of class Int32IdsScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.size = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE, name="size")
                self.mov_times_gm2ub = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="mov_times_gm2ub")
                self.ele_num_ub_front_part = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="ele_num_ub_front_part")
                self.front_burst_len = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="front_burst_len")
                self.ele_num_ub_last_part = \
                    tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="ele_num_ub_last_part")
                self.last_burst_len = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="last_burst_len")

        class Int32ENumScalar():
            def __init__(self, tik_instance):
                """
                constructor of class Int32ENumScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.e_num = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE, name="e_num")
                self.e_mov_times_gm2ub = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="e_mov_times_gm2ub")
                self.e_ub2gm_front_burst_len = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="e_ub2gm_front_burst_len")
                self.e_num_front_part = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="e_num_front_part")
                self.repeat_time_front_part = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="repeat_time_front_part")
                self.e_ub2gm_last_burst_len = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="e_ub2gm_last_burst_len")
                self.e_num_last_part = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="e_num_last_part")
                self.repeat_time_last_part = tik_instance.Scalar(
                    dtype=TILING_PARAM_DTYPE,
                    name="repeat_time_last_part")

        self.obj_gm_tensor = GmTensor(self.tik_instance, self.input_dtype,
                                      self.ids_dtype, self.num_segments_dtype)
        self.obj_ub_tensor = UbTensor()
        self.obj_common_scalar = CommonScalar(self.tik_instance,
                                              self.num_segments_dtype,
                                              self.ids_dtype)
        self.obj_fp32_input_data_input_scalar = Fp32InputDataInputScalar(
            self.tik_instance)
        self.obj_fp32_e_num_input_scalar = Fp32ENumInputScalar(
            self.tik_instance)
        self.obj_fp32_ids_input_scalar = Fp32IdsInputScalar(self.tik_instance)
        self.obj_fp32_output_init_input_scalar = Fp32OutputInitInputScalar(
            self.tik_instance)
        self.obj_int32_ids_input_scalar = Int32IdsScalar(self.tik_instance)
        self.obj_int32_e_num_input_scalar = Int32ENumScalar(self.tik_instance)

        with self.tik_instance.new_stmt_scope():
            # num_segments
            self.obj_ub_tensor.num_segments_ub = self.tik_instance.Tensor(
                self.num_segments_dtype, (MIN_TENSOR_ELE_NUM,),
                name="num_segments_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.obj_ub_tensor.num_segments_ub,
                                        self.obj_gm_tensor.num_segments_gm, 0,
                                        1, 1, 0, 0)
            self.obj_common_scalar.num_segments_scalar.set_as(
                self.obj_ub_tensor.num_segments_ub[1])

        with self.tik_instance.new_stmt_scope():
            self.obj_ub_tensor.tiling_ub = self.tik_instance.Tensor(
                TILING_PARAM_DTYPE, (TILING_PARAMS_NUM,), name="tiling_ub",
                scope=tik.scope_ubuf)
            # mov tiling params from gm to ub
            self.tik_instance.data_move(self.obj_ub_tensor.tiling_ub,
                                        self.obj_gm_tensor.tiling_gm, 0, 1,
                                        TILING_PARAMS_NUM * BYTE_INT32 // \
                                        BYTE_BLOCK,
                                        0, 0)
            # input scalar in flowtable
            input_scalar_index = 0

            # common params
            self.obj_common_scalar.select_key.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_common_scalar.need_core_num.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1

            # input data params
            # front core
            self.\
        obj_fp32_input_data_input_scalar.ele_num_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # front part front core
            self.\
        obj_fp32_input_data_input_scalar.\
                mov_times_gm2ub_front_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_burst_len_front_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_burst_len_front_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_ele_num_ub_front_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_ele_num_ub_front_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_rows_front_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_rows_front_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # last part front core
            self.obj_fp32_input_data_input_scalar.\
                mov_times_gm2ub_last_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_burst_len_last_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_burst_len_last_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_ele_num_ub_last_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_ele_num_ub_last_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_rows_last_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_rows_last_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # last core
            self.obj_fp32_input_data_input_scalar.\
                ele_num_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # front part last core
            self.obj_fp32_input_data_input_scalar.\
                mov_times_gm2ub_front_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_burst_len_front_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_burst_len_front_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_ele_num_ub_front_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_ele_num_ub_front_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_rows_front_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_rows_front_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # last part last core
            self.obj_fp32_input_data_input_scalar.\
                mov_times_gm2ub_last_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_burst_len_last_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_burst_len_last_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_ele_num_ub_last_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_ele_num_ub_last_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                front_rows_last_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_input_data_input_scalar.\
                last_rows_last_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1

            # e num params
            self.obj_fp32_e_num_input_scalar.e_num.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.e_mov_times_gm2ub.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.\
                e_ub2gm_front_burst_len.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.e_num_front_part.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.\
                e_ub2gm_last_burst_len.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_e_num_input_scalar.e_num_last_part.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1

            # ids params
            self.obj_fp32_ids_input_scalar.size.set_as(
                self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                ele_num_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                mov_times_gm2ub_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                front_burst_len_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                last_burst_len_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                ele_num_ub_front_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                ele_num_ub_last_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.ele_num_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                mov_times_gm2ub_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                front_burst_len_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                last_burst_len_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                ele_num_ub_front_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_ids_input_scalar.\
                ele_num_ub_last_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1

            # output init params
            self.obj_fp32_output_init_input_scalar.\
                last_repeat_time_front_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.\
                init_times_front_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.\
                last_repeat_time_last_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.\
                init_times_last_part_front_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.\
                last_repeat_time_front_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.\
                init_times_front_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.\
                last_repeat_time_last_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.\
                init_times_last_part_last_core.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.\
                last_axis_align_front_part.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.\
                last_axis_align_floor.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_fp32_output_init_input_scalar.\
                last_part_vadd_mask.\
                set_as(self.obj_ub_tensor.tiling_ub[input_scalar_index])

        with self.tik_instance.new_stmt_scope():
            self.obj_ub_tensor.tiling_ub = self.tik_instance.Tensor(
                TILING_PARAM_DTYPE, (TILING_PARAMS_NUM,), name="tiling_ub",
                scope=tik.scope_ubuf)
            # mov tiling params from gm to ub
            self.tik_instance.data_move(self.obj_ub_tensor.tiling_ub,
                                        self.obj_gm_tensor.tiling_gm, 0, 1,
                                        TILING_PARAMS_NUM * BYTE_INT32 // \
                                        BYTE_BLOCK,
                                        0, 0)
            # input scalar in flowtable
            input_scalar_index = 0

    def unsorted_segment_sum(self):
        """
        main process of unsorted_segment_sum

        Parameters
        ----------
        None

        Returns:
        -------
        None
        """
        _enable_atomic_add(self.tik_instance)
        with self.tik_instance.new_stmt_scope():
            with self.tik_instance.for_range(0, self.core_num,
                                             block_num=self.core_num) \
                    as block_index:
                with self.tik_instance.if_scope(
                        block_index <
                        self.obj_common_scalar.need_core_num):
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key
                            == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN):
                        # fp32 last axis 32B align
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (self.ub_size // 2 // BYTE_FP32,),
                            name="input_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (self.ub_size // 2 // BYTE_INT32,),
                            name="ids_ub", scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_align_small_e(block_index,
                            self.tik_instance,
                            self.obj_gm_tensor,
                            self.obj_ub_tensor,
                            self.obj_common_scalar,
                            self.obj_fp32_input_data_input_scalar,
                            self.obj_fp32_e_num_input_scalar,
                            self.obj_fp32_ids_input_scalar)
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key
                            == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE):
                        # fp32 last axis is 1
                        def _compute_input_ub_row():
                            one_row_size = BYTE_FP32 + BYTE_INT32 + \
                                           BYTE_FP32 * \
                                           self.fp32_ele_num_one_block
                            return _floor(self.ub_size // one_row_size,
                                          self.fp32_ele_num_one_block)

                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (_compute_input_ub_row(),),
                            name="input_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (_compute_input_ub_row(),),
                            name="ids_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(
                            self.output_dtype, (_compute_input_ub_row(),
                                                self.fp32_ele_num_one_block),
                            name="output_ub", scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_one(block_index,
                            self.tik_instance,
                            self.obj_gm_tensor,
                            self.obj_ub_tensor,
                            self.obj_common_scalar,
                            self.obj_fp32_input_data_input_scalar,
                            self.obj_fp32_e_num_input_scalar,
                            self.obj_fp32_ids_input_scalar,
                            self.obj_fp32_output_init_input_scalar,
                            self.fp32_ele_num_one_block)
                    with self.tik_instance.if_scope(
                        self.obj_common_scalar.select_key
                        == SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MODIFY):
                        # fp32 last axis is 1 modify
                        def _compute_input_ub_row():
                            one_row_size = BYTE_FP32 + BYTE_INT32
                            return _floor(self.ub_size // one_row_size,
                                          MASK_FP32)

                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (_compute_input_ub_row(), ),
                            name="input_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (_compute_input_ub_row(), ),
                            name="ids_ub", scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_one_modify(block_index,
                            self.tik_instance,
                            self.obj_gm_tensor,
                            self.obj_ub_tensor,
                            self.obj_common_scalar,
                            self.obj_fp32_input_data_input_scalar,
                            self.obj_fp32_ids_input_scalar,
                            self.obj_fp32_output_init_input_scalar)
                    with self.tik_instance.if_scope(
                        self.obj_common_scalar.select_key ==
                        SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ONE_MULTI):
                        # fp32 last axis is 1 multi 64
                        def _compute_input_ub_row():
                            one_row_size = BYTE_FP32 + BYTE_INT32
                            return _floor(self.ub_size // one_row_size,
                                          16 * MASK_FP32)
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (_compute_input_ub_row(), ),
                            name="input_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (_compute_input_ub_row(), ),
                            name="ids_ub", scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_one_multi(block_index,
                            self.tik_instance,
                            self.obj_gm_tensor,
                            self.obj_ub_tensor,
                            self.obj_common_scalar,
                            self.obj_fp32_input_data_input_scalar,
                            self.obj_fp32_ids_input_scalar,
                            self.obj_fp32_output_init_input_scalar)
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key ==
                            SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN):
                        # fp32 last axis 32B not align
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (self.ub_size // 3 // BYTE_FP32,),
                            name="input_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (self.ub_size // 3 // BYTE_INT32,),
                            name="ids_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(
                            self.output_dtype,
                            (self.ub_size // 3 // BYTE_FP32,), name="output_ub",
                            scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_not_align_small_e(block_index,
                            self.tik_instance,
                            self.obj_gm_tensor,
                            self.obj_ub_tensor,
                            self.obj_common_scalar,
                            self.obj_fp32_input_data_input_scalar,
                            self.obj_fp32_e_num_input_scalar,
                            self.obj_fp32_ids_input_scalar,
                            self.obj_fp32_output_init_input_scalar,
                            self.fp32_ele_num_one_block)
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key ==
                            SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_ALIGN_BIG_E):
                        # fp32 last axis 32B align and big e
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (self.ub_size // 2 // BYTE_FP32,),
                            name="input_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (self.ub_size // 2 // BYTE_INT32,),
                            name="ids_ub", scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_align_big_e(block_index,
                            self.tik_instance,
                            self.obj_gm_tensor,
                            self.obj_ub_tensor,
                            self.obj_common_scalar,
                            self.obj_fp32_input_data_input_scalar,
                            self.obj_fp32_e_num_input_scalar,
                            self.obj_fp32_ids_input_scalar)
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key ==
                            SELECT_KEY_MODE_FP32_INPUT_LAST_AXIS_NOT_ALIGN_BIG_E):
                        # fp32 last axis 32B not align and big e
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype, (self.ub_size // 3 // BYTE_FP32,),
                            name="input_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (self.ub_size // 3 // BYTE_INT32,),
                            name="ids_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(
                            self.output_dtype,
                            (self.ub_size // 3 // BYTE_FP32,), name="output_ub",
                            scope=tik.scope_ubuf)
                        _tik_atomic_add_last_axis_not_align_big_e(block_index,
                            self.tik_instance,
                            self.obj_gm_tensor,
                            self.obj_ub_tensor,
                            self.obj_common_scalar,
                            self.obj_fp32_input_data_input_scalar,
                            self.obj_fp32_e_num_input_scalar,
                            self.obj_fp32_ids_input_scalar,
                            self.obj_fp32_output_init_input_scalar)
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key ==
                            SELECT_KEY_MODE_INT32_SMALL_E):
                        # int32 last axis 32B align
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype,
                            (self.ub_size // 3 // BYTE_INT32,), name="input_ub",
                            scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (self.ub_size // 3 // BYTE_INT32,),
                            name="ids_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(
                            self.output_dtype,
                            (self.ub_size // 3 // BYTE_INT32,),
                            name="output_ub", scope=tik.scope_ubuf)
                        _tik_int32_add_small_e(block_index,
                            self.tik_instance,
                            self.obj_gm_tensor,
                            self.obj_ub_tensor,
                            self.obj_common_scalar,
                            self.obj_int32_ids_input_scalar,
                            self.obj_int32_e_num_input_scalar)
                    with self.tik_instance.if_scope(
                            self.obj_common_scalar.select_key ==
                            SELECT_KEY_MODE_INT32_BIG_E):
                        # int32 last axis 32B align
                        self.obj_ub_tensor.input_ub = self.tik_instance.Tensor(
                            self.input_dtype,
                            (self.ub_size // 3 // BYTE_INT32,), name="input_ub",
                            scope=tik.scope_ubuf)
                        self.obj_ub_tensor.ids_ub = self.tik_instance.Tensor(
                            self.ids_dtype, (self.ub_size // 3 // BYTE_INT32,),
                            name="ids_ub", scope=tik.scope_ubuf)
                        self.obj_ub_tensor.output_ub = self.tik_instance.Tensor(
                            self.output_dtype,
                            (self.ub_size // 3 // BYTE_INT32,),
                            name="output_ub", scope=tik.scope_ubuf)
                        _tik_int32_add_big_e(block_index,
                                             self.tik_instance,
                                             self.obj_gm_tensor,
                                             self.obj_ub_tensor,
                                             self.obj_common_scalar,
                                             self.obj_int32_ids_input_scalar,
                                             self.obj_int32_e_num_input_scalar)

        _disable_atomic_add(self.tik_instance)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.obj_gm_tensor.input_gm,
                                           self.obj_gm_tensor.ids_gm,
                                           self.obj_gm_tensor.num_segments_gm],
                                   outputs=[self.obj_gm_tensor.output_gm],
                                   flowtable=[self.obj_gm_tensor.tiling_gm])


def _enable_atomic_add(tik_inst):
    """
    enable atomic add

    Parameters
    ----------
    tik_inst: tik instance

    Returns
    -------
    None
    """
    if platform.cce_conf.get_soc_spec("SOC_VERSION") in (
            "Ascend910", "Ascend610"):
        tik_inst.set_atomic_add(1)


def _disable_atomic_add(tik_inst):
    """
    disable atomic add

    Parameters
    ----------
    tik_inst: tik instance

    Returns
    -------
    None
    """
    if platform.cce_conf.get_soc_spec("SOC_VERSION") in (
            "Ascend910", "Ascend610"):
        tik_inst.set_atomic_add(0)


def _tik_get_ub_size(is_double_buffer=True):
    """
    get ub size

    Parameters
    ----------
    is_double_buffer: is_double_buffer flag

    Returns
    -------
    ub_size
    """
    ub_size = platform.cce_conf.get_soc_spec(platform.cce_conf.UB_SIZE)
    if is_double_buffer:
        return ub_size // 2
    return ub_size


def _tik_get_core_num():
    """
    get core num

    Parameters
    ----------
    None

    Returns
    -------
    core num
    """
    return platform.cce_conf.get_soc_spec(platform.cce_conf.CORE_NUM)


def _tik_init_ub_tensor(tik_inst,
                        ub_tensor,
                        init_last_repeat_time,
                        init_times):
    """
    init ub tensor

    Parameters
    ----------
    tik_inst: tik instance
    ub_tensor: ub_tensor
    init_last_repeat_time: last repeat time
    init_times: init times

    Returns
    -------
    None
    """
    with tik_inst.for_range(0, init_times) as init_index:
        with tik_inst.if_scope(init_index == init_times - 1):
            tik_inst.vector_dup(MASK_FP32, ub_tensor[
                init_index * MASK_FP32 * MAX_REPEAT_TIME], 0,
                                init_last_repeat_time, 1, 8)
        with tik_inst.else_scope():
            tik_inst.vector_dup(MASK_FP32, ub_tensor[
                init_index * MASK_FP32 * MAX_REPEAT_TIME], 0, MAX_REPEAT_TIME,
                                1, 8)


def _tik_init_ub_tensor_once(tik_inst,
                             ub_tensor,
                             repeat_time,
                             mask):
    """
    init ub tensor once

    Parameters
    ----------
    tik_inst: tik instance
    ub_tensor: ub_tensor
    repeat_time: repeat time
    mask: mask

    Returns
    -------
    None
    """
    tik_inst.vector_dup(mask, ub_tensor, 0, repeat_time, 1, 8)


def _tik_vadd(tik_inst,
              input_ub,
              output_ub,
              repeat_time,
              mask):
    """
    tik_vadd

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input ub tensor
    output_ub: output ub tensor
    repeat_time: repeat time
    mask: mask

    Returns
    -------
    None
    """
    tik_inst.vadd(mask, output_ub, output_ub, input_ub, repeat_time, 1, 1, 1, 8,
                  8, 8)


def _tik_mov_output_ub2gm_continue(tik_inst,
                                   output_gm,
                                   output_ub,
                                   output_offset_gm,
                                   output_offset_ub,
                                   output_n_burst,
                                   output_burst_len):
    """
    tik_mov_output_ub2gm_continue

    Parameters
    ----------
    tik_inst: tik instance
    output_gm: output gm tensor
    output_ub: output ub tensor
    output_offset_gm: output offset gm
    output_offset_ub: output offset ub
    output_n_burst: n_burst
    output_burst_len: burst_len

    Returns
    -------
    None
    """
    tik_inst.data_move(output_gm[output_offset_gm], output_ub[output_offset_ub],
                       0, output_n_burst, output_burst_len, 0, 0)


def _tik_mov_input_gm2ub_continue(tik_inst,
                                  input_gm,
                                  input_ub,
                                  input_offset_gm,
                                  input_offset_ub,
                                  input_n_burst,
                                  input_burst_len):
    """
    tik_mov_input_gm2ub_continue

    Parameters
    ----------
    tik_inst: tik instance
    input_gm: input gm tensor
    input_ub: input ub tensor
    input_offset_gm: input offset gm
    input_offset_ub: input offset ub
    input_n_burst: n_burst
    input_burst_len: burst_len

    Returns
    -------
    None
    """
    tik_inst.data_move(input_ub[input_offset_ub], input_gm[input_offset_gm], 0,
                       input_n_burst, input_burst_len, 0, 0)


def _tik_mov_input_gm2ub_discrete(tik_inst,
                                  input_gm,
                                  input_ub,
                                  input_offset_gm,
                                  input_offset_ub,
                                  input_n_burst,
                                  input_burst_len,
                                  input_mov_times,
                                  input_ele_num_one_row,
                                  input_ele_num_one_row_align_32B):
    """
    tik_mov_input_gm2ub_discrete

    Parameters
    ----------
    tik_inst: tik instance
    input_gm: input gm tensor
    input_ub: input ub tensor
    input_offset_gm: input offset gm
    input_offset_ub: input offset ub
    input_n_burst: n_burst
    input_burst_len: burst_len
    input_mov_times: mov times
    input_ele_num_one_row: input ele num one row
    input_ele_num_one_row_align_32B: input ele num one row align 32B

    Returns
    -------
    None
    """
    with tik_inst.for_range(0, input_mov_times) as input_mov_index:
        tik_inst.data_move(input_ub[
                               input_offset_ub + input_mov_index *
                               input_ele_num_one_row_align_32B],
                           input_gm[
                               input_offset_gm + input_mov_index *
                               input_ele_num_one_row],
                           0, input_n_burst,
                           input_burst_len, 0, 0)


def _tik_mov_ids_gm2ub(tik_inst,
                       ids_gm,
                       ids_ub,
                       ids_offset_gm,
                       ids_offset_ub,
                       ids_n_burst,
                       ids_burst_len):
    """
    tik_mov_ids_gm2ub

    Parameters
    ----------
    tik_inst: tik instance
    ids_gm: ids_gm tensor
    ids_ub: ids_ub tensor
    ids_offset_gm: ids_offset_gm
    ids_offset_ub: ids_offset_ub
    ids_n_burst: ids_n_burst
    ids_burst_len: ids_burst_len

    Returns
    -------
    None
    """
    tik_inst.data_move(ids_ub[ids_offset_ub], ids_gm[ids_offset_gm], 0,
                       ids_n_burst, ids_burst_len, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_one(tik_inst,
                                              input_ub,
                                              ids_ub,
                                              output_ub,
                                              output_gm,
                                              ub2gm_burst_len,
                                              ids_num,
                                              output_ele_num_one_row,
                                              id_val_scalar):
    """
    tik_atomic_add_ub2gm_by_id_last_axis_one

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    ids_ub: ids_ub tensor
    output_ub: output_ub tensor
    output_gm: output_gm tensor
    ub2gm_burst_len: ub2gm_burst_len
    ids_num: ids_num
    output_ele_num_one_row: output_ele_num_one_row
    id_val_scalar: id_val_scalar

    Returns
    -------
    None
    """
    input_ele_scalar = tik_inst.Scalar(dtype="float32", name="input_ele_scalar")
    with tik_inst.for_range(0, ids_num) as ids_index:
        input_ele_scalar.set_as(input_ub[ids_index])
        output_ub[ids_index * output_ele_num_one_row].set_as(input_ele_scalar)
        id_val_scalar.set_as(ids_ub[ids_index])
        tik_inst.data_move(output_gm[id_val_scalar],
                           output_ub[ids_index * output_ele_num_one_row], 0, 1,
                           ub2gm_burst_len, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_one_modify(tik_inst,
                                                     input_ub,
                                                     ids_ub,
                                                     times_by_mask,
                                                     output_gm,
                                                     id_val_scalar):
    """
    modify float32 atomic add when last axis of input is one

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    ids_ub: ids_ub tensor
    output_gm: output_gm tensor
    id_val_scalar: id_val_scalar

    Returns
    -------
    None
    """
    id_val_fp32 = tik_inst.Scalar(DTYPE_FP32, "id_val_fp32")
    input_val = tik_inst.Scalar(DTYPE_FP32, "input_val")
    neg_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                             tik.scope_ubuf, "neg_ub")
    tik_inst.vector_dup(MASK_FP32, neg_ub[0], NEG_ONE, 1, 1, 8)
    with tik_inst.for_range(0, times_by_mask) as index:
        # times divided by mask
        conv_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                  tik.scope_ubuf, "conv_ub")
        tik_inst.vconv(MASK_FP32, "", conv_ub[0], ids_ub[index * MASK_FP32],
                       1, 1, 1, 8, 8)
        with tik_inst.for_range(0, MASK_FP32) as ids_index:
            # traversal ids
            id_val_fp32.set_as(conv_ub[ids_index])
            with tik_inst.if_scope(id_val_fp32 >= ZERO):
                # new id
                zero_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32,),
                                          tik.scope_ubuf, "zero_ub")
                tik_inst.vector_dup(MASK_FP32, zero_ub[0], ZERO, 1, 1, 8)
                dup_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                         tik.scope_ubuf, "dup_ub")
                tik_inst.vector_dup(MASK_FP32, dup_ub[0], id_val_fp32,
                                    1, 1, 8)
                cmpmask = tik_inst.vcmp_eq(MASK_FP32, dup_ub[0], conv_ub[0],
                                           1, 1)
                tik_inst.vsel(MASK_FP32, 0, conv_ub[0], cmpmask,
                              neg_ub[0], conv_ub[0], 1, 1, 1, 1, 8, 8, 8)
                sel_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                         tik.scope_ubuf, "sel_ub")
                tik_inst.vsel(MASK_FP32, 0, sel_ub[0], cmpmask,
                              input_ub[index * MASK_FP32], zero_ub[0],
                              1, 1, 1, 1, 8, 8, 8)
                cadd_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32,),
                                          tik.scope_ubuf, "cadd_ub")
                tik_inst.vcadd(MASK_FP32, cadd_ub[0], sel_ub[0], 1, 1, 1, 8)
                input_val.set_as(cadd_ub[0])
                zero_ub[0].set_as(input_val)
                id_val_scalar.set_as(ids_ub[index * MASK_FP32 + ids_index])
                tik_inst.data_move(output_gm[id_val_scalar], zero_ub[0], 0,
                                   1, 1, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_one_modify_last_part(tik_inst,
                                                            input_ub,
                                                            ids_ub,
                                                            last_mask,
                                                            output_gm,
                                                            id_val_scalar,
                                                            offset_last_part):
    """
    modify float32 atomic add last part when last axis of input is one

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    ids_ub: ids_ub tensor
    last_mask: last part ele num
    output_gm: output_gm tensor
    id_val_scalar: id_val_scalar
    offset_last_part: offset to last part

    Returns
    -------
    None
    """
    id_val_fp32 = tik_inst.Scalar(DTYPE_FP32, "id_val_fp32")
    input_val = tik_inst.Scalar(DTYPE_FP32, "input_val")
    neg_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                             tik.scope_ubuf, "neg_ub")
    tik_inst.vector_dup(MASK_FP32, neg_ub[0], NEG_ONE, 1, 1, 8)
    conv_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                              tik.scope_ubuf, "conv_ub")
    tik_inst.vector_dup(MASK_FP32, conv_ub[0], NEG_ONE, 1, 1, 8)
    tik_inst.vconv(last_mask, "", conv_ub[0], ids_ub[offset_last_part],
                   1, 1, 1, 8, 8)
    with tik_inst.for_range(0, last_mask) as ids_index:
        # traversal ids
        id_val_fp32.set_as(conv_ub[ids_index])
        with tik_inst.if_scope(id_val_fp32 >= ZERO):
            # new id
            zero_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                      tik.scope_ubuf, "zero_ub")
            tik_inst.vector_dup(MASK_FP32, zero_ub[0], ZERO, 1, 1, 8)
            dup_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                     tik.scope_ubuf, "dup_ub")
            tik_inst.vector_dup(MASK_FP32, dup_ub[0], id_val_fp32,
                                1, 1, 8)
            cmpmask = tik_inst.vcmp_eq(MASK_FP32, dup_ub[0], conv_ub[0],
                                       1, 1)
            tik_inst.vsel(MASK_FP32, 0, conv_ub[0], cmpmask,
                          neg_ub[0], conv_ub[0], 1, 1, 1, 1, 8, 8, 8)
            sel_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                     tik.scope_ubuf, "sel_ub")
            tik_inst.vsel(MASK_FP32, 0, sel_ub[0], cmpmask,
                          input_ub[offset_last_part], zero_ub[0],
                          1, 1, 1, 1, 8, 8, 8)
            cadd_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32,),
                                      tik.scope_ubuf, "cadd_ub")
            tik_inst.vcadd(MASK_FP32, cadd_ub[0], sel_ub[0], 1, 1, 1, 8)
            input_val.set_as(cadd_ub[0])
            zero_ub[0].set_as(input_val)
            id_val_scalar.set_as(ids_ub[offset_last_part + ids_index])
            tik_inst.data_move(output_gm[id_val_scalar], zero_ub[0],
                               0, 1, 1, 0, 0)


def last_axis_one_modify_multi(tik_inst,
                               input_ub,
                               ids_ub,
                               times_by_multi,
                               output_gm,
                               id_val_scalar):
    id_val_fp32 = tik_inst.Scalar(DTYPE_FP32, "id_val_fp32")
    neg_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                             tik.scope_ubuf, "neg_ub")
    tik_inst.vector_dup(MASK_FP32, neg_ub[0], NEG_ONE, 1, 1, 8)
    multi = 4
    with tik_inst.for_range(0, times_by_multi) as index:
        # times divided by mask
        conv_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32 * multi, ),
                                  tik.scope_ubuf, "conv_ub")
        tik_inst.vconv(MASK_FP32, "", conv_ub[0],
                       ids_ub[index * multi * MASK_FP32],
                       multi, 1, 1, 8, 8)
        with tik_inst.for_range(0, multi) as multi_index:
            with tik_inst.for_range(0, MASK_FP32) as ids_index:
                # traversal ids
                id_val_fp32.set_as(conv_ub[multi_index * MASK_FP32 +
                                           ids_index])
                with tik_inst.if_scope(id_val_fp32 >= ZERO):
                    output_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32,),
                                                tik.scope_ubuf, "output_ub")
                    tik_inst.vector_dup(MASK_FP32, output_ub[0], ZERO, 1, 1, 8)
                    # new id
                    zero_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                              tik.scope_ubuf, "zero_ub")
                    tik_inst.vector_dup(MASK_FP32, zero_ub[0], ZERO, 1, 1, 8)
                    dup_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                             tik.scope_ubuf, "dup_ub")
                    tik_inst.vector_dup(MASK_FP32, dup_ub[0], id_val_fp32,
                                        1, 1, 8)
                    with tik_inst.for_range(multi_index, multi) as cmp_index:
                        cmpmask = tik_inst.vcmp_eq(MASK_FP32,
                                                   dup_ub[0],
                                                   conv_ub[cmp_index *
                                                           MASK_FP32], 1, 1)
                        tik_inst.vsel(MASK_FP32, 0,
                                      conv_ub[cmp_index * MASK_FP32], cmpmask,
                                      neg_ub[0],
                                      conv_ub[cmp_index * MASK_FP32],
                                      1, 1, 1, 1, 8, 8, 8)
                        sel_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                                 tik.scope_ubuf, "sel_ub")
                        tik_inst.vsel(MASK_FP32, 0, sel_ub[0], cmpmask,
                                      input_ub[index * multi * MASK_FP32 +
                                               cmp_index * MASK_FP32],
                                      zero_ub[0],
                                      1, 1, 1, 1, 8, 8, 8)
                        cadd_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                                  tik.scope_ubuf, "cadd_ub")
                        tik_inst.vector_dup(MASK_FP32, cadd_ub[0], ZERO,
                                            1, 1, 8)
                        tik_inst.vcadd(MASK_FP32, cadd_ub[0],
                                       sel_ub[0], 1, 1, 1, 8)
                        tik_inst.vadd(MASK_FP32, output_ub[0],
                                      output_ub[0], cadd_ub[0], 1, 1,
                                      1, 1, 8, 8, 8)
                    id_val_scalar.set_as(ids_ub[index * multi * MASK_FP32 +
                                                multi_index * MASK_FP32 +
                                                ids_index])
                    tik_inst.data_move(output_gm[id_val_scalar], output_ub[0],
                                       0, 1, 1, 0, 0)


def last_axis_one_modify_single(tik_inst,
                                input_ub,
                                ids_ub,
                                times_by_mask,
                                output_gm,
                                id_val_scalar,
                                offset):
    id_val_fp32 = tik_inst.Scalar(DTYPE_FP32, "id_val_fp32")
    input_val = tik_inst.Scalar(DTYPE_FP32, "input_val")
    neg_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                             tik.scope_ubuf, "neg_ub")
    tik_inst.vector_dup(MASK_FP32, neg_ub[0], NEG_ONE, 1, 1, 8)
    with tik_inst.for_range(0, times_by_mask) as index:
        # times divided by mask
        conv_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                  tik.scope_ubuf, "conv_ub")
        tik_inst.vconv(MASK_FP32, "", conv_ub[0],
                       ids_ub[offset + index * MASK_FP32],
                       1, 1, 1, 8, 8)
        with tik_inst.for_range(0, MASK_FP32) as ids_index:
            # traversal ids
            id_val_fp32.set_as(conv_ub[ids_index])
            with tik_inst.if_scope(id_val_fp32 >= ZERO):
                # new id
                zero_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32,),
                                          tik.scope_ubuf, "zero_ub")
                tik_inst.vector_dup(MASK_FP32, zero_ub[0], ZERO, 1, 1, 8)
                dup_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                         tik.scope_ubuf, "dup_ub")
                tik_inst.vector_dup(MASK_FP32, dup_ub[0], id_val_fp32,
                                    1, 1, 8)
                cmpmask = tik_inst.vcmp_eq(MASK_FP32, dup_ub[0], conv_ub[0],
                                           1, 1)
                tik_inst.vsel(MASK_FP32, 0, conv_ub[0], cmpmask,
                              neg_ub[0], conv_ub[0], 1, 1, 1, 1, 8, 8, 8)
                sel_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32, ),
                                         tik.scope_ubuf, "sel_ub")
                tik_inst.vsel(MASK_FP32, 0, sel_ub[0], cmpmask,
                              input_ub[offset + index * MASK_FP32], zero_ub[0],
                              1, 1, 1, 1, 8, 8, 8)
                cadd_ub = tik_inst.Tensor(DTYPE_FP32, (MASK_FP32,),
                                          tik.scope_ubuf, "cadd_ub")
                tik_inst.vcadd(MASK_FP32, cadd_ub[0], sel_ub[0], 1, 1, 1, 8)
                input_val.set_as(cadd_ub[0])
                zero_ub[0].set_as(input_val)
                id_val_scalar.set_as(
                    ids_ub[offset + index * MASK_FP32 + ids_index])
                tik_inst.data_move(output_gm[id_val_scalar], zero_ub[0], 0,
                                   1, 1, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_align(tik_inst,
                                                input_ub,
                                                output_gm,
                                                ub2gm_burst_len,
                                                input_ub_offset,
                                                output_gm_offset):
    """
    tik_atomic_add_ub2gm_by_id_last_axis_align

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    output_gm: output_gm tensor
    ub2gm_burst_len: ub2gm_burst_len
    input_ub_offset: input_ub_offset
    output_gm_offset: output_gm_offset

    Returns
    -------
    None
    """
    tik_inst.data_move(output_gm[output_gm_offset], input_ub[input_ub_offset],
                       0, 1, ub2gm_burst_len, 0, 0)


def _tik_atomic_add_ub2gm_by_id_last_axis_not_align(tik_inst,
                                                    input_ub,
                                                    output_ub,
                                                    output_gm,
                                                    input_ub_offset,
                                                    output_ub_offset,
                                                    output_gm_offset,
                                                    vadd_mask):
    """
    tik_atomic_add_ub2gm_by_id_last_axis_not_align

    Parameters
    ----------
    tik_inst: tik instance
    input_ub: input_ub tensor
    output_ub: output_ub tensor
    output_gm: output_gm tensor
    input_ub_offset: input_ub_offset
    output_ub_offset: output_ub_offset
    output_gm_offset: output_gm_offset
    vadd_mask: vadd_mask

    Returns
    -------
    None
    """
    tik_inst.vadd(vadd_mask, output_ub[output_ub_offset],
                  input_ub[input_ub_offset], output_ub[output_ub_offset], 1, 1,
                  1, 1, 8, 8, 8)
    tik_inst.data_move(output_gm[output_gm_offset], output_ub[output_ub_offset],
                       0, 1, 1, 0, 0)


def _tik_atomic_add_last_axis_align_small_e(block_index,
                                            tik_inst,
                                            obj_gm_tensor,
                                            obj_ub_tensor,
                                            obj_common_scalar,
                                            obj_fp32_input_data_input_scalar,
                                            obj_fp32_e_num_input_scalar,
                                            obj_fp32_ids_input_scalar):
    """
    _tik_atomic_add_last_axis_align_small_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(
            block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,\
        obj_fp32_ids_input_scalar.
                mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index <
                    obj_fp32_ids_input_scalar.
                            mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                                        obj_fp32_input_data_input_scalar.
                    mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(
                            input_mov_tims_front_part_front_core_index <
                            obj_fp32_input_data_input_scalar.
                    mov_times_gm2ub_front_part_front_core -
                            1):
                        # front part ids front part front core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                            ele_num_front_core + \
                            ids_mov_times_front_core_index * \
                            obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_front_core * \
                            obj_fp32_e_num_input_scalar.\
                            e_num + \
                            input_mov_tims_front_part_front_core_index * \
                            obj_fp32_input_data_input_scalar.\
                            front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.\
                            front_burst_len_front_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst,
                                                      obj_gm_tensor.input_gm,
                                                      obj_ub_tensor.input_ub,
                                                      input_offset_gm,
                                                      input_offset_ub,
                                                      input_n_burst,
                                                      input_burst_len)
                        with tik_inst.for_range(0,\
                            obj_fp32_input_data_input_scalar.
                            front_rows_front_part_front_core) as \
                    rows_index:
                            # visit ids
                            input_ub_offset = \
                    rows_index * \
                    obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_front_core_index *
                                obj_fp32_input_data_input_scalar.
                                front_rows_front_part_front_core +
                    rows_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.\
                                                   e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                    with tik_inst.if_scope(
                        input_mov_tims_front_part_front_core_index ==
                        obj_fp32_input_data_input_scalar.
                        mov_times_gm2ub_front_part_front_core - 1):
                        # last part ids front part front core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                            ele_num_front_core + \
                            ids_mov_times_front_core_index * \
                            obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_front_core * \
                            obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_front_part_front_core_index * \
                            obj_fp32_input_data_input_scalar.\
                            front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.\
                            last_burst_len_front_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst,
                                                      obj_gm_tensor.input_gm,
                                                      obj_ub_tensor.input_ub,
                                                      input_offset_gm,
                                                      input_offset_ub,
                                                      input_n_burst,
                                                      input_burst_len)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                            last_rows_front_part_front_core) as \
                    rows_index:
                            # visit ids
                            input_ub_offset = \
                    rows_index * \
                    obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                    input_mov_tims_front_part_front_core_index *
                    obj_fp32_input_data_input_scalar.
                    front_rows_front_part_front_core +
                    rows_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index ==
                    obj_fp32_ids_input_scalar.
                            mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                    obj_fp32_ids_input_scalar.\
                    ele_num_front_core + \
                    ids_mov_times_front_core_index * \
                    obj_fp32_ids_input_scalar.\
                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                    mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(
                            input_mov_tims_last_part_front_core_index <
                            obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_last_part_front_core -
                            1):
                        # front part ids last part front core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                            ele_num_front_core + \
                            ids_mov_times_front_core_index * \
                            obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_front_core * \
                            obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_last_part_front_core_index * \
                            obj_fp32_input_data_input_scalar.\
                            front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.\
                            front_burst_len_last_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst,
                                                      obj_gm_tensor.input_gm,
                                                      obj_ub_tensor.input_ub,
                                                      input_offset_gm,
                                                      input_offset_ub,
                                                      input_n_burst,
                                                      input_burst_len)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                            front_rows_last_part_front_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = rows_index * \
                                obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_front_core_index *
                                obj_fp32_input_data_input_scalar.
                                front_rows_last_part_front_core +
                                rows_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.\
                                e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                    with tik_inst.if_scope(
                            input_mov_tims_last_part_front_core_index ==
                            obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_last_part_front_core -
                            1):
                        # last part ids last part front core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                            ele_num_front_core + \
                            ids_mov_times_front_core_index * \
                            obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_front_core * \
                            obj_fp32_e_num_input_scalar.\
                            e_num + \
                            input_mov_tims_last_part_front_core_index * \
                            obj_fp32_input_data_input_scalar.\
                            front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.\
                            last_burst_len_last_part_front_core
                        _tik_mov_input_gm2ub_continue(tik_inst,
                                                      obj_gm_tensor.input_gm,
                                                      obj_ub_tensor.input_ub,
                                                      input_offset_gm,
                                                      input_offset_ub,
                                                      input_n_burst,
                                                      input_burst_len)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                            last_rows_last_part_front_core) as \
                            rows_index:
                            # visit ids
                            input_ub_offset = rows_index * \
                                obj_fp32_e_num_input_scalar.e_num
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_front_core_index *
                                obj_fp32_input_data_input_scalar.
                                front_rows_last_part_front_core +
                                rows_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.\
                                e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
    with tik_inst.if_scope(
            block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.
                                mov_times_gm2ub_last_core) as \
                                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index <
                    obj_fp32_ids_input_scalar.
                            mov_times_gm2ub_last_core - 1):
                # ids front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                    mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(
                            input_mov_tims_front_part_last_core_index <
                            obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_front_part_last_core -
                            1):
                        # front part ids front part last core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                            ele_num_front_core + \
                            ids_mov_times_last_core_index * \
                            obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_last_core * \
                            obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_front_part_last_core_index * \
                            obj_fp32_input_data_input_scalar.\
                            front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.\
                            front_burst_len_front_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst,
                                                      obj_gm_tensor.input_gm,
                                                      obj_ub_tensor.input_ub,
                                                      input_offset_gm,
                                                      input_offset_ub,
                                                      input_n_burst,
                                                      input_burst_len)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                            front_rows_front_part_last_core) as \
                            rows_index:
                            # visit ids
                            input_ub_offset = \
                                rows_index * \
                                obj_fp32_e_num_input_scalar.\
                                    e_num
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_last_core_index *
                                obj_fp32_input_data_input_scalar.
                                front_rows_front_part_last_core +
                                rows_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.\
                                e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                    with tik_inst.if_scope(
                            input_mov_tims_front_part_last_core_index ==
                            obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_front_part_last_core -
                            1):
                        # last part ids front part last core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.\
                                          ele_num_front_core + \
                                          ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_last_core * \
                            obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_front_part_last_core_index * \
                            obj_fp32_input_data_input_scalar.\
                            front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.\
                            last_burst_len_front_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst,
                                                      obj_gm_tensor.input_gm,
                                                      obj_ub_tensor.input_ub,
                                                      input_offset_gm,
                                                      input_offset_ub,
                                                      input_n_burst,
                                                      input_burst_len)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                                last_rows_front_part_last_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = rows_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_last_core_index *
                                obj_fp32_input_data_input_scalar.
                                front_rows_front_part_last_core +
                                rows_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.\
                                    e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index ==
                    obj_fp32_ids_input_scalar.
                            mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_last_part_last_core) as\
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(
                            input_mov_tims_last_part_last_core_index <
                            obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_last_part_last_core -
                            1):
                        # front part ids last part last core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                            ele_num_front_core + \
                            ids_mov_times_last_core_index * \
                            obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_last_core * \
                            obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_last_part_last_core_index * \
                            obj_fp32_input_data_input_scalar.\
                            front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.\
                            front_burst_len_last_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst,
                                                      obj_gm_tensor.input_gm,
                                                      obj_ub_tensor.input_ub,
                                                      input_offset_gm,
                                                      input_offset_ub,
                                                      input_n_burst,
                                                      input_burst_len)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                                front_rows_last_part_last_core) as \
                                rows_index:
                            # visit ids
                            input_ub_offset = rows_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_last_core_index *
                                obj_fp32_input_data_input_scalar.
                                front_rows_last_part_last_core +
                                                     rows_index])
                            output_gm_offset = id_val_scalar * \
                                               obj_fp32_e_num_input_scalar.\
                                                   e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                    with tik_inst.if_scope(
                            input_mov_tims_last_part_last_core_index ==
                            obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_last_part_last_core -
                            1):
                        # last part ids last part last core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                            ele_num_front_core + \
                            ids_mov_times_last_core_index * \
                            obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_last_core * \
                            obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_last_part_last_core_index * \
                            obj_fp32_input_data_input_scalar.\
                            front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_input_data_input_scalar.\
                            last_burst_len_last_part_last_core
                        _tik_mov_input_gm2ub_continue(tik_inst,
                                                      obj_gm_tensor.input_gm,
                                                      obj_ub_tensor.input_ub,
                                                      input_offset_gm,
                                                      input_offset_ub,
                                                      input_n_burst,
                                                      input_burst_len)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                            last_rows_last_part_last_core) as \
                            rows_index:
                            # visit ids
                            input_ub_offset = rows_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_last_core_index *
                                obj_fp32_input_data_input_scalar.
                                front_rows_last_part_last_core +
                                rows_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.\
                                e_num
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)


def _tik_atomic_add_last_axis_one(block_index,
                                  tik_inst,
                                  obj_gm_tensor,
                                  obj_ub_tensor,
                                  obj_common_scalar,
                                  obj_fp32_input_data_input_scalar,
                                  obj_fp32_e_num_input_scalar,
                                  obj_fp32_ids_input_scalar,
                                  obj_fp32_output_init_input_scalar,
                                  fp32_ele_num_one_block):
    """
    _tik_atomic_add_last_axis_one

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    obj_fp32_output_init_input_scalar: obj_fp32_output_init_input_scalar
    fp32_ele_num_one_block: fp32_ele_num_one_block

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(
            block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.
                                mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index <
                    obj_fp32_ids_input_scalar.
                            mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input front part front core
                input_offset_gm = block_index * \
                                  obj_fp32_input_data_input_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_fp32_input_data_input_scalar.\
                    front_burst_len_front_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # init output
                _tik_init_ub_tensor(tik_inst,
                                    obj_ub_tensor.output_ub,
                                    obj_fp32_output_init_input_scalar.
                                    last_repeat_time_front_part_front_core,
                                    obj_fp32_output_init_input_scalar.
                                    init_times_front_part_front_core)
                # ub2gm by id
                _tik_atomic_add_ub2gm_by_id_last_axis_one(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_ub_tensor.output_ub,
                    obj_gm_tensor.output_gm,
                    obj_fp32_e_num_input_scalar.
                        e_ub2gm_front_burst_len,
                    obj_fp32_input_data_input_scalar.
                        front_rows_front_part_front_core,
                    fp32_ele_num_one_block,
                    id_val_scalar)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index ==
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input front part front core
                input_offset_gm = block_index * \
                                  obj_fp32_input_data_input_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_fp32_input_data_input_scalar.\
                    front_burst_len_last_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # init output
                _tik_init_ub_tensor(tik_inst,
                                    obj_ub_tensor.output_ub,
                                    obj_fp32_output_init_input_scalar.
                                    last_repeat_time_last_part_front_core,
                                    obj_fp32_output_init_input_scalar.
                                    init_times_last_part_front_core)
                # ub2gm by id
                _tik_atomic_add_ub2gm_by_id_last_axis_one(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_ub_tensor.output_ub,
                    obj_gm_tensor.output_gm,
                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                    obj_fp32_input_data_input_scalar.
                        front_rows_last_part_front_core,
                    fp32_ele_num_one_block,
                    id_val_scalar)
    with tik_inst.if_scope(
            block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.
                                        mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index <
                    obj_fp32_ids_input_scalar.
                            mov_times_gm2ub_last_core - 1):
                # front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input front part last core
                input_offset_gm = block_index * \
                                  obj_fp32_input_data_input_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_fp32_input_data_input_scalar.\
                    front_burst_len_front_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # init output
                _tik_init_ub_tensor(tik_inst,
                                    obj_ub_tensor.output_ub,
                                    obj_fp32_output_init_input_scalar.
                                        last_repeat_time_front_part_last_core,
                                    obj_fp32_output_init_input_scalar.
                                        init_times_front_part_last_core)
                # ub2gm by id
                _tik_atomic_add_ub2gm_by_id_last_axis_one(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_ub_tensor.output_ub,
                    obj_gm_tensor.output_gm,
                    obj_fp32_e_num_input_scalar.e_ub2gm_front_burst_len,
                    obj_fp32_input_data_input_scalar.
                        front_rows_front_part_last_core,
                    fp32_ele_num_one_block,
                    id_val_scalar)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index ==
                        obj_fp32_ids_input_scalar.
                                mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input last part last core
                input_offset_gm = block_index * \
                                  obj_fp32_input_data_input_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_fp32_input_data_input_scalar.\
                    front_burst_len_last_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # init output
                _tik_init_ub_tensor(tik_inst,
                                    obj_ub_tensor.output_ub,
                                    obj_fp32_output_init_input_scalar.
                                    last_repeat_time_last_part_last_core,
                                    obj_fp32_output_init_input_scalar.
                                    init_times_last_part_last_core)
                # ub2gm by id
                _tik_atomic_add_ub2gm_by_id_last_axis_one(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_ub_tensor.output_ub,
                    obj_gm_tensor.output_gm,
                    obj_fp32_e_num_input_scalar.
                        e_ub2gm_front_burst_len,
                    obj_fp32_input_data_input_scalar.
                        front_rows_last_part_last_core,
                    fp32_ele_num_one_block,
                    id_val_scalar)


def _tik_atomic_add_last_axis_one_modify(block_index,
                                         tik_inst,
                                         obj_gm_tensor,
                                         obj_ub_tensor,
                                         obj_common_scalar,
                                         obj_input_data_scalar,
                                         obj_fp32_ids_input_scalar,
                                         obj_output_init_scalar):
    """
    modify float32 atomic add when last axis of input is one

    Parameters
    ----------
    block_index: block index
    tik_inst: tik instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_input_data_scalar: obj_input_data_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    obj_output_init_scalar: obj_output_init_scalar

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
            obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(ids_mov_times_front_core_index <
                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input front part front core
                input_offset_gm = block_index * \
                                  obj_input_data_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.\
                    front_burst_len_front_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # cadd front part front core
                _tik_atomic_add_ub2gm_by_id_last_axis_one_modify(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.
                        init_times_front_part_front_core,
                    obj_gm_tensor.output_gm,
                    id_val_scalar)
            with tik_inst.if_scope(ids_mov_times_front_core_index ==
                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input last part front core
                input_offset_gm = block_index * \
                                  obj_input_data_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.\
                    front_burst_len_last_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # cadd last part front core
                _tik_atomic_add_ub2gm_by_id_last_axis_one_modify(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.
                        init_times_last_part_front_core,
                    obj_gm_tensor.output_gm,
                    id_val_scalar)
    with tik_inst.if_scope(block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
            obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            # ids tiling by ub last core
            with tik_inst.if_scope(ids_mov_times_last_core_index <
                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # ids front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input front part last core
                input_offset_gm = block_index * \
                                  obj_input_data_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.\
                    front_burst_len_front_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # cadd front part last core
                _tik_atomic_add_ub2gm_by_id_last_axis_one_modify(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.
                        init_times_front_part_last_core,
                    obj_gm_tensor.output_gm,
                    id_val_scalar)
            with tik_inst.if_scope(ids_mov_times_last_core_index ==
                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # ids last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input last part last core
                input_offset_gm = block_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.\
                    front_burst_len_last_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # cadd last part last core
                _tik_atomic_add_ub2gm_by_id_last_axis_one_modify(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.
                        init_times_last_part_last_core - 1,
                    obj_gm_tensor.output_gm,
                    id_val_scalar)
                # last part
                offset_last_part = (obj_output_init_scalar.
                                    init_times_last_part_last_core - 1) * \
                                   MASK_FP32
                _tik_atomic_add_ub2gm_by_id_last_axis_one_modify_last_part(
                    tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.last_part_vadd_mask,
                    obj_gm_tensor.output_gm,
                    id_val_scalar,
                    offset_last_part)


def _tik_atomic_add_last_axis_one_multi(block_index,
                                        tik_inst,
                                        obj_gm_tensor,
                                        obj_ub_tensor,
                                        obj_common_scalar,
                                        obj_input_data_scalar,
                                        obj_fp32_ids_input_scalar,
                                        obj_output_init_scalar):
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
            obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(ids_mov_times_front_core_index <
                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input front part front core
                input_offset_gm = block_index * \
                                  obj_input_data_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.\
                    front_burst_len_front_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # cadd front part front core
                # multi 64 part
                last_axis_one_modify_multi(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.
                        init_times_front_part_front_core,
                    obj_gm_tensor.output_gm,
                    id_val_scalar)
                # single 64 part
                offset_multi = obj_output_init_scalar.\
                                   init_times_front_part_front_core * \
                               MULTI * MASK_FP32
                times_by_mask = obj_output_init_scalar.\
                    last_repeat_time_front_part_front_core
                last_axis_one_modify_single(tik_inst,
                                            obj_ub_tensor.input_ub,
                                            obj_ub_tensor.ids_ub,
                                            times_by_mask,
                                            obj_gm_tensor.output_gm,
                                            id_val_scalar,
                                            offset_multi)
            with tik_inst.if_scope(ids_mov_times_front_core_index ==
                obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input last part front core
                input_offset_gm = block_index * \
                                  obj_input_data_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_front_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_front_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.\
                    front_burst_len_last_part_front_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # cadd last part front core
                # multi 64 part
                last_axis_one_modify_multi(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.
                        init_times_last_part_front_core,
                    obj_gm_tensor.output_gm,
                    id_val_scalar)
                # single 64 part
                offset_multi = obj_output_init_scalar.\
                                   init_times_last_part_front_core * \
                               MULTI * MASK_FP32
                times_by_mask = obj_output_init_scalar.\
                    last_repeat_time_last_part_front_core
                last_axis_one_modify_single(tik_inst,
                                            obj_ub_tensor.input_ub,
                                            obj_ub_tensor.ids_ub,
                                            times_by_mask,
                                            obj_gm_tensor.output_gm,
                                            id_val_scalar,
                                            offset_multi)
    with tik_inst.if_scope(block_index ==
                           obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0, obj_fp32_ids_input_scalar.
            mov_times_gm2ub_last_core) as ids_mov_times_last_core_index:
            # ids tiling by ub last core
            with tik_inst.if_scope(ids_mov_times_last_core_index <
                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # ids front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input front part last core
                input_offset_gm = block_index * \
                                  obj_input_data_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.\
                    front_burst_len_front_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # cadd front part last core
                # multi 64 part
                last_axis_one_modify_multi(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.
                        init_times_front_part_last_core,
                    obj_gm_tensor.output_gm,
                    id_val_scalar)
                # single 64 part
                offset_multi = obj_output_init_scalar. \
                                   init_times_front_part_last_core * \
                               MULTI * MASK_FP32
                times_by_mask = obj_output_init_scalar.\
                    last_repeat_time_front_part_last_core
                last_axis_one_modify_single(tik_inst,
                                            obj_ub_tensor.input_ub,
                                            obj_ub_tensor.ids_ub,
                                            times_by_mask,
                                            obj_gm_tensor.output_gm,
                                            id_val_scalar,
                                            offset_multi)
            with tik_inst.if_scope(ids_mov_times_last_core_index ==
                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # ids last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                # input last part last core
                input_offset_gm = block_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_front_core + \
                                  ids_mov_times_last_core_index * \
                                  obj_fp32_ids_input_scalar.\
                                      ele_num_ub_front_part_last_core
                input_offset_ub = 0
                input_n_burst = 1
                input_burst_len = obj_input_data_scalar.\
                    front_burst_len_last_part_last_core
                _tik_mov_input_gm2ub_continue(tik_inst,
                                              obj_gm_tensor.input_gm,
                                              obj_ub_tensor.input_ub,
                                              input_offset_gm,
                                              input_offset_ub,
                                              input_n_burst,
                                              input_burst_len)
                # cadd last part last core
                # multi 64 part
                last_axis_one_modify_multi(tik_inst,
                    obj_ub_tensor.input_ub,
                    obj_ub_tensor.ids_ub,
                    obj_output_init_scalar.
                        init_times_last_part_last_core,
                    obj_gm_tensor.output_gm,
                    id_val_scalar)
                # single 64 part
                offset_multi = obj_output_init_scalar.\
                    init_times_last_part_last_core * MULTI * MASK_FP32
                times_by_mask = obj_output_init_scalar.\
                    last_repeat_time_last_part_last_core
                last_axis_one_modify_single(tik_inst,
                                            obj_ub_tensor.input_ub,
                                            obj_ub_tensor.ids_ub,
                                            times_by_mask,
                                            obj_gm_tensor.output_gm,
                                            id_val_scalar,
                                            offset_multi)
                # last mask part
                with tik_inst.if_scope(obj_output_init_scalar.
                                               last_part_vadd_mask > 0):
                    offset_last_part = offset_multi + times_by_mask * MASK_FP32
                    _tik_atomic_add_ub2gm_by_id_last_axis_one_modify_last_part(
                        tik_inst,
                        obj_ub_tensor.input_ub,
                        obj_ub_tensor.ids_ub,
                        obj_output_init_scalar.last_part_vadd_mask,
                        obj_gm_tensor.output_gm,
                        id_val_scalar,
                        offset_last_part)


def _tik_atomic_add_last_axis_not_align_small_e(block_index,
                                            tik_inst,
                                            obj_gm_tensor,
                                            obj_ub_tensor,
                                            obj_common_scalar,
                                            obj_fp32_input_data_input_scalar,
                                            obj_fp32_e_num_input_scalar,
                                            obj_fp32_ids_input_scalar,
                                            obj_fp32_output_init_input_scalar,
                                            fp32_ele_num_one_block):
    """
    _tik_atomic_add_last_axis_not_align_small_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    obj_fp32_output_init_input_scalar: obj_fp32_output_init_input_scalar
    fp32_ele_num_one_block: fp32_ele_num_one_block

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(
            block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.
                                        mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index <
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                    mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(
                            input_mov_tims_front_part_front_core_index <
                            obj_fp32_input_data_input_scalar.
                                    mov_times_gm2ub_front_part_front_core - 1):
                        # front part ids front part front core
                        input_offset_gm = block_index * \
                                          obj_fp32_input_data_input_scalar.\
                                              ele_num_front_core + \
                                          ids_mov_times_front_core_index * \
                                          obj_fp32_ids_input_scalar.\
                                          ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                                    input_mov_tims_front_part_front_core_index * \
                                          obj_fp32_input_data_input_scalar.\
                                        front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.\
                            front_rows_front_part_front_core
                        input_ele_num_one_row = obj_fp32_e_num_input_scalar.\
                            e_num
                        input_ele_num_one_row_align_32B = \
                            obj_fp32_output_init_input_scalar.\
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst,
                            obj_gm_tensor.input_gm,
                            obj_ub_tensor.input_ub,
                            input_offset_gm,
                            input_offset_ub,
                            input_n_burst,
                            input_burst_len,
                            input_mov_times,
                            input_ele_num_one_row,
                            input_ele_num_one_row_align_32B)
                        _tik_init_ub_tensor(tik_inst,
                            obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.
                            last_repeat_time_front_part_front_core,
                            obj_fp32_output_init_input_scalar.
                            init_times_front_part_front_core)
                        with tik_inst.for_range(0,
                                obj_fp32_input_data_input_scalar.
                                        front_rows_front_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_front_core_index *
                                obj_fp32_input_data_input_scalar.
                                    front_rows_front_part_front_core +
                                rows_index])
                            # align part
                            with tik_inst.if_scope(
                                    obj_fp32_e_num_input_scalar.
                                            e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                    obj_fp32_output_init_input_scalar.\
                                        last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                    obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst,
                                    obj_ub_tensor.input_ub,
                                    obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.
                                        e_ub2gm_front_burst_len,
                                    input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                            obj_fp32_output_init_input_scalar.\
                                                last_axis_align_floor + \
                                            obj_fp32_output_init_input_scalar.\
                                                last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                            obj_fp32_e_num_input_scalar.\
                                                e_num + \
                                            obj_fp32_output_init_input_scalar.\
                                                last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.\
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)
                    with tik_inst.if_scope(
                            input_mov_tims_front_part_front_core_index ==
                            obj_fp32_input_data_input_scalar.
                                    mov_times_gm2ub_front_part_front_core - 1):
                        # last part ids front part front core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                                ele_num_front_core + \
                            ids_mov_times_front_core_index * \
                            obj_fp32_ids_input_scalar.\
                                ele_num_ub_front_part_front_core * \
                            obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_front_part_front_core_index * \
                            obj_fp32_input_data_input_scalar.\
                                front_ele_num_ub_front_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.\
                            last_rows_front_part_front_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32B = \
                            obj_fp32_output_init_input_scalar.\
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst,
                            obj_gm_tensor.input_gm,
                            obj_ub_tensor.input_ub,
                            input_offset_gm,
                            input_offset_ub,
                            input_n_burst,
                            input_burst_len,
                            input_mov_times,
                            input_ele_num_one_row,
                            input_ele_num_one_row_align_32B)
                        _tik_init_ub_tensor(tik_inst,
                            obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.
                                last_repeat_time_front_part_front_core,
                            obj_fp32_output_init_input_scalar.
                                init_times_front_part_front_core)
                        with tik_inst.for_range(0,
                                obj_fp32_input_data_input_scalar.
                                        last_rows_front_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_front_core_index *
                                obj_fp32_input_data_input_scalar.
                                    front_rows_front_part_front_core +
                                rows_index])
                            # align part
                            with tik_inst.if_scope(
                                    obj_fp32_e_num_input_scalar.
                                            e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                    obj_fp32_output_init_input_scalar.\
                                        last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                    obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst,
                                    obj_ub_tensor.input_ub,
                                    obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.
                                        e_ub2gm_front_burst_len,
                                    input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_floor + \
                                obj_fp32_output_init_input_scalar.\
                                                  last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                obj_fp32_output_init_input_scalar.\
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.\
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index ==
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                                    obj_fp32_input_data_input_scalar.
                                    mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(
                            input_mov_tims_last_part_front_core_index <
                            obj_fp32_input_data_input_scalar.
                                    mov_times_gm2ub_last_part_front_core - 1):
                        # front part ids last part front core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                                ele_num_front_core + \
                            ids_mov_times_front_core_index * \
                            obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_front_core * \
                            obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_last_part_front_core_index * \
                            obj_fp32_input_data_input_scalar.\
                            front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.\
                            front_rows_last_part_front_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32B = \
                            obj_fp32_output_init_input_scalar.\
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst,
                            obj_gm_tensor.input_gm,
                            obj_ub_tensor.input_ub,
                            input_offset_gm,
                            input_offset_ub,
                            input_n_burst,
                            input_burst_len,
                            input_mov_times,
                            input_ele_num_one_row,
                            input_ele_num_one_row_align_32B)
                        _tik_init_ub_tensor(tik_inst,
                            obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.
                                last_repeat_time_last_part_front_core,
                            obj_fp32_output_init_input_scalar.
                                init_times_last_part_front_core)
                        with tik_inst.for_range(0,
                                obj_fp32_input_data_input_scalar.
                                        front_rows_last_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_front_core_index *
                                obj_fp32_input_data_input_scalar.
                                    front_rows_last_part_front_core +
                                rows_index])
                            # align part
                            with tik_inst.if_scope(
                                    obj_fp32_e_num_input_scalar.
                                            e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                    obj_fp32_output_init_input_scalar.\
                                        last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                    obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst,
                                    obj_ub_tensor.input_ub,
                                    obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.
                                        e_ub2gm_front_burst_len,
                                    input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_floor + \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                obj_fp32_output_init_input_scalar.\
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.\
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)
                    with tik_inst.if_scope(
                            input_mov_tims_last_part_front_core_index ==
                            obj_fp32_input_data_input_scalar.
                                    mov_times_gm2ub_last_part_front_core - 1):
                        # last part ids last part front core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                                ele_num_front_core + \
                            ids_mov_times_front_core_index * \
                            obj_fp32_ids_input_scalar.\
                                ele_num_ub_front_part_front_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_last_part_front_core_index * \
                                obj_fp32_input_data_input_scalar.\
                                front_ele_num_ub_last_part_front_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.\
                            last_rows_last_part_front_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32B = \
                            obj_fp32_output_init_input_scalar.\
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst,
                            obj_gm_tensor.input_gm,
                            obj_ub_tensor.input_ub,
                            input_offset_gm,
                            input_offset_ub,
                            input_n_burst,
                            input_burst_len,
                            input_mov_times,
                            input_ele_num_one_row,
                            input_ele_num_one_row_align_32B)
                        _tik_init_ub_tensor(tik_inst,
                            obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.
                            last_repeat_time_last_part_front_core,
                            obj_fp32_output_init_input_scalar.
                                            init_times_last_part_front_core)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                                    last_rows_last_part_front_core) as \
                                rows_index:
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_front_core_index *
                                obj_fp32_input_data_input_scalar.
                                    front_rows_last_part_front_core +
                                rows_index])
                            # align part
                            with tik_inst.if_scope(
                                    obj_fp32_e_num_input_scalar.
                                            e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                    obj_fp32_output_init_input_scalar.\
                                        last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                    obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst,
                                    obj_ub_tensor.input_ub,
                                    obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.
                                        e_ub2gm_front_burst_len,
                                    input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_floor + \
                                obj_fp32_output_init_input_scalar.\
                                                  last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                obj_fp32_output_init_input_scalar.\
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.\
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)
    with tik_inst.if_scope(
            block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index <
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                    mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(
                            input_mov_tims_front_part_last_core_index <
                            obj_fp32_input_data_input_scalar.
                                    mov_times_gm2ub_front_part_last_core - 1):
                        # front part ids front part last core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                                ele_num_front_core + \
                            ids_mov_times_last_core_index * \
                                          obj_fp32_ids_input_scalar.\
                                        ele_num_ub_front_part_last_core * \
                                        obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_front_part_last_core_index * \
                                        obj_fp32_input_data_input_scalar.\
                                        front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.\
                            front_rows_front_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32B = \
                            obj_fp32_output_init_input_scalar.\
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst,
                            obj_gm_tensor.input_gm,
                            obj_ub_tensor.input_ub,
                            input_offset_gm,
                            input_offset_ub,
                            input_n_burst,
                            input_burst_len,
                            input_mov_times,
                            input_ele_num_one_row,
                            input_ele_num_one_row_align_32B)
                        _tik_init_ub_tensor(tik_inst,
                            obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.
                                last_repeat_time_front_part_last_core,
                            obj_fp32_output_init_input_scalar.
                                init_times_front_part_last_core)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                                    front_rows_front_part_last_core) as \
                                rows_index:
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_last_core_index *
                                obj_fp32_input_data_input_scalar.
                                    front_rows_front_part_last_core +
                                rows_index])
                            # align part
                            with tik_inst.if_scope(
                                    obj_fp32_e_num_input_scalar.
                                            e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                    obj_fp32_output_init_input_scalar.\
                                    last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                    obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst,
                                    obj_ub_tensor.input_ub,
                                    obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                    input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_floor + \
                                obj_fp32_output_init_input_scalar.\
                                                  last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                obj_fp32_output_init_input_scalar.\
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.\
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)
                    with tik_inst.if_scope(
                            input_mov_tims_front_part_last_core_index ==
                            obj_fp32_input_data_input_scalar.
                                    mov_times_gm2ub_front_part_last_core - 1):
                        # last part ids front part last core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                                ele_num_front_core + \
                            ids_mov_times_last_core_index * \
                            obj_fp32_ids_input_scalar.\
                                ele_num_ub_front_part_last_core * \
                            obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_front_part_last_core_index * \
                                obj_fp32_input_data_input_scalar.\
                                front_ele_num_ub_front_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.\
                                          e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.\
                            last_rows_front_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32B = \
                            obj_fp32_output_init_input_scalar.\
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst,
                            obj_gm_tensor.input_gm,
                            obj_ub_tensor.input_ub,
                            input_offset_gm,
                            input_offset_ub,
                            input_n_burst,
                            input_burst_len,
                            input_mov_times,
                            input_ele_num_one_row,
                            input_ele_num_one_row_align_32B)
                        _tik_init_ub_tensor(tik_inst,
                            obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.
                                last_repeat_time_front_part_last_core,
                            obj_fp32_output_init_input_scalar.
                                init_times_front_part_last_core)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                                    last_rows_front_part_last_core) as \
                                rows_index:
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_last_core_index *
                                obj_fp32_input_data_input_scalar.
                                    front_rows_front_part_last_core +
                                rows_index])
                            # align part
                            with tik_inst.if_scope(
                                    obj_fp32_e_num_input_scalar.
                                            e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                    obj_fp32_output_init_input_scalar.\
                                        last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                    obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst,
                                    obj_ub_tensor.input_ub,
                                    obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.
                                        e_ub2gm_front_burst_len,
                                    input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_floor + \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                obj_fp32_output_init_input_scalar.\
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.\
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index ==
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                    obj_fp32_ids_input_scalar.ele_num_front_core + \
                    ids_mov_times_last_core_index * \
                    obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_last_part_last_core) as \
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.if_scope(
                            input_mov_tims_last_part_last_core_index <
                            obj_fp32_input_data_input_scalar.
                                    mov_times_gm2ub_last_part_last_core - 1):
                        # front part ids last part last core
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                            ele_num_front_core + \
                            ids_mov_times_last_core_index * \
                            obj_fp32_ids_input_scalar.\
                            ele_num_ub_front_part_last_core * \
                            obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_last_part_last_core_index * \
                            obj_fp32_input_data_input_scalar.\
                            front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_front_burst_len + \
                                          obj_fp32_e_num_input_scalar.\
                                              e_ub2gm_last_burst_len
                        input_mov_times = obj_fp32_input_data_input_scalar.\
                            front_rows_last_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32B = \
                            obj_fp32_output_init_input_scalar.\
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst,
                            obj_gm_tensor.input_gm,
                            obj_ub_tensor.input_ub,
                            input_offset_gm,
                            input_offset_ub,
                            input_n_burst,
                            input_burst_len,
                            input_mov_times,
                            input_ele_num_one_row,
                            input_ele_num_one_row_align_32B)
                        _tik_init_ub_tensor(tik_inst,
                            obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.
                                last_repeat_time_last_part_last_core,
                            obj_fp32_output_init_input_scalar.
                                init_times_last_part_last_core)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                                    front_rows_last_part_last_core) as \
                                rows_index:
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_last_core_index *
                                obj_fp32_input_data_input_scalar.
                                    front_rows_last_part_last_core +
                                rows_index])
                            # align part
                            with tik_inst.if_scope(
                                obj_fp32_e_num_input_scalar.\
                                    e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                    obj_fp32_output_init_input_scalar.\
                                        last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                    obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst,
                                    obj_ub_tensor.input_ub,
                                    obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.
                                        e_ub2gm_front_burst_len,
                                    input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_floor + \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                obj_fp32_output_init_input_scalar.\
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.\
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)
                    with tik_inst.if_scope(
                            input_mov_tims_last_part_last_core_index ==
                            obj_fp32_input_data_input_scalar.
                                    mov_times_gm2ub_last_part_last_core - 1):
                        input_offset_gm = block_index * \
                            obj_fp32_input_data_input_scalar.\
                                ele_num_front_core + \
                            ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_last_core * \
                                          obj_fp32_e_num_input_scalar.e_num + \
                            input_mov_tims_last_part_last_core_index * \
                                obj_fp32_input_data_input_scalar.\
                                    front_ele_num_ub_last_part_last_core
                        input_offset_ub = 0
                        input_n_burst = 1
                        input_burst_len = \
                            obj_fp32_e_num_input_scalar.\
                                e_ub2gm_front_burst_len + \
                            obj_fp32_e_num_input_scalar.\
                                e_ub2gm_last_burst_len
                        input_mov_times = \
                            obj_fp32_input_data_input_scalar.\
                                last_rows_last_part_last_core
                        input_ele_num_one_row = \
                            obj_fp32_e_num_input_scalar.e_num
                        input_ele_num_one_row_align_32B = \
                            obj_fp32_output_init_input_scalar.\
                                last_axis_align_floor
                        _tik_mov_input_gm2ub_discrete(tik_inst,
                            obj_gm_tensor.input_gm,
                            obj_ub_tensor.input_ub,
                            input_offset_gm,
                            input_offset_ub,
                            input_n_burst,
                            input_burst_len,
                            input_mov_times,
                            input_ele_num_one_row,
                            input_ele_num_one_row_align_32B)
                        _tik_init_ub_tensor(tik_inst,
                            obj_ub_tensor.output_ub,
                            obj_fp32_output_init_input_scalar.
                                last_repeat_time_last_part_last_core,
                            obj_fp32_output_init_input_scalar.
                                init_times_last_part_last_core)
                        with tik_inst.for_range(0,
                            obj_fp32_input_data_input_scalar.
                            last_rows_last_part_last_core) as rows_index:
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_last_core_index *
                                obj_fp32_input_data_input_scalar.
                                    front_rows_last_part_last_core +
                                rows_index])
                            # align part
                            with tik_inst.if_scope(
                                    obj_fp32_e_num_input_scalar.
                                            e_ub2gm_front_burst_len > 0):
                                input_ub_offset = rows_index * \
                                    obj_fp32_output_init_input_scalar.\
                                        last_axis_align_floor
                                output_gm_offset = id_val_scalar * \
                                        obj_fp32_e_num_input_scalar.e_num
                                _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                    tik_inst,
                                    obj_ub_tensor.input_ub,
                                    obj_gm_tensor.output_gm,
                                    obj_fp32_e_num_input_scalar.
                                        e_ub2gm_front_burst_len,
                                    input_ub_offset,
                                    output_gm_offset)
                            # last part
                            input_ub_offset = rows_index * \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_floor + \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_front_part
                            output_ub_offset = rows_index * \
                                               fp32_ele_num_one_block
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                obj_fp32_output_init_input_scalar.\
                                                   last_axis_align_front_part
                            vadd_mask = obj_fp32_output_init_input_scalar.\
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)


def _tik_atomic_add_last_axis_align_big_e(block_index,
                                          tik_inst,
                                          obj_gm_tensor,
                                          obj_ub_tensor,
                                          obj_common_scalar,
                                          obj_fp32_input_data_input_scalar,
                                          obj_fp32_e_num_input_scalar,
                                          obj_fp32_ids_input_scalar):
    """
    _tik_atomic_add_last_axis_align_big_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(
            block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
            obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index <
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                    mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                        obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(
                                e_mov_index <
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_front_part_front_core_index * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.\
                                e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                obj_gm_tensor.input_gm,
                                obj_ub_tensor.input_ub,
                                input_offset_gm,
                                input_offset_ub,
                                input_n_burst,
                                input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                        with tik_inst.if_scope(
                                e_mov_index == obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_front_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num + \
                                e_mov_index * obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.\
                                e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                   e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_last_burst_len,
                                input_ub_offset,
                                output_gm_offset)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index ==
                    obj_fp32_ids_input_scalar.
                            mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                        obj_fp32_input_data_input_scalar.
                        mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                        obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(
                                e_mov_index <
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_last_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.\
                                    e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                        with tik_inst.if_scope(
                                e_mov_index ==
                                obj_fp32_e_num_input_scalar.e_mov_times_gm2ub -
                                1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_last_part_front_core_index * \
                                    obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * obj_fp32_e_num_input_scalar.\
                                    e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.\
                                e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                obj_gm_tensor.input_gm,
                                obj_ub_tensor.input_ub,
                                input_offset_gm,
                                input_offset_ub,
                                input_n_burst,
                                input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_last_burst_len,
                                input_ub_offset,
                                output_gm_offset)
    with tik_inst.if_scope(
            block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
            obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index <
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # front part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                        obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(
                                e_mov_index <
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_front_part_last_core_index * \
                                    obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.\
                                    e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                        with tik_inst.if_scope(
                                e_mov_index ==
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_front_part_last_core_index * \
                                    obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.\
                                    e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[
                                    input_mov_tims_front_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                   e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_last_burst_len,
                                input_ub_offset,
                                output_gm_offset)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index ==
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_last_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                        obj_fp32_input_data_input_scalar.
                        mov_times_gm2ub_last_part_last_core) as \
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                        obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(
                                e_mov_index <
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_last_part_last_core_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.\
                                e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                    input_mov_tims_last_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                   e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                        with tik_inst.if_scope(
                                e_mov_index ==
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_last_part_last_core_index * \
                                    obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.\
                                e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                obj_gm_tensor.input_gm,
                                obj_ub_tensor.input_ub,
                                input_offset_gm,
                                input_offset_ub,
                                input_n_burst,
                                input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_last_burst_len,
                                input_ub_offset,
                                output_gm_offset)


def _tik_atomic_add_last_axis_not_align_big_e(block_index,
                                            tik_inst,
                                            obj_gm_tensor,
                                            obj_ub_tensor,
                                            obj_common_scalar,
                                            obj_fp32_input_data_input_scalar,
                                            obj_fp32_e_num_input_scalar,
                                            obj_fp32_ids_input_scalar,
                                            obj_fp32_output_init_input_scalar):
    """
    _tik_atomic_add_last_axis_not_align_big_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_fp32_input_data_input_scalar: obj_fp32_input_data_input_scalar
    obj_fp32_e_num_input_scalar: obj_fp32_e_num_input_scalar
    obj_fp32_ids_input_scalar: obj_fp32_ids_input_scalar
    obj_fp32_output_init_input_scalar: obj_fp32_output_init_input_scalar

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(
            block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_fp32_ids_input_scalar.
                                        mov_times_gm2ub_front_core) as \
                ids_mov_times_front_core_index:
            # ids tiling by ub front core
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index <
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids front part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    front_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                    mov_times_gm2ub_front_part_front_core) as \
                        input_mov_tims_front_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                        obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(
                                e_mov_index <
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_front_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num + \
                                e_mov_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.\
                                e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                   e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                        with tik_inst.if_scope(
                                e_mov_index == obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_front_part_front_core_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num + \
                                e_mov_index * obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.\
                                e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_last_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                            _tik_init_ub_tensor(tik_inst,
                                                obj_ub_tensor.output_ub, 1, 1)
                            input_ub_offset = \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_front_part
                            output_ub_offset = 0
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                   e_num_front_part + \
                                input_ub_offset
                            vadd_mask = obj_fp32_output_init_input_scalar.\
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)
            with tik_inst.if_scope(
                    ids_mov_times_front_core_index ==
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_front_core - 1):
                # ids last part front core
                ids_offset_gm = block_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_front_core + \
                                ids_mov_times_front_core_index * \
                                obj_fp32_ids_input_scalar.\
                                    ele_num_ub_front_part_front_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_front_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                    mov_times_gm2ub_last_part_front_core) as \
                        input_mov_tims_last_part_front_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                        obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(
                                e_mov_index <
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_last_part_front_core_index * \
                                    obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.\
                                    e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                   e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                        with tik_inst.if_scope(
                                e_mov_index ==
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_last_part_front_core_index * \
                                    obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.\
                                    e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_front_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_last_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                            _tik_init_ub_tensor(tik_inst,
                                                obj_ub_tensor.output_ub, 1, 1)
                            input_ub_offset = \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_front_part
                            output_ub_offset = 0
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.\
                                                   e_num_front_part + \
                                input_ub_offset
                            vadd_mask = \
                                obj_fp32_output_init_input_scalar.\
                                    last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)
    with tik_inst.if_scope(
            block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
            obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core) as \
                ids_mov_times_last_core_index:
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index <
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # front part last core
                ids_offset_gm = block_index * \
                    obj_fp32_ids_input_scalar.ele_num_front_core + \
                    ids_mov_times_last_core_index * \
                    obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = \
                    obj_fp32_ids_input_scalar.front_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_front_part_last_core) as \
                        input_mov_tims_front_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                        obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(
                                e_mov_index <
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_front_part_last_core_index * \
                                    obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.\
                                e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_front_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                        with tik_inst.if_scope(
                                e_mov_index ==
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_front_part_last_core_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num + \
                                e_mov_index * \
                                              obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_fp32_e_num_input_scalar.\
                                e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                    obj_gm_tensor.input_gm,
                                                    obj_ub_tensor.input_ub,
                                                    input_offset_gm,
                                                    input_offset_ub,
                                                    input_n_burst,
                                                    input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(
                                obj_ub_tensor.ids_ub[
                                    input_mov_tims_front_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_last_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                            _tik_init_ub_tensor(tik_inst,
                                obj_ub_tensor.output_ub, 1, 1)
                            input_ub_offset = \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_front_part
                            output_ub_offset = 0
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part + \
                                input_ub_offset
                            vadd_mask = \
                                obj_fp32_output_init_input_scalar.\
                                    last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)
            with tik_inst.if_scope(
                    ids_mov_times_last_core_index ==
                    obj_fp32_ids_input_scalar.mov_times_gm2ub_last_core - 1):
                # last part last core
                ids_offset_gm = block_index * \
                    obj_fp32_ids_input_scalar.ele_num_front_core + \
                    ids_mov_times_last_core_index * \
                    obj_fp32_ids_input_scalar.ele_num_ub_front_part_last_core
                ids_offset_ub = 0
                ids_n_burst = 1
                ids_burst_len = obj_fp32_ids_input_scalar.\
                    last_burst_len_last_core
                _tik_mov_ids_gm2ub(tik_inst,
                                   obj_gm_tensor.ids_gm,
                                   obj_ub_tensor.ids_ub,
                                   ids_offset_gm,
                                   ids_offset_ub,
                                   ids_n_burst,
                                   ids_burst_len)
                with tik_inst.for_range(0,
                    obj_fp32_input_data_input_scalar.
                            mov_times_gm2ub_last_part_last_core) as \
                        input_mov_tims_last_part_last_core_index:
                    # input data tiling by ids and ub
                    with tik_inst.for_range(0,
                        obj_fp32_e_num_input_scalar.e_mov_times_gm2ub) as \
                            e_mov_index:
                        # e_num tiling
                        with tik_inst.if_scope(
                                e_mov_index <
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num front part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_last_part_last_core_index * \
                                    obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.\
                                    e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                input_mov_tims_last_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                obj_fp32_e_num_input_scalar.e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_front_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                        with tik_inst.if_scope(
                                e_mov_index ==
                                obj_fp32_e_num_input_scalar.
                                        e_mov_times_gm2ub - 1):
                            # e_num last part
                            input_offset_gm = block_index * \
                                obj_fp32_input_data_input_scalar.\
                                    ele_num_front_core + \
                                input_mov_tims_last_part_last_core_index * \
                                    obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                  e_num_front_part
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = \
                                obj_fp32_e_num_input_scalar.\
                                    e_ub2gm_last_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            input_ub_offset = 0
                            id_val_scalar.set_as(obj_ub_tensor.ids_ub[
                                    input_mov_tims_last_part_last_core_index])
                            output_gm_offset = id_val_scalar * \
                                        obj_fp32_e_num_input_scalar.e_num + \
                                        e_mov_index * \
                                        obj_fp32_e_num_input_scalar.\
                                            e_num_front_part
                            _tik_atomic_add_ub2gm_by_id_last_axis_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_gm_tensor.output_gm,
                                obj_fp32_e_num_input_scalar.
                                    e_ub2gm_last_burst_len,
                                input_ub_offset,
                                output_gm_offset)
                            _tik_init_ub_tensor(tik_inst,
                                                obj_ub_tensor.output_ub, 1, 1)
                            input_ub_offset = \
                                obj_fp32_output_init_input_scalar.\
                                    last_axis_align_front_part
                            output_ub_offset = 0
                            output_gm_offset = id_val_scalar * \
                                obj_fp32_e_num_input_scalar.e_num + \
                                e_mov_index * \
                                    obj_fp32_e_num_input_scalar.\
                                                   e_num_front_part + \
                                input_ub_offset
                            vadd_mask = obj_fp32_output_init_input_scalar.\
                                last_part_vadd_mask
                            _tik_atomic_add_ub2gm_by_id_last_axis_not_align(
                                tik_inst,
                                obj_ub_tensor.input_ub,
                                obj_ub_tensor.output_ub,
                                obj_gm_tensor.output_gm,
                                input_ub_offset,
                                output_ub_offset,
                                output_gm_offset,
                                vadd_mask)


def _tik_int32_add_small_e(block_index,
                           tik_inst,
                           obj_gm_tensor,
                           obj_ub_tensor,
                           obj_common_scalar,
                           obj_int32_ids_input_scalar,
                           obj_int32_e_num_input_scalar):
    """
    _tik_int32_add_small_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_int32_ids_input_scalar: obj_int32_ids_input_scalar
    obj_int32_e_num_input_scalar: obj_int32_e_num_input_scalar

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(
            block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
            obj_common_scalar.num_segments_front_core) as \
                segment_index_front_core:
            # num_segments divide by core
            # init output_ub
            _tik_init_ub_tensor_once(tik_inst,
                                     obj_ub_tensor.output_ub,
                                     obj_int32_e_num_input_scalar.
                                     repeat_time_front_part,
                                     MASK_INT32)
            with tik_inst.for_range(0,
                obj_int32_ids_input_scalar.mov_times_gm2ub) as \
                    ids_mov_index_gm2ub:
                # ids divide by ub
                with tik_inst.if_scope(
                        ids_mov_index_gm2ub <
                        obj_int32_ids_input_scalar.
                                mov_times_gm2ub - 1):
                    # ids front part
                    ids_offset_gm = ids_mov_index_gm2ub * \
                                    obj_int32_ids_input_scalar.\
                                        ele_num_ub_front_part
                    ids_offset_ub = 0
                    ids_n_burst = 1
                    ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                    _tik_mov_ids_gm2ub(tik_inst,
                                       obj_gm_tensor.ids_gm,
                                       obj_ub_tensor.ids_ub,
                                       ids_offset_gm,
                                       ids_offset_ub,
                                       ids_n_burst,
                                       ids_burst_len)
                    with tik_inst.for_range(0,
                        obj_int32_ids_input_scalar.ele_num_ub_front_part) as \
                            ids_index_front_part:
                        # visit ids
                        id_val_scalar.set_as(
                            obj_ub_tensor.ids_ub[ids_index_front_part])
                        with tik_inst.if_scope(
                                segment_index_front_core + block_index *
                                obj_common_scalar.
                                    num_segments_front_core ==
                                id_val_scalar):
                            # id in segment
                            input_offset_gm = (
                                                      ids_offset_gm +
                                                      ids_index_front_part) * \
                                              obj_int32_e_num_input_scalar.\
                                                  e_num
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.\
                                e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            _tik_vadd(tik_inst,
                                      obj_ub_tensor.input_ub,
                                      obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.
                                          repeat_time_front_part,
                                      MASK_INT32)
                with tik_inst.if_scope(
                        ids_mov_index_gm2ub == obj_int32_ids_input_scalar.
                                mov_times_gm2ub - 1):
                    # ids last part
                    ids_offset_gm = ids_mov_index_gm2ub * \
                                    obj_int32_ids_input_scalar.\
                                        ele_num_ub_front_part
                    ids_offset_ub = 0
                    ids_n_burst = 1
                    ids_burst_len = obj_int32_ids_input_scalar.\
                        last_burst_len
                    _tik_mov_ids_gm2ub(tik_inst,
                                       obj_gm_tensor.ids_gm,
                                       obj_ub_tensor.ids_ub,
                                       ids_offset_gm,
                                       ids_offset_ub,
                                       ids_n_burst,
                                       ids_burst_len)
                    with tik_inst.for_range(0,
                                            obj_int32_ids_input_scalar.
                                                    ele_num_ub_last_part) as \
                            ids_index_last_part:
                        # visit ids
                        id_val_scalar.set_as(
                            obj_ub_tensor.ids_ub[ids_index_last_part])
                        with tik_inst.if_scope(
                                segment_index_front_core + block_index *
                                obj_common_scalar.
                                    num_segments_front_core ==
                                id_val_scalar):
                            # id in segment
                            input_offset_gm = (
                                                      ids_offset_gm +
                                                      ids_index_last_part) * \
                                              obj_int32_e_num_input_scalar.\
                                                  e_num
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.\
                                e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            _tik_vadd(tik_inst,
                                      obj_ub_tensor.input_ub,
                                      obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.
                                      repeat_time_front_part,
                                      MASK_INT32)
            output_offset_gm = (
                                       block_index *
                                       obj_common_scalar.
                                       num_segments_front_core +
                                       segment_index_front_core) * \
                               obj_int32_e_num_input_scalar.e_num
            output_offset_ub = 0
            output_n_burst = 1
            output_burst_len = obj_int32_e_num_input_scalar.\
                e_ub2gm_front_burst_len
            _tik_mov_output_ub2gm_continue(tik_inst,
                                           obj_gm_tensor.output_gm,
                                           obj_ub_tensor.output_ub,
                                           output_offset_gm,
                                           output_offset_ub,
                                           output_n_burst,
                                           output_burst_len)
    with tik_inst.if_scope(
            block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0,
                                obj_common_scalar.
                                num_segments_last_core) as \
                segment_index_last_core:
            # num_segments divide by core
            # init output_ub
            _tik_init_ub_tensor_once(tik_inst,
                                     obj_ub_tensor.output_ub,
                                     obj_int32_e_num_input_scalar.
                                     repeat_time_front_part,
                                     MASK_INT32)
            with tik_inst.for_range(0,
                                    obj_int32_ids_input_scalar.
                                            mov_times_gm2ub) as \
                    ids_mov_index_gm2ub:
                # ids divide by ub
                with tik_inst.if_scope(
                        ids_mov_index_gm2ub <
                        obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                    # ids front part
                    ids_offset_gm = ids_mov_index_gm2ub * \
                                    obj_int32_ids_input_scalar.\
                                        ele_num_ub_front_part
                    ids_offset_ub = 0
                    ids_n_burst = 1
                    ids_burst_len = obj_int32_ids_input_scalar.front_burst_len
                    _tik_mov_ids_gm2ub(tik_inst,
                                       obj_gm_tensor.ids_gm,
                                       obj_ub_tensor.ids_ub,
                                       ids_offset_gm,
                                       ids_offset_ub,
                                       ids_n_burst,
                                       ids_burst_len)
                    with tik_inst.for_range(0,
                                            obj_int32_ids_input_scalar.
                                                    ele_num_ub_front_part) as \
                            ids_index_front_part:
                        # visit ids
                        id_val_scalar.set_as(
                            obj_ub_tensor.ids_ub[ids_index_front_part])
                        with tik_inst.if_scope(
                                segment_index_last_core + block_index *
                                obj_common_scalar.
                                    num_segments_front_core ==
                                id_val_scalar):
                            # id in segment
                            input_offset_gm = (
                                                      ids_offset_gm +
                                                      ids_index_front_part) * \
                                              obj_int32_e_num_input_scalar.\
                                                  e_num
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.\
                                e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            _tik_vadd(tik_inst,
                                      obj_ub_tensor.input_ub,
                                      obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.
                                      repeat_time_front_part,
                                      MASK_INT32)
                with tik_inst.if_scope(
                        ids_mov_index_gm2ub ==
                        obj_int32_ids_input_scalar.mov_times_gm2ub - 1):
                    # ids last part
                    ids_offset_gm = ids_mov_index_gm2ub * \
                                    obj_int32_ids_input_scalar.\
                                        ele_num_ub_front_part
                    ids_offset_ub = 0
                    ids_n_burst = 1
                    ids_burst_len = obj_int32_ids_input_scalar.last_burst_len
                    _tik_mov_ids_gm2ub(tik_inst,
                                       obj_gm_tensor.ids_gm,
                                       obj_ub_tensor.ids_ub,
                                       ids_offset_gm,
                                       ids_offset_ub,
                                       ids_n_burst,
                                       ids_burst_len)
                    with tik_inst.for_range(0,
                                            obj_int32_ids_input_scalar.
                                                    ele_num_ub_last_part) as \
                            ids_index_last_part:
                        # visit ids
                        id_val_scalar.set_as(
                            obj_ub_tensor.ids_ub[ids_index_last_part])
                        with tik_inst.if_scope(
                                segment_index_last_core + block_index *
                                obj_common_scalar.
                                    num_segments_front_core ==
                                id_val_scalar):
                            # id in segment
                            input_offset_gm = (
                                                      ids_offset_gm +
                                                      ids_index_last_part) * \
                                            obj_int32_e_num_input_scalar.e_num
                            input_offset_ub = 0
                            input_n_burst = 1
                            input_burst_len = obj_int32_e_num_input_scalar.\
                                e_ub2gm_front_burst_len
                            _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                            _tik_vadd(tik_inst,
                                      obj_ub_tensor.input_ub,
                                      obj_ub_tensor.output_ub,
                                      obj_int32_e_num_input_scalar.
                                      repeat_time_front_part,
                                      MASK_INT32)
            output_offset_gm = (
                                       block_index *
                                       obj_common_scalar.
                                       num_segments_front_core +
                                       segment_index_last_core) * \
                               obj_int32_e_num_input_scalar.e_num
            output_offset_ub = 0
            output_n_burst = 1
            output_burst_len = obj_int32_e_num_input_scalar.\
                e_ub2gm_front_burst_len
            _tik_mov_output_ub2gm_continue(tik_inst,
                                           obj_gm_tensor.output_gm,
                                           obj_ub_tensor.output_ub,
                                           output_offset_gm,
                                           output_offset_ub,
                                           output_n_burst,
                                           output_burst_len)


def _tik_int32_add_big_e(block_index,
                         tik_inst,
                         obj_gm_tensor,
                         obj_ub_tensor,
                         obj_common_scalar,
                         obj_int32_ids_input_scalar,
                         obj_int32_e_num_input_scalar):
    """
    _tik_int32_add_big_e

    Parameters
    ----------
    block_index: block_index
    tik_inst: tik_instance
    obj_gm_tensor: obj_gm_tensor
    obj_ub_tensor: obj_ub_tensor
    obj_common_scalar: obj_common_scalar
    obj_int32_ids_input_scalar: obj_int32_ids_input_scalar
    obj_int32_e_num_input_scalar: obj_int32_e_num_input_scalar

    Returns
    -------
    None
    """
    id_val_scalar = obj_common_scalar.id_val_scalar
    with tik_inst.if_scope(
            block_index < obj_common_scalar.need_core_num - 1):
        # front core
        with tik_inst.for_range(0,
                                obj_int32_e_num_input_scalar.
                                        e_mov_times_gm2ub) as \
                e_mov_index_gm2ub:
            # e divide by ub
            with tik_inst.if_scope(
                    e_mov_index_gm2ub < obj_int32_e_num_input_scalar.
                            e_mov_times_gm2ub - 1):
                # front part e
                with tik_inst.for_range(0,
                                        obj_common_scalar.
                                    num_segments_front_core) as \
                        segment_index_front_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst,
                                             obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.
                                             repeat_time_front_part,
                                             MASK_INT32)
                    with tik_inst.for_range(0,
                                            obj_int32_ids_input_scalar.
                                                    mov_times_gm2ub) as \
                            ids_mov_index_gm2ub:
                        # ids divide by ub
                        with tik_inst.if_scope(
                                ids_mov_index_gm2ub <
                                obj_int32_ids_input_scalar.
                                        mov_times_gm2ub - 1):
                            # ids front part
                            ids_offset_gm = ids_mov_index_gm2ub * \
                                            obj_int32_ids_input_scalar.\
                                                ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.\
                                front_burst_len
                            _tik_mov_ids_gm2ub(tik_inst,
                                               obj_gm_tensor.ids_gm,
                                               obj_ub_tensor.ids_ub,
                                               ids_offset_gm,
                                               ids_offset_ub,
                                               ids_n_burst,
                                               ids_burst_len)
                            with tik_inst.for_range(0,
                                obj_int32_ids_input_scalar.
                                        ele_num_ub_front_part) as \
                                    ids_index_front_part:
                                # visit ids
                                id_val_scalar.set_as(
                                    obj_ub_tensor.ids_ub[ids_index_front_part])
                                with tik_inst.if_scope(
                                        segment_index_front_core +
                                        block_index *
                                        obj_common_scalar.
                                        num_segments_front_core ==
                                        id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (
                                        ids_offset_gm +
                                        ids_index_front_part) * \
                                        obj_int32_e_num_input_scalar.e_num + \
                                        e_mov_index_gm2ub * \
                                        obj_int32_e_num_input_scalar.\
                                                          e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = \
                                        obj_int32_e_num_input_scalar.\
                                            e_ub2gm_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                                    _tik_vadd(tik_inst,
                                              obj_ub_tensor.input_ub,
                                              obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.
                                              repeat_time_front_part,
                                              MASK_INT32)
                        with tik_inst.if_scope(
                                ids_mov_index_gm2ub ==
                                obj_int32_ids_input_scalar.
                                        mov_times_gm2ub - 1):
                            # ids last part
                            ids_offset_gm = ids_mov_index_gm2ub * \
                                            obj_int32_ids_input_scalar.\
                                                ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.\
                                last_burst_len
                            _tik_mov_ids_gm2ub(tik_inst,
                                               obj_gm_tensor.ids_gm,
                                               obj_ub_tensor.ids_ub,
                                               ids_offset_gm,
                                               ids_offset_ub,
                                               ids_n_burst,
                                               ids_burst_len)
                            with tik_inst.for_range(0,
                                obj_int32_ids_input_scalar.
                                        ele_num_ub_last_part) as \
                                    ids_index_last_part:
                                # visit ids
                                id_val_scalar.set_as(
                                    obj_ub_tensor.ids_ub[ids_index_last_part])
                                with tik_inst.if_scope(
                                        segment_index_front_core +
                                        block_index * obj_common_scalar.
                                        num_segments_front_core ==
                                        id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (
                                        ids_offset_gm +
                                        ids_index_last_part) * \
                                        obj_int32_e_num_input_scalar.e_num + \
                                        e_mov_index_gm2ub * \
                                        obj_int32_e_num_input_scalar.\
                                                          e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = \
                                        obj_int32_e_num_input_scalar.\
                                            e_ub2gm_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst,
                                                    obj_gm_tensor.input_gm,
                                                    obj_ub_tensor.input_ub,
                                                    input_offset_gm,
                                                    input_offset_ub,
                                                    input_n_burst,
                                                    input_burst_len)
                                    _tik_vadd(tik_inst,
                                              obj_ub_tensor.input_ub,
                                              obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.
                                              repeat_time_front_part,
                                              MASK_INT32)
                    # mov output data ub2gm
                    output_offset_gm = (
                        block_index *
                        obj_common_scalar.
                        num_segments_front_core +
                        segment_index_front_core) * \
                        obj_int32_e_num_input_scalar.e_num + \
                        e_mov_index_gm2ub * \
                        obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.\
                        e_ub2gm_front_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst,
                                                   obj_gm_tensor.output_gm,
                                                   obj_ub_tensor.output_ub,
                                                   output_offset_gm,
                                                   output_offset_ub,
                                                   output_n_burst,
                                                   output_burst_len)
            with tik_inst.if_scope(
                    e_mov_index_gm2ub ==
                    obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # last part e
                with tik_inst.for_range(0,
                    obj_common_scalar.
                        num_segments_front_core) as \
                        segment_index_front_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst,
                                             obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.
                                             repeat_time_last_part,
                                             MASK_INT32)
                    with tik_inst.for_range(0,
                                            obj_int32_ids_input_scalar.
                                                    mov_times_gm2ub) as \
                            ids_mov_index_gm2ub:
                        # ids divide by ub
                        with tik_inst.if_scope(
                                ids_mov_index_gm2ub <
                                obj_int32_ids_input_scalar.
                                        mov_times_gm2ub - 1):
                            # ids front part
                            ids_offset_gm = ids_mov_index_gm2ub * \
                                            obj_int32_ids_input_scalar.\
                                                ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.\
                                front_burst_len
                            _tik_mov_ids_gm2ub(tik_inst,
                                               obj_gm_tensor.ids_gm,
                                               obj_ub_tensor.ids_ub,
                                               ids_offset_gm,
                                               ids_offset_ub,
                                               ids_n_burst,
                                               ids_burst_len)
                            with tik_inst.for_range(0,
                                obj_int32_ids_input_scalar.
                                        ele_num_ub_front_part) as \
                                    ids_index_front_part:
                                # visit ids
                                id_val_scalar.set_as(
                                    obj_ub_tensor.ids_ub[ids_index_front_part])
                                with tik_inst.if_scope(
                                        segment_index_front_core +
                                        block_index *
                                        obj_common_scalar.
                                        num_segments_front_core ==
                                        id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm +
                                        ids_index_front_part) * \
                                        obj_int32_e_num_input_scalar.\
                                            e_num + \
                                        e_mov_index_gm2ub * \
                                        obj_int32_e_num_input_scalar.\
                                            e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = \
                                        obj_int32_e_num_input_scalar.\
                                            e_ub2gm_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst,
                                        obj_gm_tensor.input_gm,
                                        obj_ub_tensor.input_ub,
                                        input_offset_gm,
                                        input_offset_ub,
                                        input_n_burst,
                                        input_burst_len)
                                    _tik_vadd(tik_inst,
                                              obj_ub_tensor.input_ub,
                                              obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.
                                              repeat_time_front_part,
                                              MASK_INT32)
                        with tik_inst.if_scope(
                                ids_mov_index_gm2ub ==
                                obj_int32_ids_input_scalar.
                                        mov_times_gm2ub - 1):
                            # ids last part
                            ids_offset_gm = ids_mov_index_gm2ub * \
                                            obj_int32_ids_input_scalar.\
                                                ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.\
                                last_burst_len
                            _tik_mov_ids_gm2ub(tik_inst,
                                               obj_gm_tensor.ids_gm,
                                               obj_ub_tensor.ids_ub,
                                               ids_offset_gm,
                                               ids_offset_ub,
                                               ids_n_burst,
                                               ids_burst_len)
                            with tik_inst.for_range(0,
                                obj_int32_ids_input_scalar.
                                        ele_num_ub_last_part) as \
                                    ids_index_last_part:
                                # visit ids
                                id_val_scalar.set_as(
                                    obj_ub_tensor.ids_ub[ids_index_last_part])
                                with tik_inst.if_scope(
                                    segment_index_front_core +
                                    block_index *
                                    obj_common_scalar.
                                    num_segments_front_core ==
                                    id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm +
                                        ids_index_last_part) * \
                                        obj_int32_e_num_input_scalar.e_num + \
                                        e_mov_index_gm2ub * \
                                        obj_int32_e_num_input_scalar.\
                                                          e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = \
                                    obj_int32_e_num_input_scalar.\
                                        e_ub2gm_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst,
                                        obj_gm_tensor.input_gm,
                                        obj_ub_tensor.input_ub,
                                        input_offset_gm,
                                        input_offset_ub,
                                        input_n_burst,
                                        input_burst_len)
                                    _tik_vadd(tik_inst,
                                              obj_ub_tensor.input_ub,
                                              obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.
                                              repeat_time_front_part,
                                              MASK_INT32)
                    # mov output data ub2gm
                    output_offset_gm = (block_index *
                                        obj_common_scalar.
                                        num_segments_front_core +
                                        segment_index_front_core) * \
                                       obj_int32_e_num_input_scalar.e_num + \
                                       e_mov_index_gm2ub * \
                                       obj_int32_e_num_input_scalar.\
                                           e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.\
                        e_ub2gm_front_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst,
                                                   obj_gm_tensor.output_gm,
                                                   obj_ub_tensor.output_ub,
                                                   output_offset_gm,
                                                   output_offset_ub,
                                                   output_n_burst,
                                                   output_burst_len)
    with tik_inst.if_scope(
            block_index == obj_common_scalar.need_core_num - 1):
        # last core
        with tik_inst.for_range(0, obj_int32_e_num_input_scalar.
                e_mov_times_gm2ub) as e_mov_index_gm2ub:
            # e divide by ub
            with tik_inst.if_scope(
                    e_mov_index_gm2ub <
                    obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # front part e
                with tik_inst.for_range(0, obj_common_scalar.
                        num_segments_last_core) as \
                        segment_index_last_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst,
                                             obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.
                                             repeat_time_front_part,
                                             MASK_INT32)
                    with tik_inst.for_range(0, obj_int32_ids_input_scalar.
                            mov_times_gm2ub) as ids_mov_index_gm2ub:
                        # ids divide by ub
                        with tik_inst.if_scope(
                                ids_mov_index_gm2ub <
                                obj_int32_ids_input_scalar.
                                        mov_times_gm2ub - 1):
                            # ids front part
                            ids_offset_gm = ids_mov_index_gm2ub * \
                                            obj_int32_ids_input_scalar.\
                                                ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.\
                                front_burst_len
                            _tik_mov_ids_gm2ub(tik_inst,
                                               obj_gm_tensor.ids_gm,
                                               obj_ub_tensor.ids_ub,
                                               ids_offset_gm,
                                               ids_offset_ub,
                                               ids_n_burst,
                                               ids_burst_len)
                            with tik_inst.for_range(0,
                                    obj_int32_ids_input_scalar.
                                            ele_num_ub_front_part) as \
                                    ids_index_front_part:
                                # visit ids
                                id_val_scalar.set_as(
                                    obj_ub_tensor.ids_ub[ids_index_front_part])
                                with tik_inst.if_scope(
                                        segment_index_last_core +
                                        block_index *
                                        obj_common_scalar.
                                        num_segments_front_core ==
                                        id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm +
                                        ids_index_front_part) * \
                                        obj_int32_e_num_input_scalar.e_num + \
                                            e_mov_index_gm2ub * \
                                        obj_int32_e_num_input_scalar.\
                                                          e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = \
                                        obj_int32_e_num_input_scalar.\
                                            e_ub2gm_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst,
                                                    obj_gm_tensor.input_gm,
                                                    obj_ub_tensor.input_ub,
                                                    input_offset_gm,
                                                    input_offset_ub,
                                                    input_n_burst,
                                                    input_burst_len)
                                    _tik_vadd(tik_inst,
                                              obj_ub_tensor.input_ub,
                                              obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.
                                              repeat_time_front_part,
                                              MASK_INT32)
                        with tik_inst.if_scope(
                                ids_mov_index_gm2ub ==
                                obj_int32_ids_input_scalar.
                                        mov_times_gm2ub - 1):
                            # ids last part
                            ids_offset_gm = ids_mov_index_gm2ub * \
                                            obj_int32_ids_input_scalar.\
                                                ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.\
                                last_burst_len
                            _tik_mov_ids_gm2ub(tik_inst,
                                               obj_gm_tensor.ids_gm,
                                               obj_ub_tensor.ids_ub,
                                               ids_offset_gm,
                                               ids_offset_ub,
                                               ids_n_burst,
                                               ids_burst_len)
                            with tik_inst.for_range(0,
                                obj_int32_ids_input_scalar.
                                        ele_num_ub_last_part) as \
                                    ids_index_last_part:
                                # visit ids
                                id_val_scalar.set_as(
                                    obj_ub_tensor.ids_ub[ids_index_last_part])
                                with tik_inst.if_scope(
                                        segment_index_last_core +
                                        block_index *
                                        obj_common_scalar.
                                        num_segments_front_core ==
                                        id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm +
                                                       ids_index_last_part) * \
                                        obj_int32_e_num_input_scalar.e_num + \
                                        e_mov_index_gm2ub * \
                                        obj_int32_e_num_input_scalar.\
                                                          e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = \
                                        obj_int32_e_num_input_scalar.\
                                            e_ub2gm_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                                    _tik_vadd(tik_inst,
                                              obj_ub_tensor.input_ub,
                                              obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.
                                              repeat_time_front_part,
                                              MASK_INT32)
                    # mov output data ub2gm
                    output_offset_gm = (block_index *
                        obj_common_scalar.
                            num_segments_front_core +
                        segment_index_last_core) * \
                        obj_int32_e_num_input_scalar.e_num + \
                        e_mov_index_gm2ub * \
                        obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.\
                        e_ub2gm_front_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst,
                                                   obj_gm_tensor.output_gm,
                                                   obj_ub_tensor.output_ub,
                                                   output_offset_gm,
                                                   output_offset_ub,
                                                   output_n_burst,
                                                   output_burst_len)
            with tik_inst.if_scope(
                    e_mov_index_gm2ub ==
                    obj_int32_e_num_input_scalar.e_mov_times_gm2ub - 1):
                # last part e
                with tik_inst.for_range(0, obj_common_scalar.
                        num_segments_last_core) as \
                        segment_index_last_core:
                    # num_segments divide by core
                    # init output_ub
                    _tik_init_ub_tensor_once(tik_inst,
                                             obj_ub_tensor.output_ub,
                                             obj_int32_e_num_input_scalar.
                                             repeat_time_last_part,
                                             MASK_INT32)
                    with tik_inst.for_range(0,
                                            obj_int32_ids_input_scalar.
                                                    mov_times_gm2ub) as \
                            ids_mov_index_gm2ub:
                        # ids divide by ub
                        with tik_inst.if_scope(
                                ids_mov_index_gm2ub <
                                obj_int32_ids_input_scalar.
                                        mov_times_gm2ub - 1):
                            # ids front part
                            ids_offset_gm = ids_mov_index_gm2ub * \
                                            obj_int32_ids_input_scalar.\
                                                ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.\
                                front_burst_len
                            _tik_mov_ids_gm2ub(tik_inst,
                                               obj_gm_tensor.ids_gm,
                                               obj_ub_tensor.ids_ub,
                                               ids_offset_gm,
                                               ids_offset_ub,
                                               ids_n_burst,
                                               ids_burst_len)
                            with tik_inst.for_range(0,
                                obj_int32_ids_input_scalar.
                                        ele_num_ub_front_part) as \
                                    ids_index_front_part:
                                # visit ids
                                id_val_scalar.set_as(
                                    obj_ub_tensor.ids_ub[ids_index_front_part])
                                with tik_inst.if_scope(
                                        segment_index_last_core +
                                        block_index *
                                        obj_common_scalar.
                                        num_segments_front_core ==
                                        id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm +
                                                       ids_index_front_part) * \
                                        obj_int32_e_num_input_scalar.e_num + \
                                                      e_mov_index_gm2ub * \
                                        obj_int32_e_num_input_scalar.\
                                                          e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = \
                                        obj_int32_e_num_input_scalar.\
                                            e_ub2gm_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst,
                                                    obj_gm_tensor.input_gm,
                                                    obj_ub_tensor.input_ub,
                                                    input_offset_gm,
                                                    input_offset_ub,
                                                    input_n_burst,
                                                    input_burst_len)
                                    _tik_vadd(tik_inst,
                                              obj_ub_tensor.input_ub,
                                              obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.
                                              repeat_time_front_part,
                                              MASK_INT32)
                        with tik_inst.if_scope(
                                ids_mov_index_gm2ub ==
                                obj_int32_ids_input_scalar.
                                        mov_times_gm2ub - 1):
                            # ids last part
                            ids_offset_gm = ids_mov_index_gm2ub * \
                                            obj_int32_ids_input_scalar.\
                                                ele_num_ub_front_part
                            ids_offset_ub = 0
                            ids_n_burst = 1
                            ids_burst_len = obj_int32_ids_input_scalar.\
                                last_burst_len
                            _tik_mov_ids_gm2ub(tik_inst,
                                               obj_gm_tensor.ids_gm,
                                               obj_ub_tensor.ids_ub,
                                               ids_offset_gm,
                                               ids_offset_ub,
                                               ids_n_burst,
                                               ids_burst_len)
                            with tik_inst.for_range(0,
                                obj_int32_ids_input_scalar.
                                        ele_num_ub_last_part) as \
                                    ids_index_last_part:
                                # visit ids
                                id_val_scalar.set_as(
                                    obj_ub_tensor.ids_ub[ids_index_last_part])
                                with tik_inst.if_scope(
                                        segment_index_last_core +
                                        block_index *
                                        obj_common_scalar.
                                        num_segments_front_core ==
                                        id_val_scalar):
                                    # id in segment
                                    input_offset_gm = (ids_offset_gm +
                                        ids_index_last_part) * \
                                        obj_int32_e_num_input_scalar.e_num + \
                                        e_mov_index_gm2ub * \
                                        obj_int32_e_num_input_scalar.\
                                            e_num_front_part
                                    input_offset_ub = 0
                                    input_n_burst = 1
                                    input_burst_len = \
                                        obj_int32_e_num_input_scalar.\
                                            e_ub2gm_front_burst_len
                                    _tik_mov_input_gm2ub_continue(tik_inst,
                                                        obj_gm_tensor.input_gm,
                                                        obj_ub_tensor.input_ub,
                                                        input_offset_gm,
                                                        input_offset_ub,
                                                        input_n_burst,
                                                        input_burst_len)
                                    _tik_vadd(tik_inst,
                                              obj_ub_tensor.input_ub,
                                              obj_ub_tensor.output_ub,
                                              obj_int32_e_num_input_scalar.
                                              repeat_time_front_part,
                                              MASK_INT32)
                    # mov output data ub2gm
                    output_offset_gm = (block_index *
                        obj_common_scalar.
                            num_segments_front_core +
                            segment_index_last_core) * \
                        obj_int32_e_num_input_scalar.e_num + \
                            e_mov_index_gm2ub * \
                        obj_int32_e_num_input_scalar.e_num_front_part
                    output_offset_ub = 0
                    output_n_burst = 1
                    output_burst_len = obj_int32_e_num_input_scalar.\
                        e_ub2gm_front_burst_len
                    _tik_mov_output_ub2gm_continue(tik_inst,
                                                   obj_gm_tensor.output_gm,
                                                   obj_ub_tensor.output_ub,
                                                   output_offset_gm,
                                                   output_offset_ub,
                                                   output_n_burst,
                                                   output_burst_len)


@te.op.register_operator("UnsortedSegmentSum")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def unsorted_segment_sum(x_dict, segment_ids_dict, num_segments_dict, y_dict,
                         kernel_name="UnsortedSegmentSum"):
    """
    unsorted_segment_sum entry interface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    segment_ids_dict: segment_ids shape, dtype and range
    num_segments_dict: num_segments shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of UnsortedSegmentSum op

    Returns
    -------
    compile info
    """
    x_dtype = x_dict.get("dtype").lower()
    x_dtype_check_list = ("float32")
    check_dtype(x_dtype, x_dtype_check_list, param_name="x_dict")

    segment_ids_dtype = segment_ids_dict.get("dtype").lower()
    segment_ids_dtype_check_list = ("int32")
    check_dtype(segment_ids_dtype, segment_ids_dtype_check_list,
                param_name="segment_ids_dict")

    num_segments_dtype = num_segments_dict.get("dtype").lower()
    num_segments_dtype_check_list = ("int32")
    check_dtype(num_segments_dtype, num_segments_dtype_check_list, param_name="num_segments_dict")

    y_dtype = y_dict.get("dtype").lower()
    check_dtype(y_dtype, x_dtype_check_list, param_name="y_dict")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name,
                                                              "x",
                                                              "y",
                                                              x_dtype,
                                                              y_dtype)
    obj = UnsortedSegmentSum(x_dict, segment_ids_dict, num_segments_dict,
                             y_dict, kernel_name)
    obj.unsorted_segment_sum()
    # add compile info
    te.op.add_compile_info("vars",
                           {"ub_size": obj.ub_size, "core_num": obj.core_num,
                            "dtype": obj.obj_gm_tensor.input_gm.dtype,
                            "ub_tensor_num": obj.ub_tensor_num})
