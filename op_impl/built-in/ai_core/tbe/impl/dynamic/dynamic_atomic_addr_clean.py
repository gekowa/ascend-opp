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
atomic_addr_clean
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
from te.utils.op_utils import REQUIRED_ATTR_LIST_INT
from te.utils.op_utils import KERNEL_NAME

# max_int32
MAX_INT32 = 2 ** 31 - 1

# full mask for fp32
MASK_FP32 = 64

# max repeat time of vector instruction
MAX_REPEAT_TIME = 255

# max tiling params num
MAX_TILING_PARAMS_NUM = 64

# int32 byte
INT32_BYTE = 4

# block byte
BLOCK_BYTE = 32

ZERO_FP32 = 0.0


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


class DynamicAtomicAddrClean(object):
    def __init__(self):
        """
        constructor of class DynamicAtomicAddrClean

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.core_num = platform.cce_conf.get_soc_spec(
            platform.cce_conf.CORE_NUM)
        self.is_double_buffer = True
        self.ub_size = _tik_get_ub_size(self.is_double_buffer)
        self.gm_tensor = self.tik_instance.Tensor("float32", (MAX_INT32,),
                                                  tik.scope_gm, "gm_tensor")
        self.tiling_gm = self.tik_instance.Tensor("int32",
                                                  (MAX_TILING_PARAMS_NUM,),
                                                  tik.scope_gm, "tiling_gm")

        class CommonInputScalar():
            def __init__(self, tik_instance):
                """
                constructor of class CommonInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.select_key = tik_instance.Scalar(
                    dtype="int32", name="select_key")
                self.need_core_num = tik_instance.Scalar(
                    dtype="int32", name="need_core_num")
                self.ele_num_full_mask_repeat_time = \
                    tik_instance.Scalar(
                    dtype="int32",
                    name="ele_num_full_mask_repeat_time")
                self.burst_len_full_mask_repeat_time = \
                    tik_instance.Scalar(
                    dtype="int32",
                    name="burst_len_full_mask_repeat_time")

        class InitInputScalar():
            def __init__(self, tik_instance):
                """
                constructor of class InitInputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                # front core
                self.ele_num_front_core = tik_instance.Scalar(
                    dtype="int32", name="ele_num_front_core")
                # front part full mask full repeat time front core
                self.init_times_full_mask_repeat_time_front_core = \
                    tik_instance.Scalar(
                    dtype="int32",
                    name="init_times_full_mask_repeat_time_front_core")
                self.ele_num_front_part_front_core = tik_instance.Scalar(
                    dtype="int32",
                    name="ele_num_front_part_front_core")
                # last part front
                self.burst_len_last_part_front_core = tik_instance.Scalar(
                    dtype="int32",
                    name="burst_len_last_part_front_core")
                self.repeat_time_last_part_front_core = tik_instance.Scalar(
                    dtype="int32",
                    name="repeat_time_last_part_front_core")

                # last core
                self.ele_num_last_core = tik_instance.Scalar(
                    dtype="int32", name="ele_num_last_core")
                # front part full mask full repeat time last core
                self.init_times_full_mask_repeat_time_last_core = \
                    tik_instance.Scalar(
                    dtype="int32",
                    name="init_times_full_mask_repeat_time_last_core")
                self.ele_num_front_part_last_core = tik_instance.Scalar(
                    dtype="int32",
                    name="ele_num_front_part_last_core")
                # last part last core
                self.burst_len_last_part_last_core = tik_instance.Scalar(
                    dtype="int32",
                    name="burst_len_last_part_last_core")
                self.repeat_time_last_part_last_core = tik_instance.Scalar(
                    dtype="int32",
                    name="repeat_time_last_part_last_core")

        self.obj_common_input_scalar = CommonInputScalar(self.tik_instance)
        self.obj_init_input_scalar = InitInputScalar(self.tik_instance)

        with self.tik_instance.new_stmt_scope():
            # mov tiling data from gm to ub, and set_as scalar
            tiling_ub = self.tik_instance.Tensor("int32",
                                                 (MAX_TILING_PARAMS_NUM,),
                                                 tik.scope_ubuf, "tiling_ub")
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1,
                MAX_TILING_PARAMS_NUM * INT32_BYTE // BLOCK_BYTE,
                0, 0)
            input_scalar_index = 0
            # common part input scalar
            self.obj_common_input_scalar.select_key.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_common_input_scalar.need_core_num.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_common_input_scalar.ele_num_full_mask_repeat_time.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_common_input_scalar.burst_len_full_mask_repeat_time.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            # init part input scalar
            self.obj_init_input_scalar.ele_num_front_core.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_init_input_scalar.\
                init_times_full_mask_repeat_time_front_core.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_init_input_scalar.ele_num_front_part_front_core.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_init_input_scalar.burst_len_last_part_front_core.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_init_input_scalar.repeat_time_last_part_front_core.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_init_input_scalar.ele_num_last_core.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_init_input_scalar.\
                init_times_full_mask_repeat_time_last_core.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_init_input_scalar.ele_num_front_part_last_core.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_init_input_scalar.burst_len_last_part_last_core.set_as(
                tiling_ub[input_scalar_index])
            input_scalar_index = input_scalar_index + 1
            self.obj_init_input_scalar.repeat_time_last_part_last_core.set_as(
                tiling_ub[input_scalar_index])

        self.ub_tensor = self.tik_instance.Tensor("float32", (
            MASK_FP32 * MAX_REPEAT_TIME,), tik.scope_ubuf, "ub_tensor")

    def addr_clean(self, kernel_name):
        with self.tik_instance.for_range(0, self.core_num,
                                         block_num=self.core_num) as core_index:
            with self.tik_instance.if_scope(
                    core_index <
                    self.obj_common_input_scalar.need_core_num - 1):
                # front core
                with self.tik_instance.for_range(0,
                        self.obj_init_input_scalar.
                            init_times_full_mask_repeat_time_front_core) as \
                        init_index:
                    # front part front core full mask full repeat time
                    self.tik_instance.vector_dup(MASK_FP32, self.ub_tensor[0],
                                            ZERO_FP32, MAX_REPEAT_TIME, 1, 8)
                    gm_offset = core_index * \
                                self.obj_init_input_scalar.\
                                    ele_num_front_core + \
                                init_index * \
                                self.obj_common_input_scalar.\
                                    ele_num_full_mask_repeat_time
                    ub_offset = 0
                    self.tik_instance.data_move(self.gm_tensor[gm_offset],
                                                self.ub_tensor[ub_offset], 0, 1,
                                                self.obj_common_input_scalar.
                                                burst_len_full_mask_repeat_time,
                                                0, 0)
                # last part front core
                with self.tik_instance.if_scope(
                        self.obj_init_input_scalar.
                            init_times_full_mask_repeat_time_front_core == 0):
                    self.tik_instance.vector_dup(MASK_FP32, self.ub_tensor[0],
                                                 ZERO_FP32,
                        self.obj_init_input_scalar.
                        repeat_time_last_part_front_core,
                        1, 8)
                gm_offset = core_index * \
                            self.obj_init_input_scalar.ele_num_front_core + \
                            self.obj_init_input_scalar.\
                                ele_num_front_part_front_core
                self.tik_instance.data_move(self.gm_tensor[gm_offset],
                                            self.ub_tensor[0], 0, 1,
                                            self.obj_init_input_scalar.
                                            burst_len_last_part_front_core,
                                            0, 0)
            with self.tik_instance.if_scope(
                    core_index ==
                    self.obj_common_input_scalar.need_core_num - 1):
                # last core
                with self.tik_instance.for_range(0,
                        self.obj_init_input_scalar.
                            init_times_full_mask_repeat_time_last_core) as \
                        init_index:
                    # front part last core full mask full repeat time
                    self.tik_instance.vector_dup(MASK_FP32, self.ub_tensor[0],
                                                 ZERO_FP32,
                                                 MAX_REPEAT_TIME, 1, 8)
                    gm_offset = core_index * \
                                self.obj_init_input_scalar.ele_num_front_core + \
                                init_index * \
                                self.obj_common_input_scalar.\
                                    ele_num_full_mask_repeat_time
                    ub_offset = 0
                    self.tik_instance.data_move(self.gm_tensor[gm_offset],
                                                self.ub_tensor[ub_offset], 0, 1,
                                                self.obj_common_input_scalar.
                                                burst_len_full_mask_repeat_time,
                                                0, 0)
                # last part last core
                with self.tik_instance.if_scope(
                        self.obj_init_input_scalar.
                            init_times_full_mask_repeat_time_last_core == 0):
                    self.tik_instance.vector_dup(MASK_FP32, self.ub_tensor[0],
                        ZERO_FP32,
                        self.obj_init_input_scalar.
                        repeat_time_last_part_last_core,
                        1, 8)
                gm_offset = core_index * \
                            self.obj_init_input_scalar.ele_num_front_core + \
                            self.obj_init_input_scalar.\
                                ele_num_front_part_last_core
                self.tik_instance.data_move(self.gm_tensor[gm_offset],
                                            self.ub_tensor[0], 0, 1,
                                            self.obj_init_input_scalar.
                                            burst_len_last_part_last_core,
                                            0, 0)
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=[self.gm_tensor],
                                   outputs=[],
                                   flowtable=[self.tiling_gm])
        return self.tik_instance


@te.op.register_operator("DynamicAtomicAddrClean")
@check_op_params(REQUIRED_ATTR_LIST_INT, KERNEL_NAME)
def dynamic_atomic_addr_clean(size_list, kernel_name="DynamicAtomicAddrClean"):
    """
    clean memory of workspace list
    Parameters
    ----------
    size_list :  list
        sizes of workspaces
    kernel_name : str
        kernel name, default value is "DynamicAtomicAddrClean"

    Returns
    ------- 
    compile info
    """
    obj_dynamic_atomic_addr_clean = DynamicAtomicAddrClean()
    obj_dynamic_atomic_addr_clean.addr_clean(kernel_name)
    # add compile info
    te.op.add_compile_info("vars",
                           {"ub_size": obj_dynamic_atomic_addr_clean.ub_size,
                            "core_num": obj_dynamic_atomic_addr_clean.core_num})
