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
nll_loss
"""
# pylint: disable=ungrouped-imports,import-error
import math
from te import tik
from topi.cce import util
from te import platform as tbe_platform
from impl.constant_util import MASK64

DIM2 = 2
NEGATIVE = -1
TWO_KB = 2048
MAXREPEAT = 255
NUM_SIXTYFOUR = MASK64


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
def _shape_and_dtype_check(x, target, weight, kernel_name):
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    target_shape = target.get("shape")
    target_dtype = target.get("dtype").lower()
    weight_shape = weight.get("shape")
    weight_dtype = weight.get("dtype").lower()

    util.check_shape_rule(x_shape)
    util.check_shape_rule(target_shape)
    util.check_shape_rule(weight_shape)
    util.check_tensor_shape_size(x_shape)
    util.check_tensor_shape_size(target_shape)
    util.check_tensor_shape_size(weight_shape)
    util.check_kernel_name(kernel_name)
    util.check_dtype_rule(x_dtype, "float32")
    util.check_dtype_rule(target_dtype, "int32")
    util.check_dtype_rule(weight_dtype, "float32")
    if len(x_shape) > DIM2:
        raise RuntimeError("The dimension of x should be equal to"
                           "or less than 2")
    if len(target_shape) != 1:
        raise RuntimeError("The dimension of target only support 1")
    if len(x_shape) == DIM2 and x_shape[0] != target_shape[0]:
        raise RuntimeError("The first dimension of x and"
                           " target should be equal")
    if len(weight_shape) != 1:
        raise RuntimeError("The dimension of weight only support 1")
    if x_shape[-1] != weight_shape[0]:
        raise RuntimeError("The last dimension of x and the first dimension"
                           "of weight should be equal")


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-statements
# pylint: disable=attribute-defined-outside-init,too-many-lines
class NllLossCompute:
    """
    NLLLOSS

    Returns
    -------
    None
    """
    def __init__(self, x, target, weight, reduction, kernel_name):
        self.init_tik_instance()
        self.target = target
        self.weight = weight
        self.reduction = reduction
        self.kernel_name = kernel_name
        self.x_dtype = x.get("dtype").lower()
        self.x_shape = x.get("shape")
        self.target_shape = target.get("shape")
        self.target_dtype = target.get("dtype").lower()
        self.weight_shape = weight.get("shape")
        self.weight_dtype = weight.get("dtype").lower()
        self.x_dims = len(self.x_shape)
        self.n_dim = self.x_shape[0]
        self.c_dim = self.x_shape[-1]
        self.ub_size_bytes = tbe_platform.CceProductParams().getParams(
            "Unified_Buffer") - TWO_KB
        self.init_gm_size()
        self.init_gm()
        self.init_tiling_size()

    def init_tik_instance(self):
        """
        init the tik_instance.

        Parameters
        ----------

        Returns
        -------
        None
        """
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile)
        self.core_num = profile.get_aicore_num()
        self.real_core_num = self.core_num

    def init_gm_size(self):
        """
        init the size of gm.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.x_gm_size = 1
        self.target_gm_size = self.target_shape[0]
        self.weight_gm_size = self.weight_shape[0]
        self.out_gm_size = self.weight_shape[0]
        self.total_weight_size = 1
        if self.x_dims == DIM2 and self.reduction == "none":
            self.output_gm_size = self.n_dim
        else:
            self.output_gm_size = 1

    def init_tiling_size(self):
        """
        init the size of args.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.big_weight = False
        self.big_target = False
        self.none_reduction_is_multi_core = False
        self.target_ub_size = math.ceil(self.target_shape[0] /
                                        NUM_SIXTYFOUR)*NUM_SIXTYFOUR
        self.refactor_weight_size = self.target_ub_size
        self.weight_ub_size = math.ceil(self.weight_shape[0] /
                                        NUM_SIXTYFOUR+1)*NUM_SIXTYFOUR
        self.work_tensor_size = math.ceil(self.target_shape[0]/512)*8
        if self.reduction == "none":
            self.work_tensor_size = 0
        self.no_align_last_ub_size = self.ub_size_bytes/4 - \
                                     self.target_ub_size - \
                                     self.refactor_weight_size - \
                                     self.weight_ub_size - \
                                     self.work_tensor_size
        self.last_ub_size = int((self.no_align_last_ub_size//64)*64)
        self.move_max_line = (self.last_ub_size//(self.x_shape[-1]+1)//8)*8
        self.refactor_x_size = math.ceil(self.move_max_line/64)*64
        if self.move_max_line < 8 or self.reduction == "none":
            self.no_align_last_ub_size = self.ub_size_bytes/4 - \
                                         self.weight_ub_size
            self.last_ub_size = int((self.no_align_last_ub_size//64)*64)
            if self.reduction == "none":
                self.move_max_line = self.last_ub_size // (self.x_shape[-1]+3)
                if self.move_max_line > 8:
                    self.none_reduction_is_multi_core = True
                else:
                    self.move_max_line = 0
            else:
                self.big_target = True
                self.move_max_line = self.last_ub_size // (self.x_shape[-1]+4)
            self.target_ub_size = math.ceil(self.move_max_line /
                                            NUM_SIXTYFOUR)*NUM_SIXTYFOUR
            self.refactor_x_size = self.target_ub_size
            self.refactor_weight_size = self.target_ub_size
            self.work_tensor_size = math.ceil(self.move_max_line/512)*8
        self.move_max_burst = math.ceil(self.move_max_line*self.x_shape[-1]/8)
        if self.move_max_line < 8:
            self.big_weight = True
            self.big_weight_tiling()
        else:
            self.move_last_line = self.target_shape[0] % self.move_max_line
            self.max_vmul_repeat = math.ceil(self.move_max_line/64)
            self.last_vmul_repeat = math.ceil(self.move_last_line/64)
            self.compute_offset = (self.max_vmul_repeat - 1)*64
            if self.move_last_line != 0:
                self.move_last_burst = math.ceil(self.move_last_line *
                                                 self.x_shape[-1]/8)
            else:
                self.move_last_line = self.move_max_line
                self.move_last_burst = self.move_max_burst
                self.last_vmul_repeat = self.max_vmul_repeat
            self.move_times = math.ceil(self.target_shape[0] /
                                        self.move_max_line)
            if self.move_times == 1:
                self.max_vmul_repeat = self.last_vmul_repeat
                self.move_max_line = self.move_last_line
                self.move_max_burst = self.move_last_burst
            self.x_offset = self.move_max_line*self.x_shape[-1]

    def _recalculate_core_num(self, total_line):
        """
        init the size of args.

        Parameters
        ----------

        Returns
        -------
        None
        """
        core_num = self.real_core_num
        if self.reduction == "none":
            if total_line <= self.real_core_num*8:
                core_num = math.ceil(total_line/8)
        elif self.reduction == "sum":
            if total_line < self.real_core_num:
                core_num = total_line
        elif self.reduction == "mean":
            if total_line < self.real_core_num*8:
                core_num = math.ceil(total_line/8)

        return core_num

    def _compute_none_and_sum_size(self):
        """
        compute the size of tiling.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.loop_time = math.ceil(self.avg_line/self.move_max_line)
        self.redundant_line = self.n_dim - self.move_max_line * \
            (self.loop_time - 1)*self.core_num
        self.fake_core = self._recalculate_core_num(self.redundant_line)
        self.avg_line = math.ceil(self.redundant_line/self.fake_core)
        self.move_last_line = self.redundant_line // self.fake_core
        self.break_core = self.redundant_line % self.fake_core

        # if avg_line lower or equal to eight, have three situations of core
        if self.avg_line <= 8 and self.reduction != "sum":
            self.avg_line = 8
            self.break_core = self.fake_core - 1
            self.move_last_line = self.redundant_line - 8*self.break_core
            self.lower_eight_line = 0
        elif self.reduction == "mean":
            align_eight_line = self.redundant_line//8
            self.lower_eight_line = self.redundant_line % 8
            self.fake_core = self._recalculate_core_num(self.redundant_line)
            self.avg_line = math.ceil(align_eight_line/self.fake_core)*8
            self.move_last_line = (align_eight_line // self.fake_core)*8
            self.break_core = align_eight_line % self.fake_core


        self.avg_repeat = math.ceil(self.avg_line/64)
        self.last_vmul_repeat = math.ceil(self.move_last_line/64)
        self.avg_in_burst = math.ceil(self.avg_line*self.c_dim/8)
        self.last_in_burst = math.ceil(self.move_last_line*self.c_dim/8)
        if self.loop_time == 1:
            self.thread_num = 1

    def _db_tiling(self):
        self.ub_size_bytes = tbe_platform.CceProductParams().getParams(
            "Unified_Buffer")//2 - 2048
        self.init_tiling_size()

        # cancel db when cannot move weight or target to ub one time
        # with open db
        if self.big_weight or self.big_target:
            self.ub_size_bytes = tbe_platform.CceProductParams().getParams(
                "Unified_Buffer") - 2048
            self.thread_num = 1
            self.init_tiling_size()
        self._compute_none_and_sum_size()

    def init_reduction_none_and_sum_and_mean_tiling(self):
        """
        Recalculate tiling when reduction is none or sum or mean.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.core_num = self._recalculate_core_num(self.n_dim)
        self.avg_line = math.ceil(self.n_dim/self.core_num)
        if self.avg_line <= self.move_max_line:
            self.thread_num = 1
            self._compute_none_and_sum_size()
        else:
            self.thread_num = 2
            self._db_tiling()

    def init_gm(self):
        """
        init the gm of input and output.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.data_x = self.tik_instance.Tensor(self.x_dtype, [self.x_gm_size],
                                               name="data_x",
                                               scope=tik.scope_gm)
        self.data_target = self.tik_instance.Tensor(self.target_dtype,
                                                    [self.target_gm_size],
                                                    name="data_target",
                                                    scope=tik.scope_gm)
        self.data_weight = self.tik_instance.Tensor(self.weight_dtype,
                                                    [self.weight_gm_size],
                                                    name="data_weight",
                                                    scope=tik.scope_gm)
        if self.reduction == "none" or self.x_dims == 1:
            self.output = self.tik_instance.Tensor(self.x_dtype,
                                                   [self.output_gm_size],
                                                   name="output",
                                                   scope=tik.scope_gm)
        else:
            self.output = self.tik_instance.Tensor(self.x_dtype,
                                                   [self.output_gm_size],
                                                   name="output",
                                                   scope=tik.scope_gm,
                                                   is_atomic_add=True)
        if self.x_dims == DIM2 and self.reduction == "sum":
            self.total_weight = self.tik_instance.Tensor(
                self.x_dtype, [self.total_weight_size], name="total_weight",
                scope=tik.scope_gm, is_atomic_add=True)
        elif self.x_dims == DIM2 and self.reduction == "mean":
            self.total_weight = self.tik_instance.Tensor(
                self.x_dtype, [self.total_weight_size], name="total_weight",
                scope=tik.scope_gm)

    def init_normal_ub(self):
        """
        init the ub of input and output.

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.reduction != "mean" or self.big_target:
            self.refactor_weight_ub = self.tik_instance.Tensor(
                "float32", [self.refactor_weight_size],
                name="refactor_weight_ub", scope=tik.scope_ubuf)
            self.temp_weight_ub = self.tik_instance.Tensor(
                "float32", [self.weight_ub_size],
                name="temp_weight_ub", scope=tik.scope_ubuf)
            self.target_ub = self.tik_instance.Tensor(
                "float32", [self.target_ub_size],
                name="target_ub", scope=tik.scope_ubuf)
        self.refactor_x_ub = self.tik_instance.Tensor(
            "float32", [self.refactor_x_size],
            name="refactor_x_ub", scope=tik.scope_ubuf)
        self.x_ub = self.tik_instance.Tensor(
            "float32", [self.move_max_burst*8],
            name="x_ub", scope=tik.scope_ubuf)
        self.index_x = self.tik_instance.Scalar(dtype="int32")
        if self.reduction in ("mean", "sum"):
            self.work_tensor_ub = self.tik_instance.Tensor(
                "float32", [self.work_tensor_size],
                name="work_tensor_ub", scope=tik.scope_ubuf)
            self.temp_total_weight_ub = self.tik_instance.Tensor(
                "float32", [2*NUM_SIXTYFOUR], name="temp_total_weight_ub",
                scope=tik.scope_ubuf)
        if self.reduction == "none":
            self.align_tensor_ub = self.tik_instance.Tensor(
                "float32", [8],
                name="align_tensor_ub", scope=tik.scope_ubuf)
        if self.big_target and self.reduction == "mean":
            self.temp_output_ub = self.tik_instance.Tensor(
                "float32", [NUM_SIXTYFOUR], name="temp_output_ub",
                scope=tik.scope_ubuf)
            self.temp_total_x_ub = self.tik_instance.Tensor(
                "float32", [NUM_SIXTYFOUR], name="temp_total_x_ub",
                scope=tik.scope_ubuf)
        if self.reduction == "sum" or self.big_target is True:
            self.temp_total_x_ub = self.tik_instance.Tensor(
                "float32", [NUM_SIXTYFOUR], name="temp_total_x_ub",
                scope=tik.scope_ubuf)

    def init_one_dim_ub(self):
        """
        init the ub of input and output when x is 1D.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.x_ub = self.tik_instance.Tensor("float32", [NUM_SIXTYFOUR],
                                             name="x_ub",
                                             scope=tik.scope_ubuf)
        self.target_ub = self.tik_instance.Tensor("float32", [NUM_SIXTYFOUR],
                                                  name="target_ub",
                                                  scope=tik.scope_ubuf)
        self.weight_ub = self.tik_instance.Tensor("float32", [NUM_SIXTYFOUR],
                                                  name="weight_ub",
                                                  scope=tik.scope_ubuf)
        self.index_x = self.tik_instance.Scalar(dtype="int32")

    def init_big_weight_ub(self):
        """
        init the ub of input and output when weight is big.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.x_ub = self.tik_instance.Tensor("float32", [self.max_move_line*8],
                                             name="x_ub",
                                             scope=tik.scope_ubuf)
        self.target_ub = self.tik_instance.Tensor(
            "float32", [math.ceil(self.max_move_line/8)*8], name="target_ub",
            scope=tik.scope_ubuf)
        self.weight_ub = self.tik_instance.Tensor(
            "float32", [self.max_move_line*8], name="weight_ub",
            scope=tik.scope_ubuf)
        self.index_x = self.tik_instance.Scalar(dtype="int32")
        self.temp_total_weight_ub = self.tik_instance.Tensor(
            "float32", [math.ceil(self.max_move_line/8)*8],
            name="temp_total_weight_ub", scope=tik.scope_ubuf)
        self.temp_total_x_ub = self.tik_instance.Tensor(
            "float32", [math.ceil(self.max_move_line/8)*8],
            name="temp_total_x_ub", scope=tik.scope_ubuf)
        self.work_tensor_ub = self.tik_instance.Tensor(
            "float32", [self.work_tensor_size],
            name="work_tensor_ub", scope=tik.scope_ubuf)
        self.align_tensor_ub = self.tik_instance.Tensor(
            "float32", [NUM_SIXTYFOUR], name="align_tensor_ub",
            scope=tik.scope_ubuf)
        self.reduce_x_ub = self.tik_instance.Tensor(
            "float32", [NUM_SIXTYFOUR], name="reduce_x_ub",
            scope=tik.scope_ubuf)
        self.reduce_weight_ub = self.tik_instance.Tensor(
            "float32", [NUM_SIXTYFOUR], name="reduce_weight_ub",
            scope=tik.scope_ubuf)

    def one_dim_compute(self):
        """
        The normal calculate process when x is 1D and batch is 1D.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.init_one_dim_ub()
        self.tik_instance.data_move(self.target_ub, self.data_target,
                                    1, 1, 1, 0, 0)
        self.index_x.set_as(self.target_ub(0))
        self.tik_instance.data_move(self.x_ub, self.data_x[self.index_x],
                                    1, 1, 1, 0, 0)
        self.tik_instance.data_move(self.weight_ub,
                                    self.data_weight[self.index_x],
                                    1, 1, 1, 0, 0)
        self.tik_instance.vmul(1, self.x_ub, self.x_ub, self.weight_ub,
                               1, 1, 1, 1, 1, 0, 0)
        self.tik_instance.vmuls(1, self.x_ub, self.x_ub, NEGATIVE,
                                1, 1, 1, 0, 0)

        if self.x_dims == 2:
            if self.reduction == "mean":
                self.tik_instance.vdiv(1, self.x_ub, self.x_ub,
                                       self.weight_ub, 1, 1, 1, 1, 8, 8, 8)
            if self.reduction != "none":
                self.tik_instance.data_move(self.total_weight,
                                            self.weight_ub, 1, 1, 1, 0, 0)
        self.tik_instance.data_move(self.output, self.x_ub, 1, 1, 1, 0, 0)

    def select_valid_value(self, line_num, src1_line_size, src2_line_size,
                           dst1, src1, dst2, src2, target, line_offset=0):
        """
        Rearrange tensor according to target.

        Parameters
        ----------

        Returns
        -------
        None
        """
        if line_num >= 8:
            vars_num = 8
            loop_num = int(line_num//8)
            last_line = int(line_num) % 8
        else:
            vars_num = int(line_num)
            loop_num = 0
            last_line = int(line_num)
        names = locals()
        for i in range(0, vars_num):
            names["index_x" + str(i)] = self.tik_instance.Scalar(dtype="int32")
        with self.tik_instance.for_range(0, loop_num) as time:
            offset_set = 8 * time
            for i in range(0, vars_num):
                names["index_x" + str(i)].set_as(
                    target[offset_set + i + line_offset])
            for i in range(0, vars_num):
                dst_offset = offset_set+i
                src1_offset = dst_offset*src1_line_size + \
                              names["index_x" + str(i)]
                src2_offset = dst_offset*src2_line_size + \
                              names["index_x" + str(i)]
                dst1[dst_offset].set_as(src1[src1_offset])
                if self.reduction != "mean" or self.big_target:
                    dst2[dst_offset].set_as(src2[src2_offset])
        for i in range(0, last_line):
            names["index_x" + str(i)].set_as(
                target[loop_num*8 + i + line_offset])
        for i in range(0, last_line):
            dst_offset = loop_num*8+i
            src1_offset = dst_offset*src1_line_size + names["index_x" + str(i)]
            src2_offset = dst_offset*src2_line_size + names["index_x" + str(i)]
            dst1[dst_offset].set_as(src1[src1_offset])
            if self.reduction != "mean" or self.big_target:
                dst2[dst_offset].set_as(src2[src2_offset])

    def _none_and_sum_compute_process(self, line, repeat, offset,
                                      x_burst, out_burst):
        self.tik_instance.data_move(self.temp_weight_ub, self.data_weight,
                                    0, 1, math.ceil(self.c_dim/8), 0, 0)
        self.tik_instance.data_move(
            self.x_ub, self.data_x[offset*self.c_dim], 0, 1,
            x_burst, 0, 0)
        self.tik_instance.data_move(
            self.target_ub, self.data_target[offset],
            0, 1, math.ceil(line/8), 0, 0)
        self.select_valid_value(line, 0, self.c_dim, self.refactor_weight_ub,
                                self.temp_weight_ub, self.refactor_x_ub,
                                self.x_ub, self.target_ub)
        self.tik_instance.vmuls(MASK64, self.refactor_x_ub,
                                self.refactor_x_ub, NEGATIVE,
                                repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(
            MASK64, self.refactor_x_ub, self.refactor_x_ub,
            self.refactor_weight_ub, repeat, 1, 1, 1, 8, 8, 8)

        if self.reduction == "none":
            if line % 8 != 0 and self.core_num != 1 and out_burst > 1:
                with self.tik_instance.for_range(0, 8) as i:
                    self.align_tensor_ub[i].set_as(
                        self.refactor_x_ub[line - 8 + i])
                self.tik_instance.data_move(
                    self.output[offset], self.refactor_x_ub,
                    0, 1, out_burst-1, 8, 8)
                self.tik_instance.data_move(
                    self.output[offset + line - 8],
                    self.align_tensor_ub, 0, 1, 1, 0, 0)
            else:
                self.tik_instance.data_move(
                    self.output[offset],
                    self.refactor_x_ub, 0, 1, out_burst, 0, 0)
        if self.reduction == "sum":
            self.sum_compute(line, self.x_ub, self.refactor_x_ub,
                             self.work_tensor_ub)
            self.sum_compute(line, self.temp_weight_ub,
                             self.refactor_weight_ub, self.work_tensor_ub)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(
                self.output, self.x_ub, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(
                self.total_weight, self.temp_weight_ub, 0, 1, 1, 0, 0)
            self.tik_instance.set_atomic_add(0)

    def reduction_is_none_and_sum_compute(self):
        """
        The normal calculate process when x is 2D and reduction is none or sum.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.init_reduction_none_and_sum_and_mean_tiling()
        max_move_out_burst = math.ceil(self.move_max_line/8)
        last_move_out_burst = math.ceil(self.move_last_line/8)
        avg_move_out_burst = math.ceil(self.avg_line/8)
        with self.tik_instance.for_range(0, self.core_num,
                                         block_num=self.core_num) as cycle:
            with self.tik_instance.for_range(
                    0, self.loop_time, thread_num=self.thread_num) as loop:
                core_offset = cycle*self.avg_line
                lower_core_offset = cycle*self.move_last_line+self.break_core
                if self.avg_line == 8:
                    lower_core_offset = cycle*8
                loop_offset = loop*self.core_num*self.move_max_line
                self.init_normal_ub()
                with self.tik_instance.if_scope(loop < self.loop_time - 1):
                    self._none_and_sum_compute_process(
                        self.move_max_line, self.max_vmul_repeat,
                        loop_offset+self.move_max_line*cycle,
                        self.move_max_burst, max_move_out_burst)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(cycle < self.fake_core):
                        with self.tik_instance.if_scope(
                                cycle < self.break_core):
                            self._none_and_sum_compute_process(
                                self.avg_line, self.avg_repeat,
                                loop_offset + core_offset, self.avg_in_burst,
                                avg_move_out_burst)
                        with self.tik_instance.else_scope():
                            self._none_and_sum_compute_process(
                                self.move_last_line, self.last_vmul_repeat,
                                loop_offset+lower_core_offset,
                                self.last_in_burst, last_move_out_burst)

    def sum_compute(self, sum_size, dst_ub, src_ub, work_ub):
        """
        calculate tensor sum.

        Parameters
        ----------

        Returns
        -------
        None
        """
        vcadd_repeat_time = math.ceil(sum_size/NUM_SIXTYFOUR)
        head_repeat = vcadd_repeat_time - 1
        tail_mask = int(sum_size % NUM_SIXTYFOUR)
        if tail_mask == 0:
            self.tik_instance.vec_reduce_add(MASK64, dst_ub, src_ub,
                                             work_ub, vcadd_repeat_time, 8)
        else:
            self.tik_instance.vcadd(
                tail_mask, dst_ub,
                src_ub[head_repeat*NUM_SIXTYFOUR], 1, 1, 1, 8)
            if head_repeat > 0:
                self.tik_instance.vec_reduce_add(MASK64, dst_ub[NUM_SIXTYFOUR],
                                                 src_ub, work_ub,
                                                 head_repeat, 8)
                self.tik_instance.vadd(1, dst_ub, dst_ub,
                                       dst_ub[NUM_SIXTYFOUR],
                                       1, 1, 1, 1, 8, 8, 8)

    def mean_compute_process(self, line, offset, x_burst, mask=MASK64):
        """
        The normal calculate process when x is 2D and reduction is mean.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.tik_instance.data_move(
            self.x_ub, self.data_x[offset*self.c_dim], 0, 1,
            x_burst, 8, 8)
        self.select_valid_value(
            line, self.c_dim, 0, self.refactor_x_ub, self.x_ub,
            self.refactor_x_ub, self.x_ub, self.target_ub, offset)
        self.tik_instance.vmul(
            mask, self.refactor_x_ub, self.refactor_x_ub,
            self.refactor_weight_ub[offset],
            math.ceil(line/64), 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.refactor_x_ub,
                                self.refactor_x_ub, NEGATIVE,
                                math.ceil(line/64), 1, 1, 8, 8)
        self.sum_compute(line, self.x_ub,
                         self.refactor_x_ub, self.work_tensor_ub)

        self.tik_instance.vdiv(1, self.x_ub, self.x_ub,
                               self.temp_weight_ub,
                               1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.set_atomic_add(1)
        self.tik_instance.data_move(self.output, self.x_ub,
                                    0, 1, 1, 0, 0)
        self.tik_instance.set_atomic_add(0)

    def _init_mean_ub(self):
        self.temp_weight_ub = self.tik_instance.Tensor(
            "float32", [self.weight_ub_size],
            name="temp_weight_ub", scope=tik.scope_ubuf)
        self.target_ub = self.tik_instance.Tensor(
            "float32", [self.target_ub_size],
            name="target_ub", scope=tik.scope_ubuf)
        self.refactor_weight_ub = self.tik_instance.Tensor(
            "float32", [self.refactor_weight_size],
            name="refactor_weight_ub", scope=tik.scope_ubuf)
        self.work_tensor_weight_ub = self.tik_instance.Tensor(
            "float32", [self.work_tensor_size],
            name="work_tensor_weight_ub", scope=tik.scope_ubuf)

    def reduction_is_mean_compute(self):
        """
        The normal calculate process when x is 2D and reduction is mean.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.init_reduction_none_and_sum_and_mean_tiling()
        with self.tik_instance.for_range(0, self.core_num,
                                         block_num=self.core_num) as cycle:
            self._init_mean_ub()
            self.tik_instance.data_move(self.temp_weight_ub, self.data_weight,
                                        0, 1, math.ceil(self.c_dim/8), 0, 0)
            self.tik_instance.data_move(self.target_ub, self.data_target, 0, 1,
                                        math.ceil(self.n_dim/8), 0, 0)
            self.select_valid_value(
                self.n_dim, 0, 0, self.refactor_weight_ub, self.temp_weight_ub,
                self.refactor_weight_ub, self.temp_weight_ub, self.target_ub)
            self.sum_compute(self.n_dim, self.temp_weight_ub,
                             self.refactor_weight_ub,
                             self.work_tensor_weight_ub)

            with self.tik_instance.for_range(
                    0, self.loop_time, thread_num=self.thread_num) as loop:
                self.init_normal_ub()
                core_offset = cycle*self.avg_line
                lower_core_offset = cycle*self.move_last_line+self.break_core*8
                if self.avg_line == 8:
                    lower_core_offset = cycle*8
                loop_offset = loop*self.core_num*self.move_max_line
                with self.tik_instance.if_scope(loop < self.loop_time - 1):
                    self.mean_compute_process(
                        self.move_max_line,
                        loop_offset+cycle*self.move_max_line,
                        self.move_max_burst)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(cycle < self.fake_core):
                        with self.tik_instance.if_scope(
                                cycle < self.break_core):
                            self.mean_compute_process(
                                self.avg_line, loop_offset + core_offset,
                                math.ceil(self.avg_line*self.c_dim/8))
                        with self.tik_instance.else_scope():
                            with self.tik_instance.if_scope(
                                    cycle < self.fake_core-1):
                                self.mean_compute_process(
                                    self.move_last_line,
                                    loop_offset + lower_core_offset,
                                    math.ceil(self.move_last_line *
                                              self.c_dim/8))
                            with self.tik_instance.else_scope():
                                line = self.move_last_line
                                if self.lower_eight_line != 0:
                                    line = self.lower_eight_line + \
                                           self.move_last_line
                                self.mean_compute_process(
                                    self.lower_eight_line+self.move_last_line,
                                    loop_offset + lower_core_offset,
                                    math.ceil(line * self.c_dim/8))
                                self.tik_instance.data_move(
                                    self.total_weight, self.temp_weight_ub,
                                    0, 1, 1, 0, 0)

    def big_target_compute(self):
        """
        The calculate process of target cannot move to ub one time.

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.move_times) as cycle:
            self.init_normal_ub()
            self.tik_instance.data_move(self.temp_weight_ub, self.data_weight,
                                        0, 1, math.ceil(self.c_dim/8), 0, 0)
            with self.tik_instance.if_scope(cycle < self.move_times - 1):
                self.tik_instance.data_move(
                    self.x_ub, self.data_x[cycle*self.x_offset], 0, 1,
                    self.move_max_burst, 0, 0)
                self.tik_instance.data_move(
                    self.target_ub, self.data_target[cycle*self.move_max_line],
                    0, 1, math.ceil(self.move_max_line/8), 0, 0)
                self.select_valid_value(
                    self.move_max_line, self.c_dim, 0, self.refactor_x_ub,
                    self.x_ub, self.refactor_weight_ub, self.temp_weight_ub,
                    self.target_ub)

                self.tik_instance.vmul(
                    MASK64, self.refactor_x_ub, self.refactor_x_ub,
                    self.refactor_weight_ub, self.max_vmul_repeat,
                    1, 1, 1, 8, 8, 8)
                self.sum_compute(self.move_max_line, self.x_ub,
                                 self.refactor_x_ub, self.work_tensor_ub)
                self.sum_compute(self.move_max_line, self.temp_total_weight_ub,
                                 self.refactor_weight_ub, self.work_tensor_ub)
                with self.tik_instance.if_scope(cycle == 0):
                    self.tik_instance.data_move(
                        self.temp_output_ub, self.temp_total_weight_ub,
                        0, 1, 1, 0, 0)
                    self.tik_instance.data_move(
                        self.temp_total_x_ub, self.x_ub, 0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.vadd(
                        1, self.temp_output_ub, self.temp_output_ub,
                        self.temp_total_weight_ub, 1, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(
                        1, self.temp_total_x_ub, self.temp_total_x_ub,
                        self.x_ub, 1, 1, 1, 1, 8, 8, 8)
            with self.tik_instance.if_scope(cycle == self.move_times - 1):
                self.tik_instance.data_move(
                    self.x_ub, self.data_x[cycle*self.x_offset], 0, 1,
                    self.move_last_burst, 0, 0)
                self.tik_instance.data_move(
                    self.target_ub, self.data_target[cycle*self.move_max_line],
                    0, 1, math.ceil(self.move_last_line/8), 0, 0)
                self.select_valid_value(
                    self.move_last_line, self.c_dim, 0, self.refactor_x_ub,
                    self.x_ub, self.refactor_weight_ub, self.temp_weight_ub,
                    self.target_ub)

                self.tik_instance.vmul(
                    MASK64, self.refactor_x_ub, self.refactor_x_ub,
                    self.refactor_weight_ub, self.last_vmul_repeat,
                    1, 1, 1, 8, 8, 8)
                self.sum_compute(self.move_last_line, self.x_ub,
                                 self.refactor_x_ub, self.work_tensor_ub)
                self.sum_compute(self.move_last_line,
                                 self.temp_total_weight_ub,
                                 self.refactor_weight_ub, self.work_tensor_ub)
                with self.tik_instance.if_scope(cycle == 0):
                    self.tik_instance.data_move(
                        self.temp_output_ub, self.temp_total_weight_ub,
                        0, 1, 1, 0, 0)
                    self.tik_instance.data_move(
                        self.temp_total_x_ub, self.x_ub, 0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.vadd(
                        1, self.temp_output_ub, self.temp_output_ub,
                        self.temp_total_weight_ub, 1, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(
                        1, self.temp_total_x_ub, self.temp_total_x_ub,
                        self.x_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vdiv(
                    1, self.temp_total_x_ub, self.temp_total_x_ub,
                    self.temp_output_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vmuls(
                    1, self.temp_total_x_ub, self.temp_total_x_ub,
                    NEGATIVE, 1, 1, 1, 8, 8)
                self.tik_instance.data_move(self.output, self.temp_total_x_ub,
                                            0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.total_weight,
                                            self.temp_output_ub, 0, 1, 1, 0, 0)

    def big_weight_tiling(self):
        """
        The tiling process when weight cannot move to ub one time.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.thread_num = 1
        if self.n_dim <= 8*self.core_num:
            self.core_num = math.ceil(self.n_dim/8)
            self.max_move_line = 8
            self.redundant_line = 0
            self.last_move_line = self.n_dim - (self.core_num - 1)*8
            if self.core_num == 1:
                self.max_move_line = self.target_shape[0]
                self.last_move_line = self.max_move_line
        else:
            self.max_move_line = math.ceil(self.target_shape[0] /
                                           self.core_num)
            self.last_move_line = self.target_shape[0] // self.core_num
            self.redundant_line = self.n_dim % self.core_num

        # one_line_size include two block for gm to ub and two number of target
        # and refactor x.
        one_line_size = 18
        size = (self.ub_size_bytes*0.5)//4 - (self.max_move_line*one_line_size)
        if size < 0 or self.reduction == "mean":
            self.max_move_line = int((self.ub_size_bytes*0.5) // 4 //
                                     one_line_size)
            self.last_move_line = self.n_dim % self.max_move_line
            self.redundant_line = 0

        if self.last_move_line == 0:
            self.last_move_line = self.max_move_line
        self.move_time = math.ceil(self.target_shape[0] / self.max_move_line)
        if self.reduction == "mean" and self.move_time == 1:
            self.max_move_line = self.last_move_line
        self.max_burst_len = math.ceil(self.max_move_line/8)
        self.last_burst_len = math.ceil(self.last_move_line/8)
        self.loop = math.ceil(self.move_time/self.core_num)
        self.work_tensor_size = math.ceil(self.max_move_line/512)*8
        if self.loop > 1:
            self.thread_num = 2

    def _big_weight_data_move(self, total_x_ub, total_weight_ub, sort_index,
                              burst_len, repeat, offset):
        """
        data move process, place x or weight one by one at a block interval.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.tik_instance.data_move(
            self.target_ub, self.data_target[offset],
            0, 1, burst_len, 0, 0)
        with self.tik_instance.for_range(0, sort_index) as index:
            self.index_x.set_as(self.target_ub[index])
            self.tik_instance.data_move(
                self.x_ub[8*index],
                self.data_x[self.c_dim*(index+offset)+self.index_x],
                0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.weight_ub[8*index],
                                        self.data_weight[self.index_x],
                                        0, 1, 1, 0, 0)
        if repeat > MAXREPEAT:
            loop_repeat = math.ceil(repeat/MAXREPEAT)
            max_repeat = MAXREPEAT
        else:
            max_repeat = 1
            loop_repeat = 1
        tail_offset = (loop_repeat - 1)*MAXREPEAT*8
        tail_repeat = int(repeat % MAXREPEAT)
        if tail_repeat == 0:
            tail_repeat = MAXREPEAT
        with self.tik_instance.for_range(0, loop_repeat-1) as loop:
            self.tik_instance.vmul(1, self.x_ub[loop*MAXREPEAT*8],
                                   self.x_ub[loop*MAXREPEAT*8],
                                   self.weight_ub[loop*MAXREPEAT*8],
                                   max_repeat, 1, 1, 1, 1, 1, 1)
            self.tik_instance.vmuls(1, self.x_ub[loop*MAXREPEAT*8],
                                    self.x_ub[loop*MAXREPEAT*8],
                                    NEGATIVE, max_repeat, 1, 1, 1, 1)
        self.tik_instance.vmul(1, self.x_ub[tail_offset],
                               self.x_ub[tail_offset],
                               self.weight_ub[tail_offset],
                               tail_repeat, 1, 1, 1, 1, 1, 1)
        self.tik_instance.vmuls(1, self.x_ub[tail_offset],
                                self.x_ub[tail_offset], NEGATIVE, tail_repeat,
                                1, 1, 1, 1)
        with self.tik_instance.for_range(0, sort_index) as index:
            total_x_ub[index].set_as(self.x_ub[8*index])
            if self.reduction != "none":
                total_weight_ub[index].set_as(self.weight_ub[8*index])

    def big_weight_with_none_and_sum(self, move_line, burst_len, offset):
        """
        The normal calculate process of reduction is none or sum.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self._big_weight_data_move(
            self.temp_total_x_ub, self.temp_total_weight_ub,
            move_line, burst_len, move_line, offset)
        if self.reduction == "none":
            if move_line % 8 != 0 and self.core_num != 1:
                with self.tik_instance.for_range(0, 8) as i:
                    self.align_tensor_ub[i].set_as(
                        self.temp_total_x_ub[move_line - 8 + i])
                self.tik_instance.data_move(
                    self.output[offset], self.temp_total_x_ub, 0, 1,
                    burst_len-1, 0, 0)
                self.tik_instance.data_move(
                    self.output(offset + move_line - 8),
                    self.align_tensor_ub, 0, 1, 1, 0, 0)
            else:
                self.tik_instance.data_move(
                    self.output(offset), self.temp_total_x_ub, 0, 1,
                    burst_len, 0, 0)
        elif self.reduction == "sum":
            self.sum_compute(move_line, self.x_ub,
                             self.temp_total_x_ub,
                             self.work_tensor_ub)
            self.sum_compute(move_line, self.weight_ub,
                             self.temp_total_weight_ub,
                             self.work_tensor_ub)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(
                self.output, self.x_ub, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(
                self.total_weight, self.weight_ub, 0, 1, 1, 0, 0)
            self.tik_instance.set_atomic_add(0)

    def big_weight_compute(self):
        """
        The calculate process when weight cannot move to ub one time.

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.core_num,
                                         block_num=self.core_num) as cycle:
            with self.tik_instance.for_range(
                    0, self.loop, thread_num=self.thread_num) as loop:
                self.init_big_weight_ub()
                current_time = self.core_num*loop+cycle
                offset = current_time*self.max_move_line
                if self.loop == 1:
                    with self.tik_instance.if_scope(
                            current_time < self.redundant_line):
                        self.big_weight_with_none_and_sum(
                            self.max_move_line, self.max_burst_len, offset)
                    with self.tik_instance.else_scope():
                        offset = current_time*self.last_move_line + \
                                 self.redundant_line
                        self.big_weight_with_none_and_sum(
                            self.last_move_line, self.last_burst_len, offset)
                else:
                    with self.tik_instance.if_scope(
                            current_time < self.move_time - 1):
                        self.big_weight_with_none_and_sum(
                            self.max_move_line, self.max_burst_len, offset)
                    with self.tik_instance.if_scope(
                            current_time == self.move_time - 1):
                        self.big_weight_with_none_and_sum(
                            self.last_move_line, self.last_burst_len, offset)

    def big_weight_with_mean(self):
        """
        The calculate process when weight cannot move to ub one time and
        reduction is mean.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.init_big_weight_ub()
        with self.tik_instance.for_range(0, self.move_time) as cycle:
            offset = cycle*self.max_move_line
            with self.tik_instance.if_scope(cycle < self.move_time - 1):
                self._big_weight_data_move(
                    self.temp_total_x_ub, self.temp_total_weight_ub,
                    self.max_move_line, self.max_burst_len,
                    self.max_move_line, offset)
                self.sum_compute(self.max_move_line, self.x_ub,
                                 self.temp_total_x_ub,
                                 self.work_tensor_ub)
                self.sum_compute(self.max_move_line, self.weight_ub,
                                 self.temp_total_weight_ub,
                                 self.work_tensor_ub)
                with self.tik_instance.if_scope(cycle == 0):
                    self.tik_instance.data_move(
                        self.reduce_x_ub, self.x_ub,
                        0, 1, 1, 0, 0)
                    self.tik_instance.data_move(
                        self.reduce_weight_ub, self.weight_ub, 0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.vadd(1, self.reduce_weight_ub,
                                           self.reduce_weight_ub,
                                           self.weight_ub, 1, 1, 1,
                                           1, 8, 8, 8)
                    self.tik_instance.vadd(
                        1, self.reduce_x_ub, self.reduce_x_ub,
                        self.x_ub, 1, 1, 1, 1, 8, 8, 8)
            with self.tik_instance.if_scope(cycle == self.move_time-1):
                self._big_weight_data_move(
                    self.temp_total_x_ub, self.temp_total_weight_ub,
                    self.last_move_line, self.last_burst_len,
                    self.last_move_line, offset)
                self.sum_compute(self.last_move_line, self.x_ub,
                                 self.temp_total_x_ub, self.work_tensor_ub)
                self.sum_compute(self.last_move_line, self.weight_ub,
                                 self.temp_total_weight_ub,
                                 self.work_tensor_ub)
                with self.tik_instance.if_scope(cycle == 0):
                    self.tik_instance.data_move(
                        self.reduce_x_ub, self.x_ub,
                        0, 1, 1, 0, 0)
                    self.tik_instance.data_move(
                        self.reduce_weight_ub, self.weight_ub, 0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.vadd(
                        1, self.reduce_weight_ub, self.reduce_weight_ub,
                        self.weight_ub, 1, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vadd(
                        1, self.reduce_x_ub, self.reduce_x_ub,
                        self.x_ub, 1, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vdiv(1, self.reduce_x_ub, self.reduce_x_ub,
                               self.reduce_weight_ub, 1, 1, 1, 1, 0, 0, 0)
        self.tik_instance.data_move(self.output, self.reduce_x_ub,
                                    0, 1, 1, 0, 0)
        self.tik_instance.data_move(self.total_weight, self.reduce_weight_ub,
                                    0, 1, 1, 0, 0)

    def nll_loss_compute_start(self):
        """
        Different calculation methods

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.x_dims == 1 or self.n_dim == 1:
            self.one_dim_compute()
        elif self.big_weight:
            if self.reduction == "mean":
                self.big_weight_with_mean()
            else:
                self.big_weight_compute()
        elif self.x_dims == DIM2 and self.reduction == "mean":
            if self.big_target:
                self.big_target_compute()
            else:
                self.reduction_is_mean_compute()
        elif self.x_dims == DIM2:
            self.reduction_is_none_and_sum_compute()
        else:
            raise RuntimeError("No algorithm matched, please check"
                               "distribution rules.")

        if self.x_dims == DIM2 and (self.reduction == "sum" or
                                    self.reduction == "mean"):
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=[self.data_x, self.data_target,
                                               self.data_weight],
                                       outputs=[self.output, self.total_weight])
        else:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=[self.data_x, self.data_target,
                                               self.data_weight],
                                       outputs=[self.output])
        return self.tik_instance


@util.check_input_type(dict, dict, dict, dict, dict, str, str)
def nll_loss(x, target, weight, y, total_weight, reduction="mean",
             kernel_name="nll_loss"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input, the length of shape should be two or one.
    target : dict
        shape and dtype of input, the length of shape only support one.
    weight : dict or None
        the length of shape only support one when weight is dict.
    y:dict
        It’s a tensor with shape(minibatch, ) when reduction == ‘none’ and
        the input is 2D. Otherwise, the output is a scalar.
    total_weight:
        shape and dtype of output, should be same type as weight
    reduction: str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "nll_loss"

    Returns
    -------
    None
    """
    _shape_and_dtype_check(x, target, weight, kernel_name)
    nll_loss_function = NllLossCompute(x, target, weight,
                                         reduction, kernel_name)
    return nll_loss_function.nll_loss_compute_start()
