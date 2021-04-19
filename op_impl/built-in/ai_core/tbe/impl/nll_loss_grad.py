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
nll_loss_grad
"""
# pylint: disable=ungrouped-imports,import-error
import math
from te import tik
from topi.cce import util
from te import platform as tbe_platform
from impl.constant_util import MASK64

DIM2 = 2
NUM_EIGHT = 8
NUM_FOUR = 4
NEGATIVE = -1
MAX_REPEAT = 255
ONE_KB = 1024
NUM_SIXTYFOUR = MASK64


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# pylint: disable=too-many-arguments
def _shape_and_dtype_check(x, y_grad, target, weight, total_weight, reduction,
                           kernel_name):
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    y_grad_shape = y_grad.get("shape")
    y_grad_dtype = y_grad.get("dtype").lower()
    target_shape = target.get("shape")
    target_dtype = target.get("dtype").lower()
    total_weight_shape = total_weight.get("shape")
    total_weight_dtype = total_weight.get("dtype").lower()
    weight_shape = weight.get("shape")
    weight_dtype = weight.get("dtype").lower()
    util.check_tensor_shape_size(weight_shape)
    util.check_shape_rule(weight_shape)

    util.check_shape_rule(x_shape)
    util.check_shape_rule(y_grad_shape)
    util.check_shape_rule(target_shape)
    util.check_tensor_shape_size(y_grad_shape)
    util.check_tensor_shape_size(target_shape)

    util.check_kernel_name(kernel_name)
    util.check_dtype_rule(x_dtype, "float32")
    util.check_dtype_rule(y_grad_dtype, "float32")
    util.check_dtype_rule(target_dtype, "int32")
    util.check_dtype_rule(weight_dtype, "float32")
    util.check_dtype_rule(total_weight_dtype, "float32")

    if reduction in ("mean", "sum") and y_grad_shape[0] != 1:
        raise RuntimeError("The shape of y_grad must be (1,),"
                           " while reduction is mean or sum. ")
    if len(x_shape) == 1 and y_grad_shape[0] != 1:
        raise RuntimeError("The shape of y_grad must be (1,),"
                           " while input x is 1D. ")
    if len(x_shape) > DIM2:
        raise RuntimeError("The dimension of x should be equal to"
                           "or less than two.")
    if len(x_shape) == DIM2 and x_shape[0] != target_shape[0]:
        raise RuntimeError("The first dimension of x and"
                           " target should be equal")
    if x_shape[-1] != weight_shape[0]:
        raise RuntimeError("The last dimension of x and the first dimension"
                           " of weight should be equal")
    if len(y_grad_shape) != 1:
        raise RuntimeError("The dimension of y_grad should be 1D.")
    if len(weight_shape) != 1:
        raise RuntimeError("The dimension of weight should be 1D.")
    if len(target_shape) != 1:
        raise RuntimeError("The dimension of target should be 1D.")
    if total_weight_shape[0] != 1:
        raise RuntimeError("The shape of total_weight must be (1,)")


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-statements
# pylint: disable=attribute-defined-outside-init
class NllLossGradCompute:
    """
    NLLLOSSGRAD

    Returns
    -------
    None
    """
    def __init__(self, x, y_grad, target, weight, reduction, kernel_name):
        self.init_tik_instance()
        self.reduction = reduction
        self.kernel_name = kernel_name
        self.x_shape = x.get("shape")
        self.x_dtype = x.get("dtype").lower()
        self.y_grad_shape = y_grad.get("shape")
        self.y_grad_dtype = y_grad.get("dtype").lower()
        self.target_shape = target.get("shape")
        self.target_dtype = target.get("dtype").lower()
        self.weight_shape = weight.get("shape")
        self.weight_dtype = weight.get("dtype").lower()
        self.x_dim = len(self.x_shape)
        self.init_size()
        self.init_gm()

    def init_tik_instance(self):
        """
        init the tik_instance

        Parameters
        ----------

        Returns
        -------
        None
        """
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile)
        self.real_core_num = profile.get_aicore_num()
        self.l1_buffer_size = profile.get_l1_buffer_size()

    def init_size(self):
        """
        init the size of args.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.big_weight = False
        self.core_num = self.x_shape[0]
        self.ub_size_bytes = tbe_platform.CceProductParams().getParams(
            "Unified_Buffer") - 2048
        self.ub_size_float = self.ub_size_bytes//4
        self.db_ub_size = self.ub_size_bytes/2
        if len(self.x_shape) == DIM2:
            self.output_gm_size = self.x_shape[0] * self.x_shape[1]
        else:
            self.output_gm_size = self.x_shape[0]
        self.n_dim = self.x_shape[0]
        self.c_dim = self.x_shape[-1]
        self.repeat_time = math.ceil(self.c_dim / NUM_SIXTYFOUR)
        self.move_len = math.ceil(self.c_dim / NUM_EIGHT)
        if self.c_dim > NUM_EIGHT:
            self.end_num = self.c_dim % NUM_EIGHT
        self.y_grad_ub_size = math.ceil(self.y_grad_shape[0]/NUM_SIXTYFOUR) * \
                              NUM_SIXTYFOUR
        self.target_ub_size = math.ceil(self.target_shape[0]/NUM_SIXTYFOUR) * \
                              NUM_SIXTYFOUR
        self.total_weight_ub_size = NUM_SIXTYFOUR
        self.weight_ub_size = math.ceil(self.weight_shape[0]/NUM_SIXTYFOUR) * \
                              NUM_SIXTYFOUR
        self.dup_ub_size = math.ceil(self.x_shape[-1]/NUM_SIXTYFOUR) * \
                           NUM_SIXTYFOUR
        self.y_grad_gm_size = self.y_grad_shape[0]
        self.target_gm_size = self.target_shape[0]
        self.data_total_weight_size = 1
        self.weight_gm_size = self.weight_shape[0]
        self.dup_repeat = math.ceil(self.c_dim/NUM_SIXTYFOUR)
        self.last_ub_size = self.ub_size_float - self.weight_ub_size

        # one_line_size include one num of refactor weight, one num of target
        # and one num of y_grad or total_weight
        one_line_size = self.c_dim + 3

        self.max_move_line = self.last_ub_size//one_line_size
        if self.max_move_line < 2:
            self.big_weight = True

    def init_gm(self):
        """
        init the gm of input and output.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.data_x = self.tik_instance.Tensor(self.y_grad_dtype,
                                               [self.y_grad_gm_size],
                                               name="data_x",
                                               scope=tik.scope_gm)
        self.data_y_grad = self.tik_instance.Tensor(self.y_grad_dtype,
                                                    [self.y_grad_gm_size],
                                                    name="data_y_grad",
                                                    scope=tik.scope_gm)
        self.data_weight = self.tik_instance.Tensor(self.weight_dtype,
                                                    [self.weight_gm_size],
                                                    name="data_weight",
                                                    scope=tik.scope_gm)
        self.data_target = self.tik_instance.Tensor(self.target_dtype,
                                                    [self.target_gm_size],
                                                    name="data_target",
                                                    scope=tik.scope_gm)
        self.data_total_weight = self.tik_instance.Tensor(
            self.x_dtype, [self.data_total_weight_size],
            name="data_total_weight", scope=tik.scope_gm)
        self.output = self.tik_instance.Tensor(self.x_dtype,
                                               [self.output_gm_size],
                                               name="output",
                                               scope=tik.scope_gm)

    def init_ub(self):
        """
        init the ub of input and output.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.y_grad_ub = self.tik_instance.Tensor("float32",
                                                  [self.y_grad_ub_size],
                                                  name="y_grad_ub",
                                                  scope=tik.scope_ubuf)
        self.target_ub = self.tik_instance.Tensor(self.weight_dtype,
                                                  [self.target_ub_size],
                                                  name="target_ub",
                                                  scope=tik.scope_ubuf)
        self.total_weight_ub = self.tik_instance.Tensor(
            self.x_dtype, [self.total_weight_ub_size], name="total_weight_ub",
            scope=tik.scope_ubuf)
        self.weight_ub = self.tik_instance.Tensor(self.weight_dtype,
                                                  [self.weight_ub_size],
                                                  name="weight_ub",
                                                  scope=tik.scope_ubuf)

        self.dup_ub = self.tik_instance.Tensor(self.x_dtype,
                                               [self.dup_ub_size],
                                               name="dup_ub",
                                               scope=tik.scope_ubuf)
        self.index_x = self.tik_instance.Scalar(dtype="int32")

    def init_one_dim_and_big_weight_ub(self):
        """
        init the ub of input and output when input is one dim or big weight.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.dup_ub = self.tik_instance.Tensor(self.x_dtype,
                                               [self.dup_ub_size],
                                               name="dup_ub",
                                               scope=tik.scope_ubuf)
        self.y_grad_ub = self.tik_instance.Tensor("float32",
                                                  [NUM_SIXTYFOUR],
                                                  name="y_grad_ub",
                                                  scope=tik.scope_ubuf)
        self.target_ub = self.tik_instance.Tensor(self.weight_dtype,
                                                  [NUM_SIXTYFOUR],
                                                  name="target_ub",
                                                  scope=tik.scope_ubuf)
        if self.big_weight is True:
            self.total_weight_ub = self.tik_instance.Tensor(
                self.x_dtype, [NUM_SIXTYFOUR],
                name="total_weight_ub", scope=tik.scope_ubuf)
        self.weight_ub = self.tik_instance.Tensor(self.weight_dtype,
                                                  [NUM_SIXTYFOUR],
                                                  name="weight_ub",
                                                  scope=tik.scope_ubuf)
        if self.data_weight_size < 1024*256:
            self.data_weight_l1 = self.tik_instance.Tensor(
                self.weight_dtype, [self.data_weight_size],
                name="data_weight_l1", scope=tik.scope_cbuf)
        self.index_x = self.tik_instance.Scalar(dtype="int32")

    def init_normal_two_dim_ub(self):
        """
        init the ub of input and output when input is normal two dim.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.y_grad_ub = self.tik_instance.Tensor("float32",
                                                  [self.y_grad_ub_size],
                                                  name="y_grad_ub",
                                                  scope=tik.scope_ubuf)
        self.target_ub = self.tik_instance.Tensor(self.target_dtype,
                                                  [self.target_ub_size],
                                                  name="target_ub",
                                                  scope=tik.scope_ubuf)
        self.weight_ub = self.tik_instance.Tensor(self.weight_dtype,
                                                  [self.weight_ub_size],
                                                  name="weight_ub",
                                                  scope=tik.scope_ubuf)
        self.refactor_weight_ub = self.tik_instance.Tensor(
            self.weight_dtype, [self.refactor_weight_ub_size],
            name="refactor_weight_ub", scope=tik.scope_ubuf)
        if self.reduction != "none":
            self.total_weight_ub = self.tik_instance.Tensor(
                self.x_dtype, [NUM_SIXTYFOUR],
                name="total_weight_ub", scope=tik.scope_ubuf)
        self.dup_ub = self.tik_instance.Tensor(
            self.x_dtype, [self.dup_ub_size], name="dup_ub",
            scope=tik.scope_ubuf)
        self.index_x = self.tik_instance.Scalar(dtype="int32")

    def one_dim_and_big_weight_tiling(self):
        """
        tiling process when input is one dim or big weight.

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.n_dim <= self.real_core_num:
            self.core_num = self.n_dim
        else:
            self.core_num = self.real_core_num
        self.loop_time = math.ceil(self.n_dim/self.core_num)
        self.align_repeat_size = (self.ub_size_float//64)*64
        self.move_out_time = math.ceil(self.c_dim/self.align_repeat_size)
        self.single_max_repeat = self.align_repeat_size//64
        self.tail_repeat = self.dup_repeat % self.single_max_repeat
        self.tail_mask = self.c_dim % 64
        self.max_out_burst = self.single_max_repeat*8
        self.last_out_burst = math.ceil(self.c_dim/8) - \
            (self.move_out_time - 1)*self.max_out_burst
        if self.move_out_time == 1:
            self.tail_repeat = math.ceil(self.c_dim/64)
            self.last_out_burst = math.ceil(self.c_dim/8)
            self.single_max_repeat = self.tail_repeat
            self.max_out_burst = self.last_out_burst
        self.offet = self.max_out_burst*8
        self.data_weight_size = self.c_dim
        self.dup_ub_size = math.ceil(self.max_out_burst*8/64)*64

    def one_dim_compute(self):
        """
        calculate process when input is 1D.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.one_dim_and_big_weight_tiling()
        self.init_ub()
        self.tik_instance.data_move(self.y_grad_ub, self.data_y_grad,
                                    1, 1, 1, 0, 0)
        self.tik_instance.data_move(self.total_weight_ub,
                                    self.data_total_weight, 1, 1, 1, 0, 0)
        self.tik_instance.data_move(self.target_ub, self.data_target,
                                    1, 1, 1, 0, 0)
        self.index_x.set_as(self.target_ub(0))
        self.tik_instance.data_move(self.weight_ub,
                                    self.data_weight[self.index_x],
                                    1, 1, 1, 0, 0)
        self.tik_instance.vmul(1, self.weight_ub, self.weight_ub,
                               self.y_grad_ub, 1, 1, 1, 1, 1, 0, 0)
        self.tik_instance.vmuls(1, self.weight_ub, self.weight_ub,
                                NEGATIVE, 1, 1, 1, 0, 0)
        if self.move_out_time > 1:
            self.tik_instance.vector_dup(MASK64, self.dup_ub, 0,
                                         self.single_max_repeat, 1, NUM_EIGHT)
        else:
            self.tik_instance.vector_dup(MASK64, self.dup_ub, 0,
                                         self.tail_repeat, 1, NUM_EIGHT)
        with self.tik_instance.for_range(0, self.move_out_time) as cycle:
            with self.tik_instance.if_scope(cycle < self.move_out_time - 1):
                self.tik_instance.data_move(self.output[cycle*self.offet],
                                            self.dup_ub,
                                            0, 1, self.max_out_burst, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.output[cycle*self.offet],
                                            self.dup_ub,
                                            0, 1, self.last_out_burst, 0, 0)
                self.tik_instance.data_move(self.output[self.index_x],
                                            self.weight_ub, 0, 1, 1, 0, 0)

    def vector_dup_process(self, dup_up, repeat):
        """
        vector dup process.

        Parameters
        ----------

        Returns
        -------
        None
        """
        max_repeat_num = MAX_REPEAT*NUM_SIXTYFOUR
        max_repeat_loop = math.ceil(repeat/MAX_REPEAT)
        last_repeat = repeat % MAX_REPEAT
        if last_repeat == 0:
            last_repeat = MAX_REPEAT
        for i in range(max_repeat_loop-1):
            self.tik_instance.vector_dup(MASK64, dup_up[i*max_repeat_num],
                                         0, MAX_REPEAT, 1, NUM_EIGHT)

        self.tik_instance.vector_dup(
            MASK64, dup_up[(max_repeat_loop-1)*max_repeat_num], 0,
            last_repeat, 1, NUM_EIGHT)

    def insert_valid_value(self, dst, src, index, left_value, right_value):
        """
        insert valid value.

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.index_x >= left_value):
            with self.tik_instance.if_scope(self.index_x < right_value):
                dst[index - left_value].set_as(src[0])

    def tail_block_refactor(self, dst, src, valid_value, index, burst,
                            start, offset):
        """
        refactor tail block.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.tik_instance.data_move(
            dst[offset], src, 0, 1, burst - 1, 0, 0)
        temp_out_ub = self.tik_instance.Tensor(
            self.x_dtype, [NUM_EIGHT],
            name="temp_out_ub", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(8, temp_out_ub, 0, 1, 1, 0)
        with self.tik_instance.if_scope(index > self.c_dim - 8):
            temp_out_ub[index - self.c_dim + 8].set_as(valid_value[0])
            self.tik_instance.data_move(dst[start + self.c_dim - 8],
                                        temp_out_ub, 0, 1, 1, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(dst[start + self.c_dim - 8],
                                        temp_out_ub, 0, 1, 1, 0, 0)
            temp_out_ub[0].set_as(valid_value[0])
            self.tik_instance.data_move(dst[start + index],
                                        temp_out_ub, 0, 1, 1, 0, 0)

    def two_dim_with_big_weight_compute(self):
        """
        calculate process when x is 2D and the shape of weight
        is big.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.one_dim_and_big_weight_tiling()
        with self.tik_instance.for_range(0, self.core_num,
                                         block_num=self.core_num) as cycle:
            self.init_one_dim_and_big_weight_ub()
            if self.move_out_time > 1:
                self.vector_dup_process(self.dup_ub,
                                        self.single_max_repeat)
            else:
                self.vector_dup_process(self.dup_ub, self.tail_repeat)
            with self.tik_instance.for_range(0, self.loop_time) as loop:
                line_num = cycle + loop*self.core_num
                with self.tik_instance.if_scope(line_num < self.n_dim):
                    self.tik_instance.data_move(self.y_grad_ub, self.data_x,
                                                0, 1, 1, 0, 0)
                    if self.reduction == "none":
                        self.tik_instance.data_move(self.y_grad_ub,
                                                    self.data_y_grad[line_num],
                                                    0, 1, 1, 0, 0)
                    else:
                        self.tik_instance.data_move(self.y_grad_ub,
                                                    self.data_y_grad,
                                                    0, 1, 1, 0, 0)
                    self.tik_instance.data_move(self.target_ub,
                                                self.data_target[line_num],
                                                0, 1, 1, 0, 0)
                    self.index_x.set_as(self.target_ub[0])
                    self.tik_instance.data_move(self.weight_ub,
                                                self.data_weight[self.index_x],
                                                0, 1, 1, 0, 0)
                    self.tik_instance.data_move(self.total_weight_ub,
                                                self.data_total_weight,
                                                0, 1, 1, 0, 0)
                    if self.reduction == "mean":
                        if tbe_platform.cce_conf.api_check_support(
                                "te.lang.cce.vdiv", "float32"):
                            self.tik_instance.vdiv(
                                1, self.weight_ub, self.weight_ub,
                                self.total_weight_ub, 1, 1, 1, 1,
                                NUM_EIGHT, NUM_EIGHT, NUM_EIGHT)
                        else:
                            self.tik_instance.vrec(1, self.weight_ub,
                                                   self.weight_ub,
                                                   1, 1, 1, 8, 8)
                            self.tik_instance.vmul(
                                1, self.weight_ub, self.weight_ub,
                                self.total_weight_ub, 1, 1, 1, 1,
                                NUM_EIGHT, NUM_EIGHT, NUM_EIGHT)
                    self.tik_instance.vmuls(1, self.weight_ub, self.weight_ub,
                                            NEGATIVE, 1, 1, 1,
                                            NUM_EIGHT, NUM_EIGHT)
                    self.tik_instance.vmul(1, self.weight_ub, self.weight_ub,
                                           self.y_grad_ub, 1, 1, 1, 1,
                                           NUM_EIGHT, NUM_EIGHT, NUM_EIGHT)
                    with self.tik_instance.for_range(
                            0, self.move_out_time) as time:
                        out_put_offset = line_num*self.c_dim + time*self.offet
                        with self.tik_instance.if_scope(
                                time < self.move_out_time - 1):
                            self.tik_instance.data_move(
                                self.output[out_put_offset],
                                self.dup_ub, 0, 1, self.max_out_burst, 0, 0)
                        with self.tik_instance.else_scope():
                            self.tail_block_refactor(
                                self.output, self.dup_ub, self.weight_ub,
                                self.index_x, self.last_out_burst,
                                line_num*self.c_dim, out_put_offset)

    def weight_no_32b_tiling(self):
        """
        calculate weight lower 32b tiling

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.core_num = 1
        self. max_line = self.n_dim
        self.lower_line = self.n_dim
        self.loop_time = 1
        self.max_total_num = self.n_dim*self.c_dim
        self.lower_total_num = self.max_total_num
        self.redundant_line = 0

    def normal_two_dim_tiling(self):
        """
        tiling of normal two dim.

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.n_dim <= self.real_core_num:
            self.core_num = self.n_dim
        else:
            self.core_num = self.real_core_num
        self.max_line = self.max_move_line
        self.lower_line = self.n_dim % self.max_line
        self.loop_time = math.ceil(self.n_dim/(self.max_line*self.core_num))
        self.fake_core = math.ceil(self.n_dim/self.max_line)
        self.redundant_line = self.n_dim % self.core_num

        if self.loop_time == 1:
            if self.redundant_line == 0:
                self.max_line = self.n_dim//self.core_num
            else:
                self.max_line = self.n_dim//self.core_num + 1
            self.lower_line = self.n_dim//self.core_num
        if self.lower_line == 0:
            self.lower_line = self.max_line
        self.max_total_num = self.max_line*self.c_dim
        self.lower_total_num = self.lower_line*self.c_dim
        if self.lower_total_num < 8:
            self.weight_no_32b_tiling()
        self.tail_mask = self.max_total_num % 64

        #compute ub size
        self.dup_ub_size = math.ceil(self.max_total_num/64)*64
        self.target_ub_size = math.ceil(self.max_line/64)*64
        self.total_weight_ub_size = NUM_SIXTYFOUR
        self.weight_ub_size = math.ceil(self.c_dim/64)*64
        self.refactor_weight_ub_size = self.target_ub_size

        #compute burst and repeat time
        self.weight_burst = math.ceil(self.c_dim/8)
        self.target_burst = math.ceil(self.max_line/8)
        self.lower_target_burst = math.ceil(self.lower_line/8)
        self.max_vmul_repeat = math.ceil(self.max_line/64)
        self.lower_vmul_repeat = math.ceil(self.lower_line/64)
        self.last_target_burst = math.ceil(self.lower_line/8)
        self.last_vmul_repeat = math.ceil(self.lower_line/64)
        self.core_dup_repeat = math.ceil(self.max_total_num/64)
        self.last_dup_repeat = math.ceil(self.lower_total_num/64)
        self.max_out_burst = math.ceil(self.max_total_num / 8)
        self.last_out_burst = math.ceil(self.lower_total_num / 8)

        if self.reduction == "none":
            self.y_grad_ub_size = self.target_ub_size
        else:
            self.y_grad_ub_size = NUM_SIXTYFOUR

    def compute_valid_value(self, dst, src, index, offset, repeat):
        """
        compute valid value.

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.reduction != "none":
            scalar = self.tik_instance.Scalar(dtype="float32")
            scalar.set_as(src[0])
            self.tik_instance.vmuls(MASK64, dst[index*offset],
                                    dst[index*offset], scalar,
                                    repeat, 1, 1, 8, 8)
        else:
            self.tik_instance.vmul(
                MASK64, dst[index*offset], dst[index*offset],
                src[index*offset], repeat, 1, 1, 1,
                NUM_EIGHT, NUM_EIGHT, NUM_EIGHT)
        self.tik_instance.vmuls(MASK64, dst[index*offset], dst[index*offset],
                                NEGATIVE, repeat, 1, 1, 8, 8)
        if self.reduction == "mean":
            if tbe_platform.cce_conf.api_check_support(
                    "te.lang.cce.vdiv", "float32"):
                self.tik_instance.vdiv(MASK64, dst[index*offset],
                                       dst[index*offset], self.total_weight_ub,
                                       repeat, 1, 1, 1, 8, 8, 0)
            else:
                self.tik_instance.vrec(MASK64, dst[index*offset],
                                       dst[index*offset], repeat, 1, 1, 8, 8)
                self.tik_instance.vmul(MASK64, dst[index*offset],
                                       dst[index*offset], self.total_weight_ub,
                                       repeat, 1, 1, 1, 8, 8, 0)

    def select_valid_value(self, line_num, line_size, dst, src, target,
                           dst_need_index=True, src_need_index=True):
        """
        select valid value with .

        Parameters
        ----------

        Returns
        -------
        None
        """
        if line_num >= 8:
            vars_num = 8
            loop_num = line_num//8
            last_line = line_num % 8
        else:
            vars_num = line_num
            loop_num = 0
            last_line = line_num
        names = locals()
        for i in range(0, vars_num):
            names["index_x" + str(i)] = self.tik_instance.Scalar(dtype="int32")
        with self.tik_instance.for_range(0, loop_num) as time:
            offset_set = 8 * time
            for i in range(0, vars_num):
                names["index_x" + str(i)].set_as(target[offset_set + i])
            for i in range(0, vars_num):
                dst_offset = (offset_set+i)*line_size +\
                             names["index_x" + str(i)]
                src_offset = names["index_x" + str(i)]
                if not dst_need_index:
                    dst_offset = (offset_set+i)*line_size
                if not src_need_index:
                    src_offset = offset_set+i
                dst[dst_offset].set_as(src[src_offset])
        for i in range(0, last_line):
            names["index_x" + str(i)].set_as(target[loop_num*8 + i])
        for i in range(0, last_line):
            dst_offset = (loop_num*8+i)*line_size + \
                             names["index_x" + str(i)]
            src_offset = names["index_x" + str(i)]
            if not dst_need_index:
                dst_offset = (loop_num*8+i)*line_size
            if not src_need_index:
                src_offset = loop_num*8+i
            dst[dst_offset].set_as(src[src_offset])

    def _normal_two_tim_process(self, line_num, core_offset,
                                repeat, burst, output_burst):
        if self.reduction == "none":
            self.tik_instance.data_move(
                self.y_grad_ub, self.data_y_grad[core_offset],
                0, 1, burst, 0, 0)
        self.tik_instance.data_move(
            self.target_ub, self.data_target[core_offset],
            0, 1, burst, 0, 0)
        self.vector_dup_process(self.dup_ub, self.core_dup_repeat)
        if self.reduction == "mean":
            total_weight = self.tik_instance.Scalar(dtype="float32")
            total_weight.set_as(self.total_weight_ub[0])
            self.tik_instance.vector_dup(MASK64, self.total_weight_ub,
                                         total_weight, 1, 1, 8)
        if self.reduction == "none" and self.c_dim != 1:
            self.select_valid_value(line_num, 1, self.refactor_weight_ub,
                                    self.weight_ub, self.target_ub,
                                    dst_need_index=False)
        else:
            self.select_valid_value(line_num, self.c_dim, self.dup_ub,
                                    self.weight_ub, self.target_ub)

        vmul_repeat_times = math.ceil(repeat/MAX_REPEAT)
        max_repeat_num = MAX_REPEAT*NUM_SIXTYFOUR
        last_vmul_offset = max_repeat_num
        last_vmul_repeat = repeat % MAX_REPEAT
        if last_vmul_repeat == 0:
            last_vmul_repeat = MAX_REPEAT

        if self.reduction == "none" and self.c_dim != 1:
            compute_ub = self.refactor_weight_ub
        else:
            compute_ub = self.dup_ub
        for i in range(0, vmul_repeat_times-1):
            self.compute_valid_value(compute_ub, self.y_grad_ub, i,
                                     max_repeat_num, MAX_REPEAT)
        self.compute_valid_value(
            compute_ub, self.y_grad_ub, vmul_repeat_times - 1,
            last_vmul_offset, last_vmul_repeat)
        if self.reduction == "none" and self.c_dim != 1:
            self.select_valid_value(line_num, self.c_dim, self.dup_ub,
                                    self.refactor_weight_ub, self.target_ub,
                                    dst_need_index=True, src_need_index=False)

        if line_num*self.c_dim % 8 == 0 or self.core_num == 1:
            self.tik_instance.data_move(self.output[core_offset*self.c_dim],
                                        self.dup_ub, 0, 1, output_burst, 8, 8)
        else:
            temp_out_ub = self.tik_instance.Tensor(
                self.x_dtype, [NUM_EIGHT],
                name="temp_out_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(
                self.output[core_offset*self.c_dim], self.dup_ub, 0, 1,
                output_burst - 1, 8, 8)
            for i in range(0, 8):
                temp_out_ub[i].set_as(self.dup_ub[line_num*self.c_dim - 8 + i])
            self.tik_instance.data_move(
                self.output[(core_offset + line_num)*self.c_dim - 8],
                temp_out_ub, 0, 1, 1, 8, 8)

    def normal_two_dim_compute(self):
        """
        calculate process of normal two dim.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.normal_two_dim_tiling()
        with self.tik_instance.for_range(0, self.core_num,
                                         block_num=self.core_num) as cycle:
            self.init_normal_two_dim_ub()
            core_offset = cycle*self.max_line
            lower_core_offset = cycle*self.lower_line+self.redundant_line
            self.tik_instance.data_move(self.y_grad_ub, self.data_x,
                                        0, 1, 1, 0, 0)
            if self.reduction != "none":
                self.tik_instance.data_move(
                    self.total_weight_ub, self.data_total_weight,
                    0, 1, 1, 0, 0)
                self.tik_instance.data_move(
                    self.y_grad_ub, self.data_y_grad, 0, 1, 1, 0, 0)

            self.tik_instance.data_move(self.weight_ub, self.data_weight,
                                        0, 1, self.weight_burst, 0, 0)
            with self.tik_instance.for_range(0, self.loop_time) as loop:
                loop_offset = loop*self.max_line*self.core_num
                if self.reduction == "none" and self.c_dim != 1:
                    compute_max_repeat = self.max_vmul_repeat
                    compute_last_repeat = self.last_vmul_repeat
                else:
                    compute_max_repeat = self.core_dup_repeat
                    compute_last_repeat = self.last_dup_repeat
                with self.tik_instance.if_scope(self.loop_time == 1):
                    with self.tik_instance.if_scope(
                            cycle < self.redundant_line):
                        self._normal_two_tim_process(
                            self.max_line, core_offset, compute_max_repeat,
                            self.target_burst, self.max_out_burst)
                    with self.tik_instance.else_scope():
                        self._normal_two_tim_process(
                            self.lower_line, lower_core_offset,
                            compute_last_repeat, self.lower_target_burst,
                            self.last_out_burst)
                with self.tik_instance.if_scope(self.loop_time > 1):
                    with self.tik_instance.if_scope(
                            loop*self.core_num + cycle < self.fake_core - 1):
                        self._normal_two_tim_process(
                            self.max_line, loop_offset + core_offset,
                            compute_max_repeat, self.target_burst,
                            self.max_out_burst)
                    with self.tik_instance.if_scope(
                            loop*self.core_num + cycle == self.fake_core - 1):
                        self._normal_two_tim_process(
                            self.lower_line, loop_offset + core_offset,
                            compute_last_repeat, self.last_target_burst,
                            self.last_out_burst)

    def nll_loss_compute_start(self):
        """
        Different calculation methods

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.x_dim == 1:
            self.one_dim_compute()
        if self.big_weight:
            self.two_dim_with_big_weight_compute()
        elif self.x_dim == DIM2:
            self.normal_two_dim_compute()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.data_x,
                                           self.data_y_grad,
                                           self.data_target,
                                           self.data_weight,
                                           self.data_total_weight],
                                   outputs=[self.output])
        return self.tik_instance


@util.check_input_type(dict, dict, dict, dict, dict, dict, str, str)
def nll_loss_grad(x, y_grad, target, weight, total_weight, x_grad,
                  reduction="mean", kernel_name="nll_loss_grad"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input, the length of shape should be two or one.
    y_grad : dict
        shape and dtype of input, the length of shape must be one.
    target : dict
        shape and dtype of input, the length of shape only support one.
    total_weight : dict
        shape and dtype of input, it is a scalar.
    weight : dict or None
        the length of shape only support one when weight is dict.
    x_grad: dict
        It’s a tensor with shape(minibatch, ) when reduction == ‘none’ and
        the input is 2D. Otherwise, the output is a scalar.
    reduction: str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "nll_loss_grad"

    Returns
    -------
    None
    """
    _shape_and_dtype_check(x, y_grad, target, weight, total_weight,
                           reduction, kernel_name)
    nll_loss_function = NllLossGradCompute(x, y_grad, target, weight,
                                              reduction, kernel_name)
    return nll_loss_function.nll_loss_compute_start()
