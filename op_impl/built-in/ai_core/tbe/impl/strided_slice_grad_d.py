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
strided_slice_grad_d
"""
from impl import pad_d
from topi.cce import util
from te import platform as tbe_platform
from te import tik
from impl.strided_slice_d import _init_parameter
from te.utils.op_utils import *
# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
BLOCK_SIZE = 32


# pylint: disable=invalid-name, too-many-instance-attributes
# pylint: disable=too-many-arguments, useless-object-inheritance
# pylint: disable=too-many-locals, too-many-statements
# pylint: disable=attribute-defined-outside-init, unused-argument
# pylint: disable=attribute-defined-outside-init, chained-comparison
class StridedSliceGradLastDimCompute(object):
    """
    the compute for stridedslicegrad in last dim situation
    """
    def __init__(self, shape, begin, size, dtype, kernel_name):
        self.dim_product = 1
        self.input_dim_last = 1
        self.output_dim_last = 1
        self.begin_last = 1
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.ele_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        # align size for product dim, to make sure out data is 32B align
        self.product_dim_align_size = BLOCK_SIZE // self.ele_size

        # check only last dim to be sliced
        for i, (shape_i, begin_i, size_i) in \
                enumerate(zip(reversed(shape),
                              reversed(begin), reversed(size))):
            if i != 0:
                if shape_i != size_i:
                    self.check_result = False
                    return

                self.dim_product *= shape_i
            else:
                if begin_i < 0:
                    begin_i += shape_i
                self.input_dim_last = shape_i
                self.begin_last = begin_i
                self.output_dim_last = size_i

        # for moving data continuously, only small last dim is allowed
        # last dim data size <= 340B
        if self.input_dim_last * self.ele_size > 340:
            self.check_result = False
            return

        # for dividing cores easily, only big product dim is allowed
        # product dim >= aicore_num * 32 // ele_size
        aicore_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        if self.dim_product < self.product_dim_align_size * aicore_num:
            self.check_result = False
            return

        self.check_result = True

    def check(self):
        """
        return check_result
        """
        return self.check_result

    def check_perf(self):
        """
        return if can enter performance template
        """
        is_satisfied_perf = (self.output_dim_last % self.product_dim_align_size == 0) and \
                            (self.input_dim_last % self.product_dim_align_size == 0) and \
                            (self.begin_last % self.product_dim_align_size == 0)
        return is_satisfied_perf

    def _get_block_tiling(self, product, core, block_idx):
        task_size = self.product_dim_align_size
        if product % task_size == 0:
            tasks = product // task_size
        else:
            tasks = product // task_size + 1

        begin = self.tik_instance.Scalar(dtype="int64")
        size = self.tik_instance.Scalar(dtype="int64")
        if tasks % core == 0:
            begin.set_as(block_idx * (tasks // core) * task_size)
            size.set_as((tasks // core) * task_size)
        else:
            pack1 = tasks // core + 1
            pack2 = tasks // core
            with self.tik_instance.if_scope(block_idx >= tasks % core):
                begin.set_as(pack1 * block_idx * task_size - (block_idx - tasks % core) * task_size)
                size.set_as(pack2 * task_size)
            with self.tik_instance.else_scope():
                begin.set_as(pack1 * block_idx * task_size)
                size.set_as(pack1 * task_size)

        with self.tik_instance.if_scope(block_idx == (core - 1)):
            size.set_as(product - begin)
        return begin, size

    def strided_slice_grad(self):
        """
        schedule for strided_slice_grad
        """
        if not self.check_result:
            raise RuntimeError("conditions of SliceLastDimCompute are not fulfilled")

        tik_instance = tik.Tik()
        self.tik_instance = tik_instance
        aicore_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        ub_size = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE)

        pad_value = tik_instance.Scalar(dtype=self.dtype, init_value=0)
        x = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.input_dim_last),
                                name="x", scope=tik.scope_gm)
        y = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.output_dim_last),
                                name="y", scope=tik.scope_gm)

        with tik_instance.for_range(0, aicore_num,
                                    block_num=aicore_num) as block_idx:
            dim_product_begin, dim_product_size = \
                self._get_block_tiling(self.dim_product, aicore_num, block_idx)
            max_dim_product = ub_size // self.ele_size \
                              // (self.input_dim_last + self.output_dim_last) \
                              // self.product_dim_align_size * self.product_dim_align_size
            loops = tik_instance.Scalar(dtype="int64")
            loops.set_as(dim_product_size // max_dim_product)
            with tik_instance.if_scope(dim_product_size % max_dim_product == 0):
                loops.set_as(loops - 1)

            with tik_instance.for_range(0, loops) as i:
                dim_product_begin_in_loop = i * max_dim_product
                dim_product_size_in_loop = max_dim_product

                x_ub = tik_instance.Tensor(self.dtype,
                                           (max_dim_product,
                                            self.input_dim_last), \
                                           name="x_ub", scope=tik.scope_ubuf)
                y_ub = tik_instance.Tensor(self.dtype,
                                           (max_dim_product,
                                            self.output_dim_last), \
                                           name="y_ub", scope=tik.scope_ubuf)

                output_size_in_loop = dim_product_size_in_loop \
                                      * self.output_dim_last * self.ele_size
                burst_length_out = output_size_in_loop // BLOCK_SIZE
                tik_instance.data_move(y_ub,
                                       y[(dim_product_begin +
                                          dim_product_begin_in_loop)
                                         * self.output_dim_last],
                                       0, 1, burst_length_out, 0, 0)

                with tik_instance.for_range(0, dim_product_size_in_loop) as j:
                    idx_x = j * self.input_dim_last
                    idx_y = j * self.output_dim_last
                    for k in range(self.input_dim_last):
                        max_num = self.begin_last + self.output_dim_last
                        if (k >= self.begin_last) and (k < max_num):
                            x_ub[idx_x + k] = y_ub[idx_y + k - self.begin_last]
                        else:
                            x_ub[idx_x + k] = pad_value

                input_size_in_loop = dim_product_size_in_loop \
                                     * self.input_dim_last * self.ele_size
                burst_length = input_size_in_loop // BLOCK_SIZE
                tik_instance.data_move(x[(dim_product_begin +
                                          dim_product_begin_in_loop)
                                         * self.input_dim_last],
                                       x_ub,
                                       0, 1, burst_length, 0, 0)

            # last loop
            i = loops
            dim_product_begin_in_loop = i * max_dim_product
            dim_product_size_in_loop = dim_product_size - dim_product_begin_in_loop

            x_ub = tik_instance.Tensor(self.dtype, (max_dim_product, self.input_dim_last), \
                                       name="x_ub", scope=tik.scope_ubuf)
            y_ub = tik_instance.Tensor(self.dtype, (max_dim_product, self.output_dim_last), \
                                       name="y_ub", scope=tik.scope_ubuf)

            output_size_in_loop = dim_product_size_in_loop * self.output_dim_last * self.ele_size
            burst_length_out = tik_instance.Scalar(dtype="int64")
            burst_length_out.set_as(output_size_in_loop // BLOCK_SIZE)
            with tik_instance.if_scope(output_size_in_loop % BLOCK_SIZE != 0):
                burst_length_out.set_as(burst_length_out + 1)
            tik_instance.data_move(y_ub,
                                   y[(dim_product_begin +
                                      dim_product_begin_in_loop)
                                     * self.output_dim_last],
                                   0, 1, burst_length_out, 0, 0)

            with tik_instance.for_range(0, dim_product_size_in_loop) as j:
                idx_x = j * self.input_dim_last
                idx_y = j * self.output_dim_last
                for k in range(self.input_dim_last):
                    max_num = (self.begin_last + self.output_dim_last)
                    if (k >= self.begin_last) and (k < max_num):
                        x_ub[idx_x + k] = y_ub[idx_y + k - self.begin_last]
                    else:
                        x_ub[idx_x + k] = pad_value

            input_size_in_loop = dim_product_size_in_loop * self.input_dim_last * self.ele_size
            burst_length = tik_instance.Scalar(dtype="int64")
            burst_length.set_as(input_size_in_loop // BLOCK_SIZE)
            with tik_instance.if_scope(input_size_in_loop % BLOCK_SIZE != 0):
                burst_length.set_as(burst_length + 1)
            tik_instance.data_move(x[(dim_product_begin +
                                      dim_product_begin_in_loop)
                                     * self.input_dim_last],
                                   x_ub,
                                   0, 1, burst_length, 0, 0)

        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[y], outputs=[x])

    def strided_slice_grad_perf(self):
        """
        high performance schedule for strided_slice_grad
        self.input_dim_last, self.input_dim_last and self.begin_last should divided by block size
        """
        if not self.check_result:
            raise RuntimeError("conditions of SliceLastDimCompute are not fullfilled")

        tik_instance = tik.Tik()
        self.tik_instance = tik_instance
        aicore_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)

        self.pad_value = tik_instance.Scalar(dtype=self.dtype, init_value=0)
        self.x = tik_instance.Tensor(self.dtype,
                           (self.dim_product, self.input_dim_last),
                           name="x", scope=tik.scope_gm)
        self.y = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.output_dim_last),
                                name="y", scope=tik.scope_gm)

        self.max_dim_product = ub_size // self.ele_size \
                            // self.input_dim_last \
                            // self.product_dim_align_size * self.product_dim_align_size

        product_per_core = self.dim_product // aicore_num
        is_same_core = 0 if self.dim_product % aicore_num == 0 else 1
        product_last_core = product_per_core if is_same_core == 0 else self.dim_product % aicore_num
        aicore_num += is_same_core

        self.x_ub = self.tik_instance.Tensor(self.dtype, (self.max_dim_product, self.input_dim_last), \
                    name="x_ub", scope=tik.scope_ubuf)
        self.vector_mask = 256 // self.ele_size

        with self.tik_instance.for_range(0, aicore_num, block_num=aicore_num) as block_idx:
            with self.tik_instance.if_scope(block_idx != aicore_num - 1):
                self.compute_each_core(block_idx * product_per_core, product_per_core)
            with self.tik_instance.else_scope():
                self.compute_each_core(block_idx * product_per_core, product_last_core)

        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.y], outputs=[self.x])
        return tik_instance

    def compute_each_core(self, move_product_offset, move_product_len):
        product_repeat = move_product_len // self.max_dim_product
        if product_repeat > 0:
            with self.tik_instance.for_range(0, product_repeat) as product_index:
                self.compute_each_loop(move_product_offset + product_index * \
                self.max_dim_product, self.max_dim_product)

        product_tail = move_product_len % self.max_dim_product
        if product_tail > 0:
            self.compute_each_loop(move_product_offset + product_repeat * self.max_dim_product, product_tail)

    def compute_each_loop(self, move_product_offset, move_product_len):
        # vector dup 0 to x_ub
        repeat_loop = (move_product_len * self.input_dim_last) // (self.vector_mask * 255)
        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as repeat_index:
                self.tik_instance.vector_dup(self.vector_mask, self.x_ub[repeat_index * self.vector_mask * 255], \
                self.pad_value, 255, 1, 8)
        repeat_tail = (move_product_len * self.input_dim_last) % (self.vector_mask * 255) // self.vector_mask
        if repeat_tail > 0:
            self.tik_instance.vector_dup(self.vector_mask, self.x_ub[repeat_loop * self.vector_mask * 255], \
            self.pad_value, repeat_tail, 1, 8)
        mask_tail = (move_product_len * self.input_dim_last) % self.vector_mask
        if mask_tail > 0:
            self.tik_instance.vector_dup(mask_tail, self.x_ub[repeat_loop * self.vector_mask * 255 + \
            repeat_tail * self.vector_mask], self.pad_value, 1, 1, 8)

        # move y to x_ub and pad
        mv_stride = (self.input_dim_last - self.output_dim_last) // self.product_dim_align_size
        move_loop = move_product_len // 4095
        if move_loop > 0:
            with self.tik_instance.for_range(0, move_loop) as move_index:
                self.tik_instance.data_move(self.x_ub[self.begin_last + move_index * 4095 * self.input_dim_last], \
                self.y[(move_product_offset + move_index * 4095) * self.output_dim_last], 0, 4095, \
                self.output_dim_last // self.product_dim_align_size, 0, mv_stride)
        move_tail = move_product_len % 4095
        if move_tail > 0:
            self.tik_instance.data_move(self.x_ub[self.begin_last + move_loop * 4095 * self.input_dim_last], \
            self.y[(move_product_offset + move_loop * 4095) * self.output_dim_last], 0, \
            move_tail, self.output_dim_last // self.product_dim_align_size, 0, mv_stride)

        # move x_ub to x
        burst_len = (move_product_len * self.input_dim_last) // self.product_dim_align_size
        if burst_len > 65535:
            nburst = burst_len // 65535
            burst_len = 65535
        else:
            nburst = 1
        self.tik_instance.data_move(self.x[move_product_offset * self.input_dim_last], \
            self.x_ub, 0, nburst, burst_len, 0, 0)


def _update_begin_end(input_shape, begin, end, begin_mask, end_mask):
    """ Calculate the value of padding by input parameters.

    Parameters
    ----------
    input_shape: list or tuple.
        shape of input.
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    begin_mask: int
        a bit mask where a bit i being 1 means to ignore the begin value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.

    Returns
    -------
    begin_shape: list.
        shape of 'begin' after mask handle
    end_shape: list.
        shape of 'end' after mask handle
    """
    begin_shape = list(begin)
    end_shape = list(end)

    if end_shape[-1] > input_shape[-1]:
        end_shape[-1] = input_shape[-1]

    # If the ith bit of begin_mask is set, begin[i] is ignored,
    # and the fullest possible range in that dimension is used instead.
    # end_mask works analogously, except with the end range.
    for i, _ in enumerate(zip(input_shape, begin_shape, end_shape)):
        # process begin_mask
        if (begin_mask & 2**i) == 2**i:
            begin_shape[i] = 0
        # process end_mask
        if (end_mask & 2**i) == 2**i:
            end_shape[i] = input_shape[i]

    return begin_shape, end_shape


def _get_paddings(shape_x, begin_shape, end_shape):
    """ Calculate the value of padding by input parameters.

    Parameters
    ----------
    shape_x: list or tuple.
        shape of output.
    begin_shape: list or tuple.
        represents the index of the first value to select.
    end_shape: list or tuple.
        represents the index of the last value to select.

    Returns
    -------
    paddings: list.
        indicates how many zeros to add after the contents of `shape_dy` in every dimension
    """
    paddings = []
    for begin_i, shape_x_i, end_i in zip(begin_shape, shape_x, end_shape):
        if begin_i < 0:
            begin_i += shape_x_i
        if end_i < 0:
            end_i += shape_x_i
        paddings.append([begin_i, shape_x_i - end_i])

    return paddings


def _check_shape_parameter(shape_x, shape_dy, begin, end, strides):
    """ Check whether the input shape meets the requirements.

    Parameters
    ----------
    shape_x: list or tuple.
        shape of output.
    shape_dy: list or tuple.
        shape of input.
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.

    Returns
    -------
    None.
    """
    # length of 'shape_x, shape_dy, begin, end, strides' must be the same
    if not (len(end) == len(begin) and \
            len(shape_x) == len(begin) and \
            len(shape_x) == len(strides)):
        raise RuntimeError("shape length mismatch!")

    # value of begin must less equal to end, and it's range is (0, shape_x_i).
    for i, (shape_x_i, begin_i, end_i) in enumerate(zip(shape_x, begin, end)):
        if begin_i < 0:
            begin_i += shape_x_i
        if end_i < 0:
            end_i += shape_x_i
        if not ((begin_i >= 0) and (end_i <= shape_x_i)
                and (begin_i <= end_i)):
            raise RuntimeError("Bound Over: begin[%d]:%d, end[%d]:%d, shape_x[%d]:%d\n" \
                               % (i, begin[i], i, end[i], i, shape_x_i))

    # value of strides must all be 1.
    for i, strides_i in enumerate(strides):
        if strides_i != 1:
            raise RuntimeError("Value of the strides[%d]:%d must be 1!" % (i, strides_i))


def _check_mask(input_mask, is_shrink=False):
    """ Check whether the value of the input mask is 0.

    Parameters
    ----------
    input_mask: int.
        value of the input mask.

    Returns
    -------
    None.
    """
    if is_shrink:
        if input_mask != 0 and input_mask != 2:
            raise RuntimeError("shrink_axis_mask only support 0/2 currently")
    elif input_mask != 0:
        raise RuntimeError("ellipsis_mask,new_axis_mask"
                           " only support 0 currently")


def _check_is_not_aligned_shape(shape, begin, ellipsis_mask, shrink_axis_mask):
    """Check whether the shape of begin and shape is not equal,
       and masks are not 0

    Parameters
    ----------
    shape : list or tuple.
        shape of input
    begin: list or tuple.
        represents the index of the first value to select.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th
        position is actually an ellipsis.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th
        specification should shrink the dimensionality.
    Returns
    -------
    bool result
    """
    is_check_pass = False
    if len(shape) > len(begin) and len(begin) == 2 and \
            ellipsis_mask == 1 and shrink_axis_mask == 2:
        is_check_pass = True

    return is_check_pass, begin


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals

@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_LIST_INT,
                 REQUIRED_ATTR_LIST_INT, REQUIRED_ATTR_LIST_INT, OPTION_ATTR_INT, OPTION_ATTR_INT,
                 OPTION_ATTR_INT, OPTION_ATTR_INT, OPTION_ATTR_INT, KERNEL_NAME)
def strided_slice_grad_d(dy, output, shape, begin, end, strides, begin_mask=0,
                         end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                         kernel_name="strided_slice_grad_d"):

    """ Since `StridedSlice` cuts out pieces of its `input` which is size`shape_dy`, its gradient
    will have the same shape (which is passed here as `shape_x`). The gradient will be zero in any
    element that the slice does not select.

    Parameters
    ----------
    dy : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of out
    shape : list or tuple.
        shape of input
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification should shrink
        the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice_grad_d"

    Returns
    -------
    None.
    """
    shape_dy = dy.get("shape")
    ori_shape_dy = dy.get("ori_shape")
    format_dy = dy.get("format")
    ori_format_dy = dy.get("ori_format")
    input_dtype = dy.get("dtype").lower()

    check_dtype(input_dtype, ("float16", "float32", "int8", "uint8", "int32"), param_name="dy")
    check_shape(shape, param_name="shape")
    check_shape(shape_dy, param_name="dy")

    _check_mask(new_axis_mask)
    _check_mask(shrink_axis_mask, True)

    is_not_aligned, ori_begin = _check_is_not_aligned_shape(shape, begin,
                                                            ellipsis_mask,
                                                            shrink_axis_mask)

    shape = list(shape)
    begin = list(begin)
    end = list(end)
    strides = list(strides)
    begin_shape, end_shape, stride_shape = _init_parameter(shape, begin, end,
                                                           strides, begin_mask, end_mask,
                                                           ellipsis_mask, new_axis_mask,
                                                           shrink_axis_mask)

    _check_shape_parameter(shape, shape_dy, begin_shape, end_shape, stride_shape)

    last_dim_compute = StridedSliceGradLastDimCompute(shape,
                                                      begin_shape,
                                                      shape_dy,
                                                      input_dtype, kernel_name)

    if last_dim_compute.check():
        if last_dim_compute.check_perf() and ellipsis_mask == 0:
            last_dim_compute.strided_slice_grad_perf()
        else:
            last_dim_compute.strided_slice_grad()
    elif is_not_aligned:
        shape_dy = list(shape_dy)
        shape_dy += [1]
        paddings = [[0, 0]] * (len(shape) - 1) + \
                   [[ori_begin[1], shape[-1] - ori_begin[1] - 1]]
        dy_dict = {"shape": shape_dy, "ori_shape": ori_shape_dy,
                   "format": format_dy, "ori_format": ori_format_dy,
                   "dtype": input_dtype}
        pad_d(dy_dict, dy_dict, paddings, kernel_name)
    else:
        paddings = _get_paddings(shape, begin_shape, end_shape)

        # Call the pad operator due to gradient of 'StridedSlice' is the same as 'pad'
        # when the strides is 1.
        # pad.pad_cce(shape_dy, paddings, dtype, "CONSTANT", pad_value, kernel_name, need_build,
        #            need_print)
        dy_dict = {"shape": shape_dy, "ori_shape": ori_shape_dy, "format": format_dy, "ori_format": ori_format_dy, "dtype": input_dtype}
        pad_d(dy_dict, dy_dict, paddings, kernel_name)
