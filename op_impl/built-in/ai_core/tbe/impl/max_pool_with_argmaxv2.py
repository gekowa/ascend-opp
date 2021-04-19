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
max_pool_with_argmaxv2
"""
from te import tik
from topi.cce import util
from te import platform as tbe_platform
from te.platform.cce_conf import CceProductParams

# min value of fp16
MIN_VALUE_FP16 = -65504.0
# define dilation size
DILATION = 1
# parameters for vector instruct
MASK = 128
REPEAT_2 = 2
DSTSTRIDEM0 = 1
SRC0STRIDEM0 = 1
SRC1STRIDEM0 = 1
DSTSTRIDEM1 = 8
SRC0STRIDEM1 = 8
SRC1STRIDEM1 = 8
MAX_ALLOW_UB = 253952
DT_INT32 = 3
DT_INT64 = 9
# get available ub size
UB_SIZE = CceProductParams().getParams("Unified_Buffer")
UB_SIZE = MAX_ALLOW_UB if UB_SIZE > MAX_ALLOW_UB else UB_SIZE
# get available l1 size
L1_SIZE = CceProductParams().getParams("L1_Buffer")


def _ceil_div(value, factor):
    """
    caculate ceil value of div

    Parameters
    ----------
    value: dtype of int or float
        original value
    factor: dtype of int or float
        dividend value

    Returns
    -------
    value: dtype of int or float
    """
    if value % factor == 0:
        quotient = value // factor
    else:
        quotient = value // factor + 1

    return quotient


def _check_param(x, ksize, strides, padding, dtype, dilation, ceil_mode,
                 kernel_name):
    """
    check parameters, if one is invalid, then raise error

    Parameters
    ----------
    x: dict
        shape and datatype
    ksize: list or tuple
        the size of the window
    strides: list or tuple
        the stride of the sliding window
    padding: list or tuple
    kernel_name: str

    Returns
    -------
    None
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_tensor_shape_size(input_shape)
    util.check_dtype_rule(input_dtype, ("float16",))

    # the format of x must be NC1HWC0
    if len(input_shape) != 5:
        raise RuntimeError("invalid shape params, input feature map must be "
                           "5D format in kernel.")
    # get shape info of feature map in NC1HWC0 format
    input_h = input_shape[2]
    input_w = input_shape[3]
    c0_size = input_shape[4]

    if c0_size != 16:
        raise RuntimeError("invalid featur map shape params, "
                           "C0 must be equal to 16")

    if len(ksize) != 4:
        raise RuntimeError("Invalid ksize params, ksize dim must be 4.")

    if ksize[0] != 1 or ksize[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling "
                           "across width/height, and other ksize "
                           "dimension should be one")
    if len(strides) != 4:
        raise RuntimeError("Invalid strides params, strides dim must be 4.")

    if strides[0] != 1 or strides[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across "
                           "width/height, and other strides dimension "
                           "should be one")
    if len(padding) != 4:
        raise RuntimeError("Invalid padding params, padding dim must be 4.")

    if padding[0] != 1 or padding[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across "
                           "width/height, and other padding dimension "
                           "should be one")
    if len(strides) != 4:
        raise RuntimeError("Invalid strides params, strides dim must be 4.")

    if strides[0] != 1 or strides[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across "
                           "width/height, and other strides dimension "
                           "should be one")
    if len(dilation) != 4:
        raise RuntimeError("Invalid dilation params, dilation dim must be 4.")

    if dilation[0] != 1 or dilation[3] != 1:
        raise RuntimeError("MaxPoolWithArgmax only supports pooling across "
                           "width/height, and other dilation dimension "
                           "should be one")
    if ceil_mode != True and ceil_mode != False:
        raise RuntimeError("MaxPoolWithArgmax only supports ceil_mode across "
                           "True/False, and other string not support!")
    if dtype != DT_INT32 and dtype != DT_INT64:
        raise RuntimeError(
            "MaxPoolWithArgmax only supports output indices data type: "
            "int32, int64, and other data type not support!")

    if ksize[1] >= input_h or ksize[2] >= input_w:
        raise RuntimeError("can not support global pooling now")

    if ksize[1] * ksize[2] > 255:
        raise RuntimeError("invalid window params, kernel_h*kernel_w "
                           "should be <= 255")


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class MaxPoolWithargmaxPytorch():
    """
    Function: use to finish MaxPoolWithargmax main functions
    Modify : 2019-10-16
    """

    def __init__(self, x, ksize, strides, padding, dtype, dilation, ceil_mode,
                 kernel_name):
        """
        init MaxPoolWithargmax parameters

        Parameters
        ----------
        x: dict
            shape and datatype
        ksize: list or tuple
            The size of the window for each dimension of the input tensor.
        strides: list or tuple
            The stride of the sliding window of the input tensor.
        padding: list int
            The value of padding in all dimention, (1, padh, padw, 1).
        dilation: list int
            A parameter that controls the stride of elements in the window.
        ceil_mode: Bool
            If True, will use ceil instead of floor to compute the output
            shape
        dtype: int
            The output indices data type, only support int32 or int64.
        kernel_name: str
            The kernel's name
        Returns
        -------
        None
        """
        self.input_shape = x.get("shape")
        self.input_dtype = x.get("dtype").lower()

        self.tik_instance = tik.Tik()

        self.ksize = ksize
        self.strides = strides
        self.ceil_mode = ceil_mode
        self.dilation_h = dilation[1]
        self.dilation_w = dilation[2]
        self.pad_h = padding[1]
        self.pad_w = padding[2]
        self.dtype = dtype
        self.kernel_name = kernel_name

        self.batch_size = self.input_shape[0]
        self.c1_size = self.input_shape[1]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        self.c0_size = self.input_shape[4]
        self.input_wh = self.input_h * self.input_w

        self.kernel_h = self.ksize[1]
        self.kernel_w = self.ksize[2]
        self.stride_h = self.strides[1]
        self.stride_w = self.strides[2]
        self.nc1 = self.batch_size * self.c1_size
        # scalar for load3d
        self.scalar_source_h = self.tik_instance.Scalar(dtype="int64")
        self.scalar_source_w = self.tik_instance.Scalar(dtype="int64")

        # caculate pad and output size
        self.pad, self.output_h, self.output_w = \
            self._calc_out_size_and_pad()
        # output_shape
        self.fmap_h = self.output_h * self.output_w
        self.fmap_w = self.kernel_h * self.kernel_w
        self.fmap_h_num = _ceil_div(self.fmap_h, self.c0_size)
        self.output_wh = self.output_h * self.output_w
        mask_tmp = self.fmap_h_num * 16 - self.fmap_h
        self.mask_zero = 2 ** 16 - 2 ** (16 - mask_tmp)

        if self.input_dtype == "float16":
            self.pad_value = MIN_VALUE_FP16
        # famp is NC1HWC0 format
        fmap_gm_shape = (self.batch_size, self.c1_size, self.input_h,
                         self.input_w, self.c0_size)
        output_gm_shape = (self.batch_size, self.c1_size, self.output_h,
                           self.output_w, self.c0_size)
        output_mask_gm_shape = (self.batch_size, self.c1_size,
                                self.fmap_w, (self.fmap_h_num + 1),
                                self.c0_size)
        # input and output
        self.input_fmap_gm = self.tik_instance.Tensor(self.input_dtype,
                                                      fmap_gm_shape,
                                                      name="input_fmap_gm",
                                                      scope=tik.scope_gm)
        self.output_max_gm = self.tik_instance.Tensor(self.input_dtype,
                                                      output_gm_shape,
                                                      name="output_max_gm",
                                                      scope=tik.scope_gm)
        self.output_mask_gm = self.tik_instance.Tensor("uint16",
                                                       output_mask_gm_shape,
                                                       name="output_mask_gm",
                                                       scope=tik.scope_gm)

    # pylint: disable=too-many-locals, too-many-function-args
    def tik_instance_function(self, kernel_name):
        """
        tik_instance_function

        Parameters
        ----------
        kernel_name: str
            kernel_name

        Returns
        -------
        tik_instance
        """
        core_counts = tik.Dprofile().get_aicore_num()
        need_cut_h, need_cut_h_w, need_cut = self._check_if_need_cut_h_w()
        if need_cut_h or need_cut:
            cut_h_size, cut_stride, cut_h_num = \
                self._calc_cut_h_size_fun(need_cut)

            flag_cut_h = False
            out_size_cut_h = \
                (cut_h_size - self.kernel_h + self.stride_h) // self.stride_h
            fmap_cut_h = self.output_w * out_size_cut_h
            if (fmap_cut_h % 16) == 0:
                flag_cut_h = True
                nc1_cuth = self.nc1 * cut_h_num
            else:
                nc1_cuth = self.nc1

            if need_cut_h_w:
                cut_w_size, cut_w_stride, cut_w_num = \
                    self._calc_cut_w_size_fun()
                cut_w_tail = self.in_size_w + self.pad[0] - \
                             cut_w_stride * (cut_w_num - 1)
                if (cut_w_tail % 16) == 0 and cut_w_size > 0:
                    flag_cut_h = True
                    nc1_cuth = self.nc1 * cut_h_num
                else:
                    nc1_cuth = self.nc1
                    flag_cut_h = False

            nc1_cuth_size = nc1_cuth // core_counts + (
                1 if nc1_cuth % core_counts > 0 else 0)
            if (nc1_cuth % core_counts == 0) or (
                    nc1_cuth % nc1_cuth_size == 0):
                is_same_percore = 0
            else:
                is_same_percore = 1

            block_dim = nc1_cuth // nc1_cuth_size + (
                0 if nc1_cuth // core_counts == 0 else is_same_percore)
            with self.tik_instance.for_range(0, block_dim,
                                             block_num=block_dim) as block_index:
                with self.tik_instance.if_scope(block_index != block_dim - 1):
                    with self.tik_instance.for_range(0,
                                                     nc1_cuth_size) as nc1_cuth_index:
                        # size of ub is not enough, need cutH
                        if need_cut_h_w:
                            self._fun_need_cut_h_w(block_index, nc1_cuth_index,
                                                   cut_h_size,
                                                   cut_stride, cut_h_num,
                                                   nc1_cuth_size,
                                                   flag_cut_h)
                        else:
                            # come into this logical 2020/6/1
                            self._fun_only_cut_h(block_index, nc1_cuth_index,
                                                 cut_h_size, cut_stride,
                                                 cut_h_num,
                                                 nc1_cuth_size, flag_cut_h)

                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, nc1_cuth - (block_dim - 1) * nc1_cuth_size) as nc1_cuth_index:
                        # size of ub is not enough, need cutH
                        if need_cut_h_w:
                            self._fun_need_cut_h_w(block_index, nc1_cuth_index,
                                                   cut_h_size,
                                                   cut_stride, cut_h_num,
                                                   nc1_cuth_size,
                                                   flag_cut_h)
                        else:
                            self._fun_only_cut_h(
                                block_index, nc1_cuth_index, cut_h_size,
                                cut_stride, cut_h_num, nc1_cuth_size,
                                flag_cut_h)
        # no need cut
        else:
            nc1_size = self.nc1 // core_counts + (
                1 if self.nc1 % core_counts > 0 else 0)
            if (self.nc1 % core_counts == 0) or (self.nc1 % nc1_size == 0):
                is_same_percore = 0
            else:
                is_same_percore = 1

            block_dim = self.nc1 // nc1_size + (
                0 if self.nc1 // core_counts == 0 else is_same_percore)

            with self.tik_instance.for_range(0, block_dim,
                                             block_num=block_dim) as block_index:
                with self.tik_instance.if_scope(block_index != block_dim - 1):
                    with self.tik_instance.for_range(0, nc1_size) as nc1_index:
                        self._fun_no_cut(block_index, nc1_index, nc1_size)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.nc1 - (block_dim - 1) * nc1_size) as nc1_index:
                        self._fun_no_cut(block_index, nc1_index, nc1_size)

        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=self.input_fmap_gm,
                                   outputs=(self.output_max_gm,
                                            self.output_mask_gm))
        return self.tik_instance

    def _check_if_need_cut_h_w(self):
        """
        funtion check if need cutH or cutW

        Parameters
        ----------
            none

        Returns
        -------
        need_cut_h: bool
        need_cut_h_w: bool
        need_cut: bool

        """
        need_cut_h = False
        need_cut_h_w = False
        need_cut = False
        ub_size_used_max = self.output_wh * \
                           self.c0_size * self.kernel_h * self.kernel_w * 2
        ub_size_cut_h_max = self.output_w * self.c0_size * \
                            self.kernel_h * self.kernel_w * 2

        if ub_size_used_max > (UB_SIZE / 2):
            need_cut_h = True

        if ub_size_cut_h_max > (UB_SIZE / 2):
            need_cut_h_w = True

        if self.kernel_h * self.input_w * self.c0_size * 2 > L1_SIZE:
            raise RuntimeError( \
                "cutC0 is needed and this scene is not supported")

        if not need_cut_h:
            if self.input_wh * self.c0_size * 2 > L1_SIZE:
                need_cut = True

        return need_cut_h, need_cut_h_w, need_cut

    # pylint: disable=too-many-locals
    def _fun_no_cut(self, block_index, nc1_index, nc1_size):
        """
        funtion while no need cut H

        Parameters
        ----------
        block_index: index of block
        nc1_index: index of nc1

        Returns
        -------
        none
        """

        fmap_l1_shape = (self.input_h, self.input_w, self.c0_size)
        input_fmap_l1 = self.tik_instance.Tensor(self.input_dtype,
                                                 fmap_l1_shape,
                                                 name="input_fmap_l1",
                                                 scope=tik.scope_cbuf)
        fmap_shape_ub = (self.fmap_h_num * 16, self.kernel_h,
                         self.kernel_w, self.c0_size)
        fmap_ub = self.tik_instance.Tensor(self.input_dtype,
                                           fmap_shape_ub, name="fmap_ub",
                                           scope=tik.scope_ubuf)
        mask_shape_ub = (self.kernel_h, self.kernel_w,
                         self.fmap_h_num, self.c0_size)
        mask_ub = self.tik_instance.Tensor("uint16", mask_shape_ub,
                                           name="mask_ub",
                                           scope=tik.scope_ubuf)
        data_x_max = self.tik_instance.Tensor("float16",
                                              (self.fmap_h_num, 16, 16),
                                              name="data_x_max",
                                              scope=tik.scope_ubuf)

        # copy input fmap from gm to l1
        gm_l1_burst_len = int(self.input_wh * self.c0_size // 16)
        self.tik_instance.data_move(input_fmap_l1, \
                                    self.input_fmap_gm[
                                        (block_index * nc1_size + nc1_index) * \
                                        self.input_wh * self.c0_size], 0, 1,
                                    gm_l1_burst_len, 0, 0)

        # load3dv1, from l1 to ub, every time process 16x  kernels
        with self.tik_instance.for_range(0, self.fmap_h_num) as h_index:
            # ??? 此处有疑问
            source_h = (((h_index * 256 * self.fmap_w) // \
                         (16 * self.fmap_w)) // self.output_w) * \
                       self.stride_h - self.pad[2]
            source_w = (((h_index * 256 * self.fmap_w) //
                         (16 * self.fmap_w)) % self.output_w) * \
                       self.stride_w - self.pad[0]
            self.scalar_source_h.set_as(source_h)
            self.scalar_source_w.set_as(source_w)

            self.tik_instance.load3dv1(fmap_ub[h_index * 256 * self.fmap_w], \
                                       input_fmap_l1[0], self.pad,
                                       self.input_h, self.input_w, \
                                       0, 0, 0, self.scalar_source_w,
                                       self.scalar_source_h, \
                                       self.stride_w, self.stride_h,
                                       self.kernel_w, self.kernel_h, \
                                       self.dilation_w, self.dilation_h, 1, 0,
                                       self.fmap_w, 0, \
                                       self.pad_value)

        if self.fmap_w != 1:
            # calc max_pool and max_indices
            self._calc_max_and_mask(self.fmap_h_num, fmap_ub, \
                                    data_x_max, mask_ub)

            # move max output to gm
            self.tik_instance.data_move(
                self.output_max_gm[(block_index * nc1_size + nc1_index) * \
                                   self.output_wh * self.c0_size],
                data_x_max[0], 0, 1, \
                self.fmap_h, 0, 0)

            # remove kernel size same maxval position
            self._remove_repeated_fun(mask_ub)

        else:
            # move max output to gm
            self.tik_instance.data_move(
                self.output_max_gm[(block_index * nc1_size + nc1_index) * \
                                   self.output_wh * self.c0_size], fmap_ub[0],
                0, 1, \
                self.fmap_h, 0, 0)
            self._dup_mask_fun(mask_ub, mask_shape_ub)

        with self.tik_instance.for_range(0, self.fmap_w) as w_index:
            offset_output_mask = (block_index * nc1_size + nc1_index) * \
                                 (
                                         self.fmap_h_num + 1) * self.fmap_w * self.c0_size
            if self.mask_zero != 0 and self.fmap_w != 1:
                self.tik_instance.vector_dup([0, self.mask_zero], \
                                             mask_ub[
                                                 w_index * self.fmap_h_num * self.c0_size + \
                                                 self.fmap_h_num * 16 - 16], 0,
                                             1, 1, 8)

            self.tik_instance.data_move(self.output_mask_gm[ \
                                            offset_output_mask + w_index * (
                                                    self.fmap_h_num + 1) * 16], \
                                        mask_ub[
                                            w_index * self.fmap_h_num * self.c0_size],
                                        0, 1, \
                                        self.fmap_h_num, 0, 0)

    # pylint: disable=too-many-locals,too-many-statements
    def _calc_only_cut_h(self, cur_h_idx, cut_h_size, cut_stride, cut_h_num,
                         input_fmap_l1, fmap_ub, fmap_cut_h,
                         mask_shape_ub, nc1_num):
        """
        calc only cut H

        Parameters
        ----------
        cur_h_idx: index of cuth
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        input_fmap_l1: fmag in l1
        fmap_ub: fmag in ub
        fmap_cut_h: fmag cutH
        mask_shape_ub: shape of mask
        nc1_num: num of n*c1

        Returns
        -------
        none
        """
        fmap_cut_h_num = _ceil_div(fmap_cut_h, 16)
        mask_ub = self.tik_instance.Tensor("uint16", mask_shape_ub, \
                                           name="mask_ub",
                                           scope=tik.scope_ubuf)
        data_x_max = self.tik_instance.Tensor("float16", ( \
            fmap_cut_h_num, 16, 16), name="data_x_max", scope=tik.scope_ubuf)

        with self.tik_instance.if_scope(cur_h_idx != 0):
            with self.tik_instance.if_scope(cur_h_idx != (cut_h_num - 1)):
                # copy input fmap from gm to l1
                gm_l1_burst_len = int(cut_h_size * self.input_w)
                with self.tik_instance.if_scope(cur_h_idx == (cut_h_num - 2)):
                    len_tmp = min(cut_h_size, (self.input_h + self.pad[2] - \
                                               cut_stride * (cut_h_num - 2)))
                    gm_l1_burst_len_1 = len_tmp * self.input_w
                    pad_max = max(0, self.pad[3] - cut_stride)

                    self.tik_instance.data_move(input_fmap_l1, \
                                                self.input_fmap_gm[
                                                    nc1_num * self.input_h * \
                                                    self.input_w * self.c0_size + \
                                                    (cur_h_idx * cut_stride -
                                                     self.pad[2]) * \
                                                    self.input_w * self.c0_size], \
                                                0, 1, gm_l1_burst_len_1, 0, 0)

                    with self.tik_instance.for_range(0, fmap_cut_h_num) \
                            as h_index:
                        source_h = (((h_index * 256 * self.fmap_w) // \
                                     (16 * self.fmap_w)) // self.output_w) * \
                                   self.stride_h
                        source_w = (((h_index * 256 * self.fmap_w) // \
                                     (16 * self.fmap_w)) % self.output_w) * \
                                   self.stride_w - self.pad[0]

                        self.scalar_source_h.set_as(source_h)
                        self.scalar_source_w.set_as(source_w)

                        self.tik_instance.load3dv1(fmap_ub[h_index * \
                                                           256 * self.fmap_w],
                                                   input_fmap_l1[0], \
                                                   (
                                                       self.pad[0], self.pad[1], 0,
                                                       pad_max), \
                                                   cut_h_size - pad_max,
                                                   self.input_w, 0, 0, 0, \
                                                   self.scalar_source_w,
                                                   self.scalar_source_h, \
                                                   self.stride_w,
                                                   self.stride_h,
                                                   self.kernel_w, \
                                                   self.kernel_h,
                                                   self.dilation_w,
                                                   self.dilation_h,
                                                   1, 0, self.fmap_w, 0,
                                                   self.pad_value)

                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(cur_h_idx != 1):
                        self.tik_instance.data_move(input_fmap_l1,
                                                    self.input_fmap_gm[
                                                        nc1_num * self.input_wh * \
                                                        self.c0_size + (
                                                                cur_h_idx * cut_stride - \
                                                                self.pad[
                                                                    2]) * self.input_w * self.c0_size], \
                                                    0, 1, gm_l1_burst_len, 0,
                                                    0)

                        with self.tik_instance.for_range(0, \
                                                         fmap_cut_h_num) as h_index:
                            source_h = (((h_index * 256 * self.fmap_w) // \
                                         (16 * self.fmap_w)) // \
                                        self.output_w) * self.stride_h
                            source_w = (((h_index * 256 * self.fmap_w) // \
                                         (16 * self.fmap_w)) % self.output_w) * \
                                       self.stride_w - self.pad[0]

                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)

                            self.tik_instance.load3dv1( \
                                fmap_ub[h_index * 256 * \
                                        self.fmap_w], input_fmap_l1[0], \
                                (self.pad[0], self.pad[1], 0, 0), \
                                cut_h_size, self.input_w, 0, 0, 0, \
                                self.scalar_source_w, self.scalar_source_h, \
                                self.stride_w, self.stride_h, self.kernel_w, \
                                self.kernel_h, self.dilation_w, \
                                self.dilation_h, 1, 0, \
                                self.fmap_w, 0, self.pad_value)
                    with self.tik_instance.else_scope():
                        # copy input fmap from gm to l1
                        len_tmp1 = min(cut_h_size, \
                                       (cut_h_size - self.pad[2] + cut_stride))
                        pad_max1 = max(0, self.pad[2] - cut_stride)
                        gm_l1_burst_len2 = int(len_tmp1 * self.input_w)
                        gm_tem = max(0, cut_stride - self.pad[2])
                        self.tik_instance.data_move(input_fmap_l1, \
                                                    self.input_fmap_gm[
                                                        nc1_num * self.input_wh * \
                                                        self.c0_size + gm_tem * self.input_w * \
                                                        self.c0_size], 0, 1,
                                                    gm_l1_burst_len2, 0, 0)

                        with self.tik_instance.for_range(0, fmap_cut_h_num) \
                                as h_index:
                            source_h = (((h_index * 256 * self.fmap_w) // \
                                         (16 * self.fmap_w)) // \
                                        self.output_w) * self.stride_h - pad_max1
                            source_w = (((h_index * 256 * self.fmap_w) // \
                                         (16 * self.fmap_w)) % self.output_w) * \
                                       self.stride_w - self.pad[0]

                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(fmap_ub[ \
                                                           h_index * 256 * self.fmap_w], \
                                                       input_fmap_l1[0], (
                                                           self.pad[0],
                                                           self.pad[1], \
                                                           pad_max1, 0),
                                                       (cut_h_size - pad_max1), \
                                                       self.input_w, 0, 0, 0,
                                                       self.scalar_source_w, \
                                                       self.scalar_source_h,
                                                       self.stride_w, \
                                                       self.stride_h,
                                                       self.kernel_w,
                                                       self.kernel_h, \
                                                       self.dilation_w,
                                                       self.dilation_h, 1, 0, \
                                                       self.fmap_w, 0,
                                                       self.pad_value)

                if self.fmap_w != 1:
                    self._calc_max_and_mask(fmap_cut_h_num, \
                                            fmap_ub, data_x_max, mask_ub)

                    # move max output to gm
                    gm_max_burst_len = int(fmap_cut_h)
                    self.tik_instance.data_move(
                        self.output_max_gm[nc1_num * self.output_wh * \
                                           self.c0_size + cur_h_idx * fmap_cut_h * \
                                           self.c0_size], data_x_max[0], 0, 1, \
                        gm_max_burst_len, 0, 0)
                    self._remove_repeated_fun(mask_ub, fmap_cut_h)
                else:
                    # move max output to gm
                    gm_max_burst_len = int(fmap_cut_h)
                    self.tik_instance.data_move( \
                        self.output_max_gm[nc1_num * self.output_wh * \
                                           self.c0_size + cur_h_idx * fmap_cut_h * \
                                           self.c0_size], fmap_ub[0], 0, 1, \
                        gm_max_burst_len, 0, 0)
                    self._dup_mask_fun(mask_ub, mask_shape_ub)

                with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                    offset_output_mask = \
                        nc1_num * (self.fmap_h_num + 1) * \
                        self.fmap_w * self.c0_size + \
                        cur_h_idx * fmap_cut_h

                    self.tik_instance.data_move( \
                        self.output_mask_gm[offset_output_mask + \
                                            w_index * (
                                                    self.fmap_h_num + 1) * 16], \
                        mask_ub[w_index * fmap_cut_h_num * \
                                self.c0_size], 0, 1, fmap_cut_h_num, 0, 0)
            with self.tik_instance.else_scope():
                cut_h_tail = self.input_h + self.pad[2] - cut_stride * \
                             (cut_h_num - 1)
                if cut_h_tail > cut_h_size:
                    cut_h_tail = cut_h_size

                output_h_tail = (cut_h_tail - self.kernel_h + self.stride_h + \
                                 self.pad[3]) // self.stride_h

                fmap_h_tail = self.output_w * output_h_tail
                fmap_h_tail_num = _ceil_div(fmap_h_tail, 16)
                # copy input fmap from gm to l1
                gm_l1_burst_len = int(cut_h_tail * self.input_w * \
                                      self.c0_size // 16)

                self.tik_instance.data_move(input_fmap_l1, \
                                            self.input_fmap_gm[
                                                nc1_num * self.input_wh * \
                                                self.c0_size + (
                                                        cur_h_idx * cut_stride - \
                                                        self.pad[
                                                            2]) * self.input_w * self.c0_size],
                                            0, 1, \
                                            gm_l1_burst_len, 0, 0)

                with self.tik_instance.for_range(
                        0, fmap_h_tail_num) as h_index:
                    source_h = (((h_index * 256 * self.fmap_w)
                                 / (16 * self.fmap_w)) /
                                self.output_w) * self.stride_h
                    source_w = (((h_index * 256 *
                                  self.fmap_w) /
                                 (16 * self.fmap_w)) %
                                self.output_w) * self.stride_w - \
                               self.pad[0]
                    self.scalar_source_h.set_as(source_h)
                    self.scalar_source_w.set_as(source_w)
                    self.tik_instance.load3dv1(
                        fmap_ub[h_index * 256 *
                                self.fmap_w],
                        input_fmap_l1[0],
                        (self.pad[0], self.pad[1], 0, self.pad[3]),
                        cut_h_tail, self.input_w, 0, 0, 0,
                        self.scalar_source_w,
                        self.scalar_source_h, self.stride_w,
                        self.stride_h, self.kernel_w, self.kernel_h,
                        self.dilation_w, self.dilation_h, 1, 0, self.fmap_w,
                        0, self.pad_value)
                if self.fmap_w != 1:
                    self._calc_max_and_mask(fmap_h_tail_num,
                                            fmap_ub,
                                            data_x_max, mask_ub,
                                            fmap_cut_h_num)
                    # move max output to gm
                    gm_max_burst_len = int(fmap_h_tail)
                    self.tik_instance.data_move(
                        self.output_max_gm[
                            nc1_num * self.output_h *
                            self.output_w * self.c0_size +
                            cur_h_idx * fmap_cut_h *
                            self.c0_size],
                        data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._remove_repeated_fun(mask_ub, fmap_h_tail, 0,
                                              0, fmap_cut_h)
                else:
                    # move max output to gm
                    gm_max_burst_len = int(fmap_h_tail)
                    self.tik_instance.data_move(
                        self.output_max_gm[
                            nc1_num * self.output_h *
                            self.output_w * self.c0_size +
                            cur_h_idx * fmap_cut_h *
                            self.c0_size],
                        fmap_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                    self._dup_mask_fun(mask_ub, mask_shape_ub)
                mask_cut = fmap_h_tail_num * 16 - fmap_h_tail
                mask_zero_cut = 2 ** 16 - 2 ** (16 - mask_cut)
                with self.tik_instance.for_range(
                        0, self.fmap_w) as w_index:
                    offset_output_mask = \
                        nc1_num * (self.fmap_h_num + 1) * \
                        self.fmap_w * self.c0_size + \
                        cur_h_idx * fmap_cut_h
                    if mask_zero_cut != 0 and self.fmap_w != 1:
                        self.tik_instance.vector_dup([
                            0, mask_zero_cut], mask_ub[
                            w_index * fmap_cut_h_num * self.c0_size +
                            fmap_h_tail_num * 16 - 16], 0, 1, 1, 8)

                    self.tik_instance.data_move(
                        self.output_mask_gm[
                            offset_output_mask +
                            w_index * (self.fmap_h_num + 1) * 16],
                        mask_ub[w_index * fmap_cut_h_num *
                                self.c0_size],
                        0, 1, fmap_h_tail_num, 0, 0)
        with self.tik_instance.else_scope():
            # copy input fmap from gm to l1
            gm_l1_burst_len = int(
                (cut_h_size - self.pad[2]) * self.input_w *
                self.c0_size // 16)
            self.tik_instance.data_move(input_fmap_l1, self.input_fmap_gm[ \
                nc1_num * self.input_wh * self.c0_size], 0, 1, \
                                        gm_l1_burst_len, 0, 0)

            with self.tik_instance.for_range(0, fmap_cut_h_num) \
                    as h_index:
                source_h = (((h_index * 256 * self.fmap_w) //
                             (16 * self.fmap_w)) //
                            self.output_w) * self.stride_h - self.pad[2]
                source_w = (((h_index * 256 * self.fmap_w) //
                             (16 * self.fmap_w)) %
                            self.output_w) * self.stride_w - self.pad[0]

                self.scalar_source_h.set_as(source_h)
                self.scalar_source_w.set_as(source_w)
                self.tik_instance.load3dv1(
                    fmap_ub[h_index * 256 * self.fmap_w],
                    input_fmap_l1[0],
                    (self.pad[0], self.pad[1], self.pad[2], 0),
                    (cut_h_size - self.pad[2]),
                    self.input_w, 0, 0, 0, self.scalar_source_w,
                    self.scalar_source_h, self.stride_w, self.stride_h,
                    self.kernel_w, self.kernel_h, self.dilation_w,
                    self.dilation_h, 1, 0,
                    self.fmap_w, 0, self.pad_value)

            if self.fmap_w != 1:
                self._calc_max_and_mask(fmap_cut_h_num,
                                        fmap_ub, data_x_max, mask_ub)
                # move max output to gm
                gm_max_burst_len = int(fmap_cut_h)
                self.tik_instance.data_move(
                    self.output_max_gm[
                        nc1_num * self.output_wh *
                        self.c0_size],
                    data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                self._remove_repeated_fun(mask_ub, fmap_cut_h)
            else:
                # move max output to gm
                gm_max_burst_len = int(fmap_cut_h)
                self.tik_instance.data_move(
                    self.output_max_gm[
                        nc1_num * self.output_wh *
                        self.c0_size],
                    fmap_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                self._dup_mask_fun(mask_ub, mask_shape_ub)
            with self.tik_instance.for_range(
                    0, self.fmap_w) as w_index:
                offset_output_mask = \
                    nc1_num * (self.fmap_h_num + 1) * \
                    self.fmap_w * self.c0_size + \
                    cur_h_idx * fmap_cut_h
                self.tik_instance.data_move(
                    self.output_mask_gm[
                        offset_output_mask + w_index *
                        (self.fmap_h_num + 1) * 16],
                    mask_ub[w_index *
                            fmap_cut_h_num *
                            self.c0_size], 0,
                    1, fmap_cut_h_num, 0, 0)

    # pylint: disable=too-many-locals,too-many-statements
    def _fun_only_cut_h(self, block_index, nc1_cuth_index, cut_h_size,
                        cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h):
        """
        funtion only cut H

        Parameters
        ----------
        block_index: index of block
        nc1_cuth_index: index of nc1_cuth
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        nc1_cuth_size: size of nc1_cuth
        flag_cut_h: bool

        Returns
        -------
        none
        """

        fmap_l1_shape = (cut_h_size, self.input_w, self.c0_size)
        input_fmap_l1 = self.tik_instance.Tensor(self.input_dtype,
                                                 fmap_l1_shape,
                                                 name="input_fmap_l1",
                                                 scope=tik.scope_cbuf)
        out_size_cut_h = \
            (cut_h_size - self.kernel_h + self.stride_h) // self.stride_h
        fmap_cut_h = self.output_w * out_size_cut_h
        fmap_cut_h_num = _ceil_div(fmap_cut_h, 16)
        fmap_shape_ub = (fmap_cut_h_num * 16, self.kernel_h,
                         self.kernel_w, self.c0_size)

        fmap_ub = self.tik_instance.Tensor(self.input_dtype,
                                           fmap_shape_ub,
                                           name="fmap_ub",
                                           scope=tik.scope_ubuf)
        mask_shape_ub = (self.kernel_h, self.kernel_w, fmap_cut_h_num,
                         self.c0_size)
        if flag_cut_h:
            cur_h_idx = (block_index * nc1_cuth_size + nc1_cuth_index) \
                        % cut_h_num
            nc1_num = (block_index * nc1_cuth_size + nc1_cuth_index) \
                      // cut_h_num

            self._calc_only_cut_h(cur_h_idx, cut_h_size, cut_stride,
                                  cut_h_num,
                                  input_fmap_l1, fmap_ub,
                                  fmap_cut_h,
                                  mask_shape_ub, nc1_num)
        else:
            nc1_num = block_index * nc1_cuth_size + nc1_cuth_index
            with self.tik_instance.for_range(0, cut_h_num) as cur_h_idx:
                self._calc_only_cut_h(cur_h_idx, cut_h_size, cut_stride,
                                      cut_h_num, input_fmap_l1, fmap_ub,
                                      fmap_cut_h, mask_shape_ub, nc1_num)

    # pylint: disable=too-many-statements,too-many-branches
    def _calc_need_cut_h_w(self, nc1_num, cut_h_size, cut_h_num, cur_h_idx,
                           cut_stride):
        """
        funtion need cut H and W while l1 not enough

        Parameters
        ----------
        nc1_num: num of n*c1
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        cur_h_idx: index of cuth

        Returns
        -------
        none
        """
        cut_w_size, cut_w_stride, cut_w_num = self._calc_cut_w_size_fun()
        fmap_l1_shape = (cut_h_size, self.input_w, self.c0_size)
        input_fmap_l1 = self.tik_instance.Tensor(self.input_dtype,
                                                 fmap_l1_shape,
                                                 name="input_fmap_l1",
                                                 scope=tik.scope_cbuf)
        with self.tik_instance.for_range(0, cut_w_num) as cut_w_index:
            out_size_cut_h = (cut_h_size - self.kernel_h +
                              self.stride_h) // self.stride_h
            fmap_cut_h = self.output_w * out_size_cut_h
            out_size_cut_w = (cut_w_size - self.kernel_w +
                              self.stride_w) // self.stride_w
            fmap_cut_w = out_size_cut_w
            fmap_cut_w_num = _ceil_div(fmap_cut_w, 16)
            fmap_shape_ub = (
                fmap_cut_w_num * 16, self.kernel_h,
                self.kernel_w, self.c0_size)
            fmap_ub = \
                self.tik_instance.Tensor(self.input_dtype,
                                         fmap_shape_ub,
                                         name="fmap_ub",
                                         scope=tik.scope_ubuf)
            mask_shape_ub = (self.kernel_h, self.kernel_w,
                             fmap_cut_w_num,
                             self.c0_size)
            mask_ub = self.tik_instance.Tensor("uint16",
                                               mask_shape_ub,
                                               name="mask_ub",
                                               scope=tik.scope_ubuf)
            data_x_max = self.tik_instance.Tensor("float16",
                                                  (fmap_cut_w_num,
                                                   16, 16),
                                                  name="data_x_max",
                                                  scope=tik.scope_ubuf)
            with self.tik_instance.if_scope(cur_h_idx != 0):
                with self.tik_instance.if_scope(cur_h_idx != (cut_h_num - 1)):
                    # copy input fmap from gm to l1
                    gm_l1_burst_len = int(cut_h_size * self.input_w *
                                          self.c0_size // 16)
                    self.tik_instance.data_move(
                        input_fmap_l1,
                        self.input_fmap_gm[
                            nc1_num * self.input_wh *
                            self.c0_size +
                            (cur_h_idx * cut_stride - self.pad[2]) *
                            self.input_w *
                            self.c0_size],
                        0, 1, gm_l1_burst_len, 0, 0)
                    with self.tik_instance.if_scope(cut_w_index != 0):
                        with self.tik_instance.if_scope(
                                cut_w_index != (cut_w_num - 1)):
                            with self.tik_instance. \
                                    for_range(0, fmap_cut_w_num) \
                                    as h_index:
                                source_h = 0
                                source_w = \
                                    (((h_index * 256 * self.fmap_w) /
                                      (16 * self.fmap_w)) %
                                     out_size_cut_w) * self.stride_w + \
                                    cut_w_stride * cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_ub[
                                        h_index * 256 *
                                        self.fmap_w],
                                    input_fmap_l1[0],
                                    (self.pad[0], self.pad[1], 0, 0),
                                    cut_h_size, self.input_w,
                                    0, 0, 0, self.scalar_source_w,
                                    self.scalar_source_h,
                                    self.stride_w, self.stride_h,
                                    self.kernel_w, self.kernel_h,
                                    self.dilation_w, self.dilation_h, 1, 0,
                                    self.fmap_w, 0, self.pad_value)
                            if self.fmap_w != 1:
                                self._calc_max_and_mask(fmap_cut_w_num,
                                                        fmap_ub,
                                                        data_x_max, mask_ub)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.output_h *
                                        self.output_w * self.c0_size +
                                        cur_h_idx * self.output_w *
                                        self.c0_size +
                                        cut_w_index * fmap_cut_w *
                                        self.c0_size],
                                    data_x_max[0], 0, 1, gm_max_burst_len, 0,
                                    0)
                                self._remove_repeated_fun(mask_ub,
                                                          fmap_cut_h,
                                                          fmap_cut_w)
                            else:
                                # move max output to gm
                                gm_max_burst_len = int(fmap_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.output_h *
                                        self.output_w * self.c0_size +
                                        cur_h_idx * self.output_w *
                                        self.c0_size +
                                        cut_w_index * fmap_cut_w *
                                        self.c0_size],
                                    fmap_ub[0], 0, 1, gm_max_burst_len,
                                    0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            with self.tik_instance. \
                                    for_range(0, self.fmap_w) \
                                    as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_h_num + 1) * \
                                    self.fmap_w * \
                                    self.c0_size + \
                                    cur_h_idx * fmap_cut_h + \
                                    cut_w_index * fmap_cut_w
                                self.tik_instance.data_move(
                                    self.output_mask_gm[
                                        offset_output_mask
                                        + w_index *
                                        (self.fmap_h_num
                                         + 1) * 16],
                                    mask_ub[w_index * fmap_cut_w_num *
                                            self.c0_size],
                                    0, 1, fmap_cut_w_num, 0, 0)
                        with self.tik_instance.else_scope():
                            cut_w_tail = self.input_w + self.pad[
                                0] - cut_w_stride * (cut_w_num - 1)
                            if cut_w_tail > cut_w_size:
                                cut_w_tail = cut_w_size
                            out_size_tail_w = (cut_w_tail - self.kernel_w +
                                               self.stride_w +
                                               self.pad[1]) // self.stride_w
                            fmap_tail_w = out_size_tail_w
                            fmap_tail_w_num = _ceil_div(
                                fmap_tail_w, 16)
                            with self.tik_instance.for_range(
                                    0, fmap_tail_w_num) as h_index:
                                source_h = 0
                                source_w = \
                                    (((h_index * 256 *
                                       self.fmap_w) /
                                      (16 * self.fmap_w)) %
                                     out_size_tail_w
                                     ) * self.stride_w + \
                                    cut_w_stride * cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_ub[
                                        h_index * 256 * self.fmap_w],
                                    input_fmap_l1[0],
                                    (self.pad[0], self.pad[1], 0, 0),
                                    cut_h_size, self.input_w,
                                    0, 0, 0, self.scalar_source_w,
                                    self.scalar_source_h,
                                    self.stride_w, self.stride_h,
                                    self.kernel_w, self.kernel_h,
                                    self.dilation_w, self.dilation_h, 1, 0,
                                    self.fmap_w, 0, self.pad_value)
                            if self.fmap_w != 1:
                                self._calc_max_and_mask(fmap_tail_w_num,
                                                        fmap_ub,
                                                        data_x_max, mask_ub,
                                                        fmap_cut_w_num)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_tail_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.output_wh *
                                        self.c0_size + cur_h_idx *
                                        self.output_w * self.c0_size +
                                        cut_w_index * fmap_cut_w *
                                        self.c0_size],
                                    data_x_max[0], 0, 1, gm_max_burst_len, 0,
                                    0)
                                self._remove_repeated_fun(mask_ub,
                                                          fmap_cut_h,
                                                          fmap_tail_w,
                                                          fmap_cut_w)
                            else:
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.output_wh *
                                        self.c0_size + cur_h_idx *
                                        self.output_w * self.c0_size +
                                        cut_w_index * fmap_cut_w *
                                        self.c0_size],
                                    fmap_ub[0], 0, 1, gm_max_burst_len,
                                    0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            with self.tik_instance.for_range(
                                    0, self.fmap_w) as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_h_num + 1) * \
                                    self.fmap_w * \
                                    self.c0_size + cur_h_idx * \
                                    fmap_cut_h + cut_w_index * \
                                    fmap_cut_w
                                self.tik_instance.data_move(
                                    self.output_mask_gm[
                                        offset_output_mask
                                        + w_index *
                                        (self.fmap_h_num
                                         + 1) * 16],
                                    mask_ub[w_index *
                                            fmap_cut_w_num *
                                            self.c0_size],
                                    0, 1, fmap_tail_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(
                                0, fmap_cut_w_num) as h_index:
                            source_h = 0
                            source_w = (((h_index * 256 *
                                          self.fmap_w) /
                                         (16 * self.fmap_w))
                                        % out_size_cut_w
                                        ) * self.stride_w - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_ub[h_index * 256 *
                                        self.fmap_w],
                                input_fmap_l1[0],
                                (self.pad[0], self.pad[1], 0, 0),
                                cut_h_size, self.input_w,
                                0, 0, 0, self.scalar_source_w,
                                self.scalar_source_h,
                                self.stride_w, self.stride_h,
                                self.kernel_w, self.kernel_h,
                                self.dilation_w, self.dilation_h,
                                1, 0, self.fmap_w,
                                0, self.pad_value)
                        if self.fmap_w != 1:
                            self._calc_max_and_mask(fmap_cut_w_num,
                                                    fmap_ub, data_x_max,
                                                    mask_ub)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.output_wh *
                                    self.c0_size + cur_h_idx *
                                    self.output_w * self.c0_size],
                                data_x_max[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub,
                                                      fmap_cut_h,
                                                      fmap_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.output_wh *
                                    self.c0_size + cur_h_idx *
                                    self.output_w * self.c0_size],
                                fmap_ub[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        with self.tik_instance.for_range(
                                0, self.fmap_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_h_num + 1) * \
                                self.fmap_w * self.c0_size + \
                                cur_h_idx * fmap_cut_h
                            self.tik_instance.data_move(
                                self.output_mask_gm[
                                    offset_output_mask + w_index *
                                    (self.fmap_h_num + 1) * 16],
                                mask_ub[w_index * fmap_cut_w_num *
                                        self.c0_size],
                                0, 1, fmap_cut_w_num, 0, 0)
                with self.tik_instance.else_scope():
                    # copy input fmap from gm to l1
                    if self.input_h - cut_stride * (cut_h_num - 1) + \
                            self.pad[2] <= cut_h_size:
                        gm_l1_burst_len = int((self.input_h - cut_stride *
                                               (cut_h_num - 1) + self.pad[2]) *
                                              self.input_w *
                                              self.c0_size // 16)
                    else:
                        gm_l1_burst_len = int(cut_h_size * self.input_w *
                                              self.c0_size // 16)
                    self.tik_instance.data_move(
                        input_fmap_l1, self.input_fmap_gm[
                            nc1_num * self.input_h *
                            self.input_w * self.c0_size +
                            (cur_h_idx * cut_stride - self.pad[2]) *
                            self.input_w * self.c0_size], 0,
                        1, gm_l1_burst_len, 0, 0)
                    with self.tik_instance.if_scope(cut_w_index != 0):
                        with self.tik_instance.if_scope(
                                cut_w_index != (cut_w_num - 1)):
                            with self.tik_instance.for_range(
                                    0, fmap_cut_w_num) as h_index:
                                source_h = 0
                                source_w = \
                                    (((h_index * 256 *
                                       self.fmap_w) /
                                      (16 * self.fmap_w)) %
                                     out_size_cut_w
                                     ) * self.stride_w + \
                                    cut_w_stride * cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_ub[h_index * 256 * self.fmap_w],
                                    input_fmap_l1[0],
                                    (self.pad[0], self.pad[1],
                                     0, self.pad[3]),
                                    (cut_h_size - self.pad[3]), self.input_w,
                                    0, 0, 0,
                                    self.scalar_source_w,
                                    self.scalar_source_h,
                                    self.stride_w, self.stride_h,
                                    self.kernel_w, self.kernel_h,
                                    self.dilation_w, self.dilation_h, 1, 0,
                                    self.fmap_w, 0,
                                    self.pad_value)
                            if self.fmap_w != 1:
                                self._calc_max_and_mask(fmap_cut_w_num,
                                                        fmap_ub,
                                                        data_x_max, mask_ub)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.output_h *
                                        self.output_w * self.c0_size +
                                        cur_h_idx * out_size_cut_h *
                                        self.output_w * self.c0_size +
                                        cut_w_index * fmap_cut_w *
                                        self.c0_size],
                                    data_x_max[0], 0, 1,
                                    gm_max_burst_len, 0, 0)
                                self._remove_repeated_fun(mask_ub,
                                                          fmap_cut_h,
                                                          fmap_cut_w)
                            else:
                                # move max output to gm
                                gm_max_burst_len = int(fmap_cut_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.output_h *
                                        self.output_w * self.c0_size +
                                        cur_h_idx * out_size_cut_h *
                                        self.output_w * self.c0_size +
                                        cut_w_index * fmap_cut_w *
                                        self.c0_size],
                                    fmap_ub[0], 0, 1,
                                    gm_max_burst_len, 0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            with self.tik_instance.for_range(
                                    0, self.fmap_w) as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_h_num + 1) * \
                                    self.fmap_w * \
                                    self.c0_size + \
                                    cur_h_idx * fmap_cut_h + \
                                    cut_w_index * fmap_cut_w
                                self.tik_instance.data_move(
                                    self.output_mask_gm[
                                        offset_output_mask + w_index *
                                        (self.fmap_h_num + 1) * 16],
                                    mask_ub[w_index * fmap_cut_w_num *
                                            self.c0_size],
                                    0, 1, fmap_cut_w_num, 0, 0)
                        with self.tik_instance.else_scope():
                            cut_w_tail = self.input_w + self.pad[
                                0] - cut_w_stride * (cut_w_num - 1)
                            if cut_w_tail > cut_w_size:
                                cut_w_tail = cut_w_size
                            out_size_tail_w = (cut_w_tail - self.kernel_w +
                                               self.stride_w +
                                               self.pad[1]) // self.stride_w
                            fmap_tail_w = out_size_tail_w
                            fmap_tail_w_num = _ceil_div(
                                fmap_tail_w, 16)
                            with self.tik_instance.for_range(
                                    0, fmap_tail_w_num) as h_index:
                                source_h = 0
                                source_w = \
                                    (((h_index * 256 *
                                       self.fmap_w) /
                                      (16 * self.fmap_w)) %
                                     out_size_tail_w
                                     ) * self.stride_w + cut_w_stride * \
                                    cut_w_index - self.pad[0]
                                self.scalar_source_h.set_as(source_h)
                                self.scalar_source_w.set_as(source_w)
                                self.tik_instance.load3dv1(
                                    fmap_ub[h_index * 256 * self.fmap_w],
                                    input_fmap_l1[0],
                                    (self.pad[0], self.pad[1],
                                     0, self.pad[3]),
                                    (cut_h_size - self.pad[3]),
                                    self.input_w, 0, 0, 0,
                                    self.scalar_source_w,
                                    self.scalar_source_h,
                                    self.stride_w, self.stride_h,
                                    self.kernel_w, self.kernel_h,
                                    self.dilation_w, self.dilation_h, 1, 0,
                                    self.fmap_w, 0, self.pad_value)
                            if self.fmap_w != 1:
                                self._calc_max_and_mask(fmap_tail_w_num,
                                                        fmap_ub,
                                                        data_x_max, mask_ub,
                                                        fmap_cut_w_num)
                                # move max output to gm
                                gm_max_burst_len = int(fmap_tail_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.output_h *
                                        self.output_w * self.c0_size +
                                        cur_h_idx * out_size_cut_h *
                                        self.output_w * self.c0_size +
                                        cut_w_index * fmap_cut_w *
                                        self.c0_size],
                                    data_x_max[0], 0, 1, gm_max_burst_len, 0,
                                    0)
                                self._remove_repeated_fun(mask_ub,
                                                          fmap_cut_h,
                                                          fmap_tail_w,
                                                          fmap_cut_w)
                            else:
                                # move max output to gm
                                gm_max_burst_len = int(fmap_tail_w)
                                self.tik_instance.data_move(
                                    self.output_max_gm[
                                        nc1_num * self.output_h *
                                        self.output_w * self.c0_size +
                                        cur_h_idx * out_size_cut_h *
                                        self.output_w * self.c0_size +
                                        cut_w_index * fmap_cut_w *
                                        self.c0_size],
                                    fmap_ub[0], 0, 1, gm_max_burst_len,
                                    0, 0)
                                self._dup_mask_fun(mask_ub, mask_shape_ub)
                            mask_cut_w = fmap_tail_w_num * 16 - fmap_tail_w
                            mask_zero_w = 2 ** 16 - 2 ** (16 - mask_cut_w)
                            with self.tik_instance.for_range(
                                    0, self.fmap_w) as w_index:
                                offset_output_mask = \
                                    nc1_num * (self.fmap_h_num + 1) * \
                                    self.fmap_w * \
                                    self.c0_size + \
                                    cur_h_idx * fmap_cut_h + \
                                    cut_w_index * fmap_cut_w
                                if mask_zero_w != 0 and self.fmap_w != 1:
                                    self.tik_instance.vector_dup([
                                        0, mask_zero_w], mask_ub[
                                        w_index * fmap_cut_w_num * self.c0_size +
                                        fmap_tail_w_num * 16 - 16], 0,
                                        1, 1, 8)

                                self.tik_instance.data_move(
                                    self.output_mask_gm[
                                        offset_output_mask + w_index *
                                        (self.fmap_h_num + 1) * 16],
                                    mask_ub[w_index * fmap_cut_w_num *
                                            self.c0_size],
                                    0, 1, fmap_tail_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        with self.tik_instance.for_range(
                                0, fmap_cut_w_num) as h_index:
                            source_h = 0
                            source_w = \
                                (((h_index * 256 *
                                   self.fmap_w) /
                                  (16 * self.fmap_w)) %
                                 out_size_cut_w) * \
                                self.stride_w - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_ub[h_index * 256 *
                                        self.fmap_w],
                                input_fmap_l1[0],
                                (self.pad[0], self.pad[1], 0, self.pad[3]),
                                (cut_h_size - self.pad[3]),
                                self.input_w, 0, 0,
                                0, self.scalar_source_w,
                                self.scalar_source_h,
                                self.stride_w, self.stride_h,
                                self.kernel_w, self.kernel_h,
                                self.dilation_w, self.dilation_h, 1, 0,
                                self.fmap_w, 0, self.pad_value)
                        if self.fmap_w != 1:
                            self._calc_max_and_mask(fmap_cut_w_num,
                                                    fmap_ub, data_x_max,
                                                    mask_ub)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.output_wh *
                                    self.c0_size + cur_h_idx *
                                    self.output_w * self.c0_size],
                                data_x_max[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub,
                                                      fmap_cut_h,
                                                      fmap_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.output_wh *
                                    self.c0_size + cur_h_idx *
                                    self.output_w * self.c0_size],
                                fmap_ub[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        with self.tik_instance.for_range(
                                0, self.fmap_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_h_num + 1) * \
                                self.fmap_w * self.c0_size + \
                                cur_h_idx * fmap_cut_h
                            self.tik_instance.data_move(
                                self.output_mask_gm[
                                    offset_output_mask + w_index *
                                    (self.fmap_h_num + 1) * 16],
                                mask_ub[w_index * fmap_cut_w_num *
                                        self.c0_size],
                                0, 1, fmap_cut_w_num, 0, 0)
            with self.tik_instance.else_scope():
                # copy input fmap from gm to l1
                gm_l1_burst_len = int(
                    (cut_h_size - self.pad[2]) * self.input_w *
                    self.c0_size // 16)
                self.tik_instance.data_move(
                    input_fmap_l1,
                    self.input_fmap_gm[
                        nc1_num * self.input_wh * self.c0_size],
                    0, 1, gm_l1_burst_len, 0, 0)
                with self.tik_instance.if_scope(cut_w_index != 0):
                    with self.tik_instance.if_scope(
                            cut_w_index != (cut_w_num - 1)):
                        with self.tik_instance.for_range(
                                0, fmap_cut_w_num) as h_index:
                            source_h = -self.pad[2]
                            source_w = \
                                (((h_index * 256 *
                                   self.fmap_w) /
                                  (16 * self.fmap_w))
                                 % out_size_cut_w
                                 ) * self.stride_w + cut_w_stride * \
                                cut_w_index - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_ub[h_index * 256 *
                                        self.fmap_w],
                                input_fmap_l1[0],
                                (self.pad[0], self.pad[1], self.pad[2], 0),
                                (cut_h_size - self.pad[2]),
                                self.input_w, 0, 0, 0,
                                self.scalar_source_w, self.scalar_source_h,
                                self.stride_w, self.stride_h,
                                self.kernel_w, self.kernel_h,
                                self.dilation_w, self.dilation_h,
                                1, 0, self.fmap_w,
                                0, self.pad_value)
                        if self.fmap_w != 1:
                            self._calc_max_and_mask(fmap_cut_w_num,
                                                    fmap_ub, data_x_max,
                                                    mask_ub)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.output_wh *
                                    self.c0_size +
                                    cut_w_index * fmap_cut_w *
                                    self.c0_size],
                                data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub,
                                                      fmap_cut_h,
                                                      fmap_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_cut_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.output_wh *
                                    self.c0_size +
                                    cut_w_index * fmap_cut_w *
                                    self.c0_size],
                                fmap_ub[0], 0, 1, gm_max_burst_len,
                                0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_h_num + 1) * \
                                self.fmap_w * self.c0_size + \
                                cur_h_idx * fmap_cut_h + \
                                cut_w_index * fmap_cut_w
                            self.tik_instance.data_move(
                                self.output_mask_gm[
                                    offset_output_mask +
                                    w_index * (self.fmap_h_num + 1) *
                                    16],
                                mask_ub[w_index * fmap_cut_w_num *
                                        self.c0_size],
                                0, 1, fmap_cut_w_num, 0, 0)
                    with self.tik_instance.else_scope():
                        cut_w_tail = self.input_w + self.pad[
                            0] - cut_w_stride * (cut_w_num - 1)
                        if cut_w_tail > cut_w_size:
                            cut_w_tail = cut_w_size
                        out_size_tail_w = (cut_w_tail - self.kernel_w +
                                           self.stride_w +
                                           self.pad[1]) // self.stride_w
                        fmap_tail_w = out_size_tail_w
                        fmap_tail_w_num = _ceil_div(
                            fmap_tail_w, 16)
                        with self.tik_instance. \
                                for_range(0, fmap_tail_w_num) \
                                as h_index:
                            source_h = -self.pad[2]
                            source_w = \
                                (((h_index * 256 * self.fmap_w) /
                                  (16 * self.fmap_w)) %
                                 out_size_tail_w) * self.stride_w + \
                                cut_w_stride * cut_w_index - self.pad[0]
                            self.scalar_source_h.set_as(source_h)
                            self.scalar_source_w.set_as(source_w)
                            self.tik_instance.load3dv1(
                                fmap_ub[h_index * 256 * self.fmap_w],
                                input_fmap_l1[0],
                                (self.pad[0], self.pad[1],
                                 self.pad[2], 0), (cut_h_size - self.pad[2]),
                                self.input_w, 0, 0,
                                0, self.scalar_source_w,
                                self.scalar_source_h,
                                self.stride_w, self.stride_h,
                                self.kernel_w, self.kernel_h,
                                self.dilation_w, self.dilation_h,
                                1, 0, self.fmap_w, 0, self.pad_value)
                        if self.fmap_w != 1:
                            self._calc_max_and_mask(fmap_tail_w_num,
                                                    fmap_ub, data_x_max,
                                                    mask_ub,
                                                    fmap_cut_w_num)
                            # move max output to gm
                            gm_max_burst_len = int(fmap_tail_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.output_wh *
                                    self.c0_size + cut_w_index *
                                    fmap_cut_w * self.c0_size],
                                data_x_max[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._remove_repeated_fun(mask_ub,
                                                      fmap_cut_h,
                                                      fmap_tail_w,
                                                      fmap_cut_w)
                        else:
                            # move max output to gm
                            gm_max_burst_len = int(fmap_tail_w)
                            self.tik_instance.data_move(
                                self.output_max_gm[
                                    nc1_num * self.output_wh *
                                    self.c0_size + cut_w_index *
                                    fmap_cut_w * self.c0_size],
                                fmap_ub[0], 0, 1,
                                gm_max_burst_len, 0, 0)
                            self._dup_mask_fun(mask_ub, mask_shape_ub)
                        mask_cut_w = fmap_tail_w_num * 16 - fmap_tail_w
                        mask_zero_w = 2 ** 16 - 2 ** (16 - mask_cut_w)
                        with self.tik_instance.for_range(0, self.fmap_w) as w_index:
                            offset_output_mask = \
                                nc1_num * (self.fmap_h_num + 1) * \
                                self.fmap_w * self.c0_size + \
                                cur_h_idx * fmap_cut_h + \
                                cut_w_index * fmap_cut_w
                            if mask_zero_w != 0 and self.fmap_w != 1 \
                                    and cut_h_num == 1:
                                self.tik_instance.vector_dup(
                                    [0, mask_zero_w],
                                    mask_ub[w_index * fmap_cut_w_num * self.c0_size +
                                            fmap_tail_w_num * 16 - 16], 0, 1, 1, 8)

                            self.tik_instance.data_move(
                                self.output_mask_gm[
                                    offset_output_mask +
                                    w_index *
                                    (self.fmap_h_num + 1) * 16],
                                mask_ub[w_index * fmap_cut_w_num *
                                        self.c0_size],
                                0, 1, fmap_tail_w_num, 0, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, fmap_cut_w_num) as h_index:
                        source_h = -self.pad[2]
                        source_w = (((h_index * 256 * self.fmap_w) / (16 * self.fmap_w)) % out_size_cut_w) * \
                                   self.stride_w - self.pad[0]
                        self.scalar_source_h.set_as(source_h)
                        self.scalar_source_w.set_as(source_w)
                        self.tik_instance.load3dv1(
                            fmap_ub[h_index * 256 * self.fmap_w],
                            input_fmap_l1[0],
                            (self.pad[0], self.pad[1], self.pad[2], 0),
                            (cut_h_size - self.pad[2]),
                            self.input_w, 0, 0, 0,
                            self.scalar_source_w,
                            self.scalar_source_h,
                            self.stride_w, self.stride_h,
                            self.kernel_w, self.kernel_h, self.dilation_w,
                            self.dilation_h, 1,
                            0, self.fmap_w, 0, self.pad_value)
                    if self.fmap_w != 1:
                        self._calc_max_and_mask(fmap_cut_w_num,
                                                fmap_ub,
                                                data_x_max, mask_ub)
                        # move max output to gm
                        gm_max_burst_len = int(fmap_cut_w)
                        self.tik_instance.data_move(
                            self.output_max_gm[nc1_num * self.output_h *
                                               self.output_w *
                                               self.c0_size],
                            data_x_max[0], 0, 1, gm_max_burst_len, 0, 0)
                        self._remove_repeated_fun(mask_ub, fmap_cut_h,
                                                  fmap_cut_w)
                    else:
                        # move max output to gm
                        gm_max_burst_len = int(fmap_cut_w)
                        self.tik_instance.data_move(
                            self.output_max_gm[nc1_num * self.output_h *
                                               self.output_w *
                                               self.c0_size],
                            fmap_ub[0], 0, 1, gm_max_burst_len, 0, 0)
                        self._dup_mask_fun(mask_ub, mask_shape_ub)
                    with self.tik_instance. \
                            for_range(0, self.fmap_w) as w_index:
                        offset_output_mask = \
                            nc1_num * (self.fmap_h_num + 1) * \
                            self.fmap_w * self.c0_size + \
                            cur_h_idx * fmap_cut_h
                        self.tik_instance.data_move(
                            self.output_mask_gm[offset_output_mask +
                                                w_index *
                                                (self.fmap_h_num +
                                                 1) * 16],
                            mask_ub[w_index * fmap_cut_w_num *
                                    self.c0_size],
                            0, 1, fmap_cut_w_num, 0, 0)

    # pylint: disable=too-many-statements,too-many-branches
    def _fun_need_cut_h_w(self, block_index, nc1_cuth_index, cut_h_size,
                          cut_stride, cut_h_num, nc1_cuth_size, flag_cut_h):
        """
        funtion need cut H and W while l1 not enough

        Parameters
        ----------
        block_index: index of block
        nc1_index: index of nc1
        cut_h_size: size of cuth
        cut_stride: stride of cuth
        cut_h_num: number of cuth
        flag_cut_h:bool

        Returns
        -------
        none
        """
        if flag_cut_h:
            cur_h_idx = (block_index * nc1_cuth_size + nc1_cuth_index) % cut_h_num
            nc1_num = (block_index * nc1_cuth_size + nc1_cuth_index) // cut_h_num
            self._calc_need_cut_h_w(nc1_num, cut_h_size, cut_h_num, cur_h_idx,
                                    cut_stride)
        else:
            nc1_num = block_index * nc1_cuth_size + nc1_cuth_index
            with self.tik_instance.for_range(0, cut_h_num) as cur_h_idx:
                self._calc_need_cut_h_w(nc1_num, cut_h_size, cut_h_num,
                                        cur_h_idx, cut_stride)

    def _pooling_output_shape_pad_lr(self, input_size, kernel_size, pad_l, \
                                     pad_r, stride, dilation, ceil_mode):
        temp = input_size + pad_l + pad_r - dilation * (kernel_size - 1) - 1

        if ceil_mode == True:
            output_size = ((temp + (stride - 1)) // stride) + 1
        else:
            output_size = (temp // stride) + 1

        if pad_l > 0:
            # ensure that the last pooling starts inside the image
            # needed to avoid problems in ceil mode
            if (output_size - 1) * stride >= (input_size + pad_l):
                output_size = output_size - 1

        return output_size

    def _pooling_output_shape(self, input_size, kernel_size, pad, stride, \
                              dilation, ceil_mode):
        return self._pooling_output_shape_pad_lr(input_size, kernel_size, \
                                                 pad, pad, stride, dilation,
                                                 ceil_mode)

    def _pool2d_shape_check(self, kernel_h, kernel_w, stride_h, stride_w, \
                            pad_h, pad_w, dilation_h, dilation_w, output_h,
                            output_w):
        if kernel_w <= 0 or kernel_h <= 0:
            raise RuntimeError("kernel size should be greater than zero, but \
                got ", "kH: ", kernel_h, " kW: ", kernel_w)

        if stride_h <= 0 or stride_w <= 0:
            raise RuntimeError("stride should be greater than zero, but got ", \
                               "dH= ", stride_h, "dW= ", stride_w)

        if dilation_h <= 0 or dilation_w <= 0:
            raise RuntimeError("dilation should be greater than 0, but got", \
                               "dilationH= ", dilation_h, ", dilationW= ",
                               dilation_w)

        if (kernel_w // 2) < pad_w or (kernel_h // 2) < pad_h:
            raise RuntimeError("pad should be smaller than half of kernel " \
                               "size, but got", "padW=", pad_w, ", padH= ", \
                               pad_h, ", kW= ", kernel_w, ", kH= ", kernel_h)

        if output_h < 1 or output_w < 1:
            raise RuntimeError("Output size is too small ", "outW= ", \
                               output_w, "outH= ", output_h)

    def _calc_out_size_and_pad(self):
        """
        caculate output size and padding size
        -------
        pad: include pad_t, pad_b, pad_l, pad_r
        output_h: out_size in h direction
        output_w: out_size in w direction
        """
        output_h = self._pooling_output_shape(self.input_h, self.kernel_h,
                                              self.pad_h, self.stride_h,
                                              self.dilation_h, self.ceil_mode)

        output_w = self._pooling_output_shape(self.input_w, self.kernel_w,
                                              self.pad_w, self.stride_w,
                                              self.dilation_w, self.ceil_mode)

        self._pool2d_shape_check(self.kernel_h, self.kernel_w, self.stride_h,
                                 self.stride_w, self.pad_h, self.pad_w,
                                 self.dilation_h, self.dilation_w, output_h,
                                 output_w)

        if self.ceil_mode == False:
            pad_t = self.pad_h
            pad_b = self.pad_h
            pad_l = self.pad_w
            pad_r = self.pad_w
        else:
            pad_t = self.pad_h
            pad_b = self.pad_h + self.stride_h - 1
            pad_l = self.pad_w
            pad_r = self.pad_w + self.stride_w - 1

        pad = (pad_l, pad_r, pad_t, pad_b)

        return pad, output_h, output_w

    # pylint: disable=too-many-branches
    def _calc_cut_h_size_fun(self, need_cut=False):
        """
        caculate cut_h size

        Parameters
        ----------
        need_cut :bool
            if need cut

        Returns
        -------
        cut_h_size: cut size
        cut_stride: cut stride
        fh_loop: loop number
        """
        img2col_w = self.kernel_h * self.kernel_w * self.c0_size
        img2col_h = UB_SIZE // 2 // (img2col_w * 2 + (32 * 5))
        if self.kernel_h >= self.stride_h:
            cut_h_size = ((img2col_h //
                           ((self.input_w + self.pad[0] + self.pad[1]) //
                            self.stride_w + 1)) - 1) * self.stride_h + \
                         self.kernel_h - self.stride_h
            if cut_h_size < self.kernel_h:
                cut_h_size = self.kernel_h
            cut_stride = cut_h_size - (self.kernel_h - self.stride_h)
        else:
            cut_h_size = ((img2col_h //
                           ((self.input_w + self.pad[0] + self.pad[1]) //
                            self.stride_w + 1)) - 1) * self.stride_h
            if cut_h_size < self.kernel_h:
                cut_h_size = self.kernel_h
                cut_stride = self.stride_h
            else:
                cut_stride = cut_h_size

        if cut_h_size >= cut_stride:
            fh_loop = _ceil_div(
                ((self.input_h + self.pad[2] + self.pad[3]) - cut_h_size),
                cut_stride) + 1
        else:
            if (self.input_h + self.pad[2] + self.pad[3]) % cut_stride == 0:
                fh_loop = (self.input_h + self.pad[2] + self.pad[
                    3]) // cut_stride
            else:
                fh_loop = _ceil_div(
                    (self.input_h + self.pad[2] + self.pad[3]),
                    cut_stride)

        if cut_h_size * self.input_w * self.c0_size * 2 > L1_SIZE:
            need_cut = True

        if need_cut:
            cut_h_size = self.kernel_h
            cut_stride = self.stride_h
            if cut_h_size >= cut_stride:
                fh_loop = _ceil_div(
                    ((self.input_h + self.pad[2] + self.pad[3]) - cut_h_size),
                    cut_stride) + 1
            else:
                if (self.input_h + self.pad[2] + self.pad[3]) % \
                        cut_stride == 0:
                    fh_loop = (self.input_h + self.pad[2] + self.pad[
                        3]) // cut_stride
                else:
                    fh_loop = _ceil_div(
                        (self.input_h + self.pad[2] + self.pad[3]),
                        cut_stride)
        """
        if self.padding == "VALID":
            if cut_h_size == self.kernel_h:
                if cut_h_size >= cut_stride:
                    fh_loop = (self.input_h - cut_h_size + cut_stride) // \
                        cut_stride
                else:
                    fh_loop = self.input_h // cut_stride
            else:
                if cut_h_size >= cut_stride:
                    fh_loop = (self.input_h - cut_h_size + cut_stride) // \
                        cut_stride
                else:
                    fh_loop = self.input_h // cut_stride
                if (self.input_h - cut_stride * fh_loop) >= self.kernel_h:
                    fh_loop = fh_loop + 1
        """
        return int(cut_h_size), int(cut_stride), int(fh_loop)

    def _calc_cut_w_size_fun(self):
        """
        caculate cut_w size

        Parameters
        ----------
        none

        Returns
        -------
        cut_w_size: cut size
        cut_w_stride: cut stride
        fw_loop: loop number
        """
        img2col_w = self.kernel_h * self.kernel_w * 16
        img2col_h = UB_SIZE / 2 / (img2col_w * 2 + (32 * 5))
        if self.kernel_w >= self.stride_w:
            cut_w_size = (img2col_h // 1 - 1) * self.stride_w + self.kernel_w - \
                         self.stride_w
            cut_w_stride = cut_w_size - (self.kernel_w - self.stride_w)
        else:
            cut_w_size = (img2col_h // 1 - 1) * self.stride_w
            cut_w_stride = cut_w_size

        if cut_w_size < self.kernel_w:
            raise RuntimeError(
                "cutC0 is needed and this scene is not supported")

        if cut_w_size >= cut_w_stride:
            fw_loop = _ceil_div(
                ((self.input_w + self.pad[0] + self.pad[1]) - cut_w_size),
                cut_w_stride) + 1
        else:
            if (self.input_w + self.pad[0] + self.pad[1]) % cut_w_stride == 0:
                fw_loop = (self.input_w + self.pad[0] + self.pad[1]) // \
                          cut_w_stride
            else:
                fw_loop = _ceil_div(
                    (self.input_w + self.pad[0] + self.pad[1]),
                    cut_w_stride)
        return int(cut_w_size), int(cut_w_stride), int(fw_loop)

    def _calc_max_fun(self, data_input, data_input_ub, index_w, index_h):
        """
        caculate max of data_input

        Parameters
        ----------
        data_input: input data
        data_input_ub: input data in ub
        index_w: input size in w direction
        index_h: input size in h direction

        Returns
        -------
        data_input: output tensor
        """
        self.tik_instance.vmax(
            MASK, data_input[index_h * 256], data_input[index_h * 256],
            data_input_ub[index_w * 256 + index_h * self.fmap_w * 256],
            REPEAT_2, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0, DSTSTRIDEM1,
            SRC0STRIDEM1, SRC1STRIDEM1)
        return data_input

    def _calc_mask_fun(self, data_input_max, data_input_ub, index_w, index_h,
                       fmap_h_num, mask_ub):
        """
        caculate mask of data_input_max

        Parameters
        ----------
        data_input_max: max value in input data
        data_input_ub: input data in ub
        index_w: index of w, along to kernel, 3x3
        index_h: index of h, alogn to output, 6x6
        fmap_h_num: num of fmap in h
        mask_ub: mask in ub, 3 x 3 x 3 x 16

        Returns
        -------
        mask_ub: mask in ub
        """
        self.tik_instance.vcmpv_eq(mask_ub[index_w * fmap_h_num * 16 +
                                           index_h * 16],
                                   data_input_ub[index_w * 256 + index_h *
                                                 self.fmap_w * 256],
                                   data_input_max[index_h * 256], REPEAT_2,
                                   SRC0STRIDEM0, SRC1STRIDEM0, SRC0STRIDEM1,
                                   SRC1STRIDEM1)
        return mask_ub

    def _calc_max_and_mask(self, fmap_h_num, fmap_ub, data_x_max,
                           mask_ub, fmap_cut_w_num=0,
                           fmap_h_tail_num=0):
        """
        caculate max and mask of data_input

        Parameters
        ----------
        exampel for: 1x12x12x16
        fmap_h_num: num of fmap_h, 3
        fmap_ub: fmap in ub, 48x3x3x16
        data_x_max: max value in input data, 48x16
        mask_ub: mask in ub, 3x3x3x16
        fmap_cut_w_num: cut number of w, default as 0
        fmap_h_tail_num: num of h tail, default as 0

        Returns
        -------
        data_input_ub: output tensor
        """
        scalar_repeat_times = int(fmap_h_num * 2)
        repeat_times = _ceil_div(scalar_repeat_times, 254)
        # dup  output max with a given value
        if scalar_repeat_times > 255:
            with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                with self.tik_instance.if_scope(repeat_index !=
                                                (repeat_times - 1)):
                    self.tik_instance.vector_dup(
                        MASK, data_x_max[repeat_index * 254 * 128],
                        MIN_VALUE_FP16,
                        254, DSTSTRIDEM0, SRC0STRIDEM1)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(
                        MASK, data_x_max[repeat_index * 254 * 128],
                        MIN_VALUE_FP16,
                        (scalar_repeat_times - repeat_index * 254),
                        DSTSTRIDEM0, SRC0STRIDEM1)
        else:
            self.tik_instance.vector_dup(MASK, data_x_max, MIN_VALUE_FP16,
                                         scalar_repeat_times, DSTSTRIDEM0,
                                         SRC0STRIDEM1)
        # cal max, iter ker_h x ker_w x 3:ceil_div(outh x outw, 16)
        with self.tik_instance.for_range(0, self.fmap_w) as index_w:
            with self.tik_instance.for_range(0, fmap_h_num) as index_h:
                # the first 128
                data_x_max = self._calc_max_fun(data_x_max, fmap_ub,
                                                index_w, index_h)

        # calc mask indices
        with self.tik_instance.for_range(0, self.fmap_w) as index_w:
            with self.tik_instance.for_range(0, fmap_h_num) as index_h:
                if fmap_cut_w_num == 0:
                    if fmap_h_tail_num == 0:
                        mask_ub = self._calc_mask_fun(data_x_max,
                                                      fmap_ub,
                                                      index_w, index_h,
                                                      fmap_h_num, mask_ub)
                    else:
                        mask_ub = self._calc_mask_fun(data_x_max,
                                                      fmap_ub,
                                                      index_w, index_h,
                                                      fmap_h_tail_num, mask_ub)
                else:
                    mask_ub = self._calc_mask_fun(data_x_max, fmap_ub,
                                                  index_w, index_h,
                                                  fmap_cut_w_num,
                                                  mask_ub)

    def _remove_repeated_fun(self, mask_ub, fmap_cut_h=0,
                             fmap_cut_w=0, fmap_tail_w=0,
                             fmap_tail_h=0):
        """
        caculate max and mask of data_input

        Parameters
        ----------
        fmap_h: size of fmap_h
        mask_ub: mask in ub
        fmap_cut_h: size of fmap_cut_h
        fmap_cut_w: size of fmap_cut_w
        fmap_tail_w: size of fmap_tail_w
        fmap_tail_h: size of tail_h

        Returns
        -------
        data_input_ub: output tensor
        """
        if fmap_cut_h != 0:
            if fmap_cut_w != 0:
                fmap_h_num = _ceil_div(fmap_cut_w, 16)
            else:
                fmap_h_num = _ceil_div(fmap_cut_h, 16)
        else:
            fmap_h_num = _ceil_div(self.fmap_h, 16)

        mask_or_shape_ub = (fmap_h_num, 16)
        mask_or = self.tik_instance.Tensor(
            "uint16", mask_or_shape_ub, name="mask_or", scope=tik.scope_ubuf)
        mask_not = self.tik_instance.Tensor(
            "uint16", mask_or_shape_ub, name="mask_not", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.fmap_w) as index_w:
            with self.tik_instance.if_scope(index_w > 0):
                if fmap_tail_w == 0:
                    if fmap_tail_h == 0:
                        self.tik_instance.vor(
                            16, mask_or[0],
                            mask_ub[index_w * fmap_h_num * 16],
                            mask_or[0],
                            fmap_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0)
                        self.tik_instance.vand(
                            16, mask_ub[index_w * fmap_h_num * 16],
                            mask_not[0],
                            mask_ub[index_w * fmap_h_num * 16],
                            fmap_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0)
                    else:
                        fmap_tail_num = _ceil_div(fmap_tail_h, 16)
                        self.tik_instance.vor(
                            16, mask_or[0],
                            mask_ub[index_w * fmap_tail_num * 16],
                            mask_or[0],
                            fmap_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0)
                        self.tik_instance.vand(
                            16, mask_ub[index_w * fmap_tail_num * 16],
                            mask_not[0],
                            mask_ub[index_w * fmap_tail_num * 16],
                            fmap_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0,
                            SRC1STRIDEM0)
                else:
                    fmap_tail_num = _ceil_div(fmap_tail_w, 16)
                    self.tik_instance.vor(
                        16, mask_or[0],
                        mask_ub[index_w * fmap_tail_num * 16],
                        mask_or[0],
                        fmap_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                        SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                    self.tik_instance.vand(
                        16, mask_ub[index_w * fmap_tail_num * 16],
                        mask_not[0],
                        mask_ub[index_w * fmap_tail_num * 16],
                        fmap_h_num, DSTSTRIDEM0, SRC0STRIDEM0,
                        SRC1STRIDEM0, DSTSTRIDEM0, SRC0STRIDEM0, SRC1STRIDEM0)
                self.tik_instance.vnot(16, mask_not[0], mask_or[0],
                                       fmap_h_num, SRC0STRIDEM0,
                                       SRC1STRIDEM0, SRC0STRIDEM0,
                                       SRC1STRIDEM0)
            with self.tik_instance.else_scope():
                self.tik_instance.vnot(16, mask_not[0], mask_ub[0],
                                       fmap_h_num, SRC0STRIDEM0,
                                       SRC1STRIDEM0, SRC0STRIDEM0,
                                       SRC1STRIDEM0)
                self.tik_instance.data_move(mask_or[0], mask_ub[0], 0, 1,
                                            fmap_h_num, 0, 0)

    def _dup_mask_fun(self, mask_ub, mask_shape_ub):
        """
         caculate max and mask of data_input

         Parameters
         ----------
         mask_ub: mask in ub
         mask_shape_ub: shape of mask_ub

         Returns
         -------
         none
         """
        scalar_repeat_times = mask_shape_ub[2]
        repeat_times = _ceil_div(scalar_repeat_times, 240)
        # dup 8*blocks init 1 into a buffer:
        if scalar_repeat_times > 240:
            with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                with self.tik_instance.if_scope(repeat_index !=
                                                (repeat_times - 1)):
                    self.tik_instance.vector_dup(
                        MASK, mask_ub[repeat_index * 240 * 16],
                        65535, 30, DSTSTRIDEM0, SRC0STRIDEM1)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(
                        16, mask_ub[repeat_index * 240 * 16],
                        65535, (scalar_repeat_times - repeat_index * 240),
                        DSTSTRIDEM0, DSTSTRIDEM0)
        else:
            self.tik_instance.vector_dup(16, mask_ub, 65535,
                                         scalar_repeat_times, DSTSTRIDEM0,
                                         DSTSTRIDEM0)


# pylint: disable=invalid-name
def max_pool_with_argmax(x, y, argmax, ksize, strides,
                         pads, dtype=DT_INT32, dilation=(1, 1, 1, 1), ceil_mode=False,
                         kernel_name="max_pool_with_argmaxv2"):
    """
    implementation of max_pool_with_argmax for pytorch and return the \
    tik instance
    :param x: dict of shape and dtype of the input x
    :param y: dict of shape and dtype of the output y
    :param argmax: dict of shape and dtype of the output argmax
    :param ksize: the size of the window to take a max over
    :param strides: the stride of the window
    :param pads: implicit zero padding to be added on both sides
    :param dilation: a parameter that controls the stride of elements \
                     in the window
    :param ceil_mode: when True, will use ceil instead of floor to compute \
                      the output shape
    :param dtype: input data type, only support int32 or int64
    :param kernel_name: the kernel's name
    :return: tik_instance
    """
    _check_param(x, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
    max_pool_grad = MaxPoolWithargmaxPytorch(x, ksize, strides, pads, dtype,
                                             dilation, ceil_mode, kernel_name)
    return max_pool_grad.tik_instance_function(kernel_name)
