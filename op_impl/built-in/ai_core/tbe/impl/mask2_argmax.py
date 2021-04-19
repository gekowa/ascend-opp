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
mask2_argmax
"""
from te import tik
from topi.cce import util
from te import platform as tbe_platform

# define dilation size
DILATION = 1
UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)


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

# pylint: disable=too-many-arguments,unused-argument
@util.check_input_type(dict, dict, dict, (list, tuple), (list, tuple), str, (list, tuple), str)
def mask2_argmax(input_x, mask, argmax, ksize, strides, padding, originshape,
                 kernel_name="mask2_argmax"):
    """
    the main function of the Mask2Argmax
    Parameters
    ----------
    input_x: input of maxpool, useless for Mask2Argmax
    mask: output mask of maxpool
    argmax:output of Mask2Argmax
    ksize: kernel or windows size,minimum length is 4,
           just like [1, poolingWindowH, poolingWindowW, 1]
    strides: stride , minimum length is 4, just like [1, poolingStrideH, poolingStrideW, 1]
    padding: pad mode, just support "SAME" or "VALID"
    originshape: originshape of input
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    """
    argmax_reslut = Mask2Argmax(input_x, mask, ksize, strides, padding, originshape)
    return argmax_reslut.tik_instance_function(kernel_name)


# pylint: disable=too-many-instance-attributes
class Mask2Argmax():
    """
       Function: use to finish Mask2Argmax main functions
       Modify : 2020-6-16
    """
    def __init__(self, input_x, mask, ksize, strides, padding, originshape):
        """
        init Mask2Argmax parameters

        Parameters
        ----------
        input_x: dict
            shape and datatype
        ksize: list or tuple
            The size of the window for each dimension of the input tensor.
        strides: list or tuple
            The stride of the sliding window of the input tensor.
        padding: str
            The type of padding algorithm to use.
        originshape: list or tuple
            The origin shape of the input tensor.

        Returns
        -------
        None
        """
        self.input_shape = input_x.get("shape")
        self.input_dtype = input_x.get("dtype").lower()
        self.tik_instance = tik.Tik()
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.batch_size = self.input_shape[0]
        self.c1_size = self.input_shape[1]
        self.in_size_h = self.input_shape[2]
        self.in_size_w = self.input_shape[3]
        self.c_block_size = self.input_shape[4]
        self.c_size = originshape[3]

        self.window_h = self.ksize[1]
        self.window_w = self.ksize[2]
        self.stride_h = self.strides[1]
        self.stride_w = self.strides[2]
        self.nc1 = self.batch_size * self.c1_size

        # caculate pad and output size
        self.pad, self.out_size_h, self.out_size_w = \
            self.calc_out_size_and_pad()
        # output_shape
        self.fmap_img2col_h = self.out_size_h * self.out_size_w
        self.fmap_img2col_w = self.window_h * self.window_w
        self.fmap_img2col_h_num = _ceil_div(self.fmap_img2col_h,
                                            self.c_block_size)
        # famp is NC1HWC0 format
        self.output_argmax_shape = (self.batch_size, self.c1_size, self.out_size_h,
                                    self.out_size_w, self.c_block_size)

        self.mask_gm_shape = (
            self.batch_size*self.c1_size*self.fmap_img2col_w*(
                self.fmap_img2col_h_num + 1) * self.c_block_size,)
        # input and output
        self.output_argmax_gm = self.tik_instance.Tensor("float32",
                                                         self.output_argmax_shape,
                                                         name="output_argmax_gm",
                                                         scope=tik.scope_gm)
        self.data_input_gm = self.tik_instance.Tensor(self.input_dtype,
                                                      self.input_shape,
                                                      name="data_input_gm",
                                                      scope=tik.scope_gm)
        self.input_mask_gm = self.tik_instance.Tensor("uint16",
                                                      self.mask_gm_shape,
                                                      name="input_mask_gm",
                                                      scope=tik.scope_gm)

        # use scalar
        self.stride_h_scalar = self.tik_instance.Scalar(dtype='float32')
        self.stride_w_scalar = self.tik_instance.Scalar(dtype='float32')
        self.in_size_w_scalar = self.tik_instance.Scalar(dtype="float32")
        self.c_size_scalar = self.tik_instance.Scalar(dtype="float32")

        self.stride_h_scalar.set_as(self.stride_h)
        self.stride_w_scalar.set_as(self.stride_w)
        self.in_size_w_scalar.set_as(self.in_size_w)
        self.c_size_scalar.set_as(self.c_size)

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
        core_counts = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        # caculate block number
        nc1 = self.batch_size * self.c1_size
        nc1_size = nc1 // core_counts + (1 if nc1 % core_counts > 0 else 0)
        if (nc1 % core_counts == 0) or (nc1 % nc1_size == 0):
            is_same_core = 0
        else:
            is_same_core = 1

        block_dim = nc1 // nc1_size + (0 if nc1 // core_counts == 0
                                       else is_same_core)
        if self.out_size_h * self.out_size_w * 16 * 4 * 5 < UB_SIZE:
            with self.tik_instance.for_range(0, block_dim, block_num=block_dim) \
                    as block_index:
                with self.tik_instance.if_scope(block_index != block_dim - 1):
                    with self.tik_instance.for_range(0, nc1_size) as nc1_index:
                        self.fun_no_cut(block_index, nc1_size, nc1_index)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(
                            0, nc1 - (block_dim - 1)*nc1_size) as nc1_index:
                        self.fun_no_cut(block_index, nc1_size, nc1_index)
        else:
            with self.tik_instance.for_range(0, block_dim, block_num=block_dim) \
                    as block_index:
                with self.tik_instance.if_scope(block_index != block_dim - 1):
                    with self.tik_instance.for_range(0, nc1_size) as nc1_index:
                        self.fun_cut_mask(block_index, nc1_size, nc1_index)
                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(
                            0, nc1 - (block_dim - 1)*nc1_size) as nc1_index:
                        self.fun_cut_mask(block_index, nc1_size, nc1_index)
        images_buf = self.tik_instance.Tensor(self.input_dtype, (16,),
                                              name="one_value_buf",
                                              scope=tik.scope_ubuf)
        self.tik_instance.data_move(images_buf[0], self.data_input_gm[0],
                                    0, 1, 1, 0, 0)
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=(self.data_input_gm, self.input_mask_gm),
                                   outputs=(self.output_argmax_gm))
        return self.tik_instance

    def calc_out_size_and_pad(self):
        """
        caculate output size and padding size

        Parameters
        ----------
        none

        Returns
        -------
        pad: include pad_t, pad_b, pad_l, pad_r
        out_size_h: out_size in h direction
        out_size_w: out_size in w direction
        """
        # pad_l, pad_r, pad_t, pad_b is for pad on the left, right, top, bottom
        pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0

        if self.padding == "SAME":
            # Hout = ceil(Hi, Sh), Wout = ceil(Wi, Sw)
            out_size_h = (self.in_size_h + self.stride_h - 1) // self.stride_h
            out_size_w = (self.in_size_w + self.stride_w - 1) // self.stride_w

            # get total pad rows or pad columns
            pad_rows = (out_size_h - 1) * self.stride_h + \
                       ((self.window_h - 1) * DILATION + 1) - self.in_size_h
            pad_cols = (out_size_w - 1) * self.stride_w + \
                       ((self.window_w - 1) * DILATION + 1) - self.in_size_w

            # pad_rows and pad_columns is odd or even number
            if pad_rows % 2 == 0:
                pad_t = pad_rows // 2
                pad_b = pad_rows // 2
            else:
                pad_t = pad_rows // 2
                pad_b = pad_rows - pad_t

            if pad_cols % 2 == 0:
                pad_l = pad_cols // 2
                pad_r = pad_cols // 2
            else:
                pad_l = pad_cols // 2
                pad_r = pad_cols - pad_l

            if pad_t < 0:
                pad_t = 0

            if pad_b < 0:
                pad_b = 0

            if pad_l < 0:
                pad_l = 0

            if pad_r < 0:
                pad_r = 0

        # caculate output size in VALID mode
        if self.padding == "VALID":
            # Hout = ceil(Hi - Fh + 1, Sh), Wout = ceil(Wi - Fw + 1, Sw)
            out_size_h = (self.in_size_h - self.window_h + 1 +
                          (self.stride_h - 1)) // self.stride_h
            out_size_w = (self.in_size_w - self.window_w + 1 +
                          (self.stride_w - 1)) // self.stride_w
        pad = (pad_l, pad_r, pad_t, pad_b)

        return pad, out_size_h, out_size_w

    def dup_one_repeat(self, dup_input_ub, index, dtype):
        """
        The fun just for dup ub
        """
        data_vsel_scalar = self.tik_instance.Scalar(dtype)
        data_vsel_scalar.set_as(index)
        if dup_input_ub.shape[0] > 255:
            repeat_dups = _ceil_div(dup_input_ub.shape[0], 255)
            with self.tik_instance.for_range(0, repeat_dups) as repeat_index:
                with self.tik_instance.if_scope(repeat_index !=
                                                (repeat_dups - 1)):
                    self.tik_instance.vector_dup(
                        64, dup_input_ub[repeat_index * 255 * 64],
                        data_vsel_scalar,
                        255,
                        1,
                        8)
                with self.tik_instance.else_scope():
                    self.tik_instance.vector_dup(
                        64, dup_input_ub[repeat_index * 255 * 64],
                        data_vsel_scalar,
                        (repeat_dups - repeat_index * 255),
                        1,
                        8)
        else:
            self.tik_instance.vector_dup(64,
                                         dup_input_ub[0],
                                         data_vsel_scalar,
                                         dup_input_ub.shape[0],
                                         1,
                                         8)

    def expr2tensor(self, expr, int32_ub, float16_ub, float32_ub):
        """
        Transform tik.Expr to float32 tensor
        :param expr:
        :param int32_ub:
        :param float16_ub:
        :param float32_ub:
        :return:
        """
        self.tik_instance.vector_dup(16, int32_ub, expr, 1, 1, 1)
        self.tik_instance.vec_conv(16, '', float16_ub, int32_ub, 1, 1, 1, 1.0)
        self.tik_instance.vec_conv(16, '', float32_ub, float16_ub, 1, 1, 1)

    def calc_argmax(self, out_h_index_ub3, out_w_index_ub3, window_h_ub3, window_w_ub3, reg_argmax_ub):
        """
        calculate the value of reg_argmax
        :param out_h_index_ub3:
        :param out_w_index_ub3:
        :param window_h_ub3:
        :param window_w_ub3:
        :param reg_argmax_ub:
        :return:
        """
        self.tik_instance.vec_muls(16, out_h_index_ub3, out_h_index_ub3, self.stride_h_scalar, 1, 1, 1)
        self.tik_instance.vec_add(16, out_h_index_ub3, out_h_index_ub3, window_h_ub3, 1, 1, 1, 1)
        self.tik_instance.vec_muls(16, out_h_index_ub3, out_h_index_ub3, self.in_size_w_scalar, 1, 1, 1)
        self.tik_instance.vec_muls(16, out_h_index_ub3, out_h_index_ub3, self.c_size_scalar, 1, 1, 1)

        self.tik_instance.vec_muls(16, out_w_index_ub3, out_w_index_ub3, self.stride_w_scalar, 1, 1, 1)
        self.tik_instance.vec_add(16, out_w_index_ub3, out_w_index_ub3, window_w_ub3, 1, 1, 1, 1)
        self.tik_instance.vec_muls(16, out_w_index_ub3, out_w_index_ub3, self.c_size_scalar, 1, 1, 1)

        self.tik_instance.vec_add(16, reg_argmax_ub, out_h_index_ub3, out_w_index_ub3, 1, 1, 1, 1)

    # pylint: disable=too-many-statements
    def fun_no_cut(self, block_index, nc1_size, nc1_index):
        """
        funtion no need cut

        Parameters
        ----------
        block_index: index of block
        nc1_size: n*c1
        nc1_index: index of nc1

        Returns
        -------
        none
        """
        c1_value = (block_index * nc1_size + nc1_index) % self.c1_size
        cmp_num = _ceil_div(self.fmap_img2col_h, 8)
        data_vsel_ub_zero = self.tik_instance.Tensor("float16", (128,),
                                                     name="data_vsel_ub_zero",
                                                     scope=tik.scope_ubuf)
        data_vsel_ub_one = self.tik_instance.Tensor("float16", (128,),
                                                    name="data_vsel_ub_one",
                                                    scope=tik.scope_ubuf)
        out_h_index_ub = self.tik_instance.Tensor("int32", (16,), name='out_h_index_ub',
                                                  scope=tik.scope_ubuf)
        out_h_index_ub2 = self.tik_instance.Tensor("float16", (16,), name="out_h_index_ub2",
                                                   scope=tik.scope_ubuf)
        out_h_index_ub3 = self.tik_instance.Tensor("float32", (16,), name="out_h_index_ub3",
                                                   scope=tik.scope_ubuf)
        out_w_index_ub = self.tik_instance.Tensor("int32", (16,), name='out_w_index_ub',
                                                  scope=tik.scope_ubuf)
        out_w_index_ub2 = self.tik_instance.Tensor("float16", (16,), name="out_w_index_ub2",
                                                   scope=tik.scope_ubuf)
        out_w_index_ub3 = self.tik_instance.Tensor("float32", (16,), name="out_w_index_ub3",
                                                   scope=tik.scope_ubuf)
        window_h_ub = self.tik_instance.Tensor("int32", (16,), name='window_h_ub',
                                               scope=tik.scope_ubuf)
        window_h_ub2 = self.tik_instance.Tensor("float16", (16,), name='window_h_ub2',
                                                scope=tik.scope_ubuf)
        window_h_ub3 = self.tik_instance.Tensor("float32", (16,), name='window_h_ub3',
                                                scope=tik.scope_ubuf)
        window_w_ub = self.tik_instance.Tensor("int32", (16,), name='window_w_ub',
                                               scope=tik.scope_ubuf)
        window_w_ub2 = self.tik_instance.Tensor("float16", (16,), name='window_w_ub2',
                                                scope=tik.scope_ubuf)
        window_w_ub3 = self.tik_instance.Tensor("float32", (16,), name='window_w_ub3',
                                                scope=tik.scope_ubuf)
        reg_argmax_ub = self.tik_instance.Tensor("float32", (16,), name='reg_argmax_ub',
                                                 scope=tik.scope_ubuf)
        reg_argmax = self.tik_instance.Scalar(dtype="float32")
        self.tik_instance.vector_dup(128, data_vsel_ub_zero, 0, 1, 1, 8)
        self.tik_instance.vector_dup(128, data_vsel_ub_one, 1, 1, 1, 8)
        data_out_ub = self.tik_instance.Tensor("float32", (cmp_num*2, 64),
                                               name="data_out_ub",
                                               scope=tik.scope_ubuf)
        self.dup_one_repeat(data_out_ub, 0, "float32")
        with self.tik_instance.for_range(0, self.window_h) as window_h_index:
            with self.tik_instance.for_range(0, self.window_w) as window_w_index:
                mask_ub_shape = ((self.fmap_img2col_h_num + 1) * self.c_block_size,)
                data_mask_ub = self.tik_instance.Tensor("uint16",
                                                        mask_ub_shape,
                                                        name="data_mask_ub",
                                                        scope=tik.scope_ubuf)
                data_vsel_ub = self.tik_instance.Tensor("float16", (cmp_num, 128),
                                                        name="data_vsel_ub",
                                                        scope=tik.scope_ubuf)
                data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (cmp_num*2, 64),
                                                             name="data_vsel_ub",
                                                             scope=tik.scope_ubuf)
                self.tik_instance.\
                    data_move(data_mask_ub[0],
                              self.input_mask_gm[(block_index *
                                                  nc1_size + nc1_index) *
                                                 self.fmap_img2col_w * (
                                                     self.fmap_img2col_h_num + 1) *
                                                 self.c_block_size + (
                                                     window_h_index *
                                                     self.window_w +
                                                     window_w_index) * (
                                                         self.fmap_img2col_h_num +
                                                         1) * self.c_block_size],
                              0,
                              1,
                              (self.fmap_img2col_h_num + 1),
                              0,
                              0)
                data_out_tmp = self.tik_instance.Tensor("float32", (cmp_num*2, 64),
                                                        name="data_out_tmp",
                                                        scope=tik.scope_ubuf)
                self.dup_one_repeat(data_out_tmp, 1, "float32")

                with self.tik_instance.for_range(0, cmp_num) as cmp_index:
                    cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                        data_mask_ub[cmp_index * 8])
                    self.tik_instance.vsel(128, 0,
                                           data_vsel_ub[cmp_index * 128],
                                           cmpmask,
                                           data_vsel_ub_one[0],
                                           data_vsel_ub_zero[0],
                                           1,
                                           1, 1, 1,
                                           8, 8, 8)
                if cmp_num*2 < 255:
                    self.tik_instance.vconv(64, '', data_vsel_ub_fp32[0],
                                            data_vsel_ub[0], cmp_num*2,
                                            1, 1, 8, 4)
                else:
                    repeat_times = _ceil_div(cmp_num*2, 255)
                    with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                        with self.tik_instance.if_scope(repeat_index !=
                                                        (repeat_times - 1)):
                            self.tik_instance.vconv(64,
                                                    '',
                                                    data_vsel_ub_fp32[repeat_index * 255 * 64],
                                                    data_vsel_ub[repeat_index * 255 * 64], 255,
                                                    1, 1, 8, 4)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vconv(64,
                                                    '',
                                                    data_vsel_ub_fp32[repeat_index * 255 * 64],
                                                    data_vsel_ub[repeat_index * 255 * 64],
                                                    cmp_num*2 - repeat_index * 255,
                                                    1, 1, 8, 4)

                with self.tik_instance.for_range(0, self.out_size_h) as out_h_index:
                    with self.tik_instance.for_range(0, self.out_size_w) as out_w_index:
                        window_h_index_pad = window_h_index - self.pad[2]
                        window_w_index_pad = window_w_index - self.pad[0]

                        self.expr2tensor(out_h_index, out_h_index_ub, out_h_index_ub2,
                                         out_h_index_ub3)
                        self.expr2tensor(out_w_index, out_w_index_ub, out_w_index_ub2,
                                         out_w_index_ub3)
                        self.expr2tensor(window_h_index_pad, window_h_ub, window_h_ub2, window_h_ub3)
                        self.expr2tensor(window_w_index_pad, window_w_ub, window_w_ub2, window_w_ub3)

                        self.calc_argmax(out_h_index_ub3, out_w_index_ub3, window_h_ub3, window_w_ub3, reg_argmax_ub)

                        reg_argmax.set_as(reg_argmax_ub[0])

                        self.tik_instance.vmuls(16,
                                                data_out_tmp[(out_h_index * self.out_size_w +
                                                              out_w_index) * 16],
                                                data_out_tmp[(out_h_index * self.out_size_w +
                                                              out_w_index) * 16],
                                                reg_argmax,
                                                1,
                                                1, 1,
                                                8, 8)
                if cmp_num*2 < 255:
                    self.tik_instance.vmul(64, data_out_tmp[0],
                                           data_vsel_ub_fp32[0],
                                           data_out_tmp[0],
                                           cmp_num*2,
                                           1, 1, 1,
                                           8, 8, 8)
                    self.tik_instance.vadd(64,
                                           data_out_ub[0],
                                           data_out_tmp[0],
                                           data_out_ub[0],
                                           cmp_num*2,
                                           1, 1, 1,
                                           8, 8, 8)
                else:
                    repeat_times = _ceil_div(cmp_num*2, 255)
                    with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                        with self.tik_instance.if_scope(repeat_index !=
                                                        (repeat_times - 1)):
                            self.tik_instance.vmul(64, data_out_tmp[repeat_index * 255 * 64],
                                                   data_vsel_ub_fp32[repeat_index * 255 * 64],
                                                   data_out_tmp[repeat_index * 255 * 64],
                                                   255,
                                                   1, 1, 1,
                                                   8, 8, 8)
                            self.tik_instance.vadd(64,
                                                   data_out_ub[repeat_index * 255 * 64],
                                                   data_out_tmp[repeat_index * 255 * 64],
                                                   data_out_ub[repeat_index * 255 * 64],
                                                   255,
                                                   1, 1, 1,
                                                   8, 8, 8)
                        with self.tik_instance.else_scope():
                            self.tik_instance.vmul(64, data_out_tmp[repeat_index * 255 * 64],
                                                   data_vsel_ub_fp32[repeat_index * 255 * 64],
                                                   data_out_tmp[repeat_index * 255 * 64],
                                                   cmp_num*2 - repeat_index*255,
                                                   1, 1, 1,
                                                   8, 8, 8)
                            self.tik_instance.vadd(64,
                                                   data_out_ub[repeat_index * 255 * 64],
                                                   data_out_tmp[repeat_index * 255 * 64],
                                                   data_out_ub[repeat_index * 255 * 64],
                                                   cmp_num*2 - repeat_index*255,
                                                   1, 1, 1,
                                                   8, 8, 8)
        index_16 = self.tik_instance.Tensor("float32", (64,),
                                            name="index_16",
                                            scope=tik.scope_ubuf)
        tmp_w = self.tik_instance.Tensor("int32", (16,), name="tmp_w", scope=tik.scope_ubuf)
        tmp_w2 = self.tik_instance.Tensor("float16", (16,), name="tmp_w2", scope=tik.scope_ubuf)
        tmp_w3 = self.tik_instance.Tensor("float32", (16,), name="tmp_w3", scope=tik.scope_ubuf)
        reg_w_tmp = self.tik_instance.Scalar(dtype="float32")

        order = 0
        while order < 64:
            tmp_scalar = (order % 16) + c1_value * 16

            self.tik_instance.vector_dup(16, tmp_w, tmp_scalar, 1, 1, 1)
            self.tik_instance.vec_conv(16, '', tmp_w2, tmp_w, 1, 1, 1, 1.0)
            self.tik_instance.vec_conv(16, '', tmp_w3, tmp_w2, 1, 1, 1)
            reg_w_tmp.set_as(tmp_w3[0])
            index_16[order].set_as(reg_w_tmp)
            order = order + 1

        if cmp_num*2 < 255:
            self.tik_instance.vadd(64,
                                   data_out_ub[0],
                                   index_16[0],
                                   data_out_ub[0],
                                   cmp_num*2,
                                   1, 1, 1,
                                   8, 0, 8)
        else:
            repeat_times = _ceil_div(cmp_num*2, 255)
            with self.tik_instance.for_range(0, repeat_times) as repeat_index:
                with self.tik_instance.if_scope(repeat_index !=
                                                (repeat_times - 1)):
                    self.tik_instance.vadd(64,
                                           data_out_ub[repeat_index * 255 * 64],
                                           index_16[0],
                                           data_out_ub[repeat_index * 255 * 64],
                                           255,
                                           1, 1, 1,
                                           8, 0, 8)
                with self.tik_instance.else_scope():
                    self.tik_instance.vadd(64,
                                           data_out_ub[repeat_index * 255 * 64],
                                           index_16[0],
                                           data_out_ub[repeat_index * 255 * 64],
                                           cmp_num*2 - repeat_index * 255,
                                           1, 1, 1,
                                           8, 0, 8)
        self.tik_instance.data_move(self.output_argmax_gm[(block_index * nc1_size + nc1_index) *
                                                          self.fmap_img2col_h *
                                                          self.c_block_size],
                                    data_out_ub[0],
                                    0,
                                    1,
                                    self.fmap_img2col_h * 2,
                                    0,
                                    0)

    # pylint: disable=too-many-statements
    def fun_cut_mask(self, block_index, nc1_size, nc1_index):
        """
        funtion need cut

        Parameters
        ----------
        block_index: index of block
        nc1_size: n*c1
        nc1_index: index of nc1

        Returns
        -------
        none
        """
        cut_num = _ceil_div(self.fmap_img2col_h, 800)
        c1_value = (block_index * nc1_size + nc1_index) % self.c1_size

        out_h_index_ub = self.tik_instance.Tensor("int32", (16,), name='out_h_index_ub',
                                                  scope=tik.scope_ubuf)
        out_h_index_ub2 = self.tik_instance.Tensor("float16", (16,), name="out_h_index_ub2",
                                                   scope=tik.scope_ubuf)
        out_h_index_ub3 = self.tik_instance.Tensor("float32", (16,), name="out_h_index_ub3",
                                                   scope=tik.scope_ubuf)
        out_w_index_ub = self.tik_instance.Tensor("int32", (16,), name='out_w_index_ub',
                                                  scope=tik.scope_ubuf)
        out_w_index_ub2 = self.tik_instance.Tensor("float16", (16,), name="out_w_index_ub2",
                                                   scope=tik.scope_ubuf)
        out_w_index_ub3 = self.tik_instance.Tensor("float32", (16,), name="out_w_index_ub3",
                                                   scope=tik.scope_ubuf)
        window_h_ub = self.tik_instance.Tensor("int32", (16,), name='window_h_ub',
                                               scope=tik.scope_ubuf)
        window_h_ub2 = self.tik_instance.Tensor("float16", (16,), name='window_h_ub2',
                                                scope=tik.scope_ubuf)
        window_h_ub3 = self.tik_instance.Tensor("float32", (16,), name='window_h_ub3',
                                                scope=tik.scope_ubuf)
        window_w_ub = self.tik_instance.Tensor("int32", (16,), name='window_w_ub',
                                               scope=tik.scope_ubuf)
        window_w_ub2 = self.tik_instance.Tensor("float16", (16,), name='window_w_ub2',
                                                scope=tik.scope_ubuf)
        window_w_ub3 = self.tik_instance.Tensor("float32", (16,), name='window_w_ub3',
                                                scope=tik.scope_ubuf)
        reg_argmax_ub = self.tik_instance.Tensor("float32", (16,), name='reg_argmax_ub',
                                                 scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, cut_num) as cut_index:
            with self.tik_instance.if_scope(cut_index != cut_num - 1):
                cmp_num = 100
                data_vsel_ub_zero = self.tik_instance.Tensor("float16", (128,),
                                                             name="data_vsel_ub_zero",
                                                             scope=tik.scope_ubuf)
                data_vsel_ub_one = self.tik_instance.Tensor("float16", (128,),
                                                            name="data_vsel_ub_one",
                                                            scope=tik.scope_ubuf)
                reg_argmax = self.tik_instance.Scalar(dtype="float32")
                self.tik_instance.vector_dup(128, data_vsel_ub_zero, 0, 1, 1, 8)
                self.tik_instance.vector_dup(128, data_vsel_ub_one, 1, 1, 1, 8)
                data_out_ub = self.tik_instance.Tensor("float32", (cmp_num*2, 64),
                                                       name="data_out_ub",
                                                       scope=tik.scope_ubuf)
                self.dup_one_repeat(data_out_ub, 0, "float32")
                with self.tik_instance.for_range(0, self.window_h) as window_h_index:
                    with self.tik_instance.for_range(0, self.window_w) as window_w_index:
                        mask_ub_shape = (800,)
                        data_mask_ub = self.tik_instance.Tensor("uint16",
                                                                mask_ub_shape,
                                                                name="data_mask_ub",
                                                                scope=tik.scope_ubuf)
                        data_vsel_ub = self.tik_instance.Tensor("float16", (cmp_num, 128),
                                                                name="data_vsel_ub",
                                                                scope=tik.scope_ubuf)
                        data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (cmp_num*2, 64),
                                                                     name="data_vsel_ub",
                                                                     scope=tik.scope_ubuf)
                        self.tik_instance. \
                            data_move(data_mask_ub[0],
                                      self.input_mask_gm[(block_index *
                                                          nc1_size +
                                                          nc1_index) *
                                                         self.fmap_img2col_w * (
                                                             self.fmap_img2col_h_num + 1) *
                                                         self.c_block_size + (
                                                             window_h_index * self.window_w +
                                                             window_w_index) * (
                                                                 self.fmap_img2col_h_num + 1) *
                                                         self.c_block_size +
                                                         cut_index * 800],
                                      0,
                                      1,
                                      50,
                                      0,
                                      0)
                        data_out_tmp = self.tik_instance.Tensor("float32", (cmp_num*2, 64),
                                                                name="data_out_tmp",
                                                                scope=tik.scope_ubuf)
                        self.dup_one_repeat(data_out_tmp, 1, "float32")

                        with self.tik_instance.for_range(0, cmp_num) as cmp_index:
                            cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                data_mask_ub[cmp_index * 8])
                            self.tik_instance.vsel(128, 0,
                                                   data_vsel_ub[cmp_index * 128],
                                                   cmpmask,
                                                   data_vsel_ub_one[0],
                                                   data_vsel_ub_zero[0],
                                                   1,
                                                   1, 1, 1,
                                                   8, 8, 8)
                        self.tik_instance.vconv(64, '', data_vsel_ub_fp32[0],
                                                data_vsel_ub[0], cmp_num*2,
                                                1, 1, 8, 4)
                        with self.tik_instance.for_range(0, 800) as cut_h_w_index:
                            out_h_index = (cut_index * 800 + cut_h_w_index) // self.out_size_w
                            out_w_index = (cut_index * 800 + cut_h_w_index) % self.out_size_w

                            window_h_index_pad = window_h_index - self.pad[2]
                            window_w_index_pad = window_w_index - self.pad[0]

                            self.expr2tensor(out_h_index, out_h_index_ub, out_h_index_ub2,
                                             out_h_index_ub3)
                            self.expr2tensor(out_w_index, out_w_index_ub, out_w_index_ub2,
                                             out_w_index_ub3)
                            self.expr2tensor(window_h_index_pad, window_h_ub, window_h_ub2, window_h_ub3)
                            self.expr2tensor(window_w_index_pad, window_w_ub, window_w_ub2, window_w_ub3)

                            self.calc_argmax(out_h_index_ub3, out_w_index_ub3, window_h_ub3, window_w_ub3,
                                             reg_argmax_ub)

                            reg_argmax.set_as(reg_argmax_ub[0])
                            self.tik_instance.vmuls(16,
                                                    data_out_tmp[cut_h_w_index * 16],
                                                    data_out_tmp[cut_h_w_index * 16],
                                                    reg_argmax,
                                                    1,
                                                    1, 1,
                                                    8, 8)
                        self.tik_instance.vmul(64, data_out_tmp[0],
                                               data_vsel_ub_fp32[0],
                                               data_out_tmp[0],
                                               cmp_num*2,
                                               1, 1, 1,
                                               8, 8, 8)
                        self.tik_instance.vadd(64,
                                               data_out_ub[0],
                                               data_out_tmp[0],
                                               data_out_ub[0],
                                               cmp_num*2,
                                               1, 1, 1,
                                               8, 8, 8)
                index_16 = self.tik_instance.Tensor("float32", (64,),
                                                    name="index_16",
                                                    scope=tik.scope_ubuf)
                tmp_w = self.tik_instance.Tensor("int32", (16,), name="tmp_w", scope=tik.scope_ubuf)
                tmp_w2 = self.tik_instance.Tensor("float16", (16,), name="tmp_w2", scope=tik.scope_ubuf)
                tmp_w3 = self.tik_instance.Tensor("float32", (16,), name="tmp_w3", scope=tik.scope_ubuf)
                reg_w_tmp = self.tik_instance.Scalar(dtype="float32")

                order = 0
                while order < 64:
                    tmp_scalar = (order % 16) + c1_value * 16

                    self.tik_instance.vector_dup(16, tmp_w, tmp_scalar, 1, 1, 1)
                    self.tik_instance.vec_conv(16, '', tmp_w2, tmp_w, 1, 1, 1, 1.0)
                    self.tik_instance.vec_conv(16, '', tmp_w3, tmp_w2, 1, 1, 1)
                    reg_w_tmp.set_as(tmp_w3[0])
                    index_16[order].set_as(reg_w_tmp)
                    order = order + 1

                self.tik_instance.vadd(64,
                                       data_out_ub[0],
                                       index_16[0],
                                       data_out_ub[0],
                                       cmp_num*2,
                                       1, 1, 1,
                                       8, 0, 8)
                self.tik_instance.data_move(self.output_argmax_gm[(block_index *
                                                                   nc1_size + nc1_index) *
                                                                  self.fmap_img2col_h *
                                                                  self.c_block_size +
                                                                  cut_index * 800 *
                                                                  self.c_block_size],
                                            data_out_ub[0],
                                            0,
                                            1,
                                            800 * 2,
                                            0,
                                            0)
            with self.tik_instance.else_scope():
                cmp_num = _ceil_div(self.fmap_img2col_h - (cut_num - 1) * 800, 8)
                data_vsel_ub_zero = self.tik_instance.Tensor("float16", (128,),
                                                             name="data_vsel_ub_zero",
                                                             scope=tik.scope_ubuf)
                data_vsel_ub_one = self.tik_instance.Tensor("float16", (128,),
                                                            name="data_vsel_ub_one",
                                                            scope=tik.scope_ubuf)
                reg_argmax = self.tik_instance.Scalar(dtype="float32")
                self.tik_instance.vector_dup(128, data_vsel_ub_zero, 0, 1, 1, 8)
                self.tik_instance.vector_dup(128, data_vsel_ub_one, 1, 1, 1, 8)
                data_out_ub = self.tik_instance.Tensor("float32", (cmp_num*2, 64),
                                                       name="data_out_ub",
                                                       scope=tik.scope_ubuf)
                self.dup_one_repeat(data_out_ub, 0, "float32")
                with self.tik_instance.for_range(0, self.window_h) as window_h_index:
                    with self.tik_instance.for_range(0, self.window_w) as window_w_index:

                        mask_ub_shape = ((self.fmap_img2col_h_num + 1) *
                                         self.c_block_size - (cut_num - 1) * 800,)
                        data_mask_ub = self.tik_instance.Tensor("uint16",
                                                                mask_ub_shape,
                                                                name="data_mask_ub",
                                                                scope=tik.scope_ubuf)
                        data_vsel_ub = self.tik_instance.Tensor("float16", (cmp_num, 128),
                                                                name="data_vsel_ub",
                                                                scope=tik.scope_ubuf)
                        data_vsel_ub_fp32 = self.tik_instance.Tensor("float32", (cmp_num*2, 64),
                                                                     name="data_vsel_ub",
                                                                     scope=tik.scope_ubuf)
                        reg_num = self.tik_instance.Scalar(dtype="int32")
                        reg_num.set_as((self.fmap_img2col_h_num + 1) - (cut_num - 1) * 50)
                        self.tik_instance. \
                            data_move(data_mask_ub[0],
                                      self.input_mask_gm[(block_index *
                                                          nc1_size + nc1_index) *
                                                         self.fmap_img2col_w * (
                                                             self.fmap_img2col_h_num + 1) *
                                                         self.c_block_size + (
                                                             window_h_index * self.window_w +
                                                             window_w_index) * (
                                                                 self.fmap_img2col_h_num + 1) *
                                                         self.c_block_size +
                                                         cut_index * 800],
                                      0,
                                      1,
                                      reg_num,
                                      0,
                                      0)
                        data_out_tmp = self.tik_instance.Tensor("float32", (cmp_num*2, 64),
                                                                name="data_out_tmp",
                                                                scope=tik.scope_ubuf)
                        self.dup_one_repeat(data_out_tmp, 1, "float32")
                        with self.tik_instance.for_range(0, cmp_num) as cmp_index:
                            cmpmask = self.tik_instance.mov_tensor_to_cmpmask(
                                data_mask_ub[cmp_index * 8])
                            self.tik_instance.vsel(128, 0,
                                                   data_vsel_ub[cmp_index * 128],
                                                   cmpmask,
                                                   data_vsel_ub_one[0],
                                                   data_vsel_ub_zero[0],
                                                   1,
                                                   1, 1, 1,
                                                   8, 8, 8)
                        self.tik_instance.vconv(64, '', data_vsel_ub_fp32[0],
                                                data_vsel_ub[0], cmp_num*2,
                                                1, 1, 8, 4)
                        with self.tik_instance. \
                                for_range(0, self.fmap_img2col_h - (cut_num - 1) * 800) \
                                as cut_h_w_index:
                            out_h_index = (cut_index * 800 + cut_h_w_index) // self.out_size_w
                            out_w_index = (cut_index * 800 + cut_h_w_index) % self.out_size_w
                            window_h_index_pad = window_h_index - self.pad[2]
                            window_w_index_pad = window_w_index - self.pad[0]

                            self.expr2tensor(out_h_index, out_h_index_ub, out_h_index_ub2,
                                             out_h_index_ub3)
                            self.expr2tensor(out_w_index, out_w_index_ub, out_w_index_ub2,
                                             out_w_index_ub3)
                            self.expr2tensor(window_h_index_pad, window_h_ub, window_h_ub2, window_h_ub3)
                            self.expr2tensor(window_w_index_pad, window_w_ub, window_w_ub2, window_w_ub3)

                            self.calc_argmax(out_h_index_ub3, out_w_index_ub3, window_h_ub3, window_w_ub3,
                                             reg_argmax_ub)

                            reg_argmax.set_as(reg_argmax_ub[0])

                            self.tik_instance.vmuls(16,
                                                    data_out_tmp[cut_h_w_index * 16],
                                                    data_out_tmp[cut_h_w_index * 16],
                                                    reg_argmax,
                                                    1,
                                                    1, 1,
                                                    8, 8)
                        self.tik_instance.vmul(64, data_out_tmp[0],
                                               data_vsel_ub_fp32[0],
                                               data_out_tmp[0],
                                               cmp_num*2,
                                               1, 1, 1,
                                               8, 8, 8)
                        self.tik_instance.vadd(64,
                                               data_out_ub[0],
                                               data_out_tmp[0],
                                               data_out_ub[0],
                                               cmp_num*2,
                                               1, 1, 1,
                                               8, 8, 8)
                index_16 = self.tik_instance.Tensor("float32", (64,),
                                                    name="index_16",
                                                    scope=tik.scope_ubuf)
                tmp_w = self.tik_instance.Tensor("int32", (16,), name="tmp_w", scope=tik.scope_ubuf)
                tmp_w2 = self.tik_instance.Tensor("float16", (16,), name="tmp_w2", scope=tik.scope_ubuf)
                tmp_w3 = self.tik_instance.Tensor("float32", (16,), name="tmp_w3", scope=tik.scope_ubuf)
                reg_w_tmp = self.tik_instance.Scalar(dtype="float32")
                order = 0
                while order < 64:
                    tmp_scalar = (order % 16) + c1_value * 16

                    self.tik_instance.vector_dup(16, tmp_w, tmp_scalar, 1, 1, 1)
                    self.tik_instance.vec_conv(16, '', tmp_w2, tmp_w, 1, 1, 1, 1.0)
                    self.tik_instance.vec_conv(16, '', tmp_w3, tmp_w2, 1, 1, 1)
                    reg_w_tmp.set_as(tmp_w3[0])
                    index_16[order].set_as(reg_w_tmp)
                    order = order + 1

                self.tik_instance.vadd(64,
                                       data_out_ub[0],
                                       index_16[0],
                                       data_out_ub[0],
                                       cmp_num*2,
                                       1, 1, 1,
                                       8, 0, 8)
                reg_move = self.tik_instance.Scalar(dtype="int32")
                reg_move.set_as((self.fmap_img2col_h - cut_index*800) * 2)
                self.tik_instance.data_move(self.output_argmax_gm[(block_index *
                                                                   nc1_size + nc1_index) *
                                                                  self.fmap_img2col_h *
                                                                  self.c_block_size +
                                                                  cut_index * 800 *
                                                                  self.c_block_size],
                                            data_out_ub[0],
                                            0,
                                            1,
                                            reg_move,
                                            0,
                                            0)
