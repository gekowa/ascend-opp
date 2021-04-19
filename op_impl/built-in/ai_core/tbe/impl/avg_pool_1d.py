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
avg_pool_1d
"""

from te.platform import build_config
from te.platform import insn_cmd
from te.platform import scope_ubuf

from te.platform.cce_conf import CORE_NUM
from te.platform.cce_conf import UB_SIZE
from te.platform.cce_conf import get_soc_spec
from te.platform.cce_intrin import get_bit_len
from te.platform.fusion_manager import fusion_manager

from te.tvm import api as tvm
from te.tvm.build_module import build
from te.tvm.schedule import create_schedule

from te.utils import check_para


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# pylint: disable=line-too-long,too-many-statements,too-many-branches
def parameter_check(shape, div_shape, shape_out, dtype, ksize, pads):
    """
    avg_pool_1d_check_rule

    Parameters
    ----------
    shape: input shape
    div_shape: input shape
    shape_out: output shape
    dtype: data type
    ksize: kernel size
    pads: zero paddings on both sides
    kernel_name: kernel name

    Returns
    -------
    None
    """
    check_para.check_shape(shape)
    check_para.check_shape(div_shape)
    check_para.check_shape(shape_out)
    half_kernel_size = ksize // 2
    if dtype != "float16" and dtype != "float32":
        raise RuntimeError("input dtype only support float16, float32")
    if pads[0] > half_kernel_size:
        raise RuntimeError("pad should be smaller than half of kernel size, but got pad = %d, ksize = %d" %
                           (pads[0], ksize))


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# pylint: disable=line-too-long,too-many-statements,too-many-branches,invalid-name
@fusion_manager.register("avg_pool_1d")
def avg_pool_1d_compute(x,
                        div,
                        out_dict,
                        kernel,
                        pad,
                        stride,
                        ceil_mode=True,
                        count_include_pad=False,
                        kernel_name="avg_pool_1d"):
    """
    avg_pool_1d compute

    Parameters
    ----------
    x: input tensor dict
    div: matrix tensor dict
    out_dict: output dict
    kernel: the size of the window
    pad: implicit zero padding to be added on both sides
    stride: the stride of the window
    ceil_mode: when True, will use ceil instead of floor to compute the output shape
    count_include_pad: when True, will include the zero-padding in the averaging calculation
    kernel_name: kernel name

    Returns
    -------
    output tensor, reduce_tensor_list, tensor_list
    """
    shape = [i.value for i in x.shape]
    x_wi = shape[-2]
    pad_l, pad_r = pad

    if ceil_mode:
        x_wo = (x_wi + pad_l + pad_r - kernel + stride - 1) // stride + 1
    else:
        x_wo = ((x_wi + pad_l + pad_r) - kernel) // stride + 1

    if x_wo <= 0:
        raise RuntimeError("Given input Win num: %d. Calculated output Wout num: %d. should keep Woutput > 0" %
                           (x_wi, x_wo))

    if pad_l:
        # ensure that the last pooling starts inside the image needed to avoid problems in ceil mode
        # existing bug in pytorch code
        # pad_l = 0 and stride is big, but kernel is small, return nan
        if ((x_wo - 1) * stride) >= (x_wi + pad_l):
            x_wo -= 1
    pad_r = (x_wo - 1) * stride + kernel - x_wi - pad_l

    # set padding
    x_fused_axis, x_w, x_c0 = shape
    mid_shape = (x_fused_axis, x_w + pad_l + pad_r, x_c0)
    tensor_mid_shape_in_ub = tvm.compute(
        mid_shape,
        lambda x_fused_axis, w, c0: tvm.select(tvm.any(w < pad_l, w >= x_wi + pad_l), tvm.const(0, dtype=x.dtype), x[
            x_fused_axis, w - pad_l, c0]),
        name="tensor_mid_shape_in_ub")
    reduce_tensor_list = []
    # reduce w
    re_shape = (x_fused_axis, x_wo, x_c0)
    if kernel > 1:

        tensor_w = tvm.compute(
            re_shape,
            lambda fused_axis, w, c0: tvm.sum(tensor_mid_shape_in_ub[fused_axis, w * stride + 0, c0],
                                              tensor_mid_shape_in_ub[fused_axis, w * stride + 1, c0]),
            name="tensor_w")
        reduce_tensor_list.append(tensor_w)
        for j in range(2, kernel):
            tensor_w_tmp = tvm.compute(
                re_shape,
                lambda fused_axis, w, c0: tvm.sum(tensor_mid_shape_in_ub[fused_axis, w * stride + j, c0], tensor_w[
                    fused_axis, w, c0]),
                name="tensor_w" + str(j))
            tensor_w = tensor_w_tmp
            reduce_tensor_list.append(tensor_w)
    elif kernel == 1:
        tensor_w = tensor_mid_shape_in_ub

    tensor_list = []
    tensor_list.append(x)
    tensor_list.append(div)
    tensor_list.append(tensor_mid_shape_in_ub)
    res = tvm.compute(re_shape,
                      lambda i, j, k: tensor_w(i, j, k) * div(0, j, k),
                      attrs={
                          "stride": stride,
                          "kernel": kernel
                      },
                      name="res")
    return res, reduce_tensor_list, tensor_list


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# pylint: disable=line-too-long,too-many-statements,too-many-branches,invalid-name
def avg_pool_1d_schedule(res, reduce_tensor_list, tensor_list):
    """
    avg_pool_1d schedule

    Parameters
    ----------
    res: result of compute
    reduce_tensor_list: list of reduce tensor
    tensor_list: list of tensors

    Returns
    -------
    output sch
    """
    tensor_x = tensor_list[0]
    tensor_div = tensor_list[1]
    tensor_mid_shape_in_ub = tensor_list[2]

    def _ceil(m, n):
        return (m + n - 1) // n

    def _tiling(shape, dtype, wo_out, stride, kernel):
        ub_size_bytes = get_soc_spec(UB_SIZE)
        dtype_bytes_size = get_bit_len(dtype) // 8
        total_ele = ub_size_bytes // dtype_bytes_size // 2

        nc1h, _, c0 = shape
        core_num = get_soc_spec(CORE_NUM)
        fused_axis_block_factor, w_block_factor = nc1h, wo_out

        if fused_axis_block_factor >= core_num:
            fused_axis_block_factor = _ceil(fused_axis_block_factor, core_num)
        else:
            w_block_factor = _ceil(wo_out, _ceil(core_num, fused_axis_block_factor))
            fused_axis_block_factor = 1
        wo_buffer_num = 4
        wi_buffer_num = 2

        # for wi = (wo - 1) * stride + kernel
        # wo_buffer_num * N * C1 * H * Wo * C0 + wi_buffer_num * N * C1 * H * Wi * C0 <= total_ele
        nc1wo_limit = (total_ele // (c0) +
                       (stride - kernel) * wi_buffer_num) // (wo_buffer_num + stride * wi_buffer_num)
        nc1_limit = (total_ele // (c0)) // (wo_buffer_num * wo_out + wi_buffer_num * wo_out * stride - wi_buffer_num *
                                            (stride - kernel))

        if nc1_limit > 1:
            fused_factor, wo_factor = min(fused_axis_block_factor, nc1_limit), w_block_factor
        else:
            fused_factor, wo_factor = 1, nc1wo_limit // 8 * 8
        return [fused_axis_block_factor, w_block_factor], [fused_factor, wo_factor]

    x_shape = [i.value for i in tensor_x.shape]
    stride = int(res.op.attrs['stride'].value)
    kernel = int(res.op.attrs['kernel'].value)
    [fused_b_factor, w_b_factor], [fused_factor, wo_factor] = _tiling(x_shape, tensor_x.dtype,
                                                                      tensor_div.shape[-2].value, stride, kernel)

    sch = create_schedule(res.op)
    # set output ub
    tensor_div_in_ub = sch.cache_read(tensor_div, scope_ubuf, [res])
    sch[tensor_mid_shape_in_ub].set_scope(scope_ubuf)

    for tensor in reduce_tensor_list:
        sch[tensor].set_scope(scope_ubuf)
    tensor_ub_mul = sch.cache_write(res, scope_ubuf)
    fused_b_out, fused_b_in = sch[res].split(res.op.axis[0], fused_b_factor)
    fused_out, fused_in = sch[res].split(fused_b_in, fused_factor)
    wo_b_out, wo_b_in = sch[res].split(res.op.axis[1], w_b_factor)
    wo_out, wo_in = sch[res].split(wo_b_in, wo_factor)
    sch[res].reorder(fused_b_out, wo_b_out, wo_out, fused_out, fused_in, wo_in, res.op.axis[-1])

    # split tensor_w for reduce sum
    sch[tensor_div_in_ub].compute_at(sch[res], wo_out)
    sch[tensor_mid_shape_in_ub].compute_at(sch[res], fused_out)
    for tensor in reduce_tensor_list:
        sch[tensor].compute_at(sch[res], fused_out)
    sch[tensor_ub_mul].compute_at(sch[res], fused_out)

    sch[tensor_div_in_ub].double_buffer()
    sch[tensor_mid_shape_in_ub].preload()
    sch[tensor_mid_shape_in_ub].double_buffer()
    for tensor in reduce_tensor_list:
        sch[tensor].double_buffer()
    sch[tensor_ub_mul].double_buffer()

    # for multi cores
    block = tvm.thread_axis("blockIdx.x")
    block_axis = sch[res].fuse(fused_b_out, wo_b_out)
    sch[res].bind(block_axis, block)

    # set emit_insn
    sch[tensor_div_in_ub].emit_insn(tensor_div_in_ub.op.axis[0], insn_cmd.DMA_COPY)
    sch[tensor_mid_shape_in_ub].emit_insn(tensor_mid_shape_in_ub.op.axis[0], insn_cmd.DMA_PADDING)
    for tensor in reduce_tensor_list:
        sch[tensor].emit_insn(tensor.op.axis[0], insn_cmd.ADD)
    sch[tensor_ub_mul].emit_insn(tensor_ub_mul.op.axis[0], insn_cmd.MUL)
    sch[res].emit_insn(fused_in, insn_cmd.DMA_COPY)
    return sch


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# pylint: disable=line-too-long,too-many-statements,too-many-branches
# pylint: disable=unused-argument,invalid-name
@check_para.check_op_params(check_para.REQUIRED_INPUT, check_para.REQUIRED_INPUT, check_para.REQUIRED_OUTPUT,
                            check_para.REQUIRED_ATTR_INT, check_para.REQUIRED_ATTR_INT,
                            check_para.REQUIRED_ATTR_LIST_INT, check_para.OPTION_ATTR_BOOL, check_para.OPTION_ATTR_BOOL,
                            check_para.KERNEL_NAME)
def avg_pool_1d(x_dict,
                div_dict,
                out_dict,
                ksize,
                strides,
                pads,
                ceil_mode=True,
                count_include_pad=False,
                kernel_name="avg_pool_1d"):
    """
    Parameters
    ----------
    x_dict : dict, shape and dtype of input_data

    div_dict : dict, shape and dtype of matrix_data

    out_dict : dict, shape and dtype of output_data

    ksize : the size of the window

    strides : the strides of the window.

    pads : implicit zero padding to be added on both sides

    ceil_mode: when True, will use ceil instead of floor to compute the output shape

    count_include_pad: when True, will include the zero-padding in the averaging calculation

    kernel_name : cce kernel name, default value is "avg_pool_1d"

    Returns
    -------
    None
    """

    shape = x_dict.get("shape")
    div_shape = div_dict.get("shape")
    out_shape = out_dict.get("shape")
    dtype = x_dict.get("dtype")
    dtype_div = div_dict.get("dtype")
    parameter_check(shape, div_shape, out_shape, dtype, ksize, pads)

    x_n, x_c1, x_h, x_w, x_c0 = shape
    div_x_n, div_x_c1, div_x_h, div_x_w, div_x_c0 = div_shape
    shape = [x_n * x_c1 * x_h, x_w, x_c0]
    div_shape = [1, div_x_w, div_x_c0]

    tensor_a = tvm.placeholder(shape, name="tensor_a", dtype=dtype)
    tensor_div = tvm.placeholder(div_shape, name="tensor_div", dtype=dtype_div)

    res, reduce_tensor_list, tensor_list = avg_pool_1d_compute(tensor_a, tensor_div, out_dict, ksize, pads, strides,
                                                               ceil_mode, count_include_pad, kernel_name)
    sch = avg_pool_1d_schedule(res, reduce_tensor_list, tensor_list)

    with build_config:
        build(sch, [tensor_a, tensor_div, res], "cce", name=kernel_name)
