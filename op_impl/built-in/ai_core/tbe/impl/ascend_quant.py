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
ascend_quant
"""
from functools import reduce as function_reduce
import te.lang.cce
import topi
from te import tvm
from te.utils.op_utils import check_shape
from te.utils.op_utils import check_op_params
from te.utils.op_utils import *
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce.util import is_lhisi_version

# define the tag of quant
ASCEND_QUANT_TAG = "quant"


# pylint: disable=too-many-arguments,invalid-name,unused-argument
# pylint: disable=unnecessary-lambda,too-many-locals
def _check_params(x, y, scale, offset, sqrt_mode, round_mode, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr.
    """
    shape = x.get("shape")
    x_format = x.get("format")
    dtype = x.get("dtype").lower()
    format_list = ["NC1HWC0", "FRACTAL_NZ"]
    if x_format not in format_list:
        raise RuntimeError(
            "ascend quant only support [NC1HWC0, FRACTAL_NZ]")
    if x_format == "NC1HWC0":
        if len(shape) != 5:
            raise RuntimeError(
                "ascend quant only support the length of shape is 4 or 5")
    if x_format == "FRACTAL_NZ":
        if len(shape) < 4:
            raise RuntimeError(
                "ascend quant only support the length of shape is >= 4")
    check_shape(shape, param_name="x")
    if is_lhisi_version():
        # es
        check_list = ["float16"]
    else:
        check_list = ["float16", "float32"]

    if dtype not in check_list:
        raise RuntimeError("ascend quant only support %s"
                           % (",".join(check_list)))
    round_mode_list = ["Round", "Ceil", "Floor", "Trunc"]
    if round_mode not in round_mode_list:
        raise RuntimeError(
            "ascend quant only support %s while" % (",".join(round_mode_list)))


def _check_l1_fusion(x, y):
    """
    check the l1 fusion parameters
    """
    x_addr_type = x.get("addr_type", 0)
    x_valid_shape = x.get("valid_shape", [])
    x_slice_offset = x.get("slice_offset", [])
    x_l1_fusion_type = x.get("L1_fusion_type", -1)

    y_valid_shape = y.get("valid_shape", [])
    y_l1_fusion_type = y.get("L1_fusion_type", -1)

    if x_l1_fusion_type not in (-1, 0):
        raise RuntimeError("quant L1_fusion_type only  support (-1, 0)")

    if y_l1_fusion_type not in (-1, 0):
        raise RuntimeError("quant L1_fusion_type only  support (-1, 0)")

    if x_valid_shape and len(x_valid_shape) != 5:
        raise RuntimeError("the len of valid shape should be 5")

    if y_valid_shape and len(y_valid_shape) != 5:
        raise RuntimeError("the len of valid shape should be 5")

    if x_slice_offset and len(x_slice_offset) != 5:
        raise RuntimeError("the len of slice_offset shape should be 5")

    attr = {"addr_type": x_addr_type,
            "valid_shape": x_valid_shape,
            "slice_offset": x_slice_offset,
            "L1_fusion_type": x_l1_fusion_type}

    return x_l1_fusion_type, y_l1_fusion_type, attr


def _reform_compute_generate(tensor, in_shape, out_shape, val_info,
                             nz_format_flag):
    """
    generate lambda func
    Parameters
    ----------
    tensor : input tensor

    in_shape : the shape of input tensor

    out_shape :the shape of output tensor

    val_info : the val info of offset,scale

    nz_format_flag: the format of input tensor

    Returns
    -------
    res lambda_func
    """
    in_shape = list(in_shape)
    out_shape = list(out_shape)
    n_dim = len(in_shape)

    c0_index = n_dim - 1
    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4

    def lambda_func(*indice):
        new_indice = [0] * n_dim
        for i in range(n_dim):
            if i == c0_index:
                new_indice[i] = (indice[c1_index] * out_shape[c0_index] +
                                 indice[c0_index]) % in_shape[c0_index]
            elif i == c1_index:
                new_indice[i] = (indice[c1_index] * out_shape[c0_index] +
                                 indice[c0_index]) // in_shape[c0_index]
            else:
                new_indice[i] = indice[i]

        if val_info[0]:
            return tensor(*new_indice) + val_info[1]

        return tensor(*new_indice) * val_info[2]

    return lambda_func


def _input_compute_generate(x, in_shape, read_shape, c1_dim, c1_index):
    """
    generate lambda func
    """
    x_shape = te.lang.cce.util.shape_to_list(x.shape)
    dtype = x.dtype
    x_slice_offset = _get_input_attr(x, "slice_offset", [], True)
    l1_fusion_flag = _get_input_attr(x, "l1_fusion_flag", -1, False)
    if not x_slice_offset:
        x_slice_offset = [0, 0, 0, 0, 0]

    if l1_fusion_flag != -1:
        x_w = x_shape[3]
        n_offset, _, h_offset, w_offset, _ = x_slice_offset
        if c1_dim % 2 == 0:
            input_ub = tvm.compute(
                in_shape, lambda n, c1, m, c0: x(n + n_offset,
                                                 c1,
                                                 (m // x_w) + h_offset,
                                                 (m % x_w) + w_offset,
                                                 c0),
                name="input_ub",
                attrs={"c_out": c1_dim})
        else:
            input_ub = tvm.compute(
                read_shape,
                lambda n, c1, m, c0: tvm.select(c1 <= in_shape[c1_index] - 1,
                                                x(n + n_offset,
                                                  c1,
                                                  (m // x_w) + h_offset,
                                                  (m % x_w) + w_offset,
                                                  c0),
                                                tvm.const(0, dtype=dtype)),
                name='input_ub',
                attrs={"c_out": c1_dim})
    else:
        if c1_dim % 2 == 0:
            input_ub = tvm.compute(in_shape, lambda *i: x(*i),
                                   name="input_ub",
                                   attrs={"c_out": c1_dim})
        else:
            input_ub = tvm.compute(read_shape,
                                   lambda *indice: tvm.select(
                                       indice[c1_index] <= in_shape[
                                           c1_index] - 1,
                                       x(*indice),
                                       tvm.const(0, dtype=dtype)),
                                   name='input_ub',
                                   attrs={"c_out": c1_dim})
    return input_ub


def _reform_by_vadds(input_tensor, input_shape, output_shape, offset_val,
                     nz_format_flag):
    """
    5 dim input tensor C0 change
    Parameters
    ----------
    input_tensor : input tensor

    input_shape : the shape of input tensor

    output_shape :the shape of output tensor

    offset_val : the val of offset

    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    vadds_vector = tvm.compute(output_shape,
                               _reform_compute_generate(
                                   input_tensor, input_shape,
                                   output_shape, (True, offset_val, -1),
                                   nz_format_flag),
                               name='reform_by_vadds')

    return vadds_vector


def _reform_by_vmuls(input_tensor, input_shape, output_shape, scale_val,
                     nz_format_flag):
    """
    5 dim input tensor C0 change
    Parameters
    ----------
    input_tensor : input tensor

    input_shape : the shape of input tensor

    output_shape :the shape of output tensor

    scale_val : the val of scale

    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    vmuls_vector = tvm.compute(output_shape,
                               _reform_compute_generate(
                                   input_tensor, input_shape,
                                   output_shape, (False, -1, scale_val),
                                   nz_format_flag),
                               name='reform_by_vmuls')

    return vmuls_vector


def _compute_scale(in_tensor, in_shape, out_shape, attr_list, nz_format_flag):
    """
    the compute of scale
    Parameters
    ----------
    in_tensor : input tensor

    in_shape : the shape of input tensor

    out_shape :the shape of output tensor

    attr_list : the attr list

    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    scale = attr_list[0]
    offset = attr_list[1]
    sqrt_mode = attr_list[2]
    if scale != 1:
        scale_value = tvm.const(scale, "float16")
        scale_ub = _reform_by_vmuls(in_tensor, in_shape, out_shape,
                                    scale_value, nz_format_flag)
        if sqrt_mode:
            scale_sqrt_ub = tvm.compute(
                out_shape,
                lambda *indice: scale_ub(*indice) * scale_value,
                name="scale_sqrt_ub")
            res = _compute_offset(scale_sqrt_ub, in_shape, out_shape,
                                  (offset, False, scale), nz_format_flag)
        else:
            res = _compute_offset(scale_ub, in_shape, out_shape,
                                  (offset, False, scale), nz_format_flag)
    else:
        res = _compute_offset(in_tensor, in_shape, out_shape,
                              (offset, True, scale), nz_format_flag)
    return res


def _compute_offset(in_tensor, in_shape, out_shape, attr_list, nz_format_flag):
    """
    the compute of scale
    Parameters
    ----------
    in_tensor : input tensor

    in_shape : the shape of input tensor

    out_shape :the shape of output tensor

    attr_list : the attr list

    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    offset = attr_list[0]
    reform_flag = attr_list[1]
    scale = attr_list[2]
    if offset != 0 or scale == 1:
        offset_value = tvm.const(offset, "float16")
        if reform_flag:
            offset_ub = _reform_by_vadds(in_tensor, in_shape, out_shape,
                                         offset_value, nz_format_flag)
        else:
            offset_ub = tvm.compute(
                out_shape,
                lambda *indice: in_tensor(*indice) + offset_value,
                name="offset_ub")
        cast_i8_ub = tvm.compute(out_shape,
                                 lambda *indice: topi.cast(
                                     offset_ub(*indice), "int8"),
                                 name='cast_i8_ub')
    else:
        cast_i8_ub = tvm.compute(out_shape,
                                 lambda *indice: topi.cast(
                                     in_tensor(*indice),
                                     "int8"),
                                 name='cast_i8_ub')
    return cast_i8_ub


def _get_shape_info(in_shape, nz_format_flag):
    """
    the compute of scale
    Parameters
    ----------
    in_shape : the shape of input tensor

    nz_format_flag : the format of output tensor

    Returns
    -------
    read_shape, out_shape
    """
    c0_index = len(in_shape) - 1
    c1_index = 1
    c1_dim = in_shape[1]
    if nz_format_flag:
        c1_index = len(in_shape) - 4
        c1_dim = in_shape[c1_index]
    out_shape = in_shape[:]
    read_shape = in_shape[:]
    read_shape[c1_index] = read_shape[c1_index] + 1 * (c1_dim % 2)
    for dim, _ in enumerate(in_shape):
        if dim == c0_index:
            out_shape[dim] = in_shape[dim] * 2
        if dim == c1_index:
            out_shape[dim] = in_shape[dim] // 2 + 1 * (c1_dim % 2)
    return read_shape, out_shape


def _get_input_attr(x, attr_name, default_value, is_list):
    """
    get the attrs of input tensor
    """
    value = default_value
    if x.op.attrs:
        if attr_name in x.op.attrs:
            if is_list:
                value = x.op.attrs[attr_name]
            else:
                value = x.op.attrs[attr_name].value
    return value


def _get_input_l1_info(x):
    """
    get the l1 fusion info from input tensor
    """
    x_valid_shape = _get_input_attr(x, "valid_shape", [], True)
    x_slice_offset = _get_input_attr(x, "slice_offset", [], True)
    l1_fusion_flag = _get_input_attr(x, "l1_fusion_flag", -1, False)
    if not x_slice_offset:
        x_slice_offset = [0, 0, 0, 0, 0]
    x_shape = te.lang.cce.util.shape_to_list(x.shape)
    in_shape = x_shape

    if l1_fusion_flag != -1:
        v_shape = te.lang.cce.util.shape_to_list(x_valid_shape)
        if v_shape:
            in_shape = [v_shape[0],
                        v_shape[1],
                        v_shape[2] * v_shape[3],
                        v_shape[4]]
        else:
            in_shape = [x_shape[0],
                        x_shape[1],
                        x_shape[2] * x_shape[3],
                        x_shape[4]]
    return x_valid_shape, x_slice_offset, in_shape, l1_fusion_flag


def _get_out_l1_info(y):
    """
    get the l1 fusion info from output tensor
    """
    y_addr_type = 0
    y_valid_shape = []
    if isinstance(y, dict):
        y_addr_type = y.get("addr_type", 0)
        y_valid_shape = y.get("valid_shape", [])
    elif isinstance(y, tvm.tensor.Tensor):
        y_addr_type = _get_input_attr(y, "addr_type", 0, False)
        y_valid_shape = _get_input_attr(y, "valid_shape", [], True)

    hwc0 = 0
    if y_valid_shape:
        _, _, h_valid, w_valid, c0_valid = y_valid_shape
        hwc0 = h_valid * w_valid * c0_valid
    return y_addr_type, y_valid_shape, hwc0


def _is_nz_format(x):
    """
    check is nz format
    """
    tensor_format = "NC1HWC0"
    if x.op.attrs:
        if 'format' in x.op.attrs:
            # NZ format,UB convergence scenario, input shape ..C1,N1,N0,C0
            tensor_format = x.op.attrs['format']
    if tensor_format == "FRACTAL_NZ":
        return True, tensor_format

    return False, tensor_format


@fusion_manager.register("ascend_quant")
def ascend_quant_compute(x, y, scale, offset, sqrt_mode=False,
                         round_mode="Round", kernel_name="ascend_quant"):
    """
    float16/float32 -> int8

    Parameters:
    ----------
    x : the tensor of input

    y : the dict of output

    scale : the data of scale

    offset : the data of offset

    sqrt_mode : the sqrt mode when true the result to do sqrt

    round_mode : the data conversion mode

    kernel_name : cce kernel name, default value is "ascend_quant"

    Returns:
    -------
    None
    """
    dtype = x.dtype
    _, _, in_shape, l1_fusion_flag = _get_input_l1_info(x)
    y_addr_type, _, hwc0 = _get_out_l1_info(y)

    nz_format_flag, tensor_format = _is_nz_format(x)
    c1_dim = in_shape[1]
    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4
        c1_dim = in_shape[c1_index]

    read_shape, out_shape = _get_shape_info(in_shape, nz_format_flag)

    input_ub = _input_compute_generate(x, in_shape, read_shape, c1_dim,
                                       c1_index)
    if dtype == "float32":
        cast_f16_ub = tvm.compute(read_shape,
                                  lambda *indice: topi.cast(
                                      input_ub(*indice),
                                      "float16"),
                                  name='cast_f16_ub')
        cast_i8_ub = _compute_scale(
            cast_f16_ub, in_shape, out_shape, (scale, offset, sqrt_mode),
            nz_format_flag)
    else:
        cast_i8_ub = _compute_scale(
            input_ub, in_shape, out_shape, (scale, offset, sqrt_mode),
            nz_format_flag)
    res = tvm.compute(out_shape, lambda *indice: cast_i8_ub(*indice),
                      name="res", tag=ASCEND_QUANT_TAG,
                      attrs={'scale': scale,
                             'sqrt_mode': sqrt_mode,
                             'offset': offset,
                             'round_mode': round_mode,
                             'input_format': tensor_format,
                             'c1_dim': c1_dim,
                             'l1_fusion_flag': l1_fusion_flag,
                             'addr_type': y_addr_type,
                             'HWC0': hwc0
                             })
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_FLOAT, OPTION_ATTR_FLOAT,
                 OPTION_ATTR_BOOL, OPTION_ATTR_STR, KERNEL_NAME)
def ascend_quant(x, y, scale, offset, sqrt_mode=False, round_mode="Round",
                 kernel_name="ascend_quant"):
    """
    float16/float32 -> int8

    Parameters:
    ----------
    x : the dict of input

    y : the dict of output

    scale : the data of scale

    offset : the data of offset

    sqrt_mode : the sqrt mode when true the result to do sqrt

    round_mode : the data conversion mode

    kernel_name : cce kernel name, default value is "ascend_quant"

    Returns:
    -------
    None
    """
    _check_params(x, y, scale, offset, sqrt_mode, round_mode, kernel_name)
    shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    input_format = x.get("format")

    x_l1_fusion_type, y_l1_fusion_type, attr = _check_l1_fusion(x, y)

    if input_format == "NC1HWC0":
        if x_l1_fusion_type != -1:
            input_shape = shape
            attr["l1_fusion_flag"] = x_l1_fusion_type
        else:
            # change to N,C1,H*W,C0
            input_shape = (shape[0],
                           shape[1],
                           shape[2] * shape[3],
                           shape[4])
    else:
        # nz change to 1,C1,N1*N0,C0 equivalence N,C1,H*W,C0
        batch = 1
        if len(shape) > 4:
            batch = function_reduce(lambda x, y: x * y, shape[:-4])
        input_shape = (batch,
                       shape[-4],
                       shape[-3] * shape[-2],
                       shape[-1])
    input_x = tvm.placeholder(input_shape,
                              name="input_x",
                              dtype=input_dtype,
                              attrs=attr)

    res = ascend_quant_compute(input_x, y, scale, offset, sqrt_mode,
                               round_mode, kernel_name)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_x, res]}

    te.lang.cce.cce_build_code(sch, config)
