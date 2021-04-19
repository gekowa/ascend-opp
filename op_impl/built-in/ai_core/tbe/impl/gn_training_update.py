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
gn_training_update
"""
from __future__ import division

import te.lang.cce
from te import tvm
from te.utils.op_utils import refine_shapes_for_broadcast
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils.op_utils import *
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json

NONETYPE = type(None)


# pylint: disable=locally-disabled,unused-argument,invalid-name
# pylint: disable=locally-disabled,redefined-builtin
def op_select_format(x, sum, square_sum, scale, offset, mean, variance,
                     y, batch_mean, batch_variance,
                     epsilon=0.0001, num_groups=2,
                     kernel_name="gn_training_update"):
    """
    select format dynamically
    """
    input0 = gen_param(classify="input0", name="x",
                       datatype="float16,float,float16,float",
                       format="NCHW,NCHW,NHWC,NHWC")
    input1 = gen_param(classify="input1", name="sum",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")
    input2 = gen_param(classify="input2", name="square_sum",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")
    input3 = gen_param(classify="input3", name="scale",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")
    input4 = gen_param(classify="input4", name="offset",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")
    input5 = gen_param(classify="input5", name="mean",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")
    input6 = gen_param(classify="input6", name="variance",
                       datatype="float,float,float,float",
                       format="ND,ND,ND,ND")

    output0 = gen_param(classify="output0", name="y",
                        datatype="float16,float,float16,float",
                        format="NCHW,NCHW,NHWC,NHWC")
    output1 = gen_param(classify="output1", name="batch_mean",
                        datatype="float,float,float,float",
                        format="ND,ND,ND,ND")
    output2 = gen_param(classify="output2", name="batch_variance",
                        datatype="float,float,float,float",
                        format="ND,ND,ND,ND")
    param_list = [input0, input1, input2, input3, input4,
                  input5, input6,
                  output0, output1, output2]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def check_rule(data, rule_desc, param_name=PARAM_NAME):
    """
    The special check rule for tensor
    """
    if data is None or rule_desc is None:
        return
    error_info = {}
    error_info['errCode'] = OP_ERROR_CODE_009
    error_info['op_name'] = OP_NAME
    error_info['param_name'] = param_name
    error_info['rule_desc'] = rule_desc
    error_info['param_value'] = data
    raise RuntimeError(error_info,
                       "Op[%s] has rule: %s, but [%s] is [%s]."\
                        % (error_info['op_name'],
                        error_info['rule_desc'],
                        error_info['param_name'],
                        error_info['param_value']))


def check_input_shape(shape, data_format="NCHW", num_groups=2):
    check_shape(shape, min_rank=4, max_rank=4, param_name="x")
    c_index = data_format.index("C")
    if shape[c_index] % num_groups != 0:
        check_rule("{} and {}".format(shape[c_index], num_groups),
                   "num_groups must divide C channel",
                   "channel and num_groups")


def check_couple_shape(shape_a, shape_b, ori_shape, data_format,
                       num_groups, first_index=False):
    if first_index:
        first_value = ori_shape[0]
    else:
        first_value = 1
    check_shape(shape_a, min_rank=5, max_rank=5, param_name="x")
    check_shape(shape_b, min_rank=5, max_rank=5, param_name="x")
    if data_format == "NCHW":
        aim_shape = (first_value, num_groups, 1, 1, 1)
    else:
        aim_shape = (first_value, 1, 1, num_groups, 1)
    if tuple(shape_a) != aim_shape:
        check_rule("{} and {}".format(shape_a, aim_shape),
                   "shape_a must match with aim_shape",
                   "shape_a and aim_shape")
    if tuple(shape_b) != aim_shape:
        check_rule("{} and {}".format(shape_a, aim_shape),
                   "shape_b must match with aim_shape",
                   "shape_b and aim_shape")


@fusion_manager.register("gn_training_update")
def gn_training_update_compute(x,
                               scale, offset, mean, variance,
                               sum, square_sum,
                               data_format,
                               y, batch_mean, batch_variance,
                               epsilon, num_groups,
                               kernel_name="gn_training_update"):
    """
    algorithm: group_norm
    group normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    data_format: str
        data format
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    batch_variance: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "in_training_update_v2"

    Returns
    -------
    res: TVM tensor list
        the result of in_training_update_v2 compute
    """
    dtype = x.dtype
    shape = te.lang.cce.util.shape_to_list(x.shape)
    if dtype == "float16":
        x = te.lang.cce.cast_to(x, "float32")
    if data_format == "NCHW":
        num = shape[2] * shape[3] * shape[4]
    else:
        num = shape[1] * shape[2] * shape[4]

    num_rec = 1.0 / num
    # compute the saved mean of x
    compute_mean = te.lang.cce.vmuls(sum, num_rec)
    mean_boardcast = te.lang.cce.broadcast(compute_mean, shape)

    # compute the saved variance of x
    variance_div = te.lang.cce.vmuls(square_sum, num_rec)
    variance_square = te.lang.cce.vmul(compute_mean, compute_mean)
    compute_var = te.lang.cce.vsub(variance_div, variance_square)

    # (x - mean) / sqrt(var + eps)
    # x_mean = x - mean
    # multiplier_add = var + eps
    # multiplier_sqrt = sqrt(var + eps)

    x_mean = te.lang.cce.vsub(x, mean_boardcast)
    multiplier_add = te.lang.cce.vadds(compute_var, epsilon)
    multiplier_sqrt = te.lang.cce.vsqrt(multiplier_add)
    sqrt_boardcast = te.lang.cce.broadcast(multiplier_sqrt, shape)
    mean_wo_scale = te.lang.cce.vdiv(x_mean, sqrt_boardcast)
    result = mean_wo_scale
    if scale is not None and offset is not None:
        scale = te.lang.cce.broadcast(scale, shape)
        offset = te.lang.cce.broadcast(offset, shape)
        scale_scale = te.lang.cce.vmul(result, scale)
        result = te.lang.cce.vadd(scale_scale, offset)

    if dtype == "float16":
        result = te.lang.cce.cast_to(result, "float16")

    res = [result, compute_mean, compute_var]
    return res


@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_INPUT,
                 OPTION_INPUT, OPTION_INPUT,
                 OPTION_INPUT, OPTION_INPUT,
                 REQUIRED_OUTPUT, REQUIRED_OUTPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_FLOAT, OPTION_ATTR_INT, KERNEL_NAME)
def gn_training_update(x, sum, square_sum,
                       scale, offset, mean, variance,
                       y, batch_mean, batch_variance,
                       epsilon=0.0001, num_groups=2,
                       kernel_name="gn_training_update"):
    """
    algorithm: group_norm
    group normalization.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    sum: dict
        dict of sum, A Tensor for sum.
        The output of instance_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A Tensor for square_sum.
        The output of instance_normalization_forward_training_reduce.
    scale: dict
        dict of scale, A Tensor for scale.
    offset: dict
        dict of offset, A Tensor for offset.
    mean: dict
        dict of mean, A Tensor for mean.
    variance: dict
        dict of variance, A Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of mean, A `Tensor`. The update mean of save mean and running mean.
    batch_variance: dict
        dict of variance, A `Tensor`.
        The update variance of save variance and running variance.
    epsilon: float
        A small float number added to the variance of x.
    num_groups: int
        group num
    kernel_name: str
        kernel name, default value is "gn_training_update"

    Returns
    -------
    None
    """
    data_format = x.get("format")
    check_format(data_format, ("NCHW", "NHWC"), param_name="x")

    # Process x, sum, square_sum
    shape_origin = x.get("shape")
    dtype_x = x.get("dtype")
    check_input_shape(shape_origin, data_format, num_groups)

    if data_format == "NCHW":
        shape_x = [shape_origin[0], num_groups,
                   shape_origin[1] // num_groups, shape_origin[2],
                   shape_origin[3]]

    # Reshape NHWC -> NHW[GD]
    elif data_format == "NHWC":
        shape_x = [shape_origin[0], shape_origin[1], shape_origin[2],
                   num_groups, shape_origin[3] // num_groups]

    check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    shape_sum = sum.get("shape")
    shape_square_sum = square_sum.get("shape")
    check_couple_shape(shape_sum, shape_square_sum, shape_origin,
                       data_format, num_groups, True)

    dtype_sum = sum.get("dtype")
    dtype_square_sum = square_sum.get("dtype")
    check_dtype(dtype_sum.lower(), ("float32",), param_name="sum")
    check_dtype(dtype_square_sum.lower(), ("float32",),
                param_name="square_sum")

    x_input = tvm.placeholder(shape_x, name="x_input",
                              dtype=dtype_x.lower())
    sum_input = tvm.placeholder(shape_sum, name="sum_input",
                                dtype=dtype_sum.lower())
    square_sum_input = tvm.placeholder(shape_square_sum,
                                       name="square_sum_input",
                                       dtype=dtype_square_sum.lower())
    scale_input, offset_input, mean_input, var_input = None, None, None, None

    # Process scale and offset

    affine = False
    if scale is not None and offset is not None:
        affine = True
        shape_scale = scale.get("shape")
        dtype_scale = scale.get("dtype")
        shape_offset = offset.get("shape")
        dtype_offset = offset.get("dtype")

        check_couple_shape(shape_scale, shape_offset, shape_origin,
                           data_format,
                           num_groups)

        check_dtype(dtype_scale.lower(), ("float32",), param_name="scale")
        check_dtype(dtype_offset.lower(), ("float32",), param_name="offset")

        scale_input = tvm.placeholder(shape_scale, name="scale_input",
                                      dtype=dtype_scale.lower())
        offset_input = tvm.placeholder(shape_offset, name="offset_input",
                                       dtype=dtype_offset.lower())

    res = gn_training_update_compute(x_input, scale_input, offset_input,
                                     mean_input, var_input,
                                     sum_input, square_sum_input,
                                     data_format,
                                     y, batch_mean, batch_variance,
                                     epsilon, num_groups,
                                     kernel_name=kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    if affine:
        tensor_list = [x_input, sum_input, square_sum_input,
                       scale_input, offset_input] + list(res)
    else:
        tensor_list = [x_input, sum_input, square_sum_input] \
                      + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
