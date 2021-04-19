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
cosine_embedding_loss
"""
from functools import reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from te import platform as cceconf


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
def _shape_check(shape_x1, shape_x2, shape_tgt):
    # check whether the shape meets the broadcast requirements, and output broadcast shape
    try:
        _, _, x_shape = util.produce_shapes(shape_x1, shape_x2)
    except RuntimeError:
        raise RuntimeError("x1 and x2 can't be broadcast")

    x_shape_reduce = x_shape[:]
    x_shape_reduce.pop(1)
    try:
        _, _, tgt_shape = util.produce_shapes(x_shape_reduce, shape_tgt)
    except RuntimeError:
        raise RuntimeError("x and target can't be broadcast")
    min_dim = min(len(shape_x1), len(shape_x2), len(shape_tgt))
    if min_dim >= 3:
        reduce_dim = -1
        for i in range(-1, -min_dim, -1):
            if(shape_x1[i] == shape_x2) or (
                    shape_x1[i] == shape_tgt[i]):
                reduce_dim = i
            else:
                break
        if reduce_dim != -1:
            shape_x1 = list(shape_x1[:reduce_dim]) + [
                reduce(lambda x, y:x*y, shape_x1[reduce_dim:])]
            shape_x2 = list(shape_x2[:reduce_dim]) + [
                reduce(lambda x, y:x*y, shape_x2[reduce_dim:])]
            shape_tgt = list(shape_tgt[:reduce_dim]) + [
                reduce(lambda x, y:x*y, shape_tgt[reduce_dim:])]
            x_shape = list(x_shape[:reduce_dim]) + [
                reduce(lambda x, y:x*y, x_shape[reduce_dim:])]
            tgt_shape = list(tgt_shape[:reduce_dim]) + [
                reduce(lambda x, y:x*y, tgt_shape[reduce_dim:])]
    util.check_shape_rule(shape_x1)
    util.check_shape_rule(shape_x2)
    util.check_shape_rule(shape_tgt)
    util.check_tensor_shape_size(shape_x1)
    util.check_tensor_shape_size(shape_x2)
    util.check_tensor_shape_size(shape_tgt)

    return x_shape, tgt_shape, shape_x1, shape_x2, shape_tgt


def _dtype_check(input_dtype_x1, input_dtype_x2, target_dtype, reduction):
    # cast_to not support "int16", "int64", ISA not support float64(double)
    x_check_list = ["int8", "uint8", "int32", "float16", "float32"]
    if not input_dtype_x1 in x_check_list:
        raise RuntimeError("x1 dtype %s not support" % input_dtype_x1)
    if not input_dtype_x2 in x_check_list:
        raise RuntimeError("x2 dtype %s not support" % input_dtype_x2)

    # cast_to not support "int16", "int64", "uint8" can't indicate -1
    tgt_check_list = ["int8", "int32", "float16", "float32"]
    if not target_dtype in tgt_check_list:
        raise RuntimeError("target dtype %s not support" % target_dtype)

    reduce_check_list = ['mean', 'sum', 'none']
    if reduction not in reduce_check_list:
        raise RuntimeError("reduction method not support")


# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("cosine_embedding_loss")
def cosine_embedding_loss_compute(x1, x2, target, output_y, x_shape_broadcat,
                                  tgt_shape_broadcast, margin=0,
                                  reduction='mean',
                                  kernel_name="cosine_embedding_loss"):
    """
    DSL description of the cosine_embedding_loss operator's calculation process

    Parameters
    ----------
    x1: TVM tensor
        the placeholder of x1 input data
    x2: TVM tensor
        the placeholder of x2 input data
    target: TVM tensor
        the placeholder of target input data
    output_y: TVM tensor
        the placeholder of beta output data
    x_shape_broadcat: list,
        x1 and x2 broadcast shape
    tgt_shape_broadcast: list
        x and target broadcast shape
    margin: float
        margin, default value is "0.0"
    reduction: str
        string indicate reduce method, default value is "mean"
    kernel_name: str
        cce kernel name, default value is "group_norm"

    Returns
    -------
    res: TVM tensor
    """
    cce_plat = cceconf.get_soc_spec('SOC_VERSION')
    cast_dtype = 'float32'
    epsilon = tvm.const(1e-12, dtype="float32")

    if cce_plat == 'Ascend310':
        cast_dtype = 'float16'
        epsilon = tvm.const(5e-8, dtype="float16")

    if x1.dtype.lower() != cast_dtype and x1.dtype.lower() != 'float32':
        x1 = te.lang.cce.cast_to(x1, cast_dtype)

    if x2.dtype.lower() != cast_dtype and x2.dtype.lower() != 'float32':
        x2 = te.lang.cce.cast_to(x2, cast_dtype)

    target = te.lang.cce.cast_to(target, x1.dtype)

    x1_broadcast = te.lang.cce.broadcast(x1, x_shape_broadcat)
    x2_broadcast = te.lang.cce.broadcast(x2, x_shape_broadcat)
    target_broadcast = te.lang.cce.broadcast(target, tgt_shape_broadcast)

    # DSL description for cosine similarity compute
    prod = te.lang.cce.vmul(x1_broadcast, x2_broadcast)

    mag1 = te.lang.cce.vmul(x1_broadcast, x1_broadcast)
    mag2 = te.lang.cce.vmul(x2_broadcast, x2_broadcast)
    mag_square1 = te.lang.cce.sum(mag1, axis=1)
    mag_square2 = te.lang.cce.sum(mag2, axis=1)

    x1_epsilon = te.lang.cce.vadds(mag_square1, epsilon)
    x2_epsilon = te.lang.cce.vadds(mag_square2, epsilon)
    x1_sqrt = te.lang.cce.vsqrt(x1_epsilon)
    x2_sqrt = te.lang.cce.vsqrt(x2_epsilon)
    mode_num = te.lang.cce.vmul(x1_sqrt, x2_sqrt)
    prod_num = te.lang.cce.sum(prod, axis=1)
    cos_res = te.lang.cce.vdiv(prod_num, mode_num)

    # DSL description for 1 - cos(x1, x2)
    zero_tensor = te.lang.cce.vmuls(target_broadcast, 0)
    one_tensor = te.lang.cce.vadds(zero_tensor, 1)

    neg_one_tensor = te.lang.cce.vsub(zero_tensor, one_tensor)
    pos = te.lang.cce.vsub(one_tensor, cos_res)

    # DSL description for max(0, cos(x1, x2) - margin)
    margin_const = tvm.const(margin, dtype="float32")
    margin_tensor = te.lang.cce.vmuls(one_tensor, margin_const)
    neg_sub = te.lang.cce.vsub(cos_res, margin_tensor)
    neg = te.lang.cce.vmax(zero_tensor, neg_sub)

    # DSL description for output = pos if y == 1 else neg
    output_pos = te.lang.cce.vcmpsel(target_broadcast, one_tensor, 'eq',
                                     pos, zero_tensor)
    output_neg = te.lang.cce.vcmpsel(target_broadcast, neg_one_tensor, 'eq',
                                     neg, zero_tensor)
    res = te.lang.cce.vadd(output_pos, output_neg)
    if reduction in ['sum', 'mean']:
        if reduction == 'mean':
            num = reduce(lambda x, y: x * y, tgt_shape_broadcast)
            mean_cof = num ** (-1)
            res = te.lang.cce.vmuls(res, mean_cof)
            res = te.lang.cce.cast_to(res, 'float32')

        reduce_axis = [index for index, _ in enumerate(tgt_shape_broadcast)]
        res_sum = te.lang.cce.sum(res, axis=reduce_axis)
        return res_sum

    return te.lang.cce.cast_to(res, 'float32')


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@util.check_input_type(dict, dict, dict, dict, float, str, str)
def cosine_embedding_loss(input_x1, input_x2, target, y,
                          margin=0, reduction='mean',
                          kernel_name="cosine_embedding_loss"):
    """
    algorithm: cosine_embedding_loss
    cosine embedding loss = // 1-cos(x1, x2),                if y == 1
                            \\ max(0, cos(x1, x2) - margin), if y == -1
    Note that the size of 5D Tensors are defined by "NC1HWC0".
    The input tensor's dimension C should be equal.

    Parameters
    ----------
    x1: dict
        dict of input x1, A Tensor for input data.
    x2: dict
        dict of input x1, A Tensor for input data.
    target: dict
        dict of target, A Tensor for target, include 1 and -1.
    output_y: dict
        dict of output, A Tensor for output
    margin: float
        float of margin, A float number subtracted when y == -1
    reduction: str
        str of output reduce method.
    kernel_name: str
        kernel name, default value is "cosine_embedding_loss"

    Returns
    -------
    None
    """
    shape_x1 = input_x1.get("shape")
    dtype_x1 = input_x1.get("dtype")
    input_dtype_x1 = dtype_x1.lower()
    shape_x2 = input_x2.get("shape")
    dtype_x2 = input_x2.get("dtype")
    input_dtype_x2 = dtype_x2.lower()
    shape_tgt = target.get("shape")
    dtype_tgt = target.get("dtype")
    target_dtype = dtype_tgt.lower()

    util.check_kernel_name(kernel_name)
    x_shape_broadcat, tgt_shape_broadcast, shape_x1, shape_x2, shape_tgt = \
        _shape_check(shape_x1, shape_x2, shape_tgt)
    _dtype_check(input_dtype_x1, input_dtype_x2, target_dtype, reduction)

    data_input1 = tvm.placeholder(shape_x1, name="data_input1",
                                  dtype=input_dtype_x1)
    data_input2 = tvm.placeholder(shape_x2, name="data_input2",
                                  dtype=input_dtype_x2)
    data_target = tvm.placeholder(shape_tgt, name="data_target",
                                  dtype=target_dtype)

    res = cosine_embedding_loss_compute(data_input1, data_input2, data_target,
                                        y, x_shape_broadcat,
                                        tgt_shape_broadcast, margin, reduction,
                                        kernel_name)

    with tvm.target.cce():
        schedule = generic.auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": [data_input1, data_input2, data_target, res],
    }

    te.lang.cce.cce_build_code(schedule, config)
