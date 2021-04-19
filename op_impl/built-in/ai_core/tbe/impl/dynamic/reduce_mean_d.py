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
dynamic reduce mean
"""
from collections import Iterable

import te
import te.lang.dynamic
from te import platform as tbe_platform
from te import tvm
from te.platform import operation
from te.platform.operation import add_compile_info
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.op_utils import REQUIRED_ATTR_LIST_INT
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import variable_shape
from topi import generic


def fused_reduce_axis(shape, shape_range, reduce_axis):
    # convert reduce axis
    for idx, value in enumerate(reduce_axis):
        if value < 0:
            reduce_axis[idx] = len(shape) + value

    # if all reduce axis is 1, do not del axis which is 1
    is_del_axis = False
    for idx, _ in enumerate(shape):
        if shape[idx] != 1 and idx in reduce_axis:
            is_del_axis = True

    shape_new = []
    reduce_axis_new = []
    shape_range_new = []
    fused_rel_dic = {}
    last_axis_type = "n"
    for idx, value in enumerate(shape):
        shape_range_value = list(shape_range[idx])
        if (is_del_axis and value != 1) or (not is_del_axis):
            if last_axis_type == "n":
                if idx in reduce_axis:
                    reduce_axis_new.append(len(shape_new))
                    shape_new.append(value)
                    shape_range_new.append(shape_range_value)
                    fused_rel_dic[len(shape_new) - 1] = [idx]
                    last_axis_type = "r"
                else:
                    shape_new.append(value)
                    shape_range_new.append(shape_range_value)
                    fused_rel_dic[len(shape_new) - 1] = [idx]
                    last_axis_type = "a"
            elif last_axis_type == "a":
                if idx in reduce_axis:
                    reduce_axis_new.append(len(shape_new))
                    shape_new.append(value)
                    shape_range_new.append(shape_range_value)
                    fused_rel_dic[len(shape_new) - 1] = [idx]
                    last_axis_type = "r"
                else:
                    shape_new[-1] = -1 if shape_new[-1] == -1 or value == -1 \
                        else shape_new[-1] * value
                    shape_range_new[-1][0] = None if shape_range_new[-1][0] \
                        is None or shape_range_value[0] is None or \
                        shape_range_new[-1][0] * shape_range_value[0] \
                        > 2147483647 else shape_range_new[-1][0] * \
                        shape_range_value[0]
                    shape_range_new[-1][1] = None if shape_range_new[-1][1] \
                        is None or shape_range_value[1] is None or \
                        shape_range_new[-1][1] * shape_range_value[1] > \
                        2147483647 else shape_range_new[-1][1] * \
                        shape_range_value[1]
                    fused_rel_dic[len(shape_new) - 1].append(idx)
                    last_axis_type = "a"
            elif last_axis_type == "r":
                if idx in reduce_axis:
                    shape_new[-1] = -1 if shape_new[-1] == -1 or value == -1 \
                        else shape_new[-1] * value
                    shape_range_new[-1][0] = None if shape_range_new[-1][0] \
                        is None or shape_range_value[0] is None or \
                        shape_range_new[-1][0] * shape_range_value[0] > \
                        2147483647 else shape_range_new[-1][0] * \
                        shape_range_value[0]
                    shape_range_new[-1][1] = None if shape_range_new[-1][1] \
                        is None or shape_range_value[1] is None or \
                        shape_range_new[-1][1] * shape_range_value[1] > \
                        2147483647 else shape_range_new[-1][1] * \
                        shape_range_value[1]
                    fused_rel_dic[len(shape_new) - 1].append(idx)
                    last_axis_type = "r"
                else:
                    shape_new.append(value)
                    shape_range_new.append(shape_range_value)
                    fused_rel_dic[len(shape_new) - 1] = [idx]
                    last_axis_type = "a"

    return shape_new, shape_range_new, reduce_axis_new, fused_rel_dic


def reduce_mean_d_compute(x,
                          y,
                          axes,
                          keepdims=None,
                          kernel_name="reduce_mean_d",
                          impl_mode="high_performance",
                          is_5hdc=False):
    """reduce_mean_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axes: int, list, tuple or NoneType
        the axes for reduce.
    keepdims: bool or NoneType
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_mean_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    shape_x = x.shape

    reduce_elts = 1.0
    if isinstance(axes, Iterable):
        for i in axes:
            if isinstance(shape_x[i], tvm.expr.IntImm):
                reduce_elts *= shape_x[i].value
            else:
                reduce_elts *= shape_x[i]
    else:
        reduce_elts = shape_x[axes]

    dtype = x.dtype
    if dtype == "float32":
        calc_dtype = "float32"
    elif dtype == "float16":
        cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
        if not tbe_platform.cce_conf.api_check_support("te.lang.cce.sum",
                                                       "float32"):
            calc_dtype = "float16"
        elif cce_product == "Ascend310" and impl_mode == "high_performance":
            calc_dtype = "float16"
        else:
            calc_dtype = "float32"
    else:
        # int8 and uint8
        calc_dtype = "float16"

    if isinstance(reduce_elts, float):
        cof = reduce_elts ** (-1)
        cof = tvm.const(cof, dtype=calc_dtype)
    else:
        cof = operation.var("cof", dtype=calc_dtype)
        if calc_dtype == "float16":
            operation.var("cof_empty", dtype=calc_dtype)
        add_compile_info("reduce_mean_cof_dtype", calc_dtype)

    if dtype != calc_dtype:
        data_input_tmp = te.lang.dynamic.cast_to(x, calc_dtype)
    else:
        data_input_tmp = x

    data_input_tmp = te.lang.dynamic.vmuls(data_input_tmp, cof)
    res = te.lang.dynamic.sum(data_input_tmp, axis=axes, keepdims=keepdims)

    if dtype != calc_dtype:
        if dtype in ("int8", "uint8"):
            res = te.lang.dynamic.cast_to(res, dtype, False)
        else:
            res = te.lang.dynamic.cast_to(res, dtype)

    return res


@te.op.register_operator("ReduceMeanD")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_INT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def reduce_mean_d(input_x, output_y, axes,
                  keepdims=None, kernel_name="reduce_mean_d",
                  impl_mode="high_performance"):
    """
    Reduce a tensor on a certa in axes based on mean.

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input
    output_y: dict
        shape and dtype of output
    axes : int, list, tuple, NoneType
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    keepdims : bool, NoneType
        if true, retains reduced dimensions with length 1,
        default value is None.
    kernel_name : str
        cce kernel name, default value is reduce_mean_d

    Returns
    -------
    None
    """
    dtype = input_x["dtype"]
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32", "int8", "uint8")
    check_dtype(dtype_lower, check_list)

    with te.op.compute():
        shape = input_x["shape"]
        shape_range = input_x["range"]

        shape_len = len(shape)
        if not axes:
            axes = range(shape_len)
        if hasattr(axes, 'index'):
            axes = list(axes)
        # not support 5HD
        is_5hdc = False

        shape_new, shape_range_new, axes_new, fused_rel_dic = \
            fused_reduce_axis(shape, shape_range, axes)

        add_compile_info("fused_rel_dic", fused_rel_dic)
        input_x["shape"] = shape_new
        input_x["range"] = shape_range_new
        shape_var_new = variable_shape([input_x])[0]

        data_input = tvm.placeholder(shape_var_new, name="data_input",
                                     dtype=dtype_lower)
        res = reduce_mean_d_compute(data_input, output_y, axes_new,
                                    keepdims, impl_mode=impl_mode,
                                    is_5hdc=is_5hdc)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.dynamic.build(sch, config)
