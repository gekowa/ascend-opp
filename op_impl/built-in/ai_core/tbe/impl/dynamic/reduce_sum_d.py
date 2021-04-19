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
dynamic reduce sum
"""
import te
import te.lang.dynamic
from te import tvm
from topi import generic
from topi.cce import util as cce_util
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import REQUIRED_ATTR_LIST_INT
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import variable_shape
from te.platform.operation import add_compile_info

NONETYPE = type(None)


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


# 'pylint: disable=unused-argument,invalid-name
def reduce_sum_d_compute(x,
                         y,
                         axis=None,
                         keepdims=None,
                         kernel_name="reduce_sum_d"):
    """redusce_sum_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    dtype = x.dtype
    if dtype == "float16":
        x = te.lang.dynamic.cast_to(x, "float32")
    res_sum = te.lang.dynamic.sum(x, axis=axis, keepdims=keepdims)
    res = te.lang.dynamic.cast_to(res_sum, dtype)

    return res

# 'pylint: disable=too-many-locals,invalid-name
@te.op.register_operator("ReduceSumD")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_INT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def reduce_sum_d(x, y, axis=None, keepdims=None, kernel_name="reduce_sum_d"):
    """reduce a tensor on a certain axis based on sum.

    Parameters:
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    None
    """

    dtype = x["dtype"]
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32")
    check_dtype(dtype_lower, check_list, param_name="x")

    with te.op.compute():
        shape = x["shape"]
        shape_range = x["range"]

        axes = []
        shape_len = len(shape)
        if not axis:
            for i, _ in enumerate(shape):
                axes.append(i)
        else:
            axes = list(axis)
        axes = cce_util.axis_check(shape_len, axes)

        shape_new, shape_range_new, axes_new, fused_rel_dic = \
            fused_reduce_axis(shape, shape_range, axes)

        add_compile_info("fused_rel_dic", fused_rel_dic)
        x["shape"] = shape_new
        x["range"] = shape_range_new
        shape_var_new = variable_shape([x])[0]

        data_input = tvm.placeholder(shape_var_new, name="data_input",
                                     dtype=dtype_lower)
        res = reduce_sum_d_compute(data_input, y, axes_new, keepdims)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    # build
    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}
    te.lang.dynamic.build(sch, config)
