"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

reduce sum
"""
from itertools import combinations
import te
import te.lang.dynamic
from te import tvm
from topi import generic
from te import platform as tbe_platform
from te.platform.shape_classifier import classify
from te.platform.shape_classifier import Mode
from topi.cce import util as cce_util
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import OPTION_ATTR_BOOL
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import variable_shape
from te.platform.operation import add_compile_info


NONETYPE = type(None)


def _check_data_shape_const(input_shape):
    """
    check whether the input data shape is const
    """

    for dim in input_shape:
        if dim < 0:
            return False
    return True


def _calc_tiling_key(axes, dim_len):
    """
    calculate tiling key when data shape is const
    """
    tiling_key = 0
    for dim_idx, _ in enumerate(range(dim_len)):
        if dim_idx in axes:
            tiling_key += 2 * (2 ** (dim_len - 1 - dim_idx))
        else:
            tiling_key += 1 * (2 ** (dim_len - 1 - dim_idx))

    return tiling_key


def _const_shape_compute(data_input_x, axis, keep_dims):
    """
    const shape compute
    """
    dtype = data_input_x.dtype
    cce_product = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")

    if cce_product not in ("Ascend310",) and dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support(
                "te.lang.cce.sum", "float32"):
        data_input_x = te.lang.cce.cast_to(data_input_x, "float32")
    res_sum = te.lang.cce.sum(data_input_x, axis=axis, keepdims=keep_dims)
    res = te.lang.cce.cast_to(res_sum, dtype)

    return res


def _adjust_tensor_list(sch, ori_tensor_list):
    """
    adjust tensor list in special cases
    """
    real_out_tensors = sch.cce_special["real_out_tensor"]
    orign_out_tensors = sch.cce_special["orign_out_tensor"]
    # update the config_tensor_list:update 1 auto_cast tensor 2 compute
    # group tensor
    config_tensor_list_tmp = []
    for tensor in ori_tensor_list:
        if tensor not in orign_out_tensors:
            config_tensor_list_tmp.append(tensor)
        else:
            index = orign_out_tensors.index(tensor)
            config_tensor_list_tmp.append(real_out_tensors[index])
    # update special_tensor_list:if the spec node is a output, no need
    # to use workspace
    special_tensor_list = []
    for tensor in sch.cce_special["tensor_list"]:
        if tensor not in config_tensor_list_tmp:
            special_tensor_list.append(tensor)
    tensor_list = config_tensor_list_tmp + special_tensor_list

    return tensor_list


# pylint: disable=invalid-name
def _reduce_sum_const(x, axes, keepdims, kernel_name):
    """
    build process when shape is const
    """
    input_shape = x.get("shape")
    axes_shape = axes.get("shape")
    dtype_lower_x = x.get("dtype").lower()
    dtype_lower_axes = axes.get("dtype").lower()
    min_combination_order = 0 if axes_shape[0] == -1 else axes_shape[0] - 1
    max_combination_order = len(input_shape) if axes_shape[0] == -1 else axes_shape[0]
    const_schedules, const_tensors, tiling_keys = [], [], []
    block_dim_dic = {}

    build_config_items = {
        "parse_ddr_args": True,
        "dynamic_shape": False,
        "build_fatbin": True,
    }
    build_config = tbe_platform.cce_build.build_config_update_list(
        tbe_platform.cce_build.dynamic_build_config,
        build_config_items)
    data_input_axes = tvm.placeholder(axes_shape, name="data_input_axes",
                                      dtype=dtype_lower_axes)
    with build_config:
        for _, dim_idx in enumerate(range(min_combination_order, max_combination_order)):
            # list of combinations of axes
            axes_list = list(combinations(range(len(input_shape)), dim_idx + 1))
            for axes in axes_list:
                tiling_key = _calc_tiling_key(axes, len(input_shape))
                data_input_x = tvm.placeholder(input_shape, name="data_input_x",
                                               dtype=dtype_lower_x)
                res_sum = _const_shape_compute(data_input_x, axes, keepdims)

                with tvm.target.cce():
                    sch = te.lang.cce.te_schedule.cce_schedule.schedule_cce(res_sum)
                    block_dim_dic[str(tiling_key)] = 1
                    # obtain block_dim
                    schedule_ir = tvm.build_module.form_body(sch)

                    def _verify(var):
                        if isinstance(var, tvm.stmt.AttrStmt):
                            if var.attr_key == "thread_extent":
                                block_dim_dic[str(tiling_key)] = int(var.value)

                    tvm.ir_pass.PostOrderVisit(schedule_ir, _verify)
                    const_schedules.append(sch)
                    tiling_keys.append(tiling_key)
                    const_tensors.append(_adjust_tensor_list(sch, [data_input_x, data_input_axes, res_sum]))

        tvm.build(const_schedules, const_tensors, rules=tiling_keys, target="cce", name=kernel_name)

    te.op.add_compile_info("block_dim", block_dim_dic)
    te.op.add_compile_info("reduce_shape_known", 1)
    te.op.add_compile_info("reduce_axis_unknown", 1)


# 'pylint: disable=unused-argument,invalid-name
def reduce_sum_compute(x, axes, y, keepdims=None, kernel_name="reduce_sum"):
    """
    reduce_sum compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    axes: int, list, tuple or NONETYPE
        the axes for reduce.
    y: dict
        the dict of output tensor.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same type as input tensor.
    """
    dtype = x.dtype
    if dtype == "float16":
        x = te.lang.dynamic.cast_to(x, "float32")
    res_sum = te.lang.dynamic.sum(x, axis=axes, keepdims=keepdims)
    res = te.lang.dynamic.cast_to(res_sum, dtype)

    return res


# 'pylint: disable=too-many-locals,invalid-name
@te.op.register_operator("ReduceSum")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT,
                 OPTION_ATTR_BOOL, KERNEL_NAME)
def reduce_sum(x, axes, y, keepdims=False, kernel_name="reduce_sum"):
    """reduce a tensor on a certain axes based on sum.

    Parameters:
    ----------
    x: dict
        the dict of input tensor.
    axes: dict
        the axes for reduce.
    y: dict
        the dict of output tensor.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum".

    Returns
    -------
    None
    """

    dtype_x = x["dtype"]
    dtype_lower_x = dtype_x.lower()
    check_list_x = ("float16", "float32")
    check_dtype(dtype_lower_x, check_list_x, param_name="x")

    dtype_axes = axes["dtype"]
    dtype_lower_axes = dtype_axes.lower()
    check_list_axes = ("int32", "int64")
    check_dtype(dtype_lower_axes, check_list_axes, param_name="axes")
    input_shape = x.get("shape")

    if not _check_data_shape_const(input_shape):
        schedules = []
        ins = classify([x, axes], Mode.REDUCE)
        tensors = []
        shape_axes = variable_shape([axes])[0]
        data_input_axes = tvm.placeholder(shape_axes, name="data_input_axes",
                                          dtype=dtype_lower_axes)

        for (x, axes) in ins:
            with te.op.compute():
                shape_x = variable_shape([x])[0]
                data_input_x = tvm.placeholder(shape_x, name="data_input_x",
                                               dtype=dtype_lower_x)
                shape_len = len(shape_x)
                axes_d = cce_util.axis_check(shape_len, axes)
                res = reduce_sum_compute(data_input_x, axes_d, y, keepdims)

                tensors.append([data_input_x, data_input_axes, res])

            with tvm.target.cce():
                schedule = generic.auto_schedule(res)
            schedules.append(schedule)

        # build
        config = {"name": kernel_name,
                  "tensor_list": tensors}
        te.lang.dynamic.build(schedules, config)
        add_compile_info("reduce_axis_unknown", 1)

    else:
        _reduce_sum_const(x, axes, keepdims, kernel_name)
