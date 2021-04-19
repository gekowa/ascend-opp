#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

dynamic unpack
"""


import te
from te import tvm
from te.platform import cce_conf
from te.platform import insn_cmd
from te.platform import cce_intrin
from te.platform import cce_params
from te.platform import cce_build
from te.utils.op_utils import check_op_params
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import DYNAMIC_OUTPUT
from te.utils.op_utils import OPTION_ATTR_INT
from te.utils.op_utils import REQUIRED_ATTR_INT
from te.utils.op_utils import KERNEL_NAME


class CompileVar:
    """
    Compile var
    """
    def __init__(self, name, bound):
        self.tvm_var = tvm.var(name)
        self.name = name
        self.bound = bound

    def get_tvm_var(self):
        return self.tvm_var

    def get_name(self):
        return self.name

    def get_bound(self):
        return self.bound


class Unpack:
    """
    Base Class for Unpack Op, includes Unpack op info.
    """
    def __init__(self, input_x, output_y, num, axis, kernel_name):
        self.input_x = input_x
        self.num = num
        self.kernel_name = kernel_name
        self.dtype = input_x.get("dtype").lower()

        self.dim_info_vars = []
        self.ub_tensor_list = []
        self.res_tensor_list = []
        self.virtual_node = None
        self.sch_list = []
        self.arg_list = []
        self.rules = []
        self.compile_vars = {}
        self.dim_vars = []
        self.dim_bounds = []
        self.output_shape = []
        self.x_reshape = None
        self.left_range = None
        self.right_range = None

        self._normalize_shape()
        self._trans_input_shape(axis)
        self.new_axis = 1

        self._input_placeholder = None
        self.block_idx = tvm.thread_axis('blockIdx.x')

        self.ub_size = cce_conf.get_soc_spec(cce_conf.UB_SIZE)
        self.core_num = cce_conf.get_soc_spec(cce_conf.CORE_NUM)

    def _compute(self):
        """
        Unpack compute function
        """
        self._input_placeholder = tvm.placeholder(self.x_reshape,
                                                  dtype=self.dtype,
                                                  name="input_x")
        self.output_shape = [self._input_placeholder.shape[0],
                             1,
                             self._input_placeholder.shape[2]]

        offset = 0
        for i in range(self.num):
            tensor_ub = tvm.compute(self.output_shape,
                                    lambda *index:
                                    self._input_placeholder(
                                        *self._index_offset(offset,
                                                            *index)),
                                    name="tensor" + str(i))
            self.ub_tensor_list.append(tensor_ub)

            res_tensor = tvm.compute(self.output_shape,
                                     lambda *index: tensor_ub(*index),
                                     name="res" + str(i))
            self.res_tensor_list.append(res_tensor)
            offset = offset + self.output_shape[self.new_axis]

        # create virtual node
        def _add_compute(*index):
            virtual_tensor = self.res_tensor_list[0](*index)
            for res_tensor in self.res_tensor_list[1:]:
                virtual_tensor += res_tensor(*index)
            return virtual_tensor

        self.virtual_node = tvm.compute(self.output_shape,
                                        lambda *index: _add_compute(*index),
                                        name="virtual_node")

    def _unpack_schedule(self,
                         block_tiling_axis,
                         right_dim_in,
                         ub_tiling_axis,
                         split_factor):
        """
        unpack schedule function
        Parameters
        ----------
        block_tiling_axis: int
            identify spilt axis for multicore
        right_dim_in: tvm.var
            the var identify right_dim of output_shape
        ub_tiling_axis: int
            identify spilt axis for ub_tiling
        split_factor: tvm.var
            the var identify spilt_factor
        Returns
        ---------
        sch: tvm.schedule
            the compute schedule
        build_list: list
            include tvm.tensor of input and tvm.tensor of res
        """
        build_list = [self._input_placeholder]
        for res_tensor in self.res_tensor_list:
            build_list.append(res_tensor)

        sch = tvm.create_schedule(self.virtual_node.op)
        sch.disable_allocate(cce_params.scope_ubuf)

        for tensor in self.ub_tensor_list:
            sch[tensor].set_scope(cce_params.scope_ubuf)

        right_dim_outer, right_dim_inner = sch[self.virtual_node].split(
            self.virtual_node.op.axis[block_tiling_axis], factor=right_dim_in)
        sch[self.virtual_node].bind(right_dim_outer, self.block_idx)

        if ub_tiling_axis == 0:
            axis_outer, axis_inner = sch[self.virtual_node].split(
                self.virtual_node.op.axis[0], factor=1)
        else:
            axis_outer, axis_inner = sch[self.virtual_node].split(
                right_dim_inner, factor=split_factor)

        for i in range(self.num):
            sch[self.ub_tensor_list[i]].compute_at(
                sch[self.virtual_node], axis_outer)
            sch[self.res_tensor_list[i]].compute_at(
                sch[self.virtual_node], axis_outer)
            sch[self.ub_tensor_list[i]].emit_insn(
                self.ub_tensor_list[i].op.axis[ub_tiling_axis],
                insn_cmd.DMA_COPY)
            sch[self.res_tensor_list[i]].emit_insn(
                self.res_tensor_list[i].op.axis[ub_tiling_axis],
                insn_cmd.DMA_COPY)

        sch[self.virtual_node].emit_insn(axis_inner, insn_cmd.PHONY_INSN)

        return sch, build_list

    def build_cce(self):
        """
        Build cce
        """
        self._compute()
        tiling_cases = self._calc_tiling_case()
        for case in tiling_cases:
            tvm_vars = self.dim_info_vars.copy()
            right_dim_in = CompileVar("right_dim_in", self.right_range)
            tvm_vars.append(right_dim_in)
            split_factor = CompileVar("split_factor",
                                      case.get("ub_factor_bound"))
            tvm_vars.append(split_factor)

            var_list = [var.get_tvm_var() for var in tvm_vars]
            sch, tensor_list = self._unpack_schedule(
                case.get("block_tiling_axis"), right_dim_in.get_tvm_var(),
                case.get("ub_tiling_axis"), split_factor.get_tvm_var())

            # set var bound
            for var in tvm_vars:
                sch.set_var_range(var.get_tvm_var(), *(var.get_bound()))

            self.sch_list.append(sch)
            self.arg_list.append(var_list + tensor_list)

            self.rules.append(case.get("key"))
            self.compile_vars[case.get("key")] = \
                [var.get_name() for var in tvm_vars]

        build_config_items = {"parse_ddr_args": True,
                              "build_fatbin": True}
        build_config = cce_build.build_config_update_list(
            cce_build.dynamic_build_config,
            build_config_items)

        with build_config:
            tvm.build(self.sch_list, self.arg_list, rules=self.rules,
                      target="cce", name=self.kernel_name)

    def _normalize_shape(self):
        """
        Let input dimensions be represented by tvm.var
        """
        x_shape = list(self.input_x["shape"])
        x_range = list(self.input_x["range"])
        if len(x_shape) != len(x_range):
            raise RuntimeError("Unpack:input_x shape is invalid")

        for key, value in enumerate(x_shape):
            if value == -1:
                dim_info_var = CompileVar("dim_" + str(key), x_range[key])
                self.dim_info_vars.append(dim_info_var)
                self.dim_vars.append(dim_info_var.get_tvm_var())
            else:
                self.dim_vars.append(x_shape[key])
            self.dim_bounds.append(x_range[key])

        te.op.add_compile_info("input_shapes", x_shape)

    def _trans_input_shape(self, axis):
        """
        trans the input shape into three dimensions (left, mid, right) and
        get the range of different dims of the input shape.
        Returns:
        -------
        x_reshape: new input shape of format with (left, mid, right)
        left_range:left dim range
        right_range:right dim range
        """
        real_axis = axis + len(self.dims_var) if axis < 0 else axis
        left_dim = tvm.const(1)
        left_upper = 1
        for idx in range(real_axis):
            left_dim *= self.dim_vars[idx]
            left_upper *= self.dim_bounds[idx][1]
        self.left_range = (1, left_upper)

        right_dim = tvm.const(1)
        right_upper = 1
        for idx in range(real_axis + 1, len(self.dim_vars)):
            right_dim *= self.dim_vars[idx]
            right_upper *= self.dim_bounds[idx][1]
        self.right_range = (1, right_upper)
        self.x_reshape = (left_dim, self.dim_vars[real_axis], right_dim)

    def _index_offset(self, offset, *index):
        """
        Compute the output offset in input_tensor
        """
        input_index = list(index)
        output_index = ()
        for idx, _ in enumerate(self.output_shape):
            if idx == self.new_axis:
                input_index[idx] = input_index[idx] + offset
            output_index += (input_index[idx], )
        return output_index

    def _calc_tiling_case(self):
        """
        calc different tiling strategy
        """
        tiling_cases = []
        dtype_sie = cce_intrin.get_bit_len(self.dtype) // 8
        ub_ele_num = self.ub_size // dtype_sie
        right_dim_bound = [(1, ub_ele_num), (ub_ele_num, None)]
        ub_factor_bound = (1, ub_ele_num)
        for case_id, _ in enumerate(right_dim_bound):
            tiling_cases.append({"key": case_id,
                                 "block_tiling_axis": 2,
                                 "ub_tiling_axis": 0 if case_id == 0 else 2,
                                 "ub_factor_bound": ub_factor_bound})
        return tiling_cases


@te.op.register_operator("Unpack")
@check_op_params(REQUIRED_INPUT,
                 DYNAMIC_OUTPUT,
                 OPTION_ATTR_INT,
                 REQUIRED_ATTR_INT,
                 KERNEL_NAME)
def unpack(input_x, output_y, num=None, axis=0, kernel_name="unpack"):
    """
    unpacks the given dimension of a rank R tensor into rank (R-1) tensors.

    Parameters
    ----------
    input_x : dict.
        shape, dtype and format of value to be unpacked.
    output_y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis, automatically inferred if None(default).
    axis: int.
        the axis to unpack along.
    kernel_name : str
        cce kernel name, default value is "unpack".

    Returns
    -------
    None
    """
    unpack_obj = Unpack(input_x, output_y, num, axis, kernel_name)
    unpack_obj.build_cce()

    # Add compile info
    te.op.add_compile_info("axis", axis)
    te.op.add_compile_info("ub_size", unpack_obj.ub_size)
    te.op.add_compile_info("core_num", unpack_obj.core_num)
    te.op.add_compile_info("dtype", unpack_obj.dtype)
    te.op.add_compile_info("vars", unpack_obj.compile_vars)