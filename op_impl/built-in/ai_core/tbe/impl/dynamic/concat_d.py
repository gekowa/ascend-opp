#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

concat
"""

import te.lang.dynamic
from te.utils.op_utils import *
from impl.dynamic.concat_v2_d import concat_v2_d
from impl.dynamic.concat_v2_d import op_select_format as select_format_concat_v2


# pylint: disable=locally-disabled,unused-argument,too-many-branches
# pylint: disable=too-many-locals,too-many-statements,unused-variable
# pylint: disable=too-many-boolean-expressions
def op_select_format(input_values,
                     output_data,
                     axis,
                     kernel_name="concat"):
    return select_format_concat_v2(input_values, output_data, axis, kernel_name)


@te.op.register_operator("ConcatD")
@check_op_params(DYNAMIC_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_INT, KERNEL_NAME)
def concat_d(input_values, output_data, concat_dim, kernel_name="concat"):
    """
    algorithm: concat
    Concatenates tensors along one dimension.
    Parameters
    ----------
    input_values : A list of `dict`.dict include keys shape and dtype
    output_data: dict of output_data, dict include keys shape and dtype
    concat_dim : scalar, in the range [-rank(values), rank(values))]
    kernel_name : cce kernel name, default value is "concat"
    Returns
    -------
    None
    """
    # concat_d is the same as concat_v2_d
    # use concat_v2_d to replace
    return concat_v2_d(input_values, output_data, concat_dim, kernel_name)