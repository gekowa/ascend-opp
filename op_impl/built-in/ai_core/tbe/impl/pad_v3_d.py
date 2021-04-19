#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

pad_v3
"""
from te.utils.op_utils import *
from impl.pad_d import pad_d


# pylint: disable=locally-disabled,too-many-arguments,too-many-branches,
# pylint: disable=locally-disabled,too-many-statements
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_LIST_LIST_INT,
                 OPTION_ATTR_INT, OPTION_ATTR_STR, OPTION_ATTR_BOOL, KERNEL_NAME)
def pad_v3_d(input_x, output_x, paddings, constant_values=0, mode="constant", paddings_contiguous=True,
             kernel_name="pad_v3_d"):
    """ calculating pad tensor by paddings parameters

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        shape and dtype of output
    paddings: list or tuple.
        For each dimension D of input, paddings[D, 0] indicates how many
        values to add
        before the contents of tensor in that dimension, and paddings[D, 1]
        indicates
        how many values to add after the contents of tensor in that dimension.
    mode: string
        Defaults to "constant", indicates paddings mode,
        support "constant", "reflect", "edge"
    paddings_contiguous: bool
        Determining the parsing sequence of paddings
        True : paddings is arranged as [[begin0, end0], [begin1, end1], ...]
        False: paddings is arranged as [[begin0, begin1], ..., [end0, end1]]
    kernel_name : str
        cce kernel name, default value is "pad_d"

    Returns
    -------
    None.
    """
    shape = list(input_x.get("shape"))
    paddings = list(paddings)
    check_shape(shape, param_name="input_x")
    if len(paddings) is not len(shape):
        raise RuntimeError("Paddings and shape are not the same length.")
    for padding in paddings:
        if len(padding) != 2:
            raise RuntimeError("Paddings's shape is not in the form of (n,2)")
        if (not isinstance(padding[0], int)) or (not isinstance(padding[1], int)):
            raise RuntimeError("Paddings only suppot int")

    mode_list = ["constant", "reflect", "edge"]
    if mode.lower() not in mode_list:
        raise RuntimeError("mode should be one of \"constant\", \"reflect\", \"edge\"")

    # [[begin0, begin1], ..., [end0, end1], ...] --> [[begin0, end0], [begin1, end1], ...]
    flatten_paddings = [i for item in paddings for i in item]
    if not paddings_contiguous:
        rank = len(paddings)
        for i in range(rank):
            paddings[i] = [flatten_paddings[i], flatten_paddings[i + rank]]

    if mode.lower() == "constant":
        if constant_values != 0:
            raise RuntimeError("only support constant_values=0 now")
        pad_d(input_x, output_x, paddings, kernel_name)
    else:
        raise RuntimeError("only support mode=\"constant\" now")
