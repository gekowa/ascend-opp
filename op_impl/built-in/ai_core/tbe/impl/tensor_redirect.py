# /usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

GetFloatStatus
"""

from te import platform as tbe_platform
from topi.cce import util
from te import tik
from te.utils.op_utils import *


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def tensor_redirect(x, output_x, kernel_name="tensor_redirect"):
    """
    the main function of TensorRedirect

    Parameters
    ----------
    x: dict,shape and datatype
    output_x: dict,shape and datatype
    kernel_name: cce kernel name, default value is "tensor_redirect"

    Returns
    -------
    tik_instance: tik_instance
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    input_dtype = dtype.lower()
    check_list = ["float16", "float32", "int8", "int32", "uint8",
                  "int16", "uint16", "uint32", "int64", "uint64"]
    check_dtype(input_dtype, check_list)

    tik_instance = tik.Tik()

    input_addr = tik_instance.Tensor(input_dtype, shape, name="input_addr",
                                     scope=tik.scope_gm)
    output_data = tik_instance.Tensor(input_dtype, shape,
                                      name="output_data",
                                      scope=tik.scope_gm)

    data_ub = tik_instance.Tensor(input_dtype, shape, name="data_ub",
                                  scope=tik.scope_ubuf)
    tik_instance.data_move(data_ub, input_addr, 0, 1, 1, 0, 0)
    tik_instance.data_move(output_data, data_ub, 0, 1, 1, 0, 0)
    tik_instance.BuildCCE(kernel_name, inputs=[input_addr],
                          outputs=[output_data])
    return tik_instance
