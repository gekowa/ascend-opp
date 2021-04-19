/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: tensor_equal
 * Create: 2020-07-28
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
**/
/**
*
* @par Inputs:
* input_x:A Tensor of type float16 or float32 or int32 or int8
* input_y:A Tensor of type float16 or float32 or int32 or int8

* @par Outputs:
* out_z: A Tensor of type bool True or False
**/

#ifndef GE_OP_TENSOR_EQUAL_H
#define GE_OP_TENSOR_EQUAL_H

#include "graph/operator_reg.h"

namespace ge {
REG_OP(TensorEqual)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8})) /* "First operand." */
    .INPUT(input_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8})) /* "Second operand." */
    .OUTPUT(output_z, TensorType({DT_BOOL}))  /* "Result, True or False" */
    .OP_END_FACTORY_REG(TensorEqual)
} // namespace ge

#endif // GE_OP_TENSOR_EQUAL_H
