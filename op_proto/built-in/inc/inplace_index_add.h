/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: InplaceIndexAdd
 * Create: 2020-08-10
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef GE_OP_INPLACE_INDEX_ADD_H
#define GE_OP_INPLACE_INDEX_ADD_H
#include "graph/operator_reg.h"
namespace ge {
        REG_OP(InplaceIndexAdd)
        .INPUT(var, TensorType({DT_INT16, DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))     /* "First operand." */
        .INPUT(indices, TensorType({DT_INT32}))                                            /* "Second operand." */
        .INPUT(updates, TensorType({DT_INT16, DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16})) /* "Third operand." */
        .OUTPUT(var, TensorType({DT_INT16, DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))    /* "Result, has same element type as two inputs" */
        .REQUIRED_ATTR(axis, Int)
        .OP_END_FACTORY_REG(InplaceIndexAdd)
    } // namespace ge
#endif // GE_OP_INPLACE_INDEX_ADD_H
