/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: index_add
 * Create: 2020-09-01
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 **/
#ifndef GE_OP_INDEX_ADD_H
#define GE_OP_INDEX_ADD_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(IndexAdd)
    .INPUT(var, TensorType({DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))
    .OUTPUT(var_out, TensorType({DT_INT32, DT_INT8, DT_UINT8, DT_FLOAT32, DT_FLOAT16}))
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(IndexAdd)
} // namespace ge
#endif // GE_OP_INDEX_ADD_H
