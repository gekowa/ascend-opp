/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: MASKED_FILL
 * Create: 2020-08-11
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */


/**
*
* @par Inputs:
* x:A Tensor of type float16 or float32 or int32 or int8
* mask:A Tensor of type float16 or float32 or int32 or int8
* value:A Tensor or scalar of type float16 or float32 or int32 or int8

* @par Outputs:
* y: A Tensor of type float16 or float32 or int32 or int8
**/


#ifndef GE_OP_MASKED_FILL_H
#define GE_OP_MASKED_FILL_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(MaskedFill)
.INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
.INPUT(mask, TensorType({DT_BOOL}))
.INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
.OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
.OP_END_FACTORY_REG(MaskedFill)
}
#endif // GE_OP_MASK_FILL_H
