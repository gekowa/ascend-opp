/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: ADDCDIV
 * Create: 2020-08-19
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef GE_OP_ADDCDIV_ADDCMUL_H
#define GE_OP_ADDCDIV_ADDCMUL_H

#include "graph/operator_reg.h"

namespace ge {
    REG_OP(Addcdiv)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8}))
    .INPUT(x3, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8}))
    .ATTR(value, Float, 1.0)
    .OP_END_FACTORY_REG(Addcdiv)
    
    REG_OP(Addcmul)
    .INPUT(input_data, TensorType({ DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8 }))
    .INPUT(x1, TensorType({ DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8 }))
    .INPUT(x2, TensorType({ DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8 }))
    .OUTPUT(y, TensorType({ DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8 }))
    .ATTR(value, Float, 1.0)
    .OP_END_FACTORY_REG(Addcmul)
}

#endif // GE_OP_Addcdiv_Addcmul_H
