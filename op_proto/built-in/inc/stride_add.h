/** 
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * @file embedding_dense_grad.harderr
 *
 * @version 1.0
 */

 
#ifndef GE_OP_STRIDE_ADD_H
#define GE_OP_STRIDE_ADD_H
#include "graph/operator_reg.h"

namespace ge {
REG_OP(StrideAdd)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(x1_c1_offset, Int)
    .REQUIRED_ATTR(x2_c1_offset, Int)
    .REQUIRED_ATTR(c1_len, Int)
    .OP_END_FACTORY_REG(StrideAdd)
}
#endif // GE_OP_STRIDE_ADD_H
