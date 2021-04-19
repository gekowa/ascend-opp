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
 * @file 
 *
 * @version 1.0
 */


#ifndef GE_OP_FILL_V2_D_H
#define GE_OP_FILL_V2_D_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(FillV2D)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64}))
    .ATTR(value, Float, 0)
    .REQUIRED_ATTR(dims, ListInt)
    .OP_END_FACTORY_REG(FillV2D)
}
#endif // GE_OP_FILL_V2_D_H
