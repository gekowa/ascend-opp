/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: GE_OP_SOFTPLUS_V2_H
 * Create: 2020-08-24
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef GE_OP_SOFTPLUS_V2_H
#define GE_OP_SOFTPLUS_V2_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(SoftplusV2)
    .INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .ATTR(beta, Float, 1.0)
    .ATTR(threshold, Float, 20.0)
    .OP_END_FACTORY_REG(SoftplusV2)
}
#endif // GE_OP_SOFTPLUS_V2_H
