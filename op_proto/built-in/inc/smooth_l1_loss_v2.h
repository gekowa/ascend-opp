/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: GE_OP_SMOOTH_L1_LOSS_H
 * Create: 2020-07-22
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */


#ifndef GE_OP_SMOOTH_L1_LOSS_V2_H
#define GE_OP_SMOOTH_L1_LOSS_V2_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(SmoothL1LossV2)
    .INPUT(predict, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(label, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(loss, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .ATTR(sigma, Float, 1.0)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SmoothL1LossV2)
}
#endif // GE_OP_SMOOTH_L1_LOSS_H
