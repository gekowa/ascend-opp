/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @brief adaptive avg pool2d op proto
 *
 */

#ifndef GE_OP_ADAPTIVE_AVG_POOL2D_H
#define GE_OP_ADAPTIVE_AVG_POOL2D_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
REG_OP(AdaptiveAvgPool2d)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(output_size, ListInt)
    .OP_END_FACTORY_REG(AdaptiveAvgPool2d)
}  // namespace ge

#endif  // GE_OP_ADAPTIVE_AVG_POOL2D_H