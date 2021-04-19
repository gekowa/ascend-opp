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
 * @file L1LossGrad
 *
 * @version 1.0
 */

#ifndef GE_OP_L1_LOSS_BACKWARD_H
#define GE_OP_L1_LOSS_BACKWARD_H

#include "graph/operator_reg.h"

namespace ge {
    REG_OP(L1LossGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(L1LossGrad)
}

#endif // GE_OP_L1_LOSS_BACKWARD_H
