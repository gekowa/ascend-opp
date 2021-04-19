/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: GE_OP_SOFTPLUS_V2_GRAD_H
 * Create: 2020-08-17
 * Notes: This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef GE_OP_SOFTPLUS_V2_GRAD_H
#define GE_OP_SOFTPLUS_V2_GRAD_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(SoftplusV2Grad)
    .INPUT(input_gradients, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .INPUT(input_features, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .OUTPUT(output_backprops, TensorType({ DT_FLOAT, DT_FLOAT16 }))
    .ATTR(beta, Float, 1.0)
    .ATTR(threshold, Float, 20.0)
    .OP_END_FACTORY_REG(SoftplusV2Grad)
}
#endif // GE_OP_SOFTPLUS_V2_GRAD_H
