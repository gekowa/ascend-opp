/* * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
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
 * @file hardtanh_grad.h
 *
 * @brief
 *
 * @version 1.0
 */

#ifndef GE_OP_HARDTANH_GRAD_H
#define GE_OP_HARDTANH_GRAD_H

#include "graph/operator_reg.h"

/**
 * @brief pytorch hardtanh_backward operator.
 *
 * @par Inputs:
 * 2 inputs, including:
 * @li result, minimum tensor of the linear region range, datatype:float16/float32, format:ND/5HD.
 * @li grad, maximum tensor of the linear region range, datatype:float16/float32, format:ND/5HD.
 *
 * @par Attributes:
 * 2 attributes, including:
 * @li min_val, minimum value of the linear region range, datatype:float.
 * @li max_val, maximum value of the linear region range, datatype:float.
 *
 * @par Outputs:
 * 1 output, including:
 * @li y, hardtanh_backward output tensor, datatype and format is same as input result.
 *
 * @attention Constraints:
 * This operator only supports dataType: float16/float32, format: ND/5HD.
 */
namespace ge {
REG_OP(HardtanhGrad)
    .INPUT(result, TensorType({ DT_FLOAT16, DT_FLOAT })) /* "First operand." */
    .INPUT(grad, TensorType({ DT_FLOAT16, DT_FLOAT }))   /* "Second operand." */
    .OUTPUT(y, TensorType({ DT_FLOAT16, DT_FLOAT }))     /* "Result, has same element type as two inputs" */
    .ATTR(min_val, Float, -1.0)
    .ATTR(max_val, Float, 1.0)
    .OP_END_FACTORY_REG(HardtanhGrad)
}

#endif // GE_OP_HARDTANH_GRAD_H
