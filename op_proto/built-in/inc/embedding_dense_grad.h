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

# ifndef GE_OP_EMBEDDING_DENSE_GRAD_H
# define GE_OP_EMBEDDING_DENSE_GRAD_H

# include "graph/operator_reg.h"

namespace ge {
    REG_OP(EmbeddingDenseGrad)
    .INPUT(grad, TensorType({DT_FLOAT32})) /* "First operand." */
    .INPUT(indices, TensorType({DT_INT32})) /* "Second operand." */
    .OUTPUT(y, TensorType({DT_FLOAT32}))  /* "Result, has same element type as two inputs" */
    .REQUIRED_ATTR(num_weights, Int)
    .ATTR(padding_idx, Int, -1)
    .ATTR(scale_grad_by_freq, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingDenseGrad)
} // namespace ge

#endif // GE_OP_EMBEDDING_DENSE_GRAD_H