/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file nn_pooling_ops.h
 * \brief
 */
#ifndef GE_OP_NN_POOLING_OPS_H
#define GE_OP_NN_POOLING_OPS_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Performs pooling on the input.
*@par Inputs:
*@li x: An NCHW tensor of type float16, float32, int8.
*@par Attributes:
*@li mode: An optional int32, specifying the pooling algorithm, either "1" (max pooling) or "0" (avg pooling). Defaults to "0".
*@li global_pooling: An optional bool. Defaults to "false".
*@li window: Optional, including:
*window[0]: An optional int32, specifying the window size along in the H dimension. The value range is [1, 32768]. Defaults to "1".
*window[1]: An optional int32, specifying the window size along in the W dimension. The value range is [1, 32768]. Defaults to "1".
*@li stride: Optional, including:
*stride[0]: An optional int32, specifying the stride along in the H dimension. The value range is [1, 63]. Defaults to "1".
*stride[1]: An optional int32, specifying the stride along in the W dimension. The value range is [1, 63]. Defaults to "1".
*@li pad: Optional, including:
*pad[0]: An optional int32, specifying the up padding. Defaults to "0".
*pad[1]: An optional int32, specifying the bottom padding. Defaults to "0".
*pad[2]: An optional int32, specifying the left padding. Defaults to "0".
*pad[3]: An optional int32, specifying the right padding. Defaults to "0".
*@li dilation: Optional, including:
*dilation[0]: An optional int32, specifying the up dilation. Defaults to "1".
*dilation[1]: An optional int32, specifying the bottom dilation. Defaults to "1".
*dilation[2]: An optional int32, specifying the left dilation. Defaults to "1".
*dilation[3]: An optional int32, specifying the right dilation. Defaults to "1".
*@li ceil_mode: An optional int32, either "0" (ceil mode) or "1" (floor mode). Defaults to "0".
*@par Outputs:
*y: An NCHW tensor of type float16, float32, int32.
*@attention Constraints:
*@li window[0] * window[1] < 256;
*@li 1<=input_h<=4096,1<=input_w<=4096
*@li If input tensor N is a prime number, it should be less than 65535.
*@par Third-party framework compatibility
*@li Compatible with the Caffe operator Pooling.
*@li Compatible with the TensorFlow operator Pooling.
*/
REG_OP(Pooling)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT32}))
    .ATTR(mode, Int, 0)                 // 0:max pooling or 1:avg pooling
    .ATTR(global_pooling, Bool, false)
    .ATTR(window, ListInt, {1,1})       // kernel size
    .ATTR(stride, ListInt, {1,1})       // stride size
    .ATTR(pad, ListInt, {0,0,0,0})      // pad size
    .ATTR(dilation, ListInt, {1,1,1,1})
    .ATTR(ceil_mode, Int, 0)
    .OP_END_FACTORY_REG(Pooling)

/**
*@brief Performs average pooling on the input . \n

*@par Inputs:
*x: A tensor of type float16, float32, double . \n

*@par Attributes:
*@li ksize: A required list of 4 ints, specifying the size (N, C, H, and W) of the sliding window, where N = C = 1, and H and W are positive integers within the range [1, 32768].
*@li strides: A required list of 4 ints, specifying the stride of the sliding window. The strides of the N and C dimensions are 1. The strides of the H and W dimensions are positive integers within the range [1, 63].
*@li padding: A required string, specifying the padding algorithm, either "VALID" or "SAME". With "SAME" means that the outputs will have the same spatial dimensions as its inputs. With "VALID" means no padding.
*@li data_format: An optional string, specifying the data format of "ksize" and "strides", either "NCHW", "NC1HWC0", or "NHWC" (default) . \n

*@par Outputs:
*y: The average pooled output tensor. Has the same type and format as input "x" . \n

*@attention Constraints:
*@li This operator applies only to a TensorFlow network.
*@li Only single input and single output are supported.
*@li Global pooling is supported.
*@li "ksize_H" and "ksize_W" are positive integers within the range [1, 32768]. ksize_H * ksize_W < 256
*@li Due to instruction restrictions, the values of "strides_h" and "strides_w" are positive integers within the range [1, 63].
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPool.
*/
REG_OP(AvgPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(AvgPool)

/**
*@brief Performs average pooling on the input . \n

*@par Inputs:
*x: A 5-D Tensor of shape [batch, depth, height, width, channels] and type float16, float32, double . \n

*@par Attributes:
*@li ksize: List of ints that has length 1, 3 or 5. The size of the window for each dimension of the input tensor.
*@li strides:List of ints that has length 1, 3 or 5. The stride of the sliding window for each dimension of the input tensor.
*@li pads: List of ints, implicit zero paddings on both sides of the input.
*@li ceil_mode: When true, will use ceil instead of floor in the formula to compute the output shape.
*@li count_include_pad: When true, will include the zero-padding in the averaging calculation.
*@li divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.
*@li data_format: A string, format of input data . \n

*@par Outputs:
*y: The average pooled output tensor . \n

*@attention Constraints:
*@li "ksize" is in the range [1, 255]. "strides" is in the range [1, 63]

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPool3D.
*/
REG_OP(AvgPool3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, true)
    .ATTR(divisor_override, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(AvgPool3D)

/**
*@brief Performs max_pool_ext2 on the input . \n

*@par Inputs:
* One input:
*x: An NC1HWC0 Tensor of type float16.


*@par Attributes:
*@li ksize: A required list of int8, int16, int32, or int64 values, specifying the size of the window for each dimension of the input tensor. No default value.
*@li strides: A required list of int8, int16, int32, or int64 values, specifying the stride of the sliding window for each dimension of the input tensor. No default value.
*@li padding: A required string. No default value.
*@li data_format: An optional string. Defaults to "NC1HWC0" . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
*@li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1, strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
*@li "padding" is either "SAME" or "VALID" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolV2.
*/
REG_OP(MaxPoolExt2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                          DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                          DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                           DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                           DT_UINT16, DT_QINT8}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolExt2)

/**
*@brief Performs max pooling on the input . \n

*@par Inputs:
* One input:
*x: An NC1HWC0 Tensor. Supported type:float16, float32, double, int8, int16,
 * int32, int64, uint8, uint16, qint8

*@par Attributes:
*@li ksize: A required list of int8, int16, int32, or int64 values,
 * specifying the size of the window for each dimension of the input tensor.
 * No default value.
*@li strides: A required list of int8, int16, int32, or int64 values,
 * specifying the stride of the sliding window for each dimension of
 * the input tensor. No default value.
*@li padding: A required string. No default value.
*@li data_format: An optional string. Defaults to "NHWC" . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
 * ksize[1] * ksize[2] <= 255.
*@li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1,
 * strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
*@li "padding" is either "SAME" or "VALID".


*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPool.
*/
REG_OP(MaxPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                          DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                          DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                           DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPool)

/**
*@brief Performs max 3d pooling on the input . \n

*@par Inputs:
*x: An NC1HWC0 Tensor. Supported type float16, float32, double . \n

*@par Attributes:
*@li ksize: A required list of int8, int16, int32, or int64 values,
specifying the size of the window for each dimension of the input tensor.
No default value.
*@li strides: A required list of int8, int16, int32, or int64 values,
specifying the stride of the sliding window for each dimension of
the input tensor. No default value.
*@li padding: A required string type of float16.
*@li pads: A list type of int32. Default value {0, 0, 0}.
*@li dilation: A list type of int32. Default value {1, 1, 1}.
*@li ceil_mode: A ceil mode number of int32 . Default value 0.
*@li data_format: An optional string. Defaults to "NDHWC" . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
 * ksize[1] * ksize[2] <= 255.
*@li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1,
 * strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
*@li "padding" is either "SAME" or "VALID" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPool3D.
*/
REG_OP(MaxPool3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(pads, ListInt, {0,0,0})
    .ATTR(dilation, ListInt, {1,1,1})
    .ATTR(ceil_mode, Int, 0)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(MaxPool3D)


/**
* @brief Computes second-order gradients of the maxpooling3d function . \n

* @par Inputs:
* @li orig_x: Original forward input tensor(NDC1HWC0) of type float16
* @li orig_y: Original forward output tensor(NDC1HWC0) of type float16
* @li grads: Gradient tensor(NDC1HWC0) of type float16
* @li assist: Assist tensor(NDC1HWC0) of type float16

* @par Attributes:
* @li ksize: A required list or tuple,
* specifying the size of the sliding window.
* @li strides: A required list or tuple,
* specifying the stride of the sliding window.
* @li pads: A required list or tuple
* @li padding: A required string, window sliding mode. Either SAME or VALID.
* @li data_format: An optional string.
* Format of the original input, either NCDHW or NDHWC. Defaults to NDHWC . \n

* @attention Constraints:
* @li Only the Ascend 910 platform is supported.
* @li "orig_x" and "grads" must have the same shape.
* @li "orig_y" and "y" must have the same shape. Otherwise, an error is reported.
* @li "orig_x", "orig_y", "grads", and "y" must be NDC1HWC0 tensors . \n

* @par Outputs:
* @li y: Result tensor of type float16

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator MaxPool3DGradGrad.
*/

REG_OP(MaxPool3DGradGrad)
    .INPUT(orig_x, TensorType::RealNumberType())
    .INPUT(orig_y, TensorType::RealNumberType())
    .INPUT(grads, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(MaxPool3DGradGrad)


/**
* @brief Computes gradients of the maxpooling function . \n

* @par Inputs:
* @li x1: A mutable NC1HWC0 tensor of type RealNumberType.
* @li x2: A mutable NC1HWC0 tensor of type RealNumberTypex.
* @li grad: A mutable NC1HWC0 tensor of type RealNumberType . \n

* @par Attributes:
* @li ksize: A required tuple or list, specifying the size of the window for
* each dimension of the input tensor.
* @li strides: A required tuple or list, specifying the stride of the sliding
* window for each dimension of the input tensor.
* @li padding: A required string, specifying the type of padding algorithm
* to use.
* @li data_format: An optional string, Specify the data format of the input and
* output data. With the default format "NHWC" . \n

* @par Outputs:
* y: A mutable tensor. Has the same shape and type as "x1" . \n

* @attention Constraints:
* @li Computing gradients of global pooling is not supported, which means
* "ksize < x1".
* @li "ksize" is in the range [1, 255]. "strides" is in the range [1, 63]

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolGrad.
*/
REG_OP(MaxPoolGrad)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolGrad)

/**
* @brief Computes second-order gradients of the maxpooling function . \n

* @par Inputs:
* @li x1: Original forward input tensor. Supported type:float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
* @li x2: Has the same type and format as input "x1".
* @li grad:Has the same type and format as input "x1" . \n

* @par Attributes:
* @li ksize: A required list or tuple,
* specifying the size of the sliding window.
* @li strides: A required list or tuple,
* specifying the stride of the sliding window.
* @li padding: A required string, window sliding mode. Either SAME or VALID.
* @li data_format: An optional string.
* Format of the original input, either NCHW or NHWC. Defaults to NHWC . \n

* @attention Constraints:
* @li Only the Ascend 910 platform is supported.
* @li "x1" and "grads" must have the same shape.
* @li "x2" and "y" must have the same shape. Otherwise, an error is reported.
* @li "x1", "x2", "grads", and "y" must be 5D tensors.
* @li ksize[H] and ksize[W] is in the range [1, 255].
* @li strides[H] and strides[W] is in the range [1, 63].
* @li Other dimensions of ksize and strides is 1 . \n

* @par Outputs:
* @li y: Has the same type and format as input "x1" . \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator MaxPoolGradGrad.
*/
REG_OP(MaxPoolGradGrad)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolGradGrad)

/**
*@brief Performs max_pool_ext2 on the input . \n

*@par Inputs:
* Two inputs:
*@li x: An NC1HWC0 Tensor of type float16.
*@li strides: A required type of int32 values, specifying the stride of the sliding window for each dimension of the input tensor. No default value.
*@li ksize: A required type of int32 values, specifying the size of the window for each dimension of the input tensor. No default value.


*@par Attributes:
*@li padding: A required string. No default value.
*@li data_format: An optional string. Defaults to "NC1HWC0" . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
*@li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1, strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
*@li "padding" is either "SAME" or "VALID" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolV2.
*/
REG_OP(MaxPoolV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(ksize, TensorType({DT_INT32}))
    .INPUT(strides, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolV2)

/**
*@brief Performs max pooling on the input and outputs both max values and
 * indices . \n

*@par Inputs:
* One input:
*x: An NC1HWC0 Tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64 . \n

*@par Attributes:
*@li ksize: A required list of int8, int16, int32, or int64 values,
 * specifying the size of the window for each dimension of the input tensor.
 * No default value.
*@li strides: A required list of int8, int16, int32, or int64 values,
 * specifying the stride of the sliding window for each dimension of
 * the input tensor. No default value.
*@li padding: A required string. No default value . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".
*argmax: A Tensor. Has the same type and format as input "x".
*@attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
 * ksize[1] * ksize[2] <= 255.
*@li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1,
 * strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
*@li "padding" is either "SAME" or "VALID" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolWithArgmax.
*/
REG_OP(MaxPoolWithArgmax)
    .INPUT(x, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OUTPUT(argmax, TensorType::IndexNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(Targmax, Int, 7)
    .OP_END_FACTORY_REG(MaxPoolWithArgmax)

/**
*@brief Performs the backpropagation of MaxPoolWithArgmax . \n

*@par Inputs:
* Three inputs, including:
*@li x: An NC1HWC0 tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
*@li grad: An NC1HWC0 tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
*@li argmx: An NC1HWC0 tensor of type int32 or int64 . \n

*@par Attributes:
*@li ksize: A required list of int8, int16, int32, or int64 values,
 * specifying the size of the window for each dimension of the input tensor.
 * No default value.
*@li strides: A required list of int8, int16, int32, or int64 values,
 * specifying the stride of the sliding window for each dimension of
 * the input tensor. No default value.
*@li padding: A required string. No default value . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
 * ksize[1] * ksize[2] <= 255.
*@li "strides" is a list that has length 4: strides[0] = 1 or strides[3] = 1
*@li "padding" is either "SAME" or "VALID".


*@see max_pool_with_argmax
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolGradWithArgmax.
*/
REG_OP(MaxPoolGradWithArgmax)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .INPUT(argmax, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .OP_END_FACTORY_REG(MaxPoolGradWithArgmax)

/**
*@brief Performs transform mask to argmax . \n

*@par Inputs:
* Two input:
*x: An NC1HWC0 Tensor of type float16.
*mask: An NC1HWC0 Tensor of type uint16 . \n

*@par Attributes:
*@li ksize: A required list of int8, int16, int32, or int64 values, specifying the size of the window for each dimension of the input tensor. No default value.
*@li strides: A required list of int8, int16, int32, or int64 values, specifying the stride of the sliding window for each dimension of the input tensor. No default value.
*@li padding: A required string. No default value . \n

*@par Outputs:
*argmax: An NC1HWC0 Tensor of type int32 . \n

*@attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
*@li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1, strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1.
*@li "padding" is either "SAME" or "VALID" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Mask2Argmax.
*/
REG_OP(Mask2Argmax)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(mask, TensorType::IndexNumberType())
    .OUTPUT(argmax, TensorType::IndexNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .REQUIRED_ATTR(originshape, ListInt)
    .OP_END_FACTORY_REG(Mask2Argmax)

/**
* @brief Computes second-order gradients of the maxpooling function . \n

* @par Inputs:
* @li x: Original forward input tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
* @li grad: Gradient tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64.
* @li argmax: An tensor of type int32 or int64.
* @par Attributes:
* @li ksize: A required list, specifying the size of the sliding window.
* @li strides: A required list, specifying the stride of the sliding window.
* @li padding: A required string, window sliding mode. Either SAME or VALID.
* @par Outputs:
* @li y:Result tensor. Supported type: float, double, int32,
 * uint8, int16, int8, int64, uint16, half, uint32, uint64

* @attention Constraints:
* @li Only the cloud platform is supported.
* @li "x1" and "grads" must have the same shape.
* @li length of the shape of x, grads, argmax, y must be 5.
* @li shape of argmax must be (fmap_n, fmap_c1, kernel_h * kernel_w,
* (shape_max_pool[2] * shape_max_pool[3] + 15) // 16 * 16, 1),
* or (fmap_n, fmap_c1, kernel_h * kernel_w,
* (shape_max_pool[2] * shape_max_pool[3] + 31) // 16, 16), else failed . \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator MaxPoolGradGradWithArgmax.
*/
REG_OP(MaxPoolGradGradWithArgmax)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .INPUT(argmax, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .OP_END_FACTORY_REG(MaxPoolGradGradWithArgmax)

/**
* @brief Computes avgpoograd function . \n

* @par Inputs:
* @li orig_input_shape: An NHWC tensor of type int32.
* @li input_grad: An NHWC tensor of type float16, float32, or double . \n

* @par Attributes:
* @li ksize: A required tuple or list, specifying the size of the window for
* each dimension of the input tensor.
* @li strides: A required tuple or list, specifying the stride of the sliding
* window for each dimension of the input tensor.
* @li padding: A required string, specifying the type of
* the padding algorithm to use.
* @li data_format: An optional string. Defaults to "NHWC" . \n

* @par Outputs:
* @out_grad: A mutable tensor with the same shape and type as "orig_input" . \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator AvgPoolGrad.
*/
REG_OP(AvgPoolGrad)
    .INPUT(orig_input_shape, TensorType({DT_INT32}))
    .INPUT(input_grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(out_grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(AvgPoolGrad)

/**
* @brief Computes gradients of average pooling function . \n

* @par Inputs:
* @input_grad: An NHWC tensor of type float16.
* @mean_matrix: Assist matrix, an NHWC tensor of type float16.
* @kernel_matrix: Assist matrix, an NHWC tensor of type float16. \n

* @par Attributes:
* @li orig_input_shape: A required Original input dimensions.
* @li ksize: A required tuple or list, specifying the size of the window
* for each dimension of the input tensor.
* @li strides: A required tuple or list, specifying the stride of
* the sliding window for each dimension of the input tensor.
* @li padding: A required string, specifying the type of the padding algorithm
* to use.
* @li data_format: An optional string. Defaults to "NHWC" . \n

* @par Outputs:
* @out_grad: A mutable tensor with the same shape and type as "orig_input".
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use AvgPoolGrad instead.
*/
REG_OP(AvgPoolGradD)
    .INPUT(input_grad, TensorType({DT_FLOAT16}))
    .INPUT(mean_matrix, TensorType({DT_FLOAT16}))
    .INPUT(kernel_matrix, TensorType({DT_FLOAT16}))
    .OUTPUT(out_grad, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(orig_input_shape, ListInt)
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(AvgPoolGradD)


/**
*@brief :upsample the layer

*@par Inputs:
* one input, including:
*@li x: A tensor of type float16 or float32.
*@par Attributes:
*@li  scale: A optional float32, scale factor of x. Defaults to "1.0".
*@li  stride_h: An optional int32, broadcast the axis of h. Defaults to "2".
*@li  stride_w: An optional int32, broadcast the axis of w. Defaults to "2".
*@par Outputs:
*y: A tensor of type float16 or float32.
*/
REG_OP(Upsample)
   .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
   .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
   .ATTR(scale, Float, 1)
   .ATTR(stride_h, Int, 2)
   .ATTR(stride_w, Int, 2)
   .OP_END_FACTORY_REG(Upsample)

/**
*@brief Computes gradient of the FractionalMaxPool function . \n

*@par Inputs:
*Inputs include:
* @li orig_input: A Tensor. Must be one of the following types: float32, float64, int32, int64.
* @li orig_output: A Tensor. Must have the same type as orig_input.
* @li out_backprop: A Tensor. Must have the same type as orig_input.
      4-D with shape [batch, height, width, channels].
* @li row_pooling_sequence: A Tensor of type int64.
* @li col_pooling_sequence: A Tensor of type int64 . \n

*@par Attributes:
*overlapping: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as orig_input . \n

*@attention Constraints:
*The implementation for FractionalMaxPoolGrad on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow FractionalMaxPoolGrad operator.
*/
REG_OP(FractionalMaxPoolGrad)
    .INPUT(orig_input, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(orig_output, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(out_backprop, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(row_pooling_sequence, TensorType({ DT_INT64 }))
    .INPUT(col_pooling_sequence, TensorType({ DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64 }))
    .ATTR(overlapping, Bool, false)
    .OP_END_FACTORY_REG(FractionalMaxPoolGrad)

/**
*@brief Performs fractional average pooling on the input . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: float32, float64, int32, int64.
 4-D with shape [batch, height, width, channels] . \n

*@par Attributes:
*@li pooling_ratio: A list of floats that has length >= 4.
*@li pseudo_random: An optional bool. Defaults to False.
*@li overlapping: An optional bool. Defaults to False. When set to True, it means when pooling.
*@li deterministic: An optional bool. Defaults to False.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*@li y: A Tensor. Has the same type as x.
*@li row_pooling_sequence: A Tensor of type int64.
*@li col_pooling_sequence: A Tensor of type int64 . \n

*@attention Constraints:
*The implementation for FractionalAvgPool on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow FractionalAvgPool operator.
*/
REG_OP(FractionalAvgPool)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(row_pooling_sequence, TensorType({DT_INT64}))
    .OUTPUT(col_pooling_sequence, TensorType({DT_INT64}))
    .ATTR(pooling_ratio, ListFloat, {})
    .ATTR(pseudo_random, Bool, false)
    .ATTR(overlapping, Bool, false)
    .ATTR(deterministic, Bool, false)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(FractionalAvgPool)

/**
*@brief Performs fractional max pooling on the input . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: float32, float64, int32, int64.
 4-D with shape [batch, height, width, channels] . \n

*@par Attributes:
*@li pooling_ratio: A list of floats that has length >= 4. Pooling ratio for each dimension of value.
*@li pseudo_random: An optional bool. Defaults to False.
*@li overlapping: An optional bool. Defaults to False.
*@li deterministic: An optional bool. Defaults to False.
*@li seed: An optional int. Defaults to 0.
*@li seed2: An optional int. Defaults to 0 . \n

*@par Outputs:
*@li y: A Tensor. Has the same type as x.
*@li row_pooling_sequence: A Tensor of type int64.
*@li col_pooling_sequence: A Tensor of type int64 . \n

*@attention Constraints:
*The implementation for FractionalMaxPool on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow FractionalMaxPool operator.
*/
REG_OP(FractionalMaxPool)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .OUTPUT(row_pooling_sequence, TensorType({DT_INT64}))
    .OUTPUT(col_pooling_sequence, TensorType({DT_INT64}))
    .ATTR(pooling_ratio, ListFloat, {})
    .ATTR(pseudo_random, Bool, false)
    .ATTR(overlapping, Bool, false)
    .ATTR(deterministic, Bool, false)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(FractionalMaxPool)

/**
*@brief Finds values of the n-th order statistic for the last dimension . \n

*@par Inputs:
*Inputs include:
* @li x: A Tensor. Must be one of the following types: float32, float64, int32, uint8,
      int16, int8, int64, bfloat16, uint16, half, uint32, uint64.
* @li n: A Tensor of type int32. 0-D . \n

*@par Attributes:
*reverse: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for NthElement on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow NthElement operator.
*/
REG_OP(NthElement)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                          DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(n, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                          DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(NthElement)

/**
*@brief Computes gradient of the FractionalAvgPool function . \n

*@par Inputs:
*Inputs include:
* @li orig_input_tensor_shape: A Tensor of type int64.
* @li out_backprop: A Tensor. Must be one of the following types: float32, float64,
      int32, int64. 4-D with shape [batch, height, width, channels].
* @li row_pooling_sequence: A Tensor of type int64.
* @li col_pooling_sequence: A Tensor of type int64 . \n

*@par Attributes:
*overlapping: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as out_backprop . \n

*@attention Constraints:
*The implementation for FractionalAvgPoolGrad on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow FractionalAvgPoolGrad operator.
*/
REG_OP(FractionalAvgPoolGrad)
    .INPUT(orig_input_tensor_shape, TensorType({DT_INT64}))
    .INPUT(out_backprop, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .INPUT(row_pooling_sequence, TensorType({DT_INT64}))
    .INPUT(col_pooling_sequence, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .ATTR(overlapping, Bool, false)
    .OP_END_FACTORY_REG(FractionalAvgPoolGrad)

/**
*@brief Returns the permuted vector/tensor in the destination data format given the . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: int32, int64. Vector of size 4
 or Tensor of shape (4, 2) in source data format . \n

*@par Attributes:
*@li src_format: An optional string. Defaults to "NHWC". source data format.
*@li dst_format: An optional string. Defaults to "NCHW". destination data format . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for DataFormatVecPermute on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow DataFormatVecPermute operator.
*/
REG_OP(DataFormatVecPermute)
    .INPUT(x, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT32, DT_INT64 }))
    .ATTR(src_format, String, "NHWC")
    .ATTR(dst_format, String, "NCHW")
    .OP_END_FACTORY_REG(DataFormatVecPermute)

/**
* @brief Computes gradients of the MaxPool3D function . \n

* @par Inputs:
* @li orig_x: A mutable NDC1HWC0 tensor of type float16.
* @li orig_y: A mutable NDC1HWC0 tensor of type float16.
* @li grads: A mutable NDC1HWC0 tensor of type float16 . \n

* @par Attributes:
* @li ksize: A required tuple or list, specifying the size of the window for
* each dimension of the input tensor.
* @li strides: A required tuple or list, specifying the stride of the sliding
* window for each dimension of the input tensor.
* @li pads: A list of 6 ints. Supports only padding along the D,
* H and W dimensions in sequence of head, tail, top, bottom, left and right.
* to use.
* @li data_format: An optional string, Specify the data format of the input and
* output data. With the default format "NDHWC" . \n

* @par Outputs:
* y: A mutable tensor. Has the same shape as "orig_x", but type is float32 . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPool3DGrad.
*/
REG_OP(MaxPool3DGrad)
    .INPUT(orig_x, TensorType::RealNumberType())
    .INPUT(orig_y, TensorType::RealNumberType())
    .INPUT(grads, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(MaxPool3DGrad)

/**
*@brief Performs AvgPool1D on the input . \n

*@par Inputs:
*x: A Tensor. Must be one of the following types: int8, uint8, int16, int32, int64, float16, float32, float64 . \n

*@par Attributes:
*@li ksize: An required int, specifying the size of the window.
*@li strides: An required int.
*@li pads: A required tuple or list.
*@li ceil_mode: An optional bool. Defaults to False.
*@li count_include_pad: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@par Third-party framework compatibility
*@li compatible with pytorch AvgPool1D operator.
*/
REG_OP(AvgPool1D)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, Int)
    .REQUIRED_ATTR(strides, Int)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, false)
    .OP_END_FACTORY_REG(AvgPool1D)

/**
*@brief Performs AvgPool1D on the input . \n

*@par Inputs:
*x: A Tensor. Must be one of the following types: int8, uint8, int16, int32, int64, float16, float32, float64 . \n

*@par Attributes:
*@li ksize: An required int, specifying the size of the window.
*@li strides: An required int.
*@li pads: A required tuple or list.
*@li ceil_mode: An optional bool. Defaults to False.
*@li count_include_pad: An optional bool. Defaults to False . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@par Third-party framework compatibility
*@li compatible with pytorch AvgPool1D operator.
*
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use AvgPool1D instead.
*/
REG_OP(AvgPool1DD)
    .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(assist_matrix, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, Int)
    .REQUIRED_ATTR(strides, Int)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(ceil_mode, Bool, false)
    .ATTR(count_include_pad, Bool, false)
    .OP_END_FACTORY_REG(AvgPool1DD)
/**
*@brief Performs max pooling on the input and outputs both max values and indices . \n

*@par Inputs:
* One input:
*x: An NC1HWC0 Tensor of type float16.
*@par Attributes:
*@li ksize: A required list of int8, int16, int32, or int64 values, specifying the size of the window for
* each dimension of the input tensor. No default value.
*@li strides: A required list of int8, int16, int32, or int64 values, specifying the stride of the sliding window for
* each dimension of the input tensor. No default value.
*@li pads: A required string. No default value.
*@li dtype: A optional int. default value is 3.
*@li dilation: A optional list of int8, int16, int32, or int64 values.
*@li ceil_mode: A optional bool. default value is false . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x".
*argmax:  A Tensor. type:uint16, format:NC1HWC0.
*@attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
*@li "strides is a list that has length 4: strides[0] = 1 or strides[3] = 1, strides[1] <= 63, strides[0] >= 1,
* strides[2] <= 63, strides[2] >= 1.
*@li "dilation" is a list that has length 4.
*@li "ceil_mode" is a bool, default is false . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolWithArgmax.
*/
REG_OP(MaxPoolWithArgmaxV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OUTPUT(argmax, TensorType({DT_UINT16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1, 1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolWithArgmaxV2)

/**
*@brief Performs the backpropagation of MaxPoolWithArgmaxV2 . \n

*@par Inputs:
* Three inputs, including:
*@li x: An NC1HWC0 tensor of type float16.
*@li grad: An NC1HWC0 tensor of type float16.
*@li argmx: An NC1HWC0 tensor of type uint16 or int64 . \n

*@par Attributes:
*@li ksize: A required list of int8, int16, int32, or int64 values, specifying the size of the window for
 * each dimension of the input tensor. No default value.
*@li strides: A required list of int8, int16, int32, or int64 values, specifying the stride of the sliding window for
 * each dimension of the input tensor. No default value.
*@li pads: A required string. No default value.
*@li dtype: A optional int. default value is 3.
*@li dilation: A optional list of int8, int16, int32, or int64 values.
*@li ceil_mode: A optional bool. default value is false . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as input "x" . \n

*@attention Constraints:
*@li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
*@li "strides" is a list that has length 4: strides[0] = 1 or strides[3] = 1
*@li "dilation" is a list that has length 4.
*@li "ceil_mode" is a bool, default is false . \n

*@see max_pool_grad_with_argmaxv2
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolGradWithArgmaxV2.
*/

REG_OP(MaxPoolGradWithArgmaxV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(grad, TensorType({DT_FLOAT16}))
    .INPUT(argmax, TensorType({DT_UINT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1,1,1,1})
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolGradWithArgmaxV2)
}  // namespace ge

#endif  // GE_OP_NN_POOLING_OPS_H
