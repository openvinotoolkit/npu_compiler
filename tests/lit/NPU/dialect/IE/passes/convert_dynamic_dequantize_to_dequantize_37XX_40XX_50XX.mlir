//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-dynamic-dequantize-to-dequantize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertForDirectConnect
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4096x4096x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4096x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4096xf16>
func.func @ConvertForDirectConnect(%arg0: tensor<4096x4096x!qElemType>, %arg1: tensor<4096x1xf16>, %arg2: tensor<1x4096xf16>) -> tensor<1x4096xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    %1 = IE.FullyConnected(%arg2, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>

    return %1 : tensor<1x4096xf16>

    // CHECK:  [[SCALE_RESHAPE:%.+]] = IE.Reshape([[INPUT_1]]) {shape_value = [1, 4096]} : tensor<4096x1xf16> -> tensor<1x4096xf16>
    // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<4096x4096x!qElemType> -> tensor<4096x4096xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[DEQUANT]]) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[SCALE_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4096xf16>, tensor<1x4096xf16> -> tensor<1x4096xf16>
    // CHECK:  return [[MULTIPLY]] : tensor<1x4096xf16>
}

// -----

!qElemType = !quant.uniform<i2:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertForDirectConnectI2
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4096x4096x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4096x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4096xf16>
func.func @ConvertForDirectConnectI2(%arg0: tensor<4096x4096x!qElemType>, %arg1: tensor<4096x1xf16>, %arg2: tensor<1x4096xf16>) -> tensor<1x4096xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    %1 = IE.FullyConnected(%arg2, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>

    return %1 : tensor<1x4096xf16>

    // CHECK:  [[SCALE_RESHAPE:%.+]] = IE.Reshape([[INPUT_1]]) {shape_value = [1, 4096]} : tensor<4096x1xf16> -> tensor<1x4096xf16>
    // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<4096x4096x!qElemType> -> tensor<4096x4096xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[DEQUANT]]) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[SCALE_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4096xf16>, tensor<1x4096xf16> -> tensor<1x4096xf16>
    // CHECK:  return [[MULTIPLY]] : tensor<1x4096xf16>
}

// -----

!qElemType = !quant.uniform<u2:f16, 1.000000e+00:2>

// CHECK-LABEL: @ConvertForDirectConnectU2WithSymmetricZP
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4096x4096x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4096x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4096xf16>
func.func @ConvertForDirectConnectU2WithSymmetricZP(%arg0: tensor<4096x4096x!qElemType>, %arg1: tensor<4096x1xf16>, %arg2: tensor<1x4096xf16>) -> tensor<1x4096xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    %1 = IE.FullyConnected(%arg2, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>

    return %1 : tensor<1x4096xf16>

    // CHECK:  [[SCALE_RESHAPE:%.+]] = IE.Reshape([[INPUT_1]]) {shape_value = [1, 4096]} : tensor<4096x1xf16> -> tensor<1x4096xf16>
    // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<4096x4096x!qElemType> -> tensor<4096x4096xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[DEQUANT]]) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[SCALE_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4096xf16>, tensor<1x4096xf16> -> tensor<1x4096xf16>
    // CHECK:  return [[MULTIPLY]] : tensor<1x4096xf16>
}

// -----

!qElemType = !quant.uniform<u2:f16, 1.000000e+00:1>

// CHECK-LABEL: @ConvertForDirectConnectU2WithAsymmetricZP
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4096x4096x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4096x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4096xf16>
func.func @ConvertForDirectConnectU2WithAsymmetricZP(%arg0: tensor<4096x4096x!qElemType>, %arg1: tensor<4096x1xf16>, %arg2: tensor<1x4096xf16>) -> tensor<1x4096xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    %1 = IE.FullyConnected(%arg2, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>

    return %1 : tensor<1x4096xf16>

    // CHECK:  [[SCALE_RESHAPE:%.+]] = IE.Reshape([[INPUT_1]]) {shape_value = [1, 4096]} : tensor<4096x1xf16> -> tensor<1x4096xf16>
    // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<4096x4096x!qElemType> -> tensor<4096x4096xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[DEQUANT]]) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[SCALE_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4096xf16>, tensor<1x4096xf16> -> tensor<1x4096xf16>
    // CHECK:  return [[MULTIPLY]] : tensor<1x4096xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @NotConvertForDirectConnectDueToShapeMismatch
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4096x4096x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x4096xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4096xf16>
func.func @NotConvertForDirectConnectDueToShapeMismatch(%arg0: tensor<4096x4096x!qElemType>, %arg1: tensor<1x4096xf16>, %arg2: tensor<1x4096xf16>) -> tensor<1x4096xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<1x4096xf16> -> tensor<4096x4096xf16>
    %1 = IE.FullyConnected(%arg2, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>

    return %1 : tensor<1x4096xf16>

    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.DynamicDequantize
    // CHECK:  [[FC:%.+]] = IE.FullyConnected
    // CHECK:  return [[FC]] : tensor<1x4096xf16>
}


// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertForDirectConnectOnlyOneScale
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4096x4096x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4096xf16>
func.func @ConvertForDirectConnectOnlyOneScale(%arg0: tensor<4096x4096x!qElemType>, %arg1: tensor<1x1xf16>, %arg2: tensor<1x4096xf16>) -> tensor<1x4096xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<1x1xf16> -> tensor<4096x4096xf16>
    %1 = IE.FullyConnected(%arg2, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>

    return %1 : tensor<1x4096xf16>

    // CHECK:  [[DEQUANT:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<4096x4096x!qElemType> -> tensor<4096x4096xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[DEQUANT]]) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4096xf16>, tensor<1x1xf16> -> tensor<1x4096xf16>
    // CHECK:  return [[MULTIPLY]] : tensor<1x4096xf16>
}


// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertForReshapeTranpose
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x128x512x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x512xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @ConvertForReshapeTranpose(%arg0: tensor<1x128x512x!qElemType>, %arg1: tensor<1x1x512xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x128x512x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x1x512xf16> -> tensor<1x128x512xf16>
    %1 = IE.Reshape(%0) {shape_value = [128, 512]} : tensor<1x128x512xf16> -> tensor<128x512xf16>
    %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<128x512xf16> -> tensor<512x128xf16>
    %3 = IE.FullyConnected(%arg2, %2) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %3 : tensor<1x512xf16>

    // CHECK:  [[RESHAPE_SCALE:%.+]] = IE.Reshape([[INPUT_1]]) {shape_value = [1, 512]} : tensor<1x1x512xf16> -> tensor<1x512xf16>
    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<1x128x512x!qElemType> -> tensor<1x128x512xf16>
    // CHECK:  [[RESHAPE_IN:%.+]] = IE.Reshape([[DYN_DEQUANTIZE]]) {shape_value = [128, 512]} : tensor<1x128x512xf16> -> tensor<128x512xf16>
    // CHECK:  [[TRANSPOSE:%.+]]  = IE.Transpose([[RESHAPE_IN]]) {order_value = #CN} : tensor<128x512xf16> -> tensor<512x128xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[TRANSPOSE]]) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[RESHAPE_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512xf16>, tensor<1x512xf16> -> tensor<1x512xf16>
    // CHECK:  return  [[MULTIPLY]] : tensor<1x512xf16>
}


// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @NotConvertForReshapeTranposeDueToShapeMismatch
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x128x512x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x128x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @NotConvertForReshapeTranposeDueToShapeMismatch(%arg0: tensor<1x128x512x!qElemType>, %arg1: tensor<1x128x1xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x128x512x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x128x1xf16> -> tensor<1x128x512xf16>
    %1 = IE.Reshape(%0) {shape_value = [128, 512]} : tensor<1x128x512xf16> -> tensor<128x512xf16>
    %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<128x512xf16> -> tensor<512x128xf16>
    %3 = IE.FullyConnected(%arg2, %2) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %3 : tensor<1x512xf16>

    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.DynamicDequantize
    // CHECK:  [[RESHAPE:%.+]] =  IE.Reshape
    // CHECK:  [[TRANSPOSE:%.+]] =  IE.Transpose
    // CHECK:  [[FC:%.+]] = IE.FullyConnected
    // CHECK:  return [[FC]] : tensor<1x512xf16>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertForTransposeReshape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x128x512x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x512xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @ConvertForTransposeReshape(%arg0: tensor<1x128x512x!qElemType>, %arg1: tensor<1x1x512xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x128x512x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x1x512xf16> -> tensor<1x128x512xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<1x128x512xf16> -> tensor<1x512x128xf16>
    %2 = IE.Reshape(%1) {shape_value = [512, 128]} : tensor<1x512x128xf16> -> tensor<512x128xf16>
    %3 = IE.FullyConnected(%arg2, %2) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %3 : tensor<1x512xf16>

    // CHECK:  [[RESHAPE_SCALE:%.+]] = IE.Reshape([[INPUT_1]]) {shape_value = [1, 512]} : tensor<1x1x512xf16> -> tensor<1x512xf16>
    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<1x128x512x!qElemType> -> tensor<1x128x512xf16>
    // CHECK:  [[TRANSPOSE:%.+]]  = IE.Transpose([[DYN_DEQUANTIZE]]) {order_value = #map} : tensor<1x128x512xf16> -> tensor<1x512x128xf16>
    // CHECK:  [[RESHAPE_IN:%.+]] = IE.Reshape([[TRANSPOSE]]) {shape_value = [512, 128]} : tensor<1x512x128xf16> -> tensor<512x128xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[RESHAPE_IN]]) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[RESHAPE_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512xf16>, tensor<1x512xf16> -> tensor<1x512xf16>
    // CHECK:  return  [[MULTIPLY]] : tensor<1x512xf16>
}


// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @NotConvertForTransposeReshapeDueToShapeMismatch
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x128x512x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x128x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @NotConvertForTransposeReshapeDueToShapeMismatch(%arg0: tensor<1x128x512x!qElemType>, %arg1: tensor<1x128x1xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x128x512x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x128x1xf16> -> tensor<1x128x512xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<1x128x512xf16> -> tensor<1x512x128xf16>
    %2 = IE.Reshape(%1) {shape_value = [512, 128]} : tensor<1x512x128xf16> -> tensor<512x128xf16>
    %3 = IE.FullyConnected(%arg2, %2) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %3 : tensor<1x512xf16>

    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.DynamicDequantize
    // CHECK:  [[TRANSPOSE:%.+]] =  IE.Transpose
    // CHECK:  [[RESHAPE:%.+]] =  IE.Reshape
    // CHECK:  [[FC:%.+]] = IE.FullyConnected
    // CHECK:  return [[FC]] : tensor<1x512xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertForReshapeOnly
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x512x128x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x512x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @ConvertForReshapeOnly(%arg0: tensor<1x512x128x!qElemType>, %arg1: tensor<1x512x1xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x512x128x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x512x1xf16> -> tensor<1x512x128xf16>
    %1 = IE.Reshape(%0) {shape_value = [512, 128]} : tensor<1x512x128xf16> -> tensor<512x128xf16>
    %2 = IE.FullyConnected(%arg2, %1) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %2 : tensor<1x512xf16>

    // CHECK:  [[RESHAPE_SCALE:%.+]] = IE.Reshape([[INPUT_1]]) {shape_value = [1, 512]} : tensor<1x512x1xf16> -> tensor<1x512xf16>
    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<1x512x128x!qElemType> -> tensor<1x512x128xf16>
    // CHECK:  [[RESHAPE_IN:%.+]] = IE.Reshape([[DYN_DEQUANTIZE]]) {shape_value = [512, 128]} : tensor<1x512x128xf16> -> tensor<512x128xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[RESHAPE_IN]]) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[RESHAPE_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512xf16>, tensor<1x512xf16> -> tensor<1x512xf16>
    // CHECK:  return  [[MULTIPLY]] : tensor<1x512xf16>
}


// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @NotConvertForReshapeOnlyDueToShapeMismatch
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x512x128x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x128xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @NotConvertForReshapeOnlyDueToShapeMismatch(%arg0: tensor<1x512x128x!qElemType>, %arg1: tensor<1x1x128xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x512x128x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x1x128xf16> -> tensor<1x512x128xf16>
    %1 = IE.Reshape(%0) {shape_value = [512, 128]} : tensor<1x512x128xf16> -> tensor<512x128xf16>
    %2 = IE.FullyConnected(%arg2, %1) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %2 : tensor<1x512xf16>

    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.DynamicDequantize
    // CHECK:  [[RESHAPE:%.+]] =  IE.Reshape
    // CHECK:  [[FC:%.+]] = IE.FullyConnected
    // CHECK:  return [[FC]] : tensor<1x512xf16>

}


// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertForTransposeOnly
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<128x512x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x512xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @ConvertForTransposeOnly(%arg0: tensor<128x512x!qElemType>, %arg1: tensor<1x512xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<128x512x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x512xf16> -> tensor<128x512xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<128x512xf16> -> tensor<512x128xf16>
    %2 = IE.FullyConnected(%arg2, %1) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %2 : tensor<1x512xf16>

    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<128x512x!qElemType> -> tensor<128x512xf16>
    // CHECK:  [[TRANSPOSE:%.+]] = IE.Transpose([[DYN_DEQUANTIZE]]) {order_value = #CN} : tensor<128x512xf16> -> tensor<512x128xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[TRANSPOSE]]) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512xf16>, tensor<1x512xf16> -> tensor<1x512xf16>
    // CHECK:  return  [[MULTIPLY]] : tensor<1x512xf16>
}


// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @NotConvertForTransposeOnlyDueToShapeMismatch
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<128x512x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<128x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @NotConvertForTransposeOnlyDueToShapeMismatch(%arg0: tensor<128x512x!qElemType>, %arg1: tensor<128x1xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<128x512x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<128x1xf16> -> tensor<128x512xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<128x512xf16> -> tensor<512x128xf16>
    %2 = IE.FullyConnected(%arg2, %1) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %2 : tensor<1x512xf16>

    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.DynamicDequantize
    // CHECK:  [[TRANSPOSE:%.+]] =  IE.Transpose
    // CHECK:  [[FC:%.+]] = IE.FullyConnected
    // CHECK:  return [[FC]] : tensor<1x512xf16>

}


// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @NotConvertForReshapeOnlyDueToNotSqueezeReshape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x128x512x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x512xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @NotConvertForReshapeOnlyDueToNotSqueezeReshape(%arg0: tensor<1x128x512x!qElemType>, %arg1: tensor<1x1x512xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x128x512x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x1x512xf16> -> tensor<1x128x512xf16>
    %1 = IE.Reshape(%0) {shape_value = [512, 128]} : tensor<1x128x512xf16> -> tensor<512x128xf16>
    %2 = IE.FullyConnected(%arg2, %1) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %2 : tensor<1x512xf16>

    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.DynamicDequantize
    // CHECK:  [[RESHAPE:%.+]] =  IE.Reshape
    // CHECK:  [[FC:%.+]] = IE.FullyConnected
    // CHECK:  return [[FC]] : tensor<1x512xf16>

}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @NotConvertHasZPInput
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4096x4096x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4096x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4096xf16>
func.func @NotConvertHasZPInput(%arg0: tensor<4096x4096x!qElemType>, %arg1: tensor<4096x1xf16>, %arg2: tensor<4096x1xsi4>, %arg3: tensor<1x4096xf16>) -> tensor<1x4096xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1, %arg2) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<4096x1xf16>, tensor<4096x1xsi4> -> tensor<4096x4096xf16>
    %1 = IE.FullyConnected(%arg3, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>

    return %1 : tensor<1x4096xf16>

    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.DynamicDequantize
    // CHECK:  [[FC:%.+]] = IE.FullyConnected
    // CHECK:  return [[FC]] : tensor<1x4096xf16>
}


// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @NotConvertForMultiAxes
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4096x4096x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4096x4096xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4096xf16>
func.func @NotConvertForMultiAxes(%arg0: tensor<4096x4096x!qElemType>, %arg1: tensor<4096x4096xf16>, %arg2: tensor<1x4096xf16>) -> tensor<1x4096xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<4096x4096xf16> -> tensor<4096x4096xf16>
    %1 = IE.FullyConnected(%arg2, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>

    return %1 : tensor<1x4096xf16>

    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.DynamicDequantize
    // CHECK:  [[FC:%.+]] = IE.FullyConnected
    // CHECK:  return [[FC]] : tensor<1x4096xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<i8:f16, 0.055118110030889511>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForI8WeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960xsi8>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForI8WeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960xsi8>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<1536x8960xsi8> -> tensor<1536x8960x!qElemType>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>

    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960xsi8> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<1.814060e+01> : tensor<1xf16>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<i8:f16, 0.055118110030889511>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForI8WeightsWithExternalZPCorrection
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960xsi8>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>,
// CHECK-SAME:      [[INPUT_3:%.+]]: tensor<1536x1xf16>
func.func @RescaleForI8WeightsWithExternalZPCorrection(%arg0: tensor<1x8960xf16>, %arg1: tensor<1536x8960xsi8>, %arg2: tensor<1536x1xf16>, %arg3: tensor<1536x1xf16>) -> tensor<1x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<1536x8960xsi8> -> tensor<1536x8960x!qElemType>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1x8960xf16>, tensor<1536x8960xf16> -> tensor<1x1536xf16>

    %3 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x8960xf16> -> tensor<1x1xf16>
    %4 = IE.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1536x1xf16>, tensor<1536x1xf16> -> tensor<1536x1xf16>
    %5 = IE.Reshape(%4) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    %6 = IE.Multiply(%3, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xf16>, tensor<1x1536xf16> -> tensor<1x1536xf16>
    %7 = IE.Subtract(%2, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<1x1536xf16>

    return %7 : tensor<1x1536xf16>

    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960xsi8> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<1.814060e+01> : tensor<1xf16>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1x8960xf16>, tensor<1536x8960xf16> -> tensor<1x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<1x1536xf16>
    // CHECK:       [[REDUCESUM0:%.+]] = IE.ReduceSum([[INPUT_0]]) {axes_value = [1], keep_dims} : tensor<1x8960xf16> -> tensor<1x1xf16>
    // CHECK:       [[MULTIPLY2:%.+]] = IE.Multiply([[INPUT_2]], [[INPUT_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1536x1xf16>, tensor<1536x1xf16> -> tensor<1536x1xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[MULTIPLY2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[MULTIPLY3:%.+]] = IE.Multiply([[REDUCESUM0]], [[RESHAPE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xf16>, tensor<1x1536xf16> -> tensor<1x1536xf16>
    // CHECK:       [[SUBTRACT0:%.+]] = IE.Subtract([[MULTIPLY1]], [[MULTIPLY3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<1x1536xf16>
    // CHECK:       return [[SUBTRACT0]] : tensor<1x1536xf16>
}

// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>
!quantileType = !quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-6.961928e-01,-5.250731e-01,-3.949175e-01,-2.844414e-01,-1.847734e-01,-9.105004e-02,0.000000e+00,7.958030e-02,1.609302e-01,2.461123e-01,3.379152e-01,4.407098e-01,5.626170e-01,7.229568e-01,1.000000e+00}>:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForNF4SmallLUTWeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960x!quantileType>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForNF4SmallLUTWeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960x!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<1536x8960x!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<1536x8960x!qElemType>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>

    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960x!quantileType> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[RESHAPE0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}

// -----

!quantileType = !quant.quantile<ui4:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}>
!qElemType1 = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}>:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+01,-6.961928e-01,-5.250731e-01,-3.949175e-01,-2.844414e-01,-1.847734e-01,-9.105004e-02,0.000000e+00,7.958030e-02,1.609302e-01,2.461123e-01,3.379152e-01,4.407098e-01,5.626170e-01,7.229568e-01,1.000000e+01}>:f16, 0.69999998807907104>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForNF4LargeLUTWeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960x!quantileType>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForNF4LargeLUTWeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960x!quantileType>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<1536x8960x!quantileType> -> tensor<1536x8960x!qElemType1>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType1>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>


    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960x!quantileType> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<1.428710e+00> : tensor<1xf16>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}

// -----

!quantileType = !quant.quantile<ui4:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}>
!qElemType1 = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}>:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+01,-6.961928e-01,-5.250731e-01,-3.949175e-01,-2.844414e-01,-1.847734e-01,-9.105004e-02,0.000000e+00,7.958030e-02,1.609302e-01,2.461123e-01,3.379152e-01,4.407098e-01,5.626170e-01,7.229568e-01,1.000000e+01}>:f16, 0.69999998807907104>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForNF4LargeLUTWeightsTwoConsumers
// CHECK-SAME:      [[INPUT_0:%[^:]+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%[^:]+]]: tensor<1536x8960x!quantileType>,
// CHECK-SAME:      [[INPUT_2:%[^:]+]]: tensor<1536x1xf16>,
// CHECK-SAME:      [[INPUT_3:%[^:]+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_4:%[^:]+]]: tensor<1536x1xf16>
func.func @RescaleForNF4LargeLUTWeightsTwoConsumers(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960x!quantileType>, %arg2: tensor<1536x1xf16>, %arg3: tensor<1024x8960xf16>, %arg4: tensor<1536x1xf16>) -> (tensor<1024x1536xf16>, tensor<1024x1536xf16>) {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<1536x8960x!quantileType> -> tensor<1536x8960x!qElemType1>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType1>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    %3 = IE.DynamicDequantize(%0, %arg4) {dstElemType = f16} : tensor<1536x8960x!qElemType1>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %4 = IE.FullyConnected(%arg3, %3) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2, %4 : tensor<1024x1536xf16>, tensor<1024x1536xf16>

    // Both DynamicDequantize consumers of the shared IE.QuantizeCast get their own rescaled cast,
    // and the original (non-rescaled) QuantizeCast is fully replaced rather than left as a stacked
    // cast, once both of its uses are rewired away from it.
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960x!quantileType> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<1.428710e+00> : tensor<1xf16>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>

    // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[INPUT_4]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST1:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960x!quantileType> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[CONST1:%.+]] = const.Declare tensor<1xf16> = dense<1.428710e+00> : tensor<1xf16>
    // CHECK:       [[MULTIPLY2:%.+]] = IE.Multiply([[RESHAPE1]], [[CONST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE1:%.+]] = IE.Dequantize([[QUANTIZECAST1]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED1:%.+]] = IE.FullyConnected([[INPUT_3]], [[DEQUANTIZE1]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY3:%.+]] = IE.Multiply([[FULLYCONNECTED1]], [[MULTIPLY2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>

    // CHECK:       return [[MULTIPLY1]], [[MULTIPLY3]] : tensor<1024x1536xf16>, tensor<1024x1536xf16>
}

// -----

!quantileType = !quant.quantile<ui4:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}>
!qElemType1 = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}>:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_MID:!.+]] = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+01,-6.961928e-01,-5.250731e-01,-3.949175e-01,-2.844414e-01,-1.847734e-01,-9.105004e-02,0.000000e+00,7.958030e-02,1.609302e-01,2.461123e-01,3.379152e-01,4.407098e-01,5.626170e-01,7.229568e-01,1.000000e+01}>:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+01,-6.961928e-01,-5.250731e-01,-3.949175e-01,-2.844414e-01,-1.847734e-01,-9.105004e-02,0.000000e+00,7.958030e-02,1.609302e-01,2.461123e-01,3.379152e-01,4.407098e-01,5.626170e-01,7.229568e-01,1.000000e+01}>:f16, 0.69999998807907104>

// CHECK:       @RescaleForNF4LargeLUTWeightsSliceAsDirectProducer
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x3072xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<16384x3072x!quantileType>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<16384x1xf16>
func.func @RescaleForNF4LargeLUTWeightsSliceAsDirectProducer(%arg0: tensor<1024x3072xf16>, %arg1: tensor<16384x3072x!quantileType>, %arg2: tensor<16384x1xf16>) -> (tensor<1024x8192xf16>, tensor<1024x8192xf16>) {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<16384x3072x!quantileType> -> tensor<16384x3072x!qElemType1>

    %1 = IE.Slice %0 [0, 0] [8192, 3072] : tensor<16384x3072x!qElemType1> to tensor<8192x3072x!qElemType1>
    %2 = IE.Slice %arg2 [0, 0] [8192, 1] : tensor<16384x1xf16> to tensor<8192x1xf16>
    %3 = IE.DynamicDequantize(%1, %2) {dstElemType = f16} : tensor<8192x3072x!qElemType1>, tensor<8192x1xf16> -> tensor<8192x3072xf16>

    %4 = IE.Slice %0 [8192, 0] [8192, 3072] : tensor<16384x3072x!qElemType1> to tensor<8192x3072x!qElemType1>
    %5 = IE.Slice %arg2 [8192, 0] [8192, 1] : tensor<16384x1xf16> to tensor<8192x1xf16>
    %6 = IE.DynamicDequantize(%4, %5) {dstElemType = f16} : tensor<8192x3072x!qElemType1>, tensor<8192x1xf16> -> tensor<8192x3072xf16>

    %7 = IE.FullyConnected(%arg0, %3) : tensor<1024x3072xf16>, tensor<8192x3072xf16> -> tensor<1024x8192xf16>
    %8 = IE.FullyConnected(%arg0, %6) : tensor<1024x3072xf16>, tensor<8192x3072xf16> -> tensor<1024x8192xf16>

    return %7, %8 : tensor<1024x8192xf16>, tensor<1024x8192xf16>

    // The QuantizeCast is shared by both branches through a Slice (weight producer chain is
    // QuantizeCast -> Slice -> DynamicDequantize). Each branch's Slice output gets its own
    // rescaled QuantizeCast inserted right after it; the original (unscaled) QuantizeCast is left
    // untouched since it still feeds both Slices.
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_MID]]} : tensor<16384x3072x!quantileType> -> tensor<16384x3072x[[QELEMTYPE_MID]]>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[QUANTIZECAST0]] [0, 0] [8192, 3072] : tensor<16384x3072x[[QELEMTYPE_MID]]> to tensor<8192x3072x[[QELEMTYPE_MID]]>
    // CHECK:       [[SLICE_SCALE0:%.+]] = IE.Slice [[INPUT_2]] [0, 0] [8192, 1] : tensor<16384x1xf16> to tensor<8192x1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[SLICE_SCALE0]]) {shape_value = [1, 8192]} : tensor<8192x1xf16> -> tensor<1x8192xf16>
    // CHECK:       [[QUANTIZECAST1:%.+]] = IE.QuantizeCast([[SLICE0]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<8192x3072x[[QELEMTYPE_MID]]> -> tensor<8192x3072x[[QELEMTYPE_OUT]]>
    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<1.428710e+00> : tensor<1xf16>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8192xf16>, tensor<1xf16> -> tensor<1x8192xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST1]]) {dstElemType = f16} : tensor<8192x3072x[[QELEMTYPE_OUT]]> -> tensor<8192x3072xf16>

    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[QUANTIZECAST0]] [8192, 0] [8192, 3072] : tensor<16384x3072x[[QELEMTYPE_MID]]> to tensor<8192x3072x[[QELEMTYPE_MID]]>
    // CHECK:       [[SLICE_SCALE1:%.+]] = IE.Slice [[INPUT_2]] [8192, 0] [8192, 1] : tensor<16384x1xf16> to tensor<8192x1xf16>
    // CHECK:       [[RESHAPE1:%.+]] = IE.Reshape([[SLICE_SCALE1]]) {shape_value = [1, 8192]} : tensor<8192x1xf16> -> tensor<1x8192xf16>
    // CHECK:       [[QUANTIZECAST2:%.+]] = IE.QuantizeCast([[SLICE1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<8192x3072x[[QELEMTYPE_MID]]> -> tensor<8192x3072x[[QELEMTYPE_OUT]]>
    // CHECK:       [[CONST1:%.+]] = const.Declare tensor<1xf16> = dense<1.428710e+00> : tensor<1xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[RESHAPE1]], [[CONST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8192xf16>, tensor<1xf16> -> tensor<1x8192xf16>
    // CHECK:       [[DEQUANTIZE1:%.+]] = IE.Dequantize([[QUANTIZECAST2]]) {dstElemType = f16} : tensor<8192x3072x[[QELEMTYPE_OUT]]> -> tensor<8192x3072xf16>

    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x3072xf16>, tensor<8192x3072xf16> -> tensor<1024x8192xf16>
    // CHECK:       [[MULTIPLY2:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x8192xf16>, tensor<1x8192xf16> -> tensor<1024x8192xf16>
    // CHECK:       [[FULLYCONNECTED1:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE1]]) : tensor<1024x3072xf16>, tensor<8192x3072xf16> -> tensor<1024x8192xf16>
    // CHECK:       [[MULTIPLY3:%.+]] = IE.Multiply([[FULLYCONNECTED1]], [[MULTIPLY1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x8192xf16>, tensor<1x8192xf16> -> tensor<1024x8192xf16>

    // CHECK:       return [[MULTIPLY2]], [[MULTIPLY3]] : tensor<1024x8192xf16>, tensor<1024x8192xf16>
}

// -----

!quantileType = !quant.quantile<ui4:f16, {-1.600000e+01,-1.500000e+01,-1.400000e+01,-1.300000e+01,-1.200000e+01,-1.100000e+01,-1.000000e+01,-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>
!qElemType1 = !quant.uniform<!quant.quantile<ui4:f16, {-1.600000e+01,-1.500000e+01,-1.400000e+01,-1.300000e+01,-1.200000e+01,-1.100000e+01,-1.000000e+01,-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<!quant.quantile<ui4:f16, {-1.600000e+01,-1.500000e+01,-1.400000e+01,-1.300000e+01,-1.200000e+01,-1.100000e+01,-1.000000e+01,-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16, 5.000000e-01>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForNF4AsymmetricNegativeLUTWeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960x!quantileType>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForNF4AsymmetricNegativeLUTWeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960x!quantileType>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<1536x8960x!quantileType> -> tensor<1536x8960x!qElemType1>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType1>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>

    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960x!quantileType> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<1xf16>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}

// -----

!quantileType = !quant.quantile<ui4:f16, {1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01,1.600000e+01}>
!qElemType1 = !quant.uniform<!quant.quantile<ui4:f16, {1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01,1.600000e+01}>:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<!quant.quantile<ui4:f16, {1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01,1.600000e+01}>:f16, 4.375000e-01>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForNF4AsymmetricPositiveLUTWeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960x!quantileType>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForNF4AsymmetricPositiveLUTWeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960x!quantileType>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<1536x8960x!quantileType> -> tensor<1536x8960x!qElemType1>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType1>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>

    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960x!quantileType> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<2.285160e+00> : tensor<1xf16>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}


// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertForAffineReshapeOnly
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x512x128x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x512x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @ConvertForAffineReshapeOnly(%arg0: tensor<1x512x128x!qElemType>, %arg1: tensor<1x512x1xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x512x128x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x512x1xf16> -> tensor<1x512x128xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1]], shape_value = [512, 128]} : tensor<1x512x128xf16> -> tensor<512x128xf16>
    %2 = IE.FullyConnected(%arg2, %1) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %2 : tensor<1x512xf16>

    // CHECK:  [[RESHAPE_SCALE:%.+]] = IE.Reshape([[INPUT_1]]) {shape_value = [1, 512]} : tensor<1x512x1xf16> -> tensor<1x512xf16>
    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<1x512x128x!qElemType> -> tensor<1x512x128xf16>
    // CHECK:  [[AFFINE_RESHAPE_IN:%.+]] = IE.AffineReshape([[DYN_DEQUANTIZE]])
    // CHECK-SAME{LITERAL}        {dim_mapping = [[0], [0], [1]], shape_value = [512, 128]} : tensor<1x512x128xf16> -> tensor<512x128xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[AFFINE_RESHAPE_IN]]) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[RESHAPE_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512xf16>, tensor<1x512xf16> -> tensor<1x512xf16>
    // CHECK:  return  [[MULTIPLY]] : tensor<1x512xf16>
}


// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertForTransposeAffineReshape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x128x512x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x512xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x128xf16>
func.func @ConvertForTransposeAffineReshape(%arg0: tensor<1x128x512x!qElemType>, %arg1: tensor<1x1x512xf16>, %arg2: tensor<1x128xf16>) -> tensor<1x512xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x128x512x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x1x512xf16> -> tensor<1x128x512xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<1x128x512xf16> -> tensor<1x512x128xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1]], shape_value = [512, 128]} : tensor<1x512x128xf16> -> tensor<512x128xf16>
    %3 = IE.FullyConnected(%arg2, %2) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>

    return %3 : tensor<1x512xf16>

    // CHECK:  [[RESHAPE_SCALE:%.+]] = IE.Reshape([[INPUT_1]]) {shape_value = [1, 512]} : tensor<1x1x512xf16> -> tensor<1x512xf16>
    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.Dequantize([[INPUT_0]]) {dstElemType = f16} : tensor<1x128x512x!qElemType> -> tensor<1x128x512xf16>
    // CHECK:  [[TRANSPOSE:%.+]]  = IE.Transpose([[DYN_DEQUANTIZE]]) {order_value = #map} : tensor<1x128x512xf16> -> tensor<1x512x128xf16>
    // CHECK:  [[AFFINE_RESHAPE_IN:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}      {dim_mapping = [[0], [0], [1]], shape_value = [512, 128]} : tensor<1x512x128xf16> -> tensor<512x128xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[AFFINE_RESHAPE_IN]]) : tensor<1x128xf16>, tensor<512x128xf16> -> tensor<1x512xf16>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[FC]], [[RESHAPE_SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512xf16>, tensor<1x512xf16> -> tensor<1x512xf16>
    // CHECK:  return  [[MULTIPLY]] : tensor<1x512xf16>
}
