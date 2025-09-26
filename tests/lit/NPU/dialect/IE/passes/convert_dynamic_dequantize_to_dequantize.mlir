//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-dynamic-dequantize-to-dequantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


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
func.func @NotConvertHasZPInput(%arg0: tensor<4096x4096x!qElemType>, %arg1: tensor<4096x1xf16>, %arg2: tensor<4096x1xi4>, %arg3: tensor<1x4096xf16>) -> tensor<1x4096xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1, %arg2) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<4096x1xf16>, tensor<4096x1xi4> -> tensor<4096x4096xf16>
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

    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<1.814060e+01> : tensor<1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960xsi8> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}

// -----

!qElemType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForNF4SmallLUTWeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForNF4SmallLUTWeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<1536x8960x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<1536x8960x!qElemType>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>

    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[RESHAPE0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}

// -----

!qElemType0 = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}>
!qElemType1 = !quant.quantile<u4:f16:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}:1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.quantile<u4:f16:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}:0.69999998807907104>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForNF4LargeLUTWeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}>>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForNF4LargeLUTWeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960x!qElemType0>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<1536x8960x!qElemType0> -> tensor<1536x8960x!qElemType1>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType1>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>

    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<1.428710e+00> : tensor<1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960x!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+01,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+01}>> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}

// -----

!qElemType0 = !QuantileFloat.quantileFloat<ui4:f16, {-1.600000e+01,-1.500000e+01,-1.400000e+01,-1.300000e+01,-1.200000e+01,-1.100000e+01,-1.000000e+01,-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>
!qElemType1 = !quant.quantile<u4:f16:f16, {-1.600000e+01,-1.500000e+01,-1.400000e+01,-1.300000e+01,-1.200000e+01,-1.100000e+01,-1.000000e+01,-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}:1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.quantile<u4:f16:f16, {-1.600000e+01,-1.500000e+01,-1.400000e+01,-1.300000e+01,-1.200000e+01,-1.100000e+01,-1.000000e+01,-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}:5.000000e-01>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForNF4AsymmetricNegativeLUTWeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960x!QuantileFloat.quantileFloat<ui4:f16, {-1.600000e+01,-1.500000e+01,-1.400000e+01,-1.300000e+01,-1.200000e+01,-1.100000e+01,-1.000000e+01,-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForNF4AsymmetricNegativeLUTWeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960x!qElemType0>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<1536x8960x!qElemType0> -> tensor<1536x8960x!qElemType1>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType1>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>

    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960x!QuantileFloat.quantileFloat<ui4:f16, {-1.600000e+01,-1.500000e+01,-1.400000e+01,-1.300000e+01,-1.200000e+01,-1.100000e+01,-1.000000e+01,-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}

// -----

!qElemType0 = !QuantileFloat.quantileFloat<ui4:f16, {1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01,1.600000e+01}>
!qElemType1 = !quant.quantile<u4:f16:f16, {1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01,1.600000e+01}:1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.quantile<u4:f16:f16, {1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01,1.600000e+01}:4.375000e-01>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForNF4AsymmetricPositiveLUTWeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960x!QuantileFloat.quantileFloat<ui4:f16, {1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01,1.600000e+01}>>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForNF4AsymmetricPositiveLUTWeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960x!qElemType0>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<1536x8960x!qElemType0> -> tensor<1536x8960x!qElemType1>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType1>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>

    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<2.285160e+00> : tensor<1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960x!QuantileFloat.quantileFloat<ui4:f16, {1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01,1.600000e+01}>> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}
