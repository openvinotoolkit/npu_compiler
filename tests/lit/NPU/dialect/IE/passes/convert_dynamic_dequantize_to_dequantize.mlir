//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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


!qElemType = !quant.uniform<i4:f16, 1.000000e+00:8>

// CHECK-LABEL: @NotConvertZPIsNotZero
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4096x4096x!qElemType>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4096x1xf16>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4096xf16>
func.func @NotConvertZPIsNotZero(%arg0: tensor<4096x4096x!qElemType>, %arg1: tensor<4096x1xf16>, %arg2: tensor<1x4096xf16>) -> tensor<1x4096xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    %1 = IE.FullyConnected(%arg2, %0) : tensor<1x4096xf16>, tensor<4096x4096xf16> -> tensor<1x4096xf16>

    return %1 : tensor<1x4096xf16>

    // CHECK:  [[DYN_DEQUANTIZE:%.+]] = IE.DynamicDequantize
    // CHECK:  [[FC:%.+]] = IE.FullyConnected
    // CHECK:  return [[FC]] : tensor<1x4096xf16>
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
