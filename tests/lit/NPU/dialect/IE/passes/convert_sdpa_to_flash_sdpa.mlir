//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-sdpa-to-flash-sdpa  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @SdpaNoMaskNoScale
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x32x128xf16>
func.func @SdpaNoMaskNoScale(%arg0: tensor<8x64x64xf16>, %arg1: tensor<8x32x64xf16>, %arg2: tensor<8x32x128xf16>) -> tensor<8x64x128xf16> {
    %0 = IE.SDPA(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16> -> tensor<8x64x128xf16>
    return %0 : tensor<8x64x128xf16>

    // CHECK:       [[TRANSPOSED_VALUE:%.+]] = IE.Transpose([[VALUE]]) {order_value = #map} : tensor<8x32x128xf16> -> tensor<8x128x32xf16>
    // CHECK:       [[FLASH_SDPA:%.+]] = IE.FlashSDPA([[QUERY]], [[KEY]], [[TRANSPOSED_VALUE]])
    // CHECK-SAME:      -> tensor<8x64x128xf16>

    // CHECK:       return [[FLASH_SDPA]]
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @SdpaNoMaskYesScale
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x32x128xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<1xf16>,
// CHECK-SAME: [[SCALE:%[^, ]+]]: tensor<1xf16>
func.func @SdpaNoMaskYesScale(%arg0: tensor<8x64x64xf16>, %arg1: tensor<8x32x64xf16>, %arg2: tensor<8x32x128xf16>, %arg3: tensor<1xf16>, %arg4: tensor<1xf16>) -> tensor<8x64x128xf16> {
    %0 = IE.SDPA(%arg0, %arg1, %arg2, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<1xf16> -> tensor<8x64x128xf16>
    return %0 : tensor<8x64x128xf16>

    // CHECK:       [[TRANSPOSED_VALUE:%.+]] = IE.Transpose([[VALUE]]) {order_value = #map} : tensor<8x32x128xf16> -> tensor<8x128x32xf16>
    // CHECK:       [[FLASH_SDPA:%.+]] = IE.FlashSDPA([[QUERY]], [[KEY]], [[TRANSPOSED_VALUE]], [[SCALE]])
    // CHECK-SAME:      -> tensor<8x64x128xf16>

    // CHECK:       return [[FLASH_SDPA]]
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @SdpaYesMaskNoScale
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x32x128xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<8x64x32xf16>
func.func @SdpaYesMaskNoScale(%arg0: tensor<8x64x64xf16>, %arg1: tensor<8x32x64xf16>, %arg2: tensor<8x32x128xf16>, %arg3: tensor<8x64x32xf16>) -> tensor<8x64x128xf16> {
    %0 = IE.SDPA(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<8x64x32xf16> -> tensor<8x64x128xf16>
    return %0 : tensor<8x64x128xf16>

    // CHECK:       [[TRANSPOSED_VALUE:%.+]] = IE.Transpose([[VALUE]]) {order_value = #map} : tensor<8x32x128xf16> -> tensor<8x128x32xf16>
    // CHECK:       [[FLASH_SDPA:%.+]] = IE.FlashSDPA([[QUERY]], [[KEY]], [[TRANSPOSED_VALUE]], [[ATTENTION_MASK]])
    // CHECK-SAME:      -> tensor<8x64x128xf16>

    // CHECK:       return [[FLASH_SDPA]]
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @SdpaYesMaskYesScale
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x32x128xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<8x64x32xf16>,
// CHECK-SAME: [[SCALE:%[^, ]+]]: tensor<1xf16>
func.func @SdpaYesMaskYesScale(%arg0: tensor<8x64x64xf16>, %arg1: tensor<8x32x64xf16>, %arg2: tensor<8x32x128xf16>, %arg3: tensor<8x64x32xf16>, %arg4: tensor<1xf16>) -> tensor<8x64x128xf16> {
    %0 = IE.SDPA(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<8x64x32xf16>, tensor<1xf16> -> tensor<8x64x128xf16>
    return %0 : tensor<8x64x128xf16>

    // CHECK:       [[TRANSPOSED_VALUE:%.+]] = IE.Transpose([[VALUE]]) {order_value = #map} : tensor<8x32x128xf16> -> tensor<8x128x32xf16>
    // CHECK:       [[FLASH_SDPA:%.+]] = IE.FlashSDPA([[QUERY]], [[KEY]], [[TRANSPOSED_VALUE]], [[ATTENTION_MASK]], [[SCALE]])
    // CHECK-SAME:      -> tensor<8x64x128xf16>

    // CHECK:       return [[FLASH_SDPA]]
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @SdpaNoMaskNoScaleCausal
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x32x128xf16>
func.func @SdpaNoMaskNoScaleCausal(%arg0: tensor<8x64x64xf16>, %arg1: tensor<8x32x64xf16>, %arg2: tensor<8x32x128xf16>) -> tensor<8x64x128xf16> {
    // The consant will have an actual causal mask. Replaced with dense 0-s to make it concise.
    %cst = const.Declare tensor<64x32xf16> = dense<0.000000e+00> : tensor<64x32xf16>
    %0 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<64x32xf16> -> tensor<8x64x128xf16>
    return %0 : tensor<8x64x128xf16>

    // CHECK:       [[CAUSAL_MASK:%.+]] = const.Declare tensor<64x32xf16>
    // CHECK:       [[TRANSPOSED_VALUE:%.+]] = IE.Transpose([[VALUE]]) {order_value = #map} : tensor<8x32x128xf16> -> tensor<8x128x32xf16>
    // CHECK:       [[FLASH_SDPA:%.+]] = IE.FlashSDPA([[QUERY]], [[KEY]], [[TRANSPOSED_VALUE]], [[CAUSAL_MASK]])
    // CHECK-SAME:      -> tensor<8x64x128xf16>

    // CHECK:       return [[FLASH_SDPA]]
}
