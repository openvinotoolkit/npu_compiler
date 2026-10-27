//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-sdpa-to-flash-sdpa  %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @SdpaNoMaskNoScale
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x32x128xf16>
func.func @SdpaNoMaskNoScale(%arg0: tensor<8x64x64xf16>, %arg1: tensor<8x32x64xf16>, %arg2: tensor<8x32x128xf16>) -> tensor<8x64x128xf16> {
    %0 = IE.SDPA(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16> -> tensor<8x64x128xf16>
    return %0 : tensor<8x64x128xf16>

    // CHECK-DAG:   [[SCALE_CONST:%.+]] = const.Declare tensor<1xf16> = dense<1.250000e-01> : tensor<1xf16>
    // CHECK-DAG:   [[SCALED_QUERY:%.+]] = IE.Multiply([[QUERY]], [[SCALE_CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x64x64xf16>, tensor<1xf16> -> tensor<8x64x64xf16>

    // CHECK-DAG:   [[RUNNING_OUTPUT:%.+]] = const.Declare tensor<8x64x128xf16> = dense<0.000000e+00> : tensor<8x64x128xf16>
    // The running max seed is fp16 lowest, not -Inf (see convert_sdpa_to_flash_sdpa.cpp).
    // CHECK-DAG:   [[RUNNING_MAX:%.+]] = const.Declare tensor<8x64xf16> = dense<-6.550400e+04> : tensor<8x64xf16>
    // CHECK-DAG:   [[RUNNING_SUM:%.+]] = const.Declare tensor<8x64xf32> = dense<0.000000e+00> : tensor<8x64xf32>

    // CHECK:       [[OUTPUT:%.+]], [[MAX:%.+]], [[SUM:%.+]] = IE.FlashSDPA([[SCALED_QUERY]], [[KEY]], [[VALUE]], [[RUNNING_OUTPUT]], [[RUNNING_MAX]], [[RUNNING_SUM]])
    // CHECK-SAME:      -> tensor<8x64x128xf16>, tensor<8x64xf16>, tensor<8x64xf32>

    // CHECK:       return [[OUTPUT]]
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
    %0 = IE.SDPA(%arg0, %arg1, %arg2, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<1xf16> -> tensor<8x64x128xf16>
    return %0 : tensor<8x64x128xf16>

    // CHECK-DAG:   [[SCALED_QUERY:%.+]] = IE.Multiply([[QUERY]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x64x64xf16>, tensor<1xf16> -> tensor<8x64x64xf16>

    // CHECK-DAG:   [[RUNNING_OUTPUT:%.+]] = const.Declare tensor<8x64x128xf16> = dense<0.000000e+00> : tensor<8x64x128xf16>
    // CHECK-DAG:   [[RUNNING_MAX:%.+]] = const.Declare tensor<8x64xf16> = dense<-6.550400e+04> : tensor<8x64xf16>
    // CHECK-DAG:   [[RUNNING_SUM:%.+]] = const.Declare tensor<8x64xf32> = dense<0.000000e+00> : tensor<8x64xf32>

    // CHECK:       [[OUTPUT:%.+]], [[MAX:%.+]], [[SUM:%.+]] = IE.FlashSDPA([[SCALED_QUERY]], [[KEY]], [[VALUE]], [[RUNNING_OUTPUT]], [[RUNNING_MAX]], [[RUNNING_SUM]])
    // CHECK-SAME:      -> tensor<8x64x128xf16>, tensor<8x64xf16>, tensor<8x64xf32>

    // CHECK:       return [[OUTPUT]]
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @SdpaYesMaskNoScale
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x32x128xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<8x64x32xf16>
func.func @SdpaYesMaskNoScale(%arg0: tensor<8x64x64xf16>, %arg1: tensor<8x32x64xf16>, %arg2: tensor<8x32x128xf16>, %arg3: tensor<8x64x32xf16>) -> tensor<8x64x128xf16> {
    %0 = IE.SDPA(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<8x64x32xf16> -> tensor<8x64x128xf16>
    return %0 : tensor<8x64x128xf16>

    // CHECK-DAG:   [[SCALE_CONST:%.+]] = const.Declare tensor<1xf16> = dense<1.250000e-01> : tensor<1xf16>
    // CHECK-DAG:   [[SCALED_QUERY:%.+]] = IE.Multiply([[QUERY]], [[SCALE_CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x64x64xf16>, tensor<1xf16> -> tensor<8x64x64xf16>

    // CHECK-DAG:   [[RUNNING_OUTPUT:%.+]] = const.Declare tensor<8x64x128xf16> = dense<0.000000e+00> : tensor<8x64x128xf16>
    // CHECK-DAG:   [[RUNNING_MAX:%.+]] = const.Declare tensor<8x64xf16> = dense<-6.550400e+04> : tensor<8x64xf16>
    // CHECK-DAG:   [[RUNNING_SUM:%.+]] = const.Declare tensor<8x64xf32> = dense<0.000000e+00> : tensor<8x64xf32>

    // CHECK:       [[OUTPUT:%.+]], [[MAX:%.+]], [[SUM:%.+]] = IE.FlashSDPA([[SCALED_QUERY]], [[KEY]], [[VALUE]], [[RUNNING_OUTPUT]], [[RUNNING_MAX]], [[RUNNING_SUM]], [[ATTENTION_MASK]])
    // CHECK-SAME:      -> tensor<8x64x128xf16>, tensor<8x64xf16>, tensor<8x64xf32>

    // CHECK:       return [[OUTPUT]]
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
    %0 = IE.SDPA(%arg0, %arg1, %arg2, %arg3, %arg4) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<8x64x32xf16>, tensor<1xf16> -> tensor<8x64x128xf16>
    return %0 : tensor<8x64x128xf16>

    // CHECK-DAG:   [[SCALED_QUERY:%.+]] = IE.Multiply([[QUERY]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x64x64xf16>, tensor<1xf16> -> tensor<8x64x64xf16>

    // CHECK-DAG:   [[RUNNING_OUTPUT:%.+]] = const.Declare tensor<8x64x128xf16> = dense<0.000000e+00> : tensor<8x64x128xf16>
    // CHECK-DAG:   [[RUNNING_MAX:%.+]] = const.Declare tensor<8x64xf16> = dense<-6.550400e+04> : tensor<8x64xf16>
    // CHECK-DAG:   [[RUNNING_SUM:%.+]] = const.Declare tensor<8x64xf32> = dense<0.000000e+00> : tensor<8x64xf32>

    // CHECK:       [[OUTPUT:%.+]], [[MAX:%.+]], [[SUM:%.+]] = IE.FlashSDPA([[SCALED_QUERY]], [[KEY]], [[VALUE]], [[RUNNING_OUTPUT]], [[RUNNING_MAX]], [[RUNNING_SUM]], [[ATTENTION_MASK]])
    // CHECK-SAME:      -> tensor<8x64x128xf16>, tensor<8x64xf16>, tensor<8x64xf32>

    // CHECK:       return [[OUTPUT]]
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
    %0 = IE.SDPA(%arg0, %arg1, %arg2, %cst) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>} : tensor<8x64x64xf16>, tensor<8x32x64xf16>, tensor<8x32x128xf16>, tensor<64x32xf16> -> tensor<8x64x128xf16>
    return %0 : tensor<8x64x128xf16>

    // CHECK-DAG:   [[CAUSAL_MASK:%.+]] = const.Declare tensor<64x32xf16>

    // CHECK-DAG:   [[SCALE_CONST:%.+]] = const.Declare tensor<1xf16> = dense<1.250000e-01> : tensor<1xf16>
    // CHECK-DAG:   [[SCALED_QUERY:%.+]] = IE.Multiply([[QUERY]], [[SCALE_CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x64x64xf16>, tensor<1xf16> -> tensor<8x64x64xf16>

    // CHECK-DAG:   [[RUNNING_OUTPUT:%.+]] = const.Declare tensor<8x64x128xf16> = dense<0.000000e+00> : tensor<8x64x128xf16>
    // CHECK-DAG:   [[RUNNING_MAX:%.+]] = const.Declare tensor<8x64xf16> = dense<-6.550400e+04> : tensor<8x64xf16>
    // CHECK-DAG:   [[RUNNING_SUM:%.+]] = const.Declare tensor<8x64xf32> = dense<0.000000e+00> : tensor<8x64xf32>

    // CHECK:       [[OUTPUT:%.+]], [[MAX:%.+]], [[SUM:%.+]] = IE.FlashSDPA([[SCALED_QUERY]], [[KEY]], [[VALUE]], [[RUNNING_OUTPUT]], [[RUNNING_MAX]], [[RUNNING_SUM]], [[CAUSAL_MASK]])
    // CHECK-SAME:      -> tensor<8x64x128xf16>, tensor<8x64xf16>, tensor<8x64xf32>

    // CHECK:       return [[OUTPUT]]
}
