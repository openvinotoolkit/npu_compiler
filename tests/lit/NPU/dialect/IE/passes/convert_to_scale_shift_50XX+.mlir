//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-scale-shift %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @ConvertMultiplyToScaleShiftWithBroadcastAndWithinDPULimits
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x9216x1024x1xf16>, [[INPUT1:%.+]]: tensor<1x9216x1x1xf16>
func.func @ConvertMultiplyToScaleShiftWithBroadcastAndWithinDPULimits(%arg0: tensor<1x9216x1024x1xf16>, %arg1: tensor<1x9216x1x1xf16>) -> tensor<1x9216x1024x1xf16> {
    %0 = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x9216x1024x1xf16>, tensor<1x9216x1x1xf16> -> tensor<1x9216x1024x1xf16>

    return %0 : tensor<1x9216x1024x1xf16>

    // CHECK: [[SCALESHIFT:%.+]] = IE.ScaleShift([[INPUT0]], [[INPUT1]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x9216x1024x1xf16>, tensor<1x9216x1x1xf16> -> tensor<1x9216x1024x1xf16>

    // CHECK: return [[SCALESHIFT]]
}

// -----
// CHECK-LABEL: @NotConvertMultiplyToScaleShiftWithBroadcastAndBeyondDPULimits
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x24578x1x2048xf16>, [[INPUT1:%.+]]: tensor<1x24578x1x1xf16>
func.func @NotConvertMultiplyToScaleShiftWithBroadcastAndBeyondDPULimits(%arg0: tensor<1x24578x1x2048xf16>, %arg1: tensor<1x24578x1x1xf16>) -> tensor<1x24578x1x2048xf16> {
    %0 = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x24578x1x2048xf16>, tensor<1x24578x1x1xf16> -> tensor<1x24578x1x2048xf16>

    return %0 : tensor<1x24578x1x2048xf16>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24578x1x2048xf16>, tensor<1x24578x1x1xf16> -> tensor<1x24578x1x2048xf16>

    // CHECK: return [[MULTIPLY]]
}
