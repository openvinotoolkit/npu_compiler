//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-to-scale-shift %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @ConvertMultiplyToScaleShiftWithBroadcastAndWithinDPUBenefitFactor
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x17920x1024x1xf16>, [[INPUT1:%.+]]: tensor<1x17920x1x1xf16>
func.func @ConvertMultiplyToScaleShiftWithBroadcastAndWithinDPUBenefitFactor(%arg0: tensor<1x17920x1024x1xf16>, %arg1: tensor<1x17920x1x1xf16>) -> tensor<1x17920x1024x1xf16> {
    %0 = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x17920x1024x1xf16>, tensor<1x17920x1x1xf16> -> tensor<1x17920x1024x1xf16>

    return %0 : tensor<1x17920x1024x1xf16>

    // CHECK: [[SCALESHIFT:%.+]] = IE.ScaleShift([[INPUT0]], [[INPUT1]]) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<1x17920x1024x1xf16>, tensor<1x17920x1x1xf16> -> tensor<1x17920x1024x1xf16>

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
