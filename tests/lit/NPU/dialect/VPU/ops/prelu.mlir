//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----

// Per-channel slope (C == input C, others 1): accepted.
// CHECK-LABEL: @PReluPerChannelSlopeAccepted
// CHECK-SAME:    ([[INPUT:%[^:]+]]: tensor<1x4x2x3xf16>, [[SLOPE:%[^:]+]]: tensor<1x4x1x1xf16>)
func.func @PReluPerChannelSlopeAccepted(%arg0: tensor<1x4x2x3xf16>, %arg1: tensor<1x4x1x1xf16>) -> tensor<1x4x2x3xf16> {
    // CHECK: [[PRELU:%.+]] = VPU.PRelu([[INPUT]], [[SLOPE]])
    // CHECK-SAME:   : tensor<1x4x2x3xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x2x3xf16>
    // CHECK: return [[PRELU]]
    %0 = VPU.PRelu(%arg0, %arg1) : tensor<1x4x2x3xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x2x3xf16>
    return %0 : tensor<1x4x2x3xf16>
}

// -----

// Per-W slope (W == input W, others 1): accepted (numpy broadcast).
// CHECK-LABEL: @PReluPerWSlopeAccepted
// CHECK-SAME:    ([[INPUT:%[^:]+]]: tensor<1x4x2x3xf16>, [[SLOPE:%[^:]+]]: tensor<1x1x1x3xf16>)
func.func @PReluPerWSlopeAccepted(%arg0: tensor<1x4x2x3xf16>, %arg1: tensor<1x1x1x3xf16>) -> tensor<1x4x2x3xf16> {
    // CHECK: [[PRELU:%.+]] = VPU.PRelu([[INPUT]], [[SLOPE]])
    // CHECK-SAME:   : tensor<1x4x2x3xf16>, tensor<1x1x1x3xf16> -> tensor<1x4x2x3xf16>
    // CHECK: return [[PRELU]]
    %0 = VPU.PRelu(%arg0, %arg1) : tensor<1x4x2x3xf16>, tensor<1x1x1x3xf16> -> tensor<1x4x2x3xf16>
    return %0 : tensor<1x4x2x3xf16>
}

// -----

// Per-H slope (H == input H, others 1): accepted (numpy broadcast).
// CHECK-LABEL: @PReluPerHSlopeAccepted
// CHECK-SAME:    ([[INPUT:%[^:]+]]: tensor<1x4x2x3xf16>, [[SLOPE:%[^:]+]]: tensor<1x1x2x1xf16>)
func.func @PReluPerHSlopeAccepted(%arg0: tensor<1x4x2x3xf16>, %arg1: tensor<1x1x2x1xf16>) -> tensor<1x4x2x3xf16> {
    // CHECK: [[PRELU:%.+]] = VPU.PRelu([[INPUT]], [[SLOPE]])
    // CHECK-SAME:   : tensor<1x4x2x3xf16>, tensor<1x1x2x1xf16> -> tensor<1x4x2x3xf16>
    // CHECK: return [[PRELU]]
    %0 = VPU.PRelu(%arg0, %arg1) : tensor<1x4x2x3xf16>, tensor<1x1x2x1xf16> -> tensor<1x4x2x3xf16>
    return %0 : tensor<1x4x2x3xf16>
}

// -----

// Scalar slope (all dims 1): accepted.
// CHECK-LABEL: @PReluScalarSlopeAccepted
// CHECK-SAME:    ([[INPUT:%[^:]+]]: tensor<1x4x2x3xf16>, [[SLOPE:%[^:]+]]: tensor<1x1x1x1xf16>)
func.func @PReluScalarSlopeAccepted(%arg0: tensor<1x4x2x3xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x4x2x3xf16> {
    // CHECK: [[PRELU:%.+]] = VPU.PRelu([[INPUT]], [[SLOPE]])
    // CHECK: return [[PRELU]]
    %0 = VPU.PRelu(%arg0, %arg1) : tensor<1x4x2x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x2x3xf16>
    return %0 : tensor<1x4x2x3xf16>
}

// -----

// Identical shape slope: accepted.
// CHECK-LABEL: @PReluIdenticalSlopeAccepted
// CHECK-SAME:    ([[INPUT:%[^:]+]]: tensor<1x4x2x3xf16>, [[SLOPE:%[^:]+]]: tensor<1x4x2x3xf16>)
func.func @PReluIdenticalSlopeAccepted(%arg0: tensor<1x4x2x3xf16>, %arg1: tensor<1x4x2x3xf16>) -> tensor<1x4x2x3xf16> {
    // CHECK: [[PRELU:%.+]] = VPU.PRelu([[INPUT]], [[SLOPE]])
    // CHECK: return [[PRELU]]
    %0 = VPU.PRelu(%arg0, %arg1) : tensor<1x4x2x3xf16>, tensor<1x4x2x3xf16> -> tensor<1x4x2x3xf16>
    return %0 : tensor<1x4x2x3xf16>
}
