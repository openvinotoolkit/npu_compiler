//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Invalid: slope C=3 but input C=4 (neither 1 nor equal to input dim).
func.func @PReluInvalidSlopeDimMismatch(%arg0: tensor<1x4x2x3xf16>, %arg1: tensor<1x3x1x1xf16>) -> tensor<1x4x2x3xf16> {
    // expected-error @below {{Unsupported slope shape for PRelu: slope must numpy-broadcast to input}}
    %0 = VPU.PRelu(%arg0, %arg1) : tensor<1x4x2x3xf16>, tensor<1x3x1x1xf16> -> tensor<1x4x2x3xf16>
    return %0 : tensor<1x4x2x3xf16>
}
