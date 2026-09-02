//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @verification_disabled
module @verification_disabled {
func.func @main(%arg0: tensor<8x1024xf32>) -> tensor<8x1024xf32> {
    // No expected error: shape verification is not enabled on this module.
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<8x1024xf32>, tensor<8x1024xf32> -> tensor<8x1024xf32>
    return %0 : tensor<8x1024xf32>
}
}

// -----

// CHECK-LABEL: @verification_enabled_multiply
module @verification_enabled_multiply attributes {IE.shape_verification_enabled} {
// expected-error@+1 {{Expected 4D input, got rank 2}}
func.func @main(%arg0: tensor<8x1024xf32>) -> tensor<8x1024xf32> {
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<8x1024xf32>, tensor<8x1024xf32> -> tensor<8x1024xf32>
    return %0 : tensor<8x1024xf32>
}
}

// -----

// CHECK-LABEL: @verification_enabled_convolution
module @verification_enabled_convolution attributes {IE.shape_verification_enabled} {
// expected-error@+1 {{Expected 4D input, got rank 3}}
func.func @main(%arg0: tensor<1x224x224xf32>) -> tensor<16x224x224xf32> {
    %filter = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<16x1x1x1xf32>
    %0 = IE.Convolution(%arg0, %filter)
        {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x224x224xf32>, tensor<16x1x1x1xf32> -> tensor<16x224x224xf32>
    return %0 : tensor<16x224x224xf32>
}
}
