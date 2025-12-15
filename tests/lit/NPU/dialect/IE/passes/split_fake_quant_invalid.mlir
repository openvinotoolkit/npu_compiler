//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-fake-quant --verify-diagnostics %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @BroadcastDiffDims
func.func @BroadcastDiffDims(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-1.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>
    %input_high = const.Declare tensor<1x3x1x1xf32> = dense<[[[[2.0]],[[2.0]],[[2.0]]]]> : tensor<1x3x1x1xf32>
    %output_low = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-1.0]],[[-1.0]],[[-1.0]]]]> : tensor<1x3x1x1xf32>
    %output_high = const.Declare tensor<1x3x1x1xf32> = dense<[[[[2.0]],[[2.0]],[[2.0]]]]> : tensor<1x3x1x1xf32>

    // expected-error@+2 {{Got non broadcastable dimensions pair : '3' and 2'}}
    // expected-error@+1 {{Eltwise inputs cannot be broadcast}}
    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x2x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>
}
