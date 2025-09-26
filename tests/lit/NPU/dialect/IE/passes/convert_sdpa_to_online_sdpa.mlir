//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-sdpa-to-online-sdpa  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @SdpaWithAttentionMaskNoScale
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<8x512x64xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<8x512x512xf16>
func.func @SdpaWithAttentionMaskNoScale(%arg0: tensor<8x512x64xf16>, %arg1: tensor<8x512x64xf16>, %arg2: tensor<8x512x64xf16>, %arg3: tensor<8x512x512xf16>) -> tensor<8x512x64xf16> {
    %0 = IE.SDPA(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>} : tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512x64xf16>, tensor<8x512x512xf16> -> tensor<8x512x64xf16>
    return %0 : tensor<8x512x64xf16>

    // CHECK:       [[ONLINE_SDPA:%.+]] = IE.OnlineSDPA([[QUERY]], [[KEY]], [[VALUE]], [[ATTENTION_MASK]])
    // CHECK-SAME:      -> tensor<8x512x64xf16>

    // CHECK:       return [[ONLINE_SDPA]]
}

