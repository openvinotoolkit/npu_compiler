//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-nce-ops-with-i32-inputs="convert-fc-to-conv=false" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @FC_NoConvert
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<2x3xsi32>, [[INPUT1:%.+]]: tensor<2x3xsi32>
func.func @FC_NoConvert(%arg0: tensor<2x3xsi32>, %arg1: tensor<2x3xsi32>) -> tensor<2x2xsi32> {
    %0 = IE.FullyConnected(%arg0, %arg1) : tensor<2x3xsi32>, tensor<2x3xsi32> -> tensor<2x2xsi32>
    return %0 : tensor<2x2xsi32>

    // CHECK-NOT:  IE.Convert
    // CHECK:      IE.FullyConnected([[INPUT0]], [[INPUT1]])
    // CHECK-SAME:      : tensor<2x3xsi32>, tensor<2x3xsi32> -> tensor<2x2xsi32>
}
