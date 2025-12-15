//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-nce-ops-with-i32-inputs="convert-fc-to-conv=false" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FC_NoConvert
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<2x3xsi32>, [[INPUT1:%.+]]: tensor<2x3xsi32>
func.func @FC_NoConvert(%arg0: tensor<2x3xsi32>, %arg1: tensor<2x3xsi32>) -> tensor<2x2xsi32> {
    %0 = IE.FullyConnected(%arg0, %arg1) : tensor<2x3xsi32>, tensor<2x3xsi32> -> tensor<2x2xsi32>
    return %0 : tensor<2x2xsi32>

    // CHECK-NOT:  IE.Convert
    // CHECK:      IE.FullyConnected([[INPUT0]], [[INPUT1]])
    // CHECK-SAME:      : tensor<2x3xsi32>, tensor<2x3xsi32> -> tensor<2x2xsi32>
}
