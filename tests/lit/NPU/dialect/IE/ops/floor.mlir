//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @FoldFloorWithSI32
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x128x4xsi32>
func.func @FoldFloorWithSI32(%arg0: tensor<1x16x128x4xsi32>) -> tensor<1x16x128x4xsi32> {
    %floor = IE.Floor(%arg0) : tensor<1x16x128x4xsi32> -> tensor<1x16x128x4xsi32>
    return %floor : tensor<1x16x128x4xsi32>

    // CHECK-NOT:   IE.Floor
}

// -----

// CHECK-LABEL: @FoldFloorWithSI64
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x48x4xsi64>
func.func @FoldFloorWithSI64(%arg0: tensor<1x16x48x4xsi64>) -> tensor<1x16x48x4xsi64> {
    %floor = IE.Floor(%arg0) : tensor<1x16x48x4xsi64> -> tensor<1x16x48x4xsi64>
    return %floor : tensor<1x16x48x4xsi64>

    // CHECK-NOT:   IE.Floor
}
