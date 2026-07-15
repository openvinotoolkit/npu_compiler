//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --inline %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Note: This test checks if UnifiedFuncInlinerInterface uses the fallback implementation.
module @FuncCallOpFallback {
    func.func nested @foo(%arg: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32> {
        %0 = VPU.Add(%arg, %arg) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
        return %0: tensor<1x1x1x1xf32>
    }

    // CHECK-NOT: func.func nested @foo

    func.func @main(%arg: tensor<1x1x1x1xf32>) -> (tensor<1x1x1x1xf32>) {
        %0 = func.call @foo(%arg) : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
        return %0: tensor<1x1x1x1xf32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<1x1x1x1xf32>)
    // CHECK: [[ADD0:%.+]] = VPU.Add([[ARG]], [[ARG]])
    // CHECK: return [[ADD0]]
}
