//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --inline %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// Note: This test checks if UnifiedFuncInlinerInterface uses the fallback implementation.
module @FuncCallOpFallback {
    func.func private @foo(%arg: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32> {
        %0 = VPU.Add(%arg, %arg) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
        return %0: tensor<1x1x1x1xf32>
    }

    // CHECK-NOT: func.func private @foo

    func.func @main(%arg: tensor<1x1x1x1xf32>) -> (tensor<1x1x1x1xf32>) {
        %0 = func.call @foo(%arg) : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
        return %0: tensor<1x1x1x1xf32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<1x1x1x1xf32>)
    // CHECK: [[ADD0:%.+]] = VPU.Add([[ARG]], [[ARG]])
    // CHECK: return [[ADD0]]
}
