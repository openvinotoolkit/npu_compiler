//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --convert-shape-to-4d="enable-shape-verification=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: module @ConvertShapeTo4DShapeVerification attributes {
// CHECK: IE.shape_verification_enabled
module @ConvertShapeTo4DShapeVerification attributes {test.attr = "keep-existing-attr"} {
    func.func @main(%arg0: tensor<8x1024xf32>) -> tensor<8x1024xf32> {
        %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
            : tensor<8x1024xf32>, tensor<8x1024xf32> -> tensor<8x1024xf32>
        return %0 : tensor<8x1024xf32>
    }
}
