//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE ./shape_of.xml | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK: module @shape_of {
// CHECK:   func.func @main(%arg0: tensor<1x8x?x?xf16, {bounds = [1, 8, 384, 384], order = #NCHW}>)
// CHECK-SAME:      -> tensor<4xsi64> {
// CHECK:       [[SHAPE_OF:%.*]] = IE.ShapeOf(%arg0) {dstElemType = si64} :
// CHECK-SAME:      tensor<1x8x?x?xf16, {bounds = [1, 8, 384, 384], order = #NCHW}> -> tensor<4xsi64>
// CHECK:       return [[SHAPE_OF]] : tensor<4xsi64>
// CHECK:   }
// CHECK: }
