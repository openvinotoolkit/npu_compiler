//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --platform=%platform% --import-IE ./IR/dead_scatter_nd_update_zero_sized_broadcast.xml -o %t
// RUN: FileCheck %s --input-file %t
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// If ov::pass::EliminateScatterUpdate does not run before ConstantFolding, the dead Broadcast survives
// and crashes the process. If it runs first, the whole dead branch (Broadcast + ScatterNDUpdate)
// is removed before ConstantFolding.

// CHECK-LABEL: @DeadScatterNDUpdateZeroSizedBroadcast
// CHECK: net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:   DataInfo "data" : tensor<8x4xf16>
// CHECK: } outputsInfo : {
// CHECK:   DataInfo "scatter" friendlyName = "out" : tensor<8x4xf32>
// CHECK: }

// CHECK: func.func @main([[ARG_0:%.+]]: tensor<8x4xf16>) -> tensor<8x4xf32>
// CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG_0]]) {dstElemType = f32} : tensor<8x4xf16> -> tensor<8x4xf32>
// CHECK: return [[CONVERT]] : tensor<8x4xf32>
