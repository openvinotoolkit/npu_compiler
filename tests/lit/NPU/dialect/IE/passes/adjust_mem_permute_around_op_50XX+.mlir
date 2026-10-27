//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-mem-permute-around-op %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseAffineReshapePermuteCastAroundMultiply
// CHECK-SAME:  ([[INPUT1:%.+]]: tensor<1x128x16x64xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x128x16x64xf16, {order = #NHWC}>)
func.func @FuseAffineReshapePermuteCastAroundMultiply(%arg0: tensor<1x128x16x64xf16, {order = #NHWC}>, %arg1: tensor<1x128x16x64xf16, {order = #NHWC}>) -> tensor<1x128x16x64xf16, {order = #NHWC}> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x16x64x128xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1024, 128]} : tensor<1x16x64x128xf16> -> tensor<1024x128xf16>
    %2 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x16x64x128xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1024, 128]} : tensor<1x16x64x128xf16> -> tensor<1024x128xf16>
    %4 = IE.Multiply(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x128xf16>, tensor<1024x128xf16> -> tensor<1024x128xf16>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 16, 64, 128]} : tensor<1024x128xf16> -> tensor<1x16x64x128xf16>
    %6 = IE.PermuteCast(%5) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x64x128xf16> -> tensor<1x128x16x64xf16, {order = #NHWC}>
    return %6 : tensor<1x128x16x64xf16, {order = #NHWC}>

    // CHECK-NOT:  IE.PermuteCast
    // CHECK-NOT:  IE.AffineReshape
    // CHECK:      [[MULTIPLY:%.+]] = IE.Multiply([[INPUT1]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x16x64xf16, {order = #NHWC}>, tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x128x16x64xf16, {order = #NHWC}>
    // CHECK:      return [[MULTIPLY]] : tensor<1x128x16x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseAffineReshapePermuteCastRealBroadcast
// CHECK-SAME:  ([[INPUT1:%.+]]: tensor<1x4x1x1xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x4x2x1xf16, {order = #NHWC}>)
func.func @NotFuseAffineReshapePermuteCastRealBroadcast(%arg0: tensor<1x4x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x4x2x1xf16, {order = #NHWC}>) -> tensor<1x4x2x1xf16, {order = #NHWC}> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x4x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x4xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 4]} : tensor<1x1x1x4xf16> -> tensor<1x4xf16>
    %2 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x4x2x1xf16, {order = #NHWC}> -> tensor<1x2x1x4xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1]], shape_value = [2, 4]} : tensor<1x2x1x4xf16> -> tensor<2x4xf16>
    %4 = IE.Multiply(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4xf16>, tensor<2x4xf16> -> tensor<2x4xf16>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 1, 4]} : tensor<2x4xf16> -> tensor<1x2x1x4xf16>
    %6 = IE.PermuteCast(%5) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x1x4xf16> -> tensor<1x4x2x1xf16, {order = #NHWC}>
    return %6 : tensor<1x4x2x1xf16, {order = #NHWC}>

    // CHECK:      [[PC0:%.+]] = IE.PermuteCast([[INPUT1]])
    // CHECK:      [[AR0:%.+]] = IE.AffineReshape([[PC0]])
    // CHECK:      [[PC1:%.+]] = IE.PermuteCast([[INPUT2]])
    // CHECK:      [[AR1:%.+]] = IE.AffineReshape([[PC1]])
    // CHECK:      [[MULTIPLY:%.+]] = IE.Multiply([[AR0]], [[AR1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4xf16>, tensor<2x4xf16> -> tensor<2x4xf16>
    // CHECK:      [[AR2:%.+]] = IE.AffineReshape([[MULTIPLY]])
    // CHECK:      [[PC2:%.+]] = IE.PermuteCast([[AR2]])
    // CHECK:      return [[PC2]] : tensor<1x4x2x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @NotFuseAffineReshapePermuteCastMismatchedOrder
// CHECK-SAME:  ([[INPUT1:%.+]]: tensor<1x4x1x1xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x4x1x1xf16, {order = #NHWC}>)
func.func @NotFuseAffineReshapePermuteCastMismatchedOrder(%arg0: tensor<1x4x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x4x1x1xf16, {order = #NHWC}>) -> tensor<1x4x1x1xf16, {order = #NWHC}> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x4x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x4xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 4]} : tensor<1x1x1x4xf16> -> tensor<1x4xf16>
    %2 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x4x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x4xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 4]} : tensor<1x1x1x4xf16> -> tensor<1x4xf16>
    %4 = IE.Multiply(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4xf16>, tensor<1x4xf16> -> tensor<1x4xf16>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 4]} : tensor<1x4xf16> -> tensor<1x1x1x4xf16>
    %6 = IE.PermuteCast(%5) {dst_order = #NWHC, mem_perm = #NCHW} : tensor<1x1x1x4xf16> -> tensor<1x4x1x1xf16, {order = #NWHC}>
    return %6 : tensor<1x4x1x1xf16, {order = #NWHC}>

    // CHECK:      [[PC0:%.+]] = IE.PermuteCast([[INPUT1]])
    // CHECK:      [[AR0:%.+]] = IE.AffineReshape([[PC0]])
    // CHECK:      [[PC1:%.+]] = IE.PermuteCast([[INPUT2]])
    // CHECK:      [[AR1:%.+]] = IE.AffineReshape([[PC1]])
    // CHECK:      [[MULTIPLY:%.+]] = IE.Multiply([[AR0]], [[AR1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4xf16>, tensor<1x4xf16> -> tensor<1x4xf16>
    // CHECK:      [[AR2:%.+]] = IE.AffineReshape([[MULTIPLY]])
    // CHECK:      [[PC2:%.+]] = IE.PermuteCast([[AR2]])
    // CHECK:      return [[PC2]] : tensor<1x4x1x1xf16, {order = #NWHC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseAffineReshapePermuteCastNotNCEConvertible
// CHECK-SAME:  ([[INPUT1:%.+]]: tensor<1x4x1x1xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x4x1x1xf16, {order = #NHWC}>)
func.func @NotFuseAffineReshapePermuteCastNotNCEConvertible(%arg0: tensor<1x4x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x4x1x1xf16, {order = #NHWC}>) -> tensor<1x4x1x1xf16, {order = #NHWC}> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x4x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x4xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 4]} : tensor<1x1x1x4xf16> -> tensor<1x4xf16>
    %2 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x4x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x4xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 4]} : tensor<1x1x1x4xf16> -> tensor<1x4xf16>
    %4 = IE.Multiply(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4xf16>, tensor<1x4xf16> -> tensor<1x4xf16>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 4]} : tensor<1x4xf16> -> tensor<1x1x1x4xf16>
    %6 = IE.PermuteCast(%5) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x1x4xf16> -> tensor<1x4x1x1xf16, {order = #NHWC}>
    return %6 : tensor<1x4x1x1xf16, {order = #NHWC}>

    // CHECK:      [[PC0:%.+]] = IE.PermuteCast([[INPUT1]])
    // CHECK:      [[AR0:%.+]] = IE.AffineReshape([[PC0]])
    // CHECK:      [[PC1:%.+]] = IE.PermuteCast([[INPUT2]])
    // CHECK:      [[AR1:%.+]] = IE.AffineReshape([[PC1]])
    // CHECK:      [[MULTIPLY:%.+]] = IE.Multiply([[AR0]], [[AR1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4xf16>, tensor<1x4xf16> -> tensor<1x4xf16>
    // CHECK:      [[AR2:%.+]] = IE.AffineReshape([[MULTIPLY]])
    // CHECK:      [[PC2:%.+]] = IE.PermuteCast([[AR2]])
    // CHECK:      return [[PC2]] : tensor<1x4x1x1xf16, {order = #NHWC}>
}
