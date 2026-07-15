//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --broadcast-input-for-multiply="broadcast-input-for-multiply=false"  %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @FuseMultiplyBroadcastRightInput
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x64x64x128xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x64x1xf16>
func.func @FuseMultiplyBroadcastRightInput(%arg0: tensor<1x64x64x128xf16>, %arg1: tensor<1x64x64x1xf16>) -> tensor<1x64x64x128xf16> {
    %0 = IE.Tile(%arg1) {repeats_values = [1, 1, 1, 128]} : tensor<1x64x64x1xf16> -> tensor<1x64x64x128xf16>
    %1 = IE.Multiply(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x128xf16>, tensor<1x64x64x128xf16> -> tensor<1x64x64x128xf16>
    return %1 : tensor<1x64x64x128xf16>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x128xf16>, tensor<1x64x64x1xf16> -> tensor<1x64x64x128xf16>
    // CHECK:       return [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @FuseMultiplyBroadcastLeftInput
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x64x64x1xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x64x128xf16>
func.func @FuseMultiplyBroadcastLeftInput(%arg0: tensor<1x64x64x1xf16>, %arg1: tensor<1x64x64x128xf16>) -> tensor<1x64x64x128xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 128]} : tensor<1x64x64x1xf16> -> tensor<1x64x64x128xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x128xf16>, tensor<1x64x64x128xf16> -> tensor<1x64x64x128xf16>
    return %1 : tensor<1x64x64x128xf16>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x1xf16>, tensor<1x64x64x128xf16> -> tensor<1x64x64x128xf16>
    // CHECK:       return [[MULTIPLY]]
}

// -----

// CHECK-LABEL: @FuseMultiplyBroadcastDiffInputShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x64x64x1xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x1x128xf16>
func.func @FuseMultiplyBroadcastDiffInputShape(%arg0: tensor<1x64x64x1xf16>, %arg1: tensor<1x64x1x128xf16>) -> tensor<1x64x64x128xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1, 128]} : tensor<1x64x64x1xf16> -> tensor<1x64x64x128xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x128xf16>, tensor<1x64x1x128xf16> -> tensor<1x64x64x128xf16>
    return %1 : tensor<1x64x64x128xf16>

    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x1xf16>, tensor<1x64x1x128xf16> -> tensor<1x64x64x128xf16>
    // CHECK:       return [[MULTIPLY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @FuseTilePermuteCastIntoMultiply
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x1024x1x128xf16>
// CHECK-SAME:  [[INPUT1:%arg[0-9]]]: tensor<1x1024x64x1xf16, {order = #map}>
func.func @FuseTilePermuteCastIntoMultiply(
        %arg0: tensor<1x1024x1x128xf16>,
        %arg1: tensor<1x1024x64x1xf16, {order = #HCNW}>)
        -> tensor<1x1024x64x128xf16, {order = #HCNW}> {
    %0 = IE.Tile(%arg0) {repeats_values = [64, 1, 1, 1]} : tensor<1x1024x1x128xf16> -> tensor<64x1024x1x128xf16>
    %1 = IE.PermuteCast(%0) {dst_order = #HCNW, mem_perm = #NCHW} : tensor<64x1024x1x128xf16> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    %2 = IE.Multiply(%1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x64x128xf16, {order = #HCNW}>, tensor<1x1024x64x1xf16, {order = #HCNW}> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    return %2 : tensor<1x1024x64x128xf16, {order = #HCNW}>

    // CHECK-NOT: IE.Tile
    // CHECK:     [[PC:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #map, mem_perm = #NCHW} : tensor<1x1024x1x128xf16> -> tensor<1x1024x1x128xf16, {order = #map}>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[PC]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x1024x1x128xf16, {order = #map}>, tensor<1x1024x64x1xf16, {order = #map}>
    // CHECK-SAME:    -> tensor<1x1024x64x128xf16, {order = #map}>
    // CHECK:     return [[MUL]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// Tile->PermuteCast feeds input2 of Multiply (commuted case)
// CHECK-LABEL: @FuseTilePermuteCastIntoMultiplyInput2
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x1024x64x1xf16, {order = #map}>
// CHECK-SAME:  [[INPUT1:%arg[0-9]]]: tensor<1x1024x1x128xf16>
func.func @FuseTilePermuteCastIntoMultiplyInput2(
        %arg0: tensor<1x1024x64x1xf16, {order = #HCNW}>,
        %arg1: tensor<1x1024x1x128xf16>)
        -> tensor<1x1024x64x128xf16, {order = #HCNW}> {
    %0 = IE.Tile(%arg1) {repeats_values = [64, 1, 1, 1]} : tensor<1x1024x1x128xf16> -> tensor<64x1024x1x128xf16>
    %1 = IE.PermuteCast(%0) {dst_order = #HCNW, mem_perm = #NCHW} : tensor<64x1024x1x128xf16> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    %2 = IE.Multiply(%arg0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x64x1xf16, {order = #HCNW}>, tensor<1x1024x64x128xf16, {order = #HCNW}> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    return %2 : tensor<1x1024x64x128xf16, {order = #HCNW}>

    // CHECK-NOT: IE.Tile
    // CHECK:     [[PC:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #map, mem_perm = #NCHW} : tensor<1x1024x1x128xf16> -> tensor<1x1024x1x128xf16, {order = #map}>
    // CHECK:     [[MUL:%.+]] = IE.Multiply([[INPUT0]], [[PC]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x1024x64x1xf16, {order = #map}>, tensor<1x1024x1x128xf16, {order = #map}>
    // CHECK-SAME:    -> tensor<1x1024x64x128xf16, {order = #map}>
    // CHECK:     return [[MUL]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// PermuteCast has two Multiply users; rewriter fires once per Multiply.
// After both rewrites, the original Tile and PermuteCast become dead and are removed.
// CHECK-LABEL: @FuseTilePermuteCastIntoMultiplyMultiUser
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x1024x1x128xf16>
// CHECK-SAME:  [[INPUT1:%arg[0-9]]]: tensor<1x1024x64x1xf16, {order = #map}>
// CHECK-SAME:  [[INPUT2:%arg[0-9]]]: tensor<1x1024x64x1xf16, {order = #map}>
func.func @FuseTilePermuteCastIntoMultiplyMultiUser(
        %arg0: tensor<1x1024x1x128xf16>,
        %arg1: tensor<1x1024x64x1xf16, {order = #HCNW}>,
        %arg2: tensor<1x1024x64x1xf16, {order = #HCNW}>)
        -> (tensor<1x1024x64x128xf16, {order = #HCNW}>, tensor<1x1024x64x128xf16, {order = #HCNW}>) {
    %0 = IE.Tile(%arg0) {repeats_values = [64, 1, 1, 1]} : tensor<1x1024x1x128xf16> -> tensor<64x1024x1x128xf16>
    %1 = IE.PermuteCast(%0) {dst_order = #HCNW, mem_perm = #NCHW} : tensor<64x1024x1x128xf16> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    %2 = IE.Multiply(%1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x64x128xf16, {order = #HCNW}>, tensor<1x1024x64x1xf16, {order = #HCNW}> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    %3 = IE.Multiply(%1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x64x128xf16, {order = #HCNW}>, tensor<1x1024x64x1xf16, {order = #HCNW}> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    return %2, %3 : tensor<1x1024x64x128xf16, {order = #HCNW}>, tensor<1x1024x64x128xf16, {order = #HCNW}>

    // CHECK-NOT: IE.Tile
    // Two new small PermuteCasts (one per Multiply rewrite, processed in IR order)
    // CHECK-DAG: [[PC0:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #map, mem_perm = #NCHW} : tensor<1x1024x1x128xf16> -> tensor<1x1024x1x128xf16, {order = #map}>
    // CHECK:     [[MUL0:%.+]] = IE.Multiply([[PC0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x1024x1x128xf16, {order = #map}>, tensor<1x1024x64x1xf16, {order = #map}>
    // CHECK-SAME:    -> tensor<1x1024x64x128xf16, {order = #map}>
    // CHECK-DAG: [[PC1:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #map, mem_perm = #NCHW} : tensor<1x1024x1x128xf16> -> tensor<1x1024x1x128xf16, {order = #map}>
    // CHECK:     [[MUL1:%.+]] = IE.Multiply([[PC1]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x1024x1x128xf16, {order = #map}>, tensor<1x1024x64x1xf16, {order = #map}>
    // CHECK-SAME:    -> tensor<1x1024x64x128xf16, {order = #map}>
    // CHECK:     return [[MUL0]], [[MUL1]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @FuseTilePermuteCastIntoMultiplyMultiUserMixed
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x1024x1x128xf16>
// CHECK-SAME:  [[INPUT1:%arg[0-9]]]: tensor<1x1024x64x128xf16, {order = #map}>
// CHECK-SAME:  [[INPUT2:%arg[0-9]]]: tensor<1x1024x64x1xf16, {order = #map}>
func.func @FuseTilePermuteCastIntoMultiplyMultiUserMixed(
        %arg0: tensor<1x1024x1x128xf16>,
        %arg1: tensor<1x1024x64x128xf16, {order = #HCNW}>,
        %arg2: tensor<1x1024x64x1xf16, {order = #HCNW}>)
        -> (tensor<1x1024x64x128xf16, {order = #HCNW}>, tensor<1x1024x64x128xf16, {order = #HCNW}>) {
    %0 = IE.Tile(%arg0) {repeats_values = [64, 1, 1, 1]} : tensor<1x1024x1x128xf16> -> tensor<64x1024x1x128xf16>
    %1 = IE.PermuteCast(%0) {dst_order = #HCNW, mem_perm = #NCHW} : tensor<64x1024x1x128xf16> -> tensor<1x1024x64x128xf16, {order = #HCNW}>

    %2 = IE.Multiply(%1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x64x128xf16, {order = #HCNW}>, tensor<1x1024x64x128xf16, {order = #HCNW}> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    %3 = IE.Multiply(%1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x64x128xf16, {order = #HCNW}>, tensor<1x1024x64x1xf16, {order = #HCNW}> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    return %2, %3 : tensor<1x1024x64x128xf16, {order = #HCNW}>, tensor<1x1024x64x128xf16, {order = #HCNW}>

    // CHECK: [[PC_NEW_0:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #map, mem_perm = #NCHW} : tensor<1x1024x1x128xf16> -> tensor<1x1024x1x128xf16, {order = #map}>
    // CHECK: [[MUL0:%.+]] = IE.Multiply([[PC_NEW_0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x1024x1x128xf16, {order = #map}>, tensor<1x1024x64x128xf16, {order = #map}>
    // CHECK-SAME:    -> tensor<1x1024x64x128xf16, {order = #map}>

    // CHECK: [[PC_NEW_1:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #map, mem_perm = #NCHW} : tensor<1x1024x1x128xf16> -> tensor<1x1024x1x128xf16, {order = #map}>
    // CHECK: [[MUL1:%.+]] = IE.Multiply([[PC_NEW_1]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:    tensor<1x1024x1x128xf16, {order = #map}>, tensor<1x1024x64x1xf16, {order = #map}>
    // CHECK-SAME:    -> tensor<1x1024x64x128xf16, {order = #map}>
    // CHECK: return [[MUL0]], [[MUL1]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// Negative: Tile repeats non-1 on a dim whose input size is NOT 1 (C=1024 tiled by 2).
// Rewriter must NOT fire.
// CHECK-LABEL: @NoFuseTileNotBroadcast
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x1024x1x128xf16>
// CHECK-SAME:  [[INPUT1:%arg[0-9]]]: tensor<1x2048x64x128xf16, {order = #map}>
func.func @NoFuseTileNotBroadcast(
        %arg0: tensor<1x1024x1x128xf16>,
        %arg1: tensor<1x2048x64x128xf16, {order = #HCNW}>)
        -> tensor<1x2048x64x128xf16, {order = #HCNW}> {
    %0 = IE.Tile(%arg0) {repeats_values = [64, 2, 1, 1]} : tensor<1x1024x1x128xf16> -> tensor<64x2048x1x128xf16>
    %1 = IE.PermuteCast(%0) {dst_order = #HCNW, mem_perm = #NCHW} : tensor<64x2048x1x128xf16> -> tensor<1x2048x64x128xf16, {order = #HCNW}>
    %2 = IE.Multiply(%1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2048x64x128xf16, {order = #HCNW}>, tensor<1x2048x64x128xf16, {order = #HCNW}> -> tensor<1x2048x64x128xf16, {order = #HCNW}>
    return %2 : tensor<1x2048x64x128xf16, {order = #HCNW}>

    // CHECK: [[TILE:%.+]] = IE.Tile([[INPUT0]])
    // CHECK: [[PERMUTE:%.+]] = IE.PermuteCast([[TILE]])
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[PERMUTE]], [[INPUT1]])
    // CHECK: return [[MULTIPLY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// Negative: after fusion the small PermuteCast output has dim2=1, but the other Multiply input
// also has dim2=1 (not equal to output dim2=64). The other input cannot cover that broadcast dim,
// so the rewrite must NOT fire.
// CHECK-LABEL: @NoFuseBroadcastDimCoverage
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x1024x1x128xf16>
// CHECK-SAME:  [[INPUT1:%arg[0-9]]]: tensor<1x1024x1x128xf16, {order = #map}>
func.func @NoFuseBroadcastDimCoverage(
        %arg0: tensor<1x1024x1x128xf16>,
        %arg1: tensor<1x1024x1x128xf16, {order = #HCNW}>)
        -> tensor<1x1024x64x128xf16, {order = #HCNW}> {
    %0 = IE.Tile(%arg0) {repeats_values = [64, 1, 1, 1]} : tensor<1x1024x1x128xf16> -> tensor<64x1024x1x128xf16>
    %1 = IE.PermuteCast(%0) {dst_order = #HCNW, mem_perm = #NCHW} : tensor<64x1024x1x128xf16> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    %2 = IE.Multiply(%1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x64x128xf16, {order = #HCNW}>, tensor<1x1024x1x128xf16, {order = #HCNW}> -> tensor<1x1024x64x128xf16, {order = #HCNW}>
    return %2 : tensor<1x1024x64x128xf16, {order = #HCNW}>

    // CHECK: [[TILE:%.+]] = IE.Tile([[INPUT0]])
    // CHECK: [[PERMUTE:%.+]] = IE.PermuteCast([[TILE]])
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[PERMUTE]], [[INPUT1]])
    // CHECK: return [[MULTIPLY]]
}
