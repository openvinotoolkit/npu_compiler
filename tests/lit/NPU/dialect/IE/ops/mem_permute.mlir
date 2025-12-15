//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @FoldMemPermute
func.func @FoldMemPermute(%arg0: tensor<1x16x2x3xf32>) -> tensor<1x16x2x3xf32> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<1x16x2x3xf32> -> tensor<1x16x2x3xf32>
    return %0 : tensor<1x16x2x3xf32>

    // CHECK-NOT: IE.MemPermute
    // CHECK:     return %arg0 : tensor<1x16x2x3xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConstFoldAndFuse
func.func @ConstFoldAndFuse() -> tensor<1x2x3x4xf32> {
    %0 = const.Declare tensor<1x2x3x4xf32, {order = #NHWC}> = dense<5.0> : tensor<1x2x3x4xf32>, [#const.Reorder<#NHWC>]
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>
    return %1 : tensor<1x2x3x4xf32>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<1x2x3x4xf32> = dense<5.000000e+00> : tensor<1x2x3x4xf32>
    // CHECK-NOT:   #const.Reorder
    // CHECK-NOT:   #const.MemPermute

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       return [[CST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<1x2x3x4xf32, {order = #NHWC}> {
    %0 = const.Declare tensor<1x2x3x4xf32> = dense<5.0> : tensor<1x2x3x4xf32>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    return %1 : tensor<1x2x3x4xf32, {order = #NHWC}>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<1x2x3x4xf32, {order = #NHWC}> = dense<5.000000e+00> : tensor<1x2x3x4xf32>,
    // CHECK-SAME:        [#const.MemPermute<#NHWC, #NHWC>]
    // CHECK-NOT:   IE.MemPermute
    // CHECK:       return [[CST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL:   @FuseMemPermutes
func.func @FuseMemPermutes(%arg0: tensor<1x16x2x3xf32>, %arg1: tensor<1x16x2x3xf32, {order = #NHWC}>) ->
        (tensor<1x3x2x16xf32>, tensor<1x3x16x2xf32>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NCWH} :
        tensor<1x3x16x2xf32> -> tensor<1x3x2x16xf32>

    %2 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NWCH} :
        tensor<1x16x2x3xf32, {order = #NHWC}> -> tensor<1x3x16x2xf32, {order = #NHWC}>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x3x16x2xf32, {order = #NHWC}> -> tensor<1x3x16x2xf32>

    return %1, %3 : tensor<1x3x2x16xf32>, tensor<1x3x16x2xf32>

    // CHECK-NOT: IE.MemPermute
    // CHECK-NOT: IE.MemPermute
    // CHECK:     %[[VAL_0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x16x2x3xf32> -> tensor<1x3x2x16xf32>
    // CHECK:     %[[VAL_1:.*]] = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x2x3xf32, {order = #NHWC}> -> tensor<1x3x16x2xf32>
    // CHECK:     return %[[VAL_0]], %[[VAL_1]] : tensor<1x3x2x16xf32>, tensor<1x3x16x2xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @ConvertToPermuteCast
func.func @ConvertToPermuteCast(
        %arg0: tensor<1x100x1x1xf32>,
        %arg1: tensor<1x100x1x1xf32, {order = #NHWC}>,
        %arg2: tensor<1x1x256x32xf32, {order = #NHWC}>,
        %arg3: tensor<1x512x2x1xf32, {order = #NHWC}>) ->
            (tensor<1x1x100x1xf32>, tensor<1x1x1x100xf32>, tensor<1x1x256x32xf32>, tensor<1x2x1x512xf32>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x100x1x1xf32> -> tensor<1x1x100x1xf32>

    %1 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<1x100x1x1xf32, {order = #NHWC}> -> tensor<1x1x1x100xf32>

    %2 = IE.MemPermute(%arg2) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x1x256x32xf32, {order = #NHWC}> -> tensor<1x1x256x32xf32>

    %3 = IE.MemPermute(%arg3) {dst_order = #NCHW, mem_perm = #NCHW} :
        tensor<1x512x2x1xf32, {order = #NHWC}> -> tensor<1x2x1x512xf32>

    return %0, %1, %2, %3 : tensor<1x1x100x1xf32>, tensor<1x1x1x100xf32>, tensor<1x1x256x32xf32>, tensor<1x2x1x512xf32>

    //CHECK: [[VAR0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NWCH}
    //CHECK: [[VAR1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NCHW}
    //CHECK: [[VAR2:%.+]] = IE.PermuteCast(%arg2) {dst_order = #NCHW, mem_perm = #NWCH}
    //CHECK: [[VAR3:%.+]] = IE.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW}
    //CHECK: return [[VAR0]], [[VAR1]], [[VAR2]], [[VAR3]] : tensor<1x1x100x1xf32>, tensor<1x1x1x100xf32>, tensor<1x1x256x32xf32>, tensor<1x2x1x512xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @FusePermCastAndMemPerm
func.func @FusePermCastAndMemPerm(%arg0: tensor<1x1000x1x1xf32, {order = #NHWC}>) ->
            tensor<1x1x1000x1xf32> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NWCH} :
            tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x1x1000x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>

    return %1 : tensor<1x1x1000x1xf32>

    // CHECK:     %[[VAL_0:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>
    // CHECK:     return %[[VAL_0]] : tensor<1x1x1000x1xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL:   @NotFuseMemPermute
func.func @NotFuseMemPermute(%arg0: tensor<1x8x4x64xf16>) -> (tensor<1x64x8x4xf16>, tensor<1x64x4x8xf16>, tensor<1x4x64x8xf16, {order = #NHWC}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x8x4x64xf16> -> tensor<1x64x8x4xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NCWH} :
        tensor<1x64x8x4xf16> -> tensor<1x64x4x8xf16>
    %2 = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW} :
        tensor<1x64x8x4xf16> -> tensor<1x4x64x8xf16, {order = #NHWC}>

    return %0, %1, %2 : tensor<1x64x8x4xf16>, tensor<1x64x4x8xf16>, tensor<1x4x64x8xf16, {order = #NHWC}>

    // CHECK:     %[[VAL_0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4x64xf16> -> tensor<1x64x8x4xf16>
    // CHECK:     %[[VAL_1:.*]] = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x64x8x4xf16> -> tensor<1x64x4x8xf16>
    // CHECK:     %[[VAL_2:.*]] = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x8x4xf16> -> tensor<1x4x64x8xf16, {order = #NHWC}>
    // CHECK:     return %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]  : tensor<1x64x8x4xf16>, tensor<1x64x4x8xf16>, tensor<1x4x64x8xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL:   @FuseMemPermuteForAllUserMemPerm
func.func @FuseMemPermuteForAllUserMemPerm(%arg0: tensor<1x8x4x64xf16>) -> tensor<1x64x8x4xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x8x4x64xf16> -> tensor<1x64x8x4xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NCWH} :
        tensor<1x64x8x4xf16> -> tensor<1x64x4x8xf16>
    %2 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NCHW} :
        tensor<1x64x8x4xf16> -> tensor<1x4x64x8xf16, {order = #NHWC}>

    return %0 : tensor<1x64x8x4xf16>

    // CHECK:     %[[VAL_0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x4x64xf16> -> tensor<1x64x8x4xf16>
    // CHECK-NOT: IE.MemPermute
    // CHECK-NOT: IE.MemPermute
    // CHECK:     return %[[VAL_0]] : tensor<1x64x8x4xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL:   @FuseFP16PermuteQuantizeAndPermute
func.func @FuseFP16PermuteQuantizeAndPermute(%arg0: tensor<1x3x13x26xf16>) -> tensor<1x3x169x2xf16> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x13x26xf16> -> tensor<1x3x13x26xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x13x26xf16, {order = #NHWC}> -> tensor<1x3x13x26xf16>
    %2 = IE.Reshape(%1) {shape_value = [1, 3, 169, 2]} : tensor<1x3x13x26xf16> -> tensor<1x3x169x2xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3x169x2xf16>, tensor<1x3x169x2xf16> -> tensor<1x3x169x2xf16>

    return %3 : tensor<1x3x169x2xf16>

    // CHECK-NOT: IE.PermuteQuantize
    // CHECK-NOT: IE.MemPermute
    // CHECK:     %[[VAL_0:.*]] = IE.Reshape(%arg0) {shape_value = [1, 3, 169, 2]} : tensor<1x3x13x26xf16> -> tensor<1x3x169x2xf16>
    // CHECK:     %[[VAL_1:.*]] = IE.Add(%[[VAL_0]], %[[VAL_0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3x169x2xf16>, tensor<1x3x169x2xf16> -> tensor<1x3x169x2xf16>

    // CHECK:     return %[[VAL_1]] : tensor<1x3x169x2xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0033655795396543018>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL:   @NotFuseU8PermuteQuantizeAndPermute
func.func @NotFuseU8PermuteQuantizeAndPermute(%arg0: tensor<1x3x13x26xf16>) -> tensor<1x3x169x2xf16> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x13x26xf16> -> tensor<1x3x13x26x!qElemType, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x13x26x!qElemType, {order = #NHWC}> -> tensor<1x3x13x26x!qElemType>
    %2 = IE.Reshape(%1) {shape_value = [1, 3, 169, 2]} : tensor<1x3x13x26x!qElemType> -> tensor<1x3x169x2x!qElemType>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3x169x2x!qElemType>, tensor<1x3x169x2x!qElemType> -> tensor<1x3x169x2xf16>

    return %3 : tensor<1x3x169x2xf16>

    // CHECK:     %[[VAL_0:.*]] = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x13x26xf16> -> tensor<1x3x13x26x!qElemType, {order = #NHWC}>
    // CHECK:     %[[VAL_1:.*]] = IE.MemPermute(%[[VAL_0]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x13x26x!qElemType, {order = #NHWC}> -> tensor<1x3x13x26x!qElemType>
    // CHECK:     %[[VAL_3:.*]] = IE.Reshape(%[[VAL_1]]) {shape_value = [1, 3, 169, 2]} : tensor<1x3x13x26x!qElemType> -> tensor<1x3x169x2x!qElemType>
    // CHECK:     %[[VAL_4:.*]] = IE.Add(%[[VAL_3]], %[[VAL_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3x169x2x!qElemType>, tensor<1x3x169x2x!qElemType> -> tensor<1x3x169x2xf16>

    // CHECK:     return %[[VAL_4]] : tensor<1x3x169x2xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @FuseMemPermThroughConcat
func.func @FuseMemPermThroughConcat(%arg0: tensor<1x16x2x3xf32>, %arg1: tensor<1x16x2x3xf32>) ->
            tensor<1x2x6x16xf32> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>
    %1 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x3x16x2xf32>, tensor<1x3x16x2xf32> -> tensor<1x6x16x2xf32>

    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x6x16x2xf32> -> tensor<1x2x6x16xf32>

    return %3 : tensor<1x2x6x16xf32>

    // CHECK-NOT:     IE.MemPermute

    // CHECK:     %[[VAL_0:.*]] = IE.Concat(%arg0, %arg1)
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 3]]}
    // CHECK-SAME:     tensor<1x16x2x3xf32>, tensor<1x16x2x3xf32> -> tensor<1x16x2x6xf32>

    // CHECK:     %[[VAL_1:.*]] = IE.MemPermute(%[[VAL_0:.*]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x2x6xf32> -> tensor<1x2x6x16xf32>
    // CHECK:     return %[[VAL_1:.*]] : tensor<1x2x6x16xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @NotFuseMemPermThroughConcat
// CHECK-SAME:    ([[INPUT_0:%.+]]: tensor<1x16x2x3xf32>, [[INPUT_1:%.+]]: tensor<1x16x2x3xf32>)
func.func @NotFuseMemPermThroughConcat(%arg0: tensor<1x16x2x3xf32>, %arg1: tensor<1x16x2x3xf32>) ->
            (tensor<1x6x16x2xf32>, tensor<1x2x6x16xf32>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>
    %1 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x3x16x2xf32>, tensor<1x3x16x2xf32> -> tensor<1x6x16x2xf32>

    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x6x16x2xf32> -> tensor<1x2x6x16xf32>

    return %2, %3 : tensor<1x6x16x2xf32>, tensor<1x2x6x16xf32>

    // CHECK:     [[MEM_PERMUTE_0:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>
    // CHECK:     [[MEM_PERMUTE_1:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>

    // CHECK:     [[CONCAT:%.+]] = IE.Concat([[MEM_PERMUTE_0]], [[MEM_PERMUTE_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0]]}
    // CHECK-SAME:     tensor<1x3x16x2xf32>, tensor<1x3x16x2xf32> -> tensor<1x6x16x2xf32>

    // CHECK:     [[MEM_PERMUTE_OUT:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x6x16x2xf32> -> tensor<1x2x6x16xf32>
    // CHECK:     return [[CONCAT]], [[MEM_PERMUTE_OUT]] : tensor<1x6x16x2xf32>, tensor<1x2x6x16xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL:   @FuseMemPermThroughExpand
func.func @FuseMemPermThroughExpand(%arg0: tensor<1x289x289x1xf16>) ->
            tensor<1x1x289x304xf16>  {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x289x289x1xf16> -> tensor<1x1x289x289xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 15, 0]} : tensor<1x1x289x289xf16> -> tensor<1x1x304x289xf16>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x304x289xf16> -> tensor<1x1x289x304xf16>

    return %2 : tensor<1x1x289x304xf16>

    // CHECK-NOT:     IE.MemPermute

    // CHECK:     %[[VAL_0:.*]] = IE.Expand(%arg0)
    // CHECK-SAME{LITERAL}:     {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 15, 0]}
    // CHECK-SAME:     tensor<1x289x289x1xf16> -> tensor<1x289x304x1xf16>

    // CHECK:     %[[VAL_1:.*]] = IE.PermuteCast(%[[VAL_0]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x289x304x1xf16> -> tensor<1x1x289x304xf16>

    // CHECK:     return %[[VAL_1:.*]] : tensor<1x1x289x304xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL:   @NotFuseMemPermThroughExpandForNonTrivialCase
func.func @NotFuseMemPermThroughExpandForNonTrivialCase(%arg0: tensor<1x289x289x10xf16>) ->
            tensor<1x10x289x304xf16>  {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x289x289x10xf16> -> tensor<1x10x289x289xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 15, 0]} : tensor<1x10x289x289xf16> -> tensor<1x10x304x289xf16>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x10x304x289xf16> -> tensor<1x10x289x304xf16>

    return %2 : tensor<1x10x289x304xf16>

    // CHECK:     %[[VAL_0:.*]] = IE.MemPermute(%arg0)
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #NWHC}
    // CHECK-SAME:     tensor<1x289x289x10xf16> -> tensor<1x10x289x289xf16>

    // CHECK:     %[[VAL_1:.*]] = IE.Expand(%[[VAL_0:.*]])
    // CHECK-SAME{LITERAL}:     {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 15, 0]}
    // CHECK-SAME:     tensor<1x10x289x289xf16> -> tensor<1x10x304x289xf16>

    // CHECK:     %[[VAL_2:.*]] = IE.MemPermute(%[[VAL_1:.*]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #NCWH}
    // CHECK-SAME:     tensor<1x10x304x289xf16> -> tensor<1x10x289x304xf16>

    // CHECK:     return %[[VAL_2:.*]] : tensor<1x10x289x304xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHWN = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#WCNH = affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @FuseMemPermThroughExpandWithDifferentAxisOnChannel
func.func @FuseMemPermThroughExpandWithDifferentAxisOnChannel(%arg0: tensor<1x2x71x1xf16, {order = #NHWC}>) ->
            tensor<71x1x1x16xf16>  {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #WCNH} : tensor<1x2x71x1xf16, {order = #NHWC}> -> tensor<2x71x1x1xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [14, 0, 0, 0]} : tensor<2x71x1x1xf16> -> tensor<16x71x1x1xf16>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #CHWN} : tensor<16x71x1x1xf16> -> tensor<71x1x1x16xf16>

    return %2 : tensor<71x1x1x16xf16>

    // CHECK-NOT:     IE.MemPermute

    // CHECK:     %[[VAL_0:.*]] = IE.Expand(%arg0)
    // CHECK-SAME{LITERAL}:     {pads_begin = [0, 0, 0, 0], pads_end = [0, 14, 0, 0]}
    // CHECK-SAME:     tensor<1x2x71x1xf16, {order = #NHWC}> -> tensor<1x16x71x1xf16, {order = #NHWC}>

    // CHECK:     %[[VAL_1:.*]] = IE.PermuteCast(%[[VAL_0]]) {dst_order = #NCHW, mem_perm = #map} : tensor<1x16x71x1xf16, {order = #NHWC}> -> tensor<71x1x1x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @FuseMemPermThroughExpandWithDifferentAxisOnHeight
func.func @FuseMemPermThroughExpandWithDifferentAxisOnHeight(%arg0: tensor<1x1x289x289xf16, {order = #NHWC}>) -> tensor<1x1x289x304xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x1x289x289xf16, {order = #NHWC}> -> tensor<1x1x289x289xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 15, 0]} : tensor<1x1x289x289xf16> -> tensor<1x1x304x289xf16>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x304x289xf16> -> tensor<1x1x289x304xf16>

    return %2: tensor<1x1x289x304xf16>

    // CHECK-NOT:     IE.MemPermute

    // CHECK:     %[[VAL_0:.*]] = IE.Expand(%arg0)
    // CHECK-SAME{LITERAL}:     {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]}
    // CHECK-SAME:     tensor<1x1x289x289xf16, {order = #NHWC}> -> tensor<1x1x289x304xf16, {order = #NHWC}>

    // CHECK:     %[[VAL_1:.*]] = IE.PermuteCast(%[[VAL_0]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x289x304xf16, {order = #NHWC}> -> tensor<1x1x289x304xf16>
}


// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#HNWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

// CHECK-LABEL:   @FuseMemPermThroughExpandComplexLayoutCase
func.func @FuseMemPermThroughExpandComplexLayoutCase(%arg0: tensor<1x16x75x1xf16, {order = #NHWC}>) ->
            tensor<80x16x1x1xf16, {order = #NHWC}>  {
    %0 = IE.MemPermute(%arg0) {dst_order = #HCNW, mem_perm = #NWCH} : tensor<1x16x75x1xf16, {order = #NHWC}> -> tensor<75x16x1x1xf16, {order = #HCNW}>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [5, 0, 0, 0]} : tensor<75x16x1x1xf16, {order = #HCNW}> -> tensor<80x16x1x1xf16, {order = #HCNW}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #HNWC} : tensor<80x16x1x1xf16, {order = #HCNW}> -> tensor<80x16x1x1xf16, {order = #NHWC}>

    return %2 : tensor<80x16x1x1xf16, {order = #NHWC}>

    // CHECK-NOT:     IE.MemPermute

    // CHECK:     %[[VAL_0:.*]] = IE.Expand(%arg0)
    // CHECK-SAME{LITERAL}:     {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 5, 0]}
    // CHECK-SAME:     tensor<1x16x75x1xf16, {order = #NHWC}> -> tensor<1x16x80x1xf16, {order = #NHWC}>

    // CHECK:     %[[VAL_1:.*]] = IE.PermuteCast(%[[VAL_0]]) {dst_order = #NHWC, mem_perm = #map} : tensor<1x16x80x1xf16, {order = #NHWC}> -> tensor<80x16x1x1xf16, {order = #NHWC}>
    // CHECK:     return %[[VAL_1]] : tensor<80x16x1x1xf16, {order = #NHWC}>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL:   @NotFuseFP16PaddingPermuteQuantizeAndPermute
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x3x13x25xf16>
func.func @NotFuseFP16PaddingPermuteQuantizeAndPermute(%arg0: tensor<1x3x13x25xf16>) -> tensor<1x3x169x2xf16> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 1]} : tensor<1x3x13x25xf16> -> tensor<1x3x13x26xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x13x26xf16, {order = #NHWC}> -> tensor<1x3x13x26xf16>
    %2 = IE.Reshape(%1) {shape_value = [1, 3, 169, 2]} : tensor<1x3x13x26xf16> -> tensor<1x3x169x2xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3x169x2xf16>, tensor<1x3x169x2xf16> -> tensor<1x3x169x2xf16>

    return %3 : tensor<1x3x169x2xf16>

    // CHECK:     [[PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize([[INPUT]])
    // CHECK:     [[MEMPERMUTE:%.*]] = IE.MemPermute([[PERMUTEQUANTIZE]])
    // CHECK:     [[RESHAPE:%.*]] = IE.Reshape([[MEMPERMUTE]])
    // CHECK:     [[ADD:%.*]] = IE.Add([[RESHAPE]], [[RESHAPE]])

    // CHECK:     return [[ADD]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @ConvertShapeCastToPermuteCast
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x49x1x1024xf16, {order = #NHWC}>
func.func @ConvertShapeCastToPermuteCast(%arg0: tensor<1x49x1x1024xf16, {order = #NHWC}>) -> tensor<1x49x1x1024xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x1x49x1024xf16>
    %1 = IE.ShapeCast {shape = [1, 49, 1, 1024]} inputs(%0 : tensor<1x1x49x1024xf16>) -> tensor<1x49x1x1024xf16>

    return %1 : tensor<1x49x1x1024xf16>

    // CHECK:     [[MEMPERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16>
    // CHECK:     return [[MEMPERMUTE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL:   @ConvertShapeCastToPermuteCastWhenProducerHasMultiUser
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x49x1x1024xf16, {order = #NHWC}>
func.func @ConvertShapeCastToPermuteCastWhenProducerHasMultiUser(%arg0: tensor<1x49x1x1024xf16, {order = #NHWC}>) -> (tensor<1x49x1x1024xf16>, tensor<1x1x49x1024xf16>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x1x49x1024xf16>
    %1 = IE.ShapeCast {shape = [1, 49, 1, 1024]} inputs(%0 : tensor<1x1x49x1024xf16>) -> tensor<1x49x1x1024xf16>
    %2 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x49x1024xf16>, tensor<1x1x49x1024xf16> -> tensor<1x1x49x1024xf16>

    return %1, %2 : tensor<1x49x1x1024xf16>, tensor<1x1x49x1024xf16>

    // CHECK:     [[MEMPERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x1x49x1024xf16>
    // CHECK:     [[SHAPECAST:%.+]] = IE.PermuteCast([[MEMPERMUTE]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x1x49x1024xf16> -> tensor<1x49x1x1024xf16>
    // CHECK:     [[ADD:%.+]] = IE.Add([[MEMPERMUTE]], [[MEMPERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x49x1024xf16>, tensor<1x1x49x1024xf16> -> tensor<1x1x49x1024xf16>
    // CHECK:     return [[SHAPECAST]], [[ADD]] : tensor<1x49x1x1024xf16>, tensor<1x1x49x1024xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL:   @NotConvertShapeCastWhenShapeChanged
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x49x1x1024xf16, {order = #NHWC}>
func.func @NotConvertShapeCastWhenShapeChanged(%arg0: tensor<1x49x1x1024xf16, {order = #NHWC}>) -> tensor<1x16x49x64xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x1x49x1024xf16>
    %1 = IE.ShapeCast {shape = [1, 16, 49, 64]} inputs(%0 : tensor<1x1x49x1024xf16>) -> tensor<1x16x49x64xf16>

    return %1 : tensor<1x16x49x64xf16>

    // CHECK:     [[MEMPERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x1x49x1024xf16>
    // CHECK:     [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 16, 49, 64]} inputs([[MEMPERMUTE]] : tensor<1x1x49x1024xf16>) -> tensor<1x16x49x64xf16>
    // CHECK:     return [[SHAPECAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL:   @ConvertShapeCastWhenPermuteCastFound
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x2x2xf16, {order = #NHWC}>
func.func @ConvertShapeCastWhenPermuteCastFound(%arg0: tensor<1x1x2x2xf16, {order = #NHWC}>) -> tensor<1x1x2x2xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x1x2x2xf16, {order = #NHWC}> -> tensor<1x2x2x1xf16>
    %1 = IE.ShapeCast {shape = [1, 1, 2, 2]} inputs(%0 : tensor<1x2x2x1xf16>) -> tensor<1x1x2x2xf16>

    return %1 : tensor<1x1x2x2xf16>

    // CHECK:     [[MEMPERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x1x2x2xf16, {order = #NHWC}> -> tensor<1x1x2x2xf16>
    // CHECK:     return [[MEMPERMUTE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @NotFuseMemPermThroughConcat
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x256x2x256xsi8, {order = #NHWC}>
// CHECK-SAME:    [[INPUT_1:%.+]]: tensor<1x2x256x3840xsi8>
func.func @NotFuseMemPermThroughConcat(%arg0: tensor<1x256x2x256xsi8, {order = #NHWC}>, %arg1: tensor<1x2x256x3840xsi8>) -> tensor<1x2x256x4096xsi8> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x2x256xsi8, {order = #NHWC}> -> tensor<1x2x256x256xsi8, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x256x3840xsi8> -> tensor<1x2x256x3840xsi8, {order = #NHWC}>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 256]]} : tensor<1x2x256x256xsi8, {order = #NHWC}>, tensor<1x2x256x3840xsi8, {order = #NHWC}> -> tensor<1x2x256x4096xsi8, {order = #NHWC}>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x256x4096xsi8, {order = #NHWC}> -> tensor<1x2x256x4096xsi8>

    return %3 : tensor<1x2x256x4096xsi8>

    // CHECK:     [[IN_0:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x2x256xsi8, {order = #NHWC}> -> tensor<1x2x256x256xsi8, {order = #NHWC}>
    // CHECK:     [[IN_1:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x256x3840xsi8> -> tensor<1x2x256x3840xsi8, {order = #NHWC}>
    // CHECK:     [[CONCAT:%.+]] = IE.Concat([[IN_0]], [[IN_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 256]]} : tensor<1x2x256x256xsi8, {order = #NHWC}>, tensor<1x2x256x3840xsi8, {order = #NHWC}> -> tensor<1x2x256x4096xsi8, {order = #NHWC}>
    // CHECK:     [[OUT:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x256x4096xsi8, {order = #NHWC}> -> tensor<1x2x256x4096xsi8>

    // CHECK:     return [[OUT]] : tensor<1x2x256x4096xsi8>
}
