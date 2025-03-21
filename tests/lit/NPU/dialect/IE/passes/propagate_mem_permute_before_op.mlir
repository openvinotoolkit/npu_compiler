//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-mem-permute-before-op %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @PropagateMemPermuteThroughAffineReshape(%arg0: tensor<1x1280x1x4096xf16>) -> tensor<1x1280x4096x1xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 4096, 1]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x4096x1xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    return %1 : tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1280x1x4096xf16> -> tensor<1x4096x1280x1xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 4096, 1, 1280]} : tensor<1x4096x1280x1xf16> -> tensor<1x4096x1x1280xf16>
    // CHECK:               [[PERMUTECAST:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4096x1x1280xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               return [[PERMUTECAST]] : tensor<1x1280x4096x1xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

func.func @PropagateMemPermuteThroughAffineReshapeWithPermNWHC(%arg0: tensor<1x64x64x320xf16, {order = #NWCH}>) -> tensor<1x4096x320x1xf16, {order = #NCWH}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4096, 320, 1]} : tensor<1x64x64x320xf16, {order = #NWCH}> -> tensor<1x4096x320x1xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x4096x320x1xf16, {order = #NHWC}> -> tensor<1x4096x320x1xf16, {order = #NCWH}>
    return %1 : tensor<1x4096x320x1xf16, {order = #NCWH}>
    // CHECK:                [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x64x64x320xf16, {order = #NWCH}> -> tensor<1x64x64x320xf16>
    // CHECK:                [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4096, 1, 320]} : tensor<1x64x64x320xf16> -> tensor<1x4096x1x320xf16>
    // CHECK:                [[PERMUTECAST:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x4096x1x320xf16> -> tensor<1x4096x320x1xf16, {order = #NCWH}>
    // CHECK:                return [[PERMUTECAST]] : tensor<1x4096x320x1xf16, {order = #NCWH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @NotPropagateMemPermuteIfBrokenSplitAxis(%arg0: tensor<1x1280x1x4096xf16>) -> tensor<1x1280x2048x2xf16, {order = #NHCW}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 2048, 2]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x2048x2xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NHCW, mem_perm = #NHCW} : tensor<1x1280x2048x2xf16> -> tensor<1x1280x2048x2xf16, {order = #NHCW}>

    return %1 : tensor<1x1280x2048x2xf16, {order = #NHCW}>

    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 2048, 2]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x2048x2xf16>
    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute([[RESHAPE]]) {dst_order = #NHCW, mem_perm = #NHCW} : tensor<1x1280x2048x2xf16> -> tensor<1x1280x2048x2xf16, {order = #NHCW}>

    // CHECK:               return [[MEMPERMUTE]] : tensor<1x1280x2048x2xf16, {order = #NHCW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @PropagateMemPermuteIfBreakSplitAxisWithSingleNonTrivialMemShape(%arg0: tensor<1x8x4096x40xf16>) -> tensor<8x40x4096x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [8, 4096, 40, 1]} : tensor<1x8x4096x40xf16> -> tensor<8x4096x40x1xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<8x4096x40x1xf16> -> tensor<8x40x4096x1xf16>

    return %1 : tensor<8x40x4096x1xf16>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x4096x40xf16> -> tensor<1x8x40x4096xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [8, 40, 4096, 1]} : tensor<1x8x40x4096xf16> -> tensor<8x40x4096x1xf16>

    // CHECK:               return [[RESHAPE]] : tensor<8x40x4096x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @PropagateMemPermuteInCaseSplitAxisIsTrivial(%arg0: tensor<1x768x14x14xf16, {order = #NHWC}>) -> tensor<1x1x196x768xf16, {order = #NCWH}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 768, 196]} : tensor<1x768x14x14xf16, {order = #NHWC}> -> tensor<1x1x768x196xf16, {order = #NCWH}>
    %1 = IE.MemPermute(%0) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x1x768x196xf16, {order = #NCWH}> -> tensor<1x1x196x768xf16, {order = #NCWH}>

    return %1 : tensor<1x1x196x768xf16, {order = #NCWH}>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x768x14x14xf16, {order = #NHWC}> -> tensor<1x768x14x14xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 768, 196]} : tensor<1x768x14x14xf16> -> tensor<1x1x768x196xf16>
    // CHECK:               [[PERMUTECAST:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x1x768x196xf16> -> tensor<1x1x196x768xf16, {order = #NCWH}>

    // CHECK:               return [[PERMUTECAST]] : tensor<1x1x196x768xf16, {order = #NCWH}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PropagateMemPermuteNewCaseWithSplitAxisTrivial
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x128x128xf16, {order = #NHWC}>
func.func @PropagateMemPermuteNewCaseWithSplitAxisTrivial(%arg0: tensor<1x1x128x128xf16, {order = #NHWC}>) -> tensor<1x1x1x16384xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 16384]} : tensor<1x1x128x128xf16, {order = #NHWC}> -> tensor<1x1x1x16384xf16, {order = #NWCH}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x1x16384xf16, {order = #NWCH}> -> tensor<1x1x1x16384xf16>

    return %1 : tensor<1x1x1x16384xf16>

    // CHECK:               [[PERMUTECAST1:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x128x128xf16, {order = #NHWC}> -> tensor<1x128x128x1xf16>
    // CHECK:               [[PERMUTECAST2:%.+]] = IE.PermuteCast([[PERMUTECAST1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x128x128x1xf16> -> tensor<1x1x128x128xf16>
    // CHECK:               [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[PERMUTECAST2]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 1, 16384]} : tensor<1x1x128x128xf16> -> tensor<1x1x1x16384xf16>

    // CHECK:               return [[AFFINERESHAPE]] : tensor<1x1x1x16384xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @PropagateMemPermuteThroughAffineReshapeNHWCInput(%arg0: tensor<1x4096x1x320xf16, {order = #NHWC}>) -> tensor<1x64x64x320xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 64, 64, 320]} : tensor<1x4096x1x320xf16, {order = #NHWC}> -> tensor<1x64x64x320xf16, {order = #NWCH}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x64x64x320xf16, {order = #NWCH}> -> tensor<1x64x64x320xf16>

    return %1 : tensor<1x64x64x320xf16>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4096x1x320xf16, {order = #NHWC}> -> tensor<1x4096x1x320xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 64, 64, 320]} : tensor<1x4096x1x320xf16> -> tensor<1x64x64x320xf16>

    // CHECK:               return [[RESHAPE]] : tensor<1x64x64x320xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

func.func @PropagateMemPermuteWithOrigDimMapping(%arg0: tensor<1x512x1x1500xf16>) -> tensor<1x1x1500x512xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1, 512, 1500]} : tensor<1x512x1x1500xf16> -> tensor<1x1x512x1500xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x1x512x1500xf16> -> tensor<1x1x1500x512xf16, {order = #NHWC}>

    return %1 : tensor<1x1x1500x512xf16, {order = #NHWC}>

    // CHECK: [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC}
    // CHECK-SAME:                 tensor<1x512x1x1500xf16> -> tensor<1x1x1500x512xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 1500, 512, 1]}
    // CHECK-SAME:             tensor<1x1x1500x512xf16> -> tensor<1x1500x512x1xf16>
    // CHECK: [[PERMUTECAST:%.*]] = IE.PermuteCast([[AFFINERESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:                   tensor<1x1500x512x1xf16> -> tensor<1x1x1500x512xf16, {order = #NHWC}>
    // CHECK: return [[PERMUTECAST]] : tensor<1x1x1500x512xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

func.func @PropagateMemPermuteWithChangedDimMapping(%arg0: tensor<1x512x1x1500xf16>) -> tensor<1x1x1500x512xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 512, 1500]} : tensor<1x512x1x1500xf16> -> tensor<1x1x512x1500xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x1x512x1500xf16> -> tensor<1x1x1500x512xf16, {order = #NHWC}>

    return %1 : tensor<1x1x1500x512xf16, {order = #NHWC}>

    // CHECK: [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC}
    // CHECK-SAME:                 tensor<1x512x1x1500xf16> -> tensor<1x1x1500x512xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 1500, 512, 1]}
    // CHECK-SAME:             tensor<1x1x1500x512xf16> -> tensor<1x1500x512x1xf16>
    // CHECK: [[PERMUTECAST:%.*]] = IE.PermuteCast([[AFFINERESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:                   tensor<1x1500x512x1xf16> -> tensor<1x1x1500x512xf16, {order = #NHWC}>
    // CHECK: return [[PERMUTECAST]] : tensor<1x1x1500x512xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @PropagatePermuteQuantizeAndCancelPermute
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4096x1x1280xf16>
func.func @PropagatePermuteQuantizeAndCancelPermute(%arg0: tensor<1x4096x1x1280xf16>) -> tensor<1x1280x4096x1xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x4096x1x1280xf16> -> tensor<1x1280x1x4096xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 4096, 1]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x4096x1xf16>
    %2 = IE.PermuteQuantize(%1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    return %2 : tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               [[IN_PERMUTECAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x4096x1x1280xf16> -> tensor<1x4096x1280x1xf16>
    // CHECK:               [[RESHAPE:%.+]] = IE.AffineReshape([[IN_PERMUTECAST]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 4096, 1, 1280]} : tensor<1x4096x1280x1xf16> -> tensor<1x4096x1x1280xf16>
    // CHECK:               [[OUT_PERMUTECAST:%.+]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4096x1x1280xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               return [[OUT_PERMUTECAST]] : tensor<1x1280x4096x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @PropagatePermuteQuantizeAndFuseMemPermute
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4096x4x1280xf16>
func.func @PropagatePermuteQuantizeAndFuseMemPermute(%arg0: tensor<1x4096x4x1280xf16>) -> tensor<1280x1x4096x4xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x4096x4x1280xf16> -> tensor<1x1280x4x4096xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1280, 1, 4, 4096]} : tensor<1x1280x4x4096xf16> -> tensor<1280x1x4x4096xf16>
    %2 = IE.PermuteQuantize(%1) {dstElemType = f16, dst_order = #NCHW, mem_perm = #NCWH, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1280x1x4x4096xf16> -> tensor<1280x1x4096x4xf16>

    return %2 : tensor<1280x1x4096x4xf16>

    // CHECK:               [[MEMPERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4096x4x1280xf16> -> tensor<1x1280x4096x4xf16>
    // CHECK:               [[RESHAPE:%.+]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1280, 1, 4096, 4]} : tensor<1x1280x4096x4xf16> -> tensor<1280x1x4096x4xf16>

    // CHECK:               return [[RESHAPE]] : tensor<1280x1x4096x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MovePermuteQuantizeThroughAffineReshape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x1x3584xf16>
func.func @MovePermuteQuantizeThroughAffineReshape(%arg0: tensor<1x1024x1x3584xf16>) -> tensor<1x1024x64x56xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1024, 64, 56]} : tensor<1x1024x1x3584xf16> -> tensor<1x1024x64x56xf16>
    %1 = IE.PermuteQuantize(%0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1024x64x56xf16> -> tensor<1x1024x64x56xf16, {order = #NHWC}>

    return %1 : tensor<1x1024x64x56xf16, {order = #NHWC}>

    // CHECK:               [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1024x1x3584xf16> -> tensor<1x1024x1x3584xf16, {order = #NHWC}>
    // CHECK:               [[RESHAPE:%.+]] = IE.ShapeCast {shape = [1, 1024, 64, 56]} inputs([[PERMUTEQUANTIZE]] : tensor<1x1024x1x3584xf16, {order = #NHWC}>) -> tensor<1x1024x64x56xf16, {order = #NHWC}>

    // CHECK:               return [[RESHAPE]] : tensor<1x1024x64x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotMovePermuteQuantizeDueToMergedPermutationChanged
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x2x3584xf16>
func.func @NotMovePermuteQuantizeDueToMergedPermutationChanged(%arg0: tensor<1x1024x2x3584xf16>) -> tensor<1x2048x64x56xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 2048, 64, 56]} : tensor<1x1024x2x3584xf16> -> tensor<1x2048x64x56xf16>
    %1 = IE.PermuteQuantize(%0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2048x64x56xf16> -> tensor<1x2048x64x56xf16, {order = #NHWC}>

    return %1 : tensor<1x2048x64x56xf16, {order = #NHWC}>

    // CHECK:               [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 2048, 64, 56]} : tensor<1x1024x2x3584xf16> -> tensor<1x2048x64x56xf16>
    // CHECK:               [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[RESHAPE]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2048x64x56xf16> -> tensor<1x2048x64x56xf16, {order = #NHWC}>

    // CHECK:               return [[PERMUTEQUANTIZE]] : tensor<1x2048x64x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotMovePermuteQuantizeDueToUnalignedWidth
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x2x8xf16>
func.func @NotMovePermuteQuantizeDueToUnalignedWidth(%arg0: tensor<1x1024x2x8xf16>) -> tensor<1x1024x1x16xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1024, 1, 16]} : tensor<1x1024x2x8xf16> -> tensor<1x1024x1x16xf16>
    %1 = IE.PermuteQuantize(%0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1024x1x16xf16> -> tensor<1x1024x1x16xf16, {order = #NHWC}>

    return %1 : tensor<1x1024x1x16xf16, {order = #NHWC}>

    // CHECK:               [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 1024, 1, 16]} : tensor<1x1024x2x8xf16> -> tensor<1x1024x1x16xf16>
    // CHECK:               [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[RESHAPE]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1024x1x16xf16> -> tensor<1x1024x1x16xf16, {order = #NHWC}>

    // CHECK:               return [[PERMUTEQUANTIZE]] : tensor<1x1024x1x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @NotPropagatePermuteQuantizeIfChangesElemType(%arg0: tensor<1x4096x1x1280xf16>) -> tensor<1x1280x4096x1x!qElemType, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x4096x1x1280xf16> -> tensor<1x1280x1x4096xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 4096, 1]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x4096x1xf16>
    %2 = IE.PermuteQuantize(%1) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1x!qElemType, {order = #NHWC}>

    return %2 : tensor<1x1280x4096x1x!qElemType, {order = #NHWC}>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x4096x1x1280xf16> -> tensor<1x1280x1x4096xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1280, 4096, 1]} : tensor<1x1280x1x4096xf16> -> tensor<1x1280x4096x1xf16>
    // CHECK:               [[PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize([[RESHAPE]]) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1x!qElemType, {order = #NHWC}>

    // CHECK:               return [[PERMUTEQUANTIZE]] : tensor<1x1280x4096x1x!qElemType, {order = #NHWC}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NotPropagatePermuteQuantizeForInvalidRank(%arg0: tensor<4096x1x1280xf16>) -> tensor<1x1280x4096x1xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #CHW, mem_perm = #map} : tensor<4096x1x1280xf16> -> tensor<1280x4096x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1280, 4096, 1]} : tensor<1280x4096x1xf16> -> tensor<1x1280x4096x1xf16>
    %2 = IE.PermuteQuantize(%1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    return %2 : tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #CHW, mem_perm = #map} : tensor<4096x1x1280xf16> -> tensor<1280x4096x1xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1280, 4096, 1]} : tensor<1280x4096x1xf16> -> tensor<1x1280x4096x1xf16>
    // CHECK:               [[PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize([[RESHAPE]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1xf16, {order = #NHWC}>

    // CHECK:               return [[PERMUTEQUANTIZE]] : tensor<1x1280x4096x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MoveThroughMVN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x320x64x64xf16, {order = #NHWC}>
func.func @MoveThroughMVN(%arg0: tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x320x4096x1xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x320x64x64xf16, {order = #NHWC}>, tensor<1x320x64x64xf16, {order = #NHWC}> -> tensor<1x320x64x64xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #map, mem_perm = #map} : tensor<1x320x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x320xf16, {order = #map}>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4096, 320, 1]} : tensor<1x64x64x320xf16, {order = #map}> -> tensor<1x4096x320x1xf16, {order = #NHWC}>
    %3 = IE.MVN(%2) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x4096x320x1xf16, {order = #NHWC}> -> tensor<1x4096x320x1xf16, {order = #NHWC}>
    %4 = IE.MemPermute(%3) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} : tensor<1x4096x320x1xf16, {order = #NHWC}> -> tensor<1x320x4096x1xf16, {order = #NHWC}>
    return %4 : tensor<1x320x4096x1xf16, {order = #NHWC}>

    // CHECK:               [[ADD:%.*]] = IE.Add([[INPUT]], [[INPUT]])  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320x64x64xf16, {order = #NHWC}>, tensor<1x320x64x64xf16, {order = #NHWC}> -> tensor<1x320x64x64xf16, {order = #NHWC}>
    // CHECK:               [[PERMUTECAST0:%.*]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x320x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x320xf16>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[PERMUTECAST0]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4096, 1, 320]} : tensor<1x64x64x320xf16> -> tensor<1x4096x1x320xf16>
    // CHECK:               [[PERMUTECAST1:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x4096x1x320xf16> -> tensor<1x4096x320x1xf16, {order = #NCWH}>
    // CHECK:               [[MVN:%.*]] = IE.MVN([[PERMUTECAST1]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x4096x320x1xf16, {order = #NCWH}> -> tensor<1x4096x320x1xf16, {order = #NCWH}>
    // CHECK:               [[PERMUTECAST2:%.*]] = IE.PermuteCast([[MVN]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4096x320x1xf16, {order = #NCWH}> -> tensor<1x320x4096x1xf16, {order = #NHWC}>
    // CHECK:               return [[PERMUTECAST2]] : tensor<1x320x4096x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
!qElemType = !quant.uniform<u8:f16, 0.29441935221354165:128>
!qElemType1 = !quant.uniform<u8:f16, 0.14720967610677083:128>

// CHECK-LABEL: @MoveThroughQuantizeCast
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1536x150x1x!qElemType, {order = #NHWC}>
func.func @MoveThroughQuantizeCast(%arg0: tensor<1x1536x150x1x!qElemType, {order = #NHWC}>) -> tensor<1x1536x150x1x!qElemType1, {order = #NHWC}> {
  %0 = IE.MemPermute(%arg0) {dst_order = #HCNW, mem_perm = #NWCH} : tensor<1x1536x150x1x!qElemType, {order = #NHWC}> -> tensor<150x1536x1x1x!qElemType, {order = #HCNW}>
  %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<150x1536x1x1x!qElemType, {order = #HCNW}> -> tensor<150x1536x1x1x!qElemType1, {order = #HCNW}>
  %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<150x1536x1x1x!qElemType1, {order = #HCNW}> -> tensor<1x1536x150x1x!qElemType1, {order = #NHWC}>
  return %2 : tensor<1x1536x150x1x!qElemType1, {order = #NHWC}>

  // CHECK:  [[PERMUTECAST0:%.*]] = IE.PermuteCast([[INPUT]]) {dst_order = #map, mem_perm = #NCHW} : tensor<1x1536x150x1x!qElemType, {order = #NHWC}> -> tensor<150x1536x1x1x!qElemType, {order = #map}>
  // CHECK:  [[QUANTIZECAST:%.*]] = IE.QuantizeCast([[PERMUTECAST0]]) {dstElemType = !qElemType1} : tensor<150x1536x1x1x!qElemType, {order = #map}> -> tensor<150x1536x1x1x!qElemType1, {order = #map}>
  // CHECK:  [[PERMUTECAST1:%.*]] = IE.PermuteCast([[QUANTIZECAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<150x1536x1x1x!qElemType1, {order = #map}> -> tensor<1x1536x150x1x!qElemType1, {order = #NHWC}>
  // CHECK:  return [[PERMUTECAST1]]  : tensor<1x1536x150x1x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
!qElemType = !quant.uniform<u8:f16:1, {1.0:128, 0.5:64}>
!qElemType1 = !quant.uniform<u8:f16:1, {0.4:128, 0.8:128}>

// CHECK-LABEL: @DoNotMoveThroughQuantizeCastPerAxis
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x2x150x1x!qElemType, {order = #NHWC}>
func.func @DoNotMoveThroughQuantizeCastPerAxis(%arg0: tensor<1x2x150x1x!qElemType, {order = #NHWC}>) -> tensor<1x2x150x1x!qElemType1, {order = #NHWC}> {
  %0 = IE.MemPermute(%arg0) {dst_order = #HCNW, mem_perm = #NWCH} : tensor<1x2x150x1x!qElemType, {order = #NHWC}> -> tensor<150x2x1x1x!qElemType, {order = #HCNW}>
  %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<150x2x1x1x!qElemType, {order = #HCNW}> -> tensor<150x2x1x1x!qElemType1, {order = #HCNW}>
  %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<150x2x1x1x!qElemType1, {order = #HCNW}> -> tensor<1x2x150x1x!qElemType1, {order = #NHWC}>
  return %2 : tensor<1x2x150x1x!qElemType1, {order = #NHWC}>

  // CHECK:  [[MEMPERMUTE0:%.*]] = IE.MemPermute([[INPUT]]) {dst_order = #map, mem_perm = #NWCH} : tensor<1x2x150x1x!qElemType, {order = #NHWC}> -> tensor<150x2x1x1x!qElemType, {order = #map}>
  // CHECK:  [[QUANTIZECAST:%.*]] = IE.QuantizeCast([[MEMPERMUTE0]]) {dstElemType = !qElemType1} : tensor<150x2x1x1x!qElemType, {order = #map}> -> tensor<150x2x1x1x!qElemType1, {order = #map}>
  // CHECK:  [[MEMPERMUTE1:%.*]] = IE.MemPermute([[QUANTIZECAST]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<150x2x1x1x!qElemType1, {order = #map}> -> tensor<1x2x150x1x!qElemType1, {order = #NHWC}>
  // CHECK:  return [[MEMPERMUTE1]] : tensor<1x2x150x1x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
!qElemType = !quant.uniform<u8:f16, 0.29441935221354165:128>
!qElemType1 = !quant.uniform<u8:f16, 0.14720967610677083:128>

// CHECK-LABEL: @DoNotMoveThroughQuantizeCastBlockArgument
// CHECK-SAME:    [[INPUT:%.+]]: tensor<150x2x1x1x!qElemType, {order = #map}>
func.func @DoNotMoveThroughQuantizeCastBlockArgument(%arg0: tensor<150x2x1x1x!qElemType, {order = #map}>) -> tensor<1x2x150x1x!qElemType1, {order = #NHWC}> {
  %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType1} : tensor<150x2x1x1x!qElemType, {order = #map}> -> tensor<150x2x1x1x!qElemType1, {order = #map}>
  %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<150x2x1x1x!qElemType1, {order = #map}> -> tensor<1x2x150x1x!qElemType1, {order = #NHWC}>
  return %1 : tensor<1x2x150x1x!qElemType1, {order = #NHWC}>

  // CHECK:  [[QUANTIZECAST:%.*]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !qElemType1} : tensor<150x2x1x1x!qElemType, {order = #map}> -> tensor<150x2x1x1x!qElemType1, {order = #map}>
  // CHECK:  [[MEMPERMUTE1:%.*]] = IE.MemPermute([[QUANTIZECAST]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<150x2x1x1x!qElemType1, {order = #map}> -> tensor<1x2x150x1x!qElemType1, {order = #NHWC}>
  // CHECK:  return [[MEMPERMUTE1]] : tensor<1x2x150x1x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
!qElemType = !quant.uniform<u4:f16, 1.1534313725490195>
!qElemType1 = !quant.uniform<u4:f16, 0.14720967610677083:128>

// CHECK-LABEL: @DoNotMoveThroughQuantizeCastSubbyteQuantisation
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x2x150x1x!qElemType, {order = #NHWC}>
func.func @DoNotMoveThroughQuantizeCastSubbyteQuantisation(%arg0: tensor<1x2x150x1x!qElemType, {order = #NHWC}>) -> tensor<1x2x150x1x!qElemType1, {order = #NHWC}> {
  %0 = IE.MemPermute(%arg0) {dst_order = #HCNW, mem_perm = #NWCH} : tensor<1x2x150x1x!qElemType, {order = #NHWC}> -> tensor<150x2x1x1x!qElemType, {order = #HCNW}>
  %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<150x2x1x1x!qElemType, {order = #HCNW}> -> tensor<150x2x1x1x!qElemType1, {order = #HCNW}>
  %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<150x2x1x1x!qElemType1, {order = #HCNW}> -> tensor<1x2x150x1x!qElemType1, {order = #NHWC}>
  return %2 : tensor<1x2x150x1x!qElemType1, {order = #NHWC}>

  // CHECK:  [[MEMPERMUTE0:%.*]] = IE.MemPermute([[INPUT]]) {dst_order = #map, mem_perm = #NWCH} : tensor<1x2x150x1x!qElemType, {order = #NHWC}> -> tensor<150x2x1x1x!qElemType, {order = #map}>
  // CHECK:  [[QUANTIZECAST:%.*]] = IE.QuantizeCast([[MEMPERMUTE0]]) {dstElemType = !qElemType1} : tensor<150x2x1x1x!qElemType, {order = #map}> -> tensor<150x2x1x1x!qElemType1, {order = #map}>
  // CHECK:  [[MEMPERMUTE1:%.*]] = IE.MemPermute([[QUANTIZECAST]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<150x2x1x1x!qElemType1, {order = #map}> -> tensor<1x2x150x1x!qElemType1, {order = #NHWC}>
  // CHECK:  return [[MEMPERMUTE1]] : tensor<1x2x150x1x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:254>:f16:1, {0.002174197219488189:127,0.0013370063361220473:127,8.1604484498031492E-4:127,7.6702448326771658E-4:127}>

!qElemType1 = !quant.uniform<i8<-127:127>:f16:1, {0.002174197219488189,0.0013370063361220473,8.1604484498031492E-4,7.6702448326771658E-4}>

!qElemType2 = !quant.uniform<u8<0:254>:f16:0, {0.002174197219488189:127,0.0013370063361220473:127,8.1604484498031492E-4:127,7.6702448326771658E-4:127}>

func.func @PropagateMemPermuteThroughAffineReshapeChangesQuantAxis(%arg0: tensor<1x4x48x25x!qElemType>) -> tensor<4x48x5x5xf16, {order = #NHWC}> {
    %cst_7 = const.Declare tensor<1x4x48x25x!qElemType> = dense<1.000000e+00> : tensor<2x2x48x5x5xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 4, 48, 25]>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
	%8 = IE.Add(%arg0, %cst_7) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x48x25x!qElemType>, tensor<1x4x48x25x!qElemType> -> tensor<1x4x48x25x!qElemType>
    %9 = IE.AffineReshape(%8) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [4, 48, 5, 5]} : tensor<1x4x48x25x!qElemType> -> tensor<4x48x5x5x!qElemType2>
    %10 = IE.MemPermute(%9) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x48x5x5x!qElemType2> -> tensor<4x48x5x5x!qElemType2, {order = #NHWC}>
    %11 = IE.Dequantize(%10) {dstElemType = f16} : tensor<4x48x5x5x!qElemType2, {order = #NHWC}> -> tensor<4x48x5x5xf16, {order = #NHWC}>
    return %11 : tensor<4x48x5x5xf16, {order = #NHWC}>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x4x48x25x!qElemType> = dense<1.000000e+00> : tensor<2x2x48x5x5xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 4, 48, 25]>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    // CHECK:               [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48x25x!qElemType>, tensor<1x4x48x25x!qElemType> -> tensor<1x4x48x25x!qElemType>
    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute([[ADD]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x4x48x25x!qElemType> -> tensor<1x4x25x48x!qElemType>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[MEMPERMUTE]])
    // CHECK-SAME{LITERAL}:       {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [4, 5, 5, 48]} : tensor<1x4x25x48x!qElemType> -> tensor<4x5x5x48x!qElemType2>
    // CHECK:               [[PERMUTECAST:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<4x5x5x48x!qElemType2> -> tensor<4x48x5x5x!qElemType2, {order = #NHWC}>
    // CHECK:               [[DEQUANT:%.*]] = IE.Dequantize([[PERMUTECAST]]) {dstElemType = f16} : tensor<4x48x5x5x!qElemType2, {order = #NHWC}> -> tensor<4x48x5x5xf16, {order = #NHWC}>
    // CHECK:               return [[DEQUANT]] : tensor<4x48x5x5xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MoveThroughGelu
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x2x512x1500xf16>
func.func @MoveThroughGelu(%arg0: tensor<1x2x512x1500xf16>) -> tensor<1x512x1500x2xf16> {
    %0 = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x2x512x1500xf16>, tensor<1x2x512x1500xf16> -> tensor<1x2x512x1500xf16>
    %1 = IE.Gelu(%0) : tensor<1x2x512x1500xf16> -> tensor<1x2x512x1500xf16>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x2x512x1500xf16> -> tensor<1x512x1500x2xf16>
    return %2 : tensor<1x512x1500x2xf16>

    // CHECK:       [[ADD:%.*]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x512x1500xf16>, tensor<1x2x512x1500xf16> -> tensor<1x2x512x1500xf16>
    // CHECK:       [[MEMPERMUTE:%.*]] = IE.MemPermute([[ADD]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x512x1500xf16> -> tensor<1x2x512x1500xf16, {order = #NHWC}>
    // CHECK:       [[GELU:%.*]] = IE.Gelu([[MEMPERMUTE]]) : tensor<1x2x512x1500xf16, {order = #NHWC}> -> tensor<1x2x512x1500xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTCAST:%.*]] = IE.PermuteCast([[GELU]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x2x512x1500xf16, {order = #NHWC}> -> tensor<1x512x1500x2xf16>
    // CHECK:       return  [[PERMUTCAST]] : tensor<1x512x1500x2xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: @MoveThroughMVNWithSupportLayout
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x75x48x2xf16>
func.func @MoveThroughMVNWithSupportLayout(%arg0: tensor<1x75x48x2xf16>) -> tensor<1x48x75x2xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x75x48x2xf16>, tensor<1x75x48x2xf16> -> tensor<1x75x48x2xf16>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x75x48x2xf16> -> tensor<1x75x48x2xf16>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x75x48x2xf16> -> tensor<1x48x75x2xf16, {order = #NHWC}>
    return %2 : tensor<1x48x75x2xf16, {order = #NHWC}>

    // CHECK:       [[ADD:%.*]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x75x48x2xf16>, tensor<1x75x48x2xf16> -> tensor<1x75x48x2xf16>
    // CHECK:       [[MEMPERMUTE:%.*]] = IE.MemPermute([[ADD]]) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x75x48x2xf16> -> tensor<1x75x48x2xf16, {order = #NCWH}>
    // CHECK:       [[MVN:%.*]] = IE.MVN([[MEMPERMUTE]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x75x48x2xf16, {order = #NCWH}> -> tensor<1x75x48x2xf16, {order = #NCWH}>
    // CHECK:       [[PERMUTECAST:%.*]] = IE.PermuteCast([[MVN]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x75x48x2xf16, {order = #NCWH}> -> tensor<1x48x75x2xf16, {order = #NHWC}>
    // CHECK:       return  [[PERMUTECAST]] : tensor<1x48x75x2xf16, {order = #NHWC}>
}


// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
func.func @NotMoveThroughMVNUnSupportLayout(%arg0: tensor<1x75x48x2xf16, {order = #NWHC}>) -> tensor<1x2x75x48xf16> {
    %0 = IE.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x75x48x2xf16, {order = #NWHC}> -> tensor<1x75x48x2xf16, {order = #NWHC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x75x48x2xf16, {order = #NWHC}> -> tensor<1x2x75x48xf16>
    return %1 : tensor<1x2x75x48xf16>

    // CHECK:       [[MVN:%.*]] = IE.MVN(%arg0)
    // CHECK:       [[MEMPERMUTE:%.*]] = IE.MemPermute([[MVN]])
    // CHECK:       return  [[MEMPERMUTE]] : tensor<1x2x75x48xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @MoveThroughConcat(%arg0: tensor<1x8x64x447xf16>, %arg1: tensor<1x8x64x447xf16>) -> tensor<1x8x64x894xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x64x447xf16> -> tensor<1x8x447x64xf16>
    %1 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x64x447xf16> -> tensor<1x8x447x64xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 447, 0]]} : tensor<1x8x447x64xf16>, tensor<1x8x447x64xf16> -> tensor<1x8x894x64xf16>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x894x64xf16> -> tensor<1x8x64x894xf16>

    return %3 : tensor<1x8x64x894xf16>

    // CHECK:       [[PERMUTCAST_1:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x8x64x447xf16> -> tensor<1x8x447x64xf16, {order = #NCWH}>
    // CHECK:       [[PERMUTCAST_2:%.*]] = IE.PermuteCast(%arg1) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x8x64x447xf16> -> tensor<1x8x447x64xf16, {order = #NCWH}>

    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[PERMUTCAST_1]], [[PERMUTCAST_2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 447, 0]]} : tensor<1x8x447x64xf16, {order = #NCWH}>, tensor<1x8x447x64xf16, {order = #NCWH}> -> tensor<1x8x894x64xf16, {order = #NCWH}>

    // CHECK:       [[PERMUTCAST_OUT:%.*]] = IE.PermuteCast([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x8x894x64xf16, {order = #NCWH}> -> tensor<1x8x64x894xf16>

    // CHECK:       return  [[PERMUTCAST_OUT]] : tensor<1x8x64x894xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @MoveThroughConcat_OutputLayoutIsChanged(%arg0: tensor<1x8x64x447xf16>, %arg1: tensor<1x8x64x447xf16>) -> tensor<1x8x64x894xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x8x64x447xf16> -> tensor<1x64x8x447xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x8x64x447xf16> -> tensor<1x64x8x447xf16, {order = #NHWC}>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 447]]} : tensor<1x64x8x447xf16, {order = #NHWC}>, tensor<1x64x8x447xf16, {order = #NHWC}> -> tensor<1x64x8x894xf16, {order = #NHWC}>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x64x8x894xf16, {order = #NHWC}> -> tensor<1x8x64x894xf16>

    return %3 : tensor<1x8x64x894xf16>

    // CHECK:       [[PERMUTCAST_1:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x8x64x447xf16> -> tensor<1x64x8x447xf16, {order = #NHCW}>
    // CHECK:       [[PERMUTCAST_2:%.*]] = IE.PermuteCast(%arg1) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x8x64x447xf16> -> tensor<1x64x8x447xf16, {order = #NHCW}>

    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[PERMUTCAST_1]], [[PERMUTCAST_2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 447]]} : tensor<1x64x8x447xf16, {order = #NHCW}>, tensor<1x64x8x447xf16, {order = #NHCW}> -> tensor<1x64x8x894xf16, {order = #NHCW}>

    // CHECK:       [[PERMUTCAST_OUT:%.*]] = IE.PermuteCast([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x8x894xf16, {order = #NHCW}> -> tensor<1x8x64x894xf16>

    // CHECK:       return  [[PERMUTCAST_OUT]] : tensor<1x8x64x894xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @MoveThroughConcat_NotAllInputsHaveMemPermute(%arg0: tensor<1x8x64x447xf16>, %arg1: tensor<1x512x1x1xf16, {order = #NHWC}>) -> tensor<1x8x64x448xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x64x447xf16> -> tensor<1x8x447x64xf16>
    %1 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16>
    %2 = IE.Reshape(%1) {shape_value = [1, 8, 1, 64]} : tensor<1x512x1x1xf16> -> tensor<1x8x1x64xf16>
    %3 = IE.Concat(%0, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 447, 0]]} : tensor<1x8x447x64xf16>, tensor<1x8x1x64xf16> -> tensor<1x8x448x64xf16>
    %4 = IE.MemPermute(%3) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x448x64xf16> -> tensor<1x8x64x448xf16>

    return %4 : tensor<1x8x64x448xf16>

    // CHECK:       [[ORIG_PERMUTCAST:%.*]] = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16>
    // CHECK:       [[RESHAPE:%.*]] = IE.Reshape([[ORIG_PERMUTCAST]]) {shape_value = [1, 8, 1, 64]} : tensor<1x512x1x1xf16> -> tensor<1x8x1x64xf16>

    // CHECK:       [[NEW_INPUT_PERMUTCAST_1:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x8x64x447xf16> -> tensor<1x8x447x64xf16, {order = #NCWH}>
    // CHECK:       [[NEW_INPUT_PERMUTCAST_2:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x8x1x64xf16> -> tensor<1x8x1x64xf16, {order = #NCWH}>

    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[NEW_INPUT_PERMUTCAST_1]], [[NEW_INPUT_PERMUTCAST_2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 447, 0]]} : tensor<1x8x447x64xf16, {order = #NCWH}>, tensor<1x8x1x64xf16, {order = #NCWH}> -> tensor<1x8x448x64xf16, {order = #NCWH}>

    // CHECK:       [[NEW_OUTPUT_PERMUTCAST:%.*]] = IE.PermuteCast([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x8x448x64xf16, {order = #NCWH}> -> tensor<1x8x64x448xf16>

    // CHECK:       return  [[NEW_OUTPUT_PERMUTCAST]] : tensor<1x8x64x448xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @MoveThroughConcat_NotAllInputsHaveMemPermute_OutputLayoutIsChanged(%arg0: tensor<1x8x64x447xf16>, %arg1: tensor<1x512x1x1xf16, {order = #NHWC}>) -> tensor<1x448x8x64xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x64x447xf16> -> tensor<1x8x447x64xf16>
    %1 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16>
    %2 = IE.Reshape(%1) {shape_value = [1, 8, 1, 64]} : tensor<1x512x1x1xf16> -> tensor<1x8x1x64xf16>
    %3 = IE.Concat(%0, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 447, 0]]} : tensor<1x8x447x64xf16>, tensor<1x8x1x64xf16> -> tensor<1x8x448x64xf16>
    %4 = IE.MemPermute(%3) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x8x448x64xf16> -> tensor<1x448x8x64xf16, {order = #NHWC}>

    return %4 : tensor<1x448x8x64xf16, {order = #NHWC}>

    // CHECK:       [[ORIG_PERMUTCAST:%.*]] = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x512x1x1xf16, {order = #NHWC}> -> tensor<1x512x1x1xf16>
    // CHECK:       [[RESHAPE:%.*]] = IE.Reshape([[ORIG_PERMUTCAST]]) {shape_value = [1, 8, 1, 64]} : tensor<1x512x1x1xf16> -> tensor<1x8x1x64xf16>

    // CHECK:       [[NEW_INPUT_PERMUTCAST_1:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x8x64x447xf16> -> tensor<1x8x447x64xf16, {order = #NCWH}>
    // CHECK:       [[NEW_INPUT_PERMUTCAST_2:%.*]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x8x1x64xf16> -> tensor<1x8x1x64xf16, {order = #NCWH}>

    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[NEW_INPUT_PERMUTCAST_1]], [[NEW_INPUT_PERMUTCAST_2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 447, 0]]} : tensor<1x8x447x64xf16, {order = #NCWH}>, tensor<1x8x1x64xf16, {order = #NCWH}> -> tensor<1x8x448x64xf16, {order = #NCWH}>

    // CHECK:       [[NEW_OUTPUT_PERMUTCAST:%.*]] = IE.PermuteCast([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x8x448x64xf16, {order = #NCWH}> -> tensor<1x448x8x64xf16, {order = #NHWC}>

    // CHECK:       return  [[NEW_OUTPUT_PERMUTCAST]] : tensor<1x448x8x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MoveThroughConcat_IfAnyInputBranchIsBenificial
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4x512x36x36xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x2048x36x36xf16>
func.func @MoveThroughConcat_IfAnyInputBranchIsBenificial(%arg0: tensor<4x512x36x36xf16, {order = #NHWC}>, %arg1: tensor<1x2048x36x36xf16>) -> (tensor<4x512x1296x1xf16>, tensor<4x1024x36x36xf16, {order = #NHWC}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x512x36x36xf16, {order = #NHWC}> -> tensor<4x512x36x36xf16>

    %2 = IE.ShapeCast {shape = [4, 512, 36, 36]} inputs(%arg1 : tensor<1x2048x36x36xf16>) -> tensor<4x512x36x36xf16>
    %3 = IE.ShapeCast {shape = [4, 512, 1296, 1]} inputs(%2 : tensor<4x512x36x36xf16>) -> tensor<4x512x1296x1xf16>

    %4 = IE.Concat(%0, %2) {static_offsets = [[0, 0, 0, 0], [0, 512, 0, 0]]} : tensor<4x512x36x36xf16>, tensor<4x512x36x36xf16> -> tensor<4x1024x36x36xf16>
    %5 = IE.MemPermute(%4) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x1024x36x36xf16> -> tensor<4x1024x36x36xf16, {order = #NHWC}>

    return %3, %5 : tensor<4x512x1296x1xf16>, tensor<4x1024x36x36xf16, {order = #NHWC}>

    // CHECK:    [[RESHAPE0:%.+]] = IE.ShapeCast {shape = [4, 512, 36, 36]} inputs([[INPUT_1]] : tensor<1x2048x36x36xf16>) -> tensor<4x512x36x36xf16>
    // CHECK:    [[RESHAPE1:%.+]]  = IE.ShapeCast {shape = [4, 512, 1296, 1]} inputs([[RESHAPE0]] : tensor<4x512x36x36xf16>) -> tensor<4x512x1296x1xf16>
    // CHECK:    [[PERMUTE:%.+]] = IE.MemPermute([[RESHAPE0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x512x36x36xf16> -> tensor<4x512x36x36xf16, {order = #NHWC}>
    // CHECK:    [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[PERMUTE]])
    // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 512, 0, 0]]} : tensor<4x512x36x36xf16, {order = #NHWC}>, tensor<4x512x36x36xf16, {order = #NHWC}> -> tensor<4x1024x36x36xf16, {order = #NHWC}>

    // CHECK:    return [[RESHAPE1]], [[CONCAT]] : tensor<4x512x1296x1xf16>, tensor<4x1024x36x36xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MoveThroughConcat_IfNoneOfInputBranchIsBenificial
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x2048x36x36xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x2048x36x36xf16>
func.func @MoveThroughConcat_IfNoneOfInputBranchIsBenificial(%arg0: tensor<1x2048x36x36xf16>, %arg1: tensor<1x2048x36x36xf16>) -> tensor<4x1024x36x36xf16, {order = #NHWC}> {
    %2 = IE.ShapeCast {shape = [4, 512, 36, 36]} inputs(%arg0 : tensor<1x2048x36x36xf16>) -> tensor<4x512x36x36xf16>
    %3 = IE.ShapeCast {shape = [4, 512, 36, 36]} inputs(%arg1 : tensor<1x2048x36x36xf16>) -> tensor<4x512x36x36xf16>

    %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 512, 0, 0]]} : tensor<4x512x36x36xf16>, tensor<4x512x36x36xf16> -> tensor<4x1024x36x36xf16>
    %5 = IE.MemPermute(%4) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x1024x36x36xf16> -> tensor<4x1024x36x36xf16, {order = #NHWC}>

    return %5 : tensor<4x1024x36x36xf16, {order = #NHWC}>

    // CHECK:    [[RESHAPE0:%.+]] = IE.ShapeCast {shape = [4, 512, 36, 36]} inputs([[INPUT_0]] : tensor<1x2048x36x36xf16>) -> tensor<4x512x36x36xf16>
    // CHECK:    [[RESHAPE1:%.+]] = IE.ShapeCast {shape = [4, 512, 36, 36]} inputs([[INPUT_1]] : tensor<1x2048x36x36xf16>) -> tensor<4x512x36x36xf16>
    // CHECK:    [[PERMUTE0:%.+]] = IE.MemPermute([[RESHAPE0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x512x36x36xf16> -> tensor<4x512x36x36xf16, {order = #NHWC}>
    // CHECK:    [[PERMUTE1:%.+]] = IE.MemPermute([[RESHAPE1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x512x36x36xf16> -> tensor<4x512x36x36xf16, {order = #NHWC}>
    // CHECK:    [[CONCAT:%.+]] = IE.Concat([[PERMUTE0]], [[PERMUTE1]])
    // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 512, 0, 0]]} : tensor<4x512x36x36xf16, {order = #NHWC}>, tensor<4x512x36x36xf16, {order = #NHWC}> -> tensor<4x1024x36x36xf16, {order = #NHWC}>

    // CHECK:    return [[CONCAT]] : tensor<4x1024x36x36xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @NotMoveThroughConcat_NotBeneficial(%arg0: tensor<1x8x64x447xf16>, %arg1: tensor<1x8x447x64xf16>) -> tensor<1x8x64x894xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x64x447xf16> -> tensor<1x8x447x64xf16>
    %1 = IE.Concat(%0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 447, 0]]} : tensor<1x8x447x64xf16>, tensor<1x8x447x64xf16> -> tensor<1x8x894x64xf16>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x894x64xf16> -> tensor<1x8x64x894xf16>

    return %2 : tensor<1x8x64x894xf16>

    // CHECK:       [[PERMUTE_IN1:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x64x447xf16> -> tensor<1x8x447x64xf16>

    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[PERMUTE_IN1]], %arg1) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 447, 0]]} : tensor<1x8x447x64xf16>, tensor<1x8x447x64xf16> -> tensor<1x8x894x64xf16>

    // CHECK:       [[PERMUTE_OUT:%.*]] = IE.MemPermute([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x894x64xf16> -> tensor<1x8x64x894xf16>

    // CHECK:       return  [[PERMUTE_OUT]] : tensor<1x8x64x894xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: @MoveThroughSlice
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x64x48x32xf16>
func.func @MoveThroughSlice(%arg0: tensor<1x64x48x32xf16>) -> tensor<1x48x16x32xf16, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] }
            : tensor<1x64x48x32xf16> -> tensor<1x64x48x32xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 16, 48, 32] : tensor<1x64x48x32xf16> to tensor<1x16x48x32xf16>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x16x48x32xf16> -> tensor<1x48x16x32xf16, {order = #NHWC}>
    return %2 : tensor<1x48x16x32xf16, {order = #NHWC}>

    // CHECK:       [[POOL:%.*]] = IE.AvgPool([[INPUT]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x48x32xf16> -> tensor<1x64x48x32xf16>
    // CHECK:       [[PERMUTE:%.*]] = IE.MemPermute([[POOL]]) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x64x48x32xf16> -> tensor<1x64x48x32xf16, {order = #NCWH}>
    // CHECK:       [[SLICE:%.*]] = IE.Slice [[PERMUTE]] [0, 0, 0, 0] [1, 16, 48, 32] : tensor<1x64x48x32xf16, {order = #NCWH}> to tensor<1x16x48x32xf16, {order = #NCWH}>
    // CHECK:       [[PERMUTECAST:%.*]] = IE.PermuteCast([[SLICE]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x48x32xf16, {order = #NCWH}> -> tensor<1x48x16x32xf16, {order = #NHWC}>
    // CHECK:       return  [[PERMUTECAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MoveThroughSliceWithConv
// CHECK-SAME:      [[INPUT:%arg[0-9]]]:  tensor<1x64x4x4xf16, {order = #NHWC}>
func.func @MoveThroughSliceWithConv(%arg0: tensor<1x64x4x4xf16, {order = #NHWC}>) -> tensor<1x56x4x4xf16> {
    %cst = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16, {order = #NHWC}>
    %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x64x4x4xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x4x4xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 56, 4, 4] : tensor<1x64x4x4xf16, {order = #NHWC}> to tensor<1x56x4x4xf16, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x56x4x4xf16, {order = #NHWC}> -> tensor<1x56x4x4xf16>

    return %2 : tensor<1x56x4x4xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:          dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x64x4x4xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x4x4xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE:%.+]] = IE.MemPermute([[CONV]]) {
    // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x64x4x4xf16, {order = #NHWC}> -> tensor<1x64x4x4xf16>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERMUTE]] [0, 0, 0, 0] [1, 56, 4, 4] : tensor<1x64x4x4xf16> to tensor<1x56x4x4xf16>

    // CHECK:       return [[SLICE]] : tensor<1x56x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: @NotMoveThroughSliceOpWithOutNCE
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x64x48x32xf16>
func.func @NotMoveThroughSliceOpWithOutNCE(%arg0: tensor<1x64x48x32xf16>) -> tensor<1x48x16x32xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 48, 32] : tensor<1x64x48x32xf16> to tensor<1x16x48x32xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x16x48x32xf16> -> tensor<1x48x16x32xf16, {order = #NHWC}>
    return %1 : tensor<1x48x16x32xf16, {order = #NHWC}>

    // CHECK:       [[SLICE:%.*]] = IE.Slice [[INPUT]]
    // CHECK:       [[PERMUTE:%.*]] = IE.MemPermute([[SLICE]])
    // CHECK:       return  [[PERMUTE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: @NotMoveThroughSliceForMakingSliceDimLower
// CHECK-SAME:    [[INPUT:%.*]]: tensor<1x64x48x32xf16>
func.func @NotMoveThroughSliceForMakingSliceDimLower(%arg0: tensor<1x64x48x32xf16>) -> tensor<1x6x64x32xf16, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) { kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1] }
            : tensor<1x64x48x32xf16> -> tensor<1x64x48x32xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 64, 6, 32] : tensor<1x64x48x32xf16> to tensor<1x64x6x32xf16>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x64x6x32xf16> -> tensor<1x6x64x32xf16, {order = #NHWC}>
    return %2 : tensor<1x6x64x32xf16, {order = #NHWC}>

    // CHECK:       [[POOL:%.*]] = IE.AvgPool([[INPUT]])
    // CHECK:       [[SLICE:%.*]] = IE.Slice [[POOL]]
    // CHECK:       [[PERMUTE:%.*]] = IE.MemPermute([[SLICE]])
    // CHECK:       return  [[PERMUTE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#WCNH = affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @PropagatePermuteQuantizeNewCaseWithSplitAxisPermute
// CHECK-SAME:    [[INPUT:%.+]]: tensor<32x1x157x64xf16>
func.func @PropagatePermuteQuantizeNewCaseWithSplitAxisPermute(%arg0: tensor<32x1x157x64xf16>) -> tensor<1x64x32x157xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #WCNH} : tensor<32x1x157x64xf16> -> tensor<64x1x32x157xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 64, 32, 157]} : tensor<64x1x32x157xf16> -> tensor<1x64x32x157xf16>
    %2 = IE.PermuteQuantize(%1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x64x32x157xf16> -> tensor<1x64x32x157xf16, {order = #NHWC}>

    return %2 : tensor<1x64x32x157xf16, {order = #NHWC}>

    // CHECK:               [[PERMUTECAST1:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #map} : tensor<32x1x157x64xf16> -> tensor<1x32x157x64xf16>
    // CHECK:               [[PERMUTECAST2:%.+]] = IE.PermuteCast([[PERMUTECAST1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x157x64xf16> -> tensor<1x64x32x157xf16, {order = #NHWC}>

    // CHECK:               return [[PERMUTECAST2]] : tensor<1x64x32x157xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MoveThroughShapeCast
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x384x24xf16, {order = #NHWC}>
func.func @MoveThroughShapeCast(%arg0: tensor<1x16x384x24xf16, {order = #NHWC}>) -> tensor<1x1x384x384xf16, {order = #NHWC}> {
    %reshape = IE.ShapeCast {shape = [1, 1, 384, 384]} inputs(%arg0 : tensor<1x16x384x24xf16, {order = #NHWC}>) -> tensor<1x1x384x384xf16, {order = #NHWC}>
    %mem_permute = IE.MemPermute(%reshape) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x1x384x384xf16, {order = #NHWC}> -> tensor<1x1x384x384xf16, {order = #NHWC}>

    return %mem_permute : tensor<1x1x384x384xf16, {order = #NHWC}>

    // CHECK:    [[MEMPERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x384x24xf16, {order = #NHWC}> -> tensor<1x24x16x384xf16>
    // CHECK:    [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 384, 384, 1]} inputs([[MEMPERMUTE]] : tensor<1x24x16x384xf16>) -> tensor<1x384x384x1xf16>
    // CHECK:    [[PERMUTECAST:%.+]] = IE.PermuteCast([[SHAPECAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x384x384x1xf16> -> tensor<1x1x384x384xf16, {order = #NHWC}>
    // CHECK:    return [[PERMUTECAST]] : tensor<1x1x384x384xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @NotMoveThroughShapeCast_AxesSplit
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x24x384xf16, {order = #NHWC}>
func.func @NotMoveThroughShapeCast_AxesSplit(%arg0: tensor<1x16x24x384xf16, {order = #NHWC}>) -> tensor<1x1x384x384xf16, {order = #NHWC}> {
    %reshape = IE.ShapeCast {shape = [1, 1, 384, 384]} inputs(%arg0 : tensor<1x16x24x384xf16, {order = #NHWC}>) -> tensor<1x1x384x384xf16, {order = #NHWC}>
    %mem_permute = IE.MemPermute(%reshape) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x1x384x384xf16, {order = #NHWC}> -> tensor<1x1x384x384xf16, {order = #NHWC}>

    return %mem_permute : tensor<1x1x384x384xf16, {order = #NHWC}>

    // CHECK:    [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 1, 384, 384]} inputs([[INPUT]] : tensor<1x16x24x384xf16, {order = #NHWC}>) -> tensor<1x1x384x384xf16, {order = #NHWC}>
    // CHECK:    [[MEMPERMUTE:%.+]] = IE.MemPermute([[SHAPECAST]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x1x384x384xf16, {order = #NHWC}> -> tensor<1x1x384x384xf16, {order = #NHWC}>
    // CHECK:    return [[MEMPERMUTE]] : tensor<1x1x384x384xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// CHECK-LABEL: @NotMoveThroughShapeCastWithIOShapeNot4D
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x4x96x24xf16, {order = #NCDHW}>
func.func @NotMoveThroughShapeCastWithIOShapeNot4D(%arg0: tensor<1x16x4x96x24xf16, {order = #NCDHW}>) -> tensor<1x384x24x16xf16> {
    %reshape = IE.ShapeCast {shape = [1, 16, 384, 24]} inputs(%arg0 : tensor<1x16x4x96x24xf16, {order = #NCDHW}>) -> tensor<1x16x384x24xf16>
    %mem_permute = IE.MemPermute(%reshape) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x384x24xf16> -> tensor<1x384x24x16xf16>

    return %mem_permute : tensor<1x384x24x16xf16>

    //CHECK:    [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 16, 384, 24]} inputs(%arg0 : tensor<1x16x4x96x24xf16, {order = #NCDHW}>) -> tensor<1x16x384x24xf16>
    //CHECK:    [[MEMPERMUTE:%.+]] = IE.MemPermute([[SHAPECAST]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x384x24xf16> -> tensor<1x384x24x16xf16>
    //CHECK:    return [[MEMPERMUTE]] : tensor<1x384x24x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MovePermuteQuantizeThroughMultiply
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x3584xf16>
func.func @MovePermuteQuantizeThroughMultiply(%arg0: tensor<1x1024x3584xf16>) -> tensor<1x1024x1x3584xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1024, 1, 3584]} : tensor<1x1024x3584xf16> -> tensor<1x1024x1x3584xf16>
    %1 = IE.Multiply(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x1x3584xf16>, tensor<1x1024x1x3584xf16> -> tensor<1x1024x1x3584xf16>
    %2 = IE.PermuteQuantize(%1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1024x1x3584xf16> -> tensor<1x1024x1x3584xf16, {order = #NHWC}>

    return %2 : tensor<1x1024x1x3584xf16, {order = #NHWC}>

    //CHECK:    [[RESHAPE:%.+]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1024, 1, 3584]} : tensor<1x1024x3584xf16> -> tensor<1x1024x1x3584xf16>
    //CHECK:    [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[RESHAPE]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1024x1x3584xf16> -> tensor<1x1024x1x3584xf16, {order = #NHWC}>
    //CHECK:    [[MULTIPLY:%.+]] = IE.Multiply([[PERMUTEQUANTIZE]], [[PERMUTEQUANTIZE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x1x3584xf16, {order = #NHWC}>, tensor<1x1024x1x3584xf16, {order = #NHWC}> -> tensor<1x1024x1x3584xf16, {order = #NHWC}>

    //CHECK:    return [[MULTIPLY]] : tensor<1x1024x1x3584xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotMovePermuteQuantizeWhenMultiplyHasDifferentInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1024x3584xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1024x3584xf16>
func.func @NotMovePermuteQuantizeWhenMultiplyHasDifferentInputs(%arg0: tensor<1x1024x3584xf16>, %arg1: tensor<1x1024x3584xf16>) -> tensor<1x1024x1x3584xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1024, 1, 3584]} : tensor<1x1024x3584xf16> -> tensor<1x1024x1x3584xf16>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1024, 1, 3584]} : tensor<1x1024x3584xf16> -> tensor<1x1024x1x3584xf16>

    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x1x3584xf16>, tensor<1x1024x1x3584xf16> -> tensor<1x1024x1x3584xf16>
    %3 = IE.PermuteQuantize(%2) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1024x1x3584xf16> -> tensor<1x1024x1x3584xf16, {order = #NHWC}>

    return %3 : tensor<1x1024x1x3584xf16, {order = #NHWC}>

    //CHECK:    [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1024, 1, 3584]} : tensor<1x1024x3584xf16> -> tensor<1x1024x1x3584xf16>
    //CHECK:    [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1024, 1, 3584]} : tensor<1x1024x3584xf16> -> tensor<1x1024x1x3584xf16>
    //CHECK:    [[MULTIPLY:%.+]] = IE.Multiply([[RESHAPE0]], [[RESHAPE1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1024x1x3584xf16>, tensor<1x1024x1x3584xf16> -> tensor<1x1024x1x3584xf16>
    //CHECK:    [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[MULTIPLY]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1024x1x3584xf16> -> tensor<1x1024x1x3584xf16, {order = #NHWC}>

    //CHECK:    return [[PERMUTEQUANTIZE]] : tensor<1x1024x1x3584xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PropagateMemPermuteThroughReshape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x128x64x64xf16>
func.func @PropagateMemPermuteThroughReshape(%arg0: tensor<1x128x64x64xf16>) -> tensor<16x8x64x64xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [16, 8, 64, 64]} : tensor<1x128x64x64xf16> -> tensor<16x8x64x64xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<16x8x64x64xf16> -> tensor<16x8x64x64xf16>

    return %1 :  tensor<16x8x64x64xf16>

    //CHECK:    [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x128x64x64xf16> -> tensor<1x128x64x64xf16>
    //CHECK:    [[RESHAPE:%.+]] = IE.Reshape([[MEM_PERMUTE]]) {shape_value = [16, 8, 64, 64]} : tensor<1x128x64x64xf16> -> tensor<16x8x64x64xf16>

    //CHECK:    return [[RESHAPE]] : tensor<16x8x64x64xf16>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotPropagateMemPermuteThroughReshapeDueToPermuteDimsChanged
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x128x64x64xf16>
func.func @NotPropagateMemPermuteThroughReshapeDueToPermuteDimsChanged(%arg0: tensor<1x128x64x64xf16>) -> tensor<16x64x64x8xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [16, 8, 64, 64]} : tensor<1x128x64x64xf16> -> tensor<16x8x64x64xf16>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<16x8x64x64xf16> -> tensor<16x64x64x8xf16>

    return %1 :  tensor<16x64x64x8xf16>

    //CHECK:    [[RESHAPE:%.+]] = IE.Reshape([[INPUT_0]]) {shape_value = [16, 8, 64, 64]} : tensor<1x128x64x64xf16> -> tensor<16x8x64x64xf16>
    //CHECK:    [[MEM_PERMUTE:%.+]] = IE.MemPermute([[RESHAPE]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<16x8x64x64xf16> -> tensor<16x64x64x8xf16>

    //CHECK:    return [[MEM_PERMUTE]] : tensor<16x64x64x8xf16>
}
