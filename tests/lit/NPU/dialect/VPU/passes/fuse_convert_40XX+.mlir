//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --fuse-convert --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseD2sConvert
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x20x40xf16, {order = #NHWC}>
func.func @FuseD2sConvert(%input: tensor<1x16x20x40xf16, {order = #NHWC}>) -> tensor<1x40x80x4xf32> {
    %0 = VPU.DepthToSpace(%input) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
         : tensor<1x16x20x40xf16, {order = #NHWC}> -> tensor<1x4x40x80xf16, {order = #NHWC}>
    %1 = VPU.Convert(%0) {dstElemType = f32} : tensor<1x4x40x80xf16, {order = #NHWC}> -> tensor<1x4x40x80xf32, {order = #NHWC}>
    %2 = VPU.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x4x40x80xf32, {order = #NHWC}> -> tensor<1x40x80x4xf32>
    return %2 : tensor<1x40x80x4xf32>

    // CHECK:       [[D2S_CVT:%.+]] = VPU.DepthToSpace([[INPUT]])
    // CHECK-SAME:                       {block_size = 2 : i64, dstElemType = f32, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
    // CHECK-SAME:                       : tensor<1x16x20x40xf16, {order = #NHWC}> -> tensor<1x4x40x80xf32, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]]  = VPU.PermuteCast([[D2S_CVT]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:                       : tensor<1x4x40x80xf32, {order = #NHWC}> -> tensor<1x40x80x4xf32>

    // CHECK:       return [[OUTPUT]] : tensor<1x40x80x4xf32>
}

// -----

// CHECK-LABEL: @FuseScalesConvertGroupQuantCorrection
// CHECK-SAME:  [[ACT_OUT:%.+]]: tensor<8x96x1x1xf16>
// CHECK-SAME:  [[ACT_IN:%.+]]: tensor<8x1x1x64xf16>
// CHECK-SAME:  [[SCALES:%.+]]: tensor<1x8x1x96xf32>
// CHECK-SAME:  [[ZP:%.+]]: tensor<8x96x1x1xui2>
func.func @FuseScalesConvertGroupQuantCorrection(%actOut: tensor<8x96x1x1xf16>, %actIn: tensor<8x1x1x64xf16>,
                                                 %scales: tensor<1x8x1x96xf32>, %zp: tensor<8x96x1x1xui2>)
        -> tensor<8x96x1x1xf16> {
    %0 = VPU.Convert(%scales) {dstElemType = f16} : tensor<1x8x1x96xf32> -> tensor<1x8x1x96xf16>
    %1 = VPU.AffineReshape(%0) {
        dim_mapping = [[0], [0], [0], [1, 2, 3]],
        shape_value = [8, 96, 1, 1]
    } : tensor<1x8x1x96xf16> -> tensor<8x96x1x1xf16>
    %2 = VPU.GroupQuantCorrection(%actOut, %actIn, %1, %zp)
        : tensor<8x96x1x1xf16>, tensor<8x1x1x64xf16>, tensor<8x96x1x1xf16>, tensor<8x96x1x1xui2>
        -> tensor<8x96x1x1xf16>
    return %2 : tensor<8x96x1x1xf16>

    // CHECK-NOT:   VPU.Convert
    // CHECK:       [[SCALES_RESHAPE:%.+]] = VPU.AffineReshape([[SCALES]])
    // CHECK-SAME:      : tensor<1x8x1x96xf32> -> tensor<8x96x1x1xf32>
    // CHECK:       [[GQC:%.+]] = VPU.GroupQuantCorrection([[ACT_OUT]], [[ACT_IN]], [[SCALES_RESHAPE]], [[ZP]])
    // CHECK-SAME:      : tensor<8x96x1x1xf16>, tensor<8x1x1x64xf16>, tensor<8x96x1x1xf32>, tensor<8x96x1x1xui2>
    // CHECK-SAME:      -> tensor<8x96x1x1xf16>
    // CHECK:       return [[GQC]] : tensor<8x96x1x1xf16>
}

// -----

// CHECK-LABEL: @NotFuseScalesConvertGroupQuantCorrectionNoConvert
// CHECK-SAME:  [[ACT_OUT:%.+]]: tensor<8x96x1x1xf16>
// CHECK-SAME:  [[ACT_IN:%.+]]: tensor<8x1x1x64xf16>
// CHECK-SAME:  [[SCALES:%.+]]: tensor<1x8x1x96xf16>
// CHECK-SAME:  [[ZP:%.+]]: tensor<8x96x1x1xui2>
func.func @NotFuseScalesConvertGroupQuantCorrectionNoConvert(%actOut: tensor<8x96x1x1xf16>, %actIn: tensor<8x1x1x64xf16>,
                                                             %scales: tensor<1x8x1x96xf16>, %zp: tensor<8x96x1x1xui2>)
        -> tensor<8x96x1x1xf16> {
    %0 = VPU.AffineReshape(%scales) {
        dim_mapping = [[0], [0], [0], [1, 2, 3]],
        shape_value = [8, 96, 1, 1]
    } : tensor<1x8x1x96xf16> -> tensor<8x96x1x1xf16>
    %1 = VPU.GroupQuantCorrection(%actOut, %actIn, %0, %zp)
        : tensor<8x96x1x1xf16>, tensor<8x1x1x64xf16>, tensor<8x96x1x1xf16>, tensor<8x96x1x1xui2>
        -> tensor<8x96x1x1xf16>
    return %1 : tensor<8x96x1x1xf16>

    // CHECK:       [[SCALES_RESHAPE:%.+]] = VPU.AffineReshape([[SCALES]])
    // CHECK-SAME:      : tensor<1x8x1x96xf16> -> tensor<8x96x1x1xf16>
    // CHECK:       [[GQC:%.+]] = VPU.GroupQuantCorrection([[ACT_OUT]], [[ACT_IN]], [[SCALES_RESHAPE]], [[ZP]])
    // CHECK-SAME:      : tensor<8x96x1x1xf16>, tensor<8x1x1x64xf16>, tensor<8x96x1x1xf16>, tensor<8x96x1x1xui2>
    // CHECK-SAME:      -> tensor<8x96x1x1xf16>
    // CHECK:       return [[GQC]] : tensor<8x96x1x1xf16>
}
