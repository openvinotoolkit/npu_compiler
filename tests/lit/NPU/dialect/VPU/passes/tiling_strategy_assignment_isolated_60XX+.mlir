//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true enable-is-reduce-supported=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --cmx-additional-stack-frames-reserve-mem --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @Test {

config.Resources 6 of @NCE {
config.MemoryResource 1571840 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @NCEMatMulSOGAndGTile
func.func @NCEMatMulSOGAndGTile(%arg0: tensor<64x8x64x32xf16>, %arg1: tensor<64x8x64x32xf16>) -> tensor<512x1x64x64x1xf16, {order = #GNHWC}> {
  %scale_table = const.Declare tensor<1x64x1x1x1xf32> = dense<1.0> : tensor<1x64x1x1x1xf32>
  %bias_table = const.Declare tensor<1x64x1x1x1xf32> = dense<2.0> : tensor<1x64x1x1x1xf32>
  %0 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg0 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %1 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg1 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<512x64x32x1x1xf16> -> tensor<512x1x32x64x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<512x64x32x1x1xf16> -> tensor<512x64x32x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [512, 1, 32, 16, 4]} : tensor<512x1x32x64x1xf16, {order = #GNHWC}> -> tensor<512x1x32x16x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %scale_table, %bias_table) rawFilterShape [512, 64, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
     strides = [1, 1], tilingStrategy = [2, 1, 1, 1, 1],
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} -> tensor<512x1x64x16x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [512, 1, 64, 64, 1]} : tensor<512x1x64x16x4xf16, {order = #GNHWC}> -> tensor<512x1x64x64x1xf16, {order = #GNHWC}>
  return %8 : tensor<512x1x64x64x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [2, 1, 1, 1, 1]
}
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @Test {

config.Resources 6 of @NCE {
config.MemoryResource 1571840 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @NCEMatMulSOGAndHTile
func.func @NCEMatMulSOGAndHTile(%arg0: tensor<6x1x512x512xf16>, %arg1: tensor<6x1x512x512xf16>) -> tensor<6x1x512x512x1xf16, {order = #GNHWC}> {
  %scale_table = const.Declare tensor<1x512x1x1x1xf32> = dense<1.0> : tensor<1x512x1x1x1xf32>
  %bias_table = const.Declare tensor<1x512x1x1x1xf32> = dense<2.0> : tensor<1x512x1x1x1xf32>
  %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %1 = VPU.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<6x512x512x1x1xf16> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<6x512x512x1x1xf16> -> tensor<6x512x512x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [6, 1, 512, 128, 4]} : tensor<6x1x512x512x1xf16, {order = #GNHWC}> -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %scale_table, %bias_table) rawFilterShape [6, 512, 512, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [6, 1, 512, 512, 1]} : tensor<6x1x512x128x4xf16, {order = #GNHWC}> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  return %8 : tensor<6x1x512x512x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [1, 1, 1, 2, 1]
}
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 1 of @NCE {
        config.MemoryResource 2096128 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitSingleClusterDepthConvHeightByMinNumTiles
    func.func @SplitSingleClusterDepthConvHeightByMinNumTiles(%arg0: tensor<1x16x10272x1xf16, {order = #NHWC}>) -> tensor<1x16x10272x1xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x1x1x1xf16, {order = #NHWC}>, [#const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<0.0> : tensor<16x1x1x1xf32>
        %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0, 1, 1, 0>} -> tensor<1x16x10272x1xf16, {order = #NHWC}>
        return %0 : tensor<1x16x10272x1xf16, {order = #NHWC}>

        // CHECK:      VPU.NCE.DepthConvolution
        // CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 2 of @NCE {
        config.MemoryResource 2096128 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitSOHDepthConvHeightByMinNumTiles
    func.func @SplitSOHDepthConvHeightByMinNumTiles(%arg0: tensor<1x16x20000x1xf16, {order = #NHWC}>) -> tensor<1x16x20000x1xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x1x1x1xf16, {order = #NHWC}>, [#const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<0.0> : tensor<16x1x1x1xf32>
        %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0, 1, 1, 0>} -> tensor<1x16x20000x1xf16, {order = #NHWC}>
        return %0 : tensor<1x16x20000x1xf16, {order = #NHWC}>

        // CHECK:      VPU.NCE.DepthConvolution
        // CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 2 of @NCE {
        config.MemoryResource 2096128 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitSOKDepthConvHeightByMinNumTiles
    func.func @SplitSOKDepthConvHeightByMinNumTiles(%arg0: tensor<1x32x20000x1xf16, {order = #NHWC}>) -> tensor<1x32x20000x1xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x1x1x1xf16, {order = #NHWC}>, [#const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
        %scale_table = const.Declare tensor<32x1x1x1xf32> = dense<1.0> : tensor<32x1x1x1xf32>
        %bias_table = const.Declare tensor<32x1x1x1xf32> = dense<0.0> : tensor<32x1x1x1xf32>
        %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0, 1, 1, 0>} -> tensor<1x32x20000x1xf16, {order = #NHWC}>
        return %0 : tensor<1x32x20000x1xf16, {order = #NHWC}>

        // CHECK:      VPU.NCE.DepthConvolution
        // CHECK-SAME: tilingStrategy = [1, 1, 3, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 1 of @NCE

    // CHECK-LABEL: @AdjustSplitDepthConvResultErrata
    func.func @AdjustSplitDepthConvResultErrata(%arg0: tensor<1x2416x20x10xf16, {order = #NHWC}>) -> tensor<1x2416x20x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<2416x16x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.0> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.Broadcast<0 : i64, 2401 : i64>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>>, #const.Reshape<[2416, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
        %scale_table = const.Declare tensor<2416x1x1x1xf32> = dense<1.0> : tensor<2416x1x1x1xf32>
        %bias_table = const.Declare tensor<2416x1x1x1xf32> = dense<0.0> : tensor<2416x1x1x1xf32>
        %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [2416, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0, 1, 1, 0>} -> tensor<1x2416x20x10xf16, {order = #NHWC}>
        return %0 : tensor<1x2416x20x10xf16, {order = #NHWC}>

        // CHECK:      VPU.NCE.DepthConvolution
        // CHECK-SAME: tilingStrategy = [1, 2, 1, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @TilingForDWConvSEP {
    config.Resources 4 of @NCE at 6.000000e+02 MHz

// CHECK-LABEL: @DWConvWithSEPSOK
func.func @DWConvWithSEPSOK(%arg0: tensor<1x1152x1x1xf16, {order = #NHWC}>) -> tensor<1x1152x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<1152x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1152x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x1152x2x2xi1> = dense<1> : tensor<1x1152x2x2xi1>

    %storage_element = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 18, seSize = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        dataShape = [1, 1152, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 1152, 2, 2]>
    } -> tensor<1x18x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 1152, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x1152x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x1152x2x2xi1>,
                           storage_element_table=tensor<1x18x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 1152, 2, 2]>>

    %interpolate = VPU.NCE.DepthConvolution(%input, %weights) rawFilterShape [1152, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,

        strides = [1, 1]
    } -> tensor<1x1152x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x1152x2x2xf16, {order = #NHWC}>

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-SAME:     tilingStrategy = [1, 2, 1, 1]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DWConvWithSEPNoMC
func.func @DWConvWithSEPNoMC(%arg0: tensor<1x1152x1x1xf16, {order = #NHWC}>) -> tensor<1x1152x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<1152x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1152x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x1152x2x2xi1> = dense<1> : tensor<1x1152x2x2xi1>

    %storage_element = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 18, seSize = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        dataShape = [1, 1152, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 1152, 2, 2]>
    } -> tensor<1x18x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 1152, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x1152x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x1152x2x2xi1>,
                           storage_element_table=tensor<1x18x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 1152, 2, 2]>>

    %interpolate = VPU.NCE.DepthConvolution(%input, %weights) rawFilterShape [1152, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,

        strides = [1, 1]
    } -> tensor<1x1152x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x1152x2x2xf16, {order = #NHWC}>

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-SAME:     tilingStrategy = [1, 5, 1, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 4 of @NCE

    // CHECK-LABEL: @AdjustSplitDepthConvResultErrataMultiCluster
    func.func @AdjustSplitDepthConvResultErrataMultiCluster(%arg0: tensor<1x2416x20x10xf16, {order = #NHWC}>) -> tensor<1x2416x20x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<2416x16x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.0> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.Broadcast<0 : i64, 2401 : i64>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>>, #const.Reshape<[2416, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
        %scale_table = const.Declare tensor<2416x1x1x1xf32> = dense<1.0> : tensor<2416x1x1x1xf32>
        %bias_table = const.Declare tensor<2416x1x1x1xf32> = dense<0.0> : tensor<2416x1x1x1xf32>
        %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [2416, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0, 1, 1, 0>} -> tensor<1x2416x20x10xf16, {order = #NHWC}>
        return %0 : tensor<1x2416x20x10xf16, {order = #NHWC}>

        // CHECK:      VPU.NCE.DepthConvolution
        // CHECK-SAME: tilingStrategy = [1, 2, 1, 1]
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.13278423384124158:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 4 of @NCE {
        config.MemoryResource 2096128 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @DWConvWithKernelOptimizationSplitOverC
    func.func @DWConvWithKernelOptimizationSplitOverC(%arg0: tensor<1x960x65x65x!qElemType, {order = #NHWC}>) -> tensor<1x960x65x65x!qElemType, {order = #NHWC}> {
        %cst = const.Declare tensor<960x96x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<960x9x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.Reshape<[960, 81, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
        %0 = VPU.NCE.DepthConvolution(%arg0, %cst) rawFilterShape [960, 1, 9, 9] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 4 : i64, right = 4 : i64, top = 4 : i64, bottom = 4 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,  strides = [1, 1]} -> tensor<1x960x65x65x!qElemType, {order = #NHWC}>
        return %0 : tensor<1x960x65x65x!qElemType, {order = #NHWC}>

        // CHECK:      [[DW:%.+]] = VPU.NCE.DepthConvolution
        // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        // CHECK-SAME:     tilingStrategy = [1, 2, 1, 1]
        // CHECK:      return [[DW]]
    }
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 4 of @NCE {
        config.MemoryResource 2096128 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }
// CHECK-LABEL: func.func @SplitNCEConvNoReduce
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x768x768xf16, {order = #NHWC}>
func.func @SplitNCEConvNoReduce(%arg0: tensor<1x16x768x768xf16, {order = #NHWC}>) -> tensor<1x16x768x768xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 16, 3, 3],
        strides = [1, 1]
    } : tensor<1x16x768x768xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x768x768xf16, {order = #NHWC}>

    return %0 : tensor<1x16x768x768xf16, {order = #NHWC}>

    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS:%.+]])
    // CHECK-SAME: tilingStrategy = [1, 1, 20, 1]
    // CHECK: return [[CONV]] : tensor<1x16x768x768xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 4 of @NCE {
        config.MemoryResource 2096128 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

// CHECK-LABEL: func.func @SplitNCEConvWithReduce
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x768x768xf16, {order = #NHWC}>
func.func @SplitNCEConvWithReduce(%arg0: tensor<1x16x768x768xf16, {order = #NHWC}>) -> (tensor<1x16x768x768xf16, {order = #NHWC}>, tensor<1x1x768x768xf16, {order = #NHWC}>) {
    %weights = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %out, %rmax = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [16, 16, 3, 3] {axes_value = [1],
        resultSegmentSizes = array<i32: 1, 1, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 16, 3, 3],
        strides = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>
    } : tensor<1x16x768x768xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x768x768xf16, {order = #NHWC}>, tensor<1x1x768x768xf16, {order = #NHWC}>

    return %out, %rmax : tensor<1x16x768x768xf16, {order = #NHWC}>, tensor<1x1x768x768xf16, {order = #NHWC}>

    // CHECK: [[CONV:%.+]], [[RMAX:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS:%.+]])
    // CHECK-SAME: axes_value = [1]
    // CHECK-SAME: tilingStrategy = [1, 1, 21, 1]
    // CHECK: return [[CONV]], [[RMAX]] : tensor<1x16x768x768xf16, {order = #NHWC}>, tensor<1x1x768x768xf16, {order = #NHWC}>
}
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 4 of @NCE {
        config.MemoryResource 2096128 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

// CHECK-LABEL: func.func @SplitNCEMaxPoolNoReduce
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x768x768xf16, {order = #NHWC}>
func.func @SplitNCEMaxPoolNoReduce(%arg0: tensor<1x16x768x768xf16, {order = #NHWC}>) -> tensor<1x16x768x768xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [3, 3],
        strides = [1, 1],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>
    } : tensor<1x16x768x768xf16, {order = #NHWC}> -> tensor<1x16x768x768xf16, {order = #NHWC}>

    return %0 : tensor<1x16x768x768xf16, {order = #NHWC}>

    // CHECK: [[POOL:%.+]] = VPU.NCE.MaxPool([[INPUT]])
    // CHECK-SAME: tilingStrategy = [1, 1, 20, 1]
    // CHECK: return [[POOL]] : tensor<1x16x768x768xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 4 of @NCE {
        config.MemoryResource 2096128 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

// CHECK-LABEL: func.func @SplitNCEMaxPoolWithReduce
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x768x768xf16, {order = #NHWC}>
func.func @SplitNCEMaxPoolWithReduce(%arg0: tensor<1x16x768x768xf16, {order = #NHWC}>) -> (tensor<1x16x768x768xf16, {order = #NHWC}>, tensor<1x1x768x768xf16, {order = #NHWC}>) {
    %out, %rmax = VPU.NCE.MaxPool(%arg0) {
        axes_value = [1],
        resultSegmentSizes = array<i32: 1, 1, 0, 0>,
        kernel_size = [3, 3],
        strides = [1, 1],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>
    } : tensor<1x16x768x768xf16, {order = #NHWC}> -> tensor<1x16x768x768xf16, {order = #NHWC}>, tensor<1x1x768x768xf16, {order = #NHWC}>

    return %out, %rmax : tensor<1x16x768x768xf16, {order = #NHWC}>, tensor<1x1x768x768xf16, {order = #NHWC}>

    // CHECK: [[POOL:%.+]], [[RMAX:%.+]] = VPU.NCE.MaxPool([[INPUT]])
    // CHECK-SAME: axes_value = [1]
    // CHECK-SAME: tilingStrategy = [1, 1, 21, 1]
    // CHECK: return [[POOL]], [[RMAX]] : tensor<1x16x768x768xf16, {order = #NHWC}>, tensor<1x1x768x768xf16, {order = #NHWC}>
}
}
