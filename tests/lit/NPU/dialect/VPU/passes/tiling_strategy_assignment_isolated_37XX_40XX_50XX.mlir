//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 1 of @NCE {
        config.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitSingleClusterDepthConvHeightByMinNumTiles
    func.func @SplitSingleClusterDepthConvHeightByMinNumTiles(%arg0: tensor<1x16x10272x1xf16, {order = #NHWC}>) -> tensor<1x16x10272x1xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x1x1x1xf16, {order = #NHWC}>, [#const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
        %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %weights_table) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} -> tensor<1x16x10272x1xf16, {order = #NHWC}>
        return %0 : tensor<1x16x10272x1xf16, {order = #NHWC}>

        // CHECK:      VPU.NCE.DepthConvolution
        // CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 2 of @NCE {
        config.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitSOHDepthConvHeightByMinNumTiles
    func.func @SplitSOHDepthConvHeightByMinNumTiles(%arg0: tensor<1x16x20000x1xf16, {order = #NHWC}>) -> tensor<1x16x20000x1xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x1x1x1xf16, {order = #NHWC}>, [#const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
        %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %weights_table) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} -> tensor<1x16x20000x1xf16, {order = #NHWC}>
        return %0 : tensor<1x16x20000x1xf16, {order = #NHWC}>

        // CHECK:      VPU.NCE.DepthConvolution
        // CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 2 of @NCE {
        config.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitSOKDepthConvHeightByMinNumTiles
    func.func @SplitSOKDepthConvHeightByMinNumTiles(%arg0: tensor<1x32x20000x1xf16, {order = #NHWC}>) -> tensor<1x32x20000x1xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x1x1x1xf16, {order = #NHWC}>, [#const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
        %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
        %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %weights_table) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} -> tensor<1x32x20000x1xf16, {order = #NHWC}>
        return %0 : tensor<1x32x20000x1xf16, {order = #NHWC}>

        // CHECK:      VPU.NCE.DepthConvolution
        // CHECK-SAME: tilingStrategy = [1, 1, 3, 1]
    }
}
