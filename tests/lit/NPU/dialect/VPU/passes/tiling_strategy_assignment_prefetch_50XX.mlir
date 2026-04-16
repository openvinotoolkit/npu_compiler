//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
!quantileType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>

module @executors {
    config.Resources 3 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @QuantileConvWithGatherDMA
    func.func @QuantileConvWithGatherDMA(
            %arg0: tensor<1x2880x1x1xf16, {order = #NHWC}>,
            %arg1: tensor<32x2880x2880x!quantileFloatType>)
                -> tensor<1x2880x1x1xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<2880x1x1x4xsi32> = dense<1> : tensor<2880x1x1x4xsi32>
        %indices = const.Declare tensor<11520x1x1x1xi64> = dense<1> : tensor<11520x1x1x1xi64>

        %0 = VPU.AffineReshape(%arg1) {dim_mapping = [[0], [0], [1]], shape_value = [92160, 2880]} :
            tensor<32x2880x2880x!quantileFloatType> ->
            tensor<92160x2880x!quantileFloatType>

        %1 = VPU.QuantizeCast(%0) {dstElemType = !quantileType} :
            tensor<92160x2880x!quantileFloatType> ->
            tensor<92160x2880x!quantileType>

        %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [1, 2, 3]], shape_value = [92160, 2880, 1, 1]} :
            tensor<92160x2880x!quantileType> ->
            tensor<92160x2880x1x1x!quantileType>

        %3 = VPU.GatherDMA(%2, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>} :
            tensor<92160x2880x1x1x!quantileType>, tensor<11520x1x1x1xi64> ->
            tensor<11520x2880x1x1x!quantileType>

        %4 = VPU.PermuteCast(%3) {dst_order = #NHWC, mem_perm = #NHWC} :
            tensor<11520x2880x1x1x!quantileType> ->
            tensor<11520x2880x1x1x!quantileType, {order = #NHWC}>

        %5 = VPU.Slice %4 [8640, 0, 0, 0] [2880, 2880, 1, 1] :
            tensor<11520x2880x1x1x!quantileType, {order = #NHWC}> to
            tensor<2880x2880x1x1x!quantileType, {order = #NHWC}>

        %6 = VPU.NCE.Convolution(%arg0, %5, %cst) {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            rawFilterShape = [2880, 2880, 1, 1],
            strides = [1, 1]
        } : tensor<1x2880x1x1xf16, {order = #NHWC}>, tensor<2880x2880x1x1x!quantileType, {order = #NHWC}>, tensor<2880x1x1x4xsi32> -> tensor<1x2880x1x1xf16, {order = #NHWC}>

        return %6 : tensor<1x2880x1x1xf16, {order = #NHWC}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
        // CHECK-SAME:      tilingStrategy = [1, 3, 1, 1]

        // CHECK:       return [[CONV]] : tensor<1x2880x1x1xf16, {order = #NHWC}>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!quantileType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>

module @executors {
    config.Resources 3 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @QuantileConvLargeKernel
    func.func @QuantileConvLargeKernel(
            %arg0: tensor<1x2880x1x1xf16, {order = #NHWC}>,
            %arg1: tensor<5760x2880x1x1x!quantileType, {order = #NHWC}>)
                -> tensor<1x5760x1x1xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<5760x1x1x4xsi32> = dense<1> : tensor<5760x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %arg1, %cst) {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            rawFilterShape = [5760, 2880, 1, 1],
            strides = [1, 1]
        } : tensor<1x2880x1x1xf16, {order = #NHWC}>, tensor<5760x2880x1x1x!quantileType, {order = #NHWC}>, tensor<5760x1x1x4xsi32> -> tensor<1x5760x1x1xf16, {order = #NHWC}>

        return %0 : tensor<1x5760x1x1xf16, {order = #NHWC}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
        // CHECK-SAME:      tilingStrategy = [1, 6, 1, 1]

        // CHECK:       return [[CONV]] : tensor<1x5760x1x1xf16, {order = #NHWC}>
    }
}

// -----

// CHECK-LABEL: func.func @RMSSOK_TileOverC
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x192x192x512xf16>)
func.func @RMSSOK_TileOverC(%arg0: tensor<1x192x192x512xf16>) -> tensor<1x192x192x512xf16> {
    %cst = const.Declare tensor<1x1x1x512xf16> = dense<1.0> : tensor<1x1x1x512xf16>
    %0 = VPU.RMS(%arg0, %cst) {
        eps = 9.9999999747524271E-7 : f64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
        : tensor<1x192x192x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x192x192x512xf16>
    return %0 : tensor<1x192x192x512xf16>

    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+00> : tensor<1x1x1x512xf16>
    // CHECK:      [[OUTPUT:%.+]] = VPU.RMS([[INPUT]], [[CST]]) {
    // CHECK-SAME:    eps = 9.9999999747524271E-7 : f64
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:    tilingStrategy = [1, 64, 1, 1]

    // CHECK:      return [[OUTPUT]] : tensor<1x192x192x512xf16>
}
