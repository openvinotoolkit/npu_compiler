//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment="enable-vpunn-cost-for-tiling=true" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 3 of @NCE at 1.700000e+03 MHz
    // CHECK-LABEL: @TileConvolutionOnHeightWithRestraintOfMinimalH
    func.func @TileConvolutionOnHeightWithRestraintOfMinimalH(%arg0: tensor<1x3840x256x4xf16, {order = #NHWC}>, %arg1: tensor<1536x3840x1x1xf16, {order = #NHWC}>) -> tensor<1x1536x256x4xf16, {order = #NHWC}>  {
        %cst = const.Declare tensor<1536x1x1x4xsi32> = dense<1> : tensor<1536x1x1x4xsi32>

        %conv = VPU.NCE.Convolution(%arg0, %arg1, %cst) {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            rawFilterShape = [1536, 3840, 1, 1],
            strides = [1, 1]
        } : tensor<1x3840x256x4xf16, {order = #NHWC}>, tensor<1536x3840x1x1xf16, {order = #NHWC}>, tensor<1536x1x1x4xsi32> -> tensor<1x1536x256x4xf16, {order = #NHWC}>


        return %conv : tensor<1x1536x256x4xf16, {order = #NHWC}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
        // CHECK-SAME:          tilingStrategy = [1, 4, 19, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
  config.Resources 4 of @NCE at 1.950000e+03 MHz

  // CHECK-LABEL: @pipeliningTilingForBigFilter
  func.func @pipeliningTilingForBigFilter(%arg0: tensor<1x1536x1x1xf16, {order = #NHWC}>, %arg1: tensor<8160x1536x1x1x!quant.uniform<i8:f16, 1.000000e+00>, {order = #NHWC}>) -> tensor<1x8160x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<8160x1x1x4xsi32> = dense<10> : tensor<8160x1x1x4xsi32>
    %310 = VPU.NCE.Convolution(%arg0, %arg1, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [8160, 1536, 1, 1], strides = [1, 1]} : tensor<1x1536x1x1xf16, {order = #NHWC}>, tensor<8160x1536x1x1x!quant.uniform<i8:f16, 1.000000e+00>, {order = #NHWC}>, tensor<8160x1x1x4xsi32> -> tensor<1x8160x1x1xf16, {order = #NHWC}>
    return %310 : tensor<1x8160x1x1xf16, {order = #NHWC}>

    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:    tilingStrategy = [1, 10, 1, 1]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType1 = !quant.quantile<u4:f16:f16, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}:1.000000e+00>

module @executors {
  config.PipelineOptions @Options {
    config.Option @config.EnableVPUNNPreSplit : true
  }
  config.Resources 3 of @NCE at 1.700000e+03 MHz

  // CHECK-LABEL: @multiDimPipelineTilingForSpatialFirst
  func.func @multiDimPipelineTilingForSpatialFirst(%arg0: tensor<1x3072x256x4xf16, {order = #NHWC}>, %arg1: tensor<8192x3072x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x8192x256x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<8192x1x1x4xsi32> = dense<10> : tensor<8192x1x1x4xsi32>
    %0 = VPU.NCE.Convolution(%arg0, %arg1, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [8192, 3072, 1, 1], strides = [1, 1]} : tensor<1x3072x256x4xf16, {order = #NHWC}>, tensor<8192x3072x1x1x!qElemType1, {order = #NHWC}>, tensor<8192x1x1x4xsi32> -> tensor<1x8192x256x4xf16, {order = #NHWC}>
    return %0 : tensor<1x8192x256x4xf16, {order = #NHWC}>

    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:    tilingStrategy = [1, 128, 3, 1]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
  config.PipelineOptions @Options {
    config.Option @config.EnableVPUNNPreSplit : true
  }
  config.Resources 3 of @NCE at 1.700000e+03 MHz

  // CHECK-LABEL: @multiDimPipelineTilingForWeightFirst
  func.func @multiDimPipelineTilingForWeightFirst(%arg0: tensor<1x2048x256x4xf16, {order = #NHWC}>, %arg1: tensor<2048x2048x1x1xf16, {order = #NHWC}>) -> tensor<1x2048x256x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<2048x1x1x4xsi32> = dense<10> : tensor<2048x1x1x4xsi32>
    %0 = VPU.NCE.Convolution(%arg0, %arg1, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [2048, 2048, 1, 1], strides = [1, 1]} : tensor<1x2048x256x4xf16, {order = #NHWC}>, tensor<2048x2048x1x1xf16, {order = #NHWC}>, tensor<2048x1x1x4xsi32> -> tensor<1x2048x256x4xf16, {order = #NHWC}>
    return %0 : tensor<1x2048x256x4xf16, {order = #NHWC}>

    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:    tilingStrategy = [1, 4, 32, 1]
  }
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
// CHECK-LABEL: @NCEMatMulSOGAndMultiDimPipelined
  func.func @NCEMatMulSOGAndMultiDimPipelined(%arg0: tensor<32x1x128x1024x1xf16, {order = #GNHWC}>) ->  tensor<32x1x1408x1024x1xf16, {order = #GNHWC}>{
    %weight = const.Declare tensor<32x1408x128x1x1xf16, {order = #GNHWC}> = dense<10.0> : tensor<32x1408x128x1x1xf16, {order = #GNHWC}>
    %weight_table = const.Declare tensor<32x1408x1x1x4xsi32> = dense<0> : tensor<32x1408x1x1x4xsi32>
    %grouped_matmul = VPU.NCE.MatMul(%arg0, %weight, %weight_table) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1408, 128, 1, 1], strides = [1, 1]} -> tensor<32x1x1408x1024x1xf16, {order = #GNHWC}>

    return %grouped_matmul : tensor<32x1x1408x1024x1xf16, {order = #GNHWC}>
    // CHECK:         VPU.NCE.MatMul
    // CHECK-SAME:    tilingStrategy = [8, 1, 1, 7, 1]
  }
}
