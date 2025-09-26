//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convolution-split-over-input-channel %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 2 of @NCE
// CHECK-LABEL: func.func @SplitOverInputChannelOn2T
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x3072x128x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<768x3072x1x1xf16, {order = #NHWC}>
func.func @SplitOverInputChannelOn2T(%arg0: tensor<1x3072x128x4xf16, {order = #NHWC}>, %arg1: tensor<768x3072x1x1xf16, {order = #NHWC}>) -> tensor<1x768x128x4xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %cst)  {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [768, 3072, 1, 1], strides = [1, 1], tilingStrategy = [1, 2, 5, 1]} : tensor<1x3072x128x4xf16, {order = #NHWC}>, tensor<768x3072x1x1xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32> -> tensor<1x768x128x4xf16, {order = #NHWC}>

  return %0 : tensor<1x768x128x4xf16, {order = #NHWC}>

        // CHECK:       [[CONST0:%.+]] = const.Declare tensor<768x1x1x4xsi32>
        // CHECK:       [[CONST1:%.+]] = const.Declare tensor<768x1x1x4xsi32>
        // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT_0]] 
        // CHECK-SAME:  [0, 0, 0, 0] [1, 1536, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1536x128x4xf16, {order = #NHWC}>
        // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_1]] 
        // CHECK-SAME:  [0, 0, 0, 0] [768, 1536, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1536x1x1xf16, {order = #NHWC}>
        // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[SLICE1]], [[CONST1]]) 
        // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        // CHECK-SAME:  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        // CHECK-SAME:  tilingStrategy = [1, 6, 1, 1]
        // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}>
        // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT_0]]
        // CHECK-SAME:  [0, 1536, 0, 0] [1, 1536, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1536x128x4xf16, {order = #NHWC}>
        // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT_1]] 
        // CHECK-SAME:  [0, 1536, 0, 0] [768, 1536, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1536x1x1xf16, {order = #NHWC}>
        // CHECK:       [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[SLICE3]], [[CONST0]]) 
        // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        // CHECK-SAME:  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        // CHECK-SAME:  tilingStrategy = [1, 6, 1, 1]
        // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}> 
        // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[CONVOLUTION0]], [[CONVOLUTION1]]) 
        // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        // CHECK-SAME:  op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 1]}
        // CHECK:       return [[ELTWISE0]] : tensor<1x768x128x4xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 1 of @NCE
// CHECK-LABEL: func.func @SplitOverInputChannelOn1T
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x3072x128x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<768x3072x1x1xf16, {order = #NHWC}>
func.func @SplitOverInputChannelOn1T(%arg0: tensor<1x3072x128x4xf16, {order = #NHWC}>, %arg1: tensor<768x3072x1x1xf16, {order = #NHWC}>) -> tensor<1x768x128x4xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %cst)  {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [768, 3072, 1, 1], strides = [1, 1], tilingStrategy = [1, 3, 10, 1]} : tensor<1x3072x128x4xf16, {order = #NHWC}>, tensor<768x3072x1x1xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32> -> tensor<1x768x128x4xf16, {order = #NHWC}>

  return %0 : tensor<1x768x128x4xf16, {order = #NHWC}>

        // CHECK:       [[CONST0:%.+]] = const.Declare tensor<768x1x1x4xsi32>
        // CHECK:       [[CONST1:%.+]] = const.Declare tensor<768x1x1x4xsi32>
        // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 1024, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
        // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [768, 1024, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>
        // CHECK:       [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[SLICE1]], [[CONST1]]) 
        // CHECK-SAME:  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        // CHECK-SAME:  tilingStrategy = [1, 8, 1, 1]
        // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}> 
        // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 1024, 0, 0] [1, 1024, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
        // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 1024, 0, 0] [768, 1024, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>
        // CHECK:       [[CONVOLUTION1:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[SLICE3]], [[CONST0]]) 
        // CHECK-SAME:  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        // CHECK-SAME:  tilingStrategy = [1, 8, 1, 1]
        // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}> 
        // CHECK:       [[SLICE4:%.+]] = VPU.Slice [[INPUT_0]] [0, 2048, 0, 0] [1, 1024, 128, 4] : tensor<1x3072x128x4xf16, {order = #NHWC}> to tensor<1x1024x128x4xf16, {order = #NHWC}>
        // CHECK:       [[SLICE5:%.+]] = VPU.Slice [[INPUT_1]] [0, 2048, 0, 0] [768, 1024, 1, 1] : tensor<768x3072x1x1xf16, {order = #NHWC}> to tensor<768x1024x1x1xf16, {order = #NHWC}>
        // CHECK:       [[CONVOLUTION2:%.+]] = VPU.NCE.Convolution([[SLICE4]], [[SLICE5]], [[CONST0]]) 
        // CHECK-SAME:  ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        // CHECK-SAME:  tilingStrategy = [1, 8, 1, 1]
        // CHECK-SAME:  -> tensor<1x768x128x4xf16, {order = #NHWC}> 
        // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[CONVOLUTION0]], [[CONVOLUTION1]]) 
        // CHECK-SAME:  op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]} -> tensor<1x768x128x4xf16, {order = #NHWC}> 
        // CHECK:       [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]], [[CONVOLUTION2]]) 
        // CHECK-SAME:  op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]} -> tensor<1x768x128x4xf16, {order = #NHWC}> 
        // CHECK:       return [[ELTWISE1]] : tensor<1x768x128x4xf16, {order = #NHWC}>
}
}
