//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotMergeVFDueToParentTilingFailure
func.func @NotMergeVFDueToParentTilingFailure(%arg0: tensor<1x256x128x128xf16, {order = #NHWC}>,
                                               %weights: tensor<512x256x3x3xf16, {order = #NHWC}>,
                                               %wt: tensor<512x1x1x4xsi32>) -> tensor<1x512x64x64xf16, {order = #NHWC}> {

    // Parent VF: Conv with SOK
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x256x128x128xf16, {order = #NHWC}>,
                             %weights as %arg2: tensor<512x256x3x3xf16, {order = #NHWC}>,
                             %wt as %arg3: tensor<512x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 10], isManualConfigured} -> tensor<1x512x64x64xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3) rawFilterShape [512, 256, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
          pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
           strides = [2, 2]} : tensor<1x256x128x128xf16, {order = #NHWC}>, tensor<512x256x3x3xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x64x64xf16, {order = #NHWC}>
      VPU.Yield %inner
    }

    // MVN with SOK
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x512x64x64xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x512x64x64xf16, {order = #NHWC}> {
      %inner = VPU.MVN(%arg1) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x64x64xf16, {order = #NHWC}>
      VPU.Yield %inner
    }

    // AveragePool with SOH
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x512x64x64xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 3]} -> tensor<1x512x64x64xf16, {order = #NHWC}> {
      %inner = VPU.NCE.AveragePool(%arg1) {kernel_size = [1, 1],
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]} -> tensor<1x512x64x64xf16, {order = #NHWC}>
      VPU.Yield %inner
    }

    return %2 : tensor<1x512x64x64xf16, {order = #NHWC}>

    // Merging MVN+AvgPool would adjust the input op's MC strategy,
    // but the parent VF (Conv with stride=2, tiling=[1,1,1,10]) cannot
    // recalculate tiling regions after the strategy change.

    //CHECK: [[VF_CONV:%.+]] = VPU.VerticalFusion
    //CHECK:    VPU.NCE.Convolution
    //CHECK:    VPU.Yield
    //CHECK: [[VF_MVN:%.+]] = VPU.VerticalFusion
    //CHECK:    VPU.MVN
    //CHECK:    VPU.Yield
    //CHECK: [[VF_AVG:%.+]] = VPU.VerticalFusion
    //CHECK:    VPU.NCE.AveragePool
    //CHECK:    VPU.Yield
    //CHECK: return [[VF_AVG]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.053211316875382969>

module @executors {
  config.Resources 3 of @NCE at 1.700000e+03 MHz {
      config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

// CHECK-LABEL: @MergeVFSubgraphsLargeWorkloadChannel
func.func @MergeVFSubgraphsLargeWorkloadChannel(%arg0: tensor<1x256x92x92x!qElemType, {order = #NHWC}>) -> tensor<1x256x46x46x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<256x256x3x3x!qElemType, {order = #NHWC}> = dense<1> : tensor<256x256x3x3xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %scale = const.Declare tensor<256x1x1x1xf32> = dense<1.0> : tensor<256x1x1x1xf32>
    %bias  = const.Declare tensor<256x1x1x1xf32> = dense<0.0> : tensor<256x1x1x1xf32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x256x92x92x!qElemType, {order = #NHWC}>, %weights as %arg2: tensor<256x256x3x3x!qElemType, {order = #NHWC}>, %scale as %arg3: tensor<256x1x1x1xf32>, %bias as %arg4: tensor<256x1x1x1xf32>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x256x92x92x!qElemType, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3, %arg4) rawFilterShape [256, 256, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64,
            prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
            
            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        }: tensor<1x256x92x92x!qElemType, {order = #NHWC}>, tensor<256x256x3x3x!qElemType, {order = #NHWC}>, tensor<256x1x1x1xf32>, tensor<256x1x1x1xf32> -> tensor<1x256x92x92x!qElemType, {order = #NHWC}>
        VPU.Yield %inner
    }

    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x256x92x92x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x256x46x46x!qElemType, {order = #NHWC}> {
        %inner = VPU.NCE.MaxPool(%arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            kernel_size = [2, 2],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [2, 2],
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 1, 1, 0>
        } -> tensor<1x256x46x46x!qElemType, {order = #NHWC}>
        VPU.Yield %inner
    }

    return %1 : tensor<1x256x46x46x!qElemType, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    //CHECK-SAME:           attributes {scenario = #VPU.vf_scenario<LASTOP_PREFETCHING>
    //CHECK: [[CONV:%.+]] = VPU.NCE.Convolution(
    //CHECK: [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[CONV]]
    //CHECK: VPU.Yield [[MAXPOOL]]
    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x256x46x46x!qElemType, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#HNWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @MergeVFSubgraphsMemPermuteDynamicDequantize
func.func @MergeVFSubgraphsMemPermuteDynamicDequantize(
        %arg0: tensor<1x32x4096x128x!qElemType>,
        %arg1: tensor<1x4096x32x1xf16>) -> tensor<1x4096x32x128xf16> {
    %cst_scale = const.Declare tensor<4096x1x1x1xf32> = dense<1.0> : tensor<4096x1x1x1xf32>
    %cst_bias  = const.Declare tensor<4096x1x1x1xf32> = dense<0.0> : tensor<4096x1x1x1xf32>

    %0 = VPU.VerticalFusion (%arg0 as %in0: tensor<1x32x4096x128x!qElemType>)
            attributes {tilingStrategy = [1, 3, 1, 1]} -> tensor<1x4096x32x128x!qElemType> {
        %inner = VPU.MemPermute(%in0) {
            dst_order = #NCHW,
            mem_perm  = #NHCW,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x32x4096x128x!qElemType> -> tensor<1x4096x32x128x!qElemType>
        VPU.Yield %inner
    }

    %1 = VPU.VerticalFusion (
            %0 as %in0: tensor<1x4096x32x128x!qElemType>,
            %arg1 as %in1: tensor<1x4096x32x1xf16>)
            attributes {tilingStrategy = [1, 11, 1, 1]} -> tensor<1x4096x32x128xf16> {
        %inner = VPU.DynamicDequantize(%in0, %in1) {
            dstElemType = f16,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x4096x32x128x!qElemType>, tensor<1x4096x32x1xf16> -> tensor<1x4096x32x128xf16>
        VPU.Yield %inner
    }

    return %1 : tensor<1x4096x32x128xf16>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (
    //CHECK-SAME:     %arg0 as [[IN0:%.+]]: tensor<1x32x4096x128x!qElemType>,
    //CHECK-SAME:     %arg1 as [[IN1:%.+]]: tensor<1x4096x32x1xf16>)
    //CHECK-SAME:     attributes {scenario = #VPU.vf_scenario<WEIGHTS_PREFETCHING>, tilingStrategy = [1, 11, 1, 1]}
    //CHECK-SAME:     -> tensor<1x4096x32x128xf16> {
    //CHECK:        [[PERM:%.+]] = VPU.MemPermute([[IN0]])
    //CHECK-SAME:       {dst_order = #NCHW, mem_perm = #NHCW, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       : tensor<1x32x4096x128x!qElemType> -> tensor<1x4096x32x128x!qElemType>
    //CHECK:        [[DEQ:%.+]] = VPU.DynamicDequantize([[PERM]], [[IN1]])
    //CHECK-SAME:       {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       : tensor<1x4096x32x128x!qElemType>, tensor<1x4096x32x1xf16> -> tensor<1x4096x32x128xf16>
    //CHECK:        VPU.Yield [[DEQ]]
    //CHECK:      }
    //CHECK:      return [[VERTICAL_FUSION]] : tensor<1x4096x32x128xf16>
}
