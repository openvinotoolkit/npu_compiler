//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true workload-management-mode=PWLM_V0_1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.00565029593075023:128>
!qElemType1 = !quant.uniform<u8:f16, 0.013744638480392157:128>

func.func @MergeVFWithoutGenericVFPipelining(
                %arg0: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
                %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
   %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
        %arg1 as %arg3: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg2, %arg3) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
       strides = [1, 1]} : tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %2
   }

   %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %2 = VPU.SoftMax(%arg2) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %2
   }
   return %1: tensor<1x4096x1024x4xf16, {order = #NHWC}>

   // CHECK: [[VF0:%.+]] = VPU.VerticalFusion
   // CHECK-SAME: scenario = #VPU.vf_scenario<FULL_PREFETCHING>
   // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution
   // CHECK-NEXT: [[SOFTMAX:%.+]] = VPU.SoftMax
   // CHECK: return [[VF0]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

//CHECK-LABEL: @MergeSubgraphsWithSoftMaxPermuteCastAffineReshapeConv
func.func @MergeSubgraphsWithSoftMaxPermuteCastAffineReshapeConv(
              %arg0: tensor<1x1x1024x512xf16, {order = #NHCW}>,
              %cst: tensor<256x512x1x1x!quant.uniform<i8:f16, 0.047244105488061905>, {order = #NHWC}>)
              -> tensor<1x256x256x4xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x1x1024x512xf16, {order = #NHCW}>) attributes {tilingStrategy = [1, 1, 24, 1]}
                             -> tensor<1x1x1024x512xf16, {order = #NHCW}> {
      %2 = VPU.SoftMax(%arg1) {axisInd = 3 : i64} : tensor<1x1x1024x512xf16, {order = #NHCW}> -> tensor<1x1x1024x512xf16, {order = #NHCW}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x1x1024x512xf16, {order = #NHCW}>,
                             %cst as %arg3: tensor<256x512x1x1x!quant.uniform<i8:f16, 0.047244105488061905>, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 24, 1]}
                             -> tensor<1x256x256x4xf16, {order = #NHWC}> {
      %2 = VPU.PermuteCast(%arg2) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1x1024x512xf16, {order = #NHCW}> -> tensor<1x1x1024x512xf16>
      %3 = VPU.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 512, 1, 1]} : tensor<1x1x1024x512xf16> -> tensor<1024x512x1x1xf16>
      %4 = VPU.PermuteCast(%3) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>} : tensor<1024x512x1x1xf16> -> tensor<1x512x1024x1xf16, {order = #NHWC}>
      %5 = VPU.AffineReshape(%4) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 256, 4]} : tensor<1x512x1024x1xf16, {order = #NHWC}> -> tensor<1x512x256x4xf16, {order = #NHWC}>
      %6 = VPU.NCE.Convolution(%5, %arg3) rawFilterShape [256, 512, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                                   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                                   ppe = #VPU.PPEFp<mode = <NOOP>,
                                                   clamp_low = -3.4028234663852886E+38 : f64,
                                                   clamp_high = 3.4028234663852886E+38 : f64,
                                                   scale = 0.047244105488061905 : f64,
                                                   prelu_alpha = [1.000000e+00],
                                                   bias = 0.000000e+00 : f64,
                                                   adder = 0.000000e+00 : f64>,
                                                   resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                                                   strides = [1, 1]} : tensor<1x512x256x4xf16, {order = #NHWC}>,
                                                                        tensor<256x512x1x1x!quant.uniform<i8:f16, 0.047244105488061905>, {order = #NHWC}>
                                                                        -> tensor<1x256x256x4xf16, {order = #NHWC}>
      VPU.Yield %6
    }
    return %1 : tensor<1x256x256x4xf16, {order = #NHWC}>


    //CHECK: [[VF:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 24, 1]}
    //CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax
    //CHECK: [[PERMUTE1:%.+]] = VPU.PermuteCast([[SOFTMAX]]
    //CHECK: [[RESHAPE1:%.+]] = VPU.AffineReshape([[PERMUTE1]]
    //CHECK: [[PERMUTE2:%.+]] = VPU.PermuteCast([[RESHAPE1]]
    //CHECK: [[RESHAPE2:%.+]] = VPU.AffineReshape([[PERMUTE2]]
    //CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[RESHAPE2]]
    //CHECK: VPU.Yield [[CONV]]
    //CHECK: return [[VF]] : tensor<1x256x256x4xf16, {order = #NHWC}>
}
