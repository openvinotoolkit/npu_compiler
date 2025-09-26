//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true workload-management-mode=PWLM_V0_LCA" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.00565029593075023:128>
!qElemType1 = !quant.uniform<u8:f16, 0.013744638480392157:128>

func.func @MergeVFWithoutGenericVFPipelining(
                %arg0: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
                %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
   %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
        %arg1 as %arg3: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg2, %arg3)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
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
