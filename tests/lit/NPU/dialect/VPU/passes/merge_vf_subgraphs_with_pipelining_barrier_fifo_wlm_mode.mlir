//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true workload-management-mode=PWLM_V1_BARRIER_FIFO" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.00565029593075023:128>
!qElemType1 = !quant.uniform<u8:f16, 0.013744638480392157:128>

func.func @MergeVFWithGenericVFPipelining(
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
   return %1 : tensor<1x4096x1024x4xf16, {order = #NHWC}>

   // CHECK: [[VF0:%.+]] = VPU.VerticalFusion
   // CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>
   // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution
   // CHECK-NEXT: [[SOFTMAX:%.+]] = VPU.SoftMax
   // CHECK: return [[VF0]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @BuildSubgraphEltwise(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType> {
      %3 = VPU.MemPermute(%arg1)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x32x256x256x!qElemType, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType>
      VPU.Yield %3
    }
    return %1 : tensor<1x32x256x256x!qElemType>
    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    //CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution
    //CHECK:      [[PERMUTE:%.+]] = VPU.MemPermute([[CONV]])
    //CHECK:      return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.00565029593075023:128>
!qElemType1 = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType2 = !quant.uniform<u8:f16, 0.043729394959228592:128>

func.func @MergeVFWithViewOpAndGenericVFPipelining(
                %arg0: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
                %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
   %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
        %arg1 as %arg3: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %2 = VPU.QuantizeCast(%arg2) {dstElemType = !qElemType2} : tensor<1x48x1024x4x!qElemType0, {order = #NHWC}> -> tensor<1x48x1024x4x!qElemType2, {order = #NHWC}>
      %3 = VPU.NCE.Convolution(%2, %arg3)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x1024x4x!qElemType2, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

   %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %2 = VPU.SoftMax(%arg2) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %2
   }
   return %1 : tensor<1x4096x1024x4xf16, {order = #NHWC}>

   // CHECK: [[VF0:%.+]] = VPU.VerticalFusion
   // CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>
   // CHECK: [[CAST:%.+]] = VPU.QuantizeCast
   // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution
   // CHECK-NEXT: [[SOFTMAX:%.+]] = VPU.SoftMax
   // CHECK: return [[VF0]] : tensor<1x4096x1024x4xf16, {order = #NHWC}>
}
