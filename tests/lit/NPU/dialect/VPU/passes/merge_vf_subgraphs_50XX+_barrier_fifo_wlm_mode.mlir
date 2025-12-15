//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs="workload-management-mode=PWLM_V1_BARRIER_FIFO" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

//CHECK-LABEL: @MergeDueToLowerCost
func.func @MergeDueToLowerCost(%arg0: tensor<1x1x1x4096xf16>, %arg1: tensor<5504x4096x1x1x!qElemType, {order = #NHWC}>, %arg2: tensor<1x5504x1x1xf16, {order = #NHWC}>) -> tensor<1x2752x1x1xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<1x1x1x4096xf16> = dense<1.0> : tensor<1x1x1x4096xf16>
  %0 = VPU.RMS(%arg0, %cst_0) {eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x4096xf16>, tensor<1x1x1x4096xf16> -> tensor<1x1x1x4096xf16>
  %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 4096, 1, 1]} : tensor<1x1x1x4096xf16> -> tensor<1x4096x1x1xf16>
  %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4096x1x1xf16> -> tensor<1x4096x1x1xf16, {order = #NHWC}>
  %3 = VPU.VerticalFusion (%2 as %arg3: tensor<1x4096x1x1xf16, {order = #NHWC}>,
                           %arg1 as %arg4: tensor<5504x4096x1x1x!qElemType, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 8, 1, 1]} -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg3, %arg4) {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,
            rawFilterShape = [5504, 4096, 1, 1],
            strides = [1, 1]} : tensor<1x4096x1x1xf16, {order = #NHWC}>, tensor<5504x4096x1x1x!qElemType, {order = #NHWC}> -> tensor<1x5504x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }
  %4 = VPU.VerticalFusion (%3 as %arg3: tensor<1x5504x1x1xf16, {order = #NHWC}>,
                           %arg2 as %arg4: tensor<1x5504x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Eltwise(%arg3, %arg4) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            op_type = #VPU.eltwise_type<MULTIPLY>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>} -> tensor<1x5504x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 2752, 1, 1] : tensor<1x5504x1x1xf16, {order = #NHWC}> to tensor<1x2752x1x1xf16, {order = #NHWC}>
  %6 = VPU.VerticalFusion (%5 as %arg3: tensor<1x2752x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x2752x1x1xf16, {order = #NHWC}> {
    %inner = VPU.Swish(%arg3) {
            beta_value = 1.000000e+00 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x2752x1x1xf16, {order = #NHWC}> -> tensor<1x2752x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  return %6 : tensor<1x2752x1x1xf16, {order = #NHWC}>

  //CHECK:  [[RMS:%.+]] = VPU.RMS
  //CHECK:  [[RESHAPE:%.+]] = VPU.AffineReshape([[RMS]])
  //CHECK:  [[PERMUTE:%.+]] = VPU.PermuteCast([[RESHAPE]])
  //CHECK:  [[VF_0:%.+]] = VPU.VerticalFusion ([[PERMUTE]]
  //CHECK:  VPU.NCE.Convolution
  //CHECK:  VPU.NCE.Eltwise
  //CHECK:  [[SLICE:%.+]] = VPU.Slice [[VF_0]]
  //CHECK:  [[VF_1:%.+]] = VPU.VerticalFusion ([[SLICE]]
  //CHECK:  VPU.Swish
  //CHECK:  return [[VF_1]] : tensor<1x2752x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @NotBuildSubgraphOutOfSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> (tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<1x32x256x256x!qElemType, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg1, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg1, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }
    %2 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg1, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }
    %3 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %2 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg1, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    return %1, %3 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<1x32x256x256x!qElemType, {order = #NHWC}>


    //CHECK: [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK: [[CONV:%.+]] = VPU.NCE.Convolution(%arg1, %arg2)
    //CHECK: VPU.Yield [[CONV]]

    //CHECK: [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion ([[VERTICAL_FUSION0]] as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK: [[CONV:%.+]] = VPU.NCE.Convolution(%arg1, %arg2)
    //CHECK: VPU.Yield [[CONV]]

    //CHECK: [[VERTICAL_FUSION2:%.+]] = VPU.VerticalFusion ([[VERTICAL_FUSION0]] as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>)
    //CHECK-SAME: attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>
    //CHECK: [[CONV:%.+]] = VPU.NCE.Convolution(%arg1, %arg2)
    //CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise(%arg1, [[CONV]])
    //CHECK: VPU.Yield [[ELTWISE]]

    //CHECK: return [[VERTICAL_FUSION1]], [[VERTICAL_FUSION2]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}
