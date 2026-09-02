//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @MergeDueToLowerCost
func.func @MergeDueToLowerCost(%arg0: tensor<1x1x1x4096xf16>, %arg1: tensor<5504x4096x1x1x!qElemType, {order = #NHWC}>, %arg2: tensor<1x5504x1x1xf16, {order = #NHWC}>) -> tensor<1x2752x1x1xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<1x1x1x4096xf16> = dense<1.0> : tensor<1x1x1x4096xf16>
  %0 = VPU.RMS(%arg0, %cst_0) {eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x4096xf16>, tensor<1x1x1x4096xf16> -> tensor<1x1x1x4096xf16>
  %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 4096, 1, 1]} : tensor<1x1x1x4096xf16> -> tensor<1x4096x1x1xf16>
  %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4096x1x1xf16> -> tensor<1x4096x1x1xf16, {order = #NHWC}>
  %3 = VPU.VerticalFusion (%2 as %arg3: tensor<1x4096x1x1xf16, {order = #NHWC}>,
                           %arg1 as %arg4: tensor<5504x4096x1x1x!qElemType, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 8, 1, 1]} -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg3, %arg4) rawFilterShape [5504, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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

            strides = [1, 1]} : tensor<1x4096x1x1xf16, {order = #NHWC}>, tensor<5504x4096x1x1x!qElemType, {order = #NHWC}> -> tensor<1x5504x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }
  %4 = VPU.VerticalFusion (%3 as %arg3: tensor<1x5504x1x1xf16, {order = #NHWC}>,
                           %arg2 as %arg4: tensor<1x5504x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Eltwise(%arg3, %arg4) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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

  // CHECK:  [[RMS:%.+]] = VPU.RMS
  // CHECK:  [[RESHAPE:%.+]] = VPU.AffineReshape([[RMS]])
  // CHECK:  [[PERMUTE:%.+]] = VPU.PermuteCast([[RESHAPE]])
  // CHECK:  [[VF_0:%.+]] = VPU.VerticalFusion ([[PERMUTE]]
  // CHECK:  VPU.NCE.Convolution
  // CHECK:  VPU.NCE.Eltwise
  // CHECK:  [[SLICE:%.+]] = VPU.Slice [[VF_0]]
  // CHECK:  [[VF_1:%.+]] = VPU.VerticalFusion ([[SLICE]]
  // CHECK:  VPU.Swish
  // CHECK:  return [[VF_1]] : tensor<1x2752x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

// CHECK-LABEL: @NotBuildSubgraphOutOfSubgraph
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>
func.func @NotBuildSubgraphOutOfSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> (tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<1x32x256x256x!qElemType, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
          strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
          strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }
    %2 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
          strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }
    %3 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %2 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg1, %arg2)
         {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    return %1, %3 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<1x32x256x256x!qElemType, {order = #NHWC}>


    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>

    // CHECK: [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, [[CST]] as [[ARG_2:%[^:]+]]: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]])
    // CHECK: VPU.Yield [[CONV]]

    // CHECK: [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion ([[VERTICAL_FUSION0]] as [[ARG_1:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, [[CST]] as [[ARG_2:%[^:]+]]: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]])
    // CHECK: VPU.Yield [[CONV]]

    // CHECK: [[VERTICAL_FUSION2:%.+]] = VPU.VerticalFusion ([[VERTICAL_FUSION0]] as [[ARG_1:%[^:]+]]: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, [[CST_0]] as [[ARG_2:%[^:]+]]: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>)
    // CHECK-SAME: attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>
    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]])
    // CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[ARG_1]], [[CONV]])
    // CHECK: VPU.Yield [[ELTWISE]]

    // CHECK: return [[VERTICAL_FUSION1]], [[VERTICAL_FUSION2]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// The test is to check the case when two subgraphs are not aligned with MC strategies
// Eltwise (SOH) -> PermuteCast (NCHW) -> AffineReshape -> Multiply (SOH) -/> PermuteCast (NHWC) -> Eltwise (SOH)
// The procedure is aligning strategies trying to propagate a new distribution through the PermuteCast (NCHW) and AffineReshape
// The conclusion must be that two subgraphs cannot be merged cause there are no suitable MC strategies to assign
// CHECK-LABEL: @CheckAligningMCStrategiesWithViewLikeChain
func.func @CheckAligningMCStrategiesWithViewLikeChain(%arg0: tensor<1x2048x256x1xf16, {order = #NHWC}>,
%arg1: tensor<1x2048x256x1xf16, {order = #NHWC}>,
%arg2: tensor<1x4x256x1xf16>,
%arg3: tensor<1x2048x4x256xf16, {order = #NHWC}>) -> tensor<1x2048x4x256xf16, {order = #NHWC}> {

   %0 = VPU.VerticalFusion (%arg0 as %arg4: tensor<1x2048x256x1xf16, {order = #NHWC}>, %arg3 as %arg5: tensor<1x2048x256x1xf16, {order = #NHWC}>, %arg2 as %arg6: tensor<1x4x256x1xf16>) attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 3, 1]} -> tensor<1x4x256x2048xf16> {
  	%2 = VPU.NCE.Eltwise(%arg4, %arg5) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x2048x256x1xf16, {order = #NHWC}>
  	%3 = VPU.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x2048x256x1xf16, {order = #NHWC}> -> tensor<1x256x1x2048xf16>
  	%4 = VPU.AffineReshape(%3) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x1x2048xf16> -> tensor<1x1x256x2048xf16>
  	%5 = VPU.Multiply(%arg6, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x256x1xf16>, tensor<1x1x256x2048xf16> -> tensor<1x4x256x2048xf16>
  	VPU.Yield %5
	}
  %1 = VPU.VerticalFusion (%0 as %arg4: tensor<1x4x256x2048xf16>, %arg3 as %arg5: tensor<1x2048x4x256xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 3, 1, 1]} -> tensor<1x2048x4x256xf16, {order = #NHWC}> {
  	%2 = VPU.PermuteCast(%arg4) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x256x2048xf16> -> tensor<1x2048x4x256xf16, {order = #NHWC}>
  	%3 = VPU.NCE.Eltwise(%arg5, %2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x2048x4x256xf16, {order = #NHWC}>
  	VPU.Yield %3
	}
  return %1 : tensor<1x2048x4x256xf16, {order = #NHWC}>

  // CHECK: [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion
  // CHECK: VPU.NCE.Eltwise
  // CHECK: VPU.PermuteCast
  // CHECK: VPU.AffineReshape
  // CHECK: VPU.Multiply
  // CHECK: VPU.Yield

  // CHECK: [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion ([[VERTICAL_FUSION0]]
  // CHECK: VPU.PermuteCast
  // CHECK: VPU.NCE.Eltwise
  // CHECK: VPU.Yield

  // CHECK: return [[VERTICAL_FUSION1]] : tensor<1x2048x4x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>


// CHECK-LABEL: @DecomposedOpsInSdpaSoftmax
module @DecomposedOpsInSdpaSoftmax {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  // CHECK:   ([[ARG0:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
  // CHECK:   [[ARG1:%.+]]: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:   [[ARG2:%.+]]: tensor<48x4096x1x1xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16> {
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
    %cst_1 = const.Declare tensor<48x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reshape<[48, 1, 1, 1]>, #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 15]>]
    %cst_2 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %129 = VPU.VerticalFusion (%arg0 as %206: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1 as %207: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %cst_0 as %arg5: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 32, 1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %208 = VPU.NCE.Convolution(%206, %207) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %208
    }
    %130 = VPU.VerticalFusion (%129 as %206: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 32, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %207 = VPU.SoftMax(%206) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %207
    }
    %156 = VPU.VerticalFusion (%130 as %206: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 29, 1]} -> tensor<1x1x1024x4xf16, {order = #NHWC}> {
      // TODO: replace mode = <EXP> with mode = <INV> once it's supported, E#207875
      %207 = VPU.NCE.Reduce(%206) {axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEFp<mode = <EXP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64, sprlut = dense_resource<__elided__> : tensor<512xui16>>} -> tensor<1x1x1024x4xf16, {order = #NHWC}>
      VPU.Yield %207
    }
    %157 = VPU.VerticalFusion (%156 as %206: tensor<1x1x1024x4xf16, {order = #NHWC}>, %cst_1 as %arg4: tensor<48x1x1x16xf16, {order = #NHWC}>, %cst_2 as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %207 = VPU.NCE.Convolution(%206, %arg4) rawFilterShape [48, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x1x1024x4xf16, {order = #NHWC}>, tensor<48x1x1x16xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %207
    }
    %158 = VPU.VerticalFusion (%130 as %206: tensor<1x4096x1024x4xf16, {order = #NHWC}>, %arg2 as %207: tensor<48x4096x1x1xf16, {order = #NHWC}>, %cst_2 as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 29, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %208 = VPU.NCE.Convolution(%206, %207) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.562500e-02 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %208
    }
    %159 = VPU.VerticalFusion (%158 as %206: tensor<1x48x1024x4xf16, {order = #NHWC}>, %157 as %207: tensor<1x48x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x1024x4xf16> {
      %208 = VPU.NCE.Eltwise(%206, %207) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 6.400000e+01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x48x1024x4xf16>
      VPU.Yield %208
    }

    //  CHECK: [[CST_0:%.+]] = const.Declare
    //  CHECK: [[CST_1:%.+]] = const.Declare
    //  CHECK: [[CST_2:%.+]] = const.Declare
    //  CHECK-NEXT: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    //  CHECK-NEXT: VPU.NCE.Convolution
    //  CHECK-NEXT: VPU.SoftMax
    //  CHECK-NEXT: VPU.NCE.Reduce
    //  CHECK-NEXT: VPU.NCE.Convolution
    //  CHECK-NEXT: VPU.NCE.Convolution
    //  CHECK-NEXT: [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    //  CHECK-NEXT: VPU.Yield [[ELTWISE]]

    //  CHECK: return [[VERTICAL_FUSION]] : tensor<1x48x1024x4xf16>
    return %159 : tensor<1x48x1024x4xf16>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeConvShapeCastSliceShapeCastEltwise
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x256x2048xf16, {order = #NHWC}>
// CHECK-SAME: [[ADDEND:%.+]]: tensor<1x16x256x512xf16, {order = #NHWC}>
func.func @MergeConvShapeCastSliceShapeCastEltwise(
    %input:  tensor<1x16x256x2048xf16, {order = #NHWC}>,
    %addend: tensor<1x16x256x512xf16,  {order = #NHWC}>)
    -> tensor<1x16x256x512xf16, {order = #NHWC}> {
  %cst_weight = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

  %vf_conv = VPU.VerticalFusion (
      %input      as %arg0: tensor<1x16x256x2048xf16, {order = #NHWC}>,
      %cst_weight as %arg1: tensor<16x16x1x1xf16,     {order = #NHWC}>
  ) attributes {tilingStrategy = [1, 1, 8, 1]} -> tensor<1x16x256x2048xf16, {order = #NHWC}> {
    %conv = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [16, 16, 1, 1] {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,

        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        strides = [1, 1]
    } : tensor<1x16x256x2048xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
      -> tensor<1x16x256x2048xf16, {order = #NHWC}>
    VPU.Yield %conv
  }

  %vf_add = VPU.VerticalFusion (
      %vf_conv as %arg0: tensor<1x16x256x2048xf16, {order = #NHWC}>,
      %addend  as %arg1: tensor<1x16x256x512xf16,  {order = #NHWC}>
  ) attributes {tilingStrategy = [1, 1, 8, 1]} -> tensor<1x16x256x512xf16, {order = #NHWC}> {
    %0 = VPU.ShapeCast {shape = [1, 4, 256, 8192]}
        inputs(%arg0 : tensor<1x16x256x2048xf16, {order = #NHWC}>)
       -> tensor<1x4x256x8192xf16, {order = #NHWC}>
    %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 4, 256, 2048]
        : tensor<1x4x256x8192xf16, {order = #NHWC}> to tensor<1x4x256x2048xf16, {order = #NHWC}>
    %2 = VPU.ShapeCast {shape = [1, 16, 256, 512]}
        inputs(%1 : tensor<1x4x256x2048xf16, {order = #NHWC}>)
       -> tensor<1x16x256x512xf16, {order = #NHWC}>
    %3 = VPU.NCE.Eltwise(%2, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        is_inplace = true,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>
    } -> tensor<1x16x256x512xf16, {order = #NHWC}>
    VPU.Yield %3
  }

  return %vf_add : tensor<1x16x256x512xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>

  // CHECK:       [[VF:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[ARG_0:%[^:]+]]: tensor<1x16x256x2048xf16, {order = #NHWC}>,
  // CHECK-SAME:    [[CST]] as [[ARG_1:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>,
  // CHECK-SAME:    [[ADDEND]] as [[ARG_2:%[^:]+]]: tensor<1x16x256x512xf16, {order = #NHWC}>)
  // CHECK:       VPU.NCE.Convolution([[ARG_0]], [[ARG_1]])
  // CHECK:       VPU.ShapeCast
  // CHECK:       VPU.Slice
  // CHECK:       [[SHAPECAST:%.+]] = VPU.ShapeCast
  // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SHAPECAST]], [[ARG_2]])
  // CHECK:       VPU.Yield [[ELTWISE]]

  // CHECK: return [[VF]] : tensor<1x16x256x512xf16, {order = #NHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @MergeMatMulChainSplitOverGroup
// CHECK-SAME: [[INPUT:%.+]]: tensor<256x1x64x16x16xf16, {order = #GNHWC}>
func.func @MergeMatMulChainSplitOverGroup(
    %input: tensor<256x1x64x16x16xf16, {order = #GNHWC}>)
    -> tensor<256x1x256x16x16xf32, {order = #GNHWC}> {
  %cst_0 = const.Declare tensor<256x128x64x1x1xf16, {order = #GNHWC}> = dense<0.0> : tensor<256x128x64x1x1xf16>, [#const.Reorder<#GNHWC>]
  %cst = const.Declare tensor<256x256x128x1x1xf16, {order = #GNHWC}> = dense<0.0> : tensor<256x256x128x1x1xf16>, [#const.Reorder<#GNHWC>]
  %matmul0 = VPU.VerticalFusion (
      %input as %arg1: tensor<256x1x64x16x16xf16, {order = #GNHWC}>,
      %cst_0 as %arg2: tensor<256x128x64x1x1xf16, {order = #GNHWC}>) attributes {tilingStrategy = [3, 1, 1, 1, 1]} -> tensor<256x1x128x16x16xf16, {order = #GNHWC}> {
    %0 = VPU.NCE.MatMul(%arg1, %arg2) rawFilterShape [256, 128, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,
            strides = [1, 1],
            resultSegmentSizes = array<i32: 1, 0, 0, 0>
    } -> tensor<256x1x128x16x16xf16, {order = #GNHWC}>
    VPU.Yield %0
  }
  %matmul1 = VPU.VerticalFusion (
      %matmul0 as %arg1: tensor<256x1x128x16x16xf16, {order = #GNHWC}>,
      %cst as %arg2: tensor<256x256x128x1x1xf16, {order = #GNHWC}>) attributes {tilingStrategy = [10, 1, 1, 1, 1]} -> tensor<256x1x256x16x16xf32, {order = #GNHWC}> {
    %1 = VPU.NCE.MatMul(%arg1, %arg2) rawFilterShape [256, 256, 128, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,
            strides = [1, 1],
            resultSegmentSizes = array<i32: 1, 0, 0, 0>
    } -> tensor<256x1x256x16x16xf32, {order = #GNHWC}>
    VPU.Yield %1
  }

  return %matmul1 : tensor<256x1x256x16x16xf32, {order = #GNHWC}>

  // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<256x128x64x1x1xf16, {order = #GNHWC}>
  // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<256x256x128x1x1xf16, {order = #GNHWC}>

  // CHECK: [[VF:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[ARG_IN:%.+]]: tensor<256x1x64x16x16xf16, {order = #GNHWC}>
  // CHECK-SAME: [[CST_0]] as [[ARG_W0:%[^:]+]]
  // CHECK-SAME: [[CST_1]] as [[ARG_W1:%[^:]+]]
  // CHECK: [[MM0:%.+]] = VPU.NCE.MatMul([[ARG_IN]], [[ARG_W0]])
  // CHECK: [[MM1:%.+]] = VPU.NCE.MatMul([[MM0]], [[ARG_W1]])
  // CHECK: VPU.Yield [[MM1]]
  // CHECK: return [[VF]] : tensor<256x1x256x16x16xf32, {order = #GNHWC}>
}


// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!qOutputEltwise = !quant.uniform<u8:f16, 0.0078431372549019607>
!qOutputEltwiseVfBlock = !quant.uniform<u8:f16, 0.0039215686274509803>
!qWeight = !quant.uniform<u8:f16, 0.025612333709118414:146>
!qOutput = !quant.uniform<u8:f16, 0.019869573443543679:116>

// CHECK-LABEL: @MergePermuteCastEltwiseWithQuantizeCastMatMul
func.func @MergePermuteCastEltwiseWithQuantizeCastMatMul(
    %arg0: tensor<1x12x256x256xf16, {order = #NHCW}>)
    -> tensor<12x1x64x64x4x!qOutput, {order = #GNHWC}> {
    %weights = const.Declare tensor<12x64x256x1x1x!qWeight, {order = #GNHWC}> = dense<1> : tensor<12x64x256x1x1xui8>, [#const.CastElemType<!qWeight>, #const.Reorder<#GNHWC>]

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x12x256x256xf16, {order = #NHCW}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x256x256x12x!qOutputEltwise, {order = #NWHC}> {
      %1 = VPU.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW}
          : tensor<1x12x256x256xf16, {order = #NHCW}> -> tensor<1x256x256x12xf16, {order = #NHWC}>
      %2 = VPU.NCE.Eltwise(%1, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          op_type = #VPU.eltwise_type<ADD>,
          ppe = #VPU.PPEStub<>
      } -> tensor<1x256x256x12x!qOutputEltwise, {order = #NWHC}>
      VPU.Yield %2
    }

    %3 = VPU.VerticalFusion (
        %0 as %arg1: tensor<1x256x256x12x!qOutputEltwise, {order = #NWHC}>,
        %weights as %arg2: tensor<12x64x256x1x1x!qWeight, {order = #GNHWC}>
    ) attributes {tilingStrategy = [1, 1, 1, 1, 1]} -> tensor<12x1x64x64x4x!qOutput, {order = #GNHWC}> {
      %4 = VPU.QuantizeCast(%arg1) {dstElemType = !qOutputEltwiseVfBlock}
          : tensor<1x256x256x12x!qOutputEltwise, {order = #NWHC}> -> tensor<1x256x256x12x!qOutputEltwiseVfBlock, {order = #NWHC}>
      %5 = VPU.PermuteCast(%4) {dst_order = #NCHW, mem_perm = #NCHW}
          : tensor<1x256x256x12x!qOutputEltwiseVfBlock, {order = #NWHC}> -> tensor<1x12x256x256x!qOutputEltwiseVfBlock>
      %6 = VPU.AffineReshape(%5) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [12, 256, 256, 1, 1]}
          : tensor<1x12x256x256x!qOutputEltwiseVfBlock> -> tensor<12x256x256x1x1x!qOutputEltwiseVfBlock>
      %7 = VPU.PermuteCast(%6) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>}
          : tensor<12x256x256x1x1x!qOutputEltwiseVfBlock> -> tensor<12x1x256x256x1x!qOutputEltwiseVfBlock, {order = #GNHWC}>
      %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [12, 1, 256, 64, 4]}
          : tensor<12x1x256x256x1x!qOutputEltwiseVfBlock, {order = #GNHWC}> -> tensor<12x1x256x64x4x!qOutputEltwiseVfBlock, {order = #GNHWC}>
      %9 = VPU.NCE.MatMul(%8, %arg2) rawFilterShape [12, 64, 256, 1, 1] {
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
      } -> tensor<12x1x64x64x4x!qOutput, {order = #GNHWC}>
      VPU.Yield %9
    }

    return %3 : tensor<12x1x64x64x4x!qOutput, {order = #GNHWC}>

    // CHECK:     [[VF:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 4, 1]
    // CHECK:            [[PERM_CAST0:%.+]] = VPU.PermuteCast
    // CHECK:            [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[PERM_CAST0]]
    // CHECK-NOT: VPU.VerticalFusion
    // CHECK:            [[Q_CAST:%.+]] = VPU.QuantizeCast([[ELTWISE]])
    // CHECK:            [[PERM_CAST1:%.+]] = VPU.PermuteCast([[Q_CAST]])
    // CHECK:            [[AFFINE_RESHAPE0:%.+]] = VPU.AffineReshape([[PERM_CAST1]])
    // CHECK:            [[PERM_CAST2:%.+]] = VPU.PermuteCast([[AFFINE_RESHAPE0]])
    // CHECK:            [[AFFINE_RESHAPE1:%.+]] = VPU.AffineReshape([[PERM_CAST2]])
    // CHECK:            [[MATMUL:%.+]] = VPU.NCE.MatMul([[AFFINE_RESHAPE1]]
    // CHECK:            VPU.Yield [[MATMUL]]

    // CHECK:       return [[VF]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>


// Test: eltwise has two VF parents.
// First parent: VF_convert (Convert f32->f16, SplitOverKernel) — incompatible MC strategy with eltwise.
// Second parent: VF_conv (NCE.Convolution, SplitOverKernel) — the "other parent".
// Expected: eltwise cannot merge with VF_convert, so it merges with VF_conv instead.
// CHECK-LABEL: @MergeEltwiseWithTheOtherParent
func.func @MergeEltwiseWithTheOtherParent(
    %arg0: tensor<1x1x1x4096xf16>,
    %arg1: tensor<5504x4096x1x1x!qElemType, {order = #NHWC}>,
    %arg2: tensor<1x5504x1x1xf32, {order = #NHWC}>
) -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<1x1x1x4096xf16> = dense<1.0> : tensor<1x1x1x4096xf16>
  %0 = VPU.RMS(%arg0, %cst_0) {eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x4096xf16>, tensor<1x1x1x4096xf16> -> tensor<1x1x1x4096xf16>
  %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 4096, 1, 1]} : tensor<1x1x1x4096xf16> -> tensor<1x4096x1x1xf16>
  %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4096x1x1xf16> -> tensor<1x4096x1x1xf16, {order = #NHWC}>

  // First VF parent of eltwise: Convert (f32->f16) with SplitOverKernel.
  // This MC strategy is incompatible with the eltwise (Clustering), so eltwise
  // should skip this parent and try merging with the other parent (Conv) instead.
  %3 = VPU.VerticalFusion (%arg2 as %arg3: tensor<1x5504x1x1xf32, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
    %inner = VPU.Convert(%arg3) {dstElemType = f16,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
            : tensor<1x5504x1x1xf32, {order = #NHWC}> -> tensor<1x5504x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  // Second VF parent of eltwise: NCE.Convolution with SplitOverKernel.
  %4 = VPU.VerticalFusion (%2 as %arg3: tensor<1x4096x1x1xf16, {order = #NHWC}>,
                           %arg1 as %arg4: tensor<5504x4096x1x1x!qElemType, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 8, 1, 1]} -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg3, %arg4) rawFilterShape [5504, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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
            strides = [1, 1]} : tensor<1x4096x1x1xf16, {order = #NHWC}>, tensor<5504x4096x1x1x!qElemType, {order = #NHWC}> -> tensor<1x5504x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  // Eltwise: first operand = VF_convert (SOK, incompatible), second operand = VF_conv.
  // The pass should merge eltwise with the second (other) parent: VF_conv.
  %5 = VPU.VerticalFusion (%3 as %arg3: tensor<1x5504x1x1xf16, {order = #NHWC}>,
                           %4 as %arg4: tensor<1x5504x1x1xf16, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Eltwise(%arg3, %arg4) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>} -> tensor<1x5504x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  return %5 : tensor<1x5504x1x1xf16, {order = #NHWC}>

  // VF_convert stays separate because its SplitOverKernel strategy is incompatible.
  // CHECK:  [[RMS:%.+]] = VPU.RMS
  // CHECK:  [[RESHAPE:%.+]] = VPU.AffineReshape([[RMS]])
  // CHECK:  [[PERMUTE:%.+]] = VPU.PermuteCast([[RESHAPE]])
  // CHECK:  [[VF_CONVERT:%.+]] = VPU.VerticalFusion
  // CHECK:  VPU.Convert
  // Conv + Eltwise merge into one VF (eltwise merges with the "other parent").
  // CHECK:  [[VF_CONV_ELTWISE:%.+]] = VPU.VerticalFusion ([[PERMUTE]]
  // CHECK:  VPU.NCE.Convolution
  // CHECK:  VPU.NCE.Eltwise
  // CHECK:  return [[VF_CONV_ELTWISE]] : tensor<1x5504x1x1xf16, {order = #NHWC}>
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qPoolOut = !quant.uniform<u8:f16, 0.12574929630055148>
!qW64x512 = !quant.uniform<u8:f16, 0.029752286275227864:128>

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FuseMultiplyWithStrategyChangeAndBroadcastedInput0
// CHECK-SAME: [[BROADCAST_IN:%.+]]: tensor<1x1x1x500xf16, {order = #NHWC}>
// CHECK-SAME: [[FEATURE_IN:%.+]]: tensor<1x512x4x500xf16, {order = #NHWC}>
func.func @FuseMultiplyWithStrategyChangeAndBroadcastedInput0(
    %arg0: tensor<1x1x1x500xf16, {order = #NHWC}>,
    %arg1: tensor<1x512x4x500xf16, {order = #NHWC}>,
    %arg2: tensor<64x512x1x1x!qW64x512, {order = #NHWC}>
) -> tensor<1x64x4x500xf16, {order = #NHWC}> {

  // VF1: SW Multiply — arg0 (H=1, C=1) broadcasts over H=4, C=512 of arg1; SOW.
  %0 = VPU.VerticalFusion (%arg0 as %in0: tensor<1x1x1x500xf16, {order = #NHWC}>,
                            %arg1 as %in1: tensor<1x512x4x500xf16, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 2, 1, 1]}
                           -> tensor<1x512x4x500xf16, {order = #NHWC}> {
    %out = VPU.Multiply(%in0, %in1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>
    } : tensor<1x1x1x500xf16, {order = #NHWC}>, tensor<1x512x4x500xf16, {order = #NHWC}>
      -> tensor<1x512x4x500xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // VF2: NCE.AveragePool — quantizes the Multiply output; SOH.
  %1 = VPU.VerticalFusion (%0 as %in0: tensor<1x512x4x500xf16, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 4, 1, 1]}
                           -> tensor<1x512x4x500x!qPoolOut, {order = #NHWC}> {
    %out = VPU.NCE.AveragePool(%in0) {
            kernel_size = [1, 1],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
    } -> tensor<1x512x4x500x!qPoolOut, {order = #NHWC}>
    VPU.Yield %out
  }

  // VF3: NCE.Convolution 512->64; SOH. Strategy changes from SOW (Multiply) to SOH.
  %2 = VPU.VerticalFusion (%1 as %in0: tensor<1x512x4x500x!qPoolOut, {order = #NHWC}>,
                            %arg2 as %in1: tensor<64x512x1x1x!qW64x512, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x64x4x500xf16, {order = #NHWC}> {
    %out = VPU.NCE.Convolution(%in0, %in1) rawFilterShape [64, 512, 1, 1] {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
    } : tensor<1x512x4x500x!qPoolOut, {order = #NHWC}>, tensor<64x512x1x1x!qW64x512, {order = #NHWC}>
        -> tensor<1x64x4x500xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  return %2 : tensor<1x64x4x500xf16, {order = #NHWC}>

  // All three VF blocks merge into one despite the MC strategy change (SOW->SOH)
  // that the broadcast Multiply introduces.
  // CHECK:       [[VF:%.+]] = VPU.VerticalFusion
  // CHECK:           VPU.Multiply
  // CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
  // CHECK:           VPU.NCE.AveragePool
  // CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
  // CHECK:           VPU.NCE.Convolution
  // CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
  // CHECK:           VPU.Yield
  // CHECK-NOT:   VPU.VerticalFusion
  // CHECK:       return [[VF]] : tensor<1x64x4x500xf16, {order = #NHWC}>
}
