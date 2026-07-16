//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=false" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

// CHECK-LABEL: @NotBuildNotTiledSubgraph
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>)
func.func @NotBuildNotTiledSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
          strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
          strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    return %1 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>

    //CHECK:      [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST]] as [[ARG_2:%[^:]+]]: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>)
    //CHECK-SAME:                         attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]])
    //CHECK:      VPU.Yield [[CONV0]]

    //CHECK:      [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion ([[VERTICAL_FUSION0]] as [[ARG_1:%[^:]+]]: tensor<1x32x256x256x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_0]] as [[ARG_2:%[^:]+]]: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>)
    //CHECK-SAME:                         attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]])
    //CHECK:      VPU.Yield [[CONV1]]

    //CHECK: return [[VERTICAL_FUSION1]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotBuildLargeWeights
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x256x26x26xf16, {order = #NHWC}>)
func.func @NotBuildLargeWeights(%arg0: tensor<1x256x26x26xf16, {order = #NHWC}>) -> tensor<1x256x26x26xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %cst_1 = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (
        %arg0 as %arg1: tensor<1x256x26x26xf16, {order = #NHWC}>,
        %cst_1 as %arg2: tensor<512x256x3x3xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 4, 1]}
            -> tensor<1x512x26x26xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [512, 256, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad =  #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]} : tensor<1x256x26x26xf16, {order = #NHWC}>, tensor<512x256x3x3xf16, {order = #NHWC}> -> tensor<1x512x26x26xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (
        %0 as %arg1: tensor<1x512x26x26xf16, {order = #NHWC}>,
        %cst as %arg2: tensor<256x512x3x3xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 4, 1]}
            -> tensor<1x256x26x26xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [256, 512, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]} : tensor<1x512x26x26xf16, {order = #NHWC}>, tensor<256x512x3x3xf16, {order = #NHWC}> -> tensor<1x256x26x26xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %1 : tensor<1x256x26x26xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x256x3x3xf16>

    //CHECK: [[VF_0:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x256x26x26xf16, {order = #NHWC}>,
    //CHECK-SAME: [[CST_0]] as [[ARG_2:%[^:]+]]: tensor<512x256x3x3xf16, {order = #NHWC}>
    //CHECK-SAME: ) attributes {tilingStrategy = [1, 1, 4, 1]}
    //CHECK-SAME: -> tensor<1x512x26x26xf16, {order = #NHWC}>
    //CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]])

    //CHECK: [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]] as [[ARG_1:%[^:]+]]: tensor<1x512x26x26xf16, {order = #NHWC}>,
    //CHECK-SAME: [[CST]] as [[ARG_2:%[^:]+]]: tensor<256x512x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME: attributes {tilingStrategy = [1, 1, 4, 1]}
    //CHECK-SAME: -> tensor<1x256x26x26xf16, {order = #NHWC}>
    //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]])

    //CHECK: return [[VF_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeAdjacentSinSubgraphs
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x448x392xf16, {order = #NHWC}>
func.func @MergeAdjacentSinSubgraphs(%arg0: tensor<1x16x448x392xf16, {order = #NHWC}>) -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x448x392xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x448x392xf16, {order = #NHWC}> {
      %2 = VPU.Sin(%arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x448x392xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x448x392xf16, {order = #NHWC}> {
      %2 = VPU.Sin(%arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %1 : tensor<1x16x448x392xf16, {order = #NHWC}>

    // CHECK:      [[VF:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[ARG_1:[^:]+]]: tensor<1x16x448x392xf16, {order = #NHWC}>)
    // CHECK-SAME: attributes {{{.*}}tilingStrategy = [1, 1, {{[0-9]+}}, 1]}
    // CHECK-SAME: -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    // CHECK:      [[SIN_0:%.+]] = VPU.Sin([[ARG_1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    // CHECK:      [[SIN_1:%.+]] = VPU.Sin([[SIN_0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    // CHECK:      VPU.Yield [[SIN_1]]
    // CHECK:      return [[VF]]
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
    %3 = VPU.NCE.Eltwise(%2, %arg1) {
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

// CHECK-LABEL: @MergeRoPEWithAdd
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x1x64xf16>, [[ARG_1:%[^:]+]]: tensor<1x32x1x64xf16>)
func.func @MergeRoPEWithAdd(%arg0: tensor<1x32x1x64xf16>, %arg1: tensor<1x32x1x64xf16>) -> tensor<1x32x1x64xf16> {
    %cos = const.Declare tensor<1x1x1x64xf16> = dense<1.0> : tensor<1x1x1x64xf16>
    %sin = const.Declare tensor<1x1x1x64xf16> = dense<0.0> : tensor<1x1x1x64xf16>

    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x1x64xf16>, %arg1 as %arg3: tensor<1x32x1x64xf16>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x32x1x64xf16> {
      %2 = VPU.Add(%arg2, %arg3)
         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x64xf16>, tensor<1x32x1x64xf16> -> tensor<1x32x1x64xf16>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x32x1x64xf16>, %cos as %arg3: tensor<1x1x1x64xf16>, %sin as %arg4: tensor<1x1x1x64xf16>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x32x1x64xf16> {
      %2 = VPU.RoPE(%arg2, %arg3, %arg4) {mode = #IE.rope_mode<SPLIT_HALF>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x64xf16>, tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x32x1x64xf16>
      VPU.Yield %2
    }

    return %1 : tensor<1x32x1x64xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x64xf16> = dense<1.000000e+00>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x64xf16> = dense<0.000000e+00>
    // CHECK:       [[VF:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[VF_IN0:%[^:]+]]: tensor<1x32x1x64xf16>,
    // CHECK-SAME:    [[ARG_1]] as [[VF_IN1:%[^:]+]]: tensor<1x32x1x64xf16>,
    // CHECK-SAME:    [[CST]] as [[VF_IN2:%[^:]+]]: tensor<1x1x1x64xf16>,
    // CHECK-SAME:    [[CST_0]] as [[VF_IN3:%[^:]+]]: tensor<1x1x1x64xf16>)
    // CHECK-SAME:    tilingStrategy = [1, 2, 1, 1]
    // CHECK:           [[ADD:%.+]] = VPU.Add([[VF_IN0]], [[VF_IN1]])
    // CHECK:           [[ROPE:%.+]] = VPU.RoPE([[ADD]], [[VF_IN2]], [[VF_IN3]])
    // CHECK-SAME:      mode = #IE.rope_mode<SPLIT_HALF>
    // CHECK:           VPU.Yield [[ROPE]]
    // CHECK:       return [[VF]] : tensor<1x32x1x64xf16>
}

// -----

// CHECK-LABEL: @MergeRMSWithAdd
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x128x1x256xf16>, [[ARG_1:%[^:]+]]: tensor<1x128x1x256xf16>)
func.func @MergeRMSWithAdd(%arg0: tensor<1x128x1x256xf16>, %arg1: tensor<1x128x1x256xf16>) -> tensor<1x128x1x256xf16> {
    %cst = const.Declare tensor<1x1x1x256xf16> = dense<1.0> : tensor<1x1x1x256xf16>

    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x128x1x256xf16>, %arg1 as %arg3: tensor<1x128x1x256xf16>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x128x1x256xf16> {
      %2 = VPU.Add(%arg2, %arg3)
         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x128x1x256xf16>, tensor<1x128x1x256xf16> -> tensor<1x128x1x256xf16>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x128x1x256xf16>, %cst as %arg3: tensor<1x1x1x256xf16>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x128x1x256xf16> {
      %2 = VPU.RMS(%arg2, %arg3) {eps = 1.0E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x128x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x128x1x256xf16>
      VPU.Yield %2
    }

    return %1 : tensor<1x128x1x256xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x256xf16> = dense<1.000000e+00>
    // CHECK:       [[VF:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[VF_IN0:%[^:]+]]: tensor<1x128x1x256xf16>,
    // CHECK-SAME:    [[ARG_1]] as [[VF_IN1:%[^:]+]]: tensor<1x128x1x256xf16>,
    // CHECK-SAME:    [[CST]] as [[VF_IN2:%[^:]+]]: tensor<1x1x1x256xf16>)
    // CHECK-SAME:    tilingStrategy = [1, 2, 1, 1]
    // CHECK:           [[ADD:%.+]] = VPU.Add([[VF_IN0]], [[VF_IN1]])
    // CHECK:           [[RMS:%.+]] = VPU.RMS([[ADD]], [[VF_IN2]])
    // CHECK-SAME:      eps = 9.9999999999999995E-7
    // CHECK:           VPU.Yield [[RMS]]
    // CHECK:       return [[VF]] : tensor<1x128x1x256xf16>
}

// -----

// CHECK-LABEL: @MergeRoundWithAdd
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x64x64xf16>, [[ARG_1:%[^:]+]]: tensor<1x32x64x64xf16>)
func.func @MergeRoundWithAdd(%arg0: tensor<1x32x64x64xf16>, %arg1: tensor<1x32x64x64xf16>) -> tensor<1x32x64x64xf16> {
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x64x64xf16>, %arg1 as %arg3: tensor<1x32x64x64xf16>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x64x64xf16> {
      %2 = VPU.Add(%arg2, %arg3)
         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x64x64xf16>, tensor<1x32x64x64xf16> -> tensor<1x32x64x64xf16>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x32x64x64xf16>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x64x64xf16> {
      %2 = VPU.Round(%arg2) {mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x64x64xf16> -> tensor<1x32x64x64xf16>
      VPU.Yield %2
    }

    return %1 : tensor<1x32x64x64xf16>

    // CHECK:       [[VF:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[VF_IN0:%[^:]+]]: tensor<1x32x64x64xf16>,
    // CHECK-SAME:    [[ARG_1]] as [[VF_IN1:%[^:]+]]: tensor<1x32x64x64xf16>)
    // CHECK-SAME:    tilingStrategy = [1, 1, 2, 1]
    // CHECK:           [[ADD:%.+]] = VPU.Add([[VF_IN0]], [[VF_IN1]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:           [[ROUND:%.+]] = VPU.Round([[ADD]])
    // CHECK-SAME:      mode = #IE.round_mode<HALF_TO_EVEN>
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:           VPU.Yield [[ROUND]]
    // CHECK:       return [[VF]] : tensor<1x32x64x64xf16>
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
