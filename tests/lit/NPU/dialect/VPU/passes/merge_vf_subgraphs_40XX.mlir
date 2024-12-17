//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @VerticalFuseConvSoftmaxConv(
    %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
    %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
    %cst_1 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

    %0 = VPU.VerticalFusion (
      %arg0 as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %cst_0 as %arg6: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 9, 1]}
            -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg4, %arg5, %arg6) {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          opaque_ppe = #VPU.PPEStub<>,
          rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    %1 = VPU.VerticalFusion (
      %0 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 8, 1]}
            -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.SoftMax(%arg4) {
          axisInd = 1 : i64,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}>
            -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    %2 = VPU.VerticalFusion (
      %1 as %arg4: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
      %arg2 as %arg5: tensor<48x4096x1x1xf16, {order = #NHWC}>,
      %cst_1 as %arg6: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 15, 1]}
            -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg4, %arg5, %arg6) {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          opaque_ppe = #VPU.PPEStub<>,
          rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %2 : tensor<1x48x1024x4xf16, {order = #NHWC}>

    //CHECK: [[VF:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: tilingStrategy = [1, 1, 19, 1]
    //CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution
    //CHECK:    [[SOFTMAX:%.+]] = VPU.SoftMax
    //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution

    //CHECK: return [[VF]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MergeSubgraphsWithCompatibleDistributedTensorType(%arg0: tensor<1x96x176x176xf16, {order = #NHWC}>) -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    %cst_1 = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.VerticalFusion (
        %arg0 as %arg1: tensor<1x96x176x176xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]}
             -> tensor<1x96x176x176xf16, {order = #NHWC}> {
        %2 = VPU.Swish(%arg1)
            {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
            tensor<1x96x176x176xf16, {order = #NHWC}>
                -> tensor<1x96x176x176xf16, {order = #NHWC}>
        VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (
        %0 as %arg1: tensor<1x96x176x176xf16, {order = #NHWC}>,
        %cst as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>,
        %cst_0 as %arg3: tensor<96x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
             -> tensor<1x96x176x176xf16, {order = #NHWC}> {
        %3 = VPU.NCE.DepthConvolution(
            %arg1, %arg2, %arg3) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            opaque_ppe = #VPU.PPEStub<>,
            rawFilterShape = [96, 1, 1, 1],
            strides = [1, 1]
        } -> tensor<1x96x176x176xf16, {order = #NHWC}>
        VPU.Yield %3
    }

    return %1 : tensor<1x96x176x176xf16, {order = #NHWC}>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x96x176x176xf16, {order = #NHWC}>,
    //CHECK-SAME:                        %cst as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>,
    //CHECK-SAME:                        %cst_0 as %arg3: tensor<96x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    //CHECK:      [[SWISH0:%.+]] = VPU.Swish(%arg1) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x176x176xf16, {order = #NHWC}> -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:      [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[SWISH0]], %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, opaque_ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 1, 1, 1], strides = [1, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:        VPU.Yield [[DWCONV0]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x96x176x176xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BuildSubgraphCTilingEltwise
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x512x12x512xf16, {order = #NHWC}>
func.func @BuildSubgraphCTilingEltwise(%arg0: tensor<1x512x12x512xf16, {order = #NHWC}>) -> tensor<1x512x12x512xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x512x12x512xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 3, 1, 1]} -> tensor<1x512x12x512xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Eltwise(%arg1, %arg1)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, opaque_ppe = #VPU.PPEStub<>} -> tensor<1x512x12x512xf16, {order = #NHWC}>
      VPU.Yield %2
  }

  %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x512x12x512xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x512x12x512xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Eltwise(%arg2, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, opaque_ppe = #VPU.PPEStub<>} -> tensor<1x512x12x512xf16, {order = #NHWC}>
      VPU.Yield %2
  }
  return %1 : tensor<1x512x12x512xf16, {order = #NHWC}>

  //CHECK:  VPU.VerticalFusion ([[INPUT]] as [[ARG1:%.+]]: tensor<1x512x12x512xf16, {order = #NHWC}>)
  //CHECK-SAME: attributes {tilingStrategy = [1, 4, 1, 1]}
  //CHECK:  [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[ARG1]], [[ARG1]])
  //CHECK-NEXT:  [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[ELTWISE_0]], [[ELTWISE_0]])
  //CHECK-NEXT:  VPU.Yield [[ELTWISE_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotBuildSubgraphCTilingNCEOpwithWeights
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x512x12x512xf16, {order = #NHWC}>
func.func @NotBuildSubgraphCTilingNCEOpwithWeights(%arg0: tensor<1x512x12x512xf16, {order = #NHWC}>) -> tensor<1x512x12x512xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<512x512x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<512x512xf32>, [#const.Reshape<[512, 512, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_1 = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>

  %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x512x12x512xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 3, 1, 1]} -> tensor<1x512x12x512xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Eltwise(%arg1, %arg1)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, opaque_ppe = #VPU.PPEStub<>} -> tensor<1x512x12x512xf16, {order = #NHWC}>
      VPU.Yield %2
  }

  %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x512x12x512xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<512x512x1x1xf16, {order = #NHWC}>, %cst_1 as %arg3: tensor<512x1x1x4xsi32>) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x512x12x512xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, opaque_ppe = #VPU.PPEStub<>, rawFilterShape = [512, 512, 1, 1], strides = [1, 1]} -> tensor<1x512x12x512xf16, {order = #NHWC}>
      VPU.Yield %2
  }
  return %1 : tensor<1x512x12x512xf16, {order = #NHWC}>

  //CHECK:  [[VF_0:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[ARG1:%.+]]: tensor<1x512x12x512xf16, {order = #NHWC}>)
  //CHECK-SAME: attributes {tilingStrategy = [1, 3, 1, 1]}
  //CHECK:  [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]] as [[ARG1:%.+]]: tensor<1x512x12x512xf16, {order = #NHWC}>,
  //CHECK-SAME: attributes {tilingStrategy = [1, 4, 1, 1]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PipeliningWithoutCostCondition
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x768x128x4xf16, {order = #NHWC}>
func.func @PipeliningWithoutCostCondition(%arg0: tensor<1x768x128x4xf16, {order = #NHWC}>) -> tensor<1x3072x128x4xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<3072x768x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<3072x768xf32>, [#const.Reshape<[3072, 768, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %cst_1 = const.Declare tensor<3072x1x1x4xsi32> = dense<1> : tensor<3072x1x1x4xsi32>

  %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x768x128x4xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<3072x768x1x1xf16, {order = #NHWC}>, %cst_1 as %arg3: tensor<3072x1x1x4xsi32>) attributes {tilingStrategy = [1, 8, 1, 1]} -> tensor<1x3072x128x4xf16, {order = #NHWC}> {
    %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, opaque_ppe = #VPU.PPEStub<>, rawFilterShape = [3072, 768, 1, 1], strides = [1, 1]} -> tensor<1x3072x128x4xf16, {order = #NHWC}>
    VPU.Yield %2
  }

  %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x3072x128x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x3072x128x4xf16, {order = #NHWC}> {
    %2 = VPU.Gelu(%arg1)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x3072x128x4xf16, {order = #NHWC}> -> tensor<1x3072x128x4xf16, {order = #NHWC}>
    VPU.Yield %2
  }

  return %1 : tensor<1x3072x128x4xf16, {order = #NHWC}>

    // E#128159 regions should be merged together

    //CHECK: [[VF0:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: tilingStrategy = [1, 8, 1, 1]
    //CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution
    //CHECK:    [[GELU:%.+]] = VPU.Gelu

    //CHECK: return [[VF0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>

func.func @BuildSubgraphMinimalRequirements(%arg0: tensor<1x768x128x4xf16, {order = #NHWC}>) -> tensor<1x3072x128x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<3072x768x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<3072x768x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<3072x1x1x4xsi32> = dense<1> : tensor<3072x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x768x128x4xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<3072x768x1x1x!qElemType, {order = #NHWC}>, %cst_1 as %arg4: tensor<3072x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 5, 1]} -> tensor<1x3072x128x4xf16, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, opaque_ppe = #VPU.PPEStub<>, rawFilterShape = [3072, 768, 1, 1], strides = [1, 1]} -> tensor<1x3072x128x4xf16, {order = #NHWC}>
      VPU.Yield %4
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x3072x128x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x3072x128x4xf16, {order = #NHWC}> {
      %4 = VPU.Gelu(%arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x3072x128x4xf16, {order = #NHWC}> -> tensor<1x3072x128x4xf16, {order = #NHWC}>
      VPU.Yield %4
    }
    return %1 : tensor<1x3072x128x4xf16, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x768x128x4xf16, {order = #NHWC}>, %cst as %arg2: tensor<3072x768x1x1x!qElemType, {order = #NHWC}>, %cst_0 as %arg3: tensor<3072x1x1x4xsi32>)
    //CHECK-SAME: attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x3072x128x4xf16, {order = #NHWC}> {
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    //CHECK: [[GELU:%.+]] = VPU.Gelu([[CONV0]])
    //CHECK:  VPU.Yield [[GELU]]
    //CHECK: return [[VERTICAL_FUSION]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

module @executors {
    IE.TileResource 4 of @NCE at 1.850000e+03 MHz


  // CHECK-LABEL: @BuildSubgraphWithOtherAxis
  // CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x48x512x128xf16, {order = #NHWC}>
  // CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x12x512x512xf16>
  func.func @BuildSubgraphWithOtherAxis(%arg0: tensor<1x48x512x128xf16, {order = #NHWC}>, %arg1: tensor<1x12x512x512xf16>) -> tensor<1x12x512x512xf16, {order = #NHCW}> {
    %shapecast = VPU.ShapeCast {shape = [1, 512, 12, 512]} inputs(%arg0 : tensor<1x48x512x128xf16, {order = #NHWC}>) -> tensor<1x512x12x512xf16, {order = #NHWC}>
    %permutecast0 = VPU.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x12x512x512xf16> -> tensor<1x512x12x512xf16, {order = #NHWC}>

    %0 = VPU.VerticalFusion (%shapecast as %arg3: tensor<1x512x12x512xf16, {order = #NHWC}>,
                            %permutecast0 as %arg4: tensor<1x512x12x512xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 3, 1, 1]} -> tensor<1x512x12x512xf16, {order = #NHWC}> {
        %2 = VPU.NCE.Eltwise(%arg3, %arg4) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              op_type = #VPU.eltwise_type<ADD>, opaque_ppe = #VPU.PPEStub<>}
              -> tensor<1x512x12x512xf16, {order = #NHWC}>
        VPU.Yield %2
    }

    %1 = VPU.VerticalFusion (%0 as %arg3: tensor<1x512x12x512xf16, {order = #NHWC}>, %permutecast0 as %arg4: tensor<1x512x12x512xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x512x12x512xf16, {order = #NWHC}> {
        %2 = VPU.NCE.Eltwise(%arg3, %arg4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>, opaque_ppe = #VPU.PPEStub<>}
            -> tensor<1x512x12x512xf16, {order = #NWHC}>
        VPU.Yield %2
    }

    %permutecast2 = VPU.PermuteCast(%1) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x512x12x512xf16, {order = #NWHC}> -> tensor<1x12x512x512xf16, {order = #NHCW}>


    return %permutecast2 : tensor<1x12x512x512xf16, {order = #NHCW}>

    //CHECK:      [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 512, 12, 512]}
    //CHECK:      [[PERMUTECAST0:%.+]] = VPU.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW}
    //CHECK:      VPU.VerticalFusion ([[SHAPECAST]] as [[ARG0:%.+]]: tensor<1x512x12x512xf16, {order = #NHWC}>, [[PERMUTECAST0]] as [[ARG1:%.+]]: tensor<1x512x12x512xf16, {order = #NHWC}>
    //CHECK-SAME: attributes {tilingStrategy = [1, 1, 1, 6]}
    //CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[ARG0]], [[ARG1]])
    //CHECK-NEXT:  [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[ELTWISE_0]], [[ARG1]])
    //CHECK-NEXT:  VPU.Yield [[ELTWISE_1]]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BuildSubgraphWhenSpillIsBetweenVFBlocks
// CHECK-SAME:      ([[INPUT_0:%.+]]: tensor<1x64x160x160xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x160x160xf16, {order = #NHWC}>
func.func @BuildSubgraphWhenSpillIsBetweenVFBlocks(
            %arg0: tensor<1x64x160x160xf16, {order = #NHWC}>,
            %arg1: tensor<1x64x160x160xf16, {order = #NHWC}>) -> tensor<1x64x160x160xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<64x64x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x3x3xf16>, [#const.Reorder<#NHWC>]
  %cst_1 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

  %0 = VPU.VerticalFusion (
            %arg0 as %arg2: tensor<1x64x160x160xf16, {order = #NHWC}>,
            %cst_0 as %arg3: tensor<64x64x3x3xf16, {order = #NHWC}>,
            %cst_1 as %arg4: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x64x160x160xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            opaque_ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 64, 3, 3],
            strides = [1, 1]} -> tensor<1x64x160x160xf16, {order = #NHWC}>
      VPU.Yield %2
  }

  %1 = VPU.VerticalFusion (
            %0 as %arg2: tensor<1x64x160x160xf16, {order = #NHWC}>,
            %arg1 as %arg3: tensor<1x64x160x160xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x160x160xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Eltwise(%arg2, %arg3) {
            is_inplace = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            opaque_ppe = #VPU.PPEStub<>} -> tensor<1x64x160x160xf16, {order = #NHWC}>
      VPU.Yield %2
  }

  return %1 : tensor<1x64x160x160xf16, {order = #NHWC}>

  //CHECK: [[VF:%.+]] = VPU.VerticalFusion
  //CHECK-SAME: tilingStrategy = [1, 1, 1, 2]
  //CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution
  //CHECK:    [[ELTWISE:%.+]] = VPU.NCE.Eltwise

  //CHECK: return [[VF]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @BuildSubgraphTwoConvTileChannelVF
//CHECK-SAME:  [[INPUT:%.+]]: tensor<1x128x112x112xf16, {order = #NHWC}>
func.func @BuildSubgraphTwoConvTileChannelVF(%arg0: tensor<1x128x112x112xf16, {order = #NHWC}>) -> tensor<1x128x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x128x1x1xf16, {order = #NHWC}>
    %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x128x112x112xf16, {order = #NHWC}>,
                             %cst_0 as %arg3: tensor<128x128x1x1xf16, {order = #NHWC}>,
                             %cst_1 as %arg4: tensor<128x1x1x4xsi32>)
                  attributes {tilingStrategy = [1, 3, 1, 1]} -> tensor<1x128x112x112xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) {
              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              opaque_ppe = #VPU.PPEStub<>,
              rawFilterShape = [128, 128, 1, 1], strides = [1, 1]
            } -> tensor<1x128x112x112xf16, {order = #NHWC}>
      VPU.Yield %2
    }

    %1 = VPU.VerticalFusion (%0 as %arg5: tensor<1x128x112x112xf16, {order = #NHWC}>,
                             %cst_0 as %arg6: tensor<128x128x1x1xf16, {order = #NHWC}>,
                             %cst_1 as %arg7: tensor<128x1x1x4xsi32>)
                  attributes {tilingStrategy = [1, 3, 1, 1]} -> tensor<1x128x112x112xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg5, %arg6, %arg7) {
              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              opaque_ppe = #VPU.PPEStub<>,
              rawFilterShape = [128, 128, 1, 1], strides = [1, 1]
            } -> tensor<1x128x112x112xf16, {order = #NHWC}>
      VPU.Yield %2
    }

    return %1: tensor<1x128x112x112xf16, {order = #NHWC}>

    //CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>
    //CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<128x1x1x4xsi32>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (
    //CHECK-SAME:                        [[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x128x112x112xf16, {order = #NHWC}>,
    //CHECK-SAME:                        [[WEIGHTS]] as [[INNER_ARG2:[^:]+]]: tensor<128x128x1x1xf16, {order = #NHWC}>,
    //CHECK-SAME:                        [[WT]] as [[INNER_ARG3:[^:]+]]: tensor<128x1x1x4xsi32>
    //CHECK-SAME:             attributes {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x128x112x112xf16, {order = #NHWC}> {
    //CHECK:      [[NCE0:%.+]] = VPU.NCE.Convolution([[INNER_ARG1]], [[INNER_ARG2]], [[INNER_ARG3]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:      [[NCE1:%.+]] = VPU.NCE.Convolution([[NCE0]], [[INNER_ARG2]], [[INNER_ARG3]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x128x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType =  !quant.uniform<u8:f16, 0.34327301324582565:126>
!qElemType1 = !quant.uniform<u8:f16, 0.33446191525926777:126>
!qElemType2 = !quant.uniform<u8:f16, 0.045267969019272748:126>
!qElemType3 = !quant.uniform<u8:f16, 40.502450980392155:255>
//CHECK:      func.func @BuildSubgraphWithViewLikeOps([[INPUT:%.+]]: tensor<1x512x12x512x!qElemType, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x512x12x512x!qElemType1, {order = #NHWC}>) -> tensor<1x512x12x512xf16, {order = #NHWC}> {
func.func @BuildSubgraphWithViewLikeOps(
  %arg0: tensor<1x512x12x512x!qElemType, {order = #NHWC}>,
  %arg1: tensor<1x512x12x512x!qElemType3, {order = #NHWC}>) -> tensor<1x512x12x512xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x512x12x512x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x512x12x512xf16, {order = #NHWC}> {
    %3 = VPU.NCE.AveragePool(%arg2) {kernel_size = [1, 1],  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, opaque_ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x512x12x512xf16, {order = #NHWC}>
    VPU.Yield %3
  }
  %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x512x12x512xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x512x12x512x!qElemType1, {order = #NHWC}> {
    %3 = VPU.NCE.AveragePool(%arg2) {kernel_size = [1, 1],  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, opaque_ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x512x12x512x!qElemType1, {order = #NHWC}>
    VPU.Yield %3
  }
  %2 = VPU.VerticalFusion (%1 as %arg2: tensor<1x512x12x512x!qElemType1, {order = #NHWC}>, %arg1 as %arg3: tensor<1x512x12x512x!qElemType3, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 3]} -> tensor<1x512x12x512xf16, {order = #NHWC}> {
    %3 = VPU.QuantizeCast(%arg2) {dstElemType = !qElemType2} : tensor<1x512x12x512x!qElemType1, {order = #NHWC}> -> tensor<1x512x12x512x!qElemType2, {order = #NHWC}>
    %4 = VPU.NCE.Eltwise(%3, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, opaque_ppe = #VPU.PPEStub<>} -> tensor<1x512x12x512xf16, {order = #NHWC}>
    VPU.Yield %4
  }
    return %2 :tensor<1x512x12x512xf16, {order = #NHWC}>

    //CHECK:	  [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG0:%.+]]: tensor<1x512x12x512x!qElemType, {order = #NHWC}>, [[INPUT1]] as [[INNER_ARG1:%.+]]: tensor<1x512x12x512x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x512x12x512xf16, {order = #NHWC}> {
    //CHECK:      [[AVGPOOL0:%.+]] = VPU.NCE.AveragePool([[INNER_ARG0]]) {kernel_size = [1, 1],  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, opaque_ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x512x12x512xf16, {order = #NHWC}>
    //CHECK:      [[AVGPOOL1:%.+]] = VPU.NCE.AveragePool([[AVGPOOL0]]) {kernel_size = [1, 1],  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, opaque_ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x512x12x512x!qElemType2, {order = #NHWC}>
    //CHECK:      [[QC:%.+]] = VPU.QuantizeCast([[AVGPOOL1]]) {dstElemType = !qElemType3} : tensor<1x512x12x512x!qElemType2, {order = #NHWC}> -> tensor<1x512x12x512x!qElemType3, {order = #NHWC}>
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[QC]], [[INNER_ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, opaque_ppe = #VPU.PPEStub<>} -> tensor<1x512x12x512xf16, {order = #NHWC}>
    //CHECK:      VPU.Yield [[ELTWISE]]

    //CHECK:    return [[VERTICAL_FUSION]] : tensor<1x512x12x512xf16, {order = #NHWC}>
  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.1:128>

// CHECK-LABEL: @DontMergeSpillingWeightsToVF
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x512x26x26x!qElemType, {order = #NHWC}>
func.func @DontMergeSpillingWeightsToVF(%arg0: tensor<1x512x26x26x!qElemType, {order = #NHWC}>) -> tensor<1x1024x13x13x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<256x512x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<256x512x1x1xf16>, [#const.CastElemType<ui8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.VerticalFusion (
            %arg0 as %arg1: tensor<1x1024x13x13x!qElemType, {order = #NHWC}>,
            %cst_0 as %arg2: tensor<512x1024x1x1x!qElemType, {order = #NHWC}>,
            %cst_1 as %arg3: tensor<512x1x1x4xsi32>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x512x13x13x!qElemType, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, opaque_ppe = #VPU.PPEStub<>,
                rawFilterShape = [512, 1024, 1, 1], strides = [1, 1]}
                    -> tensor<1x512x13x13x!qElemType, {order = #NHWC}>
        VPU.Yield %inner
    }
    %cst_2 = const.Declare tensor<1024x512x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<1024x512x3x3xf16>, [#const.CastElemType<ui8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
    %1 = VPU.VerticalFusion (
            %0 as %arg1: tensor<1x512x13x13x!qElemType, {order = #NHWC}>,
            %cst_2 as %arg2: tensor<1024x512x3x3x!qElemType, {order = #NHWC}>,
            %cst_3 as %arg3: tensor<1024x1x1x4xsi32>) attributes {tilingStrategy = [1, 8, 1, 1]} -> tensor<1x1024x13x13x!qElemType, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, opaque_ppe = #VPU.PPEStub<>,
                rawFilterShape = [1024, 512, 3, 3], strides = [1, 1]}
                    -> tensor<1x1024x13x13x!qElemType, {order = #NHWC}>
        VPU.Yield %inner
    }

    return %1 : tensor<1x1024x13x13x!qElemType, {order = #NHWC}>

    // Don't merge because the weights are too big
    //CHECK: [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]] as {{[^:]+}}: tensor<1x1024x13x13x!qElemType, {order = #NHWC}>
    //CHECK-SAME: tilingStrategy = [1, 2, 1, 1]
    //CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution
    //CHECK:    VPU.Yield [[CONV_0]]
    //CHECK: [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]] as {{[^:]+}}: tensor<1x512x13x13x!qElemType, {order = #NHWC}>
    //CHECK-SAME: tilingStrategy = [1, 8, 1, 1]
    //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution
    //CHECK:    VPU.Yield [[CONV_1]]

    //CHECK: return [[VF1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeVFWithoutTilingC
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x64x1x19284xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x64x1x19284xf16, {order = #NHWC}>)
func.func @MergeVFWithoutTilingC(%arg0: tensor<1x64x1x19284xf16, {order = #NHWC}>, %arg1: tensor<1x64x1x19284xf16, {order = #NHWC}>)
             -> tensor<1x64x1x19284xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (
            %arg0 as %arg2: tensor<1x64x1x19284xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x1x19284xf16, {order = #NHWC}> {
        %inner = VPU.Sigmoid(%arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x1x19284xf16, {order = #NHWC}>
               -> tensor<1x64x1x19284xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    %1 = VPU.VerticalFusion (
            %arg1 as %arg2: tensor<1x64x1x19284xf16, {order = #NHWC}>,
            %0 as %arg3: tensor<1x64x1x19284xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x64x1x19284xf16, {order = #NHWC}> {
        %inner = VPU.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x1x19284xf16, {order = #NHWC}>, tensor<1x64x1x19284xf16, {order = #NHWC}>
                   -> tensor<1x64x1x19284xf16, {order = #NHWC}>
        VPU.Yield %inner
    }

    return %1 : tensor<1x64x1x19284xf16, {order = #NHWC}>

    // Don't tiling on C since it has larger stride dma cost when dim C size is small
    //CHECK:	  [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[INPUT0]] as [[INNER_ARG0:%.+]]: tensor<1x64x1x19284xf16, {order = #NHWC}>,
    //CHECK-SAME:                                             [[INPUT1]] as [[INNER_ARG1:%.+]]: tensor<1x64x1x19284xf16, {order = #NHWC}>)
    //CHECK-SAME:            attributes {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x64x1x19284xf16, {order = #NHWC}> {
    //CHECK:    [[SIGMOID:%.*]] = VPU.Sigmoid([[INNER_ARG0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x1x19284xf16, {order = #NHWC}> -> tensor<1x64x1x19284xf16, {order = #NHWC}>
    //CHECK:    [[MULTIPLY:%.*]]= VPU.Multiply([[INNER_ARG1]], [[SIGMOID]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x64x1x19284xf16, {order = #NHWC}>, tensor<1x64x1x19284xf16, {order = #NHWC}> -> tensor<1x64x1x19284xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield [[MULTIPLY]]
    //CHECK:    return [[VERTICAL_FUSION]] : tensor<1x64x1x19284xf16, {order = #NHWC}>
}