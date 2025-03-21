//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MergeVFWithoutVFPipelining(
    %arg0: tensor<1x48x1024x4xf16, {order = #NHWC}>,
    %arg1: tensor<4096x48x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
    %cst_1 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

    %0 = VPU.VerticalFusion (
      %arg0 as %arg4: tensor<1x48x1024x4xf16, {order = #NHWC}>,
      %arg1 as %arg5: tensor<4096x48x1x1xf16, {order = #NHWC}>,
      %cst_0 as %arg6: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 8, 1, 1]}
            -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg4, %arg5, %arg6) {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
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
          ppe = #VPU.PPEStub<>,
          rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %2 : tensor<1x48x1024x4xf16, {order = #NHWC}>

    //CHECK: [[VF:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: scenario = #VPU.vf_scenario<FULL_PREFETCHING>
    //CHECK-SAME: tilingStrategy = [1, 1, 15, 1]
    //CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution
    //CHECK:    [[SOFTMAX:%.+]] = VPU.SoftMax
    //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution

    //CHECK: return [[VF]]
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 0.17455944734461168:132>
!qElemType1 = !quant.uniform<u8:f16, 0.00174561528598561:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @BuildPipeliningWithWTiling(%arg0: tensor<1x32x101x480x!qElemType0, {order = #NHWC}>, %arg1: tensor<1x32x101x480xf16, {order = #NHWC}>) -> tensor<1x32x101x480xf16, {order = #NHWC}> {

    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x101x480x!qElemType0, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x101x480xf16, {order = #NHWC}> {
      %4 = VPU.NCE.AveragePool(%arg2) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x32x101x480xf16, {order = #NHWC}>
      VPU.Yield %4
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x1x101x1xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x32x101x480xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x32x101x480xf16, {order = #NHWC}> {
      %4 = VPU.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x101x1xf16, {order = #NHWC}>, tensor<1x32x101x480xf16, {order = #NHWC}> -> tensor<1x32x101x480xf16, {order = #NHWC}>
      VPU.Yield %4
    }
    %2 = VPU.VerticalFusion (%1 as %arg2: tensor<1x32x101x480xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x101x480x!qElemType1, {order = #NHWC}> {
      %4 = VPU.NCE.AveragePool(%arg2) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x32x101x480x!qElemType1, {order = #NHWC}>
      VPU.Yield %4
    }
    %3 = VPU.VerticalFusion (%2 as %arg24: tensor<1x32x101x480x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x101x480xf16, {order = #NHWC}> {
      %4 = VPU.NCE.AveragePool(%arg24) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x32x101x480xf16, {order = #NHWC}>
      VPU.Yield %4
    }

    return %3  : tensor<1x32x101x480xf16, {order = #NHWC}>

    //CHECK:  [[VF_0:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:  as [[INPUT0:%.+]]: tensor<1x32x101x480x!qElemType, {order = #NHWC}>
    // CHECK-SAME:  as [[INPUT1:%.+]]: tensor<1x32x101x480xf16, {order = #NHWC}>
    //CHECK-SAME: scenario = #VPU.vf_scenario<FULL_PREFETCHING>
    //CHECK-SAME: tilingStrategy = [1, 1, 1, 3]

    //CHECK: [[POOL0:%.+]] = VPU.NCE.AveragePool([[INPUT0]])
    //CHECK: [[MULT:%.+]] = VPU.Multiply([[POOL0]], [[INPUT1]])
    //CHECK: [[POOL1:%.+]] = VPU.NCE.AveragePool([[MULT]])
    //CHECK: [[POOL2:%.+]] = VPU.NCE.AveragePool([[POOL1]])
    //CHECK: VPU.Yield [[POOL2]]

    //CHECK: return [[VF_0]]  : tensor<1x32x101x480xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeVFWithConsideringEarlyScheduledParent
func.func @MergeVFWithConsideringEarlyScheduledParent(
      %arg0: tensor<1x128x256x4xf16, {order = #NHWC}>,
      %arg1: tensor<1024x128x1x1xf16, {order = #NHWC}>,
      %arg2: tensor<1x1x1024x1024xf32>,
      %arg3: tensor<128x1024x1x1xf16, {order = #NHWC}>)
             -> tensor<1x128x256x4xf16, {order = #NHWC}> {
    // CMX operation 0
    %cst_0 = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
    %cmx_op_0 = VPU.VerticalFusion (
          %arg0 as %arg20: tensor<1x128x256x4xf16, {order = #NHWC}>,
          %arg1 as %arg21: tensor<1024x128x1x1xf16, {order = #NHWC}>,
          %cst_0 as %arg22: tensor<1024x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1024x256x4xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Convolution(%arg20, %arg21, %arg22) {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
          rawFilterShape = [1024, 128, 1, 1],
          strides = [1, 1]
      } -> tensor<1x1024x256x4xf16, {order = #NHWC}>
      VPU.Yield %inner
    }

    // CMX operation 1
    %cmx_op_1 = VPU.VerticalFusion (%arg2 as %arg20: tensor<1x1x1024x1024xf32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1x1024x1024xf16> {
      %inner = VPU.Convert(%arg20) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x1024x1024xf32> -> tensor<1x1x1024x1024xf16>
      VPU.Yield %inner
    }
    %0 = VPU.AffineReshape(%cmx_op_1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 1024, 1, 1]} : tensor<1x1x1024x1024xf16> -> tensor<1024x1024x1x1xf16>
    %1 = VPU.PermuteCast(%0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>} : tensor<1024x1024x1x1xf16> -> tensor<1x1024x1024x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %2 = VPU.ShapeCast {shape = [1, 1024, 256, 4]} inputs(%1 : tensor<1x1024x1024x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>) -> tensor<1x1024x256x4xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    %3 = VPU.VerticalFusion (
          %cmx_op_0 as %arg20: tensor<1x1024x256x4xf16, {order = #NHWC}>,
          %2 as %arg21: tensor<1x1024x256x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1024x256x4xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg20, %arg21) {
          is_inplace = true,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          op_type = #VPU.eltwise_type<ADD>,
          ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
      } -> tensor<1x1024x256x4xf16, {order = #NHWC}>
      VPU.Yield %inner
    }

    %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %4 = VPU.VerticalFusion (
          %3 as %arg20: tensor<1x1024x256x4xf16, {order = #NHWC}>,
          %arg3 as %arg21: tensor<128x1024x1x1xf16, {order = #NHWC}>,
          %cst_1 as %arg22: tensor<128x1x1x4xsi32>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x256x4xf16, {order = #NHWC}> {
      %inner_0 = VPU.SoftMax(%arg20) {
          axisInd = 1 : i64,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
      } : tensor<1x1024x256x4xf16, {order = #NHWC}> -> tensor<1x1024x256x4xf16, {order = #NHWC}>
      %inner_1 = VPU.NCE.Convolution(%inner_0, %arg21, %arg22) {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
          rawFilterShape = [128, 1024, 1, 1],
          strides = [1, 1]
      } -> tensor<1x128x256x4xf16, {order = #NHWC}>
      VPU.Yield %inner_1
    }

    return %4 : tensor<1x128x256x4xf16, {order = #NHWC}>


    //CHECK:    [[CMX_OP_1:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: tilingStrategy = [1, 1, 1, 1]
    //CHECK:        [[CONV0:%.+]] = VPU.Convert

    //CHECK:    [[VF:%.+]] = VPU.VerticalFusion (
    //CHECK-SAME: scenario = #VPU.vf_scenario<FULL_PREFETCHING>,
    //CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
    //CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution
    //CHECK:        [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    //CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax
    //CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.00565029593075023:128>
!qElemType1 = !quant.uniform<u8:f16, 0.013744638480392157:128>

func.func @VFTestCorrectTilingAxisWhenGetPrefetchingCost(
                %arg0: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
                %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>,
                %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
    %cst_1 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x1024x4x!qElemType0, {order = #NHWC}>,
        %arg1 as %arg4: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>,
        %cst_0 as %arg5: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

    %1 = VPU.VerticalFusion (%0 as %arg3: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.SoftMax(%arg3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

   %2 = VPU.VerticalFusion (%1 as %arg3: tensor<1x4096x1024x4xf16, {order = #NHWC}>,
        %arg2 as %arg4: tensor<48x4096x1x1xf16, {order = #NHWC}>,
        %cst_1 as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 10, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}>
      VPU.Yield %3
   }

   return %2: tensor<1x48x1024x4xf16, {order = #NHWC}>

   // CHECK: VPU.VerticalFusion
   // CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 29, 1]

}
