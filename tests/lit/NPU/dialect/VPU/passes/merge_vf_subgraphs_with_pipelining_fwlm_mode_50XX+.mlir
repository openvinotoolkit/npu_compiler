//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16, 0.015444456480392157:128>

// CHECK-LABEL: @MergeVFWithSOHToSOK
func.func @MergeVFWithSOHToSOK(%arg0: tensor<1x64x368x480x!qElemType, {order = #NHWC}>) -> tensor<1x128x184x240x!qElemType1, {order = #NHWC}> {
    %cst = const.Declare tensor<128x64x3x3x!qElemType1, {order = #NHWC}> = dense<1> : tensor<128x64x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x368x480x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 10]} -> tensor<1x64x184x240x!qElemType, {order = #NHWC}> {
         %inner = VPU.NCE.MaxPool(%arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                  kernel_size = [2, 2],
                  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                  ppe = #VPU.PPEStub<>,
                  strides = [2, 2]} -> tensor<1x64x184x240x!qElemType, {order = #NHWC}>
         VPU.Yield %inner
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x64x184x240x!qElemType, {order = #NHWC}>,
                             %cst as %arg2: tensor<128x64x3x3x!qElemType1, {order = #NHWC}>
                            ) attributes {tilingStrategy = [1, 1, 1, 15]} -> tensor<1x128x184x240x!qElemType1, {order = #NHWC}> {
         %inner = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [128, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                  mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                  ppe = #VPU.PPEStub<>,
                  strides = [1, 1]} : tensor<1x64x184x240x!qElemType, {order = #NHWC}>, tensor<128x64x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x128x184x240x!qElemType1, {order = #NHWC}>
         VPU.Yield %inner
    }
    return %1 : tensor<1x128x184x240x!qElemType1, {order = #NHWC}>

    //CHECK:        [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    //CHECK:          VPU.NCE.MaxPool
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
    //CHECK:          VPU.NCE.Convolution
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK: return [[VERTICAL_FUSION]]
}

// -----

// Verify that standalone VPU.Slice on the channel dimension between two VF blocks
// prevents them from being merged.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvSliceOnCConvert
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x96x64x64xf16, {order = #NHWC}>
func.func @ConvSliceOnCConvert(%arg0: tensor<1x96x64x64xf16, {order = #NHWC}>) -> tensor<1x3x64x64xf32, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x96x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x96x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x96x64x64xf16, {order = #NHWC}>,
                             %cst_0 as %arg2: tensor<16x96x1x1xf16, {order = #NHWC}>
                            ) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x64x64xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [16, 96, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                   mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                   ppe = #VPU.PPEStub<>,
                   strides = [1, 1]} : tensor<1x96x64x64xf16, {order = #NHWC}>, tensor<16x96x1x1xf16, {order = #NHWC}>
                    -> tensor<1x16x64x64xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x64x64xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x3x64x64xf32, {order = #NHWC}> {
      %3 = VPU.Slice %arg1 [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x16x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>
      %4 = VPU.Convert(%3) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x3x64x64xf32, {order = #NHWC}>
      VPU.Yield %4
    }
    return %1 : tensor<1x3x64x64xf32, {order = #NHWC}>

    //CHECK:      [[VF:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK-SAME:          scenario = #VPU.vf_scenario<VF_PIPELINING>
    //CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    //CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution
    //CHECK:        [[SLICE:%.+]] = VPU.Slice [[CONV]] [0, 0, 0, 0] [1, 3, 64, 64]
    //CHECK:        [[CONVERT:%.+]] = VPU.Convert([[SLICE]])
    //CHECK:        VPU.Yield [[CONVERT]]
    //CHECK:      return [[VF]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType2 = !quant.uniform<u8:f16, 0.0025564837689493218:130>

// CHECK-LABEL: @DoNotMergeUntiledWeightDequantVF
func.func @DoNotMergeUntiledWeightDequantVF(
    %arg0: tensor<1x768x16x4xf16, {order = #NHWC}>,
    %arg1: tensor<768x768x1x1x!qElemType2, {order = #NHWC}>) -> tensor<1x768x16x4xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg1 as %wq: tensor<768x768x1x1x!qElemType2, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<768x768x1x1xf16, {order = #NHWC}> {
        %2 = VPU.Dequantize(%wq) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<768x768x1x1x!qElemType2, {order = #NHWC}> -> tensor<768x768x1x1xf16, {order = #NHWC}>
        VPU.Yield %2
    }

    %1 = VPU.VerticalFusion (%arg0 as %act: tensor<1x768x16x4xf16, {order = #NHWC}>, %0 as %wf: tensor<768x768x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x768x16x4xf16, {order = #NHWC}> {
        %3 = VPU.NCE.Convolution(%act, %wf) rawFilterShape [768, 768, 1, 1] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                strides = [1, 1]
             } : tensor<1x768x16x4xf16, {order = #NHWC}>, tensor<768x768x1x1xf16, {order = #NHWC}> -> tensor<1x768x16x4xf16, {order = #NHWC}>
        VPU.Yield %3
    }
    return %1 : tensor<1x768x16x4xf16, {order = #NHWC}>

    // CHECK:      VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 1]
    // CHECK:        [[DEQ:%.+]] = VPU.Dequantize
    // CHECK:        VPU.Yield [[DEQ]]

    // CHECK:      [[CONV_VF:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 1]
    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK:        VPU.Yield [[CONV]]
    // CHECK:      return [[CONV_VF]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qW512x256 = !quant.uniform<u8:f16, 0.001:128>
!qW256x512 = !quant.uniform<u8:f16, 0.002:128>
module @module {
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// Graph:
//   arg0(1x192) → Conv192→256(%0) → DWConv(%1) ─────────────────┐
//   arg1(1x256) → Conv256→256(%2) ──────────────────────────────-┤
//                                                                 └→ Eltwise-ADD(%3) ─┬→ Conv256→512(%4) ─┐
//                                                (residual) ──────────────────────── │  Conv256→512(%5) ─┴→ Eltwise-ADD(%6) → Conv512→256(%7) → Eltwise-ADD(%8)
//                                                                                    └──────────────────────────────────────────────────────────────────────── ↗
//   cst + arg5(1x16) → Eltwise-SUBTRACT(%9) ──── Multiply(%10, input: %8)
//   arg6(raw)  ──────── NCE.Permute(%11) ──── Eltwise-ADD(%12, inputs: %11, %10) → Conv256→256(%13)
//   Note: %12 (Eltwise-ADD), %10 (Multiply), %8 (Eltwise-ADD) should NOT be merged
func.func @NotFuseLastMultiplyAndSurroundingEltwiseOps(
    %arg0: tensor<1x192x46x80xf16, {order = #NHWC}>,
    %arg1: tensor<1x256x46x80xf16, {order = #NHWC}>,
    %arg2: tensor<512x256x1x1x!qW512x256, {order = #NHWC}>,
    %arg3: tensor<512x256x1x1x!qW512x256, {order = #NHWC}>,
    %arg4: tensor<256x512x1x1x!qW256x512, {order = #NHWC}>,
    %arg5: tensor<1x16x1x1xf16, {order = #NHWC}>,
    %arg6: tensor<1x256x46x80xf16>,
    %arg7: tensor<256x256x1x1xf16, {order = #NHWC}>
) -> tensor<1x256x46x80xf16, {order = #NHWC}> {

  %cst_0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1xf16>,
      [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
  %cst_2 = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x16x1x1xf16>, [#const.Reorder<#NHWC>]
  %cst_4 = const.Declare tensor<256x192x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x192x1x1xf16>, [#const.Reorder<#NHWC>]
  %cst_5 = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<4.0> : tensor<256x256x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]

  // Conv 192->256.
  %0 = VPU.VerticalFusion (%arg0 as %in0: tensor<1x192x46x80xf16, {order = #NHWC}>,
                            %cst_4 as %in1: tensor<256x192x1x1xf16, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x256x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.Convolution(%in0, %in1) rawFilterShape [256, 192, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
    } : tensor<1x192x46x80xf16, {order = #NHWC}>, tensor<256x192x1x1xf16, {order = #NHWC}>
      -> tensor<1x256x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // DWConv 256.
  %1 = VPU.VerticalFusion (%0 as %in0: tensor<1x256x46x80xf16, {order = #NHWC}>,
                            %cst_2 as %in1: tensor<256x16x1x1xf16, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x256x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.DepthConvolution(%in0, %in1) rawFilterShape [256, 1, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
    } -> tensor<1x256x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // Conv 256->256.
  %2 = VPU.VerticalFusion (%arg1 as %in0: tensor<1x256x46x80xf16, {order = #NHWC}>,
                            %cst_5 as %in1: tensor<256x256x1x1xf16, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x256x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.Convolution(%in0, %in1) rawFilterShape [256, 256, 1, 1] {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
    } : tensor<1x256x46x80xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}>
      -> tensor<1x256x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // Eltwise-ADD: merges DWConv and Conv branches.
  %3 = VPU.VerticalFusion (%1 as %in0: tensor<1x256x46x80xf16, {order = #NHWC}>,
                            %2 as %in1: tensor<1x256x46x80xf16, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x256x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.Eltwise(%in0, %in1) {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
    } -> tensor<1x256x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // Conv 256->512 (first branch).
  %4 = VPU.VerticalFusion (%3 as %in0: tensor<1x256x46x80xf16, {order = #NHWC}>,
                            %arg2 as %in1: tensor<512x256x1x1x!qW512x256, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x512x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.Convolution(%in0, %in1) rawFilterShape [512, 256, 1, 1] {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
    } : tensor<1x256x46x80xf16, {order = #NHWC}>, tensor<512x256x1x1x!qW512x256, {order = #NHWC}>
      -> tensor<1x512x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // Conv 256->512 (second branch).
  %5 = VPU.VerticalFusion (%3 as %in0: tensor<1x256x46x80xf16, {order = #NHWC}>,
                            %arg3 as %in1: tensor<512x256x1x1x!qW512x256, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x512x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.Convolution(%in0, %in1) rawFilterShape [512, 256, 1, 1] {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
    } : tensor<1x256x46x80xf16, {order = #NHWC}>, tensor<512x256x1x1x!qW512x256, {order = #NHWC}>
      -> tensor<1x512x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // Eltwise-ADD: merges the two 512-channel branches.
  %6 = VPU.VerticalFusion (%4 as %in0: tensor<1x512x46x80xf16, {order = #NHWC}>,
                            %5 as %in1: tensor<1x512x46x80xf16, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x512x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.Eltwise(%in0, %in1) {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
    } -> tensor<1x512x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // Conv 512->256.
  %7 = VPU.VerticalFusion (%6 as %in0: tensor<1x512x46x80xf16, {order = #NHWC}>,
                            %arg4 as %in1: tensor<256x512x1x1x!qW256x512, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x256x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.Convolution(%in0, %in1) rawFilterShape [256, 512, 1, 1] {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
    } : tensor<1x512x46x80xf16, {order = #NHWC}>, tensor<256x512x1x1x!qW256x512, {order = #NHWC}>
      -> tensor<1x256x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // Eltwise-ADD: residual skip from %3.
  %8 = VPU.VerticalFusion (%7 as %in0: tensor<1x256x46x80xf16, {order = #NHWC}>,
                            %3 as %in1: tensor<1x256x46x80xf16, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x256x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.Eltwise(%in0, %in1) {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
    } -> tensor<1x256x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // Eltwise-SUBTRACT: produces the 1x1x1x1 scalar for Multiply.
  %9 = VPU.VerticalFusion (%cst_0 as %in0: tensor<1x16x1x1xf16, {order = #NHWC}>,
                            %arg5 as %in1: tensor<1x16x1x1xf16, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 1, 1, 1]}
                           -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %out = VPU.NCE.Eltwise(%in0, %in1) {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 15, 0, 0],
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            op_type = #VPU.eltwise_type<SUBTRACT>,
            ppe = #VPU.PPEStub<>
    } -> tensor<1x1x1x1xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // Multiply: scalar %9 broadcasts over feature map %8.
  %10 = VPU.VerticalFusion (%9 as %in0: tensor<1x1x1x1xf16, {order = #NHWC}>,
                             %8 as %in1: tensor<1x256x46x80xf16, {order = #NHWC}>
                            ) attributes {tilingStrategy = [1, 1, 1, 1]}
                            -> tensor<1x256x46x80xf16, {order = #NHWC}> {
    %out = VPU.Multiply(%in0, %in1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x1x1x1xf16, {order = #NHWC}>, tensor<1x256x46x80xf16, {order = #NHWC}>
      -> tensor<1x256x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // NCE.Permute: reorders arg6 to NHWC.
  %11 = VPU.NCE.Permute(%arg6) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 256 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEStub<>} -> tensor<1x256x46x80xf16, {order = #NHWC}>

  // Eltwise-ADD: %11 + %10. Should NOT merge with Multiply %10.
  %12 = VPU.VerticalFusion (%11 as %in0: tensor<1x256x46x80xf16, {order = #NHWC}>,
                             %10 as %in1: tensor<1x256x46x80xf16, {order = #NHWC}>
                            ) attributes {tilingStrategy = [1, 1, 1, 1]}
                            -> tensor<1x256x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.Eltwise(%in0, %in1) {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
    } -> tensor<1x256x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  // Conv 256->256 output.
  %13 = VPU.VerticalFusion (%12 as %in0: tensor<1x256x46x80xf16, {order = #NHWC}>,
                             %arg7 as %in1: tensor<256x256x1x1xf16, {order = #NHWC}>
                            ) attributes {tilingStrategy = [1, 1, 1, 1]}
                            -> tensor<1x256x46x80xf16, {order = #NHWC}> {
    %out = VPU.NCE.Convolution(%in0, %in1) rawFilterShape [256, 256, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
    } : tensor<1x256x46x80xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}>
      -> tensor<1x256x46x80xf16, {order = #NHWC}>
    VPU.Yield %out
  }

  return %13 : tensor<1x256x46x80xf16, {order = #NHWC}>

  // CHECK-LABEL: @NotFuseLastMultiplyAndSurroundingEltwiseOps
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.Convolution
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.DepthConvolution
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.Convolution
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.Eltwise
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.Convolution
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.Convolution
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.Eltwise
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.Convolution
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.Eltwise
  // CHECK:      VPU.VerticalFusion

  // The following op sequence must not be fused:
  //    NCE.Eltwise -> NCE.Multiply -> NCE.Eltwise
  // as it is less performant at this time.

  // CHECK-NEXT:   VPU.NCE.Eltwise
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.Multiply
  // CHECK:      VPU.NCE.Permute
  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.Eltwise

  // CHECK:      VPU.VerticalFusion
  // CHECK-NEXT:   VPU.NCE.Convolution
}
}
