//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true compilation-mode=DefaultHW" --vertical-fusion-tiling="enable-vertical-fusion-pipelining=true" %s | FileCheck %s
// REQUIRES: platform-NPU6010

// Tests VF tiling of a fused SDPA softmax sub-graph containing:
//   - NCE.Convolution with 2 results (output + reduce_xy_max, i.e., resultSegmentSizes [1,1,0,0])
//   - SoftMax consuming both results (SkipNormalization mode)
//   - NCE.Convolution (output projection matmul)
//   - NCE.Reduce (sum for softmax denominator)
//   - NCE.Convolution (tile/scale conv)
//   - NCE.Eltwise (element-wise multiply)
// The VF has a single output and tilingStrategy along H.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemTypeAct = !quant.uniform<f8E4M3FN:f16, 0.012199813766138894>
!qElemTypeWt  = !quant.uniform<f8E4M3FN:f16, 0.010631477194172996>

// CHECK-DAG: [[QELEMTYPE_ACT:!.+]] = !quant.uniform<f8E4M3FN:f16, 0.012199813766138894>
// CHECK-DAG: [[QELEMTYPE_WT:!.+]] = !quant.uniform<f8E4M3FN:f16, 0.010631477194172996>

module @module {
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
}

config.Resources 4 of @NCE  {
    config.ExecutorResource 1 of @DPU
}

// CHECK: @VFPipeliningWithMultiResultConvAndReduce
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x64x8x4x[[QELEMTYPE_ACT]], {order = #NHWC}>
// CHECK-SAME: [[FILTER_QK:%.+]]: tensor<4096x64x1x1x[[QELEMTYPE_WT]], {order = #NHWC}>
// CHECK-SAME: [[FILTER_OUT:%.+]]: tensor<64x4096x1x1xf16, {order = #NHWC}>
func.func @VFPipeliningWithMultiResultConvAndReduce(
        %arg0: tensor<1x64x8x4x!qElemTypeAct, {order = #NHWC}>,
        %arg1: tensor<4096x64x1x1x!qElemTypeWt, {order = #NHWC}>,
        %arg2: tensor<64x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x64x8x4xf16> {
    %cst_filter_tile = const.Declare tensor<64x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>,
                           [#const.Broadcast<0 : i64, 64 : i64>, #const.Reshape<[64, 1, 1, 1]>,
                            #const.Reorder<#NHWC>, #const.Reshape<[64, 1, 1, 1]>,
                            #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 15]>]

    %0 = VPU.VerticalFusion (
            %arg0            as %arg3:  tensor<1x64x8x4x!qElemTypeAct, {order = #NHWC}>,
            %arg1            as %arg4:  tensor<4096x64x1x1x!qElemTypeWt, {order = #NHWC}>,
            %cst_filter_tile as %arg7:  tensor<64x1x1x16xf16, {order = #NHWC}>,
            %arg2            as %arg10: tensor<64x4096x1x1xf16, {order = #NHWC}>)
            attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 2, 1]}
            -> tensor<1x64x8x4xf16> {

        // QK matmul: produces output activation and reduce_xy_max (row-wise max for online softmax)
        %output_qk, %reduce_xy_max = VPU.NCE.Convolution(%arg3, %arg4)
            rawFilterShape [4096, 64, 1, 1] {
                axes_value = [1],
                input_padding = [0, 8, 0, 0],
                mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                output_padding = [0, 0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                resultSegmentSizes = array<i32: 1, 1, 0, 0>,
                strides = [1, 1]}
            : tensor<1x64x8x4x!qElemTypeAct, {order = #NHWC}>,
              tensor<4096x64x1x1x!qElemTypeWt, {order = #NHWC}>
            -> tensor<1x4096x8x4xf16, {order = #NHWC}>, tensor<1x1x8x4xf16, {order = #NHWC}>

        // Fused softmax using online max normalization
        %softmax_out = VPU.SoftMax(%output_qk, %reduce_xy_max) {
                SkipNormalization,
                axisInd = 1 : i64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
            : tensor<1x4096x8x4xf16, {order = #NHWC}>, tensor<1x1x8x4xf16, {order = #NHWC}>
            -> tensor<1x4096x8x4xf16, {order = #NHWC}>

        // Output projection matmul
        %output_matmul = VPU.NCE.Convolution(%softmax_out, %arg10)
            rawFilterShape [64, 4096, 1, 1] {
                input_padding = [0, 0, 0, 0],
                mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                output_padding = [0, 8, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                strides = [1, 1]}
            : tensor<1x4096x8x4xf16, {order = #NHWC}>,
              tensor<64x4096x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x8x4xf16, {order = #NHWC}>

        // Reduce sum of softmax output
        %reduce_sum = VPU.NCE.Reduce(%softmax_out) {
                axes = [1],
                input_padding = [0, 0, 0, 0],
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.reduce_type<SUM>,
                ppe = #VPU.PPEStub<>}
            -> tensor<1x1x8x4xf16, {order = #NHWC}>

        %scale_conv = VPU.NCE.Convolution(%reduce_sum, %arg7)
            rawFilterShape [64, 1, 1, 1] {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                strides = [1, 1]}
            : tensor<1x1x8x4xf16, {order = #NHWC}>,
              tensor<64x1x1x16xf16, {order = #NHWC}>
            -> tensor<1x64x8x4xf16, {order = #NHWC}>

        %result = VPU.NCE.Eltwise(%output_matmul, %scale_conv) {
                is_inplace = false,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<MULTIPLY>,
                ppe = #VPU.PPEStub<>,
                resultSegmentSizes = array<i32: 1, 0, 0, 0>}
            -> tensor<1x64x8x4xf16>
        VPU.Yield %result
    }

    return %0 : tensor<1x64x8x4xf16>

    // Constants are preserved
    // CHECK-DAG: [[FILTER_TILE:%.+]] = const.Declare tensor<64x1x1x16xf16, {order = #NHWC}>

    // Tile 0: rows [0, 0, 0, 0] -> [1, 64, 4, 4]

    // CHECK: [[SLICE_T0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 64, 4, 4]

    // QK matmul for tile 0: two-result convolution (output + row-wise max)
    // CHECK: [[CONV_QK_T0:%.+]], [[RMAX_T0:%.+]] = VPU.NCE.Convolution([[SLICE_T0]], [[FILTER_QK]])
    // CHECK-SAME: resultSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME: -> tensor<1x4096x4x4xf16, {order = #NHWC}>, tensor<1x1x4x4xf16, {order = #NHWC}>

    // Fused softmax for tile 0
    // CHECK: [[SOFTMAX_T0:%.+]] = VPU.SoftMax([[CONV_QK_T0]], [[RMAX_T0]])
    // CHECK-SAME: SkipNormalization

    // Output matmul for tile 0
    // CHECK: [[CONV_OUT_T0:%.+]] = VPU.NCE.Convolution([[SOFTMAX_T0]], [[FILTER_OUT]])

    // Reduce sum for tile 0
    // CHECK: [[REDUCE_T0:%.+]] = VPU.NCE.Reduce([[SOFTMAX_T0]])
    // CHECK-SAME: op_type = #VPU.reduce_type<SUM>

    // Scale tile conv for tile 0
    // CHECK: [[CONV_TILE_T0:%.+]] = VPU.NCE.Convolution([[REDUCE_T0]], [[FILTER_TILE]])

    // Eltwise multiply for tile 0
    // CHECK: [[ELTWISE_T0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT_T0]], [[CONV_TILE_T0]])
    // CHECK-SAME: op_type = #VPU.eltwise_type<MULTIPLY>

    // Tile 1: rows [0, 0, 4, 0] -> [1, 64, 4, 4]

    // CHECK: [[SLICE_T1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 4, 0] [1, 64, 4, 4]

    // QK matmul for tile 1
    // CHECK: [[CONV_QK_T1:%.+]], [[RMAX_T1:%.+]] = VPU.NCE.Convolution([[SLICE_T1]], [[FILTER_QK]])
    // CHECK-SAME: resultSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME: -> tensor<1x4096x4x4xf16, {order = #NHWC}>, tensor<1x1x4x4xf16, {order = #NHWC}>

    // Fused softmax for tile 1
    // CHECK: [[SOFTMAX_T1:%.+]] = VPU.SoftMax([[CONV_QK_T1]], [[RMAX_T1]])
    // CHECK-SAME: SkipNormalization

    // Output matmul for tile 1
    // CHECK: [[CONV_OUT_T1:%.+]] = VPU.NCE.Convolution([[SOFTMAX_T1]], [[FILTER_OUT]])

    // Reduce sum for tile 1
    // CHECK: [[REDUCE_T1:%.+]] = VPU.NCE.Reduce([[SOFTMAX_T1]])
    // CHECK-SAME: op_type = #VPU.reduce_type<SUM>

    // Scale tile conv for tile 1
    // CHECK: [[CONV_TILE_T1:%.+]] = VPU.NCE.Convolution([[REDUCE_T1]], [[FILTER_TILE]])

    // Eltwise multiply for tile 1
    // CHECK: [[ELTWISE_T1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT_T1]], [[CONV_TILE_T1]])
    // CHECK-SAME: op_type = #VPU.eltwise_type<MULTIPLY>

    // Final concatenation of both tiles along H
    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_T0]], [[ELTWISE_T1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]}
    // CHECK-SAME: tensor<1x64x8x4xf16>

    // CHECK: return [[CONCAT]] : tensor<1x64x8x4xf16>
}
}
