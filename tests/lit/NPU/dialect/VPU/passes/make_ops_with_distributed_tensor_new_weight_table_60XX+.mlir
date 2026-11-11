//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor="enable-explicit-distributed-attr=true" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseInputsSameOffsets
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x128x72x72xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x128x72x72xf16, {order = #NHWC}>)
func.func @EltwiseInputsSameOffsets(%arg0: tensor<1x128x72x72xf16, {order = #NHWC}>, %arg1: tensor<1x128x72x72xf16, {order = #NHWC}>) -> tensor<1x128x72x72xf16> {
    %weights = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %scaleTable = const.Declare tensor<64x1x1x1xf32> = dense<1.0> : tensor<64x1x1x1xf32>
    %biasTable = const.Declare tensor<64x1x1x1xf32> = dense<2.0> : tensor<64x1x1x1xf32>
    %cst_1 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %weights, %scaleTable, %biasTable) rawFilterShape [64, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,  strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x128x72x72xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}>, tensor<64x1x1x1xf32>, tensor<64x1x1x1xf32> -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %1 = VPU.NCE.DepthConvolution(%0, %cst_1) rawFilterShape [64, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}> -> tensor<1x128x72x72xf16, {order = #NHWC}>

    %3 = VPU.NCE.Eltwise(%2, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
        } -> tensor<1x128x72x72xf16>

    return %3 : tensor<1x128x72x72xf16>

// CHECK: [[WT0:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
// CHECK: [[SCALE_TABLE0:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<64x1x1x1xf32>
// CHECK: [[BIAS_TABLE0:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<2.000000e+00> : tensor<64x1x1x1xf32>
// CHECK: [[WT1:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]

// CHECK:               [[INPUT0_CP:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 128, 24, 72], [1, 128, 24, 72], [1, 128, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 128, 24, 72], [1, 128, 24, 72], [1, 128, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]]}>
// CHECK:               [[WT0_CP:%.+]] = VPU.UnrolledType([[WT0]] : tensor<64x128x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[64, 128, 1, 1], [64, 128, 1, 1], [64, 128, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[SCALE_TABLE0_CP:%.+]] = VPU.UnrolledType([[SCALE_TABLE0]] : tensor<64x1x1x1xf32>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<64x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[BIAS_TABLE0_CP:%.+]] = VPU.UnrolledType([[BIAS_TABLE0]] : tensor<64x1x1x1xf32>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<64x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[64, 1, 1, 1], [64, 1, 1, 1], [64, 1, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT0_CP]], [[WT0_CP]], [[SCALE_TABLE0_CP]], [[BIAS_TABLE0_CP]]) rawFilterShape [64, 128, 1, 1] {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME{LITERAL}:                                                    ppe = #VPU.PPEStub<>,
// CHECK-SAME:          strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 64, 24, 72], [1, 64, 24, 72], [1, 64, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 64, 25, 72], [1, 64, 26, 72], [1, 64, 25, 72]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 47, 0]]}>
// CHECK:               [[CONV_CP0:%.+]] = VPU.UnrolledType([[OUT_CONV]] : !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 64, 24, 72], [1, 64, 24, 72], [1, 64, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 64, 25, 72], [1, 64, 26, 72], [1, 64, 25, 72]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 47, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK:               [[CONV_CP1:%.+]] = VPU.UnrolledType([[CONV_CP0]] : tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 64, 24, 72], [1, 64, 24, 72], [1, 64, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 64, 25, 72], [1, 64, 26, 72], [1, 64, 25, 72]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 47, 0]]}>
// CHECK:               [[WT1_CP:%.+]] = VPU.UnrolledType([[WT1]] : tensor<64x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<64x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1], [64, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[OUT_DCONV:%.+]] = VPU.NCE.DepthConvolution([[CONV_CP1]], [[WT1_CP]]) rawFilterShape [64, 1, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
// CHECK-SAME{LITERAL}:                                                    ppe = #VPU.PPEStub<>,
// CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 64, 24, 72], [1, 64, 24, 72], [1, 64, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 64, 25, 72], [1, 64, 26, 72], [1, 64, 25, 72]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 47, 0]]}>
// CHECK:               [[DCONV_CP0:%.+]] = VPU.UnrolledType([[OUT_DCONV]] : !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 64, 24, 72], [1, 64, 24, 72], [1, 64, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 64, 25, 72], [1, 64, 26, 72], [1, 64, 25, 72]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 47, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK:               [[OUT_CONCAT:%.+]] = VPU.Concat([[CONV_CP0]], [[DCONV_CP0]])
// CHECK-SAME{LITERAL}:                                     {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK:               [[CONCAT_CP:%.+]] = VPU.UnrolledType([[OUT_CONCAT]] : tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 128, 24, 72], [1, 128, 24, 72], [1, 128, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 128, 25, 72], [1, 128, 26, 72], [1, 128, 25, 72]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 47, 0]]}>
// CHECK:               [[INPUT1_CP:%.+]] = VPU.UnrolledType([[ARG1]] : tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 128, 24, 72], [1, 128, 24, 72], [1, 128, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 128, 25, 72], [1, 128, 26, 72], [1, 128, 25, 72]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 47, 0]]}>
// CHECK:               [[OUT_ELTW:%.+]] = VPU.NCE.Eltwise([[CONCAT_CP]], [[INPUT1_CP]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x128x72x72xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 128, 24, 72], [1, 128, 24, 72], [1, 128, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 128, 24, 72], [1, 128, 24, 72], [1, 128, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_ELTW]] : !VPU.DistributedTensor<1x128x72x72xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 128, 24, 72], [1, 128, 24, 72], [1, 128, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 128, 24, 72], [1, 128, 24, 72], [1, 128, 24, 72]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0], [0, 0, 48, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x128x72x72xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x128x72x72xf16>

}
}
