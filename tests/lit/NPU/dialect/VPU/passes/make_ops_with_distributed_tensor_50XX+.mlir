//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor="enable-explicit-distributed-attr=true" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvMulticlusterSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>
func.func @ConvMulticlusterSOHOverlapped(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [80, 64, 3, 3],
        strides = [1, 1]}
      : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

// CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]


// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x64x28x28xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 64, 10, 28], [1, 64, 9, 28], [1, 64, 9, 28]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 64, 11, 28], [1, 64, 11, 28], [1, 64, 10, 28]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 18, 0]]}>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[WEIGHTS]] : tensor<80x64x3x3xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[CONV:%.*]] = VPU.NCE.Convolution([[IN_CP0]], [[IN_CP1]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]]}>

// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[CONV]] : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]]}>
// CHECK-SAME{LITERAL}:                                                -> tensor<1x80x28x28xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}
}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @DepthConvDistributedTensorSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>

func.func @DepthConvDistributedTensorSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

// CHECK:    [[WEIGHTS:%.*]]  = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 39, 112], [1, 32, 39, 112], [1, 32, 38, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 37, 0], [0, 0, 74, 0]]}>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[WEIGHTS]] : tensor<32x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[32, 16, 1, 1], [32, 16, 1, 1], [32, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[DEPTH_CONV:%.*]] = VPU.NCE.DepthConvolution([[IN_CP0]], [[IN_CP1]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[DEPTH_CONV]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK-SAME{LITERAL}:                                                    -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:                return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>


}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @MaxPoolMulticlusterSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @MaxPoolMulticlusterSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[MAXPOOL:%.*]] = VPU.NCE.MaxPool([[IN_CP0]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[MAXPOOL]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK-SAME{LITERAL}:                                                    -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>


}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseAddMulticlusterSOHOverlapped
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>,
// CHECK-SAME:   [[ARG1:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @EltwiseAddMulticlusterSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<> } :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[ARG1]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[IN_CP0]], [[IN_CP1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[ELTWISE]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK-SAME{LITERAL}:                                    -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @AvgPoolMulticlusterSOHOverlapped
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @AvgPoolMulticlusterSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 39, 112], [1, 32, 39, 112], [1, 32, 38, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 37, 0], [0, 0, 74, 0]]}>
// CHECK:               [[AVGPOOL:%.*]] = VPU.NCE.AveragePool([[IN_CP0]]) {kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[AVGPOOL]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SparseConvMulticlusterSOHOverlapped
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x64x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:   [[ARG1:%.*]]: tensor<1x64x28x28xi1, {order = #NHWC}>)
func.func @SparseConvMulticlusterSOHOverlapped(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 01 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<80x1x1x640xi1>, is_weights> -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>


// CHECK:      [[INPUT_SPARSE:%.*]] = VPU.GroupSparseTensor([[ARG0]], [[ARG1]]) -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>
// CHECK:      [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
// CHECK:      [[WEIGHTS_SM:%.*]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
// CHECK:      [[WEIGHTS_SPARSE:%.*]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<80x1x1x640xi1>, is_weights>
// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[INPUT_SPARSE]] : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 64, 10, 28], [1, 64, 9, 28], [1, 64, 9, 28]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 64, 11, 28], [1, 64, 11, 28], [1, 64, 10, 28]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 18, 0]]}>,
// CHECK-SAME{LITERAL}:                                                    sparsity_map=!VPU.DistributedTensor<1x64x28x28xi1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 64, 10, 28], [1, 64, 9, 28], [1, 64, 9, 28]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 64, 11, 28], [1, 64, 11, 28], [1, 64, 10, 28]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 18, 0]]}>>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[WEIGHTS_SPARSE]] : !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<80x1x1x640xi1>, is_weights>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.SparseTensor<data=!VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[80, 64, 3, 3], [80, 64, 3, 3], [80, 64, 3, 3]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
// CHECK-SAME{LITERAL}:                                                    sparsity_map=!VPU.DistributedTensor<80x1x1x640xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[80, 1, 1, 640], [80, 1, 1, 640], [80, 1, 1, 640]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>, is_weights>
// CHECK:               [[CONV:%.*]] = VPU.NCE.Convolution([[IN_CP0]], [[IN_CP1]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]]}>,
// CHECK-SAME{LITERAL}:                                                    sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]]}>> {
// CHECK:                                                                  VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <VECTOR_FP16>
// CHECK:                                               }
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[CONV]] : !VPU.SparseTensor<data=!VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]]}>,
// CHECK-SAME{LITERAL}:                                                    sparsity_map=!VPU.DistributedTensor<1x80x28x28xi1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 80, 10, 28], [1, 80, 9, 28], [1, 80, 9, 28]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]]}>>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
// CHECK:               return [[OUT_CP]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

}

}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVN6SOK4
module @MVN6SOK4 {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @MVN6SOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x32x15x64xf16>)
func.func @MVN6SOK(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [2], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[INPUT_DATA]] : tensor<1x32x15x64xf16>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 11, 15, 64], [1, 11, 15, 64], [1, 10, 15, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 11, 15, 64], [1, 11, 15, 64], [1, 10, 15, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>
// CHECK:               [[MVN6:%.*]] = VPU.MVN6([[IN_CP0]]) {axes = [2], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} :
// CHECK-SAME{LITERAL}:                                                    !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 11, 15, 64], [1, 11, 15, 64], [1, 10, 15, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 11, 15, 64], [1, 11, 15, 64], [1, 10, 15, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 11, 15, 64], [1, 11, 15, 64], [1, 10, 15, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 11, 15, 64], [1, 11, 15, 64], [1, 10, 15, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>
// CHECK:               [[CP_OUT:%.*]] = VPU.UnrolledType([[MVN6]] : !VPU.DistributedTensor<1x32x15x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 11, 15, 64], [1, 11, 15, 64], [1, 10, 15, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 11, 15, 64], [1, 11, 15, 64], [1, 10, 15, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                          -> tensor<1x32x15x64xf16>
// CHECK:               return [[CP_OUT]] : tensor<1x32x15x64xf16>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @MVN6SOH
// CHECK-SAME:   [[INPUT:%.*]]: tensor<1x1x150x512xf16>, [[SCALE:%.*]]: tensor<1x1x1x512xf16>, [[BIAS:%.*]]: tensor<1x1x1x512xf16>
func.func @MVN6SOH(%arg0: tensor<1x1x150x512xf16>, %arg1: tensor<1x1x1x512xf16>, %arg2: tensor<1x1x1x512xf16>) -> tensor<1x1x150x512xf16> {
    %0 = VPU.MVN6(%arg0, %arg1, %arg2) {axes = [3], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x1x150x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x150x512xf16>
    return %0 : tensor<1x1x150x512xf16>

// CHECK:               [[IN_CP:%.+]] = VPU.UnrolledType([[INPUT]] : tensor<1x1x150x512xf16>
// CHECK-SAME{LITERAL}:                                                   -> !VPU.DistributedTensor<1x1x150x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                   compute_shapes = [[1, 1, 50, 512], [1, 1, 50, 512], [1, 1, 50, 512]],
// CHECK-SAME{LITERAL}:                                                   compute_offsets = [[0, 0, 0, 0], [0, 0, 50, 0], [0, 0, 100, 0]],
// CHECK-SAME{LITERAL}:                                                   memory_shapes = [[1, 1, 50, 512], [1, 1, 50, 512], [1, 1, 50, 512]],
// CHECK-SAME{LITERAL}:                                                   memory_offsets = [[0, 0, 0, 0], [0, 0, 50, 0], [0, 0, 100, 0]]}>

// CHECK:              [[SCALE_CP:%.+]] = VPU.UnrolledType([[SCALE]] : tensor<1x1x1x512xf16>
// CHECK-SAME{LITERAL}:                                                   -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                   compute_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
// CHECK-SAME{LITERAL}:                                                   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                   memory_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
// CHECK-SAME{LITERAL}:                                                   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:              [[BIAS_CP:%.+]] = VPU.UnrolledType([[BIAS]] : tensor<1x1x1x512xf16>
// CHECK-SAME{LITERAL}:                                                   -> !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                   compute_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
// CHECK-SAME{LITERAL}:                                                   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                   memory_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
// CHECK-SAME{LITERAL}:                                                   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:             [[MVN_RES:%.+]] = VPU.MVN6([[IN_CP]], [[SCALE_CP]], [[BIAS_CP]]) {axes = [3], eps = 1.000000e-02 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 1>} :
// CHECK-SAME{LITERAL}:                                              !VPU.DistributedTensor<1x1x150x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                   compute_shapes = [[1, 1, 50, 512], [1, 1, 50, 512], [1, 1, 50, 512]],
// CHECK-SAME{LITERAL}:                                                   compute_offsets = [[0, 0, 0, 0], [0, 0, 50, 0], [0, 0, 100, 0]],
// CHECK-SAME{LITERAL}:                                                   memory_shapes = [[1, 1, 50, 512], [1, 1, 50, 512], [1, 1, 50, 512]],
// CHECK-SAME{LITERAL}:                                                   memory_offsets = [[0, 0, 0, 0], [0, 0, 50, 0], [0, 0, 100, 0]]}>,
// CHECK-SAME{LITERAL}:                                              !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                   compute_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
// CHECK-SAME{LITERAL}:                                                   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                   memory_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
// CHECK-SAME{LITERAL}:                                                   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
// CHECK-SAME{LITERAL}:                                              !VPU.DistributedTensor<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                   compute_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
// CHECK-SAME{LITERAL}:                                                   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                   memory_shapes = [[1, 1, 1, 512], [1, 1, 1, 512], [1, 1, 1, 512]],
// CHECK-SAME{LITERAL}:                                                   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                           -> !VPU.DistributedTensor<1x1x150x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                   compute_shapes = [[1, 1, 50, 512], [1, 1, 50, 512], [1, 1, 50, 512]],
// CHECK-SAME{LITERAL}:                                                   compute_offsets = [[0, 0, 0, 0], [0, 0, 50, 0], [0, 0, 100, 0]],
// CHECK-SAME{LITERAL}:                                                   memory_shapes = [[1, 1, 50, 512], [1, 1, 50, 512], [1, 1, 50, 512]],
// CHECK-SAME{LITERAL}:                                                   memory_offsets = [[0, 0, 0, 0], [0, 0, 50, 0], [0, 0, 100, 0]]}>

// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[MVN_RES]] : !VPU.DistributedTensor<1x1x150x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                   compute_shapes = [[1, 1, 50, 512], [1, 1, 50, 512], [1, 1, 50, 512]],
// CHECK-SAME{LITERAL}:                                                   compute_offsets = [[0, 0, 0, 0], [0, 0, 50, 0], [0, 0, 100, 0]],
// CHECK-SAME{LITERAL}:                                                   memory_shapes = [[1, 1, 50, 512], [1, 1, 50, 512], [1, 1, 50, 512]],
// CHECK-SAME{LITERAL}:                                                   memory_offsets = [[0, 0, 0, 0], [0, 0, 50, 0], [0, 0, 100, 0]]}>) -> tensor<1x1x150x512xf16>

// CHECK:           return [[OUT_CP]] : tensor<1x1x150x512xf16>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PadSOK4
module @PadSOK4 {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: func.func @PadSwSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x30x50xf16>)
func.func @PadSwSOK(%arg0: tensor<1x16x30x50xf16>) -> tensor<1x16x33x53xf16> {
    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 3, 3]} : tensor<1x16x30x50xf16> -> tensor<1x16x33x53xf16>
    return %0 : tensor<1x16x33x53xf16>

// CHECK:               [[INPUT_CP:%.+]] = VPU.UnrolledType([[INPUT_DATA]] : tensor<1x16x30x50xf16>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x16x30x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 6, 30, 50], [1, 5, 30, 50], [1, 5, 30, 50]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 11, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 6, 30, 50], [1, 5, 30, 50], [1, 5, 30, 50]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 11, 0, 0]]}>
// CHECK:               [[OUT_PAD:%.+]] = VPU.Pad([[INPUT_CP]]) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 3, 3]} :
// CHECK-SAME{LITERAL}:                                                    !VPU.DistributedTensor<1x16x30x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 6, 30, 50], [1, 5, 30, 50], [1, 5, 30, 50]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 11, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 6, 30, 50], [1, 5, 30, 50], [1, 5, 30, 50]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 11, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x16x33x53xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 6, 33, 53], [1, 5, 33, 53], [1, 5, 33, 53]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 11, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 6, 33, 53], [1, 5, 33, 53], [1, 5, 33, 53]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 11, 0, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_PAD]] : !VPU.DistributedTensor<1x16x33x53xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 6, 33, 53], [1, 5, 33, 53], [1, 5, 33, 53]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 11, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 6, 33, 53], [1, 5, 33, 53], [1, 5, 33, 53]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 11, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x16x33x53xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x16x33x53xf16>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @EltwiseInputsSameOffsets
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x128x72x72xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x128x72x72xf16, {order = #NHWC}>)
func.func @EltwiseInputsSameOffsets(%arg0: tensor<1x128x72x72xf16, {order = #NHWC}>, %arg1: tensor<1x128x72x72xf16, {order = #NHWC}>) -> tensor<1x128x72x72xf16> {
    %cst = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} : tensor<1x128x72x72xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}> -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %1 = VPU.NCE.DepthConvolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}> -> tensor<1x128x72x72xf16, {order = #NHWC}>

    %3 = VPU.NCE.Eltwise(%2, %arg1) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
        } -> tensor<1x128x72x72xf16>

    return %3 : tensor<1x128x72x72xf16>

// CHECK: [[WT0:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
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
// CHECK:               [[OUT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT0_CP]], [[WT0_CP]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME{LITERAL}:                                                    ppe = #VPU.PPEStub<>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]}
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
// CHECK:               [[OUT_DCONV:%.+]] = VPU.NCE.DepthConvolution([[CONV_CP1]], [[WT1_CP]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
// CHECK-SAME{LITERAL}:                                                    ppe = #VPU.PPEStub<>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]}
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

// -----
module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz
// CHECK-LABEL: func.func @RMSNormClustering
// CHECK: ([[ARG0:%.*]]: tensor<1x1x32x6xf16>)
func.func @RMSNormClustering(%arg0: tensor<1x1x32x6xf16>) -> tensor<1x1x32x6xf16> {
  %cst = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00>: tensor<1x1x1x6xf16>
  %0 = VPU.RMS(%arg0, %cst) {eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x32x6xf16>, tensor<1x1x1x6xf16> -> tensor<1x1x32x6xf16>
  return %0 : tensor<1x1x32x6xf16>

    // CHECK:     [[CST:%.*]] = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00> : tensor<1x1x1x6xf16>
    // CHECK:     [[DATA:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x1x32x6xf16>)
    // CHECK-SAME: 	-> !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: 	compute_shapes = [[1, 1, 32, 6], [1, 1, 32, 6], [1, 1, 32, 6]],
    // CHECK-SAME{LITERAL}: 	compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: 	memory_shapes = [[1, 1, 32, 6], [1, 1, 32, 6], [1, 1, 32, 6]],
    // CHECK-SAME{LITERAL}: 	memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:     [[GAMMA:%.*]] = VPU.UnrolledType([[CST]] : tensor<1x1x1x6xf16>)
    // CHECK-SAME: 	-> !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: 	compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}: 	compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: 	memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}: 	memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:     [[RMS:%.*]] = VPU.RMS([[DATA]], [[GAMMA]]) {eps = 9.9999997473787516E-6 : f64}
    // CHECK-SAME: 	: !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: 	compute_shapes = [[1, 1, 32, 6], [1, 1, 32, 6], [1, 1, 32, 6]],
    // CHECK-SAME{LITERAL}: 	compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: 	memory_shapes = [[1, 1, 32, 6], [1, 1, 32, 6], [1, 1, 32, 6]],
    // CHECK-SAME{LITERAL}: 	memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK-SAME: 	!VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: 	compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}: 	compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: 	memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}: 	memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK-SAME: 	-> !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: 	compute_shapes = [[1, 1, 32, 6], [1, 1, 32, 6], [1, 1, 32, 6]],
    // CHECK-SAME{LITERAL}: 	compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: 	memory_shapes = [[1, 1, 32, 6], [1, 1, 32, 6], [1, 1, 32, 6]],
    // CHECK-SAME{LITERAL}: 	memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:     [[OUT:%.*]] = VPU.UnrolledType([[RMS]] : !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: 	compute_shapes = [[1, 1, 32, 6], [1, 1, 32, 6], [1, 1, 32, 6]],
    // CHECK-SAME{LITERAL}: 	compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: 	memory_shapes = [[1, 1, 32, 6], [1, 1, 32, 6], [1, 1, 32, 6]],
    // CHECK-SAME{LITERAL}: 	memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) -> tensor<1x1x32x6xf16>
    // CHECK:     return [[OUT]] : tensor<1x1x32x6xf16>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @UnrollSOKDequantConv
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x512x5x5xf16, {order = #NHWC}>
func.func @UnrollSOKDequantConv(%input: tensor<1x512x5x5xf16, {order = #NHWC}>) -> tensor<1x1024x5x5xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<1024x512x1x1x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<1024x512x1x1xf16>, [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]
    %dequant = VPU.Dequantize(%weights) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1024x512x1x1x!qElemType, {order = #NHWC}> -> tensor<1024x512x1x1xf16, {order = #NHWC}>
    %conv = VPU.NCE.Convolution(%input, %dequant) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [1024, 512, 1, 1], strides = [1, 1]
            } : tensor<1x512x5x5xf16, {order = #NHWC}>, tensor<1024x512x1x1xf16, {order = #NHWC}> -> tensor<1x1024x5x5xf16, {order = #NHWC}>

    return %conv : tensor<1x1024x5x5xf16, {order = #NHWC}>
}
    // CHECK:	 [[WEIGHTS:%.+]] = const.Declare tensor<1024x512x1x1x!qElemType, {order = #NHWC}>
    // CHECK:    [[DEQUANTIZE_IN:%.+]] = VPU.UnrolledType([[WEIGHTS]] : tensor<1024x512x1x1x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:    -> !VPU.DistributedTensor<1024x512x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:    compute_shapes = [[352, 512, 1, 1], [336, 512, 1, 1], [336, 512, 1, 1]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[352, 512, 1, 1], [336, 512, 1, 1], [336, 512, 1, 1]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>
    // CHECK:    [[DEQUANTIZE:%.+]] = VPU.Dequantize([[DEQUANTIZE_IN]]) {dstElemType = f16} : !VPU.DistributedTensor<1024x512x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME:    -> !VPU.DistributedTensor<1024x512x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK:    [[DEQUANTIZE_OUT:%.+]] = VPU.UnrolledType([[DEQUANTIZE]] : !VPU.DistributedTensor<1024x512x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK-SAME{LITERAL}:    compute_shapes = [[352, 512, 1, 1], [336, 512, 1, 1], [336, 512, 1, 1]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[352, 512, 1, 1], [336, 512, 1, 1], [336, 512, 1, 1]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>
    // CHECK-SAME:     -> tensor<1024x512x1x1xf16, {order = #NHWC}>
    // CHECK:    [[CONV_INPUT:%.+]] = VPU.UnrolledType([[ARG0]]
    // CHECK:    [[CONV_WEIGHTS_IN:%.+]] = VPU.UnrolledType([[DEQUANTIZE_OUT]] : tensor<1024x512x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:     -> !VPU.DistributedTensor<1024x512x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution([[CONV_INPUT]], [[CONV_WEIGHTS_IN]])
    // CHECK:    [[CONV_OUT:%.+]] = VPU.UnrolledType([[CONV]] : !VPU.DistributedTensor<1x1024x5x5xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME:     -> tensor<1x1024x5x5xf16, {order = #NHWC}>
    // CHECK:    return [[CONV_OUT]] : tensor<1x1024x5x5xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz


// CHECK-LABEL: @NCEReduceSOH
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x25x25xf16, {order = #NHWC}>)
func.func @NCEReduceSOH(%arg0: tensor<1x16x25x25xf16, {order = #NHWC}>) -> tensor<1x1x25x25xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x25x25xf16, {order = #NHWC}>
  return %0 : tensor<1x1x25x25xf16, {order = #NHWC}>
}

// CHECK:    [[REDUCE_INPUT:%.+]] = VPU.UnrolledType([[ARG0]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x16x25x25xf16, #NHWC, @CMX_NN,
// CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 9, 25], [1, 16, 8, 25], [1, 16, 8, 25]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 9, 25], [1, 16, 8, 25], [1, 16, 8, 25]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>

// CHECK:     [[REDUCE_OUT:%.+]] = VPU.NCE.Reduce([[REDUCE_INPUT]])
// CHECK-SAME:   -> !VPU.DistributedTensor<1x1x25x25xf16, #NHWC, @CMX_NN,
// CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 9, 25], [1, 1, 8, 25], [1, 1, 8, 25]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 9, 25], [1, 1, 8, 25], [1, 1, 8, 25]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>


// CHECK:    [[UNROLLED_OUT:%.+]] = VPU.UnrolledType([[REDUCE_OUT]]
// CHECK-SAME:   -> tensor<1x1x25x25xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz


// CHECK-LABEL: @NCEReduceSOW
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x25x25xf16, {order = #NHWC}>)
func.func @NCEReduceSOW(%arg0: tensor<1x16x25x25xf16, {order = #NHWC}>) -> tensor<1x1x25x25xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>, op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x25x25xf16, {order = #NHWC}>
  return %0 : tensor<1x1x25x25xf16, {order = #NHWC}>
}

// CHECK:    [[REDUCE_INPUT:%.+]] = VPU.UnrolledType([[ARG0]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x16x25x25xf16, #NHWC, @CMX_NN,
// CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 25, 9], [1, 16, 25, 8], [1, 16, 25, 8]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 9], [0, 0, 0, 17]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 25, 9], [1, 16, 25, 8], [1, 16, 25, 8]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 9], [0, 0, 0, 17]]}>

// CHECK:     [[REDUCE_OUT:%.+]] = VPU.NCE.Reduce([[REDUCE_INPUT]])
// CHECK-SAME:   -> !VPU.DistributedTensor<1x1x25x25xf16, #NHWC, @CMX_NN,
// CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 25, 9], [1, 1, 25, 8], [1, 1, 25, 8]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 9], [0, 0, 0, 17]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 25, 9], [1, 1, 25, 8], [1, 1, 25, 8]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 9], [0, 0, 0, 17]]}>


// CHECK:    [[UNROLLED_OUT:%.+]] = VPU.UnrolledType([[REDUCE_OUT]]
// CHECK-SAME:   -> tensor<1x1x25x25xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz


// CHECK-LABEL: @NCEReduceHKSwitch
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x25x25xf16, {order = #NHWC}>)
func.func @NCEReduceHKSwitch(%arg0: tensor<1x16x25x25xf16, {order = #NHWC}>) -> tensor<1x1x25x25xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>, op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x25x25xf16, {order = #NHWC}>
  return %0 : tensor<1x1x25x25xf16, {order = #NHWC}>
}

// CHECK:    [[REDUCE_INPUT:%.+]] = VPU.UnrolledType([[ARG0]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x16x25x25xf16, #NHWC, @CMX_NN,
// CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 9, 25], [1, 16, 8, 25], [1, 16, 8, 25]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 9, 25], [1, 16, 8, 25], [1, 16, 8, 25]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>

// CHECK:     [[REDUCE_OUT:%.+]] = VPU.NCE.Reduce([[REDUCE_INPUT]])
// CHECK-SAME:   -> !VPU.DistributedTensor<1x1x25x25xf16, #NHWC, @CMX_NN,
// CHECK-SAME:          {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 9, 25], [1, 1, 8, 25], [1, 1, 8, 25]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 25, 25], [1, 1, 25, 25], [1, 1, 25, 25]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:    [[UNROLLED_OUT:%.+]] = VPU.UnrolledType([[REDUCE_OUT]]
// CHECK-SAME:   -> tensor<1x1x25x25xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz


// CHECK-LABEL: @NCEReduceClustering
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x25x25xf16, {order = #NHWC}>)
func.func @NCEReduceClustering(%arg0: tensor<1x16x25x25xf16, {order = #NHWC}>) -> tensor<1x1x25x25xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x25x25xf16, {order = #NHWC}>
  return %0 : tensor<1x1x25x25xf16, {order = #NHWC}>
}

// CHECK:    [[REDUCE_INPUT:%.+]] = VPU.UnrolledType([[ARG0]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x16x25x25xf16, #NHWC, @CMX_NN,
// CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 25, 25], [1, 16, 25, 25], [1, 16, 25, 25]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 25, 25], [1, 16, 25, 25], [1, 16, 25, 25]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:     [[REDUCE_OUT:%.+]] = VPU.NCE.Reduce([[REDUCE_INPUT]])
// CHECK-SAME:   -> !VPU.DistributedTensor<1x1x25x25xf16, #NHWC, @CMX_NN,
// CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 25, 25], [1, 1, 25, 25], [1, 1, 25, 25]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 25, 25], [1, 1, 25, 25], [1, 1, 25, 25]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:    [[UNROLLED_OUT:%.+]] = VPU.UnrolledType([[REDUCE_OUT]]
// CHECK-SAME:   -> tensor<1x1x25x25xf16, {order = #NHWC}>
}

// -----

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz


// CHECK-LABEL:   @RoPEAssignedSplitOverKernelSameK
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x32x1x64xf16>, [[INPUT1:%.+]]: tensor<1x32x1x64xf16>, [[INPUT2:%.+]]: tensor<1x32x1x64xf16>
func.func @RoPEAssignedSplitOverKernelSameK(%arg0: tensor<1x32x1x64xf16>, %arg1: tensor<1x32x1x64xf16>, %arg2: tensor<1x32x1x64xf16>) -> tensor<1x32x1x64xf16> {
    %0 = VPU.RoPE(%arg0, %arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
      : tensor<1x32x1x64xf16>, tensor<1x32x1x64xf16>, tensor<1x32x1x64xf16> -> tensor<1x32x1x64xf16>
    return %0 : tensor<1x32x1x64xf16>

// CHECK:    [[INPUT:%.+]] = VPU.UnrolledType([[INPUT0]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>

// CHECK:    [[COS:%.+]] = VPU.UnrolledType([[INPUT1]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>

// CHECK:    [[SIN:%.+]] = VPU.UnrolledType([[INPUT2]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>

// CHECK:     [[ROPE:%.+]] = VPU.RoPE([[INPUT]], [[COS]], [[SIN]])
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>

// CHECK:    [[UNROLLED_OUT:%.+]] = VPU.UnrolledType([[ROPE]]
// CHECK-SAME:   -> tensor<1x32x1x64xf16>
}
}

// -----

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @RoPEAssignedSplitOverKernelDiffK
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x32x1x64xf16>, [[INPUT1:%.+]]: tensor<1x1x1x64xf16>, [[INPUT2:%.+]]: tensor<1x1x1x64xf16>
func.func @RoPEAssignedSplitOverKernelDiffK(%arg0: tensor<1x32x1x64xf16>, %arg1: tensor<1x1x1x64xf16>, %arg2: tensor<1x1x1x64xf16>) -> tensor<1x32x1x64xf16> {
    %0 = VPU.RoPE(%arg0, %arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
      : tensor<1x32x1x64xf16>, tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x32x1x64xf16>
    return %0 : tensor<1x32x1x64xf16>

// CHECK:    [[INPUT:%.+]] = VPU.UnrolledType([[INPUT0]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>

// CHECK:    [[COS:%.+]] = VPU.UnrolledType([[INPUT1]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:    [[SIN:%.+]] = VPU.UnrolledType([[INPUT2]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x1x1x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:     [[ROPE:%.+]] = VPU.RoPE([[INPUT]], [[COS]], [[SIN]])
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 11, 1, 64], [1, 11, 1, 64], [1, 10, 1, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>

// CHECK:    [[UNROLLED_OUT:%.+]] = VPU.UnrolledType([[ROPE]]
// CHECK-SAME:   -> tensor<1x32x1x64xf16>
}
}

// -----

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @RoPEAssignedSplitOverHeightSameH
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x32x64x64xf16>, [[INPUT1:%.+]]: tensor<1x1x64x64xf16>, [[INPUT2:%.+]]: tensor<1x1x64x64xf16>
func.func @RoPEAssignedSplitOverHeightSameH(%arg0: tensor<1x32x64x64xf16>, %arg1: tensor<1x1x64x64xf16>, %arg2: tensor<1x1x64x64xf16>) -> tensor<1x32x64x64xf16> {
    %0 = VPU.RoPE(%arg0, %arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x32x64x64xf16>, tensor<1x1x64x64xf16>, tensor<1x1x64x64xf16> -> tensor<1x32x64x64xf16>
    return %0 : tensor<1x32x64x64xf16>

// CHECK:    [[INPUT:%.+]] = VPU.UnrolledType([[INPUT0]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 22, 64], [1, 32, 21, 64], [1, 32, 21, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 22, 64], [1, 32, 21, 64], [1, 32, 21, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>

// CHECK:    [[COS:%.+]] = VPU.UnrolledType([[INPUT1]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x1x64x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 22, 64], [1, 1, 21, 64], [1, 1, 21, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 22, 64], [1, 1, 21, 64], [1, 1, 21, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>

// CHECK:    [[SIN:%.+]] = VPU.UnrolledType([[INPUT2]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x1x64x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 22, 64], [1, 1, 21, 64], [1, 1, 21, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 22, 64], [1, 1, 21, 64], [1, 1, 21, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>

// CHECK:     [[ROPE:%.+]] = VPU.RoPE([[INPUT]], [[COS]], [[SIN]])
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 22, 64], [1, 32, 21, 64], [1, 32, 21, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 22, 64], [1, 32, 21, 64], [1, 32, 21, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>

// CHECK:    [[UNROLLED_OUT:%.+]] = VPU.UnrolledType([[ROPE]]
// CHECK-SAME:   -> tensor<1x32x64x64xf16>
}
}

// -----

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   @RoPEAssignedSplitOverHeightDiffH
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x32x64x64xf16>, [[INPUT1:%.+]]: tensor<1x32x1x64xf16>, [[INPUT2:%.+]]: tensor<1x32x1x64xf16>
func.func @RoPEAssignedSplitOverHeightDiffH(%arg0: tensor<1x32x64x64xf16>, %arg1: tensor<1x32x1x64xf16>, %arg2: tensor<1x32x1x64xf16>) -> tensor<1x32x64x64xf16> {
    %0 = VPU.RoPE(%arg0, %arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
      : tensor<1x32x64x64xf16>, tensor<1x32x1x64xf16>, tensor<1x32x1x64xf16> -> tensor<1x32x64x64xf16>
    return %0 : tensor<1x32x64x64xf16>

// CHECK:    [[INPUT:%.+]] = VPU.UnrolledType([[INPUT0]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 22, 64], [1, 32, 21, 64], [1, 32, 21, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 22, 64], [1, 32, 21, 64], [1, 32, 21, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>

// CHECK:    [[COS:%.+]] = VPU.UnrolledType([[INPUT1]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 1, 64], [1, 32, 1, 64], [1, 32, 1, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 1, 64], [1, 32, 1, 64], [1, 32, 1, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:    [[SIN:%.+]] = VPU.UnrolledType([[INPUT2]]
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x1x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 1, 64], [1, 32, 1, 64], [1, 32, 1, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 1, 64], [1, 32, 1, 64], [1, 32, 1, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:     [[ROPE:%.+]] = VPU.RoPE([[INPUT]], [[COS]], [[SIN]])
// CHECK-SAME:   -> !VPU.DistributedTensor<1x32x64x64xf16, #NCHW, @CMX_NN,
// CHECK-SAME:            {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 22, 64], [1, 32, 21, 64], [1, 32, 21, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 22, 64], [1, 32, 21, 64], [1, 32, 21, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0], [0, 0, 43, 0]]}>

// CHECK:    [[UNROLLED_OUT:%.+]] = VPU.UnrolledType([[ROPE]]
// CHECK-SAME:   -> tensor<1x32x64x64xf16>
}
}
