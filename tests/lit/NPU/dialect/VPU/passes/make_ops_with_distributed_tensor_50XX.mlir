//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor="enable-explicit-distributed-attr=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @ConvMulticlusterSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>
func.func @ConvMulticlusterSOHOverlapped(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x3x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<3x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [3, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [1, 1]}
      : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<3x64x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x3x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x3x28x28xf16, {order = #NHWC}>

// CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
// CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<3x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x64x3x3xf16>, [#const.Reorder<#NHWC>]

// CHECK:               [[IN_CP0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x64x28x28xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 64, 10, 28], [1, 64, 9, 28], [1, 64, 9, 28]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 64, 11, 28], [1, 64, 11, 28], [1, 64, 10, 28]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 18, 0]]}>

// CHECK:               [[IN_CP1:%.+]] = VPU.UnrolledType([[WEIGHTS]] : tensor<3x64x3x3xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<3x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[3, 64, 3, 3], [3, 64, 3, 3], [3, 64, 3, 3]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[3, 64, 3, 3], [3, 64, 3, 3], [3, 64, 3, 3]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[IN_CP2:%.+]] = VPU.UnrolledType([[WEIGHTSTABLE]] : tensor<16x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[CONV:%.+]] = VPU.NCE.Convolution([[IN_CP0]], [[IN_CP1]], [[IN_CP2]]) rawFilterShape [3, 64, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, 
// CHECK-SAME:                                                         strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x3x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 3, 10, 28], [1, 3, 9, 28], [1, 3, 9, 28]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 3, 10, 28], [1, 3, 9, 28], [1, 3, 9, 28]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]]}>

// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[CONV]] : !VPU.DistributedTensor<1x3x28x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 3, 10, 28], [1, 3, 9, 28], [1, 3, 9, 28]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 3, 10, 28], [1, 3, 9, 28], [1, 3, 9, 28]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 19, 0]]}>
// CHECK-SAME{LITERAL}:                                                -> tensor<1x3x28x28xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x3x28x28xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @ConvMulticlusterSOHLargeWeights
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x768x128x4xf16, {order = #NHWC}>
func.func @ConvMulticlusterSOHLargeWeights(%arg0: tensor<1x768x128x4xf16, {order = #NHWC}>) -> tensor<1x1x128x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1x768x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x768x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [1, 768, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [1, 1]}
      : tensor<1x768x128x4xf16, {order = #NHWC}>, tensor<1x768x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x1x128x4xf16, {order = #NHWC}>
    return %0 : tensor<1x1x128x4xf16, {order = #NHWC}>

// CHECK:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
// CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<1x768x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x768x1x1xf16>, [#const.Reorder<#NHWC>]

// CHECK:               [[IN_CP0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x768x128x4xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x768x128x4xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 768, 43, 4], [1, 768, 43, 4], [1, 768, 42, 4]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 768, 43, 4], [1, 768, 43, 4], [1, 768, 42, 4]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0]]}>

// CHECK:               [[IN_CP1:%.+]] = VPU.UnrolledType([[WEIGHTS]] : tensor<1x768x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x768x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 768, 1, 1], [1, 768, 1, 1], [1, 768, 1, 1]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 768, 1, 1], [1, 768, 1, 1], [1, 768, 1, 1]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[IN_CP2:%.+]] = VPU.UnrolledType([[WEIGHTSTABLE]] : tensor<16x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[CONV:%.+]] = VPU.NCE.Convolution([[IN_CP0]], [[IN_CP1]], [[IN_CP2]]) rawFilterShape [1, 768, 1, 1] {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, 
// CHECK-SAME:                                                         strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x1x128x4xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 1, 43, 4], [1, 1, 43, 4], [1, 1, 42, 4]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 1, 43, 4], [1, 1, 43, 4], [1, 1, 42, 4]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0]]}>

// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[CONV]] : !VPU.DistributedTensor<1x1x128x4xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 1, 43, 4], [1, 1, 43, 4], [1, 1, 42, 4]],
// CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0]],
// CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 1, 43, 4], [1, 1, 43, 4], [1, 1, 42, 4]],
// CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0]]}>
// CHECK-SAME{LITERAL}:                                                -> tensor<1x1x128x4xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x1x128x4xf16, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @DepthConvDistributedTensorSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>

func.func @DepthConvDistributedTensorSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x3x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare tensor<3x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) rawFilterShape [3, 1, 3, 3] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} -> tensor<1x3x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x3x112x112xf16, {order = #NHWC}>

// CHECK:    [[WEIGHTSTABLE:%.+]]  = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
// CHECK:    [[WEIGHTS:%.+]]  = const.Declare tensor<3x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x16x1x1xf16>, [#const.Reorder<#NHWC>]

// CHECK:               [[IN_CP0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 38, 112], [1, 32, 37, 112], [1, 32, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 39, 112], [1, 32, 39, 112], [1, 32, 38, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 37, 0], [0, 0, 74, 0]]}>
// CHECK:               [[IN_CP1:%.+]] = VPU.UnrolledType([[WEIGHTS]] : tensor<3x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<3x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[3, 16, 1, 1], [3, 16, 1, 1], [3, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[3, 16, 1, 1], [3, 16, 1, 1], [3, 16, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[IN_CP2:%.+]] = VPU.UnrolledType([[WEIGHTSTABLE]] : tensor<16x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4], [16, 1, 1, 4]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[DEPTH_CONV:%.+]] = VPU.NCE.DepthConvolution([[IN_CP0]], [[IN_CP1]], [[IN_CP2]]) rawFilterShape [3, 1, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, 
// CHECK-SAME:                                                             strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x3x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[DEPTH_CONV]] : !VPU.DistributedTensor<1x3x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK-SAME{LITERAL}:                                                    -> tensor<1x3x112x112xf16, {order = #NHWC}>
// CHECK:                return [[OUT_CP]] : tensor<1x3x112x112xf16, {order = #NHWC}>


}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @MaxPoolMulticlusterSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x16x112x112xf16, {order = #NHWC}>
func.func @MaxPoolMulticlusterSOHOverlapped(%arg0: tensor<1x16x112x112xf16, {order = #NHWC}>) -> tensor<1x3x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1],
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 0, 0, 0]
         } -> tensor<1x3x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x3x112x112xf16, {order = #NHWC}>

// CHECK:               [[IN_CP0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x16x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 16, 38, 112], [1, 16, 37, 112], [1, 16, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 16, 38, 112], [1, 16, 37, 112], [1, 16, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[IN_CP0]]) {input_padding = [0, 13, 0, 0], kernel_size = [1, 1], output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
// CHECK-SAME:                                                             strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x3x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[MAXPOOL]] : !VPU.DistributedTensor<1x3x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK-SAME{LITERAL}:                                                    -> tensor<1x3x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x3x112x112xf16, {order = #NHWC}>


}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @EltwiseAddMulticlusterSOHOverlapped
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x16x112x112xf16, {order = #NHWC}>,
// CHECK-SAME:   [[ARG1:%.+]]: tensor<1x16x112x112xf16, {order = #NHWC}>)
func.func @EltwiseAddMulticlusterSOHOverlapped(%arg0: tensor<1x16x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x16x112x112xf16, {order = #NHWC}>) -> tensor<1x3x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, input_padding = [0, 13, 0, 0], output_padding = [0, 0, 0, 0]} :
         tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<1x16x112x112xf16, {order = #NHWC}>
         -> tensor<1x3x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x3x112x112xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x16x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 16, 38, 112], [1, 16, 37, 112], [1, 16, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 16, 38, 112], [1, 16, 37, 112], [1, 16, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[IN_CP1:%.+]] = VPU.UnrolledType([[ARG1]] : tensor<1x16x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 16, 38, 112], [1, 16, 37, 112], [1, 16, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 16, 38, 112], [1, 16, 37, 112], [1, 16, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[IN_CP0]], [[IN_CP1]]) {input_padding = [0, 13, 0, 0], op_type = #VPU.eltwise_type<ADD>, output_padding = [0, 0, 0, 0], ppe = #VPU.PPEStub<>}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x3x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[ELTWISE]] : !VPU.DistributedTensor<1x3x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK-SAME{LITERAL}:                                    -> tensor<1x3x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x3x112x112xf16, {order = #NHWC}>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @AvgPoolMulticlusterSOHOverlapped
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x16x112x112xf16, {order = #NHWC}>
func.func @AvgPoolMulticlusterSOHOverlapped(%arg0: tensor<1x16x112x112xf16, {order = #NHWC}>) -> tensor<1x3x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [3, 3],
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 0, 0, 0]
         } -> tensor<1x3x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x3x112x112xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x16x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 16, 38, 112], [1, 16, 37, 112], [1, 16, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 16, 39, 112], [1, 16, 39, 112], [1, 16, 38, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 37, 0], [0, 0, 74, 0]]}>
// CHECK:               [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[IN_CP0]]) {input_padding = [0, 13, 0, 0], kernel_size = [3, 3], output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x3x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[AVGPOOL]] : !VPU.DistributedTensor<1x3x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 3, 38, 112], [1, 3, 37, 112], [1, 3, 37, 112]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 75, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x3x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x3x112x112xf16, {order = #NHWC}>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @PermuteMulticlusterSOKWithChannelAlignment
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x48x16x16xf16>
func.func @PermuteMulticlusterSOKWithChannelAlignment(%arg0: tensor<1x48x16x16xf16>) -> tensor<1x48x16x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 48 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        ppe = #VPU.PPEStub<>
        } -> tensor<1x48x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x48x16x16xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x48x16x16xf16>)
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x48x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]}>
// CHECK:               [[PERMUTE:%.+]] = VPU.NCE.Permute([[IN_CP0]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64, ppe = #VPU.PPEStub<>}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x48x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[PERMUTE]] : !VPU.DistributedTensor<1x48x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x48x16x16xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x48x16x16xf16, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @PermuteMulticlusterSOKWithoutChannelAlignment
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x49x16x16xf16>
func.func @PermuteMulticlusterSOKWithoutChannelAlignment(%arg0: tensor<1x49x16x16xf16>) -> tensor<1x49x16x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 49 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        ppe = #VPU.PPEStub<>
        } -> tensor<1x49x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x49x16x16xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x49x16x16xf16>)
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x49x16x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 17, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 17, 0, 0], [0, 33, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 17, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 17, 0, 0], [0, 33, 0, 0]]}>
// CHECK:               [[PERMUTE:%.+]] = VPU.NCE.Permute([[IN_CP0]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 49 : i64, ppe = #VPU.PPEStub<>}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x49x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 17, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 17, 0, 0], [0, 33, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 17, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 17, 0, 0], [0, 33, 0, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[PERMUTE]] : !VPU.DistributedTensor<1x49x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 17, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 17, 0, 0], [0, 33, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 17, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 17, 0, 0], [0, 33, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x49x16x16xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x49x16x16xf16, {order = #NHWC}>
}
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @PermuteMulticlusterSOKWithAlignedParent
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x64x64x64xf16, {order = #NHWC}>
func.func @PermuteMulticlusterSOKWithAlignedParent(%arg0: tensor<1x64x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) rawFilterShape [64, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
             strides = [1, 1]
            } : tensor<1x64x64x64xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x64x64xf16>
    %1 = VPU.MVN(%0) {
            across_channels = false, eps = 1.0E-4 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true
            } : tensor<1x64x64x64xf16>
                -> tensor<1x64x64x64xf16>
    %2 = VPU.NCE.Permute(%1) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 64 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        ppe = #VPU.PPEStub<>
        } -> tensor<1x64x64x64xf16, {order = #NHWC}>
    return %2 : tensor<1x64x64x64xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x64x64x64xf16, {order = #NHWC}>)
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:                                                    uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64]]
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 64, 64, 64], [1, 64, 64, 64], [1, 64, 64, 64]]
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:              [[CONV:%.+]] = VPU.NCE.Convolution([[IN_CP0]]
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x64x64x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>
// CHECK:              [[INTER_CP0:%.+]] = VPU.UnrolledType([[CONV]] : !VPU.DistributedTensor<1x64x64x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> tensor<1x64x64x64xf16>
// CHECK:              [[INTER_CP1:%.+]] = VPU.UnrolledType([[INTER_CP0]] : tensor<1x64x64x64xf16>)
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x64x64x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>
// CHECK:              [[MVN:%.+]] = VPU.MVN([[INTER_CP1]])
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x64x64x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>
// CHECK:              [[INTER_CP2:%.+]] = VPU.UnrolledType([[MVN]] : !VPU.DistributedTensor<1x64x64x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> tensor<1x64x64x64xf16>
// CHECK:              [[INTER_CP3:%.+]] = VPU.UnrolledType([[INTER_CP2]] : tensor<1x64x64x64xf16>)
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x64x64x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>
// CHECK:              [[PERMUTE:%.+]] = VPU.NCE.Permute([[INTER_CP3]])
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>
// CHECK:              [[OUT_CP:%.+]] = VPU.UnrolledType([[PERMUTE]] : !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:                                                    alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 32, 64, 64], [1, 16, 64, 64], [1, 16, 64, 64]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>
// CHECK-SAME{LITERAL}:                                                     -> tensor<1x64x64x64xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x64x64x64xf16, {order = #NHWC}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @ReverseSequenceMulticlusterSplitOverHeight
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x1x48x16xf16>,
// CHECK-SAME:   [[ARG1:%.+]]: tensor<1x1x1x1xsi32>)
func.func @ReverseSequenceMulticlusterSplitOverHeight(%arg0: tensor<1x1x48x16xf16>, %arg1: tensor<1x1x1x1xsi32>) -> tensor<1x1x48x16xf16, {order = #NCHW}> {
    %0 = VPU.ReverseSequence(%arg0, %arg1) {batch_axis = 0 : i64, seq_axis = 2 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
         tensor<1x1x48x16xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x48x16xf16, {order = #NCHW}>
    return %0 : tensor<1x1x48x16xf16, {order = #NCHW}>
// CHECK:               [[IN_CP0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x1x48x16xf16>)
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x1x48x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 1, 16, 16], [1, 1, 16, 16], [1, 1, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 1, 16, 16], [1, 1, 16, 16], [1, 1, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0]]}>
// CHECK:               [[IN_CP1:%.+]] = VPU.UnrolledType([[ARG1]] : tensor<1x1x1x1xsi32>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x1x1x1xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:               [[REVERSESEQ:%.+]] = VPU.ReverseSequence([[IN_CP0]], [[IN_CP1]]) {batch_axis = 0 : i64, seq_axis = 2 : i64}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x1x48x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 1, 16, 16], [1, 1, 16, 16], [1, 1, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 1, 16, 16], [1, 1, 16, 16], [1, 1, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0]]}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[REVERSESEQ]] : !VPU.DistributedTensor<1x1x48x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:                                                    compute_shapes = [[1, 1, 16, 16], [1, 1, 16, 16], [1, 1, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0]],
// CHECK-SAME{LITERAL}:                                                    memory_shapes = [[1, 1, 16, 16], [1, 1, 16, 16], [1, 1, 16, 16]],
// CHECK-SAME{LITERAL}:                                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0]]}>
// CHECK-SAME{LITERAL}:                                    -> tensor<1x1x48x16xf16, {order = #NCHW}>
// CHECK:        return [[OUT_CP]] : tensor<1x1x48x16xf16, {order = #NCHW}>
}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_SoH
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<1x8x64x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<1x8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<1x8x32x128xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<1x8x64x32xf16>
func.func @FlashSDPA_SoH(%arg0: tensor<1x8x64x64xf16>, %arg1: tensor<1x8x32x64xf16>, %arg2: tensor<1x8x32x128xf16>, %arg3: tensor<1x8x64x32xf16>) -> tensor<1x8x64x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x32x4xsi32> = dense<0> : tensor<1x1x32x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x2x64x32xf16> = dense<0.000000e+00> : tensor<1x2x64x32xf16>
    %cst_3 = const.Declare tensor<1x8x64x1xf32> = dense<0.000000e+00> : tensor<1x8x64x1xf32>
    %cst_4 = const.Declare tensor<1x8x64x1xf16> = dense<0xFC00> : tensor<1x8x64x1xf16>
    %cst_5 = const.Declare tensor<1x8x64x128xf16> = dense<0.000000e+00> : tensor<1x8x64x128xf16>

    %value_reordered = IE.Reorder(%arg2) {dstOrder = #NCWH} : tensor<1x8x32x128xf16> -> tensor<1x8x32x128xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x8x64x64xf16>, tensor<1x8x32x64xf16>, tensor<1x8x32x128xf16, {order = #NCWH}>,
            tensor<1x2x64x32xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x32x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x8x64x128xf16>, tensor<1x8x64x1xf16>,
            tensor<1x8x64x1xf32>, tensor<1x8x64x32xf16>
        -> tensor<1x8x64x128xf16>, tensor<1x8x64x1xf16>, tensor<1x8x64x1xf32>

    return %result_running_output : tensor<1x8x64x128xf16>

    // CHECK-DAG:       [[DPU_DESCRIPTORS_BUF:%.+]] = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    // CHECK-DAG:       [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<1x1x32x4xsi32> = dense
    // CHECK-DAG:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<1x1x128x4xsi32> = dense
    // CHECK-DAG:       [[IN_AUX:%.+]] = const.Declare tensor<1x2x64x32xf16> = dense<0.000000e+00> : tensor<1x2x64x32xf16>
    // CHECK-DAG:       [[IN_SUM:%.+]] = const.Declare tensor<1x8x64x1xf32> = dense<0.000000e+00> : tensor<1x8x64x1xf32>
    // CHECK-DAG:       [[IN_MAX:%.+]] = const.Declare tensor<1x8x64x1xf16> = dense<0xFC00> : tensor<1x8x64x1xf16>
    // CHECK-DAG:       [[IN_OUT:%.+]] = const.Declare tensor<1x8x64x128xf16> = dense<0.000000e+00> : tensor<1x8x64x128xf16>

    // CHECK-DAG:       [[VALUE_REORDERED:%.+]] = IE.Reorder([[VALUE]]) {dstOrder = #NCWH} : tensor<1x8x32x128xf16> -> tensor<1x8x32x128xf16, {order = #NCWH}>

    // CHECK:           [[IN0:%.+]] = VPU.UnrolledType([[QUERY]] : tensor<1x8x64x64xf16>) -> !VPU.DistributedTensor<1x8x64x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN1:%.+]] = VPU.UnrolledType([[KEY]] : tensor<1x8x32x64xf16>) -> !VPU.DistributedTensor<1x8x32x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED"
    // CHECK:           [[IN2:%.+]] = VPU.UnrolledType([[VALUE_REORDERED]] : tensor<1x8x32x128xf16, {order = #NCWH}>) -> !VPU.DistributedTensor<1x8x32x128xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED"
    // CHECK:           [[IN3:%.+]] = VPU.UnrolledType([[IN_AUX]] : tensor<1x2x64x32xf16>) -> !VPU.DistributedTensor<1x2x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN4:%.+]] = VPU.UnrolledType([[DPU_DESCRIPTORS_BUF]] : tensor<1x1x2x256xsi32>) -> !VPU.DistributedTensor<1x1x2x256xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED"
    // CHECK:           [[IN5:%.+]] = VPU.UnrolledType([[WEIGHTS_TABLE0]] : tensor<1x1x32x4xsi32>) -> !VPU.DistributedTensor<1x1x32x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED"
    // CHECK:           [[IN6:%.+]] = VPU.UnrolledType([[WEIGHTS_TABLE1]] : tensor<1x1x128x4xsi32>) -> !VPU.DistributedTensor<1x1x128x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED"
    // CHECK:           [[IN7:%.+]] = VPU.UnrolledType([[IN_OUT]] : tensor<1x8x64x128xf16>) -> !VPU.DistributedTensor<1x8x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN8:%.+]] = VPU.UnrolledType([[IN_MAX]] : tensor<1x8x64x1xf16>) -> !VPU.DistributedTensor<1x8x64x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN9:%.+]] = VPU.UnrolledType([[IN_SUM]] : tensor<1x8x64x1xf32>) -> !VPU.DistributedTensor<1x8x64x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN10:%.+]] = VPU.UnrolledType([[ATTENTION_MASK]] : tensor<1x8x64x32xf16>) -> !VPU.DistributedTensor<1x8x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"

    // CHECK:           [[RES_OUT_DIST:%[^, ]+]], [[RES_MAX_DIST:%[^, ]+]], [[RES_SUM_DIST:%[^, ]+]] =
    // CHECK-SAME:          VPU.FlashSDPA([[IN0]], [[IN1]], [[IN2]], [[IN3]], [[IN4]], [[IN5]], [[IN6]], [[IN7]], [[IN8]], [[IN9]], [[IN10]]) {
    // CHECK-SAME:              is_head = true,
    // CHECK-SAME:              is_tail = true,
    // CHECK-SAME:              operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>
    // CHECK-SAME:              source_seq_len_pad_size = 0 : i64
    // CHECK-SAME:          }
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x8x64x128xf16
    // CHECK-SAME:             !VPU.DistributedTensor<1x8x64x1xf16
    // CHECK-SAME:             !VPU.DistributedTensor<1x8x64x1xf32

    // CHECK:           [[RES_OUT:%.+]] = VPU.UnrolledType([[RES_OUT_DIST]]
    // CHECK-SAME:          : !VPU.DistributedTensor<1x8x64x128xf16
    // CHECK:           [[RES_MAX:%.+]] = VPU.UnrolledType([[RES_MAX_DIST]]
    // CHECK-SAME:          : !VPU.DistributedTensor<1x8x64x1xf16
    // CHECK:           [[RES_SUM:%.+]] = VPU.UnrolledType([[RES_SUM_DIST]]
    // CHECK-SAME:          : !VPU.DistributedTensor<1x8x64x1xf32

    //CHECK:            return [[RES_OUT]] : tensor<1x8x64x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: @FlashSDPA_SoK
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<1x8x1x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<1x8x32x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<1x8x32x128xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<1x8x1x32xf16>
func.func @FlashSDPA_SoK(%arg0: tensor<1x8x1x64xf16>, %arg1: tensor<1x8x32x64xf16>, %arg2: tensor<1x8x32x128xf16>, %arg3: tensor<1x8x1x32xf16>) -> tensor<1x8x1x128xf16> {
    %cst = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>
    %cst_0 = const.Declare tensor<1x1x32x4xsi32> = dense<0> : tensor<1x1x32x4xsi32>
    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x2x1x32xf16> = dense<0.000000e+00> : tensor<1x2x1x32xf16>
    %cst_3 = const.Declare tensor<1x8x1x1xf32> = dense<0.000000e+00> : tensor<1x8x1x1xf32>
    %cst_4 = const.Declare tensor<1x8x1x1xf16> = dense<0xFC00> : tensor<1x8x1x1xf16>
    %cst_5 = const.Declare tensor<1x8x1x128xf16> = dense<0.000000e+00> : tensor<1x8x1x128xf16>

    %value_reordered = IE.Reorder(%arg2) {dstOrder = #NCWH} : tensor<1x8x32x128xf16> -> tensor<1x8x32x128xf16, {order = #NCWH}>

    %result_running_output, %result_running_max, %result_running_sum =
        VPU.FlashSDPA(%arg0, %arg1, %value_reordered, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
            is_head = true,
            is_tail = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,
            source_seq_len_pad_size = 0 : i64
        } : tensor<1x8x1x64xf16>, tensor<1x8x32x64xf16>, tensor<1x8x32x128xf16, {order = #NCWH}>,
            tensor<1x2x1x32xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x32x4xsi32>,
            tensor<1x1x128x4xsi32>, tensor<1x8x1x128xf16>, tensor<1x8x1x1xf16>,
            tensor<1x8x1x1xf32>, tensor<1x8x1x32xf16>
        -> tensor<1x8x1x128xf16>, tensor<1x8x1x1xf16>, tensor<1x8x1x1xf32>

    return %result_running_output : tensor<1x8x1x128xf16>

    // CHECK-DAG:       [[DPU_DESCRIPTORS_BUF:%.+]] = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    // CHECK-DAG:       [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<1x1x32x4xsi32> = dense
    // CHECK-DAG:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<1x1x128x4xsi32> = dense
    // CHECK-DAG:       [[IN_AUX:%.+]] = const.Declare tensor<1x2x1x32xf16> = dense<0.000000e+00> : tensor<1x2x1x32xf16>
    // CHECK-DAG:       [[IN_SUM:%.+]] = const.Declare tensor<1x8x1x1xf32> = dense<0.000000e+00> : tensor<1x8x1x1xf32>
    // CHECK-DAG:       [[IN_MAX:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<0xFC00> : tensor<1x8x1x1xf16>
    // CHECK-DAG:       [[IN_OUT:%.+]] = const.Declare tensor<1x8x1x128xf16> = dense<0.000000e+00> : tensor<1x8x1x128xf16>

    // CHECK-DAG:       [[VALUE_REORDERED:%.+]] = IE.Reorder([[VALUE]]) {dstOrder = #NCWH} : tensor<1x8x32x128xf16> -> tensor<1x8x32x128xf16, {order = #NCWH}>

    // CHECK:           [[IN0:%.+]] = VPU.UnrolledType([[QUERY]] : tensor<1x8x1x64xf16>) -> !VPU.DistributedTensor<1x8x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN1:%.+]] = VPU.UnrolledType([[KEY]] : tensor<1x8x32x64xf16>) -> !VPU.DistributedTensor<1x8x32x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN2:%.+]] = VPU.UnrolledType([[VALUE_REORDERED]] : tensor<1x8x32x128xf16, {order = #NCWH}>) -> !VPU.DistributedTensor<1x8x32x128xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN3:%.+]] = VPU.UnrolledType([[IN_AUX]] : tensor<1x2x1x32xf16>) -> !VPU.DistributedTensor<1x2x1x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED"
    // CHECK:           [[IN4:%.+]] = VPU.UnrolledType([[DPU_DESCRIPTORS_BUF]] : tensor<1x1x2x256xsi32>) -> !VPU.DistributedTensor<1x1x2x256xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED"
    // CHECK:           [[IN5:%.+]] = VPU.UnrolledType([[WEIGHTS_TABLE0]] : tensor<1x1x32x4xsi32>) -> !VPU.DistributedTensor<1x1x32x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED"
    // CHECK:           [[IN6:%.+]] = VPU.UnrolledType([[WEIGHTS_TABLE1]] : tensor<1x1x128x4xsi32>) -> !VPU.DistributedTensor<1x1x128x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED"
    // CHECK:           [[IN7:%.+]] = VPU.UnrolledType([[IN_OUT]] : tensor<1x8x1x128xf16>) -> !VPU.DistributedTensor<1x8x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN8:%.+]] = VPU.UnrolledType([[IN_MAX]] : tensor<1x8x1x1xf16>) -> !VPU.DistributedTensor<1x8x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN9:%.+]] = VPU.UnrolledType([[IN_SUM]] : tensor<1x8x1x1xf32>) -> !VPU.DistributedTensor<1x8x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:           [[IN10:%.+]] = VPU.UnrolledType([[ATTENTION_MASK]] : tensor<1x8x1x32xf16>) -> !VPU.DistributedTensor<1x8x1x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"

    // CHECK:           [[RES_OUT_DIST:%[^, ]+]], [[RES_MAX_DIST:%[^, ]+]], [[RES_SUM_DIST:%[^, ]+]] =
    // CHECK-SAME:          VPU.FlashSDPA([[IN0]], [[IN1]], [[IN2]], [[IN3]], [[IN4]], [[IN5]], [[IN6]], [[IN7]], [[IN8]], [[IN9]], [[IN10]]) {
    // CHECK-SAME:              is_head = true,
    // CHECK-SAME:              is_tail = true,
    // CHECK-SAME:              operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>
    // CHECK-SAME:              source_seq_len_pad_size = 0 : i64
    // CHECK-SAME:          }
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x8x1x128xf16
    // CHECK-SAME:             !VPU.DistributedTensor<1x8x1x1xf16
    // CHECK-SAME:             !VPU.DistributedTensor<1x8x1x1xf32

    // CHECK:           [[RES_OUT:%.+]] = VPU.UnrolledType([[RES_OUT_DIST]]
    // CHECK-SAME:          : !VPU.DistributedTensor<1x8x1x128xf16
    // CHECK:           [[RES_MAX:%.+]] = VPU.UnrolledType([[RES_MAX_DIST]]
    // CHECK-SAME:          : !VPU.DistributedTensor<1x8x1x1xf16
    // CHECK:           [[RES_SUM:%.+]] = VPU.UnrolledType([[RES_SUM_DIST]]
    // CHECK-SAME:          : !VPU.DistributedTensor<1x8x1x1xf32

    //CHECK:            return [[RES_OUT]] : tensor<1x8x1x128xf16>
}



// -----

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SubByteInputNeedAlignment
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x1x512x1xui4>

func.func @SubByteInputNeedAlignment(%arg0: tensor<1x1x512x1xui4>) -> tensor<1x1x512x1xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x512x1xui4> -> tensor<1x1x512x1xf16>

    return %0 : tensor<1x1x512x1xf16>

    // CHECK:               [[IN_CP:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x1x512x1xui4>)
    // CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x1x512x1xui4, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]],
    // CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]]}>

    // CHECK:               [[CONVERT:%.+]] = VPU.Convert([[IN_CP]]) {dstElemType = f16}
    // CHECK-SAME{LITERAL}:                                                : !VPU.DistributedTensor<1x1x512x1xui4, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]],
    // CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]]}>
    // CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x1x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]],
    // CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]]}>

    // CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[CONVERT]] : !VPU.DistributedTensor<1x1x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                                                compute_shapes = [[1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME{LITERAL}:                                                compute_offsets = [[0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]],
    // CHECK-SAME{LITERAL}:                                                memory_shapes = [[1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME{LITERAL}:                                                memory_offsets = [[0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]]}>
    // CHECK-SAME{LITERAL}:                                                -> tensor<1x1x512x1xf16>
    // CHECK:               return [[OUT_CP]] : tensor<1x1x512x1xf16>
}
}

// -----

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SubByteOutputNeedAlignment
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x1x512x1xf16>

func.func @SubByteOutputNeedAlignment(%arg0: tensor<1x1x512x1xf16>) -> tensor<1x1x512x1xui4> {
    %0 = VPU.Convert(%arg0) {dstElemType = ui4, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x512x1xf16> -> tensor<1x1x512x1xui4>

    return %0 : tensor<1x1x512x1xui4>

    // CHECK:       [[UNROLLED_0:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x1x512x1xf16>)
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x1x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME:          compute_shapes = {{\[\[}}1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME:          compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]],
    // CHECK-SAME:          memory_shapes = {{\[\[}}1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME:          memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]]}>

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[UNROLLED_0]]) {dstElemType = ui4}
    // CHECK-SAME:      !VPU.DistributedTensor<1x1x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME:          compute_shapes = {{\[\[}}1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME:          compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]],
    // CHECK-SAME:          memory_shapes = {{\[\[}}1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME:          memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]]}>
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x1x512x1xui4, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME:          compute_shapes = {{\[\[}}1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME:          compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]],
    // CHECK-SAME:          memory_shapes = {{\[\[}}1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME:          memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]]}>

    // CHECK:       [[UNROLLED_1:%.+]] = VPU.UnrolledType([[CONVERT]]
    // CHECK-SAME:      !VPU.DistributedTensor<1x1x512x1xui4, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME:          compute_shapes = {{\[\[}}1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME:          compute_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]],
    // CHECK-SAME:          memory_shapes = {{\[\[}}1, 1, 172, 1], [1, 1, 170, 1], [1, 1, 170, 1]],
    // CHECK-SAME:          memory_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 172, 0], [0, 0, 342, 0]]}>)
    // CHECK-SAME:      -> tensor<1x1x512x1xui4>

    // CHECK:       return [[UNROLLED_1]] : tensor<1x1x512x1xui4>
}
}



// -----


module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @GatherDMASOB
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<184320x4x1x1xsi4>, [[INDICES:%.+]]: tensor<12x1x1x1xi64>)
func.func @GatherDMASOB(%input: tensor<184320x4x1x1xsi4>, %indices: tensor<12x1x1x1xi64>) -> tensor<12x4x1x1xsi4> {
    %gatherDMA = VPU.GatherDMA(%input, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>} : tensor<184320x4x1x1xsi4>, tensor<12x1x1x1xi64> -> tensor<12x4x1x1xsi4>
    return %gatherDMA : tensor<12x4x1x1xsi4>

    // CHECK:       [[INPUT_UNROLLED:%.+]]  = VPU.UnrolledType([[INPUT]] : tensor<184320x4x1x1xsi4>) -> tensor<184320x4x1x1xsi4>

    // CHECK:       [[INDICES_UNROLLED:%.+]] = VPU.UnrolledType([[INDICES]] : tensor<12x1x1x1xi64>) -> !VPU.DistributedTensor<12x1x1x1xi64, #NCHW, @CMX_NN, {
    //CHECK-SAME{LITERAL}:          mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:          compute_shapes = [[4, 1, 1, 1], [4, 1, 1, 1], [4, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:          memory_shapes = [[4, 1, 1, 1], [4, 1, 1, 1], [4, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]]}>

    // CHECK:       [[GATHERDMA:%.+]] = VPU.GatherDMA([[INPUT_UNROLLED]], [[INDICES_UNROLLED]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<184320x4x1x1xsi4>,
    // CHECK-SAME               !VPU.DistributedTensor<12x1x1x1xi64, #NCHW, @CMX_NN, {
    //CHECK-SAME{LITERAL}:          mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:          compute_shapes = [[4, 1, 1, 1], [4, 1, 1, 1], [4, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:          memory_shapes = [[4, 1, 1, 1], [4, 1, 1, 1], [4, 1, 1, 1]],
    //CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]]}>
    //CHECK-SAME                -> !VPU.DistributedTensor<12x4x1x1xsi4, #NCHW, @CMX_NN, {
    //CHECK-SAME{LITERAL}:          mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:          compute_shapes = [[4, 4, 1, 1], [4, 4, 1, 1], [4, 4, 1, 1]],
    //CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:          memory_shapes = [[4, 4, 1, 1], [4, 4, 1, 1], [4, 4, 1, 1]],
    //CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]]}>

    // CHECK:       [[OUTPUT_UNROLLED:%.+]] = VPU.UnrolledType([[GATHERDMA]] : !VPU.DistributedTensor<12x4x1x1xsi4, #NCHW, @CMX_NN, {
    //CHECK-SAME{LITERAL}:          mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:          compute_shapes = [[4, 4, 1, 1], [4, 4, 1, 1], [4, 4, 1, 1]],
    //CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:          memory_shapes = [[4, 4, 1, 1], [4, 4, 1, 1], [4, 4, 1, 1]],
    //CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]]}>) -> tensor<12x4x1x1xsi4>

    // CHECK:       return [[OUTPUT_UNROLLED]] : tensor<12x4x1x1xsi4>
}
}
