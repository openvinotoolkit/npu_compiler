//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor="enable-explicit-distributed-attr=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @SWEltwiseNeedAlignmentSOW
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x3x1x255xf16>
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<1x3x1x1xf16>
func.func @SWEltwiseNeedAlignmentSOW(%arg0: tensor<1x3x1x255xf16>, %arg1: tensor<1x3x1x1xf16>) -> tensor<1x3x1x255xf16> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>} : tensor<1x3x1x255xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x1x255xf16>
    return %0 : tensor<1x3x1x255xf16>

    // CHECK:     [[DATA0:%.+]] = VPU.UnrolledType([[INPUT0]] : tensor<1x3x1x255xf16>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x3x1x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, alignment = [1, 1, 1, 32], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]}>

    // CHECK:     [[DATA1:%.+]] = VPU.UnrolledType([[INPUT1]] : tensor<1x3x1x1xf16>)
    // CHECK-SAME:           -> !VPU.DistributedTensor<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 1, 1], [1, 3, 1, 1], [1, 3, 1, 1]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 1, 1], [1, 3, 1, 1], [1, 3, 1, 1]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:     [[MUL:%.+]] = VPU.Multiply([[DATA0]], [[DATA1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:  -> !VPU.DistributedTensor<1x3x1x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, alignment = [1, 1, 1, 32], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]}>

    // CHECK:     [[OUT:%.+]] = VPU.UnrolledType([[MUL]] : !VPU.DistributedTensor<1x3x1x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, alignment = [1, 1, 1, 32], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 1, 96], [1, 3, 1, 96], [1, 3, 1, 63]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192]]}>)
    // CHECK-SAME:           -> tensor<1x3x1x255xf16>

    // CHECK:     return [[OUT]] : tensor<1x3x1x255xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.0>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @INT8DepthConvWith32AlignmentSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x256x28x28x!qElemType, {order = #NHWC}>
func.func @INT8DepthConvWith32AlignmentSOK(%input: tensor<1x256x28x28x!qElemType, {order = #NHWC}>) -> tensor<1x256x28x28x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<256x16x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<256x1x1x3x3xsi8>, [#const.Reshape<[1, 256, 3, 3]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.AffineReshape<[[0], [0], [1, 2], [3]], [256, 1, 3, 3]>, #const.ConvertElemType<!qElemType>, #const.Reshape<[256, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%input, %weights) rawFilterShape [256, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,  strides = [1, 1]} -> tensor<1x256x28x28x!qElemType, {order = #NHWC}>
    return %0 : tensor<1x256x28x28x!qElemType, {order = #NHWC}>

    // CHECK:               [[WEIGHTS:%.+]] = const.Declare tensor<256x16x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<256x1x1x3x3xsi8>,
    // CHECK-SAME{LITERAL}:     [#const.Reshape<[1, 256, 3, 3]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.AffineReshape<[[0], [0], [1, 2], [3]], [256, 1, 3, 3]>, #const.ConvertElemType<!qElemType>, #const.Reshape<[256, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    // CHECK:               [[UNROLLED_INPUT:%.+]] = VPU.UnrolledType([[INPUT]] : tensor<1x256x28x28x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:              -> !VPU.DistributedTensor<1x256x28x28x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 32, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 64, 28, 28]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 64, 28, 28]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0]]}>

    // CHECK:               [[UNROLLED_WEIGHTS:%.+]] = VPU.UnrolledType([[WEIGHTS]] : tensor<256x16x1x1x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:              -> !VPU.DistributedTensor<256x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [32, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[96, 16, 1, 1], [96, 16, 1, 1], [64, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[96, 16, 1, 1], [96, 16, 1, 1], [64, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]]}>

    // CHECK:               [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[UNROLLED_INPUT]], [[UNROLLED_WEIGHTS]]) rawFilterShape [256, 1, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
    // CHECK-SAME:              -> !VPU.DistributedTensor<1x256x28x28x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 32, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 64, 28, 28]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 256, 28, 28], [1, 256, 28, 28], [1, 256, 28, 28]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:               [[UNROLLED_OUTPUT:%.+]] = VPU.UnrolledType([[DWCONV]] :
    // CHECK-SAME:              !VPU.DistributedTensor<1x256x28x28x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 32, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 96, 28, 28], [1, 96, 28, 28], [1, 64, 28, 28]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 256, 28, 28], [1, 256, 28, 28], [1, 256, 28, 28]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:              -> tensor<1x256x28x28x!qElemType, {order = #NHWC}>

    // CHECK:               return [[UNROLLED_OUTPUT]]
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.0>

module @executors {
config.Resources 3 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @INT8DepthConvWith16AlignmentSOK
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x48x28x28x!qElemType, {order = #NHWC}>
func.func @INT8DepthConvWith16AlignmentSOK(%input: tensor<1x48x28x28x!qElemType, {order = #NHWC}>) -> tensor<1x48x28x28x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<48x16x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<48x1x1x3x3xsi8>, [#const.Reshape<[1, 48, 3, 3]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.AffineReshape<[[0], [0], [1, 2], [3]], [48, 1, 3, 3]>, #const.ConvertElemType<!qElemType>, #const.Reshape<[48, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%input, %weights) rawFilterShape [48, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,  strides = [1, 1]} -> tensor<1x48x28x28x!qElemType, {order = #NHWC}>
    return %0 : tensor<1x48x28x28x!qElemType, {order = #NHWC}>

    // CHECK:               [[WEIGHTS:%.+]] = const.Declare tensor<48x16x1x1x!qElemType, {order = #NHWC}> = dense<1> : tensor<48x1x1x3x3xsi8>,
    // CHECK-SAME{LITERAL}:     [#const.Reshape<[1, 48, 3, 3]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.AffineReshape<[[0], [0], [1, 2], [3]], [48, 1, 3, 3]>, #const.ConvertElemType<!qElemType>, #const.Reshape<[48, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    // CHECK:               [[UNROLLED_INPUT:%.+]] = VPU.UnrolledType([[INPUT]] : tensor<1x48x28x28x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:              -> !VPU.DistributedTensor<1x48x28x28x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]}>

    // CHECK:               [[UNROLLED_WEIGHTS:%.+]] = VPU.UnrolledType([[WEIGHTS]] : tensor<48x16x1x1x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:              -> !VPU.DistributedTensor<48x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0]]}>

    // CHECK:               [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[UNROLLED_INPUT]], [[UNROLLED_WEIGHTS]]) rawFilterShape [48, 1, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
    // CHECK-SAME:              -> !VPU.DistributedTensor<1x48x28x28x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:               [[UNROLLED_OUTPUT:%.+]] = VPU.UnrolledType([[DWCONV]] :
    // CHECK-SAME:              !VPU.DistributedTensor<1x48x28x28x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 28, 28], [1, 16, 28, 28], [1, 16, 28, 28]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:              -> tensor<1x48x28x28x!qElemType, {order = #NHWC}>

    // CHECK:               return [[UNROLLED_OUTPUT]]
}

}

// -----

// si4 per-cluster sizes must be even (2 elements per byte).
// [1, 1, 4094, 1] si4 over 4 clusters → [1024, 1024, 1024, 1022], all byte-aligned.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvertSi4ToF16ByteAlignedTiling4Cluster
func.func @ConvertSi4ToF16ByteAlignedTiling4Cluster(%arg0: tensor<1x1x4094x1xsi4>) -> tensor<1x1x4094x1xf16> {
    %0 = VPU.Convert(%arg0) {
        dstElemType = f16,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } : tensor<1x1x4094x1xsi4> -> tensor<1x1x4094x1xf16>
    return %0 : tensor<1x1x4094x1xf16>

    // CHECK:           [[IN:%.+]] = VPU.UnrolledType([[ARG0:%.+]] : tensor<1x1x4094x1xsi4>)
    // CHECK-SAME{LITERAL}:        -> !VPU.DistributedTensor<1x1x4094x1xsi4, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1]
    // CHECK-SAME{LITERAL}:        num_clusters = 4 : i64, alignment = [1, 1, 2, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1022, 1]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0], [0, 0, 3072, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1022, 1]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0], [0, 0, 3072, 0]]}>

    // CHECK:           [[CONV:%.+]] = VPU.Convert([[IN]]) {dstElemType = f16}
    // CHECK-SAME{LITERAL}:        -> !VPU.DistributedTensor<1x1x4094x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1],
    // CHECK-SAME{LITERAL}:        num_clusters = 4 : i64, alignment = [1, 1, 2, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:        compute_shapes = [[1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1022, 1]],
    // CHECK-SAME{LITERAL}:        compute_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0], [0, 0, 3072, 0]],
    // CHECK-SAME{LITERAL}:        memory_shapes = [[1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1024, 1], [1, 1, 1022, 1]],
    // CHECK-SAME{LITERAL}:        memory_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0], [0, 0, 3072, 0]]}>

    // CHECK:           [[OUT:%.+]] = VPU.UnrolledType([[CONV]] :
    // CHECK-SAME:          !VPU.DistributedTensor<1x1x4094x1xf16
    // CHECK-SAME:          -> tensor<1x1x4094x1xf16>
    // CHECK:           return [[OUT]] : tensor<1x1x4094x1xf16>
}
}
