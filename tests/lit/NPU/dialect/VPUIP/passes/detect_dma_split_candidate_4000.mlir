//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt  --split-input-file --init-compiler="platform=%platform%" --detect-dma-split-candidate %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DummyT = memref<1x3x224x224xf16, @DDR>

!DistributedType = !VPUIP.DistributedBuffer<
    1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @AssignSplitCandidateToBroadcastDMA
func.func @AssignSplitCandidateToBroadcastDMA(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %cst = const.Declare memref<1x1x1x368768xf16> = dense<1.0> : memref<1x1x1x368768xf16>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3, 4, 5] <163840> -> !DistributedType

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %3 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst : memref<1x1x1x368768xf16>) outputs(%2 : !DistributedType) -> !DistributedType
    }

    // CHECK:       [[NNDMA:%.+]] = VPUIP.NNDMA <{port = 0 : i64, split_candidate = true}>

    return %arg0 : !DummyT
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DummyT = memref<1x3x224x224xf16, @DDR>

!DistributedType = !VPUIP.DistributedBuffer<
    1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// Port 0:           |---------- DMA 1 ----------|
// Port 1: |- DMA 0 -|                           |- DMA 2 -|
// CHECK-LABEL: @AssignSplitCandidateWhenAnotherPortIsTotallyIdle
func.func @AssignSplitCandidateWhenAnotherPortIsTotallyIdle(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %2 = VPURT.DeclareBuffer <CMX_NN> [5] <901376> -> memref<1x16x1x8xf16, [@CMX_NN, 5]>

    %3 = VPURT.DeclareBuffer <DDR> <94320> -> memref<1x16x1x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3, 4, 5] <163840> -> !DistributedType
    %5 = VPURT.DeclareBuffer <CMX_NN> [1] <901888> -> memref<1x16x2x8xf16, [@CMX_NN, 1]>
    %6 = VPURT.DeclareBuffer <DDR> <96288> -> memref<1x16x2x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>
    %cst = const.Declare memref<1x1x1x368768xf16> = dense<1.0> : memref<1x1x1x368768xf16>

    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %7 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%2 : memref<1x16x1x8xf16, [@CMX_NN, 5]>) outputs(%3 : memref<1x16x1x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>) -> memref<1x16x1x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>
    }
    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %7 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst : memref<1x1x1x368768xf16>) outputs(%4 : !DistributedType) -> !DistributedType
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %7 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%5 : memref<1x16x2x8xf16, [@CMX_NN, 1]>) outputs(%6 : memref<1x16x2x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>) -> memref<1x16x2x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>
    }

    // CHECK:       [[NNDMA:%.+]] = VPUIP.NNDMA <{port = 0 : i64, split_candidate = true}>

    return %arg0 : !DummyT
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DummyT = memref<1x3x224x224xf16, @DDR>

!DistributedType = !VPUIP.DistributedBuffer<
    1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768], [1, 1, 1, 368768]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// Port 0: |-------------- DMA 0 --------------|
// Port 1: |- DMA 1 -|
// CHECK-LABEL: @AssignSplitCandidateWhenAnotherPortIsPartiallyIdle
func.func @AssignSplitCandidateWhenAnotherPortIsPartiallyIdle(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %1 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3, 4, 5] <163840> -> !DistributedType
    %2 = VPURT.DeclareBuffer <CMX_NN> [1] <901888> -> memref<1x16x2x8xf16, [@CMX_NN, 1]>
    %3 = VPURT.DeclareBuffer <DDR> <96288> -> memref<1x16x2x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>
    %cst = const.Declare memref<1x1x1x368768xf16> = dense<1.0> : memref<1x1x1x368768xf16>

    VPURT.Task {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst : memref<1x1x1x368768xf16>) outputs(%1 : !DistributedType) -> !DistributedType
    }
    VPURT.Task {
      %5 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%2 : memref<1x16x2x8xf16, [@CMX_NN, 1]>) outputs(%3 : memref<1x16x2x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>) -> memref<1x16x2x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>
    }

    // CHECK:       [[NNDMA:%.+]] = VPUIP.NNDMA <{port = 0 : i64, split_candidate = true}>

    return %arg0 : !DummyT
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DummyT = memref<1x3x224x224xf16, @DDR>

// Port 0: |-------------- DMA 0 --------------|
// Port 1: |-------------- DMA 1 --------------|
// CHECK-LABEL: @NotAssignSplitCandidateWithCycleCostOverlap
func.func @NotAssignSplitCandidateWithCycleCostOverlap(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <901888> -> memref<1x1x1x368768xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [1] <901888> -> memref<1x1x1x368768xf16, [@CMX_NN, 1]>
    %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @DDR>
    %3 = VPURT.DeclareBuffer <DDR> <800000> -> memref<1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @DDR>

    VPURT.Task {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x1x1x368768xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @DDR>) -> memref<1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @DDR>
    }
    VPURT.Task {
      %5 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%1 : memref<1x1x1x368768xf16, [@CMX_NN, 1]>) outputs(%3 : memref<1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @DDR>) -> memref<1x1x1x368768xf16, {order = #NCHW, strides = [368768, 368768, 368768, 1]}, @DDR>
    }

    // CHECK:       [[NNDMA:%.+]] = VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-NOT:           split_candidate = true

    return %arg0 : !DummyT
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @AssignSplitCandidateToCMX2DDRDMA
func.func @AssignSplitCandidateToCMX2DDRDMA(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <425984> {swizzlingKey = 5 : i64} -> memref<32x1280x3x3xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <DDR> <5242880> {swizzlingKey = 5 : i64} -> memref<32x1280x3x3xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}>
          inputs(%2 : memref<32x1280x3x3xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>)
          outputs(%3 : memref<32x1280x3x3xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>) -> memref<32x1280x3x3xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
    }

    // CHECK:       [[NNDMA:%.+]] = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64, split_candidate = true}>

    return %arg0: !DummyT
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @NotAssignSplitCandidateForCompressCandidate
func.func @NotAssignSplitCandidateForCompressCandidate(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x512x7x28xf16, {allocSize = 203872 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC}, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> [2] <901888> -> memref<1x512x7x28xf16, {order = #NHWC}, [@CMX_NN, 2]>

    VPURT.Task {
      %2 = VPUIP.NNDMA <{compress_candidate, port = 0 : i64, spillId = 0 : i64}>
          inputs(%0 : memref<1x512x7x28xf16, {allocSize = 203872 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC}, @DDR>)
          outputs(%1 : memref<1x512x7x28xf16, {order = #NHWC}, [@CMX_NN, 2]>) -> memref<1x512x7x28xf16, {order = #NHWC}, [@CMX_NN, 2]>
    }

    // CHECK:       [[NNDMA:%.+]] = VPUIP.NNDMA <{compress_candidate, port = 0 : i64, spillId = 0 : i64}>
    // CHECK-NOT:           split_candidate = true

    return %arg0 : !DummyT
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @AvoidTrivialSplitCandidate
func.func @AvoidTrivialSplitCandidate(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, {order = #NHWC}, @DDR>
    %1 = VPURT.DeclareBuffer <CMX_NN> [2] <901888> -> memref<1x1x1x1000xf16, {order = #NHWC}, [@CMX_NN, 2]>

    VPURT.Task {
      %2 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}>
          inputs(%0 : memref<1x1x1x1000xf16, {order = #NHWC}, @DDR>)
          outputs(%1 : memref<1x1x1x1000xf16, {order = #NHWC}, [@CMX_NN, 2]>) -> memref<1x1x1x1000xf16, {order = #NHWC}, [@CMX_NN, 2]>
    }

    // CHECK:       [[NNDMA:%.+]] = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}>
    // CHECK-NOT:           split_candidate = true

    return %arg0 : !DummyT
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DummyT = memref<1x3x224x224xf16, @DDR>

// CHECK-LABEL: @AssignSplitCandidateToNon4DDMA
func.func @AssignSplitCandidateToNon4DDMA(%arg0: !DummyT) -> !DummyT {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <425984> -> memref<1024x683xf16, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <DDR> <5242880> -> memref<1024x683xf16, {order = #NC, strides = [2048, 1]}, @DDR>

    %4 = VPURT.DeclareBuffer <CMX_NN> [1] <901888> -> memref<1x16x2x8xf16, [@CMX_NN, 1]>
    %5 = VPURT.DeclareBuffer <DDR> <96288> -> memref<1x16x2x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %6 = VPUIP.NNDMA <{port = 0 : i64}>
          inputs(%2 : memref<1024x683xf16, [@CMX_NN, 0]>)
          outputs(%3 : memref<1024x683xf16, {order = #NC, strides = [2048, 1]}, @DDR>) -> memref<1024x683xf16, {order = #NC, strides = [2048, 1]}, @DDR>
    }

    VPURT.Task {
      %7 = VPUIP.NNDMA <{port = 1 : i64}>
          inputs(%4 : memref<1x16x2x8xf16, [@CMX_NN, 1]>)
          outputs(%5 : memref<1x16x2x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>) -> memref<1x16x2x8xf16, {order = #NCHW, strides = [81920, 64, 8, 1]}, @DDR>
    }

    // CHECK:       [[NNDMA:%.+]] = VPUIP.NNDMA <{port = 0 : i64, split_candidate = true}>

    return %arg0: !DummyT
}
