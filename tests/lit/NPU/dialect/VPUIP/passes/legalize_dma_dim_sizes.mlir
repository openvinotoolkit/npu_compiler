//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --legalize-nndma-dim-sizes %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SmallDDR = memref<1x1024x512x1xf16, {order = #NCHW}, @DDR>

// The compact payload (1 MiB) fits into the 2 MiB per-task preemption budget, so the pass does not split
// the DMA.
func.func @SmallContigDMAFitsBudget(%input: !SmallDDR, %output: !SmallDDR) -> !SmallDDR {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <DDR> <0> -> !SmallDDR
    %1 = VPURT.DeclareBuffer <DDR> <2000000> -> !SmallDDR

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !SmallDDR) outputs(%1 : !SmallDDR) -> !SmallDDR
    }

    return %1 : !SmallDDR

    // CHECK:    [[INPUT:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1024x512x1xf16, {order = #NCHW}, @DDR>
    // CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <DDR> <2000000> -> memref<1x1024x512x1xf16, {order = #NCHW}, @DDR>

    // CHECK:      VPURT.Task waits({{%.+}} : !VPURT.Barrier) updates({{%.+}} : !VPURT.Barrier)
    // CHECK:        VPUIP.NNDMA
    // CHECK-SAME:     inputs([[INPUT]] : memref<1x1024x512x1xf16, {order = #NCHW}, @DDR>)
    // CHECK-SAME:     outputs([[OUTPUT]] : memref<1x1024x512x1xf16, {order = #NCHW}, @DDR>)
    // CHECK-NOT:  VPURT.Task
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InCMX = memref<5000x1x1x2xf16, {order = #NCHW}, [@CMX_NN, 0]>
!OutDDR = memref<5000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>

// CMX -> DDR path: tiling is computed on the (strided) output side with scaleBudget=true.
func.func @CmxToDdrStridedOutput(%input: !InCMX, %output: !OutDDR) -> !OutDDR {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !InCMX
    %1 = VPURT.DeclareBuffer <DDR> <1000000> -> !OutDDR

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !InCMX) outputs(%1 : !OutDDR) -> !OutDDR
    }

    return %1 : !OutDDR

    // CHECK:    [[IN_TILE0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2500x1x1x2xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_TILE1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <10000> -> memref<2500x1x1x2xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUT_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <1000000> -> memref<2500x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>
    // CHECK:    [[OUT_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <1080000> -> memref<2500x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>

    // CHECK:      VPURT.Task waits({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME:     inputs([[IN_TILE0]] : memref<2500x1x1x2xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:     outputs([[OUT_TILE0]] : memref<2500x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>)

    // CHECK:      VPURT.Task updates({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME:     inputs([[IN_TILE1]] : memref<2500x1x1x2xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:     outputs([[OUT_TILE1]] : memref<2500x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InDDR = memref<6000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>
!OutCMX = memref<6000x1x1x2xf16, {order = #NCHW}, [@CMX_NN, 0]>

// DDR -> CMX path: tiling is computed on the (strided) input side with scaleBudget=true.
func.func @DdrStridedInputToCmx(%input: !InDDR, %output: !OutCMX) -> !OutCMX {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <DDR> <0> -> !InDDR
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !OutCMX

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !InDDR) outputs(%1 : !OutCMX) -> !OutCMX
    }

    return %1 : !OutCMX

    // CHECK:    [[IN_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<3000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>
    // CHECK:    [[IN_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <96000> -> memref<3000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>
    // CHECK:    [[OUT_TILE0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<3000x1x1x2xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUT_TILE1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <12000> -> memref<3000x1x1x2xf16, [@CMX_NN, 0]>

    // CHECK:      VPURT.Task waits({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME:     inputs([[IN_TILE0]] : memref<3000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>)
    // CHECK-SAME:     outputs([[OUT_TILE0]] : memref<3000x1x1x2xf16, [@CMX_NN, 0]>)

    // CHECK:      VPURT.Task updates({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME:     inputs([[IN_TILE1]] : memref<3000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>)
    // CHECK-SAME:     outputs([[OUT_TILE1]] : memref<3000x1x1x2xf16, [@CMX_NN, 0]>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InCMXStrided = memref<100000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, [@CMX_NN, 0]>
!OutCMX = memref<100000x1x1x2xf16, {order = #NCHW}, [@CMX_NN, 0]>

// CMX -> CMX path: scaleBudget=false, so the effective per-tile budget is the full 2 MiB regardless of the
// contiguous chunk size.
func.func @CmxToCmxNotScaled(%input: !InCMXStrided, %output: !OutCMX) -> !OutCMX {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !InCMXStrided
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <4000000> -> !OutCMX

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !InCMXStrided) outputs(%1 : !OutCMX) -> !OutCMX
    }

    return %1 : !OutCMX

    // CHECK:    [[INPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<100000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4000000> -> memref<100000x1x1x2xf16, {order = #NCHW}, [@CMX_NN, 0]>

    // CHECK:      VPURT.Task waits({{%.+}} : !VPURT.Barrier) updates({{%.+}} : !VPURT.Barrier)
    // CHECK:        VPUIP.NNDMA
    // CHECK-SAME:     inputs([[INPUT]] : memref<100000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:     outputs([[OUTPUT]] : memref<100000x1x1x2xf16, {order = #NCHW}, [@CMX_NN, 0]>)
    // CHECK-NOT:  VPURT.Task
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InDDRStrided = memref<9000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>
!OutDDR = memref<9000x1x1x2xf16, {order = #NCHW}, @DDR>

// DDR -> DDR strided path: scaleBudget=true, contigBytes=4 B -> scaled budget = 16 384 B, elPerPart = 4096.
// ceil(9000/4096) = 3 tiles; planesPerTile = divUp(9000, 3) = 3000, so three balanced tiles of 3000 planes.
// Only the first tile keeps the original wait barrier and only the last keeps the update barrier; the middle
// tile runs back-to-back on the same DMA port without extra synchronization.
func.func @DdrStridedToDdrScaled(%input: !InDDRStrided, %output: !OutDDR) -> !OutDDR {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <DDR> <0> -> !InDDRStrided
    %1 = VPURT.DeclareBuffer <DDR> <1000000> -> !OutDDR

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !InDDRStrided) outputs(%1 : !OutDDR) -> !OutDDR
    }

    return %1 : !OutDDR

    // CHECK:    [[IN_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<3000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>
    // CHECK:    [[IN_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <96000> -> memref<3000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>
    // CHECK:    [[IN_TILE2:%.+]] = VPURT.DeclareBuffer <DDR> <192000> -> memref<3000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>
    // CHECK:    [[OUT_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <1000000> -> memref<3000x1x1x2xf16, @DDR>
    // CHECK:    [[OUT_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <1012000> -> memref<3000x1x1x2xf16, @DDR>
    // CHECK:    [[OUT_TILE2:%.+]] = VPURT.DeclareBuffer <DDR> <1024000> -> memref<3000x1x1x2xf16, @DDR>

    // CHECK:      VPURT.Task waits({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME:     inputs([[IN_TILE0]] : memref<3000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>)
    // CHECK-SAME:     outputs([[OUT_TILE0]] : memref<3000x1x1x2xf16, @DDR>)

    // CHECK:      VPURT.Task {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME:     inputs([[IN_TILE1]] : memref<3000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>)
    // CHECK-SAME:     outputs([[OUT_TILE1]] : memref<3000x1x1x2xf16, @DDR>)

    // CHECK:      VPURT.Task updates({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME:     inputs([[IN_TILE2]] : memref<3000x1x1x2xf16, {order = #NCHW, strides = [16, 8, 4, 1]}, @DDR>)
    // CHECK-SAME:     outputs([[OUT_TILE2]] : memref<3000x1x1x2xf16, @DDR>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDDR = memref<2x1050000x1x1xf16, {order = #NCHW}, @DDR>
!OutputDDR = memref<2x1050000x1x1xf16, {order = #NCHW}, @DDR>

// The outermost dimension is only 2, so the preemption split must skip it and use the next dimension, which is
// large enough to produce three tiles and satisfy the 2 MiB budget.
func.func @SplitOnNextDimWhenOuterDimTooSmall(%input: !InputDDR, %output: !OutputDDR) -> !OutputDDR {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <DDR> <0> -> !InputDDR
    %1 = VPURT.DeclareBuffer <DDR> <1000000> -> !OutputDDR

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !InputDDR) outputs(%1 : !OutputDDR) -> !OutputDDR
    }

    return %1: !OutputDDR

    // CHECK:      VPURT.Task waits([[BAR0:%.+]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs({{%[0-9]+}} : memref<2x350000x1x1xf16, {order = #NCHW, strides = [1050000, 1, 1, 1]}, @DDR>)
    // CHECK-SAME: outputs({{%[0-9]+}} : memref<2x350000x1x1xf16, {order = #NCHW, strides = [1050000, 1, 1, 1]}, @DDR>)

    // CHECK:      VPURT.Task {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs({{%[0-9]+}} : memref<2x350000x1x1xf16, {order = #NCHW, strides = [1050000, 1, 1, 1]}, @DDR>)
    // CHECK-SAME: outputs({{%[0-9]+}} : memref<2x350000x1x1xf16, {order = #NCHW, strides = [1050000, 1, 1, 1]}, @DDR>)

    // CHECK:      VPURT.Task updates([[BAR1:%.+]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs({{%[0-9]+}} : memref<2x350000x1x1xf16, {order = #NCHW, strides = [1050000, 1, 1, 1]}, @DDR>)
    // CHECK-SAME: outputs({{%[0-9]+}} : memref<2x350000x1x1xf16, {order = #NCHW, strides = [1050000, 1, 1, 1]}, @DDR>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDDR = memref<1x1024x800x2xf16, {order = #NHWC}, @DDR>
!OutputCMX_Dup = !VPUIP.DistributedBuffer<1x1024x800x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 4 : i64,
    compute_shapes = [[1, 1024, 800, 2], [1, 1024, 800, 2], [1, 1024, 800, 2], [1, 1024, 800, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1024, 800, 2], [1, 1024, 800, 2], [1, 1024, 800, 2], [1, 1024, 800, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// DUPLICATED broadcast surviving UnrollDistributedOps: getCompactAllocSize returns the full 3.125 MiB
// payload (>2 MiB budget), so the pass splits along H (the first mem-order dim with size >= numTiles).
func.func @DuplicatedDistributedSplits(%input: !InputDDR) -> !OutputCMX_Dup {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer <DDR> <0> -> !InputDDR
    %1 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <0> -> !OutputCMX_Dup
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !InputDDR) outputs(%1 : !OutputCMX_Dup) -> !OutputCMX_Dup
    }
    return %1 : !OutputCMX_Dup

    // CHECK:    [[IN_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1024x400x2xf16, {order = #NHWC, strides = [1638400, 1, 2048, 1024]}, @DDR>
    // CHECK:    [[IN_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <1638400> -> memref<1x1024x400x2xf16, {order = #NHWC, strides = [1638400, 1, 2048, 1024]}, @DDR>
    // CHECK:    [[OUT_TILE0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <0> -> !VPUIP.DistributedBuffer<1x1024x400x2xf16, {order = #NHWC, strides = [1638400, 1, 2048, 1024]}, @CMX_NN,
    // CHECK-SAME:    mode = "DUPLICATED", num_clusters = 4 : i64
    // CHECK-SAME:    compute_shapes = {{\[}}[1, 1024, 400, 2], [1, 1024, 400, 2], [1, 1024, 400, 2], [1, 1024, 400, 2]]
    // CHECK:    [[OUT_TILE1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <1638400> -> !VPUIP.DistributedBuffer<1x1024x400x2xf16, {order = #NHWC, strides = [1638400, 1, 2048, 1024]}, @CMX_NN,

    // CHECK:      VPURT.Task waits({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE0]] : memref<1x1024x400x2xf16, {order = #NHWC, strides = [1638400, 1, 2048, 1024]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE0]] : !VPUIP.DistributedBuffer<1x1024x400x2xf16, {{.*}})

    // CHECK:      VPURT.Task updates({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE1]] : memref<1x1024x400x2xf16, {order = #NHWC, strides = [1638400, 1, 2048, 1024]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE1]] : !VPUIP.DistributedBuffer<1x1024x400x2xf16, {{.*}})
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InStridedC = memref<1x2x1050x1000xf16, {order = #NCHW, strides = [2100002, 1050001, 1000, 1]}, @DDR>
!OutCompact = memref<1x2x1050x1000xf16, {order = #NCHW}, @DDR>

// Strided per-C DMA: C carries the plane iteration (with 1-element slack), so H*W is squashed as the inner
// contiguous chunk. That chunk (1050*1000*2 = 2 100 000 B) already exceeds the scaled 2 MiB budget, so the
// split must land *inside* the contiguous region. H = 1050 is large enough to carry the required 3 parts, so
// the pass prefers H (outermost dim inside the contig chunk) and slices it into ceil(1050/3) = 350 planes per
// tile.
func.func @SplitOnHInsideContigChunk(%input: !InStridedC, %output: !OutCompact) -> !OutCompact {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <DDR> <0> -> !InStridedC
    %1 = VPURT.DeclareBuffer <DDR> <8000000> -> !OutCompact

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !InStridedC) outputs(%1 : !OutCompact) -> !OutCompact
    }

    return %1 : !OutCompact

    // CHECK:    [[IN_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100002, 1050001, 1000, 1]}, @DDR>
    // CHECK:    [[IN_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <700000> -> memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100002, 1050001, 1000, 1]}, @DDR>
    // CHECK:    [[IN_TILE2:%.+]] = VPURT.DeclareBuffer <DDR> <1400000> -> memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100002, 1050001, 1000, 1]}, @DDR>
    // CHECK:    [[OUT_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <8000000> -> memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100000, 1050000, 1000, 1]}, @DDR>
    // CHECK:    [[OUT_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <8700000> -> memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100000, 1050000, 1000, 1]}, @DDR>
    // CHECK:    [[OUT_TILE2:%.+]] = VPURT.DeclareBuffer <DDR> <9400000> -> memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100000, 1050000, 1000, 1]}, @DDR>

    // CHECK:      VPURT.Task waits({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE0]] : memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100002, 1050001, 1000, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE0]] : memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100000, 1050000, 1000, 1]}, @DDR>)

    // CHECK:      VPURT.Task {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE1]] : memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100002, 1050001, 1000, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE1]] : memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100000, 1050000, 1000, 1]}, @DDR>)

    // CHECK:      VPURT.Task updates({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE2]] : memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100002, 1050001, 1000, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE2]] : memref<1x2x350x1000xf16, {order = #NCHW, strides = [2100000, 1050000, 1000, 1]}, @DDR>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InStridedCSmallH = memref<1x2x2x600000xf16, {order = #NCHW, strides = [2400002, 1200001, 600000, 1]}, @DDR>
!OutCompactSmallH = memref<1x2x2x600000xf16, {order = #NCHW}, @DDR>

// Same strided-per-C setup as the previous test, but H is only 2 -- too small to carry the required 3 parts.
// The pass skips H and falls through to W (the next dim inside the contig chunk), which is large enough:
// W = 600 000 -> ceil(600000/3) = 200 000 elements per tile.
func.func @SplitOnWWhenHTooSmall(%input: !InStridedCSmallH, %output: !OutCompactSmallH) -> !OutCompactSmallH {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <DDR> <0> -> !InStridedCSmallH
    %1 = VPURT.DeclareBuffer <DDR> <9000000> -> !OutCompactSmallH

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !InStridedCSmallH) outputs(%1 : !OutCompactSmallH) -> !OutCompactSmallH
    }

    return %1 : !OutCompactSmallH

    // CHECK:    [[IN_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400002, 1200001, 600000, 1]}, @DDR>
    // CHECK:    [[IN_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <400000> -> memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400002, 1200001, 600000, 1]}, @DDR>
    // CHECK:    [[IN_TILE2:%.+]] = VPURT.DeclareBuffer <DDR> <800000> -> memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400002, 1200001, 600000, 1]}, @DDR>
    // CHECK:    [[OUT_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <9000000> -> memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400000, 1200000, 600000, 1]}, @DDR>
    // CHECK:    [[OUT_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <9400000> -> memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400000, 1200000, 600000, 1]}, @DDR>
    // CHECK:    [[OUT_TILE2:%.+]] = VPURT.DeclareBuffer <DDR> <9800000> -> memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400000, 1200000, 600000, 1]}, @DDR>

    // CHECK:      VPURT.Task waits({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE0]] : memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400002, 1200001, 600000, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE0]] : memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400000, 1200000, 600000, 1]}, @DDR>)

    // CHECK:      VPURT.Task {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE1]] : memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400002, 1200001, 600000, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE1]] : memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400000, 1200000, 600000, 1]}, @DDR>)

    // CHECK:      VPURT.Task updates({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE2]] : memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400002, 1200001, 600000, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE2]] : memref<1x2x2x200000xf16, {order = #NCHW, strides = [2400000, 1200000, 600000, 1]}, @DDR>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InStridedI4C = memref<1x3x1x4194306xi4, {order = #NCHW, strides = [12582924, 4194308, 4194306, 1]}, @DDR>
!OutCompactI4C = memref<1x3x1x4194306xi4, {order = #NCHW}, @DDR>

// Strided i4 DMA that jointly exercises the analyzeContigChunk bit-based walk (previously threw on i4
// because getElemTypeSize().to<Byte>() cannot represent a 4-bit element) AND the planesPerTile sub-byte
// alignment fix. Contig chunk = H*W*4/8 = 1*4194306*4/8 = 2 097 153 B, just above the 2 MiB scaledBudget,
// so the split lands inside the contig chunk on the innermost dim W (stride = 4 bits, sub-byte). Naive
// planesPerTile = divUp(4194306, 4) = 1 048 577 is odd and would place the per-tile byte offset on a
// half-byte boundary (1 048 577 * 4 bits = 524 288.5 B). The fix rounds planesPerTile up to 1 048 578,
// giving a byte-aligned per-tile step of 524 289 B; the trailing tile absorbs the remaining
// 4 194 306 - 3 * 1 048 578 = 1 048 572 elements. C-slack is 2 elements (8 bits) so the C stride stays
// byte-aligned as reduceDimsForDma requires.
func.func @StridedI4SplitInsideContigNeedsAlign(%input: !InStridedI4C, %output: !OutCompactI4C) -> !OutCompactI4C {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <DDR> <0> -> !InStridedI4C
    %1 = VPURT.DeclareBuffer <DDR> <10000000> -> !OutCompactI4C

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !InStridedI4C) outputs(%1 : !OutCompactI4C) -> !OutCompactI4C
    }

    return %1 : !OutCompactI4C

    // CHECK:    [[IN_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582924, 4194308, 4194306, 1]}, @DDR>
    // CHECK:    [[IN_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <524289> -> memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582924, 4194308, 4194306, 1]}, @DDR>
    // CHECK:    [[IN_TILE2:%.+]] = VPURT.DeclareBuffer <DDR> <1048578> -> memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582924, 4194308, 4194306, 1]}, @DDR>
    // CHECK:    [[IN_TILE3:%.+]] = VPURT.DeclareBuffer <DDR> <1572867> -> memref<1x3x1x1048572xi4, {order = #NCHW, strides = [12582924, 4194308, 4194306, 1]}, @DDR>
    // CHECK:    [[OUT_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <10000000> -> memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582918, 4194306, 4194306, 1]}, @DDR>
    // CHECK:    [[OUT_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <10524289> -> memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582918, 4194306, 4194306, 1]}, @DDR>
    // CHECK:    [[OUT_TILE2:%.+]] = VPURT.DeclareBuffer <DDR> <11048578> -> memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582918, 4194306, 4194306, 1]}, @DDR>
    // CHECK:    [[OUT_TILE3:%.+]] = VPURT.DeclareBuffer <DDR> <11572867> -> memref<1x3x1x1048572xi4, {order = #NCHW, strides = [12582918, 4194306, 4194306, 1]}, @DDR>

    // CHECK:      VPURT.Task waits({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE0]] : memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582924, 4194308, 4194306, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE0]] : memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582918, 4194306, 4194306, 1]}, @DDR>)

    // CHECK:      VPURT.Task {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE1]] : memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582924, 4194308, 4194306, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE1]] : memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582918, 4194306, 4194306, 1]}, @DDR>)

    // CHECK:      VPURT.Task {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE2]] : memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582924, 4194308, 4194306, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE2]] : memref<1x3x1x1048578xi4, {order = #NCHW, strides = [12582918, 4194306, 4194306, 1]}, @DDR>)

    // CHECK:      VPURT.Task updates({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE3]] : memref<1x3x1x1048572xi4, {order = #NCHW, strides = [12582924, 4194308, 4194306, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE3]] : memref<1x3x1x1048572xi4, {order = #NCHW, strides = [12582918, 4194306, 4194306, 1]}, @DDR>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InI4Flat = memref<1x1x1x4194306xi4, {order = #NCHW}, @DDR>
!OutI4Flat = memref<1x1x1x4194306xi4, {order = #NCHW}, @DDR>

// Contiguous i4 DMA that hits the planesPerTile byte-alignment fix. The pass picks the innermost dim as
// tileDim (stride = 4 bits). Naive planesPerTile = ceil(4194306 / 2) = 2 097 153 is odd, which would produce
// a non-byte-aligned per-tile byte offset (2097153 * 4 bits = 1 048 576.5 B). The fix rounds planesPerTile up
// to 2 097 154 so the intermediate byte offset becomes exactly 1 048 577 B; the trailing (short) tile
// absorbs the remainder of 2 097 152 elements.
func.func @ContigI4NeedsPlanesAlign(%input: !InI4Flat, %output: !OutI4Flat) -> !OutI4Flat {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <DDR> <0> -> !InI4Flat
    %1 = VPURT.DeclareBuffer <DDR> <5000000> -> !OutI4Flat

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %2 = VPUIP.NNDMA inputs(%0 : !InI4Flat) outputs(%1 : !OutI4Flat) -> !OutI4Flat
    }

    return %1 : !OutI4Flat

    // CHECK:    [[IN_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x2097154xi4, {order = #NCHW, strides = [4194306, 4194306, 4194306, 1]}, @DDR>
    // CHECK:    [[IN_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <1048577> -> memref<1x1x1x2097152xi4, {order = #NCHW, strides = [4194306, 4194306, 4194306, 1]}, @DDR>
    // CHECK:    [[OUT_TILE0:%.+]] = VPURT.DeclareBuffer <DDR> <5000000> -> memref<1x1x1x2097154xi4, {order = #NCHW, strides = [4194306, 4194306, 4194306, 1]}, @DDR>
    // CHECK:    [[OUT_TILE1:%.+]] = VPURT.DeclareBuffer <DDR> <6048577> -> memref<1x1x1x2097152xi4, {order = #NCHW, strides = [4194306, 4194306, 4194306, 1]}, @DDR>

    // CHECK:      VPURT.Task waits({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE0]] : memref<1x1x1x2097154xi4, {order = #NCHW, strides = [4194306, 4194306, 4194306, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE0]] : memref<1x1x1x2097154xi4, {order = #NCHW, strides = [4194306, 4194306, 4194306, 1]}, @DDR>)

    // CHECK:      VPURT.Task updates({{%.+}} : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME: inputs([[IN_TILE1]] : memref<1x1x1x2097152xi4, {order = #NCHW, strides = [4194306, 4194306, 4194306, 1]}, @DDR>)
    // CHECK-SAME: outputs([[OUT_TILE1]] : memref<1x1x1x2097152xi4, {order = #NCHW, strides = [4194306, 4194306, 4194306, 1]}, @DDR>)
}
