//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-view-ops-to-declarations %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK: func.func @Reshape([[ARG0:%.+]]: memref<1x512xf16>, [[ARG1:%.+]]: memref<1x512xf16>)
func.func @Reshape(%arg0: memref<1x512xf16>, %arg1: memref<1x512xf16>) -> memref<1x512xf16> {
    %in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x512xf16>
    %out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x512xf16>

    %0 = VPUIP.GenericReshape inputs(%in : memref<1x512xf16>) -> memref<1x512x1x1xf16>
    %1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x512x1x1xf16, @DDR>
    %2 = VPUIP.NNDMA inputs(%0 : memref<1x512x1x1xf16>) outputs(%1 : memref<1x512x1x1xf16, @DDR>) -> memref<1x512x1x1xf16, @DDR>
    %3 = VPUIP.GenericReshape inputs(%2 : memref<1x512x1x1xf16, @DDR>) -> memref<1x512xf16, @DDR>
    %4 = VPUIP.NNDMA inputs(%3 : memref<1x512xf16, @DDR>) outputs(%out : memref<1x512xf16>) -> memref<1x512xf16>
    return %arg1 : memref<1x512xf16>


    // CHECK-DAG:       [[OUT:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x512xf16>
    // CHECK-DAG:       [[IN:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x512x1x1xf16>

    // CHECK-DAG:       [[VAR1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x512x1x1xf16, @DDR>

    // CHECK:       [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[IN]] : memref<1x512x1x1xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<1x512x1x1xf16, @DDR>)

    // CHECK:       [[VAR3:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x512xf16, @DDR>

    // CHECK:       [[VAR4:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR3]] : memref<1x512xf16, @DDR>)
    // CHECK-SAME:      outputs([[OUT]] : memref<1x512xf16>) -> memref<1x512xf16>

    // CHECK: return [[ARG1]] : memref<1x512xf16>
}

// -----

// CHECK: func.func @SubView([[ARG0:%.+]]: memref<4x4xf16>, [[ARG1:%.+]]: memref<4x4xf16>)
func.func @SubView(%arg0: memref<4x4xf16>, %arg1: memref<4x4xf16>) -> memref<4x4xf16> {
    %in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<4x4xf16>
    %out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<4x4xf16>

    %0 = VPUIP.SubView %in [0, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %1 = VPUIP.SubView %out [0, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %2 = VPUIP.NNDMA inputs(%0 : memref<2x4xf16>) outputs(%1 : memref<2x4xf16>) -> memref<2x4xf16>

    %3 = VPUIP.SubView %in [2, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %4 = VPUIP.SubView %out [2, 0][2, 4] : memref<4x4xf16> to memref<2x4xf16>
    %5 = VPUIP.NNDMA inputs(%3 : memref<2x4xf16>) outputs(%4 : memref<2x4xf16>) -> memref<2x4xf16>

    return %arg1 : memref<4x4xf16>

    // CHECK:       [[VAR0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<2x4xf16>
    // CHECK:       [[VAR1:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<2x4xf16>
    // CHECK:       [[VAR2:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<2x4xf16>)
    // CHECK-SAME:      outputs([[VAR1]] : memref<2x4xf16>) -> memref<2x4xf16>

    // CHECK:       [[VAR3:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <16> -> memref<2x4xf16>
    // CHECK:       [[VAR4:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <16> -> memref<2x4xf16>
    // CHECK:       [[VAR5:%.*]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR3]] : memref<2x4xf16>)
    // CHECK-SAME:      outputs([[VAR4]] : memref<2x4xf16>) -> memref<2x4xf16>

    // CHECK:       return [[ARG1]] : memref<4x4xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @PermuteCast([[ARG0:%.+]]: memref<1x12x16x16xf16, #NHWC>, [[ARG1:%.+]]: memref<1x16x16x12xf16>) -> memref<1x16x16x12xf16> {
func.func @PermuteCast(%arg0: memref<1x12x16x16xf16, #NHWC>, %arg1: memref<1x16x16x12xf16>) -> memref<1x16x16x12xf16> {
    %in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x12x16x16xf16, #NHWC>
    %out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x16x12xf16>

    %0 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW}
        inputs(%in : memref<1x12x16x16xf16, #NHWC>)
        -> memref<1x16x16x12xf16>

    %1 = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x16x16x12xf16, @DDR>
    %2 = VPUIP.NNDMA
        inputs(%0 : memref<1x16x16x12xf16>)
        outputs(%1 : memref<1x16x16x12xf16, @DDR>) -> memref<1x16x16x12xf16, @DDR>
    %3 = VPUIP.NNDMA
        inputs(%2 : memref<1x16x16x12xf16, @DDR>)
        outputs(%out : memref<1x16x16x12xf16>) -> memref<1x16x16x12xf16>
    return %arg1 : memref<1x16x16x12xf16>

    //CHECK-DAG:    [[OUT:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x16x12xf16>
    //CHECK-DAG:    [[IN:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x12xf16>
    //CHECK-DAG:    [[VAR1:%.*]] = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x16x16x12xf16, @DDR>
    //CHECK:        [[VAR2:%.*]] = VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN]] : memref<1x16x16x12xf16>)
    //CHECK-SAME:       outputs([[VAR1]] : memref<1x16x16x12xf16, @DDR>) -> memref<1x16x16x12xf16, @DDR>
    //CHECK:        [[VAR3:%.*]] = VPUIP.NNDMA
    //CHECK-SAME:       inputs([[VAR2]] : memref<1x16x16x12xf16, @DDR>)
    //CHECK-SAME:       outputs([[OUT]] : memref<1x16x16x12xf16>) -> memref<1x16x16x12xf16>
    //CHECK:        return [[ARG1]] : memref<1x16x16x12xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @DistributedCast
func.func @DistributedCast(%arg0: memref<1x128x16x16xf16, #NHWC>) -> memref<1x128x16x16xf16, #NHWC> {
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedBuffer
    %1 = VPUIP.DistributedCast inputs(%0 : !InputDistributedBuffer) -> !OutputDistributedBuffer
    return %arg0 : memref<1x128x16x16xf16, #NHWC>

    // CHECK:       VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:        {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK-NOT:   VPUIP.DistributedCast
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>


// CHECK-LABEL: @NonDistributedCast
func.func @NonDistributedCast() -> memref<1x128x16x16xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = VPURT.DeclareBuffer <CMX_NN> <1000> -> !InputDistributedBuffer
    %1 = VPUIP.NonDistributedCastOp inputs(%0 : !InputDistributedBuffer) -> memref<1x128x16x16xf16, #NHWC, [@CMX_NN, 0]>
    return %1 : memref<1x128x16x16xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       VPURT.DeclareBuffer <CMX_NN> <1000> -> !VPUIP.DistributedBuffer<1x128x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK:       VPURT.DeclareBuffer <CMX_NN> [0] <1000> -> memref<1x128x16x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-NOT:   VPUIP.NonDistributedCastOp
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!InputSliceDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x8x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!InputSliceBuffer = memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>
!OutputBuffer = memref<1x64x8x16xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @VPUIPSubViewDistributed
func.func @VPUIPSubViewDistributed(%arg0: !OutputBuffer) -> !OutputBuffer {

    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedBuffer
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributedBuffer
    %2 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 8, 16] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %3 = VPUIP.NNDMA
        inputs(%2 : !InputSliceDistributedBuffer)
        outputs(%1 : !OutputDistributedBuffer) ->  !OutputDistributedBuffer

    return %arg0 : !OutputBuffer

    // CHECK-DAG:       [[BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK-DAG:       [[BUF_OUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
    // CHECK-DAG:       [[BUF_SLICE:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView

    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:       inputs([[BUF_SLICE]]
    // CHECK-SAME:       outputs([[BUF_OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!InputSliceDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x8x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!InputSliceBuffer = memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>
!OutputBuffer = memref<1x64x8x16xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @VPUIPSubViewDistributedSegmentedOnSubviewAxis
func.func @VPUIPSubViewDistributedSegmentedOnSubviewAxis(%arg0: !OutputBuffer) -> !OutputBuffer {

    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedBuffer
    %1 = VPURT.DeclareBuffer <CMX_NN> <16384> -> !OutputDistributedBuffer
    %2 = VPUIP.SubView %0 [0, 0, 8, 0] [1, 64, 8, 16] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %3 = VPUIP.NNDMA
        inputs(%2 : !InputSliceDistributedBuffer)
        outputs(%1 : !OutputDistributedBuffer) ->  !OutputDistributedBuffer

    return %arg0 : !OutputBuffer

    // CHECK:       [[BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUF_OUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <16384>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x64x8x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUF_SLICE:%.*]] = VPURT.DeclareBuffer <CMX_NN> <4096>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!InputSliceDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x8x8xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x8x8xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!InputSliceBuffer = memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>
!OutputBuffer = memref<1x64x8x16xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @VPUIPSubViewOnHAndWDistributedSegmentedOnH
func.func @VPUIPSubViewOnHAndWDistributedSegmentedOnH(%arg0: !OutputBuffer) -> !OutputBuffer {

    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedBuffer
    %1 = VPURT.DeclareBuffer <CMX_NN> <16384> -> !OutputDistributedBuffer
    %2 = VPUIP.SubView %0 [0, 0, 8, 8] [1, 64, 8, 8] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %3 = VPUIP.NNDMA
        inputs(%2 : !InputSliceDistributedBuffer)
        outputs(%1 : !OutputDistributedBuffer) ->  !OutputDistributedBuffer

    return %arg0 : !OutputBuffer

    // CHECK:       [[BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUF_OUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <16384>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x64x8x8xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUF_SLICE:%.*]] = VPURT.DeclareBuffer <CMX_NN> <5120>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x64x8x8xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x15x16xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!InputSliceDistributedBuffer = !VPUIP.DistributedBuffer<
    1x32x15x16xf16, {order = #NCHW, strides = [15360, 240, 16, 1]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x32x15x16xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64
}>

!InputSliceBuffer = memref<1x32x15x16xf16, {order = #NCHW, strides = [15360, 240, 16, 1]}, @CMX_NN>
!OutputBuffer = memref<1x32x15x16xf16, #NCHW, @CMX_NN>

// CHECK-LABEL: @VPUIPSubViewOnCDistributedSegmentedOnH
func.func @VPUIPSubViewOnCDistributedSegmentedOnH(%arg0: !OutputBuffer) -> !OutputBuffer {

    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedBuffer
    %1 = VPURT.DeclareBuffer <CMX_NN> <16384> -> !OutputDistributedBuffer
    %2 = VPUIP.SubView %0 [0, 32, 0, 0] [1, 32, 15, 16] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %3 = VPUIP.NNDMA
        inputs(%2 : !InputSliceDistributedBuffer)
        outputs(%1 : !OutputDistributedBuffer) ->  !OutputDistributedBuffer

    return %arg0 : !OutputBuffer

    // CHECK:       [[BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x64x15x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUF_OUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <16384>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x32x15x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUF_SLICE:%.*]] = VPURT.DeclareBuffer <CMX_NN> <4096>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x32x15x16xf16, {order = #NCHW, strides = [15360, 240, 16, 1]}, @CMX_NN,
    // CHECK-SAME:         {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}

    // CHECK-NOT:   VPUIP.SubView
}

// -----

#NCHW= affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x2x128x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!InputSliceDistributedBuffer = !VPUIP.DistributedBuffer<
    1x1x128x128xf16, {order = #NCHW, strides = [32768, 16384, 128, 1]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x2x128x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!InputBufferDDR = memref<1x1x128x128xf16, #NCHW, @DDR>
!InputSliceBuffer = memref<1x1x128x128xf16, {order = #NCHW, strides = [32768, 16384, 128, 1]}, @CMX_NN>
!OutputBuffer = memref<1x2x128x128xf16, #NCHW, @CMX_NN>
!OutputBufferDDR = memref<1x2x128x128xf16, #NCHW, @DDR>

// CHECK: func.func @ImplicitConcatViewOnCAndDistrubedSegmentedOnH([[IN0:%.+]]: memref<1x1x128x128xf16, @DDR>, [[IN1:%.+]]: memref<1x1x128x128xf16, @DDR>, [[OUT:%.+]]: memref<1x2x128x128xf16, @DDR>)
func.func @ImplicitConcatViewOnCAndDistrubedSegmentedOnH(%arg0: !InputBufferDDR, %arg1: !InputBufferDDR, %arg2: !OutputBufferDDR) -> !OutputBufferDDR {
    // There is no explicit Concat, but %0 is "concat" buffer
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedBuffer

    %2 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 1, 128, 128] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %3 = VPUIP.NNDMA
        inputs(%arg0 : !InputBufferDDR)
        outputs(%2 : !InputSliceDistributedBuffer) ->  !InputSliceDistributedBuffer

    %4 = VPUIP.SubView %0 [0, 1, 0, 0] [1, 1, 128, 128] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %5 = VPUIP.NNDMA
        inputs(%arg1 : !InputBufferDDR)
        outputs(%4 : !InputSliceDistributedBuffer) ->  !InputSliceDistributedBuffer
    %6 = VPUIP.NNDMA
        inputs(%0 : !InputDistributedBuffer)
        outputs(%arg2 : !OutputBufferDDR) ->  !OutputBufferDDR

    return %arg2 : !OutputBufferDDR

    // CHECK:       [[BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0>
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x2x128x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK-NOT:   VPUIP.SubView

    // CHECK:       [[BUF0:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0>
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x128x128xf16, {order = #NCHW, strides = [32768, 16384, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[IN0]]
    // CHECK-SAME:      outputs([[BUF0]]

    // CHECK-NOT:   VPUIP.SubView

    // CHECK:       [[BUF1:%.*]] = VPURT.DeclareBuffer <CMX_NN> <16384>
    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[IN1]]
    // CHECK-SAME:      outputs([[BUF1]]

    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF]]
    // CHECK-SAME:      outputs([[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!InputSliceDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!InputBufferDdr = memref<1x64x8x16xf16, #NHWC, @DDR>
!InputBuffer = memref<1x64x8x16xf16, #NHWC, @CMX_NN>
!InputSliceBuffer = memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>
!OutputBuffer = memref<1x64x16x16xf16, #NHWC, @CMX_NN>
!OutputBufferDdr = memref<1x64x16x16xf16, #NHWC, @DDR>

// CHECK: func.func @ImplicitConcatView([[ARG0:%.+]]: memref<1x64x8x16xf16, #NHWC, @DDR>, [[ARG1:%.+]]: memref<1x64x16x16xf16, #NHWC, @DDR>)
func.func @ImplicitConcatView(%arg0: !InputBufferDdr, %arg1: !OutputBufferDdr) -> !OutputBufferDdr {
    // There is no explicit Concat, but %0 is "concat" buffer
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedBuffer

    %2 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 8, 16] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %3 = VPUIP.NNDMA
        inputs(%arg0 : !InputBufferDdr)
        outputs(%2 : !InputSliceDistributedBuffer) ->  !InputSliceDistributedBuffer

    %4 = VPUIP.SubView %0 [0, 0, 8, 0] [1, 64, 8, 16] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    %5 = VPUIP.NNDMA
        inputs(%arg0 : !InputBufferDdr)
        outputs(%4 : !InputSliceDistributedBuffer) ->  !InputSliceDistributedBuffer
    %6 = VPUIP.NNDMA
        inputs(%0 : !InputDistributedBuffer)
        outputs(%arg1 : !OutputBufferDdr) ->  !OutputBufferDdr

    return %arg1 : !OutputBufferDdr

    // CHECK-DAG:       [[BUF_INPUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView
    // CHECK:       [[BUF_INPUT_SLICE1:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:       inputs([[ARG0]]
    // CHECK-SAME:       outputs([[BUF_INPUT_SLICE1]]

    // CHECK-NOT:   VPUIP.SubView
    // CHECK:       [[BUF_INPUT_SLICE2:%.*]] = VPURT.DeclareBuffer <CMX_NN> <16384> -> !VPUIP.DistributedBuffer<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:       inputs([[ARG0]]
    // CHECK-SAME:       outputs([[BUF_INPUT_SLICE2]]

    // CHECK:       VPUIP.NNDMA
    // CHECK-SAME:       inputs([[BUF_INPUT]]
    // CHECK-SAME:       outputs([[ARG1]]

    // CHECK:       return [[ARG1]] : memref<1x64x16x16xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ShapeCast
func.func @ShapeCast(%arg0: memref<64x3x7x7xf16, #NHWC>, %arg1: memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]> {

%0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>
%weights = VPUIP.NNDMA
    inputs(%arg0: memref<64x3x7x7xf16, #NHWC>)
    outputs(%0: memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>

%weights_align = VPUIP.ShapeCast{shape = [64, 16, 7, 7]}
    inputs(%weights: memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>)
     -> memref<64x16x7x7xf16, #NHWC, [@CMX_NN, 0]>
%weight_table = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>

%in = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>

%1 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        kernel_size = [7, 7],
        kernel_strides = [2, 2],
        task_type = #VPUIP.nce_task_type<CONV>
    }
    input(%in : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>)
    weights(%weights_align : memref<64x16x7x7xf16, #NHWC, [@CMX_NN, 0]>)
    weight_table(%weight_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    parent_input(%in : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>)
    parent_output(%arg1 : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>)
    outputs(%arg1 : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>
    variants :
    {
        DPUTask {outEnd = [55, 55, 63], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    }
    PPE :  {
    }

return %1 : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>

//CHECK:        [[VAR0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>
//CHECK:        [[VAR1:%.*]] = VPUIP.NNDMA inputs(%arg0 : memref<64x3x7x7xf16, #NHWC>) outputs([[VAR0]] : memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<64x3x7x7xf16, #NHWC, [@CMX_NN, 0]>
//CHECK:        [[VAR2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x16x7x7xf16, #NHWC, [@CMX_NN, 0]>
//CHECK:        [[VAR3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
//CHECK:        [[VAR4:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>
//CHECK:        [[VAR5:%.*]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}
//CHECK-SAME:           input([[VAR4]] : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>)
//CHECK-SAME:           weights([[VAR2]] : memref<64x16x7x7xf16, #NHWC, [@CMX_NN, 0]>)
//CHECK-SAME:           weight_table([[VAR3]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
//CHECK-SAME:           parent_input([[VAR4]] : memref<1x16x112x112xf16, #NHWC, [@CMX_NN, 0]>)
//CHECK-SAME:           parent_output(%arg1 : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>)
//CHECK-SAME:           outputs(%arg1 : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>)
//CHECK-SAME:           -> memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>
//CHECK-SAME:           variants :  {
//CHECK:       DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [55, 55, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
//CHECK:       return [[VAR5]] : memref<1x64x56x56xf16, #NHWC, [@CMX_NN, 0]>
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}
>
!OutType = memref<1x1x1x128xf16, [@CMX_NN, 2]>

// CHECK-LABEL: @ExtractFlatSlice
func.func @ExtractFlatSlice() -> !OutType {
    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedBuffer
    %1 = VPUIP.ExtractFlatSlice {offset = 19 : i64} inputs(%0 : !InputDistributedBuffer) -> memref<1x1x1x128xf16, [@CMX_NN, 2]>
    return %1 : !OutType

    // CHECK:       [[NEW_SOURCE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <768> -> memref<1x1x1x128xf16, [@CMX_NN, 2]>
    // CHECK:       return [[NEW_SOURCE]]
    // CHECK-NOT:   VPUIP.ExtractFlatSlice
}

// -----

// CHECK: func.func @ReinterpretCast([[ARG0:%.+]]: memref<8xi8, @DDR>, [[ARG1:%.+]]: memref<8xi8, @DDR>)
// CHECK-SAME: -> memref<8xi8, @DDR>
func.func @ReinterpretCast(%arg0: memref<8xi8, @DDR>, %arg1: memref<8xi8, @DDR>)
        -> memref<8xi8, @DDR> {
    %in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<8xi8, @DDR>
    %out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<8xi8, @DDR>

    %cast_in = Core.ReinterpretCast(%in) : memref<8xi8, @DDR> -> memref<4x1x1x1xf16, @DDR>
    %tmp = VPURT.DeclareBuffer <DDR> <0> -> memref<4x1x1x1xf16, @DDR>
    %copy = VPUIP.NNDMA {port = 0 : i64}
        inputs(%cast_in : memref<4x1x1x1xf16, @DDR>) outputs(%tmp : memref<4x1x1x1xf16, @DDR>)
        -> memref<4x1x1x1xf16, @DDR>

    %cast_out = Core.ReinterpretCast(%copy) : memref<4x1x1x1xf16, @DDR> -> memref<8xi8, @DDR>
    %res = VPUIP.NNDMA {port = 0 : i64}
        inputs(%cast_out : memref<8xi8, @DDR>) outputs(%out : memref<8xi8, @DDR>)
        -> memref<8xi8, @DDR>

    return %arg1 : memref<8xi8, @DDR>

    // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<4x1x1x1xf16, @DDR>
    // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<8xi8, @DDR>

    // CHECK: [[TMP:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<4x1x1x1xf16, @DDR>
    // CHECK: [[COPY:%.+]] = VPUIP.NNDMA
    // CHECK-SAME: inputs([[IN]]
    // CHECK-SAME: outputs([[TMP]]

    // CHECK: [[CAST_BUF:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<8xi8, @DDR>
    // CHECK: [[RES:%.+]] = VPUIP.NNDMA
    // CHECK-SAME: inputs([[CAST_BUF]]
    // CHECK-SAME: outputs([[OUT]]

    // CHECK: return [[ARG1]]
}
