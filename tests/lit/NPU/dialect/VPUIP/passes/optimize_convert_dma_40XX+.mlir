//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-convert-dma-op %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDMACopySequence
func.func @ConvertDMACopySequence(
        %arg0: memref<1x16x112x112xf32, #NHWC, @CMX>,
        %arg1: memref<1x16x112x112xf16, #NHWC, @CMX>)
        -> memref<1x16x112x112xf16, #NHWC, @CMX> {

    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX>
    %1 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x16x112x112xf32, #NHWC, @CMX>) outputs(%0 : memref<1x16x112x112xf16, #NHWC, @CMX>) -> memref<1x16x112x112xf16, #NHWC, @CMX>

    %2 = VPUIP.Copy inputs(%1 : memref<1x16x112x112xf16, #NHWC, @CMX>)
        outputs(%arg1 : memref<1x16x112x112xf16, #NHWC, @CMX>)
        -> memref<1x16x112x112xf16, #NHWC, @CMX>

    return %2 : memref<1x16x112x112xf16, #NHWC, @CMX>

    //CHECK: [[CONVERTDMA:%.*]] = VPUIP.ConvertDMA inputs(%arg0 : memref<1x16x112x112xf32, #NHWC, @CMX>)
    //CHECK-NOT: VPUIP.Copy
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!SpilledOutput_DDR = memref<1x16x112x112xf16, #NHWC, @DDR>

// CHECK-LABEL: @ConvertDMAClusterCopySequence
func.func @ConvertDMAClusterCopySequence(%arg0: memref<1x16x112x112xf32, #NHWC, @CMX_NN>) -> !SpilledOutput_DDR {

    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x16x112x112xf32, #NHWC, @CMX_NN>) outputs(%0 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

    %2 = memref.alloc() : !SpilledOutput_DDR
    %3 = VPUIP.Copy
        inputs(%1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
        outputs(%2 : !SpilledOutput_DDR)  ->  !SpilledOutput_DDR

    return %3 : !SpilledOutput_DDR

    // CHECK:   [[BUF_0:%.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    // CHECK:    [[ConvertDMA:%.*]] = VPUIP.ConvertDMA
    // CHECK-SAME:     inputs(%arg0 : memref<1x16x112x112xf32, #NHWC, @CMX_NN>)
    // CHECK-SAME:     outputs([[BUF_0]] : memref<1x16x112x112xf16, #NHWC, @DDR>) ->  memref<1x16x112x112xf16, #NHWC, @DDR>
    // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvertDMASubView
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x1x5462x128xf32, @DDR>)
func.func @ConvertDMASubView(%arg0: memref<1x1x5462x128xf32, @DDR>) -> memref<1x1x1024x128xf16, @CMX_NN> {
    %0 = memref.alloc() : memref<1x1x5462x128xf16, @DDR>
    %1 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x1x5462x128xf32, @DDR>) outputs(%0 : memref<1x1x5462x128xf16, @DDR>) -> memref<1x1x5462x128xf16, @DDR>
    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 1, 1024, 128] : memref<1x1x5462x128xf16, @DDR> to memref<1x1x1024x128xf16, {order = #NCHW, strides = [699136, 699136, 128, 1]}, @DDR>
    %3 = memref.alloc() : memref<1x1x1024x128xf16, @CMX_NN>
    %4 = VPUIP.Copy inputs(%2 : memref<1x1x1024x128xf16, {order = #NCHW, strides = [699136, 699136, 128, 1]}, @DDR>) outputs(%3 : memref<1x1x1024x128xf16, @CMX_NN>) -> memref<1x1x1024x128xf16, @CMX_NN>
    return %4 : memref<1x1x1024x128xf16, @CMX_NN>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x1x1024x128xf16, @CMX_NN>

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 1, 1024, 128]
    // CHECK-SAME: memref<1x1x5462x128xf32, @DDR> to memref<1x1x1024x128xf32, {order = #NCHW, strides = [699136, 699136, 128, 1]}, @DDR>

    // CHECK: [[CONVERT:%.+]] = VPUIP.ConvertDMA
    // CHECK:   inputs([[SUBVIEW]] : memref<1x1x1024x128xf32, {order = #NCHW, strides = [699136, 699136, 128, 1]}, @DDR>)
    // CHECK:   outputs([[OUT_BUF]] : memref<1x1x1024x128xf16, @CMX_NN>) -> memref<1x1x1024x128xf16, @CMX_NN>
    // CHECK: return [[CONVERT]] : memref<1x1x1024x128xf16, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!outDistributedType = !VPUIP.DistributedBuffer<
    32x2048x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    uniform_distributed_segments,
    compute_shapes = [[32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @ConvertDMAClusterCopySequenceWithViewLike
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x1x32x2048xf32, @DDR>)
func.func @ConvertDMAClusterCopySequenceWithViewLike(%arg0: memref<1x1x32x2048xf32, @DDR>) -> !outDistributedType {
    %alloc_0 = memref.alloc() : memref<1x1x32x2048xf16, @DDR>
    %0 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x1x32x2048xf32, @DDR>) outputs(%alloc_0 : memref<1x1x32x2048xf16, @DDR>) -> memref<1x1x32x2048xf16, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x1x32x2048xf16, @DDR>) -> memref<32x2048x1x1xf16, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%1 : memref<32x2048x1x1xf16, @DDR>) -> memref<32x2048x1x1xf16, #NHWC, @DDR>

    %3 = VPURT.AllocDistributed -> !outDistributedType
    %4 = VPUIP.Copy inputs(%2 : memref<32x2048x1x1xf16, #NHWC, @DDR>) outputs(%3 : !outDistributedType) -> !outDistributedType
    return %4 : !outDistributedType

    // CHECK: [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK: -> !VPUIP.DistributedBuffer<32x2048x1x1xf16, #NHWC, @CMX_NN

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : memref<1x1x32x2048xf32, @DDR>) -> memref<32x2048x1x1xf32, @DDR>
    // CHECK: [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : memref<32x2048x1x1xf32, @DDR>)
    // CHECK-SAME: -> memref<32x2048x1x1xf32, #NHWC, @DDR>

    // CHECK: [[CONVERT:%.+]] = VPUIP.ConvertDMA
    // CHECK: inputs([[PERMUTE]] : memref<32x2048x1x1xf32, #NHWC, @DDR>)
    // CHECK: outputs([[OUT_BUFF]] : !VPUIP.DistributedBuffer<32x2048x1x1xf16, #NHWC, @CMX_NN,
    // CHECK: {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK: -> !VPUIP.DistributedBuffer<32x2048x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1], [32, 2048, 1, 1]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: return [[CONVERT]] : !VPUIP.DistributedBuffer
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!outDistributedType = !VPUIP.DistributedBuffer<
    1x1x1x32xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 32], [1, 1, 1, 32], [1, 1, 1, 32], [1, 1, 1, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 32], [1, 1, 1, 32], [1, 1, 1, 32], [1, 1, 1, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!reshapeDistributedType = !VPUIP.DistributedBuffer<
    32x1x1x1xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    uniform_distributed_segments,
    compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!permuteDistributedType = !VPUIP.DistributedBuffer<
    32x1x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    uniform_distributed_segments,
    compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @NotFuseConvertDMAAndCopyDueToDistOutputAndViewLikeOp
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x1x1x32xf32, @DDR>)
func.func @NotFuseConvertDMAAndCopyDueToDistOutputAndViewLikeOp(%arg0: memref<1x1x1x32xf32, @DDR>) -> memref<32x1x1x1xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !outDistributedType
    %1 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x1x1x32xf32, @DDR>) outputs(%0 : !outDistributedType) -> !outDistributedType
    %2 = VPUIP.GenericReshape inputs(%1 : !outDistributedType) -> !reshapeDistributedType
    %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%2 : !reshapeDistributedType) -> !permuteDistributedType
    %alloc_0 = memref.alloc() : memref<32x1x1x1xf16, #NHWC, @DDR>
    %4 = VPUIP.Copy inputs(%3 : !permuteDistributedType) outputs(%alloc_0 : memref<32x1x1x1xf16, #NHWC, @DDR>) -> memref<32x1x1x1xf16, #NHWC, @DDR>
    return %4 : memref<32x1x1x1xf16, #NHWC, @DDR>

    // CHECK: [[ALLOC_DIST:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK: [[CONVERT_DMA:%.+]] = VPUIP.ConvertDMA
    // CHECK:  inputs([[INPUT]] : memref<1x1x1x32xf32, @DDR>)
    // CHECK:  outputs(%0 : !VPUIP.DistributedBuffer<1x1x1x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape
    // CHECK:  inputs([[CONVERT_DMA]] : !VPUIP.DistributedBuffer<1x1x1x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK:  -> !VPUIP.DistributedBuffer<32x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments

    // CHECK: [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:  inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<32x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK:  -> !VPUIP.DistributedBuffer<32x1x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments

    // CHECK: [[MEM_ALLOC:%.+]] = memref.alloc() : memref<32x1x1x1xf16, #NHWC, @DDR>
    // CHECK: [[COPY:%.+]] = VPUIP.Copy
    // CHECK:  inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<32x1x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK:  outputs([[MEM_ALLOC]] : memref<32x1x1x1xf16, #NHWC, @DDR>) -> memref<32x1x1x1xf16, #NHWC, @DDR>
    // CHECK: return [[COPY]] : memref<32x1x1x1xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!inDistributedType = !VPUIP.DistributedBuffer<
    1x1x32x2048xf32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
    memory_shapes = [[1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]
}>

!outDistributedType = !VPUIP.DistributedBuffer<
    1x1x32x2048xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
    memory_shapes = [[1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]
}>

// CHECK-LABEL: @DistConvertDMAAndCopyWithoutViewLikeOp
// CHECK-SAME: ([[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x1x32x2048xf32, #NCHW, @CMX_NN
func.func @DistConvertDMAAndCopyWithoutViewLikeOp(%arg0: !inDistributedType) -> memref<1x1x32x2048xf16, @DDR> {
    %0 = VPURT.AllocDistributed -> !outDistributedType
    %1 = VPUIP.ConvertDMA inputs(%arg0 : !inDistributedType) outputs(%0 : !outDistributedType) -> !outDistributedType

    %2 = memref.alloc() : memref<1x1x32x2048xf16, @DDR>
    %3 = VPUIP.Copy inputs(%1 : !outDistributedType) outputs(%2 : memref<1x1x32x2048xf16, @DDR>) -> memref<1x1x32x2048xf16, @DDR>
    return %3 : memref<1x1x32x2048xf16, @DDR>

    // CHECK: [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x1x32x2048xf16, @DDR>
    // CHECK: [[CONVERT_DMA:%.+]] = VPUIP.ConvertDMA
    // CHECK: inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x1x32x2048xf32, #NCHW, @CMX_NN,
    // CHECK:   {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048]]
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048], [1, 1, 8, 2048]]
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}>)
    // CHECK: outputs([[OUT_BUFF]] : memref<1x1x32x2048xf16, @DDR>) -> memref<1x1x32x2048xf16, @DDR>

    // CHECK: return [[CONVERT_DMA]] : memref<1x1x32x2048xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!inDDRDistributedType = !VPUIP.DistributedBuffer<
    1x2048x1x1xf32, #NHWC, @DDR, {
    mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>


!outDDRDistributedType = !VPUIP.DistributedBuffer<
    1x2048x1x1xf16, #NHWC, @DDR, {
    mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!outCmxDistributedType = !VPUIP.DistributedBuffer<
    1x2048x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @DistConvertDMAAndDistCopyWithoutViewLikeOp
// CHECK-SAME: ([[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x2048x1x1xf32, #NHWC, @DDR
func.func @DistConvertDMAAndDistCopyWithoutViewLikeOp(%arg0: !inDDRDistributedType) -> !outCmxDistributedType {
    %0 = VPURT.AllocDistributed -> !outDDRDistributedType
    %1 = VPUIP.ConvertDMA inputs(%arg0 : !inDDRDistributedType) outputs(%0 : !outDDRDistributedType) -> !outDDRDistributedType

    %2 = VPURT.AllocDistributed -> !outCmxDistributedType
    %3 = VPUIP.Copy inputs(%1 : !outDDRDistributedType) outputs(%2 : !outCmxDistributedType) -> !outCmxDistributedType
    return %3 : !outCmxDistributedType

    // CHECK: [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK: -> !VPUIP.DistributedBuffer<1x2048x1x1xf16, #NHWC, @CMX_NN,
    // CHECK: {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[CONVERT_DMA:%.+]] = VPUIP.ConvertDMA
    // CHECK: inputs([[INPUT]] : !VPUIP.DistributedBuffer<1x2048x1x1xf32, #NHWC, @DDR,
    // CHECK:   {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK: outputs([[OUT_BUFF]] : !VPUIP.DistributedBuffer<1x2048x1x1xf16, #NHWC, @CMX_NN,
    // CHECK:  {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]]
    // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK: -> !VPUIP.DistributedBuffer<1x2048x1x1xf16, #NHWC, @CMX_NN,
    // CHECK:  {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,

    // CHECK: return [[CONVERT_DMA]] : !VPUIP.DistributedBuffer<1x2048x1x1xf16, #NHWC, @CMX_NN,
}

// -----

// E#179283: this case is not supported.
func.func @NotOptimizeConvertDmaWithReinterpretCast(%out: memref<32xi8, @DDR>) -> memref<32xi8, @DDR> {
    %cst = const.Declare memref<1x4x4x1xf32> = dense<42.0> : tensor<4x4x1xf32>,
        [#const.AffineReshape<[[0, 1], [2], [3]], [1, 4, 4, 1]>]
    %tmp_alloc = memref.alloc() : memref<1x4x4x1xf16, @DDR>
    %cvt = VPUIP.ConvertDMA inputs(%cst : memref<1x4x4x1xf32>) outputs(%tmp_alloc : memref<1x4x4x1xf16, @DDR>)
        -> memref<1x4x4x1xf16, @DDR>
    %rcast = Core.ReinterpretCast(%cvt) : memref<1x4x4x1xf16, @DDR> -> memref<32xi8, @DDR>
    %copy = VPUIP.Copy inputs(%rcast : memref<32xi8, @DDR>) outputs(%out : memref<32xi8, @DDR>)
        -> memref<32xi8, @DDR>
    return %copy : memref<32xi8, @DDR>

    // CHECK: [[CST:%.+]] = const.Declare
    // CHECK: [[CVT_DMA:%.+]] = VPUIP.ConvertDMA inputs([[CST]]
    // CHECK: [[RCAST:%.+]] = Core.ReinterpretCast([[CVT_DMA]])
    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[RCAST]]
    // CHECK: return [[COPY]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Output_CMX = memref<1x16x112x112xf16, #NHWC, @DDR>
!Input_CMX = memref<1x16x112x112xf32, #NHWC, @CMX_NN>

// CHECK-LABEL: @ClusterConvertDMAClusterCopySequence
func.func @ClusterConvertDMAClusterCopySequence(%arg0: !Input_CMX) -> !Output_CMX {

    %0 = memref.alloc() : !Output_CMX
    %1 = VPUIP.ConvertDMA
        inputs(%arg0 : !Input_CMX)
        outputs(%0 : !Output_CMX) ->  !Output_CMX

    %2 = memref.alloc() : !Output_CMX
    %3 = VPUIP.Copy
        inputs(%1 : !Output_CMX)
        outputs(%2 : !Output_CMX)  ->  !Output_CMX

    return %3 : !Output_CMX

    // CHECK:   [[BUF_0:%.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    // CHECK:   [[ConvertDMA:%.*]] = VPUIP.ConvertDMA
    // CHECK-SAME:     inputs(%arg0 : memref<1x16x112x112xf32, #NHWC, @CMX_NN>)
    // CHECK-SAME:     outputs([[BUF_0]] : memref<1x16x112x112xf16, #NHWC, @DDR>) ->  memref<1x16x112x112xf16, #NHWC, @DDR>
    // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x30x120x120xf32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!InputStub_CMX = memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x30x120x120xf16, #NHWC, @CMX>

func.func @ClusterConvertDMACopySequence() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !OutputStub_CMX
  %2 = VPUIP.ConvertDMA
      inputs(%0 : !InputDistributedType)
      outputs(%1 : !OutputStub_CMX) ->  !OutputStub_CMX

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !OutputStub_CMX) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x120x120xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[BUF_1:%.*]] = memref.alloc() : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   [[COPY_0:%.*]] = VPUIP.ConvertDMA inputs(%0 : !VPUIP.DistributedBuffer<1x30x120x120xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs(%alloc : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   return [[COPY_0]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedType = !VPUIP.DistributedBuffer<
  1x4x512x1xf32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments
  }
>

!OutputDistributedType = !VPUIP.DistributedBuffer<
  1x4x512x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments
  }
>

func.func @DoNotFuseIncompatibleDistributedCopy() -> !OutputDistributedType {
  %1 = VPURT.AllocDistributed -> !InputDistributedType
  %alloc = memref.alloc() : memref<1x4x512x1xf16, @DDR>
  %2 = VPUIP.ConvertDMA
      inputs(%1 : !InputDistributedType)
      outputs(%alloc : memref<1x4x512x1xf16, @DDR>) ->  memref<1x4x512x1xf16, @DDR>
  %3 = VPURT.AllocDistributed -> !OutputDistributedType
  %4 = VPUIP.Copy
      inputs(%2 : memref<1x4x512x1xf16, @DDR>)
      outputs(%3 : !OutputDistributedType)  ->  !OutputDistributedType

  return %4 : !OutputDistributedType

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x512x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[BUF_1:%.*]] = memref.alloc() : memref<1x4x512x1xf16, @DDR>
  // CHECK:    [[CONVERT_0:%.*]] = VPUIP.ConvertDMA
  // CHECK-SAME:     inputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x4x512x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>)
  // CHECK-SAME:     outputs([[BUF_1]] : memref<1x4x512x1xf16, @DDR>) ->  memref<1x4x512x1xf16, @DDR>
  // CHECK:   [[BUF_2:%.*]] =  VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:    [[COPY_0:%.*]] = VPUIP.Copy
  // CHECK-SAME:     inputs([[CONVERT_0]] : memref<1x4x512x1xf16, @DDR>)
  // CHECK-SAME:     outputs([[BUF_2]] : !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>)  ->  !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   return [[COPY_0]] : !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x3x224x224xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x3x224x224xf32, #NHWC, @DDR>
!Input_CMX = memref<1x3x224x224xf32, #NHWC, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @CopyClusterConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, #NHWC, @DDR>)
func.func @CopyClusterConvertDMASequence(%arg0: !Input_DDR) -> !OutputDistributedType {
  %0 = memref.alloc() : !Input_CMX
  %1 = VPUIP.Copy inputs(%arg0 : !Input_DDR) outputs(%0 : !Input_CMX) -> !Input_CMX

  %2 = VPURT.AllocDistributed -> !OutputDistributedType
  %3 = VPUIP.ConvertDMA
      inputs(%1 : !Input_CMX)
      outputs(%2 : !OutputDistributedType) ->  !OutputDistributedType
  return %3 : !OutputDistributedType

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[ConvertDMA:%.+]] = VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x3x224x224xf32, #NHWC, @DDR>) outputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x3x224x224xf32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x3x224x224xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x3x224x224xf32, #NHWC, @DDR>
!Input_CMX = memref<1x3x224x224xf32, #NHWC, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @ClusterCopyClusterConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, #NHWC, @DDR>)
func.func @ClusterCopyClusterConvertDMASequence(%arg0: !Input_DDR) -> !OutputDistributedType {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = VPUIP.Copy
      inputs(%arg0 : !Input_DDR)
      outputs(%0 : !InputDistributedType)  ->  !InputDistributedType
  %2 = VPURT.AllocDistributed -> !OutputDistributedType
  %3 = VPUIP.ConvertDMA
      inputs(%1 : !InputDistributedType)
      outputs(%2 : !OutputDistributedType) ->  !OutputDistributedType

  return %3 : !OutputDistributedType

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[ConvertDMA:%.*]] = VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x3x224x224xf32, #NHWC, @DDR>) outputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x3x224x224xf32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x3x224x224xf32, #NHWC, @DDR>
!Input_CMX = memref<1x3x224x224xf32, #NHWC, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @ClusterCopyConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, #NHWC, @DDR>)
func.func @ClusterCopyConvertDMASequence(%arg0: !Input_DDR) -> !Output_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = VPUIP.Copy
      inputs(%arg0 : !Input_DDR)
      outputs(%0 : !InputDistributedType)  ->  !InputDistributedType
  %2 = memref.alloc() : !Output_CMX
  %3 = VPUIP.ConvertDMA inputs(%1 : !InputDistributedType) outputs(%2 : !Output_CMX) -> !Output_CMX

  return %3 : !Output_CMX

  // CHECK:   [[BUF_0:%.*]] = memref.alloc() : memref<1x3x224x224xf16, #NHWC, @CMX_NN>
  // CHECK:   [[ConvertDMA:%.*]] = VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x3x224x224xf32,  #NHWC, @DDR>) outputs([[BUF_0]] : memref<1x3x224x224xf16, #NHWC, @CMX_NN>) -> memref<1x3x224x224xf16, #NHWC, @CMX_NN>
  // CHECK:   }
  // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----

!Input_DDR = memref<1x3x224x224xf32, @DDR>
!Input_CMX = memref<1x3x224x224xf32, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, @CMX_NN>

// CHECK-LABEL: @CopyConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, @DDR>)
func.func @CopyConvertDMASequence(%arg0: !Input_DDR) -> !Output_CMX {
  %0 = memref.alloc() : !Input_CMX
  %1 = VPUIP.Copy inputs(%arg0 : !Input_DDR) outputs(%0 : !Input_CMX) -> !Input_CMX

  %2 = memref.alloc() : !Output_CMX
  %3 = VPUIP.ConvertDMA inputs(%1 : !Input_CMX) outputs(%2 : !Output_CMX) -> !Output_CMX

  return %3 : !Output_CMX

  // CHECK:   [[BUF_0:%.*]] = memref.alloc() : memref<1x3x224x224xf16, @CMX_NN>
  // CHECK:   [[ConvertDMA:%.*]] =  VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x3x224x224xf32, @DDR>) outputs([[BUF_0]] : memref<1x3x224x224xf16, @CMX_NN>) -> memref<1x3x224x224xf16, @CMX_NN>
  // CHECK:   }
  // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!copyOutDistedType = !VPUIP.DistributedBuffer<
    64x5504x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    memory_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]
}>

// CHECK-LABEL: @ConvertDMAViewLikeOpMultiSubViewCopy
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x1x64x11008xf32, @DDR>)
func.func @ConvertDMAViewLikeOpMultiSubViewCopy(%arg0: memref<1x1x64x11008xf32, @DDR>) -> (!copyOutDistedType, !copyOutDistedType) {
    %alloc_0 = memref.alloc() : memref<1x1x64x11008xf16, @DDR>
    %0 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x1x64x11008xf32, @DDR>) outputs(%alloc_0 : memref<1x1x64x11008xf16, @DDR>) -> memref<1x1x64x11008xf16, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x1x64x11008xf16, @DDR>) -> memref<64x11008x1x1xf16, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%1 : memref<64x11008x1x1xf16, @DDR>) -> memref<64x11008x1x1xf16, #NHWC, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf16, #NHWC, @DDR> to memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>
    %4 = VPUIP.SubView %2 [0, 5504, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf16, #NHWC, @DDR> to memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>
    %alloc_1 = VPURT.AllocDistributed -> !copyOutDistedType
    %5 = VPUIP.Copy inputs(%4 : memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>) outputs(%alloc_1 : !copyOutDistedType) -> !copyOutDistedType
    %alloc_2 = VPURT.AllocDistributed -> !copyOutDistedType
    %6 = VPUIP.Copy inputs(%3 : memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>) outputs(%alloc_2 : !copyOutDistedType) -> !copyOutDistedType
    return %5, %6 : !copyOutDistedType, !copyOutDistedType

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : memref<1x1x64x11008xf32, @DDR>) -> memref<64x11008x1x1xf32, @DDR>
    // CHECK: [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : memref<64x11008x1x1xf32, @DDR>)
    // CHECK-SAME: -> memref<64x11008x1x1xf32, #NHWC, @DDR>

    // CHECK: [[COPY_OUT_0:%.+]] = VPURT.AllocDistributed
    // CHECK: -> !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 5504, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf32, #NHWC, @DDR>
    // CHECK:  to memref<64x5504x1x1xf32, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>

    // CHECK: [[CONVERT_0:%.+]] = VPUIP.ConvertDMA
    // CHECK: inputs([[SUBVIEW_0]] : memref<64x5504x1x1xf32, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>)
    // CHECK: outputs([[COPY_OUT_0]] : !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN,
    // CHECK: {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>)
    // CHECK: -> !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN

    // CHECK: [[COPY_OUT_1:%.+]] = VPURT.AllocDistributed
    // CHECK: -> !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN

    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 0, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf32, #NHWC, @DDR>
    // CHECK:  to memref<64x5504x1x1xf32, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>

    // CHECK: [[CONVERT_1:%.+]] = VPUIP.ConvertDMA
    // CHECK: inputs([[SUBVIEW_1]] : memref<64x5504x1x1xf32, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>)
    // CHECK: outputs([[COPY_OUT_1]] : !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN,
    // CHECK: {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>)
    // CHECK: -> !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN
    // CHECK: return [[CONVERT_0]], [[CONVERT_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!copyOutDistedType = !VPUIP.DistributedBuffer<
    64x5504x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    memory_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]
}>

// CHECK-LABEL: @ConvertDMAMultiSubViewCopy
// CHECK-SAME: ([[INPUT:%.+]]: memref<64x11008x1x1xf32, #NHWC, @DDR>)
func.func @ConvertDMAMultiSubViewCopy(%arg0: memref<64x11008x1x1xf32, #NHWC, @DDR>) -> (!copyOutDistedType, !copyOutDistedType) {
    %alloc_0 = memref.alloc() : memref<64x11008x1x1xf16, #NHWC, @DDR>
    %0 = VPUIP.ConvertDMA inputs(%arg0 : memref<64x11008x1x1xf32, #NHWC, @DDR>) outputs(%alloc_0 : memref<64x11008x1x1xf16, #NHWC, @DDR>) -> memref<64x11008x1x1xf16, #NHWC, @DDR>
    %3 = VPUIP.SubView %0 [0, 0, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf16, #NHWC, @DDR> to memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>
    %4 = VPUIP.SubView %0 [0, 5504, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf16, #NHWC, @DDR> to memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>
    %alloc_1 = VPURT.AllocDistributed -> !copyOutDistedType
    %5 = VPUIP.Copy inputs(%4 : memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>) outputs(%alloc_1 : !copyOutDistedType) -> !copyOutDistedType
    %alloc_2 = VPURT.AllocDistributed -> !copyOutDistedType
    %6 = VPUIP.Copy inputs(%3 : memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>) outputs(%alloc_2 : !copyOutDistedType) -> !copyOutDistedType
    return %5, %6 : !copyOutDistedType, !copyOutDistedType

    // CHECK: [[COPY_OUT_0:%.+]] = VPURT.AllocDistributed
    // CHECK: -> !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 5504, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf32, #NHWC, @DDR>
    // CHECK:  to memref<64x5504x1x1xf32, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>

    // CHECK: [[CONVERT_0:%.+]] = VPUIP.ConvertDMA
    // CHECK: inputs([[SUBVIEW_0]] : memref<64x5504x1x1xf32, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>)
    // CHECK: outputs([[COPY_OUT_0]] : !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN,
    // CHECK: {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>)
    // CHECK: -> !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN

    // CHECK: [[COPY_OUT_1:%.+]] = VPURT.AllocDistributed
    // CHECK: -> !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN

    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf32, #NHWC, @DDR>
    // CHECK:  to memref<64x5504x1x1xf32, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>

    // CHECK: [[CONVERT_1:%.+]] = VPUIP.ConvertDMA
    // CHECK: inputs([[SUBVIEW_1]] : memref<64x5504x1x1xf32, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>)
    // CHECK: outputs([[COPY_OUT_1]] : !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN,
    // CHECK: {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1], [16, 5504, 1, 1]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>)
    // CHECK: -> !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN
    // CHECK: return [[CONVERT_0]], [[CONVERT_1]]
}
