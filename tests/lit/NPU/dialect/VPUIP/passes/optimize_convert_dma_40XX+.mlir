//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-convert-dma-op %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDMACopySequence
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x16x112x112xf32, {order = #NHWC}, @CMX>, [[ARG_1:%.+]]: memref<1x16x112x112xf16, {order = #NHWC}, @CMX>)
func.func @ConvertDMACopySequence(
        %arg0: memref<1x16x112x112xf32, {order = #NHWC}, @CMX>,
        %arg1: memref<1x16x112x112xf16, {order = #NHWC}, @CMX>)
        -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX> {

    %0 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>
    %1 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x16x112x112xf32, {order = #NHWC}, @CMX>) outputs(%0 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX>

    %2 = VPUIP.Copy inputs(%1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>)
        outputs(%arg1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>)
        -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX>

    return %2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX>

    //CHECK: [[CONVERTDMA:%.+]] = VPUIP.ConvertDMA inputs([[ARG_0]] : memref<1x16x112x112xf32, {order = #NHWC}, @CMX>)
    //CHECK-NOT: VPUIP.Copy
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!SpilledOutput_DDR = memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

// CHECK-LABEL: @ConvertDMAClusterCopySequence
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x16x112x112xf32, {order = #NHWC}, @CMX_NN>)
func.func @ConvertDMAClusterCopySequence(%arg0: memref<1x16x112x112xf32, {order = #NHWC}, @CMX_NN>) -> !SpilledOutput_DDR {

    %0 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %1 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x16x112x112xf32, {order = #NHWC}, @CMX_NN>) outputs(%0 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>

    %2 = memref.alloc() : !SpilledOutput_DDR
    %3 = VPUIP.Copy
        inputs(%1 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%2 : !SpilledOutput_DDR)  ->  !SpilledOutput_DDR

    return %3 : !SpilledOutput_DDR

    // CHECK:   [[BUF_0:%.+]] = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[ConvertDMA:%.+]] = VPUIP.ConvertDMA
    // CHECK-SAME:     inputs([[ARG_0]] : memref<1x16x112x112xf32, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     outputs([[BUF_0]] : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>) ->  memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
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
    %2 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%1 : memref<32x2048x1x1xf16, @DDR>) -> memref<32x2048x1x1xf16, {order = #NHWC}, @DDR>

    %3 = VPURT.AllocDistributed -> !outDistributedType
    %4 = VPUIP.Copy inputs(%2 : memref<32x2048x1x1xf16, {order = #NHWC}, @DDR>) outputs(%3 : !outDistributedType) -> !outDistributedType
    return %4 : !outDistributedType

    // CHECK: [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK: -> !VPUIP.DistributedBuffer<32x2048x1x1xf16, #NHWC, @CMX_NN

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : memref<1x1x32x2048xf32, @DDR>) -> memref<32x2048x1x1xf32, @DDR>
    // CHECK: [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : memref<32x2048x1x1xf32, @DDR>)
    // CHECK-SAME: -> memref<32x2048x1x1xf32, {order = #NHWC}, @DDR>

    // CHECK: [[CONVERT:%.+]] = VPUIP.ConvertDMA
    // CHECK: inputs([[PERMUTE]] : memref<32x2048x1x1xf32, {order = #NHWC}, @DDR>)
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
func.func @NotFuseConvertDMAAndCopyDueToDistOutputAndViewLikeOp(%arg0: memref<1x1x1x32xf32, @DDR>) -> memref<32x1x1x1xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.AllocDistributed -> !outDistributedType
    %1 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x1x1x32xf32, @DDR>) outputs(%0 : !outDistributedType) -> !outDistributedType
    %2 = VPUIP.GenericReshape inputs(%1 : !outDistributedType) -> !reshapeDistributedType
    %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%2 : !reshapeDistributedType) -> !permuteDistributedType
    %alloc_0 = memref.alloc() : memref<32x1x1x1xf16, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%3 : !permuteDistributedType) outputs(%alloc_0 : memref<32x1x1x1xf16, {order = #NHWC}, @DDR>) -> memref<32x1x1x1xf16, {order = #NHWC}, @DDR>
    return %4 : memref<32x1x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK: [[ALLOC_DIST:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK: [[CONVERT_DMA:%.+]] = VPUIP.ConvertDMA
    // CHECK:  inputs([[INPUT]] : memref<1x1x1x32xf32, @DDR>)
    // CHECK:  outputs([[ALLOC_DIST]] : !VPUIP.DistributedBuffer<1x1x1x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape
    // CHECK:  inputs([[CONVERT_DMA]] : !VPUIP.DistributedBuffer<1x1x1x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK:  -> !VPUIP.DistributedBuffer<32x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments

    // CHECK: [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:  inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<32x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK:  -> !VPUIP.DistributedBuffer<32x1x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments

    // CHECK: [[MEM_ALLOC:%.+]] = memref.alloc() : memref<32x1x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK: [[COPY:%.+]] = VPUIP.Copy
    // CHECK:  inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<32x1x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK:  outputs([[MEM_ALLOC]] : memref<32x1x1x1xf16, {order = #NHWC}, @DDR>) -> memref<32x1x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK: return [[COPY]] : memref<32x1x1x1xf16, {order = #NHWC}, @DDR>
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
!Output_CMX = memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
!Input_CMX = memref<1x16x112x112xf32, {order = #NHWC}, @CMX_NN>

// CHECK-LABEL: @ClusterConvertDMAClusterCopySequence
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x16x112x112xf32, {order = #NHWC}, @CMX_NN>)
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

    // CHECK:   [[BUF_0:%.+]] = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
    // CHECK:   [[ConvertDMA:%.+]] = VPUIP.ConvertDMA
    // CHECK-SAME:     inputs([[ARG_0]] : memref<1x16x112x112xf32, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     outputs([[BUF_0]] : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>) ->  memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
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

!InputStub_CMX = memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x30x120x120xf16, {order = #NHWC}, @CMX>

func.func @ClusterConvertDMACopySequence() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !OutputStub_CMX
  %2 = VPUIP.ConvertDMA
      inputs(%0 : !InputDistributedType)
      outputs(%1 : !OutputStub_CMX) ->  !OutputStub_CMX

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !OutputStub_CMX) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX

  // CHECK:   [[BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x120x120xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[BUF_1:%.+]] = memref.alloc() : memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK:   [[COPY_0:%.+]] = VPUIP.ConvertDMA inputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x30x120x120xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) outputs([[BUF_1]] : memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK:   return [[COPY_0]] : memref<1x30x120x120xf16, {order = #NHWC}, [@CMX_NN, 0]>
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

  // CHECK:   [[BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x512x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[BUF_1:%.+]] = memref.alloc() : memref<1x4x512x1xf16, @DDR>
  // CHECK:    [[CONVERT_0:%.+]] = VPUIP.ConvertDMA
  // CHECK-SAME:     inputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x4x512x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>)
  // CHECK-SAME:     outputs([[BUF_1]] : memref<1x4x512x1xf16, @DDR>) ->  memref<1x4x512x1xf16, @DDR>
  // CHECK:   [[BUF_2:%.+]] =  VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
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

!Input_DDR = memref<1x3x224x224xf32, {order = #NHWC}, @DDR>
!Input_CMX = memref<1x3x224x224xf32, {order = #NHWC}, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, {order = #NHWC}, @CMX_NN>

// CHECK-LABEL: @CopyClusterConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, {order = #NHWC}, @DDR>)
func.func @CopyClusterConvertDMASequence(%arg0: !Input_DDR) -> !OutputDistributedType {
  %0 = memref.alloc() : !Input_CMX
  %1 = VPUIP.Copy inputs(%arg0 : !Input_DDR) outputs(%0 : !Input_CMX) -> !Input_CMX

  %2 = VPURT.AllocDistributed -> !OutputDistributedType
  %3 = VPUIP.ConvertDMA
      inputs(%1 : !Input_CMX)
      outputs(%2 : !OutputDistributedType) ->  !OutputDistributedType
  return %3 : !OutputDistributedType

  // CHECK:   [[BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[ConvertDMA:%.+]] = VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x3x224x224xf32, {order = #NHWC}, @DDR>) outputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
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

!Input_DDR = memref<1x3x224x224xf32, {order = #NHWC}, @DDR>
!Input_CMX = memref<1x3x224x224xf32, {order = #NHWC}, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, {order = #NHWC}, @CMX_NN>

// CHECK-LABEL: @ClusterCopyClusterConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, {order = #NHWC}, @DDR>)
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

  // CHECK:   [[BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[ConvertDMA:%.+]] = VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x3x224x224xf32, {order = #NHWC}, @DDR>) outputs([[BUF_0]] : !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
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

!Input_DDR = memref<1x3x224x224xf32, {order = #NHWC}, @DDR>
!Input_CMX = memref<1x3x224x224xf32, {order = #NHWC}, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, {order = #NHWC}, @CMX_NN>

// CHECK-LABEL: @ClusterCopyConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, {order = #NHWC}, @DDR>)
func.func @ClusterCopyConvertDMASequence(%arg0: !Input_DDR) -> !Output_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = VPUIP.Copy
      inputs(%arg0 : !Input_DDR)
      outputs(%0 : !InputDistributedType)  ->  !InputDistributedType
  %2 = memref.alloc() : !Output_CMX
  %3 = VPUIP.ConvertDMA inputs(%1 : !InputDistributedType) outputs(%2 : !Output_CMX) -> !Output_CMX

  return %3 : !Output_CMX

  // CHECK:   [[BUF_0:%.+]] = memref.alloc() : memref<1x3x224x224xf16, {order = #NHWC}, @CMX_NN>
  // CHECK:   [[ConvertDMA:%.+]] = VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x3x224x224xf32, {order = #NHWC}, @DDR>) outputs([[BUF_0]] : memref<1x3x224x224xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x3x224x224xf16, {order = #NHWC}, @CMX_NN>
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

  // CHECK:   [[BUF_0:%.+]] = memref.alloc() : memref<1x3x224x224xf16, @CMX_NN>
  // CHECK:   [[ConvertDMA:%.+]] =  VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x3x224x224xf32, @DDR>) outputs([[BUF_0]] : memref<1x3x224x224xf16, @CMX_NN>) -> memref<1x3x224x224xf16, @CMX_NN>
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
func.func @ConvertDMAViewLikeOpMultiSubViewCopy(%arg0: memref<1x1x64x11008xf32, @DDR>)
        -> (!copyOutDistedType, !copyOutDistedType, !copyOutDistedType, !copyOutDistedType) {
    %alloc_0 = memref.alloc() : memref<1x1x64x11008xf16, @DDR>
    %0 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x1x64x11008xf32, @DDR>) outputs(%alloc_0 : memref<1x1x64x11008xf16, @DDR>) -> memref<1x1x64x11008xf16, @DDR>
    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x1x64x11008xf16, @DDR>) -> memref<64x11008x1x1xf16, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%1 : memref<64x11008x1x1xf16, @DDR>) -> memref<64x11008x1x1xf16, {order = #NHWC}, @DDR>
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf16, {order = #NHWC}, @DDR> to memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>
    %4 = VPUIP.SubView %2 [0, 5504, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf16, {order = #NHWC}, @DDR> to memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>
    %alloc_1 = VPURT.AllocDistributed -> !copyOutDistedType
    %5 = VPUIP.Copy inputs(%4 : memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>) outputs(%alloc_1 : !copyOutDistedType) -> !copyOutDistedType
    %alloc_2 = VPURT.AllocDistributed -> !copyOutDistedType
    %6 = VPUIP.Copy inputs(%3 : memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>) outputs(%alloc_2 : !copyOutDistedType) -> !copyOutDistedType
    // Each Copy result has multiple SSA uses. They must all be replaced by the corresponding ConvertDMA result.
    return %5, %5, %6, %6 :
        !copyOutDistedType, !copyOutDistedType, !copyOutDistedType, !copyOutDistedType

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : memref<1x1x64x11008xf32, @DDR>) -> memref<64x11008x1x1xf32, @DDR>
    // CHECK: [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : memref<64x11008x1x1xf32, @DDR>)
    // CHECK-SAME: -> memref<64x11008x1x1xf32, {order = #NHWC}, @DDR>

    // CHECK: [[COPY_OUT_0:%.+]] = VPURT.AllocDistributed
    // CHECK: -> !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 5504, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf32, {order = #NHWC}, @DDR>
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

    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 0, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf32, {order = #NHWC}, @DDR>
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
    // CHECK: return [[CONVERT_0]], [[CONVERT_0]], [[CONVERT_1]], [[CONVERT_1]]
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
// CHECK-SAME: ([[INPUT:%.+]]: memref<64x11008x1x1xf32, {order = #NHWC}, @DDR>)
func.func @ConvertDMAMultiSubViewCopy(%arg0: memref<64x11008x1x1xf32, {order = #NHWC}, @DDR>) -> (!copyOutDistedType, !copyOutDistedType) {
    %alloc_0 = memref.alloc() : memref<64x11008x1x1xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.ConvertDMA inputs(%arg0 : memref<64x11008x1x1xf32, {order = #NHWC}, @DDR>) outputs(%alloc_0 : memref<64x11008x1x1xf16, {order = #NHWC}, @DDR>) -> memref<64x11008x1x1xf16, {order = #NHWC}, @DDR>
    %3 = VPUIP.SubView %0 [0, 0, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf16, {order = #NHWC}, @DDR> to memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>
    %4 = VPUIP.SubView %0 [0, 5504, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf16, {order = #NHWC}, @DDR> to memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>
    %alloc_1 = VPURT.AllocDistributed -> !copyOutDistedType
    %5 = VPUIP.Copy inputs(%4 : memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>) outputs(%alloc_1 : !copyOutDistedType) -> !copyOutDistedType
    %alloc_2 = VPURT.AllocDistributed -> !copyOutDistedType
    %6 = VPUIP.Copy inputs(%3 : memref<64x5504x1x1xf16, {order = #NHWC, strides = [11008, 1, 11008, 11008]}, @DDR>) outputs(%alloc_2 : !copyOutDistedType) -> !copyOutDistedType
    return %5, %6 : !copyOutDistedType, !copyOutDistedType

    // CHECK: [[COPY_OUT_0:%.+]] = VPURT.AllocDistributed
    // CHECK: -> !VPUIP.DistributedBuffer<64x5504x1x1xf16, #NHWC, @CMX_NN

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 5504, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf32, {order = #NHWC}, @DDR>
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

    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [64, 5504, 1, 1] : memref<64x11008x1x1xf32, {order = #NHWC}, @DDR>
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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvertDMASubViewCopyWithStrides
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x1x342x1536xf32, {order = #NCHW, strides = [1048576, 1048576, 1536, 1]}, @DDR>)
func.func @ConvertDMASubViewCopyWithStrides(
        %arg0: memref<1x1x342x1536xf32, {order = #NCHW, strides = [1048576, 1048576, 1536, 1]}, @DDR>)
        -> (memref<1x1x100x1536xf16, @DDR>, memref<1x1x100x1536xf16, @DDR>) {
    %alloc_ddr = memref.alloc() : memref<1x1x342x1536xf16, @DDR>
    %0 = VPUIP.ConvertDMA
            inputs(%arg0 : memref<1x1x342x1536xf32, {order = #NCHW, strides = [1048576, 1048576, 1536, 1]}, @DDR>)
            outputs(%alloc_ddr : memref<1x1x342x1536xf16, @DDR>)
            -> memref<1x1x342x1536xf16, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 1, 100, 1536] [1, 1, 2, 1]
            : memref<1x1x342x1536xf16, @DDR>
            to memref<1x1x100x1536xf16, {order = #NCHW, strides = [525312, 525312, 3072, 1]}, @DDR>
    %alloc_ddr0 = memref.alloc() : memref<1x1x100x1536xf16, @DDR>
    %2 = VPUIP.Copy
            inputs(%1 : memref<1x1x100x1536xf16, {order = #NCHW, strides = [525312, 525312, 3072, 1]}, @DDR>)
            outputs(%alloc_ddr0 : memref<1x1x100x1536xf16, @DDR>)
            -> memref<1x1x100x1536xf16, @DDR>
    %3 = VPUIP.SubView %0 [0, 0, 100, 0] [1, 1, 100, 1536]
            : memref<1x1x342x1536xf16, @DDR>
            to memref<1x1x100x1536xf16, {order = #NCHW, strides = [525312, 525312, 1536, 1]}, @DDR>
    %alloc_ddr1 = memref.alloc() : memref<1x1x100x1536xf16, @DDR>
    %4 = VPUIP.Copy
            inputs(%3 : memref<1x1x100x1536xf16, {order = #NCHW, strides = [525312, 525312, 1536, 1]}, @DDR>)
            outputs(%alloc_ddr1 : memref<1x1x100x1536xf16, @DDR>)
            -> memref<1x1x100x1536xf16, @DDR>
    return %2, %4 : memref<1x1x100x1536xf16, @DDR>, memref<1x1x100x1536xf16, @DDR>

    // Both SubViews move before ConvertDMA; intermediate f16 DDR allocation is eliminated.
    // CHECK:     [[DDR_BUF0:%.+]] = memref.alloc() : memref<1x1x100x1536xf16, @DDR>
    // CHECK:     [[SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 1, 100, 1536] [1, 1, 2, 1]
    // CHECK-SAME:  memref<1x1x342x1536xf32, {order = #NCHW, strides = [1048576, 1048576, 1536, 1]}, @DDR>
    // CHECK-SAME:  to memref<1x1x100x1536xf32, {order = #NCHW, strides = [1048576, 1048576, 3072, 1]}, @DDR>
    // CHECK:     [[CONVERT0:%.+]] = VPUIP.ConvertDMA
    // CHECK-SAME:  inputs([[SUBVIEW0]] : memref<1x1x100x1536xf32, {order = #NCHW, strides = [1048576, 1048576, 3072, 1]}, @DDR>)
    // CHECK-SAME:  outputs([[DDR_BUF0]] : memref<1x1x100x1536xf16, @DDR>) -> memref<1x1x100x1536xf16, @DDR>
    // CHECK:     [[DDR_BUF1:%.+]] = memref.alloc() : memref<1x1x100x1536xf16, @DDR>
    // CHECK:     [[SUBVIEW1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 100, 0] [1, 1, 100, 1536]
    // CHECK-SAME:  memref<1x1x342x1536xf32, {order = #NCHW, strides = [1048576, 1048576, 1536, 1]}, @DDR>
    // CHECK-SAME:  to memref<1x1x100x1536xf32, {order = #NCHW, strides = [1048576, 1048576, 1536, 1]}, @DDR>
    // CHECK:     [[CONVERT1:%.+]] = VPUIP.ConvertDMA
    // CHECK-SAME:  inputs([[SUBVIEW1]] : memref<1x1x100x1536xf32, {order = #NCHW, strides = [1048576, 1048576, 1536, 1]}, @DDR>)
    // CHECK-SAME:  outputs([[DDR_BUF1]] : memref<1x1x100x1536xf16, @DDR>) -> memref<1x1x100x1536xf16, @DDR>
    // CHECK-NOT:   VPUIP.Copy
    // CHECK:     return [[CONVERT0]], [[CONVERT1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!output = !VPUIP.DistributedBuffer<
    1x8x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

// CHECK-LABEL: @ConvertDMAConcatViewSubViewCopy
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x1x4x8xf32, @DDR>)
func.func @ConvertDMAConcatViewSubViewCopy(%arg0: memref<1x1x4x8xf32, @DDR>) -> !output {
    // ConvertDMASubViewCopy recognizes the partitioned producer through ConcatView, recovers the whole-tensor
    // source, and reuses its view-chain movement and per-consumer ConvertDMA rewrite directly.
    // CHECK-NOT:  memref.alloc() : memref<1x1x4x8xf16, @DDR>
    // CHECK-NOT:  VPUIP.ConcatView
    // CHECK-NOT:  VPUIP.Copy

    %full_f16 = memref.alloc() : memref<1x1x4x8xf16, @DDR>

    %input_0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 4, 4] :
        memref<1x1x4x8xf32, @DDR> to memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %output_0 = VPUIP.SubView %full_f16 [0, 0, 0, 0] [1, 1, 4, 4] :
        memref<1x1x4x8xf16, @DDR> to memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %convert_0 = VPUIP.ConvertDMA inputs(%input_0 : memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        outputs(%output_0 : memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        -> memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>

    %input_1 = VPUIP.SubView %arg0 [0, 0, 0, 4] [1, 1, 4, 4] :
        memref<1x1x4x8xf32, @DDR> to memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %output_1 = VPUIP.SubView %full_f16 [0, 0, 0, 4] [1, 1, 4, 4] :
        memref<1x1x4x8xf16, @DDR> to memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %convert_1 = VPUIP.ConvertDMA inputs(%input_1 : memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        outputs(%output_1 : memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        -> memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>

    %concat = VPUIP.ConcatView inputs(
        %convert_0, %convert_1 :
        memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>,
        memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        outputs(%full_f16 : memref<1x1x4x8xf16, @DDR>) -> memref<1x1x4x8xf16, @DDR>

    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x1x4x8xf16, @DDR>)
        -> memref<4x8x1x1xf16, @DDR>
    %permute = VPUIP.PermuteCast {
        dst_order = #NHWC,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>}
        inputs(%reshape : memref<4x8x1x1xf16, @DDR>)
        -> memref<1x8x4x1xf16, {order = #NHWC}, @DDR>
    %final_reshape = VPUIP.GenericReshape inputs(%permute : memref<1x8x4x1xf16, {order = #NHWC}, @DDR>)
        -> memref<1x8x2x2xf16, {order = #NHWC}, @DDR>

    %tile_0 = VPUIP.SubView %final_reshape [0, 0, 0, 0] [1, 8, 1, 2] :
        memref<1x8x2x2xf16, {order = #NHWC}, @DDR> to memref<1x8x1x2xf16, {order = #NHWC, strides = [32, 1, 16, 8]}, @DDR>
    %cmx_0 = VPURT.AllocDistributed -> !output
    %copy_0 = VPUIP.Copy inputs(%tile_0 : memref<1x8x1x2xf16, {order = #NHWC, strides = [32, 1, 16, 8]}, @DDR>)
        outputs(%cmx_0 : !output) -> !output

    return %copy_0 : !output

    // CHECK:      [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : memref<1x1x4x8xf32, @DDR>) -> memref<4x8x1x1xf32, @DDR>
    // CHECK:      [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #map} inputs([[RESHAPE]] : memref<4x8x1x1xf32, @DDR>)
    // CHECK:      [[FINAL_RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[PERMUTE]] : memref<1x8x4x1xf32, {order = #NHWC}, @DDR>)
    // CHECK-SAME:   -> memref<1x8x2x2xf32, {order = #NHWC}, @DDR>

    // CHECK:      [[CMX_BUF_0:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x8x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED"
    // CHECK:      [[SUBVIEW_0:%.+]] = VPUIP.SubView [[FINAL_RESHAPE]] [0, 0, 0, 0] [1, 8, 1, 2] : memref<1x8x2x2xf32, {order = #NHWC}, @DDR>
    // CHECK-SAME:   to memref<1x8x1x2xf32, {order = #NHWC, strides = [32, 1, 16, 8]}, @DDR>
    // CHECK:      [[CONVERT_0:%.+]] = VPUIP.ConvertDMA
    // CHECK-SAME:   inputs([[SUBVIEW_0]] : memref<1x8x1x2xf32, {order = #NHWC, strides = [32, 1, 16, 8]}, @DDR>)
    // CHECK-SAME:   outputs([[CMX_BUF_0]] : !VPUIP.DistributedBuffer<1x8x1x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED"

    // CHECK-NOT:  VPUIP.ConcatView
    // CHECK-NOT:  VPUIP.Copy
    // CHECK:      return [[CONVERT_0]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// The two ConvertDMA operations swap the halves, so their input and output SubViews do not describe
// the same tiling. The ConvertDMASubViewCopy extension must not rewrite this chain from the whole source.

// CHECK-LABEL: @NotFusedSwappedPartition
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x1x4x8xf32, @DDR>)
func.func @NotFusedSwappedPartition(%arg0: memref<1x1x4x8xf32, @DDR>) -> memref<4x8x1x1xf16, @DDR> {
    %full_f16 = memref.alloc() : memref<1x1x4x8xf16, @DDR>

    %input_0 = VPUIP.SubView %arg0 [0, 0, 0, 4] [1, 1, 4, 4] :
        memref<1x1x4x8xf32, @DDR> to memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %output_0 = VPUIP.SubView %full_f16 [0, 0, 0, 0] [1, 1, 4, 4] :
        memref<1x1x4x8xf16, @DDR> to memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %convert_0 = VPUIP.ConvertDMA inputs(%input_0 : memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        outputs(%output_0 : memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        -> memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>

    %input_1 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 4, 4] :
        memref<1x1x4x8xf32, @DDR> to memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %output_1 = VPUIP.SubView %full_f16 [0, 0, 0, 4] [1, 1, 4, 4] :
        memref<1x1x4x8xf16, @DDR> to memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %convert_1 = VPUIP.ConvertDMA inputs(%input_1 : memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        outputs(%output_1 : memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        -> memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>

    %concat = VPUIP.ConcatView inputs(
        %convert_0, %convert_1 :
        memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>,
        memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        outputs(%full_f16 : memref<1x1x4x8xf16, @DDR>) -> memref<1x1x4x8xf16, @DDR>

    // Keep a single whole-tensor view user so the matcher reaches the partition-offset check.
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x1x4x8xf16, @DDR>)
        -> memref<4x8x1x1xf16, @DDR>
    return %reshape : memref<4x8x1x1xf16, @DDR>

    // CHECK:      [[DDR_BUF:%.+]] = memref.alloc() : memref<1x1x4x8xf16, @DDR>
    // CHECK:      [[CONVERT_IN_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 4] [1, 1, 4, 4] : memref<1x1x4x8xf32, @DDR>
    // CHECK:      [[CONVERT_OUT_0:%.+]] = VPUIP.SubView [[DDR_BUF]] [0, 0, 0, 0] [1, 1, 4, 4] : memref<1x1x4x8xf16, @DDR>
    // CHECK:      [[CONVERT_0:%.+]] = VPUIP.ConvertDMA
    // CHECK-SAME:   inputs([[CONVERT_IN_0]]
    // CHECK-SAME:   outputs([[CONVERT_OUT_0]]
    // CHECK:      [[CONVERT_IN_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 1, 4, 4] : memref<1x1x4x8xf32, @DDR>
    // CHECK:      [[CONVERT_OUT_1:%.+]] = VPUIP.SubView [[DDR_BUF]] [0, 0, 0, 4] [1, 1, 4, 4] : memref<1x1x4x8xf16, @DDR>
    // CHECK:      [[CONVERT_1:%.+]] = VPUIP.ConvertDMA
    // CHECK-SAME:   inputs([[CONVERT_IN_1]]
    // CHECK-SAME:   outputs([[CONVERT_OUT_1]]
    // CHECK:      [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CONVERT_0]], [[CONVERT_1]]
    // CHECK-SAME:   outputs([[DDR_BUF]]
    // CHECK:      [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]]
    // CHECK:      return [[RESHAPE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// The W=475 f32 slice has a 1900-byte contiguous region, which is not 512-byte aligned.
// A bad multi-use candidate keeps every original Copy, including the compact candidate.

// CHECK-LABEL: @NotFuseInefficientMultiUseSlice
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x8x25x950xf32, @DDR>)
func.func @NotFuseInefficientMultiUseSlice(%arg0: memref<1x8x25x950xf32, @DDR>)
        -> (memref<1x8x25x950xf16, @DDR>, memref<1x8x25x950xf16, @DDR>,
            memref<1x8x25x475xf16, @DDR>, memref<1x8x25x475xf16, @DDR>) {
    %full = memref.alloc() : memref<1x8x25x950xf16, @DDR>
    %convert = VPUIP.ConvertDMA
        inputs(%arg0 : memref<1x8x25x950xf32, @DDR>)
        outputs(%full : memref<1x8x25x950xf16, @DDR>)
        -> memref<1x8x25x950xf16, @DDR>

    %full_tile = VPUIP.SubView %convert [0, 0, 0, 0] [1, 8, 25, 950] :
        memref<1x8x25x950xf16, @DDR> to memref<1x8x25x950xf16, @DDR>
    %full_out = memref.alloc() : memref<1x8x25x950xf16, @DDR>
    %full_copy = VPUIP.Copy
        inputs(%full_tile : memref<1x8x25x950xf16, @DDR>)
        outputs(%full_out : memref<1x8x25x950xf16, @DDR>)
        -> memref<1x8x25x950xf16, @DDR>

    %bad_tile = VPUIP.SubView %convert [0, 0, 0, 0] [1, 8, 25, 475] :
        memref<1x8x25x950xf16, @DDR> to
        memref<1x8x25x475xf16, {order = #NCHW, strides = [190000, 23750, 950, 1]}, @DDR>
    %bad_out = memref.alloc() : memref<1x8x25x475xf16, @DDR>
    %bad_copy = VPUIP.Copy
        inputs(%bad_tile : memref<1x8x25x475xf16, {order = #NCHW, strides = [190000, 23750, 950, 1]}, @DDR>)
        outputs(%bad_out : memref<1x8x25x475xf16, @DDR>)
        -> memref<1x8x25x475xf16, @DDR>

    return %full_copy, %full_copy, %bad_copy, %bad_copy :
        memref<1x8x25x950xf16, @DDR>, memref<1x8x25x950xf16, @DDR>,
        memref<1x8x25x475xf16, @DDR>, memref<1x8x25x475xf16, @DDR>

    // CHECK:      [[FULL:%.+]] = memref.alloc() : memref<1x8x25x950xf16, @DDR>
    // CHECK:      [[CONVERT:%.+]] = VPUIP.ConvertDMA
    // CHECK-SAME:   inputs([[INPUT]]
    // CHECK-SAME:   outputs([[FULL]]
    // CHECK:      [[GOOD_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:   inputs([[CONVERT]]
    // CHECK:      [[BAD_TILE:%.+]] = VPUIP.SubView [[CONVERT]] [0, 0, 0, 0] [1, 8, 25, 475]
    // CHECK:      [[BAD_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:   inputs([[BAD_TILE]]
    // CHECK:      return [[GOOD_COPY]], [[GOOD_COPY]], [[BAD_COPY]], [[BAD_COPY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// The producer tiles are joined before the consumers slice along another axis. Each slice then owns
// a separate view chain, so there is no whole-tensor view chain for ConvertDMASubViewCopy to move ahead
// of the partitioned ConvertDMA producers. This is the group-128 topology and must remain partitioned.

// CHECK-LABEL: @NotCollapsedWhenSlicesPrecedeViews
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x1x4x8xf32, @DDR>)
func.func @NotCollapsedWhenSlicesPrecedeViews(%arg0: memref<1x1x4x8xf32, @DDR>)
        -> (memref<2x8x1x1xf16, @DDR>, memref<2x8x1x1xf16, @DDR>) {
    %full_f16 = memref.alloc() : memref<1x1x4x8xf16, @DDR>

    %input_0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 4, 4] :
        memref<1x1x4x8xf32, @DDR> to memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %output_0 = VPUIP.SubView %full_f16 [0, 0, 0, 0] [1, 1, 4, 4] :
        memref<1x1x4x8xf16, @DDR> to memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %convert_0 = VPUIP.ConvertDMA inputs(%input_0 : memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        outputs(%output_0 : memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        -> memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>

    %input_1 = VPUIP.SubView %arg0 [0, 0, 0, 4] [1, 1, 4, 4] :
        memref<1x1x4x8xf32, @DDR> to memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %output_1 = VPUIP.SubView %full_f16 [0, 0, 0, 4] [1, 1, 4, 4] :
        memref<1x1x4x8xf16, @DDR> to memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %convert_1 = VPUIP.ConvertDMA inputs(%input_1 : memref<1x1x4x4xf32, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        outputs(%output_1 : memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        -> memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>

    %concat = VPUIP.ConcatView inputs(
        %convert_0, %convert_1 :
        memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>,
        memref<1x1x4x4xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        outputs(%full_f16 : memref<1x1x4x8xf16, @DDR>) -> memref<1x1x4x8xf16, @DDR>

    %slice_0 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 1, 2, 8] :
        memref<1x1x4x8xf16, @DDR> to memref<1x1x2x8xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %reshape_0 = VPUIP.GenericReshape inputs(%slice_0 : memref<1x1x2x8xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        -> memref<2x8x1x1xf16, @DDR>
    %slice_1 = VPUIP.SubView %concat [0, 0, 2, 0] [1, 1, 2, 8] :
        memref<1x1x4x8xf16, @DDR> to memref<1x1x2x8xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>
    %reshape_1 = VPUIP.GenericReshape inputs(%slice_1 : memref<1x1x2x8xf16, {order = #NCHW, strides = [32, 32, 8, 1]}, @DDR>)
        -> memref<2x8x1x1xf16, @DDR>

    return %reshape_0, %reshape_1 : memref<2x8x1x1xf16, @DDR>, memref<2x8x1x1xf16, @DDR>

    // CHECK:      [[BUF:%.+]] = memref.alloc() : memref<1x1x4x8xf16, @DDR>
    // CHECK:      [[CONVERT_0:%.+]] = VPUIP.ConvertDMA
    // CHECK:      [[CONVERT_1:%.+]] = VPUIP.ConvertDMA
    // CHECK:      [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CONVERT_0]], [[CONVERT_1]]
    // CHECK-SAME:   outputs([[BUF]]
    // CHECK:      [[SLICE_0:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 1, 2, 8]
    // CHECK:      [[RESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[SLICE_0]]
    // CHECK:      [[SLICE_1:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 2, 0] [1, 1, 2, 8]
    // CHECK:      [[RESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[SLICE_1]]
    // CHECK:      return [[RESHAPE_0]], [[RESHAPE_1]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDMAViewChainWithLayoutChangingReshape
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x1x203x12xf32, {order = #NCWH}, @DDR>)
func.func @ConvertDMAViewChainWithLayoutChangingReshape(
        %arg0: memref<1x1x203x12xf32, {order = #NCWH}, @DDR>)
        -> memref<1x203x1x12xf16, {order = #NHWC}, @CMX_NN> {
    %convert_buffer = memref.alloc() : memref<1x1x203x12xf16, {order = #NCWH}, @CMX_NN>
    %convert = VPUIP.ConvertDMA
        inputs(%arg0 : memref<1x1x203x12xf32, {order = #NCWH}, @DDR>)
        outputs(%convert_buffer : memref<1x1x203x12xf16, {order = #NCWH}, @CMX_NN>)
        -> memref<1x1x203x12xf16, {order = #NCWH}, @CMX_NN>

    %copy_buffer = memref.alloc() : memref<1x1x203x12xf16, {order = #NCWH}, @DDR>
    %copy = VPUIP.Copy
        inputs(%convert : memref<1x1x203x12xf16, {order = #NCWH}, @CMX_NN>)
        outputs(%copy_buffer : memref<1x1x203x12xf16, {order = #NCWH}, @DDR>)
        -> memref<1x1x203x12xf16, {order = #NCWH}, @DDR>
    %reshape = VPUIP.GenericReshape
        inputs(%copy : memref<1x1x203x12xf16, {order = #NCWH}, @DDR>)
        -> memref<1x203x1x12xf16, {order = #NWCH}, @DDR>
    %permute = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NWCH}
        inputs(%reshape : memref<1x203x1x12xf16, {order = #NWCH}, @DDR>)
        -> memref<1x203x1x12xf16, {order = #NHWC}, @DDR>

    %output = memref.alloc() : memref<1x203x1x12xf16, {order = #NHWC}, @CMX_NN>
    %result = VPUIP.Copy
        inputs(%permute : memref<1x203x1x12xf16, {order = #NHWC}, @DDR>)
        outputs(%output : memref<1x203x1x12xf16, {order = #NHWC}, @CMX_NN>)
        -> memref<1x203x1x12xf16, {order = #NHWC}, @CMX_NN>
    return %result : memref<1x203x1x12xf16, {order = #NHWC}, @CMX_NN>

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : memref<1x1x203x12xf32, {order = #NCWH}, @DDR>)
    // CHECK-SAME: -> memref<1x203x1x12xf32, {order = #NWCH}, @DDR>
    // CHECK: [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK-SAME: inputs([[RESHAPE]] : memref<1x203x1x12xf32, {order = #NWCH}, @DDR>)
    // CHECK-SAME: -> memref<1x203x1x12xf32, {order = #NHWC}, @DDR>
    // CHECK: [[CONVERT:%.+]] = VPUIP.ConvertDMA
    // CHECK-SAME: inputs([[PERMUTE]] : memref<1x203x1x12xf32, {order = #NHWC}, @DDR>)
    // CHECK-SAME: outputs([[OUTPUT:%.+]] : memref<1x203x1x12xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-NOT: VPUIP.Copy
    // CHECK: return [[CONVERT]]
}
