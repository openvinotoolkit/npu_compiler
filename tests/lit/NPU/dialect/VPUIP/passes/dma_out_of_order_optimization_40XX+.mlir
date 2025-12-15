//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dma-out-of-order-optimization %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x64x64xf16, #NHWC, @DDR>

//CHECK-LABEL: @DMAOutOfOrderOptimization
func.func @DMAOutOfOrderOptimization() -> !type_DDR {

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %buf1 = VPURT.DeclareBuffer <DDR> <24576> -> !type_DDR
    %output = VPURT.DeclareBuffer <DDR> <49152> -> !type_DDR

    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !type_DDR) outputs(%buf0: !type_DDR) -> !type_DDR
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf1 : !type_DDR) outputs(%output: !type_DDR) -> !type_DDR
    }
    return %output : !type_DDR

    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-NOT:     is_out_of_order
    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-SAME:    is_out_of_order
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x64x64xf16, #NHWC, @DDR>

//CHECK-LABEL: @NoDMAOutOfOrderOptimizationDueToMemOverlap
func.func @NoDMAOutOfOrderOptimizationDueToMemOverlap() -> !type_DDR {

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR
    %buf = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output = VPURT.DeclareBuffer <DDR> <49152> -> !type_DDR

    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !type_DDR) outputs(%buf: !type_DDR) -> !type_DDR
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf : !type_DDR) outputs(%output: !type_DDR) -> !type_DDR
    }
    return %output : !type_DDR

    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-NOT:     is_out_of_order
    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-NOT:     is_out_of_order
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x64x64xf16, #NHWC, @DDR>

//CHECK-LABEL: @NoDMAOutOfOrderOptimizationOn3rdTaskDueToMemOverlapWith1stTask
func.func @NoDMAOutOfOrderOptimizationOn3rdTaskDueToMemOverlapWith1stTask() -> !type_DDR {

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %buf1 = VPURT.DeclareBuffer <DDR> <24576> -> !type_DDR
    %buf2 = VPURT.DeclareBuffer <DDR> <49152> -> !type_DDR
    %output = VPURT.DeclareBuffer <DDR> <73728> -> !type_DDR

    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !type_DDR) outputs(%buf0: !type_DDR) -> !type_DDR
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf1 : !type_DDR) outputs(%buf2: !type_DDR) -> !type_DDR
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf0 : !type_DDR) outputs(%output: !type_DDR) -> !type_DDR
    }
    return %output : !type_DDR

    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-NOT:     is_out_of_order
    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-SAME:    is_out_of_order
    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-NOT:     is_out_of_order
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x64x64xf16, #NHWC, @DDR>

//CHECK-LABEL: @DMAOutOfOrderOptimizationOn4thTaskEvenWithMemOverlapWith1stTask
func.func @DMAOutOfOrderOptimizationOn4thTaskEvenWithMemOverlapWith1stTask() -> !type_DDR {

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %buf1 = VPURT.DeclareBuffer <DDR> <24576> -> !type_DDR
    %buf2 = VPURT.DeclareBuffer <DDR> <49152> -> !type_DDR
    %buf3 = VPURT.DeclareBuffer <DDR> <73728> -> !type_DDR
    %buf4 = VPURT.DeclareBuffer <DDR> <98304> -> !type_DDR
    %output = VPURT.DeclareBuffer <DDR> <122880> -> !type_DDR

    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !type_DDR) outputs(%buf0: !type_DDR) -> !type_DDR
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf1 : !type_DDR) outputs(%buf2: !type_DDR) -> !type_DDR
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf3 : !type_DDR) outputs(%buf4: !type_DDR) -> !type_DDR
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf0 : !type_DDR) outputs(%output: !type_DDR) -> !type_DDR
    }
    return %output : !type_DDR

    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-NOT:     is_out_of_order
    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-SAME:    is_out_of_order
    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-SAME:    is_out_of_order
    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-SAME:    is_out_of_order
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x64x64xf16, #NHWC, @DDR>

//CHECK-LABEL: @ParallelDMAOutOfOrderOptimization
func.func @ParallelDMAOutOfOrderOptimization() -> !type_DDR {

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %buf1 = VPURT.DeclareBuffer <DDR> <24576> -> !type_DDR
    %output = VPURT.DeclareBuffer <DDR> <49152> -> !type_DDR

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : !type_DDR) outputs(%buf0: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar0: !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf1 : !type_DDR) outputs(%output: !type_DDR) -> !type_DDR
    }
    return %output : !type_DDR

    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-NOT:     is_out_of_order
    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-SAME:     is_out_of_order
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!type_CMX_DUPLICATED = !VPUIP.DistributedBuffer<1x1x1x128xf16, #NCHW, @CMX_NN, {
  mode = "DUPLICATED",
  num_clusters = 2 : i64,
  uniform_distributed_segments
}>

//CHECK-LABEL: @NoDMAOutOfOrderOptimizationDueToMemOverlapWithDUPLICATEDBuffer
func.func @NoDMAOutOfOrderOptimizationDueToMemOverlapWithDUPLICATEDBuffer() -> memref<1x128xf16, [@CMX_NN, 0]> {
    %inbuf0 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x1x1x128xf32, [@CMX_NN, 0]>
    %outbuf0 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <1024> -> !type_CMX_DUPLICATED

    %inbuf1 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x128xf16, [@CMX_NN, 0]>
    %outbuf1 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x128xf16, [@CMX_NN, 0]>

    VPURT.Task {
      %0 = VPUIP.ConvertDMA {channelType = 1 : i64, port = 0 : i64} inputs(%inbuf0 : memref<1x1x1x128xf32, [@CMX_NN, 0]>) outputs(%outbuf0 : !type_CMX_DUPLICATED) -> !type_CMX_DUPLICATED
    }
    VPURT.Task {
      %0 = VPUIP.NNDMA {channelType = 1 : i64, port = 0 : i64} inputs(%inbuf1 : memref<1x128xf16, [@CMX_NN, 0]>) outputs(%outbuf1 : memref<1x128xf16, [@CMX_NN, 0]>) -> memref<1x128xf16, [@CMX_NN, 0]>
    }

    return %outbuf1 : memref<1x128xf16, [@CMX_NN, 0]>

    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.ConvertDMA
    // CHECK-NOT:     is_out_of_order
    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-NOT:     is_out_of_order
}

// -----

!type_DDR = memref<1x2x1x1640xf16, @DDR>
!type_CMX = memref<1x2x1x1640xf16, [@CMX_NN, 0]>
!indices_type_DDR = memref<1x2x1x1xi64, @DDR>
!indices_type_CMX = memref<1x2x1x1xi64, [@CMX_NN, 0]>

//CHECK-LABEL: @NoGatherDMAOutOfOrderOptimizationDueToIndicesMemOverlap
func.func @NoGatherDMAOutOfOrderOptimizationDueToIndicesMemOverlap() -> !type_CMX {
    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR
    %indices = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> !indices_type_DDR
    %indices_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !indices_type_CMX
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <10000> -> !type_CMX

    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%indices : !indices_type_DDR) outputs(%indices_cmx: !indices_type_CMX) -> !indices_type_CMX
    }
    VPURT.Task {
      %0 = VPUIP.GatherDMA {channelType = 0 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}
                inputs(%input : !type_DDR) indices(%indices_cmx : !indices_type_CMX)
                outputs(%output : !type_CMX) -> !type_CMX
    }
    return %output : !type_CMX

    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.NNDMA
    // CHECK-NOT:     is_out_of_order
    // CHECK:    VPURT.Task
    // CHECK-NEXT:    VPUIP.GatherDMA
    // CHECK-NOT:     is_out_of_order
}
