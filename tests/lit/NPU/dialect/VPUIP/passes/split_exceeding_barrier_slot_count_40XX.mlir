//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-exceeding-barrier-slot-count="max-variant-count=8 max-variant-sum=4" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExceedingVariantSumByProducers
func.func @ExceedingVariantSumByProducers() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //   0 1 2 3
    //      |
    //     bar0
    //      |
    //      4

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    // active barrier consumer

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //    0 1    2 3
    //     |      |
    //    bar0   bar1
    //     |      |
    //     4      4

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)

    // CHECK: VPURT.Task updates([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR1]] : !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR0]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExceedingVariantSumByConsumers
func.func @ExceedingVariantSumByConsumers() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //        0
    //        |
    //       bar0
    //        |
    //     1 2 3 4

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    // multiple active barrier consumers

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //     0         0
    //     |         |
    //    bar0      bar1
    //     |         |
    //    1 2       3 4

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoChangeForGoodVariantSum
func.func @NoChangeForGoodVariantSum() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //    0 1
    //     |
    //    bar0
    //     |
    //    2 3

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    // multiple active barrier consumers

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //    0 1
    //     |
    //    bar0
    //     |
    //    2 3

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoChangeForGoodVariantSumProducersUnEqual
func.func @NoChangeForGoodVariantSumProducersUnEqual() -> memref<1x16x1x1xf16, #NHWC, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, #NHWC, @DDR>

    //   0 1 2
    //     |
    //    bar0
    //     |
    //     3

    // multiple active barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    // multiple active barrier consumers

    VPURT.Task waits(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, #NHWC, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, #NHWC, @DDR>)
            -> memref<1x16x1x1xf16, #NHWC, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, #NHWC, @DDR>

    //   0 1 2
    //     |
    //    bar0
    //     |
    //     3

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)

    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExceedingVariantSumByMultipleWorkloads
func.func @ExceedingVariantSumByMultipleWorkloads(%arg0: memref<1x16x8x32xf16, #NHWC>, %arg1: memref<1x16x8x32xf16, #NHWC>) -> memref<1x16x8x32xf16, #NHWC> {
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CMX buffers
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //      0 1 2 (DMA)
    //        |
    //       bar0
    //        |
    //        3 (DPU, 2 variants)
    //        |
    //       bar1
    //        |
    //        4 (DMA)

    // multiple active barrier producers
    // DPU task has 2 variants but only the first variant waits for bar0 hence 
    // no linearization is needed.

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%cst0: memref<16x16x1x1xf16, #NHWC>)
            outputs(%buf2: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%cst1: memref<16x1x1x4xsi32>)
            outputs(%buf3: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%arg0: memref<1x16x8x32xf16, #NHWC>)
            outputs(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    }

    // DPU task

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%buf2: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%buf3: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf1: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf1: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [31, 7, 7],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>
                }
                DPUTask {
                    outStart = [0, 0, 8],
                    outEnd = [31, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>
                }
            } PPE : {
            }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf1: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%arg1: memref<1x16x8x32xf16, #NHWC>)
            -> memref<1x16x8x32xf16, #NHWC>
    }

    return %arg1 : memref<1x16x8x32xf16, #NHWC>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)

    // DPU Conv
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)

    // DMA
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
}
