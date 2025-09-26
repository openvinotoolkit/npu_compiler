//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --order-barriers-for-wlm %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OrderBarriers() -> memref<1x16x8x32xf16,  #NHWC, [@CMX_NN, 0]> {
    // Input order follows barrier production order
    %bar0 = VPURT.ConfigureBarrier<0> <{isStartBarrier}> -> !VPURT.Barrier
    %bar1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %bar2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %bar3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
    %bar4 = VPURT.ConfigureBarrier<4> <{isFinalBarrier}> -> !VPURT.Barrier

    // dummy buffer
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>

    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // Simple subgraph with dummy ops:
    //              DMA0
    //               |
    //            bar0[0]
    //             /   \
    //          DMA1     DPU0
    //           |       |
    //        bar1[1]    |
    //           |       |
    //          DMA2     |
    //           |       |
    //        bar3[3]    |
    //           |       |
    //          DMA3     |
    //            \     /
    //             bar2[2]
    //               |
    //              DMA4
    //               |
    //             bar4[4]

    VPURT.Task updates(%bar0: !VPURT.Barrier)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier)
               updates(%bar1: !VPURT.Barrier)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier)
               updates(%bar2: !VPURT.Barrier)
    {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%buf2: memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }

    VPURT.Task waits(%bar1: !VPURT.Barrier)
               updates(%bar3: !VPURT.Barrier)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier)
               updates(%bar2: !VPURT.Barrier)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier)
               updates(%bar4: !VPURT.Barrier)
    {
         VPUIP.NNDMA inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>

    // Result order follows barrier consumption order

    // CHECK:   VPURT.ConfigureBarrier<0> <{isStartBarrier}> -> !VPURT.Barrier
    // CHECK:   VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK:   VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
    // CHECK:   VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    // CHECK:   VPURT.ConfigureBarrier<4> <{isFinalBarrier}> -> !VPURT.Barrier
}
