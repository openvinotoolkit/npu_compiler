//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --assign-physical-barriers="num-barriers=33 color-bin-enable=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ColorbinAssignment
func.func @ColorbinAssignment() -> memref<1x16x8x32xf16,  #NHWC, [@CMX_NN, 0]> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar11 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar12 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar13 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar14 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar15 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar16 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar17 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar18 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar19 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar20 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar21 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar22 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar23 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar24 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar25 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar26 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar27 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar28 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar29 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar30 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar31 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar32 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar33 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffer
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %buf3 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>

    // Simple subgraph with dummy ops:
    //              DMA
    //               |
    //            bar[0-3]
    //               |
    //              DPU
    //               |
    //            bar[4-7]
    //               |
    //              DMA
    //               |
    //            bar[8-11]
    //               |
    //              DPU
    //               |
    //            bar[12-15]
    //               |
    //              DMA
    //               |
    //            bar[16-19]
    //               |
    //              DPU
    //               |
    //            bar[20-23]
    //               |
    //              DMA
    //               |
    //            bar[24-27]
    //               |
    //              DPU
    //               |
    //            bar[28-31]
    //               |
    //              DPU
    //               |
    //             bar32
    //               |
    //              DMA
    //               |
    //             bar33
    //               |
    //              DMA

    // Step 1: During the assignment, 34 virtual barriers will set bin type according to its consumer task type:
    // Virtual barriers with DMA bin type: [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 32, 33]
    // Virtual barriers with DPU bin type: [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]

    // Step 2: 33 physical barriers will be in two bins firstly:
    //  DMA bin:  0 - 16
    //  DPU bin: 17 - 32

    // Step 3: Assign physical barrier from the related bin

    // CHECK: VPURT.ConfigureBarrier<0>
    // CHECK: VPURT.ConfigureBarrier<1>
    // CHECK: VPURT.ConfigureBarrier<2>
    // CHECK: VPURT.ConfigureBarrier<3>
    // CHECK: VPURT.ConfigureBarrier<4>
    // CHECK: VPURT.ConfigureBarrier<5>
    // CHECK: VPURT.ConfigureBarrier<6>
    // CHECK: VPURT.ConfigureBarrier<7>
    // CHECK: VPURT.ConfigureBarrier<8>
    // CHECK: VPURT.ConfigureBarrier<9>
    // CHECK: VPURT.ConfigureBarrier<10>
    // CHECK: VPURT.ConfigureBarrier<11>
    // CHECK: VPURT.ConfigureBarrier<12>
    // CHECK: VPURT.ConfigureBarrier<13>
    // CHECK: VPURT.ConfigureBarrier<14>
    // CHECK: VPURT.ConfigureBarrier<15>
    // CHECK: VPURT.ConfigureBarrier<16>
    // CHECK: VPURT.ConfigureBarrier<17>
    // CHECK: VPURT.ConfigureBarrier<18>
    // CHECK: VPURT.ConfigureBarrier<19>
    // CHECK: VPURT.ConfigureBarrier<20>
    // CHECK: VPURT.ConfigureBarrier<21>
    // CHECK: VPURT.ConfigureBarrier<22>
    // CHECK: VPURT.ConfigureBarrier<23>
    // CHECK: VPURT.ConfigureBarrier<24>
    // CHECK: VPURT.ConfigureBarrier<25>
    // CHECK: VPURT.ConfigureBarrier<26>
    // CHECK: VPURT.ConfigureBarrier<27>
    // CHECK: VPURT.ConfigureBarrier<28>
    // CHECK: VPURT.ConfigureBarrier<29>
    // CHECK: VPURT.ConfigureBarrier<30>
    // CHECK: VPURT.ConfigureBarrier<31>
    // CHECK: VPURT.ConfigureBarrier<32>
    // CHECK: VPURT.ConfigureBarrier<0>

    VPURT.Task updates(%bar0, %bar1, %bar2, %bar3: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar0, %bar1, %bar2, %bar3: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
               updates(%bar4, %bar5, %bar6, %bar7: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
    {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}
    }
    VPURT.Task waits(%bar4, %bar5, %bar6, %bar7: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
               updates(%bar8, %bar9, %bar10, %bar11: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }


    VPURT.Task waits(%bar8, %bar9, %bar10, %bar11: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
               updates(%bar12, %bar13, %bar14, %bar15: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
    {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}
    }
    VPURT.Task waits(%bar12, %bar13, %bar14, %bar15: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
               updates(%bar16, %bar17, %bar18, %bar19: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
    {

        VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar16, %bar17, %bar18, %bar19: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
               updates(%bar20, %bar21, %bar22, %bar23: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
    {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}

    }
    VPURT.Task waits(%bar20, %bar21, %bar22, %bar23: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
               updates(%bar24, %bar25, %bar26, %bar27: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar24, %bar25, %bar26, %bar27: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
               updates(%bar28, %bar29, %bar30, %bar31: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
    {
         VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%buf0: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {DPUTask {outStart = [0, 0, 0], outEnd = [31, 7, 15], pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>}} PPE : {}
    }
    VPURT.Task waits(%bar28, %bar29, %bar30, %bar31: !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier)
               updates(%bar32: !VPURT.Barrier)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar32: !VPURT.Barrier)
               updates(%bar33: !VPURT.Barrier)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%bar33: !VPURT.Barrier)
    {
         VPUIP.NNDMA {port = 0 : i64} inputs(%cst0: memref<16x16x1x1xf16, #NHWC>) outputs(%buf1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    return %buf3: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
}
