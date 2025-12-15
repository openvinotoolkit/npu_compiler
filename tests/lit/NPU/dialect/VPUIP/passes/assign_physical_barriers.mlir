//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% workload-management-enable=false" --assign-physical-barriers="num-barriers=4" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @LinearDMA
func.func @LinearDMA(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    // CHECK-NOT: VPURT.DeclareVirtualBarrier
    // CHECK: VPURT.ConfigureBarrier<0>
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<10xf16, @DDR>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA {port = 0 : i64}
            inputs(
                %arg0 : memref<10xf16>
            ) outputs(
                %buf0 : memref<10xf16, @DDR>
            ) -> memref<10xf16, @DDR>
    }
    // CHECK-NOT: VPURT.DeclareVirtualBarrier
    // CHECK: VPURT.ConfigureBarrier<1>
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %buf1 = VPURT.DeclareBuffer <DDR> <2048> -> memref<10xf16, @DDR>
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        %1 = VPUIP.NNDMA {port = 0 : i64}
            inputs(
                %buf0 : memref<10xf16, @DDR>
            ) outputs(
                %buf1 : memref<10xf16, @DDR>
            ) -> memref<10xf16, @DDR>
    }
    // CHECK-NOT: VPURT.DeclareVirtualBarrier
    // CHECK: VPURT.ConfigureBarrier<2>
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
        %2 = VPUIP.NNDMA {port = 0 : i64}
            inputs(
                %buf1 : memref<10xf16, @DDR>
            ) outputs(
                %arg1 : memref<10xf16>
            ) -> memref<10xf16>
    }
    return %arg1 : memref<10xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultipleExecutors
func.func @MultipleExecutors(%arg0: memref<1x16x32x32xf16>, %arg1: memref<1x16x32x32xf16>) -> memref<1x16x32x32xf16> {
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // input buffers for SOH tiling
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x8x32xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <8192> -> memref<1x16x8x32xf16, #NHWC, @DDR>
    %buf3 = VPURT.DeclareBuffer <DDR> <16384> -> memref<1x16x8x32xf16, #NHWC, @DDR>
    %buf4 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x16x8x32xf16, #NHWC, @DDR>

    // output buffers for SOH tiling
    %buf5 = VPURT.DeclareBuffer <DDR> <32768> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    %buf6 = VPURT.DeclareBuffer <DDR> <32768> -> memref<1x16x8x32xf16, #NHWC, @DDR>
    %buf7 = VPURT.DeclareBuffer <DDR> <40960> -> memref<1x16x8x32xf16, #NHWC, @DDR>
    %buf8 = VPURT.DeclareBuffer <DDR> <49152> -> memref<1x16x8x32xf16, #NHWC, @DDR>
    %buf9 = VPURT.DeclareBuffer <DDR> <57344> -> memref<1x16x8x32xf16, #NHWC, @DDR>

    // CMX buffers (double-buffers)
    %buf10 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf11 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf12 = VPURT.DeclareBuffer <CMX_NN> [0] <16384> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf13 = VPURT.DeclareBuffer <CMX_NN> [0] <24576> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf14 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>

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
    // CHECK: VPURT.ConfigureBarrier<0>
    // CHECK: VPURT.ConfigureBarrier<1>
    // CHECK: VPURT.ConfigureBarrier<2>
    // CHECK: VPURT.ConfigureBarrier<3>
    // CHECK: VPURT.ConfigureBarrier<0>
    // CHECK: VPURT.ConfigureBarrier<1>
    // CHECK: VPURT.ConfigureBarrier<2>
    // CHECK: VPURT.ConfigureBarrier<3>
    // CHECK: VPURT.ConfigureBarrier<1>
    // CHECK: VPURT.ConfigureBarrier<2>

    // Upload weights and weights table

    VPURT.Task {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%cst0: memref<16x16x1x1xf16, #NHWC>)
            outputs(%buf14: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    // Reorder input

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64}
            inputs(%arg0: memref<1x16x32x32xf16>)
            outputs(%buf0: memref<1x16x32x32xf16, #NHWC, @DDR>)
            -> memref<1x16x32x32xf16, #NHWC, @DDR>
    }

    // Upload 1st input tile

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf1: memref<1x16x8x32xf16, #NHWC, @DDR>)
            outputs(%buf10: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    }

    // 1st tile

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf10: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%buf10: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf11: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf11: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [31, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>
                }
            } PPE : {
            }
    }

    // Prefetch 2nd tile (in parallel)

    VPURT.Task updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf2: memref<1x16x8x32xf16, #NHWC, @DDR>)
            outputs(%buf12: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    }

    // Copyback 1st result tile

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf11: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf6: memref<1x16x8x32xf16, #NHWC, @DDR>)
            -> memref<1x16x8x32xf16, #NHWC, @DDR>
    }

    // 2nd tile

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf12: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%buf12: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf13: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf13: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [31, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>
                }
            } PPE : {
            }
    }

    // Prefetch 3rd tile (in parallel)

    VPURT.Task updates(%bar6: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf3: memref<1x16x8x32xf16, #NHWC, @DDR>)
            outputs(%buf10: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    }

    // Copyback 2nd result tile

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf13: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf7: memref<1x16x8x32xf16, #NHWC, @DDR>)
            -> memref<1x16x8x32xf16, #NHWC, @DDR>
    }

    // 3rd tile

    VPURT.Task waits(%bar6: !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf10: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%buf10: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf11: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf11: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [31, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>
                }
            } PPE : {
            }
    }

    // Prefetch 4th tile (in parallel)

    VPURT.Task updates(%bar8: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf4: memref<1x16x8x32xf16, #NHWC, @DDR>)
            outputs(%buf12: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    }

    // Copyback 3rd result tile

    VPURT.Task waits(%bar7: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf11: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf8: memref<1x16x8x32xf16, #NHWC, @DDR>)
            -> memref<1x16x8x32xf16, #NHWC, @DDR>
    }

    // 4th tile

    VPURT.Task waits(%bar8: !VPURT.Barrier) updates(%bar9: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf12: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%buf12: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%buf13: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf13: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
            variants : {
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [31, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>
                }
            } PPE : {
            }
    }

    // Copyback 4th result tile

    VPURT.Task waits(%bar9: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64}
            inputs(%buf13: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf9: memref<1x16x8x32xf16, #NHWC, @DDR>)
            -> memref<1x16x8x32xf16, #NHWC, @DDR>
    }

    // Reorder output

    VPURT.Task waits(%bar4: !VPURT.Barrier) {
        VPUIP.PermuteDMA {mem_perm = #NCHW, port = 0 : i64}
            inputs(%buf5: memref<1x16x32x32xf16, #NHWC, @DDR>)
            outputs(%arg1: memref<1x16x32x32xf16>)
            -> memref<1x16x32x32xf16>
    }

    return %arg1 : memref<1x16x32x32xf16>
}
