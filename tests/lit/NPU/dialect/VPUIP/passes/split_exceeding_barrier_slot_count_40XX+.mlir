//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-exceeding-barrier-slot-count="max-variant-count=6" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExceedingVariantCount
func.func @ExceedingVariantCount(%arg0: memref<1x16x32x32xf16, #NHWC>, %arg1: memref<1x16x32x32xf16>) -> memref<1x16x32x32xf16> {
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // input buffers for SOH tiling
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x8x32xf16, #NHWC, @DDR>
    %buf2 = VPURT.DeclareBuffer <DDR> <8192> -> memref<1x16x8x32xf16, #NHWC, @DDR>

    // output buffers for SOH tiling
    %buf6 = VPURT.DeclareBuffer <DDR> <32768> -> memref<1x16x8x32xf16, #NHWC, @DDR>
    %buf7 = VPURT.DeclareBuffer <DDR> <40960> -> memref<1x16x8x32xf16, #NHWC, @DDR>

    // CMX buffers
    %buf10 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf11 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf12 = VPURT.DeclareBuffer <CMX_NN> [0] <16384> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf13 = VPURT.DeclareBuffer <CMX_NN> [0] <24576> -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    %buf14 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %buf15 = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>

    %buf16 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    %buf17 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>

    %buf18 = VPURT.DeclareBuffer <CMX_NN> [0] <51200> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    %buf19 = VPURT.DeclareBuffer <CMX_NN> [0] <53248> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // parallel barrier producers

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%cst0: memref<16x16x1x1xf16, #NHWC>)
            outputs(%buf14: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%cst1: memref<16x1x1x4xsi32>)
            outputs(%buf15: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA
            inputs(%arg0: memref<1x16x32x32xf16, #NHWC>)
            outputs(%buf0: memref<1x16x32x32xf16, #NHWC, @DDR>)
            -> memref<1x16x32x32xf16, #NHWC, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf1: memref<1x16x8x32xf16, #NHWC, @DDR>)
            outputs(%buf10: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    }

    // DPU task

    VPURT.Task waits(%bar0, %bar1: !VPURT.Barrier, !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf10: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%buf15: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
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

    // parallel barrier producers with DPU task

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp
            inputs(%buf16  as %arg2: memref<1x4x256x1xf16, [@CMX_NN, 0]>)
            outputs(%buf18 as %arg3: memref<1x4x256x1xf16, [@CMX_NN, 0]>) on tile 0
                -> memref<1x4x256x1xf16, [@CMX_NN, 0]>{
                    VPUIP.SW.Kernel.run
                        {attrs = [0]}(%arg2, %arg3) : memref<1x4x256x1xf16, [@CMX_NN, 0]>, memref<1x4x256x1xf16, [@CMX_NN, 0]>
            }
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_TanhOp
            inputs(%buf17  as %arg2: memref<1x4x256x1xf16, [@CMX_NN, 0]>)
            outputs(%buf19 as %arg3: memref<1x4x256x1xf16, [@CMX_NN, 0]>) on tile 1
                -> memref<1x4x256x1xf16, [@CMX_NN, 0]> {
                    VPUIP.SW.Kernel.run
                        {attrs = [0]}(%arg2, %arg3) : memref<1x4x256x1xf16, [@CMX_NN, 0]>, memref<1x4x256x1xf16, [@CMX_NN, 0]>
                    }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%cst0: memref<16x16x1x1xf16, #NHWC>)
            outputs(%buf14: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%cst1: memref<16x1x1x4xsi32>)
            outputs(%buf15: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
            -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf2: memref<1x16x8x32xf16, #NHWC, @DDR>)
            outputs(%buf12: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf11: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf6: memref<1x16x8x32xf16, #NHWC, @DDR>)
            -> memref<1x16x8x32xf16, #NHWC, @DDR>
    }

    // DPU task

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%buf12: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weight_table(%buf15: memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
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

    VPURT.Task waits(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf13: memref<1x16x8x32xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf7: memref<1x16x8x32xf16, #NHWC, @DDR>)
            -> memref<1x16x8x32xf16, #NHWC, @DDR>
    }

    return %arg1 : memref<1x16x32x32xf16>

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // New barriers introduced

    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR6:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR7:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR0]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK: VPURT.Task updates([[BAR0]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR2]], [[BAR3]] : !VPURT.Barrier, !VPURT.Barrier)

    // SW Tanh
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)

    // DMA
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)

    // DMA
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier)

    // DPU Conv
    // CHECK: VPURT.Task waits([[BAR0]], [[BAR2]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)

    // DMA
    // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)

    // DMA
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)

    // DPU
    // CHECK: VPURT.Task waits([[BAR4]], [[BAR5]], [[BAR6]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier)
    // CHECK: VPURT.Task waits([[BAR7]] : !VPURT.Barrier)
}
