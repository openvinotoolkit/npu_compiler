// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB" --assign-physical-barriers="num-barriers=4" %s | FileCheck %s

// CHECK-LABEL: @LinearDMA
func @LinearDMA(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    // CHECK-NOT: VPURT.DeclareVirtualBarrier
    // CHECK: VPURT.ConfigureBarrier<0>
    %bar0 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %buf0 = VPURT.DeclareBuffer "DDR" <0> -> memref<10xf16>
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA
            inputs(
                %arg0 : memref<10xf16>
            ) outputs(
                %buf0 : memref<10xf16>
            ) -> memref<10xf16>
    }
    // CHECK-NOT: VPURT.DeclareVirtualBarrier
    // CHECK: VPURT.ConfigureBarrier<1>
    %bar1 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %buf1 = VPURT.DeclareBuffer "DDR" <2048> -> memref<10xf16>
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
        %1 = VPUIP.NNDMA
            inputs(
                %buf0 : memref<10xf16>
            ) outputs(
                %buf1 : memref<10xf16>
            ) -> memref<10xf16>
    }
    // CHECK-NOT: VPURT.DeclareVirtualBarrier
    // CHECK: VPURT.ConfigureBarrier<2>
    %bar2 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
        %2 = VPUIP.NNDMA
            inputs(
                %buf1 : memref<10xf16>
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
func @MultipleExecutors(%arg0: memref<1x16x32x32xf16>, %arg1: memref<1x16x32x32xf16>) -> memref<1x16x32x32xf16> {
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        #const.Content<dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %cst1 = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<1> : tensor<16x1x1x4xsi32>>

    // input buffers for SOH tiling
    %buf0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x32x32xf16, #NHWC>
    %buf1 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x8x32xf16, #NHWC>
    %buf2 = VPURT.DeclareBuffer "DDR" <8192> -> memref<1x16x8x32xf16, #NHWC>
    %buf3 = VPURT.DeclareBuffer "DDR" <16384> -> memref<1x16x8x32xf16, #NHWC>
    %buf4 = VPURT.DeclareBuffer "DDR" <24576> -> memref<1x16x8x32xf16, #NHWC>

    // output buffers for SOH tiling
    %buf5 = VPURT.DeclareBuffer "DDR" <32768> -> memref<1x16x32x32xf16, #NHWC>
    %buf6 = VPURT.DeclareBuffer "DDR" <32768> -> memref<1x16x8x32xf16, #NHWC>
    %buf7 = VPURT.DeclareBuffer "DDR" <40960> -> memref<1x16x8x32xf16, #NHWC>
    %buf8 = VPURT.DeclareBuffer "DDR" <49152> -> memref<1x16x8x32xf16, #NHWC>
    %buf9 = VPURT.DeclareBuffer "DDR" <57344> -> memref<1x16x8x32xf16, #NHWC>

    // CMX buffers (double-buffers)
    %buf10 = VPURT.DeclareBuffer "CMX_NN" <0> -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
    %buf11 = VPURT.DeclareBuffer "CMX_NN" <8192> -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
    %buf12 = VPURT.DeclareBuffer "CMX_NN" <16384> -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
    %buf13 = VPURT.DeclareBuffer "CMX_NN" <24576> -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
    %buf14 = VPURT.DeclareBuffer "CMX_NN" <32768> -> memref<16x16x1x1xf16, #NHWC, "CMX_NN">
    %buf15 = VPURT.DeclareBuffer "CMX_NN" <33280> -> memref<16x1x1x4xsi32, "CMX_NN">

    %bar0 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %bar6 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %bar7 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %bar8 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier
    %bar9 = VPURT.DeclareVirtualBarrier {id = 0 : i64} -> !VPURT.Barrier

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
         VPUIP.NNDMA
            inputs(%cst0: memref<16x16x1x1xf16, #NHWC>)
            outputs(%buf14: memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
            -> memref<16x16x1x1xf16, #NHWC, "CMX_NN">
    }

    VPURT.Task {
         VPUIP.NNDMA
            inputs(%cst1: memref<16x1x1x4xsi32>)
            outputs(%buf15: memref<16x1x1x4xsi32, "CMX_NN">)
            -> memref<16x1x1x4xsi32, "CMX_NN">
    }

    // Reorder input

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NHWC}
            inputs(%arg0: memref<1x16x32x32xf16>)
            outputs(%buf0: memref<1x16x32x32xf16, #NHWC>)
            -> memref<1x16x32x32xf16, #NHWC>
    }

    // Upload 1st input tile

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf1: memref<1x16x8x32xf16, #NHWC>)
            outputs(%buf10: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
    }

    // 1st tile

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%buf10: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
            weight_table(%buf15: memref<16x1x1x4xsi32, "CMX_NN">)
            parent_input(%buf10: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            parent_output(%buf11: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            outputs(%buf11: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
            variants : {
                DPUTask {
                    start = [0, 0, 0],
                    end = [31, 7, 15],
                    pad = {bottom = 0, left = 0, right = 0, top = 0},
                    mpe_mode = "VECTOR_FP16"
                }
            } PPE : {
            }
    }

    // Prefetch 2nd tile (in parallel)

    VPURT.Task updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf2: memref<1x16x8x32xf16, #NHWC>)
            outputs(%buf12: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
    }

    // Copyback 1st result tile

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf11: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            outputs(%buf6: memref<1x16x8x32xf16, #NHWC>)
            -> memref<1x16x8x32xf16, #NHWC>
    }

    // 2nd tile

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar5: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%buf12: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
            weight_table(%buf15: memref<16x1x1x4xsi32, "CMX_NN">)
            parent_input(%buf12: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            parent_output(%buf13: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            outputs(%buf13: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
            variants : {
                DPUTask {
                    start = [0, 0, 0],
                    end = [31, 7, 15],
                    pad = {bottom = 0, left = 0, right = 0, top = 0},
                    mpe_mode = "VECTOR_FP16"
                }
            } PPE : {
            }
    }

    // Prefetch 3rd tile (in parallel)

    VPURT.Task updates(%bar6: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf3: memref<1x16x8x32xf16, #NHWC>)
            outputs(%buf10: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
    }

    // Copyback 2nd result tile

    VPURT.Task waits(%bar5: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf13: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            outputs(%buf7: memref<1x16x8x32xf16, #NHWC>)
            -> memref<1x16x8x32xf16, #NHWC>
    }

    // 3rd tile

    VPURT.Task waits(%bar6: !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%buf10: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
            weight_table(%buf15: memref<16x1x1x4xsi32, "CMX_NN">)
            parent_input(%buf10: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            parent_output(%buf11: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            outputs(%buf11: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
            variants : {
                DPUTask {
                    start = [0, 0, 0],
                    end = [31, 7, 15],
                    pad = {bottom = 0, left = 0, right = 0, top = 0},
                    mpe_mode = "VECTOR_FP16"
                }
            } PPE : {
            }
    }

    // Prefetch 4th tile (in parallel)

    VPURT.Task updates(%bar8: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf4: memref<1x16x8x32xf16, #NHWC>)
            outputs(%buf12: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
    }

    // Copyback 3rd result tile

    VPURT.Task waits(%bar7: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf11: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            outputs(%buf8: memref<1x16x8x32xf16, #NHWC>)
            -> memref<1x16x8x32xf16, #NHWC>
    }

    // 4th tile

    VPURT.Task waits(%bar8: !VPURT.Barrier) updates(%bar9: !VPURT.Barrier) {
        VPUIP.NCEClusterTask {
                kernel_padding = {bottom = 0, left = 0, right = 0, top = 0},
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = "CONV"
            }
            input(%buf12: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            weights(%buf14: memref<16x16x1x1xf16, #NHWC, "CMX_NN">)
            weight_table(%buf15: memref<16x1x1x4xsi32, "CMX_NN">)
            parent_input(%buf12: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            parent_output(%buf13: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            outputs(%buf13: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            -> memref<1x16x8x32xf16, #NHWC, "CMX_NN">
            variants : {
                DPUTask {
                    start = [0, 0, 0],
                    end = [31, 7, 15],
                    pad = {bottom = 0, left = 0, right = 0, top = 0},
                    mpe_mode = "VECTOR_FP16"
                }
            } PPE : {
            }
    }

    // Copyback 4th result tile

    VPURT.Task waits(%bar9: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf13: memref<1x16x8x32xf16, #NHWC, "CMX_NN">)
            outputs(%buf9: memref<1x16x8x32xf16, #NHWC>)
            -> memref<1x16x8x32xf16, #NHWC>
    }

    // Reorder output

    VPURT.Task waits(%bar4: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NCHW}
            inputs(%buf5: memref<1x16x32x32xf16, #NHWC>)
            outputs(%arg1: memref<1x16x32x32xf16>)
            -> memref<1x16x32x32xf16>
    }

    return %arg1 : memref<1x16x32x32xf16>
}
