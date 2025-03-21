//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-func-args-to-declarations --canonicalize --move-declarations-to-top %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
module @RepeatingBlocks {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x4x5x5xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x4x5x5xf16>
    }

    // CHECK: func.func @foo([[ARG0:%.+]]: memref<1x4x5x5xf16, @DDR>, [[ARG1:%.+]]: memref<1x4x5x5xf16, @DDR>)
    func.func @foo(%arg0: memref<1x4x5x5xf16, @DDR>, %arg1: memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR> {
        VPURT.Task {
            %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg0 as %arg2: memref<1x4x5x5xf16, @DDR>)
                                      outputs(%arg1 as %arg3: memref<1x4x5x5xf16, @DDR>) on tile 0 -> memref<1x4x5x5xf16, @DDR>{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3) : memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>
            }
        }
        return %arg1 : memref<1x4x5x5xf16, @DDR>

        //CHECK: [[TMP_IN:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x5x5xf16, @DDR>
        //CHECK: [[TMP_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <100> -> memref<1x4x5x5xf16, @DDR>

        //CHECK:    VPURT.Task {
        //CHECK:        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        //CHECK-SAME:                inputs([[TMP_IN]] as {{[^:]+}}: memref<1x4x5x5xf16, @DDR>)
        //CHECK-SAME:                outputs([[TMP_OUT]] as {{[^:]+}}: memref<1x4x5x5xf16, @DDR>) on tile 0

        //CHECK: return [[ARG1]] : memref<1x4x5x5xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x4x5x5xf16, @DDR>, [[ARG1:%.+]]: memref<1x4x5x5xf16, @DDR>)
    func.func @main(%arg0: memref<1x4x5x5xf16, @DDR>, %arg1: memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR> {
        %tmp_in = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x5x5xf16, @DDR>
        %tmp_out = VPURT.DeclareBuffer <DDR> <100> -> memref<1x4x5x5xf16, @DDR>

        %barr1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %barr2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %barr3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %barr4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // copy network input to temporary (tmp_in)buffer to comply with fixed offest for offset
        // TODO: #-122828 recalculate the offset at the inlining stage to get rid of this copy
        VPURT.Task updates(%barr1 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x4x5x5xf16, @DDR>) outputs(%tmp_in : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
        }
        VPURT.Task waits(%barr1 : !VPURT.Barrier) updates(%barr2 : !VPURT.Barrier) {
            %0 = func.call @foo(%tmp_in, %tmp_out) : (memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
        }

        // copy output from first call of repeating block
        // to temporary (tmp_in)buffer to comply with fixed offest for in/out buffers offset
        // TODO: #-122828 recalculate the offset at the inlining stage to get rid of this copy
        VPURT.Task waits(%barr2 : !VPURT.Barrier) updates(%barr3 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA inputs(%tmp_out : memref<1x4x5x5xf16, @DDR>) outputs(%tmp_in : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
        }
        VPURT.Task waits(%barr3 : !VPURT.Barrier) updates(%barr4 : !VPURT.Barrier) {
            %0 = func.call @foo(%tmp_in, %tmp_out) : (memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
        }

        VPURT.Task waits(%barr4 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA inputs(%tmp_out : memref<1x4x5x5xf16, @DDR>) outputs(%arg1 : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
        }
        return %arg1 : memref<1x4x5x5xf16, @DDR>

        //CHECK: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4x5x5xf16, @DDR>
        //CHECK: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x4x5x5xf16, @DDR>
        //CHECK: [[TMP_IN:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x5x5xf16, @DDR>
        //CHECK: [[TMP_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <100> -> memref<1x4x5x5xf16, @DDR>

        //CHECK:    VPURT.Task
        //CHECK:        VPUIP.NNDMA inputs([[IN]] : memref<1x4x5x5xf16, @DDR>) outputs([[TMP_IN]] : memref<1x4x5x5xf16, @DDR>)

        //CHECK:    VPURT.Task
        //CHECK:        func.call @foo([[TMP_IN]], [[TMP_OUT]]) : (memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>)

        //CHECK:    VPURT.Task
        //CHECK:        VPUIP.NNDMA inputs([[TMP_OUT]] : memref<1x4x5x5xf16, @DDR>) outputs([[TMP_IN]] : memref<1x4x5x5xf16, @DDR>)

        //CHECK:    VPURT.Task
        //CHECK:        func.call @foo([[TMP_IN]], [[TMP_OUT]]) : (memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>)

        //CHECK:    VPURT.Task
        //CHECK:        VPUIP.NNDMA inputs([[TMP_OUT]] : memref<1x4x5x5xf16, @DDR>) outputs([[OUT]] : memref<1x4x5x5xf16, @DDR>)

        //CHECK: return [[ARG1]] : memref<1x4x5x5xf16, @DDR>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @RepeatingBlocksWithViewOps {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x8x5x5xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x8x5x5xf16>
    }

    // CHECK: func.func @foo([[ARG0:%.+]]: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>, [[ARG1:%.+]]: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>, [[ARG2:%.+]]: memref<1x8x5x5xf16, @DDR>)
    func.func @foo(%arg0: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>,
                    %arg1: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>,
                     %arg2: memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR> {
        %out_part1 = VPUIP.SubView %arg2 [0, 0, 0, 0] [1, 4, 5, 5] : memref<1x8x5x5xf16, @DDR> to memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>
        %out_part2 = VPUIP.SubView %arg2 [0, 4, 0, 0] [1, 4, 5, 5] : memref<1x8x5x5xf16, @DDR> to memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>

        VPURT.Task {
            %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
                                inputs(%arg0 as %arg3: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>)
                                outputs(%out_part1 as %arg4: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>) on tile 0
                                    -> memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) :
                        memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>, memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>
            }
        }
        VPURT.Task {
            %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
                                inputs(%arg1 as %arg3: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>)
                                outputs(%out_part2 as %arg4: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>) on tile 0
                                    -> memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) :
                        memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>, memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>
            }
        }
        return %arg2 : memref<1x8x5x5xf16, @DDR>

        // CHECK: [[IN1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x8x5x5xf16, @DDR>
        // CHECK: [[IN2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x8x5x5xf16, @DDR>
        // CHECK: [[OUT:%.+]] = VPURT.DeclareBuffer <DDR> <200> -> memref<1x8x5x5xf16, @DDR>

        // CHECK: [[IN_PART1:%.+]] = VPUIP.SubView [[IN1]] [0, 0, 0, 0] [1, 4, 5, 5] : memref<1x8x5x5xf16, @DDR> to memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>
        // CHECK: [[IN_PART2:%.+]] = VPUIP.SubView [[IN2]] [0, 4, 0, 0] [1, 4, 5, 5] : memref<1x8x5x5xf16, @DDR> to memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>
        // CHECK: [[OUT_PART1:%.+]] = VPUIP.SubView [[OUT]] [0, 0, 0, 0] [1, 4, 5, 5] : memref<1x8x5x5xf16, @DDR> to memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>
        // CHECK: [[OUT_PART2:%.+]] = VPUIP.SubView [[OUT]] [0, 4, 0, 0] [1, 4, 5, 5] : memref<1x8x5x5xf16, @DDR> to memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>

        // CHECK:   VPURT.Task
        // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK-SAME:      inputs([[IN_PART1]] as {{[^:]+}}: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>)
        // CHECK-SAME:      outputs([[OUT_PART1]] as {{[^:]+}}: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>) on tile 0

        // CHECK:   VPURT.Task
        // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK:           inputs([[IN_PART2]] as {{[^:]+}}: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>)
        // CHECK:           outputs([[OUT_PART2]] as {{[^:]+}}: memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>) on tile 0

        // CHECK: return [[ARG2]] : memref<1x8x5x5xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x8x5x5xf16, @DDR>, [[ARG1:%.+]]: memref<1x8x5x5xf16, @DDR>)
    func.func @main(%arg0: memref<1x8x5x5xf16, @DDR>, %arg1: memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR> {
        %tmp_in_full = VPURT.DeclareBuffer <DDR> <0> -> memref<1x8x5x5xf16, @DDR>
        %tmp_out = VPURT.DeclareBuffer <DDR> <200> -> memref<1x8x5x5xf16, @DDR>

        %tmp_in_part1 = VPUIP.SubView %tmp_in_full [0, 0, 0, 0] [1, 4, 5, 5] : memref<1x8x5x5xf16, @DDR>
                            to memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>
        %tmp_in_part2 = VPUIP.SubView %tmp_in_full [0, 4, 0, 0] [1, 4, 5, 5] : memref<1x8x5x5xf16, @DDR>
                            to memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>

        %barr1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %barr2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %barr3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %barr4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // copy network input to temporary (tmp_in)buffer to comply with fixed offest for offset
        // TODO: #-122828 recalculate the offset at the inlining stage to get rid of this copy
        VPURT.Task updates(%barr1 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x8x5x5xf16, @DDR>) outputs(%tmp_in_full : memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR>
        }
        VPURT.Task waits(%barr1 : !VPURT.Barrier) updates(%barr2 : !VPURT.Barrier) {
            %0 = func.call @foo(%tmp_in_part1, %tmp_in_part2, %tmp_out) :
                    (memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>,
                        memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>, memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR>
        }

        // copy output from first call of repeating block
        // to temporary (tmp_in)buffer to comply with fixed offest for in/out buffers offset
        // TODO: #-122828 recalculate the offset at the inlining stage to get rid of this copy
        VPURT.Task waits(%barr2 : !VPURT.Barrier) updates(%barr3 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA inputs(%tmp_out : memref<1x8x5x5xf16, @DDR>) outputs(%tmp_in_full : memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR>
        }
        VPURT.Task waits(%barr3 : !VPURT.Barrier) updates(%barr4 : !VPURT.Barrier) {
            %0 = func.call @foo(%tmp_in_part1, %tmp_in_part2, %tmp_out) :
                    (memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>,
                        memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>, memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR>
        }

        VPURT.Task waits(%barr4 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA inputs(%tmp_out : memref<1x8x5x5xf16, @DDR>) outputs(%arg1 : memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR>
        }

        return %arg1 : memref<1x8x5x5xf16, @DDR>

        // CHECK: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x8x5x5xf16, @DDR>
        // CHECK: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x8x5x5xf16, @DDR>
        // CHECK: [[TMP_IN:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x8x5x5xf16, @DDR>
        // CHECK: [[TMP_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <200> -> memref<1x8x5x5xf16, @DDR>

        // CHECK: [[TMP_IN_1:%.+]] = VPUIP.SubView [[TMP_IN]] [0, 0, 0, 0] [1, 4, 5, 5] : memref<1x8x5x5xf16, @DDR> to memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>
        // CHECK: [[TMP_IN_2:%.+]] = VPUIP.SubView [[TMP_IN]] [0, 4, 0, 0] [1, 4, 5, 5] : memref<1x8x5x5xf16, @DDR> to memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>

        // CHECK:   VPURT.Task
        // CHECK:       VPUIP.NNDMA inputs([[IN]] : memref<1x8x5x5xf16, @DDR>) outputs([[TMP_IN]] : memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR>

        // CHECK:   VPURT.Task
        // CHECK:       func.call @foo([[TMP_IN_1]], [[TMP_IN_2]], [[TMP_OUT]]) : (memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>, memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>, memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR>

        // CHECK:   VPURT.Task
        // CHECK:       VPUIP.NNDMA inputs([[TMP_OUT]] : memref<1x8x5x5xf16, @DDR>) outputs([[TMP_IN]] : memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR>

        // CHECK:   VPURT.Task
        // CHECK:       func.call @foo([[TMP_IN_1]], [[TMP_IN_2]], [[TMP_OUT]]) : (memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>, memref<1x4x5x5xf16, {order = #NCHW, strides = [200, 25, 5, 1]}, @DDR>, memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR>

        // CHECK:   VPURT.Task
        // CHECK:       VPUIP.NNDMA inputs([[TMP_OUT]] : memref<1x8x5x5xf16, @DDR>) outputs([[OUT]] : memref<1x8x5x5xf16, @DDR>) -> memref<1x8x5x5xf16, @DDR>

        // CHECK: return [[ARG1]] : memref<1x8x5x5xf16, @DDR>
    }
}
