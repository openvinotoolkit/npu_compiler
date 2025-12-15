//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --insert-sync-tasks %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// Note: 'idx' added since tasks can be reordered

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<1x3x64x64xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x3x64x64xf16>
    }

    VPURT.SW.Runtime entryPoint: @VPU.SW::@runtime stack_configuration: [4096, 4096, 4096, 4096]

    module @VPU.SW {
        func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE }
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func private @foo1(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // original input
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // allocated by main
        %1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // tmp buffer
        %2 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>

        VPURT.Task {
            VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                        @VPU.SW::@builtin_relu
                        inputs(%0 as %arg2: memref<1x3x64x64xf16, @DDR>)
                        outputs(%1 as %arg3: memref<1x3x64x64xf16, @DDR>)
                        on tile 0 -> memref<1x3x64x64xf16, @DDR> {
                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3)
                        : memref<1x3x64x64xf16, @DDR>
                        , memref<1x3x64x64xf16, @DDR>
            }
        }
        VPURT.Task {
            VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                        @VPU.SW::@builtin_relu
                        inputs(%0 as %arg2: memref<1x3x64x64xf16, @DDR>)
                        outputs(%2 as %arg3: memref<1x3x64x64xf16, @DDR>)
                        on tile 1 -> memref<1x3x64x64xf16, @DDR> {
                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3)
                        : memref<1x3x64x64xf16, @DDR>
                        , memref<1x3x64x64xf16, @DDR>
            }        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

    func.func private @foo2(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // input from foo1 allocated by main
        %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // original output
        %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // tmp buffer
        %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%3 : !VPURT.Barrier) {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%2 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%3 : !VPURT.Barrier) {
            %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%2 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

    func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%3 : !VPURT.Barrier) {
            %4 = func.call @foo1(%0, %2) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%3 : !VPURT.Barrier) {
            %4 = func.call @foo2(%2, %1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK: func.func private @foo1
    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier

    // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.SyncDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.SW.Kernel
    // CHECK-SAME:    on tile 0
    // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.SW.Kernel
    // CHECK-SAME:    on tile 1
    // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.SyncDMA {port = 0 : i64}

    // CHECK: func.func private @foo2
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier
    // CHECK: [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier

    // CHECK: VPURT.Task updates([[BAR3]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.SyncDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.NNDMA {port = 1 : i64}
    // CHECK: VPURT.Task waits([[BAR5]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.SyncDMA {port = 0 : i64}

    // CHECK: func.func @main
    // CHECK: [[BAR6:%.*]] = VPURT.DeclareVirtualBarrier
    // CHECK: [[BAR7:%.*]] = VPURT.DeclareVirtualBarrier
    // CHECK: [[BAR8:%.*]] = VPURT.DeclareVirtualBarrier

    // CHECK: VPURT.Task updates([[BAR6]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.SyncDMA {port = 0 : i64}
    // CHECK: VPURT.Task waits([[BAR6]] : !VPURT.Barrier) updates([[BAR7]] : !VPURT.Barrier)
    // CHECK-NEXT: func.call @foo1
    // CHECK: VPURT.Task waits([[BAR7]] : !VPURT.Barrier) updates([[BAR8]] : !VPURT.Barrier)
    // CHECK-NEXT: func.call @foo2
    // CHECK: VPURT.Task waits([[BAR8]] : !VPURT.Barrier)
    // CHECK-NEXT: VPUIP.SyncDMA {port = 0 : i64}
}
