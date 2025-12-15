//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" --split-input-file --dispatched-inliner --move-declarations-to-top %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// foo1 -> foo2

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<1x3x64x64xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK-NOT: func.func private @foo1
    func.func private @foo1(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // original input
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // allocated by main
        %1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
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

    // CHECK-NOT: func.func private @foo2
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

    // CHECK-LABEL: @main
        // CHECK-DAG: [[CROSS_FUNC_BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO1_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO1_TMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK-DAG: [[FOO1_BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[FOO2_IN:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO2_TMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK-DAG: [[FOO2_BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[FOO1_BARR]] : !VPURT.Barrier)
        // CHECK:     VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[FOO1_TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        // CHECK: VPURT.Task waits([[FOO1_BARR]] : !VPURT.Barrier) updates([[CROSS_FUNC_BARR]] : !VPURT.Barrier)
        // CHECK:     VPUIP.NNDMA {port = 1 : i64} inputs([[FOO1_TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[FOO1_OUT]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task waits([[CROSS_FUNC_BARR]] : !VPURT.Barrier) updates([[FOO2_BARR]] : !VPURT.Barrier)
        // CHECK:     VPUIP.NNDMA {port = 0 : i64} inputs([[FOO2_IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[FOO2_TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        // CHECK: VPURT.Task waits([[FOO2_BARR]] : !VPURT.Barrier)
        // CHECK:     VPUIP.NNDMA {port = 1 : i64} inputs([[FOO2_TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: return {{[^:]+}} : memref<1x3x64x64xf16, @DDR>
}

// -----

// foo1 -> foo2 -> foo3

// CHECK-LABEL: @ThreeFunctions
module @ThreeFunctions {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<1x3x64x64xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK-NOT: func.func private @foo1
    func.func private @foo1(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // original input
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // allocated by main
        %1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
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

    // CHECK-NOT: func.func private @foo2
    func.func private @foo2(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // input from foo1 allocated by main
        %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // output allocated by main
        %1 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>

        VPURT.Task {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK-NOT: func.func private @foo3
    func.func private @foo3(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // input from foo1 allocated by main
        %0 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
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
        %3 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%4 : !VPURT.Barrier) {
            %6 = func.call @foo1(%0, %2) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
            %6 = func.call @foo2(%2, %3) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%5 : !VPURT.Barrier) {
            %6 = func.call @foo3(%3, %1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK-LABEL: @main
        // CHECK-DAG: [[FOO1_TO_FOO2_BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[FOO2_TO_FOO3_BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT_FOO1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[TMP_FOO1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK-DAG: [[FOO1_BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK-DAG: [[IN_FOO2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT_FOO2:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>

        // CHECK-DAG: [[IN_FOO3:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[TMP_FOO3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK-DAG: [[FOO3_BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[FOO1_BARR]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP_FOO1]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>)

        // CHECK: VPURT.Task waits([[FOO1_BARR]] : !VPURT.Barrier) updates([[FOO1_TO_FOO2_BAR]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP_FOO1]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[OUT_FOO1]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: VPURT.Task waits([[FOO1_TO_FOO2_BAR]] : !VPURT.Barrier) updates([[FOO2_TO_FOO3_BAR]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN_FOO2]] : memref<1x3x64x64xf16, @DDR>) outputs([[OUT_FOO2]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: VPURT.Task waits([[FOO2_TO_FOO3_BAR]] : !VPURT.Barrier) updates([[FOO3_BARR]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN_FOO3]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP_FOO3]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>)

        // CHECK: VPURT.Task waits([[FOO3_BARR]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP_FOO3]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return {{[^:]+}} : memref<1x3x64x64xf16, @DDR>
}

// -----

// op -> .. -> op ->
//                   foo
// op -> .. -> op ->

// CHECK-LABEL: @ThreeFunctions
module @ThreeFunctions {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<1x3x64x64xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x6x64x64xf16>
    }

    // CHECK-NOT: func.func private @foo
    func.func private @foo(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>, %arg2: memref<1x6x64x64xf16, @DDR>) -> memref<1x6x64x64xf16, @DDR> {
        // 1st input allocated by main
        %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // 2nd input allocated by main
        %1 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // aliases for output
        %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        %3 = VPURT.DeclareBuffer <NetworkOutput> [0] <24576> -> memref<1x3x64x64xf16, @DDR>

        VPURT.Task {
            %5 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%2 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task {
            %5 = VPUIP.NNDMA {port = 1 : i64} inputs(%1 : memref<1x3x64x64xf16, @DDR>) outputs(%3 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg2 : memref<1x6x64x64xf16, @DDR>
    }

    func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x6x64x64xf16, @DDR>) -> memref<1x6x64x64xf16, @DDR> {
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x6x64x64xf16, @DDR>
        %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        %3 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // tmp buffer #1
        %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // tmp buffer #2
        %5 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>

        %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // branch #1
        VPURT.Task updates(%6 : !VPURT.Barrier) {
            %10 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%4 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%6 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) {
            %10 = VPUIP.NNDMA {port = 1 : i64} inputs(%4 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }

        // branch #2
        VPURT.Task updates(%7 : !VPURT.Barrier) {
            %10 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%5 : memref<1x3x64x64xf16, [@CMX_NN, 1]>) -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        }
        VPURT.Task waits(%7 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) {
            %10 = VPUIP.NNDMA {port = 1 : i64} inputs(%5 : memref<1x3x64x64xf16, [@CMX_NN, 1]>) outputs(%3 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }

        VPURT.Task waits(%8, %9 : !VPURT.Barrier, !VPURT.Barrier) {
            %10 = func.call @foo(%2, %3, %1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>,memref<1x6x64x64xf16, @DDR>) -> memref<1x6x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x6x64x64xf16, @DDR>
    }

    // CHECK-LABEL: @main
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>

        // CHECK-DAG: [[TMP1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK-DAG: [[TMP2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>

        // CHECK-DAG: [[TMP_OUT1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[TMP_OUT2:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>

        // CHECK-DAG: [[TMP_BARR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[TMP_BARR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[BARR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[BARR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[TMP_OUT1_ALIAS:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[TMP_OUT2_ALIAS:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT2:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <24576> -> memref<1x3x64x64xf16, @DDR>

        // branch #1
        // CHECK: VPURT.Task updates([[TMP_BARR1]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP1]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>)

        // CHECK: VPURT.Task waits([[TMP_BARR1]] : !VPURT.Barrier) updates([[BARR1]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP1]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[TMP_OUT1]] : memref<1x3x64x64xf16, @DDR>)

        // branch #2
        // CHECK: VPURT.Task updates([[TMP_BARR2]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP2]] : memref<1x3x64x64xf16, [@CMX_NN, 1]>)

        // CHECK: VPURT.Task waits([[TMP_BARR2]] : !VPURT.Barrier) updates([[BARR2]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP2]] : memref<1x3x64x64xf16, [@CMX_NN, 1]>) outputs([[TMP_OUT2]] : memref<1x3x64x64xf16, @DDR>)

        // foo
        // CHECK: VPURT.Task waits([[BARR1]], [[BARR2]] : !VPURT.Barrier, !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[TMP_OUT1_ALIAS]] : memref<1x3x64x64xf16, @DDR>) outputs([[OUT1]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BARR1]], [[BARR2]] : !VPURT.Barrier, !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP_OUT2_ALIAS]] : memref<1x3x64x64xf16, @DDR>) outputs([[OUT2]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return {{[^:]+}} : memref<1x6x64x64xf16, @DDR>
}

// -----

//      -> op -> .. -> op
// foo1
//      -> op -> .. -> op

// CHECK-LABEL: @ThreeFunctions
module @ThreeFunctions {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<1x6x64x64xf16>
    } outputsInfo : {
      DataInfo "output1" : tensor<1x3x64x64xf16>
      DataInfo "output2" : tensor<1x3x64x64xf16>
    }

    // CHECK-NOT: func.func private @foo1
    func.func private @foo1(%arg0: memref<1x6x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>, %arg2: memref<1x3x64x64xf16, @DDR>)
                                                                                        -> (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) {
        // original input
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x6x64x64xf16, @DDR>
        // output allocated by main
        %1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // output allocated by main
        %2 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // aliases for input
        %3 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        %4 = VPURT.DeclareBuffer <NetworkInput> [0] <24576> -> memref<1x3x64x64xf16, @DDR>

        VPURT.Task {
            %5 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x64x64xf16, @DDR>) outputs(%1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task {
            %5 = VPUIP.NNDMA {port = 1 : i64} inputs(%4 : memref<1x3x64x64xf16, @DDR>) outputs(%2 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1, %arg2 : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
    }

    func.func @main(%arg0: memref<1x6x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>, %arg2: memref<1x3x64x64xf16, @DDR>) -> (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) {
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x6x64x64xf16, @DDR>
        %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        %2 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x3x64x64xf16, @DDR>
        %3 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        %4 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%5, %6 : !VPURT.Barrier, !VPURT.Barrier) {
            %7:2 = func.call @foo1(%0, %3, %4) : (memref<1x6x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>)
        }
        VPURT.Task waits(%5 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x64x64xf16, @DDR>) outputs(%1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%6 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 1 : i64} inputs(%4 : memref<1x3x64x64xf16, @DDR>) outputs(%2 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1, %arg2 : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
    }

    // CHECK-LABEL: @main
        // CHECK-DAG: [[OUT1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT2:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO_OUT1_ALIAS:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO_OUT2_ALIAS:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO_BARR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[FOO_BARR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[FOO_OUT1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO_OUT2:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[IN1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[IN2:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <24576> -> memref<1x3x64x64xf16, @DDR>

        // foo
        // CHECK: VPURT.Task updates([[FOO_BARR1]], [[FOO_BARR2]] : !VPURT.Barrier, !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN1]] : memref<1x3x64x64xf16, @DDR>) outputs([[FOO_OUT1]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: VPURT.Task updates([[FOO_BARR1]], [[FOO_BARR2]] : !VPURT.Barrier, !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[IN2]] : memref<1x3x64x64xf16, @DDR>) outputs([[FOO_OUT2]] : memref<1x3x64x64xf16, @DDR>)

        // branch #1
        // CHECK: VPURT.Task waits([[FOO_BARR1]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[FOO_OUT1_ALIAS]] : memref<1x3x64x64xf16, @DDR>) outputs([[OUT1]] : memref<1x3x64x64xf16, @DDR>)

        // branch #2
        // CHECK: VPURT.Task waits([[FOO_BARR2]] : !VPURT.Barrier)
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[FOO_OUT2_ALIAS]] : memref<1x3x64x64xf16, @DDR>) outputs([[OUT2]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return {{[^:]+}}, {{[^:]+}} : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
}

// -----

// foo1 -> foo2

// CHECK-LABEL: @TwoFunctionsEachWithSingleExecutionQueue
module @TwoFunctionsEachWithSingleExecutionQueue {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<1x3x64x64xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK-NOT: func.func private @foo1
    func.func private @foo1(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // original input
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // allocated by main
        %1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // tmp buffer
        %2 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>

        VPURT.Task {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%2 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x3x64x64xf16, @DDR>) outputs(%1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK-NOT: func.func private @foo2
    func.func private @foo2(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // input from foo1 allocated by main
        %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // original output
        %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // tmp buffer
        %2 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>

        VPURT.Task {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%2 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x3x64x64xf16, @DDR>) outputs(%1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
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

    // CHECK-LABEL: @main
        // CHECK-DAG: [[CROSS_FUNC_BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO1_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO1_TMP:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO2_IN:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[FOO2_TMP:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task
        // CHECK:     VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[FOO1_TMP]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task updates([[CROSS_FUNC_BARR]] : !VPURT.Barrier)
        // CHECK:     VPUIP.NNDMA {port = 0 : i64} inputs([[FOO1_TMP]] : memref<1x3x64x64xf16, @DDR>) outputs([[FOO1_OUT]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task waits([[CROSS_FUNC_BARR]] : !VPURT.Barrier)
        // CHECK:     VPUIP.NNDMA {port = 0 : i64} inputs([[FOO2_IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[FOO2_TMP]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task
        // CHECK:     VPUIP.NNDMA {port = 0 : i64} inputs([[FOO2_TMP]] : memref<1x3x64x64xf16, @DDR>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: return {{[^:]+}} : memref<1x3x64x64xf16, @DDR>
}

//
// -----
//

// CHECK-LABEL: @MultiShaveFunction
module @MultiShaveFunction {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<1x1x2x64xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x1x2x64xf16>
    }

    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64)
            attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    // CHECK-NOT: func.func private @foo1
    func.func private @foo(%arg0: memref<1x1x2x64xf16, @DDR>, %arg1: memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR> {
        %0 = VPURT.DeclareBuffer <DDR> <4096> -> memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
        %1 = VPURT.DeclareBuffer <DDR> <4352> -> memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
        %2 = VPURT.DeclareBuffer <DDR> <4224> -> memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
        %3 = VPURT.DeclareBuffer <DDR> <4480> -> memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
        VPURT.Task {
            %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%0 as %arg2: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>) outputs(%1 as %arg3: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>) on tile 0 -> memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>{
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3) : memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>, memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
            }
        }
        VPURT.Task {
            %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%2 as %arg2: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>) outputs(%3 as %arg3: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>) on tile 0 -> memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>{
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3) : memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>, memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
            }
        }
        return %arg1 : memref<1x1x2x64xf16, @DDR>
    }

    // CHECK-LABEL: @main
    func.func @main(%arg0: memref<1x1x2x64xf16, @DDR>, %arg1: memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR> {
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x2x64xf16, @DDR>
        %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x2x64xf16, @DDR>
        %2 = VPURT.DeclareBuffer <DDR> <4096> -> memref<1x1x2x64xf16, @DDR>
        %3 = VPURT.DeclareBuffer <DDR> <4352> -> memref<1x1x2x64xf16, @DDR>
        %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%4 : !VPURT.Barrier) {
            %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x1x2x64xf16, @DDR>) outputs(%2 : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
        }
        VPURT.Task waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
            %8 = func.call @foo(%2, %3) : (memref<1x1x2x64xf16, @DDR>, memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
        }
        VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
            %8 = VPUIP.NNDMA {port = 1 : i64} inputs(%3 : memref<1x1x2x64xf16, @DDR>) outputs(%2 : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
        }
        return %arg1 : memref<1x1x2x64xf16, @DDR>

        // CHECK-DAG: [[FOO1_WAIT:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[FOO1_UPDATE:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[POST_DMA2_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[FOO1_WAIT]] : !VPURT.Barrier)
        // CHECK-NEXT:  VPUIP.NNDMA {port = 0 : i64}

        // CHECK: VPURT.Task waits([[FOO1_WAIT]] : !VPURT.Barrier) updates([[FOO1_UPDATE]] : !VPURT.Barrier)
        // CHECK-NEXT:  VPUIP.SW.Kernel

        // CHECK: VPURT.Task waits([[FOO1_WAIT]] : !VPURT.Barrier) updates([[FOO1_UPDATE]] : !VPURT.Barrier)
        // CHECK-NEXT:  VPUIP.SW.Kernel

        // CHECK: VPURT.Task waits([[FOO1_UPDATE]] : !VPURT.Barrier) updates([[POST_DMA2_BARRIER]] : !VPURT.Barrier)
        // CHECK-NEXT:  VPUIP.NNDMA {port = 1 : i64}

    }
}
