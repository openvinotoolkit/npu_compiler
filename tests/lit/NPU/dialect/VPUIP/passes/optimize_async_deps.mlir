//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-async-deps %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @LinearGraph
func.func @LinearGraph(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = memref.alloc() : memref<10xf16>
    %buf1 = memref.alloc() : memref<10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<10xf16>> {
        %1 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16>) -> memref<10xf16>
        async.yield %1 : memref<10xf16>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> {
        %2 = IERT.ReLU inputs(%1 : memref<10xf16>) outputs(%buf1 : memref<10xf16>) -> memref<10xf16>
        async.yield %2 : memref<10xf16>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> {
        %3 = IERT.ReLU inputs(%2 : memref<10xf16>) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %3 : memref<10xf16>
    }

    %3 = async.await %f3 : !async.value<memref<10xf16>>
    return %3 : memref<10xf16>

    // CHECK:       [[BUF0:%.+]] = memref.alloc() : memref<10xf16>
    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%arg[0-9]]]: !async.value<memref<10xf16>>)
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           IERT.ReLU inputs([[VAL1]] : memref<10xf16>) outputs([[BUF1]] : memref<10xf16>) -> memref<10xf16>

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          ([[F2]] as [[VAL2:%arg[0-9]]]: !async.value<memref<10xf16>>)
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           IERT.ReLU inputs([[VAL2]] : memref<10xf16>) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]] : !async.value<memref<10xf16>>
    // CHECK:       return [[VAL3]]
}

// -----

// CHECK-LABEL: @IndependentBranchesLinearSched
func.func @IndependentBranchesLinearSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf = memref.alloc() : memref<20xf16>

    %t0, %f0 = async.execute -> !async.value<memref<10xf16>> {
        %buf0 = VPUIP.SubView %buf[0][10] : memref<20xf16> to memref<10xf16>
        %0 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16>) -> memref<10xf16>
        async.yield %0 : memref<10xf16>
    }

    %t1, %f1 = async.execute [%t0] -> !async.value<memref<10xf16>> {
        %buf1 = VPUIP.SubView %buf[10][10] : memref<20xf16> to memref<10xf16>
        %1 = IERT.ReLU inputs(%arg1 : memref<10xf16>) outputs(%buf1 : memref<10xf16>) -> memref<10xf16>
        async.yield %1 : memref<10xf16>
    }

    %t3, %f3 = async.execute [%t0, %t1] (
                %f0 as %0 : !async.value<memref<10xf16>>,
                %f1 as %1 : !async.value<memref<10xf16>>
            ) -> !async.value<memref<20xf16>> {
        %2 = VPUIP.ConcatView inputs(%0, %1 : memref<10xf16>, memref<10xf16>) outputs(%buf : memref<20xf16>) -> memref<20xf16>
        %3 = VPUIP.Copy inputs(%2 : memref<20xf16>) outputs(%arg2 : memref<20xf16>) -> memref<20xf16>
        async.yield %3 : memref<20xf16>
    }

    %3 = async.await %f3 : !async.value<memref<20xf16>>
    return %3 : memref<20xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           {{%.+}} = IERT.ReLU inputs(%arg0 : memref<10xf16>)

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           {{%.+}} = IERT.ReLU inputs(%arg1 : memref<10xf16>)

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-NOT:           [[T0]]
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          [[F0]] as [[VAL0:%arg[0-9]]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          [[F1]] as [[VAL1:%arg[0-9]]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          -> !async.value<memref<20xf16>>
    // CHECK:           {{%.+}} = VPUIP.ConcatView
    // CHECK-SAME:          inputs([[VAL0]], [[VAL1]] : memref<10xf16>, memref<10xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]] : !async.value<memref<20xf16>>
    // CHECK:       return [[VAL3]]
}

// -----

// CHECK-LABEL: @IndependentBranchesParallelSched
func.func @IndependentBranchesParallelSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf = memref.alloc() : memref<20xf16>

    %t0, %f0 = async.execute -> !async.value<memref<10xf16>> {
        %buf0 = VPUIP.SubView %buf[0][10] : memref<20xf16> to memref<10xf16>
        %0 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16>) -> memref<10xf16>
        async.yield %0 : memref<10xf16>
    }

    %t1, %f1 = async.execute -> !async.value<memref<10xf16>> {
        %buf1 = VPUIP.SubView %buf[10][10] : memref<20xf16> to memref<10xf16>
        %1 = IERT.ReLU inputs(%arg1 : memref<10xf16>) outputs(%buf1 : memref<10xf16>) -> memref<10xf16>
        async.yield %1 : memref<10xf16>
    }

    %t3, %f3 = async.execute [%t0, %t1] (
                %f0 as %0 : !async.value<memref<10xf16>>,
                %f1 as %1 : !async.value<memref<10xf16>>
            ) -> !async.value<memref<20xf16>> {
        %2 = VPUIP.ConcatView inputs(%0, %1 : memref<10xf16>, memref<10xf16>) outputs(%buf : memref<20xf16>) -> memref<20xf16>
        %3 = VPUIP.Copy inputs(%2 : memref<20xf16>) outputs(%arg2 : memref<20xf16>) -> memref<20xf16>
        async.yield %3 : memref<20xf16>
    }

    %3 = async.await %f3 : !async.value<memref<20xf16>>
    return %3 : memref<20xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           {{%.+}} = IERT.ReLU inputs(%arg0 : memref<10xf16>)

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           {{%.+}} = IERT.ReLU inputs(%arg1 : memref<10xf16>)

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T0]], [[T1]]
    // CHECK-SAME:          [[F0]] as [[VAL0:%arg[0-9]]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          [[F1]] as [[VAL1:%arg[0-9]]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          -> !async.value<memref<20xf16>>
    // CHECK:           {{%.+}} = VPUIP.ConcatView
    // CHECK-SAME:          inputs([[VAL0]], [[VAL1]] : memref<10xf16>, memref<10xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]] : !async.value<memref<20xf16>>
    // CHECK:       return [[VAL3]]
}

// -----

// CHECK-LABEL: @TwoOutputs
func.func @TwoOutputs(%arg0: memref<2xf16>, %arg1: memref<2xf16>, %arg2: memref<2xf16>) -> (memref<2xf16>, memref<2xf16>) {
    %0 = const.Declare memref<2xf16> = dense<1.0> : tensor<2xf16>

    %t1, %f1 = async.execute -> !async.value<memref<2xf16>> {
        %buf1 = VPUIP.StaticAlloc<0> -> memref<2xf16>
        %1 = IERT.ReLU inputs(%arg0 : memref<2xf16>) outputs(%buf1 : memref<2xf16>) -> memref<2xf16>
        async.yield %1 : memref<2xf16>
    }

    %t2, %f2 = async.execute [%t1] -> !async.value<memref<2xf16>> {
        %buf2 = VPUIP.StaticAlloc<4> -> memref<2xf16>
        %2 = IERT.ReLU inputs(%0 : memref<2xf16>) outputs(%buf2 : memref<2xf16>) -> memref<2xf16>
        async.yield %2 : memref<2xf16>
    }

    %t3, %f3 = async.execute [%t1, %t2] (%f1 as %1 : !async.value<memref<2xf16>>) -> !async.value<memref<2xf16>> {
        %3 = VPUIP.Copy inputs(%1 : memref<2xf16>) outputs(%arg1 : memref<2xf16>) -> memref<2xf16>
        async.yield %3 : memref<2xf16>
    }

    %t4, %f4 = async.execute [%t2, %t3] (%f2 as %2 : !async.value<memref<2xf16>>) -> !async.value<memref<2xf16>> {
        %4 = VPUIP.Copy inputs(%2 : memref<2xf16>) outputs(%arg2 : memref<2xf16>) -> memref<2xf16>
        async.yield %4 : memref<2xf16>
    }

    %3 = async.await %f3 : !async.value<memref<2xf16>>
    %4 = async.await %f4 : !async.value<memref<2xf16>>

    return %3, %4 : memref<2xf16>, memref<2xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK:           {{%.+}} = VPUIP.StaticAlloc<0>

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK:           {{%.+}} = VPUIP.StaticAlloc<4>

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-NOT:           [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          [[F1]] as [[VAL1:%arg[0-9]]]: !async.value<memref<2xf16>>
    // CHECK:           {{%.+}} = VPUIP.Copy inputs([[VAL1]] : memref<2xf16>) outputs(%arg1 : memref<2xf16>)

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute
    // CHECK-NOT:           [[T1]]
    // CHECK-NOT:           [[T2]]
    // CHECK-SAME:          [[T3]]
    // CHECK-SAME:          [[F2]] as [[VAL2:%arg[0-9]]]: !async.value<memref<2xf16>>
    // CHECK:           {{%.+}} = VPUIP.Copy inputs([[VAL2]] : memref<2xf16>) outputs(%arg2 : memref<2xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       [[VAL4:%.+]] = async.await [[F4]]
    // CHECK:       return [[VAL3]], [[VAL4]]
}

// -----

// CHECK-LABEL: @DiamondGraph
func.func @DiamondGraph(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = memref.alloc() : memref<10xf16>
    %buf1 = memref.alloc() : memref<10xf16>
    %buf2 = memref.alloc() : memref<10xf16>
    %buf3 = memref.alloc() : memref<10xf16>

    %t0, %f0 = async.execute -> !async.value<memref<10xf16>> {
        %0 = IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16>) -> memref<10xf16>
        async.yield %0 : memref<10xf16>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> {
        %1 = IERT.Sigmoid inputs(%0 : memref<10xf16>) outputs(%buf1 : memref<10xf16>) -> memref<10xf16>
        async.yield %1 : memref<10xf16>
    }

    %t2, %f2 = async.execute [%t1] -> !async.value<memref<10xf16>> {
        %2 = IERT.Tanh inputs(%arg0 : memref<10xf16>) outputs(%buf2 : memref<10xf16>) -> memref<10xf16>
        async.yield %2 : memref<10xf16>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> {
        %3 = IERT.Exp inputs(%2 : memref<10xf16>) outputs(%buf3 : memref<10xf16>) -> memref<10xf16>
        async.yield %3 : memref<10xf16>
    }

    %t4, %f4 = async.execute [%t1, %t3] (
                %f1 as %1 : !async.value<memref<10xf16>>,
                %f3 as %3 : !async.value<memref<10xf16>>
            ) -> !async.value<memref<10xf16>> {
        %4 = IERT.Add inputs(%1 : memref<10xf16>, %3 : memref<10xf16>) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %4 : memref<10xf16>
    }

    %4 = async.await %f4 : !async.value<memref<10xf16>>
    return %4 : memref<10xf16>

    // CHECK:   [[BUF0:%.+]] = memref.alloc() : memref<10xf16>
    // CHECK:   [[BUF1:%.+]] = memref.alloc() : memref<10xf16>
    // CHECK:   [[BUF2:%.+]] = memref.alloc() : memref<10xf16>
    // CHECK:   [[BUF3:%.+]] = memref.alloc() : memref<10xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           IERT.ReLU inputs(%arg0 : memref<10xf16>) outputs([[BUF0]] : memref<10xf16>)

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          [[F0]] as [[VAL0:%.+]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           IERT.Sigmoid inputs([[VAL0]] : memref<10xf16>) outputs([[BUF1]] : memref<10xf16>)

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-NOT:           [[T0]]
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           IERT.Tanh inputs(%arg0 : memref<10xf16>) outputs([[BUF2]] : memref<10xf16>)

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-NOT:           [[T0]]
    // CHECK-NOT:           [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          [[F2]] as [[VAL2:%.+]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           IERT.Exp inputs([[VAL2]] : memref<10xf16>) outputs([[BUF3]] : memref<10xf16>)

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute
    // CHECK-NOT:           [[T0]]
    // CHECK-NOT:           [[T1]]
    // CHECK-NOT:           [[T2]]
    // CHECK-SAME:          [[T3]]
    // CHECK-SAME:          [[F1]] as [[VAL1:%.+]]: !async.value<memref<10xf16>>,
    // CHECK-SAME:          [[F3]] as [[VAL3:%.+]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           IERT.Add inputs([[VAL1]] : memref<10xf16>, [[VAL3]] : memref<10xf16>) outputs(%arg1 : memref<10xf16>)

    // CHECK:       [[VAL4:%.+]] = async.await [[F4]] : !async.value<memref<10xf16>>
    // CHECK:       return [[VAL4]] : memref<10xf16>
}
