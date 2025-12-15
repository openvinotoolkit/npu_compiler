//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-async-deps %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {
        VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE
    }
}

// CHECK-LABEL: @LinearGraph
func.func @LinearGraph(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = memref.alloc() : memref<10xf16>
    %buf1 = memref.alloc() : memref<10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<10xf16>> {
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%arg0 as %input: memref<10xf16>)
            outputs(%buf0 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<10xf16>, memref<10xf16>
        }
        async.yield %1 : memref<10xf16>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%1 as %input: memref<10xf16>)
            outputs(%buf1 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<10xf16>, memref<10xf16>
        }
        async.yield %2 : memref<10xf16>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> {
        %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%2 as %input: memref<10xf16>)
            outputs(%arg1 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<10xf16>, memref<10xf16>
        }
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
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:      inputs([[VAL1]]
    // CHECK-SAME:      outputs([[BUF1]]

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          ([[F2]] as [[VAL2:%arg[0-9]]]: !async.value<memref<10xf16>>)
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:      inputs([[VAL2]]
    // CHECK-SAME:      outputs(%arg1


    // CHECK:       [[VAL3:%.+]] = async.await [[F3]] : !async.value<memref<10xf16>>
    // CHECK:       return [[VAL3]]
}

// -----

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {
        VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE
    }
}

// CHECK-LABEL: @IndependentBranchesLinearSched
func.func @IndependentBranchesLinearSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf = memref.alloc() : memref<20xf16>

    %t0, %f0 = async.execute -> !async.value<memref<10xf16>> {
        %buf0 = VPUIP.SubView %buf[0][10] : memref<20xf16> to memref<10xf16>
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%arg0 as %input: memref<10xf16>)
            outputs(%buf0 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<10xf16>, memref<10xf16>
        }
        async.yield %0 : memref<10xf16>
    }

    %t1, %f1 = async.execute [%t0] -> !async.value<memref<10xf16>> {
        %buf1 = VPUIP.SubView %buf[10][10] : memref<20xf16> to memref<10xf16>
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%arg1 as %input: memref<10xf16>)
            outputs(%buf1 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<10xf16>, memref<10xf16>
        }
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
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:      inputs(%arg0

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:      inputs(%arg1

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

module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {
        VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE
    }
}

// CHECK-LABEL: @IndependentBranchesParallelSched
func.func @IndependentBranchesParallelSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf = memref.alloc() : memref<20xf16>

    %t0, %f0 = async.execute -> !async.value<memref<10xf16>> {
        %buf0 = VPUIP.SubView %buf[0][10] : memref<20xf16> to memref<10xf16>
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%arg0 as %input: memref<10xf16>)
            outputs(%buf0 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<10xf16>, memref<10xf16>
        }
        async.yield %0 : memref<10xf16>
    }

    %t1, %f1 = async.execute -> !async.value<memref<10xf16>> {
        %buf1 = VPUIP.SubView %buf[10][10] : memref<20xf16> to memref<10xf16>
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%arg1 as %input: memref<10xf16>)
            outputs(%buf1 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<10xf16>, memref<10xf16>
        }
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
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:      inputs(%arg0

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:      inputs(%arg1

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

module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {
        VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE
    }
}

// CHECK-LABEL: @TwoOutputs
func.func @TwoOutputs(%arg0: memref<2xf16>, %arg1: memref<2xf16>, %arg2: memref<2xf16>) -> (memref<2xf16>, memref<2xf16>) {
    %0 = const.Declare memref<2xf16> = dense<1.0> : tensor<2xf16>

    %t1, %f1 = async.execute -> !async.value<memref<2xf16>> {
        %buf1 = VPUIP.StaticAlloc<0> -> memref<2xf16>
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%arg0 as %input: memref<2xf16>)
            outputs(%buf1 as %output: memref<2xf16>) on tile 0 -> memref<2xf16> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<2xf16>, memref<2xf16>
        }
        async.yield %1 : memref<2xf16>
    }

    %t2, %f2 = async.execute [%t1] -> !async.value<memref<2xf16>> {
        %buf2 = VPUIP.StaticAlloc<4> -> memref<2xf16>
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%0 as %input: memref<2xf16>)
            outputs(%buf2 as %output: memref<2xf16>) on tile 0 -> memref<2xf16> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<2xf16>, memref<2xf16>
        }
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

module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE}
    func.func private @builtin_sigmoid(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE}
    func.func private @builtin_Tanh(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
    func.func private @builtin_exp(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_exp.cpp", VPU.kernel_entry = "activation_exp"}
    func.func private @builtin_Add(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_add.cpp", VPU.kernel_entry = "eltwise_add", VPU.task_type = @COMPUTE}
}

// CHECK-LABEL: @DiamondGraph
func.func @DiamondGraph(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = memref.alloc() : memref<10xf16>
    %buf1 = memref.alloc() : memref<10xf16>
    %buf2 = memref.alloc() : memref<10xf16>
    %buf3 = memref.alloc() : memref<10xf16>

    %t0, %f0 = async.execute -> !async.value<memref<10xf16>> {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%arg0 as %input: memref<10xf16>)
            outputs(%buf0 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<10xf16>, memref<10xf16>
        }
        async.yield %0 : memref<10xf16>
    }

    %t1, %f1 = async.execute [%t0] (%f0 as %0 : !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> {
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_sigmoid
            inputs(%0 as %input: memref<10xf16>)
            outputs(%buf1 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run (%input, %output) : memref<10xf16>, memref<10xf16>
        }
        async.yield %1 : memref<10xf16>
    }

    %t2, %f2 = async.execute [%t1] -> !async.value<memref<10xf16>> {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_tanh
            inputs(%arg0 as %input: memref<10xf16>)
            outputs(%buf2 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run (%input, %output) : memref<10xf16>, memref<10xf16>
        }
        async.yield %2 : memref<10xf16>
    }

    %t3, %f3 = async.execute [%t2] (%f2 as %2 : !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> {
        %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_exp
            inputs(%2 as %input: memref<10xf16>)
            outputs(%buf3 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run (%input, %output) : memref<10xf16>, memref<10xf16>
        }
        async.yield %3 : memref<10xf16>
    }

    %t4, %f4 = async.execute [%t1, %t3] (
                %f1 as %1 : !async.value<memref<10xf16>>,
                %f3 as %3 : !async.value<memref<10xf16>>
            ) -> !async.value<memref<10xf16>> {
        %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_add
            inputs(%1 as %input_0: memref<10xf16>, %3 as %input_1: memref<10xf16>)
            outputs(%arg1 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
            VPUIP.SW.Kernel.run (%input_0, %input_1, %output) : memref<10xf16>, memref<10xf16>, memref<10xf16>
        }
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
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:          @builtin_relu
    // CHECK-SAME:          inputs(%arg0
    // CHECK-SAME:          outputs([[BUF0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          [[F0]] as [[VAL0:%.+]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:          @builtin_sigmoid
    // CHECK-SAME:          inputs([[VAL0]]
    // CHECK-SAME:          outputs([[BUF1]]

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-NOT:           [[T0]]
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:          @builtin_tanh
    // CHECK-SAME:          inputs(%arg0
    // CHECK-SAME:          outputs([[BUF2]]

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-NOT:           [[T0]]
    // CHECK-NOT:           [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          [[F2]] as [[VAL2:%.+]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:          @builtin_exp
    // CHECK-SAME:          inputs([[VAL2]]
    // CHECK-SAME:          outputs([[BUF3]]

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute
    // CHECK-NOT:           [[T0]]
    // CHECK-NOT:           [[T1]]
    // CHECK-NOT:           [[T2]]
    // CHECK-SAME:          [[T3]]
    // CHECK-SAME:          [[F1]] as [[VAL1:%.+]]: !async.value<memref<10xf16>>,
    // CHECK-SAME:          [[F3]] as [[VAL3:%.+]]: !async.value<memref<10xf16>>
    // CHECK-SAME:          -> !async.value<memref<10xf16>>
    // CHECK:           VPUIP.SW.Kernel
    // CHECK-SAME:          @builtin_add
    // CHECK-SAME:          inputs([[VAL1]]
    // CHECK-SAME:                 [[VAL3]]
    // CHECK-SAME:          outputs(%arg1

    // CHECK:       [[VAL4:%.+]] = async.await [[F4]] : !async.value<memref<10xf16>>
    // CHECK:       return [[VAL4]] : memref<10xf16>
}
