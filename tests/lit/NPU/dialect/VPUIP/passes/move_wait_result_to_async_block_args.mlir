//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-wait-result-to-async-block-args %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @LinearCase
func.func @LinearCase(%arg0: memref<1x1x1x10xf16>, %arg1: memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16> {
    %buf0 = memref.alloc() : memref<1x1x1x10xf16>
    %buf1 = memref.alloc() : memref<1x1x1x10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %input_0: memref<1x1x1x10xf16>)
                    outputs(%buf0 as %output_0: memref<1x1x1x10xf16>)
                    on tile 0 -> memref<1x1x1x10xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x10xf16>
                    , memref<1x1x1x10xf16>
        }
        async.yield %1 : memref<1x1x1x10xf16>
    }
    %1 = async.await %f1 : !async.value<memref<1x1x1x10xf16>>

    %t2, %f2 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%1 as %input_0: memref<1x1x1x10xf16>)
                    outputs(%buf1 as %output_0: memref<1x1x1x10xf16>)
                    on tile 0 -> memref<1x1x1x10xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x10xf16>
                    , memref<1x1x1x10xf16>
        }
        async.yield %2 : memref<1x1x1x10xf16>
    }
    %2 = async.await %f2 : !async.value<memref<1x1x1x10xf16>>

    %t3, %f3 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%2 as %input_0: memref<1x1x1x10xf16>)
                    outputs(%arg1 as %output_0: memref<1x1x1x10xf16>)
                    on tile 0 -> memref<1x1x1x10xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x10xf16>
                    , memref<1x1x1x10xf16>
        }
        async.yield %3 : memref<1x1x1x10xf16>
    }
    %3 = async.await %f3 : !async.value<memref<1x1x1x10xf16>>

    return %3 : memref<1x1x1x10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<1x1x1x10xf16>>)
    // CHECK:           VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK-SAME:          inputs(
    // CHECK-SAME:              [[VAL1]] as {{[^:]+}}: memref<1x1x1x10xf16>
    // CHECK-SAME:          )
    // CHECK-NOT:   async.await

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-NOT:           [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          ([[F2]] as [[VAL2:%.+]]: !async.value<memref<1x1x1x10xf16>>)
    // CHECK:           VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK-SAME:          inputs(
    // CHECK-SAME:              [[VAL2]] as {{[^:]+}}: memref<1x1x1x10xf16>
    // CHECK-SAME:          )

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       return [[VAL3]]
}

// -----

module @VPU.SW {
    func.func private @builtin_Add(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_add.cpp", VPU.kernel_entry = "eltwise_add", VPU.task_type = @COMPUTE}
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @MultipleUsesInOneRegion
func.func @MultipleUsesInOneRegion(%arg0: memref<1x1x1x10xf16>, %arg1: memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16> {
    %buf0 = memref.alloc() : memref<1x1x1x10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %input_0: memref<1x1x1x10xf16>)
                    outputs(%buf0 as %output_0: memref<1x1x1x10xf16>)
                    on tile 0 -> memref<1x1x1x10xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x10xf16>
                    , memref<1x1x1x10xf16>
        }
        async.yield %1 : memref<1x1x1x10xf16>
    }
    %1 = async.await %f1 : !async.value<memref<1x1x1x10xf16>>

    %t2, %f2 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
                    inputs(%1 as %input_0: memref<1x1x1x10xf16>,
                           %1 as %input_1: memref<1x1x1x10xf16>)
                    outputs(%arg1 as %output_0: memref<1x1x1x10xf16>)
                    on tile 0 -> memref<1x1x1x10xf16> {
                VPUIP.SW.Kernel.run (%input_0, %input_1, %output_0)
                    : memref<1x1x1x10xf16>, memref<1x1x1x10xf16>
                    , memref<1x1x1x10xf16>
        }
        async.yield %2 : memref<1x1x1x10xf16>
    }
    %2 = async.await %f2 : !async.value<memref<1x1x1x10xf16>>

    return %2 : memref<1x1x1x10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<1x1x1x10xf16>>)
    // CHECK:           VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add

    // CHECK:       [[VAL2:%.+]] = async.await [[F2]]
    // CHECK:       return [[VAL2]]
}

// -----

module @VPU.SW {
    func.func private @builtin_Add(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_add.cpp", VPU.kernel_entry = "eltwise_add", VPU.task_type = @COMPUTE}
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @UsesFromMultipleWaits
func.func @UsesFromMultipleWaits(%arg0: memref<1x1x1x10xf16>, %arg1: memref<1x1x1x10xf16>) -> memref<1x1x1x10xf16> {
    %buf0 = memref.alloc() : memref<1x1x1x10xf16>
    %buf1 = memref.alloc() : memref<1x1x1x10xf16>

    %t1, %f1 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %input_0: memref<1x1x1x10xf16>)
                    outputs(%buf0 as %output_0: memref<1x1x1x10xf16>)
                    on tile 0 -> memref<1x1x1x10xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x10xf16>
                    , memref<1x1x1x10xf16>
        }
        async.yield %1 : memref<1x1x1x10xf16>
    }
    %1 = async.await %f1 : !async.value<memref<1x1x1x10xf16>>

    %t2, %f2 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %input_0: memref<1x1x1x10xf16>)
                    outputs(%buf1 as %output_0: memref<1x1x1x10xf16>)
                    on tile 0 -> memref<1x1x1x10xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x10xf16>
                    , memref<1x1x1x10xf16>
        }
        async.yield %2 : memref<1x1x1x10xf16>
    }
    %2 = async.await %f2 : !async.value<memref<1x1x1x10xf16>>

    %t3, %f3 = async.execute -> !async.value<memref<1x1x1x10xf16>> {
        %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
                    inputs(%1 as %input_0: memref<1x1x1x10xf16>,
                           %2 as %input_1: memref<1x1x1x10xf16>)
                    outputs(%arg1 as %output_0: memref<1x1x1x10xf16>)
                    on tile 0 -> memref<1x1x1x10xf16> {
                VPUIP.SW.Kernel.run (%input_0, %input_1, %output_0)
                    : memref<1x1x1x10xf16>, memref<1x1x1x10xf16>
                    , memref<1x1x1x10xf16>
        }
        async.yield %3 : memref<1x1x1x10xf16>
    }
    %3 = async.await %f3 : !async.value<memref<1x1x1x10xf16>>

    return %3 : memref<1x1x1x10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-NOT:   async.await

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%.+]]: !async.value<memref<1x1x1x10xf16>>,
    // CHECK-SAME:           [[F2]] as [[VAL2:%.+]]: !async.value<memref<1x1x1x10xf16>>)
    // CHECK:           VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       return [[VAL3]]
}

// -----

module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TwoOutputs
func.func @TwoOutputs(%arg0: memref<1x1x1x2xf16>, %arg1: memref<1x1x1x2xf16>, %arg2: memref<1x1x1x2xf16>) -> (memref<1x1x1x2xf16>, memref<1x1x1x2xf16>) {
    %cst = const.Declare memref<1x1x1x2xf16> = dense<1.0> : tensor<1x1x1x2xf16>

    %buf1 = memref.alloc() : memref<1x1x1x2xf16>
    %buf2 = memref.alloc() : memref<1x1x1x2xf16>

    %t1, %f1 = async.execute -> !async.value<memref<1x1x1x2xf16>> {
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %input_0: memref<1x1x1x2xf16>)
                    outputs(%buf1 as %output_0: memref<1x1x1x2xf16>)
                    on tile 0 -> memref<1x1x1x2xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x2xf16>
                    , memref<1x1x1x2xf16>
        }
        async.yield %1 : memref<1x1x1x2xf16>
    }
    %1 = async.await %f1 : !async.value<memref<1x1x1x2xf16>>


    %t2, %f2 = async.execute -> !async.value<memref<1x1x1x2xf16>> {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%cst as %input_0: memref<1x1x1x2xf16>)
                    outputs(%buf2 as %output_0: memref<1x1x1x2xf16>)
                    on tile 0 -> memref<1x1x1x2xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x2xf16>
                    , memref<1x1x1x2xf16>
        }
        async.yield %2 : memref<1x1x1x2xf16>
    }
    %2 = async.await %f2 : !async.value<memref<1x1x1x2xf16>>

    %t3, %f3 = async.execute -> !async.value<memref<1x1x1x2xf16>> {
        %3 = VPUIP.Copy inputs(%1 : memref<1x1x1x2xf16>) outputs(%arg1 : memref<1x1x1x2xf16>) -> memref<1x1x1x2xf16>
        async.yield %3 : memref<1x1x1x2xf16>
    }
    %3 = async.await %f3 : !async.value<memref<1x1x1x2xf16>>

    %t4, %f4 = async.execute -> !async.value<memref<1x1x1x2xf16>> {
        %4 = VPUIP.Copy inputs(%2 : memref<1x1x1x2xf16>) outputs(%arg2 : memref<1x1x1x2xf16>) -> memref<1x1x1x2xf16>
        async.yield %4 : memref<1x1x1x2xf16>
    }
    %4 = async.await %f4 : !async.value<memref<1x1x1x2xf16>>

    return %3, %4 : memref<1x1x1x2xf16>, memref<1x1x1x2xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare

    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<1x1x1x2xf16>
    // CHECK:       [[BUF2:%.+]] = memref.alloc() : memref<1x1x1x2xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK:           VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK-SAME:          inputs(
    // CHECK-SAME:              %arg0 as {{[^:]+}}: memref<1x1x1x2xf16>
    // CHECK-SAME:          ) outputs(
    // CHECK-SAME:              [[BUF1]] as {{[^:]+}}: memref<1x1x1x2xf16>
    // CHECK-SAME:          )

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK:           VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK-SAME:          inputs(
    // CHECK-SAME:              [[CST]] as {{[^:]+}}: memref<1x1x1x2xf16>
    // CHECK-SAME:          ) outputs(
    // CHECK-SAME:              [[BUF2]] as {{[^:]+}}: memref<1x1x1x2xf16>
    // CHECK-SAME:          )

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-NOT:           [[T2]]
    // CHECK-SAME:          [[F1]] as [[VAL1:%.+]]: !async.value<memref<1x1x1x2xf16>>
    // CHECK:           VPUIP.Copy inputs([[VAL1]] : memref<1x1x1x2xf16>) outputs(%arg1 : memref<1x1x1x2xf16>)

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute
    // CHECK-NOT:           [[T1]]
    // CHECK-SAME:          [[T2]]
    // CHECK-NOT:           [[T3]]
    // CHECK-SAME:          [[F2]] as [[VAL2:%.+]]: !async.value<memref<1x1x1x2xf16>>
    // CHECK:           VPUIP.Copy inputs([[VAL2]] : memref<1x1x1x2xf16>) outputs(%arg2 : memref<1x1x1x2xf16>)

    // CHECK:       [[VAL3:%.+]] = async.await [[F3]]
    // CHECK:       [[VAL4:%.+]] = async.await [[F4]]
    // CHECK:       return [[VAL3]], [[VAL4]]
}
