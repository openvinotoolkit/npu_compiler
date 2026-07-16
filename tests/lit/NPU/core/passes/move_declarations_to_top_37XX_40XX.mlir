//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --move-declarations-to-top %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000


module @VPU.SW {
    func.func nested @builtin_Add(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {
        VPU.kernel_code = "eltwise_add.cpp", VPU.kernel_entry = "eltwise_add", VPU.task_type = @COMPUTE
    }
}

// CHECK-LABEL: @MoveToTopOfBlock
// CHECK-SAME:    [[ARG_0:%[^:]+]]: memref<16xf16>
// CHECK-SAME:    [[ARG_1:%[^:]+]]: memref<8xf16>
func.func @MoveToTopOfBlock(%arg0: memref<16xf16>, %arg1: memref<8xf16>) -> memref<8xf16> {
    %0 = VPUIP.SubView %arg0 [0] [8] : memref<16xf16> to memref<8xf16>
    %decl0 = VPUIP.StaticAlloc<0> -> memref<8xf16>
    %cst2 = const.Declare memref<8xf16> = dense<2.000000e+00> : tensor<8xf16>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
        inputs(%0 as %input_0: memref<8xf16>, %cst2 as %input_1: memref<8xf16>)
        outputs(%decl0 as %output: memref<8xf16>) on tile 0 -> memref<8xf16> {
        VPUIP.SW.Kernel.run (%input_0, %input_1, %output) : memref<8xf16>, memref<8xf16>, memref<8xf16>
    }

    %cst4 = const.Declare memref<8xf16> = dense<4.000000e+00> : tensor<8xf16>
    %decl16 = VPUIP.StaticAlloc<16> -> memref<8xf16>
    %t2, %f2 = async.execute -> !async.value<memref<8xf16>> {
        %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
            inputs(%1 as %input_0: memref<8xf16>, %cst4 as %input_1: memref<8xf16>)
            outputs(%decl16 as %output: memref<8xf16>) on tile 0 -> memref<8xf16> {
            VPUIP.SW.Kernel.run (%input_0, %input_1, %output) : memref<8xf16>, memref<8xf16>, memref<8xf16>
        }
        async.yield %2 : memref<8xf16>
    }
    %3 = async.await %f2 : !async.value<memref<8xf16>>

    %decl32 = VPUIP.StaticAlloc<32> -> memref<8xf16>
    %cst9 = const.Declare memref<8xf16> = dense<9.000000e+00> : tensor<8xf16>
    %t4, %f4 = async.execute -> !async.value<memref<8xf16>> {
        %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
            inputs(%3 as %input_0: memref<8xf16>, %cst9 as %input_1: memref<8xf16>)
            outputs(%decl32 as %output: memref<8xf16>) on tile 0 -> memref<8xf16> {
            VPUIP.SW.Kernel.run (%input_0, %input_1, %output) : memref<8xf16>, memref<8xf16>, memref<8xf16>
        }
        async.yield %4 : memref<8xf16>
    }
    %5 = async.await %f4 : !async.value<memref<8xf16>>

    %6 = VPUIP.Copy inputs(%5 : memref<8xf16>) outputs(%arg1 : memref<8xf16>) -> memref<8xf16>
    return %6 : memref<8xf16>

    // CHECK-DAG:   [[CST2:%.+]] = const.Declare memref<8xf16> = dense<2.000000e+00> : tensor<8xf16>
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare memref<8xf16> = dense<4.000000e+00> : tensor<8xf16>
    // CHECK-DAG:   [[CST9:%.+]] = const.Declare memref<8xf16> = dense<9.000000e+00> : tensor<8xf16>

    // CHECK-DAG:   [[DECL0:%.+]] = VPUIP.StaticAlloc<0> -> memref<8xf16>
    // CHECK-DAG:   [[DECL16:%.+]] = VPUIP.StaticAlloc<16> -> memref<8xf16>
    // CHECK-DAG:   [[DECL32:%.+]] = VPUIP.StaticAlloc<32> -> memref<8xf16>

    // CHECK:       [[VAR0:%.+]] = VPUIP.SubView [[ARG_0]] [0] [8] : memref<16xf16> to memref<8xf16>

    // CHECK:       [[VAR1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
    // CHECK-SAME:      inputs([[VAR0]]
    // CHECK-SAME:             [[CST2]]
    // CHECK-SAME:      outputs([[DECL0]]

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute -> !async.value<memref<8xf16>> {
    // CHECK:           [[VAR2:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
    // CHECK-SAME:          inputs([[VAR1]]
    // CHECK-SAME:                 [[CST4]]
    // CHECK-SAME:          outputs([[DECL16]]
    // CHECK:           async.yield [[VAR2]] : memref<8xf16>
    // CHECK:       }
    // CHECK:       [[VAR3:%.+]] = async.await [[F2]] : !async.value<memref<8xf16>>

    // CHECK:       [[T4:%.+]], [[F4:%.+]] = async.execute -> !async.value<memref<8xf16>> {
    // CHECK:           [[VAR4:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Add
    // CHECK-SAME:          inputs([[VAR3]]
    // CHECK-SAME:                 [[CST9]]
    // CHECK-SAME:          outputs([[DECL32]]
    // CHECK:           async.yield [[VAR4]] : memref<8xf16>
    // CHECK:       }
    // CHECK:       [[VAR5:%.+]] = async.await [[F4]] : !async.value<memref<8xf16>>

    // CHECK:       [[VAR6:%.+]] = VPUIP.Copy inputs([[VAR5]] : memref<8xf16>) outputs([[ARG_1]] : memref<8xf16>)
    // CHECK:       return [[VAR6]] : memref<8xf16>
}
