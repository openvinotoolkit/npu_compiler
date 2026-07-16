//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --async-scheduling %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

module @VPU.SW {
func.func nested @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func nested @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: func @main
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x1x1x100xf16>, [[ARG_1:%.+]]: memref<100xf16>) -> memref<100xf16>
func.func @main(%arg0: memref<1x1x1x100xf16>, %arg1: memref<100xf16>) -> memref<100xf16> {
    %buf0 = memref.alloc() : memref<1x1x1x100xf16>
    %buf1 = memref.alloc() : memref<1x1x1x100xf16>

    %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %input_0: memref<1x1x1x100xf16>)
                    outputs(%buf0 as %output_0: memref<1x1x1x100xf16>)
                    on tile 0 -> memref<1x1x1x100xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x100xf16>
                    , memref<1x1x1x100xf16>
    }
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%0 as %input_0: memref<1x1x1x100xf16>)
                    outputs(%buf1 as %output_0: memref<1x1x1x100xf16>)
                    on tile 0 -> memref<1x1x1x100xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x100xf16>
                    , memref<1x1x1x100xf16>
    }
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x1x1x100xf16>) -> memref<100xf16>
    %3 = VPUIP.Copy inputs(%2 : memref<100xf16>) outputs(%arg1 : memref<100xf16>) -> memref<100xf16>

    return %3: memref<100xf16>

    // CHECK:       [[BUF0:%.+]] = memref.alloc() : memref<1x1x1x100xf16>
    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<1x1x1x100xf16>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-SAME:          VPUIP.executor = @SHAVE_ACT
    // CHECK:           [[VAR0:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK-SAME:          inputs([[ARG_0]] as {{%[^:]+}}: memref<1x1x1x100xf16>)
    // CHECK-SAME:          outputs([[BUF0]] as {{%[^:]+}}: memref<1x1x1x100xf16>)
    // CHECK:           async.yield [[VAR0]]

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK-SAME:          [[T0]]
    // CHECK-SAME:          ([[F0]] as [[VAR0:%.+]]: !async.value<memref<1x1x1x100xf16>>)
    // CHECK-SAME:          VPUIP.executor = @SHAVE_ACT
    // CHECK:           [[VAR1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK-SAME:          inputs([[VAR0]] as {{%[^:]+}}: memref<1x1x1x100xf16>)
    // CHECK-SAME:          outputs([[BUF1]] as {{%[^:]+}}: memref<1x1x1x100xf16>)
    // CHECK:           async.yield [[VAR1]]

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAR1:%.+]]: !async.value<memref<1x1x1x100xf16>>)
    // CHECK-SAME:          VPUIP.executor = @DMA_NN
    // CHECK:           [[VAR2:%.+]] = VPUIP.GenericReshape inputs([[VAR1]] : memref<1x1x1x100xf16>) -> memref<100xf16>
    // CHECK:           [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<100xf16>) outputs([[ARG_1]] : memref<100xf16>)
    // CHECK:           async.yield [[VAR3]]

    // CHECK:       [[VAR3:%.+]] = async.await [[F3]]
    // CHECK:       return [[VAR3]]
}
