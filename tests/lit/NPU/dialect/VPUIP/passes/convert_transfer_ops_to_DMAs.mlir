//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-transfer-ops-to-DMAs %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @CopyToDMA
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x2x2x2xf16>)
func.func @CopyToDMA(%arg0: memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16> {
    %0 = const.Declare memref<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf16>
    %1 = VPUIP.Copy inputs(%0 : memref<1x2x2x2xf16>) outputs(%arg0 : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>
    return %1: memref<1x2x2x2xf16>

    // CHECK-DAG:       [[VAR0:%.+]] = const.Declare memref<1x2x2x2xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x2x2x2xf16>

    // CHECK:       [[VAR1:%.+]] = VPUIP.NNDMA
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:      outputs([[ARG_0]] : memref<1x2x2x2xf16>) -> memref<1x2x2x2xf16>

    // CHECK: return [[VAR1]] : memref<1x2x2x2xf16>
}
