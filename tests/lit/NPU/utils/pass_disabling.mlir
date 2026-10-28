//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% disabled-passes=set-memory-space" --set-memory-space="memory-space=DDR" %s | FileCheck --check-prefix=CHECK-DISABLED %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --set-memory-space="memory-space=DDR" %s | FileCheck --check-prefix=CHECK-ENABLED %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

net.NetworkInfo entryPoint : @TestPassDisabled inputsInfo :  {
    DataInfo "input" : tensor<1x1000xf16>
} outputsInfo :  {
    DataInfo "output" : tensor<1x1000xf16>
}

// CHECK-DISABLED-LABEL: func.func @TestPassDisabled
// CHECK-DISABLED-SAME:  ([[ARG0:%.+]]: memref<1x1000xf16>) -> memref<1x1000xf16>
func.func @TestPassDisabled(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
    return %arg0 : memref<1x1000xf16>
    // CHECK-DISABLED:   return [[ARG0]] : memref<1x1000xf16>
}

// -----

net.NetworkInfo entryPoint : @TestPassEnabled inputsInfo :  {
    DataInfo "input" : tensor<1x1000xf16>
} outputsInfo :  {
    DataInfo "output" : tensor<1x1000xf16>
}

// CHECK-ENABLED-LABEL: func.func @TestPassEnabled
// CHECK-ENABLED-SAME:  ([[ARG0:%.+]]: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
func.func @TestPassEnabled(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
    return %arg0 : memref<1x1000xf16>
    // CHECK-ENABLED:   return [[ARG0]] : memref<1x1000xf16, @DDR>
}
