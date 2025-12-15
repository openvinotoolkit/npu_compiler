//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-assign-read-value --mlir-print-debuginfo %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertAssignAndReadValue
net.NetworkInfo entryPoint : @ConvertAssignAndReadValue inputsInfo : {
    DataInfo "input1" : tensor<1x768xf32> loc(fused<{name = "input", type = "Parameter"}>["input"])
} outputsInfo : {
    DataInfo "Gemm_9" : tensor<1x768xf32> loc(fused<{name = "output", type = "Result"}>["output"])
}
func.func @ConvertAssignAndReadValue(%arg0: tensor<1x768xf32>) -> tensor<1x768xf32> {
    %cst = const.Declare tensor<1x768xf32> = dense<1.100000e+00> : tensor<1x768xf32>
    %0 = IE.ReadValue(%cst) {name = "inner_h1"} : tensor<1x768xf32> -> tensor<1x768xf32> loc(fused<{name = "inner_h1_r", type = "ReadValue"}>["inner_h1_r"])
    %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768xf32>, tensor<1x768xf32> -> tensor<1x768xf32>
    %2 = IE.Assign(%1) {name = "inner_h1"} : tensor<1x768xf32> -> tensor<1x768xf32> loc(fused<{name = "inner_h1_w", type = "Assign"}>["inner_h1_w"])
    return %1 : tensor<1x768xf32>

    // CHECK-NOT:   IE.ReadValue
    // CHECK-NOT:   IE.Assign
    // CHECK:       DataInfo "vpux_ie_read_value_inner_h1" : tensor<1x768xf32> loc([[LOC_READ:#.+]])
    // CHECK:       DataInfo "vpux_ie_assign_inner_h1" : tensor<1x768xf32> loc([[LOC_ASSIGN:#.+]])

    // CHECK:       @ConvertAssignAndReadValue(%arg0: tensor<1x768xf32> loc([[LOC_FOO1_ARG0:.+]]), %arg1: tensor<1x768xf32> loc([[LOC_FOO2_ARG0:.+]])) -> (tensor<1x768xf32>, tensor<1x768xf32>)
    // CHECK:       [[VAR0:%.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768xf32>, tensor<1x768xf32> -> tensor<1x768xf32>
    // CHECK:       return [[VAR0]], [[VAR0]] : tensor<1x768xf32>, tensor<1x768xf32>

    // CHECK: [[BASE_LOC_READ:#.+]] = loc("inner_h1_r")
    // CHECK: [[READ_SUFFIX:#.+]] = loc("read_inner_h1")
    // CHECK: [[BASE_LOC_ASSIGN:#.+]] = loc("inner_h1_w")
    // CHECK: [[WRITE_SUFFIX:#.+]] = loc("assign_inner_h1")

    // CHECK: [[LOC_READ]] = loc(fused<{name = "inner_h1_r", type = "ReadValue"}>[[[BASE_LOC_READ]], [[READ_SUFFIX]]])
    // CHECK: [[LOC_ASSIGN]] = loc(fused<{name = "inner_h1_w", type = "Assign"}>[[[BASE_LOC_ASSIGN]], [[WRITE_SUFFIX]]])
}
