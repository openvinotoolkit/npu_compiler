//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @IrdftConvertConstToAttrAndNormalize
// CHECK-SAME:   ([[INPUT:%arg[0-9]]]: tensor<10x4x2xf32>) -> tensor<10x6xf32>
func.func @IrdftConvertConstToAttrAndNormalize(%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32> {
    %cst = const.Declare tensor<2xsi32> = dense<[0, 1]> : tensor<2xsi64>, [#const.CastElemType<si32>]
    %0 = IE.IRDFT(%arg0, %cst) {operandSegmentSizes = array<i32: 1, 1, 0>} : tensor<10x4x2xf32>, tensor<2xsi32> -> tensor<10x6xf32>
    return %0 : tensor<10x6xf32>

    // CHECK: [[OUTPUT:%.+]] = IE.IRDFT([[INPUT]]) {axes_attr = [0, 1], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
    // CHECK: return [[OUTPUT]] : tensor<10x6xf32>
}
