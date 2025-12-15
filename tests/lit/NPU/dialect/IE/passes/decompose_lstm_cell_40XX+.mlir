//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-lstm-cell %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// Configurations with small input and hidden sizes are optimized for SHAVES,
// so they should not be decomposed.

// CHECK-LABEL: func.func @DoNotDecomposeLSTMCell(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x32xf16>, [[VAL_1:%.+]]: tensor<1x64xf16>, [[VAL_2:%.+]]: tensor<1x64xf16>, [[VAL_3:%.+]]: tensor<256x32xf16>, [[VAL_4:%.+]]: tensor<256x64xf16>, [[VAL_5:%.+]]: tensor<256xf16>)
// CHECK-SAME:      -> (tensor<1x64xf16>, tensor<1x64xf16>) {
func.func @DoNotDecomposeLSTMCell(%arg0: tensor<1x32xf16>, %arg1: tensor<1x64xf16>, %arg2: tensor<1x64xf16>, %arg3: tensor<256x32xf16>, %arg4: tensor<256x64xf16>, %arg5: tensor<256xf16>) -> (tensor<1x64xf16>, tensor<1x64xf16>) {
    %cst = const.Declare tensor<256x32xf16> = dense<1.0> : tensor<256x32xf16>
    %cst_0 = const.Declare tensor<256x64xf16> = dense<2.0> : tensor<256x64xf16>
    %cst_1 = const.Declare tensor<256xf16> = dense<3.0> : tensor<256xf16>
    %outputHiddenState, %outputCellState = IE.LSTMCell(%arg0, %arg1, %arg2, %cst, %cst_0, %cst_1) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, hiddenSize = 64 : i64} : tensor<1x32xf16>, tensor<1x64xf16>, tensor<1x64xf16>, tensor<256x32xf16>, tensor<256x64xf16>, tensor<256xf16> -> tensor<1x64xf16>, tensor<1x64xf16>
    return %outputHiddenState, %outputCellState : tensor<1x64xf16>, tensor<1x64xf16>

// CHECK:   [[VAL_6:%.+]] = const.Declare tensor<256x32xf16> = dense<1.000000e+00> : tensor<256x32xf16>
// CHECK:   [[VAL_7:%.+]] = const.Declare tensor<256x64xf16> = dense<2.000000e+00> : tensor<256x64xf16>
// CHECK:   [[VAL_8:%.+]] = const.Declare tensor<256xf16> = dense<3.000000e+00> : tensor<256xf16>
// CHECK:   [[VAL_9:%.+]], [[VAL_10:%.+]] = IE.LSTMCell([[VAL_0]], [[VAL_1]], [[VAL_2]], [[VAL_6]], [[VAL_7]], [[VAL_8]]) {hiddenSize = 64 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>} : tensor<1x32xf16>, tensor<1x64xf16>, tensor<1x64xf16>, tensor<256x32xf16>, tensor<256x64xf16>, tensor<256xf16> -> tensor<1x64xf16>, tensor<1x64xf16>
// CHECK:   return [[VAL_9]], [[VAL_10]] : tensor<1x64xf16>, tensor<1x64xf16>
// CHECK: }
}
