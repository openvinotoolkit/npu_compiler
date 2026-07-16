// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FoldLogicalNotConstSI8AllZero
func.func @FoldLogicalNotConstSI8AllZero() -> tensor<1x2xi8> {
    %0 = const.Declare tensor<1x2xi8> = dense<0> : tensor<1x2xi8>
    %1 = IE.LogicalNot(%0) : tensor<1x2xi8> -> tensor<1x2xi8>
    return %1 : tensor<1x2xi8>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x2xi8> = dense<0> : tensor<1x2xi8>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @FoldLogicalNotConstSI8AllNonZero
func.func @FoldLogicalNotConstSI8AllNonZero() -> tensor<1x2xi8> {
    %0 = const.Declare tensor<1x2xi8> = dense<5> : tensor<1x2xi8>
    %1 = IE.LogicalNot(%0) : tensor<1x2xi8> -> tensor<1x2xi8>
    return %1 : tensor<1x2xi8>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x2xi8> = dense<5> : tensor<1x2xi8>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @FoldLogicalNotSplatZeroSI8
func.func @FoldLogicalNotSplatZeroSI8() -> tensor<2x3x4xi8> {
    %0 = const.Declare tensor<2x3x4xi8> = dense<0> : tensor<2x3x4xi8>
    %1 = IE.LogicalNot(%0) : tensor<2x3x4xi8> -> tensor<2x3x4xi8>
    return %1 : tensor<2x3x4xi8>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2x3x4xi8> = dense<0> : tensor<2x3x4xi8>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @FoldLogicalNotSplatNonZeroSI8
func.func @FoldLogicalNotSplatNonZeroSI8() -> tensor<2x3x4xi8> {
    %0 = const.Declare tensor<2x3x4xi8> = dense<5> : tensor<2x3x4xi8>
    %1 = IE.LogicalNot(%0) : tensor<2x3x4xi8> -> tensor<2x3x4xi8>
    return %1 : tensor<2x3x4xi8>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2x3x4xi8> = dense<5> : tensor<2x3x4xi8>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @FoldLogicalNotConstSI32
func.func @FoldLogicalNotConstSI32() -> tensor<3xsi32> {
    %0 = const.Declare tensor<3xsi32> = dense<[0, 2, -1]> : tensor<3xsi32>
    %1 = IE.LogicalNot(%0) : tensor<3xsi32> -> tensor<3xsi32>
    return %1 : tensor<3xsi32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<3xsi32> = dense<[0, 2, -1]> : tensor<3xsi32>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @FoldLogicalNotConstSI64
func.func @FoldLogicalNotConstSI64() -> tensor<3xsi64> {
    %0 = const.Declare tensor<3xsi64> = dense<[0, 9, -2]> : tensor<3xsi64>
    %1 = IE.LogicalNot(%0) : tensor<3xsi64> -> tensor<3xsi64>
    return %1 : tensor<3xsi64>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<3xsi64> = dense<[0, 9, -2]> : tensor<3xsi64>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @FoldLogicalNotSplatNonZeroSI64
func.func @FoldLogicalNotSplatNonZeroSI64() -> tensor<2x3x4xsi64> {
    %0 = const.Declare tensor<2x3x4xsi64> = dense<7> : tensor<2x3x4xsi64>
    %1 = IE.LogicalNot(%0) : tensor<2x3x4xsi64> -> tensor<2x3x4xsi64>
    return %1 : tensor<2x3x4xsi64>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2x3x4xsi64> = dense<7> : tensor<2x3x4xsi64>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @FoldLogicalNotConstF16Mixed
func.func @FoldLogicalNotConstF16Mixed() -> tensor<4xf16> {
    %0 = const.Declare tensor<4xf16> = dense<[0.0, 1.0, 0.0, -3.5]> : tensor<4xf16>
    %1 = IE.LogicalNot(%0) : tensor<4xf16> -> tensor<4xf16>
    return %1 : tensor<4xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<4xf16> = dense<[0.000000e+00, 1.000000e+00, 0.000000e+00, -3.500000e+00]> : tensor<4xf16>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @FoldLogicalNotSplatNonZeroF16
func.func @FoldLogicalNotSplatNonZeroF16() -> tensor<2x3xf16> {
    %0 = const.Declare tensor<2x3xf16> = dense<2.5> : tensor<2x3xf16>
    %1 = IE.LogicalNot(%0) : tensor<2x3xf16> -> tensor<2x3xf16>
    return %1 : tensor<2x3xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2x3xf16> = dense<2.500000e+00> : tensor<2x3xf16>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @FoldLogicalNotConstF32Mixed
func.func @FoldLogicalNotConstF32Mixed() -> tensor<4xf32> {
    %0 = const.Declare tensor<4xf32> = dense<[0.0, 1.0, 0.0, -3.5]> : tensor<4xf32>
    %1 = IE.LogicalNot(%0) : tensor<4xf32> -> tensor<4xf32>
    return %1 : tensor<4xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<4xf32> = dense<[0.000000e+00, 1.000000e+00, 0.000000e+00, -3.500000e+00]> : tensor<4xf32>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @DoNotFoldLogicalNotNonConst
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4xf32>
func.func @DoNotFoldLogicalNotNonConst(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = IE.LogicalNot(%arg0) : tensor<4xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>

    // CHECK: IE.LogicalNot([[INPUT]])
}

// -----

// CHECK-LABEL: @FoldLogicalNotConstF64Mixed
func.func @FoldLogicalNotConstF64Mixed() -> tensor<4xf64> {
    %0 = const.Declare tensor<4xf64> = dense<[0.0, 7.25, 0.0, -1.0]> : tensor<4xf64>
    %1 = IE.LogicalNot(%0) : tensor<4xf64> -> tensor<4xf64>
    return %1 : tensor<4xf64>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<4xf64> = dense<[0.000000e+00, 7.250000e+00, 0.000000e+00, -1.000000e+00]> : tensor<4xf64>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}

// -----

// CHECK-LABEL: @FoldLogicalNotSplatZeroF64
func.func @FoldLogicalNotSplatZeroF64() -> tensor<2x2xf64> {
    %0 = const.Declare tensor<2x2xf64> = dense<0.0> : tensor<2x2xf64>
    %1 = IE.LogicalNot(%0) : tensor<2x2xf64> -> tensor<2x2xf64>
    return %1 : tensor<2x2xf64>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2x2xf64> = dense<0.000000e+00> : tensor<2x2xf64>, [#const.LogicalNot]
    // CHECK-NOT: IE.LogicalNot
    // CHECK:     return [[CST]]
}
