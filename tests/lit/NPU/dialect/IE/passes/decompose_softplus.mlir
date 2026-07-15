//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --decompose-softplus %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @DecomposeSoftPlusWithAbsNeg4D
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x28x28xf16>
func.func @DecomposeSoftPlusWithAbsNeg4D(%arg0: tensor<1x16x28x28xf16>) -> tensor<1x16x28x28xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>

    %abs = IE.Abs(%arg0) : tensor<1x16x28x28xf16> -> tensor<1x16x28x28xf16>
    %neg = IE.Multiply(%abs, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x28x28xf16>
    %sp = IE.SoftPlus(%neg) : tensor<1x16x28x28xf16> -> tensor<1x16x28x28xf16>

    return %sp : tensor<1x16x28x28xf16>

    // CHECK-NOT: IE.SoftPlus
    // CHECK:     [[ABS:%.+]] = IE.Abs([[INPUT]])
    // CHECK:     [[NEG:%.+]] = IE.Multiply([[ABS]], {{%.+}})
    // CHECK:     [[EXP:%.+]] = IE.Exp([[NEG]])
    // CHECK:     [[ONE:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00>
    // CHECK:     [[ADD:%.+]] = IE.Add([[EXP]], [[ONE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:     [[LOG:%.+]] = IE.Log([[ADD]])

    // CHECK:     return [[LOG]]
}

// -----

// CHECK-LABEL: @DecomposeSoftPlusWithAbsNeg3D
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x700x80xf16>
func.func @DecomposeSoftPlusWithAbsNeg3D(%arg0: tensor<1x700x80xf16>) -> tensor<1x700x80xf16> {
    %cst = const.Declare tensor<1x1x1xf16> = dense<-1.0> : tensor<1x1x1xf16>

    %abs = IE.Abs(%arg0) : tensor<1x700x80xf16> -> tensor<1x700x80xf16>
    %neg = IE.Multiply(%abs, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x700x80xf16>, tensor<1x1x1xf16> -> tensor<1x700x80xf16>
    %sp = IE.SoftPlus(%neg) : tensor<1x700x80xf16> -> tensor<1x700x80xf16>

    return %sp : tensor<1x700x80xf16>

    // CHECK-NOT: IE.SoftPlus
    // CHECK:     [[ABS:%.+]] = IE.Abs([[INPUT]])
    // CHECK:     [[NEG:%.+]] = IE.Multiply([[ABS]], {{%.+}})
    // CHECK:     [[EXP:%.+]] = IE.Exp([[NEG]])
    // CHECK:     [[ONE:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00>
    // CHECK:     [[ADD:%.+]] = IE.Add([[EXP]], [[ONE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:     [[LOG:%.+]] = IE.Log([[ADD]])

    // CHECK:     return [[LOG]]
}

// -----

// CHECK-LABEL: @NoDecomposeWithoutAbs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x28x28xf16>
func.func @NoDecomposeWithoutAbs(%arg0: tensor<1x16x28x28xf16>) -> tensor<1x16x28x28xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>

    %neg = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x28x28xf16>
    %sp = IE.SoftPlus(%neg) : tensor<1x16x28x28xf16> -> tensor<1x16x28x28xf16>

    return %sp : tensor<1x16x28x28xf16>

    // CHECK: IE.SoftPlus
    // CHECK-NOT: IE.Exp
}

// -----

// CHECK-LABEL: @NoDecomposePositiveMultiplier
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x28x28xf16>
func.func @NoDecomposePositiveMultiplier(%arg0: tensor<1x16x28x28xf16>) -> tensor<1x16x28x28xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>

    %abs = IE.Abs(%arg0) : tensor<1x16x28x28xf16> -> tensor<1x16x28x28xf16>
    %mul = IE.Multiply(%abs, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x28x28xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x28x28xf16>
    %sp = IE.SoftPlus(%mul) : tensor<1x16x28x28xf16> -> tensor<1x16x28x28xf16>

    return %sp : tensor<1x16x28x28xf16>

    // CHECK: IE.SoftPlus
    // CHECK-NOT: IE.Exp
}
