//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FoldPowerWithExponentEqual1
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x900x161xf16>
func.func @FoldPowerWithExponentEqual1(%arg0: tensor<1x900x161xf16>) -> tensor<1x900x161xf16> {
    %cst = const.Declare tensor<1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %power = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x900x161xf16>, tensor<1x1x1xf16> -> tensor<1x900x161xf16>
    return %power : tensor<1x900x161xf16>

    // CHECK-NOT:   IE.Power
    // CHECK:       return [[INPUT]] : tensor<1x900x161xf16>
}

// -----

// CHECK-LABEL: @NotFoldPowerWithExponentNotEqual1
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x900x161xf16>
func.func @NotFoldPowerWithExponentNotEqual1(%arg0: tensor<1x900x161xf16>) -> tensor<1x900x161xf16> {
    %cst = const.Declare tensor<1x1x1xf16> = dense<1.200000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %power = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x900x161xf16>, tensor<1x1x1xf16> -> tensor<1x900x161xf16>
    return %power : tensor<1x900x161xf16>

    // CHECK-DAG:   [[EXPONENT:%.+]] = const.Declare tensor<1x1x1xf16> = dense<1.200000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[POWER:%.+]] = IE.Power([[INPUT]], [[EXPONENT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x900x161xf16>, tensor<1x1x1xf16> -> tensor<1x900x161xf16>
    // CHECK:       return [[POWER]]
}

// -----

// CHECK-LABEL: @FuseSqrtPowerWithExponent2
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x900x161xf16>
func.func @FuseSqrtPowerWithExponent2(%arg0: tensor<1x900x161xf16>) -> tensor<1x900x161xf16> {
    %cst = const.Declare tensor<1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %sqrt = IE.Sqrt(%arg0) : tensor<1x900x161xf16> -> tensor<1x900x161xf16>
    %power = IE.Power(%sqrt, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x900x161xf16>, tensor<1x1x1xf16> -> tensor<1x900x161xf16>
    return %power : tensor<1x900x161xf16>

    // CHECK-NOT:   IE.Sqrt
    // CHECK:       return [[INPUT]] : tensor<1x900x161xf16>
}

// -----

// CHECK-LABEL: @NoteFuseSqrtPowerWithExponentNotEqual2
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x900x161xf16>
func.func @NoteFuseSqrtPowerWithExponentNotEqual2(%arg0: tensor<1x900x161xf16>) -> tensor<1x900x161xf16> {
    %cst = const.Declare tensor<1x1x1xf16> = dense<2.200000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %sqrt = IE.Sqrt(%arg0) : tensor<1x900x161xf16> -> tensor<1x900x161xf16>
    %power = IE.Power(%sqrt, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x900x161xf16>, tensor<1x1x1xf16> -> tensor<1x900x161xf16>
    return %power : tensor<1x900x161xf16>

    // CHECK-DAG:   [[EXPONENT:%.+]] = const.Declare tensor<1x1x1xf16> = dense<2.200000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[SQRT:%.+]] = IE.Sqrt([[INPUT]]) : tensor<1x900x161xf16> -> tensor<1x900x161xf16>
    // CHECK:       [[POWER:%.+]] = IE.Power([[SQRT]], [[EXPONENT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x900x161xf16>, tensor<1x1x1xf16> -> tensor<1x900x161xf16>
    // CHECK:       return [[POWER]]
}

// -----

// CHECK-LABEL: @NoteFuseSqrtPowerWithMultiUsers
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x900x161xf16>
func.func @NoteFuseSqrtPowerWithMultiUsers(%arg0: tensor<1x900x161xf16>) -> (tensor<1x900x161xf16>, tensor<1x900x161xf16>) {
    %cst = const.Declare tensor<1x1x1xf16> = dense<2.200000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %sqrt = IE.Sqrt(%arg0) : tensor<1x900x161xf16> -> tensor<1x900x161xf16>
    %power = IE.Power(%sqrt, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x900x161xf16>, tensor<1x1x1xf16> -> tensor<1x900x161xf16>
    %sigmoid = IE.Sigmoid(%sqrt) : tensor<1x900x161xf16> -> tensor<1x900x161xf16>
    return %power, %sigmoid : tensor<1x900x161xf16>, tensor<1x900x161xf16>

    // CHECK-DAG:   [[EXPONENT:%.+]] = const.Declare tensor<1x1x1xf16> = dense<2.200000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[SQRT:%.+]] = IE.Sqrt([[INPUT]]) : tensor<1x900x161xf16> -> tensor<1x900x161xf16>
    // CHECK:       [[POWER:%.+]] = IE.Power([[SQRT]], [[EXPONENT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x900x161xf16>, tensor<1x1x1xf16> -> tensor<1x900x161xf16>
    // CHECK:       [[SIGMOID:%.+]] = IE.Sigmoid([[SQRT]]) : tensor<1x900x161xf16> -> tensor<1x900x161xf16>
    // CHECK:       return [[POWER]], [[SIGMOID]]
}
