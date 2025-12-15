//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-power-to-mult %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertPowerWithExponent2ToMult
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16xf16>
func.func @ConvertPowerWithExponent2ToMult(%arg0: tensor<1x16xf16>) -> tensor<1x16xf16> {
    %cst_exponent = const.Declare tensor<1x1xf16> = dense<2.0> : tensor<1x1xf16>
    %power = IE.Power(%arg0, %cst_exponent) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16xf16>, tensor<1x1xf16> -> tensor<1x16xf16>
    return %power : tensor<1x16xf16>

    // CHECK-NOT:   IE.Power
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK-SAME:      tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>
    // CHECK:       return [[MUL]]
}

// -----

// CHECK-LABEL: @ConvertPowerWithExponent3ToMult
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16xf16>
func.func @ConvertPowerWithExponent3ToMult(%arg0: tensor<1x16xf16>) -> tensor<1x16xf16> {
    %cst_exponent = const.Declare tensor<1x1xf16> = dense<3.0> : tensor<1x1xf16>
    %power = IE.Power(%arg0, %cst_exponent) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16xf16>, tensor<1x1xf16> -> tensor<1x16xf16>
    return %power : tensor<1x16xf16>

    // CHECK-NOT:   IE.Power
    // CHECK:       [[MUL_0:%.+]] = IE.Multiply([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK-SAME:      tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>
    // CHECK:       [[MUL_1:%.+]] = IE.Multiply([[MUL_0]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK-SAME:      tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>
    // CHECK:       return [[MUL_1]]
}

// -----

// CHECK-LABEL: @NotConvertPowerWithExponent4ToMult
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16xf16>
func.func @NotConvertPowerWithExponent4ToMult(%arg0: tensor<1x16xf16>) -> tensor<1x16xf16> {
    %cst_exponent = const.Declare tensor<1x1xf16> = dense<4.0> : tensor<1x1xf16>
    %power = IE.Power(%arg0, %cst_exponent) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16xf16>, tensor<1x1xf16> -> tensor<1x16xf16>
    return %power : tensor<1x16xf16>

    // CHECK-DAG:   [[EXPONENT:%.+]] = const.Declare tensor<1x1xf16> = dense<4.000000e+00> : tensor<1x1xf16>
    // CHECK:       [[POWER:%.+]] = IE.Power([[INPUT]], [[EXPONENT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x16xf16>, tensor<1x1xf16> -> tensor<1x16xf16>
    // CHECK:       return [[POWER]]
}

// -----

// CHECK-LABEL: @NotConvertPowerWithoutIntegerExponentToMult
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16xf16>
func.func @NotConvertPowerWithoutIntegerExponentToMult(%arg0: tensor<1x16xf16>) -> tensor<1x16xf16> {
    %cst_exponent = const.Declare tensor<1x1xf16> = dense<2.2> : tensor<1x1xf16>
    %power = IE.Power(%arg0, %cst_exponent) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16xf16>, tensor<1x1xf16> -> tensor<1x16xf16>
    return %power : tensor<1x16xf16>

    // CHECK-DAG:   [[EXPONENT:%.+]] = const.Declare tensor<1x1xf16> = dense<2.199220e+00> : tensor<1x1xf16>
    // CHECK:       [[POWER:%.+]] = IE.Power([[INPUT]], [[EXPONENT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x16xf16>, tensor<1x1xf16> -> tensor<1x16xf16>
    // CHECK:       return [[POWER]]
}
