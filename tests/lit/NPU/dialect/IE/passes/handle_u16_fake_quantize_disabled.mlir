//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=512 --init-compiler="platform=%platform% enable-adaptive-stripping=false" --handle-u16-fake-quantize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @RemoveFQU16
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x640x640xf16>)
func.func @RemoveFQU16(%arg0: tensor<1x3x640x640xf16>) -> tensor<1x4x640x640xf16> {
    %cst = const.Declare tensor<1x4x1x1xf16> = dense<[[[[0.539961219]], [[-1.85364282]], [[1.99483931]], [[-2.17146444]]]]> : tensor<1x4x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<4x3x3x3xf16> = dense<[[[[-7.72567116E-4, 0.00231770123, 0.00103008945], [0.00154513423, 0.00386283547, 0.00231770123], [0.00128761178, 2.57522363E-4, 7.72567116E-4]], [[-7.72567116E-4, 0.00360531313, 0.0020601789], [0.00154513423, 0.0056654918, 0.00231770123], [0.000000e+00, 2.57522363E-4, -7.72567116E-4]], [[2.57522363E-4, 0.0066955816, 0.00180265657], [0.0028327459, 0.00515044713, 7.72567116E-4], [0.00154513423, -0.00103008945, -0.00154513423]]], [[[-7.72567116E-4, 0.00231770123, 0.00103008945], [0.00154513423, 0.00386283547, 0.00231770123], [0.00128761178, 2.57522363E-4, 7.72567116E-4]], [[-7.72567116E-4, 0.00360531313, 0.0020601789], [0.00154513423, 0.0056654918, 0.00231770123], [0.000000e+00, 2.57522363E-4, -7.72567116E-4]], [[2.57522363E-4, 0.0066955816, 0.00180265657], [0.0028327459, 0.00515044713, 7.72567116E-4], [0.00154513423, -0.00103008945, -0.00154513423]]],[[[-7.72567116E-4, 0.00231770123, 0.00103008945], [0.00154513423, 0.00386283547, 0.00231770123], [0.00128761178, 2.57522363E-4, 7.72567116E-4]], [[-7.72567116E-4, 0.00360531313, 0.0020601789], [0.00154513423, 0.0056654918, 0.00231770123], [0.000000e+00, 2.57522363E-4, -7.72567116E-4]], [[2.57522363E-4, 0.0066955816, 0.00180265657], [0.0028327459, 0.00515044713, 7.72567116E-4], [0.00154513423, -0.00103008945, -0.00154513423]]], [[[-7.72567116E-4, 0.00231770123, 0.00103008945], [0.00154513423, 0.00386283547, 0.00231770123], [0.00128761178, 2.57522363E-4, 7.72567116E-4]], [[-7.72567116E-4, 0.00360531313, 0.0020601789], [0.00154513423, 0.0056654918, 0.00231770123], [0.000000e+00, 2.57522363E-4, -7.72567116E-4]], [[2.57522363E-4, 0.0066955816, 0.00180265657], [0.0028327459, 0.00515044713, 7.72567116E-4], [0.00154513423, -0.00103008945, -0.00154513423]]]]> : tensor<4x3x3x3xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<57.9222679> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-57.1374702> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.Convolution(%arg0, %cst_0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x640x640xf16>, tensor<4x3x3x3xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x640x640xf16>
    %1 = IE.FakeQuantize(%0, %cst_2, %cst_1, %cst_2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x4x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x640x640xf16>
    %2 = IE.Sigmoid(%1) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>
    return %2 : tensor<1x4x640x640xf16>

    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x4x1x1xf16> =
    // CHECK-SAME{LITERAL}:   dense<[[[[0.539961219]], [[-1.85364282]], [[1.99483931]], [[-2.17146444]]]]> : tensor<1x4x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<4x3x3x3xf16> =
    // CHECK-SAME{LITERAL}:   dense<[[[[-7.72567116E-4, 0.00231770123, 0.00103008945], [0.00154513423, 0.00386283547, 0.00231770123], [0.00128761178, 2.57522363E-4, 7.72567116E-4]], [[-7.72567116E-4, 0.00360531313, 0.0020601789], [0.00154513423, 0.0056654918, 0.00231770123], [0.000000e+00, 2.57522363E-4, -7.72567116E-4]], [[2.57522363E-4, 0.0066955816, 0.00180265657], [0.0028327459, 0.00515044713, 7.72567116E-4], [0.00154513423, -0.00103008945, -0.00154513423]]], [[[-7.72567116E-4, 0.00231770123, 0.00103008945], [0.00154513423, 0.00386283547, 0.00231770123], [0.00128761178, 2.57522363E-4, 7.72567116E-4]], [[-7.72567116E-4, 0.00360531313, 0.0020601789], [0.00154513423, 0.0056654918, 0.00231770123], [0.000000e+00, 2.57522363E-4, -7.72567116E-4]], [[2.57522363E-4, 0.0066955816, 0.00180265657], [0.0028327459, 0.00515044713, 7.72567116E-4], [0.00154513423, -0.00103008945, -0.00154513423]]], [[[-7.72567116E-4, 0.00231770123, 0.00103008945], [0.00154513423, 0.00386283547, 0.00231770123], [0.00128761178, 2.57522363E-4, 7.72567116E-4]], [[-7.72567116E-4, 0.00360531313, 0.0020601789], [0.00154513423, 0.0056654918, 0.00231770123], [0.000000e+00, 2.57522363E-4, -7.72567116E-4]], [[2.57522363E-4, 0.0066955816, 0.00180265657], [0.0028327459, 0.00515044713, 7.72567116E-4], [0.00154513423, -0.00103008945, -0.00154513423]]], [[[-7.72567116E-4, 0.00231770123, 0.00103008945], [0.00154513423, 0.00386283547, 0.00231770123], [0.00128761178, 2.57522363E-4, 7.72567116E-4]], [[-7.72567116E-4, 0.00360531313, 0.0020601789], [0.00154513423, 0.0056654918, 0.00231770123], [0.000000e+00, 2.57522363E-4, -7.72567116E-4]], [[2.57522363E-4, 0.0066955816, 0.00180265657], [0.0028327459, 0.00515044713, 7.72567116E-4], [0.00154513423, -0.00103008945, -0.00154513423]]]]> : tensor<4x3x3x3xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST_1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x640x640xf16>, tensor<4x3x3x3xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x640x640xf16>
    // CHECK:       [[SIGMOID:%.+]] = IE.Sigmoid([[CONV]]) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>

    // CHECK: return [[SIGMOID]]
}

// -----

// CHECK-LABEL: @ReplaceFQU16WithReLU
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x4x640x640xf16>)
func.func @ReplaceFQU16WithReLU(%arg0: tensor<1x4x640x640xf16>) -> tensor<1x4x640x640xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<57.1374702> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x4x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x640x640xf16>
    %1 = IE.Sigmoid(%0) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>
    return %1 : tensor<1x4x640x640xf16>

    // CHECK: [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>
    // CHECK: [[SIGMOID:%.+]] = IE.Sigmoid([[RELU]]) : tensor<1x4x640x640xf16> -> tensor<1x4x640x640xf16>

    // CHECK: return [[SIGMOID]]
}

// -----

// With adaptive stripping disabled, the ParentFQ U16 -> ChildFQ U16 chain is not
// collapsed; each FQ is handled independently (parent dropped on il=ol, child -> ReLU).

// CHECK-LABEL: @DoNotChainCollapseAdaptiveStrippingDisabled
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x512x51x39xf16>)
func.func @DoNotChainCollapseAdaptiveStrippingDisabled(%arg0: tensor<1x512x51x39xf16>) -> tensor<1x512x51x39xf16> {
    %fq1_low  = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %fq1_high = const.Declare tensor<1x1x1x1xf16> = dense<12.7559032> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %fq2_low  = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %fq2_high = const.Declare tensor<1x1x1x1xf16> = dense<14.7559032> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %parent = IE.FakeQuantize(%arg0, %fq1_low, %fq1_high, %fq1_low, %fq1_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x512x51x39xf16>
    %child  = IE.FakeQuantize(%parent, %fq2_low, %fq2_high, %fq2_low, %fq2_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x512x51x39xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x512x51x39xf16>
    %sigmoid = IE.Sigmoid(%child) : tensor<1x512x51x39xf16> -> tensor<1x512x51x39xf16>
    return %sigmoid : tensor<1x512x51x39xf16>

    // With adaptive stripping disabled: parent FQ is dropped (il=ol so it is replaced
    // with its input), child FQ is per-tensor il=ol=0 so it is replaced with a ReLU.
    // CHECK:    [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x512x51x39xf16> -> tensor<1x512x51x39xf16>
    // CHECK:    [[SIGMOID:%.+]] = IE.Sigmoid([[RELU]]) : tensor<1x512x51x39xf16> -> tensor<1x512x51x39xf16>
    // CHECK:    return [[SIGMOID]]
}
