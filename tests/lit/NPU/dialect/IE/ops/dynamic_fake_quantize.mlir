//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL:  func.func @DynamicFQWeights
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x16x16xf16>)
func.func @DynamicFQWeights(%input: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %zp = const.Declare tensor<1x1x1x1xi4> = dense<1.0> : tensor<1x1x1x1xf16>,
            [#const.CastElemType<i4>]

    %0 = IE.DynamicFakeQuantize(%input, %scale, %zp)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256} :
        tensor<1x3x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xi4> -> tensor<1x3x16x16xf16>

    return %0 : tensor<1x3x16x16xf16>

    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[ZP:%.+]] = const.Declare tensor<1x1x1x1xi4> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<i4>]
    // CHECK:       [[DYNAMICFQ:%.+]] = IE.DynamicFakeQuantize([[INPUT]], [[SCALE]], [[ZP]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xi4> -> tensor<1x3x16x16xf16>
    // CHECK:       return [[DYNAMICFQ]] : tensor<1x3x16x16xf16>
}

// -----

// CHECK-LABEL:  func.func @DynamicFQWeightsInScale
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x3x16x16xf16>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x3x1x1xf16>
func.func @DynamicFQWeightsInScale(%input: tensor<1x3x16x16xf16>, %scale: tensor<1x3x1x1xf16>) -> tensor<1x3x16x16xf16> {
    %zp = const.Declare tensor<1x3x16x16xi4> = dense<1.0> : tensor<1x3x16x16xf16>,
            [#const.CastElemType<i4>]

    %0 = IE.DynamicFakeQuantize(%input, %scale, %zp)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256} :
        tensor<1x3x16x16xf16>, tensor<1x3x1x1xf16>, tensor<1x3x16x16xi4> -> tensor<1x3x16x16xf16>

    return %0 : tensor<1x3x16x16xf16>

    // CHECK:       [[ZP:%.+]] = const.Declare tensor<1x3x16x16xi4> = dense<1.000000e+00> : tensor<1x3x16x16xf16>, [#const.CastElemType<i4>]
    // CHECK:       [[DYNAMICFQ:%.+]] = IE.DynamicFakeQuantize([[INPUT]], [[SCALE]], [[ZP]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<1x3x1x1xf16>, tensor<1x3x16x16xi4> -> tensor<1x3x16x16xf16>
    // CHECK:       return [[DYNAMICFQ]] : tensor<1x3x16x16xf16>
}
