//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
func.func @FuseFQ(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    %1 = IE.FakeQuantize(%0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>
    // CHECK-DAG:   %[[ILOW:.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG:   %[[IHIGH:.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%arg0, %[[ILOW]], %[[IHIGH]], %[[ILOW]], %[[IHIGH]])

    // CHECK-NOT:   IE.FakeQuantize

    // CHECK:       return %[[FQ]]
}

// -----

func.func @DoNotFuseFQ(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high_1 = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %input_high_2 = const.Declare tensor<f32> = dense<128.0> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high_1, %input_low, %input_high_1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    %1 = IE.FakeQuantize(%0, %input_low, %input_high_1, %input_low, %input_high_2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x16x16xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>
    // CHECK-DAG:   %[[ILOW:.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG:   %[[IHIGH1:.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG:   %[[IHIGH2:.*]] = const.Declare tensor<f32> = dense<1.280000e+02> : tensor<f32>

    // CHECK:   %[[FQ1:.*]] = IE.FakeQuantize(%arg0, %[[ILOW]], %[[IHIGH1]], %[[ILOW]], %[[IHIGH1]])

    // CHECK:   %[[FQ2:.*]] = IE.FakeQuantize(%[[FQ1]], %[[ILOW]], %[[IHIGH1]], %[[ILOW]], %[[IHIGH2]])

    // CHECK:       return %[[FQ2]]
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @TransposeGroups
func.func @TransposeGroups() -> tensor<1280x20x128xf32> {
    %DATA = const.Declare tensor<1280x20x128xf32> = dense<4.500000e+01>  : tensor<1280x20x128xf32>
    // CHECK-DAG:   [[DATA:%.*]] = const.Declare tensor<128x1280x20xf32> = dense<4.500000e+01> : tensor<1280x20x128xf32>, [#const.Transpose<#map>]

    %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<0.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Transpose<#map>]

    %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<15.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>, [#const.Transpose<#map>]

    %OUT_LOW = const.Declare tensor<1280x20x1xf32> = dense<-16.0>  : tensor<1280x20x1xf32>
    // CHECK-DAG:   [[OUT_LOW:%.*]] = const.Declare tensor<1x1280x20xf32> = dense<-1.600000e+01> : tensor<1280x20x1xf32>, [#const.Transpose<#map>]

    %OUT_HIGH = const.Declare tensor<1280x20x1xf32> = dense<14.0>  : tensor<1280x20x1xf32>
    // CHECK-DAG:   [[OUT_HIGH:%.*]] = const.Declare tensor<1x1280x20xf32> = dense<1.400000e+01> : tensor<1280x20x1xf32>, [#const.Transpose<#map>]

    %FQ = IE.FakeQuantize(%DATA, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 16 : i64
    } : tensor<1280x20x128xf32>,
        tensor<1x1x1xf32>,
        tensor<1x1x1xf32>,
        tensor<1280x20x1xf32>,
        tensor<1280x20x1xf32> -> tensor<1280x20x128xf32>

    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 16 : i64
    // CHECK-SAME:  } : tensor<128x1280x20xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1280x20xf32>, tensor<1x1280x20xf32> -> tensor<128x1280x20xf32>

    // CHECK:   [[TRANSPOSE:%.*]] = IE.Transpose([[FQ]]) {
    // CHECK-SAME:      order_value = #HWC
    // CHECK-SAME:  } : tensor<128x1280x20xf32> -> tensor<1280x20x128xf32>

    return %FQ : tensor<1280x20x128xf32>
    // CHECK:   return [[TRANSPOSE]] : tensor<1280x20x128xf32>
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @PropagateGroupsReshapeLastTwoDims
func.func @PropagateGroupsReshapeLastTwoDims() -> tensor<1280x2560xf32> {
    %DATA = const.Declare tensor<1280x20x128xf32> = dense<4.500000e+01>  : tensor<1280x20x128xf32>
    // CHECK-DAG:   [[DATA:%.*]] = const.Declare tensor<20x128x1280xf32> = dense<4.500000e+01> : tensor<1280x20x128xf32>, [#const.Transpose<#HWC>]

    %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<0.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Transpose<#HWC>]

    %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<15.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>, [#const.Transpose<#HWC>]

    %OUT_LOW = const.Declare tensor<1280x20x1xf32> = dense<-16.0>  : tensor<1280x20x1xf32>
    // CHECK-DAG:   [[OUT_LOW:%.*]] = const.Declare tensor<20x1x1280xf32> = dense<-1.600000e+01> : tensor<1280x20x1xf32>, [#const.Transpose<#HWC>]

    %OUT_HIGH = const.Declare tensor<1280x20x1xf32> = dense<14.0>  : tensor<1280x20x1xf32>
    // CHECK-DAG:   [[OUT_HIGH:%.*]] = const.Declare tensor<20x1x1280xf32> = dense<1.400000e+01> : tensor<1280x20x1xf32>, [#const.Transpose<#HWC>]

    %FQ = IE.FakeQuantize(%DATA, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 16 : i64
    } : tensor<1280x20x128xf32>,
        tensor<1x1x1xf32>,
        tensor<1x1x1xf32>,
        tensor<1280x20x1xf32>,
        tensor<1280x20x1xf32> -> tensor<1280x20x128xf32>

    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 16 : i64
    // CHECK-SAME:  } : tensor<20x128x1280xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<20x1x1280xf32>, tensor<20x1x1280xf32> -> tensor<20x128x1280xf32>

    // CHECK:   [[TRANSPOSE:%.*]] = IE.Transpose([[FQ]]) {
    // CHECK-SAME:      order_value = #map
    // CHECK-SAME:  } : tensor<20x128x1280xf32> -> tensor<1280x20x128xf32>

    %RESHAPE = IE.AffineReshape(%FQ) {
        dim_mapping = [[0], [1], [1]],
        shape_value = [1280, 2560]
    } : tensor<1280x20x128xf32> -> tensor<1280x2560xf32>

    // CHECK:   [[RESHAPE:%.*]] = IE.AffineReshape([[TRANSPOSE]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [1], [1]],
    // CHECK-SAME:          shape_value = [1280, 2560]
    // CHECK-SAME:  } : tensor<1280x20x128xf32> -> tensor<1280x2560xf32>

    return %RESHAPE : tensor<1280x2560xf32>
    // CHECK:   return [[RESHAPE]] : tensor<1280x2560xf32>
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @PropagateGroupsReshapeFirstTwoDims
func.func @PropagateGroupsReshapeFirstTwoDims() -> tensor<5120x128xf32> {
    %DATA = const.Declare tensor<256x20x128xf32> = dense<4.500000e+01> : tensor<256x20x128xf32>
    // CHECK-DAG:   [[DATA:%.*]] = const.Declare tensor<128x256x20xf32> = dense<4.500000e+01> : tensor<256x20x128xf32>, [#const.Transpose<#map>]

    %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<0.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.Transpose<#map>]

    %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<15.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>, [#const.Transpose<#map>]

    %OUT_LOW = const.Declare tensor<256x1x128xf32> = dense<-16.0> : tensor<256x1x128xf32>
    // CHECK-DAG:   [[OUT_LOW:%.*]] = const.Declare tensor<128x256x1xf32> = dense<-1.600000e+01> : tensor<256x1x128xf32>, [#const.Transpose<#map>]

    %OUT_HIGH = const.Declare tensor<256x1x128xf32> = dense<14.0> : tensor<256x1x128xf32>
    // CHECK-DAG:   [[OUT_HIGH:%.*]] = const.Declare tensor<128x256x1xf32> = dense<1.400000e+01> : tensor<256x1x128xf32>, [#const.Transpose<#map>]

    %FQ = IE.FakeQuantize(%DATA, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 16 : i64
    } : tensor<256x20x128xf32>,
        tensor<1x1x1xf32>,
        tensor<1x1x1xf32>,
        tensor<256x1x128xf32>,
        tensor<256x1x128xf32> -> tensor<256x20x128xf32>

    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 16 : i64
    // CHECK-SAME:  } : tensor<128x256x20xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<128x256x1xf32>, tensor<128x256x1xf32> -> tensor<128x256x20xf32>

    // CHECK:   [[TRANSPOSE:%.*]] = IE.Transpose([[FQ]]) {
    // CHECK-SAME:      order_value = #HWC
    // CHECK-SAME:  } : tensor<128x256x20xf32> -> tensor<256x20x128xf32>

    %RESHAPE = IE.AffineReshape(%FQ) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [5120, 128]
    } : tensor<256x20x128xf32> -> tensor<5120x128xf32>

    // CHECK:   [[RESHAPE:%.*]] = IE.AffineReshape([[TRANSPOSE]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [0], [1]],
    // CHECK-SAME:          shape_value = [5120, 128]
    // CHECK-SAME:  } : tensor<256x20x128xf32> -> tensor<5120x128xf32>

    return %RESHAPE : tensor<5120x128xf32>
    // CHECK:   return [[RESHAPE]] : tensor<5120x128xf32>
}

// -----

// CHECK-LABEL: @NotBeneficialToTranspose
func.func @NotBeneficialToTranspose() -> tensor<64x25600xf32> {
    %DATA = const.Declare tensor<64x200x128xf32> = dense<4.500000e+01>  : tensor<64x200x128xf32>
    // CHECK-DAG:   [[DATA:%.*]] = const.Declare tensor<64x200x128xf32> = dense<4.500000e+01>  : tensor<64x200x128xf32>

    %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>

    %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>

    %OUT_LOW = const.Declare tensor<64x200x1xf32> = dense<-1.600000e+01>  : tensor<64x200x1xf32>
    // CHECK-DAG:   [[OUT_LOW:%.*]] = const.Declare tensor<64x200x1xf32> = dense<-1.600000e+01>  : tensor<64x200x1xf32>

    %OUT_HIGH = const.Declare tensor<64x200x1xf32> = dense<1.400000e+01>  : tensor<64x200x1xf32>
    // CHECK-DAG:   [[OUT_HIGH:%.*]] = const.Declare tensor<64x200x1xf32> = dense<1.400000e+01>  : tensor<64x200x1xf32>

    %FQ = IE.FakeQuantize(%DATA, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 16 : i64
    } : tensor<64x200x128xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<64x200x1xf32>, tensor<64x200x1xf32> -> tensor<64x200x128xf32>
    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-NOT:   IE.Transpose

    %RESHAPE = IE.AffineReshape(%FQ) {
        dim_mapping = [[0], [1], [1]],
        shape_value = [64, 25600]
    } : tensor<64x200x128xf32> -> tensor<64x25600xf32>
    // CHECK:   [[RESHAPE:%.*]] = IE.AffineReshape([[FQ]])

    return %RESHAPE : tensor<64x25600xf32>
    // CHECK:   return [[RESHAPE]] : tensor<64x25600xf32>
}

// -----

// CHECK-LABEL: @NotBeneficialToTranspose4dData
func.func @NotBeneficialToTranspose4dData() -> tensor<512x64x200x128xf32> {
    %DATA = const.Declare tensor<512x64x200x128xf32> = dense<4.500000e+01>  : tensor<512x64x200x128xf32>
    // CHECK-DAG:   [[DATA:%.*]] = const.Declare tensor<512x64x200x128xf32> = dense<4.500000e+01>  : tensor<512x64x200x128xf32>

    %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>

    %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>

    %OUT_LOW = const.Declare tensor<64x200x1xf32> = dense<-1.600000e+01>  : tensor<64x200x1xf32>
    // CHECK-DAG:   [[OUT_LOW:%.*]] = const.Declare tensor<64x200x1xf32> = dense<-1.600000e+01>  : tensor<64x200x1xf32>

    %OUT_HIGH = const.Declare tensor<64x200x1xf32> = dense<1.400000e+01>  : tensor<64x200x1xf32>
    // CHECK-DAG:   [[OUT_HIGH:%.*]] = const.Declare tensor<64x200x1xf32> = dense<1.400000e+01>  : tensor<64x200x1xf32>

    %FQ = IE.FakeQuantize(%DATA, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 16 : i64
    } : tensor<512x64x200x128xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<64x200x1xf32>, tensor<64x200x1xf32> -> tensor<512x64x200x128xf32>
    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-NOT:   IE.Transpose

    return %FQ : tensor<512x64x200x128xf32>
    // CHECK:   return [[FQ]] : tensor<512x64x200x128xf32>
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @TransposeGroupsWAI
// CHECK-SAME: ([[ARG:%.+]]: tensor<1280x20x128xf32>)
func.func @TransposeGroupsWAI(%arg0: tensor<1280x20x128xf32>) -> tensor<1280x2560xf32> {

    %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<0.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>

    %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<15.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>

    %OUT_LOW = const.Declare tensor<1280x20x1xf32> = dense<-16.0>  : tensor<1280x20x1xf32>
    // CHECK-DAG:   [[OUT_LOW:%.*]] = const.Declare tensor<1280x20x1xf32> = dense<-1.600000e+01> : tensor<1280x20x1xf32>

    %OUT_HIGH = const.Declare tensor<1280x20x1xf32> = dense<14.0>  : tensor<1280x20x1xf32>
    // CHECK-DAG:   [[OUT_HIGH:%.*]] = const.Declare tensor<1280x20x1xf32> = dense<1.400000e+01> : tensor<1280x20x1xf32>

    // CHECK-NOT:    IE.Transpose

    %FQ = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 16 : i64
    } : tensor<1280x20x128xf32>,
        tensor<1x1x1xf32>,
        tensor<1x1x1xf32>,
        tensor<1280x20x1xf32>,
        tensor<1280x20x1xf32> -> tensor<1280x20x128xf32>

    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize([[ARG]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 16 : i64
    // CHECK-SAME:  } : tensor<1280x20x128xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1280x20x1xf32>, tensor<1280x20x1xf32> -> tensor<1280x20x128xf32>

    // CHECK-NOT:    IE.Transpose
    
    %RESHAPE = IE.AffineReshape(%FQ) {
        dim_mapping = [[0], [1], [1]],
        shape_value = [1280, 2560]
    } : tensor<1280x20x128xf32> -> tensor<1280x2560xf32>

    // CHECK:   [[RESHAPE:%.*]] = IE.AffineReshape([[FQ]]) {

    return %RESHAPE : tensor<1280x2560xf32>
    // CHECK:   return [[RESHAPE]] : tensor<1280x2560xf32>
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @WAIConvertDoNotInsertTranspose
// CHECK-SAME: ([[ARG:%.+]]: tensor<1280x20x128xui4>)
func.func @WAIConvertDoNotInsertTranspose(%arg0: tensor<1280x20x128xui4>) -> tensor<1280x2560xf32> {

    %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<0.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>

    %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<15.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>

    %OUT_LOW = const.Declare tensor<1280x20x1xf32> = dense<-16.0>  : tensor<1280x20x1xf32>
    // CHECK-DAG:   [[OUT_LOW:%.*]] = const.Declare tensor<1280x20x1xf32> = dense<-1.600000e+01> : tensor<1280x20x1xf32>

    %OUT_HIGH = const.Declare tensor<1280x20x1xf32> = dense<14.0>  : tensor<1280x20x1xf32>
    // CHECK-DAG:   [[OUT_HIGH:%.*]] = const.Declare tensor<1280x20x1xf32> = dense<1.400000e+01> : tensor<1280x20x1xf32>
    
    // CHECK:   [[CONVERT:%.*]] = IE.Convert([[ARG]]) {
    %CONVERT = IE.Convert(%arg0) {dstElemType = f32} : tensor<1280x20x128xui4> -> tensor<1280x20x128xf32>

    // CHECK-NOT:    IE.Transpose

    %FQ = IE.FakeQuantize(%CONVERT, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 16 : i64
    } : tensor<1280x20x128xf32>,
        tensor<1x1x1xf32>,
        tensor<1x1x1xf32>,
        tensor<1280x20x1xf32>,
        tensor<1280x20x1xf32> -> tensor<1280x20x128xf32>

    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 16 : i64
    // CHECK-SAME:  } : tensor<1280x20x128xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1280x20x1xf32>, tensor<1280x20x1xf32> -> tensor<1280x20x128xf32>

    // CHECK-NOT:    IE.Transpose
    
    %RESHAPE = IE.AffineReshape(%FQ) {
        dim_mapping = [[0], [1], [1]],
        shape_value = [1280, 2560]
    } : tensor<1280x20x128xf32> -> tensor<1280x2560xf32>

    // CHECK:   [[RESHAPE:%.*]] = IE.AffineReshape([[FQ]]) {

    return %RESHAPE : tensor<1280x2560xf32>
    // CHECK:   return [[RESHAPE]] : tensor<1280x2560xf32>
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @WAINoInsertTranspose
// CHECK-SAME: ([[ARG:%.+]]: tensor<1280x2560xf32>)
func.func @WAINoInsertTranspose(%arg0: tensor<1280x2560xf32>) -> tensor<1280x2560xf32> {

    %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<0.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>

    %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<15.0> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1xf32>

    %OUT_LOW = const.Declare tensor<1280x20x1xf32> = dense<-16.0>  : tensor<1280x20x1xf32>
    // CHECK-DAG:   [[OUT_LOW:%.*]] = const.Declare tensor<1280x20x1xf32> = dense<-1.600000e+01> : tensor<1280x20x1xf32>

    %OUT_HIGH = const.Declare tensor<1280x20x1xf32> = dense<14.0>  : tensor<1280x20x1xf32>
    // CHECK-DAG:   [[OUT_HIGH:%.*]] = const.Declare tensor<1280x20x1xf32> = dense<1.400000e+01> : tensor<1280x20x1xf32>
    
    // CHECK:   [[RESHAPE1:%.*]] = IE.AffineReshape([[ARG]])
    %RESHAPE1 = IE.AffineReshape(%arg0) {
        dim_mapping = [[0], [1, 2]],
        shape_value = [1280, 20, 128]
    } : tensor<1280x2560xf32> -> tensor<1280x20x128xf32>

    // CHECK-NOT:   IE.Transpose

    %FQ = IE.FakeQuantize(%RESHAPE1, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 16 : i64
    } : tensor<1280x20x128xf32>,
        tensor<1x1x1xf32>,
        tensor<1x1x1xf32>,
        tensor<1280x20x1xf32>,
        tensor<1280x20x1xf32> -> tensor<1280x20x128xf32>

    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize([[RESHAPE1]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 16 : i64
    // CHECK-SAME:  } : tensor<1280x20x128xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1280x20x1xf32>, tensor<1280x20x1xf32> -> tensor<1280x20x128xf32>

    // CHECK-NOT:   IE.Transpose
    
    %RESHAPE2 = IE.AffineReshape(%FQ) {
        dim_mapping = [[0], [1], [1]],
        shape_value = [1280, 2560]
    } : tensor<1280x20x128xf32> -> tensor<1280x2560xf32>

    // CHECK:   [[RESHAPE2:%.*]] = IE.AffineReshape([[FQ]]) {

    return %RESHAPE2 : tensor<1280x2560xf32>
    // CHECK:   return [[RESHAPE2]] : tensor<1280x2560xf32>
}

