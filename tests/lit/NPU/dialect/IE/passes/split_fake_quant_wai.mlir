//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-fake-quant %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<i8:f16:1, {0.43933364269780179,0.24145462933708639,0.78572959151922483,0.58888816085516238,0.56745297301049324,0.93348080504174324,0.71047830020680147,0.15292347926719516,0.12209666383032705,0.099555625167547484,0.85396088244868262,0.038385636198754403,0.14808043685613895,0.25258062026079964,0.29266679053213079,0.59047199324065569}>
// CHECK-LABEL: @SplitFakeQuantForI8WeightsAsInputs

  func.func @SplitFakeQuantForI8WeightsAsInputs(%arg0: tensor<16x256xf32>, %arg1: tensor<16x256xsi8>) -> tensor<1x16x1x256xf16> {
    %cst = const.Declare tensor<1x16x1x1xf16> = dense<[[55.795372], [30.6647377], [99.7876586], [74.7887955], [72.0665283], [118.552063], [90.2307434], [19.4212818], [15.5062761], [12.6435642], [108.453033], [4.87497568], [18.8062153], [32.0777397], [37.1686821], [74.9899444]]> : tensor<16x1xf32>, [#const.Reshape<[1, 16, 1, 1]>, #const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<[[-56.2347069], [-30.9061928], [-100.573387], [-75.3776855], [-72.6339798], [-119.485542], [-90.9412231], [-19.5742054], [-15.6283731], [-12.7431202], [-109.306992], [-4.91336155], [-18.9542961], [-32.3303185], [-37.4613495], [-75.5804138]]> : tensor<16x1xf32>, [#const.Reshape<[1, 16, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %0 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 256]} : tensor<16x256xsi8> -> tensor<1x1x16x256xsi8>
    %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 256]} : tensor<16x256xf32> -> tensor<1x1x16x256xf32>
    %2 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x16x256xsi8> -> tensor<1x1x16x256xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 16, 1, 256]} : tensor<1x1x16x256xf16> -> tensor<1x16x1x256xf16>
    %4 = IE.FakeQuantize(%3, %cst_2, %cst_1, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x256xf16>
    return %4 : tensor<1x16x1x256xf16>

    // CHECK: [[VAL1:%.+]] = IE.Quantize([[VAL0:%.+]]) {dstElemType = !qElemType} : tensor<1x16x1x256xf16> -> tensor<1x16x1x256x!qElemType>
    // CHECK: [[VAL2:%.+]] = IE.QuantizeCast([[VAL1]]) {dstElemType = !qElemType1} : tensor<1x16x1x256x!qElemType> -> tensor<1x16x1x256x!qElemType1>
    // CHECK: [[VAL3:%.+]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x16x1x256x!qElemType1> -> tensor<1x16x1x256xf16>
    // CHECK: return [[VAL3]] : tensor<1x16x1x256xf16>
  }

// -----

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<i4:f16:1, {0.43933364550272624,0.24145463307698567,0.78572959899902339,0.58888816833496094,0.5674529711405436,0.9334808031717936,0.71047830581665039,0.15292348066965739,0.12209666570027669,0.099555627504984534,0.85396086374918622,0.038385637601216632,0.14808043638865154,0.25258061091105144,0.29266678492228188,0.59047196706136063}>
// CHECK-LABEL: @SplitFakeQuantForI4WeightsAsInputs

  func.func @SplitFakeQuantForI4WeightsAsInputs(%arg0: tensor<16x256xf32>, %arg1: tensor<16x256xsi4>) -> tensor<1x16x1x256xf16> {
    %cst = const.Declare tensor<1x16x1x1xf16> = dense<[[3.0753355], [1.69018245], [5.50010729], [4.12221718], [3.97217083], [6.53436565], [4.97334814], [1.07046437], [0.854676663], [0.6968894], [5.97772598], [0.268699467], [1.03656304], [1.76806426], [2.04866743], [4.13330364]]> : tensor<16x1xf32>, [#const.Reshape<[1, 16, 1, 1]>, #const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<[[-3.51466918], [-1.93163705], [-6.2858367], [-4.71110535], [-4.53962374], [-7.46784639], [-5.68382645], [-1.22338784], [-0.976773321], [-7.964450e-01], [-6.83168697], [-0.307085097], [-1.18464351], [-2.0206449], [-2.34133434], [-4.72377586]]> : tensor<16x1xf32>, [#const.Reshape<[1, 16, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<7.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-8.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %0 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 256]} : tensor<16x256xsi4> -> tensor<1x1x16x256xsi4>
    %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 256]} : tensor<16x256xf32> -> tensor<1x1x16x256xf32>
    %2 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x16x256xsi4> -> tensor<1x1x16x256xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 16, 1, 256]} : tensor<1x1x16x256xf16> -> tensor<1x16x1x256xf16>
    %4 = IE.FakeQuantize(%3, %cst_2, %cst_1, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x16x1x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x256xf16>
    return %4 : tensor<1x16x1x256xf16>

    // CHECK: [[VAL1:%.+]] = IE.Quantize([[VAL0:%.+]]) {dstElemType = !qElemType} : tensor<1x16x1x256xf16> -> tensor<1x16x1x256x!qElemType>
    // CHECK: [[VAL2:%.+]] = IE.QuantizeCast([[VAL1]]) {dstElemType = !qElemType1} : tensor<1x16x1x256x!qElemType> -> tensor<1x16x1x256x!qElemType1>
    // CHECK: [[VAL3:%.+]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x16x1x256x!qElemType1> -> tensor<1x16x1x256xf16>
    // CHECK: return [[VAL3]] : tensor<1x16x1x256xf16>
  }

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<u8:f16:1, {0.43933364269780179:128,0.24145462933708639:128,0.78572959151922483:128,0.58888816085516238:128,0.56745297301049324:128,0.93348080504174324:128,0.71047830020680147:128,0.15292347926719516:128,0.12209666383032705:128,0.099555625167547484:128,0.85396088244868262:128,0.038385636198754403:128,0.14808043685613895:128,0.25258062026079964:128,0.29266679053213079:128,0.59047199324065569:128}>
// CHECK-LABEL: @SplitFakeQuantForU8WeightsAsInputs

  func.func @SplitFakeQuantForU8WeightsAsInputs(%arg0: tensor<16x256xf32>, %arg1: tensor<16x256xui8>) -> tensor<1x16x1x256xf16> {
    %cst = const.Declare tensor<1x16x1x1xf16> = dense<[[55.795372], [30.6647377], [99.7876586], [74.7887955], [72.0665283], [118.552063], [90.2307434], [19.4212818], [15.5062761], [12.6435642], [108.453033], [4.87497568], [18.8062153], [32.0777397], [37.1686821], [74.9899444]]> : tensor<16x1xf32>, [#const.Reshape<[1, 16, 1, 1]>, #const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<[[-56.2347069], [-30.9061928], [-100.573387], [-75.3776855], [-72.6339798], [-119.485542], [-90.9412231], [-19.5742054], [-15.6283731], [-12.7431202], [-109.306992], [-4.91336155], [-18.9542961], [-32.3303185], [-37.4613495], [-75.5804138]]> : tensor<16x1xf32>, [#const.Reshape<[1, 16, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %0 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 256]} : tensor<16x256xui8> -> tensor<1x1x16x256xui8>
    %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 256]} : tensor<16x256xf32> -> tensor<1x1x16x256xf32>
    %2 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x16x256xui8> -> tensor<1x1x16x256xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 16, 1, 256]} : tensor<1x1x16x256xf16> -> tensor<1x16x1x256xf16>
    %4 = IE.FakeQuantize(%3, %cst_2, %cst_1, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x1x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x256xf16>
    return %4 : tensor<1x16x1x256xf16>

    // CHECK: [[VAL1:%.+]] = IE.Quantize([[VAL0:%.+]]) {dstElemType = !qElemType} : tensor<1x16x1x256xf16> -> tensor<1x16x1x256x!qElemType>
    // CHECK: [[VAL2:%.+]] = IE.QuantizeCast([[VAL1]]) {dstElemType = !qElemType1} : tensor<1x16x1x256x!qElemType> -> tensor<1x16x1x256x!qElemType1>
    // CHECK: [[VAL3:%.+]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x16x1x256x!qElemType1> -> tensor<1x16x1x256xf16>
    // CHECK: return [[VAL3]] : tensor<1x16x1x256xf16>
  }

// -----

// CHECK: !qElemType = !quant.uniform<u4:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<u4:f16:1, {0.43933364550272624:8,0.24145463307698567:8,0.78572959899902339:8,0.58888816833496094:8,0.5674529711405436:8,0.9334808031717936:8,0.71047830581665039:8,0.15292348066965739:8,0.12209666570027669:8,0.099555627504984534:8,0.85396086374918622:8,0.038385637601216632:8,0.14808043638865154:8,0.25258061091105144:8,0.29266678492228188:8,0.59047196706136063:8}>
// CHECK-LABEL: @SplitFakeQuantForU4WeightsAsInputs

  func.func @SplitFakeQuantForU4WeightsAsInputs(%arg0: tensor<16x256xf32>, %arg1: tensor<16x256xui4>) -> tensor<1x16x1x256xf16> {
    %cst = const.Declare tensor<1x16x1x1xf16> = dense<[[3.0753355], [1.69018245], [5.50010729], [4.12221718], [3.97217083], [6.53436565], [4.97334814], [1.07046437], [0.854676663], [0.6968894], [5.97772598], [0.268699467], [1.03656304], [1.76806426], [2.04866743], [4.13330364]]> : tensor<16x1xf32>, [#const.Reshape<[1, 16, 1, 1]>, #const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x16x1x1xf16> = dense<[[-3.51466918], [-1.93163705], [-6.2858367], [-4.71110535], [-4.53962374], [-7.46784639], [-5.68382645], [-1.22338784], [-0.976773321], [-7.964450e-01], [-6.83168697], [-0.307085097], [-1.18464351], [-2.0206449], [-2.34133434], [-4.72377586]]> : tensor<16x1xf32>, [#const.Reshape<[1, 16, 1, 1]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.500000e+01> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %0 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 256]} : tensor<16x256xui4> -> tensor<1x1x16x256xui4>
    %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 256]} : tensor<16x256xf32> -> tensor<1x1x16x256xf32>
    %2 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x16x256xui4> -> tensor<1x1x16x256xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 16, 1, 256]} : tensor<1x1x16x256xf16> -> tensor<1x16x1x256xf16>
    %4 = IE.FakeQuantize(%3, %cst_2, %cst_1, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x16x1x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x256xf16>
    return %4 : tensor<1x16x1x256xf16>

    // CHECK: [[VAL1:%.+]] = IE.Quantize([[VAL0:%.+]]) {dstElemType = !qElemType} : tensor<1x16x1x256xf16> -> tensor<1x16x1x256x!qElemType>
    // CHECK: [[VAL2:%.+]] = IE.QuantizeCast([[VAL1]]) {dstElemType = !qElemType1} : tensor<1x16x1x256x!qElemType> -> tensor<1x16x1x256x!qElemType1>
    // CHECK: [[VAL3:%.+]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x16x1x256x!qElemType1> -> tensor<1x16x1x256xf16>
    // CHECK: return [[VAL3]] : tensor<1x16x1x256xf16>
  }

// -----

#NWHC = affine_map < (d0, d1, d2, d3)->(d0, d3, d2, d1)>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<i8:f16:3,
// {0.43933364269780179,0.24145462933708639,0.78572959151922483,0.58888816085516238,0.56745297301049324,0.93348080504174324,0.71047830020680147,0.15292347926719516,0.12209666383032705,0.099555625167547484,0.85396088244868262,0.038385636198754403,0.14808043685613895,0.25258062026079964,0.29266679053213079,0.59047199324065569}>
// CHECK-LABEL: @SplitFakeQuantForI8WeightsAsInputsWithTranspose
// CHECK-SAME:  [[INPUT:%.+]]: tensor<16x256xsi8>
func.func @SplitFakeQuantForI8WeightsAsInputsWithTranspose(%arg0 : tensor<16x256xsi8>)->tensor<1x256x1x16xf16> {
    %cst = const.Declare tensor<1x1x1x16xf16> = dense<[[55.795372, 30.6647377, 99.7876586, 74.7887955, 72.0665283, 118.552063, 90.2307434, 19.4212818, 15.5062761, 12.6435642, 108.453033, 4.87497568, 18.8062153, 32.0777397, 37.1686821, 74.9899444]]> : tensor<1x16xf32>, [#const.Reshape<[1, 1, 1, 16]>, #const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x16xf16> = dense<[[-56.2347069, -30.9061928, -100.573387, -75.3776855, -72.6339798, -119.485542, -90.9412231, -19.5742054, -15.6283731, -12.7431202, -109.306992, -4.91336155, -18.9542961, -32.3303185, -37.4613495, -75.5804138]]> : tensor<1x16xf32>, [#const.Reshape<[1, 1, 1, 16]>, #const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 256]} : tensor<16x256xsi8> -> tensor<1x1x16x256xsi8>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x16x256xsi8> -> tensor<1x1x16x256xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 16, 1, 256]} : tensor<1x1x16x256xf16> -> tensor<1x16x1x256xf16>
    %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} : tensor<1x16x1x256xf16> -> tensor<1x256x1x16xf16>
    %4 = IE.FakeQuantize(%3, %cst_2, %cst_1, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x256x1x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x16xf16>, tensor<1x1x1x16xf16> -> tensor<1x256x1x16xf16>

    return %4 : tensor<1x256x1x16xf16>

    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 256]} : tensor<16x256xsi8> -> tensor<1x1x16x256xsi8>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[RESHAPE0]]) {dstElemType = f16} : tensor<1x1x16x256xsi8> -> tensor<1x1x16x256xf16>
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[CONVERT]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [1, 16, 1, 256]} : tensor<1x1x16x256xf16> -> tensor<1x16x1x256xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE1]]) {order_value = #NWHC} : tensor<1x16x1x256xf16> -> tensor<1x256x1x16xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[TRANSPOSE]]) {dstElemType = !qElemType} : tensor<1x256x1x16xf16> -> tensor<1x256x1x16x!qElemType>
    // CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = !qElemType1} : tensor<1x256x1x16x!qElemType> -> tensor<1x256x1x16x!qElemType1>
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<1x256x1x16x!qElemType1> -> tensor<1x256x1x16xf16>

    // CHECK: return [[DEQUANTIZE]] : tensor<1x256x1x16xf16>
}
