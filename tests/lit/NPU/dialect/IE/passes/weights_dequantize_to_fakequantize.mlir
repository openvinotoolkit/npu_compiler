//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --weights-dequantize-to-fake-quantize="enable-wd-blockarg-input=true" --mlir-print-elementsattrs-with-hex-if-larger -1 %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @WeightsMultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @WeightsMultToFakeQuantize(%input: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, -70], [36, 72, -82]], [[31, -67, 22], [-70, -55, 12], [-99, 42, 90]], [[6, -18, 95], [-8, -37, -64], [40, 31, -41]], [[35, -2, -98], [-94, -60, -68], [-3, -39, 88]]], [[[-43, -95, 64], [46, -125, -63], [-21, -25, -25]], [[-118, -103, -12], [84, 67, 55], [-105, 13, -10]], [[97, -124, 39], [-28, -112, 116], [74, 104, 72]], [[14, 58, 0], [37, -48, 26], [33, -64, 53]]], [[[-124, 104, 105], [-14, 0, -25], [104, -46, -87]], [[87, -105, 69], [94, 88, 47], [53, 93, -34]], [[-62, -44, -10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, -50, -39], [-108, 7, -12]]], [[[-73, -47, 7], [72, -17, 90], [-113, 44, 80]], [[-60, -102, -79], [-111, -43, 68], [-21, 53, 120]], [[-109, -69, 30], [120, -7, 107], [-30, 42, 66]], [[43, 16, -57], [95, 125, -99], [-30, 1, 126]]]]> : tensor<4x4x3x3xsi8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.00294781756]]], [[[0.00312666874]]], [[[0.00260377093]]], [[[0.00269700377]]]]> : tensor<4x1x1x1xf32>
  %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411> : tensor<1x1x1x1xf32>
  %0 = IE.FakeQuantize(%input, %cst_3, %cst_2, %cst_3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%0, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK: [[ACT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411>
  // CHECK: [[ACT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102>

  // CHECK: [[WT_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02>
  // CHECK: [[WT_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02>
  // CHECK: [[WT_OUT_LOW:%.+]] = const.Declare tensor<4x1x1x1xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[-0.377320647]]], [[[-0.400213599]]], [[[-0.333282679]]], [[[-0.345216483]]]]>
  // CHECK: [[WT_OUT_HIGH:%.+]] = const.Declare tensor<4x1x1x1xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[0.37437284]]], [[[0.397086918]]], [[[0.33067891]]], [[[0.342519492]]]]>

  // CHECK: [[WT_DATA:%.+]] = const.Declare tensor<4x4x3x3xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[73, 69, 95], [47, 85, -70], [36, 72, -82]], [[31, -67, 22], [-70, -55, 12], [-99, 42, 90]], [[6, -18, 95], [-8, -37, -64], [40, 31, -41]], [[35, -2, -98], [-94, -60, -68], [-3, -39, 88]]], [[[-43, -95, 64], [46, -125, -63], [-21, -25, -25]], [[-118, -103, -12], [84, 67, 55], [-105, 13, -10]], [[97, -124, 39], [-28, -112, 116], [74, 104, 72]], [[14, 58, 0], [37, -48, 26], [33, -64, 53]]], [[[-124, 104, 105], [-14, 0, -25], [104, -46, -87]], [[87, -105, 69], [94, 88, 47], [53, 93, -34]], [[-62, -44, -10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, -50, -39], [-108, 7, -12]]], [[[-73, -47, 7], [72, -17, 90], [-113, 44, 80]], [[-60, -102, -79], [-111, -43, 68], [-21, 53, 120]], [[-109, -69, 30], [120, -7, 107], [-30, 42, 66]], [[43, 16, -57], [95, 125, -99], [-30, 1, 126]]]]>

  // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[WT_DATA]], [[WT_IN_LOW]], [[WT_IN_HIGH]], [[WT_OUT_LOW]], [[WT_OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<4x4x3x3xf32>

  // CHECK: [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<1x4x28x28xf32>

  // CHECK: [[CONV:%.+]] = IE.Convolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsMultSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x32x32xf32>
// CHECK-SAME: -> tensor<1x16x32x32xf32>
func.func @WeightsMultSubToFakeQuantize(%input: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %cst_2 = const.Declare tensor<16x1x1x1xf32> = dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]> : tensor<16x1x1x1xsi8>, [#const.CastElemType<f32>]
  %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<2.500000e+01> : tensor<1x1x1x1xf32>
  %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<0.0566197559> : tensor<1x1x1x1xf32>
  %0 = IE.FakeQuantize(%input, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  %1 = IE.Subtract(%cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  %2 = IE.Multiply(%1, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>
  return %3 : tensor<1x16x32x32xf32>

  // CHECK: [[ACT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158>
  // CHECK: [[ACT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>

  // CHECK: [[WT_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02>
  // CHECK: [[WT_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02>
  // CHECK: [[WT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-8.66282272>
  // CHECK: [[WT_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.77521514>

  // CHECK: [[WT_DATA:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]>

  // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[WT_DATA]], [[WT_IN_LOW]], [[WT_IN_HIGH]], [[WT_OUT_LOW]], [[WT_OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<16x1x1x1xf32>

  // CHECK: [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<1x16x32x32xf32>

  // CHECK: [[GROUP_CONV:%.+]] = IE.GroupConvolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>

  // CHECK: return [[GROUP_CONV]]
}

// -----

// CHECK-LABEL: @WeightsMultSubToNon4DFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x48xf16>
// CHECK-SAME: -> tensor<1x4x48xf16>
func.func @WeightsMultSubToNon4DFakeQuantize(%input: tensor<1x4x48xf16>) -> tensor<1x4x48xf16> {
  %cst_0 = const.Declare tensor<48xf16> = dense<[9, -5, -25, 52, -77, -123, 24, 67, -32, 11, -24, 93, -17, -127, -46, -38, 53, -88, -108, -60, 9, -8, -78, 106, -33, 14, -11, -21, -94, -72, 49, 125, -58, -93, 91, 44, 123, -99, -59, 15, -124, -13, 89, -92, -97, 10, -16, 38]> : tensor<48xsi8>, [#const.CastElemType<f16>]
  %cst_1 = const.Declare tensor<1xf16> = dense<8.800000e+01> : tensor<1xf16>
  %cst_2 = const.Declare tensor<1xf16> = dense<9.88533836E-4> : tensor<1xf16>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %2 = IE.Add(%input, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  return %2 : tensor<1x4x48xf16>

  // CHECK: [[IN_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.280000e+02>
  // CHECK: [[IN_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.270000e+02>
  // CHECK: [[OUT_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-2.136230e-01>
  // CHECK: [[OUT_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<3.857420e-02>

  // CHECK: [[DATA:%.+]] = const.Declare tensor<48xf16>
  // CHECK-SAME{LITERAL}: dense<[9, -5, -25, 52, -77, -123, 24, 67, -32, 11, -24, 93, -17, -127, -46, -38, 53, -88, -108, -60, 9, -8, -78, 106, -33, 14, -11, -21, -94, -72, 49, 125, -58, -93, 91, 44, 123, -99, -59, 15, -124, -13, 89, -92, -97, 10, -16, 38]>

  // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<48xf16>

  // CHECK: [[ADD:%.+]] = IE.Add([[INPUT]], [[WT_FQ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

  // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @WeightsMultScalarSubFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x6x12x12xf32>
// CHECK-SAME: -> tensor<1x6x12x12xf32>
func.func @WeightsMultScalarSubFakeQuantize(%input: tensor<1x6x12x12xf32>) -> tensor<1x6x12x12xf32> {
  %cst_0 = const.Declare tensor<6x6xf32> = dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]> : tensor<6x6xsi8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<1xf32> = dense<-22> : tensor<si8>, [#const.CastElemType<f32>, #const.Reshape<[1]>]
  %cst_2 = const.Declare tensor<1x1xf32> = dense<0.00704713073> : tensor<1x1xf32>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<1xf32> -> tensor<6x6xf32>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<1x1xf32> -> tensor<6x6xf32>
  %2 = IE.Reshape(%1) {shape_value = [6, 6, 1, 1]} : tensor<6x6xf32> -> tensor<6x6x1x1xf32>
  %3 = IE.Convolution(%input, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x12x12xf32>, tensor<6x6x1x1xf32> -> tensor<1x6x12x12xf32>

  return %3 : tensor<1x6x12x12xf32>

  // CHECK:   [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK:   [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK:   [[OUT_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-0.746995866>
  // CHECK:   [[OUT_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.05002248>

  // CHECK:   [[DATA:%.+]] = const.Declare tensor<6x6xf32>
  // CHECK-SAME{LITERAL}: dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]>

  // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<6x6xf32>

  // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[WT_FQ]]) {shape_value = [6, 6, 1, 1]} : tensor<6x6xf32> -> tensor<6x6x1x1xf32>

  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @NonSplatScale
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x6x12x12xf32>
func.func @NonSplatScale(%input: tensor<1x6x12x12xf32>) -> tensor<1x6x12x12xf32> {
  %cst_0 = const.Declare tensor<6x6xf32> = dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]> : tensor<6x6xsi8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<1xf32> = dense<10> : tensor<si8>, [#const.CastElemType<f32>, #const.Reshape<[1]>]
  %cst_2 = const.Declare tensor<6xf32> = dense<[1.0, 0.5, 0.25, 0.125, 0.06, 0.03]> : tensor<6xf32>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<1xf32> -> tensor<6x6xf32>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<6xf32> -> tensor<6x6xf32>
  %2 = IE.Reshape(%1) {shape_value = [6, 6, 1, 1]} : tensor<6x6xf32> -> tensor<6x6x1x1xf32>
  %3 = IE.Convolution(%input, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x12x12xf32>, tensor<6x6x1x1xf32> -> tensor<1x6x12x12xf32>

  return %3 : tensor<1x6x12x12xf32>

  // CHECK: [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK: [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK: [[OUT_LOW:%.+]] = const.Declare tensor<6xf32>
  // CHECK-SAME{LITERAL}: dense<[-1.380000e+02, -6.900000e+01, -3.450000e+01, -1.725000e+01, -8.27999973, -4.140000e+00]>
  // CHECK: [[OUT_HIGH:%.+]] = const.Declare tensor<6xf32>
  // CHECK-SAME{LITERAL}: dense<[1.170000e+02, 5.850000e+01, 2.925000e+01, 1.462500e+01, 7.020000e+00, 3.510000e+00]>

  // CHECK: [[DATA:%.+]] = const.Declare tensor<6x6xf32>
  // CHECK-SAME{LITERAL}: dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]>

  // CHECK:   [[WT_FQ:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<6x6xf32>

  // CHECK:   [[RESHAPE:%.+]] = IE.Reshape([[WT_FQ]]) {shape_value = [6, 6, 1, 1]}
  // CHECK:   [[CONV:%.+]] = IE.Convolution([[INPUT]], [[RESHAPE]])

  // CHECK:   return [[CONV]] : tensor<1x6x12x12xf32>
}

// -----

// CHECK-LABEL: @NonSplatOffset
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x6x12x12xf32>
func.func @NonSplatOffset(%input: tensor<1x6x12x12xf32>) -> tensor<1x6x12x12xf32> {
  %cst_0 = const.Declare tensor<6x6xf32> = dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]> : tensor<6x6xsi8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<6xf32> = dense<[0, 1, 2, 3, 4, 5]> : tensor<6xsi8>, [#const.CastElemType<f32>]
  %cst_2 = const.Declare tensor<1xf32> = dense<0.5> : tensor<1xf32>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<6xf32> -> tensor<6x6xf32>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<1xf32> -> tensor<6x6xf32>
  %2 = IE.Reshape(%1) {shape_value = [6, 6, 1, 1]} : tensor<6x6xf32> -> tensor<6x6x1x1xf32>
  %3 = IE.Convolution(%input, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x12x12xf32>, tensor<6x6x1x1xf32> -> tensor<1x6x12x12xf32>

  return %3 : tensor<1x6x12x12xf32>

  // CHECK: [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK: [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK: [[OUT_LOW:%.+]] = const.Declare tensor<6xf32>
  // CHECK-SAME{LITERAL}: dense<[-6.400000e+01, -6.450000e+01, -6.500000e+01, -6.550000e+01, -6.600000e+01, -6.650000e+01]>
  // CHECK: [[OUT_HIGH:%.+]] = const.Declare tensor<6xf32>
  // CHECK-SAME{LITERAL}: dense<[6.350000e+01, 6.300000e+01, 6.250000e+01, 6.200000e+01, 6.150000e+01, 6.100000e+01]>

  // CHECK: [[DATA:%.+]] = const.Declare tensor<6x6xf32>
  // CHECK-SAME{LITERAL}: dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]>

  // CHECK:   [[WT_FQ:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<6x6xf32>

  // CHECK:   [[RESHAPE:%.+]] = IE.Reshape([[WT_FQ]]) {shape_value = [6, 6, 1, 1]}
  // CHECK:   [[CONV:%.+]] = IE.Convolution([[INPUT]], [[RESHAPE]])

  // CHECK:   return [[CONV]] : tensor<1x6x12x12xf32>
}

// -----

{-#
  dialect_resources: {
    // Note: first 4 bytes in the dense_resource blob specify alignment
    builtin: {
      // Note: 9 == -7; 14 == E == -2 in bit representation for two's complement
      blob: "0x0400000076545932E0"
    }
  }
#-}

// CHECK-LABEL: @WeightsMultToFakeQuantizeI4WithU8Storage
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
// CHECK-SAME: -> tensor<1x1x28x28xf16>
func.func @WeightsMultToFakeQuantizeI4WithU8Storage(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x3x3xsi4> = dense_resource<blob> : tensor<1x1x3x3xsi4>, [#const.ConvertElemType<si8>, #const.CastElemType<si4>]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %0 = IE.Convert(%cst_0) { dstElemType = f16 } : tensor<1x1x3x3xsi4> -> tensor<1x1x3x3xf16>
  %1 = IE.Multiply(%0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>

  return %2 : tensor<1x1x28x28xf16>

  // CHECK:   [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-8.000000e+00>
  // CHECK:   [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<7.000000e+00>
  // CHECK:   [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.359010e-02>
  // CHECK:   [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.064510e-02>

  // CHECK:   [[DATA:%.+]] = const.Declare tensor<1x1x3x3xf16>
  // CHECK-SAME{LITERAL}: dense_resource<blob>

  // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
  // CHECK-SAME: -> tensor<1x1x3x3xf16>

  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}

  // CHECK: return [[CONV]]
}

// -----

{-#
  dialect_resources: {
    // Note: first 4 bytes in the dense_resource blob specify alignment
    builtin: {
      blob: "0x0400000076545832F0"
    }
  }
#-}

// CHECK-LABEL: @WeightsMultToFakeQuantizeU4WithFP16Storage
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
// CHECK-SAME: -> tensor<1x1x28x28xf16>
func.func @WeightsMultToFakeQuantizeU4WithFP16Storage(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x3x3xui4> = dense_resource<blob> : tensor<1x1x3x3xui4>, [#const.ConvertElemType<ui8>, #const.CastElemType<ui4>]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %0 = IE.Convert(%cst_0) { dstElemType = f16 } : tensor<1x1x3x3xui4> -> tensor<1x1x3x3xf16>
  %1 = IE.Multiply(%0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>

  return %2 : tensor<1x1x28x28xf16>

  // CHECK: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00>
  // CHECK: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.500000e+01>
  // CHECK: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.422000e-02>

  // CHECK: [[DATA:%.+]] = const.Declare tensor<1x1x3x3xf16>
  // CHECK-SAME{LITERAL}: dense_resource<blob>

  // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
  // CHECK-SAME: -> tensor<1x1x3x3xf16>

  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsUI8MultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @WeightsUI8MultToFakeQuantize(%input: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]], [[31, 67, 22], [70, 55, 12], [99, 42, 90]], [[6, 18, 95], [8, 37, 64], [40, 31, 41]], [[35, 2, 98], [94, 60, 68], [3, 39, 88]]], [[[43, 95, 64], [46, 125, 63], [21, 25, 25]], [[118, 103, 12], [84, 67, 55], [105, 13, 10]], [[97, 124, 39], [28, 112, 116], [74, 104, 72]], [[14, 58, 0], [37, 48, 26], [33, 64, 53]]], [[[124, 104, 105], [14, 0, 25], [104, 46, 87]], [[87, 105, 69], [94, 88, 47], [53, 93, 34]], [[62, 44, 10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, 50, 39], [108, 7, 12]]], [[[73, 47, 7], [72, 17, 90], [113, 44, 80]], [[60, 102, 79], [111, 43, 68], [21, 53, 120]], [[109, 69, 30], [120, 7, 107], [30, 42, 66]], [[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]> : tensor<4x4x3x3xui8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.00294781756]]], [[[0.00312666874]]], [[[0.00260377093]]], [[[0.00269700377]]]]> : tensor<4x1x1x1xf32>
  %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411> : tensor<1x1x1x1xf32>
  %0 = IE.FakeQuantize(%input, %cst_3, %cst_2, %cst_3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%0, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK: [[ACT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411>
  // CHECK: [[ACT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102>

  // CHECK: [[WT_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>
  // CHECK: [[WT_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02>
  // CHECK: [[WT_OUT_LOW:%.+]] = const.Declare tensor<4x1x1x1xf32> = dense<0.000000e+00>
  // CHECK: [[WT_OUT_HIGH:%.+]] = const.Declare tensor<4x1x1x1xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[0.751693487]]], [[[0.797300517]]], [[[0.663961589]]], [[[0.687735974]]]]>

  // CHECK: [[WT_DATA:%.+]] = const.Declare tensor<4x4x3x3xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]], [[31, 67, 22], [70, 55, 12], [99, 42, 90]], [[6, 18, 95], [8, 37, 64], [40, 31, 41]], [[35, 2, 98], [94, 60, 68], [3, 39, 88]]], [[[43, 95, 64], [46, 125, 63], [21, 25, 25]], [[118, 103, 12], [84, 67, 55], [105, 13, 10]], [[97, 124, 39], [28, 112, 116], [74, 104, 72]], [[14, 58, 0], [37, 48, 26], [33, 64, 53]]], [[[124, 104, 105], [14, 0, 25], [104, 46, 87]], [[87, 105, 69], [94, 88, 47], [53, 93, 34]], [[62, 44, 10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, 50, 39], [108, 7, 12]]], [[[73, 47, 7], [72, 17, 90], [113, 44, 80]], [[60, 102, 79], [111, 43, 68], [21, 53, 120]], [[109, 69, 30], [120, 7, 107], [30, 42, 66]], [[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]>

  // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[WT_DATA]], [[WT_IN_LOW]], [[WT_IN_HIGH]], [[WT_OUT_LOW]], [[WT_OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<4x4x3x3xf32>

  // CHECK: [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<1x4x28x28xf32>

  // CHECK: [[CONV:%.+]] = IE.Convolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsUI8SubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @WeightsUI8SubToFakeQuantize(%input: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]], [[31, 67, 22], [70, 55, 12], [99, 42, 90]], [[6, 18, 95], [8, 37, 64], [40, 31, 41]], [[35, 2, 98], [94, 60, 68], [3, 39, 88]]], [[[43, 95, 64], [46, 125, 63], [21, 25, 25]], [[118, 103, 12], [84, 67, 55], [105, 13, 10]], [[97, 124, 39], [28, 112, 116], [74, 104, 72]], [[14, 58, 0], [37, 48, 26], [33, 64, 53]]], [[[124, 104, 105], [14, 0, 25], [104, 46, 87]], [[87, 105, 69], [94, 88, 47], [53, 93, 34]], [[62, 44, 10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, 50, 39], [108, 7, 12]]], [[[73, 47, 7], [72, 17, 90], [113, 44, 80]], [[60, 102, 79], [111, 43, 68], [21, 53, 120]], [[109, 69, 30], [120, 7, 107], [30, 42, 66]], [[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]> : tensor<4x4x3x3xui8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<128.0> : tensor<1x1x1x1xf32>
  %1 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>
  // CHECK: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02>
  // CHECK: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02>
  // CHECK: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02>

  // CHECK: [[DATA:%.+]] = const.Declare tensor<4x4x3x3xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]], [[31, 67, 22], [70, 55, 12], [99, 42, 90]], [[6, 18, 95], [8, 37, 64], [40, 31, 41]], [[35, 2, 98], [94, 60, 68], [3, 39, 88]]], [[[43, 95, 64], [46, 125, 63], [21, 25, 25]], [[118, 103, 12], [84, 67, 55], [105, 13, 10]], [[97, 124, 39], [28, 112, 116], [74, 104, 72]], [[14, 58, 0], [37, 48, 26], [33, 64, 53]]], [[[124, 104, 105], [14, 0, 25], [104, 46, 87]], [[87, 105, 69], [94, 88, 47], [53, 93, 34]], [[62, 44, 10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, 50, 39], [108, 7, 12]]], [[[73, 47, 7], [72, 17, 90], [113, 44, 80]], [[60, 102, 79], [111, 43, 68], [21, 53, 120]], [[109, 69, 30], [120, 7, 107], [30, 42, 66]], [[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]>

  // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<4x4x3x3xf32>

  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsUI8ToFakeQuantizeNoSubNoMult
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @WeightsUI8ToFakeQuantizeNoSubNoMult(%input: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]], [[31, 67, 22], [70, 55, 12], [99, 42, 90]], [[6, 18, 95], [8, 37, 64], [40, 31, 41]], [[35, 2, 98], [94, 60, 68], [3, 39, 88]]], [[[43, 95, 64], [46, 125, 63], [21, 25, 25]], [[118, 103, 12], [84, 67, 55], [105, 13, 10]], [[97, 124, 39], [28, 112, 116], [74, 104, 72]], [[14, 58, 0], [37, 48, 26], [33, 64, 53]]], [[[124, 104, 105], [14, 0, 25], [104, 46, 87]], [[87, 105, 69], [94, 88, 47], [53, 93, 34]], [[62, 44, 10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, 50, 39], [108, 7, 12]]], [[[73, 47, 7], [72, 17, 90], [113, 44, 80]], [[60, 102, 79], [111, 43, 68], [21, 53, 120]], [[109, 69, 30], [120, 7, 107], [30, 42, 66]], [[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]> : tensor<4x4x3x3xui8>, [#const.CastElemType<f32>]
  %2 = IE.Convolution(%input, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>
  // CHECK: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02>

  // CHECK: [[DATA:%.+]] = const.Declare tensor<4x4x3x3xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]], [[31, 67, 22], [70, 55, 12], [99, 42, 90]], [[6, 18, 95], [8, 37, 64], [40, 31, 41]], [[35, 2, 98], [94, 60, 68], [3, 39, 88]]], [[[43, 95, 64], [46, 125, 63], [21, 25, 25]], [[118, 103, 12], [84, 67, 55], [105, 13, 10]], [[97, 124, 39], [28, 112, 116], [74, 104, 72]], [[14, 58, 0], [37, 48, 26], [33, 64, 53]]], [[[124, 104, 105], [14, 0, 25], [104, 46, 87]], [[87, 105, 69], [94, 88, 47], [53, 93, 34]], [[62, 44, 10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, 50, 39], [108, 7, 12]]], [[[73, 47, 7], [72, 17, 90], [113, 44, 80]], [[60, 102, 79], [111, 43, 68], [21, 53, 120]], [[109, 69, 30], [120, 7, 107], [30, 42, 66]], [[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]>

  // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<4x4x3x3xf32>

  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WT_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsUI8MultSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x32x32xf32>
// CHECK-SAME: -> tensor<1x16x32x32xf32>
func.func @WeightsUI8MultSubToFakeQuantize(%input: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158> : tensor<1x1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %cst_2 = const.Declare tensor<16x1x1x1xf32> = dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]> : tensor<16x1x1x1xui8>, [#const.CastElemType<f32>]
  %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<2.500000e+01> : tensor<1x1x1x1xf32>
  %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<0.0566197559> : tensor<1x1x1x1xf32>
  %0 = IE.FakeQuantize(%input, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x32x32xf32>
  %1 = IE.Subtract(%cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  %2 = IE.Multiply(%1, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x1x1x1xf32>
  %3 = IE.GroupConvolution(%0, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x32x32xf32>
  return %3 : tensor<1x16x32x32xf32>

  // CHECK: [[ACT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158>
  // CHECK: [[ACT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>

  // CHECK: [[WT_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02>
  // CHECK: [[WT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.41549385>
  // CHECK: [[WT_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<13.0225439>

  // CHECK: [[WT_DATA:%.+]] = const.Declare tensor<16x1x1x1xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]>

  // CHECK: [[WT_FQ:%.+]] = IE.FakeQuantize([[WT_DATA]], [[ACT_LOW]], [[WT_IN_HIGH]], [[WT_OUT_LOW]], [[WT_OUT_HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<16x1x1x1xf32>

  // CHECK: [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<1x16x32x32xf32>

  // CHECK: [[GRUP_CONV:%.+]] = IE.GroupConvolution([[ACT_FQ]], [[WT_FQ]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK: return [[GRUP_CONV]]
}


// -----

// CHECK-LABEL: @MultipleConsumerConstSubMult
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x48xf16>
// CHECK-SAME: -> tensor<1x4x48xf16>
func.func @MultipleConsumerConstSubMult(%input: tensor<1x4x48xf16>) -> tensor<1x4x48xf16> {
  %cst_0 = const.Declare tensor<48xf16> = dense<[9, -5, -25, 52, -77, -123, 24, 67, -32, 11, -24, 93, -17, -127, -46, -38, 53, -88, -108, -60, 9, -8, -78, 106, -33, 14, -11, -21, -94, -72, 49, 125, -58, -93, 91, 44, 123, -99, -59, 15, -124, -13, 89, -92, -97, 10, -16, 38]> : tensor<48xsi8>, [#const.CastElemType<f16>]
  %cst_1 = const.Declare tensor<1xf16> = dense<8.800000e+01> : tensor<1xf16>
  %cst_2 = const.Declare tensor<1xf16> = dense<9.88533836E-4> : tensor<1xf16>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %2 = IE.Add(%input, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  %3 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %4 = IE.Multiply(%3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %5 = IE.Add(%2, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  %6 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %7 = IE.Add(%5, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  %8 = IE.Add(%7, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  %9 = IE.Multiply(%cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %10 = IE.Add(%8, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  %11 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %12 = IE.Add(%10, %11) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>

  return %12 : tensor<1x4x48xf16>

  // CHECK-DAG: [[OUT_LOW1:%.+]] = const.Declare tensor<1xf16> = dense<-2.160000e+02>
  // CHECK-DAG: [[OUT_HIGH1:%.+]] = const.Declare tensor<1xf16> = dense<3.900000e+01>

  // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1xf16> = dense<-1.280000e+02>
  // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1xf16> = dense<1.270000e+02>
  // CHECK-DAG: [[OUT_LOW0:%.+]] = const.Declare tensor<1xf16> = dense<-2.136230e-01>
  // CHECK-DAG: [[OUT_HIGH0:%.+]] = const.Declare tensor<1xf16> = dense<3.857420e-02>
  // CHECK-DAG: [[OUT_LOW2:%.+]] = const.Declare tensor<1xf16> = dense<-1.265870e-01>
  // CHECK-DAG: [[OUT_HIGH2:%.+]] = const.Declare tensor<1xf16> = dense<1.256100e-01>

  // CHECK: [[DATA:%.+]] = const.Declare tensor<48xf16>
  // CHECK-SAME{LITERAL}: dense<[9, -5, -25, 52, -77, -123, 24, 67, -32, 11, -24, 93, -17, -127, -46, -38, 53, -88, -108, -60, 9, -8, -78, 106, -33, 14, -11, -21, -94, -72, 49, 125, -58, -93, 91, 44, 123, -99, -59, 15, -124, -13, 89, -92, -97, 10, -16, 38]>

  // CHECK: [[WT_FQ_0:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])

  // CHECK: [[WT_FQ_1:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])

  // CHECK: [[WT_FQ_2:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])

  // CHECK: [[WT_FQ_3:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW1]], [[OUT_HIGH1]])

  // CHECK: [[WT_FQ_4:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW2]], [[OUT_HIGH2]])

  // CHECK: [[ADD_0:%.+]] = IE.Add([[INPUT]], [[WT_FQ_0]])
  // CHECK: [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[WT_FQ_2]])
  // CHECK: [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[WT_FQ_3]])
  // CHECK: [[ADD_3:%.+]] = IE.Add([[ADD_2]], [[WT_FQ_0]])
  // CHECK: [[ADD_4:%.+]] = IE.Add([[ADD_3]], [[WT_FQ_4]])
  // CHECK: [[ADD_5:%.+]] = IE.Add([[ADD_4]], [[WT_FQ_1]])

  // CHECK: return [[ADD_5]]
}

// -----

// CHECK-LABEL: @DontConvertFP16WeightsMultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
// CHECK-SAME: -> tensor<1x1x28x28xf16>
func.func @DontConvertFP16WeightsMultToFakeQuantize(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x3x3xf16> = dense<[[[[7.0, 6.0, 5.0], [4.0, 5.0, -8.0], [3.0, 2.0, -2.0]]]]> : tensor<1x1x3x3xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>

  return %2 : tensor<1x1x28x28xf16>

  // CHECK:   [[CST_0:%.+]] = const.Declare tensor<1x1x3x3xf16>
  // CHECK-SAME{LITERAL}: dense<[[[[7.000000e+00, 6.000000e+00, 5.000000e+00], [4.000000e+00, 5.000000e+00, -8.000000e+00], [3.000000e+00, 2.000000e+00, -2.000000e+00]]]]>
  // CHECK:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03>
  // CHECK:   [[MULTI:%.+]] = IE.Multiply([[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:   [[CONV:%.+]] = IE.Convolution([[INPUT]], [[MULTI]])
  // CHECK-SAME: {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @DontConvertNonFloatStorageWeightsMultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
// CHECK-SAME: -> tensor<1x1x28x28xf16>
func.func @DontConvertNonFloatStorageWeightsMultToFakeQuantize(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x28x28xui8> = dense<6> : tensor<1x1x28x28xui8>
  %cst_1 = const.Declare tensor<1x1x1x1xui8> = dense<[[[[2]]]]> : tensor<1x1x1x1xui8>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x28x28xui8>, tensor<1x1x1x1xui8> -> tensor<1x1x28x28xui8>
  %2 = IE.Add(%input, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x28x28xf16>, tensor<1x1x28x28xui8> -> tensor<1x1x28x28xf16>

  return %2 : tensor<1x1x28x28xf16>

  // CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x28x28xui8> = dense<6>
  // CHECK:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xui8> = dense<2>
  // CHECK:   [[MULTI:%.+]] = IE.Multiply([[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:   [[ADD:%.+]] = IE.Add([[INPUT]], [[MULTI]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @DontReconvertWeightsMultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x4x3x3xf32>
// CHECK-SAME: -> tensor<4x4x3x3xf32>
func.func @DontReconvertWeightsMultToFakeQuantize(%input: tensor<4x4x3x3xf32>) -> tensor<4x4x3x3xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<5> : tensor<4x4x3x3xui8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-0.275> : tensor<1x1x1x1xf32>
  %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<0.407> : tensor<1x1x1x1xf32>

  %0 = IE.FakeQuantize(%cst_0, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %1 = IE.Add(%input, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<4x4x3x3xf32>, tensor<4x4x3x3xf32> -> tensor<4x4x3x3xf32>

  return %1 : tensor<4x4x3x3xf32>

  // CHECK:   [[DATA:%.+]] = const.Declare tensor<4x4x3x3xf32> = dense<5> {{.*}} [#const.CastElemType<f32>]
  // CHECK:   [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.750000e-01>
  // CHECK:   [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<4.070000e-01>

  // CHECK:   [[FQ:%.+]] = IE.FakeQuantize([[DATA]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<4x4x3x3xf32>

  // CHECK:  [[ADD:%.+]] = IE.Add([[INPUT]], [[FQ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

  // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @DontBlockArgMultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xui8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @DontBlockArgMultToFakeQuantize(%input: tensor<1x4x28x28xui8>) -> tensor<1x4x28x28xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%input) { dstElemType = f32 } : tensor<1x4x28x28xui8> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %2 = IE.Add(%1, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102>
  // CHECK:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01>
  // CHECK:   [[CONV:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32}
  // CHECK:   [[MULTI:%.+]] = IE.Multiply([[CONV]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK:   [[ADD:%.+]] = IE.Add([[MULTI]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

  // CHECK: return [[ADD]]
}

// -----

// Note: this case serves to document the current pattern matching behavior,
// which looks wrong, yet it may be sufficient for "real models" that only have
// simple patterns.

// CHECK-LABEL: @MisbehavingMatch
// CHECK-SAME: -> (tensor<1xf32>, tensor<1xf32>)
func.func @MisbehavingMatch() -> (tensor<1xf32>, tensor<1xf32>) {
  %weights = const.Declare tensor<1xf32> = dense<127> : tensor<1xsi8>, [#const.CastElemType<f32>]
  %offset = const.Declare tensor<1xf32> = dense<-27.0> : tensor<1xf32>
  %scale = const.Declare tensor<1xf32> = dense<0.5> : tensor<1xf32>
  %cst = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00>
  // CHECK: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<5.000000e-01>

  // CHECK: [[IN_LOW:%.+]] = const.Declare tensor<1xf32> = dense<-1.280000e+02>
  // CHECK: [[IN_HIGH:%.+]] = const.Declare tensor<1xf32> = dense<1.270000e+02>
  // CHECK: [[OUT_LOW:%.+]] = const.Declare tensor<1xf32> = dense<-1.010000e+02>
  // CHECK: [[OUT_HIGH:%.+]] = const.Declare tensor<1xf32> = dense<1.540000e+02>

  // CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<1xf32> = dense<127>

  %sub = IE.Subtract(%weights, %offset) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // Note: instead of 'Subtract -> Multiply' treated as 'FQ', we only converted Subtract!
  // CHECK: [[PARTIAL_FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]],

  %mul = IE.Multiply(%sub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK: [[MULT:%.+]] = IE.Multiply([[PARTIAL_FQ]], [[SCALE]])

  %add = IE.Add(%sub, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK: [[ADD:%.+]] = IE.Add([[PARTIAL_FQ]], [[CST]])

  return %mul, %add : tensor<1xf32>, tensor<1xf32>
}

// -----

// Note: this test has some weird MLIR behavior w.r.t. operation users. It is
// not yet clear whether it is a bug of this pass or of MLIR or not a bug at all.
// Anyhow, it's practically necessary to use CHECK-DAG instead of CHECK in order
// to be able to run this test successfully locally and in CI...

// CHECK-LABEL: @TooLongWdStructure
// CHECK-SAME: -> tensor<15xf32>
func.func @TooLongWdStructure() -> tensor<15xf32> {
  %weights = const.Declare tensor<1xf32> = dense<127> : tensor<1xsi8>, [#const.CastElemType<f32>]
  %scale = const.Declare tensor<1xf32> = dense<0.00157480314> : tensor<1xf32>

  %offset = const.Declare tensor<1xf32> = dense<25.0> : tensor<1xf32>
  %scale2 = const.Declare tensor<1xf32> = dense<0.05> : tensor<1xf32>

  %cst = const.Declare tensor<1xf32> = dense<0.5> : tensor<1xf32>

  // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1xf32> = dense<-1.280000e+02>
  // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1xf32> = dense<1.270000e+02>
  // CHECK-DAG: [[OUT_LOW0:%.+]] = const.Declare tensor<1xf32> = dense<-0.201574802>
  // CHECK-DAG: [[OUT_HIGH0:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e-01>

  // CHECK-DAG: [[OUT_LOW1:%.+]] = const.Declare tensor<1xf32> = dense<-7.650000e+00>
  // CHECK-DAG: [[OUT_HIGH1:%.+]] = const.Declare tensor<1xf32> = dense<5.100000e+00>

  // CHECK-DAG: [[OUT_LOW2:%.+]] = const.Declare tensor<1xf32> = dense<-7.650000e+01>
  // CHECK-DAG: [[OUT_HIGH2:%.+]] = const.Declare tensor<1xf32> = dense<5.100000e+01>

  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<1xf32> = dense<127>

  %0 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ0:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %1 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ1:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %2 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ2:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %3 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ3:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %4 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ4:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %5 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ5:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %6 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ6:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %7 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ7:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %8 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ8:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %9 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ9:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %10 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ10:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])
  %11 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ11:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW0]], [[OUT_HIGH0]])

  %12 = IE.Subtract(%weights, %offset) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  %subMul0 = IE.Multiply(%12, %scale2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  %extraUser = IE.Multiply(%12, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // Note: two *separate* FQ blocks are produced here!
  // CHECK-DAG: [[FQ_SUB_MUL0:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW1]], [[OUT_HIGH1]])
  // CHECK-DAG: [[EXTRA_USER:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW2]], [[OUT_HIGH2]])

  %14 = IE.Subtract(%weights, %offset) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  %subMul1 = IE.Multiply(%14, %scale2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[FQ_SUB_MUL1:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW1]], [[OUT_HIGH1]])

  // Note: real network wouldn't have this concat - used here for simplicity.
  %res = IE.Concat(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %subMul0, %subMul1, %extraUser) {
    per_axis = #IE.Concat<axis = 0 : i64>
  }
  : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>,
    tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>,
    tensor<1xf32>
  -> tensor<15xf32>
  // CHECK: [[CONCAT:%.+]] = IE.Concat([[FQ0]], [[FQ1]], [[FQ2]], [[FQ3]], [[FQ4]], [[FQ5]], [[FQ6]],
  // CHECK-SAME:  [[FQ7]], [[FQ8]], [[FQ9]], [[FQ10]], [[FQ11]], [[FQ_SUB_MUL0]], [[FQ_SUB_MUL1]], [[EXTRA_USER]])

  return %res : tensor<15xf32>
  // CHECK: return [[CONCAT]]
}

// -----

!quantFloatType = !QuantileFloat.quantileFloat<4, {-1.0, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0}>

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000002222"
    }
  }
#-}

// CHECK-LABEL: @WeightsMultToFakeQuantizeNF4
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @WeightsMultToFakeQuantizeNF4(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
  %cst_0 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2x!quantFloatType>,
    [
      #const.ConvertElemType<si8>,
      #const.CastElemType<f16>
    ]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>

  return %2 : tensor<1x1x29x29xf16>

  // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.948760e-03> : tensor<1x1x1x1xf16>
  // CHECK: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03> : tensor<1x1x1x1xf16>
  // CHECK: [[CST_4:%.+]] = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2x!QuantileFloat.quantileFloat<4, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}>> isSplat, [#const.ConvertElemType<si8>, #const.CastElemType<f16>]
  // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[CST_4]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = !QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsMultToFakeQuantizeNegativeScales
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3xf16>
func.func @WeightsMultToFakeQuantizeNegativeScales(%arg0: tensor<1x3xf16>) -> tensor<1x3xf16> {
  %cst = const.Declare tensor<3x1xf16> = dense<[[-1.083370e-03], [6.713870e-04], [-6.370540e-04]]> : tensor<3x1xf16>
  %cst_0 = const.Declare tensor<3x3xf16> = dense<[[64, 63, 112], [-8, 62, -8], [8, 63, 16]]> : tensor<3x3xsi8>, [#const.CastElemType<f16>]
  %0 = IE.Multiply(%cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x3xf16>, tensor<3x1xf16> -> tensor<3x3xf16>
  %1 = IE.FullyConnected(%arg0, %0) : tensor<1x3xf16>, tensor<3x3xf16> -> tensor<1x3xf16>
  return %1 : tensor<1x3xf16>
  // CHECK:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf16> = dense<-1.280000e+02> : tensor<1x1xf16>
  // CHECK:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf16> = dense<1.270000e+02> : tensor<1x1xf16>
  // CHECK:  [[OUT_LOW:%.+]] = const.Declare tensor<3x1xf16>
  // CHECK-SAME-LITERAL: dense<[[1.386720e-01], [-8.593750e-02], [8.154300e-02]]> : tensor<3x1xf16>
  // CHECK:  [[OUT_HIGH:%.+]] = const.Declare tensor<3x1xf16>
  // CHECK-SAME-LITERAL: dense<[[-1.375730e-01], [8.526610e-02], [-8.093260e-02]]> : tensor<3x1xf16>
  // CHECK:  [[WT:%.+]] = const.Declare tensor<3x3xf16>
  // CHECK-SAME-LITERAL: dense<[[64, 63, 112], [-8, 62, -8], [8, 63, 16]]> : tensor<3x3xsi8>, [#const.CastElemType<f16>]
  // CHECK:  [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x3xf16>, tensor<1x1xf16>, tensor<1x1xf16>, tensor<3x1xf16>, tensor<3x1xf16> -> tensor<3x3xf16>
  // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[FQ]]) : tensor<1x3xf16>, tensor<3x3xf16> -> tensor<1x3xf16>
  // CHECK:  return [[FC]] : tensor<1x3xf16>
}
