//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-initial-low-precision-transformations-rewriters="rewriter=weights-dequantize-to-dynamic-dequantize" --mlir-print-elementsattrs-with-hex-if-larger -1 %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @WeightsMultToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
func.func @WeightsMultToDynamicDequantize(%input: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %weights = const.Declare tensor<4x4x3x3xf32> = dense<3> : tensor<4x4x3x3xsi8>, [#const.CastElemType<f32>]
  %scale = const.Declare tensor<4x1x1x1xf32> = dense<2.500000e-03> : tensor<4x1x1x1xf32>
  %mul = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  return %conv : tensor<1x4x28x28xf32>

  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<4x4x3x3xsi8>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x1x1x1xf32> = dense<2.500000e-03>
  // CHECK:     [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant}
  // CHECK-NOT: IE.Multiply
  // CHECK:     IE.Convolution([[INPUT]], [[DDQ]])
}

// -----

// CHECK-LABEL: @WeightsMultToDynamicDequantizePerChannel
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
func.func @WeightsMultToDynamicDequantizePerChannel(%input: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, -70], [36, 72, -82]], [[31, -67, 22], [-70, -55, 12], [-99, 42, 90]], [[6, -18, 95], [-8, -37, -64], [40, 31, -41]], [[35, -2, -98], [-94, -60, -68], [-3, -39, 88]]], [[[-43, -95, 64], [46, -125, -63], [-21, -25, -25]], [[-118, -103, -12], [84, 67, 55], [-105, 13, -10]], [[97, -124, 39], [-28, -112, 116], [74, 104, 72]], [[14, 58, 0], [37, -48, 26], [33, -64, 53]]], [[[-124, 104, 105], [-14, 0, -25], [104, -46, -87]], [[87, -105, 69], [94, 88, 47], [53, 93, -34]], [[-62, -44, -10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, -50, -39], [-108, 7, -12]]], [[[-73, -47, 7], [72, -17, 90], [-113, 44, 80]], [[-60, -102, -79], [-111, -43, 68], [-21, 53, 120]], [[-109, -69, 30], [120, -7, 107], [-30, 42, 66]], [[43, 16, -57], [95, 125, -99], [-30, 1, 126]]]]> : tensor<4x4x3x3xsi8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.00294781756]]], [[[0.00312666874]]], [[[0.00260377093]]], [[[0.00269700377]]]]> : tensor<4x1x1x1xf32>
  %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411> : tensor<1x1x1x1xf32>
  %0 = IE.FakeQuantize(%input, %cst_3, %cst_2, %cst_3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%0, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG: [[ACT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411>
  // CHECK-DAG: [[ACT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x1x1x1xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[0.00294781756]]], [[[0.00312666874]]], [[[0.00260377093]]], [[[0.00269700377]]]]>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<4x4x3x3xsi8>
  // CHECK-SAME{LITERAL}: dense<[[[[73, 69, 95], [47, 85, -70], [36, 72, -82]], [[31, -67, 22], [-70, -55, 12], [-99, 42, 90]], [[6, -18, 95], [-8, -37, -64], [40, 31, -41]], [[35, -2, -98], [-94, -60, -68], [-3, -39, 88]]], [[[-43, -95, 64], [46, -125, -63], [-21, -25, -25]], [[-118, -103, -12], [84, 67, 55], [-105, 13, -10]], [[97, -124, 39], [-28, -112, 116], [74, 104, 72]], [[14, 58, 0], [37, -48, 26], [33, -64, 53]]], [[[-124, 104, 105], [-14, 0, -25], [104, -46, -87]], [[87, -105, 69], [94, 88, 47], [53, 93, -34]], [[-62, -44, -10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, -50, -39], [-108, 7, -12]]], [[[-73, -47, 7], [72, -17, 90], [-113, 44, 80]], [[-60, -102, -79], [-111, -43, 68], [-21, 53, 120]], [[-109, -69, 30], [120, -7, 107], [-30, 42, 66]], [[43, 16, -57], [95, 125, -99], [-30, 1, 126]]]]>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x3x3xsi8>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  // CHECK: [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ACT_FQ]], [[DDQ]])
  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsMultSubToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x32x32xf32>
func.func @WeightsMultSubToDynamicDequantize(%input: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
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

  // CHECK-DAG: [[ACT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158>
  // CHECK-DAG: [[ACT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.0566197559>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<16x1x1x1xsi8>
  // CHECK-SAME{LITERAL}: dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xsi8> = dense<2.500000e+01>

  // CHECK:     [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<16x1x1x1xsi8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xsi8> -> tensor<16x1x1x1xf32>
  // CHECK-NOT: IE.Subtract
  // CHECK-NOT: IE.Multiply
  // CHECK:     [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK:     IE.GroupConvolution([[ACT_FQ]], [[DDQ]])
}

// -----

// CHECK-LABEL: @WeightsSubToDynamicDequantizeImplicitUnitScale
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x16x16xf32>
func.func @WeightsSubToDynamicDequantizeImplicitUnitScale(%input: tensor<1x4x16x16xf32>) -> tensor<1x4x16x16xf32> {
  %weights = const.Declare tensor<4x4x1x1xf32> = dense<6> : tensor<4x4x1x1xsi8>, [#const.CastElemType<f32>]
  %zp = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
  %sub = IE.Subtract(%weights, %zp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x1x1xf32>
  %conv = IE.Convolution(%input, %sub) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x16x16xf32>, tensor<4x4x1x1xf32> -> tensor<1x4x16x16xf32>
  return %conv : tensor<1x4x16x16xf32>

  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<4x4x1x1xsi8> = dense<6>
  // The implicit unit scale is broadcast per-output-channel (Dim(0) matches the weights' leading
  // dim), not collapsed to [1, 1, 1, 1] -- consumers such as the New Weight Table ScaleTable
  // builder require the scale's Dim(0) to equal the number of output channels.
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x1x1x1xf32> = dense<1.000000e+00>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xsi8>
  // CHECK:     [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant}
  // CHECK-NOT: IE.Subtract
  // CHECK:     IE.Convolution([[INPUT]], [[DDQ]])
}

// -----

// Gather-fed WD chains are intentionally not converted by this rewriter.
// CHECK-LABEL: @GatherFedInt4StaysAsArithmeticWD
// CHECK-SAME:      [[INDICES:%.+]]: tensor<2xsi32>
func.func @GatherFedInt4StaysAsArithmeticWD(%indices: tensor<2xsi32>) -> tensor<2x8xf32> {
  %weights = const.Declare tensor<4x8xf32> = dense<1> : tensor<4x8xsi4>, [#const.ConvertElemType<si8>, #const.CastElemType<f32>]
  %scale = const.Declare tensor<4x1xf32> = dense<3.9215686e-3> : tensor<4x1xf32>
  %mul = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x8xf32>, tensor<4x1xf32> -> tensor<4x8xf32>
  %gather = IE.Gather(%mul, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<4x8xf32>, tensor<2xsi32> -> tensor<2x8xf32>
  return %gather : tensor<2x8xf32>

  // CHECK:      [[MUL:%.+]] = IE.Multiply
  // CHECK:      IE.Gather([[MUL]], [[INDICES]])
  // CHECK-NOT:  vpux.weights_import_dyn_dequant
}

// -----

// CHECK-LABEL: @WeightsMultSubToNon4DDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x48xf16>
func.func @WeightsMultSubToNon4DDynamicDequantize(%input: tensor<1x4x48xf16>) -> tensor<1x4x48xf16> {
  %cst_0 = const.Declare tensor<48xf16> = dense<[9, -5, -25, 52, -77, -123, 24, 67, -32, 11, -24, 93, -17, -127, -46, -38, 53, -88, -108, -60, 9, -8, -78, 106, -33, 14, -11, -21, -94, -72, 49, 125, -58, -93, 91, 44, 123, -99, -59, 15, -124, -13, 89, -92, -97, 10, -16, 38]> : tensor<48xsi8>, [#const.CastElemType<f16>]
  %cst_1 = const.Declare tensor<1xf16> = dense<8.800000e+01> : tensor<1xf16>
  %cst_2 = const.Declare tensor<1xf16> = dense<9.88533836E-4> : tensor<1xf16>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48xf16>, tensor<1xf16> -> tensor<48xf16>
  %2 = IE.Add(%input, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x48xf16>, tensor<48xf16> -> tensor<1x4x48xf16>
  return %2 : tensor<1x4x48xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf16> = dense<9.889600e-04>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<48xsi8>
  // CHECK-SAME{LITERAL}: dense<[9, -5, -25, 52, -77, -123, 24, 67, -32, 11, -24, 93, -17, -127, -46, -38, 53, -88, -108, -60, 9, -8, -78, 106, -33, 14, -11, -21, -94, -72, 49, 125, -58, -93, 91, 44, 123, -99, -59, 15, -124, -13, 89, -92, -97, 10, -16, 38]>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1xsi8> = dense<8.800000e+01>
  // CHECK:     [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f16, vpux.weights_import_dyn_dequant}
  // CHECK:     [[ADD:%.+]] = IE.Add([[INPUT]], [[DDQ]])
  // CHECK:     return [[ADD]]
}

// -----

// CHECK-LABEL: @WeightsMultScalarSubDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x6x12x12xf32>
func.func @WeightsMultScalarSubDynamicDequantize(%input: tensor<1x6x12x12xf32>) -> tensor<1x6x12x12xf32> {
  %cst_0 = const.Declare tensor<6x6xf32> = dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]> : tensor<6x6xsi8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<1xf32> = dense<-22> : tensor<si8>, [#const.CastElemType<f32>, #const.Reshape<[1]>]
  %cst_2 = const.Declare tensor<1x1xf32> = dense<0.00704713073> : tensor<1x1xf32>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<1xf32> -> tensor<6x6xf32>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<1x1xf32> -> tensor<6x6xf32>
  %2 = IE.Reshape(%1) {shape_value = [6, 6, 1, 1]} : tensor<6x6xf32> -> tensor<6x6x1x1xf32>
  %3 = IE.Convolution(%input, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x12x12xf32>, tensor<6x6x1x1xf32> -> tensor<1x6x12x12xf32>
  return %3 : tensor<1x6x12x12xf32>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1xf32> = dense<0.00704713073>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<6x6xsi8>
  // CHECK-SAME{LITERAL}: dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1xsi8> = dense<-22>
  // CHECK:     [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant}
  // CHECK:     [[RESHAPE:%.+]] = IE.Reshape([[DDQ]]) {shape_value = [6, 6, 1, 1]}
  // CHECK:     [[CONV:%.+]] = IE.Convolution([[INPUT]], [[RESHAPE]])
  // CHECK:     return [[CONV]]
}

// -----

// CHECK-LABEL: @NonSplatScaleDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x6x12x12xf32>
func.func @NonSplatScaleDynamicDequantize(%input: tensor<1x6x12x12xf32>) -> tensor<1x6x12x12xf32> {
  %cst_0 = const.Declare tensor<6x6xf32> = dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]> : tensor<6x6xsi8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<1xf32> = dense<10> : tensor<si8>, [#const.CastElemType<f32>, #const.Reshape<[1]>]
  %cst_2 = const.Declare tensor<6xf32> = dense<[1.0, 0.5, 0.25, 0.125, 0.06, 0.03]> : tensor<6xf32>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<1xf32> -> tensor<6x6xf32>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<6xf32> -> tensor<6x6xf32>
  %2 = IE.Reshape(%1) {shape_value = [6, 6, 1, 1]} : tensor<6x6xf32> -> tensor<6x6x1x1xf32>
  %3 = IE.Convolution(%input, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x12x12xf32>, tensor<6x6x1x1xf32> -> tensor<1x6x12x12xf32>
  return %3 : tensor<1x6x12x12xf32>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<6xf32>
  // CHECK-SAME{LITERAL}: dense<[1.000000e+00, 5.000000e-01, 2.500000e-01, 1.250000e-01, 6.000000e-02, 3.000000e-02]>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<6x6xsi8>
  // CHECK-SAME{LITERAL}: dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1xsi8> = dense<10>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<6x6xsi8>, tensor<6xf32>, tensor<1x1xsi8> -> tensor<6x6xf32>
  // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[DDQ]]) {shape_value = [6, 6, 1, 1]}
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[RESHAPE]])
  // CHECK: return [[CONV]] : tensor<1x6x12x12xf32>
}

// -----

// CHECK-LABEL: @NonSplatOffsetDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x6x12x12xf32>
func.func @NonSplatOffsetDynamicDequantize(%input: tensor<1x6x12x12xf32>) -> tensor<1x6x12x12xf32> {
  %cst_0 = const.Declare tensor<6x6xf32> = dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]> : tensor<6x6xsi8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<6xf32> = dense<[0, 1, 2, 3, 4, 5]> : tensor<6xsi8>, [#const.CastElemType<f32>]
  %cst_2 = const.Declare tensor<1xf32> = dense<0.5> : tensor<1xf32>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<6xf32> -> tensor<6x6xf32>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x6xf32>, tensor<1xf32> -> tensor<6x6xf32>
  %2 = IE.Reshape(%1) {shape_value = [6, 6, 1, 1]} : tensor<6x6xf32> -> tensor<6x6x1x1xf32>
  %3 = IE.Convolution(%input, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6x12x12xf32>, tensor<6x6x1x1xf32> -> tensor<1x6x12x12xf32>
  return %3 : tensor<1x6x12x12xf32>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<5.000000e-01>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<6x6xsi8>
  // CHECK-SAME{LITERAL}: dense<[[-63, -6, 67, -62, 46, 40], [95, 56, -24, 20, -53, -43], [-41, -76, 113, 0, 87, -107], [-121, 105, -89, 64, -91, -39], [92, -16, 89, 5, 92, 27], [-112, 112, -101, 62, 61, -29]]>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x6xsi8>
  // CHECK-SAME{LITERAL}: dense<[0, 1, 2, 3, 4, 5]>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<6x6xsi8>, tensor<1xf32>, tensor<1x6xsi8> -> tensor<6x6xf32>
  // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[DDQ]]) {shape_value = [6, 6, 1, 1]}
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[RESHAPE]])
  // CHECK: return [[CONV]] : tensor<1x6x12x12xf32>
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

// CHECK-LABEL: @WeightsMultToDynamicDequantizeI4WithU8Storage
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @WeightsMultToDynamicDequantizeI4WithU8Storage(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x3x3xsi4> = dense_resource<blob> : tensor<1x1x3x3xsi4>, [#const.ConvertElemType<si8>, #const.CastElemType<si4>]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %0 = IE.Convert(%cst_0) { dstElemType = f16 } : tensor<1x1x3x3xsi4> -> tensor<1x1x3x3xf16>
  %1 = IE.Multiply(%0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>
  return %2 : tensor<1x1x28x28xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<1x1x3x3xsi4> = dense_resource<blob>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<1x1x3x3xsi4>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
  // CHECK: return [[CONV]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
      blob: "0x0400000076545832F0"
    }
  }
#-}

// CHECK-LABEL: @WeightsMultToDynamicDequantizeU4WithFP16Storage
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @WeightsMultToDynamicDequantizeU4WithFP16Storage(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x3x3xui4> = dense_resource<blob> : tensor<1x1x3x3xui4>, [#const.ConvertElemType<ui8>, #const.CastElemType<ui4>]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %0 = IE.Convert(%cst_0) { dstElemType = f16 } : tensor<1x1x3x3xui4> -> tensor<1x1x3x3xf16>
  %1 = IE.Multiply(%0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>
  return %2 : tensor<1x1x28x28xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<1x1x3x3xui4> = dense_resource<blob>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<1x1x3x3xui4>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
  // CHECK: return [[CONV]]
}

// -----

!quantFloatType = !QuantileType.quantile<ui4:f16, {-1.0, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0}>

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000002222"
    }
  }
#-}

// CHECK-LABEL: @WeightsMultToDynamicDequantizeNF4
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @WeightsMultToDynamicDequantizeNF4(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
  %cst_0 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2xui4>,
    [
      #const.ConvertElemType<si8>,
      #const.CastElemType<!quantFloatType>,
      #const.CastElemType<f16>
    ]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
  return %2 : tensor<1x1x29x29xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<1x1x2x2x!QuantileType.quantile<ui4:f16, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}>> = dense_resource<blob>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]]) {dstElemType = f16, vpux.weights_import_dyn_dequant}
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsMultToDynamicDequantizeNegativeScales
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3xf16>
func.func @WeightsMultToDynamicDequantizeNegativeScales(%arg0: tensor<1x3xf16>) -> tensor<1x3xf16> {
  %cst = const.Declare tensor<3x1xf16> = dense<[[-1.083370e-03], [6.713870e-04], [-6.370540e-04]]> : tensor<3x1xf16>
  %cst_0 = const.Declare tensor<3x3xf16> = dense<[[64, 63, 112], [-8, 62, -8], [8, 63, 16]]> : tensor<3x3xsi8>, [#const.CastElemType<f16>]
  %0 = IE.Multiply(%cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x3xf16>, tensor<3x1xf16> -> tensor<3x3xf16>
  %1 = IE.FullyConnected(%arg0, %0) : tensor<1x3xf16>, tensor<3x3xf16> -> tensor<1x3xf16>
  return %1 : tensor<1x3xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<3x1xf16>
  // CHECK-SAME{LITERAL}: dense<[[-1.083370e-03], [6.713870e-04], [-6.370540e-04]]>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<3x3xsi8>
  // CHECK-SAME{LITERAL}: dense<[[64, 63, 112], [-8, 62, -8], [8, 63, 16]]>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<3x3xsi8>, tensor<3x1xf16> -> tensor<3x3xf16>
  // CHECK: [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[DDQ]]) : tensor<1x3xf16>, tensor<3x3xf16> -> tensor<1x3xf16>
  // CHECK: return [[FC]] : tensor<1x3xf16>
}

// -----

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000001B"
    }
  }
#-}

// CHECK-LABEL: @WeightsMultToDynamicDequantizeI2
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @WeightsMultToDynamicDequantizeI2(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
  %cst_0 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2xsi2>,
    [
      #const.ConvertElemType<si8>,
      #const.CastElemType<f16>
    ]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
  return %2 : tensor<1x1x29x29xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<1x1x2x2xsi2> = dense_resource<blob>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<1x1x2x2xsi2>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
  // CHECK: return [[CONV]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000001B"
    }
  }
#-}

// CHECK-LABEL: @WeightsMultToDynamicDequantizeU2
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @WeightsMultToDynamicDequantizeU2(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
  %cst_0 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2xui2>,
    [
      #const.ConvertElemType<ui8>,
      #const.CastElemType<f16>
    ]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[2.0]]]]> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
  return %2 : tensor<1x1x29x29xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<1x1x2x2xui2> = dense_resource<blob>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xui2> = dense<2.000000e+00>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<1x1x2x2xui2>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xui2> -> tensor<1x1x2x2xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
  // CHECK: return [[CONV]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
      blob: "0x0400000013"
    }
  }
#-}

// CHECK-LABEL: @WeightsMultToDynamicDequantizeI2TernaryWeights
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @WeightsMultToDynamicDequantizeI2TernaryWeights(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
  %cst_0 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2xsi2>,
    [
      #const.ConvertElemType<si8>,
      #const.CastElemType<f16>
    ]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
  return %2 : tensor<1x1x29x29xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<1x1x2x2xsi2> = dense_resource<blob>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<1x1x2x2xsi2>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
  // CHECK: return [[CONV]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
      blob: "0x0400000012"
    }
  }
#-}

// CHECK-LABEL: @WeightsMultToDynamicDequantizeU2TernaryWeights
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @WeightsMultToDynamicDequantizeU2TernaryWeights(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
  %cst_0 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2xui2>,
    [
      #const.ConvertElemType<ui8>,
      #const.CastElemType<f16>
    ]
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[1.0]]]]> : tensor<1x1x1x1xf16>
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %0 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  %1 = IE.Multiply(%0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
  return %2 : tensor<1x1x29x29xf16>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<1x1x2x2xui2> = dense_resource<blob>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xui2> = dense<1.000000e+00>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<1x1x2x2xui2>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xui2> -> tensor<1x1x2x2xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsUI8MultToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
func.func @WeightsUI8MultToDynamicDequantize(%input: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]], [[31, 67, 22], [70, 55, 12], [99, 42, 90]], [[6, 18, 95], [8, 37, 64], [40, 31, 41]], [[35, 2, 98], [94, 60, 68], [3, 39, 88]]], [[[43, 95, 64], [46, 125, 63], [21, 25, 25]], [[118, 103, 12], [84, 67, 55], [105, 13, 10]], [[97, 124, 39], [28, 112, 116], [74, 104, 72]], [[14, 58, 0], [37, 48, 26], [33, 64, 53]]], [[[124, 104, 105], [14, 0, 25], [104, 46, 87]], [[87, 105, 69], [94, 88, 47], [53, 93, 34]], [[62, 44, 10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, 50, 39], [108, 7, 12]]], [[[73, 47, 7], [72, 17, 90], [113, 44, 80]], [[60, 102, 79], [111, 43, 68], [21, 53, 120]], [[109, 69, 30], [120, 7, 107], [30, 42, 66]], [[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]> : tensor<4x4x3x3xui8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.00294781756]]], [[[0.00312666874]]], [[[0.00260377093]]], [[[0.00269700377]]]]> : tensor<4x1x1x1xf32>
  %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102> : tensor<1x1x1x1xf32>
  %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411> : tensor<1x1x1x1xf32>
  %0 = IE.FakeQuantize(%input, %cst_3, %cst_2, %cst_3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x28x28xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x28x28xf32>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%0, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG: [[ACT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.273143411>
  // CHECK-DAG: [[ACT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.407326102>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x1x1x1xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[0.00294781756]]], [[[0.00312666874]]], [[[0.00260377093]]], [[[0.00269700377]]]]>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<4x4x3x3xui8>
  // CHECK-SAME{LITERAL}: dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]], [[31, 67, 22], [70, 55, 12], [99, 42, 90]], [[6, 18, 95], [8, 37, 64], [40, 31, 41]], [[35, 2, 98], [94, 60, 68], [3, 39, 88]]], [[[43, 95, 64], [46, 125, 63], [21, 25, 25]], [[118, 103, 12], [84, 67, 55], [105, 13, 10]], [[97, 124, 39], [28, 112, 116], [74, 104, 72]], [[14, 58, 0], [37, 48, 26], [33, 64, 53]]], [[[124, 104, 105], [14, 0, 25], [104, 46, 87]], [[87, 105, 69], [94, 88, 47], [53, 93, 34]], [[62, 44, 10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, 50, 39], [108, 7, 12]]], [[[73, 47, 7], [72, 17, 90], [113, 44, 80]], [[60, 102, 79], [111, 43, 68], [21, 53, 120]], [[109, 69, 30], [120, 7, 107], [30, 42, 66]], [[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x3x3xui8>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  // CHECK: [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ACT_FQ]], [[DDQ]])
  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsUI8SubToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
func.func @WeightsUI8SubToDynamicDequantize(%input: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]], [[31, 67, 22], [70, 55, 12], [99, 42, 90]], [[6, 18, 95], [8, 37, 64], [40, 31, 41]], [[35, 2, 98], [94, 60, 68], [3, 39, 88]]], [[[43, 95, 64], [46, 125, 63], [21, 25, 25]], [[118, 103, 12], [84, 67, 55], [105, 13, 10]], [[97, 124, 39], [28, 112, 116], [74, 104, 72]], [[14, 58, 0], [37, 48, 26], [33, 64, 53]]], [[[124, 104, 105], [14, 0, 25], [104, 46, 87]], [[87, 105, 69], [94, 88, 47], [53, 93, 34]], [[62, 44, 10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, 50, 39], [108, 7, 12]]], [[[73, 47, 7], [72, 17, 90], [113, 44, 80]], [[60, 102, 79], [111, 43, 68], [21, 53, 120]], [[109, 69, 30], [120, 7, 107], [30, 42, 66]], [[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]> : tensor<4x4x3x3xui8>, [#const.CastElemType<f32>]
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<128.0> : tensor<1x1x1x1xf32>
  %1 = IE.Subtract(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  return %2 : tensor<1x4x28x28xf32>

  // Shift-only chain (no Multiply): implicit unit scale is synthesized broadcast per-output-channel,
  // and the shift becomes the ZP.
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<4x4x3x3xui8>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x1x1x1xf32> = dense<1.000000e+00>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xui8> = dense<1.280000e+02>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x3x3xui8>, tensor<4x1x1x1xf32>, tensor<1x1x1x1xui8> -> tensor<4x4x3x3xf32>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]])
  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsUI8ToDynamicDequantizeNoSubNoMult
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
func.func @WeightsUI8ToDynamicDequantizeNoSubNoMult(%input: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xf32> = dense<[[[[73, 69, 95], [47, 85, 70], [36, 72, 82]], [[31, 67, 22], [70, 55, 12], [99, 42, 90]], [[6, 18, 95], [8, 37, 64], [40, 31, 41]], [[35, 2, 98], [94, 60, 68], [3, 39, 88]]], [[[43, 95, 64], [46, 125, 63], [21, 25, 25]], [[118, 103, 12], [84, 67, 55], [105, 13, 10]], [[97, 124, 39], [28, 112, 116], [74, 104, 72]], [[14, 58, 0], [37, 48, 26], [33, 64, 53]]], [[[124, 104, 105], [14, 0, 25], [104, 46, 87]], [[87, 105, 69], [94, 88, 47], [53, 93, 34]], [[62, 44, 10], [81, 110, 32], [10, 72, 30]], [[117, 64, 41], [0, 50, 39], [108, 7, 12]]], [[[73, 47, 7], [72, 17, 90], [113, 44, 80]], [[60, 102, 79], [111, 43, 68], [21, 53, 120]], [[109, 69, 30], [120, 7, 107], [30, 42, 66]], [[43, 16, 57], [95, 125, 99], [30, 1, 126]]]]> : tensor<4x4x3x3xui8>, [#const.CastElemType<f32>]
  %2 = IE.Convolution(%input, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  return %2 : tensor<1x4x28x28xf32>

  // No Subtract, no Multiply: implicit unit scale broadcast per-output-channel, no ZP.
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<4x4x3x3xui8>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x1x1x1xf32> = dense<1.000000e+00>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x3x3xui8>, tensor<4x1x1x1xf32> -> tensor<4x4x3x3xf32>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]])
  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @WeightsUI8MultSubToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x32x32xf32>
func.func @WeightsUI8MultSubToDynamicDequantize(%input: tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32> {
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

  // CHECK-DAG: [[ACT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.99976158>
  // CHECK-DAG: [[ACT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.0566197559>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<16x1x1x1xui8>
  // CHECK-SAME{LITERAL}: dense<[[[[27]]], [[[25]]], [[[39]]], [[[22]]], [[[27]]], [[[25]]], [[[21]]], [[[27]]], [[[31]]], [[[29]]], [[[42]]], [[[27]]], [[[27]]], [[[28]]], [[[33]]], [[[33]]]]>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xui8> = dense<2.500000e+01>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<16x1x1x1xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8> -> tensor<16x1x1x1xf32>
  // CHECK: [[ACT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[ACT_LOW]], [[ACT_HIGH]], [[ACT_LOW]], [[ACT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK: [[GRUP_CONV:%.+]] = IE.GroupConvolution([[ACT_FQ]], [[DDQ]])
  // CHECK: return [[GRUP_CONV]]
}

// -----

// Multiple users of the same const weights/shift/scale each get their own DynamicDequantize.
// CHECK-LABEL: @MultipleConsumerConstSubMultDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x48xf16>
func.func @MultipleConsumerConstSubMultDynamicDequantize(%input: tensor<1x4x48xf16>) -> tensor<1x4x48xf16> {
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

  // CHECK-DAG: [[SCALE_1:%.+]] = const.Declare tensor<48xf16> = dense<1.000000e+00>
  // CHECK-DAG: [[SCALE_0:%.+]] = const.Declare tensor<1xf16> = dense<9.889600e-04>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<48xsi8>
  // CHECK-SAME{LITERAL}: dense<[9, -5, -25, 52, -77, -123, 24, 67, -32, 11, -24, 93, -17, -127, -46, -38, 53, -88, -108, -60, 9, -8, -78, 106, -33, 14, -11, -21, -94, -72, 49, 125, -58, -93, 91, 44, 123, -99, -59, 15, -124, -13, 89, -92, -97, 10, -16, 38]>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1xsi8> = dense<8.800000e+01>

  // CHECK: [[DDQ_0:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE_0]], [[ZP]])
  // CHECK: [[DDQ_1:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE_0]], [[ZP]])
  // CHECK: [[DDQ_2:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE_0]], [[ZP]])
  // CHECK: [[DDQ_3:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE_1]], [[ZP]])
  // CHECK: [[DDQ_4:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE_0]])

  // CHECK: [[ADD_0:%.+]] = IE.Add([[INPUT]], [[DDQ_0]])
  // CHECK: [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[DDQ_1]])
  // CHECK: [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[DDQ_3]])
  // CHECK: [[ADD_3:%.+]] = IE.Add([[ADD_2]], [[DDQ_0]])
  // CHECK: [[ADD_4:%.+]] = IE.Add([[ADD_3]], [[DDQ_4]])
  // CHECK: [[ADD_5:%.+]] = IE.Add([[ADD_4]], [[DDQ_2]])

  // CHECK: return [[ADD_5]]
}

// -----

// CHECK-LABEL: @DontConvertFP16WeightsMultToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @DontConvertFP16WeightsMultToDynamicDequantize(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x3x3xf16> = dense<[[[[7.0, 6.0, 5.0], [4.0, 5.0, -8.0], [3.0, 2.0, -2.0]]]]> : tensor<1x1x3x3xf16>
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<[[[[0.00294781756]]]]> : tensor<1x1x1x1xf16>
  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x3x3xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x3x3xf16> -> tensor<1x1x28x28xf16>
  return %2 : tensor<1x1x28x28xf16>

  // CHECK-NOT: IE.DynamicDequantize

  // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x3x3xf16>
  // CHECK-SAME{LITERAL}: dense<[[[[7.000000e+00, 6.000000e+00, 5.000000e+00], [4.000000e+00, 5.000000e+00, -8.000000e+00], [3.000000e+00, 2.000000e+00, -2.000000e+00]]]]>
  // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03>
  // CHECK: [[MULTI:%.+]] = IE.Multiply([[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[MULTI]])
  // CHECK-SAME: {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @DontConvertNonFloatStorageWeightsMultToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @DontConvertNonFloatStorageWeightsMultToDynamicDequantize(%input: tensor<1x1x28x28xf16>) -> tensor<1x1x28x28xf16> {
  %cst_0 = const.Declare tensor<1x1x28x28xui8> = dense<6> : tensor<1x1x28x28xui8>
  %cst_1 = const.Declare tensor<1x1x1x1xui8> = dense<[[[[2]]]]> : tensor<1x1x1x1xui8>

  %1 = IE.Multiply(%cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x28x28xui8>, tensor<1x1x1x1xui8> -> tensor<1x1x28x28xui8>
  %2 = IE.Add(%input, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x28x28xf16>, tensor<1x1x28x28xui8> -> tensor<1x1x28x28xf16>
  return %2 : tensor<1x1x28x28xf16>

  // CHECK-NOT: IE.DynamicDequantize

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x28x28xui8> = dense<6>
  // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xui8> = dense<2>
  // CHECK: [[MULTI:%.+]] = IE.Multiply([[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[ADD:%.+]] = IE.Add([[INPUT]], [[MULTI]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: return [[ADD]]
}

// -----

// CHECK-LABEL: @DontReconvertWeightsAlreadyDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x4x3x3xf32>
func.func @DontReconvertWeightsAlreadyDynamicDequantize(%input: tensor<4x4x3x3xf32>) -> tensor<4x4x3x3xf32> {
  %cst_0 = const.Declare tensor<4x4x3x3xsi8> = dense<5> : tensor<4x4x3x3xsi8>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<0.00159420295> : tensor<1x1x1x1xf32>

  %0 = IE.DynamicDequantize(%cst_0, %cst_1) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x3x3xsi8>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %1 = IE.Add(%input, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<4x4x3x3xf32>, tensor<4x4x3x3xf32> -> tensor<4x4x3x3xf32>

  return %1 : tensor<4x4x3x3xf32>

  // CHECK:      [[DDQ:%.+]] = IE.DynamicDequantize
  // CHECK-SAME:   {dstElemType = f32, vpux.weights_import_dyn_dequant}
  // CHECK:      [[ADD:%.+]] = IE.Add([[INPUT]], [[DDQ]])
  // CHECK:      return [[ADD]]
}

// -----

// CHECK-LABEL: @U2WeightsConstPrefillPatternToDynamicDequantize
// CHECK-SAME:      [[ACT:%.+]]: tensor<1x3x4xf32>
func.func @U2WeightsConstPrefillPatternToDynamicDequantize(%act : tensor<1x3x4xf32>) -> tensor<1x3x4xf32> {
  %weights = const.Declare tensor<4x2x2xf16> = dense_resource<weights_blob> : tensor<16xui2>, [#const.ConvertElemType<ui8>, #const.Reshape<[4, 2, 2]>, #const.CastElemType<f16>]
  %scale = const.Declare tensor<4x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]
  %shift = const.Declare tensor<4x2x1xf16> = dense_resource<shift_blob> : tensor<8xui2>, [#const.ConvertElemType<ui8>, #const.Reshape<[4, 2, 1]>, #const.CastElemType<f16>]

  %1 = IE.Subtract(%weights, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %4 = IE.Convert(%3) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %5 = IE.Reshape(%act) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  %6 = IE.FullyConnected(%5, %4) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  %7 = IE.Reshape(%6) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  return %7 : tensor<1x3x4xf32>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16>
  // CHECK-SAME{LITERAL}: dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]>
  // CHECK-DAG: [[WT_RAW:%.+]] = const.Declare tensor<4x2x2xui2> = dense_resource<weights_blob>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<4x2x1xui2> = dense_resource<shift_blob>

  // CHECK-NOT: vpux.weights_import_dyn_dequant
  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WT_RAW]], [[SCALE]], [[ZP]]) {dstElemType = f16} : tensor<4x2x2xui2>, tensor<4x2x1xf16>, tensor<4x2x1xui2> -> tensor<4x2x2xf16>
  // CHECK: [[RESHAPE_1:%.+]] = IE.AffineReshape([[DDQ]])
  // CHECK: [[CONVERT:%.+]] = IE.Convert([[RESHAPE_1]]) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[ACT]]) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  // CHECK: [[FC:%.+]] = IE.FullyConnected([[RESHAPE_2]], [[CONVERT]]) : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>
  // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[FC]]) {shape_value = [1, 3, 4]} : tensor<3x4xf32> -> tensor<1x3x4xf32>
  // CHECK: return [[RESHAPE_3]]
}

{-#
  dialect_resources: {
    builtin: {
      weights_blob: "0x010000001AA12345",
      shift_blob: "0x010000001AA1"
    }
  }
#-}

// -----

// CHECK-LABEL: @DontLhsConstScaleToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2xf32>
func.func @DontLhsConstScaleToDynamicDequantize(%arg0: tensor<1x2xf32>) -> tensor<1x6xf32> {
  %lhs = const.Declare tensor<6x2xf32> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]> : tensor<12xf32>, [#const.Reshape<[6, 2]>]
  %rhs = const.Declare tensor<6x2xf32> = dense<[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]> : tensor<12xui8>, [#const.Reshape<[6, 2]>, #const.CastElemType<f32>]

  %0 = IE.Multiply(%lhs, %rhs) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<6x2xf32>, tensor<6x2xf32> -> tensor<6x2xf32>
  %1 = IE.FullyConnected(%arg0, %0) : tensor<1x2xf32>, tensor<6x2xf32> -> tensor<1x6xf32>
  return %1 : tensor<1x6xf32>

  // The quantized (integer) side is the RHS of the Multiply, not the "weights-like" LHS the matcher
  // expects: the const folder simply rescales LHS into the RHS instead of forming a WD chain.
  // CHECK-NOT: IE.DynamicDequantize

  // CHECK: [[CST:%.+]] = const.Declare tensor<6x2xf32> =
  // CHECK-SAME{LITERAL}: dense<[1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01, 0.899999976, 1.000000e+00, 1.100000e+00, 1.200000e+00]> : tensor<12xf32>,
  // CHECK-SAME{LITERAL}:   [#const.Reshape<[6, 2]>, #const.Rescale<Content<dense<[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]> : tensor<12xui8>, [#const.Reshape<[6, 2]>, #const.CastElemType<f32>]>>]

  // CHECK: [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[CST]])
  // CHECK: return [[FC]]
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

  // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1xf32> = dense<5.000000e-01>
  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<1xsi8> = dense<127>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1xsi8> = dense<-2.700000e+01>

  %sub = IE.Subtract(%weights, %offset) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // Note: instead of 'Subtract -> Multiply' treated as one DynDeq chain, we only converted Subtract!
  // CHECK: [[PARTIAL_DDQ:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[CST]], [[ZP]])

  %mul = IE.Multiply(%sub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK: [[MULT:%.+]] = IE.Multiply([[PARTIAL_DDQ]], [[SCALE]])

  %add = IE.Add(%sub, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK: [[ADD:%.+]] = IE.Add([[PARTIAL_DDQ]], [[CST]])

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

  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<1xsi8> = dense<127>
  // CHECK-DAG: [[SCALE0:%.+]] = const.Declare tensor<1xf32> = dense<0.00157480314>
  // CHECK-DAG: [[SCALE1:%.+]] = const.Declare tensor<1xf32> = dense<5.000000e-02>
  // CHECK-DAG: [[SCALE2:%.+]] = const.Declare tensor<1xf32> = dense<5.000000e-01>
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1xsi8> = dense<2.500000e+01>

  %0 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ0:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %1 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ1:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %2 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ2:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %3 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ3:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %4 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ4:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %5 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ5:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %6 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ6:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %7 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ7:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %8 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ8:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %9 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ9:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %10 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ10:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])
  %11 = IE.Multiply(%weights, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ11:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE0]])

  %12 = IE.Subtract(%weights, %offset) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  %subMul0 = IE.Multiply(%12, %scale2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  %extraUser = IE.Multiply(%12, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // Note: two *separate* DynamicDequantize ops are produced here!
  // CHECK-DAG: [[DDQ_SUB_MUL0:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE1]], [[ZP]])
  // CHECK-DAG: [[EXTRA_USER:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE2]], [[ZP]])

  %14 = IE.Subtract(%weights, %offset) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  %subMul1 = IE.Multiply(%14, %scale2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // CHECK-DAG: [[DDQ_SUB_MUL1:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE1]], [[ZP]])

  // Note: real network wouldn't have this concat - used here for simplicity.
  %res = IE.Concat(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %subMul0, %subMul1, %extraUser) {
    per_axis = #IE.Concat<axis = 0 : i64>
  }
  : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>,
    tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>,
    tensor<1xf32>
  -> tensor<15xf32>
  // CHECK: [[CONCAT:%.+]] = IE.Concat(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}},
  // CHECK-SAME:  %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, [[DDQ_SUB_MUL0]], [[DDQ_SUB_MUL1]], [[EXTRA_USER]])

  return %res : tensor<15xf32>
  // CHECK: return [[CONCAT]]
}

// -----

// U16 weights are a supported quantized weights type: the WD chain is converted to a marked
// IE.DynamicDequantize (raw ui16 storage + scale + integer zero-point).

// CHECK-LABEL: @U16ConstWeightsToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
func.func @U16ConstWeightsToDynamicDequantize(%input: tensor<1x4x28x28xf32>) -> tensor<1x4x28x28xf32> {
  %weights = const.Declare tensor<4x4x3x3xf32> = dense<400> : tensor<4x4x3x3xui16>, [#const.CastElemType<f32>]
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.6> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
  %sub = IE.Subtract(%weights, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %mul = IE.Multiply(%sub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %conv = IE.Convolution(%input, %mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  return %conv : tensor<1x4x28x28xf32>

  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.000000e-01> : tensor<1x1x1x1xf32>
  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x4x3x3xui16> = dense<400> : tensor<4x4x3x3xui16>, [#const.CastElemType<ui16>]
  // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xui16> = dense<1.000000e+01> : tensor<1x1x1x1xf32>, [#const.CastElemType<ui16>]
  // CHECK: [[DD:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x3x3xui16>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui16> -> tensor<4x4x3x3xf32>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DD]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  // CHECK: return [[CONV]] : tensor<1x4x28x28xf32>
}

// -----

// WD chain whose last op feeds a single GatherOp: the DynDeq const rewriter must bail out and leave the
// chain intact so that ConsolidateWeightsDequantization can produce a DynamicDequantizeOp itself.
// Deferring avoids dequantizing the full embedding table offline; DQ runs after Gather instead.

// CHECK-LABEL: @GatherFedInt4StaysAsArithmeticWDConst
// CHECK-SAME:      [[INDICES:%.+]]: tensor<256xsi32>
// CHECK-SAME: -> tensor<256x768xf32>
func.func @GatherFedInt4StaysAsArithmeticWDConst(%indices: tensor<256xsi32>) -> tensor<256x768xf32> {
  // int4 embedding table: vocab_size=262144, hidden_dim=768
  %cst_wt = const.Declare tensor<262144x768xf32> = dense<1> : tensor<262144x768xsi4>,
      [#const.ConvertElemType<si8>, #const.CastElemType<f32>]
  // Per-row (multi-axis) scale: shape [262144, 1]
  %cst_scale = const.Declare tensor<262144x1xf32> = dense<3.9215686e-3> : tensor<262144x1xf32>
  %cst_splat = const.Declare tensor<1x1xf32> = dense<2.0> : tensor<1x1xf32>

  %mul_wd = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<262144x768xf32>, tensor<262144x1xf32> -> tensor<262144x768xf32>
  %gather = IE.Gather(%mul_wd, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      : tensor<262144x768xf32>, tensor<256xsi32> -> tensor<256x768xf32>
  %mul_out = IE.Multiply(%gather, %cst_splat) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<256x768xf32>, tensor<1x1xf32> -> tensor<256x768xf32>
  return %mul_out : tensor<256x768xf32>

  // CHECK-DAG: [[WT:%.+]]    = const.Declare tensor<262144x768xf32>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<262144x1xf32>
  // CHECK-DAG: [[SPLAT:%.+]] = const.Declare tensor<1x1xf32>
  // CHECK-NOT: IE.DynamicDequantize
  // CHECK: [[MUL_WD:%.+]]  = IE.Multiply([[WT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME: tensor<262144x768xf32>, tensor<262144x1xf32> -> tensor<262144x768xf32>
  // CHECK: [[GATHER:%.+]]  = IE.Gather([[MUL_WD]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
  // CHECK-SAME: tensor<262144x768xf32>, tensor<256xsi32> -> tensor<256x768xf32>
  // CHECK: [[MUL_OUT:%.+]] = IE.Multiply([[GATHER]], [[SPLAT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME: tensor<256x768xf32>, tensor<1x1xf32> -> tensor<256x768xf32>
  // CHECK: return [[MUL_OUT]]
}

// -----

// WD chain with no Gather consumer: the const rewriter converts it to DynamicDequantize as normal.
// CHECK-LABEL: @NonGatherFedInt4GoesThroughDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x768x28x28xf32>
func.func @NonGatherFedInt4GoesThroughDynamicDequantize(%input: tensor<1x768x28x28xf32>) -> tensor<1x768x28x28xf32> {
  %cst_wt = const.Declare tensor<768x768x1x1xf32> = dense<1> : tensor<768x768x1x1xsi4>,
      [#const.ConvertElemType<si8>, #const.CastElemType<f32>]
  %cst_scale = const.Declare tensor<768x1x1x1xf32> = dense<3.9215686e-3> : tensor<768x1x1x1xf32>
  %mul_wd = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<768x768x1x1xf32>, tensor<768x1x1x1xf32> -> tensor<768x768x1x1xf32>
  %conv = IE.Convolution(%input, %mul_wd) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
      : tensor<1x768x28x28xf32>, tensor<768x768x1x1xf32> -> tensor<1x768x28x28xf32>
  return %conv : tensor<1x768x28x28xf32>

  // CHECK:     [[DDQ:%.+]] = IE.DynamicDequantize
  // CHECK-NOT: IE.Multiply{{.*}}768x768x1x1
  // CHECK:     [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]])
  // CHECK:     return [[CONV]]
}

// -----

// CHECK-LABEL: @EmbeddingInt8PerRowWithGatherStaysAsArithmeticWD
// CHECK-SAME:      [[INDICES:%.+]]: tensor<256xsi32>
// CHECK-SAME: -> tensor<256x512xf32>
func.func @EmbeddingInt8PerRowWithGatherStaysAsArithmeticWD(%indices: tensor<256xsi32>) -> tensor<256x512xf32> {
  %cst_wt    = const.Declare tensor<65536x512xf32> = dense<1> : tensor<65536x512xsi8>, [#const.CastElemType<f32>]
  %cst_scale = const.Declare tensor<65536x1xf32> = dense<3.9215686e-3> : tensor<65536x1xf32>
  %mul = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<65536x512xf32>, tensor<65536x1xf32> -> tensor<65536x512xf32>
  %gather = IE.Gather(%mul, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      : tensor<65536x512xf32>, tensor<256xsi32> -> tensor<256x512xf32>
  return %gather : tensor<256x512xf32>

  // CHECK-DAG: [[WT:%.+]]    = const.Declare tensor<65536x512xf32>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<65536x1xf32>
  // CHECK-NOT: IE.DynamicDequantize
  // CHECK: [[MUL:%.+]]    = IE.Multiply([[WT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME: tensor<65536x512xf32>, tensor<65536x1xf32> -> tensor<65536x512xf32>
  // CHECK: [[GATHER:%.+]] = IE.Gather([[MUL]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
  // CHECK-SAME: tensor<65536x512xf32>, tensor<256xsi32> -> tensor<256x512xf32>
  // CHECK: return [[GATHER]]
}

// -----

// CHECK-LABEL: @EmbeddingInt8PerTensorWithGatherStaysAsArithmeticWD
// CHECK-SAME:      [[INDICES:%.+]]: tensor<256xsi32>
// CHECK-SAME: -> tensor<256x512xf32>
func.func @EmbeddingInt8PerTensorWithGatherStaysAsArithmeticWD(%indices: tensor<256xsi32>) -> tensor<256x512xf32> {
  %cst_wt    = const.Declare tensor<65536x512xf32> = dense<1> : tensor<65536x512xsi8>, [#const.CastElemType<f32>]
  %cst_scale = const.Declare tensor<1x1xf32> = dense<3.9215686e-3> : tensor<1x1xf32>
  %mul = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<65536x512xf32>, tensor<1x1xf32> -> tensor<65536x512xf32>
  %gather = IE.Gather(%mul, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      : tensor<65536x512xf32>, tensor<256xsi32> -> tensor<256x512xf32>
  return %gather : tensor<256x512xf32>

  // CHECK-DAG: [[WT:%.+]]    = const.Declare tensor<65536x512xf32>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1xf32>
  // CHECK-NOT: IE.DynamicDequantize
  // CHECK: [[MUL:%.+]]    = IE.Multiply([[WT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME: tensor<65536x512xf32>, tensor<1x1xf32> -> tensor<65536x512xf32>
  // CHECK: [[GATHER:%.+]] = IE.Gather([[MUL]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
  // CHECK-SAME: tensor<65536x512xf32>, tensor<256xsi32> -> tensor<256x512xf32>
  // CHECK: return [[GATHER]]
}

// -----

// CHECK-LABEL: @EmbeddingInt4ViaConvertDynamicDequantize
// CHECK-SAME:      [[INDICES:%.+]]: tensor<2xsi32>
// CHECK-SAME: -> tensor<2x8xf32>
func.func @EmbeddingInt4ViaConvertDynamicDequantize(%indices: tensor<2xsi32>) -> tensor<2x8xf32> {
  %cst_wt = const.Declare tensor<4x8xf16> = dense<1> : tensor<4x8xsi4>,
      [#const.ConvertElemType<si8>, #const.CastElemType<f16>]
  %cst_scale = const.Declare tensor<4x1xf16> = dense<3.922e-3> : tensor<4x1xf16>
  %mul = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      : tensor<4x8xf16>, tensor<4x1xf16> -> tensor<4x8xf16>
  %convert = IE.Convert(%mul) {dstElemType = f32} : tensor<4x8xf16> -> tensor<4x8xf32>
  %gather = IE.Gather(%convert, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      : tensor<4x8xf32>, tensor<2xsi32> -> tensor<2x8xf32>
  return %gather : tensor<2x8xf32>

  // The WD chain's Multiply result feeds (through an intervening Convert) a Gather: the
  // isQuantizedConsumedByGather guard defers conversion, same as the plain GatherFedInt4 case.
  // CHECK-NOT: IE.DynamicDequantize

  // CHECK: [[WT:%.+]]    = const.Declare tensor<4x8xf16>
  // CHECK: [[SCALE:%.+]] = const.Declare tensor<4x1xf16>
  // CHECK: [[MUL:%.+]] = IE.Multiply([[WT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK: [[CONVERT:%.+]] = IE.Convert([[MUL]]) {dstElemType = f32} : tensor<4x8xf16> -> tensor<4x8xf32>
  // CHECK: [[GATHER:%.+]]  = IE.Gather([[CONVERT]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
  // CHECK-SAME: tensor<4x8xf32>, tensor<2xsi32> -> tensor<2x8xf32>
  // CHECK: return [[GATHER]]
}

// -----

// CHECK-LABEL: @BlockArgPerGroupScaleToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui8>
func.func @BlockArgPerGroupScaleToDynamicDequantize(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xui8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x3x3xf32> = dense<[[[[0.4, 0.3, 0.1], [0.2, 0.3, 0.2], [0.1, 0.5, 0.2]]]]> : tensor<1x1x3x3xf32>

  %0 = IE.Convert(%weights) { dstElemType = f32 } : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x3x3xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG:  [[SCALE:%.+]] = const.Declare tensor<1x1x3x3xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[4.000000e-01, 3.000000e-01, 1.000000e-01], [2.000000e-01, 3.000000e-01, 2.000000e-01], [1.000000e-01, 5.000000e-01, 2.000000e-01]]]]>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x3x3xui8>, tensor<1x1x3x3xf32> -> tensor<4x4x3x3xf32>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  // CHECK: return [[CONV]] : tensor<1x4x28x28xf32>
}

// -----

// Shift-only block-arg chain (no Multiply): the mandatory IE.DynamicDequantize scale operand is
// synthesized as an implicit splat 1.0, mirroring the const rewriter's createImplicitUnitScale.
// CHECK-LABEL: @BlockArgPerGroupShiftToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui8>
func.func @BlockArgPerGroupShiftToDynamicDequantize(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xui8>) -> tensor<1x4x28x28xf32> {
  %shift = const.Declare tensor<1x1x3x3xf32> = dense<[[[[0.4, 0.3, 0.1], [0.2, 0.3, 0.2], [0.1, 0.5, 0.2]]]]> : tensor<1x1x3x3xf32>

  %0 = IE.Convert(%weights) { dstElemType = f32 } : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x3x3xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  return %2 : tensor<1x4x28x28xf32>

  // CHECK: [[ZP:%.+]] = const.Declare tensor<1x1x3x3xui8>
  // CHECK-SAME{LITERAL}: dense<[[[[4.000000e-01, 3.000000e-01, 1.000000e-01], [2.000000e-01, 3.000000e-01, 2.000000e-01], [1.000000e-01, 5.000000e-01, 2.000000e-01]]]]>
  // CHECK: [[SCALE:%.+]] = const.Declare tensor<4x1x1x1xf32> = dense<1.000000e+00>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x3x3xui8>, tensor<4x1x1x1xf32>, tensor<1x1x3x3xui8> -> tensor<4x4x3x3xf32>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  // CHECK: return [[CONV]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @BlockArgPerGroupScalePerAxisShiftToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui8>
func.func @BlockArgPerGroupScalePerAxisShiftToDynamicDequantize(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xui8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x3x3xf32> = dense<[[[[0.4, 0.3, 0.1], [0.2, 0.3, 0.2], [0.1, 0.5, 0.2]]]]> : tensor<1x1x3x3xf32>
  %shift = const.Declare tensor<1x1x3x1xf32> = dense<[[[[100.0], [50.0], [75.0]]]]> : tensor<1x1x3x1xf32>

  %0 = IE.Convert(%weights) { dstElemType = f32 } : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x3x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x3x3xf32> -> tensor<4x4x3x3xf32>
  %3 = IE.Convolution(%input, %2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  return %3 : tensor<1x4x28x28xf32>

  // CHECK-DAG:  [[SCALE:%.+]] = const.Declare tensor<1x1x3x3xf32>
  // CHECK-SAME{LITERAL}: dense<[[[[4.000000e-01, 3.000000e-01, 1.000000e-01], [2.000000e-01, 3.000000e-01, 2.000000e-01], [1.000000e-01, 5.000000e-01, 2.000000e-01]]]]>
  // CHECK-DAG:  [[ZP:%.+]] = const.Declare tensor<1x1x3x1xui8>
  // CHECK-SAME{LITERAL}: dense<[[[[1.000000e+02], [5.000000e+01], [7.500000e+01]]]]>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x3x3xui8>, tensor<1x1x3x3xf32>, tensor<1x1x3x1xui8> -> tensor<4x4x3x3xf32>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  // CHECK: return [[CONV]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @BlockArgPerTensorScalePerGroupShiftToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui8>
func.func @BlockArgPerTensorScalePerGroupShiftToDynamicDequantize(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xui8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.6> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x3x3xf32> = dense<[[[[40.0, 30.0, 10.0], [20.0, 30.0, 20.0], [10.0, 50.0, 20.0]]]]> : tensor<1x1x3x3xf32>

  %0 = IE.Convert(%weights) { dstElemType = f32 } : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x3x3xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %3 = IE.Convolution(%input, %2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  return %3 : tensor<1x4x28x28xf32>

  // CHECK-DAG:  [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.000000e-01>
  // CHECK-DAG:  [[ZP:%.+]] = const.Declare tensor<1x1x3x3xui8>
  // CHECK-SAME{LITERAL}: dense<[[[[4.000000e+01, 3.000000e+01, 1.000000e+01], [2.000000e+01, 3.000000e+01, 2.000000e+01], [1.000000e+01, 5.000000e+01, 2.000000e+01]]]]>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x3x3xui8>, tensor<1x1x1x1xf32>, tensor<1x1x3x3xui8> -> tensor<4x4x3x3xf32>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DDQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>
  // CHECK: return [[CONV]] : tensor<1x4x28x28xf32>
}

// -----

// CHECK-LABEL: @BlockArgPerGroupScalePerGroupShiftToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x8xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<8x2x4xui2>
func.func @BlockArgPerGroupScalePerGroupShiftToDynamicDequantize(%input: tensor<1x1x8xf16>, %weights: tensor<8x2x4xui2>) -> tensor<1x1x1x8xf16> {
  %scale = const.Declare tensor<8x2x1xf16> = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]> : tensor<16xf16>, [#const.Reshape<[8, 2, 1]>]
  %shift = const.Declare tensor<8x2x1xf16> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tensor<16xui8>, [#const.Reshape<[8, 2, 1]>, #const.CastElemType<f16>]

  %0 = IE.Convert(%weights) { dstElemType = f16 } : tensor<8x2x4xui2> -> tensor<8x2x4xf16>
  %1 = IE.Subtract(%0, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x2x4xf16>, tensor<8x2x1xf16> -> tensor<8x2x4xf16>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x2x4xf16>, tensor<8x2x1xf16> -> tensor<8x2x4xf16>
  %3 = IE.Reshape(%input) {shape_value = [1, 1, 1, 8]} : tensor<1x1x8xf16> -> tensor<1x1x1x8xf16>
  %4 = IE.Reshape(%2) {shape_value = [8, 8]} : tensor<8x2x4xf16> -> tensor<8x8xf16>
  %5 = IE.MatMul(%3, %4) {transpose_b} : tensor<1x1x1x8xf16>, tensor<8x8xf16> -> tensor<1x1x1x8xf16>

  return %5 : tensor<1x1x1x8xf16>

  // CHECK-DAG:  [[SCALE:%.+]] = const.Declare tensor<8x2x1xf16>
  // CHECK-SAME{LITERAL}: dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01, 8.999020e-01, 1.000000e+00, 1.099610e+00, 1.200200e+00, 1.299800e+00, 1.400390e+00, 1.500000e+00, 1.599610e+00]>
  // CHECK-DAG:  [[ZP:%.+]] = const.Declare tensor<8x2x1xui2>
  // CHECK-SAME{LITERAL}: dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<8x2x4xui2>, tensor<8x2x1xf16>, tensor<8x2x1xui2> -> tensor<8x2x4xf16>
  // CHECK: [[RESHAPE_ACT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 1, 8]} : tensor<1x1x8xf16> -> tensor<1x1x1x8xf16>
  // CHECK: [[RESHAPE_WGT:%.+]] = IE.Reshape([[DDQ]]) {shape_value = [8, 8]} : tensor<8x2x4xf16> -> tensor<8x8xf16>
  // CHECK: [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_ACT]], [[RESHAPE_WGT]]) {transpose_b} : tensor<1x1x1x8xf16>, tensor<8x8xf16> -> tensor<1x1x1x8xf16>

  // CHECK: return [[MATMUL]] : tensor<1x1x1x8xf16>
}

// -----

// CHECK-LABEL: @BlockArgPerGroupScaleExtraConvertToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x4xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x2x2xui2>
func.func @BlockArgPerGroupScaleExtraConvertToDynamicDequantize(%input: tensor<1x1x4xf32>, %weights: tensor<4x2x2xui2>) -> tensor<1x4xf32> {
  %scale = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]

  %0 = IE.Convert(%weights) {dstElemType = f16} : tensor<4x2x2xui2> -> tensor<4x2x2xf16>
  %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %4 = IE.Reshape(%input) {shape_value = [1, 4]} : tensor<1x1x4xf32> -> tensor<1x4xf32>
  %5 = IE.MatMul(%4, %3) {transpose_b} : tensor<1x4xf32>, tensor<4x4xf32> -> tensor<1x4xf32>

  return %5 : tensor<1x4xf32>

  // CHECK-DAG:  [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16>
  // CHECK-SAME{LITERAL}: dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]>

  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<4x2x2xui2>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  // CHECK: [[RESHAPE_WGT:%.+]] = IE.AffineReshape([[DDQ]]) {
  // CHECK-SAME{LITERAL}: dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[RESHAPE_WGT]]) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>

  // CHECK: [[RESHAPE_ACT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 4]} : tensor<1x1x4xf32> -> tensor<1x4xf32>
  // CHECK: [[MATMUL:%.+]] = IE.MatMul([[RESHAPE_ACT]], [[CONVERT_1]]) {transpose_b} : tensor<1x4xf32>, tensor<4x4xf32> -> tensor<1x4xf32>

  // CHECK: return [[MATMUL]] : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @DontBlockArgPrefillPatternToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3x4xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x2x2xui2>
func.func @DontBlockArgPrefillPatternToDynamicDequantize(%input: tensor<1x3x4xf32>, %weights: tensor<4x2x2xui2>) -> tensor<3x4xf32> {
  %scale = const.Declare tensor<4x2x1xf16> = dense<[9.997550e-02, 1.999510e-01, 3.000490e-01, 3.999020e-01, 5.000000e-01, 6.000980e-01, 7.001950e-01, 7.998050e-01]> : tensor<8xf16>, [#const.Reshape<[4, 2, 1]>]

  %0 = IE.Convert(%weights) {dstElemType = f16} : tensor<4x2x2xui2> -> tensor<4x2x2xf16>
  %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x2x2xf16>, tensor<4x2x1xf16> -> tensor<4x2x2xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1]], shape_value = [4, 4]} : tensor<4x2x2xf16> -> tensor<4x4xf16>
  %3 = IE.Convert(%2) {dstElemType = f32} : tensor<4x4xf16> -> tensor<4x4xf32>
  %4 = IE.Reshape(%input) {shape_value = [3, 4]} : tensor<1x3x4xf32> -> tensor<3x4xf32>
  %5 = IE.MatMul(%4, %3) {transpose_b} : tensor<3x4xf32>, tensor<4x4xf32> -> tensor<3x4xf32>

  return %5 : tensor<3x4xf32>

  // CHECK-NOT: vpux.weights_import_dyn_dequant
  // CHECK: [[SCALE:%.+]] = const.Declare tensor<4x2x1xf16>
  // CHECK: [[DDQ:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f16}
  // CHECK: [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[DDQ]])
  // CHECK: [[CONVERT_1:%.+]] = IE.Convert([[AFFINERESHAPE]])
  // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[INPUT]])
  // CHECK: [[MATMUL:%.+]] = IE.MatMul([[RESHAPE]], [[CONVERT_1]])
  // CHECK: return [[MATMUL]]
}

// -----

// CHECK-LABEL: @DontBlockArgPerTensorScaleToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui8>
func.func @DontBlockArgPerTensorScaleToDynamicDequantize(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xui8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %0 = IE.Convert(%weights) { dstElemType = f32 } : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // Per-tensor scale (quantized-axis count < 2): handled by the const rewriter's territory, not the
  // block-arg rewriter; the block-arg weights are a block argument, so nothing matches and it stays
  // as a plain Convert+Multiply.
  // CHECK-NOT:  IE.DynamicDequantize

  // CHECK:  [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01>
  // CHECK:  [[CONVERT:%.+]] = IE.Convert([[WEIGHTS]])
  // CHECK:  [[MUL:%.+]] = IE.Multiply([[CONVERT]], [[SCALE]])
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[MUL]])
  // CHECK:  return [[CONV]]
}

// -----

// CHECK-LABEL: @DontBlockArgPerAxisScaleToDynamicDequantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<4x4x3x3xui8>
func.func @DontBlockArgPerAxisScaleToDynamicDequantize(%input: tensor<1x4x28x28xf32>, %weights: tensor<4x4x3x3xui8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x3x1xf32> = dense<0.5> : tensor<1x1x3x1xf32>

  %0 = IE.Convert(%weights) { dstElemType = f32 } : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x3x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // Single non-one axis (quantized-axis count < 2): same territory as above, stays untouched.
  // CHECK-NOT:  IE.DynamicDequantize

  // CHECK:  [[SCALE:%.+]] = const.Declare tensor<1x1x3x1xf32> = dense<5.000000e-01>
  // CHECK:  [[CONVERT:%.+]] = IE.Convert([[WEIGHTS]])
  // CHECK:  [[MUL:%.+]] = IE.Multiply([[CONVERT]], [[SCALE]])
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[MUL]])
  // CHECK:  return [[CONV]]
}
