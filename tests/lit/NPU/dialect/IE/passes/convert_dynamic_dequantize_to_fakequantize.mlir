//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-dynamic-dequantize-to-fake-quantize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU5010

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @WeightsImportDynDeqQuantI8ToFakeQuantize
func.func @WeightsImportDynDeqQuantI8ToFakeQuantize() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4x!qElemType> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<!qElemType>]
  %scale = const.Declare tensor<4x1xf32> = dense<2.000000e+00> : tensor<4x1xf32>
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4x!qElemType>, tensor<4x1xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x4xf32> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<!qElemType>, #const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<-2.560000e+02>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<2.540000e+02>

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64,
  // CHECK-SAME:    vpux.fq_precise_scales = array<f32: 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00>, vpux.fq_precise_zero_points = array<i32: 0, 0, 0, 0>}
  // CHECK-SAME:   -> tensor<4x4xf32>
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// CHECK-LABEL: @WeightsImportDynDeqRawI8ToFakeQuantize
func.func @WeightsImportDynDeqRawI8ToFakeQuantize() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4xsi8> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>]
  %scale = const.Declare tensor<4x1xf32> = dense<2.000000e+00> : tensor<4x1xf32>
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4xsi8>, tensor<4x1xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x4xf32> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<-2.560000e+02>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<2.540000e+02>

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
  // CHECK-SAME:    vpux.fq_precise_scales = array<f32: 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00>, vpux.fq_precise_zero_points = array<i32: 0, 0, 0, 0>}
  // CHECK-SAME:   -> tensor<4x4xf32>
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// CHECK-LABEL: @WeightsImportDynDeqBlockArgToFakeQuantize
// CHECK-SAME:      ([[ARG0:%.+]]: tensor<4x8xf32>, [[WEIGHTS:%.+]]: tensor<4x8xsi8>)
func.func @WeightsImportDynDeqBlockArgToFakeQuantize(%arg0: tensor<4x8xf32>, %weights: tensor<4x8xsi8>) -> tensor<4x8xf32> {
  %scale = const.Declare tensor<4x1xf32> = dense<2.000000e+00> : tensor<4x1xf32>
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x8xsi8>, tensor<4x1xf32> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>

  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<-2.560000e+02>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<2.540000e+02>

  // CHECK:      [[WT:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f32} : tensor<4x8xsi8> -> tensor<4x8xf32>
  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
  // CHECK-SAME:    vpux.fq_precise_scales = array<f32: 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00>, vpux.fq_precise_zero_points = array<i32: 0, 0, 0, 0>}
  // CHECK-SAME:   -> tensor<4x8xf32>
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// CHECK-LABEL: @WeightsImportDynDeqRawI8WithZPToFakeQuantize
func.func @WeightsImportDynDeqRawI8WithZPToFakeQuantize() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4xsi8> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>]
  %scale = const.Declare tensor<4x1xf32> = dense<2.000000e+00> : tensor<4x1xf32>
  %zp = const.Declare tensor<4x1xsi8> = dense<10> : tensor<4x1xsi8>
  %0 = IE.DynamicDequantize(%weights, %scale, %zp) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4xsi8>, tensor<4x1xf32>, tensor<4x1xsi8> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x4xf32> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<-2.760000e+02>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<2.340000e+02>

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
  // CHECK-SAME:    vpux.fq_precise_scales = array<f32: 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00>, vpux.fq_precise_zero_points = array<i32: 10, 10, 10, 10>}
  // CHECK-SAME:   -> tensor<4x4xf32>
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @WeightsImportDynDeqI4ToFakeQuantize
func.func @WeightsImportDynDeqI4ToFakeQuantize() -> tensor<4x8xf32> {
  %weights = const.Declare tensor<4x8x!qElemType> = dense<1> : tensor<4x8xsi4>, [#const.CastElemType<f32>, #const.CastElemType<!qElemType>]
  %scale = const.Declare tensor<4x1xf32> = dense<1.500000e+00> : tensor<4x1xf32>
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x8x!qElemType>, tensor<4x1xf32> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x8xf32> = dense<1> : tensor<4x8xsi4>, [#const.CastElemType<f32>, #const.CastElemType<!qElemType>, #const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-8.000000e+00>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<7.000000e+00>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<-1.200000e+01>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<1.050000e+01>

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64,
  // CHECK-SAME:   vpux.fq_precise_scales = array<f32: 1.500000e+00, 1.500000e+00, 1.500000e+00, 1.500000e+00>, vpux.fq_precise_zero_points = array<i32: 0, 0, 0, 0>}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @KeepUnmarkedDynDeq
func.func @KeepUnmarkedDynDeq() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4x!qElemType> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<!qElemType>]
  %scale = const.Declare tensor<4x1xf32> = dense<2.000000e+00> : tensor<4x1xf32>
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32} : tensor<4x4x!qElemType>, tensor<4x1xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK:      IE.DynamicDequantize
  // CHECK-NOT:  IE.FakeQuantize
}

// -----

// A synthetic-from-Dequantize DynamicDequantize (different marker) must also be left untouched.

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK-LABEL: @KeepSyntheticDynDeq
func.func @KeepSyntheticDynDeq() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4x!qElemType> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<!qElemType>]
  %scale = const.Declare tensor<4x1xf32> = dense<2.000000e+00> : tensor<4x1xf32>
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.synthetic_dyn_dequant} : tensor<4x4x!qElemType>, tensor<4x1xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK:      IE.DynamicDequantize
  // CHECK-NOT:  IE.FakeQuantize
}

// -----

// CHECK-LABEL: @WeightsImportDynDeqRawU2ToFakeQuantize
func.func @WeightsImportDynDeqRawU2ToFakeQuantize() -> tensor<4x8xf32> {
  %weights = const.Declare tensor<4x8xui2> = dense<3> : tensor<4x8xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<ui2>]
  %scale = const.Declare tensor<4x1xf32> = dense<2.000000e+00> : tensor<4x1xf32>
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x8xui2>, tensor<4x1xf32> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x8xf32> = dense<3> : tensor<4x8xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<0.000000e+00>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<3.000000e+00>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<0.000000e+00>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<6.000000e+00>
  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64
  // CHECK-SAME:   vpux.fq_precise_scales = array<f32: 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00>, vpux.fq_precise_zero_points = array<i32: 0, 0, 0, 0>}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// CHECK-LABEL: @WeightsImportDynDeqRawI2ToFakeQuantize
func.func @WeightsImportDynDeqRawI2ToFakeQuantize() -> tensor<4x8xf32> {
  %weights = const.Declare tensor<4x8xsi2> = dense<1> : tensor<4x8xsi2>, [#const.ConvertElemType<si8>, #const.CastElemType<si2>]
  %scale = const.Declare tensor<4x1xf32> = dense<2.000000e+00> : tensor<4x1xf32>
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x8xsi2>, tensor<4x1xf32> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x8xf32> = dense<1> : tensor<4x8xsi2>, [#const.ConvertElemType<si8>, #const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-2.000000e+00>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.000000e+00>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<-4.000000e+00>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<2.000000e+00>
  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64
  // CHECK-SAME:   vpux.fq_precise_scales = array<f32: 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00>, vpux.fq_precise_zero_points = array<i32: 0, 0, 0, 0>}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// CHECK-LABEL: @WeightsImportDynDeqRawU2WithZPToFakeQuantize
func.func @WeightsImportDynDeqRawU2WithZPToFakeQuantize() -> tensor<4x8xf32> {
  %weights = const.Declare tensor<4x8xui2> = dense<3> : tensor<4x8xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<ui2>]
  %scale = const.Declare tensor<4x1xf32> = dense<2.000000e+00> : tensor<4x1xf32>
  %zp = const.Declare tensor<4x1xui2> = dense<1> : tensor<4x1xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<ui2>]
  %0 = IE.DynamicDequantize(%weights, %scale, %zp) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x8xui2>, tensor<4x1xf32>, tensor<4x1xui2> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x8xf32> = dense<3> : tensor<4x8xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<0.000000e+00>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<3.000000e+00>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<-2.000000e+00>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<4.000000e+00>
  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64
  // The unsigned sub-byte ZP operand must be zero-extended, not reinterpreted as signed.
  // CHECK-SAME:   vpux.fq_precise_zero_points = array<i32: 1, 1, 1, 1>}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// CHECK-LABEL: @WeightsImportDynDeqSubViewScalePreservesChain
func.func @WeightsImportDynDeqSubViewScalePreservesChain() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4xsi8> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>]
  %scale = const.Declare tensor<4x1xf32> = dense<[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]> : tensor<8x1xf32>, [#const.SubView<[4, 0], [4, 1]>]
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4xsi8>, tensor<4x1xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x4xf32> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[1.270000e+02], [2.540000e+02], [3.810000e+02], [5.080000e+02], [6.350000e+02], [7.620000e+02], [8.890000e+02], [1.016000e+03]]> : tensor<8x1xf32>, [#const.SubView<[4, 0], [4, 1]>]
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[-1.280000e+02], [-2.560000e+02], [-3.840000e+02], [-5.120000e+02], [-6.400000e+02], [-7.680000e+02], [-8.960000e+02], [-1.024000e+03]]> : tensor<8x1xf32>, [#const.SubView<[4, 0], [4, 1]>]

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64,
  // CHECK-SAME:   vpux.fq_precise_scales = array<f32: 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00>,
  // CHECK-SAME:   vpux.fq_precise_zero_points = array<i32: 0, 0, 0, 0>}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// A splat scale must stay splat: the bounds are built from the single splat value instead of being
// expanded to the full element count.

// CHECK-LABEL: @WeightsImportDynDeqSplatScaleStaysSplat
func.func @WeightsImportDynDeqSplatScaleStaysSplat() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4xsi8> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>]
  %scale = const.Declare tensor<4x1xf32> = dense<2.000000e+00> : tensor<4x1xf32>
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4xsi8>, tensor<4x1xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x4xf32> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<2.540000e+02> : tensor<4x1xf32>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<-2.560000e+02> : tensor<4x1xf32>

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
  // CHECK-SAME:   vpux.fq_precise_scales = array<f32: 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00>, vpux.fq_precise_zero_points = array<i32: 0, 0, 0, 0>}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// CHECK-LABEL: @WeightsImportDynDeqNonCommutingScaleChainIsFolded
func.func @WeightsImportDynDeqNonCommutingScaleChainIsFolded() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4xsi8> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>]
  %scale = const.Declare tensor<4x1xf32> = dense<[[1.0], [2.0], [3.0], [4.0]]> : tensor<4x1xf32>, [#const.Add<1.000000e+00 : f64>]
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4xsi8>, tensor<4x1xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x4xf32> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[-2.560000e+02], [-3.840000e+02], [-5.120000e+02], [-6.400000e+02]]> : tensor<4x1xf32>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[2.540000e+02], [3.810000e+02], [5.080000e+02], [6.350000e+02]]> : tensor<4x1xf32>

  // The re-attached chain must be gone -- bounds are fully materialized.
  // CHECK-NOT:  #const.Add
  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
  // CHECK-SAME:   vpux.fq_precise_scales = array<f32: 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00>, vpux.fq_precise_zero_points = array<i32: 0, 0, 0, 0>}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// A non-f32 scale base cannot be reinterpreted as f32, so the chain is not re-attached and the bounds are
// materialized instead.

// CHECK-LABEL: @WeightsImportDynDeqNonF32ScaleBaseIsFolded
func.func @WeightsImportDynDeqNonF32ScaleBaseIsFolded() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4xsi8> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>]
  %scale = const.Declare tensor<4x1xf32> = dense<[[1.0], [2.0], [3.0], [4.0]]> : tensor<4x1xf16>, [#const.CastElemType<f32>]
  %0 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4xsi8>, tensor<4x1xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x4xf32> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[-1.280000e+02], [-2.560000e+02], [-3.840000e+02], [-5.120000e+02]]> : tensor<4x1xf32>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[1.270000e+02], [2.540000e+02], [3.810000e+02], [5.080000e+02]]> : tensor<4x1xf32>

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
  // The f16 scale base is read through the storage-type conversion, not reinterpreted as f32.
  // CHECK-SAME:   vpux.fq_precise_scales = array<f32: 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00>
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// Per-channel (non-splat) ZP cannot be absorbed into a scalar factor, so the pass falls back to
// materializing the bounds via revertScaleShift.

// CHECK-LABEL: @WeightsImportDynDeqPerChannelZPIsFolded
func.func @WeightsImportDynDeqPerChannelZPIsFolded() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4xsi8> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>]
  %scale = const.Declare tensor<4x1xf32> = dense<[[1.0], [2.0], [3.0], [4.0]]> : tensor<4x1xf32>
  %zp = const.Declare tensor<4x1xsi8> = dense<[[1], [2], [3], [4]]> : tensor<4x1xsi8>
  %0 = IE.DynamicDequantize(%weights, %scale, %zp) {dstElemType = f32, vpux.weights_import_dyn_dequant}
      : tensor<4x4xsi8>, tensor<4x1xf32>, tensor<4x1xsi8> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x4xf32> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[-1.290000e+02], [-2.600000e+02], [-3.930000e+02], [-5.280000e+02]]> : tensor<4x1xf32>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[1.260000e+02], [2.500000e+02], [3.720000e+02], [4.920000e+02]]> : tensor<4x1xf32>

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
  // CHECK-SAME:   vpux.fq_precise_scales = array<f32: 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00>, vpux.fq_precise_zero_points = array<i32: 1, 2, 3, 4>}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// Per-channel scale with a per-tensor ZP: the scalar ZP is replicated so that the cached arrays hold one
// zero-point per scale, which is what SplitFakeQuant expects.

// CHECK-LABEL: @WeightsImportDynDeqPerTensorZPIsBroadcastToScales
func.func @WeightsImportDynDeqPerTensorZPIsBroadcastToScales() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4xsi8> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>]
  %scale = const.Declare tensor<4x1xf32> = dense<[[1.0], [2.0], [3.0], [4.0]]> : tensor<4x1xf32>
  %zp = const.Declare tensor<1x1xsi8> = dense<10> : tensor<1x1xsi8>
  %0 = IE.DynamicDequantize(%weights, %scale, %zp) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4xsi8>, tensor<4x1xf32>, tensor<1x1xsi8> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x4xf32> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[-1.380000e+02], [-2.760000e+02], [-4.140000e+02], [-5.520000e+02]]> : tensor<4x1xf32>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[1.170000e+02], [2.340000e+02], [3.510000e+02], [4.680000e+02]]> : tensor<4x1xf32>

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
  // CHECK-SAME:   vpux.fq_precise_scales = array<f32: 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00>, vpux.fq_precise_zero_points = array<i32: 10, 10, 10, 10>}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// Mirror case: a per-tensor scale with a per-channel ZP replicates the scale instead.

// CHECK-LABEL: @WeightsImportDynDeqPerTensorScaleIsBroadcastToZPs
func.func @WeightsImportDynDeqPerTensorScaleIsBroadcastToZPs() -> tensor<4x4xf32> {
  %weights = const.Declare tensor<4x4xsi8> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>]
  %scale = const.Declare tensor<1x1xf32> = dense<2.000000e+00> : tensor<1x1xf32>
  %zp = const.Declare tensor<4x1xsi8> = dense<[[1], [2], [3], [4]]> : tensor<4x1xsi8>
  %0 = IE.DynamicDequantize(%weights, %scale, %zp) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x4xsi8>, tensor<1x1xf32>, tensor<4x1xsi8> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>

  // CHECK-DAG:  [[WT:%.+]] = const.Declare tensor<4x4xf32> = dense<3> : tensor<4x4xsi8>, [#const.CastElemType<f32>]
  // CHECK-DAG:  [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.280000e+02>
  // CHECK-DAG:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
  // CHECK-DAG:  [[OUT_LOW:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[-2.580000e+02], [-2.600000e+02], [-2.620000e+02], [-2.640000e+02]]> : tensor<4x1xf32>
  // CHECK-DAG:  [[OUT_HIGH:%.+]] = const.Declare tensor<4x1xf32> = dense<{{\[}}[2.520000e+02], [2.500000e+02], [2.480000e+02], [2.460000e+02]]> : tensor<4x1xf32>

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
  // CHECK-SAME:   vpux.fq_precise_scales = array<f32: 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00>, vpux.fq_precise_zero_points = array<i32: 1, 2, 3, 4>}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}

// -----

// Scale and ZP broadcasting along different axes cannot be paired into flat per-channel arrays, so the
// precise-parameter attributes are dropped and SplitFakeQuant recomputes them from the output limits.

// CHECK-LABEL: @WeightsImportDynDeqCrossAxisBroadcastDropsPreciseParams
func.func @WeightsImportDynDeqCrossAxisBroadcastDropsPreciseParams() -> tensor<4x8xf32> {
  %weights = const.Declare tensor<4x8xsi8> = dense<3> : tensor<4x8xsi8>, [#const.CastElemType<f32>, #const.CastElemType<si8>]
  %scale = const.Declare tensor<4x1xf32> = dense<[[1.0], [2.0], [3.0], [4.0]]> : tensor<4x1xf32>
  %zp = const.Declare tensor<1x8xsi8> = dense<[[1, 2, 3, 4, 5, 6, 7, 8]]> : tensor<1x8xsi8>
  %0 = IE.DynamicDequantize(%weights, %scale, %zp) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<4x8xsi8>, tensor<4x1xf32>, tensor<1x8xsi8> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>

  // CHECK:      [[FQ:%.+]] = IE.FakeQuantize
  // CHECK-NOT:  vpux.fq_precise_scales
  // CHECK-NOT:  vpux.fq_precise_zero_points
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}
