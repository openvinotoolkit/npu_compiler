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
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
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
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
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
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
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
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
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
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
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
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64}
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
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64}
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
  // CHECK-SAME:   {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64}
  // CHECK-NOT:  IE.DynamicDequantize
  // CHECK:      return [[FQ]]
}
