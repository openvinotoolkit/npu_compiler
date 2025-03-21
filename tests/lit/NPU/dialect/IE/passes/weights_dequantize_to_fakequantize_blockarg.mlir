//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --weights-dequantize-to-fake-quantize="enable-wd-blockarg-input=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @BlockArgMultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHT:%.+]]: tensor<4x4x3x3xui8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @BlockArgMultToFakeQuantize(%input: tensor<1x4x28x28xf32>, %weight: tensor<4x4x3x3xui8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weight) { dstElemType = f32 } : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  %1 = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.275000e+02> : tensor<1x1x1x1xf32>

  // CHECK:    [[CONV_WEIGHT:%.+]] = IE.Convert([[WEIGHT]]) {dstElemType = f32} : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV_WEIGHT]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<4x4x3x3xf32>

  // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]])

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @BlockArgSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHT:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @BlockArgSubToFakeQuantize(%input: tensor<1x4x28x28xf16>, %weight: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf16> {
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%weight) { dstElemType = f16 } : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf16>
  %1 = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %2 : tensor<1x4x28x28xf16>

  // CHECK-DAG:    [[IN_LOW:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[IN_HIGH:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_LOW:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-2.280000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_HIGH:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<2.700000e+01> : tensor<1x1x1x1xf16>

  // CHECK:    [[CONV_WEIGHT:%.+]] = IE.Convert([[WEIGHT]]) {dstElemType = f16} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf16>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV_WEIGHT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME:-> tensor<4x4x3x3xf16>

  // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]])

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @BlockArgMultSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHT:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @BlockArgMultSubToFakeQuantize(%input: tensor<1x4x28x28xf32>, %weight: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weight) { dstElemType = f32 } : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf32>
  %1 = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %3 = IE.Convolution(%input, %2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %3 : tensor<1x4x28x28xf32>

  // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.140000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.350000e+01> : tensor<1x1x1x1xf32>

  // CHECK:    [[CONV_WEIGHT:%.+]] = IE.Convert([[WEIGHT]]) {dstElemType = f32} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf32>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV_WEIGHT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<4x4x3x3xf32>

  // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]])

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @BlockArgMultiConvertMultSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHT:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @BlockArgMultiConvertMultSubToFakeQuantize(%input: tensor<1x4x28x28xf32>, %weight: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>
  %shift = const.Declare tensor<1x1x1x1xf32> = dense<100.0> : tensor<1x1x1x1xf32>

  %convert_0 = IE.Convert(%weight) { dstElemType = ui8 } : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xui8>
  %convert_1 = IE.Convert(%convert_0) { dstElemType = f16 } : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf16>
  %convert_2 = IE.Convert(%convert_1) { dstElemType = f32 } : tensor<4x4x3x3xf16> -> tensor<4x4x3x3xf32>

  %1 = IE.Subtract(%convert_2, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %3 = IE.Convolution(%input, %2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %3 : tensor<1x4x28x28xf32>

  // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.140000e+02> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.350000e+01> : tensor<1x1x1x1xf32>

  // CHECK:    [[CONV_WEIGHT:%.+]] = IE.Convert([[WEIGHT]]) {dstElemType = f32} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf32>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV_WEIGHT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME: -> tensor<4x4x3x3xf32>

  // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]])

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @BlockArgUI4MultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHT:%.+]]: tensor<4x4x3x3xui4>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @BlockArgUI4MultToFakeQuantize(%input: tensor<1x4x28x28xf32>, %weight: tensor<4x4x3x3xui4>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weight) { dstElemType = f32 } : tensor<4x4x3x3xui4> -> tensor<4x4x3x3xf32>
  %1 = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.500000e+01> : tensor<1x1x1x1xf32>
  // CHECK-DAG:    [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<7.500000e+00> : tensor<1x1x1x1xf32>

  // CHECK:    [[CONV_WEIGHT:%.+]] = IE.Convert([[WEIGHT]]) {dstElemType = f32} : tensor<4x4x3x3xui4> -> tensor<4x4x3x3xf32>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV_WEIGHT]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
  // CHECK-SAME: -> tensor<4x4x3x3xf32>

  // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]])

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @BlockArgSI4SubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHT:%.+]]: tensor<4x4x3x3xsi4>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @BlockArgSI4SubToFakeQuantize(%input: tensor<1x4x28x28xf16>, %weight: tensor<4x4x3x3xsi4>) -> tensor<1x4x28x28xf16> {
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%weight) { dstElemType = f16 } : tensor<4x4x3x3xsi4> -> tensor<4x4x3x3xf16>
  %1 = IE.Subtract(%convert, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %2 : tensor<1x4x28x28xf16>

  // CHECK-DAG:    [[IN_LOW:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-8.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[IN_HIGH:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<7.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_LOW:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-1.080000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_HIGH:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-9.300000e+01> : tensor<1x1x1x1xf16>

  // CHECK:    [[CONV_WEIGHT:%.+]] = IE.Convert([[WEIGHT]]) {dstElemType = f16} : tensor<4x4x3x3xsi4> -> tensor<4x4x3x3xf16>

  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV_WEIGHT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
  // CHECK-SAME:-> tensor<4x4x3x3xf16>

  // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]])

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @DontBlockArgConvertNoMultNoSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHT:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @DontBlockArgConvertNoMultNoSubToFakeQuantize(%input: tensor<1x4x28x28xf16>, %weight: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf16> {
  %convert = IE.Convert(%weight) { dstElemType = f16 } : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf16>
  %0 = IE.Convolution(%input, %convert) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %0 : tensor<1x4x28x28xf16>

  // CHECK:    [[CONV_WEIGHT:%.+]] = IE.Convert([[WEIGHT]]) {dstElemType = f16} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf16>
  // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CONV_WEIGHT]])

  // CHECK: return [[CONV]]
}

// -----

#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @BlockArgWithTransposeSubToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf16>
// CHECK-SAME:      [[WEIGHT:%.+]]: tensor<4x4x3x3xsi8>
// CHECK-SAME: -> tensor<1x4x28x28xf16>
func.func @BlockArgWithTransposeSubToFakeQuantize(%input: tensor<1x4x28x28xf16>, %weight: tensor<4x4x3x3xsi8>) -> tensor<1x4x28x28xf16> {
  %shift = const.Declare tensor<1x1x1x1xf16> = dense<100.0> : tensor<1x1x1x1xf16>

  %convert = IE.Convert(%weight) { dstElemType = f16 } : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf16>
  %transpose = IE.Transpose(%convert) {order_value = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>} : tensor<4x4x3x3xf16> -> tensor<4x4x3x3xf16>
  %1 = IE.Subtract(%transpose, %shift) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf16>, tensor<1x1x1x1xf16> -> tensor<4x4x3x3xf16>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf16>, tensor<4x4x3x3xf16> -> tensor<1x4x28x28xf16>

  return %2 : tensor<1x4x28x28xf16>

  // CHECK-DAG:    [[IN_LOW:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[IN_HIGH:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_LOW:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<-2.280000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[OUT_HIGH:%.+]]  = const.Declare tensor<1x1x1x1xf16> = dense<2.700000e+01> : tensor<1x1x1x1xf16>

  // CHECK:    [[CONV_WEIGHT:%.+]] = IE.Convert([[WEIGHT]]) {dstElemType = f16} : tensor<4x4x3x3xsi8> -> tensor<4x4x3x3xf16>
  // CHECK:    [[TRANSPOSE:%.+]] = IE.Transpose([[CONV_WEIGHT]]) {order_value = #map} : tensor<4x4x3x3xf16> -> tensor<4x4x3x3xf16>
  // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
  // CHECK-SAME: {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
  // CHECK-SAME:-> tensor<4x4x3x3xf16>

  // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]])

  // CHECK: return [[CONV]]
}

// -----

// CHECK-LABEL: @BlockArgNegativeMultToFakeQuantize
// CHECK-SAME:      [[INPUT:%.+]]:  tensor<1x4x28x28xf32>
// CHECK-SAME:      [[WEIGHT:%.+]]: tensor<4x4x3x3xui8>
// CHECK-SAME: -> tensor<1x4x28x28xf32>
func.func @BlockArgNegativeMultToFakeQuantize(%input: tensor<1x4x28x28xf32>, %weight: tensor<4x4x3x3xui8>) -> tensor<1x4x28x28xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<-0.5> : tensor<1x1x1x1xf32>

  %convert = IE.Convert(%weight) { dstElemType = f32 } : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  %1 = IE.Multiply(%convert, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
  %2 = IE.Convolution(%input, %1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x28x28xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x28x28xf32>

  return %2 : tensor<1x4x28x28xf32>

  // CHECK:  [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK:  [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
  // CHECK:  [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK:  [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.275000e+02> : tensor<1x1x1x1xf32>

  // CHECK:  [[CONV_WEIGHT:%.+]] = IE.Convert([[WEIGHT]]) {dstElemType = f32} : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
  // CHECK:  [[FQ:%.+]] = IE.FakeQuantize([[CONV_WEIGHT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>

  // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FQ]])

  // CHECK:  return [[CONV]] : tensor<1x4x28x28xf32>
}
