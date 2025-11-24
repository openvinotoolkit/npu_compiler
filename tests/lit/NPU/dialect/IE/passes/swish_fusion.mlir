//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swish-fusion --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// CHECK-LABEL: func @SwishFusion
// CHECK-SAME:        [[INPUT:%arg0]]: tensor<1x24x640x640xf16>
func.func @SwishFusion(%arg0: tensor<1x24x640x640xf16>) -> tensor<1x24x640x640xf16> {
  %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-10.0031395> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<10.0819044> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<0.999958157> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<-2.845580e-01> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_5 = const.Declare tensor<1x1x1x1xf16> = dense<10.0814838> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %cst_6 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x24x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x24x640x640xf16>
  %1 = IE.Sigmoid(%0) : tensor<1x24x640x640xf16> -> tensor<1x24x640x640xf16>
  %2 = IE.FakeQuantize(%1, %cst_6, %cst_3, %cst_6, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x24x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x24x640x640xf16>
  %3 = IE.Multiply(%0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x640x640xf16>, tensor<1x24x640x640xf16> -> tensor<1x24x640x640xf16>
  
  return %3 : tensor<1x24x640x640xf16>

  // CHECK:  [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-10.0031395> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:  [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<10.0819044> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
  // CHECK:  [[FAKE_QUANT:%.+]] = IE.FakeQuantize([[INPUT]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x24x640x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x24x640x640xf16>
  // CHECK:  [[SWISH:%.+]] = IE.Swish([[FAKE_QUANT]]) {beta_value = 1.000000e+00 : f64} : tensor<1x24x640x640xf16> -> tensor<1x24x640x640xf16>
  // CHECK:  return [[SWISH]] : tensor<1x24x640x640xf16>

}

// -----

// CHECK-LABEL: func @DequantizedSwishFusion
// CHECK-SAME:        [[INPUT:%arg0]]: tensor<1x128x512x512xf16>
func.func @DequantizedSwishFusion(%arg0: tensor<1x128x512x512xf16>) -> tensor<1x128x512x512xf16> {
  %1 = IE.Sigmoid(%arg0) : tensor<1x128x512x512xf16> -> tensor<1x128x512x512xf16>
  %2 = IE.Multiply(%arg0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x512x512xf16>, tensor<1x128x512x512xf16> -> tensor<1x128x512x512xf16>
  
  return %2 : tensor<1x128x512x512xf16>

  // CHECK-NOT:   IE.Sigmoid
  // CHECK-NOT:   IE.Multiply
  // CHECK:       [[SWISH:%.+]] = IE.Swish([[INPUT]]) {beta_value = 1.000000e+00 : f64} : tensor<1x128x512x512xf16> -> tensor<1x128x512x512xf16>
  // CHECK:       return [[SWISH]] : tensor<1x128x512x512xf16>
}

// -----

// CHECK-LABEL: func @NoSwishFusion
// CHECK-SAME:        [[INPUT0:%arg0]]: tensor<2x1x2x3x2x121xf16>, [[INPUT1:%arg1]]: tensor<2x2x2x3x2x121xf16>
func.func @NoSwishFusion(%arg0: tensor<2x1x2x3x2x121xf16>, %arg1: tensor<2x2x2x3x2x121xf16> ) -> tensor<1x2x12x121xf16> {
  %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [2], [2], [3]], shape_value = [1, 2, 12, 121]} : tensor<2x1x2x3x2x121xf16> -> tensor<1x2x12x121xf16>
  %2 = IE.Slice %arg1 [0, 1, 0, 0, 0, 0] [2, 1, 2, 3, 2, 121] : tensor<2x2x2x3x2x121xf16> to tensor<2x1x2x3x2x121xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1], [2], [2], [2], [2], [3]], shape_value = [1, 2, 12, 121]} : tensor<2x1x2x3x2x121xf16> -> tensor<1x2x12x121xf16>
  %4 = IE.Sigmoid(%3) : tensor<1x2x12x121xf16> -> tensor<1x2x12x121xf16>
  %5 = IE.Multiply(%1, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x12x121xf16>, tensor<1x2x12x121xf16> -> tensor<1x2x12x121xf16>

  return %5 : tensor<1x2x12x121xf16>

  // CHECK:   [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]]) 
  //CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [2], [2], [2], [3]], shape_value = [1, 2, 12, 121]} :

  // CHECK:   [[SLICE:%.+]] = IE.Slice [[INPUT1]] [0, 1, 0, 0, 0, 0] [2, 1, 2, 3, 2, 121] : tensor<2x2x2x3x2x121xf16> to tensor<2x1x2x3x2x121xf16>
  // CHECK:   [[RESHAPE1:%.+]] = IE.AffineReshape([[SLICE]]) 
  //CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [2], [2], [2], [3]], shape_value = [1, 2, 12, 121]} :

  // CHECK:   [[SIGMOID:%.+]] = IE.Sigmoid([[RESHAPE1]]) : tensor<1x2x12x121xf16> -> tensor<1x2x12x121xf16>
  // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[RESHAPE0]], [[SIGMOID]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x12x121xf16>, tensor<1x2x12x121xf16> -> tensor<1x2x12x121xf16>
  // CHECK:   return [[MULTIPLY]] : tensor<1x2x12x121xf16>

}
