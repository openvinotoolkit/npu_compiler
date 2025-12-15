//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE ./IR/mixed_precision_conv_nf4_wac.xml -o %t
// RUN: FileCheck %s --input-file %t
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// Verifying that QuantileFloat const.Declare dense_resource is actually stored as u4 raw data

// CHECK: module @MixedPrecisionConvolutionNF4 {
// CHECK:   net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:       DataInfo "Parameter_10" : tensor<1x16x16x16xf16>
// CHECK:   } outputsInfo : {
// CHECK:       DataInfo "Convolution_15" friendlyName = "Result_16" : tensor<1x16x16x16xf16>
// CHECK:   }
// CHECK:   func.func @main([[ARG0:[^:]+]]: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {

// CHECK:       [[CONST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f16
// CHECK-SAME:       = dense_resource<vpux_ow_0> : tensor<16x16x1x1xui4>
// CHECK-SAME:       [#const.ConvertElemType<ui8>, #const.CastElemType<!QuantileFloat.quantileFloat<ui4:f16,

// CHECK:       [[CONVERT:%.+]] = IE.Convert([[CONST_WEIGHTS]])
// CHECK-SAME:      {dstElemType = f16}
// CHECK-SAME:      : tensor<16x16x1x1x!QuantileFloat.quantileFloat<ui4:f16
// CHECK-SAME:      -> tensor<16x16x1x1xf16>

// CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<7.873530e-02> : tensor<16x1x1x1xf16>

// CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT]], [[CST]]) {
// CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
// CHECK-SAME:  } : tensor<16x16x1x1xf16>, tensor<16x1x1x1xf16> -> tensor<16x16x1x1xf16>

// CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[MULTIPLY]]) {
// CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
// CHECK-SAME:  } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

// CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
// CHECK:   }
// CHECK: }
