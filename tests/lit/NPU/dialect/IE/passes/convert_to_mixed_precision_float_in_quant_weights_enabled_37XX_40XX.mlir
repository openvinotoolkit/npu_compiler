//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-mixed-precision="enable-float-in-quant-weights-mixed-mode=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


!qElemType = !quant.uniform<i8:f16, 1.1534313725490195:10>
// CHECK-LABEL: @InvalidZPMixedPrecisionFloatInputQuantWeightsConv
func.func @InvalidZPMixedPrecisionFloatInputQuantWeightsConv(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %qweights = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]])
  //CHECK: [[VAL2:%.+]] = IE.Convolution(%arg0, [[VAL1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL2]]
}
