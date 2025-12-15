//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-dynamic-dequantize-to-dequantize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

!qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<f8E4M3FN:f16, 1.562500e-02>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForF8E4M3FNWeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960xf8E4M3FN>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForF8E4M3FNWeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960xf8E4M3FN>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<1536x8960xf8E4M3FN> -> tensor<1536x8960x!qElemType>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>

    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<6.400000e+01> : tensor<1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960xf8E4M3FN> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}

// -----

!qElemType = !quant.uniform<f8E5M2:f16, 1.000000e+00>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<f8E5M2:f16, 1.220703125E-4>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @RescaleForF8E5M2WeightsAsInputs
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1024x8960xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1536x8960xf8E5M2>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1536x1xf16>
func.func @RescaleForF8E5M2WeightsAsInputs(%arg0: tensor<1024x8960xf16>, %arg1: tensor<1536x8960xf8E5M2>, %arg2: tensor<1536x1xf16>) -> tensor<1024x1536xf16> {
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<1536x8960xf8E5M2> -> tensor<1536x8960x!qElemType>
    %1 = IE.DynamicDequantize(%0, %arg2) {dstElemType = f16} : tensor<1536x8960x!qElemType>, tensor<1536x1xf16> -> tensor<1536x8960xf16>
    %2 = IE.FullyConnected(%arg0, %1) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>

    return %2 : tensor<1024x1536xf16>

    // CHECK:       [[CONST0:%.+]] = const.Declare tensor<1xf16> = dense<8.192000e+03> : tensor<1xf16>
    // CHECK:       [[RESHAPE0:%.+]] = IE.Reshape([[INPUT_2]]) {shape_value = [1, 1536]} : tensor<1536x1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = [[QELEMTYPE_OUT]]} : tensor<1536x8960xf8E5M2> -> tensor<1536x8960x[[QELEMTYPE_OUT]]>
    // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[RESHAPE0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536xf16>, tensor<1xf16> -> tensor<1x1536xf16>
    // CHECK:       [[DEQUANTIZE0:%.+]] = IE.Dequantize([[QUANTIZECAST0]]) {dstElemType = f16} : tensor<1536x8960x[[QELEMTYPE_OUT]]> -> tensor<1536x8960xf16>
    // CHECK:       [[FULLYCONNECTED0:%.+]] = IE.FullyConnected([[INPUT_0]], [[DEQUANTIZE0]]) : tensor<1024x8960xf16>, tensor<1536x8960xf16> -> tensor<1024x1536xf16>
    // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[FULLYCONNECTED0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1536xf16>, tensor<1x1536xf16> -> tensor<1024x1536xf16>
    // CHECK:       return [[MULTIPLY1]] : tensor<1024x1536xf16>
}
