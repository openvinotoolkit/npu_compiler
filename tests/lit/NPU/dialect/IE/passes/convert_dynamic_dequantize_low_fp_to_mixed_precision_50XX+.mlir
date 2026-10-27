//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-dq-raw-data-type-to-quantized --convert-const-dynamic-dequantize-to-dequantize --convert-to-mixed-precision="enable-float-in-quant-weights-mixed-mode=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010

// Behavioural guard for the signedness of quant types rebuilt from low-FP storage types.
//
// ConvertDQRawDataTypeToQuantized and ConvertConstantDynamicDequantizeToDequantize rebuild a
// quant.uniform type around raw weights. MixedFloatInQuantWeightsRewriter only accepts signed quant
// types, so dropping mlir::quant::QuantizationFlags::Signed for low-FP storage leaves the
// IE.Dequantize in place and DequantizeConst then materializes the weights to f16
//
// Flag can't be checked directly through the IR printing so the tests below check for the presence of the signedness Quantization flag.

// -----
// Per-tensor f8E4M3FN weights: mixed precision consumes the dequantize and the convolution keeps
// quantized weights

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16, 2.500000e-01>
// CHECK-LABEL: @PerTensorF8E4M3FNWeights
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x64x1x1xf16>)
func.func @PerTensorF8E4M3FNWeights(%arg0: tensor<1x64x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %wt = const.Declare tensor<16x64x1x1xf8E4M3FN> = dense<1.000000e+00> : tensor<16x64x1x1xf8E4M3FN>
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.25> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %dq = IE.DynamicDequantize(%wt, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<16x64x1x1xf8E4M3FN>, tensor<1x1x1x1xf16> -> tensor<16x64x1x1xf16>
    %conv = IE.Convolution(%arg0, %dq) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
            : tensor<1x64x1x1xf16>, tensor<16x64x1x1xf16> -> tensor<1x16x1x1xf16>
    return %conv : tensor<1x16x1x1xf16>

    // CHECK:       [[WT:%.+]] = const.Declare tensor<16x64x1x1x!qElemType> = dense<1.000000e+00> : tensor<16x64x1x1xf8E4M3FN>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WT]])
    // CHECK-SAME:      tensor<1x64x1x1xf16>, tensor<16x64x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x16x1x1xf16>
    // CHECK-NOT:   IE.Dequantize
    // CHECK-NOT:   IE.DynamicDequantize
}

// -----
// Per-axis f8E4M3FN weights: same expectation with one scale per channel

// CHECK: !qElemType = !quant.uniform<f8E4M3FN:f16:0, {2.500000e-01,5.000000e-01,7.500000e-01,1.000000e+00}>
// CHECK-LABEL: @PerAxisF8E4M3FNWeights
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x64x1x1xf16>)
func.func @PerAxisF8E4M3FNWeights(%arg0: tensor<1x64x1x1xf16>) -> tensor<1x4x1x1xf16> {
    %wt = const.Declare tensor<4x64x1x1xf8E4M3FN> = dense<1.000000e+00> : tensor<4x64x1x1xf8E4M3FN>
    %scale = const.Declare tensor<4x1x1x1xf16> = dense<[[[[0.25]]], [[[0.5]]], [[[0.75]]], [[[1.0]]]]> : tensor<4x1x1x1xf32>, [#const.CastElemType<f16>]
    %dq = IE.DynamicDequantize(%wt, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<4x64x1x1xf8E4M3FN>, tensor<4x1x1x1xf16> -> tensor<4x64x1x1xf16>
    %conv = IE.Convolution(%arg0, %dq) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
            : tensor<1x64x1x1xf16>, tensor<4x64x1x1xf16> -> tensor<1x4x1x1xf16>
    return %conv : tensor<1x4x1x1xf16>

    // CHECK:       [[WT:%.+]] = const.Declare tensor<4x64x1x1x!qElemType> = dense<1.000000e+00> : tensor<4x64x1x1xf8E4M3FN>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WT]])
    // CHECK-SAME:      tensor<1x64x1x1xf16>, tensor<4x64x1x1x!qElemType> -> tensor<1x4x1x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x4x1x1xf16>
    // CHECK-NOT:   IE.Dequantize
    // CHECK-NOT:   IE.DynamicDequantize
}

// -----
// Per-tensor f8E5M2 weights: the second low-FP storage type takes the same path.

// CHECK: !qElemType = !quant.uniform<f8E5M2:f16, 2.500000e-01>
// CHECK-LABEL: @PerTensorF8E5M2Weights
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x64x1x1xf16>)
func.func @PerTensorF8E5M2Weights(%arg0: tensor<1x64x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %wt = const.Declare tensor<16x64x1x1xf8E5M2> = dense<1.000000e+00> : tensor<16x64x1x1xf8E5M2>
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.25> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %dq = IE.DynamicDequantize(%wt, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<16x64x1x1xf8E5M2>, tensor<1x1x1x1xf16> -> tensor<16x64x1x1xf16>
    %conv = IE.Convolution(%arg0, %dq) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
            : tensor<1x64x1x1xf16>, tensor<16x64x1x1xf16> -> tensor<1x16x1x1xf16>
    return %conv : tensor<1x16x1x1xf16>

    // CHECK:       [[WT:%.+]] = const.Declare tensor<16x64x1x1x!qElemType> = dense<1.000000e+00> : tensor<16x64x1x1xf8E5M2>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WT]])
    // CHECK-SAME:      tensor<1x64x1x1xf16>, tensor<16x64x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x16x1x1xf16>
    // CHECK-NOT:   IE.Dequantize
    // CHECK-NOT:   IE.DynamicDequantize
}

// -----
// Negative: unsigned integer weights must not be fused, since MixedFloatInQuantWeightsRewriter only
// accepts signed quant types

// CHECK: !qElemType = !quant.uniform<u8:f16, 2.500000e-01>
// CHECK-LABEL: @UnsignedWeightsNotFused
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x64x1x1xf16>)
func.func @UnsignedWeightsNotFused(%arg0: tensor<1x64x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %wt = const.Declare tensor<16x64x1x1xui8> = dense<1> : tensor<16x64x1x1xui8>
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.25> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %dq = IE.DynamicDequantize(%wt, %scale) {dstElemType = f16, vpux.synthetic_dyn_dequant}
            : tensor<16x64x1x1xui8>, tensor<1x1x1x1xf16> -> tensor<16x64x1x1xf16>
    %conv = IE.Convolution(%arg0, %dq) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
            : tensor<1x64x1x1xf16>, tensor<16x64x1x1xf16> -> tensor<1x16x1x1xf16>
    return %conv : tensor<1x16x1x1xf16>

    // CHECK:       [[WT:%.+]] = const.Declare tensor<16x64x1x1x!qElemType> = dense<1> : tensor<16x64x1x1xui8>
    // CHECK:       [[DEQU:%.+]] = IE.Dequantize([[WT]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<16x64x1x1x!qElemType> -> tensor<16x64x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQU]])
    // CHECK-SAME:      tensor<1x64x1x1xf16>, tensor<16x64x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       return [[CONV]] : tensor<1x16x1x1xf16>
}
