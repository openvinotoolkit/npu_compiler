//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-dequantize-to-dynamic-dequantize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!qElemType = !quant.uniform<i8:f16, 5.000000e-01>

// CHECK-LABEL: @PerTensorSymmetricI8ZeroPointF16Scale
func.func @PerTensorSymmetricI8ZeroPointF16Scale() -> tensor<4x2x1x1xf16> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<1> : tensor<4x2x1x1xi8>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f16}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf16>
    return %0 : tensor<4x2x1x1xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xsi8> = dense<1> : tensor<4x2x1x1xi8>, [#const.CastElemType<!qElemType>, #const.CastElemType<si8>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e-01>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<1x1x1x1xsi8> = dense<0>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x2x1x1xsi8>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi8> -> tensor<4x2x1x1xf16>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf16>
}

// -----
!qElemType = !quant.uniform<u8:f32, 5.000000e-01:255>

// CHECK-LABEL: @PerTensorAsymmetricU8ZeroPointF32Scale
func.func @PerTensorAsymmetricU8ZeroPointF32Scale() -> tensor<4x2x1x1xf32> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<1> : tensor<4x2x1x1xui8>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f32}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf32>
    return %0 : tensor<4x2x1x1xf32>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xui8> = dense<1> : tensor<4x2x1x1xui8>, [#const.CastElemType<!qElemType>, #const.CastElemType<ui8>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<1x1x1x1xui8> = dense<255>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f32}
    // CHECK-SAME:      tensor<4x2x1x1xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8> -> tensor<4x2x1x1xf32>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf32>
}

// -----
!qElemType = !quant.uniform<i2:f16, 5.000000e-01:-2>

// CHECK-LABEL: @PerTensorAsymmetricI2ZeroPointF16Scale
func.func @PerTensorAsymmetricI2ZeroPointF16Scale() -> tensor<4x2x1x1xf16> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<1> : tensor<4x2x1x1xi2>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f16}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf16>
    return %0 : tensor<4x2x1x1xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xsi2> = dense<1> : tensor<4x2x1x1xi2>, [#const.CastElemType<!qElemType>, #const.CastElemType<si2>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e-01>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<1x1x1x1xsi2> = dense<-2>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x2x1x1xsi2>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi2> -> tensor<4x2x1x1xf16>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf16>
}

// -----
!qElemType = !quant.uniform<u2:f16, 5.000000e-01:3>

// CHECK-LABEL: @PerTensorAsymmetricU2ZeroPointF16Scale
func.func @PerTensorAsymmetricU2ZeroPointF16Scale() -> tensor<4x2x1x1xf16> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<1> : tensor<4x2x1x1xui2>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f16}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf16>
    return %0 : tensor<4x2x1x1xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xui2> = dense<1> : tensor<4x2x1x1xui2>, [#const.CastElemType<!qElemType>, #const.CastElemType<ui2>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e-01>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<1x1x1x1xui2> = dense<3>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x2x1x1xui2>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xui2> -> tensor<4x2x1x1xf16>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf16>
}

// -----
!qElemType = !quant.uniform<i4:f16, 5.000000e-01:-5>

// CHECK-LABEL: @PerTensorAsymmetricI4ZeroPointF16Scale
func.func @PerTensorAsymmetricI4ZeroPointF16Scale() -> tensor<4x2x1x1xf16> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<1> : tensor<4x2x1x1xi4>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f16}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf16>
    return %0 : tensor<4x2x1x1xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xsi4> = dense<1> : tensor<4x2x1x1xi4>, [#const.CastElemType<!qElemType>, #const.CastElemType<si4>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e-01>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<1x1x1x1xsi4> = dense<-5>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x2x1x1xsi4>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi4> -> tensor<4x2x1x1xf16>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf16>
}

// -----
!qElemType = !quant.uniform<ui4:f16, 5.000000e-01:5>

// CHECK-LABEL: @PerTensorAsymmetricUI4ZeroPointF16Scale
func.func @PerTensorAsymmetricUI4ZeroPointF16Scale() -> tensor<4x2x1x1xf16> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<1> : tensor<4x2x1x1xui4>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f16}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf16>
    return %0 : tensor<4x2x1x1xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xui4> = dense<1> : tensor<4x2x1x1xui4>, [#const.CastElemType<!qElemType>, #const.CastElemType<ui4>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e-01>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<1x1x1x1xui4> = dense<5>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x2x1x1xui4>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xui4> -> tensor<4x2x1x1xf16>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf16>
}

// -----
!qElemType = !quant.uniform<i8:f32:0, {1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01}>

// CHECK-LABEL: @PerChannelSymmetricI8ZeroPointF32Scale
func.func @PerChannelSymmetricI8ZeroPointF32Scale() -> tensor<4x2x1x1xf32> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<1> : tensor<4x2x1x1xi8>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f32}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf32>
    return %0 : tensor<4x2x1x1xf32>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xsi8> = dense<1> : tensor<4x2x1x1xi8>, [#const.CastElemType<!qElemType>, #const.CastElemType<si8>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<4x1x1x1xf32>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<4x1x1x1xsi8> = dense<0> : tensor<4x1x1x1xsi8>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f32}
    // CHECK-SAME:      tensor<4x2x1x1xsi8>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xsi8> -> tensor<4x2x1x1xf32>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf32>
}

// -----
!qElemType = !quant.uniform<u8:f16:0, {1.000000e-01:5,2.000000e-01:10,3.000000e-01:15,4.000000e-01:20}>

// CHECK-LABEL: @PerChannelAsymmetricU8ZeroPointF16Scale
func.func @PerChannelAsymmetricU8ZeroPointF16Scale() -> tensor<4x2x1x1xf16> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<1> : tensor<4x2x1x1xi8>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f16}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf16>
    return %0 : tensor<4x2x1x1xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xui8> = dense<1> : tensor<4x2x1x1xi8>, [#const.CastElemType<!qElemType>, #const.CastElemType<ui8>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<4x1x1x1xf16>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<4x1x1x1xui8>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x2x1x1xui8>, tensor<4x1x1x1xf16>, tensor<4x1x1x1xui8> -> tensor<4x2x1x1xf16>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf16>
}

// -----
!qElemType = !quant.uniform<f8E4M3FN:f32, 5.000000e-01>

// CHECK-LABEL: @PerTensorF8E4M3FNZeroPointF32Scale
func.func @PerTensorF8E4M3FNZeroPointF32Scale() -> tensor<4x2x1x1xf32> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<0.5> : tensor<4x2x1x1xf8E4M3FN>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f32}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf32>
    return %0 : tensor<4x2x1x1xf32>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xf8E4M3FN> = dense<5.000000e-01> : tensor<4x2x1x1xf8E4M3FN>, [#const.CastElemType<!qElemType>, #const.CastElemType<f8E4M3FN>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<1x1x1x1xf8E4M3FN> = dense<0.000000e+00>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f32}
    // CHECK-SAME:      tensor<4x2x1x1xf8E4M3FN>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf8E4M3FN> -> tensor<4x2x1x1xf32>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf32>
}

// -----
!qElemType = !quant.uniform<f8E5M2:f32, 5.000000e-01>

// CHECK-LABEL: @PerTensorF8E5M2ZeroPointF32Scale
func.func @PerTensorF8E5M2ZeroPointF32Scale() -> tensor<4x2x1x1xf32> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<0.5> : tensor<4x2x1x1xf8E5M2>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f32}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf32>
    return %0 : tensor<4x2x1x1xf32>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xf8E5M2> = dense<5.000000e-01> : tensor<4x2x1x1xf8E5M2>, [#const.CastElemType<!qElemType>, #const.CastElemType<f8E5M2>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<1x1x1x1xf8E5M2> = dense<0.000000e+00>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f32}
    // CHECK-SAME:      tensor<4x2x1x1xf8E5M2>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf8E5M2> -> tensor<4x2x1x1xf32>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf32>
}

// -----
!qElemType = !quant.uniform<f4E2M1FN:f16, 5.000000e-01>

// CHECK-LABEL: @PerTensorF4E2M1FNZeroPointF16Scale
func.func @PerTensorF4E2M1FNZeroPointF16Scale() -> tensor<4x2x1x1xf16> {
    %weights = const.Declare tensor<4x2x1x1x!qElemType> = dense<0.5> : tensor<4x2x1x1xf4E2M1FN>, [#const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%weights) {dstElemType = f16}
            : tensor<4x2x1x1x!qElemType> -> tensor<4x2x1x1xf16>
    return %0 : tensor<4x2x1x1xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x2x1x1xf4E2M1FN> = dense<5.000000e-01> : tensor<4x2x1x1xf4E2M1FN>, [#const.CastElemType<!qElemType>, #const.CastElemType<f4E2M1FN>]
    // CHECK-DAG: [[SCALE:%.+]]   = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e-01>
    // CHECK-DAG: [[ZP:%.+]]      = const.Declare tensor<1x1x1x1xf4E2M1FN> = dense<0.000000e+00>

    // CHECK-NOT:   IE.Dequantize
    // CHECK:       [[OUT:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f16}
    // CHECK-SAME:      tensor<4x2x1x1xf4E2M1FN>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf4E2M1FN> -> tensor<4x2x1x1xf16>
    // CHECK:       return [[OUT]] : tensor<4x2x1x1xf16>
}
