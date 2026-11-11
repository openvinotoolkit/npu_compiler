//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true enable-se-ptrs-operations=true" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType = !quant.uniform<u8:f16, 5.000000e-01:128>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01:128>
!qElemType2 = !quant.uniform<u16:f16, 5.000000e-01:2888>

// CHECK-LABEL: @DoNotFuseIntoConvDqU8InAsymmetricQU16Out
module @DoNotFuseIntoConvDqU8InAsymmetricQU16Out attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf16>
    func.func @main(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
        %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
        %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
        %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.3127431869506836 : f64>, strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
        %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
        %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

        return %6 : tensor<1x3x14x14xf16>

        // CHECK:   [[QUANT0:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType}
        // CHECK:   [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16}
        // CHECK:   [[CST:%.+]] = const.Declare
        // CHECK-SAME:   tensor<3x3x3x3x!qElemType> = dense<1.000000e+00> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
        // CHECK:   [[DEQUANT1:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16}
        // CHECK:   [[CONV:%.+]] = IE.Convolution([[DEQUANT0]], [[DEQUANT1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.3127431869506836 : f64>, strides = [1, 1]}
        // CHECK:   [[QUANT1:%.+]]  = IE.Quantize([[CONV]]) {dstElemType = !qElemType1}
        // CHECK:   [[DEQUANT2:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16}
        // CHECK:   return [[DEQUANT2]]
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 5.000000e-01:128>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:128>
!qElemType1 = !quant.uniform<u16:f16, 5.000000e-01:4325>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<u16:f16, 5.000000e-01:4325>
!qElemType2 = !quant.uniform<u16:f16, 5.000000e-02:3215>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u16:f16, 5.000000e-02:3215>

// CHECK: @FuseParamsIntoConvDqU16InAsymmetricQU16Out
module @FuseParamsIntoConvDqU16InAsymmetricQU16Out attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf16>
    // CHECK-SAME: -> tensor<1x3x14x14xf16>
    func.func @main(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
        %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
        %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
        %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.3127431869506836 : f64>, strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
        %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
        %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

        return %6 : tensor<1x3x14x14xf16>

        // CHECK:   [[QUANT0:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QTYPE1]]}
        // CHECK:   [[CST:%.+]] = const.Declare
        // CHECK-SAME:   tensor<3x3x3x3x[[QTYPE]]> = dense<1.000000e+00> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[QTYPE]]>]
        // CHECK:   [[CONV:%.+]] = IE.Convolution([[QUANT0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.3127431869506836 : f64>, strides = [1, 1]} : tensor<1x3x16x16x[[QTYPE1]]>, tensor<3x3x3x3x[[QTYPE]]> -> tensor<1x3x14x14x[[QTYPE2]]>
        // CHECK:   [[DEQUANT0:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16}
        // CHECK:   return [[DEQUANT0]]
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 5.000000e-01:128>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01:128>
!qElemType2 = !quant.uniform<u16:f16, 5.000000e-01>
// CHECK-LABEL: @FuseParamsIntoConvDqU8InSymmetricQU16Out
module @FuseParamsIntoConvDqU8InSymmetricQU16Out attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf16>
    func.func @main(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
        %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
        %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
        %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.3127431869506836 : f64>, strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
        %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
        %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>
        return %6 : tensor<1x3x14x14xf16>

        // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType}
        // CHECK: [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.000000e+00> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
        // CHECK: [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.3127431869506836 : f64>, strides = [1, 1]} : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType1>
        // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16}
        // CHECK: return [[DEQUANT]]
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.574509803921568:23>
!qElemType1 = !quant.uniform<u16:f16, 4.000000e-02>
// CHECK-LABEL: @FuseQuantParamsIntoAvgPoolInQU8OutSymmetricQU16
module @FuseQuantParamsIntoAvgPoolInQU8OutSymmetricQU16 attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf16>
    func.func @main(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x15x14xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
        %3 = IE.AvgPool(%2) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x15x14xf16>
        %4 = IE.Quantize(%3) {dstElemType = !qElemType1} : tensor<1x3x15x14xf16> -> tensor<1x3x15x14x!qElemType1>
        %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x15x14x!qElemType1> -> tensor<1x3x15x14xf16>
        return %5 : tensor<1x3x15x14xf16>

        // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
        // CHECK: [[AVGPOOL:%.+]] = IE.AvgPool([[QUANT]]) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x15x14x!qElemType1>
        // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[AVGPOOL]]) {dstElemType = f16} : tensor<1x3x15x14x!qElemType1> -> tensor<1x3x15x14xf16>
        // CHECK: return [[DEQUANT]]
    }
}
// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858:23>
!qElemType2 = !quant.uniform<u16:f16, 5.000000e-02:3215>
// CHECK-LABEL: @DoNotFuseQuantParamsIntoAvgPoolInQU8OutAsymmetricQU16
module @DoNotFuseQuantParamsIntoAvgPoolInQU8OutAsymmetricQU16 attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf16>
    func.func @main(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x15x14xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
        %3 = IE.AvgPool(%2) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x15x14xf16>
        %4 = IE.Quantize(%3) {dstElemType = !qElemType2} : tensor<1x3x15x14xf16> -> tensor<1x3x15x14x!qElemType2>
        %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x15x14x!qElemType2> -> tensor<1x3x15x14xf16>
        return %5 : tensor<1x3x15x14xf16>

        // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
        // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
        // CHECK: [[AVGPOOL:%.+]] = IE.AvgPool([[DEQUANT]]) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x15x14xf16>
        // CHECK: [[QUANT1:%.+]] = IE.Quantize([[AVGPOOL]]) {dstElemType = !qElemType1} : tensor<1x3x15x14xf16> -> tensor<1x3x15x14x!qElemType1>
        // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x3x15x14x!qElemType1> -> tensor<1x3x15x14xf16>
        // CHECK: return [[DEQUANT1]] : tensor<1x3x15x14xf16>
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
!qElemType1 = !quant.uniform<u16:f16, 3.1368405211205576E-4>
// CHECK-LABEL: @FuseQuantParamsIntoGroupConvOutSymmetricQU16
module @FuseQuantParamsIntoGroupConvOutSymmetricQU16 attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x10x10xf16>
    func.func @main(%arg0: tensor<1x3x10x10xf16>) -> tensor<1x3x10x10xf16> {
        %cst = const.Declare tensor<3x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

        %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
        %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x10x10x!qElemType> -> tensor<1x3x10x10xf16>
        %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>

        %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x10x10xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x10x10xf16>

        %4 = IE.Quantize(%3) {dstElemType = !qElemType1} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType1>
        %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x10x10x!qElemType1> -> tensor<1x3x10x10xf16>

        return %5 : tensor<1x3x10x10xf16>

        // CHECK: [[CST:%.+]] = const.Declare tensor<3x1x3x3x!qElemType>
        // CHECK-SAME:     = dense<2.000000e+00> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
        // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
        // CHECK: [[GCONV:%.+]] = IE.GroupConvolution([[QUANT]], [[CST]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x10x10x!qElemType>, tensor<3x1x3x3x!qElemType> -> tensor<1x3x10x10x!qElemType1>
        // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[GCONV]]) {dstElemType = f16} : tensor<1x3x10x10x!qElemType1> -> tensor<1x3x10x10xf16>

        // CHECK: return [[DEQUANT]] : tensor<1x3x10x10xf16>
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
!qElemType1 = !quant.uniform<u16:f16, 3.1368405211205576E-4:5342>
// CHECK-LABEL: @DoNotFuseQuantParamsIntoGroupConvOutAsymmetricQU16
module @DoNotFuseQuantParamsIntoGroupConvOutAsymmetricQU16 attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x10x10xf16>
    func.func @main(%arg0: tensor<1x3x10x10xf16>) -> tensor<1x3x10x10xf16> {
        %cst = const.Declare tensor<3x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

        %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
        %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x10x10x!qElemType> -> tensor<1x3x10x10xf16>
        %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>

        %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x10x10xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x10x10xf16>

        %4 = IE.Quantize(%3) {dstElemType = !qElemType1} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType1>
        %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x10x10x!qElemType1> -> tensor<1x3x10x10xf16>

        return %5 : tensor<1x3x10x10xf16>

        // CHECK: [[CST:%.+]] = const.Declare tensor<3x1x3x3x!qElemType>
        // CHECK-SAME:     = dense<2.000000e+00> : tensor<3x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
        // CHECK: [[QUANT0:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType>
        // CHECK: [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16} : tensor<1x3x10x10x!qElemType> -> tensor<1x3x10x10xf16>
        // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x1x3x3x!qElemType> -> tensor<3x1x3x3xf16>
        // CHECK: [[GCONV:%.+]] = IE.GroupConvolution([[DEQUANT0]], [[DEQUANT1]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x10x10xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x10x10xf16>
        // CHECK: [[QUANT1:%.+]] = IE.Quantize([[GCONV]]) {dstElemType = !qElemType1} : tensor<1x3x10x10xf16> -> tensor<1x3x10x10x!qElemType1>
        // CHECK: [[DEQUANT2:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x3x10x10x!qElemType1> -> tensor<1x3x10x10xf16>
        // CHECK: return [[DEQUANT2]] : tensor<1x3x10x10xf16>
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.051102941176470587:121>
!qElemType1 = !quant.uniform<u16:f16, 0.034064797794117647>
// CHECK-LABEL: @FuseQuantParamLeakyReluQuantOutQU16Symmetric
module @FuseQuantParamLeakyReluQuantOutQU16Symmetric attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x128x512x!qElemType>
    func.func @main(%arg0: tensor<1x1x128x512x!qElemType>) -> tensor<1x1x128x512x!qElemType1> {
        %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512xf16>
        %1 = IE.LeakyRelu(%0) {negative_slope = 0.300048828125 : f64} : tensor<1x1x128x512xf16> -> tensor<1x1x128x512xf16>
        %2 = IE.Quantize(%1) {dstElemType = !qElemType1 } : tensor<1x1x128x512xf16> -> tensor<1x1x128x512x!qElemType1>

        return %2 : tensor<1x1x128x512x!qElemType1>

        // CHECK: [[LEAKYRELU:%.+]] = IE.LeakyRelu([[INPUT]]) {negative_slope = 0.300048828125 : f64} : tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512x!qElemType1>
        // CHECK: return [[LEAKYRELU]] : tensor<1x1x128x512x!qElemType1>
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.051102941176470587:121>
!qElemType1 = !quant.uniform<u16:f16, 0.034064797794117647:55>
// CHECK-LABEL: @DoNotFuseQuantParamLeakyReluQuantOutQU16Asymmetric
module @DoNotFuseQuantParamLeakyReluQuantOutQU16Asymmetric attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // This LeakyRelu will end up on DPU so it is needed to prevent fusing of asymmetric U16 quantization from output
    // CHECK-LABEL: @main
    // CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x128x512x!qElemType>
    func.func @main(%arg0: tensor<1x1x128x512x!qElemType>) -> tensor<1x1x128x512x!qElemType1> {
        %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512xf16>
        %1 = IE.LeakyRelu(%0) {negative_slope = 0.300048828125 : f64} : tensor<1x1x128x512xf16> -> tensor<1x1x128x512xf16>
        %2 = IE.Quantize(%1) {dstElemType = !qElemType1 } : tensor<1x1x128x512xf16> -> tensor<1x1x128x512x!qElemType1>

        return %2 : tensor<1x1x128x512x!qElemType1>

        // CHECK:   [[DEQUANT:%.+]] = IE.Dequantize([[INPUT]])
        // CHECK-SAME:  tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512xf16>
        // CHECK:   [[LEAKYRELU:%.+]] = IE.LeakyRelu([[DEQUANT]])
        // CHECK-SAME:  {negative_slope = 0.300048828125 : f64} : tensor<1x1x128x512xf16> -> tensor<1x1x128x512xf16>
        // CHECK:   [[QUANT:%.+]] = IE.Quantize([[LEAKYRELU]])
        // CHECK-SAME:  tensor<1x1x128x512xf16> -> tensor<1x1x128x512x!qElemType1>

        // CHECK: return [[QUANT]] : tensor<1x1x128x512x!qElemType1>
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u16:f16, 5.000000e-02>

// CHECK:  !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK:  !qElemType1 = !quant.uniform<u16:f16, 5.000000e-02>
// CHECK-LABEL: @FuseQuantParamsIntoInterpOutQU16Symmetric
module @FuseQuantParamsIntoInterpOutQU16Symmetric attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x10x10xf16>
    func.func @main(%arg0: tensor<1x16x10x10xf16>) -> tensor<1x16x20x20xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x10x10xf16> -> tensor<1x16x10x10x!qElemType>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x10x10x!qElemType> -> tensor<1x16x10x10xf16>
        %4 = IE.Interpolate(%2) {
                    attr = #IE.Interpolate<mode = <NEAREST>,
                                        shape_calc_mode = <SCALES>,
                                        coord_mode = <ASYMMETRIC>,
                                        nearest_mode = <FLOOR>,
                                        antialias = false,
                                        pads_begin = [0, 0, 0, 0],
                                        pads_end = [0, 0, 0, 0],
                                        cube_coeff = -7.500000e-01 : f64>,
                                        axes_attr = [2, 3],
                                        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                        scales_attr = [2.000000e+00, 2.000000e+00],
                                        sizes_attr = [20, 20]} :
                tensor<1x16x10x10xf16> -> tensor<1x16x20x20xf16>
        %5 = IE.Quantize(%4) {dstElemType = !qElemType1}: tensor<1x16x20x20xf16> -> tensor<1x16x20x20x!qElemType1>
        %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x16x20x20x!qElemType1> -> tensor<1x16x20x20xf16>

        return %6 : tensor<1x16x20x20xf16>

        // CHECK:      [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType}
        // CHECK-SAME:   tensor<1x16x10x10xf16> -> tensor<1x16x10x10x!qElemType>

        // CHECK:      [[INTERP:%.+]] = IE.Interpolate([[QUANT]])
        // CHECK-SAME:   tensor<1x16x10x10x!qElemType> -> tensor<1x16x20x20x!qElemType1>

        // CHECK:      [[DEQUANT:%.+]] = IE.Dequantize([[INTERP]]) {dstElemType = f16}
        // CHECK-SAME:   tensor<1x16x20x20x!qElemType1> -> tensor<1x16x20x20xf16>

        // CHECK:      return [[DEQUANT]] : tensor<1x16x20x20xf16>
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u16:f16, 5.000000e-02:5>

// CHECK:  !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK:  !qElemType1 = !quant.uniform<u16:f16, 5.000000e-02:5>
// CHECK-LABEL: @DoNotFuseQuantParamsIntoInterpOutQU16Asymmetric
module @DoNotFuseQuantParamsIntoInterpOutQU16Asymmetric attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x10x10xf16>
    func.func @main(%arg0: tensor<1x16x10x10xf16>) -> tensor<1x16x20x20xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x10x10xf16> -> tensor<1x16x10x10x!qElemType>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x10x10x!qElemType> -> tensor<1x16x10x10xf16>
        %4 = IE.Interpolate(%2) {
                    attr = #IE.Interpolate<mode = <NEAREST>,
                                        shape_calc_mode = <SCALES>,
                                        coord_mode = <ASYMMETRIC>,
                                        nearest_mode = <FLOOR>,
                                        antialias = false,
                                        pads_begin = [0, 0, 0, 0],
                                        pads_end = [0, 0, 0, 0],
                                        cube_coeff = -7.500000e-01 : f64>,
                                        axes_attr = [2, 3],
                                        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                        scales_attr = [2.000000e+00, 2.000000e+00],
                                        sizes_attr = [20, 20]} :
                tensor<1x16x10x10xf16> -> tensor<1x16x20x20xf16>
        %5 = IE.Quantize(%4) {dstElemType = !qElemType1}: tensor<1x16x20x20xf16> -> tensor<1x16x20x20x!qElemType1>
        %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x16x20x20x!qElemType1> -> tensor<1x16x20x20xf16>

        return %6 : tensor<1x16x20x20xf16>

        // CHECK:      [[QUANT:%.+]] = IE.Quantize([[ARG0]])
        // CHECK-SAME:   {dstElemType = !qElemType} : tensor<1x16x10x10xf16> -> tensor<1x16x10x10x!qElemType>

        // CHECK:      [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]])
        // CHECK-SAME:   {dstElemType = f16} : tensor<1x16x10x10x!qElemType> -> tensor<1x16x10x10xf16>

        // CHECK:      [[INTERP:%.+]] = IE.Interpolate([[DEQUANT]])
        // CHECK-SAME:   tensor<1x16x10x10xf16> -> tensor<1x16x20x20xf16>

        // CHECK:      [[QUANT1:%.+]] = IE.Quantize([[INTERP]])
        // CHECK-SAME:   {dstElemType = !qElemType1} : tensor<1x16x20x20xf16> -> tensor<1x16x20x20x!qElemType1>

        // CHECK:      [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]])
        // CHECK-SAME:   {dstElemType = f16} : tensor<1x16x20x20x!qElemType1> -> tensor<1x16x20x20xf16>

        // CHECK:      return [[DEQUANT1]] : tensor<1x16x20x20xf16>
    }
}

// -----

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u16:f16, 2.4627450980392158>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u16:f16, 2.4627450980392158>

// CHECK: @FuseQuantParamsIntoTransposedConvOutQU16Symmetric
module @FuseQuantParamsIntoTransposedConvOutQU16Symmetric attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf16>
    // CHECK-SAME: -> tensor<1x3x32x32xf16>
    func.func @main(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x32x32xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
        %weights = const.Declare tensor<3x3x4x4x!qElemType> = dense<1.0> : tensor<3x3x4x4xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
        %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x4x4x!qElemType> -> tensor<3x3x4x4xf16>
        %4 = IE.TransposedConvolution(%2, %3) {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            spatial_output_padding = [0, 0],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [2, 2]
            } : tensor<1x3x16x16xf16>, tensor<3x3x4x4xf16> -> tensor<1x3x32x32xf16>
        %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x32x32xf16> -> tensor<1x3x32x32x!qElemType2>
        %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>

        return %6 : tensor<1x3x32x32xf16>

        // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QTYPE1]]}
        // CHECK: [[CST:%.+]] = const.Declare tensor<3x3x4x4x[[QTYPE]]> = dense<1.000000e+00> : tensor<3x3x4x4xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[QTYPE]]>]
        // CHECK: [[CONV:%.+]] = IE.TransposedConvolution([[QUANT]], [[CST]]) {
        // CHECK-SAME:   dilations = [1, 1],
        // CHECK-SAME:   operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        // CHECK-SAME:   pads_begin = [1, 1],
        // CHECK-SAME:   pads_end = [1, 1],
        // CHECK-SAME:   spatial_output_padding = [0, 0],
        // CHECK-SAME:   strides = [2, 2]
        // CHECK-SAME:   } : tensor<1x3x16x16x[[QTYPE1]]>, tensor<3x3x4x4x[[QTYPE]]>
        // CHECK-SAME:    -> tensor<1x3x32x32x[[QTYPE2]]>
        // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16}
        // CHECK: return [[DEQUANT]]
    }
}

// -----

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u16:f16, 5.000000e-02:5>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u16:f16, 5.000000e-02:5>

// CHECK: @DoNotFuseQuantParamsIntoTransposedConvOutQU16Asymmetric
module @DoNotFuseQuantParamsIntoTransposedConvOutQU16Asymmetric attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf16>
    // CHECK-SAME: -> tensor<1x3x32x32xf16>
    func.func @main(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x32x32xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
        %weights = const.Declare tensor<3x3x4x4x!qElemType> = dense<1.0> : tensor<3x3x4x4xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
        %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x4x4x!qElemType> -> tensor<3x3x4x4xf16>
        %4 = IE.TransposedConvolution(%2, %3) {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            spatial_output_padding = [0, 0],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [2, 2]
            } : tensor<1x3x16x16xf16>, tensor<3x3x4x4xf16> -> tensor<1x3x32x32xf16>
        %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x32x32xf16> -> tensor<1x3x32x32x!qElemType2>
        %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x32x32x!qElemType2> -> tensor<1x3x32x32xf16>

        return %6 : tensor<1x3x32x32xf16>

        // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QTYPE1]]}
        // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16}
        // CHECK: [[CST:%.+]] = const.Declare tensor<3x3x4x4x[[QTYPE]]> = dense<1.000000e+00> : tensor<3x3x4x4xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[QTYPE]]>]
        // CHECK: [[DEQUANT_CST:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16}
        // CHECK: [[TCONV:%.+]] = IE.TransposedConvolution([[DEQUANT]], [[DEQUANT_CST]]) {
        // CHECK-SAME:   dilations = [1, 1],
        // CHECK-SAME:   operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        // CHECK-SAME:   pads_begin = [1, 1],
        // CHECK-SAME:   pads_end = [1, 1],
        // CHECK-SAME:   spatial_output_padding = [0, 0],
        // CHECK-SAME:   strides = [2, 2]
        // CHECK-SAME:   }
        // CHECK: [[QUANT1:%.+]] = IE.Quantize([[TCONV]]) {dstElemType = [[QTYPE2]]}
        // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16}
        // CHECK: return [[DEQUANT1]]
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
!qElemType1 = !quant.uniform<u16:f16, 5.000000e-02>
// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
// CHECK:  !qElemType1 = !quant.uniform<u16:f16, 5.000000e-02>

// CHECK-LABEL: FuseQuantParamsIntoReduceOpOutQU16Symmetric
module @FuseQuantParamsIntoReduceOpOutQU16Symmetric attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    config.PipelineOptions @Options {
            config.Option @config.AutoPaddingODU : true
            config.Option @config.ReduceSupported : true
    }

    // CHECK-LABEL:    func.func @FuseQuantParamsIntoReduceMean
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x30xf16>)
    func.func @FuseQuantParamsIntoReduceMean(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x1x30x30xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30x!qElemType>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x30x30x!qElemType> -> tensor<1x16x30x30xf16>
        %3 = IE.ReduceMean(%2) {axes_value = [1], keep_dims} : tensor<1x16x30x30xf16> -> tensor<1x1x30x30xf16>
        %4 = IE.Quantize(%3) {dstElemType = !qElemType1} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType1>
        %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x1x30x30x!qElemType1> -> tensor<1x1x30x30xf16>
        return %5 : tensor<1x1x30x30xf16>

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30x!qElemType>
    // CHECK: [[REDUCEMEAN:%.+]] = IE.ReduceMean([[QUANT]]) {axes_value = [1], keep_dims} : tensor<1x16x30x30x!qElemType> -> tensor<1x1x30x30x!qElemType1>
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[REDUCEMEAN]]) {dstElemType = f16} : tensor<1x1x30x30x!qElemType1> -> tensor<1x1x30x30xf16>
    // CHECK: return [[DEQUANT]]
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
!qElemType1 = !quant.uniform<u16:f16, 5.000000e-02:1>
// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
// CHECK:  !qElemType1 = !quant.uniform<u16:f16, 5.000000e-02:1>

// CHECK-LABEL: DoNotFuseQuantParamsIntoReduceOpOutQU16Asymmetric
module @DoNotFuseQuantParamsIntoReduceOpOutQU16Asymmetric attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    config.PipelineOptions @Options {
            config.Option @config.AutoPaddingODU : true
            config.Option @config.ReduceSupported : true
    }

    // CHECK-LABEL:    func.func @main
    // CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x30x30xf16>
    func.func @main(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x1x30x30xf16> {
        %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30x!qElemType>
        %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x30x30x!qElemType> -> tensor<1x16x30x30xf16>
        %3 = IE.ReduceSum(%2) {axes_value = [1], keep_dims} : tensor<1x16x30x30xf16> -> tensor<1x1x30x30xf16>
        %4 = IE.Quantize(%3) {dstElemType = !qElemType1} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType1>
        %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x1x30x30x!qElemType1> -> tensor<1x1x30x30xf16>
        return %5 : tensor<1x1x30x30xf16>

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30x!qElemType>
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x16x30x30x!qElemType> -> tensor<1x16x30x30xf16>
    // CHECK: [[REDUCE:%.+]] = IE.ReduceSum([[DEQUANT]]) {axes_value = [1], keep_dims} : tensor<1x16x30x30xf16> -> tensor<1x1x30x30xf16>
    // CHECK: [[QUANT1:%.+]] = IE.Quantize([[REDUCE]]) {dstElemType = !qElemType1} : tensor<1x1x30x30xf16> -> tensor<1x1x30x30x!qElemType1>
    // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x1x30x30x!qElemType1> -> tensor<1x1x30x30xf16>
    // CHECK: return [[DEQUANT1]]
    }
}
