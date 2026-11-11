//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% revision-id=3 compilation-mode=DefaultHW allow-custom-values=true enable-se-ptrs-operations=true" --fuse-quantized-ops %s | FileCheck %s
// REQUIRES: platform-NPU6010
// Baseline: REVISION_B via revision-id=3.

!qElemType = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<u8<1:255>:f16:0, {0.010680671751968504:128,0.0081200787401574797:128,0.010596087598425197:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u16:f16, 5.000000e-02:3215>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u16:f16, 5.000000e-02:3215>

// CHECK: @FuseQuantParamsIntoTransposedConvOutQU16Asymmetric
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf16>
// CHECK-SAME: -> tensor<1x3x32x32xf16>
func.func @FuseQuantParamsIntoTransposedConvOutQU16Asymmetric(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x32x32xf16> {
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
    // CHECK: [[TCONV:%.+]] = IE.TransposedConvolution([[QUANT]], [[CST]]) {
    // CHECK-SAME:   dilations = [1, 1],
    // CHECK-SAME:   operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:   pads_begin = [1, 1],
    // CHECK-SAME:   pads_end = [1, 1],
    // CHECK-SAME:   spatial_output_padding = [0, 0],
    // CHECK-SAME:   strides = [2, 2]
    // CHECK-SAME:   } : tensor<1x3x16x16x[[QTYPE1]]>, tensor<3x3x4x4x[[QTYPE]]>
    // CHECK-SAME:    -> tensor<1x3x32x32x[[QTYPE2]]>
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[TCONV]]) {dstElemType = f16}
    // CHECK: return [[DEQUANT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 5.000000e-01:128>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01:128>
!qElemType2 = !quant.uniform<u16:f16, 5.000000e-01:2888>

// CHECK-LABEL: @FuseIntoConvDqU8InAsymmetricQU16Out
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf16>
func.func @FuseIntoConvDqU8InAsymmetricQU16Out(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
    %weights = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.3127431869506836 : f64>, strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x3x14x14xf16> -> tensor<1x3x14x14x!qElemType2>
    %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x3x14x14x!qElemType2> -> tensor<1x3x14x14xf16>

    return %6 : tensor<1x3x14x14xf16>

    // CHECK:   [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType}
    // CHECK-NOT: IE.Dequantize([[QUANT]])
    // CHECK:   [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> = dense<1.000000e+00> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    // CHECK-NOT: IE.Dequantize([[CST]])
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.3127431869506836 : f64>, strides = [1, 1]} : tensor<1x3x16x16x!qElemType>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType1>
    // CHECK:   [[OUT_DQ:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16}

    // CHECK:   return [[OUT_DQ]] : tensor<1x3x14x14xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 5.000000e-01:4325>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01:128>
!qElemType2 = !quant.uniform<u16:f16, 5.000000e-02:3215>

module @FuseParamsIntoConvU8InU8FilterU16Out {
    // CHECK-LABEL: func.func @main
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

    // CHECK:   [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[IN_Q:!.+]]}
    // CHECK:   [[CST:%.+]] = const.Declare tensor<3x3x3x3x[[W_Q:!.+]]> = dense<1.000000e+00> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[W_Q]]>]
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.3127431869506836 : f64>, strides = [1, 1]} : tensor<1x3x16x16x[[IN_Q]]>, tensor<3x3x3x3x[[W_Q]]> -> tensor<1x3x14x14x[[OUT_Q:!.+]]>
    // CHECK:   [[DEQUANT:%.+]] = IE.Dequantize([[CONV]]) {dstElemType = f16}
    // CHECK:   return [[DEQUANT]] : tensor<1x3x14x14xf16>

    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858:23>
!qElemType2 = !quant.uniform<u16:f16, 5.000000e-02:3215>

// CHECK-LABEL: @FuseQuantParamsIntoAvgPoolInQU8OutAsymmetricQU16
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x16xf16>
func.func @FuseQuantParamsIntoAvgPoolInQU8OutAsymmetricQU16(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x15x14xf16> {
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %3 = IE.AvgPool(%2) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16xf16> -> tensor<1x3x15x14xf16>
    %4 = IE.Quantize(%3) {dstElemType = !qElemType2} : tensor<1x3x15x14xf16> -> tensor<1x3x15x14x!qElemType2>
    %5 = IE.Dequantize(%4) {dstElemType = f16} : tensor<1x3x15x14x!qElemType2> -> tensor<1x3x15x14xf16>
    return %5 : tensor<1x3x15x14xf16>

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[AVGPOOL:%.+]] = IE.AvgPool([[QUANT]]) {kernel_size = [2, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x15x14x!qElemType1>
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[AVGPOOL]]) {dstElemType = f16} : tensor<1x3x15x14x!qElemType1> -> tensor<1x3x15x14xf16>
    // CHECK: return [[DEQUANT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
!qElemType1 = !quant.uniform<u16:f16, 3.1368405211205576E-4:5342>

// CHECK-LABEL: @FuseQuantParamsIntoGroupConvOutAsymmetricQU16
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x10x10xf16>
func.func @FuseQuantParamsIntoGroupConvOutAsymmetricQU16(%arg0: tensor<1x3x10x10xf16>) -> tensor<1x3x10x10xf16> {
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

// -----

!qElemType = !quant.uniform<u8:f16, 0.051102941176470587:121>
!qElemType1 = !quant.uniform<u16:f16, 0.034064797794117647:55>

// CHECK-LABEL: @LeakyReluQuantOutQU16Asymmetric
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x128x512x!qElemType>)
func.func @LeakyReluQuantOutQU16Asymmetric(%arg0: tensor<1x1x128x512x!qElemType>) -> tensor<1x1x128x512x!qElemType1> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512xf16>
    %1 = IE.LeakyRelu(%0) {negative_slope = 0.300048828125 : f64} : tensor<1x1x128x512xf16> -> tensor<1x1x128x512xf16>
    %2 = IE.Quantize(%1) {dstElemType = !qElemType1 } : tensor<1x1x128x512xf16> -> tensor<1x1x128x512x!qElemType1>

    return %2 : tensor<1x1x128x512x!qElemType1>

    //CHECK: [[LEAKYRELU:%.+]] = IE.LeakyRelu([[INPUT]]) {negative_slope = 0.300048828125 : f64} : tensor<1x1x128x512x!qElemType> -> tensor<1x1x128x512x!qElemType1>
    //CHECK: return [[LEAKYRELU]] : tensor<1x1x128x512x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u16:f16, 5.000000e-02:3000>

// CHECK-LABEL: @FuseQuantParamsIntoInterpOutQU16Asymmetric
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x10x10xf16>
func.func @FuseQuantParamsIntoInterpOutQU16Asymmetric(%arg0: tensor<1x16x10x10xf16>) -> tensor<1x16x20x20xf16> {
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

// -----

!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
!qElemType1 = !quant.uniform<u16:f16, 5.000000e-02:1>
// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
// CHECK:  !qElemType1 = !quant.uniform<u16:f16, 5.000000e-02:1>

// CHECK-LABEL: FuseQuantParamsIntoReduceOpOutQU16Asymmetric
module @FuseQuantParamsIntoReduceOpOutQU16Asymmetric {

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
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<u8:f16, 0.57450980392156858>
!qElemType1 = !quant.uniform<u8:f16, 1.0534313725490195:128>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<u8:f16, 1.0534313725490194:128>
!qElemType2 = !quant.uniform<u16:f16, 3.1368405211205576E-7>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u16:f16, 3.1368405211205576E-7>

// CHECK: @FuseQuantParamsIntoMatMulU8InU16OutQuantStorageType
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x8x16x16xf16>
// CHECK-SAME: -> tensor<1x8x16x32xf16>
func.func @FuseQuantParamsIntoMatMulU8InU16OutQuantStorageType(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x8x16x32xf16> {
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x8x16x16xf16> -> tensor<1x8x16x16x!qElemType1>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x8x16x16x!qElemType1> -> tensor<1x8x16x16xf16>
    %weights = const.Declare tensor<1x8x16x32x!qElemType> = dense<1.0> : tensor<1x8x16x32xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<1x8x16x32x!qElemType> -> tensor<1x8x16x32xf16>
    %4 = IE.MatMul(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8x16x16xf16>, tensor<1x8x16x32xf16> -> tensor<1x8x16x32xf16>
    %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x8x16x32xf16> -> tensor<1x8x16x32x!qElemType2>
    %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x8x16x32x!qElemType2> -> tensor<1x8x16x32xf16>

    return %6 : tensor<1x8x16x32xf16>

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QTYPE1]]}
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x8x16x32x[[QTYPE]]> = dense<1.000000e+00> :
    // CHECK-SAME: tensor<1x8x16x32xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[QTYPE]]>]
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[QUANT]], [[CST]])
    // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[MATMUL]]) {dstElemType = f16}
    // CHECK: return [[DEQUANT1]]
}

// -----

!qElemType = !quant.uniform<u16:f16, 0.57450980392156858>
// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<u16:f16, 0.57450980392156858>
!qElemType1 = !quant.uniform<u8:f16, 1.0534313725490195:128>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<u8:f16, 1.0534313725490194:128>
!qElemType2 = !quant.uniform<u16:f16, 3.1368405211205576E-7>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u16:f16, 3.1368405211205576E-7>

// CHECK: @DoNotFuseQuantParamsIntoMatMulUnequalQuantStorageType
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x8x16x16xf16>
// CHECK-SAME: -> tensor<1x8x16x32xf16>
func.func @DoNotFuseQuantParamsIntoMatMulUnequalQuantStorageType(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x8x16x32xf16> {
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x8x16x16xf16> -> tensor<1x8x16x16x!qElemType1>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x8x16x16x!qElemType1> -> tensor<1x8x16x16xf16>
    %weights = const.Declare tensor<1x8x16x32x!qElemType> = dense<1.0> : tensor<1x8x16x32xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<1x8x16x32x!qElemType> -> tensor<1x8x16x32xf16>
    %4 = IE.MatMul(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8x16x16xf16>, tensor<1x8x16x32xf16> -> tensor<1x8x16x32xf16>
    %5 = IE.Quantize(%4) {dstElemType = !qElemType2}: tensor<1x8x16x32xf16> -> tensor<1x8x16x32x!qElemType2>
    %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x8x16x32x!qElemType2> -> tensor<1x8x16x32xf16>

    return %6 : tensor<1x8x16x32xf16>

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QTYPE1]]}
    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16}
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x8x16x32x[[QTYPE]]> = dense<1.000000e+00> :
    // CHECK-SAME: tensor<1x8x16x32xf16>, [#const.CastElemType<ui8>, #const.CastElemType<[[QTYPE]]>]
    // CHECK: [[DEQUANT_CST:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16}
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[DEQUANT]], [[DEQUANT_CST]])
    // CHECK: [[QUANT1:%.+]] = IE.Quantize([[MATMUL]]) {dstElemType = [[QTYPE2]]}
    // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16}
    // CHECK: return [[DEQUANT1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0034409466911764705>
!qElemType1 = !quant.uniform<u16:f16, 0.12503063725490196>

// CHECK-LABEL:    func.func @DoNotFuseQuantParamsIntoEltwiseSubtractUnequalQuantStorageType
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x3x16x16xf16>, [[INPUT1:%.+]]: tensor<1x3x16x16xf16>)
func.func @DoNotFuseQuantParamsIntoEltwiseSubtractUnequalQuantStorageType(%arg0: tensor<1x3x16x16xf16>, %arg1: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %3 = IE.Quantize(%arg1) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
    %5 = IE.Subtract(%2, %4) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    %6 = IE.Quantize(%5) {dstElemType = !qElemType}: tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %7 = IE.Dequantize(%6) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>

    return %7 : tensor<1x3x16x16xf16>

    // CHECK: [[QUANT0:%.+]] = IE.Quantize([[INPUT0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK: [[QUANT1:%.+]] = IE.Quantize([[INPUT1]]) {dstElemType = !qElemType1} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType1>
    // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType1> -> tensor<1x3x16x16xf16>
    // CHECK: [[SUB:%.+]] = IE.Subtract([[DEQUANT0]], [[DEQUANT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    // CHECK: [[QUANT2:%.+]] = IE.Quantize([[SUB]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[DEQUANT2:%.+]] = IE.Dequantize([[QUANT2]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK: return [[DEQUANT2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195>
!qElemType1 = !quant.uniform<u16:f16, 5.000000e-02>

// CHECK-LABEL: @DoNotFuseQuantParamsIntoSliceUnequalQuantStorageType
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x3x16x16xf16>)
func.func @DoNotFuseQuantParamsIntoSliceUnequalQuantStorageType(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x8xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    %2 = IE.Slice %1 [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16xf16> to tensor<1x3x16x8xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType1}: tensor<1x3x16x8xf16> -> tensor<1x3x16x8x!qElemType1>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x3x16x8x!qElemType1> -> tensor<1x3x16x8xf16>

    return %4 : tensor<1x3x16x8xf16>

    // CHECK: [[QUANT0:%.+]] = IE.Quantize([[INPUT0]]) {dstElemType = !qElemType} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType>
    // CHECK: [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType> -> tensor<1x3x16x16xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[DEQUANT0]] [0, 0, 0, 8] [1, 3, 16, 8] : tensor<1x3x16x16xf16> to tensor<1x3x16x8xf16>
    // CHECK: [[QUANT1:%.+]] = IE.Quantize([[SLICE]]) {dstElemType = !qElemType1} : tensor<1x3x16x8xf16> -> tensor<1x3x16x8x!qElemType1>
    // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x3x16x8x!qElemType1> -> tensor<1x3x16x8xf16>
    // CHECK: return [[DEQUANT1]] : tensor<1x3x16x8xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195>
!qElemType1 = !quant.uniform<u16:f16, 5.000000e-02>

// CHECK-LABEL: @DoNotFuseQuantParamsIntoTileUnequalQuantStorageType
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x64x1x1xf16>)
func.func @DoNotFuseQuantParamsIntoTileUnequalQuantStorageType(%arg0: tensor<1x64x1x1xf16>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x64x1x1x!qElemType> -> tensor<1x64x1x1xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16> -> tensor<1x64x11x11xf16>
    %3 = IE.Quantize(%2) {dstElemType = !qElemType1} : tensor<1x64x11x11xf16> -> tensor<1x64x11x11x!qElemType1>
    %4 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x64x11x11x!qElemType1> -> tensor<1x64x11x11xf16>

    return %4 : tensor<1x64x11x11xf16>

    // CHECK: [[QUANT0:%.+]] = IE.Quantize([[INPUT0]])  {dstElemType = !qElemType} : tensor<1x64x1x1xf16> -> tensor<1x64x1x1x!qElemType>
    // CHECK: [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16} : tensor<1x64x1x1x!qElemType> -> tensor<1x64x1x1xf16>
    // CHECK: [[TILE:%.+]] = IE.Tile([[DEQUANT0]]) {repeats_values = [1, 1, 11, 11]} : tensor<1x64x1x1xf16> -> tensor<1x64x11x11xf16>
    // CHECK: [[QUANT1:%.+]] = IE.Quantize([[TILE]]) {dstElemType = !qElemType1} : tensor<1x64x11x11xf16> -> tensor<1x64x11x11x!qElemType1>
    // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x64x11x11x!qElemType1> -> tensor<1x64x11x11xf16>

    // CHECK: return [[DEQUANT1]] : tensor<1x64x11x11xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u16:f16, 5.000000e-02>

// CHECK-LABEL: @FuseQuantParamsIntoConcatUnequalQuantStorageType
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x2x3x4xf16>, [[INPUT1:%.+]]: tensor<1x2x3x4xf16>)
func.func @FuseQuantParamsIntoConcatUnequalQuantStorageType(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x2x3x4xf16>) -> tensor<1x4x3x4xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %2 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>

    %4 = IE.Concat (%1, %3) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>

    %5 = IE.Quantize(%4) {dstElemType = !qElemType1}: tensor<1x4x3x4xf16> -> tensor<1x4x3x4x!qElemType1>
    %6 = IE.Dequantize(%5) {dstElemType = f16} : tensor<1x4x3x4x!qElemType1> -> tensor<1x4x3x4xf16>

    return %6 : tensor<1x4x3x4xf16>

    // CHECK: [[QUANT0:%.+]] = IE.Quantize([[INPUT0]]) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    // CHECK: [[DEQUANT0:%.+]] = IE.Dequantize([[QUANT0]]) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    // CHECK: [[QUANT1:%.+]] = IE.Quantize([[INPUT1]]) {dstElemType = !qElemType} : tensor<1x2x3x4xf16> -> tensor<1x2x3x4x!qElemType>
    // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[QUANT1]]) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[DEQUANT0]], [[DEQUANT1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    // CHECK: [[QUANT2:%.+]] = IE.Quantize([[CONCAT]]) {dstElemType = !qElemType1} : tensor<1x4x3x4xf16> -> tensor<1x4x3x4x!qElemType1>
    // CHECK: [[DEQUANT2:%.+]] = IE.Dequantize([[QUANT2]]) {dstElemType = f16} : tensor<1x4x3x4x!qElemType1> -> tensor<1x4x3x4xf16>
    // CHECK: return [[DEQUANT2]] : tensor<1x4x3x4xf16>
}
