//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-nce-ops-to-4d %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConvertNceOpsTo4DConvolution
func.func @ConvertNceOpsTo4DConvolution(%arg0: tensor<1x16x64xf16>) -> tensor<1x1x61xf16> {
    %FILTERS = const.Declare tensor<1x16x5xf16> = dense<1.000000e+00> : tensor<1x16x5xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [2], pads_begin = [3], pads_end = [2], strides = [1]} : tensor<1x16x64xf16>, tensor<1x16x5xf16> -> tensor<1x1x61xf16>
    return %RESULT : tensor<1x1x61xf16>

    // CHECK:       %[[VAL0:.*]] = IE.Convolution
    // CHECK-SAME:      dilations = [2, 1]
    // CHECK-SAME:      pads_begin = [3, 0]
    // CHECK-SAME:      pads_end = [2, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x64x1xf16>, tensor<1x16x5x1xf16> -> tensor<1x1x61x1xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 1, 61]} : tensor<1x1x61x1xf16> -> tensor<1x1x61xf16>
    // CHECK:       return %[[RESULT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK: [[QTYPE:!.+]] = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK: func.func @ConvertNceOpsTo4DConvolution_MixedPrecision
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x16x64x[[QTYPE]]>) -> tensor<1x1x61xf16>
func.func @ConvertNceOpsTo4DConvolution_MixedPrecision(%arg0: tensor<1x16x64x!qElemType>) -> tensor<1x1x61xf16> {
    %FILTERS = const.Declare tensor<1x16x5x!qElemType> = dense<1.000000e+00> : tensor<1x16x5xf16>,
        [#const.CastElemType<!qElemType>]
    %RESULT = IE.Convolution(%arg0, %FILTERS)
        {dilations = [2], pads_begin = [3], pads_end = [2], strides = [1]}
        : tensor<1x16x64x!qElemType>, tensor<1x16x5x!qElemType> -> tensor<1x1x61xf16>
    return %RESULT : tensor<1x1x61xf16>

    // CHECK: %[[RESHAPE:.+]] = IE.Reshape(%[[ARG0]]) {shape_value = [1, 16, 64, 1]}
    // CHECK: %[[FILTERS:.+]] = const.Declare tensor<1x16x5x1x[[QTYPE]]>

    // CHECK:       %[[VAL0:.*]] = IE.Convolution(%[[RESHAPE]], %[[FILTERS]])
    // CHECK-SAME:      dilations = [2, 1]
    // CHECK-SAME:      pads_begin = [3, 0]
    // CHECK-SAME:      pads_end = [2, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x64x1x[[QTYPE]]>, tensor<1x16x5x1x[[QTYPE]]> -> tensor<1x1x61x1xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 1, 61]} : tensor<1x1x61x1xf16> -> tensor<1x1x61xf16>
    // CHECK:       return %[[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolutionWithPostOp
func.func @ConvertNceOpsTo4DConvolutionWithPostOp(%arg0: tensor<1x16x64xf16>) -> tensor<1x1x64xf16> {
    %cts = const.Declare tensor<1x16x5xf16> = dense<1.000000e+00> : tensor<1x16x5xf16>
    %0 = IE.Convolution(%arg0, %cts) {dilations = [1], pads_begin = [2], pads_end = [2], strides = [1]} : tensor<1x16x64xf16>, tensor<1x16x5xf16> -> tensor<1x1x64xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x1x64xf16> -> tensor<1x1x64xf16>

    return %1 : tensor<1x1x64xf16>

    // CHECK:       [[VAL0:%.*]] = IE.Convolution
    // CHECK-SAME:       {dilations = [1, 1], pads_begin = [2, 0], pads_end = [2, 0],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:       tensor<1x16x64x1xf16>, tensor<1x16x5x1xf16> -> tensor<1x1x64x1xf16>

    // CHECK:       [[VAL1:%.*]] = IE.Reshape([[VAL0]]) {shape_value = [1, 1, 64]} : tensor<1x1x64x1xf16> -> tensor<1x1x64xf16>

    // CHECK:       [[VAL2:%.*]] = IE.ReLU([[VAL1]]) : tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    // CHECK:       return [[VAL2]] : tensor<1x1x64xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DGroupConvolution
func.func @ConvertNceOpsTo4DGroupConvolution(%arg0: tensor<1x16x30xf16>) -> tensor<1x8x28xf16>{
    %FILTERS = const.Declare tensor<8x8x3xf16> = dense<1.000000e+00> : tensor<2x4x8x3xf32>, [#const.Reshape<[8, 8, 3]>, #const.CastElemType<f16>]
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1], groups = 2, pads_begin = [0], pads_end = [0], strides = [1]} : tensor<1x16x30xf16>, tensor<8x8x3xf16> -> tensor<1x8x28xf16>
    return %RESULT : tensor<1x8x28xf16>

    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x30x1xf16>, tensor<8x8x3x1xf16> -> tensor<1x8x28x1xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 8, 28]} : tensor<1x8x28x1xf16> -> tensor<1x8x28xf16>
    // CHECK:       return %[[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DTransposedConvolution
func.func @ConvertNceOpsTo4DTransposedConvolution(%arg0: tensor<1x384x1344xf16>) -> tensor<1x192x5380xf16> {
    %FILTERS = const.Declare tensor<192x384x8xf16> = dense<1.000000e+00> : tensor<192x384x8xf16>
    %RESULT = IE.TransposedConvolution(%arg0, %FILTERS) {
        dilations = [1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0], pads_begin = [0], pads_end = [0], strides = [4]} : tensor<1x384x1344xf16>, tensor<192x384x8xf16> -> tensor<1x192x5380xf16>

    return %RESULT : tensor<1x192x5380xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 384, 1344, 1]} : tensor<1x384x1344xf16> -> tensor<1x384x1344x1xf16>
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<192x384x8x1xf16> = dense<1.000000e+00> : tensor<192x384x8xf16>, [#const.Reshape<[192, 384, 8, 1]>]

    // CHECK:       [[TRANSPOSED_CONV:%.*]] = IE.TransposedConvolution([[RESHAPE_INPUT]], [[CST_0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-SAME:      output_padding = [0, 0]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [4, 1]
    // CHECK-SAME:      tensor<1x384x1344x1xf16>, tensor<192x384x8x1xf16> -> tensor<1x192x5380x1xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[TRANSPOSED_CONV]]) {shape_value = [1, 192, 5380]} : tensor<1x192x5380x1xf16> -> tensor<1x192x5380xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x192x5380xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DMaxpool
func.func @ConvertNceOpsTo4DMaxpool(%arg0: tensor<1x512x16xf16>) -> tensor<1x512x1xf16> {
    %RESULT = IE.MaxPool(%arg0) {
        kernel_size = [16], pads_begin = [0], pads_end = [0], rounding_type = #IE.rounding_type<FLOOR>, strides = [16]} : tensor<1x512x16xf16> -> tensor<1x512x1xf16>
    return %RESULT : tensor<1x512x1xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 512, 16, 1]} : tensor<1x512x16xf16> -> tensor<1x512x16x1xf16>

    // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool([[RESHAPE_INPUT]])
    // CHECK-SAME:      kernel_size = [16, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [16, 1]
    // CHECK-SAME:      tensor<1x512x16x1xf16> -> tensor<1x512x1x1xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[MAXPOOL]]) {shape_value = [1, 512, 1]} : tensor<1x512x1x1xf16> -> tensor<1x512x1xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x512x1xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DAvgpool
func.func @ConvertNceOpsTo4DAvgpool(%arg0: tensor<1x512x16xf16>) -> tensor<1x512x1xf16> {
    %RESULT = IE.AvgPool(%arg0) {
        kernel_size = [16], pads_begin = [0], pads_end = [0], rounding_type = #IE.rounding_type<FLOOR>, strides = [16]} : tensor<1x512x16xf16> -> tensor<1x512x1xf16>
    return %RESULT : tensor<1x512x1xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 512, 16, 1]} : tensor<1x512x16xf16> -> tensor<1x512x16x1xf16>

    // CHECK:       [[AVGPOOL:%.*]] = IE.AvgPool([[RESHAPE_INPUT]])
    // CHECK-SAME:      kernel_size = [16, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [16, 1]
    // CHECK-SAME:      tensor<1x512x16x1xf16> -> tensor<1x512x1x1xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[AVGPOOL]]) {shape_value = [1, 512, 1]} : tensor<1x512x1x1xf16> -> tensor<1x512x1xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x512x1xf16>
}

// -----

!qElemType1 = !quant.uniform<i8:f16, 0.3>
!qElemType2 = !quant.uniform<u8:f16, 0.3:128>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16, 3.000000e-01>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16, 3.000000e-01:128>

// CHECK: func.func @ConvertNceOpsTo4DAvgpool_DifferentInOutTypes
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x4332x320x[[QTYPE1]]>) -> tensor<1x4332x320x[[QTYPE2]]>
func.func @ConvertNceOpsTo4DAvgpool_DifferentInOutTypes(%arg0: tensor<1x4332x320x!qElemType1>)
        -> tensor<1x4332x320x!qElemType2> {
    %RESULT = IE.AvgPool(%arg0)
        {exclude_pads, kernel_size = [1], pads_begin = [0], pads_end = [0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1]}
        : tensor<1x4332x320x!qElemType1> -> tensor<1x4332x320x!qElemType2>
    return %RESULT : tensor<1x4332x320x!qElemType2>

    // CHECK: [[RESHAPE_TO_4D:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [1, 4332, 320, 1]}

    // CHECK: [[AVGPOOL:%.+]] = IE.AvgPool([[RESHAPE_TO_4D]])
    // CHECK-SAME:  exclude_pads
    // CHECK-SAME:  kernel_size = [1, 1]
    // CHECK-SAME:  pads_begin = [0, 0]
    // CHECK-SAME:  pads_end = [0, 0]
    // CHECK-SAME:  rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  tensor<1x4332x320x1x[[QTYPE1]]> -> tensor<1x4332x320x1x[[QTYPE2]]>

    // CHECK: [[RESHAPE_FROM_4D:%.+]] = IE.Reshape([[AVGPOOL]]) {shape_value = [1, 4332, 320]}
    // CHECK: return [[RESHAPE_FROM_4D]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4D3DMaxpoolwithDepthOne
func.func @ConvertNceOpsTo4D3DMaxpoolwithDepthOne(%arg0: tensor<1x32x25x28x28xf16>) -> tensor<1x32x25x14x14xf16> {
    %RESULT = IE.MaxPool(%arg0) {
        kernel_size = [1, 3, 3], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2, 2]} : tensor<1x32x25x28x28xf16> -> tensor<1x32x25x14x14xf16>
    return %RESULT : tensor<1x32x25x14x14xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 800, 28, 28]} : tensor<1x32x25x28x28xf16> -> tensor<1x800x28x28xf16>

    // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool([[RESHAPE_INPUT]])
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:      tensor<1x800x28x28xf16> -> tensor<1x800x14x14xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[MAXPOOL]]) {shape_value = [1, 32, 25, 14, 14]} : tensor<1x800x14x14xf16> -> tensor<1x32x25x14x14xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x32x25x14x14xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4D3DMaxpoolwithDepthNotOne
func.func @ConvertNceOpsTo4D3DMaxpoolwithDepthNotOne(%arg0: tensor<1x32x25x28x28xf16>) -> tensor<1x32x23x14x14xf16> {
    %RESULT = IE.MaxPool(%arg0) {
        kernel_size = [3, 3, 3], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2, 2]} : tensor<1x32x25x28x28xf16> -> tensor<1x32x23x14x14xf16>
    return %RESULT : tensor<1x32x23x14x14xf16>

    // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool(%arg0)
    // CHECK-SAME:      kernel_size = [3, 3, 3]
    // CHECK-SAME:      pads_begin = [0, 1, 1]
    // CHECK-SAME:      pads_end = [0, 1, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [1, 2, 2]
    // CHECK-SAME:      tensor<1x32x25x28x28xf16> -> tensor<1x32x23x14x14xf16>

    // CHECK: return [[MAXPOOL]] : tensor<1x32x23x14x14xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4D3DAvgpoolwithDepthOne
func.func @ConvertNceOpsTo4D3DAvgpoolwithDepthOne(%arg0: tensor<1x32x25x28x28xf16>) -> tensor<1x32x25x14x14xf16> {
    %RESULT = IE.AvgPool(%arg0) {
        kernel_size = [1, 3, 3], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2, 2]} : tensor<1x32x25x28x28xf16> -> tensor<1x32x25x14x14xf16>
    return %RESULT : tensor<1x32x25x14x14xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 800, 28, 28]} : tensor<1x32x25x28x28xf16> -> tensor<1x800x28x28xf16>

    // CHECK:       [[AVGPOOL:%.*]] = IE.AvgPool([[RESHAPE_INPUT]])
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [2, 2]
    // CHECK-SAME:      tensor<1x800x28x28xf16> -> tensor<1x800x14x14xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[AVGPOOL]]) {shape_value = [1, 32, 25, 14, 14]} : tensor<1x800x14x14xf16> -> tensor<1x32x25x14x14xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x32x25x14x14xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4D3DAvgpoolwithDepthNotOne
func.func @ConvertNceOpsTo4D3DAvgpoolwithDepthNotOne(%arg0: tensor<1x32x25x28x28xf16>) -> tensor<1x32x23x14x14xf16> {
    %RESULT = IE.AvgPool(%arg0) {
        kernel_size = [3, 3, 3], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2, 2]} : tensor<1x32x25x28x28xf16> -> tensor<1x32x23x14x14xf16>
    return %RESULT : tensor<1x32x23x14x14xf16>

    // CHECK:       [[AVGPOOL:%.*]] = IE.AvgPool(%arg0)
    // CHECK-SAME:      kernel_size = [3, 3, 3]
    // CHECK-SAME:      pads_begin = [0, 1, 1]
    // CHECK-SAME:      pads_end = [0, 1, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [1, 2, 2]
    // CHECK-SAME:      tensor<1x32x25x28x28xf16> -> tensor<1x32x23x14x14xf16>

    // CHECK: return [[AVGPOOL]] : tensor<1x32x23x14x14xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DExtendOnH
func.func @ConvertNceOpsTo4DExtendOnH(%arg0: tensor<1x4x1x128xf16>) -> tensor<1x8x1x128xf16> {
    %reshape_in = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 4, 128]} : tensor<1x4x1x128xf16> -> tensor<1x4x128xf16>
    %cst = const.Declare tensor<8x4x1xf16> = dense<1.000000e+00> : tensor<8x4x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[8, 4, 1]>]
    %conv = IE.Convolution(%reshape_in, %cst) {dilations = [1], pads_begin = [0], pads_end = [0], strides = [1]} : tensor<1x4x128xf16>, tensor<8x4x1xf16> -> tensor<1x8x128xf16>
    %reshape_out = IE.AffineReshape(%conv) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 8, 1, 128]} : tensor<1x8x128xf16> -> tensor<1x8x1x128xf16>

    return %reshape_out : tensor<1x8x1x128xf16>

    // CHECK:               [[CONV:%.+]] = IE.Convolution
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:              tensor<1x4x1x128xf16>, tensor<8x4x1x1xf16> -> tensor<1x8x1x128xf16>
    // CHECK:               [[RESHAPE:%.+]] = IE.Reshape([[CONV]]) {
    // CHECK-SAME:              shape_value = [1, 8, 128]
    // CHECK-SAME:          } : tensor<1x8x1x128xf16> -> tensor<1x8x128xf16>
    // CHECK:               [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[RESHAPE]]) {
    // CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1, 2], [3]],
    // CHECK-SAME{LITERAL}:     shape_value = [1, 8, 1, 128]
    // CHECK-SAME{LITERAL}: } : tensor<1x8x128xf16> -> tensor<1x8x1x128xf16>
    // CHECK: return [[AFFINE_RESHAPE]] : tensor<1x8x1x128xf16>
}

// -----

// CHECK-LABEL: @ConvertBatchNceOpsTo4DExtendOnH
// CHECK-SAME:      [[INPUT:%.+]]: tensor<2x9x121xf16>
func.func @ConvertBatchNceOpsTo4DExtendOnH(%arg0: tensor<2x9x121xf16>) -> tensor<2x32x121xf16> {
    %cst = const.Declare tensor<32x9x1xf16> = dense<1.000000e+00> : tensor<32x9x1xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1], pads_begin = [0], pads_end = [0], strides = [1]} : tensor<2x9x121xf16>, tensor<32x9x1xf16> -> tensor<2x32x121xf16>

    return %conv : tensor<2x32x121xf16>

    // CHECK:               [[RESHAPE_0:%.+]] = IE.Reshape([[INPUT]]) {
    // CHECK-SAME:              shape_value = [2, 9, 1, 121]
    // CHECK-SAME:          } : tensor<2x9x121xf16> -> tensor<2x9x1x121xf16>
    // CHECK:               [[CONV:%.+]] = IE.Convolution
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:              tensor<2x9x1x121xf16>, tensor<32x9x1x1xf16> -> tensor<2x32x1x121xf16>
    // CHECK:               [[RESHAPE_1:%.+]] = IE.Reshape([[CONV]]) {
    // CHECK-SAME:              shape_value = [2, 32, 121]
    // CHECK-SAME:          } : tensor<2x32x1x121xf16> -> tensor<2x32x121xf16>
    // CHECK:               return [[RESHAPE_1]] : tensor<2x32x121xf16>
}
