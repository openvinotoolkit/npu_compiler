//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-fc-to-conv %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertFullyConnectedToConvolution
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x16xf32>)
func.func @ConvertFullyConnectedToConvolution(%arg0: tensor<1x16xf32>) -> tensor<1x64xf32> {
    %weights = const.Declare tensor<64x16xf32> = dense<1.0> : tensor<64x16xf32>
    %bias = const.Declare tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf32>, tensor<64x16xf32>, tensor<1x64xf32> -> tensor<1x64xf32>

    return %0 : tensor<1x64xf32>

    // CHECK-NOT:   IE.FullyConnected
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<64x16xf32> = dense<1.000000e+00> : tensor<64x16xf32>
    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x64xf32> = dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK:       [[VAL0:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [1, 16, 1, 1]} : tensor<1x16xf32> -> tensor<1x16x1x1xf32>
    // CHECK:       [[VAL1:%.+]] = IE.Reshape([[WEIGHTS]]) {shape_value = [64, 16, 1, 1]} : tensor<64x16xf32> -> tensor<64x16x1x1xf32>
    // CHECK:       [[VAL2:%.+]] = IE.Reshape([[BIAS]]) {shape_value = [1, 64, 1, 1]} : tensor<1x64xf32> -> tensor<1x64x1x1xf32>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[VAL0]], [[VAL1]], [[VAL2]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       [[VAL3:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 64]} : tensor<1x64x1x1xf32> -> tensor<1x64xf32>
    // CHECK:       return [[VAL3]]
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>

// A FullyConnected with a dynamic batch cannot be lowered to a convolution directly: nothing on the NCE path
// accepts a dynamic tensor. DynamicFullyConnectedRewriter first pads the rows up to their bound with
// IE.DynamicExpand, so the convolution itself is fully static, then restores the dynamic result type with an
// IE.DynamicReshape whose target shape is computed at runtime from ShapeOf(input)[0]. A constant -1
// placeholder would not work here - it would be resolved against the already-padded activation.

// CHECK-LABEL: @DynamicFullyConnected
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<?x128xf32, {bounds = #const.OpaqueI64Elements<[14, 128]> : tensor<2xsi64>, order = #NC}>)
func.func @DynamicFullyConnected(%arg0: tensor<?x128xf32, {bounds = #const.OpaqueI64Elements<[14, 128]> : tensor<2xsi64>, order = #NC}>) -> tensor<?x1xf32, {bounds = #const.OpaqueI64Elements<[14, 1]> : tensor<2xsi64>, order = #NC}> {
    %weights = const.Declare tensor<1x128xf32> = dense<1.0> : tensor<1x128xf32>
    %0 = IE.FullyConnected(%arg0, %weights) : tensor<?x128xf32, {bounds = #const.OpaqueI64Elements<[14, 128]> : tensor<2xsi64>, order = #NC}>, tensor<1x128xf32> -> tensor<?x1xf32, {bounds = #const.OpaqueI64Elements<[14, 1]> : tensor<2xsi64>, order = #NC}>

    return %0 : tensor<?x1xf32, {bounds = #const.OpaqueI64Elements<[14, 1]> : tensor<2xsi64>, order = #NC}>

    // CHECK-NOT:   IE.FullyConnected
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<1x128xf32> = dense<1.000000e+00> : tensor<1x128xf32>
    // CHECK:       [[EXPANDED:%.+]] = IE.DynamicExpand([[ARG_0]]) : tensor<?x128xf32, {bounds = #const.OpaqueI64Elements<[14, 128]> : tensor<2xsi64>, order = #NC}> -> tensor<14x128xf32>
    // CHECK:       [[IN_4D:%.+]] = IE.Reshape([[EXPANDED]]) {shape_value = [14, 128, 1, 1]} : tensor<14x128xf32> -> tensor<14x128x1x1xf32>
    // CHECK:       [[FILTER_4D:%.+]] = IE.Reshape([[WEIGHTS]]) {shape_value = [1, 128, 1, 1]} : tensor<1x128xf32> -> tensor<1x128x1x1xf32>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[IN_4D]], [[FILTER_4D]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<14x128x1x1xf32>, tensor<1x128x1x1xf32> -> tensor<14x1x1x1xf32>
    // CHECK:       [[OUT_2D:%.+]] = IE.Reshape([[CONV]]) {shape_value = [14, 1]} : tensor<14x1x1x1xf32> -> tensor<14x1xf32>
    // CHECK:       [[SHAPE_OF:%.+]] = IE.ShapeOf([[ARG_0]]) {dstElemType = si32} : tensor<?x128xf32, {bounds = #const.OpaqueI64Elements<[14, 128]> : tensor<2xsi64>, order = #NC}> -> tensor<2xsi32>
    // CHECK:       [[ROW_DIM:%.+]] = IE.Slice [[SHAPE_OF]] [0] [1] : tensor<2xsi32> to tensor<1xsi32>
    // CHECK:       [[COL_DIM:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    // CHECK:       [[TARGET_SHAPE:%.+]] = IE.Concat([[ROW_DIM]], [[COL_DIM]]) {static_offsets = {{\[\[}}0], [1]]} : tensor<1xsi32>, tensor<1xsi32> -> tensor<2xsi32>
    // CHECK:       [[OUT:%.+]] = IE.DynamicReshape([[OUT_2D]], [[TARGET_SHAPE]]) {output_bounds = [14, 1], output_shape = [-9223372036854775808, 1]}
    // CHECK-SAME:      tensor<14x1xf32>, tensor<2xsi32> -> tensor<?x1xf32, {bounds = #const.OpaqueI64Elements<[14, 1]> : tensor<2xsi64>, order = #NC}>
    // CHECK:       return [[OUT]] : tensor<?x1xf32, {bounds = #const.OpaqueI64Elements<[14, 1]> : tensor<2xsi64>, order = #NC}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>

// Same as above, with a bias. Weights and bias are static, so they keep using plain IE.Reshape; only the
// activation is padded and only the result is re-narrowed.

// CHECK-LABEL: @DynamicFullyConnectedWithBias
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<?x128xf32, {bounds = #const.OpaqueI64Elements<[14, 128]> : tensor<2xsi64>, order = #NC}>)
func.func @DynamicFullyConnectedWithBias(%arg0: tensor<?x128xf32, {bounds = #const.OpaqueI64Elements<[14, 128]> : tensor<2xsi64>, order = #NC}>) -> tensor<?x64xf32, {bounds = #const.OpaqueI64Elements<[14, 64]> : tensor<2xsi64>, order = #NC}> {
    %weights = const.Declare tensor<64x128xf32> = dense<1.0> : tensor<64x128xf32>
    %bias = const.Declare tensor<1x64xf32> = dense<1.0> : tensor<1x64xf32>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<?x128xf32, {bounds = #const.OpaqueI64Elements<[14, 128]> : tensor<2xsi64>, order = #NC}>, tensor<64x128xf32>, tensor<1x64xf32> -> tensor<?x64xf32, {bounds = #const.OpaqueI64Elements<[14, 64]> : tensor<2xsi64>, order = #NC}>

    return %0 : tensor<?x64xf32, {bounds = #const.OpaqueI64Elements<[14, 64]> : tensor<2xsi64>, order = #NC}>

    // CHECK-NOT:   IE.FullyConnected
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<64x128xf32> = dense<1.000000e+00> : tensor<64x128xf32>
    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x64xf32> = dense<1.000000e+00> : tensor<1x64xf32>
    // CHECK:       [[EXPANDED:%.+]] = IE.DynamicExpand([[ARG_0]]) : tensor<?x128xf32, {bounds = #const.OpaqueI64Elements<[14, 128]> : tensor<2xsi64>, order = #NC}> -> tensor<14x128xf32>
    // CHECK:       [[IN_4D:%.+]] = IE.Reshape([[EXPANDED]]) {shape_value = [14, 128, 1, 1]} : tensor<14x128xf32> -> tensor<14x128x1x1xf32>
    // CHECK:       [[FILTER_4D:%.+]] = IE.Reshape([[WEIGHTS]]) {shape_value = [64, 128, 1, 1]} : tensor<64x128xf32> -> tensor<64x128x1x1xf32>
    // CHECK:       [[BIAS_4D:%.+]] = IE.Reshape([[BIAS]]) {shape_value = [1, 64, 1, 1]} : tensor<1x64xf32> -> tensor<1x64x1x1xf32>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[IN_4D]], [[FILTER_4D]], [[BIAS_4D]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<14x128x1x1xf32>, tensor<64x128x1x1xf32>, tensor<1x64x1x1xf32> -> tensor<14x64x1x1xf32>
    // CHECK:       [[OUT_2D:%.+]] = IE.Reshape([[CONV]]) {shape_value = [14, 64]} : tensor<14x64x1x1xf32> -> tensor<14x64xf32>
    // CHECK:       [[SHAPE_OF:%.+]] = IE.ShapeOf([[ARG_0]]) {dstElemType = si32} : tensor<?x128xf32, {bounds = #const.OpaqueI64Elements<[14, 128]> : tensor<2xsi64>, order = #NC}> -> tensor<2xsi32>
    // CHECK:       [[ROW_DIM:%.+]] = IE.Slice [[SHAPE_OF]] [0] [1] : tensor<2xsi32> to tensor<1xsi32>
    // CHECK:       [[COL_DIM:%.+]] = const.Declare tensor<1xsi32> = dense<64> : tensor<1xsi32>
    // CHECK:       [[TARGET_SHAPE:%.+]] = IE.Concat([[ROW_DIM]], [[COL_DIM]]) {static_offsets = {{\[\[}}0], [1]]} : tensor<1xsi32>, tensor<1xsi32> -> tensor<2xsi32>
    // CHECK:       [[OUT:%.+]] = IE.DynamicReshape([[OUT_2D]], [[TARGET_SHAPE]]) {output_bounds = [14, 64], output_shape = [-9223372036854775808, 64]}
    // CHECK-SAME:      tensor<14x64xf32>, tensor<2xsi32> -> tensor<?x64xf32, {bounds = #const.OpaqueI64Elements<[14, 64]> : tensor<2xsi64>, order = #NC}>
    // CHECK:       return [[OUT]] : tensor<?x64xf32, {bounds = #const.OpaqueI64Elements<[14, 64]> : tensor<2xsi64>, order = #NC}>
}
