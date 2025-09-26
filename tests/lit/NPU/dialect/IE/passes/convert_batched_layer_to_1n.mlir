//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-batched-layer-to-1n %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @main(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<4x16x1x1xf16>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x16x1x1xf16>, tensor<4x16x1x1xf16> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[IN_TRANSPOSE:%.+]] = IE.Transpose(%arg0) {order_value = #map} : tensor<5x16x1x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[IN_TRANSPOSE]], %arg1)
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:   tensor<1x16x5x1xf16>, tensor<4x16x1x1xf16> -> tensor<1x4x5x1xf16>
    // CHECK: [[OUT_TRANSPOSE:%.+]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x4x5x1xf16> -> tensor<5x4x1x1xf16>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<5x4x1x1xf16>
}

// -----

#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16:0, {0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @MixedPrecisionCase(%arg0: tensor<5x16x1x1x!qElemType>, %arg1: tensor<4x16x1x1x!qElemType1>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x16x1x1x!qElemType>, tensor<4x16x1x1x!qElemType1> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[IN_TRANSPOSE:%.+]] = IE.Transpose(%arg0) {order_value = #map} : tensor<5x16x1x1x!qElemType> -> tensor<1x16x5x1x!qElemType>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[IN_TRANSPOSE]], %arg1)
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:   tensor<1x16x5x1x!qElemType>, tensor<4x16x1x1x!qElemType1> -> tensor<1x4x5x1xf16>
    // CHECK: [[OUT_TRANSPOSE:%.+]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x4x5x1xf16> -> tensor<5x4x1x1xf16>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<5x4x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @ConvertWithWNotEq1(%arg0: tensor<512x48x1x336xf16>, %arg1: tensor<6x48x1x3xf16>) -> tensor<512x6x1x336xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<512x48x1x336xf16>, tensor<6x48x1x3xf16> -> tensor<512x6x1x336xf16>
    return %0 : tensor<512x6x1x336xf16>

    // CHECK: [[IN_TRANSPOSE:%.+]] = IE.Transpose(%arg0) {order_value = #map} : tensor<512x48x1x336xf16> -> tensor<1x48x512x336xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[IN_TRANSPOSE]], %arg1)
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} :
    // CHECK-SAME:   tensor<1x48x512x336xf16>, tensor<6x48x1x3xf16> -> tensor<1x6x512x336xf16>
    // CHECK: [[OUT_TRANSPOSE:%.+]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x6x512x336xf16> -> tensor<512x6x1x336xf16>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<512x6x1x336xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>
!qElemType1 = !quant.uniform<u8:f16:0, {0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @NoChagesPerAxisQuantization(%arg0: tensor<5x8x1x1x!qElemType>, %arg1: tensor<4x8x1x1x!qElemType1>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x8x1x1x!qElemType>, tensor<4x8x1x1x!qElemType1> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution(%arg0, %arg1)
    // CHECK: return [[CONV]] : tensor<5x4x1x1xf16>
}

// -----

func.func @NoChangesFilterPlaneNotEq1(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<4x16x2x2xf16>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<5x16x1x1xf16>, tensor<4x16x2x2xf16> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution(%arg0, %arg1)
    // CHECK: return [[CONV]] : tensor<5x4x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @ConvertAdd(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<5x16x1x1xf16>) -> tensor<5x16x1x1xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<5x16x1x1xf16>, tensor<5x16x1x1xf16> -> tensor<5x16x1x1xf16>
    return %0 : tensor<5x16x1x1xf16>

    // CHECK: [[IN_TRANSPOSE1:%.+]] = IE.Transpose(%arg0) {order_value = #map} : tensor<5x16x1x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[IN_TRANSPOSE2:%.+]] = IE.Transpose(%arg1) {order_value = #map} : tensor<5x16x1x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[ADD:%.+]] = IE.Add([[IN_TRANSPOSE1]], [[IN_TRANSPOSE2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x5x1xf16>, tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[OUT_TRANSPOSE:%.+]] = IE.Transpose([[ADD]]) {order_value = #map} : tensor<1x16x5x1xf16> -> tensor<5x16x1x1xf16>
    // CHECK: return [[OUT_TRANSPOSE]] : tensor<5x16x1x1xf16>
}

// -----

func.func @NoChangesAddOp(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<5x16x1x1xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<5x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<5x16x1x1xf16>
    return %0 : tensor<5x16x1x1xf16>

    // CHECK: [[ADD:%.+]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<5x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<5x16x1x1xf16>
    // CHECK: return [[ADD]] : tensor<5x16x1x1xf16>
}

// -----

func.func @ConvertAddWithHW(%arg0: tensor<50x16x10x10xf16>, %arg1: tensor<50x16x10x10xf16>) -> tensor<50x16x10x10xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<50x16x10x10xf16>, tensor<50x16x10x10xf16> -> tensor<50x16x10x10xf16>
    return %0 : tensor<50x16x10x10xf16>

    // CHECK: [[SHAPE_CAST_IN1:%.+]]  = IE.ShapeCast {shape = [1, 800, 10, 10]} inputs({{[^:]+}} : tensor<50x16x10x10xf16>) -> tensor<1x800x10x10xf16>
    // CHECK: [[SHAPE_CAST_IN2:%.+]] = IE.ShapeCast {shape = [1, 800, 10, 10]} inputs({{[^:]+}} : tensor<50x16x10x10xf16>) -> tensor<1x800x10x10xf16>
    // CHECK: [[ADD:%.+]] = IE.Add([[SHAPE_CAST_IN1]], [[SHAPE_CAST_IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x800x10x10xf16>, tensor<1x800x10x10xf16> -> tensor<1x800x10x10xf16>
    // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [50, 16, 10, 10]} inputs([[ADD]] : tensor<1x800x10x10xf16>) -> tensor<50x16x10x10xf16>
    // CHECK: return [[SHAPE_CAST_OUT]] : tensor<50x16x10x10xf16>
}

// -----

func.func @ConvertAddWithHWWithDiffInputShape(%arg0: tensor<50x16x10x10xf16>, %arg1: tensor<50x16x1x1xf16>) -> tensor<50x16x10x10xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<50x16x10x10xf16>, tensor<50x16x1x1xf16> -> tensor<50x16x10x10xf16>
    return %0 : tensor<50x16x10x10xf16>

    // CHECK: [[SHAPE_CAST_IN1:%.+]] = IE.ShapeCast {shape = [1, 800, 10, 10]} inputs({{[^:]+}} : tensor<50x16x10x10xf16>) -> tensor<1x800x10x10xf16>
    // CHECK: [[SHAPE_CAST_IN2:%.+]] = IE.ShapeCast {shape = [1, 800, 1, 1]} inputs({{[^:]+}} : tensor<50x16x1x1xf16>) -> tensor<1x800x1x1xf16>
    // CHECK: [[ADD:%.+]] = IE.Add([[SHAPE_CAST_IN1]], [[SHAPE_CAST_IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x800x10x10xf16>, tensor<1x800x1x1xf16> -> tensor<1x800x10x10xf16>
    // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [50, 16, 10, 10]} inputs([[ADD]] : tensor<1x800x10x10xf16>) -> tensor<50x16x10x10xf16>
    // CHECK: return [[SHAPE_CAST_OUT]] : tensor<50x16x10x10xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

// CHECK-LABEL: func.func @ConvertAddWithQuant
// CHECK-SAME: ([[ARG:%.+]]: tensor<2x3x224x224xf16>)
func.func @ConvertAddWithQuant(%arg0: tensor<2x3x224x224xf16>) -> tensor<2x3x224x224x!quant.uniform<u8:f16, 2.000000e+00>> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x3x224x224xf16>, tensor<2x3x224x224xf16> -> tensor<2x3x224x224x!quant.uniform<u8:f16, 2.000000e+00>>
    return %0 : tensor<2x3x224x224x!quant.uniform<u8:f16, 2.000000e+00>>
    // CHECK: [[SHAPE_CAST_IN1:%.+]] = IE.ShapeCast {shape = [1, 6, 224, 224]} inputs([[ARG]] : tensor<2x3x224x224xf16>) -> tensor<1x6x224x224xf16>
    // CHECK: [[SHAPE_CAST_IN2:%.+]] = IE.ShapeCast {shape = [1, 6, 224, 224]} inputs([[ARG]] : tensor<2x3x224x224xf16>) -> tensor<1x6x224x224xf16>
    // CHECK: [[ADD:%.+]] = IE.Add([[SHAPE_CAST_IN1]], [[SHAPE_CAST_IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x6x224x224xf16>, tensor<1x6x224x224xf16> -> tensor<1x6x224x224x!qElemType>
    // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [2, 3, 224, 224]} inputs([[ADD]] : tensor<1x6x224x224x!qElemType>) -> tensor<2x3x224x224x!qElemType>
    // CHECK: return [[SHAPE_CAST_OUT]] : tensor<2x3x224x224x!qElemType>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @ConvertGroupConvolution(%arg0: tensor<16x16x1x32xf16>) -> tensor<16x16x1x32xf16> {
    %filters = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %2 = IE.GroupConvolution(%arg0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<16x16x1x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<16x16x1x32xf16>
    return %2 : tensor<16x16x1x32xf16>

    // CHECK: [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    // CHECK: [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK: [[IN_TRANSPOSE:%.+]] = IE.Transpose(%arg0) {order_value = #map} : tensor<16x16x1x32xf16> -> tensor<1x16x16x32xf16>
    // CHECK: [[CONV:%.+]] = IE.GroupConvolution([[IN_TRANSPOSE]], [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x16x32xf16>
    // CHECK: [[OUT_TRANSPOSE:%.+]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x16x16x32xf16> -> tensor<16x16x1x32xf16>
    // CHECK: return [[OUT_TRANSPOSE]] : tensor<16x16x1x32xf16>

}

// -----

func.func @NoChangesGroupConvolution(%arg0: tensor<16x16x2x32xf16>) -> tensor<16x16x2x32xf16> {
    %filters = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    %bias = const.Declare tensor<1x16x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    %2 = IE.GroupConvolution(%arg0, %filters, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<16x16x2x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<16x16x2x32xf16>
    return %2 : tensor<16x16x2x32xf16>

    // CHECK: [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<0 : i64, 16 : i64>]
    // CHECK: [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK-NOT: IE.Transpose
    // CHECK: [[CONV:%.+]] = IE.GroupConvolution(%arg0, [[FILTER]], [[BIAS]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<16x16x2x32xf16>, tensor<16x1x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<16x16x2x32xf16>
    // CHECK: return [[CONV]] : tensor<16x16x2x32xf16>
}

// -----

// CHECK: func.func @ConvertIdentityMaxPool([[ARG0:%.+]]: tensor<16x16x1x3xf16>)
func.func @ConvertIdentityMaxPool(%arg0: tensor<16x16x1x3xf16>) -> tensor<16x16x1x3xf16> {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<16x16x1x3xf16> -> tensor<16x16x1x3xf16>

    return %0 : tensor<16x16x1x3xf16>

    // CHECK:               [[IN_RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16, 16, 3]} : tensor<16x16x1x3xf16> -> tensor<1x16x16x3xf16>

    // CHECK:               [[MAXPOOL:%.+]] = IE.MaxPool([[IN_RESHAPE]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:                                  rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x16x3xf16> -> tensor<1x16x16x3xf16>

    // CHECK:               [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[MAXPOOL]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [16, 16, 1, 3]} : tensor<1x16x16x3xf16> -> tensor<16x16x1x3xf16>
    // CHECK:               return [[OUT_RESHAPE]] : tensor<16x16x1x3xf16>
}

// -----

// CHECK: [[MAP:#.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK: func.func @ConvertMaxPool([[ARG0:%.+]]: tensor<16x16x1x6xf16>)
func.func @ConvertMaxPool(%arg0: tensor<16x16x1x6xf16>) -> tensor<16x16x1x5xf16> {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [1, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<16x16x1x6xf16> -> tensor<16x16x1x5xf16>

    return %0 : tensor<16x16x1x5xf16>

    // CHECK:       [[IN_TRANSPOSE:%.+]] = IE.Transpose([[ARG0]]) {order_value = [[MAP]]} : tensor<16x16x1x6xf16> -> tensor<1x16x16x6xf16>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[IN_TRANSPOSE]]) {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:                                  rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x16x6xf16> -> tensor<1x16x16x5xf16>
    // CHECK:       [[OUT_TRANSPOSE:%.+]] = IE.Transpose([[MAXPOOL]]) {order_value = [[MAP]]} : tensor<1x16x16x5xf16> -> tensor<16x16x1x5xf16>

    // CHECK:       return [[OUT_TRANSPOSE]]  : tensor<16x16x1x5xf16>
}

// -----

// CHECK: func.func @NoChangesMaxPool([[ARG0:%.+]]: tensor<16x16x3x6xf16>)
func.func @NoChangesMaxPool(%arg0: tensor<16x16x3x6xf16>) -> tensor<16x16x3x5xf16> {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [1, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<16x16x3x6xf16> -> tensor<16x16x3x5xf16>

    return %0 : tensor<16x16x3x5xf16>

    // CHECK-NOT:       IE.Transpose
    // CHECK-NOT:       IE.AffineReshape

    // CHECK:           [[MAXPOOL:%.+]] = IE.MaxPool([[ARG0]]) {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:                          rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<16x16x3x6xf16> -> tensor<16x16x3x5xf16>
    // CHECK:           return [[MAXPOOL]]  : tensor<16x16x3x5xf16>
}

// -----

!qElemType1 = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK: func.func @ConvertIdentityAvgPool([[ARG0:%.+]]: tensor<16x16x1x3x!qElemType>)
func.func @ConvertIdentityAvgPool(%arg0: tensor<16x16x1x3x!qElemType1>) -> tensor<16x16x1x3x!qElemType2> {
    %0 = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<16x16x1x3x!qElemType1> -> tensor<16x16x1x3x!qElemType2>

    return %0 : tensor<16x16x1x3x!qElemType2>

    // CHECK:               [[IN_RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16, 16, 3]} : tensor<16x16x1x3x!qElemType> -> tensor<1x16x16x3x!qElemType>

    // CHECK:               [[MAXPOOL:%.+]] = IE.AvgPool([[IN_RESHAPE]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:                                  rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x16x3x!qElemType> -> tensor<1x16x16x3x!qElemType1>

    // CHECK:               [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[MAXPOOL]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [16, 16, 1, 3]} : tensor<1x16x16x3x!qElemType1> -> tensor<16x16x1x3x!qElemType1>
    // CHECK:               return [[OUT_RESHAPE]] : tensor<16x16x1x3x!qElemType1>
}

// -----

// CHECK-LABEL: @ConvertSigmoid
// CHECK-SAME: [[INPUT:%.+]]: tensor<9376x3x1x1xf16>
func.func @ConvertSigmoid(%arg0: tensor<9376x3x1x1xf16>) -> tensor<9376x3x1x1xf16> {
    %0 = IE.Sigmoid(%arg0) : tensor<9376x3x1x1xf16> -> tensor<9376x3x1x1xf16>

    return %0 : tensor<9376x3x1x1xf16>

    // CHECK: [[IN_AFFINERESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-LITERAL: {dim_mapping = [[0, 1], [1], [2], [3]], shape_value = [1, 28128, 1, 1]} : tensor<9376x3x1x1xf16> -> tensor<1x28128x1x1xf16>
    // CHECK: [[SIGMOID:%.+]] = IE.Sigmoid([[IN_AFFINERESHAPE]]) : tensor<1x28128x1x1xf16> -> tensor<1x28128x1x1xf16>
    // CHECK: [[OUT_AFFINERESHAPE:%.+]] = IE.AffineReshape([[SIGMOID]])
    // CHECK-LITERAL: {dim_mapping = [[0], [0, 1], [2], [3]], shape_value = [9376, 3, 1, 1]} : tensor<1x28128x1x1xf16> -> tensor<9376x3x1x1xf16>

    // CHECK: return [[OUT_AFFINERESHAPE]] : tensor<9376x3x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @ConvertConvWithSharedInputAndWeight
// CHECK-SAME:      [[INPUT:%.+]]: tensor<16x96x1x1xf16>
func.func @ConvertConvWithSharedInputAndWeight(%arg0: tensor<16x96x1x1xf16>) -> tensor<16x16x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<16x96x1x1xf16>, tensor<16x96x1x1xf16> -> tensor<16x16x1x1xf16>
    return %0 : tensor<16x16x1x1xf16>
    // CHECK: [[IN_TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #map} : tensor<16x96x1x1xf16> -> tensor<1x96x16x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[IN_TRANSPOSE]], [[INPUT]])
    // CHECK-SAME:          {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:          tensor<1x96x16x1xf16>, tensor<16x96x1x1xf16> -> tensor<1x16x16x1xf16>
    // CHECK: [[OUT_TRANSPOSE:%.+]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x16x16x1xf16> -> tensor<16x16x1x1xf16>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<16x16x1x1xf16>
}

// -----

// CHECK-LABEL: @ReshapeGroupConv
// CHECK-SAME: [[INPUT:%.+]]: tensor<1024x1x16x32xf16>
func.func @ReshapeGroupConv(%arg0: tensor<1024x1x16x32xf16>) -> tensor<1024x1x16x32xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %grp_conv = IE.GroupConvolution(%arg0, %cst) {
        dilations = [1, 1], groups = 1 : i64,
        pads_begin = [0, 0], pads_end = [0, 0],
        strides = [1, 1]
        } : tensor<1024x1x16x32xf16>, tensor<1x1x1x1xf16> -> tensor<1024x1x16x32xf16>

    return %grp_conv: tensor<1024x1x16x32xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1024x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 1024 : i64>]
    // CHECK-DAG: [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 1024, 16, 32]} inputs([[INPUT]] : tensor<1024x1x16x32xf16>) -> tensor<1x1024x16x32xf16>
    // CHECK: [[GRP_CONV:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[CST]]) {
    // CHECK-SAME:                  dilations = [1, 1], groups = 1024 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:                  } : tensor<1x1024x16x32xf16>, tensor<1024x1x1x1xf16> -> tensor<1x1024x16x32xf16>
    // CHECK: [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [1024, 1, 16, 32]} inputs([[GRP_CONV]] : tensor<1x1024x16x32xf16>) -> tensor<1024x1x16x32xf16>
    // CHECK: return [[SHAPECAST_OUT]] : tensor<1024x1x16x32xf16>
}

// -----

// CHECK-LABEL: @ReshapeMultiply
// CHECK-SAME: [[INPUT:%.+]]: tensor<1024x1x16x32xf16>
func.func @ReshapeMultiply(%arg0: tensor<1024x1x16x32xf16>) -> tensor<1024x1x16x32xf16> {
    %cst = const.Declare tensor<1024x1x1x32xf16> = dense<-1.0> : tensor<1024x1x1x32xf16>
    %multiply = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1024x1x16x32xf16>, tensor<1024x1x1x32xf16> -> tensor<1024x1x16x32xf16>

    return %multiply : tensor<1024x1x16x32xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1024x1x1x32xf16> = dense<-1.000000e+00> : tensor<1024x1x1x32xf16>
    // CHECK: [[SHAPE_CAST1:%.+]] = IE.ShapeCast {shape = [1, 1024, 16, 32]} inputs([[INPUT]] : tensor<1024x1x16x32xf16>) -> tensor<1x1024x16x32xf16>
    // CHECK: [[SHAPE_CAST2:%.+]] = IE.ShapeCast {shape = [1, 1024, 1, 32]} inputs([[CST]] : tensor<1024x1x1x32xf16>) -> tensor<1x1024x1x32xf16>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[SHAPE_CAST1]], [[SHAPE_CAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x16x32xf16>, tensor<1x1024x1x32xf16> -> tensor<1x1024x16x32xf16>
    // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1024, 1, 16, 32]} inputs([[MULTIPLY]] : tensor<1x1024x16x32xf16>) -> tensor<1024x1x16x32xf16>
    // CHECK: return [[SHAPE_CAST_OUT]] : tensor<1024x1x16x32xf16>
}

// -----

// CHECK-LABEL: @NotReshapeMultiplyNotBroadcastable
// CHECK-SAME: [[INPUT0:%.+]]: tensor<128x2x2x1xf16>,
// CHECK-SAME: [[INPUT1:%.+]]: tensor<128x1x2x8192xf16>
func.func @NotReshapeMultiplyNotBroadcastable(%arg0: tensor<128x2x2x1xf16>, %arg1: tensor<128x1x2x8192xf16>) -> tensor<128x2x2x8192xf16> {
    %multiply = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<128x2x2x1xf16>, tensor<128x1x2x8192xf16> -> tensor<128x2x2x8192xf16>

    return %multiply : tensor<128x2x2x8192xf16>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<128x2x2x1xf16>, tensor<128x1x2x8192xf16> -> tensor<128x2x2x8192xf16>
    // CHECK: return [[MULTIPLY]] : tensor<128x2x2x8192xf16>
}

// -----

// CHECK-LABEL: @NotReshapeMultiplyWithSmallBatchSize
// CHECK-SAME: [[INPUT0:%.+]]: tensor<4x128x2x8192xf16>,
// CHECK-SAME: [[INPUT1:%.+]]: tensor<4x128x2x8192xf16>
func.func @NotReshapeMultiplyWithSmallBatchSize(%arg0: tensor<4x128x2x8192xf16>, %arg1: tensor<4x128x2x8192xf16>) -> tensor<4x128x2x8192xf16> {
    %multiply = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x128x2x8192xf16>, tensor<4x128x2x8192xf16> -> tensor<4x128x2x8192xf16>

    return %multiply : tensor<4x128x2x8192xf16>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x128x2x8192xf16>, tensor<4x128x2x8192xf16> -> tensor<4x128x2x8192xf16>
    // CHECK: return [[MULTIPLY]] : tensor<4x128x2x8192xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @ConvertSubtract
// CHECK-SAME: [[INPUT0:%.+]]: tensor<5x16x1x1xf16>,
// CHECK-SAME: [[INPUT1:%.+]]: tensor<5x16x1x1xf16>
func.func @ConvertSubtract(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<5x16x1x1xf16>) -> tensor<5x16x1x1xf16> {
    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<5x16x1x1xf16>, tensor<5x16x1x1xf16> -> tensor<5x16x1x1xf16>
    return %0 : tensor<5x16x1x1xf16>

    // CHECK: [[IN_TRANSPOSE1:%.+]] = IE.Transpose([[INPUT0]]) {order_value = #map} : tensor<5x16x1x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[IN_TRANSPOSE2:%.+]] = IE.Transpose([[INPUT1]]) {order_value = #map} : tensor<5x16x1x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[SUB:%.+]] = IE.Subtract([[IN_TRANSPOSE1]], [[IN_TRANSPOSE2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x5x1xf16>, tensor<1x16x5x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[OUT_TRANSPOSE:%.+]] = IE.Transpose([[SUB]]) {order_value = #map} : tensor<1x16x5x1xf16> -> tensor<5x16x1x1xf16>
    // CHECK: return [[OUT_TRANSPOSE]] : tensor<5x16x1x1xf16>
}

// -----

// CHECK-LABEL: @NoChangesSubtractOpDueToNotBroadcastable
// CHECK-SAME: [[INPUT0:%.+]]: tensor<5x16x1x1xf16>,
// CHECK-SAME: [[INPUT1:%.+]]: tensor<1x16x1x1xf16>
func.func @NoChangesSubtractOpDueToNotBroadcastable(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<5x16x1x1xf16> {
    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<5x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<5x16x1x1xf16>
    return %0 : tensor<5x16x1x1xf16>

    // CHECK: [[SUB:%.+]] = IE.Subtract([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<5x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<5x16x1x1xf16>
    // CHECK: return [[SUB]] : tensor<5x16x1x1xf16>
}

// -----

// CHECK-LABEL: @NoChangesSubtractOpDueToNotBroadcastableButHWNotOne
// CHECK-SAME: [[INPUT0:%.+]]: tensor<2x3x576x576xf16>,
// CHECK-SAME: [[INPUT1:%.+]]: tensor<2x1x576x576xf16>
func.func @NoChangesSubtractOpDueToNotBroadcastableButHWNotOne(%arg0: tensor<2x3x576x576xf16>, %arg1: tensor<2x1x576x576xf16>) -> tensor<2x3x576x576xf16> {
    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x576x576xf16>, tensor<2x1x576x576xf16> -> tensor<2x3x576x576xf16>
    return %0 : tensor<2x3x576x576xf16>

    // CHECK: [[SUB:%.+]] = IE.Subtract([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x576x576xf16>, tensor<2x1x576x576xf16> -> tensor<2x3x576x576xf16>
    // CHECK: return [[SUB]] : tensor<2x3x576x576xf16>
}

// -----

// CHECK-LABEL: @ConvertSubtractWithHW
// CHECK-SAME: [[INPUT0:%.+]]: tensor<50x16x10x10xf16>,
// CHECK-SAME: [[INPUT1:%.+]]: tensor<50x16x10x10xf16>
func.func @ConvertSubtractWithHW(%arg0: tensor<50x16x10x10xf16>, %arg1: tensor<50x16x10x10xf16>) -> tensor<50x16x10x10xf16> {
    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<50x16x10x10xf16>, tensor<50x16x10x10xf16> -> tensor<50x16x10x10xf16>
    return %0 : tensor<50x16x10x10xf16>

    // CHECK: [[SHAPE_CAST_IN1:%.+]] = IE.ShapeCast {shape = [1, 800, 10, 10]} inputs([[INPUT0]] : tensor<50x16x10x10xf16>) -> tensor<1x800x10x10xf16>
    // CHECK: [[SHAPE_CAST_IN2:%.+]] = IE.ShapeCast {shape = [1, 800, 10, 10]} inputs([[INPUT1]] : tensor<50x16x10x10xf16>) -> tensor<1x800x10x10xf16>
    // CHECK: [[SUB:%.+]] = IE.Subtract([[SHAPE_CAST_IN1]], [[SHAPE_CAST_IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x800x10x10xf16>, tensor<1x800x10x10xf16> -> tensor<1x800x10x10xf16>
    // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [50, 16, 10, 10]} inputs([[SUB]] : tensor<1x800x10x10xf16>) -> tensor<50x16x10x10xf16>
    // CHECK: return [[SHAPE_CAST_OUT]] : tensor<50x16x10x10xf16>
}

// -----

// CHECK-LABEL: @ConvertSubtractWithHWWithDiffInputShape
// CHECK-SAME: [[INPUT0:%.+]]: tensor<50x16x10x10xf16>,
// CHECK-SAME: [[INPUT1:%.+]]: tensor<50x16x1x1xf16>
func.func @ConvertSubtractWithHWWithDiffInputShape(%arg0: tensor<50x16x10x10xf16>, %arg1: tensor<50x16x1x1xf16>) -> tensor<50x16x10x10xf16> {
    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<50x16x10x10xf16>, tensor<50x16x1x1xf16> -> tensor<50x16x10x10xf16>
    return %0 : tensor<50x16x10x10xf16>

    // CHECK: [[SHAPE_CAST_IN1:%.+]] = IE.ShapeCast {shape = [1, 800, 10, 10]} inputs([[INPUT0]] : tensor<50x16x10x10xf16>) -> tensor<1x800x10x10xf16>
    // CHECK: [[SHAPE_CAST_IN2:%.+]] = IE.ShapeCast {shape = [1, 800, 1, 1]} inputs([[INPUT1]] : tensor<50x16x1x1xf16>) -> tensor<1x800x1x1xf16>
    // CHECK: [[SUB:%.+]] = IE.Subtract([[SHAPE_CAST_IN1]], [[SHAPE_CAST_IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x800x10x10xf16>, tensor<1x800x1x1xf16> -> tensor<1x800x10x10xf16>
    // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [50, 16, 10, 10]} inputs([[SUB]] : tensor<1x800x10x10xf16>) -> tensor<50x16x10x10xf16>
    // CHECK: return [[SHAPE_CAST_OUT]] : tensor<50x16x10x10xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

// CHECK-LABEL: @ConvertSubtractWithQuant
// CHECK-SAME: [[INPUT:%.+]]: tensor<2x3x224x224xf16>
func.func @ConvertSubtractWithQuant(%arg0: tensor<2x3x224x224xf16>) -> tensor<2x3x224x224x!quant.uniform<u8:f16, 2.000000e+00>> {
    %0 = IE.Subtract(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<2x3x224x224xf16>, tensor<2x3x224x224xf16> -> tensor<2x3x224x224x!quant.uniform<u8:f16, 2.000000e+00>>
    return %0 : tensor<2x3x224x224x!quant.uniform<u8:f16, 2.000000e+00>>
    // CHECK: [[SHAPE_CAST_IN1:%.+]] = IE.ShapeCast {shape = [1, 6, 224, 224]} inputs([[INPUT]] : tensor<2x3x224x224xf16>) -> tensor<1x6x224x224xf16>
    // CHECK: [[SHAPE_CAST_IN2:%.+]] = IE.ShapeCast {shape = [1, 6, 224, 224]} inputs([[INPUT]] : tensor<2x3x224x224xf16>) -> tensor<1x6x224x224xf16>
    // CHECK: [[SUB:%.+]] = IE.Subtract([[SHAPE_CAST_IN1]], [[SHAPE_CAST_IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x6x224x224xf16>, tensor<1x6x224x224xf16> -> tensor<1x6x224x224x!qElemType>
    // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [2, 3, 224, 224]} inputs([[SUB]] : tensor<1x6x224x224x!qElemType>) -> tensor<2x3x224x224x!qElemType>
    // CHECK: return [[SHAPE_CAST_OUT]] : tensor<2x3x224x224x!qElemType>
}

// -----

// CHECK-LABEL: @BatchConvertTo1N
// CHECK-SAME: [[INPUT:%.+]]: tensor<2x64x38x96xf16>
func.func @BatchConvertTo1N(%arg0: tensor<2x64x38x96xf16>) -> tensor<2x64x38x96xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<2x64x38x96xf16> -> tensor<2x64x38x96xf32>

    return %0 : tensor<2x64x38x96xf32>

    // CHECK: [[IN_AFFINERESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-LITERAL: {dim_mapping = [[0, 1], [1], [2], [3]], shape_value = [1, 128, 38, 96]} : tensor<2x64x38x96xf16> -> tensor<1x128x38x96xf16>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[IN_AFFINERESHAPE]]) {dstElemType = f32} : tensor<1x128x38x96xf16> -> tensor<1x128x38x96xf32>
    // CHECK: [[OUT_AFFINERESHAPE:%.+]] = IE.AffineReshape([[CONVERT]])
    // CHECK-LITERAL: {dim_mapping = [[0], [0, 1], [2], [3]], shape_value = [2, 64, 38, 96]} : tensor<1x128x38x96xf32> -> tensor<2x64x38x96xf32>

    // CHECK: return [[OUT_AFFINERESHAPE]] : tensor<2x64x38x96xf32>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK: !qElemType = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @MixedPrecisionF32Case
// CHECK-SAME: [[INPUT:%.+]]: tensor<5x16x1x1x!qElemType>
// CHECK-SAME: [[WEIGHTS:%.+]]: tensor<4x16x1x1x!qElemType1>
func.func @MixedPrecisionF32Case(%arg0: tensor<5x16x1x1x!qElemType>, %arg1: tensor<4x16x1x1x!qElemType1>) -> tensor<5x4x1x1xf32> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x16x1x1x!qElemType>, tensor<4x16x1x1x!qElemType1> -> tensor<5x4x1x1xf32>
    return %0 : tensor<5x4x1x1xf32>

    // CHECK: [[IN_TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #map} : tensor<5x16x1x1x!qElemType> -> tensor<1x16x5x1x!qElemType>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[IN_TRANSPOSE]], [[WEIGHTS]])
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:   tensor<1x16x5x1x!qElemType>, tensor<4x16x1x1x!qElemType1> -> tensor<1x4x5x1xf32>
    // CHECK: [[OUT_TRANSPOSE:%.+]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x4x5x1xf32> -> tensor<5x4x1x1xf32>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<5x4x1x1xf32>
}

// -----

// CHECK-LABEL: @ConvertConvToGroupConv
// CHECK-SAME: [[INPUT:%.+]]: tensor<64x1x516x516xf16>
func.func @ConvertConvToGroupConv(%arg0: tensor<64x1x516x516xf16>) -> tensor<64x1x513x513xf16> {
    %cst = const.Declare tensor<1x1x4x4xf16> = dense<1.0> : tensor<1x1x4x4xf16>
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<64x1x516x516xf16>, tensor<1x1x4x4xf16> -> tensor<64x1x513x513xf16>
    return %0 : tensor<64x1x513x513xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<64x1x4x4xf16> = dense<1.000000e+00> : tensor<1x1x4x4xf16>, [#const.Broadcast<0 : i64, 64 : i64>]
    // CHECK: [[INPUT_RESHAPE:%.+]] = IE.ShapeCast {shape = [1, 64, 516, 516]} inputs([[INPUT]] : tensor<64x1x516x516xf16>) -> tensor<1x64x516x516xf16>
    // CHECK: [[GROUP_CONV:%.+]] = IE.GroupConvolution([[INPUT_RESHAPE]], [[CST]]) {dilations = [1, 1], groups = 64 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x516x516xf16>, tensor<64x1x4x4xf16> -> tensor<1x64x513x513xf16>
    // CHECK: [[OUTPUT_RESHAPE:%.+]] = IE.ShapeCast {shape = [64, 1, 513, 513]} inputs([[GROUP_CONV]] : tensor<1x64x513x513xf16>) -> tensor<64x1x513x513xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<64x1x513x513xf16>
}

// -----

// CHECK-LABEL: @ConvertConvToGroupConvWithBias
// CHECK-SAME: [[INPUT:%.+]]: tensor<32x1x256x256xf16>
func.func @ConvertConvToGroupConvWithBias(%arg0: tensor<32x1x256x256xf16>) -> tensor<32x1x255x255xf16> {
    %cst = const.Declare tensor<1x1x2x2xf16> = dense<0.5> : tensor<1x1x2x2xf16>
    %bias = const.Declare tensor<1x1x1x1xf16> = dense<0.1> : tensor<1x1x1x1xf16>
    %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<32x1x256x256xf16>, tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<32x1x255x255xf16>
    return %0 : tensor<32x1x255x255xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<9.997550e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<32x1x2x2xf16> = dense<5.000000e-01> : tensor<1x1x2x2xf16>, [#const.Broadcast<0 : i64, 32 : i64>]
    // CHECK: [[INPUT_RESHAPE:%.+]] = IE.ShapeCast {shape = [1, 32, 256, 256]} inputs([[INPUT]] : tensor<32x1x256x256xf16>) -> tensor<1x32x256x256xf16>
    // CHECK: [[GROUP_CONV:%.+]] = IE.GroupConvolution([[INPUT_RESHAPE]], [[CST_0]], [[CST]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x256x256xf16>, tensor<32x1x2x2xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x255x255xf16>
    // CHECK: [[OUTPUT_RESHAPE:%.+]] = IE.ShapeCast {shape = [32, 1, 255, 255]} inputs([[GROUP_CONV]] : tensor<1x32x255x255xf16>) -> tensor<32x1x255x255xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<32x1x255x255xf16>
}

// -----

// CHECK-LABEL: @NoConvertConvToGroupConvWithMultipleChannels
// CHECK-SAME: [[INPUT:%.+]]: tensor<64x3x516x516xf16>
func.func @NoConvertConvToGroupConvWithMultipleChannels(%arg0: tensor<64x3x516x516xf16>) -> tensor<64x1x513x513xf16> {
    %cst = const.Declare tensor<1x3x4x4xf16> = dense<1.0> : tensor<1x3x4x4xf16>
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<64x3x516x516xf16>, tensor<1x3x4x4xf16> -> tensor<64x1x513x513xf16>
    return %0 : tensor<64x1x513x513xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x3x4x4xf16> = dense<1.000000e+00> : tensor<1x3x4x4xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<64x3x516x516xf16>, tensor<1x3x4x4xf16> -> tensor<64x1x513x513xf16>
    // CHECK: return [[CONV]] : tensor<64x1x513x513xf16>
}

// -----

// CHECK-LABEL: @NoConvertConvToGroupConvWithSingleBatch
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x516x516xf16>
func.func @NoConvertConvToGroupConvWithSingleBatch(%arg0: tensor<1x1x516x516xf16>) -> tensor<1x1x513x513xf16> {
    %cst = const.Declare tensor<1x1x4x4xf16> = dense<1.0> : tensor<1x1x4x4xf16>
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x516x516xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x513x513xf16>
    return %0 : tensor<1x1x513x513xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x4x4xf16> = dense<1.000000e+00> : tensor<1x1x4x4xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x516x516xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x513x513xf16>
    // CHECK: return [[CONV]] : tensor<1x1x513x513xf16>
}

// -----

// CHECK-LABEL: @ConvertConvToGroupConvWithPostOp
// CHECK-SAME: [[INPUT:%.+]]: tensor<128x1x64x64xf16>
func.func @ConvertConvToGroupConvWithPostOp(%arg0: tensor<128x1x64x64xf16>) -> tensor<128x1x62x62xf16> {
    %cst = const.Declare tensor<1x1x3x3xf16> = dense<0.25> : tensor<1x1x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.1 : f64>, strides = [1, 1]} : tensor<128x1x64x64xf16>, tensor<1x1x3x3xf16> -> tensor<128x1x62x62xf16>
    return %0 : tensor<128x1x62x62xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<128x1x3x3xf16> = dense<2.500000e-01> : tensor<1x1x3x3xf16>, [#const.Broadcast<0 : i64, 128 : i64>]
    // CHECK: [[INPUT_RESHAPE:%.+]] = IE.ShapeCast {shape = [1, 128, 64, 64]} inputs([[INPUT]] : tensor<128x1x64x64xf16>) -> tensor<1x128x64x64xf16>
    // CHECK: [[GROUP_CONV:%.+]] = IE.GroupConvolution([[INPUT_RESHAPE]], [[CST]]) {dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [1, 1]} : tensor<1x128x64x64xf16>, tensor<128x1x3x3xf16> -> tensor<1x128x62x62xf16>
    // CHECK: [[OUTPUT_RESHAPE:%.+]] = IE.ShapeCast {shape = [128, 1, 62, 62]} inputs([[GROUP_CONV]] : tensor<1x128x62x62xf16>) -> tensor<128x1x62x62xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<128x1x62x62xf16>
}
