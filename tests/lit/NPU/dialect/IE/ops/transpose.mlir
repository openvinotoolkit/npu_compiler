//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK: #NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @ConvertConstToAttr
func.func @ConvertConstToAttr(%arg0: tensor<1x16x200x300xf32>) -> tensor<1x16x300x200xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<[0, 1, 3, 2]> : tensor<4xsi64>
    %1 = IE.Transpose(%arg0, %0) :
        tensor<1x16x200x300xf32>, tensor<4xsi64> -> tensor<1x16x300x200xf32>
    return %1 : tensor<1x16x300x200xf32>

    // CHECK:       %[[VAL0:.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x16x200x300xf32> -> tensor<1x16x300x200xf32>
    // CHECK:       return %[[VAL0]] : tensor<1x16x300x200xf32>
}

// -----

// CHECK: #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @FuseTransposes(%arg0: tensor<1x16x2x3xf32>) -> tensor<1x3x16x2xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<[0, 1, 3, 2]> : tensor<4xsi64>
    %1 = IE.Transpose(%arg0, %0) :
        tensor<1x16x2x3xf32>, tensor<4xsi64> -> tensor<1x16x3x2xf32>
    %2 = const.Declare tensor<4xsi64> = dense<[0, 2, 1, 3]> : tensor<4xsi64>
    %3 = IE.Transpose(%1, %2) :
        tensor<1x16x3x2xf32>, tensor<4xsi64> -> tensor<1x3x16x2xf32>
    return %3 : tensor<1x3x16x2xf32>

    // CHECK-NOT: const.Declare
    // CHECK-NOT: IE.Transpose
    // CHECK-NOT: const.Declare
    // CHECK:     [[VAL0:%.*]] = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>
    // CHECK:     return [[VAL0]] : tensor<1x3x16x2xf32>
}

// -----

func.func @FoldConstTranspose() -> tensor<40x512xf32> {
    %weights = const.Declare tensor<512x40xf32> = dense<1.0> : tensor<512x40xf32>
    %order = const.Declare tensor<2xsi64> = dense<[1, 0]> : tensor<2xsi64>
    %1 = IE.Transpose(%weights, %order) : tensor<512x40xf32>, tensor<2xsi64> -> tensor<40x512xf32>

    return %1 : tensor<40x512xf32>

    // CHECK-NOT:  IE.Transpose
    // CHECK-DAG:      [[VAL0:%.*]] = const.Declare tensor<40x512xf32>
    // CHECK-SAME:     dense<1.000000e+00> : tensor<512x40xf32>, [#const.Transpose<#CN>]
    // CHECK:      return [[VAL0]] : tensor<40x512xf32>

}

// -----

!qElemType = !quant.uniform<u8:f16:2, {0.0016649433210784313, 0.0026649433210784313, 0.0036649433210784313, 0.0046649433210784313}>
!qElemType1 = !quant.uniform<u8:f16:3, {0.0016649433210784313, 0.0026649433210784313, 0.0036649433210784313, 0.0046649433210784313}>

func.func @FoldConstTransposeWithQuantizeDimChanged() -> tensor<1x1x4x128x!qElemType> {
    %0 = const.Declare tensor<1x1x128x4x!qElemType1> = dense<1.0> : tensor<1x1x128x4xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
    %1 = const.Declare tensor<4xsi64> = dense<[0, 1, 3, 2]> : tensor<4xsi64>
    %2= IE.Transpose(%0, %1) : tensor<1x1x128x4x!qElemType1>, tensor<4xsi64> -> tensor<1x1x4x128x!qElemType>

    return %2 : tensor<1x1x4x128x!qElemType>

    // CHECK-DAG:      [[VAL0:%.*]] = const.Declare tensor<1x1x4x128x!qElemType>
    // CHECK-SAME:     dense<1.000000e+00> : tensor<1x1x128x4xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Transpose<#NCWH>]
    // CHECK:      return [[VAL0]] : tensor<1x1x4x128x!qElemType>
    // CHECK-NOT:  IE.Transpose
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func.func @CollapseTransposes(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x70x28xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 28, 70]} : tensor<1x1x28x70xf16> -> tensor<1x28x70xf16>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<1x28x70xf16> -> tensor<1x70x28xf16>

    return %2 : tensor<1x70x28xf16>

    // CHECK-NOT:           IE.Transpose
    // CHECK:               %[[SQUEEZE:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:      shape_value = [1, 70, 28]
    // CHECK-SAME:          } : tensor<1x70x1x28xf16> -> tensor<1x70x28xf16>

    // CHECK:       return %[[SQUEEZE]] : tensor<1x70x28xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func.func @DoNotCollapseNonTrivialTransposes(%arg0: tensor<1x70x2x28xf16>) -> tensor<1x70x56xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x2x28xf16> -> tensor<1x2x28x70xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 56, 70]} : tensor<1x2x28x70xf16> -> tensor<1x56x70xf16>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<1x56x70xf16> -> tensor<1x70x56xf16>

    return %2 : tensor<1x70x56xf16>

    // CHECK:               %[[FIRST_TRANSPOSE:.+]] = IE.Transpose(%arg0) {order_value = #NHWC}
    // CHECK-SAME:              : tensor<1x70x2x28xf16> -> tensor<1x2x28x70xf16>

    // CHECK:               %[[RESHAPE:.+]] = IE.AffineReshape(%[[FIRST_TRANSPOSE]]) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [0], [1], [2]],
    // CHECK-SAME{LITERAL}:      shape_value = [1, 56, 70]
    // CHECK-SAME:          } : tensor<1x2x28x70xf16> -> tensor<1x56x70xf16>

    // CHECK:               %[[LAST_TRANSPOSE:.+]] = IE.Transpose(%[[RESHAPE]]) {order_value = #map}
    // CHECK-SAME:              : tensor<1x56x70xf16> -> tensor<1x70x56xf16>

    // CHECK:       return %[[LAST_TRANSPOSE]] : tensor<1x70x56xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func.func @CollapseTransposesWithSqueeze(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x70x28xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.Squeeze(%0) { axes_value = [0] } : tensor<1x1x28x70xf16> -> tensor<1x28x70xf16>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<1x28x70xf16> -> tensor<1x70x28xf16>

    return %2 : tensor<1x70x28xf16>

    // CHECK-NOT:           IE.Transpose
    // CHECK:               %[[SQUEEZE:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:      shape_value = [1, 70, 28]
    // CHECK-SAME:          } : tensor<1x70x1x28xf16> -> tensor<1x70x28xf16>

    // CHECK:       return %[[SQUEEZE]] : tensor<1x70x28xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func.func @CollapseTransposesWithReshape(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x70x28xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.Reshape(%0) { shape_value = [1, 28, 70] } : tensor<1x1x28x70xf16> -> tensor<1x28x70xf16>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<1x28x70xf16> -> tensor<1x70x28xf16>

    return %2 : tensor<1x70x28xf16>

    // CHECK-NOT:           IE.Transpose
    // CHECK:               %[[SQUEEZE:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:      shape_value = [1, 70, 28]
    // CHECK-SAME:          } : tensor<1x70x1x28xf16> -> tensor<1x70x28xf16>

    // CHECK:       return %[[SQUEEZE]] : tensor<1x70x28xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func.func @CollapseTransposesWithUnsqueeze(%arg0: tensor<1x28x70xf16>) -> tensor<1x1x28x70xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x28x70xf16> -> tensor<1x70x28xf16>
    %1 = IE.Unsqueeze(%0) {axes_value = [0]} : tensor<1x70x28xf16> -> tensor<1x1x70x28xf16>
    %2 = IE.Transpose(%1) {order_value = #NCWH} : tensor<1x1x70x28xf16> -> tensor<1x1x28x70xf16>

    return %2 : tensor<1x1x28x70xf16>

    // CHECK-NOT:           IE.Transpose
    // CHECK:               %[[SQUEEZE:.+]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME{LITERAL}:      dim_mapping = [[0, 1], [2], [3]],
    // CHECK-SAME{LITERAL}:      shape_value = [1, 1, 28, 70]
    // CHECK-SAME:          } : tensor<1x28x70xf16> -> tensor<1x1x28x70xf16>

    // CHECK:       return %[[SQUEEZE]] : tensor<1x1x28x70xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @FoldTrivial4dTranspose(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x32x64xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NCHW} : tensor<1x16x32x64xf16> -> tensor<1x16x32x64xf16>

    return %0 : tensor<1x16x32x64xf16>

    // CHECK:       return %arg0 : tensor<1x16x32x64xf16>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @FoldTrivial3dTranspose(%arg0: tensor<1x16x32xf16>) -> tensor<1x16x32xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x16x32xf16> -> tensor<1x16x32xf16>

    return %0 : tensor<1x16x32xf16>

    // CHECK:       return %arg0 : tensor<1x16x32xf16>
}

// -----

#CN = affine_map<(d0, d1) -> (d0, d1)>

func.func @FoldTrivial2dTranspose(%arg0: tensor<1x16xf16>) -> tensor<1x16xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #CN} : tensor<1x16xf16> -> tensor<1x16xf16>

    return %0 : tensor<1x16xf16>

    // CHECK:       return %arg0 : tensor<1x16xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @DoNotFold4dTransposeWithSameShape(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:   return [[VAL0]] : tensor<1x16x32x32xf16>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

func.func @DoNotFold3dTransposeWithSameShape(%arg0: tensor<1x32x32xf16>) -> tensor<1x32x32xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x32x32xf16> -> tensor<1x32x32xf16>

    return %0 : tensor<1x32x32xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<1x32x32xf16> -> tensor<1x32x32xf16>
    // CHECK:   return [[VAL0]] : tensor<1x32x32xf16>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @DoNotFold2dTransposeWithSameShape(%arg0: tensor<32x32xf16>) -> tensor<32x32xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #CN} : tensor<32x32xf16> -> tensor<32x32xf16>

    return %0 : tensor<32x32xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Transpose(%arg0) {order_value = #CN} : tensor<32x32xf16> -> tensor<32x32xf16>
    // CHECK:   return [[VAL0]] : tensor<32x32xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @ConvertTrivialTransposeToReshape(%arg0: tensor<1x160x1x55xf16>) -> tensor<1x160x55x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x160x1x55xf16> -> tensor<1x160x55x1xf16>

    return %0 : tensor<1x160x55x1xf16>

    // CHECK-NOT:   IE.Transpose
    // CHECK:       IE.AffineReshape
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 160, 55, 1]}
    // CHECK-SAME:              tensor<1x160x1x55xf16> -> tensor<1x160x55x1xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

func.func @FuseTrivialTransposeAndReshape(%arg0: tensor<1x1024x1x1xf16>) -> tensor<1x1x1024x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 1024]} : tensor<1x1024x1x1xf16> -> tensor<1x1x1x1024xf16>
    %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x1x1x1024xf16> -> tensor<1x1x1024x1xf16>

    return %1 : tensor<1x1x1024x1xf16>

    // CHECK-NOT:   IE.Transpose
    // CHECK:       IE.AffineReshape
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 1024, 1]}
    // CHECK-SAME:              tensor<1x1024x1x1xf16> -> tensor<1x1x1024x1xf16>
}
