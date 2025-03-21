//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-convert-with-transpose-reshape --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SwapTransposeWithConvert(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>

    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%arg0) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xui8> -> tensor<1x1x28x70xui8>

    // CHECK:   %[[CONVERT:.*]] = IE.Convert(%[[TRANSPOSE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x1x28x70xui8> -> tensor<1x1x28x70xf16>

    // CHECK:   return %[[CONVERT]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SwapReshapeWithConvert(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>

    %1 = IE.Reshape(%0) {shape_value = [1, 1, 28, 70]} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK:   %[[RESHAPE:.*]] = IE.Reshape(%arg0) {shape_value = [1, 1, 28, 70]}
    // CHECK-SAME:  : tensor<1x70x1x28xui8> -> tensor<1x1x28x70xui8>

    // CHECK:   %[[CONVERT:.*]] = IE.Convert(%[[RESHAPE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x1x28x70xui8> -> tensor<1x1x28x70xf16>

    // CHECK:   return %[[CONVERT]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SwapAffineReshapeWithConvert(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>

    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2], [3], [3]], shape_value = [1, 1, 28, 70]} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK:   %[[AFFINERESHAPE:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL} {dim_mapping = [[0], [1], [2], [2], [3], [3]], shape_value = [1, 1, 28, 70]}
    // CHECK-SAME:  : tensor<1x70x1x28xui8> -> tensor<1x1x28x70xui8>

    // CHECK:   %[[CONVERT:.*]] = IE.Convert(%[[AFFINERESHAPE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x1x28x70xui8> -> tensor<1x1x28x70xf16>

    // CHECK:   return %[[CONVERT]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SwapSqueezeWithConvert(%arg0: tensor<1x1x70x28xui8>) -> tensor<70x28xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<1x1x70x28xui8> -> tensor<1x1x70x28xf16>

    %1 = IE.Squeeze(%0) {axes_value = [0, 1]} : tensor<1x1x70x28xf16> -> tensor<70x28xf16>
    return %1 : tensor<70x28xf16>

    // CHECK:   %[[SQUEEZE:.*]] = IE.Squeeze(%arg0) {axes_value = [0, 1]}
    // CHECK-SAME:  : tensor<1x1x70x28xui8> -> tensor<70x28xui8>

    // CHECK:   %[[CONVERT:.*]] = IE.Convert(%[[SQUEEZE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<70x28xui8> -> tensor<70x28xf16>

    // CHECK:   return %[[CONVERT]] : tensor<70x28xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SwapUnsqueezeWithConvert(%arg0: tensor<70x28xui8>) -> tensor<1x1x70x28xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16}
        : tensor<70x28xui8> -> tensor<70x28xf16>

    %1 = IE.Unsqueeze(%0) {axes_value = [0, 1]} : tensor<70x28xf16> -> tensor<1x1x70x28xf16>
    return %1 : tensor<1x1x70x28xf16>

    // CHECK:   %[[UNSQUEEZE:.*]] = IE.Unsqueeze(%arg0) {axes_value = [0, 1]}
    // CHECK-SAME:  : tensor<70x28xui8> -> tensor<1x1x70x28xui8>

    // CHECK:   %[[CONVERT:.*]] = IE.Convert(%[[UNSQUEEZE]])
    // CHECK-SAME:  {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x1x70x28xui8> -> tensor<1x1x70x28xf16>

    // CHECK:   return %[[CONVERT]] : tensor<1x1x70x28xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.0>

func.func @DoNotSwapTransposeWithConvert(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<1x70x1x28xui8> -> tensor<1x70x1x28x!qElemType>
    %1 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x70x1x28x!qElemType>, tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28x!qElemType>
    %2 = IE.Convert(%1) {dstElemType = f16} : tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28xf16>

    %3 = IE.Transpose(%2) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %3 : tensor<1x1x28x70xf16>

    // CHECK:   %[[VAR0:.*]] = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} :
    // CHECK-SAME:     tensor<1x70x1x28xui8> -> tensor<1x70x1x28x!qElemType>

    // CHECK:   %[[ADD:.*]] = IE.Add(%[[VAR0]], %[[VAR0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:  : tensor<1x70x1x28x!qElemType>, tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28x!qElemType>

    // CHECK:   %[[CONVERT:.*]] = IE.Convert(%[[ADD]]) {dstElemType = f16} : tensor<1x70x1x28x!qElemType> -> tensor<1x70x1x28xf16>
    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%[[CONVERT]]) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    // CHECK:   return %[[TRANSPOSE]] : tensor<1x1x28x70xf16>
}
