//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-transpose-with-fq --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0024337469362745098>

func.func @SwapTransposeWithPerTensorQuant(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70x!qElemType> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType}
        : tensor<1x70x1x28xf16> -> tensor<1x70x1x28x!qElemType>

    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28x!qElemType> -> tensor<1x1x28x70x!qElemType>
    return %1 : tensor<1x1x28x70x!qElemType>

    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%arg0) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    // CHECK:   %[[FQ:.*]] = IE.Quantize(%[[TRANSPOSE]])
    // CHECK-SAME:  {dstElemType = !qElemType}
    // CHECK-SAME:  : tensor<1x1x28x70xf16> -> tensor<1x1x28x70x!qElemType>

    // CHECK:   return %[[FQ]] : tensor<1x1x28x70x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SwapTransposeWithPerTensorFQ(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70xf16> {
    %cst_lo = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %cst_hi = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %0 = IE.FakeQuantize(%arg0, %cst_lo, %cst_hi, %cst_lo, %cst_hi)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 }
        : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>

    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK-DAG:   %[[CST_HI:.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG:   %[[CST_LO:.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%arg0) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[TRANSPOSE]], %[[CST_LO]], %[[CST_HI]], %[[CST_LO]], %[[CST_HI]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK-SAME:  : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    // CHECK:   return %[[FQ]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DoNotSwapTransposeWithPerAxisFQ(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70xf16> {
    %cst_lo = const.Declare tensor<1x70x1x1xf32> = dense<0.0> : tensor<1x70x1x1xf32>
    %cst_hi = const.Declare tensor<1x70x1x1xf32> = dense<255.0> : tensor<1x70x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_lo, %cst_hi, %cst_lo, %cst_hi)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 }
        : tensor<1x70x1x28xf16>, tensor<1x70x1x1xf32>, tensor<1x70x1x1xf32>, tensor<1x70x1x1xf32>, tensor<1x70x1x1xf32> -> tensor<1x70x1x28xf16>

    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK-DAG:   %[[CST_HI:.*]] = const.Declare tensor<1x70x1x1xf32> = dense<2.550000e+02> : tensor<1x70x1x1xf32>
    // CHECK-DAG:   %[[CST_LO:.*]] = const.Declare tensor<1x70x1x1xf32> = dense<0.000000e+00> : tensor<1x70x1x1xf32>

    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%arg0, %[[CST_LO]], %[[CST_HI]], %[[CST_LO]], %[[CST_HI]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK-SAME:  : tensor<1x70x1x28xf16>, tensor<1x70x1x1xf32>, tensor<1x70x1x1xf32>, tensor<1x70x1x1xf32>, tensor<1x70x1x1xf32>
    // CHECK-SAME:  -> tensor<1x70x1x28xf16>

    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%[[FQ]]) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    // CHECK:   return %[[TRANSPOSE]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SwapConvertTransposeWithFQ(%arg0: tensor<1x70x1x28xui8>) -> tensor<1x1x28x70xf16> {
    %cst_lo = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %cst_hi = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>
    %1 = IE.FakeQuantize(%0, %cst_lo, %cst_hi, %cst_lo, %cst_hi)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 }
        : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>

    %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %3 = IE.FakeQuantize(%2, %cst_lo, %cst_hi, %cst_lo, %cst_hi)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 }
        : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    return %3 : tensor<1x1x28x70xf16>

    // CHECK-DAG:   %[[CST_HI:.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG:   %[[CST_LO:.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK:   %[[CONVERT:.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x70x1x28xui8> -> tensor<1x70x1x28xf16>
    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%[[CONVERT]]) {order_value = #NHWC}
    // CHECK-SAME:  : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[TRANSPOSE]], %[[CST_LO]], %[[CST_HI]], %[[CST_LO]], %[[CST_HI]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK-SAME:  : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    // CHECK:   return %[[FQ]] : tensor<1x1x28x70xf16>
}
