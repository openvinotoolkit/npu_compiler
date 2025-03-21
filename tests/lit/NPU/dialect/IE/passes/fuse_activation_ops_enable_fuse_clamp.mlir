//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --fuse-activation-ops="enable-fuse-clamp=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
func.func @Conv2dWithClampTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Clamp(%0)
        {
            max = 6.000000e+00,
            min = 0.000000e+00
        } :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:  {clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:   dilations = [1, 1],
    // CHECK-SAME:   pads_begin = [0, 0],
    // CHECK-SAME:   pads_end = [0, 0],
    // CHECK-SAME:   strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>
    // CHECK-NOT:   IE.Clamp
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:127>
!qElemType1 = !quant.uniform<u8<0:254>:f16, 1.0>
!qElemType2 = !quant.uniform<u8:f16, 0.15748031466614967:128>

func.func @QuantizedConv2dWithClampTest(%arg0: tensor<1x16x20x20x!qElemType>) -> tensor<1x32x20x20x!qElemType2> {
    %filters = const.Declare tensor<32x16x1x1x!qElemType1> = dense<1.0> : tensor<32x16x1x1xf32>,
                    [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]

    %0 = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } :
        tensor<1x16x20x20x!qElemType>, tensor<32x16x1x1x!qElemType1> -> tensor<1x32x20x20x!qElemType2>

    %1 = IE.Clamp(%0)
        {
            max = 5.000000e+00 : f64,
            min = -5.000000e+00 : f64
        } :
        tensor<1x32x20x20x!qElemType2> -> tensor<1x32x20x20x!qElemType2>

    return %1 : tensor<1x32x20x20x!qElemType2>

    // CHECK:       IE.Convolution
    // CHECK-SAME:  {clamp = {max = 5.000000e+00 : f64, min = -5.000000e+00 : f64},
    // CHECK-SAME:   dilations = [1, 1],
    // CHECK-SAME:   pads_begin = [0, 0],
    // CHECK-SAME:   pads_end = [0, 0],
    // CHECK-SAME:   strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x20x20x!qElemType>, tensor<32x16x1x1x!qElemType2> -> tensor<1x32x20x20x!qElemType1>
    // CHECK-NOT:   IE.Clamp
}

// -----

func.func @Conv2dWithMultipleClampsTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Clamp(%0)
        {
            max = 6.000000e+00,
            min = 0.000000e+00
        } :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %2 = IE.Clamp(%1)
        {
            max = 4.000000e+00,
            min = 0.000000e+00
        } :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %3 = IE.Clamp(%2)
        {
            max = 5.000000e+00,
            min = 0.000000e+00
        } :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:  {clamp = {max = 4.000000e+00 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:   dilations = [1, 1],
    // CHECK-SAME:   pads_begin = [0, 0],
    // CHECK-SAME:   pads_end = [0, 0],
    // CHECK-SAME:   strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>
    // CHECK-NOT:   IE.Clamp
}

// -----

// CHECK-LABEL: @Conv2dWithReluAndClamp
func.func @Conv2dWithReluAndClamp(%arg0: tensor<4x512x1x1xf16>) -> tensor<4x2048x1x1xf16> {
    %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    %1 = IE.ReLU(%0) : tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    %2 = IE.Clamp(%1) {max = 0.700000e+00 : f64, min = 0.000000e+00 : f64} : tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>

    return %2 : tensor<4x2048x1x1xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    // CHECK:       [[CONV:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:   clamp = {max = 0.69999999999999996 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:   dilations = [1, 1],
    // CHECK-SAME:   pads_begin = [0, 0],
    // CHECK-SAME:   pads_end = [0, 0],
    // CHECK-SAME:   post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>,
    // CHECK-SAME:   strides = [1, 1]
    // CHECK-SAME:   } : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK:        return [[CONV]] : tensor<4x2048x1x1xf16>
}
