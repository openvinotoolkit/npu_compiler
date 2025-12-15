//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --low-precision %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 0.99607843137254903:127>
!qElemType1 = !quant.uniform<i8:f16, 0.99607843137254903:-1>
!qElemType2 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType3 = !quant.uniform<u8:f16, 2.000000e+00>

// CHECK-LABEL: @PropagateDequantizeTwiceToFuseMul
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<4096x320x1x1xf16>,
// CHECK-SAME:       [[WEIGHTS1:%.+]]: tensor<320x320x1x1xf16>)
func.func @PropagateDequantizeTwiceToFuseMul(
        %input: tensor<4096x320x1x1xf16>,
        %weights1: tensor<320x320x1x1xf16>) -> tensor<4096x4096x1x1xf16> {
    %conv1 = IE.Convolution(%input, %weights1)
        {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        tensor<4096x320x1x1xf16>, tensor<320x320x1x1xf16> -> tensor<4096x320x1x1xf16>

    %input_low = const.Declare tensor<f16> = dense<0.0> : tensor<f16>
    %input_high = const.Declare tensor<f16> = dense<255.0> : tensor<f16>
    %input_fq = IE.FakeQuantize(%conv1, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<4096x320x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<4096x320x1x1xf16>

    %reshape = IE.Reshape(%input_fq) {shape_value = [1, 4096, 8, 40]} : tensor<4096x320x1x1xf16> -> tensor<1x4096x8x40xf16>

    %transpose = IE.Transpose(%reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} :
        tensor<1x4096x8x40xf16> -> tensor<1x8x4096x40xf16>

    %groupconv_weights = const.Declare tensor<8x1x1x1xf16> = dense<2.0> : tensor<8x1x1x1xf16>

    %mul = IE.GroupConvolution(%transpose, %groupconv_weights)
        {dilations = [1, 1], groups = 8 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        tensor<1x8x4096x40xf16>, tensor<8x1x1x1xf16> -> tensor<1x8x4096x40xf16>

    %slice = IE.Slice %mul [0, 0, 0, 0] [1, 1, 4096, 40] : tensor<1x8x4096x40xf16> to tensor<1x1x4096x40xf16>

    %affine_reshape = IE.AffineReshape(%slice) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4096, 40, 1, 1]} :
        tensor<1x1x4096x40xf16> -> tensor<4096x40x1x1xf16>

    %weights2 = const.Declare tensor<4096x40x1x1xf16> = dense<10.0> : tensor<4096x40x1x1xf16>
    %weights_low = const.Declare tensor<4096x1x1x1xf16> = dense<-127.0> : tensor<4096x1x1x1xf16>
    %weights_high = const.Declare tensor<4096x1x1x1xf16> = dense<127.0> : tensor<4096x1x1x1xf16>
    %weights_fq = IE.FakeQuantize(%weights2, %weights_low, %weights_high, %weights_low, %weights_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<4096x40x1x1xf16>, tensor<4096x1x1x1xf16>, tensor<4096x1x1x1xf16>, tensor<4096x1x1x1xf16>, tensor<4096x1x1x1xf16> -> tensor<4096x40x1x1xf16>

    %conv2 = IE.Convolution(%affine_reshape, %weights_fq)
        {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        tensor<4096x40x1x1xf16>, tensor<4096x40x1x1xf16> -> tensor<4096x4096x1x1xf16>

    return %conv2 : tensor<4096x4096x1x1xf16>

    // CHECK-DAG: [[WEIGHTS2:%.+]] = const.Declare tensor<4096x40x1x1x!qElemType> = dense<1.000000e+01> :
    // CHECK-SAME:  tensor<4096x40x1x1xf16>, [#const.Quantize<!qElemType1>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>]

    // CHECK:     [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS1]])
    // CHECK-SAME:  tensor<4096x320x1x1xf16>, tensor<320x320x1x1xf16> -> tensor<4096x320x1x1x!qElemType2>

    // CHECK:     [[RESHAPE:%.+]] = IE.Reshape([[CONV1]])

    // CHECK:     [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])

    // CHECK:     [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[TRANSPOSE]]) {dstElemType = !qElemType3} :
    // CHECK-SAME:  tensor<1x8x4096x40x!qElemType2> -> tensor<1x8x4096x40x!qElemType3>

    // CHECK:     [[SLICE:%.+]] = IE.Slice [[QUANTIZECAST]]

    // CHECK:     [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[SLICE]])

    // CHECK:     [[CONV2:%.+]] = IE.Convolution([[AFFINE_RESHAPE]], [[WEIGHTS2]])
    // CHECK-SAME:  tensor<4096x40x1x1x!qElemType3>, tensor<4096x40x1x1x!qElemType> -> tensor<4096x4096x1x1xf16>

    // CHECK:     return [[CONV2]]
}
