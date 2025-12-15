//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-dynamic-quant-to-VPU-NCE %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i4:f16, 1.0:8>

// CHECK-LABEL: @ConvToNCEWeightsAsInputs
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<16x16x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<16x1x1x1xf16, {order = #NHWC}>
func.func @ConvToNCEWeightsAsInputs(%input: tensor<1x16x16x16xf16, {order = #NHWC}>,
                        %weights: tensor<16x16x1x1x!qElemType, {order = #NHWC}>,
                        %scale: tensor<16x1x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %zp = const.Declare tensor<1x16x1x1xi4, {order = #NHWC}> = dense<8.0> : tensor<1x16x1x1xf16, {order = #NHWC}>,
            [#const.CastElemType<i4>]
    %dynamic_dequant = IE.DynamicDequantize(%weights, %scale, %zp) {dstElemType = f16} :
        tensor<16x16x1x1x!qElemType, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xi4, {order = #NHWC}> -> tensor<16x16x1x1xf16, {order = #NHWC}>

    %conv = IE.Convolution(%input, %dynamic_dequant) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %conv : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS]]) {shape_value = [16, 1, 1, 16]} : tensor<16x16x1x1x!qElemType, {order = #NHWC}> -> tensor<16x1x1x16x!qElemType>

    // CHECK:   [[EXPAND:%.+]] = VPU.Expand([[RESHAPE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 16]} : tensor<16x1x1x16x!qElemType> -> tensor<16x1x1x32x!qElemType>

    // CHECK:   [[LAYOUT_CAST:%.+]] = VPU.LayoutCast([[EXPAND]]) {dst_order = #NHWC} : tensor<16x1x1x32x!qElemType> -> tensor<16x1x1x32x!qElemType, {order = #NHWC}>

    // CHECK:   [[WT:%.+]] = VPU.PopulateWeightTable([[SCALE]]) {base = 0 : i64, step = 0 : i64}
    // CHECK-SAME:     : tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[LAYOUT_CAST]], [[WT]])
    // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:     clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:   rawFilterShape = [16, 16, 1, 1], strides = [1, 1]}
    // CHECK-SAME:   -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:   return [[CONV]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}
