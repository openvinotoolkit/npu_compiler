//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true enable-auto-padding-odu=true enable-is-reduce-supported=true enable-dcim=true power-optimized=false" --mlir-elide-elementsattrs-if-larger 64 --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @AvgPoolToNCE(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}> {
    %ave_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x6x6xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>

    return %ave_pool : tensor<1x16x4x4xf16, {order = #NHWC}>

    // CHECK:         [[OUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [3, 3]
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:        ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:           clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:           clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:           scale = 0.1111111111111111 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

    // CHECK:           return [[OUT]] : tensor<1x16x4x4xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.049356617647058822>
!qElemType1 = !quant.uniform<u8:f16, 0.01013327205882353>
!qElemType2 = !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddWithDifferentScales
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28x!qElemType, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
func.func @EltwiseAddWithDifferentScales(%arg0: tensor<1x64x28x28x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28x!qElemType, {order = #NHWC}>, tensor<1x64x28x28x!qElemType1, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,

    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64
    // CHECK-SAME:          scale = 3.5798177123069763E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64
    // CHECK-SAME:          in1_mult = [2.587700e+04], in2_mult = [5.312000e+03]>
    // CHECK-SAME:      } -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>
}
