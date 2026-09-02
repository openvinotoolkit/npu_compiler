//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% num-of-dpu-groups=1 allow-custom-values=true enable-auto-padding-odu enable-auto-padding-idu" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>
!qElemType2 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127}>
!qElemType3 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType4 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
// CHECK:  !qElemType1 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>
// CHECK:  !qElemType2 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127}>
// CHECK:  !qElemType3 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127}>
// CHECK:  !qElemType4 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127}>

// CHECK-LABEL: func.func @ExpandQuantConvolutionChannels
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30x!qElemType, {order = #NHWC}>
func.func @ExpandQuantConvolutionChannels(%input: tensor<1x3x30x30x!qElemType, {order = #NHWC}>)
            -> tensor<1x5x28x28x!qElemType1, {order = #NHWC}> {
    %filter = const.Declare tensor<5x3x3x3x!qElemType2, {order = #NHWC}> =
        dense<1.0> : tensor<5x3x3x3xf16, {order = #NHWC}>, [
        #const.CastElemType<ui8>,
        #const.CastElemType<!qElemType2>
    ]
    %1 = IE.Convolution(%input, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30x!qElemType, {order = #NHWC}>, tensor<5x3x3x3x!qElemType2, {order = #NHWC}> -> tensor<1x5x28x28x!qElemType1, {order = #NHWC}>
    return %1 : tensor<1x5x28x28x!qElemType1, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3x!qElemType2, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x30x30x!qElemType, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28x!qElemType4, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType1 = !quant.uniform<!QuantileType.quantile<ui4:si8, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16:0, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>

// CHECK: [[QQ:!.+]] = !quant.uniform<!QuantileType.quantile<ui4:si8,

// CHECK:  @NCEConvolutionWithQuantileWeights
// CHECK-SAME:    ([[IN:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
module @module {
    func.func @NCEConvolutionWithQuantileWeights(%input: tensor<1x16x1x1xf16, {order = #NHWC}>, %weights_raw: tensor<16x4x1x1xsi4>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
        // First operand side: keep exactly one producer op.
        %prod_w = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %prod = VPU.NCE.Convolution(%input, %prod_w) rawFilterShape [16, 16, 1, 1] {
            input_padding = [0, 0, 0, 0],
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            output_padding = [0, 0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>

        %cst_0 = const.Declare tensor<16x12x1x1x!qElemType1, {order = #NHWC}> = dense<0> : tensor<16x12x1x1xui8, {order = #NHWC}>

        %q4 = VPU.QuantizeCast(%weights_raw) {dstElemType = !qElemType1} : tensor<16x4x1x1xsi4> -> tensor<16x4x1x1x!qElemType1>
        %q5 = VPU.PermuteCast(%q4) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<16x4x1x1x!qElemType1> -> tensor<16x4x1x1x!qElemType1, {order = #NHWC}>
        %q6 = VPU.Concat(%q5, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]} : tensor<16x4x1x1x!qElemType1, {order = #NHWC}>, tensor<16x12x1x1x!qElemType1, {order = #NHWC}> -> tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
        %q7 = VPU.Reshape(%q6) {shape_value = [16, 1, 1, 16]} : tensor<16x16x1x1x!qElemType1, {order = #NHWC}> -> tensor<16x1x1x16x!qElemType1>
        %q8 = VPU.Expand(%q7) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 16]} : tensor<16x1x1x16x!qElemType1> -> tensor<16x1x1x32x!qElemType1>
        %q9 = VPU.LayoutCast(%q8) {dst_order = #NHWC} : tensor<16x1x1x32x!qElemType1> -> tensor<16x1x1x32x!qElemType1, {order = #NHWC}>

        %nce = VPU.NCE.Convolution(%prod, %q9) rawFilterShape [16, 16, 1, 1] {
            input_padding = [0, 12, 0, 0],
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>, weight_zp = 0 : i64>,
            output_padding = [0, 0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x32x!qElemType1, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>

        return %nce : tensor<1x16x1x1xf16, {order = #NHWC}>

        // CHECK-DAG:   [[PW:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:   [[ZC:%.+]] = const.Declare tensor<16x12x1x1x[[QQ]], {order = #NHWC}> = dense<0>

        // CHECK:       [[ACT:%.+]] = VPU.NCE.Convolution([[IN]], [[PW]])
        // CHECK-SAME:    -> tensor<1x16x1x1xf16, {order = #NHWC}>

        // CHECK:       [[PC:%.+]] = VPU.PermuteCast
        // CHECK-SAME:    -> tensor<16x4x1x1x[[QQ]], {order = #NHWC}>
        // CHECK:       [[CAT:%.+]] = VPU.Concat([[PC]], [[ZC]])
        // CHECK-SAME:    -> tensor<16x16x1x1x[[QQ]], {order = #NHWC}>
        // CHECK:       [[RSH:%.+]] = VPU.Reshape([[CAT]]) {shape_value = [16, 1, 1, 16]}
        // CHECK:       [[EXP:%.+]] = VPU.Expand([[RSH]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 16]}
        // CHECK:       [[LC:%.+]] = VPU.LayoutCast([[EXP]]) {dst_order = #NHWC}
        // CHECK-SAME:    -> tensor<16x1x1x32x[[QQ]], {order = #NHWC}>

        // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[ACT]], [[LC]])
        // CHECK-SAME:    input_padding = [0, 12, 0, 0]
        // CHECK-SAME:    output_padding = [0, 0, 0, 0]
        // CHECK:       return [[OUT]]
    }
}
