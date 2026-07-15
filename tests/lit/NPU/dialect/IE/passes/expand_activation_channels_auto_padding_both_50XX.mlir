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
