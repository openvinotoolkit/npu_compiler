//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% num-of-dpu-groups=1 enable-auto-padding-odu" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:254>:f16:0, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>
!qElemType1 = !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType2 = !quant.uniform<u8:f16, 0.012699142156862745>
// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 0.012699142156862745>
// CHECK:  !qElemType2 = !quant.uniform<u8<0:254>:f16:0, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127}>
// CHECK:  !qElemType3 = !quant.uniform<u8<0:254>:f16:0, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>

// CHECK-LABEL: @ExpandZMajorConvChannelsQuant
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x30x30x!qElemType, {order = #NHWC}>)
func.func @ExpandZMajorConvChannelsQuant(%arg0: tensor<1x3x30x30x!qElemType1, {order = #NHWC}>) -> tensor<1x5x28x28x!qElemType2, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x3x3x!qElemType, {order = #NHWC}> =
        dense<1.0> : tensor<5x3x3x3xf16>, [
            #const.Reorder<#NHWC>,
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>
    ]

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30x!qElemType1, {order = #NHWC}>, tensor<5x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x5x28x28x!qElemType2, {order = #NHWC}>

    return %1 : tensor<1x5x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x4x3x3x!qElemType2, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x4x30x30x!qElemType, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 1, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28x!qElemType1, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandZMajorConvChannelsOnlyWeightFP16
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x30x30xf16, {order = #NHWC}>)
func.func @ExpandZMajorConvChannelsOnlyWeightFP16(%arg0: tensor<1x4x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x4x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<5x4x3x3xf16>, [#const.Reorder<#NHWC>]

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x4x30x30xf16, {order = #NHWC}>, tensor<5x4x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x4x3x3xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 0, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}
