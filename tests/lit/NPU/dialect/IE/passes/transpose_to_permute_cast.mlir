//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --transpose-to-permute-cast %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @Transpose_d0_d3_d1_d2(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x14x30xf16> {
    %cst = const.Declare tensor<16x64x3x3xf16> = dense<1.0> : tensor<16x64x3x3xf16>
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x16x32x64xf16> -> tensor<1x64x16x32xf16>
    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x64x16x32xf16>, tensor<16x64x3x3xf16> -> tensor<1x16x14x30xf16>
    return %1 : tensor<1x16x14x30xf16>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<16x64x3x3xf16>
    // CHECK: %[[PERM_CAST:.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16> -> tensor<1x64x16x32xf16, {order = #NHWC}>

    // CHECK: %[[REORDER:.*]] = IE.Reorder(%[[PERM_CAST]]) {dstOrder = #NCHW} :
    // CHECK-SAME:  tensor<1x64x16x32xf16, {order = #NHWC}> -> tensor<1x64x16x32xf16>

    // CHECK: %[[CONV:.*]] = IE.Convolution(%[[REORDER]], %[[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x64x16x32xf16>, tensor<16x64x3x3xf16> -> tensor<1x16x14x30xf16>

    // CHECK: return %[[CONV]] : tensor<1x16x14x30xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
func.func @Transpose_d0_d3_d2_d1(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x30x14xf16> {
    %cst = const.Declare tensor<16x64x3x3xf16> = dense<1.0> : tensor<16x64x3x3xf16>
    %0 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x16x32x64xf16> -> tensor<1x64x32x16xf16>
    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x64x32x16xf16>, tensor<16x64x3x3xf16> -> tensor<1x16x30x14xf16>
    return %1 : tensor<1x16x30x14xf16>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<16x64x3x3xf16>
    // CHECK: %[[PERM_CAST:.*]] = IE.PermuteCast(%arg0) {dst_order = #NWHC, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16> -> tensor<1x64x32x16xf16, {order = #NWHC}>

    // CHECK: %[[REORDER:.*]] = IE.Reorder(%[[PERM_CAST]]) {dstOrder = #NCHW} :
    // CHECK-SAME:  tensor<1x64x32x16xf16, {order = #NWHC}> -> tensor<1x64x32x16xf16>

    // CHECK: %[[CONV:.*]] = IE.Convolution(%[[REORDER]], %[[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x64x32x16xf16>, tensor<16x64x3x3xf16> -> tensor<1x16x30x14xf16>

    // CHECK: return %[[CONV]] : tensor<1x16x30x14xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
func.func @Transpose_d0_d2_d1_d3(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x14x62xf16> {
    %cst = const.Declare tensor<16x32x3x3xf16> = dense<1.0> : tensor<16x32x3x3xf16>
    %0 = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x16x32x64xf16> -> tensor<1x32x16x64xf16>
    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x32x16x64xf16>, tensor<16x32x3x3xf16> -> tensor<1x16x14x62xf16>
    return %1 : tensor<1x16x14x62xf16>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<16x32x3x3xf16> = dense<1.000000e+00> : tensor<16x32x3x3xf16>
    // CHECK: %[[PERM_CAST:.*]] = IE.PermuteCast(%arg0) {dst_order = #NHCW, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16> -> tensor<1x32x16x64xf16, {order = #NHCW}>

    // CHECK: %[[REORDER:.*]] = IE.Reorder(%[[PERM_CAST]]) {dstOrder = #NCHW} :
    // CHECK-SAME:  tensor<1x32x16x64xf16, {order = #NHCW}> -> tensor<1x32x16x64xf16>

    // CHECK: %[[CONV:.*]] = IE.Convolution(%[[REORDER]], %[[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x32x16x64xf16>, tensor<16x32x3x3xf16> -> tensor<1x16x14x62xf16>

    // CHECK: return %[[CONV]] : tensor<1x16x14x62xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
func.func @Transpose_d0_d2_d3_d1(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x62x14xf16> {
    %cst = const.Declare tensor<16x32x3x3xf16> = dense<1.0> : tensor<16x32x3x3xf16>
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x16x32x64xf16> -> tensor<1x32x64x16xf16>
    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x32x64x16xf16>, tensor<16x32x3x3xf16> -> tensor<1x16x62x14xf16>
    return %1 : tensor<1x16x62x14xf16>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<16x32x3x3xf16> = dense<1.000000e+00> : tensor<16x32x3x3xf16>
    // CHECK: %[[PERM_CAST:.*]] = IE.PermuteCast(%arg0) {dst_order = #NWCH, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16> -> tensor<1x32x64x16xf16, {order = #NWCH}>

    // CHECK: %[[REORDER:.*]] = IE.Reorder(%[[PERM_CAST]]) {dstOrder = #NCHW} :
    // CHECK-SAME:  tensor<1x32x64x16xf16, {order = #NWCH}> -> tensor<1x32x64x16xf16>

    // CHECK: %[[CONV:.*]] = IE.Convolution(%[[REORDER]], %[[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x32x64x16xf16>, tensor<16x32x3x3xf16> -> tensor<1x16x62x14xf16>

    // CHECK: return %[[CONV]] : tensor<1x16x62x14xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
func.func @Transpose_d0_d1_d3_d2(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x62x30xf16> {
    %cst = const.Declare tensor<16x16x3x3xf16> = dense<1.0> : tensor<16x16x3x3xf16>
    %0 = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x16x32x64xf16> -> tensor<1x16x64x32xf16>
    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x64x32xf16>, tensor<16x16x3x3xf16> -> tensor<1x16x62x30xf16>
    return %1 : tensor<1x16x62x30xf16>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<16x16x3x3xf16>
    // CHECK: %[[PERM_CAST:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCWH, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16> -> tensor<1x16x64x32xf16, {order = #NCWH}>

    // CHECK: %[[REORDER:.*]] = IE.Reorder(%[[PERM_CAST]]) {dstOrder = #NCHW} :
    // CHECK-SAME:  tensor<1x16x64x32xf16, {order = #NCWH}> -> tensor<1x16x64x32xf16>

    // CHECK: %[[CONV:.*]] = IE.Convolution(%[[REORDER]], %[[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x16x64x32xf16>, tensor<16x16x3x3xf16> -> tensor<1x16x62x30xf16>

    // CHECK: return %[[CONV]] : tensor<1x16x62x30xf16>
}

// -----

!QUANT_IN = !quant.uniform<u8<0:254>:f16:1, {
    0.25196850393700787:127,
    0.50393700787401574:127,
    0.75590551181102361:127,
    0.25196850393700787:127
}>

!QUANT_OUT = !quant.uniform<u8<0:254>:f16:3, {
    0.25196850393700787:127,
    0.50393700787401574:127,
    0.75590551181102361:127,
    0.25196850393700787:127
}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL-DAG: @TransposeWithPerAxisQuant
// CHECK-DAG:   [[QUANT_IN:!.*]] = !quant.uniform<u8<0:254>:f16:1
// CHECK-DAG:   [[QUANT_OUT:!.*]] = !quant.uniform<u8<0:254>:f16:3
// CHECK-DAG: ([[DATA:%arg[0-9]]]: tensor<2x4x8x16x[[QUANT_IN]]>)
func.func @TransposeWithPerAxisQuant(%arg0: tensor<2x4x8x16x!QUANT_IN>) -> tensor<2x8x16x4x!QUANT_OUT> {
    %TRANSPOSE = IE.Transpose(%arg0) {
        order_value = #NHWC
    } : tensor<2x4x8x16x!QUANT_IN> -> tensor<2x8x16x4x!QUANT_OUT>

    // CHECK:   [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[DATA]]) {
    // CHECK-SAME:      dst_order = #NWCH,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<2x4x8x16x[[QUANT_IN]]>
    // CHECK-SAME:      -> tensor<2x8x16x4x[[QUANT_OUT]], {order = #NWCH}>

    // CHECK:   [[REORDER:%.*]] = IE.Reorder([[PERMUTE_CAST]]) {
    // CHECK-SAME:      dstOrder = #NCHW
    // CHECK-SAME:  } : tensor<2x8x16x4x[[QUANT_OUT]], {order = #NWCH}>
    // CHECK-SAME:      -> tensor<2x8x16x4x[[QUANT_OUT]]>

    return %TRANSPOSE : tensor<2x8x16x4x!QUANT_OUT>
    // CHECK:   return [[REORDER]] : tensor<2x8x16x4x[[QUANT_OUT]]>
}
