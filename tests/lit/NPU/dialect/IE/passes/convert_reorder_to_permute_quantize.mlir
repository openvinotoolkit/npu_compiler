//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-reorder-to-permute-quantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertReorder
func.func @ConvertReorder(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16, {order = #NHWC}>

    return %0 : tensor<1x3x224x224xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Reorder
    // CHECK:       IE.PermuteQuantize(%arg0) {
    // CHECK-SAME:      dstElemType = f16,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SkipNHWCInput
func.func @SkipNHWCInput(%arg0: tensor<1x3x224x224xf16, {order = #NHWC}>) -> tensor<1x3x224x224xf16, {order = #NWHC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NWHC
    } : tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x3x224x224xf16, {order = #NWHC}>

    return %0 : tensor<1x3x224x224xf16, {order = #NWHC}>

    // CHECK-NOT:   IE.PermuteQuantize
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SkipNWHCOutput
func.func @SkipNWHCOutput(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16, {order = #NWHC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NWHC
    } : tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16, {order = #NWHC}>

    return %0 : tensor<1x3x224x224xf16, {order = #NWHC}>

    // CHECK-NOT:   IE.PermuteQuantize
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipU8Input
func.func @SkipU8Input(%arg0: tensor<1x3x224x224xui8>) -> tensor<1x3x224x224xui8, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x3x224x224xui8> -> tensor<1x3x224x224xui8, {order = #NHWC}>

    return %0 : tensor<1x3x224x224xui8, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipIncompatibleShape
func.func @SkipIncompatibleShape(%arg0: tensor<1x3x225x225xui8>) -> tensor<1x3x225x225xui8, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x3x225x225xui8> -> tensor<1x3x225x225xui8, {order = #NHWC}>

    return %0 : tensor<1x3x225x225xui8, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertReorderForAlignedShape
func.func @ConvertReorderForAlignedShape(%arg0: tensor<1x320x384x1xf16>) -> tensor<1x320x384x1xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x320x384x1xf16> -> tensor<1x320x384x1xf16, {order = #NHWC}>

    return %0 : tensor<1x320x384x1xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Reorder
    // CHECK:       IE.PermuteQuantize(%arg0) {
    // CHECK-SAME:      dstElemType = f16,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<1x320x384x1xf16> -> tensor<1x320x384x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipInappropriateShape
func.func @SkipInappropriateShape(%arg0: tensor<1x384x150x1xf16>) -> tensor<1x384x150x1xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x384x150x1xf16> -> tensor<1x384x150x1xf16, {order = #NHWC}>

    return %0 : tensor<1x384x150x1xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipShapeBeyondNCEInvariant
func.func @SkipShapeBeyondNCEInvariant(%arg0: tensor<1x12288x2x2xf16>) -> tensor<1x12288x2x2xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x12288x2x2xf16> -> tensor<1x12288x2x2xf16, {order = #NHWC}>

    return %0 : tensor<1x12288x2x2xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @DontConvertReorderForQuantizedAvgPool
// CHECK-SAME:        [[INPUT1:%.+]]: tensor<1x12x512x512xf16>
// CHECK-SAME:        [[INPUT2:%.+]]: tensor<1x12x512x512xf16, {order = #NHWC}>
func.func @DontConvertReorderForQuantizedAvgPool(%arg0: tensor<1x12x512x512xf16>, %arg1: tensor<1x12x512x512xf16, {order = #NHWC}>) -> tensor<1x12x512x512xf16, {order = #NHWC}> {
    %reorder = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16, {order = #NHWC}>

    %avgpool1 = IE.AvgPool(%reorder) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x12x512x512xf16, {order = #NHWC}> -> tensor<1x12x512x512x!qElemType, {order = #NHWC}>

    %avgpool2 = IE.AvgPool(%avgpool1) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x12x512x512x!qElemType, {order = #NHWC}> -> tensor<1x12x512x512xf16, {order = #NHWC}>

    %add = IE.Add(%avgpool2, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x12x512x512xf16, {order = #NHWC}>, tensor<1x12x512x512xf16, {order = #NHWC}> -> tensor<1x12x512x512xf16, {order = #NHWC}>

    return %add : tensor<1x12x512x512xf16, {order = #NHWC}>

    // Don't convert the reorder to permuteQuantize because of the quantized avgpool
    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[INPUT1]])
    // CHECK-NOT:   IE.PermuteQuantize
    // CHECK:       [[AVGPOOL1:%.+]] = IE.AvgPool([[REORDER]])
    // CHECK:       [[AVGPOOL2:%.+]] = IE.AvgPool([[AVGPOOL1]])
    // CHECK:       [[ADD:%.+]] = IE.Add([[AVGPOOL2]], [[INPUT2]])
    // CHECK:       return [[ADD]]
}
