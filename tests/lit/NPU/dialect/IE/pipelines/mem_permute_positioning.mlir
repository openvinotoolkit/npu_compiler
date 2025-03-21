//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --mempermute-positioning %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MemPemutePositioning
// CHECK-SAME:      ([[INPUT:%.*]]: tensor<1x16x32x64xf16>)
func.func @MemPemutePositioning(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x64x32xi8> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x32x64xf16> -> tensor<1x16x32x64xf16, {order = #NHWC}>
    %1 = IE.MaxPool(%0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.0, min = 0.0}>
        } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16>

    %3 = IE.Convert(%2) {dstElemType = i8} : tensor<1x16x32x64xf16> -> tensor<1x16x32x64xi8>

    %4 = IE.Transpose(%3) {order_value = #NCWH} : tensor<1x16x32x64xi8> -> tensor<1x16x64x32xi8>

    return %4 : tensor<1x16x64x32xi8>

    // CHECK:   [[MEMPERMUTE0:%.*]] = IE.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x32x64xf16> -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:   [[POOL:%.*]] = IE.MaxPool([[MEMPERMUTE0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.Clamp", attrs = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:   [[CONVERT:%.*]] = IE.Convert([[POOL]]) {dstElemType = i8} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xi8, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE1:%.*]] = IE.MemPermute([[CONVERT]]) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x16x32x64xi8, {order = #NHWC}> -> tensor<1x16x64x32xi8>
    // CHECK:   return [[MEMPERMUTE1]] : tensor<1x16x64x32xi8>
}

// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateThroughAvgpoolAndAdd
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x12x512x512xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x12x512x512xf16>,
// CHECK-SAME:        [[INPUT3:%arg[0-9]]]: tensor<1x12x512x512x!qElemType>
func.func @PropagateThroughAvgpoolAndAdd(
        %arg0: tensor<1x12x512x512xf16>,
        %arg1: tensor<1x12x512x512xf16>,
        %arg2: tensor<1x12x512x512x!qElemType>) -> tensor<1x12x512x512xf16> {
    // Eltwise input branch 1
    %reorder1 = IE.Reorder(%arg0) {
        dstOrder = #NHWC
    } : tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16, {order = #NHWC}>
    %avgpool1_1 = IE.AvgPool(%reorder1) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x12x512x512xf16, {order = #NHWC}> -> tensor<1x12x512x512x!qElemType, {order = #NHWC}>
    %avgpool1_2 = IE.AvgPool(%avgpool1_1) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x12x512x512x!qElemType, {order = #NHWC}> -> tensor<1x12x512x512xf16, {order = #NHWC}>

    // Eltwise input branch 2
    %reorder2_1 = IE.Reorder(%arg1) {
        dstOrder = #NHWC
    } : tensor<1x12x512x512xf16> -> tensor<1x12x512x512xf16, {order = #NHWC}>
    %avgpool2 = IE.AvgPool(%reorder2_1) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x12x512x512xf16, {order = #NHWC}> -> tensor<1x12x512x512x!qElemType, {order = #NHWC}>
    %reorder2_2 = IE.Reorder(%arg2) {
        dstOrder = #NHWC
    } : tensor<1x12x512x512x!qElemType> -> tensor<1x12x512x512x!qElemType, {order = #NHWC}>
    %add1 = IE.Add(%avgpool2, %reorder2_2) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x12x512x512x!qElemType, {order = #NHWC}>, tensor<1x12x512x512x!qElemType, {order = #NHWC}> -> tensor<1x12x512x512xf16, {order = #NHWC}>

    %add = IE.Add(%avgpool1_2, %add1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x12x512x512xf16, {order = #NHWC}>, tensor<1x12x512x512xf16, {order = #NHWC}> -> tensor<1x12x512x512xf16, {order = #NHWC}>
    %reorder = IE.Reorder(%add) {
        dstOrder = #NCHW
    } : tensor<1x12x512x512xf16, {order = #NHWC}> -> tensor<1x12x512x512xf16>

    return %reorder : tensor<1x12x512x512xf16>

    // The Reorders are propagated and fused so only permuteCasts are left after the conversion
    // Eltwise input branch 1
    // CHECK:       [[PERMUTECAST1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK:       [[AVGPOOL1:%.+]] = IE.AvgPool([[PERMUTECAST1]])
    // CHECK:       [[AVGPOOL2:%.+]] = IE.AvgPool([[AVGPOOL1]])
    // Eltwise input branch 2
    // CHECK:       [[PERMUTECAST2:%.+]] = IE.PermuteCast([[INPUT2]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK:       [[AVGPOOL3:%.+]] = IE.AvgPool([[PERMUTECAST2]])
    // CHECK:       [[PERMUTECAST3:%.+]] = IE.PermuteCast([[INPUT3]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK:       [[ADD1:%.+]] = IE.Add([[AVGPOOL3]], [[PERMUTECAST3]]
    // CHECK:       [[ADD:%.+]] = IE.Add([[AVGPOOL2]], [[ADD1]]
    // CHECK:       [[PERMUTECAST4:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // return [[PERMUTECAST4]]
}
