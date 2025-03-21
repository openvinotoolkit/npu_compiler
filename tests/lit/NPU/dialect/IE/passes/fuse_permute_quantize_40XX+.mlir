//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-permute-quantize="dpu-only=true"  %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01>

func.func @SkipConvertWithReshape(%arg0: tensor<3x62x62xf32>) -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}> {
    %0 = IE.Convert(%arg0) {
      dstElemType = f16
    } : tensor<3x62x62xf32> -> tensor<3x62x62xf16>

    %1 = IE.AffineReshape(%0) {
      dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 3, 62, 62]
    } : tensor<3x62x62xf16> -> tensor<1x3x62x62xf16>

    %2 = IE.Reorder(%1) {
        dstOrder = #NHWC
    } : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>

    %3 = IE.Add(%2, %2) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x3x62x62xf16, {order = #NHWC}>,
        tensor<1x3x62x62xf16, {order = #NHWC}>
        -> tensor<1x3x62x62x!qElemType, {order = #NHWC}>

    %4 = IE.QuantizeCast(%3) {
        dstElemType = !qElemType1
    } : tensor<1x3x62x62x!qElemType, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

    return %4 : tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

    // CHECK:   [[CONVERT:%.*]] = IE.Convert(%arg0)
    // CHECK:   [[RESHAPE:%.*]] = IE.AffineReshape([[CONVERT]])
    // CHECK:   [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize([[RESHAPE]]) {
    // CHECK-SAME:      dstElemType = !qElemType1,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x62x62xf16> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

    // CHECK:   [[QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[PERMUTE_QUANTIZE]])
    // CHECK:   return [[QUANTIZE_CAST]] : tensor<1x3x62x62x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = !quant.uniform<u8:f16, 1.000000e+00>

func.func @SkipPermuteQuantizeF32(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}> {
    %0 = IE.Convert(%arg0) {
      dstElemType = f16
    } : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf16>

    %1 = IE.Reorder(%0) {
        dstOrder = #NHWC
    } : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>

    %2 = IE.Add(%1, %1) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x3x62x62xf16, {order = #NHWC}>,
        tensor<1x3x62x62xf16, {order = #NHWC}>
        -> tensor<1x3x62x62x!qElemType, {order = #NHWC}>

    %3 = IE.QuantizeCast(%2) {
        dstElemType = !qElemType1
    } : tensor<1x3x62x62x!qElemType, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

    return %3 : tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

    // CHECK:  [[CONVERT:%.*]] = IE.Convert(%arg0) {dstElemType = f16}

    // CHECK: [[PERM_QUANT:%.*]] = IE.PermuteQuantize([[CONVERT]]) {
    // CHECK-SAME:        dstElemType = [[PERM_QUANT_TYPE]],
    // CHECK-SAME:        dst_order = #NHWC,
    // CHECK-SAME:        mem_perm = #NHWC,
    // CHECK-SAME:        pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:        pads_end = [0, 0, 0, 0]
    // CHECK-SAME:    } : tensor<1x3x62x62xf16> -> tensor<1x3x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>

    // CHECK: [[QUANT_CAST:%.*]] = IE.QuantizeCast([[PERM_QUANT]]) {
    // CHECK-SAME:        dstElemType = [[QUANT_CAST_TYPE]]
    // CHECK-SAME:    } : tensor<1x3x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x3x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>

    // CHECK: return [[QUANT_CAST]] : tensor<1x3x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

func.func @SkipWidth1(%arg0: tensor<1x64x2x1xf16>) -> tensor<1x64x2x1x!qElemType1, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {
        dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    } : tensor<1x64x2x1xf16> -> tensor<1x64x2x1xf16, {order = #NHWC}>

    %1 = IE.Add(%0, %0) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x64x2x1xf16, {order = #NHWC}>, tensor<1x64x2x1xf16, {order = #NHWC}>
      -> tensor<1x64x2x1x!qElemType, {order = #NHWC}>

    %2 = IE.QuantizeCast(%1) {
        dstElemType = !quant.uniform<u8:f16, 1.000000e+00>
    } : tensor<1x64x2x1x!qElemType, {order = #NHWC}> -> tensor<1x64x2x1x!qElemType1, {order = #NHWC}>

    return %2 : tensor<1x64x2x1x!qElemType1, {order = #NHWC}>

    // CHECK:   [[REORDER:%.*]] = IE.Reorder(%arg0)
    // CHECK:   [[ADD:%.*]] = IE.Add([[REORDER]], [[REORDER]])
    // CHECK:   [[QUANT_CAST:%.*]] = IE.QuantizeCast([[ADD]])
    // CHECK:   return [[QUANT_CAST]] : tensor<1x64x2x1x!qElemType, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplyMemPermuteNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x16xf16>
func.func @MultiplyMemPermuteNHWC(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x3x16x16xf16>
        = dense<12.000000e+00> : tensor<1x3x16x16xf16>

    %0 = IE.Multiply(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x3x16x16xf16, {order = #NHWC}>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x3x16x16xf16> = dense<1.200000e+01> : tensor<1x3x16x16xf16>
    // CHECK:   %[[MUL:.*]] = IE.Multiply(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    // CHECK:   %[[RESULT:.*]] = IE.MemPermute(%[[MUL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // return   %[[RESULT]]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SubtractMemPermuteNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x16xf16>
func.func @SubtractMemPermuteNHWC(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x3x16x16xf16>
        = dense<12.000000e+00> : tensor<1x3x16x16xf16>

    %0 = IE.Subtract(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x3x16x16xf16, {order = #NHWC}>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x3x16x16xf16> = dense<1.200000e+01> : tensor<1x3x16x16xf16>
    // CHECK:   %[[MUL:.*]] = IE.Subtract(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    // CHECK:   %[[RESULT:.*]] = IE.MemPermute(%[[MUL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // return   %[[RESULT]]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AndMemPermuteNHWC
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x16x16xf16>
func.func @AndMemPermuteNHWC(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x3x16x16xf16>
        = dense<12.000000e+00> : tensor<1x3x16x16xf16>

    %0 = IE.And(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>

    %1 = IE.MemPermute(%0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x3x16x16xf16, {order = #NHWC}>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x3x16x16xf16> = dense<1.200000e+01> : tensor<1x3x16x16xf16>
    // CHECK:   %[[MUL:.*]] = IE.And(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x16x16xf16>, tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16>
    // CHECK:   %[[RESULT:.*]] = IE.MemPermute(%[[MUL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // return   %[[RESULT]]

}
