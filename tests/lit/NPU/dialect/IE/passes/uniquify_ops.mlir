//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --uniquify-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UniquifyReorders(%arg0: tensor<1x16x227x227xf16>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x227x227xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x227x227xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %2 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK: [[REORDER:%.+]] = IE.Reorder
    // CHECK-NOT: IE.Reorder
    // CHECK: [[RESULT:%.+]] = IE.Add([[REORDER]], [[REORDER]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ReordersWithDifferentConsumerOps(%arg0: tensor<1x16x112x112xf16>) -> tensor<1x16x112x112xf16, {order = #NHWC}> {
    %cst_69 = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<16x1x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_90 = const.Declare tensor<1x16x1x1xf16> = dense<0.0> : tensor<1x16x1x1xf16>

    %8 = IE.HSwish(%arg0) : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16>
    %9 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %10 = IE.GroupConvolution(%9, %cst_69, %cst_90) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<16x1x3x3xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %14 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %15 = IE.Add(%10, %14) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<1x16x112x112xf16, {order = #NHWC}> -> tensor<1x16x112x112xf16, {order = #NHWC}>

    return %15 : tensor<1x16x112x112xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x16x1x1xf16>

    // CHECK: IE.HSwish
    // CHECK: [[REORDER:%.+]] = IE.Reorder
    // CHECK: [[CONV:%.+]] = IE.GroupConvolution([[REORDER]], [[CST0]], [[CST1]])
    // CHECK-NOT: IE.Reorder
    // CHECK: IE.Add([[CONV]], [[REORDER]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TripleReordersWithDifferentConsumerOps
func.func @TripleReordersWithDifferentConsumerOps(%arg0: tensor<1x16x112x112xf16>) -> tensor<1x16x112x112x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>  {
    %cst_69 = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}> = dense<0.0> : tensor<16x1x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_90 = const.Declare tensor<1x16x1x1xf16> = dense<0.0> : tensor<1x16x1x1xf16>

    %8 = IE.HSwish(%arg0) : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16>
    %9 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %10 = IE.GroupConvolution(%9, %cst_69, %cst_90) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<16x1x3x3xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %14 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %15 = IE.Add(%10, %14) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<1x16x112x112xf16, {order = #NHWC}> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %16 = IE.Reorder(%8) {dstOrder = #NHWC} : tensor<1x16x112x112xf16> -> tensor<1x16x112x112xf16, {order = #NHWC}>
    %17 = IE.Add(%15, %16) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x112x112xf16, {order = #NHWC}>, tensor<1x16x112x112xf16, {order = #NHWC}> -> tensor<1x16x112x112x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %17 : tensor<1x16x112x112x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x16x1x1xf16>

    // CHECK: IE.HSwish
    // CHECK: [[REORDER:%.+]] = IE.Reorder
    // CHECK: [[CONV:%.+]] = IE.GroupConvolution([[REORDER]], [[CST0]], [[CST1]])
    // CHECK-NOT: IE.Reorder
    // CHECK: [[ADD:%.+]] = IE.Add([[CONV]], [[REORDER]])
    // CHECK-NOT: IE.Reorder
    // CHECK: IE.Add([[ADD]], [[REORDER]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UniquifyExpands(%arg0: tensor<1x13x227x227xf16, {order = #NHWC}>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x13x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %1 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x13x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %2 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK: [[EXPAND:%.+]] = IE.Expand
    // CHECK-NOT: IE.Expand
    // CHECK: [[RESULT:%.+]] = IE.Add([[EXPAND]], [[EXPAND]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipUniquifyExpandsIfAttrDiff
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x10x227x227xf16, {order = #NHWC}>
func.func @SkipUniquifyExpandsIfAttrDiff(%arg0: tensor<1x10x227x227xf16, {order = #NHWC}>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 6, 0, 0]} : tensor<1x10x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %1 = IE.Expand(%arg0) {pads_begin = [0, 3, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x10x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %2 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    //CHECK:      [[EXPAND_0:%.+]] = IE.Expand([[INPUT1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 6, 0, 0]} : tensor<1x10x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    //CHECK:      [[EXPAND_1:%.+]] = IE.Expand([[INPUT1]]) {pads_begin = [0, 3, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x10x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    //CHECK:      [[ADD:%.+]] = IE.Add([[EXPAND_0]], [[EXPAND_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!qElemType, {order = #NHWC}>

    // CHECK: return [[ADD]] : tensor<1x16x227x227x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UniquifyExpandReorders(%arg0: tensor<1x13x227x227xf16>) -> tensor<2x16x227x227xf16, {order = #NHWC}> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x13x227x227xf16> -> tensor<1x13x227x227xf16, {order = #NHWC}>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x13x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>

    %3 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x13x227x227xf16> -> tensor<1x13x227x227xf16, {order = #NHWC}>
    %4 = IE.Expand(%3) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x13x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %5 = IE.Add(%4, %4) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>

    %6 = IE.Concat(%2, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<2x16x227x227xf16, {order = #NHWC}>

    return %6 : tensor<2x16x227x227xf16, {order = #NHWC}>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<1x13x227x227xf16> -> tensor<1x13x227x227xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[REORDER]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} :
    // CHECK-SAME:                      tensor<1x13x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227xf16, {order = #NHWC}>

    // CHECK: [[AND0:%.+]] = IE.Add([[EXPAND]], [[EXPAND]])
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[AND0]], [[AND0]])
    // CHECK: return [[CONCAT]] : tensor<2x16x227x227xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UniquifyPermuteCast(%arg0: tensor<1x227x227x16xf16>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm=#NCHW} : tensor<1x227x227x16xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %1 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm=#NCHW} : tensor<1x227x227x16xf16> -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %2 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK: [[PERMUTE_CAST:%.+]] = IE.PermuteCast
    // CHECK-NOT: IE.PermuteCast
    // CHECK: [[RESULT:%.+]] = IE.Add([[PERMUTE_CAST]], [[PERMUTE_CAST]])
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @UniquifyShapeCast(%arg0: tensor<1x16x1x51529xf16, {order = #NHWC}>) -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}> {
    %0 = IE.ShapeCast {shape = [1, 16, 227, 227]} inputs(%arg0 :tensor<1x16x1x51529xf16, {order = #NHWC}>) -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %1 = IE.ShapeCast {shape = [1, 16, 227, 227]} inputs(%arg0 :tensor<1x16x1x51529xf16, {order = #NHWC}>) -> tensor<1x16x227x227xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    return %2 : tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

    // CHECK: [[SHAPE_CAST:%.+]] = IE.ShapeCast
    // CHECK-NOT: IE.ShapeCast
    // CHECK: [[RESULT:%.+]] = IE.Add([[SHAPE_CAST]], [[SHAPE_CAST]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutType = tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

func.func @UniquifyAdd(%arg0: tensor<1x16x227x227xf16, {order = #NHWC}>) -> (!OutType, !OutType) {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType
    %1 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType

    return %0, %1 : !OutType, !OutType

    // CHECK: [[ADD:%.+]] = IE.Add
    // CHECK-NOT: IE.Add
    // CHECK: return [[ADD]], [[ADD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutType = tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

func.func @NotUniquifyAddWithSameInput(%arg0: tensor<1x16x227x227xf16, {order = #NHWC}>) -> !OutType {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType

    return %0 : !OutType

    // CHECK: [[ADD:%.+]] = IE.Add
    // CHECK: return [[ADD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutType = tensor<1x16x227x227x!quant.uniform<u8:f16, 1.1534313725490195:128>, {order = #NHWC}>

func.func @UniquifyAddWithCrossInputs(%arg0: tensor<1x16x227x227xf16, {order = #NHWC}>, %arg1: tensor<1x16x227x227xf16, {order = #NHWC}>) -> (!OutType, !OutType) {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType
    %1 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType

    return %0, %1 : !OutType, !OutType

    // CHECK: [[ADD:%.+]] = IE.Add
    // CHECK-NOT: IE.Add
    // CHECK: return [[ADD]], [[ADD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutType = tensor<1x16x227x227xf16, {order = #NHWC}>

// CHECK-LABEL: @UniquifyAnd
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x227x227xf16, {order = #NHWC}>
func.func @UniquifyAnd(%arg0: tensor<1x16x227x227xf16, {order = #NHWC}>) -> (!OutType, !OutType) {
    %0 = IE.And(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType
    %1 = IE.And(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x227x227xf16, {order = #NHWC}>, tensor<1x16x227x227xf16, {order = #NHWC}> -> !OutType

    return %0, %1 : !OutType, !OutType

    // CHECK: [[AND:%.+]] = IE.And([[INPUT]], [[INPUT]])
    // CHECK-NOT: IE.And
    // CHECK: return [[AND]], [[AND]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0082835477941176471:128>
!qElemType1 = !quant.uniform<u8:f16, 0.017072610294117645:128>

!OutType = tensor<1x3x512x512x!qElemType1, {order = #NHWC}>

func.func @UniquifyQuantizeCast(%arg0: tensor<1x3x512x512x!qElemType, {order = #NHWC}>) -> (!OutType, !OutType) {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType1} : tensor<1x3x512x512x!qElemType, {order = #NHWC}> -> !OutType
    %1 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType1} : tensor<1x3x512x512x!qElemType, {order = #NHWC}> -> !OutType

    return %0, %1 : !OutType, !OutType

    // CHECK: [[CAST:%.+]] = IE.QuantizeCast
    // CHECK-NOT: IE.QuantizeCast
    // CHECK: return [[CAST]], [[CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UniquifyLayoutCast
func.func @UniquifyLayoutCast(%arg0: tensor<1x4x8x64xf16>) -> (tensor<1x4x8x64xf16, {order = #NHWC}>, tensor<1x4x8x64xf16, {order = #NHWC}>) {
    %0 = IE.LayoutCast(%arg0) {
        dst_order = #NHWC
    } : tensor<1x4x8x64xf16> -> tensor<1x4x8x64xf16, {order = #NHWC}>

    %1 = IE.LayoutCast(%arg0) {
        dst_order = #NHWC
    } : tensor<1x4x8x64xf16> -> tensor<1x4x8x64xf16, {order = #NHWC}>

    return %0, %1 : tensor<1x4x8x64xf16, {order = #NHWC}>, tensor<1x4x8x64xf16, {order = #NHWC}>

    // CHECK:   [[LAYOUT_CAST:%.+]] = IE.LayoutCast({{[^:]+}}) {
    // CHECK-NOT:  IE.LayoutCast
    // CHECK:   return [[LAYOUT_CAST]], [[LAYOUT_CAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @UniquifyMemPermute
func.func @UniquifyMemPermute(%arg0: tensor<1x16x2x3xf16>) ->
        (tensor<1x3x16x2xf16>, tensor<1x3x16x2xf16>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x16x2x3xf16> -> tensor<1x3x16x2xf16>
    %1 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
        tensor<1x16x2x3xf16> -> tensor<1x3x16x2xf16>

    return %0, %1 : tensor<1x3x16x2xf16>, tensor<1x3x16x2xf16>

    // CHECK:     [[PERMUTE:%.+]] = IE.MemPermute({{[^:]+}})
    // CHECK-NOT: IE.MemPermute
    // CHECK:     return [[PERMUTE]], [[PERMUTE]] : tensor<1x3x16x2xf16>, tensor<1x3x16x2xf16>
}

// -----

// CHECK-LABEL: @UniquifyAffineReshape
func.func @UniquifyAffineReshape(%arg0: tensor<15x2xf16>) -> (tensor<30xf16>, tensor<30xf16>) {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %1 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>

    return %0, %1 : tensor<30xf16>, tensor<30xf16>

    // CHECK: [[VAL0:%.+]] = IE.AffineReshape({{[^:]+}})
    // CHECK-NOT: IE.AffineReshape
    // CHECK: return [[VAL0]], [[VAL0]] : tensor<30xf16>, tensor<30xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @UniquifyMemPermute
func.func @UniquifyMemPermute(%arg0: tensor<1x16x2x3xf16>) ->
        (tensor<1x16x2x3xf16, {order = #NHWC}>, tensor<1x16x2x3xf16, {order = #NHWC}>) {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} :
         tensor<1x16x2x3xf16> -> tensor<1x16x2x3xf16, {order = #NHWC}>
    %1 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} :
         tensor<1x16x2x3xf16> -> tensor<1x16x2x3xf16, {order = #NHWC}>

    return %0, %1 : tensor<1x16x2x3xf16, {order = #NHWC}>, tensor<1x16x2x3xf16, {order = #NHWC}>

    // CHECK:     [[PERMUTE:%.+]] = IE.PermuteQuantize({{[^:]+}})
    // CHECK-NOT: IE.PermuteQuantize
    // CHECK:     return [[PERMUTE]], [[PERMUTE]] : tensor<1x16x2x3xf16, {order = #NHWC}>, tensor<1x16x2x3xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL:   @UniquifyMemPermuteForTheSameMergedPermutation
func.func @UniquifyMemPermuteForTheSameMergedPermutation(%arg0: tensor<1x1x512x1500xf16, {order = #NHWC}>) ->
        (tensor<1x1x1500x512xf16>, tensor<1x1x1500x512xf16, {order = #NHWC}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} :
        tensor<1x1x512x1500xf16, {order = #NHWC}> -> tensor<1x1x1500x512xf16>

    %1 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHCW} :
        tensor<1x1x512x1500xf16, {order = #NHWC}> -> tensor<1x1x1500x512xf16, {order = #NHWC}>

    return %0, %1 : tensor<1x1x1500x512xf16>, tensor<1x1x1500x512xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE:%.+]] = IE.MemPermute({{[^:]+}})
    // CHECK-SAME:  {dst_order = #NCHW, mem_perm = #NWHC} :
    // CHECK-SAME:  tensor<1x1x512x1500xf16, {order = #NHWC}> -> tensor<1x1x1500x512xf16>

    // CHECK:   [[LAYOUTCAST:%.+]] = IE.LayoutCast([[PERMUTE]])
    // CHECK-SAME:  {dst_order = #NHWC} :
    // CHECK-SAME:  tensor<1x1x1500x512xf16> -> tensor<1x1x1500x512xf16, {order = #NHWC}>

    // CHECK:     return [[PERMUTE]], [[LAYOUTCAST]] : tensor<1x1x1500x512xf16>, tensor<1x1x1500x512xf16, {order = #NHWC}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>
#perm1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>
#perm2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK:  #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>

// CHECK-LABEL:   @UniquifyMemPermuteForTheSame3DMergedPermutation
func.func @UniquifyMemPermuteForTheSame3DMergedPermutation(%arg0: tensor<1x2x3x4x1xf16, {order = #NCDHW}>) ->
        (tensor<1x2x4x3x1xf16>, tensor<1x3x2x4x1xf16, {order = #NDHWC}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCDHW, mem_perm = #perm1} :
        tensor<1x2x3x4x1xf16, {order = #NCDHW}> -> tensor<1x2x4x3x1xf16>

    %1 = IE.MemPermute(%arg0) {dst_order = #NDHWC, mem_perm = #perm2} :
        tensor<1x2x3x4x1xf16, {order = #NCDHW}> -> tensor<1x3x2x4x1xf16, {order = #NDHWC}>

    return %0, %1 : tensor<1x2x4x3x1xf16>, tensor<1x3x2x4x1xf16, {order = #NDHWC}>

    // CHECK:   [[PERMUTE:%.+]] = IE.MemPermute({{[^:]+}})
    // CHECK-SAME:  {dst_order = #NCDHW, mem_perm = #map} :
    // CHECK-SAME:  tensor<1x2x3x4x1xf16, {order = #NCDHW}> -> tensor<1x2x4x3x1xf16>

    // CHECK:   [[LAYOUTCAST:%.+]] = IE.LayoutCast([[PERMUTE]])
    // CHECK-SAME:  {dst_order = #NDHWC} :
    // CHECK-SAME:  tensor<1x2x4x3x1xf16> -> tensor<1x2x4x3x1xf16, {order = #NDHWC}>

    // CHECK:   [[SHAPECAST:%.+]] = IE.ShapeCast
    // CHECK-SAME:  {shape = [1, 3, 2, 4, 1]}
    // CHECK-SAME:  inputs([[LAYOUTCAST]] : tensor<1x2x4x3x1xf16, {order = #NDHWC}>) -> tensor<1x3x2x4x1xf16, {order = #NDHWC}>

    // CHECK:     return [[PERMUTE]], [[SHAPECAST]] : tensor<1x2x4x3x1xf16>, tensor<1x3x2x4x1xf16, {order = #NDHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CannotUniquifyAddsWithDifferentInputs
func.func @CannotUniquifyAddsWithDifferentInputs(%arg0: tensor<1x128x4x4xf16>, %arg1: tensor<1x128x4x4xf16>) -> (tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16>) {
    %cst_10 = const.Declare tensor<128x1x3x3xf16> = dense<0.0> : tensor<128x1x3x3xf16>
    %cst_11 = const.Declare tensor<1x128x1x1xf16> = dense<0.0> : tensor<1x128x1x1xf16>

    %cst_20 = const.Declare tensor<128x1x3x3xf16> = dense<1.0> : tensor<128x1x3x3xf16>
    %cst_21 = const.Declare tensor<1x128x1x1xf16> = dense<1.0> : tensor<1x128x1x1xf16>

    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    %1 = IE.GroupConvolution(%0, %cst_10, %cst_11) {dilations = [1, 1], groups = 128 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x4x4xf16>, tensor<128x1x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x4xf16>
    %2 = IE.GroupConvolution(%0, %cst_20, %cst_21) {dilations = [1, 1], groups = 128 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x4x4xf16>, tensor<128x1x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x4xf16>
    %3 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    %4 = IE.Add(%0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>

    return %3, %4 : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<128x1x3x3xf16> = dense<0.000000e+00> : tensor<128x1x3x3xf16>
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<0.000000e+00> : tensor<1x128x1x1xf16>
    // CHECK-DAG: [[CST1:%.+]] = const.Declare tensor<128x1x3x3xf16> = dense<1.000000e+00> : tensor<128x1x3x3xf16>
    // CHECK-DAG: [[CST2:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+00> : tensor<1x128x1x1xf16>
    // CHECK:     [[ADD:%.+]] = IE.Add({{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:     [[CONV1:%.+]] = IE.GroupConvolution([[ADD]], [[CST]], [[CST0]]) {dilations = [1, 1], groups = 128 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x4x4xf16>, tensor<128x1x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x4xf16>
    // CHECK:     [[CONV2:%.+]] = IE.GroupConvolution([[ADD]], [[CST1]], [[CST2]]) {dilations = [1, 1], groups = 128 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x4x4xf16>, tensor<128x1x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x4xf16>
    // CHECK:     [[ADD1:%.+]] = IE.Add([[ADD]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:     [[ADD2:%.+]] = IE.Add([[ADD]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:     return [[ADD1]], [[ADD2]] : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16>

}

// -----

// CHECK-LABEL: @UniquifyAddsSwappedInputs
func.func @UniquifyAddsSwappedInputs(%arg0: tensor<1x128x4x4xf16>, %arg1: tensor<1x128x4x4xf16>) -> (tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16>) {
    %cst_10 = const.Declare tensor<128x1x3x3xf16> = dense<0.0> : tensor<128x1x3x3xf16>
    %cst_11 = const.Declare tensor<1x128x1x1xf16> = dense<0.0> : tensor<1x128x1x1xf16>

    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    %1 = IE.GroupConvolution(%0, %cst_10, %cst_11) {dilations = [1, 1], groups = 128 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x4x4xf16>, tensor<128x1x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x4xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    %3 = IE.Add(%1, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>

    return %2, %3 : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<128x1x3x3xf16> = dense<0.000000e+00> : tensor<128x1x3x3xf16>
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<0.000000e+00> : tensor<1x128x1x1xf16>
    // CHECK:     [[ADD:%.+]] = IE.Add({{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:     [[CONV:%.+]] = IE.GroupConvolution([[ADD]], [[CST]], [[CST0]]) {dilations = [1, 1], groups = 128 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x4x4xf16>, tensor<128x1x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x4xf16>
    // CHECK:     [[ADD1:%.+]] = IE.Add([[ADD]], [[CONV]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK-NOT: IE.ADD
    // CHECK:     return [[ADD1]], [[ADD1]] : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16>

}

// -----

// CHECK-LABEL: @UniquifyReshape
func.func @UniquifyReshape(%arg0: tensor<1x32x44x44xf16>) -> (tensor<1x32x16x121xf16>, tensor<1x32x16x121xf16>) {
    %0 = IE.Reshape(%arg0) {shape_value = [1, 32, 16, 121]} : tensor<1x32x44x44xf16> -> tensor<1x32x16x121xf16>
    %1 = IE.Reshape(%arg0) {shape_value = [1, 32, 16, 121]} : tensor<1x32x44x44xf16> -> tensor<1x32x16x121xf16>

    return %0, %1 : tensor<1x32x16x121xf16>, tensor<1x32x16x121xf16>

    // CHECK: [[VAL0:%.+]] = IE.Reshape({{[^:]+}})
    // CHECK-NOT: IE.Reshape
    // CHECK: return [[VAL0]], [[VAL0]] : tensor<1x32x16x121xf16>, tensor<1x32x16x121xf16>
}


// -----

// CHECK-LABEL: @UniquifyConcat
func.func @UniquifyConcat(%arg0: tensor<1x32x44x44xf16>, %arg1: tensor<1x32x44x44xf16>) -> (tensor<1x64x44x44xf16>, tensor<1x64x44x44xf16>) {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x64x44x44xf16>
    %1 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x64x44x44xf16>
    return %0, %1 : tensor<1x64x44x44xf16>, tensor<1x64x44x44xf16>

    //CHECK:        [[CONCAT0:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}}) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x64x44x44xf16>
    //CHECK:        return [[CONCAT0]], [[CONCAT0]] : tensor<1x64x44x44xf16>, tensor<1x64x44x44xf16>

}

// -----

// CHECK-LABEL: @NotUniquifyConcatDifferentInput
func.func @NotUniquifyConcatDifferentInput(%arg0: tensor<1x32x44x44xf16>, %arg1: tensor<1x32x44x44xf16>) -> (tensor<1x64x44x44xf16>, tensor<1x64x44x44xf16>) {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x64x44x44xf16>
    %1 = IE.Concat(%arg1, %arg0) {per_axis = #IE.Concat<axis = 1>} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x64x44x44xf16>
    return %0, %1 : tensor<1x64x44x44xf16>, tensor<1x64x44x44xf16>

    //CHECK:     [[CONCAT0:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}})
    //CHECK:     [[CONCAT1:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}})
    //CHECK:     return [[CONCAT0]], [[CONCAT1]] : tensor<1x64x44x44xf16>, tensor<1x64x44x44xf16>
}

// -----

// CHECK-LABEL: @NotUniquifyConcatDifferentInputNumber
func.func @NotUniquifyConcatDifferentInputNumber(%arg0: tensor<1x32x44x44xf16>, %arg1: tensor<1x32x44x44xf16>) -> (tensor<1x96x44x44xf16>, tensor<1x64x44x44xf16>) {
    %0 = IE.Concat(%arg0, %arg1, %arg0) {per_axis = #IE.Concat<axis = 1>} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x96x44x44xf16>
    %1 = IE.Concat(%arg1, %arg0) {per_axis = #IE.Concat<axis = 1>} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x64x44x44xf16>
    return %0, %1 : tensor<1x96x44x44xf16>, tensor<1x64x44x44xf16>

    //CHECK:     [[CONCAT0:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    //CHECK:     [[CONCAT1:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}})
    //CHECK:     return [[CONCAT0]], [[CONCAT1]] : tensor<1x96x44x44xf16>, tensor<1x64x44x44xf16>
}

// -----

// CHECK-LABEL: @NotUniquifyConcatDifferentAxis
func.func @NotUniquifyConcatDifferentAxis(%arg0: tensor<1x32x44x44xf16>, %arg1: tensor<1x32x44x44xf16>) -> (tensor<1x32x88x44xf16>, tensor<1x32x44x88xf16>) {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 2>} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x32x88x44xf16>
    %1 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 3>}: tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x32x44x88xf16>
    return %0, %1 : tensor<1x32x88x44xf16>, tensor<1x32x44x88xf16>

    //CHECK:     [[CONCAT0:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}})
    //CHECK:     [[CONCAT1:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}})
    //CHECK:     return [[CONCAT0]], [[CONCAT1]] : tensor<1x32x88x44xf16>, tensor<1x32x44x88xf16>
}

// -----

// CHECK-LABEL: @NotUniquifyConcatDifferentOffsetStride
func.func @NotUniquifyConcatDifferentOffsetStride(%arg0: tensor<1x32x44x44xf16>, %arg1: tensor<1x32x44x44xf16>) -> (tensor<1x64x44x44xf16>, tensor<1x64x44x44xf16>) {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1, offset = 1, stride = 2>} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x64x44x44xf16>
    %1 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x64x44x44xf16>
    return %0, %1 : tensor<1x64x44x44xf16>, tensor<1x64x44x44xf16>

    //CHECK:     [[CONCAT0:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}})
    //CHECK:     [[CONCAT1:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}})
    //CHECK:     return [[CONCAT0]], [[CONCAT1]] : tensor<1x64x44x44xf16>, tensor<1x64x44x44xf16>
}

// -----

// CHECK-LABEL: @NotUniquifyConcatDifferentStaticOffset
func.func @NotUniquifyConcatDifferentStaticOffset(%arg0: tensor<1x32x44x44xf16>, %arg1: tensor<1x32x44x44xf16>) -> (tensor<1x32x131x44xf16>, tensor<1x32x131x44xf16>) {
    %0 = IE.Concat(%arg0, %arg1, %arg0) {static_offsets = [[0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 87, 0]]} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x32x131x44xf16>
    %1 = IE.Concat(%arg0, %arg1, %arg0) {static_offsets = [[0, 0, 0, 0], [0, 0, 44, 0], [0, 0, 87, 0]]} : tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16>, tensor<1x32x44x44xf16> -> tensor<1x32x131x44xf16>
    return %0, %1 : tensor<1x32x131x44xf16>, tensor<1x32x131x44xf16>

    //CHECK:     [[CONCAT0:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    //CHECK:     [[CONCAT1:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    //CHECK:     return [[CONCAT0]], [[CONCAT1]] : tensor<1x32x131x44xf16>, tensor<1x32x131x44xf16>
}


// -----

// CHECK-LABEL: @UniquifyTile
func.func @UniquifyTile(%arg0: tensor<1x1x1x44xf16>) -> (tensor<1x2x16x44xf16>, tensor<1x2x16x44xf16>) {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 16, 1]} : tensor<1x1x1x44xf16> -> tensor<1x2x16x44xf16>
    %1 = IE.Tile(%arg0) {repeats_values = [1, 2, 16, 1]} : tensor<1x1x1x44xf16> -> tensor<1x2x16x44xf16>
    return %0, %1 : tensor<1x2x16x44xf16>, tensor<1x2x16x44xf16>

    //CHECK:      [[TILE:%.+]] = IE.Tile({{[^:]+}}) {repeats_values = [1, 2, 16, 1]} : tensor<1x1x1x44xf16> -> tensor<1x2x16x44xf16>
    //CHECK:      return  [[TILE]], [[TILE]] : tensor<1x2x16x44xf16>, tensor<1x2x16x44xf16>
}

// -----

// CHECK-LABEL: @UniquifyAvgPool
func.func @UniquifyAvgPool(%arg0: tensor<1x1x16x16xf16>) -> (tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>) {
    %0 = IE.AvgPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    %1 = IE.AvgPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    return %0, %1 : tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>

    //CHECK:      [[AVGPOOL:%.+]] = IE.AvgPool({{[^:]+}})
    //CHECK-SAME: {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]}
    //CHECK-SAME: tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>
    //CHECK:      return  [[AVGPOOL]], [[AVGPOOL]] : tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>
}

// -----

// CHECK-LABEL: @UniquifyMaxPool
func.func @UniquifyMaxPool(%arg0: tensor<1x1x16x16xf16>) -> (tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>) {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    %1 = IE.MaxPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    return %0, %1 : tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>

    //CHECK:      [[MAXPOOL:%.+]] = IE.MaxPool({{[^:]+}})
    //CHECK-SAME: {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]}
    //CHECK-SAME: tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>
    //CHECK:      return  [[MAXPOOL]], [[MAXPOOL]] : tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>
}

// CHECK-LABEL: @DoNotUniquifyAvgPool
func.func @DoNotUniquifyAvgPool(%arg0: tensor<1x1x16x16xf16>) -> (tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>) {
    %0 = IE.AvgPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    %1 = IE.AvgPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    return %0, %1 : tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>

    //CHECK:      [[AVGPOOL0:%.+]] = IE.AvgPool({{[^:]+}})
    //CHECK-SAME: {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]}
    //CHECK-SAME: tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    //CHECK:      [[AVGPOOL1:%.+]] = IE.AvgPool({{[^:]+}})
    //CHECK-SAME: {kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]}
    //CHECK-SAME: tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    //CHECK:      return  [[AVGPOOL0]], [[AVGPOOL1]] : tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>
}

// CHECK-LABEL: @DoNotUniquifyMaxPool
func.func @DoNotUniquifyMaxPool(%arg0: tensor<1x1x16x16xf16>) -> (tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>) {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    %1 = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 2]
    } : tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    return %0, %1 : tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>

    //CHECK:      [[MAXPOOL0:%.+]] = IE.MaxPool({{[^:]+}})
    //CHECK-SAME: {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]}
    //CHECK-SAME: tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    //CHECK:      [[MAXPOOL1:%.+]] = IE.MaxPool({{[^:]+}})
    //CHECK-SAME: {kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]}
    //CHECK-SAME: tensor<1x1x16x16xf16> -> tensor<1x1x8x8xf16>

    //CHECK:      return  [[MAXPOOL0]], [[MAXPOOL1]] : tensor<1x1x8x8xf16>, tensor<1x1x8x8xf16>
}

// -----

// CHECK-LABEL: @UniquifySlice
func.func @UniquifySlice(%arg0: tensor<1x128x96x64xf16>) -> (tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    return %0, %1 : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>

    //CHECK:      [[SLICE:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    //CHECK:      return  [[SLICE]], [[SLICE]] : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>
}

// -----

// CHECK-LABEL: @UniquifyMultiSlice
func.func @UniquifyMultiSlice(%arg0: tensor<1x128x96x64xf16>) -> (tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    return %0, %1, %2 : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>

    //CHECK:      [[SLICE:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    //CHECK:      return  [[SLICE]], [[SLICE]], [[SLICE]] : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>
}

// -----

// CHECK-LABEL: @UniquifyMultiSliceWithMultiOffset
func.func @UniquifyMultiSliceWithMultiOffset(%arg0: tensor<1x128x96x64xf16>) -> (tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %1 = IE.Slice %arg0 [0, 4, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %2 = IE.Slice %arg0 [0, 8, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %4 = IE.Slice %arg0 [0, 4, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %5 = IE.Slice %arg0 [0, 8, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    return %0, %1, %2, %3, %4, %5 : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>

    //CHECK:      [[SLICE_OFFSET0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    //CHECK:      [[SLICE_OFFSET4:%.+]] = IE.Slice {{[^:]+}} [0, 4, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    //CHECK:      [[SLICE_OFFSET8:%.+]] = IE.Slice {{[^:]+}} [0, 8, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    //CHECK:      return  [[SLICE_OFFSET0]], [[SLICE_OFFSET4]], [[SLICE_OFFSET8]], [[SLICE_OFFSET0]], [[SLICE_OFFSET4]], [[SLICE_OFFSET8]]
}

// -----

// CHECK-LABEL: @NotUniquifySliceWithDifferentOffset
func.func @NotUniquifySliceWithDifferentOffset(%arg0: tensor<1x128x96x64xf16>) -> (tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %1 = IE.Slice %arg0 [0, 4, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    return %0, %1 : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>

    //CHECK:      [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    //CHECK:      [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 4, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    //CHECK:      return  [[SLICE_0]], [[SLICE_1]] : tensor<1x64x96x64xf16>, tensor<1x64x96x64xf16>
}

// -----

// CHECK-LABEL: @NotUniquifySliceWithDifferentSize
func.func @NotUniquifySliceWithDifferentSize(%arg0: tensor<1x128x96x64xf16>) -> (tensor<1x64x96x64xf16>, tensor<1x32x96x64xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 32, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x32x96x64xf16>
    return %0, %1 : tensor<1x64x96x64xf16>, tensor<1x32x96x64xf16>

    //CHECK:      [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0] [1, 64, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x64x96x64xf16>
    //CHECK:      [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0] [1, 32, 96, 64] : tensor<1x128x96x64xf16> to tensor<1x32x96x64xf16>
    //CHECK:      return  [[SLICE_0]], [[SLICE_1]] : tensor<1x64x96x64xf16>, tensor<1x32x96x64xf16>
}

// -----

// CHECK-LABEL: @UniquifyFloor
func.func @UniquifyFloor(%arg0: tensor<1x1x1x880xf16>) -> (tensor<1x1x1x880xf16>, tensor<1x1x1x880xf16>) {
    %0 = IE.Floor(%arg0) : tensor<1x1x1x880xf16> -> tensor<1x1x1x880xf16>
    %1 = IE.Floor(%arg0) : tensor<1x1x1x880xf16> -> tensor<1x1x1x880xf16>

    return %0, %1 : tensor<1x1x1x880xf16>, tensor<1x1x1x880xf16>

    // CHECK: [[VAL0:%.+]] = IE.Floor({{[^:]+}})
    // CHECK-NOT: IE.Floor
    // CHECK: return [[VAL0]], [[VAL0]] : tensor<1x1x1x880xf16>, tensor<1x1x1x880xf16>
}

// -----

// CHECK-LABEL: @UniquifyConvert
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x880xf32>
func.func @UniquifyConvert(%arg0: tensor<1x1x1x880xf32>) -> (tensor<1x1x1x880xf16>, tensor<1x1x1x880xf16>) {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x1x880xf32> -> tensor<1x1x1x880xf16>
    %1 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x1x880xf32> -> tensor<1x1x1x880xf16>
    return %0, %1 : tensor<1x1x1x880xf16>, tensor<1x1x1x880xf16>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x1x1x880xf32> -> tensor<1x1x1x880xf16>
    // CHECK-NOT: IE.Convert
    // CHECK: return [[CONVERT]], [[CONVERT]] : tensor<1x1x1x880xf16>, tensor<1x1x1x880xf16>
}

// -----

// CHECK-LABEL: @UniquifyAffineReshapeWithMaxIterations
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<15x2xf16>
func.func @UniquifyAffineReshapeWithMaxIterations(%arg0: tensor<15x2xf16>) -> tensor<390xf16> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %1 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %2 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %3 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %4 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %5 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %6 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %7 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %8 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %9 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %10 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %11 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %12 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [30] } : tensor<15x2xf16> -> tensor<30xf16>
    %13 = IE.Concat(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12) {static_offsets = [[0], [30], [60], [90], [120], [150], [180], [210], [240], [270], [300], [330], [360]]}
         : tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16> -> tensor<390xf16>

    return %13 : tensor<390xf16>

    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1]], shape_value = [30]} : tensor<15x2xf16> -> tensor<30xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0], [30], [60], [90], [120], [150], [180], [210], [240], [270], [300], [330], [360]]} : tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16>, tensor<30xf16> -> tensor<390xf16>

    // CHECK: return [[CONCAT]] : tensor<390xf16>
}
