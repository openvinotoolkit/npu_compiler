//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-permute-post-eltwise --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!qElemType = !quant.uniform<u8:f16, 0.001>
!qElemType1 = !quant.uniform<u8:f16, 0.002>
!qElemType2 = !quant.uniform<u8:f16, 0.003>
!qElemType3 = !quant.uniform<u8:f16, 0.004>
!qElemType4 = !quant.uniform<u8:f16, 0.005>


// CHECK-LABEL: @MovePermutePostAdd
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x256x256xf16>
func.func @MovePermutePostAdd(%arg0: tensor<1x3x256x256xf16>, %arg1: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}>
            -> tensor<1x3x256x256xf16, {order = #NHWC}>
    return %2 : tensor<1x3x256x256xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_2:%.+]] = IE.PermuteCast([[INPUT2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[PERMUTE_CAST_1]], [[PERMUTE_CAST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x3x256xf16, {order = #NHWC}>, tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[ADD]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x3x256x256xf16, {order = #NHWC}>
}

// CHECK-LABEL: @MovePermuteQuantizePostAdd
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x256x256xf16>
func.func @MovePermuteQuantizePostAdd(%arg0: tensor<1x3x256x256xf16>, %arg1: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
            : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %1 = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
            : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}>
            -> tensor<1x3x256x256xf16, {order = #NHWC}>
    return %2 : tensor<1x3x256x256xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_2:%.+]] = IE.PermuteCast([[INPUT2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[PERMUTE_CAST_1]], [[PERMUTE_CAST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x3x256xf16, {order = #NHWC}>, tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[ADD]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x256x256xf16, {order = #NHWC}>
}

// CHECK-LABEL: @MovePermutePostAddWithOneInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x256x256xf16>
func.func @MovePermutePostAddWithOneInput(%arg0: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}>
            -> tensor<1x3x256x256xf16, {order = #NHWC}>
    return %2 : tensor<1x3x256x256xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[PERMUTE_CAST]], [[PERMUTE_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x3x256xf16, {order = #NHWC}>, tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[ADD]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x3x256x256xf16, {order = #NHWC}>
}

// CHECK-LABEL: @MovePermutePostAddWithQuantizeCast
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256x!qElemType>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x256x256x!qElemType1>
func.func @MovePermutePostAddWithQuantizeCast(%arg0: tensor<1x3x256x256x!qElemType>, %arg1: tensor<1x3x256x256x!qElemType1>)
        -> tensor<1x3x256x256x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
    %permute1 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256x!qElemType> -> tensor<1x3x256x256x!qElemType, {order = #NHWC}>
    %qc1 = IE.QuantizeCast(%permute1) {dstElemType = !qElemType2} : tensor<1x3x256x256x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
            -> tensor<1x3x256x256x!qElemType2, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %permute2 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256x!qElemType1> -> tensor<1x3x256x256x!qElemType1, {order = #NHWC}>
    %qc2 = IE.QuantizeCast(%permute2) {dstElemType = !qElemType3} : tensor<1x3x256x256x!qElemType1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
            -> tensor<1x3x256x256x!qElemType3, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %add = IE.Add(%qc1, %qc2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256x!qElemType2, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x3x256x256x!qElemType3, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
            -> tensor<1x3x256x256x!qElemType4, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %qc3 = IE.QuantizeCast(%add) {dstElemType = !qElemType} : tensor<1x3x256x256x!qElemType4, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
            -> tensor<1x3x256x256x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    return %qc3 : tensor<1x3x256x256x!qElemType, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    // CHECK:       [[PERM_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256x!qElemType>
    // CHECK-SAME:          -> tensor<1x256x3x256x!qElemType, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 3, 256, 256]} inputs([[PERM_CAST_1]] : tensor<1x256x3x256x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType, {order = #NHWC}>
    // CHECK:       [[QUANT_CAST_1:%.+]] = IE.QuantizeCast([[SHAPE_CAST_1]]) {dstElemType = !qElemType2} : tensor<1x3x256x256x!qElemType, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType2, {order = #NHWC}>
    // CHECK:       [[PERM_CAST_2:%.+]] = IE.PermuteCast([[INPUT2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256x!qElemType1>
    // CHECK-SAME:          -> tensor<1x256x3x256x!qElemType1, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [1, 3, 256, 256]} inputs([[PERM_CAST_2]] : tensor<1x256x3x256x!qElemType1, {order = #NHWC}>)
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType1, {order = #NHWC}>
    // CHECK:       [[QUANT_CAST_2:%.+]] = IE.QuantizeCast([[SHAPE_CAST_2]]) {dstElemType = !qElemType3} : tensor<1x3x256x256x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType3, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[QUANT_CAST_1]], [[QUANT_CAST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256x!qElemType2, {order = #NHWC}>, tensor<1x3x256x256x!qElemType3, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType4, {order = #NHWC}>
    // CHECK:       [[QUANT_CAST_OUT:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType} : tensor<1x3x256x256x!qElemType4, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 256, 3, 256]} inputs([[QUANT_CAST_OUT]] : tensor<1x3x256x256x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:          -> tensor<1x256x3x256x!qElemType, {order = #NHWC}>
    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[SHAPE_CAST_OUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256x!qElemType, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x3x256x256x!qElemType, {order = #NHWC}>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x3x256x256x!qElemType, {order = #NHWC}>
}

// CHECK-LABEL: @MovePermutePostAddWithSameLayout
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x256x256xf16, {order = #NHWC}>
func.func @MovePermutePostAddWithSameLayout(%arg0: tensor<1x3x256x256xf16, {order = #NHWC}>, %arg1: tensor<1x3x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x3xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x3xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x3xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x256x3xf16, {order = #NHWC}>, tensor<1x256x256x3xf16, {order = #NHWC}>
            -> tensor<1x256x256x3xf16, {order = #NHWC}>
    return %2 : tensor<1x256x256x3xf16, {order = #NHWC}>


    // CHECK:       [[SHAPE_CAST_1:%.+]]  = IE.ShapeCast {shape = [1, 256, 256, 3]} inputs([[INPUT1]] : tensor<1x3x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x3xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_2:%.+]]  = IE.ShapeCast {shape = [1, 256, 256, 3]} inputs([[INPUT2]] : tensor<1x3x256x256xf16, {order = #NHWC}>) -> tensor<1x256x256x3xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]]  = IE.Add([[SHAPE_CAST_1]], [[SHAPE_CAST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x256x3xf16, {order = #NHWC}>, tensor<1x256x256x3xf16, {order = #NHWC}> -> tensor<1x256x256x3xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]]  = IE.ShapeCast {shape = [1, 3, 256, 256]} inputs([[ADD]] : tensor<1x256x256x3xf16, {order = #NHWC}>) -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       [[MEMPERMUTE:%.+]]  = IE.MemPermute([[SHAPE_CAST_OUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x256x256x3xf16, {order = #NHWC}>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x256x256x3xf16, {order = #NHWC}>
}

// CHECK-LABEL: @MovePermutePostAddDifferentDstOrderAndMemPerm
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x512x64x8xf16>
func.func @MovePermutePostAddDifferentDstOrderAndMemPerm(%arg0: tensor<1x512x64x8xf16>) -> tensor<1x64x8x512x!qElemType, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x512x64x8xf16> -> tensor<1x64x8x512xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x8x512xf16, {order = #NHWC}>, tensor<1x64x8x512xf16, {order = #NHWC}> -> tensor<1x64x8x512x!qElemType1, {order = #NHWC}>
    %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType} : tensor<1x64x8x512x!qElemType1, {order = #NHWC}> -> tensor<1x64x8x512x!qElemType, {order = #NHWC}>
    return %2 : tensor<1x64x8x512x!qElemType, {order = #NHWC}>

    // CHECK:       [[PERM_CAST_1:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x512x64x8xf16> -> tensor<1x8x512x64xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 64, 8, 512]} inputs(%0 : tensor<1x8x512x64xf16, {order = #NHWC}>) -> tensor<1x64x8x512xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPE_CAST_1]], [[SHAPE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x8x512xf16, {order = #NHWC}>, tensor<1x64x8x512xf16, {order = #NHWC}> -> tensor<1x64x8x512x!qElemType1, {order = #NHWC}>
    // CHECK:       [[QUANT_CAST_OUT:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType} : tensor<1x64x8x512x!qElemType1, {order = #NHWC}> -> tensor<1x64x8x512x!qElemType, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [1, 8, 512, 64]} inputs(%3 : tensor<1x64x8x512x!qElemType, {order = #NHWC}>) -> tensor<1x8x512x64x!qElemType, {order = #NHWC}>
    // CHECK:       [[MEM_PERM:%.+]] = IE.MemPermute([[SHAPE_CAST_2]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x8x512x64x!qElemType, {order = #NHWC}> -> tensor<1x64x8x512x!qElemType, {order = #NHWC}>
    // CHECK:       return [[MEM_PERM]] : tensor<1x64x8x512x!qElemType, {order = #NHWC}>
}

// CHECK-LABEL: @DoNotMovePermutePostAddWhenDifferentMemPermAttributes
// CHECK-SAME:        [[INPUT1:%arg0]]: tensor<1x2x3x4xf16>
// CHECK-SAME:        [[INPUT1:%arg1]]: tensor<1x3x4x2xf16>
func.func @DoNotMovePermutePostAddWhenDifferentMemPermAttributes(%arg0: tensor<1x2x3x4xf16>, %arg1: tensor<1x3x4x2xf16>) -> tensor<1x4x2x3xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf16> -> tensor<1x4x2x3xf16>
    %1 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x3x4x2xf16> -> tensor<1x4x2x3xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x2x3xf16>, tensor<1x4x2x3xf16> -> tensor<1x4x2x3xf16>
    return %2 : tensor<1x4x2x3xf16>

    // CHECK:       [[MEM_PERM_1:%.+]]  = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf16> -> tensor<1x4x2x3xf16>
    // CHECK:       [[MEM_PERM_2:%.+]]  = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x3x4x2xf16> -> tensor<1x4x2x3xf16>
    // CHECK:       [[ADD:%.+]]  = IE.Add([[MEM_PERM_1]], [[MEM_PERM_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x2x3xf16>, tensor<1x4x2x3xf16> -> tensor<1x4x2x3xf16>
    // CHECK:       return [[ADD]] : tensor<1x4x2x3xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @PropagatePermuteThroughGroupConv
// CHECK-SAME:        [[BLOCK_ARG:%arg0]]: tensor<1x4x8x64xf16>
func.func @PropagatePermuteThroughGroupConv(%arg0: tensor<1x4x8x64xf16>) -> tensor<1x8x4x64xf16> {
    %SCALE = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x1x1xf16>, [
        #const.Broadcast<0 : i64, 16 : i64>,
        #const.Reshape<[16, 1, 1, 1]>,
        #const.Reorder<#NHWC>
    ]

    %IN_MEM_PERM = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x64xf16> -> tensor<1x8x4x64xf16, {order = #NHWC}>

    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 16, 8]
    } inputs(%IN_MEM_PERM : tensor<1x8x4x64xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>

    %SCALE_SHIFT = IE.GroupConvolution(%IN_SHAPE_CAST, %SCALE) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x16x8xf16, {order = #NHWC}>,
        tensor<16x1x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x16x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 64]
    } inputs(%SCALE_SHIFT : tensor<1x16x16x8xf16, {order = #NHWC}>) -> tensor<1x8x4x64xf16, {order = #NHWC}>

    %OUT_MEM_PERM = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW,
        mem_perm = #NWCH
    } : tensor<1x8x4x64xf16, {order = #NHWC}> -> tensor<1x8x4x64xf16>

    return %OUT_MEM_PERM : tensor<1x8x4x64xf16>

    // CHECK-DAG:       [[SCALE:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01>
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[BLOCK_ARG]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x4x8x64xf16> -> tensor<1x64x4x8xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 16, 8]
    // CHECK-SAME:  } inputs([[PERMUTE_CAST]] : tensor<1x64x4x8xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x16x16x8xf16, {order = #NHWC}>

    // CHECK:       [[GROUP_CONVOLUTION:%.+]] = IE.GroupConvolution([[IN_SHAPE_CAST]], [[SCALE]])

    // CHECK:       [[OUT_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 64, 4, 8]
    // CHECK-SAME:  } inputs([[GROUP_CONVOLUTION]] : tensor<1x16x16x8xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x64x4x8xf16, {order = #NHWC}>

    // CHECK:       [[OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[OUT_SHAPE_CAST]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NHCW
    // CHECK-SAME:  } : tensor<1x64x4x8xf16, {order = #NHWC}> -> tensor<1x8x4x64xf16>

    // CHECK:       return [[OUT_MEM_PERMUTE]] : tensor<1x8x4x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @PropagatePermuteQuantizeThroughGroupConv
// CHECK-SAME:        [[BLOCK_ARG:%arg0]]: tensor<1x4x8x64xf16>
func.func @PropagatePermuteQuantizeThroughGroupConv(%arg0: tensor<1x4x8x64xf16>) -> tensor<1x8x4x64xf16> {
    %SCALE = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x1x1xf16>, [
        #const.Broadcast<0 : i64, 16 : i64>,
        #const.Reshape<[16, 1, 1, 1]>,
        #const.Reorder<#NHWC>
    ]

    %IN_MEM_PERM = IE.PermuteQuantize(%arg0) {
        dstElemType = f16,
        dst_order = #NHWC,
        mem_perm = #NCWH,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x4x8x64xf16> -> tensor<1x8x4x64xf16, {order = #NHWC}>

    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 16, 8]
    } inputs(%IN_MEM_PERM : tensor<1x8x4x64xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>

    %SCALE_SHIFT = IE.GroupConvolution(%IN_SHAPE_CAST, %SCALE) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x16x8xf16, {order = #NHWC}>,
        tensor<16x1x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x16x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 64]
    } inputs(%SCALE_SHIFT : tensor<1x16x16x8xf16, {order = #NHWC}>) -> tensor<1x8x4x64xf16, {order = #NHWC}>

    %OUT_MEM_PERM = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW,
        mem_perm = #NWCH
    } : tensor<1x8x4x64xf16, {order = #NHWC}> -> tensor<1x8x4x64xf16>

    return %OUT_MEM_PERM : tensor<1x8x4x64xf16>

    // CHECK-DAG:       [[SCALE:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01>
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[BLOCK_ARG]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x4x8x64xf16> -> tensor<1x64x4x8xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 16, 8]
    // CHECK-SAME:  } inputs([[PERMUTE_CAST]] : tensor<1x64x4x8xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x16x16x8xf16, {order = #NHWC}>

    // CHECK:       [[GROUP_CONVOLUTION:%.+]] = IE.GroupConvolution([[IN_SHAPE_CAST]], [[SCALE]])

    // CHECK:       [[OUT_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 64, 4, 8]
    // CHECK-SAME:  } inputs([[GROUP_CONVOLUTION]] : tensor<1x16x16x8xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x64x4x8xf16, {order = #NHWC}>

    // CHECK:       [[OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[OUT_SHAPE_CAST]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NHCW
    // CHECK-SAME:  } : tensor<1x64x4x8xf16, {order = #NHWC}> -> tensor<1x8x4x64xf16>

    // CHECK:       return [[OUT_MEM_PERMUTE]] : tensor<1x8x4x64xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SkipGroupConvInputNotEqualOutput(%arg0: tensor<1x128x51x64xf16>) -> tensor<1x128x16x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<128x1x9x1xf16, {order = #NHWC}> = dense<1.838680e-03> : tensor<128x1x9x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>
  %1 = IE.GroupConvolution(%0, %cst) {dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [3, 0], strides = [3, 1]} : tensor<1x128x51x64xf16, {order = #NHWC}>, tensor<128x1x9x1xf16, {order = #NHWC}> -> tensor<1x128x16x64xf16, {order = #NHWC}>
  return  %1 : tensor<1x128x16x64xf16, {order = #NHWC}>

    // CHECK:       [[CONST:%.+]] = const.Declare tensor<128x1x9x1xf16,
    // CHECK-SAME:      {order = #NHWC}> = dense<1.838680e-03>
    // CHECK-SAME:      : tensor<128x1x9x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute(%arg0)
    // CHECK-SAME:      {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[MEMPERMUTE]], [[CONST]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [3, 0], strides = [3, 1]}
    // CHECK-SAME:      : tensor<1x128x51x64xf16, {order = #NHWC}>, tensor<128x1x9x1xf16, {order = #NHWC}> -> tensor<1x128x16x64xf16, {order = #NHWC}>

    // CHECK:       return [[GROUPCONV]] : tensor<1x128x16x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SkipGroupConvInputEqualOutput(%arg0: tensor<1x128x51x64xf16>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<128x1x3x3xf16, {order = #NHWC}> = dense<1.838680e-03> : tensor<128x1x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>
  %1 = IE.GroupConvolution(%0, %cst) {dilations = [1, 1], groups = 128 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x51x64xf16, {order = #NHWC}>, tensor<128x1x3x3xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
  return  %1 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:       [[CONST:%.+]] = const.Declare tensor<128x1x3x3xf16,
    // CHECK-SAME:      {order = #NHWC}> = dense<1.838680e-03>
    // CHECK-SAME:      : tensor<128x1x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute(%arg0)
    // CHECK-SAME:      {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[MEMPERMUTE]], [[CONST]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 128 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x128x51x64xf16, {order = #NHWC}>, tensor<128x1x3x3xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:       return [[GROUPCONV]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @PropagatePermuteThroughGroupConvCase2
// CHECK-SAME:        [[BLOCK_ARG:%arg0]]: tensor<1x4x8x64xf16, {order = #NCWH}>
func.func @PropagatePermuteThroughGroupConvCase2(%arg0: tensor<1x4x8x64xf16, {order = #NCWH}>)-> tensor<1x16x16x8xf16, {order = #NHWC}> {
    %SCALE = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x1x1xf16>, [
        #const.Broadcast<0 : i64, 16 : i64>,
        #const.Reshape<[16, 1, 1, 1]>,
        #const.Reorder<#NHWC>
    ]

    %IN_MEM_PERM = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x4x8x64xf16,{order = #NCWH}> -> tensor<1x4x8x64xf16, {order = #NHWC}>

    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 16, 8]
    } inputs(%IN_MEM_PERM : tensor<1x4x8x64xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>

    %SCALE_SHIFT = IE.GroupConvolution(%IN_SHAPE_CAST, %SCALE) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x16x8xf16, {order = #NHWC}>,
        tensor<16x1x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x16x8xf16, {order = #NHWC}>

    return %SCALE_SHIFT : tensor<1x16x16x8xf16, {order = #NHWC}>

    // CHECK-DAG:       [[SCALE:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01>
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[BLOCK_ARG]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x4x8x64xf16, {order = #NCWH}> -> tensor<1x8x4x64xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 16, 8]
    // CHECK-SAME:  } inputs([[PERMUTE_CAST]] : tensor<1x8x4x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x16x16x8xf16, {order = #NHWC}>

    // CHECK:       [[GROUP_CONVOLUTION:%.+]] = IE.GroupConvolution([[IN_SHAPE_CAST]], [[SCALE]])

    // CHECK:       [[OUT_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 4, 64]
    // CHECK-SAME:  } inputs([[GROUP_CONVOLUTION]] : tensor<1x16x16x8xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x8x4x64xf16, {order = #NHWC}>

    // CHECK:       [[OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[OUT_SHAPE_CAST]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  } : tensor<1x8x4x64xf16, {order = #NHWC}> -> tensor<1x4x8x64xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 16, 8]
    // CHECK-SAME:  } inputs([[OUT_MEM_PERMUTE]] : tensor<1x4x8x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x16x16x8xf16, {order = #NHWC}>


    // CHECK:       return [[OUTPUT_SHAPE_CAST]] : tensor<1x16x16x8xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @SkipGroupConvWith2ShapeCastConsumers
// CHECK-SAME:        [[BLOCK_ARG:%arg0]]: tensor<1x4x8x64xf16>
func.func @SkipGroupConvWith2ShapeCastConsumers(%arg0: tensor<1x4x8x64xf16>)
        -> (tensor<1x8x4x64xf16, {order = #NHWC}>, tensor<1x8x4x64xf16, {order = #NHWC}>) {
    %SCALE = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x1x1xf16>, [
        #const.Broadcast<0 : i64, 16 : i64>,
        #const.Reshape<[16, 1, 1, 1]>,
        #const.Reorder<#NHWC>
    ]

    %IN_MEM_PERM = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x64xf16> -> tensor<1x8x4x64xf16, {order = #NHWC}>

    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 16, 8]
    } inputs(%IN_MEM_PERM : tensor<1x8x4x64xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>

    %SCALE_SHIFT = IE.GroupConvolution(%IN_SHAPE_CAST, %SCALE) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x16x8xf16, {order = #NHWC}>,
        tensor<16x1x1x1xf16, {order = #NHWC}>
            -> tensor<1x16x16x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST_1 = IE.ShapeCast {
        shape = [1, 8, 4, 64]
    } inputs(%SCALE_SHIFT : tensor<1x16x16x8xf16, {order = #NHWC}>) -> tensor<1x8x4x64xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST_2 = IE.ShapeCast {
        shape = [1, 8, 4, 64]
    } inputs(%SCALE_SHIFT : tensor<1x16x16x8xf16, {order = #NHWC}>) -> tensor<1x8x4x64xf16, {order = #NHWC}>

    return %OUT_SHAPE_CAST_1, %OUT_SHAPE_CAST_2 : tensor<1x8x4x64xf16, {order = #NHWC}>, tensor<1x8x4x64xf16, {order = #NHWC}>

    // CHECK-DAG:       [[SCALE:%.*]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01>
    // CHECK:       [[IN_MEM_PERM:%.+]] = IE.MemPermute([[BLOCK_ARG]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NCWH
    // CHECK-SAME:  } : tensor<1x4x8x64xf16> -> tensor<1x8x4x64xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 16, 8]
    // CHECK-SAME:  } inputs([[IN_MEM_PERM]] : tensor<1x8x4x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x16x16x8xf16, {order = #NHWC}>

    // CHECK:       [[GROUP_CONVOLUTION:%.+]] = IE.GroupConvolution([[IN_SHAPE_CAST]], [[SCALE]])

    // CHECK:       [[OUT_SHAPE_CAST_1:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 4, 64]
    // CHECK-SAME:  } inputs([[GROUP_CONVOLUTION]] : tensor<1x16x16x8xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x8x4x64xf16, {order = #NHWC}>

    // CHECK:       [[OUT_SHAPE_CAST_2:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 4, 64]
    // CHECK-SAME:  } inputs([[GROUP_CONVOLUTION]] : tensor<1x16x16x8xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x8x4x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_SHAPE_CAST_1]], [[OUT_SHAPE_CAST_2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!qElemType = !quant.uniform<u8:f16, 0.5>
!qElemType1 = !quant.uniform<u8:f16, 0.25>

// CHECK-LABEL: @PropagatePermuteThroughAddWith2QuantCasts
// CHECK-SAME:        [[BLOCK_ARG:%arg0]]: tensor<1x4x8x64x!qElemType>
func.func @PropagatePermuteThroughAddWith2QuantCasts(%arg0: tensor<1x4x8x64x!qElemType>)
    -> (tensor<1x8x4x64x!qElemType1, {order = #NHWC}>, tensor<1x8x4x64x!qElemType1, {order = #NHWC}>)  {
    %IN_MEM_PERM = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x64x!qElemType> -> tensor<1x8x4x64x!qElemType, {order = #NHWC}>

    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 16, 8]
    } inputs(%IN_MEM_PERM : tensor<1x8x4x64x!qElemType, {order = #NHWC}>) -> tensor<1x16x16x8x!qElemType, {order = #NHWC}>

    %ADD = IE.Add(%IN_SHAPE_CAST, %IN_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x16x8x!qElemType, {order = #NHWC}>, tensor<1x16x16x8x!qElemType, {order = #NHWC}>
            -> tensor<1x16x16x8x!qElemType, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 64]
    } inputs(%ADD : tensor<1x16x16x8x!qElemType, {order = #NHWC}>) -> tensor<1x8x4x64x!qElemType, {order = #NHWC}>

    %OUT_QUANT_CAST_1 = IE.QuantizeCast(%OUT_SHAPE_CAST) {
        dstElemType = !qElemType1
    } : tensor<1x8x4x64x!qElemType, {order = #NHWC}>
            -> tensor<1x8x4x64x!qElemType1, {order = #NHWC}>

    %OUT_QUANT_CAST_2 = IE.QuantizeCast(%OUT_SHAPE_CAST) {
        dstElemType = !qElemType1
    } : tensor<1x8x4x64x!qElemType, {order = #NHWC}>
            -> tensor<1x8x4x64x!qElemType1, {order = #NHWC}>

    return %OUT_QUANT_CAST_1, %OUT_QUANT_CAST_2 : tensor<1x8x4x64x!qElemType1, {order = #NHWC}>, tensor<1x8x4x64x!qElemType1, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[BLOCK_ARG]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x4x8x64x!qElemType> -> tensor<1x64x4x8x!qElemType, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 16, 8]
    // CHECK-SAME:  } inputs([[PERMUTE_CAST]] : tensor<1x64x4x8x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x16x16x8x!qElemType, {order = #NHWC}>

    // CHECK:       [[ADD:%.+]] = IE.Add([[IN_SHAPE_CAST]], [[IN_SHAPE_CAST]])

    // CHECK:       [[OUT_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 64, 4, 8]
    // CHECK-SAME:  } inputs([[ADD]] : tensor<1x16x16x8x!qElemType, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x64x4x8x!qElemType, {order = #NHWC}>

    // CHECK:       [[OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[OUT_SHAPE_CAST]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NCWH
    // CHECK-SAME:  } : tensor<1x64x4x8x!qElemType, {order = #NHWC}> -> tensor<1x8x4x64x!qElemType, {order = #NHWC}>

    // CHECK:       [[OUT_QUANT_CAST_1:%.+]] = IE.QuantizeCast([[OUT_MEM_PERMUTE]])
    // CHECK:       [[OUT_QUANT_CAST_2:%.+]] = IE.QuantizeCast([[OUT_MEM_PERMUTE]])

    // CHECK:       return [[OUT_QUANT_CAST_1]], [[OUT_QUANT_CAST_2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK-LABEL: @PropagatePermuteThroughGroupConvWithDifferentShape
func.func @PropagatePermuteThroughGroupConvWithDifferentShape(%arg0: tensor<1x16x16x2xf16, {order = #NHWC}>) -> tensor<1x2x16x16xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<2x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
  %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x16x2xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
  %1 = IE.ShapeCast {shape = [1, 16, 16, 2]} inputs(%0 : tensor<1x2x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x2xf16, {order = #NHWC}>
  %2 = IE.GroupConvolution(%1, %cst) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x2xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x2xf16, {order = #NHWC}>
  %6 = IE.ShapeCast {shape = [1, 16, 8, 4]} inputs(%2 : tensor<1x16x16x2xf16, {order = #NHWC}>) -> tensor<1x16x8x4xf16, {order = #NHWC}>
  %8 = IE.ShapeCast {shape = [1, 2, 16, 16]} inputs(%6 : tensor<1x16x8x4xf16, {order = #NHWC}>) -> tensor<1x2x16x16xf16, {order = #NHWC}>
  return %8 : tensor<1x2x16x16xf16, {order = #NHWC}>

  // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<2x1x1x1xf16>
  // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution(%arg0, [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x2xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x2xf16, {order = #NHWC}>
  // CHECK:       [[RET:%.+]] = IE.MemPermute([[GROUP_CONV]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x16x2xf16, {order = #NHWC}> -> tensor<1x2x16x16xf16, {order = #NHWC}>
  // CHECK:       return [[RET]] : tensor<1x2x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!qElemType = !quant.uniform<u8:f16, 0.43661021812289369:128>
// CHECK-LABEL: @PropagatePermuteThroughAddWithDifferentType
func.func @PropagatePermuteThroughAddWithDifferentType(%arg0:tensor<1x768x128x1xf16, {order = #NHWC}>) -> tensor<1x128x768x1x!quant.uniform<u8:f16, 0.43661021812289369:128>, {order = #NHWC}> {
  %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} : tensor<1x768x128x1xf16, {order = #NHWC}> -> tensor<1x128x768x1xf16, {order = #NHWC}>
  %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x128x768x1xf16, {order = #NHWC}>, tensor<1x128x768x1xf16, {order = #NHWC}> -> tensor<1x128x768x1x!qElemType, {order = #NHWC}>
  return %1 : tensor<1x128x768x1x!qElemType, {order = #NHWC}>
  // CHECK:       [[CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 128, 768, 1]} inputs(%arg0 : tensor<1x768x128x1xf16, {order = #NHWC}>) -> tensor<1x128x768x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_RET:%.+]] = IE.Add([[CAST_IN]], [[CAST_IN]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
  // CHECK-SAME:                    tensor<1x128x768x1xf16, {order = #NHWC}>, tensor<1x128x768x1xf16, {order = #NHWC}> -> tensor<1x128x768x1x!qElemType, {order = #NHWC}>
  // CHECK:       [[CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 768, 128, 1]} inputs([[ADD_RET]] : tensor<1x128x768x1x!qElemType, {order = #NHWC}>) -> tensor<1x768x128x1x!qElemType, {order = #NHWC}>
  // CHECK:       [[RET:%.+]] = IE.MemPermute([[CAST_OUT]]) {dst_order = #NHWC, mem_perm = #NWHC} : tensor<1x768x128x1x!qElemType, {order = #NHWC}> -> tensor<1x128x768x1x!qElemType, {order = #NHWC}>
  // CHECK:       return [[RET]] : tensor<1x128x768x1x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagatePermuteThroughAvgPool
func.func @PropagatePermuteThroughAvgPool(%arg0: tensor<1x64x64x192xf16>) -> tensor<1x64x64x192xf16, {order = #NHWC}> {
  %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x64x64x192xf16> -> tensor<1x64x64x192xf16, {order = #NHWC}>
  %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.01000213623046875 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x192xf16, {order = #NHWC}>

  // CHECK:       [[PERM_CAST:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x64x192xf16> -> tensor<1x192x64x64xf16, {order = #NHWC}>
  // CHECK:       [[AVG_POOL:%.*]] = IE.AvgPool([[PERM_CAST]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.01000213623046875 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x192x64x64xf16, {order = #NHWC}> -> tensor<1x192x64x64xf16, {order = #NHWC}>
  // CHECK:       [[MEM_PERM:%.*]] = IE.MemPermute([[AVG_POOL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x192x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
  // CHECK:       return [[MEM_PERM]] : tensor<1x64x64x192xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagatePermuteQuantizeThroughAvgPool
func.func @PropagatePermuteQuantizeThroughAvgPool(%arg0: tensor<1x64x64x192xf16>) -> tensor<1x64x64x192xf16, {order = #NHWC}> {
  %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x64x64x192xf16> -> tensor<1x64x64x192xf16, {order = #NHWC}>
  %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.01000213623046875 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x192xf16, {order = #NHWC}>

  // CHECK:       [[PERM_CAST_IN:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x64x192xf16> -> tensor<1x192x64x64xf16, {order = #NHWC}>
  // CHECK:       [[AVG_POOL:%.*]] = IE.AvgPool([[PERM_CAST_IN]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.01000213623046875 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x192x64x64xf16, {order = #NHWC}> -> tensor<1x192x64x64xf16, {order = #NHWC}>
  // CHECK:       [[MEM_PERMUTE:%.*]] = IE.MemPermute([[AVG_POOL]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x192x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
  // CHECK:       return [[MEM_PERMUTE]] : tensor<1x64x64x192xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipAvgPoolNonEltwise
func.func @SkipAvgPoolNonEltwise(%arg0: tensor<1x64x64x192xf16>) -> tensor<1x64x64x192xf16, {order = #NHWC}> {
  %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x64x64x192xf16> -> tensor<1x64x64x192xf16, {order = #NHWC}>
  %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.LeakyRelu<negative_slope = 0.01000213623046875 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
  return %1 : tensor<1x64x64x192xf16, {order = #NHWC}>

  // CHECK:       [[MEM_PERM:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x64x64x192xf16> -> tensor<1x64x64x192xf16, {order = #NHWC}>
  // CHECK:       [[AVG_POOL:%.*]] = IE.AvgPool([[MEM_PERM]]) {exclude_pads, kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.LeakyRelu<negative_slope = 0.01000213623046875 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x64x192xf16, {order = #NHWC}> -> tensor<1x64x64x192xf16, {order = #NHWC}>
  // CHECK:       return [[AVG_POOL]] : tensor<1x64x64x192xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16:1, {0.010182879952823415:128,0.0086226547465604892:128,0.0071560340769150676:128,0.0071187512547362082:128}>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagatePermuteThrough2EltwiseGroupConvs
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x4x8x1x1xf16>
func.func @PropagatePermuteThrough2EltwiseGroupConvs(%input: tensor<1x4x8x1x1xf16>) -> tensor<1x4x8x1x1xf16> {
  %cst = const.Declare tensor<4x1x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %cst_0 = const.Declare tensor<4x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.AffineReshape(%input) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 4, 8, 1]} : tensor<1x4x8x1x1xf16> -> tensor<1x4x8x1xf16>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 4, 2, 4]} : tensor<1x4x8x1xf16> -> tensor<1x4x2x4xf16>
  %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x2x4xf16> -> tensor<1x4x2x4xf16, {order = #NHWC}>
  %3 = IE.GroupConvolution(%2, %cst_0) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x2x4xf16, {order = #NHWC}>, tensor<4x1x1x1xf16, {order = #NHWC}> -> tensor<1x4x2x4x!qElemType1, {order = #NHWC}>
  %4 = IE.GroupConvolution(%3, %cst) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x2x4x!qElemType1, {order = #NHWC}>, tensor<4x1x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4x2x4xf16, {order = #NHWC}>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4, 8, 1]} : tensor<1x4x2x4xf16, {order = #NHWC}> -> tensor<1x4x8x1xf16, {order = #NHWC}>
  %6 = IE.MemPermute(%5) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x1xf16, {order = #NHWC}> -> tensor<1x4x8x1xf16>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 4, 8, 1, 1]} : tensor<1x4x8x1xf16> -> tensor<1x4x8x1x1xf16>
  return %7 : tensor<1x4x8x1x1xf16>

  // CHECK:       [[CONST:%.*]] = const.Declare tensor<4x1x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

  // CHECK:       [[CONST_0:%.*]] = const.Declare tensor<4x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1x1x1xf16>, [#const.Reorder<#NHWC>]

  // CHECK:       [[PERMUTECAST_INPUT_1:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 4, 2, 4]} : tensor<1x4x8x1x1xf16> -> tensor<1x4x2x4xf16>

  // CHECK:       [[SHAPECAST_INPUT_1:%.+]] = IE.PermuteCast([[PERMUTECAST_INPUT_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x2x4xf16> -> tensor<1x4x4x2xf16, {order = #NHWC}>

  // CHECK:       [[GROUPCONV_INPUT_1:%.+]] = IE.ShapeCast {shape = [1, 4, 2, 4]} inputs([[SHAPECAST_INPUT_1]] : tensor<1x4x4x2xf16, {order = #NHWC}>) -> tensor<1x4x2x4xf16, {order = #NHWC}>

  // CHECK:       [[GROUPCONV_INPUT_2:%.+]] = IE.GroupConvolution([[GROUPCONV_INPUT_1]], [[CONST_0]]) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x2x4xf16, {order = #NHWC}>, tensor<4x1x1x1xf16, {order = #NHWC}> -> tensor<1x4x2x4x!qElemType1, {order = #NHWC}>

  // CHECK:       [[SHAPECAST_INPUT_2:%.+]] = IE.GroupConvolution([[GROUPCONV_INPUT_2]], [[CONST]]) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x2x4x!qElemType1, {order = #NHWC}>, tensor<4x1x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4x2x4xf16, {order = #NHWC}>

  // CHECK:       [[MEMPERMUTE_INPUT_1:%.+]] = IE.ShapeCast {shape = [1, 4, 4, 2]} inputs([[SHAPECAST_INPUT_2]] : tensor<1x4x2x4xf16, {order = #NHWC}>) -> tensor<1x4x4x2xf16, {order = #NHWC}>

  // CHECK:       [[AFFINERESHAPE_INPUT_1:%.+]] = IE.MemPermute([[MEMPERMUTE_INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x4x2xf16, {order = #NHWC}> -> tensor<1x4x2x4xf16, {order = #NHWC}>

  // CHECK:       [[MEMPERMUTE_INPUT_2:%.+]] = IE.AffineReshape([[AFFINERESHAPE_INPUT_1]]) {
  // CHECK-SAME:         shape_value = [1, 4, 8, 1]
  // CHECK-SAME:     } : tensor<1x4x2x4xf16, {order = #NHWC}> -> tensor<1x4x8x1xf16, {order = #NHWC}>

  // CHECK:       [[AFFINERESHAPE_INPUT_2:%.+]] = IE.MemPermute([[MEMPERMUTE_INPUT_2]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x1xf16, {order = #NHWC}> -> tensor<1x4x8x1xf16>

  // CHECK:       [[RET:%.+]] = IE.AffineReshape([[AFFINERESHAPE_INPUT_2]]) {
  // CHECK-SAME:         shape_value = [1, 4, 8, 1, 1]
  // CHECK-SAME:     } : tensor<1x4x8x1xf16> -> tensor<1x4x8x1x1xf16>

  // CHECK:       return [[RET:%.+]] : tensor<1x4x8x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16:1, {0.010182879952823415:128,0.0086226547465604892:128,0.0071560340769150676:128,0.0071187512547362082:128}>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagatePermuteThrough3EltwiseGroupConvs
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x4x8x1x1xf16>
func.func @PropagatePermuteThrough3EltwiseGroupConvs(%input: tensor<1x4x8x1x1xf16>) -> tensor<1x4x8x1x1xf16> {
  %cst = const.Declare tensor<4x1x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %cst_0 = const.Declare tensor<4x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.AffineReshape(%input) {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 4, 8, 1]} : tensor<1x4x8x1x1xf16> -> tensor<1x4x8x1xf16>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 4, 2, 4]} : tensor<1x4x8x1xf16> -> tensor<1x4x2x4xf16>
  %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x2x4xf16> -> tensor<1x4x2x4xf16, {order = #NHWC}>
  %3 = IE.GroupConvolution(%2, %cst_0) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x2x4xf16, {order = #NHWC}>, tensor<4x1x1x1xf16, {order = #NHWC}> -> tensor<1x4x2x4x!qElemType1, {order = #NHWC}>
  %4 = IE.GroupConvolution(%3, %cst) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x2x4x!qElemType1, {order = #NHWC}>, tensor<4x1x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4x2x4xf16, {order = #NHWC}>
  %5 = IE.GroupConvolution(%4, %cst_0) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x2x4xf16, {order = #NHWC}>, tensor<4x1x1x1xf16, {order = #NHWC}> -> tensor<1x4x2x4xf16, {order = #NHWC}>
  %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4, 8, 1]} : tensor<1x4x2x4xf16, {order = #NHWC}> -> tensor<1x4x8x1xf16, {order = #NHWC}>
  %7 = IE.MemPermute(%6) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x1xf16, {order = #NHWC}> -> tensor<1x4x8x1xf16>
  %8 = IE.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 4, 8, 1, 1]} : tensor<1x4x8x1xf16> -> tensor<1x4x8x1x1xf16>
  return %8 : tensor<1x4x8x1x1xf16>

  // CHECK:       [[CONST:%.*]] = const.Declare tensor<4x1x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

  // CHECK:       [[CONST_0:%.*]] = const.Declare tensor<4x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1x1x1xf16>, [#const.Reorder<#NHWC>]

  // CHECK:       [[PERMUTECAST_INPUT_1:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 4, 2, 4]} : tensor<1x4x8x1x1xf16> -> tensor<1x4x2x4xf16>

  // CHECK:       [[SHAPECAST_INPUT_1:%.+]] = IE.PermuteCast([[PERMUTECAST_INPUT_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x2x4xf16> -> tensor<1x4x4x2xf16, {order = #NHWC}>

  // CHECK:       [[GROUPCONV_INPUT_1:%.+]] = IE.ShapeCast {shape = [1, 4, 2, 4]} inputs([[SHAPECAST_INPUT_1]] : tensor<1x4x4x2xf16, {order = #NHWC}>) -> tensor<1x4x2x4xf16, {order = #NHWC}>

  // CHECK:       [[GROUPCONV_INPUT_2:%.+]] = IE.GroupConvolution([[GROUPCONV_INPUT_1]], [[CONST_0]]) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x2x4xf16, {order = #NHWC}>, tensor<4x1x1x1xf16, {order = #NHWC}> -> tensor<1x4x2x4x!qElemType1, {order = #NHWC}>

  // CHECK:       [[GROUPCONV_INPUT_3:%.+]] = IE.GroupConvolution([[GROUPCONV_INPUT_2]], [[CONST]]) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x2x4x!qElemType1, {order = #NHWC}>, tensor<4x1x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4x2x4xf16, {order = #NHWC}>

  // CHECK:       [[SHAPECAST_INPUT_2:%.+]] = IE.GroupConvolution([[GROUPCONV_INPUT_3]], [[CONST_0]]) {dilations = [1, 1], groups = 4 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x2x4xf16, {order = #NHWC}>, tensor<4x1x1x1xf16, {order = #NHWC}> -> tensor<1x4x2x4xf16, {order = #NHWC}>

  // CHECK:       [[MEMPERMUTE_INPUT_1:%.+]] = IE.ShapeCast {shape = [1, 4, 4, 2]} inputs([[SHAPECAST_INPUT_2]] : tensor<1x4x2x4xf16, {order = #NHWC}>) -> tensor<1x4x4x2xf16, {order = #NHWC}>

  // CHECK:       [[AFFINERESHAPE_INPUT_1:%.+]] = IE.MemPermute([[MEMPERMUTE_INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x4x2xf16, {order = #NHWC}> -> tensor<1x4x2x4xf16, {order = #NHWC}>

  // CHECK:       [[MEMPERMUTE_INPUT_2:%.+]] = IE.AffineReshape([[AFFINERESHAPE_INPUT_1]]) {
  // CHECK-SAME:         shape_value = [1, 4, 8, 1]
  // CHECK-SAME:     } : tensor<1x4x2x4xf16, {order = #NHWC}> -> tensor<1x4x8x1xf16, {order = #NHWC}>

  // CHECK:       [[AFFINERESHAPE_INPUT_2:%.+]] = IE.MemPermute([[MEMPERMUTE_INPUT_2]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x1xf16, {order = #NHWC}> -> tensor<1x4x8x1xf16>

  // CHECK:       [[RET:%.+]] = IE.AffineReshape([[AFFINERESHAPE_INPUT_2]]) {
  // CHECK-SAME:         shape_value = [1, 4, 8, 1, 1]
  // CHECK-SAME:     } : tensor<1x4x8x1xf16> -> tensor<1x4x8x1x1xf16>

  // CHECK:       return [[RET:%.+]] : tensor<1x4x8x1x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Skip2ndNonEltwiseGroupConv
// CHECK-SAME:      [[INPUT:%.*]]: tensor<1x128x51x64xf16>
func.func @Skip2ndNonEltwiseGroupConv(%input: tensor<1x128x51x64xf16>) -> tensor<1x128x51x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<128x1x3x3xf16, {order = #NHWC}> = dense<1.838680e-03> : tensor<128x1x3x3xf16>, [#const.Reorder<#NHWC>]
  %cst_0 = const.Declare tensor<128x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.MemPermute(%input) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x128x51x64xf16> -> tensor<1x128x51x64xf16, {order = #NHWC}>
  %1 = IE.GroupConvolution(%0, %cst_0) {dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x51x64xf16, {order = #NHWC}>, tensor<128x1x1x1xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
  %2 = IE.GroupConvolution(%1, %cst) {dilations = [1, 1], groups = 128 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x51x64xf16, {order = #NHWC}>, tensor<128x1x3x3xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>
  return  %2 : tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:       [[CONST:%.+]] = const.Declare tensor<128x1x3x3xf16,
    // CHECK-SAME:      {order = #NHWC}> = dense<1.838680e-03>
    // CHECK-SAME:      : tensor<128x1x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[CONST_0:%.+]] = const.Declare tensor<128x1x1x1xf16,
    // CHECK-SAME:      {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<128x1x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[SHAPECAST_INPUT_1:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x128x51x64xf16> -> tensor<1x64x128x51xf16, {order = #NHWC}>

    // CHECK:       [[GROUPCONV_INPUT_1:%.+]] = IE.ShapeCast {shape = [1, 128, 51, 64]} inputs([[SHAPECAST_INPUT_1]] : tensor<1x64x128x51xf16, {order = #NHWC}>) -> tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:       [[SHAPECAST_INPUT_2:%.+]] = IE.GroupConvolution([[GROUPCONV_INPUT_1]], [[CONST_0]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 128 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x128x51x64xf16, {order = #NHWC}>, tensor<128x1x1x1xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:       [[MEMPERMUTE_INPUT_1:%.+]] = IE.ShapeCast {shape = [1, 64, 128, 51]} inputs([[SHAPECAST_INPUT_2]] : tensor<1x128x51x64xf16, {order = #NHWC}>) -> tensor<1x64x128x51xf16, {order = #NHWC}>

    // CHECK:       [[GROUPCONV_INPUT_2:%.+]] = IE.MemPermute([[MEMPERMUTE_INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x64x128x51xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:       [[RET:%.+]] = IE.GroupConvolution([[GROUPCONV_INPUT_2]], [[CONST]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 128 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x128x51x64xf16, {order = #NHWC}>, tensor<128x1x3x3xf16, {order = #NHWC}> -> tensor<1x128x51x64xf16, {order = #NHWC}>

    // CHECK:       return [[RET]] : tensor<1x128x51x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipGroupConvIfNotEligible
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x880x1x960xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x880x1x1xf16, {order = #NHWC}>
func.func @SkipGroupConvIfNotEligible(%arg0: tensor<1x880x1x960xf16>, %arg1: tensor<1x880x1x1xf16, {order = #NHWC}>) -> tensor<1x880x1x960xf16, {order = #NHWC}> {
    %CONST_ADD = const.Declare tensor<1x880x1x960xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x880x1x960xf16, {order = #NHWC}>

    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [880, 1, 1, 1]
    } inputs(%arg1 : tensor<1x880x1x1xf16, {order = #NHWC}>) -> tensor<880x1x1x1xf16, {order = #NHWC}>


    %IN_MEM_PERM = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x880x1x960xf16> -> tensor<1x880x1x960xf16, {order = #NHWC}>

    %ADD = IE.Add(%IN_MEM_PERM, %CONST_ADD) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x880x1x960xf16, {order = #NHWC}>, tensor<1x880x1x960xf16, {order = #NHWC}>
            -> tensor<1x880x1x960xf16, {order = #NHWC}>

    %SCALE_SHIFT = IE.GroupConvolution(%ADD, %IN_SHAPE_CAST) {
        dilations = [1, 1],
        groups = 880 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x880x1x960xf16, {order = #NHWC}>,
        tensor<880x1x1x1xf16, {order = #NHWC}>
            -> tensor<1x880x1x960xf16, {order = #NHWC}>

    return  %SCALE_SHIFT : tensor<1x880x1x960xf16, {order = #NHWC}>

    // CHECK:       [[CONST:%.+]] = const.Declare tensor<1x880x1x960xf16,
    // CHECK-SAME:      {order = #NHWC}> = dense<1.250000e-01>
    // CHECK-SAME:      : tensor<1x880x1x960xf16, {order = #NHWC}>

    // CHECK:       [[SHAPECAST_INPUT:%.+]] = IE.ShapeCast {shape = [880, 1, 1, 1]} inputs([[INPUT2]] : tensor<1x880x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<880x1x1x1xf16, {order = #NHWC}>

    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x880x1x960xf16> -> tensor<1x880x1x960xf16, {order = #NHWC}>

    // CHECK:       [[IN_ADD:%.+]] = IE.Add([[MEM_PERMUTE]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x880x1x960xf16, {order = #NHWC}>, tensor<1x880x1x960xf16, {order = #NHWC}> -> tensor<1x880x1x960xf16, {order = #NHWC}>

    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[IN_ADD]], [[SHAPECAST_INPUT]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 880 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x880x1x960xf16, {order = #NHWC}>, tensor<880x1x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x880x1x960xf16, {order = #NHWC}>

    // CHECK:       return [[GROUP_CONV]] : tensor<1x880x1x960xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @PropagatePermuteThroughAvgPoolWithGroupConvUser([[ARG0:%.+]]: tensor<1x880x1x960xf16>, [[ARG1:%.+]]: tensor<1x880x1x1xf16, {order = #NHWC}>)
func.func @PropagatePermuteThroughAvgPoolWithGroupConvUser(%arg0: tensor<1x880x1x960xf16>, %arg1: tensor<1x880x1x1xf16, {order = #NHWC}>) -> tensor<1x880x1x960xf16, {order = #NHWC}> {
    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [880, 1, 1, 1]
    } inputs(%arg1 : tensor<1x880x1x1xf16, {order = #NHWC}>) -> tensor<880x1x1x1xf16, {order = #NHWC}>


    %IN_MEM_PERM = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x880x1x960xf16> -> tensor<1x880x1x960xf16, {order = #NHWC}>

    %AVG_POOL = IE.AvgPool(%IN_MEM_PERM) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
                                            post_op = #IE.LeakyRelu<negative_slope = 0.01000213623046875 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
                                            : tensor<1x880x1x960xf16, {order = #NHWC}> -> tensor<1x880x1x960xf16, {order = #NHWC}>

    %SCALE_SHIFT = IE.GroupConvolution(%AVG_POOL, %IN_SHAPE_CAST) {
        dilations = [1, 1],
        groups = 880 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x880x1x960xf16, {order = #NHWC}>,
        tensor<880x1x1x1xf16, {order = #NHWC}>
            -> tensor<1x880x1x960xf16, {order = #NHWC}>

    return %SCALE_SHIFT : tensor<1x880x1x960xf16, {order = #NHWC}>

    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [880, 1, 1, 1]} inputs([[ARG1]] : tensor<1x880x1x1xf16, {order = #NHWC}>)
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x880x1x960xf16>

    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[PERMUTE_CAST]])
    // CHECK-SAME:                         -> tensor<1x960x880x1xf16, {order = #NHWC}>
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[AVG_POOL]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:                          : tensor<1x960x880x1xf16, {order = #NHWC}> -> tensor<1x880x1x960xf16, {order = #NHWC}>

    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[MEM_PERMUTE]], [[SHAPE_CAST]]) {dilations = [1, 1], groups = 880 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

    // CHECK:       return [[GROUP_CONV]] : tensor<1x880x1x960xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MovePermutePostMultiply
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x3x256x256xf16>,
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256xf16>
func.func @MovePermutePostMultiply(%arg0: tensor<1x3x256x256xf16>, %arg1: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}>
            -> tensor<1x3x256x256xf16, {order = #NHWC}>
    return %2 : tensor<1x3x256x256xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x3x256xf16, {order = #NHWC}>, tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x3x256x256xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MovePermuteQuantizePostMultiply
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x3x256x256xf16>,
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256xf16>
func.func @MovePermuteQuantizePostMultiply(%arg0: tensor<1x3x256x256xf16>, %arg1: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
            : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %1 = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
            : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}>
            -> tensor<1x3x256x256xf16, {order = #NHWC}>
    return %2 : tensor<1x3x256x256xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x3x256xf16, {order = #NHWC}>, tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x256x256xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MovePermuteQuantizePostMultiplyDifferentShapeInputs
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x20x20x64xf16>,
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x20x20x1xf16>
func.func @MovePermuteQuantizePostMultiplyDifferentShapeInputs(%arg0: tensor<1x20x20x64xf16>, %arg1: tensor<1x20x20x1xf16>) -> tensor<1x20x20x64xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
            : tensor<1x20x20x64xf16> -> tensor<1x20x20x64xf16, {order = #NHWC}>
    %1 = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
            : tensor<1x20x20x1xf16> -> tensor<1x20x20x1xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x20x20x64xf16, {order = #NHWC}>, tensor<1x20x20x1xf16, {order = #NHWC}>
            -> tensor<1x20x20x64xf16, {order = #NHWC}>
    return %2 : tensor<1x20x20x64xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x20x20x64xf16> -> tensor<1x64x20x20xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x20x20x1xf16> -> tensor<1x1x20x20xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x20x20xf16, {order = #NHWC}>, tensor<1x1x20x20xf16, {order = #NHWC}> -> tensor<1x64x20x20xf16, {order = #NHWC}>
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x64x20x20xf16, {order = #NHWC}> -> tensor<1x20x20x64xf16, {order = #NHWC}>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x20x20x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MovePermuteQuantizePostMultiplyAndAdd
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x24x1x64xf16>,
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x24x1x64xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x24x1x64xf16>,
// CHECK-SAME:        [[INPUT3:%arg[0-9]]]: tensor<1x24x1x64xf16>
func.func @MovePermuteQuantizePostMultiplyAndAdd(%arg0: tensor<1x24x1x64xf16>, %arg1: tensor<1x24x1x64xf16>, %arg2: tensor<1x24x1x64xf16>, %arg3: tensor<1x24x1x64xf16>) -> tensor<1x24x1x64xf16> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x24x1x64xf16> -> tensor<1x24x1x64xf16, {order = #NHWC}>
    %1 = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x24x1x64xf16> -> tensor<1x24x1x64xf16, {order = #NHWC}>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x1x64xf16, {order = #NHWC}>, tensor<1x24x1x64xf16, {order = #NHWC}> -> tensor<1x24x1x64xf16, {order = #NHWC}>

    %3 = IE.PermuteQuantize(%arg2) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x24x1x64xf16> -> tensor<1x24x1x64xf16, {order = #NHWC}>
    %4 = IE.PermuteQuantize(%arg3) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x24x1x64xf16> -> tensor<1x24x1x64xf16, {order = #NHWC}>
    %5 = IE.Multiply(%3, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x1x64xf16, {order = #NHWC}>, tensor<1x24x1x64xf16, {order = #NHWC}> -> tensor<1x24x1x64xf16, {order = #NHWC}>

    %6 = IE.Add(%2, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x24x1x64xf16, {order = #NHWC}>, tensor<1x24x1x64xf16, {order = #NHWC}> -> tensor<1x24x1x64xf16, {order = #NHWC}>
    %7 = IE.MemPermute(%6) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x24x1x64xf16, {order = #NHWC}> -> tensor<1x24x1x64xf16>

    return  %7 : tensor<1x24x1x64xf16>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x24x1x64xf16> -> tensor<1x64x24x1xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x24x1x64xf16> -> tensor<1x64x24x1xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY_0:%.+]] = IE.Multiply([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x24x1xf16, {order = #NHWC}>, tensor<1x64x24x1xf16, {order = #NHWC}> -> tensor<1x64x24x1xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_2:%.+]] = IE.PermuteCast([[INPUT2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x24x1x64xf16> -> tensor<1x64x24x1xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_3:%.+]] = IE.PermuteCast([[INPUT3]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x24x1x64xf16> -> tensor<1x64x24x1xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY_1:%.+]] = IE.Multiply([[PERMUTE_CAST_2]], [[PERMUTE_CAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x24x1xf16, {order = #NHWC}>, tensor<1x64x24x1xf16, {order = #NHWC}> -> tensor<1x64x24x1xf16, {order = #NHWC}>

    // CHECK:       [[ADD:%.+]] = IE.Add([[MULTIPLY_0]], [[MULTIPLY_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x24x1xf16, {order = #NHWC}>, tensor<1x64x24x1xf16, {order = #NHWC}> -> tensor<1x64x24x1xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_4:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x24x1xf16, {order = #NHWC}> -> tensor<1x24x1x64xf16>

    // CHECK:       return [[PERMUTE_CAST_4]] : tensor<1x24x1x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MovePermutePostSubtract
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x3x256x256xf16>,
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256xf16>
func.func @MovePermutePostSubtract(%arg0: tensor<1x3x256x256xf16>, %arg1: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %2 = IE.Subtract(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}>
            -> tensor<1x3x256x256xf16, {order = #NHWC}>
    return %2 : tensor<1x3x256x256xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[SUBTRACT:%.+]] = IE.Subtract([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x3x256xf16, {order = #NHWC}>, tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[SUBTRACT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x3x256x256xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MovePermuteQuantizePostSubtract
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x3x256x256xf16>,
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256xf16>
func.func @MovePermuteQuantizePostSubtract(%arg0: tensor<1x3x256x256xf16>, %arg1: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf16, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
            : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %1 = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
            : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %2 = IE.Subtract(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}>
            -> tensor<1x3x256x256xf16, {order = #NHWC}>
    return %2 : tensor<1x3x256x256xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x256x256xf16> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[SUBTRACT:%.+]] = IE.Subtract([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x3x256xf16, {order = #NHWC}>, tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x256x3x256xf16, {order = #NHWC}>
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[SUBTRACT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x256x3x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MovePermuteQuantizePostGroupConvWithPermuteOutShapeInconsistent
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x128x256x3xf16>
func.func @MovePermuteQuantizePostGroupConvWithPermuteOutShapeInconsistent(%arg0: tensor<1x128x256x3xf16>) -> tensor<1x3x128x256xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x1x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x128x256x3xf16> -> tensor<1x128x256x3xf16, {order = #NHWC}>
    %1 = IE.ShapeCast {shape = [1, 48, 128, 16]} inputs(%0 : tensor<1x128x256x3xf16, {order = #NHWC}>) -> tensor<1x48x128x16xf16, {order = #NHWC}>
    %2 = IE.GroupConvolution(%1, %cst) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x128x16xf16, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}> -> tensor<1x48x128x16xf16, {order = #NHWC}>
    %3 = IE.ShapeCast {shape = [1, 3, 128, 256]} inputs(%2 : tensor<1x48x128x16xf16, {order = #NHWC}>) -> tensor<1x3x128x256xf16, {order = #NHWC}>
    %4 = IE.ShapeCast {shape = [1, 128, 256, 3]} inputs(%3 : tensor<1x3x128x256xf16, {order = #NHWC}>) -> tensor<1x128x256x3xf16, {order = #NHWC}>
    %5 = IE.MemPermute(%4) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x128x256x3xf16, {order = #NHWC}> -> tensor<1x3x128x256xf16, {order = #NHWC}>

    return  %5 : tensor<1x3x128x256xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x1x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x128x256x3xf16> -> tensor<1x3x128x256xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 128, 16]} inputs([[PERMUTE_CAST]] : tensor<1x3x128x256xf16, {order = #NHWC}>) -> tensor<1x48x128x16xf16, {order = #NHWC}>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[SHAPE_CAST]], [[WEIGHTS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x128x16xf16, {order = #NHWC}>, tensor<48x1x1x1xf16, {order = #NHWC}> -> tensor<1x48x128x16xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 128, 256]} inputs([[GROUP_CONV]] : tensor<1x48x128x16xf16, {order = #NHWC}>) -> tensor<1x3x128x256xf16, {order = #NHWC}>

    // CHECK:       return [[SHAPE_CAST_OUT]] : tensor<1x3x128x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotMovePermutePostFP32Add
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x256x256xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x256x256xf16>
func.func @DoNotMovePermutePostFP32Add(%arg0: tensor<1x3x256x256xf16>, %arg1: tensor<1x3x256x256xf16>) -> tensor<1x3x256x256xf32, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}>
            -> tensor<1x3x256x256xf32, {order = #NHWC}>
    return %2 : tensor<1x3x256x256xf32, {order = #NHWC}>

    // CHECK:       [[MEMPERMUTE_1:%.+]] = IE.MemPermute([[INPUT1]]
    // CHECK:       [[MEMPERMUTE_2:%.+]] = IE.MemPermute([[INPUT2]]
    // CHECK:       [[ADD:%.+]] = IE.Add([[MEMPERMUTE_1]], [[MEMPERMUTE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x3x256x256xf32, {order = #NHWC}>
    // CHECK:       return [[ADD]] : tensor<1x3x256x256xf32, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL: @MoveDifferentInLayoutPermutesPostAdd
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1024x49x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x49x1024xf16, {order = #NHWC}>
func.func @MoveDifferentInLayoutPermutesPostAdd(%arg0: tensor<1x1024x49x1xf16, {order = #NHWC}>, %arg1: tensor<1x1x49x1024xf16, {order = #NHWC}>) -> tensor<1x49x1x1024xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1024x1024x1x1xf16, {order = #NHWC}> = dense<0.230000e+00> : tensor<1024x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x49x1xf16, {order = #NHWC}>, tensor<1024x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #map} : tensor<1x1024x49x1xf16, {order = #NHWC}> -> tensor<49x1024x1x1xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 49, 1, 1024]} : tensor<49x1024x1x1xf16> -> tensor<1x49x1x1024xf16>
    %3 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x49x1x1024xf16> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    %4 = IE.ShapeCast {shape = [1, 1024, 49, 1]} inputs(%arg1 : tensor<1x1x49x1024xf16, {order = #NHWC}>) -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    %5 = IE.MemPermute(%4) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1024x49x1xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    %6 = IE.Add(%5, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x49x1x1024xf16, {order = #NHWC}>, tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    return %6 : tensor<1x49x1x1024xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1024x1024x1x1xf16, {order = #NHWC}> = dense<2.299800e-01> : tensor<1024x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x1024x49x1xf16, {order = #NHWC}>, tensor<1024x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK-SAME:      : tensor<1x1x49x1024xf16, {order = #NHWC}> -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[PERMUTE_CAST_0]], [[CONV]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x1024x49x1xf16, {order = #NHWC}>, tensor<1x1024x49x1xf16, {order = #NHWC}> -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #map}
    // CHECK-SAME:      : tensor<1x1024x49x1xf16, {order = #NHWC}> -> tensor<49x1024x1x1xf16>
    // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[PERMUTE_CAST_1]]) {
    // CHECK-SAME:      shape_value = [1, 49, 1, 1024]}
    // CHECK-SAME:      : tensor<49x1024x1x1xf16> -> tensor<1x49x1x1024xf16>
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[AFFINE_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x49x1x1024xf16> -> tensor<1x49x1x1024xf16, {order = #NHWC}>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x49x1x1024xf16, {order = #NHWC}>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL: @NotMoveDifferentInLayoutPermutesPostAdd
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1024x49x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x2x512x49xf16, {order = #NHWC}>
func.func @NotMoveDifferentInLayoutPermutesPostAdd(%arg0: tensor<1x1024x49x1xf16, {order = #NHWC}>, %arg1: tensor<1x2x512x49xf16, {order = #NHWC}>) -> tensor<1x49x1x1024xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1024x1024x1x1xf16, {order = #NHWC}> = dense<0.230000e+00> : tensor<1024x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x49x1xf16, {order = #NHWC}>, tensor<1024x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #map} : tensor<1x1024x49x1xf16, {order = #NHWC}> -> tensor<49x1024x1x1xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 49, 1, 1024]} : tensor<49x1024x1x1xf16> -> tensor<1x49x1x1024xf16>
    %3 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x49x1x1024xf16> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    %4 = IE.ShapeCast {shape = [1, 1024, 49, 1]} inputs(%arg1 : tensor<1x2x512x49xf16, {order = #NHWC}>) -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    %5 = IE.MemPermute(%4) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1024x49x1xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    %6 = IE.Add(%5, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x49x1x1024xf16, {order = #NHWC}>, tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    return %6 : tensor<1x49x1x1024xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1024x1024x1x1xf16, {order = #NHWC}> = dense<2.299800e-01> : tensor<1024x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x1024x49x1xf16, {order = #NHWC}>, tensor<1024x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #map}
    // CHECK-SAME:      : tensor<1x1024x49x1xf16, {order = #NHWC}> -> tensor<49x1024x1x1xf16>
    // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[PERMUTE_CAST]]) {
    // CHECK-SAME:      shape_value = [1, 49, 1, 1024]}
    // CHECK-SAME:      : tensor<49x1024x1x1xf16> -> tensor<1x49x1x1024xf16>
    // CHECK:       [[MEM_PERMUTE_0:%.+]] = IE.MemPermute([[AFFINE_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x49x1x1024xf16> -> tensor<1x49x1x1024xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 1024, 49, 1]}
    // CHECK-SAME:      inputs([[INPUT_1]] : tensor<1x2x512x49xf16, {order = #NHWC}>) -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    // CHECK:       [[MEM_PERMUTE_1:%.+]] = IE.MemPermute([[SHAPE_CAST]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x1024x49x1xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[MEM_PERMUTE_1]], [[MEM_PERMUTE_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x49x1x1024xf16, {order = #NHWC}>, tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    // CHECK:       return [[ADD]] : tensor<1x49x1x1024xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL: @NotMoveDifferentInLayoutPermutesPostAddNoNHWC
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x49x1024x1xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x49x1024xf16, {order = #NHWC}>
func.func @NotMoveDifferentInLayoutPermutesPostAddNoNHWC(%arg0: tensor<1x49x1024x1xf16>, %arg1: tensor<1x1x49x1024xf16, {order = #NHWC}>) -> tensor<1x49x1x1024xf16, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x49x1024x1xf16>, tensor<1x49x1024x1xf16> -> tensor<1x49x1024x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 49, 1, 1024]} : tensor<1x49x1024x1xf16> -> tensor<1x49x1x1024xf16>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x49x1x1024xf16> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    %3 = IE.ShapeCast {shape = [1, 1024, 49, 1]} inputs(%arg1 : tensor<1x1x49x1024xf16, {order = #NHWC}>) -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    %4 = IE.MemPermute(%3) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1024x49x1xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    %5 = IE.Add(%4, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x49x1x1024xf16, {order = #NHWC}>, tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    return %5 : tensor<1x49x1x1024xf16, {order = #NHWC}>

    // CHECK:       [[ADD_1:%.+]] = IE.Add([[INPUT_0]], [[INPUT_0]])
    // CHECK-SAME:      : tensor<1x49x1024x1xf16>, tensor<1x49x1024x1xf16> -> tensor<1x49x1024x1xf16>
    // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[ADD_1]]) {
    // CHECK-SAME:      shape_value = [1, 49, 1, 1024]}
    // CHECK-SAME:      : tensor<1x49x1024x1xf16> -> tensor<1x49x1x1024xf16>
    // CHECK:       [[MEM_PERMUTE_0:%.+]] = IE.MemPermute([[AFFINE_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x49x1x1024xf16> -> tensor<1x49x1x1024xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 1024, 49, 1]}
    // CHECK-SAME:      inputs([[INPUT_1]] : tensor<1x1x49x1024xf16, {order = #NHWC}>) -> tensor<1x1024x49x1xf16, {order = #NHWC}>
    // CHECK:       [[MEM_PERMUTE_1:%.+]] = IE.MemPermute([[SHAPE_CAST]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x1024x49x1xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16, {order = #NHWC}>
    // CHECK:       [[ADD_2:%.+]] = IE.Add([[MEM_PERMUTE_1]], [[MEM_PERMUTE_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x49x1x1024xf16, {order = #NHWC}>, tensor<1x49x1x1024xf16, {order = #NHWC}> -> tensor<1x49x1x1024xf16, {order = #NHWC}>

    // CHECK:       return [[ADD_2]] : tensor<1x49x1x1024xf16, {order = #NHWC}>
}
