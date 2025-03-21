//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-nd-mem-permute %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @IgnoreLegalMemPermutes
func.func @IgnoreLegalMemPermutes(%arg0: tensor<1x16x2x3xf32, {order = #NHWC}>) -> tensor<1x16x2x3xf32, {order = #NCHW}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #map} :
        tensor<1x16x2x3xf32, {order = #NHWC}> -> tensor<1x16x2x3xf32, {order = #NCHW}>
    return %0 : tensor<1x16x2x3xf32, {order = #NCHW}>

    // CHECK:     %[[VAL_0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x2x3xf32, {order = #NHWC}> -> tensor<1x16x2x3xf32, {order = #NCHW}>
    // CHECK:     return %[[VAL_0]] : tensor<1x16x2x3xf32, {order = #NCHW}>
}

// -----

#srcOrder = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#dstOrder = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d7, d2, d3, d5, d6, d1, d4)>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d7, d2, d3, d5, d6, d1, d4)>

// CHECK:     #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
// CHECK:     #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d7, d2, d3, d5, d6, d1, d4)>
// CHECK:     #map2 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL:   @MultipleMerge
func.func @MultipleMerge(%arg0: tensor<1x2x3x4x5x6x7x1xf32, {order = #srcOrder}>) -> tensor<1x2x3x4x5x6x7x1xf32, {order = #dstOrder}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #dstOrder, mem_perm = #map} :
        tensor<1x2x3x4x5x6x7x1xf32, {order = #srcOrder}> -> tensor<1x2x3x4x5x6x7x1xf32, {order = #dstOrder}>
    return %0 : tensor<1x2x3x4x5x6x7x1xf32, {order = #dstOrder}>

    // CHECK:     %[[CAST_1:.*]] = IE.PermuteCast(%arg0) {dst_order = #map, mem_perm = #map} : tensor<1x2x3x4x5x6x7x1xf32, {order = #map}> -> tensor<1x2x3x4x5x6x7x1xf32>
    // CHECK:     %[[RESHAPE_1:.*]] = IE.Reshape(%[[CAST_1]]) {shape_value = [2, 12, 5, 42]} : tensor<1x2x3x4x5x6x7x1xf32> -> tensor<2x12x5x42xf32>
    // CHECK:     %[[PERM:.*]] = IE.MemPermute(%[[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #map2} : tensor<2x12x5x42xf32> -> tensor<12x42x2x5xf32>
    // CHECK:     %[[RESHAPE_2:.*]] = IE.Reshape(%[[PERM]]) {shape_value = [1, 1, 3, 4, 6, 7, 2, 5]} : tensor<12x42x2x5xf32> -> tensor<1x1x3x4x6x7x2x5xf32>
    // CHECK:     %[[CAST_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #map1, mem_perm = #map} : tensor<1x1x3x4x6x7x2x5xf32> -> tensor<1x2x3x4x5x6x7x1xf32, {order = #map1}>
    // CHECK:     return %[[CAST_2]] : tensor<1x2x3x4x5x6x7x1xf32, {order = #map1}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL:   @LegalizeNDMemPermute
func.func @LegalizeNDMemPermute(%arg0: tensor<1x2x3x4x5xf32, {order = #NCDHW}>) -> tensor<1x2x3x4x5xf32, {order = #NDHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NDHWC, mem_perm = #NDHWC} :
        tensor<1x2x3x4x5xf32, {order = #NCDHW}> -> tensor<1x2x3x4x5xf32, {order = #NDHWC}>
    return %0 : tensor<1x2x3x4x5xf32, {order = #NDHWC}>

    // CHECK:     %[[CAST_1:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<1x2x3x4x5xf32, {order = #NCDHW}> -> tensor<1x2x3x4x5xf32>
    // CHECK:     %[[RESHAPE_1:.*]] = IE.Reshape(%[[CAST_1]]) {shape_value = [1, 2, 12, 5]} : tensor<1x2x3x4x5xf32> -> tensor<1x2x12x5xf32>
    // CHECK:     %[[PERM:.*]] = IE.MemPermute(%[[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x2x12x5xf32> -> tensor<1x12x5x2xf32>
    // CHECK:     %[[RESHAPE_2:.*]] = IE.Reshape(%[[PERM]]) {shape_value = [1, 3, 4, 5, 2]} : tensor<1x12x5x2xf32> -> tensor<1x3x4x5x2xf32>
    // CHECK:     %[[CAST_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #NDHWC, mem_perm = #NCDHW} : tensor<1x3x4x5x2xf32> -> tensor<1x2x3x4x5xf32, {order = #NDHWC}>
    // CHECK:     return %[[CAST_2]] : tensor<1x2x3x4x5xf32, {order = #NDHWC}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>

// CHECK-LABEL:   @LegalizeNDMemPermuteNo1s
func.func @LegalizeNDMemPermuteNo1s(%arg0: tensor<2x2x3x4x5xf32, {order = #NCDHW}>) -> tensor<2x2x3x4x5xf32, {order = #NDHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NDHWC, mem_perm = #NDHWC} :
        tensor<2x2x3x4x5xf32, {order = #NCDHW}> -> tensor<2x2x3x4x5xf32, {order = #NDHWC}>
    return %0 : tensor<2x2x3x4x5xf32, {order = #NDHWC}>

    // CHECK:     %[[CAST_1:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<2x2x3x4x5xf32, {order = #NCDHW}> -> tensor<2x2x3x4x5xf32>
    // CHECK:     %[[RESHAPE_1:.*]] = IE.Reshape(%[[CAST_1]]) {shape_value = [1, 2, 2, 60]} : tensor<2x2x3x4x5xf32> -> tensor<1x2x2x60xf32>
    // CHECK:     %[[PERM:.*]] = IE.MemPermute(%[[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x2x2x60xf32> -> tensor<1x2x60x2xf32>
    // CHECK:     %[[RESHAPE_2:.*]] = IE.Reshape(%[[PERM]]) {shape_value = [2, 3, 4, 5, 2]} : tensor<1x2x60x2xf32> -> tensor<2x3x4x5x2xf32>
    // CHECK:     %[[CAST_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #NDHWC, mem_perm = #NCDHW} : tensor<2x3x4x5x2xf32> -> tensor<2x2x3x4x5xf32, {order = #NDHWC}>
    // CHECK:     return %[[CAST_2]] : tensor<2x2x3x4x5xf32, {order = #NDHWC}>
}

// -----

#SIXD = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#SIXD_PERM = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d2, d3, d1, d5)>

// CHECK:     #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK:     #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d2, d3, d1, d5)>
// CHECK:     #map2 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>


// CHECK-LABEL:   @LegalizeNDMemPermute6D
func.func @LegalizeNDMemPermute6D(%arg0: tensor<1x2x3x4x5x6xf32, {order = #SIXD}>) -> tensor<1x2x3x4x5x6xf32, {order = #SIXD_PERM}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #SIXD_PERM, mem_perm = #SIXD_PERM} :
        tensor<1x2x3x4x5x6xf32, {order = #SIXD}> -> tensor<1x2x3x4x5x6xf32, {order = #SIXD_PERM}>
    return %0 : tensor<1x2x3x4x5x6xf32, {order = #SIXD_PERM}>

    // CHECK:     %[[CAST_1:.*]] = IE.PermuteCast(%arg0) {dst_order = #map, mem_perm = #map} : tensor<1x2x3x4x5x6xf32, {order = #map}> -> tensor<1x2x3x4x5x6xf32>
    // CHECK:     %[[RESHAPE_1:.*]] = IE.Reshape(%[[CAST_1]]) {shape_value = [2, 12, 5, 6]} : tensor<1x2x3x4x5x6xf32> -> tensor<2x12x5x6xf32>
    // CHECK:     %[[PERM:.*]] = IE.MemPermute(%[[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #map2} : tensor<2x12x5x6xf32> -> tensor<5x12x2x6xf32>
    // CHECK:     %[[RESHAPE_2:.*]] = IE.Reshape(%[[PERM]]) {shape_value = [1, 5, 3, 4, 2, 6]} : tensor<5x12x2x6xf32> -> tensor<1x5x3x4x2x6xf32>
    // CHECK:     %[[CAST_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #map1, mem_perm = #map} : tensor<1x5x3x4x2x6xf32> -> tensor<1x2x3x4x5x6xf32, {order = #map1}>
    // CHECK:     return %[[CAST_2]] : tensor<1x2x3x4x5x6xf32, {order = #map1}>
}

// -----

#CWH = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL:   @LegalizeNDMemPermute3D
func.func @LegalizeNDMemPermute3D(%arg0: tensor<8x40x4096xf16, {order = #CWH}>) -> tensor<8x40x4096xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #CHW, mem_perm = #map} :
        tensor<8x40x4096xf16, {order = #CWH}> -> tensor<8x40x4096xf16>
    return %0 : tensor<8x40x4096xf16>

    // CHECK:     %[[CAST_1:.*]] = IE.PermuteCast(%arg0) {dst_order = #CHW, mem_perm = #CHW} : tensor<8x40x4096xf16, {order = #map}> -> tensor<8x4096x40xf16>
    // CHECK:     %[[RESHAPE_1:.*]] = IE.Reshape(%[[CAST_1]]) {shape_value = [1, 8, 4096, 40]} : tensor<8x4096x40xf16> -> tensor<1x8x4096x40xf16>
    // CHECK:     %[[PERM:.*]] = IE.MemPermute(%[[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x4096x40xf16> -> tensor<1x8x40x4096xf16>
    // CHECK:     %[[RESHAPE_2:.*]] = IE.Reshape(%[[PERM]]) {shape_value = [8, 40, 4096]} : tensor<1x8x40x4096xf16> -> tensor<8x40x4096xf16>
    // CHECK:     return %[[RESHAPE_2]] : tensor<8x40x4096xf16>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1, d3)>

// CHECK-LABEL:   @DecomposeMemPermute_InOrderNCDHW
func.func @DecomposeMemPermute_InOrderNCDHW(%arg0: tensor<3x128x4x128x4xf16>) -> tensor<3x4x4x128x128xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCDHW, mem_perm = #perm} : tensor<3x128x4x128x4xf16> -> tensor<3x4x4x128x128xf16>

    return %0 : tensor<3x4x4x128x128xf16>

    // CHECK:     %[[RESHAPE_IN_1:.*]] = IE.Reshape({{[^:]+}}) {shape_value = [1, 3, 128, 2048]} : tensor<3x128x4x128x4xf16> -> tensor<1x3x128x2048xf16>
    // CHECK:     %[[PERM_1:.*]] = IE.MemPermute(%[[RESHAPE_IN_1]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x3x128x2048xf16> -> tensor<1x3x2048x128xf16>
    // CHECK:     %[[RESHAPE_OUT_1:.*]] = IE.Reshape(%[[PERM_1]]) {shape_value = [3, 4, 128, 4, 128]} : tensor<1x3x2048x128xf16> -> tensor<3x4x128x4x128xf16>

    // CHECK:     %[[RESHAPE_IN_2:.*]] = IE.Reshape(%[[RESHAPE_OUT_1]]) {shape_value = [1, 12, 128, 512]} : tensor<3x4x128x4x128xf16> -> tensor<1x12x128x512xf16>
    // CHECK:     %[[PERM_2:.*]] = IE.MemPermute(%[[RESHAPE_IN_2]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x12x128x512xf16> -> tensor<1x12x512x128xf16>
    // CHECK:     %[[RESHAPE_OUT_2:.*]] = IE.Reshape(%[[PERM_2]]) {shape_value = [3, 4, 4, 128, 128]} : tensor<1x12x512x128xf16> -> tensor<3x4x4x128x128xf16>

    // CHECK:     return %[[RESHAPE_OUT_2]] : tensor<3x4x4x128x128xf16>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>
#NHCDW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>
#perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1, d3)>

// CHECK:     #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>

// CHECK-LABEL:   @DecomposeMemPermute_InOrderNDHWC
func.func @DecomposeMemPermute_InOrderNDHWC(%arg0: tensor<3x128x4x128x4xf16, {order = #NDHWC}>) -> tensor<3x128x4x128x4xf16, {order = #NHCDW}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHCDW, mem_perm = #perm} : tensor<3x128x4x128x4xf16, {order = #NDHWC}> -> tensor<3x128x4x128x4xf16, {order = #NHCDW}>

    return %0 : tensor<3x128x4x128x4xf16, {order = #NHCDW}>

    // CHECK:     %[[CAST_IN:.*]] = IE.PermuteCast({{[^:]+}}) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<3x128x4x128x4xf16, {order = #NDHWC}> -> tensor<3x4x128x4x128xf16>

    // CHECK:     %[[RESHAPE_IN_1:.*]] = IE.Reshape(%[[CAST_IN]]) {shape_value = [1, 3, 4, 65536]} : tensor<3x4x128x4x128xf16> -> tensor<1x3x4x65536xf16>
    // CHECK:     %[[PERM_1:.*]] = IE.MemPermute(%[[RESHAPE_IN_1]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x3x4x65536xf16> -> tensor<1x3x65536x4xf16>
    // CHECK:     %[[RESHAPE_OUT_1:.*]] = IE.Reshape(%[[PERM_1]]) {shape_value = [3, 128, 4, 128, 4]} : tensor<1x3x65536x4xf16> -> tensor<3x128x4x128x4xf16>

    // CHECK:     %[[RESHAPE_IN_2:.*]] = IE.Reshape(%[[RESHAPE_OUT_1]]) {shape_value = [1, 384, 4, 512]} : tensor<3x128x4x128x4xf16> -> tensor<1x384x4x512xf16>
    // CHECK:     %[[PERM_2:.*]] = IE.MemPermute(%[[RESHAPE_IN_2]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x384x4x512xf16> -> tensor<1x384x512x4xf16>
    // CHECK:     %[[RESHAPE_OUT_2:.*]] = IE.Reshape(%[[PERM_2]]) {shape_value = [3, 128, 128, 4, 4]} : tensor<1x384x512x4xf16> -> tensor<3x128x128x4x4xf16>

    // CHECK:     %[[CAST_OUT:.*]] = IE.PermuteCast(%[[RESHAPE_OUT_2]]) {dst_order = #map, mem_perm = #NCDHW} : tensor<3x128x128x4x4xf16> -> tensor<3x128x4x128x4xf16, {order = #map}>


    // CHECK:     return %[[CAST_OUT]] : tensor<3x128x4x128x4xf16, {order = #map}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#perm = affine_map<(d0, d1, d2, d3, d4) -> (d3, d0, d2, d1, d4)>


// CHECK-LABEL:   @DecomposeMemPermuteForRDFT
func.func @DecomposeMemPermuteForRDFT(%arg0: tensor<2x4x8x10x2xf16>) -> tensor<10x2x8x4x2xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCDHW, mem_perm = #perm} : tensor<2x4x8x10x2xf16> -> tensor<10x2x8x4x2xf16>

    return %0 : tensor<10x2x8x4x2xf16>

    // CHECK:     %[[RESHAPE_IN_1:.*]] = IE.Reshape({{[^:]+}}) {shape_value = [1, 2, 320, 2]} : tensor<2x4x8x10x2xf16> -> tensor<1x2x320x2xf16>
    // CHECK:     %[[PERM_1:.*]] = IE.MemPermute(%[[RESHAPE_IN_1]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x2x320x2xf16> -> tensor<1x320x2x2xf16>
    // CHECK:     %[[RESHAPE_OUT_1:.*]] = IE.Reshape(%[[PERM_1]]) {shape_value = [4, 8, 10, 2, 2]} : tensor<1x320x2x2xf16> -> tensor<4x8x10x2x2xf16>

    // CHECK:     %[[RESHAPE_IN_2:.*]] = IE.Reshape(%[[RESHAPE_OUT_1]]) {shape_value = [1, 4, 8, 40]} : tensor<4x8x10x2x2xf16> -> tensor<1x4x8x40xf16>
    // CHECK:     %[[PERM_2:.*]] = IE.MemPermute(%[[RESHAPE_IN_2]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x8x40xf16> -> tensor<1x8x4x40xf16>
    // CHECK:     %[[RESHAPE_OUT_2:.*]] = IE.Reshape(%[[PERM_2]]) {shape_value = [8, 4, 10, 2, 2]} : tensor<1x8x4x40xf16> -> tensor<8x4x10x2x2xf16>

    // CHECK:     %[[RESHAPE_IN_3:.*]] = IE.Reshape(%[[RESHAPE_OUT_2]]) {shape_value = [1, 32, 20, 2]} : tensor<8x4x10x2x2xf16> -> tensor<1x32x20x2xf16>
    // CHECK:     %[[PERM_3:.*]] = IE.MemPermute(%[[RESHAPE_IN_3]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x32x20x2xf16> -> tensor<1x20x32x2xf16>
    // CHECK:     %[[RESHAPE_OUT_3:.*]] = IE.Reshape(%[[PERM_3]]) {shape_value = [10, 2, 8, 4, 2]} : tensor<1x20x32x2xf16> -> tensor<10x2x8x4x2xf16>

    // CHECK:     return %[[RESHAPE_OUT_3]] : tensor<10x2x8x4x2xf16>
}

// -----

#srcOrder = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8)>
#dstOrder = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8)>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d3, d2, d6, d4, d1, d5, d7, d8)>

// CHECK-LABEL:   @DecomposeMemPermuteND
func.func @DecomposeMemPermuteND(%arg0: tensor<4x6x8x10x12x14x16x18x20xf16>) -> tensor<4x10x8x16x12x6x14x18x20xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #dstOrder, mem_perm = #map} : tensor<4x6x8x10x12x14x16x18x20xf16> -> tensor<4x10x8x16x12x6x14x18x20xf16>

    return %0 : tensor<4x10x8x16x12x6x14x18x20xf16>

    // CHECK:     %[[RESHAPE0:.*]] = IE.Reshape({{[^:]+}}) {shape_value = [4, 6, 8, 10, 12, 14, 16, 360]} : tensor<4x6x8x10x12x14x16x18x20xf16> -> tensor<4x6x8x10x12x14x16x360xf16>

    // CHECK:     %[[RESHAPE_IN_1:.*]] = IE.Reshape(%[[RESHAPE0]]) {shape_value = [4, 6, 960, 80640]} : tensor<4x6x8x10x12x14x16x360xf16> -> tensor<4x6x960x80640xf16>
    // CHECK:     %[[PERM_1:.*]] = IE.MemPermute(%[[RESHAPE_IN_1]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<4x6x960x80640xf16> -> tensor<4x960x6x80640xf16>
    // CHECK:     %[[RESHAPE_OUT_1:.*]] = IE.Reshape(%[[PERM_1]]) {shape_value = [4, 8, 10, 12, 6, 14, 16, 360]} : tensor<4x960x6x80640xf16> -> tensor<4x8x10x12x6x14x16x360xf16>

    // CHECK:     %[[RESHAPE_IN_2:.*]] = IE.Reshape(%[[RESHAPE_OUT_1]]) {shape_value = [4, 8, 10, 5806080]} : tensor<4x8x10x12x6x14x16x360xf16> -> tensor<4x8x10x5806080xf16>
    // CHECK:     %[[PERM_2:.*]] = IE.MemPermute(%[[RESHAPE_IN_2]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<4x8x10x5806080xf16> -> tensor<4x10x8x5806080xf16>
    // CHECK:     %[[RESHAPE_OUT_2:.*]] = IE.Reshape(%[[PERM_2]]) {shape_value = [4, 10, 8, 12, 6, 14, 16, 360]} : tensor<4x10x8x5806080xf16> -> tensor<4x10x8x12x6x14x16x360xf16>

    // CHECK:     %[[RESHAPE_IN_3:.*]] = IE.Reshape(%[[RESHAPE_OUT_2]]) {shape_value = [320, 1008, 16, 360]} : tensor<4x10x8x12x6x14x16x360xf16> -> tensor<320x1008x16x360xf16>
    // CHECK:     %[[PERM_3:.*]] = IE.MemPermute(%[[RESHAPE_IN_3]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<320x1008x16x360xf16> -> tensor<320x16x1008x360xf16>
    // CHECK:     %[[RESHAPE_OUT_3:.*]] = IE.Reshape(%[[PERM_3]]) {shape_value = [4, 10, 8, 16, 12, 6, 14, 360]} : tensor<320x16x1008x360xf16> -> tensor<4x10x8x16x12x6x14x360xf16>

    // CHECK:     %[[RESHAPE4:.*]] = IE.Reshape(%[[RESHAPE_OUT_3]]) {shape_value = [4, 10, 8, 16, 12, 6, 14, 18, 20]} : tensor<4x10x8x16x12x6x14x360xf16> -> tensor<4x10x8x16x12x6x14x18x20xf16>

    // CHECK:     return %[[RESHAPE4]] : tensor<4x10x8x16x12x6x14x18x20xf16>
}
