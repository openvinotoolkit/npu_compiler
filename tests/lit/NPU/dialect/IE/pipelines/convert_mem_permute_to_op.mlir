//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-mem-permute-to-op --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MemPermuteNHCWInNCHWOutNHCWPerm
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x48x224xf16, {order = #NHCW}>
func.func @MemPermuteNHCWInNCHWOutNHCWPerm(%arg0: tensor<1x32x48x224xf16, {order = #NHCW}>)
        -> tensor<1x32x48x224xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW,
        mem_perm = #NHCW
    } : tensor<1x32x48x224xf16, {order = #NHCW}>
        -> tensor<1x32x48x224xf16>

    return %MEM_PERMUTE : tensor<1x32x48x224xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x32x48x224xf16, {order = #NHCW}>
    // CHECK-SAME:      -> tensor<1x224x48x32xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[IN_PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x224x48x32xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x224x48x32xf16, {order = #NWHC}>

    // CHECK:       [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x224x48x32xf16, {order = #NWHC}>
    // CHECK-SAME:      -> tensor<1x32x48x224xf16>

    // CHECK:       return [[OUT_PERMUTE_CAST]] : tensor<1x32x48x224xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MemPermuteNCHWInNCHWOutNHWCPerm
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x48x64xf16>
func.func @MemPermuteNCHWInNCHWOutNHWCPerm(%arg0: tensor<1x32x48x64xf16>)
        -> tensor<1x48x64x32xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW,
        mem_perm = #NHWC
    } : tensor<1x32x48x64xf16> -> tensor<1x48x64x32xf16>

    return %MEM_PERMUTE : tensor<1x48x64x32xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x32x48x64xf16>
    // CHECK-SAME:      -> tensor<1x64x32x48xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[IN_PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x32x48xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x32x48xf16, {order = #NWCH}>

    // CHECK:       [[OUTPUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x64x32x48xf16, {order = #NWCH}>
    // CHECK-SAME:      -> tensor<1x48x64x32xf16>

    // CHECK:       return [[OUTPUT_PERMUTE_CAST]] : tensor<1x48x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MemPermuteNCHWInNCHWOutNHCWPerm
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x48x224xf16>
func.func @MemPermuteNCHWInNCHWOutNHCWPerm(%arg0: tensor<1x32x48x224xf16>)
        -> tensor<1x48x32x224xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
    } : tensor<1x32x48x224xf16> -> tensor<1x48x32x224xf16>

    return %MEM_PERMUTE : tensor<1x48x32x224xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x32x48x224xf16>
    // CHECK-SAME:      -> tensor<1x224x32x48xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[IN_PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x224x32x48xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x224x32x48xf16, {order = #NWHC}>

    // CHECK:       [[OUTPUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x224x32x48xf16, {order = #NWHC}>
    // CHECK-SAME:      -> tensor<1x48x32x224xf16>

    // CHECK:       return [[OUTPUT_PERMUTE_CAST]] : tensor<1x48x32x224xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MemPermuteNCWHInNHWCOutNWHCPerm
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x48x64xf16, {order = #NCWH}>
func.func @MemPermuteNCWHInNHWCOutNWHCPerm(%arg0: tensor<1x32x48x64xf16, {order = #NCWH}>)
        -> tensor<1x32x48x64xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x32x48x64xf16, {order = #NCWH}>
        -> tensor<1x32x48x64xf16, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<1x32x48x64xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x32x48x64xf16, {order = #NCWH}>
    // CHECK-SAME:      -> tensor<1x48x32x64xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[IN_PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x32x64xf16, {order = #NCWH}>

    // CHECK:       [[OUTPUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x48x32x64xf16, {order = #NCWH}>
    // CHECK-SAME:      -> tensor<1x32x48x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT_PERMUTE_CAST]] : tensor<1x32x48x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MemPermuteNCHWInNCHWOutNCWHPerm
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x1500x64xf16>
func.func @MemPermuteNCHWInNCHWOutNCWHPerm(%arg0: tensor<1x8x1500x64xf16>) -> tensor<1x8x64x1500xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
    } : tensor<1x8x1500x64xf16> -> tensor<1x8x64x1500xf16>

    return %MEM_PERMUTE : tensor<1x8x64x1500xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x8x1500x64xf16>
    // CHECK-SAME:      -> tensor<1x64x8x1500xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[IN_PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x8x1500xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x8x1500xf16, {order = #NHCW}>

    // CHECK:       [[OUTPUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x64x8x1500xf16, {order = #NHCW}>
    // CHECK-SAME:      -> tensor<1x8x64x1500xf16>

    // CHECK:       return [[OUTPUT_PERMUTE_CAST]] : tensor<1x8x64x1500xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MemPermuteWithMisalignedShape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x47x64xf16, {order = #NCWH}>
func.func @MemPermuteWithMisalignedShape(%arg0: tensor<1x32x47x64xf16, {order = #NCWH}>)
        -> tensor<1x32x47x64xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x32x47x64xf16, {order = #NCWH}> -> tensor<1x32x47x64xf16, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<1x32x47x64xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x32x47x64xf16, {order = #NCWH}>
    // CHECK-SAME:      -> tensor<1x47x32x64xf16, {order = #NHWC}>

    // CHECK:       [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 32, 188]
    // CHECK-SAME:  } inputs([[IN_PERMUTE_CAST]] : tensor<1x47x32x64xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x16x32x188xf16, {order = #NHWC}>

    // CHECK:       [[POOLING_0:%.+]] = IE.MaxPool([[SHAPE_CAST_0]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x32x188xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x32x188xf16, {order = #NWCH}>

    // CHECK:       [[LAYOUT_CAST:%.+]] = IE.LayoutCast([[POOLING_0]]) {
    // CHECK-SAME:      dst_order = #NHWC
    // CHECK-SAME:  } : tensor<1x16x32x188xf16, {order = #NWCH}>
    // CHECK-SAME:      -> tensor<1x16x32x188xf16, {order = #NHWC}>

    // CHECK:       [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 32, 64, 47]
    // CHECK-SAME:  } inputs([[LAYOUT_CAST]] : tensor<1x16x32x188xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x32x64x47xf16, {order = #NHWC}>

    // CHECK:       [[POOLING_1:%.+]] = IE.MaxPool([[SHAPE_CAST_1]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x32x64x47xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x64x47xf16, {order = #NWHC}>

    // CHECK:       [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOLING_1]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x32x64x47xf16, {order = #NWHC}>
    // CHECK-SAME:      -> tensor<1x32x47x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_PERMUTE_CAST]] : tensor<1x32x47x64xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SkipTrivialMemPermute
func.func @SkipTrivialMemPermute(%arg0: tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>)
        -> tensor<1x48x64x32xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    } : tensor<1x32x48x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        -> tensor<1x48x64x32xf16>

    return %MEM_PERMUTE : tensor<1x48x64x32xf16>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeDimN
func.func @ReshapeDimN(%arg0: tensor<25x14x14x2304xf16>) -> tensor<25x14x2304x14xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW,
        mem_perm = #NHWC
    } : tensor<25x14x14x2304xf16> -> tensor<25x14x2304x14xf16>

    return %MEM_PERMUTE : tensor<25x14x2304x14xf16>

    // CHECK:       [[SHAPE_CAST1:%.+]] = IE.ShapeCast {shape = [1, 25, 14, 32256]} inputs(%arg0 : tensor<25x14x14x2304xf16>) -> tensor<1x25x14x32256xf16>
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[SHAPE_CAST1]]) {dst_order = #NCHW, mem_perm = #NCWH} :
    // CHECK-SAME:  tensor<1x25x14x32256xf16> -> tensor<1x25x32256x14xf16>
    // CHECK:       [[SHAPE_CAST2:%.+]] = IE.ShapeCast {shape = [25, 14, 2304, 14]} inputs([[MEM_PERMUTE]] : tensor<1x25x32256x14xf16>) -> tensor<25x14x2304x14xf16>
    // CHECK:       return [[SHAPE_CAST2]] : tensor<25x14x2304x14xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithAlignChannel(%arg0: tensor<1x32x255x511xf16, {order = #NHWC}>) -> tensor<1x32x255x511xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x32x255x511xf16, {order = #NHWC}> -> tensor<1x32x255x511xf16>
    return %MEM_PERMUTE : tensor<1x32x255x511xf16>

    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:        kernel_size = [1, 1],
    // CHECK-SAME:        pads_begin = [0, 0],
    // CHECK-SAME:        pads_end = [0, 0],
    // CHECK-SAME:        rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:        strides = [1, 1]} : tensor<1x32x255x511xf16, {order = #NHWC}> -> tensor<1x32x255x511xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x32x255x511xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithUnalignedChannel
func.func @MemPermuteNHWCInNCHWOutWithUnalignedChannel(%arg0: tensor<1x3x256x512xf16, {order = #NHWC}>) -> tensor<1x3x256x512xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x256x512xf16, {order = #NHWC}> -> tensor<1x3x256x512xf16>
    return %MEM_PERMUTE : tensor<1x3x256x512xf16>

    // CHECK:       [[SHAPE_CAST_WC_IN_W:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 96]} inputs(%arg0 : tensor<1x3x256x512xf16, {order = #NHWC}>) -> tensor<1x16x256x96xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.+]] = IE.MaxPool([[SHAPE_CAST_WC_IN_W]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x16x256x96xf16, {order = #NHWC}> -> tensor<1x16x256x96xf16, {order = #NWCH}>
    // CHECK:       [[LAYOUT_CAST_0:%.+]] = IE.LayoutCast([[MAXPOOL_0]]) {dst_order = #NHWC} : tensor<1x16x256x96xf16, {order = #NWCH}> -> tensor<1x16x256x96xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_WC_IN_H:%.+]] = IE.ShapeCast {shape = [1, 256, 512, 3]} inputs([[LAYOUT_CAST_0]] : tensor<1x16x256x96xf16, {order = #NHWC}>) -> tensor<1x256x512x3xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.+]] = IE.MaxPool([[SHAPE_CAST_WC_IN_H]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x256x512x3xf16, {order = #NHWC}> -> tensor<1x256x512x3xf16, {order = #NWCH}>
    // CHECK:       [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[MAXPOOL_1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x256x512x3xf16, {order = #NWCH}> -> tensor<1x3x256x512xf16>
    // CHECK:       return [[OUT_PERMUTE_CAST]] : tensor<1x3x256x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithHCNotAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithHCNotAlignChannel(%arg0: tensor<1x3x255x512xf16, {order = #NHWC}>) -> tensor<1x3x255x512xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x255x512xf16, {order = #NHWC}> -> tensor<1x3x255x512xf16>
    return %MEM_PERMUTE : tensor<1x3x255x512xf16>


    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
    // CHECK-SAME:  tensor<1x3x255x512xf16, {order = #NHWC}> -> tensor<1x3x255x512xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x255x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithWCNotAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithWCNotAlignChannel(%arg0: tensor<1x3x256x511xf16, {order = #NHWC}>) -> tensor<1x3x256x511xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x256x511xf16, {order = #NHWC}> -> tensor<1x3x256x511xf16>
    return %MEM_PERMUTE : tensor<1x3x256x511xf16>


    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
    // CHECK-SAME:  tensor<1x3x256x511xf16, {order = #NHWC}> -> tensor<1x3x256x511xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x256x511xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNWCHInNHWCOut
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x48x289xf16, {order = #NWCH}>
func.func @MemPermuteNWCHInNHWCOut(%arg0: tensor<1x16x48x289xf16, {order = #NWCH}>) -> tensor<1x16x48x289xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x48x289xf16, {order = #NWCH}> -> tensor<1x16x48x289xf16, {order = #NHWC}>
    return %MEM_PERMUTE : tensor<1x16x48x289xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x16x48x289xf16, {order = #NWCH}>
    // CHECK-SAME:      -> tensor<1x48x289x16xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[IN_PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x289x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x289x16xf16>

    // CHECK:       [[OUTPUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x48x289x16xf16>
    // CHECK-SAME:      -> tensor<1x16x48x289xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT_PERMUTE_CAST]] : tensor<1x16x48x289xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteNWCHInNCHWOut
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x48x289xf16, {order = #NWCH}>
func.func @MemPermuteNWCHInNCHWOut(%arg0: tensor<1x16x48x289xf16, {order = #NWCH}>) -> tensor<1x16x48x289xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x48x289xf16, {order = #NWCH}> -> tensor<1x16x48x289xf16>
    return %MEM_PERMUTE : tensor<1x16x48x289xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x16x48x289xf16, {order = #NWCH}>
    // CHECK-SAME:      -> tensor<1x48x289x16xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[IN_PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x289x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x289x16xf16, {order = #NWCH}>

    // CHECK:       [[OUTPUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x48x289x16xf16, {order = #NWCH}>
    // CHECK-SAME:      -> tensor<1x16x48x289xf16>

    // CHECK:       return [[OUTPUT_PERMUTE_CAST]] : tensor<1x16x48x289xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>


// CHECK-LABEL: @MemPermuteNHWCInNCHWOutWithWCNotAlignChannel
func.func @MemPermuteNHWCInNCHWOutWithWCNotAlignChannel(%arg0: tensor<1x3x256x511xf16, {order = #NHWC}>) -> tensor<1x3x256x511xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x256x511xf16, {order = #NHWC}> -> tensor<1x3x256x511xf16>
    return %MEM_PERMUTE : tensor<1x3x256x511xf16>


    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} :
    // CHECK-SAME:  tensor<1x3x256x511xf16, {order = #NHWC}> -> tensor<1x3x256x511xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x3x256x511xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @UnsupportedMemPermuteForUnalignedChannel
func.func @UnsupportedMemPermuteForUnalignedChannel(%arg0: tensor<1x4x20x36xf16, {order = #NHWC}>) -> tensor<1x4x20x36xf16, {order = #NCWH}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x4x20x36xf16, {order = #NHWC}> -> tensor<1x4x20x36xf16, {order = #NCWH}>
    return %MEM_PERMUTE : tensor<1x4x20x36xf16, {order = #NCWH}>


    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NWHC} :
    // CHECK-SAME:      tensor<1x4x20x36xf16, {order = #NHWC}> -> tensor<1x4x20x36xf16, {order = #NCWH}>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x4x20x36xf16, {order = #NCWH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK-LABEL: @MemPermuteDimWandDimHCAligned
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x20x48xf16, {order = #NHWC}>
func.func @MemPermuteDimWandDimHCAligned(%arg0: tensor<1x4x20x48xf16, {order = #NHWC}>) -> tensor<1x4x20x48xf16, {order = #NCWH}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NWHC} : tensor<1x4x20x48xf16, {order = #NHWC}> -> tensor<1x4x20x48xf16, {order = #NCWH}>
    return %MEM_PERMUTE : tensor<1x4x20x48xf16, {order = #NCWH}>

    // CHECK:       [[IN_SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 20, 12]} inputs([[INPUT]] : tensor<1x4x20x48xf16, {order = #NHWC}>) -> tensor<1x16x20x12xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.+]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x16x20x12xf16, {order = #NHWC}> -> tensor<1x16x20x12xf16, {order = #NWCH}>
    // CHECK:       [[LAYOUT_CAST_0:%.+]] = IE.LayoutCast([[MAXPOOL_0]]) {dst_order = #NHWC} : tensor<1x16x20x12xf16, {order = #NWCH}> -> tensor<1x16x20x12xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 48, 5]} inputs([[LAYOUT_CAST_0]] : tensor<1x16x20x12xf16, {order = #NHWC}>) -> tensor<1x16x48x5xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.+]] = IE.MaxPool([[SHAPE_CAST_0]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x16x48x5xf16, {order = #NHWC}> -> tensor<1x16x48x5xf16, {order = #NWCH}>
    // CHECK:       [[LAYOUT_CAST_1:%.+]] = IE.LayoutCast([[MAXPOOL_1]]) {dst_order = #NHWC} : tensor<1x16x48x5xf16, {order = #NWCH}> -> tensor<1x16x48x5xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 48, 4, 20]} inputs([[LAYOUT_CAST_1]] : tensor<1x16x48x5xf16, {order = #NHWC}>) -> tensor<1x48x4x20xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_2:%.*]] = IE.MaxPool([[SHAPE_CAST_1]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x48x4x20xf16, {order = #NHWC}> -> tensor<1x48x4x20xf16, {order = #NHCW}>
    // CHECK:       [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[MAXPOOL_2]]) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x48x4x20xf16, {order = #NHCW}> -> tensor<1x4x20x48xf16, {order = #NCWH}>

    // CHECK:       return [[OUT_PERMUTE_CAST]] : tensor<1x4x20x48xf16, {order = #NCWH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @UnsupportedMemPermuteForIntegerInputType
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x1x10x100xsi32>
func.func @UnsupportedMemPermuteForIntegerInputType(%arg0: tensor<1x1x10x100xsi32>) -> tensor<1x1x100x10xsi32> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x10x100xsi32> -> tensor<1x1x100x10xsi32>
    return %MEM_PERMUTE : tensor<1x1x100x10xsi32>

    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT0]]) {dst_order = #NCHW, mem_perm = #NCWH} :
    // CHECK-SAME:      -> tensor<1x1x100x10xsi32>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x1x100x10xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @UnsupportedMemPermuteForFP32InputType
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x1x1024x768xf32>
func.func @UnsupportedMemPermuteForFP32InputType(%arg0: tensor<1x1x1024x768xf32>) -> tensor<1x1x768x1024xf32> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x1024x768xf32> -> tensor<1x1x768x1024xf32>
    return %MEM_PERMUTE : tensor<1x1x768x1024xf32>

    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT0]]) {dst_order = #NCHW, mem_perm = #NCWH} :
    // CHECK-SAME:      -> tensor<1x1x768x1024xf32>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x1x768x1024xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @UnsupportedMemPermuteForFP8InputType
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x1x1024x768xf8E4M3FN>
func.func @UnsupportedMemPermuteForFP8InputType(%arg0: tensor<1x1x1024x768xf8E4M3FN>) -> tensor<1x1x768x1024xf8E4M3FN> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x1024x768xf8E4M3FN> -> tensor<1x1x768x1024xf8E4M3FN>
    return %MEM_PERMUTE : tensor<1x1x768x1024xf8E4M3FN>

    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT0]]) {dst_order = #NCHW, mem_perm = #NCWH} :
    // CHECK-SAME:      -> tensor<1x1x768x1024xf8E4M3FN>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x1x768x1024xf8E4M3FN>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MemPermuteWithPermNHCW
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x640x640xf16, {order = #NHWC}>
func.func @MemPermuteWithPermNHCW(%arg0: tensor<1x4x640x640xf16, {order = #NHWC}>) -> tensor<1x4x640x640xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x4x640x640xf16, {order = #NHWC}> -> tensor<1x4x640x640xf16, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<1x4x640x640xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 640, 160]} inputs([[INPUT]] : tensor<1x4x640x640xf16, {order = #NHWC}>) -> tensor<1x16x640x160xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.+]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x16x640x160xf16, {order = #NHWC}> -> tensor<1x16x640x160xf16, {order = #NWCH}>
    // CHECK:       [[LAYOUT_CAST_0:%.+]] = IE.LayoutCast([[MAXPOOL_0]]) {dst_order = #NHWC} : tensor<1x16x640x160xf16, {order = #NWCH}> -> tensor<1x16x640x160xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 640, 640, 4]} inputs([[LAYOUT_CAST_0]] : tensor<1x16x640x160xf16, {order = #NHWC}>) -> tensor<1x640x640x4xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.+]] = IE.MaxPool([[SHAPE_CAST_0]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x640x640x4xf16, {order = #NHWC}> -> tensor<1x640x640x4xf16, {order = #NHCW}>
    // CHECK:       [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[MAXPOOL_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x640x640x4xf16, {order = #NHCW}> -> tensor<1x4x640x640xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_PERMUTE_CAST]] : tensor<1x4x640x640xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SkipConversionForSmallHeightNum
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4096x4x40xf16>
func.func @SkipConversionForSmallHeightNum(%arg0: tensor<1x4096x4x40xf16>) -> tensor<1x4x4096x40xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4096x4x40xf16> -> tensor<1x4x4096x40xf16>

    return %MEM_PERMUTE : tensor<1x4x4096x40xf16>

    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4096x4x40xf16> -> tensor<1x4x4096x40xf16>

    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x4x4096x40xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16:3, {
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329,
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329}>

!qElemType1 = !quant.uniform<i4:f16:1, {
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329,
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329}>

// CHECK-LABEL: @ConvertPerAxisQuantTypeMemPermute
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x1x32x!qElemType>
func.func @ConvertPerAxisQuantTypeMemPermute(%arg0: tensor<1x128x1x32x!qElemType>) -> tensor<1x32x1x128x!qElemType1> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x128x1x32x!qElemType> -> tensor<1x32x1x128x!qElemType1>

    return %MEM_PERMUTE : tensor<1x32x1x128x!qElemType1>

    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x128x1x32x!qElemType>
    // CHECK-SAME:      -> tensor<1x32x128x1x!qElemType1, {order = #NHWC}>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[IN_PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x32x128x1x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x128x1x!qElemType1, {order = #NCWH}>

    // CHECK:       [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[MAX_POOL]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x32x128x1x!qElemType1, {order = #NCWH}>
    // CHECK-SAME:      -> tensor<1x32x1x128x!qElemType1>

    // CHECK:       return [[OUT_PERMUTE_CAST]] : tensor<1x32x1x128x!qElemType1>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReshapeIOAndConvertMemPermute
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3x768x768xf16, {order = #NHWC}>
func.func @ReshapeIOAndConvertMemPermute(%arg0: tensor<1x3x768x768xf16, {order = #NHWC}>) -> tensor<1x768x3x768xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x3x768x768xf16, {order = #NHWC}> -> tensor<1x768x3x768xf16>

    return %MEM_PERMUTE : tensor<1x768x3x768xf16>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[IN_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 768, 144]
    // CHECK-SAME:  } inputs([[INPUT]] : tensor<1x3x768x768xf16, {order = #NHWC}>)
    // CHECK-SAME:      -> tensor<1x16x768x144xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[IN_SHAPE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x768x144xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x768x144xf16, {order = #NWCH}>

    // CHECK:       [[OUTPUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x16x768x144xf16, {order = #NWCH}>
    // CHECK-SAME:      -> tensor<1x144x16x768xf16>

    // CHECK:       [[OUTPUT_SHAPE_CAST:%.+]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 768, 3, 768]
    // CHECK-SAME:  } inputs([[OUTPUT_PERMUTE_CAST]] : tensor<1x144x16x768xf16>)
    // CHECK-SAME:      -> tensor<1x768x3x768xf16>

    // CHECK:       return [[OUTPUT_SHAPE_CAST]] : tensor<1x768x3x768xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#WNCH = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @MemPermuteWithDimNChanged
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x80x7975xf16>
func.func @MemPermuteWithDimNChanged(%arg0: tensor<1x32x80x7975xf16>) -> tensor<7975x1x32x80xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #WNCH} : tensor<1x32x80x7975xf16> -> tensor<7975x1x32x80xf16>

    return %MEM_PERMUTE : tensor<7975x1x32x80xf16>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x80x7975xf16> -> tensor<1x7975x32x80xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 32, 39875]} inputs([[PERMUTE_CAST_0]] : tensor<1x7975x32x80xf16, {order = #NHWC}>) -> tensor<1x16x32x39875xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.+]]  = IE.MaxPool([[SHAPE_CAST_0]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x16x32x39875xf16, {order = #NHWC}> -> tensor<1x16x32x39875xf16, {order = #NWCH}>
    // CHECK:       [[LAYOUT_CAST:%.+]] = IE.LayoutCast([[MAXPOOL_0]]) {dst_order = #NHWC} : tensor<1x16x32x39875xf16, {order = #NWCH}> -> tensor<1x16x32x39875xf16, {order = #NHWC}>
    // CHECK:       [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 32, 80, 7975]} inputs([[LAYOUT_CAST]] : tensor<1x16x32x39875xf16, {order = #NHWC}>) -> tensor<1x32x80x7975xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.+]]  = IE.MaxPool([[SHAPE_CAST_1]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x32x80x7975xf16, {order = #NHWC}> -> tensor<1x32x80x7975xf16, {order = #NWCH}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[MAXPOOL_1]]) {dst_order = #NCHW, mem_perm = #map} : tensor<1x32x80x7975xf16, {order = #NWCH}> -> tensor<7975x1x32x80xf16>

    // CHECK:       return [[PERMUTE_CAST_1]] : tensor<7975x1x32x80xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>

// CHECK-LABEL: @SkipConversionForMemPermuteWithDimNChangedAndFP32InputType
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x80x7975xf32>
func.func @SkipConversionForMemPermuteWithDimNChangedAndFP32InputType(%arg0: tensor<1x32x80x7975xf32>) -> tensor<7975x1x32x80xf32> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #map} : tensor<1x32x80x7975xf32> -> tensor<7975x1x32x80xf32>

    return %MEM_PERMUTE : tensor<7975x1x32x80xf32>

    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NCHW, mem_perm = #map} : tensor<1x32x80x7975xf32> -> tensor<7975x1x32x80xf32>

    // CHECK:       return [[MEM_PERMUTE]] : tensor<7975x1x32x80xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#HWNC = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>

// CHECK-LABEL: @AdjustMemPermuteShape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x16x128xf16, {order = #NCHW}>
func.func @AdjustMemPermuteShape(%arg0: tensor<1x1024x16x128xf16, {order = #NCHW}>) -> tensor<16x128x1x1024xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #HWNC} : tensor<1x1024x16x128xf16, {order = #NCHW}> -> tensor<16x128x1x1024xf16>

    return %MEM_PERMUTE : tensor<16x128x1x1024xf16>

    // CHECK:       [[PERMUTECAST_IN:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1024x16x128xf16, {order = #NCHW}> -> tensor<1x128x1024x16xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x1024x16xf16, {order = #NHWC}> -> tensor<1x128x1024x16xf16, {order = #NWCH}>
    // CHECK:       [[PERMUTECAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #map} : tensor<1x128x1024x16xf16, {order = #NWCH}> -> tensor<16x128x1x1024xf16>
    // CHECK:       return [[PERMUTECAST_OUT]] : tensor<16x128x1x1024xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#WCHN = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2, d0)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @AdjustMemPermuteShapeWithDimsOne
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1024x1x1x128xf16, {order = #NCHW}>
func.func @AdjustMemPermuteShapeWithDimsOne(%arg0: tensor<1024x1x1x128xf16, {order = #NCHW}>) -> tensor<128x1024x1x1xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #WCHN} : tensor<1024x1x1x128xf16, {order = #NCHW}> -> tensor<128x1024x1x1xf16, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<128x1024x1x1xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #map}
    // CHECK-SAME:     : tensor<1024x1x1x128xf16, {order = #NCHW}> -> tensor<1x128x1024x1xf16, {order = #NHWC}>
    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x128x1024x1xf16, {order = #NHWC}> -> tensor<1x128x1024x1xf16, {order = #NCWH}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[MAX_POOL]]) {dst_order = #NHWC, mem_perm = #map}
    // CHECK-SAME:     : tensor<1x128x1024x1xf16, {order = #NCWH}> -> tensor<128x1024x1x1xf16, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_CAST_1]] : tensor<128x1024x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#HNWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @AdjustMemPermuteWithDimNChanged
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x192xf16, {order = #NHWC}>
func.func @AdjustMemPermuteWithDimNChanged(%arg0: tensor<1x1x256x192xf16, {order = #NHWC}>) -> tensor<1x1x256x192xf16, {order = #map}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #map, mem_perm = #HNWC}
            : tensor<1x1x256x192xf16, {order = #NHWC}> -> tensor<1x1x256x192xf16, {order = #map}>

    return %MEM_PERMUTE : tensor<1x1x256x192xf16, {order = #map}>

    // CHECK:       [[PERMUTECAST_IN:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x1x256x192xf16, {order = #NHWC}> -> tensor<1x192x256x1xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x192x256x1xf16, {order = #NHWC}> -> tensor<1x192x256x1xf16, {order = #NCWH}>
    // CHECK:       [[PERMUTECAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #map, mem_perm = #map1} : tensor<1x192x256x1xf16, {order = #NCWH}> -> tensor<1x1x256x192xf16, {order = #map}>
    // CHECK:       return [[PERMUTECAST_OUT]] : tensor<1x1x256x192xf16, {order = #map}>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#WNHC = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2, d1)>
!qElemType = !quant.uniform<i4:f16:3, {
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329,
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329
    }>
!qElemType1 = !quant.uniform<i4:f16:0, {
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329,
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329
    }>

// CHECK-LABEL: @AdjustMemPermuteForPerAxisQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x12800x1x32x!qElemType, {order = #NCHW}>
func.func @AdjustMemPermuteForPerAxisQuantize(%arg0: tensor<1x12800x1x32x!qElemType, {order = #NCHW}>) -> tensor<32x12800x1x1x!qElemType1, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #WNHC} : tensor<1x12800x1x32x!qElemType, {order = #NCHW}> -> tensor<32x12800x1x1x!qElemType1, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<32x12800x1x1x!qElemType1, {order = #NHWC}>

    // CHECK:       [[PERMUTE_CAST_IN:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x12800x1x32x!qElemType, {order = #NCHW}> -> tensor<1x32x12800x1x!qElemType2, {order = #NHWC}>
    // CHECK:       [[MAX_POOL:%.+]]  = IE.MaxPool([[PERMUTE_CAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x12800x1x!qElemType2, {order = #NHWC}>
    // CHECK-SAME:                  -> tensor<1x32x12800x1x!qElemType2, {order = #NCWH}>
    // CHECK:       [[PERMUTE_CAST_OUT:%.+]] = IE.PermuteCast([[MAX_POOL]]) {dst_order = #NHWC, mem_perm = #map} : tensor<1x32x12800x1x!qElemType2, {order = #NCWH}> -> tensor<32x12800x1x1x!qElemType1, {order = #NHWC}>
    // CHECK:       return [[PERMUTE_CAST_OUT]] : tensor<32x12800x1x1x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#HNWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @BigMemPermute
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4x16x19320xf16>
func.func @BigMemPermute(%arg0: tensor<1x4x16x19320xf16>) -> tensor<1x16x4x19320xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCWH } :
    tensor<1x4x16x19320xf16> -> tensor<1x16x4x19320xf16, {order = #NHWC}>


    return %MEM_PERMUTE : tensor<1x16x4x19320xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x4x16x19320xf16> -> tensor<1x16x4x19320xf16, {order = #NHWC}>

    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x16x4x19320xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16:1, {
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329,
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254}>

// CHECK-LABEL: @NotConvertPerAxisQuantTypeMemPermute
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x30x4x128x!qElemType, {order = #NHWC}>
func.func @NotConvertPerAxisQuantTypeMemPermute(%arg0: tensor<1x30x4x128x!qElemType, {order = #NHWC}>) -> tensor<1x30x4x128x!qElemType> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x30x4x128x!qElemType, {order = #NHWC}> -> tensor<1x30x4x128x!qElemType>

    return %MEM_PERMUTE : tensor<1x30x4x128x!qElemType>

    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NWCH
    // CHECK-SAME:  } : tensor<1x30x4x128x!qElemType, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x30x4x128x!qElemType>

    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x30x4x128x!qElemType>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MemPermuteToPoolWithMemPermNCWHAndExpand
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x24x47x63xf16, {order = #NHWC}>
func.func @MemPermuteToPoolWithMemPermNCWHAndExpand(%arg0: tensor<1x24x47x63xf16, {order = #NHWC}>)
        -> tensor<1x47x24x63xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW, mem_perm = #NCWH
    } : tensor<1x24x47x63xf16, {order = #NHWC}> -> tensor<1x47x24x63xf16>

    return %MEM_PERMUTE : tensor<1x47x24x63xf16>

    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {
    // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]
    // CHECK-SAME:      } : tensor<1x24x47x63xf16, {order = #NHWC}> -> tensor<1x32x47x63xf16, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[EXPAND]]) {
    // CHECK-SAME:          kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x32x47x63xf16, {order = #NHWC}> -> tensor<1x32x47x63xf16, {order = #NHCW}>

    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[POOLING]]) {
    // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NCHW
    // CHECK-SAME:      } : tensor<1x32x47x63xf16, {order = #NHCW}> -> tensor<1x47x32x63xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERMUTE_CAST]] [0, 0, 0, 0] [1, 47, 24, 63] : tensor<1x47x32x63xf16> to tensor<1x47x24x63xf16>

    // CHECK:       return [[SLICE]] : tensor<1x47x24x63xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.100000e+00>

// CHECK-LABEL: @MemPermuteToPoolWithMemPermNWCHAndExpand
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x48x47x63x!qElemType>
func.func @MemPermuteToPoolWithMemPermNWCHAndExpand(%arg0: tensor<1x48x47x63x!qElemType>)
        -> tensor<1x47x63x48x!qElemType, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #NWCH
    } : tensor<1x48x47x63x!qElemType> -> tensor<1x47x63x48x!qElemType, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<1x47x63x48x!qElemType, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {
    // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 1]
    // CHECK-SAME:      } : tensor<1x48x47x63x!qElemType> -> tensor<1x48x47x64x!qElemType>

    // CHECK:       [[PERMUTE_CAST_IN:%.+]] = IE.PermuteCast([[EXPAND]]) {
    // CHECK-SAME:          dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:      } : tensor<1x48x47x64x!qElemType> -> tensor<1x64x48x47x!qElemType, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[PERMUTE_CAST_IN]]) {
    // CHECK-SAME:          kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x64x48x47x!qElemType, {order = #NHWC}> -> tensor<1x64x48x47x!qElemType>

    // CHECK:       [[PERMUTE_CAST_OUT:%.+]] = IE.PermuteCast([[POOLING]]) {
    // CHECK-SAME:          dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:      } : tensor<1x64x48x47x!qElemType> -> tensor<1x47x64x48x!qElemType, {order = #NHWC}>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERMUTE_CAST_OUT]] [0, 0, 0, 0] [1, 47, 63, 48] : tensor<1x47x64x48x!qElemType, {order = #NHWC}> to tensor<1x47x63x48x!qElemType, {order = #NHWC}>

    // CHECK:       return [[SLICE]] : tensor<1x47x63x48x!qElemType, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16:3, {1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,
                                       6.000000e-01,7.100000e-01,8.000000e-01,9.000000e-01,1.100000e+00}>
!qElemType1 = !quant.uniform<u8:f16:1, {1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,
                                        6.000000e-01,7.100000e-01,8.000000e-01,9.000000e-01,1.100000e+00}>
!qElemType2 = !quant.uniform<u8:f16:3, {1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,
                                        6.000000e-01,7.100000e-01,8.000000e-01,9.000000e-01,1.100000e+00,
                                        1.100000e+00,1.100000e+00,1.100000e+00,1.100000e+00,1.100000e+00,1.100000e+00}>
!qElemType3 = !quant.uniform<u8:f16:1, {1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,
                                        6.000000e-01,7.100000e-01,8.000000e-01,9.000000e-01,1.100000e+00,
                                        1.100000e+00,1.100000e+00,1.100000e+00,1.100000e+00,1.100000e+00,1.100000e+00}>

// CHECK-LABEL: @PerAxisQuantizeMemPermuteToPoolWithMemPermNWCHAndExpand
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x48x329x10x!qElemType>
func.func @PerAxisQuantizeMemPermuteToPoolWithMemPermNWCHAndExpand(%arg0: tensor<1x48x329x10x!qElemType>)
        -> tensor<1x10x48x329x!qElemType1> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x48x329x10x!qElemType> -> tensor<1x10x48x329x!qElemType1>

    return %MEM_PERMUTE : tensor<1x10x48x329x!qElemType1>

    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {
    // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 6]
    // CHECK-SAME:      } : tensor<1x48x329x10x!qElemType> -> tensor<1x48x329x16x!qElemType2>

    // CHECK:       [[PERMUTE_CAST1:%.+]] = IE.PermuteCast([[EXPAND]]) {
    // CHECK-SAME:          dst_order = #NHWC, mem_perm = #NCHW
    // CHECK-SAME:      } : tensor<1x48x329x16x!qElemType2> -> tensor<1x16x48x329x!qElemType3, {order = #NHWC}>

    // CHECK:       [[POOLING:%.+]] = IE.MaxPool([[PERMUTE_CAST1]]) {
    // CHECK-SAME:          kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x16x48x329x!qElemType3, {order = #NHWC}> -> tensor<1x16x48x329x!qElemType3>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[POOLING]] [0, 0, 0, 0] [1, 10, 48, 329] : tensor<1x16x48x329x!qElemType3> to tensor<1x10x48x329x!qElemType1>

    // CHECK:       return [[SLICE]] : tensor<1x10x48x329x!qElemType1>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotConvertMemPermuteToPoolWithExpandDueToSmallSize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x24x21x63xf16, {order = #NHWC}>
func.func @NotConvertMemPermuteToPoolWithExpandDueToSmallSize(%arg0: tensor<1x24x21x63xf16, {order = #NHWC}>)
        -> tensor<1x21x24x63xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW, mem_perm = #NCWH
    } : tensor<1x24x21x63xf16, {order = #NHWC}> -> tensor<1x21x24x63xf16>

    return %MEM_PERMUTE : tensor<1x21x24x63xf16>

    // A larger total size is preferable for ODU permute
    // This case is 62_KB but threshold is 128_KB

    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {
    // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NCWH
    // CHECK-SAME:      } : tensor<1x24x21x63xf16, {order = #NHWC}> -> tensor<1x21x24x63xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x21x24x63xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotConvertMemPermuteToPoolWithExpandDueToLargeExpansion
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x24x57x65xf16>
func.func @NotConvertMemPermuteToPoolWithExpandDueToLargeExpansion(%arg0: tensor<1x24x57x65xf16>)
        -> tensor<1x24x65x57xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW, mem_perm = #NCWH
    } : tensor<1x24x57x65xf16> -> tensor<1x24x65x57xf16>

    return %MEM_PERMUTE : tensor<1x24x65x57xf16>

    // A smaller expand size is preferable as it minimizes unnecessary data movement
    // This case is 15 but threshold is "alignedChannel / 2 = 8"

    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {
    // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NCWH
    // CHECK-SAME:      } : tensor<1x24x57x65xf16> -> tensor<1x24x65x57xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x24x65x57xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotConvertMemPermuteToPoolWithExpandDueToSmallMemH
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x1368x65xf16>
func.func @NotConvertMemPermuteToPoolWithExpandDueToSmallMemH(%arg0: tensor<1x1x1368x65xf16>)
        -> tensor<1x1x65x1368xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW, mem_perm = #NCWH
    } : tensor<1x1x1368x65xf16> -> tensor<1x1x65x1368xf16>

    return %MEM_PERMUTE : tensor<1x1x65x1368xf16>

    // Larger H size improves HW efficiency and evenly distributing H across clusters
    // This case is 1 but threshold is "_numClusters * 2"

    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {
    // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NCWH
    // CHECK-SAME:      } : tensor<1x1x1368x65xf16> -> tensor<1x1x65x1368xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<1x1x65x1368xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertMemPermuteToPermuteQuantize
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x9x86016x1xf16>
func.func @ConvertMemPermuteToPermuteQuantize(%arg0: tensor<1x9x86016x1xf16>)
        -> tensor<1x9x86016x1xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
         dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x9x86016x1xf16> -> tensor<1x9x86016x1xf16, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<1x9x86016x1xf16, {order = #NHWC}>

    // CHECK: [[PERMUTE_QUANT:%.+]] = IE.PermuteQuantize([[INPUT]])
    // CHECK-SAME:     {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
    // CHECK-SAME:     tensor<1x9x86016x1xf16> -> tensor<1x9x86016x1xf16, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANT]] : tensor<1x9x86016x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertMemPermuteToPermuteQuantizeDueToHW
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x13x13x3xf16>
func.func @NotConvertMemPermuteToPermuteQuantizeDueToHW(%arg0: tensor<1x13x13x3xf16>)
        -> tensor<1x13x13x3xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x13x13x3xf16> -> tensor<1x13x13x3xf16, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<1x13x13x3xf16, {order = #NHWC}>

    // CHECK-NOT: IE.PermuteQuantize
    // CHECK: [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT]])
    // CHECK:   {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   tensor<1x13x13x3xf16> -> tensor<1x13x13x3xf16, {order = #NHWC}>
    // CHECK: return [[MEM_PERMUTE]] : tensor<1x13x13x3xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#CHWN = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @ConvertMemPermuteWithCNHWInput
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x13x16x80xf16, {order = #map}>
func.func @ConvertMemPermuteWithCNHWInput(%arg0: tensor<1x13x16x80xf16, {order = #CNHW}>)
        -> tensor<1x13x16x80xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #CHWN
    } : tensor<1x13x16x80xf16, {order = #CNHW}> -> tensor<1x13x16x80xf16, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<1x13x16x80xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.MemPermute

    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #map
    // CHECK-SAME:  } : tensor<1x13x16x80xf16, {order = #map}> -> tensor<1x13x16x80xf16>

    // CHECK:       [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[IN_PERMUTE_CAST]]) {
    // CHECK-SAME:      dstElemType = f16,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<1x13x16x80xf16> -> tensor<1x13x16x80xf16, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x13x16x80xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#CHWN = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @ConvertMemPermuteWithCNHWInputNCHWOutput
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x13x16x80xf16, {order = #map}>
func.func @ConvertMemPermuteWithCNHWInputNCHWOutput(%arg0: tensor<1x13x16x80xf16, {order = #CNHW}>)
        -> tensor<1x16x80x13xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW, mem_perm = #CHWN
    } : tensor<1x13x16x80xf16, {order = #CNHW}> -> tensor<1x16x80x13xf16>

    return %MEM_PERMUTE : tensor<1x16x80x13xf16>

    // CHECK-NOT:   IE.MemPermute

    // CHECK:       [[IN_PERMUTE_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #map
    // CHECK-SAME:  } : tensor<1x13x16x80xf16, {order = #map}> -> tensor<1x13x16x80xf16>

    // CHECK:       [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[IN_PERMUTE_CAST]]) {
    // CHECK-SAME:      dstElemType = f16,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<1x13x16x80xf16> -> tensor<1x13x16x80xf16, {order = #NHWC}>

    // CHECK:       [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[PERMUTE_QUANTIZE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  } : tensor<1x13x16x80xf16, {order = #NHWC}> -> tensor<1x16x80x13xf16>

    // CHECK:       return [[OUT_PERMUTE_CAST]] : tensor<1x16x80x13xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#WCHN = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2, d0)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

!intype = tensor<32x3072x1x1x!quant.uniform<i8:f16, 9.8075869027525187E-4>, {order = #map}>
!outtype = tensor<32x3072x1x1x!quant.uniform<i8:f16, 9.8075869027525187E-4>, {order = #NHWC}>

// CHECK-LABEL: @ConvertToDPUPermuteWith2AxisAdaptation
// CHECK-SAME: [[ARG0:%.+]]: tensor<32x3072x1x1x!qElemType, {order = #map}>
func.func @ConvertToDPUPermuteWith2AxisAdaptation(%arg0: !intype) -> !outtype {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #WCHN} : !intype -> !outtype

    return %0 : !outtype
}

// CHECK:     [[IN_PERM:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NHWC, mem_perm = #map1}
// CHECK-SAME:    : tensor<32x3072x1x1x!qElemType, {order = #map}> -> tensor<1x32x3072x1x!qElemType, {order = #NHWC}>
// CHECK:     [[MAX_POOL:%.+]] = IE.MaxPool([[IN_PERM]])
// CHECK-SAME:    {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
// CHECK-SAME:    : tensor<1x32x3072x1x!qElemType, {order = #NHWC}> -> tensor<1x32x3072x1x!qElemType, {order = #NCWH}>
// CHECK:     [[OUT_PERM:%.+]] = IE.PermuteCast([[MAX_POOL]]) {dst_order = #NHWC, mem_perm = #map1}
// CHECK-SAME:    : tensor<1x32x3072x1x!qElemType, {order = #NCWH}> -> tensor<32x3072x1x1x!qElemType, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#WCHN = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2, d0)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

!intype = tensor<32x1x3072x1x!quant.uniform<i8:f16, 9.8075869027525187E-4>, {order = #map}>
!outtype = tensor<32x1x3072x1x!quant.uniform<i8:f16, 9.8075869027525187E-4>, {order = #NHWC}>

// CHECK-LABEL: @ConvertToDPUPermuteWith2AxisAdaptationScenario2
// CHECK-SAME: [[ARG0:%.+]]: tensor<32x1x3072x1x!qElemType, {order = #map}>
func.func @ConvertToDPUPermuteWith2AxisAdaptationScenario2(%arg0: !intype) -> !outtype {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #WCHN} : !intype -> !outtype

    return %0 : !outtype
}

// CHECK:     [[IN_PERM:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW}
// CHECK-SAME:    : tensor<32x1x3072x1x!qElemType, {order = #map}> -> tensor<1x32x3072x1x!qElemType, {order = #NHWC}>
// CHECK:     [[MAX_POOL:%.+]] = IE.MaxPool([[IN_PERM]])
// CHECK-SAME:    {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
// CHECK-SAME:    : tensor<1x32x3072x1x!qElemType, {order = #NHWC}> -> tensor<1x32x3072x1x!qElemType, {order = #NCWH}>
// CHECK:     [[OUT_PERM:%.+]] = IE.PermuteCast([[MAX_POOL]]) {dst_order = #NHWC, mem_perm = #map1}
// CHECK-SAME:    : tensor<1x32x3072x1x!qElemType, {order = #NCWH}> -> tensor<32x1x3072x1x!qElemType, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#HWCN = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1, d0)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

!intype = tensor<64x64x1x1xf16>
!outtype = tensor<1x1x64x64xf16>

// CHECK-LABEL: @ConvertToDPUPermuteSameValOn2Axis
// CHECK-SAME: [[ARG0:%.+]]: tensor<64x64x1x1xf16>
func.func @ConvertToDPUPermuteSameValOn2Axis(%arg0: !intype) -> !outtype {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #HWCN} : !intype -> !outtype

    return %0 : !outtype
}

// CHECK:     [[IN_PERM:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NHWC, mem_perm = #map}
// CHECK-SAME:    : tensor<64x64x1x1xf16> -> tensor<1x64x64x1xf16, {order = #NHWC}>
// CHECK:     [[MAX_POOL:%.+]] = IE.MaxPool([[IN_PERM]])
// CHECK-SAME:    {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
// CHECK-SAME:    : tensor<1x64x64x1xf16, {order = #NHWC}> -> tensor<1x64x64x1xf16, {order = #NCWH}>
// CHECK:     [[OUT_PERM:%.+]] = IE.PermuteCast([[MAX_POOL]]) {dst_order = #NCHW, mem_perm = #NHCW}
// CHECK-SAME:    : tensor<1x64x64x1xf16, {order = #NCWH}> -> tensor<1x1x64x64xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#WCHN = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2, d0)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

!qElemType = !quant.uniform<u8:f16:0, {0.956:128, 0.785:128, 0.567:128, 0.785:128,
                                       0.956:128, 0.785:128, 0.567:128, 0.785:128,
                                       0.956:128, 0.785:128, 0.567:128, 0.785:128,
                                       0.956:128, 0.785:128, 0.567:128, 0.785:128}>

!intype = tensor<16x1x3072x1x!qElemType, {order = #map}>
!outtype = tensor<16x1x3072x1x!qElemType, {order = #NHWC}>

// CHECK-DAG: [[QELEM_TYPE:!.+]] = !quant.uniform<u8:f16:0,

// CHECK: @ConvertToDPUPermuteWith2AxisAdaptationPerAxisQuant
// CHECK-SAME: [[ARG0:%.+]]: tensor<16x1x3072x1x[[QELEM_TYPE]], {order = #map}>
func.func @ConvertToDPUPermuteWith2AxisAdaptationPerAxisQuant(%arg0: !intype) -> !outtype {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #WCHN} : !intype -> !outtype

    return %0 : !outtype

// CHECK:      [[PERMUTECAST_IN:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<16x1x3072x1x!qElemType, {order = #map}> -> tensor<1x16x3072x1x!qElemType1, {order = #NHWC}>
// CHECK:      [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x3072x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x3072x1x!qElemType1, {order = #NCWH}>
// CHECK:      [[PERMUTECAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NHWC, mem_perm = #map1} : tensor<1x16x3072x1x!qElemType1, {order = #NCWH}> -> tensor<16x1x3072x1x!qElemType, {order = #NHWC}>
// CHECK:      return [[PERMUTECAST_OUT]] : tensor<16x1x3072x1x!qElemType, {order = #NHWC}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ConvertMemPermuteWithReshapeN_0
func.func @ConvertMemPermuteWithReshapeN_0(%arg0: tensor<2x256x16x64xf16>)
    -> tensor<2x16x64x256xf16> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW,
        mem_perm = #NHWC
    } : tensor<2x256x16x64xf16> -> tensor<2x16x64x256xf16>

    return %PERMUTE : tensor<2x16x64x256xf16>

    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 2, 256, 1024]} inputs(%arg0 : tensor<2x256x16x64xf16>) -> tensor<1x2x256x1024xf16>
    // CHECK:       [[PERMUTECAST_IN:%.+]] = IE.PermuteCast([[SHAPECAST_IN]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x256x1024xf16> -> tensor<1x1024x2x256xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x2x256xf16, {order = #NHWC}> -> tensor<1x1024x2x256xf16, {order = #NHCW}>
    // CHECK:       [[PERMUTECAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1024x2x256xf16, {order = #NHCW}> -> tensor<1x2x1024x256xf16>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [2, 16, 64, 256]} inputs([[PERMUTECAST_OUT]] : tensor<1x2x1024x256xf16>) -> tensor<2x16x64x256xf16>
    // CHECK:       return [[SHAPECAST_OUT]] : tensor<2x16x64x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ConvertMemPermuteWithReshapeN_1
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<4x75x16x64xf16>
func.func @ConvertMemPermuteWithReshapeN_1(%arg0: tensor<4x75x16x64xf16>) -> tensor<4x16x64x75xf16> {
    %0 = IE.MemPermute(%arg0) {dstElemType = f16, dst_order = #NCHW, mem_perm = #NHWC} :
        tensor<4x75x16x64xf16> -> tensor<4x16x64x75xf16>

    return %0 : tensor<4x16x64x75xf16>

    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 4, 75, 1024]} inputs([[INPUT]] : tensor<4x75x16x64xf16>) -> tensor<1x4x75x1024xf16>
    // CHECK:       [[PERMUTECAST_IN:%.+]] = IE.PermuteCast([[SHAPECAST_IN]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x75x1024xf16> -> tensor<1x1024x4x75xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x4x75xf16, {order = #NHWC}> -> tensor<1x1024x4x75xf16, {order = #NHCW}>
    // CHECK:       [[PERMUTECAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1024x4x75xf16, {order = #NHCW}> -> tensor<1x4x1024x75xf16>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [4, 16, 64, 75]} inputs([[PERMUTECAST_OUT]] : tensor<1x4x1024x75xf16>) -> tensor<4x16x64x75xf16>
    // CHECK:       return [[SHAPECAST_OUT]] : tensor<4x16x64x75xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>

// CHECK-LABEL: @ConvertMemPermuteWithPermuteCast_Batched
func.func @ConvertMemPermuteWithPermuteCast_Batched(%arg0: tensor<2x256x400x1xf16>)
    -> tensor<2x400x1x256xf16> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW,
        mem_perm = #NHWC
    } : tensor<2x256x400x1xf16> -> tensor<2x400x1x256xf16>

    return %PERMUTE : tensor<2x400x1x256xf16>

    // CHECK:       [[PERMUTECAST_IN:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #map} : tensor<2x256x400x1xf16> -> tensor<1x400x2x256xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x400x2x256xf16, {order = #NHWC}> -> tensor<1x400x2x256xf16, {order = #NHCW}>
    // CHECK:       [[PERMUTECAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #map1} : tensor<1x400x2x256xf16, {order = #NHCW}> -> tensor<2x400x1x256xf16>
    // CHECK:       return [[PERMUTECAST_OUT]] : tensor<2x400x1x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertMemPermuteWithPermuteCast_1CH1
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x380x720x1xf16>
func.func @ConvertMemPermuteWithPermuteCast_1CH1(%arg0: tensor<1x380x720x1xf16>) -> tensor<1x720x380x1xf16> {
    %0 = IE.MemPermute(%arg0) {dstElemType = f16, dst_order = #NCHW, mem_perm = #NHCW} :
        tensor<1x380x720x1xf16> -> tensor<1x720x380x1xf16>

    return %0 : tensor<1x720x380x1xf16>

    // CHECK:       [[PERMUTECAST_IN:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x380x720x1xf16> -> tensor<1x720x380x1xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x720x380x1xf16, {order = #NHWC}> -> tensor<1x720x380x1xf16, {order = #NCWH}>
    // CHECK:       [[PERMUTECAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x720x380x1xf16, {order = #NCWH}> -> tensor<1x720x380x1xf16>
    // CHECK:       return [[PERMUTECAST_OUT]] : tensor<1x720x380x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertMemPermuteWithReshape1C1W_0
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x1x128xf16, {order = #NHWC}>
func.func @ConvertMemPermuteWithReshape1C1W_0(%arg0: tensor<1x16x1x128xf16, {order = #NHWC}>) -> tensor<1x16x1x128xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x1x128xf16>

    return %0 : tensor<1x16x1x128xf16>

    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[INPUT]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1x128xf16, {order = #NHWC}> -> tensor<1x16x1x128xf16>
    // CHECK:       return [[MAXPOOL]] : tensor<1x16x1x128xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertMemPermuteWithReshape1C1W_1
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x32x3072xf16>
func.func @ConvertMemPermuteWithReshape1C1W_1(%arg0: tensor<1x1x32x3072xf16>) -> tensor<1x1x3072x32xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x32x3072xf16> -> tensor<1x1x3072x32xf16>

    return %0 : tensor<1x1x3072x32xf16>

    // CHECK:       [[PERMUTECAST_IN:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x32x3072xf16> -> tensor<1x3072x1x32xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3072x1x32xf16, {order = #NHWC}> -> tensor<1x3072x1x32xf16, {order = #NHCW}>
    // CHECK:       [[PERMUTECAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3072x1x32xf16, {order = #NHCW}> -> tensor<1x1x3072x32xf16>
    // CHECK:       return [[PERMUTECAST_OUT:%.+]] : tensor<1x1x3072x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertMemPermuteWithReshape1C1W_2
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x64x64xf16>
func.func @ConvertMemPermuteWithReshape1C1W_2(%arg0: tensor<1x1x64x64xf16>) -> tensor<1x1x64x64xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x64x64xf16> -> tensor<1x1x64x64xf16>

    return %0 : tensor<1x1x64x64xf16>

    // CHECK:       [[PERMUTECAST_IN:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x64x64xf16> -> tensor<1x64x1x64xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x1x64xf16, {order = #NHWC}> -> tensor<1x64x1x64xf16, {order = #NHCW}>
    // CHECK:       [[PERMUTECAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x1x64xf16, {order = #NHCW}> -> tensor<1x1x64x64xf16>
    // CHECK:       return [[PERMUTECAST_OUT]] : tensor<1x1x64x64xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039686492845123888>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @DoNotConvertMemPermuteIntoMultipleMaxPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<12x64x64x4x!qElemType>
func.func @DoNotConvertMemPermuteIntoMultipleMaxPool(%arg0: tensor<12x64x64x4x!qElemType>) -> tensor<12x4x64x64x!qElemType> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<12x64x64x4x!qElemType> -> tensor<12x4x64x64x!qElemType>

    return %0 : tensor<12x4x64x64x!qElemType>

    // CHECK:       [[SHAPE_CAST1:%.+]] = IE.ShapeCast {shape = [1, 12, 4096, 4]} inputs([[INPUT]] : tensor<12x64x64x4x!qElemType>) -> tensor<1x12x4096x4x!qElemType>
    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[SHAPE_CAST1]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x12x4096x4x!qElemType> -> tensor<1x12x4x4096x!qElemType>
    // CHECK:       [[SHAPE_CAST2:%.+]] = IE.ShapeCast {shape = [12, 4, 64, 64]} inputs([[MEMPERMUTE]] : tensor<1x12x4x4096x!qElemType>) -> tensor<12x4x64x64x!qElemType>
    // CHECK:       return [[SHAPE_CAST2]] : tensor<12x4x64x64x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @InsertMaxPoolBeforeMemPermute
func.func @InsertMaxPoolBeforeMemPermute(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>)
    -> tensor<1x32x16x64xf16, {order = #NHWC}> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x16x64xf16, {order = #NHWC}>

    return %PERMUTE : tensor<1x32x16x64xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool(%arg0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16, {order = #NCWH}>
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x32x64xf16, {order = #NCWH}> -> tensor<1x32x16x64xf16, {order = #NHWC}>
    // CHECK:       return [[PERMUTECAST]] : tensor<1x32x16x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MemPermuteWithSmallByteSize
func.func @MemPermuteWithSmallByteSize(%arg0: tensor<1x16x32x2xf16, {order = #NHWC}>)
    -> tensor<1x32x16x2xf16, {order = #NHWC}> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x16x32x2xf16, {order = #NHWC}> -> tensor<1x32x16x2xf16, {order = #NHWC}>

    return %PERMUTE : tensor<1x32x16x2xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool(%arg0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x32x2xf16, {order = #NHWC}> -> tensor<1x16x32x2xf16, {order = #NCWH}>
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x32x2xf16, {order = #NCWH}> -> tensor<1x32x16x2xf16, {order = #NHWC}>
    // CHECK:       return [[PERMUTECAST]] : tensor<1x32x16x2xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u16:f16, 0.0039686492845123888>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @DoNotConvertQuantized16bitsMemPermuteIntoMaxPoolNBatches
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<12x64x64x4x!qElemType>
func.func @DoNotConvertQuantized16bitsMemPermuteIntoMaxPoolNBatches(%arg0: tensor<12x64x64x4x!qElemType>) -> tensor<12x4x64x64x!qElemType> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<12x64x64x4x!qElemType> -> tensor<12x4x64x64x!qElemType>

    return %0 : tensor<12x4x64x64x!qElemType>

    // CHECK:       [[SHAPE_CAST1:%.+]] = IE.ShapeCast {shape = [1, 12, 4096, 4]} inputs([[INPUT]] : tensor<12x64x64x4x!qElemType>) -> tensor<1x12x4096x4x!qElemType>
    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[SHAPE_CAST1]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x12x4096x4x!qElemType> -> tensor<1x12x4x4096x!qElemType>
    // CHECK:       [[SHAPE_CAST2:%.+]] = IE.ShapeCast {shape = [12, 4, 64, 64]} inputs([[MEMPERMUTE]] : tensor<1x12x4x4096x!qElemType>) -> tensor<12x4x64x64x!qElemType>
    // CHECK:       return [[SHAPE_CAST2]] : tensor<12x4x64x64x!qElemType>
}

// -----

!qElemType = !quant.uniform<u16:f16, 0.0039686492845123888>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @DoNotConvertQuantized16bitsMemPermuteIntoMaxPool1Batch
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x64x4x!qElemType>
func.func @DoNotConvertQuantized16bitsMemPermuteIntoMaxPool1Batch(%arg0: tensor<1x64x64x4x!qElemType>) -> tensor<1x4x64x64x!qElemType> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x64x64x4x!qElemType> -> tensor<1x4x64x64x!qElemType>

    return %0 : tensor<1x4x64x64x!qElemType>

    // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x64x64x4x!qElemType> -> tensor<1x4x64x64x!qElemType>
    // CHECK:       return [[MEMPERMUTE]] : tensor<1x4x64x64x!qElemType>
}



