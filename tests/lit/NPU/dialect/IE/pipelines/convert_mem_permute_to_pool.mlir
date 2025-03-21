//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-mem-permute-to-pool --canonicalize %s | FileCheck %s
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

// CHECK-LABEL: @UnsupportedDimN
func.func @UnsupportedDimN(%arg0: tensor<25x14x14x2304xf16>) -> tensor<25x14x2304x14xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW,
        mem_perm = #NHWC
    } : tensor<25x14x14x2304xf16> -> tensor<25x14x2304x14xf16>

    return %MEM_PERMUTE : tensor<25x14x2304x14xf16>

    // CHECK-NOT:   IE.ShapeCast
    // CHECK-NOT:   IE.LayoutCast
    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[MEM_PERMUTE:%.+]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} :
    // CHECK-SAME:  tensor<25x14x14x2304xf16> -> tensor<25x14x2304x14xf16>
    // CHECK:       return [[MEM_PERMUTE]] : tensor<25x14x2304x14xf16>
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
#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

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
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[MAXPOOL_1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x80x7975xf16, {order = #NWCH}> -> tensor<1x7975x32x80xf16>
    // CHECK:       [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [7975, 1, 32, 80]} inputs([[PERMUTE_CAST_1]] : tensor<1x7975x32x80xf16>) -> tensor<7975x1x32x80xf16>

    // CHECK:       return [[SHAPE_CAST_OUT]] : tensor<7975x1x32x80xf16>
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

// CHECK-LABEL: @AdjustMemPermuteShape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x16x128xf16, {order = #NCHW}>
func.func @AdjustMemPermuteShape(%arg0: tensor<1x1024x16x128xf16, {order = #NCHW}>) -> tensor<16x128x1x1024xf16> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #HWNC} : tensor<1x1024x16x128xf16, {order = #NCHW}> -> tensor<16x128x1x1024xf16>

    return %MEM_PERMUTE : tensor<16x128x1x1024xf16>

    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 1024, 16, 128]} inputs(%arg0 : tensor<1x1024x16x128xf16, {order = #NCHW}>) -> tensor<1x1024x16x128xf16>
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[SHAPECAST_IN]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1024x16x128xf16> -> tensor<1x128x1024x16xf16, {order = #NHWC}>
    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x128x1024x16xf16, {order = #NHWC}> -> tensor<1x128x1024x16xf16, {order = #NWCH}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[MAX_POOL]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x1024x16xf16, {order = #NWCH}> -> tensor<1x16x128x1024xf16>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [16, 128, 1, 1024]} inputs([[PERMUTE_CAST_1]] : tensor<1x16x128x1024xf16>) -> tensor<16x128x1x1024xf16>

    // CHECK:       return [[SHAPECAST_OUT]] : tensor<16x128x1x1024xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#WCHN = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2, d0)>

// CHECK-LABEL: @AdjustMemPermuteShapeWithDimsOne
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1024x1x1x128xf16, {order = #NCHW}>
func.func @AdjustMemPermuteShapeWithDimsOne(%arg0: tensor<1024x1x1x128xf16, {order = #NCHW}>) -> tensor<128x1024x1x1xf16, {order = #NHWC}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #WCHN} : tensor<1024x1x1x128xf16, {order = #NCHW}> -> tensor<128x1024x1x1xf16, {order = #NHWC}>

    return %MEM_PERMUTE : tensor<128x1024x1x1xf16, {order = #NHWC}>

    // CHECK:       [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 1024, 1, 128]} inputs([[INPUT]] : tensor<1024x1x1x128xf16, {order = #NCHW}>) -> tensor<1x1024x1x128xf16>
    // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[SHAPECAST_IN]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1024x1x128xf16> -> tensor<1x128x1024x1xf16, {order = #NHWC}>
    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[PERMUTE_CAST]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x128x1024x1xf16, {order = #NHWC}> -> tensor<1x128x1024x1xf16, {order = #NCWH}>
    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[MAX_POOL]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x128x1024x1xf16, {order = #NCWH}> -> tensor<1x1024x128x1xf16, {order = #NHWC}>
    // CHECK:       [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [128, 1024, 1, 1]} inputs([[PERMUTE_CAST_1]] : tensor<1x1024x128x1xf16, {order = #NHWC}>) -> tensor<128x1024x1x1xf16, {order = #NHWC}>

    // CHECK:       return [[SHAPECAST_OUT]] : tensor<128x1024x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#HNWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>

// CHECK-LABEL: @AdjustMemPermuteWithDimNChanged
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x256x192xf16, {order = #NHWC}>
func.func @AdjustMemPermuteWithDimNChanged(%arg0: tensor<1x1x256x192xf16, {order = #NHWC}>) -> tensor<1x1x256x192xf16, {order = #map}> {
    %MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #map, mem_perm = #HNWC}
            : tensor<1x1x256x192xf16, {order = #NHWC}> -> tensor<1x1x256x192xf16, {order = #map}>

    return %MEM_PERMUTE : tensor<1x1x256x192xf16, {order = #map}>

    // CHECK:        [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 192, 256, 1]} inputs([[INPUT]] : tensor<1x1x256x192xf16, {order = #NHWC}>)
    // CHECK-SAME:           -> tensor<1x192x256x1xf16, {order = #NHWC}>
    // CHECK:        [[MAX_POOL:%.+]] = IE.MaxPool([[SHAPE_CAST]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:           : tensor<1x192x256x1xf16, {order = #NHWC}> -> tensor<1x192x256x1xf16, {order = #NCWH}>
    // CHECK:        [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[MAX_POOL]]) {dst_order = #map, mem_perm = #NCHW}
    // CHECK-SAME:           : tensor<1x192x256x1xf16, {order = #NCWH}> -> tensor<192x1x256x1xf16, {order = #map}>
    // CHECK:        [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [1, 1, 256, 192]} inputs([[PERMUTE_CAST]] : tensor<192x1x256x1xf16, {order = #map}>)
    // CHECK-SAME:           -> tensor<1x1x256x192xf16, {order = #map}>

    // CHECK:        return [[SHAPE_CAST_2]] : tensor<1x1x256x192xf16, {order = #map}>
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
