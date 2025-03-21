//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-mem-permute-through-add %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertAddNWCH
func.func @ConvertAddNWCH(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x8x4x76xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16>

    return %OUT_MEM_PERMUTE : tensor<1x8x4x76xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4x76xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4x76xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 4, 76]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x4x76xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @ConvertAddNWHC
func.func @ConvertAddNWHC(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x8x76x4xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NWHC
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x76x4xf16>

    return %OUT_MEM_PERMUTE : tensor<1x8x76x4xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x76x4xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x76x4xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 76, 4]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x76x4xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAddNCWH
func.func @ConvertAddNCWH(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x4x8x76xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NCWH
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x4x8x76xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x8x76xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NCWH
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x4x8x76xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NCWH
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x4x8x76xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 4, 8, 76]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x4x8x76xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAddNCHW
func.func @ConvertAddNCHW(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x4x76x8xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NCHW
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x4x76x8xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x76x8xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x4x76x8xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x4x76x8xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 4, 76, 8]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x4x76x8xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ConvertAddNHCW
func.func @ConvertAddNHCW(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x76x4x8xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NHCW
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x76x4x8xf16>

    return %OUT_MEM_PERMUTE : tensor<1x76x4x8xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NHCW
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x76x4x8xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NHCW
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x76x4x8xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 76, 4, 8]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x76x4x8xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertAddWithPostOp
func.func @ConvertAddWithPostOp(%arg0 : tensor<1x8x4x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x8x4x76xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.1}>
    } : tensor<1x16x19x8xf16, {order = #NHWC}>,
        tensor<1x16x19x8xf16, {order = #NHWC}>
            -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 76]
    } inputs(%ADD : tensor<1x16x19x8xf16, {order = #NHWC}>) -> tensor<1x8x4x76xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x8x4x76xf16, {order = #NHWC}> -> tensor<1x8x4x76xf16>

    return %OUT_MEM_PERMUTE : tensor<1x8x4x76xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  }

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x8x4x76xf16>)

    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  }

    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 16, 19, 8]
    // CHECK-SAME:  } inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x8x4x76xf16>)

    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC}

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]])
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW}
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {
    // CHECK-SAME:      shape = [1, 8, 4, 76]
    // CHECK-SAME:  } inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8xf16>)

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x8x4x76xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteAddWithQuantizeCast
func.func @ConvertPermuteAddWithQuantizeCast(%arg0 : tensor<1x4x8x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x4x8x76x!qElemType> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs(%LHS_MEM_PERMUTE : tensor<1x4x8x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs(%RHS_MEM_PERMUTE : tensor<1x4x8x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType1, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs(%ADD : tensor<1x16x19x8x!qElemType1, {order = #NHWC}>) -> tensor<1x4x8x76x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%OUT_SHAPE_CAST) {dstElemType = !qElemType} : tensor<1x4x8x76x!qElemType1, {order = #NHWC}> -> tensor<1x4x8x76x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76x!qElemType, {order = #NHWC}> -> tensor<1x4x8x76x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x4x8x76x!qElemType>

    // CHECK:   [[LHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76xf16, {order = #NHWC}> -> tensor<1x4x8x76xf16>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76xf16, {order = #NHWC}> -> tensor<1x4x8x76xf16>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x16x19x8x!qElemType1, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType1>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8x!qElemType1>) -> tensor<1x4x8x76x!qElemType1>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_SHAPE_CAST]]) {dstElemType = !qElemType} : tensor<1x4x8x76x!qElemType1> -> tensor<1x4x8x76x!qElemType>

    // CHECK:   return [[OUT_QUANTIZE_CAST]] : tensor<1x4x8x76x!qElemType>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteShapeCastAdd
func.func @ConvertPermuteShapeCastAdd(%arg0 : tensor<1x4x512x512xf16>, %arg1 : tensor<1x4x512x512xf16>) -> tensor<1x4x512x512xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 512, 128]} inputs(%LHS_MEM_PERMUTE : tensor<1x4x512x512xf16, {order = #NHWC}>) -> tensor<1x16x512x128xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 512, 128]} inputs(%RHS_MEM_PERMUTE : tensor<1x4x512x512xf16, {order = #NHWC}>) -> tensor<1x16x512x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x512x128xf16, {order = #NHWC}>, tensor<1x16x512x128xf16, {order = #NHWC}> -> tensor<1x16x512x128xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 4, 512, 512]} inputs(%ADD : tensor<1x16x512x128xf16, {order = #NHWC}>) -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x512x512xf16>

    // CHECK:   [[LHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_OUT_MEM_PERMUTE]], [[RHS_OUT_MEM_PERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_PERMUTE_CAST:%.*]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[OUT_PERMUTE_CAST]] : tensor<1x4x512x512xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>


// CHECK-LABEL: @ConvertPermuteShapeCastAddWithQuantizeCast
func.func @ConvertPermuteShapeCastAddWithQuantizeCast(%arg0 : tensor<1x4x512x512xf16>, %arg1 : tensor<1x4x512x512xf16>) -> tensor<1x4x512x512x!qElemType> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 512, 128]} inputs(%LHS_MEM_PERMUTE : tensor<1x4x512x512xf16, {order = #NHWC}>) -> tensor<1x16x512x128xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 512, 128]} inputs(%RHS_MEM_PERMUTE : tensor<1x4x512x512xf16, {order = #NHWC}>) -> tensor<1x16x512x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x512x128xf16, {order = #NHWC}>, tensor<1x16x512x128xf16, {order = #NHWC}> -> tensor<1x16x512x128x!qElemType1, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 4, 512, 512]} inputs(%ADD : tensor<1x16x512x128x!qElemType1, {order = #NHWC}>) -> tensor<1x4x512x512x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%OUT_SHAPE_CAST) {dstElemType = !qElemType} : tensor<1x4x512x512x!qElemType1, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512x!qElemType, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x4x512x512x!qElemType>

    // CHECK:   [[LHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_OUT_MEM_PERMUTE]], [[RHS_OUT_MEM_PERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_PERMUTE_CAST:%.*]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512x!qElemType1, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType1>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_PERMUTE_CAST]]) {dstElemType = !qElemType} : tensor<1x4x512x512x!qElemType1> -> tensor<1x4x512x512x!qElemType>
    // CHECK:   return [[OUT_QUANTIZE_CAST]] : tensor<1x4x512x512x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteAddWithQuantizeCastNoShapeCast
func.func @ConvertPermuteAddWithQuantizeCastNoShapeCast(%arg0 : tensor<1x8x8x8xf16>, %arg1 : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%ADD) {dstElemType = !qElemType} : tensor<1x8x8x8x!qElemType1, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x8x8x!qElemType>

    // CHECK:   [[LHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_OUT_MEM_PERMUTE]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_OUT_MEM_PERMUTE]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x8x8x8x!qElemType1, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_LAYOUT_CAST]]) {dstElemType = !qElemType} : tensor<1x8x8x8x!qElemType1> -> tensor<1x8x8x8x!qElemType>

    // CHECK:   return [[OUT_QUANTIZE_CAST]] : tensor<1x8x8x8x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteAddNoShapeCast
func.func @ConvertPermuteAddNoShapeCast(%arg0 : tensor<1x8x8x8xf16>, %arg1 : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x8x8x!qElemType>

    // CHECK:   [[LHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_OUT_MEM_PERMUTE]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_OUT_MEM_PERMUTE]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>

    // CHECK:   return [[OUT_LAYOUT_CAST]] : tensor<1x8x8x8x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddWithQuantizeCast
func.func @ConvertPermuteQuantizeAddWithQuantizeCast(%arg0 : tensor<1x4x8x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x4x8x76x!qElemType> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x4x8x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x4x8x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType1, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs(%ADD : tensor<1x16x19x8x!qElemType1, {order = #NHWC}>) -> tensor<1x4x8x76x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%OUT_SHAPE_CAST) {dstElemType = !qElemType} : tensor<1x4x8x76x!qElemType1, {order = #NHWC}> -> tensor<1x4x8x76x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76x!qElemType, {order = #NHWC}> -> tensor<1x4x8x76x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x4x8x76x!qElemType>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76xf16, {order = #NHWC}> -> tensor<1x4x8x76xf16>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76xf16, {order = #NHWC}> -> tensor<1x4x8x76xf16>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x16x19x8x!qElemType1, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType1>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8x!qElemType1>) -> tensor<1x4x8x76x!qElemType1>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_SHAPE_CAST]]) {dstElemType = !qElemType} : tensor<1x4x8x76x!qElemType1> -> tensor<1x4x8x76x!qElemType>

    // CHECK:   return [[OUT_QUANTIZE_CAST]] : tensor<1x4x8x76x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NoPopagateIfPemutationsCanNotFold
func.func @NoPopagateIfPemutationsCanNotFold(%arg0 : tensor<1x8x4096x4096xf16>, %arg1 : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs(%ADD : tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>) -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%OUT_SHAPE_CAST) {dstElemType = !qElemType} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>

    return %OUT_MEM_PERMUTE : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[LHS_PERMUTEQUANTIZE]] : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[RHS_PERMUTEQUANTIZE]] : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_SHAPE_CAST]], [[RHS_SHAPE_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[ADD]] : tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>) -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_SHAPE_CAST]]) {dstElemType = !qElemType} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>

    // CHECK:   return [[OUT_QUANTIZE_CAST]] : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAdd
func.func @ConvertPermuteQuantizeAdd(%arg0 : tensor<1x4x8x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x4x8x76x!qElemType> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x4x8x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x4x8x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs(%ADD : tensor<1x16x19x8x!qElemType, {order = #NHWC}>) -> tensor<1x4x8x76x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76x!qElemType, {order = #NHWC}> -> tensor<1x4x8x76x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x4x8x76x!qElemType>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76xf16, {order = #NHWC}> -> tensor<1x4x8x76xf16>
    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[LHS_OUT_MEM_PERMUTE]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76xf16, {order = #NHWC}> -> tensor<1x4x8x76xf16>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[RHS_OUT_MEM_PERMUTE]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_SHAPE_CAST]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x16x19x8x!qElemType, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType>
    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs([[OUT_LAYOUT_CAST]] : tensor<1x16x19x8x!qElemType>) -> tensor<1x4x8x76x!qElemType>

    // CHECK:   return [[OUT_SHAPE_CAST]] : tensor<1x4x8x76x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddWithQuantizeCastNoShapeCast
func.func @ConvertPermuteQuantizeAddWithQuantizeCastNoShapeCast(%arg0 : tensor<1x8x8x8xf16>, %arg1 : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_PERMUTEQUANTIZE, %RHS_PERMUTEQUANTIZE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%ADD) {dstElemType = !qElemType} : tensor<1x8x8x8x!qElemType1, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x8x8x!qElemType>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_OUT_MEM_PERMUTE]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_OUT_MEM_PERMUTE]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x8x8x8x!qElemType1, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1>
    // CHECK:   [[OUT_QUANTIZE_CAST:%.*]] = IE.QuantizeCast([[OUT_LAYOUT_CAST]]) {dstElemType = !qElemType} : tensor<1x8x8x8x!qElemType1> -> tensor<1x8x8x8x!qElemType>

    // CHECK:   return [[OUT_QUANTIZE_CAST]] : tensor<1x8x8x8x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddNoShapeCast
func.func @ConvertPermuteQuantizeAddNoShapeCast(%arg0 : tensor<1x8x8x8xf16>, %arg1 : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_PERMUTEQUANTIZE, %RHS_PERMUTEQUANTIZE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x8x8x!qElemType>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    // CHECK:   [[LHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[LHS_OUT_MEM_PERMUTE]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_PERMUTEQUANTIZE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16>
    // CHECK:   [[RHS_LAYOUT_CAST:%.*]] = IE.LayoutCast([[RHS_OUT_MEM_PERMUTE]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_LAYOUT_CAST]], [[RHS_LAYOUT_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>
    // CHECK:   [[OUT_LAYOUT_CAST:%.*]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>

    // CHECK:   return [[OUT_LAYOUT_CAST]] : tensor<1x8x8x8x!qElemType>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddWithSoftmax
func.func @ConvertPermuteQuantizeAddWithSoftmax(%arg0 : tensor<1x2x512x512xf16>, %arg1 : tensor<1x2x512x512xf16>) -> tensor<1x2x512x512xf16> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>
    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs(%ADD : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %OUT_SOFTMAX = IE.SoftMax(%OUT_SHAPE_CAST) {axisInd = 3} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SOFTMAX) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>

    return %OUT_MEM_PERMUTE : tensor<1x2x512x512xf16>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.*]] = IE.MemPermute([[RHS_PERMUTEQUANTIZE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_OUT_MEM_PERMUTE]], [[RHS_OUT_MEM_PERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x2x512xf16, {order = #NHWC}>, tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_PERMUTE_CAST:%.*]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    // CHECK:   [[OUT_SOFTMAX:%.*]] = IE.SoftMax([[OUT_PERMUTE_CAST]]) {axisInd = 3 : i64} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16>

    // CHECK:   return [[OUT_SOFTMAX]] : tensor<1x2x512x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @NotSwapSoftmaxMemPermuteIfCanNotFuse
func.func @NotSwapSoftmaxMemPermuteIfCanNotFuse(%arg0 : tensor<1x2x512x512xf16>, %arg1 : tensor<1x2x512x512xf16>) -> tensor<1x2x512x512xf16, {order = #NHWC}> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>
    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs(%ADD : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %OUT_SOFTMAX = IE.SoftMax(%OUT_SHAPE_CAST) {axisInd = 3} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SOFTMAX) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    return %OUT_MEM_PERMUTE : tensor<1x2x512x512xf16, {order = #NHWC}>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    // CHECK:   [[LHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs([[LHS_PERMUTEQUANTIZE]] : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK:   [[RHS_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs([[RHS_PERMUTEQUANTIZE]] : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_SHAPE_CAST]], [[RHS_SHAPE_CAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>

    // CHECK:   [[OUT_SHAPE_CAST:%.*]] = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs([[ADD]] : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_SOFTMAX:%.*]] = IE.SoftMax([[OUT_SHAPE_CAST]]) {axisInd = 3 : i64} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_MEMPERMUTE:%.*]] = IE.MemPermute([[OUT_SOFTMAX]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    // CHECK:   return [[OUT_MEMPERMUTE]] : tensor<1x2x512x512xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertOnlyOnePermuteLikeInput
func.func @ConvertOnlyOnePermuteLikeInput(%arg0 : tensor<1x2x512x512xf16>, %arg1 : tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %RHS_TILE = IE.Tile(%arg1) {repeats_values = [1, 2, 512, 1]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%RHS_TILE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs(%ADD : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>

    return %OUT_MEM_PERMUTE : tensor<1x2x512x512xf16>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_TILE:%.*]] = IE.Tile(%arg1) {repeats_values = [1, 2, 512, 1]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    // CHECK:   [[LHS_MEMPERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEMPERMUTE:%.*]] = IE.MemPermute([[RHS_TILE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_MEMPERMUTE]], [[RHS_MEMPERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x2x512xf16, {order = #NHWC}>, tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_PERMUTE_CAST:%.*]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    // CHECK:   return [[OUT_PERMUTE_CAST]] : tensor<1x2x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertOnlyOnePermuteLikeAndWithoutShapeCastInput
func.func @ConvertOnlyOnePermuteLikeAndWithoutShapeCastInput(%arg0 : tensor<1x16x128x128xf16>, %arg1 : tensor<1x1x1x128xf16, {order = #NHWC}>) -> tensor<1x16x128x128xf16> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x128x128xf16> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    %RHS_TILE = IE.Tile(%arg1) {repeats_values = [1, 16, 128, 1]} : tensor<1x1x1x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_PERMUTEQUANTIZE, %RHS_TILE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16>

    return %OUT_MEM_PERMUTE : tensor<1x16x128x128xf16>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x128x128xf16> -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK:   [[RHS_TILE:%.*]] = IE.Tile(%arg1) {repeats_values = [1, 16, 128, 1]} : tensor<1x1x1x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    // CHECK:   [[LHS_MEMPERMUTE:%.*]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x128x16x128xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEMPERMUTE:%.*]] = IE.MemPermute([[RHS_TILE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x128x16x128xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_MEMPERMUTE]], [[RHS_MEMPERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x16x128xf16, {order = #NHWC}>, tensor<1x128x16x128xf16, {order = #NHWC}> -> tensor<1x128x16x128xf16, {order = #NHWC}>
    // CHECK:   [[OUT_PERMUTE_CAST:%.*]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16>
    // CHECK:   return [[OUT_PERMUTE_CAST]] : tensor<1x16x128x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NoPopagateIfAddWithTwoOutputs
func.func @NoPopagateIfAddWithTwoOutputs(%arg0 : tensor<1x16x128x128xf16>, %arg1 : tensor<1x1x1x128xf16, {order = #NHWC}>) -> tensor<1x48x128x128xf16> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x128x128xf16> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    %RHS_TILE = IE.Tile(%arg1) {repeats_values = [1, 16, 128, 1]} : tensor<1x1x1x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_PERMUTEQUANTIZE, %RHS_TILE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    %MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16>

    %CONCAT = IE.Concat(%MEM_PERMUTE, %MEM_PERMUTE) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x128x128xf16>, tensor<1x16x128x128xf16> -> tensor<1x32x128x128xf16>

    %OUT_ADD = IE.Add(%ADD, %ADD) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16>

    %OUT_CONCAT = IE.Concat(%CONCAT, %OUT_MEM_PERMUTE) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x128x128xf16>, tensor<1x16x128x128xf16> -> tensor<1x48x128x128xf16>

    return %OUT_CONCAT : tensor<1x48x128x128xf16>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x128x128xf16> -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK:   [[RHS_TILE:%.*]] = IE.Tile(%arg1) {repeats_values = [1, 16, 128, 1]} : tensor<1x1x1x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[LHS_PERMUTEQUANTIZE]], [[RHS_TILE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    // CHECK:   [[MEM_PERMUTE_1:%.*]] = IE.MemPermute([[ADD]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16>
    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat([[MEM_PERMUTE_1]], [[MEM_PERMUTE_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x128x128xf16>, tensor<1x16x128x128xf16> -> tensor<1x32x128x128xf16>
    // CHECK:   [[ADD_2:%.*]] = IE.Add([[ADD]], [[ADD]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK:   [[MEM_PERMUTE_2:%.*]] = IE.MemPermute([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16>
    // CHECK:   [[CONCAT_OUTPUT:%.*]] = IE.Concat([[CONCAT_1]], [[MEM_PERMUTE_2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x128x128xf16>, tensor<1x16x128x128xf16> -> tensor<1x48x128x128xf16>

    // CHECK:   return [[CONCAT_OUTPUT]] : tensor<1x48x128x128xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertTwoPermuteLikeAndNoShapeCastInput
func.func @ConvertTwoPermuteLikeAndNoShapeCastInput(%arg0 : tensor<1x64x64x768xf16, {order = #NHWC}>, %arg1 : tensor<1x64x64x768xf16>) -> tensor<1x768x64x64xf16, {order = #NHWC}> {
   %0 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x64x64x768xf16> -> tensor<1x64x64x768xf16, {order = #NHWC}>
   %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x768xf16, {order = #NHWC}>, tensor<1x64x64x768xf16, {order = #NHWC}> -> tensor<1x64x64x768xf16, {order = #NHWC}>
   %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x64x64x768xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16, {order = #NHWC}>
   return %2 : tensor<1x768x64x64xf16, {order = #NHWC}>

    // CHECK:   [[MEM_PERMUTE_0:%.*]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x64x64x768xf16> -> tensor<1x64x64x768xf16, {order = #NHWC}>
    // CHECK:   [[MEM_PERMUTE_1:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x64x64x768xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16, {order = #NHWC}>
    // CHECK:   [[MEM_PERMUTE_2:%.*]] = IE.MemPermute([[MEM_PERMUTE_0]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x64x64x768xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[MEM_PERMUTE_1]], [[MEM_PERMUTE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768x64x64xf16, {order = #NHWC}>, tensor<1x768x64x64xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16, {order = #NHWC}>

    // CHECK:   return [[ADD]] : tensor<1x768x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertOneInputIsConst
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x16x256x128xf16>)
func.func @ConvertOneInputIsConst(%arg0: tensor<1x16x256x128xf16>) -> tensor<1x16x256x128xf16> {
    %CST = const.Declare tensor<1x16x256x128xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x256x128xf16>, [#const.Reorder<#NHWC>]
    %PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NHWC}>
    %ADD = IE.Add(%PERMUTEQUANTIZE , %CST) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>
    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16>

    return %OUT_MEM_PERMUTE : tensor<1x16x256x128xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<1x128x16x256xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x256x128xf16>, [#const.MemPermute<#NHWC, #NCHW>]
    // CHECK: [[PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize([[INPUT0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK: [[PERMUTE:%.*]] = IE.MemPermute([[PERMUTEQUANTIZE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x128x16x256xf16, {order = #NHWC}>
    // CHECK: [[ADD:%.*]] = IE.Add([[PERMUTE]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x16x256xf16, {order = #NHWC}>, tensor<1x128x16x256xf16, {order = #NHWC}> -> tensor<1x128x16x256xf16, {order = #NHWC}>
    // CHECK: [[PERMUTECAST:%.*]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x256xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16>
    // CHECK: return [[PERMUTECAST]] : tensor<1x16x256x128xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotConvertOneInputIsConst
func.func @NotConvertOneInputIsConst(%arg0 : tensor<1x2x512x512xf16>, %arg1 : tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16> {
    %CST = const.Declare tensor<1x16x256x128xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x256x128xf16>, [#const.Reorder<#NHWC>]
    %PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%PERMUTEQUANTIZE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>
    %ADD = IE.Add(%SHAPE_CAST, %CST) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>
    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs(%ADD : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>

    return %OUT_MEM_PERMUTE : tensor<1x2x512x512xf16>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1x16x256x128xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x256x128xf16>, [#const.Reorder<#NHWC>]
    // CHECK:    [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:    [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs([[PERMUTE_QUANTIZE]] : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK:    [[ADD:%.+]] = IE.Add([[SHAPE_CAST]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK:    [[OUT_SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs([[ADD]] : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:    [[OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[OUT_SHAPE_CAST]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    // CHECK:    return [[OUT_MEM_PERMUTE]] : tensor<1x2x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: func.func @NotPropagateWithIllegalShapeCastNumb
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x2x1x1024xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x2x1x1024xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x2x1x1024xf16, {order = #NHWC}>)
func.func @NotPropagateWithIllegalShapeCastNumb(%arg0 : tensor<1x2x1x1024xf16, {order = #NHWC}>,
                                                %arg1 : tensor<1x2x1x1024xf16, {order = #NHWC}>,
                                                %arg2 : tensor<1x2x1x1024xf16, {order = #NHWC}>) -> tensor<1x2x1x1024xf16> {
    %0 = IE.ShapeCast {shape = [1, 16, 16, 8]}
            inputs(%arg0 : tensor<1x2x1x1024xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>
    %1 = IE.ShapeCast {shape = [1, 16, 16, 8]}
            inputs(%arg1 : tensor<1x2x1x1024xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x16x16x8xf16, {order = #NHWC}>, tensor<1x16x16x8xf16, {order = #NHWC}> -> tensor<1x16x16x8xf16, {order = #NHWC}>
    %3 = IE.ShapeCast {shape = [1, 16, 16, 8]}
            inputs(%arg2 : tensor<1x2x1x1024xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>
    %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x16x16x8xf16, {order = #NHWC}>, tensor<1x16x16x8xf16, {order = #NHWC}> -> tensor<1x16x16x8xf16, {order = #NHWC}>
    %5 = IE.ShapeCast {shape = [1, 2, 1, 1024]}
            inputs(%4 : tensor<1x16x16x8xf16, {order = #NHWC}>) -> tensor<1x2x1x1024xf16, {order = #NHWC}>
    %6 = IE.MemPermute(%5) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x1x1024xf16, {order = #NHWC}> -> tensor<1x2x1x1024xf16>

    return %6 : tensor<1x2x1x1024xf16>

    // CHECK:    [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 8]} inputs([[INPUT0]] : tensor<1x2x1x1024xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>
    // CHECK:    [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 8]} inputs([[INPUT1]] : tensor<1x2x1x1024xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>
    // CHECK:    [[ADD_0:%.+]] = IE.Add([[SHAPE_CAST_0]], [[SHAPE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x8xf16, {order = #NHWC}>, tensor<1x16x16x8xf16, {order = #NHWC}> -> tensor<1x16x16x8xf16, {order = #NHWC}>
    // CHECK:    [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 8]} inputs([[INPUT2]] : tensor<1x2x1x1024xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>
    // CHECK:    [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[SHAPE_CAST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x8xf16, {order = #NHWC}>, tensor<1x16x16x8xf16, {order = #NHWC}> -> tensor<1x16x16x8xf16, {order = #NHWC}>
    // CHECK:    [[SHAPE_CAST_3:%.+]] = IE.ShapeCast {shape = [1, 2, 1, 1024]} inputs([[ADD_1]] : tensor<1x16x16x8xf16, {order = #NHWC}>) -> tensor<1x2x1x1024xf16, {order = #NHWC}>
    // CHECK:    [[MEMPERMUTE:%.+]] = IE.MemPermute([[SHAPE_CAST_3]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x1x1024xf16, {order = #NHWC}> -> tensor<1x2x1x1024xf16>

    // CHECK:    return [[MEMPERMUTE]] : tensor<1x2x1x1024xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertPermuteAddWithODUPermute
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x16x16x16xf16>, [[INPUT1:%.+]]: tensor<1x16x16x16xf16>)
func.func @ConvertPermuteAddWithODUPermute(%arg0 : tensor<1x16x16x16xf16>, %arg1 : tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %ADD = IE.Add(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    return %ADD: tensor<1x16x16x16xf16>

    // CHECK: [[LHS_MEM_PERMUTE0:%.*]] = IE.MemPermute([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK: [[RHS_MEM_PERMUTE0:%.*]] = IE.MemPermute([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK: [[LHS_MEM_PERMUTE1:%.*]] = IE.MemPermute([[LHS_MEM_PERMUTE0]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK: [[RHS_MEM_PERMUTE1:%.*]] = IE.MemPermute([[RHS_MEM_PERMUTE0]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK: [[ADD:%.*]] = IE.Add([[LHS_MEM_PERMUTE1]], [[RHS_MEM_PERMUTE1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK: [[PERMUTE_CAST:%.*]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    // CHECK: return [[PERMUTE_CAST]] : tensor<1x16x16x16xf16>

}
