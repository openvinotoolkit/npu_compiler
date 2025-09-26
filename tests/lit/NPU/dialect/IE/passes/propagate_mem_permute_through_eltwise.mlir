//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-mem-permute-through-eltwise %s | FileCheck %s
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
        post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
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

!qElemType = !quant.uniform<u8:f16, 4.9280512566659965E-4>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>

// CHECK-LABEL: @PropagatePermuteAddPermuteReserveShapeCast
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x129x16x48xf16>
func.func @PropagatePermuteAddPermuteReserveShapeCast(%arg0 : tensor<1x129x16x48xf16>) -> tensor<129x1x16x48x!qElemType> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x129x16x48xf16> -> tensor<1x129x16x48xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x129x16x48xf16, {order = #NHWC}>, tensor<1x129x16x48xf16, {order = #NHWC}> -> tensor<1x129x16x48x!qElemType, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #map} : tensor<1x129x16x48x!qElemType, {order = #NHWC}> -> tensor<129x1x16x48x!qElemType>

    return %2 : tensor<129x1x16x48x!qElemType>

    // CHECK:   [[PERMUTE_0:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x129x16x48xf16> -> tensor<1x129x16x48xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE_1:%.+]] = IE.MemPermute([[PERMUTE_0]]) {dst_order = #NHWC, mem_perm = #map} : tensor<1x129x16x48xf16, {order = #NHWC}> -> tensor<129x48x1x16xf16, {order = #NHWC}>
    // CHECK:   [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 6192, 1, 16]} inputs([[PERMUTE_1]] : tensor<129x48x1x16xf16, {order = #NHWC}>) -> tensor<1x6192x1x16xf16, {order = #NHWC}>
    // CHECK:   [[ADD:%.+]] = IE.Add([[SHAPE_CAST_0]], [[SHAPE_CAST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x6192x1x16xf16, {order = #NHWC}>, tensor<1x6192x1x16xf16, {order = #NHWC}> -> tensor<1x6192x1x16x!qElemType, {order = #NHWC}>
    // CHECK:   [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [129, 48, 1, 16]} inputs([[ADD]] : tensor<1x6192x1x16x!qElemType, {order = #NHWC}>) -> tensor<129x48x1x16x!qElemType, {order = #NHWC}>
    // CHECK:   [[PERMUTE_2:%.+]] = IE.PermuteCast([[SHAPE_CAST_1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<129x48x1x16x!qElemType, {order = #NHWC}> -> tensor<129x1x16x48x!qElemType>

    // CHECK:   return [[PERMUTE_2]] : tensor<129x1x16x48x!qElemType>
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

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertPermuteAddWithODUPermuteWithInputMultiUser
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x16x16x16xf16>, [[INPUT1:%.+]]: tensor<1x16x16x16xf16>, [[INPUT2:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @NotConvertPermuteAddWithODUPermuteWithInputMultiUser(%arg0 : tensor<1x16x16x16xf16>, %arg1 : tensor<1x16x16x16xf16>, %arg2 : tensor<1x16x16x16xf16, {order = #NHWC}>)
        ->(tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16>->tensor<1x16x16x16xf16, {order = #NHWC}>
    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16>->tensor<1x16x16x16xf16, {order = #NHWC}>
    %ADD0 = IE.Add(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
            tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>->tensor<1x16x16x16xf16>
    %ADD1 = IE.Add(%LHS_MEM_PERMUTE, %arg2){auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
            tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>->tensor<1x16x16x16xf16, {order = #NHWC}>
    return %ADD0, %ADD1 : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK: [[LHS_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK: [[RHS_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK: [[ADD_0:%.+]] = IE.Add([[LHS_MEM_PERMUTE]], [[RHS_MEM_PERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:   tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    // CHECK: [[ADD_1:%.+]] = IE.Add([[LHS_MEM_PERMUTE]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:   tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK: return [[ADD_0]], [[ADD_1]] : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForMultiplyWithShapecast
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x8x4x96xf16>
// CHECK-SAME:  [[INPUT_1:%.+]]: tensor<1x4x8x96xf16>
func.func @PropagateForMultiplyWithShapecast(%arg0 : tensor<1x8x4x96xf16>, %arg1 : tensor<1x4x8x96xf16>) -> tensor<1x8x4x96xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NHWC
    } : tensor<1x8x4x96xf16> -> tensor<1x8x4x96xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x4x8x96xf16> -> tensor<1x8x4x96xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 24, 8]
    } inputs(%LHS_MEM_PERMUTE : tensor<1x8x4x96xf16, {order = #NHWC}>) -> tensor<1x16x24x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 24, 8]
    } inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x96xf16, {order = #NHWC}>) -> tensor<1x16x24x8xf16, {order = #NHWC}>

    %MULTIPLY = IE.Multiply(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x24x8xf16, {order = #NHWC}>,
        tensor<1x16x24x8xf16, {order = #NHWC}>
            -> tensor<1x16x24x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 8, 4, 96]
    } inputs(%MULTIPLY : tensor<1x16x24x8xf16, {order = #NHWC}>) -> tensor<1x8x4x96xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x8x4x96xf16, {order = #NHWC}> -> tensor<1x8x4x96xf16>

    return %OUT_MEM_PERMUTE : tensor<1x8x4x96xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NCWH}
    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[LHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  }

    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[RHS_IN_MEM_PERMUTE]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  }

    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[LHS_OUT_MEM_PERMUTE]], [[RHS_OUT_MEM_PERMUTE]])
    // CHECK:   [[PERMUTECAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NCHW
    // CHECK-SAME:  }

    // CHECK:   return [[PERMUTECAST]] : tensor<1x8x4x96xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForMultiply
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x256xf16>, [[INPUT_1:%.+]]: tensor<1x4x512x256xf16>
func.func @PropagateForMultiply(%arg0 : tensor<1x4x512x256xf16>, %arg1 : tensor<1x4x512x256xf16>) -> tensor<1x4x512x256xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x256xf16> -> tensor<1x4x512x256xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x256xf16> -> tensor<1x4x512x256xf16, {order = #NHWC}>

    %MULTIPLY = IE.Multiply(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x256xf16, {order = #NHWC}>, tensor<1x4x512x256xf16, {order = #NHWC}> -> tensor<1x4x512x256xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%MULTIPLY) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x256xf16, {order = #NHWC}> -> tensor<1x4x512x256xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x512x256xf16>

    // CHECK:   [[LHS_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x256xf16> -> tensor<1x4x512x256xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x256xf16> -> tensor<1x4x512x256xf16, {order = #NHWC}>

    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x4x512x256xf16, {order = #NHWC}> -> tensor<1x256x4x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x4x512x256xf16, {order = #NHWC}> -> tensor<1x256x4x512xf16, {order = #NHWC}>

    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[LHS_OUT_MEM_PERMUTE]], [[RHS_OUT_MEM_PERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x256x4x512xf16, {order = #NHWC}>, tensor<1x256x4x512xf16, {order = #NHWC}> -> tensor<1x256x4x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x256x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x256xf16>
    // CHECK:   return [[OUT_PERMUTE_CAST]] : tensor<1x4x512x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @PropagateForMultiplyWithDifferentPermutation
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x256xf16>, [[INPUT_1:%.+]]: tensor<1x512x4x256xf16>
func.func @PropagateForMultiplyWithDifferentPermutation(%arg0 : tensor<1x4x512x256xf16>, %arg1 : tensor<1x512x4x256xf16>) -> tensor<1x4x512x256xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x256xf16> -> tensor<1x4x512x256xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x512x4x256xf16> -> tensor<1x4x512x256xf16, {order = #NHWC}>

    %MULTIPLY = IE.Multiply(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x256xf16, {order = #NHWC}>, tensor<1x4x512x256xf16, {order = #NHWC}> -> tensor<1x4x512x256xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%MULTIPLY) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x256xf16, {order = #NHWC}> -> tensor<1x4x512x256xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x512x256xf16>

    // CHECK:   [[LHS_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x256xf16> -> tensor<1x4x512x256xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x512x4x256xf16> -> tensor<1x4x512x256xf16, {order = #NHWC}>

    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x4x512x256xf16, {order = #NHWC}> -> tensor<1x256x4x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x4x512x256xf16, {order = #NHWC}> -> tensor<1x256x4x512xf16, {order = #NHWC}>

    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[LHS_OUT_MEM_PERMUTE]], [[RHS_OUT_MEM_PERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x256x4x512xf16, {order = #NHWC}>, tensor<1x256x4x512xf16, {order = #NHWC}> -> tensor<1x256x4x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x256x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x256xf16>
    // CHECK:   return [[OUT_PERMUTE_CAST]] : tensor<1x4x512x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForAddWithSoftmax
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x2x512x512xf16>, [[INPUT_1:%.+]]: tensor<1x2x512x512xf16>
func.func @PropagateForAddWithSoftmax(%arg0 : tensor<1x2x512x512xf16>, %arg1 : tensor<1x2x512x512xf16>) -> tensor<1x2x512x512xf16> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>
    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %MULTIPLY = IE.Multiply(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs(%MULTIPLY : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %OUT_SOFTMAX = IE.SoftMax(%OUT_SHAPE_CAST) {axisInd = 3} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SOFTMAX) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>

    return %OUT_MEM_PERMUTE : tensor<1x2x512x512xf16>

    // CHECK:   [[LHS_PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[INPUT_0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[INPUT_1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[LHS_PERMUTEQUANTIZE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[RHS_PERMUTEQUANTIZE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>

    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[LHS_OUT_MEM_PERMUTE]], [[RHS_OUT_MEM_PERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x2x512xf16, {order = #NHWC}>, tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    // CHECK:   [[OUT_SOFTMAX:%.+]] = IE.SoftMax([[OUT_PERMUTE_CAST]]) {axisInd = 3 : i64} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16>

    // CHECK:   return [[OUT_SOFTMAX]] : tensor<1x2x512x512xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForBroadCastMultiply
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_1:%.+]]: tensor<1x1x512x512xf16>
func.func @PropagateForBroadCastMultiply(%arg0 : tensor<1x4x512x512xf16>, %arg1 : tensor<1x1x512x512xf16>) -> tensor<1x4x512x512xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>

    %MULTIPLY = IE.Multiply(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%MULTIPLY) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x512x512xf16>

    // CHECK:   [[LHS_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>

    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x512x1x512xf16, {order = #NHWC}>

    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[LHS_OUT_MEM_PERMUTE]], [[RHS_OUT_MEM_PERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_PERMUTE_CAST:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[OUT_PERMUTE_CAST]] : tensor<1x4x512x512xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForMultiplyNCHW
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x256xf16>, [[INPUT_1:%.+]]: tensor<1x4x512x256xf16>
func.func @PropagateForMultiplyNCHW(%arg0 : tensor<1x4x512x256xf16>, %arg1 : tensor<1x4x512x256xf16>) -> tensor<1x4x512x256xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x4x512x256xf16> -> tensor<1x512x256x4xf16>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x4x512x256xf16> -> tensor<1x512x256x4xf16>

    %MULTIPLY = IE.Multiply(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x256x4xf16>, tensor<1x512x256x4xf16> -> tensor<1x512x256x4xf16>

    %OUT_MEM_PERMUTE = IE.MemPermute(%MULTIPLY) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x512x256x4xf16> -> tensor<1x4x512x256xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x512x256xf16>

    // CHECK:   [[LHS_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x4x512x256xf16> -> tensor<1x512x256x4xf16>
    // CHECK:   [[RHS_MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x4x512x256xf16> -> tensor<1x512x256x4xf16>

    // CHECK:   [[LHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[LHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x512x256x4xf16> -> tensor<1x4x512x256xf16>
    // CHECK:   [[RHS_OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[RHS_MEM_PERMUTE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x512x256x4xf16> -> tensor<1x4x512x256xf16>
    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[LHS_OUT_MEM_PERMUTE]], [[RHS_OUT_MEM_PERMUTE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x256xf16>, tensor<1x4x512x256xf16> -> tensor<1x4x512x256xf16>

    // CHECK:   return [[MULTIPLY]] : tensor<1x4x512x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForBroadCastMultiplyAndAdd
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_1:%.+]]: tensor<1x1x512x512xf16>, [[INPUT_2:%.+]]: tensor<1x4x512x512xf16>
func.func @PropagateForBroadCastMultiplyAndAdd(%arg0 : tensor<1x4x512x512xf16>, %arg1 : tensor<1x1x512x512xf16>, %arg2 : tensor<1x4x512x512xf16>) -> tensor<1x4x512x512xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>
    %MULTIPLY = IE.Multiply(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %ADD_RHS_PERMUTE = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %ADD = IE.Add(%MULTIPLY, %ADD_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    return %OUT_MEM_PERMUTE : tensor<1x4x512x512xf16>

    // CHECK:   [[LHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>

    // CHECK:   [[MUL_LHS_PERMUTE:%.+]] = IE.MemPermute([[LHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL_RHS_PERMUTE:%.+]] = IE.MemPermute([[RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x512x1x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[MUL_LHS_PERMUTE]], [[MUL_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[IN_ADD_RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_2]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_RHS_PERMUTE:%.+]] = IE.MemPermute([[IN_ADD_RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.+]] = IE.Add([[MULTIPLY]], [[ADD_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[OUT_CAST:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[OUT_CAST]] : tensor<1x4x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.081444568260043274>
!qElemType1 = !quant.uniform<u8:f16, 0.0019852942111445409>
!qElemType2 = !quant.uniform<u8:f16, 0.0063430973127776499:134>
!qElemType3 = !quant.uniform<u8:f16, 0.012491718928019205:128>
!qElemType4 = !quant.uniform<u8:f16, 0.0039705884222890819>

// CHECK-LABEL: @PropagateForBroadCastMultiplyAndAddQuantized
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x512x!qElemType>, [[INPUT_1:%.+]]: tensor<1x1x512x512x!qElemType1>, [[INPUT_2:%.+]]: tensor<1x4x512x512x!qElemType2>
func.func @PropagateForBroadCastMultiplyAndAddQuantized(%arg0: tensor<1x4x512x512x!qElemType>, %arg1: tensor<1x1x512x512x!qElemType1>, %arg2: tensor<1x4x512x512x!qElemType2>) -> tensor<1x4x512x512x!quant.uniform<u8:f16, 0.012491718928019205:128>> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512x!qElemType> -> tensor<1x4x512x512x!qElemType, {order = #NHWC}>
    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x512x512x!qElemType1> -> tensor<1x1x512x512x!qElemType1, {order = #NHWC}>
    %MULTIPLY = IE.Multiply(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x512x512x!qElemType, {order = #NHWC}>, tensor<1x1x512x512x!qElemType1, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType4, {order = #NHWC}>

    %ADD_RHS_PERMUTE = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512x!qElemType2> -> tensor<1x4x512x512x!qElemType2, {order = #NHWC}>
    %ADD = IE.Add(%MULTIPLY, %ADD_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512x!qElemType4, {order = #NHWC}>, tensor<1x4x512x512x!qElemType2, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType3, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512x!qElemType3, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType3>
    return %OUT_MEM_PERMUTE : tensor<1x4x512x512x!quant.uniform<u8:f16, 0.012491718928019205:128>>

    // CHECK:   [[LHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512x!qElemType> -> tensor<1x4x512x512x!qElemType, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x1x512x512x!qElemType1> -> tensor<1x1x512x512x!qElemType1, {order = #NHWC}>

    // CHECK:   [[MUL_LHS_PERMUTE:%.+]] = IE.MemPermute([[LHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512x!qElemType, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType, {order = #NHWC}>
    // CHECK:   [[MUL_RHS_PERMUTE:%.+]] = IE.MemPermute([[RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x1x512x512x!qElemType1, {order = #NHWC}> -> tensor<1x512x1x512x!qElemType1, {order = #NHWC}>
    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[MUL_LHS_PERMUTE]], [[MUL_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:       tensor<1x512x4x512x!qElemType, {order = #NHWC}>, tensor<1x512x1x512x!qElemType1, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType4, {order = #NHWC}>

    // CHECK:   [[IN_ADD_RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_2]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512x!qElemType2> -> tensor<1x4x512x512x!qElemType2, {order = #NHWC}>
    // CHECK:   [[ADD_RHS_PERMUTE:%.+]] = IE.MemPermute([[IN_ADD_RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512x!qElemType2, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType2, {order = #NHWC}>

    // CHECK:   [[ADD:%.+]] = IE.Add([[MULTIPLY]], [[ADD_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512x!qElemType4, {order = #NHWC}>, tensor<1x512x4x512x!qElemType2, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType3, {order = #NHWC}>

    // CHECK:   [[OUT_CAST:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK:       tensor<1x512x4x512x!qElemType3, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType3>
    // CHECK:   return [[OUT_CAST]] : tensor<1x4x512x512x!qElemType3>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForBroadCastMultiplyAndMultiply
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_1:%.+]]: tensor<1x1x512x512xf16>, [[INPUT_2:%.+]]: tensor<1x4x512x512xf16>
func.func @PropagateForBroadCastMultiplyAndMultiply(%arg0 : tensor<1x4x512x512xf16>, %arg1 : tensor<1x1x512x512xf16>, %arg2 : tensor<1x4x512x512xf16>) -> tensor<1x4x512x512xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>
    %MULTIPLY = IE.Multiply(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %MUL1_RHS_PERMUTE = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %MUL1 = IE.Multiply(%MULTIPLY, %MUL1_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%MUL1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    return %OUT_MEM_PERMUTE : tensor<1x4x512x512xf16>

    // CHECK:   [[LHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>

    // CHECK:   [[MUL_LHS_PERMUTE:%.+]] = IE.MemPermute([[LHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL_RHS_PERMUTE:%.+]] = IE.MemPermute([[RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x512x1x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[MUL_LHS_PERMUTE]], [[MUL_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[IN_MUL1_RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_2]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL1_RHS_PERMUTE:%.+]] = IE.MemPermute([[IN_MUL1_RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[MUL1:%.+]] = IE.Multiply([[MULTIPLY]], [[MUL1_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[OUT_CAST:%.+]] = IE.PermuteCast([[MUL1]])
    // CHECK:       {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[OUT_CAST]] : tensor<1x4x512x512xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.081444568260043274>
!qElemType1 = !quant.uniform<u8:f16, 0.0019852942111445409>
!qElemType2 = !quant.uniform<u8:f16, 0.0063430973127776499:134>
!qElemType3 = !quant.uniform<u8:f16, 0.012491718928019205:128>
!qElemType4 = !quant.uniform<u8:f16, 0.0039705884222890819>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForAddAndBroadCastMultiplyQuantized
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x512x!qElemType>, [[INPUT_1:%.+]]: tensor<1x4x512x512x!qElemType1>, [[INPUT_2:%.+]]: tensor<1x1x512x512x!qElemType2>
func.func @PropagateForAddAndBroadCastMultiplyQuantized(%arg0 : tensor<1x4x512x512x!qElemType>, %arg1 : tensor<1x4x512x512x!qElemType1>, %arg2 : tensor<1x1x512x512x!qElemType2>) -> tensor<1x4x512x512x!qElemType3> {
    %ADD_LHS_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512x!qElemType> -> tensor<1x4x512x512x!qElemType, {order = #NHWC}>
    %ADD_RHS_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512x!qElemType1> -> tensor<1x4x512x512x!qElemType1, {order = #NHWC}>
    %ADD = IE.Add(%ADD_LHS_PERMUTE, %ADD_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512x!qElemType, {order = #NHWC}>, tensor<1x4x512x512x!qElemType1, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType4, {order = #NHWC}>

    %MUL_RHS_PERMUTE = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x512x512x!qElemType2> -> tensor<1x1x512x512x!qElemType2, {order = #NHWC}>
    %MULTIPLY = IE.Multiply(%ADD, %MUL_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x512x512x!qElemType4, {order = #NHWC}>, tensor<1x1x512x512x!qElemType2, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType3, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%MULTIPLY) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512x!qElemType3, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType3>
    return %OUT_MEM_PERMUTE : tensor<1x4x512x512x!qElemType3>

    // CHECK:   [[LHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512x!qElemType> -> tensor<1x4x512x512x!qElemType, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512x!qElemType1> -> tensor<1x4x512x512x!qElemType1, {order = #NHWC}>

    // CHECK:   [[ADD_LHS_PERMUTE:%.+]] = IE.MemPermute([[LHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512x!qElemType, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType, {order = #NHWC}>
    // CHECK:   [[ADD_RHS_PERMUTE:%.+]] = IE.MemPermute([[RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512x!qElemType1, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType1, {order = #NHWC}>
    // CHECK:   [[ADD:%.+]] = IE.Add([[ADD_LHS_PERMUTE]], [[ADD_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512x!qElemType, {order = #NHWC}>, tensor<1x512x4x512x!qElemType1, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType4, {order = #NHWC}>
    // CHECK:   [[IN_MUL_RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_2]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x1x512x512x!qElemType2> -> tensor<1x1x512x512x!qElemType2, {order = #NHWC}>
    // CHECK:   [[MUL_IN_CAST_RHS:%.+]] = IE.MemPermute([[IN_MUL_RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x1x512x512x!qElemType2, {order = #NHWC}> -> tensor<1x512x1x512x!qElemType2, {order = #NHWC}>
    // CHECK:   [[MUL:%.+]] = IE.Multiply([[ADD]], [[MUL_IN_CAST_RHS]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:       tensor<1x512x4x512x!qElemType4, {order = #NHWC}>, tensor<1x512x1x512x!qElemType2, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType3, {order = #NHWC}>
    // CHECK:   [[OUT_CAST:%.+]] = IE.PermuteCast([[MUL]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK:       tensor<1x512x4x512x!qElemType3, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType3>
    // CHECK:   return [[OUT_CAST]] : tensor<1x4x512x512x!qElemType3>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForAddAndBroadCastMultiply
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_1:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_2:%.+]]: tensor<1x1x512x512xf16>
func.func @PropagateForAddAndBroadCastMultiply(%arg0 : tensor<1x4x512x512xf16>, %arg1 : tensor<1x4x512x512xf16>, %arg2 : tensor<1x1x512x512xf16>) -> tensor<1x4x512x512xf16> {
    %ADD_LHS_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %ADD_RHS_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %ADD = IE.Add(%ADD_LHS_PERMUTE, %ADD_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %MUL_RHS_PERMUTE = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>
    %MULTIPLY = IE.Multiply(%ADD, %MUL_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%MULTIPLY) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    return %OUT_MEM_PERMUTE : tensor<1x4x512x512xf16>

    // CHECK:   [[LHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    // CHECK:   [[ADD_LHS_PERMUTE:%.+]] = IE.MemPermute([[LHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_RHS_PERMUTE:%.+]] = IE.MemPermute([[RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD:%.+]] = IE.Add([[ADD_LHS_PERMUTE]], [[ADD_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[IN_MUL_RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_2]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL_IN_CAST_RHS:%.+]] = IE.MemPermute([[IN_MUL_RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x512x1x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL:%.+]] = IE.Multiply([[ADD]], [[MUL_IN_CAST_RHS]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_CAST:%.+]] = IE.PermuteCast([[MUL]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[OUT_CAST]] : tensor<1x4x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForBroadCastMultiplyAndAddInputConst
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x512xf16>
func.func @PropagateForBroadCastMultiplyAndAddInputConst(%arg0 : tensor<1x4x512x512xf16>) -> tensor<1x4x512x512xf16> {
    %MUL_CST = const.Declare tensor<1x1x512x512xf16, {order = #NHWC}> = dense<2.0> : tensor<1x1x512x512xf16>, [#const.Reorder<#NHWC>]
    %ADD_CST = const.Declare tensor<1x4x512x512xf16, {order = #NHWC}> = dense<2.0> : tensor<1x4x512x512xf16>, [#const.Reorder<#NHWC>]

    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %MULTIPLY = IE.Multiply(%LHS_MEM_PERMUTE, %MUL_CST) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %ADD = IE.Add(%MULTIPLY, %ADD_CST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    return %OUT_MEM_PERMUTE : tensor<1x4x512x512xf16>

    // CHECK-DAG: [[ADD_CST:%.+]] = const.Declare tensor<1x512x4x512xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x4x512x512xf16>, [#const.MemPermute<#NHWC, #NCHW>]
    // CHECK-DAG: [[MUL_CST:%.+]] = const.Declare tensor<1x512x1x512xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x1x512x512xf16>, [#const.MemPermute<#NHWC, #NCHW>]

    // CHECK:   [[LHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL_LHS_PERMUTE:%.+]] = IE.MemPermute([[LHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL:%.+]] = IE.Multiply([[MUL_LHS_PERMUTE]], [[MUL_CST]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD:%.+]] = IE.Add([[MUL]], [[ADD_CST]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_OUT_CAST:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[ADD_OUT_CAST]] : tensor<1x4x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateForMultiplyAndAdd
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_1:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_2:%.+]]: tensor<1x4x512x512xf16>
func.func @PropagateForMultiplyAndAdd(%arg0 : tensor<1x4x512x512xf16>, %arg1 : tensor<1x4x512x512xf16>, %arg2 : tensor<1x4x512x512xf16>) -> tensor<1x4x512x512xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %MULTIPLY = IE.Multiply(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %ADD_RHS_PERMUTE = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %ADD = IE.Add(%MULTIPLY, %ADD_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    return %OUT_MEM_PERMUTE : tensor<1x4x512x512xf16>

    // CHECK:   [[LHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    // CHECK:   [[MUL_LHS_PERMUTE:%.+]] = IE.MemPermute([[LHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL_RHS_PERMUTE:%.+]] = IE.MemPermute([[RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[MUL_LHS_PERMUTE]], [[MUL_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[IN_ADD_RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_2]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_RHS_PERMUTE:%.+]]  = IE.MemPermute([[IN_ADD_RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD:%.+]] = IE.Add([[MULTIPLY]], [[ADD_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_CAST:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[OUT_CAST]] : tensor<1x4x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @PropagateForMultiplyAndAddWithDiffInOutOrder
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_1:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_2:%.+]]: tensor<1x4x512x512xf16>
func.func @PropagateForMultiplyAndAddWithDiffInOutOrder(%arg0 : tensor<1x4x512x512xf16>, %arg1 : tensor<1x4x512x512xf16>, %arg2 : tensor<1x4x512x512xf16>)
  -> tensor<1x512x4x512xf16, {order = #NHCW}> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %MULTIPLY = IE.Multiply(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %ADD_RHS_PERMUTE = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %ADD = IE.Add(%MULTIPLY, %ADD_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NHCW, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHCW}>
    return %OUT_MEM_PERMUTE : tensor<1x512x4x512xf16, {order = #NHCW}>

    // CHECK:   [[LHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL_LHS_PERMUTE:%.+]] = IE.MemPermute([[LHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL_RHS_PERMUTE:%.+]] = IE.MemPermute([[RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL:%.+]] = IE.Multiply([[MUL_LHS_PERMUTE]], [[MUL_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[IN_ADD_RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_2]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_RHS_PERMUTE:%.+]]  = IE.MemPermute([[IN_ADD_RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD:%.+]] = IE.Add([[MUL]], [[ADD_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_CAST:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NHCW, mem_perm = #NCHW}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHCW}>
    // CHECK:   return [[OUT_CAST]] : tensor<1x512x4x512xf16, {order = #NHCW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @PropagateForAddAndBroadCastMultiplyWithDiffInOutOrder
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_1:%.+]]: tensor<1x4x512x512xf16>, [[INPUT_2:%.+]]: tensor<1x1x512x512xf16>
func.func @PropagateForAddAndBroadCastMultiplyWithDiffInOutOrder(%arg0 : tensor<1x4x512x512xf16>, %arg1 : tensor<1x4x512x512xf16>, %arg2 : tensor<1x1x512x512xf16>) -> tensor<1x512x4x512xf16, {order = #NHCW}> {
    %ADD_LHS_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %ADD_RHS_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    %ADD = IE.Add(%ADD_LHS_PERMUTE, %ADD_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %MUL_RHS_PERMUTE = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>
    %MULTIPLY = IE.Multiply(%ADD, %MUL_RHS_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x512x512xf16, {order = #NHWC}>, tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%MULTIPLY) {dst_order = #NHCW, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHCW}>
    return %OUT_MEM_PERMUTE : tensor<1x512x4x512xf16, {order = #NHCW}>

    // CHECK:   [[LHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>
    // CHECK:   [[RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    // CHECK:   [[ADD_LHS_PERMUTE:%.+]] = IE.MemPermute([[LHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_RHS_PERMUTE:%.+]] = IE.MemPermute([[RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD:%.+]] = IE.Add([[ADD_LHS_PERMUTE]], [[ADD_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>

    // CHECK:   [[IN_MUL_RHS_PERMUTE:%.+]] = IE.MemPermute([[INPUT_2]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:       tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL_RHS_PERMUTE:%.+]] = IE.MemPermute([[IN_MUL_RHS_PERMUTE]]) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK:       tensor<1x1x512x512xf16, {order = #NHWC}> -> tensor<1x512x1x512xf16, {order = #NHWC}>
    // CHECK:   [[MUL:%.+]] = IE.Multiply([[ADD]], [[MUL_RHS_PERMUTE]])
    // CHECK:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[OUT_CAST:%.+]] = IE.PermuteCast([[MUL]]) {dst_order = #NHCW, mem_perm = #NCHW}
    // CHECK:       tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHCW}>
    // CHECK:   return [[OUT_CAST]] : tensor<1x512x4x512xf16, {order = #NHCW}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.081444568260043274:128>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagatePermuteThroughAvgPoolWithLayoutCast
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x1575x72xf16>
func.func @PropagatePermuteThroughAvgPoolWithLayoutCast(%arg0 : tensor<1x16x1575x72xf16>) -> tensor<1x16x1575x72x!quant.uniform<u8:f16, 0.081444568260043274:128>> {
    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
                } : tensor<1x16x1575x72xf16> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                } : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72x!quant.uniform<u8:f16, 0.081444568260043274:128>, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #NWCH
                } : tensor<1x16x1575x72x!quant.uniform<u8:f16, 0.081444568260043274:128>, {order = #NHWC}> -> tensor<1x16x1575x72x!quant.uniform<u8:f16, 0.081444568260043274:128>>

    return %2 : tensor<1x16x1575x72x!quant.uniform<u8:f16, 0.081444568260043274:128>>

    // CHECK:   [[PERMUTE_IN:%.+]] = IE.PermuteQuantize([[INPUT]]) {
    // CHECK-SAME:          dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
    // CHECK-SAME:      } : tensor<1x16x1575x72xf16> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE_OUT:%.+]] = IE.MemPermute([[PERMUTE_IN]]) {
    // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72xf16>
    // CHECK:   [[LAYOUT_CAST_IN:%.+]] = IE.LayoutCast([[PERMUTE_OUT]]) {dst_order = #NHWC} : tensor<1x16x1575x72xf16> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    // CHECK:   [[AVGPOOL:%.+]] = IE.AvgPool([[LAYOUT_CAST_IN]]) {
    // CHECK-SAME:          exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72x!qElemType, {order = #NHWC}>
    // CHECK:   [[LAYOUT_CAST_OUT:%.+]] = IE.LayoutCast([[AVGPOOL]]) {dst_order = #NCHW} : tensor<1x16x1575x72x!qElemType, {order = #NHWC}> -> tensor<1x16x1575x72x!qElemType>

    // CHECK:   return [[LAYOUT_CAST_OUT]] : tensor<1x16x1575x72x!qElemType>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagatePermuteThroughMaxPoolWithLayoutCast
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x1575x72xf16>
func.func @PropagatePermuteThroughMaxPoolWithLayoutCast(%arg0 : tensor<1x16x1575x72xf16>) -> tensor<1x16x1575x72xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC
                } : tensor<1x16x1575x72xf16> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    %1 = IE.MaxPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                } : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {dst_order = #NCHW, mem_perm = #NWCH
                } : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72xf16>

    return %2 : tensor<1x16x1575x72xf16>

    // CHECK:   [[PERMUTE_IN:%.+]] = IE.MemPermute([[INPUT]]) {
    // CHECK-SAME:          dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x1575x72xf16> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE_OUT:%.+]] = IE.MemPermute([[PERMUTE_IN]]) {
    // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72xf16>
    // CHECK:   [[LAYOUT_CAST_IN:%.+]] = IE.LayoutCast([[PERMUTE_OUT]]) {dst_order = #NHWC} : tensor<1x16x1575x72xf16> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    // CHECK:   [[MAXPOOL:%.+]] = IE.MaxPool([[LAYOUT_CAST_IN]]) {
    // CHECK-SAME:          exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUT_CAST_OUT:%.+]] = IE.LayoutCast([[MAXPOOL]]) {dst_order = #NCHW} : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72xf16>

    // CHECK:   return [[LAYOUT_CAST_OUT]] : tensor<1x16x1575x72xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagatePermuteThroughAvgPoolWithShapeCast
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x768x1152x3xf16>
func.func @PropagatePermuteThroughAvgPoolWithShapeCast(%arg0 : tensor<1x768x1152x3xf16>) -> tensor<1x3x768x1152x!quant.uniform<u8:f16, 1.0287820255055147>, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
                dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
            } : tensor<1x768x1152x3xf16> -> tensor<1x768x1152x3xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {
                exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
            } : tensor<1x768x1152x3xf16, {order = #NHWC}> -> tensor<1x768x1152x3x!quant.uniform<u8:f16, 1.0287820255055147>, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {
                dst_order = #NHWC, mem_perm = #NWCH
            } : tensor<1x768x1152x3x!quant.uniform<u8:f16, 1.0287820255055147>, {order = #NHWC}> -> tensor<1x3x768x1152x!quant.uniform<u8:f16, 1.0287820255055147>, {order = #NHWC}>

    return %2 : tensor<1x3x768x1152x!quant.uniform<u8:f16, 1.0287820255055147>, {order = #NHWC}>

    // CHECK:   [[PERMUTE_IN:%.+]] = IE.PermuteQuantize([[INPUT]]) {
    // CHECK-SAME:          dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x768x1152x3xf16> -> tensor<1x768x1152x3xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE_OUT:%.+]] = IE.MemPermute([[PERMUTE_IN]]) {
    // CHECK-SAME:          dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x768x1152x3xf16, {order = #NHWC}> -> tensor<1x3x768x1152xf16, {order = #NHWC}>
    // CHECK:   [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 768, 1152, 3]
    // CHECK-SAME:          } inputs([[PERMUTE_OUT]] : tensor<1x3x768x1152xf16, {order = #NHWC}>) -> tensor<1x768x1152x3xf16, {order = #NHWC}>
    // CHECK:   [[AVGPOOL:%.+]] = IE.AvgPool([[SHAPE_CAST_IN]]) {
    // CHECK-SAME:          exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x768x1152x3xf16, {order = #NHWC}> -> tensor<1x768x1152x3x!qElemType, {order = #NHWC}>
    // CHECK:   [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 768, 1152]
    // CHECK-SAME:          } inputs([[AVGPOOL]] : tensor<1x768x1152x3x!qElemType, {order = #NHWC}>) -> tensor<1x3x768x1152x!qElemType, {order = #NHWC}>

    // CHECK:   return [[SHAPE_CAST_OUT]] : tensor<1x3x768x1152x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @PropagatePermuteThroughMaxPoolWithShapeCastAndLayoutCast
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x768x1152x3xf16>
func.func @PropagatePermuteThroughMaxPoolWithShapeCastAndLayoutCast(%arg0 : tensor<1x768x1152x3xf16>)
                            -> tensor<1x1152x768x3x!quant.uniform<u8:f16, 1.0287820255055147>, {order = #NHCW}> {
    %0 = IE.PermuteQuantize(%arg0) {
                dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
            } : tensor<1x768x1152x3xf16> -> tensor<1x768x1152x3xf16, {order = #NHWC}>
    %1 = IE.MaxPool(%0) {
                exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
            } : tensor<1x768x1152x3xf16, {order = #NHWC}> -> tensor<1x768x1152x3x!quant.uniform<u8:f16, 1.0287820255055147>, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {
                dst_order = #NHCW, mem_perm = #NWCH
            } : tensor<1x768x1152x3x!quant.uniform<u8:f16, 1.0287820255055147>, {order = #NHWC}> -> tensor<1x1152x768x3x!quant.uniform<u8:f16, 1.0287820255055147>, {order = #NHCW}>

    return %2 : tensor<1x1152x768x3x!quant.uniform<u8:f16, 1.0287820255055147>, {order = #NHCW}>

    // CHECK:   [[PERMUTE_IN:%.+]] = IE.PermuteQuantize([[INPUT]]) {
    // CHECK-SAME:          dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x768x1152x3xf16> -> tensor<1x768x1152x3xf16, {order = #NHWC}>
    // CHECK:   [[MEM_PERMUTE:%.+]] = IE.MemPermute([[PERMUTE_IN]]) {
    // CHECK-SAME:          dst_order = #NHCW, mem_perm = #NWCH} : tensor<1x768x1152x3xf16, {order = #NHWC}> -> tensor<1x1152x768x3xf16, {order = #NHCW}>
    // CHECK:   [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 768, 1152, 3]
    // CHECK-SAME:          } inputs([[MEM_PERMUTE]] : tensor<1x1152x768x3xf16, {order = #NHCW}>) -> tensor<1x768x1152x3xf16, {order = #NHCW}>
    // CHECK:   [[LAYOUT_CAST_IN:%.+]] = IE.LayoutCast([[SHAPE_CAST_IN]]) {
    // CHECK-SAME:          dst_order = #NHWC} : tensor<1x768x1152x3xf16, {order = #NHCW}> -> tensor<1x768x1152x3xf16, {order = #NHWC}>
    // CHECK:   [[MAXPOOL:%.+]] = IE.MaxPool([[LAYOUT_CAST_IN]]) {
    // CHECK-SAME:          exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x768x1152x3xf16, {order = #NHWC}> -> tensor<1x768x1152x3x!qElemType, {order = #NHWC}>
    // CHECK:   [[LAYOUT_CAST_OUT:%.+]] = IE.LayoutCast([[MAXPOOL]]) {
    // CHECK-SAME:          dst_order = #NHCW} : tensor<1x768x1152x3x!qElemType, {order = #NHWC}> -> tensor<1x768x1152x3x!qElemType, {order = #NHCW}>
    // CHECK:   [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 1152, 768, 3]
    // CHECK-SAME:          } inputs([[LAYOUT_CAST_OUT]] : tensor<1x768x1152x3x!qElemType, {order = #NHCW}>) -> tensor<1x1152x768x3x!qElemType, {order = #NHCW}>

    // CHECK:   return [[SHAPE_CAST_OUT]] : tensor<1x1152x768x3x!qElemType, {order = #NHCW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @DoNotPropagateMemPermuteWithMultipleUsers
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xf16>
func.func @DoNotPropagateMemPermuteWithMultipleUsers(%arg0: tensor<1x4x1600x2560xf16>) -> tensor<1x4x1600x2560xf16> {
  %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC}
                    : tensor<1x4x1600x2560xf16>  -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
  %1 = IE.Gelu(%0) : tensor<1x4x1600x2560xf16, {order = #NHWC}>  -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
  %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                    : tensor<1x4x1600x2560xf16, {order = #NHWC}>, tensor<1x4x1600x2560xf16, {order = #NHWC}> -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
  %3 = IE.MemPermute(%2) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = #NWCH}
                    : tensor<1x4x1600x2560xf16, {order = #NHWC}> -> tensor<1x4x1600x2560xf16>
  return %3 : tensor<1x4x1600x2560xf16>

  // CHECK:  [[MEM_PERMUTE:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NHWC}
  // CHECK-SAME:                : tensor<1x4x1600x2560xf16> -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
  // CHECK:  [[GELU:%.+]] = IE.Gelu([[MEM_PERMUTE]]) : tensor<1x4x1600x2560xf16, {order = #NHWC}>
  // CHECK-SAME:                -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
  // CHECK:  [[ADD:%.+]] = IE.Add([[MEM_PERMUTE]], [[GELU]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:                : tensor<1x4x1600x2560xf16, {order = #NHWC}>, tensor<1x4x1600x2560xf16, {order = #NHWC}> -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
  // CHECK:  [[OUT_MEM_PERMUTE:%.+]] = IE.MemPermute([[ADD]]) {dst_order = #NCHW, mem_perm = #NWCH}
  // CHECK-SAME:                : tensor<1x4x1600x2560xf16, {order = #NHWC}> -> tensor<1x4x1600x2560xf16>
  // CHECK:  return [[OUT_MEM_PERMUTE]] : tensor<1x4x1600x2560xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotPropagatePermuteThroughMultiply
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x32x64x256xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x16x128x256xf16>
func.func @NotPropagatePermuteThroughMultiply(%arg0 : tensor<1x32x64x256xf16>, %arg1 : tensor<1x16x128x256xf16>) -> tensor<1x16x128x256xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x32x64x256xf16> -> tensor<1x64x256x32xf16>
    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 512, 64, 16]} inputs(%LHS_MEM_PERMUTE : tensor<1x64x256x32xf16>) -> tensor<1x512x64x16xf16>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x128x256xf16> -> tensor<1x128x256x16xf16>
    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 512, 64, 16]} inputs(%RHS_MEM_PERMUTE : tensor<1x128x256x16xf16>) -> tensor<1x512x64x16xf16>

    %MULTIPLY = IE.Multiply(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x64x16xf16>, tensor<1x512x64x16xf16> -> tensor<1x512x64x16xf16>
    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 128, 256, 16]} inputs(%MULTIPLY : tensor<1x512x64x16xf16>) -> tensor<1x128x256x16xf16>
    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x128x256x16xf16> -> tensor<1x16x128x256xf16>

    return %OUT_MEM_PERMUTE : tensor<1x16x128x256xf16>

    // CHECK:   [[LHS_IN_MEM_PERMUTE:%.+]] = IE.MemPermute
    // CHECK:   [[LHS_SHAPE_CAST:%.+]] = IE.ShapeCast

    // CHECK:   [[RHS_IN_MEM_PERMUTE:%.+]] = IE.MemPermute
    // CHECK:   [[RHS_SHAPE_CAST:%.+]] = IE.ShapeCast

    // CHECK:   [[MULTIPLY:%.+]] =  IE.Multiply([[LHS_SHAPE_CAST]], [[RHS_SHAPE_CAST]])

    // CHECK:   [[OUT_SHAPE_CAST:%.+]] = IE.ShapeCast
    // CHECK:   [[OUT_MEM_PERMUTE:%.*]] = IE.MemPermute
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagatePermuteWhenDimNIsNotOneNeedBroadcast
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4x16x32x56xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4x16x1x1xf16, {order = #NHWC}>
func.func @PropagatePermuteWhenDimNIsNotOneNeedBroadcast(%arg0 : tensor<4x16x32x56xf16, {order = #NHWC}>, %arg1 : tensor<4x16x1x1xf16, {order = #NHWC}>) -> tensor<4x16x32x56xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x16x32x56xf16, {order = #NHWC}> -> tensor<4x16x32x56xf16>
    %1 = IE.ShapeCast {shape = [1, 64, 32, 56]} inputs(%0 : tensor<4x16x32x56xf16>) -> tensor<1x64x32x56xf16>
    %2 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x16x1x1xf16, {order = #NHWC}> -> tensor<4x16x1x1xf16>
    %3 = IE.ShapeCast {shape = [1, 64, 1, 1]} inputs(%2 : tensor<4x16x1x1xf16>) -> tensor<1x64x1x1xf16>
    %4 = IE.Multiply(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x32x56xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x32x56xf16>
    %5 = IE.ShapeCast {shape = [4, 16, 32, 56]} inputs(%4 : tensor<1x64x32x56xf16>) -> tensor<4x16x32x56xf16>
    %6 = IE.MemPermute(%5) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x16x32x56xf16> -> tensor<4x16x32x56xf16, {order = #NHWC}>

    return %6 : tensor<4x16x32x56xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE_0:%.+]] = IE.MemPermute([[INPUT_0]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x16x32x56xf16, {order = #NHWC}> -> tensor<4x16x32x56xf16>
    // CHECK:   [[PERMUTE_1:%.+]] = IE.MemPermute([[INPUT_1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x16x1x1xf16, {order = #NHWC}> -> tensor<4x16x1x1xf16>
    // CHECK:   [[PERMUTE_2:%.+]] = IE.MemPermute([[PERMUTE_0]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<4x16x32x56xf16> -> tensor<4x32x56x16xf16>
    // CHECK:   [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 32, 56, 64]} inputs([[PERMUTE_2]] : tensor<4x32x56x16xf16>) -> tensor<1x32x56x64xf16>
    // CHECK:   [[PERMUTE_3:%.+]] = IE.MemPermute([[PERMUTE_1]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<4x16x1x1xf16> -> tensor<4x1x1x16xf16>
    // CHECK:   [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 64]} inputs([[PERMUTE_3]] : tensor<4x1x1x16xf16>) -> tensor<1x1x1x64xf16>
    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[SHAPE_CAST_0]], [[SHAPE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x56x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x32x56x64xf16>
    // CHECK:   [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [4, 32, 56, 16]} inputs([[MULTIPLY]] : tensor<1x32x56x64xf16>) -> tensor<4x32x56x16xf16>
    // CHECK:   [[PERMUTE_4:%.+]] = IE.PermuteCast([[SHAPE_CAST_2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<4x32x56x16xf16> -> tensor<4x16x32x56xf16, {order = #NHWC}>

    // CHECK:   return [[PERMUTE_4]] : tensor<4x16x32x56xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotPropagatePermuteWhenDimNOfOneInputIsNotOne
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4x1x32x56xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x56x32xf16, {order = #NHWC}>
func.func @NotPropagatePermuteWhenDimNOfOneInputIsNotOne(%arg0 : tensor<4x1x32x56xf16, {order = #NHWC}>, %arg1 : tensor<1x1x56x32xf16, {order = #NHWC}>) -> tensor<4x1x32x56xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x1x32x56xf16, {order = #NHWC}> -> tensor<4x1x32x56xf16>
    %1 = IE.ShapeCast {shape = [1, 4, 32, 56]} inputs(%0 : tensor<4x1x32x56xf16>) -> tensor<1x4x32x56xf16>
    %2 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x56x32xf16, {order = #NHWC}> -> tensor<1x1x56x32xf16>
    %3 = IE.ShapeCast {shape = [1, 1, 32, 56]} inputs(%2 : tensor<1x1x56x32xf16>) -> tensor<1x1x32x56xf16>
    %4 = IE.Multiply(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x32x56xf16>, tensor<1x1x32x56xf16> -> tensor<1x4x32x56xf16>
    %5 = IE.ShapeCast {shape = [4, 1, 32, 56]} inputs(%4 : tensor<1x4x32x56xf16>) -> tensor<4x1x32x56xf16>
    %6 = IE.MemPermute(%5) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x1x32x56xf16> -> tensor<4x1x32x56xf16, {order = #NHWC}>

    return %6 : tensor<4x1x32x56xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE_0:%.+]] = IE.MemPermute([[INPUT_0]])
    // CHECK:   [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 4, 32, 56]}
    // CHECK:   [[PERMUTE_1:%.+]] = IE.MemPermute([[INPUT_1]])
    // CHECK:   [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 1, 32, 56]}
    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[SHAPE_CAST_0]], [[SHAPE_CAST_1]])
    // CHECK:   [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [4, 1, 32, 56]}
    // CHECK:   [[PERMUTE_2:%.+]] = IE.MemPermute

    // CHECK:   return [[PERMUTE_2]] : tensor<4x1x32x56xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotPropagatePermuteWhenDimNOfOneInputIsNotOneSameHW
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<4x1x32x56xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x32x56xf16, {order = #NHWC}>
func.func @NotPropagatePermuteWhenDimNOfOneInputIsNotOneSameHW(%arg0 : tensor<4x1x32x56xf16, {order = #NHWC}>, %arg1 : tensor<1x1x32x56xf16, {order = #NHWC}>) -> tensor<4x1x32x56xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x1x32x56xf16, {order = #NHWC}> -> tensor<4x1x32x56xf16>
    %1 = IE.ShapeCast {shape = [1, 4, 32, 56]} inputs(%0 : tensor<4x1x32x56xf16>) -> tensor<1x4x32x56xf16>
    %2 = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x32x56xf16, {order = #NHWC}> -> tensor<1x1x32x56xf16>
    %3 = IE.Multiply(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x32x56xf16>, tensor<1x1x32x56xf16> -> tensor<1x4x32x56xf16>
    %4 = IE.ShapeCast {shape = [4, 1, 32, 56]} inputs(%3 : tensor<1x4x32x56xf16>) -> tensor<4x1x32x56xf16>
    %5 = IE.MemPermute(%4) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x1x32x56xf16> -> tensor<4x1x32x56xf16, {order = #NHWC}>

    return %5 : tensor<4x1x32x56xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE_0:%.+]] = IE.MemPermute([[INPUT_0]])
    // CHECK:   [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 4, 32, 56]}
    // CHECK:   [[PERMUTE_1:%.+]] = IE.MemPermute([[INPUT_1]])
    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[SHAPE_CAST_0]], [[PERMUTE_1]])
    // CHECK:   [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [4, 1, 32, 56]}
    // CHECK:   [[PERMUTE_2:%.+]] = IE.MemPermute

    // CHECK:   return [[PERMUTE_2]] : tensor<4x1x32x56xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PermuteQuantizeAddDynamic
func.func @PermuteQuantizeAddDynamic(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>,
                %arg1: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>)
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    %1 = IE.PermuteQuantize(%0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    %14 = IE.Add(%arg1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    %15 = IE.MemPermute(%14) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    return %15 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

}
    // Check that bounds order also modified same way as the logical order

    // CHECK: IE.MemPermute({{.*}}) {dst_order = #NHWC, mem_perm = #NWCH}
    // CHECK-SAME: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> ->
    // CHECK-SAME: tensor<1x?x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: IE.MemPermute({{.*}})
    // CHECK-SAME: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> ->
    // CHECK-SAME: tensor<1x?x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: IE.Add({{.*}}, {{.*}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME: tensor<1x?x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME: tensor<1x?x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}> ->
    // CHECK-SAME: tensor<1x?x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: IE.PermuteCast({{.*}})
    // CHECK-SAME: tensor<1x?x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}> ->
    // CHECK-SAME: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: return {{.*}} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
