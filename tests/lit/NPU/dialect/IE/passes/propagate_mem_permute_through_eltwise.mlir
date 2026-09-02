//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --propagate-mem-permute-through-eltwise %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertAddNWCH
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x4x76xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x8x76xf16>) -> tensor<1x8x4x76xf16>
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

    // CHECK:   [[SHAPECAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[ARG_0]] : tensor<1x8x4x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[SHAPECAST_0]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_2:%.+]] = IE.MemPermute([[ARG_1]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16>
    // CHECK:   [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[MEMPERMUTE_2]] : tensor<1x8x4x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_4:%.+]] = IE.LayoutCast([[SHAPECAST_3]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_5:%.+]] = IE.Add([[LAYOUTCAST_1]], [[LAYOUTCAST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_6:%.+]] = IE.LayoutCast([[ADD_5]]) {dst_order = #NCHW} : tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16>
    // CHECK:   [[SHAPECAST_7:%.+]] = IE.ShapeCast {shape = [1, 8, 4, 76]} inputs([[LAYOUTCAST_6]] : tensor<1x16x19x8xf16>) -> tensor<1x8x4x76xf16>
    // CHECK:   return [[SHAPECAST_7]] : tensor<1x8x4x76xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @ConvertAddNWHC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x4x76xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x8x76xf16>) -> tensor<1x8x76x4xf16>
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

    // CHECK:   [[MEMPERMUTE_0:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x8x4x76xf16> -> tensor<1x8x76x4xf16>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[MEMPERMUTE_0]] : tensor<1x8x76x4xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_2:%.+]] = IE.LayoutCast([[SHAPECAST_1]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_3:%.+]] = IE.MemPermute([[ARG_1]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x4x8x76xf16> -> tensor<1x8x76x4xf16>
    // CHECK:   [[SHAPECAST_4:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[MEMPERMUTE_3]] : tensor<1x8x76x4xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_5:%.+]] = IE.LayoutCast([[SHAPECAST_4]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_6:%.+]] = IE.Add([[LAYOUTCAST_2]], [[LAYOUTCAST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_7:%.+]] = IE.LayoutCast([[ADD_6]]) {dst_order = #NCHW} : tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16>
    // CHECK:   [[SHAPECAST_8:%.+]] = IE.ShapeCast {shape = [1, 8, 76, 4]} inputs([[LAYOUTCAST_7]] : tensor<1x16x19x8xf16>) -> tensor<1x8x76x4xf16>
    // CHECK:   return [[SHAPECAST_8]] : tensor<1x8x76x4xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAddNCWH
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x4x76xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x8x76xf16>) -> tensor<1x4x8x76xf16>
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

    // CHECK:   [[MEMPERMUTE_0:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x8x4x76xf16> -> tensor<1x4x8x76xf16>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[MEMPERMUTE_0]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_2:%.+]] = IE.LayoutCast([[SHAPECAST_1]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[ARG_1]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_4:%.+]] = IE.LayoutCast([[SHAPECAST_3]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_5:%.+]] = IE.Add([[LAYOUTCAST_2]], [[LAYOUTCAST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_6:%.+]] = IE.LayoutCast([[ADD_5]]) {dst_order = #NCHW} : tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16>
    // CHECK:   [[SHAPECAST_7:%.+]] = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs([[LAYOUTCAST_6]] : tensor<1x16x19x8xf16>) -> tensor<1x4x8x76xf16>
    // CHECK:   return [[SHAPECAST_7]] : tensor<1x4x8x76xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAddNCHW
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x4x76xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x8x76xf16>) -> tensor<1x4x76x8xf16>
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

    // CHECK:   [[MEMPERMUTE_0:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x8x4x76xf16> -> tensor<1x4x76x8xf16>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[MEMPERMUTE_0]] : tensor<1x4x76x8xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_2:%.+]] = IE.LayoutCast([[SHAPECAST_1]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_3:%.+]] = IE.MemPermute([[ARG_1]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x4x8x76xf16> -> tensor<1x4x76x8xf16>
    // CHECK:   [[SHAPECAST_4:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[MEMPERMUTE_3]] : tensor<1x4x76x8xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_5:%.+]] = IE.LayoutCast([[SHAPECAST_4]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_6:%.+]] = IE.Add([[LAYOUTCAST_2]], [[LAYOUTCAST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_7:%.+]] = IE.LayoutCast([[ADD_6]]) {dst_order = #NCHW} : tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16>
    // CHECK:   [[SHAPECAST_8:%.+]] = IE.ShapeCast {shape = [1, 4, 76, 8]} inputs([[LAYOUTCAST_7]] : tensor<1x16x19x8xf16>) -> tensor<1x4x76x8xf16>
    // CHECK:   return [[SHAPECAST_8]] : tensor<1x4x76x8xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ConvertAddNHCW
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x4x76xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x8x76xf16>) -> tensor<1x76x4x8xf16>
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

    // CHECK:   [[MEMPERMUTE_0:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x8x4x76xf16> -> tensor<1x76x4x8xf16>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[MEMPERMUTE_0]] : tensor<1x76x4x8xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_2:%.+]] = IE.LayoutCast([[SHAPECAST_1]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_3:%.+]] = IE.MemPermute([[ARG_1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76xf16> -> tensor<1x76x4x8xf16>
    // CHECK:   [[SHAPECAST_4:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[MEMPERMUTE_3]] : tensor<1x76x4x8xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_5:%.+]] = IE.LayoutCast([[SHAPECAST_4]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_6:%.+]] = IE.Add([[LAYOUTCAST_2]], [[LAYOUTCAST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_7:%.+]] = IE.LayoutCast([[ADD_6]]) {dst_order = #NCHW} : tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16>
    // CHECK:   [[SHAPECAST_8:%.+]] = IE.ShapeCast {shape = [1, 76, 4, 8]} inputs([[LAYOUTCAST_7]] : tensor<1x16x19x8xf16>) -> tensor<1x76x4x8xf16>
    // CHECK:   return [[SHAPECAST_8]] : tensor<1x76x4x8xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertAddWithPostOp
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x4x76xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x8x76xf16>) -> tensor<1x8x4x76xf16>
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

    // CHECK:   [[SHAPECAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[ARG_0]] : tensor<1x8x4x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[SHAPECAST_0]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_2:%.+]] = IE.MemPermute([[ARG_1]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x8x76xf16> -> tensor<1x8x4x76xf16>
    // CHECK:   [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[MEMPERMUTE_2]] : tensor<1x8x4x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_4:%.+]] = IE.LayoutCast([[SHAPECAST_3]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_5:%.+]] = IE.Add([[LAYOUTCAST_1]], [[LAYOUTCAST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_6:%.+]] = IE.LayoutCast([[ADD_5]]) {dst_order = #NCHW} : tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8xf16>
    // CHECK:   [[SHAPECAST_7:%.+]] = IE.ShapeCast {shape = [1, 8, 4, 76]} inputs([[LAYOUTCAST_6]] : tensor<1x16x19x8xf16>) -> tensor<1x8x4x76xf16>
    // CHECK:   return [[SHAPECAST_7]] : tensor<1x8x4x76xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteAddWithQuantizeCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x8x76xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x8x76xf16>) -> tensor<1x4x8x76x!qElemType>
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

    // CHECK:   [[SHAPECAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[ARG_0]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[SHAPECAST_0]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[ARG_1]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_3:%.+]] = IE.LayoutCast([[SHAPECAST_2]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_4:%.+]] = IE.Add([[LAYOUTCAST_1]], [[LAYOUTCAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType1, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_5:%.+]] = IE.LayoutCast([[ADD_4]]) {dst_order = #NCHW} : tensor<1x16x19x8x!qElemType1, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType1>
    // CHECK:   [[SHAPECAST_6:%.+]] = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs([[LAYOUTCAST_5]] : tensor<1x16x19x8x!qElemType1>) -> tensor<1x4x8x76x!qElemType1>
    // CHECK:   [[QUANTIZECAST_7:%.+]] = IE.QuantizeCast([[SHAPECAST_6]]) {dstElemType = !qElemType} : tensor<1x4x8x76x!qElemType1> -> tensor<1x4x8x76x!qElemType>
    // CHECK:   return [[QUANTIZECAST_7]] : tensor<1x4x8x76x!qElemType>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteShapeCastAdd
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x512x512xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x512x512xf16>) -> tensor<1x4x512x512xf16>
func.func @ConvertPermuteShapeCastAdd(%arg0 : tensor<1x4x512x512xf16>, %arg1 : tensor<1x4x512x512xf16>) -> tensor<1x4x512x512xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x512x512xf16> -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 512, 128]} inputs(%LHS_MEM_PERMUTE : tensor<1x4x512x512xf16, {order = #NHWC}>) -> tensor<1x16x512x128xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 512, 128]} inputs(%RHS_MEM_PERMUTE : tensor<1x4x512x512xf16, {order = #NHWC}>) -> tensor<1x16x512x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x512x128xf16, {order = #NHWC}>, tensor<1x16x512x128xf16, {order = #NHWC}> -> tensor<1x16x512x128xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 4, 512, 512]} inputs(%ADD : tensor<1x16x512x128xf16, {order = #NHWC}>) -> tensor<1x4x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x512x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x512x512xf16>

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<1x4x512x512xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x129x16x48xf16> -> tensor<1x48x129x16xf16, {order = #NHWC}>
    // CHECK:   [[ADD_1:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x129x16xf16, {order = #NHWC}>, tensor<1x48x129x16xf16, {order = #NHWC}> -> tensor<1x48x129x16x!qElemType, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [129, 48, 1, 16]} inputs([[ADD_1]] : tensor<1x48x129x16x!qElemType, {order = #NHWC}>) -> tensor<129x48x1x16x!qElemType, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[SHAPECAST_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<129x48x1x16x!qElemType, {order = #NHWC}> -> tensor<129x1x16x48x!qElemType>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<129x1x16x48x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>


// CHECK-LABEL: @ConvertPermuteShapeCastAddWithQuantizeCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x512x512xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x512x512xf16>) -> tensor<1x4x512x512x!qElemType>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType1, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512x!qElemType1, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType1>
    // CHECK:   [[QUANTIZECAST_4:%.+]] = IE.QuantizeCast([[PERMUTECAST_3]]) {dstElemType = !qElemType} : tensor<1x4x512x512x!qElemType1> -> tensor<1x4x512x512x!qElemType>
    // CHECK:   return [[QUANTIZECAST_4]] : tensor<1x4x512x512x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteAddWithQuantizeCastNoShapeCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x8x8xf16>, [[ARG_1:%[^:]+]]: tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType>
func.func @ConvertPermuteAddWithQuantizeCastNoShapeCast(%arg0 : tensor<1x8x8x8xf16>, %arg1 : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%ADD) {dstElemType = !qElemType} : tensor<1x8x8x8x!qElemType1, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x8x8x!qElemType>

    // CHECK:   [[LAYOUTCAST_0:%.+]] = IE.LayoutCast([[ARG_0]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[ARG_1]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[LAYOUTCAST_0]], [[LAYOUTCAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_3:%.+]] = IE.LayoutCast([[ADD_2]]) {dst_order = #NCHW} : tensor<1x8x8x8x!qElemType1, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1>
    // CHECK:   [[QUANTIZECAST_4:%.+]] = IE.QuantizeCast([[LAYOUTCAST_3]]) {dstElemType = !qElemType} : tensor<1x8x8x8x!qElemType1> -> tensor<1x8x8x8x!qElemType>
    // CHECK:   return [[QUANTIZECAST_4]] : tensor<1x8x8x8x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteAddNoShapeCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x8x8xf16>, [[ARG_1:%[^:]+]]: tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType>
func.func @ConvertPermuteAddNoShapeCast(%arg0 : tensor<1x8x8x8xf16>, %arg1 : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x8x8x!qElemType>

    // CHECK:   [[LAYOUTCAST_0:%.+]] = IE.LayoutCast([[ARG_0]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[ARG_1]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[LAYOUTCAST_0]], [[LAYOUTCAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_3:%.+]] = IE.LayoutCast([[ADD_2]]) {dst_order = #NCHW} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>
    // CHECK:   return [[LAYOUTCAST_3]] : tensor<1x8x8x8x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddWithQuantizeCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x8x76xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x8x76xf16>) -> tensor<1x4x8x76x!qElemType>
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

    // CHECK:   [[SHAPECAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[ARG_0]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[SHAPECAST_0]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[ARG_1]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_3:%.+]] = IE.LayoutCast([[SHAPECAST_2]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_4:%.+]] = IE.Add([[LAYOUTCAST_1]], [[LAYOUTCAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType1, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_5:%.+]] = IE.LayoutCast([[ADD_4]]) {dst_order = #NCHW} : tensor<1x16x19x8x!qElemType1, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType1>
    // CHECK:   [[SHAPECAST_6:%.+]] = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs([[LAYOUTCAST_5]] : tensor<1x16x19x8x!qElemType1>) -> tensor<1x4x8x76x!qElemType1>
    // CHECK:   [[QUANTIZECAST_7:%.+]] = IE.QuantizeCast([[SHAPECAST_6]]) {dstElemType = !qElemType} : tensor<1x4x8x76x!qElemType1> -> tensor<1x4x8x76x!qElemType>
    // CHECK:   return [[QUANTIZECAST_7]] : tensor<1x4x8x76x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NoPropagateIfPermutationsCanNotFold
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x4096x4096xf16>, [[ARG_1:%[^:]+]]: tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>
func.func @NoPropagateIfPermutationsCanNotFold(%arg0 : tensor<1x8x4096x4096xf16>, %arg1 : tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs(%ADD : tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>) -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%OUT_SHAPE_CAST) {dstElemType = !qElemType} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>

    return %OUT_MEM_PERMUTE : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>

    // CHECK:   [[PERMUTEQUANTIZE_0:%.+]] = IE.PermuteQuantize([[ARG_0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTEQUANTIZE_1:%.+]] = IE.PermuteQuantize([[ARG_1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[PERMUTEQUANTIZE_0]] : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 16, 4096, 2048]} inputs([[PERMUTEQUANTIZE_1]] : tensor<1x8x4096x4096xf16, {order = #NHWC}>) -> tensor<1x16x4096x2048xf16, {order = #NHWC}>
    // CHECK:   [[ADD_4:%.+]] = IE.Add([[SHAPECAST_2]], [[SHAPECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x4096x2048xf16, {order = #NHWC}>, tensor<1x16x4096x2048xf16, {order = #NHWC}> -> tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_5:%.+]] = IE.ShapeCast {shape = [1, 8, 4096, 4096]} inputs([[ADD_4]] : tensor<1x16x4096x2048x!qElemType1, {order = #NHWC}>) -> tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}>
    // CHECK:   [[QUANTIZECAST_6:%.+]] = IE.QuantizeCast([[SHAPECAST_5]]) {dstElemType = !qElemType} : tensor<1x8x4096x4096x!qElemType1, {order = #NHWC}> -> tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>
    // CHECK:   return [[QUANTIZECAST_6]] : tensor<1x8x4096x4096x!qElemType, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAdd
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x8x76xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x8x76xf16>) -> tensor<1x4x8x76x!qElemType>
func.func @ConvertPermuteQuantizeAdd(%arg0 : tensor<1x4x8x76xf16>, %arg1 : tensor<1x4x8x76xf16>) -> tensor<1x4x8x76x!qElemType> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x8x76xf16> -> tensor<1x4x8x76xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x4x8x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs(%RHS_PERMUTEQUANTIZE : tensor<1x4x8x76xf16, {order = #NHWC}>) -> tensor<1x16x19x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs(%ADD : tensor<1x16x19x8x!qElemType, {order = #NHWC}>) -> tensor<1x4x8x76x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x76x!qElemType, {order = #NHWC}> -> tensor<1x4x8x76x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x4x8x76x!qElemType>

    // CHECK:   [[SHAPECAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[ARG_0]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[SHAPECAST_0]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [1, 16, 19, 8]} inputs([[ARG_1]] : tensor<1x4x8x76xf16>) -> tensor<1x16x19x8xf16>
    // CHECK:   [[LAYOUTCAST_3:%.+]] = IE.LayoutCast([[SHAPECAST_2]]) {dst_order = #NHWC} : tensor<1x16x19x8xf16> -> tensor<1x16x19x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_4:%.+]] = IE.Add([[LAYOUTCAST_1]], [[LAYOUTCAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x19x8xf16, {order = #NHWC}>, tensor<1x16x19x8xf16, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_5:%.+]] = IE.LayoutCast([[ADD_4]]) {dst_order = #NCHW} : tensor<1x16x19x8x!qElemType, {order = #NHWC}> -> tensor<1x16x19x8x!qElemType>
    // CHECK:   [[SHAPECAST_6:%.+]] = IE.ShapeCast {shape = [1, 4, 8, 76]} inputs([[LAYOUTCAST_5]] : tensor<1x16x19x8x!qElemType>) -> tensor<1x4x8x76x!qElemType>
    // CHECK:   return [[SHAPECAST_6]] : tensor<1x4x8x76x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddWithQuantizeCastNoShapeCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x8x8xf16>, [[ARG_1:%[^:]+]]: tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType>
func.func @ConvertPermuteQuantizeAddWithQuantizeCastNoShapeCast(%arg0 : tensor<1x8x8x8xf16>, %arg1 : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_PERMUTEQUANTIZE, %RHS_PERMUTEQUANTIZE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1, {order = #NHWC}>

    %OUT_QUANTIZE_CAST = IE.QuantizeCast(%ADD) {dstElemType = !qElemType} : tensor<1x8x8x8x!qElemType1, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_QUANTIZE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x8x8x!qElemType>

    // CHECK:   [[LAYOUTCAST_0:%.+]] = IE.LayoutCast([[ARG_0]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[ARG_1]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[LAYOUTCAST_0]], [[LAYOUTCAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_3:%.+]] = IE.LayoutCast([[ADD_2]]) {dst_order = #NCHW} : tensor<1x8x8x8x!qElemType1, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType1>
    // CHECK:   [[QUANTIZECAST_4:%.+]] = IE.QuantizeCast([[LAYOUTCAST_3]]) {dstElemType = !qElemType} : tensor<1x8x8x8x!qElemType1> -> tensor<1x8x8x8x!qElemType>
    // CHECK:   return [[QUANTIZECAST_4]] : tensor<1x8x8x8x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddNoShapeCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x8x8xf16>, [[ARG_1:%[^:]+]]: tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType>
func.func @ConvertPermuteQuantizeAddNoShapeCast(%arg0 : tensor<1x8x8x8xf16>, %arg1 : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8x!qElemType> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %RHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_PERMUTEQUANTIZE, %RHS_PERMUTEQUANTIZE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>

    return %OUT_MEM_PERMUTE : tensor<1x8x8x8x!qElemType>

    // CHECK:   [[LAYOUTCAST_0:%.+]] = IE.LayoutCast([[ARG_0]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_1:%.+]] = IE.LayoutCast([[ARG_1]]) {dst_order = #NHWC} : tensor<1x8x8x8xf16> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[LAYOUTCAST_0]], [[LAYOUTCAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_3:%.+]] = IE.LayoutCast([[ADD_2]]) {dst_order = #NCHW} : tensor<1x8x8x8x!qElemType, {order = #NHWC}> -> tensor<1x8x8x8x!qElemType>
    // CHECK:   return [[LAYOUTCAST_3]] : tensor<1x8x8x8x!qElemType>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertPermuteQuantizeAddWithSoftmax
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x2x512x512xf16>, [[ARG_1:%[^:]+]]: tensor<1x2x512x512xf16>) -> tensor<1x2x512x512xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x512x512xf16> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x512x512xf16> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x2x512xf16, {order = #NHWC}>, tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    // CHECK:   [[SOFTMAX_4:%.+]] = IE.SoftMax([[PERMUTECAST_3]]) {axisInd = 3 : i64} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16>
    // CHECK:   return [[SOFTMAX_4]] : tensor<1x2x512x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @NotSwapSoftmaxMemPermuteIfCanNotFuse
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x2x512x512xf16>, [[ARG_1:%[^:]+]]: tensor<1x2x512x512xf16>) -> tensor<1x2x512x512xf16, {order = #NHWC}>
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

    // CHECK:   [[PERMUTEQUANTIZE_0:%.+]] = IE.PermuteQuantize([[ARG_0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTEQUANTIZE_1:%.+]] = IE.PermuteQuantize([[ARG_1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_2:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs([[PERMUTEQUANTIZE_0]] : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs([[PERMUTEQUANTIZE_1]] : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK:   [[ADD_4:%.+]] = IE.Add([[SHAPECAST_2]], [[SHAPECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_5:%.+]] = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs([[ADD_4]] : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[SOFTMAX_6:%.+]] = IE.SoftMax([[SHAPECAST_5]]) {axisInd = 3 : i64} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_7:%.+]] = IE.MemPermute([[SOFTMAX_6]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   return [[MEMPERMUTE_7]] : tensor<1x2x512x512xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertOnlyOnePermuteLikeInput
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x2x512x512xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16>
func.func @ConvertOnlyOnePermuteLikeInput(%arg0 : tensor<1x2x512x512xf16>, %arg1 : tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %RHS_TILE = IE.Tile(%arg1) {repeats_values = [1, 2, 512, 1]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%LHS_PERMUTEQUANTIZE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%RHS_TILE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs(%ADD : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>

    return %OUT_MEM_PERMUTE : tensor<1x2x512x512xf16>

    // CHECK:   [[TILE_0:%.+]] = IE.Tile([[ARG_1]]) {repeats_values = [1, 2, 512, 1]} : tensor<1x1x1x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x512x512xf16> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_2:%.+]] = IE.MemPermute([[TILE_0]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_3:%.+]] = IE.Add([[PERMUTECAST_1]], [[MEMPERMUTE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x2x512xf16, {order = #NHWC}>, tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_4:%.+]] = IE.PermuteCast([[ADD_3]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    // CHECK:   return [[PERMUTECAST_4]] : tensor<1x2x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertOnlyOnePermuteLikeAndWithoutShapeCastInput
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x128x128xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x128xf16, {order = #NHWC}>) -> tensor<1x16x128x128xf16>
func.func @ConvertOnlyOnePermuteLikeAndWithoutShapeCastInput(%arg0 : tensor<1x16x128x128xf16>, %arg1 : tensor<1x1x1x128xf16, {order = #NHWC}>) -> tensor<1x16x128x128xf16> {
    %LHS_PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x128x128xf16> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    %RHS_TILE = IE.Tile(%arg1) {repeats_values = [1, 16, 128, 1]} : tensor<1x1x1x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_PERMUTEQUANTIZE, %RHS_TILE) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16>

    return %OUT_MEM_PERMUTE : tensor<1x16x128x128xf16>

    // CHECK:   [[TILE_0:%.+]] = IE.Tile([[ARG_1]]) {repeats_values = [1, 16, 128, 1]} : tensor<1x1x1x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x128x128xf16> -> tensor<1x128x16x128xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_2:%.+]] = IE.MemPermute([[TILE_0]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x128x16x128xf16, {order = #NHWC}>
    // CHECK:   [[ADD_3:%.+]] = IE.Add([[PERMUTECAST_1]], [[MEMPERMUTE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x16x128xf16, {order = #NHWC}>, tensor<1x128x16x128xf16, {order = #NHWC}> -> tensor<1x128x16x128xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_4:%.+]] = IE.PermuteCast([[ADD_3]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16>
    // CHECK:   return [[PERMUTECAST_4]] : tensor<1x16x128x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NoPopagateIfAddWithTwoOutputs
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x128x128xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x128xf16, {order = #NHWC}>) -> tensor<1x48x128x128xf16>
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

    // CHECK:   [[PERMUTEQUANTIZE_0:%.+]] = IE.PermuteQuantize([[ARG_0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x128x128xf16> -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK:   [[TILE_1:%.+]] = IE.Tile([[ARG_1]]) {repeats_values = [1, 16, 128, 1]} : tensor<1x1x1x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTEQUANTIZE_0]], [[TILE_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_3:%.+]] = IE.MemPermute([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16>
    // CHECK:   [[CONCAT_4:%.+]] = IE.Concat([[MEMPERMUTE_3]], [[MEMPERMUTE_3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x128x128xf16>, tensor<1x16x128x128xf16> -> tensor<1x32x128x128xf16>
    // CHECK:   [[ADD_5:%.+]] = IE.Add([[ADD_2]], [[ADD_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_6:%.+]] = IE.MemPermute([[ADD_5]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x128x128xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16>
    // CHECK:   [[CONCAT_7:%.+]] = IE.Concat([[CONCAT_4]], [[MEMPERMUTE_6]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x128x128xf16>, tensor<1x16x128x128xf16> -> tensor<1x48x128x128xf16>
    // CHECK:   return [[CONCAT_7]] : tensor<1x48x128x128xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ConvertTwoPermuteLikeAndNoShapeCastInput
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x64x768xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x64x64x768xf16>) -> tensor<1x768x64x64xf16, {order = #NHWC}>
func.func @ConvertTwoPermuteLikeAndNoShapeCastInput(%arg0 : tensor<1x64x64x768xf16, {order = #NHWC}>, %arg1 : tensor<1x64x64x768xf16>) -> tensor<1x768x64x64xf16, {order = #NHWC}> {
   %0 = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x64x64x768xf16> -> tensor<1x64x64x768xf16, {order = #NHWC}>
   %1 = IE.Add(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x768xf16, {order = #NHWC}>, tensor<1x64x64x768xf16, {order = #NHWC}> -> tensor<1x64x64x768xf16, {order = #NHWC}>
   %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x64x64x768xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16, {order = #NHWC}>
   return %2 : tensor<1x768x64x64xf16, {order = #NHWC}>

    // CHECK:   [[MEMPERMUTE_0:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x64x64x768xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x64x768xf16> -> tensor<1x768x64x64xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[MEMPERMUTE_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768x64x64xf16, {order = #NHWC}>, tensor<1x768x64x64xf16, {order = #NHWC}> -> tensor<1x768x64x64xf16, {order = #NHWC}>
    // CHECK:   return [[ADD_2]] : tensor<1x768x64x64xf16, {order = #NHWC}>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x256x128xf16> -> tensor<1x128x16x256xf16, {order = #NHWC}>
    // CHECK:   [[ADD_1:%.+]] = IE.Add([[PERMUTECAST_0]], %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x16x256xf16, {order = #NHWC}>, tensor<1x128x16x256xf16, {order = #NHWC}> -> tensor<1x128x16x256xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_2:%.+]] = IE.PermuteCast([[ADD_1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x128x16x256xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16>
    // CHECK:   return [[PERMUTECAST_2]] : tensor<1x16x256x128xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotConvertOneInputIsConst
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x2x512x512xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16>
func.func @NotConvertOneInputIsConst(%arg0 : tensor<1x2x512x512xf16>, %arg1 : tensor<1x1x1x512xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16> {
    %CST = const.Declare tensor<1x16x256x128xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x256x128xf16>, [#const.Reorder<#NHWC>]
    %PERMUTEQUANTIZE = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs(%PERMUTEQUANTIZE : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>
    %ADD = IE.Add(%SHAPE_CAST, %CST) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>
    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs(%ADD : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>
    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>

    return %OUT_MEM_PERMUTE : tensor<1x2x512x512xf16>

    // CHECK:   [[PERMUTEQUANTIZE_0:%.+]] = IE.PermuteQuantize([[ARG_0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 128]} inputs([[PERMUTEQUANTIZE_0]] : tensor<1x2x512x512xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[SHAPECAST_1]], %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<1x16x256x128xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 2, 512, 512]} inputs([[ADD_2]] : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x2x512x512xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_4:%.+]] = IE.MemPermute([[SHAPECAST_3]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x512x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    // CHECK:   return [[MEMPERMUTE_4]] : tensor<1x2x512x512xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<1x16x16x16xf16>
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

    // CHECK:   [[MEMPERMUTE_0:%.+]] = IE.MemPermute([[INPUT0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_1:%.+]] = IE.MemPermute([[INPUT1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[MEMPERMUTE_0]], [[MEMPERMUTE_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    // CHECK:   [[ADD_3:%.+]] = IE.Add([[MEMPERMUTE_0]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   return [[ADD_2]], [[ADD_3]] : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotConvertAddWithIdentityODUPermute
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x16x16x16xf16>, [[INPUT1:%.+]]: tensor<1x16x16x16xf16>)
func.func @NotConvertAddWithIdentityODUPermute(%arg0 : tensor<1x16x16x16xf16>, %arg1 : tensor<1x16x16x16xf16>)
        -> tensor<1x16x16x16xf16, {order = #NCWH}> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NCWH}>
    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {dst_order = #NCWH, mem_perm = #NCWH} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NCWH}>
    %ADD = IE.Add(%LHS_MEM_PERMUTE, %RHS_MEM_PERMUTE) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
            tensor<1x16x16x16xf16, {order = #NCWH}>, tensor<1x16x16x16xf16, {order = #NCWH}> -> tensor<1x16x16x16xf16, {order = #NCWH}>
    return %ADD : tensor<1x16x16x16xf16, {order = #NCWH}>

    // CHECK:       [[LHS:%.+]] = IE.MemPermute([[INPUT0]]) {dst_order = #[[$NCWH:.+]], mem_perm = #[[$NCWH]]} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #[[$NCWH]]}>
    // CHECK:       [[RHS:%.+]] = IE.MemPermute([[INPUT1]]) {dst_order = #[[$NCWH]], mem_perm = #[[$NCWH]]} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #[[$NCWH]]}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x16xf16, {order = #[[$NCWH]]}>, tensor<1x16x16x16xf16, {order = #[[$NCWH]]}> -> tensor<1x16x16x16xf16, {order = #[[$NCWH]]}>
    // CHECK-NOT:   IE.MemPermute
    // CHECK:       return [[ADD]] : tensor<1x16x16x16xf16, {order = #[[$NCWH]]}>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x8x4x96xf16> -> tensor<1x96x8x4xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_1:%.+]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x4x8x96xf16> -> tensor<1x96x8x4xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[PERMUTECAST_0]], [[MEMPERMUTE_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x96x8x4xf16, {order = #NHWC}>, tensor<1x96x8x4xf16, {order = #NHWC}> -> tensor<1x96x8x4xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[MULTIPLY_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x96x8x4xf16, {order = #NHWC}> -> tensor<1x8x4x96xf16>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<1x8x4x96xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x256xf16> -> tensor<1x256x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x256xf16> -> tensor<1x256x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x256x4x512xf16, {order = #NHWC}>, tensor<1x256x4x512xf16, {order = #NHWC}> -> tensor<1x256x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[MULTIPLY_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x256x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x256xf16>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<1x4x512x256xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x256xf16> -> tensor<1x256x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_1:%.+]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x512x4x256xf16> -> tensor<1x256x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[PERMUTECAST_0]], [[MEMPERMUTE_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x256x4x512xf16, {order = #NHWC}>, tensor<1x256x4x512xf16, {order = #NHWC}> -> tensor<1x256x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[MULTIPLY_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x256x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x256xf16>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<1x4x512x256xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x512x512xf16> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x512x512xf16> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x2x512xf16, {order = #NHWC}>, tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x512x2x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[MULTIPLY_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x2x512xf16, {order = #NHWC}> -> tensor<1x2x512x512xf16>
    // CHECK:   [[SOFTMAX_4:%.+]] = IE.SoftMax([[PERMUTECAST_3]]) {axisInd = 3 : i64} : tensor<1x2x512x512xf16> -> tensor<1x2x512x512xf16>
    // CHECK:   return [[SOFTMAX_4]] : tensor<1x2x512x512xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x512x512xf16> -> tensor<1x512x1x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[MULTIPLY_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<1x4x512x512xf16>
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

    // CHECK:   [[MULTIPLY_0:%.+]] = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x512x256xf16>, tensor<1x4x512x256xf16> -> tensor<1x4x512x256xf16>
    // CHECK:   return [[MULTIPLY_0]] : tensor<1x4x512x256xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x512x512xf16> -> tensor<1x512x1x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast(%arg2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_4:%.+]] = IE.Add([[MULTIPLY_2]], [[PERMUTECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_5:%.+]] = IE.PermuteCast([[ADD_4]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[PERMUTECAST_5]] : tensor<1x4x512x512xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512x!qElemType> -> tensor<1x512x4x512x!qElemType, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x512x512x!qElemType1> -> tensor<1x512x1x512x!qElemType1, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x512x!qElemType, {order = #NHWC}>, tensor<1x512x1x512x!qElemType1, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType4, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast(%arg2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512x!qElemType2> -> tensor<1x512x4x512x!qElemType2, {order = #NHWC}>
    // CHECK:   [[ADD_4:%.+]] = IE.Add([[MULTIPLY_2]], [[PERMUTECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512x!qElemType4, {order = #NHWC}>, tensor<1x512x4x512x!qElemType2, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType3, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_5:%.+]] = IE.PermuteCast([[ADD_4]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512x!qElemType3, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType3>
    // CHECK:   return [[PERMUTECAST_5]] : tensor<1x4x512x512x!qElemType3>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x512x512xf16> -> tensor<1x512x1x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast(%arg2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_4:%.+]] = IE.Multiply([[MULTIPLY_2]], [[PERMUTECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_5:%.+]] = IE.PermuteCast([[MULTIPLY_4]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[PERMUTECAST_5]] : tensor<1x4x512x512xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512x!qElemType> -> tensor<1x512x4x512x!qElemType, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512x!qElemType1> -> tensor<1x512x4x512x!qElemType1, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512x!qElemType, {order = #NHWC}>, tensor<1x512x4x512x!qElemType1, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType4, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast(%arg2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x512x512x!qElemType2> -> tensor<1x512x1x512x!qElemType2, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_4:%.+]] = IE.Multiply([[ADD_2]], [[PERMUTECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x512x!qElemType4, {order = #NHWC}>, tensor<1x512x1x512x!qElemType2, {order = #NHWC}> -> tensor<1x512x4x512x!qElemType3, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_5:%.+]] = IE.PermuteCast([[MULTIPLY_4]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512x!qElemType3, {order = #NHWC}> -> tensor<1x4x512x512x!qElemType3>
    // CHECK:   return [[PERMUTECAST_5]] : tensor<1x4x512x512x!qElemType3>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast(%arg2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x512x512xf16> -> tensor<1x512x1x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_4:%.+]] = IE.Multiply([[ADD_2]], [[PERMUTECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_5:%.+]] = IE.PermuteCast([[MULTIPLY_4]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[PERMUTECAST_5]] : tensor<1x4x512x512xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_1:%.+]] = IE.Multiply([[PERMUTECAST_0]], %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[MULTIPLY_1]], %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<1x4x512x512xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast(%arg2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_4:%.+]] = IE.Add([[MULTIPLY_2]], [[PERMUTECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_5:%.+]] = IE.PermuteCast([[ADD_4]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x4x512x512xf16>
    // CHECK:   return [[PERMUTECAST_5]] : tensor<1x4x512x512xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_2:%.+]] = IE.Multiply([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast(%arg2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_4:%.+]] = IE.Add([[MULTIPLY_2]], [[PERMUTECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_5:%.+]] = IE.PermuteCast([[ADD_4]]) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHCW}>
    // CHECK:   return [[PERMUTECAST_5]] : tensor<1x512x4x512xf16, {order = #NHCW}>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x512x512xf16> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast(%arg2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x512x512xf16> -> tensor<1x512x1x512xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_4:%.+]] = IE.Multiply([[ADD_2]], [[PERMUTECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x512xf16, {order = #NHWC}>, tensor<1x512x1x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_5:%.+]] = IE.PermuteCast([[MULTIPLY_4]]) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x512x4x512xf16, {order = #NHWC}> -> tensor<1x512x4x512xf16, {order = #NHCW}>
    // CHECK:   return [[PERMUTECAST_5]] : tensor<1x512x4x512xf16, {order = #NHCW}>
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

    // CHECK:   [[LAYOUTCAST_0:%.+]] = IE.LayoutCast(%arg0) {dst_order = #NHWC} : tensor<1x16x1575x72xf16> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    // CHECK:   [[AVGPOOL_1:%.+]] = IE.AvgPool([[LAYOUTCAST_0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72x!qElemType, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_2:%.+]] = IE.LayoutCast([[AVGPOOL_1]]) {dst_order = #NCHW} : tensor<1x16x1575x72x!qElemType, {order = #NHWC}> -> tensor<1x16x1575x72x!qElemType>
    // CHECK:   return [[LAYOUTCAST_2]] : tensor<1x16x1575x72x!qElemType>
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

    // CHECK:   [[LAYOUTCAST_0:%.+]] = IE.LayoutCast(%arg0) {dst_order = #NHWC} : tensor<1x16x1575x72xf16> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    // CHECK:   [[MAXPOOL_1:%.+]] = IE.MaxPool([[LAYOUTCAST_0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72xf16, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_2:%.+]] = IE.LayoutCast([[MAXPOOL_1]]) {dst_order = #NCHW} : tensor<1x16x1575x72xf16, {order = #NHWC}> -> tensor<1x16x1575x72xf16>
    // CHECK:   return [[LAYOUTCAST_2]] : tensor<1x16x1575x72xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x768x1152x3xf16> -> tensor<1x3x768x1152xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 768, 1152, 3]} inputs([[PERMUTECAST_0]] : tensor<1x3x768x1152xf16, {order = #NHWC}>) -> tensor<1x768x1152x3xf16, {order = #NHWC}>
    // CHECK:   [[AVGPOOL_2:%.+]] = IE.AvgPool([[SHAPECAST_1]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x768x1152x3xf16, {order = #NHWC}> -> tensor<1x768x1152x3x!qElemType, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 3, 768, 1152]} inputs([[AVGPOOL_2]] : tensor<1x768x1152x3x!qElemType, {order = #NHWC}>) -> tensor<1x3x768x1152x!qElemType, {order = #NHWC}>
    // CHECK:   return [[SHAPECAST_3]] : tensor<1x3x768x1152x!qElemType, {order = #NHWC}>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x768x1152x3xf16> -> tensor<1x1152x768x3xf16, {order = #NHCW}>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 768, 1152, 3]} inputs([[PERMUTECAST_0]] : tensor<1x1152x768x3xf16, {order = #NHCW}>) -> tensor<1x768x1152x3xf16, {order = #NHCW}>
    // CHECK:   [[LAYOUTCAST_2:%.+]] = IE.LayoutCast([[SHAPECAST_1]]) {dst_order = #NHWC} : tensor<1x768x1152x3xf16, {order = #NHCW}> -> tensor<1x768x1152x3xf16, {order = #NHWC}>
    // CHECK:   [[MAXPOOL_3:%.+]] = IE.MaxPool([[LAYOUTCAST_2]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x768x1152x3xf16, {order = #NHWC}> -> tensor<1x768x1152x3x!qElemType, {order = #NHWC}>
    // CHECK:   [[LAYOUTCAST_4:%.+]] = IE.LayoutCast([[MAXPOOL_3]]) {dst_order = #NHCW} : tensor<1x768x1152x3x!qElemType, {order = #NHWC}> -> tensor<1x768x1152x3x!qElemType, {order = #NHCW}>
    // CHECK:   [[SHAPECAST_5:%.+]] = IE.ShapeCast {shape = [1, 1152, 768, 3]} inputs([[LAYOUTCAST_4]] : tensor<1x768x1152x3x!qElemType, {order = #NHCW}>) -> tensor<1x1152x768x3x!qElemType, {order = #NHCW}>
    // CHECK:   return [[SHAPECAST_5]] : tensor<1x1152x768x3x!qElemType, {order = #NHCW}>
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

    // CHECK:   [[MEMPERMUTE_0:%.+]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x1600x2560xf16> -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
    // CHECK:   [[GELU_1:%.+]] = IE.Gelu([[MEMPERMUTE_0]]) : tensor<1x4x1600x2560xf16, {order = #NHWC}> -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[MEMPERMUTE_0]], [[GELU_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf16, {order = #NHWC}>, tensor<1x4x1600x2560xf16, {order = #NHWC}> -> tensor<1x4x1600x2560xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_3:%.+]] = IE.MemPermute([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x1600x2560xf16, {order = #NHWC}> -> tensor<1x4x1600x2560xf16>
    // CHECK:   return [[MEMPERMUTE_3]] : tensor<1x4x1600x2560xf16>
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

    // CHECK:   [[MEMPERMUTE_0:%.+]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x32x64x256xf16> -> tensor<1x64x256x32xf16>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 512, 64, 16]} inputs([[MEMPERMUTE_0]] : tensor<1x64x256x32xf16>) -> tensor<1x512x64x16xf16>
    // CHECK:   [[MEMPERMUTE_2:%.+]] = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x128x256xf16> -> tensor<1x128x256x16xf16>
    // CHECK:   [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 512, 64, 16]} inputs([[MEMPERMUTE_2]] : tensor<1x128x256x16xf16>) -> tensor<1x512x64x16xf16>
    // CHECK:   [[MULTIPLY_4:%.+]] = IE.Multiply([[SHAPECAST_1]], [[SHAPECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x64x16xf16>, tensor<1x512x64x16xf16> -> tensor<1x512x64x16xf16>
    // CHECK:   [[SHAPECAST_5:%.+]] = IE.ShapeCast {shape = [1, 128, 256, 16]} inputs([[MULTIPLY_4]] : tensor<1x512x64x16xf16>) -> tensor<1x128x256x16xf16>
    // CHECK:   [[MEMPERMUTE_6:%.+]] = IE.MemPermute([[SHAPECAST_5]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x128x256x16xf16> -> tensor<1x16x128x256xf16>
    // CHECK:   return [[MEMPERMUTE_6]] : tensor<1x16x128x256xf16>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<4x16x32x56xf16, {order = #NHWC}> -> tensor<4x32x56x16xf16>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 32, 56, 64]} inputs([[PERMUTECAST_0]] : tensor<4x32x56x16xf16>) -> tensor<1x32x56x64xf16>
    // CHECK:   [[PERMUTECAST_2:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<4x16x1x1xf16, {order = #NHWC}> -> tensor<4x1x1x16xf16>
    // CHECK:   [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 64]} inputs([[PERMUTECAST_2]] : tensor<4x1x1x16xf16>) -> tensor<1x1x1x64xf16>
    // CHECK:   [[MULTIPLY_4:%.+]] = IE.Multiply([[SHAPECAST_1]], [[SHAPECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x56x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x32x56x64xf16>
    // CHECK:   [[SHAPECAST_5:%.+]] = IE.ShapeCast {shape = [4, 32, 56, 16]} inputs([[MULTIPLY_4]] : tensor<1x32x56x64xf16>) -> tensor<4x32x56x16xf16>
    // CHECK:   [[PERMUTECAST_6:%.+]] = IE.PermuteCast([[SHAPECAST_5]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<4x32x56x16xf16> -> tensor<4x16x32x56xf16, {order = #NHWC}>
    // CHECK:   return [[PERMUTECAST_6]] : tensor<4x16x32x56xf16, {order = #NHWC}>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x1x32x56xf16, {order = #NHWC}> -> tensor<4x1x32x56xf16>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 4, 32, 56]} inputs([[PERMUTECAST_0]] : tensor<4x1x32x56xf16>) -> tensor<1x4x32x56xf16>
    // CHECK:   [[PERMUTECAST_2:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x56x32xf16, {order = #NHWC}> -> tensor<1x1x56x32xf16>
    // CHECK:   [[SHAPECAST_3:%.+]] = IE.ShapeCast {shape = [1, 1, 32, 56]} inputs([[PERMUTECAST_2]] : tensor<1x1x56x32xf16>) -> tensor<1x1x32x56xf16>
    // CHECK:   [[MULTIPLY_4:%.+]] = IE.Multiply([[SHAPECAST_1]], [[SHAPECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x32x56xf16>, tensor<1x1x32x56xf16> -> tensor<1x4x32x56xf16>
    // CHECK:   [[SHAPECAST_5:%.+]] = IE.ShapeCast {shape = [4, 1, 32, 56]} inputs([[MULTIPLY_4]] : tensor<1x4x32x56xf16>) -> tensor<4x1x32x56xf16>
    // CHECK:   [[PERMUTECAST_6:%.+]] = IE.PermuteCast([[SHAPECAST_5]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x1x32x56xf16> -> tensor<4x1x32x56xf16, {order = #NHWC}>
    // CHECK:   return [[PERMUTECAST_6]] : tensor<4x1x32x56xf16, {order = #NHWC}>
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

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x1x32x56xf16, {order = #NHWC}> -> tensor<4x1x32x56xf16>
    // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 4, 32, 56]} inputs([[PERMUTECAST_0]] : tensor<4x1x32x56xf16>) -> tensor<1x4x32x56xf16>
    // CHECK:   [[PERMUTECAST_2:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x32x56xf16, {order = #NHWC}> -> tensor<1x1x32x56xf16>
    // CHECK:   [[MULTIPLY_3:%.+]] = IE.Multiply([[SHAPECAST_1]], [[PERMUTECAST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x32x56xf16>, tensor<1x1x32x56xf16> -> tensor<1x4x32x56xf16>
    // CHECK:   [[SHAPECAST_4:%.+]] = IE.ShapeCast {shape = [4, 1, 32, 56]} inputs([[MULTIPLY_3]] : tensor<1x4x32x56xf16>) -> tensor<4x1x32x56xf16>
    // CHECK:   [[PERMUTECAST_5:%.+]] = IE.PermuteCast([[SHAPECAST_4]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x1x32x56xf16> -> tensor<4x1x32x56xf16, {order = #NHWC}>
    // CHECK:   return [[PERMUTECAST_5]] : tensor<4x1x32x56xf16, {order = #NHWC}>
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

    // CHECK:   [[CONVERT_0:%.+]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[MEMPERMUTE_1:%.+]] = IE.MemPermute(%arg1) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x?x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[PERMUTECAST_2:%.+]] = IE.PermuteCast([[CONVERT_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x?x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[ADD_3:%.+]] = IE.Add([[MEMPERMUTE_1]], [[PERMUTECAST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x?x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x?x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x?x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[PERMUTECAST_4:%.+]] = IE.PermuteCast([[ADD_3]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x?x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 2560, 3, 1600]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   return [[PERMUTECAST_4]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>

// CHECK-LABEL: @PropagatePermuteWhenDimNIsNotOneSwapDim
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x4x256x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4x4x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4x1x1xf16>,
// CHECK-SAME:      [[INPUT_3:%.+]]: tensor<1x2048x256x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_4:%.+]]: tensor<1x4x256x2048xf16>
func.func @PropagatePermuteWhenDimNIsNotOneSwapDim(%arg0: tensor<1x4x256x1xf16, {order = #NHWC}>,
                %arg1: tensor<4x4x1x1xf16, {order = #NHWC}>,
                %arg2: tensor<1x4x1x1xf16>,
                %arg3: tensor<1x2048x256x1xf16, {order = #NHWC}>,
                %arg4: tensor<1x4x256x2048xf16>) -> tensor<4x1x256x2048xf16> {
    %0 = IE.Convolution(%arg0, %arg1, %arg2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x256x1xf16, {order = #NHWC}>, tensor<4x4x1x1xf16, {order = #NHWC}>, tensor<1x4x1x1xf16> -> tensor<1x4x256x1xf16, {order = #NHWC}>
    %1 = IE.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x2048x256x1xf16, {order = #NHWC}> -> tensor<1x256x1x2048xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x1x2048xf16> -> tensor<1x1x256x2048xf16>
    %3= IE.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048xf16, {order = #NHWC}>
    %4 = IE.Multiply(%0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x1xf16, {order = #NHWC}>, tensor<1x1x256x2048xf16, {order = #NHWC}> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
    %5 = IE.PermuteQuantize(%arg4) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x256x2048xf16> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
    %6 = IE.Add(%5, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x2048xf16, {order = #NHWC}>, tensor<1x4x256x2048xf16, {order = #NHWC}> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
    %7 = IE.MemPermute(%6) {dst_order = #NCHW, mem_perm = #map1} : tensor<1x4x256x2048xf16, {order = #NHWC}> -> tensor<4x1x256x2048xf16>

    return %7 : tensor<4x1x256x2048xf16>

    // CHECK:   [[CONVOLUTION_0:%.+]] = IE.Convolution(%arg0, %arg1, %arg2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x256x1xf16, {order = #NHWC}>, tensor<4x4x1x1xf16, {order = #NHWC}>, tensor<1x4x1x1xf16> -> tensor<1x4x256x1xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x2048x256x1xf16, {order = #NHWC}> -> tensor<1x256x1x2048xf16>
    // CHECK:   [[AFFINERESHAPE_2:%.+]] = IE.AffineReshape([[PERMUTECAST_1]]) {dim_mapping = {{\[\[}}0, 1], [2], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x1x2048xf16> -> tensor<1x1x256x2048xf16>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[AFFINERESHAPE_2]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048xf16, {order = #NHWC}>
    // CHECK:   [[MULTIPLY_4:%.+]] = IE.Multiply([[CONVOLUTION_0]], [[PERMUTECAST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x1xf16, {order = #NHWC}>, tensor<1x1x256x2048xf16, {order = #NHWC}> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_5:%.+]] = IE.PermuteCast(%arg4) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x256x2048xf16> -> tensor<1x2048x4x256xf16, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_6:%.+]] = IE.MemPermute([[MULTIPLY_4]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x4x256x2048xf16, {order = #NHWC}> -> tensor<1x2048x4x256xf16, {order = #NHWC}>
    // CHECK:   [[ADD_7:%.+]] = IE.Add([[PERMUTECAST_5]], [[MEMPERMUTE_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2048x4x256xf16, {order = #NHWC}>, tensor<1x2048x4x256xf16, {order = #NHWC}> -> tensor<1x2048x4x256xf16, {order = #NHWC}>
    // CHECK:   [[SHAPECAST_8:%.+]] = IE.ShapeCast {shape = [4, 2048, 1, 256]} inputs([[ADD_7]] : tensor<1x2048x4x256xf16, {order = #NHWC}>) -> tensor<4x2048x1x256xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_9:%.+]] = IE.PermuteCast([[SHAPECAST_8]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<4x2048x1x256xf16, {order = #NHWC}> -> tensor<4x1x256x2048xf16>
    // CHECK:   return [[PERMUTECAST_9]] : tensor<4x1x256x2048xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateMemPermuteThroughPermuteCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16>, [[ARG_1:%[^:]+]]: tensor<1x16x16x16xf16>)
func.func @PropagateMemPermuteThroughPermuteCast(
    %arg0 : tensor<1x16x16x16xf16>,
    %arg1 : tensor<1x16x16x16xf16>
) -> tensor<1x16x16x16xf16> {
    %MEM_PERMUTE_0 = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %PERMUTE_CAST_0 = IE.PermuteCast(%MEM_PERMUTE_0) {
        dst_order = #NHWC, mem_perm = #NCHW
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %MEM_PERMUTE_1 = IE.MemPermute(%arg1) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %PERMUTE_CAST_1 = IE.PermuteCast(%MEM_PERMUTE_1) {
        dst_order = #NHWC, mem_perm = #NCHW
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %ADD = IE.Add(%PERMUTE_CAST_0, %PERMUTE_CAST_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x16x16xf16, {order = #NHWC}>,
        tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>

    return %OUT_MEM_PERMUTE : tensor<1x16x16x16xf16>

    // Both branches have PermuteCast between MemPermute and Add.
    // The pass traverses PermuteCast to find MemPermute, then propagates the
    // output MemPermute through the eltwise. Consecutive MemPermutes fuse and
    // the trivial PermuteCasts remain as lightweight layout changes.

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<1x16x16x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MixedBranchesOneDirectOneIndirectViaPermuteCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16>, [[ARG_1:%[^:]+]]: tensor<1x16x16x16xf16>)
func.func @MixedBranchesOneDirectOneIndirectViaPermuteCast(
    %arg0 : tensor<1x16x16x16xf16>,
    %arg1 : tensor<1x16x16x16xf16>
) -> tensor<1x16x16x16xf16> {
    %MEM_PERMUTE_0 = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %PERMUTE_CAST = IE.PermuteCast(%MEM_PERMUTE_0) {
        dst_order = #NHWC, mem_perm = #NCHW
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %MEM_PERMUTE_1 = IE.MemPermute(%arg1) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %ADD = IE.Add(%PERMUTE_CAST, %MEM_PERMUTE_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x16x16xf16, {order = #NHWC}>,
        tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>

    return %OUT_MEM_PERMUTE : tensor<1x16x16x16xf16>

    // Branch 0 (indirect via PermuteCast): processNonPermuteBranch fuses the PermuteCast chain
    // into PermuteCast(arg0). Branch 1 (direct MemPermute): fuses via FuseMemPermutes +
    // ConvertToPermuteCast into PermuteCast(arg1). Both paths produce the same zero-copy result.

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<1x16x16x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NoPropagateMixedUsesIndirectPermuteLike
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16>, [[ARG_1:%[^:]+]]: tensor<1x16x16x16xf16>)
func.func @NoPropagateMixedUsesIndirectPermuteLike(
    %arg0 : tensor<1x16x16x16xf16>,
    %arg1 : tensor<1x16x16x16xf16>
) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16>) {
    %MEM_PERMUTE_A = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %PERMUTE_CAST = IE.PermuteCast(%MEM_PERMUTE_A) {
        dst_order = #NHWC, mem_perm = #NCHW
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %MEM_PERMUTE_B = IE.MemPermute(%arg1) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %ADD = IE.Add(%PERMUTE_CAST, %MEM_PERMUTE_B) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x16x16xf16, {order = #NHWC}>,
        tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>

    return %MEM_PERMUTE_A, %OUT_MEM_PERMUTE : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16>

    // PERMUTE_CAST {dst_order=#NHWC, mem_perm=#NCHW} has the same input/output type and identity mem_perm,
    // so PermuteCastOp::fold collapses it to its input (MEM_PERMUTE_A). After folding, MEM_PERMUTE_A
    // directly feeds Add with two users: Add and the return op. hasInputWithMultiUseMemPermute fires
    // because MEM_PERMUTE_A is directly connected to Add and !hasOneUse(). The pass must not transform.

    // CHECK:   [[MP_A:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[MP_B:%.+]] = IE.MemPermute([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[ADD:%.+]] = IE.Add([[MP_A]], [[MP_B]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[OUT_MP:%.+]] = IE.MemPermute([[ADD]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    // CHECK:   return [[MP_A]], [[OUT_MP]] : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NoPropagateMismatchedShapesNoneOrExplicit
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x8x64xf16>, [[ARG_1:%[^:]+]]: tensor<1x8x4x64xf16>)
func.func @NoPropagateMismatchedShapesNoneOrExplicit(
    %arg0 : tensor<1x4x8x64xf16>,
    %arg1 : tensor<1x8x4x64xf16>
) -> tensor<1x4x8x64xf16> {
    %LHS_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x4x8x64xf16> -> tensor<1x4x8x64xf16, {order = #NHWC}>

    %LHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 16, 8]}
        inputs(%LHS_MEM_PERMUTE : tensor<1x4x8x64xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE = IE.MemPermute(%arg1) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x8x4x64xf16> -> tensor<1x8x4x64xf16, {order = #NHWC}>

    %RHS_SHAPE_CAST = IE.ShapeCast {shape = [1, 16, 16, 8]}
        inputs(%RHS_MEM_PERMUTE : tensor<1x8x4x64xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>

    %ADD = IE.Add(%LHS_SHAPE_CAST, %RHS_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x16x16x8xf16, {order = #NHWC}>,
        tensor<1x16x16x8xf16, {order = #NHWC}>
        -> tensor<1x16x16x8xf16, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {shape = [1, 4, 8, 64]}
        inputs(%ADD : tensor<1x16x16x8xf16, {order = #NHWC}>) -> tensor<1x4x8x64xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%OUT_SHAPE_CAST) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x4x8x64xf16, {order = #NHWC}> -> tensor<1x4x8x64xf16>

    return %OUT_MEM_PERMUTE : tensor<1x4x8x64xf16>

    // The two MemPermutes have different underlying shapes (1x4x8x64 vs 1x8x4x64), so after applying
    // the output mem_perm=NWCH the new aligned shapes differ: [1,64,4,8] vs [1,64,8,4].
    // With NONE_OR_EXPLICIT broadcast, newAlignedShape[0] != newAlignedShape[1] triggers matchFailed.

    // CHECK:   [[LHS_MP:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x8x64xf16> -> tensor<1x4x8x64xf16, {order = #NHWC}>
    // CHECK:   [[LHS_SC:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 8]} inputs([[LHS_MP]] : tensor<1x4x8x64xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>
    // CHECK:   [[RHS_MP:%.+]] = IE.MemPermute([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x8x4x64xf16> -> tensor<1x8x4x64xf16, {order = #NHWC}>
    // CHECK:   [[RHS_SC:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 8]} inputs([[RHS_MP]] : tensor<1x8x4x64xf16, {order = #NHWC}>) -> tensor<1x16x16x8xf16, {order = #NHWC}>
    // CHECK:   [[ADD:%.+]] = IE.Add([[LHS_SC]], [[RHS_SC]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x8xf16, {order = #NHWC}>, tensor<1x16x16x8xf16, {order = #NHWC}> -> tensor<1x16x16x8xf16, {order = #NHWC}>
    // CHECK:   [[OUT_SC:%.+]] = IE.ShapeCast {shape = [1, 4, 8, 64]} inputs([[ADD]] : tensor<1x16x16x8xf16, {order = #NHWC}>) -> tensor<1x4x8x64xf16, {order = #NHWC}>
    // CHECK:   [[OUT_MP:%.+]] = IE.MemPermute([[OUT_SC]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x8x64xf16, {order = #NHWC}> -> tensor<1x4x8x64xf16>
    // CHECK:   return [[OUT_MP]] : tensor<1x4x8x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @FuseMemPermuteAndPermuteQuantizeIntoPermuteCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16>, [[ARG_1:%[^:]+]]: tensor<1x16x16x16xf16>)
func.func @FuseMemPermuteAndPermuteQuantizeIntoPermuteCast(
    %arg0 : tensor<1x16x16x16xf16>,
    %arg1 : tensor<1x16x16x16xf16>
) -> tensor<1x16x16x16xf16> {
    %PQ = IE.PermuteQuantize(%arg0) {
        dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
    } : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %MP = IE.MemPermute(%arg1) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %ADD = IE.Add(%PQ, %MP) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x16x16xf16, {order = #NHWC}>,
        tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>

    return %OUT_MEM_PERMUTE : tensor<1x16x16x16xf16>

    // The pass propagates OUT_MEM_PERMUTE through Add. For the PermuteQuantize branch the pass
    // creates MemPermute(PQ.result). The newly-registered FuseMemPermuteAndPermuteQuantize
    // canonicalization fuses this into a single MemPermute(arg0) whose composed
    // permutation is identity; ConvertToPermuteCast then converts it to a zero-copy PermuteCast.
    // The MemPermute branch similarly fuses via FuseMemPermutes + ConvertToPermuteCast.

    // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[ADD_2:%.+]] = IE.Add([[PERMUTECAST_0]], [[PERMUTECAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[ADD_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16>
    // CHECK:   return [[PERMUTECAST_3]] : tensor<1x16x16x16xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateMemPermuteThroughQuantizeCastOnly
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16x!qElemType>, [[ARG_1:%[^:]+]]: tensor<1x16x16x16x!qElemType>)
func.func @PropagateMemPermuteThroughQuantizeCastOnly(
    %arg0 : tensor<1x16x16x16x!qElemType>,
    %arg1 : tensor<1x16x16x16x!qElemType>
) -> tensor<1x16x16x16x!qElemType1> {
    %MEM_PERMUTE_0 = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    %QUANTIZE_CAST_0 = IE.QuantizeCast(%MEM_PERMUTE_0) {dstElemType = !qElemType1}
        : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>

    %MEM_PERMUTE_1 = IE.MemPermute(%arg1) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    %QUANTIZE_CAST_1 = IE.QuantizeCast(%MEM_PERMUTE_1) {dstElemType = !qElemType1}
        : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>

    %ADD = IE.Add(%QUANTIZE_CAST_0, %QUANTIZE_CAST_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x16x16x!qElemType1, {order = #NHWC}>,
        tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
        -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1>

    return %OUT_MEM_PERMUTE : tensor<1x16x16x16x!qElemType1>

    // Both branches have only a QuantizeCast (no PermuteCast) between MemPermute and Add.
    // getInputPermuteLikeOp traverses QuantizeCast to find each MemPermute, but
    // isPermuteLikeDirectlyReachable returns false (QuantizeCast is not ShapeCast).
    // processNonPermuteBranch inserts MemPermute after each QuantizeCast; FuseMemPermAndPermCast
    // then merges the resulting MemPermute→PermuteCast pairs into single MemPermutes with NHWC output.

    // CHECK:   [[MEMPERMUTE_0:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>
    // CHECK:   [[QUANTIZECAST_1:%.+]] = IE.QuantizeCast([[MEMPERMUTE_0]]) {dstElemType = !qElemType1} : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_2:%.+]] = IE.MemPermute([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>
    // CHECK:   [[QUANTIZECAST_3:%.+]] = IE.QuantizeCast([[MEMPERMUTE_2]]) {dstElemType = !qElemType1} : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_4:%.+]] = IE.MemPermute([[QUANTIZECAST_1]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_5:%.+]] = IE.MemPermute([[QUANTIZECAST_3]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:   [[ADD_6:%.+]] = IE.Add([[MEMPERMUTE_4]], [[MEMPERMUTE_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16x!qElemType1, {order = #NHWC}>, tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_7:%.+]] = IE.PermuteCast([[ADD_6]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1>
    // CHECK:   return [[PERMUTECAST_7]] : tensor<1x16x16x16x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// Negative test: indirect upstream IE.MemPermute has a mixed user set (QuantizeCast and a
// non-MemPermute user IE.Sigmoid). canFuseAfterInsertion() returns false because
// fusePermutations cannot fold a MemPermute that feeds non-MemPermute consumers.
// The pass must bail out; the IR must remain unchanged.
//
// CHECK-LABEL: @NoOptimizeIndirectMultiUseMemPermute
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16x!qElemType>)
func.func @NoOptimizeIndirectMultiUseMemPermute(
    %arg0 : tensor<1x16x16x16x!qElemType>
) -> (tensor<1x16x16x16x!qElemType1>, tensor<1x16x16x16x!qElemType, {order = #NHWC}>) {
    // Upstream MemPermute — has two users below: QuantizeCast and MaxPool.
    %MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // Non-MemPermute user — makes the upstream MemPermute ineligible for fusion.
    %POOL = IE.MaxPool(%MEM_PERMUTE) {
        kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    } : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // Indirect path: QuantizeCast sits between MemPermute and Add.
    %QUANTIZE_CAST = IE.QuantizeCast(%MEM_PERMUTE) {dstElemType = !qElemType1}
        : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>

    %ADD = IE.Add(%QUANTIZE_CAST, %QUANTIZE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x16x16x16x!qElemType1, {order = #NHWC}>,
        tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
     -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>

    // Trailing MemPermute — this is the op OptimizeEltwise would normally rewrite.
    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1>

    return %OUT_MEM_PERMUTE, %POOL
        : tensor<1x16x16x16x!qElemType1>, tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    // Pass does NOT fire: MEM_PERMUTE has a non-MemPermute user (POOL), so
    // canFuseAfterInsertion() returns false and the rewriter bails out.
    // All original ops remain in the output.

    // CHECK:   [[MP:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK:   [[POOL:%.+]] = IE.MaxPool([[MP]])
    // CHECK:   [[QC:%.+]] = IE.QuantizeCast([[MP]]) {dstElemType = !qElemType1}
    // CHECK:   [[ADD:%.+]] = IE.Add([[QC]], [[QC]])
    // CHECK:   [[OUT:%.+]] = IE.MemPermute([[ADD]]) {dst_order = #NCHW, mem_perm = #NWCH}
    // CHECK:   return [[OUT]], [[POOL]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 7.013997026518279E-4>
!qElemType1 = !quant.uniform<u8:f16, 0.0014027994053036558>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateMemPermuteThroughQuantizeCastAndPermuteCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16x!qElemType>, [[ARG_1:%[^:]+]]: tensor<1x16x16x16x!qElemType>)
func.func @PropagateMemPermuteThroughQuantizeCastAndPermuteCast(
    %arg0 : tensor<1x16x16x16x!qElemType>,
    %arg1 : tensor<1x16x16x16x!qElemType>
) -> tensor<1x16x16x16x!qElemType1> {
    %MEM_PERMUTE_0 = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    %QUANTIZE_CAST_0 = IE.QuantizeCast(%MEM_PERMUTE_0) {dstElemType = !qElemType1}
        : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>

    %PERMUTE_CAST_0 = IE.PermuteCast(%QUANTIZE_CAST_0) {
        dst_order = #NHWC, mem_perm = #NCHW
    } : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>

    %MEM_PERMUTE_1 = IE.MemPermute(%arg1) {
        dst_order = #NHWC, mem_perm = #NHWC
    } : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>

    %QUANTIZE_CAST_1 = IE.QuantizeCast(%MEM_PERMUTE_1) {dstElemType = !qElemType1}
        : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>

    %PERMUTE_CAST_1 = IE.PermuteCast(%QUANTIZE_CAST_1) {
        dst_order = #NHWC, mem_perm = #NCHW
    } : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>

    %ADD = IE.Add(%PERMUTE_CAST_0, %PERMUTE_CAST_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x16x16x!qElemType1, {order = #NHWC}>,
        tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
        -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {
        dst_order = #NCHW, mem_perm = #NWCH
    } : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1>

    return %OUT_MEM_PERMUTE : tensor<1x16x16x16x!qElemType1>

    // Both branches have QuantizeCast and PermuteCast between MemPermute and Add.
    // The pass traverses QuantizeCast (and PermuteCast, which folds) to find the
    // upstream MemPermute. QuantizeCast blocks direct reachability, so
    // processNonPermuteBranch handles each branch: the output MemPermute becomes
    // a lightweight PermuteCast and a new MemPermute is introduced after each
    // QuantizeCast for downstream fusion.

    // CHECK:   [[MEMPERMUTE_0:%.+]] = IE.MemPermute([[ARG_0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>
    // CHECK:   [[QUANTIZECAST_1:%.+]] = IE.QuantizeCast([[MEMPERMUTE_0]]) {dstElemType = !qElemType1} : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_2:%.+]] = IE.MemPermute([[ARG_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16x!qElemType, {order = #NHWC}>
    // CHECK:   [[QUANTIZECAST_3:%.+]] = IE.QuantizeCast([[MEMPERMUTE_2]]) {dstElemType = !qElemType1} : tensor<1x16x16x16x!qElemType, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_4:%.+]] = IE.MemPermute([[QUANTIZECAST_1]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:   [[MEMPERMUTE_5:%.+]] = IE.MemPermute([[QUANTIZECAST_3]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:   [[ADD_6:%.+]] = IE.Add([[MEMPERMUTE_4]], [[MEMPERMUTE_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16x!qElemType1, {order = #NHWC}>, tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:   [[PERMUTECAST_7:%.+]] = IE.PermuteCast([[ADD_6]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x16x16x!qElemType1>
    // CHECK:   return [[PERMUTECAST_7]] : tensor<1x16x16x16x!qElemType1>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @ExtractODUPermuteFromAddViewOps
// CHECK-SAME:  ([[ARG_0:%.+]]: tensor<1x1024x13x1xf16, {order = #NHWC}>, [[ARG_1:%.+]]: tensor<1x1024x13x1xf16, {order = #NHWC}>)
func.func @ExtractODUPermuteFromAddViewOps(
        %arg0: tensor<1x1024x13x1xf16, {order = #NHWC}>,
        %arg1: tensor<1x1024x13x1xf16, {order = #NHWC}>)
        -> tensor<1x16x13x64xf16> {
    %ADD = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x1024x13x1xf16, {order = #NHWC}>,
        tensor<1x1024x13x1xf16, {order = #NHWC}>
        -> tensor<1x1024x13x1xf16, {order = #NHWC}>

    %PERMUTE_CAST = IE.PermuteCast(%ADD) {
        dst_order = #NCHW,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
    } : tensor<1x1024x13x1xf16, {order = #NHWC}> -> tensor<13x1024x1x1xf16>

    %RESHAPE = IE.Reshape(%PERMUTE_CAST) {shape_value = [1, 13, 16, 64]} :
        tensor<13x1024x1x1xf16> -> tensor<1x13x16x64xf16>

    %MEM_PERMUTE = IE.MemPermute(%RESHAPE) {
        dst_order = #NCHW,
        mem_perm = #NHCW
    } : tensor<1x13x16x64xf16> -> tensor<1x16x13x64xf16>

    return %MEM_PERMUTE : tensor<1x16x13x64xf16>

    // CHECK:   [[LHS:%.+]] = IE.ShapeCast {shape = [1, 64, 13, 16]} inputs([[ARG_0]] : tensor<1x1024x13x1xf16, {order = #NHWC}>) -> tensor<1x64x13x16xf16, {order = #NHWC}>
    // CHECK:   [[RHS:%.+]] = IE.ShapeCast {shape = [1, 64, 13, 16]} inputs([[ARG_1]] : tensor<1x1024x13x1xf16, {order = #NHWC}>) -> tensor<1x64x13x16xf16, {order = #NHWC}>
    // CHECK:   [[ADD:%.+]] = IE.Add([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x13x16xf16, {order = #NHWC}>, tensor<1x64x13x16xf16, {order = #NHWC}> -> tensor<1x64x13x16xf16, {order = #NHWC}>
    // CHECK:   [[MEM_PERMUTE:%.+]] = IE.MemPermute([[ADD]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x64x13x16xf16, {order = #NHWC}> -> tensor<1x16x13x64xf16>
    // CHECK:   return [[MEM_PERMUTE]] : tensor<1x16x13x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @NotExtractODUPermuteFromAddViewOpsMultiUsePermuteCast
// CHECK-SAME:  ([[ARG_0:%.+]]: tensor<1x1024x13x1xf16, {order = #NHWC}>, [[ARG_1:%.+]]: tensor<1x1024x13x1xf16, {order = #NHWC}>)
func.func @NotExtractODUPermuteFromAddViewOpsMultiUsePermuteCast(
        %arg0: tensor<1x1024x13x1xf16, {order = #NHWC}>,
        %arg1: tensor<1x1024x13x1xf16, {order = #NHWC}>)
        -> (tensor<1x16x13x64xf16>, tensor<13x1024x1x1xf16>) {
    %ADD = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x1024x13x1xf16, {order = #NHWC}>,
        tensor<1x1024x13x1xf16, {order = #NHWC}>
        -> tensor<1x1024x13x1xf16, {order = #NHWC}>

    %PERMUTE_CAST = IE.PermuteCast(%ADD) {
        dst_order = #NCHW,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
    } : tensor<1x1024x13x1xf16, {order = #NHWC}> -> tensor<13x1024x1x1xf16>

    %RESHAPE = IE.Reshape(%PERMUTE_CAST) {shape_value = [1, 13, 16, 64]} :
        tensor<13x1024x1x1xf16> -> tensor<1x13x16x64xf16>

    %MEM_PERMUTE = IE.MemPermute(%RESHAPE) {
        dst_order = #NCHW,
        mem_perm = #NHCW
    } : tensor<1x13x16x64xf16> -> tensor<1x16x13x64xf16>

    return %MEM_PERMUTE, %PERMUTE_CAST : tensor<1x16x13x64xf16>, tensor<13x1024x1x1xf16>

    // CHECK:   [[ADD:%.+]] = IE.Add([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x13x1xf16, {order = #NHWC}>, tensor<1x1024x13x1xf16, {order = #NHWC}> -> tensor<1x1024x13x1xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[ADD]])
    // CHECK-SAME: tensor<1x1024x13x1xf16, {order = #NHWC}> -> tensor<13x1024x1x1xf16>
    // CHECK:   [[RESHAPE:%.+]] = IE.Reshape([[PERMUTE_CAST]]) {shape_value = [1, 13, 16, 64]} : tensor<13x1024x1x1xf16> -> tensor<1x13x16x64xf16>
    // CHECK:   [[MEM_PERMUTE:%.+]] = IE.MemPermute([[RESHAPE]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x13x16x64xf16> -> tensor<1x16x13x64xf16>
    // CHECK:   return [[MEM_PERMUTE]], [[PERMUTE_CAST]] : tensor<1x16x13x64xf16>, tensor<13x1024x1x1xf16>
}
