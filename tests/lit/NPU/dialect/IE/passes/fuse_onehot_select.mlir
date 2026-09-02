//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --fuse-onehot-select %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FuseGroupedGatherSelect
// CHECK-SAME:    ([[GID:%[^:]+]]: tensor<1xsi64>, [[TOK:%[^:]+]]: tensor<1xsi64>)
func.func @FuseGroupedGatherSelect(%gid: tensor<1xsi64>, %tok: tensor<1xsi64>) -> tensor<1x8xf32> {
    %depth = const.Declare tensor<si64> = dense<4> : tensor<si64>
    %on = const.Declare tensor<f32> = dense<1.0> : tensor<f32>
    %off = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %onehot = IE.OneHot(%gid, %depth, %on, %off) {axis_attr = 0 : i64, mode = #IE.one_hot_mode<IGNORE_NEGATIVE>, operandSegmentSizes = array<i32: 1, 1, 1, 1>, outputType = f32} : tensor<1xsi64>, tensor<si64>, tensor<f32>, tensor<f32> -> tensor<4x1xf32>
    %sel = IE.Reshape(%onehot) {shape_value = [4, 1, 1]} : tensor<4x1xf32> -> tensor<4x1x1xf32>

    %tok_i32 = IE.Convert(%tok) {dstElemType = si32} : tensor<1xsi64> -> tensor<1xsi32>
    %e0 = const.Declare tensor<4x8xf16> = dense<1.0> : tensor<4x8xf16>
    %e1 = const.Declare tensor<4x8xf16> = dense<2.0> : tensor<4x8xf16>
    %e2 = const.Declare tensor<4x8xf16> = dense<3.0> : tensor<4x8xf16>
    %e3 = const.Declare tensor<4x8xf16> = dense<4.0> : tensor<4x8xf16>
    %g0 = IE.Gather(%e0, %tok_i32) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<4x8xf16>, tensor<1xsi32> -> tensor<1x8xf16>
    %c0 = IE.Convert(%g0) {dstElemType = f32} : tensor<1x8xf16> -> tensor<1x8xf32>
    %r0 = IE.Reshape(%c0) {shape_value = [1, 1, 8]} : tensor<1x8xf32> -> tensor<1x1x8xf32>
    %g1 = IE.Gather(%e1, %tok_i32) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<4x8xf16>, tensor<1xsi32> -> tensor<1x8xf16>
    %c1 = IE.Convert(%g1) {dstElemType = f32} : tensor<1x8xf16> -> tensor<1x8xf32>
    %r1 = IE.Reshape(%c1) {shape_value = [1, 1, 8]} : tensor<1x8xf32> -> tensor<1x1x8xf32>
    %g2 = IE.Gather(%e2, %tok_i32) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<4x8xf16>, tensor<1xsi32> -> tensor<1x8xf16>
    %c2 = IE.Convert(%g2) {dstElemType = f32} : tensor<1x8xf16> -> tensor<1x8xf32>
    %r2 = IE.Reshape(%c2) {shape_value = [1, 1, 8]} : tensor<1x8xf32> -> tensor<1x1x8xf32>
    %g3 = IE.Gather(%e3, %tok_i32) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<4x8xf16>, tensor<1xsi32> -> tensor<1x8xf16>
    %c3 = IE.Convert(%g3) {dstElemType = f32} : tensor<1x8xf16> -> tensor<1x8xf32>
    %r3 = IE.Reshape(%c3) {shape_value = [1, 1, 8]} : tensor<1x8xf32> -> tensor<1x1x8xf32>
    %concat = IE.Concat(%r0, %r1, %r2, %r3) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x8xf32>, tensor<1x1x8xf32>, tensor<1x1x8xf32>, tensor<1x1x8xf32> -> tensor<4x1x8xf32>

    %mul = IE.Multiply(%sel, %concat) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x1x1xf32>, tensor<4x1x8xf32> -> tensor<4x1x8xf32>
    %out = IE.ReduceSum(%mul) {axes_value = [0]} : tensor<4x1x8xf32> -> tensor<1x8xf32>
    return %out : tensor<1x8xf32>

    // CHECK-NOT:   IE.OneHot
    // CHECK-NOT:   IE.ReduceSum
    // CHECK-NOT:   IE.Concat
    // CHECK-DAG:   [[STACKED:%.+]] = const.Declare tensor<16x8xf16>
    // CHECK:       [[GIDI:%.+]] = IE.Convert([[GID]]) {dstElemType = si32} : tensor<1xsi64> -> tensor<1xsi32>
    // CHECK:       [[SCALE:%.+]] = IE.Multiply([[GIDI]], {{%.+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xsi32>, tensor<1xsi32> -> tensor<1xsi32>
    // CHECK:       [[FLAT:%.+]] = IE.Add([[SCALE]], [[TOKI:%.+]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xsi32>, tensor<1xsi32> -> tensor<1xsi32>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[STACKED]], [[FLAT]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<16x8xf16>, tensor<1xsi32> -> tensor<1x8xf16>
    // CHECK:       [[CVT:%.+]] = IE.Convert([[GATHER]]) {dstElemType = f32} : tensor<1x8xf16> -> tensor<1x8xf32>
    // CHECK:       return [[CVT]] : tensor<1x8xf32>
}

// -----

// CHECK-LABEL: @NonSharedIndexGatherUntouched
func.func @NonSharedIndexGatherUntouched(%gid: tensor<1xsi64>, %tok0: tensor<1xsi32>, %tok1: tensor<1xsi32>) -> tensor<1x8xf32> {
    %depth = const.Declare tensor<si64> = dense<2> : tensor<si64>
    %on = const.Declare tensor<f32> = dense<1.0> : tensor<f32>
    %off = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %onehot = IE.OneHot(%gid, %depth, %on, %off) {axis_attr = 0 : i64, mode = #IE.one_hot_mode<IGNORE_NEGATIVE>, operandSegmentSizes = array<i32: 1, 1, 1, 1>, outputType = f32} : tensor<1xsi64>, tensor<si64>, tensor<f32>, tensor<f32> -> tensor<2x1xf32>
    %sel = IE.Reshape(%onehot) {shape_value = [2, 1, 1]} : tensor<2x1xf32> -> tensor<2x1x1xf32>

    %e0 = const.Declare tensor<4x8xf16> = dense<1.0> : tensor<4x8xf16>
    %e1 = const.Declare tensor<4x8xf16> = dense<2.0> : tensor<4x8xf16>
    %g0 = IE.Gather(%e0, %tok0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<4x8xf16>, tensor<1xsi32> -> tensor<1x8xf16>
    %c0 = IE.Convert(%g0) {dstElemType = f32} : tensor<1x8xf16> -> tensor<1x8xf32>
    %r0 = IE.Reshape(%c0) {shape_value = [1, 1, 8]} : tensor<1x8xf32> -> tensor<1x1x8xf32>
    %g1 = IE.Gather(%e1, %tok1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<4x8xf16>, tensor<1xsi32> -> tensor<1x8xf16>
    %c1 = IE.Convert(%g1) {dstElemType = f32} : tensor<1x8xf16> -> tensor<1x8xf32>
    %r1 = IE.Reshape(%c1) {shape_value = [1, 1, 8]} : tensor<1x8xf32> -> tensor<1x1x8xf32>
    %concat = IE.Concat(%r0, %r1) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x8xf32>, tensor<1x1x8xf32> -> tensor<2x1x8xf32>

    %mul = IE.Multiply(%sel, %concat) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x1x1xf32>, tensor<2x1x8xf32> -> tensor<2x1x8xf32>
    %out = IE.ReduceSum(%mul) {axes_value = [0]} : tensor<2x1x8xf32> -> tensor<1x8xf32>
    return %out : tensor<1x8xf32>

    // CHECK:       IE.Gather
    // CHECK:       IE.ReduceSum
}

// -----

// CHECK-LABEL: @FuseGroupedMatMulSelect
// CHECK-SAME:    ([[HID:%[^:]+]]: tensor<1x16xf16>, [[GID:%[^:]+]]: tensor<1xsi64>)
func.func @FuseGroupedMatMulSelect(%hidden: tensor<1x16xf16>, %gid: tensor<1xsi64>) -> tensor<1x32xf16> {
    %depth = const.Declare tensor<si64> = dense<4> : tensor<si64>
    %on = const.Declare tensor<f16> = dense<1.0> : tensor<f16>
    %off = const.Declare tensor<f16> = dense<0.0> : tensor<f16>
    %onehot = IE.OneHot(%gid, %depth, %on, %off) {axis_attr = 0 : i64, mode = #IE.one_hot_mode<IGNORE_NEGATIVE>, operandSegmentSizes = array<i32: 1, 1, 1, 1>, outputType = f16} : tensor<1xsi64>, tensor<si64>, tensor<f16>, tensor<f16> -> tensor<4x1xf16>
    %sel = IE.Reshape(%onehot) {shape_value = [4, 1, 1]} : tensor<4x1xf16> -> tensor<4x1x1xf16>

    %w0 = const.Declare tensor<32x16xf16> = dense<1.0> : tensor<32x16xf16>
    %w1 = const.Declare tensor<32x16xf16> = dense<2.0> : tensor<32x16xf16>
    %w2 = const.Declare tensor<32x16xf16> = dense<3.0> : tensor<32x16xf16>
    %w3 = const.Declare tensor<32x16xf16> = dense<4.0> : tensor<32x16xf16>
    %mm0 = IE.MatMul(%hidden, %w0) {transpose_b} : tensor<1x16xf16>, tensor<32x16xf16> -> tensor<1x32xf16>
    %r0 = IE.Reshape(%mm0) {shape_value = [1, 1, 32]} : tensor<1x32xf16> -> tensor<1x1x32xf16>
    %mm1 = IE.MatMul(%hidden, %w1) {transpose_b} : tensor<1x16xf16>, tensor<32x16xf16> -> tensor<1x32xf16>
    %r1 = IE.Reshape(%mm1) {shape_value = [1, 1, 32]} : tensor<1x32xf16> -> tensor<1x1x32xf16>
    %mm2 = IE.MatMul(%hidden, %w2) {transpose_b} : tensor<1x16xf16>, tensor<32x16xf16> -> tensor<1x32xf16>
    %r2 = IE.Reshape(%mm2) {shape_value = [1, 1, 32]} : tensor<1x32xf16> -> tensor<1x1x32xf16>
    %mm3 = IE.MatMul(%hidden, %w3) {transpose_b} : tensor<1x16xf16>, tensor<32x16xf16> -> tensor<1x32xf16>
    %r3 = IE.Reshape(%mm3) {shape_value = [1, 1, 32]} : tensor<1x32xf16> -> tensor<1x1x32xf16>
    %concat = IE.Concat(%r0, %r1, %r2, %r3) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16> -> tensor<4x1x32xf16>

    %mul = IE.Multiply(%sel, %concat) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x1x1xf16>, tensor<4x1x32xf16> -> tensor<4x1x32xf16>
    %out = IE.ReduceSum(%mul) {axes_value = [0]} : tensor<4x1x32xf16> -> tensor<1x32xf16>
    return %out : tensor<1x32xf16>

    // CHECK-NOT:   IE.ReduceSum
    // CHECK-NOT:   IE.OneHot
    // CHECK-NOT:   IE.Convert
    // CHECK-DAG:   [[STACKED:%.+]] = const.Declare tensor<4x32x16xf16>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[STACKED]], [[GID]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4x32x16xf16>, tensor<1xsi64> -> tensor<1x32x16xf16>
    // CHECK:       [[W:%.+]] = IE.Reshape([[GATHER]]) {shape_value = [32, 16]} : tensor<1x32x16xf16> -> tensor<32x16xf16>
    // CHECK:       [[MM:%.+]] = IE.MatMul([[HID]], [[W]]) {transpose_b} : tensor<1x16xf16>, tensor<32x16xf16> -> tensor<1x32xf16>
    // CHECK:       return [[MM]] : tensor<1x32xf16>
}

// -----

// CHECK-LABEL: @NonSelectReduceSumUntouched
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<2x1x3xf32>, [[ARG_1:%[^:]+]]: tensor<2x1x3xf32>)
func.func @NonSelectReduceSumUntouched(%arg0: tensor<2x1x3xf32>, %arg1: tensor<2x1x3xf32>) -> tensor<1x3xf32> {
    %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x1x3xf32>, tensor<2x1x3xf32> -> tensor<2x1x3xf32>
    %out = IE.ReduceSum(%mul) {axes_value = [0]} : tensor<2x1x3xf32> -> tensor<1x3xf32>
    return %out : tensor<1x3xf32>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x1x3xf32>, tensor<2x1x3xf32> -> tensor<2x1x3xf32>
    // CHECK:       [[OUT:%.+]] = IE.ReduceSum([[MUL]]) {axes_value = [0]} : tensor<2x1x3xf32> -> tensor<1x3xf32>
    // CHECK:       return [[OUT]] : tensor<1x3xf32>
}
