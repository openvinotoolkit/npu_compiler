//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --opt-dynamic-eltwise-shapeof --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @OptDynamicAddWithShapeOf
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1xsi64>
func.func @OptDynamicAddWithShapeOf(%input: tensor<1xsi64>) -> tensor<3xsi64> {
    %cst = const.Declare tensor<1x1x1xsi64> = dense<1> : tensor<1x1x1xsi64>
    %cst_0 = const.Declare tensor<12xsi64> = dense<[40, 57, 1, 32, 43, 86, 94, 52, 30, 59, 78, 27]> : tensor<12xsi64>
    %0 = IE.StridedSlice(%cst_0, %input) {begin_mask = [], ellipsis_mask = [], end_mask = [], ends_attr = [150], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 0, 0>, shrink_axis_mask = [], strides_attr = [1]} : tensor<12xsi64>, tensor<1xsi64> -> tensor<?xsi64, {bounds = [12], order = affine_map<(d0) -> (d0)>}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<?xsi64, {bounds = [12], order = affine_map<(d0) -> (d0)>}>, tensor<1x1x1xsi64> -> tensor<1x1x?xsi64, {bounds = [1, 1, 12], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    %2 = IE.ShapeOf(%1) {dstElemType = si64} : tensor<1x1x?xsi64, {bounds = [1, 1, 12], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> -> tensor<3xsi64>

    return %2 : tensor<3xsi64>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<3xsi64> = dense<[1, 1, -1]> : tensor<3xsi64>
    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<12xsi64> = dense<[40, 57, 1, 32, 43, 86, 94, 52, 30, 59, 78, 27]> : tensor<12xsi64>
    // CHECK:       [[STRIDEDSLICE:%.+]] = IE.StridedSlice([[CST0]], [[INPUT]]) {begin_mask = [], ellipsis_mask = [], end_mask = [], ends_attr = [150], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 0, 0>, shrink_axis_mask = [], strides_attr = [1]} : tensor<12xsi64>, tensor<1xsi64> -> tensor<?xsi64, {bounds = [12], order = #C}>
    // CHECK:       [[RESHAPE:%.+]] = IE.DynamicReshape([[STRIDEDSLICE]], [[CST]]) {output_bounds = [1, 1, 12], output_shape = [1, 1, -9223372036854775808]} : tensor<?xsi64, {bounds = [12], order = #C}>, tensor<3xsi64> -> tensor<1x1x?xsi64, {bounds = [1, 1, 12], order = #CHW}>
    // CHECK:       [[SHAPEOF:%.+]]  = IE.ShapeOf([[RESHAPE]]) {dstElemType = si64} : tensor<1x1x?xsi64, {bounds = [1, 1, 12], order = #CHW}> -> tensor<3xsi64>

    // CHECK:       return [[SHAPEOF]] : tensor<3xsi64>
}

// -----

// CHECK-LABEL: @OptDynamicSigmoidWithShapeOf
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = #NCHW}>
func.func @OptDynamicSigmoidWithShapeOf(%input: tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<4xsi64> {
    %0 = IE.Sigmoid(%input) : tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> -> tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %1 = IE.ShapeOf(%0) {dstElemType = si64} : tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> -> tensor<4xsi64>

    return %1 : tensor<4xsi64>

    // CHECK:       [[SHAPEOF:%.+]] = IE.ShapeOf([[INPUT]]) {dstElemType = si64} : tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = #NCHW}> -> tensor<4xsi64>

    // CHECK:       return [[SHAPEOF]] : tensor<4xsi64>
}

// -----

// CHECK-LABEL: @NonCompliantNonDynamicOnes
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1xsi64>
func.func @NonCompliantNonDynamicOnes(%input: tensor<1xsi64>) -> tensor<3xsi64> {
    %cst = const.Declare tensor<3x2x1xsi64> = dense<1> : tensor<3x2x1xsi64>
    %cst_0 = const.Declare tensor<12xsi64> = dense<[40, 57, 1, 32, 43, 86, 94, 52, 30, 59, 78, 27]> : tensor<12xsi64>
    %0 = IE.StridedSlice(%cst_0, %input) {begin_mask = [], ellipsis_mask = [], end_mask = [], ends_attr = [150], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 0, 0>, shrink_axis_mask = [], strides_attr = [1]} : tensor<12xsi64>, tensor<1xsi64> -> tensor<?xsi64, {bounds = [12], order = affine_map<(d0) -> (d0)>}>
    %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<?xsi64, {bounds = [12], order = affine_map<(d0) -> (d0)>}>, tensor<3x2x1xsi64> -> tensor<3x2x?xsi64, {bounds = [3, 2, 12], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    %2 = IE.ShapeOf(%1) {dstElemType = si64} : tensor<3x2x?xsi64, {bounds = [3, 2, 12], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> -> tensor<3xsi64>

    return %2 : tensor<3xsi64>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<3x2x1xsi64> = dense<1> : tensor<3x2x1xsi64>
    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<12xsi64> = dense<[40, 57, 1, 32, 43, 86, 94, 52, 30, 59, 78, 27]> : tensor<12xsi64>
    // CHECK:       [[STRIDEDSLICE:%.+]] = IE.StridedSlice([[CST0]], [[INPUT]]) {begin_mask = [], ellipsis_mask = [], end_mask = [], ends_attr = [150], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 0, 0>, shrink_axis_mask = [], strides_attr = [1]} : tensor<12xsi64>, tensor<1xsi64> -> tensor<?xsi64, {bounds = [12], order = #C}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[STRIDEDSLICE]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<?xsi64, {bounds = [12], order = #C}>, tensor<3x2x1xsi64> -> tensor<3x2x?xsi64, {bounds = [3, 2, 12], order = #CHW}>
    // CHECK:       [[SHAPEOF:%.+]]  = IE.ShapeOf([[ADD]]) {dstElemType = si64} : tensor<3x2x?xsi64, {bounds = [3, 2, 12], order = #CHW}> -> tensor<3xsi64>

    // CHECK:       return [[SHAPEOF]] : tensor<3xsi64>
}
