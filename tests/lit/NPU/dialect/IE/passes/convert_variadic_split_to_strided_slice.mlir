//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-variadic-split-to-strided-slice %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @VariadicSplit(%arg: tensor<2x3x4x5xf32, {order = #NCWH}>) -> (tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>) {
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, 2, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>

    // CHECK: func.func @VariadicSplit([[ARG0:%.+]]: tensor<2x3x4x5xf32, {order = #NCWH}>)
    // CHECK:     [[SLICE0:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 2], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>
    // CHECK:     [[SLICE1:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 2], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 4], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>
    // CHECK:     [[SLICE2:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 4], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 5], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x1xf32, {order = #NCWH}>
    // CHECK:     return [[SLICE0]], [[SLICE1]], [[SLICE2]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @VariadicSplitNegativeSplitLength(%arg: tensor<2x3x4x5xf32, {order = #NCWH}>) -> (tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>) {
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, 2, -1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>

    // CHECK: func.func @VariadicSplitNegativeSplitLength([[ARG0:%.+]]: tensor<2x3x4x5xf32, {order = #NCWH}>)
    // CHECK:     [[SLICE0:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 2], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>
    // CHECK:     [[SLICE1:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 2], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 4], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>
    // CHECK:     [[SLICE2:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 4], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 5], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x1xf32, {order = #NCWH}>
    // CHECK:     return [[SLICE0]], [[SLICE1]], [[SLICE2]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @VariadicSplitNegativeAxis(%arg: tensor<2x3x4x5xf32, {order = #NCWH}>) -> (tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>) {
    %variadic:3 = IE.VariadicSplit(%arg) {axis=-1, split_lengths=[2, 2, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>

    // CHECK: func.func @VariadicSplitNegativeAxis([[ARG0:%.+]]: tensor<2x3x4x5xf32, {order = #NCWH}>)
    // CHECK:     [[SLICE0:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 2], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>
    // CHECK:     [[SLICE1:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 2], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 4], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>
    // CHECK:     [[SLICE2:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 4], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 5], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x1xf32, {order = #NCWH}>
    // CHECK:     return [[SLICE0]], [[SLICE1]], [[SLICE2]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @MultipleUsers(%arg: tensor<2x3x4x5xf32, {order = #NCWH}>) -> (tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>) {
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, 2, -1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>
    %cst = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
    %add = IE.Add(%variadic#0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<1xf32> -> tensor<2x3x4x2xf32, {order = #NCWH}>
    return %variadic#0, %variadic#1, %variadic#2 : tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x2xf32, {order = #NCWH}>, tensor<2x3x4x1xf32, {order = #NCWH}>

    // CHECK: func.func @MultipleUsers([[ARG0:%.+]]: tensor<2x3x4x5xf32, {order = #NCWH}>)
    // CHECK:     [[CST:%.+]] = const.Declare
    // CHECK:     [[SLICE0:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 2], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>
    // CHECK:     [[ADD:%.+]] = IE.Add([[SLICE0]], [[CST]])
    // CHECK:     [[SLICE1:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 2], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 4], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x2xf32, {order = #NCWH}>
    // CHECK:     [[SLICE2:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 4], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 5], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32, {order = #NCWH}> -> tensor<2x3x4x1xf32, {order = #NCWH}>
    // CHECK:     return [[SLICE0]], [[SLICE1]], [[SLICE2]]
}

// -----

func.func @FirstUserInsertionPoint(%arg: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>) {
    %variadic:3 = IE.VariadicSplit(%arg) {axis=3, split_lengths=[2, 2, -1]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x1xf32>
    %cst0 = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
    %add1 = IE.Add(%variadic#0, %cst0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x2xf32>, tensor<1xf32> -> tensor<2x3x4x2xf32>
    %cst1 = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>
    %add2 = IE.Add(%variadic#0, %cst1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x2xf32>, tensor<1xf32> -> tensor<2x3x4x2xf32>
    return %variadic#0, %variadic#1, %variadic#1 : tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>

    // Note: This test checks if the pass chooses the correct insertion points, as this affects the schedule. This is related to #-155244.
    // CHECK: func.func @FirstUserInsertionPoint([[ARG0:%.+]]: tensor<2x3x4x5xf32>)
    //            Note: %variadic#2 has no users. That's why it's inserted at the original op location.
    // CHECK:     [[SLICE2:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 4], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 5], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x1xf32>
    // CHECK:     [[CST0:%.+]] = const.Declare
    // CHECK:     [[SLICE0:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 2], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>
    // CHECK:     [[ADD0:%.+]] = IE.Add([[SLICE0]], [[CST0]])
    // CHECK:     [[CST1:%.+]] = const.Declare
    // CHECK:     [[ADD1:%.+]] = IE.Add([[SLICE0]], [[CST1]])
    // CHECK:     [[SLICE1:%.+]] = IE.StridedSlice([[ARG0]]) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 2], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [2, 3, 4, 4], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<2x3x4x5xf32> -> tensor<2x3x4x2xf32>
    // CHECK:     return [[SLICE0]], [[SLICE1]], [[SLICE1]]
}
