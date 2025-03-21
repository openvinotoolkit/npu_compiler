//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-mem-permute-through-softmax %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateMemPermuteThroughSoftmaxWithAxis3
func.func @PropagateMemPermuteThroughSoftmaxWithAxis3(%arg0: tensor<1x4620x4x16xf16, {order = #NHWC}>) -> tensor<1x16x4x4620xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<[[[[0.000000e+00]], [[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]], [[7.000000e+00]], [[8.000000e+00]], [[9.000000e+00]], [[1.000000e+01]], [[1.100000e+01]], [[1.200000e+01]], [[1.300000e+01]], [[1.400000e+01]], [[1.500000e+01]]]]> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
    %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x4620x4x16xf16, {order = #NHWC}> -> tensor<1x4620x4x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x4620x4x16xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    %2 = IE.Convolution(%1, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4620xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    return %2 : tensor<1x16x4x4620xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x4620x4x16xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:               [[SOFTMAX:%.*]] = IE.SoftMax([[MEMPERMUTE]]) {axisInd = 1 : i64} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[SOFTMAX]], [[CST]])

    // CHECK:               return [[CONV]] : tensor<1x16x4x4620xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @PropagateMemPermuteThroughSoftmaxAxis3PermutationNWCH
func.func @PropagateMemPermuteThroughSoftmaxAxis3PermutationNWCH(%arg0: tensor<1x8x1x448xf16, {order = #NHWC}>) -> tensor<1x8x1x448xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x8x1x448xf16, {order = #NHWC}> -> tensor<1x8x1x448xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x1x448xf16, {order = #NHWC}> -> tensor<1x8x1x448xf16>
    return %1 : tensor<1x8x1x448xf16>

    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x8x1x448xf16, {order = #NHWC}> -> tensor<1x8x1x448xf16>
    // CHECK:               [[SOFTMAX:%.*]] = IE.SoftMax([[MEMPERMUTE]]) {axisInd = 3 : i64} : tensor<1x8x1x448xf16> -> tensor<1x8x1x448xf16>

    // CHECK:               return [[SOFTMAX]] : tensor<1x8x1x448xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @PropagateMemPermuteThroughSoftmaxWithAxis1
func.func @PropagateMemPermuteThroughSoftmaxWithAxis1(%arg0: tensor<1x16x4x4620xf16, {order = #NHCW}>) -> tensor<1x16x4x4620xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<[[[[0.000000e+00]], [[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]], [[7.000000e+00]], [[8.000000e+00]], [[9.000000e+00]], [[1.000000e+01]], [[1.100000e+01]], [[1.200000e+01]], [[1.300000e+01]], [[1.400000e+01]], [[1.500000e+01]]]]> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHCW}>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    %2 = IE.Convolution(%1, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4620xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    return %2 : tensor<1x16x4x4620xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:               [[SOFTMAX:%.*]] = IE.SoftMax([[MEMPERMUTE]]) {axisInd = 1 : i64} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[SOFTMAX]], [[CST]])

    // CHECK:               return [[CONV]] : tensor<1x16x4x4620xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @DoNotPropagateMemPermuteThroughSoftmax
func.func @DoNotPropagateMemPermuteThroughSoftmax(%arg0: tensor<1x16x4x4620xf16, {order = #NHCW}>) -> tensor<1x16x4x4620xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<[[[[0.000000e+00]], [[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]], [[7.000000e+00]], [[8.000000e+00]], [[9.000000e+00]], [[1.000000e+01]], [[1.100000e+01]], [[1.200000e+01]], [[1.300000e+01]], [[1.400000e+01]], [[1.500000e+01]]]]> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
    %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHCW}>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    %2 = IE.Convolution(%1, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4620xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    return %2 : tensor<1x16x4x4620xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:               [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHCW}>
    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute([[SOFTMAX]]) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[MEMPERMUTE]], [[CST]])

    // CHECK:               return [[CONV]] : tensor<1x16x4x4620xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateMemPermuteThroughSoftmaxWithAxis1PermutationNHWC
func.func @PropagateMemPermuteThroughSoftmaxWithAxis1PermutationNHWC(%arg0: tensor<1x16x4620x4xf16>) -> tensor<1x4620x4x16xf16> {
  %0 = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x16x4620x4xf16> -> tensor<1x16x4620x4xf16>
  %1 = IE.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x4620x4xf16> -> tensor<1x4620x4x16xf16>
  return %1 : tensor<1x4620x4x16xf16>

    // CHECK:        [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x4620x4xf16> -> tensor<1x4620x4x16xf16>
    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax([[MEMPERMUTE]]) {axisInd = 3 : i64} : tensor<1x4620x4x16xf16> -> tensor<1x4620x4x16xf16>

    // CHECK:         return [[SOFTMAX]] : tensor<1x4620x4x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @InsertMemPermuteBeforeAndAfterSoftmaxNCHW
func.func @InsertMemPermuteBeforeAndAfterSoftmaxNCHW(%arg0: tensor<1x16x4620x4xf16>) -> tensor<1x16x4620x4xf16> {
  %0 = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x16x4620x4xf16> -> tensor<1x16x4620x4xf16>
  return %0 : tensor<1x16x4620x4xf16>

    // CHECK:        [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x4620x4xf16> -> tensor<1x16x4620x4xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax([[MEMPERMUTE]]) {axisInd = 1 : i64} : tensor<1x16x4620x4xf16, {order = #NHWC}> -> tensor<1x16x4620x4xf16, {order = #NHWC}>
    // CHECK:        [[MEMPERMUTE_RESULT:%.*]] = IE.MemPermute([[SOFTMAX]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x4620x4xf16, {order = #NHWC}> -> tensor<1x16x4620x4xf16>
    // CHECK:         return [[MEMPERMUTE_RESULT]] : tensor<1x16x4620x4xf16>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @InsertMemPermuteBeforeAndAfterSoftmaxNHWC
func.func @InsertMemPermuteBeforeAndAfterSoftmaxNHWC(%arg0: tensor<1x16x4620x4xf16, {order = #NHWC}>) -> tensor<1x16x4620x4xf16, {order = #NHWC}> {
  %0 = IE.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x16x4620x4xf16, {order = #NHWC}> -> tensor<1x16x4620x4xf16, {order = #NHWC}>
  return %0 : tensor<1x16x4620x4xf16, {order = #NHWC}>

    // CHECK:        [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NWCH, mem_perm = #NHWC} : tensor<1x16x4620x4xf16, {order = #NHWC}> -> tensor<1x16x4620x4xf16, {order = #NWCH}>
    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax([[MEMPERMUTE]]) {axisInd = 2 : i64} : tensor<1x16x4620x4xf16, {order = #NWCH}> -> tensor<1x16x4620x4xf16, {order = #NWCH}>
    // CHECK:        [[MEMPERMUTE_RESULT:%.*]] = IE.MemPermute([[SOFTMAX]]) {dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x4620x4xf16, {order = #NWCH}> -> tensor<1x16x4620x4xf16, {order = #NHWC}>
    // CHECK:        return [[MEMPERMUTE_RESULT]] : tensor<1x16x4620x4xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @InsertMemPermuteBeforeAndAfterSoftmaxNWCH
func.func @InsertMemPermuteBeforeAndAfterSoftmaxNWCH(%arg0: tensor<1x16x4620x4xf16, {order = #NWCH}>) -> tensor<1x16x4620x4xf16, {order = #NWCH}> {
  %0 = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x16x4620x4xf16, {order = #NWCH}> -> tensor<1x16x4620x4xf16, {order = #NWCH}>
  return %0 : tensor<1x16x4620x4xf16, {order = #NWCH}>

    // CHECK:        [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NWHC, mem_perm = #NCWH} : tensor<1x16x4620x4xf16, {order = #NWCH}> -> tensor<1x16x4620x4xf16, {order = #NWHC}>
    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax([[MEMPERMUTE]]) {axisInd = 1 : i64} : tensor<1x16x4620x4xf16, {order = #NWHC}> -> tensor<1x16x4620x4xf16, {order = #NWHC}>
    // CHECK:        [[MEMPERMUTE_RESULT:%.*]] = IE.MemPermute([[SOFTMAX]]) {dst_order = #NWCH, mem_perm = #NCWH} : tensor<1x16x4620x4xf16, {order = #NWHC}> -> tensor<1x16x4620x4xf16, {order = #NWCH}>
    // CHECK:        return [[MEMPERMUTE_RESULT]] : tensor<1x16x4620x4xf16, {order = #NWCH}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @PropagateMemPermuteThroughSoftmaxWithOnlyOneNonPermuteUse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x4x4620xf16, {order = #NHWC}>
func.func @PropagateMemPermuteThroughSoftmaxWithOnlyOneNonPermuteUse(%arg0: tensor<1x16x4x4620xf16, {order = #NHWC}>) -> (tensor<1x16x4x4620xf16, {order = #NHCW}>, tensor<1x16x4x4620xf16, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<[[[[0.000000e+00]], [[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]], [[7.000000e+00]], [[8.000000e+00]], [[9.000000e+00]], [[1.000000e+01]], [[1.100000e+01]], [[1.200000e+01]], [[1.300000e+01]], [[1.400000e+01]], [[1.500000e+01]]]]> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
    %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NHCW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHCW}>
    %2 = IE.Convolution(%0, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4620xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x16x4x4620xf16, {order = #NHCW}>, tensor<1x16x4x4620xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:               [[MEMPERMUTE_0:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NHCW, mem_perm = #NCWH} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHCW}>
    // CHECK:               [[SOFTMAX:%.+]] = IE.SoftMax([[MEMPERMUTE_0]]) {axisInd = 3 : i64} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHCW}>
    // CHECK:               [[MEMPERMUTE_1:%.+]] = IE.MemPermute([[SOFTMAX]]) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:               [[MEMPERMUTE_2:%.+]] = IE.MemPermute([[MEMPERMUTE_1]]) {dst_order = #NHCW, mem_perm = #NCWH} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHCW}>
    // CHECK:               [[CONV:%.+]] = IE.Convolution([[MEMPERMUTE_1]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4620xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:               return [[MEMPERMUTE_2]], [[CONV]] : tensor<1x16x4x4620xf16, {order = #NHCW}>, tensor<1x16x4x4620xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotPropagateMemPermuteThroughSoftmaxWithMultiUses
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x4x4620xf16, {order = #NHWC}>
func.func @DoNotPropagateMemPermuteThroughSoftmaxWithMultiUses(%arg0: tensor<1x16x4x4620xf16, {order = #NHWC}>) -> (tensor<1x16x4x4620xf32, {order = #NHWC}>, tensor<1x16x4x4620xf16, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<[[[[0.000000e+00]], [[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]], [[7.000000e+00]], [[8.000000e+00]], [[9.000000e+00]], [[1.000000e+01]], [[1.100000e+01]], [[1.200000e+01]], [[1.300000e+01]], [[1.400000e+01]], [[1.500000e+01]]]]> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
    %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf32, {order = #NHWC}>
    %2 = IE.Convolution(%0, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4620xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    return %1, %2 : tensor<1x16x4x4620xf32, {order = #NHWC}>, tensor<1x16x4x4620xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:               [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 3 : i64} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:               [[CONVERT:%.+]] = IE.Convert([[SOFTMAX]]) {dstElemType = f32} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf32, {order = #NHWC}>
    // CHECK:               [[CONV:%.+]] = IE.Convolution([[SOFTMAX]], [[CST]])

    // CHECK:               return [[CONVERT]], [[CONV]] : tensor<1x16x4x4620xf32, {order = #NHWC}>, tensor<1x16x4x4620xf16, {order = #NHWC}>
}
