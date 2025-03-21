//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --mvn-fusion --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
func.func @FuseMVNInsideSqrt(%arg0: tensor<1x1500x512xf32>) -> tensor<1500x512xf32> {
    %1 = IE.Reshape(%arg0) {shape_value = [1500, 512]} : tensor<1x1500x512xf32> -> tensor<1500x512xf32>

    %mean1Axes = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %mean1 = IE.ReduceMean(%1, %mean1Axes) {keep_dims} : tensor<1500x512xf32>, tensor<si32> -> tensor<1500x1xf32>

    %sub1 = IE.Subtract(%1, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x1xf32> -> tensor<1500x512xf32>
    %mul1 = IE.Multiply(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x512xf32> -> tensor<1500x512xf32>

    %mean2Axes = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %mean2 = IE.ReduceMean(%mul1, %mean2Axes) {keep_dims} : tensor<1500x512xf32>, tensor<1xsi32> -> tensor<1500x1xf32>

    %mul2 = IE.Multiply(%mean1, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1500x1xf32> -> tensor<1500x1xf32>
    %sub2 = IE.Subtract(%mean2, %mul2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1500x1xf32> -> tensor<1500x1xf32>

    %eps = const.Declare tensor<1xf32> = dense<0.000001> : tensor<1xf32>
    %insideAdd = IE.Add(%sub2, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1xf32> -> tensor<1500x1xf32>

    %sqrt = IE.Sqrt(%insideAdd) : tensor<1500x1xf32> -> tensor<1500x1xf32>
    %div = IE.Divide(%sub1, %sqrt) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x1xf32> -> tensor<1500x512xf32>
    return %div : tensor<1500x512xf32>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.ReduceMean
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Sqrt
    // CHECK:  [[PRE_RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK:        tensor<1x1500x512xf32> -> tensor<1x1500x512x1xf32>
    // CHECK:  [[MVN:%.*]] = IE.MVN([[PRE_RESHAPE]])
    // CHECK-SAME:    across_channels = false,
    // CHECK:         eps
    // CHECK-SAME:    normalize_variance = true} : tensor<1x1500x512x1xf32> -> tensor<1x1500x512x1xf32>
    // CHECK:  [[POST_RESHAPE:%.*]] = IE.AffineReshape([[MVN]])
    // CHECK:       tensor<1x1500x512x1xf32> -> tensor<1500x512xf32>
    // CHECK:  return [[POST_RESHAPE]] : tensor<1500x512xf32>
}

// -----

func.func @FuseMVNOutsideSqrt(%arg0: tensor<1500x512xf32>) -> tensor<1500x512xf32> {
    %mean1Axes = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %mean1 = IE.ReduceMean(%arg0, %mean1Axes) {keep_dims} : tensor<1500x512xf32>, tensor<si32> -> tensor<1500x1xf32>

    %sub1 = IE.Subtract(%arg0, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x1xf32> -> tensor<1500x512xf32>
    %mul1 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x512xf32> -> tensor<1500x512xf32>

    %mean2Axes = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %mean2 = IE.ReduceMean(%mul1, %mean2Axes) {keep_dims} : tensor<1500x512xf32>, tensor<1xsi32> -> tensor<1500x1xf32>

    %mul2 = IE.Multiply(%mean1, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1500x1xf32> -> tensor<1500x1xf32>
    %sub2 = IE.Subtract(%mean2, %mul2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1500x1xf32> -> tensor<1500x1xf32>

    %sqrt = IE.Sqrt(%sub2) : tensor<1500x1xf32> -> tensor<1500x1xf32>
    %eps = const.Declare tensor<1xf32> = dense<0.000001> : tensor<1xf32>
    %outsideAdd = IE.Add(%sqrt, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x1xf32>, tensor<1xf32> -> tensor<1500x1xf32>

    %div = IE.Divide(%sub1, %outsideAdd) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1500x512xf32>, tensor<1500x1xf32> -> tensor<1500x512xf32>
    return %div : tensor<1500x512xf32>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.ReduceMean
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Sqrt
    // CHECK:  [[PRE_RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK:        tensor<1500x512xf32> -> tensor<1x1500x512x1xf32>
    // CHECK:  [[MVN:%.*]] = IE.MVN([[PRE_RESHAPE]])
    // CHECK-SAME:    across_channels = false,
    // CHECK:         eps
    // CHECK-SAME:    normalize_variance = true} : tensor<1x1500x512x1xf32> -> tensor<1x1500x512x1xf32>
    // CHECK:  [[POST_RESHAPE:%.*]] = IE.AffineReshape([[MVN]])
    // CHECK:       tensor<1x1500x512x1xf32> -> tensor<1500x512xf32>
    // CHECK:  return [[POST_RESHAPE]] : tensor<1500x512xf32>
}

// -----

func.func @FuseMVNAxes2D(%arg0: tensor<16x1500x512xf32>) -> tensor<16x1500x512xf32> {
    %mean1Axes = const.Declare tensor<2xsi32> = dense<[1,2]> : tensor<2xsi32>
    %mean1 = IE.ReduceMean(%arg0, %mean1Axes) {keep_dims} : tensor<16x1500x512xf32>, tensor<2xsi32> -> tensor<16x1x1xf32>

    %sub1 = IE.Subtract(%arg0, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1500x512xf32>, tensor<16x1x1xf32> -> tensor<16x1500x512xf32>
    %mul1 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1500x512xf32>, tensor<16x1500x512xf32> -> tensor<16x1500x512xf32>

    %mean2Axes = const.Declare tensor<2xsi32> = dense<[1,2]> : tensor<2xsi32>
    %mean2 = IE.ReduceMean(%mul1, %mean2Axes) {keep_dims} : tensor<16x1500x512xf32>, tensor<2xsi32> -> tensor<16x1x1xf32>

    %mul2 = IE.Multiply(%mean1, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1xf32>, tensor<16x1x1xf32> -> tensor<16x1x1xf32>
    %sub2 = IE.Subtract(%mean2, %mul2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1xf32>, tensor<16x1x1xf32> -> tensor<16x1x1xf32>

    %sqrt = IE.Sqrt(%sub2) : tensor<16x1x1xf32> -> tensor<16x1x1xf32>
    %eps = const.Declare tensor<1xf32> = dense<0.000001> : tensor<1xf32>
    %outsideAdd = IE.Add(%sqrt, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1x1xf32>, tensor<1xf32> -> tensor<16x1x1xf32>

    %div = IE.Divide(%sub1, %outsideAdd) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x1500x512xf32>, tensor<16x1x1xf32> -> tensor<16x1500x512xf32>
    return %div : tensor<16x1500x512xf32>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.ReduceMean
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Sqrt
    // CHECK:  [[PRE_RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK:        tensor<16x1500x512xf32> -> tensor<1x16x1500x512xf32>
    // CHECK:  [[MVN:%.*]] = IE.MVN([[PRE_RESHAPE]])
    // CHECK-SAME:    across_channels = false,
    // CHECK:         eps
    // CHECK-SAME:    normalize_variance = true} : tensor<1x16x1500x512xf32> -> tensor<1x16x1500x512xf32>
    // CHECK:  [[POST_RESHAPE:%.*]] = IE.AffineReshape([[MVN]])
    // CHECK:       tensor<1x16x1500x512xf32> -> tensor<16x1500x512xf32>
    // CHECK:  return [[POST_RESHAPE]] : tensor<16x1500x512xf32>
}

// -----

func.func @FuseMVNAcrossChannel(%arg0: tensor<1x16x1500x512xf32>) -> tensor<1x16x1500x512xf32> {
    %mean1Axes = const.Declare tensor<3xsi32> = dense<[1,2,3]> : tensor<3xsi32>
    %mean1 = IE.ReduceMean(%arg0, %mean1Axes) {keep_dims} : tensor<1x16x1500x512xf32>, tensor<3xsi32> -> tensor<1x1x1x1xf32>

    %sub1 = IE.Subtract(%arg0, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1500x512xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x1500x512xf32>
    %mul1 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1500x512xf32>, tensor<1x16x1500x512xf32> -> tensor<1x16x1500x512xf32>

    %mean2Axes = const.Declare tensor<3xsi32> = dense<[1,2,3]> : tensor<3xsi32>
    %mean2 = IE.ReduceMean(%mul1, %mean2Axes) {keep_dims} : tensor<1x16x1500x512xf32>, tensor<3xsi32> -> tensor<1x1x1x1xf32>

    %mul2 = IE.Multiply(%mean1, %mean1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
    %sub2 = IE.Subtract(%mean2, %mul2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>

    %sqrt = IE.Sqrt(%sub2) : tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
    %eps = const.Declare tensor<1xf32> = dense<0.000001> : tensor<1xf32>
    %outsideAdd = IE.Add(%sqrt, %eps) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1xf32> -> tensor<1x1x1x1xf32>

    %div = IE.Divide(%sub1, %outsideAdd) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1500x512xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x1500x512xf32>
    return %div : tensor<1x16x1500x512xf32>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Add
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.ReduceMean
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Sqrt
    // CHECK:  [[MVN:%.*]] = IE.MVN(%arg0)
    // CHECK-SAME:    across_channels = true,
    // CHECK:         eps
    // CHECK-SAME:    normalize_variance = true} : tensor<1x16x1500x512xf32> -> tensor<1x16x1500x512xf32>
    // CHECK:  return [[MVN]] : tensor<1x16x1500x512xf32>
}
