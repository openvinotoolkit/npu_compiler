//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK: func.func @foo(
func.func @foo(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x3x8x4xf32> {
  // CHECK: IE.CodeGenCapsule inputs({{.*}} as [[ARG1:%.+]]: tensor<1x3x16x16xf32>) {
  // CHECK-NEXT:      [[EXTRACT_SLICE:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 8, 12] [1, 3, 8, 4] [1, 1, 1, 1] : tensor<1x3x16x16xf32> to tensor<1x3x8x4xf32>
  // CHECK-NEXT:      IE.CGCYield [[EXTRACT_SLICE]] : tensor<1x3x8x4xf32>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x16x16xf32>) {
    %1 = IE.Slice %arg1 [0, 0, 8, 12] [1, 3, 8, 4] : tensor<1x3x16x16xf32> to tensor<1x3x8x4xf32>
    IE.CGCYield %1 : tensor<1x3x8x4xf32>
  } -> tensor<1x3x8x4xf32>
  return %0 : tensor<1x3x8x4xf32>
}

// -----

// CHECK: func.func @bar(
func.func @bar(%arg0: tensor<1x1x16x4xf32>) -> tensor<1x1x16x1xf32> {
  // CHECK: IE.CodeGenCapsule inputs({{.*}} as [[ARG1:%.+]]: tensor<1x1x16x4xf32>) {
  // CHECK-NEXT:    [[EXTRACT_SLICE:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 3] [1, 1, 16, 1] [1, 1, 1, 1] : tensor<1x1x16x4xf32> to tensor<1x1x16x1xf32>
  // CHECK-NEXT:    IE.CGCYield [[EXTRACT_SLICE]] : tensor<1x1x16x1xf32>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x16x4xf32>) {
    %1 = IE.Slice %arg1 [0, 0, 0, 3] [1, 1, 16, 1] : tensor<1x1x16x4xf32> to tensor<1x1x16x1xf32>
    IE.CGCYield %1 : tensor<1x1x16x1xf32>
  } -> tensor<1x1x16x1xf32>
  return %0 : tensor<1x1x16x1xf32>
}

// -----

// CHECK: func.func @baz(
func.func @baz(%arg0: tensor<1x1x16x4xf32>) -> tensor<1x1x2x1xf32> {
  // CHECK: IE.CodeGenCapsule inputs({{.*}} as [[ARG1:%.+]]: tensor<1x1x16x4xf32>) {
  // CHECK-NEXT:    [[EXTRACT_SLICE:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 14, 3] [1, 1, 2, 1] [1, 1, 1, 1] : tensor<1x1x16x4xf32> to tensor<1x1x2x1xf32>
  // CHECK-NEXT:    IE.CGCYield [[EXTRACT_SLICE]] : tensor<1x1x2x1xf32>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x16x4xf32>) {
    %1 = IE.Slice %arg1 [0, 0, 14, 3] [1, 1, 2, 1] : tensor<1x1x16x4xf32> to tensor<1x1x2x1xf32>
    IE.CGCYield %1 : tensor<1x1x2x1xf32>
  } -> tensor<1x1x2x1xf32>
  return %0 : tensor<1x1x2x1xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @bif(
func.func @bif(%arg0: tensor<1x3x16x16xf32, {order = #NHWC}>) -> tensor<1x3x8x4xf32, {order = #NHWC}> {
  // CHECK: IE.CodeGenCapsule inputs({{.*}} as [[ARG1:%.+]]: tensor<1x3x16x16xf32, {order = [[NHWC]]}>) {
  // CHECK-NEXT:      [[PCAST:%.+]] = IE.PermuteCast([[ARG1]]) {dst_order = [[NCHW]], mem_perm = [[NCHW]]} : tensor<1x3x16x16xf32, {order = [[NHWC]]}> -> tensor<1x16x16x3xf32>
  // CHECK-NEXT:      [[EXTRACT_SLICE:%.+]] = tensor.extract_slice [[PCAST]][0, 8, 12, 0] [1, 8, 4, 3] [1, 1, 1, 1] : tensor<1x16x16x3xf32> to tensor<1x8x4x3xf32>
  // CHECK-NEXT:      [[PCAST1:%.+]] = IE.PermuteCast([[EXTRACT_SLICE]]) {dst_order = [[NHWC]], mem_perm = [[NCHW]]} : tensor<1x8x4x3xf32> -> tensor<1x3x8x4xf32, {order = [[NHWC]]}>
  // CHECK-NEXT:      IE.CGCYield [[PCAST1]] : tensor<1x3x8x4xf32, {order = [[NHWC]]}>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x16x16xf32, {order = #NHWC}>) {
    %1 = IE.Slice %arg1 [0, 0, 8, 12] [1, 3, 8, 4] :
      tensor<1x3x16x16xf32, {order = #NHWC}> to
      tensor<1x3x8x4xf32, {order = #NHWC}>
    IE.CGCYield %1 : tensor<1x3x8x4xf32, {order = #NHWC}>
  } -> tensor<1x3x8x4xf32, {order = #NHWC}>
  return %0 : tensor<1x3x8x4xf32, {order = #NHWC}>
}
