//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK: func.func @foo(
func.func @foo(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x3x8x4xf32> {
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x3x16x16xf32>) {
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
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x1x16x4xf32>) {
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
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x1x16x4xf32>) {
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

// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @bif(
func.func @bif(%arg0: tensor<1x3x16x16xf32, {order = #NHWC}>) -> tensor<1x3x8x4xf32, {order = #NHWC}> {
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x16x16x3xf32>) {
  // CHECK-NEXT:      [[EXTRACT_SLICE:%.+]] = tensor.extract_slice [[ARG1]][0, 8, 12, 0] [1, 8, 4, 3] [1, 1, 1, 1] : tensor<1x16x16x3xf32> to tensor<1x8x4x3xf32>
  // CHECK-NEXT:      IE.CGCYield [[EXTRACT_SLICE]] : tensor<1x8x4x3xf32>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x16x16xf32, {order = #NHWC}>) {
    %1 = IE.Slice %arg1 [0, 0, 8, 12] [1, 3, 8, 4] :
      tensor<1x3x16x16xf32, {order = #NHWC}> to
      tensor<1x3x8x4xf32, {order = #NHWC}>
    IE.CGCYield %1 : tensor<1x3x8x4xf32, {order = #NHWC}>
  } -> tensor<1x3x8x4xf32, {order = #NHWC}>
  return %0 : tensor<1x3x8x4xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16:1, {0.030591299019607842,0.032935049019607844,0.34542432}>
!qElemType1 = !quant.uniform<u8:f16:1, {0.031372549019607843,0.030591299019607842,0.032935049019607844,0.34542432}>

// CHECK-DAG: [[QT1:!.+]] = !quant.uniform<u8:f16:3, {0.031372549019607843,0.030591299019607842,0.032935049019607844,0.34542432000000001}>
// CHECK-DAG: [[QT2:!.+]] = !quant.uniform<u8:f16:3, {0.030591299019607842,0.032935049019607844,0.34542432000000001}>

// CHECK: func.func @bat(
func.func @bat(%arg0: tensor<1x4x8x32xf16, {order = #NHWC}>) -> tensor<1x3x8x32x!qElemType, {order = #NHWC}> {
  %foo = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x4x8x32xf16, {order = #NHWC}> -> tensor<1x4x8x32x!qElemType1, {order = #NHWC}>
  %0 = IE.CodeGenCapsule inputs(%foo as %arg1: tensor<1x4x8x32x!qElemType1, {order = #NHWC}>) {
    %1 = IE.Slice %arg1 [0, 1, 0, 0] [1, 3, 8, 32] : tensor<1x4x8x32x!qElemType1, {order = #NHWC}> to tensor<1x3x8x32x!qElemType, {order = #NHWC}>
    IE.CGCYield %1 : tensor<1x3x8x32x!qElemType, {order = #NHWC}>
  } -> tensor<1x3x8x32x!qElemType, {order = #NHWC}>
  return %0 : tensor<1x3x8x32x!qElemType, {order = #NHWC}>

  // CHECK: IE.CodeGenCapsule inputs({{%.+}} as [[ARG1:%.+]]: tensor<1x8x32x4xi8>) {
  // CHECK:  [[ARGCAST:%.+]] = quant.scast [[ARG1]] : tensor<1x8x32x4xi8> to tensor<1x8x32x4x[[QT1]]>
  // CHECK:  [[INSLICECAST:%.+]] = quant.scast [[ARGCAST]] : tensor<1x8x32x4x[[QT1]]> to tensor<1x8x32x4xi8>
  // CHECK:  [[SLICE:%.+]] = tensor.extract_slice [[INSLICECAST]][0, 0, 0, 1] [1, 8, 32, 3] [1, 1, 1, 1] : tensor<1x8x32x4xi8> to tensor<1x8x32x3xi8>
  // CHECK:  [[OUTSLICECAST:%.+]] = quant.scast [[SLICE]] : tensor<1x8x32x3xi8> to tensor<1x8x32x3x[[QT2]]>
  // CHECK:  [[YIELDCAST:%.+]] = quant.scast [[OUTSLICECAST]] : tensor<1x8x32x3x[[QT2]]> to tensor<1x8x32x3xi8>
  // CHECK:  IE.CGCYield [[YIELDCAST]] : tensor<1x8x32x3xi8>
}
