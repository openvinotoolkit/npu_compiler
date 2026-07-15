//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK: func.func @foo(
func.func @foo(%arg0: tensor<15x2xf32>) -> tensor<10x3x1xf32> {
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<15x2xf32>) {
  // CHECK-NEXT:    [[COLLAPSE:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[}}0, 1{{\]\]}} : tensor<15x2xf32> into tensor<30xf32>
  // CHECK-NEXT:    [[EXPAND:%.+]] = tensor.expand_shape [[COLLAPSE]] {{\[\[0, 1\]\]}} output_shape [30, 1] : tensor<30xf32> into tensor<30x1xf32>
  // CHECK-NEXT:    IE.CGCYield [[EXPAND]] : tensor<30x1xf32>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<15x2xf32>) {
    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0, 1]], shape_value = [30, 1]} : tensor<15x2xf32> -> tensor<30x1xf32>
    IE.CGCYield %2 : tensor<30x1xf32>
  } -> tensor<30x1xf32>

  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1]]: tensor<30x1xf32>) {
  // CHECK-NEXT:      [[COLLAPSE:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[0, 1\]\]}} : tensor<30x1xf32> into tensor<30xf32>
  // CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[COLLAPSE]] {{\[\[0, 1, 2\]\]}} output_shape [10, 3, 1] : tensor<30xf32> into tensor<10x3x1xf32>
  // CHECK-NEXT:      IE.CGCYield [[EXPAND]] : tensor<10x3x1xf32>
  %1 = IE.CodeGenCapsule inputs(%0 as %arg1: tensor<30x1xf32>) {
    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2]], shape_value = [10, 3, 1]} : tensor<30x1xf32> -> tensor<10x3x1xf32>
    IE.CGCYield %2 : tensor<10x3x1xf32>
  } -> tensor<10x3x1xf32>
  return %1 : tensor<10x3x1xf32>
}

// -----

// CHECK: func.func @bar(
func.func @bar(%arg0: tensor<15x2x1xf32>) -> tensor<30xf32> {
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<15x2x1xf32>) {
  // CHECK-NEXT:    [[COLLAPSE:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[0, 1, 2\]\]}} : tensor<15x2x1xf32> into tensor<30xf32>
  // CHECK-NEXT:    IE.CGCYield [[COLLAPSE]] : tensor<30xf32>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<15x2x1xf32>) {
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0]], shape_value = [30]} : tensor<15x2x1xf32> -> tensor<30xf32>
    IE.CGCYield %1 : tensor<30xf32>
  } -> tensor<30xf32>
  return %0 : tensor<30xf32>
}

// -----

// CHECK: func.func @baz(
func.func @baz(%arg0: tensor<1x512x64x64xf16>) -> tensor<1x1x512x4096xf16> {
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x512x64x64xf16>) {
  // CHECK-NEXT:      [[COLLAPSE:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[0, 1, 2, 3\]\]}} : tensor<1x512x64x64xf16> into tensor<2097152xf16>
  // CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[COLLAPSE]] {{\[\[0, 1, 2, 3\]\]}} output_shape [1, 1, 512, 4096] : tensor<2097152xf16> into tensor<1x1x512x4096xf16>
  // CHECK-NEXT:      IE.CGCYield [[EXPAND]] : tensor<1x1x512x4096xf16>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x512x64x64xf16>) {
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 512, 4096]} : tensor<1x512x64x64xf16> -> tensor<1x1x512x4096xf16>
    IE.CGCYield %1 : tensor<1x1x512x4096xf16>
  } -> tensor<1x1x512x4096xf16>
  return %0 : tensor<1x1x512x4096xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK: [[NCWH:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK: func.func @bat(
func.func @bat(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x1x512x4096xf16, {order = #NCWH}> {

  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x64x64x512xf16>) {
  // CHECK-NEXT:      [[COLLAPSE:%.+]] = tensor.collapse_shape [[ARG1:%.+]] {{\[\[}}0, 1, 2, 3{{\]\]}} : tensor<1x64x64x512xf16> into tensor<2097152xf16>
  // CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[COLLAPSE]] {{\[\[0, 1, 2\]\]}} output_shape [1, 4096, 512] : tensor<2097152xf16> into tensor<1x4096x512xf16>
  // CHECK-NEXT:      IE.CGCYield [[EXPAND]] : tensor<1x4096x512xf16>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x512x64x64xf16, {order = #NHWC}>) {
    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 512, 4096]} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x4096xf16, {order = #map}>
    IE.CGCYield %2 : tensor<1x512x4096xf16, {order = #map}>
  } -> tensor<1x512x4096xf16, {order = #map}>

  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x4096x512xf16>) {
  // CHECK-NEXT:      [[COLLAPSE:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[0, 1, 2\]\]}} : tensor<1x4096x512xf16> into tensor<2097152xf16>
  // CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[COLLAPSE]] {{\[\[0, 1, 2, 3\]\]}} output_shape [1, 1, 4096, 512] : tensor<2097152xf16> into tensor<1x1x4096x512xf16>
  // CHECK-NEXT:      IE.CGCYield [[EXPAND:%.+]] : tensor<1x1x4096x512xf16>
  %1 = IE.CodeGenCapsule inputs(%0 as %arg1: tensor<1x512x4096xf16, {order = #map}>) {
    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 512, 4096]} : tensor<1x512x4096xf16, {order = #map}> -> tensor<1x1x512x4096xf16, {order = #NCWH}>
    IE.CGCYield %2 : tensor<1x1x512x4096xf16, {order = #NCWH}>
  } -> tensor<1x1x512x4096xf16, {order = #NCWH}>
  return %1 : tensor<1x1x512x4096xf16, {order = #NCWH}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @fiz(
func.func @fiz(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x512x4096x1xf16, {order = #NHWC}> {
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x64x64x512xf16>) {
  // CHECK-NEXT:      [[COLLAPSE:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[0, 1, 2, 3\]\]}} : tensor<1x64x64x512xf16> into tensor<2097152xf16>
  // CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[COLLAPSE]] {{\[\[0, 1, 2, 3\]\]}} output_shape [1, 4096, 1, 512] : tensor<2097152xf16> into tensor<1x4096x1x512xf16>
  // CHECK-NEXT:      IE.CGCYield [[EXPAND]] : tensor<1x4096x1x512xf16>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x512x64x64xf16, {order = #NHWC}>) {
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 512, 4096, 1]} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>
    IE.CGCYield %1 : tensor<1x512x4096x1xf16, {order = #NHWC}>
  } -> tensor<1x512x4096x1xf16, {order = #NHWC}>
  return %0 : tensor<1x512x4096x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK: func.func @fuz(
func.func @fuz(%arg0: tensor<1x1000x1x1xf32>) ->  tensor<1x1x1x1000xf32> {
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x1000x1x1xf32>) {
  // CHECK-NEXT:      [[COLLAPSE:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[0, 1, 2, 3\]\]}} : tensor<1x1000x1x1xf32> into tensor<1000xf32>
  // CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[COLLAPSE]] {{\[\[0, 1, 2, 3\]\]}} output_shape [1, 1, 1, 1000] : tensor<1000xf32> into tensor<1x1x1x1000xf32>
  // CHECK-NEXT:      IE.CGCYield [[EXPAND]] : tensor<1x1x1x1000xf32>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1 : tensor<1x1000x1x1xf32>) {
    %1 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x1000x1x1xf32> -> tensor<1x1x1x1000xf32>
    IE.CGCYield %1 : tensor<1x1x1x1000xf32>
  } ->  tensor<1x1x1x1000xf32>
  return %0 : tensor<1x1x1x1000xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @qiz(
func.func @qiz(%arg0: tensor<1x1000x1x1xf32, {order = #NHWC}>) -> tensor<1x1x1000x1xf32> {
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x1x1x1000xf32>) {
  // CHECK-NEXT:      [[COLLAPSE:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[0, 1, 2, 3\]\]}} : tensor<1x1x1x1000xf32> into tensor<1000xf32>
  // CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[COLLAPSE]] {{\[\[0, 1, 2, 3\]\]}} output_shape [1, 1, 1000, 1] : tensor<1000xf32> into tensor<1x1x1000x1xf32>
  // CHECK-NEXT:      IE.CGCYield [[EXPAND]] : tensor<1x1x1000x1xf32>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1 : tensor<1x1000x1x1xf32, {order = #NHWC}>) {
    %1 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1x1000x1xf32>
    IE.CGCYield %1 : tensor<1x1x1000x1xf32>
  } -> tensor<1x1x1000x1xf32>
  return %0 : tensor<1x1x1000x1xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: [[NHCW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @zip(
func.func @zip(%arg0: tensor<1x1000x1x1xf32, {order = #NHWC}>) ->  tensor<1x1000x1x1xf32, {order = #NHCW}> {
  // CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x1x1x1000xf32>) {
  // CHECK-NEXT:      [[COLLAPSE:%.+]] = tensor.collapse_shape [[ARG1]] {{\[\[0, 1, 2, 3\]\]}} : tensor<1x1x1x1000xf32> into tensor<1000xf32>
  // CHECK-NEXT:      [[EXPAND:%.+]] = tensor.expand_shape [[COLLAPSE]] {{\[\[0, 1, 2, 3\]\]}} output_shape [1, 1, 1000, 1] : tensor<1000xf32> into tensor<1x1x1000x1xf32>
  // CHECK-NEXT:      IE.CGCYield [[EXPAND]] : tensor<1x1x1000x1xf32>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1 : tensor<1x1000x1x1xf32, {order = #NHWC}>) {
    %1 = IE.PermuteCast(%arg1) {dst_order = #NHCW, mem_perm = #NHWC} : tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1000x1x1xf32, {order = #NHCW}>
    IE.CGCYield %1 : tensor<1x1000x1x1xf32, {order = #NHCW}>
  } -> tensor<1x1000x1x1xf32, {order = #NHCW}>
  return %0 : tensor<1x1000x1x1xf32, {order = #NHCW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16:1, {0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329}>
!qElemType1 = !quant.uniform<i8:f16:0, {0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329}>

// CHECK-DAG: [[QT:!.+]] = !quant.uniform<i8:f16:1, {0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329}>
// CHECK-DAG: [[QT1:!.+]] = !quant.uniform<i8:f16:0, {0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329}>

// CHECK: func.func @biz(
func.func @biz(%arg0: tensor<1x4x1x4096x!qElemType>) -> tensor<4x4096x1x1x!qElemType1> {
// CHECK: IE.CodeGenCapsule inputs({{%.+}} as [[ARG1:%.+]]: tensor<1x4x1x4096xi8>) {
// CHECK-NEXT:        [[ARGCAST:%.+]] = quant.scast [[ARG1]] : tensor<1x4x1x4096xi8> to tensor<1x4x1x4096x[[QT]]>
// CHECK-NEXT:        [[SCAST:%.+]] = quant.scast [[ARGCAST]] : tensor<1x4x1x4096x[[QT]]> to tensor<1x4x1x4096xi8>
// CHECK-NEXT:        [[COLLAPSE:%.+]] = tensor.collapse_shape [[SCAST]] {{\[\[}}0, 1, 2, 3{{\]\]}} : tensor<1x4x1x4096xi8> into tensor<16384xi8>
// CHECK-NEXT:        [[EXPAND:%.+]] = tensor.expand_shape [[COLLAPSE]] {{\[\[}}0, 1, 2, 3{{\]\]}} output_shape [4, 4096, 1, 1] : tensor<16384xi8> into tensor<4x4096x1x1xi8>
// CHECK-NEXT:        [[SCAST:%.+]] = quant.scast [[EXPAND]] : tensor<4x4096x1x1xi8> to tensor<4x4096x1x1x[[QT1]]>
// CHECK-NEXT:        [[YIELDCAST:%.+]] = quant.scast [[SCAST]] : tensor<4x4096x1x1x[[QT1]]> to tensor<4x4096x1x1xi8>
// CHECK-NEXT:        IE.CGCYield [[YIELDCAST]] : tensor<4x4096x1x1xi8>
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x4x1x4096x!qElemType>) {
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 4096, 1, 1]} :
        tensor<1x4x1x4096x!qElemType> -> tensor<4x4096x1x1x!qElemType1>
    IE.CGCYield %1 : tensor<4x4096x1x1x!qElemType1>
  } -> tensor<4x4096x1x1x!qElemType1>
  return %0 : tensor<4x4096x1x1x!qElemType1>
}
