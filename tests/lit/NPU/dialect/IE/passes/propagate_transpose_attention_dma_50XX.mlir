//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --propagate-transpose %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// MoveThroughAttentionDMA folds a WH-swapping Transpose feeding pastV, and the inverse Transpose
// consuming presentV, into AttentionDMA itself (vTransposed attribute). The fold's dimension check
// requires pastV to genuinely be in the [Ev, S] layout, so Ev (128) must differ from pastK's Seq (8192).

// CHECK-LABEL: @MoveTransposeThroughAttentionDMA
// CHECK-SAME:      ([[Q:%arg[0-9]]]: tensor<1x40x1x128xf16>, [[K:%arg[0-9]]]: tensor<1x40x1x128xf16>, [[V:%arg[0-9]]]: tensor<1x40x1x128xf16>, [[PASTK:%arg[0-9]]]: tensor<1x40x8192x128xf16>, [[PASTV:%arg[0-9]]]: tensor<1x40x128x8192xf16>, [[SEQLENK:%arg[0-9]]]: tensor<1x1xsi32>)
func.func @MoveTransposeThroughAttentionDMA(%q: tensor<1x40x1x128xf16>, %k: tensor<1x40x1x128xf16>, %v: tensor<1x40x1x128xf16>, %pastK: tensor<1x40x8192x128xf16>, %pastV: tensor<1x40x128x8192xf16>, %seqLenK: tensor<1x1xsi32>) -> (tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>) {
  %pastVT = IE.Transpose(%pastV) {order_value = #NCWH} : tensor<1x40x128x8192xf16> -> tensor<1x40x8192x128xf16>
  %output, %presentK, %presentV = IE.AttentionDMA(%q, %k, %v, %pastK, %pastVT, %seqLenK) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0>} : tensor<1x40x1x128xf16>, tensor<1x40x1x128xf16>, tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x1xsi32> -> tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x8192x128xf16>
  %presentVT = IE.Transpose(%presentV) {order_value = #NCWH} : tensor<1x40x8192x128xf16> -> tensor<1x40x128x8192xf16>
  return %output, %presentK, %presentVT : tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>

  // CHECK-NOT: IE.Transpose
  // CHECK:       [[OUTPUT:%.+]], [[PRESENTK:%.+]], [[PRESENTV:%.+]] = IE.AttentionDMA([[Q]], [[K]], [[V]], [[PASTK]], [[PASTV]], [[SEQLENK]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0>, vTransposed}
  // CHECK-SAME:  tensor<1x40x1x128xf16>, tensor<1x40x1x128xf16>, tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>, tensor<1x1xsi32> -> tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>
  // CHECK:       return [[OUTPUT]], [[PRESENTK]], [[PRESENTV]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// presentV feeds a second consumer besides the Transpose, so the cache cannot be rewired: the
// Transpose result is still required as a distinct value.

// CHECK-LABEL: @NotMoveTransposeThroughAttentionDMAMultiUsePresentV
// CHECK-SAME:      ([[Q:%arg[0-9]]]: tensor<1x40x1x128xf16>, [[K:%arg[0-9]]]: tensor<1x40x1x128xf16>, [[V:%arg[0-9]]]: tensor<1x40x1x128xf16>, [[PASTK:%arg[0-9]]]: tensor<1x40x8192x128xf16>, [[PASTV:%arg[0-9]]]: tensor<1x40x128x8192xf16>, [[SEQLENK:%arg[0-9]]]: tensor<1x1xsi32>)
func.func @NotMoveTransposeThroughAttentionDMAMultiUsePresentV(%q: tensor<1x40x1x128xf16>, %k: tensor<1x40x1x128xf16>, %v: tensor<1x40x1x128xf16>, %pastK: tensor<1x40x8192x128xf16>, %pastV: tensor<1x40x128x8192xf16>, %seqLenK: tensor<1x1xsi32>) -> (tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>) {
  %pastVT = IE.Transpose(%pastV) {order_value = #NCWH} : tensor<1x40x128x8192xf16> -> tensor<1x40x8192x128xf16>
  %output, %presentK, %presentV = IE.AttentionDMA(%q, %k, %v, %pastK, %pastVT, %seqLenK) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0>} : tensor<1x40x1x128xf16>, tensor<1x40x1x128xf16>, tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x1xsi32> -> tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x8192x128xf16>
  %presentVT = IE.Transpose(%presentV) {order_value = #NCWH} : tensor<1x40x8192x128xf16> -> tensor<1x40x128x8192xf16>
  return %output, %presentK, %presentV, %presentVT : tensor<1x40x1x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x8192x128xf16>, tensor<1x40x128x8192xf16>

  // CHECK:       [[PASTVT:%.+]] = IE.Transpose([[PASTV]]) {order_value = #NCWH}
  // CHECK:       [[OUTPUT:%.+]], [[PRESENTK:%.+]], [[PRESENTV:%.+]] = IE.AttentionDMA([[Q]], [[K]], [[V]], [[PASTK]], [[PASTVT]], [[SEQLENK]])
  // CHECK-NOT:   vTransposed
  // CHECK:       [[PRESENTVT:%.+]] = IE.Transpose([[PRESENTV]]) {order_value = #NCWH}
  // CHECK:       return [[OUTPUT]], [[PRESENTK]], [[PRESENTV]], [[PRESENTVT]]
}
