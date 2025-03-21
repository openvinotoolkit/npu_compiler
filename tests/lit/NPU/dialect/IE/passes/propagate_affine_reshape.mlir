//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-affine-reshape %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @ShiftReshapeAndTranspose(%arg0: tensor<1x12x7x1xf32>, %arg1: tensor<1x1x12x7xf32>) -> (tensor<7x12x1x1xf32>, tensor<7x12x1x1xf32>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x12x7x1xf32> -> tensor<12x7x1x1xf32>
    %1 = IE.Transpose(%0) {order_value = #CNHW} : tensor<12x7x1x1xf32> -> tensor<7x12x1x1xf32>

    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf32> -> tensor<12x7x1x1xf32>
    %3 = IE.Transpose(%2) {order_value = #CNHW} : tensor<12x7x1x1xf32> -> tensor<7x12x1x1xf32>

    return %1, %3 : tensor<7x12x1x1xf32>, tensor<7x12x1x1xf32>

    // CHECK:               [[TRANSPOSE0:%.*]] = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x12x7x1xf32> -> tensor<1x7x12x1xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[TRANSPOSE0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [7, 12, 1, 1]} : tensor<1x7x12x1xf32> -> tensor<7x12x1x1xf32>

    // CHECK:               [[TRANSPOSE1:%.*]] = IE.Transpose(%arg1) {order_value = #NCWH} : tensor<1x1x12x7xf32> -> tensor<1x1x7x12xf32>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [7, 12, 1, 1]} : tensor<1x1x7x12xf32> -> tensor<7x12x1x1xf32>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<7x12x1x1xf32>, tensor<7x12x1x1xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>
#WHC = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

func.func @CollapseReshapeAndTranspose(%arg0: tensor<1x1x12x7xf32>) -> (tensor<7x12xf32>, tensor<7x1x12xf32>) {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [12, 7] } : tensor<1x1x12x7xf32> -> tensor<12x7xf32>
    %1 = IE.Transpose(%0) {order_value = #CN} : tensor<12x7xf32> -> tensor<7x12xf32>

    %2 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 1, 7] } : tensor<1x1x12x7xf32> -> tensor<12x1x7xf32>
    %3 = IE.Transpose(%2) {order_value = #WHC} : tensor<12x1x7xf32> -> tensor<7x1x12xf32>

    return %1, %3 : tensor<7x12xf32>, tensor<7x1x12xf32>

    // CHECK:               [[TRANSPOSE0:%.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x1x12x7xf32> -> tensor<1x1x7x12xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[TRANSPOSE0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [7, 12]} : tensor<1x1x7x12xf32> -> tensor<7x12xf32>

    // CHECK:               [[TRANSPOSE1:%.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x1x12x7xf32> -> tensor<1x1x7x12xf32>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [7, 1, 12]} : tensor<1x1x7x12xf32> -> tensor<7x1x12xf32>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<7x12xf32>, tensor<7x1x12xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @ExtendReshapeAndTranspose(%arg0: tensor<7x12xf32>) -> (tensor<1x1x12x7xf32>, tensor<1x12x7x1xf32>) {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 7, 12] } : tensor<7x12xf32> -> tensor<1x1x7x12xf32>
    %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x1x7x12xf32> -> tensor<1x1x12x7xf32>

    %2 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 12, 1]} : tensor<7x12xf32> -> tensor<1x7x12x1xf32>
    %3 = IE.Transpose(%2) {order_value = #NHCW} : tensor<1x7x12x1xf32> -> tensor<1x12x7x1xf32>

    return %1, %3 : tensor<1x1x12x7xf32>, tensor<1x12x7x1xf32>

    // CHECK:               [[TRANSPOSE0:%.*]] = IE.Transpose(%arg0) {order_value = #CN} : tensor<7x12xf32> -> tensor<12x7xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[TRANSPOSE0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 12, 7]} : tensor<12x7xf32> -> tensor<1x1x12x7xf32>

    // CHECK:               [[TRANSPOSE1:%.*]] = IE.Transpose(%arg0) {order_value = #CN} : tensor<7x12xf32> -> tensor<12x7xf32>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[TRANSPOSE1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 12, 7, 1]} : tensor<12x7xf32> -> tensor<1x12x7x1xf32>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<1x1x12x7xf32>, tensor<1x12x7x1xf32>
}

// -----

func.func @ShiftReshapeAndExpand(%arg0: tensor<1x1x12x7xf32>, %arg1: tensor<1x19x80x1xf16>) -> (tensor<12x16x1x1xf32>, tensor<1x16x19x80xf16>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf32> -> tensor<12x7x1x1xf32>
    %1 = IE.Expand(%0) {pads_begin = [0, 3, 0, 0], pads_end = [0, 6, 0, 0]} : tensor<12x7x1x1xf32> -> tensor<12x16x1x1xf32>

    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 19, 80]} : tensor<1x19x80x1xf16> -> tensor<1x1x19x80xf16>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x19x80xf16> -> tensor<1x16x19x80xf16>

    return %1, %3 : tensor<12x16x1x1xf32>, tensor<1x16x19x80xf16>

    // CHECK:               [[EXPAND0:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 3], pads_end = [0, 0, 0, 6]} : tensor<1x1x12x7xf32> -> tensor<1x1x12x16xf32>
    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[EXPAND0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 16, 1, 1]} : tensor<1x1x12x16xf32> -> tensor<12x16x1x1xf32>

    // CHECK:               [[EXPAND1:%.*]] = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [15, 0, 0, 0]} : tensor<1x19x80x1xf16> -> tensor<16x19x80x1xf16>
    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[EXPAND1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16, 19, 80]} : tensor<16x19x80x1xf16> -> tensor<1x16x19x80xf16>

    // CHECK:               return [[RESHAPE0]], [[RESHAPE1]] : tensor<12x16x1x1xf32>, tensor<1x16x19x80xf16>
}

// -----

func.func @CollapseReshapeAndExpand(%arg0: tensor<1x1x12x7xf32>) -> tensor<16x7xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [12, 7] } : tensor<1x1x12x7xf32> -> tensor<12x7xf32>
    %1 = IE.Expand(%0) {pads_begin = [0, 0], pads_end = [4, 0]}  : tensor<12x7xf32> -> tensor<16x7xf32>
    return %1 : tensor<16x7xf32>

    // CHECK:               [[EXPAND:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<1x1x12x7xf32> -> tensor<1x1x16x7xf32>
    // CHECK:               [[RESHAPE:%.*]]  = IE.AffineReshape([[EXPAND]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [16, 7]} : tensor<1x1x16x7xf32> -> tensor<16x7xf32>
    // CHECK:               return [[RESHAPE]] : tensor<16x7xf32>
}

// -----

func.func @ExtendReshapeAndExpand(%arg0: tensor<7x12xf32>) -> tensor<1x7x16x1xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 12, 1] } : tensor<7x12xf32> -> tensor<1x7x12x1xf32>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 2, 0], pads_end = [0, 0, 2, 0]} : tensor<1x7x12x1xf32> -> tensor<1x7x16x1xf32>
    return %1 : tensor<1x7x16x1xf32>

    // CHECK:               [[EXPAND:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 2], pads_end = [0, 2]} : tensor<7x12xf32> -> tensor<7x16xf32>
    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape([[EXPAND]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 7, 16, 1]} : tensor<7x16xf32> -> tensor<1x7x16x1xf32>
    // CHECK:               return [[RESHAPE]] : tensor<1x7x16x1xf32>
}

// -----

#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

func.func @NoChangesShiftReshape(%arg0: tensor<1x1x12x8xf32>) -> tensor<4x12x2x1xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 4, 2, 1]} : tensor<1x1x12x8xf32> -> tensor<12x4x2x1xf32>
    %1 = IE.Transpose(%0) {order_value = #CNHW} : tensor<12x4x2x1xf32> -> tensor<4x12x2x1xf32>

    return %1 : tensor<4x12x2x1xf32>

    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 4, 2, 1]} : tensor<1x1x12x8xf32> -> tensor<12x4x2x1xf32>
    // CHECK:               [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE]]) {order_value = #map} : tensor<12x4x2x1xf32> -> tensor<4x12x2x1xf32>
    // CHECK:               return [[TRANSPOSE]] : tensor<4x12x2x1xf32>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @NoChangesCollapseReshape(%arg0: tensor<1x2x12x7xf32>, %arg1: tensor<1x1x12x7xf32>, %arg2: tensor<1x1x12x6xf32>)
                                    -> (tensor<7x24xf32>, tensor<12x7x16xf32>, tensor<12x3x16xf32>) {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [24, 7] } : tensor<1x2x12x7xf32> -> tensor<24x7xf32>
    %1 = IE.Transpose(%0) {order_value = #CN} : tensor<24x7xf32> -> tensor<7x24xf32>

    %2 = IE.AffineReshape(%arg1) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 7, 1] } : tensor<1x1x12x7xf32> -> tensor<12x7x1xf32>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0], pads_end = [0, 0, 15]}  : tensor<12x7x1xf32> -> tensor<12x7x16xf32>

    %4 = IE.AffineReshape(%arg2) { dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 3, 2] } : tensor<1x1x12x6xf32> -> tensor<12x3x2xf32>
    %5 = IE.Expand(%4) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]}  : tensor<12x3x2xf32> -> tensor<12x3x16xf32>

    return %1, %3, %5 : tensor<7x24xf32>, tensor<12x7x16xf32>, tensor<12x3x16xf32>

    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [24, 7]} : tensor<1x2x12x7xf32> -> tensor<24x7xf32>
    // CHECK:               [[TRANSPOSE:%.*]] = IE.Transpose([[RESHAPE0]]) {order_value = #CN} : tensor<24x7xf32> -> tensor<7x24xf32>

    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 7, 1]} : tensor<1x1x12x7xf32> -> tensor<12x7x1xf32>
    // CHECK:               [[EXPAND0:%.*]] = IE.Expand([[RESHAPE1]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 15]} : tensor<12x7x1xf32> -> tensor<12x7x16xf32>

    // CHECK:               [[RESHAPE2:%.*]] = IE.AffineReshape(%arg2)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2]], shape_value = [12, 3, 2]} : tensor<1x1x12x6xf32> -> tensor<12x3x2xf32>
    // CHECK:               [[EXPAND1:%.*]] = IE.Expand([[RESHAPE2]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]} : tensor<12x3x2xf32> -> tensor<12x3x16xf32>
    // CHECK:               return [[TRANSPOSE]], [[EXPAND0]], [[EXPAND1]] : tensor<7x24xf32>, tensor<12x7x16xf32>, tensor<12x3x16xf32>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

func.func @NoChangesExpandReshape(%arg0: tensor<6x12xf32>) -> tensor<1x2x12x3xf32> {
    %0 = IE.AffineReshape(%arg0) { dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 3, 12] } : tensor<6x12xf32> -> tensor<1x2x3x12xf32>
    %1 = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x2x3x12xf32> -> tensor<1x2x12x3xf32>
    return %1 : tensor<1x2x12x3xf32>

    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 3, 12]} : tensor<6x12xf32> -> tensor<1x2x3x12xf32>
    // CHECK:               [[TRANSPOSE:%.*]] = IE.Transpose(%0) {order_value = #NCWH} : tensor<1x2x3x12xf32> -> tensor<1x2x12x3xf32>
    // CHECK:               return [[TRANSPOSE]] : tensor<1x2x12x3xf32>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeWithConcatWithOffsets
func.func @SwapAffineReshapeWithConcatWithOffsets(%arg0: tensor<1x1024x1x1xf16>, %arg1: tensor<1x1024x1x1xf16>) ->
                        tensor<2x1x1024xf16> {
     %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [1, 1, 1024]} : tensor<1x1024x1x1xf16> -> tensor<1x1x1024xf16>

      %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [1, 1, 1024]} : tensor<1x1024x1x1xf16> -> tensor<1x1x1024xf16>

     %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0], [1, 0, 0]]} : tensor<1x1x1024xf16>, tensor<1x1x1024xf16> -> tensor<2x1x1024xf16>

     return %2 : tensor<2x1x1024xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x1024x1x1xf16>, tensor<1x1024x1x1xf16> -> tensor<2x1024x1x1xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [2, 1, 1024]}
     // CHECK-SAME:  tensor<2x1024x1x1xf16> -> tensor<2x1x1024xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeWithConcatWithOffsets
func.func @NotSwapAffineReshapeWithConcatWithOffsets(%arg0: tensor<1x40x80x2xf16>, %arg1: tensor<1x20x40x4xf16>) ->
                        tensor<1x60x1x160xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 40, 1, 160]} : tensor<1x40x80x2xf16> -> tensor<1x40x1x160xf16>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 20, 1, 160]} : tensor<1x20x40x4xf16> -> tensor<1x20x1x160xf16>

    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 40, 0, 0]]} : tensor<1x40x1x160xf16>, tensor<1x20x1x160xf16> -> tensor<1x60x1x160xf16>

    return %2 : tensor<1x60x1x160xf16>

    // CHECK:       IE.AffineReshape
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 40, 1, 160]}
    // CHECK-SAME:  tensor<1x40x80x2xf16> -> tensor<1x40x1x160xf16>
    // CHECK:       IE.AffineReshape
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 20, 1, 160]}
    // CHECK-SAME:  tensor<1x20x40x4xf16> -> tensor<1x20x1x160xf16>
    // CHECK:       IE.Concat
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 40, 0, 0]]} : tensor<1x40x1x160xf16>, tensor<1x20x1x160xf16> -> tensor<1x60x1x160xf16>
}

// -----

// CHECK-LABEL: @NoSwapAffineReshapeConcatDiffMapping
func.func @NoSwapAffineReshapeConcatDiffMapping(%arg0: tensor<1x70x1x28xf16>, %arg1: tensor<1x70x28x1xf16>) ->
                        tensor<1x70x56xf16> {

      %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 70, 28]} : tensor<1x70x1x28xf16> -> tensor<1x70x28xf16>
      %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 70, 28]} : tensor<1x70x28x1xf16> -> tensor<1x70x28xf16>
      %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0], [0, 0, 28]]} : tensor<1x70x28xf16>, tensor<1x70x28xf16> -> tensor<1x70x56xf16>

      return %2 : tensor<1x70x56xf16>

     // CHECK:       IE.AffineReshape(%arg0)
     // CHECK:       IE.AffineReshape(%arg1)
     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0], [0, 0, 28]]}
     // CHECK-SAME:           tensor<1x70x28xf16>, tensor<1x70x28xf16> -> tensor<1x70x56xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConcatSameDims
func.func @SwapAffineReshapeConcatSameDims(%arg0: tensor<1x1x12x7xf16>, %arg1: tensor<1x1x12x7xf16>) ->
                        tensor<24x7x1x1xf16> {

      %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf16> -> tensor<12x7x1x1xf16>
      %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [12, 7, 1, 1]} : tensor<1x1x12x7xf16> -> tensor<12x7x1x1xf16>
      %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [12, 0, 0, 0]]} : tensor<12x7x1x1xf16>, tensor<12x7x1x1xf16> -> tensor<24x7x1x1xf16>

      return %2 : tensor<24x7x1x1xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 12, 0]]}
     // CHECK-SAME:           tensor<1x1x12x7xf16>, tensor<1x1x12x7xf16> -> tensor<1x1x24x7xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [24, 7, 1, 1]}
     // CHECK-SAME:  tensor<1x1x24x7xf16> -> tensor<24x7x1x1xf16>
}

// -----

// CHECK-LABEL: @NoPropagateAfiineReshapeDifferentOutput
func.func @NoPropagateAfiineReshapeDifferentOutput(%arg0: tensor<1x32x32x16xf16>, %arg1: tensor<1x16x16x20xf16>) -> tensor<1x21504xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 16384]} : tensor<1x32x32x16xf16> -> tensor<1x16384xf16>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 5120]} : tensor<1x16x16x20xf16> -> tensor<1x5120xf16>
    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0], [0, 16384]]} : tensor<1x16384xf16>, tensor<1x5120xf16> -> tensor<1x21504xf16>

    return %2 : tensor<1x21504xf16>
    // CHECK: IE.AffineReshape
    // CHECK: IE.AffineReshape
    // CHECK: IE.Concat
    // CHECK-SAME: tensor<1x16384xf16>, tensor<1x5120xf16> -> tensor<1x21504xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:3, {0.023082332985073912:128,0.022744074054792816:128,0.020550023808198817:128,0.021637948354085286:128}>

// CHECK-LABEL:  @NoPropagateAfiineReshapeForQuantizeWithPerAxisQuantOutput
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x4x1x1xf16>) -> tensor<1x1x1x4x!qElemType> {
func.func @NoPropagateAfiineReshapeForQuantizeWithPerAxisQuantOutput(%arg0: tensor<1x4x1x1xf16>) -> tensor<1x1x1x4x!qElemType> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 4]} : tensor<1x4x1x1xf16> -> tensor<1x1x1x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x1x1x4xf16> -> tensor<1x1x1x4x!qElemType>

    return %1 : tensor<1x1x1x4x!qElemType>

    // CHECK:                [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 4]} : tensor<1x4x1x1xf16> -> tensor<1x1x1x4xf16>
    // CHECK:                [[QUANTIZE:%.+]] = IE.Quantize([[RESHAPE]]) {dstElemType = !qElemType} : tensor<1x1x1x4xf16> -> tensor<1x1x1x4x!qElemType>
    // CHECK:                return [[QUANTIZE]] : tensor<1x1x1x4x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:3, {0.023082332985073912:128,0.022744074054792816:128,0.020550023808198817:128,0.021637948354085286:128}>

// CHECK-LABEL:  @NoPropagateAfiineReshapeForOneInputEltwiseWithPerAxisQuantIO
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x1x2x4x!qElemType>) -> tensor<1x2x1x4x!qElemType> {
func.func @NoPropagateAfiineReshapeForOneInputEltwiseWithPerAxisQuantIO(%arg0: tensor<1x1x2x4x!qElemType>) -> tensor<1x2x1x4x!qElemType> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 2, 1, 4]} : tensor<1x1x2x4x!qElemType> -> tensor<1x2x1x4x!qElemType>
    %1 = IE.LeakyRelu(%0) {negative_slope = 1.500000e-01 : f64} : tensor<1x2x1x4x!qElemType> -> tensor<1x2x1x4x!qElemType>

    return %1 : tensor<1x2x1x4x!qElemType>

    // CHECK:                [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 2, 1, 4]} : tensor<1x1x2x4x!qElemType> -> tensor<1x2x1x4x!qElemType>
    // CHECK:                [[LEAKYRELU:%.+]] = IE.LeakyRelu([[RESHAPE]]) {negative_slope = 1.500000e-01 : f64} : tensor<1x2x1x4x!qElemType> -> tensor<1x2x1x4x!qElemType>
    // CHECK:                return [[LEAKYRELU]] : tensor<1x2x1x4x!qElemType>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConcatOnNewOneDim
func.func @SwapAffineReshapeConcatOnNewOneDim(%arg0: tensor<1x68x128x128xf16>, %arg1: tensor<1x68x128x128xf16>) ->
                        tensor<1x4x68x128x128xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %3 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 1, 68, 128, 128]} : tensor<1x68x128x128xf16> -> tensor<1x1x68x128x128xf16>

    %4 = IE.Concat(%0, %1, %2, %3) {static_offsets = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 3, 0, 0, 0]]} : tensor<1x1x68x128x128xf16>, tensor<1x1x68x128x128xf16>, tensor<1x1x68x128x128xf16>, tensor<1x1x68x128x128xf16> -> tensor<1x4x68x128x128xf16>

    return %4 :tensor<1x4x68x128x128xf16>

     // CHECK:       IE.Concat
     // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]} : tensor<1x68x128x128xf16>, tensor<1x68x128x128xf16>, tensor<1x68x128x128xf16>, tensor<1x68x128x128xf16> -> tensor<4x68x128x128xf16>
     // CHECK:       IE.AffineReshape
     // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 4, 68, 128, 128]}
     // CHECK-SAME:  tensor<4x68x128x128xf16> -> tensor<1x4x68x128x128xf16>
}

// -----

// CHECK-LABEL: @SwapWithSoftmax
func.func @SwapWithSoftmax(%arg0: tensor<1x24x16x1xf32>) -> tensor<1x1x24x16xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 24, 16]} : tensor<1x24x16x1xf32> -> tensor<1x1x24x16xf32>
    %1 = IE.SoftMax(%0) {axisInd = 2 : i64} : tensor<1x1x24x16xf32> -> tensor<1x1x24x16xf32>
    return %1: tensor<1x1x24x16xf32>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 24, 16]} : tensor<1x24x16x1xf32> -> tensor<1x1x24x16xf32>
    // CHECK:        return [[RESHAPE]] : tensor<1x1x24x16xf32>
}

// -----

// CHECK-LABEL: @SwapWithAvgPool
func.func @SwapWithAvgPool(%arg0: tensor<1x512x768x1xf32>) -> tensor<512x768x1x1xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [512, 768, 1, 1]} : tensor<1x512x768x1xf32> -> tensor<512x768x1x1xf32>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<512x768x1x1xf32> -> tensor<512x768x1x1xf32>
    return %1: tensor<512x768x1x1xf32>

    // CHECK:        [[AVGPOOL:%.*]] = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1],
    // CHECK-SAME:     tensor<1x512x768x1xf32> -> tensor<1x512x768x1xf32>
    // CHECK-NEXT:   [[RESHAPE:%.*]] = IE.AffineReshape([[AVGPOOL]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [512, 768, 1, 1]} : tensor<1x512x768x1xf32> -> tensor<512x768x1x1xf32>
    // CHECK-NEXT:   return [[RESHAPE]] : tensor<512x768x1x1xf32>
}

// -----

// CHECK-LABEL: @SwapWithSoftmaxWithPadSize
func.func @SwapWithSoftmaxWithPadSize(%arg0: tensor<1x1504x1x1500xf16>) -> tensor<1x1504x1500x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1504, 1500, 1]} : tensor<1x1504x1x1500xf16> -> tensor<1x1504x1500x1xf16>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64, padSize = 4 : i64} : tensor<1x1504x1500x1xf16> -> tensor<1x1504x1500x1xf16>
    return %1: tensor<1x1504x1500x1xf16>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64, padSize = 4 : i64} : tensor<1x1504x1x1500xf16> -> tensor<1x1504x1x1500xf16>
    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1504, 1500, 1]} : tensor<1x1504x1x1500xf16> -> tensor<1x1504x1500x1xf16>
    // CHECK: return [[RESHAPE]] : tensor<1x1504x1500x1xf16>
}

// -----

// CHECK-LABEL: @NotSwapWithSoftmaxWhenPadSizeButAxisChanged
func.func @NotSwapWithSoftmaxWhenPadSizeButAxisChanged(%arg0: tensor<1504x1x1500x1xf16>) -> tensor<1x1504x1500x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1504, 1500, 1]} : tensor<1504x1x1500x1xf16> -> tensor<1x1504x1500x1xf16>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64, padSize = 4 : i64} : tensor<1x1504x1500x1xf16> -> tensor<1x1504x1500x1xf16>
    return %1: tensor<1x1504x1500x1xf16>

     // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
     // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1504, 1500, 1]} : tensor<1504x1x1500x1xf16> -> tensor<1x1504x1500x1xf16>
    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax([[RESHAPE]]) {axisInd = 1 : i64, padSize = 4 : i64} : tensor<1x1504x1500x1xf16> -> tensor<1x1504x1500x1xf16>
    // CHECK: return [[SOFTMAX]] : tensor<1x1504x1500x1xf16>
}

// -----

// CHECK-LABEL: @NoSwapWithSoftmaxWhenAxisMerged
func.func @NoSwapWithSoftmaxWhenAxisMerged(%arg0: tensor<1x24x16x1xf32>) -> tensor<1x2x12x16xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 12, 16]} : tensor<1x24x16x1xf32> -> tensor<1x2x12x16xf32>
    %1 = IE.SoftMax(%0) {axisInd = 2 : i64} : tensor<1x2x12x16xf32> -> tensor<1x2x12x16xf32>
    return %1: tensor<1x2x12x16xf32>

    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 12, 16]} : tensor<1x24x16x1xf32> -> tensor<1x2x12x16xf32>
    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax([[RESHAPE]]) {axisInd = 2 : i64} : tensor<1x2x12x16xf32> -> tensor<1x2x12x16xf32>
    // CHECK:        return [[SOFTMAX]] : tensor<1x2x12x16xf32>
}

// -----

// CHECK-LABEL: @SwapWithSoftmaxWhenAxisMergedWithOnes
func.func @SwapWithSoftmaxWhenAxisMergedWithOnes(%arg0: tensor<1x24x16x1xf32>) -> tensor<1x24x1x16xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 24, 1, 16]} : tensor<1x24x16x1xf32> -> tensor<1x24x1x16xf32>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x24x1x16xf32> -> tensor<1x24x1x16xf32>
    return %1: tensor<1x24x1x16xf32>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 24, 1, 16]} : tensor<1x24x16x1xf32> -> tensor<1x24x1x16xf32>
    // CHECK:        return [[RESHAPE]] : tensor<1x24x1x16xf32>
}

// -----

// CHECK-LABEL: @NoSwapWithSoftmaxWhenAxisSplit
func.func @NoSwapWithSoftmaxWhenAxisSplit(%arg0: tensor<4x2x8x1xf32>) -> tensor<1x4x16x1xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 4, 16, 1]} : tensor<4x2x8x1xf32> -> tensor<1x4x16x1xf32>
    %1 = IE.SoftMax(%0) {axisInd = 2 : i64} : tensor<1x4x16x1xf32> -> tensor<1x4x16x1xf32>
    return %1: tensor<1x4x16x1xf32>

    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 4, 16, 1]} : tensor<4x2x8x1xf32> -> tensor<1x4x16x1xf32>
    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax([[RESHAPE]]) {axisInd = 2 : i64} : tensor<1x4x16x1xf32> -> tensor<1x4x16x1xf32>
    // CHECK:        return [[SOFTMAX]] : tensor<1x4x16x1xf32>
}

// -----

// CHECK-LABEL: @SwapWithSoftmaxWhenAxisSplitWithOnes
func.func @SwapWithSoftmaxWhenAxisSplitWithOnes(%arg0: tensor<4x8x1x2xf32>) -> tensor<1x4x8x2xf32> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 4, 8, 2]} : tensor<4x8x1x2xf32> -> tensor<1x4x8x2xf32>
    %1 = IE.SoftMax(%0) {axisInd = 2 : i64} : tensor<1x4x8x2xf32> -> tensor<1x4x8x2xf32>
    return %1: tensor<1x4x8x2xf32>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<4x8x1x2xf32> -> tensor<4x8x1x2xf32>
    // CHECK:        [[RESHAPE:%.*]] = IE.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 4, 8, 2]} : tensor<4x8x1x2xf32> -> tensor<1x4x8x2xf32>
    // CHECK:        return [[RESHAPE]] : tensor<1x4x8x2xf32>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeGelu
func.func @SwapAffineReshapeGelu(%arg0: tensor<1x2048x375x4xf16>) -> tensor<1x2048x1500x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 2048, 1500, 1]} : tensor<1x2048x375x4xf16> -> tensor<1x2048x1500x1xf16>
    %1 = IE.Gelu(%0) : tensor<1x2048x1500x1xf16> -> tensor<1x2048x1500x1xf16>
    return %1 : tensor<1x2048x1500x1xf16>

    // CHECK: [[GELU:%.*]] = IE.Gelu(%arg0) : tensor<1x2048x375x4xf16> -> tensor<1x2048x375x4xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[GELU]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 2048, 1500, 1]}
    // CHECK-SAME:          tensor<1x2048x375x4xf16> -> tensor<1x2048x1500x1xf16>
    // CHECK: return [[AFFINERESHAPE]] : tensor<1x2048x1500x1xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeGeluDueToDifferentRank
func.func @NotSwapAffineReshapeGeluDueToDifferentRank(%arg0: tensor<2048x375x4xf16>) -> tensor<1x2048x1500x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2, 3]], shape_value = [1, 2048, 1500, 1]} : tensor<2048x375x4xf16> -> tensor<1x2048x1500x1xf16>
    %1 = IE.Gelu(%0) : tensor<1x2048x1500x1xf16> -> tensor<1x2048x1500x1xf16>
    return %1 : tensor<1x2048x1500x1xf16>

    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2, 3]], shape_value = [1, 2048, 1500, 1]}
    // CHECK-SAME:          tensor<2048x375x4xf16> -> tensor<1x2048x1500x1xf16>

    // CHECK: [[GELU:%.*]] = IE.Gelu([[AFFINERESHAPE]]) : tensor<1x2048x1500x1xf16> -> tensor<1x2048x1500x1xf16>
    // CHECK: return [[GELU]] : tensor<1x2048x1500x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @SwapAffineReshapeGeluChangeLayout
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x512x4096x1xf16, {order = #NHWC}>
func.func @SwapAffineReshapeGeluChangeLayout(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x512x4096x1xf16, {order = #NHWC}> {
    %0 = IE.AffineReshape(%arg0) {
        dim_mapping = [[0, 1], [2], [3], [3]],
        shape_value = [1, 1, 512, 4096]
    } : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x1x512x4096xf16, {order = #NCWH}>

    %1 = IE.Gelu(%0) : tensor<1x1x512x4096xf16, {order = #NCWH}> -> tensor<1x1x512x4096xf16, {order = #NCWH}>

    %2 = IE.AffineReshape(%1) {
        dim_mapping = [[0], [0], [1], [2, 3]],
        shape_value = [1, 512, 4096, 1]
    } : tensor<1x1x512x4096xf16, {order = #NCWH}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>

    return %2 : tensor<1x512x4096x1xf16, {order = #NHWC}>

    // CHECK:      [[GELU:%.+]] = IE.Gelu([[INPUT]]) :
    // CHECK-SAME:     tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x64x64xf16, {order = #NHWC}>

    // CHECK:      [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[GELU]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0], [1], [2], [2, 3]],
    // CHECK-SAME:     shape_value = [1, 512, 4096, 1]
    // CJECL-SAME: } : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>

    // CHECK:      return [[AFFINERESHAPE]] : tensor<1x512x4096x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeSwish
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1280x1x1xf16>)
func.func @SwapAffineReshapeSwish(%arg0: tensor<1x1280x1x1xf16>) -> tensor<1x1x1x1280xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 1280]} : tensor<1x1280x1x1xf16> -> tensor<1x1x1x1280xf16>
    %1 = IE.Swish(%0) {beta_value = 1.000000e+00 : f64} : tensor<1x1x1x1280xf16> -> tensor<1x1x1x1280xf16>

    return %1 : tensor<1x1x1x1280xf16>

    // CHECK: [[SWISH:%.*]] = IE.Swish([[INPUT]]) {beta_value = 1.000000e+00 : f64} : tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[SWISH]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 1280]}
    // CHECK-SAME:          tensor<1x1280x1x1xf16> -> tensor<1x1x1x1280xf16>

    // CHECK: return [[AFFINERESHAPE]] : tensor<1x1x1x1280xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeSwishDueToDifferentRank
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<2048x375x4xf16>)
func.func @NotSwapAffineReshapeSwishDueToDifferentRank(%arg0: tensor<2048x375x4xf16>) -> tensor<1x2048x1500x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2, 3]], shape_value = [1, 2048, 1500, 1]} : tensor<2048x375x4xf16> -> tensor<1x2048x1500x1xf16>
    %1 = IE.Swish(%0) {beta_value = 1.000000e+00 : f64} : tensor<1x2048x1500x1xf16> -> tensor<1x2048x1500x1xf16>

    return %1 : tensor<1x2048x1500x1xf16>

    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2, 3]], shape_value = [1, 2048, 1500, 1]}
    // CHECK-SAME:          tensor<2048x375x4xf16> -> tensor<1x2048x1500x1xf16>
    // CHECK: [[SWISH:%.*]] = IE.Swish([[AFFINERESHAPE]]) {beta_value = 1.000000e+00 : f64} : tensor<1x2048x1500x1xf16> -> tensor<1x2048x1500x1xf16>

    // CHECK: return [[SWISH]] : tensor<1x2048x1500x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @NotSwapAffineReshapeAbsBecausePreviousTransposeHaveMultipleUses
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<1x256x106x1xf16>)
func.func @NotSwapAffineReshapeAbsBecausePreviousTransposeHaveMultipleUses(%arg0: tensor<1x256x106x1xf16>) -> (tensor<1x1x106x256xf16>, tensor<1x106x1x256xf16>) {
    %0 = IE.Transpose(%arg0) {
        order_value = #map
    } : tensor<1x256x106x1xf16> -> tensor<106x256x1x1xf16>

    %1 = IE.AffineReshape(%0) {
        dim_mapping = [[0, 1, 2], [3], [3], [3]],
        shape_value = [1, 1, 106, 256]
    } : tensor<106x256x1x1xf16> -> tensor<1x1x106x256xf16>

    %2 = IE.Abs(%1) : tensor<1x1x106x256xf16> -> tensor<1x1x106x256xf16>

    %3 = IE.AffineReshape(%0) {
        dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 106, 1, 256]
    } : tensor<106x256x1x1xf16> -> tensor<1x106x1x256xf16>

    return %2, %3 : tensor<1x1x106x256xf16>, tensor<1x106x1x256xf16>

    // CHECK:      [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {
    // CHECK-SAME:     order_value = #map
    // CHECK-SAME: } : tensor<1x256x106x1xf16> -> tensor<106x256x1x1xf16>

    // CHECK:      [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[TRANSPOSE]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0, 1, 2], [3], [3], [3]],
    // CHECK-SAME:     shape_value = [1, 1, 106, 256]
    // CHECK-SAME: } : tensor<106x256x1x1xf16> -> tensor<1x1x106x256xf16>

    // CHECK:      [[ABS:%.+]] = IE.Abs([[AFFINERESHAPE0]]) : tensor<1x1x106x256xf16> -> tensor<1x1x106x256xf16>

    // CHECK:      [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[TRANSPOSE]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 106, 1, 256]
    // CHECK-SAME: } : tensor<106x256x1x1xf16> -> tensor<1x106x1x256xf16>

    // CHECK:      return [[ABS]], [[AFFINERESHAPE1]] : tensor<1x1x106x256xf16>, tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeMultiply
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<4096x2560x1x1xf16>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<4096x2560x1x1xf16>)
func.func @SwapAffineReshapeMultiply(%arg0: tensor<4096x2560x1x1xf16>, %arg1: tensor<4096x2560x1x1xf16>) -> tensor<1x1x4096x2560xf16> {
     %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 4096, 2560]} : tensor<4096x2560x1x1xf16> -> tensor<1x1x4096x2560xf16>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 4096, 2560]} : tensor<4096x2560x1x1xf16> -> tensor<1x1x4096x2560xf16>

    %2 = IE.Multiply(%0, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x4096x2560xf16>, tensor<1x1x4096x2560xf16> -> tensor<1x1x4096x2560xf16>
    return %2 : tensor<1x1x4096x2560xf16>

    // CHECK: [[MULTIPLY:%.*]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x2560x1x1xf16>, tensor<4096x2560x1x1xf16> -> tensor<4096x2560x1x1xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[MULTIPLY]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 4096, 2560]}
    // CHECK-SAME:          tensor<4096x2560x1x1xf16> -> tensor<1x1x4096x2560xf16>
    // CHECK: return [[AFFINERESHAPE]] : tensor<1x1x4096x2560xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeMultiplyWhenRankIsChanged
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<16x32x32xf16>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<16x32x32xf16>)
func.func @NotSwapAffineReshapeMultiplyWhenRankIsChanged(%arg0: tensor<16x32x32xf16>, %arg1: tensor<16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 32, 32]} : tensor<16x32x32xf16> -> tensor<1x16x32x32xf16>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 32, 32]} : tensor<16x32x32xf16> -> tensor<1x16x32x32xf16>

    %2 = IE.Multiply(%0, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    return %2 : tensor<1x16x32x32xf16>

    // CHECK: [[RESHAPE1:%.*]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 32, 32]} :
    // CHECK-SAME:          tensor<16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK: [[RESHAPE2:%.*]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 32, 32]} :
    // CHECK-SAME:          tensor<16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK: [[MULTIPLY:%.*]] = IE.Multiply([[RESHAPE1]], [[RESHAPE2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

    // CHECK: return [[MULTIPLY]] : tensor<1x16x32x32xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeMultiplySplitAndMergeAxes
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<1x16x32x32xf16>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<1x16x32x32xf16>)
func.func @SwapAffineReshapeMultiplySplitAndMergeAxes(%arg0: tensor<1x16x32x32xf16>, %arg1: tensor<1x16x32x32xf16>) -> tensor<1x512x1x32xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 512, 1, 32]} : tensor<1x16x32x32xf16> -> tensor<1x512x1x32xf16>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 512, 1, 32]} : tensor<1x16x32x32xf16> -> tensor<1x512x1x32xf16>

    %2 = IE.Multiply(%0, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x512x1x32xf16>, tensor<1x512x1x32xf16> -> tensor<1x512x1x32xf16>
    return %2 : tensor<1x512x1x32xf16>

    // CHECK: [[MULTIPLY:%.*]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[MULTIPLY]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 512, 1, 32]}
    // CHECK-SAME:          tensor<1x16x32x32xf16> -> tensor<1x512x1x32xf16>
    // CHECK: return [[AFFINERESHAPE]] : tensor<1x512x1x32xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeMultiplyWithScalarConst
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x512x768x1xf16>
func.func @SwapAffineReshapeMultiplyWithScalarConst(%arg0: tensor<1x512x768x1xf16>) -> tensor<1x512x1x768xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 512, 1, 768]} : tensor<1x512x768x1xf16> -> tensor<1x512x1x768xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x1x768xf16>, tensor<1x1x1x1xf16> -> tensor<1x512x1x768xf16>
    return %1 : tensor<1x512x1x768xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[MULTIPLY:%.*]] = IE.Multiply([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x768x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x512x768x1xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[MULTIPLY]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 512, 1, 768]} : tensor<1x512x768x1xf16> -> tensor<1x512x1x768xf16>
    // CHECK: return [[AFFINERESHAPE]] : tensor<1x512x1x768xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeMultiplyWithConstBroadCastOnOneDim
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x512x768x1xf16>
func.func @SwapAffineReshapeMultiplyWithConstBroadCastOnOneDim(%arg0: tensor<1x512x768x1xf16>) -> tensor<1x512x1x768xf16> {
    %cst = const.Declare tensor<1x1x1x768xf16> = dense<2.0> : tensor<1x1x1x768xf16>
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 512, 1, 768]} : tensor<1x512x768x1xf16> -> tensor<1x512x1x768xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x1x768xf16>, tensor<1x1x1x768xf16> -> tensor<1x512x1x768xf16>
    return %1 : tensor<1x512x1x768xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<1x1x768x1xf16> = dense<2.000000e+00> : tensor<1x1x1x768xf16>, [#const.Reshape<[1, 1, 768, 1]>]
    // CHECK: [[MULTIPLY:%.*]] = IE.Multiply([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x768x1xf16>, tensor<1x1x768x1xf16> -> tensor<1x512x768x1xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[MULTIPLY]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 512, 1, 768]} : tensor<1x512x768x1xf16> -> tensor<1x512x1x768xf16>
    // CHECK: return [[AFFINERESHAPE]] : tensor<1x512x1x768xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeMultiplyWithMultiNonBroadCastDims
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<2x256x4x192xf16>
func.func @NotSwapAffineReshapeMultiplyWithMultiNonBroadCastDims(%arg0: tensor<2x256x4x192xf16>) -> tensor<1x512x4x192xf16> {
    %cst = const.Declare tensor<1x1x4x192xf16> = dense<2.0> : tensor<1x1x4x192xf16>
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0, 1], [2], [3]], shape_value = [1, 512, 4, 192]} : tensor<2x256x4x192xf16> -> tensor<1x512x4x192xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x192xf16>, tensor<1x1x4x192xf16> -> tensor<1x512x4x192xf16>
    return %1 : tensor<1x512x4x192xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<1x1x4x192xf16> = dense<2.000000e+00> : tensor<1x1x4x192xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0, 1], [2], [3]], shape_value = [1, 512, 4, 192]} : tensor<2x256x4x192xf16> -> tensor<1x512x4x192xf16>
    // CHECK: [[MULTIPLY:%.*]] = IE.Multiply([[AFFINERESHAPE]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x192xf16>, tensor<1x1x4x192xf16> -> tensor<1x512x4x192xf16>
    // CHECK: return [[MULTIPLY]] : tensor<1x512x4x192xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeMultiplyWithNonBroadCastDimChanged
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<2x4x1x1xf16>
func.func @NotSwapAffineReshapeMultiplyWithNonBroadCastDimChanged(%arg0: tensor<2x4x1x1xf16>) -> tensor<1x2x2x2xf16> {
    %cst = const.Declare tensor<1x1x2x1xf16> = dense<2.0> : tensor<1x1x2x1xf16>
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1, 2], [2, 3]], shape_value = [1, 2, 2, 2]} : tensor<2x4x1x1xf16> -> tensor<1x2x2x2xf16>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x2x2xf16>, tensor<1x1x2x1xf16> -> tensor<1x2x2x2xf16>
    return %1 : tensor<1x2x2x2xf16>

    // CHECK: [[CST:%.*]] = const.Declare tensor<1x1x2x1xf16> = dense<2.000000e+00> : tensor<1x1x2x1xf16>
    // CHECK: [[AFFINERESHAPE:%.*]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [2, 3]], shape_value = [1, 2, 2, 2]} : tensor<2x4x1x1xf16> -> tensor<1x2x2x2xf16>
    // CHECK: [[MULTIPLY:%.*]] = IE.Multiply([[AFFINERESHAPE]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x2x2xf16>, tensor<1x1x2x1xf16> -> tensor<1x2x2x2xf16>
    // CHECK: return [[MULTIPLY]] : tensor<1x2x2x2xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeMultiplyForReshapeDifferentInTypes
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<1x16x32x32xf16>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<1x1x512x1xf16>)
func.func @NotSwapAffineReshapeMultiplyForReshapeDifferentInTypes(%arg0: tensor<1x16x32x32xf16>, %arg1: tensor<1x1x512x1xf16>) -> tensor<1x512x1x32xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 512, 1, 32]} : tensor<1x16x32x32xf16> -> tensor<1x512x1x32xf16>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 512, 1, 1]} : tensor<1x1x512x1xf16> -> tensor<1x512x1x1xf16>

    %2 = IE.Multiply(%0, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x512x1x32xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x1x32xf16>
    return %2 : tensor<1x512x1x32xf16>

    // CHECK: [[RESHAPE1:%.*]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 512, 1, 32]} :
    // CHECK-SAME:          tensor<1x16x32x32xf16> -> tensor<1x512x1x32xf16>
    // CHECK: [[RESHAPE2:%.*]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 512, 1, 1]} :
    // CHECK-SAME:          tensor<1x1x512x1xf16> -> tensor<1x512x1x1xf16>
    // CHECK: [[MULTIPLY:%.*]] = IE.Multiply([[RESHAPE1]], [[RESHAPE2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x1x32xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x1x32xf16>

    // CHECK: return [[MULTIPLY]] : tensor<1x512x1x32xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeMultiplyForReshapeDifferentDimMapping
// CHECK-SAME:    ([[INPUT0:%.*]]: tensor<16x1x1x32xf16>,
// CHECK-SAME:     [[INPUT1:%.*]]: tensor<16x1x1x32xf16>)
func.func @NotSwapAffineReshapeMultiplyForReshapeDifferentDimMapping(%arg0: tensor<16x1x1x32xf16>, %arg1: tensor<16x1x1x32xf16>) -> tensor<1x16x1x32xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 16, 1, 32]} : tensor<16x1x1x32xf16> -> tensor<1x16x1x32xf16>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16, 1, 32]} : tensor<16x1x1x32xf16> -> tensor<1x16x1x32xf16>

    %2 = IE.Multiply(%0, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x16x1x32xf16>, tensor<1x16x1x32xf16> -> tensor<1x16x1x32xf16>
    return %2 : tensor<1x16x1x32xf16>

    // CHECK: [[RESHAPE1:%.*]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 16, 1, 32]} :
    // CHECK-SAME:          tensor<16x1x1x32xf16> -> tensor<1x16x1x32xf16>
    // CHECK: [[RESHAPE2:%.*]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 16, 1, 32]} :
    // CHECK-SAME:          tensor<16x1x1x32xf16> -> tensor<1x16x1x32xf16>
    // CHECK: [[MULTIPLY:%.*]] = IE.Multiply([[RESHAPE1]], [[RESHAPE2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x32xf16>, tensor<1x16x1x32xf16> -> tensor<1x16x1x32xf16>

    // CHECK: return [[MULTIPLY]] : tensor<1x16x1x32xf16>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: SwapConcatAffineReshape
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x3x40x40x81xf16, {order = #NCDHW}>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x3x40x40x2xf16, {order = #NCDHW}>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<1x3x40x40x2xf16, {order = #NCDHW}>,
// CHECK-SAME:     [[INPUT3:%arg[0-9]]]: tensor<1x4800x85xf16>)
func.func @SwapConcatAffineReshape(%input0: tensor<1x3x40x40x81xf16, {order = #NCDHW}>,
                                   %input1: tensor<1x3x40x40x2xf16, {order = #NCDHW}>,
                                   %input2: tensor<1x3x40x40x2xf16, {order = #NCDHW}>,
                                   %input3: tensor<1x4800x85xf16>)
    -> (tensor<1x9600x85xf16>) {
    %concat0 = IE.Concat(%input0, %input1, %input2) {
        static_offsets = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 81],
            [0, 0, 0, 0, 83]
        ]
    } : tensor<1x3x40x40x81xf16, {order = #NCDHW}>,
        tensor<1x3x40x40x2xf16, {order = #NCDHW}>,
        tensor<1x3x40x40x2xf16, {order = #NCDHW}>
            -> tensor<1x3x40x40x85xf16, {order = #NCDHW}>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0], [1], [1], [1], [2]],
            shape_value = [1, 4800, 85]
        } : tensor<1x3x40x40x85xf16, {order = #NCDHW}> -> tensor<1x4800x85xf16>
    %concat1 = IE.Concat(%reshape, %input3) {
        static_offsets = [
            [0, 0, 0],
            [0, 4800, 0]
        ]
    } : tensor<1x4800x85xf16>,
        tensor<1x4800x85xf16>
            -> tensor<1x9600x85xf16>
    return %concat1 : tensor<1x9600x85xf16>

    // CHECK:                [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME:            {dim_mapping =
    // CHECK-SAME{LITERAL}:    [[0], [1], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:    shape_value = [1, 4800, 81]} :
    // CHECK-SAME{LITERAL}:    tensor<1x3x40x40x81xf16, {order = #NCDHW}> -> tensor<1x4800x81xf16>
    // CHECK:                [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME:            {dim_mapping =
    // CHECK-SAME{LITERAL}:    [[0], [1], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:    shape_value = [1, 4800, 2]} :
    // CHECK-SAME{LITERAL}:    tensor<1x3x40x40x2xf16, {order = #NCDHW}> -> tensor<1x4800x2xf16>
    // CHECK:                [[RESHAPE2:%.+]] = IE.AffineReshape([[INPUT2]])
    // CHECK-SAME:            {dim_mapping =
    // CHECK-SAME{LITERAL}:    [[0], [1], [1], [1], [2]],
    // CHECK-SAME{LITERAL}:    shape_value = [1, 4800, 2]} :
    // CHECK-SAME{LITERAL}:    tensor<1x3x40x40x2xf16, {order = #NCDHW}> -> tensor<1x4800x2xf16>
    // CHECK:                [[CONCAT0:%.+]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]], [[RESHAPE2]]) {
    // CHECK-SAME{LITERAL}:    static_offsets = [[0, 0, 0], [0, 0, 81], [0, 0, 83]]} :
    // CHECK-SAME:             tensor<1x4800x81xf16>, tensor<1x4800x2xf16>, tensor<1x4800x2xf16> -> tensor<1x4800x85xf16>
    // CHECK:                [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[INPUT3]]) {
    // CHECK-SAME{LITERAL}:    static_offsets = [[0, 0, 0], [0, 4800, 0]]} :
    // CHECK-SAME:             tensor<1x4800x85xf16>, tensor<1x4800x85xf16> -> tensor<1x9600x85xf16>
    // CHECK:                return [[CONCAT1]] : tensor<1x9600x85xf16>
}

// -----

// CHECK-LABEL: ConcatAndCollapseReshape
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x1x6x7xf32>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x1x6x7xf32>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<12x7xf32>)
func.func @ConcatAndCollapseReshape(%arg0: tensor<1x1x6x7xf32>,
                                    %arg1: tensor<1x1x6x7xf32>,
                                    %arg2: tensor<12x7xf32>) -> tensor<12x14xf32> {
    %concat0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 6, 0]
        ]
    } : tensor<1x1x6x7xf32>,
        tensor<1x1x6x7xf32>
            -> tensor<1x1x12x7xf32>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0], [0], [0], [1]],
            shape_value = [12, 7]
        } : tensor<1x1x12x7xf32> -> tensor<12x7xf32>
    %concat1 = IE.Concat(%reshape, %arg2) {
        static_offsets = [
            [0, 0],
            [0, 7]
        ]
    } : tensor<12x7xf32>,
        tensor<12x7xf32>
            -> tensor<12x14xf32>
    return %concat1 : tensor<12x14xf32>

    // CHECK:             [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [6, 7]} : tensor<1x1x6x7xf32> -> tensor<6x7xf32>
    // CHECK:             [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [6, 7]} : tensor<1x1x6x7xf32> -> tensor<6x7xf32>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0], [6, 0]]} : tensor<6x7xf32>, tensor<6x7xf32> -> tensor<12x7xf32>
    // CHECK:             [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[INPUT2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0], [0, 7]]} : tensor<12x7xf32>, tensor<12x7xf32> -> tensor<12x14xf32>
    // CHECK:             return [[CONCAT1]] : tensor<12x14xf32>
}

// -----

// CHECK-LABEL: ConcatAndExtendReshape
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<6x7xf32>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<6x7xf32>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<1x12x7x1xf32>)
func.func @ConcatAndExtendReshape(%arg0: tensor<6x7xf32>,
                                  %arg1: tensor<6x7xf32>,
                                  %arg2: tensor<1x12x7x1xf32>) -> tensor<1x12x14x1xf32> {
    %concat0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0],
            [6, 0]
        ]
    } : tensor<6x7xf32>,
        tensor<6x7xf32>
            -> tensor<12x7xf32>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0, 1], [2, 3]],
            shape_value = [1, 12, 7, 1]
        } : tensor<12x7xf32> -> tensor<1x12x7x1xf32>
    %concat1 = IE.Concat(%reshape, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 7, 0]
        ]
    } : tensor<1x12x7x1xf32>,
        tensor<1x12x7x1xf32>
            -> tensor<1x12x14x1xf32>
    return %concat1 : tensor<1x12x14x1xf32>

    // CHECK:             [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 6, 7, 1]} : tensor<6x7xf32> -> tensor<1x6x7x1xf32>
    // CHECK:             [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 6, 7, 1]} : tensor<6x7xf32> -> tensor<1x6x7x1xf32>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 6, 0, 0]]} : tensor<1x6x7x1xf32>, tensor<1x6x7x1xf32> -> tensor<1x12x7x1xf32>
    // CHECK:             [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[INPUT2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]} : tensor<1x12x7x1xf32>, tensor<1x12x7x1xf32> -> tensor<1x12x14x1xf32>
    // CHECK:             return [[CONCAT1]] : tensor<1x12x14x1xf32>
}

// -----

// CHECK-LABEL: ConcatAndShiftReshape
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x1x6x7xf32>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x1x6x7xf32>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<12x7x1x1xf32>)
func.func @ConcatAndShiftReshape(%arg0: tensor<1x1x6x7xf32>,
                                 %arg1: tensor<1x1x6x7xf32>,
                                 %arg2: tensor<12x7x1x1xf32>) -> tensor<12x14x1x1xf32> {
    %concat0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 6, 0]
        ]
    } : tensor<1x1x6x7xf32>,
        tensor<1x1x6x7xf32>
            -> tensor<1x1x12x7xf32>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0], [0], [0], [1, 2, 3]],
            shape_value = [12, 7, 1, 1]
        } : tensor<1x1x12x7xf32> -> tensor<12x7x1x1xf32>
    %concat1 = IE.Concat(%reshape, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 7, 0, 0]
        ]
    } : tensor<12x7x1x1xf32>,
        tensor<12x7x1x1xf32>
            -> tensor<12x14x1x1xf32>
    return %concat1 : tensor<12x14x1x1xf32>

    // CHECK:             [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [6, 7, 1, 1]} : tensor<1x1x6x7xf32> -> tensor<6x7x1x1xf32>
    // CHECK:             [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [6, 7, 1, 1]} : tensor<1x1x6x7xf32> -> tensor<6x7x1x1xf32>
    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [6, 0, 0, 0]]} : tensor<6x7x1x1xf32>, tensor<6x7x1x1xf32> -> tensor<12x7x1x1xf32>
    // CHECK:             [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[INPUT2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 7, 0, 0]]} : tensor<12x7x1x1xf32>, tensor<12x7x1x1xf32> -> tensor<12x14x1x1xf32>
    // CHECK:             return [[CONCAT1]] : tensor<12x14x1x1xf32>
}

// -----

// CHECK-LABEL: NoSwapConcatAxisSplit
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x16x42xf32>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x16x42xf32>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<1x16x12x7xf32>)
func.func @NoSwapConcatAxisSplit(%arg0: tensor<1x16x42xf32>,
                                 %arg1: tensor<1x16x42xf32>,
                                 %arg2: tensor<1x16x12x7xf32>) -> tensor<1x16x12x14xf32> {
    %concat0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0],
            [0, 0, 42]
        ]
    } : tensor<1x16x42xf32>,
        tensor<1x16x42xf32>
            -> tensor<1x16x84xf32>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0], [1], [2, 3]],
            shape_value = [1, 16, 12, 7]
        } : tensor<1x16x84xf32> -> tensor<1x16x12x7xf32>
    %concat1 = IE.Concat(%reshape, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 0, 7]
        ]
    } : tensor<1x16x12x7xf32>,
        tensor<1x16x12x7xf32>
            -> tensor<1x16x12x14xf32>
    return %concat1 : tensor<1x16x12x14xf32>

    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0], [0, 0, 42]]} : tensor<1x16x42xf32>, tensor<1x16x42xf32> -> tensor<1x16x84xf32>
    // CHECK:             [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 16, 12, 7]} : tensor<1x16x84xf32> -> tensor<1x16x12x7xf32>
    // CHECK:             [[CONCAT1:%.+]] = IE.Concat([[RESHAPE]], [[INPUT2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 7]]} : tensor<1x16x12x7xf32>, tensor<1x16x12x7xf32> -> tensor<1x16x12x14xf32>
    // CHECK:             return [[CONCAT1]] : tensor<1x16x12x14xf32>
}

// -----

// CHECK-LABEL: NoSwapConcatAxisMerged
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x16x6x7xf32>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x16x6x7xf32>,
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<1x16x84xf32>)
func.func @NoSwapConcatAxisMerged(%arg0: tensor<1x16x6x7xf32>,
                                  %arg1: tensor<1x16x6x7xf32>,
                                  %arg2: tensor<1x16x84xf32>) -> tensor<1x16x168xf32> {
    %concat0 = IE.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 6, 0]
        ]
    } : tensor<1x16x6x7xf32>,
        tensor<1x16x6x7xf32>
            -> tensor<1x16x12x7xf32>
    %reshape = IE.AffineReshape(%concat0) {
            dim_mapping = [[0], [1], [2], [2]],
            shape_value = [1, 16, 84]
        } : tensor<1x16x12x7xf32> -> tensor<1x16x84xf32>
    %concat1 = IE.Concat(%reshape, %arg2) {
        static_offsets = [
            [0, 0, 0],
            [0, 0, 84]
        ]
    } : tensor<1x16x84xf32>,
        tensor<1x16x84xf32>
            -> tensor<1x16x168xf32>
    return %concat1 : tensor<1x16x168xf32>

    // CHECK:             [[CONCAT0:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]} : tensor<1x16x6x7xf32>, tensor<1x16x6x7xf32> -> tensor<1x16x12x7xf32>
    // CHECK:             [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 16, 84]} : tensor<1x16x12x7xf32> -> tensor<1x16x84xf32>
    // CHECK:             [[CONCAT1:%.+]] = IE.Concat([[RESHAPE]], [[INPUT2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0], [0, 0, 84]]} : tensor<1x16x84xf32>, tensor<1x16x84xf32> -> tensor<1x16x168xf32>
    // CHECK:             return [[CONCAT1:%.+]] : tensor<1x16x168xf32>
}

// -----

// CHECK-LABEL: @SwapSqueezeAffineReshapeConcat
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x1x1x256xf16>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x1x1x256xf16>
func.func @SwapSqueezeAffineReshapeConcat(%arg0: tensor<1x1x1x256xf16>, %arg1: tensor<1x1x1x256xf16>) ->
                        tensor<2x256xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x1x256xf16> -> tensor<1x256xf16>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x1x256xf16> -> tensor<1x256xf16>

    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0], [1, 0]]} : tensor<1x256xf16>, tensor<1x256xf16> -> tensor<2x256xf16>

    return %2 : tensor<2x256xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<2x1x1x256xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [2, 256]} : tensor<2x1x1x256xf16> -> tensor<2x256xf16>
    // CHECK:       return [[RESHAPE]] : tensor<2x256xf16>
}

// CHECK-LABEL: @SwapBackConcatSqueezeAffineReshapeConcat
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x1x1x256xf16>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x1x1x256xf16>
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<2x256xf16>
func.func @SwapBackConcatSqueezeAffineReshapeConcat(%arg0: tensor<1x1x1x256xf16>, %arg1: tensor<1x1x1x256xf16>, %arg2: tensor<2x256xf16>) ->
                        tensor<2x512xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<2x1x1x256xf16>

    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [2, 256]} : tensor<2x1x1x256xf16> -> tensor<2x256xf16>

    %2 = IE.Concat(%1, %arg2) {static_offsets = [[0, 0], [0, 256]]} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x512xf16>

    return %2 : tensor<2x512xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x1x256xf16> -> tensor<1x256xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x1x256xf16> -> tensor<1x256xf16>
    // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[RESHAPE_0]], [[RESHAPE_1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0], [1, 0]]} : tensor<1x256xf16>, tensor<1x256xf16> -> tensor<2x256xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CONCAT_0]], [[INPUT2]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0], [0, 256]]} : tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x512xf16>
    // CHECK:       return [[CONCAT_1]] : tensor<2x512xf16>
}

// -----

// CHECK-LABEL: @SwapUnsqueezeAffineReshapeConcat
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x256xf16>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x256xf16>
func.func @SwapUnsqueezeAffineReshapeConcat(%arg0: tensor<1x256xf16>, %arg1: tensor<1x256xf16>) ->
                        tensor<1x2x1x256xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x256xf16>

    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x256xf16>

    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x2x1x256xf16>

    return %2 : tensor<1x2x1x256xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0], [1, 0]]} : tensor<1x256xf16>, tensor<1x256xf16> -> tensor<2x256xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 1, 256]} : tensor<2x256xf16> -> tensor<1x2x1x256xf16>
    // CHECK:       return [[RESHAPE]] : tensor<1x2x1x256xf16>
}

// CHECK-LABEL: @SwapBackConcatUnsqueezeAffineReshapeConcat
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x256xf16>
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x256xf16>
// CHECK-SAME:     [[INPUT2:%arg[0-9]]]: tensor<1x2x1x256xf16>
func.func @SwapBackConcatUnsqueezeAffineReshapeConcat(%arg0: tensor<1x256xf16>, %arg1: tensor<1x256xf16>, %arg2: tensor<1x2x1x256xf16>) ->
                        tensor<1x2x1x512xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0], [1, 0]]} : tensor<1x256xf16>, tensor<1x256xf16> -> tensor<2x256xf16>

    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 2, 1, 256]} : tensor<2x256xf16> -> tensor<1x2x1x256xf16>

    %2 = IE.Concat(%1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 256]]} : tensor<1x2x1x256xf16>, tensor<1x2x1x256xf16> -> tensor<1x2x1x512xf16>

    return %2 : tensor<1x2x1x512xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x256xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 256]} : tensor<1x256xf16> -> tensor<1x1x1x256xf16>
    // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[RESHAPE_0]], [[RESHAPE_1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x2x1x256xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CONCAT_0]], [[INPUT2]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 256]]} : tensor<1x2x1x256xf16>, tensor<1x2x1x256xf16> -> tensor<1x2x1x512xf16>
    // CHECK:       return [[CONCAT_1]] : tensor<1x2x1x512xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeSlice
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x3x144x1439xf16>
func.func @SwapAffineReshapeSlice(%arg0: tensor<1x3x144x1439xf16>) -> (tensor<1x3x9x16x478xf16>, tensor<1x3x9x16x956xf16>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 1439]} : tensor<1x3x144x1439xf16> -> tensor<1x3x9x16x1439xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0, 4] [1, 3, 9, 16, 478] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x16x478xf16>
    %2 = IE.Slice %0 [0, 0, 0, 0, 482] [1, 3, 9, 16, 956] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x16x956xf16>

    return %1, %2 : tensor<1x3x9x16x478xf16>, tensor<1x3x9x16x956xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[INPUT0]]
    // CHECK-SAME:      [0, 0, 0, 4] [1, 3, 144, 478] : tensor<1x3x144x1439xf16> to tensor<1x3x144x478xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 478]} : tensor<1x3x144x478xf16> -> tensor<1x3x9x16x478xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[INPUT0]]
    // CHECK-SAME:      [0, 0, 0, 482] [1, 3, 144, 956] : tensor<1x3x144x1439xf16> to tensor<1x3x144x956xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 956]} : tensor<1x3x144x956xf16> -> tensor<1x3x9x16x956xf16>
    // CHECK:       return [[RESHAPE_0]], [[RESHAPE_1]] : tensor<1x3x9x16x478xf16>, tensor<1x3x9x16x956xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeSliceOnMultiAxes
// CHECK-SAME:     [[INPUT0:%arg[0-9]]]: tensor<1x3x144x1439xf16>
func.func @NotSwapAffineReshapeSliceOnMultiAxes(%arg0: tensor<1x3x144x1439xf16>) -> (tensor<1x3x9x16x478xf16>, tensor<1x3x9x8x1439xf16>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 1439]} : tensor<1x3x144x1439xf16> -> tensor<1x3x9x16x1439xf16>
    %1 = IE.Slice %0 [0, 0, 0, 0, 4] [1, 3, 9, 16, 478] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x16x478xf16>
    %2 = IE.Slice %0 [0, 0, 0, 0, 0] [1, 3, 9, 8, 1439] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x8x1439xf16>

    return %1, %2 : tensor<1x3x9x16x478xf16>, tensor<1x3x9x8x1439xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 3, 9, 16, 1439]} : tensor<1x3x144x1439xf16> -> tensor<1x3x9x16x1439xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[RESHAPE]]
    // CHECK-SAME:      [0, 0, 0, 0, 4] [1, 3, 9, 16, 478] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x16x478xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[RESHAPE]]
    // CHECK-SAME:      [0, 0, 0, 0, 0] [1, 3, 9, 8, 1439] : tensor<1x3x9x16x1439xf16> to tensor<1x3x9x8x1439xf16>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x3x9x16x478xf16>, tensor<1x3x9x8x1439xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d2, d0)>

// CHECK-LABEL: @SwapWithConstAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x16x128x1xf16>
func.func @SwapWithConstAdd(%arg0: tensor<1x16x128x1xf16>) -> tensor<1x16x1x128xf16> {
    %cst = const.Declare tensor<1x16x1x128xf16> = dense<1.0> : tensor<1x16x1x128xf16>
    %0 = IE.Transpose(%arg0) {order_value = #map1} : tensor<1x16x128x1xf16> -> tensor<16x1x128x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [3, 3]], shape_value = [1, 16, 1, 128]} : tensor<16x1x128x1xf16> -> tensor<1x16x1x128xf16>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x1x128xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x1x128xf16>

    return %2 : tensor<1x16x1x128xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x16x128x1xf16> = dense<1.000000e+00> : tensor<1x16x1x128xf16>, [#const.Reshape<[16, 1, 128, 1]>, #const.Transpose<#map>]
    // CHECK:           [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x128x1xf16>, tensor<1x16x128x1xf16> -> tensor<1x16x128x1xf16>
    // CHECK:           [[TRANSPOSE:%.+]] = IE.Transpose([[ADD]]) {order_value = #map1} : tensor<1x16x128x1xf16> -> tensor<16x1x128x1xf16>
    // CHECK:           [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [2], [3, 3]], shape_value = [1, 16, 1, 128]} : tensor<16x1x128x1xf16> -> tensor<1x16x1x128xf16>
    // CHECK:           return [[AFFINE_RESHAPE]] : tensor<1x16x1x128xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0031202451855528589>
!qElemType1 = !quant.uniform<u8:f16, 0.0062404903711057178>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @SwapWithQuantizeAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x144x2304x1xf16>
func.func @SwapWithQuantizeAdd(%arg0: tensor<1x144x2304x1xf16>) -> tensor<1x1x2304x144x!qElemType> {
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} : tensor<1x144x2304x1xf16> -> tensor<2304x144x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 2304, 144]} : tensor<2304x144x1x1xf16> -> tensor<1x1x2304x144xf16>
    %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x2304x144xf16>, tensor<1x1x2304x144xf16> -> tensor<1x1x2304x144x!qElemType1>
    %3 = IE.QuantizeCast(%2) {dstElemType = !qElemType} : tensor<1x1x2304x144x!qElemType1> -> tensor<1x1x2304x144x!qElemType>

    return %3 : tensor<1x1x2304x144x!qElemType>

    // CHECK:   [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x144x2304x1xf16>, tensor<1x144x2304x1xf16> -> tensor<1x144x2304x1x!qElemType1>
    // CHECK:   [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType} : tensor<1x144x2304x1x!qElemType1> -> tensor<1x144x2304x1x!qElemType>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[QUANTIZE_CAST]]) {order_value = #map} : tensor<1x144x2304x1x!qElemType> -> tensor<2304x144x1x1x!qElemType>
    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 2304, 144]} : tensor<2304x144x1x1x!qElemType> -> tensor<1x1x2304x144x!qElemType>
    // CHECK:   return [[AFFINE_RESHAPE]] : tensor<1x1x2304x144x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0037092456630631989:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithDeQuantizeAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x512x2x1x!qElemType>
func.func @SwapWithDeQuantizeAdd(%arg0: tensor<1x512x2x1x!qElemType>) -> tensor<1x2x256x2xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x512x2x1x!qElemType> -> tensor<1x2x1x512x!qElemType>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 256, 2]} : tensor<1x2x1x512x!qElemType> -> tensor<1x2x256x2x!qElemType>
    %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x256x2x!qElemType>, tensor<1x2x256x2x!qElemType> -> tensor<1x2x256x2xf16>

    return %2 : tensor<1x2x256x2xf16>

    // CHECK:   [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x2x1x!qElemType>, tensor<1x512x2x1x!qElemType> -> tensor<1x512x2x1xf16>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[ADD]]) {order_value = #NHWC} : tensor<1x512x2x1xf16> -> tensor<1x2x1x512xf16>
    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 256, 2]} : tensor<1x2x1x512xf16> -> tensor<1x2x256x2xf16>
    // CHECK:   return [[AFFINE_RESHAPE]] : tensor<1x2x256x2xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapAffineReshapeWithAddToCloseToConv
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x128x256x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1024x128x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x1024x1024x1xf16, {order = #NHWC}>
func.func @SwapAffineReshapeWithAddToCloseToConv(
            %arg0: tensor<1x128x256x4xf16, {order = #NHWC}>,
            %arg1: tensor<1024x128x1x1xf16, {order = #NHWC}>,
            %arg2: tensor<1x1024x1024x1xf16, {order = #NHWC}>) -> tensor<1x1024x1024x1xf16, {order = #NHWC}> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<1024x128x1x1xf16, {order = #NHWC}> -> tensor<1x1024x256x4xf16, {order = #NHWC}>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 1024, 1024, 1]} : tensor<1x1024x256x4xf16, {order = #NHWC}> -> tensor<1x1024x1024x1xf16, {order = #NHWC}>
    %2 = IE.Add(%1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1024x1xf16, {order = #NHWC}>, tensor<1x1024x1024x1xf16, {order = #NHWC}> -> tensor<1x1024x1024x1xf16, {order = #NHWC}>

    return %2 : tensor<1x1024x1024x1xf16, {order = #NHWC}>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[INPUT_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<1024x128x1x1xf16, {order = #NHWC}> -> tensor<1x1024x256x4xf16, {order = #NHWC}>
    // CHECK: [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 1024, 256, 4]} inputs([[INPUT_2]] : tensor<1x1024x1024x1xf16, {order = #NHWC}>) -> tensor<1x1024x256x4xf16, {order = #NHWC}>
    // CHECK: [[ADD:%.+]] = IE.Add([[CONV]], [[SHAPECAST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x256x4xf16, {order = #NHWC}>, tensor<1x1024x256x4xf16, {order = #NHWC}> -> tensor<1x1024x256x4xf16, {order = #NHWC}>
    // CHECK: [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[ADD]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 1024, 1024, 1]}
    // CHECK-SAME:          tensor<1x1024x256x4xf16, {order = #NHWC}> -> tensor<1x1024x1024x1xf16, {order = #NHWC}>

    // CHECK: return [[AFFINERESHAPE]] : tensor<1x1024x1024x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapAffineReshapeWithAddToCloseToAdd
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x3584x256x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x3584x256x4xf16, {order = #NHWC}>
func.func @SwapAffineReshapeWithAddToCloseToAdd(
            %arg0: tensor<1x3584x256x4xf16, {order = #NHWC}>,
            %arg1: tensor<1x3584x256x4xf16, {order = #NHWC}>) -> tensor<1x3584x1024x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x3584x1024x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x3584x1024x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3584x256x4xf16, {order = #NHWC}>, tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 3584, 1024, 1]} : tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x1024x1xf16, {order = #NHWC}>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584x1024x1xf16, {order = #NHWC}>, tensor<1x3584x1024x1xf16, {order = #NHWC}> -> tensor<1x3584x1024x1xf16, {order = #NHWC}>

    return %2 : tensor<1x3584x1024x1xf16, {order = #NHWC}>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x3584x256x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x3584x1024x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[1, 3584, 256, 4]>]
    // CHECK: [[ADD_0:%.+]] = IE.Add([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3584x256x4xf16, {order = #NHWC}>, tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    // CHECK: [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584x256x4xf16, {order = #NHWC}>, tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    // CHECK: [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[ADD_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 3584, 1024, 1]} :
    // CHECK-SAME:          tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x1024x1xf16, {order = #NHWC}>

    // CHECK: return [[AFFINERESHAPE]] : tensor<1x3584x1024x1xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.032288775724523211:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapAffineReshapeWithMixedPrecisionAdd
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x256x1x768xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x256x768xf16>
func.func @SwapAffineReshapeWithMixedPrecisionAdd(
            %arg0: tensor<1x256x1x768xf16>,
            %arg1: tensor<1x1x256x768xf16>) -> tensor<1x1x256x768x!qElemType> {
    %cst = const.Declare tensor<1x256x1x768xf16> = dense<1.0> : tensor<1x256x1x768xf16>
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x768xf16>, tensor<1x256x1x768xf16> -> tensor<1x256x1x768xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 256, 768]} : tensor<1x256x1x768xf16> -> tensor<1x1x256x768xf16>
    %2 = IE.Add(%arg1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x768xf16>, tensor<1x1x256x768xf16> -> tensor<1x1x256x768x!qElemType>

    return %2 : tensor<1x1x256x768x!qElemType>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x256x1x768xf16> = dense<1.000000e+00> : tensor<1x256x1x768xf16>
    // CHECK: [[ADD_0:%.+]] = IE.Add([[INPUT_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x768xf16>, tensor<1x256x1x768xf16> -> tensor<1x256x1x768xf16>
    // CHECK: [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 256, 1, 768]} inputs([[INPUT_1]] : tensor<1x1x256x768xf16>) -> tensor<1x256x1x768xf16>
    // CHECK: [[ADD_1:%.+]] = IE.Add([[SHAPECAST]], [[ADD_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x768xf16>, tensor<1x256x1x768xf16> -> tensor<1x256x1x768x!qElemType>
    // CHECK: [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[ADD_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 256, 768]} :
    // CHECK-SAME:          tensor<1x256x1x768x!qElemType> -> tensor<1x1x256x768x!qElemType>

    // CHECK: return [[AFFINERESHAPE]] : tensor<1x1x256x768x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotSwapAffineReshapeWithPerAxisQuantizedAdd
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x32x1x4xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x16x2x4xf16>
func.func @NotSwapAffineReshapeWithPerAxisQuantizedAdd(
            %arg0: tensor<1x32x1x4xf16>,
            %arg1: tensor<1x16x2x4xf16>) -> tensor<1x16x2x4x!qElemType> {
    %cst = const.Declare tensor<1x32x1x4xf16> = dense<1.0> : tensor<1x32x1x4xf16>
    %0 = IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x4xf16>, tensor<1x32x1x4xf16> -> tensor<1x32x1x4xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 16, 2, 4]} : tensor<1x32x1x4xf16> -> tensor<1x16x2x4xf16>
    %2 = IE.Add(%arg1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x2x4xf16>, tensor<1x16x2x4xf16> -> tensor<1x16x2x4x!qElemType>

    return %2 : tensor<1x16x2x4x!qElemType>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x32x1x4xf16> = dense<1.000000e+00> : tensor<1x32x1x4xf16>
    // CHECK: [[ADD_0:%.+]] = IE.Add([[INPUT_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x4xf16>, tensor<1x32x1x4xf16> -> tensor<1x32x1x4xf16>
    // CHECK: [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[ADD_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 16, 2, 4]} :
    // CHECK-SAME:          tensor<1x32x1x4xf16> -> tensor<1x16x2x4xf16>
    // CHECK: [[ADD_1:%.+]] = IE.Add([[INPUT_1]], [[AFFINERESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x2x4xf16>, tensor<1x16x2x4xf16> -> tensor<1x16x2x4x!qElemType>

    // CHECK: return [[ADD_1]] : tensor<1x16x2x4x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithAddWhenBothInputsAreAffineReshapes
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x512x256x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1x512x256x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<1x512x256x4xf16, {order = #NHWC}>
func.func @SwapWithAddWhenBothInputsAreAffineReshapes(
            %arg0: tensor<1x512x256x4xf16, {order = #NHWC}>,
            %arg1: tensor<1x512x256x4xf16, {order = #NHWC}>,
            %arg2: tensor<1x512x256x4xf16, {order = #NHWC}>) -> tensor<1x512x1024x1xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x256x4xf16, {order = #NHWC}>, tensor<1x512x256x4xf16, {order = #NHWC}> -> tensor<1x512x256x4xf16, {order = #NHWC}>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 512, 1024, 1]} : tensor<1x512x256x4xf16, {order = #NHWC}> -> tensor<1x512x1024x1xf16, {order = #NHWC}>
    %2 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 512, 1024, 1]} : tensor<1x512x256x4xf16, {order = #NHWC}> -> tensor<1x512x1024x1xf16, {order = #NHWC}>
    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x1024x1xf16, {order = #NHWC}>, tensor<1x512x1024x1xf16, {order = #NHWC}> -> tensor<1x512x1024x1xf16, {order = #NHWC}>

    return %3 : tensor<1x512x1024x1xf16, {order = #NHWC}>

    // CHECK: [[ADD_0:%.+]] = IE.Add([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x256x4xf16, {order = #NHWC}>, tensor<1x512x256x4xf16, {order = #NHWC}> -> tensor<1x512x256x4xf16, {order = #NHWC}>
    // CHECK: [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[INPUT_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x256x4xf16, {order = #NHWC}>, tensor<1x512x256x4xf16, {order = #NHWC}> -> tensor<1x512x256x4xf16, {order = #NHWC}>
    // CHECK: [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[ADD_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 512, 1024, 1]} :
    // CHECK-SAME:          tensor<1x512x256x4xf16, {order = #NHWC}> -> tensor<1x512x1024x1xf16, {order = #NHWC}>

    // CHECK: return [[AFFINERESHAPE]] : tensor<1x512x1024x1xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithAddWhenBothInputsAreAffineReshapesForDynamicQuantCase
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<3584x128x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<3584x1x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_3:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_4:%arg[0-9]]]: tensor<3584x128x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_5:%arg[0-9]]]: tensor<3584x1x1x1xf16, {order = #NHWC}>
func.func @SwapWithAddWhenBothInputsAreAffineReshapesForDynamicQuantCase(
            %arg0: tensor<1x128x256x4xf16, {order = #NHWC}>,
            %arg1: tensor<3584x128x1x1x!qElemType, {order = #NHWC}>,
            %arg2: tensor<3584x1x1x1xf16, {order = #NHWC}>,
            %arg3: tensor<1x128x256x4xf16, {order = #NHWC}>,
            %arg4: tensor<3584x128x1x1x!qElemType, {order = #NHWC}>,
            %arg5: tensor<3584x1x1x1xf16, {order = #NHWC}>) -> tensor<1x3584x1024x1xf16, {order = #NHWC}> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<3584x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    %1 = IE.GroupConvolution(%0, %arg2) {dilations = [1, 1], groups = 3584 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3584x256x4xf16, {order = #NHWC}>, tensor<3584x1x1x1xf16, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 3584, 1024, 1]} : tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x1024x1xf16, {order = #NHWC}>

    %3 = IE.Convolution(%arg3, %arg4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<3584x128x1x1x!quant.uniform<i4:f16, 1.000000e+00>, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    %4 = IE.GroupConvolution(%3, %arg5) {dilations = [1, 1], groups = 3584 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3584x256x4xf16, {order = #NHWC}>, tensor<3584x1x1x1xf16, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 3584, 1024, 1]} : tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x1024x1xf16, {order = #NHWC}>

    %6 = IE.Add(%2, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584x1024x1xf16, {order = #NHWC}>, tensor<1x3584x1024x1xf16, {order = #NHWC}> -> tensor<1x3584x1024x1xf16, {order = #NHWC}>

    return %6 : tensor<1x3584x1024x1xf16, {order = #NHWC}>

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT_0]], [[INPUT_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<3584x128x1x1x!qElemType, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    // CHECK: [[GROUPCONV0:%.+]] = IE.GroupConvolution([[CONV0]], [[INPUT_2]]) {dilations = [1, 1], groups = 3584 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3584x256x4xf16, {order = #NHWC}>, tensor<3584x1x1x1xf16, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT_3]], [[INPUT_4]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<3584x128x1x1x!qElemType, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    // CHECK: [[GROUPCONV1:%.+]] = IE.GroupConvolution([[CONV1]], [[INPUT_5]]) {dilations = [1, 1], groups = 3584 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3584x256x4xf16, {order = #NHWC}>, tensor<3584x1x1x1xf16, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    // CHECK: [[ADD:%.+]] = IE.Add([[GROUPCONV0]], [[GROUPCONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3584x256x4xf16, {order = #NHWC}>, tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
    // CHECK: [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[ADD]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 3584, 1024, 1]}
    // CHECK-SAME:          tensor<1x3584x256x4xf16, {order = #NHWC}> -> tensor<1x3584x1024x1xf16, {order = #NHWC}>

    // CHECK: return [[AFFINERESHAPE]] : tensor<1x3584x1024x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SwapWithMvn
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x1x1x1280xf16>
func.func @SwapWithMvn(%arg0: tensor<1x1x1x1280xf16>) -> tensor<1x1x1280x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1, 1280, 1]} : tensor<1x1x1x1280xf16> -> tensor<1x1x1280x1xf16>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1x1280x1xf16> -> tensor<1x1x1280x1xf16>

    return %1: tensor<1x1x1280x1xf16>

    // CHECK:        [[MVN:%.+]] = IE.MVN([[INPUT]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1x1x1280xf16> -> tensor<1x1x1x1280xf16>
    // CHECK:        [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1, 1280, 1]} : tensor<1x1x1x1280xf16> -> tensor<1x1x1280x1xf16>

    // CHECK:        return [[AFFINERESHAPE]] : tensor<1x1x1280x1xf16>
}

// -----

// CHECK-LABEL: @SwapWithMvnWhenChannelChangedAndAcrossChannelFalse
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x1280x1x1xf16>
func.func @SwapWithMvnWhenChannelChangedAndAcrossChannelFalse(%arg0: tensor<1x1280x1x1xf16>) -> tensor<1x1x1280x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 1280, 1]} : tensor<1x1280x1x1xf16> -> tensor<1x1x1280x1xf16>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1x1280x1xf16> -> tensor<1x1x1280x1xf16>

    return %1: tensor<1x1x1280x1xf16>

    // CHECK:        [[MVN:%.+]] = IE.MVN([[INPUT]]) {across_channels = true, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1280x1x1xf16> -> tensor<1x1280x1x1xf16>
    // CHECK:        [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 1280, 1]} : tensor<1x1280x1x1xf16> -> tensor<1x1x1280x1xf16>

    // CHECK:        return [[AFFINERESHAPE]] : tensor<1x1x1280x1xf16>
}

// -----

// CHECK-LABEL: @NotSwapWithMvnWhenChannelChangedAndAcrossChannelFalse
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x2560x1x1xf16>
func.func @NotSwapWithMvnWhenChannelChangedAndAcrossChannelFalse(%arg0: tensor<1x2560x1x1xf16>) -> tensor<1x2x1280x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 1280, 1]} : tensor<1x2560x1x1xf16> -> tensor<1x2x1280x1xf16>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x2x1280x1xf16> -> tensor<1x2x1280x1xf16>

    return %1: tensor<1x2x1280x1xf16>

    // CHECK:        [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 1280, 1]} : tensor<1x2560x1x1xf16> -> tensor<1x2x1280x1xf16>
    // CHECK:        [[MVN:%.+]] = IE.MVN([[AFFINERESHAPE]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x2x1280x1xf16> -> tensor<1x2x1280x1xf16>

    // CHECK:        return [[MVN]] : tensor<1x2x1280x1xf16>
}

// -----

// CHECK-LABEL: @SwapWithMvnWhenBatchNotChangedAndAcrossChannelFalse
// CHECK-SAME:     [[INPUT:%.+]]: tensor<2x121x1x32xf16>
func.func @SwapWithMvnWhenBatchNotChangedAndAcrossChannelFalse(%arg0: tensor<2x121x1x32xf16>) -> tensor<2x1x121x32xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [2, 1, 121, 32]} : tensor<2x121x1x32xf16> -> tensor<2x1x121x32xf16>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.765625E-4 : f64, normalize_variance = true} : tensor<2x1x121x32xf16> -> tensor<2x1x121x32xf16>

    return %1: tensor<2x1x121x32xf16>

    // CHECK:        [[MVN:%.+]] = IE.MVN([[INPUT]]) {across_channels = true, eps = 9.765625E-4 : f64, normalize_variance = true} : tensor<2x121x1x32xf16> -> tensor<2x121x1x32xf16>
    // CHECK:        [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [2, 1, 121, 32]} : tensor<2x121x1x32xf16> -> tensor<2x1x121x32xf16>

    // CHECK:        return [[AFFINERESHAPE]] : tensor<2x1x121x32xf16>
}

// -----

// CHECK-LABEL: @NotSwapWithMvnWhenBatchChangedAndAcrossChannelFalse
// CHECK-SAME:     [[INPUT:%.+]]: tensor<2x1x121x32xf16>
func.func @NotSwapWithMvnWhenBatchChangedAndAcrossChannelFalse(%arg0: tensor<2x1x121x32xf16>) -> tensor<242x1x32x1xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [242, 1, 32, 1]} : tensor<2x1x121x32xf16> -> tensor<242x1x32x1xf16>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.765625E-4 : f64, normalize_variance = true} : tensor<242x1x32x1xf16> -> tensor<242x1x32x1xf16>

    return %1: tensor<242x1x32x1xf16>

    // CHECK:        [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [242, 1, 32, 1]} : tensor<2x1x121x32xf16> -> tensor<242x1x32x1xf16>
    // CHECK:        [[MVN:%.+]] = IE.MVN([[AFFINERESHAPE]]) {across_channels = false, eps = 9.765625E-4 : f64, normalize_variance = true} : tensor<242x1x32x1xf16> -> tensor<242x1x32x1xf16>

    // CHECK:        return [[MVN]] : tensor<242x1x32x1xf16>
}

// -----

// CHECK-LABEL: @SwapWithMvnWhenBatchChangedAndAcrossChannelFalse
// CHECK-SAME:     [[INPUT:%.+]]: tensor<4x2x1x320xf16>
func.func @SwapWithMvnWhenBatchChangedAndAcrossChannelFalse(%arg0: tensor<4x2x1x320xf16>) -> tensor<2x4x1x320xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [1], [2], [3]], shape_value = [2, 4, 1, 320]} : tensor<4x2x1x320xf16> -> tensor<2x4x1x320xf16>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.765625E-4 : f64, normalize_variance = true} : tensor<2x4x1x320xf16> -> tensor<2x4x1x320xf16>

    return %1: tensor<2x4x1x320xf16>

    // CHECK:        [[MVN:%.+]] = IE.MVN([[INPUT]]) {across_channels = false, eps = 9.765625E-4 : f64, normalize_variance = true} : tensor<4x2x1x320xf16> -> tensor<4x2x1x320xf16>
    // CHECK:        [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[MVN]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [1], [2], [3]], shape_value = [2, 4, 1, 320]} : tensor<4x2x1x320xf16> -> tensor<2x4x1x320xf16>

    // CHECK:        return [[AFFINERESHAPE]] : tensor<2x4x1x320xf16>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConvert
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x1280x1x1xf32>
func.func @SwapAffineReshapeConvert(%arg0 : tensor<1x1280x1x1xf32>) -> tensor<1x1x1x1280xf16> {
    %1 = IE.AffineReshape(%arg0) {
        dim_mapping = [[0, 1, 2], [3], [3], [3]],
        shape_value = [1, 1, 1, 1280]
    } : tensor<1x1280x1x1xf32> -> tensor<1x1x1x1280xf32>

    %2 = IE.Convert(%1) {
        dstElemType = f16
    } : tensor<1x1x1x1280xf32> -> tensor<1x1x1x1280xf16>

    %3 = IE.Add(%2, %2) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x1x1x1280xf16>, tensor<1x1x1x1280xf16> -> tensor<1x1x1x1280xf16>

    return %3 : tensor<1x1x1x1280xf16>

    // CHECK:      [[LAYER:%.+]] = IE.Convert([[INPUT]]) {
    // CHECK-SAME:     dstElemType = f16
    // CHECK-SAME: } : tensor<1x1280x1x1xf32> -> tensor<1x1280x1x1xf16>

    // CHECK:      [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[LAYER]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0, 1, 2], [3], [3], [3]],
    // CHECK-SAME:     shape_value = [1, 1, 1, 1280]
    // CHECK-SAME: } : tensor<1x1280x1x1xf16> -> tensor<1x1x1x1280xf16>

    // CHECK:      [[ADD:%.+]] = IE.Add([[AFFINERESHAPE]], [[AFFINERESHAPE]]) {
    // CHECK-SAME:     auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    // CHECK-SAME: } : tensor<1x1x1x1280xf16>, tensor<1x1x1x1280xf16> -> tensor<1x1x1x1280xf16>

    // CHECK:      return [[ADD]] : tensor<1x1x1x1280xf16>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeConvert
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x1280x1x1xf16>
func.func @NotSwapAffineReshapeConvert(%arg0 : tensor<1x1280x1x1xf16>) -> tensor<1x1x1x1280xf32> {
    %1 = IE.AffineReshape(%arg0) {
        dim_mapping = [[0, 1, 2], [3], [3], [3]],
        shape_value = [1, 1, 1, 1280]
    } : tensor<1x1280x1x1xf16> -> tensor<1x1x1x1280xf16>

    %2 = IE.Convert(%1) {
        dstElemType = f32
    } : tensor<1x1x1x1280xf16> -> tensor<1x1x1x1280xf32>

    return %2 : tensor<1x1x1x1280xf32>

    // CHECK:      [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0, 1, 2], [3], [3], [3]],
    // CHECK-SAME:     shape_value = [1, 1, 1, 1280]
    // CHECK-SAME: } : tensor<1x1280x1x1xf16> -> tensor<1x1x1x1280xf16>

    // CHECK:      [[LAYER:%.+]] = IE.Convert([[AFFINERESHAPE]]) {
    // CHECK-SAME:     dstElemType = f32
    // CHECK-SAME: } : tensor<1x1x1x1280xf16> -> tensor<1x1x1x1280xf32>

    // CHECK:      return [[LAYER]] : tensor<1x1x1x1280xf32>
}

// -----

// CHECK-LABEL: @SwapAffineReshapeConvertSubByte
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x1280x1x1xi4>
func.func @SwapAffineReshapeConvertSubByte(%arg0 : tensor<1x1280x1x1xi4>) -> tensor<1x1x1x1280xi1> {
    %1 = IE.AffineReshape(%arg0) {
        dim_mapping = [[0, 1, 2], [3], [3], [3]],
        shape_value = [1, 1, 1, 1280]
    } : tensor<1x1280x1x1xi4> -> tensor<1x1x1x1280xi4>

    %2 = IE.Convert(%1) {
        dstElemType = i1
    } : tensor<1x1x1x1280xi4> -> tensor<1x1x1x1280xi1>

    return %2 : tensor<1x1x1x1280xi1>

    // CHECK:      [[LAYER:%.+]] = IE.Convert([[INPUT]]) {
    // CHECK-SAME:     dstElemType = i1
    // CHECK-SAME: } : tensor<1x1280x1x1xi4> -> tensor<1x1280x1x1xi1>

    // CHECK:      [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[LAYER]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0, 1, 2], [3], [3], [3]],
    // CHECK-SAME:     shape_value = [1, 1, 1, 1280]
    // CHECK-SAME: } : tensor<1x1280x1x1xi1> -> tensor<1x1x1x1280xi1>

    // CHECK:      return [[AFFINERESHAPE]] : tensor<1x1x1x1280xi1>
}

// -----

// CHECK-LABEL: @NotSwapAffineReshapeConvertSubByte
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x1280x1x1xi1>
func.func @NotSwapAffineReshapeConvertSubByte(%arg0 : tensor<1x1280x1x1xi1>) -> tensor<1x1x1x1280xi4> {
    %1 = IE.AffineReshape(%arg0) {
        dim_mapping = [[0, 1, 2], [3], [3], [3]],
        shape_value = [1, 1, 1, 1280]
    } : tensor<1x1280x1x1xi1> -> tensor<1x1x1x1280xi1>

    %2 = IE.Convert(%1) {
        dstElemType = i4
    } : tensor<1x1x1x1280xi1> -> tensor<1x1x1x1280xi4>

    return %2 : tensor<1x1x1x1280xi4>

    // CHECK:      [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME{LITERAL}: dim_mapping = [[0, 1, 2], [3], [3], [3]],
    // CHECK-SAME:     shape_value = [1, 1, 1, 1280]
    // CHECK-SAME: } : tensor<1x1280x1x1xi1> -> tensor<1x1x1x1280xi1>

    // CHECK:      [[LAYER:%.+]] = IE.Convert([[AFFINERESHAPE]]) {
    // CHECK-SAME:     dstElemType = i4
    // CHECK-SAME: } : tensor<1x1x1x1280xi1> -> tensor<1x1x1x1280xi4>

    // CHECK:      return [[LAYER]] : tensor<1x1x1x1280xi4>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL: @NotPropogateAffineReshapePermuteCastDueToCMXConcat
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x32x16x4xf16, {order = #NHWC}>
func.func @NotPropogateAffineReshapePermuteCastDueToCMXConcat(%arg0 : tensor<1x32x16x4xf16, {order = #NHWC}>) -> tensor<1x2x64x64xf16> {
    %cst_0 = const.Declare tensor<64x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x32x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]

    %13 = IE.Convolution(%arg0, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            static_scale = 0.135327876 : f32,
            strides = [1, 1]
        } : tensor<1x32x16x4xf16, {order = #NHWC}>, tensor<64x32x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x4xf16, {order = #NHWC}>
    %14 = IE.AffineReshape(%13) {
            dim_mapping = [[0], [1], [2], [2, 3]],
            shape_value = [1, 64, 64, 1]
        } : tensor<1x64x16x4xf16, {order = #NHWC}> -> tensor<1x64x64x1xf16, {order = #NHWC}>
    %15 = IE.PermuteCast(%14) {
            dst_order = #NCHW,
            mem_perm = #map1
        } : tensor<1x64x64x1xf16, {order = #NHWC}> -> tensor<64x64x1x1xf16>

    %20 = IE.Convolution(%arg0, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            static_scale = 0.135327876 : f32,
            strides = [1, 1]
        } : tensor<1x32x16x4xf16, {order = #NHWC}>, tensor<64x32x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x4xf16, {order = #NHWC}>
    %21 = IE.AffineReshape(%20) {
            dim_mapping = [[0], [1], [2], [2, 3]],
            shape_value = [1, 64, 64, 1]
        } : tensor<1x64x16x4xf16, {order = #NHWC}> -> tensor<1x64x64x1xf16, {order = #NHWC}>
    %22 = IE.PermuteCast(%21) {
            dst_order = #NCHW,
            mem_perm = #map1
        } : tensor<1x64x64x1xf16, {order = #NHWC}> -> tensor<64x64x1x1xf16>

    %30 = IE.AffineReshape(%15) {
            dim_mapping = [[0, 1, 2], [3], [3], [3]],
            shape_value = [1, 1, 64, 64]} : tensor<64x64x1x1xf16> -> tensor<1x1x64x64xf16>
    %31 = IE.AffineReshape(%22) {
            dim_mapping = [[0, 1, 2], [3], [3], [3]],
            shape_value = [1, 1, 64, 64]} : tensor<64x64x1x1xf16> -> tensor<1x1x64x64xf16>

    %33 = IE.Concat(%30, %31) {
            static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
        } : tensor<1x1x64x64xf16>, tensor<1x1x64x64xf16> -> tensor<1x2x64x64xf16>

    return %33 : tensor<1x2x64x64xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<64x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]
    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0],
    // CHECK-SAME:              static_scale = 0.135327876 : f32,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              } : tensor<1x32x16x4xf16, {order = #NHWC}>, tensor<64x32x1x1xf16, {order = #NHWC}> -> tensor<1x64x16x4xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 64, 1]} : tensor<1x64x16x4xf16, {order = #NHWC}> -> tensor<1x64x64x1xf16, {order = #NHWC}>

    // CHECK: [[PERMUTECAST1:%.+]] = IE.PermuteCast([[RESHAPE1]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #map} : tensor<1x64x64x1xf16, {order = #NHWC}> -> tensor<64x64x1x1xf16>

    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0],
    // CHECK-SAME:              static_scale = 0.135327876 : f32,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              } : tensor<1x32x16x4xf16, {order = #NHWC}>, tensor<64x32x1x1xf16, {order = #NHWC}> -> tensor<1x64x16x4xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[CONV1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 64, 1]} : tensor<1x64x16x4xf16, {order = #NHWC}> -> tensor<1x64x64x1xf16, {order = #NHWC}>

    // CHECK: [[PERMUTECAST2:%.+]] = IE.PermuteCast([[RESHAPE2]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #map} : tensor<1x64x64x1xf16, {order = #NHWC}> -> tensor<64x64x1x1xf16>

    // CHECK: [[RESHAPE3:%.+]] = IE.AffineReshape([[PERMUTECAST1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 64, 64]} : tensor<64x64x1x1xf16> -> tensor<1x1x64x64xf16>

    // CHECK: [[RESHAPE4:%.+]] = IE.AffineReshape([[PERMUTECAST2]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 64, 64]} : tensor<64x64x1x1xf16> -> tensor<1x1x64x64xf16>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[RESHAPE3]], [[RESHAPE4]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x64x64xf16>, tensor<1x1x64x64xf16> -> tensor<1x2x64x64xf16>

    // CHECK:      return [[CONCAT]] : tensor<1x2x64x64xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL: @PropogateAffineReshapePermuteCastThroughConcat
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x64x256x32xf16, {order = #NHWC}>
func.func @PropogateAffineReshapePermuteCastThroughConcat(%arg0 : tensor<1x64x256x32xf16, {order = #NHWC}>) -> tensor<1x3x8192x64xf16> {
    %cst_0 = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]

    %13 = IE.Convolution(%arg0, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            static_scale = 0.135327876 : f32,
            strides = [1, 1]
        } : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x256x32xf16, {order = #NHWC}>
    %14 = IE.AffineReshape(%13) {
            dim_mapping = [[0], [1], [2], [2, 3]],
            shape_value = [1, 64, 8192, 1]
        } : tensor<1x64x256x32xf16, {order = #NHWC}> -> tensor<1x64x8192x1xf16, {order = #NHWC}>
    %15 = IE.PermuteCast(%14) {
            dst_order = #NCHW,
            mem_perm = #map1
        } : tensor<1x64x8192x1xf16, {order = #NHWC}> -> tensor<8192x64x1x1xf16>

    %20 = IE.Convolution(%arg0, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            static_scale = 0.135327876 : f32,
            strides = [1, 1]
        } : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x256x32xf16, {order = #NHWC}>
    %21 = IE.AffineReshape(%20) {
            dim_mapping = [[0], [1], [2], [2, 3]],
            shape_value = [1, 64, 8192, 1]
        } : tensor<1x64x256x32xf16, {order = #NHWC}> -> tensor<1x64x8192x1xf16, {order = #NHWC}>
    %22 = IE.PermuteCast(%21) {
            dst_order = #NCHW,
            mem_perm = #map1
        } : tensor<1x64x8192x1xf16, {order = #NHWC}> -> tensor<8192x64x1x1xf16>

    %27 = IE.Convolution(%arg0, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            static_scale = 0.135327876 : f32,
            strides = [1, 1]
        } : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x256x32xf16, {order = #NHWC}>
    %28 = IE.AffineReshape(%27) {
            dim_mapping = [[0], [1], [2], [2, 3]],
            shape_value = [1, 64, 8192, 1]
        } : tensor<1x64x256x32xf16, {order = #NHWC}> -> tensor<1x64x8192x1xf16, {order = #NHWC}>
    %29 = IE.PermuteCast(%28) {
            dst_order = #NCHW,
            mem_perm = #map1
        } : tensor<1x64x8192x1xf16, {order = #NHWC}> -> tensor<8192x64x1x1xf16>

    %30 = IE.AffineReshape(%15) {
            dim_mapping = [[0, 1, 2], [3], [3], [3]],
            shape_value = [1, 1, 8192, 64]} : tensor<8192x64x1x1xf16> -> tensor<1x1x8192x64xf16>
    %31 = IE.AffineReshape(%22) {
            dim_mapping = [[0, 1, 2], [3], [3], [3]],
            shape_value = [1, 1, 8192, 64]} : tensor<8192x64x1x1xf16> -> tensor<1x1x8192x64xf16>
    %32 = IE.AffineReshape(%29) {
            dim_mapping = [[0, 1, 2], [3], [3], [3]],
            shape_value = [1, 1, 8192, 64]} : tensor<8192x64x1x1xf16> -> tensor<1x1x8192x64xf16>

    %33 = IE.Concat(%30, %31, %32) {
            static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]]
        } : tensor<1x1x8192x64xf16>, tensor<1x1x8192x64xf16>, tensor<1x1x8192x64xf16> -> tensor<1x3x8192x64xf16>

    return %33 : tensor<1x3x8192x64xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0],
    // CHECK-SAME:              static_scale = 0.135327876 : f32,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              } : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x256x32xf16, {order = #NHWC}>

    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0],
    // CHECK-SAME:              static_scale = 0.135327876 : f32,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              } : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x256x32xf16, {order = #NHWC}>

    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0],
    // CHECK-SAME:              static_scale = 0.135327876 : f32,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              } : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x256x32xf16, {order = #NHWC}>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]])
    // CHECK-SAME{LITERAL}:     {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<1x64x256x32xf16, {order = #NHWC}> -> tensor<3x64x256x32xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [3, 64, 8192, 1]} : tensor<3x64x256x32xf16, {order = #NHWC}> -> tensor<3x64x8192x1xf16, {order = #NHWC}>

    // CHECK: [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[RESHAPE1]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #NCHW} : tensor<3x64x8192x1xf16, {order = #NHWC}> -> tensor<3x8192x1x64xf16>

    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[PERMUTE_CAST]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 3, 8192, 64]} : tensor<3x8192x1x64xf16> -> tensor<1x3x8192x64xf16>

    // CHECK:      return [[RESHAPE2]] : tensor<1x3x8192x64xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL: @NotPropogateAffineReshapePermuteCastDueToConcatNotOnC
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x64x256x32xf16, {order = #NHWC}>
func.func @NotPropogateAffineReshapePermuteCastDueToConcatNotOnC(%arg0 : tensor<1x64x256x32xf16, {order = #NHWC}>) -> tensor<1x1x8192x128xf16> {
    %cst_0 = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]

    %13 = IE.Convolution(%arg0, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            static_scale = 0.135327876 : f32,
            strides = [1, 1]
        } : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x256x32xf16, {order = #NHWC}>
    %14 = IE.AffineReshape(%13) {
            dim_mapping = [[0], [1], [2], [2, 3]],
            shape_value = [1, 64, 8192, 1]
        } : tensor<1x64x256x32xf16, {order = #NHWC}> -> tensor<1x64x8192x1xf16, {order = #NHWC}>
    %15 = IE.PermuteCast(%14) {
            dst_order = #NCHW,
            mem_perm = #map1
        } : tensor<1x64x8192x1xf16, {order = #NHWC}> -> tensor<8192x64x1x1xf16>

    %20 = IE.Convolution(%arg0, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            static_scale = 0.135327876 : f32,
            strides = [1, 1]
        } : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x256x32xf16, {order = #NHWC}>
    %21 = IE.AffineReshape(%20) {
            dim_mapping = [[0], [1], [2], [2, 3]],
            shape_value = [1, 64, 8192, 1]
        } : tensor<1x64x256x32xf16, {order = #NHWC}> -> tensor<1x64x8192x1xf16, {order = #NHWC}>
    %22 = IE.PermuteCast(%21) {
            dst_order = #NCHW,
            mem_perm = #map1
        } : tensor<1x64x8192x1xf16, {order = #NHWC}> -> tensor<8192x64x1x1xf16>

    %30 = IE.AffineReshape(%15) {
            dim_mapping = [[0, 1, 2], [3], [3], [3]],
            shape_value = [1, 1, 8192, 64]} : tensor<8192x64x1x1xf16> -> tensor<1x1x8192x64xf16>
    %31 = IE.AffineReshape(%22) {
            dim_mapping = [[0, 1, 2], [3], [3], [3]],
            shape_value = [1, 1, 8192, 64]} : tensor<8192x64x1x1xf16> -> tensor<1x1x8192x64xf16>

    %32 = IE.Concat(%30, %31) {
            static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64]]
        } : tensor<1x1x8192x64xf16>, tensor<1x1x8192x64xf16> -> tensor<1x1x8192x128xf16>

    return %32 : tensor<1x1x8192x128xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16, {order = #NHWC}>, [#const.Reorder<#NHWC>]

    // CHECK: [[CONV0:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0],
    // CHECK-SAME:              static_scale = 0.135327876 : f32,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              } : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x256x32xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[CONV0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 8192, 1]} : tensor<1x64x256x32xf16, {order = #NHWC}> -> tensor<1x64x8192x1xf16, {order = #NHWC}>

    // CHECK: [[PERMUTE_CAST0:%.+]] = IE.PermuteCast([[RESHAPE0]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #map} : tensor<1x64x8192x1xf16, {order = #NHWC}> -> tensor<8192x64x1x1xf16>

    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0],
    // CHECK-SAME:              static_scale = 0.135327876 : f32,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              } : tensor<1x64x256x32xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x256x32xf16, {order = #NHWC}>

    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[CONV1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 8192, 1]} : tensor<1x64x256x32xf16, {order = #NHWC}> -> tensor<1x64x8192x1xf16, {order = #NHWC}>

    // CHECK: [[PERMUTE_CAST1:%.+]] = IE.PermuteCast([[RESHAPE1]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #map} : tensor<1x64x8192x1xf16, {order = #NHWC}> -> tensor<8192x64x1x1xf16>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[PERMUTE_CAST0]], [[PERMUTE_CAST1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<8192x64x1x1xf16>, tensor<8192x64x1x1xf16> -> tensor<8192x128x1x1xf16>

    // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 8192, 128]} : tensor<8192x128x1x1xf16> -> tensor<1x1x8192x128xf16>

    // CHECK:      return [[RESHAPE_OUT]] : tensor<1x1x8192x128xf16>
}

// -----

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: NotPropogateAffineReshapePermuteCastDueToNon4DReshape
// CHECK-SAME:    ([[INPUT0:%arg[0-9]]]: tensor<1x12x7x1xf32, {order = #NHWC}>,
// CHECK-SAME:     [[INPUT1:%arg[0-9]]]: tensor<1x12x7x1xf32, {order = #NHWC}>
func.func @NotPropogateAffineReshapePermuteCastDueToNon4DReshape(%arg0: tensor<1x12x7x1xf32, {order = #NHWC}>,
                                    %arg1: tensor<1x12x7x1xf32, {order = #NHWC}>) -> tensor<12x14xf32> {
    %in1 = IE.AffineReshape(%arg0) {
            dim_mapping = [[0], [1], [2], [2, 3]],
            shape_value = [1, 1, 12, 7]
        } : tensor<1x12x7x1xf32, {order = #NHWC}> -> tensor<1x1x12x7xf32, {order = #NHWC}>

    %in2 = IE.AffineReshape(%arg1) {
            dim_mapping = [[0], [1], [2], [2, 3]],
            shape_value = [1, 1, 12, 7]
        } : tensor<1x12x7x1xf32, {order = #NHWC}> -> tensor<1x1x12x7xf32, {order = #NHWC}>

    %permute_cast1 = IE.PermuteCast(%in1) {
            dst_order = #NCHW,
            mem_perm = #map1
        } : tensor<1x1x12x7xf32, {order = #NHWC}> -> tensor<1x1x12x7xf32>

    %permute_cast2 = IE.PermuteCast(%in2) {
            dst_order = #NCHW,
            mem_perm = #map1
        } : tensor<1x1x12x7xf32, {order = #NHWC}> -> tensor<1x1x12x7xf32>

    %reshape1 = IE.AffineReshape(%permute_cast1) {
            dim_mapping = [[0], [0], [0], [1]],
            shape_value = [12, 7]
        } : tensor<1x1x12x7xf32> -> tensor<12x7xf32>

    %reshape2 = IE.AffineReshape(%permute_cast2) {
            dim_mapping = [[0], [0], [0], [1]],
            shape_value = [12, 7]
        } : tensor<1x1x12x7xf32> -> tensor<12x7xf32>

    %concat1 = IE.Concat(%reshape1, %reshape2) {
        static_offsets = [
            [0, 0],
            [0, 7]
        ]
    } : tensor<12x7xf32>,
        tensor<12x7xf32>
            -> tensor<12x14xf32>
    return %concat1 : tensor<12x14xf32>

    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 1, 12, 7]} : tensor<1x12x7x1xf32, {order = #NHWC}> -> tensor<1x1x12x7xf32, {order = #NHWC}>

    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[INPUT1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 1, 12, 7]} : tensor<1x12x7x1xf32, {order = #NHWC}> -> tensor<1x1x12x7xf32, {order = #NHWC}>

    // CHECK: [[PERMUTE_CAST0:%.+]] = IE.PermuteCast([[RESHAPE0]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x12x7xf32, {order = #NHWC}> -> tensor<1x1x12x7xf32>

    // CHECK: [[PERMUTE_CAST1:%.+]] = IE.PermuteCast([[RESHAPE1]])
    // CHECK-SAME{LITERAL}:     {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x12x7xf32, {order = #NHWC}> -> tensor<1x1x12x7xf32>

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[PERMUTE_CAST0]], [[PERMUTE_CAST1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 7]]} : tensor<1x1x12x7xf32>, tensor<1x1x12x7xf32> -> tensor<1x1x12x14xf32>

    // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [12, 14]} : tensor<1x1x12x14xf32> -> tensor<12x14xf32>

    // CHECK:      return [[RESHAPE_OUT]] : tensor<12x14xf32>
}
