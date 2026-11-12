//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-split-concat-to-affinereshape %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @ConvertSplitConcatToAffineReshape
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x2x120x96x49xf16>
func.func @ConvertSplitConcatToAffineReshape(%arg0: tensor<1x2x120x96x49xf16>) -> tensor<1x1x240x96x49xf16> {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x120x96x49xf16> -> tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16>
    %1 = IE.Concat(%0#0, %0#1) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 120, 0, 0]]} : tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16> -> tensor<1x1x240x96x49xf16>

    return %1 : tensor<1x1x240x96x49xf16>

   // CHECK:       [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
   // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [2], [3], [4]], shape_value = [1, 1, 240, 96, 49]} : tensor<1x2x120x96x49xf16> -> tensor<1x1x240x96x49xf16>
   // CHECK:       return [[AFFINERESHAPE]] : tensor<1x1x240x96x49xf16>
}
