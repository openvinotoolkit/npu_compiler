//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-d2s-and-scale-shift %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: func.func @SwapD2SAndScaleShift
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x48x224x224xf16>)
func.func @SwapD2SAndScaleShift(%arg0: tensor<1x48x224x224xf16>) -> tensor<1x3x896x896xf16> {
    %cst = const.Declare tensor<1x3x1x1xf16> = dense<5.000000e-01> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 3 : i64>]
    %d2s = IE.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x48x224x224xf16> -> tensor<1x3x896x896xf16>
    %scale_shift = IE.ScaleShift(%d2s, %cst) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x3x896x896xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x896x896xf16>

    return %scale_shift : tensor<1x3x896x896xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x48x1x1xf16> = dense<5.000000e-01> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 3 : i64>, #const.Broadcast<1 : i64, 48 : i64>]
    //CHECK: [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[INPUT]], [[CST]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x48x224x224xf16>, tensor<1x48x1x1xf16> -> tensor<1x48x224x224xf16>
    //CHECK: [[D2S:%.+]] = IE.DepthToSpace([[SCALE_SHIFT]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x48x224x224xf16> -> tensor<1x3x896x896xf16>
    //CHECK: return [[D2S]] : tensor<1x3x896x896xf16>
}
