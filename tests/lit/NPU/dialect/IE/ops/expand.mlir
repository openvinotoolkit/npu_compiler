//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
func.func @FoldExpand(%arg0: tensor<1x8x4x4xf16>) -> tensor<1x8x4x4xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x8x4x4xf16> -> tensor<1x8x4x4xf16>
    return %0 : tensor<1x8x4x4xf16>

    // CHECK: return %arg0 : tensor<1x8x4x4xf16>
}

func.func @FoldSliceExpandWithSameZeroOffsets(%arg0: tensor<1x80x56x56xf16>) -> tensor<1x80x56x56xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 72, 56, 56] : tensor<1x80x56x56xf16> to tensor<1x72x56x56xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x72x56x56xf16> -> tensor<1x80x56x56xf16>
    return %1 : tensor<1x80x56x56xf16>

    // CHECK: return %arg0 : tensor<1x80x56x56xf16>
}

func.func @FoldSliceExpandWithSameNoneZeroOffsets(%arg0: tensor<1x16x320x320xf16>) -> tensor<1x16x320x320xf16> {
    %0 = IE.Slice %arg0 [0, 3, 0, 0] [1, 1, 320, 320] : tensor<1x16x320x320xf16> to tensor<1x1x320x320xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 3, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x1x320x320xf16> -> tensor<1x16x320x320xf16>
    return %1 : tensor<1x16x320x320xf16>

    // CHECK: return %arg0 : tensor<1x16x320x320xf16>
}

func.func @NotFoldSliceExpandWithDiffOffsets(%arg0: tensor<1x16x320x320xf16>) -> tensor<1x16x320x320xf16> {
    %0 = IE.Slice %arg0 [0, 3, 0, 0] [1, 1, 320, 320] : tensor<1x16x320x320xf16> to tensor<1x1x320x320xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x320x320xf16> -> tensor<1x16x320x320xf16>
    return %1 : tensor<1x16x320x320xf16>

    // CHECK:   [[SLICE:%.+]] = IE.Slice %arg0 [0, 3, 0, 0] [1, 1, 320, 320] : tensor<1x16x320x320xf16> to tensor<1x1x320x320xf16>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[SLICE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x320x320xf16> -> tensor<1x16x320x320xf16>
    // CHECK:   return [[EXPAND]] : tensor<1x16x320x320xf16>
}

func.func @ConstantFolding() -> tensor<1x11x12x12xf16> {
    %cst = const.Declare tensor<1x5x10x11xf16> = dense<1.0> : tensor<1x5x10x11xf16>
    %0 = IE.Expand(%cst) {pads_begin = [0, 3, 0, 1], pads_end = [0, 3, 2, 0]} : tensor<1x5x10x11xf16> -> tensor<1x11x12x12xf16>
    return %0 : tensor<1x11x12x12xf16>

    // CHECK-DAG:       %[[CST:.*]] = const.Declare tensor<1x11x12x12xf16> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x5x10x11xf16>, [#const.PadWithZero<[0, 3, 0, 1], [0, 3, 2, 0]>]
    // CHECK:       return %[[CST]] : tensor<1x11x12x12xf16>
}

func.func @KeepExpandForEltwiseToBenefitFromAdjustInputShape(%arg0: tensor<1x80x56x56xf16>) -> tensor<1x80x56x56xf16> {
    %cst = const.Declare tensor<1x80x56x56xf16> = dense<1.0> : tensor<1x80x56x56xf16>
    %slice = IE.Slice %arg0 [0, 0, 0, 0] [1, 72, 56, 56] : tensor<1x80x56x56xf16> to tensor<1x72x56x56xf16>
    %expand = IE.Expand(%slice) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x72x56x56xf16> -> tensor<1x80x56x56xf16>
    %add = IE.Add(%expand, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x80x56x56xf16>, tensor<1x80x56x56xf16> -> tensor<1x80x56x56xf16>
    return %add : tensor<1x80x56x56xf16>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<1x80x56x56xf16> = dense<1.000000e+00> : tensor<1x80x56x56xf16>
    // CHECK:     %[[SLICE:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 72, 56, 56] : tensor<1x80x56x56xf16> to tensor<1x72x56x56xf16>
    // CHECK:     %[[EXPAND:.*]] = IE.Expand(%[[SLICE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x72x56x56xf16> -> tensor<1x80x56x56xf16>
    // CHECK:     %[[ADD:.*]] = IE.Add(%[[EXPAND]], %[[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x80x56x56xf16>, tensor<1x80x56x56xf16> -> tensor<1x80x56x56xf16>
    // CHECK: return %[[ADD]] : tensor<1x80x56x56xf16>
}
