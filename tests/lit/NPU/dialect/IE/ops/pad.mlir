//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

func.func @OperandsToAttrs(%arg0: tensor<1x5x10x11xf16>) -> tensor<1x11x12x12xf16> {
    %0 = const.Declare tensor<4xsi64> = dense<[0, 3, 0, 1]> : tensor<4xsi64>
    %1 = const.Declare tensor<4xsi64> = dense<[0, 3, 2, 0]> : tensor<4xsi64>
    %2 = const.Declare tensor<f16> = dense<1.000000e+00> : tensor<f16>
    // CHECK-NOT:   const.Declare

    %3 = IE.Pad(%arg0)[%0, %1, %2] {mode = #IE.pad_mode<SYMMETRIC>} : tensor<1x5x10x11xf16>, tensor<4xsi64>, tensor<4xsi64>, tensor<f16> -> tensor<1x11x12x12xf16>
    // CHECK:       %[[VAL0:.*]] = IE.Pad(%arg0) {
    // CHECK-SAME:      mode = #IE.pad_mode<SYMMETRIC>
    // CHECK-SAME:      pad_value_attr = 1.000000e+00
    // CHECK-SAME:      pads_begin_attr = [0, 3, 0, 1]
    // CHECK-SAME:      pads_end_attr = [0, 3, 2, 0]
    // CHECK-SAME:      : tensor<1x5x10x11xf16> -> tensor<1x11x12x12xf16>

    return %3 : tensor<1x11x12x12xf16>
    // CHECK:       return %[[VAL0]]
}

func.func @ConstantFolding() -> tensor<1x11x12x12xf16> {
    %0 = const.Declare tensor<1x5x10x11xf16> = dense<1.0> : tensor<1x5x10x11xf16>

    %1 = IE.Pad(%0) {
            mode = #IE.pad_mode<CONSTANT>,
            pads_begin_attr = [0, 3, 0, 1],
            pads_end_attr = [0, 3, 2, 0],
            pad_value_attr = 0.0
        } : tensor<1x5x10x11xf16> -> tensor<1x11x12x12xf16>

    return %1 : tensor<1x11x12x12xf16>

    // CHECK-DAG:       %[[VAL0:.*]] = const.Declare tensor<1x11x12x12xf16> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x5x10x11xf16>, [#const.PadWithZero<[0, 3, 0, 1], [0, 3, 2, 0]>]
    // CHECK:       return %[[VAL0]]
}
