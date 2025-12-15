//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-maxpool-with-act %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

func.func @MaxPoolWithReluTest(%arg0: tensor<1x16x5x5xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x5x5xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x4x4xf16>

    %1 = IE.MaxPool(%0)
         {
             kernel_size = [2, 2],
             pads_begin = [0, 0],
             pads_end = [0, 0],
             strides = [1, 1],
             rounding_type = #IE.rounding_type<CEIL>
         } :
         tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %2 = IE.ReLU(%1) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %2 : tensor<1x16x3x3xf16>

    // CHECK-DAG:       [[FILTERS:%.*]] = const.Declare
    // CHECK:       [[VAR0:%.*]] = IE.Convolution(%arg0, [[FILTERS]])
    // CHECK:       [[VAR1:%.*]] = IE.ReLU([[VAR0]])
    // CHECK:       [[VAR2:%.*]] = IE.MaxPool([[VAR1]])

    // CHECK:       return [[VAR2]] : tensor<1x16x3x3xf16>
}
