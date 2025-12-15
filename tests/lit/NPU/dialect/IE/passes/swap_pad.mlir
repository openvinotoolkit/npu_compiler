//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-pad-layer %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @SwapTransposeWithPerTensorQuant(%arg0: tensor<1x64x128x8xf16>) -> tensor<1x8x70x134xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 3, 3, 0], pads_end_attr = [0, 3, 3, 0]} : tensor<1x64x128x8xf16> -> tensor<1x70x134x8xf16>
    %1 = IE.Transpose(%0) {order_value = #NWCH} : tensor<1x70x134x8xf16> -> tensor<1x8x70x134xf16>

    return %1 : tensor<1x8x70x134xf16>

    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%arg0) {order_value = #NWCH}
    // CHECK-SAME:  : tensor<1x64x128x8xf16> -> tensor<1x8x64x128xf16>

    // CHECK:   %[[PAD:.*]] = IE.Pad(%[[TRANSPOSE]])
    // CHECK-SAME:  : tensor<1x8x64x128xf16> -> tensor<1x8x70x134xf16>

    // CHECK:   return %[[PAD]] : tensor<1x8x70x134xf16>

}
