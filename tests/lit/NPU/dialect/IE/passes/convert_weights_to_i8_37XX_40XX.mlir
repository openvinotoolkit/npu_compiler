//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-weights-to-i8 --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:127>
// CHECK: !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:127>

// We don't convert u8 to i8 because of the zero point value of U8 which must be 128.
// Conversion converts from u8 ZP = 128 to i8 ZP = 0
// CHECK-LABEL: @NotConvertU8Weights
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x16x16xf16>)
func.func @NotConvertU8Weights(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %cst = const.Declare tensor<3x3x3x3x!qElemType> =
        dense<9.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    return %1 : tensor<1x3x14x14xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<9.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<ui8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<3x3x3x3x!qElemType> -> tensor<3x3x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[CONV]]
}

