//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-outstanding-dequant %s | FileCheck %s
// REQUIRES: platform-NPU5010

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.012968034837760177:121>

// CHECK-LABEL: func.func @FuseWithClampNotZeroAndOutputNotQuantized
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x384x20x20xf16>) -> tensor<1x384x20x20xf16>
func.func @FuseWithClampNotZeroAndOutputNotQuantized(%arg0: tensor<1x384x20x20xf16>) -> tensor<1x384x20x20xf16> {
    %cst = const.Declare tensor<384x384x1x1x!quant.uniform<i8:f16, 1.000000e+00>> = dense<1> : tensor<384x384x1x1xsi8>

    %0 = IE.Convolution(%arg0, %cst) {clamp = {max = 1.671716570854187 : f64, min = -1.4173249006271362 : f64}, dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x384x20x20xf16>, tensor<384x384x1x1x!qElemType> -> tensor<1x384x20x20x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x384x20x20x!qElemType1> -> tensor<1x384x20x20xf16>
    return %1 : tensor<1x384x20x20xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<384x384x1x1x!qElemType> = dense<1> : tensor<384x384x1x1xsi8>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {clamp = {max = 1.671716570854187 : f64, min = -1.4173249006271362 : f64}, dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x384x20x20xf16>, tensor<384x384x1x1x!qElemType> -> tensor<1x384x20x20xf16>

    // CHECK-NOT: IE.Dequantize
    // CHECK: return [[CONV]] : tensor<1x384x20x20xf16>
}
