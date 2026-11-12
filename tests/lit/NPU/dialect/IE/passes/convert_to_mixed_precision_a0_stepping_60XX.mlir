//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --convert-to-mixed-precision %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType = !quant.uniform<u16:f32, 8.0507534851041623E-5>

// CHECK-LABEL: @GroupConvWithQuantizeSymmetricU16Quant
module @GroupConvWithQuantizeSymmetricU16Quant attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x3x3xf16>
    func.func @main(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
        %cst = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1x1xf16>

        %0 = IE.GroupConvolution(%arg0, %cst) {
            dilations = [1, 1],
            groups = 16 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

        %1 = IE.Quantize(%0) {
            dstElemType = !qElemType
        } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

        return %1 : tensor<1x16x3x3x!qElemType>

        // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> :
        // CHECK-SAME:  tensor<16x1x1x1xf16>

        // CHECK:   [[GCONV:%.+]] = IE.GroupConvolution([[ARG0]], [[CST]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      groups = 16 : i64,
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

        // CHECK:   return [[GCONV]] : tensor<1x16x3x3x!qElemType>
    }
}

// -----

!qElemType = !quant.uniform<u16:f32, 8.0507534851041623E-5:3325>

// CHECK-LABEL: @DoNotGroupConvWithQuantizeAsymmetricU16Quant
module @DoNotGroupConvWithQuantizeAsymmetricU16Quant attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x3x3xf16>
    func.func @main(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
        %cst = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1x1xf16>

        %0 = IE.GroupConvolution(%arg0, %cst) {
            dilations = [1, 1],
            groups = 16 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

        %1 = IE.Quantize(%0) {
            dstElemType = !qElemType
        } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

        return %1 : tensor<1x16x3x3x!qElemType>

        // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> :
        // CHECK-SAME:  tensor<16x1x1x1xf16>

        // CHECK:   [[GCONV:%.+]] = IE.GroupConvolution([[ARG0]], [[CST]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      groups = 16 : i64,
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

        // CHECK:   [[QUANT:%.+]] = IE.Quantize([[GCONV]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

        // CHECK:   return [[QUANT]] : tensor<1x16x3x3x!qElemType>
    }
}
