//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --convert-quantize-ops-to-nce-ops %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType = !quant.uniform<u16:f32, 8.0507534851041623E-5:3325>

// CHECK-LABEL: @DoNotConvertToGroupConvAsymmetricU16Quant
module @DoNotConvertToGroupConvAsymmetricU16Quant attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: func.func @main
    // CHECK:       ([[INPUT:%.+]]: tensor<1x320x64x64xf16>)
    func.func @main(%arg0 : tensor<1x320x64x64xf16>) -> tensor<1x320x64x64x!qElemType> {
        %0 = IE.Quantize(%arg0) {dstElemType = !qElemType } : tensor<1x320x64x64xf16> -> tensor<1x320x64x64x!qElemType>
        return %0 : tensor<1x320x64x64x!qElemType>

        // CHECK: [[QUANT:%.+]] = IE.Quantize([[INPUT]]) {dstElemType = !qElemType} : tensor<1x320x64x64xf16> -> tensor<1x320x64x64x!qElemType>
        // CHECK: return [[QUANT]] : tensor<1x320x64x64x!qElemType>
    }
}

// -----

!qElemType = !quant.uniform<u16:f32, 8.0507534851041623E-5>

// CHECK-LABEL: @ConvertToGroupConvSymmetricU16Quant
module @ConvertToGroupConvSymmetricU16Quant attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: func.func @main
    // CHECK:       ([[INPUT:%.+]]: tensor<1x320x64x64xf16>)
    func.func @main(%arg0 : tensor<1x320x64x64xf16>) -> tensor<1x320x64x64x!qElemType> {
        %0 = IE.Quantize(%arg0) {dstElemType = !qElemType } : tensor<1x320x64x64xf16> -> tensor<1x320x64x64x!qElemType>
        return %0 : tensor<1x320x64x64x!qElemType>

        // CHECK:  [[CST:%.+]] = const.Declare tensor<320x1x1x1xf16> = dense<1.000000e+00> : tensor<320x1x1x1xf16>
        // CHECK:  [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST]]) {dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x64x64xf16>, tensor<320x1x1x1xf16> -> tensor<1x320x64x64x!qElemType>
        // CHECK:  return [[CONV]] : tensor<1x320x64x64x!qElemType>
    }
}
