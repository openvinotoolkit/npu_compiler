//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --fuse-outstanding-quant %s | FileCheck %s
// REQUIRES: platform-NPU6010

// CHECK: !qElemType = !quant.uniform<u16:f16, 8.2007573709031714E-6:23225>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0013098501691631243:122>
// CHECK: !qElemType2 = !quant.uniform<u16:f16, 4.5571154766645928E-6:25096>
!qElemType = !quant.uniform<u16:f16, 8.2007573709031714E-6:23225>
!qElemType1 = !quant.uniform<u8:f16, 0.0013098501691631243:122>
!qElemType2 = !quant.uniform<u16:f16, 4.5571154766645928E-6:25096>

// CHECK-LABEL: @DoNotFuseConvolutionWithOutstandingU16QuantAndAsymetricU16Output
module @DoNotFuseConvolutionWithOutstandingU16QuantAndAsymetricU16Output attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_A0>} {
    // CHECK-LABEL: func.func @main
    // CHECK:    ([[ARG0:%.+]]: tensor<1x96x64x64xf16>) -> tensor<1x96x64x64x!qElemType>
    func.func @main(%arg0: tensor<1x96x64x64xf16>) -> tensor<1x96x64x64x!qElemType> {
        %weights = const.Declare tensor<96x96x3x3x!qElemType1> = dense<1> : tensor<96x96x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>]
        %quant = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x96x64x64xf16> -> tensor<1x96x64x64x!qElemType2>
        %conv = IE.Convolution(%quant, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Swish<beta = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x96x64x64x!qElemType2>, tensor<96x96x3x3x!qElemType1> -> tensor<1x96x64x64x!qElemType>
        return %conv : tensor<1x96x64x64x!qElemType>
        // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<96x96x3x3x!qElemType1> =
        // CHECK-DAG:    dense<1> : tensor<96x96x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>]
        // CHECK:    [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType2} : tensor<1x96x64x64xf16> -> tensor<1x96x64x64x!qElemType2>
        // CHECK:    [[CONV:%.+]] = IE.Convolution([[QUANTIZE]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Swish<beta = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x96x64x64x!qElemType2>, tensor<96x96x3x3x!qElemType1> -> tensor<1x96x64x64x!qElemType>
        // CHECK:    return [[CONV]] : tensor<1x96x64x64x!qElemType>
    }
}
