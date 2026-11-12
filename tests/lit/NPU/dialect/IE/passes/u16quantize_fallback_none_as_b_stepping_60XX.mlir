//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% allow-custom-values=true" --u16-quantize-fallback --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType = !quant.uniform<u16:f16, 1.2090807877116612E-5:222>
!qElemType1 = !quant.uniform<u8:f16, 3.4662022017965132E-4:128>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @ConvTest attributes {
    config.compilationMode = #config.compilation_mode<DefaultHW>,
    config.revisionID = #config.revision_id<REVISION_NONE>} {
    // CHECK-LABEL: func.func @RevisionNoneBehavesAsBStepping
    // CHECK-SAME: ([[ARG_0:%.+]]: tensor<1x3x16x16x!qElemType, {order = #NHWC}>)
    func.func @RevisionNoneBehavesAsBStepping(%arg0: tensor<1x3x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x16x18x18x!qElemType, {order = #NHWC}> {
        %weights = const.Declare tensor<16x3x1x1x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<16x3x1x1xf16>, [#const.ConvertElemType<!qElemType1>, #const.Reorder<#NHWC>]
        %0 = IE.Convolution(%arg0, %weights)
              {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
              : tensor<1x3x16x16x!qElemType, {order = #NHWC}>, tensor<16x3x1x1x!qElemType1, {order = #NHWC}>
                -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>

        return %0 : tensor<1x16x18x18x!qElemType, {order = #NHWC}>

        // CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<16x3x1x1x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x3x1x1xf16>, [#const.ConvertElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.ConvertElemType<!qElemType1>]
        // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[ARG_0]]) {dstElemType = f16} : tensor<1x3x16x16x!qElemType, {order = #NHWC}> -> tensor<1x3x16x16xf16, {order = #NHWC}>
        // CHECK: [[CONV:%.+]] = IE.Convolution([[DEQUANT]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16xf16, {order = #NHWC}>, tensor<16x3x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x18x18x!qElemType, {order = #NHWC}>
        // CHECK-NEXT: return [[CONV]] : tensor<1x16x18x18x!qElemType, {order = #NHWC}>
    }
}
