//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

//  CHECK-LABEL: @ConvertConvertLikeToConvertF16ToF32
//  CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<2x2xf16>)
func.func @ConvertConvertLikeToConvertF16ToF32(%arg0: tensor<2x2xf16>) -> tensor<2x2xf32> {
    %cst = const.Declare tensor<2xf32> = dense<0.000000e+00> : tensor<2xf32>
    %0 = IE.ConvertLike(%arg0, %cst) : tensor<2x2xf16>, tensor<2xf32> -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>

    //CHECK:   [[CONVERT:%.+]] = IE.Convert([[ARG_0]]) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
    //CHECK:   return [[CONVERT]] : tensor<2x2xf32>
}

//  CHECK-LABEL: @ConvertConvertLikeToConvertI8ToI32
//  CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<2x2xsi8>)
func.func @ConvertConvertLikeToConvertI8ToI32(%arg0: tensor<2x2xsi8>) -> tensor<2x2xsi32> {
    %cst = const.Declare tensor<2xsi32> = dense<0> : tensor<2xsi32>
    %0 = IE.ConvertLike(%arg0, %cst) : tensor<2x2xsi8>, tensor<2xsi32> -> tensor<2x2xsi32>
    return %0 : tensor<2x2xsi32>

    //CHECK:   [[CONVERT:%.+]] = IE.Convert([[ARG_0]]) {dstElemType = si32} : tensor<2x2xsi8> -> tensor<2x2xsi32>
    //CHECK:   return [[CONVERT]] : tensor<2x2xsi32>
}

//  CHECK-LABEL: @ConvertConvertLikeToConvertF32ToUI8
//  CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<2x2xf32>)
func.func @ConvertConvertLikeToConvertF32ToUI8(%arg0: tensor<2x2xf32>) -> tensor<2x2xui8> {
    %cst = const.Declare tensor<2xui8> = dense<0> : tensor<2xui8>
    %0 = IE.ConvertLike(%arg0, %cst) : tensor<2x2xf32>, tensor<2xui8> -> tensor<2x2xui8>
    return %0 : tensor<2x2xui8>

    //CHECK:   [[CONVERT:%.+]] = IE.Convert([[ARG_0]]) {dstElemType = ui8} : tensor<2x2xf32> -> tensor<2x2xui8>
    //CHECK:   return [[CONVERT]] : tensor<2x2xui8>
}

//  CHECK-LABEL: @ConvertLikeFold
//  CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<2x2xf32>)
func.func @ConvertLikeFold(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = const.Declare tensor<2xf32> = dense<0.000000e+00> : tensor<2xf32>
    %0 = IE.ConvertLike(%arg0, %cst) : tensor<2x2xf32>, tensor<2xf32> -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>

    //CHECK-NOT: IE.ConvertLike
    //CHECK-NOT: IE.Convert
    //CHECK:   return [[ARG_0]] : tensor<2x2xf32>
}
