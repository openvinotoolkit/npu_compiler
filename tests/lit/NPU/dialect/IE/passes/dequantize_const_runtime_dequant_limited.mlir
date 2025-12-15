//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dequantize-const="enable-runtime-dequant=true runtime-dequantization-limit=600000" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.0045055291231940776:174>

// CHECK-LABEL: @KeepLargeDequantizeAboveLimitAsSwKernel()
func.func @KeepLargeDequantizeAboveLimitAsSwKernel() -> tensor<1x320x64x64xf16>  {
   %activation = const.Declare  tensor<1x320x64x64xf16> = dense<1.0> :  tensor<1x320x64x64xf16>
   %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.0> : tensor<1x320x1x1xf32>, [#const.CastElemType<f16>]
   %weights = const.Declare tensor<320x320x3x3x!qElemType> = dense<1> : tensor<320x320x3x3xui8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
   %dequantize = IE.Dequantize(%weights) {dstElemType = f16} : tensor<320x320x3x3x!qElemType> -> tensor<320x320x3x3xf16>
   %conv = IE.Convolution(%activation, %dequantize, %bias) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16>, tensor<320x320x3x3xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x64x64xf16>
   return %conv : tensor<1x320x64x64xf16>

   // CHECK:        [[ACT:%.+]] =  const.Declare  tensor<1x320x64x64xf16>
   // CHECK:        [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16>
   // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<320x320x3x3x!qElemType>
   // CHECK:        [[DEQUANTIZE:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16}
   // CHECK:        [[CONV:%.+]] = IE.Convolution([[ACT]], [[DEQUANTIZE]], [[BIAS]])
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.0045055291231940776:174>

// CHECK-LABEL: @FoldSmallDequantizeBelowLimit()
func.func @FoldSmallDequantizeBelowLimit() -> tensor<1x180x64x64xf16>  {
   %activation = const.Declare  tensor<1x320x64x64xf16> = dense<1.0> :  tensor<1x320x64x64xf16>
   %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.0> : tensor<1x320x1x1xf32>, [#const.CastElemType<f16>]
   %weights = const.Declare tensor<180x320x3x3x!qElemType> = dense<1> : tensor<180x320x3x3xui8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
   %dequantize = IE.Dequantize(%weights) {dstElemType = f16} : tensor<180x320x3x3x!qElemType> -> tensor<180x320x3x3xf16>
   %conv = IE.Convolution(%activation, %dequantize, %bias) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16>, tensor<180x320x3x3xf16>, tensor<1x320x1x1xf16> -> tensor<1x180x64x64xf16>
   return %conv : tensor<1x180x64x64xf16>

   // CHECK:        [[ACT:%.+]] =  const.Declare  tensor<1x320x64x64xf16>
   // CHECK:        [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16>
   // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<180x320x3x3xf16>
   // CHECK-SAME:    #const.Dequantize
   // CHECK:        [[CONV:%.+]] = IE.Convolution([[ACT]], [[WEIGHTS]], [[BIAS]])
}
