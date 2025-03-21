//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dequantize-const %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8:f32:0, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @PerAxis
func.func @PerAxis() -> tensor<4x1x1x1xf32> {
    %0 = const.Declare tensor<4x1x1x1x!qElemType> =
        dense<129> : tensor<4x1x1x1xui8>, [#const.CastElemType<!qElemType>]
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<4x1x1x1x!qElemType> -> tensor<4x1x1x1xf32>
    return %1 : tensor<4x1x1x1xf32>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare
    // CHECK-SAME:      dense<129> : tensor<4x1x1x1xui8
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK-SAME:      #const.Dequantize

    // CHECK:       return [[CST]]
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.0045055291231940776:174>

// CHECK-LABEL: @FoldDequantize()
func.func @FoldDequantize() -> tensor<1x320x64x64xf16>  {
   %activation = const.Declare  tensor<1x320x64x64xf16> = dense<1.0> :  tensor<1x320x64x64xf16>
   %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.0> : tensor<1x320x1x1xf32>, [#const.CastElemType<f16>]
   %weights = const.Declare tensor<320x320x3x3x!qElemType> = dense<1> : tensor<320x320x3x3xui8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
   %dequantize = IE.Dequantize(%weights) {dstElemType = f16} : tensor<320x320x3x3x!qElemType> -> tensor<320x320x3x3xf16>
   %conv = IE.Convolution(%activation, %dequantize, %bias) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16>, tensor<320x320x3x3xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x64x64xf16>
   return %conv : tensor<1x320x64x64xf16>

   // CHECK:        [[ACT:%.+]] =  const.Declare  tensor<1x320x64x64xf16>
   // CHECK:        [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16>
   // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<320x320x3x3xf16>
   // CHECK-SAME:    #const.Dequantize
   // CHECK:        [[CONV:%.+]] = IE.Convolution([[ACT]], [[WEIGHTS]], [[BIAS]])
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.quantile<i4:f16:f16, {-1.0000,-0.8667,-0.7333,-0.6000,-0.4667,-0.3333,-0.2000,-0.0667,0.0667,0.2000,0.3333,0.4667,0.6000,0.7333,0.8667,1.0000}:0.0045055291231940776>

// CHECK-LABEL: @FoldDequantizeNF4()
func.func @FoldDequantizeNF4() -> tensor<1x320x64x64xf16>  {
   %activation = const.Declare  tensor<1x320x64x64xf16> = dense<1.0> :  tensor<1x320x64x64xf16>
   %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.0> : tensor<1x320x1x1xf32>, [#const.CastElemType<f16>]
   %weights = const.Declare tensor<320x320x3x3x!qElemType> = dense<1> : tensor<320x320x3x3xui8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
   %dequantize = IE.Dequantize(%weights) {dstElemType = f16} : tensor<320x320x3x3x!qElemType> -> tensor<320x320x3x3xf16>
   %conv = IE.Convolution(%activation, %dequantize, %bias) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16>, tensor<320x320x3x3xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x64x64xf16>
   return %conv : tensor<1x320x64x64xf16>

   // CHECK:        [[ACT:%.+]] =  const.Declare  tensor<1x320x64x64xf16>
   // CHECK:        [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16>
   // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<320x320x3x3xf16>
   // CHECK-SAME:    #const.Dequantize
   // CHECK:        [[CONV:%.+]] = IE.Convolution([[ACT]], [[WEIGHTS]], [[BIAS]])
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.quantile<i4:i8:f16:2, {-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0}:{1.000000e-01:0,2.000000e-01:0,3.000000e-01}>

// CHECK-LABEL: @FoldDequantizeQuantilePerAxis()
func.func @FoldDequantizeQuantilePerAxis() -> tensor<1x320x64x64xf16>  {
   %activation = const.Declare  tensor<1x320x64x64xf16> = dense<1.0> :  tensor<1x320x64x64xf16>
   %bias = const.Declare tensor<1x320x1x1xf16> = dense<1.0> : tensor<1x320x1x1xf32>, [#const.CastElemType<f16>]
   %weights = const.Declare tensor<320x320x3x3x!qElemType> = dense<1> : tensor<320x320x3x3xui8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
   %dequantize = IE.Dequantize(%weights) {dstElemType = f16} : tensor<320x320x3x3x!qElemType> -> tensor<320x320x3x3xf16>
   %conv = IE.Convolution(%activation, %dequantize, %bias) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16>, tensor<320x320x3x3xf16>, tensor<1x320x1x1xf16> -> tensor<1x320x64x64xf16>
   return %conv : tensor<1x320x64x64xf16>

   // CHECK:        [[ACT:%.+]] =  const.Declare  tensor<1x320x64x64xf16>
   // CHECK:        [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16>
   // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<320x320x3x3xf16>
   // CHECK-SAME:    #const.Dequantize
   // CHECK:        [[CONV:%.+]] = IE.Convolution([[ACT]], [[WEIGHTS]], [[BIAS]])
}
