//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-permute-cast-through-dequantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.006552408723270192:147>

// CHECK-LABEL: @OptimizeDequantizeMemPermuteReorder()
func.func @OptimizeDequantizeMemPermuteReorder() -> tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
   %cst = const.Declare tensor<320x1280x1x1x!qElemType> = dense<1> : tensor<320x1280xui8>, [#const.Reshape<[1, 1, 320, 1280]>, #const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reshape<[320, 1280, 1, 1]>]
   %dequantize = IE.Dequantize(%cst) {dstElemType = f16} : tensor<320x1280x1x1x!qElemType> -> tensor<320x1280x1x1xf16>
   %permute_cast = IE.PermuteCast(%dequantize) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<320x1280x1x1xf16> -> tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

   return %permute_cast : tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
   // CHECK:        [[CST:%.+]] =  const.Declare tensor<320x1280x1x1x!qElemType, {order = #NHWC}>
   // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<320x1280x1x1x!qElemType, {order = #NHWC}> -> tensor<320x1280x1x1xf16, {order = #NHWC}>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.006552408723270192:147>

// CHECK-LABEL: @IncorrectLayoutDontPropagatePermuteThroughDequantize()
func.func @IncorrectLayoutDontPropagatePermuteThroughDequantize() -> tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> {
   %cst = const.Declare tensor<320x1280x1x1x!qElemType> = dense<1> : tensor<320x1280xui8>, [#const.Reshape<[1, 1, 320, 1280]>, #const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reshape<[320, 1280, 1, 1]>]
   %dequantize = IE.Dequantize(%cst) {dstElemType = f16} : tensor<320x1280x1x1x!qElemType> -> tensor<320x1280x1x1xf16>
   %permute_cast = IE.PermuteCast(%dequantize) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<320x1280x1x1xf16> -> tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>

   return %permute_cast : tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
   // CHECK:        [[CST:%.+]] =  const.Declare tensor<320x1280x1x1x!qElemType>
   // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<320x1280x1x1x!qElemType> -> tensor<320x1280x1x1xf16>
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[DEQUANT]]) {dst_order = #NWHC, mem_perm = #NHWC} : tensor<320x1280x1x1xf16> -> tensor<320x1280x1x1xf16, {order = #NWHC}>

}
