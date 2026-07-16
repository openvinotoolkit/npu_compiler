//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --extract-weights --concat-host-consts %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

{-#
  dialect_resources: {
    builtin: {
      vpux_ow_0: "0x1000000001020304",
      vpux_ow_1: "0x100000000506"
    }
  }
#-}
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @BaseCase {
  net.NetworkInfo entryPoint : @main inputsInfo : {
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x1x1xf16, {order = #NHWC}>
    DataInfo "output_1" : tensor<1x8x1x1xf16, {order = #NHWC}>
  }
  // CHECK:       func.func private @kernel_func0([[ARG:%.+]]: tensor<48xi8>)
  func.func private @kernel_func0() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
    // CHECK-NOT: const.Declare
    %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
    // CHECK:     [[SLICE_0:%.+]] = VPU.Slice [[ARG]] [0] [32] : tensor<48xi8> to tensor<32xi8>
    // CHECK:     [[CAST_0:%.+]] = Core.ReinterpretCast([[SLICE_0]]) : tensor<32xi8> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:     [[SLICE_1:%.+]] = VPU.Slice [[ARG]] [32] [16] : tensor<48xi8> to tensor<16xi8>
    // CHECK:     [[CAST_1:%.+]] = Core.ReinterpretCast([[SLICE_1]]) : tensor<16xi8> -> tensor<1x8x1x1xf16, {order = #NHWC}>
    // CHECK:     return [[CAST_0]], [[CAST_1]] : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
    return %cst0, %cst1: tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
  }
  // CHECK:       func.func private @kernel_func1([[ARG:%.+]]: tensor<48xi8>)
  func.func private @kernel_func1() -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>) {
    // CHECK-NOT: const.Declare
    %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
    // CHECK:     [[SLICE_0:%.+]] = VPU.Slice [[ARG]] [0] [32] : tensor<48xi8> to tensor<32xi8>
    // CHECK:     [[CAST_0:%.+]] = Core.ReinterpretCast([[SLICE_0]]) : tensor<32xi8> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:     [[SLICE_1:%.+]] = VPU.Slice [[ARG]] [32] [16] : tensor<48xi8> to tensor<16xi8>
    // CHECK:     [[CAST_1:%.+]] = Core.ReinterpretCast([[SLICE_1]]) : tensor<16xi8> -> tensor<1x8x1x1xf16, {order = #NHWC}>
    // CHECK:     return [[CAST_1]], [[CAST_0]] : tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
    return %cst1, %cst0: tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
  }
  func.func @main() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
    // CHECK-NOT: const.Declare
    // CHECK:     [[CST:%.+]] = arith.constant dense<[0, 60, 0, 64, 0, 66, 0, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<48xi8>
    // CHECK:     [[CALL_0:%.+]]:2 = call @kernel_func0([[CST]]) : (tensor<48xi8>) -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
    %call0:2 = func.call @kernel_func0() : () -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
    // CHECK:     [[CALL_1:%.+]]:2 = call @kernel_func1([[CST]]) : (tensor<48xi8>) -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>)
    %call1:2 = func.call @kernel_func1() : () -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>)
    return %call0#0, %call1#0: tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
  }
}

// -----
{-#
  dialect_resources: {
    builtin: {
      vpux_ow_2: "0x1000000006070809",
      vpux_ow_0: "0x1000000001020304",
      vpux_ow_1: "0x100000000506"
    }
  }
#-}
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @OneExtraConst {
  net.NetworkInfo entryPoint : @main inputsInfo : {
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x1x1xf16, {order = #NHWC}>
    DataInfo "output_1" : tensor<1x8x1x1xf16, {order = #NHWC}>
  }
  // CHECK:       func.func private @kernel_func0([[ARG:%.+]]: tensor<48xi8>)
  func.func private @kernel_func0() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {

    %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
    // CHECK:     [[SLICE_0:%.+]] = VPU.Slice [[ARG]] [0] [32] : tensor<48xi8> to tensor<32xi8>
    // CHECK:     [[CAST_0:%.+]] = Core.ReinterpretCast([[SLICE_0]]) : tensor<32xi8> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:     [[SLICE_1:%.+]] = VPU.Slice [[ARG]] [32] [16] : tensor<48xi8> to tensor<16xi8>
    // CHECK:     [[CAST_1:%.+]] = Core.ReinterpretCast([[SLICE_1]]) : tensor<16xi8> -> tensor<1x8x1x1xf16, {order = #NHWC}>
    // CHECK:     return [[CAST_0]], [[CAST_1]] : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
    return %cst0, %cst1: tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
  }
  // CHECK:       func.func private @kernel_func1([[ARG:%.+]]: tensor<48xi8>)
  func.func private @kernel_func1() -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>) {
    %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
    // CHECK:     [[SLICE_0:%.+]] = VPU.Slice [[ARG]] [0] [32] : tensor<48xi8> to tensor<32xi8>
    // CHECK:     [[CAST_0:%.+]] = Core.ReinterpretCast([[SLICE_0]]) : tensor<32xi8> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:     [[SLICE_1:%.+]] = VPU.Slice [[ARG]] [32] [16] : tensor<48xi8> to tensor<16xi8>
    // CHECK:     [[CAST_1:%.+]] = Core.ReinterpretCast([[SLICE_1]]) : tensor<16xi8> -> tensor<1x8x1x1xf16, {order = #NHWC}>
    // CHECK:     [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_2>
    // CHECK:     return [[CAST_1]], [[CAST_0]] : tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
    %cst = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_2> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    return %cst1, %cst0: tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
  }
  func.func @main() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
    // CHECK-NOT: const.Declare
    // CHECK:     [[CST:%.+]] = arith.constant dense<[0, 60, 0, 64, 0, 66, 0, 68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<48xi8>
    // CHECK:     [[CALL_0:%.+]]:2 = call @kernel_func0([[CST]]) : (tensor<48xi8>) -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
    %call0:2 = call @kernel_func0() : () -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
    // CHECK:     [[CALL_1:%.+]]:2 = call @kernel_func1([[CST]]) : (tensor<48xi8>) -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>)
    %call1:2 = call @kernel_func1() : () -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>)
    return %call0#0, %call1#0: tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
  }
}

// -----
{-#
  dialect_resources: {
    builtin: {
      vpux_ow_0: "0x1000000001020304"
    }
  }
#-}
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @DifferentConsts_NoExtraction {
  net.NetworkInfo entryPoint : @main inputsInfo : {
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x1x1xf16, {order = #NHWC}>
    DataInfo "output_1" : tensor<1x4x1x1xf16, {order = #NHWC}>
  }
  func.func private @kernel_func0() -> tensor<1x16x1x1xf16, {order = #NHWC}> {
    // CHECK:     [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0>
    %cst = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    return %cst : tensor<1x16x1x1xf16, {order = #NHWC}>
  }
  func.func private @kernel_func1() -> tensor<1x4x1x1xf16, {order = #NHWC}> {
    // CHECK:     [[CST:%.+]] = const.Declare tensor<1x4x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0>
    %cst = const.Declare tensor<1x4x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>]
    return %cst : tensor<1x4x1x1xf16, {order = #NHWC}>
  }
  func.func @main() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x4x1x1xf16, {order = #NHWC}>) {
    %0 = call @kernel_func0() : () -> tensor<1x16x1x1xf16, {order = #NHWC}>
    %1 = call @kernel_func1() : () -> tensor<1x4x1x1xf16, {order = #NHWC}>
    return %0, %1 : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x4x1x1xf16, {order = #NHWC}>
  }
}

// -----
{-#
  dialect_resources: {
    builtin: {
      vpux_ow_0: "0x1000000001020304",
      vpux_ow_1: "0x100000000506"
    }
  }
#-}
!qElemType = !quant.uniform<u8:f16, 0.034925088695451328:128>
!qElemType1 = !quant.uniform<i8:f16, 0.034925088695451328>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @QuantizedType {
  net.NetworkInfo entryPoint : @main inputsInfo : {
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x1x1xf16, {order = #NHWC}>
    DataInfo "output_1" : tensor<1x8x1x1xf16, {order = #NHWC}>
  }
  // CHECK:       func.func private @kernel_func0([[ARG:%.+]]: tensor<32xi8>)
  func.func private @kernel_func0() -> (tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
    // CHECK-NOT: const.Declare
    %cst0 = const.Declare tensor<1x16x1x1x!qElemType, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
    // CHECK:     [[SLICE_0:%.+]] = VPU.Slice [[ARG]] [0] [16] : tensor<32xi8> to tensor<16xi8>
    // CHECK:     [[CAST_0:%.+]] = Core.ReinterpretCast([[SLICE_0]]) : tensor<16xi8> -> tensor<1x16x1x1xui8, {order = #NHWC}>
    // CHECK:     [[SLICE_1:%.+]] = VPU.Slice [[ARG]] [16] [16] : tensor<32xi8> to tensor<16xi8>
    // CHECK:     [[CAST_1:%.+]] = Core.ReinterpretCast([[SLICE_1]]) : tensor<16xi8> -> tensor<1x8x1x1xf16, {order = #NHWC}>
    // CHECK:     [[QUANT_CAST_0:%.+]] = VPU.QuantizeCast([[CAST_0]]) {dstElemType = !qElemType} : tensor<1x16x1x1xui8, {order = #NHWC}> -> tensor<1x16x1x1x!qElemType, {order = #NHWC}>
    // CHECK:     [[QUANT_CAST_1:%.+]] = VPU.QuantizeCast([[QUANT_CAST_0]]) {dstElemType = si8} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x1x1xsi8, {order = #NHWC}>
    // CHECK:     return [[QUANT_CAST_1]], [[CAST_1]] : tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
    %qcast = VPU.QuantizeCast(%cst0) {dstElemType = si8} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x1x1xsi8, {order = #NHWC}>
    return %qcast, %cst1: tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
  }
  // CHECK:       func.func private @kernel_func1([[ARG:%.+]]: tensor<32xi8>)
  func.func private @kernel_func1() -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xsi8, {order = #NHWC}>) {
    // CHECK-NOT: const.Declare
    %cst0 = const.Declare tensor<1x16x1x1x!qElemType, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
    // CHECK:     [[SLICE_0:%.+]] = VPU.Slice [[ARG]] [0] [16] : tensor<32xi8> to tensor<16xi8>
    // CHECK:     [[CAST_0:%.+]] = Core.ReinterpretCast([[SLICE_0]]) : tensor<16xi8> -> tensor<1x16x1x1xui8, {order = #NHWC}>
    // CHECK:     [[SLICE_1:%.+]] = VPU.Slice [[ARG]] [16] [16] : tensor<32xi8> to tensor<16xi8>
    // CHECK:     [[CAST_1:%.+]] = Core.ReinterpretCast([[SLICE_1]]) : tensor<16xi8> -> tensor<1x8x1x1xf16, {order = #NHWC}>
    // CHECK:     [[QUANT_CAST_0:%.+]] = VPU.QuantizeCast([[CAST_0]]) {dstElemType = !qElemType} : tensor<1x16x1x1xui8, {order = #NHWC}> -> tensor<1x16x1x1x!qElemType, {order = #NHWC}>
    // CHECK:     [[QUANT_CAST_1:%.+]] = VPU.QuantizeCast([[QUANT_CAST_0]]) {dstElemType = si8} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x1x1xsi8, {order = #NHWC}>
    // CHECK:     return [[CAST_1]], [[QUANT_CAST_1]] : tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xsi8, {order = #NHWC}>
    %qcast = VPU.QuantizeCast(%cst0) {dstElemType = si8} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x1x1xsi8, {order = #NHWC}>
    return %cst1, %qcast: tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xsi8, {order = #NHWC}>
  }
  func.func @main() -> (tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
    // CHECK-NOT: const.Declare
    // CHECK:     [[CST:%.+]] = arith.constant dense<[-127, -126, -125, -124, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 0, 69, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<32xi8>
    // CHECK:     [[CALL_0:%.+]]:2 = call @kernel_func0([[CST]]) : (tensor<32xi8>) -> (tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
    %call0:2 = func.call @kernel_func0() : () -> (tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
    // CHECK:     [[CALL_1:%.+]]:2 = call @kernel_func1([[CST]]) : (tensor<32xi8>) -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xsi8, {order = #NHWC}>)
    %call1:2 = func.call @kernel_func1() : () -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xsi8, {order = #NHWC}>)
    return %call0#0, %call1#0: tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
  }
}

// -----

{-#
  dialect_resources: {
    builtin: {
      vpux_ow_0: "0x1000000000450044003C0044804800400045004600470048003C0046003C0048003C0046804800408048004000450040003C004800450044003C004600450040003C0040004500000045004080480044004500468048004400420044003C004000450044004700400047004600470048004500400042000000450044004500460042004080480046003C004400420046003C00488048004400420000003C00488048004800420046003C004000470040004500400045004400420048004500400045004880480046004500480045004680480040003C0046804800000047004000450048004500000047004080480000804800460045000000470048004700460042000000450040003C00000047004000470048003C000080480040004200460042004800420048003C004000420044003C00440042000000420048003C0000003C00008048000000450040804800480047000000450000004500460045004800450040004200480047004800420044004200440045004800420040004700488048000000450000003C0046003C004600450000004500440045000000450044004500000047004400420048003C004400450040004500460047000080480044804800400042004680480048003C00460042000080480044003C00400042004400450040003C004000470040003C00400045004400450046004700000045004800420000004500000047004000450040004200448048004600450048003C0044804800000042004600470044003C00440042004600470040004200480047004400420048804800440042004800470044003C004600420044003C004400450000004200460047000000450000003C004680480000804800440045004000470044003C00468048000080480048003C00400045004800450040003C0000003C0044004500408048004600470044804800400045004880480044003C00460047004800420040003C00480045004600470046004200000042004400420040004500460042004400470040003C0048804800000045004000450000003C00400045000000450000003C0000003C004600470048003C0046004700440042004080480046003C004000420048004200440047004000420040004200488048004800470048003C00460047004800450048004700440045004880480040004500000047000000420040"
    }
  }
#-}
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @SingleInput {
  config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" friendlyName = "Result_16" : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
  }
  func.func private @kernel_func0(%main: tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 256, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 256, 1280]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Permute(%main) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 3 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, tiling_loop_index = 0 : i64} -> tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 256, 1280]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 256, 1280]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:   func.func private @kernel_func1([[ARG0:%.+]]: tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>, [[ARG1:%.+]]: tensor<4608xi8>)
  func.func private @kernel_func1(%main: tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}> {
    // VPU.Slice will be folded during canonicalization
    // CHECK: [[SLICE:%.+]] = VPU.Slice [[ARG1]] [0] [4608] : tensor<4608xi8> to tensor<4608xi8>
    // CHECK: [[CAST:%.+]] = Core.ReinterpretCast([[SLICE]]) : tensor<4608xi8> -> tensor<16x1x1x144xf16, {order = #NHWC}>
    %cst = const.Declare tensor<16x1x1x144xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<16x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[16, 1, 1, 27]>, #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 117]>]
    // CHECK: VPU.NCE.Convolution([[ARG0]], [[CAST]])
    %0 = VPU.NCE.Convolution(%main, %cst) rawFilterShape [16, 3, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tiling_loop_index = 1 : i64} : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x1x1x144xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
  }

  // CHECK:   func.func private @kernel_func2([[ARG0:%.+]]: tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>, [[ARG1:%.+]]: tensor<4608xi8>)
  func.func private @kernel_func2(%main: tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK: [[SLICE:%.+]] = VPU.Slice [[ARG1]] [0] [4608] : tensor<4608xi8> to tensor<4608xi8>
    // CHECK: [[CAST:%.+]] = Core.ReinterpretCast([[SLICE]]) : tensor<4608xi8> -> tensor<16x1x1x144xf16, {order = #NHWC}>
    %cst = const.Declare tensor<16x1x1x144xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<16x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[16, 1, 1, 27]>, #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 117]>]
    // CHECK: VPU.NCE.Convolution([[ARG0]], [[CAST]])
    %0 = VPU.NCE.Convolution(%main, %cst) rawFilterShape [16, 3, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tiling_loop_index = 1 : i64} : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x1x1x144xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
  }

  // CHECK:   func.func private @kernel_func3([[ARG0:%.+]]: tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 88, 1280]> : tensor<4xsi64>, order = #NHWC}>, [[ARG1:%.+]]: tensor<4608xi8>)
  func.func private @kernel_func3(%main: tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 88, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK: [[SLICE:%.+]] = VPU.Slice [[ARG1]] [0] [4608] : tensor<4608xi8> to tensor<4608xi8>
    // CHECK: [[CAST:%.+]] = Core.ReinterpretCast([[SLICE]]) : tensor<4608xi8> -> tensor<16x1x1x144xf16, {order = #NHWC}>
    %cst = const.Declare tensor<16x1x1x144xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<16x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[16, 1, 1, 27]>, #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 117]>]
    // CHECK: VPU.NCE.Convolution([[ARG0]], [[CAST]])
    %0 = VPU.NCE.Convolution(%main, %cst) rawFilterShape [16, 3, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], tiling_loop_index = 1 : i64} : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 88, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x1x1x144xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
  }

  func.func @main(%arg: tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK: [[CST:%.+]] = arith.constant dense<"0x{{([0-9]|[A-Z])+}}"> : tensor<4608xi8>
    %index = arith.constant 0 : index // hardcoded index to simplify the test
    %switch = scf.index_switch %index -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
    case 0 {
      // CHECK: [[SLICE_0:%.+]] = tensor.extract_slice
      %extract_slice_0 = tensor.extract_slice %arg[0, 0, 0, 0] [1, 3, 88, 1280] [1, 1, 1, 1] : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x3x88x1280xf16, {order = #NHWC}>
      // CHECK: [[CAST_0:%.+]] = tensor.cast [[SLICE_0]]
      %cast_0 = tensor.cast %extract_slice_0 : tensor<1x3x88x1280xf16, {order = #NHWC}> to tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 88, 1280]> : tensor<4xsi64>, order = #NHWC}>
      // CHECK: func.call @kernel_func3([[CAST_0]], [[CST]]) : (tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 88, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<4608xi8>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
      %kernel_output_0 = func.call @kernel_func3(%cast_0) : (tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 88, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
      scf.yield %kernel_output_0 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
    }
    case 1 {
      // CHECK: [[SLICE_1:%.+]] = tensor.extract_slice
      %extract_slice_1 = tensor.extract_slice %arg[0, 0, 0, 0] [1, 3, 87, 1280] [1, 1, 1, 1] : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x3x87x1280xf16, {order = #NHWC}>
      // CHECK: [[CAST_1:%.+]] = tensor.cast [[SLICE_1]]
      %cast_1 = tensor.cast %extract_slice_1 : tensor<1x3x87x1280xf16, {order = #NHWC}> to tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>
      // CHECK: func.call @kernel_func2([[CAST_1]], [[CST]]) : (tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<4608xi8>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
      %kernel_output_1 = func.call @kernel_func2(%cast_1) : (tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
      scf.yield %kernel_output_1 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
    }
    case 2 {
      // CHECK: [[SLICE_2:%.+]] = tensor.extract_slice
      %extract_slice_2 = tensor.extract_slice %arg[0, 0, 0, 0] [1, 3, 87, 1280] [1, 1, 1, 1] : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x3x87x1280xf16, {order = #NHWC}>
      // CHECK: [[CAST_2:%.+]] = tensor.cast [[SLICE_2]]
      %cast_2 = tensor.cast %extract_slice_2 : tensor<1x3x87x1280xf16, {order = #NHWC}> to tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>
      // CHECK: func.call @kernel_func1([[CAST_2]], [[CST]]) : (tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<4608xi8>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
      %kernel_output_2 = func.call @kernel_func1(%cast_2) : (tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 87, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
      scf.yield %kernel_output_2 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
    }
    default {
      %cst = arith.constant false
      cf.assert %cst, "Unsupported case"
      // CHECK: [[SLICE_DEF:%.+]] = tensor.extract_slice
      %extract_slice_def = tensor.extract_slice %arg[0, 0, 0, 0] [1, 3, 88, 1280] [1, 1, 1, 1] : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x3x88x1280xf16, {order = #NHWC}>
      // CHECK: [[CAST_DEF:%.+]] = tensor.cast [[SLICE_DEF]]
      %cast_def = tensor.cast %extract_slice_def : tensor<1x3x88x1280xf16, {order = #NHWC}> to tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 88, 1280]> : tensor<4xsi64>, order = #NHWC}>
      // CHECK: func.call @kernel_func3([[CAST_DEF]], [[CST]]) : (tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 88, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<4608xi8>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
      %kernel_output_def = func.call @kernel_func3(%cast_def) : (tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 88, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
      scf.yield %kernel_output_def : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
    }
    return %switch: tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 86, 1280]> : tensor<4xsi64>, order = #NCHW}>
  }
}
