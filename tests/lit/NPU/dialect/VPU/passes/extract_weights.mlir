//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile" --extract-weights %s | FileCheck %s
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
// CHECK-LABEL: @BaseCase
module @BaseCase {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x1x1xf16, {order = #NHWC}>
        DataInfo "output_1" : tensor<1x8x1x1xf16, {order = #NHWC}>
    }

    // CHECK: func.func nested @kernel_func0([[ARG0:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x8x1x1xf16, {order = #NHWC}>)
    func.func nested @kernel_func0() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
        %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
        // CHECK-NOT: const.Declare
        // CHECK: return [[ARG0]], [[ARG1]] : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
        return %cst0, %cst1: tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
    }

    // CHECK: func.func nested @kernel_func1([[ARG0:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x8x1x1xf16, {order = #NHWC}>)
    func.func nested @kernel_func1() -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>) {
        %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
        // CHECK-NOT: const.Declare
        // CHECK: return [[ARG1]], [[ARG0]] : tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
        return %cst1, %cst0: tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
    }

    // CHECK: func.func @main()
    func.func @main() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
        // CHECK: [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]

        // CHECK: [[CALL_0:%.+]]:2 = call @kernel_func0([[CST]], [[CST_0]]) : (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
        %call0:2 = func.call @kernel_func0() : () -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)

        // CHECK: [[CALL_1:%.+]]:2 = call @kernel_func1([[CST]], [[CST_0]]) : (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>)
        %call1:2 = func.call @kernel_func1() : () -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>)
        // CHECK: return [[CALL_0]]#0, [[CALL_1]]#0 : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
        return %call0#0, %call1#0: tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
    }
}

// -----

{-#
dialect_resources: {
    builtin: {
        vpux_ow_0: "0x1000000001020304",
        vpux_ow_1: "0x100000000506",
        vpux_ow_2: "0x1000000006070809"
    }
}
#-}
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @OneExtraConst
module @OneExtraConst {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x1x1xf16, {order = #NHWC}>
        DataInfo "output_1" : tensor<1x8x1x1xf16, {order = #NHWC}>
    }

    // CHECK: func.func nested @kernel_func0([[ARG0:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x8x1x1xf16, {order = #NHWC}>)
    func.func nested @kernel_func0() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
        %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
        // CHECK-NOT: const.Declare
        // CHECK: return [[ARG0]], [[ARG1]] : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
        return %cst0, %cst1: tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
    }

    // CHECK: func.func nested @kernel_func1([[ARG0:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x8x1x1xf16, {order = #NHWC}>)
    func.func nested @kernel_func1() -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>) {
        %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
        // CHECK: [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_2>
        %cst = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_2> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        // CHECK: return [[ARG1]], [[ARG0]] : tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
        return %cst1, %cst0: tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
    }

    // CHECK: func.func @main()
    func.func @main() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
        // CHECK: [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]

        // CHECK: [[CALL_0:%.+]]:2 = call @kernel_func0([[CST]], [[CST_0]]) : (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
        %call0:2 = call @kernel_func0() : () -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
        // CHECK: [[CALL_1:%.+]]:2 = call @kernel_func1([[CST]], [[CST_0]]) : (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>)
        %call1:2 = call @kernel_func1() : () -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>)

        // CHECK: return [[CALL_0]]#0, [[CALL_1]]#0 : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
        return %call0#0, %call1#0 : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
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
// CHECK-LABEL: @DifferentConsts_NoExtraction
module @DifferentConsts_NoExtraction {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x1x1xf16, {order = #NHWC}>
        DataInfo "output_1" : tensor<1x4x1x1xf16, {order = #NHWC}>
    }

    func.func nested @kernel_func0() -> (tensor<1x16x1x1xf16, {order = #NHWC}>) {
        // CHECK: const.Declare
        %cst0 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]

        return %cst0: tensor<1x16x1x1xf16, {order = #NHWC}>
    }
    func.func nested @kernel_func1() -> (tensor<1x4x1x1xf16, {order = #NHWC}>) {
        // CHECK: const.Declare
        %cst0 = const.Declare tensor<1x4x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>]

        return %cst0: tensor<1x4x1x1xf16, {order = #NHWC}>
    }
    func.func @main() -> (tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x4x1x1xf16, {order = #NHWC}>) {
        // CHECK-NOT: const.Declare

        // CHECK: call @kernel_func0()
        %call0 = func.call @kernel_func0() : () -> (tensor<1x16x1x1xf16, {order = #NHWC}>)
        // CHECK: call @kernel_func1()
        %call1 = func.call @kernel_func1() : () -> (tensor<1x4x1x1xf16, {order = #NHWC}>)
        return %call0, %call1: tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x4x1x1xf16, {order = #NHWC}>
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
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.034925088695451328:128>
!qElemType1 = !quant.uniform<i8:f16, 0.034925088695451328>
// CHECK-LABEL: @QuantizedType
module @QuantizedType {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x1x1xf16, {order = #NHWC}>
        DataInfo "output_1" : tensor<1x8x1x1xf16, {order = #NHWC}>
    }

    // CHECK: func.func nested @kernel_func0([[ARG0:%.+]]: tensor<1x16x1x1xui8, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x8x1x1xf16, {order = #NHWC}>)
    func.func nested @kernel_func0() -> (tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
        // CHECK-NOT: const.Declare
        %cst0 = const.Declare tensor<1x16x1x1x!qElemType, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
        // CHECK: [[EXTRACTED_CAST:%.+]] = VPU.QuantizeCast([[ARG0]]) {dstElemType = !qElemType} : tensor<1x16x1x1xui8, {order = #NHWC}> -> tensor<1x16x1x1x!qElemType, {order = #NHWC}>

        // CHECK: [[Q_CAST:%.+]] = VPU.QuantizeCast([[EXTRACTED_CAST]]) {dstElemType = si8} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x1x1xsi8, {order = #NHWC}>
        %qcast = VPU.QuantizeCast(%cst0) {dstElemType = si8} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x1x1xsi8, {order = #NHWC}>

        // CHECK: return [[Q_CAST]], [[ARG1]] : tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
        return %qcast, %cst1: tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
    }

    // CHECK: func.func nested @kernel_func1([[ARG0:%.+]]: tensor<1x16x1x1xui8, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x8x1x1xf16, {order = #NHWC}>)
    func.func nested @kernel_func1() -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xsi8, {order = #NHWC}>) {
        // CHECK-NOT: const.Declare
        %cst0 = const.Declare tensor<1x16x1x1x!qElemType, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
        %cst1 = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
        // CHECK: [[EXTRACTED_CAST:%.+]] = VPU.QuantizeCast([[ARG0]]) {dstElemType = !qElemType} : tensor<1x16x1x1xui8, {order = #NHWC}> -> tensor<1x16x1x1x!qElemType, {order = #NHWC}>

        // CHECK: [[Q_CAST:%.+]] = VPU.QuantizeCast([[EXTRACTED_CAST]]) {dstElemType = si8} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x1x1xsi8, {order = #NHWC}>
        %qcast = VPU.QuantizeCast(%cst0) {dstElemType = si8} : tensor<1x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x1x1xsi8, {order = #NHWC}>

        // CHECK: return [[ARG1]], [[Q_CAST]] : tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xsi8, {order = #NHWC}>
        return %cst1, %qcast: tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xsi8, {order = #NHWC}>
    }

    // CHECK: func.func @main()
    func.func @main() -> (tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) {
        // CHECK: [[CST:%.+]] = const.Declare tensor<1x16x1x1xui8, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>, #const.CastElemType<ui8>]
        // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<2xui8>, [#const.Reshape<[1, 2, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]

        // CHECK: [[CALL_0:%.+]]:2 = call @kernel_func0([[CST]], [[CST_0]]) : (tensor<1x16x1x1xui8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
        %call0:2 = func.call @kernel_func0() : () -> (tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>)
        // CHECK: [[CALL_1:%.+]]:2 = call @kernel_func1([[CST]], [[CST_0]]) : (tensor<1x16x1x1xui8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>) -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xsi8, {order = #NHWC}>)
        %call1:2 = func.call @kernel_func1() : () -> (tensor<1x8x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xsi8, {order = #NHWC}>)

        // CHECK: return [[CALL_0]]#0, [[CALL_1]]#0 : tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
        return %call0#0, %call1#0: tensor<1x16x1x1xsi8, {order = #NHWC}>, tensor<1x8x1x1xf16, {order = #NHWC}>
    }
}
