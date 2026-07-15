//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --mlir-elide-elementsattrs-if-larger=8 --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --scf-ops-outlining %s | FileCheck %s --check-prefix=DISABLED
// RUN: vpux-opt --mlir-elide-elementsattrs-if-larger=8 --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --scf-ops-outlining="weights-extraction=true" %s | FileCheck %s --check-prefix=ENABLED
// REQUIRES: platform-NPU4000

{-#
dialect_resources: {
    builtin: {
        vpux_ow_0: "0x1000000001020304"
    }
}
#-}
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (d0 - 1, 0)>

module @ConvolutionWeightsExtractionOption {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x32x64x64xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x256x64x64xf16>
    }

    func.func @main(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>)
            -> tensor<1x256x64x64xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<4xui8>, [#const.Reshape<[1, 4, 1, 1]>, #const.Reorder<#NHWC>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [255, 28, 2, 2]>]
        %output = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %first = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %output) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
            %offset = affine.max #map(%arg1)
            %slice = tensor.extract_slice %arg0[0, 0, %offset, 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
            %conv = VPU.NCE.Convolution(%slice, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
            %inserted = tensor.insert_slice %conv into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
            scf.yield %inserted : tensor<1x256x64x64xf16, {order = #NHWC}>
        }
        %second = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %first) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
            %offset = affine.max #map(%arg1)
            %slice = tensor.extract_slice %arg0[0, 0, %offset, 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
            %conv = VPU.NCE.Convolution(%slice, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
            %inserted = tensor.insert_slice %conv into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
            scf.yield %inserted : tensor<1x256x64x64xf16, {order = #NHWC}>
        }
        return %second : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
}

// DISABLED-LABEL: module @ConvolutionWeightsExtractionOption
// DISABLED:       module @Module0
// DISABLED-NOT:   DataInfo "in_1"
// DISABLED:       func.func @main_func0_static([[ARG0:%.+]]: tensor<1x33x64x32xf16>) -> tensor<1x32x64x256xf16> {
// DISABLED:       const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>
// DISABLED:       VPU.NCE.Convolution
// DISABLED:       module @Module1
// DISABLED-NOT:   DataInfo "in_1"
// DISABLED:       func.func @main_func1_static([[ARG1:%.+]]: tensor<1x33x64x32xf16>) -> tensor<1x32x64x256xf16> {
// DISABLED:       const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>
// DISABLED:       VPU.NCE.Convolution
// DISABLED:       func.func @main(
// DISABLED-NOT:   arith.constant dense
// DISABLED:       Core.NestedCall @Module0::@main_func0_static({{.*}}) : (tensor<1x33x64x32xf16>) -> tensor<1x32x64x256xf16>
// DISABLED:       Core.NestedCall @Module1::@main_func1_static({{.*}}) : (tensor<1x33x64x32xf16>) -> tensor<1x32x64x256xf16>

// ENABLED-LABEL:  module @ConvolutionWeightsExtractionOption
// ENABLED-NOT:    DataInfo "in_1"
// ENABLED:        module @Module0
// ENABLED:        DataInfo "in_1" tensorNames = ["in_1"] : tensor<147456xi8>
// ENABLED:        func.func @main_func0_static([[ARG0:%.+]]: tensor<1x33x64x32xf16>, [[WEIGHTS0:%.+]]: tensor<147456xi8>) -> tensor<1x32x64x256xf16> {
// ENABLED-NOT:    const.Declare
// ENABLED:        [[CAST_WEIGHTS0:%.+]] = Core.ReinterpretCast([[WEIGHTS0]]) : tensor<147456xi8> -> tensor<256x32x3x3xf16, {order = #NHWC}>
// ENABLED:        VPU.NCE.Convolution({{.*}}, [[CAST_WEIGHTS0]])
// ENABLED:        module @Module1
// ENABLED:        DataInfo "in_1" tensorNames = ["in_1"] : tensor<147456xi8>
// ENABLED:        func.func @main_func1_static([[ARG1:%.+]]: tensor<1x33x64x32xf16>, [[WEIGHTS1:%.+]]: tensor<147456xi8>) -> tensor<1x32x64x256xf16> {
// ENABLED-NOT:    const.Declare
// ENABLED:        [[CAST_WEIGHTS1:%.+]] = Core.ReinterpretCast([[WEIGHTS1]]) : tensor<147456xi8> -> tensor<256x32x3x3xf16, {order = #NHWC}>
// ENABLED:        VPU.NCE.Convolution({{.*}}, [[CAST_WEIGHTS1]])
// ENABLED:        func.func @main(
// ENABLED:        [[CST:%.+]] = arith.constant dense_resource<__elided__> : tensor<147456xi8>
// ENABLED:        Core.NestedCall @Module0::@main_func0_static({{.*}}, [[CST]]) : (tensor<1x33x64x32xf16>, tensor<147456xi8>) -> tensor<1x32x64x256xf16>
// ENABLED:        Core.NestedCall @Module1::@main_func1_static({{.*}}, [[CST]]) : (tensor<1x33x64x32xf16>, tensor<147456xi8>) -> tensor<1x32x64x256xf16>
