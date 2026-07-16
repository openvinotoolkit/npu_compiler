//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% compilation-mode=DefaultHW" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCECompressConv
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x3x224x224xf16, {order = #NHWC}>)
func.func @NCECompressConv(%arg0: tensor<1x3x224x224xf16, {order = #NHWC}>) -> tensor<1x64x112x112xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.0> : tensor<64x4x7x7xf16>, [#const.CastElemType<ui8>,
            #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>,
            #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]
    %weight_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
            : tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%expand, %filter, %weight_table) rawFilterShape [64, 4, 7, 7] {
            cm_sp_pattern = 15 : i64,
            pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
            ppe = #VPU.PPEStub<>,
             strides = [2, 2]
        } -> tensor<1x64x112x112xf16, {order = #NHWC}>
    return %compress_conv : tensor<1x64x112x112xf16, {order = #NHWC}>

    // CHECK-DAG:   [[FILTERS:%.+]] = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x4x7x7xf16>
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[ARG_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
    // CHECK-SAME:      tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {order = #NHWC}>
    // CHECK:       [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[EXPAND]], [[FILTERS]], [[WEIGHTS_TABLE]]) rawFilterShape [64, 4, 7, 7] {
    // CHECK-SAME:      cm_sp_pattern = 15 : i64,
    // CHECK-SAME:      pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [2, 2]
    // CHECK:       return [[VAL0]]
}
