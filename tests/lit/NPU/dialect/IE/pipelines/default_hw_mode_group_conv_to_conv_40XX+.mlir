//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:254>:f16, 0.0031488373523622048:127>
!qElemType1 = !quant.uniform<i8<-127:127>:f16, 0.0031488373523622048>
!qElemType2 = !quant.uniform<u8:f16, 0.0078431372549019607:128>
!qElemType3 = !quant.uniform<u8:f16, 0.015686274509803921:128>

// CHECK-LABEL: @GroupConvolutionToSingleConvolution
module @GroupConvolutionToSingleConvolution {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x96x128x256xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x96x128x256xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x96x128x256xf16>) -> tensor<1x96x128x256xf16>
    func.func @main(%arg0: tensor<1x96x128x256xf16>) -> tensor<1x96x128x256xf16> {
        %cst = const.Declare tensor<3x32x32x3x3xf16> = dense<1.0> : tensor<3x32x32x3x3xf16>
        %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf16>
        %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf16>
        %cst_2 = const.Declare tensor<3x32x1x1x1xf16> = dense<-0.40> : tensor<3x32x1x1x1xf16>
        %cst_3 = const.Declare tensor<3x32x1x1x1xf16> = dense<0.40> : tensor<3x32x1x1x1xf16>
        %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>
        %cst_5 = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
        %cst_6 = const.Declare tensor<1x1x1x1xf16> = dense<-2.0> : tensor<1x1x1x1xf16>
        %cst_7 = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>

        %0 = IE.FakeQuantize(%arg0, %cst_4, %cst_5, %cst_6, %cst_7) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 256 : i64
        } : tensor<1x96x128x256xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x96x128x256xf16>

        %1 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_2, %cst_3) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 255 : i64
        } : tensor<3x32x32x3x3xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<3x32x1x1x1xf16>, tensor<3x32x1x1x1xf16> -> tensor<3x32x32x3x3xf16>

        %2 = IE.GroupConvolution(%0, %1) {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } : tensor<1x96x128x256xf16>, tensor<3x32x32x3x3xf16> -> tensor<1x96x128x256xf16>

        return %2 : tensor<1x96x128x256xf16>

        // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<96x96x3x3x!qElemType, {order = #NHWC}> = dense_resource<__elided__> : tensor<96x96x3x3xf16>, [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType>, #const.Reorder<#NHWC>]

        // CHECK:           [[PERMUTE_CAST0:%.+]] = IE.PermuteQuantize([[ARG0]])
        // CHECK-SAME:          {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
        // CHECK-SAME:          : tensor<1x96x128x256xf16> -> tensor<1x96x128x256xf16, {order = #NHWC}>

        // CHECK:           [[AVGPOOL:%.+]] = IE.AvgPool([[PERMUTE_CAST0]]) {
        // CHECK-SAME:          exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
        // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
        // CHECK-SAME:          : tensor<1x96x128x256xf16, {order = #NHWC}> -> tensor<1x96x128x256x!qElemType2, {order = #NHWC}>

        // CHECK:           [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[AVGPOOL]]) {dstElemType = !qElemType3}
        // CHECK-SAME:        : tensor<1x96x128x256x!qElemType2, {order = #NHWC}> -> tensor<1x96x128x256x!qElemType3, {order = #NHWC}>

        // CHECK:           [[CONV:%.+]] = IE.Convolution([[QUANTIZE_CAST]], [[CST]]) {
        // CHECK-SAME:          dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        // CHECK-SAME:        : tensor<1x96x128x256x!qElemType3, {order = #NHWC}>, tensor<96x96x3x3x!qElemType, {order = #NHWC}>
        // CHECK-SAME:        -> tensor<1x96x128x256xf16>
        // CHECK:        return [[CONV]] : tensor<1x96x128x256xf16>
    }
}
