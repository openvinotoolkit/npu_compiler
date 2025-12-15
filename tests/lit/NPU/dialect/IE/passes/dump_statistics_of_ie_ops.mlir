//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env OV_NPU_LOG_LEVEL=LOG_INFO env IE_NPU_LOG_FILTER=dump-statistics-of-ie-ops vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dump-statistics-of-ie-ops -o /dev/null %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 1.0:128>

module @NonComputationalOps {
    net.NetworkInfo entryPoint : @init inputsInfo : {
        DataInfo "vpux_ow_1" : tensor<2x1xui8>
        DataInfo "vpux_ow_2" : tensor<2x1xf16>
    } outputsInfo : {
        DataInfo "vpux_tw_1" : tensor<2x1xui8>
        DataInfo "vpux_tw_2" : tensor<1x2xf16>
    }

    func.func @init(%ov1: tensor<2x1xui8>, %ov2: tensor<2x1xf16>) -> (tensor<2x1xui8>, tensor<1x2xf16>) {
        %0 = IE.QuantizeCast(%ov1) {dstElemType = !qElemType} : tensor<2x1xui8> -> tensor<2x1x!qElemType>
        %res0 = IE.QuantizeCast(%0) {dstElemType = ui8} : tensor<2x1x!qElemType> -> tensor<2x1xui8>
        %res1 = IE.Reshape(%ov2) {shape_value = [1, 2]} : tensor<2x1xf16> -> tensor<1x2xf16>
        return %res0, %res1 : tensor<2x1xui8>, tensor<1x2xf16>
    }

    // CHECK:   IE dialect statistics:
    // CHECK:   IE - 3
    // CHECK:     Non-computational - 3 (100.00%)
    // CHECK:       IE.QuantizeCast - 2 (66.67%)
    // CHECK:       IE.Reshape - 1 (33.33%)
}

// -----

module @ComputationalOps {
    net.NetworkInfo entryPoint : @init inputsInfo : {
        DataInfo "vpux_ow_1" : tensor<2x1xf32>
        DataInfo "vpux_ow_2" : tensor<2x1xf16>
    } outputsInfo : {
        DataInfo "vpux_tw_1" : tensor<10x1xf16>
        DataInfo "vpux_tw_2" : tensor<2x1xf16>
    }

    func.func @init(%ov1: tensor<2x1xf32>, %ov2: tensor<2x1xf16>) -> (tensor<10x1xf16>, tensor<2x1xf16>) {
        %0 = IE.Convert(%ov1) {dstElemType = f16} : tensor<2x1xf32> -> tensor<2x1xf16>
        %fourty_two = const.Declare tensor<1xf16> = dense<42.0> : tensor<1xf16>
        %1 = IE.Add(%0, %fourty_two) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<2x1xf16>, tensor<1xf16> -> tensor<2x1xf16>
        %res0 = IE.Pad(%1) {
            mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.0 : f64, pads_begin_attr = [0, 0],
            pads_end_attr = [8, 0]
        } : tensor<2x1xf16> -> tensor<10x1xf16>
        %two = const.Declare tensor<1xf16> = dense<2.0> : tensor<1xf16>
        %res1 = IE.Divide(%ov2, %two) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<2x1xf16>, tensor<1xf16> -> tensor<2x1xf16>
        return %res0, %res1 : tensor<10x1xf16>, tensor<2x1xf16>
    }

    // CHECK:   IE dialect statistics:
    // CHECK:   IE - 4
    // CHECK:     Computational - 4 (100.00%)
    // CHECK:       IE.Convert - 1 (25.00%)
    // CHECK:         f32 -> f16 - 1 (25.00%)
    // CHECK:       IE.Add - 1 (25.00%)
    // CHECK:       IE.Divide - 1 (25.00%)
    // CHECK:       IE.Pad - 1 (25.00%)
}

// -----

!qElemType1 = !quant.uniform<i8:f16, 0.8359>
!qElemType2 = !quant.uniform<u8:f16, 0.8359:128>

module @WeightsSeparation_InterQuantizedConvert {
    net.NetworkInfo entryPoint : @init inputsInfo : {
        DataInfo "vpux_ow_1" : tensor<2x1xf16>
    } outputsInfo : {
        DataInfo "vpux_tw_1" : tensor<2x1xui8>
    }

    func.func @init(%ov1: tensor<2x1xf16>) -> tensor<2x1xui8> {
        %1 = IE.Convert(%ov1) {dstElemType = i8} : tensor<2x1xf16> -> tensor<2x1xi8>
        %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType1} : tensor<2x1xi8> -> tensor<2x1x!qElemType1>
        %3 = IE.Reshape(%2) {shape_value = [1, 1, 2, 1]} : tensor<2x1x!qElemType1> -> tensor<1x1x2x1x!qElemType1>
        %4 = IE.AvgPool(%3) {
            exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
        } : tensor<1x1x2x1x!qElemType1> -> tensor<1x1x2x1x!qElemType2>
        %5 = IE.Reshape(%4) {shape_value = [2, 1]} : tensor<1x1x2x1x!qElemType2> -> tensor<2x1x!qElemType2>
        %res = IE.QuantizeCast(%5) {dstElemType = ui8} : tensor<2x1x!qElemType2> -> tensor<2x1xui8>
        return %res : tensor<2x1xui8>
    }

    // CHECK:   IE dialect statistics:
    // CHECK:   IE - 6
    // CHECK:     Non-computational - 4 (66.67%)
    // CHECK:       IE.QuantizeCast - 2 (33.33%)
    // CHECK:       IE.Reshape - 2 (33.33%)
    // CHECK:     Computational - 2 (33.33%)
    // CHECK:       IE.Convert - 1 (16.67%)
    // CHECK:         f16 -> i8 - 1 (16.67%)
    // CHECK:       IE.AvgPool - 1 (16.67%)
    // CHECK:         qtype<i8:f16, 0.836:0> -> qtype<ui8:f16, 0.836:128> - 1 (16.67%)
}

// -----

// same scales, same zero points
!qElemType_simple = !quant.uniform<i8:f16:0, {0.8359:42,0.8359:42}>
// scales = {0.84, 1.5}
!qElemType_diff_scales = !quant.uniform<i8:f16:0, {0.84,1.5}>
// zero points = {1, 42}
!qElemType_diff_zps = !quant.uniform<i8:f16:0, {0.8359:42,0.8359:-42}>

module @PerAxisConvert {
    net.NetworkInfo entryPoint : @init inputsInfo : {
        DataInfo "vpux_ow_1" : tensor<2x1xf16>
    } outputsInfo : {
        DataInfo "out1" : tensor<2x1xi8>
        DataInfo "out2" : tensor<2x1xi8>
        DataInfo "out3" : tensor<2x1xi8>
    }

    func.func @init(%ov1: tensor<2x1xf16>) -> (tensor<2x1xi8>, tensor<2x1xi8>, tensor<2x1xi8>) {
        %1 = IE.Convert(%ov1) {dstElemType = i8} : tensor<2x1xf16> -> tensor<2x1xi8>

        %simple = IE.Convert(%1) {dstElemType = !qElemType_simple}
            : tensor<2x1xi8> -> tensor<2x1x!qElemType_simple>
        %res0 = IE.QuantizeCast(%simple) {dstElemType = i8}
            : tensor<2x1x!qElemType_simple> -> tensor<2x1xi8>

        %diff_scales = IE.Convert(%1) {dstElemType = !qElemType_diff_scales}
            : tensor<2x1xi8> -> tensor<2x1x!qElemType_diff_scales>
        %res1 = IE.QuantizeCast(%diff_scales) {dstElemType = i8}
            : tensor<2x1x!qElemType_diff_scales> -> tensor<2x1xi8>

        %diff_zps = IE.Convert(%1) {dstElemType = !qElemType_diff_zps}
            : tensor<2x1xi8> -> tensor<2x1x!qElemType_diff_zps>
        %res2 = IE.QuantizeCast(%diff_zps) {dstElemType = i8}
            : tensor<2x1x!qElemType_diff_zps> -> tensor<2x1xi8>

        return %res0, %res1, %res2 : tensor<2x1xi8>, tensor<2x1xi8>, tensor<2x1xi8>
    }

    // CHECK:   IE dialect statistics:
    // CHECK:   IE - 7
    // CHECK:     Non-computational - 3 (42.86%)
    // CHECK:       IE.QuantizeCast - 3 (42.86%)
    // CHECK:     Computational - 4 (57.14%)
    // CHECK:       IE.Convert - 4 (57.14%)
    // CHECK:         f16 -> i8 - 1 (14.29%)
    // CHECK:         i8 -> qtype<i8:f16:0, per-axis> - 3 (42.86%)
}
