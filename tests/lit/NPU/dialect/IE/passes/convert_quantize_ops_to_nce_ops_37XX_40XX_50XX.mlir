//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-quantize-ops-to-nce-ops  %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.013961060374390846:109>
!dynType = tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
!dynQuantType = tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.027922120748781691:109>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 0.013961060374390846:109>

// CHECK-LABEL:  func.func @DynamicShapeToAdd
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>)
func.func @DynamicShapeToAdd(%arg0 : !dynType) -> !dynType {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : !dynType -> !dynQuantType
    %1 = IE.Dequantize(%0) {dstElemType = f16} : !dynQuantType -> !dynType
    return %1 : !dynType

    // CHECK:       [[Q_ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME:    tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[Q_CAST:%.+]] = IE.QuantizeCast([[Q_ADD]]) {dstElemType = !qElemType1} :
    // CHECK-SAME:    tensor<1x1x?x2x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2x!qElemType1, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[D_AVG:%.+]] = IE.AvgPool([[Q_CAST]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} :
    // CHECK-SAME:    tensor<1x1x?x2x!qElemType1, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:  return [[D_AVG]] : tensor<1x1x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 500, 2]> : tensor<4xsi64>, order = #NCHW}>
}
