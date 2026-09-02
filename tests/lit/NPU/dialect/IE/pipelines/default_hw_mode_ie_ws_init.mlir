//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --default-hw-mode-ie="enable-adjust-precision-pipeline=false verify-locations=off" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// This test validates that the enable-adjust-precision-pipeline option is not missing.
// At the same time can help catch other possible errors in the WS Init function.

!qElemType = !quant.uniform<i8:f16:0, {0.0011582261791416243,0.0016320897083656461,0.0020050289584141153,0.0031276913250193874}>
!qElemType1 = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f16, 1.000000e+00:128>
!qElemType3 = !quant.uniform<u8:f16:0, {0.0011582261791416243:128,0.0016320897083656461:128,0.0020050289584141153:128,0.0031276913250193874:128}>

module @onnx_Frontend_IR_init_0 attributes {VPU.WsTotalInitPartCount = 1 : i64} {
  net.NetworkInfo entryPoint : @init inputsInfo : {
    DataInfo "vpux_ow_0" : tensor<4x4x1x1xsi8>
    DataInfo "vpux_ow_1" : tensor<4x4x1x1xsi8>
  } outputsInfo : {
    DataInfo "vpux_tw_0_concat" : tensor<32xi8>
  }
  func.func @init(%init_cst0: tensor<4x4x1x1xsi8>, %init_cst1: tensor<4x4x1x1xsi8>) -> tensor<32xi8> {
    %convert_to_f16_0 = IE.Convert(%init_cst0) {dstElemType = f16} : tensor<4x4x1x1xsi8> -> tensor<4x4x1x1xf16>
    %convert_to_i8_0 = IE.Convert(%convert_to_f16_0) {dstElemType = si8} : tensor<4x4x1x1xf16> -> tensor<4x4x1x1xsi8>
    %quant_cast_in_0 = IE.QuantizeCast(%convert_to_i8_0) {dstElemType = !qElemType} : tensor<4x4x1x1xsi8> -> tensor<4x4x1x1x!qElemType>
    %cast_to_i8_0 = IE.QuantizeCast(%quant_cast_in_0) {dstElemType = si8} : tensor<4x4x1x1x!qElemType> -> tensor<4x4x1x1xsi8>
    %quant_cast_pool_in_0 = IE.QuantizeCast(%cast_to_i8_0) {dstElemType = !qElemType1} : tensor<4x4x1x1xsi8> -> tensor<4x4x1x1x!qElemType1>
    %avg_pool_0 = IE.AvgPool(%quant_cast_pool_in_0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<4x4x1x1x!qElemType1> -> tensor<4x4x1x1x!qElemType2>
    %cast_to_u8_0 = IE.QuantizeCast(%avg_pool_0) {dstElemType = ui8} : tensor<4x4x1x1x!qElemType2> -> tensor<4x4x1x1xui8>
    %quant_cast_u8_0 = IE.QuantizeCast(%cast_to_u8_0) {dstElemType = !qElemType3} : tensor<4x4x1x1xui8> -> tensor<4x4x1x1x!qElemType3>
    %cast_out_0 = IE.QuantizeCast(%quant_cast_u8_0) {dstElemType = ui8} : tensor<4x4x1x1x!qElemType3> -> tensor<4x4x1x1xui8>

    %convert_to_f16_1 = IE.Convert(%init_cst1) {dstElemType = f16} : tensor<4x4x1x1xsi8> -> tensor<4x4x1x1xf16>
    %convert_to_i8_1 = IE.Convert(%convert_to_f16_1) {dstElemType = si8} : tensor<4x4x1x1xf16> -> tensor<4x4x1x1xsi8>
    %quant_cast_in_1 = IE.QuantizeCast(%convert_to_i8_1) {dstElemType = !qElemType} : tensor<4x4x1x1xsi8> -> tensor<4x4x1x1x!qElemType>
    %cast_to_i8_1 = IE.QuantizeCast(%quant_cast_in_1) {dstElemType = si8} : tensor<4x4x1x1x!qElemType> -> tensor<4x4x1x1xsi8>
    %quant_cast_pool_in_1 = IE.QuantizeCast(%cast_to_i8_1) {dstElemType = !qElemType1} : tensor<4x4x1x1xsi8> -> tensor<4x4x1x1x!qElemType1>
    %avg_pool_1 = IE.AvgPool(%quant_cast_pool_in_1) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<4x4x1x1x!qElemType1> -> tensor<4x4x1x1x!qElemType2>
    %cast_to_u8_1 = IE.QuantizeCast(%avg_pool_1) {dstElemType = ui8} : tensor<4x4x1x1x!qElemType2> -> tensor<4x4x1x1xui8>
    %quant_cast_u8_1 = IE.QuantizeCast(%cast_to_u8_1) {dstElemType = !qElemType3} : tensor<4x4x1x1xui8> -> tensor<4x4x1x1x!qElemType3>
    %cast_out_1 = IE.QuantizeCast(%quant_cast_u8_1) {dstElemType = ui8} : tensor<4x4x1x1x!qElemType3> -> tensor<4x4x1x1xui8>

    %reinterpret_0 = Core.ReinterpretCast(%cast_out_0) : tensor<4x4x1x1xui8> -> tensor<16xi8>
    %reinterpret_1 = Core.ReinterpretCast(%cast_out_1) : tensor<4x4x1x1xui8> -> tensor<16xi8>
    %concat = IE.Concat(%reinterpret_0, %reinterpret_1) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<16xi8>, tensor<16xi8> -> tensor<32xi8>
    return %concat : tensor<32xi8>
  }
}

// CHECK: [[$qElemTypeIn:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: [[$qElemTypeOut:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: func.func @init
// CHECK-SAME:      ([[ARG0:%.+]]: tensor<4x4x1x1xsi8>, [[ARG1:%.+]]: tensor<4x4x1x1xsi8>) -> tensor<32xi8>

// CHECK:       [[QC0:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = [[$qElemTypeIn]]} : tensor<4x4x1x1xsi8> -> tensor<4x4x1x1x[[$qElemTypeIn]]>
// CHECK:       [[PC0:%.+]] = IE.PermuteCast([[QC0]]) {dst_order = #NHWC, mem_perm = #NHWC}
// CHECK:       [[SC0:%.+]] = IE.ShapeCast {shape = [1, 16, 1, 1]} inputs([[PC0]]
// CHECK:       [[POOL0:%.+]] = IE.AvgPool([[SC0]])
// CHECK-SAME:      : tensor<1x16x1x1x[[$qElemTypeIn]], {order = #NHWC}> -> tensor<1x16x1x1x[[$qElemTypeOut]], {order = #NHWC}>
// CHECK:       [[SC0OUT:%.+]] = IE.ShapeCast {shape = [4, 4, 1, 1]} inputs([[POOL0]]
// CHECK:       [[QC0OUT:%.+]] = IE.QuantizeCast([[SC0OUT]]) {dstElemType = ui8}
// CHECK:       [[PC0OUT:%.+]] = IE.PermuteCast([[QC0OUT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x4x1x1xui8, {order = #NHWC}> -> tensor<4x4x1x1xui8>

// CHECK:       [[QC1:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = [[$qElemTypeIn]]} : tensor<4x4x1x1xsi8> -> tensor<4x4x1x1x[[$qElemTypeIn]]>
// CHECK:       [[PC1:%.+]] = IE.PermuteCast([[QC1]]) {dst_order = #NHWC, mem_perm = #NHWC}
// CHECK:       [[SC1:%.+]] = IE.ShapeCast {shape = [1, 16, 1, 1]} inputs([[PC1]]
// CHECK:       [[POOL1:%.+]] = IE.AvgPool([[SC1]])
// CHECK-SAME:      : tensor<1x16x1x1x[[$qElemTypeIn]], {order = #NHWC}> -> tensor<1x16x1x1x[[$qElemTypeOut]], {order = #NHWC}>
// CHECK:       [[SC1OUT:%.+]] = IE.ShapeCast {shape = [4, 4, 1, 1]} inputs([[POOL1]]
// CHECK:       [[QC1OUT:%.+]] = IE.QuantizeCast([[SC1OUT]]) {dstElemType = ui8}
// CHECK:       [[PC1OUT:%.+]] = IE.PermuteCast([[QC1OUT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<4x4x1x1xui8, {order = #NHWC}> -> tensor<4x4x1x1xui8>

// CHECK:       [[RC0:%.+]] = Core.ReinterpretCast([[PC0OUT]]) : tensor<4x4x1x1xui8> -> tensor<16xi8>
// CHECK:       [[RC1:%.+]] = Core.ReinterpretCast([[PC1OUT]]) : tensor<4x4x1x1xui8> -> tensor<16xi8>
// If the enable-adjust-precision-pipeline were missing, we would see lots of Convert operations around this Concat.
// CHECK:       [[CONCAT:%.+]] = IE.Concat([[RC0]], [[RC1]]) {static_offsets = {{\[\[}}0], [16]]} : tensor<16xi8>, tensor<16xi8> -> tensor<32xi8>
// CHECK:       return [[CONCAT]] : tensor<32xi8>
