//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --low-precision %s | FileCheck %s
// REQUIRES: platform-NPU6010


!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u16:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<u16:f16, 1.000000e+00>
// CHECK-LABEL: @FuseAddU8InputsU16Output
module @FuseAddU8InputsU16Output {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "lhs" : tensor<1x16x4x4xui8>
        DataInfo "rhs" : tensor<1x16x4x4xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x4x4xui16>
    }

    // CHECK: func.func @main([[LHS:%.+]]: tensor<1x16x4x4x!qElemType>, [[RHS:%.+]]: tensor<1x16x4x4x!qElemType>)
    func.func @main(%lhs: tensor<1x16x4x4x!qElemType>, %rhs: tensor<1x16x4x4x!qElemType>)
            -> tensor<1x16x4x4x!qElemType1> {
        %lhsDequant = IE.Dequantize(%lhs) {dstElemType = f16} :
                tensor<1x16x4x4x!qElemType> -> tensor<1x16x4x4xf16>
        %rhsDequant = IE.Dequantize(%rhs) {dstElemType = f16} :
                tensor<1x16x4x4x!qElemType> -> tensor<1x16x4x4xf16>
        %add = IE.Add(%lhsDequant, %rhsDequant) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
        %quant = IE.Quantize(%add) {dstElemType = !qElemType1} :
                tensor<1x16x4x4xf16> -> tensor<1x16x4x4x!qElemType1>
        return %quant : tensor<1x16x4x4x!qElemType1>

        // CHECK-NEXT: [[ADD:%.+]] = IE.Add([[LHS]], [[RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        // CHECK-SAME: tensor<1x16x4x4x!qElemType>, tensor<1x16x4x4x!qElemType> -> tensor<1x16x4x4x!qElemType1>
        // CHECK-NEXT: return [[ADD]] : tensor<1x16x4x4x!qElemType1>
    }
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0039215688593685627>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.0039215686274509803>
// CHECK: !qElemType2 = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK: !qElemType3 = !quant.uniform<u8:f16, 0.0080296076980291634>
// CHECK: !qElemType4 = !quant.uniform<u8:f16, 0.0040148038490145817>
// CHECK: !qElemType5 = !quant.uniform<u8:f16, 0.0020074019245072908>

// TODO: Investigate and resolve issue before allowing fusion E#234073
// CHECK-LABEL: @DoNotFuseQuantizeAndActivationOps
// CHECK-SAME:      ([[ARG0:%.+]]: tensor<1x2x1x1xf16>) -> tensor<1x2x1x1xf16>
func.func @DoNotFuseQuantizeAndActivationOps(%arg0: tensor<1x2x1x1xf16>) -> tensor<1x2x1x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<2x2x1x1xf16> = dense<[[[[0.000000e+00]], [[2.550000e+02]]], [[[1.310000e+02]], [[1.290000e+02]]]]> : tensor<2x2x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<1.02377498> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_1, %cst_2, %cst_3, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<2x2x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<2x2x1x1xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_2, %cst_4, %cst_2, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x2x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x1xf16>
    %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x1x1xf16>, tensor<2x2x1x1xf16> -> tensor<1x2x1x1xf16>
    %3 = IE.FakeQuantize(%2, %cst_2, %cst_3, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x2x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x1xf16>
    %4 = IE.LeakyRelu(%3) {negative_slope = 0.20000000298023224 : f64} : tensor<1x2x1x1xf16> -> tensor<1x2x1x1xf16>
    %5 = IE.FakeQuantize(%4, %cst_2, %cst_4, %cst_2, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x2x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x1xf16>
    return %5 : tensor<1x2x1x1xf16>

    // CHECK:      [[CST:%.+]] = const.Declare tensor<2x2x1x1x!qElemType>
    // CHECK-SAME:    [#const.CastElemType<f16>, #const.Quantize<!qElemType1>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType>]
    // CHECK:      [[ADD0:%.+]] = IE.Add([[ARG0]], [[ARG0]])
    // CHECK:      [[QCAST1:%.+]] = IE.QuantizeCast([[ADD0]])
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[QCAST1]], [[CST]])
    // CHECK-SAME:   dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:   post_op = #IE.LeakyRelu<negative_slope = 0.20000000298023224 : f64>, strides = [1, 1]
    // CHECK-SAME: } : tensor<1x2x1x1x!qElemType4>, tensor<2x2x1x1x!qElemType> -> tensor<1x2x1x1x!qElemType4>
    // CHECK:      [[QCAST2:%.+]] = IE.QuantizeCast([[CONV]]) {dstElemType = !qElemType5} : tensor<1x2x1x1x!qElemType4> -> tensor<1x2x1x1x!qElemType5>
    // CHECK:      [[ADD1:%.+]] = IE.Add([[QCAST2]], [[QCAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x1x1x!qElemType5>, tensor<1x2x1x1x!qElemType5> -> tensor<1x2x1x1xf16>
    // CHECK:      return [[ADD1]] : tensor<1x2x1x1xf16>
}
