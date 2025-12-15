//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-dequant-through-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @PerTensorConcatDequantAndConst
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x3x4x!qElemType>)
func.func @PerTensorConcatDequantAndConst(%input: tensor<1x2x3x4x!qElemType>) -> tensor<1x4x3x4xf16> {
    %cst = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    %dequant = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %concat = IE.Concat(%dequant, %cst) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    return %concat : tensor<1x4x3x4xf16>

    //CHECK: [[CONST:%.+]] = const.Declare tensor<1x2x3x4x!qElemType> = dense<0.000000e+00> : tensor<1x2x3x4xf16>, [#const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
    //CHECK: [[CONCAT:%.+]] = IE.Concat([[INPUT]], [[CONST]])
    //CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[CONCAT]])
    //CHECK: return [[DEQUANTIZE]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @PerTensorConcatDequantAndTwoConsts
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x3x4x!qElemType>)
func.func @PerTensorConcatDequantAndTwoConsts(%input: tensor<1x2x3x4x!qElemType>) -> tensor<1x6x3x4xf16> {
    %cst1 = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    %cst2 = const.Declare tensor<1x2x3x4xf16> = dense<1.000000e+00> :  tensor<1x2x3x4xf16>
    %dequant = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %concat = IE.Concat(%dequant, %cst1, %cst2) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x6x3x4xf16>
    return %concat : tensor<1x6x3x4xf16>

    //CHECK-DAG: [[CONST1:%.+]] = const.Declare tensor<1x2x3x4x!qElemType> = dense<0.000000e+00> : tensor<1x2x3x4xf16>, [#const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
    //CHECK-DAG: [[CONST2:%.+]] = const.Declare tensor<1x2x3x4x!qElemType> = dense<1.000000e+00> : tensor<1x2x3x4xf16>, [#const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
    //CHECK: [[CONCAT:%.+]] = IE.Concat([[INPUT]], [[CONST1]], [[CONST2]])
    //CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[CONCAT]])
    //CHECK: return [[DEQUANTIZE]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @PerTensorConcatDequantAndTwoConstsOrderPreserved
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x3x4x!qElemType>)
func.func @PerTensorConcatDequantAndTwoConstsOrderPreserved(%input: tensor<1x2x3x4x!qElemType>) -> tensor<1x6x3x4xf16> {
    %cst1 = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    %cst2 = const.Declare tensor<1x2x3x4xf16> = dense<1.000000e+00> :  tensor<1x2x3x4xf16>
    %dequant = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %concat = IE.Concat(%cst1, %dequant, %cst2) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x6x3x4xf16>
    return %concat : tensor<1x6x3x4xf16>

    //CHECK-DAG: [[CONST1:%.+]] = const.Declare tensor<1x2x3x4x!qElemType> = dense<0.000000e+00> : tensor<1x2x3x4xf16>, [#const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
    //CHECK-DAG: [[CONST2:%.+]] = const.Declare tensor<1x2x3x4x!qElemType> = dense<1.000000e+00> : tensor<1x2x3x4xf16>, [#const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
    //CHECK: [[CONCAT:%.+]] = IE.Concat([[CONST1]], [[INPUT]], [[CONST2]])
    //CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[CONCAT]])
    //CHECK: return [[DEQUANTIZE]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @PerTensorTwoConsecutiveConcats
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x3x4x!qElemType>)
func.func @PerTensorTwoConsecutiveConcats(%input: tensor<1x2x3x4x!qElemType>) -> tensor<1x2x6x8xf16> {
    %cst1 = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    %cst2 = const.Declare tensor<1x2x6x4xf16> = dense<1.000000e+00> :  tensor<1x2x6x4xf16>
    %dequant = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %concat1 = IE.Concat(%dequant, %cst1) {per_axis = #IE.Concat<axis = 2>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x2x6x4xf16>
    %concat2 = IE.Concat(%concat1, %cst2) {per_axis = #IE.Concat<axis = 3>} : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16> -> tensor<1x2x6x8xf16>
    return %concat2 : tensor<1x2x6x8xf16>

    //CHECK-DAG: [[CONST1:%.+]] = const.Declare tensor<1x2x3x4x!qElemType> = dense<0.000000e+00> : tensor<1x2x3x4xf16>, [#const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
    //CHECK-DAG: [[CONST2:%.+]] = const.Declare tensor<1x2x6x4x!qElemType> = dense<1.000000e+00> : tensor<1x2x6x4xf16>, [#const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
    //CHECK: [[CONCAT1:%.+]] = IE.Concat([[INPUT]], [[CONST1]])
    //CHECK: [[CONCAT2:%.+]] = IE.Concat([[CONCAT1]], [[CONST2]])
    //CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[CONCAT2]])
    //CHECK: return [[DEQUANTIZE]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @NoPropagationPerTensorConcatDequantAndConstTwoUsers
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x3x4x!qElemType>)
func.func @NoPropagationPerTensorConcatDequantAndConstTwoUsers(%input: tensor<1x2x3x4x!qElemType>) -> tensor<1x4x3x4xf16> {
    %cst = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    %dequant = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %concat = IE.Concat(%dequant, %cst) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    %relu1 = IE.ReLU(%concat) : tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    %relu2 = IE.ReLU(%concat) : tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    %add = IE.Add(%relu1, %relu2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x3x4xf16>, tensor<1x4x3x4xf16> -> tensor<1x4x3x4xf16>
    return %add : tensor<1x4x3x4xf16>

    //CHECK: [[CONST:%.+]] = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> : tensor<1x2x3x4xf16>
    //CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[INPUT]])
    //CHECK: [[CONCAT:%.+]] = IE.Concat([[DEQUANTIZE]], [[CONST]])
    //CHECK: [[RELU1:%.+]] = IE.ReLU([[CONCAT]])
    //CHECK: [[RELU2:%.+]] = IE.ReLU([[CONCAT]])
    //CHECK: [[ADD:%.+]] = IE.Add([[RELU1]], [[RELU2]])
    //CHECK: return [[ADD]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16, 2.0000000000000000E-1>

// CHECK-LABEL: @NoPropagationPerTensorConcatTwoDequantsAndConst
// CHECK-SAME: ([[INPUT1:%.+]]: tensor<1x2x3x4x!qElemType>, [[INPUT2:%.+]]: tensor<1x2x3x4x!qElemType1>)
func.func @NoPropagationPerTensorConcatTwoDequantsAndConst(%input1: tensor<1x2x3x4x!qElemType>, %input2: tensor<1x2x3x4x!qElemType1>) -> tensor<1x6x3x4xf16> {
    %cst = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    %dequant1 = IE.Dequantize(%input1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %dequant2 = IE.Dequantize(%input2) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %concat = IE.Concat(%dequant1, %dequant2, %cst) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x6x3x4xf16>
    return %concat : tensor<1x6x3x4xf16>

    //CHECK: [[CONST:%.+]] = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    //CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[INPUT1]])
    //CHECK: [[DEQUANTIZE2:%.+]] = IE.Dequantize([[INPUT2]])
    //CHECK: [[CONCAT:%.+]] = IE.Concat([[DEQUANTIZE1]], [[DEQUANTIZE2]], [[CONST]])
    //CHECK: return [[CONCAT]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @NoPropagationPerAxisConcatDequantAndConst
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x3x4x!qElemType>)
func.func @NoPropagationPerAxisConcatDequantAndConst(%input: tensor<1x2x3x4x!qElemType>) -> tensor<1x4x3x4xf16> {
    %cst = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    %dequant = IE.Dequantize(%input) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %concat = IE.Concat(%dequant, %cst) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>
    return %concat : tensor<1x4x3x4xf16>

    //CHECK: [[CONST:%.+]] = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    //CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[INPUT]])
    //CHECK: [[CONCAT:%.+]] = IE.Concat([[DEQUANTIZE]], [[CONST]])
    //CHECK: return [[CONCAT]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @NotPropagateDueToAllInputsAreTensors
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x2x3x4xf16>, [[INPUT1:%.+]]: tensor<1x2x3x4x!qElemType>)
func.func @NotPropagateDueToAllInputsAreTensors(%input0: tensor<1x2x3x4xf16>, %input1: tensor<1x2x3x4x!qElemType>) -> tensor<1x4x3x4xf16> {
    %relu1 = IE.ReLU(%input0) : tensor<1x2x3x4xf16> -> tensor<1x2x3x4xf16>
    %dequant = IE.Dequantize(%input1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %concat = IE.Concat(%dequant, %relu1) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x4x3x4xf16>

    return %concat : tensor<1x4x3x4xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
!qElemType1 = !quant.uniform<u8:f16:1, {1.0000000000000000E-1, 2.0000000000000000E-1}>

// CHECK-LABEL: @NoPropagationMixedQuantConcatTwoDequantsAndConst
// CHECK-SAME: ([[INPUT1:%.+]]: tensor<1x2x3x4x!qElemType>, [[INPUT2:%.+]]: tensor<1x2x3x4x!qElemType1>)
func.func @NoPropagationMixedQuantConcatTwoDequantsAndConst(%input1: tensor<1x2x3x4x!qElemType>, %input2: tensor<1x2x3x4x!qElemType1>) -> tensor<1x6x3x4xf16> {
    %cst = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    %dequant1 = IE.Dequantize(%input1) {dstElemType = f16} : tensor<1x2x3x4x!qElemType> -> tensor<1x2x3x4xf16>
    %dequant2 = IE.Dequantize(%input2) {dstElemType = f16} : tensor<1x2x3x4x!qElemType1> -> tensor<1x2x3x4xf16>
    %concat = IE.Concat(%dequant1, %dequant2, %cst) {per_axis = #IE.Concat<axis = 1>} : tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16>, tensor<1x2x3x4xf16> -> tensor<1x6x3x4xf16>
    return %concat : tensor<1x6x3x4xf16>

    //CHECK: [[CONST:%.+]] = const.Declare tensor<1x2x3x4xf16> = dense<0.000000e+00> :  tensor<1x2x3x4xf16>
    //CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[INPUT1]])
    //CHECK: [[DEQUANTIZE2:%.+]] = IE.Dequantize([[INPUT2]])
    //CHECK: [[CONCAT:%.+]] = IE.Concat([[DEQUANTIZE1]], [[DEQUANTIZE2]], [[CONST]])
    //CHECK: return [[CONCAT]]
}
