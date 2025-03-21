//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --outliner="function-outlining=\"batching=''\"" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
module @OneInputOneOutput {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<3x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<3x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<3x3x62x62xf32>) -> tensor<3x48x60x60xf32> {
        %0 = builtin.unrealized_conversion_cast %arg0 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %1 = IE.Convolution(%0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %3 = IE.Add(%2, %2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %4 = IE.SoftMax(%3) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %5 = builtin.unrealized_conversion_cast %4: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        return %5: tensor<3x48x60x60xf32>
    }
}

// CHECK-LABEL: @OneInputOneOutput

// CHECK: DataInfo "input" : tensor<3x3x62x62xf16>

// CHECK: DataInfo "output" : tensor<3x48x60x60xf16>

// CHECK: func.func private @main_batching1([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[ADD:%.+]] = IE.Add([[SOFT]], [[SOFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RET_SOFT:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RET_SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<3x3x62x62xf32>) -> tensor<3x48x60x60xf32> {
// CHECK:   [[VAL0:%0]] = builtin.unrealized_conversion_cast [[ARG0]] : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
// CHECK:   [[PART:%.+]] = call @main_batching1([[VAL0]]) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[VAL1:%.+]] = builtin.unrealized_conversion_cast [[PART]] : tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
// CHECK:   return [[VAL1]] : tensor<3x48x60x60xf32>
// CHECK: }

//
// -----
//

module @MultipleInputsOneOutput {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<3x3x62x62xf32>
        DataInfo "input1" : tensor<3x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<3x48x60x60xf32>
    }

    func.func @main(%arg0: tensor<3x3x62x62xf32>, %arg1: tensor<3x48x60x60xf32>) -> tensor<3x48x60x60xf32> {
        %0 = builtin.unrealized_conversion_cast %arg0 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %1 = builtin.unrealized_conversion_cast %arg1 : tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %2 = IE.Convolution(%0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %4 = IE.Add(%3, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %5 = IE.SoftMax(%4) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %6 = builtin.unrealized_conversion_cast %5: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        return %6: tensor<3x48x60x60xf32>
    }
}

// CHECK-LABEL: @MultipleInputsOneOutput

// CHECK: DataInfo "input0" : tensor<3x3x62x62xf32>
// CHECK: DataInfo "input1" : tensor<3x48x60x60xf32>

// CHECK: DataInfo "output" : tensor<3x48x60x60xf32>

// CHECK: func.func private @main_batching1([[ARG0:%.+]]: tensor<1x3x62x62xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[ADD:%.+]] = IE.Add([[SOFT]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RET_SOFT:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RET_SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main(%arg0: tensor<3x3x62x62xf32>, %arg1: tensor<3x48x60x60xf32>) -> tensor<3x48x60x60xf32> {
// CHECK:   [[VAL0:%.+]] = builtin.unrealized_conversion_cast %arg0 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
// CHECK:   [[VAL1:%.+]] = builtin.unrealized_conversion_cast %arg1 : tensor<3x48x60x60xf32> to tensor<1x48x60x60xf32>
// CHECK:   [[FUNC_RES:%.+]] = call @main_batching1([[VAL0]], [[VAL1]]) : (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[RET:%.+]] = builtin.unrealized_conversion_cast [[FUNC_RES]] : tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
// CHECK:   return [[RET]] : tensor<3x48x60x60xf32>
// CHECK: }

//
// -----
//

module @OneInputMultipleOutputsFirstSlice {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<3x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output0" : tensor<3x48x60x60xf16>
        DataInfo "output1" : tensor<1x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<3x3x62x62xf32>) -> (tensor<3x48x60x60xf32>, tensor<1x48x60x60xf32>) {
        %0 = builtin.unrealized_conversion_cast %arg0 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %1 = IE.Convolution(%0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %3 = IE.Add(%2, %2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %4 = IE.SoftMax(%3) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %5 = builtin.unrealized_conversion_cast %4: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        return %5, %1: tensor<3x48x60x60xf32>, tensor<1x48x60x60xf32>
    }
}

// CHECK-LABEL: @OneInputMultipleOutputsFirstSlice

// CHECK: DataInfo "input" : tensor<3x3x62x62xf16>

// CHECK: DataInfo "output0" : tensor<3x48x60x60xf16>
// CHECK: DataInfo "output1" : tensor<1x48x60x60xf16>

// CHECK: func.func private @main_batching1([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[ADD:%.+]] = IE.Add([[SOFT]], [[SOFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RET_SOFT:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[CONV]], [[RET_SOFT]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<3x3x62x62xf32>) -> (tensor<3x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[VAL0:%0]] = builtin.unrealized_conversion_cast [[ARG0]] : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
// CHECK:   [[PART:%.+]]:2 = call @main_batching1([[VAL0]]) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   [[VAL1:%.+]] = builtin.unrealized_conversion_cast [[PART]]#1 : tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
// CHECK:   return [[VAL1]], [[PART]]#0 : tensor<3x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

//
// -----
//

module @OneInputMultipleOutputsLastSlice {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<3x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output0" : tensor<3x48x60x60xf16>
        DataInfo "output1" : tensor<3x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<3x3x62x62xf32>) -> (tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>) {
        %0 = builtin.unrealized_conversion_cast %arg0 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %1 = IE.Convolution(%0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %2 = IE.SoftMax(%1) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %3 = IE.Add(%2, %2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %4 = IE.SoftMax(%3) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %5 = builtin.unrealized_conversion_cast %4: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        %6 = builtin.unrealized_conversion_cast %3: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        return %5, %6: tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>
    }
}

// CHECK-LABEL: @OneInputMultipleOutputsLastSlice

// CHECK: DataInfo "input" : tensor<3x3x62x62xf16>

// CHECK: DataInfo "output0" : tensor<3x48x60x60xf16>
// CHECK: DataInfo "output1" : tensor<3x48x60x60xf16>

// CHECK: func.func private @main_batching1([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RET_ADD:%.+]] = IE.Add([[SOFT]], [[SOFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RET_SOFT:%.+]] = IE.SoftMax([[RET_ADD]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RET_SOFT]], [[RET_ADD]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<3x3x62x62xf32>) -> (tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>) {
// CHECK:   [[VAL0:%0]] = builtin.unrealized_conversion_cast [[ARG0]] : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
// CHECK:   [[PART:%.+]]:2 = call @main_batching1([[VAL0]]) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   [[VAL1:%.+]] = builtin.unrealized_conversion_cast [[PART]]#0 : tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
// CHECK:   [[VAL2:%.+]] = builtin.unrealized_conversion_cast [[PART]]#1 : tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
// CHECK:   return [[VAL1]], [[VAL2]] : tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>
// CHECK: }

//
// -----
//

module @MultipleInputsMultipleOutputs {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<3x3x62x62xf16>
        DataInfo "input1" : tensor<3x3x62x62xf16>
        DataInfo "input2" : tensor<3x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output0" : tensor<3x48x60x60xf16>
        DataInfo "output1" : tensor<3x48x60x60xf16>
        DataInfo "output2" : tensor<3x48x60x60xf16>
        DataInfo "output3" : tensor<3x48x60x60xf16>
        DataInfo "output4" : tensor<3x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<3x3x62x62xf32>, %arg1: tensor<3x3x62x62xf32>, %arg2: tensor<3x3x62x62xf32>) -> (tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>) {
        %0 = builtin.unrealized_conversion_cast %arg0 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %1 = builtin.unrealized_conversion_cast %arg1 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %2 = builtin.unrealized_conversion_cast %arg2 : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %3 = IE.Convolution(%0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %4 = IE.SoftMax(%3) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %5 = IE.Add(%4, %3) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %6 = IE.Add(%4, %4) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %7 = IE.SoftMax(%5) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %8 = builtin.unrealized_conversion_cast %3: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        %9 = builtin.unrealized_conversion_cast %4: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        %10 = builtin.unrealized_conversion_cast %5: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        %11 = builtin.unrealized_conversion_cast %6: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        %12 = builtin.unrealized_conversion_cast %7: tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
        return %8, %9, %10, %11, %8: tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>
    }
}

// CHECK-LABEL: @MultipleInputsMultipleOutputs

// CHECK: DataInfo "input0" : tensor<3x3x62x62xf16>
// CHECK: DataInfo "input1" : tensor<3x3x62x62xf16>
// CHECK: DataInfo "input2" : tensor<3x3x62x62xf16>

// CHECK: DataInfo "output0" : tensor<3x48x60x60xf16>
// CHECK: DataInfo "output1" : tensor<3x48x60x60xf16>
// CHECK: DataInfo "output2" : tensor<3x48x60x60xf16>
// CHECK: DataInfo "output3" : tensor<3x48x60x60xf16>
// CHECK: DataInfo "output4" : tensor<3x48x60x60xf16>

// CHECK: func.func private @main_batching1([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RET_ADD0:%.+]] = IE.Add([[SOFT]], [[CONV]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RET_ADD1:%.+]] = IE.Add([[SOFT]], [[SOFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RET_SOFT:%.+]] = IE.SoftMax([[RET_ADD0]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[CONV]], [[SOFT]], [[RET_ADD0]], [[RET_ADD1]], [[RET_SOFT]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<3x3x62x62xf32>, [[ARG1:%.+]]: tensor<3x3x62x62xf32>, [[ARG2:%.+]]: tensor<3x3x62x62xf32>) -> (tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>) {
// CHECK:   [[VAL0:%0]] = builtin.unrealized_conversion_cast [[ARG0]] : tensor<3x3x62x62xf32> to tensor<1x3x62x62xf32>
// CHECK:   [[PART:%.+]]:5 = call @main_batching1([[VAL0]]) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   [[VAL11:%.+]] = builtin.unrealized_conversion_cast [[PART]]#0 : tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
// CHECK:   [[VAL22:%.+]] = builtin.unrealized_conversion_cast [[PART]]#1 : tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
// CHECK:   [[VAL33:%.+]] = builtin.unrealized_conversion_cast [[PART]]#2 : tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
// CHECK:   [[VAL44:%.+]] = builtin.unrealized_conversion_cast [[PART]]#3 : tensor<1x48x60x60xf32> to tensor<3x48x60x60xf32>
// CHECK:   return [[VAL11]], [[VAL22]], [[VAL33]], [[VAL44]], [[VAL11]] : tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>, tensor<3x48x60x60xf32>
// CHECK: }
