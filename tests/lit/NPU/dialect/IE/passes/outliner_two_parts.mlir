//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --outliner="function-outlining=\"naive='num-parts=2'\"" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

module @OneInputOneOutput {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %2 = IE.Add(%1, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %3: tensor<1x48x60x60xf32>
    }
}

// CHECK-LABEL: @OneInputOneOutput

// CHECK: DataInfo "input" : tensor<1x3x62x62xf16>

// CHECK: DataInfo "output" : tensor<1x48x60x60xf16>

// CHECK: func.func private @main_part1([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[PART2]] : tensor<1x48x60x60xf32>
// CHECK: }

//
// -----
//

module @MultipleInputsOneOutput {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x3x62x62xf16>
        DataInfo "input1" : tensor<1x48x60x60xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<1x3x62x62xf32>, %arg1: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %2 = IE.Add(%1, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %3: tensor<1x48x60x60xf32>
    }
}

// CHECK-LABEL: @MultipleInputsOneOutput

// CHECK: DataInfo "input0" : tensor<1x3x62x62xf16>
// CHECK: DataInfo "input1" : tensor<1x48x60x60xf16>

// CHECK: DataInfo "output" : tensor<1x48x60x60xf16>

// CHECK: func.func private @main_part1([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]], [[ARG1]]) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[PART2]] : tensor<1x48x60x60xf32>
// CHECK: }

//
// -----
//

module @OneInputMultipleOutputsFirstSlice {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output0" : tensor<1x48x60x60xf16>
        DataInfo "output1" : tensor<1x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %2 = IE.Add(%1, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %3, %0: tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    }
}

// CHECK-LABEL: @OneInputMultipleOutputsFirstSlice

// CHECK: DataInfo "input" : tensor<1x3x62x62xf16>

// CHECK: DataInfo "output0" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output1" : tensor<1x48x60x60xf16>

// CHECK: func.func private @main_part1([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[CONV]], [[SOFT]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[PART1:%.+]]:2 = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]]#1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[PART2]], [[PART1]]#0 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

//
// -----
//

module @OneInputMultipleOutputsLastSlice {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output0" : tensor<1x48x60x60xf16>
        DataInfo "output1" : tensor<1x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %2 = IE.Add(%1, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %3, %2: tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    }
}

// CHECK-LABEL: @OneInputMultipleOutputsLastSlice

// CHECK: DataInfo "input" : tensor<1x3x62x62xf16>

// CHECK: DataInfo "output0" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output1" : tensor<1x48x60x60xf16>

// CHECK: func.func private @main_part1([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[ADD]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[ADD]], [[SOFT]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[PART2:%.+]]:2 = call @main_part2([[PART1]]) : (tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   return [[PART2]]#1, [[PART2]]#0 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

//
// -----
//

module @MultipleInputsMultipleOutputs {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x3x62x62xf16>
        DataInfo "input1" : tensor<1x3x62x62xf16>
        DataInfo "input2" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output0" : tensor<1x48x60x60xf16>
        DataInfo "output1" : tensor<1x48x60x60xf16>
        DataInfo "output2" : tensor<1x48x60x60xf16>
        DataInfo "output3" : tensor<1x48x60x60xf16>
        DataInfo "output4" : tensor<1x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<1x3x62x62xf32>, %arg1: tensor<1x3x62x62xf32>, %arg2: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %2 = IE.Add(%1, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %3 = IE.Add(%1, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %4 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %0, %1, %2, %3, %0: tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    }
}

// CHECK-LABEL: @MultipleInputsMultipleOutputs

// CHECK: DataInfo "input0" : tensor<1x3x62x62xf16>
// CHECK: DataInfo "input1" : tensor<1x3x62x62xf16>
// CHECK: DataInfo "input2" : tensor<1x3x62x62xf16>

// CHECK: DataInfo "output0" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output1" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output2" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output3" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output4" : tensor<1x48x60x60xf16>

// CHECK: func.func private @main_part1([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[CONV]], [[SOFT]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[ADD1]], [[ADD2]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>, [[ARG1:%.+]]: tensor<1x3x62x62xf32>, [[ARG2:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[PART1:%.+]]:2 = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   [[PART2:%.+]]:2 = call @main_part2([[PART1]]#1, [[PART1]]#0) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   return [[PART1]]#0, [[PART1]]#1, [[PART2]]#0, [[PART2]]#1, [[PART1]]#0 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

//
// -----
//

module @MultipleInputsMultipleOutputsFourOutputsInSlice {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x3x62x62xf16>
        DataInfo "input1" : tensor<1x3x62x62xf16>
        DataInfo "input2" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output0" : tensor<1x48x60x60xf16>
        DataInfo "output1" : tensor<1x48x60x60xf16>
        DataInfo "output2" : tensor<1x48x60x60xf16>
        DataInfo "output3" : tensor<1x48x60x60xf16>
        DataInfo "output4" : tensor<1x48x60x60xf16>
        DataInfo "output5" : tensor<1x48x60x60xf16>
    }

    func.func @main(%arg0: tensor<1x3x62x62xf32>, %arg1: tensor<1x3x62x62xf32>, %arg2: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %2 = IE.Add(%1, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %3 = IE.Add(%1, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %4 = IE.Add(%1, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %5 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %0, %1, %2, %3, %4, %5: tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    }
}

// CHECK-LABEL: @MultipleInputsMultipleOutputsFourOutputsInSlice

// CHECK: DataInfo "input0" : tensor<1x3x62x62xf16>
// CHECK: DataInfo "input1" : tensor<1x3x62x62xf16>
// CHECK: DataInfo "input2" : tensor<1x3x62x62xf16>

// CHECK: DataInfo "output0" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output1" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output2" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output3" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output4" : tensor<1x48x60x60xf16>
// CHECK: DataInfo "output5" : tensor<1x48x60x60xf16>

// CHECK: func.func private @main_part1([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[CONV]], [[SOFT]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[ADD3:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[ADD1]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[ADD1]], [[ADD2]], [[ADD3]], [[SOFT]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>, [[ARG1:%.+]]: tensor<1x3x62x62xf32>, [[ARG2:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[PART1:%.+]]:2 = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   [[PART2:%.+]]:4 = call @main_part2([[PART1]]#1, [[PART1]]#0) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   return [[PART1]]#0, [[PART1]]#1, [[PART2]]#0, [[PART2]]#1, [[PART2]]#2, [[PART2]]#3 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

//
// -----
//

module @SharedConstant {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x300x300xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x300x300xf32>
    }

    func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
        %filter = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32>
            %conv1 = IE.Convolution(%input, %filter) {
                strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
            } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>
        %maxpool = IE.MaxPool(%conv1) {
            kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
        } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
        %conv2 = IE.Convolution(%maxpool, %filter) {
            strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
        } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>
        return %conv2 : tensor<1x3x300x300xf32>
    }
}

// CHECK-LABEL: @SharedConstant

// CHECK: DataInfo "input" : tensor<1x3x300x300xf32>

// CHECK: DataInfo "output" : tensor<1x3x300x300xf32>

// CHECK: func.func private @main_part1([[INPUT:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<3x3x3x3xf32> = dense<1.000000e+00> : tensor<3x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>
// CHECK:   [[MAXPOOL:%.+]] = IE.MaxPool([[CONV]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
// CHECK:   return [[MAXPOOL]] : tensor<1x3x300x300xf32>
// CHECK: }

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<3x3x3x3xf32> = dense<1.000000e+00> : tensor<3x3x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0:%.+]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>
// CHECK:   return [[CONV]] : tensor<1x3x300x300xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
// CHECK:   [[PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
// CHECK:   return [[PART2]] : tensor<1x3x300x300xf32>
// CHECK: }

//
// -----
//

module @OneInputOneOutputWithPreprocessOp {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x62x62xf32>
    }

    func.func @main(%input: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
        %cst = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32>
        %convert = IE.Convert(%input) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32>
        %conv = IE.Convolution(%convert, %cst) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x62x62xf32>
        %soft_max = IE.SoftMax(%conv) {axisInd = 1} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>

        %add = IE.Add(%soft_max, %convert) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x62x62xf32>, tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>

        return %add: tensor<1x3x62x62xf32>
    }
}

// CHECK-LABEL: @OneInputOneOutputWithPreprocessOp

// CHECK: DataInfo "input" : tensor<1x3x62x62xf16>

// CHECK: DataInfo "output" : tensor<1x3x62x62xf32>

// CHECK: func.func private @main_part1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<3x3x3x3xf32> = dense<1.000000e+00> : tensor<3x3x3x3xf32>
// CHECK:   [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[CONVERT]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x62x62xf32>
// CHECK:   return [[CONV]] : tensor<1x3x62x62xf32>
// CHECK: }

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x3x62x62xf32>, [[ARG1:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>
// CHECK:   [[CONVERT:%.+]] = IE.Convert([[ARG1]]) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32>
// CHECK:   [[ADD:%.+]] = IE.Add([[SOFT]], [[CONVERT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x62x62xf32>, tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>
// CHECK:   return [[ADD]] : tensor<1x3x62x62xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
// CHECK:   [[PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32>
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]], [[ARG0]]) : (tensor<1x3x62x62xf32>, tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32>
// CHECK:   return [[PART2]] : tensor<1x3x62x62xf32>
// CHECK: }

//
// -----
//

module @DontOutlineFunc {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x62x62xf32>
    }

    func.func @main(%input: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
        %cst = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32>
        %transpose_const1 = const.Declare tensor<4xsi64> = dense<[0, 1, 3, 2]> : tensor<4xsi64>

        %transpose1 = IE.Transpose(%input, %transpose_const1) : tensor<1x3x62x62xf16>, tensor<4xsi64> -> tensor<1x3x62x62xf16>
        %convert = IE.Convert(%transpose1) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32>
        %transpose2 = IE.Transpose(%convert, %transpose_const1) : tensor<1x3x62x62xf32>, tensor<4xsi64> -> tensor<1x3x62x62xf32>

        %conv = IE.Convolution(%transpose2, %cst) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x62x62xf32>
        %soft_max = IE.SoftMax(%conv) {axisInd = 1} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>

        %add = IE.Add(%soft_max, %transpose2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x62x62xf32>, tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>

        return %add: tensor<1x3x62x62xf32>
    }
}

// CHECK-LABEL: @DontOutlineFunc

// CHECK: DataInfo "input" : tensor<1x3x62x62xf16>

// CHECK: DataInfo "output" : tensor<1x3x62x62xf32>

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<3x3x3x3xf32> = dense<1.000000e+00> : tensor<3x3x3x3xf32>
// CHECK:   [[TRANSPOSE_1:%.+]] = IE.Transpose([[ARG0]]) {order_value = #NCWH} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16>
// CHECK:   [[CONVERT:%.+]] = IE.Convert([[TRANSPOSE_1]]) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32>
// CHECK:   [[TRANSPOSE_2:%.+]] = IE.Transpose([[CONVERT]]) {order_value = #NCWH} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[TRANSPOSE_2]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x62x62xf32>
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[CONV]]) {axisInd = 1 : i64} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>
// CHECK:   [[ADD:%.+]] = IE.Add([[SOFT]], [[TRANSPOSE_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x62x62xf32>, tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>
// CHECK:   return [[ADD]] : tensor<1x3x62x62xf32>
// CHECK: }

//
// -----
//

module @OneInputOneOutputWithMultiplePreprocessOps {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x62x62xf32>
    }

    func.func @main(%input: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
        %cst = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32>
        %cst_0 = const.Declare tensor<4xsi64> = dense<[0, 1, 3, 2]> : tensor<4xsi64>
        %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32>
        %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
        %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-2.0> : tensor<1x1x1x1xf32>
        %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>

        %transpose = IE.Transpose(%input, %cst_0) : tensor<1x3x62x62xf16>, tensor<4xsi64> -> tensor<1x3x62x62xf16>
        %convert = IE.Convert(%transpose) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32>
        %fake_quant = IE.FakeQuantize(%convert, %cst_1, %cst_2, %cst_3, %cst_4)
            {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                levels = 256 : i64
            } :
            tensor<1x3x62x62xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x62x62xf32>

        %conv = IE.Convolution(%fake_quant, %cst) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x62x62xf32>
        %soft_max1 = IE.SoftMax(%conv) {axisInd = 1} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>
        %soft_max2 = IE.SoftMax(%soft_max1) {axisInd = 1} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>

        %add = IE.Add(%soft_max2, %fake_quant) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x62x62xf32>, tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>

        return %add: tensor<1x3x62x62xf32>
    }
}

// CHECK-LABEL: @OneInputOneOutputWithMultiplePreprocessOps

// CHECK: DataInfo "input" : tensor<1x3x62x62xf16>

// CHECK: DataInfo "output" : tensor<1x3x62x62xf32>

// CHECK: func.func private @main_part1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<3x3x3x3xf32> = dense<1.000000e+00> : tensor<3x3x3x3xf32>
// CHECK:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
// CHECK:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1x1xf32>
// CHECK:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
// CHECK:   [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
// CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]]) {order_value = #NCWH} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16>
// CHECK:   [[CONVERT:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32>
// CHECK:   [[FAKE_QUANT:%.+]] = IE.FakeQuantize([[CONVERT]], [[CST_3]], [[CST_2]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x62x62xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x62x62xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[FAKE_QUANT]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x62x62xf32>
// CHECK:   return [[CONV]] : tensor<1x3x62x62xf32>
// CHECK: }

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x3x62x62xf32>, [[ARG1:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
// CHECK:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1x1xf32>
// CHECK:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
// CHECK:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
// CHECK:   [[SOFT1:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>
// CHECK:   [[SOFT2:%.+]] = IE.SoftMax([[SOFT1]]) {axisInd = 1 : i64} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>
// CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[ARG1]]) {order_value = #NCWH} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16>
// CHECK:   [[CONVERT:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32>
// CHECK:   [[FAKE_QUANT:%.+]] = IE.FakeQuantize([[CONVERT]], [[CST_2]], [[CST_1]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x62x62xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x62x62xf32>
// CHECK:   [[ADD:%.+]] = IE.Add([[SOFT2]], [[FAKE_QUANT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x62x62xf32>, tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32>
// CHECK:   return [[ADD]] : tensor<1x3x62x62xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
// CHECK:   [[PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32>
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]], [[ARG0]]) : (tensor<1x3x62x62xf32>, tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32>
// CHECK:   return [[PART2]] : tensor<1x3x62x62xf32>
// CHECK: }
