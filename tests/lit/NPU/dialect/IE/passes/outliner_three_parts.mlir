//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --outliner="function-outlining=\"naive='num-parts=3'\"" --canonicalize %s | FileCheck %s
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
// CHECK:   return [[ADD]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part3(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFT:%.+]] = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[PART3:%.+]] = call @main_part3([[PART2]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[PART3]] : tensor<1x48x60x60xf32>
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
// CHECK:   return [[ADD]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part3([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]], [[ARG1]]) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[PART3:%.+]] = call @main_part3([[PART2]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[PART3]] : tensor<1x48x60x60xf32>
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
// CHECK:   return [[ADD]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part3([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[PART1:%.+]]:2 = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]]#1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[PART3:%.+]] = call @main_part3([[PART2]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[PART3]], [[PART1]]#0 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
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

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[ADD]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part3([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[PART3:%.+]] = call @main_part3([[PART2]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[PART3]], [[PART2]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

//
// -----
//

module @MultipleInputsMultipleOutputsThreeOutputsInSlice {

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

// CHECK-LABEL: @MultipleInputsMultipleOutputsThreeOutputsInSlice

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

// CHECK: func.func private @main_part2([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[ADD3:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[ADD1]], [[ADD2]], [[ADD3]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func private @main_part3([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFT:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFT]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf32>, [[ARG1:%.+]]: tensor<1x3x62x62xf32>, [[ARG2:%.+]]: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[PART1:%.+]]:2 = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   [[PART2:%.+]]:3 = call @main_part2([[PART1]]#1, [[PART1]]#0) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   [[PART3:%.+]] = call @main_part3([[PART2]]#0) : (tensor<1x48x60x60xf32>
// CHECK:   return [[PART1]]#0, [[PART1]]#1, [[PART2]]#0, [[PART2]]#1, [[PART2]]#2, [[PART3]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }
