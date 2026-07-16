//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --outliner="function-outlining=\"repeating-blocks='min-ops-in-block=2 max-num-iterations=10 weights-as-inputs=false'\"" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module @TwoInstances {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %softmax = IE.SoftMax(%input) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %cst_weights1 = const.Declare tensor<48x48x3x3xf32> = dense<1.0> : tensor<48x48x3x3xf32>
        %conv1 = IE.Convolution(%softmax, %cst_weights1) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%conv1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %cst_weights2 = const.Declare tensor<48x48x3x3xf32> = dense<2.0> : tensor<48x48x3x3xf32>
        %conv2 = IE.Convolution(%relu1, %cst_weights2) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%conv2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %relu2: tensor<1x48x60x60xf32>
    }
}

// CHECK-LABEL: @TwoInstances

// CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
// CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

// CHECK: func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RELU:%.+]] = IE.ReLU([[CONV]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RELU]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func nested @main_rest1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFTMAX]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFTMAX:%.+]] = call @main_rest1([[INPUT]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[CALL1:%.+]] = call @main_fn1([[SOFTMAX]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[CALL2]] : tensor<1x48x60x60xf32>
// CHECK: }

// -----

module @TwoInstancesSameConstant {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %softmax = IE.SoftMax(%input) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %cst_weights = const.Declare tensor<48x48x3x3xf32> = dense<1.0> : tensor<48x48x3x3xf32>
        %conv1 = IE.Convolution(%softmax, %cst_weights) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%conv1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %conv2 = IE.Convolution(%relu1, %cst_weights) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%conv2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %relu2: tensor<1x48x60x60xf32>
    }
}

// CHECK-LABEL: @TwoInstancesSameConstant

// CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
// CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

// CHECK: func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RELU:%.+]] = IE.ReLU([[CONV]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RELU]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func nested @main_rest1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFTMAX]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFTMAX:%.+]] = call @main_rest1([[INPUT]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[CALL1:%.+]] = call @main_fn1([[SOFTMAX]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[CALL2]] : tensor<1x48x60x60xf32>
// CHECK: }
