//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --outliner="function-outlining=\"repeating-blocks-separate-functions='min-ops-in-block=2 max-num-iterations=10'\"" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

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

    // CHECK-LABEL: @TwoInstances

    // CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK: func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[CST1:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK:   [[CONV1:%.+]] = IE.Convolution([[ARG0]], [[CST1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[RELU1:%.+]] = IE.ReLU([[CONV1]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[RELU1]] : tensor<1x48x60x60xf32>
    // CHECK: }

    // CHECK: func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[CST2:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<2.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK:   [[CONV2:%.+]] = IE.Convolution([[ARG0]], [[CST2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[RELU2:%.+]] = IE.ReLU([[CONV2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[RELU2]] : tensor<1x48x60x60xf32>
    // CHECK: }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK1:%.+]] = call @main_fn1_block1([[SOFTMAX]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   return [[BLOCK2]] : tensor<1x48x60x60xf32>
    // CHECK: }
}

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

    // CHECK-LABEL: @TwoInstancesSameConstant

    // CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK: func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[CST1:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK:   [[CONV1:%.+]] = IE.Convolution([[ARG0]], [[CST1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[RELU1:%.+]] = IE.ReLU([[CONV1]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[RELU1]] : tensor<1x48x60x60xf32>
    // CHECK: }

    // CHECK: func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[CST2:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK:   [[CONV2:%.+]] = IE.Convolution([[ARG0]], [[CST2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[RELU2:%.+]] = IE.ReLU([[CONV2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[RELU2]] : tensor<1x48x60x60xf32>
    // CHECK: }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK1:%.+]] = call @main_fn1_block1([[SOFTMAX]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   return [[BLOCK2]] : tensor<1x48x60x60xf32>
    // CHECK: }
}

// -----

module @TwoInstancesQuantizedWeights {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %softmax = IE.SoftMax(%input) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %cst_weights1 = const.Declare tensor<48x48x3x3xf32> = dense<1.0> : tensor<48x48x3x3xf32>
        %cst_weights1_sub_value = const.Declare tensor<48x1x1x1xf32> = dense<1.1> : tensor<48x1x1x1xf32>
        %cst_weights1_sub = IE.Subtract(%cst_weights1, %cst_weights1_sub_value) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
        %cst_weights1_mul_value = const.Declare tensor<48x1x1x1xf32> = dense<1.2> : tensor<48x1x1x1xf32>
        %cst_weights1_mul = IE.Multiply(%cst_weights1_sub, %cst_weights1_mul_value) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
        %conv1 = IE.Convolution(%softmax, %cst_weights1_mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%conv1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %cst_weights2 = const.Declare tensor<48x48x3x3xf32> = dense<2.0> : tensor<48x48x3x3xf32>
        %cst_weights2_sub_value = const.Declare tensor<48x1x1x1xf32> = dense<2.1> : tensor<48x1x1x1xf32>
        %cst_weights2_sub = IE.Subtract(%cst_weights2, %cst_weights2_sub_value) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
        %cst_weights2_mul_value = const.Declare tensor<48x1x1x1xf32> = dense<2.2> : tensor<48x1x1x1xf32>
        %cst_weights2_mul = IE.Multiply(%cst_weights2_sub, %cst_weights2_mul_value) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
        %conv2 = IE.Convolution(%relu1, %cst_weights2_mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%conv2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %relu2: tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @TwoInstancesQuantizedWeights

    // CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK:     func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:   [[CST_WEIGHTS1:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK-DAG:   [[CST_WEIGHTS1_SUB_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.100000e+00> : tensor<48x1x1x1xf32>
    // CHECK-DAG:   [[CST_WEIGHTS1_MUL_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.200000e+00> : tensor<48x1x1x1xf32>
    // CHECK:       [[SUB1:%.+]] = IE.Subtract([[CST_WEIGHTS1]], [[CST_WEIGHTS1_SUB_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[MUL1:%.+]] = IE.Multiply([[SUB1]], [[CST_WEIGHTS1_MUL_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[ARG0]], [[MUL1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       [[RELU1:%.+]] = IE.ReLU([[CONV1]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       return [[RELU1]] : tensor<1x48x60x60xf32>
    // CHECK:     }

    // CHECK:     func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:   [[CST_WEIGHTS2:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<2.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK-DAG:   [[CST_WEIGHTS2_SUB_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<2.100000e+00> : tensor<48x1x1x1xf32>
    // CHECK-DAG:   [[CST_WEIGHTS2_MUL_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<2.200000e+00> : tensor<48x1x1x1xf32>
    // CHECK:       [[SUB2:%.+]] = IE.Subtract([[CST_WEIGHTS2]], [[CST_WEIGHTS2_SUB_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[MUL2:%.+]] = IE.Multiply([[SUB2]], [[CST_WEIGHTS2_MUL_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[CONV2:%.+]] = IE.Convolution([[ARG0]], [[MUL2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       [[RELU2:%.+]] = IE.ReLU([[CONV2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       return [[RELU2]] : tensor<1x48x60x60xf32>
    // CHECK:     }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK1:%.+]] = call @main_fn1_block1([[SOFTMAX]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   return [[BLOCK2]] : tensor<1x48x60x60xf32>
    // CHECK: }
}

// -----

module @TwoInstancesQuantizedWeightsSameConstants {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %softmax = IE.SoftMax(%input) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %cst_weights = const.Declare tensor<48x48x3x3xf32> = dense<1.0> : tensor<48x48x3x3xf32>
        %cst_weights_sub_value = const.Declare tensor<48x1x1x1xf32> = dense<1.1> : tensor<48x1x1x1xf32>
        %cst_weights_sub = IE.Subtract(%cst_weights, %cst_weights_sub_value) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
        %cst_weights_mul_value = const.Declare tensor<48x1x1x1xf32> = dense<1.2> : tensor<48x1x1x1xf32>
        %cst_weights_mul = IE.Multiply(%cst_weights_sub, %cst_weights_mul_value) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>

        %conv1 = IE.Convolution(%softmax, %cst_weights_mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%conv1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %conv2 = IE.Convolution(%relu1, %cst_weights_mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%conv2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %relu2: tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @TwoInstancesQuantizedWeightsSameConstants

    // CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK:     func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:   [[CST_WEIGHTS1:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK-DAG:   [[CST_WEIGHTS1_SUB_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.100000e+00> : tensor<48x1x1x1xf32>
    // CHECK-DAG:   [[CST_WEIGHTS1_MUL_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.200000e+00> : tensor<48x1x1x1xf32>
    // CHECK:       [[SUB1:%.+]] = IE.Subtract([[CST_WEIGHTS1]], [[CST_WEIGHTS1_SUB_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[MUL1:%.+]] = IE.Multiply([[SUB1]], [[CST_WEIGHTS1_MUL_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[ARG0]], [[MUL1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       [[RELU1:%.+]] = IE.ReLU([[CONV1]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       return [[RELU1]] : tensor<1x48x60x60xf32>
    // CHECK:     }

    // CHECK:     func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:   [[CST_WEIGHTS2:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK-DAG:   [[CST_WEIGHTS2_SUB_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.100000e+00> : tensor<48x1x1x1xf32>
    // CHECK-DAG:   [[CST_WEIGHTS2_MUL_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.200000e+00> : tensor<48x1x1x1xf32>
    // CHECK:       [[SUB2:%.+]] = IE.Subtract([[CST_WEIGHTS2]], [[CST_WEIGHTS2_SUB_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[MUL2:%.+]] = IE.Multiply([[SUB2]], [[CST_WEIGHTS2_MUL_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[CONV2:%.+]] = IE.Convolution([[ARG0]], [[MUL2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       [[RELU2:%.+]] = IE.ReLU([[CONV2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       return [[RELU2]] : tensor<1x48x60x60xf32>
    // CHECK:     }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK1:%.+]] = call @main_fn1_block1([[SOFTMAX]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   return [[BLOCK2]] : tensor<1x48x60x60xf32>
    // CHECK: }
}

// -----

module @TwoInstancesQuantizedIntermediateChain {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %cst_weights1 = const.Declare tensor<48x48x1x9xf32> = dense<1.0> : tensor<48x48x1x9xf32>
        %cst_weights1_mul_value = const.Declare tensor<48x1x1x1xf32> = dense<1.1> : tensor<48x1x1x1xf32>
        %cst_weights1_mul = IE.Multiply(%cst_weights1, %cst_weights1_mul_value) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x1x9xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x1x9xf32>
        %cst_weights1_reshape = IE.AffineReshape(%cst_weights1_mul) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [48, 48, 3, 3]} : tensor<48x48x1x9xf32> -> tensor<48x48x3x3xf32>
        %conv1 = IE.Convolution(%input, %cst_weights1_reshape) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%conv1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %cst_weights2 = const.Declare tensor<48x48x1x9xf32> = dense<2.0> : tensor<48x48x1x9xf32>
        %cst_weights2_mul_value = const.Declare tensor<48x1x1x1xf32> = dense<2.1> : tensor<48x1x1x1xf32>
        %cst_weights2_mul = IE.Multiply(%cst_weights2, %cst_weights2_mul_value) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x1x9xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x1x9xf32>
        %cst_weights2_reshape = IE.AffineReshape(%cst_weights2_mul) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [48, 48, 3, 3]} : tensor<48x48x1x9xf32> -> tensor<48x48x3x3xf32>
        %conv2 = IE.Convolution(%relu1, %cst_weights2_reshape) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%conv2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %relu2: tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @TwoInstancesQuantizedIntermediateChain

    // CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK:     func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:   [[CST_WEIGHTS1:%.+]] = const.Declare tensor<48x48x1x9xf32> = dense<1.000000e+00> : tensor<48x48x1x9xf32>
    // CHECK-DAG:   [[CST_WEIGHTS1_MUL_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.100000e+00> : tensor<48x1x1x1xf32>
    // CHECK:       [[MUL1:%.+]] = IE.Multiply([[CST_WEIGHTS1]], [[CST_WEIGHTS1_MUL_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x1x9xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x1x9xf32>
    // CHECK:       [[RESHAPE1:%.+]] = IE.AffineReshape([[MUL1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [48, 48, 3, 3]} : tensor<48x48x1x9xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[ARG0]], [[RESHAPE1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       [[RELU1:%.+]] = IE.ReLU([[CONV1]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       return [[RELU1]] : tensor<1x48x60x60xf32>
    // CHECK:     }

    // CHECK:     func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:   [[CST_WEIGHTS2:%.+]] = const.Declare tensor<48x48x1x9xf32> = dense<2.000000e+00> : tensor<48x48x1x9xf32>
    // CHECK-DAG:   [[CST_WEIGHTS2_MUL_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<2.100000e+00> : tensor<48x1x1x1xf32>
    // CHECK:       [[MUL2:%.+]] = IE.Multiply([[CST_WEIGHTS2]], [[CST_WEIGHTS2_MUL_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x1x9xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x1x9xf32>
    // CHECK:       [[RESHAPE2:%.+]] = IE.AffineReshape([[MUL2]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [48, 48, 3, 3]} : tensor<48x48x1x9xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[CONV2:%.+]] = IE.Convolution([[ARG0]], [[RESHAPE2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       [[RELU2:%.+]] = IE.ReLU([[CONV2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       return [[RELU2]] : tensor<1x48x60x60xf32>
    // CHECK:     }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[BLOCK1:%.+]] = call @main_fn1_block1([[INPUT]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   return [[BLOCK2]] : tensor<1x48x60x60xf32>
    // CHECK: }
}

// -----

module @TwoInstancesQuantizedWeightsAsInputs {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
        DataInfo "weights" : tensor<48x48x3x3xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>, %weights: tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32> {
        %softmax = IE.SoftMax(%input) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %cst_weights_sub_value = const.Declare tensor<48x1x1x1xf32> = dense<1.1> : tensor<48x1x1x1xf32>
        %cst_weights_sub = IE.Subtract(%weights, %cst_weights_sub_value) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
        %cst_weights_mul_value = const.Declare tensor<48x1x1x1xf32> = dense<1.2> : tensor<48x1x1x1xf32>
        %cst_weights_mul = IE.Multiply(%cst_weights_sub, %cst_weights_mul_value) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>

        %conv1 = IE.Convolution(%softmax, %cst_weights_mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%conv1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %conv2 = IE.Convolution(%relu1, %cst_weights_mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%conv2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %relu2: tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @TwoInstancesQuantizedWeightsAsInputs

    // CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK:     func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:   [[CST_WEIGHTS1_SUB_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.100000e+00> : tensor<48x1x1x1xf32>
    // CHECK-DAG:   [[CST_WEIGHTS1_MUL_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.200000e+00> : tensor<48x1x1x1xf32>
    // CHECK:       [[SUB1:%.+]] = IE.Subtract([[ARG1]], [[CST_WEIGHTS1_SUB_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[MUL1:%.+]] = IE.Multiply([[SUB1]], [[CST_WEIGHTS1_MUL_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[ARG0]], [[MUL1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       [[RELU1:%.+]] = IE.ReLU([[CONV1]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       return [[RELU1]] : tensor<1x48x60x60xf32>
    // CHECK:     }

    // CHECK:     func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:   [[CST_WEIGHTS2_SUB_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.100000e+00> : tensor<48x1x1x1xf32>
    // CHECK-DAG:   [[CST_WEIGHTS2_MUL_VALUE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.200000e+00> : tensor<48x1x1x1xf32>
    // CHECK:       [[SUB2:%.+]] = IE.Subtract([[ARG1]], [[CST_WEIGHTS2_SUB_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[MUL2:%.+]] = IE.Multiply([[SUB2]], [[CST_WEIGHTS2_MUL_VALUE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x1x1x1xf32> -> tensor<48x48x3x3xf32>
    // CHECK:       [[CONV2:%.+]] = IE.Convolution([[ARG0]], [[MUL2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       [[RELU2:%.+]] = IE.ReLU([[CONV2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:       return [[RELU2]] : tensor<1x48x60x60xf32>
    // CHECK:     }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>, [[WEIGHTS:%.+]]: tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK1:%.+]] = call @main_fn1_block1([[SOFTMAX]], [[WEIGHTS]]) : (tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]], [[WEIGHTS]]) : (tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   return [[BLOCK2]] : tensor<1x48x60x60xf32>
    // CHECK: }
}

// -----

module @TwoInstancesDifferentNumberOfOutputs {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x48x60x60xf32>
        DataInfo "input2" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input1: tensor<1x48x60x60xf32>, %input2: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %add1 = IE.Add(%input1, %input2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%add1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %add2 = IE.Add(%add1, %relu1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%add2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %relu2: tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @TwoInstancesDifferentNumberOfOutputs

    // CHECK: DataInfo "input1" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "input2" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK: func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
    // CHECK:   [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[RELU1:%.+]] = IE.ReLU([[ADD1]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[ADD1]], [[RELU1]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    // CHECK: }

    // CHECK: func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[RELU2:%.+]] = IE.ReLU([[ADD2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[RELU2]] : tensor<1x48x60x60xf32>
    // CHECK: }

    // CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x48x60x60xf32>, [[INPUT2:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[BLOCK1:%.+]]:2 = call @main_fn1_block1([[INPUT1]], [[INPUT2]]) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:   [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]]#0, [[BLOCK1]]#1) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   return [[BLOCK2]] : tensor<1x48x60x60xf32>
    // CHECK: }
}

// -----

module @FirstInstanceInputReuse {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %add1 = IE.Add(%input, %input) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%add1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %add2 = IE.Add(%add1, %relu1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%add2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %relu2: tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @FirstInstanceInputReuse

    // CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK:  func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
    // CHECK:      [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[RELU1:%.+]] = IE.ReLU([[ADD1]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      return [[ADD1]], [[RELU1]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    // CHECK:  }

    // CHECK:  func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:      [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[RELU2:%.+]] = IE.ReLU([[ADD2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      return [[RELU2]] : tensor<1x48x60x60xf32>
    // CHECK:  }

    // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:      [[BLOCK1:%.+]]:2 = call @main_fn1_block1([[INPUT]]) : (tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:      [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]]#0, [[BLOCK1]]#1) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:      return [[BLOCK2]] : tensor<1x48x60x60xf32>
    // CHECK:  }
}

// -----

module @InputReuseMultipleOps {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %add1 = IE.Add(%input, %input) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %add2 = IE.Add(%add1, %input) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%add2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %add3 = IE.Add(%add1, %relu1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %add4 = IE.Add(%add3, %relu1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%add4) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %relu2: tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @InputReuseMultipleOps

    // CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK:  func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
    // CHECK:      [[ADD1_1:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[ADD1_2:%.+]] = IE.Add([[ADD1_1]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[RELU1:%.+]] = IE.ReLU([[ADD1_2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      return [[ADD1_1]], [[RELU1]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    // CHECK:  }

    // CHECK:  func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:      [[ADD2_1:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[ADD2_2:%.+]] = IE.Add([[ADD2_1]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[RELU2:%.+]] = IE.ReLU([[ADD2_2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      return [[RELU2]] : tensor<1x48x60x60xf32>
    // CHECK:  }

    // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:      [[BLOCK1:%.+]]:2 = call @main_fn1_block1([[INPUT]]) : (tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:      [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]]#0, [[BLOCK1]]#1) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:      return [[BLOCK2]] : tensor<1x48x60x60xf32>
    // CHECK:  }
}

// -----

module @TwoRepeatingBlockTypes {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x300x300xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x300x300xf32>
    }

    func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
        %maxpool1 = IE.MaxPool(%input) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
        %avgpool1 = IE.AvgPool(%maxpool1) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
        %avgpool2 = IE.AvgPool(%avgpool1) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

        %softmax = IE.SoftMax(%avgpool2) {axisInd = -1} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

        %add1 = IE.Add(%softmax, %softmax) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
        %multiply1 = IE.Multiply(%add1, %add1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

        %maxpool2 = IE.MaxPool(%multiply1) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
        %avgpool3 = IE.AvgPool(%maxpool2) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
        %avgpool4 = IE.AvgPool(%avgpool3) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

        %add2 = IE.Add(%avgpool4, %avgpool4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
        %multiply2 = IE.Multiply(%add2, %add2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

        return %multiply2 : tensor<1x3x300x300xf32>
    }

    // CHECK-LABEL: @TwoRepeatingBlockTypes

    // CHECK: DataInfo "input" : tensor<1x3x300x300xf32>
    // CHECK: DataInfo "output" : tensor<1x3x300x300xf32>

    // CHECK:      func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[MAXPOOL1:%.+]] = IE.MaxPool([[ARG0]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[AVGPOOL1_1:%.+]] = IE.AvgPool([[MAXPOOL1]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[AVGPOOL1_2:%.+]] = IE.AvgPool([[AVGPOOL1_1]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[AVGPOOL1_2]] : tensor<1x3x300x300xf32>
    // CHECK:      }
    // CHECK:      func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[MAXPOOL2:%.+]] = IE.MaxPool([[ARG0]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:         : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[AVGPOOL2_1:%.+]] = IE.AvgPool([[MAXPOOL2]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:         : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[AVGPOOL2_2:%.+]] = IE.AvgPool([[AVGPOOL2_1]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:         : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return %2 : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func private @main_fn2_block1([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[MUL1:%.+]] = IE.Multiply([[ADD1]], [[ADD1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[MUL1]] : tensor<1x3x300x300xf32>
    // CHECK:      }
    // CHECK:      func.func private @main_fn2_block2([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[MUL2:%.+]] = IE.Multiply([[ADD2]], [[ADD2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[MUL2]] : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func @main([[INPUT:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[FN1_BLOCK1:%.+]] = call @main_fn1_block1([[INPUT]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          [[SOFTMAX:%.+]] = IE.SoftMax([[FN1_BLOCK1]]) {axisInd = 3 : i64} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[FN2_BLOCK1:%.+]] = call @main_fn2_block1([[SOFTMAX]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          [[FN1_BLOCK2:%.+]] = call @main_fn1_block2([[FN2_BLOCK1]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          [[FN2_BLOCK2:%.+]] = call @main_fn2_block2([[FN1_BLOCK2]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          return [[FN2_BLOCK2]] : tensor<1x3x300x300xf32>
    // CHECK:      }
}

// -----

module @MixedOperationsOrder {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x16x16xf32>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x64x16x16xf32>
        DataInfo "output2" : tensor<1x64x16x16xf32>
    }

    func.func @main(%input: tensor<1x64x16x16xf32>) -> (tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32>) {
        %add1 = IE.Add(%input, %input) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
        %add2 = IE.Add(%add1, %add1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
        %softmax = IE.SoftMax(%input) {axisInd = -1} : tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
        %sub1 = IE.Subtract(%add1, %softmax) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
        %mult1 = IE.Multiply(%sub1, %sub1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>

        %sub2 = IE.Subtract(%add2, %softmax) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
        %mul2 = IE.Multiply(%sub2, %sub2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>

        return %mult1, %mul2: tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32>
    }

    // CHECK-LABEL: @MixedOperationsOrder

    // CHECK: DataInfo "input" : tensor<1x64x16x16xf32>
    // CHECK: DataInfo "output1" : tensor<1x64x16x16xf32>
    // CHECK: DataInfo "output2" : tensor<1x64x16x16xf32>

    // CHECK: func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x64x16x16xf32>, [[ARG1:%.+]]: tensor<1x64x16x16xf32>) -> (tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32>) {
    // CHECK:   [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
    // CHECK:   [[SUB1:%.+]] = IE.Subtract([[ADD1]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
    // CHECK:   [[MUL1:%.+]] = IE.Multiply([[SUB1]], [[SUB1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
    // CHECK:   return [[ADD1]], [[MUL1]] : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32>
    // CHECK: }

    // CHECK: func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x64x16x16xf32>, [[ARG1:%.+]]: tensor<1x64x16x16xf32>) -> tensor<1x64x16x16xf32> {
    // CHECK:   [[ADD2:%.+]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
    // CHECK:   [[SUB2:%.+]] = IE.Subtract([[ADD2]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
    // CHECK:   [[MUL2:%.+]] = IE.Multiply([[SUB2]], [[SUB2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
    // CHECK:   return [[MUL2]] : tensor<1x64x16x16xf32>
    // CHECK: }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x64x16x16xf32>) -> (tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32>) {
    // CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 3 : i64} : tensor<1x64x16x16xf32> -> tensor<1x64x16x16xf32>
    // CHECK:   [[BLOCK1:%.+]]:2 = call @main_fn1_block1([[INPUT]], [[SOFTMAX]]) : (tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32>) -> (tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32>)
    // CHECK:   [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]]#0, [[SOFTMAX]]) : (tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32>) -> tensor<1x64x16x16xf32>
    // CHECK:   return [[BLOCK1]]#1, [[BLOCK2]] : tensor<1x64x16x16xf32>, tensor<1x64x16x16xf32>
    // CHECK: }
}

// -----

module @QuantizedSubgraph {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %softmax = IE.SoftMax(%input) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
        %input_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
        %output_low = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
        %output_high = const.Declare tensor<1x1x1x1xf32> = dense<254.0> : tensor<1x1x1x1xf32>
        %softmax_fq = IE.FakeQuantize(%softmax, %input_low, %input_high, %output_low, %output_high) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32
            } : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>

        %cst_weights1 = const.Declare tensor<48x48x3x3xf32> = dense<1.0> : tensor<48x48x3x3xf32>
        %cst_weights1_fq = IE.FakeQuantize(%cst_weights1, %input_low, %input_high, %output_low, %output_high) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32
            } : tensor<48x48x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<48x48x3x3xf32>
        %conv1 = IE.Convolution(%softmax_fq, %cst_weights1_fq) {
                dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
            } : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%conv1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu1_fq = IE.FakeQuantize(%relu1, %input_low, %input_high, %output_low, %output_high) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32
            } : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>

        %cst_weights2 = const.Declare tensor<48x48x3x3xf32> = dense<2.0> : tensor<48x48x3x3xf32>
        %cst_weights2_fq = IE.FakeQuantize(%cst_weights2, %input_low, %input_high, %output_low, %output_high) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32
            } : tensor<48x48x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<48x48x3x3xf32>
        %conv2 = IE.Convolution(%relu1_fq, %cst_weights2_fq) {
                dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
            } : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%conv2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu2_fq = IE.FakeQuantize(%relu2, %input_low, %input_high, %output_low, %output_high) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32
            } : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>

        return %relu2_fq: tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @QuantizedSubgraph

    // CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK:      func.func private @main_fn1_block1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:    [[INPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[INPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[OUTPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[OUTPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[CST_WEIGHTS1:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK-DAG:    [[FQ_WEIGHTS1:%.+]] = IE.FakeQuantize([[CST_WEIGHTS1]], [[INPUT_LOW]], [[INPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]])
    // CHECK:        [[CONV1:%.+]] = IE.Convolution([[ARG0]], [[FQ_WEIGHTS1]])
    // CHECK:        [[RELU1:%.+]] = IE.ReLU([[CONV1]])
    // CHECK:        [[FQ_RELU1:%.+]] = IE.FakeQuantize([[RELU1]], [[INPUT_LOW]], [[INPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]])
    // CHECK:        return [[FQ_RELU1]]
    // CHECK:      }

    // CHECK:      func.func private @main_fn1_block2([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:    [[INPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[INPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[OUTPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[OUTPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[CST_WEIGHTS2:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<2.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK-DAG:    [[FQ_WEIGHTS2:%.+]] = IE.FakeQuantize([[CST_WEIGHTS2]], [[INPUT_LOW]], [[INPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]])
    // CHECK:        [[CONV2:%.+]] = IE.Convolution([[ARG0]], [[FQ_WEIGHTS2]])
    // CHECK:        [[RELU2:%.+]] = IE.ReLU([[CONV2]])
    // CHECK:        [[FQ_RELU2:%.+]] = IE.FakeQuantize([[RELU2]], [[INPUT_LOW]], [[INPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]])
    // CHECK:        return [[FQ_RELU2]]
    // CHECK:      }

    // CHECK:      func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:    [[INPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[INPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[OUTPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:    [[OUTPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK:        [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:        [[FQ_SOFTMAX:%.+]] = IE.FakeQuantize([[SOFTMAX]], [[INPUT_LOW]], [[INPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]])
    // CHECK:        [[BLOCK1:%.+]] = call @main_fn1_block1([[FQ_SOFTMAX]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:        [[BLOCK2:%.+]] = call @main_fn1_block2([[BLOCK1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:        return [[BLOCK2]]
    // CHECK:      }
}

// -----

module @UnsupportedCyclicalPattern {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %softmax1 = IE.SoftMax(%input)  {axisInd = -1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        // Pattern leads to a cycle in the IR since the ReLU operation would end up as an input to the call operation while also having a result of the call op as an input
        //     %relu = IE.ReLU(%call)
        //     %call = call @fn(%input, %relu)
        // Same for the Sigmoid below
        %relu = IE.ReLU(%softmax1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %add1 = IE.Add(%softmax1, %relu) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %softmax2 = IE.SoftMax(%add1)  {axisInd = -1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %sigmoid = IE.Sigmoid(%softmax2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %add2 = IE.Add(%softmax2, %sigmoid) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %add2: tensor<1x48x60x60xf32>

        // CHECK-LABEL: @UnsupportedCyclicalPattern
        // CHECK:       func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        // CHECK-NOT:     call
    }
}

// -----

module @UnsupportedCyclicalPatternSubgraph {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %softmax1 = IE.SoftMax(%input)  {axisInd = -1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        // Pattern leads to a cycle in the IR since the ReLU+Add operations would end up as an input to the call operation while also having a result of the call op as an input
        //     %relu = IE.ReLU(%call)
        //     %add = IE.Add(%call, %relu)
        //     %call = call @fn(%input, %add)
        // Same for the Sigmoid+Subtract below
        %relu = IE.ReLU(%softmax1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %multiply = IE.Add(%softmax1, %relu) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %add1 = IE.Add(%softmax1, %multiply) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %softmax2 = IE.SoftMax(%add1)  {axisInd = -1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %sigmoid = IE.Sigmoid(%softmax2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %subtract = IE.Subtract(%softmax2, %sigmoid) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %add2 = IE.Add(%softmax2, %subtract) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %add2: tensor<1x48x60x60xf32>

        // CHECK-LABEL: @UnsupportedCyclicalPatternSubgraph
        // CHECK:       func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        // CHECK-NOT:     call
    }
}
