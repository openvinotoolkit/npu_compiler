//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --outliner="function-outlining=\"repeating-blocks='min-ops-in-block=2 max-num-iterations=10 weights-as-inputs=true'\"" --canonicalize %s | FileCheck %s
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
}

// CHECK-LABEL: @TwoInstances

// CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
// CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

// CHECK: func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RELU:%.+]] = IE.ReLU([[CONV]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RELU]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK:     func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
// CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<2.000000e+00> : tensor<48x48x3x3xf32>
// CHECK-DAG:   [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:       [[CALL1:%.+]] = call @main_fn1([[SOFTMAX]], [[CST1]]) : (tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32>
// CHECK:       [[CALL2:%.+]] = call @main_fn1([[CALL1]], [[CST2]]) : (tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32>
// CHECK:       return [[CALL2]] : tensor<1x48x60x60xf32>
// CHECK:     }

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

// CHECK: func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RELU:%.+]] = IE.ReLU([[CONV]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RELU]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
// CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[CALL1:%.+]] = call @main_fn1([[SOFTMAX]], [[CST]]) : (tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[CALL2:%.+]] = call @main_fn1([[CALL1]], [[CST]]) : (tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[CALL2]] : tensor<1x48x60x60xf32>
// CHECK: }

// -----

module @ReuseMultipleOpsSubtractMultiply {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %cst1 = const.Declare tensor<1x48x60x60xf32> = dense<1.000000e+00> : tensor<1x48x60x60xf32>
        %sub1 = IE.Subtract(%cst1, %input) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %cst3 = const.Declare tensor<1x48x60x60xf32> = dense<2.000000e+00> : tensor<1x48x60x60xf32>
        %mul1 = IE.Multiply(%sub1, %cst3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%mul1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %cst2 = const.Declare tensor<1x48x60x60xf32> = dense<3.000000e+00> : tensor<1x48x60x60xf32>
        %sub2 = IE.Subtract(%cst2, %cst1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %mul2 = IE.Multiply(%sub2, %relu1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%mul2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        return %relu2: tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @ReuseMultipleOpsSubtractMultiply

    // CHECK: DataInfo "input" : tensor<1x48x60x60xf32>
    // CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

    // CHECK:  func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>, [[ARG2:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:      [[SUB:%.+]] = IE.Subtract([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[MUL:%.+]] = IE.Multiply([[SUB]], [[ARG2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[REL:%.+]] = IE.ReLU([[MUL]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      return [[REL]] : tensor<1x48x60x60xf32>
    // CHECK:  }
    // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:  [[CST1:%.+]] = const.Declare tensor<1x48x60x60xf32> = dense<1.000000e+00> : tensor<1x48x60x60xf32>
    // CHECK-DAG:  [[CST3:%.+]] = const.Declare tensor<1x48x60x60xf32> = dense<2.000000e+00> : tensor<1x48x60x60xf32>
    // CHECK-DAG:  [[CST2:%.+]] = const.Declare tensor<1x48x60x60xf32> = dense<3.000000e+00> : tensor<1x48x60x60xf32>
    // CHECK:      [[CALL1:%.+]] = call @main_fn1([[CST1]], [[INPUT]], [[CST3]]) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:      [[CALL2:%.+]] = call @main_fn1([[CST2]], [[CST1]], [[CALL1]]) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:      return [[CALL2]] : tensor<1x48x60x60xf32>
    // CHECK:  }
}
