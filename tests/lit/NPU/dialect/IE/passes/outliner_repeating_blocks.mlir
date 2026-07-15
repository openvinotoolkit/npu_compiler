//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --outliner="function-outlining=\"repeating-blocks='min-ops-in-block=2 max-num-iterations=10'\"" --canonicalize %s | FileCheck %s
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

// CHECK: func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[CST_ARG1:%.+]]: tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST_ARG1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RELU:%.+]] = IE.ReLU([[CONV]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RELU]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func nested @main_rest1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFTMAX]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK:     func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
// CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<2.000000e+00> : tensor<48x48x3x3xf32>
// CHECK:       [[SOFTMAX:%.+]] = call @main_rest1([[INPUT]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK-NOT:   IE.
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

// CHECK: func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[CST_ARG1:%.+]]: tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST_ARG1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RELU:%.+]] = IE.ReLU([[CONV]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RELU]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func nested @main_rest1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[SOFTMAX]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK-DAG: [[CST:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
// CHECK:     [[SOFTMAX:%.+]] = call @main_rest1([[INPUT]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK-NOT: IE.
// CHECK:     [[CALL1:%.+]] = call @main_fn1([[SOFTMAX]], [[CST]]) : (tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32>
// CHECK:     [[CALL2:%.+]] = call @main_fn1([[CALL1]], [[CST]]) : (tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32>) -> tensor<1x48x60x60xf32>
// CHECK:     return [[CALL2]] : tensor<1x48x60x60xf32>
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

    // CHECK:  func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>, [[ARG2:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
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
}

// CHECK-LABEL: @TwoInstancesDifferentNumberOfOutputs

// CHECK: DataInfo "input1" : tensor<1x48x60x60xf32>
// CHECK: DataInfo "input2" : tensor<1x48x60x60xf32>
// CHECK: DataInfo "output" : tensor<1x48x60x60xf32>

// CHECK: func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
// CHECK:   [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RELU:%.+]] = IE.ReLU([[ADD]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[ADD]], [[RELU]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x48x60x60xf32>, [[INPUT2:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CALL1:%.+]]:2 = call @main_fn1([[INPUT1]], [[INPUT2]]) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   [[CALL2:%.+]]:2 = call @main_fn1([[CALL1]]#0, [[CALL1]]#1) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
// CHECK:   return [[CALL2]]#1 : tensor<1x48x60x60xf32>
// CHECK: }

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

    // CHECK:  func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
    // CHECK:      [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[RELU:%.+]] = IE.ReLU([[ADD]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      return [[ADD]], [[RELU]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    // CHECK:  }
    // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:      [[CALL1:%.+]]:2 = call @main_fn1([[INPUT]], [[INPUT]]) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:      [[CALL2:%.+]]:2 = call @main_fn1([[CALL1]]#0, [[CALL1]]#1) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:      return [[CALL2]]#1 : tensor<1x48x60x60xf32>
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

    // CHECK:  func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>, [[ARG2:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
    // CHECK:      [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[ADD2:%.+]] = IE.Add([[ADD1]], [[ARG2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      [[RELU:%.+]] = IE.ReLU([[ADD2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:      return [[ADD1]], [[RELU]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    // CHECK:  }
    // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:      [[CALL1:%.+]]:2 = call @main_fn1([[INPUT]], [[INPUT]], [[INPUT]]) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:      [[CALL2:%.+]]:2 = call @main_fn1([[CALL1]]#0, [[CALL1]]#1, [[CALL1]]#1) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:      return [[CALL2]]#1 : tensor<1x48x60x60xf32>
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

    // CHECK:      func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[MAXPOOL1:%.+]] = IE.MaxPool([[ARG0]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[AVGPOOL1_1:%.+]] = IE.AvgPool([[MAXPOOL1]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[AVGPOOL1_2:%.+]] = IE.AvgPool([[AVGPOOL1_1]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[AVGPOOL1_2]] : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func nested @main_fn2([[ARG0:%.+]]: tensor<1x3x300x300xf32>, [[ARG1:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[MUL1:%.+]] = IE.Multiply([[ADD1]], [[ADD1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[MUL1]] : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func nested @main_rest1([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[SOFTMAX]] : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func @main([[INPUT:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[FN1_CALL1:%.+]] = call @main_fn1([[INPUT]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          [[SOFTMAX:%.+]] = call @main_rest1([[FN1_CALL1]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK-NOT:      IE.
    // CHECK:          [[FN2_CALL1:%.+]] = call @main_fn2([[SOFTMAX]], [[SOFTMAX]]) : (tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          [[FN1_CALL2:%.+]] = call @main_fn1([[FN2_CALL1]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          [[FN2_CALL2:%.+]] = call @main_fn2([[FN1_CALL2]], [[FN1_CALL2]]) : (tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          return [[FN2_CALL2]] : tensor<1x3x300x300xf32>
    // CHECK:      }
}

// -----

module @TwoInstancesInterleaved {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x300x300xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x300x300xf32>
    }

    func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
        %relu = IE.ReLU(%input) : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

        %maxpool1 = IE.MaxPool(%relu) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
        %avgpool1 = IE.AvgPool(%maxpool1) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

        %softmax = IE.SoftMax(%avgpool1) {axisInd = -1} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

        %maxpool2 = IE.MaxPool(%softmax) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
        %avgpool2 = IE.AvgPool(%maxpool2) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

        %lrelu = IE.LeakyRelu(%avgpool2) {negative_slope = 1.000000e-02 : f64} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
        return %lrelu : tensor<1x3x300x300xf32>
    }

    // CHECK-LABEL: @TwoInstancesInterleaved

    // CHECK:      func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[MAXPOOL:%.+]] = IE.MaxPool([[ARG0]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[AVGPOOL:%.+]] = IE.AvgPool([[MAXPOOL]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[AVGPOOL]] : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func nested @main_rest1([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[RELU_FN:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[RELU_FN]] : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func nested @main_rest2([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[SOFTMAX_FN:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[SOFTMAX_FN]] : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func nested @main_rest3([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[LRELU_FN:%.+]] = IE.LeakyRelu([[ARG0]]) {negative_slope = 1.000000e-02 : f64} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[LRELU_FN]] : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func @main([[INPUT:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[RELU:%.+]] = call @main_rest1([[INPUT]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK-NOT:      IE.
    // CHECK:          [[CALL1:%.+]] = call @main_fn1([[RELU]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          [[SOFTMAX:%.+]] = call @main_rest2([[CALL1]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK-NOT:      IE.
    // CHECK:          [[CALL2:%.+]] = call @main_fn1([[SOFTMAX]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>

    // CHECK:          [[LRELU:%.+]] = call @main_rest3([[CALL2]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK-NOT:      IE.
    // CHECK:          return [[LRELU]] : tensor<1x3x300x300xf32>
    // CHECK:      }
}

// -----

module @NoRepeatedBlocks {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %softmax = IE.SoftMax(%input) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu = IE.ReLU(%softmax) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %relu : tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @NoRepeatedBlocks
    // CHECK:      func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:          [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:          [[RELU:%.+]] = IE.ReLU([[SOFTMAX]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:          return [[RELU]] : tensor<1x48x60x60xf32>
    // CHECK:      }
}

// -----

module @TwoInstancesWithLoop {
        net.NetworkInfo entryPoint : @main
        inputsInfo : {
                DataInfo "input" : tensor<3x5xf32>
        } outputsInfo : {
                DataInfo "output" : tensor<3x5xf32>
        }

        func.func @main(%input: tensor<3x5xf32>) -> tensor<3x5xf32> {
                %num_iterations = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi32>
                %exec_cond = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>

                %loop1 = IE.Loop(%num_iterations, %exec_cond, %input) : tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32> -> tensor<3x5xf32>
                (num_iterations : 3 current_iter_index : -1 exec_cond_index : 1)
                    slice_input_descs : []
                    invariant_input_descs : []
                    feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
                    concat_output_descs : []
                    invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
                    body_module : {
                ^bb0(%arg0: tensor<3x5xf32>):
                        %loop_cond1 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
                        %softmax1 = IE.SoftMax(%arg0) {axisInd = 0 : i64} : tensor<3x5xf32> -> tensor<3x5xf32>
                        "IE.LoopTerminator"(%softmax1, %loop_cond1) : (tensor<3x5xf32>, tensor<1xi8>) -> ()
                    }
                %relu1 = IE.ReLU(%loop1) : tensor<3x5xf32> -> tensor<3x5xf32>

                %loop2 = IE.Loop(%num_iterations, %exec_cond, %relu1) : tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32> -> tensor<3x5xf32>
                (num_iterations : 3 current_iter_index : -1 exec_cond_index : 1)
                    slice_input_descs : []
                    invariant_input_descs : []
                    feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
                    concat_output_descs : []
                    invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
                    body_module : {
                ^bb0(%arg1: tensor<3x5xf32>):
                        %loop_cond2 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
                        %softmax2 = IE.SoftMax(%arg1) {axisInd = 0 : i64} : tensor<3x5xf32> -> tensor<3x5xf32>
                        "IE.LoopTerminator"(%softmax2, %loop_cond2) : (tensor<3x5xf32>, tensor<1xi8>) -> ()
                    }
                %relu2 = IE.ReLU(%loop2) : tensor<3x5xf32> -> tensor<3x5xf32>

                return %relu2 : tensor<3x5xf32>
        }

        // CHECK-LABEL: @TwoInstancesWithLoop

        // CHECK: func.func nested @main_fn1([[ARG0:%.+]]: tensor<1xsi32>, [[ARG1:%.+]]: tensor<1xi8>, [[ARG2:%.+]]: tensor<3x5xf32>) -> tensor<3x5xf32> {
        // CHECK:   [[LOOP_COND:%.+]] = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
        // CHECK:   [[LOOP:%.+]] = IE.Loop([[ARG0]], [[ARG1]], [[ARG2]]) : tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32> -> tensor<3x5xf32>
        // CHECK:   body_module : {
        // CHECK:   ^bb0([[LOOP_ARG:%.+]]: tensor<3x5xf32>):
        // CHECK:     [[SOFTMAX:%.+]] = IE.SoftMax([[LOOP_ARG]]) {axisInd = 0 : i64} : tensor<3x5xf32> -> tensor<3x5xf32>
        // CHECK:     "IE.LoopTerminator"([[SOFTMAX]], [[LOOP_COND]]) : (tensor<3x5xf32>, tensor<1xi8>) -> ()
        // CHECK:   }
        // CHECK:   [[RELU:%.+]] = IE.ReLU([[LOOP]]) : tensor<3x5xf32> -> tensor<3x5xf32>
        // CHECK:   return [[RELU]] : tensor<3x5xf32>
        // CHECK: }

        // CHECK: func.func @main([[INPUT:%.+]]: tensor<3x5xf32>) -> tensor<3x5xf32> {
        // CHECK-DAG: [[NUM_ITER:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi32>
        // CHECK-DAG: [[EXEC_COND:%.+]] = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
        // CHECK: [[CALL1:%.+]] = call @main_fn1([[NUM_ITER]], [[EXEC_COND]], [[INPUT]]) : (tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32>) -> tensor<3x5xf32>
        // CHECK: [[CALL2:%.+]] = call @main_fn1([[NUM_ITER]], [[EXEC_COND]], [[CALL1]]) : (tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32>) -> tensor<3x5xf32>
        // CHECK: return [[CALL2]] : tensor<3x5xf32>
        // CHECK: }
}

// -----

module @WeightsCreatedInHeadUsedInTail {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %weights = const.Declare tensor<48x48x3x3xf32> = dense<3.0> : tensor<48x48x3x3xf32>
        %tw1 = IE.Multiply(%weights, %weights) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
        %tw2 = IE.Subtract(%tw1, %weights) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>

        %tinput = IE.Add(%input, %input) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %add1 = IE.Add(%tinput, %tinput) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %add2 = IE.Add(%add1, %tinput) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%add2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %add3 = IE.Add(%add1, %relu1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %add4 = IE.Add(%add3, %relu1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%add4) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %result = IE.Convolution(%relu2, %tw2) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        return %result: tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @WeightsCreatedInHeadUsedInTail

    // CHECK: func.func nested @main_fn1([[FN_ARG0:%.+]]: tensor<1x48x60x60xf32>, [[FN_ARG1:%.+]]: tensor<1x48x60x60xf32>, [[FN_ARG2:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
    // CHECK:   [[FN_ADD1:%.+]] = IE.Add([[FN_ARG0]], [[FN_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[FN_ADD2:%.+]] = IE.Add([[FN_ADD1]], [[FN_ARG2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[FN_RELU:%.+]] = IE.ReLU([[FN_ADD2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[FN_ADD1]], [[FN_RELU]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    // CHECK: }
    // CHECK: func.func nested @main_rest1([[REST_ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[REST_ADD:%.+]] = IE.Add([[REST_ARG0]], [[REST_ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[REST_ADD]] : tensor<1x48x60x60xf32>
    // CHECK: }
    // CHECK: func.func nested @main_rest2([[REST_ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[REST_CST:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<3.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK:   [[REST_MUL:%.+]] = IE.Multiply([[REST_CST]], [[REST_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
    // CHECK:   [[REST_SUB:%.+]] = IE.Subtract([[REST_MUL]], [[REST_CST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
    // CHECK:   [[REST_CONV:%.+]] = IE.Convolution([[REST_ARG0]], [[REST_SUB]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[REST_CONV]] : tensor<1x48x60x60xf32>
    // CHECK: }
    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[MAIN_0:%.+]] = call @main_rest1([[ARG0]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   [[MAIN_1:%.+]]:2 = call @main_fn1([[MAIN_0]], [[MAIN_0]], [[MAIN_0]]) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:   [[MAIN_2:%.+]]:2 = call @main_fn1([[MAIN_1]]#0, [[MAIN_1]]#1, [[MAIN_1]]#1) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:   [[MAIN_3:%.+]] = call @main_rest2([[MAIN_2]]#1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   return [[MAIN_3]] : tensor<1x48x60x60xf32>
    // CHECK: }
}

// -----

module @SameWeightsInHeadAndTail {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %weights = const.Declare tensor<48x48x3x3xf32> = dense<3.0> : tensor<48x48x3x3xf32>
        %head = IE.Convolution(%input, %weights) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>

        %add1 = IE.Add(%head, %head) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %add2 = IE.Add(%add1, %head) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu1 = IE.ReLU(%add2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %add3 = IE.Add(%add1, %relu1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %add4 = IE.Add(%add3, %relu1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%add4) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

        %tail = IE.Convolution(%relu2, %weights) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        return %tail : tensor<1x48x60x60xf32>
    }

    // CHECK-LABEL: @SameWeightsInHeadAndTail

    // CHECK: func.func nested @main_fn1([[FN_ARG0:%.+]]: tensor<1x48x60x60xf32>, [[FN_ARG1:%.+]]: tensor<1x48x60x60xf32>, [[FN_ARG2:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
    // CHECK:   [[FN_ADD1:%.+]] = IE.Add([[FN_ARG0]], [[FN_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[FN_ADD2:%.+]] = IE.Add([[FN_ADD1]], [[FN_ARG2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   [[FN_RELU:%.+]] = IE.ReLU([[FN_ADD2]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[FN_ADD1]], [[FN_RELU]] : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
    // CHECK: }
    // CHECK: func.func nested @main_rest1([[REST_ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[REST1_CST:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<3.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK:   [[REST1_CONV:%.+]] = IE.Convolution([[REST_ARG0]], [[REST1_CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[REST1_CONV]] : tensor<1x48x60x60xf32>
    // CHECK: }
    // CHECK: func.func nested @main_rest2([[REST_ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[REST2_CST:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<3.000000e+00> : tensor<48x48x3x3xf32>
    // CHECK:   [[REST2_CONV:%.+]] = IE.Convolution([[REST_ARG0]], [[REST2_CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK:   return [[REST2_CONV]] : tensor<1x48x60x60xf32>
    // CHECK: }
    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:   [[MAIN_0:%.+]] = call @main_rest1([[ARG0]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   [[MAIN_1:%.+]]:2 = call @main_fn1([[MAIN_0]], [[MAIN_0]], [[MAIN_0]]) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:   [[MAIN_2:%.+]]:2 = call @main_fn1([[MAIN_1]]#0, [[MAIN_1]]#1, [[MAIN_1]]#1) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
    // CHECK:   [[MAIN_3:%.+]] = call @main_rest2([[MAIN_2]]#1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   return [[MAIN_3]] : tensor<1x48x60x60xf32>
    // CHECK: }
}
