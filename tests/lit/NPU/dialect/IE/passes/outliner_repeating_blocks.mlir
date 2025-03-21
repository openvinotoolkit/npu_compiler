//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --outliner="function-outlining=\"repeating-blocks='min-ops-in-block=2 max-num-iterations=10'\"" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
module @TwoInstances {
    IE.CNNNetwork entryPoint : @main
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

// CHECK: func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RELU:%.+]] = IE.ReLU([[CONV]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RELU]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[CALL1:%.+]] = call @main_fn1([[SOFTMAX]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[CALL2]] : tensor<1x48x60x60xf32>
// CHECK: }

// -----

module @TwoInstancesSameConstant {
    IE.CNNNetwork entryPoint : @main
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

// CHECK: func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[CST:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
// CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[RELU:%.+]] = IE.ReLU([[CONV]]) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   return [[RELU]] : tensor<1x48x60x60xf32>
// CHECK: }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
// CHECK:   [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
// CHECK:   [[CALL1:%.+]] = call @main_fn1([[SOFTMAX]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
// CHECK:   return [[CALL2]] : tensor<1x48x60x60xf32>
// CHECK: }


// -----

module @TwoInstancesDifferentNumberOfOutputs {
    IE.CNNNetwork entryPoint : @main
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

// CHECK: func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
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
    IE.CNNNetwork entryPoint : @main
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

    // CHECK:  func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
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
    IE.CNNNetwork entryPoint : @main
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

    // CHECK:  func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf32>, [[ARG1:%.+]]: tensor<1x48x60x60xf32>, [[ARG2:%.+]]: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
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
    IE.CNNNetwork entryPoint : @main
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

    // CHECK:      func.func private @main_fn1([[ARG0:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[MAXPOOL1:%.+]] = IE.MaxPool([[ARG0]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[AVGPOOL1_1:%.+]] = IE.AvgPool([[MAXPOOL1]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[AVGPOOL1_2:%.+]] = IE.AvgPool([[AVGPOOL1_1]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[AVGPOOL1_2]] : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func private @main_fn2([[ARG0:%.+]]: tensor<1x3x300x300xf32>, [[ARG1:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[ADD1:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[MUL1:%.+]] = IE.Multiply([[ADD1]], [[ADD1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          return [[MUL1]] : tensor<1x3x300x300xf32>
    // CHECK:      }

    // CHECK:      func.func @main([[INPUT:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    // CHECK:          [[FN1_CALL1:%.+]] = call @main_fn1([[INPUT]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          [[SOFTMAX:%.+]] = IE.SoftMax([[FN1_CALL1]]) {axisInd = 3 : i64} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:          [[FN2_CALL1:%.+]] = call @main_fn2([[SOFTMAX]], [[SOFTMAX]]) : (tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          [[FN1_CALL2:%.+]] = call @main_fn1([[FN2_CALL1]]) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          [[FN2_CALL2:%.+]] = call @main_fn2([[FN1_CALL2]], [[FN1_CALL2]]) : (tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>
    // CHECK:          return [[FN2_CALL2]] : tensor<1x3x300x300xf32>
    // CHECK:      }
}
