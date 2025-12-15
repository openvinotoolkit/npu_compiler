//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --concat-repeating-blocks-outlining="min-seq-length=2 single-function-per-concat=false" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @OutlineConcat {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x96x32x32xf16>
    }

    func.func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        %softmax1 = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu1 = VPU.ReLU(%softmax1) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %softmax2 = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu2 = VPU.ReLU(%softmax2) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

        %concat = VPU.Concat(%relu1, %relu2) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
            : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>

        return %concat : tensor<1x96x32x32xf16, {order = #NHWC}>

        // CHECK:  func.func private @main_concat1_input1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
        // CHECK:      [[RELU:%.+]] = VPU.ReLU([[SOFTMAX]])
        // CHECK:      return [[RELU]]
        // CHECK:  }
        // CHECK:  func.func private @main_concat1_input2([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
        // CHECK:      [[RELU:%.+]] = VPU.ReLU([[SOFTMAX]])
        // CHECK:      return [[RELU]]
        // CHECK:  }
        // CHECK:  func.func private @main_concat2_input1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[ARG0]], [[ARG1]])
        // CHECK:      return [[CONCAT]]
        // CHECK:  }
        // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[CALL1:%.+]] = call @main_concat1_input1([[INPUT]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
        // CHECK:      [[CALL2:%.+]] = call @main_concat1_input2([[INPUT]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
        // CHECK:      [[CALL3:%.+]] = call @main_concat2_input1([[CALL1]], [[CALL2]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}>
        // CHECK:      return [[CALL3]]
        // CHECK:  }

    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @DoNotOutlineConcatsWithoutRepeatingInputBranches {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x96x32x32xf16>
    }

    func.func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        %softmax1 = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu1 = VPU.ReLU(%softmax1) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        // Different axis for Softmax in the input second branch
        %softmax2 = VPU.SoftMax(%input) {axisInd = 3 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu2 = VPU.ReLU(%softmax2) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

        %concat = VPU.Concat(%relu1, %relu2) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
            : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>

        return %concat : tensor<1x96x32x32xf16, {order = #NHWC}>

        // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[SOFTMAX1:%.+]] = VPU.SoftMax([[ARG0]])
        // CHECK:      [[RELU1:%.+]] = VPU.ReLU([[SOFTMAX1]])
        // CHECK:      [[SOFTMAX2:%.+]] = VPU.SoftMax([[ARG0]])
        // CHECK:      [[RELU2:%.+]] = VPU.ReLU([[SOFTMAX2]])
        // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[RELU1]], [[RELU2]])
        // CHECK:      return [[CONCAT]]
        // CHECK:  }
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @OutlineConcatWithConstants {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x96x32x32xf16>
    }

    func.func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        %maxpool_wt = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

        %maxpool1 = VPU.NCE.MaxPool(%input, %maxpool_wt) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
            } -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %softmax1 = VPU.SoftMax(%maxpool1) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %maxpool2 = VPU.NCE.MaxPool(%input, %maxpool_wt) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
            } -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %softmax2 = VPU.SoftMax(%maxpool2) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %concat = VPU.Concat(%softmax1, %softmax2) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
            : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>

        return %concat : tensor<1x96x32x32xf16, {order = #NHWC}>

        // CHECK:  func.func private @main_concat1_input1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[CST:%.+]] = const.Declare tensor<48x1x1x4xsi32>
        // CHECK:      [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[CST]] )
        // CHECK:      [[SOFTMAX:%.+]] = VPU.SoftMax([[MAXPOOL]])
        // CHECK:      return [[SOFTMAX]]
        // CHECK:  }
        // CHECK:  func.func private @main_concat1_input2([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[CST:%.+]] = const.Declare tensor<48x1x1x4xsi32>
        // CHECK:      [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[CST]] )
        // CHECK:      [[SOFTMAX:%.+]] = VPU.SoftMax([[MAXPOOL]])
        // CHECK:      return [[SOFTMAX]]
        // CHECK:  }
        // CHECK:  func.func private @main_concat2_input1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[ARG0]], [[ARG1]])
        // CHECK:      return [[CONCAT]]
        // CHECK:  }
        // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[CALL1:%.+]] = call @main_concat1_input1([[INPUT]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
        // CHECK:      [[CALL2:%.+]] = call @main_concat1_input2([[INPUT]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
        // CHECK:      [[CALL3:%.+]] = call @main_concat2_input1([[CALL1]], [[CALL2]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}>
        // CHECK:      return [[CALL3]]
        // CHECK:  }

    }
}
