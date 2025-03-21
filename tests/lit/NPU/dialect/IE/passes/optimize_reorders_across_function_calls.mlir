//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-reorders-across-function-calls %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @OneFunction {
    func.func private @function(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
        %reorder = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax = IE.SoftMax(%reorder) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        return %softmax : tensor<1x48x60x60xf16, {order = #NHWC}>
    }
    func.func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
        %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder = IE.Reorder(%conv) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        %call = call @function(%reorder) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        return %call : tensor<1x48x60x60xf16, {order = #NHWC}>
    }

    // CHECK:  func.func private @function([[ARG0:%.+]]: tensor<1x48x60x60xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK:      return [[SOFTMAX]] : tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    // CHECK:      [[CST:%.+]] = const.Declare
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK:      [[CALL:%.+]] = call @function([[CONV]]) : (tensor<1x48x60x60xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:      return [[CALL]] : tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @MultipleFunctionsSameProducerReorder {
    func.func private @function1(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
        %reorder = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax = IE.SoftMax(%reorder) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        return %softmax : tensor<1x48x60x60xf16, {order = #NHWC}>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
        %reorder = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax = IE.SoftMax(%reorder) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        return %softmax : tensor<1x48x60x60xf16, {order = #NHWC}>
    }
    func.func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>) {
        %cst = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
        %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder = IE.Reorder(%conv) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        %call1 = call @function1(%reorder) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %call2 = call @function2(%reorder) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        return %call1, %call2 : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>
    }

    // CHECK:  func.func private @function1([[ARG0:%.+]]: tensor<1x48x60x60xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    // CHECK:      [[SOFTMAX1:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK:      return [[SOFTMAX1]] : tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:  }
    // CHECK:  func.func private @function2([[ARG0:%.+]]: tensor<1x48x60x60xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    // CHECK:      [[SOFTMAX2:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK:      return [[SOFTMAX2]] : tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>) {
    // CHECK:      [[CST:%.+]] = const.Declare
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK:      [[CALL1:%.+]] = call @function1([[CONV]]) : (tensor<1x48x60x60xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:      [[CALL2:%.+]] = call @function2([[CONV]]) : (tensor<1x48x60x60xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:      return [[CALL1]], [[CALL2]] : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @ReorderPair {
    func.func private @main_part1(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
        %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder = IE.Reorder(%conv) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %reorder : tensor<1x48x60x60xf16>
    }
    func.func private @main_part2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
        %reorder = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax = IE.SoftMax(%reorder) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        return %softmax : tensor<1x48x60x60xf16, {order = #NHWC}>
    }
    func.func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
        %call_part1 = call @main_part1(%arg0) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16>
        %call_part2 = call @main_part2(%call_part1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        return %call_part2 : tensor<1x48x60x60xf16, {order = #NHWC}>
    }

    // CHECK:  func.func private @main_part1([[PART1_ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    // CHECK:      [[CST:%.+]] = const.Declare
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:      return [[CONV]] : tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:  }
    // CHECK:  func.func private @main_part2([[PART2_ARG0:%.+]]: tensor<1x48x60x60xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[PART2_ARG0]])
    // CHECK:      return [[SOFTMAX]] : tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    // CHECK:      [[CALL_PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:      [[CALL_PART2:%.+]] = call @main_part2([[CALL_PART1]]) : (tensor<1x48x60x60xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:      return [[CALL_PART2]] : tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @ReorderPairMiddleArgPosition {
    func.func private @main_part1(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
        %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder = IE.Reorder(%conv) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %reorder : tensor<1x48x60x60xf16>
    }
    func.func private @main_part2(%arg0: tensor<1x3x62x62xf16>, %arg1: tensor<1x48x60x60xf16>, %arg2: tensor<1x3x62x62xf16>)
            -> (tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x3x62x62xf16>) {
        %softmax1 = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16>
        %reorder = IE.Reorder(%arg1) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax2 = IE.SoftMax(%reorder) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax3 = IE.SoftMax(%arg2) {axisInd = 1 : i64} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16>
        return %softmax1, %softmax2, %softmax3 : tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x3x62x62xf16>
    }
    func.func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>, %arg1: tensor<1x3x62x62xf16>, %arg2: tensor<1x3x62x62xf16>)
            -> (tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x3x62x62xf16>) {
        %call_part1 = call @main_part1(%arg0) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16>
        %call_part2:3 = call @main_part2(%arg1, %call_part1, %arg2) : (tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16>, tensor<1x3x62x62xf16>) -> (tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x3x62x62xf16>)
        return %call_part2#0, %call_part2#1, %call_part2#2 : tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x3x62x62xf16>
    }

    // CHECK:       func.func private @main_part1([[PART1_ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    // CHECK:           [[CST:%.+]] = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
    // CHECK:           [[CONV:%.+]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:           return [[CONV]] : tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       func.func private @main_part2([[PART1_ARG0:%.+]]: tensor<1x3x62x62xf16>, [[PART1_ARG1:%.+]]: tensor<1x3x62x62xf16>, [[PART1_ARG2:%.+]]: tensor<1x48x60x60xf16, {order = #NHWC}>)
    // CHECK-SAME:        -> (tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x3x62x62xf16>) {
    // CHECK:           [[SOFTMAX1:%.+]] = IE.SoftMax([[PART1_ARG0]]) {axisInd = 1 : i64} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16>
    // CHECK:           [[SOFTMAX2:%.+]] = IE.SoftMax([[PART1_ARG2]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:           [[SOFTMAX3:%.+]] = IE.SoftMax([[PART1_ARG1]]) {axisInd = 1 : i64} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16>
    // CHECK:           return [[SOFTMAX1]], [[SOFTMAX2]], [[SOFTMAX3]] : tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x3x62x62xf16>
    // CHECK:       }
    // CHECK:       func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x3x62x62xf16>, [[ARG2:%.+]]: tensor<1x3x62x62xf16>)
    // CHECK-SAME:        -> (tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x3x62x62xf16>) {
    // CHECK:           [[CALL_PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:           [[CALL_PART2:%.+]]:3 = call @main_part2([[ARG1]], [[ARG2]], [[CALL_PART1]]) : (tensor<1x3x62x62xf16>, tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>)
    // CHECK-SAME:        -> (tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x3x62x62xf16>)
    // CHECK:           return [[CALL_PART2]]#0, [[CALL_PART2]]#1, [[CALL_PART2]]#2 : tensor<1x3x62x62xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x3x62x62xf16>
    // CHECK:       }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @MultipleReorderPairs {
    func.func private @main_part1(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16>, tensor<1x48x30x30xf16>) {
        %cst = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
        %conv1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %conv2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x30x30xf16, {order = #NHWC}>
        %reorder1 = IE.Reorder(%conv1) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        %reorder2 = IE.Reorder(%conv2) {dstOrder = #NCHW} : tensor<1x48x30x30xf16, {order = #NHWC}> -> tensor<1x48x30x30xf16>
        return %reorder1, %reorder2 : tensor<1x48x60x60xf16>, tensor<1x48x30x30xf16>
    }
    func.func private @main_part2(%arg0: tensor<1x48x60x60xf16>, %arg1: tensor<1x48x30x30xf16>) -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>) {
        %reorder1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder2 = IE.Reorder(%arg1) {dstOrder = #NHWC} : tensor<1x48x30x30xf16> -> tensor<1x48x30x30xf16, {order = #NHWC}>
        %softmax1 = IE.SoftMax(%reorder1) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax2 = IE.SoftMax(%reorder2) {axisInd = 1 : i64} : tensor<1x48x30x30xf16, {order = #NHWC}> -> tensor<1x48x30x30xf16, {order = #NHWC}>
        return %softmax1, %softmax2 : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>
    }
    func.func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>) {
        %call_part1:2 = call @main_part1(%arg0) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16>, tensor<1x48x30x30xf16>)
        %call_part2:2 = call @main_part2(%call_part1#0, %call_part1#1) : (tensor<1x48x60x60xf16>, tensor<1x48x30x30xf16>) -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>)
        return %call_part2#0, %call_part2#1 : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>
    }

    // CHECK:       func.func private @main_part1([[PART1_ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>) {
    // CHECK:           [[CST:%.+]] = const.Declare
    // CHECK:           [[CONV1:%.+]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:           [[CONV2:%.+]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:           return [[CONV1]], [[CONV2]] : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       func.func private @main_part2([[PART2_ARG0:%.+]]: tensor<1x48x60x60xf16, {order = #NHWC}>, [[PART2_ARG1:%.+]]: tensor<1x48x30x30xf16, {order = #NHWC}>)
    // CHECK-SAME:        -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>) {
    // CHECK:           [[SOFTMAX1:%.+]] = IE.SoftMax([[PART2_ARG0]])
    // CHECK:           [[SOFTMAX2:%.+]] = IE.SoftMax([[PART2_ARG1]])
    // CHECK:           return [[SOFTMAX1]], [[SOFTMAX2]] : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>)
    // CHECK-SAME:        -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>) {
    // CHECK:           [[CALL_PART1:%.+]]:2 = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>)
    // CHECK:           [[CALL_PART2:%.+]]:2 = call @main_part2([[CALL_PART1]]#0, [[CALL_PART1]]#1) : (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>)
    // CHECK-SAME:        -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>)
    // CHECK:           return [[CALL_PART2]]#0, [[CALL_PART2]]#1 : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>
    // CHECK:       }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @MultipleReorderUsers {
    func.func private @main_part1(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16>, tensor<1x48x30x30xf16, {order = #NHWC}>) {
        %cst = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
        %conv1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %conv2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x30x30xf16, {order = #NHWC}>
        %reorder = IE.Reorder(%conv1) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %reorder, %conv2 : tensor<1x48x60x60xf16>, tensor<1x48x30x30xf16, {order = #NHWC}>
    }
    func.func private @main_part2(%arg0: tensor<1x48x60x60xf16>, %arg1: tensor<1x48x30x30xf16, {order = #NHWC}>)
            -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>) {
        %reorder1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder2 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax1 = IE.SoftMax(%reorder1) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax2 = IE.SoftMax(%reorder2) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax3 = IE.SoftMax(%arg1) {axisInd = 1 : i64} : tensor<1x48x30x30xf16, {order = #NHWC}> -> tensor<1x48x30x30xf16, {order = #NHWC}>
        return %softmax1, %softmax2, %softmax3 : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>
    }
    func.func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>) {
        %call_part1:2 = call @main_part1(%arg0) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16>, tensor<1x48x30x30xf16, {order = #NHWC}>)
        %call_part2:3 = call @main_part2(%call_part1#0, %call_part1#1) : (tensor<1x48x60x60xf16>, tensor<1x48x30x30xf16, {order = #NHWC}>)
            -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>)
        return %call_part2#0, %call_part2#1, %call_part2#2 : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>
    }

    // CHECK:       func.func private @main_part1([[PART1_ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x30x30xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>) {
    // CHECK:           [[CST:%.+]] = const.Declare
    // CHECK:           [[CONV1:%.+]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:           [[CONV2:%.+]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:           return [[CONV2]], [[CONV1]] : tensor<1x48x30x30xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       func.func private @main_part2([[PART2_ARG0:%.+]]: tensor<1x48x30x30xf16, {order = #NHWC}>, [[PART2_ARG1:%.+]]: tensor<1x48x60x60xf16, {order = #NHWC}>)
    // CHECK-SAME:        -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>) {
    // CHECK:           [[SOFTMAX1:%.+]] = IE.SoftMax([[PART2_ARG1]])
    // CHECK:           [[SOFTMAX2:%.+]] = IE.SoftMax([[PART2_ARG1]])
    // CHECK:           [[SOFTMAX3:%.+]] = IE.SoftMax([[PART2_ARG0]])
    // CHECK:           return [[SOFTMAX1]], [[SOFTMAX2]], [[SOFTMAX3]] : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>)
    // CHECK-SAME:        -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>) {
    // CHECK:           [[CALL_PART1:%.+]]:2 = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x30x30xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>)
    // CHECK:           [[CALL_PART2:%.+]]:3 = call @main_part2([[CALL_PART1]]#0, [[CALL_PART1]]#1) : (tensor<1x48x30x30xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>)
    // CHECK-SAME:        -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>)
    // CHECK:           return [[CALL_PART2]]#0, [[CALL_PART2]]#1, [[CALL_PART2]]#2 : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x30x30xf16, {order = #NHWC}>
    // CHECK:       }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @MultipleReorderUsersMixed {
    func.func private @main_part1(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
        %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder = IE.Reorder(%conv) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %reorder : tensor<1x48x60x60xf16>
    }
    func.func private @main_part2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
        %reorder = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %softmax = IE.SoftMax(%reorder) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        return %softmax : tensor<1x48x60x60xf16, {order = #NHWC}>
    }
    func.func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>) {
        %call_part1 = call @main_part1(%arg0) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16>
        %call_part2 = call @main_part2(%call_part1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder = IE.Reorder(%call_part1) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        return %call_part2, %reorder : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>
    }

    // CHECK:  func.func private @main_part1([[PART1_ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>) {
    // CHECK:      [[CST:%.+]] = const.Declare
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:      [[REORDER1:%.+]] = IE.Reorder([[CONV]])
    // CHECK:      return [[REORDER1]], [[CONV]]
    // CHECK:  }
    // CHECK:  func.func private @main_part2([[PART2_ARG0:%.+]]: tensor<1x48x60x60xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}> {
    // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[PART2_ARG0]])
    // CHECK:      return [[SOFTMAX]]
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<1x48x60x60xf16, {order = #NHWC}>) {
    // CHECK:      [[CALL_PART1:%.+]]:2 = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16, {order = #NHWC}>)
    // CHECK:      [[CALL_PART2:%.+]] = call @main_part2([[CALL_PART1]]#1) : (tensor<1x48x60x60xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:      [[REORDER2:%.+]] = IE.Reorder([[CALL_PART1]]#0)
    // CHECK:      return [[CALL_PART2]], [[REORDER2]]
    // CHECK:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @DoNotOptimizeUserSameInOutOrder {
    func.func private @main_part1(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
        %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder = IE.Reorder(%conv) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %reorder : tensor<1x48x60x60xf16>
    }
    func.func private @main_part2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %softmax = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        // Softmax requires both the input and output to have the same layout, so this case would introduce a Reorder here if the producer Reorder would be optimized
        return %softmax : tensor<1x48x60x60xf16>
    }
    func.func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
        %call_part1 = call @main_part1(%arg0) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16>
        %call_part2 = call @main_part2(%call_part1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %call_part2 : tensor<1x48x60x60xf16>
    }

    // CHECK:  func.func private @main_part1([[PART1_ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[CST:%.+]] = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}>
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:      [[REORDER:%.+]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
    // CHECK:      return [[REORDER]] : tensor<1x48x60x60xf16>
    // CHECK:  }
    // CHECK:  func.func private @main_part2([[PART2_ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[PART2_ARG0]])
    // CHECK:      return [[SOFTMAX]] : tensor<1x48x60x60xf16>
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[CALL_PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16>
    // CHECK:      [[CALL_PART2:%.+]] = call @main_part2([[CALL_PART1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    // CHECK:      return [[CALL_PART2]] : tensor<1x48x60x60xf16>
    // CHECK:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @UserAnyDimsOrder {
    func.func private @main_part1(%arg0: tensor<1x48x60x1xf16, {order = #NHWC}>) -> tensor<1x48x60x1xf16> {
        %cst = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf16, {order = #NHWC}>
        %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x1xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x1xf16, {order = #NHWC}>
        %reorder = IE.Reorder(%conv) {dstOrder = #NCHW} : tensor<1x48x60x1xf16, {order = #NHWC}> -> tensor<1x48x60x1xf16>
        return %reorder : tensor<1x48x60x1xf16>
    }
    func.func private @main_part2(%arg0: tensor<1x48x60x1xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<4xsi32> = dense<[1, 48, 60, 60]> : tensor<4xsi32>
        %broadcast = IE.Broadcast(%arg0, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x48x60x1xf16>, tensor<4xsi32> -> tensor<1x48x60x60xf16>
        // Broadcast accepts the input and output having different layouts, so this case would not introduce a Reorder here if the producer Reorder would be optimized
        return %broadcast : tensor<1x48x60x60xf16>
    }
    func.func @main(%arg0: tensor<1x48x60x1xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
        %call_part1 = call @main_part1(%arg0) : (tensor<1x48x60x1xf16, {order = #NHWC}>) -> tensor<1x48x60x1xf16>
        %call_part2 = call @main_part2(%call_part1) : (tensor<1x48x60x1xf16>) -> tensor<1x48x60x60xf16>
        return %call_part2 : tensor<1x48x60x60xf16>
    }

    // CHECK:  func.func private @main_part1([[PART1_ARG0:%.+]]: tensor<1x48x60x1xf16, {order = #NHWC}>) -> tensor<1x48x60x1xf16, {order = #NHWC}> {
    // CHECK:      [[CST:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}>
    // CHECK:      [[CONV]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:      return [[CONV]] : tensor<1x48x60x1xf16, {order = #NHWC}>
    // CHECK:  }
    // CHECK:  func.func private @main_part2([[PART2_ARG0:%.+]]: tensor<1x48x60x1xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[CST:%.+]] = const.Declare tensor<4xsi32>
    // CHECK:      [[BROADCAST:%.+]] = IE.Broadcast([[PART2_ARG0]], [[CST]]) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x48x60x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<1x48x60x60xf16>
    // CHECK:      return [[BROADCAST]] : tensor<1x48x60x60xf16>
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x48x60x1xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[CALL_PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x48x60x1xf16, {order = #NHWC}>) -> tensor<1x48x60x1xf16, {order = #NHWC}>
    // CHECK:      [[CALL_PART2:%.+]] = call @main_part2([[CALL_PART1]]) : (tensor<1x48x60x1xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16>
    // CHECK:      return [[CALL_PART2]] : tensor<1x48x60x60xf16>
    // CHECK:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @DoNotOptimizeMultipleProducersSameUser {
    func.func private @main_part1(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>) {
        %cst = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
        %conv1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %conv2 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<48x3x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder1 = IE.Reorder(%conv1) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        %reorder2 = IE.Reorder(%conv2) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %reorder1, %reorder2 : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>
    }
    func.func private @main_part2(%arg0: tensor<1x48x60x60xf16>, %arg1: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %add = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        // Add requires both the inputs and output to have the same layout, so this case would introduce a Reorder here if the producer Reorders would be optimized
        return %add : tensor<1x48x60x60xf16>
    }
    func.func @main(%arg0: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
        %call_part1:2 = call @main_part1(%arg0) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>)
        %call_part2 = call @main_part2(%call_part1#0, %call_part1#1) : (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %call_part2 : tensor<1x48x60x60xf16>
    }

    // CHECK:  func.func private @main_part1([[PART1_ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>) {
    // CHECK:      [[CST:%.+]] = const.Declare tensor<48x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf16, {order = #NHWC}>
    // CHECK:      [[CONV1:%.+]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:      [[CONV2:%.+]] = IE.Convolution([[PART1_ARG0]], [[CST]])
    // CHECK:      [[REORDER1:%.+]] = IE.Reorder([[CONV1]]) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
    // CHECK:      [[REORDER2:%.+]] = IE.Reorder([[CONV2]]) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
    // CHECK:      return [[REORDER1]], [[REORDER2]] : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>
    // CHECK:  }
    // CHECK:  func.func private @main_part2([[PART2_ARG0:%.+]]: tensor<1x48x60x60xf16>, [[PART2_ARG1:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[ADD:%.+]] = IE.Add([[PART2_ARG0]], [[PART2_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
    // CHECK:      return [[ADD]] : tensor<1x48x60x60xf16>
    // CHECK:  }
    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[CALL_PART1:%.+]]:2 = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf16, {order = #NHWC}>) -> (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>)
    // CHECK:      [[CALL_PART2:%.+]] = call @main_part2([[CALL_PART1]]#0, [[CALL_PART1]]#1) : (tensor<1x48x60x60xf16>, tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    // CHECK:      return [[CALL_PART2]] : tensor<1x48x60x60xf16>
    // CHECK:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @RepeatingBlocks {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    func.func private @main_fn1(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %reorder1 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %cst = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf16>, [#const.Reorder<#NHWC>]
        %conv = IE.Convolution(%reorder1, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %reorder2 = IE.Reorder(%conv) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %reorder2 : tensor<1x48x60x60xf16>
    }

    func.func @main(%input: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %softmax = IE.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        %call1 = call @main_fn1(%softmax) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %call2 = call @main_fn1(%call1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %call2 : tensor<1x48x60x60xf16>
    }

    // CHECK:  func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[REORDER1:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NHWC} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16, {order = #NHWC}>
    // CHECK:      [[CST:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}>
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[REORDER1]], [[CST]])
    // CHECK:      [[REORDER2:%.+]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
    // CHECK:      return [[REORDER2]]
    // CHECK:  }

    // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    // CHECK:      [[SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
    // CHECK:      [[CALL1:%.+]] = call @main_fn1([[SOFTMAX]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    // CHECK:      [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    // CHECK:      return [[CALL2]]
    // CHECK:  }
}
