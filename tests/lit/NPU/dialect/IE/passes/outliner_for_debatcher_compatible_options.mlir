//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --outliner="function-outlining=\"batching=''\"" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

module @SimpleModuleForOutlining attributes {VPU.debatch = 1 : i64} {

    net.NetworkInfo entryPoint : @main
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

// CHECK-LABEL: @SimpleModuleForOutlining

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
