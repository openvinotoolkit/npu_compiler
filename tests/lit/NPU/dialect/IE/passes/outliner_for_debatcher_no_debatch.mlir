//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --outliner="function-outlining=\"naive=''\"" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module @ValidModuleWithAttrBut_NoDebatchOptions attributes {config.debatch} {

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

    // CHECK: func.func nested @main_part1
    // CHECK: func.func nested @main_part2

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<3x3x62x62xf32>) -> tensor<3x48x60x60xf32> {
    // CHECK: [[PART_0:%.+]] = call @main_part1([[ARG0]]) : (tensor<3x3x62x62xf32>) -> tensor<1x48x60x60xf32>
    // CHECK: [[PART_1:%.+]] = call @main_part2([[PART_0]]) : (tensor<1x48x60x60xf32>) -> tensor<3x48x60x60xf32>
    // CHECK: return [[PART_1]] : tensor<3x48x60x60xf32>

}
