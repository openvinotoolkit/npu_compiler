//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --debatcher="debatcher-input-coefficients-partitions=[0-2],[0-2]" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @SingleInputSingleOutputBatched
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]>)
func.func @SingleInputSingleOutputBatched(%arg: tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[6, 3, 62, 62]> : tensor<4xsi64>}>) -> tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}> {
    %cst = const.Declare tensor<48x3x3x6xf32> = dense<1.0> : tensor<48x3x3x6xf32>
    %0 = IE.Convolution(%arg, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[6, 3, 62, 62]> : tensor<4xsi64>}>, tensor<48x3x3x6xf32> -> tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}>
    %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}> -> tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}>
    return %1 : tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}>

    // CHECK: [[CASTED_ARG:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]> to tensor<2x3x62x62xf32>
    // CHECK: [[CASTED_CONST:%.+]] = builtin.unrealized_conversion_cast [[UNKN_VAR:%.+]] : tensor<2x48x60x57xf32> to tensor<?x48x60x57xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: return [[CASTED_CONST]] : tensor<?x48x60x57xf32, [[ANY_MATCH:{.+}]]>
}

// -----

// CHECK-LABEL: @MultipleInputSingleOutputBatched
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]>,
// CHECK-SAME:     [[ARG1:%.+]]: tensor<?x48x60x57xf32, [[ANY_MATCH:{.+}]]>
func.func @MultipleInputSingleOutputBatched(%arg0: tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[6, 3, 62, 62]> : tensor<4xsi64>}>, %arg1: tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}>) -> tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}> {
        %cst = const.Declare tensor<48x3x3x6xf32> = dense<1.0> : tensor<48x3x3x6xf32>
        %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[6, 3, 62, 62]> : tensor<4xsi64>}>, tensor<48x3x3x6xf32> -> tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}>
        %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}> -> tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}>
        %2 = IE.Add(%1, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}>, tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}> -> tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}>
        %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}> -> tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}>
        return %3: tensor<?x48x60x57xf32, {bounds = #const.OpaqueI64Elements<[6, 48, 60, 57]> : tensor<4xsi64>}>

        // CHECK-DAG: [[CASTED_ARG0:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]> to tensor<2x3x62x62xf32>
        // CHECK-DAG: [[CASTED_ARG1:%.+]] = builtin.unrealized_conversion_cast [[ARG1]] : tensor<?x48x60x57xf32, [[ANY_MATCH:{.+}]]> to tensor<2x48x60x57xf32>
        // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<48x3x3x6xf32> = [[ANY_DATA:.*]] : tensor<48x3x3x6xf32>
        // CHECK: [[CONV_RES:%.+]] = IE.Convolution([[CASTED_ARG0]], [[CST]]) {
        // CHECK-SAME:              dilations = [1, 1],
        // CHECK-SAME:              pads_begin = [0, 0],
        // CHECK-SAME:              pads_end = [0, 0],
        // CHECK-SAME:              strides = [1, 1]
        // CHECK-SAME:              } : tensor<2x3x62x62xf32>, tensor<48x3x3x6xf32> -> tensor<2x48x60x57xf32>
        // CHECK: [[SOFTM_RES0:%.+]] = IE.SoftMax([[CONV_RES]]) {axisInd = 1 : i64} : tensor<2x48x60x57xf32> -> tensor<2x48x60x57xf32>
        // CHECK: [[ADD_RES:%.+]] = IE.Add([[SOFTM_RES0]], [[CASTED_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x48x60x57xf32>, tensor<2x48x60x57xf32> -> tensor<2x48x60x57xf32>
        // CHECK: [[SOFTM_RES1:%.+]] = IE.SoftMax([[ADD_RES]]) {axisInd = 1 : i64} : tensor<2x48x60x57xf32> -> tensor<2x48x60x57xf32>
        // CHECK: [[CASTED_RES:%.+]] = builtin.unrealized_conversion_cast [[SOFTM_RES1]] : tensor<2x48x60x57xf32> to tensor<?x48x60x57xf32, [[ANY_MATCH:{.+}]]>
        // CHECK: return [[CASTED_RES]] : tensor<?x48x60x57xf32, [[ANY_MATCH:{.+}]]>
}
