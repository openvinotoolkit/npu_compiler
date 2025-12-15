//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --debatcher="debatcher-input-coefficients-partitions=[0-2],[0-2]" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @SingleInputSingleOutputBatched
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<6x3x62x62xf32>)
func.func @SingleInputSingleOutputBatched(%arg: tensor<6x3x62x62xf32>) -> tensor<6x48x60x57xf32> {
    %cst = const.Declare tensor<48x3x3x6xf32> = dense<1.0> : tensor<48x3x3x6xf32>
    %0 = IE.Convolution(%arg, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<6x3x62x62xf32>, tensor<48x3x3x6xf32> -> tensor<6x48x60x57xf32>
    %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<6x48x60x57xf32> -> tensor<6x48x60x57xf32>
    return %1 : tensor<6x48x60x57xf32>

    // CHECK: [[CASTED_ARG:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : tensor<6x3x62x62xf32> to tensor<2x3x62x62xf32>
    // CHECK: [[CASTED_CONST:%.+]] = builtin.unrealized_conversion_cast [[UNKN_VAR:%.+]] : tensor<2x48x60x57xf32> to tensor<6x48x60x57xf32>
    // CHECK: return [[CASTED_CONST]] : tensor<6x48x60x57xf32>
}

// -----

// CHECK-LABEL: @MultipleInputSingleOutputBatched
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<6x3x62x62xf32>,
// CHECK-SAME:     [[ARG1:%.+]]: tensor<6x48x60x57xf32>)
func.func @MultipleInputSingleOutputBatched(%arg0: tensor<6x3x62x62xf32>, %arg1: tensor<6x48x60x57xf32>) -> tensor<6x48x60x57xf32> {
        %cst = const.Declare tensor<48x3x3x6xf32> = dense<1.0> : tensor<48x3x3x6xf32>
        %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<6x3x62x62xf32>, tensor<48x3x3x6xf32> -> tensor<6x48x60x57xf32>
        %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<6x48x60x57xf32> -> tensor<6x48x60x57xf32>
        %2 = IE.Add(%1, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<6x48x60x57xf32>, tensor<6x48x60x57xf32> -> tensor<6x48x60x57xf32>
        %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<6x48x60x57xf32> -> tensor<6x48x60x57xf32>
        return %3: tensor<6x48x60x57xf32>

        // CHECK-DAG: [[CASTED_ARG0:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : tensor<6x3x62x62xf32> to tensor<2x3x62x62xf32>
        // CHECK-DAG: [[CASTED_ARG1:%.+]] = builtin.unrealized_conversion_cast [[ARG1]] : tensor<6x48x60x57xf32> to tensor<2x48x60x57xf32>
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
        // CHECK: [[CASTED_RES:%.+]] = builtin.unrealized_conversion_cast [[SOFTM_RES1]] : tensor<2x48x60x57xf32> to tensor<6x48x60x57xf32>
        // CHECK: return [[CASTED_RES]] : tensor<6x48x60x57xf32>
}
