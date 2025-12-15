//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --debatcher %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @SingleInputSingleOutputBatched
func.func @SingleInputSingleOutputBatched(%arg: tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>) -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
    %0 = IE.Convolution(%arg, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, tensor<48x3x3x3xf32> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    return %1 : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

    // CHECK-DAG: [[VAL0:%0]] = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x62x62xf32>
    // CHECK: [[VAL1:%1]] = IE.Convolution([[VAL0]], %cst) {
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0],
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK: [[VAL2:%2]] = IE.SoftMax([[VAL1]]) {axisInd = 3 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK: [[VAL3:%3]] = builtin.unrealized_conversion_cast [[VAL2]] : tensor<1x48x60x60xf32> to tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: return [[VAL3]] : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
}

// CHECK-LABEL: @MultipleInputSingleOutputBatched
func.func @MultipleInputSingleOutputBatched(%arg0: tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, %arg1: tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, tensor<48x3x3x3xf32> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
        %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
        %2 = IE.Add(%1, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
        %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
        return %3: tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

        // CHECK-DAG: [[VAL0:%.+]] = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x62x62xf32>
        // CHECK-DAG: [[VAL1:%.+]] = builtin.unrealized_conversion_cast %arg1 : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]> to tensor<1x48x60x60xf32>
        // CHECK: [[VAL2:%.+]] = IE.Convolution([[VAL0]], %cst) {
        // CHECK-SAME:              dilations = [1, 1],
        // CHECK-SAME:              pads_begin = [0, 0],
        // CHECK-SAME:              pads_end = [0, 0],
        // CHECK-SAME:              strides = [1, 1]
        // CHECK-SAME:              } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        // CHECK: [[VAL3:%.+]] = IE.SoftMax([[VAL2]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        // CHECK: [[VAL4:%.+]] = IE.Add([[VAL3]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        // CHECK: [[VAL5:%.+]] = IE.SoftMax([[VAL4]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        // CHECK: [[VAL6:%.+]] = builtin.unrealized_conversion_cast [[VAL5]] : tensor<1x48x60x60xf32> to tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
        // CHECK: return [[VAL6]] : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>

}

// -----

// CHECK-LABEL: @MultipleInputMultipleOutputBatched
func.func @MultipleInputMultipleOutputBatched(%arg0: tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, %arg1: tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) -> (tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, tensor<48x3x3x3xf32> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
        %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
        %2 = IE.Add(%1, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
        %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
        return %3, %1: tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

        // CHECK-DAG: [[VAL0:%.+]] = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x62x62xf32>
        // CHECK-DAG: [[VAL1:%.+]] = builtin.unrealized_conversion_cast %arg1 : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]> to tensor<1x48x60x60xf32>
        // CHECK: [[VAL2:%.+]] = IE.Convolution([[VAL0]], %cst) {
        // CHECK-SAME:              dilations = [1, 1],
        // CHECK-SAME:              pads_begin = [0, 0],
        // CHECK-SAME:              pads_end = [0, 0],
        // CHECK-SAME:              strides = [1, 1]
        // CHECK-SAME:              } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        // CHECK: [[VAL3:%.+]] = IE.SoftMax([[VAL2]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        // CHECK: [[VAL4:%.+]] = IE.Add([[VAL3]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        // CHECK: [[VAL5:%.+]] = IE.SoftMax([[VAL4]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        // CHECK: [[VAL6:%.+]] = builtin.unrealized_conversion_cast [[VAL5]] : tensor<1x48x60x60xf32> to tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
        // CHECK: [[VAL7:%.+]] = builtin.unrealized_conversion_cast [[VAL3]] : tensor<1x48x60x60xf32> to tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
        // CHECK: return [[VAL6]], [[VAL7]] : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
}

// -----

// CHECK-LABEL: @SingleInputSingleOutputDynamicReshapeOnlyBatchDynamic
func.func @SingleInputSingleOutputDynamicReshapeOnlyBatchDynamic(%arg: tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>) -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<1> : tensor<2xsi64>
    %0 = IE.Convolution(%arg, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, tensor<48x3x3x3xf32> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %1 = IE.DynamicReshape(%0, %cst_1) {output_bounds = [3, 48, 3600], output_shape = [-9223372036854775808, 48, 3600]} : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<2xsi64> -> tensor<?x48x3600xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 3600]> : tensor<3xsi64>}>
    %2 = IE.DynamicReshape(%1, %cst_1) {output_bounds = [3, 48, 60, 60], output_shape = [-9223372036854775808, 48, 60, 60]} : tensor<?x48x3600xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 3600]> : tensor<3xsi64>}>, tensor<2xsi64> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %3 = IE.SoftMax(%2) {axisInd = 3 : i64} : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    return %3 : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

    // CHECK-DAG: [[VAL0:%0]] = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x62x62xf32>
    // CHECK: [[VAL1:%1]] = IE.Convolution([[VAL0]], %cst) {
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0],
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
    // CHECK: [[VAL2:%.+]] = IE.Reshape([[VAL1]]) {shape_value = [1, 48, 3600]} : tensor<1x48x60x60xf32> -> tensor<1x48x3600xf32>
    // CHECK: [[VAL3:%.+]] = IE.Reshape([[VAL2]]) {shape_value = [1, 48, 60, 60]} : tensor<1x48x3600xf32> -> tensor<1x48x60x60xf32>
    // CHECK: [[VAL4:%.+]] = IE.SoftMax([[VAL3]]) {axisInd = 3 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK: [[VAL5:%.+]] = builtin.unrealized_conversion_cast [[VAL4]] : tensor<1x48x60x60xf32> to tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: return [[VAL5]] : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
}

// -----

// CHECK-LABEL: @SingleInputSingleOutputDynamicReshapeMultiDynamicDims
func.func @SingleInputSingleOutputDynamicReshapeMultiDynamicDims(%arg: tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>) -> tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
    %cst_1 = const.Declare tensor<2xsi64> = dense<1> : tensor<2xsi64>
    %0 = IE.Convolution(%arg, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, tensor<48x3x3x3xf32> -> tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %1 = IE.DynamicReshape(%0, %cst_1) {output_bounds = [3, 48, 3600], output_shape = [-9223372036854775808, 48, -9223372036854775808]} : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<2xsi64> -> tensor<?x48x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 3600]> : tensor<3xsi64>}>
    %2 = IE.DynamicReshape(%1, %cst_1) {output_bounds = [3, 48, 60, 60], output_shape = [-9223372036854775808, 48, -9223372036854775808, -9223372036854775808]} : tensor<?x48x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 3600]> : tensor<3xsi64>}>, tensor<2xsi64> -> tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %3 = IE.SoftMax(%2) {axisInd = 3 : i64} : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    return %3 : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

    // CHECK-DAG: [[VAL0:%.+]] = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: [[VAL1:%.+]] = IE.Convolution([[VAL0]], %cst) {
    // CHECK-SAME:              dilations = [1, 1],
    // CHECK-SAME:              pads_begin = [0, 0],
    // CHECK-SAME:              pads_end = [0, 0],
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              } : tensor<1x3x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<48x3x3x3xf32> -> tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: [[VAL2:%.+]] = IE.DynamicReshape([[VAL1]], [[ANY_CNST:%.+]]) {output_bounds = [1, 48, 3600], output_shape = [1, 48, -9223372036854775808]} : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<2xsi64> -> tensor<1x48x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: [[VAL3:%.+]] = IE.DynamicReshape([[VAL2]], [[ANY_CNST:%.+]]) {output_bounds = [1, 48, 60, 60], output_shape = [1, 48, -9223372036854775808, -9223372036854775808]} : tensor<1x48x?xf32, [[ANY_MATCH:{.+}]]>, tensor<2xsi64> -> tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: [[VAL4:%.+]] = IE.SoftMax([[VAL3]]) {axisInd = 3 : i64} : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]> -> tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: [[VAL5:%.+]] = builtin.unrealized_conversion_cast [[VAL4]] : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]> to tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: return [[VAL5]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
}
