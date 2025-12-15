//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --de-debatcher="debatching-inlining-method=host_pipeline" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

func.func private @SingleInputSingleOutputDeBatchedTo1_Batch1(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    return %2 : tensor<1x48x60x60xf32>
}

// CHECK-LABEL: @SingleInputSingleOutputDeBatchedTo1
func.func @SingleInputSingleOutputDeBatchedTo1(%arg0: tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>) -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}> to tensor<1x3x62x62xf32>
    %1 = call @SingleInputSingleOutputDeBatchedTo1_Batch1(%0) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
    %2 = builtin.unrealized_conversion_cast %1: tensor<1x48x60x60xf32> to tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    return %2 : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

    // CHECK: func.func @SingleInputSingleOutputDeBatchedTo1([[ARG0:%.+]]: tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]>) -> tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]> {
    // CHECK: [[BEGIN:%.+]] = arith.constant 0 : index
    // CHECK: [[END:%.+]] = tensor.dim [[ARG0]], [[ANY:%.*]] : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: [[STEP:%.+]] = arith.constant 1 : index
    // CHECK: [[OUTPUT:%.+]] = tensor.empty([[ANY:%.*]]) : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: [[RET:%.+]] = scf.for [[IND_VAR:%.+]] = [[BEGIN]] to %dim step [[STEP]] iter_args([[LOOP_CARRIED:%.+]] = [[OUTPUT]]) -> (tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>) {
    // CHECK:   [[I_SLICE:%.+]] = tensor.extract_slice [[ARG0]][[[IND_VAR]], 0, 0, 0] [1, 3, 62, 62] [1, 1, 1, 1] : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x62x62xf32>
    // CHECK:   [[FUNC:%.+]] = func.call @SingleInputSingleOutputDeBatchedTo1_Batch1([[I_SLICE]]) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:   [[O_SLICE:%.+]] = tensor.insert_slice [[FUNC]] into [[LOOP_CARRIED]][[[IND_VAR]], 0, 0, 0] [1, 48, 60, 60] [1, 1, 1, 1] : tensor<1x48x60x60xf32> into tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   scf.yield [[O_SLICE]] : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>

    // CHECK: return [[RET]]
}

// -----


func.func private @MultipleInputSingleOutputDeBatched_Batch1(%arg0: tensor<1x3x62x62xf32>, %arg1: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %3 = IE.Add(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %4 = IE.SoftMax(%3) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    return %4 : tensor<1x48x60x60xf32>
}

// CHECK-LABEL: @MultipleInputSingleOutputDeBatched
func.func @MultipleInputSingleOutputDeBatched(%arg0: tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, %arg1: tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}> to tensor<1x3x62x62xf32>
    %1 = builtin.unrealized_conversion_cast %arg1 : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> to tensor<1x48x60x60xf32>
    %2 = call @MultipleInputSingleOutputDeBatched_Batch1(%0, %1) : (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    %3 = builtin.unrealized_conversion_cast %2: tensor<1x48x60x60xf32> to tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    return %3 : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

    // CHECK: func.func @MultipleInputSingleOutputDeBatched([[ARG0:%.+]]: tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]>, [[ARG1:%.+]]: tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>) -> tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]> {
    // CHECK: [[BEGIN:%.+]] = arith.constant 0 : index
    // CHECK: [[END:%.+]] = tensor.dim [[ARG0]], [[ANY:%.*]] : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: [[STEP:%.+]] = arith.constant 1 : index
    // CHECK: [[OUTPUT:%.+]] = tensor.empty([[ANY:%.*]]) : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK: [[RET:%.+]] = scf.for [[IND_VAR:%.+]] = [[ANY_MATCH:%.+]] to [[END]] step [[STEP]] iter_args([[LOOP_CARRIED:%.+]] = [[OUTPUT]]) -> (tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>) {
    // CHECK:       [[I_SLICE_0:%.+]] = tensor.extract_slice [[ARG0]][[[IND_VAR]], 0, 0, 0] [1, 3, 62, 62] [1, 1, 1, 1] : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x62x62xf32>
    // CHECK:       [[I_SLICE_1:%.+]] = tensor.extract_slice [[ARG1]][[[IND_VAR]], 0, 0, 0] [1, 48, 60, 60] [1, 1, 1, 1] : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]> to tensor<1x48x60x60xf32>
    // CHECK:       [[FUNC:%.+]] = func.call @MultipleInputSingleOutputDeBatched_Batch1([[I_SLICE_0]], [[I_SLICE_1]]) : (tensor<1x3x62x62xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:       [[O_SLICE:%.+]] = tensor.insert_slice [[FUNC]] into [[LOOP_CARRIED]][[[IND_VAR]], 0, 0, 0] [1, 48, 60, 60] [1, 1, 1, 1] : tensor<1x48x60x60xf32> into tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       scf.yield [[O_SLICE]] : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   return [[RET]]
}

// -----

func.func private @SingleInputMultipleOutputDeBatched_Batch1(%arg0: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf16>) {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %4 = IE.SoftMax(%3) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
    %5 = IE.Convert(%4) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
    return %4, %5 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf16>
}

// CHECK-LABEL: @SingleInputMultipleOutputDeBatched
func.func @SingleInputMultipleOutputDeBatched(%arg0: tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>) -> (tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x60x60xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}> to tensor<1x3x62x62xf32>
    %1:2 = call @SingleInputMultipleOutputDeBatched_Batch1(%0) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf16>)
    %2 = builtin.unrealized_conversion_cast %1#0: tensor<1x48x60x60xf32> to tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %3 = builtin.unrealized_conversion_cast %1#1: tensor<1x48x60x60xf16> to tensor<?x48x60x60xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    return %2, %3 : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x60x60xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

    // CHECK: func.func @SingleInputMultipleOutputDeBatched([[ARG0:%.+]]: tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]>) -> (tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x60x60xf16, [[ANY_MATCH:{.+}]]>) {
    // CHECK:   [[BEGIN:%.+]] = arith.constant 0 : index
    // CHECK:   [[END:%.+]] = tensor.dim [[ARG0]], [[ANY_MATCH:%.*]] : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[STEP:%.+]] = arith.constant 1 : index
    // CHECK:   [[OUTPUT_0:%.+]] = tensor.empty([[ANY:.*]]) : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[OUTPUT_1:%.+]] = tensor.empty([[ANY:.*]]) : tensor<?x48x60x60xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[RET:%.+]]:2 = scf.for [[IND_VAR:%.+]] = [[BEGIN]] to %dim step [[STEP]] iter_args([[LOOP_CARRIED_0:%.+]] = [[OUTPUT_0]], [[LOOP_CARRIED_1:%.+]] = [[OUTPUT_1]]) -> (tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x60x60xf16, [[ANY_MATCH:{.+}]]>) {
    // CHECK:       [[I_SLICE:%.+]] = tensor.extract_slice [[ARG0]][[[IND_VAR]], 0, 0, 0] [1, 3, 62, 62] [1, 1, 1, 1] : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x62x62xf32>
    // CHECK:       [[FUNC:%.+]]:2 = func.call @SingleInputMultipleOutputDeBatched_Batch1([[I_SLICE]]) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf16>)
    // CHECK:       [[O_SLICE_0:%.+]] = tensor.insert_slice [[FUNC]]#0 into [[LOOP_CARRIED_0]][[[IND_VAR]], 0, 0, 0] [1, 48, 60, 60] [1, 1, 1, 1] : tensor<1x48x60x60xf32> into tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O_SLICE_1:%.+]] = tensor.insert_slice [[FUNC]]#1 into [[LOOP_CARRIED_1]][[[IND_VAR]], 0, 0, 0] [1, 48, 60, 60] [1, 1, 1, 1] : tensor<1x48x60x60xf16> into tensor<?x48x60x60xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       scf.yield [[O_SLICE_0]], [[O_SLICE_1]] : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x60x60xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:   return [[RET]]#0, [[RET]]#1 : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x60x60xf16, [[ANY_MATCH:{.+}]]>
}

// -----

func.func private @SingleInputSingleOutputDeBatchedTo2_Batch2(%arg0: tensor<2x3x62x62xf32>) -> tensor<2x48x60x60xf32> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<2x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<2x48x60x60xf32>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<2x48x60x60xf32> -> tensor<2x48x60x60xf32>
    return %2 : tensor<2x48x60x60xf32>
}

// CHECK-LABEL: @SingleInputSingleOutputDeBatchedTo2
func.func @SingleInputSingleOutputDeBatchedTo2(%arg0: tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>) -> tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x62x62xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}> to tensor<2x3x62x62xf32>
    %1 = call @SingleInputSingleOutputDeBatchedTo2_Batch2(%0) : (tensor<2x3x62x62xf32>) -> tensor<2x48x60x60xf32>
    %2 = builtin.unrealized_conversion_cast %1: tensor<2x48x60x60xf32> to tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    return %2 : tensor<?x48x60x60xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

    // CHECK: func.func @SingleInputSingleOutputDeBatchedTo2([[ARG0:%.+]]: tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]>) -> tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]> {
    // CHECK:   [[BEGIN:%.+]] = arith.constant 0 : index
    // CHECK:   [[END:%.+]] = tensor.dim [[ARG0]], [[ANY_MATCH:%.*]] : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[STEP:%.+]] = arith.constant 2 : index
    // CHECK:   [[OUTPUT:%.+]] = tensor.empty([[ANY:.*]]) : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[RET:%.+]] = scf.for [[IND_VAR:%.+]] = [[BEGIN]] to %dim step [[STEP]] iter_args([[LOOP_CARRIED:%.+]] = [[OUTPUT]]) -> (tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>) {
    // CHECK:       [[I_SLICE:%.+]] = tensor.extract_slice [[ARG0]][[[IND_VAR]], 0, 0, 0] [2, 3, 62, 62] [1, 1, 1, 1] : tensor<?x3x62x62xf32, [[ANY_MATCH:{.+}]]> to tensor<2x3x62x62xf32>
    // CHECK:       [[FUNC:%.+]] = func.call @SingleInputSingleOutputDeBatchedTo2_Batch2([[I_SLICE]]) : (tensor<2x3x62x62xf32>) -> tensor<2x48x60x60xf32>
    // CHECK:       [[O_SLICE:%.+]] = tensor.insert_slice [[FUNC]] into [[LOOP_CARRIED]][[[IND_VAR]], 0, 0, 0] [2, 48, 60, 60] [1, 1, 1, 1] : tensor<2x48x60x60xf32> into tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       scf.yield [[O_SLICE]] : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   return [[RET]] : tensor<?x48x60x60xf32, [[ANY_MATCH:{.+}]]>
}
