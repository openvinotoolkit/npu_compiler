//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --de-debatcher="debatching-inlining-method=host_pipeline" --canonicalize --cse --verify-diagnostics %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

net.NetworkInfo entryPoint : @SingleInputMultipleOutputMultiDynamicDimDeBatched inputsInfo : {
    DataInfo "input" : tensor<3x3x62x62xf32>
} outputsInfo : {
    DataInfo "output0" : tensor<3x48x60x60xf32>
    DataInfo "output1" : tensor<3x48x60x60xf16>
}
func.func @output_shape(%arg0: tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>) -> (tensor<4xi64>, tensor<4xi64>) {
    %c48_i64 = arith.constant 48 : i64
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %dim_n = tensor.dim %arg0, %c0 : tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>
    %n = arith.index_cast %dim_n : index to i64
    %dim_h = tensor.dim %arg0, %c2 : tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>
    %h = arith.index_cast %dim_h : index to i64
    %dim_w = tensor.dim %arg0, %c3 : tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>
    %w = arith.index_cast %dim_w : index to i64
    %from_elements = tensor.from_elements %n, %c48_i64, %h, %w : tensor<4xi64>
    return %from_elements, %from_elements : tensor<4xi64>, tensor<4xi64>
}

func.func nested @SingleInputMultipleOutputMultiDynamicDimDeBatched_Batch1(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>) -> (tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>) {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>, tensor<48x3x3x3xf32> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %4 = IE.SoftMax(%3) {axisInd = 1 : i64} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %5 = IE.Convert(%4) {dstElemType = f16} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    return %4, %5 : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
}

// CHECK-LABEL: @SingleInputMultipleOutputMultiDynamicDimDeBatched
func.func @SingleInputMultipleOutputMultiDynamicDimDeBatched(%arg0: tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>) -> (tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}> to tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>
    %1:2 = call @SingleInputMultipleOutputMultiDynamicDimDeBatched_Batch1(%0) : (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>) -> (tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>)
    %2 = builtin.unrealized_conversion_cast %1#0: tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> to tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %3 = builtin.unrealized_conversion_cast %1#1: tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> to tensor<?x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    return %2, %3 : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

    // CHECK: func.func @SingleInputMultipleOutputMultiDynamicDimDeBatched([[ARG0:%.+]]: tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>) -> (tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>)
    // CHECK-DAG:   [[DYN_W_IDX:%.+]] = arith.constant 3 : index
    // CHECK-DAG:   [[DYN_H_IDX:%.+]] = arith.constant 2 : index
    // CHECK:   [[STEP:%.+]] = arith.constant 1 : index
    // CHECK:   [[BEGIN:%.+]] = arith.constant 0 : index
    // CHECK:   [[END:%.+]] = tensor.dim [[ARG0]], [[ANY_MATCH:%.+]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_MIN_H:%.+]] = tensor.dim [[ARG0]], [[DYN_H_IDX]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_MIN_W:%.+]] = tensor.dim [[ARG0]], [[DYN_W_IDX]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[OUTPUT_0:%.+]] = tensor.empty([[END]], [[DYN_MIN_H]], [[DYN_MIN_W]]) : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[OUTPUT_1:%.+]] = tensor.empty([[END]], [[DYN_MIN_H]], [[DYN_MIN_W]]) : tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[RET:%.+]]:2 = scf.for [[IND_VAR:%.+]] = [[BEGIN]] to [[END]] step [[STEP]] iter_args([[LOOP_CARRIED_0:%.+]] = [[OUTPUT_0]], [[LOOP_CARRIED_1:%.+]] = [[OUTPUT_1]]) -> (tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>) {
    // CHECK:       [[I_SLICE:%.+]] = tensor.extract_slice [[ARG0]][[[IND_VAR]], 0, 0, 0] [1, 3, [[DYN_MIN_H]], [[DYN_MIN_W]]] [1, 1, 1, 1] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[FUNC:%.+]]:2 = func.call @SingleInputMultipleOutputMultiDynamicDimDeBatched_Batch[[ANY_SUFFIX:.*]]([[I_SLICE]]) : (tensor<1x3x?x?xf32, [[ANY_MATCH:{.+}]]>) -> (tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]>)
    // CHECK:       [[O0_DYN_H_DIM:%.+]] = tensor.dim [[LOOP_CARRIED_0]], [[DYN_H_IDX]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O0_DYN_W_DIM:%.+]] = tensor.dim [[LOOP_CARRIED_0]], [[DYN_W_IDX]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O_SLICE_0:%.+]] = tensor.insert_slice [[FUNC]]#0 into [[LOOP_CARRIED_0]][[[IND_VAR]], 0, 0, 0] [1, 48, [[O0_DYN_H_DIM]], [[O0_DYN_W_DIM]]] [1, 1, 1, 1] : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]> into tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O1_DYN_H_DIM:%.+]] = tensor.dim [[LOOP_CARRIED_1]], [[DYN_H_IDX]] : tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O1_DYN_W_DIM:%.+]] = tensor.dim [[LOOP_CARRIED_1]], [[DYN_W_IDX]] : tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O_SLICE_1:%.+]] = tensor.insert_slice [[FUNC]]#1 into [[LOOP_CARRIED_1]][[[IND_VAR]], 0, 0, 0] [1, 48, [[O1_DYN_H_DIM]], [[O1_DYN_W_DIM]]] [1, 1, 1, 1] : tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]> into tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       scf.yield [[O_SLICE_0]], [[O_SLICE_1]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:   return [[RET]]#0, [[RET]]#1 : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
}

// -----

net.NetworkInfo entryPoint : @MultipleInputMultipleOutputMultiDynamicDimDeBatched inputsInfo : {
    DataInfo "input0" : tensor<3x3x62x62xf32>
    DataInfo "input1" : tensor<3x48x60x60xf32>
} outputsInfo : {
    DataInfo "output0" : tensor<3x48x60x60xf32>
    DataInfo "output1" : tensor<3x48x60x60xf16>
}
func.func @output_shape(%arg0: tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, %arg1: tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) -> (tensor<4xi64>, tensor<4xi64>) {
    %c48_i64 = arith.constant 48 : i64
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %out0_dim_n = tensor.dim %arg0, %c0 : tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>
    %out0_n = arith.index_cast %out0_dim_n : index to i64
    %out0_dim_h = tensor.dim %arg0, %c2 : tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>
    %out0_h = arith.index_cast %out0_dim_h : index to i64
    %out0_dim_w = tensor.dim %arg0, %c3 : tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>
    %out0_w = arith.index_cast %out0_dim_w : index to i64
    %out0_from_elements = tensor.from_elements %out0_n, %c48_i64, %out0_h, %out0_w : tensor<4xi64>
    %out1_dim_n = tensor.dim %arg1, %c0 : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %out1_n = arith.index_cast %out1_dim_n : index to i64
    %out1_dim_h = tensor.dim %arg1, %c2 : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %out1_h = arith.index_cast %out1_dim_h : index to i64
    %out1_dim_w = tensor.dim %arg1, %c3 : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %out1_w = arith.index_cast %out1_dim_w : index to i64
    %out1_from_elements = tensor.from_elements %out1_n, %c48_i64, %out1_h, %out1_w : tensor<4xi64>
    return %out0_from_elements, %out1_from_elements : tensor<4xi64>, tensor<4xi64>
}

func.func nested @MultipleInputMultipleOutputMultiDynamicDimDeBatched_Batch1(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>, %arg1: tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>) -> (tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>) {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>, tensor<48x3x3x3xf32> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %3 = IE.Add(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %4 = IE.SoftMax(%3) {axisInd = 1 : i64} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %5 = IE.Convert(%4) {dstElemType = f16} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    return %4, %5 : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
}

// CHECK-LABEL: @MultipleInputMultipleOutputMultiDynamicDimDeBatched
func.func @MultipleInputMultipleOutputMultiDynamicDimDeBatched(%arg0: tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, %arg1: tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) -> (tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}> to tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>
    %1 = builtin.unrealized_conversion_cast %arg1 : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> to tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %2:2 = call @MultipleInputMultipleOutputMultiDynamicDimDeBatched_Batch1(%0, %1) : (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>, tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>) -> (tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>)
    %3 = builtin.unrealized_conversion_cast %2#0: tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> to tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %4 = builtin.unrealized_conversion_cast %2#1: tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> to tensor<?x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    return %3, %4 : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>

    // CHECK: func.func @MultipleInputMultipleOutputMultiDynamicDimDeBatched([[ARG0:%.+]]: tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>, [[ARG1:%.+]]: tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>) -> (tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>)
    // CHECK-DAG:   [[DYN_W_IDX:%.+]] = arith.constant 3 : index
    // CHECK-DAG:   [[DYN_H_IDX:%.+]] = arith.constant 2 : index
    // CHECK:   [[STEP:%.+]] = arith.constant 1 : index
    // CHECK:   [[BEGIN:%.+]] = arith.constant 0 : index
    // CHECK:   [[END:%.+]] = tensor.dim [[ARG0]], [[BEGIN]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_H_0:%.+]] = tensor.dim [[ARG0]], [[DYN_H_IDX]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_W_0:%.+]] = tensor.dim [[ARG0]], [[DYN_W_IDX]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_N_1:%.+]] = tensor.dim [[ARG1]], [[BEGIN]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_H_1:%.+]] = tensor.dim [[ARG1]], [[DYN_H_IDX]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_W_1:%.+]] = tensor.dim [[ARG1]], [[DYN_W_IDX]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[OUTPUT_0:%.+]] = tensor.empty([[END]], [[DYN_H_0]], [[DYN_W_0]]) : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[OUTPUT_1:%.+]] = tensor.empty([[DYN_N_1]], [[DYN_H_1]], [[DYN_W_1]]) : tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[RET:%.+]]:2 = scf.for [[IND_VAR:%.+]] = [[BEGIN]] to [[END]] step [[STEP]] iter_args([[LOOP_CARRIED_0:%.+]] = [[OUTPUT_0]], [[LOOP_CARRIED_1:%.+]] = [[OUTPUT_1]]) -> (tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>) {
    // CHECK:       [[I_SLICE_0:%.+]] = tensor.extract_slice [[ARG0]][[[IND_VAR]], 0, 0, 0] [1, 3, [[DYN_H_0]], [[DYN_W_0]]] [1, 1, 1, 1] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[I_SLICE_1:%.+]] = tensor.extract_slice [[ARG1]][[[IND_VAR]], 0, 0, 0] [1, 48, [[DYN_H_1]], [[DYN_W_1]]] [1, 1, 1, 1] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]> to tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[FUNC:%.+]]:2 = func.call @MultipleInputMultipleOutputMultiDynamicDimDeBatched_Batch[[ANY_SUFFIX:.*]]([[I_SLICE_0]], [[I_SLICE_1]]) : (tensor<1x3x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>) -> (tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]>)
    // CHECK:       [[O0_DYN_H_DIM:%.+]] = tensor.dim [[LOOP_CARRIED_0]], [[DYN_H_IDX]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O0_DYN_W_DIM:%.+]] = tensor.dim [[LOOP_CARRIED_0]], [[DYN_W_IDX]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O_SLICE_0:%.+]] = tensor.insert_slice [[FUNC]]#0 into [[LOOP_CARRIED_0]][[[IND_VAR]], 0, 0, 0] [1, 48, [[O0_DYN_H_DIM]], [[O0_DYN_W_DIM]]] [1, 1, 1, 1] : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]> into tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O1_DYN_H_DIM:%.+]] = tensor.dim [[LOOP_CARRIED_1]], [[DYN_H_IDX]] : tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O1_DYN_W_DIM:%.+]] = tensor.dim [[LOOP_CARRIED_1]], [[DYN_W_IDX]] : tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O_SLICE_1:%.+]] = tensor.insert_slice [[FUNC]]#1 into [[LOOP_CARRIED_1]][[[IND_VAR]], 0, 0, 0] [1, 48, [[O1_DYN_H_DIM]], [[O1_DYN_W_DIM]]] [1, 1, 1, 1] : tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]> into tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       scf.yield [[O_SLICE_0]], [[O_SLICE_1]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:   return [[RET]]#0, [[RET]]#1 : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
}


// -----
// expected-error@-1 {{HostCompile pipeline must provide the "output_shape" function}}

net.NetworkInfo entryPoint : @main_error_case inputsInfo : {
    DataInfo "input0" : tensor<3x3x62x62xf32>
    DataInfo "input1" : tensor<3x48x60x60xf32>
} outputsInfo : {
    DataInfo "output0" : tensor<3x48x60x60xf32>
    DataInfo "output1" : tensor<3x48x60x60xf16>
}
func.func nested @debatched_error_case(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>, %arg1: tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>) -> (tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>) {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
    %1 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>, tensor<48x3x3x3xf32> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %3 = IE.Add(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %4 = IE.SoftMax(%3) {axisInd = 1 : i64} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %5 = IE.Convert(%4) {dstElemType = f16} : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> -> tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    return %4, %5 : tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
}

func.func @main_error_case(%arg0: tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}>, %arg1: tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) -> (tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<?x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 3, 62, 62]> : tensor<4xsi64>}> to tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>
    %1 = builtin.unrealized_conversion_cast %arg1 : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}> to tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>
    %2:2 = call @debatched_error_case(%0, %1) : (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>, tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>) -> (tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>)
    %3 = builtin.unrealized_conversion_cast %2#0: tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> to tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    %4 = builtin.unrealized_conversion_cast %2#1: tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}> to tensor<?x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
    return %3, %4 : tensor<?x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>, tensor<?x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[3, 48, 60, 60]> : tensor<4xsi64>}>
}
