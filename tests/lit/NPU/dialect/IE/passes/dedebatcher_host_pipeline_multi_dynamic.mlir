//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --de-debatcher="debatching-inlining-method=host_pipeline" --canonicalize --cse %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// -----

func.func private @SingleInputMultipleOutputMultiDynamicDimDeBatched_Batch1(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>) -> (tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>) {
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

    // CHECK: func.func @SingleInputMultipleOutputMultiDynamicDimDeBatched([[ARG0:%.+]]: tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>) -> (tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>) {
    // CHECK:   [[DYN_W_IDX:%.+]] = arith.constant 3 : index
    // CHECK:   [[DYN_H_IDX:%.+]] = arith.constant 2 : index
    // CHECK:   [[STEP:%.+]] = arith.constant 1 : index
    // CHECK:   [[BEGIN:%.+]] = arith.constant 0 : index
    // CHECK:   [[END:%.+]] = tensor.dim [[ARG0]], [[ANY_MATCH:%.*]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_MIN_H:%.+]] = tensor.dim [[ARG0]], [[DYN_H_IDX]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_MIN_W:%.+]] = tensor.dim [[ARG0]], [[DYN_W_IDX]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[OUTPUT_0:%.+]] = tensor.empty([[END]], [[DYN_MIN_H]], [[DYN_MIN_W]]) : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[OUTPUT_1:%.+]] = tensor.empty([[END]], [[DYN_MIN_H]], [[DYN_MIN_W]]) : tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[RET:%.+]]:2 = scf.for [[IND_VAR:%.+]] = [[BEGIN]] to %dim step [[STEP]] iter_args([[LOOP_CARRIED_0:%.+]] = [[OUTPUT_0]], [[LOOP_CARRIED_1:%.+]] = [[OUTPUT_1]]) -> (tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>) {
    // CHECK:       [[I_SLICE:%.+]] = tensor.extract_slice [[ARG0]][[[IND_VAR]], 0, 0, 0] [1, 3, [[DYN_MIN_H]], [[DYN_MIN_W]]] [1, 1, 1, 1] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[FUNC:%.+]]:2 = func.call @SingleInputMultipleOutputMultiDynamicDimDeBatched_Batch1([[I_SLICE]]) : (tensor<1x3x?x?xf32, [[ANY_MATCH:{.+}]]>) -> (tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]>)
    // CHECK:       [[O0_DYN_H_DIM:%.+]] = tensor.dim [[FUNC]]#0, [[DYN_H_IDX]] : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O0_DYN_W_DIM:%.+]] = tensor.dim [[FUNC]]#0, [[DYN_W_IDX]] : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O_SLICE_0:%.+]] = tensor.insert_slice [[FUNC]]#0 into [[LOOP_CARRIED_0]][[[IND_VAR]], 0, 0, 0] [1, 48, [[O0_DYN_H_DIM]], [[O0_DYN_W_DIM]]] [1, 1, 1, 1] : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]> into tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O1_DYN_H_DIM:%.+]] = tensor.dim [[FUNC]]#1, [[DYN_H_IDX]] : tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O1_DYN_W_DIM:%.+]] = tensor.dim [[FUNC]]#1, [[DYN_W_IDX]] : tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O_SLICE_1:%.+]] = tensor.insert_slice [[FUNC]]#1 into [[LOOP_CARRIED_1]][[[IND_VAR]], 0, 0, 0] [1, 48, [[O1_DYN_H_DIM]], [[O1_DYN_W_DIM]]] [1, 1, 1, 1] : tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]> into tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       scf.yield [[O_SLICE_0]], [[O_SLICE_1]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:   return [[RET]]#0, [[RET]]#1 : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
}

// -----

func.func private @MultipleInputMultipleOutputMultiDynamicDimDeBatched_Batch1(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 62, 62]> : tensor<4xsi64>}>, %arg1: tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>) -> (tensor<1x48x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>, tensor<1x48x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 60, 60]> : tensor<4xsi64>}>) {
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

    // CHECK: func.func @MultipleInputMultipleOutputMultiDynamicDimDeBatched([[ARG0:%.+]]: tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>, [[ARG1:%.+]]: tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>) -> (tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>) {
    // CHECK:   [[DYN_W_IDX:%.+]] = arith.constant 3 : index
    // CHECK:   [[DYN_H_IDX:%.+]] = arith.constant 2 : index
    // CHECK:   [[STEP:%.+]] = arith.constant 1 : index
    // CHECK:   [[BEGIN:%.+]] = arith.constant 0 : index
    // CHECK:   [[END:%.+]] = tensor.dim [[ARG0]], [[ANY_MATCH:%.*]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_MIN_H:%.+]] = tensor.dim [[ARG0]], [[DYN_H_IDX]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[DYN_MIN_W:%.+]] = tensor.dim [[ARG0]], [[DYN_W_IDX]] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[OUTPUT_0:%.+]] = tensor.empty([[END]], [[DYN_MIN_H]], [[DYN_MIN_W]]) : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[OUTPUT_1:%.+]] = tensor.empty([[END]], [[DYN_MIN_H]], [[DYN_MIN_W]]) : tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:   [[RET:%.+]]:2 = scf.for [[IND_VAR:%.+]] = [[BEGIN]] to %dim step [[STEP]] iter_args([[LOOP_CARRIED_0:%.+]] = [[OUTPUT_0]], [[LOOP_CARRIED_1:%.+]] = [[OUTPUT_1]]) -> (tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>) {
    // CHECK:       [[I_SLICE_0:%.+]] = tensor.extract_slice [[ARG0]][[[IND_VAR]], 0, 0, 0] [1, 3, [[DYN_MIN_H]], [[DYN_MIN_W]]] [1, 1, 1, 1] : tensor<?x3x?x?xf32, [[ANY_MATCH:{.+}]]> to tensor<1x3x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[DYN_MIN_H_1:%.+]] = tensor.dim [[ARG1]], [[DYN_H_IDX]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[DYN_MIN_W_1:%.+]] = tensor.dim [[ARG1]], [[DYN_W_IDX]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[I_SLICE_1:%.+]] = tensor.extract_slice [[ARG1]][[[IND_VAR]], 0, 0, 0] [1, 48, [[DYN_MIN_H_1]], [[DYN_MIN_W_1]]] [1, 1, 1, 1] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]> to tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[FUNC:%.+]]:2 = func.call @MultipleInputMultipleOutputMultiDynamicDimDeBatched_Batch1([[I_SLICE_0]], [[I_SLICE_1]]) : (tensor<1x3x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>) -> (tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]>)
    // CHECK:       [[O0_DYN_H_DIM:%.+]] = tensor.dim [[FUNC]]#0, [[DYN_H_IDX]] : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O0_DYN_W_DIM:%.+]] = tensor.dim [[FUNC]]#0, [[DYN_W_IDX]] : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O_SLICE_0:%.+]] = tensor.insert_slice [[FUNC]]#0 into [[LOOP_CARRIED_0]][[[IND_VAR]], 0, 0, 0] [1, 48, [[O0_DYN_H_DIM]], [[O0_DYN_W_DIM]]] [1, 1, 1, 1] : tensor<1x48x?x?xf32, [[ANY_MATCH:{.+}]]> into tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O1_DYN_H_DIM:%.+]] = tensor.dim [[FUNC]]#1, [[DYN_H_IDX]] : tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O1_DYN_W_DIM:%.+]] = tensor.dim [[FUNC]]#1, [[DYN_W_IDX]] : tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       [[O_SLICE_1:%.+]] = tensor.insert_slice [[FUNC]]#1 into [[LOOP_CARRIED_1]][[[IND_VAR]], 0, 0, 0] [1, 48, [[O1_DYN_H_DIM]], [[O1_DYN_W_DIM]]] [1, 1, 1, 1] : tensor<1x48x?x?xf16, [[ANY_MATCH:{.+}]]> into tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:       scf.yield [[O_SLICE_0]], [[O_SLICE_1]] : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
    // CHECK:   return [[RET]]#0, [[RET]]#1 : tensor<?x48x?x?xf32, [[ANY_MATCH:{.+}]]>, tensor<?x48x?x?xf16, [[ANY_MATCH:{.+}]]>
}

