//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --populate-dynamic-dimensions-generic %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: Softmax
func.func @Softmax(%arg0: tensor<?x?x64xf16, {bounds = [32, 32, 64], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>) -> tensor<?x?x64xf16, {bounds = [32, 32, 64], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> {
    // CHECK:   [[IN:%.+]]: tensor<?x?x64xf16

    %0 = IE.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<?x?x64xf16, {bounds = [32, 32, 64], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> -> tensor<?x?x64xf16, {bounds = [32, 32, 64], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[IN]]) {axisInd = 2 : i64} :
    // CHECK-SAME:  tensor<?x?x64xf16, {bounds = [32, 32, 64], order = #CHW}>
    // CHECK-SAME:  -> tensor<?x?x64xf16, {bounds = [32, 32, 64], order = #CHW}>

    // CHECK-DAG:   [[DYN_DIM_IDX_0:%.+]] = arith.constant 0 : index
    // CHECK-DAG:   [[DYN_DIM_VALUE_0:%.+]] = tensor.dim [[SOFTMAX]], [[DYN_DIM_IDX_0]]
    // CHECK-DAG:   [[DYN_DIM_0_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE_0]] : index to i64
    // CHECK-DAG:   [[DYN_DIM_0_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_0_I64]] : tensor<1xi64>
    // CHECK-DAG:   [[DYN_DIM_0:%.+]] = tensor.bitcast [[DYN_DIM_0_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG:   [[DYN_DIM_IDX_1:%.+]] = arith.constant 1 : index
    // CHECK-DAG:   [[DYN_DIM_VALUE_1:%.+]] = tensor.dim [[SOFTMAX]], [[DYN_DIM_IDX_1]]
    // CHECK-DAG:   [[DYN_DIM_1_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE_1]] : index to i64
    // CHECK-DAG:   [[DYN_DIM_1_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_1_I64]] : tensor<1xi64>
    // CHECK-DAG:   [[DYN_DIM_1:%.+]] = tensor.bitcast [[DYN_DIM_1_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK:       [[STATIC_DIM_2:%.+]] = const.Declare tensor<1xsi64> = dense<64> : tensor<1xsi64>

    // CHECK:       [[NEW_SHAPE:%.+]] = IE.Concat([[DYN_DIM_0]], [[DYN_DIM_1]], [[STATIC_DIM_2]])
    // CHECK:       [[DYN_RESHAPE:%.+]] = IE.DynamicReshape([[SOFTMAX]], [[NEW_SHAPE]])

    return %0 : tensor<?x?x64xf16, {bounds = [32, 32, 64], order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    // CHECK:       return [[DYN_RESHAPE]]
}
