//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-layout --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3, d0)>

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x77x4096x1xf32>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x77x4096x1xf32>
    }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x77x4096x1xf32>) -> tensor<1x77x4096x1xf32> {
func.func @main(%arg0: tensor<1x77x4096x1xf32>) -> tensor<1x77x4096x1xf32> {
    %0 = IE.Transpose(%arg0) {order_value = #HCNW} : tensor<1x77x4096x1xf32> -> tensor<4096x77x1x1xf32>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4096, 77, 1]} : tensor<4096x77x1x1xf32> -> tensor<1x4096x77x1xf32>
    %2 = IE.SoftMax(%1) {axisInd = 2 : i64} : tensor<1x4096x77x1xf32> -> tensor<1x4096x77x1xf32>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [4096, 77, 1, 1]} : tensor<1x4096x77x1xf32> -> tensor<4096x77x1x1xf32>
    %4 = IE.Transpose(%3) {order_value = #HCNW} : tensor<4096x77x1x1xf32> -> tensor<1x77x4096x1xf32>
    return %4: tensor<1x77x4096x1xf32>

    // CHECK:        [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x77x4096x1xf32> -> tensor<1x77x4096x1xf32>
    // CHECK:        return [[SOFTMAX]] : tensor<1x77x4096x1xf32>
}

}
