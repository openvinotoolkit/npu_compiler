//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --use-user-precision %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @U8toFP32
module @U8toFP32 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x1000xui8>
        DataInfo "data" : tensor<1x1000xui8>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1000xf32>
        DataInfo "prob" : tensor<1x1000xf32>
    }

// CHECK: func.func @main(%[[ARG0:arg.*]]: tensor<1x1000xui8>) -> tensor<1x1000xf32>
func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK:       %[[VAR0:.*]] = IE.Convert(%[[ARG0]])
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:      tensor<1x1000xui8> -> tensor<1x1000xf16>

    // CHECK:       %[[VAR1:.*]] = IE.SoftMax(%[[VAR0]])
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    // CHECK:       %[[VAR2:.*]] = IE.Convert(%[[VAR1]])
    // CHECK-SAME:      dstElemType = f32
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf32>

    // CHECK:       return %[[VAR2]] : tensor<1x1000xf32>
}

}

// -----

// CHECK-LABEL: @SameTypes
module @SameTypes {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x1000xf16>
        DataInfo "data" : tensor<1x1000xf16>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1000xf16>
        DataInfo "prob" : tensor<1x1000xf16>
    }

// CHECK: func.func @main(%[[ARG0:arg.*]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK-NOT:   IE.Convert

    // CHECK:       %[[VAR0:.*]] = IE.SoftMax(%[[ARG0]])
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    // CHECK-NOT:   IE.Convert

    // CHECK:       return %[[VAR0]] : tensor<1x1000xf16>
}

}
