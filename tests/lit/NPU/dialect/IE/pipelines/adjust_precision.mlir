//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --adjust-precision %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

module @Test {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<100xf32>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x2x2x2xf32>
        DataInfo "prob2" : tensor<10xf32>
    }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<100xf32>) -> (tensor<1x2x2x2xf32>, tensor<10xf32>) {
func.func @main(%arg0: tensor<100xf32>) -> (tensor<1x2x2x2xf64>, tensor<10xf32>) {
    %0 = const.Declare tensor<1x2x2x2xf64> = dense<1.0> : tensor<1x2x2x2xf64>
    %1 = const.Declare tensor<10xsi64> = dense<1> : tensor<10xsi64>

    %2 = IE.Gather(%arg0, %1) {axis_value = 0, batch_dims = 0} : tensor<100xf32>,tensor<10xsi64> -> tensor<10xf32>
    return %0, %2 : tensor<1x2x2x2xf64>, tensor<10xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x2x2x2xf32> = dense<1.000000e+00> : tensor<1x2x2x2xf64>, [#const.CastElemType<f32>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<10xsi32> = dense<1> : tensor<10xsi64>, [#const.CastElemType<si32>]

    // CHECK: [[VAR0:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<100xf32> -> tensor<100xf16>
    // CHECK: [[VAR1:%.+]] = IE.Gather([[VAR0]], [[CST_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<100xf16>, tensor<10xsi32> -> tensor<10xf16>
    // CHECK: [[VAR2:%.+]] = IE.Convert([[VAR1]]) {dstElemType = f32} : tensor<10xf16> -> tensor<10xf32>

    // CHECK: return [[CST]], [[VAR2]] : tensor<1x2x2x2xf32>, tensor<10xf32>
}

}
