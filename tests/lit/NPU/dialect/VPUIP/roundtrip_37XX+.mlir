//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" %s | vpux-opt --init-compiler="platform=%platform% allow-custom-values=true" | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @ConstantTensor
func.func @ConstantTensor(%arg0: memref<1x1x28x28xf32>) -> memref<1x1x28x28xf32> {
    // CHECK: const.Declare memref<1x1x28x28xf32> = dense<0.000000e+00> : tensor<1x1x28x28xf32>
    %0 = const.Declare memref<1x1x28x28xf32> = dense<0.000000e+00> : tensor<1x1x28x28xf32>
    // CHECK: VPUIP.NNDMA <{port = 0 : i64}> inputs({{.+}} : memref<1x1x28x28xf32>) outputs({{.+}} : memref<1x1x28x28xf32>) -> memref<1x1x28x28xf32>
    %1 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : memref<1x1x28x28xf32>) outputs(%arg0 : memref<1x1x28x28xf32>) -> memref<1x1x28x28xf32>
    return %1 : memref<1x1x28x28xf32>
}
