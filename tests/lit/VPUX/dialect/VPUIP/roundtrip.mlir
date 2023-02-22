//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | vpux-opt | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConstantTensor
func @ConstantTensor(%arg0: memref<1x1x28x28xf32>) -> memref<1x1x28x28xf32> {
    // CHECK: const.Declare memref<1x1x28x28xf16> = dense<0.000000e+00> : tensor<1x1x28x28xf16>
    %0 = const.Declare memref<1x1x28x28xf16> = dense<0.000000e+00> : tensor<1x1x28x28xf16>
    // CHECK: VPUIP.ConvertUPA {isTrailingSWLayer} inputs(%{{.*}} : memref<1x1x28x28xf16>) outputs(%{{.*}} : memref<1x1x28x28xf32>) -> memref<1x1x28x28xf32>
    %1 = VPUIP.ConvertUPA {isTrailingSWLayer} inputs(%0 : memref<1x1x28x28xf16>) outputs(%arg0 : memref<1x1x28x28xf32>) -> memref<1x1x28x28xf32>
    return %1 : memref<1x1x28x28xf32>
}
