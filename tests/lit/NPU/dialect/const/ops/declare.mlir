//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @EraseTiledInfo
#C = affine_map<(d0) -> (d0)>

func.func @EraseTiledInfo() -> memref<8xf32> {
    %0 = const.Declare memref<8xf32, {order = #C, strides = [1]}> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    %1 = IERT.SubView %0 [0] [8] :
        memref<8xf32, {order = #C, strides = [1]}> to
        memref<8xf32>
    return %1 : memref<8xf32>
    // CHECK: [[CST:%.+]] = const.Declare memref<8xf32> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    // CHECK: return %cst : memref<8xf32>
}

// -----

// CHECK-LABEL: @EraseTiledInfoCopy
#C = affine_map<(d0) -> (d0)>

func.func @EraseTiledInfoCopy(%arg0: memref<8xf32, {order = #C}>) -> memref<8xf32, {order = #C}> {
    %0 = const.Declare memref<8xf32, {order = #C, strides = [1]}> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    %1 = IERT.Copy
        inputs(%0 : memref<8xf32, {order = #C, strides = [1]}>)
        outputs(%arg0: memref<8xf32, {order = #C}>)
        -> memref<8xf32, {order = #C}>
    return %1 : memref<8xf32, {order = #C}>
    // CHECK: [[CST:%.+]] = const.Declare memref<8xf32> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    // CHECK: [[VAR1:%.+]] = IERT.Copy inputs([[CST]] : memref<8xf32>) outputs(%arg0 : memref<8xf32, {order = #C}>) -> memref<8xf32, {order = #C}>
    // CHECK: return [[VAR1]] : memref<8xf32, {order = #C}>
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @ParseAndPrintAffineReshape
func.func @ParseAndPrintAffineReshape() -> tensor<6x2x2xf32, {order=#HWC}> {
    %cst = const.Declare tensor<6x2x2xf32, {order = #HWC}> = dense<1.000000e+00> : tensor<2x3x4xf32, {order = #map}>, [#const.AffineReshape<[[0], [0], [1, 2]], [6, 2, 2]>]
    return %cst : tensor<6x2x2xf32, {order = #HWC}>
    // CHECK: [[CST:%.+]] = const.Declare tensor<6x2x2xf32, {order = #HWC}> = dense<1.000000e+00> : tensor<2x3x4xf32, {order = #map}>, [#const.AffineReshape<{{\[\[}}0], [0], [1, 2]], [6, 2, 2]>]
    // CHECK: return [[CST]] : tensor<6x2x2xf32, {order = #HWC}>
}
