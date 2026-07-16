//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @EraseTiledInfo
#C = affine_map<(d0) -> (d0)>

func.func @EraseTiledInfo() -> memref<8xf32> {
    %0 = const.Declare memref<8xf32, {order = #C, strides = [1]}> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    %1 = VPUIP.SubView %0 [0] [8] :
        memref<8xf32, {order = #C, strides = [1]}> to
        memref<8xf32>
    return %1 : memref<8xf32>
    // CHECK: [[CST:%.+]] = const.Declare memref<8xf32> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    // CHECK: return [[CST]] : memref<8xf32>
}

// -----

// CHECK-LABEL: @EraseTiledInfoCopy
// CHECK-SAME: ([[ARG_0:%.+]]: memref<8xf32>) -> memref<8xf32>
#C = affine_map<(d0) -> (d0)>

func.func @EraseTiledInfoCopy(%arg0: memref<8xf32>) -> memref<8xf32> {
    %0 = const.Declare memref<8xf32, {order = #C, strides = [1]}> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    %1 = VPUIP.Copy
        inputs(%0 : memref<8xf32, {order = #C, strides = [1]}>)
        outputs(%arg0: memref<8xf32>)
        -> memref<8xf32>
    return %1 : memref<8xf32>
    // CHECK: [[CST:%.+]] = const.Declare memref<8xf32> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#C>]
    // CHECK: [[VAR1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<8xf32>) outputs([[ARG_0]] : memref<8xf32>) -> memref<8xf32>
    // CHECK: return [[VAR1]] : memref<8xf32>
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

// -----

// CHECK-LABEL: @ParseAndPrintRescaleAttr
func.func @ParseAndPrintRescaleAttr() -> (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) {
    %cst_two_transformations = const.Declare tensor<1x1x2x2xf32> = dense<[[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]]]> : tensor<1x1x2x2xf32>,
    [#const.Rescale<Content<dense<[[[[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]]]> : tensor<1x1x2x2xf32>>>,
     #const.Add<1.000000e+00 : f64>]


    %cst_rescale_inner_transformation = const.Declare tensor<1x1x2x2xf32> = dense<[[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]]]> : tensor<1x1x2x2xf32>,
    [#const.Rescale<Content<dense<[[[[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]]]> : tensor<1x1x2x2xf32>, [#const.Add<1.000000e+00 : f64>]>>]


    return %cst_two_transformations, %cst_rescale_inner_transformation : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x2x2xf32> =
    // CHECK-SAME{LITERAL}: dense<[[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]]]> : tensor<1x1x2x2xf32>,
    // CHECK-SAME{LITERAL}: [#const.Rescale<Content<dense<[[[[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]]]> : tensor<1x1x2x2xf32>>>, #const.Add<1.000000e+00 : f64>]

    // CHECK:  [[CST_0:%.+]] = const.Declare tensor<1x1x2x2xf32> =
    // CHECK-SAME{LITERAL}: dense<[[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]]]> : tensor<1x1x2x2xf32>,
    // CHECK-SAME{LITERAL}:  [#const.Rescale<Content<dense<[[[[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]]]> : tensor<1x1x2x2xf32>, [#const.Add<1.000000e+00 : f64>]>>]

    // CHECK-NEXT: return [[CST]], [[CST_0]] : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>
}

// -----

// CHECK-LABEL: @ParseAndPrintSplatRescaleAttr
func.func @ParseAndPrintSplatRescaleAttr() -> tensor<1x1x32x1xf16> {
    %cst = const.Declare tensor<1x1x32x1xf16> = dense<5.000000e+00> : tensor<1x1x32x1xf16>, [#const.Rescale<-1.000000e+00 : f64>]
    return %cst : tensor<1x1x32x1xf16>
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x32x1xf16> = dense<5.000000e+00> : tensor<1x1x32x1xf16>, [#const.Rescale<-1.000000e+00 : f64>]

    // CHECK-NEXT: return [[CST]] : tensor<1x1x32x1xf16>
}
