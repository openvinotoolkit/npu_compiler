//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  func.func @testDynamicReshapeWithParameters
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>)
func.func @testDynamicReshapeWithParameters(%arg0: tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}> {
     %cst_0 = const.Declare tensor<1xsi64> = dense<[1]> : tensor<1xsi64>
     %cst_1 = const.Declare tensor<1xsi64> = dense<[64]> : tensor<1xsi64>

     // Extract the shape of the input tensor
     %shape1 = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>

     // Slice to get the dynamic dimension
     %dyn_dim = IE.Slice %shape1 [3] [1] : tensor<4xsi64> to tensor<1xsi64>

     // Create a parameter shape using the dynamic dimension
     %param_shape1 = IE.Concat(%dyn_dim, %cst_0, %cst_1) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
     %param_shape2 = IE.Concat(%cst_0, %dyn_dim, %cst_1) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>

     // First DynamicReshape operation with parameter shape
     %0 = IE.DynamicReshape(%arg0, %param_shape1) {output_bounds = [10, 1, 64], output_shape = [-9223372036854775808, 1, 64]} : 
          tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>, tensor<3xsi64> -> tensor<?x1x64xf32, {bounds = #const.OpaqueI64Elements<[10, 1, 64]> : tensor<3xsi64>, order = #CHW}>

     // Second DynamicReshape operation with parameter shape
     %1 = IE.DynamicReshape(%0, %param_shape2) {output_bounds = [1, 10, 64], output_shape = [1, -9223372036854775808, 64]} : 
          tensor<?x1x64xf32, {bounds = #const.OpaqueI64Elements<[10, 1, 64]> : tensor<3xsi64>, order = #CHW}>, tensor<3xsi64> -> tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>

     return %1 : tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>

     // CHECK: [[CST:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
     // CHECK: [[CST_0:%.+]] = const.Declare tensor<1xsi64> = dense<64> : tensor<1xsi64>
     // CHECK: [[SHAPE_OF:%.+]] = IE.ShapeOf([[ARG0]]) {dstElemType = si64} : tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
     // CHECK: [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1] : tensor<4xsi64> to tensor<1xsi64>
     // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[CST]], [[CST_0]])
     // CHECK:    tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
     // CHECK: [[SLICE:%.+]] = IE.Slice [[CONCAT]] [0] [1] : tensor<3xsi64> to tensor<1xsi64>
     // CHECK: [[CONCAT_0:%.+]] = IE.Concat([[CST]], [[SLICE]], [[CST_0]])
     // CHECK:    tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
     // CHECK: [[DYN_RESHAPE:%.+]] = IE.DynamicReshape([[ARG0]], [[CONCAT_0]]) {output_bounds = [1, 10, 64], output_shape = [1, -9223372036854775808, 64]} :
     // CHECK:    tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>,
     // CHECK:    tensor<3xsi64> -> tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>
     // CHECK: return [[DYN_RESHAPE]] : tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  func.func @testDynamicReshapeWithConstants
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>)
func.func @testDynamicReshapeWithConstants(%arg0: tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}> {
     %shape1 = const.Declare tensor<3xsi64> = dense<[-1, 1, 64]> : tensor<3xsi64>
     %shape2 = const.Declare tensor<3xsi64> = dense<[1, -1, 64]> : tensor<3xsi64>

     %0 = IE.DynamicReshape(%arg0, %shape1) {output_bounds = [10, 1, 64], output_shape = [-9223372036854775808, 1, 64]} : 
          tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>, tensor<3xsi64> -> tensor<?x1x64xf32, {bounds = #const.OpaqueI64Elements<[10, 1, 64]> : tensor<3xsi64>, order = #CHW}>

     %1 = IE.DynamicReshape(%0, %shape2) {output_bounds = [1, 10, 64], output_shape = [1, -9223372036854775808, 64]} : 
          tensor<?x1x64xf32, {bounds = #const.OpaqueI64Elements<[10, 1, 64]> : tensor<3xsi64>, order = #CHW}>, tensor<3xsi64> -> tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>

     return %1 : tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>

     // CHECK: [[CST:%.+]] = const.Declare tensor<3xsi64> = dense<[1, -1, 64]> : tensor<3xsi64>
     // CHECK: [[DYN_RESHAPE:%.+]] = IE.DynamicReshape([[ARG0]], [[CST]]) {output_bounds = [1, 10, 64],
     // CHECK:    output_shape = [1, -9223372036854775808, 64]} : tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> :
     // CHECK:    tensor<4xsi64>, order = #NCHW}>, tensor<3xsi64> -> tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> :
     // CHECK:    tensor<3xsi64>, order = #CHW}>
     // CHECK: return [[DYN_RESHAPE]] : tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  func.func @testDynamicReshapeWithParameterAndConstant
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>)
func.func @testDynamicReshapeWithParameterAndConstant(%arg0: tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}> {
     %cst_0 = const.Declare tensor<1xsi64> = dense<[1]> : tensor<1xsi64>
     %cst_1 = const.Declare tensor<1xsi64> = dense<[64]> : tensor<1xsi64>
     %shape2 = const.Declare tensor<3xsi64> = dense<[1, -1, 64]> : tensor<3xsi64>

     %0 = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
     %dyn_dim = IE.Slice %0 [3] [1] : tensor<4xsi64> to tensor<1xsi64>
     %2 = IE.Concat(%dyn_dim, %cst_1, %cst_0) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
     %3 = IE.DynamicReshape(%arg0, %2) {output_bounds = [10, 1, 64], output_shape = [-9223372036854775808, 1, 64]} : 
          tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>, tensor<3xsi64> -> tensor<?x1x64xf32, {bounds = #const.OpaqueI64Elements<[10, 1, 64]> : tensor<3xsi64>, order = #CHW}>

     %4 = IE.DynamicReshape(%3, %shape2) {output_bounds = [1, 10, 64], output_shape = [1, -9223372036854775808, 64]} : 
          tensor<?x1x64xf32, {bounds = #const.OpaqueI64Elements<[10, 1, 64]> : tensor<3xsi64>, order = #CHW}>, tensor<3xsi64> -> tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>


     return %4 : tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>

     // CHECK: [[CST:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
     // CHECK: [[CST_0:%.+]] = const.Declare tensor<1xsi64> = dense<64> : tensor<1xsi64>
     // CHECK: [[SHAPE_OF:%.+]] = IE.ShapeOf([[ARG0]]) {dstElemType = si64} : tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
     // CHECK: [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1] : tensor<4xsi64> to tensor<1xsi64>
     // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[CST_0]], [[CST]])
     // CHECK:    tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
     // CHECK: [[SLICE_0:%.+]] = IE.Slice [[CONCAT]] [0] [1] : tensor<3xsi64> to tensor<1xsi64>
     // CHECK: [[CONCAT_0:%.+]] = IE.Concat([[CST]], [[SLICE_0]], [[CST_0]])
     // CHECK:    tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
     // CHECK: [[DYN_RESHAPE:%.+]] = IE.DynamicReshape([[ARG0]], [[CONCAT_0]]) {output_bounds = [1, 10, 64], output_shape = [1, -9223372036854775808, 64]} :
     // CHECK:    tensor<4x1x16x?xf32, {bounds = #const.OpaqueI64Elements<[4, 1, 16, 10]> : tensor<4xsi64>, order = #NCHW}>, tensor<3xsi64> -> tensor<1x?x64xf32,
     // CHECK:    {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>
     // CHECK: return [[DYN_RESHAPE]] : tensor<1x?x64xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 64]> : tensor<3xsi64>, order = #CHW}>
}
