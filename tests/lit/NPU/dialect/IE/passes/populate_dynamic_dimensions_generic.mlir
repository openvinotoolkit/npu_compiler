//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --populate-dynamic-dimensions-generic %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: EltwiseLessEqual
func.func @EltwiseLessEqual(%arg0: tensor<1x1x?x200xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<1x1x?x200xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW }>) -> tensor<1x1x?x200xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK-DAG: [[IN:%.+]]: tensor<1x1x?x200xf32
  
    %0 = IE.LessEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x?x200xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x200xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x200xi8, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-DAG: [[LESSEQUAL:%.+]] = IE.LessEqual(%{{.+}}, %{{.+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x?x200xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x200xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x200xi8, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}> 
    
    // CHECK-DAG: [[STATIC_DIM_0:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[DYN_DIM_IDX_1:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[DYN_DIM_VALUE_1:%.+]] = tensor.dim [[LESSEQUAL]], [[DYN_DIM_IDX_1]]
    // CHECK-DAG: [[DYN_DIM_1_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE_1]] : index to i64
    // CHECK-DAG: [[DYN_DIM_1_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_1_I64]] : tensor<1xi64>
    // CHECK-DAG: [[DYN_DIM_1:%.+]] = tensor.bitcast [[DYN_DIM_1_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>
    
    // CHECK-DAG: [[NEW_SHAPE:%.+]] = IE.Concat([[STATIC_DIM_0]], [[STATIC_DIM_1]], [[DYN_DIM_1]], %{{.+}}) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>
    // CHECK-DAG: [[DYN_RESHAPE:%.+]] = IE.DynamicReshape([[LESSEQUAL]], [[NEW_SHAPE]]) {only_set_shape, output_bounds = [1, 1, 10, 200], output_shape = [1, 1, -9223372036854775808, 200]}
    
    %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x1x?x200xi8, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x200xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-DAG: [[CONVERT:%.+]] = IE.Convert([[DYN_RESHAPE]]) {dstElemType = f32}
    
    return %1 : tensor<1x1x?x200xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 200]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-DAG: return [[CONVERT]]
}

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: Softmax
func.func @Softmax(%arg0: tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>) -> tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> {
    // CHECK:   [[IN:%.+]]: tensor<?x?x64xf16

    %0 = IE.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> -> tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[IN]]) {axisInd = 2 : i64} :
    // CHECK-SAME:  tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = #CHW}>
    // CHECK-SAME:  -> tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = #CHW}>

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

    return %0 : tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    // CHECK:       return [[DYN_RESHAPE]]
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: EltwiseLess
func.func @EltwiseLess(%arg0: tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}>) -> tensor<1x?xi8, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}> {
    // CHECK:   [[IN:%.+]]: tensor<1x?xf16

    %cst = const.Declare tensor<1x1xf16> = dense<0.500000e+00> : tensor<1x1xf16> isSplat

    %0 = IE.Less(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}>, tensor<1x1xf16> -> tensor<1x?xi8, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}>
    // CHECK: [[LESS:%.+]] = IE.Less([[IN]], %{{.+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:  tensor<1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}>, tensor<1x1xf16>
    // CHECK-SAME:  -> tensor<1x?xi8, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}>

    // CHECK-DAG:   [[STATIC_DIM_0:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK-DAG:   [[DYN_DIM_IDX_1:%.+]] = arith.constant 1 : index
    // CHECK-DAG:   [[DYN_DIM_VALUE_1:%.+]] = tensor.dim [[LESS]], [[DYN_DIM_IDX_1]]
    // CHECK-DAG:   [[DYN_DIM_1_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE_1]] : index to i64
    // CHECK-DAG:   [[DYN_DIM_1_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_1_I64]] : tensor<1xi64>
    // CHECK-DAG:   [[DYN_DIM_1:%.+]] = tensor.bitcast [[DYN_DIM_1_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK:       [[NEW_SHAPE:%.+]] = IE.Concat([[STATIC_DIM_0]], [[DYN_DIM_1]])
    // CHECK:       [[DYN_RESHAPE:%.+]] = IE.DynamicReshape([[LESS]], [[NEW_SHAPE]])

    return %0 : tensor<1x?xi8, {bounds = #const.OpaqueI64Elements<[1, 32]> : tensor<2xsi64>, order = #NC}>
    // CHECK:       return [[DYN_RESHAPE]]
}

// -----

// CHECK-LABEL: EltwiseMinimum
func.func @EltwiseMinimum(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> {
    %cst = const.Declare tensor<1x3x1x1xf32> =  dense<0.500000e+00> : tensor<1x3x1x1xf32> isSplat
    %0 = IE.Minimum(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>, tensor<1x3x1x1xf32> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    return %0 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>

    // CHECK-DAG: [[DYN_DIM_IDX_2:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[DYN_DIM_VALUE_2:%.+]] = tensor.dim %0, [[DYN_DIM_IDX_2]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-DAG: [[DYN_DIM_IDX_3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DYN_DIM_VALUE_3:%.+]] = tensor.dim %0, [[DYN_DIM_IDX_3]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-DAG: [[STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[STATIC_DIM_2:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>

    // CHECK-DAG: [[DYN_DIM_2_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE_2]] : index to i64
    // CHECK-DAG: [[DYN_DIM_2_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_2_I64]] : tensor<1xi64>
    // CHECK-DAG: [[DYN_DIM_2:%.+]] = tensor.bitcast [[DYN_DIM_2_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG: [[DYN_DIM_3_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE_3]] : index to i64
    // CHECK-DAG: [[DYN_DIM_3_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_3_I64]] : tensor<1xi64>
    // CHECK-DAG: [[DYN_DIM_3:%.+]] = tensor.bitcast [[DYN_DIM_3_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG: [[NEW_SHAPE:%.+]] = IE.Concat([[STATIC_DIM_1]], [[STATIC_DIM_2]], [[DYN_DIM_2]], [[DYN_DIM_3]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK: [[DYN_RESHAPE:%.+]] = IE.DynamicReshape(%0, [[NEW_SHAPE]]) {only_set_shape, output_bounds = [1, 3, 128, 128], output_shape = [1, 3, -9223372036854775808, -9223372036854775808]} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: return [[DYN_RESHAPE]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

// CHECK-LABEL: EltwiseMaximum
func.func @EltwiseMaximum(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> {
    %cst = const.Declare tensor<1x3x1x1xf32> =  dense<0.500000e+00> : tensor<1x3x1x1xf32> isSplat
    %0 = IE.Maximum(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>, tensor<1x3x1x1xf32> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    return %0 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>

    // CHECK-DAG: [[DYN_DIM_IDX_2:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[DYN_DIM_VALUE_2:%.+]] = tensor.dim %0, [[DYN_DIM_IDX_2]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-DAG: [[DYN_DIM_IDX_3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DYN_DIM_VALUE_3:%.+]] = tensor.dim %0, [[DYN_DIM_IDX_3]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-DAG: [[STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[STATIC_DIM_2:%.+]] = const.Declare tensor<1xsi64> = dense<3> : tensor<1xsi64>

    // CHECK-DAG: [[DYN_DIM_2_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE_2]] : index to i64
    // CHECK-DAG: [[DYN_DIM_2_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_2_I64]] : tensor<1xi64>
    // CHECK-DAG: [[DYN_DIM_2:%.+]] = tensor.bitcast [[DYN_DIM_2_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG: [[DYN_DIM_3_I64:%.+]] = arith.index_cast [[DYN_DIM_VALUE_3]] : index to i64
    // CHECK-DAG: [[DYN_DIM_3_TO_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_3_I64]] : tensor<1xi64>
    // CHECK-DAG: [[DYN_DIM_3:%.+]] = tensor.bitcast [[DYN_DIM_3_TO_TENSOR]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG: [[NEW_SHAPE:%.+]] = IE.Concat([[STATIC_DIM_1]], [[STATIC_DIM_2]], [[DYN_DIM_2]], [[DYN_DIM_3]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK: [[DYN_RESHAPE:%.+]] = IE.DynamicReshape(%0, [[NEW_SHAPE]]) {only_set_shape, output_bounds = [1, 3, 128, 128], output_shape = [1, 3, -9223372036854775808, -9223372036854775808]} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: return [[DYN_RESHAPE]] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 128, 128]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: EltwiseSubtract
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @EltwiseSubtract(%arg0: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %0 = IE.Subtract(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK: [[SUB:%.+]] = IE.Subtract([[CST]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME: tensor<1x1x1x1xf32>, tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME: -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DIM:%.+]] = tensor.dim [[SUB]], [[C3]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK-DAG: [[DIM_I64:%.+]] = arith.index_cast [[DIM]] : index to i64
    // CHECK-DAG: [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[DIM_I64]] : tensor<1xi64>

    // CHECK-DAG: [[BITCAST:%.+]] = tensor.bitcast [[FROM_ELEMENTS]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG: [[NEW_SHAPE:%.+]] = IE.Concat([[CST_0]], [[CST_1]], [[CST_2]], [[BITCAST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK: [[RESHAPE:%.+]] = IE.DynamicReshape([[SUB]], [[NEW_SHAPE]]) {only_set_shape, output_bounds = [1, 1, 1, 64], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: return [[RESHAPE]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: EltwisePower
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @EltwisePower(%arg0: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %0 = IE.Power(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK: [[POW:%.+]] = IE.Power([[CST]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME: tensor<1x1x1x1xf32>, tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME: -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DIM:%.+]] = tensor.dim [[POW]], [[C3]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK-DAG: [[DIM_I64:%.+]] = arith.index_cast [[DIM]] : index to i64
    // CHECK-DAG: [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[DIM_I64]] : tensor<1xi64>

    // CHECK-DAG: [[BITCAST:%.+]] = tensor.bitcast [[FROM_ELEMENTS]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG: [[NEW_SHAPE:%.+]] = IE.Concat([[CST_0]], [[CST_1]], [[CST_2]], [[BITCAST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK: [[RESHAPE:%.+]] = IE.DynamicReshape([[POW]], [[NEW_SHAPE]]) {only_set_shape, output_bounds = [1, 1, 1, 64], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: return [[RESHAPE]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: EltwiseMod
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @EltwiseMod(%arg0: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %0 = IE.Mod(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1xf32> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK: [[MOD:%.+]] = IE.Mod([[ARG0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1xf32>
    // CHECK-SAME: -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DIM:%.+]] = tensor.dim [[MOD]], [[C3]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK-DAG: [[DIM_I64:%.+]] = arith.index_cast [[DIM]] : index to i64
    // CHECK-DAG: [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[DIM_I64]] : tensor<1xi64>

    // CHECK-DAG: [[BITCAST:%.+]] = tensor.bitcast [[FROM_ELEMENTS]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG: [[NEW_SHAPE:%.+]] = IE.Concat([[CST_0]], [[CST_1]], [[CST_2]], [[BITCAST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK: [[RESHAPE:%.+]] = IE.DynamicReshape([[MOD]], [[NEW_SHAPE]]) {only_set_shape, output_bounds = [1, 1, 1, 64], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: return [[RESHAPE]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: EltwiseFloorMod
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @EltwiseFloorMod(%arg0: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %0 = IE.FloorMod(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1xf32> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK: [[FLOORMOD:%.+]] = IE.FloorMod([[ARG0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1xf32>
    // CHECK-SAME: -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DIM:%.+]] = tensor.dim [[FLOORMOD]], [[C3]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK-DAG: [[DIM_I64:%.+]] = arith.index_cast [[DIM]] : index to i64
    // CHECK-DAG: [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[DIM_I64]] : tensor<1xi64>

    // CHECK-DAG: [[BITCAST:%.+]] = tensor.bitcast [[FROM_ELEMENTS]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG: [[NEW_SHAPE:%.+]] = IE.Concat([[CST_0]], [[CST_1]], [[CST_2]], [[BITCAST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK: [[RESHAPE:%.+]] = IE.DynamicReshape([[FLOORMOD]], [[NEW_SHAPE]]) {only_set_shape, output_bounds = [1, 1, 1, 64], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: return [[RESHAPE]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: EltwiseDivide
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @EltwiseDivide(%arg0: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %0 = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1xf32> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[ARG0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x1xf32>
    // CHECK-SAME: -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DIM:%.+]] = tensor.dim [[DIVIDE]], [[C3]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    // CHECK-DAG: [[DIM_I64:%.+]] = arith.index_cast [[DIM]] : index to i64
    // CHECK-DAG: [[FROM_ELEMENTS:%.+]] = tensor.from_elements [[DIM_I64]] : tensor<1xi64>

    // CHECK-DAG: [[BITCAST:%.+]] = tensor.bitcast [[FROM_ELEMENTS]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK-DAG: [[NEW_SHAPE:%.+]] = IE.Concat([[CST_0]], [[CST_1]], [[CST_2]], [[BITCAST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<4xsi64>

    // CHECK: [[RESHAPE:%.+]] = IE.DynamicReshape([[DIVIDE]], [[NEW_SHAPE]]) {only_set_shape, output_bounds = [1, 1, 1, 64], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: return [[RESHAPE]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
}
