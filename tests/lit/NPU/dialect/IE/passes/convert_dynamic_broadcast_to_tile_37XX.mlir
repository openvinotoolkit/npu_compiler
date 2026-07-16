//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-broadcast-to-tile %s --canonicalize | FileCheck %s
// REQUIRES: platform-NPU3720

#C = affine_map<(d0) -> (d0)>
#CHW = affine_map<(d1, d2, d3) -> (d1, d2, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DynamicBroadcastShapeSubgraph {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x3x8x10xf32>
    DataInfo "input_1" : tensor<10xf32>
  } outputsInfo : {
    DataInfo "Broadcast_60" friendlyName = "Result_64" : tensor<1x3x10x16xf32>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}>, [[ARG1:%.+]]: tensor<?xf32, {bounds = #const.OpaqueI64Elements<[10]> : tensor<1xsi64>, order = #C}>) -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}> {
  func.func @main(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<?xf32, {bounds = #const.OpaqueI64Elements<[10]> : tensor<1xsi64>, order = #C}>) -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    %1 = IE.DynamicBroadcast(%arg1, %0) {mode = #IE.broadcast_type<NUMPY>, output_bounds = [1, 3, 10, 16], output_shape = [1, 3, -9223372036854775808, -9223372036854775808]} : tensor<?xf32, {bounds = #const.OpaqueI64Elements<[10]> : tensor<1xsi64>, order = #C}>, tensor<4xsi64> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}>
    return %1 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:               [[CONST:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, 1, -1]> : tensor<4xsi32>
    // CHECK:               [[SHAPEOF:%.+]] = IE.ShapeOf([[ARG0]]) {dstElemType = si64} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    // CHECK-NOT:           IE.Broadcast
    // CHECK-NOT:           IE.DynamicBroadcast
    // CHECK:               [[RESHAPE:%.+]] = IE.DynamicReshape([[ARG1]], [[CONST]]) {output_bounds = [1, 1, 1, 10], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<?xf32, {bounds = #const.OpaqueI64Elements<[10]> : tensor<1xsi64>, order = #C}>, tensor<4xsi32> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:               [[TILE:%.+]] = IE.DynamicTile([[RESHAPE]], [[SHAPEOF]]) {output_bounds = [1, 3, 10, 16], output_shape = [1, 3, -9223372036854775808, -9223372036854775808]} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:               return [[TILE]]
  }
}

// -----

#C = affine_map<(d0) -> (d0)>
#CHW = affine_map<(d1, d2, d3) -> (d1, d2, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DynamicBroadcastShapeSubgraph {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x3x8x10xf32>
    DataInfo "input_1" : tensor<10xf32>
  } outputsInfo : {
    DataInfo "Broadcast_60" friendlyName = "Result_64" : tensor<1x3x10x16xf32>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}>, [[ARG1:%.+]]: tensor<?xf32, {bounds = #const.OpaqueI64Elements<[10]> : tensor<1xsi64>, order = #C}>) -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}> {
  func.func @main(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<?xf32, {bounds = #const.OpaqueI64Elements<[10]> : tensor<1xsi64>, order = #C}>) -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    %1 = IE.DynamicBroadcast(%arg1, %0) {mode = #IE.broadcast_type<BIDIRECTIONAL>, output_bounds = [1, 3, 10, 16], output_shape = [1, 3, -9223372036854775808, -9223372036854775808]} : tensor<?xf32, {bounds = #const.OpaqueI64Elements<[10]> : tensor<1xsi64>, order = #C}>, tensor<4xsi64> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}>
    return %1 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:               [[CONST:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, 1, -1]> : tensor<4xsi32>
    // CHECK:               [[SHAPEOF:%.+]] = IE.ShapeOf([[ARG0]]) {dstElemType = si64} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    // CHECK-NOT:           IE.Broadcast
    // CHECK-NOT:           IE.DynamicBroadcast
    // CHECK:               [[RESHAPE:%.+]] = IE.DynamicReshape([[ARG1]], [[CONST]]) {output_bounds = [1, 1, 1, 10], output_shape = [1, 1, 1, -9223372036854775808]} : tensor<?xf32, {bounds = #const.OpaqueI64Elements<[10]> : tensor<1xsi64>, order = #C}>, tensor<4xsi32> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:               [[TILE:%.+]] = IE.DynamicTile([[RESHAPE]], [[SHAPEOF]]) {output_bounds = [1, 3, 10, 16], output_shape = [1, 3, -9223372036854775808, -9223372036854775808]} : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 10]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64> -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 10, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:               return [[TILE]]
  }
}

// -----

#C = affine_map<(d0) -> (d0)>
#CHW = affine_map<(d1, d2, d3) -> (d1, d2, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DynamicBroadcastShapeSubgraph {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<2x3x4xf32>
    DataInfo "input_1" : tensor<2x4xf32>
  } outputsInfo : {
    DataInfo "Broadcast_60" friendlyName = "Result_64" : tensor<2x3x4xf32>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<?x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 4, 5]> : tensor<3xsi64>}>, [[ARG1:%.+]]: tensor<?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 5]> : tensor<2xsi64>}>) -> tensor<?x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 4, 5]> : tensor<3xsi64>}> {
  func.func @main(%arg0: tensor<?x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 4, 5]> : tensor<3xsi64>}>, %arg1: tensor<?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 5]> : tensor<2xsi64>}>) -> tensor<?x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 4, 5]> : tensor<3xsi64>}> {
    %cst_0 = const.Declare tensor<2xsi64> = dense<[0, 2]> : tensor<2xsi64>
    %0 = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<?x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 4, 5]> : tensor<3xsi64>}> -> tensor<3xsi64>
    %1 = IE.DynamicBroadcast(%arg1, %0, %cst_0) {mode = #IE.broadcast_type<EXPLICIT>, output_bounds = [3, 4, 5], output_shape = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : tensor<?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 5]> : tensor<2xsi64>}>, tensor<3xsi64>, tensor<2xsi64> -> tensor<?x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 4, 5]> : tensor<3xsi64>}>
    return %1 : tensor<?x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 4, 5]> : tensor<3xsi64>}>

    // CHECK:               [[CONST:%.+]] = const.Declare tensor<3xsi32> = dense<[-1, 1, -1]> : tensor<3xsi32>
    // CHECK:               [[SHAPEOF:%.+]] = IE.ShapeOf([[ARG0]]) {dstElemType = si64} : tensor<?x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 4, 5]> : tensor<3xsi64>}> -> tensor<3xsi64>
    // CHECK-NOT:           IE.Broadcast
    // CHECK-NOT:           IE.DynamicBroadcast
    // CHECK:               [[RESHAPE:%.+]] = IE.DynamicReshape([[ARG1]], [[CONST]]) {output_bounds = [3, 1, 5], output_shape = [-9223372036854775808, 1, -9223372036854775808]} : tensor<?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 5]> : tensor<2xsi64>}>, tensor<3xsi32> -> tensor<?x1x?xf32, {bounds = #const.OpaqueI64Elements<[3, 1, 5]> : tensor<3xsi64>, order = #CHW}>
    // CHECK:               [[TILE:%.+]] = IE.DynamicTile([[RESHAPE]], [[SHAPEOF]]) {output_bounds = [3, 4, 5], output_shape = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : tensor<?x1x?xf32, {bounds = #const.OpaqueI64Elements<[3, 1, 5]> : tensor<3xsi64>, order = #CHW}>, tensor<3xsi64> -> tensor<?x?x?xf32, {bounds = #const.OpaqueI64Elements<[3, 4, 5]> : tensor<3xsi64>}>
    // CHECK:               return [[TILE]]
  }
}

// -----

#C = affine_map<(d0) -> (d0)>
#CHW = affine_map<(d1, d2, d3) -> (d1, d2, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DynamicBroadcastShapeSubgraph {
net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_60" : tensor<1x1x6xsi32>
    DataInfo "input_1" : tensor<1x1x1xf32>
  } outputsInfo : {
    DataInfo "Broadcast_63" friendlyName = "Result_67" : tensor<1x1x10xf32>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>, [[ARG1:%.+]]: tensor<1x1x1xf32>) -> tensor<1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}> {
  func.func @main(%arg0: tensor<1x1x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>, %arg1: tensor<1x1x1xf32>) -> tensor<1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}> {
    %0 = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<1x1x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}> -> tensor<3xsi64>
    %1 = IE.DynamicBroadcast(%arg1, %0) {mode = #IE.broadcast_type<NUMPY>, output_bounds = [1, 1, 10], output_shape = [1, 1, -9223372036854775808]} : tensor<1x1x1xf32>, tensor<3xsi64> -> tensor<1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>
    return %1 : tensor<1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>

    // CHECK:               [[SHAPEOF:%.+]] = IE.ShapeOf([[ARG0]]) {dstElemType = si64} : tensor<1x1x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}> -> tensor<3xsi64>
    // CHECK-NOT:           IE.Broadcast
    // CHECK-NOT:           IE.DynamicBroadcast
    // CHECK:               [[TILE:%.+]] = IE.DynamicTile([[ARG1]], [[SHAPEOF]]) {output_bounds = [1, 1, 10], output_shape = [1, 1, -9223372036854775808]} : tensor<1x1x1xf32>, tensor<3xsi64> -> tensor<1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>
    // CHECK:               return [[TILE]]
  }
}

// -----

#C = affine_map<(d0) -> (d0)>
#CHW = affine_map<(d1, d2, d3) -> (d1, d2, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DynamicBroadcastShapeSubgraph {
 net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_3966" : tensor<1x1x8x3xsi32>
    DataInfo "input_1" : tensor<1x1x1xf32>
  } outputsInfo : {
    DataInfo "Broadcast_3969" friendlyName = "Result_3973" : tensor<1x1x10x5xf32>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>, [[ARG1:%.+]]: tensor<1x1x1xf32>) -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> {
  func.func @main(%arg0: tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<1x1x1xf32>) -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    %1 = IE.DynamicBroadcast(%arg1, %0) {mode = #IE.broadcast_type<NUMPY>, output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x1xf32>, tensor<4xsi64> -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
    return %1 : tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:               [[SHAPEOF:%.+]] = IE.ShapeOf([[ARG0]]) {dstElemType = si64} : tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    // CHECK-NOT:           IE.Broadcast
    // CHECK-NOT:           IE.DynamicBroadcast
    // CHECK:               [[RESHAPE:%.+]] = IE.AffineReshape([[ARG1]])
    // CHECK:               [[TILE:%.+]] = IE.DynamicTile([[RESHAPE]], [[SHAPEOF]]) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x1x1xf32>, tensor<4xsi64> -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:               return [[TILE]]
  }
}

// -----

#C = affine_map<(d0) -> (d0)>
#CHW = affine_map<(d1, d2, d3) -> (d1, d2, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DynamicBroadcastShapeSubgraph {
 net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_3966" : tensor<1x1x8x3xsi32>
    DataInfo "input_1" : tensor<1x2x1xf32>
  } outputsInfo : {
    DataInfo "Broadcast_3969" friendlyName = "Result_3973" : tensor<1x1x10x5xf32>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>, [[ARG1:%.+]]: tensor<1x2x1xf32>) -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> {
  func.func @main(%arg0: tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<1x2x1xf32>) -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.ShapeOf(%arg0) {dstElemType = si64} : tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    %1 = IE.DynamicBroadcast(%arg1, %0) {mode = #IE.broadcast_type<NUMPY>, output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x2x1xf32>, tensor<4xsi64> -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
    return %1 : tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:               [[SHAPEOF:%.+]] = IE.ShapeOf([[ARG0]]) {dstElemType = si64} : tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    // CHECK-NOT:           IE.Broadcast
    // CHECK-NOT:           IE.DynamicBroadcast
    // CHECK:               [[RESHAPE:%.+]] = IE.AffineReshape([[ARG1]])
    // CHECK:               [[TILE:%.+]] = IE.DynamicTile([[RESHAPE]], [[SHAPEOF]]) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x2x1xf32>, tensor<4xsi64> -> tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:               return [[TILE]]
  }
}
