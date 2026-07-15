//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU3720

#C = affine_map<(d0) -> (d0)>
#NC = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @DynamicShapeOfGatherStridedSlice
module @DynamicShapeOfGatherStridedSlice {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<3x5xsi64>
  } outputsInfo : {
    DataInfo "output1" : tensor<5x3xsi64>
    DataInfo "output2" : tensor<5xf32>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<3x?xsi64, {bounds = #const.OpaqueI64Elements<[3, 5]> : tensor<2xsi64>, order = #NC}>) -> (tensor<?x3xsi64, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #NC}>, tensor<?xf32, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}>) {
  func.func @main(%arg0: tensor<3x?xsi64, {bounds = #const.OpaqueI64Elements<[3, 5]> : tensor<2xsi64>, order = #NC}>) -> (tensor<?x3xsi64, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #NC}>, tensor<?xf32, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}>) {
    %cst = const.Declare tensor<9xf32> = dense<[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -1.000000e+00]> : tensor<9xf32>
    %cst_0 = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
    %cst_1 = const.Declare tensor<2xsi64> = dense<[1, 0]> : tensor<2xsi64>

    %0 = IE.Transpose(%arg0, %cst_1) : tensor<3x?xsi64, {bounds = #const.OpaqueI64Elements<[3, 5]> : tensor<2xsi64>, order = #NC}>, tensor<2xsi64> -> tensor<?x3xsi64, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #NC}>
    %1 = IE.ShapeOf(%0) {dstElemType = si64} : tensor<?x3xsi64, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #NC}> -> tensor<2xsi64>

    %cst_2 = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
    %cst_3 = const.Declare tensor<si64> = dense<0> : tensor<si64>
    %2 = IE.Gather(%1, %cst_2, %cst_3) {batch_dims = 0 : i64} : tensor<2xsi64>, tensor<1xsi64>, tensor<si64> -> tensor<1xsi64>
    %cst_4 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    %3 = IE.StridedSlice(%cst, %cst_0, %2, %cst_4) {begin_mask = [0], ellipsis_mask = [], end_mask = [0], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 1, 1>, shrink_axis_mask = []}
           : tensor<9xf32>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<?xf32, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}>

    return %0, %3 : tensor<?x3xsi64, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #NC}>, tensor<?xf32, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}>

    // CHECK: [[CONST0:%.+]] = const.Declare tensor<9xf16>
    // CHECK: [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #CN, mem_perm = #NC}
    // CHECK-SAME: : tensor<3x?xsi64, {bounds = #const.OpaqueI64Elements<[3, 5]> : tensor<2xsi64>, order = #NC}> -> tensor<?x3xsi64, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #CN}>
    // CHECK: [[MEM_PERMUTE0:%.+]] = IE.MemPermute([[ARG0]]) {dst_order = #NC, mem_perm = #CN}
    // CHECK-SAME: : tensor<3x?xsi64, {bounds = #const.OpaqueI64Elements<[3, 5]> : tensor<2xsi64>, order = #NC}> -> tensor<?x3xsi64, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #NC}>
    // CHECK: [[CONVERT0:%.+]] = IE.Convert([[PERMUTE_CAST]]) {dstElemType = si32}
    // CHECK-SAME: : tensor<?x3xsi64, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #CN}> -> tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #CN}>
    // CHECK: [[MEM_PERMUTE1:%.+]] = IE.MemPermute([[CONVERT0]]) {dst_order = #NC, mem_perm = #CN}
    // CHECK-SAME: : tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #CN}> -> tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #NC}>
    // CHECK: [[SHAPE_OF:%.+]] = IE.ShapeOf([[MEM_PERMUTE1]]) {dstElemType = si32}
    // CHECK-SAME: : tensor<?x3xsi32, {bounds = #const.OpaqueI64Elements<[5, 3]> : tensor<2xsi64>, order = #NC}> -> tensor<2xsi32>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [0] [1] : tensor<2xsi32> to tensor<1xsi32>
    // CHECK: [[STRIDED_SLICE:%.+]] = IE.StridedSlice([[CONST0]], [[SLICE]])
    // CHECK-SAME: {begin_mask = [0], begins_attr = [0], ellipsis_mask = [], end_mask = [0], new_axis_mask = [],
    // CHECK-SAME:  operandSegmentSizes = array<i32: 1, 0, 1, 0>, shrink_axis_mask = [], strides_attr = [1]}
    // CHECK-SAME: : tensor<9xf16>, tensor<1xsi32> -> tensor<?xf16, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}>
    // CHECK: [[CONVERT1:%.+]] = IE.Convert([[STRIDED_SLICE]]) {dstElemType = f32}
    // CHECK-SAME: : tensor<?xf16, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}> -> tensor<?xf32, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}>

    // CHECK: return [[MEM_PERMUTE0]], [[CONVERT1]]
  }
}
