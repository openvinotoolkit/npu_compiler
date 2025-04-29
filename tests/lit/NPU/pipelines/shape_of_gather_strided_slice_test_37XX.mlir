//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE --set-upper-bounds="3 5" shape-of-gather-strided-slice-test.xml -o %t
// RUN: FileCheck %s --input-file %t
// REQUIRES: arch-NPU37XX

// CHECK:  #C = affine_map<(d0) -> (d0)>
// CHECK:  #NC = affine_map<(d0, d1) -> (d0, d1)>

// CHECK:       func.func @main([[ARG0:[^:]+]]: tensor<3x?xsi64, {bounds = [3, 5], order = #NC}>)
// CHECK-SAME:      -> (tensor<?x3xsi64, {bounds = [5, 3], order = #NC}>, tensor<?xf32, {bounds = [9], order = #C}>) {

// CHECK-DAG:   [[CST_STRIDEDSLICE_IN:%.+]] = const.Declare tensor<9xf32>
// CHECK-DAG:   [[CST_STRIDEDSLICE_BEGINS:%.+]] = const.Declare tensor<1xsi64>

// CHECK-DAG:   [[CST_TRANSPOSE:%.+]] = const.Declare tensor<2xsi64>
// CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]], [[CST_TRANSPOSE]])
// CHECK-SAME:      -> tensor<?x3xsi64, {bounds = [5, 3], order = #NC}>
// CHECK:       [[SHAPE_OF:%.+]] = IE.ShapeOf([[TRANSPOSE]]) {dstElemType = si64}
// CHECK-SAME:      -> tensor<2xsi64>
// CHECK:       [[CST_GATHER_INDICES:%.+]] = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
// CHECK:       [[CST_GATHER_AXIS:%.+]] = const.Declare tensor<si64> = dense<0> : tensor<si64>
// CHECK:       [[GATHER:%.+]] = IE.Gather([[SHAPE_OF]], [[CST_GATHER_INDICES]], [[CST_GATHER_AXIS]])
// CHECK-SAME:      -> tensor<1xsi64>
// CHECK:       [[CST_STRIDEDSLICE_STRIDES:%.+]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
// CHECK:       [[STRIDEDSLICE:%.+]] = IE.StridedSlice([[CST_STRIDEDSLICE_IN]], [[CST_STRIDEDSLICE_BEGINS]], [[GATHER]], [[CST_STRIDEDSLICE_STRIDES]])
// CHECK-SAME:      -> tensor<?xf32, {bounds = [9], order = #C}>

// CHECK:       return [[TRANSPOSE]], [[STRIDEDSLICE]]
// CHECK:     }
