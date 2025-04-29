//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE slice-to-strided-slice.xml -o %t
// RUN: FileCheck %s --input-file %t
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK: module @"Slice-8" {
// CHECK:   IE.CNNNetwork entryPoint : @main inputsInfo : {
// CHECK:       DataInfo "Parameter_52" : tensor<16xf16>
// CHECK:   } outputsInfo : {
// CHECK:   DataInfo "Slice_57" friendlyName = "Result_58" : tensor<8xf16>
// CHECK:   }
// CHECK:   func.func @main([[ARG0:[^:]+]]: tensor<16xf16>) -> tensor<8xf16> {
// CHECK:      [[CONST:%.*]] = const.Declare tensor<1xsi64> = dense<4> : tensor<1xsi64>
// CHECK:      [[CONST_0:%.*]] = const.Declare tensor<1xsi64> = dense<12> : tensor<1xsi64>
// CHECK:      [[CONST_1:%.*]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
// CHECK:      [[OUT:%.*]] = IE.StridedSlice([[ARG0]], [[CONST]], [[CONST_0]], [[CONST_1]])
// CHECK:       {begin_mask = [0], ellipsis_mask = [], end_mask = [0], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 1, 1>, shrink_axis_mask = []}
// CHECK:        : tensor<16xf16>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<8xf16>
// CHECK:      return [[OUT]]
// CHECK:   }
// CHECK: }
