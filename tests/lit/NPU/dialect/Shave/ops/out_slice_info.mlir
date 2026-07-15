//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: func.func @OutSliceInfo1D
// CHECK-SAME:      ([[LOOP_ID:%.+]]: index, [[TILE_NUM:%.+]]: index)
func.func @OutSliceInfo1D(%loopId: index, %tileNum: index) -> (index, index) {
  // CHECK: [[SIZES:%.+]], [[OFFSETS:%.+]] = Shave.OutSliceInfo([[LOOP_ID]], [[TILE_NUM]]) -> sizes(index), offsets(index)
  %info:2 = Shave.OutSliceInfo(%loopId, %tileNum) -> sizes(index), offsets(index)
  // CHECK-NEXT: return [[SIZES]], [[OFFSETS]] : index, index
  return %info#0, %info#1 : index, index
}

// -----

// CHECK-LABEL: func.func @OutSliceInfo2D
// CHECK-SAME:      ([[LOOP_ID:%.+]]: index, [[TILE_NUM:%.+]]: index)
func.func @OutSliceInfo2D(%loopId: index, %tileNum: index) -> (index, index, index, index) {
  // CHECK: [[SIZES:%.+]]:2, [[OFFSETS:%.+]]:2 = Shave.OutSliceInfo([[LOOP_ID]], [[TILE_NUM]]) -> sizes(index, index), offsets(index, index)
  %info:4 = Shave.OutSliceInfo(%loopId, %tileNum) -> sizes(index, index), offsets(index, index)
  // CHECK-NEXT: return [[SIZES]]#0, [[SIZES]]#1, [[OFFSETS]]#0, [[OFFSETS]]#1 : index, index, index, index
  return %info#0, %info#1, %info#2, %info#3 : index, index, index, index
}

// -----

// CHECK-LABEL: func.func @OutSliceInfo3D
// CHECK-SAME:      ([[LOOP_ID:%.+]]: index, [[TILE_NUM:%.+]]: index)
func.func @OutSliceInfo3D(%loopId: index, %tileNum: index) -> (index, index, index, index, index, index) {
  // CHECK: [[SIZES:%.+]]:3, [[OFFSETS:%.+]]:3 = Shave.OutSliceInfo([[LOOP_ID]], [[TILE_NUM]]) -> sizes(index, index, index), offsets(index, index, index)
  %info:6 = Shave.OutSliceInfo(%loopId, %tileNum) -> sizes(index, index, index), offsets(index, index, index)
  // CHECK-NEXT: return [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2 : index, index, index, index, index, index
  return %info#0, %info#1, %info#2, %info#3, %info#4, %info#5 : index, index, index, index, index, index
}
