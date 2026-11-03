//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --expand-non-contiguous-memref-copy %s -o - | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

module @NonContiguousSlice {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x4x1x1000xf16>, %arg1: memref<1x4x1x1000xf16>) {
      %src = memref.subview %arg0[0, 0, 0, 0] [1, 4, 1, 500] [1, 1, 1, 1] : memref<1x4x1x1000xf16> to memref<1x4x1x500xf16, strided<[4000, 1000, 1000, 1]>>
      %dst = memref.subview %arg1[0, 0, 0, 0] [1, 4, 1, 500] [1, 1, 1, 1] : memref<1x4x1x1000xf16> to memref<1x4x1x500xf16, strided<[4000, 1000, 1000, 1]>>
      memref.copy %src, %dst : memref<1x4x1x500xf16, strided<[4000, 1000, 1000, 1]>> to memref<1x4x1x500xf16, strided<[4000, 1000, 1000, 1]>>
      return
    }
  }
}

// CHECK: #[[$NCHW:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NonContiguousSlice
// CHECK:       func.func @generated_0([[ARG0:%.+]]: memref<1x4x1x1000xf16>, [[ARG1:%.+]]: memref<1x4x1x1000xf16>) {
// CHECK:           [[SRC:%.+]] = memref.subview [[ARG0]]
// CHECK:           [[DST:%.+]] = memref.subview [[ARG1]]
// CHECK-NOT:       memref.copy
// CHECK:           linalg.generic
// CHECK-SAME:          indexing_maps = [#[[$NCHW]], #[[$NCHW]]]
// CHECK-SAME:          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:          ins([[SRC]] : memref<1x4x1x500xf16, strided<[4000, 1000, 1000, 1]>>)
// CHECK-SAME:          outs([[DST]] : memref<1x4x1x500xf16, strided<[4000, 1000, 1000, 1]>>)
// CHECK:           ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK:             linalg.yield [[IN]] : f16
// CHECK-NOT:       memref.copy

// -----

module @DynamicStrided {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<?x?xf16, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf16, strided<[?, 1], offset: ?>>) {
      memref.copy %arg0, %arg1 : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      return
    }
  }
}

// CHECK: #[[$NC:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @DynamicStrided
// CHECK:       func.func @generated_0([[ARG0:%.+]]: memref<?x?xf16, strided<[?, 1], offset: ?>>, [[ARG1:%.+]]: memref<?x?xf16, strided<[?, 1], offset: ?>>) {
// CHECK-NOT:       memref.copy
// CHECK:           linalg.generic
// CHECK-SAME:          indexing_maps = [#[[$NC]], #[[$NC]]]
// CHECK-SAME:          iterator_types = ["parallel", "parallel"]
// CHECK-SAME:          ins([[ARG0]] : memref<?x?xf16, strided<[?, 1], offset: ?>>)
// CHECK-SAME:          outs([[ARG1]] : memref<?x?xf16, strided<[?, 1], offset: ?>>)
// CHECK:           ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK:             linalg.yield [[IN]] : f16
// CHECK-NOT:       memref.copy

// -----

module @IdentityLayout {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x1x1x1000xf32>, %arg1: memref<1x1x1x1000xf32>) {
      memref.copy %arg0, %arg1 : memref<1x1x1x1000xf32> to memref<1x1x1x1000xf32>
      return
    }
  }
}

// CHECK-LABEL: @IdentityLayout
// CHECK:       func.func @generated_0([[ARG0:%.+]]: memref<1x1x1x1000xf32>, [[ARG1:%.+]]: memref<1x1x1x1000xf32>) {
// CHECK-NOT:       linalg.generic
// CHECK:           memref.copy [[ARG0]], [[ARG1]] : memref<1x1x1x1000xf32> to memref<1x1x1x1000xf32>
// CHECK-NOT:       linalg.generic

// -----

module @UnitOuterDimsStrided {
  module @VPU.SW {
    func.func @generated_0(%arg0: memref<1x1x1x1000xf16>, %arg1: memref<1x1x1x1000xf16>, %idx: index) {
      %src = memref.subview %arg0[0, 0, 0, %idx] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1000xf16> to memref<1x1x1x1xf16, strided<[1000, 1000, 1000, 1], offset: ?>>
      %dst = memref.subview %arg1[0, 0, 0, %idx] [1, 1, 1, 1] [1, 1, 1, 1] : memref<1x1x1x1000xf16> to memref<1x1x1x1xf16, strided<[1000, 1000, 1000, 1], offset: ?>>
      memref.copy %src, %dst : memref<1x1x1x1xf16, strided<[1000, 1000, 1000, 1], offset: ?>> to memref<1x1x1x1xf16, strided<[1000, 1000, 1000, 1], offset: ?>>
      return
    }
  }
}

// CHECK-LABEL: @UnitOuterDimsStrided
// CHECK:       func.func @generated_0([[ARG0:%.+]]: memref<1x1x1x1000xf16>, [[ARG1:%.+]]: memref<1x1x1x1000xf16>, [[IDX:%.+]]: index) {
// CHECK:           [[SRC:%.+]] = memref.subview [[ARG0]]
// CHECK:           [[DST:%.+]] = memref.subview [[ARG1]]
// CHECK-NOT:       linalg.generic
// CHECK:           memref.copy [[SRC]], [[DST]]
// CHECK-NOT:       linalg.generic
