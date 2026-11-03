//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ShaveCodeGen SW-kernel bufferization (one-shot-bufferize-sw-kernels) for standard tensor ops.
// An outlined @VPU.SW kernel gets plain host tensors, so bufferization must match upstream MLIR
// (standard strided memrefs) and leak no vpux::MemRefAttr; the NOLEAK RUN asserts no `bounds =` / `order =`.

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --one-shot-bufferize-sw-kernels %s > %t
// RUN: FileCheck %s --input-file %t
// RUN: FileCheck %s --check-prefix=NOLEAK --input-file %t
// NPU3720 is excluded: it predates the ShaveCodeGen SW-kernel bufferization path.
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU5020

// extract_slice, dynamic tile size: strided memref.subview, result via memref.alloc + copy.
module @ExtractSlicePlain {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x16x175x512xf16>, %off: index, %sz: index) -> tensor<1x16x175x?xf16> {
      %e = tensor.extract_slice %arg0[0, 0, 0, %off] [1, 16, 175, %sz] [1, 1, 1, 1]
          : tensor<1x16x175x512xf16> to tensor<1x16x175x?xf16>
      return %e : tensor<1x16x175x?xf16>
    }
  }

  // CHECK-LABEL: @ExtractSlicePlain
  // CHECK:       [[SUB:%.+]] = memref.subview
  // CHECK-SAME:    to memref<1x16x175x?xf16, strided<[1433600, 89600, 512, 1], offset: ?>>
  // CHECK:       [[ALLOC:%.+]] = memref.alloc(%{{.+}}) {{.*}}: memref<1x16x175x?xf16>
  // CHECK:       memref.copy [[SUB]], [[ALLOC]]
  // CHECK:       return [[ALLOC]]

  // NOLEAK-LABEL: @ExtractSlicePlain
  // NOLEAK-NOT:  bounds =
  // NOLEAK-NOT:  order =
}

// -----

// cast refining a dynamic shape to static: plain memref.cast, not a vpux::MemRefAttr memref.
module @CastPlain {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x16x175x?xf16>) -> tensor<1x16x175x16xf16> {
      %c = tensor.cast %arg0 : tensor<1x16x175x?xf16> to tensor<1x16x175x16xf16>
      return %c : tensor<1x16x175x16xf16>
    }
  }

  // CHECK-LABEL: @CastPlain
  // CHECK:       [[CAST:%.+]] = memref.cast %{{.+}} : memref<1x16x175x?xf16> to memref<1x16x175x16xf16>
  // CHECK:       return [[CAST]]

  // NOLEAK-LABEL: @CastPlain
  // NOLEAK-NOT:  bounds =
  // NOLEAK-NOT:  order =
}

// -----

// cast Case 1 (unranked source): memref.cast to a fully-dynamic strided memref, then alloc + copy into
// the plain return type. No bounds.
module @CastUnrankedSource {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<*xf16>) -> tensor<1x16x175x16xf16> {
      %c = tensor.cast %arg0 : tensor<*xf16> to tensor<1x16x175x16xf16>
      return %c : tensor<1x16x175x16xf16>
    }
  }

  // CHECK-LABEL: @CastUnrankedSource
  // CHECK:       [[CAST:%.+]] = memref.cast %{{.+}} : memref<*xf16> to memref<1x16x175x16xf16, strided<[?, ?, ?, ?], offset: ?>>
  // CHECK:       [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x16x175x16xf16>
  // CHECK:       memref.copy [[CAST]], [[ALLOC]]
  // CHECK:       return [[ALLOC]]

  // NOLEAK-LABEL: @CastUnrankedSource
  // NOLEAK-NOT:  bounds =
  // NOLEAK-NOT:  order =
}

// -----

// cast Case 2 (unranked result): memref.cast to an unranked memref (element type + memory space). No bounds.
module @CastUnrankedResult {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x16x175x16xf16>) -> tensor<*xf16> {
      %c = tensor.cast %arg0 : tensor<1x16x175x16xf16> to tensor<*xf16>
      return %c : tensor<*xf16>
    }
  }

  // CHECK-LABEL: @CastUnrankedResult
  // CHECK:       [[CAST:%.+]] = memref.cast %{{.+}} : memref<1x16x175x16xf16> to memref<*xf16>
  // CHECK:       return [[CAST]]

  // NOLEAK-LABEL: @CastUnrankedResult
  // NOLEAK-NOT:  bounds =
  // NOLEAK-NOT:  order =
}

// -----

// tensor.empty (dynamic) + insert_slice accumulator: empty -> plain memref.alloc(%sz) with no bounds,
// insert_slice -> strided memref.subview + copy.
module @EmptyInsertSlicePlain {
  module @VPU.SW {
    func.func @generated_0(%arg0: tensor<1x16x175x?xf16>, %sz: index) -> tensor<1x16x175x?xf16> {
      %c0 = arith.constant 0 : index
      %0 = tensor.empty(%sz) : tensor<1x16x175x?xf16>
      %ins = tensor.insert_slice %arg0 into %0[0, 0, 0, %c0] [1, 16, 175, %sz] [1, 1, 1, 1]
          : tensor<1x16x175x?xf16> into tensor<1x16x175x?xf16>
      return %ins : tensor<1x16x175x?xf16>
    }
  }

  // CHECK-LABEL: @EmptyInsertSlicePlain
  // CHECK:       [[ALLOC:%.+]] = memref.alloc(%{{.+}}) {{.*}}: memref<1x16x175x?xf16>
  // CHECK:       memref.subview [[ALLOC]]
  // CHECK-SAME:    to memref<1x16x175x?xf16, strided<[?, ?, ?, 1], offset: ?>>
  // CHECK:       memref.copy
  // CHECK:       return [[ALLOC]]

  // NOLEAK-LABEL: @EmptyInsertSlicePlain
  // NOLEAK-NOT:  bounds =
  // NOLEAK-NOT:  order =
}
