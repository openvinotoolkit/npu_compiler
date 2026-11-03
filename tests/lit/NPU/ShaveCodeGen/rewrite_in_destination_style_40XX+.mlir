//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="platform=%platform%" --rewrite-in-destination-style | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

module @ConstantPad {
  module @VPU.SW {
    func.func @generated_0(%input: tensor<1x12x8xf16>) -> tensor<1x16x8xf16> {
      %cst = arith.constant 1.000000e+00 : f16
      %padded = tensor.pad %input low[0, 3, 0] high[0, 1, 0] {
      ^bb0(%i0: index, %i1: index, %i2: index):
        tensor.yield %cst : f16
      } : tensor<1x12x8xf16> to tensor<1x16x8xf16>
      return %padded : tensor<1x16x8xf16>
    }
  }
}

// CHECK-LABEL: module @ConstantPad
// CHECK:         func.func @generated_0
// CHECK-SAME:      ([[INPUT:%[^:]+]]: tensor<1x12x8xf16>) -> tensor<1x16x8xf16> {
// CHECK-NEXT:      arith.constant 1.000000e+00 : f16
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x8xf16>
// CHECK-NEXT:      [[PAD_VAL:%.+]] = arith.constant 1.000000e+00 : f16
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[PAD_VAL]] : f16) outs([[EMPTY]] : tensor<1x16x8xf16>) -> tensor<1x16x8xf16>
// CHECK-NEXT:      [[INSERT:%.+]] = tensor.insert_slice [[INPUT]] into [[FILL]][0, 3, 0] [1, 12, 8] [1, 1, 1] : tensor<1x12x8xf16> into tensor<1x16x8xf16>
// CHECK-NEXT:      return [[INSERT]] : tensor<1x16x8xf16>

// -----

module @Generate {
  module @VPU.SW {
    func.func @generated_0() -> tensor<4x8xf32> {
      %generated = tensor.generate {
      ^bb0(%i0: index, %i1: index):
        %idx = arith.index_cast %i0 : index to i32
        %val = arith.sitofp %idx : i32 to f32
        tensor.yield %val : f32
      } : tensor<4x8xf32>
      return %generated : tensor<4x8xf32>
    }
  }
}

// CHECK-LABEL: module @Generate
// CHECK:         func.func @generated_0() -> tensor<4x8xf32> {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<4x8xf32>
// CHECK-NEXT:      [[GEN:%.+]] = linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK-SAME:        outs([[EMPTY]] : tensor<4x8xf32>)
// CHECK-NEXT:      ^bb0({{%.+}}: f32):
// CHECK-NEXT:        [[DIM0:%.+]] = linalg.index 0 : index
// CHECK:             [[CAST:%.+]] = arith.index_cast [[DIM0]] : index to i32
// CHECK-NEXT:        [[VAL:%.+]] = arith.sitofp [[CAST]] : i32 to f32
// CHECK-NEXT:        linalg.yield [[VAL]] : f32
// CHECK-NEXT:      } -> tensor<4x8xf32>
// CHECK-NEXT:      return [[GEN]] : tensor<4x8xf32>

// -----

module @FromElements {
  module @VPU.SW {
    func.func @generated_0(%elem0: f32, %elem1: f32) -> tensor<2xf32> {
      %tensor = tensor.from_elements %elem0, %elem1 : tensor<2xf32>
      return %tensor : tensor<2xf32>
    }
  }
}

// CHECK-LABEL: module @FromElements
// CHECK:         func.func @generated_0
// CHECK-SAME:      ([[ELEM_0:%[^:]+]]: f32, [[ELEM_1:%[^:]+]]: f32) -> tensor<2xf32> {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2xf32>
// CHECK-NEXT:      [[C0:%.+]] = arith.constant 0 : index
// CHECK-NEXT:      [[C1:%.+]] = arith.constant 1 : index
// CHECK-NEXT:      [[INSERT_0:%.+]] = tensor.insert [[ELEM_0]] into [[EMPTY]]{{\[}}[[C0]]] : tensor<2xf32>
// CHECK-NEXT:      [[INSERT_1:%.+]] = tensor.insert [[ELEM_1]] into [[INSERT_0]]{{\[}}[[C1]]] : tensor<2xf32>
// CHECK-NEXT:      return [[INSERT_1]] : tensor<2xf32>
