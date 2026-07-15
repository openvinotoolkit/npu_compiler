//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --propagate-dynamic-shapes --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010


module @DirectDynamicInput {
  module @Module0 {
    func.func nested @main_func0(%arg0: memref<1x16x4x?xf16>, %arg1: memref<4xsi32>,
                                  %arg2: memref<1x16x4x?xf16>, %arg3: memref<4xsi32>)
                                  -> (memref<1x16x4x?xf16>, memref<4xsi32>) {
      return %arg2, %arg3 : memref<1x16x4x?xf16>, memref<4xsi32>
    }
  }

  func.func @main(%arg0: memref<1x16x4x?xf16>, %arg1: memref<1x16x4x?xf16>) -> memref<1x16x4x?xf16> {
    %c3 = arith.constant 3 : index
    %dim = memref.dim %arg0, %c3 : memref<1x16x4x?xf16>

    %allocOut = memref.alloc(%dim) : memref<1x16x4x?xf16>
    %allocInShape = memref.alloc() : memref<4xsi32>
    %allocOutShape = memref.alloc() : memref<4xsi32>

    %res:2 = Core.NestedCall @Module0::@main_func0(%arg0, %allocInShape, %allocOut, %allocOutShape)
            : (memref<1x16x4x?xf16>, memref<4xsi32>, memref<1x16x4x?xf16>, memref<4xsi32>)
              -> (memref<1x16x4x?xf16>, memref<4xsi32>)

    memref.copy %res#0, %arg1 : memref<1x16x4x?xf16> to memref<1x16x4x?xf16>
    return %arg1 : memref<1x16x4x?xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x16x4x?xf16>, [[ARG1:%.+]]: memref<1x16x4x?xf16>)

    // Indices:
    // CHECK:         [[C1_I32:%.+]] = arith.constant 1 : i32
    // CHECK:         [[C16_I32:%.+]] = arith.constant 16 : i32
    // CHECK:         [[C4_I32:%.+]] = arith.constant 4 : i32
    // CHECK:         [[C2:%.+]] = arith.constant 2 : index
    // CHECK:         [[C1:%.+]] = arith.constant 1 : index
    // CHECK:         [[C0:%.+]] = arith.constant 0 : index
    // CHECK:         [[C3:%.+]] = arith.constant 3 : index

    // CHECK:         [[ALLOC_IN:%.+]] = memref.alloc() : memref<4xsi32>

    // Dynamic shape construction:
    // CHECK:         [[DIM:%.+]] = memref.dim [[ARG0]], [[C3]]
    // CHECK:         [[DIMI32:%.+]] = arith.index_cast [[DIM]] : index to i32
    // CHECK:         [[S0:%.+]] = builtin.unrealized_conversion_cast [[DIMI32]] : i32 to si32
    // CHECK:         memref.store [[S0]], [[ALLOC_IN]][[[C0]]]
    // CHECK:         [[S1:%.+]] = builtin.unrealized_conversion_cast [[C4_I32]] : i32 to si32
    // CHECK:         memref.store [[S1]], [[ALLOC_IN]][[[C1]]]
    // CHECK:         [[S2:%.+]] = builtin.unrealized_conversion_cast [[C16_I32]] : i32 to si32
    // CHECK:         memref.store [[S2]], [[ALLOC_IN]][[[C2]]]
    // CHECK:         [[S3:%.+]] = builtin.unrealized_conversion_cast [[C1_I32]] : i32 to si32
    // CHECK:         memref.store [[S3]], [[ALLOC_IN]][[[C3]]]

    // CHECK:         Core.NestedCall @Module0::@main_func0([[ARG0]], [[ALLOC_IN]]
  }
}

// -----

module @MultipleInputsMultipleDynamicDims {
  module @Module0 {
    func.func nested @main_func0(%arg0: memref<2x3xf32>,
                                  %arg1: memref<?x?x548xf16>, %arg2: memref<?x?x548xf16>,
                                  %arg3: memref<3xsi32>, %arg4: memref<3xsi32>,
                                  %arg5: memref<?x?xf32>, %arg6: memref<2xsi32>)
                                  -> (memref<?x?xf32>, memref<2xsi32>) {
      return %arg5, %arg6 : memref<?x?xf32>, memref<2xsi32>
    }
  }

  func.func @main(%arg0: memref<2x3xf32>,
                  %arg1: memref<?x?x548xf16>, %arg2: memref<?x?x548xf16>,
                  %arg3: memref<?x?xf32>, %arg4: memref<?x?x548xf16>, %arg5: memref<?x?x548xf16>) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %allocInShape1 = memref.alloc() : memref<3xsi32>
    %allocInShape2 = memref.alloc() : memref<3xsi32>
    %allocOutShape = memref.alloc() : memref<2xsi32>

    %res:2 = Core.NestedCall @Module0::@main_func0(%arg0, %arg1, %arg2, %allocInShape1, %allocInShape2,
                                                    %arg3, %allocOutShape)
            : (memref<2x3xf32>, memref<?x?x548xf16>, memref<?x?x548xf16>, memref<3xsi32>, memref<3xsi32>,
               memref<?x?xf32>, memref<2xsi32>)
              -> (memref<?x?xf32>, memref<2xsi32>)

    memref.copy %res#0, %arg3 : memref<?x?xf32> to memref<?x?xf32>
    return %arg3 : memref<?x?xf32>

    // CHECK: func.func @main([[ARG0:%.+]]: memref<2x3xf32>, [[ARG1:%.+]]: memref<?x?x548xf16>, [[ARG2:%.+]]: memref<?x?x548xf16>, [[ARG3:%.+]]: memref<?x?xf32>

    // Indices:
    // CHECK:         [[C548:%.+]] = arith.constant 548 : i32
    // CHECK:         [[C2:%.+]] = arith.constant 2 : index
    // CHECK:         [[C0:%.+]] = arith.constant 0 : index
    // CHECK:         [[C1:%.+]] = arith.constant 1 : index

    // CHECK:         [[ALLOC1:%.+]] = memref.alloc() : memref<3xsi32>
    // CHECK:         [[ALLOC2:%.+]] = memref.alloc() : memref<3xsi32>

    // Dynamic shapes construction:
    // CHECK:         [[S1_0:%.+]] = builtin.unrealized_conversion_cast [[C548]] : i32 to si32
    // CHECK:         memref.store [[S1_0]], [[ALLOC1]][[[C0]]]
    // CHECK:         [[DIM1_1:%.+]] = memref.dim [[ARG1]], [[C1]]
    // CHECK:         [[DIM1_1I32:%.+]] = arith.index_cast [[DIM1_1]] : index to i32
    // CHECK:         [[S1_1:%.+]] = builtin.unrealized_conversion_cast [[DIM1_1I32]] : i32 to si32
    // CHECK:         memref.store [[S1_1]], [[ALLOC1]][[[C1]]]
    // CHECK:         [[DIM1_0:%.+]] = memref.dim [[ARG1]], [[C0]]
    // CHECK:         [[DIM1_0I32:%.+]] = arith.index_cast [[DIM1_0]] : index to i32
    // CHECK:         [[S1_2:%.+]] = builtin.unrealized_conversion_cast [[DIM1_0I32]] : i32 to si32
    // CHECK:         memref.store [[S1_2]], [[ALLOC1]][[[C2]]]
    // CHECK:         [[S2_0:%.+]] = builtin.unrealized_conversion_cast [[C548]] : i32 to si32
    // CHECK:         memref.store [[S2_0]], [[ALLOC2]][[[C0]]]
    // CHECK:         [[DIM2_1:%.+]] = memref.dim [[ARG2]], [[C1]]
    // CHECK:         [[DIM2_1I32:%.+]] = arith.index_cast [[DIM2_1]] : index to i32
    // CHECK:         [[S2_1:%.+]] = builtin.unrealized_conversion_cast [[DIM2_1I32]] : i32 to si32
    // CHECK:         memref.store [[S2_1]], [[ALLOC2]][[[C1]]]
    // CHECK:         [[DIM2_0:%.+]] = memref.dim [[ARG2]], [[C0]]
    // CHECK:         [[DIM2_0I32:%.+]] = arith.index_cast [[DIM2_0]] : index to i32
    // CHECK:         [[S2_2:%.+]] = builtin.unrealized_conversion_cast [[DIM2_0I32]] : i32 to si32
    // CHECK:         memref.store [[S2_2]], [[ALLOC2]][[[C2]]]

    // CHECK:         Core.NestedCall @Module0::@main_func0([[ARG0]], [[ARG1]], [[ARG2]], [[ALLOC1]], [[ALLOC2]], [[ARG3]]
  }
}

// -----

module @NoDynamicInputsNoop {
  module @Module0 {
    func.func nested @main_func0(%arg0: memref<1x16x4x32xf16>, %arg1: memref<1x16x4x32xf16>)
                                  -> memref<1x16x4x32xf16> {
      return %arg1 : memref<1x16x4x32xf16>
    }
  }

  func.func @main(%arg0: memref<1x16x4x32xf16>, %arg1: memref<1x16x4x32xf16>) -> memref<1x16x4x32xf16> {
    %res = Core.NestedCall @Module0::@main_func0(%arg0, %arg1)
           : (memref<1x16x4x32xf16>, memref<1x16x4x32xf16>) -> memref<1x16x4x32xf16>

    memref.copy %res, %arg1 : memref<1x16x4x32xf16> to memref<1x16x4x32xf16>
    return %arg1 : memref<1x16x4x32xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x16x4x32xf16>, [[ARG1:%.+]]: memref<1x16x4x32xf16>)
    // CHECK-NOT: memref.store
    // CHECK: [[RES:%.+]] = Core.NestedCall @Module0::@main_func0([[ARG0]], [[ARG1]])
  }
}
