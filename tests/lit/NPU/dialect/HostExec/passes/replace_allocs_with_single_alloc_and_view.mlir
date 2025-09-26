//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --replace-allocs-with-single-alloc-and-views --canonicalize --cse --verify-diagnostics %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

func.func @StaticShapeAllocs(%arg0: memref<1x1024xf16>, %arg1: memref<1x1024xf16>) -> memref<1x1024xf16> {
    %alloc = memref.alloc() : memref<1x1024xf16>
    memref.copy %arg0, %alloc : memref<1x1024xf16> to memref<1x1024xf16>

    %alloc1 = memref.alloc() : memref<1x1024xf16>
    memref.copy %alloc, %alloc1 : memref<1x1024xf16> to memref<1x1024xf16>

    memref.copy %alloc1, %arg1 : memref<1x1024xf16> to memref<1x1024xf16>

    return %arg1 : memref<1x1024xf16>
}

// CHECK: func.func @StaticShapeAllocs([[ARG0:%.+]]: memref<1x1024xf16>, [[ARG1:%.+]]: memref<1x1024xf16>) -> memref<1x1024xf16> {
// CHECK:   [[C0:%.+]] = arith.constant 0 : index
// CHECK:   [[C2048:%.+]] = arith.constant 2048 : index
// CHECK:   [[ALLOC:%.+]] = memref.alloc() {alignment = 64 : i64} : memref<4096xi8>
// CHECK:   [[VIEW:%.+]] = memref.view [[ALLOC]][[[C0]]][] : memref<4096xi8> to memref<1x1024xf16>
// CHECK:   [[VIEW_0:%.+]] = memref.view [[ALLOC]][[[C2048]]][] : memref<4096xi8> to memref<1x1024xf16>
// CHECK:   memref.copy [[ARG0]], [[VIEW]] : memref<1x1024xf16> to memref<1x1024xf16>
// CHECK:   memref.copy [[VIEW]], [[VIEW_0]] : memref<1x1024xf16> to memref<1x1024xf16>
// CHECK:   memref.copy [[VIEW_0]], [[ARG1]] : memref<1x1024xf16> to memref<1x1024xf16>

// CHECK:   return [[ARG1]] : memref<1x1024xf16>

// -----

func.func @DynamicShapeNoAlloc(%arg0: memref<1x?x?xf16>, %arg1: memref<1x?x?xf16>) -> memref<1x?x?xf16> {
    memref.copy %arg0, %arg1 : memref<1x?x?xf16> to memref<1x?x?xf16>

    return %arg1 : memref<1x?x?xf16>
    // CHECK-NOT: memref.alloc
}

// -----

func.func @DynamicShapeSingleAlloc(%arg0: memref<1x?x?xf16>, %arg1: memref<1x?x?xf16>) -> memref<1x?x?xf16> {
    %c1 = arith.constant 1 : index
    %dim1 = memref.dim %arg0, %c1 : memref<1x?x?xf16>

    %c2 = arith.constant 2 : index
    %dim2 = memref.dim %arg0, %c2 : memref<1x?x?xf16>

    %alloc = memref.alloc(%dim1, %dim2) : memref<1x?x?xf16>
    memref.copy %arg0, %alloc : memref<1x?x?xf16> to memref<1x?x?xf16>

    memref.copy %alloc, %arg1 : memref<1x?x?xf16> to memref<1x?x?xf16>

    return %arg1 : memref<1x?x?xf16>
}

// CHECK: func.func @DynamicShapeSingleAlloc([[ARG0:%.+]]: memref<1x?x?xf16>, [[ARG1:%.+]]: memref<1x?x?xf16>) -> memref<1x?x?xf16> {
// CHECK-DAG:   [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:   [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:   [[C2:%.+]] = arith.constant 2 : index

// CHECK:   [[DIM1:%.+]] = memref.dim [[ARG0]], [[C1]] : memref<1x?x?xf16>
// CHECK:   [[DIM2:%.+]] = memref.dim [[ARG0]], [[C2]] : memref<1x?x?xf16>
// CHECK:   [[MULTIPLY1:%.+]] = arith.muli [[DIM1]], [[DIM2]] : index
// CHECK:   [[MULTIPLY2:%.+]] = arith.muli [[MULTIPLY1]], [[C2]] : index

// CHECK:   [[ALLOC:%.+]] = memref.alloc([[MULTIPLY2]]) {alignment = 64 : i64} : memref<?xi8>
// CHECK:   [[VIEW:%.+]] = memref.view [[ALLOC]][[[C0]]][[[DIM1]], [[DIM2]]] : memref<?xi8> to memref<1x?x?xf16>

// CHECK:   memref.copy [[ARG0]], [[VIEW]] : memref<1x?x?xf16> to memref<1x?x?xf16>
// CHECK:   memref.copy [[VIEW]], [[ARG1]] : memref<1x?x?xf16> to memref<1x?x?xf16>

// CHECK:   return [[ARG1]] : memref<1x?x?xf16>

// -----

func.func @AllocWithSubviews(%arg0: memref<1x16x720x?xf16>, %arg1: memref<1x16x720x40xf16>, %arg2: index) -> memref<1x16x720x40xf16> {
    %c3 = arith.constant 3 : index

    %dim = memref.dim %arg0, %c3 : memref<1x16x720x?xf16>
    %alloc = memref.alloc(%dim) : memref<1x16x720x?xf16>

    %subview = memref.subview %arg0[0, 0, 0, %arg2] [1, 16, 720, 40] [1, 1, 1, 1]
             : memref<1x16x720x?xf16> to memref<1x16x720x40xf16, strided<[?, ?, ?, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[0, 0, 0, %arg2] [1, 16, 720, 40] [1, 1, 1, 1]
               : memref<1x16x720x?xf16> to memref<1x16x720x40xf16, strided<[?, ?, ?, 1], offset: ?>>
    memref.copy %subview, %subview_2 : memref<1x16x720x40xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x720x40xf16, strided<[?, ?, ?, 1], offset: ?>>
    memref.copy %subview_2, %arg1 : memref<1x16x720x40xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x720x40xf16>

    return %arg1 : memref<1x16x720x40xf16>
}

// CHECK: func.func @AllocWithSubviews([[ARG0:%.+]]: memref<1x16x720x?xf16>, [[ARG1:%.+]]: memref<1x16x720x40xf16>, [[ARG2:%.+]]: index) -> memref<1x16x720x40xf16> {
// CHECK:   [[C23040:%.+]] = arith.constant 23040 : index
// CHECK:   [[C0:%.+]] = arith.constant 0 : index
// CHECK:   [[C3:%.+]] = arith.constant 3 : index

// CHECK:   [[DIM:%.+]] = memref.dim [[ARG0]], [[C3]] : memref<1x16x720x?xf16>
// CHECK:   [[MULTIPLY:%.+]] = arith.muli [[DIM]], [[C23040]] : index
// CHECK:   [[ALLOC:%.+]] = memref.alloc([[MULTIPLY]]) {alignment = 64 : i64} : memref<?xi8>
// CHECK:   [[VIEW:%.+]] = memref.view [[ALLOC]][[[C0]]][[[DIM]]] : memref<?xi8> to memref<1x16x720x?xf16>
// CHECK:   [[SUBVIEW:%.+]] = memref.subview [[ARG0]][0, 0, 0, [[ARG2]]] [1, 16, 720, 40] [1, 1, 1, 1]
// CHECK:   [[SUBVIEW_0:%.+]] = memref.subview [[VIEW]][0, 0, 0, [[ARG2]]] [1, 16, 720, 40] [1, 1, 1, 1]
// CHECK:   memref.copy [[SUBVIEW]], [[SUBVIEW_0]]
// CHECK:   memref.copy [[SUBVIEW_0]], [[ARG1]]
// CHECK:   return [[ARG1]]

// -----

// expected-error@+1 {{ReplaceAllocsWithSingleAllocAndViewsPass cannot be applied to functions with dealloc operations}}
func.func @AllocDeallocInvalid() {
    %alloc = memref.alloc() : memref<1x16x720x40xf16>
    memref.dealloc %alloc : memref<1x16x720x40xf16>
    return
}
