//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Bounds-aware bufferization of tensor.extract_slice / insert_slice / cast when an scf.for loop's
// affine.min over the induction variable produces the dynamic slice size (the last, uneven tile).
// E#236293 Investigate uniform memory-layout representation after tensor-slice bufferization

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --one-shot-bufferize-VPU-to-VPUIP --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExtractInsertSliceDynamicDimFromAffineMin
// CHECK-SAME:  [[ARG0:%.+]]: memref<1x32x175x512xf16, {order = #NHWC}>
func.func @ExtractInsertSliceDynamicDimFromAffineMin(%arg0: tensor<1x32x175x512xf16, {order = #NHWC}>) -> tensor<1x32x175x512xf16, {order = #NHWC}> {
    %c103 = arith.constant 103 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x32x175x512xf16, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c512 step %c103 iter_args(%arg2 = %0) -> (tensor<1x32x175x512xf16, {order = #NHWC}>) {
        %2 = affine.min affine_map<(d0) -> (-d0 + 512, 103)>(%arg1)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1]
            : tensor<1x32x175x512xf16, {order = #NHWC}>
           to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %extracted_slice into %arg2[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1]
            : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
           into tensor<1x32x175x512xf16, {order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x32x175x512xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x32x175x512xf16, {order = #NHWC}>

    // CHECK-DAG:  [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG:  [[C103:%.+]] = arith.constant 103 : index
    // CHECK-DAG:  [[C512:%.+]] = arith.constant 512 : index
    // CHECK:      [[ACC:%.+]] = memref.alloc() {{.*}}memref<1x32x175x512xf16>
    // CHECK:      scf.for [[IV:%.+]] = [[C0]] to [[C512]] step [[C103]] {
    // CHECK:        [[SIZE:%.+]] = affine.min
    // CHECK:        [[SRC_SUBVIEW:%.+]] = memref.subview [[ARG0]][0, 0, 0, [[IV]]] [1, 32, 175, [[SIZE]]] [1, 1, 1, 1]
    // CHECK-SAME:       to memref<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, offset = -9223372036854775808 : i64, order = #NHWC, strides = [2867200, 1, 16384, 32]}>
    // CHECK:        [[DST_SUBVIEW:%.+]] = memref.subview [[ACC]][0, 0, 0, [[IV]]] [1, 32, 175, [[SIZE]]] [1, 1, 1, 1]
    // CHECK-SAME:       to memref<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, offset = -9223372036854775808 : i64, order = #NCHW, strides = [2867200, 89600, 512, 1]}>
    // CHECK:        memref.copy [[SRC_SUBVIEW]], [[DST_SUBVIEW]]
    // CHECK:      }
    // Identity-layout accumulator copied into the NHWC-encoded function result.
    // CHECK:      [[RESULT:%.+]] = memref.alloc() {{.*}}memref<1x32x175x512xf16, {order = #NHWC}>
    // CHECK:      memref.copy [[ACC]], [[RESULT]]
    // CHECK:      return [[RESULT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// tensor.cast refines the dynamic tile down to a static slice; its result is fully static (no bounds),
// so the cast model must not carry stale bounds into the result type.
// CHECK-LABEL: @CastRefinesDynamicSliceToStatic
// CHECK-SAME:  [[ARG0:%.+]]: memref<1x32x175x512xf16, {order = #NHWC}>
func.func @CastRefinesDynamicSliceToStatic(%arg0: tensor<1x32x175x512xf16, {order = #NHWC}>) -> tensor<1x32x175x512xf16, {order = #NHWC}> {
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x32x175x512xf16, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c512 step %c16 iter_args(%arg2 = %0) -> (tensor<1x32x175x512xf16, {order = #NHWC}>) {
        %2 = affine.min affine_map<(d0) -> (-d0 + 512, 16)>(%arg1)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 32, 175, %2] [1, 1, 1, 1]
            : tensor<1x32x175x512xf16, {order = #NHWC}>
           to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
        %cast = tensor.cast %extracted_slice
            : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
           to tensor<1x32x175x16xf16, {order = #NHWC}>
        %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, 0, %arg1] [1, 32, 175, 16] [1, 1, 1, 1]
            : tensor<1x32x175x16xf16, {order = #NHWC}>
           into tensor<1x32x175x512xf16, {order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x32x175x512xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x32x175x512xf16, {order = #NHWC}>

    // CHECK:      [[ACC:%.+]] = memref.alloc() {{.*}}memref<1x32x175x512xf16>
    // CHECK:      scf.for [[IV:%.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
    // CHECK:        [[SRC_SUBVIEW:%.+]] = memref.subview [[ARG0]][0, 0, 0, [[IV]]] [1, 32, 175, %{{.+}}] [1, 1, 1, 1]
    // CHECK-SAME:       to memref<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, offset = -9223372036854775808 : i64, order = #NHWC, strides = [2867200, 1, 16384, 32]}>
    // memref.cast to a fully-static result; bounds absence is asserted by the exact type match below.
    // CHECK:        [[CAST:%.+]] = memref.cast [[SRC_SUBVIEW]]
    // CHECK-SAME:       to memref<1x32x175x16xf16, {offset = -9223372036854775808 : i64, order = #NHWC, strides = [2867200, 1, 16384, 32]}>
    // CHECK:        [[DST_SUBVIEW:%.+]] = memref.subview [[ACC]][0, 0, 0, [[IV]]] [1, 32, 175, 16] [1, 1, 1, 1]
    // CHECK-SAME:       to memref<1x32x175x16xf16, strided<[2867200, 89600, 512, 1], offset: ?>>
    // CHECK:        memref.copy [[CAST]], [[DST_SUBVIEW]]
    // CHECK:      }
    // Identity-layout accumulator copied into the NHWC-encoded function result.
    // CHECK:      [[RESULT:%.+]] = memref.alloc() {{.*}}memref<1x32x175x512xf16, {order = #NHWC}>
    // CHECK:      memref.copy [[ACC]], [[RESULT]]
    // CHECK:      return [[RESULT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// tensor.empty (dynamic + bounds) seeding an insert_slice accumulator. empty has no custom model: it
// lowers via upstream EmptyOpInterface to memref.alloc(%sz); a final alloc + copy makes the result.
// CHECK-LABEL: @EmptyDynamicThenInsertSlice
// CHECK-SAME:  [[ARG0:%.+]]: memref<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
// CHECK-SAME:  [[SZ:%.+]]: index
func.func @EmptyDynamicThenInsertSlice(
        %arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>,
        %sz: index)
        -> tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %0 = tensor.empty(%sz) : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    %inserted = tensor.insert_slice %arg0 into %0[0, 0, 0, %c0] [1, 32, 175, %sz] [1, 1, 1, 1]
        : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
       into tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    return %inserted : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:      [[ALLOC:%.+]] = memref.alloc([[SZ]]) {{.*}}memref<1x32x175x?xf16>
    // CHECK:      [[SUBVIEW:%.+]] = memref.subview [[ALLOC]][0, 0, 0, 0] [1, 32, 175, [[SZ]]] [1, 1, 1, 1]
    // CHECK-SAME:       to memref<1x32x175x?xf16, strided<[?, ?, ?, 1]>>
    // CHECK:      memref.copy [[ARG0]], [[SUBVIEW]]
    // CHECK:      [[RESULT:%.+]] = memref.alloc([[SZ]]) {{.*}}memref<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:      memref.copy [[ALLOC]], [[RESULT]]
    // CHECK:      return [[RESULT]]
}
