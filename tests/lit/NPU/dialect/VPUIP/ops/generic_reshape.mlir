//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Fold
func.func @Fold(%arg0: memref<1x3x16x16xf32, #NHWC>) -> memref<1x3x16x16xf32, #NHWC> {
    %0 = const.Declare memref<1x3x16x16xf32, #NHWC> =
        dense<1.000000e+00> : tensor<1x3x16x16xf32>, [#const.Reorder<#NHWC>]

    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x3x16x16xf32, #NHWC>) -> memref<1x3x16x16xf32, #NHWC>

    %2 = VPUIP.Copy
        inputs(%1 : memref<1x3x16x16xf32, #NHWC>)
        outputs(%arg0 : memref<1x3x16x16xf32, #NHWC>)
        -> memref<1x3x16x16xf32, #NHWC>

    return %2 : memref<1x3x16x16xf32, #NHWC>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x3x16x16xf32, #NHWC>
    // CHECK-SAME:       dense<1.000000e+00> : tensor<1x3x16x16xf32>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAR0:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x16x16xf32, #NHWC>) outputs(%arg0 : memref<1x3x16x16xf32, #NHWC>)
    // CHECK:       return [[VAR0]] : memref<1x3x16x16xf32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FuseGenericReshapes
func.func @FuseGenericReshapes(%arg0: memref<1x3x16x2xf32>) -> memref<1x3x16x2xf32> {
    %0 = const.Declare memref<1x3x2x16xf32> =
        dense<1.000000e+00> : tensor<1x3x2x16xf32>

    %1 = memref.alloc() : memref<1x3x2x16xf32>
    %2 = VPUIP.Copy
        inputs(%0 : memref<1x3x2x16xf32>)
        outputs(%1 : memref<1x3x2x16xf32>)
        -> memref<1x3x2x16xf32>

    %3 = VPUIP.GenericReshape inputs(%2 : memref<1x3x2x16xf32>) -> memref<1x3x4x8xf32>
    %4 = VPUIP.GenericReshape inputs(%3 : memref<1x3x4x8xf32>) -> memref<1x3x16x2xf32>

    %5 = VPUIP.Copy
        inputs(%4 : memref<1x3x16x2xf32>)
        outputs(%arg0 : memref<1x3x16x2xf32>)
        -> memref<1x3x16x2xf32>

    return %5 : memref<1x3x16x2xf32>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x3x2x16xf32> = dense<1.000000e+00> : tensor<1x3x2x16xf32>
    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x3x2x16xf32>
    // CHECK:       [[VAR1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x2x16xf32>) outputs([[VAR0]] : memref<1x3x2x16xf32>)
    // CHECK:       [[VAR2:%.+]] = VPUIP.GenericReshape inputs([[VAR1]] : memref<1x3x2x16xf32>) -> memref<1x3x16x2xf32>
    // CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x3x16x2xf32>) outputs(%arg0 : memref<1x3x16x2xf32>)
    // CHECK:       return [[VAR3]] : memref<1x3x16x2xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x48xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 16, 48], [1, 16, 16, 48]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    memory_shapes = [[1, 16, 20, 48], [1, 16, 18, 48]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x16x96xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 10, 96], [1, 16, 9, 96]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    memory_shapes = [[1, 16, 10, 96], [1, 16, 9, 96]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]
}>

// CHECK-LABEL: @GenericReshapeDistributed
func.func @GenericReshapeDistributed(%arg0: !InputDistributed) -> !OutputDistributed {
    %0 = VPUIP.GenericReshape inputs(%arg0 : !InputDistributed) -> !OutputDistributed

    return %0 : !OutputDistributed

    // CHECK:       [[RES:%.+]] = VPUIP.GenericReshape
    // CHECK-SAME:          inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x48xf16, #NHWC, @CMX_NN
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 16, 48], [1, 16, 16, 48]], compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 20, 48], [1, 16, 18, 48]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x16x96xf16, #NHWC, @CMX_NN
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 10, 96], [1, 16, 9, 96]], compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 10, 96], [1, 16, 9, 96]], memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]

    // CHECK:       return [[RES]]
}
