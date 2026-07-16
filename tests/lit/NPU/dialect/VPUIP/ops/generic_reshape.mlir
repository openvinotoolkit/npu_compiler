//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize --verify-diagnostics %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Fold
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x16x16xf32, {order = #NHWC}>)
func.func @Fold(%arg0: memref<1x3x16x16xf32, {order = #NHWC}>) -> memref<1x3x16x16xf32, {order = #NHWC}> {
    %0 = const.Declare memref<1x3x16x16xf32, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x3x16x16xf32>, [#const.Reorder<#NHWC>]

    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x3x16x16xf32, {order = #NHWC}>) -> memref<1x3x16x16xf32, {order = #NHWC}>

    %2 = VPUIP.Copy
        inputs(%1 : memref<1x3x16x16xf32, {order = #NHWC}>)
        outputs(%arg0 : memref<1x3x16x16xf32, {order = #NHWC}>)
        -> memref<1x3x16x16xf32, {order = #NHWC}>

    return %2 : memref<1x3x16x16xf32, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x3x16x16xf32, {order = #NHWC}>
    // CHECK-SAME:       dense<1.000000e+00> : tensor<1x3x16x16xf32>, [#const.Reorder<#NHWC>]

    // CHECK:       [[VAR0:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x16x16xf32, {order = #NHWC}>) outputs([[ARG_0]] : memref<1x3x16x16xf32, {order = #NHWC}>)
    // CHECK:       return [[VAR0]] : memref<1x3x16x16xf32, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FuseGenericReshapes
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x16x2xf32>)
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
    // CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x3x16x2xf32>) outputs([[ARG_0]] : memref<1x3x16x2xf32>)
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
// CHECK-SAME: ([[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x48xf16, #NHWC, @CMX_NN
func.func @GenericReshapeDistributed(%arg0: !InputDistributed) -> !OutputDistributed {
    %0 = VPUIP.GenericReshape inputs(%arg0 : !InputDistributed) -> !OutputDistributed

    return %0 : !OutputDistributed

    // CHECK:       [[RES:%.+]] = VPUIP.GenericReshape
    // CHECK-SAME:          inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x48xf16, #NHWC, @CMX_NN
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 16, 48], [1, 16, 16, 48]], compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 20, 48], [1, 16, 18, 48]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x16x96xf16, #NHWC, @CMX_NN
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 10, 96], [1, 16, 9, 96]], compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 10, 96], [1, 16, 9, 96]], memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]

    // CHECK:       return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @GenericReshapeWithStrides4DTo4D
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x16x4x8xf32, {order = #NCHW, strides = [1024, 32, 8, 1]}>)
func.func @GenericReshapeWithStrides4DTo4D(%arg0: memref<1x16x4x8xf32, {order = #NCHW, strides = [1024, 32, 8, 1]}>) -> memref<16x4x4x2xf32, {order = #NCHW, strides = [32, 8, 2, 1]}> {
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x16x4x8xf32, {order = #NCHW, strides = [1024, 32, 8, 1]}>) -> memref<16x4x4x2xf32, {order = #NCHW, strides = [32, 8, 2, 1]}>
    return %0 : memref<16x4x4x2xf32, {order = #NCHW, strides = [32, 8, 2, 1]}>

    // CHECK:       [[VAR0:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]] : memref<1x16x4x8xf32, {order = #NCHW, strides = [1024, 32, 8, 1]}>) -> memref<16x4x4x2xf32, {order = #NCHW, strides = [32, 8, 2, 1]}>
    // CHECK:       return [[VAR0]] : memref<16x4x4x2xf32, {order = #NCHW, strides = [32, 8, 2, 1]}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @GenericReshapeWithStrides2DTo4D
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<2048x8192xf32, {order = #NC, strides = [16384, 1]}>)
func.func @GenericReshapeWithStrides2DTo4D(%arg0: memref<2048x8192xf32, {order = #NC, strides = [16384, 1]}>) -> memref<1x1x2048x8192xf32, {order = #NCHW, strides = [33554432, 33554432, 16384, 1]}> {
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<2048x8192xf32, {order = #NC, strides = [16384, 1]}>) -> memref<1x1x2048x8192xf32, {order = #NCHW, strides = [33554432, 33554432, 16384, 1]}>
    return %0 : memref<1x1x2048x8192xf32, {order = #NCHW, strides = [33554432, 33554432, 16384, 1]}>

    // CHECK:       [[VAR0:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]] : memref<2048x8192xf32, {order = #NC, strides = [16384, 1]}>) -> memref<1x1x2048x8192xf32, {order = #NCHW, strides = [33554432, 33554432, 16384, 1]}>
    // CHECK:       return [[VAR0]] : memref<1x1x2048x8192xf32, {order = #NCHW, strides = [33554432, 33554432, 16384, 1]}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @GenericReshapeWithStrides5DTo4D
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x1x16x1x32xf32, {order = #NCDHW, strides = [2048, 1024, 64, 32, 1]}>)
func.func @GenericReshapeWithStrides5DTo4D(%arg0: memref<1x1x16x1x32xf32, {order = #NCDHW, strides = [2048, 1024, 64, 32, 1]}>) -> memref<4x4x4x8xf32, {order = #NCHW, strides = [256, 64, 8, 1]}> {
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1x16x1x32xf32, {order = #NCDHW, strides = [2048, 1024, 64, 32, 1]}>) -> memref<4x4x4x8xf32, {order = #NCHW, strides = [256, 64, 8, 1]}>
    return %0 : memref<4x4x4x8xf32, {order = #NCHW, strides = [256, 64, 8, 1]}>

    // CHECK:       [[VAR0:%.+]] = VPUIP.GenericReshape inputs([[ARG_0]] : memref<1x1x16x1x32xf32, {order = #NCDHW, strides = [2048, 1024, 64, 32, 1]}>) -> memref<4x4x4x8xf32, {order = #NCHW, strides = [256, 64, 8, 1]}>
    // CHECK:       return [[VAR0]] : memref<4x4x4x8xf32, {order = #NCHW, strides = [256, 64, 8, 1]}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>

func.func @GenericReshapeStridesIncompatible2DSwapDims(%arg0: memref<16x32xf32, {order = #NC, strides = [512, 16]}>) -> memref<32x16xf32, {order = #NC, strides = [512, 1]}> {
// expected-error@+1 {{Incompatible strides between input}}
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<16x32xf32, {order = #NC, strides = [512, 16]}>) -> memref<32x16xf32, {order = #NC, strides = [512, 1]}>
    return %0 : memref<32x16xf32, {order = #NC, strides = [512, 1]}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @GenericReshapeStridesIncompatible3DTo3D(%arg0: memref<2x16x32xf32, {order = #CHW, strides = [2048, 64, 1]}>) -> memref<1x32x32xf32, {order = #CHW, strides = [2048, 64, 1]}> {
// expected-error@+1 {{Incompatible strides between input}}
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<2x16x32xf32, {order = #CHW, strides = [2048, 64, 1]}>) -> memref<1x32x32xf32, {order = #CHW, strides = [2048, 64, 1]}>
    return %0 : memref<1x32x32xf32, {order = #CHW, strides = [2048, 64, 1]}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>

func.func @GenericReshapeStridesIncompatibleCompactOutput(%arg0: memref<16x32xf32, {order = #NC, strides = [512, 16]}>) -> memref<32x16xf32, {order = #NC, strides = [32, 1]}> {
// expected-error@+1 {{Incompatible strides between input}}
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<16x32xf32, {order = #NC, strides = [512, 16]}>) -> memref<32x16xf32, {order = #NC, strides = [32, 1]}>
    return %0 : memref<32x16xf32, {order = #NC, strides = [32, 1]}>
}
