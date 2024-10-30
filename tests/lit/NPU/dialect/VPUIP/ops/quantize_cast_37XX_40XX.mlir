//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @Fold
func.func @Fold(%arg0: memref<1x3x16x16x!qElemType>) -> memref<1x3x16x16x!qElemType> {
    %0 = const.Declare memref<1x3x16x16x!qElemType> =
        dense<1.000000e+00> : tensor<1x3x16x16xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %1 = VPUIP.QuantizeCast
        inputs(%0 : memref<1x3x16x16x!qElemType>)
        -> memref<1x3x16x16x!qElemType>

    %2 = VPUIP.Copy
        inputs(%1 : memref<1x3x16x16x!qElemType>)
        outputs(%arg0 : memref<1x3x16x16x!qElemType>)
        -> memref<1x3x16x16x!qElemType>


    return %2 : memref<1x3x16x16x!qElemType>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare memref<1x3x16x16x!qElemType> =
    // CHECK-SAME:       dense<1.000000e+00> : tensor<1x3x16x16xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    // CHECK:       [[VAR0:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x16x16x!qElemType>)
    // CHECK-SAME:       outputs(%arg0 : memref<1x3x16x16x!qElemType>) -> memref<1x3x16x16x!qElemType>
    // CHECK:       return [[VAR0]] : memref<1x3x16x16x!qElemType>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType2 = !quant.uniform<u8:f16, 3.000000e+00>

// CHECK:  !qElemType = !quant.uniform<u8:f16, 3.000000e+00>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @FuseQuantizeCastOps
func.func @FuseQuantizeCastOps(%arg0: memref<1x3x2x16x!qElemType2>) -> memref<1x3x2x16x!qElemType2> {
    %0 = const.Declare memref<1x3x2x16x!qElemType> =
        dense<1.000000e+00> : tensor<1x3x2x16xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %1 = memref.alloc() : memref<1x3x2x16x!qElemType>
    %2 = VPUIP.Copy
        inputs(%0 : memref<1x3x2x16x!qElemType>)
        outputs(%1 : memref<1x3x2x16x!qElemType>)
        -> memref<1x3x2x16x!qElemType>

    %3 = VPUIP.QuantizeCast inputs(%2 : memref<1x3x2x16x!qElemType>) -> memref<1x3x2x16x!qElemType1>
    %4 = VPUIP.QuantizeCast inputs(%3 : memref<1x3x2x16x!qElemType1>) -> memref<1x3x2x16x!qElemType2>

    %5 = VPUIP.Copy
        inputs(%4 : memref<1x3x2x16x!qElemType2>)
        outputs(%arg0 : memref<1x3x2x16x!qElemType2>)
        -> memref<1x3x2x16x!qElemType2>

    return %5 : memref<1x3x2x16x!qElemType2>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<1x3x2x16x!qElemType1> = dense<1.000000e+00> : tensor<1x3x2x16xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x3x2x16x!qElemType1>
    // CHECK:   [[VAR1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x3x2x16x!qElemType1>) outputs([[VAR0]] : memref<1x3x2x16x!qElemType1>) -> memref<1x3x2x16x!qElemType1>
    // CHECK:   [[VAR2:%.+]] = VPUIP.QuantizeCast inputs([[VAR1]] : memref<1x3x2x16x!qElemType1>) -> memref<1x3x2x16x!qElemType>
    // CHECK:   [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x3x2x16x!qElemType>) outputs(%arg0 : memref<1x3x2x16x!qElemType>) -> memref<1x3x2x16x!qElemType>
    // CHECK:   return [[VAR3]] : memref<1x3x2x16x!qElemType>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 2.000000e+00>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x48x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 16, 48], [1, 16, 16, 48]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    memory_shapes = [[1, 16, 20, 48], [1, 16, 18, 48]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x48x!qElemType1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 16, 48], [1, 16, 16, 48]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    memory_shapes = [[1, 16, 20, 48], [1, 16, 18, 48]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]
}>

// CHECK-LABEL: @QuantizeCastDistributed
func.func @QuantizeCastDistributed(%arg0: !InputDistributed) -> !OutputDistributed {
    %0 = VPUIP.QuantizeCast inputs(%arg0 : !InputDistributed) -> !OutputDistributed

    return %0 : !OutputDistributed

    // CHECK:       [[RES:%.+]] = VPUIP.QuantizeCast
    // CHECK-SAME:          inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x32x48x!qElemType, #NHWC, @CMX_NN
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 16, 48], [1, 16, 16, 48]], compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 20, 48], [1, 16, 18, 48]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x32x48x!qElemType1, #NHWC, @CMX_NN
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 16, 48], [1, 16, 16, 48]], compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 20, 48], [1, 16, 18, 48]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]

    // CHECK:       return [[RES]]
}
