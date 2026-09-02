//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @Fold
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x16x16x!qElemType>)
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
    // CHECK-SAME:       outputs([[ARG_0]] : memref<1x3x16x16x!qElemType>) -> memref<1x3x16x16x!qElemType>
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
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x3x2x16x!qElemType>
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
    // CHECK:   [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : memref<1x3x2x16x!qElemType>) outputs([[ARG_0]] : memref<1x3x2x16x!qElemType>) -> memref<1x3x2x16x!qElemType>
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
// CHECK-SAME: ([[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x32x48x!qElemType, #NHWC, @CMX_NN,
func.func @QuantizeCastDistributed(%arg0: !InputDistributed) -> !OutputDistributed {
    %0 = VPUIP.QuantizeCast inputs(%arg0 : !InputDistributed) -> !OutputDistributed

    return %0 : !OutputDistributed

    // CHECK:       [[RES:%.+]] = VPUIP.QuantizeCast
    // CHECK-SAME:          inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x32x48x!qElemType, #NHWC, @CMX_NN
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 16, 48], [1, 16, 16, 48]], compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 20, 48], [1, 16, 18, 48]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x32x48x!qElemType1, #NHWC, @CMX_NN
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 16, 48], [1, 16, 16, 48]], compute_offsets = [[0, 0, 0, 0], [0, 0, 16, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 20, 48], [1, 16, 18, 48]], memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0]]

    // CHECK:       return [[RES]]
}

// -----

!quantileType = !QuantileType.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
!qElemType = !quant.uniform<!QuantileType.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 0.07874348958333334>

// CHECK-LABEL: @QuantizeCastNF4
// CHECK-SAME:  [[INPUT0:%.+]]: memref<16x32x1x1x!QuantileType.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
// CHECK-SAME:  [[INPUT1:%.+]]: memref<16x32x1x1x!qElemType>
func.func @QuantizeCastNF4(%arg0: memref<16x32x1x1x!quantileType>, %arg1: memref<16x32x1x1x!qElemType>) -> (memref<16x32x1x1x!qElemType>, memref<16x32x1x1x!quantileType>) {
    %0 = VPUIP.QuantizeCast inputs(%arg0 : memref<16x32x1x1x!quantileType>) -> memref<16x32x1x1x!qElemType>
    %1 = VPUIP.QuantizeCast inputs(%arg1 : memref<16x32x1x1x!qElemType>) -> memref<16x32x1x1x!quantileType>

    return %0, %1 : memref<16x32x1x1x!qElemType>, memref<16x32x1x1x!quantileType>

    // CHECK:       [[FIRST_QUANT_CAST:%.+]] = VPUIP.QuantizeCast inputs([[INPUT0]] : memref<16x32x1x1x!QuantileType.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>) -> memref<16x32x1x1x!qElemType>
    // CHECK:       [[SECOND_QUANT_CAST:%.+]] = VPUIP.QuantizeCast inputs([[INPUT1]] : memref<16x32x1x1x!qElemType>) -> memref<16x32x1x1x!QuantileType.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
    // CHECK:       return [[FIRST_QUANT_CAST]], [[SECOND_QUANT_CAST]]
}
