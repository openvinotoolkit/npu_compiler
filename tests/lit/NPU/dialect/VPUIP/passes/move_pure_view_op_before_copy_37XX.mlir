//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --move-pure-view-op-before-copy %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x239x18xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @MoveShapeCastWithAlignmentBeforeTilingCopySegmented
// CHECK-SAME: ([[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x239x18xf16, #NHWC, @CMX_NN
func.func @MoveShapeCastWithAlignmentBeforeTilingCopySegmented(%arg0: !InputDistributed) -> memref<1x16x478x9xf16, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x16x239x18xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x239x18xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x239x18xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.ShapeCast {shape = [1, 16, 478, 9]} inputs(%0 : memref<1x16x239x18xf16, {order = #NHWC}, @DDR>) -> memref<1x16x478x9xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x16x478x9xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 478, 9]} inputs([[ARG_0]] : !VPUIP.DistributedBuffer<1x16x239x18xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x478x9xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>
    // CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x478x9xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SHAPECAST]] : !VPUIP.DistributedBuffer<1x16x478x9xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>)
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x16x478x9xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x478x9xf16, {order = #NHWC}, @DDR>
    // CHECK:    return [[COPY]] : memref<1x16x478x9xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x16x8xf16, #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
    }
>

// CHECK-LABEL: @DoNotMoveShapeCastBeforeTilingCopySegmentedDueToAlignment
// CHECK-SAME: ([[ARG_0:%[^:]+]]: !VPUIP.DistributedBuffer<1x16x16x8xf16, #NHWC, @CMX_NN,
func.func @DoNotMoveShapeCastBeforeTilingCopySegmentedDueToAlignment(%arg0: !InputDistributed) -> memref<1x1024x2x1xf16, {order = #NHWC}, @DDR> {
    %out = memref.alloc() : memref<1x16x16x8xf16, {order = #NHWC}, @DDR>
    %0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%out : memref<1x16x16x8xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x16x8xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.ShapeCast {shape = [1, 1024, 2, 1]} inputs(%0 : memref<1x16x16x8xf16, {order = #NHWC}, @DDR>) -> memref<1x1024x2x1xf16, {order = #NHWC}, @DDR>

    return %1 : memref<1x1024x2x1xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[OUTBUFF:%.+]] = memref.alloc() : memref<1x16x16x8xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ARG_0]]
    // CHECK-SAME:     outputs([[OUTBUFF]] : memref<1x16x16x8xf16, {order = #NHWC}, @DDR>)  ->  memref<1x16x16x8xf16, {order = #NHWC}, @DDR>
    // CHECK:    [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 1024, 2, 1]} inputs([[COPY]] : memref<1x16x16x8xf16, {order = #NHWC}, @DDR>) -> memref<1x1024x2x1xf16, {order = #NHWC}, @DDR>
    // CHECK:    return [[SHAPECAST]] : memref<1x1024x2x1xf16, {order = #NHWC}, @DDR>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @ChangeForStridedCopy
// CHECK-SAME: ([[INPUT:%.+]]: memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>)
func.func @ChangeForStridedCopy(
        %arg0: memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) -> memref<1x1x8x8x!qElemType, {order = #NHWC}, @DDR> {

    %0 = memref.alloc() : memref<1x16x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>)
        outputs(%0 : memref<1x16x4xui8, @DDR>)
        -> memref<1x16x4xui8, @DDR>

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x16x4xui8, @DDR>) -> memref<1x1x16x4xui8, @DDR>

    %3 = VPUIP.QuantizeCast inputs(%2 : memref<1x1x16x4xui8, @DDR>) -> memref<1x1x16x4x!qElemType, @DDR>

    %4 = VPUIP.ShapeCast {shape = [1, 1, 8, 8]} inputs(%3 : memref<1x1x16x4x!qElemType, @DDR>) -> memref<1x1x8x8x!qElemType, @DDR>

    %5 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
        inputs(%4 : memref<1x1x8x8x!qElemType, @DDR>)
        -> memref<1x1x8x8x!qElemType, {order = #NHWC}, @DDR>

    return %5 : memref<1x1x8x8x!qElemType, {order = #NHWC}, @DDR>

    // CHECK:    [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[INPUT]] : memref<1x16x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) -> memref<1x1x16x4xui8, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>
    // CHECK:    [[QUANT:%.+]] = VPUIP.QuantizeCast inputs([[RESHAPE]] : memref<1x1x16x4xui8, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>) -> memref<1x1x16x4x!qElemType, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>
    // CHECK:    [[SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 1, 8, 8]} inputs([[QUANT]] : memref<1x1x16x4x!qElemType, {order = #NCHW, strides = [128, 64, 4, 1]}, @DDR>) -> memref<1x1x8x8x!qElemType, {order = #NCHW, strides = [128, 64, 8, 1]}, @DDR>
    // CHECK:    [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[SHAPECAST]] : memref<1x1x8x8x!qElemType, {order = #NCHW, strides = [128, 64, 8, 1]}, @DDR>) -> memref<1x1x8x8x!qElemType, {order = #NHWC, strides = [128, 1, 8, 1]}, @DDR>
    // CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x1x8x8x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy inputs([[PERMUTE]] : memref<1x1x8x8x!qElemType, {order = #NHWC, strides = [128, 1, 8, 1]}, @DDR>) outputs([[ALLOC]] : memref<1x1x8x8x!qElemType, {order = #NHWC}, @DDR>) -> memref<1x1x8x8x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:    return [[COPY]] : memref<1x1x8x8x!qElemType, {order = #NHWC}, @DDR>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @ChangeWithNotEffectiveStridesGenericReshape
func.func @ChangeWithNotEffectiveStridesGenericReshape(
    %arg0: memref<1x8x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) -> memref<1x1x32xui8, @DDR> {
    %0 = memref.alloc() : memref<1x8x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) outputs(%0 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4xui8, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x8x4xui8, @DDR>) -> memref<1x1x32xui8, @DDR>
    return %2 : memref<1x1x32xui8, @DDR>
    // CHECK:   VPUIP.GenericReshape
    // CHECK:   VPUIP.Copy
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @NoChangeWithEffectiveStridesGenericReshape
func.func @NoChangeWithEffectiveStridesGenericReshape(
    %arg0: memref<1x8x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) -> memref<1x1x32xui8, @DDR> {
    %0 = memref.alloc() : memref<1x8x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) outputs(%0 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4xui8, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x8x4xui8, @DDR>) -> memref<1x1x32xui8, @DDR>
    return %2 : memref<1x1x32xui8, @DDR>
    // CHECK:   VPUIP.Copy
    // CHECK:   VPUIP.GenericReshape
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @ChangeWithNotEffectiveStridesPermuteCast
func.func @ChangeWithNotEffectiveStridesPermuteCast(
    %arg0: memref<1x8x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) -> memref<1x8x4xui8, {order = #map}, @DDR> {
    %0 = memref.alloc() : memref<1x8x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) outputs(%0 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4xui8, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #map, mem_perm = #map} inputs(%1 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4xui8, {order = #map}, @DDR>
    return %2 : memref<1x8x4xui8, {order = #map}, @DDR>
    // CHECK:   VPUIP.PermuteCast
    // CHECK:   VPUIP.Copy
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @NoChangeWithEffectiveStridesPermuteCast
func.func @NoChangeWithEffectiveStridesPermuteCast(
    %arg0: memref<1x8x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) -> memref<1x8x4xui8, {order = #map}, @DDR> {
    %0 = memref.alloc() : memref<1x8x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) outputs(%0 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4xui8, @DDR>
    %2 = VPUIP.PermuteCast {dst_order = #map, mem_perm = #map} inputs(%1 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4xui8, {order = #map}, @DDR>
    return %2 : memref<1x8x4xui8, {order = #map}, @DDR>
    // CHECK:   VPUIP.Copy
    // CHECK:   VPUIP.PermuteCast
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @ChangeWithNotEffectiveStridesQuantizeCast
func.func @ChangeWithNotEffectiveStridesQuantizeCast(
    %arg0: memref<1x8x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) -> memref<1x8x4x!qElemType, @DDR> {
    %0 = memref.alloc() : memref<1x8x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) outputs(%0 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4xui8, @DDR>
    %2 = VPUIP.QuantizeCast inputs(%1 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4x!qElemType, @DDR>
    return %2 : memref<1x8x4x!qElemType, @DDR>
    // CHECK:   VPUIP.QuantizeCast
    // CHECK:   VPUIP.Copy
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @ChangeWithEffectiveStridesQuantizeCast
func.func @ChangeWithEffectiveStridesQuantizeCast(
    %arg0: memref<1x8x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) -> memref<1x8x4x!qElemType, @DDR> {
    %0 = memref.alloc() : memref<1x8x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) outputs(%0 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4xui8, @DDR>
    %2 = VPUIP.QuantizeCast inputs(%1 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4x!qElemType, @DDR>
    return %2 : memref<1x8x4x!qElemType, @DDR>
    // CHECK:   VPUIP.QuantizeCast
    // CHECK:   VPUIP.Copy
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @ChangeWithNotEffectiveStridesShapeCast
func.func @ChangeWithNotEffectiveStridesShapeCast(
    %arg0: memref<1x8x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) -> memref<1x32x1xui8, @DDR> {
    %0 = memref.alloc() : memref<1x8x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x4xui8, {order = #CHW, strides = [128, 4, 1]}, @DDR>) outputs(%0 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4xui8, @DDR>
    %2 = VPUIP.ShapeCast {shape = [1, 32, 1]} inputs(%1 : memref<1x8x4xui8, @DDR>) -> memref<1x32x1xui8, @DDR>
    return %2 : memref<1x32x1xui8, @DDR>
    // CHECK:   VPUIP.ShapeCast
    // CHECK:   VPUIP.Copy
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @NoChangeWithEffectiveStridesShapeCast
func.func @NoChangeWithEffectiveStridesShapeCast(
    %arg0: memref<1x8x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) -> memref<1x32x1xui8, @DDR> {
    %0 = memref.alloc() : memref<1x8x4xui8, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x4xui8, {order = #CHW, strides = [256, 8, 1]}, @DDR>) outputs(%0 : memref<1x8x4xui8, @DDR>) -> memref<1x8x4xui8, @DDR>
    %2 = VPUIP.ShapeCast {shape = [1, 32, 1]} inputs(%1 : memref<1x8x4xui8, @DDR>) -> memref<1x32x1xui8, @DDR>
    return %2 : memref<1x32x1xui8, @DDR>
    // CHECK:   VPUIP.Copy
    // CHECK:   VPUIP.ShapeCast
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NoChangeForStridedSliceWithShapeCast
// CHECK-SAME: ([[INPUT:%.+]]: memref<7x4x16x5xf16, @DDR>)
func.func @NoChangeForStridedSliceWithShapeCast(
        %arg0: memref<7x4x16x5xf16, @DDR>) -> memref<2x7x6x25xf16, @DDR> {

    %subview = VPUIP.SubView %arg0 [0, 0, 0, 0] [7, 4, 15, 5] : memref<7x4x16x5xf16, @DDR> to memref<7x4x15x5xf16, {order = #NCHW, strides = [320, 80, 5, 1]}, @DDR>

    %0 = memref.alloc() : memref<7x4x15x5xf16, @DDR>
    %1 = VPUIP.Copy inputs(%subview : memref<7x4x15x5xf16, {order = #NCHW, strides = [320, 80, 5, 1]}, @DDR>)
        outputs(%0 : memref<7x4x15x5xf16, @DDR>)
        -> memref<7x4x15x5xf16, @DDR>

    %2 = VPUIP.ShapeCast {shape = [2, 7, 6, 25]} inputs(%1 : memref<7x4x15x5xf16, @DDR>) -> memref<2x7x6x25xf16, @DDR>

    return %2 : memref<2x7x6x25xf16, @DDR>

    // CHECK:   VPUIP.SubView
    // CHECK:   VPUIP.Copy
    // CHECK:   VPUIP.ShapeCast
}
