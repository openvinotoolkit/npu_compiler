//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-expand="defer-to-expand-dma=true" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Coverage for `--convert-expand=defer-to-expand-dma=true`. The only case that diverges from
// the default `defer-to-expand-dma=false` is integral-storage Expand with all-zero `pads_begin`:
// the default decomposes it here to the shared-zero-constant Copy+ConcatView pattern, this
// option leaves it as `VPUIP.Expand` for downstream ConvertToDMA -> VPUIP.ExpandDMA.

!qElemType = !quant.uniform<si8:f16, 0.005>
!qElemType1 = !quant.uniform<ui8:f16, 0.0038725490663565842>

// CHECK-LABEL: func.func @MixedSignedAndUnsignedQuantizedExpandDeferred
// CHECK-SAME: ([[ARG_I8:%[^:]+]]: memref<1x3x4x4x!qElemType>, [[ARG_U8:%[^:]+]]: memref<1x3x6x6x!qElemType1>)
func.func @MixedSignedAndUnsignedQuantizedExpandDeferred(
        %arg0: memref<1x3x4x4x!qElemType>,
        %arg1: memref<1x3x6x6x!qElemType1>) ->
        (memref<1x4x4x4x!qElemType>, memref<1x4x6x6x!qElemType1>) {
    %alloc_i8 = memref.alloc() : memref<1x4x4x4x!qElemType>
    %alloc_u8 = memref.alloc() : memref<1x4x6x6x!qElemType1>

    %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} inputs(%arg0 : memref<1x3x4x4x!qElemType>) outputs(%alloc_i8 : memref<1x4x4x4x!qElemType>) -> memref<1x4x4x4x!qElemType>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} inputs(%arg1 : memref<1x3x6x6x!qElemType1>) outputs(%alloc_u8 : memref<1x4x6x6x!qElemType1>) -> memref<1x4x6x6x!qElemType1>

    return %0, %1 : memref<1x4x4x4x!qElemType>, memref<1x4x6x6x!qElemType1>

    // CHECK-NOT:   VPUIP.ConcatView
    // CHECK:       [[OUT_I8:%.+]] = VPUIP.Expand
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]
    // CHECK-SAME:      inputs([[ARG_I8]] : memref<1x3x4x4x!qElemType>)
    // CHECK:       [[OUT_U8:%.+]] = VPUIP.Expand
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]
    // CHECK-SAME:      inputs([[ARG_U8]] : memref<1x3x6x6x!qElemType1>)
    // CHECK:       return [[OUT_I8]], [[OUT_U8]]
}

// -----

// FP16 is always decomposed here regardless of `lower-to-expand-dma`, because ConvertToDMA
// does not handle FP16 Expand.

// CHECK-LABEL: func.func @F16ExpandStillDecomposed
// CHECK-SAME: ([[ARG_F16:%[^:]+]]: memref<1x3x4x4xf16>)
func.func @F16ExpandStillDecomposed(%arg0: memref<1x3x4x4xf16>) -> memref<1x8x4x4xf16> {
    %0 = memref.alloc() : memref<1x8x4x4xf16>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} inputs(%arg0 : memref<1x3x4x4xf16>) outputs(%0 : memref<1x8x4x4xf16>) -> memref<1x8x4x4xf16>
    return %1 : memref<1x8x4x4xf16>

    // CHECK-NOT:   VPUIP.Expand
    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       return [[CONCAT]]
}

// -----

// Non-zero `pads_begin` is always decomposed here regardless of `lower-to-expand-dma`,
// because the ExpandDMA descriptor generator only supports pads at the end.

!qElemType = !quant.uniform<u8:f16, 0.0038725490663565842>

// CHECK-LABEL: func.func @NonZeroPadsBeginStillDecomposed
// CHECK-SAME: ([[ARG:%[^:]+]]: memref<1x3x4x4x!qElemType>)
func.func @NonZeroPadsBeginStillDecomposed(%arg0: memref<1x3x4x4x!qElemType>) -> memref<1x5x4x4x!qElemType> {
    %0 = memref.alloc() : memref<1x5x4x4x!qElemType>
    %1 = VPUIP.Expand {pads_begin = [0, 1, 0, 0], pads_end = [0, 1, 0, 0]} inputs(%arg0 : memref<1x3x4x4x!qElemType>) outputs(%0 : memref<1x5x4x4x!qElemType>) -> memref<1x5x4x4x!qElemType>
    return %1 : memref<1x5x4x4x!qElemType>

    // CHECK-NOT:   VPUIP.Expand
    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       return [[CONCAT]]
}
