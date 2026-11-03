//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

func.func @NNDMAUnboundedInputRejected(%arg0: memref<1x1x1x?xf16, @DDR>, %arg1: memref<1x1x1x16xf16, @DDR>) -> memref<1x1x1x16xf16, @DDR> {
    // expected-error@+1 {{DMA cannot access 'memref<1x1x1x?xf16, @DDR>': a dimension has no upper bound}}
    %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x?xf16, @DDR>) outputs(%arg1 : memref<1x1x1x16xf16, @DDR>) -> memref<1x1x1x16xf16, @DDR>
    return %0 : memref<1x1x1x16xf16, @DDR>
}

// -----

func.func @NNDMAUnboundedOutputRejected(%arg0: memref<1x1x1x16xf16, @DDR>, %arg1: memref<1x1x1x?xf16, @DDR>) -> memref<1x1x1x?xf16, @DDR> {
    // expected-error@+1 {{DMA cannot access 'memref<1x1x1x?xf16, @DDR>': a dimension has no upper bound}}
    %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x1x1x16xf16, @DDR>) outputs(%arg1 : memref<1x1x1x?xf16, @DDR>) -> memref<1x1x1x?xf16, @DDR>
    return %0 : memref<1x1x1x?xf16, @DDR>
}
