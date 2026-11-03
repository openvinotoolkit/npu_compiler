//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Pins a known gap: %dst is both the insert_slice destination and a second function result, so a
// conflict-correct bufferization must copy it before the in-place write. --lower-VPU-to-VPUIP does not
// schedule the in-place analysis and inserts no copies, so the CHECK below cannot pass yet.
//
// Held out with UNSUPPORTED (not XFAIL): XFAIL gates on the environment, so it would go green for any run
// failure, not the missing-copy gap; asserting the current in-place behavior would paint the miscompile
// green. Re-enable once analysis-driven conflict copies land: drop UNSUPPORTED and tighten the checks.
// E#235075 Add DPS conflict handling to one-shot-bufferize-VPU-to-VPUIP

// UNSUPPORTED: true
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --lower-VPU-to-VPUIP %s | FileCheck %s

// CHECK-LABEL: @InsertSliceDestReadAfterWrite
func.func @InsertSliceDestReadAfterWrite(
        %src: tensor<1x32x175x16xf16>,
        %dst: tensor<1x32x175x512xf16>)
        -> (tensor<1x32x175x512xf16>, tensor<1x32x175x512xf16>) {
    %inserted = tensor.insert_slice %src into %dst[0, 0, 0, 0] [1, 32, 175, 16] [1, 1, 1, 1]
        : tensor<1x32x175x16xf16> into tensor<1x32x175x512xf16>
    // %dst is read (returned) after insert_slice wrote it in place -> genuine out-of-place conflict.
    return %inserted, %dst : tensor<1x32x175x512xf16>, tensor<1x32x175x512xf16>

    // Conflict-correct bufferization copies the destination into a fresh buffer before the in-place slice
    // write, so the second (unmodified) result does not alias the mutated buffer:
    // CHECK:      [[DST_COPY:%.+]] = memref.alloc() {{.*}}memref<1x32x175x512xf16>
    // CHECK:      memref.copy %arg1, [[DST_COPY]]
    // CHECK:      [[SUBVIEW:%.+]] = memref.subview [[DST_COPY]]
    // CHECK:      memref.copy %arg0, [[SUBVIEW]]
}
