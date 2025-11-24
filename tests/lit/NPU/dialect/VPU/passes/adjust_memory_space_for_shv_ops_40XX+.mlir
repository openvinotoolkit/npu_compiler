//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-memory-space-for-shv-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// Only conversions that cannot be done via DMA are lowered to SHAVE. This DMA feature is only supported starting with NPU4

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @SkipNonSHAVEConvert
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x3x4x4xf32>)
func.func @SkipNonSHAVEConvert(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x4x4xf32> -> tensor<1x3x4x4xf16>
    return %0 : tensor<1x3x4x4xf16>

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[INPUT_DDR]])
    // CHECK-SAME:    -> tensor<1x3x4x4xf16>
    // CHECK:       return [[CONVERT]]
}
