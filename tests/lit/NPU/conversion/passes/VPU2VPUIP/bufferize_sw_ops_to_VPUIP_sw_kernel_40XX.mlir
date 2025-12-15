//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL:  func.func @ConvertFP32ToFP16
func.func @ConvertFP32ToFP16(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x4x4xf32> -> tensor<1x3x4x4xf16>
    return %0 : tensor<1x3x4x4xf16>

    // CHECK-NOT:   VPUIP.SW.Kernel
    // CHECK:       return
    // CHECK:       <1x3x4x4xf16>
}

// -----
// CHECK-LABEL:  func.func @ConvertFP16ToFP32UsingSW
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x3x4x4xf16>
func.func @ConvertFP16ToFP32UsingSW(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf32>
    return %0 : tensor<1x3x4x4xf32>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[ALLOC_OUT:%.*]] = memref.alloc() : memref<1x3x4x4xf32>
    // CHECK: [[SW:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[INPUT]] as [[INNER_ARG0:[^:]+]]: memref<1x3x4x4xf16>) outputs([[ALLOC_OUT]] as [[INNER_ARG1:[^:]+]]: memref<1x3x4x4xf32>) on tile 0 -> memref<1x3x4x4xf32>{
    // CHECK:   VPUIP.SW.Kernel.run([[INNER_ARG0]], [[INNER_ARG1]]) : memref<1x3x4x4xf16>, memref<1x3x4x4xf32>
    // CHECK: }
    // CHECK: return [[SW]]
}
