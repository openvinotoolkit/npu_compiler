//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU4000

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
    // CHECK: [[ALLOC_OUT:%.+]] = memref.alloc() : memref<1x3x4x4xf32>
    // CHECK: [[SW:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[INPUT]] as [[INNER_ARG0:[^:]+]]: memref<1x3x4x4xf16>) outputs([[ALLOC_OUT]] as [[INNER_ARG1:[^:]+]]: memref<1x3x4x4xf32>) on tile 0 -> memref<1x3x4x4xf32>{
    // CHECK:   VPUIP.SW.Kernel.run([[INNER_ARG0]], [[INNER_ARG1]]) : memref<1x3x4x4xf16>, memref<1x3x4x4xf32>
    // CHECK: }
    // CHECK: return [[SW]]
}

// -----
// CHECK: func.func nested @builtin_Convert(memref<*xsi16>, memref<*xf16>)
// CHECK-SAME: attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"
// CHECK-LABEL:  func.func @ConvertSI16ToFP16UsingSW
func.func @ConvertSI16ToFP16UsingSW(%arg0: tensor<1x3x4x4xsi16>) -> tensor<1x3x4x4xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x4x4xsi16> -> tensor<1x3x4x4xf16>
    return %0 : tensor<1x3x4x4xf16>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[ALLOC_OUT:%.+]] = memref.alloc() : memref<1x3x4x4xf16>
    // CHECK: [[SW:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME: @VPU.SW::@builtin_Convert
    // CHECK: return [[SW]]
}

// -----
// CHECK: func.func nested @builtin_Convert(memref<*xui16>, memref<*xf16>)
// CHECK-SAME: attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"
// CHECK-LABEL:  func.func @ConvertUI16ToFP16UsingSW
func.func @ConvertUI16ToFP16UsingSW(%arg0: tensor<1x3x4x4xui16>) -> tensor<1x3x4x4xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x4x4xui16> -> tensor<1x3x4x4xf16>
    return %0 : tensor<1x3x4x4xf16>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[ALLOC_OUT:%.+]] = memref.alloc() : memref<1x3x4x4xf16>
    // CHECK: [[SW:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME: @VPU.SW::@builtin_Convert
    // CHECK: return [[SW]]
}

// -----
// CHECK: func.func nested @builtin_Convert(memref<*xf16>, memref<*xsi16>)
// CHECK-SAME: attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"
// CHECK-LABEL:  func.func @ConvertFP16ToSI16UsingSW
func.func @ConvertFP16ToSI16UsingSW(%arg0: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4xsi16> {
    %0 = VPU.Convert(%arg0) {dstElemType = si16} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4xsi16>
    return %0 : tensor<1x3x4x4xsi16>

    // CHECK-NOT: VPU.Convert
    // CHECK: [[ALLOC_OUT:%.+]] = memref.alloc() : memref<1x3x4x4xsi16>
    // CHECK: [[SW:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME: @VPU.SW::@builtin_Convert
    // CHECK: return [[SW]]
}
