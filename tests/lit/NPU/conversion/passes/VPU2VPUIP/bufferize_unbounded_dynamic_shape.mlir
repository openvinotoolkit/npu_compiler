//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter" --one-shot-bufferize-VPU-to-VPUIP --split-input-file %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// An unbounded network result gets a VPURT.Alloc placeholder without bounds.

// CHECK-LABEL:  func.func @UnboundedNetworkResult
func.func @UnboundedNetworkResult(%input: tensor<1x1x1x?xf16>) -> tensor<1x1x1x?xf16> {
// CHECK-SAME:      ([[INPUT:%.+]]: memref<1x1x1x?xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x1x1x?xf16, strided<[?, ?, ?, ?], offset: ?>>

    %softmax = VPU.SoftMax(%input) {axisInd = 3} : tensor<1x1x1x?xf16> -> tensor<1x1x1x?xf16>
// CHECK:       [[OUT_BUF:%.+]] = VPURT.Alloc -> memref<1x1x1x?xf16>
// CHECK-NOT:       bounds
// CHECK:       [[SOFTMAX:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_SoftMax
// CHECK-SAME:      inputs([[INPUT]]
// CHECK-SAME:      outputs([[OUT_BUF]]

    return %softmax : tensor<1x1x1x?xf16>
// CHECK:       [[RESULT:%.+]] = Core.ReinterpretCast([[SOFTMAX]]) : memref<1x1x1x?xf16> -> memref<1x1x1x?xf16, strided<[?, ?, ?, ?], offset: ?>>
// CHECK:       return [[RESULT]]
}

// -----

// Bufferization cannot tell which unbounded result becomes a network output, so every one gets a placeholder.
// This intermediate is never replaced, and the DDR StaticAllocation rejects it.

// CHECK-LABEL:  func.func @UnboundedIntermediate
func.func @UnboundedIntermediate(%input: tensor<1x1x1x?xf16>) -> tensor<1x1x1x?xf16> {
// CHECK-SAME:      ([[INPUT:%.+]]: memref<1x1x1x?xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x1x1x?xf16, strided<[?, ?, ?, ?], offset: ?>>

    %first = VPU.SoftMax(%input) {axisInd = 3} : tensor<1x1x1x?xf16> -> tensor<1x1x1x?xf16>
    %second = VPU.SoftMax(%first) {axisInd = 3} : tensor<1x1x1x?xf16> -> tensor<1x1x1x?xf16>
// CHECK:       [[FIRST_BUF:%.+]] = VPURT.Alloc -> memref<1x1x1x?xf16>
// CHECK:       [[FIRST:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      inputs([[INPUT]]
// CHECK-SAME:      outputs([[FIRST_BUF]]
// CHECK:       [[SECOND_BUF:%.+]] = VPURT.Alloc -> memref<1x1x1x?xf16>
// CHECK:       [[SECOND:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      inputs([[FIRST]]
// CHECK-SAME:      outputs([[SECOND_BUF]]

    return %second : tensor<1x1x1x?xf16>
// CHECK:       [[RESULT:%.+]] = Core.ReinterpretCast([[SECOND]]) : memref<1x1x1x?xf16> -> memref<1x1x1x?xf16, strided<[?, ?, ?, ?], offset: ?>>
// CHECK:       return [[RESULT]]
}
