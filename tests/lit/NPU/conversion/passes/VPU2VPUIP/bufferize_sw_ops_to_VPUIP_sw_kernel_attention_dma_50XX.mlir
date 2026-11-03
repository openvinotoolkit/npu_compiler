//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU5010

// No KV cache, no quantized K/V, matches the one statically-legal non-flash shape
// (isLegalNonFlashAttentionDMAConfig) and fits CMX: getKernelImpl picks the plain
// (non-flash) attention_dma kernel.

// CHECK-LABEL: func.func nested @builtin_AttentionDMA
// CHECK-SAME:    attributes {VPU.kernel_code = "attention_dma.cpp", VPU.kernel_entry = "attention_dma", VPU.kernel_name = "attention_dma", VPU.task_type = @COMPUTE}

// CHECK-LABEL: @AttentionDMANonFlash
func.func @AttentionDMANonFlash(%q: tensor<1x12x3600x16xf16>, %k: tensor<1x12x3600x16xf16>, %v: tensor<1x12x16x3600xf16>) -> tensor<1x12x3600x16xf16> {
  %aux = VPU.Empty : tensor<1x1x1x1024xui8>
  %output = VPU.AttentionDMA(%q, %k, %v, %aux) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>} : tensor<1x12x3600x16xf16>, tensor<1x12x3600x16xf16>, tensor<1x12x16x3600xf16>, tensor<1x1x1x1024xui8> -> tensor<1x12x3600x16xf16>
  return %output : tensor<1x12x3600x16xf16>

  // CHECK: [[RES:%.+]]:2 = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_AttentionDMA
  // CHECK: return [[RES]]#0
}

// -----

// pastK/pastV present (hasKvCache): getKernelImpl always routes a KV cache to attention_dma_flash,
// regardless of alignment.

// CHECK-LABEL: func.func nested @builtin_AttentionDMA
// CHECK-SAME:    attributes {VPU.kernel_code = "attention_dma_flash.cpp", VPU.kernel_entry = "attention_dma_flash", VPU.kernel_name = "attention_dma", VPU.task_type = @COMPUTE}

// CHECK-LABEL: @AttentionDMAFlashKvCache
func.func @AttentionDMAFlashKvCache(%q: tensor<1x1x16x16xf16>, %k: tensor<1x1x16x16xf16>, %v: tensor<1x1x16x16xf16>, %pastK: tensor<1x1x16x16xf16>, %pastV: tensor<1x1x16x16xf16>) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>) {
  %aux = VPU.Empty : tensor<1x1x1x1024xui8>
  %output, %presentK, %presentV = VPU.AttentionDMA(%q, %k, %v, %pastK, %pastV, %aux) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0>} : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<1x1x1x1024xui8> -> tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>
  return %output, %presentK, %presentV : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>

  // CHECK: [[RES:%.+]]:4 = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_AttentionDMA
  // CHECK: return [[RES]]#0, [[RES]]#1, [[RES]]#2
}

// -----

// Quantized K (si8) with tSL == 1, a runtime seqLenK, and no mask/bias: getKernelImpl picks the
// decode-step attention_dma_flash_tsl1 kernel.

// CHECK-LABEL: func.func nested @builtin_AttentionDMA
// CHECK-SAME:    attributes {VPU.kernel_code = "attention_dma_flash_tsl1.cpp", VPU.kernel_entry = "attention_dma_flash_tsl1", VPU.kernel_name = "attention_dma", VPU.task_type = @COMPUTE}

// CHECK-LABEL: @AttentionDMAFlashQuantTsl1
func.func @AttentionDMAFlashQuantTsl1(%q: tensor<1x1x1x16xf16>, %k: tensor<1x1x16x16xsi8>, %v: tensor<1x1x16x16xf16>, %seqLenK: tensor<1x1xsi32>, %kScale: tensor<1x1x16x16xf16>) -> tensor<1x1x1x16xf16> {
  %aux = VPU.Empty : tensor<1x1x1x1024xui8>
  %output = VPU.AttentionDMA(%q, %k, %v, %aux, %seqLenK, %kScale) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>} : tensor<1x1x1x16xf16>, tensor<1x1x16x16xsi8>, tensor<1x1x16x16xf16>, tensor<1x1x1x1024xui8>, tensor<1x1xsi32>, tensor<1x1x16x16xf16> -> tensor<1x1x1x16xf16>
  return %output : tensor<1x1x1x16xf16>

  // CHECK: [[RES:%.+]]:2 = VPUIP.SW.Kernel {{.*}} @VPU.SW::@builtin_AttentionDMA
  // CHECK: return [[RES]]#0
}
