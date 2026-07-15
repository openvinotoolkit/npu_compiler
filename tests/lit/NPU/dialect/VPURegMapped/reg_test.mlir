//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

  func.func nested @MLIR_VPURegMapped_CreateDpuVariantRegisterAllowOverlapTrue() {
    VPURegMapped.RegisterWrapper regAttr(<regForTest offset 12 size 32 allowOverlap {
      UINT test_1 at 0 size 8 = 0xFF,
      SINT test_2 at 8 size 8 = 0xFF
    }>)
    return
  }

// CHECK:      VPURegMapped.RegisterWrapper regAttr(<regForTest offset 12 size 32 allowOverlap {
// CHECK-NEXT:   UINT test_1 at 0 size 8 = 0xFF,
// CHECK-NEXT:   SINT test_2 at 8 size 8 = 0xFF
// CHECK-NEXT: }>)

// -----

  func.func nested @MLIR_VPURegMapped_CreateDpuVariantRegisterAllowOverlapFalse() {
    VPURegMapped.RegisterWrapper regAttr(<regForTest offset 12 size 32 {
      UINT test_1 at 0 size 8 = 0xFF,
      SINT test_2 at 8 size 8 = 0xFF
    }>)
    return
  }

// CHECK:      VPURegMapped.RegisterWrapper regAttr(<regForTest offset 12 size 32 {
// CHECK-NEXT:   UINT test_1 at 0 size 8 = 0xFF,
// CHECK-NEXT:   SINT test_2 at 8 size 8 = 0xFF
// CHECK-NEXT: }>)

// -----

  func.func nested @MLIR_VPURegMapped_CreateDpuVariantRegisterOneRegField() {
    VPURegMapped.RegisterWrapper regAttr(<regForTest offset 12 size 4 allowOverlap = UINT 0>)
    return
  }

// CHECK:      VPURegMapped.RegisterWrapper regAttr(<regForTest offset 12 size 4 allowOverlap = UINT 0>)

// -----

  func.func nested @MLIR_VPURegMapped_CreateDpuVariantRegisterRequiresVersion() {
    VPURegMapped.RegisterWrapper regAttr(<regForTest offset 12 size 32 {
      UINT test_1 at 0 size 8 = 0xFF,
      SINT test_2 at 8 size 8 = 0xFF
    }>)
    return
  }

// CHECK:      VPURegMapped.RegisterWrapper regAttr(<regForTest offset 12 size 32 {
// CHECK-NEXT:   UINT test_1 at 0 size 8 = 0xFF,
// CHECK-NEXT:   SINT test_2 at 8 size 8 = 0xFF
// CHECK-NEXT: }>)
