//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX
  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegisterAllowOverlapTrue() {
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

  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegisterAllowOverlapFalse() {
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

  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegisterOneRegField() {
    VPURegMapped.RegisterWrapper regAttr(<regForTest offset 12 size 4 allowOverlap = UINT 0>)
    return
  }

// CHECK:      VPURegMapped.RegisterWrapper regAttr(<regForTest offset 12 size 4 allowOverlap = UINT 0>)

// -----

  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegisterRequiresVersion() {
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
