//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

  func.func nested @MLIR_VPURegMapped_CreateDpuVariantRegister() {
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<UINT test at 0 size 8 = 0>)
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<SINT test at 0 size 8 = 0xFF>)
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<FP test at 0 size 64 = 0x200000>)
    return
  }

// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<UINT test at 0 size 8 = 0>)
// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<SINT test at 0 size 8 = 0xFF>)
// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<FP test at 0 size 64 = 0x200000>)

// -----

  func.func nested @MLIR_VPURegMapped_CreateDpuVariantRegisterRequiresVersion() {
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<UINT test at 0 size 8 = 0>)
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<SINT test at 0 size 8 = 0xFF>)
    VPURegMapped.RegisterFieldWrapper regFieldAttr(<FP test at 0 size 64 = 0x200000>)
    return
  }

// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<UINT test at 0 size 8 = 0>)
// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<SINT test at 0 size 8 = 0xFF>)
// CHECK: VPURegMapped.RegisterFieldWrapper regFieldAttr(<FP test at 0 size 64 = 0x200000>)
