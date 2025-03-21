//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX
  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegister() {
    VPURegMapped.RegisterMappedWrapper regMapped(<
      regMappedForTest {
        regForTest_1 offset 12 size 32 {
          UINT test_1 at 0 size 8 = 0xFF,
          SINT test_2 at 8 size 8 = 0xFF
        },
        regForTest_2 offset 12 size 32 allowOverlap {
          UINT test_1 at 0 size 8 = 0xFF,
          SINT test_2 at 8 size 8 = 0xFF
        },
        regForTest_3 offset 12 size 4 allowOverlap = FP 0x200000
      }
    >)
    return
  }

// -----

// CHECK:      VPURegMapped.RegisterMappedWrapper regMapped(<
// CHECK-NEXT:   regMappedForTest {
// CHECK-NEXT:     regForTest_1 offset 12 size 32 {
// CHECK-NEXT:       UINT test_1 at 0 size 8 = 0xFF,
// CHECK-NEXT:       SINT test_2 at 8 size 8 = 0xFF
// CHECK-NEXT:     },
// CHECK-NEXT:     regForTest_2 offset 12 size 32 allowOverlap {
// CHECK-NEXT:       UINT test_1 at 0 size 8 = 0xFF,
// CHECK-NEXT:       SINT test_2 at 8 size 8 = 0xFF
// CHECK-NEXT:     },
// CHECK-NEXT:     regForTest_3 offset 12 size 4 allowOverlap = FP 0x200000
// CHECK-NEXT:   }
// CHECK-NEXT: >)

  func.func private @MLIR_VPURegMapped_CreateDpuVariantRegisterRequiresVersion() {
    VPURegMapped.RegisterMappedWrapper regMapped(<
      regMappedForTest {
        regForTest_1 offset 12 size 32 {
          UINT test_1 at 0 size 8 = 0xFF,
          SINT test_2 at 8 size 8 = 0xFF
        },
        regForTest_2 offset 12 size 32 allowOverlap {
          UINT test_1 at 0 size 8 = 0xFF,
          SINT test_2 at 8 size 8 = 0xFF
        },
        regForTest_3 offset 12 size 4 allowOverlap = FP 0x200000
      }
    >)
    return
  }

// CHECK:      VPURegMapped.RegisterMappedWrapper regMapped(<
// CHECK-NEXT:   regMappedForTest {
// CHECK-NEXT:     regForTest_1 offset 12 size 32 {
// CHECK-NEXT:       UINT test_1 at 0 size 8 = 0xFF,
// CHECK-NEXT:       SINT test_2 at 8 size 8 = 0xFF
// CHECK-NEXT:     },
// CHECK-NEXT:     regForTest_2 offset 12 size 32 allowOverlap {
// CHECK-NEXT:       UINT test_1 at 0 size 8 = 0xFF,
// CHECK-NEXT:       SINT test_2 at 8 size 8 = 0xFF
// CHECK-NEXT:     },
// CHECK-NEXT:     regForTest_3 offset 12 size 4 allowOverlap = FP 0x200000
// CHECK-NEXT:   }
// CHECK-NEXT: >)
