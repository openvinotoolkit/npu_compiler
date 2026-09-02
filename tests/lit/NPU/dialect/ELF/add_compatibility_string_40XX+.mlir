//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --add-compatibility-string="" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

func.func @test() {
  ELF.Main {
    ELF.CreateSection @note.MappedInferenceVersion aligned(64) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
      NPUReg40XX.MappedInferenceVersion(11 _ 4 _ 10) {sym_name = "MappedInferenceVersion_0_0_0"}
    }
  }
  return
}

// CHECK: ELF.CreateSection @compatibility_string aligned(64) secType(VPU_SHT_COMPAT_STR) secFlags("SHF_NONE") secLocation(<DDR>) {
// CHECK-NEXT: ELF.CompatibilityString compatibilityString("compiler={{[0-9]+(\.[0-9]+)*}};npu={{[0-9]+}};t={{[0-9]+}};elf={{[0-9]+(\.[0-9]+)*}};mi={{[0-9]+(\.[0-9]+)*}}")
// CHECK-NEXT: }
