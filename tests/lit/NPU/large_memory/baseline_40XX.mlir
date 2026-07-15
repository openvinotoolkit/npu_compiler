//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --platform=%platform% --export-ELF %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @Test attributes {config.platform = #config.platform<NPU4000>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Input" : tensor<1x1024xui8>
  } outputsInfo : {
    DataInfo "Output" : tensor<1x1024xui8>
  }
  VPUASM.InputBindings inputDeclarations : {
    VPUASM.DeclareBuffer @Input !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1024xui8, @DDR> :  swizzling(0)>
  }
  VPUASM.OutputBindings outputDeclarations : {
    VPUASM.DeclareBuffer @Output !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1024xui8, @DDR> :  swizzling(0)>
  }
  VPUASM.ProfilingBindings profilingDeclarations : {
  }
  func.func @main() {
    ELF.Main {
      ELF.CreateLogicalSection @data.BuffersIO.DMA aligned(1) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        VPUASM.DeclareBuffer @DeclareBufferDMA !VPUASM.Buffer< "DDR"[0] <0> : memref<3072x1024x1024xui8, @DDR> :  swizzling(0)> // 3 GB
      }
      ELF.CreateLogicalSection @data.BuffersIO.LEON aligned(1) secType(SHT_NOBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        VPUASM.DeclareBuffer @DeclareBufferLEON !VPUASM.Buffer< "DDR"[0] <0> : memref<3072x1024x1024xui8, @DDR> :  swizzling(0)> // 3 GB
      }
      ELF.CreateLogicalSection @data.BuffersIO.SHAVE aligned(1) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
        VPUASM.DeclareBuffer @DeclareBufferSHAVE !VPUASM.Buffer< "DDR"[0] <0> : memref<1536x1024x1024xui8, @DDR> :  swizzling(0)> // 1.5 GB
      }
      ELF.CreateMetadataSection @MetadataSection aligned(8) secFlags("SHF_NONE")  {
        VPUASM.NetworkMetadata @NetworkMetadata
      }
    }
    return
  }

  // CHECK: ELF
  // CHECK: .strtab
  // CHECK: .symstrtab
  // CHECK: MetadataSection
  // CHECK: data.BuffersIO.DMA
  // CHECK: data.BuffersIO.LEON
  // CHECK: data.BuffersIO.SHAVE
}
