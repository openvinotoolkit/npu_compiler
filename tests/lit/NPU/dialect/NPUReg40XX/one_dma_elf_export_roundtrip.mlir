//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --platform=%platform% --export-ELF %s | FileCheck %s
// REQUIRES: platform-NPU4000

!quantileType = !QuantileType.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

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
    DataInfo "Input" : tensor<1x1024x!quantileType>
  } outputsInfo : {
    DataInfo "Output" : tensor<1x1024x!quantileType>
  }
  VPUASM.InputBindings inputDeclarations : {
    VPUASM.DeclareBuffer @Input !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1024x!quantileType, @DDR> :  swizzling(0)>
  }
  VPUASM.OutputBindings outputDeclarations : {
    VPUASM.DeclareBuffer @Output !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1024x!quantileType, @DDR> :  swizzling(0)>
  }
  VPUASM.ProfilingBindings profilingDeclarations : {
  }
  func.func @main() {
    ELF.Main {
      ELF.CreateLogicalSection @data.BuffersIO.DMA aligned(1) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        VPUASM.DeclareBuffer @DeclareBufferDMA !VPUASM.Buffer< "DDR"[0] <0> : memref<1x128x1024x!quantileType, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @data.BuffersIO.LEON aligned(1) secType(SHT_NOBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        VPUASM.DeclareBuffer @DeclareBufferLEON !VPUASM.Buffer< "DDR"[0] <0> : memref<1x128x1024x!quantileType, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @data.BuffersIO.SHAVE aligned(1) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
        VPUASM.DeclareBuffer @DeclareBufferSHAVE !VPUASM.Buffer< "DDR"[0] <0> : memref<2560x1024x1024x!quantileType, @DDR> :  swizzling(0)>
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
