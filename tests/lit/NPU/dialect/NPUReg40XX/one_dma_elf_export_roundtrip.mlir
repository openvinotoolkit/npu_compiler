//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-ELF %s | FileCheck %s
// REQUIRES: arch-NPU40XX

!quantileFloatType = !QuantileFloat.quantileFloat<4, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

module @Test attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.TileResource 6 of @NCE at 1.700000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Input" : tensor<1x1024x!quantileFloatType>
  } outputsInfo : {
    DataInfo "Output" : tensor<1x1024x!quantileFloatType>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @Input !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1024x!quantileFloatType, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @Output !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1024x!quantileFloatType, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @data.BuffersIO.DMA aligned(1) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_DMA") {
        VPUASM.DeclareBuffer @DeclareBufferDMA !VPUASM.Buffer< "DDR"[0] <0> : memref<1x128x1024x!quantileFloatType, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @data.BuffersIO.LEON aligned(1) secType(SHT_NOBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {
        VPUASM.DeclareBuffer @DeclareBufferLEON !VPUASM.Buffer< "DDR"[0] <0> : memref<1x128x1024x!quantileFloatType, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @data.BuffersIO.SHAVE aligned(1) secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") {
        VPUASM.DeclareBuffer @DeclareBufferSHAVE !VPUASM.Buffer< "DDR"[0] <0> : memref<2560x1024x1024x!quantileFloatType, @DDR> :  swizzling(0)>
      }
      ELF.CreateMetadataSection @MetadataSection aligned(8) secFlags("SHF_NONE") {
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
