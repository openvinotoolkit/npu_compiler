// RUN: vpux-translate --vpu-arch=%arch% --export-ELF %s | FileCheck %s
// REQUIRES: arch-NPU37XX

module @Test attributes {VPU.arch = #VPU.arch_kind<NPU37XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1982464 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @SHAVE_NN
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 2 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Input" : tensor<1x1024xui8>
  } outputsInfo : {
    DataInfo "Output" : tensor<1x1024xui8>
  }
  func.func @main(%arg0: memref<1x1024xui8, @DDR>, %arg1: memref<1x1024xui8, @DDR>) -> memref<1x1024xui8, @DDR> {
    %buffer_dma = VPURT.DeclareBuffer <DDR> <0> -> memref<3072x1024x1024xui8, @DDR> // 3 GB
    %buffer_leon = VPURT.DeclareBuffer <DDR> <0> -> memref<288x1024x1024xui8, @DDR> // 288 MB
    %buffer_shave = VPURT.DeclareBuffer <DDR> <0> -> memref<1536x1024x1024xui8, @DDR> // 1.5 GB
    %metadata_sec = ELFNPU37XX.CreateMetadataSection secFlags("SHF_NONE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".metadata"} -> !ELFNPU37XX.Section {
      %metadata = VPUMI37XX.NetworkMetadata -> !VPURegMapped.Index<0:0:0>
    }
    %buffer_dma_sec = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO.DMA"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %buffer_dma : memref<3072x1024x1024xui8, @DDR>
    }
    %buffer_leon_sec = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO.LEON"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %buffer_leon : memref<288x1024x1024xui8, @DDR>
    }
    %buffer_shave_sec = ELFNPU37XX.CreateLogicalSection secType(SHT_NOBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".data.BuffersIO.SHAVE"} -> !ELFNPU37XX.Section {
      ELFNPU37XX.PutOpInSection %buffer_shave : memref<1536x1024x1024xui8, @DDR>
    }
    return %arg1 : memref<1x1024xui8, @DDR>
  }

  // CHECK: ELF
  // CHECK: .strtab
  // CHECK: .symstrtab
  // CHECK: .metadata
  // CHECK: .data.BuffersIO.DMA
  // CHECK: .data.BuffersIO.LEON
  // CHECK: .data.BuffersIO.SHAVE
}
