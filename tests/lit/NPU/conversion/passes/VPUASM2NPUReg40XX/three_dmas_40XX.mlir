//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUASM-to-NPUReg40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule attributes {config.arch = #config.arch_kind<NPU40XX>} {
  config.ExecutorResource 1 of @DMA_NN
  config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @race_condition_dma_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
    DataInfo "output_1" : tensor<1x16x16x16xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x16x16x16xf16, #NHWC, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x16x16x16xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @output_1_buffDecl !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<1x16x16x16xf16, #NHWC, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func private @race_condition_dma_f16_f16() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x16x16x16xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x16x16x16xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<1x16x16x16xf16, #NHWC, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_1 idx(!VPURegMapped.Index<0:0:1>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_2 idx(!VPURegMapped.Index<0:0:2>) <DMA>
      }
      ELF.CreateLogicalSection @builtin.data.nncmx0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "CMX_NN"[0] <16> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @builtin.data.nncmx1 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "CMX_NN"[1] <32> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]> :  swizzling(0)>
      }
      ELF.CreateSection @text.Barriers aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ConfigureBarrier @ConfigureBarrier0 idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(2 : 2)
        VPUASM.ConfigureBarrier @ConfigureBarrier1 idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(2 : 2)
      }
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) links(@text.nndma0::@NNDMA_0_0_1) input(@DeclareBuffer0) outputs([@builtin.data.nncmx0::@DeclareBuffer3]) waits([]) updates([0 : ui8]) start_after(0) clean_after(0) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 8192 : i32, srcWidth = 8192 : i32, srcStride = 8192 : i32, srcPlaneStride = 0 : i32, dstWidth = 8192 : i32, dstStride = 8192 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:  UINT dma_link_address = 0
        // CHECK:  UINT dma_cfg_fields_barrier_en = 1
        // CHECK:  UINT dma_width_src = 0x2000
        // CHECK:  UINT dma_width_dst = 0x2000
        // CHECK:  UINT dma_src = 0
        // CHECK:  UINT dma_dst = 0
        // CHECK:  dma_barrier_prod_mask_lower = UINT 1
        // CHECK:  dma_barrier_cons_mask_lower = UINT 0
        // CHECK:  UINT dma_barrier_prod_mask_upper = 0
        // CHECK:  UINT dma_barrier_cons_mask_upper = 0
        VPUASM.NNDMA @NNDMA_0_0_1 idx(!VPURegMapped.Index<0:0:1>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_1) links(@text.nndma0::@NNDMA_0_0_2) input(@DeclareBuffer0) outputs([@builtin.data.nncmx0::@DeclareBuffer3]) waits([0 : ui8]) updates([1 : ui8]) start_after(0) clean_after(0) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0x2000 : i32, srcWidth = 0x2000 : i32, srcStride = 0x2000 : i32, srcPlaneStride = 0 : i32, dstWidth = 0x2000 : i32, dstStride = 0x2000 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:  UINT dma_link_address = 0
        // CHECK:  UINT dma_cfg_fields_barrier_en = 1
        // CHECK:  UINT dma_width_src = 0x2000
        // CHECK:  UINT dma_width_dst = 0x2000
        // CHECK:  UINT dma_src = 0
        // CHECK:  UINT dma_dst = 0
        // CHECK:  dma_barrier_prod_mask_lower = UINT 2
        // CHECK:  dma_barrier_cons_mask_lower = UINT 1
        // CHECK:  UINT dma_barrier_prod_mask_upper = 0
        // CHECK:  UINT dma_barrier_cons_mask_upper = 0
        VPUASM.NNDMA @NNDMA_0_0_2 idx(!VPURegMapped.Index<0:0:2>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_2) input(@builtin.data.nncmx0::@DeclareBuffer3) outputs([@DeclareBuffer1]) waits([1 : ui8]) updates([]) start_after(0) clean_after(0) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0x2000 : i32, srcWidth = 0x2000 : i32, srcStride = 0x2000 : i32, srcPlaneStride = 0 : i32, dstWidth = 0x2000 : i32, dstStride = 0x2000 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:  UINT dma_link_address = 0
        // CHECK:  UINT dma_cfg_fields_barrier_en = 1
        // CHECK:  UINT dma_width_src = 0x2000
        // CHECK:  UINT dma_width_dst = 0x2000
        // CHECK:  UINT dma_src = 0x200000
        // CHECK:  UINT dma_dst = 0
        // CHECK:  dma_barrier_prod_mask_lower = UINT 0
        // CHECK:  dma_barrier_cons_mask_lower = UINT 2
        // CHECK:  UINT dma_barrier_prod_mask_upper = 0
        // CHECK:  UINT dma_barrier_cons_mask_upper = 0
      }
    }
    return
  }
}
