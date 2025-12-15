//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUASM-to-NPUReg50XX %s | FileCheck %s
// REQUIRES: dev-build && arch-NPU50XX

module @mainModule attributes {config.arch = #config.arch_kind<NPU50XX>} {
  config.ExecutorResource 1 of @DMA_NN
  config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @dma_compressed_constant inputsInfo : {
    DataInfo "input_0" : tensor<1024x16x1x1xf16>
  } outputsInfo : {
    DataInfo "output_1" : tensor<1024x16x1x1xf16>
  }
  func.func @dma_compressed_constant() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @io.NetworkInput0 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT) secLocation(<NetworkInput>) {
        VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1024x16x1x1xf16, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @io.NetworkOutput0 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT) secLocation(<NetworkOutput>) {
        VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1024x16x1x1xf16, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @program.DMA.cmx.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }
      ELF.CreateLogicalSection @program.DMA.cmx.0.1 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) <DMA>
      }
      ELF.CreateSection @program.barrier aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ConfigureBarrier @ConfigureBarrier0 idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(1 : 1)
      }
      ELF.CreateSection @buffer.Constant.0.constant aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<Constant>) {
        // let's imagine @Declare0 is the compressed representation of tensor
        // %cst = const.Declare memref<1024x16x1x1xf16, [@DDR, 0]> = dense<0.1> : tensor<1024x16x1x1xf16>
        // after --compress-weights-btc pass
        VPUASM.ConstBuffer @Declare0 !VPUASM.Buffer< "Constant"[0] <0> : memref<100x1x1x1xui8, [@DDR, 0]> :  swizzling(0)> = dense<100> : tensor<100x1x1x1xui8>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1024x16x1x1xf16, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.DMA.cmx.0.0::@DeclareTaskBuffer_DMA_0_0_0)
          input(@buffer.Constant.0.constant::@Declare0) outputs([@buffer.CMX_NN.0::@DeclareBuffer2])
          waits([]) updates([0 : ui8]) start_after(0) clean_after(0)
          dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)
          acceleration_mode(<DECOMPRESSION>)
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg50XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_rwf_en = 0
        // CHECK:  UINT dma_cfg_fields_rws_en = 0
        // CHECK:  UINT dma_cfg_fields_acceleration_cfg = 2
        // CHECK:  UINT dma_acc_info_decompress_dtype = 1
        // CHECK:  UINT dma_acc_info_decompress_bitc_en = 1
      }
      ELF.CreateSection @task.dma.0.1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) taskLocation(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_0)
          input(@buffer.CMX_NN.0::@DeclareBuffer2) outputs([@io.NetworkOutput0::@DeclareBuffer1])
          waits([0 : ui8]) updates([]) start_after(0) clean_after(0)
          dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)
          acceleration_mode(<DISABLE>)
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg50XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_acceleration_cfg = 0
      }
    }
    return
  }
}
