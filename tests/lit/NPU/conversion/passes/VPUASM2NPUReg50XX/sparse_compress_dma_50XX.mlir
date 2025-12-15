//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-VPUASM-to-NPUReg50XX %s | FileCheck %s
// REQUIRES: dev-build && arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @OneDMAWithoutAttributes attributes {config.arch = #config.arch_kind<NPU50XX>} {
  config.ExecutorResource 1 of @DMA_NN
  config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x16x32x32xf16>
    DataInfo "input_1" : tensor<1x16x32x32xi1>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x32x32xf16>
    DataInfo "output_1" : tensor<1x1x1x25632xf16>
    DataInfo "output_2" : tensor<1x16x32x32xi1>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x16x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @input_1_buffDecl !VPUASM.Buffer< "NetworkInput"[1] <32768> : memref<1x16x32x32xi1, #NHWC, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x16x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @output_1_buffDecl !VPUASM.Buffer< "NetworkOutput"[1] <32768> : memref<1x1x1x25632xf16, @DDR> :  swizzling(0)>
    VPUASM.DeclareBuffer @output_2_buffDecl !VPUASM.Buffer< "NetworkOutput"[2] <84032> : memref<1x16x32x32xi1, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @io.NetworkInput0 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT) secLocation(<NetworkInput>) {
        VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x16x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @io.NetworkInput1 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USERINPUT) secLocation(<NetworkInput>) {
        VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkInput"[1] <32768> : memref<1x16x32x32xi1, #NHWC, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @io.NetworkOutput0 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT) secLocation(<NetworkOutput>) {
        VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x16x32x32xf16, #NHWC, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @io.NetworkOutput1 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT) secLocation(<NetworkOutput>) {
        VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "NetworkOutput"[1] <0> : memref<1x1x1x25632xf16, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @io.NetworkOutput2 aligned(1) secType(SHT_NOBITS) secFlags(VPU_SHF_USEROUTPUT) secLocation(<NetworkOutput>) {
        VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "NetworkOutput"[2] <0> : memref<1x16x32x32xi1, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @program.DMA.cmx.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_1 idx(!VPURegMapped.Index<0:0:1>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_2 idx(!VPURegMapped.Index<0:0:2>) <DMA>
      }
      ELF.CreateLogicalSection @program.DMA.cmx.0.1 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_1 idx(!VPURegMapped.Index<0:1:1>) <DMA>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "CMX_NN"[0] <0>     : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer6 !VPUASM.Buffer< "CMX_NN"[0] <32768> : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer7 !VPUASM.Buffer< "CMX_NN"[0] <65536> : memref<32xui8, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer8 !VPUASM.Buffer< "CMX_NN"[0] <65568> : memref<1x16x32x32xi1, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer9 !VPUASM.Buffer< "CMX_NN"[0] <81952> : memref<1x16x32x32xi1, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @program.barrier aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ConfigureBarrier @ConfigureBarrier0 idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(1 : 1) {elfMemOffsetAttrKey = 0 : ui64}
        VPUASM.ConfigureBarrier @ConfigureBarrier1 idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(1 : 1) {elfMemOffsetAttrKey = 8 : ui64}
        VPUASM.ConfigureBarrier @ConfigureBarrier2 idx(!VPURegMapped.Index<0:0:2>) (2) => (-1) counts(1 : 1) {elfMemOffsetAttrKey = 16 : ui64}
      }
      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.DMA.cmx.0.0::@DeclareTaskBuffer_DMA_0_0_0) links(@program.DMA.cmx.0.0::@DeclareTaskBuffer_DMA_0_0_2)
          input(@io.NetworkInput0::@DeclareBuffer0) outputs([@buffer.CMX_NN.0::@DeclareBuffer5])
          waits([]) updates([0 : ui8]) start_after(1) clean_after(0)
          dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)
          acceleration_mode(<DISABLE>) {elfMemOffsetAttrKey = 0 : ui64}
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg50XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_acceleration_cfg = 0

        VPUASM.NNDMA @NNDMA_0_0_1 idx(!VPURegMapped.Index<0:0:1>) taskLocation(@program.DMA.cmx.0.0::@DeclareTaskBuffer_DMA_0_0_1) links(@program.DMA.cmx.0.0::@DeclareTaskBuffer_DMA_0_0_2)
           input(@io.NetworkInput1::@DeclareBuffer1) outputs([@buffer.CMX_NN.0::@DeclareBuffer8])
          waits([]) updates([0 : ui8]) start_after(1) clean_after(0)
          dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)
          acceleration_mode(<DISABLE>) {elfMemOffsetAttrKey = 224 : ui64}
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg50XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_acceleration_cfg = 0

        VPUASM.NNDMA @NNDMA_0_0_2 idx(!VPURegMapped.Index<0:0:2>) taskLocation(@program.DMA.cmx.0.0::@DeclareTaskBuffer_DMA_0_0_2)
          input(@buffer.CMX_NN.0::@DeclareBuffer5) outputs([@io.NetworkOutput1::@DeclareBuffer3])
          waits([0 : ui8]) updates([1 : ui8]) start_after(2)  clean_after(0)
          dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)
          acceleration_mode(<COMPRESSION>) act_compression_size_entry(@buffer.CMX_NN.0::@DeclareBuffer7) act_compression_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer8)
          {elfMemOffsetAttrKey = 0 : ui64}
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg50XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_rwf_en = 0
        // CHECK:  UINT dma_cfg_fields_rws_en = 1
        // CHECK:  UINT dma_cfg_fields_acceleration_cfg = 1
        // CHECK:  UINT dma_acc_info_compress_dtype = 1
        // CHECK:  UINT dma_acc_info_compress_reserved1 = 0
        // CHECK:  UINT dma_acc_info_compress_sparse = 1
        // CHECK:  UINT dma_acc_info_compress_bitc_en = 1
        // CHECK:  UINT dma_acc_info_compress_z = 1
        // CHECK:  UINT dma_acc_info_compress_bitmap_buf_sz = 0x800
        // CHECK:  UINT dma_acc_info_compress_reserved2 = 0
        // CHECK:  UINT dma_acc_info_compress_bitmap_base_addr = 0x20000
      }

    ELF.CreateSection @task.dma.0.1 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) taskLocation(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_0) links(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_1)
          input(@io.NetworkOutput1::@DeclareBuffer3) outputs([@buffer.CMX_NN.0::@DeclareBuffer6])
          waits([1 : ui8]) updates([2 : ui8]) start_after(3) clean_after(0)
          dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)
          acceleration_mode(<DECOMPRESSION>) act_compression_size_entry(@buffer.CMX_NN.0::@DeclareBuffer7) act_compression_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer9)
          {elfMemOffsetAttrKey = 224 : ui64}
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg50XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_rwf_en = 1
        // CHECK:  UINT dma_cfg_fields_rws_en = 0
        // CHECK:  UINT dma_cfg_fields_acceleration_cfg = 2
        // CHECK:  UINT dma_acc_info_decompress_dtype = 1
        // CHECK:  UINT dma_acc_info_decompress_reserved1 = 0
        // CHECK:  UINT dma_acc_info_decompress_sparse = 1
        // CHECK:  UINT dma_acc_info_decompress_bitc_en = 1
        // CHECK:  UINT dma_acc_info_decompress_z = 1
        // CHECK:  UINT dma_acc_info_decompress_bitmap_buf_sz = 0x800
        // CHECK:  UINT dma_acc_info_decompress_reserved2 = 0
        // CHECK:  UINT dma_acc_info_decompress_bitmap_base_addr = 0x20000

        VPUASM.NNDMA @NNDMA_0_1_1 idx(!VPURegMapped.Index<0:1:1>) taskLocation(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_1)
          input(@buffer.CMX_NN.0::@DeclareBuffer6) outputs([@io.NetworkOutput0::@DeclareBuffer2])
          waits([2 : ui8]) updates([]) start_after(3) clean_after(0)
          dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>)
          acceleration_mode(<DISABLE>) {elfMemOffsetAttrKey = 224 : ui64}
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg50XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_acceleration_cfg = 0
      }
    }
    return
  }
}
