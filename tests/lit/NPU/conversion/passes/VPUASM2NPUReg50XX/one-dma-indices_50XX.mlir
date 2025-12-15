//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUASM-to-NPUReg50XX %s | FileCheck %s
// REQUIRES: dev-build && arch-NPU50XX

module @OneDMAWithoutAttributes {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x50257x768xf32, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1024x768xf32, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x50257x768xf32, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1024x768xf32, @DDR> :  swizzling(0)>

      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
        VPUASM.DeclareBuffer @DeclareBuffer10 !VPUASM.Buffer< "CMX_NN"[4] <0> : memref<1x1024xi64, [@CMX_NN, 4]> :  swizzling(0)>
      }
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(1) clean_after(2) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 1024 : i32, srcWidth = 1024 : i32, srcStride = 1024 : i32, srcPlaneStride = 0 : i32, dstWidth = 1024 : i32, dstStride = 1024 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>) indices( @builtin.tasks.DMA0::@DeclareBuffer10) {addressing_mode = 0 : i64}
      // CHECK-NOT:   VPUASM.NNDMA
      // CHECK:       NPUReg50XX.NNDMA
      // CHECK:  UINT dma_cfg_fields_src_list_cfg = 1
      // CHECK:  UINT dma_width_src = 0xC00
      // CHECK:  UINT dma_width_dst = 0x400
      // CHECK:  UINT dma_src = 0
      // CHECK:  UINT dma_dst = 0
      // CHECK:  UINT dma_list_size_src = 0x400
      // CHECK:  dma_stride_dst_1 = UINT 0xC00
      // CHECK:  UINT start_after_ = 1
      // CHECK:  UINT clean_after_ = 2

      }
    }
    return
  }
}

// -----

!qElemType = !quant.uniform<i4:f32, 1.000000e+00>

module @OneI4DMAWithoutAttributes {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x50257x768x!qElemType, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1024x768x!qElemType, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1x50257x768x!qElemType, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x1024x768x!qElemType, @DDR> :  swizzling(0)>

      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
        VPUASM.DeclareBuffer @DeclareBuffer10 !VPUASM.Buffer< "CMX_NN"[4] <0> : memref<1x1024xi64, [@CMX_NN, 4]> :  swizzling(0)>
      }
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
      VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(1) clean_after(2) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 1024 : i32, srcWidth = 1024 : i32, srcStride = 1024 : i32, srcPlaneStride = 0 : i32, dstWidth = 1024 : i32, dstStride = 1024 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>) indices( @builtin.tasks.DMA0::@DeclareBuffer10) {addressing_mode = 1 : i64}
      // CHECK-NOT:   VPUASM.NNDMA
      // CHECK:       NPUReg50XX.NNDMA
      // CHECK:  UINT dma_cfg_fields_src_list_cfg = 2
      // CHECK:  UINT dma_width_src = 0x180
      // CHECK:  UINT dma_width_dst = 0x400
      // CHECK:  UINT dma_src = 0
      // CHECK:  UINT dma_dst = 0
      // CHECK:  UINT dma_list_size_src = 0x400
      // CHECK:  dma_stride_dst_1 = UINT 0x180
      // CHECK:  UINT start_after_ = 1
      // CHECK:  UINT clean_after_ = 2

      }
    }
    return
  }
}
