//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUASM-to-NPUReg50XX %s | FileCheck %s
// REQUIRES: dev-build && arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @OneDMABCastToSlices024 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }

  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
      VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.2 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "CMX_NN"[2] <0> : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 2]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.4 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer6 !VPUASM.Buffer< "CMX_NN"[4] <0> : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 4]> :  swizzling(0)>
      }

      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@buffer.CMX_NN.0::@DeclareBuffer4, @buffer.CMX_NN.2::@DeclareBuffer5, @buffer.CMX_NN.4::@DeclareBuffer6]) waits([]) updates([]) start_after(1) clean_after(2) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 48 : i32, srcWidth = 48 : i32, srcStride = 48 : i32, srcPlaneStride = 48 : i32, dstWidth = 48 : i32, dstStride = 48 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)  tile_indexes([0, 2, 4])

        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg50XX.NNDMA
        // CHECK:  UINT dma_dst = 0x2A00000
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @OneDMABCastToSlices135 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }

  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x2x3x4xf16, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
      VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.2 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "CMX_NN"[2] <0> : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 2]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.4 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer6 !VPUASM.Buffer< "CMX_NN"[4] <0> : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 4]> :  swizzling(0)>
      }

      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@buffer.CMX_NN.0::@DeclareBuffer4, @buffer.CMX_NN.2::@DeclareBuffer5, @buffer.CMX_NN.4::@DeclareBuffer6]) waits([]) updates([]) start_after(1) clean_after(2) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 48 : i32, srcWidth = 48 : i32, srcStride = 48 : i32, srcPlaneStride = 48 : i32, dstWidth = 48 : i32, dstStride = 48 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)  tile_indexes([1, 3, 5])

        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg50XX.NNDMA
        // CHECK:  UINT dma_dst = 0x5400000
      }
    }
    return
  }
}
