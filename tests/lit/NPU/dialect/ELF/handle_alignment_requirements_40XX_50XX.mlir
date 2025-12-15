//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-VPUASM-to-NPUReg40XX --handle-alignment-requirements %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule attributes {config.arch = #config.arch_kind<NPU40XX>} {
  config.ExecutorResource 1 of @DMA_NN
  config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @read_after_write_act_dma_f16_f16 inputsInfo : {
    DataInfo "input" : tensor<1x10x2x3xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x10x2x3xf16>
  }
  func.func private @read_after_write_act_dma_f16_f16() {
    ELF.Main @ELFMain {
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x10x2x3xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x10x2x3xf16, #NHWC, @DDR> :  swizzling(0)>
      VPUASM.DeclareKernelEntry @DeclareKernelEntry0 : "test_kernel_elf_pad"
      VPUASM.DeclareKernelEntry @DeclareKernelEntry1 : "test_kernel_elf_pad"
      VPUASM.DeclareKernelEntry @DeclareKernelEntry2 : "test_kernel_elf_pad"
      VPUASM.DeclareKernelEntry @DeclareKernelEntry3 : "test_kernel_elf_pad"
      VPUASM.DeclareKernelEntry @DeclareKernelEntry4 : "test_kernel_elf_pad"
      VPUASM.DeclareKernelEntry @DeclareKernelEntry5 : "test_kernel_elf_pad"
      ELF.CreateLogicalSection @program.DMA.cmx.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        //CHECK:  ELF.CreateLogicalSection @program.DMA.cmx.0.0 aligned(1)
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }
      ELF.CreateLogicalSection @program.DMA.cmx.0.1 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        //CHECK:  ELF.CreateLogicalSection @program.DMA.cmx.0.1 aligned(1)
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_1 idx(!VPURegMapped.Index<0:1:1>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_2 idx(!VPURegMapped.Index<0:1:2>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_3 idx(!VPURegMapped.Index<0:1:3>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_4 idx(!VPURegMapped.Index<0:1:4>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_5 idx(!VPURegMapped.Index<0:1:5>) <DMA>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_1_6 idx(!VPURegMapped.Index<0:1:6>) <DMA>
      }
      ELF.CreateLogicalSection @program.ActKernelRange.cmx.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        //CHECK:  ELF.CreateLogicalSection @program.ActKernelRange.cmx.0.0 aligned(1)
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelRange>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange_0_0_1 idx(!VPURegMapped.Index<0:0:1>) <ActKernelRange>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange_0_0_2 idx(!VPURegMapped.Index<0:0:2>) <ActKernelRange>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange_0_0_3 idx(!VPURegMapped.Index<0:0:3>) <ActKernelRange>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange_0_0_4 idx(!VPURegMapped.Index<0:0:4>) <ActKernelRange>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelRange_0_0_5 idx(!VPURegMapped.Index<0:0:5>) <ActKernelRange>
      }
      ELF.CreateLogicalSection @program.ActKernelInvocation.cmx.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        //CHECK:  ELF.CreateLogicalSection @program.ActKernelInvocation.cmx.0.0 aligned(1)
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <ActKernelInvocation>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_1 idx(!VPURegMapped.Index<0:0:1>) <ActKernelInvocation>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_2 idx(!VPURegMapped.Index<0:0:2>) <ActKernelInvocation>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_3 idx(!VPURegMapped.Index<0:0:3>) <ActKernelInvocation>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_4 idx(!VPURegMapped.Index<0:0:4>) <ActKernelInvocation>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_ActKernelInvocation_0_0_5 idx(!VPURegMapped.Index<0:0:5>) <ActKernelInvocation>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        //CHECK:  ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1)
        VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x10x2x3xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer3 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x1x1x1xsi8, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer4 !VPUASM.Buffer< "CMX_NN"[0] <119> : memref<1x1x1x1xsi8, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer5 !VPUASM.Buffer< "CMX_NN"[0] <120> : memref<1x10x2x3xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer6 !VPUASM.Buffer< "CMX_NN"[0] <240> : memref<1x10x2x3xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer7 !VPUASM.Buffer< "CMX_NN"[0] <239> : memref<1x1x1x1xsi8, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer8 !VPUASM.Buffer< "CMX_NN"[0] <360> : memref<1x10x2x3xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer9 !VPUASM.Buffer< "CMX_NN"[0] <359> : memref<1x1x1x1xsi8, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer10 !VPUASM.Buffer< "CMX_NN"[0] <480> : memref<1x10x2x3xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer11 !VPUASM.Buffer< "CMX_NN"[0] <479> : memref<1x1x1x1xsi8, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer12 !VPUASM.Buffer< "CMX_NN"[0] <600> : memref<1x10x2x3xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer13 !VPUASM.Buffer< "CMX_NN"[0] <599> : memref<1x1x1x1xsi8, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer14 !VPUASM.Buffer< "CMX_NN"[0] <720> : memref<1x10x2x3xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer15 !VPUASM.Buffer< "CMX_NN"[0] <719> : memref<1x1x1x1xsi8, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @text.shave aligned(1024) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
        //CHECK:  ELF.CreateSection @text.shave aligned(1024)
        VPUASM.DeclareKernelText @DeclareKernelText0 : "test_kernel_elf_pad"
        //CHECK:  ELF.Pad size(880)
        VPUASM.DeclareKernelText @DeclareKernelText1 : "test_kernel_elf_pad"
        //CHECK:  ELF.Pad size(880)
        VPUASM.DeclareKernelText @DeclareKernelText2 : "test_kernel_elf_pad"
        //CHECK:  ELF.Pad size(880)
        VPUASM.DeclareKernelText @DeclareKernelText3 : "test_kernel_elf_pad"
        //CHECK:  ELF.Pad size(880)
        VPUASM.DeclareKernelText @DeclareKernelText4 : "test_kernel_elf_pad"
        //CHECK:  ELF.Pad size(880)
        VPUASM.DeclareKernelText @DeclareKernelText5 : "test_kernel_elf_pad"
        //CHECK:  ELF.Pad size(128)
      }
      ELF.CreateSection @program.shave.data aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
        //CHECK:  ELF.CreateSection @program.shave.data aligned(1024)
        VPUASM.DeclareKernelData @DeclareKernelArgs0 : "test_kernel_elf_pad"
        VPUASM.DeclareKernelData @DeclareKernelArgs1 : "test_kernel_elf_pad"
        VPUASM.DeclareKernelData @DeclareKernelArgs2 : "test_kernel_elf_pad"
        VPUASM.DeclareKernelData @DeclareKernelArgs3 : "test_kernel_elf_pad"
        VPUASM.DeclareKernelData @DeclareKernelArgs4 : "test_kernel_elf_pad"
        VPUASM.DeclareKernelData @DeclareKernelArgs5 : "test_kernel_elf_pad"
        //CHECK:  ELF.Pad size(128)
      }
      ELF.CreateSection @program.shave.parameter aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_SHAVE") secLocation(<DDR>) {
        //CHECK:  ELF.CreateSection @program.shave.parameter aligned(64)
        VPUASM.KernelParams @KernelParams0 inputs([@buffer.CMX_NN.0::@DeclareBuffer2]) outputs([@buffer.CMX_NN.0::@DeclareBuffer5]) dynamicInputShapes([]) dynamicOutputShapes([])
        kernel_type("test_kernel_elf_pad") < {inputDimsBinaryVector = [], inputStridesBinaryVector = [],
        kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        outputDimsBinaryVector = [], outputStridesBinaryVector = []}>
        //CHECK:  ELF.Pad size(56)
        VPUASM.KernelParams @KernelParams1 inputs([@buffer.CMX_NN.0::@DeclareBuffer5]) outputs([@buffer.CMX_NN.0::@DeclareBuffer6]) dynamicInputShapes([]) dynamicOutputShapes([])
        kernel_type("test_kernel_elf_pad") < {inputDimsBinaryVector = [], inputStridesBinaryVector = [],
        kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        outputDimsBinaryVector = [], outputStridesBinaryVector = []}>
        //CHECK:  ELF.Pad size(56)
        VPUASM.KernelParams @KernelParams2 inputs([@buffer.CMX_NN.0::@DeclareBuffer6]) outputs([@buffer.CMX_NN.0::@DeclareBuffer8]) dynamicInputShapes([]) dynamicOutputShapes([])
        kernel_type("test_kernel_elf_pad") < {inputDimsBinaryVector = [], inputStridesBinaryVector = [],
        kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        outputDimsBinaryVector = [], outputStridesBinaryVector = []}>
        //CHECK:  ELF.Pad size(56)
        VPUASM.KernelParams @KernelParams3 inputs([@buffer.CMX_NN.0::@DeclareBuffer8]) outputs([@buffer.CMX_NN.0::@DeclareBuffer10]) dynamicInputShapes([]) dynamicOutputShapes([])
        kernel_type("test_kernel_elf_pad") < {inputDimsBinaryVector = [], inputStridesBinaryVector = [],
        kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        outputDimsBinaryVector = [], outputStridesBinaryVector = []}>
        //CHECK:  ELF.Pad size(56)
        VPUASM.KernelParams @KernelParams4 inputs([@buffer.CMX_NN.0::@DeclareBuffer10]) outputs([@buffer.CMX_NN.0::@DeclareBuffer12]) dynamicInputShapes([]) dynamicOutputShapes([])
        kernel_type("test_kernel_elf_pad") < {inputDimsBinaryVector = [], inputStridesBinaryVector = [],
        kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        outputDimsBinaryVector = [], outputStridesBinaryVector = []}>
        //CHECK:  ELF.Pad size(56)
        VPUASM.KernelParams @KernelParams5 inputs([@buffer.CMX_NN.0::@DeclareBuffer12]) outputs([@buffer.CMX_NN.0::@DeclareBuffer14]) dynamicInputShapes([]) dynamicOutputShapes([])
        kernel_type("test_kernel_elf_pad") < {inputDimsBinaryVector = [], inputStridesBinaryVector = [],
        kernel_params = [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        outputDimsBinaryVector = [], outputStridesBinaryVector = []}>
        //CHECK:  ELF.Pad size(128)
      }
      ELF.CreateSection @program.barrier aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        //CHECK:  ELF.CreateSection @program.barrier aligned(64)
        VPUASM.ConfigureBarrier @ConfigureBarrier0 idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier1 idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier2 idx(!VPURegMapped.Index<0:0:2>) (2) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier3 idx(!VPURegMapped.Index<0:0:3>) (3) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier4 idx(!VPURegMapped.Index<0:0:4>) (4) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier5 idx(!VPURegMapped.Index<0:0:5>) (5) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier6 idx(!VPURegMapped.Index<0:0:6>) (6) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier7 idx(!VPURegMapped.Index<0:0:7>) (7) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier8 idx(!VPURegMapped.Index<0:0:8>) (8) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier9 idx(!VPURegMapped.Index<0:0:9>) (9) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier10 idx(!VPURegMapped.Index<0:0:10>) (10) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier11 idx(!VPURegMapped.Index<0:0:11>) (11) => (-1) counts(1 : 1)
        VPUASM.ConfigureBarrier @ConfigureBarrier12 idx(!VPURegMapped.Index<0:0:12>) (12) => (-1) counts(1 : 1)
      }
      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        //CHECK:  ELF.CreateSection @task.dma.0.0 aligned(64)
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.DMA.cmx.0.0::@DeclareTaskBuffer_DMA_0_0_0) input(@DeclareBuffer0) outputs([@buffer.CMX_NN.0::@DeclareBuffer2]) waits([]) updates([0 : ui8]) start_after(1) clean_after(0) dma_descriptor(<numPlanes = 0 : i4, len = -8 : i4, srcWidth = -8 : i4, srcStride = -8 : i4, srcPlaneStride = 0 : i4, dstWidth = -8 : i4, dstStride = -8 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
      }
      ELF.CreateSection @task.dma.0.1 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        //CHECK:  ELF.CreateSection @task.dma.0.1 aligned(64)
        VPUASM.NNDMA @NNDMA_0_1_0 idx(!VPURegMapped.Index<0:1:0>) taskLocation(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_0) links(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_1) input(@buffer.CMX_NN.0::@DeclareBuffer3) outputs([@buffer.CMX_NN.0::@DeclareBuffer4]) waits([1 : ui8]) updates([2 : ui8]) start_after(3) clean_after(2) dma_descriptor(<numPlanes = 0 : i4, len = 1 : i4, srcWidth = 1 : i4, srcStride = 1 : i4, srcPlaneStride = 0 : i4, dstWidth = 1 : i4, dstStride = 1 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
        VPUASM.NNDMA @NNDMA_0_1_1 idx(!VPURegMapped.Index<0:1:1>) taskLocation(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_1) links(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_2) input(@buffer.CMX_NN.0::@DeclareBuffer3) outputs([@buffer.CMX_NN.0::@DeclareBuffer7]) waits([3 : ui8]) updates([4 : ui8]) start_after(5) clean_after(4) dma_descriptor(<numPlanes = 0 : i4, len = 1 : i4, srcWidth = 1 : i4, srcStride = 1 : i4, srcPlaneStride = 0 : i4, dstWidth = 1 : i4, dstStride = 1 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
        VPUASM.NNDMA @NNDMA_0_1_2 idx(!VPURegMapped.Index<0:1:2>) taskLocation(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_2) links(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_3) input(@buffer.CMX_NN.0::@DeclareBuffer3) outputs([@buffer.CMX_NN.0::@DeclareBuffer9]) waits([5 : ui8]) updates([6 : ui8]) start_after(7) clean_after(6) dma_descriptor(<numPlanes = 0 : i4, len = 1 : i4, srcWidth = 1 : i4, srcStride = 1 : i4, srcPlaneStride = 0 : i4, dstWidth = 1 : i4, dstStride = 1 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
        VPUASM.NNDMA @NNDMA_0_1_3 idx(!VPURegMapped.Index<0:1:3>) taskLocation(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_3) links(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_4) input(@buffer.CMX_NN.0::@DeclareBuffer3) outputs([@buffer.CMX_NN.0::@DeclareBuffer11]) waits([7 : ui8]) updates([8 : ui8]) start_after(9) clean_after(8) dma_descriptor(<numPlanes = 0 : i4, len = 1 : i4, srcWidth = 1 : i4, srcStride = 1 : i4, srcPlaneStride = 0 : i4, dstWidth = 1 : i4, dstStride = 1 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
        VPUASM.NNDMA @NNDMA_0_1_4 idx(!VPURegMapped.Index<0:1:4>) taskLocation(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_4) links(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_5) input(@buffer.CMX_NN.0::@DeclareBuffer3) outputs([@buffer.CMX_NN.0::@DeclareBuffer13]) waits([9 : ui8]) updates([10 : ui8]) start_after(11) clean_after(10) dma_descriptor(<numPlanes = 0 : i4, len = 1 : i4, srcWidth = 1 : i4, srcStride = 1 : i4, srcPlaneStride = 0 : i4, dstWidth = 1 : i4, dstStride = 1 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
        VPUASM.NNDMA @NNDMA_0_1_5 idx(!VPURegMapped.Index<0:1:5>) taskLocation(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_5) links(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_6) input(@buffer.CMX_NN.0::@DeclareBuffer3) outputs([@buffer.CMX_NN.0::@DeclareBuffer15]) waits([11 : ui8]) updates([12 : ui8]) start_after(13) clean_after(12) dma_descriptor(<numPlanes = 0 : i4, len = 1 : i4, srcWidth = 1 : i4, srcStride = 1 : i4, srcPlaneStride = 0 : i4, dstWidth = 1 : i4, dstStride = 1 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
        VPUASM.NNDMA @NNDMA_0_1_6 idx(!VPURegMapped.Index<0:1:6>) taskLocation(@program.DMA.cmx.0.1::@DeclareTaskBuffer_DMA_0_1_6) input(@buffer.CMX_NN.0::@DeclareBuffer14) outputs([@DeclareBuffer1]) waits([12 : ui8]) updates([]) start_after(13) clean_after(12) dma_descriptor(<numPlanes = 0 : i4, len = -8 : i4, srcWidth = -8 : i4, srcStride = -8 : i4, srcPlaneStride = 0 : i4, dstWidth = -8 : i4, dstStride = -8 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>)
      }
      ELF.CreateSection @task.shave.range.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        //CHECK:  ELF.CreateSection @task.shave.range.0.0 aligned(64)
        VPUASM.ActKernelRange @ActKernelRange0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_0) kernelTaskType(@COMPUTE) calls @text.shave::@DeclareKernelText0 : @DeclareKernelEntry0
        VPUASM.ActKernelRange @ActKernelRange1 idx(!VPURegMapped.Index<0:0:1>) taskLocation(@program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_1) kernelTaskType(@COMPUTE) calls @text.shave::@DeclareKernelText1 : @DeclareKernelEntry1
        VPUASM.ActKernelRange @ActKernelRange2 idx(!VPURegMapped.Index<0:0:2>) taskLocation(@program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_2) kernelTaskType(@COMPUTE) calls @text.shave::@DeclareKernelText2 : @DeclareKernelEntry2
        VPUASM.ActKernelRange @ActKernelRange3 idx(!VPURegMapped.Index<0:0:3>) taskLocation(@program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_3) kernelTaskType(@COMPUTE) calls @text.shave::@DeclareKernelText3 : @DeclareKernelEntry3
        VPUASM.ActKernelRange @ActKernelRange4 idx(!VPURegMapped.Index<0:0:4>) taskLocation(@program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_4) kernelTaskType(@COMPUTE) calls @text.shave::@DeclareKernelText4 : @DeclareKernelEntry4
        VPUASM.ActKernelRange @ActKernelRange5 idx(!VPURegMapped.Index<0:0:5>) taskLocation(@program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_5) kernelTaskType(@COMPUTE) calls @text.shave::@DeclareKernelText5 : @DeclareKernelEntry5
      }
      ELF.CreateSection @task.shave.invocation.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        //CHECK:  ELF.CreateSection @task.shave.invocation.0.0 aligned(64)
        VPUASM.ActKernelInvocation @ActKernelInvocation0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.ActKernelInvocation.cmx.0.0::@DeclareTaskBuffer_ActKernelInvocation_0_0_0) -> @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_0(kernel_data : @program.shave.data::@DeclareKernelArgs0, kernel_params : @program.shave.parameter::@KernelParams0) waits([0 : ui8]) updates([1 : ui8]) tile(0) start_after(2) clean_after(1) range_index(0)
        VPUASM.ActKernelInvocation @ActKernelInvocation1 idx(!VPURegMapped.Index<0:0:1>) taskLocation(@program.ActKernelInvocation.cmx.0.0::@DeclareTaskBuffer_ActKernelInvocation_0_0_1) -> @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_1(kernel_data : @program.shave.data::@DeclareKernelArgs1, kernel_params : @program.shave.parameter::@KernelParams1) waits([2 : ui8]) updates([3 : ui8]) tile(0) start_after(4) clean_after(3) range_index(1)
        VPUASM.ActKernelInvocation @ActKernelInvocation2 idx(!VPURegMapped.Index<0:0:2>) taskLocation(@program.ActKernelInvocation.cmx.0.0::@DeclareTaskBuffer_ActKernelInvocation_0_0_2) -> @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_2(kernel_data : @program.shave.data::@DeclareKernelArgs2, kernel_params : @program.shave.parameter::@KernelParams2) waits([4 : ui8]) updates([5 : ui8]) tile(0) start_after(6) clean_after(5) range_index(2)
        VPUASM.ActKernelInvocation @ActKernelInvocation3 idx(!VPURegMapped.Index<0:0:3>) taskLocation(@program.ActKernelInvocation.cmx.0.0::@DeclareTaskBuffer_ActKernelInvocation_0_0_3) -> @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_3(kernel_data : @program.shave.data::@DeclareKernelArgs3, kernel_params : @program.shave.parameter::@KernelParams3) waits([6 : ui8]) updates([7 : ui8]) tile(0) start_after(8) clean_after(7) range_index(3)
        VPUASM.ActKernelInvocation @ActKernelInvocation4 idx(!VPURegMapped.Index<0:0:4>) taskLocation(@program.ActKernelInvocation.cmx.0.0::@DeclareTaskBuffer_ActKernelInvocation_0_0_4) -> @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_4(kernel_data : @program.shave.data::@DeclareKernelArgs4, kernel_params : @program.shave.parameter::@KernelParams4) waits([8 : ui8]) updates([9 : ui8]) tile(0) start_after(10) clean_after(9) range_index(4)
        VPUASM.ActKernelInvocation @ActKernelInvocation5 idx(!VPURegMapped.Index<0:0:5>) taskLocation(@program.ActKernelInvocation.cmx.0.0::@DeclareTaskBuffer_ActKernelInvocation_0_0_5) -> @program.ActKernelRange.cmx.0.0::@DeclareTaskBuffer_ActKernelRange_0_0_5(kernel_data : @program.shave.data::@DeclareKernelArgs5, kernel_params : @program.shave.parameter::@KernelParams5) waits([10 : ui8]) updates([11 : ui8]) tile(0) start_after(12) clean_after(11) range_index(5)
      }
      ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
        //CHECK:  ELF.CreateSection @note.MappedInferenceVersion aligned(4)
        VPUASM.MappedInferenceVersion @MappedInferenceVersion_0_0(11 _ 4 _ 10)
      }
      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        //CHECK:  ELF.CreateSection @program.mapped_inference aligned(64)
        VPUASM.MappedInference @MappedInference : dmas([[@task.dma.0.0::@NNDMA_0_0_0, @task.dma.0.1::@NNDMA_0_1_0]]) actKernelRanges([@task.shave.range.0.0::@ActKernelRange0]) actKernelInvocations([@task.shave.invocation.0.0::@ActKernelInvocation0]) barriers(@program.barrier::@ConfigureBarrier0) dmaCount([[1, 7], [0, 0]]) invariantCount([0, 0]) variantCount([0, 0]) actKernelRangesCount([6, 0]) actKernelInvocationsCount([6, 0]) mediaCount(0) barrierCount(13) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0)
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.DMA.cmx.0.0 of(@program.DMA.cmx.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.DMA.cmx.0.1 of(@program.DMA.cmx.0.1) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.ActKernelRange.cmx.0.0 of(@program.ActKernelRange.cmx.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.ActKernelInvocation.cmx.0.0 of(@program.ActKernelInvocation.cmx.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.text.shave of(@text.shave) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.shave.data of(@program.shave.data) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.shave.parameter of(@program.shave.parameter) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.barrier of(@program.barrier) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.task.dma.0.1 of(@task.dma.0.1) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.task.shave.range.0.0 of(@task.shave.range.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.task.shave.invocation.0.0 of(@task.shave.invocation.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.note.MappedInferenceVersion of(@note.MappedInferenceVersion) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @entry of(@program.mapped_inference::@MappedInference) type(<VPU_STT_ENTRY>) size(0) value(0)
      }
      ELF.CreateMetadataSection @MetadataSection aligned(8) secFlags("SHF_NONE")  {
        VPUASM.NetworkMetadata @NetworkMetadata
      }
    }
    return
  }
}
