//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUASM-to-NPUReg40XX %s | FileCheck %s
// REQUIRES: dev-build && arch-NPU40XX

module @OneDMAWithoutAttributes {
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
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(1) clean_after(2) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 48 : i32, srcWidth = 48 : i32, srcStride = 48 : i32, srcPlaneStride = 48 : i32, dstWidth = 48 : i32, dstStride = 48 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_num_dim = 0
        // CHECK:  UINT dma_cfg_fields_conversion_cfg = 0
        // CHECK:  UINT dma_width_src = 0x30
        // CHECK:  UINT dma_width_dst = 0x30
        // CHECK:  UINT dma_src = 0
        // CHECK:  UINT dma_dst = 0
        // CHECK:  UINT start_after_ = 1
        // CHECK:  UINT clean_after_ = 2
      }
    }
    return
  }
}

// -----

module @OneSyncDMA {
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
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<0x0x0x0xf16, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<0x0x0x0xf16, @DDR> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(1) clean_after(2) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>)
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:   UINT dma_cfg_fields_memset_en = 1
      }
    }
    return
  }
}

// -----

module @OneDMAWithNNDMATransactionAttr {
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
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        // All DMADescriptorAttr fields set to 0 to show that it is ignored if NNDMATransaction is present
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(1) clean_after(2) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>)
        acceleration_mode(<DISABLE>)
        {dma_transaction = #VPUMI40XX.NNDMATransaction<inputType = memref<1x2x3x4xf16, @DDR>, outputType = memref<1x2x3x4xf16, @DDR>>}
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_num_dim = 0
        // CHECK:  UINT dma_cfg_fields_conversion_cfg = 0
        // CHECK:  UINT dma_width_src = 0x30
        // CHECK:  UINT dma_width_dst = 0x30
        // CHECK:  UINT dma_src = 0
        // CHECK:  UINT dma_dst = 0
        // CHECK:  UINT start_after_ = 1
        // CHECK:  UINT clean_after_ = 2
      }
    }
    return
  }
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @OneINT4DMAWithNNDMATransactionAttr {
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
      VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x5x308x128x!qElemType, {order = #NCHW, strides = [16515072, 589824, 128, 1]}, @DDR> :  swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x5x308x128x!qElemType, [@CMX_NN, 0]> :  swizzling(0)>
      ELF.CreateLogicalSection @builtin.tasks.DMA0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0 idx(!VPURegMapped.Index<0:0:0>) <DMA>
      }
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(1) clean_after(2) dma_descriptor(<numPlanes = 0 : i32, len = 98560 : i32, srcWidth = 19712 : i32, srcStride = 294912 : i32, srcPlaneStride = 0 : i32, dstWidth = 98560 : i32, dstStride = 98560 : i32, dstPlaneStride = 0 : i32>)
        acceleration_mode(<DISABLE>)
        {dma_transaction = #VPUMI40XX.NNDMATransaction<inputType = memref<1x5x308x128x!qElemType, {order = #NCHW, strides = [16515072, 589824, 128, 1]}, @DDR>, outputType = memref<1x5x308x128x!qElemType, [@CMX_NN, 0]>>}
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:  UINT dma_width_src = 0x4D00
        // CHECK:  UINT dma_dim_size_src_1 = 4
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d1, d5, d2)>

module @OneDMAWithPermuteDMATransactionAttr {
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
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        // All DMADescriptorAttr fields set to 0 to show that it is ignored if NNDMATransaction is present
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@builtin.tasks.DMA0::@DeclareTaskBuffer_DMA_0) input(@DeclareBuffer0) outputs([@DeclareBuffer1]) waits([]) updates([]) start_after(1) clean_after(2) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>)
        acceleration_mode(<DISABLE>)
        {dma_transaction = #VPUMI40XX.PermuteDMATransaction<inputType = memref<1x2x2x4x1x6xf16, #map, [@CMX_NN, 0]>, outputType = memref<1x4x1x2x6x2xf16, #map1, [@CMX_NN, 0]>, mappingOrder = #map2, loopOrder = #map>}
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_num_dim = 2
        // CHECK:  UINT dma_cfg_fields_conversion_cfg = 0
        // CHECK:  UINT dma_width_src = 0xC0
        // CHECK:  UINT dma_width_dst = 0x10
        // CHECK:  UINT dma_src = 0
        // CHECK:  UINT dma_dst = 0
        // CHECK:  UINT dma_dim_size_src_1 = 0
        // CHECK:  UINT dma_dim_size_dst_1 = 1
        // CHECK:  dma_stride_src_1 = UINT 0,
        // CHECK:  dma_stride_dst_1 = UINT 0x60
        // CHECK:  UINT dma_dim_size_src_2 = 0
        // CHECK:  UINT dma_dim_size_dst_2 = 5
        // CHECK:  dma_stride_src_2 = UINT 0
        // CHECK:  dma_stride_dst_2 = UINT 0x10
        // CHECK:  UINT start_after_ = 1
        // CHECK:  UINT clean_after_ = 2
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @UnrollDistributedExpandDMAOutputWithTransactionAttr {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x60x8x56xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x56xf16>
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateSection @buffer.Constant.0.constant aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ConstBuffer @Declare_0 !VPUASM.Buffer< "Constant"[0] <0> : memref<1x64x8x56xf16, #NHWC, @DDR> :  swizzling(0)> = dense<1.000000e+00> : tensor<1x64x8x56xf16>, [#const.Reorder<#NHWC>]
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_0 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x56xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.1 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_1 !VPUASM.Buffer< "CMX_NN"[1] <0> : memref<1x64x8x56xf16, #NHWC, [@CMX_NN, 1]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) input(@buffer.Constant.0.constant::@Declare_0) outputs([@buffer.CMX_NN.0::@DeclareBuffer_0, @buffer.CMX_NN.1::@DeclareBuffer_1])
         waits([]) updates([]) start_after(0) clean_after(0) dma_transaction(#VPUMI40XX.ExpandDMATransaction<inputType = memref<1x64x8x56xf16, #NHWC, @DDR>,
            outputType = !VPUIP.DistributedBuffer<1x64x8x56xf16, {order = #NHWC, strides = [30464, 1, 3808, 68]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>,
            padsBegin = [0, 0, 0, 0], padsEnd = [0, 4, 0, 0]>)
            dma_descriptor(<numPlanes = 0 : i32, len = 4096 : i32, srcWidth = 4096 : i32, srcStride = 4096 : i32, srcPlaneStride = 0 : i32, dstWidth = 64 : i32, dstStride = 68 : i32, dstPlaneStride = 0 : i32>)
            acceleration_mode(<DISABLE>) tile_indexes([0, 1])
      }
      ELF.CreateSection @note.LoaderABIVersion aligned(64) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
        ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
      }
      ELF.CreateSection @perf.metrics aligned(64) secType(VPU_SHT_PERF_METRICS) secFlags("SHF_NONE") secLocation(<DDR>) {
        ELF.PerformanceMetricsSection @PerfMetrics
      }
    }
    return
  }

  // CHECK: ELF.CreateSection @buffer.Constant.0.constant aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
  // CHECK: VPUASM.ConstBuffer @Declare_0 !VPUASM.Buffer< "Constant"[0] <0> : memref<1x64x8x56xf16, #NHWC, @DDR> :  swizzling(0)> = dense<1.000000e+00> : tensor<1x64x8x56xf16>, [#const.Reorder<#NHWC>]
  // CHECK: ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
  // CHECK: VPUASM.DeclareBuffer @DeclareBuffer_0 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x56xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
  // CHECK: ELF.CreateLogicalSection @buffer.CMX_NN.1 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
  // CHECK: VPUASM.DeclareBuffer @DeclareBuffer_1 !VPUASM.Buffer< "CMX_NN"[1] <0> : memref<1x64x8x56xf16, #NHWC, [@CMX_NN, 1]> :  swizzling(0)>
     // CHECK: ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        // CHECK: NPUReg40XX.NNDMA descriptor = <
        // CHECK:  DMARegister {
        // CHECK:  UINT dma_cfg_fields_num_dim = 1,
        // CHECK: UINT dma_cfg_fields_src_burst_length = 0xF,
        // CHECK: UINT dma_cfg_fields_dst_burst_length = 0xF,
        // CHECK: UINT dma_cfg_fields_arb_qos = 0xFF,
        // CHECK: UINT dma_cfg_fields_ord = 1,
        // CHECK:    dma_remote_width_fetch = UINT 0xE000,
        // CHECK:  dma_width {
        // CHECK:    UINT dma_width_src = 0xE000,
        // CHECK:    UINT dma_width_dst = 0x78,
        // CHECK:  dma_list_size {
        // CHECK:    UINT dma_list_size_src = 0,
        // CHECK:    UINT dma_list_size_dst = 0x1BF,
        // CHECK:  dma_dim_size {
        // CHECK:    UINT dma_dim_size_src_1 = 0,
        // CHECK:    UINT dma_dim_size_dst_1 = 0x1BF,
        // CHECK:  dma_stride_src_1 = UINT 0,
        // CHECK:  dma_stride_dst_1 = UINT 0x88,
        // CHECK:  dma_dim_size_2 {
        // CHECK:    UINT dma_dim_size_src_2 = 0,
        // CHECK:    UINT dma_dim_size_dst_2 = 0,
        // CHECK:  dma_list_addr {
        // CHECK:    UINT dma_list_addr_src = 0,
        // CHECK:    UINT dma_list_addr_dst = 0,
        // CHECK:  dma_stride_dst_2 = UINT 0,
        // CHECK:  dma_remote_width_store = UINT 0,
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @OneDMAWithPerAxisTileDMATransactionAttr {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1xf16>
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @buffer.DDR.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC") secLocation(<DDR>) {
        VPUASM.DeclareBuffer @DeclareBuffer_0 !VPUASM.Buffer< "DDR"[0] <0> : memref<1x4x122x120xf16, #NHWC, @DDR> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_1 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x4x122x240xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) input(@buffer.DDR.0::@DeclareBuffer_0) outputs([@buffer.CMX_NN.0::@DeclareBuffer_1]) waits([]) updates([]) start_after(0) clean_after(0) dma_transaction(#VPUMI40XX.PerAxisTileDMATransaction<inputType = memref<1x4x122x120xf16, #NHWC, @DDR>, outputType = memref<1x4x122x240xf16, #NHWC, [@CMX_NN, 0]>, axis = 3 : i64, tiles = 2 : i64>) acceleration_mode(<DISABLE>) tile_indexes([0])
        // CHECK-NOT:   VPUASM.NNDMA
        // CHECK:       NPUReg40XX.NNDMA
        // CHECK:  UINT dma_cfg_fields_num_dim = 2
        // CHECK:  UINT dma_cfg_fields_conversion_cfg = 0
        // CHECK:  UINT dma_width_src = 0x3C0
        // CHECK:  UINT dma_width_dst = 0x39300
        // CHECK:  UINT dma_src = 0
        // CHECK:  UINT dma_dst = 0
        // CHECK:  UINT dma_dim_size_src_1 = 1
        // CHECK:  UINT dma_dim_size_dst_1 = 0
        // CHECK:  dma_stride_src_1 = UINT 0,
        // CHECK:  dma_stride_dst_1 = UINT 0
        // CHECK:  UINT dma_dim_size_src_2 = 0x79
        // CHECK:  UINT dma_dim_size_dst_2 = 0
        // CHECK:  dma_stride_src_2 = UINT 0x3C0
        // CHECK:  dma_stride_dst_2 = UINT 0
        // CHECK:  UINT start_after_ = 0
        // CHECK:  UINT clean_after_ = 0

      }
    }
    return
  }
}
