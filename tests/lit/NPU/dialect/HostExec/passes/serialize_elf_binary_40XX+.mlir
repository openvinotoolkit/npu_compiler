//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --serialize-elf-to-binary %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @StaticEltwiseNHWC

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @StaticEltwiseNHWC attributes {config.arch = #config.arch_kind<NPU40XX>, config.revisionID = #config.revision_id<REVISION_NONE>, config.compilationMode = #config.compilation_mode<HostCompile>} {
  config.PipelineOptions @Options {
    config.Option @VPU.EnableExtraStaticShapeOps : true
    config.Option @VPU.EnableAdaptiveStripping : false
    config.Option @VPU.EnableSEPtrsOperations : false
    config.Option @VPU.EnableExperimentalSEPtrsOperations : false
    config.Option @VPU.EnableVPUNNPreSplit : false
    config.Option @VPU.FP16CompressedConv : false
    config.Option @VPU.EnableDCIM : true
    config.Option @VPU.ReduceSupported : false
    config.Option @VPU.AutoPaddingODU : false
    config.Option @VPU.AutoPaddingIDU : false
    config.Option @VPU.SprLUTEnabled : false
    config.Option @VPU.FragmentationAvoidRatioPipeliningLargeWeights : 4.500000e-01 : f32
    config.Option @VPU.UseDedicatedFifoPerShaveEngine : false
    config.Option @VPU.BarrierMaxVariantSum : 64 : ui64
    config.Option @VPU.BarrierMaxVariantCount : 128 : ui64
    config.Option @VPU.MetadataMaxVariantCount : 128 : ui64
    config.Option @VPU.MetadataMaxInvariantCount : 64 : ui64
    config.Option @VPU.MetadataMaxKernelInvocationCount : 64 : ui64
    config.Option @VPU.MetadataMaxKernelRangeCount : 64 : ui64
    config.Option @VPU.MetadataMaxMediaCount : 4 : ui64
    config.Option @VPU.MaxKernelSize : 11 : si64
  }
  config.Resources 6 of @NCE at 1.850000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x1000xf16>
    DataInfo "input2" : tensor<1x16x720x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x1000xf16>
  }
  module @OneDMAWithoutAttributes attributes {config.arch = #config.arch_kind<NPU40XX>, config.revisionID = #config.revision_id<REVISION_NONE>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.PipelineOptions @Options {
    config.Option @VPU.EnableExtraStaticShapeOps : true
    config.Option @VPU.EnableAdaptiveStripping : false
    config.Option @VPU.EnableSEPtrsOperations : false
    config.Option @VPU.EnableExperimentalSEPtrsOperations : false
    config.Option @VPU.EnableVPUNNPreSplit : false
    config.Option @VPU.FP16CompressedConv : false
    config.Option @VPU.EnableDCIM : true
    config.Option @VPU.ReduceSupported : false
    config.Option @VPU.AutoPaddingODU : false
    config.Option @VPU.AutoPaddingIDU : false
    config.Option @VPU.SprLUTEnabled : false
    config.Option @VPU.FragmentationAvoidRatioPipeliningLargeWeights : 4.500000e-01 : f32
    config.Option @VPU.UseDedicatedFifoPerShaveEngine : false
    config.Option @VPU.BarrierMaxVariantSum : 64 : ui64
    config.Option @VPU.BarrierMaxVariantCount : 128 : ui64
    config.Option @VPU.MetadataMaxVariantCount : 128 : ui64
    config.Option @VPU.MetadataMaxInvariantCount : 64 : ui64
    config.Option @VPU.MetadataMaxKernelInvocationCount : 64 : ui64
    config.Option @VPU.MetadataMaxKernelRangeCount : 64 : ui64
    config.Option @VPU.MetadataMaxMediaCount : 4 : ui64
    config.Option @VPU.MaxKernelSize : 11 : si64
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main1 inputsInfo : {
    DataInfo "input_0" : tensor<1x90x1000x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x90x1000x16xf16>
  }
  config.Resources 6 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x90x1000x16xf16> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x90x1000x16xf16> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main1() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(1) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DMA> {offset = 59904 : ui64}
      }
      ELF.CreateLogicalSection @io.NetworkInput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkInput>) {
        VPUASM.DeclareBuffer @DeclareBuffer0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x90x1000x16xf16> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @io.NetworkOutput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USEROUTPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkOutput>) {
        VPUASM.DeclareBuffer @DeclareBuffer1 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x90x1000x16xf16> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer2 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<16xui32, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        NPUReg40XX.NNDMA descriptor = <
          DMARegister {
            dma_watermark {
              UINT dma_watermark = 0,
            }
            dma_link_address {
              UINT dma_link_address = 0,
            }
            dma_lra {
              UINT dma_lra = 0,
            }
            dma_lba_addr = UINT 0,
            dma_src_aub = UINT 0,
            dma_dst_aub = UINT 0,
            dma_cfg_fields {
              UINT dma_cfg_fields_num_dim = 0,
              UINT dma_cfg_fields_int_en = 0,
              UINT dma_cfg_fields_int_id = 0,
              UINT dma_cfg_fields_src_burst_length = 0xF,
              UINT dma_cfg_fields_dst_burst_length = 0xF,
              UINT dma_cfg_fields_arb_qos = 0xFF,
              UINT dma_cfg_fields_ord = 1,
              UINT dma_cfg_fields_barrier_en = 1,
              UINT dma_cfg_fields_memset_en = 0,
              UINT dma_cfg_fields_atp_en = 1,
              UINT dma_cfg_fields_watermark_en = 0,
              UINT dma_cfg_fields_rwf_en = 0,
              UINT dma_cfg_fields_rws_en = 0,
              UINT dma_cfg_fields_src_list_cfg = 0,
              UINT dma_cfg_fields_dst_list_cfg = 0,
              UINT dma_cfg_fields_conversion_cfg = 0,
              UINT dma_cfg_fields_acceleration_cfg = 0,
              UINT dma_cfg_fields_tile4_cfg = 0,
              UINT dma_cfg_fields_axi_user_bits_cfg = 0,
              UINT dma_cfg_fields_hwp_id_en = 1,
              UINT dma_cfg_fields_hwp_id = 0,
              UINT dma_cfg_fields_reserved = 0,
            }
            dma_remote_width_fetch = UINT 0x2BF200,
            dma_width {
              UINT dma_width_src = 0x2BF200,
              UINT dma_width_dst = 0x2BF200,
            }
            dma_acc_info_compress {
              UINT dma_acc_info_compress_dtype = 0,
              UINT dma_acc_info_compress_reserved1 = 0,
              UINT dma_acc_info_compress_sparse = 0,
              UINT dma_acc_info_compress_bitc_en = 0,
              UINT dma_acc_info_compress_z = 0,
              UINT dma_acc_info_compress_bitmap_buf_sz = 0,
              UINT dma_acc_info_compress_reserved2 = 0,
              UINT dma_acc_info_compress_bitmap_base_addr = 0,
            }
            dma_acc_info_decompress {
              UINT dma_acc_info_decompress_dtype = 0,
              UINT dma_acc_info_decompress_reserved1 = 0,
              UINT dma_acc_info_decompress_sparse = 0,
              UINT dma_acc_info_decompress_bitc_en = 0,
              UINT dma_acc_info_decompress_z = 0,
              UINT dma_acc_info_decompress_reserved2 = 0,
              UINT dma_acc_info_decompress_bitmap_base_addr = 0,
            }
            dma_acc_info_w_prep {
              UINT dma_acc_info_w_prep_dtype = 0,
              UINT dma_acc_info_w_prep_reserved1 = 0,
              UINT dma_acc_info_w_prep_sparse = 0,
              UINT dma_acc_info_w_prep_zeropoint = 0,
              UINT dma_acc_info_w_prep_ic = 0,
              UINT dma_acc_info_w_prep_filtersize = 0,
              UINT dma_acc_info_w_prep_reserved2 = 0,
              UINT dma_acc_info_w_prep_bitmap_base_addr = 0,
            }
            dma_mset_data = UINT 0,
            dma_src_addr {
              UINT dma_src = 0,
              UINT dma_sra = 0,
            }
            dma_dst_addr {
              UINT dma_dst = 0,
              UINT dma_dra = 0,
            }
            dma_sba_addr = UINT 0,
            dma_dba_addr = UINT 0,
            dma_barrier_prod_mask_lower = UINT 0,
            dma_barrier_cons_mask_lower = UINT 0,
            dma_barrier_prod_mask_upper {
              UINT dma_barrier_prod_mask_upper = 0,
            }
            dma_barrier_cons_mask_upper {
              UINT dma_barrier_cons_mask_upper = 0,
            }
            dma_list_size {
              UINT dma_list_size_src = 0,
              UINT dma_list_size_dst = 0,
            }
            dma_dim_size {
              UINT dma_dim_size_src_1 = 0,
              UINT dma_dim_size_dst_1 = 0,
            }
            dma_stride_src_1 = UINT 0,
            dma_stride_dst_1 = UINT 0,
            dma_dim_size_2 {
              UINT dma_dim_size_src_2 = 0,
              UINT dma_dim_size_dst_2 = 0,
            }
            dma_list_addr {
              UINT dma_list_addr_src = 0,
              UINT dma_list_addr_dst = 0,
            }
            dma_stride_src_2 = UINT 0,
            dma_stride_dst_2 = UINT 0,
            dma_remote_width_store = UINT 0,
            dma_dim_size_src_3 = UINT 0,
            dma_dim_size_src_4 = UINT 0,
            dma_dim_size_dst_3 = UINT 0,
            dma_dim_size_dst_4 = UINT 0,
            dma_dim_size_src_5 = UINT 0,
            dma_dim_size_dst_5 = UINT 0,
            dma_stride_src_3 = UINT 0,
            dma_stride_dst_3 = UINT 0,
            dma_stride_src_4 = UINT 0,
            dma_stride_dst_4 = UINT 0,
            dma_stride_src_5 = UINT 0,
            dma_stride_dst_5 = UINT 0,
            dma_word_21_reserved = UINT 0,
            dma_word_22_reserved = UINT 0,
            dma_word_23_reserved = UINT 0,
            dma_barriers_sched {
              UINT start_after_ = 0,
              UINT clean_after_ = 0xFFFFFFFF,
            }
            dma_pad_24_0 = UINT 0,
            dma_pad_24_1 = UINT 0,
            dma_pad_24_2 = UINT 0,
          } requires 11:4:10
        > {elfMemOffsetAttrKey = 0 : ui64, input = @io.NetworkInput.0::@DeclareBuffer0, output_buffs = [@io.NetworkOutput.0::@DeclareBuffer1], sym_name = "NNDMA_0_0_0"}
      }
      VPURegMapped.TaskBufferLayout {ActKernelInvocation = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(64 : ui64), startOffset(53760 : ui64), binaryTaskSize(96 : ui64)>, #VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(0 : ui64), startOffset(59904 : ui64), binaryTaskSize(96 : ui64)>]], ActKernelRange = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(64 : ui64), startOffset(51200 : ui64), binaryTaskSize(40 : ui64)>, #VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(0 : ui64), startOffset(53760 : ui64), binaryTaskSize(40 : ui64)>]], DMA = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(1 : ui64), staticTaskListSize(64 : ui64), startOffset(59904 : ui64), binaryTaskSize(224 : ui64)>, #VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(16 : ui64), startOffset(74240 : ui64), binaryTaskSize(224 : ui64)>]], DPUInvariant = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(64 : ui64), startOffset(0 : ui64), binaryTaskSize(352 : ui64)>]], DPUVariant = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(128 : ui64), startOffset(22528 : ui64), binaryTaskSize(224 : ui64)>]], M2I = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(4 : ui64), startOffset(77824 : ui64), binaryTaskSize(240 : ui64)>]]}
      ELF.CreateSection @meta.PlatformInfo aligned(8) secType(VPU_SHT_PLATFORM_INFO) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.PlatformInfo {elfMemOffsetAttrKey = 0 : ui64, sym_name = "PlatformInfo_0_0"}
      }
      ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
        NPUReg40XX.MappedInferenceVersion(11 _ 4 _ 10) {elfMemOffsetAttrKey = 0 : ui64, sym_name = "MappedInferenceVersion_0_0"}
      }
      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        NPUReg40XX.MappedInference descriptor = <
          VpuMappedInference {
            miVpuNNRTApiVer = UINT 0xB0004,
            miPad0 = UINT 0,
            miReserved0 = UINT 0,
            miLogAddrDmaHwp = UINT 0,
            miTcReserved1 = UINT 0,
            miTcReserved2 = UINT 0,
            miTcDmaDDRCount = UINT 0x40,
            miTcDmaCMXCount = UINT 0x10,
            miDPUInvariantCount = UINT 0x40,
            miTcDPUVariantCount = UINT 0x80,
            miTcActRangeCount = UINT 0x40,
            miTcActInvoCount = UINT 0x40,
            miTcMediaCount = UINT 4,
            miTaskStorageSize = UINT 0,
            taskReferenceR1_DMA_DDR_0 = UINT 0,
            taskReferenceR2_DMA_DDR_0 = UINT 0,
            taskReferenceR3_DMA_DDR_0 = UINT 0,
            taskReferenceAddr_DMA_DDR_0 = UINT 0,
            taskReferenceCount_DMA_DDR_0 = UINT 1,
            taskReferenceR1_DMA_DDR_1 = UINT 0,
            taskReferenceR2_DMA_DDR_1 = UINT 0,
            taskReferenceR3_DMA_DDR_1 = UINT 0,
            taskReferenceAddr_DMA_DDR_1 = UINT 0,
            taskReferenceCount_DMA_DDR_1 = UINT 0,
            taskReferenceR1_DMA_DDR_2 = UINT 0,
            taskReferenceR2_DMA_DDR_2 = UINT 0,
            taskReferenceR3_DMA_DDR_2 = UINT 0,
            taskReferenceAddr_DMA_DDR_2 = UINT 0,
            taskReferenceCount_DMA_DDR_2 = UINT 0,
            taskReferenceR1_DMA_DDR_3 = UINT 0,
            taskReferenceR2_DMA_DDR_3 = UINT 0,
            taskReferenceR3_DMA_DDR_3 = UINT 0,
            taskReferenceAddr_DMA_DDR_3 = UINT 0,
            taskReferenceCount_DMA_DDR_3 = UINT 0,
            taskReferenceR1_DMA_DDR_4 = UINT 0,
            taskReferenceR2_DMA_DDR_4 = UINT 0,
            taskReferenceR3_DMA_DDR_4 = UINT 0,
            taskReferenceAddr_DMA_DDR_4 = UINT 0,
            taskReferenceCount_DMA_DDR_4 = UINT 0,
            taskReferenceR1_DMA_DDR_5 = UINT 0,
            taskReferenceR2_DMA_DDR_5 = UINT 0,
            taskReferenceR3_DMA_DDR_5 = UINT 0,
            taskReferenceAddr_DMA_DDR_5 = UINT 0,
            taskReferenceCount_DMA_DDR_5 = UINT 0,
            taskReferenceR1_DMA_CMX_0 = UINT 0,
            taskReferenceR2_DMA_CMX_0 = UINT 0,
            taskReferenceR3_DMA_CMX_0 = UINT 0,
            taskReferenceAddr_DMA_CMX_0 = UINT 0,
            taskReferenceCount_DMA_CMX_0 = UINT 0,
            taskReferenceR1_DMA_CMX_1 = UINT 0,
            taskReferenceR2_DMA_CMX_1 = UINT 0,
            taskReferenceR3_DMA_CMX_1 = UINT 0,
            taskReferenceAddr_DMA_CMX_1 = UINT 0,
            taskReferenceCount_DMA_CMX_1 = UINT 0,
            taskReferenceR1_DMA_CMX_2 = UINT 0,
            taskReferenceR2_DMA_CMX_2 = UINT 0,
            taskReferenceR3_DMA_CMX_2 = UINT 0,
            taskReferenceAddr_DMA_CMX_2 = UINT 0,
            taskReferenceCount_DMA_CMX_2 = UINT 0,
            taskReferenceR1_DMA_CMX_3 = UINT 0,
            taskReferenceR2_DMA_CMX_3 = UINT 0,
            taskReferenceR3_DMA_CMX_3 = UINT 0,
            taskReferenceAddr_DMA_CMX_3 = UINT 0,
            taskReferenceCount_DMA_CMX_3 = UINT 0,
            taskReferenceR1_DMA_CMX_4 = UINT 0,
            taskReferenceR2_DMA_CMX_4 = UINT 0,
            taskReferenceR3_DMA_CMX_4 = UINT 0,
            taskReferenceAddr_DMA_CMX_4 = UINT 0,
            taskReferenceCount_DMA_CMX_4 = UINT 0,
            taskReferenceR1_DMA_CMX_5 = UINT 0,
            taskReferenceR2_DMA_CMX_5 = UINT 0,
            taskReferenceR3_DMA_CMX_5 = UINT 0,
            taskReferenceAddr_DMA_CMX_5 = UINT 0,
            taskReferenceCount_DMA_CMX_5 = UINT 0,
            taskReferenceR1_DPU_inv_0 = UINT 0,
            taskReferenceR2_DPU_inv_0 = UINT 0,
            taskReferenceR3_DPU_inv_0 = UINT 0,
            taskReferenceAddr_DPU_inv_0 = UINT 0,
            taskReferenceCount_DPU_inv_0 = UINT 0,
            taskReferenceR1_DPU_inv_1 = UINT 0,
            taskReferenceR2_DPU_inv_1 = UINT 0,
            taskReferenceR3_DPU_inv_1 = UINT 0,
            taskReferenceAddr_DPU_inv_1 = UINT 0,
            taskReferenceCount_DPU_inv_1 = UINT 0,
            taskReferenceR1_DPU_inv_2 = UINT 0,
            taskReferenceR2_DPU_inv_2 = UINT 0,
            taskReferenceR3_DPU_inv_2 = UINT 0,
            taskReferenceAddr_DPU_inv_2 = UINT 0,
            taskReferenceCount_DPU_inv_2 = UINT 0,
            taskReferenceR1_DPU_inv_3 = UINT 0,
            taskReferenceR2_DPU_inv_3 = UINT 0,
            taskReferenceR3_DPU_inv_3 = UINT 0,
            taskReferenceAddr_DPU_inv_3 = UINT 0,
            taskReferenceCount_DPU_inv_3 = UINT 0,
            taskReferenceR1_DPU_inv_4 = UINT 0,
            taskReferenceR2_DPU_inv_4 = UINT 0,
            taskReferenceR3_DPU_inv_4 = UINT 0,
            taskReferenceAddr_DPU_inv_4 = UINT 0,
            taskReferenceCount_DPU_inv_4 = UINT 0,
            taskReferenceR1_DPU_inv_5 = UINT 0,
            taskReferenceR2_DPU_inv_5 = UINT 0,
            taskReferenceR3_DPU_inv_5 = UINT 0,
            taskReferenceAddr_DPU_inv_5 = UINT 0,
            taskReferenceCount_DPU_inv_5 = UINT 0,
            taskReferenceR1_DPU_var_0 = UINT 0,
            taskReferenceR2_DPU_var_0 = UINT 0,
            taskReferenceR3_DPU_var_0 = UINT 0,
            taskReferenceAddr_DPU_var_0 = UINT 0,
            taskReferenceCount_DPU_var_0 = UINT 0,
            taskReferenceR1_DPU_var_1 = UINT 0,
            taskReferenceR2_DPU_var_1 = UINT 0,
            taskReferenceR3_DPU_var_1 = UINT 0,
            taskReferenceAddr_DPU_var_1 = UINT 0,
            taskReferenceCount_DPU_var_1 = UINT 0,
            taskReferenceR1_DPU_var_2 = UINT 0,
            taskReferenceR2_DPU_var_2 = UINT 0,
            taskReferenceR3_DPU_var_2 = UINT 0,
            taskReferenceAddr_DPU_var_2 = UINT 0,
            taskReferenceCount_DPU_var_2 = UINT 0,
            taskReferenceR1_DPU_var_3 = UINT 0,
            taskReferenceR2_DPU_var_3 = UINT 0,
            taskReferenceR3_DPU_var_3 = UINT 0,
            taskReferenceAddr_DPU_var_3 = UINT 0,
            taskReferenceCount_DPU_var_3 = UINT 0,
            taskReferenceR1_DPU_var_4 = UINT 0,
            taskReferenceR2_DPU_var_4 = UINT 0,
            taskReferenceR3_DPU_var_4 = UINT 0,
            taskReferenceAddr_DPU_var_4 = UINT 0,
            taskReferenceCount_DPU_var_4 = UINT 0,
            taskReferenceR1_DPU_var_5 = UINT 0,
            taskReferenceR2_DPU_var_5 = UINT 0,
            taskReferenceR3_DPU_var_5 = UINT 0,
            taskReferenceAddr_DPU_var_5 = UINT 0,
            taskReferenceCount_DPU_var_5 = UINT 0,
            taskReferenceR1_ActKernel_range_0 = UINT 0,
            taskReferenceR2_ActKernel_range_0 = UINT 0,
            taskReferenceR3_ActKernel_range_0 = UINT 0,
            taskReferenceAddr_ActKernel_range_0 = UINT 0,
            taskReferenceCount_ActKernel_range_0 = UINT 0,
            taskReferenceR1_ActKernel_range_1 = UINT 0,
            taskReferenceR2_ActKernel_range_1 = UINT 0,
            taskReferenceR3_ActKernel_range_1 = UINT 0,
            taskReferenceAddr_ActKernel_range_1 = UINT 0,
            taskReferenceCount_ActKernel_range_1 = UINT 0,
            taskReferenceR1_ActKernel_range_2 = UINT 0,
            taskReferenceR2_ActKernel_range_2 = UINT 0,
            taskReferenceR3_ActKernel_range_2 = UINT 0,
            taskReferenceAddr_ActKernel_range_2 = UINT 0,
            taskReferenceCount_ActKernel_range_2 = UINT 0,
            taskReferenceR1_ActKernel_range_3 = UINT 0,
            taskReferenceR2_ActKernel_range_3 = UINT 0,
            taskReferenceR3_ActKernel_range_3 = UINT 0,
            taskReferenceAddr_ActKernel_range_3 = UINT 0,
            taskReferenceCount_ActKernel_range_3 = UINT 0,
            taskReferenceR1_ActKernel_range_4 = UINT 0,
            taskReferenceR2_ActKernel_range_4 = UINT 0,
            taskReferenceR3_ActKernel_range_4 = UINT 0,
            taskReferenceAddr_ActKernel_range_4 = UINT 0,
            taskReferenceCount_ActKernel_range_4 = UINT 0,
            taskReferenceR1_ActKernel_range_5 = UINT 0,
            taskReferenceR2_ActKernel_range_5 = UINT 0,
            taskReferenceR3_ActKernel_range_5 = UINT 0,
            taskReferenceAddr_ActKernel_range_5 = UINT 0,
            taskReferenceCount_ActKernel_range_5 = UINT 0,
            taskReferenceR1_ActKernel_invo_0 = UINT 0,
            taskReferenceR2_ActKernel_invo_0 = UINT 0,
            taskReferenceR3_ActKernel_invo_0 = UINT 0,
            taskReferenceAddr_ActKernel_invo_0 = UINT 0,
            taskReferenceCount_ActKernel_invo_0 = UINT 0,
            taskReferenceR1_ActKernel_invo_1 = UINT 0,
            taskReferenceR2_ActKernel_invo_1 = UINT 0,
            taskReferenceR3_ActKernel_invo_1 = UINT 0,
            taskReferenceAddr_ActKernel_invo_1 = UINT 0,
            taskReferenceCount_ActKernel_invo_1 = UINT 0,
            taskReferenceR1_ActKernel_invo_2 = UINT 0,
            taskReferenceR2_ActKernel_invo_2 = UINT 0,
            taskReferenceR3_ActKernel_invo_2 = UINT 0,
            taskReferenceAddr_ActKernel_invo_2 = UINT 0,
            taskReferenceCount_ActKernel_invo_2 = UINT 0,
            taskReferenceR1_ActKernel_invo_3 = UINT 0,
            taskReferenceR2_ActKernel_invo_3 = UINT 0,
            taskReferenceR3_ActKernel_invo_3 = UINT 0,
            taskReferenceAddr_ActKernel_invo_3 = UINT 0,
            taskReferenceCount_ActKernel_invo_3 = UINT 0,
            taskReferenceR1_ActKernel_invo_4 = UINT 0,
            taskReferenceR2_ActKernel_invo_4 = UINT 0,
            taskReferenceR3_ActKernel_invo_4 = UINT 0,
            taskReferenceAddr_ActKernel_invo_4 = UINT 0,
            taskReferenceCount_ActKernel_invo_4 = UINT 0,
            taskReferenceR1_ActKernel_invo_5 = UINT 0,
            taskReferenceR2_ActKernel_invo_5 = UINT 0,
            taskReferenceR3_ActKernel_invo_5 = UINT 0,
            taskReferenceAddr_ActKernel_invo_5 = UINT 0,
            taskReferenceCount_ActKernel_invo_5 = UINT 0,
            taskReferenceR1_MediaTask = UINT 0,
            taskReferenceR2_MediaTask = UINT 0,
            taskReferenceR3_MediaTask = UINT 0,
            taskReferenceAddr_MediaTask = UINT 0,
            taskReferenceCount_MediaTask = UINT 0,
            taskReferenceR1_BarrierConfig = UINT 0,
            taskReferenceR2_BarrierConfig = UINT 0,
            taskReferenceR3_BarrierConfig = UINT 0,
            taskReferenceAddr_BarrierConfig = UINT 0,
            taskReferenceCount_BarrierConfig = UINT 0,
            MiNNRTCfg_reserved = UINT 0,
            MiNNRTCfg_runtime_entry = UINT 0x1C000000,
            MiNNRTCfg_act_rt_window_base = UINT 0,
            MiNNRTCfg_stack_0 = UINT 0,
            MiNNRTCfg_stack_1 = UINT 0,
            MiNNRTCfg_stack_2 = UINT 0,
            MiNNRTCfg_stack_3 = UINT 0,
            MiNNRTCfg_stack_4 = UINT 0,
            MiNNRTCfg_stack_5 = UINT 0,
            MiNNRTCfg_stack_6 = UINT 0,
            MiNNRTCfg_stack_7 = UINT 0,
            MiNNRTCfg_stack_8 = UINT 0,
            MiNNRTCfg_stack_9 = UINT 0,
            MiNNRTCfg_stack_10 = UINT 0,
            MiNNRTCfg_stack_11 = UINT 0,
            MiNNRTCfg_stack_size = UINT 0,
            MiNNRTCfg_code_window_buffer_size = UINT 0x100000,
            MiNNRTCfg_perf_metrics_mask = UINT 0,
            MiNNRTCfg_runtime_version = UINT 0,
            MiNNRTCfg_use_schedule_embedded_rt = UINT 0,
            MiNNRTCfg_dpu_perf_mode = UINT 3,
            MiNNRTCfg_pad_6 = UINT 0,
            MihHwpWorkpointCfgAddr = UINT 0,
            taskReferenceR1_ManagedInference = UINT 0,
            taskReferenceR2_ManagedInference = UINT 0,
            taskReferenceR3_ManagedInference = UINT 0,
            taskReferenceAddr_ManagedInference = UINT 0,
            taskReferenceCount_ManagedInference = UINT 0,
          } requires 11:4:10
        > {actKernelInvocationsCount = [0, 0, 0, 0, 0, 0], actKernelRangesCount = [0, 0, 0, 0, 0, 0], barrierCount = 0 : i64, dmaCMXCount = [0, 0], dmaCount = [[1, 0], [0, 0]], dmaDDRCount = [1, 0], dmaHwpBase = @buffer.CMX_NN.0::@DeclareBuffer2, dmaTasks = [[@task.dma.0.0::@NNDMA_0_0_0]], elfMemOffsetAttrKey = 0 : ui64, invariantCount = [0, 0, 0, 0, 0, 0], mappedInferenceVersion = @note.MappedInferenceVersion::@MappedInferenceVersion_0_0, mediaCount = 0 : i64, sym_name = "MappedInference", variantCount = [0, 0, 0, 0, 0, 0]}
      }
      ELF.CreateSection @note.LoaderABIVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
        ELF.ABIVersion(1 _ 2 _ 2) {elfMemOffsetAttrKey = 0 : ui64, sym_name = "LoaderABIVersion"}
      }
      ELF.CreateSection @perf.metrics aligned(8) secType(VPU_SHT_PERF_METRICS) secFlags("SHF_NONE") secLocation(<DDR>) {
        ELF.PerformanceMetricsSection {elfMemOffsetAttrKey = 0 : ui64} @PerfMetrics
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>) size(82944) value(1075854336)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(1474560) value(1075937280)
        ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.meta.PlatformInfo of(@meta.PlatformInfo) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.note.MappedInferenceVersion of(@note.MappedInferenceVersion) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.note.LoaderABIVersion of(@note.LoaderABIVersion) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.perf.metrics of(@perf.metrics) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @entry of(@program.mapped_inference::@MappedInference) type(<VPU_STT_ENTRY>) size(0) value(0)
      }
      ELF.CreateSymbolTableSection @symtab.io.NetworkInput secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") {
        ELF.Symbol @elfsym.io.NetworkInput.0 of(@io.NetworkInput.0) type(<STT_SECTION>) size(2880000) value(0)
      }
      ELF.CreateSymbolTableSection @symtab.io.NetworkOutput secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT") {
        ELF.Symbol @elfsym.io.NetworkOutput.0 of(@io.NetworkOutput.0) type(<STT_SECTION>) size(2880000) value(0)
      }
      ELF.CreateMetadataSection @MetadataSection aligned(8) secFlags("SHF_NONE") {
        VPUASM.NetworkMetadata @NetworkMetadata
      }
      ELF.CreateRelocationSection @rela.task.dma.0.0.symtab.io.NetworkInput target(@task.dma.0.0) symtab(@symtab.io.NetworkInput) secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") {
        ELF.Reloc offset(40) sourceSym(@symtab.io.NetworkInput::@elfsym.io.NetworkInput.0) relocType(<R_VPU_64>) addend(0) (description : "Input in NNDMA reloc")
      }
      ELF.CreateRelocationSection @rela.task.dma.0.0.symtab.io.NetworkOutput target(@task.dma.0.0) symtab(@symtab.io.NetworkOutput) secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT") {
        ELF.Reloc offset(48) sourceSym(@symtab.io.NetworkOutput::@elfsym.io.NetworkOutput.0) relocType(<R_VPU_64>) addend(0) (description : "Output (firstOutputBuff) in NNDMA reloc")
      }
      ELF.CreateRelocationSection @rela.program.mapped_inference.symtab target(@program.mapped_inference) symtab(@symtab) secFlags("SHF_NONE") {
        ELF.Reloc offset(88) sourceSym(@symtab::@elfsym.task.dma.0.0) relocType(<R_VPU_64>) addend(0) (description : "Dma list in mapped inference reloc")
        ELF.Reloc offset(16) sourceSym(@symtab::@elfsym.buffer.CMX_NN.0) relocType(<R_VPU_64>) addend(0) (description : "dmaHwpBase in mapped inference reloc")
      }
    }
    return
  }
}

  func.func @main(%arg0: memref<1x720x1000x16xf16>, %arg1: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> {
    %c90 = arith.constant 90 : index
    %c720 = arith.constant 720 : index
    %c0 = arith.constant 0 : index
    %0 = arith.subi %c720, %c0 : index
    %1 = arith.divsi %0, %c90 : index
    %2 = async.create_group %1 : !async.group
    scf.for %arg3 = %c0 to %c720 step %c90 {
      %subview = memref.subview %arg0[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %3 = builtin.unrealized_conversion_cast %subview : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %subview1 = memref.subview %arg1[0, %arg3, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %4 = builtin.unrealized_conversion_cast %subview1 : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %token, %bodyResults = async.execute -> !async.value<memref<1x90x1000x16xf16>> {
        %5 = Core.NestedCall @OneDMAWithoutAttributes::@main1(%3, %4) : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
        async.yield %5 : memref<1x90x1000x16xf16>
      }
      %6 = async.add_to_group %token, %2 : !async.token
      %7 = async.await %bodyResults : !async.value<memref<1x90x1000x16xf16>>
    }
    async.await_all %2
    return %arg1 : memref<1x720x1000x16xf16>
  }

  // CHECK:   HostExec.Binary @OneDMAWithoutAttributes {
  // CHECK:   HostExec.BinaryData @serialized_main
  // CHECK-SAME:   <object = "\7FELF\02\01\00\00\00\{{.*}}">
  // CHECK:   func.func private @main1(memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
  // CHECK:   }
  // CHECK:   func.func @main(%arg0: memref<1x720x1000x16xf16>, %arg1: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> {
  // CHECK:   [[IN0:%.*]] = builtin.unrealized_conversion_cast %subview : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
  // CHECK:   [[OUT0:%.*]] = builtin.unrealized_conversion_cast %subview_0 : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
  // CHECK:   [[RESULT:%.*]] = Core.NestedCall @OneDMAWithoutAttributes::@main1([[IN0]], %4) : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
  // CHECK:   async.yield [[OUT0]] : memref<1x90x1000x16xf16>
}
