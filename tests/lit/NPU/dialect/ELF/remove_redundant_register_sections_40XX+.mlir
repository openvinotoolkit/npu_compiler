//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --remove-redundant-register-sections %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

func.func @removeUnusedRegisterSection() {
  // CHECK-LABEL: @removeUnusedRegisterSection
  ELF.Main {
    ELF.CreateLogicalSection @reg.Register.0 aligned(1) secType(SHT_NOBITS) secFlags("SHF_NONE") secLocation(<Register>) {
      VPUASM.DeclareBuffer @DeclareBuffer_40 !VPUASM.Buffer< "Register"[0] <788578304> : memref<1xui32, @Register> :  swizzling(0)>
    }
    // CHECK-NOT: ELF.CreateLogicalSection @reg.Register.0
    ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
      NPUReg50XX.NNDMA descriptor = <
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
            UINT dma_cfg_fields_num_dim = 1,
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
            UINT dma_cfg_fields_hwp_skip = 1,
            UINT dma_cfg_fields_reserved = 0,
          }
          dma_remote_width_fetch = UINT 0x300,
          dma_width {
            UINT dma_width_src = 0x300,
            UINT dma_width_dst = 0x10,
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
            UINT dma_acc_info_decompress_bitmap_buf_sz = 0,
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
            UINT dma_dst = 0x2F010000,
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
            UINT dma_list_size_dst = 0x2F,
          }
          dma_dim_size {
            UINT dma_dim_size_src_1 = 0,
            UINT dma_dim_size_dst_1 = 0x2F,
          }
          dma_stride_src_1 = UINT 0,
          dma_stride_dst_1 = UINT 0x20,
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
            UINT clean_after_ = 0,
          }
          dma_pad_24_0 = UINT 0,
          dma_pad_24_1 = UINT 0,
          dma_pad_24_2 = UINT 0,
        } requires 11:13:0
      > {directLink, elfMemOffsetAttrKey = 0 : ui64, input = @reg.Register.0::@DeclareBuffer_40, next_link = @task.dma.0.0::@NNDMA_0_0_1, output_buffs = [@reg.Register.0::@DeclareBuffer_40], sym_name = "NNDMA_0_0_0"}
      // CHECK: > {directLink, elfMemOffsetAttrKey = 0 : ui64, next_link = @task.dma.0.0::@NNDMA_0_0_1, output_buffs = [], sym_name = "NNDMA_0_0_0"}
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.reg.Register.0 of(@reg.Register.0) type(<STT_SECTION>)
    }
    // CHECK-NOT: ELF.Symbol @elfsym.reg.Register.0 of(@reg.Register.0) type(<STT_SECTION>)
  }
  return
}

// -----

func.func @keepRegisterSectionUsedByRelocation() {
  // CHECK-LABEL: @keepRegisterSectionUsedByRelocation
  ELF.Main {
    ELF.CreateLogicalSection @reg.Register.0 aligned(1) secType(SHT_NOBITS) secFlags("SHF_NONE") secLocation(<Register>) {
      VPUASM.DeclareBuffer @DeclareBuffer_0 !VPUASM.Buffer< "Register"[0] <788578304> : memref<1xui32, @Register> : swizzling(0)>
    }
    // CHECK: ELF.CreateLogicalSection @reg.Register.0
    ELF.CreateSection @task.dma.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.reg.Register.0 of(@reg.Register.0) type(<STT_SECTION>)
    }
    ELF.CreateRelocationSection @relocs target(@task.dma.0.0) symtab(@symtab) secFlags("SHF_NONE") {
      ELF.Reloc offset(0) sourceSym(@symtab::@elfsym.reg.Register.0) relocType(<R_VPU_64>) addend(0)
    }
    // CHECK: ELF.Symbol @elfsym.reg.Register.0 of(@reg.Register.0) type(<STT_SECTION>)
    // CHECK: ELF.Reloc offset(0) sourceSym(@symtab::@elfsym.reg.Register.0) relocType(<R_VPU_64>) addend(0)
  }
  return
}

// -----

func.func @keepRegisterSectionUsedByNonDmaOp() {
  // CHECK-LABEL: @keepRegisterSectionUsedByNonDmaOp
  ELF.Main {
    ELF.CreateLogicalSection @reg.Register.0 aligned(1) secType(SHT_NOBITS) secFlags("SHF_NONE") secLocation(<Register>) {
      VPUASM.DeclareBuffer @DeclareBuffer_0 !VPUASM.Buffer< "Register"[0] <788578304> : memref<1xui32, @Register> : swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer_1 !VPUASM.Buffer< "Register"[0] <788578336> : memref<1xui32, @Register> : swizzling(0)>
    }
    // CHECK: ELF.CreateLogicalSection @reg.Register.0
    ELF.CreateSection @text.KernelParams aligned(1024) secType(SHT_PROGBITS) secFlags(SHF_NONE) secLocation(<DDR>) {
      VPUASM.KernelParams @KernelParams0 inputs([@reg.Register.0::@DeclareBuffer_0]) outputs([@reg.Register.0::@DeclareBuffer_1]) dynamicInputShapes([]) dynamicOutputShapes([])
      kernel_type("test_kernel") < {inputDimsBinaryVector = [], inputStridesBinaryVector = [],
      kernel_params = [0], outputDimsBinaryVector = [], outputStridesBinaryVector = []}>
      // CHECK: VPUASM.KernelParams @KernelParams0 inputs([@reg.Register.0::@DeclareBuffer_0]) outputs([@reg.Register.0::@DeclareBuffer_1])
    }
    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.reg.Register.0 of(@reg.Register.0) type(<STT_SECTION>)
    }
    // CHECK: ELF.Symbol @elfsym.reg.Register.0 of(@reg.Register.0) type(<STT_SECTION>)
  }
  return
}
