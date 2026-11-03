//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --add-relocations-for-dynamic-strides-dmas %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

VPUASM.InputBindings inputDeclarations : {
  VPUASM.DeclareBuffer @"input1" !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x4x6xui8, @DDR> : swizzling(0)>
  VPUASM.DeclareBuffer @"input2" !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x4x6xui8, @DDR> : swizzling(0)>
}

VPUASM.OutputBindings outputDeclarations : {
  VPUASM.DeclareBuffer @"output1" !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x4x6xui8, @DDR> : swizzling(0)>
  VPUASM.DeclareBuffer @"output2" !VPUASM.Buffer< "DDR"[0] <0> : memref<1x4x6xui8, @DDR> : swizzling(0)>
}

func.func @DynamicStrides() {
  ELF.Main {
    ELF.CreateLogicalSection @io.NetworkInput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_3 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x4x6xui8, @DDR> : swizzling(0)>
    }
    ELF.CreateLogicalSection @io.NetworkInput.1 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_4 !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x4x6xui8, @DDR> : swizzling(0)>
    }
    ELF.CreateLogicalSection @io.NetworkOutput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USEROUTPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkOutput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_5 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x4x6xui8, @DDR> : swizzling(0)>
    }
    ELF.CreateLogicalSection @buffer.DDR.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
      VPUASM.DeclareBuffer @DeclareBuffer_12 !VPUASM.Buffer< "DDR"[0] <0> : memref<1x4x6xui8, @DDR> : swizzling(0)>
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
            UINT dma_cfg_fields_num_dim = 3,
            UINT dma_cfg_fields_int_en = 0,
            UINT dma_cfg_fields_int_id = 0,
            UINT dma_cfg_fields_src_burst_length = 0xF,
            UINT dma_cfg_fields_dst_burst_length = 0xF,
            UINT dma_cfg_fields_arb_qos = 0xFF,
            UINT dma_cfg_fields_ord = 0,
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
          dma_remote_width_fetch = UINT 1,
          dma_width {
            UINT dma_width_src = 1,
            UINT dma_width_dst = 0x18,
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
          dma_barrier_prod_mask_lower = UINT 2,
          dma_barrier_cons_mask_lower = UINT 0,
          dma_barrier_prod_mask_upper {
            UINT dma_barrier_prod_mask_upper = 0,
          }
          dma_barrier_cons_mask_upper {
            UINT dma_barrier_cons_mask_upper = 0,
          }
          dma_list_size {
            UINT dma_list_size_src = 5,
            UINT dma_list_size_dst = 0,
          }
          dma_dim_size {
            UINT dma_dim_size_src_1 = 5,
            UINT dma_dim_size_dst_1 = 0,
          }
          dma_stride_src_1 = UINT 1,
          dma_stride_dst_1 = UINT 0,
          dma_dim_size_2 {
            UINT dma_dim_size_src_2 = 3,
            UINT dma_dim_size_dst_2 = 0,
          }
          dma_list_addr {
            UINT dma_list_addr_src = 3,
            UINT dma_list_addr_dst = 0,
          }
          dma_stride_src_2 = UINT 6,
          dma_stride_dst_2 = UINT 0,
          dma_remote_width_store = UINT 0,
          dma_dim_size_src_3 = UINT 0,
          dma_dim_size_src_4 = UINT 0,
          dma_dim_size_dst_3 = UINT 0,
          dma_dim_size_dst_4 = UINT 0,
          dma_dim_size_src_5 = UINT 0,
          dma_dim_size_dst_5 = UINT 0,
          dma_stride_src_3 = UINT 0x18,
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
        } requires 11:10:0
      > {directLink, elfMemOffsetAttrKey = 2464 : ui64, input = @io.NetworkInput.0::@DeclareBuffer_3, next_link = @task.dma.0.0::@NNDMA_0_0_16, output_buffs = [@buffer.DDR.0::@DeclareBuffer_12], stridedInput, sym_name = "NNDMA_0_0_11"}
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
            UINT dma_cfg_fields_num_dim = 3,
            UINT dma_cfg_fields_int_en = 0,
            UINT dma_cfg_fields_int_id = 0,
            UINT dma_cfg_fields_src_burst_length = 0xF,
            UINT dma_cfg_fields_dst_burst_length = 0xF,
            UINT dma_cfg_fields_arb_qos = 0xFF,
            UINT dma_cfg_fields_ord = 0,
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
          dma_remote_width_fetch = UINT 0x18,
          dma_width {
            UINT dma_width_src = 0x18,
            UINT dma_width_dst = 1,
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
          dma_barrier_prod_mask_lower = UINT 0x20,
          dma_barrier_cons_mask_lower = UINT 0x10,
          dma_barrier_prod_mask_upper {
            UINT dma_barrier_prod_mask_upper = 0,
          }
          dma_barrier_cons_mask_upper {
            UINT dma_barrier_cons_mask_upper = 0,
          }
          dma_list_size {
            UINT dma_list_size_src = 0,
            UINT dma_list_size_dst = 5,
          }
          dma_dim_size {
            UINT dma_dim_size_src_1 = 0,
            UINT dma_dim_size_dst_1 = 5,
          }
          dma_stride_src_1 = UINT 0,
          dma_stride_dst_1 = UINT 1,
          dma_dim_size_2 {
            UINT dma_dim_size_src_2 = 0,
            UINT dma_dim_size_dst_2 = 3,
          }
          dma_list_addr {
            UINT dma_list_addr_src = 0,
            UINT dma_list_addr_dst = 3,
          }
          dma_stride_src_2 = UINT 0,
          dma_stride_dst_2 = UINT 6,
          dma_remote_width_store = UINT 6,
          dma_dim_size_src_3 = UINT 0,
          dma_dim_size_src_4 = UINT 0,
          dma_dim_size_dst_3 = UINT 0,
          dma_dim_size_dst_4 = UINT 0,
          dma_dim_size_src_5 = UINT 0,
          dma_dim_size_dst_5 = UINT 0,
          dma_stride_src_3 = UINT 0,
          dma_stride_dst_3 = UINT 0x18,
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
        } requires 11:10:0
      > {directLink, elfMemOffsetAttrKey = 3584 : ui64, input = @buffer.DDR.0::@DeclareBuffer_12, output_buffs = [@io.NetworkOutput.0::@DeclareBuffer_5], stridedOutput, sym_name = "NNDMA_0_0_16"}
    }
    ELF.CreateSection @task.dma.1.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
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
            UINT dma_cfg_fields_num_dim = 3,
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
          dma_remote_width_fetch = UINT 1,
          dma_width {
            UINT dma_width_src = 1,
            UINT dma_width_dst = 0x18,
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
          dma_barrier_prod_mask_lower = UINT 2,
          dma_barrier_cons_mask_lower = UINT 1,
          dma_barrier_prod_mask_upper {
            UINT dma_barrier_prod_mask_upper = 0,
          }
          dma_barrier_cons_mask_upper {
            UINT dma_barrier_cons_mask_upper = 0,
          }
          dma_list_size {
            UINT dma_list_size_src = 5,
            UINT dma_list_size_dst = 0,
          }
          dma_dim_size {
            UINT dma_dim_size_src_1 = 5,
            UINT dma_dim_size_dst_1 = 0,
          }
          dma_stride_src_1 = UINT 1,
          dma_stride_dst_1 = UINT 0,
          dma_dim_size_2 {
            UINT dma_dim_size_src_2 = 3,
            UINT dma_dim_size_dst_2 = 0,
          }
          dma_list_addr {
            UINT dma_list_addr_src = 3,
            UINT dma_list_addr_dst = 0,
          }
          dma_stride_src_2 = UINT 6,
          dma_stride_dst_2 = UINT 0,
          dma_remote_width_store = UINT 0,
          dma_dim_size_src_3 = UINT 0,
          dma_dim_size_src_4 = UINT 0,
          dma_dim_size_dst_3 = UINT 0,
          dma_dim_size_dst_4 = UINT 0,
          dma_dim_size_src_5 = UINT 0,
          dma_dim_size_dst_5 = UINT 0,
          dma_stride_src_3 = UINT 0x18,
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
        } requires 11:10:0
      > {directLink, elfMemOffsetAttrKey = 0 : ui64, input = @io.NetworkInput.1::@DeclareBuffer_4, output_buffs = [@buffer.DDR.0::@DeclareBuffer_12], stridedInput, sym_name = "NNDMA_1_0_0"}
    }

    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.buffer.DDR.0 of(@buffer.DDR.0) type(<STT_SECTION>)
      ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>)
      ELF.Symbol @elfsym.task.dma.1.0 of(@task.dma.1.0) type(<STT_SECTION>)
    }
  }
  return

  // CHECK:       ELF.DmaSymbolSection @dmaInputSymbolsSection secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") {
  // CHECK:       ELF.DmaSymbol
  // CHECK-SAME:    @INPUT_RELOC_0
  // CHECK-SAME:    io_index 0
  // CHECK-SAME:    tensor_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    tensor_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    dma_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    dma_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    tile_offsets [0, 0, 0, 0, 0, 0]
  // CHECK-SAME:    dma_size 1
  // CHECK:       ELF.DmaSymbol
  // CHECK-SAME:    @INPUT_RELOC_1
  // CHECK-SAME:    io_index 1
  // CHECK-SAME:    tensor_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    tensor_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    dma_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    dma_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    tile_offsets [0, 0, 0, 0, 0, 0]
  // CHECK-SAME:    dma_size 1
  // CHECK:       }

  // CHECK:       ELF.DmaSymbolSection @dmaOutputSymbolsSection secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT") {
  // CHECK:       ELF.DmaSymbol
  // CHECK-SAME:    @OUTPUT_RELOC_0
  // CHECK-SAME:    io_index 0
  // CHECK-SAME:    tensor_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    tensor_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    dma_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    dma_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    tile_offsets [0, 0, 0, 0, 0, 0]
  // CHECK-SAME:    dma_size 1
  // CHECK:       }

  // CHECK:       ELF.CreateRelocationSection @rela.task.dma.0.0.dmaInputSymbolsSection
  // CHECK-SAME:    target(@task.dma.0.0)
  // CHECK-SAME:    symtab(@dmaInputSymbolsSection)
  // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT")
  // CHECK:         ELF.Reloc
  // CHECK-SAME:      offset(2464)
  // CHECK-SAME:      sourceSym(@dmaInputSymbolsSection::@INPUT_RELOC_0)
  // CHECK-SAME:      relocType(<R_VPU_DMA_DESCRIPTOR_INPUT>)
  // CHECK-SAME:      addend(0)
  // CHECK-SAME:      (description : "DMA descriptor relocation for strided input")

  // CHECK:       ELF.CreateRelocationSection @rela.task.dma.0.0.dmaOutputSymbolsSection
  // CHECK-SAME:    target(@task.dma.0.0)
  // CHECK-SAME:    symtab(@dmaOutputSymbolsSection)
  // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT")
  // CHECK:         ELF.Reloc
  // CHECK-SAME:      offset(3584)
  // CHECK-SAME:      sourceSym(@dmaOutputSymbolsSection::@OUTPUT_RELOC_0)
  // CHECK-SAME:      relocType(<R_VPU_DMA_DESCRIPTOR_OUTPUT>)
  // CHECK-SAME:      addend(0)
  // CHECK-SAME:      (description : "DMA descriptor relocation for strided output")

  // CHECK:       ELF.CreateRelocationSection @rela.task.dma.1.0.dmaInputSymbolsSection
  // CHECK-SAME:    target(@task.dma.1.0)
  // CHECK-SAME:    symtab(@dmaInputSymbolsSection)
  // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT")
  // CHECK:         ELF.Reloc
  // CHECK-SAME:      offset(0)
  // CHECK-SAME:      sourceSym(@dmaInputSymbolsSection::@INPUT_RELOC_1)
  // CHECK-SAME:      relocType(<R_VPU_DMA_DESCRIPTOR_INPUT>)
  // CHECK-SAME:      addend(0)
  // CHECK-SAME:      (description : "DMA descriptor relocation for strided input")
}

// -----

VPUASM.InputBindings inputDeclarations : {
  VPUASM.DeclareBuffer @"input1" !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<8x12xui8, @DDR> : swizzling(0)>
}

VPUASM.OutputBindings outputDeclarations : {
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DynamicStridesWithUnitExpandPermute() {
  ELF.Main {

    // CHECK: ELF.DmaSymbolSection @dmaInputSymbolsSection
    // CHECK-NEXT: ELF.DmaSymbol @INPUT_RELOC_0 io_index 0 tensor_shapes [12, 8, 1, 1, 1, 1] tensor_strides [1, 12, 0, 0, 0, 0]
    // CHECK-SAME: dma_shapes [6, 4, 1, 1, 1, 1] dma_strides [1, 12, 0, 0, 0, 0] tile_offsets [6, 4, 0, 0, 0, 0] dma_size 1

    ELF.CreateLogicalSection @io.NetworkInput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_3 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<8x12xui8, @DDR> : swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer_6 {offsets = [0, 6, 4, 0]} !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x6x4x1xui8, {order = #NHWC, strides = [96, 1, 12, 12]}, @DDR> : swizzling(0)>
    }
    ELF.CreateLogicalSection @buffer.DDR.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
      VPUASM.DeclareBuffer @DeclareBuffer_12 !VPUASM.Buffer< "DDR"[0] <0> : memref<1x6x4x1xui8, {order = #NHWC, strides = [96, 1, 12, 12]}, @DDR> : swizzling(0)>
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
            UINT dma_cfg_fields_num_dim = 3,
            UINT dma_cfg_fields_int_en = 0,
            UINT dma_cfg_fields_int_id = 0,
            UINT dma_cfg_fields_src_burst_length = 0xF,
            UINT dma_cfg_fields_dst_burst_length = 0xF,
            UINT dma_cfg_fields_arb_qos = 0xFF,
            UINT dma_cfg_fields_ord = 0,
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
          dma_remote_width_fetch = UINT 1,
          dma_width {
            UINT dma_width_src = 1,
            UINT dma_width_dst = 0x18,
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
          dma_barrier_prod_mask_lower = UINT 2,
          dma_barrier_cons_mask_lower = UINT 0,
          dma_barrier_prod_mask_upper {
            UINT dma_barrier_prod_mask_upper = 0,
          }
          dma_barrier_cons_mask_upper {
            UINT dma_barrier_cons_mask_upper = 0,
          }
          dma_list_size {
            UINT dma_list_size_src = 5,
            UINT dma_list_size_dst = 0,
          }
          dma_dim_size {
            UINT dma_dim_size_src_1 = 5,
            UINT dma_dim_size_dst_1 = 0,
          }
          dma_stride_src_1 = UINT 1,
          dma_stride_dst_1 = UINT 0,
          dma_dim_size_2 {
            UINT dma_dim_size_src_2 = 3,
            UINT dma_dim_size_dst_2 = 0,
          }
          dma_list_addr {
            UINT dma_list_addr_src = 3,
            UINT dma_list_addr_dst = 0,
          }
          dma_stride_src_2 = UINT 6,
          dma_stride_dst_2 = UINT 0,
          dma_remote_width_store = UINT 0,
          dma_dim_size_src_3 = UINT 0,
          dma_dim_size_src_4 = UINT 0,
          dma_dim_size_dst_3 = UINT 0,
          dma_dim_size_dst_4 = UINT 0,
          dma_dim_size_src_5 = UINT 0,
          dma_dim_size_dst_5 = UINT 0,
          dma_stride_src_3 = UINT 0x18,
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
        } requires 11:10:0
      > {directLink, elfMemOffsetAttrKey = 2464 : ui64, input = @io.NetworkInput.0::@DeclareBuffer_6, next_link = @task.dma.0.0::@NNDMA_0_0_16, output_buffs = [@buffer.DDR.0::@DeclareBuffer_12], stridedInput, sym_name = "NNDMA_0_0_11"}
    }

    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.buffer.DDR.0 of(@buffer.DDR.0) type(<STT_SECTION>)
      ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>)
    }
  }
  return
}

// -----

// Sub-byte (< 8 bit) element type tensors use the *_BIT relocation variants and
// report dma_size in bits rather than bytes.

VPUASM.InputBindings inputDeclarations : {
  VPUASM.DeclareBuffer @"input1" !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x4x6xsi4, @DDR> : swizzling(0)>
  VPUASM.DeclareBuffer @"input2" !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x4x6xsi4, @DDR> : swizzling(0)>
}

VPUASM.OutputBindings outputDeclarations : {
  VPUASM.DeclareBuffer @"output1" !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x4x6xsi4, @DDR> : swizzling(0)>
  VPUASM.DeclareBuffer @"output2" !VPUASM.Buffer< "DDR"[0] <0> : memref<1x4x6xsi4, @DDR> : swizzling(0)>
}

func.func @DynamicStridesSubByte() {
  ELF.Main {
    ELF.CreateLogicalSection @io.NetworkInput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_3 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x4x6xsi4, @DDR> : swizzling(0)>
    }
    ELF.CreateLogicalSection @io.NetworkInput.1 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_4 !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x4x6xsi4, @DDR> : swizzling(0)>
    }
    ELF.CreateLogicalSection @io.NetworkOutput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USEROUTPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkOutput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_5 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x4x6xsi4, @DDR> : swizzling(0)>
    }
    ELF.CreateLogicalSection @buffer.DDR.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
      VPUASM.DeclareBuffer @DeclareBuffer_12 !VPUASM.Buffer< "DDR"[0] <0> : memref<1x4x6xsi4, @DDR> : swizzling(0)>
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
            UINT dma_cfg_fields_num_dim = 3,
            UINT dma_cfg_fields_int_en = 0,
            UINT dma_cfg_fields_int_id = 0,
            UINT dma_cfg_fields_src_burst_length = 0xF,
            UINT dma_cfg_fields_dst_burst_length = 0xF,
            UINT dma_cfg_fields_arb_qos = 0xFF,
            UINT dma_cfg_fields_ord = 0,
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
          dma_remote_width_fetch = UINT 1,
          dma_width {
            UINT dma_width_src = 1,
            UINT dma_width_dst = 0x18,
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
          dma_barrier_prod_mask_lower = UINT 2,
          dma_barrier_cons_mask_lower = UINT 0,
          dma_barrier_prod_mask_upper {
            UINT dma_barrier_prod_mask_upper = 0,
          }
          dma_barrier_cons_mask_upper {
            UINT dma_barrier_cons_mask_upper = 0,
          }
          dma_list_size {
            UINT dma_list_size_src = 5,
            UINT dma_list_size_dst = 0,
          }
          dma_dim_size {
            UINT dma_dim_size_src_1 = 5,
            UINT dma_dim_size_dst_1 = 0,
          }
          dma_stride_src_1 = UINT 1,
          dma_stride_dst_1 = UINT 0,
          dma_dim_size_2 {
            UINT dma_dim_size_src_2 = 3,
            UINT dma_dim_size_dst_2 = 0,
          }
          dma_list_addr {
            UINT dma_list_addr_src = 3,
            UINT dma_list_addr_dst = 0,
          }
          dma_stride_src_2 = UINT 6,
          dma_stride_dst_2 = UINT 0,
          dma_remote_width_store = UINT 0,
          dma_dim_size_src_3 = UINT 0,
          dma_dim_size_src_4 = UINT 0,
          dma_dim_size_dst_3 = UINT 0,
          dma_dim_size_dst_4 = UINT 0,
          dma_dim_size_src_5 = UINT 0,
          dma_dim_size_dst_5 = UINT 0,
          dma_stride_src_3 = UINT 0x18,
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
        } requires 11:10:0
      > {directLink, elfMemOffsetAttrKey = 2464 : ui64, input = @io.NetworkInput.0::@DeclareBuffer_3, next_link = @task.dma.0.0::@NNDMA_0_0_16, output_buffs = [@buffer.DDR.0::@DeclareBuffer_12], stridedInput, sym_name = "NNDMA_0_0_11"}
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
            UINT dma_cfg_fields_num_dim = 3,
            UINT dma_cfg_fields_int_en = 0,
            UINT dma_cfg_fields_int_id = 0,
            UINT dma_cfg_fields_src_burst_length = 0xF,
            UINT dma_cfg_fields_dst_burst_length = 0xF,
            UINT dma_cfg_fields_arb_qos = 0xFF,
            UINT dma_cfg_fields_ord = 0,
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
          dma_remote_width_fetch = UINT 0x18,
          dma_width {
            UINT dma_width_src = 0x18,
            UINT dma_width_dst = 1,
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
          dma_barrier_prod_mask_lower = UINT 0x20,
          dma_barrier_cons_mask_lower = UINT 0x10,
          dma_barrier_prod_mask_upper {
            UINT dma_barrier_prod_mask_upper = 0,
          }
          dma_barrier_cons_mask_upper {
            UINT dma_barrier_cons_mask_upper = 0,
          }
          dma_list_size {
            UINT dma_list_size_src = 0,
            UINT dma_list_size_dst = 5,
          }
          dma_dim_size {
            UINT dma_dim_size_src_1 = 0,
            UINT dma_dim_size_dst_1 = 5,
          }
          dma_stride_src_1 = UINT 0,
          dma_stride_dst_1 = UINT 1,
          dma_dim_size_2 {
            UINT dma_dim_size_src_2 = 0,
            UINT dma_dim_size_dst_2 = 3,
          }
          dma_list_addr {
            UINT dma_list_addr_src = 0,
            UINT dma_list_addr_dst = 3,
          }
          dma_stride_src_2 = UINT 0,
          dma_stride_dst_2 = UINT 6,
          dma_remote_width_store = UINT 6,
          dma_dim_size_src_3 = UINT 0,
          dma_dim_size_src_4 = UINT 0,
          dma_dim_size_dst_3 = UINT 0,
          dma_dim_size_dst_4 = UINT 0,
          dma_dim_size_src_5 = UINT 0,
          dma_dim_size_dst_5 = UINT 0,
          dma_stride_src_3 = UINT 0,
          dma_stride_dst_3 = UINT 0x18,
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
        } requires 11:10:0
      > {directLink, elfMemOffsetAttrKey = 3584 : ui64, input = @buffer.DDR.0::@DeclareBuffer_12, output_buffs = [@io.NetworkOutput.0::@DeclareBuffer_5], stridedOutput, sym_name = "NNDMA_0_0_16"}
    }
    ELF.CreateSection @task.dma.1.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
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
            UINT dma_cfg_fields_num_dim = 3,
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
          dma_remote_width_fetch = UINT 1,
          dma_width {
            UINT dma_width_src = 1,
            UINT dma_width_dst = 0x18,
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
          dma_barrier_prod_mask_lower = UINT 2,
          dma_barrier_cons_mask_lower = UINT 1,
          dma_barrier_prod_mask_upper {
            UINT dma_barrier_prod_mask_upper = 0,
          }
          dma_barrier_cons_mask_upper {
            UINT dma_barrier_cons_mask_upper = 0,
          }
          dma_list_size {
            UINT dma_list_size_src = 5,
            UINT dma_list_size_dst = 0,
          }
          dma_dim_size {
            UINT dma_dim_size_src_1 = 5,
            UINT dma_dim_size_dst_1 = 0,
          }
          dma_stride_src_1 = UINT 1,
          dma_stride_dst_1 = UINT 0,
          dma_dim_size_2 {
            UINT dma_dim_size_src_2 = 3,
            UINT dma_dim_size_dst_2 = 0,
          }
          dma_list_addr {
            UINT dma_list_addr_src = 3,
            UINT dma_list_addr_dst = 0,
          }
          dma_stride_src_2 = UINT 6,
          dma_stride_dst_2 = UINT 0,
          dma_remote_width_store = UINT 0,
          dma_dim_size_src_3 = UINT 0,
          dma_dim_size_src_4 = UINT 0,
          dma_dim_size_dst_3 = UINT 0,
          dma_dim_size_dst_4 = UINT 0,
          dma_dim_size_src_5 = UINT 0,
          dma_dim_size_dst_5 = UINT 0,
          dma_stride_src_3 = UINT 0x18,
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
        } requires 11:10:0
      > {directLink, elfMemOffsetAttrKey = 0 : ui64, input = @io.NetworkInput.1::@DeclareBuffer_4, output_buffs = [@buffer.DDR.0::@DeclareBuffer_12], stridedInput, sym_name = "NNDMA_1_0_0"}
    }

    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.buffer.DDR.0 of(@buffer.DDR.0) type(<STT_SECTION>)
      ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>)
      ELF.Symbol @elfsym.task.dma.1.0 of(@task.dma.1.0) type(<STT_SECTION>)
    }
  }
  return

  // CHECK:       ELF.DmaSymbolSection @dmaInputSymbolsSection secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") {
  // CHECK:       ELF.DmaSymbol
  // CHECK-SAME:    @INPUT_RELOC_0
  // CHECK-SAME:    io_index 0
  // CHECK-SAME:    tensor_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    tensor_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    dma_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    dma_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    tile_offsets [0, 0, 0, 0, 0, 0]
  // CHECK-SAME:    dma_size 4
  // CHECK:       ELF.DmaSymbol
  // CHECK-SAME:    @INPUT_RELOC_1
  // CHECK-SAME:    io_index 1
  // CHECK-SAME:    tensor_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    tensor_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    dma_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    dma_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    tile_offsets [0, 0, 0, 0, 0, 0]
  // CHECK-SAME:    dma_size 4
  // CHECK:       }

  // CHECK:       ELF.DmaSymbolSection @dmaOutputSymbolsSection secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT") {
  // CHECK:       ELF.DmaSymbol
  // CHECK-SAME:    @OUTPUT_RELOC_0
  // CHECK-SAME:    io_index 0
  // CHECK-SAME:    tensor_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    tensor_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    dma_shapes [6, 4, 1, 1, 1, 1]
  // CHECK-SAME:    dma_strides [1, 6, 24, 0, 0, 0]
  // CHECK-SAME:    tile_offsets [0, 0, 0, 0, 0, 0]
  // CHECK-SAME:    dma_size 4
  // CHECK:       }

  // CHECK:       ELF.CreateRelocationSection @rela.task.dma.0.0.dmaInputSymbolsSection
  // CHECK-SAME:    target(@task.dma.0.0)
  // CHECK-SAME:    symtab(@dmaInputSymbolsSection)
  // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT")
  // CHECK:         ELF.Reloc
  // CHECK-SAME:      offset(2464)
  // CHECK-SAME:      sourceSym(@dmaInputSymbolsSection::@INPUT_RELOC_0)
  // CHECK-SAME:      relocType(<R_VPU_DMA_DESCRIPTOR_INPUT_BIT>)
  // CHECK-SAME:      addend(0)
  // CHECK-SAME:      (description : "DMA descriptor relocation for strided input")

  // CHECK:       ELF.CreateRelocationSection @rela.task.dma.0.0.dmaOutputSymbolsSection
  // CHECK-SAME:    target(@task.dma.0.0)
  // CHECK-SAME:    symtab(@dmaOutputSymbolsSection)
  // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USEROUTPUT")
  // CHECK:         ELF.Reloc
  // CHECK-SAME:      offset(3584)
  // CHECK-SAME:      sourceSym(@dmaOutputSymbolsSection::@OUTPUT_RELOC_0)
  // CHECK-SAME:      relocType(<R_VPU_DMA_DESCRIPTOR_OUTPUT_BIT>)
  // CHECK-SAME:      addend(0)
  // CHECK-SAME:      (description : "DMA descriptor relocation for strided output")

  // CHECK:       ELF.CreateRelocationSection @rela.task.dma.1.0.dmaInputSymbolsSection
  // CHECK-SAME:    target(@task.dma.1.0)
  // CHECK-SAME:    symtab(@dmaInputSymbolsSection)
  // CHECK-SAME:    secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT")
  // CHECK:         ELF.Reloc
  // CHECK-SAME:      offset(0)
  // CHECK-SAME:      sourceSym(@dmaInputSymbolsSection::@INPUT_RELOC_1)
  // CHECK-SAME:      relocType(<R_VPU_DMA_DESCRIPTOR_INPUT_BIT>)
  // CHECK-SAME:      addend(0)
  // CHECK-SAME:      (description : "DMA descriptor relocation for strided input")
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

VPUASM.InputBindings inputDeclarations : {
  VPUASM.DeclareBuffer @"input1" !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1024xui8, @DDR> : swizzling(0)>
}

VPUASM.OutputBindings outputDeclarations : {
}

func.func @DynamicStridesWithUnitContractAndExpandClipOnlyNon1Dim() {
  ELF.Main {

    // CHECK: ELF.DmaSymbolSection @dmaInputSymbolsSection
    // CHECK-NEXT: ELF.DmaSymbol @INPUT_RELOC_0 io_index 0 tensor_shapes [1024, 1, 1, 1, 1, 1] tensor_strides [1, 1024, 0, 0, 0, 0]
    // CHECK-SAME: dma_shapes [1024, 1, 1, 1, 1, 1] dma_strides [1, 0, 0, 0, 0, 0] tile_offsets [0, 0, 0, 0, 0, 0] dma_size 1

    ELF.CreateLogicalSection @io.NetworkInput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_3 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x1024xui8, @DDR> : swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer_6 {offsets = [0, 0]} !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1024x1xui8, @DDR> : swizzling(0)>
    }
    ELF.CreateLogicalSection @buffer.DDR.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
      VPUASM.DeclareBuffer @DeclareBuffer_12 !VPUASM.Buffer< "DDR"[0] <0> : memref<1024x1xui8, @DDR> : swizzling(0)>
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
            UINT dma_cfg_fields_num_dim = 3,
            UINT dma_cfg_fields_int_en = 0,
            UINT dma_cfg_fields_int_id = 0,
            UINT dma_cfg_fields_src_burst_length = 0xF,
            UINT dma_cfg_fields_dst_burst_length = 0xF,
            UINT dma_cfg_fields_arb_qos = 0xFF,
            UINT dma_cfg_fields_ord = 0,
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
          dma_remote_width_fetch = UINT 1,
          dma_width {
            UINT dma_width_src = 1,
            UINT dma_width_dst = 0x18,
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
          dma_barrier_prod_mask_lower = UINT 2,
          dma_barrier_cons_mask_lower = UINT 0,
          dma_barrier_prod_mask_upper {
            UINT dma_barrier_prod_mask_upper = 0,
          }
          dma_barrier_cons_mask_upper {
            UINT dma_barrier_cons_mask_upper = 0,
          }
          dma_list_size {
            UINT dma_list_size_src = 5,
            UINT dma_list_size_dst = 0,
          }
          dma_dim_size {
            UINT dma_dim_size_src_1 = 5,
            UINT dma_dim_size_dst_1 = 0,
          }
          dma_stride_src_1 = UINT 1,
          dma_stride_dst_1 = UINT 0,
          dma_dim_size_2 {
            UINT dma_dim_size_src_2 = 3,
            UINT dma_dim_size_dst_2 = 0,
          }
          dma_list_addr {
            UINT dma_list_addr_src = 3,
            UINT dma_list_addr_dst = 0,
          }
          dma_stride_src_2 = UINT 6,
          dma_stride_dst_2 = UINT 0,
          dma_remote_width_store = UINT 0,
          dma_dim_size_src_3 = UINT 0,
          dma_dim_size_src_4 = UINT 0,
          dma_dim_size_dst_3 = UINT 0,
          dma_dim_size_dst_4 = UINT 0,
          dma_dim_size_src_5 = UINT 0,
          dma_dim_size_dst_5 = UINT 0,
          dma_stride_src_3 = UINT 0x18,
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
        } requires 11:10:0
      > {directLink, elfMemOffsetAttrKey = 2464 : ui64, input = @io.NetworkInput.0::@DeclareBuffer_6, next_link = @task.dma.0.0::@NNDMA_0_0_16, output_buffs = [@buffer.DDR.0::@DeclareBuffer_12], stridedInput, sym_name = "NNDMA_0_0_11"}
    }

    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.buffer.DDR.0 of(@buffer.DDR.0) type(<STT_SECTION>)
      ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>)
    }
  }
  return
}

// -----

VPUASM.InputBindings inputDeclarations : {
  VPUASM.DeclareBuffer @"input1" !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<6x8x4x12x20x100xui8, @DDR> : swizzling(0)>
}

VPUASM.OutputBindings outputDeclarations : {
}

func.func @DynamicStrides6DDMA() {
  ELF.Main {

    // CHECK: ELF.DmaSymbolSection @dmaInputSymbolsSection
    // CHECK-NEXT: ELF.DmaSymbol @INPUT_RELOC_0 io_index 0 tensor_shapes [100, 20, 12, 4, 8, 6] tensor_strides [1, 100, 2000, 24000, 96000, 768000]
    // CHECK-SAME: dma_shapes [50, 10, 6, 2, 4, 3] dma_strides [1, 100, 2000, 24000, 96000, 768000] tile_offsets [2, 2, 2, 2, 2, 2] dma_size 1

    ELF.CreateLogicalSection @io.NetworkInput.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_USERINPUT|VPU_SHF_PROC_DMA") secLocation(<NetworkInput>) {
      VPUASM.DeclareBuffer @DeclareBuffer_3 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<6x8x4x12x20x100xui8, @DDR> : swizzling(0)>
      VPUASM.DeclareBuffer @DeclareBuffer_6 {offsets = [2, 2, 2, 2, 2, 2]} !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<3x4x2x6x10x50xui8, {order = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>, strides = [768000, 96000, 24000, 2000, 100, 1]}, @DDR> : swizzling(0)>
    }
    ELF.CreateLogicalSection @buffer.DDR.0 aligned(64) secType(SHT_NOBITS) secFlags("SHF_WRITE|SHF_ALLOC|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
      VPUASM.DeclareBuffer @DeclareBuffer_12 !VPUASM.Buffer< "DDR"[0] <0> : memref<1x1x4x6xui8, @DDR> : swizzling(0)>
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
            UINT dma_cfg_fields_num_dim = 3,
            UINT dma_cfg_fields_int_en = 0,
            UINT dma_cfg_fields_int_id = 0,
            UINT dma_cfg_fields_src_burst_length = 0xF,
            UINT dma_cfg_fields_dst_burst_length = 0xF,
            UINT dma_cfg_fields_arb_qos = 0xFF,
            UINT dma_cfg_fields_ord = 0,
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
          dma_remote_width_fetch = UINT 1,
          dma_width {
            UINT dma_width_src = 1,
            UINT dma_width_dst = 0x18,
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
          dma_barrier_prod_mask_lower = UINT 2,
          dma_barrier_cons_mask_lower = UINT 0,
          dma_barrier_prod_mask_upper {
            UINT dma_barrier_prod_mask_upper = 0,
          }
          dma_barrier_cons_mask_upper {
            UINT dma_barrier_cons_mask_upper = 0,
          }
          dma_list_size {
            UINT dma_list_size_src = 5,
            UINT dma_list_size_dst = 0,
          }
          dma_dim_size {
            UINT dma_dim_size_src_1 = 5,
            UINT dma_dim_size_dst_1 = 0,
          }
          dma_stride_src_1 = UINT 1,
          dma_stride_dst_1 = UINT 0,
          dma_dim_size_2 {
            UINT dma_dim_size_src_2 = 3,
            UINT dma_dim_size_dst_2 = 0,
          }
          dma_list_addr {
            UINT dma_list_addr_src = 3,
            UINT dma_list_addr_dst = 0,
          }
          dma_stride_src_2 = UINT 6,
          dma_stride_dst_2 = UINT 0,
          dma_remote_width_store = UINT 0,
          dma_dim_size_src_3 = UINT 0,
          dma_dim_size_src_4 = UINT 0,
          dma_dim_size_dst_3 = UINT 0,
          dma_dim_size_dst_4 = UINT 0,
          dma_dim_size_src_5 = UINT 0,
          dma_dim_size_dst_5 = UINT 0,
          dma_stride_src_3 = UINT 0x18,
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
        } requires 11:10:0
      > {directLink, elfMemOffsetAttrKey = 2464 : ui64, input = @io.NetworkInput.0::@DeclareBuffer_6, next_link = @task.dma.0.0::@NNDMA_0_0_16, output_buffs = [@buffer.DDR.0::@DeclareBuffer_12], stridedInput, sym_name = "NNDMA_0_0_11"}
    }

    ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
      ELF.Symbol @elfsym.buffer.DDR.0 of(@buffer.DDR.0) type(<STT_SECTION>)
      ELF.Symbol @elfsym.task.dma.0.0 of(@task.dma.0.0) type(<STT_SECTION>)
    }
  }
  return
}
