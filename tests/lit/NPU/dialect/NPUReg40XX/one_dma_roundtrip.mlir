//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// RUN: vpux-opt --emit-bytecode --init-compiler="vpu-arch=%arch%" %s | vpux-opt --vpu-arch=%arch% | FileCheck %s
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
            dma_remote_width_fetch = UINT 0,
            dma_width {
              UINT dma_width_src = 0x30,
              UINT dma_width_dst = 0x30,
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
              UINT start_after_ = 1,
              UINT clean_after_ = 2,
            }
            dma_pad_24_0 = UINT 0,
            dma_pad_24_1 = UINT 0,
            dma_pad_24_2 = UINT 0,
          }
        > {input = @DeclareBuffer0, output_buffs = [@DeclareBuffer1], sym_name = "NNDMA_0_0_0"}
      }
    }
    return
  }
}

// CHECK: NPUReg40XX.NNDMA descriptor = <
// CHECK: DMARegister {
// CHECK:   dma_watermark {
// CHECK:     UINT dma_watermark = 0,
// CHECK:   }
// CHECK:   dma_link_address {
// CHECK:     UINT dma_link_address = 0,
// CHECK:   }
// CHECK:   dma_lra {
// CHECK:     UINT dma_lra = 0,
// CHECK:   }
// CHECK:   dma_lba_addr = UINT 0,
// CHECK:   dma_src_aub = UINT 0,
// CHECK:   dma_dst_aub = UINT 0,
// CHECK:   dma_cfg_fields {
// CHECK:     UINT dma_cfg_fields_num_dim = 0,
// CHECK:     UINT dma_cfg_fields_int_en = 0,
// CHECK:     UINT dma_cfg_fields_int_id = 0,
// CHECK:     UINT dma_cfg_fields_src_burst_length = 0xF,
// CHECK:     UINT dma_cfg_fields_dst_burst_length = 0xF,
// CHECK:     UINT dma_cfg_fields_arb_qos = 0xFF,
// CHECK:     UINT dma_cfg_fields_ord = 1,
// CHECK:     UINT dma_cfg_fields_barrier_en = 1,
// CHECK:     UINT dma_cfg_fields_memset_en = 0,
// CHECK:     UINT dma_cfg_fields_atp_en = 1,
// CHECK:     UINT dma_cfg_fields_watermark_en = 0,
// CHECK:     UINT dma_cfg_fields_rwf_en = 0,
// CHECK:     UINT dma_cfg_fields_rws_en = 0,
// CHECK:     UINT dma_cfg_fields_src_list_cfg = 0,
// CHECK:     UINT dma_cfg_fields_dst_list_cfg = 0,
// CHECK:     UINT dma_cfg_fields_conversion_cfg = 0,
// CHECK:     UINT dma_cfg_fields_acceleration_cfg = 0,
// CHECK:     UINT dma_cfg_fields_tile4_cfg = 0,
// CHECK:     UINT dma_cfg_fields_axi_user_bits_cfg = 0,
// CHECK:     UINT dma_cfg_fields_hwp_id_en = 1,
// CHECK:     UINT dma_cfg_fields_hwp_id = 0,
// CHECK:     UINT dma_cfg_fields_reserved = 0,
// CHECK:   }
// CHECK:   dma_remote_width_fetch = UINT 0x30,
// CHECK:   dma_width {
// CHECK:     UINT dma_width_src = 0x30,
// CHECK:     UINT dma_width_dst = 0x30,
// CHECK:   }
// CHECK:   dma_acc_info_compress {
// CHECK:     UINT dma_acc_info_compress_dtype = 0,
// CHECK:     UINT dma_acc_info_compress_reserved1 = 0,
// CHECK:     UINT dma_acc_info_compress_sparse = 0,
// CHECK:     UINT dma_acc_info_compress_bitc_en = 0,
// CHECK:     UINT dma_acc_info_compress_z = 0,
// CHECK:     UINT dma_acc_info_compress_bitmap_buf_sz = 0,
// CHECK:     UINT dma_acc_info_compress_reserved2 = 0,
// CHECK:     UINT dma_acc_info_compress_bitmap_base_addr = 0,
// CHECK:   }
// CHECK:   dma_acc_info_decompress {
// CHECK:     UINT dma_acc_info_decompress_dtype = 0,
// CHECK:     UINT dma_acc_info_decompress_reserved1 = 0,
// CHECK:     UINT dma_acc_info_decompress_sparse = 0,
// CHECK:     UINT dma_acc_info_decompress_bitc_en = 0,
// CHECK:     UINT dma_acc_info_decompress_z = 0,
// CHECK:     UINT dma_acc_info_decompress_reserved2 = 0,
// CHECK:     UINT dma_acc_info_decompress_bitmap_base_addr = 0,
// CHECK:   }
// CHECK:   dma_acc_info_w_prep {
// CHECK:     UINT dma_acc_info_w_prep_dtype = 0,
// CHECK:     UINT dma_acc_info_w_prep_reserved1 = 0,
// CHECK:     UINT dma_acc_info_w_prep_sparse = 0,
// CHECK:     UINT dma_acc_info_w_prep_zeropoint = 0,
// CHECK:     UINT dma_acc_info_w_prep_ic = 0,
// CHECK:     UINT dma_acc_info_w_prep_filtersize = 0,
// CHECK:     UINT dma_acc_info_w_prep_reserved2 = 0,
// CHECK:     UINT dma_acc_info_w_prep_bitmap_base_addr = 0,
// CHECK:   }
// CHECK:   dma_mset_data = UINT 0,
// CHECK:   dma_src_addr {
// CHECK:     UINT dma_src = 0,
// CHECK:     UINT dma_sra = 0,
// CHECK:   }
// CHECK:   dma_dst_addr {
// CHECK:     UINT dma_dst = 0,
// CHECK:     UINT dma_dra = 0,
// CHECK:   }
// CHECK:   dma_sba_addr = UINT 0,
// CHECK:   dma_dba_addr = UINT 0,
// CHECK:   dma_barrier_prod_mask_lower = UINT 0,
// CHECK:   dma_barrier_cons_mask_lower = UINT 0,
// CHECK:   dma_barrier_prod_mask_upper {
// CHECK:     UINT dma_barrier_prod_mask_upper = 0,
// CHECK:   }
// CHECK:   dma_barrier_cons_mask_upper {
// CHECK:     UINT dma_barrier_cons_mask_upper = 0,
// CHECK:   }
// CHECK:   dma_list_size {
// CHECK:     UINT dma_list_size_src = 0,
// CHECK:     UINT dma_list_size_dst = 0,
// CHECK:   }
// CHECK:   dma_dim_size {
// CHECK:     UINT dma_dim_size_src_1 = 0,
// CHECK:     UINT dma_dim_size_dst_1 = 0,
// CHECK:   }
// CHECK:   dma_stride_src_1 = UINT 0,
// CHECK:   dma_stride_dst_1 = UINT 0,
// CHECK:   dma_dim_size_2 {
// CHECK:     UINT dma_dim_size_src_2 = 0,
// CHECK:     UINT dma_dim_size_dst_2 = 0,
// CHECK:   }
// CHECK:   dma_list_addr {
// CHECK:     UINT dma_list_addr_src = 0,
// CHECK:     UINT dma_list_addr_dst = 0,
// CHECK:   }
// CHECK:   dma_stride_src_2 = UINT 0,
// CHECK:   dma_stride_dst_2 = UINT 0,
// CHECK:   dma_remote_width_store = UINT 0,
// CHECK:   dma_dim_size_src_3 = UINT 0,
// CHECK:   dma_dim_size_src_4 = UINT 0,
// CHECK:   dma_dim_size_dst_3 = UINT 0,
// CHECK:   dma_dim_size_dst_4 = UINT 0,
// CHECK:   dma_dim_size_src_5 = UINT 0,
// CHECK:   dma_dim_size_dst_5 = UINT 0,
// CHECK:   dma_stride_src_3 = UINT 0,
// CHECK:   dma_stride_dst_3 = UINT 0,
// CHECK:   dma_stride_src_4 = UINT 0,
// CHECK:   dma_stride_dst_4 = UINT 0,
// CHECK:   dma_stride_src_5 = UINT 0,
// CHECK:   dma_stride_dst_5 = UINT 0,
// CHECK:   dma_word_21_reserved = UINT 0,
// CHECK:   dma_word_22_reserved = UINT 0,
// CHECK:   dma_word_23_reserved = UINT 0,
// CHECK:   dma_barriers_sched {
// CHECK:     UINT start_after_ = 1,
// CHECK:     UINT clean_after_ = 2,
// CHECK:   }
// CHECK:   dma_pad_24_0 = UINT 0,
// CHECK:   dma_pad_24_1 = UINT 0,
// CHECK:   dma_pad_24_2 = UINT 0,
// CHECK: } requires 11:4:10
// CHECK: > {input = @DeclareBuffer0, output_buffs = [@DeclareBuffer1], sym_name = "NNDMA_0_0_0"}

// -----

module @OneDMAWithCustomVersions {
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
            dma_remote_width_fetch = UINT 0,
            dma_width {
              UINT dma_width_src = 0x30,
              UINT dma_width_dst = 0x30,
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
              UINT start_after_ = 1,
              UINT clean_after_ = 2,
            }
            dma_pad_24_0 = UINT 0,
            dma_pad_24_1 = UINT 0,
            dma_pad_24_2 = UINT 0,
          } requires 11:4:10
        > {input = @DeclareBuffer0, output_buffs = [@DeclareBuffer1], sym_name = "NNDMA_0_0_0"}
      }
    }
    return
  }
}

// CHECK: NPUReg40XX.NNDMA descriptor = <
// CHECK: DMARegister {
// CHECK:   dma_watermark {
// CHECK:     UINT dma_watermark = 0,
// CHECK:   }
// CHECK:   dma_link_address {
// CHECK:     UINT dma_link_address = 0,
// CHECK:   }
// CHECK:   dma_lra {
// CHECK:     UINT dma_lra = 0,
// CHECK:   }
// CHECK:   dma_lba_addr = UINT 0,
// CHECK:   dma_src_aub = UINT 0,
// CHECK:   dma_dst_aub = UINT 0,
// CHECK:   dma_cfg_fields {
// CHECK:     UINT dma_cfg_fields_num_dim = 0,
// CHECK:     UINT dma_cfg_fields_int_en = 0,
// CHECK:     UINT dma_cfg_fields_int_id = 0,
// CHECK:     UINT dma_cfg_fields_src_burst_length = 0xF,
// CHECK:     UINT dma_cfg_fields_dst_burst_length = 0xF,
// CHECK:     UINT dma_cfg_fields_arb_qos = 0xFF,
// CHECK:     UINT dma_cfg_fields_ord = 1,
// CHECK:     UINT dma_cfg_fields_barrier_en = 1,
// CHECK:     UINT dma_cfg_fields_memset_en = 0,
// CHECK:     UINT dma_cfg_fields_atp_en = 1,
// CHECK:     UINT dma_cfg_fields_watermark_en = 0,
// CHECK:     UINT dma_cfg_fields_rwf_en = 0,
// CHECK:     UINT dma_cfg_fields_rws_en = 0,
// CHECK:     UINT dma_cfg_fields_src_list_cfg = 0,
// CHECK:     UINT dma_cfg_fields_dst_list_cfg = 0,
// CHECK:     UINT dma_cfg_fields_conversion_cfg = 0,
// CHECK:     UINT dma_cfg_fields_acceleration_cfg = 0,
// CHECK:     UINT dma_cfg_fields_tile4_cfg = 0,
// CHECK:     UINT dma_cfg_fields_axi_user_bits_cfg = 0,
// CHECK:     UINT dma_cfg_fields_hwp_id_en = 1,
// CHECK:     UINT dma_cfg_fields_hwp_id = 0,
// CHECK:     UINT dma_cfg_fields_reserved = 0,
// CHECK:   }
// CHECK:   dma_remote_width_fetch = UINT 0x30,
// CHECK:   dma_width {
// CHECK:     UINT dma_width_src = 0x30,
// CHECK:     UINT dma_width_dst = 0x30,
// CHECK:   }
// CHECK:   dma_acc_info_compress {
// CHECK:     UINT dma_acc_info_compress_dtype = 0,
// CHECK:     UINT dma_acc_info_compress_reserved1 = 0,
// CHECK:     UINT dma_acc_info_compress_sparse = 0,
// CHECK:     UINT dma_acc_info_compress_bitc_en = 0,
// CHECK:     UINT dma_acc_info_compress_z = 0,
// CHECK:     UINT dma_acc_info_compress_bitmap_buf_sz = 0,
// CHECK:     UINT dma_acc_info_compress_reserved2 = 0,
// CHECK:     UINT dma_acc_info_compress_bitmap_base_addr = 0,
// CHECK:   }
// CHECK:   dma_acc_info_decompress {
// CHECK:     UINT dma_acc_info_decompress_dtype = 0,
// CHECK:     UINT dma_acc_info_decompress_reserved1 = 0,
// CHECK:     UINT dma_acc_info_decompress_sparse = 0,
// CHECK:     UINT dma_acc_info_decompress_bitc_en = 0,
// CHECK:     UINT dma_acc_info_decompress_z = 0,
// CHECK:     UINT dma_acc_info_decompress_reserved2 = 0,
// CHECK:     UINT dma_acc_info_decompress_bitmap_base_addr = 0,
// CHECK:   }
// CHECK:   dma_acc_info_w_prep {
// CHECK:     UINT dma_acc_info_w_prep_dtype = 0,
// CHECK:     UINT dma_acc_info_w_prep_reserved1 = 0,
// CHECK:     UINT dma_acc_info_w_prep_sparse = 0,
// CHECK:     UINT dma_acc_info_w_prep_zeropoint = 0,
// CHECK:     UINT dma_acc_info_w_prep_ic = 0,
// CHECK:     UINT dma_acc_info_w_prep_filtersize = 0,
// CHECK:     UINT dma_acc_info_w_prep_reserved2 = 0,
// CHECK:     UINT dma_acc_info_w_prep_bitmap_base_addr = 0,
// CHECK:   }
// CHECK:   dma_mset_data = UINT 0,
// CHECK:   dma_src_addr {
// CHECK:     UINT dma_src = 0,
// CHECK:     UINT dma_sra = 0,
// CHECK:   }
// CHECK:   dma_dst_addr {
// CHECK:     UINT dma_dst = 0,
// CHECK:     UINT dma_dra = 0,
// CHECK:   }
// CHECK:   dma_sba_addr = UINT 0,
// CHECK:   dma_dba_addr = UINT 0,
// CHECK:   dma_barrier_prod_mask_lower = UINT 0,
// CHECK:   dma_barrier_cons_mask_lower = UINT 0,
// CHECK:   dma_barrier_prod_mask_upper {
// CHECK:     UINT dma_barrier_prod_mask_upper = 0,
// CHECK:   }
// CHECK:   dma_barrier_cons_mask_upper {
// CHECK:     UINT dma_barrier_cons_mask_upper = 0,
// CHECK:   }
// CHECK:   dma_list_size {
// CHECK:     UINT dma_list_size_src = 0,
// CHECK:     UINT dma_list_size_dst = 0,
// CHECK:   }
// CHECK:   dma_dim_size {
// CHECK:     UINT dma_dim_size_src_1 = 0,
// CHECK:     UINT dma_dim_size_dst_1 = 0,
// CHECK:   }
// CHECK:   dma_stride_src_1 = UINT 0,
// CHECK:   dma_stride_dst_1 = UINT 0,
// CHECK:   dma_dim_size_2 {
// CHECK:     UINT dma_dim_size_src_2 = 0,
// CHECK:     UINT dma_dim_size_dst_2 = 0,
// CHECK:   }
// CHECK:   dma_list_addr {
// CHECK:     UINT dma_list_addr_src = 0,
// CHECK:     UINT dma_list_addr_dst = 0,
// CHECK:   }
// CHECK:   dma_stride_src_2 = UINT 0,
// CHECK:   dma_stride_dst_2 = UINT 0,
// CHECK:   dma_remote_width_store = UINT 0,
// CHECK:   dma_dim_size_src_3 = UINT 0,
// CHECK:   dma_dim_size_src_4 = UINT 0,
// CHECK:   dma_dim_size_dst_3 = UINT 0,
// CHECK:   dma_dim_size_dst_4 = UINT 0,
// CHECK:   dma_dim_size_src_5 = UINT 0,
// CHECK:   dma_dim_size_dst_5 = UINT 0,
// CHECK:   dma_stride_src_3 = UINT 0,
// CHECK:   dma_stride_dst_3 = UINT 0,
// CHECK:   dma_stride_src_4 = UINT 0,
// CHECK:   dma_stride_dst_4 = UINT 0,
// CHECK:   dma_stride_src_5 = UINT 0,
// CHECK:   dma_stride_dst_5 = UINT 0,
// CHECK:   dma_word_21_reserved = UINT 0,
// CHECK:   dma_word_22_reserved = UINT 0,
// CHECK:   dma_word_23_reserved = UINT 0,
// CHECK:   dma_barriers_sched {
// CHECK:     UINT start_after_ = 1,
// CHECK:     UINT clean_after_ = 2,
// CHECK:   }
// CHECK:   dma_pad_24_0 = UINT 0,
// CHECK:   dma_pad_24_1 = UINT 0,
// CHECK:   dma_pad_24_2 = UINT 0,
// CHECK: } requires 11:4:10
// CHECK: > {input = @DeclareBuffer0, output_buffs = [@DeclareBuffer1], sym_name = "NNDMA_0_0_0"}
