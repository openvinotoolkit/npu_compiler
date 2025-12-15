//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <npu_40xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"

using namespace npu40xx;
using namespace vpux::NPUReg50XX;

class NPUReg50XX_DMARegisterTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuDMATask, vpux::NPUReg50XX::Descriptors::DMARegister> {};

#define TEST_NPU5_DMA_REG_FIELD(FieldType, DescriptorMember)                                                   \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_DMARegisterTest, FieldType, vpux::NPUReg50XX::Fields::FieldType, \
                                   DescriptorMember, 0)

#define TEST_NPU5_DMA_MULTIPLE_REGS_FIELD(ParentRegType, FieldType, DescriptorMember)             \
    HELPER_TEST_NPU_MULTIPLE_REGS_FIELD(NPUReg50XX_DMARegisterTest, ParentRegType##__##FieldType, \
                                        vpux::NPUReg50XX::Registers::ParentRegType,               \
                                        vpux::NPUReg50XX::Fields::FieldType, DescriptorMember, 0)

TEST_NPU5_DMA_REG_FIELD(dma_watermark, transaction_.watermark)
TEST_NPU5_DMA_REG_FIELD(dma_link_address, transaction_.link_address)
TEST_NPU5_DMA_REG_FIELD(dma_lra, transaction_.lra)
TEST_NPU5_DMA_REG_FIELD(dma_lba_addr, transaction_.lba_addr)
TEST_NPU5_DMA_REG_FIELD(dma_src_aub, transaction_.src_aub)
TEST_NPU5_DMA_REG_FIELD(dma_dst_aub, transaction_.dst_aub)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_num_dim, transaction_.cfg.fields.num_dim)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_int_en, transaction_.cfg.fields.int_en)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_int_id, transaction_.cfg.fields.int_id)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_src_burst_length, transaction_.cfg.fields.src_burst_length)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_dst_burst_length, transaction_.cfg.fields.dst_burst_length)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_arb_qos, transaction_.cfg.fields.arb_qos)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_ord, transaction_.cfg.fields.ord)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_barrier_en, transaction_.cfg.fields.barrier_en)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_memset_en, transaction_.cfg.fields.memset_en)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_atp_en, transaction_.cfg.fields.atp_en)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_watermark_en, transaction_.cfg.fields.watermark_en)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_rwf_en, transaction_.cfg.fields.rwf_en)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_rws_en, transaction_.cfg.fields.rws_en)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_src_list_cfg, transaction_.cfg.fields.src_list_cfg)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_dst_list_cfg, transaction_.cfg.fields.dst_list_cfg)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_conversion_cfg, transaction_.cfg.fields.conversion_cfg)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_acceleration_cfg, transaction_.cfg.fields.acceleration_cfg)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_tile4_cfg, transaction_.cfg.fields.tile4_cfg)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_axi_user_bits_cfg, transaction_.cfg.fields.axi_user_bits_cfg)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_hwp_id_en, transaction_.cfg.fields.hwp_id_en)
TEST_NPU5_DMA_REG_FIELD(dma_cfg_fields_hwp_id, transaction_.cfg.fields.hwp_id)
TEST_NPU5_DMA_REG_FIELD(dma_remote_width_fetch, transaction_.remote_width_fetch)
TEST_NPU5_DMA_REG_FIELD(dma_width_src, transaction_.width.src)
TEST_NPU5_DMA_REG_FIELD(dma_width_dst, transaction_.width.dst)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_compress_dtype, transaction_.acc_info.compress.dtype)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_compress_sparse, transaction_.acc_info.compress.sparse)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_compress_bitc_en, transaction_.acc_info.compress.bitc_en)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_compress_z, transaction_.acc_info.compress.z)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_compress_bitmap_buf_sz, transaction_.acc_info.compress.bitmap_buf_sz)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_compress_bitmap_base_addr, transaction_.acc_info.compress.bitmap_base_addr)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_decompress_dtype, transaction_.acc_info.decompress.dtype)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_decompress_sparse, transaction_.acc_info.decompress.sparse)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_decompress_bitc_en, transaction_.acc_info.decompress.bitc_en)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_decompress_z, transaction_.acc_info.decompress.z)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_decompress_bitmap_base_addr, transaction_.acc_info.decompress.bitmap_base_addr)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_w_prep_dtype, transaction_.acc_info.w_prep.dtype)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_w_prep_sparse, transaction_.acc_info.w_prep.sparse)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_w_prep_zeropoint, transaction_.acc_info.w_prep.zeropoint)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_w_prep_ic, transaction_.acc_info.w_prep.ic)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_w_prep_filtersize, transaction_.acc_info.w_prep.filtersize)
TEST_NPU5_DMA_REG_FIELD(dma_acc_info_w_prep_bitmap_base_addr, transaction_.acc_info.w_prep.bitmap_base_addr)
TEST_NPU5_DMA_REG_FIELD(dma_mset_data, transaction_.mset_data)
TEST_NPU5_DMA_REG_FIELD(dma_src, transaction_.src)
TEST_NPU5_DMA_REG_FIELD(dma_sra, transaction_.sra)
TEST_NPU5_DMA_REG_FIELD(dma_dst, transaction_.dst)
TEST_NPU5_DMA_REG_FIELD(dma_dra, transaction_.dra)
TEST_NPU5_DMA_REG_FIELD(dma_sba_addr, transaction_.sba_addr)
TEST_NPU5_DMA_REG_FIELD(dma_dba_addr, transaction_.dba_addr)
TEST_NPU5_DMA_REG_FIELD(dma_barrier_prod_mask_lower, transaction_.barrier.prod_mask_lower)
TEST_NPU5_DMA_REG_FIELD(dma_barrier_cons_mask_lower, transaction_.barrier.cons_mask_lower)
TEST_NPU5_DMA_REG_FIELD(dma_barrier_prod_mask_upper, transaction_.barrier.prod_mask_upper)
TEST_NPU5_DMA_REG_FIELD(dma_barrier_cons_mask_upper, transaction_.barrier.cons_mask_upper)
TEST_NPU5_DMA_REG_FIELD(dma_list_size_src, transaction_.list_size.src)
TEST_NPU5_DMA_REG_FIELD(dma_list_size_dst, transaction_.list_size.dst)
TEST_NPU5_DMA_REG_FIELD(dma_dim_size_src_1, transaction_.dim_size_1.src)
TEST_NPU5_DMA_REG_FIELD(dma_dim_size_dst_1, transaction_.dim_size_1.dst)
TEST_NPU5_DMA_REG_FIELD(dma_stride_src_1, transaction_.stride_src_1)
TEST_NPU5_DMA_REG_FIELD(dma_stride_dst_1, transaction_.stride_dst_1)
TEST_NPU5_DMA_REG_FIELD(dma_dim_size_src_2, transaction_.dim_size_2.src)
TEST_NPU5_DMA_REG_FIELD(dma_dim_size_dst_2, transaction_.dim_size_2.dst)
TEST_NPU5_DMA_REG_FIELD(dma_list_addr_src, transaction_.list_addr.src)
TEST_NPU5_DMA_REG_FIELD(dma_list_addr_dst, transaction_.list_addr.dst)
TEST_NPU5_DMA_REG_FIELD(dma_stride_src_2, transaction_.stride_src_2)
TEST_NPU5_DMA_REG_FIELD(dma_stride_dst_2, transaction_.stride_dst_2)
TEST_NPU5_DMA_REG_FIELD(dma_remote_width_store, transaction_.remote_width_store)
TEST_NPU5_DMA_REG_FIELD(dma_dim_size_src_3, transaction_.dim_size_src_3)
TEST_NPU5_DMA_REG_FIELD(dma_dim_size_src_4, transaction_.dim_size_src_4)
TEST_NPU5_DMA_REG_FIELD(dma_dim_size_dst_3, transaction_.dim_size_dst_3)
TEST_NPU5_DMA_REG_FIELD(dma_dim_size_dst_4, transaction_.dim_size_dst_4)
TEST_NPU5_DMA_REG_FIELD(dma_dim_size_src_5, transaction_.dim_size_src_5)
TEST_NPU5_DMA_REG_FIELD(dma_dim_size_dst_5, transaction_.dim_size_dst_5)
TEST_NPU5_DMA_REG_FIELD(dma_stride_src_3, transaction_.stride_src_3)
TEST_NPU5_DMA_REG_FIELD(dma_stride_dst_3, transaction_.stride_dst_3)
TEST_NPU5_DMA_REG_FIELD(dma_stride_src_4, transaction_.stride_src_4)
TEST_NPU5_DMA_REG_FIELD(dma_stride_dst_4, transaction_.stride_dst_4)
TEST_NPU5_DMA_REG_FIELD(dma_stride_src_5, transaction_.stride_src_5)
TEST_NPU5_DMA_REG_FIELD(dma_stride_dst_5, transaction_.stride_dst_5)
TEST_NPU5_DMA_MULTIPLE_REGS_FIELD(dma_barriers_sched, start_after_, barriers_sched_.start_after_)
TEST_NPU5_DMA_MULTIPLE_REGS_FIELD(dma_barriers_sched, clean_after_, barriers_sched_.clean_after_)
