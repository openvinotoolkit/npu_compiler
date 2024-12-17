//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <bitset>
#include <random>

#include <gtest/gtest.h>

#include <npu_40xx_nnrt.hpp>

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/descriptors.hpp"

using namespace npu40xx;

struct NPUReg40XX_DMARegisterTest : public testing::Test {
    NPUReg40XX_DMARegisterTest() = default;

    void SetUp() override {
        actual = {};
        reference = {};
    }

    bool isContentEqual() const {
        const auto actualStorage = actual.getStorage();
        return std::memcmp(&reference, actualStorage.data(), sizeof(reference)) == 0;
    }

    vpux::NPUReg40XX::Descriptors::DMARegister actual;
    nn_public::VpuDMATask reference;

    // to test big vs little endian use random set bits distribution
    // so that for a big enough fields there would be consequent bytes
    // with different values
    template <size_t bitSize>
    static uint64_t generateTestValue() {
        std::mt19937 generator(42);
        std::uniform_int_distribution<> distribution(0, 1);
        std::stringstream bitSequence;
        for ([[maybe_unused]] auto _ : vpux::irange(bitSize)) {
            bitSequence << distribution(generator);
        }
        return std::bitset<bitSize>(bitSequence.str()).to_ullong();
    }
};

#define TEST_REGISTER_COMBINATION(FieldType, DescriptorMember)                                     \
    TEST_F(NPUReg40XX_DMARegisterTest, FieldType##Test) {                                          \
        ASSERT_EQ(sizeof(reference), actual.size());                                               \
                                                                                                   \
        const auto value = generateTestValue<vpux::NPUReg40XX::Fields::FieldType::size.count()>(); \
        actual.write<vpux::NPUReg40XX::Fields::FieldType>(value);                                  \
        reference.DescriptorMember = value;                                                        \
        const auto actualValue = actual.read<vpux::NPUReg40XX::Fields::FieldType>();               \
        ASSERT_EQ(actualValue, value);                                                             \
        ASSERT_TRUE(isContentEqual());                                                             \
    }

TEST_REGISTER_COMBINATION(dma_watermark, transaction_.watermark)
TEST_REGISTER_COMBINATION(dma_link_address, transaction_.link_address)
TEST_REGISTER_COMBINATION(dma_lra, transaction_.lra)
TEST_REGISTER_COMBINATION(dma_lba_addr, transaction_.lba_addr)
TEST_REGISTER_COMBINATION(dma_src_aub, transaction_.src_aub)
TEST_REGISTER_COMBINATION(dma_dst_aub, transaction_.dst_aub)
TEST_REGISTER_COMBINATION(dma_cfg_fields_num_dim, transaction_.cfg.fields.num_dim)
TEST_REGISTER_COMBINATION(dma_cfg_fields_int_en, transaction_.cfg.fields.int_en)
TEST_REGISTER_COMBINATION(dma_cfg_fields_int_id, transaction_.cfg.fields.int_id)
TEST_REGISTER_COMBINATION(dma_cfg_fields_src_burst_length, transaction_.cfg.fields.src_burst_length)
TEST_REGISTER_COMBINATION(dma_cfg_fields_dst_burst_length, transaction_.cfg.fields.dst_burst_length)
TEST_REGISTER_COMBINATION(dma_cfg_fields_arb_qos, transaction_.cfg.fields.arb_qos)
TEST_REGISTER_COMBINATION(dma_cfg_fields_ord, transaction_.cfg.fields.ord)
TEST_REGISTER_COMBINATION(dma_cfg_fields_barrier_en, transaction_.cfg.fields.barrier_en)
TEST_REGISTER_COMBINATION(dma_cfg_fields_memset_en, transaction_.cfg.fields.memset_en)
TEST_REGISTER_COMBINATION(dma_cfg_fields_atp_en, transaction_.cfg.fields.atp_en)
TEST_REGISTER_COMBINATION(dma_cfg_fields_watermark_en, transaction_.cfg.fields.watermark_en)
TEST_REGISTER_COMBINATION(dma_cfg_fields_rwf_en, transaction_.cfg.fields.rwf_en)
TEST_REGISTER_COMBINATION(dma_cfg_fields_rws_en, transaction_.cfg.fields.rws_en)
TEST_REGISTER_COMBINATION(dma_cfg_fields_src_list_cfg, transaction_.cfg.fields.src_list_cfg)
TEST_REGISTER_COMBINATION(dma_cfg_fields_dst_list_cfg, transaction_.cfg.fields.dst_list_cfg)
TEST_REGISTER_COMBINATION(dma_cfg_fields_conversion_cfg, transaction_.cfg.fields.conversion_cfg)
TEST_REGISTER_COMBINATION(dma_cfg_fields_acceleration_cfg, transaction_.cfg.fields.acceleration_cfg)
TEST_REGISTER_COMBINATION(dma_cfg_fields_tile4_cfg, transaction_.cfg.fields.tile4_cfg)
TEST_REGISTER_COMBINATION(dma_cfg_fields_axi_user_bits_cfg, transaction_.cfg.fields.axi_user_bits_cfg)
TEST_REGISTER_COMBINATION(dma_cfg_fields_hwp_id_en, transaction_.cfg.fields.hwp_id_en)
TEST_REGISTER_COMBINATION(dma_cfg_fields_hwp_id, transaction_.cfg.fields.hwp_id)
TEST_REGISTER_COMBINATION(dma_remote_width_fetch, transaction_.remote_width_fetch)
TEST_REGISTER_COMBINATION(dma_width_src, transaction_.width.src)
TEST_REGISTER_COMBINATION(dma_width_dst, transaction_.width.dst)
TEST_REGISTER_COMBINATION(dma_acc_info_compress_dtype, transaction_.acc_info.compress.dtype)
TEST_REGISTER_COMBINATION(dma_acc_info_compress_sparse, transaction_.acc_info.compress.sparse)
TEST_REGISTER_COMBINATION(dma_acc_info_compress_bitc_en, transaction_.acc_info.compress.bitc_en)
TEST_REGISTER_COMBINATION(dma_acc_info_compress_z, transaction_.acc_info.compress.z)
TEST_REGISTER_COMBINATION(dma_acc_info_compress_bitmap_buf_sz, transaction_.acc_info.compress.bitmap_buf_sz)
TEST_REGISTER_COMBINATION(dma_acc_info_compress_bitmap_base_addr, transaction_.acc_info.compress.bitmap_base_addr)
TEST_REGISTER_COMBINATION(dma_acc_info_decompress_dtype, transaction_.acc_info.decompress.dtype)
TEST_REGISTER_COMBINATION(dma_acc_info_decompress_sparse, transaction_.acc_info.decompress.sparse)
TEST_REGISTER_COMBINATION(dma_acc_info_decompress_bitc_en, transaction_.acc_info.decompress.bitc_en)
TEST_REGISTER_COMBINATION(dma_acc_info_decompress_z, transaction_.acc_info.decompress.z)
TEST_REGISTER_COMBINATION(dma_acc_info_decompress_bitmap_base_addr, transaction_.acc_info.decompress.bitmap_base_addr)
TEST_REGISTER_COMBINATION(dma_acc_info_w_prep_dtype, transaction_.acc_info.w_prep.dtype)
TEST_REGISTER_COMBINATION(dma_acc_info_w_prep_sparse, transaction_.acc_info.w_prep.sparse)
TEST_REGISTER_COMBINATION(dma_acc_info_w_prep_zeropoint, transaction_.acc_info.w_prep.zeropoint)
TEST_REGISTER_COMBINATION(dma_acc_info_w_prep_ic, transaction_.acc_info.w_prep.ic)
TEST_REGISTER_COMBINATION(dma_acc_info_w_prep_filtersize, transaction_.acc_info.w_prep.filtersize)
TEST_REGISTER_COMBINATION(dma_acc_info_w_prep_bitmap_base_addr, transaction_.acc_info.w_prep.bitmap_base_addr)
TEST_REGISTER_COMBINATION(dma_mset_data, transaction_.mset_data)
TEST_REGISTER_COMBINATION(dma_src, transaction_.src)
TEST_REGISTER_COMBINATION(dma_sra, transaction_.sra)
TEST_REGISTER_COMBINATION(dma_dst, transaction_.dst)
TEST_REGISTER_COMBINATION(dma_dra, transaction_.dra)
TEST_REGISTER_COMBINATION(dma_sba_addr, transaction_.sba_addr)
TEST_REGISTER_COMBINATION(dma_dba_addr, transaction_.dba_addr)
TEST_REGISTER_COMBINATION(dma_barrier_prod_mask_lower, transaction_.barrier.prod_mask_lower)
TEST_REGISTER_COMBINATION(dma_barrier_cons_mask_lower, transaction_.barrier.cons_mask_lower)
TEST_REGISTER_COMBINATION(dma_barrier_prod_mask_upper, transaction_.barrier.prod_mask_upper)
TEST_REGISTER_COMBINATION(dma_barrier_cons_mask_upper, transaction_.barrier.cons_mask_upper)
TEST_REGISTER_COMBINATION(dma_list_size_src, transaction_.list_size.src)
TEST_REGISTER_COMBINATION(dma_list_size_dst, transaction_.list_size.dst)
TEST_REGISTER_COMBINATION(dma_dim_size_1_src, transaction_.dim_size_1.src)
TEST_REGISTER_COMBINATION(dma_dim_size_1_dst, transaction_.dim_size_1.dst)
TEST_REGISTER_COMBINATION(dma_stride_src_1, transaction_.stride_src_1)
TEST_REGISTER_COMBINATION(dma_stride_dst_1, transaction_.stride_dst_1)
TEST_REGISTER_COMBINATION(dma_dim_size_2_src, transaction_.dim_size_2.src)
TEST_REGISTER_COMBINATION(dma_dim_size_2_dst, transaction_.dim_size_2.dst)
TEST_REGISTER_COMBINATION(dma_list_addr_src, transaction_.list_addr.src)
TEST_REGISTER_COMBINATION(dma_list_addr_dst, transaction_.list_addr.dst)
TEST_REGISTER_COMBINATION(dma_stride_src_2, transaction_.stride_src_2)
TEST_REGISTER_COMBINATION(dma_stride_dst_2, transaction_.stride_dst_2)
TEST_REGISTER_COMBINATION(dma_remote_width_store, transaction_.remote_width_store)
TEST_REGISTER_COMBINATION(dma_dim_size_src_3, transaction_.dim_size_src_3)
TEST_REGISTER_COMBINATION(dma_dim_size_src_4, transaction_.dim_size_src_4)
TEST_REGISTER_COMBINATION(dma_dim_size_dst_3, transaction_.dim_size_dst_3)
TEST_REGISTER_COMBINATION(dma_dim_size_dst_4, transaction_.dim_size_dst_4)
TEST_REGISTER_COMBINATION(dma_dim_size_src_5, transaction_.dim_size_src_5)
TEST_REGISTER_COMBINATION(dma_dim_size_dst_5, transaction_.dim_size_dst_5)
TEST_REGISTER_COMBINATION(dma_stride_src_3, transaction_.stride_src_3)
TEST_REGISTER_COMBINATION(dma_stride_dst_3, transaction_.stride_dst_3)
TEST_REGISTER_COMBINATION(dma_stride_src_4, transaction_.stride_src_4)
TEST_REGISTER_COMBINATION(dma_stride_dst_4, transaction_.stride_dst_4)
TEST_REGISTER_COMBINATION(dma_stride_src_5, transaction_.stride_src_5)
TEST_REGISTER_COMBINATION(dma_stride_dst_5, transaction_.stride_dst_5)
TEST_REGISTER_COMBINATION(start_after_, barriers_sched_.start_after_)
TEST_REGISTER_COMBINATION(clean_after_, barriers_sched_.clean_after_)