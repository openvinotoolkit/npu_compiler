//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <npu_37xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/types.hpp"

using namespace npu37xx;

// nn_public structs which describe hw descriptors for different dialects defined in different headers but have the same
// names (VpuDMATask,VpuDPUInvariant,VpuDPUInvariant,VpuBarrierCountConfig,VpuActKernelInvocation,VpuActKernelRange) and
// are put into the same namespace. It cause problems with gtest lib because NPUReg37XX_RegisterTest class inherit
// TestWithParam class which is parametrized by tested struct from nn_public headers twice:first time in NPU40XX related
// tests, second time in NPUReg37XX related tests. Gtest lib complaining in runtime about duplicate parameterized test
// name That's why we re-define tested hw descriptor as new struct - it helps us to avoid inner gtest conflicts with the
// test for the same descriptor for different dialects
struct Npu37XXDPUVariant {
    nn_public::VpuDPUVariant variantReg;
};

#define CREATE_HW_DPU_VARIANT_DESC(field, value)                                         \
    [] {                                                                                 \
        Npu37XXDPUVariant hwDPUVariantDesc;                                              \
        memset(reinterpret_cast<void*>(&hwDPUVariantDesc), 0, sizeof(hwDPUVariantDesc)); \
        hwDPUVariantDesc.field = value;                                                  \
        return hwDPUVariantDesc;                                                         \
    }()

class NPUReg37XX_VpuDPUVariantTest :
        public MLIR_RegMappedNPUReg37XXUnitBase<Npu37XXDPUVariant, vpux::NPUReg37XX::RegMapped_DpuVariantRegisterType> {
};

TEST_P(NPUReg37XX_VpuDPUVariantTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, Npu37XXDPUVariant>> dpuVariantFieldSetNPUReg37XX = {
        // workload_size0 ---------------------------------------------------------------------
        {{
                 {"workload_size0", {{"workload_size_x", 0x3FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.workload_size0.workload_size0_bf.workload_size_x, 0x3FFF)},
        {{
                 {"workload_size0", {{"workload_size_y", 0x3FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.workload_size0.workload_size0_bf.workload_size_y, 0x3FFF)},
        // workload_size1 ---------------------------------------------------------------------
        {{
                 {"workload_size1", {{"workload_size_z", 0x3FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.workload_size1.workload_size1_bf.workload_size_z, 0x3FFF)},
        {{
                 {"workload_size1", {{"pad_count_up", 7}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.workload_size1.workload_size1_bf.pad_count_up, 7)},
        {{
                 {"workload_size1", {{"pad_count_left", 7}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.workload_size1.workload_size1_bf.pad_count_left, 7)},
        {{
                 {"workload_size1", {{"pad_count_down", 7}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.workload_size1.workload_size1_bf.pad_count_down, 7)},
        {{
                 {"workload_size1", {{"pad_count_right", 7}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.workload_size1.workload_size1_bf.pad_count_right, 7)},
        // workload_start0 ---------------------------------------------------------------------
        {{
                 {"workload_start0", {{"workload_start_x", 0x3FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.workload_start0.workload_start0_bf.workload_start_x, 0x3FFF)},
        {{
                 {"workload_start0", {{"workload_start_y", 0x3FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.workload_start0.workload_start0_bf.workload_start_y, 0x3FFF)},
        // workload_start1 ---------------------------------------------------------------------
        {{
                 {"workload_start1", {{"workload_start_z", 0x3FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.workload_start1.workload_start1_bf.workload_start_z, 0x3FFF)},
        // offset_addr ---------------------------------------------------------------------
        {{
                 {"offset_addr", {{"nthw_ntk", 3}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.nthw_ntk, 3)},
        {{
                 {"offset_addr", {{"bin_cfg", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.bin_cfg, 1)},
        {{
                 {"offset_addr", {{"conv_cond", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.conv_cond, 1)},
        {{
                 {"offset_addr", {{"dense_se", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.dense_se, 1)},
        {{
                 {"offset_addr", {{"swizzle_key_offset", 7}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.swizzle_key, 7)},
        {{
                 {"offset_addr", {{"idu_mrm_clk_en", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.idu_mrm_clk_en, 1)},
        {{
                 {"offset_addr", {{"odu_clk_en", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.odu_clk_en, 1)},
        {{
                 {"offset_addr", {{"mpe_clk_en", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.mpe_clk_en, 1)},
        {{
                 {"offset_addr", {{"ppe_clk_en", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.ppe_clk_en, 1)},
        {{
                 {"offset_addr", {{"odu_stat_en", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.odu_stat_en, 1)},
        {{
                 {"offset_addr", {{"idu_stat_en", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.idu_stat_en, 1)},
        {{
                 {"offset_addr", {{"odu_stat_clr_mode", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.odu_stat_clr_mode, 1)},
        {{
                 {"offset_addr", {{"idu_stat_clr_mode", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.idu_stat_clr_mode, 1)},
        {{
                 {"offset_addr", {{"shave_l2_cache_en", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.shave_l2_cache_en, 1)},
        {{
                 {"offset_addr", {{"idu_dbg_en", 3}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.idu_dbg_en, 3)},
        {{
                 {"offset_addr", {{"wt_swizzle_key", 7}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.wt_swizzle_key, 7)},
        {{
                 {"offset_addr", {{"wt_swizzle_sel", 1}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.offset_addr.offset_addr_bf.wt_swizzle_sel, 1)},
        // te_end0 ---------------------------------------------------------------------
        {{
                 {"te_end0", {{"te_end_y", 0x1FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.te_end0.te_end0_bf.te_end_y, 0x1FFF)},
        {{
                 {"te_end0", {{"te_end_z", 0x1FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.te_end0.te_end0_bf.te_end_z, 0x1FFF)},
        // te_end1 ---------------------------------------------------------------------
        {{
                 {"te_end1", {{"te_end_x", 0x1FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.te_end1.te_end1_bf.te_end_x, 0x1FFF)},
        // te_beg0 ---------------------------------------------------------------------
        {{
                 {"te_beg0", {{"te_beg_y", 0x1FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.te_beg0.te_beg0_bf.te_beg_y, 0x1FFF)},
        {{
                 {"te_beg0", {{"te_beg_z", 0x1FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.te_beg0.te_beg0_bf.te_beg_z, 0x1FFF)},
        // te_beg1 ---------------------------------------------------------------------
        {{
                 {"te_beg1", {{"te_beg_x", 0x1FFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.te_beg1.te_beg1_bf.te_beg_x, 0x1FFF)},
        // weight_size ---------------------------------------------------------------------
        {{
                 {"weight_size", {{"weight_size", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.weight_size, 0xFFFFFFFF)},
        // weight_num ---------------------------------------------------------------------
        {{
                 {"weight_num", {{"weight_num", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.registers_.weight_num, 0xFFFFFFFF)},
        // invariant_ ---------------------------------------------------------------------
        {{
                 {"invariant_", {{"invariant_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.invariant_, 0xFFFFFFFF)},
        // invariant_index_ ---------------------------------------------------------------------
        {{
                 {"invariant_index_", {{"invariant_index_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.invariant_index_, 0xFFFFFFFF)},
        // weight_table_offset_ ----------------------------------------------------------
        {{
                 {"weight_table_offset_", {{"weight_table_offset_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.weight_table_offset_, 0xFFFFFFFF)},
        // wload_id_ ---------------------------------------------------------------------
        {{
                 {"wload_id_", {{"wload_id_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.wload_id_, 0xFFFFFFFF)},
        // cluster_ ---------------------------------------------------------------------
        {{
                 {"cluster_variant_", {{"cluster_", 0xFF}}},
         },
         CREATE_HW_DPU_VARIANT_DESC(variantReg.cluster_, 0xFF)},
};

INSTANTIATE_TEST_SUITE_P(NPUReg37XX_MappedRegs, NPUReg37XX_VpuDPUVariantTest,
                         testing::ValuesIn(dpuVariantFieldSetNPUReg37XX));
