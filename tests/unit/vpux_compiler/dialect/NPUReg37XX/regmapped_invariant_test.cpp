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
struct Npu37XXDPUInvariant {
    nn_public::VpuDPUInvariant invReg;
};

#define CREATE_HW_DPU_INVARIANT_DESC(field, value)                                           \
    [] {                                                                                     \
        Npu37XXDPUInvariant hwDPUInvariantDesc;                                              \
        memset(reinterpret_cast<void*>(&hwDPUInvariantDesc), 0, sizeof(hwDPUInvariantDesc)); \
        hwDPUInvariantDesc.field = value;                                                    \
        return hwDPUInvariantDesc;                                                           \
    }()

class NPUReg37XX_VpuDPUInvariantTest :
        public MLIR_RegMappedNPUReg37XXUnitBase<Npu37XXDPUInvariant,
                                                vpux::NPUReg37XX::RegMapped_DpuInvariantRegisterType> {};

TEST_P(NPUReg37XX_VpuDPUInvariantTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, Npu37XXDPUInvariant>> dpuInvariantFieldSetNPUReg37XX = {
        // se_sp_addr[0] ---------------------------------------------------------------------------
        {{
                 {"se_sp_addr_0", {{"se_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_addr[0].se_addr, 0xFFFFFFFF)},
        {{
                 {"se_sp_addr_0", {{"sparsity_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_addr[0].sparsity_addr, 0xFFFFFFFF)},
        // se_sp_addr[1] ---------------------------------------------------------------------------
        {{
                 {"se_sp_addr_1", {{"se_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_addr[1].se_addr, 0xFFFFFFFF)},
        {{
                 {"se_sp_addr_1", {{"sparsity_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_addr[1].sparsity_addr, 0xFFFFFFFF)},
        // se_sp_addr[2] ---------------------------------------------------------------------------
        {{
                 {"se_sp_addr_2", {{"se_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_addr[2].se_addr, 0xFFFFFFFF)},
        {{
                 {"se_sp_addr_2", {{"sparsity_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_addr[2].sparsity_addr, 0xFFFFFFFF)},
        // se_sp_addr[3] ---------------------------------------------------------------------------
        {{
                 {"se_sp_addr_3", {{"se_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_addr[3].se_addr, 0xFFFFFFFF)},
        {{
                 {"se_sp_addr_3", {{"sparsity_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_addr[3].sparsity_addr, 0xFFFFFFFF)},
        // se_sp_size[0] ---------------------------------------------------------------------------
        {{
                 {"se_sp_size_0", {{"sp_seg_size", 0x3FFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_size[0].se_sp_size_bf.sp_seg_size, 0x3FFF)},
        {{
                 {"se_sp_size_0", {{"se_seg_size", 0x3FFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_size[0].se_sp_size_bf.se_seg_size, 0x3FFFF)},
        // se_sp_size[1] ---------------------------------------------------------------------------
        {{
                 {"se_sp_size_1", {{"sp_seg_size", 0x3FFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_size[1].se_sp_size_bf.sp_seg_size, 0x3FFF)},
        {{
                 {"se_sp_size_1", {{"se_seg_size", 0x3FFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_size[1].se_sp_size_bf.se_seg_size, 0x3FFFF)},
        // se_sp_size[2] ---------------------------------------------------------------------------
        {{
                 {"se_sp_size_2", {{"sp_seg_size", 0x3FFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_size[2].se_sp_size_bf.sp_seg_size, 0x3FFF)},
        {{
                 {"se_sp_size_2", {{"se_seg_size", 0x3FFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_sp_size[2].se_sp_size_bf.se_seg_size, 0x3FFFF)},
        // z_config ---------------------------------------------------------------------
        {{
                 {"z_config", {{"se_z_split", 15}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.z_config.z_config_bf.se_z_split, 15)},
        {{
                 {"z_config", {{"num_ses_in_z_dir", 0x1FF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.z_config.z_config_bf.num_ses_in_z_dir, 0x1FF)},
        {{
                 {"z_config", {{"cm_sp_pattern", 0xFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.z_config.z_config_bf.cm_sp_pattern, 0xFFFF)},
        {{
                 {"z_config", {{"addr_format_sel", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.z_config.z_config_bf.addr_format_sel, 1)},
        // kernel_pad_cfg ---------------------------------------------------------------------
        {{
                 {"kernel_pad_cfg", {{"kernel_y", 0xF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_y, 0xF)},
        {{
                 {"kernel_pad_cfg", {{"kernel_x", 0xF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_x, 0xF)},
        {{
                 {"kernel_pad_cfg", {{"wt_plt_cfg", 3}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.wt_plt_cfg, 3)},
        {{
                 {"kernel_pad_cfg", {{"act_dense", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense, 1)},
        {{
                 {"kernel_pad_cfg", {{"wt_dense", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense, 1)},
        {{
                 {"kernel_pad_cfg", {{"stride_y_en", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y_en, 1)},
        {{
                 {"kernel_pad_cfg", {{"stride_y", 7}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y, 7)},
        {{
                 {"kernel_pad_cfg", {{"dynamic_bw_en", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en, 1)},
        {{
                 {"kernel_pad_cfg", {{"dw_wt_sp_ins", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.dw_wt_sp_ins, 1)},
        {{
                 {"kernel_pad_cfg", {{"layer1_wt_sp_ins", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_wt_sp_ins, 1)},
        {{
                 {"kernel_pad_cfg", {{"layer1_cmp_en", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_cmp_en, 1)},
        {{
                 {"kernel_pad_cfg", {{"pool_opt_en", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pool_opt_en, 1)},
        {{
                 {"kernel_pad_cfg", {{"sp_se_tbl_segment", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment, 1)},
        {{
                 {"kernel_pad_cfg", {{"rst_ctxt", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.kernel_pad_cfg.kernel_pad_cfg_bf.rst_ctxt, 1)},
        // weight_size_placeholder
        {{
                 {"weight_size_placeholder", {{"weight_size_placeholder", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.weight_size_placeholder, 0xFFFFFFFF)},
        // weight_num_placeholder
        {{
                 {"weight_num_placeholder", {{"weight_num_placeholder", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.weight_num_placeholder, 0xFFFFFFFF)},
        // weight_start
        {{
                 {"weight_start", {{"weight_start", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.weight_start, 0xFFFFFFFF)},
        // tensor_size0 ---------------------------------------------------------------------
        {{
                 {"tensor_size0", {{"tensor_size_x", 0x3FFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.tensor_size0.tensor_size0_bf.tensor_size_x, 0x3FFF)},
        {{
                 {"tensor_size0", {{"tensor_size_y", 0x3FFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.tensor_size0.tensor_size0_bf.tensor_size_y, 0x3FFF)},
        // tensor_size1 ---------------------------------------------------------------------
        {{
                 {"tensor_size1", {{"tensor_size_z", 0x3FFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.tensor_size1.tensor_size1_bf.tensor_size_z, 0x3FFF)},
        // tensor_start ---------------------------------------------------------------------
        {{
                 {"tensor_start", {{"tensor_start", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.tensor_start, 0xFFFFFFFF)},
        // tensor_mode ---------------------------------------------------------------------
        {{
                 {"tensor_mode", {{"wmode", 0xF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.tensor_mode.tensor_mode_bf.wmode, 0xF)},
        {{
                 {"tensor_mode", {{"amode", 0xF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.tensor_mode.tensor_mode_bf.amode, 0xF)},
        {{
                 {"tensor_mode", {{"stride", 7}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.tensor_mode.tensor_mode_bf.stride, 7)},
        {{
                 {"tensor_mode", {{"zm_input", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.tensor_mode.tensor_mode_bf.zm_input, 1)},
        {{
                 {"tensor_mode", {{"dw_input", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.tensor_mode.tensor_mode_bf.dw_input, 1)},
        {{
                 {"tensor_mode", {{"workload_operation", 3}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.tensor_mode.tensor_mode_bf.workload_operation, 3)},
        // elops_sparsity_addr ---------------------------------------------------------------------
        {{
                 {"elops_sparsity_addr", {{"elops_sparsity_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.elop_sparsity_addr, 0xFFFFFFFF)},
        // elops_se_addr ---------------------------------------------------------------------
        {{
                 {"elops_se_addr", {{"elops_se_addr", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.elop_se_addr, 0xFFFFFFFF)},
        // elops_wload ---------------------------------------------------------------------
        {{
                 {"elops_wload", {{"elop_wload", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.elops_wload.elops_wload_bf.elop_wload, 1)},
        {{
                 {"elops_wload", {{"elop_wload_type", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.elops_wload.elops_wload_bf.elop_wload_type, 1)},
        {{
                 {"elops_wload", {{"pool_wt_data", 0xFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.elops_wload.elops_wload_bf.pool_wt_data, 0xFFFF)},
        {{
                 {"elops_wload", {{"pool_wt_rd_dis", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.elops_wload.elops_wload_bf.pool_wt_rd_dis, 1)},
        // act_offset ---------------------------------------------------------------------
        {{
                 {"act_offset0", {{"act_offset", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.act_offset[0], 0xFFFFFFFF)},
        {{
                 {"act_offset1", {{"act_offset", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.act_offset[1], 0xFFFFFFFF)},
        {{
                 {"act_offset2", {{"act_offset", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.act_offset[2], 0xFFFFFFFF)},
        {{
                 {"act_offset3", {{"act_offset", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.act_offset[3], 0xFFFFFFFF)},
        // base_offset_a ---------------------------------------------------------------------
        {{
                 {"base_offset_a", {{"base_offset_a", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.base_offset_a, 0xFFFFFFFF)},
        // base_offset_b ---------------------------------------------------------------------
        {{
                 {"base_offset_b", {{"base_offset_b", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.base_offset_b, 0xFFFFFFFF)},
        // wt_offset ---------------------------------------------------------------------
        {{
                 {"wt_offset", {{"wt_offset", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.wt_offset, 0xFFFFFFFF)},
        // odu_cfg ---------------------------------------------------------------------
        {{
                 {"odu_cfg", {{"dtype", 7}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.dtype, 7)},

        {{
                 {"odu_cfg", {{"sp_value", 0xFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.sp_value, 0xFF)},
        {{
                 {"odu_cfg", {{"sp_out_en", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.sp_out_en, 1)},
        {{
                 {"odu_cfg", {{"write_sp", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.write_sp, 1)},
        {{
                 {"odu_cfg", {{"write_pt", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.write_pt, 1)},
        {{
                 {"odu_cfg", {{"write_ac", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.write_ac, 1)},
        {{
                 {"odu_cfg", {{"mode", 3}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.mode, 3)},
        {{
                 {"odu_cfg", {{"grid", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.grid, 1)},
        {{
                 {"odu_cfg", {{"swizzle_key", 7}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.swizzle_key, 7)},
        {{
                 {"odu_cfg", {{"nthw", 3}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.nthw, 3)},
        {{
                 {"odu_cfg", {{"permutation", 7}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.permutation, 7)},
        {{
                 {"odu_cfg", {{"debug_mode", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cfg.odu_cfg_bf.debug_mode, 1)},
        // odu_be_size ---------------------------------------------------------------------
        {{
                 {"odu_be_size", {{"odu_be_size", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_be_size, 0xFFFFFFFF)},
        // odu_be_cnt ---------------------------------------------------------------------
        {{
                 {"odu_be_cnt", {{"odu_be_cnt", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_be_cnt, 0xFFFFFFFF)},
        // odu_se_size ---------------------------------------------------------------------
        {{
                 {"se_size", {{"se_size", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.se_size, 0xFFFFFFFF)},
        // te_dim0 ---------------------------------------------------------------------
        {{
                 {"te_dim0", {{"te_dim_y", 0x1FFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.te_dim0.te_dim0_bf.te_dim_y, 0x1FFF)},
        {{
                 {"te_dim0", {{"te_dim_z", 0x1FFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.te_dim0.te_dim0_bf.te_dim_z, 0x1FFF)},
        // te_dim1 ---------------------------------------------------------------------
        {{
                 {"te_dim1", {{"te_dim_x", 0x1FFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.te_dim1.te_dim1_bf.te_dim_x, 0x1FFF)},
        // pt_base ---------------------------------------------------------------------
        {{
                 {"pt_base", {{"pt_base", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.pt_base, 0xFFFFFFFF)},
        // sp_base ---------------------------------------------------------------------
        {{
                 {"sp_base", {{"sp_base", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.sp_base, 0xFFFFFFFF)},
        // odu_cast0 ---------------------------------------------------------------------
        {{
                 {"odu_cast0", {{"cast_enable", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cast[0].odu_cast_bf.cast_enable, 1)},
        {{
                 {"odu_cast0", {{"cast_offset", 0xFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cast[0].odu_cast_bf.cast_offset, 0xFFFFFFF)},
        // odu_cast1 ---------------------------------------------------------------------
        {{
                 {"odu_cast1", {{"cast_enable", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cast[1].odu_cast_bf.cast_enable, 1)},
        {{
                 {"odu_cast1", {{"cast_offset", 0xFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cast[1].odu_cast_bf.cast_offset, 0xFFFFFFF)},
        // odu_cast2 ---------------------------------------------------------------------
        {{
                 {"odu_cast2", {{"cast_enable", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cast[2].odu_cast_bf.cast_enable, 1)},
        {{
                 {"odu_cast2", {{"cast_offset", 0xFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.odu_cast[2].odu_cast_bf.cast_offset, 0xFFFFFFF)},
        // mpe_cfg ---------------------------------------------------------------------
        {{
                 {"mpe_cfg", {{"mpe_wtbias", 0xFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.mpe_cfg.mpe_cfg_bf.mpe_wtbias, 0xFF)},
        {{
                 {"mpe_cfg", {{"mpe_actbias", 0xFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.mpe_cfg.mpe_cfg_bf.mpe_actbias, 0xFF)},
        {{
                 {"mpe_cfg", {{"mpe_daz", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.mpe_cfg.mpe_cfg_bf.mpe_daz, 1)},
        // elop_scale ---------------------------------------------------------------------
        {{
                 {"elop_scale", {{"elop_scale_b", 0xFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.elop_scale.elop_scale_bf.elop_scale_b, 0xFFFF)},
        {{
                 {"elop_scale", {{"elop_scale_a", 0xFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.elop_scale.elop_scale_bf.elop_scale_a, 0xFFFF)},
        // ppe_cfg ---------------------------------------------------------------------
        {{
                 {"ppe_cfg", {{"ppe_g8_bias_c", 0x1FF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c, 0x1FF)},
        // ppe_bias ---------------------------------------------------------------------
        {{
                 {"ppe_bias", {{"ppe_bias", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_bias, 0xFFFFFFFF)},
        // ppe_scale ---------------------------------------------------------------------
        {{
                 {"ppe_scale", {{"ppe_scale_shift", 0x3F}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_scale.ppe_scale_bf.ppe_scale_shift, 0x3F)},
        {{
                 {"ppe_scale", {{"ppe_scale_round", 3}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_scale.ppe_scale_bf.ppe_scale_round, 3)},
        {{
                 {"ppe_scale", {{"ppe_scale_mult", 0xFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_scale.ppe_scale_bf.ppe_scale_mult, 0xFFFF)},
        // ppe_scale_ctrl ---------------------------------------------------------------------
        {{
                 {"ppe_scale_ctrl", {{"ppe_scale_override", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override, 1)},
        {{
                 {"ppe_scale_ctrl", {{"ppe_fp_scale_override", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override, 1)},
        // ppe_prelu ---------------------------------------------------------------------
        {{
                 {"ppe_prelu", {{"ppe_prelu_shift", 0x1F}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift, 0x1F)},
        {{
                 {"ppe_prelu", {{"ppe_prelu_mult", 0x7FF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult, 0x7FF)},
        // ppe_scale_hclamp ---------------------------------------------------------------------
        {{
                 {"ppe_scale_hclamp", {{"ppe_scale_hclamp", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_scale_hclamp, 0xFFFFFFFF)},
        // ppe_scale_lclamp ---------------------------------------------------------------------
        {{
                 {"ppe_scale_lclamp", {{"ppe_scale_lclamp", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_scale_lclamp, 0xFFFFFFFF)},
        // ppe_misc ---------------------------------------------------------------------
        {{
                 {"ppe_misc", {{"ppe_fp16_ftz", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_misc.ppe_misc_bf.ppe_fp16_ftz, 1)},
        {{
                 {"ppe_misc", {{"ppe_fp16_clamp", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_misc.ppe_misc_bf.ppe_fp16_clamp, 1)},
        {{
                 {"ppe_misc", {{"ppe_i32_convert", 3}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_misc.ppe_misc_bf.ppe_i32_convert, 3)},
        // ppe_fp_bias ---------------------------------------------------------------------
        {{
                 {"ppe_fp_bias", {{"ppe_fp_bias", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_fp_bias, 0xFFFFFFFF)},
        // ppe_fp_scale ---------------------------------------------------------------------
        {{
                 {"ppe_fp_scale", {{"ppe_fp_scale", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_fp_scale, 0xFFFFFFFF)},
        // ppe_fp_prelu ---------------------------------------------------------------------
        {{
                 {"ppe_fp_prelu", {{"ppe_fp_prelu", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_fp_prelu, 0xFFFFFFFF)},
        // ppe_fp_cfg ---------------------------------------------------------------------
        {{
                 {"ppe_fp_cfg", {{"ppe_fp_convert", 7}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert, 7)},
        {{
                 {"ppe_fp_cfg", {{"ppe_fp_bypass", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass, 1)},
        {{
                 {"ppe_fp_cfg", {{"ppe_bf16_round", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_bf16_round, 1)},
        {{
                 {"ppe_fp_cfg", {{"ppe_fp_prelu_en", 1}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.registers_.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en, 1)},
        // hwp_cmx_base_offset_ ---------------------------------------------------------------------
        {{
                 {"hwp_cmx_base_offset_", {{"hwp_cmx_base_offset_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.hwp_cmx_base_offset_, 0xFFFFFFFF)},
        // barriers_ ---------------------------------------------------------------------
        {{
                 {"barriers_wait_mask_", {{"barriers_wait_mask_", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.barriers_.wait_mask_, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"barriers_post_mask_", {{"barriers_post_mask_", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.barriers_.post_mask_, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"barriers_group_mask_", {{"group_", 0xFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.barriers_.group_, 0xFF)},
        {{
                 {"barriers_group_mask_", {{"mask_", 0xFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.barriers_.mask_, 0xFF)},
        // barriers_sched_ ---------------------------------------------------------------------
        {{
                 {"barriers_sched_", {{"start_after_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.barriers_sched_.start_after_, 0xFFFFFFFF)},
        {{
                 {"barriers_sched_", {{"clean_after_", 0xFFFFFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.barriers_sched_.clean_after_, 0xFFFFFFFF)},
        // variant_count_ ---------------------------------------------------------------------
        {{
                 {"variant_count_", {{"variant_count_", 0xFFFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.variant_count_, 0xFFFF)},
        // cluster_ ---------------------------------------------------------------------
        {{
                 {"cluster_invariant_", {{"cluster_", 0xFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.cluster_, 0xFF)},
        // is_cont_conv_ ---------------------------------------------------------------------
        {{
                 {"is_cont_conv_", {{"is_cont_conv_", 0xFF}}},
         },
         CREATE_HW_DPU_INVARIANT_DESC(invReg.is_cont_conv_, 0xFF)},
};

INSTANTIATE_TEST_SUITE_P(NPUReg37XX_MappedRegs, NPUReg37XX_VpuDPUInvariantTest,
                         testing::ValuesIn(dpuInvariantFieldSetNPUReg37XX));
