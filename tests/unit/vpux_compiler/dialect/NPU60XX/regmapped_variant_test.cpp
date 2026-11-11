//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <npu_60xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/descriptors.hpp"

using namespace npu60xx;
using namespace vpux::NPUReg60XX;

class NPUReg60XX_DpuVariantRegisterTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuDPUVariant, vpux::NPUReg60XX::Descriptors::DpuVariantRegister> {};

#define TEST_NPU6_DPUVARIANT_REG_FIELD(FieldType, DescriptorMember)                                                   \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg60XX_DpuVariantRegisterTest, FieldType, vpux::NPUReg60XX::Fields::FieldType, \
                                   DescriptorMember, 0)

#define TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(ParentRegType, FieldType, DescriptorMember)             \
    HELPER_TEST_NPU_MULTIPLE_REGS_FIELD(NPUReg60XX_DpuVariantRegisterTest, ParentRegType##__##FieldType, \
                                        vpux::NPUReg60XX::Registers::ParentRegType,                      \
                                        vpux::NPUReg60XX::Fields::FieldType, DescriptorMember, 0)

// invar_ptr ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(invar_ptr, registers_.invar_ptr.invar_ptr_bf.invar_ptr)
// var_tag ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(var_tag, registers_.invar_ptr.invar_ptr_bf.var_tag)
// workload_size0 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_size_x, registers_.workload_size0.workload_size0_bf.workload_size_x)
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_size_y, registers_.workload_size0.workload_size0_bf.workload_size_y)
// workload_size1 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_size_z, registers_.workload_size1.workload_size1_bf.workload_size_z)
TEST_NPU6_DPUVARIANT_REG_FIELD(pad_count_up, registers_.workload_size1.workload_size1_bf.pad_count_up)
TEST_NPU6_DPUVARIANT_REG_FIELD(pad_count_left, registers_.workload_size1.workload_size1_bf.pad_count_left)
TEST_NPU6_DPUVARIANT_REG_FIELD(pad_count_down, registers_.workload_size1.workload_size1_bf.pad_count_down)
TEST_NPU6_DPUVARIANT_REG_FIELD(pad_count_right, registers_.workload_size1.workload_size1_bf.pad_count_right)
TEST_NPU6_DPUVARIANT_REG_FIELD(halo_enable_new_desc, registers_.workload_size1.workload_size1_bf.halo_enable_new_desc)
// workload_start0 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_start_x, registers_.workload_start0.workload_start0_bf.workload_start_x)
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_start_y, registers_.workload_start0.workload_start0_bf.workload_start_y)
// workload_start1 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_start_z, registers_.workload_start1.workload_start1_bf.workload_start_z)
TEST_NPU6_DPUVARIANT_REG_FIELD(new_invar_format, registers_.workload_start1.workload_start1_bf.new_invar_format)
TEST_NPU6_DPUVARIANT_REG_FIELD(new_var_format, registers_.workload_start1.workload_start1_bf.new_var_format)
TEST_NPU6_DPUVARIANT_REG_FIELD(weight_d_ptr_start, registers_.workload_start1.workload_start1_bf.weight_d_ptr_start)
// offset_addr ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(nthw_ntk, registers_.offset_addr.offset_addr_bf.nthw_ntk)
TEST_NPU6_DPUVARIANT_REG_FIELD(bin_cfg, registers_.offset_addr.offset_addr_bf.bin_cfg)
TEST_NPU6_DPUVARIANT_REG_FIELD(conv_cond, registers_.offset_addr.offset_addr_bf.conv_cond)
TEST_NPU6_DPUVARIANT_REG_FIELD(dense_se, registers_.offset_addr.offset_addr_bf.dense_se)
TEST_NPU6_DPUVARIANT_REG_FIELD(cim_clk_en, registers_.offset_addr.offset_addr_bf.cim_clk_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(swizzle_key_offset, registers_.offset_addr.offset_addr_bf.swizzle_key)
TEST_NPU6_DPUVARIANT_REG_FIELD(idu_mrm_clk_en, registers_.offset_addr.offset_addr_bf.idu_mrm_clk_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(odu_clk_en, registers_.offset_addr.offset_addr_bf.odu_clk_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(mpe_clk_en, registers_.offset_addr.offset_addr_bf.mpe_clk_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(ppe_clk_en, registers_.offset_addr.offset_addr_bf.ppe_clk_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(odu_stat_en, registers_.offset_addr.offset_addr_bf.odu_stat_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(idu_stat_en, registers_.offset_addr.offset_addr_bf.idu_stat_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(noc_clk_en, registers_.offset_addr.offset_addr_bf.noc_clk_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(odu_stat_clr_mode, registers_.offset_addr.offset_addr_bf.odu_stat_clr_mode)
TEST_NPU6_DPUVARIANT_REG_FIELD(idu_stat_clr_mode, registers_.offset_addr.offset_addr_bf.idu_stat_clr_mode)
TEST_NPU6_DPUVARIANT_REG_FIELD(se_only_en, registers_.offset_addr.offset_addr_bf.se_only_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(shave_l2_cache_en, registers_.offset_addr.offset_addr_bf.shave_l2_cache_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(idu_dbg_en, registers_.offset_addr.offset_addr_bf.idu_dbg_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(sb_read_en, registers_.offset_addr.offset_addr_bf.sb_read_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(tensor2_act_dense, registers_.offset_addr.offset_addr_bf.tensor2_act_dense)
TEST_NPU6_DPUVARIANT_REG_FIELD(grp_conv_en, registers_.offset_addr.offset_addr_bf.grp_conv_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(zero_point_4b, registers_.offset_addr.offset_addr_bf.zero_point_4b)
TEST_NPU6_DPUVARIANT_REG_FIELD(wt_swizzle_key, registers_.offset_addr.offset_addr_bf.wt_swizzle_key)
TEST_NPU6_DPUVARIANT_REG_FIELD(wt_swizzle_sel, registers_.offset_addr.offset_addr_bf.wt_swizzle_sel)
TEST_NPU6_DPUVARIANT_REG_FIELD(gif_clk_en, registers_.offset_addr.offset_addr_bf.gif_clk_en)
// hwp_wload_id ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(hwp_wload_id, registers_.hwp_wload_id.hwp_wload_id_bf.wload_id)
// var_cfg ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(invar_plt_rd_en, registers_.var_cfg.var_cfg_bf.invar_plt_rd_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(plt_ptr_force, registers_.var_cfg.var_cfg_bf.plt_ptr_force)
TEST_NPU6_DPUVARIANT_REG_FIELD(invar_lut_rd_en, registers_.var_cfg.var_cfg_bf.invar_lut_rd_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(invar_line_cnt_en, registers_.var_cfg.var_cfg_bf.invar_line_cnt_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(invar_line_cnt, registers_.var_cfg.var_cfg_bf.invar_line_cnt_cnt)
TEST_NPU6_DPUVARIANT_REG_FIELD(invar_lptr_force, registers_.var_cfg.var_cfg_bf.invar_lptr_force)
TEST_NPU6_DPUVARIANT_REG_FIELD(next_sram_job_valid, registers_.var_cfg.var_cfg_bf.next_sram_job_valid)
TEST_NPU6_DPUVARIANT_REG_FIELD(next_sram_job_addr, registers_.var_cfg.var_cfg_bf.next_sram_job_addr)
// cbarrier_lo ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(cbarrier_lo, registers_.cbarrier_lo)
// cbarrier_hi ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(cbarrier_hi, registers_.cbarrier_hi)
// pbarrier_lo ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(pbarrier_lo, registers_.pbarrier_lo)
// pbarrier_hi ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(pbarrier_hi, registers_.pbarrier_hi)
// halo_region 0 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0A, sp_adr_offset,
                                         registers_.halo_region[0].halo_region_a.halo_region_a_bf.sp_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0A, tile_select,
                                         registers_.halo_region[0].halo_region_a.halo_region_a_bf.tile_select)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0A, rsvdA,
                                         registers_.halo_region[0].halo_region_a.halo_region_a_bf.rsvd)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0A, enable,
                                         registers_.halo_region[0].halo_region_a.halo_region_a_bf.enable)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0B, ac_adr_offset,
                                         registers_.halo_region[0].halo_region_b.halo_region_b_bf.ac_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0B, target_width_lsb,
                                         registers_.halo_region[0].halo_region_b.halo_region_b_bf.target_width_lsb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0C, begin_x,
                                         registers_.halo_region[0].halo_region_c.halo_region_c_bf.begin_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0C, begin_y,
                                         registers_.halo_region[0].halo_region_c.halo_region_c_bf.begin_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0C, target_width_msb,
                                         registers_.halo_region[0].halo_region_c.halo_region_c_bf.target_width_msb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0D, end_x,
                                         registers_.halo_region[0].halo_region_d.halo_region_d_bf.end_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0D, end_y,
                                         registers_.halo_region[0].halo_region_d.halo_region_d_bf.end_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region0D, rsvdD,
                                         registers_.halo_region[0].halo_region_d.halo_region_d_bf.rsvd)
// halo_region 1 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1A, sp_adr_offset,
                                         registers_.halo_region[1].halo_region_a.halo_region_a_bf.sp_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1A, tile_select,
                                         registers_.halo_region[1].halo_region_a.halo_region_a_bf.tile_select)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1A, rsvdA,
                                         registers_.halo_region[1].halo_region_a.halo_region_a_bf.rsvd)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1A, enable,
                                         registers_.halo_region[1].halo_region_a.halo_region_a_bf.enable)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1B, ac_adr_offset,
                                         registers_.halo_region[1].halo_region_b.halo_region_b_bf.ac_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1B, target_width_lsb,
                                         registers_.halo_region[1].halo_region_b.halo_region_b_bf.target_width_lsb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1C, begin_x,
                                         registers_.halo_region[1].halo_region_c.halo_region_c_bf.begin_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1C, begin_y,
                                         registers_.halo_region[1].halo_region_c.halo_region_c_bf.begin_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1C, target_width_msb,
                                         registers_.halo_region[1].halo_region_c.halo_region_c_bf.target_width_msb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1D, end_x,
                                         registers_.halo_region[1].halo_region_d.halo_region_d_bf.end_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1D, end_y,
                                         registers_.halo_region[1].halo_region_d.halo_region_d_bf.end_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region1D, rsvdD,
                                         registers_.halo_region[1].halo_region_d.halo_region_d_bf.rsvd)
// halo_region 2 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2A, sp_adr_offset,
                                         registers_.halo_region[2].halo_region_a.halo_region_a_bf.sp_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2A, tile_select,
                                         registers_.halo_region[2].halo_region_a.halo_region_a_bf.tile_select)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2A, rsvdA,
                                         registers_.halo_region[2].halo_region_a.halo_region_a_bf.rsvd)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2A, enable,
                                         registers_.halo_region[2].halo_region_a.halo_region_a_bf.enable)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2B, ac_adr_offset,
                                         registers_.halo_region[2].halo_region_b.halo_region_b_bf.ac_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2B, target_width_lsb,
                                         registers_.halo_region[2].halo_region_b.halo_region_b_bf.target_width_lsb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2C, begin_x,
                                         registers_.halo_region[2].halo_region_c.halo_region_c_bf.begin_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2C, begin_y,
                                         registers_.halo_region[2].halo_region_c.halo_region_c_bf.begin_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2C, target_width_msb,
                                         registers_.halo_region[2].halo_region_c.halo_region_c_bf.target_width_msb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2D, end_x,
                                         registers_.halo_region[2].halo_region_d.halo_region_d_bf.end_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2D, end_y,
                                         registers_.halo_region[2].halo_region_d.halo_region_d_bf.end_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region2D, rsvdD,
                                         registers_.halo_region[2].halo_region_d.halo_region_d_bf.rsvd)
// halo_region 3 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3A, sp_adr_offset,
                                         registers_.halo_region[3].halo_region_a.halo_region_a_bf.sp_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3A, tile_select,
                                         registers_.halo_region[3].halo_region_a.halo_region_a_bf.tile_select)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3A, rsvdA,
                                         registers_.halo_region[3].halo_region_a.halo_region_a_bf.rsvd)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3A, enable,
                                         registers_.halo_region[3].halo_region_a.halo_region_a_bf.enable)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3B, ac_adr_offset,
                                         registers_.halo_region[3].halo_region_b.halo_region_b_bf.ac_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3B, target_width_lsb,
                                         registers_.halo_region[3].halo_region_b.halo_region_b_bf.target_width_lsb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3C, begin_x,
                                         registers_.halo_region[3].halo_region_c.halo_region_c_bf.begin_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3C, begin_y,
                                         registers_.halo_region[3].halo_region_c.halo_region_c_bf.begin_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3C, target_width_msb,
                                         registers_.halo_region[3].halo_region_c.halo_region_c_bf.target_width_msb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3D, end_x,
                                         registers_.halo_region[3].halo_region_d.halo_region_d_bf.end_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3D, end_y,
                                         registers_.halo_region[3].halo_region_d.halo_region_d_bf.end_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region3D, rsvdD,
                                         registers_.halo_region[3].halo_region_d.halo_region_d_bf.rsvd)
// halo_region 4 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4A, sp_adr_offset,
                                         registers_.halo_region[4].halo_region_a.halo_region_a_bf.sp_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4A, tile_select,
                                         registers_.halo_region[4].halo_region_a.halo_region_a_bf.tile_select)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4A, rsvdA,
                                         registers_.halo_region[4].halo_region_a.halo_region_a_bf.rsvd)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4A, enable,
                                         registers_.halo_region[4].halo_region_a.halo_region_a_bf.enable)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4B, ac_adr_offset,
                                         registers_.halo_region[4].halo_region_b.halo_region_b_bf.ac_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4B, target_width_lsb,
                                         registers_.halo_region[4].halo_region_b.halo_region_b_bf.target_width_lsb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4C, begin_x,
                                         registers_.halo_region[4].halo_region_c.halo_region_c_bf.begin_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4C, begin_y,
                                         registers_.halo_region[4].halo_region_c.halo_region_c_bf.begin_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4C, target_width_msb,
                                         registers_.halo_region[4].halo_region_c.halo_region_c_bf.target_width_msb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4D, end_x,
                                         registers_.halo_region[4].halo_region_d.halo_region_d_bf.end_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4D, end_y,
                                         registers_.halo_region[4].halo_region_d.halo_region_d_bf.end_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region4D, rsvdD,
                                         registers_.halo_region[4].halo_region_d.halo_region_d_bf.rsvd)
// halo_region 5 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5A, sp_adr_offset,
                                         registers_.halo_region[5].halo_region_a.halo_region_a_bf.sp_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5A, tile_select,
                                         registers_.halo_region[5].halo_region_a.halo_region_a_bf.tile_select)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5A, rsvdA,
                                         registers_.halo_region[5].halo_region_a.halo_region_a_bf.rsvd)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5A, enable,
                                         registers_.halo_region[5].halo_region_a.halo_region_a_bf.enable)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5B, ac_adr_offset,
                                         registers_.halo_region[5].halo_region_b.halo_region_b_bf.ac_adr_offset)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5B, target_width_lsb,
                                         registers_.halo_region[5].halo_region_b.halo_region_b_bf.target_width_lsb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5C, begin_x,
                                         registers_.halo_region[5].halo_region_c.halo_region_c_bf.begin_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5C, begin_y,
                                         registers_.halo_region[5].halo_region_c.halo_region_c_bf.begin_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5C, target_width_msb,
                                         registers_.halo_region[5].halo_region_c.halo_region_c_bf.target_width_msb)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5D, end_x,
                                         registers_.halo_region[5].halo_region_d.halo_region_d_bf.end_x)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5D, end_y,
                                         registers_.halo_region[5].halo_region_d.halo_region_d_bf.end_y)
TEST_NPU6_DPUVARIANT_MULTIPLE_REGS_FIELD(halo_region5D, rsvdD,
                                         registers_.halo_region[5].halo_region_d.halo_region_d_bf.rsvd)
// dpu_cfg ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_start_odu, registers_.dpu_cfg.dpu_cfg_bf.workload_start_odu)
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_start_idu, registers_.dpu_cfg.dpu_cfg_bf.workload_start_idu)
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_prm_sel, registers_.dpu_cfg.dpu_cfg_bf.workload_prm_sel)
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_valid, registers_.dpu_cfg.dpu_cfg_bf.workload_valid)
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_shad_odu, registers_.dpu_cfg.dpu_cfg_bf.workload_shad_odu)
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_shad_idu, registers_.dpu_cfg.dpu_cfg_bf.workload_shad_idu)
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_idu_auto_upd_0, registers_.dpu_cfg.dpu_cfg_bf.workload_idu_auto_upd_0)
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_idu_auto_upd_1, registers_.dpu_cfg.dpu_cfg_bf.workload_idu_auto_upd_1)
TEST_NPU6_DPUVARIANT_REG_FIELD(workload_odu_auto_upd, registers_.dpu_cfg.dpu_cfg_bf.workload_odu_auto_upd)
TEST_NPU6_DPUVARIANT_REG_FIELD(cfg_Reserved_2, registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_2)
TEST_NPU6_DPUVARIANT_REG_FIELD(preempt_cc_en, registers_.dpu_cfg.dpu_cfg_bf.preempt_cc_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(bar_wait_preempt_en, registers_.dpu_cfg.dpu_cfg_bf.bar_wait_preempt_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(rst_ctxt_new, registers_.dpu_cfg.dpu_cfg_bf.rst_ctxt_new)
TEST_NPU6_DPUVARIANT_REG_FIELD(cfg_Reserved_3, registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_3)
TEST_NPU6_DPUVARIANT_REG_FIELD(cfg_Reserved_4, registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_4)
TEST_NPU6_DPUVARIANT_REG_FIELD(odu_stat_clr, registers_.dpu_cfg.dpu_cfg_bf.odu_stat_clr)
TEST_NPU6_DPUVARIANT_REG_FIELD(idu_stat_clr, registers_.dpu_cfg.dpu_cfg_bf.idu_stat_clr)
TEST_NPU6_DPUVARIANT_REG_FIELD(cfg_Reserved_5, registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_5)
TEST_NPU6_DPUVARIANT_REG_FIELD(cfg_Reserved_6, registers_.dpu_cfg.dpu_cfg_bf.cfg_Reserved_6)
// te_beg0 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(te_beg_y, registers_.te_beg0.te_beg0_bf.te_beg_y)
TEST_NPU6_DPUVARIANT_REG_FIELD(te_beg_z, registers_.te_beg0.te_beg0_bf.te_beg_z)
// te_beg1 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(te_beg_x, registers_.te_beg1.te_beg1_bf.te_beg_x)
TEST_NPU6_DPUVARIANT_REG_FIELD(dyn_primitive_id_var, registers_.te_beg1.te_beg1_bf.dyn_primitive_id_var)
TEST_NPU6_DPUVARIANT_REG_FIELD(dyn_primitive_id_var_en, registers_.te_beg1.te_beg1_bf.dyn_primitive_id_var_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(wt_size_max_en, registers_.te_beg1.te_beg1_bf.wt_size_max_en)
// te_end0 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(te_end_y, registers_.te_end0.te_end0_bf.te_end_y)
TEST_NPU6_DPUVARIANT_REG_FIELD(te_end_z, registers_.te_end0.te_end0_bf.te_end_z)
// te_end1 ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(te_end_x, registers_.te_end1.te_end1_bf.te_end_x)
// weight_size ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(weight_size, registers_.weight_size.weight_size_bf.weight_size)
TEST_NPU6_DPUVARIANT_REG_FIELD(scope_id_en, registers_.weight_size.weight_size_bf.scope_id_en)
TEST_NPU6_DPUVARIANT_REG_FIELD(dyn_primitive_id2_var, registers_.weight_size.weight_size_bf.dyn_primitive_id2_var)
// weight_num ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(weight_num, registers_.weight_num.weight_num_bf.weight_num)
TEST_NPU6_DPUVARIANT_REG_FIELD(weight_sp_ptr_start, registers_.weight_num.weight_num_bf.weight_sp_ptr_start)
// weight_start ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(weight_start, registers_.weight_start)
// invariant_ ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(invariant_, invariant_)
// invariant_index_ ---------------------------------------------------------------------
TEST_NPU6_DPUVARIANT_REG_FIELD(invariant_index_, invariant_index_)
