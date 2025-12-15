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

class NPUReg50XX_DpuInvariantRegisterTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuDPUInvariant,
                                       vpux::NPUReg50XX::Descriptors::DpuInvariantRegister> {};

#define TEST_NPU5_DPUINVARIANT_REG_FIELD(FieldType, DescriptorMember)              \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_DpuInvariantRegisterTest, FieldType, \
                                   vpux::NPUReg50XX::Fields::FieldType, DescriptorMember, 0)

#define TEST_NPU5_DPUINVARIANT_MULTIPLE_REGS_FIELD(ParentRegType, FieldType, DescriptorMember)             \
    HELPER_TEST_NPU_MULTIPLE_REGS_FIELD(NPUReg50XX_DpuInvariantRegisterTest, ParentRegType##__##FieldType, \
                                        vpux::NPUReg50XX::Registers::ParentRegType,                        \
                                        vpux::NPUReg50XX::Fields::FieldType, DescriptorMember, 0)

#define TEST_NPU5_DPUINVARIANT_REG_FIELD_SHIFT(FieldType, DescriptorMember, LeftShiftBitsCount) \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_DpuInvariantRegisterTest, FieldType,              \
                                   vpux::NPUReg50XX::Fields::FieldType, DescriptorMember, LeftShiftBitsCount)

// cmx_slice0_low_addr ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(cmx_slice0_low_addr, registers_.cmx_slice0_low_addr)
// cmx_slice1_low_addr ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(cmx_slice1_low_addr, registers_.cmx_slice1_low_addr)
// cmx_slice2_low_addr ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(cmx_slice2_low_addr, registers_.cmx_slice2_low_addr)
// cmx_slice3_low_addr ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(cmx_slice3_low_addr, registers_.cmx_slice3_low_addr)
// cmx_slice_size ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(cmx_slice_size, registers_.cmx_slice_size)
// se_addr ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(se_addr, registers_.se_addr)
// sparsity_addr ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(sparsity_addr, registers_.sparsity_addr)
// se_size ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(se_size, registers_.se_size)
// z_config ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(se_z_split, registers_.z_config.z_config_bf.se_z_split)
TEST_NPU5_DPUINVARIANT_REG_FIELD(num_ses_in_z_dir, registers_.z_config.z_config_bf.num_ses_in_z_dir)
TEST_NPU5_DPUINVARIANT_REG_FIELD(cm_sp_pattern, registers_.z_config.z_config_bf.cm_sp_pattern)
TEST_NPU5_DPUINVARIANT_REG_FIELD(npo2_se_z_split_en, registers_.z_config.z_config_bf.npo2_se_z_split_enable)
TEST_NPU5_DPUINVARIANT_REG_FIELD(reserved, registers_.z_config.z_config_bf.reserved)
TEST_NPU5_DPUINVARIANT_REG_FIELD(addr_format_sel, registers_.z_config.z_config_bf.addr_format_sel)
// kernel_pad_cfg ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(mpe_assign, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign)
TEST_NPU5_DPUINVARIANT_REG_FIELD(pad_right_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pad_right_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(pad_left_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pad_left_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(pad_bottom_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pad_bottom_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(pad_top_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pad_top_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(kernel_y, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_y)
TEST_NPU5_DPUINVARIANT_REG_FIELD(kernel_x, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_x)
TEST_NPU5_DPUINVARIANT_REG_FIELD(wt_plt_cfg, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.wt_plt_cfg)
TEST_NPU5_DPUINVARIANT_REG_FIELD(act_dense, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense)
TEST_NPU5_DPUINVARIANT_REG_FIELD(wt_dense, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense)
TEST_NPU5_DPUINVARIANT_REG_FIELD(stride_y_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(stride_y, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y)
TEST_NPU5_DPUINVARIANT_REG_FIELD(dynamic_bw_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(dw_wt_sp_ins, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.dw_wt_sp_ins)
TEST_NPU5_DPUINVARIANT_REG_FIELD(layer1_wt_sp_ins, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_wt_sp_ins)
TEST_NPU5_DPUINVARIANT_REG_FIELD(layer1_cmp_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_cmp_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(pool_opt_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pool_opt_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(sp_se_tbl_segment, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment)
TEST_NPU5_DPUINVARIANT_REG_FIELD(rst_ctxt, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.rst_ctxt)
// tensor_size0 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(tensor_size_x, registers_.tensor_size0.tensor_size0_bf.tensor_size_x)
TEST_NPU5_DPUINVARIANT_REG_FIELD(tensor_size_y, registers_.tensor_size0.tensor_size0_bf.tensor_size_y)
// tensor_size1 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(tensor_size_z, registers_.tensor_size1.tensor_size1_bf.tensor_size_z)
TEST_NPU5_DPUINVARIANT_REG_FIELD(npo2_se_size, registers_.tensor_size1.tensor_size1_bf.npo2_se_size)
// tensor_start ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(tensor_start, registers_.tensor_start)
// tensor_mode ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(wmode, registers_.tensor_mode.tensor_mode_bf.wmode)
TEST_NPU5_DPUINVARIANT_REG_FIELD(amode, registers_.tensor_mode.tensor_mode_bf.amode)
TEST_NPU5_DPUINVARIANT_REG_FIELD(stride, registers_.tensor_mode.tensor_mode_bf.stride)
TEST_NPU5_DPUINVARIANT_REG_FIELD(zm_input, registers_.tensor_mode.tensor_mode_bf.zm_input)
TEST_NPU5_DPUINVARIANT_REG_FIELD(dw_input, registers_.tensor_mode.tensor_mode_bf.dw_input)
TEST_NPU5_DPUINVARIANT_REG_FIELD(cm_input, registers_.tensor_mode.tensor_mode_bf.cm_input)
TEST_NPU5_DPUINVARIANT_REG_FIELD(workload_operation, registers_.tensor_mode.tensor_mode_bf.workload_operation)
TEST_NPU5_DPUINVARIANT_REG_FIELD(pad_value, registers_.tensor_mode.tensor_mode_bf.pad_value)
// elops_sparsity_addr ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(elops_sparsity_addr, registers_.elops_sparsity_addr)
// elops_se_addr ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(elops_se_addr, registers_.elops_se_addr)
// elops_wload ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(elop_wload, registers_.elops_wload.elops_wload_bf.elop_wload)
TEST_NPU5_DPUINVARIANT_REG_FIELD(seed_wload, registers_.elops_wload.elops_wload_bf.seed_wload)
TEST_NPU5_DPUINVARIANT_REG_FIELD(fifo_wr_wload, registers_.elops_wload.elops_wload_bf.fifo_wr_wload)
TEST_NPU5_DPUINVARIANT_REG_FIELD(elop_wload_type, registers_.elops_wload.elops_wload_bf.elop_wload_type)
TEST_NPU5_DPUINVARIANT_REG_FIELD(pool_wt_data, registers_.elops_wload.elops_wload_bf.pool_wt_data)
TEST_NPU5_DPUINVARIANT_REG_FIELD(pool_wt_rd_dis, registers_.elops_wload.elops_wload_bf.pool_wt_rd_dis)
// act_offset ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(act_offset0, registers_.act_offset[0])
TEST_NPU5_DPUINVARIANT_REG_FIELD(act_offset1, registers_.act_offset[1])
TEST_NPU5_DPUINVARIANT_REG_FIELD(act_offset2, registers_.act_offset[2])
TEST_NPU5_DPUINVARIANT_REG_FIELD(act_offset3, registers_.act_offset[3])
// base_offset_a ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(base_offset_a, registers_.base_offset_a)
// base_offset_b ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(base_offset_2, registers_.base_offset_b.base_offset_b_bf.base_offset2)
TEST_NPU5_DPUINVARIANT_REG_FIELD(base_offset_3, registers_.base_offset_b.base_offset_b_bf.base_offset3)
TEST_NPU5_DPUINVARIANT_REG_FIELD(dw_opt_offset, registers_.base_offset_b.base_offset_b_bf.dw_opt_offset)
TEST_NPU5_DPUINVARIANT_REG_FIELD(dw_opt_en, registers_.base_offset_b.base_offset_b_bf.dw_opt_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(dw_3x3s1_opt_dis, registers_.base_offset_b.base_offset_b_bf.dw_3x3s1_opt_dis)
TEST_NPU5_DPUINVARIANT_REG_FIELD(wt_dense_opt_en, registers_.base_offset_b.base_offset_b_bf.wt_dense_opt_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(small_hw_opt_en, registers_.base_offset_b.base_offset_b_bf.small_hw_opt_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(idu_cmx_mux_mode, registers_.base_offset_b.base_offset_b_bf.idu_cmx_mux_mode)
// wt_offset ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(wt_offset, registers_.wt_offset)
// odu_cfg ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(dtype, registers_.odu_cfg.odu_cfg_bf.dtype)
TEST_NPU5_DPUINVARIANT_REG_FIELD(wcb_ac_mode, registers_.odu_cfg.odu_cfg_bf.wcb_ac_mode)
TEST_NPU5_DPUINVARIANT_REG_FIELD(wcb_sp_mode, registers_.odu_cfg.odu_cfg_bf.wcb_sp_mode)
TEST_NPU5_DPUINVARIANT_REG_FIELD(sp_value, registers_.odu_cfg.odu_cfg_bf.sp_value)
TEST_NPU5_DPUINVARIANT_REG_FIELD(sp_out_en, registers_.odu_cfg.odu_cfg_bf.sp_out_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(cmx_port_muxing_disable, registers_.odu_cfg.odu_cfg_bf.cmx_port_muxing_disable)
TEST_NPU5_DPUINVARIANT_REG_FIELD(write_sp, registers_.odu_cfg.odu_cfg_bf.write_sp)
TEST_NPU5_DPUINVARIANT_REG_FIELD(write_pt, registers_.odu_cfg.odu_cfg_bf.write_pt)
TEST_NPU5_DPUINVARIANT_REG_FIELD(write_ac, registers_.odu_cfg.odu_cfg_bf.write_ac)
TEST_NPU5_DPUINVARIANT_REG_FIELD(mode, registers_.odu_cfg.odu_cfg_bf.mode)
TEST_NPU5_DPUINVARIANT_REG_FIELD(grid, registers_.odu_cfg.odu_cfg_bf.grid)
TEST_NPU5_DPUINVARIANT_REG_FIELD(swizzle_key, registers_.odu_cfg.odu_cfg_bf.swizzle_key)
TEST_NPU5_DPUINVARIANT_REG_FIELD(wl_bp_on_start_en, registers_.odu_cfg.odu_cfg_bf.wl_bp_on_start_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(nthw, registers_.odu_cfg.odu_cfg_bf.nthw)
TEST_NPU5_DPUINVARIANT_REG_FIELD(permutation, registers_.odu_cfg.odu_cfg_bf.permutation)
TEST_NPU5_DPUINVARIANT_REG_FIELD(wcb_stall_avoidance, registers_.odu_cfg.odu_cfg_bf.wcb_stall_avoidance)
TEST_NPU5_DPUINVARIANT_REG_FIELD(wcb_bypass, registers_.odu_cfg.odu_cfg_bf.wcb_bypass)
// odu_be_size ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(odu_be_size, registers_.odu_be_size)
// odu_be_cnt ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(odu_be_cnt, registers_.odu_be_cnt)
// odu_se_size ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(odu_se_size, registers_.odu_se_size)
// te_dim0 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(te_dim_y, registers_.te_dim0.te_dim0_bf.te_dim_y)
TEST_NPU5_DPUINVARIANT_REG_FIELD(te_dim_z, registers_.te_dim0.te_dim0_bf.te_dim_z)
// te_dim1 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(te_dim_x, registers_.te_dim1.te_dim1_bf.te_dim_x)
// pt_base ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(pt_base, registers_.pt_base)
// sp_base ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(sp_base, registers_.sp_base)
// mpe_cfg ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(mpe_wtbias, registers_.mpe_cfg.mpe_cfg_bf.mpe_wtbias)
TEST_NPU5_DPUINVARIANT_REG_FIELD(mpe_actbias, registers_.mpe_cfg.mpe_cfg_bf.mpe_actbias)
TEST_NPU5_DPUINVARIANT_REG_FIELD(mpe_mode, registers_.mpe_cfg.mpe_cfg_bf.mpe_mode)
TEST_NPU5_DPUINVARIANT_REG_FIELD(mpe_dense, registers_.mpe_cfg.mpe_cfg_bf.mpe_dense)
TEST_NPU5_DPUINVARIANT_REG_FIELD(mrm_weight_dense, registers_.mpe_cfg.mpe_cfg_bf.mrm_weight_dense)
TEST_NPU5_DPUINVARIANT_REG_FIELD(mrm_act_dense, registers_.mpe_cfg.mpe_cfg_bf.mrm_act_dense)
TEST_NPU5_DPUINVARIANT_REG_FIELD(mpe_daz, registers_.mpe_cfg.mpe_cfg_bf.mpe_daz)
TEST_NPU5_DPUINVARIANT_REG_FIELD(mpe_ftz, registers_.mpe_cfg.mpe_cfg_bf.mpe_ftz)
// mpe_bus_data_sel ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(mpe_bus_data_sel, registers_.mpe_bus_data_sel)
// elop_scale ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(elop_scale_b, registers_.elop_scale.elop_scale_bf.elop_scale_b)
TEST_NPU5_DPUINVARIANT_REG_FIELD(elop_scale_a, registers_.elop_scale.elop_scale_bf.elop_scale_a)
// ppe_cfg ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_g8_bias_c, registers_.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_g8_bias_b, registers_.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_g8_bias_a, registers_.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a)
// ppe_bias ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_bias, registers_.ppe_bias)
// ppe_scale ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_scale_shift, registers_.ppe_scale.ppe_scale_bf.ppe_scale_shift)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_scale_round, registers_.ppe_scale.ppe_scale_bf.ppe_scale_round)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_scale_mult, registers_.ppe_scale.ppe_scale_bf.ppe_scale_mult)
// ppe_scale_ctrl ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_scale_override, registers_.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp_scale_override,
                                 registers_.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override)
// ppe_prelu ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_prelu_shift, registers_.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_prelu_mult, registers_.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult)
// ppe_scale_hclamp ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp_scale_hclamp, registers_.ppe_scale_hclamp)
// ppe_scale_lclamp ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp_scale_lclamp, registers_.ppe_scale_lclamp)
// ppe_misc ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_mode, registers_.ppe_misc.ppe_misc_bf.ppe_mode)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp16_ftz, registers_.ppe_misc.ppe_misc_bf.ppe_fp16_ftz)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp16_clamp, registers_.ppe_misc.ppe_misc_bf.ppe_fp16_clamp)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_i32_convert, registers_.ppe_misc.ppe_misc_bf.ppe_i32_convert)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_sb_dtype, registers_.ppe_misc.ppe_misc_bf.ppe_sb_dtype)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_mult2_mode, registers_.ppe_misc.ppe_misc_bf.ppe_mult2_mode)
// ppe_fp_bias ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp_bias, registers_.ppe_fp_bias)
// ppe_fp_scale ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp_scale, registers_.ppe_fp_scale)
// ppe_fp_prelu ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp_prelu, registers_.ppe_fp_prelu)
// ppe_fp_cfg ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp_convert, registers_.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp_bypass, registers_.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_bf16_round, registers_.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_bf16_round)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_fp_prelu_en, registers_.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en)
// odu_ac_base ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ac_base, registers_.odu_ac_base.odu_ac_base_bf.ac_base)
// hwp_ctrl ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(hwp_en, registers_.hwp_ctrl.hwp_ctrl_bf.hwp_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(hwp_stat_mode, registers_.hwp_ctrl.hwp_ctrl_bf.hwp_stat_mode)
TEST_NPU5_DPUINVARIANT_REG_FIELD(local_timer_en, registers_.hwp_ctrl.hwp_ctrl_bf.local_timer_en)
TEST_NPU5_DPUINVARIANT_REG_FIELD(local_timer_rst, registers_.hwp_ctrl.hwp_ctrl_bf.local_timer_rst)
TEST_NPU5_DPUINVARIANT_REG_FIELD(unique_ID, registers_.hwp_ctrl.hwp_ctrl_bf.unique_id)  // TODO: E#80252
// hwp_cmx_mem_addr ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(hwp_cmx_mem_addr, registers_.hwp_cmx_mem_addr.hwp_cmx_mem_addr)
// odu_cast0 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(cast_enable0, registers_.odu_cast[0].odu_cast_bf.cast_enable)
TEST_NPU5_DPUINVARIANT_REG_FIELD(cast_offset0, registers_.odu_cast[0].odu_cast_bf.cast_offset)
// odu_cast1 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(cast_enable1, registers_.odu_cast[1].odu_cast_bf.cast_enable)
TEST_NPU5_DPUINVARIANT_REG_FIELD(cast_offset1, registers_.odu_cast[1].odu_cast_bf.cast_offset)
// odu_cast2 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(cast_enable2, registers_.odu_cast[2].odu_cast_bf.cast_enable)
TEST_NPU5_DPUINVARIANT_REG_FIELD(cast_offset2, registers_.odu_cast[2].odu_cast_bf.cast_offset)
// tensor2_start ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(tensor2_start, registers_.tensor2_start)
// ppe_lut_ptr ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_lut_ptr, registers_.ppe_lut_ptr.ppe_lut_ptr_bf.ppe_lut_ptr)
TEST_NPU5_DPUINVARIANT_REG_FIELD(ppe_lut_ptr_force, registers_.ppe_lut_ptr.ppe_lut_ptr_bf.ppe_lut_ptr_force)
// nvar_tag ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(nvar_tag, registers_.nvar_tag)
// pallet ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(plt_idx_0, registers_.pallet[0])
TEST_NPU5_DPUINVARIANT_REG_FIELD_SHIFT(plt_idx_1, registers_.pallet[0], 16)
TEST_NPU5_DPUINVARIANT_REG_FIELD(plt_idx_2, registers_.pallet[1])
TEST_NPU5_DPUINVARIANT_REG_FIELD_SHIFT(plt_idx_3, registers_.pallet[1], 16)
TEST_NPU5_DPUINVARIANT_REG_FIELD(plt_idx_4, registers_.pallet[2])
TEST_NPU5_DPUINVARIANT_REG_FIELD_SHIFT(plt_idx_5, registers_.pallet[2], 16)
TEST_NPU5_DPUINVARIANT_REG_FIELD(plt_idx_6, registers_.pallet[3])
TEST_NPU5_DPUINVARIANT_REG_FIELD_SHIFT(plt_idx_7, registers_.pallet[3], 16)
TEST_NPU5_DPUINVARIANT_REG_FIELD(plt_idx_8, registers_.pallet[4])
TEST_NPU5_DPUINVARIANT_REG_FIELD_SHIFT(plt_idx_9, registers_.pallet[4], 16)
TEST_NPU5_DPUINVARIANT_REG_FIELD(plt_idx_10, registers_.pallet[5])
TEST_NPU5_DPUINVARIANT_REG_FIELD_SHIFT(plt_idx_11, registers_.pallet[5], 16)
TEST_NPU5_DPUINVARIANT_REG_FIELD(plt_idx_12, registers_.pallet[6])
TEST_NPU5_DPUINVARIANT_REG_FIELD_SHIFT(plt_idx_13, registers_.pallet[6], 16)
TEST_NPU5_DPUINVARIANT_REG_FIELD(plt_idx_14, registers_.pallet[7])
TEST_NPU5_DPUINVARIANT_REG_FIELD_SHIFT(plt_idx_15, registers_.pallet[7], 16)
// se_addr1 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(se_addr1, registers_.se_addr1)
// sparsity_addr1 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(sparsity_addr1, registers_.sparsity_addr1)
// se_addr2 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(se_addr2, registers_.se_addr2)
// sparsity_addr2 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(sparsity_addr2, registers_.sparsity_addr2)
// se_addr3 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(se_addr3, registers_.se_addr3)
// sparsity_addr3 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(sparsity_addr3, registers_.sparsity_addr3)
// se_sp_size1 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(se_sp_size1, registers_.se_sp_size1)
// se_sp_size2 ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(se_sp_size2, registers_.se_sp_size2)
// barriers_ ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(barriers_wait_mask_hi_, barriers_.wait_mask_hi_)
TEST_NPU5_DPUINVARIANT_REG_FIELD(barriers_wait_mask_lo_, barriers_.wait_mask_lo_)
TEST_NPU5_DPUINVARIANT_REG_FIELD(barriers_post_mask_hi_, barriers_.post_mask_hi_)
TEST_NPU5_DPUINVARIANT_REG_FIELD(barriers_post_mask_lo_, barriers_.post_mask_lo_)
// barriers_sched_ ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_MULTIPLE_REGS_FIELD(barriers_sched_, start_after_, barriers_sched_.start_after_)
TEST_NPU5_DPUINVARIANT_MULTIPLE_REGS_FIELD(barriers_sched_, clean_after_, barriers_sched_.clean_after_)
// variant_count_ ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(variant_count_, variant_count_)
// cluster_ ---------------------------------------------------------------------
TEST_NPU5_DPUINVARIANT_REG_FIELD(cluster_invariant_, cluster_)
