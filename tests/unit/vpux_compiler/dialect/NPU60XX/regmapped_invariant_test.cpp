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

class NPUReg60XX_DpuInvariantRegisterTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuDPUInvariant,
                                       vpux::NPUReg60XX::Descriptors::DpuInvariantRegister> {};

#define TEST_NPU6_DPUINVARIANT_REG_FIELD(FieldType, DescriptorMember)              \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg60XX_DpuInvariantRegisterTest, FieldType, \
                                   vpux::NPUReg60XX::Fields::FieldType, DescriptorMember, 0)

#define TEST_NPU6_DPUINVARIANT_MULTIPLE_REGS_FIELD(ParentRegType, FieldType, DescriptorMember)             \
    HELPER_TEST_NPU_MULTIPLE_REGS_FIELD(NPUReg60XX_DpuInvariantRegisterTest, ParentRegType##__##FieldType, \
                                        vpux::NPUReg60XX::Registers::ParentRegType,                        \
                                        vpux::NPUReg60XX::Fields::FieldType, DescriptorMember, 0)

#define TEST_NPU6_DPUINVARIANT_REG_FIELD_SHIFT(FieldType, DescriptorMember, LeftShiftBitsCount) \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg60XX_DpuInvariantRegisterTest, FieldType,              \
                                   vpux::NPUReg60XX::Fields::FieldType, DescriptorMember, LeftShiftBitsCount)

// se_addr ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(se_addr, registers_.se_addr)
// sparsity_addr ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(sparsity_addr, registers_.sparsity_addr)
// se_size ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(se_size, registers_.se_size)
// z_config ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(se_z_split, registers_.z_config.z_config_bf.se_z_split)
TEST_NPU6_DPUINVARIANT_REG_FIELD(num_ses_in_z_dir, registers_.z_config.z_config_bf.num_ses_in_z_dir)
TEST_NPU6_DPUINVARIANT_REG_FIELD(cm_sp_pattern, registers_.z_config.z_config_bf.cm_sp_pattern)
TEST_NPU6_DPUINVARIANT_REG_FIELD(npo2_se_z_split_en, registers_.z_config.z_config_bf.npo2_se_z_split_enable)
TEST_NPU6_DPUINVARIANT_REG_FIELD(subtract_mpe_wtbias, registers_.z_config.z_config_bf.subtract_mpe_wtbias)
TEST_NPU6_DPUINVARIANT_REG_FIELD(addr_format_sel, registers_.z_config.z_config_bf.addr_format_sel)
// kernel_pad_cfg ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(mpe_assign, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign)
TEST_NPU6_DPUINVARIANT_REG_FIELD(pad_right_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pad_right_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(pad_left_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pad_left_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(pad_bottom_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pad_bottom_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(pad_top_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pad_top_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(kernel_y, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_y)
TEST_NPU6_DPUINVARIANT_REG_FIELD(kernel_x, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_x)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wt_plt_cfg, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.wt_plt_cfg)
TEST_NPU6_DPUINVARIANT_REG_FIELD(act_dense, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wt_dense, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense)
TEST_NPU6_DPUINVARIANT_REG_FIELD(stride_y_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(stride_y, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dynamic_bw_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dw_wt_sp_ins, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.dw_wt_sp_ins)
TEST_NPU6_DPUINVARIANT_REG_FIELD(layer1_wt_sp_ins, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_wt_sp_ins)
TEST_NPU6_DPUINVARIANT_REG_FIELD(layer1_cmp_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_cmp_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(pool_opt_en, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.pool_opt_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(sdp_opt_dis, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.sdp_opt_dis)
TEST_NPU6_DPUINVARIANT_REG_FIELD(ctrl_cmx_opt_dis, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.ctrl_cmx_opt_dis)
TEST_NPU6_DPUINVARIANT_REG_FIELD(ctrl_se_opt_dis, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.ctrl_se_opt_dis)
TEST_NPU6_DPUINVARIANT_REG_FIELD(sp_se_tbl_segment, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wt_plt_cfg_2, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.wt_plt_cfg_2)
TEST_NPU6_DPUINVARIANT_REG_FIELD(flush_wl, registers_.kernel_pad_cfg.kernel_pad_cfg_bf.flush_wl)
// tensor_size0 ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(tensor_size_x, registers_.tensor_size0.tensor_size0_bf.tensor_size_x)
TEST_NPU6_DPUINVARIANT_REG_FIELD(tensor_size_y, registers_.tensor_size0.tensor_size0_bf.tensor_size_y)
// tensor_size1 ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(tensor_size_z, registers_.tensor_size1.tensor_size1_bf.tensor_size_z)
TEST_NPU6_DPUINVARIANT_REG_FIELD(npo2_se_size, registers_.tensor_size1.tensor_size1_bf.npo2_se_size)
TEST_NPU6_DPUINVARIANT_REG_FIELD(tensor_cmp_size, registers_.tensor_size1.tensor_size1_bf.tensor_cmp_size)
TEST_NPU6_DPUINVARIANT_REG_FIELD(tensor_cmp_offset, registers_.tensor_size1.tensor_size1_bf.tensor_cmp_offset)
TEST_NPU6_DPUINVARIANT_REG_FIELD(weight_table_format, registers_.tensor_size1.tensor_size1_bf.weight_table_format)
// tensor_start ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(tensor_start, registers_.tensor_start)
// tensor_mode ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(wmode, registers_.tensor_mode.tensor_mode_bf.wmode)
TEST_NPU6_DPUINVARIANT_REG_FIELD(amode, registers_.tensor_mode.tensor_mode_bf.amode)
TEST_NPU6_DPUINVARIANT_REG_FIELD(stride, registers_.tensor_mode.tensor_mode_bf.stride)
TEST_NPU6_DPUINVARIANT_REG_FIELD(zm_input, registers_.tensor_mode.tensor_mode_bf.zm_input)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dw_input, registers_.tensor_mode.tensor_mode_bf.dw_input)
TEST_NPU6_DPUINVARIANT_REG_FIELD(cm_input, registers_.tensor_mode.tensor_mode_bf.cm_input)
TEST_NPU6_DPUINVARIANT_REG_FIELD(workload_operation, registers_.tensor_mode.tensor_mode_bf.workload_operation)
TEST_NPU6_DPUINVARIANT_REG_FIELD(pad_value, registers_.tensor_mode.tensor_mode_bf.pad_value)
// elops_sparsity_addr ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(elops_sparsity_addr, registers_.elops_sparsity_addr)
// elops_se_addr ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(elops_se_addr, registers_.elops_se_addr)
// elops_wload ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(elop_wload, registers_.elops_wload.elops_wload_bf.elop_wload)
TEST_NPU6_DPUINVARIANT_REG_FIELD(seed_wload, registers_.elops_wload.elops_wload_bf.seed_wload)
TEST_NPU6_DPUINVARIANT_REG_FIELD(fifo_wr_wload, registers_.elops_wload.elops_wload_bf.fifo_wr_wload)
TEST_NPU6_DPUINVARIANT_REG_FIELD(elop_wload_type, registers_.elops_wload.elops_wload_bf.elop_wload_type)
TEST_NPU6_DPUINVARIANT_REG_FIELD(pool_wt_data, registers_.elops_wload.elops_wload_bf.pool_wt_data)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wt_pack_en, registers_.elops_wload.elops_wload_bf.wt_pack_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wt_ptr_dis_pack_en, registers_.elops_wload.elops_wload_bf.wt_ptr_dis_pack_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(act_act_wt_ptr_dis, registers_.elops_wload.elops_wload_bf.act_act_wt_ptr_dis)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wt_ptr_dis, registers_.elops_wload.elops_wload_bf.wt_ptr_dis)
TEST_NPU6_DPUINVARIANT_REG_FIELD(pool_wt_rd_dis, registers_.elops_wload.elops_wload_bf.pool_wt_rd_dis)
TEST_NPU6_DPUINVARIANT_REG_FIELD(elop_operation, registers_.elops_wload.elops_wload_bf.elop_operation)
TEST_NPU6_DPUINVARIANT_REG_FIELD(repeat_dis, registers_.elops_wload.elops_wload_bf.repeat_dis)
TEST_NPU6_DPUINVARIANT_REG_FIELD(elop_tensor2_cmp, registers_.elops_wload.elops_wload_bf.elop_tensor2_cmp)
// act_offset ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(act_offset0, registers_.act_offset)
// base_offset_a ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(base_offset_a, registers_.base_offset_a)
// base_offset_b ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(base_offset_2, registers_.base_offset_b.base_offset_b_bf.base_offset2)
TEST_NPU6_DPUINVARIANT_REG_FIELD(base_offset_3, registers_.base_offset_b.base_offset_b_bf.base_offset3)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dw_opt_offset, registers_.base_offset_b.base_offset_b_bf.dw_opt_offset)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dw_opt_en, registers_.base_offset_b.base_offset_b_bf.dw_opt_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dw_3x3s1_opt_dis, registers_.base_offset_b.base_offset_b_bf.dw_3x3s1_opt_dis)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wt_dense_opt_en, registers_.base_offset_b.base_offset_b_bf.wt_dense_opt_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(small_hw_opt_en, registers_.base_offset_b.base_offset_b_bf.small_hw_opt_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(idu_cmx_mux_mode, registers_.base_offset_b.base_offset_b_bf.idu_cmx_mux_mode)
TEST_NPU6_DPUINVARIANT_REG_FIELD(per_group_zero_point, registers_.base_offset_b.base_offset_b_bf.per_group_zero_point)
TEST_NPU6_DPUINVARIANT_REG_FIELD(zero_point_only, registers_.base_offset_b.base_offset_b_bf.zero_point_only)
// wt_offset ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(wt_offset, registers_.wt_offset)
// odu_cfg ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(dtype, registers_.odu_cfg.odu_cfg_bf.dtype)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wcb_ac_mode, registers_.odu_cfg.odu_cfg_bf.wcb_ac_mode)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wcb_sp_mode, registers_.odu_cfg.odu_cfg_bf.wcb_sp_mode)
TEST_NPU6_DPUINVARIANT_REG_FIELD(sp_value, registers_.odu_cfg.odu_cfg_bf.sp_value)
TEST_NPU6_DPUINVARIANT_REG_FIELD(sp_out_en, registers_.odu_cfg.odu_cfg_bf.sp_out_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(cmx_port_muxing_disable, registers_.odu_cfg.odu_cfg_bf.cmx_port_muxing_disable)
TEST_NPU6_DPUINVARIANT_REG_FIELD(write_sp, registers_.odu_cfg.odu_cfg_bf.write_sp)
TEST_NPU6_DPUINVARIANT_REG_FIELD(write_pt, registers_.odu_cfg.odu_cfg_bf.write_pt)
TEST_NPU6_DPUINVARIANT_REG_FIELD(write_ac, registers_.odu_cfg.odu_cfg_bf.write_ac)
TEST_NPU6_DPUINVARIANT_REG_FIELD(mode, registers_.odu_cfg.odu_cfg_bf.mode)
TEST_NPU6_DPUINVARIANT_REG_FIELD(grid, registers_.odu_cfg.odu_cfg_bf.grid)
TEST_NPU6_DPUINVARIANT_REG_FIELD(swizzle_key, registers_.odu_cfg.odu_cfg_bf.swizzle_key)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wl_bp_on_start_en, registers_.odu_cfg.odu_cfg_bf.wl_bp_on_start_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(nthw, registers_.odu_cfg.odu_cfg_bf.nthw)
TEST_NPU6_DPUINVARIANT_REG_FIELD(permutation, registers_.odu_cfg.odu_cfg_bf.permutation)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wcb_stall_avoidance, registers_.odu_cfg.odu_cfg_bf.wcb_stall_avoidance)
TEST_NPU6_DPUINVARIANT_REG_FIELD(wcb_bypass, registers_.odu_cfg.odu_cfg_bf.wcb_bypass)
// odu_cfg_2 ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(mn_enable, registers_.odu_cfg_2.odu_cfg_2_bf.mn_enable)
TEST_NPU6_DPUINVARIANT_REG_FIELD(mx_enable, registers_.odu_cfg_2.odu_cfg_2_bf.mx_enable)
TEST_NPU6_DPUINVARIANT_REG_FIELD(mt_enable, registers_.odu_cfg_2.odu_cfg_2_bf.mt_enable)
TEST_NPU6_DPUINVARIANT_REG_FIELD(local_region_enable, registers_.odu_cfg_2.odu_cfg_2_bf.local_region_enable)
TEST_NPU6_DPUINVARIANT_REG_FIELD(local_region_suppress, registers_.odu_cfg_2.odu_cfg_2_bf.local_region_suppress)
TEST_NPU6_DPUINVARIANT_REG_FIELD(min_max_dtype, registers_.odu_cfg_2.odu_cfg_2_bf.min_max_dtype)
TEST_NPU6_DPUINVARIANT_REG_FIELD(min_max_total_tile_select, registers_.odu_cfg_2.odu_cfg_2_bf.min_max_total_tile_select)
TEST_NPU6_DPUINVARIANT_REG_FIELD(plt_ptr, registers_.odu_cfg_2.odu_cfg_2_bf.plt_ptr)
// odu_mn_base ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(mn_base, registers_.odu_mn_base.odu_mn_base_bf.mn_base)
// odu_mx_base ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(mx_base, registers_.odu_mx_base.odu_mx_base_bf.mx_base)
// te_dim0 ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(te_dim_y, registers_.te_dim0.te_dim0_bf.te_dim_y)
TEST_NPU6_DPUINVARIANT_REG_FIELD(te_dim_z, registers_.te_dim0.te_dim0_bf.te_dim_z)
// te_dim1 ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(te_dim_x, registers_.te_dim1.te_dim1_bf.te_dim_x)
// mt_base ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(mt_base, registers_.odu_mt_base.odu_mt_base_bf.mt_base)
// sp_base ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(sp_base, registers_.sp_base)
// mpe_cfg ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(mpe_wtbias, registers_.mpe_cfg.mpe_cfg_bf.mpe_wtbias)
TEST_NPU6_DPUINVARIANT_REG_FIELD(mpe_actbias, registers_.mpe_cfg.mpe_cfg_bf.mpe_actbias)
TEST_NPU6_DPUINVARIANT_REG_FIELD(mpe_mode, registers_.mpe_cfg.mpe_cfg_bf.mpe_mode)
TEST_NPU6_DPUINVARIANT_REG_FIELD(mpe_dense, registers_.mpe_cfg.mpe_cfg_bf.mpe_dense)
TEST_NPU6_DPUINVARIANT_REG_FIELD(mrm_weight_dense, registers_.mpe_cfg.mpe_cfg_bf.mrm_weight_dense)
TEST_NPU6_DPUINVARIANT_REG_FIELD(mrm_act_dense, registers_.mpe_cfg.mpe_cfg_bf.mrm_act_dense)
TEST_NPU6_DPUINVARIANT_REG_FIELD(mpe_daz, registers_.mpe_cfg.mpe_cfg_bf.mpe_daz)
TEST_NPU6_DPUINVARIANT_REG_FIELD(mpe_ftz, registers_.mpe_cfg.mpe_cfg_bf.mpe_ftz)
TEST_NPU6_DPUINVARIANT_REG_FIELD(cim_en, registers_.mpe_cfg.mpe_cfg_bf.cim_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(cim_ic32_en, registers_.mpe_cfg.mpe_cfg_bf.cim_ic32_en)
// mpe_bus_data_sel ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(mpe_bus_data_sel, registers_.mpe_bus_data_sel)
// elop_scale ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(elop_scale_b, registers_.elop_scale.elop_scale_bf.elop_scale_b)
TEST_NPU6_DPUINVARIANT_REG_FIELD(elop_scale_a, registers_.elop_scale.elop_scale_bf.elop_scale_a)
// ppe_cfg ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_g8_bias_c, registers_.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c)
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_g8_bias_b, registers_.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b)
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_g8_bias_a, registers_.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a)
// ppe_scale_ctrl ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_scale_override, registers_.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override)
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_fp_scale_override,
                                 registers_.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override)
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_scale_dtype, registers_.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_dtype)
// ppe_scale_hclamp ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_fp_scale_hclamp, registers_.ppe_scale_hclamp)
// ppe_scale_lclamp ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_fp_scale_lclamp, registers_.ppe_scale_lclamp)
// ppe_misc ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_mode, registers_.ppe_misc.ppe_misc_bf.ppe_mode)
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_fp16_ftz, registers_.ppe_misc.ppe_misc_bf.ppe_fp16_ftz)
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_fp16_clamp, registers_.ppe_misc.ppe_misc_bf.ppe_fp16_clamp)
// ppe_fp_bias ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_fp_bias, registers_.ppe_fp_bias)
// ppe_fp_scale ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_fp_scale, registers_.ppe_fp_scale)
// ppe_fp_prelu ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_fp_prelu, registers_.ppe_fp_prelu)
// ppe_fp_cfg ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_fp_convert, registers_.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert)
// odu_ac_base ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ac_base, registers_.odu_ac_base.odu_ac_base_bf.ac_base)
// hwp_ctrl ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(hwp_en, registers_.hwp_ctrl.hwp_ctrl_bf.hwp_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(hwp_stat_mode, registers_.hwp_ctrl.hwp_ctrl_bf.hwp_stat_mode)
TEST_NPU6_DPUINVARIANT_REG_FIELD(local_timer_en, registers_.hwp_ctrl.hwp_ctrl_bf.local_timer_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(local_timer_rst, registers_.hwp_ctrl.hwp_ctrl_bf.local_timer_rst)
TEST_NPU6_DPUINVARIANT_REG_FIELD(unique_ID, registers_.hwp_ctrl.hwp_ctrl_bf.unique_id)  // TODO: E#80252
// hwp_cmx_mem_addr ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(hwp_cmx_mem_addr, registers_.hwp_cmx_mem_addr.hwp_cmx_mem_addr)
// tensor2_start ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(tensor2_start, registers_.tensor2_start)
// ppe_lut_ptr ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_lut_ptr, registers_.ppe_lut_ptr.ppe_lut_ptr_bf.ppe_lut_ptr)
TEST_NPU6_DPUINVARIANT_REG_FIELD(ppe_lut_ptr_force, registers_.ppe_lut_ptr.ppe_lut_ptr_bf.ppe_lut_ptr_force)
// invar_tag ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(nvar_tag, registers_.nvar_tag)
// dyn_prim_cfg ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_primitive_id, registers_.dyn_prim_cfg.dyn_prim_cfg_bf.dyn_primitive_id)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_primitive_id2, registers_.dyn_prim_cfg.dyn_prim_cfg_bf.dyn_primitive_id2)
// dyn_scale ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_scale_fetch_addr, registers_.dyn_scale.dyn_scale_bf.dyn_scale_fetch_addr)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_scale_fetch_addr_en, registers_.dyn_scale.dyn_scale_bf.dyn_scale_fetch_addr_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_ppe_pad_ovrwr_en, registers_.dyn_scale.dyn_scale_bf.dyn_ppe_pad_ovrwr_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_idu_wt_num_mod_en, registers_.dyn_scale.dyn_scale_bf.dyn_idu_wt_num_mod_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_odu_z_end_mod_en, registers_.dyn_scale.dyn_scale_bf.dyn_odu_z_end_mod_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_conv_cont_mod_en, registers_.dyn_scale.dyn_scale_bf.dyn_conv_cont_mod_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_idu_ic_mod_en, registers_.dyn_scale.dyn_scale_bf.dyn_idu_ic_mod_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_scale_cfg, registers_.dyn_scale.dyn_scale_bf.dyn_scale_cfg)
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_stat_en, registers_.dyn_scale.dyn_scale_bf.dyn_stat_en)
TEST_NPU6_DPUINVARIANT_REG_FIELD(hwp_stall_dis, registers_.dyn_scale.dyn_scale_bf.hwp_stall_dis)
// dyn_ppe_data ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(dyn_ppe_data, registers_.dyn_ppe_data)
// halo_0_offset ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(halo_0_mm_adr_offset, registers_.halo_0_offset.halo_0_offset_bf.halo_0_mm_adr_offset)
// halo_1_offset ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(halo_1_mm_adr_offset, registers_.halo_1_offset.halo_1_offset_bf.halo_1_mm_adr_offset)
// halo_2_offset ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(halo_2_mm_adr_offset, registers_.halo_2_offset.halo_2_offset_bf.halo_2_mm_adr_offset)
// halo_3_offset ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(halo_3_mm_adr_offset, registers_.halo_3_offset.halo_3_offset_bf.halo_3_mm_adr_offset)
// halo_4_offset ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(halo_4_mm_adr_offset, registers_.halo_4_offset.halo_4_offset_bf.halo_4_mm_adr_offset)
// odu_local_a ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(local_region_beg_x, registers_.odu_local_a.odu_local_a_bf.local_region_beg_x)
TEST_NPU6_DPUINVARIANT_REG_FIELD(local_region_end_x, registers_.odu_local_a.odu_local_a_bf.local_region_end_x)
// odu_local_b ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(local_region_beg_y, registers_.odu_local_b.odu_local_b_bf.local_region_beg_y)
TEST_NPU6_DPUINVARIANT_REG_FIELD(local_region_end_y, registers_.odu_local_b.odu_local_b_bf.local_region_end_y)
// barriers_ ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(barriers_wait_mask_hi_, barriers_.wait_mask_hi_)
TEST_NPU6_DPUINVARIANT_REG_FIELD(barriers_wait_mask_lo_, barriers_.wait_mask_lo_)
TEST_NPU6_DPUINVARIANT_REG_FIELD(barriers_post_mask_hi_, barriers_.post_mask_hi_)
TEST_NPU6_DPUINVARIANT_REG_FIELD(barriers_post_mask_lo_, barriers_.post_mask_lo_)
// barriers_sched_ ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_MULTIPLE_REGS_FIELD(barriers_sched_, start_after_, barriers_sched_.start_after_)
TEST_NPU6_DPUINVARIANT_MULTIPLE_REGS_FIELD(barriers_sched_, clean_after_, barriers_sched_.clean_after_)
// variant_count_ ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(variant_count_, variant_count_)
// cluster_ ---------------------------------------------------------------------
TEST_NPU6_DPUINVARIANT_REG_FIELD(cluster_invariant_, cluster_)
