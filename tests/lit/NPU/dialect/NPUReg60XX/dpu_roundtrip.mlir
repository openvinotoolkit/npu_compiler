//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: dev-build && platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @DPUTest {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
    DataInfo "input_1" : tensor<16x1x1x1xi64>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }
  func.func @main() {
    ELF.Main {
      ELF.CreateLogicalSection @builtin.data.nncmx0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <196736> : memref<1x16x16x16xf16, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @builtin.tasks.DPUVariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @builtin.tasks.DPUInvariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateSection @text.invariants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        NPUReg60XX.DPUInvariant descriptor = <
          DpuInvariantRegister {
            se_addr = UINT 0,
            sparsity_addr = UINT 0,
            se_size = UINT 0,
            z_config {
              UINT se_z_split = 0,
              UINT num_ses_in_z_dir = 0,
              UINT cm_sp_pattern = 0,
              UINT npo2_se_z_split_en = 0,
              UINT subtract_mpe_wtbias = 0,
              UINT addr_format_sel = 1,
            }
            kernel_pad_cfg {
              UINT mpe_assign = 0,
              UINT pad_right_en = 0,
              UINT pad_left_en = 0,
              UINT pad_bottom_en = 0,
              UINT pad_top_en = 0,
              UINT kernel_y = 0,
              UINT kernel_x = 0,
              UINT wt_plt_cfg = 0,
              UINT act_dense = 1,
              UINT wt_dense = 0,
              UINT stride_y_en = 0,
              UINT stride_y = 0,
              UINT dynamic_bw_en = 0,
              UINT dw_wt_sp_ins = 0,
              UINT layer1_wt_sp_ins = 0,
              UINT layer1_cmp_en = 0,
              UINT pool_opt_en = 0,
              UINT sdp_opt_dis = 0,
              UINT ctrl_cmx_opt_dis = 0,
              UINT ctrl_se_opt_dis = 0,
              UINT sp_se_tbl_segment = 0,
              UINT wt_plt_cfg_2 = 0,
              UINT flush_wl = 0,
            }
            tensor_size0 {
              UINT tensor_size_x = 0x10,
              UINT tensor_size_y = 0x10,
            }
            tensor_size1 {
              UINT tensor_size_z = 0x10,
              UINT npo2_se_size = 0,
              UINT tensor_cmp_size = 0,
              UINT tensor_cmp_offset = 0,
              UINT weight_table_format = 0,
            }
            tensor_start = UINT 0,
            tensor_mode {
              UINT wmode = 0,
              UINT amode = 0,
              UINT stride = 0,
              UINT zm_input = 0,
              UINT dw_input = 0,
              UINT cm_input = 0,
              UINT workload_operation = 0,
              UINT pad_value = 0,
            }
            elops_sparsity_addr = UINT 0,
            elops_se_addr = UINT 0,
            elops_wload {
              UINT elop_wload = 0,
              UINT seed_wload = 0,
              UINT fifo_wr_wload = 0,
              UINT elop_wload_type = 0,
              UINT pool_wt_data = 0,
              UINT wt_pack_en = 0,
              UINT wt_ptr_dis_pack_en = 0,
              UINT act_act_wt_ptr_dis = 0,
              UINT wt_ptr_dis = 0,
              UINT pool_wt_rd_dis = 0,
              UINT elop_operation = 0,
              UINT repeat_dis = 0,
              UINT elop_tensor2_cmp = 0,
            }
            act_offset0 = UINT 0,
            base_offset_a = UINT 0x200,
            base_offset_b {
              UINT base_offset_2 = 0,
              UINT base_offset_3 = 0,
              UINT dw_opt_offset = 0,
              UINT dw_opt_en = 0,
              UINT dw_3x3s1_opt_dis = 0,
              UINT wt_dense_opt_en = 0,
              UINT small_hw_opt_en = 0,
              UINT idu_cmx_mux_mode = 0,
              UINT per_group_zero_point = 0,
              UINT zero_point_only = 0,
            }
            wt_offset = UINT 0,
            odu_cfg {
              UINT dtype = 4,
              UINT wcb_ac_mode = 0,
              UINT wcb_sp_mode = 0,
              UINT sp_value = 0,
              UINT sp_out_en = 0,
              UINT cmx_port_muxing_disable = 0,
              UINT write_sp = 0,
              UINT write_pt = 0,
              UINT write_ac = 1,
              UINT mode = 0,
              UINT grid = 0,
              UINT swizzle_key = 0,
              UINT wl_bp_on_start_en = 0,
              UINT nthw = 0,
              UINT permutation = 0,
              UINT wcb_stall_avoidance = 0,
              UINT wcb_bypass = 0,
            }
            odu_cfg_2 {
              UINT mn_enable = 0,
              UINT mx_enable = 0,
              UINT mt_enable = 0,
              UINT local_region_enable = 0,
              UINT local_region_suppress = 0,
              UINT min_max_dtype = 0,
              UINT min_max_total_tile_select = 0,
              UINT plt_ptr = 0,
            }
            odu_mn_base {
              UINT mn_base = 0,
            }
            odu_mx_base {
              UINT mx_base = 0,
            }
            te_dim0 {
              UINT te_dim_y = 0x3F,
              UINT te_dim_z = 0xF,
            }
            te_dim1 {
              UINT te_dim_x = 0x3F,
            }
            mt_base {
              UINT mt_base = 0,
            }
            sp_base = UINT 0,
            mpe_cfg {
              UINT mpe_wtbias = 0,
              UINT mpe_actbias = 0,
              UINT mpe_mode = 0,
              UINT mpe_dense = 0,
              UINT mrm_weight_dense = 0,
              UINT mrm_act_dense = 0,
              UINT mpe_daz = 0,
              UINT mpe_ftz = 0,
              UINT cim_en = 0,
              UINT cim_ic32_en = 0,
            }
            mpe_bus_data_sel = UINT 0,
            elop_scale {
              UINT elop_scale_b = 0,
              UINT elop_scale_a = 0,
              FP elop_scale_b_fp = 0,
              FP elop_scale_a_fp = 0,
              BF elop_scale_b_bfp = 0,
              BF elop_scale_a_bfp = 0,
            }
            ppe_cfg {
              SINT ppe_g8_bias_c = 0,
              UINT ppe_g8_bias_b = 0,
              UINT ppe_g8_bias_a = 0,
            }
            unused0_inv = UINT 0,
            unused1_inv = UINT 0,
            ppe_scale_ctrl {
              UINT ppe_scale_override = 0,
              UINT ppe_fp_scale_override = 0,
              UINT ppe_scale_dtype = 0,
            }
            unused2_inv = UINT 0,
            ppe_scale_hclamp {
              FP ppe_fp_scale_hclamp = 0x7F7FFFFF,
            }
            ppe_scale_lclamp {
              FP ppe_fp_scale_lclamp = 0xFF7FFFFF,
            }
            ppe_misc {
              UINT ppe_mode = 0,
              UINT ppe_rnd_fp = 0,
              UINT ppe_rnd_int = 0,
              UINT ppe_fp16_ftz = 0,
              UINT ppe_fp16_clamp = 0,
              UINT ppe_mult2_mode = 0,
            }
            ppe_fp_bias = FP 0,
            ppe_fp_scale = FP 0,
            ppe_fp_prelu = FP 0,
            ppe_fp_cfg {
              UINT ppe_fp_convert = 0,
            }
            odu_ac_base {
              UINT ac_base = 0,
            }
            hwp_ctrl {
              UINT hwp_en = 0,
              UINT hwp_stat_mode = 0,
              UINT local_timer_en = 0,
              UINT local_timer_rst = 0,
              UINT unique_ID = 0,
            }
            hwp_cmx_mem_addr = UINT 0,
            tensor2_start = UINT 0,
            ppe_lut_ptr {
              UINT ppe_lut_ptr = 0,
              UINT ppe_lut_ptr_force = 0,
            }
            nvar_tag = UINT 1,
            dyn_prim_cfg {
              UINT dyn_primitive_id = 0,
              UINT dyn_primitive_id2 = 0,
            }
            dyn_scale {
              UINT dyn_scale_fetch_addr = 0,
              UINT dyn_scale_fetch_addr_en = 0,
              UINT dyn_ppe_pad_ovrwr_en = 0,
              UINT dyn_idu_wt_num_mod_en = 0,
              UINT dyn_odu_z_end_mod_en = 0,
              UINT dyn_conv_cont_mod_en = 0,
              UINT dyn_idu_ic_mod_en = 0,
              UINT dyn_scale_cfg = 0,
              UINT dyn_stat_en = 0,
              UINT hwp_stall_dis = 0,
            }
            dyn_ppe_data = SINT 0,
            odu_local_a {
              UINT local_region_beg_x = 0,
              UINT local_region_end_x = 0,
            }
            odu_local_b {
              UINT local_region_beg_y = 0,
              UINT local_region_end_y = 0,
            }
            halo_0_offset {
              UINT halo_0_mm_adr_offset = 0,
            }
            halo_1_offset {
              UINT halo_1_mm_adr_offset = 0,
            }
            halo_2_offset {
              UINT halo_2_mm_adr_offset = 0,
            }
            halo_3_offset {
              UINT halo_3_mm_adr_offset = 0,
            }
            halo_4_offset {
              UINT halo_4_mm_adr_offset = 0,
            }
            unused3_inv = UINT 0,
            barriers_wait_mask_hi_ {
              UINT barriers_wait_mask_hi_ = 0,
            }
            barriers_wait_mask_lo_ = UINT 0,
            barriers_post_mask_hi_ {
              UINT barriers_post_mask_hi_ = 0,
            }
            barriers_post_mask_lo_ = UINT 0,
            barriers_group_mask_ {
              UINT group_ = 0,
              UINT mask_ = 0,
            }
            barriers_sched_ {
              UINT start_after_ = 0,
              UINT clean_after_ = 0,
            }
            deprecated0_inv = UINT 0,
            variant_count_ = UINT 0,
            cluster_invariant_ = UINT 0,
            deprecated1_inv = UINT 0,
            pad_3 = UINT 0,
          } requires 11:13:2
        > {input = @builtin.data.nncmx0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>,
         output = @builtin.data.nncmx0::@DeclareBuffer_ActOut, sym_name = "DPUInvariant_0", task_index = !VPURegMapped.Index<0:0:0>, task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0}
        // CHECK: NPUReg60XX.DPUInvariant descriptor = <
        // CHECK:   DpuInvariantRegister {
        // CHECK:     se_addr = UINT 0,
        // CHECK:     sparsity_addr = UINT 0,
        // CHECK:     se_size = UINT 0,
        // CHECK:     z_config {
        // CHECK:       UINT se_z_split = 0,
        // CHECK:       UINT num_ses_in_z_dir = 0,
        // CHECK:       UINT cm_sp_pattern = 0,
        // CHECK:       UINT npo2_se_z_split_en = 0,
        // CHECK:       UINT subtract_mpe_wtbias = 0,
        // CHECK:       UINT addr_format_sel = 1,
        // CHECK:     }
        // CHECK:     kernel_pad_cfg {
        // CHECK:       UINT mpe_assign = 0,
        // CHECK:       UINT pad_right_en = 0,
        // CHECK:       UINT pad_left_en = 0,
        // CHECK:       UINT pad_bottom_en = 0,
        // CHECK:       UINT pad_top_en = 0,
        // CHECK:       UINT kernel_y = 0,
        // CHECK:       UINT kernel_x = 0,
        // CHECK:       UINT wt_plt_cfg = 0,
        // CHECK:       UINT act_dense = 1,
        // CHECK:       UINT wt_dense = 0,
        // CHECK:       UINT stride_y_en = 0,
        // CHECK:       UINT stride_y = 0,
        // CHECK:       UINT dynamic_bw_en = 0,
        // CHECK:       UINT dw_wt_sp_ins = 0,
        // CHECK:       UINT layer1_wt_sp_ins = 0,
        // CHECK:       UINT layer1_cmp_en = 0,
        // CHECK:       UINT pool_opt_en = 0,
        // CHECK:       UINT sdp_opt_dis = 0,
        // CHECK:       UINT ctrl_cmx_opt_dis = 0,
        // CHECK:       UINT ctrl_se_opt_dis = 0,
        // CHECK:       UINT sp_se_tbl_segment = 0,
        // CHECK:       UINT wt_plt_cfg_2 = 0,
        // CHECK:       UINT flush_wl = 0,
        // CHECK:     }
        // CHECK:     tensor_size0 {
        // CHECK:       UINT tensor_size_x = 0x10,
        // CHECK:       UINT tensor_size_y = 0x10,
        // CHECK:     }
        // CHECK:     tensor_size1 {
        // CHECK:       UINT tensor_size_z = 0x10,
        // CHECK:       UINT npo2_se_size = 0,
        // CHECK:       UINT tensor_cmp_size = 0,
        // CHECK:       UINT tensor_cmp_offset = 0,
        // CHECK:       UINT weight_table_format = 0,
        // CHECK:     }
        // CHECK:     tensor_start = UINT 0,
        // CHECK:     tensor_mode {
        // CHECK:       UINT wmode = 0,
        // CHECK:       UINT amode = 0,
        // CHECK:       UINT stride = 0,
        // CHECK:       UINT zm_input = 0,
        // CHECK:       UINT dw_input = 0,
        // CHECK:       UINT cm_input = 0,
        // CHECK:       UINT workload_operation = 0,
        // CHECK:       UINT pad_value = 0,
        // CHECK:     }
        // CHECK:     elops_sparsity_addr = UINT 0,
        // CHECK:     elops_se_addr = UINT 0,
        // CHECK:     elops_wload {
        // CHECK:       UINT elop_wload = 0,
        // CHECK:       UINT seed_wload = 0,
        // CHECK:       UINT fifo_wr_wload = 0,
        // CHECK:       UINT elop_wload_type = 0,
        // CHECK:       UINT pool_wt_data = 0,
        // CHECK:       UINT wt_pack_en = 0,
        // CHECK:       UINT wt_ptr_dis_pack_en = 0,
        // CHECK:       UINT act_act_wt_ptr_dis = 0,
        // CHECK:       UINT wt_ptr_dis = 0,
        // CHECK:       UINT pool_wt_rd_dis = 0,
        // CHECK:       UINT elop_operation = 0,
        // CHECK:       UINT repeat_dis = 0,
        // CHECK:       UINT elop_tensor2_cmp = 0,
        // CHECK:     }
        // CHECK:     act_offset0 = UINT 0,
        // CHECK:     base_offset_a = UINT 0x200,
        // CHECK:     base_offset_b {
        // CHECK:       UINT base_offset_2 = 0,
        // CHECK:       UINT base_offset_3 = 0,
        // CHECK:       UINT dw_opt_offset = 0,
        // CHECK:       UINT dw_opt_en = 0,
        // CHECK:       UINT dw_3x3s1_opt_dis = 0,
        // CHECK:       UINT wt_dense_opt_en = 0,
        // CHECK:       UINT small_hw_opt_en = 0,
        // CHECK:       UINT idu_cmx_mux_mode = 0,
        // CHECK:       UINT per_group_zero_point = 0,
        // CHECK:       UINT zero_point_only = 0,
        // CHECK:     }
        // CHECK:     wt_offset = UINT 0,
        // CHECK:     odu_cfg {
        // CHECK:       UINT dtype = 4,
        // CHECK:       UINT wcb_ac_mode = 0,
        // CHECK:       UINT wcb_sp_mode = 0,
        // CHECK:       UINT sp_value = 0,
        // CHECK:       UINT sp_out_en = 0,
        // CHECK:       UINT cmx_port_muxing_disable = 0,
        // CHECK:       UINT write_sp = 0,
        // CHECK:       UINT write_pt = 0,
        // CHECK:       UINT write_ac = 1,
        // CHECK:       UINT mode = 0,
        // CHECK:       UINT grid = 0,
        // CHECK:       UINT swizzle_key = 0,
        // CHECK:       UINT wl_bp_on_start_en = 0,
        // CHECK:       UINT nthw = 0,
        // CHECK:       UINT permutation = 0,
        // CHECK:       UINT wcb_stall_avoidance = 0,
        // CHECK:       UINT wcb_bypass = 0,
        // CHECK:     }
        // CHECK:     odu_cfg_2 {
        // CHECK:       UINT mn_enable = 0,
        // CHECK:       UINT mx_enable = 0,
        // CHECK:       UINT mt_enable = 0,
        // CHECK:       UINT local_region_enable = 0,
        // CHECK:       UINT local_region_suppress = 0,
        // CHECK:       UINT min_max_dtype = 0,
        // CHECK:       UINT min_max_total_tile_select = 0,
        // CHECK:       UINT plt_ptr = 0,
        // CHECK:     }
        // CHECK:     odu_mn_base {
        // CHECK:       UINT mn_base = 0,
        // CHECK:     }
        // CHECK:     odu_mx_base {
        // CHECK:       UINT mx_base = 0,
        // CHECK:     }
        // CHECK:     te_dim0 {
        // CHECK:       UINT te_dim_y = 0x3F,
        // CHECK:       UINT te_dim_z = 0xF,
        // CHECK:     }
        // CHECK:     te_dim1 {
        // CHECK:       UINT te_dim_x = 0x3F,
        // CHECK:     }
        // CHECK:     mt_base {
        // CHECK:       UINT mt_base = 0,
        // CHECK:     }
        // CHECK:     sp_base = UINT 0,
        // CHECK:     mpe_cfg {
        // CHECK:       UINT mpe_wtbias = 0,
        // CHECK:       UINT mpe_actbias = 0,
        // CHECK:       UINT mpe_mode = 0,
        // CHECK:       UINT mpe_dense = 0,
        // CHECK:       UINT mrm_weight_dense = 0,
        // CHECK:       UINT mrm_act_dense = 0,
        // CHECK:       UINT mpe_daz = 0,
        // CHECK:       UINT mpe_ftz = 0,
        // CHECK:       UINT cim_en = 0,
        // CHECK:       UINT cim_ic32_en = 0,
        // CHECK:     }
        // CHECK:     mpe_bus_data_sel = UINT 0,
        // CHECK:     elop_scale {
        // CHECK:       UINT elop_scale_b = 0,
        // CHECK:       UINT elop_scale_a = 0,
        // CHECK:       FP elop_scale_b_fp = 0,
        // CHECK:       FP elop_scale_a_fp = 0,
        // CHECK:       BF elop_scale_b_bfp = 0,
        // CHECK:       BF elop_scale_a_bfp = 0,
        // CHECK:     }
        // CHECK:     ppe_cfg {
        // CHECK:       SINT ppe_g8_bias_c = 0,
        // CHECK:       UINT ppe_g8_bias_b = 0,
        // CHECK:       UINT ppe_g8_bias_a = 0,
        // CHECK:     }
        // CHECK:     unused0_inv = UINT 0,
        // CHECK:     unused1_inv = UINT 0,
        // CHECK:     ppe_scale_ctrl {
        // CHECK:       UINT ppe_scale_override = 0,
        // CHECK:       UINT ppe_fp_scale_override = 0,
        // CHECK:       UINT ppe_scale_dtype = 0,
        // CHECK:     }
        // CHECK:     unused2_inv = UINT 0,
        // CHECK:     ppe_scale_hclamp {
        // CHECK:       FP ppe_fp_scale_hclamp = 0x7F7FFFFF,
        // CHECK:     }
        // CHECK:     ppe_scale_lclamp {
        // CHECK:       FP ppe_fp_scale_lclamp = 0xFF7FFFFF,
        // CHECK:     }
        // CHECK:     ppe_misc {
        // CHECK:       UINT ppe_mode = 0,
        // CHECK:       UINT ppe_rnd_fp = 0,
        // CHECK:       UINT ppe_rnd_int = 0,
        // CHECK:       UINT ppe_fp16_ftz = 0,
        // CHECK:       UINT ppe_fp16_clamp = 0,
        // CHECK:       UINT ppe_mult2_mode = 0,
        // CHECK:     }
        // CHECK:     ppe_fp_bias = FP 0,
        // CHECK:     ppe_fp_scale = FP 0,
        // CHECK:     ppe_fp_prelu = FP 0,
        // CHECK:     ppe_fp_cfg {
        // CHECK:       UINT ppe_fp_convert = 0,
        // CHECK:     }
        // CHECK:     odu_ac_base {
        // CHECK:       UINT ac_base = 0,
        // CHECK:     }
        // CHECK:     hwp_ctrl {
        // CHECK:       UINT hwp_en = 0,
        // CHECK:       UINT hwp_stat_mode = 0,
        // CHECK:       UINT local_timer_en = 0,
        // CHECK:       UINT local_timer_rst = 0,
        // CHECK:       UINT unique_ID = 0,
        // CHECK:     }
        // CHECK:     hwp_cmx_mem_addr = UINT 0,
        // CHECK:     tensor2_start = UINT 0,
        // CHECK:     ppe_lut_ptr {
        // CHECK:       UINT ppe_lut_ptr = 0,
        // CHECK:       UINT ppe_lut_ptr_force = 0,
        // CHECK:     }
        // CHECK:     nvar_tag = UINT 1,
        // CHECK:     dyn_prim_cfg {
        // CHECK:       UINT dyn_primitive_id = 0,
        // CHECK:       UINT dyn_primitive_id2 = 0,
        // CHECK:     }
        // CHECK:     dyn_scale {
        // CHECK:       UINT dyn_scale_fetch_addr = 0,
        // CHECK:       UINT dyn_scale_fetch_addr_en = 0,
        // CHECK:       UINT dyn_ppe_pad_ovrwr_en = 0,
        // CHECK:       UINT dyn_idu_wt_num_mod_en = 0,
        // CHECK:       UINT dyn_odu_z_end_mod_en = 0,
        // CHECK:       UINT dyn_conv_cont_mod_en = 0,
        // CHECK:       UINT dyn_idu_ic_mod_en = 0,
        // CHECK:       UINT dyn_scale_cfg = 0,
        // CHECK:       UINT dyn_stat_en = 0,
        // CHECK:       UINT hwp_stall_dis = 0,
        // CHECK:     }
        // CHECK:     dyn_ppe_data = SINT 0,
        // CHECK:     odu_local_a {
        // CHECK:       UINT local_region_beg_x = 0,
        // CHECK:       UINT local_region_end_x = 0,
        // CHECK:     }
        // CHECK:     odu_local_b {
        // CHECK:       UINT local_region_beg_y = 0,
        // CHECK:       UINT local_region_end_y = 0,
        // CHECK:     }
        // CHECK:     halo_0_offset {
        // CHECK:       UINT halo_0_mm_adr_offset = 0,
        // CHECK:     }
        // CHECK:     halo_1_offset {
        // CHECK:       UINT halo_1_mm_adr_offset = 0,
        // CHECK:     }
        // CHECK:     halo_2_offset {
        // CHECK:       UINT halo_2_mm_adr_offset = 0,
        // CHECK:     }
        // CHECK:     halo_3_offset {
        // CHECK:       UINT halo_3_mm_adr_offset = 0,
        // CHECK:     }
        // CHECK:     halo_4_offset {
        // CHECK:       UINT halo_4_mm_adr_offset = 0,
        // CHECK:     }
        // CHECK:     unused3_inv = UINT 0,
        // CHECK:     barriers_wait_mask_hi_ {
        // CHECK:       UINT barriers_wait_mask_hi_ = 0,
        // CHECK:     }
        // CHECK:     barriers_wait_mask_lo_ = UINT 0,
        // CHECK:     barriers_post_mask_hi_ {
        // CHECK:       UINT barriers_post_mask_hi_ = 0,
        // CHECK:     }
        // CHECK:     barriers_post_mask_lo_ = UINT 0,
        // CHECK:     barriers_group_mask_ {
        // CHECK:       UINT group_ = 0,
        // CHECK:       UINT mask_ = 0,
        // CHECK:     }
        // CHECK:     barriers_sched_ {
        // CHECK:       UINT start_after_ = 0,
        // CHECK:       UINT clean_after_ = 0,
        // CHECK:     }
        // CHECK:    deprecated0_inv = UINT 0,
        // CHECK:    variant_count_ = UINT 0,
        // CHECK:    cluster_invariant_ = UINT 0,
        // CHECK:    deprecated1_inv = UINT 0,
        // CHECK:    pad_3 = UINT 0,
        // CHECK:   } requires 11:13:3
        // CHECK: >
      }
      ELF.CreateSection @text.variants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        NPUReg60XX.DPUVariant descriptor = <
          DpuVariantRegister {
            invar_ptr {
              UINT invar_ptr = 0,
              UINT var_tag = 1,
            }
            workload_size0 {
              UINT workload_size_x = 0,
              UINT workload_size_y = 0,
            }
            workload_size1 {
              UINT workload_size_z = 0,
              UINT pad_count_up = 0,
              UINT pad_count_left = 0,
              UINT pad_count_down = 0,
              UINT pad_count_right = 0,
              UINT halo_enable_new_desc = 0,
            }
            workload_start0 {
              UINT workload_start_x = 0,
              UINT workload_start_y = 0,
            }
            workload_start1 {
              UINT workload_start_z = 0,
              UINT new_invar_format = 1,
              UINT new_var_format = 1,
              UINT weight_d_ptr_start = 0,
            }
            offset_addr {
              UINT nthw_ntk = 0,
              UINT bin_cfg = 0,
              UINT conv_cond = 0,
              UINT dense_se = 0,
              UINT cim_clk_en = 0,
              UINT swizzle_key_offset = 0,
              UINT idu_mrm_clk_en = 0,
              UINT odu_clk_en = 0,
              UINT mpe_clk_en = 0,
              UINT ppe_clk_en = 0,
              UINT odu_stat_en = 0,
              UINT idu_stat_en = 0,
              UINT noc_clk_en = 1,
              UINT odu_stat_clr_mode = 0,
              UINT idu_stat_clr_mode = 0,
              UINT se_only_en = 0,
              UINT shave_l2_cache_en = 0,
              UINT idu_dbg_en = 0,
              UINT sb_read_en = 0,
              UINT tensor2_act_dense = 0,
              UINT grp_conv_en = 0,
              UINT zero_point_4b = 0,
              UINT reserved_3 = 0,
              UINT wt_swizzle_key = 0,
              UINT wt_swizzle_sel = 1,
              UINT gif_clk_en = 0,
            }
            hwp_wload_id {
              UINT hwp_wload_id = 0,
            }
            var_cfg {
              UINT invar_plt_rd_en = 0,
              UINT plt_ptr_force = 0,
              UINT invar_lut_rd_en = 0,
              UINT invar_line_cnt_en = 0,
              UINT invar_line_cnt = 0,
              UINT invar_lptr_force = 1,
              UINT next_sram_job_valid = 0,
              UINT next_sram_job_addr = 0,
            }
            cbarrier_lo = UINT 0,
            pbarrier_lo = UINT 0,
            dpu_cfg {
              UINT workload_start_odu = 1,
              UINT workload_start_idu = 1,
              UINT workload_prm_sel = 0,
              UINT workload_valid = 0,
              UINT workload_shad_odu = 0,
              UINT workload_shad_idu = 0,
              UINT workload_idu_auto_upd_0 = 1,
              UINT workload_idu_auto_upd_1 = 0,
              UINT workload_odu_auto_upd = 0,
              UINT cfg_Reserved_2 = 0,
              UINT preempt_cc_en = 0,
              UINT bar_wait_preempt_en = 1,
              UINT rst_ctxt_new = 0,
              UINT cfg_Reserved_3 = 0,
              UINT cfg_Reserved_4 = 0,
              UINT odu_stat_clr = 0,
              UINT idu_stat_clr = 0,
              UINT cfg_Reserved_5 = 0,
              UINT cfg_Reserved_6 = 0,
            }
            te_beg0 {
              UINT te_beg_y = 0x20,
              UINT te_beg_z = 0x40,
            }
            te_beg1 {
              UINT te_beg_x = 1,
              UINT dyn_primitive_id_var = 0,
              UINT dyn_primitive_id_var_en = 0,
              UINT wt_size_max_en = 0,
            }
            te_end0 {
              UINT te_end_y = 0x3F,
              UINT te_end_z = 0xF,
            }
            te_end1 {
              UINT te_end_x = 0x3F,
            }
            weight_size {
              UINT weight_size = 0,
              UINT scope_id_en = 0,
              UINT dyn_primitive_id2_var = 0,
            }
            weight_num {
              UINT weight_num = 0,
              UINT weight_sp_ptr_start = 0,
            }
            weight_start = UINT 0,
            cbarrier_hi {
              UINT cbarrier_hi = 0,
            }
            pbarrier_hi {
              UINT pbarrier_hi = 0,
            }
            halo_region0A {
              SINT sp_adr_offset = 0,
              UINT tile_select = 0,
              UINT rsvdA = 0,
              UINT enable = 0,
            }
            halo_region0B {
              SINT ac_adr_offset = 0,
              UINT target_width_lsb = 0,
            }
            halo_region0C {
              UINT begin_x = 0,
              UINT begin_y = 0,
              UINT target_width_msb = 0,
            }
            halo_region0D {
              UINT end_x = 0,
              UINT end_y = 0,
              UINT rsvdD = 0,
            }
            halo_region1A {
              SINT sp_adr_offset = 0,
              UINT tile_select = 0,
              UINT rsvdA = 0,
              UINT enable = 0,
            }
            halo_region1B {
              SINT ac_adr_offset = 0,
              UINT target_width_lsb = 0,
            }
            halo_region1C {
              UINT begin_x = 0,
              UINT begin_y = 0,
              UINT target_width_msb = 0,
            }
            halo_region1D {
              UINT end_x = 0,
              UINT end_y = 0,
              UINT rsvdD = 0,
            }
            halo_region2A {
              SINT sp_adr_offset = 0,
              UINT tile_select = 0,
              UINT rsvdA = 0,
              UINT enable = 0,
            }
            halo_region2B {
              SINT ac_adr_offset = 0,
              UINT target_width_lsb = 0,
            }
            halo_region2C {
              UINT begin_x = 0,
              UINT begin_y = 0,
              UINT target_width_msb = 0,
            }
            halo_region2D {
              UINT end_x = 0,
              UINT end_y = 0,
              UINT rsvdD = 0,
            }
            halo_region3A {
              SINT sp_adr_offset = 0,
              UINT tile_select = 0,
              UINT rsvdA = 0,
              UINT enable = 0,
            }
            halo_region3B {
              SINT ac_adr_offset = 0,
              UINT target_width_lsb = 0,
            }
            halo_region3C {
              UINT begin_x = 0,
              UINT begin_y = 0,
              UINT target_width_msb = 0,
            }
            halo_region3D {
              UINT end_x = 0,
              UINT end_y = 0,
              UINT rsvdD = 0,
            }
            halo_region4A {
              SINT sp_adr_offset = 0,
              UINT tile_select = 0,
              UINT rsvdA = 0,
              UINT enable = 0,
            }
            halo_region4B {
              SINT ac_adr_offset = 0,
              UINT target_width_lsb = 0,
            }
            halo_region4C {
              UINT begin_x = 0,
              UINT begin_y = 0,
              UINT target_width_msb = 0,
            }
            halo_region4D {
              UINT end_x = 0,
              UINT end_y = 0,
              UINT rsvdD = 0,
            }
            halo_region5A {
              SINT sp_adr_offset = 0,
              UINT tile_select = 0,
              UINT rsvdA = 0,
              UINT enable = 0,
            }
            halo_region5B {
              SINT ac_adr_offset = 0,
              UINT target_width_lsb = 0,
            }
            halo_region5C {
              UINT begin_x = 0,
              UINT begin_y = 0,
              UINT target_width_msb = 0,
            }
            halo_region5D {
              UINT end_x = 0,
              UINT end_y = 0,
              UINT rsvdD = 0,
            }
            unused0_var = UINT 0,
            unused1_var = UINT 0,
            unused2_var = UINT 0,
            unused3_var = UINT 0,
            invariant_ = UINT 0,
            invariant_index_ = UINT 0,
            deprecated0_var = UINT 0,
            deprecated1_var = UINT 0,
            pad_7_1 = UINT 0,
          } requires 11:13:2
        > {invariant_task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0,
        nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, sym_name = "DPUVariant_0", task_index = !VPURegMapped.Index<0:0:0>, task_location = @builtin.tasks.DPUVariant0::@DeclareTaskBuffer_DPUVariant_0}
        // CHECK: NPUReg60XX.DPUVariant descriptor = <
        // CHECK:   DpuVariantRegister {
        // CHECK:     invar_ptr {
        // CHECK:       UINT invar_ptr = 0,
        // CHECK:       UINT var_tag = 1,
        // CHECK:     }
        // CHECK:     workload_size0 {
        // CHECK:       UINT workload_size_x = 0,
        // CHECK:       UINT workload_size_y = 0,
        // CHECK:     }
        // CHECK:     workload_size1 {
        // CHECK:       UINT workload_size_z = 0,
        // CHECK:       UINT pad_count_up = 0,
        // CHECK:       UINT pad_count_left = 0,
        // CHECK:       UINT pad_count_down = 0,
        // CHECK:       UINT pad_count_right = 0,
        // CHECK:       UINT halo_enable_new_desc = 0,
        // CHECK:     }
        // CHECK:     workload_start0 {
        // CHECK:       UINT workload_start_x = 0,
        // CHECK:       UINT workload_start_y = 0,
        // CHECK:     }
        // CHECK:     workload_start1 {
        // CHECK:       UINT workload_start_z = 0,
        // CHECK:       UINT new_invar_format = 1,
        // CHECK:       UINT new_var_format = 1,
        // CHECK:       UINT weight_d_ptr_start = 0,
        // CHECK:     }
        // CHECK:     offset_addr {
        // CHECK:       UINT nthw_ntk = 0,
        // CHECK:       UINT bin_cfg = 0,
        // CHECK:       UINT conv_cond = 0,
        // CHECK:       UINT dense_se = 0,
        // CHECK:       UINT cim_clk_en = 0,
        // CHECK:       UINT swizzle_key_offset = 0,
        // CHECK:       UINT idu_mrm_clk_en = 0,
        // CHECK:       UINT odu_clk_en = 0,
        // CHECK:       UINT mpe_clk_en = 0,
        // CHECK:       UINT ppe_clk_en = 0,
        // CHECK:       UINT odu_stat_en = 0,
        // CHECK:       UINT idu_stat_en = 0,
        // CHECK:       UINT noc_clk_en = 1,
        // CHECK:       UINT odu_stat_clr_mode = 0,
        // CHECK:       UINT idu_stat_clr_mode = 0,
        // CHECK:       UINT se_only_en = 0,
        // CHECK:       UINT shave_l2_cache_en = 0,
        // CHECK:       UINT idu_dbg_en = 0,
        // CHECK:       UINT sb_read_en = 0,
        // CHECK:       UINT tensor2_act_dense = 0,
        // CHECK:       UINT grp_conv_en = 0,
        // CHECK:       UINT zero_point_4b = 0,
        // CHECK:       UINT reserved_3 = 0,
        // CHECK:       UINT wt_swizzle_key = 0,
        // CHECK:       UINT wt_swizzle_sel = 1,
        // CHECK:       UINT gif_clk_en = 0,
        // CHECK:     }
        // CHECK:     hwp_wload_id {
        // CHECK:       UINT hwp_wload_id = 0,
        // CHECK:     }
        // CHECK:     var_cfg {
        // CHECK:       UINT invar_plt_rd_en = 0,
        // CHECK:       UINT plt_ptr_force = 0,
        // CHECK:       UINT invar_lut_rd_en = 0,
        // CHECK:       UINT invar_line_cnt_en = 0,
        // CHECK:       UINT invar_line_cnt = 0,
        // CHECK:       UINT invar_lptr_force = 1,
        // CHECK:       UINT next_sram_job_valid = 0,
        // CHECK:       UINT next_sram_job_addr = 0,
        // CHECK:     }
        // CHECK:     cbarrier_lo = UINT 0,
        // CHECK:     pbarrier_lo = UINT 0,
        // CHECK:     dpu_cfg {
        // CHECK:       UINT workload_start_odu = 1,
        // CHECK:       UINT workload_start_idu = 1,
        // CHECK:       UINT workload_prm_sel = 0,
        // CHECK:       UINT workload_valid = 0,
        // CHECK:       UINT workload_shad_odu = 0,
        // CHECK:       UINT workload_shad_idu = 0,
        // CHECK:       UINT workload_idu_auto_upd_0 = 1,
        // CHECK:       UINT workload_idu_auto_upd_1 = 0,
        // CHECK:       UINT workload_odu_auto_upd = 0,
        // CHECK:       UINT cfg_Reserved_2 = 0,
        // CHECK:       UINT preempt_cc_en = 0,
        // CHECK:       UINT bar_wait_preempt_en = 1,
        // CHECK:       UINT rst_ctxt_new = 0,
        // CHECK:       UINT cfg_Reserved_3 = 0,
        // CHECK:       UINT cfg_Reserved_4 = 0,
        // CHECK:       UINT odu_stat_clr = 0,
        // CHECK:       UINT idu_stat_clr = 0,
        // CHECK:       UINT cfg_Reserved_5 = 0,
        // CHECK:       UINT cfg_Reserved_6 = 0,
        // CHECK:     }
        // CHECK:     te_beg0 {
        // CHECK:       UINT te_beg_y = 0x20,
        // CHECK:       UINT te_beg_z = 0x40,
        // CHECK:     }
        // CHECK:     te_beg1 {
        // CHECK:       UINT te_beg_x = 1,
        // CHECK:       UINT dyn_primitive_id_var = 0,
        // CHECK:       UINT dyn_primitive_id_var_en = 0,
        // CHECK:       UINT wt_size_max_en = 0,
        // CHECK:     }
        // CHECK:     te_end0 {
        // CHECK:       UINT te_end_y = 0x3F,
        // CHECK:       UINT te_end_z = 0xF,
        // CHECK:     }
        // CHECK:     te_end1 {
        // CHECK:       UINT te_end_x = 0x3F,
        // CHECK:     }
        // CHECK:     weight_size {
        // CHECK:       UINT weight_size = 0,
        // CHECK:       UINT scope_id_en = 0,
        // CHECK:       UINT dyn_primitive_id2_var = 0,
        // CHECK:     }
        // CHECK:     weight_num {
        // CHECK:       UINT weight_num = 0,
        // CHECK:       UINT weight_sp_ptr_start = 0,
        // CHECK:     }
        // CHECK:     weight_start = UINT 0,
        // CHECK:     cbarrier_hi {
        // CHECK:       UINT cbarrier_hi = 0,
        // CHECK:     }
        // CHECK:     pbarrier_hi {
        // CHECK:       UINT pbarrier_hi = 0,
        // CHECK:     }
        // CHECK:     halo_region0A {
        // CHECK:       SINT sp_adr_offset = 0,
        // CHECK:       UINT tile_select = 0,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 0,
        // CHECK:     }
        // CHECK:     halo_region0B {
        // CHECK:       SINT ac_adr_offset = 0,
        // CHECK:       UINT target_width_lsb = 0,
        // CHECK:     }
        // CHECK:     halo_region0C {
        // CHECK:       UINT begin_x = 0,
        // CHECK:       UINT begin_y = 0,
        // CHECK:       UINT target_width_msb = 0,
        // CHECK:     }
        // CHECK:     halo_region0D {
        // CHECK:       UINT end_x = 0,
        // CHECK:       UINT end_y = 0,
        // CHECK:       UINT rsvdD = 0,
        // CHECK:     }
        // CHECK:     halo_region1A {
        // CHECK:       SINT sp_adr_offset = 0,
        // CHECK:       UINT tile_select = 0,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 0,
        // CHECK:     }
        // CHECK:     halo_region1B {
        // CHECK:       SINT ac_adr_offset = 0,
        // CHECK:       UINT target_width_lsb = 0,
        // CHECK:     }
        // CHECK:     halo_region1C {
        // CHECK:       UINT begin_x = 0,
        // CHECK:       UINT begin_y = 0,
        // CHECK:       UINT target_width_msb = 0,
        // CHECK:     }
        // CHECK:     halo_region1D {
        // CHECK:       UINT end_x = 0,
        // CHECK:       UINT end_y = 0,
        // CHECK:       UINT rsvdD = 0,
        // CHECK:     }
        // CHECK:     halo_region2A {
        // CHECK:       SINT sp_adr_offset = 0,
        // CHECK:       UINT tile_select = 0,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 0,
        // CHECK:     }
        // CHECK:     halo_region2B {
        // CHECK:       SINT ac_adr_offset = 0,
        // CHECK:       UINT target_width_lsb = 0,
        // CHECK:     }
        // CHECK:     halo_region2C {
        // CHECK:       UINT begin_x = 0,
        // CHECK:       UINT begin_y = 0,
        // CHECK:       UINT target_width_msb = 0,
        // CHECK:     }
        // CHECK:     halo_region2D {
        // CHECK:       UINT end_x = 0,
        // CHECK:       UINT end_y = 0,
        // CHECK:       UINT rsvdD = 0,
        // CHECK:     }
        // CHECK:     halo_region3A {
        // CHECK:       SINT sp_adr_offset = 0,
        // CHECK:       UINT tile_select = 0,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 0,
        // CHECK:     }
        // CHECK:     halo_region3B {
        // CHECK:       SINT ac_adr_offset = 0,
        // CHECK:       UINT target_width_lsb = 0,
        // CHECK:     }
        // CHECK:     halo_region3C {
        // CHECK:       UINT begin_x = 0,
        // CHECK:       UINT begin_y = 0,
        // CHECK:       UINT target_width_msb = 0,
        // CHECK:     }
        // CHECK:     halo_region3D {
        // CHECK:       UINT end_x = 0,
        // CHECK:       UINT end_y = 0,
        // CHECK:       UINT rsvdD = 0,
        // CHECK:     }
        // CHECK:     halo_region4A {
        // CHECK:       SINT sp_adr_offset = 0,
        // CHECK:       UINT tile_select = 0,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 0,
        // CHECK:     }
        // CHECK:     halo_region4B {
        // CHECK:       SINT ac_adr_offset = 0,
        // CHECK:       UINT target_width_lsb = 0,
        // CHECK:     }
        // CHECK:     halo_region4C {
        // CHECK:       UINT begin_x = 0,
        // CHECK:       UINT begin_y = 0,
        // CHECK:       UINT target_width_msb = 0,
        // CHECK:     }
        // CHECK:     halo_region4D {
        // CHECK:       UINT end_x = 0,
        // CHECK:       UINT end_y = 0,
        // CHECK:       UINT rsvdD = 0,
        // CHECK:     }
        // CHECK:     halo_region5A {
        // CHECK:       SINT sp_adr_offset = 0,
        // CHECK:       UINT tile_select = 0,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 0,
        // CHECK:     }
        // CHECK:     halo_region5B {
        // CHECK:       SINT ac_adr_offset = 0,
        // CHECK:       UINT target_width_lsb = 0,
        // CHECK:     }
        // CHECK:     halo_region5C {
        // CHECK:       UINT begin_x = 0,
        // CHECK:       UINT begin_y = 0,
        // CHECK:       UINT target_width_msb = 0,
        // CHECK:     }
        // CHECK:     halo_region5D {
        // CHECK:       UINT end_x = 0,
        // CHECK:       UINT end_y = 0,
        // CHECK:       UINT rsvdD = 0,
        // CHECK:     }
        // CHECK:     unused0_var = UINT 0,
        // CHECK:     unused1_var = UINT 0,
        // CHECK:     unused2_var = UINT 0,
        // CHECK:     unused3_var = UINT 0,
        // CHECK:     invariant_ = UINT 0,
        // CHECK:     invariant_index_ = UINT 0,
        // CHECK:     deprecated0_var = UINT 0,
        // CHECK:     deprecated1_var = UINT 0,
        // CHECK:     pad_7_1 = UINT 0,
        // CHECK:   } requires 11:13:3
        // CHECK: >
      }
    }
    return
  }
}

// -----

module @DPUPVPTest {
  func.func @main() {
    ELF.Main {
      ELF.CreateSection @dpu.pvp aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        NPUReg60XX.DpuPVP descriptor = <
          VpuDpuPVP {
            VpuDpuPVP_opFPCount = UINT 0x100,
            VpuDpuPVP_totalOpCount = UINT 0x100,
            VpuDpuPVP_reserved_0 = UINT 0,
            VpuDpuPVP_reserved_1 = UINT 0,
          } requires 11:13:2
        > {sym_name = "DpuPVP_0_0_0"}
        //CHECK: NPUReg60XX.DpuPVP descriptor = <
        //CHECK:   VpuDpuPVP {
        //CHECK:     VpuDpuPVP_opFPCount = UINT 0x100,
        //CHECK:     VpuDpuPVP_totalOpCount = UINT 0x100,
        //CHECK:     VpuDpuPVP_reserved_0 = UINT 0,
        //CHECK:     VpuDpuPVP_reserved_1 = UINT 0,
        //CHECK:   } requires 11:13:3
        //CHECK: > {sym_name = "DpuPVP_0_0_0"}
      }
    }
    return
  }
}
