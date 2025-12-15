//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// RUN: vpux-opt --emit-bytecode --init-compiler="vpu-arch=%arch%" %s | vpux-opt --vpu-arch=%arch% | FileCheck %s
// REQUIRES: dev-build && arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
    DataInfo "input_1" : tensor<16x1x1x1xi64>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }
  func.func @main() {
    ELF.Main @ELFMain {
      ELF.CreateLogicalSection @builtin.data.nncmx0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <196736> : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @Weights !VPUASM.Buffer< "CMX_NN"[0] <213632> : memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, #NHWC, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateLogicalSection @builtin.tasks.DPUVariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @builtin.tasks.DPUInvariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
      }
      ELF.CreateSection @text.invariants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        NPUReg50XX.DPUInvariant descriptor = <
          DpuInvariantRegister {
            cmx_slice0_low_addr = UINT 0x4000000,
            cmx_slice1_low_addr = UINT 0x4000000,
            cmx_slice2_low_addr = UINT 0x4000000,
            cmx_slice3_low_addr = UINT 0x4000000,
            cmx_slice_size = UINT 0x18000,
            se_addr = UINT 0,
            sparsity_addr = UINT 0,
            se_size = UINT 0,
            z_config {
              UINT se_z_split = 0,
              UINT num_ses_in_z_dir = 0,
              UINT cm_sp_pattern = 0,
              UINT npo2_se_z_split_en = 0,
              UINT reserved = 0,
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
              UINT sp_se_tbl_segment = 0,
              UINT rst_ctxt = 1,
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
              UINT pool_wt_rd_dis = 0,
              UINT elop_operation = 0,
            }
            act_offset0 = UINT 0,
            act_offset1 = UINT 0,
            act_offset2 = UINT 0,
            act_offset3 = UINT 0,
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
            }
            wt_offset = UINT 0,
            odu_cfg {
              UINT dtype = 4,
              UINT wcb_ac_mode = 0,
              UINT wcb_sp_mode = 0,
              UINT sp_value = 0,
              UINT sp_out_en = 1,
              UINT cmx_port_muxing_disable = 0,
              UINT write_sp = 1,
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
            odu_be_size = UINT 0,
            odu_be_cnt = UINT 0,
            odu_se_size = UINT 0,
            te_dim0 {
              UINT te_dim_y = 0x3F,
              UINT te_dim_z = 0xF,
            }
            te_dim1 {
              UINT te_dim_x = 0x3F,
            }
            pt_base = UINT 0,
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
            ppe_bias = SINT 0,
            ppe_scale {
              UINT ppe_scale_shift = 0,
              UINT ppe_scale_round = 3,
              SINT ppe_scale_mult = 0,
            }
            ppe_scale_ctrl {
              UINT ppe_scale_override = 0,
              UINT ppe_fp_scale_override = 0,
            }
            ppe_prelu {
              UINT ppe_prelu_shift = 0,
              UINT ppe_prelu_mult = 1,
            }
            ppe_scale_hclamp {
              SINT ppe_int_scale_hclamp = 0x7F7FFFFF,
              FP ppe_fp_scale_hclamp = 0x7F7FFFFF,
            }
            ppe_scale_lclamp {
              SINT ppe_int_scale_lclamp = 0xFF7FFFFF,
              FP ppe_fp_scale_lclamp = 0xFF7FFFFF,
            }
            ppe_misc {
              UINT ppe_mode = 2,
              UINT ppe_rnd_fp = 0,
              UINT ppe_rnd_int = 0,
              UINT ppe_fp16_ftz = 0,
              UINT ppe_fp16_clamp = 0,
              UINT ppe_i32_convert = 0,
              UINT ppe_sb_dtype = 2,
              UINT ppe_mult2_mode = 0,
            }
            ppe_fp_bias = FP 0,
            ppe_fp_scale = FP 0,
            ppe_fp_prelu = FP 0,
            ppe_fp_cfg {
              UINT ppe_fp_convert = 0,
              UINT ppe_fp_bypass = 1,
              UINT ppe_bf16_round = 0,
              UINT ppe_fp_prelu_en = 0,
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
            odu_cast0 {
              UINT cast_enable0 = 0,
              UINT cast_offset0 = 0,
            }
            odu_cast1 {
              UINT cast_enable1 = 0,
              UINT cast_offset1 = 0,
            }
            odu_cast2 {
              UINT cast_enable2 = 0,
              UINT cast_offset2 = 0,
            }
            tensor2_start = UINT 0,
            ppe_lut_ptr {
              UINT ppe_lut_ptr = 0,
              UINT ppe_lut_ptr_force = 1,
            }
            nvar_tag = UINT 1,
            pallet0 {
              UINT plt_idx_0 = 0,
              UINT plt_idx_1 = 0,
            }
            pallet1 {
              UINT plt_idx_2 = 0,
              UINT plt_idx_3 = 0,
            }
            pallet2 {
              UINT plt_idx_4 = 0,
              UINT plt_idx_5 = 0,
            }
            pallet3 {
              UINT plt_idx_6 = 0,
              UINT plt_idx_7 = 0,
            }
            pallet4 {
              UINT plt_idx_8 = 0,
              UINT plt_idx_9 = 0,
            }
            pallet5 {
              UINT plt_idx_10 = 0,
              UINT plt_idx_11 = 0,
            }
            pallet6 {
              UINT plt_idx_12 = 0,
              UINT plt_idx_13 = 0,
            }
            pallet7 {
              UINT plt_idx_14 = 0,
              UINT plt_idx_15 = 0,
            }
            se_addr1 = UINT 0,
            sparsity_addr1 = UINT 0,
            se_addr2 = UINT 0,
            sparsity_addr2 = UINT 0,
            se_addr3 = UINT 0,
            sparsity_addr3 = UINT 0,
            se_sp_size1 = UINT 0,
            se_sp_size2 = UINT 0,
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
          }
        > {input = @builtin.data.nncmx0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, output = @builtin.data.nncmx0::@DeclareBuffer_ActOut,
         sym_name = "DPUInvariant_0", task_index = !VPURegMapped.Index<0:0:0>, task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0}
        // CHECK: NPUReg50XX.DPUInvariant descriptor = <
        // CHECK:   DpuInvariantRegister {
        // CHECK:     cmx_slice0_low_addr = UINT 0x4000000,
        // CHECK:     cmx_slice1_low_addr = UINT 0x4000000,
        // CHECK:     cmx_slice2_low_addr = UINT 0x4000000,
        // CHECK:     cmx_slice3_low_addr = UINT 0x4000000,
        // CHECK:     cmx_slice_size = UINT 0x18000,
        // CHECK:     se_addr = UINT 0,
        // CHECK:     sparsity_addr = UINT 0,
        // CHECK:     se_size = UINT 0,
        // CHECK:     z_config {
        // CHECK:       UINT se_z_split = 0,
        // CHECK:       UINT num_ses_in_z_dir = 0,
        // CHECK:       UINT cm_sp_pattern = 0,
        // CHECK:       UINT npo2_se_z_split_en = 0,
        // CHECK:       UINT reserved = 0,
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
        // CHECK:       UINT sp_se_tbl_segment = 0,
        // CHECK:       UINT rst_ctxt = 1,
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
        // CHECK:       UINT pool_wt_rd_dis = 0,
        // CHECK:       UINT elop_operation = 0,
        // CHECK:     }
        // CHECK:     act_offset0 = UINT 0,
        // CHECK:     act_offset1 = UINT 0,
        // CHECK:     act_offset2 = UINT 0,
        // CHECK:     act_offset3 = UINT 0,
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
        // CHECK:     }
        // CHECK:     wt_offset = UINT 0,
        // CHECK:     odu_cfg {
        // CHECK:       UINT dtype = 4,
        // CHECK:       UINT wcb_ac_mode = 0,
        // CHECK:       UINT wcb_sp_mode = 0,
        // CHECK:       UINT sp_value = 0,
        // CHECK:       UINT sp_out_en = 1,
        // CHECK:       UINT cmx_port_muxing_disable = 0,
        // CHECK:       UINT write_sp = 1,
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
        // CHECK:     odu_be_size = UINT 0,
        // CHECK:     odu_be_cnt = UINT 0,
        // CHECK:     odu_se_size = UINT 0,
        // CHECK:     te_dim0 {
        // CHECK:       UINT te_dim_y = 0x3F,
        // CHECK:       UINT te_dim_z = 0xF,
        // CHECK:     }
        // CHECK:     te_dim1 {
        // CHECK:       UINT te_dim_x = 0x3F,
        // CHECK:     }
        // CHECK:     pt_base = UINT 0,
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
        // CHECK:     ppe_bias = SINT 0,
        // CHECK:     ppe_scale {
        // CHECK:       UINT ppe_scale_shift = 0,
        // CHECK:       UINT ppe_scale_round = 3,
        // CHECK:       SINT ppe_scale_mult = 0,
        // CHECK:     }
        // CHECK:     ppe_scale_ctrl {
        // CHECK:       UINT ppe_scale_override = 0,
        // CHECK:       UINT ppe_fp_scale_override = 0,
        // CHECK:     }
        // CHECK:     ppe_prelu {
        // CHECK:       UINT ppe_prelu_shift = 0,
        // CHECK:       UINT ppe_prelu_mult = 1,
        // CHECK:     }
        // CHECK:     ppe_scale_hclamp {
        // CHECK:       SINT ppe_int_scale_hclamp = 0x7F7FFFFF,
        // CHECK:       FP ppe_fp_scale_hclamp = 0x7F7FFFFF,
        // CHECK:     }
        // CHECK:     ppe_scale_lclamp {
        // CHECK:       SINT ppe_int_scale_lclamp = 0xFF7FFFFF,
        // CHECK:       FP ppe_fp_scale_lclamp = 0xFF7FFFFF,
        // CHECK:     }
        // CHECK:     ppe_misc {
        // CHECK:       UINT ppe_mode = 2,
        // CHECK:       UINT ppe_rnd_fp = 0,
        // CHECK:       UINT ppe_rnd_int = 0,
        // CHECK:       UINT ppe_fp16_ftz = 0,
        // CHECK:       UINT ppe_fp16_clamp = 0,
        // CHECK:       UINT ppe_i32_convert = 0,
        // CHECK:       UINT ppe_sb_dtype = 2,
        // CHECK:       UINT ppe_mult2_mode = 0,
        // CHECK:     }
        // CHECK:     ppe_fp_bias = FP 0,
        // CHECK:     ppe_fp_scale = FP 0,
        // CHECK:     ppe_fp_prelu = FP 0,
        // CHECK:     ppe_fp_cfg {
        // CHECK:       UINT ppe_fp_convert = 0,
        // CHECK:       UINT ppe_fp_bypass = 1,
        // CHECK:       UINT ppe_bf16_round = 0,
        // CHECK:       UINT ppe_fp_prelu_en = 0,
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
        // CHECK:     odu_cast0 {
        // CHECK:       UINT cast_enable0 = 0,
        // CHECK:       UINT cast_offset0 = 0,
        // CHECK:     }
        // CHECK:     odu_cast1 {
        // CHECK:       UINT cast_enable1 = 0,
        // CHECK:       UINT cast_offset1 = 0,
        // CHECK:     }
        // CHECK:     odu_cast2 {
        // CHECK:       UINT cast_enable2 = 0,
        // CHECK:       UINT cast_offset2 = 0,
        // CHECK:     }
        // CHECK:     tensor2_start = UINT 0,
        // CHECK:     ppe_lut_ptr {
        // CHECK:       UINT ppe_lut_ptr = 0,
        // CHECK:       UINT ppe_lut_ptr_force = 1,
        // CHECK:     }
        // CHECK:     nvar_tag = UINT 1,
        // CHECK:     pallet0 {
        // CHECK:       UINT plt_idx_0 = 0,
        // CHECK:       UINT plt_idx_1 = 0,
        // CHECK:     }
        // CHECK:     pallet1 {
        // CHECK:       UINT plt_idx_2 = 0,
        // CHECK:       UINT plt_idx_3 = 0,
        // CHECK:     }
        // CHECK:     pallet2 {
        // CHECK:       UINT plt_idx_4 = 0,
        // CHECK:       UINT plt_idx_5 = 0,
        // CHECK:     }
        // CHECK:     pallet3 {
        // CHECK:       UINT plt_idx_6 = 0,
        // CHECK:       UINT plt_idx_7 = 0,
        // CHECK:     }
        // CHECK:     pallet4 {
        // CHECK:       UINT plt_idx_8 = 0,
        // CHECK:       UINT plt_idx_9 = 0,
        // CHECK:     }
        // CHECK:     pallet5 {
        // CHECK:       UINT plt_idx_10 = 0,
        // CHECK:       UINT plt_idx_11 = 0,
        // CHECK:     }
        // CHECK:     pallet6 {
        // CHECK:       UINT plt_idx_12 = 0,
        // CHECK:       UINT plt_idx_13 = 0,
        // CHECK:     }
        // CHECK:     pallet7 {
        // CHECK:       UINT plt_idx_14 = 0,
        // CHECK:       UINT plt_idx_15 = 0,
        // CHECK:     }
        // CHECK:     se_addr1 = UINT 0,
        // CHECK:     sparsity_addr1 = UINT 0,
        // CHECK:     se_addr2 = UINT 0,
        // CHECK:     sparsity_addr2 = UINT 0,
        // CHECK:     se_addr3 = UINT 0,
        // CHECK:     sparsity_addr3 = UINT 0,
        // CHECK:     se_sp_size1 = UINT 0,
        // CHECK:     se_sp_size2 = UINT 0,
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
        // CHECK:   }
        // CHECK: >
      }
      ELF.CreateSection @text.variants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        NPUReg50XX.DPUVariant descriptor = <
          DpuVariantRegister {
            invar_ptr {
              UINT invar_ptr = 0,
              UINT var_tag = 1,
            }
            workload_size0 {
              UINT workload_size_x = 0x20,
              UINT workload_size_y = 0x21,
            }
            workload_size1 {
              UINT workload_size_z = 0x22,
              UINT pad_count_up = 3,
              UINT pad_count_left = 4,
              UINT pad_count_down = 5,
              UINT pad_count_right = 6,
            }
            workload_start0 {
              UINT workload_start_x = 3,
              UINT workload_start_y = 4,
            }
            workload_start1 {
              UINT workload_start_z = 5,
            }
            offset_addr {
              UINT nthw_ntk = 0,
              UINT bin_cfg = 1,
              UINT conv_cond = 1,
              UINT dense_se = 1,
              UINT idx_quad = 0,
              UINT swizzle_key_offset = 1,
              UINT idu_mrm_clk_en = 0,
              UINT odu_clk_en = 0,
              UINT mpe_clk_en = 0,
              UINT ppe_clk_en = 0,
              UINT odu_stat_en = 0,
              UINT idu_stat_en = 0,
              UINT noc_clk_en = 1,
              UINT odu_stat_clr_mode = 0,
              UINT idu_stat_clr_mode = 0,
              UINT se_only_en = 1,
              UINT shave_l2_cache_en = 0,
              UINT idu_dbg_en = 0,
              UINT sb_read_en = 1,
              UINT tensor2_act_dense = 1,
              UINT wt_swizzle_key = 2,
              UINT wt_swizzle_sel = 1,
              UINT gif_clk_en = 0,
            }
            hwp_wload_id {
              UINT hwp_wload_id = 0,
            }
            var_cfg {
              UINT invar_lut_rd_en = 1,
              UINT invar_line_cnt_en = 0,
              UINT invar_line_cnt = 0,
              UINT invar_lptr_force = 1,
              UINT next_sram_job_valid = 0,
              UINT next_sram_job_addr = 0,
            }
            cbarrier_lo = UINT 6,
            cbarrier_hi {
              UINT cbarrier_hi = 0,
            }
            pbarrier_lo = UINT 0,
            pbarrier_hi {
              UINT pbarrier_hi = 0,
            }
            halo_region0A {
              SINT sp_adr_offset = 0xA,
              UINT tile_select = 6,
              UINT rsvdA = 0,
              UINT enable = 1,
            }
            halo_region0B {
              SINT ac_adr_offset = 0xA,
              UINT target_width_lsb = 0x20,
            }
            halo_region0C {
              UINT begin_x = 0,
              UINT begin_y = 0,
              UINT target_width_msb = 0,
            }
            halo_region0D {
              UINT end_x = 0xF,
              UINT end_y = 0xF,
              UINT rsvdD = 0,
            }
            halo_region1A {
              SINT sp_adr_offset = 0,
              UINT tile_select = 0xA,
              UINT rsvdA = 0,
              UINT enable = 1,
            }
            halo_region1B {
              SINT ac_adr_offset = 9,
              UINT target_width_lsb = 0x40,
            }
            halo_region1C {
              UINT begin_x = 4,
              UINT begin_y = 5,
              UINT target_width_msb = 0,
            }
            halo_region1D {
              UINT end_x = 0xF,
              UINT end_y = 0x10,
              UINT rsvdD = 0,
            }
            halo_region2A {
              SINT sp_adr_offset = 0,
              UINT tile_select = 0x3F,
              UINT rsvdA = 0,
              UINT enable = 1,
            }
            halo_region2B {
              SINT ac_adr_offset = 0x190,
              UINT target_width_lsb = 0x40,
            }
            halo_region2C {
              UINT begin_x = 9,
              UINT begin_y = 0,
              UINT target_width_msb = 0xF,
            }
            halo_region2D {
              UINT end_x = 0x3F,
              UINT end_y = 0x3F,
              UINT rsvdD = 0,
            }
            halo_region3A {
              SINT sp_adr_offset = 0,
              UINT tile_select = 0x21,
              UINT rsvdA = 0,
              UINT enable = 1,
            }
            halo_region3B {
              SINT ac_adr_offset = 9,
              UINT target_width_lsb = 0x80,
            }
            halo_region3C {
              UINT begin_x = 9,
              UINT begin_y = 0,
              UINT target_width_msb = 0xF,
            }
            halo_region3D {
              UINT end_x = 0x3F,
              UINT end_y = 0x3F,
              UINT rsvdD = 0,
            }
            halo_region4A {
              SINT sp_adr_offset = 0,
              UINT tile_select = 4,
              UINT rsvdA = 0,
              UINT enable = 1,
            }
            halo_region4B {
              SINT ac_adr_offset = 9,
              UINT target_width_lsb = 0x100,
            }
            halo_region4C {
              UINT begin_x = 9,
              UINT begin_y = 0,
              UINT target_width_msb = 3,
            }
            halo_region4D {
              UINT end_x = 0x3F,
              UINT end_y = 0x3F,
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
              UINT cfg_Reserved_0 = 0,
              UINT cfg_Reserved_1 = 0,
              UINT cfg_Reserved_2 = 0,
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
            }
            te_end0 {
              UINT te_end_y = 0x3F,
              UINT te_end_z = 0xF,
            }
            te_end1 {
              UINT te_end_x = 0x3F,
            }
            weight_size = UINT 0,
            weight_num = UINT 0,
            weight_start = UINT 0,
            invariant_ = UINT 0x200000,
            invariant_index_ = UINT 0,
            deprecated0_var = UINT 0,
            deprecated1_var = UINT 0,
            pad_7_1 = UINT 0,
          }
        > {invariant_task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, sym_name = "DPUVariant11", task_index = !VPURegMapped.Index<0:0:0>}
        // CHECK: NPUReg50XX.DPUVariant descriptor = <
        // CHECK:   DpuVariantRegister {
        // CHECK:     invar_ptr {
        // CHECK:       UINT invar_ptr = 0,
        // CHECK:       UINT var_tag = 1,
        // CHECK:     }
        // CHECK:     workload_size0 {
        // CHECK:       UINT workload_size_x = 0x20,
        // CHECK:       UINT workload_size_y = 0x21,
        // CHECK:     }
        // CHECK:     workload_size1 {
        // CHECK:       UINT workload_size_z = 0x22,
        // CHECK:       UINT pad_count_up = 3,
        // CHECK:       UINT pad_count_left = 4,
        // CHECK:       UINT pad_count_down = 5,
        // CHECK:       UINT pad_count_right = 6,
        // CHECK:     }
        // CHECK:     workload_start0 {
        // CHECK:       UINT workload_start_x = 3,
        // CHECK:       UINT workload_start_y = 4,
        // CHECK:     }
        // CHECK:     workload_start1 {
        // CHECK:       UINT workload_start_z = 5,
        // CHECK:     }
        // CHECK:     offset_addr {
        // CHECK:       UINT nthw_ntk = 0,
        // CHECK:       UINT bin_cfg = 1,
        // CHECK:       UINT conv_cond = 1,
        // CHECK:       UINT dense_se = 1,
        // CHECK:       UINT idx_quad = 0,
        // CHECK:       UINT swizzle_key_offset = 1,
        // CHECK:       UINT idu_mrm_clk_en = 0,
        // CHECK:       UINT odu_clk_en = 0,
        // CHECK:       UINT mpe_clk_en = 0,
        // CHECK:       UINT ppe_clk_en = 0,
        // CHECK:       UINT odu_stat_en = 0,
        // CHECK:       UINT idu_stat_en = 0,
        // CHECK:       UINT noc_clk_en = 1,
        // CHECK:       UINT odu_stat_clr_mode = 0,
        // CHECK:       UINT idu_stat_clr_mode = 0,
        // CHECK:       UINT se_only_en = 1,
        // CHECK:       UINT shave_l2_cache_en = 0,
        // CHECK:       UINT idu_dbg_en = 0,
        // CHECK:       UINT sb_read_en = 1,
        // CHECK:       UINT tensor2_act_dense = 1,
        // CHECK:       UINT wt_swizzle_key = 2,
        // CHECK:       UINT wt_swizzle_sel = 1,
        // CHECK:       UINT gif_clk_en = 0,
        // CHECK:     }
        // CHECK:     hwp_wload_id {
        // CHECK:       UINT hwp_wload_id = 0,
        // CHECK:     }
        // CHECK:     var_cfg {
        // CHECK:       UINT invar_lut_rd_en = 1,
        // CHECK:       UINT invar_line_cnt_en = 0,
        // CHECK:       UINT invar_line_cnt = 0,
        // CHECK:       UINT invar_lptr_force = 1,
        // CHECK:       UINT next_sram_job_valid = 0,
        // CHECK:       UINT next_sram_job_addr = 0,
        // CHECK:     }
        // CHECK:     cbarrier_lo = UINT 6,
        // CHECK:     cbarrier_hi {
        // CHECK:       UINT cbarrier_hi = 0,
        // CHECK:     }
        // CHECK:     pbarrier_lo = UINT 0,
        // CHECK:     pbarrier_hi {
        // CHECK:       UINT pbarrier_hi = 0,
        // CHECK:     }
        // CHECK:     halo_region0A {
        // CHECK:       SINT sp_adr_offset = 0xA,
        // CHECK:       UINT tile_select = 6,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 1,
        // CHECK:     }
        // CHECK:     halo_region0B {
        // CHECK:       SINT ac_adr_offset = 0xA,
        // CHECK:       UINT target_width_lsb = 0x20,
        // CHECK:     }
        // CHECK:     halo_region0C {
        // CHECK:       UINT begin_x = 0,
        // CHECK:       UINT begin_y = 0,
        // CHECK:       UINT target_width_msb = 0,
        // CHECK:     }
        // CHECK:     halo_region0D {
        // CHECK:       UINT end_x = 0xF,
        // CHECK:       UINT end_y = 0xF,
        // CHECK:       UINT rsvdD = 0,
        // CHECK:     }
        // CHECK:     halo_region1A {
        // CHECK:       SINT sp_adr_offset = 0,
        // CHECK:       UINT tile_select = 0xA,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 1,
        // CHECK:     }
        // CHECK:     halo_region1B {
        // CHECK:       SINT ac_adr_offset = 9,
        // CHECK:       UINT target_width_lsb = 0x40,
        // CHECK:     }
        // CHECK:     halo_region1C {
        // CHECK:       UINT begin_x = 4,
        // CHECK:       UINT begin_y = 5,
        // CHECK:       UINT target_width_msb = 0,
        // CHECK:     }
        // CHECK:     halo_region1D {
        // CHECK:       UINT end_x = 0xF,
        // CHECK:       UINT end_y = 0x10,
        // CHECK:       UINT rsvdD = 0,
        // CHECK:     }
        // CHECK:     halo_region2A {
        // CHECK:       SINT sp_adr_offset = 0,
        // CHECK:       UINT tile_select = 0x3F,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 1,
        // CHECK:     }
        // CHECK:     halo_region2B {
        // CHECK:       SINT ac_adr_offset = 0x190,
        // CHECK:       UINT target_width_lsb = 0x40,
        // CHECK:     }
        // CHECK:     halo_region2C {
        // CHECK:       UINT begin_x = 9,
        // CHECK:       UINT begin_y = 0,
        // CHECK:       UINT target_width_msb = 0xF,
        // CHECK:     }
        // CHECK:     halo_region2D {
        // CHECK:       UINT end_x = 0x3F,
        // CHECK:       UINT end_y = 0x3F,
        // CHECK:       UINT rsvdD = 0,
        // CHECK:     }
        // CHECK:     halo_region3A {
        // CHECK:       SINT sp_adr_offset = 0,
        // CHECK:       UINT tile_select = 0x21,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 1,
        // CHECK:     }
        // CHECK:     halo_region3B {
        // CHECK:       SINT ac_adr_offset = 9,
        // CHECK:       UINT target_width_lsb = 0x80,
        // CHECK:     }
        // CHECK:     halo_region3C {
        // CHECK:       UINT begin_x = 9,
        // CHECK:       UINT begin_y = 0,
        // CHECK:       UINT target_width_msb = 0xF,
        // CHECK:     }
        // CHECK:     halo_region3D {
        // CHECK:       UINT end_x = 0x3F,
        // CHECK:       UINT end_y = 0x3F,
        // CHECK:       UINT rsvdD = 0,
        // CHECK:     }
        // CHECK:     halo_region4A {
        // CHECK:       SINT sp_adr_offset = 0,
        // CHECK:       UINT tile_select = 4,
        // CHECK:       UINT rsvdA = 0,
        // CHECK:       UINT enable = 1,
        // CHECK:     }
        // CHECK:     halo_region4B {
        // CHECK:       SINT ac_adr_offset = 9,
        // CHECK:       UINT target_width_lsb = 0x100,
        // CHECK:     }
        // CHECK:     halo_region4C {
        // CHECK:       UINT begin_x = 9,
        // CHECK:       UINT begin_y = 0,
        // CHECK:       UINT target_width_msb = 3,
        // CHECK:     }
        // CHECK:     halo_region4D {
        // CHECK:       UINT end_x = 0x3F,
        // CHECK:       UINT end_y = 0x3F,
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
        // CHECK:       UINT cfg_Reserved_0 = 0,
        // CHECK:       UINT cfg_Reserved_1 = 0,
        // CHECK:       UINT cfg_Reserved_2 = 0,
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
        // CHECK:     }
        // CHECK:     te_end0 {
        // CHECK:       UINT te_end_y = 0x3F,
        // CHECK:       UINT te_end_z = 0xF,
        // CHECK:     }
        // CHECK:     te_end1 {
        // CHECK:       UINT te_end_x = 0x3F,
        // CHECK:     }
        // CHECK:     weight_size = UINT 0,
        // CHECK:     weight_num = UINT 0,
        // CHECK:     weight_start = UINT 0,
        // CHECK:     invariant_ = UINT 0x200000,
        // CHECK:     invariant_index_ = UINT 0,
        // CHECK:     deprecated0_var = UINT 0,
        // CHECK:     deprecated1_var = UINT 0,
        // CHECK:     pad_7_1 = UINT 0,
        // CHECK:   }
        // CHECK: >
        NPUReg50XX.DPUVariant descriptor = <
          DpuVariantRegister {
            invar_ptr {
              UINT invar_ptr = 0,
              UINT var_tag = 1,
            }
            workload_size0 {
              UINT workload_size_x = 0x20,
              UINT workload_size_y = 0x21,
            }
            workload_size1 {
              UINT workload_size_z = 0x22,
              UINT pad_count_up = 3,
              UINT pad_count_left = 4,
              UINT pad_count_down = 5,
              UINT pad_count_right = 6,
            }
            workload_start0 {
              UINT workload_start_x = 3,
              UINT workload_start_y = 4,
            }
            workload_start1 {
              UINT workload_start_z = 5,
            }
            offset_addr {
              UINT nthw_ntk = 1,
              UINT bin_cfg = 0,
              UINT conv_cond = 0,
              UINT dense_se = 0,
              UINT idx_quad = 0,
              UINT swizzle_key_offset = 3,
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
              UINT wt_swizzle_key = 0,
              UINT wt_swizzle_sel = 1,
              UINT gif_clk_en = 0,
            }
            hwp_wload_id {
              UINT hwp_wload_id = 0,
            }
            var_cfg {
              UINT invar_lut_rd_en = 0,
              UINT invar_line_cnt_en = 0,
              UINT invar_line_cnt = 0,
              UINT invar_lptr_force = 0,
              UINT next_sram_job_valid = 0,
              UINT next_sram_job_addr = 0,
            }
            cbarrier_lo = UINT 0,
            cbarrier_hi {
              UINT cbarrier_hi = 0,
            }
            pbarrier_lo = UINT 0,
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
              UINT cfg_Reserved_0 = 0,
              UINT cfg_Reserved_1 = 0,
              UINT cfg_Reserved_2 = 0,
              UINT rst_ctxt_new = 0,
              UINT cfg_Reserved_3 = 0,
              UINT cfg_Reserved_4 = 0,
              UINT odu_stat_clr = 0,
              UINT idu_stat_clr = 0,
              UINT cfg_Reserved_5 = 0,
              UINT cfg_Reserved_6 = 0,
            }
            te_beg0 {
              UINT te_beg_y = 0,
              UINT te_beg_z = 0,
            }
            te_beg1 {
              UINT te_beg_x = 0,
            }
            te_end0 {
              UINT te_end_y = 0x3F,
              UINT te_end_z = 0xF,
            }
            te_end1 {
              UINT te_end_x = 0x3F,
            }
            weight_size = UINT 0,
            weight_num = UINT 0,
            weight_start = UINT 0,
            invariant_ = UINT 0x200000,
            invariant_index_ = UINT 0,
            deprecated0_var = UINT 0,
            deprecated1_var = UINT 0,
            pad_7_1 = UINT 0,
          }
        > {invariant_task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, sym_name = "DPUVariant22", task_index = !VPURegMapped.Index<0:0:0>}
        // CHECK: NPUReg50XX.DPUVariant descriptor = <
        // CHECK:   DpuVariantRegister {
        // CHECK:     invar_ptr {
        // CHECK:       UINT invar_ptr = 0,
        // CHECK:       UINT var_tag = 1,
        // CHECK:     }
        // CHECK:     workload_size0 {
        // CHECK:       UINT workload_size_x = 0x20,
        // CHECK:       UINT workload_size_y = 0x21,
        // CHECK:     }
        // CHECK:     workload_size1 {
        // CHECK:       UINT workload_size_z = 0x22,
        // CHECK:       UINT pad_count_up = 3,
        // CHECK:       UINT pad_count_left = 4,
        // CHECK:       UINT pad_count_down = 5,
        // CHECK:       UINT pad_count_right = 6,
        // CHECK:     }
        // CHECK:     workload_start0 {
        // CHECK:       UINT workload_start_x = 3,
        // CHECK:       UINT workload_start_y = 4,
        // CHECK:     }
        // CHECK:     workload_start1 {
        // CHECK:       UINT workload_start_z = 5,
        // CHECK:     }
        // CHECK:     offset_addr {
        // CHECK:       UINT nthw_ntk = 1,
        // CHECK:       UINT bin_cfg = 0,
        // CHECK:       UINT conv_cond = 0,
        // CHECK:       UINT dense_se = 0,
        // CHECK:       UINT idx_quad = 0,
        // CHECK:       UINT swizzle_key_offset = 3,
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
        // CHECK:       UINT wt_swizzle_key = 0,
        // CHECK:       UINT wt_swizzle_sel = 1,
        // CHECK:       UINT gif_clk_en = 0,
        // CHECK:     }
        // CHECK:     hwp_wload_id {
        // CHECK:       UINT hwp_wload_id = 0,
        // CHECK:     }
        // CHECK:     var_cfg {
        // CHECK:       UINT invar_lut_rd_en = 0,
        // CHECK:       UINT invar_line_cnt_en = 0,
        // CHECK:       UINT invar_line_cnt = 0,
        // CHECK:       UINT invar_lptr_force = 0,
        // CHECK:       UINT next_sram_job_valid = 0,
        // CHECK:       UINT next_sram_job_addr = 0,
        // CHECK:     }
        // CHECK:     cbarrier_lo = UINT 0,
        // CHECK:     cbarrier_hi {
        // CHECK:       UINT cbarrier_hi = 0,
        // CHECK:     }
        // CHECK:     pbarrier_lo = UINT 0,
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
        // CHECK:       UINT cfg_Reserved_0 = 0,
        // CHECK:       UINT cfg_Reserved_1 = 0,
        // CHECK:       UINT cfg_Reserved_2 = 0,
        // CHECK:       UINT rst_ctxt_new = 0,
        // CHECK:       UINT cfg_Reserved_3 = 0,
        // CHECK:       UINT cfg_Reserved_4 = 0,
        // CHECK:       UINT odu_stat_clr = 0,
        // CHECK:       UINT idu_stat_clr = 0,
        // CHECK:       UINT cfg_Reserved_5 = 0,
        // CHECK:       UINT cfg_Reserved_6 = 0,
        // CHECK:     }
        // CHECK:     te_beg0 {
        // CHECK:       UINT te_beg_y = 0,
        // CHECK:       UINT te_beg_z = 0,
        // CHECK:     }
        // CHECK:     te_beg1 {
        // CHECK:       UINT te_beg_x = 0,
        // CHECK:     }
        // CHECK:     te_end0 {
        // CHECK:       UINT te_end_y = 0x3F,
        // CHECK:       UINT te_end_z = 0xF,
        // CHECK:     }
        // CHECK:     te_end1 {
        // CHECK:       UINT te_end_x = 0x3F,
        // CHECK:     }
        // CHECK:     weight_size = UINT 0,
        // CHECK:     weight_num = UINT 0,
        // CHECK:     weight_start = UINT 0,
        // CHECK:     invariant_ = UINT 0x200000,
        // CHECK:     invariant_index_ = UINT 0,
        // CHECK:     deprecated0_var = UINT 0,
        // CHECK:     deprecated1_var = UINT 0,
        // CHECK:     pad_7_1 = UINT 0,
        // CHECK:   }
        // CHECK: >
        NPUReg50XX.DPUVariant descriptor = <
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
            }
            workload_start0 {
              UINT workload_start_x = 0,
              UINT workload_start_y = 0,
            }
            workload_start1 {
              UINT workload_start_z = 0,
            }
            offset_addr {
              UINT nthw_ntk = 3,
              UINT bin_cfg = 0,
              UINT conv_cond = 0,
              UINT dense_se = 0,
              UINT idx_quad = 0,
              UINT swizzle_key_offset = 4,
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
              UINT wt_swizzle_key = 0,
              UINT wt_swizzle_sel = 1,
              UINT gif_clk_en = 0,
            }
            hwp_wload_id {
              UINT hwp_wload_id = 0,
            }
            var_cfg {
              UINT invar_lut_rd_en = 0,
              UINT invar_line_cnt_en = 0,
              UINT invar_line_cnt = 0,
              UINT invar_lptr_force = 0,
              UINT next_sram_job_valid = 0,
              UINT next_sram_job_addr = 0,
            }
            cbarrier_lo = UINT 0,
            cbarrier_hi {
              UINT cbarrier_hi = 0,
            }
            pbarrier_lo = UINT 0,
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
              UINT cfg_Reserved_0 = 0,
              UINT cfg_Reserved_1 = 0,
              UINT cfg_Reserved_2 = 0,
              UINT rst_ctxt_new = 0,
              UINT cfg_Reserved_3 = 0,
              UINT cfg_Reserved_4 = 0,
              UINT odu_stat_clr = 0,
              UINT idu_stat_clr = 0,
              UINT cfg_Reserved_5 = 0,
              UINT cfg_Reserved_6 = 0,
            }
            te_beg0 {
              UINT te_beg_y = 5,
              UINT te_beg_z = 4,
            }
            te_beg1 {
              UINT te_beg_x = 0,
            }
            te_end0 {
              UINT te_end_y = 0x3F,
              UINT te_end_z = 0xF,
            }
            te_end1 {
              UINT te_end_x = 0x3F,
            }
            weight_size = UINT 0x90,
            weight_num = UINT 0x20,
            weight_start = UINT 0,
            invariant_ = UINT 0x200000,
            invariant_index_ = UINT 0,
            deprecated0_var = UINT 0,
            deprecated1_var = UINT 0,
            pad_7_1 = UINT 0,
          }
        > {invariant_task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, sym_name = "DPUVariant33", task_index = !VPURegMapped.Index<0:0:0>}
        // CHECK: NPUReg50XX.DPUVariant descriptor = <
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
        // CHECK:     }
        // CHECK:     workload_start0 {
        // CHECK:       UINT workload_start_x = 0,
        // CHECK:       UINT workload_start_y = 0,
        // CHECK:     }
        // CHECK:     workload_start1 {
        // CHECK:       UINT workload_start_z = 0,
        // CHECK:     }
        // CHECK:     offset_addr {
        // CHECK:       UINT nthw_ntk = 3,
        // CHECK:       UINT bin_cfg = 0,
        // CHECK:       UINT conv_cond = 0,
        // CHECK:       UINT dense_se = 0,
        // CHECK:       UINT idx_quad = 0,
        // CHECK:       UINT swizzle_key_offset = 4,
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
        // CHECK:       UINT wt_swizzle_key = 0,
        // CHECK:       UINT wt_swizzle_sel = 1,
        // CHECK:       UINT gif_clk_en = 0,
        // CHECK:     }
        // CHECK:     hwp_wload_id {
        // CHECK:       UINT hwp_wload_id = 0,
        // CHECK:     }
        // CHECK:     var_cfg {
        // CHECK:       UINT invar_lut_rd_en = 0,
        // CHECK:       UINT invar_line_cnt_en = 0,
        // CHECK:       UINT invar_line_cnt = 0,
        // CHECK:       UINT invar_lptr_force = 0,
        // CHECK:       UINT next_sram_job_valid = 0,
        // CHECK:       UINT next_sram_job_addr = 0,
        // CHECK:     }
        // CHECK:     cbarrier_lo = UINT 0,
        // CHECK:     cbarrier_hi {
        // CHECK:       UINT cbarrier_hi = 0,
        // CHECK:     }
        // CHECK:     pbarrier_lo = UINT 0,
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
        // CHECK:       UINT cfg_Reserved_0 = 0,
        // CHECK:       UINT cfg_Reserved_1 = 0,
        // CHECK:       UINT cfg_Reserved_2 = 0,
        // CHECK:       UINT rst_ctxt_new = 0,
        // CHECK:       UINT cfg_Reserved_3 = 0,
        // CHECK:       UINT cfg_Reserved_4 = 0,
        // CHECK:       UINT odu_stat_clr = 0,
        // CHECK:       UINT idu_stat_clr = 0,
        // CHECK:       UINT cfg_Reserved_5 = 0,
        // CHECK:       UINT cfg_Reserved_6 = 0,
        // CHECK:     }
        // CHECK:     te_beg0 {
        // CHECK:       UINT te_beg_y = 5,
        // CHECK:       UINT te_beg_z = 4,
        // CHECK:     }
        // CHECK:     te_beg1 {
        // CHECK:       UINT te_beg_x = 0,
        // CHECK:     }
        // CHECK:     te_end0 {
        // CHECK:       UINT te_end_y = 0x3F,
        // CHECK:       UINT te_end_z = 0xF,
        // CHECK:     }
        // CHECK:     te_end1 {
        // CHECK:       UINT te_end_x = 0x3F,
        // CHECK:     }
        // CHECK:     weight_size = UINT 0x90,
        // CHECK:     weight_num = UINT 0x20,
        // CHECK:     weight_start = UINT 0,
        // CHECK:     invariant_ = UINT 0x200000,
        // CHECK:     invariant_index_ = UINT 0,
        // CHECK:     deprecated0_var = UINT 0,
        // CHECK:     deprecated1_var = UINT 0,
        // CHECK:     pad_7_1 = UINT 0,
        // CHECK:   }
        // CHECK: >
        NPUReg50XX.DPUVariant descriptor = <
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
            }
            workload_start0 {
              UINT workload_start_x = 0,
              UINT workload_start_y = 0,
            }
            workload_start1 {
              UINT workload_start_z = 0,
            }
            offset_addr {
              UINT nthw_ntk = 0,
              UINT bin_cfg = 0,
              UINT conv_cond = 0,
              UINT dense_se = 0,
              UINT idx_quad = 0,
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
              UINT wt_swizzle_key = 0,
              UINT wt_swizzle_sel = 1,
              UINT gif_clk_en = 0,
            }
            hwp_wload_id {
              UINT hwp_wload_id = 0,
            }
            var_cfg {
              UINT invar_lut_rd_en = 0,
              UINT invar_line_cnt_en = 0,
              UINT invar_line_cnt = 0,
              UINT invar_lptr_force = 0,
              UINT next_sram_job_valid = 0,
              UINT next_sram_job_addr = 0,
            }
            cbarrier_lo = UINT 0,
            cbarrier_hi {
              UINT cbarrier_hi = 0,
            }
            pbarrier_lo = UINT 0x38,
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
            dpu_cfg {
              UINT workload_start_odu = 1,
              UINT workload_start_idu = 1,
              UINT workload_prm_sel = 0,
              UINT workload_valid = 0,
              UINT workload_shad_odu = 0,
              UINT workload_shad_idu = 0,
              UINT workload_idu_auto_upd_0 = 1,
              UINT workload_idu_auto_upd_1 = 0,
              UINT workload_odu_auto_upd = 1,
              UINT cfg_Reserved_0 = 0,
              UINT cfg_Reserved_1 = 0,
              UINT cfg_Reserved_2 = 0,
              UINT rst_ctxt_new = 0,
              UINT cfg_Reserved_3 = 0,
              UINT cfg_Reserved_4 = 0,
              UINT odu_stat_clr = 0,
              UINT idu_stat_clr = 0,
              UINT cfg_Reserved_5 = 0,
              UINT cfg_Reserved_6 = 0,
            }
            te_beg0 {
              UINT te_beg_y = 0,
              UINT te_beg_z = 0,
            }
            te_beg1 {
              UINT te_beg_x = 0,
            }
            te_end0 {
              UINT te_end_y = 0x3F,
              UINT te_end_z = 0xF,
            }
            te_end1 {
              UINT te_end_x = 0x3F,
            }
            weight_size = UINT 0,
            weight_num = UINT 0,
            weight_start = UINT 0,
            invariant_ = UINT 0x200000,
            invariant_index_ = UINT 0,
            deprecated0_var = UINT 0,
            deprecated1_var = UINT 0,
            pad_7_1 = UINT 0,
          }
        > {invariant_task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, sym_name = "DPUVariant44", task_index = !VPURegMapped.Index<0:0:0>}
        // CHECK: NPUReg50XX.DPUVariant descriptor = <
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
        // CHECK:     }
        // CHECK:     workload_start0 {
        // CHECK:       UINT workload_start_x = 0,
        // CHECK:       UINT workload_start_y = 0,
        // CHECK:     }
        // CHECK:     workload_start1 {
        // CHECK:       UINT workload_start_z = 0,
        // CHECK:     }
        // CHECK:     offset_addr {
        // CHECK:       UINT nthw_ntk = 0,
        // CHECK:       UINT bin_cfg = 0,
        // CHECK:       UINT conv_cond = 0,
        // CHECK:       UINT dense_se = 0,
        // CHECK:       UINT idx_quad = 0,
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
        // CHECK:       UINT wt_swizzle_key = 0,
        // CHECK:       UINT wt_swizzle_sel = 1,
        // CHECK:       UINT gif_clk_en = 0,
        // CHECK:     }
        // CHECK:     hwp_wload_id {
        // CHECK:       UINT hwp_wload_id = 0,
        // CHECK:     }
        // CHECK:     var_cfg {
        // CHECK:       UINT invar_lut_rd_en = 0,
        // CHECK:       UINT invar_line_cnt_en = 0,
        // CHECK:       UINT invar_line_cnt = 0,
        // CHECK:       UINT invar_lptr_force = 0,
        // CHECK:       UINT next_sram_job_valid = 0,
        // CHECK:       UINT next_sram_job_addr = 0,
        // CHECK:     }
        // CHECK:     cbarrier_lo = UINT 0,
        // CHECK:     cbarrier_hi {
        // CHECK:       UINT cbarrier_hi = 0,
        // CHECK:     }
        // CHECK:     pbarrier_lo = UINT 0x38,
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
        // CHECK:     dpu_cfg {
        // CHECK:       UINT workload_start_odu = 1,
        // CHECK:       UINT workload_start_idu = 1,
        // CHECK:       UINT workload_prm_sel = 0,
        // CHECK:       UINT workload_valid = 0,
        // CHECK:       UINT workload_shad_odu = 0,
        // CHECK:       UINT workload_shad_idu = 0,
        // CHECK:       UINT workload_idu_auto_upd_0 = 1,
        // CHECK:       UINT workload_idu_auto_upd_1 = 0,
        // CHECK:       UINT workload_odu_auto_upd = 1,
        // CHECK:       UINT cfg_Reserved_0 = 0,
        // CHECK:       UINT cfg_Reserved_1 = 0,
        // CHECK:       UINT cfg_Reserved_2 = 0,
        // CHECK:       UINT rst_ctxt_new = 0,
        // CHECK:       UINT cfg_Reserved_3 = 0,
        // CHECK:       UINT cfg_Reserved_4 = 0,
        // CHECK:       UINT odu_stat_clr = 0,
        // CHECK:       UINT idu_stat_clr = 0,
        // CHECK:       UINT cfg_Reserved_5 = 0,
        // CHECK:       UINT cfg_Reserved_6 = 0,
        // CHECK:     }
        // CHECK:     te_beg0 {
        // CHECK:       UINT te_beg_y = 0,
        // CHECK:       UINT te_beg_z = 0,
        // CHECK:     }
        // CHECK:     te_beg1 {
        // CHECK:       UINT te_beg_x = 0,
        // CHECK:     }
        // CHECK:     te_end0 {
        // CHECK:       UINT te_end_y = 0x3F,
        // CHECK:       UINT te_end_z = 0xF,
        // CHECK:     }
        // CHECK:     te_end1 {
        // CHECK:       UINT te_end_x = 0x3F,
        // CHECK:     }
        // CHECK:     weight_size = UINT 0,
        // CHECK:     weight_num = UINT 0,
        // CHECK:     weight_start = UINT 0,
        // CHECK:     invariant_ = UINT 0x200000,
        // CHECK:     invariant_index_ = UINT 0,
        // CHECK:     deprecated0_var = UINT 0,
        // CHECK:     deprecated1_var = UINT 0,
        // CHECK:     pad_7_1 = UINT 0,
        // CHECK:   }
        // CHECK: >
      }
      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.builtin.data.nncmx0 of(@builtin.data.nncmx0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.builtin.tasks.DPUVariant0 of(@builtin.tasks.DPUVariant0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.builtin.tasks.DPUInvariant0 of(@builtin.tasks.DPUInvariant0) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.text.invariants of(@text.invariants) type(<STT_SECTION>) size(0) value(0)
        ELF.Symbol @elfsym.text.variants of(@text.variants) type(<STT_SECTION>) size(0) value(0)
      }
    }
    return
  }
}
