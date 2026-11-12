//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUReg60XX_TYPES
#define VPUX_COMPILER_DIALECT_NPUReg60XX_TYPES

include "vpux/compiler/dialect/VPURegMapped/dialect.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/dialect.td"

//
// RegField Specialization
//

class NPUReg60XX_Version<int vMajor, int vMinor, int vPatch> : VPURegMapped_Version {
    let major = vMajor;
    let minor = vMinor;
    let patch = vPatch;
}

// MI Version that acts as baseline for NPU6 NNRT API
// All registers default to this baseline unless specified otherwise
def NPUReg60XX_BaselineVersion : NPUReg60XX_Version<11, 13, 3>;

class NPUReg60XX_RegFieldWrapper<string name = "", int width = 0, int pos = 0, string dataType = "UINT", VPURegMapped_Version version = NPUReg60XX_BaselineVersion> :
    VPURegMapped_RegFieldWrapper<NPUReg60XX_Dialect, name, width, pos, dataType, version> {
    }

//
// Register Specialization
//

class NPUReg60XX_RegisterWrapper<string name = "", int size = 0, int address = 0x00, list<string> regFieldTypes = [], int resetValue = 0x0, bit allowOverlap = 0> :
    VPURegMapped_RegisterWrapper<NPUReg60XX_Dialect, "vpux::NPUReg60XX::", name, size, address, regFieldTypes, resetValue, allowOverlap> {

}

//
// DPUInvariantRegisters
//

// se_addr
def bf_SeAddr : NPUReg60XX_RegFieldWrapper<"se_addr", 32, 0>;
def reg_SeAddr : NPUReg60XX_RegisterWrapper<"se_addr", 4, 0x0, ["se_addr"]>;

// sparsity_addr
def bf_SparsityAddr : NPUReg60XX_RegFieldWrapper<"sparsity_addr", 32, 0>;
def reg_SparsityAddr : NPUReg60XX_RegisterWrapper<"sparsity_addr", 4, 0x4, ["sparsity_addr"]>;

// se_sise
def bf_SeSize : NPUReg60XX_RegFieldWrapper<"se_size", 32, 0>;
def reg_SeSize : NPUReg60XX_RegisterWrapper<"se_size", 4, 0x8, ["se_size"]>;

// z_config
def bf_SeZSplit : NPUReg60XX_RegFieldWrapper<"se_z_split", 4, 0>;
def bf_NumSesInZDir : NPUReg60XX_RegFieldWrapper<"num_ses_in_z_dir", 9, 4>;
def bf_CmSpPattern : NPUReg60XX_RegFieldWrapper<"cm_sp_pattern", 16, 13>;
def bf_Npo2SeZSplitEnable : NPUReg60XX_RegFieldWrapper<"npo2_se_z_split_en", 1, 29>;
def bf_SubtractMpeWtbias : NPUReg60XX_RegFieldWrapper<"subtract_mpe_wtbias", 1, 30>;
def bf_AddrFormatSel : NPUReg60XX_RegFieldWrapper<"addr_format_sel", 1, 31>;
def reg_ZConfig : NPUReg60XX_RegisterWrapper<"z_config", 4, 0xC,
    ["se_z_split", "num_ses_in_z_dir", "cm_sp_pattern", "npo2_se_z_split_en", "subtract_mpe_wtbias", "addr_format_sel"], 0x0>;

// kernel_pad_cfg
def bf_MpeAssign : NPUReg60XX_RegFieldWrapper<"mpe_assign", 1, 0>;
def bf_PadRightEn : NPUReg60XX_RegFieldWrapper<"pad_right_en", 1, 1>;
def bf_PadLeftEn : NPUReg60XX_RegFieldWrapper<"pad_left_en", 1, 2>;
def bf_PadBottomEn : NPUReg60XX_RegFieldWrapper<"pad_bottom_en", 1, 3>;
def bf_PadTopEn : NPUReg60XX_RegFieldWrapper<"pad_top_en", 1, 4>;
def bf_KernelY : NPUReg60XX_RegFieldWrapper<"kernel_y", 4, 5>;
def bf_KernelX : NPUReg60XX_RegFieldWrapper<"kernel_x", 4, 9>;
def bf_WtPltCfg : NPUReg60XX_RegFieldWrapper<"wt_plt_cfg", 2, 13>;
def bf_ActDense : NPUReg60XX_RegFieldWrapper<"act_dense", 1, 15>;
def bf_WtDense : NPUReg60XX_RegFieldWrapper<"wt_dense", 1, 16>;
def bf_StrideYEn : NPUReg60XX_RegFieldWrapper<"stride_y_en", 1, 17>;
def bf_StrideY : NPUReg60XX_RegFieldWrapper<"stride_y", 3, 18>;
def bf_DynamicBwEn : NPUReg60XX_RegFieldWrapper<"dynamic_bw_en", 1, 21>;
def bf_DwWtSpIns : NPUReg60XX_RegFieldWrapper<"dw_wt_sp_ins", 1, 22>;
def bf_Layer1WtSpIns : NPUReg60XX_RegFieldWrapper<"layer1_wt_sp_ins", 1, 23>;
def bf_Layer1CmpEn : NPUReg60XX_RegFieldWrapper<"layer1_cmp_en", 1, 24>;
def bf_PoolOptEn : NPUReg60XX_RegFieldWrapper<"pool_opt_en", 1, 25>;
def bf_SdpOptDis : NPUReg60XX_RegFieldWrapper<"sdp_opt_dis", 1, 26>;
def bf_CtrlCmxOptDis : NPUReg60XX_RegFieldWrapper<"ctrl_cmx_opt_dis", 1, 27>;
def bf_CtrlSeOptDis : NPUReg60XX_RegFieldWrapper<"ctrl_se_opt_dis", 1, 28>;
def bf_SpSeTblSegment : NPUReg60XX_RegFieldWrapper<"sp_se_tbl_segment", 1, 29>;
def bf_WtPltCfg2 : NPUReg60XX_RegFieldWrapper<"wt_plt_cfg_2", 1, 30>;
def bf_FlushWl : NPUReg60XX_RegFieldWrapper<"flush_wl", 1, 31>;
def reg_KernelPadCfg : NPUReg60XX_RegisterWrapper<"kernel_pad_cfg", 4, 0x10,
    ["mpe_assign", "pad_right_en", "pad_left_en", "pad_bottom_en", "pad_top_en", "kernel_y", "kernel_x", "wt_plt_cfg",
    "act_dense", "wt_dense", "stride_y_en", "stride_y", "dynamic_bw_en", "dw_wt_sp_ins", "layer1_wt_sp_ins",
    "layer1_cmp_en", "pool_opt_en", "sdp_opt_dis", "ctrl_cmx_opt_dis", "ctrl_se_opt_dis", "sp_se_tbl_segment", "wt_plt_cfg_2", "flush_wl"]>;

// tensor_size0
def bf_TensorSizeX : NPUReg60XX_RegFieldWrapper<"tensor_size_x", 14, 0>;
def bf_TensorSizeY : NPUReg60XX_RegFieldWrapper<"tensor_size_y", 14, 14>;
def reg_TensorSize0 : NPUReg60XX_RegisterWrapper<"tensor_size0", 4, 0x14, ["tensor_size_x", "tensor_size_y"]>;

// tensor_size1
def bf_TensorSizeZ : NPUReg60XX_RegFieldWrapper<"tensor_size_z", 14, 0>;
def bf_Npo2_se_size : NPUReg60XX_RegFieldWrapper<"npo2_se_size",  9, 14>;
def bf_Tensor_cmp_size : NPUReg60XX_RegFieldWrapper<"tensor_cmp_size",  4, 23>;
def bf_Tensor_cmp_offset : NPUReg60XX_RegFieldWrapper<"tensor_cmp_offset",  4, 27>;
def bf_WeightTableFormat : NPUReg60XX_RegFieldWrapper<"weight_table_format",  1, 31>;
def reg_TensorSize1 : NPUReg60XX_RegisterWrapper<"tensor_size1", 4, 0x18,
    ["tensor_size_z", "npo2_se_size", "tensor_cmp_size", "tensor_cmp_offset", "weight_table_format"]>;

// tensor_start
def bf_TensorStart : NPUReg60XX_RegFieldWrapper<"tensor_start", 32, 0>;
def reg_TensorStart : NPUReg60XX_RegisterWrapper<"tensor_start", 4, 0x1C, ["tensor_start"]>;

// tensor_mode
def bf_Wmode : NPUReg60XX_RegFieldWrapper<"wmode", 4, 0>;
def bf_Amode : NPUReg60XX_RegFieldWrapper<"amode", 4, 4>;
def bf_Stride : NPUReg60XX_RegFieldWrapper<"stride", 3, 8>;
def bf_ZmInput : NPUReg60XX_RegFieldWrapper<"zm_input", 1, 11>;
def bf_DwInput : NPUReg60XX_RegFieldWrapper<"dw_input", 1, 12>;
def bf_CmInput : NPUReg60XX_RegFieldWrapper<"cm_input", 1, 13>;
def bf_WorkloadOperation : NPUReg60XX_RegFieldWrapper<"workload_operation", 2, 14>;
def bf_PadValue : NPUReg60XX_RegFieldWrapper<"pad_value", 16, 16>;
def reg_TensorMode : NPUReg60XX_RegisterWrapper<"tensor_mode", 4, 0x20,
    ["wmode", "amode", "stride", "zm_input", "dw_input", "cm_input", "workload_operation", "pad_value"]>;

// elops_sparsity_addr
def bf_ElopsSparityAddr : NPUReg60XX_RegFieldWrapper<"elops_sparsity_addr", 32, 0>;
def reg_ElopsSparityAddr : NPUReg60XX_RegisterWrapper<"elops_sparsity_addr", 4, 0x24, ["elops_sparsity_addr"]>;

// elops_se_addr
def bf_ElopsSeAddr : NPUReg60XX_RegFieldWrapper<"elops_se_addr", 32, 0>;
def reg_ElopsSeAddr : NPUReg60XX_RegisterWrapper<"elops_se_addr", 4, 0x28, ["elops_se_addr"]>;

// elops_wload
def bf_ElopWload : NPUReg60XX_RegFieldWrapper<"elop_wload", 1, 0>;
def bf_SeedWload : NPUReg60XX_RegFieldWrapper<"seed_wload", 1, 1>;
def bf_FifoWrWload : NPUReg60XX_RegFieldWrapper<"fifo_wr_wload", 1, 2>;
def bf_ElopWloadType : NPUReg60XX_RegFieldWrapper<"elop_wload_type", 1, 3>;
def bf_PoolWtData : NPUReg60XX_RegFieldWrapper<"pool_wt_data", 16, 4>;
def bf_WtPackEn : NPUReg60XX_RegFieldWrapper<"wt_pack_en", 1, 22>;
def bf_WtPtrDisPackEn : NPUReg60XX_RegFieldWrapper<"wt_ptr_dis_pack_en", 1, 23>;
def bf_ActActWtPtrDis : NPUReg60XX_RegFieldWrapper<"act_act_wt_ptr_dis", 1, 24>;
def bf_WtPtrDis : NPUReg60XX_RegFieldWrapper<"wt_ptr_dis", 1, 25>;
def bf_PoolWtRdDis : NPUReg60XX_RegFieldWrapper<"pool_wt_rd_dis", 1, 26>;
def bf_ElopOperation : NPUReg60XX_RegFieldWrapper<"elop_operation", 3, 27>;
def bf_RepeatDis : NPUReg60XX_RegFieldWrapper<"repeat_dis", 1, 30>;
def bf_ElopTensor2Cmp : NPUReg60XX_RegFieldWrapper<"elop_tensor2_cmp", 1, 31>;
def reg_ElopsWload : NPUReg60XX_RegisterWrapper<"elops_wload", 4, 0x2C,
    ["elop_wload", "seed_wload", "fifo_wr_wload", "elop_wload_type", "pool_wt_data", "wt_pack_en", "wt_ptr_dis_pack_en",
    "act_act_wt_ptr_dis", "wt_ptr_dis", "pool_wt_rd_dis", "elop_operation", "repeat_dis", "elop_tensor2_cmp"], 0x0>;

// act0_offset
def bf_Adr0Offset : NPUReg60XX_RegFieldWrapper<"act_offset0", 32, 0>;
def reg_Act0Offset : NPUReg60XX_RegisterWrapper<"act_offset0", 4, 0x30, ["act_offset0"]>;


// base_offset_a
def bf_BaseOffset0 : NPUReg60XX_RegFieldWrapper<"base_offset_a", 32, 0>;
def reg_BaseOffsetA : NPUReg60XX_RegisterWrapper<"base_offset_a", 4, 0x34, ["base_offset_a"]>;

// base_offset_b
def bf_BaseOffset2 : NPUReg60XX_RegFieldWrapper<"base_offset_2", 9, 0>;
def bf_BaseOffset3 : NPUReg60XX_RegFieldWrapper<"base_offset_3", 9, 9>;
def bf_DwOptOffset : NPUReg60XX_RegFieldWrapper<"dw_opt_offset", 6, 18>;
def bf_DwOptEn : NPUReg60XX_RegFieldWrapper<"dw_opt_en", 1, 24>;
def bf_Dw3x3s1OptDis : NPUReg60XX_RegFieldWrapper<"dw_3x3s1_opt_dis", 1, 25>;
def bf_WtDenseOptEn : NPUReg60XX_RegFieldWrapper<"wt_dense_opt_en", 1, 26>;
def bf_SmallHwOptEn : NPUReg60XX_RegFieldWrapper<"small_hw_opt_en", 1, 27>;
def bf_IduCmxMuxMode : NPUReg60XX_RegFieldWrapper<"idu_cmx_mux_mode", 2, 28>;
def bf_PerGroupZeroPoint : NPUReg60XX_RegFieldWrapper<"per_group_zero_point", 1, 30>;
def bf_ZeroPointOnly : NPUReg60XX_RegFieldWrapper<"zero_point_only", 1, 31>;
def reg_BaseOffsetB : NPUReg60XX_RegisterWrapper<"base_offset_b", 4, 0x38,
    ["base_offset_2", "base_offset_3", "dw_opt_offset", "dw_opt_en", "dw_3x3s1_opt_dis",
    "wt_dense_opt_en", "small_hw_opt_en", "idu_cmx_mux_mode", "per_group_zero_point", "zero_point_only"]>;

// wt_offset
def bf_WtOffset : NPUReg60XX_RegFieldWrapper<"wt_offset", 32, 0>;
def reg_WtOffset : NPUReg60XX_RegisterWrapper<"wt_offset", 4, 0x3C, ["wt_offset"]>;

// odu_cfg
def bf_Dtype : NPUReg60XX_RegFieldWrapper<"dtype", 3, 0>;
def bf_WcbAcMode : NPUReg60XX_RegFieldWrapper<"wcb_ac_mode", 1, 3>;
def bf_WcbSpMode : NPUReg60XX_RegFieldWrapper<"wcb_sp_mode", 1, 4>;
def bf_SpValue : NPUReg60XX_RegFieldWrapper<"sp_value", 8, 5>;
def bf_SpOutEn : NPUReg60XX_RegFieldWrapper<"sp_out_en", 1, 13>;
def bf_CmxPortMuxingDisable : NPUReg60XX_RegFieldWrapper<"cmx_port_muxing_disable", 1, 14>;
def bf_WriteSp : NPUReg60XX_RegFieldWrapper<"write_sp", 1, 15>;
def bf_WritePt : NPUReg60XX_RegFieldWrapper<"write_pt", 1, 16>;
def bf_WriteAc : NPUReg60XX_RegFieldWrapper<"write_ac", 1, 17>;
def bf_Mode : NPUReg60XX_RegFieldWrapper<"mode", 2, 18>;
def bf_Grid : NPUReg60XX_RegFieldWrapper<"grid", 1, 20>;
def bf_SwizzleKey : NPUReg60XX_RegFieldWrapper<"swizzle_key", 3, 21>;
def bf_WlBpOnStartEn : NPUReg60XX_RegFieldWrapper<"wl_bp_on_start_en", 1, 24>;
def bf_Nthw : NPUReg60XX_RegFieldWrapper<"nthw", 2, 25>;
def bf_Permutation : NPUReg60XX_RegFieldWrapper<"permutation", 3, 27>;
def bf_WcbStallAvoidance : NPUReg60XX_RegFieldWrapper<"wcb_stall_avoidance", 1, 30>;
def bf_WcbBypass : NPUReg60XX_RegFieldWrapper<"wcb_bypass", 1, 31>;
def reg_OduCfg : NPUReg60XX_RegisterWrapper<"odu_cfg", 4, 0x40,
    ["dtype", "wcb_ac_mode", "wcb_sp_mode", "sp_value", "sp_out_en", "cmx_port_muxing_disable", "write_sp", "write_pt", "write_ac", "mode", "grid",
    "swizzle_key", "wl_bp_on_start_en", "nthw", "permutation", "wcb_stall_avoidance", "wcb_bypass"]>;

// odu_cfg_2
def bf_MnEnable: NPUReg60XX_RegFieldWrapper<"mn_enable", 1, 0>;
def bf_MxEnable : NPUReg60XX_RegFieldWrapper<"mx_enable", 1, 1>;
def bf_MtEnable : NPUReg60XX_RegFieldWrapper<"mt_enable", 1, 2>;
def bf_LocalRegionEnable : NPUReg60XX_RegFieldWrapper<"local_region_enable", 1, 3>;
def bf_LocalRegionSuppress : NPUReg60XX_RegFieldWrapper<"local_region_suppress", 1, 4>;
def bf_MinMaxDtype : NPUReg60XX_RegFieldWrapper<"min_max_dtype", 3, 5>;
def bf_MinMaxTotalTileSelect : NPUReg60XX_RegFieldWrapper<"min_max_total_tile_select", 6, 8>;
def bf_PltPtr : NPUReg60XX_RegFieldWrapper<"plt_ptr", 16, 16>;
def reg_OduCfg2 : NPUReg60XX_RegisterWrapper<"odu_cfg_2", 4, 0x44,
    ["mn_enable", "mx_enable", "mt_enable", "local_region_enable", "local_region_suppress", "min_max_dtype", "min_max_total_tile_select", "plt_ptr"]>;

// odu_mn_base
def bf_MnBase : NPUReg60XX_RegFieldWrapper<"mn_base", 28, 4>;
def reg_OduMnBase : NPUReg60XX_RegisterWrapper<"odu_mn_base", 4, 0x48, ["mn_base"]>;

// odu_mx_base
def bf_MxBase : NPUReg60XX_RegFieldWrapper<"mx_base", 28, 4>;
def reg_OduMxBase : NPUReg60XX_RegisterWrapper<"odu_mx_base", 4, 0x4C, ["mx_base"]>;

// te_dim0
def bf_TeDimY : NPUReg60XX_RegFieldWrapper<"te_dim_y", 13, 0>;
def bf_TeDimZ : NPUReg60XX_RegFieldWrapper<"te_dim_z", 13, 13>;
def reg_TeDim0 : NPUReg60XX_RegisterWrapper<"te_dim0", 4, 0x50, ["te_dim_y", "te_dim_z"]>;

// te_dim1
def bf_TeDimX : NPUReg60XX_RegFieldWrapper<"te_dim_x", 13, 0>;
def reg_TeDim1 : NPUReg60XX_RegisterWrapper<"te_dim1", 4, 0x54, ["te_dim_x"]>;

// mt_base
def bf_MtBase : NPUReg60XX_RegFieldWrapper<"mt_base", 28, 4>;
def reg_MtBase : NPUReg60XX_RegisterWrapper<"mt_base", 4, 0x58, ["mt_base"]>;

// sp_base
def bf_SpBase : NPUReg60XX_RegFieldWrapper<"sp_base", 32, 0>;
def reg_SpBase : NPUReg60XX_RegisterWrapper<"sp_base", 4, 0x5C, ["sp_base"]>;

// mpe_cfg
def bf_MpeWtbias : NPUReg60XX_RegFieldWrapper<"mpe_wtbias", 8, 0, "UINT">;   // TODO E#80099
def bf_MpeActbias : NPUReg60XX_RegFieldWrapper<"mpe_actbias", 8, 8, "UINT">; // TODO E#80099
def bf_MpeMode : NPUReg60XX_RegFieldWrapper<"mpe_mode", 3, 16>;
def bf_MpeDense : NPUReg60XX_RegFieldWrapper<"mpe_dense", 1, 19>;
def bf_MrmWeightDense : NPUReg60XX_RegFieldWrapper<"mrm_weight_dense", 1, 20>;
def bf_MrmActDense : NPUReg60XX_RegFieldWrapper<"mrm_act_dense", 1, 21>;
def bf_MpeDaz : NPUReg60XX_RegFieldWrapper<"mpe_daz", 1, 22>;
def bf_MpeFtz : NPUReg60XX_RegFieldWrapper<"mpe_ftz", 1, 23>;
def bf_CimEn : NPUReg60XX_RegFieldWrapper<"cim_en", 2, 24>;
def bf_CimIc32En : NPUReg60XX_RegFieldWrapper<"cim_ic32_en", 1, 26>;
def reg_MpeCfg : NPUReg60XX_RegisterWrapper<"mpe_cfg", 4, 0x60,
    ["mpe_wtbias", "mpe_actbias", "mpe_mode", "mpe_dense", "mrm_weight_dense", "mrm_act_dense", "mpe_daz", "mpe_ftz", "cim_en", "cim_ic32_en"]>;

// mpe_bus_data_sel
def bf_MpeBusDataSel : NPUReg60XX_RegFieldWrapper<"mpe_bus_data_sel", 32, 0>;
def reg_MpeBusDataSel : NPUReg60XX_RegisterWrapper<"mpe_bus_data_sel", 4, 0x64, ["mpe_bus_data_sel"]>;

// elop_scale
def bf_ElopScaleB : NPUReg60XX_RegFieldWrapper<"elop_scale_b", 16, 0>;
def bf_ElopScaleA : NPUReg60XX_RegFieldWrapper<"elop_scale_a", 16, 16>;
def bf_ElopScaleBFp : NPUReg60XX_RegFieldWrapper<"elop_scale_b_fp", 16, 0, "FP">;
def bf_ElopScaleAFp : NPUReg60XX_RegFieldWrapper<"elop_scale_a_fp", 16, 16, "FP">;
def bf_ElopScaleBBfp : NPUReg60XX_RegFieldWrapper<"elop_scale_b_bfp", 16, 0, "BF">;
def bf_ElopScaleABfp : NPUReg60XX_RegFieldWrapper<"elop_scale_a_bfp", 16, 16, "BF">;
def reg_ElopScale : NPUReg60XX_RegisterWrapper<"elop_scale", 4, 0x68, ["elop_scale_b", "elop_scale_a", "elop_scale_b_fp", "elop_scale_a_fp", "elop_scale_b_bfp", "elop_scale_a_bfp"], 0x0, 1>;

// ppe_cfg
def bf_PpeG8BiasC : NPUReg60XX_RegFieldWrapper<"ppe_g8_bias_c", 9, 0, "SINT">;
def bf_PpeG8BiasB : NPUReg60XX_RegFieldWrapper<"ppe_g8_bias_b", 9, 9>;
def bf_PpeG8BiasA : NPUReg60XX_RegFieldWrapper<"ppe_g8_bias_a", 9, 18>;
def reg_PpeCfg : NPUReg60XX_RegisterWrapper<"ppe_cfg", 4, 0x6C, ["ppe_g8_bias_c", "ppe_g8_bias_b", "ppe_g8_bias_a"]>;

// unused 0
def bf_Unused0Invariant : NPUReg60XX_RegFieldWrapper<"unused0_inv", 32, 0, "UINT">;
def reg_Unused0Invariant : NPUReg60XX_RegisterWrapper<"unused0_inv", 4, 0x70, ["unused0_inv"]>;

// unused 1
def bf_Unused1Invariant : NPUReg60XX_RegFieldWrapper<"unused1_inv", 32, 0, "UINT">;
def reg_Unused1Invariant : NPUReg60XX_RegisterWrapper<"unused1_inv", 4, 0x74, ["unused1_inv"]>;

// ppe_scale_ctrl
def bf_PpeScaleOverride : NPUReg60XX_RegFieldWrapper<"ppe_scale_override", 1, 0>;
def bf_PpeFpScaleOverride : NPUReg60XX_RegFieldWrapper<"ppe_fp_scale_override", 1, 1>;
def bf_PpeScaleDtype : NPUReg60XX_RegFieldWrapper<"ppe_scale_dtype", 3, 2>;
def reg_PpeScaleCtrl : NPUReg60XX_RegisterWrapper<"ppe_scale_ctrl", 4, 0x78, ["ppe_scale_override", "ppe_fp_scale_override", "ppe_scale_dtype"], 0x0>;

// unused 2
def bf_Unused2Invariant : NPUReg60XX_RegFieldWrapper<"unused2_inv", 32, 0, "UINT">;
def reg_Unused2Invariant : NPUReg60XX_RegisterWrapper<"unused2_inv", 4, 0x7C, ["unused2_inv"]>;

// ppe_scale_hclamp: fp clamp
def bf_PpeFpScaleHclamp : NPUReg60XX_RegFieldWrapper<"ppe_fp_scale_hclamp", 32, 0, "FP">;
def reg_PpeScaleHclamp : NPUReg60XX_RegisterWrapper<"ppe_scale_hclamp", 4, 0x80, ["ppe_fp_scale_hclamp"], 0x0>;

// ppe_scale_lclamp: fp clamp
def bf_fpPpeScaleLclamp : NPUReg60XX_RegFieldWrapper<"ppe_fp_scale_lclamp", 32, 0, "FP">;
def reg_PpeScaleLclamp : NPUReg60XX_RegisterWrapper<"ppe_scale_lclamp", 4, 0x84, ["ppe_fp_scale_lclamp"], 0x0>;

// ppe_misc
def bf_PpeMode : NPUReg60XX_RegFieldWrapper<"ppe_mode", 2, 0>;
def bf_PpeRndFp : NPUReg60XX_RegFieldWrapper<"ppe_rnd_fp", 2, 2>;
def bf_PpeRndInt : NPUReg60XX_RegFieldWrapper<"ppe_rnd_int", 2, 4>;
def bf_PpeFp16Ftz : NPUReg60XX_RegFieldWrapper<"ppe_fp16_ftz", 1, 6>;
def bf_PpeFp16Clamp : NPUReg60XX_RegFieldWrapper<"ppe_fp16_clamp", 1, 7>;
def bf_PpeMult2Mode: NPUReg60XX_RegFieldWrapper<"ppe_mult2_mode", 1, 12>;

def reg_PpeMisc : NPUReg60XX_RegisterWrapper<"ppe_misc", 4, 0x88, ["ppe_mode" ,"ppe_rnd_fp" ,"ppe_rnd_int" ,
    "ppe_fp16_ftz", "ppe_fp16_clamp", "ppe_mult2_mode"]>;

// ppe_fp_bias
def bf_PpeFpBias : NPUReg60XX_RegFieldWrapper<"ppe_fp_bias", 32, 0, "FP">;
def reg_PpeFpBias : NPUReg60XX_RegisterWrapper<"ppe_fp_bias", 4, 0x8C, ["ppe_fp_bias"]>;

// ppe_fp_scale
def bf_PpeFpScale : NPUReg60XX_RegFieldWrapper<"ppe_fp_scale", 32, 0, "FP">;
def reg_PpeFpScale : NPUReg60XX_RegisterWrapper<"ppe_fp_scale", 4, 0x90, ["ppe_fp_scale"]>;

// ppe_fp_prelu
def bf_PpeFpPrelu : NPUReg60XX_RegFieldWrapper<"ppe_fp_prelu", 32, 0, "FP">;
def reg_PpeFpPrelu : NPUReg60XX_RegisterWrapper<"ppe_fp_prelu", 4, 0x94, ["ppe_fp_prelu"]>;

// ppe_fp_cfg
def bf_PpeFpConvert : NPUReg60XX_RegFieldWrapper<"ppe_fp_convert", 3, 0>;
//  Note: ppe prelu needs to be off by default, only turned on explicitly
def reg_PpeFpCfg : NPUReg60XX_RegisterWrapper<"ppe_fp_cfg", 4, 0x98, ["ppe_fp_convert"], 0x00000008>;

// odu_ac_base
def bf_AcBase : NPUReg60XX_RegFieldWrapper<"ac_base", 28, 4>;
def reg_OduAcBase : NPUReg60XX_RegisterWrapper<"odu_ac_base", 4, 0x9C, ["ac_base"]>;

// hwp_ctrl
def bf_HwpEn : NPUReg60XX_RegFieldWrapper<"hwp_en", 1, 0>;
def bf_HwpStatMode : NPUReg60XX_RegFieldWrapper<"hwp_stat_mode", 3, 1>;
def bf_HwpTimerEn : NPUReg60XX_RegFieldWrapper<"local_timer_en", 1, 4>;
def bf_HwpTimerRst : NPUReg60XX_RegFieldWrapper<"local_timer_rst", 1, 5>;
// rsvd : 9
def bf_HwpUniqueID : NPUReg60XX_RegFieldWrapper<"unique_ID", 16, 16>;

def reg_HwpCtrl : NPUReg60XX_RegisterWrapper<"hwp_ctrl", 4, 0xA0, ["hwp_en", "hwp_stat_mode", "local_timer_en", "local_timer_rst", "unique_ID"]>;

// hwp_cmx_mem_addr
def bf_HwpCmxMemAddr : NPUReg60XX_RegFieldWrapper<"hwp_cmx_mem_addr", 32, 0>;
def reg_HwpCmxMemAddr : NPUReg60XX_RegisterWrapper<"hwp_cmx_mem_addr", 4, 0xA4, ["hwp_cmx_mem_addr"]>;

// tensor2_start
def bf_Tensor2Start : NPUReg60XX_RegFieldWrapper<"tensor2_start", 32, 0>;
def reg_Tensor2Start : NPUReg60XX_RegisterWrapper<"tensor2_start", 4, 0xA8, ["tensor2_start"]>;

// ppe_lut_ptr
def bf_PpeLutPtr : NPUReg60XX_RegFieldWrapper<"ppe_lut_ptr", 16, 0>;
def bf_PpeLutPtrForce : NPUReg60XX_RegFieldWrapper<"ppe_lut_ptr_force", 1, 18>;
def reg_PpeLutPtr : NPUReg60XX_RegisterWrapper<"ppe_lut_ptr", 4, 0xAC, ["ppe_lut_ptr", "ppe_lut_ptr_force"]>;

// invar_tag
def bf_NvarTag : NPUReg60XX_RegFieldWrapper<"nvar_tag", 32, 0>;
def reg_NvarTag : NPUReg60XX_RegisterWrapper<"nvar_tag", 4, 0xB0, ["nvar_tag"]>;

// dyn_prim_cfg
def bf_DynPrimitiveId : NPUReg60XX_RegFieldWrapper<"dyn_primitive_id", 10, 0>;
def bf_DynPrimitiveId2 : NPUReg60XX_RegFieldWrapper<"dyn_primitive_id2", 10, 10>;
def reg_DynPrimCfg : NPUReg60XX_RegisterWrapper<"dyn_prim_cfg", 4, 0xB4, ["dyn_primitive_id", "dyn_primitive_id2"]>;

// dyn_scale
def bf_DynScaleFetchAddr : NPUReg60XX_RegFieldWrapper<"dyn_scale_fetch_addr", 19, 0>;
def bf_DynScaleFetchAddrEn : NPUReg60XX_RegFieldWrapper<"dyn_scale_fetch_addr_en", 1, 19>;
def bf_DynPpePadOvrwrEn : NPUReg60XX_RegFieldWrapper<"dyn_ppe_pad_ovrwr_en", 1, 20>;
def bf_DynIduWtNumModEn : NPUReg60XX_RegFieldWrapper<"dyn_idu_wt_num_mod_en", 1, 21>;
def bf_DynOduZEndModEn : NPUReg60XX_RegFieldWrapper<"dyn_odu_z_end_mod_en", 1, 22>;
def bf_DynConvContModEn : NPUReg60XX_RegFieldWrapper<"dyn_conv_cont_mod_en", 1, 23>;
def bf_DynIduIcModEn : NPUReg60XX_RegFieldWrapper<"dyn_idu_ic_mod_en", 1, 24>;
def bf_DynScaleCfg : NPUReg60XX_RegFieldWrapper<"dyn_scale_cfg", 3, 25>;
def bf_DynStatEn : NPUReg60XX_RegFieldWrapper<"dyn_stat_en", 1, 28>;
def bf_HwpStallDis : NPUReg60XX_RegFieldWrapper<"hwp_stall_dis", 1, 29>;
def reg_DynScale : NPUReg60XX_RegisterWrapper<"dyn_scale", 4, 0xB8,
    ["dyn_scale_fetch_addr", "dyn_scale_fetch_addr_en", "dyn_ppe_pad_ovrwr_en", "dyn_idu_wt_num_mod_en", "dyn_odu_z_end_mod_en",
     "dyn_conv_cont_mod_en", "dyn_idu_ic_mod_en", "dyn_scale_cfg", "dyn_stat_en", "hwp_stall_dis"]>;

// dyn_ppe_data
def bf_DynPpeData : NPUReg60XX_RegFieldWrapper<"dyn_ppe_data", 32, 0, "SINT">;
def reg_DynPpeData : NPUReg60XX_RegisterWrapper<"dyn_ppe_data", 4, 0xBC, ["dyn_ppe_data"]>;

// odu_local_a
def bf_LocalRegionBegX : NPUReg60XX_RegFieldWrapper<"local_region_beg_x", 13, 0>;
def bf_LocalRegionEndX : NPUReg60XX_RegFieldWrapper<"local_region_end_x", 13, 16>;
def reg_OduLocalA : NPUReg60XX_RegisterWrapper<"odu_local_a", 4, 0xC0, ["local_region_beg_x", "local_region_end_x"]>;

// odu_local_b
def bf_LocalRegionBegY : NPUReg60XX_RegFieldWrapper<"local_region_beg_y", 13, 0>;
def bf_LocalRegionEndY : NPUReg60XX_RegFieldWrapper<"local_region_end_y", 13, 16>;
def reg_OduLocalB : NPUReg60XX_RegisterWrapper<"odu_local_b", 4, 0xC4, ["local_region_beg_y", "local_region_end_y"]>;

// halo_0_offset
def bf_Halo0MmAdrOffset : NPUReg60XX_RegFieldWrapper<"halo_0_mm_adr_offset", 21, 0>;
def reg_Halo0Offset : NPUReg60XX_RegisterWrapper<"halo_0_offset", 4, 0xC8, ["halo_0_mm_adr_offset"]>;

// halo_1_offset
def bf_Halo1MmAdrOffset : NPUReg60XX_RegFieldWrapper<"halo_1_mm_adr_offset", 21, 0>;
def reg_Halo1Offset : NPUReg60XX_RegisterWrapper<"halo_1_offset", 4, 0xCC, ["halo_1_mm_adr_offset"]>;

// halo_2_offset
def bf_Halo2MmAdrOffset : NPUReg60XX_RegFieldWrapper<"halo_2_mm_adr_offset", 21, 0>;
def reg_Halo2Offset : NPUReg60XX_RegisterWrapper<"halo_2_offset", 4, 0xD0, ["halo_2_mm_adr_offset"]>;

// halo_3_offset
def bf_Halo3MmAdrOffset : NPUReg60XX_RegFieldWrapper<"halo_3_mm_adr_offset", 21, 0>;
def reg_Halo3Offset : NPUReg60XX_RegisterWrapper<"halo_3_offset", 4, 0xD4, ["halo_3_mm_adr_offset"]>;

// halo_4_offset
def bf_Halo4MmAdrOffset : NPUReg60XX_RegFieldWrapper<"halo_4_mm_adr_offset", 21, 0>;
def reg_Halo4Offset : NPUReg60XX_RegisterWrapper<"halo_4_offset", 4, 0xD8, ["halo_4_mm_adr_offset"]>;

// unused 3
def bf_Unused3Invariant : NPUReg60XX_RegFieldWrapper<"unused3_inv", 32, 0, "UINT">;
def reg_Unused3Invariant : NPUReg60XX_RegisterWrapper<"unused3_inv", 4, 0xDC, ["unused3_inv"]>;


def bf_BarrierWaitMaskHi : NPUReg60XX_RegFieldWrapper<"barriers_wait_mask_hi_", 32, 0>;
def reg_BarrierWaitMaskHi : NPUReg60XX_RegisterWrapper<"barriers_wait_mask_hi_", 8, 0xE0, ["barriers_wait_mask_hi_"]>;

def bf_BarrierWaitMaskLo : NPUReg60XX_RegFieldWrapper<"barriers_wait_mask_lo_", 64, 0>;
def reg_BarrierWaitMaskLo : NPUReg60XX_RegisterWrapper<"barriers_wait_mask_lo_", 8, 0xE8, ["barriers_wait_mask_lo_"]>;

def bf_BarrierWaitPostHi : NPUReg60XX_RegFieldWrapper<"barriers_post_mask_hi_", 32, 0>;
def reg_BarrierWaitPostHi : NPUReg60XX_RegisterWrapper<"barriers_post_mask_hi_", 8, 0xF0, ["barriers_post_mask_hi_"]>;

def bf_BarrierWaitPostLo : NPUReg60XX_RegFieldWrapper<"barriers_post_mask_lo_", 64, 0>;
def reg_BarrierWaitPostLo : NPUReg60XX_RegisterWrapper<"barriers_post_mask_lo_", 8, 0xF8, ["barriers_post_mask_lo_"]>;

def bf_BarrierGroup : NPUReg60XX_RegFieldWrapper<"group_", 8, 0>;
def bf_BarrierMask : NPUReg60XX_RegFieldWrapper<"mask_", 8, 8>;
def reg_BarrierGroupMask : NPUReg60XX_RegisterWrapper<"barriers_group_mask_", 8, 0x100, ["group_", "mask_"]>;

def bf_StartAfter : NPUReg60XX_RegFieldWrapper<"start_after_", 32, 0>;
def bf_CleanAfter : NPUReg60XX_RegFieldWrapper<"clean_after_", 32, 32>;
def reg_BarriersSchedulingCfg : NPUReg60XX_RegisterWrapper<"barriers_sched_", 8, 0x108, ["start_after_", "clean_after_"]>;

def bf_Deprecated0Invariant : NPUReg60XX_RegFieldWrapper<"deprecated0_inv", 64, 0, "UINT">;
def reg_Deprecated0Invariant : NPUReg60XX_RegisterWrapper<"deprecated0_inv", 8, 0x110, ["deprecated0_inv"]>;

def bf_VariantCnt : NPUReg60XX_RegFieldWrapper<"variant_count_", 16, 0>;
def reg_VariantCnt : NPUReg60XX_RegisterWrapper<"variant_count_", 2, 0x118, ["variant_count_"]>;

def bf_ClusterInvariant : NPUReg60XX_RegFieldWrapper<"cluster_invariant_", 8, 0>;
def reg_ClusterInvariant : NPUReg60XX_RegisterWrapper<"cluster_invariant_", 1, 0x11A, ["cluster_invariant_"]>;

def bf_Deprecated1Invariant : NPUReg60XX_RegFieldWrapper<"deprecated1_inv", 16, 0, "UINT">;
def reg_Deprecated1Invariant : NPUReg60XX_RegisterWrapper<"deprecated1_inv", 2, 0x11B, ["deprecated1_inv"]>;

def bf_Pad3 : NPUReg60XX_RegFieldWrapper<"pad_3", 24, 0, "UINT">;
def reg_Pad3 : NPUReg60XX_RegisterWrapper<"pad_3", 3, 0x11D, ["pad_3"]>;

//
// DPUVariantRegisters
//

// invar_ptr
def bf_InvarPtr : NPUReg60XX_RegFieldWrapper<"invar_ptr", 16, 0>;
def bf_VarTag : NPUReg60XX_RegFieldWrapper<"var_tag", 16, 16>;
def reg_InvarPtr : NPUReg60XX_RegisterWrapper<"invar_ptr", 4, 0x0, ["invar_ptr", "var_tag"]>;

// workload_size0
def bf_WorkloadSizeX : NPUReg60XX_RegFieldWrapper<"workload_size_x", 14, 0>;
def bf_WorkloadSizeY : NPUReg60XX_RegFieldWrapper<"workload_size_y", 14, 14>;
def reg_WorkloadSize0 : NPUReg60XX_RegisterWrapper<"workload_size0", 4, 0x4, ["workload_size_x", "workload_size_y"]>;

// workload_size1
def bf_WorkloadSizeZ : NPUReg60XX_RegFieldWrapper<"workload_size_z", 14, 0>;
def bf_PadCountUp : NPUReg60XX_RegFieldWrapper<"pad_count_up", 3, 14>;
def bf_PadCountLeft : NPUReg60XX_RegFieldWrapper<"pad_count_left", 3, 17>;
def bf_PadCountDown : NPUReg60XX_RegFieldWrapper<"pad_count_down", 3, 20>;
def bf_PadCountRight : NPUReg60XX_RegFieldWrapper<"pad_count_right", 3, 23>;
def bf_HaloEnableNewDesc : NPUReg60XX_RegFieldWrapper<"halo_enable_new_desc", 6, 26>;
def reg_WorkloadSize1 : NPUReg60XX_RegisterWrapper<"workload_size1", 4, 0x8,
    ["workload_size_z", "pad_count_up", "pad_count_left", "pad_count_down", "pad_count_right", "halo_enable_new_desc"]>;

// workload_start0
def bf_WorkloadStartX : NPUReg60XX_RegFieldWrapper<"workload_start_x", 14, 0>;
def bf_WorkloadStartY : NPUReg60XX_RegFieldWrapper<"workload_start_y", 14, 14>;
def reg_WorkloadStart0 : NPUReg60XX_RegisterWrapper<"workload_start0", 4, 0xC, ["workload_start_x", "workload_start_y"]>;

// workload_start1
def bf_WorkloadStartZ : NPUReg60XX_RegFieldWrapper<"workload_start_z", 14, 0>;
def bf_NewInvarFormat : NPUReg60XX_RegFieldWrapper<"new_invar_format", 1, 14>;
def bf_NewVarFormat : NPUReg60XX_RegFieldWrapper<"new_var_format", 1, 15>;
def bf_WeightDPtrStart : NPUReg60XX_RegFieldWrapper<"weight_d_ptr_start", 16, 16>;
def reg_WorkloadStart1 : NPUReg60XX_RegisterWrapper<"workload_start1", 4, 0x10, ["workload_start_z", "new_invar_format", "new_var_format", "weight_d_ptr_start"]>;

// offset_addr
def bf_NthwNtk : NPUReg60XX_RegFieldWrapper<"nthw_ntk", 2, 0>;
def bf_BinCfg : NPUReg60XX_RegFieldWrapper<"bin_cfg", 1, 2>;
def bf_ConvCond : NPUReg60XX_RegFieldWrapper<"conv_cond", 1, 3>;
def bf_DenseSe : NPUReg60XX_RegFieldWrapper<"dense_se", 1, 4>;
def bf_CimClkEn : NPUReg60XX_RegFieldWrapper<"cim_clk_en", 1, 5>;
def bf_SwizzleKeyOffsetAddr : NPUReg60XX_RegFieldWrapper<"swizzle_key_offset", 3, 6>;
def bf_IduMrmClkEn : NPUReg60XX_RegFieldWrapper<"idu_mrm_clk_en", 1, 9>;
def bf_OduClkEn : NPUReg60XX_RegFieldWrapper<"odu_clk_en", 1, 10>;
def bf_MpeClkEn : NPUReg60XX_RegFieldWrapper<"mpe_clk_en", 1, 11>;
def bf_PpeClkEn : NPUReg60XX_RegFieldWrapper<"ppe_clk_en", 1, 12>;
def bf_OduStatEn : NPUReg60XX_RegFieldWrapper<"odu_stat_en", 1, 13>;
def bf_IduStatEn : NPUReg60XX_RegFieldWrapper<"idu_stat_en", 1, 14>;
def bf_NocClkEn : NPUReg60XX_RegFieldWrapper<"noc_clk_en", 1, 15>;
def bf_OduStatClrMode : NPUReg60XX_RegFieldWrapper<"odu_stat_clr_mode", 1, 16>;
def bf_IduStatClrMode : NPUReg60XX_RegFieldWrapper<"idu_stat_clr_mode", 1, 17>;
def bf_SeOnlyEn : NPUReg60XX_RegFieldWrapper<"se_only_en", 1, 18>;
def bf_ShaveL2CacheEn : NPUReg60XX_RegFieldWrapper<"shave_l2_cache_en", 1, 19>;
def bf_IduDbgEn : NPUReg60XX_RegFieldWrapper<"idu_dbg_en", 2, 20>;
def bf_SbReadEn : NPUReg60XX_RegFieldWrapper<"sb_read_en", 1, 22>;
def bf_Tensor2ActDense : NPUReg60XX_RegFieldWrapper<"tensor2_act_dense", 1, 23>;
def bf_GrpConvEn : NPUReg60XX_RegFieldWrapper<"grp_conv_en", 1, 24>;
def bf_ZeroPoint4b : NPUReg60XX_RegFieldWrapper<"zero_point_4b", 1, 25>;
def bf_Reserved3 : NPUReg60XX_RegFieldWrapper<"reserved_3", 1, 26>;
def bf_WtSwizzleKey : NPUReg60XX_RegFieldWrapper<"wt_swizzle_key", 3, 27>;
def bf_WtSwizzleSel : NPUReg60XX_RegFieldWrapper<"wt_swizzle_sel", 1, 30>;
//  Note: wt_swizzle_sel needs to be on by default to match NNRT GF behaviour
def bf_GifClkEn : NPUReg60XX_RegFieldWrapper<"gif_clk_en", 1, 31>;
def reg_OffsetAddr : NPUReg60XX_RegisterWrapper<"offset_addr", 4, 0x14,
    ["nthw_ntk", "bin_cfg", "conv_cond", "dense_se", "cim_clk_en", "swizzle_key_offset", "idu_mrm_clk_en", "odu_clk_en",
    "mpe_clk_en", "ppe_clk_en", "odu_stat_en", "idu_stat_en", "noc_clk_en", "odu_stat_clr_mode", "idu_stat_clr_mode", "se_only_en",
    "shave_l2_cache_en", "idu_dbg_en", "sb_read_en", "tensor2_act_dense", "grp_conv_en", "zero_point_4b", "reserved_3", "wt_swizzle_key",
    "wt_swizzle_sel", "gif_clk_en"], 0x40000000>;

// hwp_wload_id
def bf_HwpWLoadId : NPUReg60XX_RegFieldWrapper<"hwp_wload_id", 16, 0>;
def reg_HwpWloadId : NPUReg60XX_RegisterWrapper<"hwp_wload_id", 4, 0x18, ["hwp_wload_id"]>;

// var_cfg
def bf_InvarPltRdEn : NPUReg60XX_RegFieldWrapper<"invar_plt_rd_en", 1, 6>;
def bf_PltPtrForce : NPUReg60XX_RegFieldWrapper<"plt_ptr_force", 1, 7>;
def bf_InvarLutRdEn : NPUReg60XX_RegFieldWrapper<"invar_lut_rd_en", 1, 8>;
def bf_InvarLineCntEn : NPUReg60XX_RegFieldWrapper<"invar_line_cnt_en", 1, 9>;
def bf_InvarLineCnt : NPUReg60XX_RegFieldWrapper<"invar_line_cnt", 4, 10>;
def bf_InvarPtrForce : NPUReg60XX_RegFieldWrapper<"invar_lptr_force", 1, 14>;
def bf_NextSramJobValid : NPUReg60XX_RegFieldWrapper<"next_sram_job_valid", 1, 15>;
def bf_NextSramJobAddr : NPUReg60XX_RegFieldWrapper<"next_sram_job_addr", 16, 16>;
def reg_VarCfg : NPUReg60XX_RegisterWrapper<"var_cfg", 4, 0x1C,
    ["invar_plt_rd_en", "plt_ptr_force", "invar_lut_rd_en", "invar_line_cnt_en", "invar_line_cnt", "invar_lptr_force", "next_sram_job_valid", "next_sram_job_addr"]>;

// cbarrier_lo
def bf_CBarrierLo : NPUReg60XX_RegFieldWrapper<"cbarrier_lo", 64, 0>;
def reg_CBarrierLo : NPUReg60XX_RegisterWrapper<"cbarrier_lo", 8, 0x20, ["cbarrier_lo"]>;

// pbarrier_lo
def bf_PBarrierLo : NPUReg60XX_RegFieldWrapper<"pbarrier_lo", 64, 0>;
def reg_PBarrierLo : NPUReg60XX_RegisterWrapper<"pbarrier_lo", 8, 0x28, ["pbarrier_lo"]>;

// dpu_cfg
def bf_WorkloadStartOdu : NPUReg60XX_RegFieldWrapper<"workload_start_odu", 1, 0>;
def bf_WorkloadStartIdu : NPUReg60XX_RegFieldWrapper<"workload_start_idu", 1, 1>;
def bf_WorkloadPrmSel : NPUReg60XX_RegFieldWrapper<"workload_prm_sel", 1, 2>;
def bf_WorkloadValid : NPUReg60XX_RegFieldWrapper<"workload_valid", 1, 3>;
def bf_WorkloadShadOdu : NPUReg60XX_RegFieldWrapper<"workload_shad_odu", 1, 4>;
def bf_WorkloadShadIdu : NPUReg60XX_RegFieldWrapper<"workload_shad_idu", 1, 5>;
def bf_WorkloadIduAutoUpd0 : NPUReg60XX_RegFieldWrapper<"workload_idu_auto_upd_0", 1, 6>;
def bf_WorkloadIduAutoUpd1 : NPUReg60XX_RegFieldWrapper<"workload_idu_auto_upd_1", 1, 7>;
def bf_WorkloadOduAutoUpd : NPUReg60XX_RegFieldWrapper<"workload_odu_auto_upd", 1, 8>;
def bf_CfgReserved0 : NPUReg60XX_RegFieldWrapper<"cfg_Reserved_2", 1, 9>;
def bf_PreemptCcEn : NPUReg60XX_RegFieldWrapper<"preempt_cc_en", 1, 10>;
def bf_BarWaitPreemptEn : NPUReg60XX_RegFieldWrapper<"bar_wait_preempt_en", 1, 11>;
def bf_RstCtxtNew : NPUReg60XX_RegFieldWrapper<"rst_ctxt_new", 1, 12>;
def bf_CfgReserved3 : NPUReg60XX_RegFieldWrapper<"cfg_Reserved_3", 1, 13>;
def bf_CfgReserved4 : NPUReg60XX_RegFieldWrapper<"cfg_Reserved_4", 1, 14>;
def bf_OduStatClr : NPUReg60XX_RegFieldWrapper<"odu_stat_clr", 1, 15>;
def bf_IduStatClr : NPUReg60XX_RegFieldWrapper<"idu_stat_clr", 1, 16>;
def bf_CfgReserved5 : NPUReg60XX_RegFieldWrapper<"cfg_Reserved_5", 1, 17>;
def bf_CfgReserved6 : NPUReg60XX_RegFieldWrapper<"cfg_Reserved_6", 14, 18>;
def reg_DpuCfg : NPUReg60XX_RegisterWrapper<"dpu_cfg", 4, 0x30,
    ["workload_start_odu", "workload_start_idu", "workload_prm_sel", "workload_valid", "workload_shad_odu",
    "workload_shad_idu", "workload_idu_auto_upd_0", "workload_idu_auto_upd_1", "workload_odu_auto_upd",
    "cfg_Reserved_2", "preempt_cc_en", "bar_wait_preempt_en", "rst_ctxt_new", "cfg_Reserved_3", "cfg_Reserved_4",
    "odu_stat_clr", "idu_stat_clr", "cfg_Reserved_5", "cfg_Reserved_6"], 0x00000004>;

// te_beg0
def bf_TeBegY : NPUReg60XX_RegFieldWrapper<"te_beg_y", 13, 0>;
def bf_TeBegZ : NPUReg60XX_RegFieldWrapper<"te_beg_z", 13, 13>;
def reg_TeBeg0 : NPUReg60XX_RegisterWrapper<"te_beg0", 4, 0x34, ["te_beg_y", "te_beg_z"]>;

// te_beg1
def bf_TeBegX : NPUReg60XX_RegFieldWrapper<"te_beg_x", 13, 0>;
def bf_DynPrimitiveIdVar : NPUReg60XX_RegFieldWrapper<"dyn_primitive_id_var", 10, 13>;
def bf_DynPrimitiveIdVarEn : NPUReg60XX_RegFieldWrapper<"dyn_primitive_id_var_en", 1, 23>;
def bf_WtSizeMaxEn : NPUReg60XX_RegFieldWrapper<"wt_size_max_en", 1, 24>;
def reg_TeBeg1 : NPUReg60XX_RegisterWrapper<"te_beg1", 4, 0x38, ["te_beg_x", "dyn_primitive_id_var", "dyn_primitive_id_var_en", "wt_size_max_en"]>;

// te_end0
def bf_TeEndY : NPUReg60XX_RegFieldWrapper<"te_end_y", 13, 0>;
def bf_TeEndZ : NPUReg60XX_RegFieldWrapper<"te_end_z", 13, 13>;
def reg_TeEnd0 : NPUReg60XX_RegisterWrapper<"te_end0", 4, 0x3C, ["te_end_y", "te_end_z"]>;

// te_end1
def bf_TeEndX : NPUReg60XX_RegFieldWrapper<"te_end_x", 13, 0>;
def reg_TeEnd1 : NPUReg60XX_RegisterWrapper<"te_end1", 4, 0x40, ["te_end_x"]>;

// weight_size
def bf_WeightSize : NPUReg60XX_RegFieldWrapper<"weight_size", 21, 0>;
def bf_ScopeIdEn : NPUReg60XX_RegFieldWrapper<"scope_id_en", 1, 21>;
def bf_DynPrimitiveId2Var : NPUReg60XX_RegFieldWrapper<"dyn_primitive_id2_var", 10, 22>;
def reg_WeightSize : NPUReg60XX_RegisterWrapper<"weight_size", 4, 0x44, ["weight_size", "scope_id_en", "dyn_primitive_id2_var"]>;

// weight_num
def bf_WeighNum : NPUReg60XX_RegFieldWrapper<"weight_num", 14, 0>;
def bf_WeightSpPtrStart : NPUReg60XX_RegFieldWrapper<"weight_sp_ptr_start", 16, 16>;
def reg_WeightNum : NPUReg60XX_RegisterWrapper<"weight_num", 4, 0x48, ["weight_num", "weight_sp_ptr_start"]>;

// weight_start
def bf_WeightStart : NPUReg60XX_RegFieldWrapper<"weight_start", 32, 0>;
def reg_WeightStart : NPUReg60XX_RegisterWrapper<"weight_start", 4, 0x4C, ["weight_start"]>;

// cbarrier_hi
def bf_CBarrierHi : NPUReg60XX_RegFieldWrapper<"cbarrier_hi", 32, 0>;
def reg_CBarrierHi : NPUReg60XX_RegisterWrapper<"cbarrier_hi", 8, 0x50, ["cbarrier_hi"]>;

// pbarrier_hi
def bf_PBarrierHi : NPUReg60XX_RegFieldWrapper<"pbarrier_hi", 32, 0>;
def reg_PBarrierHi : NPUReg60XX_RegisterWrapper<"pbarrier_hi", 8, 0x58, ["pbarrier_hi"]>;

// halo_region0..5A
def bf_SpAdrOffset : NPUReg60XX_RegFieldWrapper<"sp_adr_offset", 22, 0, "SINT">;
def bf_TileSelected : NPUReg60XX_RegFieldWrapper<"tile_select", 7, 22>;
def bf_RsvdA : NPUReg60XX_RegFieldWrapper<"rsvdA", 2, 29>;
def bf_Enable : NPUReg60XX_RegFieldWrapper<"enable", 1, 31>;
def reg_HaloRegion0A : NPUReg60XX_RegisterWrapper<"halo_region0A", 4, 0x60, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion1A : NPUReg60XX_RegisterWrapper<"halo_region1A", 4, 0x70, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion2A : NPUReg60XX_RegisterWrapper<"halo_region2A", 4, 0x80, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion3A : NPUReg60XX_RegisterWrapper<"halo_region3A", 4, 0x90, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion4A : NPUReg60XX_RegisterWrapper<"halo_region4A", 4, 0xA0, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion5A : NPUReg60XX_RegisterWrapper<"halo_region5A", 4, 0xB0, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;

// halo_region0..5B
def bf_AcAdrOffset : NPUReg60XX_RegFieldWrapper<"ac_adr_offset", 22, 0, "SINT">;
def bf_TargetWidthLsb : NPUReg60XX_RegFieldWrapper<"target_width_lsb", 10, 22>;
def reg_HaloRegion0B : NPUReg60XX_RegisterWrapper<"halo_region0B", 4, 0x64, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion1B : NPUReg60XX_RegisterWrapper<"halo_region1B", 4, 0x74, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion2B : NPUReg60XX_RegisterWrapper<"halo_region2B", 4, 0x84, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion3B : NPUReg60XX_RegisterWrapper<"halo_region3B", 4, 0x94, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion4B : NPUReg60XX_RegisterWrapper<"halo_region4B", 4, 0xA4, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion5B : NPUReg60XX_RegisterWrapper<"halo_region5B", 4, 0xB4, ["ac_adr_offset", "target_width_lsb"]>;

// halo_region0..5C
def bf_BeginX : NPUReg60XX_RegFieldWrapper<"begin_x", 13, 0>;
def bf_BeginY : NPUReg60XX_RegFieldWrapper<"begin_y", 13, 13>;
def bf_TargetWidthMsb : NPUReg60XX_RegFieldWrapper<"target_width_msb", 4, 26>;
def reg_HaloRegion0C : NPUReg60XX_RegisterWrapper<"halo_region0C", 4, 0x68, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion1C : NPUReg60XX_RegisterWrapper<"halo_region1C", 4, 0x78, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion2C : NPUReg60XX_RegisterWrapper<"halo_region2C", 4, 0x88, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion3C : NPUReg60XX_RegisterWrapper<"halo_region3C", 4, 0x98, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion4C : NPUReg60XX_RegisterWrapper<"halo_region4C", 4, 0xA8, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion5C : NPUReg60XX_RegisterWrapper<"halo_region5C", 4, 0xB8, ["begin_x", "begin_y", "target_width_msb"]>;

// halo_region0..5D
def bf_EndX : NPUReg60XX_RegFieldWrapper<"end_x", 13, 0>;
def bf_EndY : NPUReg60XX_RegFieldWrapper<"end_y", 13, 13>;
def bf_RsvdD : NPUReg60XX_RegFieldWrapper<"rsvdD", 4, 26>;
def reg_HaloRegion0D : NPUReg60XX_RegisterWrapper<"halo_region0D", 4, 0x6C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion1D : NPUReg60XX_RegisterWrapper<"halo_region1D", 4, 0x7C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion2D : NPUReg60XX_RegisterWrapper<"halo_region2D", 4, 0x8C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion3D : NPUReg60XX_RegisterWrapper<"halo_region3D", 4, 0x9C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion4D : NPUReg60XX_RegisterWrapper<"halo_region4D", 4, 0xAC, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion5D : NPUReg60XX_RegisterWrapper<"halo_region5D", 4, 0xBC, ["end_x", "end_y", "rsvdD"]>;

// unused 0
def bf_Unused0Variant : NPUReg60XX_RegFieldWrapper<"unused0_var", 64, 0>;
def reg_Unused0Variant : NPUReg60XX_RegisterWrapper<"unused0_var", 8, 0xC0, ["unused0_var"]>;

// unused 1
def bf_Unused1Variant : NPUReg60XX_RegFieldWrapper<"unused1_var", 64, 0>;
def reg_Unused1Variant : NPUReg60XX_RegisterWrapper<"unused1_var", 8, 0xC8, ["unused1_var"]>;

// unused 2
def bf_Unused2Variant : NPUReg60XX_RegFieldWrapper<"unused2_var", 64, 0>;
def reg_Unused2Variant : NPUReg60XX_RegisterWrapper<"unused2_var", 8, 0xD0, ["unused2_var"]>;

// unused 3
def bf_Unused3Variant : NPUReg60XX_RegFieldWrapper<"unused3_var", 64, 0>;
def reg_Unused3Variant : NPUReg60XX_RegisterWrapper<"unused3_var", 8, 0xD8, ["unused3_var"]>;

def bf_Invariant : NPUReg60XX_RegFieldWrapper<"invariant_", 64, 0>;
def reg_Invariant : NPUReg60XX_RegisterWrapper<"invariant_", 8, 0xE0, ["invariant_"]>;

def bf_InvariantIndex : NPUReg60XX_RegFieldWrapper<"invariant_index_", 32, 0>;
def reg_InvariantIndex : NPUReg60XX_RegisterWrapper<"invariant_index_", 4, 0xE8, ["invariant_index_"]>;

// uint8_t deprecated_[13]
def bf_Deprecated0Variant : NPUReg60XX_RegFieldWrapper<"deprecated0_var", 64, 0, "UINT">;
def reg_Deprecated0Variant : NPUReg60XX_RegisterWrapper<"deprecated0_var", 8, 0xEC, ["deprecated0_var"]>;

def bf_Deprecated1Variant : NPUReg60XX_RegFieldWrapper<"deprecated1_var", 40, 0, "UINT">;
def reg_Deprecated1Variant : NPUReg60XX_RegisterWrapper<"deprecated1_var", 5, 0xF4, ["deprecated1_var"]>;

// uint8_t pad_[7]
def bf_Pad_7_1 : NPUReg60XX_RegFieldWrapper<"pad_7_1", 56, 0, "UINT">;
def reg_Pad_7_1 : NPUReg60XX_RegisterWrapper<"pad_7_1", 7, 0xF9, ["pad_7_1"]>;


//
// RegMapped Specialization
//

class NPUReg60XX_RegMappedWrapper<string name = "", list<string> regsTypes = [], bit canBeEmpty = 0> :
    VPURegMapped_RegMappedWrapper<NPUReg60XX_Dialect, "vpux::NPUReg60XX::", name, regsTypes, canBeEmpty> {

}

// DPUInvariantRegister
def reg_mapped_DpuInvariantRegister : NPUReg60XX_RegMappedWrapper<"DpuInvariantRegister",
    ["se_addr", "sparsity_addr", "se_size", "z_config", "kernel_pad_cfg", "tensor_size0", "tensor_size1", "tensor_start", "tensor_mode",
    "elops_sparsity_addr", "elops_se_addr", "elops_wload", "act_offset0", "base_offset_a", "base_offset_b", "wt_offset", "odu_cfg", "odu_cfg_2", "odu_mn_base", "odu_mx_base",
    "te_dim0", "te_dim1", "mt_base", "sp_base", "mpe_cfg", "mpe_bus_data_sel", "elop_scale", "ppe_cfg", "unused0_inv", "unused1_inv", "ppe_scale_ctrl",
    "unused2_inv", "ppe_scale_hclamp", "ppe_scale_lclamp", "ppe_misc", "ppe_fp_bias", "ppe_fp_scale", "ppe_fp_prelu",
    "ppe_fp_cfg", "odu_ac_base", "hwp_ctrl", "hwp_cmx_mem_addr", "tensor2_start", "ppe_lut_ptr", "nvar_tag",
    "dyn_prim_cfg", "dyn_scale", "dyn_ppe_data",
    "odu_local_a", "odu_local_b", "halo_0_offset", "halo_1_offset", "halo_2_offset", "halo_3_offset", "halo_4_offset", "unused3_inv",
    "barriers_wait_mask_hi_", "barriers_wait_mask_lo_", "barriers_post_mask_hi_", "barriers_post_mask_lo_", "barriers_group_mask_",
    "barriers_sched_", "deprecated0_inv", "variant_count_", "cluster_invariant_", "deprecated1_inv", "pad_3"
    ]>;

// DPUVariantRegister
def reg_mapped_DpuVariantRegister : NPUReg60XX_RegMappedWrapper<"DpuVariantRegister",
    ["invar_ptr", "workload_size0", "workload_size1", "workload_start0", "workload_start1", "offset_addr", "hwp_wload_id",
    "var_cfg", "cbarrier_lo", "pbarrier_lo", "dpu_cfg", "te_beg0", "te_beg1", "te_end0", "te_end1",
    "weight_size", "weight_num", "weight_start", "cbarrier_hi", "pbarrier_hi",
    "halo_region0A", "halo_region0B", "halo_region0C", "halo_region0D",
    "halo_region1A", "halo_region1B", "halo_region1C", "halo_region1D",
    "halo_region2A", "halo_region2B", "halo_region2C", "halo_region2D",
    "halo_region3A", "halo_region3B", "halo_region3C", "halo_region3D",
    "halo_region4A", "halo_region4B", "halo_region4C", "halo_region4D",
    "halo_region5A", "halo_region5B", "halo_region5C", "halo_region5D",
    "unused0_var", "unused1_var", "unused2_var", "unused3_var",
    "invariant_", "invariant_index_", "deprecated0_var", "deprecated1_var", "pad_7_1"
    ]>;

// DMADescriptor

//
//  dma_64_bit_word_0
//
def bf_dma_watermark : NPUReg60XX_RegFieldWrapper<"dma_watermark", 1, 0>;                                  // Watermark to indicate that the job has completed
def reg_dma_watermark : NPUReg60XX_RegisterWrapper<"dma_watermark", 1, 0x0, ["dma_watermark"]>;

def bf_dma_link_address : NPUReg60XX_RegFieldWrapper<"dma_link_address", 48, 0>;                           // Pointer to the next element in linked list
def reg_dma_link_address : NPUReg60XX_RegisterWrapper<"dma_link_address", 8, 0x0, ["dma_link_address"]>;

def bf_dma_lra : NPUReg60XX_RegFieldWrapper<"dma_lra", 1, 63>;                                             // Link Relative Address. Base address fetched from LBA_ADDR
def reg_dma_lra : NPUReg60XX_RegisterWrapper<"dma_lra", 8, 0x0, ["dma_lra"]>;

//
//  dma_64_bit_word_1
//

def bf_dma_lba_addr : NPUReg60XX_RegFieldWrapper<"dma_lba_addr", 32, 0>;       // CMX address to the location of the Base Address for Relative Addressing options
def reg_dma_lba_addr : NPUReg60XX_RegisterWrapper<"dma_lba_addr", 4, 0x8,     // 8 dec
    ["dma_lba_addr"]>;

def bf_dma_src_aub : NPUReg60XX_RegFieldWrapper<"dma_src_aub", 8, 0>;          // Source AXI user bits
def reg_dma_src_aub : NPUReg60XX_RegisterWrapper<"dma_src_aub", 1, 0xC,        // 12 dec
    ["dma_src_aub"]>;

def bf_dma_dst_aub : NPUReg60XX_RegFieldWrapper<"dma_dst_aub", 8, 0>;          // Destination AXI user bits
def reg_dma_dst_aub : NPUReg60XX_RegisterWrapper<"dma_dst_aub", 1, 0xE,        // 14 dec
    ["dma_dst_aub"]>;

//
//  dma_64_bit_word_2
//

def bf_dma_cfg_fields_num_dim : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_num_dim", 3, 0>;
def bf_dma_cfg_fields_int_en : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_int_en", 1, 3>;
def bf_dma_cfg_fields_int_id : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_int_id", 8, 4>;
def bf_dma_cfg_fields_src_burst_length : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_src_burst_length", 4, 12>;
def bf_dma_cfg_fields_dst_burst_length : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_dst_burst_length", 4, 16>;
def bf_dma_cfg_fields_arb_qos : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_arb_qos", 8, 20>;
def bf_dma_cfg_fields_ord : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_ord", 1, 28>;
def bf_dma_cfg_fields_barrier_en : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_barrier_en", 1, 29>;
def bf_dma_cfg_fields_memset_en : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_memset_en", 1, 30>;
def bf_dma_cfg_fields_atp_en : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_atp_en", 1, 31>;
def bf_dma_cfg_fields_watermark_en : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_watermark_en", 1, 32>;
def bf_dma_cfg_fields_rwf_en : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_rwf_en", 1, 33>;
def bf_dma_cfg_fields_rws_en : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_rws_en", 1, 34>;
def bf_dma_cfg_fields_src_list_cfg : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_src_list_cfg", 2, 35>;
def bf_dma_cfg_fields_dst_list_cfg : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_dst_list_cfg", 2, 37>;
def bf_dma_cfg_fields_conversion_cfg : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_conversion_cfg", 3, 39>;
def bf_dma_cfg_fields_acceleration_cfg : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_acceleration_cfg", 2, 42>;
def bf_dma_cfg_fields_tile4_cfg : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_tile4_cfg", 2, 44>;
def bf_dma_cfg_fields_axi_user_bits_cfg : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_axi_user_bits_cfg", 2, 46>;
def bf_dma_cfg_fields_hwp_id_en : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_hwp_id_en", 1, 48>;
def bf_dma_cfg_fields_hwp_id : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_hwp_id", 12, 49>;
def bf_dma_cfg_fields_hwp_skip : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_hwp_skip", 1, 61>;
def bf_dma_cfg_fields_reserved : NPUReg60XX_RegFieldWrapper<"dma_cfg_fields_reserved", 2, 62>;

def reg_dma_cfg_fields : NPUReg60XX_RegisterWrapper<"dma_cfg_fields", 8, 0x10,  // 16 dec
    ["dma_cfg_fields_num_dim", "dma_cfg_fields_int_en", "dma_cfg_fields_int_id", "dma_cfg_fields_src_burst_length", "dma_cfg_fields_dst_burst_length",
    "dma_cfg_fields_arb_qos", "dma_cfg_fields_ord", "dma_cfg_fields_barrier_en", "dma_cfg_fields_memset_en", "dma_cfg_fields_atp_en",
    "dma_cfg_fields_watermark_en", "dma_cfg_fields_rwf_en", "dma_cfg_fields_rws_en", "dma_cfg_fields_src_list_cfg", "dma_cfg_fields_dst_list_cfg",
    "dma_cfg_fields_conversion_cfg", "dma_cfg_fields_acceleration_cfg", "dma_cfg_fields_tile4_cfg", "dma_cfg_fields_axi_user_bits_cfg",
    "dma_cfg_fields_hwp_id_en", "dma_cfg_fields_hwp_id", "dma_cfg_fields_hwp_skip", "dma_cfg_fields_reserved"]>;

//
//  dma_64_bit_word_3
//

def bf_dma_remote_width_fetch : NPUReg60XX_RegFieldWrapper<"dma_remote_width_fetch", 32, 0>;
def reg_dma_remote_width_fetch : NPUReg60XX_RegisterWrapper<"dma_remote_width_fetch", 4, 0x18, // 24 dec
    ["dma_remote_width_fetch"]>;

def bf_dma_width_src : NPUReg60XX_RegFieldWrapper<"dma_width_src", 32, 0>;
def bf_dma_width_dst : NPUReg60XX_RegFieldWrapper<"dma_width_dst", 32, 32>;

// overlap because fields dma_src_width and dma_rwf_addr are placed at the same memory address
def reg_dma_width : NPUReg60XX_RegisterWrapper<"dma_width", 8, 0x18, // 24 dec
    ["dma_width_src", "dma_width_dst"]>;

//
//  dma_64_bit_word_4: compress, decompress and WPREP regs are placed at the same memory address
//

//
//  dma_64_bit_word_4_compress
//

def bf_dma_acc_info_compress_dtype : NPUReg60XX_RegFieldWrapper<"dma_acc_info_compress_dtype", 2, 0>;
def bf_dma_acc_info_compress_reserved1 : NPUReg60XX_RegFieldWrapper<"dma_acc_info_compress_reserved1", 1, 2>;
def bf_dma_acc_info_compress_sparse : NPUReg60XX_RegFieldWrapper<"dma_acc_info_compress_sparse", 1, 3>;
def bf_dma_acc_info_compress_bitc_en : NPUReg60XX_RegFieldWrapper<"dma_acc_info_compress_bitc_en", 1, 4>;
def bf_dma_acc_info_compress_z : NPUReg60XX_RegFieldWrapper<"dma_acc_info_compress_z", 10, 5>;
def bf_dma_acc_info_compress_bitmap_buf_sz : NPUReg60XX_RegFieldWrapper<"dma_acc_info_compress_bitmap_buf_sz", 19, 15>;
def bf_dma_acc_info_compress_reserved2 : NPUReg60XX_RegFieldWrapper<"dma_acc_info_compress_reserved2", 3, 34>;
def bf_dma_acc_info_compress_bitmap_base_addr : NPUReg60XX_RegFieldWrapper<"dma_acc_info_compress_bitmap_base_addr", 27, 37>;

def reg_dma_acc_info_compress : NPUReg60XX_RegisterWrapper<"dma_acc_info_compress", 8, 0x20, // 32 dec
    ["dma_acc_info_compress_dtype", "dma_acc_info_compress_reserved1", "dma_acc_info_compress_sparse",
    "dma_acc_info_compress_bitc_en", "dma_acc_info_compress_z", "dma_acc_info_compress_bitmap_buf_sz",
    "dma_acc_info_compress_reserved2", "dma_acc_info_compress_bitmap_base_addr"]>;

//
//  dma_64_bit_word_4_decompress
//

def bf_dma_acc_info_decompress_dtype : NPUReg60XX_RegFieldWrapper<"dma_acc_info_decompress_dtype", 2, 0>;
def bf_dma_acc_info_decompress_reserved1 : NPUReg60XX_RegFieldWrapper<"dma_acc_info_decompress_reserved1", 1, 2>;
def bf_dma_acc_info_decompress_sparse : NPUReg60XX_RegFieldWrapper<"dma_acc_info_decompress_sparse", 1, 3>;
def bf_dma_acc_info_decompress_bitc_en : NPUReg60XX_RegFieldWrapper<"dma_acc_info_decompress_bitc_en", 1, 4>;
def bf_dma_acc_info_decompress_z : NPUReg60XX_RegFieldWrapper<"dma_acc_info_decompress_z", 10, 5>;
def bf_dma_acc_info_decompress_bitmap_buf_sz : NPUReg60XX_RegFieldWrapper<"dma_acc_info_decompress_bitmap_buf_sz", 19, 15>;
def bf_dma_acc_info_decompress_reserved2 : NPUReg60XX_RegFieldWrapper<"dma_acc_info_decompress_reserved2", 3, 34>;
def bf_dma_acc_info_decompress_bitmap_base_addr : NPUReg60XX_RegFieldWrapper<"dma_acc_info_decompress_bitmap_base_addr", 27, 37>;

def reg_dma_acc_info_decompress_reg : NPUReg60XX_RegisterWrapper<"dma_acc_info_decompress", 8, 0x20,  // 32 dec
    ["dma_acc_info_decompress_dtype", "dma_acc_info_decompress_reserved1",
    "dma_acc_info_decompress_sparse", "dma_acc_info_decompress_bitc_en", "dma_acc_info_decompress_z",
    "dma_acc_info_decompress_bitmap_buf_sz", "dma_acc_info_decompress_reserved2", "dma_acc_info_decompress_bitmap_base_addr"]>;

//
//  dma_64_bit_word_4_w_prep
//

def bf_dma_acc_info_w_prep_dtype : NPUReg60XX_RegFieldWrapper<"dma_acc_info_w_prep_dtype", 2, 0>;
def bf_dma_acc_info_w_prep_reserved1 : NPUReg60XX_RegFieldWrapper<"dma_acc_info_w_prep_reserved1", 1, 2>;
def bf_dma_acc_info_w_prep_sparse : NPUReg60XX_RegFieldWrapper<"dma_acc_info_w_prep_sparse", 2, 3>;
def bf_dma_acc_info_w_prep_zeropoint : NPUReg60XX_RegFieldWrapper<"dma_acc_info_w_prep_zeropoint", 8, 5>;
def bf_dma_acc_info_w_prep_ic : NPUReg60XX_RegFieldWrapper<"dma_acc_info_w_prep_ic", 14, 13>;
def bf_dma_acc_info_w_prep_filtersize : NPUReg60XX_RegFieldWrapper<"dma_acc_info_w_prep_filtersize", 7, 27>;
def bf_dma_acc_info_w_prep_reserved2 : NPUReg60XX_RegFieldWrapper<"dma_acc_info_w_prep_reserved2", 3, 34>;
def bf_dma_acc_info_w_prep_bitmap_base_addr : NPUReg60XX_RegFieldWrapper<"dma_acc_info_w_prep_bitmap_base_addr", 27, 37>;

def reg_dma_acc_info_w_prep_reg : NPUReg60XX_RegisterWrapper<"dma_acc_info_w_prep", 8, 0x20,  // 32 dec
    ["dma_acc_info_w_prep_dtype", "dma_acc_info_w_prep_reserved1", "dma_acc_info_w_prep_sparse",
    "dma_acc_info_w_prep_zeropoint", "dma_acc_info_w_prep_ic", "dma_acc_info_w_prep_filtersize",
    "dma_acc_info_w_prep_reserved2", "dma_acc_info_w_prep_bitmap_base_addr"]>;

//
//  dma_64_bit_word_4_acc_info_mem_set
//

def bf_dma_mset_data : NPUReg60XX_RegFieldWrapper<"dma_mset_data", 32, 0>;

def reg_dma_mset_data : NPUReg60XX_RegisterWrapper<"dma_mset_data", 4, 0x20,  // 32 dec
    ["dma_mset_data"]>;

//
//  dma_64_bit_word_5
//

def bf_dma_src : NPUReg60XX_RegFieldWrapper<"dma_src", 48, 0>;
def bf_dma_sra : NPUReg60XX_RegFieldWrapper<"dma_sra", 1, 63>;

def reg_dma_src_addr : NPUReg60XX_RegisterWrapper<"dma_src_addr", 8, 0x28,  // 40 dec
    ["dma_src", "dma_sra"]>;

//
//  dma_64_bit_word_6
//

def bf_dma_dst : NPUReg60XX_RegFieldWrapper<"dma_dst", 48, 0>;
def bf_dma_dra : NPUReg60XX_RegFieldWrapper<"dma_dra", 1, 63>;

def reg_dma_dst_addr : NPUReg60XX_RegisterWrapper<"dma_dst_addr", 8, 0x30,  // 48 dec
    ["dma_dst", "dma_dra"]>;

//
//  dma_64_bit_word_7
//

def bf_dma_sba_addr : NPUReg60XX_RegFieldWrapper<"dma_sba_addr", 32, 0>;
def reg_dma_sba_addr : NPUReg60XX_RegisterWrapper<"dma_sba_addr", 4, 0x38,  // 56 dec
    ["dma_sba_addr"]>;

def bf_dma_dba_addr : NPUReg60XX_RegFieldWrapper<"dma_dba_addr", 32, 0>;
def reg_dma_dba_addr : NPUReg60XX_RegisterWrapper<"dma_dba_addr", 4, 0x3C,  // 56 dec
    ["dma_dba_addr"]>;

//
//  dma_64_bit_word_8
//

def bf_dma_barrier_prod_mask_lower : NPUReg60XX_RegFieldWrapper<"dma_barrier_prod_mask_lower", 64, 0>;
def reg_dma_barrier_prod_mask_lower : NPUReg60XX_RegisterWrapper<"dma_barrier_prod_mask_lower", 8, 0x40,  // 64 dec
    ["dma_barrier_prod_mask_lower"]>;

//
//  dma_64_bit_word_9
//

def bf_dma_barrier_cons_mask_lower : NPUReg60XX_RegFieldWrapper<"dma_barrier_cons_mask_lower", 64, 0>;
def reg_dma_barrier_cons_mask_lower : NPUReg60XX_RegisterWrapper<"dma_barrier_cons_mask_lower", 8, 0x48,  // 72 dec
    ["dma_barrier_cons_mask_lower"]>;

//
//  dma_64_bit_word_10
//

def bf_dma_barrier_prod_mask_upper : NPUReg60XX_RegFieldWrapper<"dma_barrier_prod_mask_upper", 32, 0>;
def reg_dma_barrier_prod_mask_upper : NPUReg60XX_RegisterWrapper<"dma_barrier_prod_mask_upper", 8, 0x50,  // 80 dec
    ["dma_barrier_prod_mask_upper"]>;

//
//  dma_64_bit_word_11
//

def bf_dma_barrier_cons_mask_upper : NPUReg60XX_RegFieldWrapper<"dma_barrier_cons_mask_upper", 32, 0>;
def reg_dma_barrier_cons_mask_upper : NPUReg60XX_RegisterWrapper<"dma_barrier_cons_mask_upper", 8, 0x58,  // 88 dec
    ["dma_barrier_cons_mask_upper"]>;

//
//  dma_64_bit_word_12
//

def bf_dma_list_size_src : NPUReg60XX_RegFieldWrapper<"dma_list_size_src", 32, 0>;
def bf_dma_list_size_dst : NPUReg60XX_RegFieldWrapper<"dma_list_size_dst", 32, 32>;

def reg_dma_list_size : NPUReg60XX_RegisterWrapper<"dma_list_size", 8, 0x60,  // 96 dec
    ["dma_list_size_src", "dma_list_size_dst"]>;

def bf_dma_dim_size_src_1 : NPUReg60XX_RegFieldWrapper<"dma_dim_size_src_1", 32, 0>;
def bf_dma_dim_size_dst_1 : NPUReg60XX_RegFieldWrapper<"dma_dim_size_dst_1", 32, 32>;

def reg_dma_dim_size : NPUReg60XX_RegisterWrapper<"dma_dim_size", 8, 0x60,  // 96 dec
    ["dma_dim_size_src_1", "dma_dim_size_dst_1"]>;

//
//  dma_64_bit_word_13
//

def bf_dma_stride_src_1 : NPUReg60XX_RegFieldWrapper<"dma_stride_src_1", 32, 0>;
def reg_dma_stride_src_1 : NPUReg60XX_RegisterWrapper<"dma_stride_src_1", 4, 0x68,  // 104 dec
    ["dma_stride_src_1"]>;

def bf_dma_stride_dst_1 : NPUReg60XX_RegFieldWrapper<"dma_stride_dst_1", 32, 0>;
def reg_dma_stride_dst_1 : NPUReg60XX_RegisterWrapper<"dma_stride_dst_1", 4, 0x6C,  // 108 dec
    ["dma_stride_dst_1"]>;

//
//  dma_64_bit_word_14
//

def bf_dma_dim_size_src_2 : NPUReg60XX_RegFieldWrapper<"dma_dim_size_src_2", 32, 0>;
def bf_dma_dim_size_dst_2 : NPUReg60XX_RegFieldWrapper<"dma_dim_size_dst_2", 32, 32>;

def reg_dma_dim_size_2 : NPUReg60XX_RegisterWrapper<"dma_dim_size_2", 8, 0x70,  // 112 dec
    ["dma_dim_size_src_2", "dma_dim_size_dst_2"]>;

def bf_dma_list_addr_src : NPUReg60XX_RegFieldWrapper<"dma_list_addr_src", 32, 0>;
def bf_dma_list_addr_dst : NPUReg60XX_RegFieldWrapper<"dma_list_addr_dst", 32, 32>;

def reg_dma_list_addr : NPUReg60XX_RegisterWrapper<"dma_list_addr", 8, 0x70,  // 112 dec
    ["dma_list_addr_src", "dma_list_addr_dst"]>;

//
//  dma_64_bit_word_15
//

def bf_dma_stride_src_2 : NPUReg60XX_RegFieldWrapper<"dma_stride_src_2", 32, 0>;
def reg_dma_stride_src_2 : NPUReg60XX_RegisterWrapper<"dma_stride_src_2", 4, 0x78,  // 120 dec
    ["dma_stride_src_2"]>;

def bf_dma_stride_dst_2 : NPUReg60XX_RegFieldWrapper<"dma_stride_dst_2", 32, 0>;
def reg_dma_stride_dst_2 : NPUReg60XX_RegisterWrapper<"dma_stride_dst_2", 4, 0x7C,  // 124 dec
    ["dma_stride_dst_2"]>;

def bf_dma_remote_width_store : NPUReg60XX_RegFieldWrapper<"dma_remote_width_store", 32, 0>;
def reg_dma_remote_width_store : NPUReg60XX_RegisterWrapper<"dma_remote_width_store", 4, 0x7C,  // 124 dec
    ["dma_remote_width_store"]>;

//
//  dma_64_bit_word_16
//

def bf_dma_dim_size_src_3 : NPUReg60XX_RegFieldWrapper<"dma_dim_size_src_3", 16, 0>;
def reg_dma_dim_size_src_3 : NPUReg60XX_RegisterWrapper<"dma_dim_size_src_3", 2, 0x80,  // 128 dec
    ["dma_dim_size_src_3"]>;

def bf_dma_dim_size_src_4 : NPUReg60XX_RegFieldWrapper<"dma_dim_size_src_4", 16, 0>;
def reg_dma_dim_size_src_4 : NPUReg60XX_RegisterWrapper<"dma_dim_size_src_4", 2, 0x82,  // 130 dec
    ["dma_dim_size_src_4"]>;

def bf_dma_dim_size_dst_3 : NPUReg60XX_RegFieldWrapper<"dma_dim_size_dst_3", 16, 0>;
def reg_dma_dim_size_dst_3 : NPUReg60XX_RegisterWrapper<"dma_dim_size_dst_3", 2, 0x84,  // 132 dec
    ["dma_dim_size_dst_3"]>;

def bf_dma_dim_size_dst_4 : NPUReg60XX_RegFieldWrapper<"dma_dim_size_dst_4", 16, 0>;
def reg_dma_dim_size_dst_4 : NPUReg60XX_RegisterWrapper<"dma_dim_size_dst_4", 2, 0x86,  // 134 dec
    ["dma_dim_size_dst_4"]>;

//
//  dma_64_bit_word_17
//

def bf_dma_dim_size_src_5 : NPUReg60XX_RegFieldWrapper<"dma_dim_size_src_5", 16, 0>;
def reg_dma_dim_size_src_5 : NPUReg60XX_RegisterWrapper<"dma_dim_size_src_5", 2, 0x88,  // 136 dec
    ["dma_dim_size_src_5"]>;

def bf_dma_dim_size_dst_5 : NPUReg60XX_RegFieldWrapper<"dma_dim_size_dst_5", 16, 0>;
def reg_dma_dst_dim_size_5 : NPUReg60XX_RegisterWrapper<"dma_dim_size_dst_5", 2, 0x8C,  // 140 dec
    ["dma_dim_size_dst_5"]>;

//
//  dma_64_bit_word_18
//

def bf_dma_stride_src_3 : NPUReg60XX_RegFieldWrapper<"dma_stride_src_3", 32, 0>;
def reg_dma_stride_src_3 : NPUReg60XX_RegisterWrapper<"dma_stride_src_3", 4, 0x90,  // 144 dec
    ["dma_stride_src_3"]>;

def bf_dma_stride_dst_3 : NPUReg60XX_RegFieldWrapper<"dma_stride_dst_3", 32, 0>;
def reg_dma_stride_dst_3 : NPUReg60XX_RegisterWrapper<"dma_stride_dst_3", 4, 0x94,  // 148 dec
    ["dma_stride_dst_3"]>;

//
//  dma_64_bit_word_19
//

def bf_dma_stride_src_4 : NPUReg60XX_RegFieldWrapper<"dma_stride_src_4", 32, 0>;
def reg_dma_stride_src_4 : NPUReg60XX_RegisterWrapper<"dma_stride_src_4", 4, 0x98,  // 152 dec
    ["dma_stride_src_4"]>;

def bf_dma_stride_dst_4 : NPUReg60XX_RegFieldWrapper<"dma_stride_dst_4", 32, 0>;
def reg_dma_stride_dst_4 : NPUReg60XX_RegisterWrapper<"dma_stride_dst_4", 4, 0x9C,  // 156 dec
    ["dma_stride_dst_4"]>;

//
//  dma_64_bit_word_20
//

def bf_dma_stride_src_5 : NPUReg60XX_RegFieldWrapper<"dma_stride_src_5", 32, 0>;
def reg_dma_stride_src_5 : NPUReg60XX_RegisterWrapper<"dma_stride_src_5", 4, 0xA0,  // 160 dec
    ["dma_stride_src_5"]>;

def bf_dma_stride_dst_5 : NPUReg60XX_RegFieldWrapper<"dma_stride_dst_5", 32, 0>;
def reg_dma_stride_dst_5 : NPUReg60XX_RegisterWrapper<"dma_stride_dst_5", 4, 0xA4,  // 164 dec
    ["dma_stride_dst_5"]>;

//
//  dma_64_bit_word_21
//

def bf_dma_word_21_reserved : NPUReg60XX_RegFieldWrapper<"dma_word_21_reserved", 64, 0>;
def reg_dma_word_21_reserved : NPUReg60XX_RegisterWrapper<"dma_word_21_reserved", 8, 0xA8,  // 168 dec
    ["dma_word_21_reserved"]>;

//
//  dma_64_bit_word_22
//

def bf_dma_word_22_reserved : NPUReg60XX_RegFieldWrapper<"dma_word_22_reserved", 64, 0>;
def reg_dma_word_22_reserved : NPUReg60XX_RegisterWrapper<"dma_word_22_reserved", 8, 0xB0,  // 176 dec
    ["dma_word_22_reserved"]>;

//
//  dma_64_bit_word_21
//

def bf_dma_word_23_reserved : NPUReg60XX_RegFieldWrapper<"dma_word_23_reserved", 64, 0>;
def reg_dma_word_23_reserved : NPUReg60XX_RegisterWrapper<"dma_word_23_reserved", 8, 0xB8,  // 184 dec
    ["dma_word_23_reserved"]>;

//
//  dma_barriers_sched
//

def reg_VpuTaskSchedulingBarrierConfig : NPUReg60XX_RegisterWrapper<"dma_barriers_sched", 8, 0xC0,
    ["start_after_", "clean_after_"]>;

def bf_dma_pad_24_0 : NPUReg60XX_RegFieldWrapper<"dma_pad_24_0", 64, 0>;
def reg_dma_pad_24_0 : NPUReg60XX_RegisterWrapper<"dma_pad_24_0", 8, 0xC8, ["dma_pad_24_0"]>;

def bf_dma_pad_24_1 : NPUReg60XX_RegFieldWrapper<"dma_pad_24_1", 64, 0>;
def reg_dma_pad_24_1 : NPUReg60XX_RegisterWrapper<"dma_pad_24_1", 8, 0xD0, ["dma_pad_24_1"]>;

def bf_dma_pad_24_2 : NPUReg60XX_RegFieldWrapper<"dma_pad_24_2", 64, 0>;
def reg_dma_pad_24_2 : NPUReg60XX_RegisterWrapper<"dma_pad_24_2", 8, 0xD8, ["dma_pad_24_2"]>;

// DMARegister descriptor
def reg_mapped_DMARegister : NPUReg60XX_RegMappedWrapper<"DMARegister",
    ["dma_watermark", "dma_link_address", "dma_lra",
    "dma_lba_addr", "dma_src_aub", "dma_dst_aub",
    "dma_cfg_fields",
    "dma_remote_width_fetch", "dma_width",
    "dma_acc_info_compress", "dma_acc_info_decompress", "dma_acc_info_w_prep", "dma_mset_data",
    "dma_src_addr",
    "dma_dst_addr",
    "dma_sba_addr", "dma_dba_addr",
    "dma_barrier_prod_mask_lower",
    "dma_barrier_cons_mask_lower",
    "dma_barrier_prod_mask_upper",
    "dma_barrier_cons_mask_upper",
    "dma_list_size", "dma_dim_size",
    "dma_stride_src_1", "dma_stride_dst_1",
    "dma_dim_size_2", "dma_list_addr",
    "dma_stride_src_2", "dma_stride_dst_2", "dma_remote_width_store",
    "dma_dim_size_src_3", "dma_dim_size_src_4", "dma_dim_size_dst_3", "dma_dim_size_dst_4",
    "dma_dim_size_src_5", "dma_dim_size_dst_5",
    "dma_stride_src_3", "dma_stride_dst_3",
    "dma_stride_src_4", "dma_stride_dst_4",
    "dma_stride_src_5", "dma_stride_dst_5",
    "dma_word_21_reserved",
    "dma_word_22_reserved",
    "dma_word_23_reserved",
    "dma_barriers_sched",
    "dma_pad_24_0", "dma_pad_24_1", "dma_pad_24_2"
    ]>;

//
//  BarrierCfg registers
//

def bf_NextSameId : NPUReg60XX_RegFieldWrapper<"next_same_id_", 32, 0, "UINT">;
def reg_NextSameId : NPUReg60XX_RegisterWrapper<"next_same_id_", 4, 0x0, ["next_same_id_"]>;

def bf_ProducerCount: NPUReg60XX_RegFieldWrapper<"producer_count_", 16, 0>;
def reg_ProducerCount : NPUReg60XX_RegisterWrapper<"producer_count_", 2, 0x4, ["producer_count_"]>;

def bf_ConsumerCount : NPUReg60XX_RegFieldWrapper<"consumer_count_", 16, 0>;
def reg_ConsumerCount : NPUReg60XX_RegisterWrapper<"consumer_count_", 2, 0x6, ["consumer_count_"]>;

def bf_BarrierId : NPUReg60XX_RegFieldWrapper<"real_id_", 8, 0>;
def reg_BarrierId : NPUReg60XX_RegisterWrapper<"real_id_", 1, 0x8, ["real_id_"]>;

def bf_BarCfg_Pad3 : NPUReg60XX_RegFieldWrapper<"barcfg_pad_3_", 24, 0>;
def reg_BarCfg_Pad3 : NPUReg60XX_RegisterWrapper<"barcfg_pad_3_", 3, 0x9, ["barcfg_pad_3_"]>;

// BarrierCfg descriptor
def reg_mapped_BarrierDescriptor : NPUReg60XX_RegMappedWrapper<"VpuBarrierCountConfig",
    ["next_same_id_", "producer_count_", "consumer_count_", "real_id_", "barcfg_pad_3_"]>;

//
// ActKernelRange registers
//

def bf_ActWLType : NPUReg60XX_RegFieldWrapper<"type", 8, 0>;
def reg_ActWLType : NPUReg60XX_RegisterWrapper<"type", 1, 0x0, ["type"]>; // WL_KERNEL = 0x00, WL_DEBUG = 0x04, WL_UNKNOWN

def bf_KernelEntry : NPUReg60XX_RegFieldWrapper<"kernel_entry", 64, 0>;
def reg_KernelEntry : NPUReg60XX_RegisterWrapper<"kernel_entry", 8, 0x8, ["kernel_entry"]>;

def bf_TextWindowBase: NPUReg60XX_RegFieldWrapper<"text_window_base", 64, 0>;
def reg_TextWindowBase : NPUReg60XX_RegisterWrapper<"text_window_base", 8, 0x10, ["text_window_base"]>;

def bf_CodeSize : NPUReg60XX_RegFieldWrapper<"code_size", 32, 0>;
def reg_CodeSize : NPUReg60XX_RegisterWrapper<"code_size", 4, 0x18, ["code_size"]>;

def bf_DeprecatedAkr : NPUReg60XX_RegFieldWrapper<"deprecated_akr", 32, 0, "UINT">;
def reg_DeprecatedAkr : NPUReg60XX_RegisterWrapper<"deprecated_akr", 4, 0x1C, ["deprecated_akr"]>;

def bf_KernelInvoCount : NPUReg60XX_RegFieldWrapper<"kernel_invo_count", 32, 0>;
def reg_KernelInvoCount : NPUReg60XX_RegisterWrapper<"kernel_invo_count", 4, 0x20, ["kernel_invo_count"]>;

def bf_Pad1_4 : NPUReg60XX_RegFieldWrapper<"pad1_4", 32, 0>;
def reg_Pad1_4 : NPUReg60XX_RegisterWrapper<"pad1_4", 4, 0x24, ["pad1_4"]>;

// ActKernelRange descriptor
def reg_mapped_ActKernelRangeDescriptor : NPUReg60XX_RegMappedWrapper<"VpuActKernelRange",
    ["type", "kernel_entry", "text_window_base", "code_size", "deprecated_akr", "kernel_invo_count", "pad1_4"]>;

//
// ActKernelInvocation registers
//

def bf_Range : NPUReg60XX_RegFieldWrapper<"range", 64, 0>;
def reg_Range : NPUReg60XX_RegisterWrapper<"range", 8, 0x0, ["range"]>;

def bf_KernelArgs : NPUReg60XX_RegFieldWrapper<"kernel_args", 64, 0>;
def reg_KernelArgs : NPUReg60XX_RegisterWrapper<"kernel_args", 8, 0x8, ["kernel_args"]>;

def bf_DataWindowBase: NPUReg60XX_RegFieldWrapper<"data_window_base", 64, 0>;
def reg_DataWindowBase : NPUReg60XX_RegisterWrapper<"data_window_base", 8, 0x10, ["data_window_base"]>;

def bf_PerfPacketOut : NPUReg60XX_RegFieldWrapper<"perf_packet_out", 64, 0>;
def reg_PerfPacketOut : NPUReg60XX_RegisterWrapper<"perf_packet_out", 8, 0x18, ["perf_packet_out"]>;

def bf_BarrierWaitMaskHiAct : NPUReg60XX_RegFieldWrapper<"barriers_wait_mask_hi_act", 32, 0>;
def reg_BarrierWaitMaskHiAct : NPUReg60XX_RegisterWrapper<"barriers_wait_mask_hi_act", 8, 0x20, ["barriers_wait_mask_hi_act"]>;

def bf_BarrierWaitMaskLoAct : NPUReg60XX_RegFieldWrapper<"barriers_wait_mask_lo_act", 64, 0>;
def reg_BarrierWaitMaskLoAct : NPUReg60XX_RegisterWrapper<"barriers_wait_mask_lo_act", 8, 0x28, ["barriers_wait_mask_lo_act"]>;

def bf_BarrierWaitPostHiAct : NPUReg60XX_RegFieldWrapper<"barriers_post_mask_hi_act", 32, 0>;
def reg_BarrierWaitPostHiAct : NPUReg60XX_RegisterWrapper<"barriers_post_mask_hi_act", 8, 0x30, ["barriers_post_mask_hi_act"]>;

def bf_BarrierWaitPostLoAct : NPUReg60XX_RegFieldWrapper<"barriers_post_mask_lo_act", 64, 0>;
def reg_BarrierWaitPostLoAct : NPUReg60XX_RegisterWrapper<"barriers_post_mask_lo_act", 8, 0x38, ["barriers_post_mask_lo_act"]>;

def bf_BarrierGroupAct : NPUReg60XX_RegFieldWrapper<"group_act", 8, 0>;
def bf_BarrierMaskAct : NPUReg60XX_RegFieldWrapper<"mask_act", 8, 8>;
def reg_BarrierGroupMaskAct : NPUReg60XX_RegisterWrapper<"barriers_group_mask_act", 8, 0x40, ["group_act", "mask_act"]>;

def reg_SchedulingBarrierConfig : NPUReg60XX_RegisterWrapper<"act_invo_barriers_sched", 8, 0x48, ["start_after_", "clean_after_"]>;

def bf_DeprecatedAki : NPUReg60XX_RegFieldWrapper<"invo_index", 32, 0, "UINT">;
def reg_DeprecatedAki : NPUReg60XX_RegisterWrapper<"invo_index", 4, 0x50, ["invo_index"]>;

def bf_InvoTile : NPUReg60XX_RegFieldWrapper<"invo_tile", 32, 0>;
def reg_InvoTile : NPUReg60XX_RegisterWrapper<"invo_tile", 4, 0x54, ["invo_tile"]>;

def bf_KernelRangeIndex : NPUReg60XX_RegFieldWrapper<"kernel_range_index", 32, 0>;
def reg_KernelRangeIndex : NPUReg60XX_RegisterWrapper<"kernel_range_index", 4, 0x58, ["kernel_range_index"]>;

def bf_NextAkiWlAddr : NPUReg60XX_RegFieldWrapper<"next_aki_wl_addr", 32, 0, "UINT">;
def reg_NextAkibf_NextAkiWlAddr : NPUReg60XX_RegisterWrapper<"next_aki_wl_addr", 4, 0x5C, ["next_aki_wl_addr"]>;

// ActKernelInvocation descriptor
def reg_mapped_ActKernelInvocationDescriptor : NPUReg60XX_RegMappedWrapper<"VpuActKernelInvocation",
    ["range", "kernel_args", "data_window_base", "perf_packet_out",
    "barriers_wait_mask_hi_act", "barriers_wait_mask_lo_act", "barriers_post_mask_hi_act", "barriers_post_mask_lo_act", "barriers_group_mask_act",
    "act_invo_barriers_sched", "invo_index", "invo_tile", "kernel_range_index", "next_aki_wl_addr"]>;

// WorkItem registers

def bf_WI_desc_ptr : NPUReg60XX_RegFieldWrapper<"desc_ptr", 64, 0>;
def reg_WI_desc_ptr : NPUReg60XX_RegisterWrapper<"desc_ptr", 8, 0x0, ["desc_ptr"]>;

def bf_WI_type : NPUReg60XX_RegFieldWrapper<"wi_type", 8, 0>;
def reg_WI_type : NPUReg60XX_RegisterWrapper<"wi_type", 1, 0x8, ["wi_type"]>;

def bf_WI_unit : NPUReg60XX_RegFieldWrapper<"wi_unit", 8, 0>;
def reg_WI_unit : NPUReg60XX_RegisterWrapper<"wi_unit", 1, 0x9, ["wi_unit"]>;

def bf_WI_sub_unit : NPUReg60XX_RegFieldWrapper<"wi_sub_unit", 8, 0>;
def reg_WI_sub_unit : NPUReg60XX_RegisterWrapper<"wi_sub_unit", 1, 0xA, ["wi_sub_unit"]>;

def bf_Pad0_8_0 : NPUReg60XX_RegFieldWrapper<"pad0_8_0", 8, 0>;
def reg_Pad0_8_0 : NPUReg60XX_RegisterWrapper<"pad0_8_0", 1, 0xB, ["pad0_8_0"]>;

def bf_WI_next_workitem_idx : NPUReg60XX_RegFieldWrapper<"next_workitem_idx", 32, 0>;
def reg_WI_next_workitem_idx : NPUReg60XX_RegisterWrapper<"next_workitem_idx", 4, 0xC, ["next_workitem_idx"]>;

// in API we have pas with 48 bytes, but in compiler we have sanity check for 8 byte per elem -> need 6 pads
def bf_Pad1_8_0 : NPUReg60XX_RegFieldWrapper<"pad1_8_0", 64, 0>;
def reg_Pad1_8_0 : NPUReg60XX_RegisterWrapper<"pad1_8_0", 8, 0x10, ["pad1_8_0"]>;

def bf_Pad1_8_1 : NPUReg60XX_RegFieldWrapper<"pad1_8_1", 64, 0>;
def reg_Pad1_8_1 : NPUReg60XX_RegisterWrapper<"pad1_8_1", 8, 0x18, ["pad1_8_1"]>;

def bf_Pad1_8_2 : NPUReg60XX_RegFieldWrapper<"pad1_8_2", 64, 0>;
def reg_Pad1_8_2 : NPUReg60XX_RegisterWrapper<"pad1_8_2", 8, 0x20, ["pad1_8_2"]>;

def bf_Pad1_8_3 : NPUReg60XX_RegFieldWrapper<"pad1_8_3", 64, 0>;
def reg_Pad1_8_3 : NPUReg60XX_RegisterWrapper<"pad1_8_3", 8, 0x28, ["pad1_8_3"]>;

def bf_Pad1_8_4 : NPUReg60XX_RegFieldWrapper<"pad1_8_4", 64, 0>;
def reg_Pad1_8_4 : NPUReg60XX_RegisterWrapper<"pad1_8_4", 8, 0x30, ["pad1_8_4"]>;

def bf_Pad1_8_5 : NPUReg60XX_RegFieldWrapper<"pad1_8_5", 64, 0>;
def reg_Pad1_8_5 : NPUReg60XX_RegisterWrapper<"pad1_8_5", 8, 0x38, ["pad1_8_5"]>;


def reg_mapped_WorkItemDescriptor : NPUReg60XX_RegMappedWrapper<"WorkItem",
    ["desc_ptr", "wi_type", "wi_unit", "wi_sub_unit", "pad0_8_0", "next_workitem_idx", "pad1_8_0", "pad1_8_1", "pad1_8_2", "pad1_8_3", "pad1_8_4", "pad1_8_5"]>;

// TaskBarrierMap registers

def bf_TaskBarrierNextSameId : NPUReg60XX_RegFieldWrapper<"tb_next_same_id", 32, 0>;
def reg_TaskBarrierNextSameId : NPUReg60XX_RegisterWrapper<"tb_next_same_id", 4, 0x0, ["tb_next_same_id"]>;

def bf_TaskBarrierProducerCount : NPUReg60XX_RegFieldWrapper<"tb_producer_count", 16, 0>;
def reg_TaskBarrierProducerCount : NPUReg60XX_RegisterWrapper<"tb_producer_count", 2, 0x4, ["tb_producer_count"]>;

def bf_TaskBarrierConsumerCount : NPUReg60XX_RegFieldWrapper<"tb_consumer_count", 16, 0>;
def reg_TaskBarrierConsumerCount : NPUReg60XX_RegisterWrapper<"tb_consumer_count", 2, 0x6, ["tb_consumer_count"]>;

def bf_TaskBarrierRealId : NPUReg60XX_RegFieldWrapper<"tb_real_id", 8, 0>;
def reg_TaskBarrierRealId : NPUReg60XX_RegisterWrapper<"tb_real_id", 1, 0x8, ["tb_real_id"]>;

def bf_TaskBarrierPad3 : NPUReg60XX_RegFieldWrapper<"tb_pad3", 24, 0>;
def reg_TaskBarrierPad3 : NPUReg60XX_RegisterWrapper<"tb_pad3", 3, 0x9, ["tb_pad3"]>;

def bf_TaskBarrierWorkItemIdx : NPUReg60XX_RegFieldWrapper<"tb_work_item_idx", 32, 0>;
def reg_TaskBarrierWorkItemIdx : NPUReg60XX_RegisterWrapper<"tb_work_item_idx", 4, 0xC, ["tb_work_item_idx"]>;

def bf_TaskBArrierEnqueueCount : NPUReg60XX_RegFieldWrapper<"tb_enqueue_count", 32, 0>;
def reg_TaskBArrierEnqueueCount : NPUReg60XX_RegisterWrapper<"tb_enqueue_count", 4, 0x10, ["tb_enqueue_count"]>;

def bf_TaskBArrierReservedNextEnqueueId : NPUReg60XX_RegFieldWrapper<"tb_reserved_next_enqueue_id", 32, 0>;
def ref_TaskBArrierReservedNextEnqueueId : NPUReg60XX_RegisterWrapper<"tb_reserved_next_enqueue_id", 4, 0x14, ["tb_reserved_next_enqueue_id"]>;

// TaskBarrierMap descriptor
def reg_mapped_TaskBarrierMapDescriptor : NPUReg60XX_RegMappedWrapper<"VpuTaskBarrierMap",
    ["tb_next_same_id", "tb_producer_count", "tb_consumer_count", "tb_real_id", "tb_pad3", "tb_work_item_idx", "tb_enqueue_count", "tb_reserved_next_enqueue_id"]>;

//
// VpuTaskReference
//

def bf_taskReferenceR1 : NPUReg60XX_RegFieldWrapper<"taskReferenceR1", 64, 0, "UINT">;
def reg_taskReferenceR1 : NPUReg60XX_RegisterWrapper<"taskReferenceR1", 8, 0x0, ["taskReferenceR1"]>;

def bf_taskReferenceR2 : NPUReg60XX_RegFieldWrapper<"taskReferenceR2", 64, 0, "UINT">;
def reg_taskReferenceR2 : NPUReg60XX_RegisterWrapper<"taskReferenceR2", 8, 0x8, ["taskReferenceR2"]>;

def bf_taskReferenceR3 : NPUReg60XX_RegFieldWrapper<"taskReferenceR3", 64, 0, "UINT">;
def reg_taskReferenceR3 : NPUReg60XX_RegisterWrapper<"taskReferenceR3", 8, 0x10, ["taskReferenceR3"]>;

def bf_taskReferenceAddr : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr", 64, 0, "UINT">;
def reg_taskReferenceAddr : NPUReg60XX_RegisterWrapper<"taskReferenceAddr", 8, 0x18, ["taskReferenceAddr"]>;

def bf_taskReferenceCount : NPUReg60XX_RegFieldWrapper<"taskReferenceCount", 64, 0>;
def reg_taskReferenceCount : NPUReg60XX_RegisterWrapper<"taskReferenceCount", 8, 0x20, ["taskReferenceCount"]>;

def reg_mapped_VpuTaskReferenceDescriptor : NPUReg60XX_RegMappedWrapper<"VpuTaskReference",
    ["taskReferenceR1", "taskReferenceR2", "taskReferenceR3", "taskReferenceAddr", "taskReferenceCount"]>;

// VpuNNShaveRuntimeConfigs
def bf_NNRTCfg_reserved : NPUReg60XX_RegFieldWrapper<"NNRTCfg_reserved", 64, 0>;
def reg_NNRTCfg_reserved : NPUReg60XX_RegisterWrapper<"NNRTCfg_reserved", 8, 0x0, ["NNRTCfg_reserved"]>;

def bf_NNRTCfg_runtime_entry : NPUReg60XX_RegFieldWrapper<"NNRTCfg_runtime_entry", 64, 0>;
def reg_NNRTCfg_runtime_entry : NPUReg60XX_RegisterWrapper<"NNRTCfg_runtime_entry", 8, 0x8, ["NNRTCfg_runtime_entry"]>;

def bf_NNRTCfg_act_rt_window_base : NPUReg60XX_RegFieldWrapper<"NNRTCfg_act_rt_window_base", 64, 0>;
def reg_NNRTCfg_act_rt_window_base : NPUReg60XX_RegisterWrapper<"NNRTCfg_act_rt_window_base", 8, 0x10, ["NNRTCfg_act_rt_window_base"]>;

// stack_frames
def bf_NNRTCfg_stack_0 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_0", 32, 0>;
def reg_NNRTCfg_stack_0 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_0", 4, 0x18, ["NNRTCfg_stack_0"]>;
def bf_NNRTCfg_stack_1 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_1", 32, 0>;
def reg_NNRTCfg_stack_1 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_1", 4, 0x1C, ["NNRTCfg_stack_1"]>;
def bf_NNRTCfg_stack_2 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_2", 32, 0>;
def reg_NNRTCfg_stack_2 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_2", 4, 0x20, ["NNRTCfg_stack_2"]>;
def bf_NNRTCfg_stack_3 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_3", 32, 0>;
def reg_NNRTCfg_stack_3 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_3", 4, 0x24, ["NNRTCfg_stack_3"]>;
def bf_NNRTCfg_stack_4 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_4", 32, 0>;
def reg_NNRTCfg_stack_4 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_4", 4, 0x28, ["NNRTCfg_stack_4"]>;
def bf_NNRTCfg_stack_5 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_5", 32, 0>;
def reg_NNRTCfg_stack_5 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_5", 4, 0x2C, ["NNRTCfg_stack_5"]>;
def bf_NNRTCfg_stack_6 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_6", 32, 0>;
def reg_NNRTCfg_stack_6 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_6", 4, 0x30, ["NNRTCfg_stack_6"]>;
def bf_NNRTCfg_stack_7 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_7", 32, 0>;
def reg_NNRTCfg_stack_7 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_7", 4, 0x34, ["NNRTCfg_stack_7"]>;
def bf_NNRTCfg_stack_8 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_8", 32, 0>;
def reg_NNRTCfg_stack_8 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_8", 4, 0x38, ["NNRTCfg_stack_8"]>;
def bf_NNRTCfg_stack_9 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_9", 32, 0>;
def reg_NNRTCfg_stack_9 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_9", 4, 0x3C, ["NNRTCfg_stack_9"]>;
def bf_NNRTCfg_stack_10 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_10", 32, 0>;
def reg_NNRTCfg_stack_10 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_10", 4, 0x40, ["NNRTCfg_stack_10"]>;
def bf_NNRTCfg_stack_11 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_11", 32, 0>;
def reg_NNRTCfg_stack_11 : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_11", 4, 0x44, ["NNRTCfg_stack_11"]>;

def bf_NNRTCfg_stack_size : NPUReg60XX_RegFieldWrapper<"NNRTCfg_stack_size", 32, 0>;
def reg_NNRTCfg_stack_size : NPUReg60XX_RegisterWrapper<"NNRTCfg_stack_size", 4, 0x48, ["NNRTCfg_stack_size"]>;

def bf_NNRTCfg_deprecated1 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_deprecated1", 32, 0>;
def reg_NNRTCfg_deprecated1 : NPUReg60XX_RegisterWrapper<"NNRTCfg_deprecated1", 4, 0x4C, ["NNRTCfg_deprecated1"]>;

def bf_NNRTCfg_perf_metrics_mask : NPUReg60XX_RegFieldWrapper<"NNRTCfg_perf_metrics_mask", 32, 0>;
def reg_NNRTCfg_perf_metrics_mask : NPUReg60XX_RegisterWrapper<"NNRTCfg_perf_metrics_mask", 4, 0x50, ["NNRTCfg_perf_metrics_mask"]>;

def bf_NNRTCfg_runtime_version : NPUReg60XX_RegFieldWrapper<"NNRTCfg_runtime_version", 32, 0>;
def reg_NNRTCfg_runtime_version : NPUReg60XX_RegisterWrapper<"NNRTCfg_runtime_version", 4, 0x54, ["NNRTCfg_runtime_version"]>;

def bf_NNRTCfg_deprecated2 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_deprecated2", 8, 0>;
def reg_NNRTCfg_deprecated2 : NPUReg60XX_RegisterWrapper<"NNRTCfg_deprecated2", 1, 0x58, ["NNRTCfg_deprecated2"]>;

def bf_NNRTCfg_deprecated3 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_deprecated3", 8, 0>;
def reg_NNRTCfg_deprecated3 : NPUReg60XX_RegisterWrapper<"NNRTCfg_deprecated3", 1, 0x59, ["NNRTCfg_deprecated3"]>;

def bf_NNRTCfg_pad_6 : NPUReg60XX_RegFieldWrapper<"NNRTCfg_pad_6", 48, 0>;
def reg_NNRTCfg_pad_6 : NPUReg60XX_RegisterWrapper<"NNRTCfg_pad_6", 6, 0x5A, ["NNRTCfg_pad_6"]>;

def reg_mapped_VpuNNShaveRuntimeConfigsDescriptor : NPUReg60XX_RegMappedWrapper<"VpuNNShaveRuntimeConfigs",
    ["NNRTCfg_reserved", "NNRTCfg_runtime_entry", "NNRTCfg_act_rt_window_base", "NNRTCfg_stack_0", "NNRTCfg_stack_1", "NNRTCfg_stack_2",
     "NNRTCfg_stack_3", "NNRTCfg_stack_4", "NNRTCfg_stack_5", "NNRTCfg_stack_6", "NNRTCfg_stack_7", "NNRTCfg_stack_8", "NNRTCfg_stack_9",
     "NNRTCfg_stack_10", "NNRTCfg_stack_11", "NNRTCfg_stack_size", "NNRTCfg_deprecated1", "NNRTCfg_perf_metrics_mask",
     "NNRTCfg_runtime_version", "NNRTCfg_deprecated2", "NNRTCfg_deprecated3", "NNRTCfg_pad_6"
    ]>;

//VpuNNRTConfig
// this contains VpuNNShaveRuntimeConfigs (see above), and it has two aditional fields:

def bf_NNRTCfg_logAddrDmaHwp : NPUReg60XX_RegFieldWrapper<"NNRTCfg_logAddrDmaHwp", 64, 0>;
def reg_NNRTCfg_logAddrDmaHwp : NPUReg60XX_RegisterWrapper<"NNRTCfg_logAddrDmaHwp", 8, 0x60, ["NNRTCfg_logAddrDmaHwp"]>;

def bf_NNRTCfg_HwpCfgAddr : NPUReg60XX_RegFieldWrapper<"NNRTCfg_HwpCfgAddr", 64, 0>;
def reg_NNRTCfg_HwpCfgAddr : NPUReg60XX_RegisterWrapper<"NNRTCfg_HwpCfgAddr", 8, 0x68, ["NNRTCfg_HwpCfgAddr"]>;

def reg_mapped_VpuNNRTConfigDescriptor : NPUReg60XX_RegMappedWrapper<"VpuNNRTConfig",
    ["NNRTCfg_reserved", "NNRTCfg_runtime_entry", "NNRTCfg_act_rt_window_base", "NNRTCfg_stack_0", "NNRTCfg_stack_1", "NNRTCfg_stack_2", "NNRTCfg_stack_3",
     "NNRTCfg_stack_4", "NNRTCfg_stack_5", "NNRTCfg_stack_6", "NNRTCfg_stack_7", "NNRTCfg_stack_8", "NNRTCfg_stack_9", "NNRTCfg_stack_10", "NNRTCfg_stack_11",
     "NNRTCfg_stack_size", "NNRTCfg_deprecated1", "NNRTCfg_perf_metrics_mask", "NNRTCfg_runtime_version", "NNRTCfg_deprecated2",
     "NNRTCfg_deprecated3", "NNRTCfg_pad_6", "NNRTCfg_logAddrDmaHwp", "NNRTCfg_HwpCfgAddr"
    ], /*canBeEmpty=*/ true>;


//
// VpuBarrierConfiguration
//
def bf_BarrierConfiguration_producerCount : NPUReg60XX_RegFieldWrapper<"BarrierConfiguration_producerCount", 8, 0, "UINT">;
def reg_BarrierConfiguration_producerCount : NPUReg60XX_RegisterWrapper<"BarrierConfiguration_producerCount", 1, 0x0, ["BarrierConfiguration_producerCount"]>;

def bf_BarrierConfiguration_producerInterruptEnabled : NPUReg60XX_RegFieldWrapper<"BarrierConfiguration_producerInterruptEnabled", 8, 0, "UINT">;
def reg_BarrierConfiguration_producerInterruptEnabled : NPUReg60XX_RegisterWrapper<"BarrierConfiguration_producerInterruptEnabled", 1, 0x1, ["BarrierConfiguration_producerInterruptEnabled"]>;

def bf_BarrierConfiguration_consumerCount : NPUReg60XX_RegFieldWrapper<"BarrierConfiguration_consumerCount", 8, 0, "UINT">;
def reg_BarrierConfiguration_consumerCount : NPUReg60XX_RegisterWrapper<"BarrierConfiguration_consumerCount", 1, 0x2, ["BarrierConfiguration_consumerCount"]>;

def bf_BarrierConfiguration_consumerInterruptEnabled : NPUReg60XX_RegFieldWrapper<"BarrierConfiguration_consumerInterruptEnabled", 8, 0, "UINT">;
def reg_BarrierConfiguration_consumerInterruptEnabled : NPUReg60XX_RegisterWrapper<"BarrierConfiguration_consumerInterruptEnabled", 1, 0x3, ["BarrierConfiguration_consumerInterruptEnabled"]>;


def reg_mapped_VpuBarrierConfigurationDescriptor : NPUReg60XX_RegMappedWrapper<"VpuBarrierConfiguration",
    [
        "BarrierConfiguration_producerCount", "BarrierConfiguration_producerInterruptEnabled",
        "BarrierConfiguration_consumerCount", "BarrierConfiguration_consumerInterruptEnabled"
    ]>;

//
// VpuDpuPVP
//
def bf_VpuDpuPVP_opFPCount : NPUReg60XX_RegFieldWrapper<"VpuDpuPVP_opFPCount", 64, 0, "UINT">;
def reg_VpuDpuPVP_opFPCount : NPUReg60XX_RegisterWrapper<"VpuDpuPVP_opFPCount", 8, 0x0, ["VpuDpuPVP_opFPCount"]>;

def bf_VpuDpuPVP_totalOpCount : NPUReg60XX_RegFieldWrapper<"VpuDpuPVP_totalOpCount", 64, 0, "UINT">;
def reg_VpuDpuPVP_totalOpCount : NPUReg60XX_RegisterWrapper<"VpuDpuPVP_totalOpCount", 8, 0x8, ["VpuDpuPVP_totalOpCount"]>;

// uint8_t reserved[16];
def bf_VpuDpuPVP_reserved_0 : NPUReg60XX_RegFieldWrapper<"VpuDpuPVP_reserved_0", 64, 0, "UINT">;
def reg_VpuDpuPVP_reserved_0 : NPUReg60XX_RegisterWrapper<"VpuDpuPVP_reserved_0", 8, 0x10, ["VpuDpuPVP_reserved_0"]>;
def bf_VpuDpuPVP_reserved_1 : NPUReg60XX_RegFieldWrapper<"VpuDpuPVP_reserved_1", 64, 0, "UINT">;
def reg_VpuDpuPVP_reserved_1 : NPUReg60XX_RegisterWrapper<"VpuDpuPVP_reserved_1", 8, 0x18, ["VpuDpuPVP_reserved_1"]>;

// canBeEmpty=1: an all-default (zero) VpuDpuPVP descriptor is a valid runtime state,
// so version deduction must fall back to the baseline rather than failing.
def reg_mapped_VpuDpuPVPDescriptor : NPUReg60XX_RegMappedWrapper<"VpuDpuPVP",
    [
        "VpuDpuPVP_opFPCount", "VpuDpuPVP_totalOpCount", "VpuDpuPVP_reserved_0", "VpuDpuPVP_reserved_1"
    ], /*canBeEmpty=*/1>;

//
// VpuManagedMappedInference
//

// uint32_t vpu_nnrt_api_ver
def bf_MMI_vpu_nnrt_api_ver : NPUReg60XX_RegFieldWrapper<"MMI_vpu_nnrt_api_ver", 32, 0, "UINT">;
def reg_MMI_vpu_nnrt_api_ver : NPUReg60XX_RegisterWrapper<"MMI_vpu_nnrt_api_ver", 4, 0x0, ["MMI_vpu_nnrt_api_ver"]>;

// uint32_t final_barrier
def bf_MMI_final_barrier : NPUReg60XX_RegFieldWrapper<"MMI_final_barrier", 32, 0, "UINT">;
def reg_MMI_final_barrier : NPUReg60XX_RegisterWrapper<"MMI_final_barrier", 4, 0x4, ["MMI_final_barrier"]>;

// VpuTaskReference<VpuWorkItem> work_items
def bf_taskReferenceR1_MMI_work_item : NPUReg60XX_RegFieldWrapper<"taskReferenceR1_MMI_work_item", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_work_item : NPUReg60XX_RegisterWrapper<"taskReferenceR1_MMI_work_item", 8, 0x8, ["taskReferenceR1_MMI_work_item"]>;
def bf_taskReferenceR2_MMI_work_item : NPUReg60XX_RegFieldWrapper<"taskReferenceR2_MMI_work_item", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_work_item : NPUReg60XX_RegisterWrapper<"taskReferenceR2_MMI_work_item", 8, 0x10, ["taskReferenceR2_MMI_work_item"]>;
def bf_taskReferenceR3_MMI_work_item : NPUReg60XX_RegFieldWrapper<"taskReferenceR3_MMI_work_item", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_work_item : NPUReg60XX_RegisterWrapper<"taskReferenceR3_MMI_work_item", 8, 0x18, ["taskReferenceR3_MMI_work_item"]>;
def bf_taskReferenceAddr_MMI_work_item : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr_MMI_work_item", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_work_item : NPUReg60XX_RegisterWrapper<"taskReferenceAddr_MMI_work_item", 8, 0x20, ["taskReferenceAddr_MMI_work_item"]>;
def bf_taskReferenceCount_MMI_work_item : NPUReg60XX_RegFieldWrapper<"taskReferenceCount_MMI_work_item", 64, 0>;
def reg_taskReferenceCount_MMI_work_item : NPUReg60XX_RegisterWrapper<"taskReferenceCount_MMI_work_item", 8, 0x28, ["taskReferenceCount_MMI_work_item"]>;

// VpuTaskReference<VpuTaskBarrierMap> task_configs
def bf_taskReferenceR1_MMI_task_configs : NPUReg60XX_RegFieldWrapper<"taskReferenceR1_MMI_task_configs", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_task_configs : NPUReg60XX_RegisterWrapper<"taskReferenceR1_MMI_task_configs", 8, 0x30, ["taskReferenceR1_MMI_task_configs"]>;
def bf_taskReferenceR2_MMI_task_configs : NPUReg60XX_RegFieldWrapper<"taskReferenceR2_MMI_task_configs", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_task_configs : NPUReg60XX_RegisterWrapper<"taskReferenceR2_MMI_task_configs", 8, 0x38, ["taskReferenceR2_MMI_task_configs"]>;
def bf_taskReferenceR3_MMI_task_configs : NPUReg60XX_RegFieldWrapper<"taskReferenceR3_MMI_task_configs", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_task_configs : NPUReg60XX_RegisterWrapper<"taskReferenceR3_MMI_task_configs", 8, 0x40, ["taskReferenceR3_MMI_task_configs"]>;
def bf_taskReferenceAddr_MMI_task_configs : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr_MMI_task_configs", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_task_configs : NPUReg60XX_RegisterWrapper<"taskReferenceAddr_MMI_task_configs", 8, 0x48, ["taskReferenceAddr_MMI_task_configs"]>;
def bf_taskReferenceCount_MMI_task_configs : NPUReg60XX_RegFieldWrapper<"taskReferenceCount_MMI_task_configs", 64, 0>;
def reg_taskReferenceCount_MMI_task_configs : NPUReg60XX_RegisterWrapper<"taskReferenceCount_MMI_task_configs", 8, 0x50, ["taskReferenceCount_MMI_task_configs"]>;

// VpuTaskReference<uint32_t> reserved0[3]
def bf_taskReferenceR1_MMI_reserved0_0 : NPUReg60XX_RegFieldWrapper<"taskReferenceR1_MMI_reserved0_0", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_reserved0_0 : NPUReg60XX_RegisterWrapper<"taskReferenceR1_MMI_reserved0_0", 8, 0x58, ["taskReferenceR1_MMI_reserved0_0"]>;
def bf_taskReferenceR2_MMI_reserved0_0 : NPUReg60XX_RegFieldWrapper<"taskReferenceR2_MMI_reserved0_0", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_reserved0_0 : NPUReg60XX_RegisterWrapper<"taskReferenceR2_MMI_reserved0_0", 8, 0x60, ["taskReferenceR2_MMI_reserved0_0"]>;
def bf_taskReferenceR3_MMI_reserved0_0 : NPUReg60XX_RegFieldWrapper<"taskReferenceR3_MMI_reserved0_0", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_reserved0_0 : NPUReg60XX_RegisterWrapper<"taskReferenceR3_MMI_reserved0_0", 8, 0x68, ["taskReferenceR3_MMI_reserved0_0"]>;
def bf_taskReferenceAddr_MMI_reserved0_0 : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr_MMI_reserved0_0", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_reserved0_0 : NPUReg60XX_RegisterWrapper<"taskReferenceAddr_MMI_reserved0_0", 8, 0x70, ["taskReferenceAddr_MMI_reserved0_0"]>;
def bf_taskReferenceCount_MMI_reserved0_0 : NPUReg60XX_RegFieldWrapper<"taskReferenceCount_MMI_reserved0_0", 64, 0>;
def reg_taskReferenceCount_MMI_reserved0_0 : NPUReg60XX_RegisterWrapper<"taskReferenceCount_MMI_reserved0_0", 8, 0x78, ["taskReferenceCount_MMI_reserved0_0"]>;

def bf_taskReferenceR1_MMI_reserved0_1 : NPUReg60XX_RegFieldWrapper<"taskReferenceR1_MMI_reserved0_1", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_reserved0_1 : NPUReg60XX_RegisterWrapper<"taskReferenceR1_MMI_reserved0_1", 8, 0x80, ["taskReferenceR1_MMI_reserved0_1"]>;
def bf_taskReferenceR2_MMI_reserved0_1 : NPUReg60XX_RegFieldWrapper<"taskReferenceR2_MMI_reserved0_1", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_reserved0_1 : NPUReg60XX_RegisterWrapper<"taskReferenceR2_MMI_reserved0_1", 8, 0x88, ["taskReferenceR2_MMI_reserved0_1"]>;
def bf_taskReferenceR3_MMI_reserved0_1 : NPUReg60XX_RegFieldWrapper<"taskReferenceR3_MMI_reserved0_1", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_reserved0_1 : NPUReg60XX_RegisterWrapper<"taskReferenceR3_MMI_reserved0_1", 8, 0x90, ["taskReferenceR3_MMI_reserved0_1"]>;
def bf_taskReferenceAddr_MMI_reserved0_1 : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr_MMI_reserved0_1", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_reserved0_1 : NPUReg60XX_RegisterWrapper<"taskReferenceAddr_MMI_reserved0_1", 8, 0x98, ["taskReferenceAddr_MMI_reserved0_1"]>;
def bf_taskReferenceCount_MMI_reserved0_1 : NPUReg60XX_RegFieldWrapper<"taskReferenceCount_MMI_reserved0_1", 64, 0>;
def reg_taskReferenceCount_MMI_reserved0_1 : NPUReg60XX_RegisterWrapper<"taskReferenceCount_MMI_reserved0_1", 8, 0xA0, ["taskReferenceCount_MMI_reserved0_1"]>;

def bf_taskReferenceR1_MMI_reserved0_2 : NPUReg60XX_RegFieldWrapper<"taskReferenceR1_MMI_reserved0_2", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_reserved0_2 : NPUReg60XX_RegisterWrapper<"taskReferenceR1_MMI_reserved0_2", 8, 0xA8, ["taskReferenceR1_MMI_reserved0_2"]>;
def bf_taskReferenceR2_MMI_reserved0_2 : NPUReg60XX_RegFieldWrapper<"taskReferenceR2_MMI_reserved0_2", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_reserved0_2 : NPUReg60XX_RegisterWrapper<"taskReferenceR2_MMI_reserved0_2", 8, 0xB0, ["taskReferenceR2_MMI_reserved0_2"]>;
def bf_taskReferenceR3_MMI_reserved0_2 : NPUReg60XX_RegFieldWrapper<"taskReferenceR3_MMI_reserved0_2", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_reserved0_2 : NPUReg60XX_RegisterWrapper<"taskReferenceR3_MMI_reserved0_2", 8, 0xB8, ["taskReferenceR3_MMI_reserved0_2"]>;
def bf_taskReferenceAddr_MMI_reserved0_2 : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr_MMI_reserved0_2", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_reserved0_2 : NPUReg60XX_RegisterWrapper<"taskReferenceAddr_MMI_reserved0_2", 8, 0xC0, ["taskReferenceAddr_MMI_reserved0_2"]>;
def bf_taskReferenceCount_MMI_reserved0_2 : NPUReg60XX_RegFieldWrapper<"taskReferenceCount_MMI_reserved0_2", 64, 0>;
def reg_taskReferenceCount_MMI_reserved0_2 : NPUReg60XX_RegisterWrapper<"taskReferenceCount_MMI_reserved0_2", 8, 0xC8, ["taskReferenceCount_MMI_reserved0_2"]>;

//  VpuTaskReference<VpuDpuPVP> dpu_pvp;
def bf_taskReferenceR1_MMI_dpu_pvp : NPUReg60XX_RegFieldWrapper<"taskReferenceR1_MMI_dpu_pvp", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_dpu_pvp : NPUReg60XX_RegisterWrapper<"taskReferenceR1_MMI_dpu_pvp", 8, 0xD0, ["taskReferenceR1_MMI_dpu_pvp"]>;
def bf_taskReferenceR2_MMI_dpu_pvp : NPUReg60XX_RegFieldWrapper<"taskReferenceR2_MMI_dpu_pvp", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_dpu_pvp : NPUReg60XX_RegisterWrapper<"taskReferenceR2_MMI_dpu_pvp", 8, 0xD8, ["taskReferenceR2_MMI_dpu_pvp"]>;
def bf_taskReferenceR3_MMI_dpu_pvp : NPUReg60XX_RegFieldWrapper<"taskReferenceR3_MMI_dpu_pvp", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_dpu_pvp : NPUReg60XX_RegisterWrapper<"taskReferenceR3_MMI_dpu_pvp", 8, 0xE0, ["taskReferenceR3_MMI_dpu_pvp"]>;
def bf_taskReferenceAddr_MMI_dpu_pvp : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr_MMI_dpu_pvp", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_dpu_pvp : NPUReg60XX_RegisterWrapper<"taskReferenceAddr_MMI_dpu_pvp", 8, 0xE8, ["taskReferenceAddr_MMI_dpu_pvp"]>;
def bf_taskReferenceCount_MMI_dpu_pvp : NPUReg60XX_RegFieldWrapper<"taskReferenceCount_MMI_dpu_pvp", 64, 0, "UINT">;
def reg_taskReferenceCount_MMI_dpu_pvp : NPUReg60XX_RegisterWrapper<"taskReferenceCount_MMI_dpu_pvp", 8, 0xF0, ["taskReferenceCount_MMI_dpu_pvp"]>;

// VpuTaskReference<VpuBarrierConfiguration> barriers_configuration
def bf_taskReferenceR1_MMI_barriers_configuration : NPUReg60XX_RegFieldWrapper<"taskReferenceR1_MMI_barriers_configuration", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_barriers_configuration : NPUReg60XX_RegisterWrapper<"taskReferenceR1_MMI_barriers_configuration", 8, 0xF8, ["taskReferenceR1_MMI_barriers_configuration"]>;
def bf_taskReferenceR2_MMI_barriers_configuration : NPUReg60XX_RegFieldWrapper<"taskReferenceR2_MMI_barriers_configuration", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_barriers_configuration : NPUReg60XX_RegisterWrapper<"taskReferenceR2_MMI_barriers_configuration", 8, 0x100, ["taskReferenceR2_MMI_barriers_configuration"]>;
def bf_taskReferenceR3_MMI_barriers_configuration : NPUReg60XX_RegFieldWrapper<"taskReferenceR3_MMI_barriers_configuration", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_barriers_configuration : NPUReg60XX_RegisterWrapper<"taskReferenceR3_MMI_barriers_configuration", 8, 0x108, ["taskReferenceR3_MMI_barriers_configuration"]>;
def bf_taskReferenceAddr_MMI_barriers_configuration : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr_MMI_barriers_configuration", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_barriers_configuration : NPUReg60XX_RegisterWrapper<"taskReferenceAddr_MMI_barriers_configuration", 8, 0x110, ["taskReferenceAddr_MMI_barriers_configuration"]>;
def bf_taskReferenceCount_MMI_barriers_configuration : NPUReg60XX_RegFieldWrapper<"taskReferenceCount_MMI_barriers_configuration", 64, 0, "UINT">;
def reg_taskReferenceCount_MMI_barriers_configuration : NPUReg60XX_RegisterWrapper<"taskReferenceCount_MMI_barriers_configuration", 8, 0x118, ["taskReferenceCount_MMI_barriers_configuration"]>;

// VpuTaskReference<uint32_t> num_of_barrier_reprogrammings
def bf_taskReferenceR1_MMI_num_of_barrier_reprogrammings : NPUReg60XX_RegFieldWrapper<"taskReferenceR1_MMI_num_of_barrier_reprogrammings", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_num_of_barrier_reprogrammings : NPUReg60XX_RegisterWrapper<"taskReferenceR1_MMI_num_of_barrier_reprogrammings", 8, 0x120, ["taskReferenceR1_MMI_num_of_barrier_reprogrammings"]>;
def bf_taskReferenceR2_MMI_num_of_barrier_reprogrammings : NPUReg60XX_RegFieldWrapper<"taskReferenceR2_MMI_num_of_barrier_reprogrammings", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_num_of_barrier_reprogrammings : NPUReg60XX_RegisterWrapper<"taskReferenceR2_MMI_num_of_barrier_reprogrammings", 8, 0x128, ["taskReferenceR2_MMI_num_of_barrier_reprogrammings"]>;
def bf_taskReferenceR3_MMI_num_of_barrier_reprogrammings : NPUReg60XX_RegFieldWrapper<"taskReferenceR3_MMI_num_of_barrier_reprogrammings", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_num_of_barrier_reprogrammings : NPUReg60XX_RegisterWrapper<"taskReferenceR3_MMI_num_of_barrier_reprogrammings", 8, 0x130, ["taskReferenceR3_MMI_num_of_barrier_reprogrammings"]>;
def bf_taskReferenceAddr_MMI_num_of_barrier_reprogrammings : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr_MMI_num_of_barrier_reprogrammings", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_num_of_barrier_reprogrammings : NPUReg60XX_RegisterWrapper<"taskReferenceAddr_MMI_num_of_barrier_reprogrammings", 8, 0x138, ["taskReferenceAddr_MMI_num_of_barrier_reprogrammings"]>;
def bf_taskReferenceCount_MMI_num_of_barrier_reprogrammings : NPUReg60XX_RegFieldWrapper<"taskReferenceCount_MMI_num_of_barrier_reprogrammings", 64, 0>;
def reg_taskReferenceCount_MMI_num_of_barrier_reprogrammings : NPUReg60XX_RegisterWrapper<"taskReferenceCount_MMI_num_of_barrier_reprogrammings", 8, 0x140, ["taskReferenceCount_MMI_num_of_barrier_reprogrammings"]>;

// VpuTaskReference<uint32_t> initial_barriers
def bf_taskReferenceR1_MMI_initial_barriers : NPUReg60XX_RegFieldWrapper<"taskReferenceR1_MMI_initial_barriers", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_initial_barriers : NPUReg60XX_RegisterWrapper<"taskReferenceR1_MMI_initial_barriers", 8, 0x148, ["taskReferenceR1_MMI_initial_barriers"]>;
def bf_taskReferenceR2_MMI_initial_barriers : NPUReg60XX_RegFieldWrapper<"taskReferenceR2_MMI_initial_barriers", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_initial_barriers : NPUReg60XX_RegisterWrapper<"taskReferenceR2_MMI_initial_barriers", 8, 0x150, ["taskReferenceR2_MMI_initial_barriers"]>;
def bf_taskReferenceR3_MMI_initial_barriers : NPUReg60XX_RegFieldWrapper<"taskReferenceR3_MMI_initial_barriers", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_initial_barriers : NPUReg60XX_RegisterWrapper<"taskReferenceR3_MMI_initial_barriers", 8, 0x158, ["taskReferenceR3_MMI_initial_barriers"]>;
def bf_taskReferenceAddr_MMI_initial_barriers : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr_MMI_initial_barriers", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_initial_barriers : NPUReg60XX_RegisterWrapper<"taskReferenceAddr_MMI_initial_barriers", 8, 0x160, ["taskReferenceAddr_MMI_initial_barriers"]>;
def bf_taskReferenceCount_MMI_initial_barriers : NPUReg60XX_RegFieldWrapper<"taskReferenceCount_MMI_initial_barriers", 64, 0>;
def reg_taskReferenceCount_MMI_initial_barriers : NPUReg60XX_RegisterWrapper<"taskReferenceCount_MMI_initial_barriers", 8, 0x168, ["taskReferenceCount_MMI_initial_barriers"]>;

// VpuTaskReference<VpuNNRTConfig> nnrt_config
def bf_taskReferenceR1_MMI_nnrt_config : NPUReg60XX_RegFieldWrapper<"taskReferenceR1_MMI_nnrt_config", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_nnrt_config : NPUReg60XX_RegisterWrapper<"taskReferenceR1_MMI_nnrt_config", 8, 0x170, ["taskReferenceR1_MMI_nnrt_config"]>;
def bf_taskReferenceR2_MMI_nnrt_config : NPUReg60XX_RegFieldWrapper<"taskReferenceR2_MMI_nnrt_config", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_nnrt_config : NPUReg60XX_RegisterWrapper<"taskReferenceR2_MMI_nnrt_config", 8, 0x178, ["taskReferenceR2_MMI_nnrt_config"]>;
def bf_taskReferenceR3_MMI_nnrt_config : NPUReg60XX_RegFieldWrapper<"taskReferenceR3_MMI_nnrt_config", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_nnrt_config : NPUReg60XX_RegisterWrapper<"taskReferenceR3_MMI_nnrt_config", 8, 0x180, ["taskReferenceR3_MMI_nnrt_config"]>;
def bf_taskReferenceAddr_MMI_nnrt_config : NPUReg60XX_RegFieldWrapper<"taskReferenceAddr_MMI_nnrt_config", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_nnrt_config : NPUReg60XX_RegisterWrapper<"taskReferenceAddr_MMI_nnrt_config", 8, 0x188, ["taskReferenceAddr_MMI_nnrt_config"]>;
def bf_taskReferenceCount_MMI_nnrt_config : NPUReg60XX_RegFieldWrapper<"taskReferenceCount_MMI_nnrt_config", 64, 0>;
def reg_taskReferenceCount_MMI_nnrt_config : NPUReg60XX_RegisterWrapper<"taskReferenceCount_MMI_nnrt_config", 8, 0x190, ["taskReferenceCount_MMI_nnrt_config"]>;

// uint8_t actshv_used
def bf_MMI_actshv_used : NPUReg60XX_RegFieldWrapper<"MMI_actshv_used", 8, 0, "UINT">;
def reg_MMI_actshv_used : NPUReg60XX_RegisterWrapper<"MMI_actshv_used", 1, 0x198, ["MMI_actshv_used"]>;

// uint8_t dpu_used
def bf_MMI_dpu_used : NPUReg60XX_RegFieldWrapper<"MMI_dpu_used", 8, 0, "UINT">;
def reg_MMI_dpu_used : NPUReg60XX_RegisterWrapper<"MMI_dpu_used", 1, 0x199, ["MMI_dpu_used"]>;

// uint8_t deprecated (was media_used)
def bf_MMI_media_deprecated : NPUReg60XX_RegFieldWrapper<"MMI_media_deprecated", 8, 0, "UINT">;
def reg_MMI_media_deprecated : NPUReg60XX_RegisterWrapper<"MMI_media_deprecated", 1, 0x19A, ["MMI_media_deprecated"]>;

// uint8_t dma_from_ddr_used
def bf_MMI_dma_from_ddr_used : NPUReg60XX_RegFieldWrapper<"MMI_dma_from_ddr_used", 8, 0, "UINT">;
def reg_MMI_dma_from_ddr_used : NPUReg60XX_RegisterWrapper<"MMI_dma_from_ddr_used", 1, 0x19B, ["MMI_dma_from_ddr_used"]>;

// uint8_t dma_from_cmx_used
def bf_MMI_dma_from_cmx_used : NPUReg60XX_RegFieldWrapper<"MMI_dma_from_cmx_used", 8, 0, "UINT">;
def reg_MMI_dma_from_cmx_used : NPUReg60XX_RegisterWrapper<"MMI_dma_from_cmx_used", 1, 0x19C, ["MMI_dma_from_cmx_used"]>;

// uint8_t pad0_[2];
def bf_MMI_pad0_ : NPUReg60XX_RegFieldWrapper<"MMI_pad0_", 16, 0, "UINT">;
def reg_MMI_pad0_ : NPUReg60XX_RegisterWrapper<"MMI_pad0_", 2, 0x19D, ["MMI_pad0_"]>;

// VpuBarrierProgrammingMode barrier_programming_mode;
def bf_MMI_barrier_programming_mode : NPUReg60XX_RegFieldWrapper<"MMI_barrier_programming_mode", 8, 0, "UINT">;
def reg_MMI_barrier_programming_mode : NPUReg60XX_RegisterWrapper<"MMI_barrier_programming_mode", 1, 0x19F, ["MMI_barrier_programming_mode"]>;

// uint8_t deprecated[40] (was VpuTaskReference<VpuManagedMappedInferenceInfo> inference_info)
def bf_MMI_deprecated_0 : NPUReg60XX_RegFieldWrapper<"MMI_deprecated_0", 64, 0, "UINT">;
def reg_MMI_deprecated_0 : NPUReg60XX_RegisterWrapper<"MMI_deprecated_0", 8, 0x1A0, ["MMI_deprecated_0"]>;
def bf_MMI_deprecated_1 : NPUReg60XX_RegFieldWrapper<"MMI_deprecated_1", 64, 0, "UINT">;
def reg_MMI_deprecated_1 : NPUReg60XX_RegisterWrapper<"MMI_deprecated_1", 8, 0x1A8, ["MMI_deprecated_1"]>;
def bf_MMI_deprecated_2 : NPUReg60XX_RegFieldWrapper<"MMI_deprecated_2", 64, 0, "UINT">;
def reg_MMI_deprecated_2 : NPUReg60XX_RegisterWrapper<"MMI_deprecated_2", 8, 0x1B0, ["MMI_deprecated_2"]>;
def bf_MMI_deprecated_3 : NPUReg60XX_RegFieldWrapper<"MMI_deprecated_3", 64, 0, "UINT">;
def reg_MMI_deprecated_3 : NPUReg60XX_RegisterWrapper<"MMI_deprecated_3", 8, 0x1B8, ["MMI_deprecated_3"]>;
def bf_MMI_deprecated_4 : NPUReg60XX_RegFieldWrapper<"MMI_deprecated_4", 64, 0, "UINT">;
def reg_MMI_deprecated_4 : NPUReg60XX_RegisterWrapper<"MMI_deprecated_4", 8, 0x1C0, ["MMI_deprecated_4"]>;
// uint32_t barrier_configuration_stride
def bf_MMI_barrier_configuration_stride : NPUReg60XX_RegFieldWrapper<"MMI_barrier_configuration_stride", 32, 0, "UINT">;
def reg_MMI_barrier_configuration_stride : NPUReg60XX_RegisterWrapper<"MMI_barrier_configuration_stride", 4, 0x1C8, ["MMI_barrier_configuration_stride"]>;

// uint8_t inference_feature_cfg
def bf_MMI_DisableDmaSwFifo : NPUReg60XX_RegFieldWrapper<"MMI_DisableDmaSwFifo", 1, 0, "UINT">;
def bf_MMI_ReservedInferenceConfig : NPUReg60XX_RegFieldWrapper<"MMI_ReservedInferenceConfig", 7, 1, "UINT">;
def reg_MMI_inference_feature_cfg : NPUReg60XX_RegisterWrapper<"MMI_inference_feature_cfg", 1, 0x1CC, ["MMI_DisableDmaSwFifo", "MMI_ReservedInferenceConfig"]>;

// uint8_t pad1_[3]
def bf_MMI_pad1_ : NPUReg60XX_RegFieldWrapper<"MMI_pad1_", 24, 0, "UINT">;
def reg_MMI_pad1_ : NPUReg60XX_RegisterWrapper<"MMI_pad1_", 3, 0x1CD, ["MMI_pad1_"]>;

// uint32_t model_identifier
def bf_MMI_model_identifier : NPUReg60XX_RegFieldWrapper<"MMI_model_identifier", 32, 0, "UINT">;
def reg_MMI_model_identifier : NPUReg60XX_RegisterWrapper<"MMI_model_identifier", 4, 0x1D0, ["MMI_model_identifier"]>;

// uint8_t pad2_[228]
def bf_MMI_pad2_0 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_0", 64, 0, "UINT">;
def reg_MMI_pad2_0 : NPUReg60XX_RegisterWrapper<"MMI_pad2_0", 8, 0x1D4, ["MMI_pad2_0"]>;

def bf_MMI_pad2_1 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_1", 64, 0, "UINT">;
def reg_MMI_pad2_1 : NPUReg60XX_RegisterWrapper<"MMI_pad2_1", 8, 0x1DC, ["MMI_pad2_1"]>;

def bf_MMI_pad2_2 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_2", 64, 0, "UINT">;
def reg_MMI_pad2_2 : NPUReg60XX_RegisterWrapper<"MMI_pad2_2", 8, 0x1E4, ["MMI_pad2_2"]>;

def bf_MMI_pad2_3 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_3", 64, 0, "UINT">;
def reg_MMI_pad2_3 : NPUReg60XX_RegisterWrapper<"MMI_pad2_3", 8, 0x1EC, ["MMI_pad2_3"]>;

def bf_MMI_pad2_4 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_4", 64, 0, "UINT">;
def reg_MMI_pad2_4 : NPUReg60XX_RegisterWrapper<"MMI_pad2_4", 8, 0x1F4, ["MMI_pad2_4"]>;

def bf_MMI_pad2_5 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_5", 64, 0, "UINT">;
def reg_MMI_pad2_5 : NPUReg60XX_RegisterWrapper<"MMI_pad2_5", 8, 0x1FC, ["MMI_pad2_5"]>;

def bf_MMI_pad2_6 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_6", 64, 0, "UINT">;
def reg_MMI_pad2_6 : NPUReg60XX_RegisterWrapper<"MMI_pad2_6", 8, 0x204, ["MMI_pad2_6"]>;

def bf_MMI_pad2_7 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_7", 64, 0, "UINT">;
def reg_MMI_pad2_7 : NPUReg60XX_RegisterWrapper<"MMI_pad2_7", 8, 0x20C, ["MMI_pad2_7"]>;

def bf_MMI_pad2_8 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_8", 64, 0, "UINT">;
def reg_MMI_pad2_8 : NPUReg60XX_RegisterWrapper<"MMI_pad2_8", 8, 0x214, ["MMI_pad2_8"]>;

def bf_MMI_pad2_9 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_9", 64, 0, "UINT">;
def reg_MMI_pad2_9 : NPUReg60XX_RegisterWrapper<"MMI_pad2_9", 8, 0x21C, ["MMI_pad2_9"]>;

def bf_MMI_pad2_10 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_10", 64, 0, "UINT">;
def reg_MMI_pad2_10 : NPUReg60XX_RegisterWrapper<"MMI_pad2_10", 8, 0x224, ["MMI_pad2_10"]>;

def bf_MMI_pad2_11 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_11", 64, 0, "UINT">;
def reg_MMI_pad2_11 : NPUReg60XX_RegisterWrapper<"MMI_pad2_11", 8, 0x22C, ["MMI_pad2_11"]>;

def bf_MMI_pad2_12 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_12", 64, 0, "UINT">;
def reg_MMI_pad2_12 : NPUReg60XX_RegisterWrapper<"MMI_pad2_12", 8, 0x234, ["MMI_pad2_12"]>;

def bf_MMI_pad2_13 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_13", 64, 0, "UINT">;
def reg_MMI_pad2_13 : NPUReg60XX_RegisterWrapper<"MMI_pad2_13", 8, 0x23C, ["MMI_pad2_13"]>;

def bf_MMI_pad2_14 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_14", 64, 0, "UINT">;
def reg_MMI_pad2_14 : NPUReg60XX_RegisterWrapper<"MMI_pad2_14", 8, 0x244, ["MMI_pad2_14"]>;

def bf_MMI_pad2_15 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_15", 64, 0, "UINT">;
def reg_MMI_pad2_15 : NPUReg60XX_RegisterWrapper<"MMI_pad2_15", 8, 0x24C, ["MMI_pad2_15"]>;

def bf_MMI_pad2_16 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_16", 64, 0, "UINT">;
def reg_MMI_pad2_16 : NPUReg60XX_RegisterWrapper<"MMI_pad2_16", 8, 0x254, ["MMI_pad2_16"]>;

def bf_MMI_pad2_17 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_17", 64, 0, "UINT">;
def reg_MMI_pad2_17 : NPUReg60XX_RegisterWrapper<"MMI_pad2_17", 8, 0x25C, ["MMI_pad2_17"]>;

def bf_MMI_pad2_18 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_18", 64, 0, "UINT">;
def reg_MMI_pad2_18 : NPUReg60XX_RegisterWrapper<"MMI_pad2_18", 8, 0x264, ["MMI_pad2_18"]>;

def bf_MMI_pad2_19 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_19", 64, 0, "UINT">;
def reg_MMI_pad2_19 : NPUReg60XX_RegisterWrapper<"MMI_pad2_19", 8, 0x26C, ["MMI_pad2_19"]>;

def bf_MMI_pad2_20 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_20", 64, 0, "UINT">;
def reg_MMI_pad2_20 : NPUReg60XX_RegisterWrapper<"MMI_pad2_20", 8, 0x274, ["MMI_pad2_20"]>;

def bf_MMI_pad2_21 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_21", 64, 0, "UINT">;
def reg_MMI_pad2_21 : NPUReg60XX_RegisterWrapper<"MMI_pad2_21", 8, 0x27C, ["MMI_pad2_21"]>;

def bf_MMI_pad2_22 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_22", 64, 0, "UINT">;
def reg_MMI_pad2_22 : NPUReg60XX_RegisterWrapper<"MMI_pad2_22", 8, 0x284, ["MMI_pad2_22"]>;

def bf_MMI_pad2_23 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_23", 64, 0, "UINT">;
def reg_MMI_pad2_23 : NPUReg60XX_RegisterWrapper<"MMI_pad2_23", 8, 0x28C, ["MMI_pad2_23"]>;

def bf_MMI_pad2_24 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_24", 64, 0, "UINT">;
def reg_MMI_pad2_24 : NPUReg60XX_RegisterWrapper<"MMI_pad2_24", 8, 0x294, ["MMI_pad2_24"]>;

def bf_MMI_pad2_25 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_25", 64, 0, "UINT">;
def reg_MMI_pad2_25 : NPUReg60XX_RegisterWrapper<"MMI_pad2_25", 8, 0x29C, ["MMI_pad2_25"]>;

def bf_MMI_pad2_26 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_26", 64, 0, "UINT">;
def reg_MMI_pad2_26 : NPUReg60XX_RegisterWrapper<"MMI_pad2_26", 8, 0x2A4, ["MMI_pad2_26"]>;

def bf_MMI_pad2_27 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_27", 64, 0, "UINT">;
def reg_MMI_pad2_27 : NPUReg60XX_RegisterWrapper<"MMI_pad2_27", 8, 0x2AC, ["MMI_pad2_27"]>;

def bf_MMI_pad2_28 : NPUReg60XX_RegFieldWrapper<"MMI_pad2_28", 32, 0, "UINT">;
def reg_MMI_pad2_28 : NPUReg60XX_RegisterWrapper<"MMI_pad2_28", 4, 0x2B4, ["MMI_pad2_28"]>;

// uint32_t bootstrap_workitems_count
def bf_MMI_bootstrap_workitems_count : NPUReg60XX_RegFieldWrapper<"MMI_bootstrap_workitems_count", 32, 0, "UINT">;
def reg_MMI_bootstrap_workitems_count : NPUReg60XX_RegisterWrapper<"MMI_bootstrap_workitems_count", 4, 0x2B8, ["MMI_bootstrap_workitems_count"]>;

// uint8_t pad2_[4]
def bf_MMI_pad3_ : NPUReg60XX_RegFieldWrapper<"MMI_pad3_", 32, 0, "UINT">;
def reg_MMI_pad3_ : NPUReg60XX_RegisterWrapper<"MMI_pad3_", 4, 0x2BC, ["MMI_pad3_"]>;

def reg_mapped_VpuManagedMappedInferenceDescriptor : NPUReg60XX_RegMappedWrapper<"VpuManagedMappedInference",
    [
        "MMI_vpu_nnrt_api_ver", "MMI_final_barrier",
        "taskReferenceR1_MMI_work_item", "taskReferenceR2_MMI_work_item", "taskReferenceR3_MMI_work_item", "taskReferenceAddr_MMI_work_item", "taskReferenceCount_MMI_work_item",
        "taskReferenceR1_MMI_task_configs", "taskReferenceR2_MMI_task_configs", "taskReferenceR3_MMI_task_configs", "taskReferenceAddr_MMI_task_configs", "taskReferenceCount_MMI_task_configs",
        "taskReferenceR1_MMI_reserved0_0", "taskReferenceR2_MMI_reserved0_0", "taskReferenceR3_MMI_reserved0_0", "taskReferenceAddr_MMI_reserved0_0", "taskReferenceCount_MMI_reserved0_0",
        "taskReferenceR1_MMI_reserved0_1", "taskReferenceR2_MMI_reserved0_1", "taskReferenceR3_MMI_reserved0_1", "taskReferenceAddr_MMI_reserved0_1", "taskReferenceCount_MMI_reserved0_1",
        "taskReferenceR1_MMI_reserved0_2", "taskReferenceR2_MMI_reserved0_2", "taskReferenceR3_MMI_reserved0_2", "taskReferenceAddr_MMI_reserved0_2", "taskReferenceCount_MMI_reserved0_2",
        "taskReferenceR1_MMI_dpu_pvp", "taskReferenceR2_MMI_dpu_pvp", "taskReferenceR3_MMI_dpu_pvp", "taskReferenceAddr_MMI_dpu_pvp", "taskReferenceCount_MMI_dpu_pvp",
        "taskReferenceR1_MMI_barriers_configuration", "taskReferenceR2_MMI_barriers_configuration", "taskReferenceR3_MMI_barriers_configuration", "taskReferenceAddr_MMI_barriers_configuration", "taskReferenceCount_MMI_barriers_configuration",
        "taskReferenceR1_MMI_num_of_barrier_reprogrammings", "taskReferenceR2_MMI_num_of_barrier_reprogrammings", "taskReferenceR3_MMI_num_of_barrier_reprogrammings", "taskReferenceAddr_MMI_num_of_barrier_reprogrammings", "taskReferenceCount_MMI_num_of_barrier_reprogrammings",
        "taskReferenceR1_MMI_initial_barriers", "taskReferenceR2_MMI_initial_barriers", "taskReferenceR3_MMI_initial_barriers", "taskReferenceAddr_MMI_initial_barriers", "taskReferenceCount_MMI_initial_barriers",
        "taskReferenceR1_MMI_nnrt_config", "taskReferenceR2_MMI_nnrt_config", "taskReferenceR3_MMI_nnrt_config", "taskReferenceAddr_MMI_nnrt_config", "taskReferenceCount_MMI_nnrt_config",
        "MMI_actshv_used", "MMI_dpu_used", "MMI_media_deprecated", "MMI_dma_from_ddr_used", "MMI_dma_from_cmx_used", "MMI_pad0_", "MMI_barrier_programming_mode",
        "MMI_deprecated_0", "MMI_deprecated_1", "MMI_deprecated_2", "MMI_deprecated_3", "MMI_deprecated_4",
        "MMI_barrier_configuration_stride", "MMI_inference_feature_cfg", "MMI_pad1_", "MMI_model_identifier", "MMI_pad2_0", "MMI_pad2_1", "MMI_pad2_2", "MMI_pad2_3", "MMI_pad2_4", "MMI_pad2_5", "MMI_pad2_6", "MMI_pad2_7",
        "MMI_pad2_8", "MMI_pad2_9", "MMI_pad2_10", "MMI_pad2_11", "MMI_pad2_12", "MMI_pad2_13", "MMI_pad2_14", "MMI_pad2_15", "MMI_pad2_16", "MMI_pad2_17",
        "MMI_pad2_18", "MMI_pad2_19", "MMI_pad2_20", "MMI_pad2_21", "MMI_pad2_22", "MMI_pad2_23", "MMI_pad2_24", "MMI_pad2_25", "MMI_pad2_26", "MMI_pad2_27",
        "MMI_pad2_28", "MMI_bootstrap_workitems_count", "MMI_pad3_"
    ]>;

#endif
