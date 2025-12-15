//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUREG50XX_TYPES
#define VPUX_COMPILER_DIALECT_NPUREG50XX_TYPES

include "vpux/compiler/dialect/VPURegMapped/dialect.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/dialect.td"

//
// RegField Specialization
//

class NPUReg50XX_Version<int vMajor, int vMinor, int vPatch> : VPURegMapped_Version {
    let major = vMajor;
    let minor = vMinor;
    let patch = vPatch;
}

// MI Version that acts as baseline for NPU5 NNRT API
// All registers default to this baseline unless specified otherwise
def NPUReg50XX_BaselineVersion : NPUReg50XX_Version<11, 13, 0>;

class NPUReg50XX_RegFieldWrapper<string name = "", int width = 0, int pos = 0, string dataType = "UINT", VPURegMapped_Version version = NPUReg50XX_BaselineVersion> :
    VPURegMapped_RegFieldWrapper<NPUReg50XX_Dialect, name, width, pos, dataType, version> {
    }

//
// Register Specialization
//

class NPUReg50XX_RegisterWrapper<string name = "", int size = 0, int address = 0x00, list<string> regFieldTypes = [], int resetValue = 0x0, bit allowOverlap = 0> :
    VPURegMapped_RegisterWrapper<NPUReg50XX_Dialect, "vpux::NPUReg50XX::", name, size, address, regFieldTypes, resetValue, allowOverlap> {

}

//
// DPUInvariantRegisters
//

// cmx_slice0_low_addr
def bf_CmxSlice0LowAddr : NPUReg50XX_RegFieldWrapper<"cmx_slice0_low_addr", 32, 0>;
def reg_CmxSlice0LowAddr : NPUReg50XX_RegisterWrapper<"cmx_slice0_low_addr", 4, 0x0, ["cmx_slice0_low_addr"], 0x4000000>;

// cmx_slice1_low_addr
def bf_CmxSlice1LowAddr : NPUReg50XX_RegFieldWrapper<"cmx_slice1_low_addr", 32, 0>;
def reg_CmxSlice1LowAddr : NPUReg50XX_RegisterWrapper<"cmx_slice1_low_addr", 4, 0x4, ["cmx_slice1_low_addr"], 0x4000000>;

// cmx_slice2_low_addr
def bf_CmxSlice2LowAddr : NPUReg50XX_RegFieldWrapper<"cmx_slice2_low_addr", 32, 0>;
def reg_CmxSlice2LowAddr : NPUReg50XX_RegisterWrapper<"cmx_slice2_low_addr", 4, 0x8, ["cmx_slice2_low_addr"], 0x4000000>;

// cmx_slice3_low_addr
def bf_CmxSlice3LowAddr : NPUReg50XX_RegFieldWrapper<"cmx_slice3_low_addr", 32, 0>;
def reg_CmxSlice3LowAddr : NPUReg50XX_RegisterWrapper<"cmx_slice3_low_addr", 4, 0xC, ["cmx_slice3_low_addr"], 0x4000000>;

// cmx_slice_size
def bf_CmxSliceSize : NPUReg50XX_RegFieldWrapper<"cmx_slice_size", 32, 0>;
def reg_CmxSliceSize : NPUReg50XX_RegisterWrapper<"cmx_slice_size", 4, 0x10, ["cmx_slice_size"], 0x00018000>;

// se_addr
def bf_SeAddr : NPUReg50XX_RegFieldWrapper<"se_addr", 32, 0>;
def reg_SeAddr : NPUReg50XX_RegisterWrapper<"se_addr", 4, 0x14, ["se_addr"]>;

// sparsity_addr
def bf_SparsityAddr : NPUReg50XX_RegFieldWrapper<"sparsity_addr", 32, 0>;
def reg_SparsityAddr : NPUReg50XX_RegisterWrapper<"sparsity_addr", 4, 0x18, ["sparsity_addr"]>;

// se_sise
def bf_SeSize : NPUReg50XX_RegFieldWrapper<"se_size", 32, 0>;
def reg_SeSize : NPUReg50XX_RegisterWrapper<"se_size", 4, 0x1C, ["se_size"]>;

// z_config
def bf_SeZSplit : NPUReg50XX_RegFieldWrapper<"se_z_split", 4, 0>;
def bf_NumSesInZDir : NPUReg50XX_RegFieldWrapper<"num_ses_in_z_dir", 9, 4>;
def bf_CmSpPattern : NPUReg50XX_RegFieldWrapper<"cm_sp_pattern", 16, 13>;
def bf_Npo2SeZSplitEnable : NPUReg50XX_RegFieldWrapper<"npo2_se_z_split_en", 1, 29>;
def bf_Reserved : NPUReg50XX_RegFieldWrapper<"reserved", 1, 30>;
def bf_AddrFormatSel : NPUReg50XX_RegFieldWrapper<"addr_format_sel", 1, 31>;
def reg_ZConfig : NPUReg50XX_RegisterWrapper<"z_config", 4, 0x20,
    ["se_z_split", "num_ses_in_z_dir", "cm_sp_pattern", "npo2_se_z_split_en", "reserved", "addr_format_sel"], 0x0>;

// kernel_pad_cfg
def bf_MpeAssign : NPUReg50XX_RegFieldWrapper<"mpe_assign", 1, 0>;
def bf_PadRightEn : NPUReg50XX_RegFieldWrapper<"pad_right_en", 1, 1>;
def bf_PadLeftEn : NPUReg50XX_RegFieldWrapper<"pad_left_en", 1, 2>;
def bf_PadBottomEn : NPUReg50XX_RegFieldWrapper<"pad_bottom_en", 1, 3>;
def bf_PadTopEn : NPUReg50XX_RegFieldWrapper<"pad_top_en", 1, 4>;
def bf_KernelY : NPUReg50XX_RegFieldWrapper<"kernel_y", 4, 5>;
def bf_KernelX : NPUReg50XX_RegFieldWrapper<"kernel_x", 4, 9>;
def bf_WtPltCfg : NPUReg50XX_RegFieldWrapper<"wt_plt_cfg", 2, 13>;
def bf_ActDense : NPUReg50XX_RegFieldWrapper<"act_dense", 1, 15>;
def bf_WtDense : NPUReg50XX_RegFieldWrapper<"wt_dense", 1, 16>;
def bf_StrideYEn : NPUReg50XX_RegFieldWrapper<"stride_y_en", 1, 17>;
def bf_StrideY : NPUReg50XX_RegFieldWrapper<"stride_y", 3, 18>;
def bf_DynamicBwEn : NPUReg50XX_RegFieldWrapper<"dynamic_bw_en", 1, 21>;
def bf_DwWtSpIns : NPUReg50XX_RegFieldWrapper<"dw_wt_sp_ins", 1, 22>;
def bf_Layer1WtSpIns : NPUReg50XX_RegFieldWrapper<"layer1_wt_sp_ins", 1, 23>;
def bf_Layer1CmpEn : NPUReg50XX_RegFieldWrapper<"layer1_cmp_en", 1, 24>;
def bf_PoolOptEn : NPUReg50XX_RegFieldWrapper<"pool_opt_en", 1, 25>;
def bf_SpSeTblSegment : NPUReg50XX_RegFieldWrapper<"sp_se_tbl_segment", 1, 29>;
def bf_RstCtxt : NPUReg50XX_RegFieldWrapper<"rst_ctxt", 1, 30>;
def reg_KernelPadCfg : NPUReg50XX_RegisterWrapper<"kernel_pad_cfg", 4, 0x24,
    ["mpe_assign", "pad_right_en", "pad_left_en", "pad_bottom_en", "pad_top_en", "kernel_y", "kernel_x", "wt_plt_cfg",
    "act_dense", "wt_dense", "stride_y_en", "stride_y", "dynamic_bw_en", "dw_wt_sp_ins", "layer1_wt_sp_ins",
    "layer1_cmp_en", "pool_opt_en", "sp_se_tbl_segment", "rst_ctxt"]>;

// tensor_size0
def bf_TensorSizeX : NPUReg50XX_RegFieldWrapper<"tensor_size_x", 14, 0>;
def bf_TensorSizeY : NPUReg50XX_RegFieldWrapper<"tensor_size_y", 14, 14>;
def reg_TensorSize0 : NPUReg50XX_RegisterWrapper<"tensor_size0", 4, 0x28, ["tensor_size_x", "tensor_size_y"]>;

// tensor_size1
def bf_TensorSizeZ : NPUReg50XX_RegFieldWrapper<"tensor_size_z", 14, 0>;
def bf_Npo2_se_size : NPUReg50XX_RegFieldWrapper<"npo2_se_size",  9, 14>;
def bf_Tensor_cmp_size : NPUReg50XX_RegFieldWrapper<"tensor_cmp_size",  4, 23>;
def bf_Tensor_cmp_offset : NPUReg50XX_RegFieldWrapper<"tensor_cmp_offset",  4, 27>;
def reg_TensorSize1 : NPUReg50XX_RegisterWrapper<"tensor_size1", 4, 0x2c, ["tensor_size_z", "npo2_se_size", "tensor_cmp_size", "tensor_cmp_offset"]>;

// tensor_start
def bf_TensorStart : NPUReg50XX_RegFieldWrapper<"tensor_start", 32, 0>;
def reg_TensorStart : NPUReg50XX_RegisterWrapper<"tensor_start", 4, 0x30, ["tensor_start"]>;

// tensor_mode
def bf_Wmode : NPUReg50XX_RegFieldWrapper<"wmode", 4, 0>;
def bf_Amode : NPUReg50XX_RegFieldWrapper<"amode", 4, 4>;
def bf_Stride : NPUReg50XX_RegFieldWrapper<"stride", 3, 8>;
def bf_ZmInput : NPUReg50XX_RegFieldWrapper<"zm_input", 1, 11>;
def bf_DwInput : NPUReg50XX_RegFieldWrapper<"dw_input", 1, 12>;
def bf_CmInput : NPUReg50XX_RegFieldWrapper<"cm_input", 1, 13>;
def bf_WorkloadOperation : NPUReg50XX_RegFieldWrapper<"workload_operation", 2, 14>;
def bf_PadValue : NPUReg50XX_RegFieldWrapper<"pad_value", 16, 16>;
def reg_TensorMode : NPUReg50XX_RegisterWrapper<"tensor_mode", 4, 0x34,
    ["wmode", "amode", "stride", "zm_input", "dw_input", "cm_input", "workload_operation", "pad_value"]>;

// elops_sparsity_addr
def bf_ElopsSparityAddr : NPUReg50XX_RegFieldWrapper<"elops_sparsity_addr", 32, 0>;
def reg_ElopsSparityAddr : NPUReg50XX_RegisterWrapper<"elops_sparsity_addr", 4, 0x38, ["elops_sparsity_addr"]>;

// elops_se_addr
def bf_ElopsSeAddr : NPUReg50XX_RegFieldWrapper<"elops_se_addr", 32, 0>;
def reg_ElopsSeAddr : NPUReg50XX_RegisterWrapper<"elops_se_addr", 4, 0x3C, ["elops_se_addr"]>;

// elops_wload
def bf_ElopWload : NPUReg50XX_RegFieldWrapper<"elop_wload", 1, 0>;
def bf_SeedWload : NPUReg50XX_RegFieldWrapper<"seed_wload", 1, 1>;
def bf_FifoWrWload : NPUReg50XX_RegFieldWrapper<"fifo_wr_wload", 1, 2>;
def bf_ElopWloadType : NPUReg50XX_RegFieldWrapper<"elop_wload_type", 1, 3>;
def bf_PoolWtData : NPUReg50XX_RegFieldWrapper<"pool_wt_data", 16, 4>;
def bf_PoolWtRdDis : NPUReg50XX_RegFieldWrapper<"pool_wt_rd_dis", 1, 26>;
def bf_ElopOperation : NPUReg50XX_RegFieldWrapper<"elop_operation", 3, 27>;
def reg_ElopsWload : NPUReg50XX_RegisterWrapper<"elops_wload", 4, 0x40,
    ["elop_wload", "seed_wload", "fifo_wr_wload", "elop_wload_type", "pool_wt_data", "pool_wt_rd_dis", "elop_operation"], 0x0>;

// act0_offset
def bf_Adr0Offset : NPUReg50XX_RegFieldWrapper<"act_offset0", 32, 0>;
def reg_Act0Offset : NPUReg50XX_RegisterWrapper<"act_offset0", 4, 0x44, ["act_offset0"]>;

// act1_offset
def bf_Adr1Offset : NPUReg50XX_RegFieldWrapper<"act_offset1", 32, 0>;
def reg_Act1Offset : NPUReg50XX_RegisterWrapper<"act_offset1", 4, 0x48, ["act_offset1"]>;

// act2_offset
def bf_Adr2Offset : NPUReg50XX_RegFieldWrapper<"act_offset2", 32, 0>;
def reg_Act2Offset : NPUReg50XX_RegisterWrapper<"act_offset2", 4, 0x4C, ["act_offset2"]>;

// act3_offset
def bf_Adr3Offset : NPUReg50XX_RegFieldWrapper<"act_offset3", 32, 0>;
def reg_Act3Offset : NPUReg50XX_RegisterWrapper<"act_offset3", 4, 0x50, ["act_offset3"]>;

// base_offset_a
def bf_BaseOffset0 : NPUReg50XX_RegFieldWrapper<"base_offset_a", 32, 0>;
def reg_BaseOffsetA : NPUReg50XX_RegisterWrapper<"base_offset_a", 4, 0x54, ["base_offset_a"]>;

// base_offset_b
def bf_BaseOffset2 : NPUReg50XX_RegFieldWrapper<"base_offset_2", 9, 0>;
def bf_BaseOffset3 : NPUReg50XX_RegFieldWrapper<"base_offset_3", 9, 9>;
def bf_DwOptOffset : NPUReg50XX_RegFieldWrapper<"dw_opt_offset", 6, 18>;
def bf_DwOptEn : NPUReg50XX_RegFieldWrapper<"dw_opt_en", 1, 24>;
def bf_Dw3x3s1OptDis : NPUReg50XX_RegFieldWrapper<"dw_3x3s1_opt_dis", 1, 25>;
def bf_WtDenseOptEn : NPUReg50XX_RegFieldWrapper<"wt_dense_opt_en", 1, 26>;
def bf_SmallHwOptEn : NPUReg50XX_RegFieldWrapper<"small_hw_opt_en", 1, 27>;
def bf_IduCmxMuxMode : NPUReg50XX_RegFieldWrapper<"idu_cmx_mux_mode", 2, 28>;
def reg_BaseOffsetB : NPUReg50XX_RegisterWrapper<"base_offset_b", 4, 0x58,
    ["base_offset_2", "base_offset_3", "dw_opt_offset", "dw_opt_en", "dw_3x3s1_opt_dis",
    "wt_dense_opt_en", "small_hw_opt_en", "idu_cmx_mux_mode"]>;

// wt_offset
def bf_WtOffset : NPUReg50XX_RegFieldWrapper<"wt_offset", 32, 0>;
def reg_WtOffset : NPUReg50XX_RegisterWrapper<"wt_offset", 4, 0x5C, ["wt_offset"]>;

// odu_cfg
def bf_Dtype : NPUReg50XX_RegFieldWrapper<"dtype", 3, 0>;
def bf_WcbAcMode : NPUReg50XX_RegFieldWrapper<"wcb_ac_mode", 1, 3>;
def bf_WcbSpMode : NPUReg50XX_RegFieldWrapper<"wcb_sp_mode", 1, 4>;
def bf_SpValue : NPUReg50XX_RegFieldWrapper<"sp_value", 8, 5>;
def bf_SpOutEn : NPUReg50XX_RegFieldWrapper<"sp_out_en", 1, 13>;
def bf_CmxPortMuxingDisable : NPUReg50XX_RegFieldWrapper<"cmx_port_muxing_disable", 1, 14>;
def bf_WriteSp : NPUReg50XX_RegFieldWrapper<"write_sp", 1, 15>;
def bf_WritePt : NPUReg50XX_RegFieldWrapper<"write_pt", 1, 16>;
def bf_WriteAc : NPUReg50XX_RegFieldWrapper<"write_ac", 1, 17>;
def bf_Mode : NPUReg50XX_RegFieldWrapper<"mode", 2, 18>;
def bf_Grid : NPUReg50XX_RegFieldWrapper<"grid", 1, 20>;
def bf_SwizzleKey : NPUReg50XX_RegFieldWrapper<"swizzle_key", 3, 21>;
def bf_WlBpOnStartEn : NPUReg50XX_RegFieldWrapper<"wl_bp_on_start_en", 1, 24>;
def bf_Nthw : NPUReg50XX_RegFieldWrapper<"nthw", 2, 25>;
def bf_Permutation : NPUReg50XX_RegFieldWrapper<"permutation", 3, 27>;
def bf_WcbStallAvoidance : NPUReg50XX_RegFieldWrapper<"wcb_stall_avoidance", 1, 30>;
def bf_WcbBypass : NPUReg50XX_RegFieldWrapper<"wcb_bypass", 1, 31>;
def reg_OduCfg : NPUReg50XX_RegisterWrapper<"odu_cfg", 4, 0x60,
    ["dtype", "wcb_ac_mode", "wcb_sp_mode", "sp_value", "sp_out_en", "cmx_port_muxing_disable", "write_sp", "write_pt", "write_ac", "mode", "grid",
    "swizzle_key", "wl_bp_on_start_en", "nthw", "permutation", "wcb_stall_avoidance", "wcb_bypass"]>;

// odu_be_size
def bf_OduBeSize : NPUReg50XX_RegFieldWrapper<"odu_be_size", 32, 0>;
def reg_OduBeSize : NPUReg50XX_RegisterWrapper<"odu_be_size", 4, 0x64, ["odu_be_size"]>;

// odu_be_cnt
def bf_OduBeCnt : NPUReg50XX_RegFieldWrapper<"odu_be_cnt", 32, 0>;
def reg_OduBeCnt : NPUReg50XX_RegisterWrapper<"odu_be_cnt", 4, 0x68, ["odu_be_cnt"]>;

// odu_se_size
def bf_OduSeSize : NPUReg50XX_RegFieldWrapper<"odu_se_size", 32, 0>;
def reg_OduSeSize : NPUReg50XX_RegisterWrapper<"odu_se_size", 4, 0x6C, ["odu_se_size"]>;

// te_dim0
def bf_TeDimY : NPUReg50XX_RegFieldWrapper<"te_dim_y", 13, 0>;
def bf_TeDimZ : NPUReg50XX_RegFieldWrapper<"te_dim_z", 13, 13>;
def reg_TeDim0 : NPUReg50XX_RegisterWrapper<"te_dim0", 4, 0x70, ["te_dim_y", "te_dim_z"]>;

// te_dim1
def bf_TeDimX : NPUReg50XX_RegFieldWrapper<"te_dim_x", 13, 0>;
def reg_TeDim1 : NPUReg50XX_RegisterWrapper<"te_dim1", 4, 0x74, ["te_dim_x"]>;

// pt_base
def bf_PtBase : NPUReg50XX_RegFieldWrapper<"pt_base", 32, 0>;
def reg_PtBase : NPUReg50XX_RegisterWrapper<"pt_base", 4, 0x78, ["pt_base"]>;

// sp_base
def bf_SpBase : NPUReg50XX_RegFieldWrapper<"sp_base", 32, 0>;
def reg_SpBase : NPUReg50XX_RegisterWrapper<"sp_base", 4, 0x7C, ["sp_base"]>;

// mpe_cfg
def bf_MpeWtbias : NPUReg50XX_RegFieldWrapper<"mpe_wtbias", 8, 0, "UINT">;   // TODO E#80099
def bf_MpeActbias : NPUReg50XX_RegFieldWrapper<"mpe_actbias", 8, 8, "UINT">; // TODO E#80099
def bf_MpeMode : NPUReg50XX_RegFieldWrapper<"mpe_mode", 3, 16>;
def bf_MpeDense : NPUReg50XX_RegFieldWrapper<"mpe_dense", 1, 19>;
def bf_MrmWeightDense : NPUReg50XX_RegFieldWrapper<"mrm_weight_dense", 1, 20>;
def bf_MrmActDense : NPUReg50XX_RegFieldWrapper<"mrm_act_dense", 1, 21>;
def bf_MpeDaz : NPUReg50XX_RegFieldWrapper<"mpe_daz", 1, 22>;
def bf_MpeFtz : NPUReg50XX_RegFieldWrapper<"mpe_ftz", 1, 23>;
def reg_MpeCfg : NPUReg50XX_RegisterWrapper<"mpe_cfg", 4, 0x80,
    ["mpe_wtbias", "mpe_actbias", "mpe_mode", "mpe_dense", "mrm_weight_dense", "mrm_act_dense", "mpe_daz", "mpe_ftz"]>;

// mpe_bus_data_sel
def bf_MpeBusDataSel : NPUReg50XX_RegFieldWrapper<"mpe_bus_data_sel", 32, 0>;
def reg_MpeBusDataSel : NPUReg50XX_RegisterWrapper<"mpe_bus_data_sel", 4, 0x84, ["mpe_bus_data_sel"]>;

// elop_scale
def bf_ElopScaleB : NPUReg50XX_RegFieldWrapper<"elop_scale_b", 16, 0>;
def bf_ElopScaleA : NPUReg50XX_RegFieldWrapper<"elop_scale_a", 16, 16>;
def bf_ElopScaleBFp : NPUReg50XX_RegFieldWrapper<"elop_scale_b_fp", 16, 0, "FP">;
def bf_ElopScaleAFp : NPUReg50XX_RegFieldWrapper<"elop_scale_a_fp", 16, 16, "FP">;
def bf_ElopScaleBBfp : NPUReg50XX_RegFieldWrapper<"elop_scale_b_bfp", 16, 0, "BF">;
def bf_ElopScaleABfp : NPUReg50XX_RegFieldWrapper<"elop_scale_a_bfp", 16, 16, "BF">;
def reg_ElopScale : NPUReg50XX_RegisterWrapper<"elop_scale", 4, 0x88, ["elop_scale_b", "elop_scale_a", "elop_scale_b_fp", "elop_scale_a_fp", "elop_scale_b_bfp", "elop_scale_a_bfp"], 0x0, 1>;

// ppe_cfg
def bf_PpeG8BiasC : NPUReg50XX_RegFieldWrapper<"ppe_g8_bias_c", 9, 0, "SINT">;
def bf_PpeG8BiasB : NPUReg50XX_RegFieldWrapper<"ppe_g8_bias_b", 9, 9>;
def bf_PpeG8BiasA : NPUReg50XX_RegFieldWrapper<"ppe_g8_bias_a", 9, 18>;
def reg_PpeCfg : NPUReg50XX_RegisterWrapper<"ppe_cfg", 4, 0x8C, ["ppe_g8_bias_c", "ppe_g8_bias_b", "ppe_g8_bias_a"]>;

// ppe_bias
def bf_PpeBias : NPUReg50XX_RegFieldWrapper<"ppe_bias", 32, 0, "SINT">;
def reg_PpeBias : NPUReg50XX_RegisterWrapper<"ppe_bias", 4, 0x90, ["ppe_bias"]>;

// ppe_scale
def bf_PpeScaleShift : NPUReg50XX_RegFieldWrapper<"ppe_scale_shift", 6, 2>;
def bf_PpeScaleRound : NPUReg50XX_RegFieldWrapper<"ppe_scale_round", 2, 10>;
def bf_PpeScaleMult : NPUReg50XX_RegFieldWrapper<"ppe_scale_mult", 16, 16, "SINT">;
def reg_PpeScale : NPUReg50XX_RegisterWrapper<"ppe_scale", 4, 0x94, ["ppe_scale_shift", "ppe_scale_round", "ppe_scale_mult"], 0x00000c00>;

// ppe_scale_ctrl
def bf_PpeScaleOverride : NPUReg50XX_RegFieldWrapper<"ppe_scale_override", 1, 0>;
def bf_PpeFpScaleOverride : NPUReg50XX_RegFieldWrapper<"ppe_fp_scale_override", 1, 1>;
def reg_PpeScaleCtrl : NPUReg50XX_RegisterWrapper<"ppe_scale_ctrl", 4, 0x98, ["ppe_scale_override", "ppe_fp_scale_override"], 0x0>;

// ppe_prelu
def bf_PpePreluShift : NPUReg50XX_RegFieldWrapper<"ppe_prelu_shift", 5, 8>;
def bf_PpePreluMult : NPUReg50XX_RegFieldWrapper<"ppe_prelu_mult", 11, 16>;
def reg_PpePrelu : NPUReg50XX_RegisterWrapper<"ppe_prelu", 4, 0x9C, ["ppe_prelu_shift", "ppe_prelu_mult"], 0x00010000>;

// ppe_scale_hclamp:  int clamp - backwards compatible mode ; fp clamp - normal mode
def bf_PpeIntScaleHclamp : NPUReg50XX_RegFieldWrapper<"ppe_int_scale_hclamp", 32, 0, "SINT">;
def bf_PpeFpScaleHclamp : NPUReg50XX_RegFieldWrapper<"ppe_fp_scale_hclamp", 32, 0, "FP">;
def reg_PpeScaleHclamp : NPUReg50XX_RegisterWrapper<"ppe_scale_hclamp", 4, 0xA0, ["ppe_int_scale_hclamp", "ppe_fp_scale_hclamp"], 0x0, 1>;

// ppe_scale_lclamp: int clamp - backwards compatible mode ; fp clamp - normal mode
def bf_intPpeScaleLclamp : NPUReg50XX_RegFieldWrapper<"ppe_int_scale_lclamp", 32, 0, "SINT">;
def bf_fpPpeScaleLclamp : NPUReg50XX_RegFieldWrapper<"ppe_fp_scale_lclamp", 32, 0, "FP">;
def reg_PpeScaleLclamp : NPUReg50XX_RegisterWrapper<"ppe_scale_lclamp", 4, 0xA4, ["ppe_int_scale_lclamp", "ppe_fp_scale_lclamp"], 0x0, 1>;

// ppe_misc
def bf_PpeMode : NPUReg50XX_RegFieldWrapper<"ppe_mode", 2, 0>;
def bf_PpeRndFp : NPUReg50XX_RegFieldWrapper<"ppe_rnd_fp", 2, 2>;
def bf_PpeRndInt : NPUReg50XX_RegFieldWrapper<"ppe_rnd_int", 2, 4>;
def bf_PpeFp16Ftz : NPUReg50XX_RegFieldWrapper<"ppe_fp16_ftz", 1, 6>;
def bf_PpeFp16Clamp : NPUReg50XX_RegFieldWrapper<"ppe_fp16_clamp", 1, 7>;
def bf_PpeI32Convert : NPUReg50XX_RegFieldWrapper<"ppe_i32_convert", 2, 8>;
def bf_PpeSbDtype : NPUReg50XX_RegFieldWrapper<"ppe_sb_dtype", 2, 10>;
def bf_PpeMult2Mode: NPUReg50XX_RegFieldWrapper<"ppe_mult2_mode", 1, 12>;

def reg_PpeMisc : NPUReg50XX_RegisterWrapper<"ppe_misc", 4, 0xA8, ["ppe_mode" ,"ppe_rnd_fp" ,"ppe_rnd_int" ,"ppe_fp16_ftz", "ppe_fp16_clamp", "ppe_i32_convert", "ppe_sb_dtype", "ppe_mult2_mode"]>;

// ppe_fp_bias
def bf_PpeFpBias : NPUReg50XX_RegFieldWrapper<"ppe_fp_bias", 32, 0, "FP">;
def reg_PpeFpBias : NPUReg50XX_RegisterWrapper<"ppe_fp_bias", 4, 0xAC, ["ppe_fp_bias"]>;

// ppe_fp_scale
def bf_PpeFpScale : NPUReg50XX_RegFieldWrapper<"ppe_fp_scale", 32, 0, "FP">;
def reg_PpeFpScale : NPUReg50XX_RegisterWrapper<"ppe_fp_scale", 4, 0xB0, ["ppe_fp_scale"]>;

// ppe_fp_prelu
def bf_PpeFpPrelu : NPUReg50XX_RegFieldWrapper<"ppe_fp_prelu", 32, 0, "FP">;
def reg_PpeFpPrelu : NPUReg50XX_RegisterWrapper<"ppe_fp_prelu", 4, 0xB4, ["ppe_fp_prelu"]>;

// ppe_fp_cfg
def bf_PpeFpConvert : NPUReg50XX_RegFieldWrapper<"ppe_fp_convert", 3, 0>;
def bf_PpeFpBypass : NPUReg50XX_RegFieldWrapper<"ppe_fp_bypass", 1, 3>;
def bf_PpeBf16Round : NPUReg50XX_RegFieldWrapper<"ppe_bf16_round", 1, 4>;
def bf_PpeFpPreluEn : NPUReg50XX_RegFieldWrapper<"ppe_fp_prelu_en", 1, 5>;
//  Note: ppe prelu needs to be off by default, only turned on explicitly
def reg_PpeFpCfg : NPUReg50XX_RegisterWrapper<"ppe_fp_cfg", 4, 0xB8, ["ppe_fp_convert", "ppe_fp_bypass", "ppe_bf16_round", "ppe_fp_prelu_en"], 0x00000008>;

// odu_ac_base
def bf_AcBase : NPUReg50XX_RegFieldWrapper<"ac_base", 28, 4>;
def reg_OduAcBase : NPUReg50XX_RegisterWrapper<"odu_ac_base", 4, 0xBC, ["ac_base"]>;

// hwp_ctrl
def bf_HwpEn : NPUReg50XX_RegFieldWrapper<"hwp_en", 1, 0>;
def bf_HwpStatMode : NPUReg50XX_RegFieldWrapper<"hwp_stat_mode", 3, 1>;
def bf_HwpTimerEn : NPUReg50XX_RegFieldWrapper<"local_timer_en", 1, 4>;
def bf_HwpTimerRst : NPUReg50XX_RegFieldWrapper<"local_timer_rst", 1, 5>;
// rsvd : 9
def bf_HwpUniqueID : NPUReg50XX_RegFieldWrapper<"unique_ID", 16, 16>;

def reg_HwpCtrl : NPUReg50XX_RegisterWrapper<"hwp_ctrl", 4, 0xC0, ["hwp_en", "hwp_stat_mode", "local_timer_en", "local_timer_rst", "unique_ID"]>;

// hwp_cmx_mem_addr
def bf_HwpCmxMemAddr : NPUReg50XX_RegFieldWrapper<"hwp_cmx_mem_addr", 32, 0>;
def reg_HwpCmxMemAddr : NPUReg50XX_RegisterWrapper<"hwp_cmx_mem_addr", 4, 0xC4, ["hwp_cmx_mem_addr"]>;

// odu_cast_0
def bf_CastEnable0 : NPUReg50XX_RegFieldWrapper<"cast_enable0", 1, 0>;
def bf_CastOffset0 : NPUReg50XX_RegFieldWrapper<"cast_offset0", 28, 4>;
def reg_OduCast0 : NPUReg50XX_RegisterWrapper<"odu_cast0", 4, 0xC8, ["cast_enable0", "cast_offset0"]>;

// odu_cast_1
def bf_CastEnable1 : NPUReg50XX_RegFieldWrapper<"cast_enable1", 1, 0>;
def bf_CastOffset1 : NPUReg50XX_RegFieldWrapper<"cast_offset1", 28, 4>;
def reg_OduCast1 : NPUReg50XX_RegisterWrapper<"odu_cast1", 4, 0xCC, ["cast_enable1", "cast_offset1"]>;

// odu_cast_2
def bf_CastEnable2 : NPUReg50XX_RegFieldWrapper<"cast_enable2", 1, 0>;
def bf_CastOffset2 : NPUReg50XX_RegFieldWrapper<"cast_offset2", 28, 4>;
def reg_OduCast2 : NPUReg50XX_RegisterWrapper<"odu_cast2", 4, 0xD0, ["cast_enable2", "cast_offset2"]>;

// tensor2_start
def bf_Tensor2Start : NPUReg50XX_RegFieldWrapper<"tensor2_start", 32, 0>;
def reg_Tensor2Start : NPUReg50XX_RegisterWrapper<"tensor2_start", 4, 0xD4, ["tensor2_start"]>;

// ppe_lut_ptr
def bf_PpeLutPtr : NPUReg50XX_RegFieldWrapper<"ppe_lut_ptr", 16, 0>;
def bf_PpeLutPtrForce : NPUReg50XX_RegFieldWrapper<"ppe_lut_ptr_force", 1, 18>;
def reg_PpeLutPtr : NPUReg50XX_RegisterWrapper<"ppe_lut_ptr", 4, 0xD8, ["ppe_lut_ptr", "ppe_lut_ptr_force"]>;

// wl_tag
def bf_NvarTag : NPUReg50XX_RegFieldWrapper<"nvar_tag", 32, 0>;
def reg_NvarTag : NPUReg50XX_RegisterWrapper<"nvar_tag", 4, 0xDC, ["nvar_tag"]>;

// pallet

// pallet tb 0
def bf_PalletIndx0 : NPUReg50XX_RegFieldWrapper<"plt_idx_0", 16, 0>;
def bf_PalletIndx1 : NPUReg50XX_RegFieldWrapper<"plt_idx_1", 16, 16>;
def reg_PalletTbl0 : NPUReg50XX_RegisterWrapper<"pallet0", 4, 0xE0, ["plt_idx_0", "plt_idx_1"]>;

// pallet tb 1
def bf_PalletIndx2 : NPUReg50XX_RegFieldWrapper<"plt_idx_2", 16, 0>;
def bf_PalletIndx3 : NPUReg50XX_RegFieldWrapper<"plt_idx_3", 16, 16>;
def reg_PalletTbl1 : NPUReg50XX_RegisterWrapper<"pallet1", 4, 0xE4, ["plt_idx_2", "plt_idx_3"]>;

// pallet tb 2
def bf_PalletIndx4 : NPUReg50XX_RegFieldWrapper<"plt_idx_4", 16, 0>;
def bf_PalletIndx5 : NPUReg50XX_RegFieldWrapper<"plt_idx_5", 16, 16>;
def reg_PalletTbl2 : NPUReg50XX_RegisterWrapper<"pallet2", 4, 0xE8, ["plt_idx_4", "plt_idx_5"]>;

// pallet tb 3
def bf_PalletIndx6 : NPUReg50XX_RegFieldWrapper<"plt_idx_6", 16, 0>;
def bf_PalletIndx7 : NPUReg50XX_RegFieldWrapper<"plt_idx_7", 16, 16>;
def reg_PalletTbl3 : NPUReg50XX_RegisterWrapper<"pallet3", 4, 0xEC, ["plt_idx_6", "plt_idx_7"]>;

// pallet tb 4
def bf_PalletIndx8 : NPUReg50XX_RegFieldWrapper<"plt_idx_8", 16, 0>;
def bf_PalletIndx9 : NPUReg50XX_RegFieldWrapper<"plt_idx_9", 16, 16>;
def reg_PalletTbl4 : NPUReg50XX_RegisterWrapper<"pallet4", 4, 0xF0, ["plt_idx_8", "plt_idx_9"]>;

// pallet tb 5
def bf_PalletIndx10 : NPUReg50XX_RegFieldWrapper<"plt_idx_10", 16, 0>;
def bf_PalletIndx11 : NPUReg50XX_RegFieldWrapper<"plt_idx_11", 16, 16>;
def reg_PalletTbl5 : NPUReg50XX_RegisterWrapper<"pallet5", 4, 0xF4, ["plt_idx_10", "plt_idx_11"]>;

// pallet tb 6
def bf_PalletIndx12 : NPUReg50XX_RegFieldWrapper<"plt_idx_12", 16, 0>;
def bf_PalletIndx13 : NPUReg50XX_RegFieldWrapper<"plt_idx_13", 16, 16>;
def reg_PalletTbl6 : NPUReg50XX_RegisterWrapper<"pallet6", 4, 0xF8, ["plt_idx_12", "plt_idx_13"]>;

// pallet tb 7
def bf_PalletIndx14 : NPUReg50XX_RegFieldWrapper<"plt_idx_14", 16, 0>;
def bf_PalletIndx15 : NPUReg50XX_RegFieldWrapper<"plt_idx_15", 16, 16>;
def reg_PalletTbl7 : NPUReg50XX_RegisterWrapper<"pallet7", 4, 0xFC, ["plt_idx_14", "plt_idx_15"]>;

// se_addr1
def bf_SeAddr1 : NPUReg50XX_RegFieldWrapper<"se_addr1", 32, 0>;
def reg_SeAddr1 : NPUReg50XX_RegisterWrapper<"se_addr1", 4, 0x100, ["se_addr1"]>;

// sparsity_addr1
def bf_SparsityAddr1 : NPUReg50XX_RegFieldWrapper<"sparsity_addr1", 32, 0>;
def reg_SparsityAddr1 : NPUReg50XX_RegisterWrapper<"sparsity_addr1", 4, 0x104, ["sparsity_addr1"]>;

// se_addr2
def bf_SeAddr2 : NPUReg50XX_RegFieldWrapper<"se_addr2", 32, 0>;
def reg_SeAddr2 : NPUReg50XX_RegisterWrapper<"se_addr2", 4, 0x108, ["se_addr2"]>;

// sparsity_addr2
def bf_SparsityAddr2 : NPUReg50XX_RegFieldWrapper<"sparsity_addr2", 32, 0>;
def reg_SparsityAddr2 : NPUReg50XX_RegisterWrapper<"sparsity_addr2", 4, 0x10C, ["sparsity_addr2"]>;

// se_addr3
def bf_SeAddr3 : NPUReg50XX_RegFieldWrapper<"se_addr3", 32, 0>;
def reg_SeAddr3 : NPUReg50XX_RegisterWrapper<"se_addr3", 4, 0x110, ["se_addr3"]>;

// sparsity_addr3
def bf_SparsityAddr3 : NPUReg50XX_RegFieldWrapper<"sparsity_addr3", 32, 0>;
def reg_SparsityAddr3 : NPUReg50XX_RegisterWrapper<"sparsity_addr3", 4, 0x114, ["sparsity_addr3"]>;

// se_sp_size1
def bf_SpSegSize1 : NPUReg50XX_RegFieldWrapper<"se_sp_size1", 32, 0>;
def reg_SeSpSize1 : NPUReg50XX_RegisterWrapper<"se_sp_size1", 4, 0x118, ["se_sp_size1"]>;

// se_sp_size2
def bf_SpSegSize2 : NPUReg50XX_RegFieldWrapper<"se_sp_size2", 32, 0>;
def reg_SeSpSize2 : NPUReg50XX_RegisterWrapper<"se_sp_size2", 4, 0x11C, ["se_sp_size2"]>;

def bf_BarrierWaitMaskHi : NPUReg50XX_RegFieldWrapper<"barriers_wait_mask_hi_", 32, 0>;
def reg_BarrierWaitMaskHi : NPUReg50XX_RegisterWrapper<"barriers_wait_mask_hi_", 8, 0x120, ["barriers_wait_mask_hi_"]>;

def bf_BarrierWaitMaskLo : NPUReg50XX_RegFieldWrapper<"barriers_wait_mask_lo_", 64, 0>;
def reg_BarrierWaitMaskLo : NPUReg50XX_RegisterWrapper<"barriers_wait_mask_lo_", 8, 0x128, ["barriers_wait_mask_lo_"]>;

def bf_BarrierWaitPostHi : NPUReg50XX_RegFieldWrapper<"barriers_post_mask_hi_", 32, 0>;
def reg_BarrierWaitPostHi : NPUReg50XX_RegisterWrapper<"barriers_post_mask_hi_", 8, 0x130, ["barriers_post_mask_hi_"]>;

def bf_BarrierWaitPostLo : NPUReg50XX_RegFieldWrapper<"barriers_post_mask_lo_", 64, 0>;
def reg_BarrierWaitPostLo : NPUReg50XX_RegisterWrapper<"barriers_post_mask_lo_", 8, 0x138, ["barriers_post_mask_lo_"]>;

def bf_BarrierGroup : NPUReg50XX_RegFieldWrapper<"group_", 8, 0>;
def bf_BarrierMask : NPUReg50XX_RegFieldWrapper<"mask_", 8, 8>;
def reg_BarrierGroupMask : NPUReg50XX_RegisterWrapper<"barriers_group_mask_", 8, 0x140, ["group_", "mask_"]>;

def bf_StartAfter : NPUReg50XX_RegFieldWrapper<"start_after_", 32, 0>;
def bf_CleanAfter : NPUReg50XX_RegFieldWrapper<"clean_after_", 32, 32>;
def reg_BarriersSchedulingCfg : NPUReg50XX_RegisterWrapper<"barriers_sched_", 8, 0x148, ["start_after_", "clean_after_"]>;

def bf_Deprecated0Invariant : NPUReg50XX_RegFieldWrapper<"deprecated0_inv", 64, 0, "UINT">;
def reg_Deprecated0Invariant : NPUReg50XX_RegisterWrapper<"deprecated0_inv", 8, 0x150, ["deprecated0_inv"]>;

def bf_VariantCnt : NPUReg50XX_RegFieldWrapper<"variant_count_", 16, 0>;
def reg_VariantCnt : NPUReg50XX_RegisterWrapper<"variant_count_", 2, 0x158, ["variant_count_"]>;

def bf_ClusterInvariant : NPUReg50XX_RegFieldWrapper<"cluster_invariant_", 8, 0>;
def reg_ClusterInvariant : NPUReg50XX_RegisterWrapper<"cluster_invariant_", 1, 0x15A, ["cluster_invariant_"]>;

def bf_Deprecated1Invariant : NPUReg50XX_RegFieldWrapper<"deprecated1_inv", 16, 0, "UINT">;
def reg_Deprecated1Invariant : NPUReg50XX_RegisterWrapper<"deprecated1_inv", 2, 0x15B, ["deprecated1_inv"]>;

def bf_Pad3 : NPUReg50XX_RegFieldWrapper<"pad_3", 24, 0, "UINT">;
def reg_Pad3 : NPUReg50XX_RegisterWrapper<"pad_3", 3, 0x15D, ["pad_3"]>;

//
// DPUVariantRegisters
//

// invar_ptr
def bf_InvarPtr : NPUReg50XX_RegFieldWrapper<"invar_ptr", 16, 0>;
def bf_VarTag : NPUReg50XX_RegFieldWrapper<"var_tag", 16, 16>;
def reg_InvarPtr : NPUReg50XX_RegisterWrapper<"invar_ptr", 4, 0x0, ["invar_ptr", "var_tag"]>;

// workload_size0
def bf_WorkloadSizeX : NPUReg50XX_RegFieldWrapper<"workload_size_x", 14, 0>;
def bf_WorkloadSizeY : NPUReg50XX_RegFieldWrapper<"workload_size_y", 14, 14>;
def reg_WorkloadSize0 : NPUReg50XX_RegisterWrapper<"workload_size0", 4, 0x4, ["workload_size_x", "workload_size_y"]>;

// workload_size1
def bf_WorkloadSizeZ : NPUReg50XX_RegFieldWrapper<"workload_size_z", 14, 0>;
def bf_PadCountUp : NPUReg50XX_RegFieldWrapper<"pad_count_up", 3, 14>;
def bf_PadCountLeft : NPUReg50XX_RegFieldWrapper<"pad_count_left", 3, 17>;
def bf_PadCountDown : NPUReg50XX_RegFieldWrapper<"pad_count_down", 3, 20>;
def bf_PadCountRight : NPUReg50XX_RegFieldWrapper<"pad_count_right", 3, 23>;
def reg_WorkloadSize1 : NPUReg50XX_RegisterWrapper<"workload_size1", 4, 0x8,
    ["workload_size_z", "pad_count_up", "pad_count_left", "pad_count_down", "pad_count_right"]>;

// workload_start0
def bf_WorkloadStartX : NPUReg50XX_RegFieldWrapper<"workload_start_x", 14, 0>;
def bf_WorkloadStartY : NPUReg50XX_RegFieldWrapper<"workload_start_y", 14, 14>;
def reg_WorkloadStart0 : NPUReg50XX_RegisterWrapper<"workload_start0", 4, 0xC, ["workload_start_x", "workload_start_y"]>;

// workload_start1
def bf_WorkloadStartZ : NPUReg50XX_RegFieldWrapper<"workload_start_z", 14, 0>;
def reg_WorkloadStart1 : NPUReg50XX_RegisterWrapper<"workload_start1", 4, 0x10, ["workload_start_z"]>;

// offset_addr
def bf_NthwNtk : NPUReg50XX_RegFieldWrapper<"nthw_ntk", 2, 0>;
def bf_BinCfg : NPUReg50XX_RegFieldWrapper<"bin_cfg", 1, 2>;
def bf_ConvCond : NPUReg50XX_RegFieldWrapper<"conv_cond", 1, 3>;
def bf_DenseSe : NPUReg50XX_RegFieldWrapper<"dense_se", 1, 4>;
def bf_IdxQuad : NPUReg50XX_RegFieldWrapper<"idx_quad", 1, 5>;
def bf_SwizzleKeyOffsetAddr : NPUReg50XX_RegFieldWrapper<"swizzle_key_offset", 3, 6>;
def bf_IduMrmClkEn : NPUReg50XX_RegFieldWrapper<"idu_mrm_clk_en", 1, 9>;
def bf_OduClkEn : NPUReg50XX_RegFieldWrapper<"odu_clk_en", 1, 10>;
def bf_MpeClkEn : NPUReg50XX_RegFieldWrapper<"mpe_clk_en", 1, 11>;
def bf_PpeClkEn : NPUReg50XX_RegFieldWrapper<"ppe_clk_en", 1, 12>;
def bf_OduStatEn : NPUReg50XX_RegFieldWrapper<"odu_stat_en", 1, 13>;
def bf_IduStatEn : NPUReg50XX_RegFieldWrapper<"idu_stat_en", 1, 14>;
def bf_NocClkEn : NPUReg50XX_RegFieldWrapper<"noc_clk_en", 1, 15>;
def bf_OduStatClrMode : NPUReg50XX_RegFieldWrapper<"odu_stat_clr_mode", 1, 16>;
def bf_IduStatClrMode : NPUReg50XX_RegFieldWrapper<"idu_stat_clr_mode", 1, 17>;
def bf_SeOnlyEn : NPUReg50XX_RegFieldWrapper<"se_only_en", 1, 18>;
def bf_ShaveL2CacheEn : NPUReg50XX_RegFieldWrapper<"shave_l2_cache_en", 1, 19>;
def bf_IduDbgEn : NPUReg50XX_RegFieldWrapper<"idu_dbg_en", 2, 20>;
def bf_SbReadEn : NPUReg50XX_RegFieldWrapper<"sb_read_en", 1, 22>;
def bf_Tensor2ActDense : NPUReg50XX_RegFieldWrapper<"tensor2_act_dense", 1, 23>;
def bf_WtSwizzleKey : NPUReg50XX_RegFieldWrapper<"wt_swizzle_key", 3, 27>;
def bf_WtSwizzleSel : NPUReg50XX_RegFieldWrapper<"wt_swizzle_sel", 1, 30>;
//  Note: wt_swizzle_sel needs to be on by default to match NNRT GF behaviour
def bf_GifClkEn : NPUReg50XX_RegFieldWrapper<"gif_clk_en", 1, 31>;
def reg_OffsetAddr : NPUReg50XX_RegisterWrapper<"offset_addr", 4, 0x14,
    ["nthw_ntk", "bin_cfg", "conv_cond", "dense_se", "idx_quad", "swizzle_key_offset", "idu_mrm_clk_en", "odu_clk_en",
    "mpe_clk_en", "ppe_clk_en", "odu_stat_en", "idu_stat_en", "noc_clk_en", "odu_stat_clr_mode", "idu_stat_clr_mode", "se_only_en",
    "shave_l2_cache_en", "idu_dbg_en", "sb_read_en", "tensor2_act_dense", "wt_swizzle_key", "wt_swizzle_sel", "gif_clk_en"],
    0x40000000>;

// hwp_wload_id
def bf_HwpWLoadId : NPUReg50XX_RegFieldWrapper<"hwp_wload_id", 16, 0>;
def reg_HwpWloadId : NPUReg50XX_RegisterWrapper<"hwp_wload_id", 4, 0x18, ["hwp_wload_id"]>;

// var_cfg
def bf_InvarLutRdEn : NPUReg50XX_RegFieldWrapper<"invar_lut_rd_en", 1, 8>;
def bf_InvarLineCntEn : NPUReg50XX_RegFieldWrapper<"invar_line_cnt_en", 1, 9>;
def bf_InvarLineCnt : NPUReg50XX_RegFieldWrapper<"invar_line_cnt", 4, 10>;
def bf_InvarPtrForce : NPUReg50XX_RegFieldWrapper<"invar_lptr_force", 1, 14>;
def bf_NextSramJobValid : NPUReg50XX_RegFieldWrapper<"next_sram_job_valid", 1, 15>;
def bf_NextSramJobAddr : NPUReg50XX_RegFieldWrapper<"next_sram_job_addr", 16, 16>;
def reg_VarCfg : NPUReg50XX_RegisterWrapper<"var_cfg", 4, 0x1C,
        ["invar_lut_rd_en", "invar_line_cnt_en", "invar_line_cnt", "invar_lptr_force", "next_sram_job_valid", "next_sram_job_addr"]>;

// cbarrier_lo
def bf_CBarrierLo : NPUReg50XX_RegFieldWrapper<"cbarrier_lo", 64, 0>;
def reg_CBarrierLo : NPUReg50XX_RegisterWrapper<"cbarrier_lo", 8, 0x20, ["cbarrier_lo"]>;

// cbarrier_hi
def bf_CBarrierHi : NPUReg50XX_RegFieldWrapper<"cbarrier_hi", 32, 0>;
def reg_CBarrierHi : NPUReg50XX_RegisterWrapper<"cbarrier_hi", 8, 0x28, ["cbarrier_hi"]>;

// pbarrier_lo
def bf_PBarrierLo : NPUReg50XX_RegFieldWrapper<"pbarrier_lo", 64, 0>;
def reg_PBarrierLo : NPUReg50XX_RegisterWrapper<"pbarrier_lo", 8, 0x30, ["pbarrier_lo"]>;

// pbarrier_hi
def bf_PBarrierHi : NPUReg50XX_RegFieldWrapper<"pbarrier_hi", 32, 0>;
def reg_PBarrierHi : NPUReg50XX_RegisterWrapper<"pbarrier_hi", 8, 0x38, ["pbarrier_hi"]>;

// halo_region0..5A
def bf_SpAdrOffset : NPUReg50XX_RegFieldWrapper<"sp_adr_offset", 22, 0, "SINT">;
def bf_TileSelected : NPUReg50XX_RegFieldWrapper<"tile_select", 7, 22>;
def bf_RsvdA : NPUReg50XX_RegFieldWrapper<"rsvdA", 2, 29>;
def bf_Enable : NPUReg50XX_RegFieldWrapper<"enable", 1, 31>;
def reg_HaloRegion0A : NPUReg50XX_RegisterWrapper<"halo_region0A", 4, 0x40, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion1A : NPUReg50XX_RegisterWrapper<"halo_region1A", 4, 0x50, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion2A : NPUReg50XX_RegisterWrapper<"halo_region2A", 4, 0x60, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion3A : NPUReg50XX_RegisterWrapper<"halo_region3A", 4, 0x70, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion4A : NPUReg50XX_RegisterWrapper<"halo_region4A", 4, 0x80, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;
def reg_HaloRegion5A : NPUReg50XX_RegisterWrapper<"halo_region5A", 4, 0x90, ["sp_adr_offset", "tile_select", "rsvdA", "enable"]>;

// halo_region0..5B
def bf_AcAdrOffset : NPUReg50XX_RegFieldWrapper<"ac_adr_offset", 22, 0, "SINT">;
def bf_TargetWidthLsb : NPUReg50XX_RegFieldWrapper<"target_width_lsb", 10, 22>;
def reg_HaloRegion0B : NPUReg50XX_RegisterWrapper<"halo_region0B", 4, 0x44, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion1B : NPUReg50XX_RegisterWrapper<"halo_region1B", 4, 0x54, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion2B : NPUReg50XX_RegisterWrapper<"halo_region2B", 4, 0x64, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion3B : NPUReg50XX_RegisterWrapper<"halo_region3B", 4, 0x74, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion4B : NPUReg50XX_RegisterWrapper<"halo_region4B", 4, 0x84, ["ac_adr_offset", "target_width_lsb"]>;
def reg_HaloRegion5B : NPUReg50XX_RegisterWrapper<"halo_region5B", 4, 0x94, ["ac_adr_offset", "target_width_lsb"]>;

// halo_region0..5C
def bf_BeginX : NPUReg50XX_RegFieldWrapper<"begin_x", 13, 0>;
def bf_BeginY : NPUReg50XX_RegFieldWrapper<"begin_y", 13, 13>;
def bf_TargetWidthMsb : NPUReg50XX_RegFieldWrapper<"target_width_msb", 4, 26>;
def reg_HaloRegion0C : NPUReg50XX_RegisterWrapper<"halo_region0C", 4, 0x48, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion1C : NPUReg50XX_RegisterWrapper<"halo_region1C", 4, 0x58, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion2C : NPUReg50XX_RegisterWrapper<"halo_region2C", 4, 0x68, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion3C : NPUReg50XX_RegisterWrapper<"halo_region3C", 4, 0x78, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion4C : NPUReg50XX_RegisterWrapper<"halo_region4C", 4, 0x88, ["begin_x", "begin_y", "target_width_msb"]>;
def reg_HaloRegion5C : NPUReg50XX_RegisterWrapper<"halo_region5C", 4, 0x98, ["begin_x", "begin_y", "target_width_msb"]>;

// halo_region0..5D
def bf_EndX : NPUReg50XX_RegFieldWrapper<"end_x", 13, 0>;
def bf_EndY : NPUReg50XX_RegFieldWrapper<"end_y", 13, 13>;
def bf_RsvdD : NPUReg50XX_RegFieldWrapper<"rsvdD", 4, 26>;
def reg_HaloRegion0D : NPUReg50XX_RegisterWrapper<"halo_region0D", 4, 0x4C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion1D : NPUReg50XX_RegisterWrapper<"halo_region1D", 4, 0x5C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion2D : NPUReg50XX_RegisterWrapper<"halo_region2D", 4, 0x6C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion3D : NPUReg50XX_RegisterWrapper<"halo_region3D", 4, 0x7C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion4D : NPUReg50XX_RegisterWrapper<"halo_region4D", 4, 0x8C, ["end_x", "end_y", "rsvdD"]>;
def reg_HaloRegion5D : NPUReg50XX_RegisterWrapper<"halo_region5D", 4, 0x9C, ["end_x", "end_y", "rsvdD"]>;

// dpu_cfg
def bf_WorkloadStartOdu : NPUReg50XX_RegFieldWrapper<"workload_start_odu", 1, 0>;
def bf_WorkloadStartIdu : NPUReg50XX_RegFieldWrapper<"workload_start_idu", 1, 1>;
def bf_WorkloadPrmSel : NPUReg50XX_RegFieldWrapper<"workload_prm_sel", 1, 2>;
def bf_WorkloadValid : NPUReg50XX_RegFieldWrapper<"workload_valid", 1, 3>;
def bf_WorkloadShadOdu : NPUReg50XX_RegFieldWrapper<"workload_shad_odu", 1, 4>;
def bf_WorkloadShadIdu : NPUReg50XX_RegFieldWrapper<"workload_shad_idu", 1, 5>;
def bf_WorkloadIduAutoUpd0 : NPUReg50XX_RegFieldWrapper<"workload_idu_auto_upd_0", 1, 6>;
def bf_WorkloadIduAutoUpd1 : NPUReg50XX_RegFieldWrapper<"workload_idu_auto_upd_1", 1, 7>;
def bf_WorkloadOduAutoUpd : NPUReg50XX_RegFieldWrapper<"workload_odu_auto_upd", 1, 8>;
def bf_CfgReserved0 : NPUReg50XX_RegFieldWrapper<"cfg_Reserved_0", 1, 9>;
def bf_CfgReserved1 : NPUReg50XX_RegFieldWrapper<"cfg_Reserved_1", 1, 10>;
def bf_CfgReserved2 : NPUReg50XX_RegFieldWrapper<"cfg_Reserved_2", 1, 11>;
def bf_RstCtxtNew : NPUReg50XX_RegFieldWrapper<"rst_ctxt_new", 1, 12>;
def bf_CfgReserved3 : NPUReg50XX_RegFieldWrapper<"cfg_Reserved_3", 1, 13>;
def bf_CfgReserved4 : NPUReg50XX_RegFieldWrapper<"cfg_Reserved_4", 1, 14>;
def bf_OduStatClr : NPUReg50XX_RegFieldWrapper<"odu_stat_clr", 1, 15>;
def bf_IduStatClr : NPUReg50XX_RegFieldWrapper<"idu_stat_clr", 1, 16>;
def bf_CfgReserved5 : NPUReg50XX_RegFieldWrapper<"cfg_Reserved_5", 1, 17>;
def bf_CfgReserved6 : NPUReg50XX_RegFieldWrapper<"cfg_Reserved_6", 14, 18>;
def reg_DpuCfg : NPUReg50XX_RegisterWrapper<"dpu_cfg", 4, 0xA0,
    ["workload_start_odu", "workload_start_idu", "workload_prm_sel", "workload_valid", "workload_shad_odu",
    "workload_shad_idu", "workload_idu_auto_upd_0", "workload_idu_auto_upd_1", "workload_odu_auto_upd",
    "cfg_Reserved_0", "cfg_Reserved_1", "cfg_Reserved_2", "rst_ctxt_new", "cfg_Reserved_3", "cfg_Reserved_4",
    "odu_stat_clr", "idu_stat_clr", "cfg_Reserved_5", "cfg_Reserved_6"], 0x00000004>;

// te_beg0
def bf_TeBegY : NPUReg50XX_RegFieldWrapper<"te_beg_y", 13, 0>;
def bf_TeBegZ : NPUReg50XX_RegFieldWrapper<"te_beg_z", 13, 13>;
def reg_TeBeg0 : NPUReg50XX_RegisterWrapper<"te_beg0", 4, 0xA4, ["te_beg_y", "te_beg_z"]>;

// te_beg1
def bf_TeBegX : NPUReg50XX_RegFieldWrapper<"te_beg_x", 13, 0>;
def reg_TeBeg1 : NPUReg50XX_RegisterWrapper<"te_beg1", 4, 0xA8, ["te_beg_x"]>;

// te_end0
def bf_TeEndY : NPUReg50XX_RegFieldWrapper<"te_end_y", 13, 0>;
def bf_TeEndZ : NPUReg50XX_RegFieldWrapper<"te_end_z", 13, 13>;
def reg_TeEnd0 : NPUReg50XX_RegisterWrapper<"te_end0", 4, 0xAC, ["te_end_y", "te_end_z"]>;

// te_end1
def bf_TeEndX : NPUReg50XX_RegFieldWrapper<"te_end_x", 13, 0>;
def reg_TeEnd1 : NPUReg50XX_RegisterWrapper<"te_end1", 4, 0xB0, ["te_end_x"]>;

// weight_size
def bf_WeightSize : NPUReg50XX_RegFieldWrapper<"weight_size", 32, 0>;
def reg_WeightSize : NPUReg50XX_RegisterWrapper<"weight_size", 4, 0xB4, ["weight_size"]>;

// weight_num
def bf_WeighNum : NPUReg50XX_RegFieldWrapper<"weight_num", 32, 0>;
def reg_WeightNum : NPUReg50XX_RegisterWrapper<"weight_num", 4, 0xB8, ["weight_num"]>;

// weight_start
def bf_WeightStart : NPUReg50XX_RegFieldWrapper<"weight_start", 32, 0>;
def reg_WeightStart : NPUReg50XX_RegisterWrapper<"weight_start", 4, 0xBC, ["weight_start"]>;

def bf_Invariant : NPUReg50XX_RegFieldWrapper<"invariant_", 64, 0>;
def reg_Invariant : NPUReg50XX_RegisterWrapper<"invariant_", 8, 0xC0, ["invariant_"]>;

def bf_InvariantIndex : NPUReg50XX_RegFieldWrapper<"invariant_index_", 32, 0>;
def reg_InvariantIndex : NPUReg50XX_RegisterWrapper<"invariant_index_", 4, 0xC8, ["invariant_index_"]>;

// uint8_t deprecated_[13]
def bf_Deprecated0Variant : NPUReg50XX_RegFieldWrapper<"deprecated0_var", 64, 0, "UINT">;
def reg_Deprecated0Variant : NPUReg50XX_RegisterWrapper<"deprecated0_var", 8, 0xCC, ["deprecated0_var"]>;

def bf_Deprecated1Variant : NPUReg50XX_RegFieldWrapper<"deprecated1_var", 40, 0, "UINT">;
def reg_Deprecated1Variant : NPUReg50XX_RegisterWrapper<"deprecated1_var", 5, 0xD4, ["deprecated1_var"]>;

// uint8_t pad_[20]
def bf_Pad_7_1 : NPUReg50XX_RegFieldWrapper<"pad_7_1", 56, 0, "UINT">;
def reg_Pad_7_1 : NPUReg50XX_RegisterWrapper<"pad_7_1", 7, 0xD9, ["pad_7_1"]>;

//
// RegMapped Specialization
//

class NPUReg50XX_RegMappedWrapper<string name = "", list<string> regsTypes = []> :
    VPURegMapped_RegMappedWrapper<NPUReg50XX_Dialect, "vpux::NPUReg50XX::", name, regsTypes> {

}

// DPUInvariantRegister
def reg_mapped_DpuInvariantRegister : NPUReg50XX_RegMappedWrapper<"DpuInvariantRegister",
    ["cmx_slice0_low_addr", "cmx_slice1_low_addr", "cmx_slice2_low_addr", "cmx_slice3_low_addr", "cmx_slice_size", "se_addr", "sparsity_addr", "se_size",
    "z_config", "kernel_pad_cfg", "tensor_size0", "tensor_size1", "tensor_start", "tensor_mode", "elops_sparsity_addr", "elops_se_addr",
    "elops_wload", "act_offset0", "act_offset1", "act_offset2", "act_offset3", "base_offset_a", "base_offset_b", "wt_offset",
    "odu_cfg", "odu_be_size", "odu_be_cnt", "odu_se_size",
    "te_dim0", "te_dim1", "pt_base", "sp_base",
    "mpe_cfg", "mpe_bus_data_sel", "elop_scale", "ppe_cfg", "ppe_bias", "ppe_scale", "ppe_scale_ctrl",
    "ppe_prelu", "ppe_scale_hclamp", "ppe_scale_lclamp", "ppe_misc", "ppe_fp_bias", "ppe_fp_scale", "ppe_fp_prelu",
    "ppe_fp_cfg", "odu_ac_base", "hwp_ctrl", "hwp_cmx_mem_addr", "odu_cast0", "odu_cast1", "odu_cast2", "tensor2_start", "ppe_lut_ptr", "nvar_tag",
    "pallet0", "pallet1", "pallet2", "pallet3", "pallet4", "pallet5", "pallet6", "pallet7",
    "se_addr1", "sparsity_addr1", "se_addr2", "sparsity_addr2", "se_addr3", "sparsity_addr3", "se_sp_size1", "se_sp_size2",
    "barriers_wait_mask_hi_", "barriers_wait_mask_lo_", "barriers_post_mask_hi_", "barriers_post_mask_lo_", "barriers_group_mask_",
    "barriers_sched_", "deprecated0_inv", "variant_count_", "cluster_invariant_", "deprecated1_inv", "pad_3"
    ]>;

// DPUVariantRegister
def reg_mapped_DpuVariantRegister : NPUReg50XX_RegMappedWrapper<"DpuVariantRegister",
    ["invar_ptr", "workload_size0", "workload_size1", "workload_start0", "workload_start1", "offset_addr", "hwp_wload_id",
    "var_cfg", "cbarrier_lo", "cbarrier_hi", "pbarrier_lo", "pbarrier_hi",
    "halo_region0A", "halo_region0B", "halo_region0C", "halo_region0D",
    "halo_region1A", "halo_region1B", "halo_region1C", "halo_region1D",
    "halo_region2A", "halo_region2B", "halo_region2C", "halo_region2D",
    "halo_region3A", "halo_region3B", "halo_region3C", "halo_region3D",
    "halo_region4A", "halo_region4B", "halo_region4C", "halo_region4D",
    "halo_region5A", "halo_region5B", "halo_region5C", "halo_region5D",
    "dpu_cfg", "te_beg0", "te_beg1", "te_end0", "te_end1",
    "weight_size", "weight_num", "weight_start",
    "invariant_", "invariant_index_", "deprecated0_var", "deprecated1_var", "pad_7_1"
    ]>;

// DMADescriptor

//
//  dma_64_bit_word_0
//
def bf_dma_watermark : NPUReg50XX_RegFieldWrapper<"dma_watermark", 1, 0>;                                  // Watermark to indicate that the job has completed
def reg_dma_watermark : NPUReg50XX_RegisterWrapper<"dma_watermark", 1, 0x0, ["dma_watermark"]>;

def bf_dma_link_address : NPUReg50XX_RegFieldWrapper<"dma_link_address", 48, 0>;                           // Pointer to the next element in linked list
def reg_dma_link_address : NPUReg50XX_RegisterWrapper<"dma_link_address", 8, 0x0, ["dma_link_address"]>;

def bf_dma_lra : NPUReg50XX_RegFieldWrapper<"dma_lra", 1, 63>;                                             // Link Relative Address. Base address fetched from LBA_ADDR
def reg_dma_lra : NPUReg50XX_RegisterWrapper<"dma_lra", 8, 0x0, ["dma_lra"]>;

//
//  dma_64_bit_word_1
//

def bf_dma_lba_addr : NPUReg50XX_RegFieldWrapper<"dma_lba_addr", 32, 0>;       // CMX address to the location of the Base Address for Relative Addressing options
def reg_dma_lba_addr : NPUReg50XX_RegisterWrapper<"dma_lba_addr", 4, 0x8,     // 8 dec
    ["dma_lba_addr"]>;

def bf_dma_src_aub : NPUReg50XX_RegFieldWrapper<"dma_src_aub", 8, 0>;          // Source AXI user bits
def reg_dma_src_aub : NPUReg50XX_RegisterWrapper<"dma_src_aub", 1, 0xC,        // 12 dec
    ["dma_src_aub"]>;

def bf_dma_dst_aub : NPUReg50XX_RegFieldWrapper<"dma_dst_aub", 8, 0>;          // Destination AXI user bits
def reg_dma_dst_aub : NPUReg50XX_RegisterWrapper<"dma_dst_aub", 1, 0xE,        // 14 dec
    ["dma_dst_aub"]>;

//
//  dma_64_bit_word_2
//

def bf_dma_cfg_fields_num_dim : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_num_dim", 3, 0>;
def bf_dma_cfg_fields_int_en : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_int_en", 1, 3>;
def bf_dma_cfg_fields_int_id : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_int_id", 8, 4>;
def bf_dma_cfg_fields_src_burst_length : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_src_burst_length", 4, 12>;
def bf_dma_cfg_fields_dst_burst_length : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_dst_burst_length", 4, 16>;
def bf_dma_cfg_fields_arb_qos : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_arb_qos", 8, 20>;
def bf_dma_cfg_fields_ord : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_ord", 1, 28>;
def bf_dma_cfg_fields_barrier_en : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_barrier_en", 1, 29>;
def bf_dma_cfg_fields_memset_en : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_memset_en", 1, 30>;
def bf_dma_cfg_fields_atp_en : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_atp_en", 1, 31>;
def bf_dma_cfg_fields_watermark_en : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_watermark_en", 1, 32>;
def bf_dma_cfg_fields_rwf_en : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_rwf_en", 1, 33>;
def bf_dma_cfg_fields_rws_en : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_rws_en", 1, 34>;
def bf_dma_cfg_fields_src_list_cfg : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_src_list_cfg", 2, 35>;
def bf_dma_cfg_fields_dst_list_cfg : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_dst_list_cfg", 2, 37>;
def bf_dma_cfg_fields_conversion_cfg : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_conversion_cfg", 3, 39>;
def bf_dma_cfg_fields_acceleration_cfg : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_acceleration_cfg", 2, 42>;
def bf_dma_cfg_fields_tile4_cfg : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_tile4_cfg", 2, 44>;
def bf_dma_cfg_fields_axi_user_bits_cfg : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_axi_user_bits_cfg", 2, 46>;
def bf_dma_cfg_fields_hwp_id_en : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_hwp_id_en", 1, 48>;
def bf_dma_cfg_fields_hwp_id : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_hwp_id", 12, 49>;
def bf_dma_cfg_fields_hwp_skip : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_hwp_skip", 1, 61>;
def bf_dma_cfg_fields_reserved : NPUReg50XX_RegFieldWrapper<"dma_cfg_fields_reserved", 2, 62>;

def reg_dma_cfg_fields : NPUReg50XX_RegisterWrapper<"dma_cfg_fields", 8, 0x10,  // 16 dec
    ["dma_cfg_fields_num_dim", "dma_cfg_fields_int_en", "dma_cfg_fields_int_id", "dma_cfg_fields_src_burst_length", "dma_cfg_fields_dst_burst_length",
    "dma_cfg_fields_arb_qos", "dma_cfg_fields_ord", "dma_cfg_fields_barrier_en", "dma_cfg_fields_memset_en", "dma_cfg_fields_atp_en",
    "dma_cfg_fields_watermark_en", "dma_cfg_fields_rwf_en", "dma_cfg_fields_rws_en", "dma_cfg_fields_src_list_cfg", "dma_cfg_fields_dst_list_cfg",
    "dma_cfg_fields_conversion_cfg", "dma_cfg_fields_acceleration_cfg", "dma_cfg_fields_tile4_cfg", "dma_cfg_fields_axi_user_bits_cfg",
    "dma_cfg_fields_hwp_id_en", "dma_cfg_fields_hwp_id", "dma_cfg_fields_hwp_skip", "dma_cfg_fields_reserved"]>;

//
//  dma_64_bit_word_3
//

def bf_dma_remote_width_fetch : NPUReg50XX_RegFieldWrapper<"dma_remote_width_fetch", 32, 0>;
def reg_dma_remote_width_fetch : NPUReg50XX_RegisterWrapper<"dma_remote_width_fetch", 4, 0x18, // 24 dec
    ["dma_remote_width_fetch"]>;

def bf_dma_width_src : NPUReg50XX_RegFieldWrapper<"dma_width_src", 32, 0>;
def bf_dma_width_dst : NPUReg50XX_RegFieldWrapper<"dma_width_dst", 32, 32>;

// overlap because fields dma_src_width and dma_rwf_addr are placed at the same memory address
def reg_dma_width : NPUReg50XX_RegisterWrapper<"dma_width", 8, 0x18, // 24 dec
    ["dma_width_src", "dma_width_dst"]>;

//
//  dma_64_bit_word_4: compress, decompress and WPREP regs are placed at the same memory address
//

//
//  dma_64_bit_word_4_compress
//

def bf_dma_acc_info_compress_dtype : NPUReg50XX_RegFieldWrapper<"dma_acc_info_compress_dtype", 2, 0>;
def bf_dma_acc_info_compress_reserved1 : NPUReg50XX_RegFieldWrapper<"dma_acc_info_compress_reserved1", 1, 2>;
def bf_dma_acc_info_compress_sparse : NPUReg50XX_RegFieldWrapper<"dma_acc_info_compress_sparse", 1, 3>;
def bf_dma_acc_info_compress_bitc_en : NPUReg50XX_RegFieldWrapper<"dma_acc_info_compress_bitc_en", 1, 4>;
def bf_dma_acc_info_compress_z : NPUReg50XX_RegFieldWrapper<"dma_acc_info_compress_z", 10, 5>;
def bf_dma_acc_info_compress_bitmap_buf_sz : NPUReg50XX_RegFieldWrapper<"dma_acc_info_compress_bitmap_buf_sz", 19, 15>;
def bf_dma_acc_info_compress_reserved2 : NPUReg50XX_RegFieldWrapper<"dma_acc_info_compress_reserved2", 3, 34>;
def bf_dma_acc_info_compress_bitmap_base_addr : NPUReg50XX_RegFieldWrapper<"dma_acc_info_compress_bitmap_base_addr", 27, 37>;

def reg_dma_acc_info_compress : NPUReg50XX_RegisterWrapper<"dma_acc_info_compress", 8, 0x20, // 32 dec
    ["dma_acc_info_compress_dtype", "dma_acc_info_compress_reserved1", "dma_acc_info_compress_sparse",
    "dma_acc_info_compress_bitc_en", "dma_acc_info_compress_z", "dma_acc_info_compress_bitmap_buf_sz",
    "dma_acc_info_compress_reserved2", "dma_acc_info_compress_bitmap_base_addr"]>;

//
//  dma_64_bit_word_4_decompress
//

def bf_dma_acc_info_decompress_dtype : NPUReg50XX_RegFieldWrapper<"dma_acc_info_decompress_dtype", 2, 0>;
def bf_dma_acc_info_decompress_reserved1 : NPUReg50XX_RegFieldWrapper<"dma_acc_info_decompress_reserved1", 1, 2>;
def bf_dma_acc_info_decompress_sparse : NPUReg50XX_RegFieldWrapper<"dma_acc_info_decompress_sparse", 1, 3>;
def bf_dma_acc_info_decompress_bitc_en : NPUReg50XX_RegFieldWrapper<"dma_acc_info_decompress_bitc_en", 1, 4>;
def bf_dma_acc_info_decompress_z : NPUReg50XX_RegFieldWrapper<"dma_acc_info_decompress_z", 10, 5>;
def bf_dma_acc_info_decompress_bitmap_buf_sz : NPUReg50XX_RegFieldWrapper<"dma_acc_info_decompress_bitmap_buf_sz", 19, 15>;
def bf_dma_acc_info_decompress_reserved2 : NPUReg50XX_RegFieldWrapper<"dma_acc_info_decompress_reserved2", 3, 34>;
def bf_dma_acc_info_decompress_bitmap_base_addr : NPUReg50XX_RegFieldWrapper<"dma_acc_info_decompress_bitmap_base_addr", 27, 37>;

def reg_dma_acc_info_decompress_reg : NPUReg50XX_RegisterWrapper<"dma_acc_info_decompress", 8, 0x20,  // 32 dec
    ["dma_acc_info_decompress_dtype", "dma_acc_info_decompress_reserved1",
    "dma_acc_info_decompress_sparse", "dma_acc_info_decompress_bitc_en", "dma_acc_info_decompress_z",
    "dma_acc_info_decompress_bitmap_buf_sz", "dma_acc_info_decompress_reserved2", "dma_acc_info_decompress_bitmap_base_addr"]>;

//
//  dma_64_bit_word_4_w_prep
//

def bf_dma_acc_info_w_prep_dtype : NPUReg50XX_RegFieldWrapper<"dma_acc_info_w_prep_dtype", 2, 0>;
def bf_dma_acc_info_w_prep_reserved1 : NPUReg50XX_RegFieldWrapper<"dma_acc_info_w_prep_reserved1", 1, 2>;
def bf_dma_acc_info_w_prep_sparse : NPUReg50XX_RegFieldWrapper<"dma_acc_info_w_prep_sparse", 2, 3>;
def bf_dma_acc_info_w_prep_zeropoint : NPUReg50XX_RegFieldWrapper<"dma_acc_info_w_prep_zeropoint", 8, 5>;
def bf_dma_acc_info_w_prep_ic : NPUReg50XX_RegFieldWrapper<"dma_acc_info_w_prep_ic", 14, 13>;
def bf_dma_acc_info_w_prep_filtersize : NPUReg50XX_RegFieldWrapper<"dma_acc_info_w_prep_filtersize", 7, 27>;
def bf_dma_acc_info_w_prep_reserved2 : NPUReg50XX_RegFieldWrapper<"dma_acc_info_w_prep_reserved2", 3, 34>;
def bf_dma_acc_info_w_prep_bitmap_base_addr : NPUReg50XX_RegFieldWrapper<"dma_acc_info_w_prep_bitmap_base_addr", 27, 37>;

def reg_dma_acc_info_w_prep_reg : NPUReg50XX_RegisterWrapper<"dma_acc_info_w_prep", 8, 0x20,  // 32 dec
    ["dma_acc_info_w_prep_dtype", "dma_acc_info_w_prep_reserved1", "dma_acc_info_w_prep_sparse",
    "dma_acc_info_w_prep_zeropoint", "dma_acc_info_w_prep_ic", "dma_acc_info_w_prep_filtersize",
    "dma_acc_info_w_prep_reserved2", "dma_acc_info_w_prep_bitmap_base_addr"]>;

//
//  dma_64_bit_word_4_acc_info_mem_set
//

def bf_dma_mset_data : NPUReg50XX_RegFieldWrapper<"dma_mset_data", 32, 0>;

def reg_dma_mset_data : NPUReg50XX_RegisterWrapper<"dma_mset_data", 4, 0x20,  // 32 dec
    ["dma_mset_data"]>;

//
//  dma_64_bit_word_5
//

def bf_dma_src : NPUReg50XX_RegFieldWrapper<"dma_src", 48, 0>;
def bf_dma_sra : NPUReg50XX_RegFieldWrapper<"dma_sra", 1, 63>;

def reg_dma_src_addr : NPUReg50XX_RegisterWrapper<"dma_src_addr", 8, 0x28,  // 40 dec
    ["dma_src", "dma_sra"]>;

//
//  dma_64_bit_word_6
//

def bf_dma_dst : NPUReg50XX_RegFieldWrapper<"dma_dst", 48, 0>;
def bf_dma_dra : NPUReg50XX_RegFieldWrapper<"dma_dra", 1, 63>;

def reg_dma_dst_addr : NPUReg50XX_RegisterWrapper<"dma_dst_addr", 8, 0x30,  // 48 dec
    ["dma_dst", "dma_dra"]>;

//
//  dma_64_bit_word_7
//

def bf_dma_sba_addr : NPUReg50XX_RegFieldWrapper<"dma_sba_addr", 32, 0>;
def reg_dma_sba_addr : NPUReg50XX_RegisterWrapper<"dma_sba_addr", 4, 0x38,  // 56 dec
    ["dma_sba_addr"]>;

def bf_dma_dba_addr : NPUReg50XX_RegFieldWrapper<"dma_dba_addr", 32, 0>;
def reg_dma_dba_addr : NPUReg50XX_RegisterWrapper<"dma_dba_addr", 4, 0x3C,  // 56 dec
    ["dma_dba_addr"]>;

//
//  dma_64_bit_word_8
//

def bf_dma_barrier_prod_mask_lower : NPUReg50XX_RegFieldWrapper<"dma_barrier_prod_mask_lower", 64, 0>;
def reg_dma_barrier_prod_mask_lower : NPUReg50XX_RegisterWrapper<"dma_barrier_prod_mask_lower", 8, 0x40,  // 64 dec
    ["dma_barrier_prod_mask_lower"]>;

//
//  dma_64_bit_word_9
//

def bf_dma_barrier_cons_mask_lower : NPUReg50XX_RegFieldWrapper<"dma_barrier_cons_mask_lower", 64, 0>;
def reg_dma_barrier_cons_mask_lower : NPUReg50XX_RegisterWrapper<"dma_barrier_cons_mask_lower", 8, 0x48,  // 72 dec
    ["dma_barrier_cons_mask_lower"]>;

//
//  dma_64_bit_word_10
//

def bf_dma_barrier_prod_mask_upper : NPUReg50XX_RegFieldWrapper<"dma_barrier_prod_mask_upper", 32, 0>;
def reg_dma_barrier_prod_mask_upper : NPUReg50XX_RegisterWrapper<"dma_barrier_prod_mask_upper", 8, 0x50,  // 80 dec
    ["dma_barrier_prod_mask_upper"]>;

//
//  dma_64_bit_word_11
//

def bf_dma_barrier_cons_mask_upper : NPUReg50XX_RegFieldWrapper<"dma_barrier_cons_mask_upper", 32, 0>;
def reg_dma_barrier_cons_mask_upper : NPUReg50XX_RegisterWrapper<"dma_barrier_cons_mask_upper", 8, 0x58,  // 88 dec
    ["dma_barrier_cons_mask_upper"]>;

//
//  dma_64_bit_word_12
//

def bf_dma_list_size_src : NPUReg50XX_RegFieldWrapper<"dma_list_size_src", 32, 0>;
def bf_dma_list_size_dst : NPUReg50XX_RegFieldWrapper<"dma_list_size_dst", 32, 32>;

def reg_dma_list_size : NPUReg50XX_RegisterWrapper<"dma_list_size", 8, 0x60,  // 96 dec
    ["dma_list_size_src", "dma_list_size_dst"]>;

def bf_dma_dim_size_src_1 : NPUReg50XX_RegFieldWrapper<"dma_dim_size_src_1", 32, 0>;
def bf_dma_dim_size_dst_1 : NPUReg50XX_RegFieldWrapper<"dma_dim_size_dst_1", 32, 32>;

def reg_dma_dim_size : NPUReg50XX_RegisterWrapper<"dma_dim_size", 8, 0x60,  // 96 dec
    ["dma_dim_size_src_1", "dma_dim_size_dst_1"]>;

//
//  dma_64_bit_word_13
//

def bf_dma_stride_src_1 : NPUReg50XX_RegFieldWrapper<"dma_stride_src_1", 32, 0>;
def reg_dma_stride_src_1 : NPUReg50XX_RegisterWrapper<"dma_stride_src_1", 4, 0x68,  // 104 dec
    ["dma_stride_src_1"]>;

def bf_dma_stride_dst_1 : NPUReg50XX_RegFieldWrapper<"dma_stride_dst_1", 32, 0>;
def reg_dma_stride_dst_1 : NPUReg50XX_RegisterWrapper<"dma_stride_dst_1", 4, 0x6C,  // 108 dec
    ["dma_stride_dst_1"]>;

//
//  dma_64_bit_word_14
//

def bf_dma_dim_size_src_2 : NPUReg50XX_RegFieldWrapper<"dma_dim_size_src_2", 32, 0>;
def bf_dma_dim_size_dst_2 : NPUReg50XX_RegFieldWrapper<"dma_dim_size_dst_2", 32, 32>;

def reg_dma_dim_size_2 : NPUReg50XX_RegisterWrapper<"dma_dim_size_2", 8, 0x70,  // 112 dec
    ["dma_dim_size_src_2", "dma_dim_size_dst_2"]>;

def bf_dma_list_addr_src : NPUReg50XX_RegFieldWrapper<"dma_list_addr_src", 32, 0>;
def bf_dma_list_addr_dst : NPUReg50XX_RegFieldWrapper<"dma_list_addr_dst", 32, 32>;

def reg_dma_list_addr : NPUReg50XX_RegisterWrapper<"dma_list_addr", 8, 0x70,  // 112 dec
    ["dma_list_addr_src", "dma_list_addr_dst"]>;

//
//  dma_64_bit_word_15
//

def bf_dma_stride_src_2 : NPUReg50XX_RegFieldWrapper<"dma_stride_src_2", 32, 0>;
def reg_dma_stride_src_2 : NPUReg50XX_RegisterWrapper<"dma_stride_src_2", 4, 0x78,  // 120 dec
    ["dma_stride_src_2"]>;

def bf_dma_stride_dst_2 : NPUReg50XX_RegFieldWrapper<"dma_stride_dst_2", 32, 0>;
def reg_dma_stride_dst_2 : NPUReg50XX_RegisterWrapper<"dma_stride_dst_2", 4, 0x7C,  // 124 dec
    ["dma_stride_dst_2"]>;

def bf_dma_remote_width_store : NPUReg50XX_RegFieldWrapper<"dma_remote_width_store", 32, 0>;
def reg_dma_remote_width_store : NPUReg50XX_RegisterWrapper<"dma_remote_width_store", 4, 0x7C,  // 124 dec
    ["dma_remote_width_store"]>;

//
//  dma_64_bit_word_16
//

def bf_dma_dim_size_src_3 : NPUReg50XX_RegFieldWrapper<"dma_dim_size_src_3", 16, 0>;
def reg_dma_dim_size_src_3 : NPUReg50XX_RegisterWrapper<"dma_dim_size_src_3", 2, 0x80,  // 128 dec
    ["dma_dim_size_src_3"]>;

def bf_dma_dim_size_src_4 : NPUReg50XX_RegFieldWrapper<"dma_dim_size_src_4", 16, 0>;
def reg_dma_dim_size_src_4 : NPUReg50XX_RegisterWrapper<"dma_dim_size_src_4", 2, 0x82,  // 130 dec
    ["dma_dim_size_src_4"]>;

def bf_dma_dim_size_dst_3 : NPUReg50XX_RegFieldWrapper<"dma_dim_size_dst_3", 16, 0>;
def reg_dma_dim_size_dst_3 : NPUReg50XX_RegisterWrapper<"dma_dim_size_dst_3", 2, 0x84,  // 132 dec
    ["dma_dim_size_dst_3"]>;

def bf_dma_dim_size_dst_4 : NPUReg50XX_RegFieldWrapper<"dma_dim_size_dst_4", 16, 0>;
def reg_dma_dim_size_dst_4 : NPUReg50XX_RegisterWrapper<"dma_dim_size_dst_4", 2, 0x86,  // 134 dec
    ["dma_dim_size_dst_4"]>;

//
//  dma_64_bit_word_17
//

def bf_dma_dim_size_src_5 : NPUReg50XX_RegFieldWrapper<"dma_dim_size_src_5", 16, 0>;
def reg_dma_dim_size_src_5 : NPUReg50XX_RegisterWrapper<"dma_dim_size_src_5", 2, 0x88,  // 136 dec
    ["dma_dim_size_src_5"]>;

def bf_dma_dim_size_dst_5 : NPUReg50XX_RegFieldWrapper<"dma_dim_size_dst_5", 16, 0>;
def reg_dma_dst_dim_size_5 : NPUReg50XX_RegisterWrapper<"dma_dim_size_dst_5", 2, 0x8C,  // 140 dec
    ["dma_dim_size_dst_5"]>;

//
//  dma_64_bit_word_18
//

def bf_dma_stride_src_3 : NPUReg50XX_RegFieldWrapper<"dma_stride_src_3", 32, 0>;
def reg_dma_stride_src_3 : NPUReg50XX_RegisterWrapper<"dma_stride_src_3", 4, 0x90,  // 144 dec
    ["dma_stride_src_3"]>;

def bf_dma_stride_dst_3 : NPUReg50XX_RegFieldWrapper<"dma_stride_dst_3", 32, 0>;
def reg_dma_stride_dst_3 : NPUReg50XX_RegisterWrapper<"dma_stride_dst_3", 4, 0x94,  // 148 dec
    ["dma_stride_dst_3"]>;

//
//  dma_64_bit_word_19
//

def bf_dma_stride_src_4 : NPUReg50XX_RegFieldWrapper<"dma_stride_src_4", 32, 0>;
def reg_dma_stride_src_4 : NPUReg50XX_RegisterWrapper<"dma_stride_src_4", 4, 0x98,  // 152 dec
    ["dma_stride_src_4"]>;

def bf_dma_stride_dst_4 : NPUReg50XX_RegFieldWrapper<"dma_stride_dst_4", 32, 0>;
def reg_dma_stride_dst_4 : NPUReg50XX_RegisterWrapper<"dma_stride_dst_4", 4, 0x9C,  // 156 dec
    ["dma_stride_dst_4"]>;

//
//  dma_64_bit_word_20
//

def bf_dma_stride_src_5 : NPUReg50XX_RegFieldWrapper<"dma_stride_src_5", 32, 0>;
def reg_dma_stride_src_5 : NPUReg50XX_RegisterWrapper<"dma_stride_src_5", 4, 0xA0,  // 160 dec
    ["dma_stride_src_5"]>;

def bf_dma_stride_dst_5 : NPUReg50XX_RegFieldWrapper<"dma_stride_dst_5", 32, 0>;
def reg_dma_stride_dst_5 : NPUReg50XX_RegisterWrapper<"dma_stride_dst_5", 4, 0xA4,  // 164 dec
    ["dma_stride_dst_5"]>;

//
//  dma_64_bit_word_21
//

def bf_dma_word_21_reserved : NPUReg50XX_RegFieldWrapper<"dma_word_21_reserved", 64, 0>;
def reg_dma_word_21_reserved : NPUReg50XX_RegisterWrapper<"dma_word_21_reserved", 8, 0xA8,  // 168 dec
    ["dma_word_21_reserved"]>;

//
//  dma_64_bit_word_22
//

def bf_dma_word_22_reserved : NPUReg50XX_RegFieldWrapper<"dma_word_22_reserved", 64, 0>;
def reg_dma_word_22_reserved : NPUReg50XX_RegisterWrapper<"dma_word_22_reserved", 8, 0xB0,  // 176 dec
    ["dma_word_22_reserved"]>;

//
//  dma_64_bit_word_21
//

def bf_dma_word_23_reserved : NPUReg50XX_RegFieldWrapper<"dma_word_23_reserved", 64, 0>;
def reg_dma_word_23_reserved : NPUReg50XX_RegisterWrapper<"dma_word_23_reserved", 8, 0xB8,  // 184 dec
    ["dma_word_23_reserved"]>;

//
//  dma_barriers_sched
//

def reg_VpuTaskSchedulingBarrierConfig : NPUReg50XX_RegisterWrapper<"dma_barriers_sched", 8, 0xC0,
    ["start_after_", "clean_after_"]>;

def bf_dma_pad_24_0 : NPUReg50XX_RegFieldWrapper<"dma_pad_24_0", 64, 0>;
def reg_dma_pad_24_0 : NPUReg50XX_RegisterWrapper<"dma_pad_24_0", 8, 0xC8, ["dma_pad_24_0"]>;

def bf_dma_pad_24_1 : NPUReg50XX_RegFieldWrapper<"dma_pad_24_1", 64, 0>;
def reg_dma_pad_24_1 : NPUReg50XX_RegisterWrapper<"dma_pad_24_1", 8, 0xD0, ["dma_pad_24_1"]>;

def bf_dma_pad_24_2 : NPUReg50XX_RegFieldWrapper<"dma_pad_24_2", 64, 0>;
def reg_dma_pad_24_2 : NPUReg50XX_RegisterWrapper<"dma_pad_24_2", 8, 0xD8, ["dma_pad_24_2"]>;

// DMARegister descriptor
def reg_mapped_DMARegister : NPUReg50XX_RegMappedWrapper<"DMARegister",
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

def bf_NextSameId : NPUReg50XX_RegFieldWrapper<"next_same_id_", 32, 0, "UINT">;
def reg_NextSameId : NPUReg50XX_RegisterWrapper<"next_same_id_", 4, 0x0, ["next_same_id_"]>;

def bf_ProducerCount: NPUReg50XX_RegFieldWrapper<"producer_count_", 16, 0>;
def reg_ProducerCount : NPUReg50XX_RegisterWrapper<"producer_count_", 2, 0x4, ["producer_count_"]>;

def bf_ConsumerCount : NPUReg50XX_RegFieldWrapper<"consumer_count_", 16, 0>;
def reg_ConsumerCount : NPUReg50XX_RegisterWrapper<"consumer_count_", 2, 0x6, ["consumer_count_"]>;

def bf_BarrierId : NPUReg50XX_RegFieldWrapper<"real_id_", 8, 0>;
def reg_BarrierId : NPUReg50XX_RegisterWrapper<"real_id_", 1, 0x8, ["real_id_"]>;

def bf_BarCfg_Pad3 : NPUReg50XX_RegFieldWrapper<"barcfg_pad_3_", 24, 0>;
def reg_BarCfg_Pad3 : NPUReg50XX_RegisterWrapper<"barcfg_pad_3_", 3, 0x9, ["barcfg_pad_3_"]>;

// BarrierCfg descriptor
def reg_mapped_BarrierDescriptor : NPUReg50XX_RegMappedWrapper<"VpuBarrierCountConfig",
    ["next_same_id_", "producer_count_", "consumer_count_", "real_id_", "barcfg_pad_3_"]>;

//
// ActKernelRange registers
//

def bf_ActWLType : NPUReg50XX_RegFieldWrapper<"type", 8, 0>;
def reg_ActWLType : NPUReg50XX_RegisterWrapper<"type", 1, 0x0, ["type"]>; // WL_KERNEL = 0x00, WL_DEBUG = 0x04, WL_UNKNOWN

def bf_KernelEntry : NPUReg50XX_RegFieldWrapper<"kernel_entry", 64, 0>;
def reg_KernelEntry : NPUReg50XX_RegisterWrapper<"kernel_entry", 8, 0x8, ["kernel_entry"]>;

def bf_TextWindowBase: NPUReg50XX_RegFieldWrapper<"text_window_base", 64, 0>;
def reg_TextWindowBase : NPUReg50XX_RegisterWrapper<"text_window_base", 8, 0x10, ["text_window_base"]>;

def bf_CodeSize : NPUReg50XX_RegFieldWrapper<"code_size", 32, 0>;
def reg_CodeSize : NPUReg50XX_RegisterWrapper<"code_size", 4, 0x18, ["code_size"]>;

def bf_DeprecatedAkr : NPUReg50XX_RegFieldWrapper<"deprecated_akr", 32, 0, "UINT">;
def reg_DeprecatedAkr : NPUReg50XX_RegisterWrapper<"deprecated_akr", 4, 0x1C, ["deprecated_akr"]>;

def bf_KernelInvoCount : NPUReg50XX_RegFieldWrapper<"kernel_invo_count", 32, 0>;
def reg_KernelInvoCount : NPUReg50XX_RegisterWrapper<"kernel_invo_count", 4, 0x20, ["kernel_invo_count"]>;

def bf_Pad1_4 : NPUReg50XX_RegFieldWrapper<"pad1_4", 32, 0>;
def reg_Pad1_4 : NPUReg50XX_RegisterWrapper<"pad1_4", 4, 0x24, ["pad1_4"]>;

// ActKernelRange descriptor
def reg_mapped_ActKernelRangeDescriptor : NPUReg50XX_RegMappedWrapper<"VpuActKernelRange",
    ["type", "kernel_entry", "text_window_base", "code_size", "deprecated_akr", "kernel_invo_count", "pad1_4"]>;

//
// ActKernelInvocation registers
//

def bf_Range : NPUReg50XX_RegFieldWrapper<"range", 64, 0>;
def reg_Range : NPUReg50XX_RegisterWrapper<"range", 8, 0x0, ["range"]>;

def bf_KernelArgs : NPUReg50XX_RegFieldWrapper<"kernel_args", 64, 0>;
def reg_KernelArgs : NPUReg50XX_RegisterWrapper<"kernel_args", 8, 0x8, ["kernel_args"]>;

def bf_DataWindowBase: NPUReg50XX_RegFieldWrapper<"data_window_base", 64, 0>;
def reg_DataWindowBase : NPUReg50XX_RegisterWrapper<"data_window_base", 8, 0x10, ["data_window_base"]>;

def bf_PerfPacketOut : NPUReg50XX_RegFieldWrapper<"perf_packet_out", 64, 0>;
def reg_PerfPacketOut : NPUReg50XX_RegisterWrapper<"perf_packet_out", 8, 0x18, ["perf_packet_out"]>;

def bf_BarrierWaitMaskHiAct : NPUReg50XX_RegFieldWrapper<"barriers_wait_mask_hi_act", 32, 0>;
def reg_BarrierWaitMaskHiAct : NPUReg50XX_RegisterWrapper<"barriers_wait_mask_hi_act", 8, 0x20, ["barriers_wait_mask_hi_act"]>;

def bf_BarrierWaitMaskLoAct : NPUReg50XX_RegFieldWrapper<"barriers_wait_mask_lo_act", 64, 0>;
def reg_BarrierWaitMaskLoAct : NPUReg50XX_RegisterWrapper<"barriers_wait_mask_lo_act", 8, 0x28, ["barriers_wait_mask_lo_act"]>;

def bf_BarrierWaitPostHiAct : NPUReg50XX_RegFieldWrapper<"barriers_post_mask_hi_act", 32, 0>;
def reg_BarrierWaitPostHiAct : NPUReg50XX_RegisterWrapper<"barriers_post_mask_hi_act", 8, 0x30, ["barriers_post_mask_hi_act"]>;

def bf_BarrierWaitPostLoAct : NPUReg50XX_RegFieldWrapper<"barriers_post_mask_lo_act", 64, 0>;
def reg_BarrierWaitPostLoAct : NPUReg50XX_RegisterWrapper<"barriers_post_mask_lo_act", 8, 0x38, ["barriers_post_mask_lo_act"]>;

def bf_BarrierGroupAct : NPUReg50XX_RegFieldWrapper<"group_act", 8, 0>;
def bf_BarrierMaskAct : NPUReg50XX_RegFieldWrapper<"mask_act", 8, 8>;
def reg_BarrierGroupMaskAct : NPUReg50XX_RegisterWrapper<"barriers_group_mask_act", 8, 0x40, ["group_act", "mask_act"]>;

def reg_SchedulingBarrierConfig : NPUReg50XX_RegisterWrapper<"act_invo_barriers_sched", 8, 0x48, ["start_after_", "clean_after_"]>;

def bf_DeprecatedAki : NPUReg50XX_RegFieldWrapper<"invo_index", 32, 0, "UINT">;
def reg_DeprecatedAki : NPUReg50XX_RegisterWrapper<"invo_index", 4, 0x50, ["invo_index"]>;

def bf_InvoTile : NPUReg50XX_RegFieldWrapper<"invo_tile", 32, 0>;
def reg_InvoTile : NPUReg50XX_RegisterWrapper<"invo_tile", 4, 0x54, ["invo_tile"]>;

def bf_KernelRangeIndex : NPUReg50XX_RegFieldWrapper<"kernel_range_index", 32, 0>;
def reg_KernelRangeIndex : NPUReg50XX_RegisterWrapper<"kernel_range_index", 4, 0x58, ["kernel_range_index"]>;

def bf_NextAkiWlAddr : NPUReg50XX_RegFieldWrapper<"next_aki_wl_addr", 32, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_NextAkiWlAddr : NPUReg50XX_RegisterWrapper<"next_aki_wl_addr", 4, 0x5C, ["next_aki_wl_addr"]>;

// ActKernelInvocation descriptor
def reg_mapped_ActKernelInvocationDescriptor : NPUReg50XX_RegMappedWrapper<"VpuActKernelInvocation",
    ["range", "kernel_args", "data_window_base", "perf_packet_out",
    "barriers_wait_mask_hi_act", "barriers_wait_mask_lo_act", "barriers_post_mask_hi_act", "barriers_post_mask_lo_act", "barriers_group_mask_act",
    "act_invo_barriers_sched", "invo_index", "invo_tile", "kernel_range_index", "next_aki_wl_addr"]>;

//
// M2I registers
//

// VpuMediaBuffDescriptor

// IN_ADDR common
def bf_InAddr : NPUReg50XX_RegFieldWrapper<"inAddr", 64, 0>;

// IN_ADDR_0
def reg_InAddr0 : NPUReg50XX_RegisterWrapper<"inAddr0", 8, 0x0, ["inAddr"]>;

// common part for InSize0,1,2
def bf_Width : NPUReg50XX_RegFieldWrapper<"width", 16, 0>;
def bf_Height : NPUReg50XX_RegFieldWrapper<"height", 16, 16>;
def bf_Ls : NPUReg50XX_RegFieldWrapper<"ls", 24, 32>;

// InSize0
def bf_PID : NPUReg50XX_RegFieldWrapper<"pid", 8, 56>;
def reg_InSize0 : NPUReg50XX_RegisterWrapper<"inSize0", 8, 0x8, ["width", "height", "ls", "pid"]>;

// IN_ADDR_1
def reg_InAddr1 : NPUReg50XX_RegisterWrapper<"inAddr1", 8, 0x10, ["inAddr"]>;

// InSize1
def bf_HWPEN : NPUReg50XX_RegFieldWrapper<"HWPEN", 1, 56>;
def bf_ExtHDR : NPUReg50XX_RegFieldWrapper<"ExtHDR", 1, 57>;
def bf_Reserved_InSize1 : NPUReg50XX_RegFieldWrapper<"Reserved_InSize1", 6, 58>;
def reg_InSize1 : NPUReg50XX_RegisterWrapper<"inSize1", 8, 0x18, ["width", "height", "ls", "HWPEN", "ExtHDR", "Reserved_InSize1"]>;

// IN_ADDR_2
def reg_InAddr2 : NPUReg50XX_RegisterWrapper<"inAddr2", 8, 0x20, ["inAddr"]>;

// InSize2
def bf_Reserved_InSize2 : NPUReg50XX_RegFieldWrapper<"Reserved_InSize2", 8, 56>;
def reg_InSize2 : NPUReg50XX_RegisterWrapper<"inSize2", 8, 0x28, ["width", "height", "ls", "Reserved_InSize2"]>;

// IOCfg
def bf_InFormat : NPUReg50XX_RegFieldWrapper<"inFormat", 8, 0>;
def bf_OutFormat : NPUReg50XX_RegFieldWrapper<"outFormat", 8, 8>;
def bf_NumRois : NPUReg50XX_RegFieldWrapper<"numRois", 16, 16>;
def bf_SampleType : NPUReg50XX_RegFieldWrapper<"sampleType", 4, 32>;
def bf_Operations : NPUReg50XX_RegFieldWrapper<"operations", 4, 36>;
def bf_IFC : NPUReg50XX_RegFieldWrapper<"IFC", 8, 40>;
def bf_IRQMask : NPUReg50XX_RegFieldWrapper<"IRQMask", 16, 48>;

def reg_IOCfg : NPUReg50XX_RegisterWrapper<"IOCfg", 8, 0x30, ["inFormat", "outFormat", "numRois", "sampleType", "operations", "IFC", "IRQMask"]>;

// NormFactor
def bf_NormFact0 : NPUReg50XX_RegFieldWrapper<"NormFact0", 16, 0>;
def bf_NormFact1 : NPUReg50XX_RegFieldWrapper<"NormFact1", 16, 16>;
def bf_NormFact2 : NPUReg50XX_RegFieldWrapper<"NormFact2", 16, 32>;
def bf_NormFact3 : NPUReg50XX_RegFieldWrapper<"NormFact3", 16, 48>;

def reg_NormFactor_0 : NPUReg50XX_RegisterWrapper<"NormFactor_0", 8, 0x38, ["NormFact0", "NormFact1", "NormFact2", "NormFact3"]>;
def reg_NormFactor_1 : NPUReg50XX_RegisterWrapper<"NormFactor_1", 8, 0x40, ["NormFact0", "NormFact1", "NormFact2", "NormFact3"]>;
def reg_NormFactor_2 : NPUReg50XX_RegisterWrapper<"NormFactor_2", 8, 0x48, ["NormFact0", "NormFact1", "NormFact2", "NormFact3"]>;

// PSOB
def bf_InPS : NPUReg50XX_RegFieldWrapper<"inPS", 32, 0>;
def bf_OutBase : NPUReg50XX_RegFieldWrapper<"outBase", 16, 32>;
def bf_HWPAddrLO : NPUReg50XX_RegFieldWrapper<"HWPAddrLO", 16, 48>;

def reg_PSOB : NPUReg50XX_RegisterWrapper<"PSOB", 8, 0x50, ["inPS", "outBase", "HWPAddrLO"]>;

// nextDesc
def bf_NextDesc : NPUReg50XX_RegFieldWrapper<"nextDesc", 48, 0>;
def reg_NextDesc : NPUReg50XX_RegisterWrapper<"nextDesc", 6, 0x58, ["nextDesc"]>;

// HWPAddrHI
def bf_HWPAddrHI : NPUReg50XX_RegFieldWrapper<"HWPAddrHI", 16, 0>;
def reg_HWPAddrHI : NPUReg50XX_RegisterWrapper<"HWPAddrHI", 2, 0x5E, ["HWPAddrHI"]>;

//VpuROIDescriptor standard
// RoiDef
def bf_RoiBase : NPUReg50XX_RegFieldWrapper<"roiBase", 32, 0>;
def bf_OutFormatLocal : NPUReg50XX_RegFieldWrapper<"outFormatLocal", 8, 32>;
def bf_SamlingTypeLocal : NPUReg50XX_RegFieldWrapper<"samlingTypeLocal", 4, 40>;
def bf_OFC : NPUReg50XX_RegFieldWrapper<"OFC", 8, 44>;
def bf_IRQLocal : NPUReg50XX_RegFieldWrapper<"IRQLocal", 8, 52>;
def bf_HWProfEN : NPUReg50XX_RegFieldWrapper<"HWProfEN", 1, 60>;
def bf_RoiDef_RESERVED : NPUReg50XX_RegFieldWrapper<"RoiDef_RESERVED", 3, 61>;
def reg_RoiDef : NPUReg50XX_RegisterWrapper<"RoiDef", 8, 0x60,
    ["roiBase", "outFormatLocal", "samlingTypeLocal", "OFC", "IRQLocal", "HWProfEN", "RoiDef_RESERVED"]>;

// RoiCfg
def bf_X_coord : NPUReg50XX_RegFieldWrapper<"X_coord", 16, 0>;
def bf_Y_coord : NPUReg50XX_RegFieldWrapper<"Y_coord", 16, 16>;
def bf_RoiWidth : NPUReg50XX_RegFieldWrapper<"roiWidth", 16, 32>;
def bf_RoiHeight : NPUReg50XX_RegFieldWrapper<"roiHeight", 16, 48>;
def reg_RoiCfg : NPUReg50XX_RegisterWrapper<"RoiCfg", 8, 0x68, ["X_coord", "Y_coord", "roiWidth", "roiHeight"]>;

// OutScaleSize
def bf_OutScale0_width : NPUReg50XX_RegFieldWrapper<"outScale0_width", 16, 0>;
def bf_OutScale0_height : NPUReg50XX_RegFieldWrapper<"outScale0_height", 16, 16>;
def bf_OutScale1_width : NPUReg50XX_RegFieldWrapper<"outScale1_width", 16, 32>;
def bf_OutScale1_height : NPUReg50XX_RegFieldWrapper<"outScale1_height", 16, 48>;
def reg_OutScaleSize : NPUReg50XX_RegisterWrapper<"OutScaleSize", 8, 0x70, ["outScale0_width", "outScale0_height", "outScale1_width", "outScale1_height"]>;

// ScPSY
def bf_PsSc0Y : NPUReg50XX_RegFieldWrapper<"psSc0Y", 32, 0>;
def bf_PsSc1Y : NPUReg50XX_RegFieldWrapper<"psSc1Y", 32, 32>;
def reg_ScPSY : NPUReg50XX_RegisterWrapper<"ScPSY", 8, 0x78, ["psSc0Y", "psSc1Y"]>;

// ScPSUV
def bf_PsSc0UV : NPUReg50XX_RegFieldWrapper<"psSc0UV", 32, 0>;
def bf_PsSc1UV : NPUReg50XX_RegFieldWrapper<"psSc1UV", 32, 32>;
def reg_ScPSUV : NPUReg50XX_RegisterWrapper<"ScPSUV", 8, 0x80, ["psSc0UV", "psSc1UV"]>;

// OutLS
def bf_LsSc0Y : NPUReg50XX_RegFieldWrapper<"lsSc0Y", 16, 0>;
def bf_LsSc1Y : NPUReg50XX_RegFieldWrapper<"lsSc1Y", 16, 16>;
def bf_LsSc0UV : NPUReg50XX_RegFieldWrapper<"lsSc0UV", 16, 32>;
def bf_LsSc1UV : NPUReg50XX_RegFieldWrapper<"lsSc1UV", 16, 48>;
def reg_OutLS : NPUReg50XX_RegisterWrapper<"OutLS", 8, 0x88, ["lsSc0Y", "lsSc1Y", "lsSc0UV", "lsSc1UV"]>;

// ScOffset
def bf_VSc_offset : NPUReg50XX_RegFieldWrapper<"vSc_offset", 32, 0>;
def bf_HSc_offset : NPUReg50XX_RegFieldWrapper<"hSc_offset", 32, 32>;
def reg_ScOffset : NPUReg50XX_RegisterWrapper<"ScOffset", 8, 0x90, ["vSc_offset", "hSc_offset"]>;

// ScFactor
def bf_VSc_factor : NPUReg50XX_RegFieldWrapper<"vSc_factor", 32, 0>;
def bf_HSc_factor : NPUReg50XX_RegFieldWrapper<"hSc_factor", 32, 32>;
def reg_ScFactor : NPUReg50XX_RegisterWrapper<"ScFactor", 8, 0x98, ["vSc_factor", "hSc_factor"]>;

def bf_BarGateMaskLO : NPUReg50XX_RegFieldWrapper<"barGateMaskLO", 64, 0>;
def reg_BarGateMaskLO : NPUReg50XX_RegisterWrapper<"barGateMaskLO", 8, 0xA0, ["barGateMaskLO"]>;
def bf_BarGateMaskHI : NPUReg50XX_RegFieldWrapper<"barGateMaskHI", 64, 0>;
def reg_BarGateMaskHI : NPUReg50XX_RegisterWrapper<"barGateMaskHI", 8, 0xA8, ["barGateMaskHI"]>;
def bf_BarUpdateLO : NPUReg50XX_RegFieldWrapper<"barUpdateLO", 64, 0>;
def reg_BarUpdateLO : NPUReg50XX_RegisterWrapper<"barUpdateLO", 8, 0xB0, ["barUpdateLO"]>;
def bf_BarUpdateHI : NPUReg50XX_RegFieldWrapper<"barUpdateHI", 64, 0>;
def reg_BarUpdateHI : NPUReg50XX_RegisterWrapper<"barUpdateHI", 8, 0xB8, ["barUpdateHI"]>;

// Media BarriersScheduling
def reg_MediaBarriersSchedulingCfg : NPUReg50XX_RegisterWrapper<"media_barriers_sched_", 8, 0xE0, ["start_after_", "clean_after_"]>;

// uint8_t pad0_[8];
def bf_Pad8_0 : NPUReg50XX_RegFieldWrapper<"pad8_0", 32, 0>;
def reg_Pad8_0 : NPUReg50XX_RegisterWrapper<"pad8_0", 4, 0xE8, ["pad8_0"]>;
def bf_Pad8_1 : NPUReg50XX_RegFieldWrapper<"pad8_1", 32, 0>;
def reg_Pad8_1 : NPUReg50XX_RegisterWrapper<"pad8_1", 4, 0xEC, ["pad8_1"]>;

// M2I descriptor
def reg_mapped_M2IDescriptor : NPUReg50XX_RegMappedWrapper<"VpuMediaTask",
    ["inAddr0", "inSize0", "inAddr1", "inSize1", "inAddr2", "inSize2", "IOCfg", "NormFactor_0", "NormFactor_1",
    "NormFactor_2", "PSOB", "nextDesc", "HWPAddrHI", "RoiDef", "RoiCfg", "OutScaleSize", "ScPSY", "ScPSUV", "OutLS",
    "ScOffset", "ScFactor", "barGateMaskLO", "barGateMaskHI", "barUpdateLO", "barUpdateHI",
    "media_barriers_sched_", "pad8_0", "pad8_1"]>;


// Extended header structure for setting VPU5 CSC coefficients
def bf_Coeff1 : NPUReg50XX_RegFieldWrapper<"coeff1", 16, 0>;
def bf_Coeff2 : NPUReg50XX_RegFieldWrapper<"coeff2", 16, 16>;
def bf_Coeff3 : NPUReg50XX_RegFieldWrapper<"coeff3", 16, 32>;
def reg_Coeff_1 : NPUReg50XX_RegisterWrapper<"coeff_1", 6, 0x60, ["coeff1", "coeff2", "coeff3"]>;
def reg_Coeff_2 : NPUReg50XX_RegisterWrapper<"coeff_2", 6, 0x66, ["coeff1", "coeff2", "coeff3"]>;
def reg_Coeff_3 : NPUReg50XX_RegisterWrapper<"coeff_3", 6, 0x6C, ["coeff1", "coeff2", "coeff3"]>;

def bf_CSCOff1 : NPUReg50XX_RegFieldWrapper<"CSCOff1", 16, 0>;
def bf_CSCOff2 : NPUReg50XX_RegFieldWrapper<"CSCOff2", 16, 16>;
def bf_CSCOff3 : NPUReg50XX_RegFieldWrapper<"CSCOff3", 16, 32>;
def reg_CSCOff : NPUReg50XX_RegisterWrapper<"CSCOff", 6, 0x72, ["CSCOff1", "CSCOff2", "CSCOff3"]>;

def bf_ExtHeader_RSVD : NPUReg50XX_RegFieldWrapper<"ExtHeader_RSVD", 64, 0>;
def reg_ExtHeader_RSVD : NPUReg50XX_RegisterWrapper<"ExtHeader_RSVD", 8, 0x78, ["ExtHeader_RSVD"]>;

//VpuROIDescriptor extended
// RoiDef
def reg_RoiDef_Ext : NPUReg50XX_RegisterWrapper<"RoiDef_Ext", 8, 0x80,
    ["roiBase", "outFormatLocal", "samlingTypeLocal", "OFC", "IRQLocal", "HWProfEN", "RoiDef_RESERVED"]>;

// RoiCfg
def reg_RoiCfg_Ext : NPUReg50XX_RegisterWrapper<"RoiCfg_Ext", 8, 0x88, ["X_coord", "Y_coord", "roiWidth", "roiHeight"]>;

// OutScaleSize
def reg_OutScaleSize_Ext : NPUReg50XX_RegisterWrapper<"OutScaleSize_Ext", 8, 0x90, ["outScale0_width", "outScale0_height", "outScale1_width", "outScale1_height"]>;

// ScPSY
def reg_ScPSY_Ext : NPUReg50XX_RegisterWrapper<"ScPSY_Ext", 8, 0x98, ["psSc0Y", "psSc1Y"]>;

// ScPSUV
def reg_ScPSUV_Ext : NPUReg50XX_RegisterWrapper<"ScPSUV_Ext", 8, 0xA0, ["psSc0UV", "psSc1UV"]>;

// OutLS
def reg_OutLS_Ext : NPUReg50XX_RegisterWrapper<"OutLS_Ext", 8, 0xA8, ["lsSc0Y", "lsSc1Y", "lsSc0UV", "lsSc1UV"]>;

// ScOffset
def reg_ScOffset_Ext : NPUReg50XX_RegisterWrapper<"ScOffset_Ext", 8, 0xB0, ["vSc_offset", "hSc_offset"]>;

// ScFactor
def reg_ScFactor_Ext : NPUReg50XX_RegisterWrapper<"ScFactor_Ext", 8, 0xB8, ["vSc_factor", "hSc_factor"]>;

def reg_BarGateMaskLO_Ext : NPUReg50XX_RegisterWrapper<"barGateMaskLO_Ext", 8, 0xC0, ["barGateMaskLO"]>;
def reg_BarGateMaskHI_Ext : NPUReg50XX_RegisterWrapper<"barGateMaskHI_Ext", 8, 0xC8, ["barGateMaskHI"]>;
def reg_BarUpdateLO_Ext : NPUReg50XX_RegisterWrapper<"barUpdateLO_Ext", 8, 0xD0, ["barUpdateLO"]>;
def reg_BarUpdateHI_Ext : NPUReg50XX_RegisterWrapper<"barUpdateHI_Ext", 8, 0xD8, ["barUpdateHI"]>;

// M2I descriptor
def reg_mapped_M2IDescriptorExtended : NPUReg50XX_RegMappedWrapper<"VpuMediaTaskExtended",
    ["inAddr0", "inSize0", "inAddr1", "inSize1", "inAddr2", "inSize2", "IOCfg", "NormFactor_0", "NormFactor_1",
    "NormFactor_2", "PSOB", "nextDesc", "HWPAddrHI", "coeff_1", "coeff_2", "coeff_3", "CSCOff", "ExtHeader_RSVD",
    "RoiDef_Ext", "RoiCfg_Ext", "OutScaleSize_Ext", "ScPSY_Ext", "ScPSUV_Ext", "OutLS_Ext", "ScOffset_Ext",
    "ScFactor_Ext", "barGateMaskLO_Ext", "barGateMaskHI_Ext", "barUpdateLO_Ext", "barUpdateHI_Ext",
    "media_barriers_sched_", "pad8_0", "pad8_1"]>;

// WorkItem registers

def bf_WI_desc_ptr : NPUReg50XX_RegFieldWrapper<"desc_ptr", 64, 0>;
def reg_WI_desc_ptr : NPUReg50XX_RegisterWrapper<"desc_ptr", 8, 0x0, ["desc_ptr"]>;

def bf_WI_type : NPUReg50XX_RegFieldWrapper<"wi_type", 8, 0>;
def reg_WI_type : NPUReg50XX_RegisterWrapper<"wi_type", 1, 0x8, ["wi_type"]>;

def bf_WI_unit : NPUReg50XX_RegFieldWrapper<"wi_unit", 8, 0>;
def reg_WI_unit : NPUReg50XX_RegisterWrapper<"wi_unit", 1, 0x9, ["wi_unit"]>;

def bf_WI_sub_unit : NPUReg50XX_RegFieldWrapper<"wi_sub_unit", 8, 0>;
def reg_WI_sub_unit : NPUReg50XX_RegisterWrapper<"wi_sub_unit", 1, 0xA, ["wi_sub_unit"]>;

def bf_Pad0_8_0 : NPUReg50XX_RegFieldWrapper<"pad0_8_0", 8, 0>;
def reg_Pad0_8_0 : NPUReg50XX_RegisterWrapper<"pad0_8_0", 1, 0xB, ["pad0_8_0"]>;

def bf_WI_next_workitem_idx : NPUReg50XX_RegFieldWrapper<"next_workitem_idx", 32, 0>;
def reg_WI_next_workitem_idx : NPUReg50XX_RegisterWrapper<"next_workitem_idx", 4, 0xC, ["next_workitem_idx"]>;

// in API we have pas with 48 bytes, but in compiler we have sanity check for 8 byte per elem -> need 6 pads
def bf_Pad1_8_0 : NPUReg50XX_RegFieldWrapper<"pad1_8_0", 64, 0>;
def reg_Pad1_8_0 : NPUReg50XX_RegisterWrapper<"pad1_8_0", 8, 0x10, ["pad1_8_0"]>;

def bf_Pad1_8_1 : NPUReg50XX_RegFieldWrapper<"pad1_8_1", 64, 0>;
def reg_Pad1_8_1 : NPUReg50XX_RegisterWrapper<"pad1_8_1", 8, 0x18, ["pad1_8_1"]>;

def bf_Pad1_8_2 : NPUReg50XX_RegFieldWrapper<"pad1_8_2", 64, 0>;
def reg_Pad1_8_2 : NPUReg50XX_RegisterWrapper<"pad1_8_2", 8, 0x20, ["pad1_8_2"]>;

def bf_Pad1_8_3 : NPUReg50XX_RegFieldWrapper<"pad1_8_3", 64, 0>;
def reg_Pad1_8_3 : NPUReg50XX_RegisterWrapper<"pad1_8_3", 8, 0x28, ["pad1_8_3"]>;

def bf_Pad1_8_4 : NPUReg50XX_RegFieldWrapper<"pad1_8_4", 64, 0>;
def reg_Pad1_8_4 : NPUReg50XX_RegisterWrapper<"pad1_8_4", 8, 0x30, ["pad1_8_4"]>;

def bf_Pad1_8_5 : NPUReg50XX_RegFieldWrapper<"pad1_8_5", 64, 0>;
def reg_Pad1_8_5 : NPUReg50XX_RegisterWrapper<"pad1_8_5", 8, 0x38, ["pad1_8_5"]>;


def reg_mapped_WorkItemDescriptor : NPUReg50XX_RegMappedWrapper<"WorkItem",
    ["desc_ptr", "wi_type", "wi_unit", "wi_sub_unit", "pad0_8_0", "next_workitem_idx", "pad1_8_0", "pad1_8_1", "pad1_8_2", "pad1_8_3", "pad1_8_4", "pad1_8_5"]>;

// TaskInfo registers
def bf_TIDescPtr : NPUReg50XX_RegFieldWrapper<"ti_desc_ptr", 64, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_TIDescPtr : NPUReg50XX_RegisterWrapper<"ti_desc_ptr", 8, 0x0, ["ti_desc_ptr"]>;

def bf_TIType : NPUReg50XX_RegFieldWrapper<"ti_type", 8, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_TIType : NPUReg50XX_RegisterWrapper<"ti_type", 1, 0x8, ["ti_type"]>;

def bf_TIUnit : NPUReg50XX_RegFieldWrapper<"ti_unit", 8, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_TIUnit : NPUReg50XX_RegisterWrapper<"ti_unit", 1, 0x9, ["ti_unit"]>;

def bf_TISubUnit : NPUReg50XX_RegFieldWrapper<"ti_sub_unit", 8, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_TISubUnit : NPUReg50XX_RegisterWrapper<"ti_sub_unit", 1, 0xA, ["ti_sub_unit"]>;

def bf_TaskInfoPad1 : NPUReg50XX_RegFieldWrapper<"ti_pad1", 8, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_TaskInfoPad1 : NPUReg50XX_RegisterWrapper<"ti_pad1", 1, 0xB, ["ti_pad1"]>;

def bf_TILinkedList : NPUReg50XX_RegFieldWrapper<"ti_linked_list_nodes", 32, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_TILinkedList : NPUReg50XX_RegisterWrapper<"ti_linked_list_nodes", 4, 0xC, ["ti_linked_list_nodes"]>;

def bf_TIDescrRefOffset : NPUReg50XX_RegFieldWrapper<"ti_descr_ref_offset", 32, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_TIDescrRefOffset : NPUReg50XX_RegisterWrapper<"ti_descr_ref_offset", 4, 0x10, ["ti_descr_ref_offset"]>;

def bf_TIParentDescrRefOffset : NPUReg50XX_RegFieldWrapper<"ti_parent_descr_ref_offset", 32, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_TIParentDescrRefOffset : NPUReg50XX_RegisterWrapper<"ti_parent_descr_ref_offset", 4, 0x14, ["ti_parent_descr_ref_offset"]>;

def bf_TIEnqueueingTaskConfig : NPUReg50XX_RegFieldWrapper<"ti_enqueueing_task_config", 32, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_TIEnqueueingTaskConfig : NPUReg50XX_RegisterWrapper<"ti_enqueueing_task_config", 4, 0x18, ["ti_enqueueing_task_config"]>;

def bf_TIWorkItemRef : NPUReg50XX_RegFieldWrapper<"ti_work_item_ref", 32, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_TIWorkItemRef : NPUReg50XX_RegisterWrapper<"ti_work_item_ref", 4, 0x1C, ["ti_work_item_ref"]>;

def reg_mapped_TaskInfoDescriptor : NPUReg50XX_RegMappedWrapper<"VpuTaskInfo",
    ["ti_desc_ptr", "ti_type", "ti_unit", "ti_sub_unit", "ti_pad1", "ti_linked_list_nodes", "ti_descr_ref_offset", "ti_parent_descr_ref_offset", "ti_enqueueing_task_config", "ti_work_item_ref"]>;

// TaskBarrierMap registers

def bf_TaskBarrierNextSameId : NPUReg50XX_RegFieldWrapper<"tb_next_same_id", 32, 0>;
def reg_TaskBarrierNextSameId : NPUReg50XX_RegisterWrapper<"tb_next_same_id", 4, 0x0, ["tb_next_same_id"]>;

def bf_TaskBarrierProducerCount : NPUReg50XX_RegFieldWrapper<"tb_producer_count", 16, 0>;
def reg_TaskBarrierProducerCount : NPUReg50XX_RegisterWrapper<"tb_producer_count", 2, 0x4, ["tb_producer_count"]>;

def bf_TaskBarrierConsumerCount : NPUReg50XX_RegFieldWrapper<"tb_consumer_count", 16, 0>;
def reg_TaskBarrierConsumerCount : NPUReg50XX_RegisterWrapper<"tb_consumer_count", 2, 0x6, ["tb_consumer_count"]>;

def bf_TaskBarrierRealId : NPUReg50XX_RegFieldWrapper<"tb_real_id", 8, 0>;
def reg_TaskBarrierRealId : NPUReg50XX_RegisterWrapper<"tb_real_id", 1, 0x8, ["tb_real_id"]>;

def bf_TaskBarrierPad3 : NPUReg50XX_RegFieldWrapper<"tb_pad3", 24, 0>;
def reg_TaskBarrierPad3 : NPUReg50XX_RegisterWrapper<"tb_pad3", 3, 0x9, ["tb_pad3"]>;

def bf_TaskBarrierWorkItemIdx : NPUReg50XX_RegFieldWrapper<"tb_work_item_idx", 32, 0>;
def reg_TaskBarrierWorkItemIdx : NPUReg50XX_RegisterWrapper<"tb_work_item_idx", 4, 0xC, ["tb_work_item_idx"]>;

def bf_TaskBArrierEnqueueCount : NPUReg50XX_RegFieldWrapper<"tb_enqueue_count", 32, 0>;
def reg_TaskBArrierEnqueueCount : NPUReg50XX_RegisterWrapper<"tb_enqueue_count", 4, 0x10, ["tb_enqueue_count"]>;

def bf_TaskBArrierReservedNextEnqueueId : NPUReg50XX_RegFieldWrapper<"tb_reserved_next_enqueue_id", 32, 0>;
def ref_TaskBArrierReservedNextEnqueueId : NPUReg50XX_RegisterWrapper<"tb_reserved_next_enqueue_id", 4, 0x14, ["tb_reserved_next_enqueue_id"]>;

// TaskBarrierMap descriptor
def reg_mapped_TaskBarrierMapDescriptor : NPUReg50XX_RegMappedWrapper<"VpuTaskBarrierMap",
    ["tb_next_same_id", "tb_producer_count", "tb_consumer_count", "tb_real_id", "tb_pad3", "tb_work_item_idx", "tb_enqueue_count", "tb_reserved_next_enqueue_id"]>;


// BarrierReferenceMap registers

def bf_BRPhysicalBarrier : NPUReg50XX_RegFieldWrapper<"br_physical_barrier", 16, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_BRPhysicalBarrier : NPUReg50XX_RegisterWrapper<"br_physical_barrier", 2, 0x0, ["br_physical_barrier"]>;

def bf_BRProducerCount : NPUReg50XX_RegFieldWrapper<"br_producer_count", 16, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_BRProducerCount : NPUReg50XX_RegisterWrapper<"br_producer_count", 2, 0x2, ["br_producer_count"]>;

def bf_BRConsumerCount : NPUReg50XX_RegFieldWrapper<"br_consumer_count", 16, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_BRConsumerCount : NPUReg50XX_RegisterWrapper<"br_consumer_count", 2, 0x4, ["br_consumer_count"]>;

def bf_BRPad2 : NPUReg50XX_RegFieldWrapper<"br_pad2", 16, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_BRPad2 : NPUReg50XX_RegisterWrapper<"br_pad2", 2, 0x6, ["br_pad2"]>;

def bf_BRProducersRefOffset : NPUReg50XX_RegFieldWrapper<"br_producers_ref_offset", 32, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_BRProducersRefOffset : NPUReg50XX_RegisterWrapper<"br_producers_ref_offset", 4, 0x8, ["br_producers_ref_offset"]>;

def bf_BRConsumersRefOffset : NPUReg50XX_RegFieldWrapper<"br_consumers_ref_offset", 32, 0, "UINT", NPUReg50XX_Version<11, 4, 10>>;
def reg_BRConsumersRefOffset : NPUReg50XX_RegisterWrapper<"br_consumers_ref_offset", 4, 0xC, ["br_consumers_ref_offset"]>;

def reg_mapped_BarrierReferenceMapDescriptor : NPUReg50XX_RegMappedWrapper<"BarrierReferenceMap",
    ["br_physical_barrier", "br_producer_count", "br_consumer_count", "br_pad2", "br_producers_ref_offset", "br_consumers_ref_offset"]>;

//
// VpuTaskReference
//

def bf_taskReferenceR1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1", 64, 0, "UINT">;
def reg_taskReferenceR1 : NPUReg50XX_RegisterWrapper<"taskReferenceR1", 8, 0x0, ["taskReferenceR1"]>;

def bf_taskReferenceR2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2", 64, 0, "UINT">;
def reg_taskReferenceR2 : NPUReg50XX_RegisterWrapper<"taskReferenceR2", 8, 0x8, ["taskReferenceR2"]>;

def bf_taskReferenceR3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3", 64, 0, "UINT">;
def reg_taskReferenceR3 : NPUReg50XX_RegisterWrapper<"taskReferenceR3", 8, 0x10, ["taskReferenceR3"]>;

def bf_taskReferenceAddr : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr", 64, 0, "UINT">;
def reg_taskReferenceAddr : NPUReg50XX_RegisterWrapper<"taskReferenceAddr", 8, 0x18, ["taskReferenceAddr"]>;

def bf_taskReferenceCount : NPUReg50XX_RegFieldWrapper<"taskReferenceCount", 64, 0>;
def reg_taskReferenceCount : NPUReg50XX_RegisterWrapper<"taskReferenceCount", 8, 0x20, ["taskReferenceCount"]>;

def reg_mapped_VpuTaskReferenceDescriptor : NPUReg50XX_RegMappedWrapper<"VpuTaskReference",
    ["taskReferenceR1", "taskReferenceR2", "taskReferenceR3", "taskReferenceAddr", "taskReferenceCount"]>;

//
// VpuMappedInference
//

def bf_miVpuNNRTApiVer : NPUReg50XX_RegFieldWrapper<"miVpuNNRTApiVer", 32, 0, "UINT">;
def reg_miVpuNNRTApiVer : NPUReg50XX_RegisterWrapper<"miVpuNNRTApiVer", 4, 0x0, ["miVpuNNRTApiVer"]>;

def bf_miPad0 : NPUReg50XX_RegFieldWrapper<"miPad0", 32, 0, "UINT">;
def reg_miPad0 : NPUReg50XX_RegisterWrapper<"miPad0", 4, 0x4, ["miPad0"]>;

def bf_miReserved0 : NPUReg50XX_RegFieldWrapper<"miReserved0", 64, 0, "UINT">;
def reg_miReserved0 : NPUReg50XX_RegisterWrapper<"miReserved0", 8, 0x8, ["miReserved0"]>;

def bf_miLogAddrDmaHwp : NPUReg50XX_RegFieldWrapper<"miLogAddrDmaHwp", 64, 0, "UINT">;
def reg_miLogAddrDmaHwp : NPUReg50XX_RegisterWrapper<"miLogAddrDmaHwp", 8, 0x10, ["miLogAddrDmaHwp"]>;

//VpuTaskCounts
def bf_miTcReserved1 : NPUReg50XX_RegFieldWrapper<"miTcReserved1", 32, 0, "UINT">;
def reg_miTcReserved1 : NPUReg50XX_RegisterWrapper<"miTcReserved1", 4, 0x18, ["miTcReserved1"]>;

def bf_miTcReserved2 : NPUReg50XX_RegFieldWrapper<"miTcReserved2", 32, 0, "UINT">;
def reg_miTcReserved2 : NPUReg50XX_RegisterWrapper<"miTcReserved2", 4, 0x1C, ["miTcReserved2"]>;

def bf_miTcDmaDDRCount : NPUReg50XX_RegFieldWrapper<"miTcDmaDDRCount", 32, 0, "UINT">;
def reg_miTcDmaDDRCount : NPUReg50XX_RegisterWrapper<"miTcDmaDDRCount", 4, 0x20, ["miTcDmaDDRCount"]>;

def bf_miTcDmaCMXCount : NPUReg50XX_RegFieldWrapper<"miTcDmaCMXCount", 32, 0, "UINT">;
def reg_miTcDmaCMXCount : NPUReg50XX_RegisterWrapper<"miTcDmaCMXCount", 4, 0x24, ["miTcDmaCMXCount"]>;

def bf_miDPUInvariantCount : NPUReg50XX_RegFieldWrapper<"miDPUInvariantCount", 32, 0, "UINT">;
def reg_miDPUInvariantCount : NPUReg50XX_RegisterWrapper<"miDPUInvariantCount", 4, 0x28, ["miDPUInvariantCount"]>;

def bf_miTcDPUVariantCount : NPUReg50XX_RegFieldWrapper<"miTcDPUVariantCount", 32, 0, "UINT">;
def reg_miTcDPUVariantCount : NPUReg50XX_RegisterWrapper<"miTcDPUVariantCount", 4, 0x2C, ["miTcDPUVariantCount"]>;

def bf_miTcActRangeCount : NPUReg50XX_RegFieldWrapper<"miTcActRangeCount", 32, 0, "UINT">;
def reg_miTcActRangeCount : NPUReg50XX_RegisterWrapper<"miTcActRangeCount", 4, 0x30, ["miTcActRangeCount"]>;

def bf_miTcActInvoCount : NPUReg50XX_RegFieldWrapper<"miTcActInvoCount", 32, 0, "UINT">;
def reg_miTcActInvoCount : NPUReg50XX_RegisterWrapper<"miTcActInvoCount", 4, 0x34, ["miTcActInvoCount"]>;

def bf_miTcMediaCount : NPUReg50XX_RegFieldWrapper<"miTcMediaCount", 32, 0, "UINT">;
def reg_miTcMediaCount : NPUReg50XX_RegisterWrapper<"miTcMediaCount", 4, 0x38, ["miTcMediaCount"]>;
// VpuTaskCounts end

def bf_miTaskStorageSize : NPUReg50XX_RegFieldWrapper<"miTaskStorageSize", 32, 0, "UINT">;
def reg_miTaskStorageSize : NPUReg50XX_RegisterWrapper<"miTaskStorageSize", 4, 0x3C, ["miTaskStorageSize"]>;


// VpuTaskReference<VpuDMATask> dma_tasks_ddr_
def bf_taskReferenceR1_DMA_DDR_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_DDR_0", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_DDR_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_DDR_0", 8, 0x40, ["taskReferenceR1_DMA_DDR_0"]>;
def bf_taskReferenceR2_DMA_DDR_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_DDR_0", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_DDR_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_DDR_0", 8, 0x48, ["taskReferenceR2_DMA_DDR_0"]>;
def bf_taskReferenceR3_DMA_DDR_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_DDR_0", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_DDR_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_DDR_0", 8, 0x50, ["taskReferenceR3_DMA_DDR_0"]>;
def bf_taskReferenceAddr_DMA_DDR_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_DDR_0", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_DDR_0 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_DDR_0", 8, 0x58, ["taskReferenceAddr_DMA_DDR_0"]>;
def bf_taskReferenceCount_DMA_DDR_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_DDR_0", 64, 0>;
def reg_taskReferenceCount_DMA_DDR_0 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_DDR_0", 8, 0x60, ["taskReferenceCount_DMA_DDR_0"]>;

def bf_taskReferenceR1_DMA_DDR_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_DDR_1", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_DDR_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_DDR_1", 8, 0x68, ["taskReferenceR1_DMA_DDR_1"]>;
def bf_taskReferenceR2_DMA_DDR_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_DDR_1", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_DDR_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_DDR_1", 8, 0x70, ["taskReferenceR2_DMA_DDR_1"]>;
def bf_taskReferenceR3_DMA_DDR_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_DDR_1", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_DDR_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_DDR_1", 8, 0x78, ["taskReferenceR3_DMA_DDR_1"]>;
def bf_taskReferenceAddr_DMA_DDR_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_DDR_1", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_DDR_1 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_DDR_1", 8, 0x80, ["taskReferenceAddr_DMA_DDR_1"]>;
def bf_taskReferenceCount_DMA_DDR_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_DDR_1", 64, 0>;
def reg_taskReferenceCount_DMA_DDR_1 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_DDR_1", 8, 0x88, ["taskReferenceCount_DMA_DDR_1"]>;

def bf_taskReferenceR1_DMA_DDR_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_DDR_2", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_DDR_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_DDR_2", 8, 0x90, ["taskReferenceR1_DMA_DDR_2"]>;
def bf_taskReferenceR2_DMA_DDR_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_DDR_2", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_DDR_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_DDR_2", 8, 0x98, ["taskReferenceR2_DMA_DDR_2"]>;
def bf_taskReferenceR3_DMA_DDR_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_DDR_2", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_DDR_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_DDR_2", 8, 0xA0, ["taskReferenceR3_DMA_DDR_2"]>;
def bf_taskReferenceAddr_DMA_DDR_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_DDR_2", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_DDR_2 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_DDR_2", 8, 0xA8, ["taskReferenceAddr_DMA_DDR_2"]>;
def bf_taskReferenceCount_DMA_DDR_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_DDR_2", 64, 0>;
def reg_taskReferenceCount_DMA_DDR_2 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_DDR_2", 8, 0xB0, ["taskReferenceCount_DMA_DDR_2"]>;

def bf_taskReferenceR1_DMA_DDR_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_DDR_3", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_DDR_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_DDR_3", 8, 0xB8, ["taskReferenceR1_DMA_DDR_3"]>;
def bf_taskReferenceR2_DMA_DDR_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_DDR_3", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_DDR_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_DDR_3", 8, 0xC0, ["taskReferenceR2_DMA_DDR_3"]>;
def bf_taskReferenceR3_DMA_DDR_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_DDR_3", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_DDR_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_DDR_3", 8, 0xC8, ["taskReferenceR3_DMA_DDR_3"]>;
def bf_taskReferenceAddr_DMA_DDR_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_DDR_3", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_DDR_3 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_DDR_3", 8, 0xD0, ["taskReferenceAddr_DMA_DDR_3"]>;
def bf_taskReferenceCount_DMA_DDR_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_DDR_3", 64, 0>;
def reg_taskReferenceCount_DMA_DDR_3 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_DDR_3", 8, 0xD8, ["taskReferenceCount_DMA_DDR_3"]>;

def bf_taskReferenceR1_DMA_DDR_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_DDR_4", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_DDR_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_DDR_4", 8, 0xE0, ["taskReferenceR1_DMA_DDR_4"]>;
def bf_taskReferenceR2_DMA_DDR_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_DDR_4", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_DDR_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_DDR_4", 8, 0xE8, ["taskReferenceR2_DMA_DDR_4"]>;
def bf_taskReferenceR3_DMA_DDR_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_DDR_4", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_DDR_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_DDR_4", 8, 0xF0, ["taskReferenceR3_DMA_DDR_4"]>;
def bf_taskReferenceAddr_DMA_DDR_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_DDR_4", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_DDR_4 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_DDR_4", 8, 0xF8, ["taskReferenceAddr_DMA_DDR_4"]>;
def bf_taskReferenceCount_DMA_DDR_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_DDR_4", 64, 0>;
def reg_taskReferenceCount_DMA_DDR_4 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_DDR_4", 8, 0x100, ["taskReferenceCount_DMA_DDR_4"]>;

def bf_taskReferenceR1_DMA_DDR_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_DDR_5", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_DDR_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_DDR_5", 8, 0x108, ["taskReferenceR1_DMA_DDR_5"]>;
def bf_taskReferenceR2_DMA_DDR_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_DDR_5", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_DDR_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_DDR_5", 8, 0x110, ["taskReferenceR2_DMA_DDR_5"]>;
def bf_taskReferenceR3_DMA_DDR_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_DDR_5", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_DDR_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_DDR_5", 8, 0x118, ["taskReferenceR3_DMA_DDR_5"]>;
def bf_taskReferenceAddr_DMA_DDR_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_DDR_5", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_DDR_5 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_DDR_5", 8, 0x120, ["taskReferenceAddr_DMA_DDR_5"]>;
def bf_taskReferenceCount_DMA_DDR_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_DDR_5", 64, 0>;
def reg_taskReferenceCount_DMA_DDR_5 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_DDR_5", 8, 0x128, ["taskReferenceCount_DMA_DDR_5"]>;

// VpuTaskReference<VpuDMATask> dma_tasks_cmx_
def bf_taskReferenceR1_DMA_CMX_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_CMX_0", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_CMX_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_CMX_0", 8, 0x130, ["taskReferenceR1_DMA_CMX_0"]>;
def bf_taskReferenceR2_DMA_CMX_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_CMX_0", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_CMX_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_CMX_0", 8, 0x138, ["taskReferenceR2_DMA_CMX_0"]>;
def bf_taskReferenceR3_DMA_CMX_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_CMX_0", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_CMX_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_CMX_0", 8, 0x140, ["taskReferenceR3_DMA_CMX_0"]>;
def bf_taskReferenceAddr_DMA_CMX_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_CMX_0", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_CMX_0 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_CMX_0", 8, 0x148, ["taskReferenceAddr_DMA_CMX_0"]>;
def bf_taskReferenceCount_DMA_CMX_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_CMX_0", 64, 0>;
def reg_taskReferenceCount_DMA_CMX_0 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_CMX_0", 8, 0x150, ["taskReferenceCount_DMA_CMX_0"]>;

def bf_taskReferenceR1_DMA_CMX_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_CMX_1", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_CMX_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_CMX_1", 8, 0x158, ["taskReferenceR1_DMA_CMX_1"]>;
def bf_taskReferenceR2_DMA_CMX_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_CMX_1", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_CMX_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_CMX_1", 8, 0x160, ["taskReferenceR2_DMA_CMX_1"]>;
def bf_taskReferenceR3_DMA_CMX_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_CMX_1", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_CMX_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_CMX_1", 8, 0x168, ["taskReferenceR3_DMA_CMX_1"]>;
def bf_taskReferenceAddr_DMA_CMX_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_CMX_1", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_CMX_1 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_CMX_1", 8, 0x170, ["taskReferenceAddr_DMA_CMX_1"]>;
def bf_taskReferenceCount_DMA_CMX_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_CMX_1", 64, 0>;
def reg_taskReferenceCount_DMA_CMX_1 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_CMX_1", 8, 0x178, ["taskReferenceCount_DMA_CMX_1"]>;

def bf_taskReferenceR1_DMA_CMX_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_CMX_2", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_CMX_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_CMX_2", 8, 0x180, ["taskReferenceR1_DMA_CMX_2"]>;
def bf_taskReferenceR2_DMA_CMX_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_CMX_2", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_CMX_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_CMX_2", 8, 0x188, ["taskReferenceR2_DMA_CMX_2"]>;
def bf_taskReferenceR3_DMA_CMX_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_CMX_2", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_CMX_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_CMX_2", 8, 0x190, ["taskReferenceR3_DMA_CMX_2"]>;
def bf_taskReferenceAddr_DMA_CMX_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_CMX_2", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_CMX_2 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_CMX_2", 8, 0x198, ["taskReferenceAddr_DMA_CMX_2"]>;
def bf_taskReferenceCount_DMA_CMX_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_CMX_2", 64, 0>;
def reg_taskReferenceCount_DMA_CMX_2 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_CMX_2", 8, 0x1A0, ["taskReferenceCount_DMA_CMX_2"]>;

def bf_taskReferenceR1_DMA_CMX_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_CMX_3", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_CMX_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_CMX_3", 8, 0x1A8, ["taskReferenceR1_DMA_CMX_3"]>;
def bf_taskReferenceR2_DMA_CMX_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_CMX_3", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_CMX_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_CMX_3", 8, 0x1B0, ["taskReferenceR2_DMA_CMX_3"]>;
def bf_taskReferenceR3_DMA_CMX_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_CMX_3", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_CMX_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_CMX_3", 8, 0x1B8, ["taskReferenceR3_DMA_CMX_3"]>;
def bf_taskReferenceAddr_DMA_CMX_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_CMX_3", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_CMX_3 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_CMX_3", 8, 0x1C0, ["taskReferenceAddr_DMA_CMX_3"]>;
def bf_taskReferenceCount_DMA_CMX_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_CMX_3", 64, 0>;
def reg_taskReferenceCount_DMA_CMX_3 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_CMX_3", 8, 0x1C8, ["taskReferenceCount_DMA_CMX_3"]>;

def bf_taskReferenceR1_DMA_CMX_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_CMX_4", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_CMX_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_CMX_4", 8, 0x1D0, ["taskReferenceR1_DMA_CMX_4"]>;
def bf_taskReferenceR2_DMA_CMX_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_CMX_4", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_CMX_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_CMX_4", 8, 0x1D8, ["taskReferenceR2_DMA_CMX_4"]>;
def bf_taskReferenceR3_DMA_CMX_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_CMX_4", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_CMX_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_CMX_4", 8, 0x1E0, ["taskReferenceR3_DMA_CMX_4"]>;
def bf_taskReferenceAddr_DMA_CMX_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_CMX_4", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_CMX_4 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_CMX_4", 8, 0x1E8, ["taskReferenceAddr_DMA_CMX_4"]>;
def bf_taskReferenceCount_DMA_CMX_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_CMX_4", 64, 0>;
def reg_taskReferenceCount_DMA_CMX_4 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_CMX_4", 8, 0x1F0, ["taskReferenceCount_DMA_CMX_4"]>;

def bf_taskReferenceR1_DMA_CMX_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DMA_CMX_5", 64, 0, "UINT">;
def reg_taskReferenceR1_DMA_CMX_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DMA_CMX_5", 8, 0x1F8, ["taskReferenceR1_DMA_CMX_5"]>;
def bf_taskReferenceR2_DMA_CMX_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DMA_CMX_5", 64, 0, "UINT">;
def reg_taskReferenceR2_DMA_CMX_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DMA_CMX_5", 8, 0x200, ["taskReferenceR2_DMA_CMX_5"]>;
def bf_taskReferenceR3_DMA_CMX_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DMA_CMX_5", 64, 0, "UINT">;
def reg_taskReferenceR3_DMA_CMX_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DMA_CMX_5", 8, 0x208, ["taskReferenceR3_DMA_CMX_5"]>;
def bf_taskReferenceAddr_DMA_CMX_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DMA_CMX_5", 64, 0, "UINT">;
def reg_taskReferenceAddr_DMA_CMX_5 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DMA_CMX_5", 8, 0x210, ["taskReferenceAddr_DMA_CMX_5"]>;
def bf_taskReferenceCount_DMA_CMX_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DMA_CMX_5", 64, 0>;
def reg_taskReferenceCount_DMA_CMX_5 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DMA_CMX_5", 8, 0x218, ["taskReferenceCount_DMA_CMX_5"]>;

// VpuTaskReference<VpuDPUInvariant> invariants
def bf_taskReferenceR1_DPU_inv_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_inv_0", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_inv_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_inv_0", 8, 0x220, ["taskReferenceR1_DPU_inv_0"]>;
def bf_taskReferenceR2_DPU_inv_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_inv_0", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_inv_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_inv_0", 8, 0x228, ["taskReferenceR2_DPU_inv_0"]>;
def bf_taskReferenceR3_DPU_inv_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_inv_0", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_inv_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_inv_0", 8, 0x230, ["taskReferenceR3_DPU_inv_0"]>;
def bf_taskReferenceAddr_DPU_inv_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_inv_0", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_inv_0 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_inv_0", 8, 0x238, ["taskReferenceAddr_DPU_inv_0"]>;
def bf_taskReferenceCount_DPU_inv_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_inv_0", 64, 0>;
def reg_taskReferenceCount_DPU_inv_0 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_inv_0", 8, 0x240, ["taskReferenceCount_DPU_inv_0"]>;

def bf_taskReferenceR1_DPU_inv_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_inv_1", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_inv_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_inv_1", 8, 0x248, ["taskReferenceR1_DPU_inv_1"]>;
def bf_taskReferenceR2_DPU_inv_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_inv_1", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_inv_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_inv_1", 8, 0x250, ["taskReferenceR2_DPU_inv_1"]>;
def bf_taskReferenceR3_DPU_inv_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_inv_1", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_inv_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_inv_1", 8, 0x258, ["taskReferenceR3_DPU_inv_1"]>;
def bf_taskReferenceAddr_DPU_inv_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_inv_1", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_inv_1 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_inv_1", 8, 0x260, ["taskReferenceAddr_DPU_inv_1"]>;
def bf_taskReferenceCount_DPU_inv_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_inv_1", 64, 0>;
def reg_taskReferenceCount_DPU_inv_1 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_inv_1", 8, 0x268, ["taskReferenceCount_DPU_inv_1"]>;

def bf_taskReferenceR1_DPU_inv_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_inv_2", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_inv_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_inv_2", 8, 0x270, ["taskReferenceR1_DPU_inv_2"]>;
def bf_taskReferenceR2_DPU_inv_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_inv_2", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_inv_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_inv_2", 8, 0x278, ["taskReferenceR2_DPU_inv_2"]>;
def bf_taskReferenceR3_DPU_inv_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_inv_2", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_inv_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_inv_2", 8, 0x280, ["taskReferenceR3_DPU_inv_2"]>;
def bf_taskReferenceAddr_DPU_inv_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_inv_2", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_inv_2 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_inv_2", 8, 0x288, ["taskReferenceAddr_DPU_inv_2"]>;
def bf_taskReferenceCount_DPU_inv_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_inv_2", 64, 0>;
def reg_taskReferenceCount_DPU_inv_2 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_inv_2", 8, 0x290, ["taskReferenceCount_DPU_inv_2"]>;

def bf_taskReferenceR1_DPU_inv_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_inv_3", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_inv_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_inv_3", 8, 0x298, ["taskReferenceR1_DPU_inv_3"]>;
def bf_taskReferenceR2_DPU_inv_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_inv_3", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_inv_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_inv_3", 8, 0x2A0, ["taskReferenceR2_DPU_inv_3"]>;
def bf_taskReferenceR3_DPU_inv_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_inv_3", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_inv_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_inv_3", 8, 0x2A8, ["taskReferenceR3_DPU_inv_3"]>;
def bf_taskReferenceAddr_DPU_inv_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_inv_3", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_inv_3 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_inv_3", 8, 0x2B0, ["taskReferenceAddr_DPU_inv_3"]>;
def bf_taskReferenceCount_DPU_inv_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_inv_3", 64, 0>;
def reg_taskReferenceCount_DPU_inv_3 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_inv_3", 8, 0x2B8, ["taskReferenceCount_DPU_inv_3"]>;

def bf_taskReferenceR1_DPU_inv_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_inv_4", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_inv_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_inv_4", 8, 0x2C0, ["taskReferenceR1_DPU_inv_4"]>;
def bf_taskReferenceR2_DPU_inv_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_inv_4", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_inv_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_inv_4", 8, 0x2C8, ["taskReferenceR2_DPU_inv_4"]>;
def bf_taskReferenceR3_DPU_inv_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_inv_4", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_inv_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_inv_4", 8, 0x2D0, ["taskReferenceR3_DPU_inv_4"]>;
def bf_taskReferenceAddr_DPU_inv_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_inv_4", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_inv_4 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_inv_4", 8, 0x2D8, ["taskReferenceAddr_DPU_inv_4"]>;
def bf_taskReferenceCount_DPU_inv_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_inv_4", 64, 0>;
def reg_taskReferenceCount_DPU_inv_4 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_inv_4", 8, 0x2E0, ["taskReferenceCount_DPU_inv_4"]>;

def bf_taskReferenceR1_DPU_inv_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_inv_5", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_inv_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_inv_5", 8, 0x2E8, ["taskReferenceR1_DPU_inv_5"]>;
def bf_taskReferenceR2_DPU_inv_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_inv_5", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_inv_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_inv_5", 8, 0x2F0, ["taskReferenceR2_DPU_inv_5"]>;
def bf_taskReferenceR3_DPU_inv_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_inv_5", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_inv_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_inv_5", 8, 0x2F8, ["taskReferenceR3_DPU_inv_5"]>;
def bf_taskReferenceAddr_DPU_inv_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_inv_5", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_inv_5 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_inv_5", 8, 0x300, ["taskReferenceAddr_DPU_inv_5"]>;
def bf_taskReferenceCount_DPU_inv_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_inv_5", 64, 0>;
def reg_taskReferenceCount_DPU_inv_5 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_inv_5", 8, 0x308, ["taskReferenceCount_DPU_inv_5"]>;

// VpuTaskReference<VpuDPUVariant> variants
def bf_taskReferenceR1_DPU_var_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_var_0", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_var_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_var_0", 8, 0x310, ["taskReferenceR1_DPU_var_0"]>;
def bf_taskReferenceR2_DPU_var_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_var_0", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_var_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_var_0", 8, 0x318, ["taskReferenceR2_DPU_var_0"]>;
def bf_taskReferenceR3_DPU_var_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_var_0", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_var_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_var_0", 8, 0x320, ["taskReferenceR3_DPU_var_0"]>;
def bf_taskReferenceAddr_DPU_var_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_var_0", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_var_0 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_var_0", 8, 0x328, ["taskReferenceAddr_DPU_var_0"]>;
def bf_taskReferenceCount_DPU_var_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_var_0", 64, 0>;
def reg_taskReferenceCount_DPU_var_0 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_var_0", 8, 0x330, ["taskReferenceCount_DPU_var_0"]>;

def bf_taskReferenceR1_DPU_var_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_var_1", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_var_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_var_1", 8, 0x338, ["taskReferenceR1_DPU_var_1"]>;
def bf_taskReferenceR2_DPU_var_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_var_1", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_var_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_var_1", 8, 0x340, ["taskReferenceR2_DPU_var_1"]>;
def bf_taskReferenceR3_DPU_var_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_var_1", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_var_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_var_1", 8, 0x348, ["taskReferenceR3_DPU_var_1"]>;
def bf_taskReferenceAddr_DPU_var_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_var_1", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_var_1 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_var_1", 8, 0x350, ["taskReferenceAddr_DPU_var_1"]>;
def bf_taskReferenceCount_DPU_var_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_var_1", 64, 0>;
def reg_taskReferenceCount_DPU_var_1 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_var_1", 8, 0x358, ["taskReferenceCount_DPU_var_1"]>;

def bf_taskReferenceR1_DPU_var_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_var_2", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_var_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_var_2", 8, 0x360, ["taskReferenceR1_DPU_var_2"]>;
def bf_taskReferenceR2_DPU_var_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_var_2", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_var_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_var_2", 8, 0x368, ["taskReferenceR2_DPU_var_2"]>;
def bf_taskReferenceR3_DPU_var_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_var_2", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_var_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_var_2", 8, 0x370, ["taskReferenceR3_DPU_var_2"]>;
def bf_taskReferenceAddr_DPU_var_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_var_2", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_var_2 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_var_2", 8, 0x378, ["taskReferenceAddr_DPU_var_2"]>;
def bf_taskReferenceCount_DPU_var_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_var_2", 64, 0>;
def reg_taskReferenceCount_DPU_var_2 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_var_2", 8, 0x380, ["taskReferenceCount_DPU_var_2"]>;

def bf_taskReferenceR1_DPU_var_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_var_3", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_var_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_var_3", 8, 0x388, ["taskReferenceR1_DPU_var_3"]>;
def bf_taskReferenceR2_DPU_var_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_var_3", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_var_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_var_3", 8, 0x390, ["taskReferenceR2_DPU_var_3"]>;
def bf_taskReferenceR3_DPU_var_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_var_3", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_var_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_var_3", 8, 0x398, ["taskReferenceR3_DPU_var_3"]>;
def bf_taskReferenceAddr_DPU_var_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_var_3", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_var_3 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_var_3", 8, 0x3A0, ["taskReferenceAddr_DPU_var_3"]>;
def bf_taskReferenceCount_DPU_var_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_var_3", 64, 0>;
def reg_taskReferenceCount_DPU_var_3 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_var_3", 8, 0x3A8, ["taskReferenceCount_DPU_var_3"]>;

def bf_taskReferenceR1_DPU_var_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_var_4", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_var_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_var_4", 8, 0x3B0, ["taskReferenceR1_DPU_var_4"]>;
def bf_taskReferenceR2_DPU_var_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_var_4", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_var_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_var_4", 8, 0x3B8, ["taskReferenceR2_DPU_var_4"]>;
def bf_taskReferenceR3_DPU_var_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_var_4", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_var_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_var_4", 8, 0x3C0, ["taskReferenceR3_DPU_var_4"]>;
def bf_taskReferenceAddr_DPU_var_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_var_4", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_var_4 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_var_4", 8, 0x3C8, ["taskReferenceAddr_DPU_var_4"]>;
def bf_taskReferenceCount_DPU_var_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_var_4", 64, 0>;
def reg_taskReferenceCount_DPU_var_4 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_var_4", 8, 0x3D0, ["taskReferenceCount_DPU_var_4"]>;

def bf_taskReferenceR1_DPU_var_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_DPU_var_5", 64, 0, "UINT">;
def reg_taskReferenceR1_DPU_var_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_DPU_var_5", 8, 0x3D8, ["taskReferenceR1_DPU_var_5"]>;
def bf_taskReferenceR2_DPU_var_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_DPU_var_5", 64, 0, "UINT">;
def reg_taskReferenceR2_DPU_var_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_DPU_var_5", 8, 0x3E0, ["taskReferenceR2_DPU_var_5"]>;
def bf_taskReferenceR3_DPU_var_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_DPU_var_5", 64, 0, "UINT">;
def reg_taskReferenceR3_DPU_var_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_DPU_var_5", 8, 0x3E8, ["taskReferenceR3_DPU_var_5"]>;
def bf_taskReferenceAddr_DPU_var_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_DPU_var_5", 64, 0, "UINT">;
def reg_taskReferenceAddr_DPU_var_5 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_DPU_var_5", 8, 0x3F0, ["taskReferenceAddr_DPU_var_5"]>;
def bf_taskReferenceCount_DPU_var_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_DPU_var_5", 64, 0>;
def reg_taskReferenceCount_DPU_var_5 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_DPU_var_5", 8, 0x3F8, ["taskReferenceCount_DPU_var_5"]>;

// VpuTaskReference<VpuActKernelRange> act_kernel_ranges
def bf_taskReferenceR1_ActKernel_range_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_range_0", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_range_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_range_0", 8, 0x400, ["taskReferenceR1_ActKernel_range_0"]>;
def bf_taskReferenceR2_ActKernel_range_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_range_0", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_range_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_range_0", 8, 0x408, ["taskReferenceR2_ActKernel_range_0"]>;
def bf_taskReferenceR3_ActKernel_range_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_range_0", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_range_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_range_0", 8, 0x410, ["taskReferenceR3_ActKernel_range_0"]>;
def bf_taskReferenceAddr_ActKernel_range_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_range_0", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_range_0 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_range_0", 8, 0x418, ["taskReferenceAddr_ActKernel_range_0"]>;
def bf_taskReferenceCount_ActKernel_range_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_range_0", 64, 0>;
def reg_taskReferenceCount_ActKernel_range_0 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_range_0", 8, 0x420, ["taskReferenceCount_ActKernel_range_0"]>;

def bf_taskReferenceR1_ActKernel_range_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_range_1", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_range_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_range_1", 8, 0x428, ["taskReferenceR1_ActKernel_range_1"]>;
def bf_taskReferenceR2_ActKernel_range_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_range_1", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_range_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_range_1", 8, 0x430, ["taskReferenceR2_ActKernel_range_1"]>;
def bf_taskReferenceR3_ActKernel_range_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_range_1", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_range_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_range_1", 8, 0x438, ["taskReferenceR3_ActKernel_range_1"]>;
def bf_taskReferenceAddr_ActKernel_range_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_range_1", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_range_1 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_range_1", 8, 0x440, ["taskReferenceAddr_ActKernel_range_1"]>;
def bf_taskReferenceCount_ActKernel_range_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_range_1", 64, 0>;
def reg_taskReferenceCount_ActKernel_range_1 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_range_1", 8, 0x448, ["taskReferenceCount_ActKernel_range_1"]>;

def bf_taskReferenceR1_ActKernel_range_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_range_2", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_range_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_range_2", 8, 0x450, ["taskReferenceR1_ActKernel_range_2"]>;
def bf_taskReferenceR2_ActKernel_range_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_range_2", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_range_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_range_2", 8, 0x458, ["taskReferenceR2_ActKernel_range_2"]>;
def bf_taskReferenceR3_ActKernel_range_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_range_2", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_range_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_range_2", 8, 0x460, ["taskReferenceR3_ActKernel_range_2"]>;
def bf_taskReferenceAddr_ActKernel_range_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_range_2", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_range_2 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_range_2", 8, 0x468, ["taskReferenceAddr_ActKernel_range_2"]>;
def bf_taskReferenceCount_ActKernel_range_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_range_2", 64, 0>;
def reg_taskReferenceCount_ActKernel_range_2 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_range_2", 8, 0x470, ["taskReferenceCount_ActKernel_range_2"]>;

def bf_taskReferenceR1_ActKernel_range_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_range_3", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_range_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_range_3", 8, 0x478, ["taskReferenceR1_ActKernel_range_3"]>;
def bf_taskReferenceR2_ActKernel_range_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_range_3", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_range_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_range_3", 8, 0x480, ["taskReferenceR2_ActKernel_range_3"]>;
def bf_taskReferenceR3_ActKernel_range_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_range_3", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_range_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_range_3", 8, 0x488, ["taskReferenceR3_ActKernel_range_3"]>;
def bf_taskReferenceAddr_ActKernel_range_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_range_3", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_range_3 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_range_3", 8, 0x490, ["taskReferenceAddr_ActKernel_range_3"]>;
def bf_taskReferenceCount_ActKernel_range_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_range_3", 64, 0>;
def reg_taskReferenceCount_ActKernel_range_3 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_range_3", 8, 0x498, ["taskReferenceCount_ActKernel_range_3"]>;

def bf_taskReferenceR1_ActKernel_range_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_range_4", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_range_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_range_4", 8, 0x4A0, ["taskReferenceR1_ActKernel_range_4"]>;
def bf_taskReferenceR2_ActKernel_range_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_range_4", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_range_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_range_4", 8, 0x4A8, ["taskReferenceR2_ActKernel_range_4"]>;
def bf_taskReferenceR3_ActKernel_range_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_range_4", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_range_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_range_4", 8, 0x4B0, ["taskReferenceR3_ActKernel_range_4"]>;
def bf_taskReferenceAddr_ActKernel_range_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_range_4", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_range_4 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_range_4", 8, 0x4B8, ["taskReferenceAddr_ActKernel_range_4"]>;
def bf_taskReferenceCount_ActKernel_range_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_range_4", 64, 0>;
def reg_taskReferenceCount_ActKernel_range_4 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_range_4", 8, 0x4C0, ["taskReferenceCount_ActKernel_range_4"]>;

def bf_taskReferenceR1_ActKernel_range_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_range_5", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_range_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_range_5", 8, 0x4C8, ["taskReferenceR1_ActKernel_range_5"]>;
def bf_taskReferenceR2_ActKernel_range_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_range_5", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_range_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_range_5", 8, 0x4D0, ["taskReferenceR2_ActKernel_range_5"]>;
def bf_taskReferenceR3_ActKernel_range_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_range_5", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_range_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_range_5", 8, 0x4D8, ["taskReferenceR3_ActKernel_range_5"]>;
def bf_taskReferenceAddr_ActKernel_range_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_range_5", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_range_5 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_range_5", 8, 0x4E0, ["taskReferenceAddr_ActKernel_range_5"]>;
def bf_taskReferenceCount_ActKernel_range_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_range_5", 64, 0>;
def reg_taskReferenceCount_ActKernel_range_5 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_range_5", 8, 0x4E8, ["taskReferenceCount_ActKernel_range_5"]>;

// VpuTaskReference<VpuActKernelInvocation> act_kernel_invocations
def bf_taskReferenceR1_ActKernel_invo_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_invo_0", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_invo_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_invo_0", 8, 0x4F0, ["taskReferenceR1_ActKernel_invo_0"]>;
def bf_taskReferenceR2_ActKernel_invo_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_invo_0", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_invo_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_invo_0", 8, 0x4F8, ["taskReferenceR2_ActKernel_invo_0"]>;
def bf_taskReferenceR3_ActKernel_invo_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_invo_0", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_invo_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_invo_0", 8, 0x500, ["taskReferenceR3_ActKernel_invo_0"]>;
def bf_taskReferenceAddr_ActKernel_invo_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_invo_0", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_invo_0 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_invo_0", 8, 0x508, ["taskReferenceAddr_ActKernel_invo_0"]>;
def bf_taskReferenceCount_ActKernel_invo_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_invo_0", 64, 0>;
def reg_taskReferenceCount_ActKernel_invo_0 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_invo_0", 8, 0x510, ["taskReferenceCount_ActKernel_invo_0"]>;

def bf_taskReferenceR1_ActKernel_invo_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_invo_1", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_invo_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_invo_1", 8, 0x518, ["taskReferenceR1_ActKernel_invo_1"]>;
def bf_taskReferenceR2_ActKernel_invo_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_invo_1", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_invo_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_invo_1", 8, 0x520, ["taskReferenceR2_ActKernel_invo_1"]>;
def bf_taskReferenceR3_ActKernel_invo_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_invo_1", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_invo_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_invo_1", 8, 0x528, ["taskReferenceR3_ActKernel_invo_1"]>;
def bf_taskReferenceAddr_ActKernel_invo_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_invo_1", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_invo_1 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_invo_1", 8, 0x530, ["taskReferenceAddr_ActKernel_invo_1"]>;
def bf_taskReferenceCount_ActKernel_invo_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_invo_1", 64, 0>;
def reg_taskReferenceCount_ActKernel_invo_1 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_invo_1", 8, 0x538, ["taskReferenceCount_ActKernel_invo_1"]>;

def bf_taskReferenceR1_ActKernel_invo_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_invo_2", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_invo_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_invo_2", 8, 0x540, ["taskReferenceR1_ActKernel_invo_2"]>;
def bf_taskReferenceR2_ActKernel_invo_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_invo_2", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_invo_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_invo_2", 8, 0x548, ["taskReferenceR2_ActKernel_invo_2"]>;
def bf_taskReferenceR3_ActKernel_invo_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_invo_2", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_invo_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_invo_2", 8, 0x550, ["taskReferenceR3_ActKernel_invo_2"]>;
def bf_taskReferenceAddr_ActKernel_invo_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_invo_2", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_invo_2 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_invo_2", 8, 0x558, ["taskReferenceAddr_ActKernel_invo_2"]>;
def bf_taskReferenceCount_ActKernel_invo_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_invo_2", 64, 0>;
def reg_taskReferenceCount_ActKernel_invo_2 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_invo_2", 8, 0x560, ["taskReferenceCount_ActKernel_invo_2"]>;

def bf_taskReferenceR1_ActKernel_invo_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_invo_3", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_invo_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_invo_3", 8, 0x568, ["taskReferenceR1_ActKernel_invo_3"]>;
def bf_taskReferenceR2_ActKernel_invo_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_invo_3", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_invo_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_invo_3", 8, 0x570, ["taskReferenceR2_ActKernel_invo_3"]>;
def bf_taskReferenceR3_ActKernel_invo_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_invo_3", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_invo_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_invo_3", 8, 0x578, ["taskReferenceR3_ActKernel_invo_3"]>;
def bf_taskReferenceAddr_ActKernel_invo_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_invo_3", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_invo_3 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_invo_3", 8, 0x580, ["taskReferenceAddr_ActKernel_invo_3"]>;
def bf_taskReferenceCount_ActKernel_invo_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_invo_3", 64, 0>;
def reg_taskReferenceCount_ActKernel_invo_3 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_invo_3", 8, 0x588, ["taskReferenceCount_ActKernel_invo_3"]>;

def bf_taskReferenceR1_ActKernel_invo_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_invo_4", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_invo_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_invo_4", 8, 0x590, ["taskReferenceR1_ActKernel_invo_4"]>;
def bf_taskReferenceR2_ActKernel_invo_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_invo_4", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_invo_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_invo_4", 8, 0x598, ["taskReferenceR2_ActKernel_invo_4"]>;
def bf_taskReferenceR3_ActKernel_invo_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_invo_4", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_invo_4 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_invo_4", 8, 0x5A0, ["taskReferenceR3_ActKernel_invo_4"]>;
def bf_taskReferenceAddr_ActKernel_invo_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_invo_4", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_invo_4 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_invo_4", 8, 0x5A8, ["taskReferenceAddr_ActKernel_invo_4"]>;
def bf_taskReferenceCount_ActKernel_invo_4 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_invo_4", 64, 0>;
def reg_taskReferenceCount_ActKernel_invo_4 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_invo_4", 8, 0x5B0, ["taskReferenceCount_ActKernel_invo_4"]>;

def bf_taskReferenceR1_ActKernel_invo_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ActKernel_invo_5", 64, 0, "UINT">;
def reg_taskReferenceR1_ActKernel_invo_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ActKernel_invo_5", 8, 0x5B8, ["taskReferenceR1_ActKernel_invo_5"]>;
def bf_taskReferenceR2_ActKernel_invo_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ActKernel_invo_5", 64, 0, "UINT">;
def reg_taskReferenceR2_ActKernel_invo_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ActKernel_invo_5", 8, 0x5C0, ["taskReferenceR2_ActKernel_invo_5"]>;
def bf_taskReferenceR3_ActKernel_invo_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ActKernel_invo_5", 64, 0, "UINT">;
def reg_taskReferenceR3_ActKernel_invo_5 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ActKernel_invo_5", 8, 0x5C8, ["taskReferenceR3_ActKernel_invo_5"]>;
def bf_taskReferenceAddr_ActKernel_invo_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ActKernel_invo_5", 64, 0, "UINT">;
def reg_taskReferenceAddr_ActKernel_invo_5 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ActKernel_invo_5", 8, 0x5D0, ["taskReferenceAddr_ActKernel_invo_5"]>;
def bf_taskReferenceCount_ActKernel_invo_5 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ActKernel_invo_5", 64, 0>;
def reg_taskReferenceCount_ActKernel_invo_5 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ActKernel_invo_5", 8, 0x5D8, ["taskReferenceCount_ActKernel_invo_5"]>;

// VpuTaskReference<VpuMediaTask> media_tasks
def bf_taskReferenceR1_MediaTask : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MediaTask", 64, 0, "UINT">;
def reg_taskReferenceR1_MediaTask : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MediaTask", 8, 0x5E0, ["taskReferenceR1_MediaTask"]>;
def bf_taskReferenceR2_MediaTask : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MediaTask", 64, 0, "UINT">;
def reg_taskReferenceR2_MediaTask : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MediaTask", 8, 0x5E8, ["taskReferenceR2_MediaTask"]>;
def bf_taskReferenceR3_MediaTask : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MediaTask", 64, 0, "UINT">;
def reg_taskReferenceR3_MediaTask : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MediaTask", 8, 0x5F0, ["taskReferenceR3_MediaTask"]>;
def bf_taskReferenceAddr_MediaTask : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MediaTask", 64, 0, "UINT">;
def reg_taskReferenceAddr_MediaTask : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MediaTask", 8, 0x5F8, ["taskReferenceAddr_MediaTask"]>;
def bf_taskReferenceCount_MediaTask : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MediaTask", 64, 0>;
def reg_taskReferenceCount_MediaTask : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MediaTask", 8, 0x600, ["taskReferenceCount_MediaTask"]>;

// VpuTaskReference<VpuBarrierCountConfig> barrier_configs
def bf_taskReferenceR1_BarrierConfig : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_BarrierConfig", 64, 0, "UINT">;
def reg_taskReferenceR1_BarrierConfig : NPUReg50XX_RegisterWrapper<"taskReferenceR1_BarrierConfig", 8, 0x608, ["taskReferenceR1_BarrierConfig"]>;
def bf_taskReferenceR2_BarrierConfig : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_BarrierConfig", 64, 0, "UINT">;
def reg_taskReferenceR2_BarrierConfig : NPUReg50XX_RegisterWrapper<"taskReferenceR2_BarrierConfig", 8, 0x610, ["taskReferenceR2_BarrierConfig"]>;
def bf_taskReferenceR3_BarrierConfig : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_BarrierConfig", 64, 0, "UINT">;
def reg_taskReferenceR3_BarrierConfig : NPUReg50XX_RegisterWrapper<"taskReferenceR3_BarrierConfig", 8, 0x618, ["taskReferenceR3_BarrierConfig"]>;
def bf_taskReferenceAddr_BarrierConfig : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_BarrierConfig", 64, 0, "UINT">;
def reg_taskReferenceAddr_BarrierConfig : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_BarrierConfig", 8, 0x620, ["taskReferenceAddr_BarrierConfig"]>;
def bf_taskReferenceCount_BarrierConfig : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_BarrierConfig", 64, 0>;
def reg_taskReferenceCount_BarrierConfig : NPUReg50XX_RegisterWrapper<"taskReferenceCount_BarrierConfig", 8, 0x628, ["taskReferenceCount_BarrierConfig"]>;

// MappedInference.VpuNNShaveRuntimeConfigs
def bf_MiNNRTCfg_reserved : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_reserved", 64, 0>;
def reg_MiNNRTCfg_reserved : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_reserved", 8, 0x630, ["MiNNRTCfg_reserved"]>;

def bf_MiNNRTCfg_runtime_entry : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_runtime_entry", 64, 0>;
def reg_MiNNRTCfg_runtime_entry : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_runtime_entry", 8, 0x638, ["MiNNRTCfg_runtime_entry"]>;

def bf_MiNNRTCfg_act_rt_window_base : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_act_rt_window_base", 64, 0>;
def reg_MiNNRTCfg_act_rt_window_base : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_act_rt_window_base", 8, 0x640, ["MiNNRTCfg_act_rt_window_base"]>;

// stack_frames
def bf_MiNNRTCfg_stack_0 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_0", 32, 0>;
def reg_MiNNRTCfg_stack_0 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_0", 4, 0x648, ["MiNNRTCfg_stack_0"]>;
def bf_MiNNRTCfg_stack_1 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_1", 32, 0>;
def reg_MiNNRTCfg_stack_1 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_1", 4, 0x64C, ["MiNNRTCfg_stack_1"]>;
def bf_MiNNRTCfg_stack_2 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_2", 32, 0>;
def reg_MiNNRTCfg_stack_2 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_2", 4, 0x650, ["MiNNRTCfg_stack_2"]>;
def bf_MiNNRTCfg_stack_3 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_3", 32, 0>;
def reg_MiNNRTCfg_stack_3 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_3", 4, 0x654, ["MiNNRTCfg_stack_3"]>;
def bf_MiNNRTCfg_stack_4 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_4", 32, 0>;
def reg_MiNNRTCfg_stack_4 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_4", 4, 0x658, ["MiNNRTCfg_stack_4"]>;
def bf_MiNNRTCfg_stack_5 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_5", 32, 0>;
def reg_MiNNRTCfg_stack_5 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_5", 4, 0x65C, ["MiNNRTCfg_stack_5"]>;
def bf_MiNNRTCfg_stack_6 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_6", 32, 0>;
def reg_MiNNRTCfg_stack_6 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_6", 4, 0x660, ["MiNNRTCfg_stack_6"]>;
def bf_MiNNRTCfg_stack_7 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_7", 32, 0>;
def reg_MiNNRTCfg_stack_7 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_7", 4, 0x664, ["MiNNRTCfg_stack_7"]>;
def bf_MiNNRTCfg_stack_8 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_8", 32, 0>;
def reg_MiNNRTCfg_stack_8 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_8", 4, 0x668, ["MiNNRTCfg_stack_8"]>;
def bf_MiNNRTCfg_stack_9 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_9", 32, 0>;
def reg_MiNNRTCfg_stack_9 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_9", 4, 0x66C, ["MiNNRTCfg_stack_9"]>;
def bf_MiNNRTCfg_stack_10 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_10", 32, 0>;
def reg_MiNNRTCfg_stack_10 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_10", 4, 0x670, ["MiNNRTCfg_stack_10"]>;
def bf_MiNNRTCfg_stack_11 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_11", 32, 0>;
def reg_MiNNRTCfg_stack_11 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_11", 4, 0x674, ["MiNNRTCfg_stack_11"]>;

def bf_MiNNRTCfg_stack_size : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_stack_size", 32, 0>;
def reg_MiNNRTCfg_stack_size : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_stack_size", 4, 0x678, ["MiNNRTCfg_stack_size"]>;

def bf_MiNNRTCfg_code_window_buffer_size : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_code_window_buffer_size", 32, 0>;
def reg_MiNNRTCfg_code_window_buffer_size : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_code_window_buffer_size", 4, 0x67C, ["MiNNRTCfg_code_window_buffer_size"]>;

def bf_MiNNRTCfg_perf_metrics_mask : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_perf_metrics_mask", 32, 0>;
def reg_MiNNRTCfg_perf_metrics_mask : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_perf_metrics_mask", 4, 0x680, ["MiNNRTCfg_perf_metrics_mask"]>;

def bf_MiNNRTCfg_runtime_version : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_runtime_version", 32, 0>;
def reg_MiNNRTCfg_runtime_version : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_runtime_version", 4, 0x684, ["MiNNRTCfg_runtime_version"]>;

def bf_MiNNRTCfg_use_schedule_embedded_rt : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_use_schedule_embedded_rt", 8, 0>;
def reg_MiNNRTCfg_use_schedule_embedded_rt : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_use_schedule_embedded_rt", 1, 0x688, ["MiNNRTCfg_use_schedule_embedded_rt"]>;

def bf_MiNNRTCfg_dpu_perf_mode : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_dpu_perf_mode", 8, 0>;
def reg_MiNNRTCfg_dpu_perf_mode : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_dpu_perf_mode", 1, 0x689, ["MiNNRTCfg_dpu_perf_mode"]>;

def bf_MiNNRTCfg_pad_6 : NPUReg50XX_RegFieldWrapper<"MiNNRTCfg_pad_6", 48, 0>;
def reg_MiNNRTCfg_pad_6 : NPUReg50XX_RegisterWrapper<"MiNNRTCfg_pad_6", 6, 0x68A, ["MiNNRTCfg_pad_6"]>;

// uint64_t hwp_workpoint_cfg_addr
def bf_MihHwpWorkpointCfgAddr : NPUReg50XX_RegFieldWrapper<"MihHwpWorkpointCfgAddr", 64, 0>;
def reg_MihHwpWorkpointCfgAddr : NPUReg50XX_RegisterWrapper<"MihHwpWorkpointCfgAddr", 8, 0x690, ["MihHwpWorkpointCfgAddr"]>;

// VpuTaskReference<uint8_t> managed_inference
def bf_taskReferenceR1_ManagedInference : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_ManagedInference", 64, 0, "UINT">;
def reg_taskReferenceR1_ManagedInference : NPUReg50XX_RegisterWrapper<"taskReferenceR1_ManagedInference", 8, 0x698, ["taskReferenceR1_ManagedInference"]>;
def bf_taskReferenceR2_ManagedInference : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_ManagedInference", 64, 0, "UINT">;
def reg_taskReferenceR2_ManagedInference : NPUReg50XX_RegisterWrapper<"taskReferenceR2_ManagedInference", 8, 0x6A0, ["taskReferenceR2_ManagedInference"]>;
def bf_taskReferenceR3_ManagedInference : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_ManagedInference", 64, 0, "UINT">;
def reg_taskReferenceR3_ManagedInference : NPUReg50XX_RegisterWrapper<"taskReferenceR3_ManagedInference", 8, 0x6A8, ["taskReferenceR3_ManagedInference"]>;
def bf_taskReferenceAddr_ManagedInference : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_ManagedInference", 64, 0, "UINT">;
def reg_taskReferenceAddr_ManagedInference : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_ManagedInference", 8, 0x6B0, ["taskReferenceAddr_ManagedInference"]>;
def bf_taskReferenceCount_ManagedInference : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_ManagedInference", 64, 0>;
def reg_taskReferenceCount_ManagedInference : NPUReg50XX_RegisterWrapper<"taskReferenceCount_ManagedInference", 8, 0x6B8, ["taskReferenceCount_ManagedInference"]>;

def reg_mapped_VpuMappedInferenceDescriptor : NPUReg50XX_RegMappedWrapper<"VpuMappedInference",
    ["miVpuNNRTApiVer", "miPad0", "miReserved0", "miLogAddrDmaHwp", "miTcReserved1", "miTcReserved2", "miTcDmaDDRCount",
     "miTcDmaCMXCount", "miDPUInvariantCount", "miTcDPUVariantCount", "miTcActRangeCount", "miTcActInvoCount", "miTcMediaCount", "miTaskStorageSize",

     "taskReferenceR1_DMA_DDR_0", "taskReferenceR2_DMA_DDR_0", "taskReferenceR3_DMA_DDR_0", "taskReferenceAddr_DMA_DDR_0", "taskReferenceCount_DMA_DDR_0",
     "taskReferenceR1_DMA_DDR_1", "taskReferenceR2_DMA_DDR_1", "taskReferenceR3_DMA_DDR_1", "taskReferenceAddr_DMA_DDR_1", "taskReferenceCount_DMA_DDR_1",
     "taskReferenceR1_DMA_DDR_2", "taskReferenceR2_DMA_DDR_2", "taskReferenceR3_DMA_DDR_2", "taskReferenceAddr_DMA_DDR_2", "taskReferenceCount_DMA_DDR_2",
     "taskReferenceR1_DMA_DDR_3", "taskReferenceR2_DMA_DDR_3", "taskReferenceR3_DMA_DDR_3", "taskReferenceAddr_DMA_DDR_3", "taskReferenceCount_DMA_DDR_3",
     "taskReferenceR1_DMA_DDR_4", "taskReferenceR2_DMA_DDR_4", "taskReferenceR3_DMA_DDR_4", "taskReferenceAddr_DMA_DDR_4", "taskReferenceCount_DMA_DDR_4",
     "taskReferenceR1_DMA_DDR_5", "taskReferenceR2_DMA_DDR_5", "taskReferenceR3_DMA_DDR_5", "taskReferenceAddr_DMA_DDR_5", "taskReferenceCount_DMA_DDR_5",

     "taskReferenceR1_DMA_CMX_0", "taskReferenceR2_DMA_CMX_0", "taskReferenceR3_DMA_CMX_0", "taskReferenceAddr_DMA_CMX_0", "taskReferenceCount_DMA_CMX_0",
     "taskReferenceR1_DMA_CMX_1", "taskReferenceR2_DMA_CMX_1", "taskReferenceR3_DMA_CMX_1", "taskReferenceAddr_DMA_CMX_1", "taskReferenceCount_DMA_CMX_1",
     "taskReferenceR1_DMA_CMX_2", "taskReferenceR2_DMA_CMX_2", "taskReferenceR3_DMA_CMX_2", "taskReferenceAddr_DMA_CMX_2", "taskReferenceCount_DMA_CMX_2",
     "taskReferenceR1_DMA_CMX_3", "taskReferenceR2_DMA_CMX_3", "taskReferenceR3_DMA_CMX_3", "taskReferenceAddr_DMA_CMX_3", "taskReferenceCount_DMA_CMX_3",
     "taskReferenceR1_DMA_CMX_4", "taskReferenceR2_DMA_CMX_4", "taskReferenceR3_DMA_CMX_4", "taskReferenceAddr_DMA_CMX_4", "taskReferenceCount_DMA_CMX_4",
     "taskReferenceR1_DMA_CMX_5", "taskReferenceR2_DMA_CMX_5", "taskReferenceR3_DMA_CMX_5", "taskReferenceAddr_DMA_CMX_5", "taskReferenceCount_DMA_CMX_5",

     "taskReferenceR1_DPU_inv_0", "taskReferenceR2_DPU_inv_0", "taskReferenceR3_DPU_inv_0", "taskReferenceAddr_DPU_inv_0", "taskReferenceCount_DPU_inv_0",
     "taskReferenceR1_DPU_inv_1", "taskReferenceR2_DPU_inv_1", "taskReferenceR3_DPU_inv_1", "taskReferenceAddr_DPU_inv_1", "taskReferenceCount_DPU_inv_1",
     "taskReferenceR1_DPU_inv_2", "taskReferenceR2_DPU_inv_2", "taskReferenceR3_DPU_inv_2", "taskReferenceAddr_DPU_inv_2", "taskReferenceCount_DPU_inv_2",
     "taskReferenceR1_DPU_inv_3", "taskReferenceR2_DPU_inv_3", "taskReferenceR3_DPU_inv_3", "taskReferenceAddr_DPU_inv_3", "taskReferenceCount_DPU_inv_3",
     "taskReferenceR1_DPU_inv_4", "taskReferenceR2_DPU_inv_4", "taskReferenceR3_DPU_inv_4", "taskReferenceAddr_DPU_inv_4", "taskReferenceCount_DPU_inv_4",
     "taskReferenceR1_DPU_inv_5", "taskReferenceR2_DPU_inv_5", "taskReferenceR3_DPU_inv_5", "taskReferenceAddr_DPU_inv_5", "taskReferenceCount_DPU_inv_5",

     "taskReferenceR1_DPU_var_0", "taskReferenceR2_DPU_var_0", "taskReferenceR3_DPU_var_0", "taskReferenceAddr_DPU_var_0", "taskReferenceCount_DPU_var_0",
     "taskReferenceR1_DPU_var_1", "taskReferenceR2_DPU_var_1", "taskReferenceR3_DPU_var_1", "taskReferenceAddr_DPU_var_1", "taskReferenceCount_DPU_var_1",
     "taskReferenceR1_DPU_var_2", "taskReferenceR2_DPU_var_2", "taskReferenceR3_DPU_var_2", "taskReferenceAddr_DPU_var_2", "taskReferenceCount_DPU_var_2",
     "taskReferenceR1_DPU_var_3", "taskReferenceR2_DPU_var_3", "taskReferenceR3_DPU_var_3", "taskReferenceAddr_DPU_var_3", "taskReferenceCount_DPU_var_3",
     "taskReferenceR1_DPU_var_4", "taskReferenceR2_DPU_var_4", "taskReferenceR3_DPU_var_4", "taskReferenceAddr_DPU_var_4", "taskReferenceCount_DPU_var_4",
     "taskReferenceR1_DPU_var_5", "taskReferenceR2_DPU_var_5", "taskReferenceR3_DPU_var_5", "taskReferenceAddr_DPU_var_5", "taskReferenceCount_DPU_var_5",

     "taskReferenceR1_ActKernel_range_0", "taskReferenceR2_ActKernel_range_0", "taskReferenceR3_ActKernel_range_0", "taskReferenceAddr_ActKernel_range_0", "taskReferenceCount_ActKernel_range_0",
     "taskReferenceR1_ActKernel_range_1", "taskReferenceR2_ActKernel_range_1", "taskReferenceR3_ActKernel_range_1", "taskReferenceAddr_ActKernel_range_1", "taskReferenceCount_ActKernel_range_1",
     "taskReferenceR1_ActKernel_range_2", "taskReferenceR2_ActKernel_range_2", "taskReferenceR3_ActKernel_range_2", "taskReferenceAddr_ActKernel_range_2", "taskReferenceCount_ActKernel_range_2",
     "taskReferenceR1_ActKernel_range_3", "taskReferenceR2_ActKernel_range_3", "taskReferenceR3_ActKernel_range_3", "taskReferenceAddr_ActKernel_range_3", "taskReferenceCount_ActKernel_range_3",
     "taskReferenceR1_ActKernel_range_4", "taskReferenceR2_ActKernel_range_4", "taskReferenceR3_ActKernel_range_4", "taskReferenceAddr_ActKernel_range_4", "taskReferenceCount_ActKernel_range_4",
     "taskReferenceR1_ActKernel_range_5", "taskReferenceR2_ActKernel_range_5", "taskReferenceR3_ActKernel_range_5", "taskReferenceAddr_ActKernel_range_5", "taskReferenceCount_ActKernel_range_5",

     "taskReferenceR1_ActKernel_invo_0", "taskReferenceR2_ActKernel_invo_0", "taskReferenceR3_ActKernel_invo_0", "taskReferenceAddr_ActKernel_invo_0", "taskReferenceCount_ActKernel_invo_0",
     "taskReferenceR1_ActKernel_invo_1", "taskReferenceR2_ActKernel_invo_1", "taskReferenceR3_ActKernel_invo_1", "taskReferenceAddr_ActKernel_invo_1", "taskReferenceCount_ActKernel_invo_1",
     "taskReferenceR1_ActKernel_invo_2", "taskReferenceR2_ActKernel_invo_2", "taskReferenceR3_ActKernel_invo_2", "taskReferenceAddr_ActKernel_invo_2", "taskReferenceCount_ActKernel_invo_2",
     "taskReferenceR1_ActKernel_invo_3", "taskReferenceR2_ActKernel_invo_3", "taskReferenceR3_ActKernel_invo_3", "taskReferenceAddr_ActKernel_invo_3", "taskReferenceCount_ActKernel_invo_3",
     "taskReferenceR1_ActKernel_invo_4", "taskReferenceR2_ActKernel_invo_4", "taskReferenceR3_ActKernel_invo_4", "taskReferenceAddr_ActKernel_invo_4", "taskReferenceCount_ActKernel_invo_4",
     "taskReferenceR1_ActKernel_invo_5", "taskReferenceR2_ActKernel_invo_5", "taskReferenceR3_ActKernel_invo_5", "taskReferenceAddr_ActKernel_invo_5", "taskReferenceCount_ActKernel_invo_5",

     "taskReferenceR1_MediaTask", "taskReferenceR2_MediaTask", "taskReferenceR3_MediaTask", "taskReferenceAddr_MediaTask", "taskReferenceCount_MediaTask",
     "taskReferenceR1_BarrierConfig", "taskReferenceR2_BarrierConfig", "taskReferenceR3_BarrierConfig", "taskReferenceAddr_BarrierConfig", "taskReferenceCount_BarrierConfig",
     "MiNNRTCfg_reserved", "MiNNRTCfg_runtime_entry", "MiNNRTCfg_act_rt_window_base", "MiNNRTCfg_stack_0", "MiNNRTCfg_stack_1", "MiNNRTCfg_stack_2", "MiNNRTCfg_stack_3",
     "MiNNRTCfg_stack_4", "MiNNRTCfg_stack_5", "MiNNRTCfg_stack_6", "MiNNRTCfg_stack_7", "MiNNRTCfg_stack_8", "MiNNRTCfg_stack_9", "MiNNRTCfg_stack_10", "MiNNRTCfg_stack_11",
     "MiNNRTCfg_stack_size", "MiNNRTCfg_code_window_buffer_size", "MiNNRTCfg_perf_metrics_mask", "MiNNRTCfg_runtime_version", "MiNNRTCfg_use_schedule_embedded_rt",
     "MiNNRTCfg_dpu_perf_mode", "MiNNRTCfg_pad_6", "MihHwpWorkpointCfgAddr", "taskReferenceR1_ManagedInference", "taskReferenceR2_ManagedInference",
     "taskReferenceR3_ManagedInference", "taskReferenceAddr_ManagedInference", "taskReferenceCount_ManagedInference"
]>;


// VpuNNShaveRuntimeConfigs
def bf_NNRTCfg_reserved : NPUReg50XX_RegFieldWrapper<"NNRTCfg_reserved", 64, 0>;
def reg_NNRTCfg_reserved : NPUReg50XX_RegisterWrapper<"NNRTCfg_reserved", 8, 0x0, ["NNRTCfg_reserved"]>;

def bf_NNRTCfg_runtime_entry : NPUReg50XX_RegFieldWrapper<"NNRTCfg_runtime_entry", 64, 0>;
def reg_NNRTCfg_runtime_entry : NPUReg50XX_RegisterWrapper<"NNRTCfg_runtime_entry", 8, 0x8, ["NNRTCfg_runtime_entry"]>;

def bf_NNRTCfg_act_rt_window_base : NPUReg50XX_RegFieldWrapper<"NNRTCfg_act_rt_window_base", 64, 0>;
def reg_NNRTCfg_act_rt_window_base : NPUReg50XX_RegisterWrapper<"NNRTCfg_act_rt_window_base", 8, 0x10, ["NNRTCfg_act_rt_window_base"]>;

// stack_frames
def bf_NNRTCfg_stack_0 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_0", 32, 0>;
def reg_NNRTCfg_stack_0 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_0", 4, 0x18, ["NNRTCfg_stack_0"]>;
def bf_NNRTCfg_stack_1 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_1", 32, 0>;
def reg_NNRTCfg_stack_1 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_1", 4, 0x1C, ["NNRTCfg_stack_1"]>;
def bf_NNRTCfg_stack_2 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_2", 32, 0>;
def reg_NNRTCfg_stack_2 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_2", 4, 0x20, ["NNRTCfg_stack_2"]>;
def bf_NNRTCfg_stack_3 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_3", 32, 0>;
def reg_NNRTCfg_stack_3 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_3", 4, 0x24, ["NNRTCfg_stack_3"]>;
def bf_NNRTCfg_stack_4 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_4", 32, 0>;
def reg_NNRTCfg_stack_4 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_4", 4, 0x28, ["NNRTCfg_stack_4"]>;
def bf_NNRTCfg_stack_5 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_5", 32, 0>;
def reg_NNRTCfg_stack_5 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_5", 4, 0x2C, ["NNRTCfg_stack_5"]>;
def bf_NNRTCfg_stack_6 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_6", 32, 0>;
def reg_NNRTCfg_stack_6 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_6", 4, 0x30, ["NNRTCfg_stack_6"]>;
def bf_NNRTCfg_stack_7 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_7", 32, 0>;
def reg_NNRTCfg_stack_7 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_7", 4, 0x34, ["NNRTCfg_stack_7"]>;
def bf_NNRTCfg_stack_8 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_8", 32, 0>;
def reg_NNRTCfg_stack_8 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_8", 4, 0x38, ["NNRTCfg_stack_8"]>;
def bf_NNRTCfg_stack_9 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_9", 32, 0>;
def reg_NNRTCfg_stack_9 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_9", 4, 0x3C, ["NNRTCfg_stack_9"]>;
def bf_NNRTCfg_stack_10 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_10", 32, 0>;
def reg_NNRTCfg_stack_10 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_10", 4, 0x40, ["NNRTCfg_stack_10"]>;
def bf_NNRTCfg_stack_11 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_11", 32, 0>;
def reg_NNRTCfg_stack_11 : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_11", 4, 0x44, ["NNRTCfg_stack_11"]>;

def bf_NNRTCfg_stack_size : NPUReg50XX_RegFieldWrapper<"NNRTCfg_stack_size", 32, 0>;
def reg_NNRTCfg_stack_size : NPUReg50XX_RegisterWrapper<"NNRTCfg_stack_size", 4, 0x48, ["NNRTCfg_stack_size"]>;

def bf_NNRTCfg_code_window_buffer_size : NPUReg50XX_RegFieldWrapper<"NNRTCfg_code_window_buffer_size", 32, 0>;
def reg_NNRTCfg_code_window_buffer_size : NPUReg50XX_RegisterWrapper<"NNRTCfg_code_window_buffer_size", 4, 0x4C, ["NNRTCfg_code_window_buffer_size"]>;

def bf_NNRTCfg_perf_metrics_mask : NPUReg50XX_RegFieldWrapper<"NNRTCfg_perf_metrics_mask", 32, 0>;
def reg_NNRTCfg_perf_metrics_mask : NPUReg50XX_RegisterWrapper<"NNRTCfg_perf_metrics_mask", 4, 0x50, ["NNRTCfg_perf_metrics_mask"]>;

def bf_NNRTCfg_runtime_version : NPUReg50XX_RegFieldWrapper<"NNRTCfg_runtime_version", 32, 0>;
def reg_NNRTCfg_runtime_version : NPUReg50XX_RegisterWrapper<"NNRTCfg_runtime_version", 4, 0x54, ["NNRTCfg_runtime_version"]>;

def bf_NNRTCfg_use_schedule_embedded_rt : NPUReg50XX_RegFieldWrapper<"NNRTCfg_use_schedule_embedded_rt", 8, 0>;
def reg_NNRTCfg_use_schedule_embedded_rt : NPUReg50XX_RegisterWrapper<"NNRTCfg_use_schedule_embedded_rt", 1, 0x58, ["NNRTCfg_use_schedule_embedded_rt"]>;

def bf_NNRTCfg_dpu_perf_mode : NPUReg50XX_RegFieldWrapper<"NNRTCfg_dpu_perf_mode", 8, 0>;
def reg_NNRTCfg_dpu_perf_mode : NPUReg50XX_RegisterWrapper<"NNRTCfg_dpu_perf_mode", 1, 0x59, ["NNRTCfg_dpu_perf_mode"]>;

def bf_NNRTCfg_pad_6 : NPUReg50XX_RegFieldWrapper<"NNRTCfg_pad_6", 48, 0>;
def reg_NNRTCfg_pad_6 : NPUReg50XX_RegisterWrapper<"NNRTCfg_pad_6", 6, 0x5A, ["NNRTCfg_pad_6"]>;

def reg_mapped_VpuNNShaveRuntimeConfigsDescriptor : NPUReg50XX_RegMappedWrapper<"VpuNNShaveRuntimeConfigs",
    ["NNRTCfg_reserved", "NNRTCfg_runtime_entry", "NNRTCfg_act_rt_window_base", "NNRTCfg_stack_0", "NNRTCfg_stack_1", "NNRTCfg_stack_2",
     "NNRTCfg_stack_3", "NNRTCfg_stack_4", "NNRTCfg_stack_5", "NNRTCfg_stack_6", "NNRTCfg_stack_7", "NNRTCfg_stack_8", "NNRTCfg_stack_9",
     "NNRTCfg_stack_10", "NNRTCfg_stack_11", "NNRTCfg_stack_size", "NNRTCfg_code_window_buffer_size", "NNRTCfg_perf_metrics_mask",
     "NNRTCfg_runtime_version", "NNRTCfg_use_schedule_embedded_rt", "NNRTCfg_dpu_perf_mode", "NNRTCfg_pad_6"
    ]>;

//VpuNNRTConfig
// this contains VpuNNShaveRuntimeConfigs (see above), and it has two aditional fields:

def bf_NNRTCfg_logAddrDmaHwp : NPUReg50XX_RegFieldWrapper<"NNRTCfg_logAddrDmaHwp", 64, 0>;
def reg_NNRTCfg_logAddrDmaHwp : NPUReg50XX_RegisterWrapper<"NNRTCfg_logAddrDmaHwp", 8, 0x60, ["NNRTCfg_logAddrDmaHwp"]>;

def bf_NNRTCfg_HwpCfgAddr : NPUReg50XX_RegFieldWrapper<"NNRTCfg_HwpCfgAddr", 64, 0>;
def reg_NNRTCfg_HwpCfgAddr : NPUReg50XX_RegisterWrapper<"NNRTCfg_HwpCfgAddr", 8, 0x68, ["NNRTCfg_HwpCfgAddr"]>;

def reg_mapped_VpuNNRTConfigDescriptor : NPUReg50XX_RegMappedWrapper<"VpuNNRTConfig",
    ["NNRTCfg_reserved", "NNRTCfg_runtime_entry", "NNRTCfg_act_rt_window_base", "NNRTCfg_stack_0", "NNRTCfg_stack_1", "NNRTCfg_stack_2", "NNRTCfg_stack_3",
     "NNRTCfg_stack_4", "NNRTCfg_stack_5", "NNRTCfg_stack_6", "NNRTCfg_stack_7", "NNRTCfg_stack_8", "NNRTCfg_stack_9", "NNRTCfg_stack_10", "NNRTCfg_stack_11",
     "NNRTCfg_stack_size", "NNRTCfg_code_window_buffer_size", "NNRTCfg_perf_metrics_mask", "NNRTCfg_runtime_version", "NNRTCfg_use_schedule_embedded_rt",
     "NNRTCfg_dpu_perf_mode", "NNRTCfg_pad_6", "NNRTCfg_logAddrDmaHwp", "NNRTCfg_HwpCfgAddr"
    ]>;


//
// VpuBarrierConfiguration
//
def bf_BarrierConfiguration_producerCount : NPUReg50XX_RegFieldWrapper<"BarrierConfiguration_producerCount", 8, 0, "UINT", NPUReg50XX_Version<11, 5, 0>>;
def reg_BarrierConfiguration_producerCount : NPUReg50XX_RegisterWrapper<"BarrierConfiguration_producerCount", 1, 0x0, ["BarrierConfiguration_producerCount"]>;

def bf_BarrierConfiguration_producerInterruptEnabled : NPUReg50XX_RegFieldWrapper<"BarrierConfiguration_producerInterruptEnabled", 8, 0, "UINT", NPUReg50XX_Version<11, 5, 0>>;
def reg_BarrierConfiguration_producerInterruptEnabled : NPUReg50XX_RegisterWrapper<"BarrierConfiguration_producerInterruptEnabled", 1, 0x1, ["BarrierConfiguration_producerInterruptEnabled"]>;

def bf_BarrierConfiguration_consumerCount : NPUReg50XX_RegFieldWrapper<"BarrierConfiguration_consumerCount", 8, 0, "UINT", NPUReg50XX_Version<11, 5, 0>>;
def reg_BarrierConfiguration_consumerCount : NPUReg50XX_RegisterWrapper<"BarrierConfiguration_consumerCount", 1, 0x2, ["BarrierConfiguration_consumerCount"]>;

def bf_BarrierConfiguration_consumerInterruptEnabled : NPUReg50XX_RegFieldWrapper<"BarrierConfiguration_consumerInterruptEnabled", 8, 0, "UINT", NPUReg50XX_Version<11, 5, 0>>;
def reg_BarrierConfiguration_consumerInterruptEnabled : NPUReg50XX_RegisterWrapper<"BarrierConfiguration_consumerInterruptEnabled", 1, 0x3, ["BarrierConfiguration_consumerInterruptEnabled"]>;


def reg_mapped_VpuBarrierConfigurationDescriptor : NPUReg50XX_RegMappedWrapper<"VpuBarrierConfiguration",
    [
        "BarrierConfiguration_producerCount", "BarrierConfiguration_producerInterruptEnabled",
        "BarrierConfiguration_consumerCount", "BarrierConfiguration_consumerInterruptEnabled"
    ]>;


//
// VpuManagedMappedInferenceInfo
//

// VpuTaskReference<VpuTaskInfo> tasks_ref_info
def bf_taskReferenceR1_MMII_tasks_ref_info : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMII_tasks_ref_info", 64, 0, "UINT">;
def reg_taskReferenceR1_MMII_tasks_ref_info : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMII_tasks_ref_info", 8, 0x0, ["taskReferenceR1_MMII_tasks_ref_info"]>;
def bf_taskReferenceR2_MMII_tasks_ref_info : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMII_tasks_ref_info", 64, 0, "UINT">;
def reg_taskReferenceR2_MMII_tasks_ref_info : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMII_tasks_ref_info", 8, 0x8, ["taskReferenceR2_MMII_tasks_ref_info"]>;
def bf_taskReferenceR3_MMII_tasks_ref_info : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMII_tasks_ref_info", 64, 0, "UINT">;
def reg_taskReferenceR3_MMII_tasks_ref_info : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMII_tasks_ref_info", 8, 0x10, ["taskReferenceR3_MMII_tasks_ref_info"]>;
def bf_taskReferenceAddr_MMII_tasks_ref_info : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMII_tasks_ref_info", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMII_tasks_ref_info : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMII_tasks_ref_info", 8, 0x18, ["taskReferenceAddr_MMII_tasks_ref_info"]>;
def bf_taskReferenceCount_MMII_tasks_ref_info : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMII_tasks_ref_info", 64, 0>;
def reg_taskReferenceCount_MMII_tasks_ref_info : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMII_tasks_ref_info", 8, 0x20, ["taskReferenceCount_MMII_tasks_ref_info"]>;

// VpuTaskReference<BarrierReferenceMap> vb_mapping
def bf_taskReferenceR1_MMII_vb_mapping : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMII_vb_mapping", 64, 0, "UINT">;
def reg_taskReferenceR1_MMII_vb_mapping : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMII_vb_mapping", 8, 0x28, ["taskReferenceR1_MMII_vb_mapping"]>;
def bf_taskReferenceR2_MMII_vb_mapping : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMII_vb_mapping", 64, 0, "UINT">;
def reg_taskReferenceR2_MMII_vb_mapping : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMII_vb_mapping", 8, 0x30, ["taskReferenceR2_MMII_vb_mapping"]>;
def bf_taskReferenceR3_MMII_vb_mapping : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMII_vb_mapping", 64, 0, "UINT">;
def reg_taskReferenceR3_MMII_vb_mapping : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMII_vb_mapping", 8, 0x38, ["taskReferenceR3_MMII_vb_mapping"]>;
def bf_taskReferenceAddr_MMII_vb_mapping : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMII_vb_mapping", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMII_vb_mapping : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMII_vb_mapping", 8, 0x40, ["taskReferenceAddr_MMII_vb_mapping"]>;
def bf_taskReferenceCount_MMII_vb_mapping : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMII_vb_mapping", 64, 0>;
def reg_taskReferenceCount_MMII_vb_mapping : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMII_vb_mapping", 8, 0x48, ["taskReferenceCount_MMII_vb_mapping"]>;

// VpuTaskReference<uint32_t> barrier_producer_ref_offsets
def bf_taskReferenceR1_MMII_barrier_producer_ref_offsets : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMII_barrier_producer_ref_offsets", 64, 0, "UINT">;
def reg_taskReferenceR1_MMII_barrier_producer_ref_offsets : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMII_barrier_producer_ref_offsets", 8, 0x50, ["taskReferenceR1_MMII_barrier_producer_ref_offsets"]>;
def bf_taskReferenceR2_MMII_barrier_producer_ref_offsets : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMII_barrier_producer_ref_offsets", 64, 0, "UINT">;
def reg_taskReferenceR2_MMII_barrier_producer_ref_offsets : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMII_barrier_producer_ref_offsets", 8, 0x58, ["taskReferenceR2_MMII_barrier_producer_ref_offsets"]>;
def bf_taskReferenceR3_MMII_barrier_producer_ref_offsets : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMII_barrier_producer_ref_offsets", 64, 0, "UINT">;
def reg_taskReferenceR3_MMII_barrier_producer_ref_offsets : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMII_barrier_producer_ref_offsets", 8, 0x60, ["taskReferenceR3_MMII_barrier_producer_ref_offsets"]>;
def bf_taskReferenceAddr_MMII_barrier_producer_ref_offsets : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMII_barrier_producer_ref_offsets", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMII_barrier_producer_ref_offsets : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMII_barrier_producer_ref_offsets", 8, 0x68, ["taskReferenceAddr_MMII_barrier_producer_ref_offsets"]>;
def bf_taskReferenceCount_MMII_barrier_producer_ref_offsets : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMII_barrier_producer_ref_offsets", 64, 0>;
def reg_taskReferenceCount_MMII_barrier_producer_ref_offsets : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMII_barrier_producer_ref_offsets", 8, 0x70, ["taskReferenceCount_MMII_barrier_producer_ref_offsets"]>;

// VpuTaskReference<uint32_t> barrier_consumer_ref_offsets
def bf_taskReferenceR1_MMII_barrier_consumer_ref_offsets : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMII_barrier_consumer_ref_offsets", 64, 0, "UINT">;
def reg_taskReferenceR1_MMII_barrier_consumer_ref_offsets : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMII_barrier_consumer_ref_offsets", 8, 0x78, ["taskReferenceR1_MMII_barrier_consumer_ref_offsets"]>;
def bf_taskReferenceR2_MMII_barrier_consumer_ref_offsets : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMII_barrier_consumer_ref_offsets", 64, 0, "UINT">;
def reg_taskReferenceR2_MMII_barrier_consumer_ref_offsets : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMII_barrier_consumer_ref_offsets", 8, 0x80, ["taskReferenceR2_MMII_barrier_consumer_ref_offsets"]>;
def bf_taskReferenceR3_MMII_barrier_consumer_ref_offsets : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMII_barrier_consumer_ref_offsets", 64, 0, "UINT">;
def reg_taskReferenceR3_MMII_barrier_consumer_ref_offsets : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMII_barrier_consumer_ref_offsets", 8, 0x88, ["taskReferenceR3_MMII_barrier_consumer_ref_offsets"]>;
def bf_taskReferenceAddr_MMII_barrier_consumer_ref_offsets : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMII_barrier_consumer_ref_offsets", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMII_barrier_consumer_ref_offsets : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMII_barrier_consumer_ref_offsets", 8, 0x90, ["taskReferenceAddr_MMII_barrier_consumer_ref_offsets"]>;
def bf_taskReferenceCount_MMII_barrier_consumer_ref_offsets : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMII_barrier_consumer_ref_offsets", 64, 0>;
def reg_taskReferenceCount_MMII_barrier_consumer_ref_offsets : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMII_barrier_consumer_ref_offsets", 8, 0x98, ["taskReferenceCount_MMII_barrier_consumer_ref_offsets"]>;

// uint64_t ref_info_base_vars[VPU_MAX_TILES]
def bf_MMII_ref_info_base_vars_0 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_vars_0", 64, 0, "UINT">;
def reg_MMII_ref_info_base_vars_0 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_vars_0", 8, 0xA0, ["MMII_ref_info_base_vars_0"]>;
def bf_MMII_ref_info_base_vars_1 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_vars_1", 64, 0, "UINT">;
def reg_MMII_ref_info_base_vars_1 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_vars_1", 8, 0xA8, ["MMII_ref_info_base_vars_1"]>;
def bf_MMII_ref_info_base_vars_2 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_vars_2", 64, 0, "UINT">;
def reg_MMII_ref_info_base_vars_2 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_vars_2", 8, 0xB0, ["MMII_ref_info_base_vars_2"]>;
def bf_MMII_ref_info_base_vars_3 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_vars_3", 64, 0, "UINT">;
def reg_MMII_ref_info_base_vars_3 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_vars_3", 8, 0xB8, ["MMII_ref_info_base_vars_3"]>;
def bf_MMII_ref_info_base_vars_4 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_vars_4", 64, 0, "UINT">;
def reg_MMII_ref_info_base_vars_4 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_vars_4", 8, 0xC0, ["MMII_ref_info_base_vars_4"]>;
def bf_MMII_ref_info_base_vars_5 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_vars_5", 64, 0, "UINT">;
def reg_MMII_ref_info_base_vars_5 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_vars_5", 8, 0xC8, ["MMII_ref_info_base_vars_5"]>;

// uint64_t ref_info_base_invars[VPU_MAX_TILES]
def bf_MMII_ref_info_base_invars_0 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_invars_0", 64, 0, "UINT">;
def reg_MMII_ref_info_base_invars_0 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_invars_0", 8, 0xD0, ["MMII_ref_info_base_invars_0"]>;
def bf_MMII_ref_info_base_invars_1 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_invars_1", 64, 0, "UINT">;
def reg_MMII_ref_info_base_invars_1 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_invars_1", 8, 0xD8, ["MMII_ref_info_base_invars_1"]>;
def bf_MMII_ref_info_base_invars_2 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_invars_2", 64, 0, "UINT">;
def reg_MMII_ref_info_base_invars_2 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_invars_2", 8, 0xE0, ["MMII_ref_info_base_invars_2"]>;
def bf_MMII_ref_info_base_invars_3 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_invars_3", 64, 0, "UINT">;
def reg_MMII_ref_info_base_invars_3 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_invars_3", 8, 0xE8, ["MMII_ref_info_base_invars_3"]>;
def bf_MMII_ref_info_base_invars_4 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_invars_4", 64, 0, "UINT">;
def reg_MMII_ref_info_base_invars_4 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_invars_4", 8, 0xF0, ["MMII_ref_info_base_invars_4"]>;
def bf_MMII_ref_info_base_invars_5 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_invars_5", 64, 0, "UINT">;
def reg_MMII_ref_info_base_invars_5 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_invars_5", 8, 0xF8, ["MMII_ref_info_base_invars_5"]>;

// uint64_t ref_info_base_akr[VPU_MAX_TILES];
def bf_MMII_ref_info_base_akr_0 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_akr_0", 64, 0, "UINT">;
def reg_MMII_ref_info_base_akr_0 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_akr_0", 8, 0x100, ["MMII_ref_info_base_akr_0"]>;
def bf_MMII_ref_info_base_akr_1 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_akr_1", 64, 0, "UINT">;
def reg_MMII_ref_info_base_akr_1 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_akr_1", 8, 0x108, ["MMII_ref_info_base_akr_1"]>;
def bf_MMII_ref_info_base_akr_2 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_akr_2", 64, 0, "UINT">;
def reg_MMII_ref_info_base_akr_2 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_akr_2", 8, 0x110, ["MMII_ref_info_base_akr_2"]>;
def bf_MMII_ref_info_base_akr_3 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_akr_3", 64, 0, "UINT">;
def reg_MMII_ref_info_base_akr_3 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_akr_3", 8, 0x118, ["MMII_ref_info_base_akr_3"]>;
def bf_MMII_ref_info_base_akr_4 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_akr_4", 64, 0, "UINT">;
def reg_MMII_ref_info_base_akr_4 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_akr_4", 8, 0x120, ["MMII_ref_info_base_akr_4"]>;
def bf_MMII_ref_info_base_akr_5 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_akr_5", 64, 0, "UINT">;
def reg_MMII_ref_info_base_akr_5 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_akr_5", 8, 0x128, ["MMII_ref_info_base_akr_5"]>;

// uint64_t ref_info_base_aki[VPU_MAX_TILES]
def bf_MMII_ref_info_base_aki_0 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_aki_0", 64, 0, "UINT">;
def reg_MMII_ref_info_base_aki_0 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_aki_0", 8, 0x130, ["MMII_ref_info_base_aki_0"]>;
def bf_MMII_ref_info_base_aki_1 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_aki_1", 64, 0, "UINT">;
def reg_MMII_ref_info_base_aki_1 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_aki_1", 8, 0x138, ["MMII_ref_info_base_aki_1"]>;
def bf_MMII_ref_info_base_aki_2 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_aki_2", 64, 0, "UINT">;
def reg_MMII_ref_info_base_aki_2 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_aki_2", 8, 0x140, ["MMII_ref_info_base_aki_2"]>;
def bf_MMII_ref_info_base_aki_3 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_aki_3", 64, 0, "UINT">;
def reg_MMII_ref_info_base_aki_3 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_aki_3", 8, 0x148, ["MMII_ref_info_base_aki_3"]>;
def bf_MMII_ref_info_base_aki_4 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_aki_4", 64, 0, "UINT">;
def reg_MMII_ref_info_base_aki_4 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_aki_4", 8, 0x150, ["MMII_ref_info_base_aki_4"]>;
def bf_MMII_ref_info_base_aki_5 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_aki_5", 64, 0, "UINT">;
def reg_MMII_ref_info_base_aki_5 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_aki_5", 8, 0x158, ["MMII_ref_info_base_aki_5"]>;

// uint64_t ref_info_base_dma_from_ddr[VPU_MAX_DMA_ENGINES]
def bf_MMII_ref_info_base_dma_from_ddr_0 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_dma_from_ddr_0", 64, 0, "UINT">;
def reg_MMII_ref_info_base_dma_from_ddr_0 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_dma_from_ddr_0", 8, 0x160, ["MMII_ref_info_base_dma_from_ddr_0"]>;
def bf_MMII_ref_info_base_dma_from_ddr_1 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_dma_from_ddr_1", 64, 0, "UINT">;
def reg_MMII_ref_info_base_dma_from_ddr_1 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_dma_from_ddr_1", 8, 0x168, ["MMII_ref_info_base_dma_from_ddr_1"]>;

// uint64_t ref_info_base_dma_from_cmx[VPU_MAX_DMA_ENGINES]
def bf_MMII_ref_info_base_dma_from_cmx_0 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_dma_from_cmx_0", 64, 0, "UINT">;
def reg_MMII_ref_info_base_dma_from_cmx_0 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_dma_from_cmx_0", 8, 0x170, ["MMII_ref_info_base_dma_from_cmx_0"]>;
def bf_MMII_ref_info_base_dma_from_cmx_1 : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_dma_from_cmx_1", 64, 0, "UINT">;
def reg_MMII_ref_info_base_dma_from_cmx_1 : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_dma_from_cmx_1", 8, 0x178, ["MMII_ref_info_base_dma_from_cmx_1"]>;

// uint64_t ref_info_base_media
def bf_MMII_ref_info_base_media : NPUReg50XX_RegFieldWrapper<"MMII_ref_info_base_media", 64, 0, "UINT">;
def reg_MMII_ref_info_base_media : NPUReg50XX_RegisterWrapper<"MMII_ref_info_base_media", 8, 0x180, ["MMII_ref_info_base_media"]>;

def reg_mapped_VpuManagedMappedInferenceInfoDescriptor : NPUReg50XX_RegMappedWrapper<"VpuManagedMappedInferenceInfo",
    [
        "taskReferenceR1_MMII_tasks_ref_info", "taskReferenceR2_MMII_tasks_ref_info", "taskReferenceR3_MMII_tasks_ref_info", "taskReferenceAddr_MMII_tasks_ref_info",
        "taskReferenceCount_MMII_tasks_ref_info", "taskReferenceR1_MMII_vb_mapping", "taskReferenceR2_MMII_vb_mapping", "taskReferenceR3_MMII_vb_mapping",
        "taskReferenceAddr_MMII_vb_mapping", "taskReferenceCount_MMII_vb_mapping", "taskReferenceR1_MMII_barrier_producer_ref_offsets",
        "taskReferenceR2_MMII_barrier_producer_ref_offsets", "taskReferenceR3_MMII_barrier_producer_ref_offsets", "taskReferenceAddr_MMII_barrier_producer_ref_offsets",
        "taskReferenceCount_MMII_barrier_producer_ref_offsets", "taskReferenceR1_MMII_barrier_consumer_ref_offsets", "taskReferenceR2_MMII_barrier_consumer_ref_offsets",
        "taskReferenceR3_MMII_barrier_consumer_ref_offsets", "taskReferenceAddr_MMII_barrier_consumer_ref_offsets",
        "taskReferenceCount_MMII_barrier_consumer_ref_offsets", "MMII_ref_info_base_vars_0", "MMII_ref_info_base_vars_1", "MMII_ref_info_base_vars_2",
        "MMII_ref_info_base_vars_3", "MMII_ref_info_base_vars_4", "MMII_ref_info_base_vars_5", "MMII_ref_info_base_invars_0", "MMII_ref_info_base_invars_1",
        "MMII_ref_info_base_invars_2", "MMII_ref_info_base_invars_3", "MMII_ref_info_base_invars_4", "MMII_ref_info_base_invars_5", "MMII_ref_info_base_akr_0",
        "MMII_ref_info_base_akr_1", "MMII_ref_info_base_akr_2", "MMII_ref_info_base_akr_3", "MMII_ref_info_base_akr_4", "MMII_ref_info_base_akr_5",
        "MMII_ref_info_base_aki_0", "MMII_ref_info_base_aki_1", "MMII_ref_info_base_aki_2", "MMII_ref_info_base_aki_3", "MMII_ref_info_base_aki_4",
        "MMII_ref_info_base_aki_5", "MMII_ref_info_base_dma_from_ddr_0", "MMII_ref_info_base_dma_from_ddr_1", "MMII_ref_info_base_dma_from_cmx_0",
        "MMII_ref_info_base_dma_from_cmx_1", "MMII_ref_info_base_media"
    ]>;


//
// VpuManagedMappedInference
//

// uint32_t vpu_nnrt_api_ver
def bf_MMI_vpu_nnrt_api_ver : NPUReg50XX_RegFieldWrapper<"MMI_vpu_nnrt_api_ver", 32, 0, "UINT">;
def reg_MMI_vpu_nnrt_api_ver : NPUReg50XX_RegisterWrapper<"MMI_vpu_nnrt_api_ver", 4, 0x0, ["MMI_vpu_nnrt_api_ver"]>;

// uint32_t final_barrier
def bf_MMI_final_barrier : NPUReg50XX_RegFieldWrapper<"MMI_final_barrier", 32, 0, "UINT">;
def reg_MMI_final_barrier : NPUReg50XX_RegisterWrapper<"MMI_final_barrier", 4, 0x4, ["MMI_final_barrier"]>;

// VpuTaskReference<VpuWorkItem> work_items
def bf_taskReferenceR1_MMI_work_item : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_work_item", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_work_item : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_work_item", 8, 0x8, ["taskReferenceR1_MMI_work_item"]>;
def bf_taskReferenceR2_MMI_work_item : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_work_item", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_work_item : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_work_item", 8, 0x10, ["taskReferenceR2_MMI_work_item"]>;
def bf_taskReferenceR3_MMI_work_item : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_work_item", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_work_item : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_work_item", 8, 0x18, ["taskReferenceR3_MMI_work_item"]>;
def bf_taskReferenceAddr_MMI_work_item : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_work_item", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_work_item : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_work_item", 8, 0x20, ["taskReferenceAddr_MMI_work_item"]>;
def bf_taskReferenceCount_MMI_work_item : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_work_item", 64, 0>;
def reg_taskReferenceCount_MMI_work_item : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_work_item", 8, 0x28, ["taskReferenceCount_MMI_work_item"]>;

// VpuTaskReference<VpuTaskBarrierMap> task_configs
def bf_taskReferenceR1_MMI_task_configs : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_task_configs", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_task_configs : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_task_configs", 8, 0x30, ["taskReferenceR1_MMI_task_configs"]>;
def bf_taskReferenceR2_MMI_task_configs : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_task_configs", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_task_configs : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_task_configs", 8, 0x38, ["taskReferenceR2_MMI_task_configs"]>;
def bf_taskReferenceR3_MMI_task_configs : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_task_configs", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_task_configs : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_task_configs", 8, 0x40, ["taskReferenceR3_MMI_task_configs"]>;
def bf_taskReferenceAddr_MMI_task_configs : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_task_configs", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_task_configs : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_task_configs", 8, 0x48, ["taskReferenceAddr_MMI_task_configs"]>;
def bf_taskReferenceCount_MMI_task_configs : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_task_configs", 64, 0>;
def reg_taskReferenceCount_MMI_task_configs : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_task_configs", 8, 0x50, ["taskReferenceCount_MMI_task_configs"]>;

// VpuTaskReference<uint32_t> reserved0[4]
def bf_taskReferenceR1_MMI_reserved0_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_reserved0_0", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_reserved0_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_reserved0_0", 8, 0x58, ["taskReferenceR1_MMI_reserved0_0"]>;
def bf_taskReferenceR2_MMI_reserved0_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_reserved0_0", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_reserved0_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_reserved0_0", 8, 0x60, ["taskReferenceR2_MMI_reserved0_0"]>;
def bf_taskReferenceR3_MMI_reserved0_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_reserved0_0", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_reserved0_0 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_reserved0_0", 8, 0x68, ["taskReferenceR3_MMI_reserved0_0"]>;
def bf_taskReferenceAddr_MMI_reserved0_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_reserved0_0", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_reserved0_0 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_reserved0_0", 8, 0x70, ["taskReferenceAddr_MMI_reserved0_0"]>;
def bf_taskReferenceCount_MMI_reserved0_0 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_reserved0_0", 64, 0>;
def reg_taskReferenceCount_MMI_reserved0_0 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_reserved0_0", 8, 0x78, ["taskReferenceCount_MMI_reserved0_0"]>;

def bf_taskReferenceR1_MMI_reserved0_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_reserved0_1", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_reserved0_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_reserved0_1", 8, 0x80, ["taskReferenceR1_MMI_reserved0_1"]>;
def bf_taskReferenceR2_MMI_reserved0_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_reserved0_1", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_reserved0_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_reserved0_1", 8, 0x88, ["taskReferenceR2_MMI_reserved0_1"]>;
def bf_taskReferenceR3_MMI_reserved0_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_reserved0_1", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_reserved0_1 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_reserved0_1", 8, 0x90, ["taskReferenceR3_MMI_reserved0_1"]>;
def bf_taskReferenceAddr_MMI_reserved0_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_reserved0_1", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_reserved0_1 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_reserved0_1", 8, 0x98, ["taskReferenceAddr_MMI_reserved0_1"]>;
def bf_taskReferenceCount_MMI_reserved0_1 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_reserved0_1", 64, 0>;
def reg_taskReferenceCount_MMI_reserved0_1 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_reserved0_1", 8, 0xA0, ["taskReferenceCount_MMI_reserved0_1"]>;

def bf_taskReferenceR1_MMI_reserved0_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_reserved0_2", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_reserved0_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_reserved0_2", 8, 0xA8, ["taskReferenceR1_MMI_reserved0_2"]>;
def bf_taskReferenceR2_MMI_reserved0_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_reserved0_2", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_reserved0_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_reserved0_2", 8, 0xB0, ["taskReferenceR2_MMI_reserved0_2"]>;
def bf_taskReferenceR3_MMI_reserved0_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_reserved0_2", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_reserved0_2 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_reserved0_2", 8, 0xB8, ["taskReferenceR3_MMI_reserved0_2"]>;
def bf_taskReferenceAddr_MMI_reserved0_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_reserved0_2", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_reserved0_2 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_reserved0_2", 8, 0xC0, ["taskReferenceAddr_MMI_reserved0_2"]>;
def bf_taskReferenceCount_MMI_reserved0_2 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_reserved0_2", 64, 0>;
def reg_taskReferenceCount_MMI_reserved0_2 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_reserved0_2", 8, 0xC8, ["taskReferenceCount_MMI_reserved0_2"]>;

def bf_taskReferenceR1_MMI_reserved0_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_reserved0_3", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_reserved0_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_reserved0_3", 8, 0xD0, ["taskReferenceR1_MMI_reserved0_3"]>;
def bf_taskReferenceR2_MMI_reserved0_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_reserved0_3", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_reserved0_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_reserved0_3", 8, 0xD8, ["taskReferenceR2_MMI_reserved0_3"]>;
def bf_taskReferenceR3_MMI_reserved0_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_reserved0_3", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_reserved0_3 : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_reserved0_3", 8, 0xE0, ["taskReferenceR3_MMI_reserved0_3"]>;
def bf_taskReferenceAddr_MMI_reserved0_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_reserved0_3", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_reserved0_3 : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_reserved0_3", 8, 0xE8, ["taskReferenceAddr_MMI_reserved0_3"]>;
def bf_taskReferenceCount_MMI_reserved0_3 : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_reserved0_3", 64, 0>;
def reg_taskReferenceCount_MMI_reserved0_3 : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_reserved0_3", 8, 0xF0, ["taskReferenceCount_MMI_reserved0_3"]>;

// VpuTaskReference<VpuBarrierConfiguration> barriers_configuration
def bf_taskReferenceR1_MMI_barriers_configuration : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_barriers_configuration", 64, 0, "UINT", NPUReg50XX_Version<11, 5, 0>>;
def reg_taskReferenceR1_MMI_barriers_configuration : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_barriers_configuration", 8, 0xF8, ["taskReferenceR1_MMI_barriers_configuration"]>;
def bf_taskReferenceR2_MMI_barriers_configuration : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_barriers_configuration", 64, 0, "UINT", NPUReg50XX_Version<11, 5, 0>>;
def reg_taskReferenceR2_MMI_barriers_configuration : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_barriers_configuration", 8, 0x100, ["taskReferenceR2_MMI_barriers_configuration"]>;
def bf_taskReferenceR3_MMI_barriers_configuration : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_barriers_configuration", 64, 0, "UINT", NPUReg50XX_Version<11, 5, 0>>;
def reg_taskReferenceR3_MMI_barriers_configuration : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_barriers_configuration", 8, 0x108, ["taskReferenceR3_MMI_barriers_configuration"]>;
def bf_taskReferenceAddr_MMI_barriers_configuration : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_barriers_configuration", 64, 0, "UINT", NPUReg50XX_Version<11, 5, 0>>;
def reg_taskReferenceAddr_MMI_barriers_configuration : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_barriers_configuration", 8, 0x110, ["taskReferenceAddr_MMI_barriers_configuration"]>;
def bf_taskReferenceCount_MMI_barriers_configuration : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_barriers_configuration", 64, 0, "UINT", NPUReg50XX_Version<11, 5, 0>>;
def reg_taskReferenceCount_MMI_barriers_configuration : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_barriers_configuration", 8, 0x118, ["taskReferenceCount_MMI_barriers_configuration"]>;

// VpuTaskReference<uint32_t> num_of_barrier_reprogrammings
def bf_taskReferenceR1_MMI_num_of_barrier_reprogrammings : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_num_of_barrier_reprogrammings", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_num_of_barrier_reprogrammings : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_num_of_barrier_reprogrammings", 8, 0x120, ["taskReferenceR1_MMI_num_of_barrier_reprogrammings"]>;
def bf_taskReferenceR2_MMI_num_of_barrier_reprogrammings : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_num_of_barrier_reprogrammings", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_num_of_barrier_reprogrammings : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_num_of_barrier_reprogrammings", 8, 0x128, ["taskReferenceR2_MMI_num_of_barrier_reprogrammings"]>;
def bf_taskReferenceR3_MMI_num_of_barrier_reprogrammings : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_num_of_barrier_reprogrammings", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_num_of_barrier_reprogrammings : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_num_of_barrier_reprogrammings", 8, 0x130, ["taskReferenceR3_MMI_num_of_barrier_reprogrammings"]>;
def bf_taskReferenceAddr_MMI_num_of_barrier_reprogrammings : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_num_of_barrier_reprogrammings", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_num_of_barrier_reprogrammings : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_num_of_barrier_reprogrammings", 8, 0x138, ["taskReferenceAddr_MMI_num_of_barrier_reprogrammings"]>;
def bf_taskReferenceCount_MMI_num_of_barrier_reprogrammings : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_num_of_barrier_reprogrammings", 64, 0>;
def reg_taskReferenceCount_MMI_num_of_barrier_reprogrammings : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_num_of_barrier_reprogrammings", 8, 0x140, ["taskReferenceCount_MMI_num_of_barrier_reprogrammings"]>;

// VpuTaskReference<uint32_t> initial_barriers
def bf_taskReferenceR1_MMI_initial_barriers : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_initial_barriers", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_initial_barriers : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_initial_barriers", 8, 0x148, ["taskReferenceR1_MMI_initial_barriers"]>;
def bf_taskReferenceR2_MMI_initial_barriers : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_initial_barriers", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_initial_barriers : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_initial_barriers", 8, 0x150, ["taskReferenceR2_MMI_initial_barriers"]>;
def bf_taskReferenceR3_MMI_initial_barriers : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_initial_barriers", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_initial_barriers : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_initial_barriers", 8, 0x158, ["taskReferenceR3_MMI_initial_barriers"]>;
def bf_taskReferenceAddr_MMI_initial_barriers : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_initial_barriers", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_initial_barriers : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_initial_barriers", 8, 0x160, ["taskReferenceAddr_MMI_initial_barriers"]>;
def bf_taskReferenceCount_MMI_initial_barriers : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_initial_barriers", 64, 0>;
def reg_taskReferenceCount_MMI_initial_barriers : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_initial_barriers", 8, 0x168, ["taskReferenceCount_MMI_initial_barriers"]>;

// VpuTaskReference<VpuNNRTConfig> nnrt_config
def bf_taskReferenceR1_MMI_nnrt_config : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_nnrt_config", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_nnrt_config : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_nnrt_config", 8, 0x170, ["taskReferenceR1_MMI_nnrt_config"]>;
def bf_taskReferenceR2_MMI_nnrt_config : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_nnrt_config", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_nnrt_config : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_nnrt_config", 8, 0x178, ["taskReferenceR2_MMI_nnrt_config"]>;
def bf_taskReferenceR3_MMI_nnrt_config : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_nnrt_config", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_nnrt_config : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_nnrt_config", 8, 0x180, ["taskReferenceR3_MMI_nnrt_config"]>;
def bf_taskReferenceAddr_MMI_nnrt_config : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_nnrt_config", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_nnrt_config : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_nnrt_config", 8, 0x188, ["taskReferenceAddr_MMI_nnrt_config"]>;
def bf_taskReferenceCount_MMI_nnrt_config : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_nnrt_config", 64, 0>;
def reg_taskReferenceCount_MMI_nnrt_config : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_nnrt_config", 8, 0x190, ["taskReferenceCount_MMI_nnrt_config"]>;

// uint8_t actshv_used
def bf_MMI_actshv_used : NPUReg50XX_RegFieldWrapper<"MMI_actshv_used", 8, 0, "UINT">;
def reg_MMI_actshv_used : NPUReg50XX_RegisterWrapper<"MMI_actshv_used", 1, 0x198, ["MMI_actshv_used"]>;

// uint8_t dpu_used
def bf_MMI_dpu_used : NPUReg50XX_RegFieldWrapper<"MMI_dpu_used", 8, 0, "UINT">;
def reg_MMI_dpu_used : NPUReg50XX_RegisterWrapper<"MMI_dpu_used", 1, 0x199, ["MMI_dpu_used"]>;

// uint8_t media_used
def bf_MMI_media_used : NPUReg50XX_RegFieldWrapper<"MMI_media_used", 8, 0, "UINT">;
def reg_MMI_media_used : NPUReg50XX_RegisterWrapper<"MMI_media_used", 1, 0x19A, ["MMI_media_used"]>;

// uint8_t dma_from_ddr_used
def bf_MMI_dma_from_ddr_used : NPUReg50XX_RegFieldWrapper<"MMI_dma_from_ddr_used", 8, 0, "UINT">;
def reg_MMI_dma_from_ddr_used : NPUReg50XX_RegisterWrapper<"MMI_dma_from_ddr_used", 1, 0x19B, ["MMI_dma_from_ddr_used"]>;

// uint8_t dma_from_cmx_used
def bf_MMI_dma_from_cmx_used : NPUReg50XX_RegFieldWrapper<"MMI_dma_from_cmx_used", 8, 0, "UINT">;
def reg_MMI_dma_from_cmx_used : NPUReg50XX_RegisterWrapper<"MMI_dma_from_cmx_used", 1, 0x19C, ["MMI_dma_from_cmx_used"]>;

// uint8_t pad0_[2];
def bf_MMI_pad0_ : NPUReg50XX_RegFieldWrapper<"MMI_pad0_", 16, 0, "UINT">;
def reg_MMI_pad0_ : NPUReg50XX_RegisterWrapper<"MMI_pad0_", 2, 0x19D, ["MMI_pad0_"]>;

// VpuBarrierProgrammingMode barrier_programming_mode;
def bf_MMI_barrier_programming_mode : NPUReg50XX_RegFieldWrapper<"MMI_barrier_programming_mode", 8, 0, "UINT">;
def reg_MMI_barrier_programming_mode : NPUReg50XX_RegisterWrapper<"MMI_barrier_programming_mode", 1, 0x19F, ["MMI_barrier_programming_mode"]>;

// VpuTaskReference<VpuManagedMappedInferenceInfo> inference_info
def bf_taskReferenceR1_MMI_inference_info : NPUReg50XX_RegFieldWrapper<"taskReferenceR1_MMI_inference_info", 64, 0, "UINT">;
def reg_taskReferenceR1_MMI_inference_info : NPUReg50XX_RegisterWrapper<"taskReferenceR1_MMI_inference_info", 8, 0x1A0, ["taskReferenceR1_MMI_inference_info"]>;
def bf_taskReferenceR2_MMI_inference_info : NPUReg50XX_RegFieldWrapper<"taskReferenceR2_MMI_inference_info", 64, 0, "UINT">;
def reg_taskReferenceR2_MMI_inference_info : NPUReg50XX_RegisterWrapper<"taskReferenceR2_MMI_inference_info", 8, 0x1A8, ["taskReferenceR2_MMI_inference_info"]>;
def bf_taskReferenceR3_MMI_inference_info : NPUReg50XX_RegFieldWrapper<"taskReferenceR3_MMI_inference_info", 64, 0, "UINT">;
def reg_taskReferenceR3_MMI_inference_info : NPUReg50XX_RegisterWrapper<"taskReferenceR3_MMI_inference_info", 8, 0x1B0, ["taskReferenceR3_MMI_inference_info"]>;
def bf_taskReferenceAddr_MMI_inference_info : NPUReg50XX_RegFieldWrapper<"taskReferenceAddr_MMI_inference_info", 64, 0, "UINT">;
def reg_taskReferenceAddr_MMI_inference_info : NPUReg50XX_RegisterWrapper<"taskReferenceAddr_MMI_inference_info", 8, 0x1B8, ["taskReferenceAddr_MMI_inference_info"]>;
def bf_taskReferenceCount_MMI_inference_info : NPUReg50XX_RegFieldWrapper<"taskReferenceCount_MMI_inference_info", 64, 0>;
def reg_taskReferenceCount_MMI_inference_info : NPUReg50XX_RegisterWrapper<"taskReferenceCount_MMI_inference_info", 8, 0x1C0, ["taskReferenceCount_MMI_inference_info"]>;

// uint32_t barrier_configuration_stride
def bf_MMI_barrier_configuration_stride : NPUReg50XX_RegFieldWrapper<"MMI_barrier_configuration_stride", 32, 0, "UINT">;
def reg_MMI_barrier_configuration_stride : NPUReg50XX_RegisterWrapper<"MMI_barrier_configuration_stride", 4, 0x1C8, ["MMI_barrier_configuration_stride"]>;

// uint8_t inference_feature_cfg
def bf_MMI_DisableDmaSwFifo : NPUReg50XX_RegFieldWrapper<"MMI_DisableDmaSwFifo", 1, 0, "UINT">;
def bf_MMI_ReservedInferenceConfig : NPUReg50XX_RegFieldWrapper<"MMI_ReservedInferenceConfig", 7, 1, "UINT">;
def reg_MMI_inference_feature_cfg : NPUReg50XX_RegisterWrapper<"MMI_inference_feature_cfg", 1, 0x1CC, ["MMI_DisableDmaSwFifo", "MMI_ReservedInferenceConfig"]>;

// uint8_t pad1_[3]
def bf_MMI_pad1_ : NPUReg50XX_RegFieldWrapper<"MMI_pad1_", 24, 0, "UINT">;
def reg_MMI_pad1_ : NPUReg50XX_RegisterWrapper<"MMI_pad1_", 3, 0x1CD, ["MMI_pad1_"]>;

// uint32_t model_identifier
def bf_MMI_model_identifier : NPUReg50XX_RegFieldWrapper<"MMI_model_identifier", 32, 0, "UINT">;
def reg_MMI_model_identifier : NPUReg50XX_RegisterWrapper<"MMI_model_identifier", 4, 0x1D0, ["MMI_model_identifier"]>;

// uint8_t pad2_[228]
def bf_MMI_pad2_0 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_0", 64, 0, "UINT">;
def reg_MMI_pad2_0 : NPUReg50XX_RegisterWrapper<"MMI_pad2_0", 8, 0x1D4, ["MMI_pad2_0"]>;

def bf_MMI_pad2_1 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_1", 64, 0, "UINT">;
def reg_MMI_pad2_1 : NPUReg50XX_RegisterWrapper<"MMI_pad2_1", 8, 0x1DC, ["MMI_pad2_1"]>;

def bf_MMI_pad2_2 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_2", 64, 0, "UINT">;
def reg_MMI_pad2_2 : NPUReg50XX_RegisterWrapper<"MMI_pad2_2", 8, 0x1E4, ["MMI_pad2_2"]>;

def bf_MMI_pad2_3 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_3", 64, 0, "UINT">;
def reg_MMI_pad2_3 : NPUReg50XX_RegisterWrapper<"MMI_pad2_3", 8, 0x1EC, ["MMI_pad2_3"]>;

def bf_MMI_pad2_4 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_4", 64, 0, "UINT">;
def reg_MMI_pad2_4 : NPUReg50XX_RegisterWrapper<"MMI_pad2_4", 8, 0x1F4, ["MMI_pad2_4"]>;

def bf_MMI_pad2_5 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_5", 64, 0, "UINT">;
def reg_MMI_pad2_5 : NPUReg50XX_RegisterWrapper<"MMI_pad2_5", 8, 0x1FC, ["MMI_pad2_5"]>;

def bf_MMI_pad2_6 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_6", 64, 0, "UINT">;
def reg_MMI_pad2_6 : NPUReg50XX_RegisterWrapper<"MMI_pad2_6", 8, 0x204, ["MMI_pad2_6"]>;

def bf_MMI_pad2_7 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_7", 64, 0, "UINT">;
def reg_MMI_pad2_7 : NPUReg50XX_RegisterWrapper<"MMI_pad2_7", 8, 0x20C, ["MMI_pad2_7"]>;

def bf_MMI_pad2_8 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_8", 64, 0, "UINT">;
def reg_MMI_pad2_8 : NPUReg50XX_RegisterWrapper<"MMI_pad2_8", 8, 0x214, ["MMI_pad2_8"]>;

def bf_MMI_pad2_9 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_9", 64, 0, "UINT">;
def reg_MMI_pad2_9 : NPUReg50XX_RegisterWrapper<"MMI_pad2_9", 8, 0x21C, ["MMI_pad2_9"]>;

def bf_MMI_pad2_10 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_10", 64, 0, "UINT">;
def reg_MMI_pad2_10 : NPUReg50XX_RegisterWrapper<"MMI_pad2_10", 8, 0x224, ["MMI_pad2_10"]>;

def bf_MMI_pad2_11 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_11", 64, 0, "UINT">;
def reg_MMI_pad2_11 : NPUReg50XX_RegisterWrapper<"MMI_pad2_11", 8, 0x22C, ["MMI_pad2_11"]>;

def bf_MMI_pad2_12 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_12", 64, 0, "UINT">;
def reg_MMI_pad2_12 : NPUReg50XX_RegisterWrapper<"MMI_pad2_12", 8, 0x234, ["MMI_pad2_12"]>;

def bf_MMI_pad2_13 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_13", 64, 0, "UINT">;
def reg_MMI_pad2_13 : NPUReg50XX_RegisterWrapper<"MMI_pad2_13", 8, 0x23C, ["MMI_pad2_13"]>;

def bf_MMI_pad2_14 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_14", 64, 0, "UINT">;
def reg_MMI_pad2_14 : NPUReg50XX_RegisterWrapper<"MMI_pad2_14", 8, 0x244, ["MMI_pad2_14"]>;

def bf_MMI_pad2_15 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_15", 64, 0, "UINT">;
def reg_MMI_pad2_15 : NPUReg50XX_RegisterWrapper<"MMI_pad2_15", 8, 0x24C, ["MMI_pad2_15"]>;

def bf_MMI_pad2_16 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_16", 64, 0, "UINT">;
def reg_MMI_pad2_16 : NPUReg50XX_RegisterWrapper<"MMI_pad2_16", 8, 0x254, ["MMI_pad2_16"]>;

def bf_MMI_pad2_17 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_17", 64, 0, "UINT">;
def reg_MMI_pad2_17 : NPUReg50XX_RegisterWrapper<"MMI_pad2_17", 8, 0x25C, ["MMI_pad2_17"]>;

def bf_MMI_pad2_18 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_18", 64, 0, "UINT">;
def reg_MMI_pad2_18 : NPUReg50XX_RegisterWrapper<"MMI_pad2_18", 8, 0x264, ["MMI_pad2_18"]>;

def bf_MMI_pad2_19 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_19", 64, 0, "UINT">;
def reg_MMI_pad2_19 : NPUReg50XX_RegisterWrapper<"MMI_pad2_19", 8, 0x26C, ["MMI_pad2_19"]>;

def bf_MMI_pad2_20 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_20", 64, 0, "UINT">;
def reg_MMI_pad2_20 : NPUReg50XX_RegisterWrapper<"MMI_pad2_20", 8, 0x274, ["MMI_pad2_20"]>;

def bf_MMI_pad2_21 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_21", 64, 0, "UINT">;
def reg_MMI_pad2_21 : NPUReg50XX_RegisterWrapper<"MMI_pad2_21", 8, 0x27C, ["MMI_pad2_21"]>;

def bf_MMI_pad2_22 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_22", 64, 0, "UINT">;
def reg_MMI_pad2_22 : NPUReg50XX_RegisterWrapper<"MMI_pad2_22", 8, 0x284, ["MMI_pad2_22"]>;

def bf_MMI_pad2_23 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_23", 64, 0, "UINT">;
def reg_MMI_pad2_23 : NPUReg50XX_RegisterWrapper<"MMI_pad2_23", 8, 0x28C, ["MMI_pad2_23"]>;

def bf_MMI_pad2_24 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_24", 64, 0, "UINT">;
def reg_MMI_pad2_24 : NPUReg50XX_RegisterWrapper<"MMI_pad2_24", 8, 0x294, ["MMI_pad2_24"]>;

def bf_MMI_pad2_25 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_25", 64, 0, "UINT">;
def reg_MMI_pad2_25 : NPUReg50XX_RegisterWrapper<"MMI_pad2_25", 8, 0x29C, ["MMI_pad2_25"]>;

def bf_MMI_pad2_26 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_26", 64, 0, "UINT">;
def reg_MMI_pad2_26 : NPUReg50XX_RegisterWrapper<"MMI_pad2_26", 8, 0x2A4, ["MMI_pad2_26"]>;

def bf_MMI_pad2_27 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_27", 64, 0, "UINT">;
def reg_MMI_pad2_27 : NPUReg50XX_RegisterWrapper<"MMI_pad2_27", 8, 0x2AC, ["MMI_pad2_27"]>;

def bf_MMI_pad2_28 : NPUReg50XX_RegFieldWrapper<"MMI_pad2_28", 32, 0, "UINT">;
def reg_MMI_pad2_28 : NPUReg50XX_RegisterWrapper<"MMI_pad2_28", 4, 0x2B4, ["MMI_pad2_28"]>;

// uint32_t bootstrap_workitems_count
def bf_MMI_bootstrap_workitems_count : NPUReg50XX_RegFieldWrapper<"MMI_bootstrap_workitems_count", 32, 0, "UINT">;
def reg_MMI_bootstrap_workitems_count : NPUReg50XX_RegisterWrapper<"MMI_bootstrap_workitems_count", 4, 0x2B8, ["MMI_bootstrap_workitems_count"]>;

// uint8_t pad2_[4]
def bf_MMI_pad3_ : NPUReg50XX_RegFieldWrapper<"MMI_pad3_", 32, 0, "UINT">;
def reg_MMI_pad3_ : NPUReg50XX_RegisterWrapper<"MMI_pad3_", 4, 0x2BC, ["MMI_pad3_"]>;

def reg_mapped_VpuManagedMappedInferenceDescriptor : NPUReg50XX_RegMappedWrapper<"VpuManagedMappedInference",
    [
        "MMI_vpu_nnrt_api_ver", "MMI_final_barrier",
        "taskReferenceR1_MMI_work_item", "taskReferenceR2_MMI_work_item", "taskReferenceR3_MMI_work_item", "taskReferenceAddr_MMI_work_item", "taskReferenceCount_MMI_work_item",
        "taskReferenceR1_MMI_task_configs", "taskReferenceR2_MMI_task_configs", "taskReferenceR3_MMI_task_configs", "taskReferenceAddr_MMI_task_configs", "taskReferenceCount_MMI_task_configs",
        "taskReferenceR1_MMI_reserved0_0", "taskReferenceR2_MMI_reserved0_0", "taskReferenceR3_MMI_reserved0_0", "taskReferenceAddr_MMI_reserved0_0", "taskReferenceCount_MMI_reserved0_0",
        "taskReferenceR1_MMI_reserved0_1", "taskReferenceR2_MMI_reserved0_1", "taskReferenceR3_MMI_reserved0_1", "taskReferenceAddr_MMI_reserved0_1", "taskReferenceCount_MMI_reserved0_1",
        "taskReferenceR1_MMI_reserved0_2", "taskReferenceR2_MMI_reserved0_2", "taskReferenceR3_MMI_reserved0_2", "taskReferenceAddr_MMI_reserved0_2", "taskReferenceCount_MMI_reserved0_2",
        "taskReferenceR1_MMI_reserved0_3", "taskReferenceR2_MMI_reserved0_3", "taskReferenceR3_MMI_reserved0_3", "taskReferenceAddr_MMI_reserved0_3", "taskReferenceCount_MMI_reserved0_3",
        "taskReferenceR1_MMI_barriers_configuration", "taskReferenceR2_MMI_barriers_configuration", "taskReferenceR3_MMI_barriers_configuration", "taskReferenceAddr_MMI_barriers_configuration", "taskReferenceCount_MMI_barriers_configuration",
        "taskReferenceR1_MMI_num_of_barrier_reprogrammings", "taskReferenceR2_MMI_num_of_barrier_reprogrammings", "taskReferenceR3_MMI_num_of_barrier_reprogrammings", "taskReferenceAddr_MMI_num_of_barrier_reprogrammings", "taskReferenceCount_MMI_num_of_barrier_reprogrammings",
        "taskReferenceR1_MMI_initial_barriers", "taskReferenceR2_MMI_initial_barriers", "taskReferenceR3_MMI_initial_barriers", "taskReferenceAddr_MMI_initial_barriers", "taskReferenceCount_MMI_initial_barriers",
        "taskReferenceR1_MMI_nnrt_config", "taskReferenceR2_MMI_nnrt_config", "taskReferenceR3_MMI_nnrt_config", "taskReferenceAddr_MMI_nnrt_config", "taskReferenceCount_MMI_nnrt_config",
        "MMI_actshv_used", "MMI_dpu_used", "MMI_media_used", "MMI_dma_from_ddr_used", "MMI_dma_from_cmx_used", "MMI_pad0_", "MMI_barrier_programming_mode",
        "taskReferenceR1_MMI_inference_info", "taskReferenceR2_MMI_inference_info", "taskReferenceR3_MMI_inference_info", "taskReferenceAddr_MMI_inference_info", "taskReferenceCount_MMI_inference_info",
        "MMI_barrier_configuration_stride", "MMI_inference_feature_cfg", "MMI_pad1_", "MMI_model_identifier", "MMI_pad2_0", "MMI_pad2_1", "MMI_pad2_2", "MMI_pad2_3", "MMI_pad2_4", "MMI_pad2_5", "MMI_pad2_6", "MMI_pad2_7",
        "MMI_pad2_8", "MMI_pad2_9", "MMI_pad2_10", "MMI_pad2_11", "MMI_pad2_12", "MMI_pad2_13", "MMI_pad2_14", "MMI_pad2_15", "MMI_pad2_16", "MMI_pad2_17",
        "MMI_pad2_18", "MMI_pad2_19", "MMI_pad2_20", "MMI_pad2_21", "MMI_pad2_22", "MMI_pad2_23", "MMI_pad2_24", "MMI_pad2_25", "MMI_pad2_26", "MMI_pad2_27",
        "MMI_pad2_28", "MMI_bootstrap_workitems_count", "MMI_pad3_"
    ]>;

#endif
