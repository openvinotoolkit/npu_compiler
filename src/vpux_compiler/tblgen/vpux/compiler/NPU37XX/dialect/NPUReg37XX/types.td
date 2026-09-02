//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUReg37XX_TYPES
#define VPUX_COMPILER_DIALECT_NPUReg37XX_TYPES

include "vpux/compiler/dialect/VPURegMapped/dialect.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/dialect.td"

//
// RegField Specialization
//

class NPUReg37XX_Version<int vMajor, int vMinor, int vPatch> : VPURegMapped_Version {
    let major = vMajor;
    let minor = vMinor;
    let patch = vPatch;
}

// MI Version that acts as baseline for NPU37XX NNRT API
// All registers default to this baseline unless specified otherwise
def NPUReg37XX_BaselineVersion : NPUReg37XX_Version<7, 0, 0>;

class NPUReg37XX_RegFieldWrapper<string name = "", int width = 0, int pos = 0, string dataType = "UINT", VPURegMapped_Version version = NPUReg37XX_BaselineVersion> :
    VPURegMapped_RegFieldWrapper<NPUReg37XX_Dialect, name, width, pos, dataType, version> {

    }

//
// Register Specialization
//

class NPUReg37XX_RegisterWrapper<string name = "", int size = 0, int address = 0x00, list<string> regFieldTypes = [], int resetValue = 0x0, bit allowOverlap = 0> :
    VPURegMapped_RegisterWrapper<NPUReg37XX_Dialect, "vpux::NPUReg37XX::", name, size, address, regFieldTypes, resetValue, allowOverlap> {

}

//
// RegMapped Specialization
//

class NPUReg37XX_RegMappedWrapper<string name = "", list<string> regsTypes = [], bit canBeEmpty = 0> :
    VPURegMapped_RegMappedWrapper<NPUReg37XX_Dialect, "vpux::NPUReg37XX::", name, regsTypes, canBeEmpty> {

}

//
// DPUInvariantRegisters
//

// se_sp_addr[4]
def bf_DpuInvSeAddr : NPUReg37XX_RegFieldWrapper<"se_addr", 32, 0>;
def bf_DpuInvSparsityAddr : NPUReg37XX_RegFieldWrapper<"sparsity_addr", 32, 32>;

def reg_SeSpAddr0 : NPUReg37XX_RegisterWrapper<"se_sp_addr_0", 8, 0x0, ["se_addr", "sparsity_addr"]>;
def reg_SeSpAddr1 : NPUReg37XX_RegisterWrapper<"se_sp_addr_1", 8, 0x8, ["se_addr", "sparsity_addr"]>;
def reg_SeSpAddr2 : NPUReg37XX_RegisterWrapper<"se_sp_addr_2", 8, 0x10, ["se_addr", "sparsity_addr"]>;
def reg_SeSpAddr3 : NPUReg37XX_RegisterWrapper<"se_sp_addr_3", 8, 0x18, ["se_addr", "sparsity_addr"]>;

// se_sp_size[3]
def bf_DpuInvSpSegSize : NPUReg37XX_RegFieldWrapper<"sp_seg_size", 14, 0>;
def bf_DpuInvSeSegSize : NPUReg37XX_RegFieldWrapper<"se_seg_size", 18, 14>;

def reg_SeSpSize0 : NPUReg37XX_RegisterWrapper<"se_sp_size_0", 8, 0x20, ["sp_seg_size", "se_seg_size"]>;
def reg_SeSpSize1 : NPUReg37XX_RegisterWrapper<"se_sp_size_1", 8, 0x24, ["sp_seg_size", "se_seg_size"]>;
def reg_SeSpSize2 : NPUReg37XX_RegisterWrapper<"se_sp_size_2", 8, 0x28, ["sp_seg_size", "se_seg_size"]>;

// z_config
def bf_SeZSplit : NPUReg37XX_RegFieldWrapper<"se_z_split", 4, 0>;
def bf_NumSesInZDir : NPUReg37XX_RegFieldWrapper<"num_ses_in_z_dir", 9, 4>;
def bf_CmSpPattern : NPUReg37XX_RegFieldWrapper<"cm_sp_pattern", 16, 13>;
def bf_AddrFormatSel : NPUReg37XX_RegFieldWrapper<"addr_format_sel", 1, 31>;

def reg_ZConfig : NPUReg37XX_RegisterWrapper<"z_config", 4, 0x2C,
    ["se_z_split", "num_ses_in_z_dir", "cm_sp_pattern", "addr_format_sel"]>;

// kernel_pad_cfg
def bf_KernelY : NPUReg37XX_RegFieldWrapper<"kernel_y", 4, 5>;
def bf_KernelX : NPUReg37XX_RegFieldWrapper<"kernel_x", 4, 9>;
def bf_WtPltCfg : NPUReg37XX_RegFieldWrapper<"wt_plt_cfg", 2, 13>;
def bf_ActDense : NPUReg37XX_RegFieldWrapper<"act_dense", 1, 15>;
def bf_WtDense : NPUReg37XX_RegFieldWrapper<"wt_dense", 1, 16>;
def bf_StrideYEn : NPUReg37XX_RegFieldWrapper<"stride_y_en", 1, 17>;
def bf_StrideY : NPUReg37XX_RegFieldWrapper<"stride_y", 3, 18>;
def bf_DynamicBwEn : NPUReg37XX_RegFieldWrapper<"dynamic_bw_en", 1, 21>;
def bf_DwWtSpIns : NPUReg37XX_RegFieldWrapper<"dw_wt_sp_ins", 1, 22>;
def bf_Layer1WtSpIns : NPUReg37XX_RegFieldWrapper<"layer1_wt_sp_ins", 1, 23>;
def bf_Layer1CmpEn : NPUReg37XX_RegFieldWrapper<"layer1_cmp_en", 1, 24>;
def bf_PoolOptEn : NPUReg37XX_RegFieldWrapper<"pool_opt_en", 1, 25>;
def bf_SpSeTblSegment : NPUReg37XX_RegFieldWrapper<"sp_se_tbl_segment", 1, 29>;
def bf_RstCtxt : NPUReg37XX_RegFieldWrapper<"rst_ctxt", 1, 30>;

// kernel_pad_cfg
def reg_KernelPadCfg : NPUReg37XX_RegisterWrapper<"kernel_pad_cfg", 4, 0x30,
    ["kernel_y", "kernel_x", "wt_plt_cfg", "act_dense", "wt_dense", "stride_y_en", "stride_y", "dynamic_bw_en",
    "dw_wt_sp_ins", "layer1_wt_sp_ins", "layer1_cmp_en", "pool_opt_en", "sp_se_tbl_segment", "rst_ctxt"]>;

// weight_size_placeholder
def bf_WeightSizePlaceholder : NPUReg37XX_RegFieldWrapper<"weight_size_placeholder", 32, 0>;

def reg_WeightSizePlaceholder : NPUReg37XX_RegisterWrapper<"weight_size_placeholder", 4, 0x34, ["weight_size_placeholder"]>;

// weight_num_placeholder
def bf_WeightNumPlaceholder : NPUReg37XX_RegFieldWrapper<"weight_num_placeholder", 32, 0>;

def reg_WeightNumPlaceholder : NPUReg37XX_RegisterWrapper<"weight_num_placeholder", 4, 0x38, ["weight_num_placeholder"]>;

// weight_start
def bf_WeightStart : NPUReg37XX_RegFieldWrapper<"weight_start", 32, 0>;

def reg_WeightStart : NPUReg37XX_RegisterWrapper<"weight_start", 4, 0x3C, ["weight_start"]>;

// tensor_size0
def bf_TensorSizeX : NPUReg37XX_RegFieldWrapper<"tensor_size_x", 14, 0>;

def bf_TensorSizeY : NPUReg37XX_RegFieldWrapper<"tensor_size_y", 14, 14>;

def reg_TensorSize0 : NPUReg37XX_RegisterWrapper<"tensor_size0", 4, 0x40, ["tensor_size_x", "tensor_size_y"]>;

// tensor_size1
def bf_TensorSizeZ : NPUReg37XX_RegFieldWrapper<"tensor_size_z", 14, 0>;

def reg_TensorSize1 : NPUReg37XX_RegisterWrapper<"tensor_size1", 4, 0x44, ["tensor_size_z"]>;

// tensor_start
def bf_TensorStart : NPUReg37XX_RegFieldWrapper<"tensor_start", 32, 0>;

def reg_TensorStart : NPUReg37XX_RegisterWrapper<"tensor_start", 4, 0x48, ["tensor_start"]>;

// tensor_mode
def bf_Wmode : NPUReg37XX_RegFieldWrapper<"wmode", 4, 0>;
def bf_Amode : NPUReg37XX_RegFieldWrapper<"amode", 4, 4>;
def bf_Stride : NPUReg37XX_RegFieldWrapper<"stride", 3, 8>;
def bf_ZmInput : NPUReg37XX_RegFieldWrapper<"zm_input", 1, 11>;
def bf_DwInput : NPUReg37XX_RegFieldWrapper<"dw_input", 1, 12>;
def bf_WorkloadOperation : NPUReg37XX_RegFieldWrapper<"workload_operation", 2, 14>;

def reg_TensorMode : NPUReg37XX_RegisterWrapper<"tensor_mode", 4, 0x4c,
    ["wmode", "amode", "stride", "zm_input", "dw_input", "workload_operation"]>;


// elops_sparsity_addr
def bf_ElopsSparityAddr : NPUReg37XX_RegFieldWrapper<"elops_sparsity_addr", 32, 0>;

def reg_ElopsSparityAddr : NPUReg37XX_RegisterWrapper<"elops_sparsity_addr", 4, 0x50, ["elops_sparsity_addr"]>;

// elops_se_addr
def bf_ElopsSeAddr : NPUReg37XX_RegFieldWrapper<"elops_se_addr", 32, 0>;

def reg_ElopsSeAddr : NPUReg37XX_RegisterWrapper<"elops_se_addr", 4, 0x54, ["elops_se_addr"]>;

// elops_wload
def bf_ElopWload : NPUReg37XX_RegFieldWrapper<"elop_wload", 1, 0>;
def bf_ElopWloadType : NPUReg37XX_RegFieldWrapper<"elop_wload_type", 1, 3>;
def bf_PoolWtData : NPUReg37XX_RegFieldWrapper<"pool_wt_data", 16, 4>;
def bf_PoolWtRdDis : NPUReg37XX_RegFieldWrapper<"pool_wt_rd_dis", 1, 26>;

def reg_ElopsWload : NPUReg37XX_RegisterWrapper<"elops_wload", 4, 0x58,
    ["elop_wload", "elop_wload_type", "pool_wt_data", "pool_wt_rd_dis"], 0x0>;


// act0_offset [4]
def bf_Adr0Offset : NPUReg37XX_RegFieldWrapper<"act_offset", 32, 0>;

def reg_Act0Offset : NPUReg37XX_RegisterWrapper<"act_offset0", 4, 0x5C, ["act_offset"]>;
def reg_Act1Offset : NPUReg37XX_RegisterWrapper<"act_offset1", 4, 0x60, ["act_offset"]>;
def reg_Act2Offset : NPUReg37XX_RegisterWrapper<"act_offset2", 4, 0x64, ["act_offset"]>;
def reg_Act3Offset : NPUReg37XX_RegisterWrapper<"act_offset3", 4, 0x68, ["act_offset"]>;

// base_offset_a
def bf_BaseOffsetA : NPUReg37XX_RegFieldWrapper<"base_offset_a", 32, 0>;
def reg_BaseOffsetA : NPUReg37XX_RegisterWrapper<"base_offset_a", 4, 0x6C, ["base_offset_a"]>;

// base_offset_b
def bf_BaseOffsetB : NPUReg37XX_RegFieldWrapper<"base_offset_b", 32, 0>;
def reg_BaseOffsetB : NPUReg37XX_RegisterWrapper<"base_offset_b", 4, 0x70, ["base_offset_b"]>;

// wt_offset
def bf_WtOffset : NPUReg37XX_RegFieldWrapper<"wt_offset", 32, 0>;
def reg_WtOffset : NPUReg37XX_RegisterWrapper<"wt_offset", 4, 0x74, ["wt_offset"]>;

// odu_cfg
def bf_Dtype : NPUReg37XX_RegFieldWrapper<"dtype", 4, 0>;
def bf_SpValue : NPUReg37XX_RegFieldWrapper<"sp_value", 8, 5>;
def bf_SpOutEn : NPUReg37XX_RegFieldWrapper<"sp_out_en", 1, 13>;
def bf_WriteSp : NPUReg37XX_RegFieldWrapper<"write_sp", 1, 15>;
def bf_WritePt : NPUReg37XX_RegFieldWrapper<"write_pt", 1, 16>;
def bf_WriteAc : NPUReg37XX_RegFieldWrapper<"write_ac", 1, 17>;
def bf_Mode : NPUReg37XX_RegFieldWrapper<"mode", 2, 18>;
def bf_Grid : NPUReg37XX_RegFieldWrapper<"grid", 1, 20>;
def bf_SwizzleKey : NPUReg37XX_RegFieldWrapper<"swizzle_key", 3, 21>;
def bf_Nthw : NPUReg37XX_RegFieldWrapper<"nthw", 2, 25>;
def bf_Permutation : NPUReg37XX_RegFieldWrapper<"permutation", 3, 27>;
def bf_WcbStallAvoidance : NPUReg37XX_RegFieldWrapper<"debug_mode", 1, 30>;

def reg_OduCfg : NPUReg37XX_RegisterWrapper<"odu_cfg", 4, 0x78,
    ["dtype", "sp_value", "sp_out_en", "write_sp", "write_pt", "write_ac", "mode", "grid",
    "swizzle_key", "nthw", "permutation", "debug_mode"]>;

// odu_be_size
def bf_OduBeSize : NPUReg37XX_RegFieldWrapper<"odu_be_size", 32, 0>;

def reg_OduBeSize : NPUReg37XX_RegisterWrapper<"odu_be_size", 4, 0x7C, ["odu_be_size"]>;

// odu_be_cnt
def bf_OduBeCnt : NPUReg37XX_RegFieldWrapper<"odu_be_cnt", 32, 0>;

def reg_OduBeCnt : NPUReg37XX_RegisterWrapper<"odu_be_cnt", 4, 0x80, ["odu_be_cnt"]>;

// odu_se_size
def bf_SeSize : NPUReg37XX_RegFieldWrapper<"se_size", 32, 0>;
def reg_SeSize : NPUReg37XX_RegisterWrapper<"se_size", 4, 0x84, ["se_size"]>;

// te_dim0
def bf_TeDimY : NPUReg37XX_RegFieldWrapper<"te_dim_y", 13, 0>;
def bf_TeDimZ : NPUReg37XX_RegFieldWrapper<"te_dim_z", 13, 13>;

def reg_TeDim0 : NPUReg37XX_RegisterWrapper<"te_dim0", 4, 0x88, ["te_dim_y", "te_dim_z"]>;

// te_dim1
def bf_TeDimX : NPUReg37XX_RegFieldWrapper<"te_dim_x", 13, 0>;

def reg_TeDim1 : NPUReg37XX_RegisterWrapper<"te_dim1", 4, 0x8C, ["te_dim_x"]>;

// pt_base
def bf_PtBase : NPUReg37XX_RegFieldWrapper<"pt_base", 32, 0>;

def reg_PtBase : NPUReg37XX_RegisterWrapper<"pt_base", 4, 0x90, ["pt_base"]>;

// sp_base
def bf_SpBase : NPUReg37XX_RegFieldWrapper<"sp_base", 32, 0>;

def reg_SpBase : NPUReg37XX_RegisterWrapper<"sp_base", 4, 0x94, ["sp_base"]>;

// base_ptr_a
def bf_BasePtrA : NPUReg37XX_RegFieldWrapper<"base_ptr_a", 32, 0>;

def reg_BasePtrA : NPUReg37XX_RegisterWrapper<"base_ptr_a", 4, 0x98, ["base_ptr_a"]>;

// base_ptr_b
def bf_BasePtrB : NPUReg37XX_RegFieldWrapper<"base_ptr_b", 32, 0>;

def reg_BasePtrB : NPUReg37XX_RegisterWrapper<"base_ptr_b", 4, 0x9C, ["base_ptr_b"]>;

// base_adr
def bf_BaseAdr : NPUReg37XX_RegFieldWrapper<"base_adr", 32, 0>;

def reg_BaseAdr : NPUReg37XX_RegisterWrapper<"base_adr0", 4, 0xa0, ["base_adr"]>;
def reg_BaseAdr1 : NPUReg37XX_RegisterWrapper<"base_adr1", 4, 0xa4, ["base_adr"]>;
def reg_BaseAdr2 : NPUReg37XX_RegisterWrapper<"base_adr2", 4, 0xa8, ["base_adr"]>;
def reg_BaseAdr3 : NPUReg37XX_RegisterWrapper<"base_adr3", 4, 0xaC, ["base_adr"]>;

// odu_cast
def bf_OduCastEn : NPUReg37XX_RegFieldWrapper<"cast_enable", 1, 0>;
def bf_OduCastOffset : NPUReg37XX_RegFieldWrapper<"cast_offset", 28, 4>;

def reg_OduCast0 : NPUReg37XX_RegisterWrapper<"odu_cast0", 4, 0xb0, ["cast_enable", "cast_offset"]>;
def reg_OduCast1 : NPUReg37XX_RegisterWrapper<"odu_cast1", 4, 0xb4, ["cast_enable", "cast_offset"]>;
def reg_OduCast2 : NPUReg37XX_RegisterWrapper<"odu_cast2", 4, 0xb8, ["cast_enable", "cast_offset"]>;
def reg_OduCast3 : NPUReg37XX_RegisterWrapper<"odu_cast3", 4, 0xbC, ["cast_enable", "cast_offset"]>;

// mpe_cfg
def bf_MpeWtbias : NPUReg37XX_RegFieldWrapper<"mpe_wtbias", 8, 0, "UINT">;
def bf_MpeActbias : NPUReg37XX_RegFieldWrapper<"mpe_actbias", 8, 8, "UINT">;
def bf_MpeDaz : NPUReg37XX_RegFieldWrapper<"mpe_daz", 1, 22>;

def reg_MpeCfg : NPUReg37XX_RegisterWrapper<"mpe_cfg", 4, 0xBC,
    ["mpe_wtbias", "mpe_actbias", "mpe_daz"]>;

// elop_scale
def bf_ElopScaleB : NPUReg37XX_RegFieldWrapper<"elop_scale_b", 16, 0>;
def bf_ElopScaleA : NPUReg37XX_RegFieldWrapper<"elop_scale_a", 16, 16>;

def reg_ElopScale : NPUReg37XX_RegisterWrapper<"elop_scale", 4, 0xC4, ["elop_scale_b", "elop_scale_a"]>;

// ppe_cfg
def bf_PpeG8BiasC : NPUReg37XX_RegFieldWrapper<"ppe_g8_bias_c", 9, 0, "SINT">;

def reg_PpeCfg : NPUReg37XX_RegisterWrapper<"ppe_cfg", 4, 0xC8, ["ppe_g8_bias_c"]>;

// ppe_bias
def bf_PpeBias : NPUReg37XX_RegFieldWrapper<"ppe_bias", 32, 0, "SINT">;

def reg_PpeBias : NPUReg37XX_RegisterWrapper<"ppe_bias", 4, 0xCC, ["ppe_bias"]>;

// ppe_scale
def bf_PpeScaleShift : NPUReg37XX_RegFieldWrapper<"ppe_scale_shift", 6, 2>;
def bf_PpeScaleRound : NPUReg37XX_RegFieldWrapper<"ppe_scale_round", 2, 10>;
def bf_PpeScaleMult : NPUReg37XX_RegFieldWrapper<"ppe_scale_mult", 16, 16, "SINT">;

def reg_PpeScale : NPUReg37XX_RegisterWrapper<"ppe_scale", 4, 0xD0, ["ppe_scale_shift", "ppe_scale_round", "ppe_scale_mult"], 0x00000c00>;

// ppe_scale_ctrl
def bf_PpeScaleOverride : NPUReg37XX_RegFieldWrapper<"ppe_scale_override", 1, 0>;
def bf_PpeFpScaleOverride : NPUReg37XX_RegFieldWrapper<"ppe_fp_scale_override", 1, 1>;

def reg_PpeScaleCtrl : NPUReg37XX_RegisterWrapper<"ppe_scale_ctrl", 4, 0xD4, ["ppe_scale_override", "ppe_fp_scale_override"], 0x0>;


// ppe_prelu
def bf_PpePreluShift : NPUReg37XX_RegFieldWrapper<"ppe_prelu_shift", 5, 8>;
def bf_PpePreluMult : NPUReg37XX_RegFieldWrapper<"ppe_prelu_mult", 11, 16>;

def reg_PpePrelu : NPUReg37XX_RegisterWrapper<"ppe_prelu", 4, 0xD8, ["ppe_prelu_shift", "ppe_prelu_mult"], 0x00010000>;

// ppe_scale_hclamp
def bf_PpeScaleHclamp : NPUReg37XX_RegFieldWrapper<"ppe_scale_hclamp", 32, 0, "SINT">;
def reg_PpeScaleHclamp : NPUReg37XX_RegisterWrapper<"ppe_scale_hclamp", 4, 0xE4, ["ppe_scale_hclamp"], 0x7fffffff>;

// ppe_scale_lclamp
def bf_PpeScaleLclamp : NPUReg37XX_RegFieldWrapper<"ppe_scale_lclamp", 32, 0, "SINT">;
def reg_PpeScaleLclamp : NPUReg37XX_RegisterWrapper<"ppe_scale_lclamp", 4, 0xE8, ["ppe_scale_lclamp"], 0x80000000>;

// ppe_misc
def bf_PpeFp16Ftz : NPUReg37XX_RegFieldWrapper<"ppe_fp16_ftz", 1, 6>;
def bf_PpeFp16Clamp : NPUReg37XX_RegFieldWrapper<"ppe_fp16_clamp", 1, 7>;
def bf_PpeI32Convert : NPUReg37XX_RegFieldWrapper<"ppe_i32_convert", 2, 8>;
def reg_PpeMisc : NPUReg37XX_RegisterWrapper<"ppe_misc", 4, 0xF0, ["ppe_fp16_ftz", "ppe_fp16_clamp", "ppe_i32_convert"]>;

// ppe_fp_bias
def bf_PpeFpBias : NPUReg37XX_RegFieldWrapper<"ppe_fp_bias", 32, 0, "FP">;
def reg_PpeFpBias : NPUReg37XX_RegisterWrapper<"ppe_fp_bias", 4, 0xF4, ["ppe_fp_bias"]>;

// ppe_fp_scale
def bf_PpeFpScale : NPUReg37XX_RegFieldWrapper<"ppe_fp_scale", 32, 0, "FP">;
def reg_PpeFpScale : NPUReg37XX_RegisterWrapper<"ppe_fp_scale", 4, 0xF8, ["ppe_fp_scale"]>;

// ppe_fp_prelu
def bf_PpeFpPrelu : NPUReg37XX_RegFieldWrapper<"ppe_fp_prelu", 32, 0, "FP">;
def reg_PpeFpPrelu : NPUReg37XX_RegisterWrapper<"ppe_fp_prelu", 4, 0xFC, ["ppe_fp_prelu"]>;

// ppe_fp_cfg
def bf_PpeFpConvert : NPUReg37XX_RegFieldWrapper<"ppe_fp_convert", 3, 0>;
def bf_PpeFpBypass : NPUReg37XX_RegFieldWrapper<"ppe_fp_bypass", 1, 3>;
def bf_PpeBf16Round : NPUReg37XX_RegFieldWrapper<"ppe_bf16_round", 1, 4>;
def bf_PpeFpPreluEn : NPUReg37XX_RegFieldWrapper<"ppe_fp_prelu_en", 1, 5>;
//  Note: ppe prelu needs to be off by default, only turned on explicitly
def reg_PpeFpCfg : NPUReg37XX_RegisterWrapper<"ppe_fp_cfg", 4, 0x100, ["ppe_fp_convert", "ppe_fp_bypass", "ppe_bf16_round", "ppe_fp_prelu_en"], 0x00000008>;

//  hwp_cmx_base_offset_
def bf_HwpCmxBaseOffset : NPUReg37XX_RegFieldWrapper<"hwp_cmx_base_offset_", 32, 0>;
def reg_HwpCmxBaseOffset : NPUReg37XX_RegisterWrapper<"hwp_cmx_base_offset_", 4, 0x104, ["hwp_cmx_base_offset_"]>;

// barriers_
def bf_BarrierWaitMask : NPUReg37XX_RegFieldWrapper<"barriers_wait_mask_", 64, 0>;
def reg_BarrierWaitMask : NPUReg37XX_RegisterWrapper<"barriers_wait_mask_", 8, 0x108, ["barriers_wait_mask_"]>;

def bf_BarrierWaitPost : NPUReg37XX_RegFieldWrapper<"barriers_post_mask_", 64, 0>;
def reg_BarrierWaitPost : NPUReg37XX_RegisterWrapper<"barriers_post_mask_", 8, 0x110, ["barriers_post_mask_"]>;

def bf_BarrierGroup : NPUReg37XX_RegFieldWrapper<"group_", 8, 0>;
def bf_BarrierMask : NPUReg37XX_RegFieldWrapper<"mask_", 8, 8>;
def reg_BarrierGroupMask : NPUReg37XX_RegisterWrapper<"barriers_group_mask_", 8, 0x118, ["group_", "mask_"]>;

// barriers_sched_
def bf_StartAfter : NPUReg37XX_RegFieldWrapper<"start_after_", 32, 0>;
def bf_CleanAfter : NPUReg37XX_RegFieldWrapper<"clean_after_", 32, 32>;
def reg_BarriersSchedulingCfg : NPUReg37XX_RegisterWrapper<"barriers_sched_", 8, 0x120, ["start_after_", "clean_after_"]>;

// variant_count_
def bf_VariantCnt : NPUReg37XX_RegFieldWrapper<"variant_count_", 16, 0>;
def reg_VariantCnt : NPUReg37XX_RegisterWrapper<"variant_count_", 2, 0x128, ["variant_count_"]>;

// cluster_
def bf_Cluster : NPUReg37XX_RegFieldWrapper<"cluster_", 8, 0>;
def reg_ClusterInvariant : NPUReg37XX_RegisterWrapper<"cluster_invariant_", 1, 0x12A, ["cluster_"]>;

// is_cont_conv_
def bf_IsContConv : NPUReg37XX_RegFieldWrapper<"is_cont_conv_", 8, 0>;
def reg_IsContConv : NPUReg37XX_RegisterWrapper<"is_cont_conv_", 1, 0x12B, ["is_cont_conv_"]>;

// dpu_prof_mode_
def bf_dpuProfMode : NPUReg37XX_RegFieldWrapper<"dpu_prof_mode_", 8, 0>;
def reg_dpuProfMode : NPUReg37XX_RegisterWrapper<"dpu_prof_mode_", 1, 0x12C, ["dpu_prof_mode_"]>;

// pad_[3]
def bf_invPad3 : NPUReg37XX_RegFieldWrapper<"inv_pad_3", 24, 0>;
def reg_invPad3 : NPUReg37XX_RegisterWrapper<"inv_pad_3", 3, 0x12D, ["inv_pad_3"]>;

// DPUInvariantRegister
def reg_mapped_DpuInvariantRegister : NPUReg37XX_RegMappedWrapper<"DpuInvariantRegister",
    ["se_sp_addr_0", "se_sp_addr_1", "se_sp_addr_2", "se_sp_addr_3",
    "se_sp_size_0", "se_sp_size_1", "se_sp_size_2",
    "z_config",
    "kernel_pad_cfg",
    "weight_size_placeholder", "weight_num_placeholder",
    "weight_start",
    "tensor_size0", "tensor_size1",
    "tensor_start",
    "tensor_mode",
    "elops_sparsity_addr",
    "elops_se_addr",
    "elops_wload",
    "tensor_mode",
    "act_offset0", "act_offset1", "act_offset2", "act_offset3",
    "base_offset_a", "base_offset_b",
    "wt_offset",
    "odu_cfg",
    "odu_be_size",
    "odu_be_cnt",
    "se_size",
    "te_dim0", "te_dim1",
    "pt_base",
    "sp_base",
    "base_ptr_a", "base_ptr_b",
    "base_adr0", "base_adr1", "base_adr2", "base_adr3",
    "odu_cast0", "odu_cast1", "odu_cast2", "odu_cast3",
    "mpe_cfg",
    "elop_scale",
    "ppe_cfg", "ppe_bias", "ppe_scale", "ppe_scale_ctrl", "ppe_prelu",
    "ppe_scale_hclamp", "ppe_scale_lclamp",
    "ppe_misc",
    "ppe_fp_bias", "ppe_fp_scale", "ppe_fp_prelu", "ppe_fp_cfg",
    "hwp_cmx_base_offset_",
    "barriers_wait_mask_", "barriers_post_mask_", "barriers_group_mask_",
    "barriers_sched_", "variant_count_", "cluster_invariant_", "is_cont_conv_",
    "dpu_prof_mode_", "inv_pad_3"
    ]>;

//
// DPUVariantRegisters
//

// workload_size0
def bf_WorkloadSizeX : NPUReg37XX_RegFieldWrapper<"workload_size_x", 14, 0>;
def bf_WorkloadSizeY : NPUReg37XX_RegFieldWrapper<"workload_size_y", 14, 14>;

def reg_WorkloadSize0 : NPUReg37XX_RegisterWrapper<"workload_size0", 4, 0x0, ["workload_size_x", "workload_size_y"]>;

// workload_size1
def bf_WorkloadSizeZ : NPUReg37XX_RegFieldWrapper<"workload_size_z", 14, 0>;
def bf_PadCountUp : NPUReg37XX_RegFieldWrapper<"pad_count_up", 3, 14>;
def bf_PadCountLeft : NPUReg37XX_RegFieldWrapper<"pad_count_left", 3, 17>;
def bf_PadCountDown : NPUReg37XX_RegFieldWrapper<"pad_count_down", 3, 20>;
def bf_PadCountRight : NPUReg37XX_RegFieldWrapper<"pad_count_right", 3, 23>;

def reg_WorkloadSize1 : NPUReg37XX_RegisterWrapper<"workload_size1", 4, 0x4,
    ["workload_size_z", "pad_count_up", "pad_count_left", "pad_count_down", "pad_count_right"]>;

// workload_start0
def bf_WorkloadStartX : NPUReg37XX_RegFieldWrapper<"workload_start_x", 14, 0>;
def bf_WorkloadStartY : NPUReg37XX_RegFieldWrapper<"workload_start_y", 14, 14>;
def reg_WorkloadStart0 : NPUReg37XX_RegisterWrapper<"workload_start0", 4, 0x8, ["workload_start_x", "workload_start_y"]>;

// workload_start1
def bf_WorkloadStartZ : NPUReg37XX_RegFieldWrapper<"workload_start_z", 14, 0>;
def reg_WorkloadStart1 : NPUReg37XX_RegisterWrapper<"workload_start1", 4, 0xC, ["workload_start_z"]>;

// offset_addr
def bf_NthwNtk : NPUReg37XX_RegFieldWrapper<"nthw_ntk", 2, 0>;
def bf_BinCfg : NPUReg37XX_RegFieldWrapper<"bin_cfg", 1, 2>;
def bf_ConvCond : NPUReg37XX_RegFieldWrapper<"conv_cond", 1, 3>;
def bf_DenseSe : NPUReg37XX_RegFieldWrapper<"dense_se", 1, 4>;
def bf_IdxQuad : NPUReg37XX_RegFieldWrapper<"idx_quad", 1, 5>;  // unused
def bf_SwizzleKeyOffsetAddr : NPUReg37XX_RegFieldWrapper<"swizzle_key_offset", 3, 6>;
def bf_IduMrmClkEn : NPUReg37XX_RegFieldWrapper<"idu_mrm_clk_en", 1, 9>;
def bf_OduClkEn : NPUReg37XX_RegFieldWrapper<"odu_clk_en", 1, 10>;
def bf_MpeClkEn : NPUReg37XX_RegFieldWrapper<"mpe_clk_en", 1, 11>;
def bf_PpeClkEn : NPUReg37XX_RegFieldWrapper<"ppe_clk_en", 1, 12>;
def bf_OduStatEn : NPUReg37XX_RegFieldWrapper<"odu_stat_en", 1, 13>;
def bf_IduStatEn : NPUReg37XX_RegFieldWrapper<"idu_stat_en", 1, 14>;
def bf_OduStatClrMode : NPUReg37XX_RegFieldWrapper<"odu_stat_clr_mode", 1, 16>;
def bf_IduStatClrMode : NPUReg37XX_RegFieldWrapper<"idu_stat_clr_mode", 1, 17>;
def bf_ShaveL2CacheEn : NPUReg37XX_RegFieldWrapper<"shave_l2_cache_en", 1, 19>;
def bf_IduDbgEn : NPUReg37XX_RegFieldWrapper<"idu_dbg_en", 2, 20>;
def bf_WtSwizzleKey : NPUReg37XX_RegFieldWrapper<"wt_swizzle_key", 3, 27>;
def bf_WtSwizzleSel : NPUReg37XX_RegFieldWrapper<"wt_swizzle_sel", 1, 30>;
//  Note: wt_swizzle_sel needs to be on by default to match NNRT GF behaviour
def reg_OffsetAddr : NPUReg37XX_RegisterWrapper<"offset_addr", 4, 0x10,
    ["nthw_ntk", "bin_cfg", "conv_cond", "dense_se", "swizzle_key_offset", "idu_mrm_clk_en", "odu_clk_en",
    "mpe_clk_en", "ppe_clk_en", "odu_stat_en", "idu_stat_en", "odu_stat_clr_mode", "idu_stat_clr_mode", "shave_l2_cache_en",
    "idu_dbg_en", "wt_swizzle_key", "wt_swizzle_sel"], 0x40000000>;

// te_end0
def bf_TeEndY : NPUReg37XX_RegFieldWrapper<"te_end_y", 13, 0>;
def bf_TeEndZ : NPUReg37XX_RegFieldWrapper<"te_end_z", 13, 13>;
def reg_TeEnd0 : NPUReg37XX_RegisterWrapper<"te_end0", 4, 0x14, ["te_end_y", "te_end_z"]>;

// te_end1
def bf_TeEndX : NPUReg37XX_RegFieldWrapper<"te_end_x", 13, 0>;
def reg_TeEnd1 : NPUReg37XX_RegisterWrapper<"te_end1", 4, 0x18, ["te_end_x"]>;

// te_beg0
def bf_TeBegY : NPUReg37XX_RegFieldWrapper<"te_beg_y", 13, 0>;
def bf_TeBegZ : NPUReg37XX_RegFieldWrapper<"te_beg_z", 13, 13>;
def reg_TeBeg0 : NPUReg37XX_RegisterWrapper<"te_beg0", 4, 0x1C, ["te_beg_y", "te_beg_z"]>;

// te_beg1
def bf_TeBegX : NPUReg37XX_RegFieldWrapper<"te_beg_x", 13, 0>;
def reg_TeBeg1 : NPUReg37XX_RegisterWrapper<"te_beg1", 4, 0x20, ["te_beg_x"]>;

// weight_size
def bf_WeightSize : NPUReg37XX_RegFieldWrapper<"weight_size", 32, 0>;
def reg_WeightSize : NPUReg37XX_RegisterWrapper<"weight_size", 4, 0x24, ["weight_size"]>;

// weight_num
def bf_WeighNum : NPUReg37XX_RegFieldWrapper<"weight_num", 32, 0>;
def reg_WeightNum : NPUReg37XX_RegisterWrapper<"weight_num", 4, 0x28, ["weight_num"]>;

// invariant_
def bf_Invariant : NPUReg37XX_RegFieldWrapper<"invariant_", 32, 0>;
def reg_Invariant : NPUReg37XX_RegisterWrapper<"invariant_", 4, 0x2C, ["invariant_"]>;

def bf_InvariantIndex : NPUReg37XX_RegFieldWrapper<"invariant_index_", 32, 0>;
def reg_InvariantIndex : NPUReg37XX_RegisterWrapper<"invariant_index_", 4, 0x30, ["invariant_index_"]>;

def bf_WeightTableOffset : NPUReg37XX_RegFieldWrapper<"weight_table_offset_", 32, 0>;
def reg_WeightTableOffset : NPUReg37XX_RegisterWrapper<"weight_table_offset_", 4, 0x34, ["weight_table_offset_"]>;

def bf_WloadId : NPUReg37XX_RegFieldWrapper<"wload_id_", 32, 0>;
def reg_WloadId : NPUReg37XX_RegisterWrapper<"wload_id_", 4, 0x38, ["wload_id_"]>;

def reg_Cluster : NPUReg37XX_RegisterWrapper<"cluster_variant_", 4, 0x3C, ["cluster_"]>;

// DPUVariantRegister
def reg_mapped_DpuVariantRegister : NPUReg37XX_RegMappedWrapper<"DpuVariantRegister",
    ["workload_size0", "workload_size1", "workload_start0", "workload_start1", "offset_addr", "te_end0", "te_end1", "te_beg0", "te_beg1",
    "weight_size", "weight_num", "invariant_", "invariant_index_", "weight_table_offset_", "wload_id_", "cluster_variant_"]>;

// DMADescriptor

//
//  dma_64_bit_word_0
//

def bf_dma_link_address : NPUReg37XX_RegFieldWrapper<"dma_link_address", 37, 0>;                           // Pointer to the next element in linked list
def reg_dma_link_address : NPUReg37XX_RegisterWrapper<"dma_link_address", 8, 0x0, ["dma_link_address"]>;

def bf_dma_watermark : NPUReg37XX_RegFieldWrapper<"dma_watermark", 1, 63>;                                  // Watermark to indicate that the job has completed
def reg_dma_watermark : NPUReg37XX_RegisterWrapper<"dma_watermark", 8, 0x0, ["dma_watermark"]>;

//
//  dma_64_bit_word_1
//

def bf_dma_type : NPUReg37XX_RegFieldWrapper<"dma_type", 2, 0>;
def bf_dma_burst_length : NPUReg37XX_RegFieldWrapper<"dma_burst_length", 8, 2>;
def bf_dma_critical : NPUReg37XX_RegFieldWrapper<"dma_critical", 1, 10>;
def bf_dma_interrupt_en : NPUReg37XX_RegFieldWrapper<"dma_interrupt_en", 1, 11>;
def bf_dma_interrupt_trigger : NPUReg37XX_RegFieldWrapper<"dma_interrupt_trigger", 7, 12>;
def bf_dma_skip_nr : NPUReg37XX_RegFieldWrapper<"dma_skip_nr", 7, 19>;
def bf_dma_order_forced : NPUReg37XX_RegFieldWrapper<"dma_order_forced", 1, 26>;
def bf_dma_watermark_en : NPUReg37XX_RegFieldWrapper<"dma_watermark_en", 1, 27>;
def bf_dma_dec_en : NPUReg37XX_RegFieldWrapper<"dma_dec_en", 1, 28>;
def bf_dma_barrier_en : NPUReg37XX_RegFieldWrapper<"dma_barrier_en", 1, 29>;

def reg_dma_cfg_bits : NPUReg37XX_RegisterWrapper<"dma_cfg_bits", 8, 0x8,
    ["dma_type", "dma_burst_length", "dma_critical", "dma_interrupt_en", "dma_interrupt_trigger", "dma_skip_nr",
     "dma_order_forced", "dma_watermark_en", "dma_dec_en", "dma_barrier_en"]>;

//
//  dma_64_bit_word_2
//

def bf_dma_src : NPUReg37XX_RegFieldWrapper<"dma_src", 38, 0>;
def reg_dma_src : NPUReg37XX_RegisterWrapper<"dma_src", 8, 0x10, ["dma_src"]>;

//
//  dma_64_bit_word_3
//

def bf_dma_dst : NPUReg37XX_RegFieldWrapper<"dma_dst", 38, 0>;
def reg_dma_dst : NPUReg37XX_RegisterWrapper<"dma_dst", 8, 0x18, ["dma_dst"]>;

//
//  dma_64_bit_word_4
//

def bf_dma_length : NPUReg37XX_RegFieldWrapper<"dma_length", 24, 0>;
def reg_dma_length : NPUReg37XX_RegisterWrapper<"dma_length", 4, 0x20, ["dma_length"]>;

def bf_dma_num_planes : NPUReg37XX_RegFieldWrapper<"dma_num_planes", 8, 0>;
def reg_dma_num_planes : NPUReg37XX_RegisterWrapper<"dma_num_planes", 1, 0x24, ["dma_num_planes"]>;

def bf_dma_task_id : NPUReg37XX_RegFieldWrapper<"dma_task_id", 24, 0>;
def reg_dma_task_id : NPUReg37XX_RegisterWrapper<"dma_task_id", 3, 0x25, ["dma_task_id"]>;

//
//  dma_64_bit_word_5
//

def bf_dma_src_plane_stride : NPUReg37XX_RegFieldWrapper<"dma_src_plane_stride", 32, 0>;
def reg_dma_src_plane_stride : NPUReg37XX_RegisterWrapper<"dma_src_plane_stride", 4, 0x28, ["dma_src_plane_stride"]>;

def bf_dma_dst_plane_stride : NPUReg37XX_RegFieldWrapper<"dma_dst_plane_stride", 32, 0>;
def reg_dma_dst_plane_stride : NPUReg37XX_RegisterWrapper<"dma_dst_plane_stride", 4, 0x2C, ["dma_dst_plane_stride"]>;

//
//  dma_64_bit_word_6
//

//  2D case

def bf_dma_attr2d_src_width : NPUReg37XX_RegFieldWrapper<"dma_attr2d_src_width", 24, 0>;
def reg_dma_attr2d_src_width : NPUReg37XX_RegisterWrapper<"dma_attr2d_src_width", 4, 0x30, ["dma_attr2d_src_width"]>;

def bf_dma_attr2d_src_stride : NPUReg37XX_RegFieldWrapper<"dma_attr2d_src_stride", 32, 0>;
def reg_dma_attr2d_src_stride : NPUReg37XX_RegisterWrapper<"dma_attr2d_src_stride", 4, 0x34, ["dma_attr2d_src_stride"]>;

// 1D case

def bf_dma_barriers_prod_mask : NPUReg37XX_RegFieldWrapper<"dma_barriers_prod_mask", 64, 0>;
def reg_dma_barriers1d_prod_mask : NPUReg37XX_RegisterWrapper<"dma_barriers1d_prod_mask", 8, 0x30, ["dma_barriers_prod_mask"]>;

//
//  dma_64_bit_word_7
//

//  2D case

def bf_dma_attr2d_dst_width : NPUReg37XX_RegFieldWrapper<"dma_attr2d_dst_width", 24, 0>;
def reg_dma_attr2d_dst_width : NPUReg37XX_RegisterWrapper<"dma_attr2d_dst_width", 4, 0x38, ["dma_attr2d_dst_width"]>;

def bf_dma_attr2d_dst_stride : NPUReg37XX_RegFieldWrapper<"dma_attr2d_dst_stride", 32, 0>;
def reg_dma_attr2d_dst_stride : NPUReg37XX_RegisterWrapper<"dma_attr2d_dst_stride", 4, 0x3C, ["dma_attr2d_dst_stride"]>;

// 1D case

def bf_dma_barriers_cons_mask : NPUReg37XX_RegFieldWrapper<"dma_barriers_cons_mask", 64, 0>;
def reg_dma_barriers1d_cons_mask : NPUReg37XX_RegisterWrapper<"dma_barriers1d_cons_mask", 8, 0x38, ["dma_barriers_cons_mask"]>;

//
//  dma_64_bit_word_8 (used for 2d case only)
//

def reg_dma_barriers_prod_mask : NPUReg37XX_RegisterWrapper<"dma_barriers_prod_mask", 8, 0x40, ["dma_barriers_prod_mask"]>;

//
//  dma_64_bit_word_9 (used for 2d case only)
//

def reg_dma_barriers_cons_mask : NPUReg37XX_RegisterWrapper<"dma_barriers_cons_mask", 8, 0x48, ["dma_barriers_cons_mask"]>;

//
//  VpuTaskSchedulingBarrierConfig
//

def bf_dma_barriers_sched_start_after : NPUReg37XX_RegFieldWrapper<"dma_barriers_sched_start_after", 32, 0>;
def bf_dma_barriers_sched_clean_after : NPUReg37XX_RegFieldWrapper<"dma_barriers_sched_clean_after", 32, 32>;
def reg_VpuTaskSchedulingBarrierConfig : NPUReg37XX_RegisterWrapper<"dma_barriers_sched", 8, 0x50,
    ["dma_barriers_sched_start_after", "dma_barriers_sched_clean_after"]>;

//
//  Pads
//

def reg_dma_pad_0 : NPUReg37XX_RegisterWrapper<"dma_pad_0", 8, 0x58, []>;
def reg_dma_pad_1 : NPUReg37XX_RegisterWrapper<"dma_pad_1", 8, 0x60, []>;
def reg_dma_pad_2 : NPUReg37XX_RegisterWrapper<"dma_pad_2", 8, 0x68, []>;
def reg_dma_pad_3 : NPUReg37XX_RegisterWrapper<"dma_pad_3", 8, 0x70, []>;
def reg_dma_pad_4 : NPUReg37XX_RegisterWrapper<"dma_pad_4", 8, 0x78, []>;


// DMA descriptor

def reg_mapped_DMARegister : NPUReg37XX_RegMappedWrapper<"DMARegister",
    ["dma_link_address", "dma_watermark",
    "dma_cfg_bits",
    "dma_src",
    "dma_dst",
    "dma_length", "dma_num_planes", "dma_task_id",
    "dma_src_plane_stride", "dma_dst_plane_stride",
    "dma_attr2d_src_width", "dma_attr2d_src_stride", "dma_barriers1d_prod_mask",
    "dma_attr2d_dst_width", "dma_attr2d_dst_stride", "dma_barriers1d_cons_mask",
    "dma_barriers_prod_mask",
    "dma_barriers_cons_mask",
    "dma_barriers_sched", "dma_pad_0",
    "dma_pad_1",
    "dma_pad_2",
    "dma_pad_3",
    "dma_pad_4"
    ]>;

//
// ActKernelRange registers
//

def bf_ActWLType : NPUReg37XX_RegFieldWrapper<"type", 8, 0>;
def reg_ActWLType : NPUReg37XX_RegisterWrapper<"type", 1, 0x0, ["type"]>; // WL_KERNEL = 0x00, WL_DEBUG = 0x04, WL_UNKNOWN

def bf_KernelEntry : NPUReg37XX_RegFieldWrapper<"kernel_entry", 32, 0>;
def reg_KernelEntry : NPUReg37XX_RegisterWrapper<"kernel_entry", 4, 0x4, ["kernel_entry"]>;

def bf_TextWindowBase: NPUReg37XX_RegFieldWrapper<"text_window_base", 32, 0>;
def reg_TextWindowBase : NPUReg37XX_RegisterWrapper<"text_window_base", 4, 0x8, ["text_window_base"]>;

def bf_CodeSize : NPUReg37XX_RegFieldWrapper<"code_size", 32, 0>;
def reg_CodeSize : NPUReg37XX_RegisterWrapper<"code_size", 4, 0xC, ["code_size"]>;

def bf_DataSecSize : NPUReg37XX_RegFieldWrapper<"data_sec_size", 32, 0>;
def reg_DataSecSize : NPUReg37XX_RegisterWrapper<"data_sec_size", 4, 0x10, ["data_sec_size"]>;

def bf_KernelInvoCount : NPUReg37XX_RegFieldWrapper<"kernel_invo_count", 32, 0>;
def reg_KernelInvoCount : NPUReg37XX_RegisterWrapper<"kernel_invo_count", 4, 0x14, ["kernel_invo_count"]>;

// ActKernelRange descriptor
def reg_mapped_ActKernelRangeDescriptor : NPUReg37XX_RegMappedWrapper<"VpuActKernelRange",
    ["type", "kernel_entry", "text_window_base", "code_size", "data_sec_size", "kernel_invo_count"]>;

//
// ActKernelInvocation registers
//

def bf_Range : NPUReg37XX_RegFieldWrapper<"range", 32, 0>;
def reg_Range : NPUReg37XX_RegisterWrapper<"range", 4, 0x0, ["range"]>;

def bf_KernelArgs : NPUReg37XX_RegFieldWrapper<"kernel_args", 32, 0>;
def reg_KernelArgs : NPUReg37XX_RegisterWrapper<"kernel_args", 4, 0x4, ["kernel_args"]>;

def bf_DataWindowBase: NPUReg37XX_RegFieldWrapper<"data_window_base", 32, 0>;
def reg_DataWindowBase : NPUReg37XX_RegisterWrapper<"data_window_base", 4, 0x8, ["data_window_base"]>;

def bf_PerfPacketOut : NPUReg37XX_RegFieldWrapper<"perf_packet_out", 32, 0>;
def reg_PerfPacketOut : NPUReg37XX_RegisterWrapper<"perf_packet_out", 4, 0xC, ["perf_packet_out"]>;

def bf_BarrierWaitMaskAct : NPUReg37XX_RegFieldWrapper<"barriers_wait_mask_act", 64, 0>;
def reg_BarrierWaitMaskAct : NPUReg37XX_RegisterWrapper<"barriers_wait_mask_act", 8, 0x10, ["barriers_wait_mask_act"]>;

def bf_BarrierWaitPostAct : NPUReg37XX_RegFieldWrapper<"barriers_post_mask_act", 64, 0>;
def reg_BarrierWaitPostAct : NPUReg37XX_RegisterWrapper<"barriers_post_mask_act", 8, 0x18, ["barriers_post_mask_act"]>;

def bf_BarrierGroupAct : NPUReg37XX_RegFieldWrapper<"group_act", 8, 0>;
def bf_BarrierMaskAct : NPUReg37XX_RegFieldWrapper<"mask_act", 8, 8>;
def reg_BarrierGroupMaskAct : NPUReg37XX_RegisterWrapper<"barriers_group_mask_act", 8, 0x20, ["group_act", "mask_act"]>;

def bf_ActInvoBarriersSchedStartAfter : NPUReg37XX_RegFieldWrapper<"act_invo_barriers_sched_start_after", 32, 0>;
def bf_ActInvoBarriersSchedCleanAfter : NPUReg37XX_RegFieldWrapper<"act_invo_barriers_sched_clean_after", 32, 32>;
def reg_SchedulingBarrierConfig : NPUReg37XX_RegisterWrapper<"act_invo_barriers_sched", 8, 0x28, ["act_invo_barriers_sched_start_after", "act_invo_barriers_sched_clean_after"]>;

def bf_InvoIndex : NPUReg37XX_RegFieldWrapper<"invo_index", 32, 0>;
def reg_InvoIndex : NPUReg37XX_RegisterWrapper<"invo_index", 4, 0x30, ["invo_index"]>;

def bf_InvoTile : NPUReg37XX_RegFieldWrapper<"invo_tile", 32, 0>;
def reg_InvoTile : NPUReg37XX_RegisterWrapper<"invo_tile", 4, 0x34, ["invo_tile"]>;

def bf_KernelRangeIndex : NPUReg37XX_RegFieldWrapper<"kernel_range_index", 32, 0>;
def reg_KernelRangeIndex : NPUReg37XX_RegisterWrapper<"kernel_range_index", 4, 0x38, ["kernel_range_index"]>;

def bf_ActInvoPad4 : NPUReg37XX_RegFieldWrapper<"actinvo_pad_4", 32, 0>;
def reg_ActInfoPad4 : NPUReg37XX_RegisterWrapper<"actinvo_pad_4", 4, 0x3C, ["actinvo_pad_4"]>;

// ActKernelInvocation descriptor
def reg_mapped_ActKernelInvocationDescriptor : NPUReg37XX_RegMappedWrapper<"VpuActKernelInvocation",
    ["range", "kernel_args", "data_window_base", "perf_packet_out",
    "barriers_wait_mask_act", "barriers_post_mask_act", "barriers_group_mask_act",
    "act_invo_barriers_sched", "invo_index", "invo_tile", "kernel_range_index", "actinvo_pad_4"]>;

//
//  BarrierCfg registers
//

def bf_NextSameId : NPUReg37XX_RegFieldWrapper<"next_same_id_", 32, 0, "UINT">;
def reg_NextSameId : NPUReg37XX_RegisterWrapper<"next_same_id_", 4, 0x0, ["next_same_id_"]>;

def bf_ProducerCount: NPUReg37XX_RegFieldWrapper<"producer_count_", 16, 0>;
def reg_ProducerCount : NPUReg37XX_RegisterWrapper<"producer_count_", 2, 0x4, ["producer_count_"]>;

def bf_ConsumerCount : NPUReg37XX_RegFieldWrapper<"consumer_count_", 16, 0>;
def reg_ConsumerCount : NPUReg37XX_RegisterWrapper<"consumer_count_", 2, 0x6, ["consumer_count_"]>;

def bf_BarrierId : NPUReg37XX_RegFieldWrapper<"real_id_", 8, 0>;
def reg_BarrierId : NPUReg37XX_RegisterWrapper<"real_id_", 1, 0x8, ["real_id_"]>;

def bf_BarCfg_Pad3 : NPUReg37XX_RegFieldWrapper<"barcfg_pad_3", 24, 0>;
def reg_BarCfg_Pad3 : NPUReg37XX_RegisterWrapper<"barcfg_pad_3", 3, 0x9, ["barcfg_pad_3"]>;

// BarrierCfg descriptor
def reg_mapped_BarrierDescriptor : NPUReg37XX_RegMappedWrapper<"VpuBarrierCountConfig",
    ["next_same_id_", "producer_count_", "consumer_count_", "real_id_", "barcfg_pad_3"]>;

#endif
