//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <npu_60xx_nnrt.hpp>
#include <type_traits>
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/lower_to_registers.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/descriptors.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/types.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

// Implementations of the lowering function that do not change between architectures:
namespace vpux::VPUIPDPU {

namespace detail {
template <typename T, typename = void>
struct has_Field_weight_table_formatType : std::false_type {};

template <typename T>
struct has_Field_weight_table_formatType<T, std::void_t<typename T::Field_weight_table_formatType>> : std::true_type {};
}  // namespace detail

// IDUGroupConvOp
template <typename Field_grp_conv_enType, typename DpuVariantDescriptorType>
void lowerToRegIDUGroupConvOp(DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_grp_conv_enType>(1);
}

// MPEEngineModeOp
template <typename Field_cim_enType, typename DpuInvariantDescriptorType>
void lowerToRegMPEEngineModeOp(VPUIPDPU::MPEEngineModeOp op, DpuInvariantDescriptorType& descriptor) {
    // 0 - DPU workloads will be directed at the Sparse Cell Array (Default value)
    // 1 - DPU workloads will be directed at the Compute in Memory
    if (op.getMpeEngineMode() == VPU::MPEEngine60XXMode::DCIM) {
        descriptor.template write<Field_cim_enType>(1);
    }
}

// IDUpalletLUTReadOp
template <typename Field_invar_plt_rd_enType, typename DpuVariantDescriptorType>
void lowerToRegIDUpalletLUTReadOp(DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_invar_plt_rd_enType>(1);
}

// WeightsTableZpBitwidthOp
template <typename Field_zero_point_4bType, typename DpuVariantDescriptorType>
void lowerToRegWeightsTableZpBitwidthOp(VPUIPDPU::WeightsTableZpBitwidthOp op, DpuVariantDescriptorType& descriptor) {
    // 0 - weights ZP is represented as an Integer on 8 bits
    // 1 - weights ZP is represented as an Integer on 4 bits
    if (op.getZpBitwidth() == VPUIPDPU::WeightsZpBitwidthType::ZP_4b) {
        descriptor.template write<Field_zero_point_4bType>(1);
    }
}

// IDUSubtractMpeWtBiasOp
template <typename Field_subtract_mpe_wtbiasType, typename DpuInvariantDescriptorType>
void lowerToRegIDUSubtractMpeWtBiasOp(DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_subtract_mpe_wtbiasType>(1);
}

}  // namespace vpux::VPUIPDPU

namespace vpux::VPUIPDPU::arch60xx {

struct RegsTypeList {
    using Type_amode = NPUReg60XX::RegField_amodeType;
    using Type_wmode = NPUReg60XX::RegField_wmodeType;
    using Type_dma_acc_info_compress_dtype = NPUReg60XX::RegField_dma_acc_info_compress_dtypeType;
    using Type_dma_acc_info_decompress_dtype = NPUReg60XX::RegField_dma_acc_info_decompress_dtypeType;
};

// IDUStorageElementOp
struct FieldsIDUStorageElementOp {
    using Field_se_z_splitType = NPUReg60XX::Fields::se_z_split;
    using Field_npo2_se_z_split_enType = NPUReg60XX::Fields::npo2_se_z_split_en;
    using Field_npo2_se_sizeType = NPUReg60XX::Fields::npo2_se_size;
    using Field_num_ses_in_z_dirType = NPUReg60XX::Fields::num_ses_in_z_dir;
};

// IDUKernelOp
struct FieldsIDUKernelOp {
    using Field_kernel_xType = NPUReg60XX::Fields::kernel_x;
    using Field_kernel_yType = NPUReg60XX::Fields::kernel_y;
};

// IDUStrideOp
struct FieldsIDUStrideOp {
    using Field_strideType = NPUReg60XX::Fields::stride;
    using Field_stride_yType = NPUReg60XX::Fields::stride_y;
    using Field_stride_y_enType = NPUReg60XX::Fields::stride_y_en;
};

// IDUInActivationsOp
struct FieldsIDUInActivationsOp {
    using Field_tensor_size_xType = NPUReg60XX::Fields::tensor_size_x;
    using Field_tensor_size_yType = NPUReg60XX::Fields::tensor_size_y;
    using Field_tensor_size_zType = NPUReg60XX::Fields::tensor_size_z;
    using Field_amodeType = NPUReg60XX::Fields::amode;
    using Field_act_denseType = NPUReg60XX::Fields::act_dense;
    using Field_tensor_cmp_sizeType = NPUReg60XX::Fields::tensor_cmp_size;
};

template <typename Type_Fields, typename FindRegType, typename RegsTypeList, typename VpuInputTensorDTypeEnum,
          typename DpuInvariantDescriptorType>
void lowerToRegIDUInActivationsOp(VPUIPDPU::IDUInActivationsOp op, DpuInvariantDescriptorType& descriptor) {
    auto inActivations = op.getInActivations();
    auto inActivationsType = mlir::cast<vpux::NDTypeInterface>(inActivations.getType()).getElementType();
    auto inActivationShape = getShape(inActivations);
    const auto dimY = inActivationShape[Dims4D::Act::H];
    const auto dimX = inActivationShape[Dims4D::Act::W];
    const auto dimZ = inActivationShape[Dims4D::Act::C];
    auto tensorMode = getTensorMode<FindRegType, RegsTypeList, VpuInputTensorDTypeEnum>(inActivationsType);
    auto actDense = !op.getInSparse();

    if (mlir::dyn_cast<mlir::quant::QuantizedType>(inActivationsType)) {
        inActivationsType = mlir::quant::QuantizedType::castToStorageType(inActivationsType);
    }
    const auto padAuto16bit =
            (inActivationsType.getIntOrFloatBitWidth() == 16) && inActivationShape[Dims4D::Act::C] < 10;

    const auto padAuto8bit = (inActivationsType.getIntOrFloatBitWidth() == 8) && inActivationShape[Dims4D::Act::C] < 16;

    if (padAuto16bit || padAuto8bit) {
        descriptor.template write<typename Type_Fields::Field_tensor_cmp_sizeType>(inActivationShape[Dims4D::Act::C]);
    }

    descriptor.template write<typename Type_Fields::Field_tensor_size_xType>(dimX);
    descriptor.template write<typename Type_Fields::Field_tensor_size_yType>(dimY);
    descriptor.template write<typename Type_Fields::Field_tensor_size_zType>(dimZ);
    descriptor.template write<typename Type_Fields::Field_amodeType>(tensorMode);
    if constexpr (has_Field_act_denseType<Type_Fields>::value) {
        descriptor.template write<typename Type_Fields::Field_act_denseType>(actDense);
    }
}

// IDUInputLayerCfgOp
struct FieldsIDUInputLayerCfgOp {
    using Field_cm_sp_patternType = NPUReg60XX::Fields::cm_sp_pattern;
    using Field_act_denseType = NPUReg60XX::Fields::act_dense;
    using Field_wt_denseType = NPUReg60XX::Fields::wt_dense;
    using Field_layer1_wt_sp_insType = NPUReg60XX::Fields::layer1_wt_sp_ins;
    using Field_layer1_cmp_enType = NPUReg60XX::Fields::layer1_cmp_en;
    using Field_tensor_size_zType = NPUReg60XX::Fields::tensor_size_z;
};

// IDUWeightsOp
struct FieldsIDUWeightsOp {
    using Field_wmodeType = NPUReg60XX::Fields::wmode;
    using Field_wt_denseType = NPUReg60XX::Fields::wt_dense;
    using Field_wt_plt_cfgType = NPUReg60XX::Fields::wt_plt_cfg;
    using Field_wt_plt_cfg_2Type = NPUReg60XX::Fields::wt_plt_cfg_2;
    using Field_pool_wt_dataType = NPUReg60XX::Fields::pool_wt_data;
};

// IDUWorkloadCfgOp
struct FieldsIDUWorkloadCfgOp {
    using Field_workload_operationType = NPUReg60XX::Fields::workload_operation;
    using Field_zm_inputType = NPUReg60XX::Fields::zm_input;
    using Field_dw_inputType = NPUReg60XX::Fields::dw_input;
    using Field_pool_wt_rd_disType = NPUReg60XX::Fields::pool_wt_rd_dis;
    using Field_dw_wt_sp_insType = NPUReg60XX::Fields::dw_wt_sp_ins;
    using Field_dynamic_bw_enType = NPUReg60XX::Fields::dynamic_bw_en;
    using Field_elop_wloadType = NPUReg60XX::Fields::elop_wload;
    using Field_flush_wlType = NPUReg60XX::Fields::flush_wl;
    using Field_amodeType = NPUReg60XX::Fields::amode;
    using Field_wmodeType = NPUReg60XX::Fields::wmode;
    using Field_ppe_modeType = NPUReg60XX::Fields::ppe_mode;
    using Field_tensor_size_xType = NPUReg60XX::Fields::tensor_size_x;
    using Field_tensor_size_yType = NPUReg60XX::Fields::tensor_size_y;
    using Field_tensor_size_zType = NPUReg60XX::Fields::tensor_size_z;
    using Field_kernel_xType = NPUReg60XX::Fields::kernel_x;
    using Field_kernel_yType = NPUReg60XX::Fields::kernel_y;
    using Field_weight_table_formatType = NPUReg60XX::Fields::weight_table_format;
};

// IDUDepthWiseCfgOp
struct FieldsIDUDepthWiseCfgOp {
    using Field_dw_3x3s1_opt_disType = NPUReg60XX::Fields::dw_3x3s1_opt_dis;
    using Field_dw_opt_enType = NPUReg60XX::Fields::dw_opt_en;
    using Field_dw_opt_offsetType = NPUReg60XX::Fields::dw_opt_offset;
    using Field_pool_opt_enType = NPUReg60XX::Fields::pool_opt_en;
};

// IDUEltWiseCfgOp
// contains new table-gen generation Field definitions
struct NewTableGenFieldsIDUEltWiseCfgOp {
    using Field_elop_scale_a_bfpType = NPUReg60XX::Fields::elop_scale_a_bfp;
    using Field_elop_scale_b_bfpType = NPUReg60XX::Fields::elop_scale_b_bfp;
    using Field_elop_scale_a_fpType = NPUReg60XX::Fields::elop_scale_a_fp;
    using Field_elop_scale_b_fpType = NPUReg60XX::Fields::elop_scale_b_fp;
    using Field_elop_scale_aType = NPUReg60XX::Fields::elop_scale_a;
    using Field_elop_scale_bType = NPUReg60XX::Fields::elop_scale_b;
};

// contains old table-gen generation Field definitions
struct OldTableGenFieldsIDUEltWiseCfgOp {
    using Field_elop_scale_a_bfpType = NPUReg60XX::RegField_elop_scale_a_bfpType;
    using Field_elop_scale_b_bfpType = NPUReg60XX::RegField_elop_scale_b_bfpType;
    using Field_elop_scale_a_fpType = NPUReg60XX::RegField_elop_scale_a_fpType;
    using Field_elop_scale_b_fpType = NPUReg60XX::RegField_elop_scale_b_fpType;
    using Field_elop_scale_aType = NPUReg60XX::RegField_elop_scale_aType;
    using Field_elop_scale_bType = NPUReg60XX::RegField_elop_scale_bType;
};

// WeightTableSplitOp
struct FieldsWeightTableSplitOp {
    using Field_weight_table_formatType = NPUReg60XX::Fields::weight_table_format;
    using Field_zero_point_onlyType = NPUReg60XX::Fields::zero_point_only;
    using Field_per_group_zero_pointType = NPUReg60XX::Fields::per_group_zero_point;
    using Field_wt_ptr_disType = NPUReg60XX::Fields::wt_ptr_dis;
    using Field_ppe_scale_dtypeType = NPUReg60XX::Fields::ppe_scale_dtype;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegWeightTableSplitOp(VPUIPDPU::WeightTableSplitOp op, DpuInvariantDescriptorType& descriptor) {
    bool zeroPointOnly = false, wtPtrDis = false, perGroupZeroPoint = false;
    switch (auto wtSplitMode = op.getWtSplitMode()) {
    case WtSplitMode::NO_WT_READS:
        zeroPointOnly = true;  // set on true to avoid HW fail - E#169829
        wtPtrDis = true;
        perGroupZeroPoint = false;
        break;
    case WtSplitMode::WEIGHTS_ONLY:
        zeroPointOnly = false;
        wtPtrDis = false;
        perGroupZeroPoint = false;
        break;
    case WtSplitMode::ZERO_POINTS_ONLY:
        zeroPointOnly = true;
        wtPtrDis = false;
        perGroupZeroPoint = true;
        break;
    case WtSplitMode::WEIGHTS_AND_ZPS:
        zeroPointOnly = false;
        wtPtrDis = false;
        perGroupZeroPoint = true;
        break;
    default:
        VPUX_THROW("Invalid value for wt_split_mode ({0}) in op {1}", wtSplitMode, op);
    }
    if constexpr (detail::has_Field_weight_table_formatType<Type_Fields>::value) {
        descriptor.template write<typename Type_Fields::Field_weight_table_formatType>(1);
    }
    descriptor.template write<typename Type_Fields::Field_zero_point_onlyType>(zeroPointOnly);
    descriptor.template write<typename Type_Fields::Field_per_group_zero_pointType>(perGroupZeroPoint);
    descriptor.template write<typename Type_Fields::Field_wt_ptr_disType>(wtPtrDis);

    if (op.getScaleDtype().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_scale_dtypeType>(op.getScaleDtype().value());
    }
}

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegWeightTableLegacyZpOp(VPUIPDPU::WeightTableLegacyZpOp op, DpuInvariantDescriptorType& descriptor) {
    bool perGroupZeroPoint = false;
    switch (auto wtLegacyZpMode = op.getWtLegacyZpMode()) {
    case WtLegacyZpMode::WEIGHTS_PTR:
        perGroupZeroPoint = false;
        break;
    case WtLegacyZpMode::WEIGHTS_PTR_AND_ZPS:
        perGroupZeroPoint = true;
        break;
    default:
        VPUX_THROW("Invalid value for wt_legacy_zp_mode ({0}) in op {1}", wtLegacyZpMode, op);
    }
    descriptor.template write<typename Type_Fields::Field_weight_table_formatType>(0);
    descriptor.template write<typename Type_Fields::Field_zero_point_onlyType>(0);
    descriptor.template write<typename Type_Fields::Field_per_group_zero_pointType>(perGroupZeroPoint);
    descriptor.template write<typename Type_Fields::Field_wt_ptr_disType>(0);
}

// FieldsPPECfgOp
struct FieldsPPECfgOp {
    using Field_ppe_g8_bias_cType = NPUReg60XX::Fields::ppe_g8_bias_c;
    using Field_ppe_g8_bias_bType = NPUReg60XX::Fields::ppe_g8_bias_b;
    using Field_ppe_g8_bias_aType = NPUReg60XX::Fields::ppe_g8_bias_a;
};

// Bitfield configuration for output activations. The zero-point is 17 bits for 16-bit activations and 9 bits for 8-bit
// activations.
template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEIntZeroPointOffsetOp(VPUIPDPU::PPEIntZeroPointOffsetOp op, DpuInvariantDescriptorType& descriptor,
                                       VPUIPDPU::ODUDataBitWidth oduDataWidth) {
    auto zp = op.getZeroPointStatic();
    int64_t zpLower = zp & 0x1FF;
    if (zp & 0x100) {
        // if the sign bit is set, restore int64 negative value
        zpLower |= ~0x1FF;
    }
    // ppe_g8_bias_c is defined as 9-bit SINT, the expected offset range is within -128, +255
    descriptor.template write<typename Type_Fields::Field_ppe_g8_bias_cType>(zpLower);
    if (oduDataWidth == VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_16BIT) {
        uint64_t zpUpper = zp >> 9 & 0xFF;
        // ppe_g8_bias_b is defined as a 9-bit unsigned integer (UINT), with bits [7:0] utilized to provide an expected
        // offset range of -128 to +256.
        descriptor.template write<typename Type_Fields::Field_ppe_g8_bias_bType>(zpUpper);
    }
}

// PPEFpBiasAddOp
struct FieldsPPEFpBiasAddOp {
    using Field_ppe_fp_scale_overrideType = NPUReg60XX::Fields::ppe_fp_scale_override;
    using Field_ppe_fp_biasType = NPUReg60XX::Fields::ppe_fp_bias;
};

// PPEFpConvertOp
struct FieldsPPEFpConvertOp {
    using Field_ppe_fp_convertType = NPUReg60XX::Fields::ppe_fp_convert;
    using Field_ppe_fp16_clampType = NPUReg60XX::Fields::ppe_fp16_clamp;
    using Field_ppe_fp16_ftzType = NPUReg60XX::Fields::ppe_fp16_ftz;
};

// PPEFpScaleMultOp
struct FieldsPPEFpScaleMultOp {
    using Field_ppe_fp_scale_overrideType = NPUReg60XX::Fields::ppe_fp_scale_override;
    using Field_ppe_fp_scaleType = NPUReg60XX::Fields::ppe_fp_scale;
};

// PPEFpSprLUTModeOp
struct FieldsPPEFpSprLUTModeOp {
    using Field_ppe_modeType = NPUReg60XX::Fields::ppe_mode;
    using Field_ppe_lut_ptr_forceType = NPUReg60XX::Fields::ppe_lut_ptr_force;
};

// PPEFpPreluMultOp
struct FieldsPPEFpPreluMultOp {
    using Field_ppe_fp_preluType = NPUReg60XX::Fields::ppe_fp_prelu;
};

// PPEFpClampOp
struct FieldsPPEFpClampOp {
    using Field_ppe_fp_scale_hclampType = NPUReg60XX::Fields::ppe_fp_scale_hclamp;
    using Field_ppe_fp_scale_lclampType = NPUReg60XX::Fields::ppe_fp_scale_lclamp;
};

// ODUOutTensorSizeOp
struct FieldsODUOutTensorSizeOp {
    using Field_te_dim_xType = NPUReg60XX::Fields::te_dim_x;
    using Field_te_dim_yType = NPUReg60XX::Fields::te_dim_y;
    using Field_te_dim_zType = NPUReg60XX::Fields::te_dim_z;
};

// ODUSparsityOp
struct FieldsODUSparsityOp {
    using Field_sp_valueType = NPUReg60XX::Fields::sp_value;
    using Field_sp_out_enType = NPUReg60XX::Fields::sp_out_en;
    using Field_write_spType = NPUReg60XX::Fields::write_sp;
};

// ODUOutActivationsOp
struct FieldsODUOutActivationsOp {
    using Field_dtypeType = NPUReg60XX::Fields::dtype;
    using Field_write_acType = NPUReg60XX::Fields::write_ac;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegODUOutActivationsOp(VPUIPDPU::ODUOutActivationsOp op, DpuInvariantDescriptorType& descriptor) {
    uint64_t dataWidth = static_cast<uint64_t>(op.getDataWidth().value());
    uint64_t writeAc = op.getOutActivations() ? 1 : 0;

    descriptor.template write<typename Type_Fields::Field_dtypeType>(dataWidth);
    descriptor.template write<typename Type_Fields::Field_write_acType>(writeAc);
}

// ODUWriteCombineBufferOp
struct FieldsODUWriteCombineBufferOp {
    using Field_wcb_bypassType = NPUReg60XX::Fields::wcb_bypass;
    using Field_wcb_ac_modeType = NPUReg60XX::Fields::wcb_ac_mode;
    using Field_wcb_sp_modeType = NPUReg60XX::Fields::wcb_sp_mode;
};

// ODUReduceLayerOp
struct FieldsODUReduceLayerOp {
    using Field_mn_enableType = NPUReg60XX::Fields::mn_enable;
    using Field_mx_enableType = NPUReg60XX::Fields::mx_enable;
    using Field_mt_enableType = NPUReg60XX::Fields::mt_enable;
    using Field_min_max_dtypeType = NPUReg60XX::Fields::min_max_dtype;
    using Field_min_max_total_tile_selectType = NPUReg60XX::Fields::min_max_total_tile_select;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegODUReduceLayerOp(VPUIPDPU::ODUReduceLayerOp op, DpuInvariantDescriptorType& descriptor) {
    uint64_t tileSelect(0);
    if (op.getTileSelect().has_value()) {
        tileSelect = op.getTileSelect().value();
    }

    descriptor.template write<typename Type_Fields::Field_mn_enableType>(op.getReduceMin());
    descriptor.template write<typename Type_Fields::Field_mx_enableType>(op.getReduceMax());
    descriptor.template write<typename Type_Fields::Field_mt_enableType>(op.getReduceTotal());
    descriptor.template write<typename Type_Fields::Field_min_max_dtypeType>(op.getReduceDtype());
    descriptor.template write<typename Type_Fields::Field_min_max_total_tile_selectType>(tileSelect);
}

// ODULocalRegionOp
struct FieldsODULocalRegionOp {
    using Field_bf_LocalRegionEnable = NPUReg60XX::Fields::local_region_enable;
    using Field_bf_LocalRegionSuppress = NPUReg60XX::Fields::local_region_suppress;
    using Field_bf_LocalRegionBegX = NPUReg60XX::Fields::local_region_beg_x;
    using Field_bf_LocalRegionEndX = NPUReg60XX::Fields::local_region_end_x;
    using Field_bf_LocalRegionBegY = NPUReg60XX::Fields::local_region_beg_y;
    using Field_bf_LocalRegionEndY = NPUReg60XX::Fields::local_region_end_y;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegODULocalRegionOp(VPUIPDPU::ODULocalRegionOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_bf_LocalRegionEnable>(1);
    descriptor.template write<typename Type_Fields::Field_bf_LocalRegionSuppress>(op.getSuppressOutput());
    descriptor.template write<typename Type_Fields::Field_bf_LocalRegionBegX>(op.getBeginCoordX());
    descriptor.template write<typename Type_Fields::Field_bf_LocalRegionEndX>(op.getEndCoordX());
    descriptor.template write<typename Type_Fields::Field_bf_LocalRegionBegY>(op.getBeginCoordY());
    descriptor.template write<typename Type_Fields::Field_bf_LocalRegionEndY>(op.getEndCoordY());
}

// InvariantBarrierCfg
struct RegistersInvariantBarrierCfg {
    using Register_barriers_sched_Type = NPUReg60XX::Registers::barriers_sched_;
};

struct FieldsInvariantBarrierCfg {
    using Field_group_Type = NPUReg60XX::Fields::group_;
    using Field_mask_Type = NPUReg60XX::Fields::mask_;
    using Field_start_after_Type = NPUReg60XX::Fields::start_after_;
    using Field_clean_after_Type = NPUReg60XX::Fields::clean_after_;
    using Field_barriers_wait_mask_hi_Type = NPUReg60XX::Fields::barriers_wait_mask_hi_;
    using Field_barriers_wait_mask_lo_Type = NPUReg60XX::Fields::barriers_wait_mask_lo_;
    using Field_barriers_post_mask_hi_Type = NPUReg60XX::Fields::barriers_post_mask_hi_;
    using Field_barriers_post_mask_lo_Type = NPUReg60XX::Fields::barriers_post_mask_lo_;
};

// IDUWorkloadSetOp
struct FieldsIDUWorkloadSetOp {
    using Field_workload_start_xType = NPUReg60XX::Fields::workload_start_x;
    using Field_workload_start_yType = NPUReg60XX::Fields::workload_start_y;
    using Field_workload_start_zType = NPUReg60XX::Fields::workload_start_z;
    using Field_workload_size_xType = NPUReg60XX::Fields::workload_size_x;
    using Field_workload_size_yType = NPUReg60XX::Fields::workload_size_y;
    using Field_workload_size_zType = NPUReg60XX::Fields::workload_size_z;
};

// IDUPaddingOp
struct FieldsIDUPaddingOp {
    using Field_pad_count_upType = NPUReg60XX::Fields::pad_count_up;
    using Field_pad_count_leftType = NPUReg60XX::Fields::pad_count_left;
    using Field_pad_count_downType = NPUReg60XX::Fields::pad_count_down;
    using Field_pad_count_rightType = NPUReg60XX::Fields::pad_count_right;
};

// IDUWeightSetOp
struct FieldsIDUWeightSetOp {
    using Field_weight_sizeType = NPUReg60XX::Fields::weight_size;
    using Field_weight_numType = NPUReg60XX::Fields::weight_num;
    using Field_weight_startType = NPUReg60XX::Fields::weight_start;
};

// IDUPerOutputChannelScalingOp
struct FieldsIDUPerOutputChannelScalingOp {
    using Field_sb_read_enType = NPUReg60XX::Fields::sb_read_en;
    using Field_tensor2_act_denseType = NPUReg60XX::Fields::tensor2_act_dense;
};

// ODUOutSubtensorOp
struct FieldsODUOutSubtensorOp {
    using Field_te_beg_xType = NPUReg60XX::Fields::te_beg_x;
    using Field_te_beg_yType = NPUReg60XX::Fields::te_beg_y;
    using Field_te_beg_zType = NPUReg60XX::Fields::te_beg_z;
    using Field_te_end_xType = NPUReg60XX::Fields::te_end_x;
    using Field_te_end_yType = NPUReg60XX::Fields::te_end_y;
    using Field_te_end_zType = NPUReg60XX::Fields::te_end_z;
};

// ODUHaloCfgOp
struct RegistersODUHaloCfgOp {
    using Register_halo_region0AType = NPUReg60XX::Registers::halo_region0A;
    using Register_halo_region0BType = NPUReg60XX::Registers::halo_region0B;
    using Register_halo_region0CType = NPUReg60XX::Registers::halo_region0C;
    using Register_halo_region0DType = NPUReg60XX::Registers::halo_region0D;

    using Register_halo_region1AType = NPUReg60XX::Registers::halo_region1A;
    using Register_halo_region1BType = NPUReg60XX::Registers::halo_region1B;
    using Register_halo_region1CType = NPUReg60XX::Registers::halo_region1C;
    using Register_halo_region1DType = NPUReg60XX::Registers::halo_region1D;

    using Register_halo_region2AType = NPUReg60XX::Registers::halo_region2A;
    using Register_halo_region2BType = NPUReg60XX::Registers::halo_region2B;
    using Register_halo_region2CType = NPUReg60XX::Registers::halo_region2C;
    using Register_halo_region2DType = NPUReg60XX::Registers::halo_region2D;

    using Register_halo_region3AType = NPUReg60XX::Registers::halo_region3A;
    using Register_halo_region3BType = NPUReg60XX::Registers::halo_region3B;
    using Register_halo_region3CType = NPUReg60XX::Registers::halo_region3C;
    using Register_halo_region3DType = NPUReg60XX::Registers::halo_region3D;

    using Register_halo_region4AType = NPUReg60XX::Registers::halo_region4A;
    using Register_halo_region4BType = NPUReg60XX::Registers::halo_region4B;
    using Register_halo_region4CType = NPUReg60XX::Registers::halo_region4C;
    using Register_halo_region4DType = NPUReg60XX::Registers::halo_region4D;

    using Register_halo_region5AType = NPUReg60XX::Registers::halo_region5A;
    using Register_halo_region5BType = NPUReg60XX::Registers::halo_region5B;
    using Register_halo_region5CType = NPUReg60XX::Registers::halo_region5C;
    using Register_halo_region5DType = NPUReg60XX::Registers::halo_region5D;
};

struct FieldsODUHaloCfgOp {
    using Field_begin_xType = NPUReg60XX::Fields::begin_x;
    using Field_begin_yType = NPUReg60XX::Fields::begin_y;
    using Field_end_xType = NPUReg60XX::Fields::end_x;
    using Field_ac_adr_offsetType = NPUReg60XX::Fields::ac_adr_offset;
    using Field_target_width_lsbType = NPUReg60XX::Fields::target_width_lsb;
    using Field_target_width_msbType = NPUReg60XX::Fields::target_width_msb;
    using Field_tile_selectType = NPUReg60XX::Fields::tile_select;
    using Field_sp_adr_offsetType = NPUReg60XX::Fields::sp_adr_offset;
    using Field_enableType = NPUReg60XX::Fields::enable;
    using Field_end_yType = NPUReg60XX::Fields::end_y;
    using Field_halo_enable_new_descType = NPUReg60XX::Fields::halo_enable_new_desc;
};

struct FunctionsODUHaloCfgOp {
    using Function_target_width_lsbType = NPUReg60XX::RegField_target_width_lsbType;
    using Function_target_width_msbType = NPUReg60XX::RegField_target_width_msbType;
};

template <typename Type_Registers, typename Type_Fields, typename Type_Functions, typename DpuVariantDescriptorType>
void lowerToRegODUHaloCfgOp(VPUIPDPU::ODUHaloCfgOp op, DpuVariantDescriptorType& descriptor) {
    VPUIPDPU::arch40xx::lowerToRegODUHaloCfgOp<Type_Registers, Type_Fields, Type_Functions>(op, descriptor);
    uint8_t haloRegionIdx(0), haloEnNewDesc(0);
    for (const auto& haloRegionOp : op.getRegion().getOps()) {
        auto opHaloReg = mlir::dyn_cast_or_null<VPUIPDPU::ODUHaloRegionOp>(&haloRegionOp);
        if (opHaloReg == nullptr) {
            VPUX_THROW("Found invalid child op under ODUHaloCfgOp: {0}", haloRegionOp);
        }
        VPUX_THROW_WHEN(haloRegionIdx >= 6,
                        "Too many ODUHaloRegionOps inside region to represent in the 6-bit register: {0}",
                        haloRegionIdx + 1);
        haloEnNewDesc |= 1 << haloRegionIdx;
        haloRegionIdx++;
    }

    descriptor.template write<typename Type_Fields::Field_halo_enable_new_descType>(haloEnNewDesc);
}

// VariantBarrierCfg
struct FieldsVariantBarrierCfg {
    using Field_cbarrier_hiType = NPUReg60XX::Fields::cbarrier_hi;
    using Field_cbarrier_loType = NPUReg60XX::Fields::cbarrier_lo;
    using Field_pbarrier_hiType = NPUReg60XX::Fields::pbarrier_hi;
    using Field_pbarrier_loType = NPUReg60XX::Fields::pbarrier_lo;
};

// ODUDataReuseOp
template <typename Field_nthwType, typename DpuInvariantDescriptorType>
void lowerToRegODUDataReuseOp(VPUIPDPU::ODUDataReuseOp op, DpuInvariantDescriptorType& descriptor) {
    uint64_t activationReuseModeRegMapping = 0;

    switch (op.getActivationReuse()) {
    case ODUActivationReuseMode::NTHW_CIM:
        activationReuseModeRegMapping = 0;
        break;
    case ODUActivationReuseMode::NTHW_4:
        activationReuseModeRegMapping = 1;
        break;
    case ODUActivationReuseMode::NTHW_8:
        activationReuseModeRegMapping = 2;
        break;
    case ODUActivationReuseMode::NTHW_16:
        activationReuseModeRegMapping = 3;
        break;
    default:
        VPUX_THROW("Found invalid case for ODUActivationReuseMode");
    }

    descriptor.template write<Field_nthwType>(activationReuseModeRegMapping);
}

template <typename Type_Fields, typename FindRegType, typename RegsTypeList, typename VpuInputTensorDTypeEnum,
          typename DpuInvariantDescriptorType>
void lowerToRegCommonIDUWeightsOp(VPUIPDPU::IDUWeightsOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_wmodeType>(
            getTensorMode<FindRegType, RegsTypeList, VpuInputTensorDTypeEnum>(op.getWmode()));
    descriptor.template write<typename Type_Fields::Field_wt_denseType>(!op.getWtSparse());

    const std::map<IDUWeightPalletMode, std::pair<uint32_t, uint32_t>> palletConfigMap = {
            {IDUWeightPalletMode::NO_PLT, {0, 0}},       {IDUWeightPalletMode::ONE_BIT_PLT, {1, 0}},
            {IDUWeightPalletMode::TWO_BIT_PLT, {2, 0}},  {IDUWeightPalletMode::THREE_BIT_PLT, {0, 1}},
            {IDUWeightPalletMode::FOUR_BIT_PLT, {3, 0}}, {IDUWeightPalletMode::FIVE_BIT_PLT, {1, 1}},
            {IDUWeightPalletMode::SIX_BIT_PLT, {2, 1}}};
    const auto palletMode = op.getWtPltCfg();
    const auto wtPltCfgValue = palletConfigMap.at(palletMode).first;
    const auto wtPltCfg2Value = palletConfigMap.at(palletMode).second;

    descriptor.template write<typename Type_Fields::Field_wt_plt_cfgType>(wtPltCfgValue);
    descriptor.template write<typename Type_Fields::Field_wt_plt_cfg_2Type>(wtPltCfg2Value);
    if (op.getPoolWtData().has_value()) {
        descriptor.template write<typename Type_Fields::Field_pool_wt_dataType>(op.getPoolWtData().value());
    }
}

struct FieldsMPEActivationBiasOp {
    using Field_mpe_actbiasType = NPUReg60XX::Fields::mpe_actbias;
    using Field_ppe_g8_bias_aType = NPUReg60XX::Fields::ppe_g8_bias_a;
};

// Bitfield configuration setting the zp value for incoming 16-bit/8-bit activations
template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegMPEActivationBiasOp(VPUIPDPU::MPEActivationBiasOp op, DpuInvariantDescriptorType& descriptor,
                                   uint8_t inActDataWidth) {
    auto zp = op.getActBias();
    uint64_t zpLower = zp & 0xFF;
    // mpe_actbias is defined as an 8-bit UINT
    descriptor.template write<typename Type_Fields::Field_mpe_actbiasType>(zpLower);
    if (inActDataWidth == 16) {
        uint64_t zpUpper = (zp >> 8) & 0xFF;
        // ppe_g8_bias_a is defined as 9-bit UINT, out of which we use bits [7:0]
        descriptor.template write<typename Type_Fields::Field_ppe_g8_bias_aType>(zpUpper);
    }
}

// DynPrimitiveIdVarOp
struct FieldsDynPrimitiveIdVarOp {
    using Field_dyn_primitive_id_var_enType = NPUReg60XX::Fields::dyn_primitive_id_var_en;
    using Field_dyn_primitive_id_varType = NPUReg60XX::Fields::dyn_primitive_id_var;
    using Field_dyn_primitive_id2_varType = NPUReg60XX::Fields::dyn_primitive_id2_var;
};

template <typename Type_Fields, typename DpuVariantDescriptorType>
void lowerToRegDynPrimitiveIdVarOp(VPUIPDPU::DynPrimitiveIdVarOp op, DpuVariantDescriptorType& descriptor) {
    // when DYN_PRIMITIVE_ID_VAR_EN = 1, Dynamic Primitive IDs are taken from the variant descriptor
    descriptor.template write<typename Type_Fields::Field_dyn_primitive_id_var_enType>(1);
    descriptor.template write<typename Type_Fields::Field_dyn_primitive_id_varType>(op.getDynPrimitiveIdVar());
    if (op.getDynPrimitiveId2Var().has_value()) {
        descriptor.template write<typename Type_Fields::Field_dyn_primitive_id2_varType>(
                op.getDynPrimitiveId2Var().value());
    }
}

// DynScaleCfgInvOp
struct FieldsDynScaleCfgInvOp {
    using Field_dyn_primitive_idType = NPUReg60XX::Fields::dyn_primitive_id;
    using Field_dyn_primitive_id2Type = NPUReg60XX::Fields::dyn_primitive_id2;
    using Field_dyn_scale_fetch_addr_enType = NPUReg60XX::Fields::dyn_scale_fetch_addr_en;
    using Field_dyn_scale_cfgType = NPUReg60XX::Fields::dyn_scale_cfg;
    using Field_dyn_idu_wt_num_mod_enType = NPUReg60XX::Fields::dyn_idu_wt_num_mod_en;
    using Field_dyn_odu_z_end_mod_enType = NPUReg60XX::Fields::dyn_odu_z_end_mod_en;
    using Field_dyn_idu_ic_mod_enType = NPUReg60XX::Fields::dyn_idu_ic_mod_en;
    using Field_dyn_ppe_pad_ovrwr_enType = NPUReg60XX::Fields::dyn_ppe_pad_ovrwr_en;
    using Field_dyn_ppe_dataType = NPUReg60XX::Fields::dyn_ppe_data;
    using Field_dyn_conv_cont_mod_enType = NPUReg60XX::Fields::dyn_conv_cont_mod_en;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegDynScaleCfgInvOp(VPUIPDPU::DynScaleCfgInvOp op, DpuInvariantDescriptorType& descriptor) {
    if (op.getDynPrimitiveId().has_value()) {
        // Dynamic Primitive ID is taken from the invariant descriptor
        descriptor.template write<typename Type_Fields::Field_dyn_primitive_idType>(op.getDynPrimitiveId().value());
    }
    if (op.getDynPrimitiveId2().has_value()) {
        // Dynamic Primitive ID 2 is taken from the invariant descriptor
        descriptor.template write<typename Type_Fields::Field_dyn_primitive_id2Type>(op.getDynPrimitiveId2().value());
    }
    if (op.getDynPpeData().has_value()) {
        // Dynamic PPE pad overwrite enable
        // Specifies the data to be inserted into the pad region in the output tensor
        descriptor.template write<typename Type_Fields::Field_dyn_ppe_pad_ovrwr_enType>(1);
        descriptor.template write<typename Type_Fields::Field_dyn_ppe_dataType>(op.getDynPpeData().value());
    }

    // Enables fetching of a scale factor from CMX
    descriptor.template write<typename Type_Fields::Field_dyn_scale_fetch_addr_enType>(1);

    // Defines the scalable dimension:
    // 0 = No Dynamic scaling
    // 1 = Scaling in spatial dimension only
    // 2 = Scaling in spatial dimension & output depth dimension
    // 3 = Scaling in spatial dimension & input depth dimension
    descriptor.template write<typename Type_Fields::Field_dyn_scale_cfgType>(op.getDynamicSkipMode());

    bool dynIduWTNumModEn = false, dynOduZendModEn = false, dynIduICModEn = false;
    switch (op.getDynamicAdjustMode()) {
    case VPUIP::DynamicAdjustMode::DISABLED:
        break;
    case VPUIP::DynamicAdjustMode::OUTPUT_CHANNELS:
        dynIduWTNumModEn = true;
        dynOduZendModEn = true;
        break;
    case VPUIP::DynamicAdjustMode::INPUT_CHANNELS:
        dynIduICModEn = true;
        break;
    default:
        VPUX_THROW("Found invalid case for DynamicAdjustMode");
    }

    // When the depth dimension is being scaled Enable scaling of the IDU weight count
    descriptor.template write<typename Type_Fields::Field_dyn_idu_wt_num_mod_enType>(dynIduWTNumModEn);
    // When the depth dimension is being scaled Enable scaling of the ODU workload Z dimension
    descriptor.template write<typename Type_Fields::Field_dyn_odu_z_end_mod_enType>(dynOduZendModEn);
    // When the IC dimension is being scaled enable scaling of the IDU weight size and workload size Z configuration
    descriptor.template write<typename Type_Fields::Field_dyn_idu_ic_mod_enType>(dynIduICModEn);
    // When the IC dimension is being scaled enable converting a convolution continue workload to a non-convolution
    // continue workload to allow flushing of accumulator contexts when workloads are skipped due to scaling
    descriptor.template write<typename Type_Fields::Field_dyn_conv_cont_mod_enType>(op.getDynConvContModEn());
}

template <typename Type_Fields, typename VpuInputTensorDTypeEnum, typename DpuInvariantDescriptorType>
bool lowerToRegIDUWorkloadCfgOp(VPUIPDPU::IDUWorkloadCfgOp op, DpuInvariantDescriptorType& descriptor) {
    auto workloadType = op.getWorkloadType();
    auto dtype = static_cast<uint64_t>(VpuInputTensorDTypeEnum::INPUT_DTYPE_UNKNOWN);

    if (IDUWorkloadType::ACCU_STORE_I32 == workloadType || IDUWorkloadType::ACCU_LOAD_I32 == workloadType) {
        dtype = static_cast<uint64_t>(VpuInputTensorDTypeEnum::I32);
    } else if (IDUWorkloadType::ACCU_STORE_FP32 == workloadType || IDUWorkloadType::ACCU_LOAD_FP32 == workloadType) {
        dtype = static_cast<uint64_t>(VpuInputTensorDTypeEnum::FP32);
    }

    if (dtype != static_cast<uint64_t>(VpuInputTensorDTypeEnum::INPUT_DTYPE_UNKNOWN)) {
        descriptor.template write<typename Type_Fields::Field_amodeType>(dtype);
        descriptor.template write<typename Type_Fields::Field_wmodeType>(dtype);
    }

    if (IDUWorkloadType::ACCU_STORE_I32 == workloadType || IDUWorkloadType::ACCU_STORE_FP32 == workloadType) {
        descriptor.template write<typename Type_Fields::Field_flush_wlType>(1);
        descriptor.template write<typename Type_Fields::Field_workload_operationType>(0b10);
        descriptor.template write<typename Type_Fields::Field_zm_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dw_wt_sp_insType>(0b1);
        descriptor.template write<typename Type_Fields::Field_tensor_size_xType>(16);
        descriptor.template write<typename Type_Fields::Field_tensor_size_yType>(16);
        descriptor.template write<typename Type_Fields::Field_tensor_size_zType>(32);
        descriptor.template write<typename Type_Fields::Field_kernel_xType>(1);
        descriptor.template write<typename Type_Fields::Field_kernel_yType>(1);
        descriptor.template write<typename Type_Fields::Field_ppe_modeType>(1);
        if constexpr (detail::has_Field_weight_table_formatType<Type_Fields>::value) {
            descriptor.template write<typename Type_Fields::Field_weight_table_formatType>(1);
        }

        return true;
    } else if (IDUWorkloadType::ACCU_LOAD_I32 == workloadType || IDUWorkloadType::ACCU_LOAD_FP32 == workloadType) {
        descriptor.template write<typename Type_Fields::Field_workload_operationType>(1);
        descriptor.template write<typename Type_Fields::Field_dw_wt_sp_insType>(0b1);
        descriptor.template write<typename Type_Fields::Field_zm_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_ppe_modeType>(1);

        return true;
    }

    return arch50xx::lowerToRegIDUWorkloadCfgOp<Type_Fields>(op, descriptor);
}

// IDUWeightTableOffsetsOp
struct FieldsIDUWeightTableOffsetsOp {
    using Field_weight_d_ptr_startType = NPUReg60XX::Fields::weight_d_ptr_start;
    using Field_weight_sp_ptr_startType = NPUReg60XX::Fields::weight_sp_ptr_start;
    using Field_weight_startType = NPUReg60XX::Fields::weight_start;
};

template <typename Type_Fields, typename DpuVariantDescriptorType>
void lowerToRegIDUWeightTableOffsetsOp(VPUIPDPU::IDUWeightTableOffsetsOp op, DpuVariantDescriptorType& descriptor) {
    constexpr uint32_t SCALE_OFF = 16;
    constexpr uint32_t SCALE_MSK = 0xFFFF0000;
    constexpr uint32_t BIAS_OFF = 0;
    constexpr uint32_t BIAS_MSK = 0x0000FFFF;

    // these address registers will be modified by relocation mechanism based on provided offset info
    if (auto dataPtrZpOffset = op.getDataPtrZpOffset()) {
        descriptor.template write<typename Type_Fields::Field_weight_d_ptr_startType>(dataPtrZpOffset.value());
    }
    if (auto spPtrZpOffset = op.getSpPtrOffset()) {
        descriptor.template write<typename Type_Fields::Field_weight_sp_ptr_startType>(spPtrZpOffset.value());
    }
    auto scaleOffset = op.getScaleOffset();
    auto biasOffset = op.getBiasOffset();
    if (scaleOffset.has_value() || biasOffset.has_value()) {
        auto weightStart = (scaleOffset.value_or(0) << SCALE_OFF) & SCALE_MSK;
        weightStart |= (biasOffset.value_or(0) << BIAS_OFF) & BIAS_MSK;
        descriptor.template write<typename Type_Fields::Field_weight_startType>(weightStart);
    }
}

}  // namespace vpux::VPUIPDPU::arch60xx
