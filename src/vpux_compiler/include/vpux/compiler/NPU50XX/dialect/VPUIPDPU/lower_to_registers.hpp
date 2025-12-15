//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/lower_to_registers.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/types.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

using namespace vpux::VPURegMapped;
using namespace npu40xx;

// Implementations of the lowering function that do not change between architectures:
namespace vpux::VPUIPDPU {

// MPEBf16ModeOp
template <typename Field_mpe_modeType, typename DpuInvariantDescriptorType>
void lowerToRegMPEBf16ModeOp(DpuInvariantDescriptorType& descriptor) {
    const auto mpeBF16Mode = 5;
    descriptor.template write<Field_mpe_modeType>(mpeBF16Mode);
}

// PPEsprLUTReadOp
template <typename Field_invar_lut_rd_enType, typename DpuVariantDescriptorType>
void lowerToRegPPEsprLUTReadOp(DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_invar_lut_rd_enType>(1);
}

}  // namespace vpux::VPUIPDPU

namespace vpux::VPUIPDPU::arch50xx {

// IDUStorageElementOp
struct FieldsIDUStorageElementOp {
    using Field_se_z_splitType = NPUReg50XX::Fields::se_z_split;
    using Field_npo2_se_z_split_enType = NPUReg50XX::Fields::npo2_se_z_split_en;
    using Field_npo2_se_sizeType = NPUReg50XX::Fields::npo2_se_size;
    using Field_num_ses_in_z_dirType = NPUReg50XX::Fields::num_ses_in_z_dir;
};

// IDUKernelOp
struct FieldsIDUKernelOp {
    using Field_kernel_xType = NPUReg50XX::Fields::kernel_x;
    using Field_kernel_yType = NPUReg50XX::Fields::kernel_y;
};

// IDUStrideOp
struct FieldsIDUStrideOp {
    using Field_strideType = NPUReg50XX::Fields::stride;
    using Field_stride_yType = NPUReg50XX::Fields::stride_y;
    using Field_stride_y_enType = NPUReg50XX::Fields::stride_y_en;
};

// IDUInActivationsOp
struct FieldsIDUInActivationsOp {
    using Field_tensor_size_xType = NPUReg50XX::Fields::tensor_size_x;
    using Field_tensor_size_yType = NPUReg50XX::Fields::tensor_size_y;
    using Field_tensor_size_zType = NPUReg50XX::Fields::tensor_size_z;
    using Field_amodeType = NPUReg50XX::Fields::amode;
    using Field_act_denseType = NPUReg50XX::Fields::act_dense;
    using Field_tensor_cmp_sizeType = NPUReg50XX::Fields::tensor_cmp_size;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegIDUInActivationsOp(VPUIPDPU::IDUInActivationsOp op, DpuInvariantDescriptorType& descriptor) {
    VPUIPDPU::arch40xx::lowerToRegIDUInActivationsOp<Type_Fields>(op, descriptor);
    auto inActivations = op.getInActivations();
    auto inActivationsType = mlir::cast<vpux::NDTypeInterface>(inActivations.getType()).getElementType();
    auto inActivationShape = getShape(inActivations);
    if (mlir::dyn_cast<mlir::quant::QuantizedType>(inActivationsType)) {
        inActivationsType = mlir::quant::QuantizedType::castToStorageType(inActivationsType);
    }
    const auto padAuto16bit =
            (inActivationsType.getIntOrFloatBitWidth() == 16) && inActivationShape[Dims4D::Act::C] < 10;

    const auto padAuto8bit = (inActivationsType.getIntOrFloatBitWidth() == 8) && inActivationShape[Dims4D::Act::C] < 16;

    if (padAuto16bit || padAuto8bit) {
        descriptor.template write<typename Type_Fields::Field_tensor_cmp_sizeType>(inActivationShape[Dims4D::Act::C]);
    }
}

// IDUInputLayerCfgOp
struct FieldsIDUInputLayerCfgOp {
    using Field_cm_sp_patternType = NPUReg50XX::Fields::cm_sp_pattern;
    using Field_act_denseType = NPUReg50XX::Fields::act_dense;
    using Field_wt_denseType = NPUReg50XX::Fields::wt_dense;
    using Field_layer1_wt_sp_insType = NPUReg50XX::Fields::layer1_wt_sp_ins;
    using Field_layer1_cmp_enType = NPUReg50XX::Fields::layer1_cmp_en;
    using Field_tensor_size_zType = NPUReg50XX::Fields::tensor_size_z;
};

// IDUWeightsOp
struct FieldsIDUWeightsOp {
    using Field_wmodeType = NPUReg50XX::Fields::wmode;
    using Field_wt_denseType = NPUReg50XX::Fields::wt_dense;
    using Field_wt_plt_cfgType = NPUReg50XX::Fields::wt_plt_cfg;
    using Field_pool_wt_dataType = NPUReg50XX::Fields::pool_wt_data;

    using Field_plt_idx_0Type = NPUReg50XX::Fields::plt_idx_0;
    using Field_plt_idx_1Type = NPUReg50XX::Fields::plt_idx_1;
    using Field_plt_idx_2Type = NPUReg50XX::Fields::plt_idx_2;
    using Field_plt_idx_3Type = NPUReg50XX::Fields::plt_idx_3;
    using Field_plt_idx_4Type = NPUReg50XX::Fields::plt_idx_4;
    using Field_plt_idx_5Type = NPUReg50XX::Fields::plt_idx_5;
    using Field_plt_idx_6Type = NPUReg50XX::Fields::plt_idx_6;
    using Field_plt_idx_7Type = NPUReg50XX::Fields::plt_idx_7;
    using Field_plt_idx_8Type = NPUReg50XX::Fields::plt_idx_8;
    using Field_plt_idx_9Type = NPUReg50XX::Fields::plt_idx_9;
    using Field_plt_idx_10Type = NPUReg50XX::Fields::plt_idx_10;
    using Field_plt_idx_11Type = NPUReg50XX::Fields::plt_idx_11;
    using Field_plt_idx_12Type = NPUReg50XX::Fields::plt_idx_12;
    using Field_plt_idx_13Type = NPUReg50XX::Fields::plt_idx_13;
    using Field_plt_idx_14Type = NPUReg50XX::Fields::plt_idx_14;
    using Field_plt_idx_15Type = NPUReg50XX::Fields::plt_idx_15;
};

// IDUWorkloadCfgOp
struct FieldsIDUWorkloadCfgOp {
    using Field_workload_operationType = NPUReg50XX::Fields::workload_operation;
    using Field_zm_inputType = NPUReg50XX::Fields::zm_input;
    using Field_dw_inputType = NPUReg50XX::Fields::dw_input;
    using Field_pool_wt_rd_disType = NPUReg50XX::Fields::pool_wt_rd_dis;
    using Field_dw_wt_sp_insType = NPUReg50XX::Fields::dw_wt_sp_ins;
    using Field_dynamic_bw_enType = NPUReg50XX::Fields::dynamic_bw_en;
    using Field_elop_wloadType = NPUReg50XX::Fields::elop_wload;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
bool lowerToRegIDUWorkloadCfgOp(VPUIPDPU::IDUWorkloadCfgOp op, DpuInvariantDescriptorType& descriptor) {
    bool successfullyLowered = vpux::VPUIPDPU::arch40xx::lowerToRegIDUWorkloadCfgOp<Type_Fields>(op, descriptor);
    if (successfullyLowered) {
        return true;
    }
    successfullyLowered = true;
    auto workloadType = op.getWorkloadType();
    switch (workloadType) {
    case VPUIPDPU::IDUWorkloadType::REDUCEMEAN:
        descriptor.template write<typename Type_Fields::Field_workload_operationType>(0);
        descriptor.template write<typename Type_Fields::Field_zm_inputType>(0b1);  // CONV workload
        descriptor.template write<typename Type_Fields::Field_pool_wt_rd_disType>(
                0b1);  // REDUCEMEAN reuses AVEPOOL mode for weights
        descriptor.template write<typename Type_Fields::Field_dynamic_bw_enType>(1);
        break;
    case VPUIPDPU::IDUWorkloadType::REDUCESUMSQUARE:
        //  Sum of Squares workload - redirects the weights read;
        //  to read instead the same data as the input; hence squaring the values
        descriptor.template write<typename Type_Fields::Field_workload_operationType>(0b11);
        descriptor.template write<typename Type_Fields::Field_zm_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_pool_wt_rd_disType>(
                0b1);  // REDUCESUMSQUARE reuses AVEPOOL mode for weights
        descriptor.template write<typename Type_Fields::Field_dynamic_bw_enType>(1);
        break;
    case VPUIPDPU::IDUWorkloadType::REDUCESUM:
        descriptor.template write<typename Type_Fields::Field_workload_operationType>(0);
        descriptor.template write<typename Type_Fields::Field_zm_inputType>(0b1);  // CONV workload
        descriptor.template write<typename Type_Fields::Field_pool_wt_rd_disType>(
                0b1);  // REDUCESUM reuses AVEPOOL mode for weights
        descriptor.template write<typename Type_Fields::Field_dynamic_bw_enType>(1);
        break;
    default:
        successfullyLowered = false;
        break;
    }

    return successfullyLowered;
}

// IDUDepthWiseCfgOp
struct FieldsIDUDepthWiseCfgOp {
    using Field_dw_3x3s1_opt_disType = NPUReg50XX::Fields::dw_3x3s1_opt_dis;
    using Field_dw_opt_enType = NPUReg50XX::Fields::dw_opt_en;
    using Field_dw_opt_offsetType = NPUReg50XX::Fields::dw_opt_offset;
    using Field_pool_opt_enType = NPUReg50XX::Fields::pool_opt_en;
};

// IDUEltWiseCfgOp
// contains new table-gen generation Field definitions
struct NewTableGenFieldsIDUEltWiseCfgOp {
    using Field_elop_scale_a_bfpType = NPUReg50XX::Fields::elop_scale_a_bfp;
    using Field_elop_scale_b_bfpType = NPUReg50XX::Fields::elop_scale_b_bfp;
    using Field_elop_scale_a_fpType = NPUReg50XX::Fields::elop_scale_a_fp;
    using Field_elop_scale_b_fpType = NPUReg50XX::Fields::elop_scale_b_fp;
    using Field_elop_scale_aType = NPUReg50XX::Fields::elop_scale_a;
    using Field_elop_scale_bType = NPUReg50XX::Fields::elop_scale_b;
};

// contains old table-gen generation Field definitions
struct OldTableGenFieldsIDUEltWiseCfgOp {
    using Field_elop_scale_a_bfpType = NPUReg50XX::RegField_elop_scale_a_bfpType;
    using Field_elop_scale_b_bfpType = NPUReg50XX::RegField_elop_scale_b_bfpType;
    using Field_elop_scale_a_fpType = NPUReg50XX::RegField_elop_scale_a_fpType;
    using Field_elop_scale_b_fpType = NPUReg50XX::RegField_elop_scale_b_fpType;
    using Field_elop_scale_aType = NPUReg50XX::RegField_elop_scale_aType;
    using Field_elop_scale_bType = NPUReg50XX::RegField_elop_scale_bType;
};

template <typename NewTableGen_Type_Fields, typename OldTableGen_Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegIDUEltWiseCfgOp(VPUIPDPU::IDUEltWiseCfgOp op, DpuInvariantDescriptorType& descriptor) {
    auto elopScaleAFpAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(op.getElopScaleAAttr());
    auto elopScaleBFpAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(op.getElopScaleBAttr());
    if (elopScaleAFpAttr && elopScaleBFpAttr) {
        if (op.getBf16FlowOn()) {
            auto elopScaleAFp = checked_cast_reg<typename OldTableGen_Type_Fields::Field_elop_scale_a_bfpType>(
                    elopScaleAFpAttr.getValueAsDouble());
            auto elopScaleBFp = checked_cast_reg<typename OldTableGen_Type_Fields::Field_elop_scale_b_bfpType>(
                    elopScaleBFpAttr.getValueAsDouble());
            descriptor.template write<typename NewTableGen_Type_Fields::Field_elop_scale_a_bfpType>(elopScaleAFp);
            descriptor.template write<typename NewTableGen_Type_Fields::Field_elop_scale_b_bfpType>(elopScaleBFp);
        } else {
            auto elopScaleAFp = checked_cast_reg<typename OldTableGen_Type_Fields::Field_elop_scale_a_fpType>(
                    elopScaleAFpAttr.getValueAsDouble());
            auto elopScaleBFp = checked_cast_reg<typename OldTableGen_Type_Fields::Field_elop_scale_b_fpType>(
                    elopScaleBFpAttr.getValueAsDouble());
            descriptor.template write<typename NewTableGen_Type_Fields::Field_elop_scale_a_fpType>(elopScaleAFp);
            descriptor.template write<typename NewTableGen_Type_Fields::Field_elop_scale_b_fpType>(elopScaleBFp);
        }
    } else {
        vpux::VPUIPDPU::arch40xx::lowerToRegIDUEltWiseCfgOp<NewTableGen_Type_Fields>(op, descriptor);
    }
}

template <typename Field_elop_operationType, typename DpuInvariantDescriptorType>
void lowerToRegIDUEltWiseModeOp(VPUIPDPU::IDUEltWiseModeOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_elop_operationType>(op.getEltwiseType());
}

// PPEFpBiasAddOp
struct FieldsPPEFpBiasAddOp {
    using Field_ppe_fp_scale_overrideType = NPUReg50XX::Fields::ppe_fp_scale_override;
    using Field_ppe_fp_biasType = NPUReg50XX::Fields::ppe_fp_bias;
};

// PPEFpScalePreluMultOp
struct FieldsPPEFpScalePreluMultOp {
    using Field_ppe_fp_scale_overrideType = NPUReg50XX::Fields::ppe_fp_scale_override;
    using Field_ppe_fp_scaleType = NPUReg50XX::Fields::ppe_fp_scale;
    using Field_ppe_fp_prelu_enType = NPUReg50XX::Fields::ppe_fp_prelu_en;
    using Field_ppe_fp_preluType = NPUReg50XX::Fields::ppe_fp_prelu;
};

// PPEFpConvertOp
struct FieldsPPEFpConvertOp {
    using Field_ppe_fp_convertType = NPUReg50XX::Fields::ppe_fp_convert;
    using Field_ppe_fp16_clampType = NPUReg50XX::Fields::ppe_fp16_clamp;
    using Field_ppe_fp16_ftzType = NPUReg50XX::Fields::ppe_fp16_ftz;
    using Field_ppe_bf16_roundType = NPUReg50XX::Fields::ppe_bf16_round;
};

// PPEIntBiasAddOp
struct FieldsPPEIntBiasAddOp {
    using Field_ppe_scale_overrideType = NPUReg50XX::Fields::ppe_scale_override;
    using Field_ppe_biasType = NPUReg50XX::Fields::ppe_bias;
};

// PPEIntScaleMultOp
struct FieldsPPEIntScaleMultOp {
    using Field_ppe_scale_overrideType = NPUReg50XX::Fields::ppe_scale_override;
    using Field_ppe_scale_multType = NPUReg50XX::Fields::ppe_scale_mult;
};

// PPEIntScaleShiftOp
struct FieldsPPEIntScaleShiftOp {
    using Field_ppe_scale_overrideType = NPUReg50XX::Fields::ppe_scale_override;
    using Field_ppe_scale_shiftType = NPUReg50XX::Fields::ppe_scale_shift;
};

// PPEIntClampOp
struct FieldsPPEIntClampOp {
    using Field_ppe_int_scale_hclampType = NPUReg50XX::Fields::ppe_int_scale_hclamp;
    using Field_ppe_int_scale_lclampType = NPUReg50XX::Fields::ppe_int_scale_lclamp;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEIntClampOp(VPUIPDPU::PPEIntClampOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_ppe_int_scale_hclampType>(op.getClampHigh());
    if (op.getClampLow().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_int_scale_lclampType>(op.getClampLow().value());
    }
}

// PPEFpScaleMultOp
struct FieldsPPEFpScaleMultOp {
    using Field_ppe_fp_scale_overrideType = NPUReg50XX::Fields::ppe_fp_scale_override;
    using Field_ppe_fp_scaleType = NPUReg50XX::Fields::ppe_fp_scale;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEFpScaleMultOp(VPUIPDPU::PPEFpScaleMultOp op, DpuInvariantDescriptorType& descriptor) {
    VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getScaleStatic().has_value(),
                      "op {0} has ambiguous parameters", op);
    if (op.getScaleTable()) {
        descriptor.template write<typename Type_Fields::Field_ppe_fp_scale_overrideType>(0);
    }
    if (op.getScaleStatic().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_fp_scale_overrideType>(1);
        descriptor.template write<typename Type_Fields::Field_ppe_fp_scaleType>(
                op.getScaleStatic().value().convertToFloat());
    }
}

// PPEFpSprLUTModeOp
struct FieldsPPEFpSprLUTModeOp {
    using Field_ppe_modeType = NPUReg50XX::Fields::ppe_mode;
    using Field_ppe_sb_dtypeType = NPUReg50XX::Fields::ppe_sb_dtype;
    using Field_ppe_lut_ptr_forceType = NPUReg50XX::Fields::ppe_lut_ptr_force;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEFpSprLUTModeOp(VPUIPDPU::PPEFpSprLUTModeOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_ppe_modeType>(op.getSprlutMode());
    descriptor.template write<typename Type_Fields::Field_ppe_sb_dtypeType>(2);
    descriptor.template write<typename Type_Fields::Field_ppe_lut_ptr_forceType>(op.getSprlutMode() ==
                                                                                 PPEsprLUTMode::ON);
}

// PPEFpPreluMultOp
struct FieldsPPEFpPreluMultOp {
    using Field_ppe_fp_prelu_enType = NPUReg50XX::Fields::ppe_fp_prelu_en;
    using Field_ppe_fp_preluType = NPUReg50XX::Fields::ppe_fp_prelu;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEFpPreluMultOp(VPUIPDPU::PPEFpPreluMultOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_ppe_fp_prelu_enType>(1);
    descriptor.template write<typename Type_Fields::Field_ppe_fp_preluType>(op.getPreluAlpha().convertToFloat());
}

// PPEFpClampOp
struct FieldsPPEFpClampOp {
    using Field_ppe_fp_scale_hclampType = NPUReg50XX::Fields::ppe_fp_scale_hclamp;
    using Field_ppe_fp_scale_lclampType = NPUReg50XX::Fields::ppe_fp_scale_lclamp;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEFpClampOp(VPUIPDPU::PPEFpClampOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_ppe_fp_scale_hclampType>(op.getClampHigh().convertToFloat());
    descriptor.template write<typename Type_Fields::Field_ppe_fp_scale_lclampType>(op.getClampLow().convertToFloat());
}

// ODUOutTensorSizeOp
struct FieldsODUOutTensorSizeOp {
    using Field_te_dim_xType = NPUReg50XX::Fields::te_dim_x;
    using Field_te_dim_yType = NPUReg50XX::Fields::te_dim_y;
    using Field_te_dim_zType = NPUReg50XX::Fields::te_dim_z;
};

// ODUSparsityOp
struct FieldsODUSparsityOp {
    using Field_sp_valueType = NPUReg50XX::Fields::sp_value;
    using Field_sp_out_enType = NPUReg50XX::Fields::sp_out_en;
    using Field_write_spType = NPUReg50XX::Fields::write_sp;
};

// ODUOutActivationsOp
struct FieldsODUOutActivationsOp {
    using Field_dtypeType = NPUReg50XX::Fields::dtype;
    using Field_write_acType = NPUReg50XX::Fields::write_ac;
};

// ODUWriteCombineBufferOp
struct FieldsODUWriteCombineBufferOp {
    using Field_wcb_bypassType = NPUReg50XX::Fields::wcb_bypass;
    using Field_wcb_ac_modeType = NPUReg50XX::Fields::wcb_ac_mode;
    using Field_wcb_sp_modeType = NPUReg50XX::Fields::wcb_sp_mode;
};

// InvariantBarrierCfg
struct RegistersInvariantBarrierCfg {
    using Register_barriers_sched_Type = NPUReg50XX::Registers::barriers_sched_;
};

struct FieldsInvariantBarrierCfg {
    using Field_group_Type = NPUReg50XX::Fields::group_;
    using Field_mask_Type = NPUReg50XX::Fields::mask_;
    using Field_start_after_Type = NPUReg50XX::Fields::start_after_;
    using Field_clean_after_Type = NPUReg50XX::Fields::clean_after_;
    using Field_barriers_wait_mask_hi_Type = NPUReg50XX::Fields::barriers_wait_mask_hi_;
    using Field_barriers_wait_mask_lo_Type = NPUReg50XX::Fields::barriers_wait_mask_lo_;
    using Field_barriers_post_mask_hi_Type = NPUReg50XX::Fields::barriers_post_mask_hi_;
    using Field_barriers_post_mask_lo_Type = NPUReg50XX::Fields::barriers_post_mask_lo_;
};

// IDUWorkloadSetOp
struct FieldsIDUWorkloadSetOp {
    using Field_workload_start_xType = NPUReg50XX::Fields::workload_start_x;
    using Field_workload_start_yType = NPUReg50XX::Fields::workload_start_y;
    using Field_workload_start_zType = NPUReg50XX::Fields::workload_start_z;
    using Field_workload_size_xType = NPUReg50XX::Fields::workload_size_x;
    using Field_workload_size_yType = NPUReg50XX::Fields::workload_size_y;
    using Field_workload_size_zType = NPUReg50XX::Fields::workload_size_z;
};

// IDUPaddingOp
struct FieldsIDUPaddingOp {
    using Field_pad_count_upType = NPUReg50XX::Fields::pad_count_up;
    using Field_pad_count_leftType = NPUReg50XX::Fields::pad_count_left;
    using Field_pad_count_downType = NPUReg50XX::Fields::pad_count_down;
    using Field_pad_count_rightType = NPUReg50XX::Fields::pad_count_right;
};

// IDUWeightSetOp
struct FieldsIDUWeightSetOp {
    using Field_weight_sizeType = NPUReg50XX::Fields::weight_size;
    using Field_weight_numType = NPUReg50XX::Fields::weight_num;
    using Field_weight_startType = NPUReg50XX::Fields::weight_start;
};

// IDUSEOnlyOp
template <typename Field_se_only_enType, typename DpuVariantDescriptorType>
void lowerToRegIDUSEOnlyOp(DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_se_only_enType>(1);
}

// IDUPerOutputChannelScalingOp
struct FieldsIDUPerOutputChannelScalingOp {
    using Field_sb_read_enType = NPUReg50XX::Fields::sb_read_en;
    using Field_tensor2_act_denseType = NPUReg50XX::Fields::tensor2_act_dense;
};

template <typename Type_Fields, typename DpuVariantDescriptorType>
void lowerToRegIDUPerOutputChannelScalingOp(VPUIPDPU::IDUPerOutputChannelScalingOp op,
                                            DpuVariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_sb_read_enType>(1);
    descriptor.template write<typename Type_Fields::Field_tensor2_act_denseType>(!op.getTensor2ActSparse());
}

// ODUOutSubtensorOp
struct FieldsODUOutSubtensorOp {
    using Field_te_beg_xType = NPUReg50XX::Fields::te_beg_x;
    using Field_te_beg_yType = NPUReg50XX::Fields::te_beg_y;
    using Field_te_beg_zType = NPUReg50XX::Fields::te_beg_z;
    using Field_te_end_xType = NPUReg50XX::Fields::te_end_x;
    using Field_te_end_yType = NPUReg50XX::Fields::te_end_y;
    using Field_te_end_zType = NPUReg50XX::Fields::te_end_z;
};

// ODUHaloCfgOp
struct RegistersODUHaloCfgOp {
    using Register_halo_region0AType = NPUReg50XX::Registers::halo_region0A;
    using Register_halo_region0BType = NPUReg50XX::Registers::halo_region0B;
    using Register_halo_region0CType = NPUReg50XX::Registers::halo_region0C;
    using Register_halo_region0DType = NPUReg50XX::Registers::halo_region0D;

    using Register_halo_region1AType = NPUReg50XX::Registers::halo_region1A;
    using Register_halo_region1BType = NPUReg50XX::Registers::halo_region1B;
    using Register_halo_region1CType = NPUReg50XX::Registers::halo_region1C;
    using Register_halo_region1DType = NPUReg50XX::Registers::halo_region1D;

    using Register_halo_region2AType = NPUReg50XX::Registers::halo_region2A;
    using Register_halo_region2BType = NPUReg50XX::Registers::halo_region2B;
    using Register_halo_region2CType = NPUReg50XX::Registers::halo_region2C;
    using Register_halo_region2DType = NPUReg50XX::Registers::halo_region2D;

    using Register_halo_region3AType = NPUReg50XX::Registers::halo_region3A;
    using Register_halo_region3BType = NPUReg50XX::Registers::halo_region3B;
    using Register_halo_region3CType = NPUReg50XX::Registers::halo_region3C;
    using Register_halo_region3DType = NPUReg50XX::Registers::halo_region3D;

    using Register_halo_region4AType = NPUReg50XX::Registers::halo_region4A;
    using Register_halo_region4BType = NPUReg50XX::Registers::halo_region4B;
    using Register_halo_region4CType = NPUReg50XX::Registers::halo_region4C;
    using Register_halo_region4DType = NPUReg50XX::Registers::halo_region4D;

    using Register_halo_region5AType = NPUReg50XX::Registers::halo_region5A;
    using Register_halo_region5BType = NPUReg50XX::Registers::halo_region5B;
    using Register_halo_region5CType = NPUReg50XX::Registers::halo_region5C;
    using Register_halo_region5DType = NPUReg50XX::Registers::halo_region5D;
};

struct FieldsODUHaloCfgOp {
    using Field_begin_xType = NPUReg50XX::Fields::begin_x;
    using Field_begin_yType = NPUReg50XX::Fields::begin_y;
    using Field_end_xType = NPUReg50XX::Fields::end_x;
    using Field_ac_adr_offsetType = NPUReg50XX::Fields::ac_adr_offset;
    using Field_target_width_lsbType = NPUReg50XX::Fields::target_width_lsb;
    using Field_target_width_msbType = NPUReg50XX::Fields::target_width_msb;
    using Field_tile_selectType = NPUReg50XX::Fields::tile_select;
    using Field_sp_adr_offsetType = NPUReg50XX::Fields::sp_adr_offset;
    using Field_enableType = NPUReg50XX::Fields::enable;
    using Field_end_yType = NPUReg50XX::Fields::end_y;
};

struct FunctionsODUHaloCfgOp {
    using Function_target_width_lsbType = NPUReg50XX::RegField_target_width_lsbType;
    using Function_target_width_msbType = NPUReg50XX::RegField_target_width_msbType;
};

// VariantBarrierCfg
struct FieldsVariantBarrierCfg {
    using Field_cbarrier_hiType = NPUReg50XX::Fields::cbarrier_hi;
    using Field_cbarrier_loType = NPUReg50XX::Fields::cbarrier_lo;
    using Field_pbarrier_hiType = NPUReg50XX::Fields::pbarrier_hi;
    using Field_pbarrier_loType = NPUReg50XX::Fields::pbarrier_lo;
};

}  // namespace vpux::VPUIPDPU::arch50xx
