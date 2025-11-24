//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/descriptors.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux::VPURegMapped;
using namespace npu40xx;

// Implementations of the lowering function that do not change between architectures:
namespace vpux::VPUIPDPU {

// MPEDenormalOperandsFTZOp
template <typename Field_mpe_dazType, typename DpuInvariantDescriptorType>
void lowerToRegMPEDenormalOperandsFTZOp(DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_mpe_dazType>(1);
}

// IDUSEDenseOp
template <typename Field_dense_seType, typename DpuVariantDescriptorType>
void lowerToRegIDUSEDenseOp(DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_dense_seType>(1);
}

// IDUConvContinueOp
template <typename Field_conv_condType, typename DpuVariantDescriptorType>
void lowerToRegIDUConvContinueOp(DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_conv_condType>(1);
}

// IDUBinaryConfigOp
template <typename Field_bin_cfgType, typename DpuVariantDescriptorType>
void lowerToRegIDUBinaryConfigOp(DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_bin_cfgType>(1);
}

// ForceInvReadOp
template <typename Field_invar_lptr_forceType, typename DpuVariantDescriptorType>
void lowerToRegForceInvReadOp(DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_invar_lptr_forceType>(1);
}

}  // namespace vpux::VPUIPDPU

namespace vpux::VPUIPDPU::arch40xx {

template <typename REG_TYPE>
uint64_t getTensorMode(mlir::Type type) {
    static_assert(std::is_same<REG_TYPE, NPUReg40XX::RegField_amodeType>::value ||
                          std::is_same<REG_TYPE, NPUReg40XX::RegField_wmodeType>::value ||
                          std::is_same<REG_TYPE, NPUReg40XX::RegField_dma_acc_info_compress_dtypeType>::value ||
                          std::is_same<REG_TYPE, NPUReg40XX::RegField_dma_acc_info_decompress_dtypeType>::value,
                  "getTensorMode: Unsupported template argument REG_TYPE");

    if (auto quantized = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
        return getTensorMode<REG_TYPE>(quantized.getStorageType());
    }
    if (std::is_same<REG_TYPE, NPUReg40XX::RegField_amodeType>::value ||
        std::is_same<REG_TYPE, NPUReg40XX::RegField_wmodeType>::value) {
        if (type.isF16()) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::FP16);
        } else if (type.isUnsignedInteger(8)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::U8);
        } else if (type.isInteger(8)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::I8);
        } else if (type.isSignedInteger(4)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::I4);
        } else if (type.isInteger(2)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::I2);
        } else if (type.isBF16()) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::BF16);
        } else if (type.isUnsignedInteger(4)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::U4);
        } else if (type.isInteger(1)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::BIN);
        } else if (mlir::isa<mlir::Float8E5M2Type>(type)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::FP8);
        }
        VPUX_THROW("Invalid tensor type for DPU configuration {0}", type);
    } else if (std::is_same<REG_TYPE, NPUReg40XX::RegField_dma_acc_info_compress_dtypeType>::value ||
               std::is_same<REG_TYPE, NPUReg40XX::RegField_dma_acc_info_decompress_dtypeType>::value) {
        if (type.isUnsignedInteger(8) || type.isInteger(8) || type.isUnsignedInteger(4) || type.isInteger(4)) {
            return DMA_ACC_DTYPE_INT8_UINT8;
        } else if (type.isBF16() || type.isF16()) {
            return DMA_ACC_DTYPE_FP16_BF16;
        }
        VPUX_THROW("Invalid tensor type for DMA Acceleration configuration {0}", type);
    }
}

// IDUStorageElementOp
struct FieldsIDUStorageElementOp {
    using Field_se_z_splitType = NPUReg40XX::Fields::se_z_split;
    using Field_npo2_se_z_split_enType = NPUReg40XX::Fields::npo2_se_z_split_en;
    using Field_npo2_se_sizeType = NPUReg40XX::Fields::npo2_se_size;
    using Field_num_ses_in_z_dirType = NPUReg40XX::Fields::num_ses_in_z_dir;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegIDUStorageElementOp(VPUIPDPU::IDUStorageElementOp op, DpuInvariantDescriptorType& descriptor) {
    uint32_t seSize = checked_cast<uint32_t>(op.getSeSize());

    // TODO: refactor the code in the if-else below (properly define hard-coded values) - E#82002
    if ((seSize != 0) && ((seSize & (seSize - 1)) == 0)) {  // seSize power of 2
        uint32_t seSizeHW = 0;

        // adjust SE size to HW supported limits
        if (seSize < 16) {
            seSize = 16;
        } else if (seSize > 8192) {
            // storage_element_size bigger than 8192, HW value adjusted for 8192;
            seSize = 8192;
        }

        while (seSize >>= 1) {
            ++seSizeHW;
        }  // seSizeHW = log2(seSize)

        // HW register NCE_DPU_Z_CONFIG.se_z_split has values: 1=16, 2=32....9=4096, 0=8192
        if (seSizeHW >= 3 && seSizeHW < 13) {
            seSizeHW -= 3;
        } else {
            seSizeHW = 0;
        }
        descriptor.template write<typename Type_Fields::Field_se_z_splitType>(seSizeHW);
    } else {
        auto seSizeHW = ((seSize + 15) >> 4) - 1;
        descriptor.template write<typename Type_Fields::Field_npo2_se_z_split_enType>(1);
        descriptor.template write<typename Type_Fields::Field_npo2_se_sizeType>(seSizeHW);
    }

    if (op.getNumSesInZDir().has_value()) {
        descriptor.template write<typename Type_Fields::Field_num_ses_in_z_dirType>(op.getNumSesInZDir().value());
    }
}

// IDUKernelOp
struct FieldsIDUKernelOp {
    using Field_kernel_xType = NPUReg40XX::Fields::kernel_x;
    using Field_kernel_yType = NPUReg40XX::Fields::kernel_y;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegIDUKernelOp(VPUIPDPU::IDUKernelOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_kernel_xType>(op.getKernelX());
    descriptor.template write<typename Type_Fields::Field_kernel_yType>(op.getKernelY());
}

// IDUStrideOp
struct FieldsIDUStrideOp {
    using Field_strideType = NPUReg40XX::Fields::stride;
    using Field_stride_yType = NPUReg40XX::Fields::stride_y;
    using Field_stride_y_enType = NPUReg40XX::Fields::stride_y_en;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegIDUStrideOp(VPUIPDPU::IDUStrideOp op, DpuInvariantDescriptorType& descriptor) {
    auto strideX = op.getStrideX() - 1;
    auto strideY = op.getStrideY() - 1;
    descriptor.template write<typename Type_Fields::Field_strideType>(strideX);
    if (op.getStrideY() != op.getStrideX()) {
        descriptor.template write<typename Type_Fields::Field_stride_yType>(strideY);
        descriptor.template write<typename Type_Fields::Field_stride_y_enType>(1);
    }
}

// IDUInActivationsOp
struct FieldsIDUInActivationsOp {
    using Field_tensor_size_xType = NPUReg40XX::Fields::tensor_size_x;
    using Field_tensor_size_yType = NPUReg40XX::Fields::tensor_size_y;
    using Field_tensor_size_zType = NPUReg40XX::Fields::tensor_size_z;
    using Field_amodeType = NPUReg40XX::Fields::amode;
    using Field_act_denseType = NPUReg40XX::Fields::act_dense;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegIDUInActivationsOp(VPUIPDPU::IDUInActivationsOp op, DpuInvariantDescriptorType& descriptor) {
    auto inActivations = op.getInActivations();
    auto inActivationsType = mlir::cast<vpux::NDTypeInterface>(inActivations.getType()).getElementType();
    auto inActivationShape = getShape(inActivations);
    const auto dimY = inActivationShape[Dims4D::Act::H];
    const auto dimX = inActivationShape[Dims4D::Act::W];
    const auto dimZ = inActivationShape[Dims4D::Act::C];
    auto tensorMode = getTensorMode<NPUReg40XX::RegField_amodeType>(inActivationsType);
    auto actDense = !op.getInSparse();

    descriptor.template write<typename Type_Fields::Field_tensor_size_xType>(dimX);
    descriptor.template write<typename Type_Fields::Field_tensor_size_yType>(dimY);
    descriptor.template write<typename Type_Fields::Field_tensor_size_zType>(dimZ);
    descriptor.template write<typename Type_Fields::Field_amodeType>(tensorMode);
    descriptor.template write<typename Type_Fields::Field_act_denseType>(actDense);
}

// IDUInputLayerCfgOp
struct FieldsIDUInputLayerCfgOp {
    using Field_cm_sp_patternType = NPUReg40XX::Fields::cm_sp_pattern;
    using Field_act_denseType = NPUReg40XX::Fields::act_dense;
    using Field_wt_denseType = NPUReg40XX::Fields::wt_dense;
    using Field_layer1_wt_sp_insType = NPUReg40XX::Fields::layer1_wt_sp_ins;
    using Field_layer1_cmp_enType = NPUReg40XX::Fields::layer1_cmp_en;
    using Field_tensor_size_zType = NPUReg40XX::Fields::tensor_size_z;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegIDUInputLayerCfgOp(VPUIPDPU::IDUInputLayerCfgOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_cm_sp_patternType>(op.getSparsityPattern());
    descriptor.template write<typename Type_Fields::Field_act_denseType>(1);
    descriptor.template write<typename Type_Fields::Field_wt_denseType>(1);
    descriptor.template write<typename Type_Fields::Field_layer1_wt_sp_insType>(1);
    descriptor.template write<typename Type_Fields::Field_layer1_cmp_enType>(op.getInputCompressed());
    descriptor.template write<typename Type_Fields::Field_tensor_size_zType>(16);
}

// IDUWeightsOp
struct FieldsIDUWeightsOp {
    using Field_wmodeType = NPUReg40XX::Fields::wmode;
    using Field_wt_denseType = NPUReg40XX::Fields::wt_dense;
    using Field_wt_plt_cfgType = NPUReg40XX::Fields::wt_plt_cfg;
    using Field_pool_wt_dataType = NPUReg40XX::Fields::pool_wt_data;

    using Field_plt_idx_0Type = NPUReg40XX::Fields::plt_idx_0;
    using Field_plt_idx_1Type = NPUReg40XX::Fields::plt_idx_1;
    using Field_plt_idx_2Type = NPUReg40XX::Fields::plt_idx_2;
    using Field_plt_idx_3Type = NPUReg40XX::Fields::plt_idx_3;
    using Field_plt_idx_4Type = NPUReg40XX::Fields::plt_idx_4;
    using Field_plt_idx_5Type = NPUReg40XX::Fields::plt_idx_5;
    using Field_plt_idx_6Type = NPUReg40XX::Fields::plt_idx_6;
    using Field_plt_idx_7Type = NPUReg40XX::Fields::plt_idx_7;
    using Field_plt_idx_8Type = NPUReg40XX::Fields::plt_idx_8;
    using Field_plt_idx_9Type = NPUReg40XX::Fields::plt_idx_9;
    using Field_plt_idx_10Type = NPUReg40XX::Fields::plt_idx_10;
    using Field_plt_idx_11Type = NPUReg40XX::Fields::plt_idx_11;
    using Field_plt_idx_12Type = NPUReg40XX::Fields::plt_idx_12;
    using Field_plt_idx_13Type = NPUReg40XX::Fields::plt_idx_13;
    using Field_plt_idx_14Type = NPUReg40XX::Fields::plt_idx_14;
    using Field_plt_idx_15Type = NPUReg40XX::Fields::plt_idx_15;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegCommonIDUWeightsOp(VPUIPDPU::IDUWeightsOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_wmodeType>(
            getTensorMode<NPUReg40XX::RegField_wmodeType>(op.getWmode()));
    descriptor.template write<typename Type_Fields::Field_wt_denseType>(!op.getWtSparse());

    const std::map<IDUWeightPalletMode, uint32_t> palletConfigMap = {{IDUWeightPalletMode::NO_PLT, 0},
                                                                     {IDUWeightPalletMode::ONE_BIT_PLT, 1},
                                                                     {IDUWeightPalletMode::TWO_BIT_PLT, 2},
                                                                     {IDUWeightPalletMode::FOUR_BIT_PLT, 3}};
    const auto palletMode = op.getWtPltCfg();
    const auto wtPltCfgValue = palletConfigMap.at(palletMode);

    descriptor.template write<typename Type_Fields::Field_wt_plt_cfgType>(wtPltCfgValue);
    if (op.getPoolWtData().has_value()) {
        descriptor.template write<typename Type_Fields::Field_pool_wt_dataType>(op.getPoolWtData().value());
    }
}

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegIDUWeightsOp(VPUIPDPU::IDUWeightsOp op, DpuInvariantDescriptorType& descriptor) {
    lowerToRegCommonIDUWeightsOp<Type_Fields>(op, descriptor);
    if (op.getQuantilesLut().has_value()) {
        auto wmode = getTensorMode<NPUReg40XX::RegField_wmodeType>(op.getWmode());
        auto quantilesLut = op.getQuantilesLut().value();
        constexpr unsigned numPalletTableEntries = 16;
        VPUX_THROW_UNLESS((quantilesLut.size() <= numPalletTableEntries),
                          "Number of palletization table entries ({0}) exceeds maximum of 16", quantilesLut.size());
        llvm::SmallVector<uint16_t, numPalletTableEntries> quantilesLutValues(numPalletTableEntries, 0);

        auto getPalletModeBitValue = [](const double value, const uint64_t wmode) -> uint16_t {
            if (wmode == static_cast<uint64_t>(nn_public::VpuInputTensorDType::FP16)) {
                vpux::type::float16 f16(value);
                return f16.to_bits();
            } else if (wmode == static_cast<uint64_t>(nn_public::VpuInputTensorDType::U8)) {
                int i8 = static_cast<int>(value);
                return (i8 < 0 ? 0 : static_cast<uint16_t>(i8));
            } else if (wmode == static_cast<uint64_t>(nn_public::VpuInputTensorDType::I8)) {
                return static_cast<uint16_t>(static_cast<int>(value));
            } else if (wmode == static_cast<uint64_t>(nn_public::VpuInputTensorDType::BF16)) {
                vpux::type::bfloat16 bf16(value);
                return bf16.to_bits();
            } else {
                VPUX_THROW("getPalletModeBitValue: Unsupported wmode for palletization table {0}", wmode);
            }
            return 0;
        };

        for (unsigned i = 0; i < quantilesLut.size(); ++i) {
            double lutEntry = mlir::dyn_cast<mlir::FloatAttr>(quantilesLut[i]).getValueAsDouble();
            quantilesLutValues[i] = getPalletModeBitValue(lutEntry, wmode);
        }

        descriptor.template write<typename Type_Fields::Field_plt_idx_0Type>(quantilesLutValues[0]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_1Type>(quantilesLutValues[1]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_2Type>(quantilesLutValues[2]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_3Type>(quantilesLutValues[3]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_4Type>(quantilesLutValues[4]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_5Type>(quantilesLutValues[5]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_6Type>(quantilesLutValues[6]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_7Type>(quantilesLutValues[7]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_8Type>(quantilesLutValues[8]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_9Type>(quantilesLutValues[9]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_10Type>(quantilesLutValues[10]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_11Type>(quantilesLutValues[11]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_12Type>(quantilesLutValues[12]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_13Type>(quantilesLutValues[13]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_14Type>(quantilesLutValues[14]);
        descriptor.template write<typename Type_Fields::Field_plt_idx_15Type>(quantilesLutValues[15]);
    }
}

// IDUWorkloadCfgOp
struct FieldsIDUWorkloadCfgOp {
    using Field_workload_operationType = NPUReg40XX::Fields::workload_operation;
    using Field_zm_inputType = NPUReg40XX::Fields::zm_input;
    using Field_dw_inputType = NPUReg40XX::Fields::dw_input;
    using Field_pool_wt_rd_disType = NPUReg40XX::Fields::pool_wt_rd_dis;
    using Field_dw_wt_sp_insType = NPUReg40XX::Fields::dw_wt_sp_ins;
    using Field_dynamic_bw_enType = NPUReg40XX::Fields::dynamic_bw_en;
    using Field_elop_wloadType = NPUReg40XX::Fields::elop_wload;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
bool lowerToRegIDUWorkloadCfgOp(VPUIPDPU::IDUWorkloadCfgOp op, DpuInvariantDescriptorType& descriptor) {
    bool successfullyLowered = true;
    auto workloadType = op.getWorkloadType();
    switch (workloadType) {
    case VPUIPDPU::IDUWorkloadType::MAXPOOL:
        descriptor.template write<typename Type_Fields::Field_workload_operationType>(0b10);
        descriptor.template write<typename Type_Fields::Field_zm_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dw_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_pool_wt_rd_disType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dw_wt_sp_insType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dynamic_bw_enType>(0b1);
        break;
    case VPUIPDPU::IDUWorkloadType::AVEPOOL:
        descriptor.template write<typename Type_Fields::Field_workload_operationType>(0);
        descriptor.template write<typename Type_Fields::Field_zm_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dw_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_pool_wt_rd_disType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dw_wt_sp_insType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dynamic_bw_enType>(0b1);
        break;
    case VPUIPDPU::IDUWorkloadType::CONV:
        descriptor.template write<typename Type_Fields::Field_workload_operationType>(0);
        descriptor.template write<typename Type_Fields::Field_zm_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dynamic_bw_enType>(0b1);
        break;
    case VPUIPDPU::IDUWorkloadType::DWCONV:
        descriptor.template write<typename Type_Fields::Field_workload_operationType>(0);
        descriptor.template write<typename Type_Fields::Field_zm_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dw_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dw_wt_sp_insType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dynamic_bw_enType>(0b1);
        break;
    case VPUIPDPU::IDUWorkloadType::ELTWISE:
        descriptor.template write<typename Type_Fields::Field_workload_operationType>(0);
        descriptor.template write<typename Type_Fields::Field_zm_inputType>(0b1);
        descriptor.template write<typename Type_Fields::Field_dynamic_bw_enType>(0b1);
        descriptor.template write<typename Type_Fields::Field_elop_wloadType>(0b1);
        break;
    default:
        successfullyLowered = false;
        break;
    }
    return successfullyLowered;
}

// IDUDepthWiseCfgOp
struct FieldsIDUDepthWiseCfgOp {
    using Field_dw_3x3s1_opt_disType = NPUReg40XX::Fields::dw_3x3s1_opt_dis;
    using Field_dw_opt_enType = NPUReg40XX::Fields::dw_opt_en;
    using Field_dw_opt_offsetType = NPUReg40XX::Fields::dw_opt_offset;
    using Field_pool_opt_enType = NPUReg40XX::Fields::pool_opt_en;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegIDUDepthWiseCfgOp(VPUIPDPU::IDUDepthWiseCfgOp op, DpuInvariantDescriptorType& descriptor) {
    if (op.getDw_3x3s1OptDis()) {
        descriptor.template write<typename Type_Fields::Field_dw_3x3s1_opt_disType>(1);
    }
    if (op.getDwOptOffset().has_value()) {
        descriptor.template write<typename Type_Fields::Field_dw_opt_enType>(1);
        descriptor.template write<typename Type_Fields::Field_dw_opt_offsetType>(op.getDwOptOffset().value());
        descriptor.template write<typename Type_Fields::Field_pool_opt_enType>(1);
    }
}

// IDUEltWiseCfgOp
struct FieldsIDUEltWiseCfgOp {
    using Field_elop_scale_aType = NPUReg40XX::Fields::elop_scale_a;
    using Field_elop_scale_bType = NPUReg40XX::Fields::elop_scale_b;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegIDUEltWiseCfgOp(VPUIPDPU::IDUEltWiseCfgOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_elop_scale_aType>(
            mlir::dyn_cast<mlir::IntegerAttr>(op.getElopScaleAAttr()).getInt());
    descriptor.template write<typename Type_Fields::Field_elop_scale_bType>(
            mlir::dyn_cast<mlir::IntegerAttr>(op.getElopScaleBAttr()).getInt());
}

// MPEActivationBiasOp
template <typename Field_mpe_actbiasType, typename DpuInvariantDescriptorType>
void lowerToRegMPEActivationBiasOp(VPUIPDPU::MPEActivationBiasOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_mpe_actbiasType>(op.getActBias());
}

// MPEWeightsBiasOp
template <typename Field_mpe_wtbiasType, typename DpuInvariantDescriptorType>
void lowerToRegMPEWeightsBiasOp(VPUIPDPU::MPEWeightsBiasOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_mpe_wtbiasType>(op.getWeightsBias());
}

// PPEFpBiasAddOp
struct FieldsPPEFpBiasAddOp {
    using Field_ppe_fp_scale_overrideType = NPUReg40XX::Fields::ppe_fp_scale_override;
    using Field_ppe_fp_biasType = NPUReg40XX::Fields::ppe_fp_bias;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEFpBiasAddOp(VPUIPDPU::PPEFpBiasAddOp op, DpuInvariantDescriptorType& descriptor) {
    VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getBiasStatic().has_value(),
                      "op {0} has ambiguous parameters", op);
    if (op.getScaleTable() != nullptr) {
        descriptor.template write<typename Type_Fields::Field_ppe_fp_scale_overrideType>(0);
    }
    if (op.getBiasStatic().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_fp_scale_overrideType>(1);
        descriptor.template write<typename Type_Fields::Field_ppe_fp_biasType>(
                op.getBiasStatic().value().convertToFloat());
    }
}

// PPEFpScalePreluMultOp
struct FieldsPPEFpScalePreluMultOp {
    using Field_ppe_fp_scale_overrideType = NPUReg40XX::Fields::ppe_fp_scale_override;
    using Field_ppe_fp_scaleType = NPUReg40XX::Fields::ppe_fp_scale;
    using Field_ppe_fp_prelu_enType = NPUReg40XX::Fields::ppe_fp_prelu_en;
    using Field_ppe_fp_preluType = NPUReg40XX::Fields::ppe_fp_prelu;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEFpScalePreluMultOp(VPUIPDPU::PPEFpScalePreluMultOp op, DpuInvariantDescriptorType& descriptor) {
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
    if (op.getPreluAlpha().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_fp_prelu_enType>(1);
        descriptor.template write<typename Type_Fields::Field_ppe_fp_preluType>(
                op.getPreluAlpha().value().convertToFloat());
    }
}

// PPEFpAddMultBypassOp
template <typename Field_ppe_fp_bypassType, typename DpuInvariantDescriptorType>
void lowerToRegPPEFpAddMultBypassOp(VPUIPDPU::PPEFpAddMultBypassOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_ppe_fp_bypassType>(op.getBypassMode());
}

// PPEFpConvertOp
struct FieldsPPEFpConvertOp {
    using Field_ppe_fp_convertType = NPUReg40XX::Fields::ppe_fp_convert;
    using Field_ppe_fp16_clampType = NPUReg40XX::Fields::ppe_fp16_clamp;
    using Field_ppe_fp16_ftzType = NPUReg40XX::Fields::ppe_fp16_ftz;
    using Field_ppe_bf16_roundType = NPUReg40XX::Fields::ppe_bf16_round;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEFpConvertOp(VPUIPDPU::PPEFpConvertOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_ppe_fp_convertType>(op.getConvertMode());
    if (op.getClampMode().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_fp16_clampType>(op.getClampMode().value());
    }
    if (op.getFtzMode().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_fp16_ftzType>(op.getFtzMode().value());
    }
    if (op.getBf16RoundMode().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_bf16_roundType>(op.getBf16RoundMode().value());
    }
}

// PPEIntBiasAddOp
struct FieldsPPEIntBiasAddOp {
    using Field_ppe_scale_overrideType = NPUReg40XX::Fields::ppe_scale_override;
    using Field_ppe_biasType = NPUReg40XX::Fields::ppe_bias;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEIntBiasAddOp(VPUIPDPU::PPEIntBiasAddOp op, DpuInvariantDescriptorType& descriptor) {
    VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getBiasStatic().has_value(),
                      "op {0} has ambiguous parameters", op);
    if (op.getScaleTable() != nullptr) {
        descriptor.template write<typename Type_Fields::Field_ppe_scale_overrideType>(0);
    }
    if (op.getBiasStatic().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_scale_overrideType>(1);
        descriptor.template write<typename Type_Fields::Field_ppe_biasType>(op.getBiasStatic().value());
    }
}

// PPEIntScaleMultOp
struct FieldsPPEIntScaleMultOp {
    using Field_ppe_scale_overrideType = NPUReg40XX::Fields::ppe_scale_override;
    using Field_ppe_scale_multType = NPUReg40XX::Fields::ppe_scale_mult;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEIntScaleMultOp(VPUIPDPU::PPEIntScaleMultOp op, DpuInvariantDescriptorType& descriptor) {
    VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getScaleStatic().has_value(),
                      "op {0} has ambiguous parameters", op);
    if (op.getScaleTable()) {
        descriptor.template write<typename Type_Fields::Field_ppe_scale_overrideType>(0);
    }
    if (op.getScaleStatic().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_scale_overrideType>(1);
        descriptor.template write<typename Type_Fields::Field_ppe_scale_multType>(op.getScaleStatic().value());
    }
}

// PPEIntPreluMultOp
template <typename Field_ppe_prelu_multType, typename DpuInvariantDescriptorType>
void lowerToRegPPEIntPreluMultOp(VPUIPDPU::PPEIntPreluMultOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_ppe_prelu_multType>(op.getPreluMultStatic());
}

// PPEIntScaleShiftOp
struct FieldsPPEIntScaleShiftOp {
    using Field_ppe_scale_overrideType = NPUReg40XX::Fields::ppe_scale_override;
    using Field_ppe_scale_shiftType = NPUReg40XX::Fields::ppe_scale_shift;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegPPEIntScaleShiftOp(VPUIPDPU::PPEIntScaleShiftOp op, DpuInvariantDescriptorType& descriptor) {
    VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getShiftStatic().has_value(),
                      "op {0} has ambiguous parameters", op);
    if (op.getScaleTable() != nullptr) {
        descriptor.template write<typename Type_Fields::Field_ppe_scale_overrideType>(0);
    }
    if (op.getShiftStatic().has_value()) {
        descriptor.template write<typename Type_Fields::Field_ppe_scale_overrideType>(1);
        descriptor.template write<typename Type_Fields::Field_ppe_scale_shiftType>(op.getShiftStatic().value());
    }
}

// PPEIntPreluShiftOp
template <typename Field_ppe_prelu_shiftType, typename DpuInvariantDescriptorType>
void lowerToRegPPEIntPreluShiftOp(VPUIPDPU::PPEIntPreluShiftOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_ppe_prelu_shiftType>(op.getPreluShiftStatic());
}

// PPEIntRoundOp
template <typename Field_ppe_scale_roundType, typename DpuInvariantDescriptorType>
void lowerToRegPPEIntRoundOp(VPUIPDPU::PPEIntRoundOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_ppe_scale_roundType>(op.getRoundMode());
}

// PPEIntZeroPointOffsetOp
template <typename Field_ppe_g8_bias_cType, typename DpuInvariantDescriptorType>
void lowerToRegPPEIntZeroPointOffsetOp(VPUIPDPU::PPEIntZeroPointOffsetOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_ppe_g8_bias_cType>(op.getZeroPointStatic());
}

// PPEIntClampOp
template <typename DpuInvariantDescriptorType>
void lowerToRegPPEIntClampOp(VPUIPDPU::PPEIntClampOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<NPUReg40XX::Fields::ppe_scale_hclamp>(op.getClampHigh());
    if (op.getClampLow().has_value()) {
        descriptor.template write<NPUReg40XX::Fields::ppe_scale_lclamp>(op.getClampLow().value());
    }
}

// PPEIntConvertOp
template <typename Field_ppe_i32_convertType, typename DpuInvariantDescriptorType>
void lowerToRegPPEIntConvertOp(VPUIPDPU::PPEIntConvertOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_ppe_i32_convertType>(op.getConvertMode());
}

// ODUOutTensorSizeOp
struct FieldsODUOutTensorSizeOp {
    using Field_te_dim_xType = NPUReg40XX::Fields::te_dim_x;
    using Field_te_dim_yType = NPUReg40XX::Fields::te_dim_y;
    using Field_te_dim_zType = NPUReg40XX::Fields::te_dim_z;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegODUOutTensorSizeOp(VPUIPDPU::ODUOutTensorSizeOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_te_dim_xType>(op.getDimX() - 1);
    descriptor.template write<typename Type_Fields::Field_te_dim_yType>(op.getDimY() - 1);
    descriptor.template write<typename Type_Fields::Field_te_dim_zType>(op.getDimZ() - 1);
}

// ODUDataReuseOp
template <typename Field_nthwType, typename DpuInvariantDescriptorType>
void lowerToRegODUDataReuseOp(VPUIPDPU::ODUDataReuseOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_nthwType>(op.getActivationReuse());
}

// ODUPermuteDataOp
template <typename Field_permutationType, typename DpuInvariantDescriptorType>
void lowerToRegODUPermuteDataOp(VPUIPDPU::ODUPermuteDataOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_permutationType>(op.getPermuteMode());
}

// ODUSparsityOp
struct FieldsODUSparsityOp {
    using Field_sp_valueType = NPUReg40XX::Fields::sp_value;
    using Field_sp_out_enType = NPUReg40XX::Fields::sp_out_en;
    using Field_write_spType = NPUReg40XX::Fields::write_sp;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegODUSparsityOp(VPUIPDPU::ODUSparsityOp op, DpuInvariantDescriptorType& descriptor) {
    uint64_t writeSp = op.getSparsityMap() ? 1 : 0;
    descriptor.template write<typename Type_Fields::Field_sp_valueType>(op.getSparseValue().value_or(0));
    descriptor.template write<typename Type_Fields::Field_sp_out_enType>(op.getCompressionEnabled().value_or(true));
    descriptor.template write<typename Type_Fields::Field_write_spType>(writeSp);
}

// ODUSwizzleDataOp
template <typename Field_swizzle_keyType, typename DpuInvariantDescriptorType>
void lowerToRegODUSwizzleDataOp(VPUIPDPU::ODUSwizzleDataOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_swizzle_keyType>(op.getSwizzleKey());
}

// ODUOutActivationsOp
struct FieldsODUOutActivationsOp {
    using Field_dtypeType = NPUReg40XX::Fields::dtype;
    using Field_write_acType = NPUReg40XX::Fields::write_ac;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegODUOutActivationsOp(VPUIPDPU::ODUOutActivationsOp op, DpuInvariantDescriptorType& descriptor) {
    uint64_t dataWidth(0);
    if (op.getDataWidth().has_value()) {
        dataWidth = static_cast<uint64_t>(op.getDataWidth().value());
    } else {
        auto outActType = mlir::cast<mlir::MemRefType>(op.getOutActivations().getType()).getElementType();
        dataWidth = static_cast<uint64_t>(getDataBitWidth(outActType));
    }

    descriptor.template write<typename Type_Fields::Field_dtypeType>(dataWidth);
    descriptor.template write<typename Type_Fields::Field_write_acType>(1);
}

// ODUMemoryModeOp
template <typename Field_modeType, typename DpuInvariantDescriptorType>
void lowerToRegODUMemoryModeOp(VPUIPDPU::ODUMemoryModeOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_modeType>(op.getMemMode());
}

// ODUCmxPortsOp
template <typename Field_cmx_port_muxing_disableType, typename DpuInvariantDescriptorType>
void lowerToRegODUCmxPortsOp(VPUIPDPU::ODUCmxPortsOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<Field_cmx_port_muxing_disableType>(op.getCmxPorts());
}

// ODUWriteCombineBufferOp
struct FieldsODUWriteCombineBufferOp {
    using Field_wcb_bypassType = NPUReg40XX::Fields::wcb_bypass;
    using Field_wcb_ac_modeType = NPUReg40XX::Fields::wcb_ac_mode;
    using Field_wcb_sp_modeType = NPUReg40XX::Fields::wcb_sp_mode;
};

template <typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegODUWriteCombineBufferOp(VPUIPDPU::ODUWriteCombineBufferOp op, DpuInvariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_wcb_bypassType>(0);
    descriptor.template write<typename Type_Fields::Field_wcb_ac_modeType>(op.getActivationsMode());
    descriptor.template write<typename Type_Fields::Field_wcb_sp_modeType>(
            op.getSparsityMode().value_or(vpux::VPUIPDPU::ODUWcbCombineMode::WCB_COMBINE_BY_CONTEXT));
}

// InvariantBarrierCfg
struct RegistersInvariantBarrierCfg {
    using Register_barriers_sched_Type = NPUReg40XX::Registers::barriers_sched_;
};

struct FieldsInvariantBarrierCfg {
    using Field_group_Type = NPUReg40XX::Fields::group_;
    using Field_mask_Type = NPUReg40XX::Fields::mask_;
    using Field_start_after_Type = NPUReg40XX::Fields::start_after_;
    using Field_clean_after_Type = NPUReg40XX::Fields::clean_after_;
    using Field_barriers_wait_mask_hi_Type = NPUReg40XX::Fields::barriers_wait_mask_hi_;
    using Field_barriers_wait_mask_lo_Type = NPUReg40XX::Fields::barriers_wait_mask_lo_;
    using Field_barriers_post_mask_hi_Type = NPUReg40XX::Fields::barriers_post_mask_hi_;
    using Field_barriers_post_mask_lo_Type = NPUReg40XX::Fields::barriers_post_mask_lo_;
};

template <typename Type_Registers, typename Type_Fields, typename DpuInvariantDescriptorType>
void lowerToRegBarrierCfgOpWithDPUInvariantParent(VPUIPDPU::DPUInvariantOp& dpuInvariantOp,
                                                  DpuInvariantDescriptorType& descriptor) {
    auto barrierCfgOps = to_small_vector(dpuInvariantOp.getRegion().getOps<VPUIPDPU::BarrierCfgOp>());
    if (barrierCfgOps.size() == 1) {
        auto barrierCfgOp = barrierCfgOps[0];

        uint64_t prodMaskLo = vpux::VPUMI40XX::computeMaskLo(barrierCfgOp.getUpdateBarriers());
        uint64_t prodMaskHi = vpux::VPUMI40XX::computeMaskHi(barrierCfgOp.getUpdateBarriers());
        uint64_t consMaskLo = vpux::VPUMI40XX::computeMaskLo(barrierCfgOp.getWaitBarriers());
        uint64_t consMaskHi = vpux::VPUMI40XX::computeMaskHi(barrierCfgOp.getWaitBarriers());

        uint64_t startAfter = barrierCfgOp.getStartAfter();
        uint64_t cleanAfter = barrierCfgOp.getCleanAfter();

        uint8_t barrierGroup = 0;
        uint8_t barrierMask = 0;
        std::tie(barrierGroup, barrierMask) = ELF::reduceWaitMaskTo8bit(consMaskLo);

        descriptor.template write<typename Type_Fields::Field_group_Type>(barrierGroup);
        descriptor.template write<typename Type_Fields::Field_mask_Type>(barrierMask);
        descriptor.template write<typename Type_Registers::Register_barriers_sched_Type,
                                  typename Type_Fields::Field_start_after_Type>(startAfter);
        descriptor.template write<typename Type_Registers::Register_barriers_sched_Type,
                                  typename Type_Fields::Field_clean_after_Type>(cleanAfter);
        descriptor.template write<typename Type_Fields::Field_barriers_wait_mask_hi_Type>(consMaskHi);
        descriptor.template write<typename Type_Fields::Field_barriers_wait_mask_lo_Type>(consMaskLo);
        descriptor.template write<typename Type_Fields::Field_barriers_post_mask_hi_Type>(prodMaskHi);
        descriptor.template write<typename Type_Fields::Field_barriers_post_mask_lo_Type>(prodMaskLo);
    } else {
        // just to explicitly show that we really intentionally only care about size == 1
        return;
    }
}

// IDUActSwizzleOp
template <typename Field_swizzle_key_offsetType, typename DpuVariantDescriptorType>
void lowerToRegIDUActSwizzleOp(VPUIPDPU::IDUActSwizzleOp op, DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_swizzle_key_offsetType>(static_cast<uint64_t>(op.getSwizzleKey()));
}

// IDUWeightSwizzleOp
template <typename Field_wt_swizzle_keyType, typename DpuVariantDescriptorType>
void lowerToRegIDUWeightSwizzleOp(VPUIPDPU::IDUWeightSwizzleOp op, DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_wt_swizzle_keyType>(static_cast<uint64_t>(op.getWtSwizzleKey()));
}

// IDUNthwNtkOp
template <typename Field_nthw_ntkType, typename DpuVariantDescriptorType>
void lowerToRegIDUNthwNtkOp(VPUIPDPU::IDUNthwNtkOp op, DpuVariantDescriptorType& descriptor) {
    descriptor.template write<Field_nthw_ntkType>(static_cast<uint64_t>(op.getNthwNtk()));
}

// IDUWorkloadSetOp
struct FieldsIDUWorkloadSetOp {
    using Field_workload_start_xType = NPUReg40XX::Fields::workload_start_x;
    using Field_workload_start_yType = NPUReg40XX::Fields::workload_start_y;
    using Field_workload_start_zType = NPUReg40XX::Fields::workload_start_z;
    using Field_workload_size_xType = NPUReg40XX::Fields::workload_size_x;
    using Field_workload_size_yType = NPUReg40XX::Fields::workload_size_y;
    using Field_workload_size_zType = NPUReg40XX::Fields::workload_size_z;
};

template <typename Type_Fields, typename DpuVariantDescriptorType>
void lowerToRegIDUWorkloadSetOp(VPUIPDPU::IDUWorkloadSetOp op, DpuVariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_workload_start_xType>(op.getStartX());
    descriptor.template write<typename Type_Fields::Field_workload_start_yType>(op.getStartY());
    descriptor.template write<typename Type_Fields::Field_workload_start_zType>(op.getStartZ());
    descriptor.template write<typename Type_Fields::Field_workload_size_xType>(op.getSizeX());
    descriptor.template write<typename Type_Fields::Field_workload_size_yType>(op.getSizeY());
    // The workload_size_z register must be aligned to 16, even in cases where IDU autopad is used and there are fewer
    // than 16 input channels for the workload
    descriptor.template write<typename Type_Fields::Field_workload_size_zType>(
            alignValUp(op.getSizeZ(), static_cast<int64_t>(16)));
}

// IDUPaddingOp
struct FieldsIDUPaddingOp {
    using Field_pad_count_upType = NPUReg40XX::Fields::pad_count_up;
    using Field_pad_count_leftType = NPUReg40XX::Fields::pad_count_left;
    using Field_pad_count_downType = NPUReg40XX::Fields::pad_count_down;
    using Field_pad_count_rightType = NPUReg40XX::Fields::pad_count_right;
};

template <typename Type_Fields, typename DpuVariantDescriptorType>
void lowerToRegIDUPaddingOp(VPUIPDPU::IDUPaddingOp op, DpuVariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_pad_count_upType>(op.getPadCount().getTop().getInt());
    descriptor.template write<typename Type_Fields::Field_pad_count_leftType>(op.getPadCount().getLeft().getInt());
    descriptor.template write<typename Type_Fields::Field_pad_count_downType>(op.getPadCount().getBottom().getInt());
    descriptor.template write<typename Type_Fields::Field_pad_count_rightType>(op.getPadCount().getRight().getInt());
}

// IDUWeightSetOp
struct FieldsIDUWeightSetOp {
    using Field_weight_sizeType = NPUReg40XX::Fields::weight_size;
    using Field_weight_numType = NPUReg40XX::Fields::weight_num;
    using Field_weight_startType = NPUReg40XX::Fields::weight_start;
};

template <typename Type_Fields, typename DpuVariantDescriptorType>
void lowerToRegIDUWeightSetOp(VPUIPDPU::IDUWeightSetOp op, DpuVariantDescriptorType& descriptor) {
    auto weightsNum = vpux::alignValUp(static_cast<std::uint64_t>(op.getWeightNum()),
                                       static_cast<std::uint64_t>(VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT));

    // weight_start register will be modified by relocation mechanism based on provided offset info
    descriptor.template write<typename Type_Fields::Field_weight_sizeType>(op.getWeightSize());
    descriptor.template write<typename Type_Fields::Field_weight_numType>(weightsNum);
    descriptor.template write<typename Type_Fields::Field_weight_startType>(op.getWeightStart());
}

// ODUOutSubtensorOp
struct FieldsODUOutSubtensorOp {
    using Field_te_beg_xType = NPUReg40XX::Fields::te_beg_x;
    using Field_te_beg_yType = NPUReg40XX::Fields::te_beg_y;
    using Field_te_beg_zType = NPUReg40XX::Fields::te_beg_z;
    using Field_te_end_xType = NPUReg40XX::Fields::te_end_x;
    using Field_te_end_yType = NPUReg40XX::Fields::te_end_y;
    using Field_te_end_zType = NPUReg40XX::Fields::te_end_z;
};

template <typename Type_Fields, typename DpuVariantDescriptorType>
void lowerToRegODUOutSubtensorOp(VPUIPDPU::ODUOutSubtensorOp op, DpuVariantDescriptorType& descriptor) {
    descriptor.template write<typename Type_Fields::Field_te_beg_xType>(op.getBeginCoordX());
    descriptor.template write<typename Type_Fields::Field_te_beg_yType>(op.getBeginCoordY());
    descriptor.template write<typename Type_Fields::Field_te_beg_zType>(op.getBeginCoordZ());
    descriptor.template write<typename Type_Fields::Field_te_end_xType>(op.getEndCoordX());
    descriptor.template write<typename Type_Fields::Field_te_end_yType>(op.getEndCoordY());
    descriptor.template write<typename Type_Fields::Field_te_end_zType>(op.getEndCoordZ());
}

// ODUHaloCfgOp
struct RegistersODUHaloCfgOp {
    using Register_halo_region0AType = NPUReg40XX::Registers::halo_region0A;
    using Register_halo_region0BType = NPUReg40XX::Registers::halo_region0B;
    using Register_halo_region0CType = NPUReg40XX::Registers::halo_region0C;
    using Register_halo_region0DType = NPUReg40XX::Registers::halo_region0D;

    using Register_halo_region1AType = NPUReg40XX::Registers::halo_region1A;
    using Register_halo_region1BType = NPUReg40XX::Registers::halo_region1B;
    using Register_halo_region1CType = NPUReg40XX::Registers::halo_region1C;
    using Register_halo_region1DType = NPUReg40XX::Registers::halo_region1D;

    using Register_halo_region2AType = NPUReg40XX::Registers::halo_region2A;
    using Register_halo_region2BType = NPUReg40XX::Registers::halo_region2B;
    using Register_halo_region2CType = NPUReg40XX::Registers::halo_region2C;
    using Register_halo_region2DType = NPUReg40XX::Registers::halo_region2D;

    using Register_halo_region3AType = NPUReg40XX::Registers::halo_region3A;
    using Register_halo_region3BType = NPUReg40XX::Registers::halo_region3B;
    using Register_halo_region3CType = NPUReg40XX::Registers::halo_region3C;
    using Register_halo_region3DType = NPUReg40XX::Registers::halo_region3D;

    using Register_halo_region4AType = NPUReg40XX::Registers::halo_region4A;
    using Register_halo_region4BType = NPUReg40XX::Registers::halo_region4B;
    using Register_halo_region4CType = NPUReg40XX::Registers::halo_region4C;
    using Register_halo_region4DType = NPUReg40XX::Registers::halo_region4D;

    using Register_halo_region5AType = NPUReg40XX::Registers::halo_region5A;
    using Register_halo_region5BType = NPUReg40XX::Registers::halo_region5B;
    using Register_halo_region5CType = NPUReg40XX::Registers::halo_region5C;
    using Register_halo_region5DType = NPUReg40XX::Registers::halo_region5D;
};

struct FieldsODUHaloCfgOp {
    using Field_begin_xType = NPUReg40XX::Fields::begin_x;
    using Field_begin_yType = NPUReg40XX::Fields::begin_y;
    using Field_end_xType = NPUReg40XX::Fields::end_x;
    using Field_ac_adr_offsetType = NPUReg40XX::Fields::ac_adr_offset;
    using Field_target_width_lsbType = NPUReg40XX::Fields::target_width_lsb;
    using Field_target_width_msbType = NPUReg40XX::Fields::target_width_msb;
    using Field_tile_selectType = NPUReg40XX::Fields::tile_select;
    using Field_sp_adr_offsetType = NPUReg40XX::Fields::sp_adr_offset;
    using Field_enableType = NPUReg40XX::Fields::enable;
    using Field_end_yType = NPUReg40XX::Fields::end_y;
};

struct FunctionsODUHaloCfgOp {
    using Function_target_width_lsbType = NPUReg40XX::RegField_target_width_lsbType;
    using Function_target_width_msbType = NPUReg40XX::RegField_target_width_msbType;
};

template <typename halo_regionA, typename halo_regionB, typename halo_regionC, typename halo_regionD,
          typename Type_Fields, typename Type_Functions, typename DpuVariantDescriptorType>
void fillValuesForHaloRegion(VPUIPDPU::ODUHaloRegionOp opHaloReg, DpuVariantDescriptorType& descriptor) {
    uint64_t lsbWidthValue(0), msbWidthValue(0);
    computeLsbAndMsbFromTargetWidth<typename Type_Functions::Function_target_width_lsbType,
                                    typename Type_Functions::Function_target_width_msbType>(
            opHaloReg.getTargetWidth(), msbWidthValue, lsbWidthValue);

    descriptor.template write<halo_regionA, typename Type_Fields::Field_sp_adr_offsetType>(
            opHaloReg.getSparsityOffset().value_or(0));
    descriptor.template write<halo_regionA, typename Type_Fields::Field_tile_selectType>(
            static_cast<uint64_t>(opHaloReg.getCastToTile()));
    descriptor.template write<halo_regionA, typename Type_Fields::Field_enableType>(1);
    descriptor.template write<halo_regionB, typename Type_Fields::Field_ac_adr_offsetType>(
            opHaloReg.getActivationsOffset());
    descriptor.template write<halo_regionB, typename Type_Fields::Field_target_width_lsbType>(lsbWidthValue);
    descriptor.template write<halo_regionC, typename Type_Fields::Field_begin_xType>(opHaloReg.getBeginCoordX());
    descriptor.template write<halo_regionC, typename Type_Fields::Field_begin_yType>(opHaloReg.getBeginCoordY());
    descriptor.template write<halo_regionC, typename Type_Fields::Field_target_width_msbType>(msbWidthValue);
    descriptor.template write<halo_regionD, typename Type_Fields::Field_end_xType>(opHaloReg.getEndCoordX());
    descriptor.template write<halo_regionD, typename Type_Fields::Field_end_yType>(opHaloReg.getEndCoordY());
}

template <typename Type_Registers, typename Type_Fields, typename Type_Functions, typename DpuVariantDescriptorType>
void fillValuesForHaloRegion(uint8_t haloRegionIdx, VPUIPDPU::ODUHaloRegionOp opHaloReg,
                             DpuVariantDescriptorType& descriptor) {
    if (haloRegionIdx == 0) {
        fillValuesForHaloRegion<typename Type_Registers::Register_halo_region0AType,
                                typename Type_Registers::Register_halo_region0BType,
                                typename Type_Registers::Register_halo_region0CType,
                                typename Type_Registers::Register_halo_region0DType, Type_Fields, Type_Functions>(
                opHaloReg, descriptor);
    } else if (haloRegionIdx == 1) {
        fillValuesForHaloRegion<typename Type_Registers::Register_halo_region1AType,
                                typename Type_Registers::Register_halo_region1BType,
                                typename Type_Registers::Register_halo_region1CType,
                                typename Type_Registers::Register_halo_region1DType, Type_Fields, Type_Functions>(
                opHaloReg, descriptor);
    } else if (haloRegionIdx == 2) {
        fillValuesForHaloRegion<typename Type_Registers::Register_halo_region2AType,
                                typename Type_Registers::Register_halo_region2BType,
                                typename Type_Registers::Register_halo_region2CType,
                                typename Type_Registers::Register_halo_region2DType, Type_Fields, Type_Functions>(
                opHaloReg, descriptor);
    } else if (haloRegionIdx == 3) {
        fillValuesForHaloRegion<typename Type_Registers::Register_halo_region3AType,
                                typename Type_Registers::Register_halo_region3BType,
                                typename Type_Registers::Register_halo_region3CType,
                                typename Type_Registers::Register_halo_region3DType, Type_Fields, Type_Functions>(
                opHaloReg, descriptor);
    } else if (haloRegionIdx == 4) {
        fillValuesForHaloRegion<typename Type_Registers::Register_halo_region4AType,
                                typename Type_Registers::Register_halo_region4BType,
                                typename Type_Registers::Register_halo_region4CType,
                                typename Type_Registers::Register_halo_region4DType, Type_Fields, Type_Functions>(
                opHaloReg, descriptor);
    } else if (haloRegionIdx == 5) {
        fillValuesForHaloRegion<typename Type_Registers::Register_halo_region5AType,
                                typename Type_Registers::Register_halo_region5BType,
                                typename Type_Registers::Register_halo_region5CType,
                                typename Type_Registers::Register_halo_region5DType, Type_Fields, Type_Functions>(
                opHaloReg, descriptor);
    }
}

template <typename Type_Registers, typename Type_Fields, typename Type_Functions, typename DpuVariantDescriptorType>
void lowerToRegODUHaloCfgOp(VPUIPDPU::ODUHaloCfgOp op, DpuVariantDescriptorType& descriptor) {
    uint8_t haloRegionIdx(0);
    for (const auto& haloRegionOp : op.getRegion().getOps()) {
        auto opHaloReg = mlir::dyn_cast_or_null<VPUIPDPU::ODUHaloRegionOp>(&haloRegionOp);
        if (opHaloReg == nullptr) {
            VPUX_THROW("Found invalid child op under ODUHaloCfgOp: {0}", haloRegionOp);
        }
        fillValuesForHaloRegion<Type_Registers, Type_Fields, Type_Functions>(haloRegionIdx, opHaloReg, descriptor);
        haloRegionIdx++;
    }
}

// VariantBarrierCfg
struct FieldsVariantBarrierCfg {
    using Field_cbarrier_hiType = NPUReg40XX::Fields::cbarrier_hi;
    using Field_cbarrier_loType = NPUReg40XX::Fields::cbarrier_lo;
    using Field_pbarrier_hiType = NPUReg40XX::Fields::pbarrier_hi;
    using Field_pbarrier_loType = NPUReg40XX::Fields::pbarrier_lo;
};

template <typename Type_Fields, typename DpuVariantDescriptorType>
void lowerToRegBarrierCfgOpWithDPUVariantParent(VPUIPDPU::DPUVariantOp& dpuVariantOp,
                                                DpuVariantDescriptorType& descriptor) {
    // TODO: E146560 - use barrierCfgOps only in variants that need to update barriers
    auto taskListCfgOps = to_small_vector(dpuVariantOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    auto barrierCfgOps = to_small_vector(dpuVariantOp.getRegion().getOps<VPUIPDPU::BarrierCfgOp>());

    if (barrierCfgOps.size() == 1 && taskListCfgOps.size() == 1) {
        auto taskListCfgOp = taskListCfgOps[0];
        auto variantCount = taskListCfgOp.getVariantCount();
        bool isFirstVariant = taskListCfgOp.getIsFirstVariant() || (variantCount == 1);
        bool isLastVariant = taskListCfgOp.getIsLastVariant() || (variantCount == 1);
        auto barrierCfgOp = barrierCfgOps[0];

        uint64_t prodMaskLo = 0;
        uint64_t prodMaskHi = 0;
        uint64_t consMaskLo = 0;
        uint64_t consMaskHi = 0;

        if (isFirstVariant) {
            consMaskLo = vpux::VPUMI40XX::computeMaskLo(barrierCfgOp.getWaitBarriers());
            consMaskHi = vpux::VPUMI40XX::computeMaskHi(barrierCfgOp.getWaitBarriers());
        }

        if (isLastVariant) {
            prodMaskLo = vpux::VPUMI40XX::computeMaskLo(barrierCfgOp.getUpdateBarriers());
            prodMaskHi = vpux::VPUMI40XX::computeMaskHi(barrierCfgOp.getUpdateBarriers());
        }

        descriptor.template write<typename Type_Fields::Field_cbarrier_hiType>(consMaskHi);
        descriptor.template write<typename Type_Fields::Field_cbarrier_loType>(consMaskLo);
        descriptor.template write<typename Type_Fields::Field_pbarrier_hiType>(prodMaskHi);
        descriptor.template write<typename Type_Fields::Field_pbarrier_loType>(prodMaskLo);
    } else {
        // just to explicitly show that we really intentionally only care about size == 1
        return;
    }
}

}  // namespace vpux::VPUIPDPU::arch40xx
