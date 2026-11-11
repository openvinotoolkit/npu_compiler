//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg60XX/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/lower_to_registers.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIPDPU/ops.hpp"

using namespace vpux;
using namespace NPUReg60XX;

namespace {

template <typename DPUInvOpsCategoryType, typename DPUInvOpType>
std::optional<DPUInvOpType> getDPUInvOp(mlir::Region& dpuInvRegion) {
    auto cfgOps = dpuInvRegion.getOps<DPUInvOpsCategoryType>();
    if (cfgOps.empty()) {
        return std::nullopt;
    }
    auto cfgOp = *cfgOps.begin();
    auto dpuInvOps = cfgOp.getRegion().template getOps<DPUInvOpType>();
    if (dpuInvOps.empty()) {
        return std::nullopt;
    }
    return std::optional<DPUInvOpType>(*dpuInvOps.begin());
}

}  // namespace

namespace vpux {
namespace vpuipdpu2npureg60xx {

DPUInvariantRewriter::DPUInvariantRewriter(mlir::MLIRContext* ctx, Logger log)
        : mlir::OpRewritePattern<VPUIPDPU::DPUInvariantOp>(ctx), _log(log) {
    setDebugName("DPUInvariant_VPUASM2NPUReg60XXRewriter");
}

mlir::LogicalResult DPUInvariantRewriter::matchAndRewrite(VPUIPDPU::DPUInvariantOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    vpux::NPUReg60XX::Descriptors::DpuInvariantRegister descriptor;

    fillIDUCfg(origOp.getRegion(), descriptor);
    fillMPECfg(origOp.getRegion(), descriptor);
    fillPPECfg(origOp.getRegion(), descriptor);
    fillODUCfg(origOp.getRegion(), descriptor);
    fillGeneralCfg(origOp.getRegion(), descriptor);
    fillBarrierCfg(origOp, descriptor);
    fillProfilingCfg(origOp, descriptor);

    auto taskListCfgOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    if (taskListCfgOp.size() == 1) {
        descriptor.write<Fields::variant_count_>(taskListCfgOp[0].getVariantCount());
    }

    descriptor.write<Fields::nvar_tag>(origOp.getIndex() + 1);

    NPUReg60XX::DPUInvariantOp::create(
            rewriter, origOp->getLoc(), origOp.getSymNameAttr(), origOp.getTaskIndexAttr(), std::move(descriptor),
            origOp.getTaskLocationAttr(), origOp.getInputAttr(), origOp.getInputSparsityMapAttr(),
            origOp.getInputStorageElementTableAttr(), origOp.getWeightsAttr(), origOp.getWeightsSparsityMapAttr(),
            origOp.getWeightTableAttr(), origOp.getWeightTableDataPtrAttr(), origOp.getWeightTableScaleAttr(),
            origOp.getSprLookupTableAttr(), origOp.getPalletLookupTableAttr(), origOp.getOutputAttr(),
            origOp.getOutputSparsityMapAttr(), origOp.getProfilingDataAttr(), origOp.getIsZeroOffsetWeightsTable(),
            origOp.getMaxPerXyAttr(), origOp.getMinPerXyAttr(), origOp.getMinMaxPerTensorAttr(),
            origOp.getDynamicSequenceLengthAttr(), origOp.getNceTaskTypeAttr(), origOp.getIsContinued());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

using DpuInvariantRegisterType = vpux::NPUReg60XX::Descriptors::DpuInvariantRegister;

void DPUInvariantRewriter::fillIDUCfg(mlir::Region& DPURegion, DpuInvariantRegisterType& descriptor) const {
    auto IDUCfgOps = DPURegion.getOps<VPUIPDPU::IDUCfgOp>();
    if (!IDUCfgOps.empty()) {
        auto IDUCfgOp = *IDUCfgOps.begin();

        for (auto& IDUOp : IDUCfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUStorageElementOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUStorageElementOp<vpux::VPUIPDPU::arch60xx::FieldsIDUStorageElementOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUKernelOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUKernelOp<vpux::VPUIPDPU::arch60xx::FieldsIDUKernelOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUStrideOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUStrideOp<vpux::VPUIPDPU::arch60xx::FieldsIDUStrideOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUInActivationsOp>(&IDUOp)) {
                VPUIPDPU::arch60xx::lowerToRegIDUInActivationsOp<
                        vpux::VPUIPDPU::arch60xx::FieldsIDUInActivationsOp, NPUReg60XX::RegField_amodeType,
                        vpux::VPUIPDPU::arch60xx::RegsTypeList, npu60xx::nn_public::VpuInputTensorDType>(op,
                                                                                                         descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUInputLayerCfgOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUInputLayerCfgOp<vpux::VPUIPDPU::arch60xx::FieldsIDUInputLayerCfgOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightsOp>(&IDUOp)) {
                VPUIPDPU::arch60xx::lowerToRegCommonIDUWeightsOp<
                        vpux::VPUIPDPU::arch60xx::FieldsIDUWeightsOp, NPUReg60XX::RegField_wmodeType,
                        vpux::VPUIPDPU::arch60xx::RegsTypeList, npu60xx::nn_public::VpuInputTensorDType>(op,
                                                                                                         descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWorkloadCfgOp>(&IDUOp)) {
                if (!VPUIPDPU::arch60xx::lowerToRegIDUWorkloadCfgOp<vpux::VPUIPDPU::arch60xx::FieldsIDUWorkloadCfgOp,
                                                                    npu60xx::nn_public::VpuInputTensorDType>(
                            op, descriptor)) {
                    VPUX_THROW("Failed to lower to registers the op IDUWorkloadCfg: invalid Wload Type");
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUDepthWiseCfgOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUDepthWiseCfgOp<vpux::VPUIPDPU::arch60xx::FieldsIDUDepthWiseCfgOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUEltWiseCfgOp>(&IDUOp)) {
                VPUIPDPU::arch50xx::lowerToRegIDUEltWiseCfgOp<
                        vpux::VPUIPDPU::arch60xx::NewTableGenFieldsIDUEltWiseCfgOp,
                        vpux::VPUIPDPU::arch60xx::OldTableGenFieldsIDUEltWiseCfgOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUEltWiseModeOp>(&IDUOp)) {
                VPUIPDPU::arch50xx::lowerToRegIDUEltWiseModeOp<Fields::elop_operation>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::WeightTableSplitOp>(&IDUOp)) {
                VPUIPDPU::arch60xx::lowerToRegWeightTableSplitOp<vpux::VPUIPDPU::arch60xx::FieldsWeightTableSplitOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::WeightTableLegacyZpOp>(&IDUOp)) {
                VPUIPDPU::arch60xx::lowerToRegWeightTableLegacyZpOp<vpux::VPUIPDPU::arch60xx::FieldsWeightTableSplitOp>(
                        op, descriptor);
            } else if (mlir::dyn_cast_or_null<VPUIPDPU::IDUSubtractMpeWtBiasOp>(&IDUOp)) {
                VPUIPDPU::lowerToRegIDUSubtractMpeWtBiasOp<Fields::subtract_mpe_wtbias>(descriptor);
            } else {
                VPUX_THROW("Unknown IDU operation: {0}", IDUOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillMPECfg(mlir::Region& DPURegion, DpuInvariantRegisterType& descriptor) const {
    auto MPECfgOps = DPURegion.getOps<VPUIPDPU::MPECfgOp>();
    if (!MPECfgOps.empty()) {
        auto MPECfgOp = *MPECfgOps.begin();

        for (auto& MPEOp : MPECfgOp.getRegion().getOps()) {
            if (mlir::dyn_cast_or_null<VPUIPDPU::MPEDenormalOperandsFTZOp>(&MPEOp)) {
                VPUIPDPU::lowerToRegMPEDenormalOperandsFTZOp<Fields::mpe_daz>(descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEActivationBiasOp>(&MPEOp)) {
                uint8_t inActDataWidth = 0;
                auto iduInActOp = getDPUInvOp<VPUIPDPU::IDUCfgOp, VPUIPDPU::IDUInActivationsOp>(DPURegion);
                if (!iduInActOp.has_value()) {
                    VPUX_THROW("Could not find the op IDUInActivations! Cannot lower to registers the op "
                               "MPEActivationBias.");
                }
                auto inActType = VPUIPDPU::getBaseType(
                        mlir::cast<mlir::MemRefType>(iduInActOp->getInActivations().getType()).getElementType());
                inActDataWidth = inActType.getIntOrFloatBitWidth();
                VPUIPDPU::arch60xx::lowerToRegMPEActivationBiasOp<vpux::VPUIPDPU::arch60xx::FieldsMPEActivationBiasOp>(
                        op, descriptor, inActDataWidth);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEWeightsBiasOp>(&MPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegMPEWeightsBiasOp<Fields::mpe_wtbias>(op, descriptor);
            } else if (mlir::dyn_cast_or_null<VPUIPDPU::MPEBf16ModeOp>(&MPEOp)) {
                VPUIPDPU::lowerToRegMPEBf16ModeOp<Fields::mpe_mode>(descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEEngineModeOp>(&MPEOp)) {
                VPUIPDPU::lowerToRegMPEEngineModeOp<Fields::cim_en>(op, descriptor);
            } else {
                VPUX_THROW("Unknown MPE operation: {0}", MPEOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillPPECfg(mlir::Region& DPURegion, DpuInvariantRegisterType& descriptor) const {
    auto PPECfgOps = DPURegion.getOps<VPUIPDPU::PPECfgOp>();
    if (!PPECfgOps.empty()) {
        // Default reg value for clamps = 0

        descriptor.template write<Fields::ppe_fp_scale_hclamp>(std::numeric_limits<float>::max());
        descriptor.template write<Fields::ppe_fp_scale_lclamp>(std::numeric_limits<float>::lowest());

        auto PPECfgOp = *PPECfgOps.begin();

        for (auto& PPEOp : PPECfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpBiasAddOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEFpBiasAddOp<vpux::VPUIPDPU::arch60xx::FieldsPPEFpBiasAddOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpConvertOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEFpConvertOp<vpux::VPUIPDPU::arch60xx::FieldsPPEFpConvertOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntZeroPointOffsetOp>(&PPEOp)) {
                auto oduOutActOp = getDPUInvOp<VPUIPDPU::ODUCfgOp, VPUIPDPU::ODUOutActivationsOp>(DPURegion);
                if (!oduOutActOp.has_value()) {
                    VPUX_THROW("Could not find the op ODUOutActivations! Cannot lower to registers the op "
                               "PPEIntZeroPointOffset.");
                }
                VPUIPDPU::ODUDataBitWidth oduDataWidth;
                if (oduOutActOp->getDataWidth().has_value()) {
                    oduDataWidth = oduOutActOp->getDataWidth().value();
                } else if (oduOutActOp->getOutActivations()) {
                    auto outActType =
                            mlir::cast<mlir::MemRefType>(oduOutActOp->getOutActivations().getType()).getElementType();
                    oduDataWidth = VPUIPDPU::getDataBitWidth(outActType);
                } else {
                    VPUX_THROW("Could not get data width from ODUOutActivations! Cannot lower to registers the op "
                               "PPEIntZeroPointOffset.");
                }
                VPUIPDPU::arch60xx::lowerToRegPPEIntZeroPointOffsetOp<vpux::VPUIPDPU::arch60xx::FieldsPPECfgOp>(
                        op, descriptor, oduDataWidth);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpScaleMultOp>(&PPEOp)) {
                VPUIPDPU::arch50xx::lowerToRegPPEFpScaleMultOp<vpux::VPUIPDPU::arch60xx::FieldsPPEFpScaleMultOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpSprLUTModeOp>(&PPEOp)) {
                VPUIPDPU::arch50xx::lowerToRegPPEFpSprLUTModeOp<vpux::VPUIPDPU::arch60xx::FieldsPPEFpSprLUTModeOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpPreluMultOp>(&PPEOp)) {
                VPUIPDPU::arch50xx::lowerToRegPPEFpPreluMultOp<vpux::VPUIPDPU::arch60xx::FieldsPPEFpPreluMultOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpClampOp>(&PPEOp)) {
                VPUIPDPU::arch50xx::lowerToRegPPEFpClampOp<vpux::VPUIPDPU::arch60xx::FieldsPPEFpClampOp>(op,
                                                                                                         descriptor);
            } else {
                VPUX_THROW("Unknown PPE operation: {0}", PPEOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillODUCfg(mlir::Region& DPURegion, DpuInvariantRegisterType& descriptor) const {
    auto ODUCfgOps = DPURegion.getOps<VPUIPDPU::ODUCfgOp>();
    if (!ODUCfgOps.empty()) {
        auto ODUCfgOp = *ODUCfgOps.begin();

        // TODO: E#80766 select optimal write combine mode and serialize based on VPUIPDU instruction
        descriptor.write<Fields::wcb_bypass>(0);

        // Statically set bits that should not be part of functional defaults

        // Not used by HW. Setting to 1 to be coeherent with GFile.
        descriptor.write<Fields::addr_format_sel>(1);

        // TODO: E#82814 should it be a default value? this is hardcoded and directly copied from POC runtime...
        descriptor.write<Fields::base_offset_a>(0x200);

        for (auto& ODUOp : ODUCfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutTensorSizeOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUOutTensorSizeOp<vpux::VPUIPDPU::arch60xx::FieldsODUOutTensorSizeOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUDataReuseOp>(&ODUOp)) {
                VPUIPDPU::arch60xx::lowerToRegODUDataReuseOp<Fields::nthw>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUPermuteDataOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUPermuteDataOp<Fields::permutation>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUSparsityOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUSparsityOp<vpux::VPUIPDPU::arch60xx::FieldsODUSparsityOp>(op,
                                                                                                           descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUSwizzleDataOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUSwizzleDataOp<Fields::swizzle_key>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutActivationsOp>(&ODUOp)) {
                VPUIPDPU::arch60xx::lowerToRegODUOutActivationsOp<vpux::VPUIPDPU::arch60xx::FieldsODUOutActivationsOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUMemoryModeOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUMemoryModeOp<Fields::mode>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUCmxPortsOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUCmxPortsOp<Fields::cmx_port_muxing_disable>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUWriteCombineBufferOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUWriteCombineBufferOp<
                        vpux::VPUIPDPU::arch60xx::FieldsODUWriteCombineBufferOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUReduceLayerOp>(&ODUOp)) {
                VPUIPDPU::arch60xx::lowerToRegODUReduceLayerOp<vpux::VPUIPDPU::arch60xx::FieldsODUReduceLayerOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODULocalRegionOp>(&ODUOp)) {
                VPUIPDPU::arch60xx::lowerToRegODULocalRegionOp<vpux::VPUIPDPU::arch60xx::FieldsODULocalRegionOp>(
                        op, descriptor);
            } else {
                VPUX_THROW("Unknown ODU operation: {0}", ODUOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillBarrierCfg(VPUIPDPU::DPUInvariantOp origOp, DpuInvariantRegisterType& descriptor) const {
    VPUIPDPU::arch40xx::lowerToRegBarrierCfgOpWithDPUInvariantParent<
            vpux::VPUIPDPU::arch60xx::RegistersInvariantBarrierCfg,
            vpux::VPUIPDPU::arch60xx::FieldsInvariantBarrierCfg>(origOp, descriptor);
}

void DPUInvariantRewriter::fillProfilingCfg(VPUIPDPU::DPUInvariantOp origOp,
                                            DpuInvariantRegisterType& descriptor) const {
    if (!origOp.getProfilingData().has_value()) {
        return;
    }
    descriptor.write<Fields::hwp_en>(1);
    descriptor.write<Fields::hwp_stat_mode>(7);
}

void DPUInvariantRewriter::fillGeneralCfg(mlir::Region& DPURegion, DpuInvariantRegisterType& descriptor) const {
    for (const auto& DPUOp : DPURegion.getOps()) {
        // DynScaleCfgInvOp
        if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::DynScaleCfgInvOp>(&DPUOp)) {
            VPUIPDPU::arch60xx::lowerToRegDynScaleCfgInvOp<vpux::VPUIPDPU::arch60xx::FieldsDynScaleCfgInvOp>(
                    op, descriptor);
        }
    }
}

}  // namespace vpuipdpu2npureg60xx
}  // namespace vpux
