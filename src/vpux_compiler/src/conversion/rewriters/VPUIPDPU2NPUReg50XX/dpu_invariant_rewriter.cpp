//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg50XX/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/types.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/lower_to_registers.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/traits_utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace vpux::VPURegMapped;
using namespace npu40xx;
using namespace NPUReg50XX;

namespace vpux {
namespace vpuipdpu2npureg50xx {

DPUInvariantRewriter::DPUInvariantRewriter(
        mlir::MLIRContext* ctx, Logger log,
        VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode)
        : mlir::OpRewritePattern<VPUIPDPU::DPUInvariantOp>(ctx),
          _log(log),
          _npu5PPEBackwardsCompatibilityMode(npu5PPEBackwardsCompatibilityMode) {
    setDebugName("DPUInvariant_VPUASM2NPUReg50XXRewriter");
}

mlir::LogicalResult DPUInvariantRewriter::matchAndRewrite(VPUIPDPU::DPUInvariantOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    vpux::NPUReg50XX::Descriptors::DpuInvariantRegister descriptor;
    // fill default configuration
    descriptor.write<Fields::cmx_slice0_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice1_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice2_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice3_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice_size>(0x00018000);
    descriptor.write<Fields::ppe_scale_round>(3);
    descriptor.write<Fields::ppe_prelu_mult>(1);
    descriptor.write<Fields::ppe_fp_bypass>(1);

    fillIDUCfg(origOp.getRegion(), descriptor);
    fillMPECfg(origOp.getRegion(), descriptor);
    fillPPECfg(origOp.getRegion(), descriptor);
    fillODUCfg(origOp.getRegion(), descriptor);
    fillBarrierCfg(origOp, descriptor);
    fillProfilingCfg(origOp, descriptor);

    auto taskListCfgOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    if (taskListCfgOp.size() == 1) {
        descriptor.write<Fields::variant_count_>(taskListCfgOp[0].getVariantCount());
    }

    descriptor.write<Fields::nvar_tag>(origOp.getIndex() + 1);

    rewriter.create<NPUReg50XX::DPUInvariantOp>(
            origOp->getLoc(), origOp.getSymNameAttr(), origOp.getTaskIndexAttr(), std::move(descriptor),
            origOp.getTaskLocationAttr(), origOp.getInputAttr(), origOp.getInputSparsityMapAttr(),
            origOp.getInputStorageElementTableAttr(), origOp.getWeightsAttr(), origOp.getWeightsSparsityMapAttr(),
            origOp.getWeightTableAttr(), origOp.getSprLookupTableAttr(), origOp.getOutputAttr(),
            origOp.getOutputSparsityMapAttr(), origOp.getProfilingDataAttr(), origOp.getIsZeroOffsetWeightsTableAttr(),
            origOp.getNceTaskTypeAttr(), origOp.getIsContinuedAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

using DpuInvariantRegisterType = vpux::NPUReg50XX::Descriptors::DpuInvariantRegister;

void DPUInvariantRewriter::fillIDUCfg(mlir::Region& DPURegion, DpuInvariantRegisterType& descriptor) const {
    auto IDUCfgOps = DPURegion.getOps<VPUIPDPU::IDUCfgOp>();
    if (!IDUCfgOps.empty()) {
        auto IDUCfgOp = *IDUCfgOps.begin();

        for (auto& IDUOp : IDUCfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUStorageElementOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUStorageElementOp<vpux::VPUIPDPU::arch50xx::FieldsIDUStorageElementOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUKernelOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUKernelOp<vpux::VPUIPDPU::arch50xx::FieldsIDUKernelOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUStrideOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUStrideOp<vpux::VPUIPDPU::arch50xx::FieldsIDUStrideOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUInActivationsOp>(&IDUOp)) {
                VPUIPDPU::arch50xx::lowerToRegIDUInActivationsOp<vpux::VPUIPDPU::arch50xx::FieldsIDUInActivationsOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUInputLayerCfgOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUInputLayerCfgOp<vpux::VPUIPDPU::arch50xx::FieldsIDUInputLayerCfgOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightsOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUWeightsOp<vpux::VPUIPDPU::arch50xx::FieldsIDUWeightsOp>(op,
                                                                                                         descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWorkloadCfgOp>(&IDUOp)) {
                VPUIPDPU::arch50xx::lowerToRegIDUWorkloadCfgOp<vpux::VPUIPDPU::arch50xx::FieldsIDUWorkloadCfgOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUDepthWiseCfgOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUDepthWiseCfgOp<vpux::VPUIPDPU::arch50xx::FieldsIDUDepthWiseCfgOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUEltWiseCfgOp>(&IDUOp)) {
                VPUIPDPU::arch50xx::lowerToRegIDUEltWiseCfgOp<
                        vpux::VPUIPDPU::arch50xx::NewTableGenFieldsIDUEltWiseCfgOp,
                        vpux::VPUIPDPU::arch50xx::OldTableGenFieldsIDUEltWiseCfgOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUEltWiseModeOp>(&IDUOp)) {
                VPUIPDPU::arch50xx::lowerToRegIDUEltWiseModeOp<Fields::elop_operation>(op, descriptor);
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
                VPUIPDPU::arch40xx::lowerToRegMPEActivationBiasOp<Fields::mpe_actbias>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEWeightsBiasOp>(&MPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegMPEWeightsBiasOp<Fields::mpe_wtbias>(op, descriptor);
            } else if (mlir::dyn_cast_or_null<VPUIPDPU::MPEBf16ModeOp>(&MPEOp)) {
                VPUIPDPU::lowerToRegMPEBf16ModeOp<Fields::mpe_mode>(descriptor);
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
        // high & low clamps are set to default max and min values here accordingly, due to both int and fp clamps
        // being at the same address. If one clamp (int/fp) has a register default value, overlay is true and
        // setting the other clamp field won't overwrite the whole initial value, just the active bits, thus
        // creating a new unrelated value
        if (_npu5PPEBackwardsCompatibilityMode == VPURegMapped::NPU5PPEBackwardsCompatibilityMode::ENABLED) {
            descriptor.write<Fields::ppe_int_scale_hclamp>(static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
            descriptor.write<Fields::ppe_int_scale_lclamp>(static_cast<int64_t>(std::numeric_limits<int32_t>::min()));
        } else {
            descriptor.write<Fields::ppe_fp_scale_hclamp>(std::numeric_limits<float>::max());
            descriptor.write<Fields::ppe_fp_scale_lclamp>(std::numeric_limits<float>::lowest());
        }
        auto PPECfgOp = *PPECfgOps.begin();

        for (auto& PPEOp : PPECfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpBiasAddOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEFpBiasAddOp<vpux::VPUIPDPU::arch50xx::FieldsPPEFpBiasAddOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpScalePreluMultOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEFpScalePreluMultOp<
                        vpux::VPUIPDPU::arch50xx::FieldsPPEFpScalePreluMultOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpAddMultBypassOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEFpAddMultBypassOp<Fields::ppe_fp_bypass>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpConvertOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEFpConvertOp<vpux::VPUIPDPU::arch50xx::FieldsPPEFpConvertOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntBiasAddOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntBiasAddOp<vpux::VPUIPDPU::arch50xx::FieldsPPEIntBiasAddOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntScaleMultOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntScaleMultOp<vpux::VPUIPDPU::arch50xx::FieldsPPEIntScaleMultOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntPreluMultOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntPreluMultOp<Fields::ppe_prelu_mult>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntScaleShiftOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntScaleShiftOp<vpux::VPUIPDPU::arch50xx::FieldsPPEIntScaleShiftOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntPreluShiftOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntPreluShiftOp<Fields::ppe_prelu_shift>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntRoundOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntRoundOp<Fields::ppe_scale_round>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntZeroPointOffsetOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntZeroPointOffsetOp<Fields::ppe_g8_bias_c>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntClampOp>(&PPEOp)) {
                VPUIPDPU::arch50xx::lowerToRegPPEIntClampOp<vpux::VPUIPDPU::arch50xx::FieldsPPEIntClampOp>(op,
                                                                                                           descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntConvertOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntConvertOp<Fields::ppe_i32_convert>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpScaleMultOp>(&PPEOp)) {
                VPUIPDPU::arch50xx::lowerToRegPPEFpScaleMultOp<vpux::VPUIPDPU::arch50xx::FieldsPPEFpScaleMultOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpSprLUTModeOp>(&PPEOp)) {
                VPUIPDPU::arch50xx::lowerToRegPPEFpSprLUTModeOp<vpux::VPUIPDPU::arch50xx::FieldsPPEFpSprLUTModeOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpPreluMultOp>(&PPEOp)) {
                VPUIPDPU::arch50xx::lowerToRegPPEFpPreluMultOp<vpux::VPUIPDPU::arch50xx::FieldsPPEFpPreluMultOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpClampOp>(&PPEOp)) {
                VPUIPDPU::arch50xx::lowerToRegPPEFpClampOp<vpux::VPUIPDPU::arch50xx::FieldsPPEFpClampOp>(op,
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
        // TODO: E#81883 need to figure this out why it's always set to 1?
        descriptor.write<Fields::rst_ctxt>(1);

        // TODO: E#82814 should it be a default value? this is hardcoded and directly copied from POC runtime...
        descriptor.write<Fields::base_offset_a>(0x200);

        for (auto& ODUOp : ODUCfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutTensorSizeOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUOutTensorSizeOp<vpux::VPUIPDPU::arch50xx::FieldsODUOutTensorSizeOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUDataReuseOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUDataReuseOp<Fields::nthw>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUPermuteDataOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUPermuteDataOp<Fields::permutation>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUSparsityOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUSparsityOp<vpux::VPUIPDPU::arch50xx::FieldsODUSparsityOp>(op,
                                                                                                           descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUSwizzleDataOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUSwizzleDataOp<Fields::swizzle_key>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutActivationsOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUOutActivationsOp<vpux::VPUIPDPU::arch50xx::FieldsODUOutActivationsOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUMemoryModeOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUMemoryModeOp<Fields::mode>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUCmxPortsOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUCmxPortsOp<Fields::cmx_port_muxing_disable>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUWriteCombineBufferOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUWriteCombineBufferOp<
                        vpux::VPUIPDPU::arch50xx::FieldsODUWriteCombineBufferOp>(op, descriptor);
            }
        }
    }
}

void DPUInvariantRewriter::fillBarrierCfg(VPUIPDPU::DPUInvariantOp origOp, DpuInvariantRegisterType& descriptor) const {
    VPUIPDPU::arch40xx::lowerToRegBarrierCfgOpWithDPUInvariantParent<
            vpux::VPUIPDPU::arch50xx::RegistersInvariantBarrierCfg,
            vpux::VPUIPDPU::arch50xx::FieldsInvariantBarrierCfg>(origOp, descriptor);
}

void DPUInvariantRewriter::fillProfilingCfg(VPUIPDPU::DPUInvariantOp origOp,
                                            DpuInvariantRegisterType& descriptor) const {
    if (!origOp.getProfilingData().has_value()) {
        return;
    }
    descriptor.write<Fields::hwp_en>(1);
    descriptor.write<Fields::hwp_stat_mode>(7);
}

}  // namespace vpuipdpu2npureg50xx
}  // namespace vpux
