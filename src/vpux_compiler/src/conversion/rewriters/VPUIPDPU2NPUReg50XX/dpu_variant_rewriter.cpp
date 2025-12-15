//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg50XX/dpu_variant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/types.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/lower_to_registers.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

using namespace vpux;
using namespace vpux::VPURegMapped;
using namespace NPUReg50XX;

namespace vpux {
namespace vpuipdpu2npureg50xx {

DPUVariantRewriter::DPUVariantRewriter(
        mlir::MLIRContext* ctx, Logger log,
        VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode)
        : mlir::OpRewritePattern<VPUIPDPU::DPUVariantOp>(ctx, mlir::PatternBenefit(2)),
          _log(log),
          _npu5PPEBackwardsCompatibilityMode(npu5PPEBackwardsCompatibilityMode) {
    setDebugName("DPUVariant_VPUASM2NPUReg50XXRewriter");
}

mlir::LogicalResult DPUVariantRewriter::matchAndRewrite(VPUIPDPU::DPUVariantOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // index incremented by one by runtime logic. Something to do with preemption
    // This value can be change if needed in the future. For now we use the index+1 just because it is
    // convinient for when we want to preempt when running on Simics
    // (E#71635)
    const uint64_t maxTaskId = (1ull << NPUReg50XX::RegField_var_tagType::getRegFieldWidth()) - 1;
    auto taskIdx = checked_cast_reg<NPUReg50XX::RegField_var_tagType>(
            static_cast<uint64_t>(origOp.getTaskIndex().getValue() % maxTaskId + 1));

    NPUReg50XX::Descriptors::DpuVariantRegister descriptor;
    descriptor.write<Fields::var_tag>(taskIdx);
    descriptor.write<Fields::invar_lptr_force>(0b1);
    descriptor.write<Fields::workload_start_odu>(0b1);
    descriptor.write<Fields::workload_start_idu>(0b1);
    descriptor.write<Fields::workload_prm_sel>(0b0);
    descriptor.write<Fields::workload_idu_auto_upd_0>(0b1);
    //  Note: wt_swizzle_sel needs to be on by default to match NNRT GF behaviour
    descriptor.write<Fields::wt_swizzle_sel>(1);
    descriptor.write<Fields::noc_clk_en>(1);

    fillDPUConfigs(origOp.getRegion(), descriptor);
    fillBarrierCfg(origOp, descriptor);
    fillProfilingCfg(origOp, descriptor);

    auto taskListCfgOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    if (!taskListCfgOp.empty()) {
        VPUX_THROW_UNLESS(taskListCfgOp.size() == 1, "Only one VPUIPDPU::DPUGroupOp should exist");
        auto tileSelectMask = VPUMI40XX::generateTileMask({taskListCfgOp[0].getInvariantIdx().getTileIdx()});
        auto forceInvReadOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::ForceInvReadOp>());
        descriptor.write<Fields::invariant_index_>(taskListCfgOp[0].getInvariantIdx().getValue());
        descriptor.write<Fields::invariant_>(static_cast<uint64_t>(tileSelectMask));
        descriptor.write<Fields::invar_lptr_force>(taskListCfgOp[0].getIsFirstVariant() || !forceInvReadOp.empty());
        descriptor.write<Fields::workload_odu_auto_upd>(taskListCfgOp[0].getIsLastVariant());
    }

    if (origOp.getNextLinkAttr()) {
        descriptor.write<Fields::next_sram_job_valid>(1);
    }

    rewriter.create<NPUReg50XX::DPUVariantOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getNextLinkAttr(),
                                              origOp.getTaskIndexAttr(), std::move(descriptor),
                                              origOp.getTaskLocationAttr(), origOp.getInvariantTaskLocationAttr(),
                                              origOp.getWeightsAttr(), origOp.getWeightTableAttr(),
                                              origOp.getNceTaskTypeAttr(), origOp.getWorkloadIdAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

using DpuVariantDescriptorType = NPUReg50XX::Descriptors::DpuVariantRegister;

void DPUVariantRewriter::fillDPUConfigs(mlir::Region& DPURegion, DpuVariantDescriptorType& descriptor) const {
    for (auto& DPUOp : DPURegion.getOps()) {
        // IDU ops
        if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUActSwizzleOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegIDUActSwizzleOp<Fields::swizzle_key_offset>(op, descriptor);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightSwizzleOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegIDUWeightSwizzleOp<Fields::wt_swizzle_key>(op, descriptor);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUNthwNtkOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegIDUNthwNtkOp<Fields::nthw_ntk>(op, descriptor);
        } else if (mlir::dyn_cast_or_null<VPUIPDPU::IDUSEDenseOp>(&DPUOp)) {
            VPUIPDPU::lowerToRegIDUSEDenseOp<Fields::dense_se>(descriptor);
        } else if (mlir::dyn_cast_or_null<VPUIPDPU::IDUConvContinueOp>(&DPUOp)) {
            VPUIPDPU::lowerToRegIDUConvContinueOp<Fields::conv_cond>(descriptor);
        } else if (mlir::dyn_cast_or_null<VPUIPDPU::IDUBinaryConfigOp>(&DPUOp)) {
            VPUIPDPU::lowerToRegIDUBinaryConfigOp<Fields::bin_cfg>(descriptor);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWorkloadSetOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegIDUWorkloadSetOp<VPUIPDPU::arch50xx::FieldsIDUWorkloadSetOp>(op, descriptor);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUPaddingOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegIDUPaddingOp<VPUIPDPU::arch50xx::FieldsIDUPaddingOp>(op, descriptor);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightSetOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegIDUWeightSetOp<VPUIPDPU::arch50xx::FieldsIDUWeightSetOp>(op, descriptor);
        } else if (mlir::dyn_cast_or_null<VPUIPDPU::IDUSEOnlyOp>(&DPUOp)) {
            VPUIPDPU::arch50xx::lowerToRegIDUSEOnlyOp<Fields::se_only_en>(descriptor);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUPerOutputChannelScalingOp>(&DPUOp)) {
            VPUIPDPU::arch50xx::lowerToRegIDUPerOutputChannelScalingOp<
                    VPUIPDPU::arch50xx::FieldsIDUPerOutputChannelScalingOp>(op, descriptor);
        }
        // ODU ops
        else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutSubtensorOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegODUOutSubtensorOp<VPUIPDPU::arch50xx::FieldsODUOutSubtensorOp>(op,
                                                                                                         descriptor);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUHaloCfgOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegODUHaloCfgOp<VPUIPDPU::arch50xx::RegistersODUHaloCfgOp,
                                                       VPUIPDPU::arch50xx::FieldsODUHaloCfgOp,
                                                       VPUIPDPU::arch50xx::FunctionsODUHaloCfgOp>(op, descriptor);
        }
        // PPEsprLUTReadOp
        else if (mlir::dyn_cast_or_null<VPUIPDPU::PPEsprLUTReadOp>(&DPUOp)) {
            if (DPUVariantRewriter::_npu5PPEBackwardsCompatibilityMode ==
                VPURegMapped::NPU5PPEBackwardsCompatibilityMode::ENABLED) {
                continue;
            }
            VPUIPDPU::lowerToRegPPEsprLUTReadOp<Fields::invar_lut_rd_en>(descriptor);
        }
        // ForceInvReadOp
        else if (mlir::dyn_cast_or_null<VPUIPDPU::ForceInvReadOp>(&DPUOp)) {
            VPUIPDPU::lowerToRegForceInvReadOp<Fields::invar_lptr_force>(descriptor);
        }
    }
}

void DPUVariantRewriter::fillBarrierCfg(VPUIPDPU::DPUVariantOp origOp, DpuVariantDescriptorType& descriptor) const {
    VPUIPDPU::arch40xx::lowerToRegBarrierCfgOpWithDPUVariantParent<VPUIPDPU::arch50xx::FieldsVariantBarrierCfg>(
            origOp, descriptor);
}

void DPUVariantRewriter::fillProfilingCfg(VPUIPDPU::DPUVariantOp origOp, DpuVariantDescriptorType& descriptor) const {
    if (!origOp.getWorkloadId().has_value()) {
        return;
    }
    uint32_t workloadId = origOp.getWorkloadId().value();

    descriptor.write<Fields::hwp_wload_id>(workloadId);
    descriptor.write<Fields::odu_stat_en>(1);
    descriptor.write<Fields::idu_stat_en>(1);
}

}  // namespace vpuipdpu2npureg50xx
}  // namespace vpux
