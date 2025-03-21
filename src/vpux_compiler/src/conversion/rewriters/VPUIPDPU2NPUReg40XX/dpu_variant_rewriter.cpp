//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg40XX/dpu_variant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/lower_to_registers.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

using namespace vpux;
using namespace vpux::VPURegMapped;

namespace vpux {
namespace vpuipdpu2npureg40xx {

DPUVariantRewriter::DPUVariantRewriter(mlir::MLIRContext* ctx, Logger log, VPU::DPUDryRunMode dryRunMode)
        : mlir::OpRewritePattern<VPUIPDPU::DPUVariantOp>(ctx, mlir::PatternBenefit(2)),
          _log(log),
          _dryRunMode(dryRunMode) {
    setDebugName("DPUVariant_VPUASM2NPUReg40XXRewriter");
}

mlir::LogicalResult DPUVariantRewriter::matchAndRewrite(VPUIPDPU::DPUVariantOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // index incremented by one by runtime logic. Something to do with preemption
    // This value can be change if needed in the future. For now we use the index+1 just because it is
    // convinient for when we want to preempt when running on Simics (E71635)
    const uint64_t maxTaskId = (1ull << NPUReg40XX::RegField_var_tagType::getRegFieldWidth()) - 1;
    auto taskIdx = checked_cast_reg<NPUReg40XX::RegField_var_tagType>(
            static_cast<uint64_t>(origOp.getTaskIndex().getValue() % maxTaskId + 1));

    NPUReg40XX::Descriptors::DpuVariantRegister descriptor;
    descriptor.write<Fields::var_tag>(taskIdx);
    descriptor.write<Fields::workload_start_odu>(1);
    descriptor.write<Fields::workload_start_idu>(1);
    descriptor.write<Fields::workload_prm_sel>(0);
    descriptor.write<Fields::workload_idu_auto_upd_0>(1);
    //  Note: wt_swizzle_sel needs to be on by default to match NNRT GF behaviour
    descriptor.write<Fields::wt_swizzle_sel>(1);

    if (_dryRunMode == VPU::DPUDryRunMode::STUB) {
        _log.trace("DPU dry run mode = 'stub', updating variant descriptor");
        fillStubCfg(descriptor);
    } else {
        fillDPUConfigs(origOp.getRegion(), descriptor);
    }
    fillBarrierCfg(origOp, descriptor);
    fillProfilingCfg(origOp, descriptor);

    auto taskListCfgOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    if (!taskListCfgOp.empty()) {
        VPUX_THROW_UNLESS(taskListCfgOp.size() == 1, "Only one VPUIPDPU::DPUGroupOp should exist");
        auto tileSelectMask = VPUMI40XX::generateTileMask({taskListCfgOp[0].getInvariantIdx().getTileIdx()});

        descriptor.write<Fields::invariant_index_>(taskListCfgOp[0].getInvariantIdx().getValue());
        descriptor.write<Fields::invariant_>(static_cast<uint64_t>(tileSelectMask));
        auto forceInvReadOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::ForceInvReadOp>());
        descriptor.write<Fields::invar_lptr_force>(taskListCfgOp[0].getIsFirstVariant() || !forceInvReadOp.empty());
        descriptor.write<Fields::workload_odu_auto_upd>(taskListCfgOp[0].getIsLastVariant());
    }

    if (origOp.getNextLinkAttr()) {
        descriptor.write<Fields::next_sram_job_valid>(1);
    }

    auto regDPUVariantAttr = DpuVariantRegisterAttr::get(rewriter.getContext(), std::move(descriptor));

    rewriter.create<NPUReg40XX::DPUVariantOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getNextLinkAttr(),
                                              origOp.getTaskIndexAttr(), regDPUVariantAttr,
                                              origOp.getTaskLocationAttr(), origOp.getInvariantTaskLocationAttr(),
                                              origOp.getWeightsAttr(), origOp.getWeightTableAttr(),
                                              origOp.getNceTaskTypeAttr(), origOp.getWorkloadIdAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

using DpuVariantDescriptorType = NPUReg40XX::Descriptors::DpuVariantRegister;

void DPUVariantRewriter::fillDPUConfigs(mlir::Region& DPURegion, DpuVariantDescriptorType& descriptor) const {
    for (const auto& DPUOp : DPURegion.getOps()) {
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
            VPUIPDPU::arch40xx::lowerToRegIDUWorkloadSetOp<VPUIPDPU::arch40xx::FieldsIDUWorkloadSetOp>(op, descriptor);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUPaddingOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegIDUPaddingOp<VPUIPDPU::arch40xx::FieldsIDUPaddingOp>(op, descriptor);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightSetOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegIDUWeightSetOp<VPUIPDPU::arch40xx::FieldsIDUWeightSetOp>(op, descriptor);
        }
        // ODU ops
        else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutSubtensorOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegODUOutSubtensorOp<VPUIPDPU::arch40xx::FieldsODUOutSubtensorOp>(op,
                                                                                                         descriptor);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUHaloCfgOp>(&DPUOp)) {
            VPUIPDPU::arch40xx::lowerToRegODUHaloCfgOp<VPUIPDPU::arch40xx::RegistersODUHaloCfgOp,
                                                       VPUIPDPU::arch40xx::FieldsODUHaloCfgOp,
                                                       VPUIPDPU::arch40xx::FunctionsODUHaloCfgOp>(op, descriptor);
        }
        // ForceInvReadOp
        else if (mlir::dyn_cast_or_null<VPUIPDPU::ForceInvReadOp>(&DPUOp)) {
            VPUIPDPU::lowerToRegForceInvReadOp<Fields::invar_lptr_force>(descriptor);
        }
    }
}

void DPUVariantRewriter::fillBarrierCfg(VPUIPDPU::DPUVariantOp origOp, DpuVariantDescriptorType& descriptor) const {
    VPUIPDPU::arch40xx::lowerToRegBarrierCfgOpWithDPUVariantParent<VPUIPDPU::arch40xx::FieldsVariantBarrierCfg>(
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
    descriptor.write<Fields::idu_stat_clr_mode>(0);
    descriptor.write<Fields::odu_stat_clr_mode>(0);
}

void DPUVariantRewriter::fillStubCfg(DpuVariantDescriptorType& descriptor) const {
    descriptor.write<Fields::workload_size_x>(0x1);
    descriptor.write<Fields::workload_size_y>(0x1);
    descriptor.write<Fields::workload_size_z>(0x10);
    descriptor.write<Fields::pad_count_up>(0x0);
    descriptor.write<Fields::pad_count_left>(0x0);
    descriptor.write<Fields::pad_count_down>(0x0);
    descriptor.write<Fields::pad_count_right>(0x0);
    descriptor.write<Fields::workload_start_x>(0x0);
    descriptor.write<Fields::workload_start_y>(0x0);
    descriptor.write<Fields::workload_start_z>(0x0);
    descriptor.write<Fields::weight_size>(0x10);
    descriptor.write<Fields::weight_num>(0x10);
    descriptor.write<Fields::te_beg_y>(0x0);
    descriptor.write<Fields::te_beg_z>(0x0);
    descriptor.write<Fields::te_beg_x>(0x0);
    descriptor.write<Fields::te_end_y>(0x0);
    descriptor.write<Fields::te_end_z>(0xF);
    descriptor.write<Fields::te_end_x>(0x0);
}

}  // namespace vpuipdpu2npureg40xx
}  // namespace vpux
