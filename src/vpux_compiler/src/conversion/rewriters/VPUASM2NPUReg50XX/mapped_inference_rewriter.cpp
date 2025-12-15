//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/mapped_inference_rewriter.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/utils.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace NPUReg50XX;
using namespace NPUReg50XX::Descriptors;
using namespace npu40xx;

namespace vpux {
namespace vpuasm2npureg50xx {

mlir::LogicalResult MappedInferenceRewriter::matchAndRewrite(VPUASM::MappedInferenceOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto dmaCount = parseIntArrayOfArrayAttr<int64_t>(origOp.getDmaCount());

    mlir::SmallVector<int64_t> dmaCountDDR;
    mlir::SmallVector<int64_t> dmaCountCMX;
    dmaCountDDR.reserve(dmaCount.size());
    dmaCountCMX.reserve(dmaCount.size());

    for (size_t dmaTileIndex = 0; dmaTileIndex < dmaCount.size(); dmaTileIndex++) {
        VPUX_THROW_UNLESS(dmaCount[dmaTileIndex].size() == 2, "Unsupported number of DMA types - '{0}'",
                          dmaCount[dmaTileIndex].size());

        dmaCountDDR.push_back(dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::DDR)]);
        dmaCountCMX.push_back(dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::CMX_NN)]);
    }

    const auto dmaCountDDRAttr = getIntArrayAttr(origOp.getContext(), ArrayRef(dmaCountDDR));
    const auto dmaCountCMXAttr = getIntArrayAttr(origOp.getContext(), ArrayRef(dmaCountCMX));

    auto moduleOp = origOp->getParentOfType<mlir::ModuleOp>();
    bool isActShaveProfilingEnabled =
            vpux::getProfilingSection(moduleOp, profiling::ExecutorType::ACTSHAVE).has_value();

    nn_public::VpuMappedInference mi = {};

    mi.barrier_configs.count = origOp.getBarrierCount();
    mi.media_tasks.count = origOp.getMediaCount();

    size_t totalDDRDmaCount = 0;
    VPUX_THROW_WHEN(dmaCountDDR.size() > nn_public::VPU_MAX_DMA_ENGINES, "Too many DMA DDR lists");
    for (size_t listIdx = 0; listIdx < dmaCountDDR.size(); ++listIdx) {
        mi.dma_tasks_ddr_[listIdx].count = dmaCountDDR[listIdx];
        totalDDRDmaCount += mi.dma_tasks_ddr_[listIdx].count;
    }

    size_t totalCMXDmaCount = 0;
    VPUX_THROW_WHEN(dmaCountCMX.size() > nn_public::VPU_MAX_DMA_ENGINES, "Too many DMA CMX lists");
    for (size_t listIdx = 0; listIdx < dmaCountCMX.size(); ++listIdx) {
        mi.dma_tasks_cmx_[listIdx].count = dmaCountCMX[listIdx];
        totalCMXDmaCount += mi.dma_tasks_cmx_[listIdx].count;
    }

    auto invariantCountVec = parseIntArrayAttr<int64_t>(origOp.getInvariantCount());
    VPUX_THROW_WHEN(invariantCountVec.size() > nn_public::VPU_MAX_TILES, "Too many Invariant lists");
    for (size_t listIdx = 0; listIdx < invariantCountVec.size(); ++listIdx) {
        mi.invariants[listIdx].count = invariantCountVec[listIdx];
    }

    auto variantCountVec = parseIntArrayAttr<int64_t>(origOp.getVariantCount());
    VPUX_THROW_WHEN(variantCountVec.size() > nn_public::VPU_MAX_TILES, "Too many Variant lists");
    for (size_t listIdx = 0; listIdx < variantCountVec.size(); ++listIdx) {
        mi.variants[listIdx].count = variantCountVec[listIdx];
    }

    auto actKernelRangesCountVec = parseIntArrayAttr<int64_t>(origOp.getActKernelRangesCount());
    VPUX_THROW_WHEN(actKernelRangesCountVec.size() > nn_public::VPU_MAX_TILES, "Too many ActKernelRange lists");
    for (size_t listIdx = 0; listIdx < actKernelRangesCountVec.size(); ++listIdx) {
        mi.act_kernel_ranges[listIdx].count = actKernelRangesCountVec[listIdx];
    }

    auto actKernelInvocationsCountVec = parseIntArrayAttr<int64_t>(origOp.getActKernelInvocationsCount());
    VPUX_THROW_WHEN(actKernelInvocationsCountVec.size() > nn_public::VPU_MAX_TILES, "Too many ActKernelInvo lists");
    for (size_t listIdx = 0; listIdx < actKernelInvocationsCountVec.size(); ++listIdx) {
        mi.act_kernel_invocations[listIdx].count = actKernelInvocationsCountVec[listIdx];
    }

    std::optional<uint64_t> stackSize;
    std::optional<std::pair<uint32_t, uint32_t>> stackFrames;
    if (origOp.getActShaveStacks().has_value()) {
        auto stackRef =
                _symRefMap.lookupSymbol(mlir::dyn_cast<mlir::SymbolRefAttr>(*origOp.getActShaveStacks()->begin()));
        auto stackOp = mlir::cast<VPUASM::ShaveStackFrameOp>(stackRef);
        stackSize = stackOp.getStackSize();
    } else {
        stackFrames = std::make_pair(NPUReg50XX::STACK_FRAME1, NPUReg50XX::STACK_FRAME2);
        stackSize = NPUReg50XX::STACK_FRAME2 - NPUReg50XX::STACK_FRAME1;
    }

    auto isActKernelInvocations = origOp.getActKernelInvocationsCount().size() > 0;
    NPUReg40XX::fillNNrtConfig<NPUReg50XX::ActShaveRtOp>(mi.shv_rt_configs, origOp, origOp.getActShaveRt(), stackSize,
                                                         isActShaveProfilingEnabled, isActKernelInvocations,
                                                         stackFrames);

    // Look only at the DMA tasks belonging to the first (and only) DMA engine
    std::tie(mi.task_storage_counts_.dma_ddr_count, mi.task_storage_counts_.dma_cmx_count) =
            VPUMI40XX::compute_dma_split(totalDDRDmaCount, totalCMXDmaCount);
    mi.task_storage_counts_.dpu_invariant_count = config::getConstraint(moduleOp, config::METADATA_MAX_INVARIANT_COUNT);
    mi.task_storage_counts_.dpu_variant_count = config::getConstraint(moduleOp, config::METADATA_MAX_VARIANT_COUNT);
    mi.task_storage_counts_.act_range_count = config::getConstraint(moduleOp, config::METADATA_MAX_KERNEL_RANGE_COUNT);
    mi.task_storage_counts_.act_invo_count =
            config::getConstraint(moduleOp, config::METADATA_MAX_KERNEL_INVOCATION_COUNT);
    mi.task_storage_counts_.media_count = config::getConstraint(moduleOp, config::METADATA_MAX_MEDIA_COUNT);

    VpuMappedInference miDesc;
    VPUX_THROW_UNLESS(sizeof(npu40xx::nn_public::VpuMappedInference) == miDesc.size(),
                      "HW VpuMappedInference size {0} != regMapped representation size {1}.",
                      sizeof(npu40xx::nn_public::VpuMappedInference), miDesc.size());
    miDesc.copyFrom(mi);

    rewriter.create<NPUReg50XX::MappedInferenceOp>(origOp->getLoc(),                           //
                                                   origOp.getSymNameAttr(),                    //
                                                   origOp.getDmaCountAttr(),                   //
                                                   dmaCountDDRAttr,                            //
                                                   dmaCountCMXAttr,                            //
                                                   origOp.getInvariantCountAttr(),             //
                                                   origOp.getVariantCountAttr(),               //
                                                   origOp.getActKernelRangesCountAttr(),       //
                                                   origOp.getActKernelInvocationsCountAttr(),  //
                                                   origOp.getMediaCountAttr(),                 //
                                                   origOp.getBarrierCountAttr(),               //
                                                   origOp.getMappedInferenceVersionAttr(),     //
                                                   origOp.getActShaveRtAttr(),                 //
                                                   origOp.getActShaveStacksAttr(),             //
                                                   origOp.getDmaHwpBaseAttr(),                 //
                                                   origOp.getHwpWorkpointCfgAttr(),            //
                                                   origOp.getDmaTasksAttr(),                   //
                                                   origOp.getInvariantTasksAttr(),             //
                                                   origOp.getVariantTasksAttr(),               //
                                                   origOp.getActKernelRangesAttr(),            //
                                                   origOp.getActKernelInvocationsAttr(),       //
                                                   origOp.getMediaTasksAttr(),                 //
                                                   origOp.getBarrierTasksAttr(),               //
                                                   origOp.getManagedMappedInferenceAttr(),     //
                                                   std::move(miDesc));                         //
    rewriter.eraseOp(origOp);

    return mlir::success();
}
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
