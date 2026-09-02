//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/nce_cluster_task_rewriter.hpp"
#include "vpux/compiler/conversion/passes/VPUIP2VPUMI40XX/buffer_conversion.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <cstdint>

namespace {

// E#145191:
// even though backend shouldn't really validate VPUIP IR
// there seem to be no such checks higher on the stack

template <class RangeT, class Property>
void checkIfDPUTasksAreDifferent(RangeT dpuTasks, Property&& functor) {
    const auto difference =
            std::adjacent_find(std::begin(dpuTasks), std::end(dpuTasks), [&functor](auto lhs, auto rhs) {
                return functor(lhs) != functor(rhs);
            });
    if (difference == dpuTasks.end()) {
        return;
    }

    auto lhs = *difference;
    auto rhs = *std::next(difference);
    VPUX_THROW("DPU tasks {} and {} from the same NCEClusterTaskOp are different: {} vs {}", lhs, rhs, functor(lhs),
               functor(rhs));
}

template <class RangeT>
void checkAllDPUTasksHaveTheSameMode(RangeT dpuTasks) {
    checkIfDPUTasksAreDifferent(std::move(dpuTasks), [](auto dpuTask) {
        return dpuTask.getMpeMode();
    });
}

template <class RangeT>
void checkAllDPUTasksHaveTheSameClusterID(RangeT dpuTasks) {
    checkIfDPUTasksAreDifferent(dpuTasks, [](auto dpuTask) {
        const auto maybeClusterID = dpuTask.getClusterId();
        assert(maybeClusterID.has_value());
        return maybeClusterID.value();
    });
}

}  // namespace

namespace vpux::vpuip2vpumi40xx {

mlir::LogicalResult NCEClusterTaskRewriter::matchAndRewrite(VPUIP::NCEClusterTaskOp origOp, OpAdaptor adaptor,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto origTaskOp = origOp->getParentOfType<VPURT::TaskOp>();
    auto dpuTasks = adaptor.getVariants().getOps<VPUIP::DPUTaskOp>();
    assert(!dpuTasks.empty());

    checkAllDPUTasksHaveTheSameMode(dpuTasks);
    // E#145191: ambiguous check, requires clarification
    // checkAllDPUTasksHaveTheSameClusterID(dpuTasks);
    // const auto tileIndex = (*dpuTasks.begin()).getClusterId().value();

    // E#145194: refactor to get cluster id easier
    uint32_t tileIndex = 0;
    if ((*dpuTasks.begin()).getClusterId().has_value()) {
        tileIndex = (*dpuTasks.begin()).getClusterId().value();
    } else if (origOp.getInput()) {
        auto bufferOp = mlir::cast<VPURT::DeclareBufferOp>(origOp.getInput().getDefiningOp());
        if (bufferOp.getSection() == VPURT::BufferSection::CMX_NN) {
            if (bufferOp.getSectionIndex().has_value() && !bufferOp.getSectionIndex().value().empty()) {
                auto tiles = parseIntArrayAttr<uint8_t>(bufferOp.getSectionIndex().value());
                tileIndex = *std::min_element(tiles.begin(), tiles.end());
            }
        }
    }

    const auto mpeModeAttr = (*dpuTasks.begin()).getMpeModeAttr();

    const auto indexWithOnlyTileSet = VPURegMapped::IndexType::get(ctx, tileIndex, 0, 0);
    const auto zeroUI64Attr = mlir::IntegerAttr::get(getUInt64Type(ctx), 0);
    auto cleanAfterAttr = !origTaskOp.getCleanAfter().has_value() ? zeroUI64Attr : origTaskOp.getCleanAfterAttr();
    auto startAfterAttr = !origTaskOp.getStartAfter().has_value() ? zeroUI64Attr : origTaskOp.getStartAfterAttr();

    auto weights = convertOrExtractBuffer(rewriter, adaptor.getWeights(), tileIndex);
    auto weightTable = convertOrExtractBuffer(rewriter, adaptor.getWeightTable(), tileIndex);
    auto weightTableDataPtr = convertOrExtractBuffer(rewriter, adaptor.getWeightTableDataPtr(), tileIndex);
    auto weightTableSpPtr = convertOrExtractBuffer(rewriter, adaptor.getWeightTableSpPtr(), tileIndex);
    auto weightTableScale = convertOrExtractBuffer(rewriter, adaptor.getWeightTableScale(), tileIndex);
    auto weightTableBias = convertOrExtractBuffer(rewriter, adaptor.getWeightTableBias(), tileIndex);
    auto weightTableAlpha = convertOrExtractBuffer(rewriter, adaptor.getWeightTableAlpha(), tileIndex);
    auto weightZeroPoints = convertOrExtractBuffer(rewriter, adaptor.getWeightZeroPoints(), tileIndex);
    auto sprLookupTable = convertOrExtractBuffer(rewriter, adaptor.getSprLookupTable(), tileIndex);
    auto palletLookupTable = convertOrExtractBuffer(rewriter, adaptor.getPalletLookupTable(), tileIndex);
    auto taskTypeAttr = adaptor.getTaskTypeAttr();
    auto dynamicSequenceLength = convertOrExtractBuffer(rewriter, adaptor.getDynamicSequenceLength(), tileIndex);
    auto maxPerXyBuff = convertOrExtractBuffer(rewriter, adaptor.getMaxPerXyBuff(), tileIndex);
    auto minPerXyBuff = convertOrExtractBuffer(rewriter, adaptor.getMinPerXyBuff(), tileIndex);

    auto invariant = rewriter.create<VPUMI40XX::DPUInvariantOp>(
            origOp.getLoc(), indexWithOnlyTileSet,
            nullptr,  // taskLocation
            nullptr,  // previousInvariant
            convertOrExtractBuffer(rewriter, adaptor.getInput(), tileIndex),
            convertOrExtractBuffer(rewriter, adaptor.getInputSparsityMap(), tileIndex),
            convertOrExtractBuffer(rewriter, adaptor.getInputStorageElementTable(), tileIndex), weights,
            convertOrExtractBuffer(rewriter, adaptor.getWeightsSparsityMap(), tileIndex), weightTable,
            weightTableDataPtr, weightTableSpPtr, weightTableScale, weightTableBias, weightTableAlpha, weightZeroPoints,
            sprLookupTable, palletLookupTable, convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff()),
            convertOrUnrollBuffer(rewriter, adaptor.getOutputSparsityMapBuff()), adaptor.getProfilingData(),
            dynamicSequenceLength, maxPerXyBuff, minPerXyBuff, adaptor.getMinMaxPerTensorBuff(), taskTypeAttr,
            adaptor.getEltwiseTypeAttr(), mpeModeAttr, adaptor.getMpeEngineAttr(), adaptor.getKernelSizeAttr(),
            adaptor.getKernelStridesAttr(), adaptor.getKernelPaddingAttr(), adaptor.getIsContinued(),
            adaptor.getCmSpPatternAttr(), adaptor.getInputChannelsCompression(), adaptor.getIsZeroOffsetWeightsTable(),
            adaptor.getOutChannelOffsetAttr(), adaptor.getIsSuperdense(), adaptor.getIsInplaceAttr(),
            adaptor.getInputSeSizeAttr(), adaptor.getOutputSeSizeAttr(), adaptor.getIsPermuteQuantize(),
            adaptor.getIsSmallKernelOptimized(), adaptor.getProfilingMetadataAttr(),
            mlir::ValueRange(),           // waitBarriers
            mlir::ValueRange(),           // updateBarriers
            startAfterAttr,               // startAfter
            cleanAfterAttr,               // cleanAfter
            nullptr,                      // enqueueBarrier
            origTaskOp.getWlmPageAttr(),  // wlmPageAttr
            adaptor.getSparsityConfigAttr(), adaptor.getDynamicScaleConfigAttr(), adaptor.getLocalRegionAttr(),
            adaptor.getS2dd2sConfigAttr()

    );
    auto createVPUMI40XXVariant = [&](auto dpuTask, bool sprLutRead = false, bool palletLutRead = false,
                                      bool forceInvRead = false) {
        rewriter.create<VPUMI40XX::DPUVariantOp>(
                dpuTask.getLoc(), indexWithOnlyTileSet,
                nullptr,  // taskLocation
                nullptr,  // previousVariant
                invariant.getResult(), weights, weightTable, weightTableDataPtr, weightTableSpPtr, weightTableScale,
                weightTableBias, weightTableAlpha, weightZeroPoints, taskTypeAttr, dpuTask.getInStartAttr(),
                dpuTask.getInEndAttr(), dpuTask.getOutStartAttr(), dpuTask.getOutEndAttr(), dpuTask.getPadAttr(),
                mpeModeAttr, mlir::IntegerAttr::get(getUInt64Type(ctx), tileIndex), dpuTask.getHaloRegionsAttr(),
                dpuTask.getWorkloadIdAttr(), sprLutRead, palletLutRead, forceInvRead, origTaskOp.getWlmPageAttr(),
                dpuTask.getVariantPrimitiveIdAttr(), dpuTask.getWeightTableOffsetAttr());
    };

    auto dpuTasksIt = dpuTasks.begin();

    const auto hasDummyVariant = (*dpuTasksIt).getIsDummy();
    if (hasDummyVariant) {
        // Processing dummy DPU task (see more info in InsertDelayDPUVariant pass)
        createVPUMI40XXVariant(*(dpuTasksIt++));

        // For the first variant that goes after the dummy one, an additional register is set:
        // - force_inv_read forces re-read of the Invariant. sprLUT read is triggered only as a part of
        // Invariant read (see DPU FSM diagram in HAS) and Invariant read may be skipped if it's already
        // loaded. As Dummy DPU variant loads Invariant for this workload, without it read of sprLUT may
        // be skipped as well, no matter what we set in readLut.

        VPUX_THROW_WHEN(dpuTasksIt == dpuTasks.end(),
                        "Expected at least one real DPU task after the dummy delay variant");

        createVPUMI40XXVariant(*dpuTasksIt,
                               /*sprLutRead=*/sprLookupTable != nullptr,
                               /*palletLutRead=*/palletLookupTable != nullptr,
                               /*forceInvRead=*/true);
        dpuTasksIt++;
    }

    // Keep LUT read enabled on every real variant that belongs to this invariant.
    // Pre-emption may cause DPU Power Cycle, which causes loss of LUT data in DPUs. When restoring a DPU after reset,
    // DPU won’t read LUT again if invar_lut_rd_en/invar_plt_rd_en are set to 0, so the DPU resumes without reloading
    // the LUT. This will cause PPE output to be misconfigured, resulting in incorrect output.
    // That is why we set sprLutRead/palletLutRead to true, if LUT is present, on each real variant.
    std::for_each(dpuTasksIt, dpuTasks.end(), [&](auto dpuTask) {
        createVPUMI40XXVariant(dpuTask, /*sprLutRead=*/sprLookupTable != nullptr,
                               /*palletLutRead=*/palletLookupTable != nullptr);
    });

    {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        auto& invariantPPERegion = invariant.getPpe();
        invariantPPERegion.emplaceBlock();
        rewriter.setInsertionPointToEnd(&invariantPPERegion.front());

        for (auto ppe : origOp.getPpe().getOps<VPUIP::PPETaskOp>()) {
            rewriter.create<VPUMI40XX::PPETaskOp>(ppe.getLoc(), ppe->getResultTypes(), ppe->getOperands(),
                                                  ppe->getAttrDictionary().getValue());
        }
    }

    rewriter.eraseOp(origOp);
    return mlir::success();
}

}  // namespace vpux::vpuip2vpumi40xx
