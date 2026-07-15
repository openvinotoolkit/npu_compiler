//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/nce_cluster_task_rewriter.hpp"
#include <cstdint>
#include <map>
#include "vpux/compiler/conversion/passes/VPUIP2VPUMI40XX/buffer_conversion.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

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

struct WorkloadZCoord {
    int64_t zStart;
    int64_t zEnd;

    WorkloadZCoord(int64_t start, int64_t end = 0): zStart(start), zEnd(end) {
    }
};

inline bool operator<(const WorkloadZCoord& lhs, const WorkloadZCoord& rhs) {
    return lhs.zStart < rhs.zStart;
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
    auto createVPUMI40XXVariant = [&](auto dpuTask, std::optional<size_t> wtOffset = std::nullopt,
                                      bool sprLutRead = false, bool palletLutRead = false, bool forceInvRead = false) {
        rewriter.create<VPUMI40XX::DPUVariantOp>(
                dpuTask.getLoc(), indexWithOnlyTileSet,
                nullptr,  // taskLocation
                nullptr,  // previousVariant
                invariant.getResult(), weights, weightTable, weightTableDataPtr, weightTableSpPtr, weightTableScale,
                weightTableBias, weightTableAlpha, weightZeroPoints, taskTypeAttr, dpuTask.getInStartAttr(),
                dpuTask.getInEndAttr(), dpuTask.getOutStartAttr(), dpuTask.getOutEndAttr(), dpuTask.getPadAttr(),
                mpeModeAttr, mlir::IntegerAttr::get(getUInt64Type(ctx), tileIndex), dpuTask.getHaloRegionsAttr(),
                dpuTask.getWorkloadIdAttr(), sprLutRead, palletLutRead, forceInvRead, origTaskOp.getWlmPageAttr(),
                dpuTask.getVariantPrimitiveIdAttr(),
                wtOffset.has_value()
                        ? mlir::IntegerAttr::get(getUInt64Type(ctx), static_cast<int64_t>(wtOffset.value()))
                        : nullptr);
    };

    // In case of more than one DPU task belonging to the same NCEClusterTaskOp, we need to compute address offsets for
    // the weight table buffers for each variant. The reason is that the buffers are attached to NceClusterTaskOp and
    // contain weight table data for all variants, while the buffer addresses are part of the DPU variant descriptors.
    // The offset computation depends on whether weight table data pointers or weight zero points only are used.
    // The computation steps are as follows:
    // 1) Collect all unique Z output ranges among DPU tasks and sort them by the start Z coordinate
    // 2) Check that the Z output ranges cover the whole output channels range without overlaps or gaps
    // 3) For each unique Z output range, compute the corresponding offset based on the workload size in Z and previous
    // offsets
    // Further on these offsets are passed down to the weight table address fields in the DPU variant descriptors and
    // added to the start of buffer addresses during address relocation phase.
    std::map<WorkloadZCoord, size_t> workloadsZData;
    auto dpuTasksIt = dpuTasks.begin();

    if (sprLookupTable || palletLookupTable || dynamicSequenceLength) {
        // Skip dummy DPU task (see more info in InsertDelayDPUVariant pass)
        // NCEClusterTask shouldn't be treated as multi variant, because of the dummy DPUTask
        dpuTasksIt++;
    }

    // Store the start iterator for real DPU tasks (after skipping dummy if present)
    const auto realDpuTasksStart = dpuTasksIt;

    auto isMultiVariantWorkload = ++dpuTasksIt != dpuTasks.end();
    if (isMultiVariantWorkload) {
        auto getZeroPointTableAlignmentForWorkload8bit = [](int32_t zSize) {
            return VPU::NCESparsity::NewWeightsTableFormatMapper::getZeroPointTableAlignmentForWorkload(false, zSize);
        };
        auto getZeroPointTableAlignmentForWorkload4bit = [](int32_t zSize) {
            return VPU::NCESparsity::NewWeightsTableFormatMapper::getZeroPointTableAlignmentForWorkload(true, zSize);
        };
        std::function<int32_t(int32_t)> wtOffsetComputationFn;
        if (adaptor.getWeightTableDataPtr()) {
            wtOffsetComputationFn =
                    VPU::NCESparsity::NewWeightsTableFormatMapper::getNewPointerTableAlignmentForWorkload;
        } else if (adaptor.getWeightZeroPoints()) {
            auto weightsElementType = mlir::cast<NDTypeInterface>(adaptor.getWeights().getType()).getElementType();
            if (auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(weightsElementType)) {
                auto storageType = quantType.getStorageType();
                // Unwrap QuantileType to reach the inner integer storage type
                if (auto quantileType = mlir::dyn_cast<vpux::type::QuantileType>(storageType)) {
                    storageType = quantileType.getStorageType();
                }
                if (auto storageTypeAsIntegerType = mlir::dyn_cast<mlir::IntegerType>(storageType)) {
                    auto numberOfBitsInZeroPoint = storageTypeAsIntegerType.getWidth();
                    if (numberOfBitsInZeroPoint == 4) {
                        wtOffsetComputationFn = getZeroPointTableAlignmentForWorkload4bit;
                    } else {
                        wtOffsetComputationFn = getZeroPointTableAlignmentForWorkload8bit;
                    }
                }
            }
        }
        if (wtOffsetComputationFn) {
            std::for_each(realDpuTasksStart, dpuTasks.end(), [&](auto dpuTask) {
                auto workloadZCoord = WorkloadZCoord(parseIntArrayAttr<int64_t>(dpuTask.getOutStartAttr())[2],
                                                     parseIntArrayAttr<int64_t>(dpuTask.getOutEndAttr())[2]);
                const auto [it, inserted] = workloadsZData.insert({workloadZCoord, /*start_offset=*/0});
                if (!inserted) {
                    if (it->first.zEnd != workloadZCoord.zEnd) {
                        VPUX_THROW("DPU tasks from the same NCEClusterTaskOp have overlapping Z output "
                                   "ranges: "
                                   "[{}, {}] vs [{}, {}]",
                                   it->first.zStart, it->first.zEnd, workloadZCoord.zStart, workloadZCoord.zEnd);
                    }
                }
            });

            auto firstWorkloadZDataIt = workloadsZData.begin();
            if (workloadsZData.size() > 1) {
                auto workloadZDataPrevIt = firstWorkloadZDataIt;
                for (auto workloadZDataIt = std::next(firstWorkloadZDataIt); workloadZDataIt != workloadsZData.end();
                     ++workloadZDataIt) {
                    if (workloadZDataIt->first.zStart != workloadZDataPrevIt->first.zEnd + 1) {
                        VPUX_THROW(
                                "DPU tasks from the same NCEClusterTaskOp have overlapping or gaps in Z output ranges: "
                                "[{}, {}] vs [{}, {}]",
                                workloadZDataPrevIt->first.zStart, workloadZDataPrevIt->first.zEnd,
                                workloadZDataIt->first.zStart, workloadZDataIt->first.zEnd);
                    }

                    // Accumulate the byte offset for this workload's section.
                    // Each workload occupies a section whose byte size is determined by the number of output
                    // channels it processes (zEnd - zStart + 1), padded to the required alignment
                    // (wtOffsetComputationFn). The resulting offset is stored as weight_table_offset on the DPU variant
                    // descriptor and later added to the buffer's base CMX address, producing the
                    // final weight_d_ptr_start register value.
                    workloadZDataIt->second =
                            workloadZDataPrevIt->second +
                            wtOffsetComputationFn(static_cast<int32_t>(workloadZDataPrevIt->first.zEnd -
                                                                       workloadZDataPrevIt->first.zStart + 1));
                    workloadZDataPrevIt = workloadZDataIt;
                }
            }
        }
    }

    auto getWeightTableOffset = [&](auto dpuTask) -> std::optional<int64_t> {
        if (workloadsZData.size() > 1) {
            const auto workloadZCoord = WorkloadZCoord(parseIntArrayAttr<int64_t>(dpuTask.getOutStartAttr())[2]);
            if (auto offset = workloadsZData.at(workloadZCoord)) {
                return offset;
            }
        }
        return std::nullopt;
    };

    dpuTasksIt = dpuTasks.begin();
    if (sprLookupTable || palletLookupTable || dynamicSequenceLength) {
        // Processing dummy DPU task (see more info in InsertDelayDPUVariant pass)
        createVPUMI40XXVariant(*(dpuTasksIt++));

        // For the first variant that goes after the dummy one, an additional register is set:
        // - force_inv_read forces re-read of the Invariant. sprLUT read is triggered only as a part of
        // Invariant read (see DPU FSM diagram in HAS) and Invariant read may be skipped if it's already
        // loaded. As Dummy DPU variant loads Invariant for this workload, without it read of sprLUT may
        // be skipped as well, no matter what we set in readLut.
        createVPUMI40XXVariant(*dpuTasksIt, getWeightTableOffset(*dpuTasksIt),
                               /*sprLutRead=*/sprLookupTable != nullptr,
                               /*palletLutRead=*/palletLookupTable != nullptr,
                               /*forceInvRead=*/true);
        dpuTasksIt++;
    }

    std::for_each(dpuTasksIt, dpuTasks.end(), [&](auto dpuTask) {
        createVPUMI40XXVariant(dpuTask, getWeightTableOffset(dpuTask), /*sprLutRead=*/sprLookupTable != nullptr,
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
