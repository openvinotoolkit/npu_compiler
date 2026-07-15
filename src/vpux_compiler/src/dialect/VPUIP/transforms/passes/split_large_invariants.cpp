//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <algorithm>
#include <climits>
#include <map>
#include <numeric>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_SPLITLARGEINVARIANTS
#define GEN_PASS_DEF_SPLITLARGEINVARIANTS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

namespace vpux::VPUIP {
namespace {

VPUIP::DpuProfilingMetadataAttr updateDpuProfilingMetadata(VPUIP::DpuProfilingMetadataAttr oldMetadata,
                                                           int64_t numVariantsInChunk, uint32_t taskId) {
    auto* ctx = oldMetadata.getContext();
    return VPUIP::DpuProfilingMetadataAttr::get(ctx, oldMetadata.getBufferId(), getIntAttr(ctx, taskId),
                                                getIntAttr(ctx, numVariantsInChunk),
                                                getIntAttr(ctx, numVariantsInChunk), oldMetadata.getClusterId());
}

VPUIP::DpuProfilingMetadataAttr updateDpuProfilingTaskId(VPUIP::DpuProfilingMetadataAttr oldMetadata, uint32_t taskId) {
    auto* ctx = oldMetadata.getContext();
    return VPUIP::DpuProfilingMetadataAttr::get(ctx, oldMetadata.getBufferId(), getIntAttr(ctx, taskId),
                                                oldMetadata.getMaxVariants(), oldMetadata.getNumVariants(),
                                                oldMetadata.getClusterId());
}

void normalizeDpuProfilingTaskIds(mlir::func::FuncOp func) {
    using ProfilingKey = std::pair<int64_t, int64_t>;
    std::map<ProfilingKey, SmallVector<VPUIP::NCEClusterTaskOp>> tasksByProfilingBuffer;

    func.walk([&](VPURT::TaskOp taskOp) {
        auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
        if (nceTask == nullptr || nceTask.getProfilingData() == nullptr) {
            return;
        }

        const auto profilingMetadata = nceTask.getProfilingMetadataAttr();
        VPUX_THROW_WHEN(profilingMetadata == nullptr,
                        "Expected profiling metadata for NCEClusterTask with profiling data at '{0}'",
                        nceTask.getLoc());
        VPUX_THROW_WHEN(profilingMetadata.getClusterId() == nullptr,
                        "Expected clusterId in DPU profiling metadata at '{0}'", nceTask.getLoc());

        const ProfilingKey key{profilingMetadata.getBufferId().getInt(), profilingMetadata.getClusterId().getInt()};
        tasksByProfilingBuffer[key].push_back(nceTask);
    });

    for (auto& [_, tasks] : tasksByProfilingBuffer) {
        if (tasks.empty()) {
            continue;
        }

        // Parser consumes DPU metadata sorted by taskId per (bufferId, clusterId).
        // Keep the same ordering contract when reassigning IDs after splitting.
        std::stable_sort(tasks.begin(), tasks.end(), [](VPUIP::NCEClusterTaskOp lhs, VPUIP::NCEClusterTaskOp rhs) {
            return lhs.getProfilingMetadataAttr().getTaskId().getInt() <
                   rhs.getProfilingMetadataAttr().getTaskId().getInt();
        });

        int64_t firstTaskId = tasks.front().getProfilingMetadataAttr().getTaskId().getInt();
        for (auto task : tasks) {
            firstTaskId = std::min(firstTaskId, task.getProfilingMetadataAttr().getTaskId().getInt());
        }

        uint32_t nextTaskId = checked_cast<uint32_t>(firstTaskId);
        for (auto task : tasks) {
            task.setProfilingMetadataAttr(updateDpuProfilingTaskId(task.getProfilingMetadataAttr(), nextTaskId++));
        }
    }
}

mlir::Value getProfilingChunkBuffer(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value profilingData,
                                    int64_t beginVariantIdx, int64_t numVariantsInChunk,
                                    uint16_t profilingWorkloadSizeBytes) {
    const auto profilingDataType = mlir::cast<vpux::NDTypeInterface>(profilingData.getType());
    const auto elemSizeBits = profilingDataType.getElemTypeSize().count();
    const auto elemSizeBytes = elemSizeBits / CHAR_BIT;
    VPUX_THROW_WHEN(elemSizeBytes == 0 || profilingWorkloadSizeBytes % elemSizeBytes != 0,
                    "Unsupported profiling data element size '{0}' bits for profiling workload size '{1}' bytes",
                    elemSizeBits, profilingWorkloadSizeBytes);

    const auto elementsPerVariant = checked_cast<int64_t>(profilingWorkloadSizeBytes / elemSizeBytes);
    const auto size = numVariantsInChunk * elementsPerVariant;
    const auto chunkType = profilingDataType.changeShape(ShapeRef(SmallVector<int64_t>{size}));

    auto declOp = profilingData.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(declOp == nullptr, "Expected VPURT.DeclareBufferOp", loc);
    const auto chunkByteOffset = declOp.getByteOffset() + beginVariantIdx * profilingWorkloadSizeBytes;
    auto chunkBuffer = builder.create<VPURT::DeclareBufferOp>(
            loc, chunkType, declOp.getSectionAttr(), declOp.getSectionIndexAttr(), getIntAttr(builder, chunkByteOffset),
            declOp.getSwizzlingKeyAttr());
    return chunkBuffer.getBuffer();
}

// Returns a sub-buffer covering the data-pointer table section for DPU variants [begin, end).
mlir::Value getDataPointerTableChunkBuffer(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value table,
                                           ArrayRef<int32_t> allVariantWorkloadSizes, int64_t begin, int64_t end) {
    const auto accumulateAlignedElements = [](int64_t acc, int32_t zSize) {
        return acc +
               VPU::NCESparsity::NewWeightsTableFormatMapper::getNewPointerTableLogicalAlignmentForWorkload(zSize);
    };
    const int64_t chunkByteOffset =
            std::accumulate(allVariantWorkloadSizes.begin(), allVariantWorkloadSizes.begin() + begin, int64_t{0},
                            accumulateAlignedElements) *
            VPU::NCESparsity::WEIGHTS_TABLE_POINTER_SIZE;
    const int64_t chunkElements =
            std::accumulate(allVariantWorkloadSizes.begin() + begin, allVariantWorkloadSizes.begin() + end, int64_t{0},
                            accumulateAlignedElements);

    const auto tableType = mlir::cast<vpux::NDTypeInterface>(table.getType());
    const auto chunkShape = VPU::NCESparsity::inferWeightsTableShape(chunkElements, /*newFormat=*/true);
    const auto chunkType = tableType.changeShape(ShapeRef(chunkShape));

    auto declOp = table.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(declOp == nullptr, "Expected VPURT.DeclareBufferOp for data-pointer table", loc);
    const auto absoluteByteOffset = declOp.getByteOffset() + chunkByteOffset;
    auto chunkBuffer = builder.create<VPURT::DeclareBufferOp>(
            loc, chunkType, declOp.getSectionAttr(), declOp.getSectionIndexAttr(),
            getIntAttr(builder, absoluteByteOffset), declOp.getSwizzlingKeyAttr());
    return chunkBuffer.getBuffer();
}

//
// SplitLargeInvariantsPass
//
// This pass mitigates runtime problem of handling preemption in case descriptor group contains only single invariant.
// It splits VPURT.Task-wrapped VPUIP.NCEClusterTask operations
// when their number of DPU variants exceeds the architecture-specific threshold
// derived from NPU constraints.
// New VPURT.Tasks are created such that each doesn't exceed the threshold anymore and overall workload stays the same:
//  - all the outputs and inputs of the new NCEClusterTasks refer to the original ones
//  - all VPURT.Tasks are chained by barriers to preserve the order
//  - the first task waits on the original wait barriers and the last task updates the original update barriers
//
// EXAMPLE: 8 variants with threshold=4
//
//                 BEFORE                                            AFTER
// [AwaitBarrier0] [...] [AwaitBarrierX]             [AwaitBarrier0] [...] [AwaitBarrierX]
//          |        |         |                              |        |         |
// ┌─────────────────────────────────────┐          ┌─────────────────────────────────────┐
// │  VPURT.Task                         │          │  VPURT.Task #0                      │
// │ ┌─────────────────────────────────┐ │          │ ┌─────────────────────────────────┐ │
// │ │  [Input0]   [...]    [InputN]   │ │          │ │  [Input0]   [...]    [InputN]   │ │
// │ │       │       │        │        │ │          │ │       │       │        │        │ │
// │ │   ┌─────────────────────────┐   │ │          │ │   ┌─────────────────────────┐   │ │
// │ │   │ NCEClusterTask          │   │ │          │ │   │ NCEClusterTask #0       │   │ │
// │ │   │─────────────────────────│   │ │          │ │   │─────────────────────────│   │ │
// │ │   │ 8 DPU variants (V0-V7)  │   │ │          │ │   │ 4 DPU variants (V0-V3)  │   │ │
// │ │   │ V0 V1 V2 V3 V4 V5 V6 V7 │   │ │          │ │   │      V0 V1 V2 V3        │   │ │
// │ │   └─────────────────────────┘   │ │          │ │   └─────────────────────────┘   │ │
// │ │      │        │        │        │ │          │ │      │        │        │        │ │
// │ │  [Output0]  [...]  [OutputM]    │ │          │ │  [Output0]  [...]  [OutputM]    │ │
// │ └─────────────────────────────────┘ │          │ └─────────────────────────────────┘ │
// └─────────────────────────────────────┘          └─────────────────┼───────────────────┘
//         |          |          |                                    |
// [UpdateBarrier0] [...] [UpdateBarrierY]                            |
//                                                         [VPURT.VirtualBarrier]
//                                                 (updated by Task #0, awaited by Task #1)
//                                                                    |
//                                                                    v
//                                                  ┌─────────────────┼───────────────────┐
//                                                  │  VPURT.Task #1                      │
//                                                  │ ┌─────────────────────────────────┐ │
//                                                  │ │  [Input0]   [...]    [InputN]   │ │
//                                                  │ │       │       │        │        │ │
//                                                  │ │   ┌─────────────────────────┐   │ │
//                                                  │ │   │ NCEClusterTask #1       │   │ │
//                                                  │ │   │─────────────────────────│   │ │
//                                                  │ │   │ 4 DPU variants (V4-V7)  │   │ │
//                                                  │ │   │       V4 V5 V6 V7       │   │ │
//                                                  │ │   └─────────────────────────┘   │ │
//                                                  │ │      │        │        │        │ │
//                                                  │ │  [Output0]  [...]  [OutputM]    │ │
//                                                  │ └─────────────────────────────────┘ │
//                                                  └─────────────────┼───────────────────┘
//                                                         |          |          |
//                                                    [UpdateBarrier0] [...] [UpdateBarrierY]
//
// When data-pointer table is present, its single buffer attached to the original NCEClusterTask contains one section
// per variant, laid out contiguously and padded to the nearest multiple of WEIGHTS_TABLE_READER_ALIGNMENT (64)
// elements. The backend implementation in compiler, when calculating the first weight_d_ptr_start register for the
// first section of data-pointer table in each NCEClusterTask starts counting offset from 0. This is because there's no
// explicit way to get an aligned offset to a start of a concrete variant's data-pointer table. The size of a workload
// can be different in each variant, let's say for variant 0, the aligned logical element size is 256 OC elements, when
// for variant 31 it could be 64 OC elements. There's no universal formula to calculate such offset, so iterative
// approach is used. Backend implementation goes through each variant and sums up an offset from the previous variant
// with the current's. See more in
// src/vpux_compiler/src/conversion/rewriters/VPUIP2VPUMI40XX/nce_cluster_task_rewriter.cpp. Since after the split, the
// second chunk's variants are renumbered starting from 0, they would read sections 0, 1, 2, ... - which are the first
// chunk's sections. To fix this, each chunk receives a VPURT.DeclareBufferOp sub-view buffer whose CMX byte offset is
// advanced by the cumulative size of all sections belonging to the preceding chunks.

class SplitLargeInvariantsPass final : public impl::SplitLargeInvariantsBase<SplitLargeInvariantsPass> {
public:
    explicit SplitLargeInvariantsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void safeRunOnFunc() final;

private:
    static void keepOnlyVariantChunk(VPUIP::NCEClusterTaskOp nceTask, int64_t begin, int64_t end);
    static void splitNceTaskByVariants(VPURT::TaskOp taskOp, int64_t variantsPerChunk);
};

void SplitLargeInvariantsPass::keepOnlyVariantChunk(VPUIP::NCEClusterTaskOp nceTask, int64_t begin, int64_t end) {
    SmallVector<VPUIP::DPUTaskOp> variants = llvm::to_vector(nceTask.getVariants().getOps<VPUIP::DPUTaskOp>());

    // When this NCE task uses a sprLUT / palletLUT, InsertDelayDPUVariant has prepended a dummy
    // DPUTask at variants[0]. The dummy is required by the LUT-load FSM (it waits until the LUT
    // DMA barriers are produced before the first real variant runs). Splitting the invariant
    // into chunks must keep that dummy at the front of every chunk, otherwise post-split chunks
    // (begin > 0) would consume sprLutRead/forceInvRead semantics on a real variant, and the
    // LUT-load wait would be skipped — a latent precision issue plus a downstream crash in
    // NCEClusterTaskRewriter when the chunk has a single variant.
    const bool hasLut = nceTask.getSprLookupTable() != nullptr || nceTask.getPalletLookupTable() != nullptr;
    const bool needCloneDummy = hasLut && begin > 0 && !variants.empty();

    for (int64_t idx = 0; idx < static_cast<int64_t>(variants.size()); ++idx) {
        if (idx >= begin && idx < end) {
            continue;
        }
        // Preserve the dummy variant at idx==0 for chunks that need it; we only delete the rest.
        if (needCloneDummy && idx == 0) {
            continue;
        }
        variants[idx].erase();
    }

    if (needCloneDummy) {
        // Move the dummy to the front of the kept chunk so it is variants[0] of this chunk too.
        // (It is already at the front of the variants region, but ensure ordering is preserved.)
        auto& variantsBlock = nceTask.getVariants().front();
        auto* dummyOp = variants[0].getOperation();
        if (&variantsBlock.front() != dummyOp) {
            dummyOp->moveBefore(&variantsBlock.front());
        }
    }

    int64_t workloadId = 0;
    for (auto variant : nceTask.getVariants().getOps<VPUIP::DPUTaskOp>()) {
        if (!variant.getWorkloadId().has_value()) {
            continue;
        }
        variant.setWorkloadIdAttr(getIntAttr(nceTask.getContext(), workloadId++));
    }
}

void SplitLargeInvariantsPass::splitNceTaskByVariants(VPURT::TaskOp taskOp, int64_t variantsPerChunk) {
    auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
    VPUX_THROW_WHEN(nceTask == nullptr, "Expected VPURT task to wrap VPUIP.NCEClusterTask");

    const auto numVariants = nceTask.getNumVariants();
    const auto numChunks = (numVariants + variantsPerChunk - 1) / variantsPerChunk;
    if (numChunks <= 1) {
        return;
    }

    VPUX_THROW_UNLESS(
            nceTask.getWeightZeroPoints() == nullptr,
            "NCEClusterTask with zero-point table must have at most 1 variant per invariant, but got {0} at '{1}'",
            numVariants, nceTask.getLoc());
    VPUX_THROW_UNLESS(nceTask.getWeightTableSpPtr() == nullptr, "Sparsity-pointer table is not supported!");

    SmallVector<mlir::Value> originalWaitBarriers(taskOp.getWaitBarriers().begin(), taskOp.getWaitBarriers().end());
    SmallVector<mlir::Value> originalUpdateBarriers(taskOp.getUpdateBarriers().begin(),
                                                    taskOp.getUpdateBarriers().end());

    const auto oldProfilingData = nceTask.getProfilingData();
    const auto oldProfilingMetadata = nceTask.getProfilingMetadataAttr();
    if (oldProfilingData != nullptr) {
        VPUX_THROW_WHEN(oldProfilingMetadata == nullptr,
                        "Expected profiling metadata for split NCEClusterTask with profiling data at '{0}'",
                        nceTask.getLoc());
    }

    uint16_t profilingWorkloadSizeBytes = 0;
    if (oldProfilingData != nullptr) {
        profilingWorkloadSizeBytes = VPUIP::getProfWorkloadSize(taskOp->getParentOfType<mlir::ModuleOp>());
    }

    const auto oldDataPointerTable = nceTask.getWeightTableDataPtr();
    SmallVector<int32_t> variantWorkloadSizes;
    if (oldDataPointerTable != nullptr) {
        for (auto variant : nceTask.getVariants().getOps<VPUIP::DPUTaskOp>()) {
            const int32_t cStart =
                    checked_cast<int32_t>(mlir::cast<mlir::IntegerAttr>(variant.getOutStart()[2]).getInt());
            const int32_t cEnd = checked_cast<int32_t>(mlir::cast<mlir::IntegerAttr>(variant.getOutEnd()[2]).getInt());
            variantWorkloadSizes.push_back(cEnd - cStart + 1);
        }
    }

    mlir::OpBuilder builder(taskOp);

    SmallVector<mlir::Value> chainBarriers;
    chainBarriers.reserve(static_cast<size_t>(numChunks - 1));
    for (int64_t idx = 0; idx < numChunks - 1; ++idx) {
        auto chainBarrierOp = builder.create<VPURT::DeclareVirtualBarrierOp>(taskOp->getLoc());
        chainBarriers.push_back(chainBarrierOp.getResult());
    }

    for (int64_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
        auto clonedTaskOp = mlir::cast<VPURT::TaskOp>(builder.clone(*taskOp.getOperation()));
        auto clonedNceTask = mlir::cast<VPUIP::NCEClusterTaskOp>(clonedTaskOp.getInnerTaskOp());

        const auto begin = chunkIdx * variantsPerChunk;
        const auto end = std::min(begin + variantsPerChunk, numVariants);
        keepOnlyVariantChunk(clonedNceTask, begin, end);
        const auto numVariantsInChunk = end - begin;

        if (oldProfilingData != nullptr) {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(clonedTaskOp);
            auto profilingChunkBuffer = getProfilingChunkBuffer(builder, clonedNceTask.getLoc(), oldProfilingData,
                                                                begin, numVariantsInChunk, profilingWorkloadSizeBytes);
            clonedNceTask.getProfilingDataMutable().assign(profilingChunkBuffer);
            // Keep profiling output type in sync with profiling_data operand type.
            int64_t profilingResultIndex = 0;
            if (clonedNceTask.getOutputBuff() != nullptr) {
                profilingResultIndex++;
            }
            if (clonedNceTask.getOutputSparsityMapBuff() != nullptr) {
                profilingResultIndex++;
            }
            clonedNceTask->getResult(profilingResultIndex).setType(profilingChunkBuffer.getType());
            const auto taskId = checked_cast<uint32_t>(oldProfilingMetadata.getTaskId().getInt() + chunkIdx);
            clonedNceTask.setProfilingMetadataAttr(
                    updateDpuProfilingMetadata(oldProfilingMetadata, numVariantsInChunk, taskId));
        }

        if (!variantWorkloadSizes.empty()) {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(clonedTaskOp);
            const auto chunkBuffer = getDataPointerTableChunkBuffer(
                    builder, clonedNceTask.getLoc(), oldDataPointerTable, variantWorkloadSizes, begin, end);
            clonedNceTask.getWeightTableDataPtrMutable().assign(chunkBuffer);
        }

        SmallVector<mlir::Value> waitBarriers;
        SmallVector<mlir::Value> updateBarriers;

        if (chunkIdx == 0) {
            waitBarriers = originalWaitBarriers;
        } else {
            waitBarriers.push_back(chainBarriers[static_cast<size_t>(chunkIdx - 1)]);
        }

        if (chunkIdx == numChunks - 1) {
            updateBarriers = originalUpdateBarriers;
        } else {
            updateBarriers.push_back(chainBarriers[static_cast<size_t>(chunkIdx)]);
        }

        clonedTaskOp.getWaitBarriersMutable().assign(waitBarriers);
        clonedTaskOp.getUpdateBarriersMutable().assign(updateBarriers);

        // Keep cloned tasks in chunk order.
        builder.setInsertionPointAfter(clonedTaskOp);
    }

    taskOp.erase();

    // The original full-sized data-pointer table DeclareBufferOps are no longer referenced after all clones have
    // been rebased to per-chunk sub-buffers. Erase them.
    if (oldDataPointerTable == nullptr) {
        return;
    }
    if (auto declOp = oldDataPointerTable.getDefiningOp<VPURT::DeclareBufferOp>()) {
        if (declOp->use_empty()) {
            declOp.erase();
        }
    }
}

void SplitLargeInvariantsPass::safeRunOnFunc() {
    auto func = getOperation();
    const auto useVariantCount = maxVariantCount.hasValue();
    const auto variantsPerNceClusterTask = useVariantCount ? checked_cast<size_t>(maxVariantCount.getValue()) / 2
                                                           : getMaxNumberOfDpuVariantsPerInvariant(func) / 2;
    VPUX_THROW_UNLESS(variantsPerNceClusterTask >= 1,
                      "Invalid NCEClusterTask split threshold: computed variants-per-task threshold must be >= 1, "
                      "but got {0}",
                      variantsPerNceClusterTask);

    SmallVector<VPURT::TaskOp> tasksToSplit;

    func.walk([&](VPURT::TaskOp taskOp) {
        auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
        if (nceTask == nullptr) {
            return;
        }

        if (nceTask.getNumVariants() > (int64_t)variantsPerNceClusterTask) {
            tasksToSplit.push_back(taskOp);
        }
    });

    if (tasksToSplit.empty()) {
        _log.trace("No NCEClusterTaskOps with variants count exceeding the threshold of {0} were found",
                   variantsPerNceClusterTask);
        return;
    }

    _log.trace("Found {0} NCEClusterTaskOps with variants count exceeding the threshold of {1}, splitting them",
               tasksToSplit.size(), variantsPerNceClusterTask);
    auto nestedLog = _log.nest();
    for (auto taskOp : tasksToSplit) {
        auto nceTask = mlir::cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
        nestedLog.trace("Splitting NCEClusterTaskOp at '{0}' with '{1}' variants into '{2}' NCEClusterTaskOps",
                        nceTask.getLoc(), nceTask.getNumVariants(),
                        (nceTask.getNumVariants() + variantsPerNceClusterTask - 1) / variantsPerNceClusterTask);
        splitNceTaskByVariants(taskOp, variantsPerNceClusterTask);
    }

    normalizeDpuProfilingTaskIds(func);
}

}  // namespace

std::unique_ptr<mlir::Pass> createSplitLargeInvariantsPass(Logger log) {
    return std::make_unique<SplitLargeInvariantsPass>(log);
}

}  // namespace vpux::VPUIP
