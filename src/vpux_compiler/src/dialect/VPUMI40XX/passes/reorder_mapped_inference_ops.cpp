//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <array>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_REORDERMPIOPS
#define GEN_PASS_DEF_REORDERMPIOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {
using TaskType = VPURegMapped::TaskType;

class ReorderMPIOpsPass : public VPUMI40XX::impl::ReorderMPIOpsBase<ReorderMPIOpsPass> {
public:
    explicit ReorderMPIOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

mlir::Operation* moveOrCloneOp(mlir::Operation* op, mlir::OpBuilder& builder) {
    const auto sameBlock = op->getBlock() == builder.getInsertionBlock();
    const auto moveSafe = sameBlock && mlir::isMemoryEffectFree(op);

    if (moveSafe) {
        op->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
        return op;
    }

    auto* clonedOp = builder.clone(*op);
    op->replaceAllUsesWith(clonedOp);
    op->erase();
    return clonedOp;
}

template <typename OpT, typename Functor = vpux::FuncRef<bool(OpT)>>
mlir::Operation* linearizeOps(
        mlir::func::FuncOp func, mlir::OpBuilder& builder, Functor&& condition = [](OpT) {
            return true;
        }) {
    // Collect first to avoid iterator invalidation while moving ops.
    // Reserve a small default to avoid repeated growth for common cases.
    constexpr size_t expectedOpsToMove = 64;
    vpux::SmallVector<mlir::Operation*> opsToMove;
    opsToMove.reserve(expectedOpsToMove);
    for (auto op : func.getOps<OpT>()) {
        if (condition(op)) {
            opsToMove.push_back(op.getOperation());
        }
    }

    mlir::Operation* lastOp = nullptr;
    for (auto* op : opsToMove) {
        lastOp = moveOrCloneOp(op, builder);

        builder.setInsertionPointAfter(lastOp);
    }

    return lastOp;
}

VPUMI40XX::OpRanges getRequiredOpRanges(mlir::func::FuncOp func) {
    auto opRanges = func.getOps<VPUMI40XX::OpRanges>();
    VPUX_THROW_UNLESS(!opRanges.empty(),
                      "Missing VPUMI40XX::OpRanges operation in function '{0}'; "
                      "ensure the op_ranges pass runs before reorder-mapped-inference-ops",
                      func.getName());
    return *opRanges.begin();
}

SmallVector<size_t> collectOrderedTiles(VPUMI40XX::OpRanges opRanges) {
    SmallVector<size_t> tiles;
    tiles.reserve(opRanges.getBegins().size());
    for (auto begin : opRanges.getBegins()) {
        const auto tile = checked_cast<size_t>(mlir::cast<vpux::VPURegMapped::IndexType>(begin.getType()).getTileIdx());
        const auto pos = std::lower_bound(tiles.begin(), tiles.end(), tile);
        if (pos == tiles.end() || *pos != tile) {
            tiles.insert(pos, tile);
        }
    }
    return tiles;
}

SmallVector<size_t> collectOrderedLists(VPUMI40XX::OpRanges opRanges, TaskType taskType, size_t tileIdx) {
    SmallVector<size_t> lists;
    const auto types = opRanges.getTypes();
    const auto begins = opRanges.getBegins();
    lists.reserve(begins.size());
    VPUX_THROW_UNLESS(types.size() == begins.size(),
                      "OpRanges 'types' and 'begins' arrays have different sizes: {0} vs {1}", types.size(),
                      begins.size());
    for (const auto idx : irange(types.size())) {
        if (mlir::cast<VPURegMapped::TaskTypeAttr>(types[idx]).getValue() != taskType) {
            continue;
        }

        const auto beginIndex = mlir::cast<vpux::VPURegMapped::IndexType>(begins[idx].getType());
        if (checked_cast<size_t>(beginIndex.getTileIdx()) == tileIdx) {
            const auto listIdx = checked_cast<size_t>(beginIndex.getListIdx());
            const auto pos = std::lower_bound(lists.begin(), lists.end(), listIdx);
            if (pos == lists.end() || *pos != listIdx) {
                lists.insert(pos, listIdx);
            }
        }
    }
    return lists;
}

SmallVector<size_t> collectDMAListsFromTaskBuffers(mlir::func::FuncOp func, size_t tileIdx) {
    SmallVector<size_t> lists;
    for (auto op : func.getOps<VPUMI40XX::DeclareTaskBufferOp>()) {
        if (op.getTaskType() != TaskType::DMA) {
            continue;
        }
        const auto index = mlir::cast<vpux::VPURegMapped::IndexType>(op.getIndex().getType());
        if (checked_cast<size_t>(index.getTileIdx()) == tileIdx) {
            const auto listIdx = checked_cast<size_t>(index.getListIdx());
            const auto pos = std::lower_bound(lists.begin(), lists.end(), listIdx);
            if (pos == lists.end() || *pos != listIdx) {
                lists.insert(pos, listIdx);
            }
        }
    }
    return lists;
}

// DeclareTaskBuffer ordering is particularly hot (many repeated scans). Bucket once and then
// emit/move in the required firmware order.
void linearizeDeclareTaskBufferOps(mlir::func::FuncOp func, mlir::OpBuilder& builder, VPUMI40XX::OpRanges opRanges) {
    auto makeKey = [](size_t tileIdx, size_t listIdx, TaskType type) -> uint64_t {
        // Pack into a stable key: [ tileIdx (16) | listIdx (16) | taskType (32) ]
        return (static_cast<uint64_t>(tileIdx) << 48) | (static_cast<uint64_t>(listIdx) << 32) |
               static_cast<uint32_t>(type);
    };

    // Reserve for expected unique (tileIdx, listIdx, taskType) combinations.
    // Rule of thumb: VPU_MAX_TILES * expected_list_count * expected_task_types.
    // 256 keeps rehashing low for typical firmware list widths and task mix.
    constexpr size_t expectedBucketCount = 256;

    vpux::DenseMap<uint64_t, vpux::SmallVector<mlir::Operation*>> buckets;
    buckets.reserve(expectedBucketCount);

    // Bucket all DeclareTaskBufferOp once in original order.
    for (auto op : func.getOps<VPUMI40XX::DeclareTaskBufferOp>()) {
        const auto index = mlir::cast<vpux::VPURegMapped::IndexType>(op.getIndex().getType());
        const auto tileIdx = checked_cast<size_t>(index.getTileIdx());
        const auto listIdx = checked_cast<size_t>(index.getListIdx());
        const auto type = op.getTaskType();

        buckets[makeKey(tileIdx, listIdx, type)].push_back(op.getOperation());
    }

    auto moveBucket = [&](size_t tileIdx, size_t listIdx, TaskType type) {
        const auto key = makeKey(tileIdx, listIdx, type);
        auto it = buckets.find(key);
        if (it == buckets.end()) {
            return;
        }
        for (auto* op : it->second) {
            auto* lastOp = moveOrCloneOp(op, builder);
            builder.setInsertionPointAfter(lastOp);
        }
    };

    auto tileOrder = collectOrderedTiles(opRanges);

    for (const auto tileIndex : tileOrder) {
        moveBucket(tileIndex, 0, TaskType::DPUInvariant);
        moveBucket(tileIndex, 0, TaskType::DPUVariant);

        auto actKernelRangeLists = collectOrderedLists(opRanges, TaskType::ActKernelRange, tileIndex);
        auto actKernelInvoLists = collectOrderedLists(opRanges, TaskType::ActKernelInvocation, tileIndex);
        // Fall back to the peer type's list indices when one task type is absent
        if (actKernelRangeLists.empty()) {
            actKernelRangeLists = actKernelInvoLists;
        }
        if (actKernelInvoLists.empty()) {
            actKernelInvoLists = actKernelRangeLists;
        }

        for (const auto listIdx : actKernelRangeLists) {
            moveBucket(tileIndex, listIdx, TaskType::ActKernelRange);
        }
        for (const auto listIdx : actKernelInvoLists) {
            moveBucket(tileIndex, listIdx, TaskType::ActKernelInvocation);
        }

        auto dmaLists = collectDMAListsFromTaskBuffers(func, tileIndex);
        for (const auto listIdx : dmaLists) {
            moveBucket(tileIndex, listIdx, TaskType::DMA);
        }
    }
}

// Linearize firmware-visible DeclareBufferOp sections in one traversal.
// Only the sections that require deterministic ordering for firmware are bucketed here.
void linearizeDeclareBufferOpsBySection(mlir::func::FuncOp func, mlir::OpBuilder& builder) {
    constexpr std::array<VPURT::BufferSection, 7> kSectionOrder = {
            VPURT::BufferSection::NetworkInput,    VPURT::BufferSection::NetworkOutput,
            VPURT::BufferSection::ProfilingOutput, VPURT::BufferSection::DDR,
            VPURT::BufferSection::CMX_NN,          VPURT::BufferSection::MAC_Accumulators,
            VPURT::BufferSection::Register,
    };

    const auto getBucketIndex = [&](VPURT::BufferSection section) -> size_t {
        for (size_t idx = 0; idx < kSectionOrder.size(); ++idx) {
            if (kSectionOrder[idx] == section) {
                return idx;
            }
        }

        return kSectionOrder.size();
    };

    std::array<vpux::SmallVector<mlir::Operation*>, kSectionOrder.size()> sectionOps = {};

    // Single scan: bucket ops by explicit section order. Unsupported sections are intentionally left in place.
    for (auto op : func.getOps<VPURT::DeclareBufferOp>()) {
        const auto bucketIdx = getBucketIndex(op.getSection());
        if (bucketIdx < kSectionOrder.size()) {
            sectionOps[bucketIdx].push_back(op.getOperation());
        }
    }

    // Emit buckets in the explicit firmware-facing section sequence.
    for (size_t sectionIdx = 0; sectionIdx < kSectionOrder.size(); ++sectionIdx) {
        for (auto* op : sectionOps[sectionIdx]) {
            auto* lastOp = moveOrCloneOp(op, builder);
            builder.setInsertionPointAfter(lastOp);
        }
    }
}

// Collect all NNDMAOps by engine in one traversal.
// Bucket ops by engine once, then emit them in deterministic engine order.
// The engine count is provided by the IR config (available DMA executors), so do not
// duplicate architecture policy in this pass.
void linearizeNNDMAOpsByEngine(mlir::func::FuncOp func, mlir::OpBuilder& builder) {
    const auto engineCount = checked_cast<size_t>(config::getNumOfDMAPorts(func.getOperation()));
    if (engineCount == 0) {
        return;
    }

    vpux::SmallVector<vpux::SmallVector<mlir::Operation*>> engineOps(engineCount);
    for (auto op : func.getOps<VPUMI40XX::NNDMAOp>()) {
        const auto engineIdx = checked_cast<size_t>(op.getPort());
        if (engineIdx >= engineCount) {
            continue;
        }
        engineOps[engineIdx].push_back(op.getOperation());
    }

    for (size_t engineIdx = 0; engineIdx < engineCount; ++engineIdx) {
        for (auto* op : engineOps[engineIdx]) {
            auto* lastOp = moveOrCloneOp(op, builder);
            builder.setInsertionPointAfter(lastOp);
        }
    }
}

void ReorderMPIOpsPass::safeRunOnFunc() {
    auto func = getOperation();
    auto builder = mlir::OpBuilder::atBlockBegin(&func.getBody().front());
    auto opRanges = getRequiredOpRanges(func);

    // When ordering these function calls take care of the dominance relationships.
    // Ideally this should be a canonicalizer-like thing, but re-ordering ops via successive pattern rewrites
    // seems overkill.
    // Since VPUMI40XX is a highly controlled dialect in terms of the forms it can take, manual implicit good
    // ordering of what to linearize first to not break dominance should be ok.

    // order of DeclareTaskBuffer is important as it must be aligned with firmware expectations
    // tile0: DPUInvariant -> DPUVariant -> Ranges -> Invocations -> DMA from DDR -> DMA from CMX
    // tile1: DPUInvariant -> DPUVariant -> Ranges -> Invocations -> DMA from DDR -> DMA from CMX
    // ...
    linearizeDeclareTaskBufferOps(func, builder, opRanges);

    linearizeOps<Const::DeclareOp>(func, builder);

    // Linearize firmware-visible DeclareBufferOp sections in a single IR scan.
    linearizeDeclareBufferOpsBySection(func, builder);

    linearizeOps<VPUMI40XX::DeclareKernelTextOp>(func, builder);
    linearizeOps<VPUMI40XX::DeclareKernelEntryOp>(func, builder);
    linearizeOps<VPUMI40XX::DeclareKernelArgsOp>(func, builder);
    linearizeOps<VPUMI40XX::KernelParamsOp>(func, builder);

    linearizeOps<VPUMI40XX::ConfigureBarrierOp>(func, builder);

    linearizeOps<VPUMI40XX::ActKernelRangeOp>(func, builder);
    linearizeOps<VPUMI40XX::ActKernelInvocationOp>(func, builder);

    linearizeOps<VPUMI40XX::DPUInvariantOp>(func, builder);
    linearizeOps<VPUMI40XX::DPUVariantOp>(func, builder);

    linearizeOps<VPURegMapped::ViewTaskRangeOp>(func, builder);

    linearizeNNDMAOpsByEngine(func, builder);
}

}  // namespace

//
// reorderMappedInferenceOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::reorderMappedInferenceOpsPass(Logger log) {
    return std::make_unique<ReorderMPIOpsPass>(log);
}
