//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_UPDATEFETCHDMAFORSKIPDMAS
#define GEN_PASS_DEF_UPDATEFETCHDMAFORSKIPDMAS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;
using DMAMap = llvm::DenseMap<int64_t, VPUMI40XX::NNDMAOp>;
using TileListToOffsetMap = llvm::DenseMap<std::pair<size_t, size_t>, uint64_t>;
using TileListToIndexMap = llvm::DenseMap<std::pair<size_t, size_t>, size_t>;

struct PlannedFetch {
    int64_t descId;
    VPUMI40XX::NNDMAOp fetchOp;

    // Store them to ensure we don't need lookup for sorting fetchDMAs
    int64_t tile;
    int64_t logical;
    int64_t list;
    int64_t memPriority;
    int64_t port;
};

namespace {

class UpdateFetchDMAForSkipDMAs : public VPUMI40XX::impl::UpdateFetchDMAForSkipDMAsBase<UpdateFetchDMAForSkipDMAs> {
public:
    explicit UpdateFetchDMAForSkipDMAs(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void collectFetchAndSkipDMAs(DMAMap& fetchMap, DMAMap& skipMap);

    uint64_t getRegionOffset(VPURegMapped::TaskType target, config::ArchKind arch, const uint64_t& shaveCountPerTile);
    mlir::MemRefType createFetchOutputMemRefs(mlir::OpBuilder& builder, uint32_t tile, size_t dmaTaskBinarySize);
    VPUMI40XX::DeclareTaskBufferOp createTaskBuffer(mlir::OpBuilder& builder, PlannedFetch plannedFetch,
                                                    mlir::Operation* insertionPoint,
                                                    TileListToIndexMap& taskBufferIndex,
                                                    TileListToOffsetMap& taskBufferOffset, size_t dmaTaskBinarySize,
                                                    uint64_t dmaMetadataOffset);

private:
    void safeRunOnFunc() final;
};

// CMX Metadata Layout
//
// CMX metadata is organized as a sequence of sections in memory.
// Each section occupies a contiguous region, and later sections are placed
// after earlier ones.
//
// Layout (low -> high addresses):
//
//  +------------------------+
//  | DPU Invar Descriptors  |
//  +------------------------+
//  | DPU Variant Descriptors|
//  +------------------------+
//  | SHV Kernel Descriptors |
//  +------------------------+
//  | SAV Invo Descriptors   |
//  +------------------------+
//  |    M2I Descriptors     |
//  +------------------------+
//  | DMA Descriptors        |
//  +------------------------+
//
// The DMA metadata region is placed after all previous metadata sections.
// Its starting offset is fixed at compile time and must account for both
// SHAVE lists per tile.
//
//
// Example (not to scale):
//
//  Address
//    |
//    v
//    0x0000  +-----------------------------+
//            | DPU Invariant Descriptors   |
//            | 64 * 352 = 22528 bytes      |
//  0x5800    +-----------------------------+
//            | DPU Variant Descriptors     |
//            | 128 * 224 = 28672 bytes     |
//  0xC800    +-----------------------------+
//            | SHV Kernel Range, list 0    |
//            | 32 * 40 = 1280 bytes        |
//  0xCD00    +-----------------------------+
//            | SHV Invocation, list 0      |
//            | 32 * 96 = 3072 bytes        |
//  0xD900    +-----------------------------+
//            | SHV Kernel Range, list 1    |
//            | 32 * 40 = 1280 bytes        |
//  0xDE00    +-----------------------------+
//            | SHV Invocation, list 1      |
//            | 32 * 96 = 3072 bytes        |
//  0xEA00    +-----------------------------+
//            | M2I Descriptors             |
//            | 4 * 240 = 960 bytes         |
//  0xEDC0    +-----------------------------+  <-- DMA metadata start (dmaMetadataOffset)
//            | DMA Descriptor 0            |
//            +-----------------------------+
//            | DMA Descriptor 1            |
//            +-----------------------------+
//            | ...                         |
//            +-----------------------------+
//
// We use a constant `dmaMetadataOffset` to point to the beginning of the DMA
// metadata section. All DMA task descriptors are allocated sequentially from
// this base offset.
//
// Each new DMA increments the offset by:
//   VPUMI40XX::getTaskBinarySize(VPURegMapped::TaskType::DMA, ...)
//
// This ensures:
//   - DMA descriptors do not overlap with earlier metadata sections
//   - DMA descriptors form a contiguous block in CMX
//   - Deterministic layout across tiles and logical tasks
// Note: We don't use all of available DMA space, only SHV tasks running in parallel have their associated Skips fetched
// at a time and then replaced by other skips as we move through logical tasks
//

//
// CMX Metadata Layout for Skip DMA Descriptors in CMX
//
// Each tile contains multiple SHV execution contexts (Act_SHV denoted by list). Each list will
// have skip DMAs targeting both DDR and CMX memory channel
//
// Descriptor layout in CMX metadata is organized as:
//
//   for each tile:
//     for each list:
//       DDR descriptor
//       CMX descriptor
//
// Example:
//
//   Tile 0 List 0 DDR -> Tile 0 List 0 CMX ->
//   Tile 0 List 1 DDR -> Tile 0 List 1 CMX -> ...
//
// Descriptors for a given (tile, list) are always placed contiguously.
// If the base address of (Tile 0, List 0, DDR) is X, then:
//
//   (Tile 0, List 0, DDR) = X
//   (Tile 0, List 0, CMX) = X + sizeof(DMA descriptor)
//
// Rationale:
//
// This is critical because SHV lists may not be populated. For example,
// if a workload does not utilize all lists on a tile (e.g. only List 0 is used
// and List 1 has no tasks), then:
//
//   Good layout:
//     [T0 L0 DDR][T0 L0 CMX]   // contiguous, no gaps
//
//   Fragmented layout (if grouped by memory type):
//     [T0 L0 DDR][T0 L1 DDR][T0 L0 CMX][T0 L1 CMX]
//
// In the fragmented case, even if List 1 has no tasks, space is still reserved
// for its descriptors between DDR and CMX regions. As a result, descriptors
// for a single active list are no longer contiguous in memory.
//
// This introduces:
//   - gaps between related descriptors
//   - additional offset computation or conditional handling
//   - inefficient memory access patterns
//
// With the chosen layout, each (tile, list) forms a compact block, and unused
// lists do not introduce fragmentation within that block.
//
// Offset assignment in this pass strictly follows this layout.
uint64_t UpdateFetchDMAForSkipDMAs::getRegionOffset(VPURegMapped::TaskType target, config::ArchKind arch,
                                                    const uint64_t& shaveCountPerTile) {
    struct Entry {
        VPURegMapped::TaskType type;
        size_t count;
    };
    auto netFunc = getOperation();
    auto invCount = config::getConstraint<size_t>(netFunc, config::METADATA_MAX_INVARIANT_COUNT);
    auto varCount = config::getConstraint<size_t>(netFunc, config::METADATA_MAX_VARIANT_COUNT);
    auto rangeCount = config::getConstraint<size_t>(netFunc, config::METADATA_MAX_KERNEL_RANGE_COUNT);
    auto invoCount = config::getConstraint<size_t>(netFunc, config::METADATA_MAX_KERNEL_INVOCATION_COUNT);
    auto dmaCount = config::getConstraint<size_t>(netFunc, config::METADATA_MAX_DMA_COUNT);
    auto m2iCount = config::getConstraint<size_t>(netFunc, config::METADATA_MAX_MEDIA_COUNT);

    Entry layout[] = {
            {VPURegMapped::TaskType::DPUInvariant, invCount},
            {VPURegMapped::TaskType::DPUVariant, varCount},
            {VPURegMapped::TaskType::ActKernelRange, rangeCount * shaveCountPerTile},
            {VPURegMapped::TaskType::ActKernelInvocation, invoCount * shaveCountPerTile},
            {VPURegMapped::TaskType::M2I, m2iCount},
            {VPURegMapped::TaskType::DMA, dmaCount},
    };

    uint64_t offset = 0;
    for (const auto& entry : layout) {
        if (entry.type == target) {
            return offset;
        }

        offset += checked_cast<uint64_t>(entry.count) *
                  checked_cast<uint64_t>(VPUMI40XX::getTaskBinarySize(entry.type, arch));
    }

    VPUX_THROW("Task Type {0} not registered in metadata layout", target);
}

mlir::MemRefType UpdateFetchDMAForSkipDMAs::createFetchOutputMemRefs(mlir::OpBuilder& builder, uint32_t tile,
                                                                     size_t dmaTaskBinarySize) {
    const auto memSpaceCMX =
            vpux::IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), tile);

    auto memrefDDR =
            vpux::getMemRefType({1, static_cast<int64_t>(dmaTaskBinarySize)}, builder.getIntegerType(8, false));
    auto memrefCMX =
            mlir::cast<mlir::MemRefType>(mlir::cast<vpux::NDTypeInterface>(memrefDDR).changeMemSpace(memSpaceCMX));
    return memrefCMX;
}

mlir::Value createFetchBuffer(mlir::OpBuilder& builder, size_t dmaTaskBinarySize) {
    auto ctx = builder.getContext();

    const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
    const auto zeroBufferMemref = vpux::getMemRefType({1, static_cast<int64_t>(dmaTaskBinarySize)}, builder.getI8Type(),
                                                      DimsOrder::NC, symbolAttr);

    const auto sectionAttr =
            VPURT::BufferSectionAttr::get(builder.getContext(), VPURT::getBufferSection(VPU::MemoryKind::DDR));

    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), zeroBufferMemref, sectionAttr,
                                                  /*sectionIndex=*/nullptr, /*byteOffset=*/getIntAttr(builder, 0),
                                                  /*swizzlingKey=*/nullptr);
}

// ------------------------------------------------------
// Enforce phase-based ordering: list then DDR -> CMX
// ------------------------------------------------------
SmallVector<PlannedFetch> getOrderedFetchList(const DMAMap& fetchMap, const DMAMap& skipMap, Logger& log) {
    SmallVector<PlannedFetch> orderedFetches;
    orderedFetches.reserve(fetchMap.size());
    for (const auto& [key, fetchOp] : fetchMap) {
        const auto skipIt = skipMap.find(key);
        if (skipIt == skipMap.end()) {
            VPUX_THROW("Fetch DMA with descId {0} does not have a corresponding Skip DMA", key);
        }
        auto skipOp = skipIt->second;
        auto skipAttr = skipOp.getSkipDmaAttr();

        const auto port = skipOp.getPort();
        const auto tile = skipAttr.getAssociatedTileIdx().getValue().getSExtValue();
        const auto logical = skipAttr.getAssociatedLogicalTaskIdx().getValue().getSExtValue();
        const auto list = skipAttr.getAssociatedListIdx().getValue().getSExtValue();

        auto input = mlir::cast<VPURT::DeclareBufferOp>(skipOp.getInput().getDefiningOp());
        const auto memPriority = input.getSection() == VPURT::getBufferSection(VPU::MemoryKind::CMX_NN) ? 1 : 0;

        orderedFetches.push_back(PlannedFetch{key, fetchOp, tile, logical, list, memPriority, port});
    }

    log.trace("Sorting fetch DMAs based on associated skip DMAs to ensure deterministic processing order");
    llvm::sort(orderedFetches, [](const PlannedFetch& a, const PlannedFetch& b) {
        return std::tuple(a.tile, a.logical, a.list, a.port, a.memPriority) <
               std::tuple(b.tile, b.logical, b.list, b.port, b.memPriority);
    });
    return orderedFetches;
}

VPUMI40XX::DeclareTaskBufferOp UpdateFetchDMAForSkipDMAs::createTaskBuffer(
        mlir::OpBuilder& builder, PlannedFetch plannedFetch, mlir::Operation* insertionPoint,
        TileListToIndexMap& taskBufferIndex, TileListToOffsetMap& taskBufferOffset, size_t dmaTaskBinarySize,
        uint64_t dmaMetadataOffset) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    if (insertionPoint != nullptr) {
        builder.setInsertionPoint(insertionPoint);
    }

    auto [itTaskIdx, unusedIdx] = taskBufferIndex.try_emplace({plannedFetch.tile, plannedFetch.list}, 0);
    auto& taskBufferIdx = itTaskIdx->second;

    auto [itTaskOffset, unusedOffset] =
            taskBufferOffset.try_emplace({plannedFetch.tile, plannedFetch.logical}, dmaMetadataOffset);
    auto& taskBufOffset = itTaskOffset->second;

    auto offsetAttr = mlir::IntegerAttr::get(getUInt64Type(builder.getContext()), taskBufOffset);
    taskBufOffset += dmaTaskBinarySize;
    auto indexAttr =
            VPURegMapped::IndexType::get(builder.getContext(), plannedFetch.tile, plannedFetch.list, taskBufferIdx++);

    return builder.create<VPUMI40XX::DeclareTaskBufferOp>(builder.getUnknownLoc(), indexAttr,
                                                          VPURegMapped::TaskType::DMA, offsetAttr);
}

void UpdateFetchDMAForSkipDMAs::collectFetchAndSkipDMAs(DMAMap& fetchMap, DMAMap& skipMap) {
    _log.trace("Collecting Fetch and Skip DMAs");
    auto netFunc = getOperation();
    for (auto dmaOp : netFunc.getOps<VPUMI40XX::NNDMAOp>()) {
        if (auto fetchAttr = dmaOp.getFetchDmaAttr()) {
            if (fetchAttr.getTargetExecutorKindAttr().getValue() == config::ExecutorKind::DMA_NN &&
                fetchAttr.getDescId()) {
                auto key = fetchAttr.getDescId().getValue().getSExtValue();
                fetchMap[key] = dmaOp;
            }
        }

        if (auto skipAttr = dmaOp.getSkipDmaAttr()) {
            auto key = skipAttr.getDescId().getValue().getSExtValue();
            skipMap[key] = dmaOp;
        }
    }
}

void UpdateFetchDMAForSkipDMAs::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi);

    constexpr size_t DMA_DDR2CMX_LISTIDX = 0;
    constexpr size_t DMA_WLM_TILEIDX = 0;

    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    auto bufferInsertionPoint = !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    auto declTaskBufferOps = netFunc.getOps<VPUMI40XX::DeclareTaskBufferOp>();
    auto declareBufferInsertionPoint =
            !declTaskBufferOps.empty() ? *declTaskBufferOps.begin() : &netFunc.getBody().front().front();

    DMAMap fetchMap, skipMap;
    collectFetchAndSkipDMAs(fetchMap, skipMap);
    if (fetchMap.empty() || skipMap.empty()) {
        _log.trace("No Fetch or Skip DMAs found, skipping UpdateFetchDMAForSkipDMAs pass");
        return;
    }

    auto parentModule = netFunc->getParentOfType<mlir::ModuleOp>();
    auto shavesCountPerTile = config::getAvailableExecutor(parentModule, config::ExecutorKind::SHAVE_ACT).getCount();
    auto dmaMetadataOffset =
            getRegionOffset(VPURegMapped::TaskType::DMA, config::getArch(getOperation()), shavesCountPerTile);
    auto dmaTaskBinarySize = VPUMI40XX::getTaskBinarySize(VPURegMapped::TaskType::DMA, config::getArch(getOperation()));

    // (tile,list) -> running task buffer index
    TileListToIndexMap taskBufferIndex;
    // logicalTask -> running offset
    TileListToOffsetMap taskBufferOffset;
    auto orderedFetches = getOrderedFetchList(fetchMap, skipMap, _log);

    builder.setInsertionPoint(bufferInsertionPoint);
    auto inputBuffer = createFetchBuffer(builder, dmaTaskBinarySize);
    for (auto& planned : orderedFetches) {
        _log.trace("Processing Fetch DMA with descId {0} associated with Skip DMA for tile {1}, logical task {2}, list "
                   "{3}, channel {4}",
                   planned.descId, planned.tile, planned.logical, planned.list, planned.memPriority);

        auto placeholderDmaOp = planned.fetchOp;
        auto dmaToFetch = skipMap.lookup(planned.descId);
        VPUX_THROW_UNLESS(dmaToFetch, "Fetch DMA with descId {0} does not have a corresponding Skip DMA",
                          planned.descId);

        // Create Memrefs for FetchDMA
        auto memrefCMX = createFetchOutputMemRefs(builder, planned.tile, dmaTaskBinarySize);

        // Create DeclareTaskBuffer with offset in CMX where SkipDMA descriptor will be stored
        auto taskBuffer = createTaskBuffer(builder, planned, declareBufferInsertionPoint, taskBufferIndex,
                                           taskBufferOffset, dmaTaskBinarySize, dmaMetadataOffset);

        // Set task location for SkipDMA to point to task buffer with offset in CMX where the descriptor will be stored
        dmaToFetch.setTaskLocation(taskBuffer.getResult());

        builder.setInsertionPoint(placeholderDmaOp);
        auto taskLocationsView = builder.create<VPURegMapped::ViewTaskRangeOp>(
                dmaToFetch.getLoc(), memrefCMX, taskBuffer.getResult(), taskBuffer.getResult());

        // Create FetchDMA
        auto fetchDma = builder.create<VPUMI40XX::NNDMAOp>(
                placeholderDmaOp.getLoc(), placeholderDmaOp.getIndexType(), nullptr, inputBuffer,
                mlir::ValueRange({taskLocationsView.getResult()}), placeholderDmaOp.getPreviousTask(),
                placeholderDmaOp.getWaitBarriers(), placeholderDmaOp.getUpdateBarriers(), /*start_after*/ 0,
                /*clean_after*/ 0, placeholderDmaOp.getIsOutOfOrder(), placeholderDmaOp.getIsCritical(),
                placeholderDmaOp.getEnableMsc(), placeholderDmaOp.getPort(), VPUIP::DMAAccMode::DISABLE, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr, placeholderDmaOp.getEnqueueBarrier(),
                placeholderDmaOp.getWlmPageAttr());

        fetchDma.setFetchDmaAttr(placeholderDmaOp.getFetchDmaAttr());
        placeholderDmaOp.getResult().replaceAllUsesWith(fetchDma.getResult());

        if (placeholderDmaOp->use_empty()) {
            placeholderDmaOp->erase();
        }
    }

    // If we have atleast 1 fetch or skip dma, we should have the list head
    auto listHead = mpi.getListHead(VPURegMapped::TaskType::DMA, DMA_WLM_TILEIDX, DMA_DDR2CMX_LISTIDX);
    VPUMI40XX::reindexList(mpi, mlir::cast<VPUMI40XX::NNDMAOp>(listHead.getDefiningOp()), 0, 0);
}
}  // namespace

//
// createUpdateFetchDMAForSkipDMAsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createUpdateFetchDMAForSkipDMAsPass(Logger log) {
    return std::make_unique<UpdateFetchDMAForSkipDMAs>(log);
}
