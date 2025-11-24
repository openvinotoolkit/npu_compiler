//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <queue>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_COMPRESSSPILLDMA
#define GEN_PASS_DEF_COMPRESSSPILLDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

// Helper class to manage CMX allocation for act_comp_size buffers for
// spill-write/reads which will be converted to compress/decompress DMAs
class CompressSpillDmaReservedMemoryAllocator {
public:
    CompressSpillDmaReservedMemoryAllocator(int64_t tileCount, int64_t dmaCount, int64_t rsvdMemOffset,
                                            int64_t rsvdMemSize)
            : _tileCount(tileCount), _dmaCount(dmaCount), _rsvdMemOffset(rsvdMemOffset), _rsvdMemSize(rsvdMemSize) {
        init();
    }

    std::optional<std::pair<int64_t, int64_t>> getMemSlot(size_t spillWriteIndex, int64_t port,
                                                          SmallVector<size_t>& deallocIndecesPerPort);

private:
    void init();

    struct SlotData {
        size_t ownerIndex;
        SmallVector<size_t> deallocIndecesPerPort;
    };

    SmallVector<SlotData> _slotDataVec;

    int64_t _numOfSlotsPerTile;

    int64_t _tileCount;
    int64_t _dmaCount;
    int64_t _rsvdMemOffset;
    int64_t _rsvdMemSize;
};

// Initialize internal variables.
// Each CMX tile has reserved memory with same size at same offset
// for act_comp_size allocation purpose. Slots will be assigned from this
// space from all available tiles
void CompressSpillDmaReservedMemoryAllocator::init() {
    int64_t numOfSlots = _rsvdMemSize * _tileCount / ACT_COMPRESSION_SIZE_ENTRY_SIZE;
    _numOfSlotsPerTile = _rsvdMemSize / ACT_COMPRESSION_SIZE_ENTRY_SIZE;

    VPUX_THROW_UNLESS(numOfSlots > 0, "Size of reserved memory ('{0}') is not able to hold single entry ('{1}')",
                      _rsvdMemSize, ACT_COMPRESSION_SIZE_ENTRY_SIZE);

    _slotDataVec.resize(numOfSlots);
    for (auto& slotData : _slotDataVec) {
        slotData.deallocIndecesPerPort.resize(_dmaCount);
    }
}

// Process request for spilling op to get act_comp_size slot (cmxId and offset).
// For each spill op store information when (at what op index) allocated buffer
// can be deallocated and reused for next tasks
std::optional<std::pair<int64_t, int64_t>> CompressSpillDmaReservedMemoryAllocator::getMemSlot(
        size_t spillWriteIndex, int64_t port, SmallVector<size_t>& deallocIndecesPerPort) {
    for (size_t index = 0; index < _slotDataVec.size(); index++) {
        // Check if for given port this slot can be treated as deallocated
        if (spillWriteIndex < _slotDataVec[index].deallocIndecesPerPort[port]) {
            continue;
        }

        // Slot can be reassigned to this op
        _slotDataVec[index].ownerIndex = spillWriteIndex;
        _slotDataVec[index].deallocIndecesPerPort = deallocIndecesPerPort;

        int64_t cmxId = index / _numOfSlotsPerTile;
        int64_t offset = _rsvdMemOffset + (index % _numOfSlotsPerTile) * ACT_COMPRESSION_SIZE_ENTRY_SIZE;

        return std::make_pair(cmxId, offset);
    }

    return std::nullopt;
}

namespace {

//
//  CompressSpillDmaPass
//

class CompressSpillDmaPass final : public VPUIP::impl::CompressSpillDmaBase<CompressSpillDmaPass> {
public:
    explicit CompressSpillDmaPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    struct SpillDataKey {
        int64_t spillId;
        int64_t cmxId;

        bool operator<(const SpillDataKey& other) const {
            if (spillId == other.spillId) {
                return cmxId < other.cmxId;
            }
            return spillId < other.spillId;
        }

        bool operator==(const SpillDataKey& other) const {
            return (cmxId == other.cmxId && spillId == other.spillId);
        }
    };

    struct SpillDataVal {
        int64_t port;
        size_t spillWriteIndex;
        std::set<size_t> spillReadIndeces;
        SmallVector<size_t> deallocIndecesPerPort;
    };

    std::map<SpillDataKey, SpillDataVal> getSpillData(const SmallVector<VPURT::TaskOp>& taskOpsVec);
    void identifySafeDeallocIndexForSpills(const SmallVector<VPURT::TaskOp>& taskOpsVec,
                                           std::map<SpillDataKey, SpillDataVal>& spillDataMap);
    void convertSpillsToCompressOps(mlir::MLIRContext* ctx, const SmallVector<VPURT::TaskOp>& taskOpsVec,
                                    std::map<SpillDataKey, SpillDataVal>& spillDataMap);

    void adjustSpillIds(const SmallVector<VPURT::TaskOp>& taskOpsVec);
    void initIndexAttr(mlir::MLIRContext* ctx, const SmallVector<VPURT::TaskOp>& taskOpsVec);
    void clearIndexAttr(const SmallVector<VPURT::TaskOp>& taskOpsVec);
    size_t getTaskIndex(VPURT::TaskOp taskOp);

    void safeRunOnModule() final;

    int64_t _tileCount = 0;
    int64_t _dmaCount = 0;
    int64_t _rsvdMemOffset = 0;
    int64_t _rsvdMemSize = 0;
    // Below is a limit of graph depth that will be analyzed during BFS when traversing IR
    // down from spill-read through control graph to not cause too big compilation
    // time bottleneck in case of big models with multiple spills
    static constexpr size_t _safeDeallocBfsDepthLimit = 200;
};

// Adjust spill-ids to be unique after function inlining which will cause multiple spill-write/read pairs
// to use the same spill-id. Since outlined functions are always sequential once same spill-id is used for
// spill-write that was already encountered before it means that it is part of new inlined function and
// ids can be adjusted (add offset) to be unique for rest of the methods in this pass
void CompressSpillDmaPass::adjustSpillIds(const SmallVector<VPURT::TaskOp>& taskOpsVec) {
    _log.trace("Traverse spill ops and adjust spill-ids to be unique");
    int64_t spillIdOffset = 0;
    mlir::DenseSet<std::tuple<int64_t, int64_t, int64_t>> encounteredSpillWritesKeys;
    int64_t prevMaxSpillIdForSpillWrite = 0;
    for (size_t i = 0; i < taskOpsVec.size(); i++) {
        auto taskOp = taskOpsVec[i];

        auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp());
        if (dmaOp == nullptr) {
            continue;
        }

        if (!dmaOp.getSpillId().has_value()) {
            continue;
        }

        auto spillId = dmaOp.getSpillId().value();

        const auto port = dmaOp.getPort();
        VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
        const auto portValue = port.value();

        const auto inType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getInput().getType());
        const auto outType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getOutput().getType());

        auto newSpillId = spillId + spillIdOffset;

        if (inType.getMemoryKind() == VPU::MemoryKind::CMX_NN && outType.getMemoryKind() == VPU::MemoryKind::DDR) {
            auto cmxIdx = inType.getMemSpace().getIndex().value_or(0);

            std::tuple<int64_t, int64_t, int64_t> spillWriteKey{spillId, cmxIdx, portValue};

            prevMaxSpillIdForSpillWrite = std::max(prevMaxSpillIdForSpillWrite, newSpillId);

            if (encounteredSpillWritesKeys.find(spillWriteKey) != encounteredSpillWritesKeys.end()) {
                spillIdOffset = prevMaxSpillIdForSpillWrite + 1;
                newSpillId = spillId + spillIdOffset;
                encounteredSpillWritesKeys.clear();
            }
            encounteredSpillWritesKeys.insert(spillWriteKey);

            _log.trace(
                    "Spill-write op, index - '{0}', port - '{1}', cmxIdx - '{2}', spillId - '{3}', newSpillId - '{4}'",
                    i, portValue, cmxIdx, spillId, newSpillId);
        } else if (inType.getMemoryKind() == VPU::MemoryKind::DDR &&
                   outType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            auto cmxIdx = outType.getMemSpace().getIndex().value_or(0);

            _log.trace(
                    "Spill-read op, index - '{0}', port - '{1}', cmxIdx - '{2}', spillId - '{3}', newSpillId - '{4}'",
                    i, portValue, cmxIdx, spillId, newSpillId);
        }

        dmaOp.setSpillId(newSpillId);
    }
}

void CompressSpillDmaPass::initIndexAttr(mlir::MLIRContext* /*ctx*/, const SmallVector<VPURT::TaskOp>& taskOpsVec) {
    for (size_t i = 0; i < taskOpsVec.size(); i++) {
        auto taskOp = taskOpsVec[i];
        taskOp.getProperties().setTaskIndex(i);
    }
}

void CompressSpillDmaPass::clearIndexAttr(const SmallVector<VPURT::TaskOp>& taskOpsVec) {
    for (size_t i = 0; i < taskOpsVec.size(); i++) {
        auto taskOp = taskOpsVec[i];
        taskOp.getProperties().setTaskIndex(std::nullopt);
    }
}

size_t CompressSpillDmaPass::getTaskIndex(VPURT::TaskOp taskOp) {
    const auto taskIndex = taskOp.getProperties().getTaskIndex();
    VPUX_THROW_UNLESS(taskIndex.has_value(), "Get: task index was not set for '{0}' operation at '{1}'",
                      taskOp->getName(), taskOp->getLoc());
    return static_cast<size_t>(taskIndex.value());
}

// Traverse IR and identify spill-write and corresponding spill-read ops
std::map<CompressSpillDmaPass::SpillDataKey, CompressSpillDmaPass::SpillDataVal> CompressSpillDmaPass::getSpillData(
        const SmallVector<VPURT::TaskOp>& taskOpsVec) {
    std::map<SpillDataKey, SpillDataVal> spillDataMap;

    for (size_t i = 0; i < taskOpsVec.size(); i++) {
        auto taskOp = taskOpsVec[i];

        auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp());
        if (dmaOp == nullptr) {
            continue;
        }

        if (!dmaOp.getSpillId().has_value()) {
            continue;
        }
        if (dmaOp.getCompressCandidateAttr() == nullptr) {
            continue;
        }

        const auto port = dmaOp.getPort();
        VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");
        const auto portValue = port.value();

        const auto spillId = dmaOp.getSpillId().value();
        const auto inType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getInput().getType());
        const auto outType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getOutput().getType());

        auto isCompactType = [](vpux::NDTypeInterface origType) {
            const auto shape = origType.getShape();
            const auto strideReqs = StrideReqs::compact(shape.size());
            return strideReqs.checkStrides(origType);
        };

        if (!isCompactType(inType) || !isCompactType(outType)) {
            _log.trace("Compress candidate which is not a flat dma - {0}", dmaOp->getLoc());
            continue;
        }

        if (inType.getMemoryKind() == VPU::MemoryKind::CMX_NN && outType.getMemoryKind() == VPU::MemoryKind::DDR &&
            isSupportedBufferSizeForCompression(inType)) {
            auto cmxIdx = inType.getMemSpace().getIndex().value_or(0);

            SpillDataKey spillDataKey{spillId, cmxIdx};

            VPUX_THROW_UNLESS(spillDataMap.find(spillDataKey) == spillDataMap.end(),
                              "Spill Write for CMX '{0}' and spillId '{1}' has already been identified before", cmxIdx,
                              spillId);
            spillDataMap[spillDataKey].port = portValue;
            spillDataMap[spillDataKey].spillWriteIndex = i;

            _log.trace("Spill-write op, index - '{0}', port - '{1}', cmxIdx - '{2}', spillId - '{3}'", i, portValue,
                       cmxIdx, spillId);
        } else if (inType.getMemoryKind() == VPU::MemoryKind::DDR &&
                   outType.getMemoryKind() == VPU::MemoryKind::CMX_NN && isSupportedBufferSizeForCompression(outType)) {
            auto cmxIdx = outType.getMemSpace().getIndex().value_or(0);

            SpillDataKey spillDataKey{spillId, cmxIdx};

            if (spillDataMap.find(spillDataKey) == spillDataMap.end()) {
                _log.trace("Unexpected Spill Read as Spill Write for CMX '{0}' and spillId '{1}' has not been "
                           "identified before",
                           cmxIdx, spillId);
                continue;
            }

            spillDataMap[spillDataKey].spillReadIndeces.insert(i);

            _log.trace("Spill-read op, index - '{0}', port - '{1}', cmxIdx - '{2}', spillId - '{3}'", i, portValue,
                       cmxIdx, spillId);
        }
    }

    return spillDataMap;
}

// For spilling ops (spillDataMap) get last spill read for each spill-write
// and find index of op when corresponding act_comp_size buffer can be considered
// deallocated and ready for reuse. This is needed because next spill-write (compress)
// happens on different DMA HW queue (channel) then spill-read (decompress) and to mitigate
// race condition dependencies in IR need to be analyzed to find moment when act_comp_size
// buffer can be reused safely
//
// Exmaple:
// DMA Port0 channel CMX: [Compress1]            [Compress2]
// DMA Port0 channel DDR:            [Decompres1]           [Decompress2]
//
// If there is no dependency between Compress2 and Decompress1 then those tasks
// can happen in parallel and Compress2 needs to be assigned different CMX slot for
// act_comp_size.
// There is a limit of graph depth that will be analyzed during BFS when traversing IR
// down from spill-read through control graph to not cause too big compilation
// time bottleneck in case of big models with multiple spills. If limit was reached identified
// safe dealloc index might not be optimal (minimal) or might not be located at all meaning
// that assigned slot will not be reused by any subsequent compress ops
void CompressSpillDmaPass::identifySafeDeallocIndexForSpills(const SmallVector<VPURT::TaskOp>& taskOpsVec,
                                                             std::map<SpillDataKey, SpillDataVal>& spillDataMap) {
    DenseMap<size_t, SmallVector<size_t>> spillReadOpsToPerPortSafeDeallocIndexesMap;

    // Below function runs BFS (with level limit) for each provided spill read op to find
    // possible closest safe deallocation task index
    // TODO E#120701: Possible compile time optimization would be to run DFS algorithm and track
    // encountered on the way other spill-read ops and reuse given graph traversal to update
    // also positions for other spills instead of calling this function for each spill read
    auto updateSafeDeallocIndexForNewSpillRead = [&](size_t spillReadOpIndex) -> void {
        auto getChildrenOps = [&](size_t taskIndex) {
            std::set<size_t> childrenOps;

            auto taskOp = taskOpsVec[taskIndex];
            for (const auto& bar : taskOp.getUpdateBarriers()) {
                for (auto& use : bar.getUses()) {
                    auto userOp = use.getOwner();
                    if (auto userTaskOp = mlir::dyn_cast<VPURT::TaskOp>(userOp)) {
                        const auto userOperandIdx = use.getOperandNumber();
                        if (userOperandIdx < userTaskOp.getWaitBarriers().size()) {
                            childrenOps.insert(getTaskIndex(userTaskOp));
                        }
                    }
                }
            }
            return childrenOps;
        };

        // Initialize safe dealloc index for both ports to be big value
        // since algorithm will try to find and maintain minimal encountered index
        spillReadOpsToPerPortSafeDeallocIndexesMap[spillReadOpIndex] =
                SmallVector<size_t>(_dmaCount, std::numeric_limits<size_t>::max());

        // Run BFS through control graph to identify ops depending on spillReadOpIndex
        // directly or through chain of other ops and are placed on DMA channel that
        // is used by activation compression (DMA channel CMX)
        // If some other spill-read decompress ops are found on the way algorithm
        // can update safe deallocation indexes also for them
        mlir::DenseSet<size_t> explored;
        std::queue<size_t> queue;
        std::set<size_t> childrenTasks;

        // Push to the queue start node as the first element to analyze
        queue.push(spillReadOpIndex);
        explored.insert(spillReadOpIndex);

        size_t bfsLevel = 0;

        while (!queue.empty() || !childrenTasks.empty()) {
            // If queue is empty start analyzing children tasks
            // We do this since children tasks are ordered and we want to find the lowest one
            if (queue.empty()) {
                if (++bfsLevel >= _safeDeallocBfsDepthLimit) {
                    // If limit was met, stop BFS from deeper traversal
                    _log.nest().trace("Reached BFS depth traversal limit ({0}). Stoping search",
                                      _safeDeallocBfsDepthLimit);
                    break;
                }

                for (const auto& c : childrenTasks) {
                    queue.push(c);
                }

                childrenTasks.clear();
            }

            // Each time pop first element from the queue and print its value
            auto taskIndex = queue.front();
            queue.pop();

            auto taskOp = taskOpsVec[taskIndex];

            if (auto dmaTypeOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(taskOp.getInnerTaskOp())) {
                VPUX_THROW_WHEN(dmaTypeOp.getPortAttribute() == nullptr, "DMA op has no port attribute, op - '{0}'",
                                dmaTypeOp->getLoc());

                const auto port = dmaTypeOp.getPortVal();
                VPUX_THROW_UNLESS(port.has_value(), "DMA port has not been set");

                const auto inType = mlir::cast<vpux::NDTypeInterface>(dmaTypeOp.getInput().getType());
                const auto outType = mlir::cast<vpux::NDTypeInterface>(dmaTypeOp.getOutput().getType());

                if (inType.getMemoryKind() == VPU::MemoryKind::CMX_NN &&
                    outType.getMemoryKind() == VPU::MemoryKind::DDR) {
                    spillReadOpsToPerPortSafeDeallocIndexesMap[spillReadOpIndex][port.value()] = std::min(
                            spillReadOpsToPerPortSafeDeallocIndexesMap[spillReadOpIndex][port.value()], taskIndex);
                }
            }

            // Check all the children. If not yet analyzed node push it
            // to the queue otherwise skip it
            for (auto c : getChildrenOps(taskIndex)) {
                if (explored.find(c) == explored.end()) {
                    explored.insert(c);
                    childrenTasks.insert(c);
                }
            }
        }
    };

    // For identified spilling ops get last spill read for each spill-write
    // and find index of op when corresponding act_comp_size buffer can be considered
    // deallocated and ready for reuse
    for (auto& spillDataPair : spillDataMap) {
        auto lastSpillReadOpIndex = *spillDataPair.second.spillReadIndeces.rbegin();

        _log.trace("Spill-id {0}, cmx-id {1}, last spill-read index {2}", spillDataPair.first.spillId,
                   spillDataPair.first.cmxId, lastSpillReadOpIndex);

        if (spillReadOpsToPerPortSafeDeallocIndexesMap.find(lastSpillReadOpIndex) ==
            spillReadOpsToPerPortSafeDeallocIndexesMap.end()) {
            // Find at which op index related act_comp_size buffer can be considered deallocated
            // and possible for reuse for next compress op based on port index
            updateSafeDeallocIndexForNewSpillRead(lastSpillReadOpIndex);
        }

        spillDataPair.second.deallocIndecesPerPort = spillReadOpsToPerPortSafeDeallocIndexesMap[lastSpillReadOpIndex];

        for (const auto& spillReadSafeDeallocIndex : spillDataPair.second.deallocIndecesPerPort | indexed) {
            _log.nest().trace("For port '{0}' safe deallocation index - '{1}'", spillReadSafeDeallocIndex.index(),
                              spillReadSafeDeallocIndex.value());
        }
    }
}

// Create buffer representing act_comp_size buffer
VPURT::DeclareBufferOp declareCompressLUTBufAlloc(mlir::MLIRContext* ctx, mlir::OpBuilder& builder,
                                                  mlir::Location taskLoc, int64_t cmxIdx, int64_t offset) {
    auto memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), static_cast<size_t>(cmxIdx));
    auto actCompressionEntryType =
            getMemRefType(ShapeRef({ACT_COMPRESSION_SIZE_ENTRY_SIZE}), getUInt8Type(ctx), DimsOrder::C, memKindAttr);

    const auto loc = appendLoc(taskLoc, "compressionLUT");

    return builder.create<VPURT::DeclareBufferOp>(loc, actCompressionEntryType, VPURT::BufferSection::CMX_NN, cmxIdx,
                                                  offset);
}

// Convert spill-write to task with compress DMA
void createCompressDma(VPURT::TaskOp spillWriteTaskOp, mlir::Value actCompSizeBuffer) {
    auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(spillWriteTaskOp.getInnerTaskOp());
    VPUX_THROW_WHEN(dmaOp == nullptr, "No DMA task provided to convert, spill-write taskOp - '{0}'",
                    spillWriteTaskOp->getLoc());

    mlir::OpBuilder builder(spillWriteTaskOp.getOperation());
    builder.setInsertionPoint(dmaOp);

    const auto loc = appendLoc(dmaOp->getLoc(), "actCompression");

    auto outputBuf = dmaOp.getOutputBuff();
    auto outputType = outputBuf.getType();
    outputType = vpux::setCompressionState(outputType, VPUIP::CompressionState::RuntimeCompressed);
    outputBuf.setType(outputType);

    builder.create<VPUIP::CompressDMAOp>(loc, dmaOp.getInput(), actCompSizeBuffer,
                                         /*act_compression_sparsity_map*/ nullptr, outputBuf, dmaOp.getPortAttr(),
                                         dmaOp.getIsOutOfOrderAttr(), dmaOp.getIsCriticalAttr(),
                                         /*dmaHwpId=*/nullptr,
                                         /*profilingMetadata=*/nullptr);
    dmaOp.erase();
}

// Convert spill-read to task with decompress DMA
void createDecompressDma(VPURT::TaskOp spillReadTaskOp, mlir::Value actCompSizeBuffer) {
    auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(spillReadTaskOp.getInnerTaskOp());
    VPUX_THROW_WHEN(dmaOp == nullptr, "No DMA task provided to convert, spill-read taskOp - '{0}'",
                    spillReadTaskOp->getLoc());

    mlir::OpBuilder builder(spillReadTaskOp.getOperation());
    builder.setInsertionPoint(dmaOp);

    const auto loc = appendLoc(dmaOp->getLoc(), "actDecompression");

    auto inputBuf = dmaOp.getInput();
    auto inputType = inputBuf.getType();
    inputType = vpux::setCompressionState(inputType, VPUIP::CompressionState::RuntimeCompressed);
    inputBuf.setType(inputType);

    builder.create<VPUIP::DecompressDMAOp>(loc, inputBuf, actCompSizeBuffer, /*act_compression_sparsity_map*/ nullptr,
                                           dmaOp.getOutputBuff(), dmaOp.getPortAttr(), dmaOp.getIsOutOfOrderAttr(),
                                           dmaOp.getIsCriticalAttr(),
                                           /* dma_hwp_id= */ nullptr,
                                           /* profilingMetadata= */ nullptr);
    dmaOp.erase();
}

void CompressSpillDmaPass::convertSpillsToCompressOps(mlir::MLIRContext* ctx,
                                                      const SmallVector<VPURT::TaskOp>& taskOpsVec,
                                                      std::map<SpillDataKey, SpillDataVal>& spillDataMap) {
    auto memSlotAllocator =
            CompressSpillDmaReservedMemoryAllocator(_tileCount, _dmaCount, _rsvdMemOffset, _rsvdMemSize);

    // Process identified spilling ops and for each spill-write try to get CMX mem slot
    // for act_comp_size. If such slot is available spill-write and related spill-reads
    // will be converted to Compress/DecompressDMA ops
    for (auto& spillDataPair : spillDataMap) {
        auto [spillId, cmxId] = spillDataPair.first;
        auto spillDataVal = spillDataPair.second;

        _log.trace("Process spill-id {0}, cmx-id {1}", spillId, cmxId);

        // Get free slot, provide current spill write index, port and information on when it will be deallocated
        _log.nest().trace("Get free slot for spill-write op {0} on port {1}", spillDataVal.spillWriteIndex,
                          spillDataVal.port);

        auto spillWriteTaskOp = taskOpsVec[spillDataVal.spillWriteIndex];

        auto actCompSizeSlot = memSlotAllocator.getMemSlot(spillDataVal.spillWriteIndex, spillDataVal.port,
                                                           spillDataVal.deallocIndecesPerPort);

        // No free slot was found. This spill cannot be converted to Compress/Decompress DMAs
        if (!actCompSizeSlot.has_value()) {
            continue;
        }

        int64_t actCompSizeCmxInd = actCompSizeSlot.value().first;
        int64_t actCompSizeOffset = actCompSizeSlot.value().second;

        _log.nest().trace("Free slot: cmxId {0}, offset {1}", actCompSizeCmxInd, actCompSizeOffset);

        mlir::OpBuilder builder(spillWriteTaskOp.getOperation());
        builder.setInsertionPoint(spillWriteTaskOp.getOperation());

        auto actCompBuffer = declareCompressLUTBufAlloc(ctx, builder, spillWriteTaskOp->getLoc(), actCompSizeCmxInd,
                                                        actCompSizeOffset)
                                     .getBuffer();

        // Assign free slot to spill write DMA, convert
        _log.nest().trace("Assign free slot to spill-write op {0}", spillDataVal.spillWriteIndex);

        createCompressDma(spillWriteTaskOp, actCompBuffer);

        // Assign free slot to each spill read DMA, convert
        for (auto& spillReadIndex : spillDataVal.spillReadIndeces) {
            _log.nest().trace("Assign free slot to spill-read op {0}", spillReadIndex);
            auto spillReadTaskOp = taskOpsVec[spillReadIndex];
            createDecompressDma(spillReadTaskOp, actCompBuffer);
        }
    }
}

void CompressSpillDmaPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp func;
    net::NetworkInfoOp::getFromModule(module, netInfo, func);
    mlir::OpBuilder builder(&func.getBody().front().front());

    auto tileOp = config::getTileExecutor(module);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE_Cluster information");
    _tileCount = tileOp.getCount();

    if (auto rsvdMem = config::getCompressDmaReservedMemory(module, VPU::MemoryKind::CMX_NN)) {
        _rsvdMemSize = rsvdMem.getByteSize();
        VPUX_THROW_UNLESS(rsvdMem.getOffset().has_value(), "No offset setting provided");
        _rsvdMemOffset = rsvdMem.getOffset().value();
    }
    VPUX_THROW_UNLESS(_rsvdMemSize > 0, "No reserved memory provided for handling compressed DMAs");

    _log.trace("Compressed DMAs reserved memory: offset - '{0}', size - '{1}'", _rsvdMemOffset, _rsvdMemSize);

    auto dmaPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    VPUX_THROW_UNLESS(dmaPorts != nullptr, "Failed to get DMA information");
    _dmaCount = dmaPorts.getCount();

    const auto taskOpsVec = to_small_vector(func.getOps<VPURT::TaskOp>());

    // Update SpillIds so that they are unique on whole module op level
    // This is needed in case of previously done function inlining
    adjustSpillIds(taskOpsVec);

    // Since this pass processing uses task index for each task add
    // a temporary task-index attribute to enable easy retrieval of index from VPURT::TaskOp
    initIndexAttr(ctx, taskOpsVec);

    // Traverse IR and identify spill-write and corresponding spill-read ops. Store data
    // For later processing
    auto spillDataMap = getSpillData(taskOpsVec);

    // No spilling identified. Exit early.
    if (spillDataMap.empty()) {
        return;
    }

    // For identified spilling ops get last spill read for each spill-write
    // and find index of op when corresponding act_comp_size buffer can be considered
    // deallocated and ready for reuse
    identifySafeDeallocIndexForSpills(taskOpsVec, spillDataMap);

    // Process identified spilling ops and for each spill-write try to get CMX mem slot
    // for act_comp_size. If such slot is available spill-write and related spill-reads
    // will be converted to Compress/DecompressDMA ops
    convertSpillsToCompressOps(ctx, taskOpsVec, spillDataMap);

    // Remove temporary task index attribute
    clearIndexAttr(taskOpsVec);
}

}  // namespace

//
// createCompressSpillDmaPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCompressSpillDmaPass(Logger log) {
    return std::make_unique<CompressSpillDmaPass>(log);
}
