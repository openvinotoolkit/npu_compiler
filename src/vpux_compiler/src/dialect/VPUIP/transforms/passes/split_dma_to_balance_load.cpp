//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/gather_dma_constants.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <unordered_set>

using namespace vpux;

namespace vpux::VPUIP {
#define GEN_PASS_DECL_SPLITDMATOBALANCELOAD
#define GEN_PASS_DEF_SPLITDMATOBALANCELOAD
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP
namespace {

using BuffersVec = SmallVector<mlir::Value>;

// DMA size below 128B is not able to utilize the bandwidth of the DMA engine.
constexpr int64_t MIN_DMA_SIZE_FOR_SPLIT = 128;

void removeUnrollIDxAttr(mlir::Operation* op) {
    if (op->hasAttr(VPUIP::UNROLL_IDX)) {
        op->removeAttr(VPUIP::UNROLL_IDX);
    }
}

void removeUnrollIDxAttr(llvm::SmallVector<VPURT::TaskOp>& ops) {
    for (auto& op : ops) {
        auto dmaOp = op.getInnerTaskOp();
        if (dmaOp && dmaOp->hasAttr(VPUIP::UNROLL_IDX)) {
            dmaOp->removeAttr(VPUIP::UNROLL_IDX);
        }
    }
}

// Replace single allocation with multiple separate allocations. These allocations cover same memory range, but each
// points to a different part based on the split sizes passed as parameter
BuffersVec getReplacementBuffers(mlir::Value originalBuffer, Dim tileDim, const SmallVector<int64_t>& partSizes,
                                 mlir::OpBuilder builder) {
    const auto bufferType = mlir::cast<NDTypeInterface>(originalBuffer.getType());
    auto bufferOp = originalBuffer.getDefiningOp<VPURT::DeclareBufferOp>();

    auto origStrides = bufferType.getStrides();
    builder.setInsertionPoint(bufferOp);
    const auto getTiledBuf = [&](int64_t newOffset, int64_t newDimSize, int64_t extraOffset,
                                 StringRef locSuffix) -> mlir::Value {
        auto newType = VPUIP::getNewBufferType(bufferType, tileDim, newOffset, newDimSize);
        newType = newType.changeStrides(origStrides);

        auto newBufferOffset = bufferOp.getByteOffset() + extraOffset;
        const auto newLoc = takeOpLoc(bufferOp, locSuffix);

        return builder
                .create<VPURT::DeclareBufferOp>(newLoc, newType, bufferOp.getSectionAttr(),
                                                bufferOp.getSectionIndexAttr(), getIntAttr(builder, newBufferOffset),
                                                bufferOp.getSwizzlingKeyAttr())
                ->getResult(0);
    };

    BuffersVec buffers;
    buffers.reserve(partSizes.size());

    int64_t cumulativeDimOffset = 0;
    int64_t cumulativeByteOffset = 0;

    for (size_t i = 0; i < partSizes.size(); ++i) {
        buffers.emplace_back(
                getTiledBuf(cumulativeDimOffset, partSizes[i], cumulativeByteOffset, "part_" + std::to_string(i + 1)));
        cumulativeDimOffset += partSizes[i];
        cumulativeByteOffset += Byte(partSizes[i] * origStrides[tileDim]).count();
    }

    return buffers;
}

BuffersVec getConstantParts(mlir::Value originalConstant, Dim tileDim, const SmallVector<int64_t>& partSizes,
                            mlir::OpBuilder builder) {
    auto cstOp = originalConstant.getDefiningOp<Const::DeclareOp>();

    const auto cstType = mlir::cast<NDTypeInterface>(cstOp.getOutput().getType());
    const auto origShape = cstType.getShape();
    builder.setInsertionPoint(cstOp);
    const auto createCstPart = [&](int64_t tileOffset, int64_t newDimSize, StringRef locSuffix) -> mlir::Value {
        Shape offset(SmallVector<int64_t>(origShape.size(), 0));
        offset[tileDim] = tileOffset;

        const auto newShape = VPUIP::getSplitShape(cstType, tileDim, newDimSize);
        const auto newLoc = takeOpLoc(cstOp, locSuffix);
        return builder.createOrFold<VPUIP::SubViewOp>(newLoc, cstOp, offset.raw(), newShape.raw());
    };

    BuffersVec constants;
    constants.reserve(partSizes.size());

    int64_t cumulativeOffset = 0;
    for (size_t i = 0; i < partSizes.size(); ++i) {
        constants.emplace_back(createCstPart(cumulativeOffset, partSizes[i], "part_" + std::to_string(i + 1)));
        cumulativeOffset += partSizes[i];
    }

    return constants;
}

VPUIP::DMATypeOpInterface createDMATask(VPURT::TaskOp originalTaskOp, VPUIP::DMATypeOpInterface originalDmaOp,
                                        mlir::Value input, mlir::Value output, StringRef locSuffix,
                                        mlir::OpBuilder builder) {
    auto newLoc = takeOpLoc(originalTaskOp, locSuffix);
    if (auto originalGatherDmaOp = mlir::dyn_cast<VPUIP::GatherDMAOp>(originalDmaOp.getOperation())) {
        auto newGatherDmaOp = VPURT::wrapIntoTaskOp<VPUIP::GatherDMAOp>(
                builder, originalTaskOp.getWaitBarriers(), originalTaskOp.getUpdateBarriers(), newLoc,
                originalGatherDmaOp.getInput(), input, output, originalGatherDmaOp.getElementSize(),
                originalGatherDmaOp.getPadding(), 0);
        return mlir::cast<VPUIP::DMATypeOpInterface>(newGatherDmaOp.getOperation());
    } else if (auto originalNNDMAOp = mlir::dyn_cast<VPUIP::NNDMAOp>(originalDmaOp.getOperation())) {
        auto newDmaOp = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                builder, originalTaskOp.getWaitBarriers(), originalTaskOp.getUpdateBarriers(), newLoc, input, output, 0,
                originalNNDMAOp.getIsOutOfOrder(), originalNNDMAOp.getIsCritical(), originalNNDMAOp.getSpillIdAttr(),
                /*compress_candidate=*/false);
        if (originalNNDMAOp.getProfilingBufferMgmt()) {
            newDmaOp.setProfilingBufferMgmt(true);
        }
        return mlir::cast<VPUIP::DMATypeOpInterface>(newDmaOp.getOperation());
    } else if (auto originalConvertDmaOp = mlir::dyn_cast<VPUIP::ConvertDMAOp>(originalDmaOp.getOperation())) {
        auto newDmaOp = VPURT::wrapIntoTaskOp<VPUIP::ConvertDMAOp>(
                builder, originalTaskOp.getWaitBarriers(), originalTaskOp.getUpdateBarriers(), newLoc, input, output,
                getIntAttr(builder, 0), originalConvertDmaOp.getIsOutOfOrder(), originalConvertDmaOp.getIsCritical(),
                originalConvertDmaOp.getDmaHwpIdAttr(), originalConvertDmaOp.getProfilingMetadataAttr(),
                /*split_candidate=*/nullptr);
        return mlir::cast<VPUIP::DMATypeOpInterface>(newDmaOp.getOperation());
    }
    VPUX_THROW("Can't create DMA task");
    return nullptr;
}

SmallVector<VPUIP::DMATypeOpInterface> replaceDmaWithMultipleParts(VPURT::TaskOp taskOp,
                                                                   VPUIP::DMATypeOpInterface dmaOp,
                                                                   const BuffersVec& inputs, const BuffersVec& outputs,
                                                                   ArrayRef<mlir::Value> removableInputs,
                                                                   mlir::OpBuilder builder, const Logger& log) {
    VPUX_THROW_UNLESS(inputs.size() == outputs.size(), "Inputs and outputs must have the same number of parts");

    builder.setInsertionPoint(taskOp);
    SmallVector<VPUIP::DMATypeOpInterface> newDmaOps;
    newDmaOps.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        newDmaOps.push_back(
                createDMATask(taskOp, dmaOp, inputs[i], outputs[i], "part_" + std::to_string(i + 1), builder));
    }

    taskOp->erase();
    for (mlir::Value input : removableInputs) {
        if (auto* op = input.getDefiningOp()) {
            if (op->use_empty()) {
                op->erase();
            }
        }
    }
    log.trace("Replaced DMA with {0} parts.", inputs.size());

    return newDmaOps;
}

// Trivial constant is constant without LAST or PREFERRED_LAST transformation, so SubView transformation can be
// attached to the end of list
bool isTrivialConst(Const::DeclareOp cstOp) {
    const auto& contentAttr = cstOp.getContentAttr();
    auto transformations = contentAttr.getTransformations();
    return transformations.empty() ||
           transformations.back().getPositionRequirement() == Const::details::PositionRequirement::NONE;
}

// Non trivial transforms requires folding and flattening to keep content correct
SmallVector<VPUIP::DMATypeOpInterface> splitFoldedConstToBufferDma(VPURT::TaskOp taskOp,
                                                                   VPUIP::DMATypeOpInterface dmaOp,
                                                                   Const::DeclareOp cstOp, Dim tileDim,
                                                                   const SmallVector<int64_t>& partSizes,
                                                                   mlir::OpBuilder builder, Logger log) {
    const auto cstType = mlir::cast<NDTypeInterface>(cstOp.getOutput().getType());
    const auto strides = cstType.getStrides();
    // In case of subbyte type, which has non-byte stride along tiling dim - don't attempt to split this constant
    const auto tileDimStride = strides[tileDim];
    if (tileDimStride.count() % CHAR_BIT != 0) {
        log.trace("Can't split constant with non-byte stride");
        return {};
    }

    for (size_t i = 1; i < partSizes.size(); ++i) {
        if (partSizes[i - 1] != partSizes[i]) {
            log.trace("Can't split folded constant - unequal parts");
            return {};
        }
    }
    const auto content = cstOp.getContent();
    const auto contentType = content.getType();
    const auto contentElemType = contentType.getElementType();
    const auto elemTypeBitSize = contentType.getElemTypeSize().count();
    const auto isUnsupportedSubByteStorageType = elemTypeBitSize < CHAR_BIT && elemTypeBitSize > 1;
    if (isUnsupportedSubByteStorageType) {
        log.trace("Can't split constant with unsupported element type");
        return {};
    }
    log.trace("Splitting FoldedConst->Buffer DMA into {0} parts", partSizes.size());
    const auto bufSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
    std::vector<char> tempBuf(bufSize);
    content.copyTo(MutableArrayRef(tempBuf.data(), bufSize));

    auto rankedTensorType = mlir::cast<mlir::RankedTensorType>(contentType);
    if (auto qtype = mlir::dyn_cast<mlir::quant::QuantizedType>(contentElemType)) {
        rankedTensorType =
                mlir::cast<mlir::RankedTensorType>(contentType.changeElemType(normalizeQuantStorageType(qtype)));
    }

    const auto rankedElemType = rankedTensorType.getElementType();
    const auto fullShape = rankedTensorType.getShape();
    SmallVector<int64_t> newShapeVec(fullShape.begin(), fullShape.end());
    auto actualTileDims = getNonOneDim(ShapeRef(newShapeVec));
    if (actualTileDims.empty()) {
        log.trace("Can't split constant with all ones shape");
        return {};
    }
    newShapeVec[actualTileDims[0].ind()] /= partSizes.size();

    builder.setInsertionPoint(cstOp);
    const auto createCstPart = [&](int64_t tileOffset, int64_t newDimSize, StringRef locSuffix) -> mlir::Value {
        const size_t offsetSize = Byte(tileOffset * tileDimStride).count();
        const size_t bufferSize = Byte(newDimSize * tileDimStride).count();
        char* baseContentPtr = tempBuf.data() + offsetSize;
        ArrayRef<char> partContent(baseContentPtr, baseContentPtr + bufferSize);
        const auto partRankedTensorType = rankedTensorType.clone(newShapeVec, rankedElemType);
        const auto denseAttr = Const::createExternalConstContent(
                partRankedTensorType, partContent, "INTERNAL_CONSTANT",
                Const::ExternalConstContentCreationOptions{/* deepCopyConstData */ true,
                                                           /* allowDuplicatesForTheSameResourceName */ false});
        const auto newLoc = takeOpLoc(cstOp, locSuffix);
        const auto newType = VPUIP::getNewBufferType(cstType, tileDim, tileOffset, newDimSize);

        return builder.create<Const::DeclareOp>(newLoc, newType, Const::ContentAttr::get(denseAttr));
    };

    BuffersVec inputBuffers;
    inputBuffers.reserve(partSizes.size());
    int64_t cumulativeOffset = 0;
    for (size_t i = 0; i < partSizes.size(); ++i) {
        inputBuffers.push_back(createCstPart(cumulativeOffset, partSizes[i], "part_" + std::to_string(i + 1)));
        cumulativeOffset += partSizes[i];
    }
    auto outputBuffers = getReplacementBuffers(dmaOp.getOutputBuff(), tileDim, partSizes, builder);
    SmallVector<mlir::Value> removableInputs;
    removableInputs.push_back(dmaOp.getInput());
    removableInputs.push_back(dmaOp.getOutputBuff());

    return replaceDmaWithMultipleParts(taskOp, dmaOp, inputBuffers, outputBuffers, removableInputs, builder, log);
}

SmallVector<VPUIP::DMATypeOpInterface> splitDMATask(
        VPURT::TaskOp taskOp, VPUIP::DMATypeOpInterface dmaOp, mlir::Value origInput, Dim tileDim,
        const SmallVector<int64_t>& partSizes,
        std::function<BuffersVec(mlir::Value, Dim, const SmallVector<int64_t>&, mlir::OpBuilder)>
                getReplacementInputBuffers,
        mlir::OpBuilder builder, Logger log) {
    // Generate input and output buffers
    auto inputBuffers = getReplacementInputBuffers(origInput, tileDim, partSizes, builder);
    auto outputBuffers = getReplacementBuffers(dmaOp.getOutputBuff(), tileDim, partSizes, builder);

    SmallVector<mlir::Value> removableInputs = {origInput, dmaOp.getOutputBuff()};
    return replaceDmaWithMultipleParts(taskOp, dmaOp, inputBuffers, outputBuffers, removableInputs, builder, log);
}

std::optional<Dim> getDMATilingDim(VPUIP::DMATypeOpInterface dmaOp) {
    if (auto nnDMAOp = mlir::dyn_cast<VPUIP::NNDMAOp>(dmaOp.getOperation())) {
        return VPUIP::getCopyDMATilingDim(nnDMAOp);
    } else if (auto convertDMAOp = mlir::dyn_cast<VPUIP::ConvertDMAOp>(dmaOp.getOperation())) {
        return VPUIP::getCopyDMATilingDim(convertDMAOp);
    } else if (auto gatherDmaOp = mlir::dyn_cast<VPUIP::GatherDMAOp>(dmaOp.getOperation())) {
        const auto indicesType = mlir::cast<NDTypeInterface>(gatherDmaOp.getIndices().getType());
        return getHighestNonTrivialDim(indicesType.getShape(), indicesType.getDimsOrder());
    }

    return std::nullopt;
}

SmallVector<VPUIP::DMATypeOpInterface> handleDMATaskSplit(VPURT::TaskOp taskOp, VPUIP::DMATypeOpInterface nnDMAOp,
                                                          Dim tileDim, const SmallVector<int64_t>& partSizes,
                                                          mlir::OpBuilder builder, Logger log) {
    mlir::Value input = nnDMAOp.getInput();
    if (auto gatherDmaOp = mlir::dyn_cast<VPUIP::GatherDMAOp>(nnDMAOp.getOperation())) {
        input = gatherDmaOp.getIndices();
    }
    if (auto inputCst = input.getDefiningOp<Const::DeclareOp>()) {
        if (!isTrivialConst(inputCst)) {
            return splitFoldedConstToBufferDma(taskOp, nnDMAOp, inputCst, tileDim, partSizes, builder, log);
        }
    }

    auto getReplacementInputBuffers = [&](mlir::Value input, Dim tileDim, const SmallVector<int64_t>& partSizes,
                                          mlir::OpBuilder builder) -> BuffersVec {
        if (auto cstOp = input.getDefiningOp<Const::DeclareOp>()) {
            return getConstantParts(input, tileDim, partSizes, builder);
        } else {
            return getReplacementBuffers(input, tileDim, partSizes, builder);
        }
    };

    return splitDMATask(taskOp, nnDMAOp, input, tileDim, partSizes, getReplacementInputBuffers, builder, log);
}

llvm::SmallVector<int64_t> getSplitPartSizes(NDTypeInterface bufferType, Dim tileDim, int64_t partsNumber) {
    const int64_t tileDimSize = bufferType.getShape()[tileDim];

    VPUX_THROW_UNLESS(partsNumber > 0, "Number of parts must be greater than zero");
    int64_t partSize = tileDimSize / partsNumber;

    // For sub-byte types, ensure the split size is byte-aligned
    const auto elemType = bufferType.getElementType();
    if (elemType.isIntOrFloat()) {
        const auto bitWidth = static_cast<int64_t>(elemType.getIntOrFloatBitWidth());
        if (bitWidth % CHAR_BIT != 0) {
            // Minimum element-count alignment so that (count * bitWidth) is divisible by CHAR_BIT.
            const auto elemCountAlignment = CHAR_BIT / std::gcd<int64_t>(CHAR_BIT, bitWidth);
            // Trailing remainder = tileDimSize - (partsNumber-1) * alignedPartSize. It can only be
            // byte-aligned if tileDimSize itself is a multiple of elemCountAlignment. Otherwise splitting
            // would produce a non-byte-aligned last part and downstream Byte() conversion would throw.
            if (tileDimSize % elemCountAlignment != 0) {
                return {tileDimSize};
            }
            partSize = (partSize / elemCountAlignment) * elemCountAlignment;
        }
    }

    llvm::SmallVector<int64_t> partSizes;
    partSizes.reserve(partsNumber);

    int64_t remainingSize = tileDimSize;
    for (int64_t i = 0; i < partsNumber - 1; ++i) {
        // Sub-byte alignment can round partSize down to 0 (e.g., partSize=1 with alignment=2 yields 0).
        // Skip creating 0-sized parts as they would cause invalid buffer splits.
        if (partSize == 0) {
            break;
        }
        partSizes.push_back(partSize);
        remainingSize -= partSize;
    }
    if (remainingSize > 0) {
        partSizes.push_back(remainingSize);
    }

    VPUX_THROW_WHEN(partSizes.empty(), "getSplitPartSizes mustn't return an empty vector");
    return partSizes;
}

bool checkGatherIndicesAlignment(VPUIP::GatherDMAOp gatherOp, Dim tileDim, size_t noOfParts, Logger log) {
    const auto indicesType = mlir::cast<NDTypeInterface>(gatherOp.getIndices().getType());
    const int64_t indicesTileDimSize = indicesType.getShape()[tileDim] / noOfParts;
    const int64_t remainingIndicesTileDimSize = indicesType.getShape()[tileDim] % noOfParts;
    if (indicesTileDimSize % VPU::INDICES_ALIGNMENT != 0 || remainingIndicesTileDimSize % VPU::INDICES_ALIGNMENT != 0) {
        log.trace("Split would create unaligned indices (size={0}, alignment={1}), skip", indicesTileDimSize,
                  VPU::INDICES_ALIGNMENT);
        return false;
    }

    return true;
}

bool canSplitDma(VPUIP::DMATypeOpInterface dmaOp, size_t noOfParts, Logger log) {
    if (mlir::isa<VPUIP::NNDMAOp, VPUIP::ConvertDMAOp>(dmaOp.getOperation())) {
        // Skip splitting for small transfers that can't fully utilize bandwidth. Don't do that for
        // GatherDMAOp since gather dma will be split on indices so bandwidth won't be affected by the split.
        const auto transferSize = getTotalSize(dmaOp.getInput());
        if (transferSize.count() < MIN_DMA_SIZE_FOR_SPLIT) {
            log.trace("Transfer size {0}B is too small to benefit from splitting (min: {1}B)", transferSize.count(),
                      MIN_DMA_SIZE_FOR_SPLIT);
            return false;
        }
        mlir::Operation* inputOp = dmaOp.getInput().getDefiningOp();
        if (!mlir::isa<VPURT::DeclareBufferOp, Const::DeclareOp>(inputOp)) {
            log.warning("Can't split op because of unsupported source");
            return false;
        }
        return true;
    } else if (auto gatherDmaOp = mlir::dyn_cast<VPUIP::GatherDMAOp>(dmaOp.getOperation())) {
        auto tilingDim = getDMATilingDim(dmaOp);
        if (!tilingDim.has_value()) {
            return false;
        }
        return checkGatherIndicesAlignment(gatherDmaOp, tilingDim.value(), noOfParts, log);
    }
    return false;
}

void handleSingleDMASplit(VPURT::TaskOp taskOp, VPUIP::DMATypeOpInterface dmaOp, int64_t portCount,
                          mlir::OpBuilder builder, Logger log) {
    if (!canSplitDma(dmaOp, 2, log)) {
        return;
    }

    const auto tileDim = getDMATilingDim(dmaOp);
    if (!tileDim.has_value()) {
        log.trace("No valid tiling dimension found, skip");
        return;
    }

    const auto outBuffType = mlir::cast<NDTypeInterface>(dmaOp.getOutputBuff().getType());
    const auto [firstPartSize, secondPartSize] = VPUIP::getSplitPartSizes(outBuffType, tileDim.value());
    if (firstPartSize <= 0 || secondPartSize <= 0) {
        log.trace("Split is not beneficial, only one part");
        return;
    }

    auto newDmas = handleDMATaskSplit(taskOp, dmaOp, tileDim.value(), {firstPartSize, secondPartSize}, builder, log);
    // Distribute split DMA tasks across available DMA ports using round-robin assignment
    // with a group-local counter that tracks all split DMA parts within the same group.
    // The counter increments for each part, ensuring continuous port assignment across
    // all split operations and balancing the load for parallel execution.
    // Example: with 4 ports, splitting 2 DMAs (each into 2 parts):
    //    DMA 1: part_1 -> port_0
    //           part_2 -> port_1
    //    DMA 2: part_1 -> port_2
    //           part_2 -> port_3
    int64_t dmaPort = 0;
    for (auto newDma : newDmas) {
        newDma.setPortAttribute(getIntAttr(newDma.getContext(), dmaPort));
        dmaPort = (dmaPort + 1) % portCount;
    }
}

bool isDmaSegmentedRead(const SmallVector<VPURT::TaskOp>& group) {
    std::unordered_set<int64_t> uniqueSourceClusters;
    for (auto task : group) {
        auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(task.getInnerTaskOp());
        if (!dmaOp) {
            return false;
        }
        auto sourceType = mlir::cast<NDTypeInterface>(dmaOp.getInput().getType());
        if (sourceType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            auto idx = sourceType.getMemSpace().getIndex();
            if (idx.has_value()) {
                uniqueSourceClusters.insert(idx.value());
            } else {
                return false;
            }
        }
    }

    if (uniqueSourceClusters.size() == group.size()) {
        return true;
    }

    return false;
}

//
// Schedules DMAs that read from CMX to avoid parallel reads from the same CMX tile,
// preventing contention on CMX access. Ports are assigned in round-robin order,
// with clusters that have the most tasks given priority. Such groups appear when unrolling
// DMA that has SEGMENTED or OVERLAPPED as input tensor.
// Algorithm:
// 1. Sort cluster task groups in descending order by number of tasks reading from each cluster.
// 2. Assign tasks to ports in round-robin order.
// This algorithm does not guarantee contention elimination for arbitrary task configurations,
// but works well for the configurations encountered in practice, as shown below.
// A typical group to schedule has the following configuration:
// N - number of clusters, Ci - cluster index, SC - number of split clusters, p - number of DMA ports
// C0:      task0, task1, ..., taskn
// C1:      task0, task1, ..., taskn
// ...
// C_{N-SC}: task0
// ...
// C_{N-1}:  task0
//
// Scheduling algorithm outer loop iterates over task index on a cluster(cti) and inner loop goes
// over clusters(ci). This means we can order the tasks in an order that scheduling algorithm sees them:
// task index in scheduling order: ti = cti == 0 ? ci : N + cti * SC + ci
// Distance between 2 tasks in scheduling order is just a number of tasks that scheduling loop goes
// over in between: distance(i, j) = ti - tj
// Contention occurs when the distance between 2 consecutive tasks on the same cluster
// is less than the number of ports. Two cases must be considered when calculating this distance:
//          1. Distance between the first and second task on the cluster.
//          2. Distance between any other 2 consecutive tasks.
// In case 1, not every cluster is split, so the distance between the first and second task
// is increased by the number of clusters with unsplit tasks(N - SC).
// Case 1:
// The distance between two consecutive tasks is N, so contention requires N < p.
// This is typically not the case in the target HW.
// Case 2:
// The distance between two consecutive tasks is SC, so contention requires SC < p.
//
// To avoid contention in case 2, either keep the number of splits to 2, or split more clusters.
// On current HW configurations this case will not materialize, but future logic should be added
// to guard against it.
//
// Example: 3 tiles, 2 DMA ports:
// C0: split_task0, split_task1
// C1: c1_task0
// C2: c2_task0
// Normal round robin port assignment:
// p0: split_task0, c1_task0
// p1: split_task1, c2_task0
// In this assignment contention arises on read from cluster 0.
// optimal port assignment that avoids contention:
// p0: split_task0, c2_task0
// p1: c1_task0, split_task1
//
void scheduleCMXReadTasks(SmallVector<SmallVector<VPURT::TaskOp>>& tasksByCluster, int64_t numDmaPorts) {
    std::stable_sort(tasksByCluster.begin(), tasksByCluster.end(),
                     [](const SmallVector<VPURT::TaskOp>& a, const SmallVector<VPURT::TaskOp>& b) {
                         return a.size() > b.size();
                     });

    int64_t tasksToSchedule =
            std::accumulate(tasksByCluster.begin(), tasksByCluster.end(), int64_t{0}, [](int64_t acc, const auto& g) {
                return acc + static_cast<int64_t>(g.size());
            });

    SmallVector<SmallVector<VPURT::TaskOp>> dmaPortTasks(numDmaPorts);

    int64_t dmaPort = 0;
    for (size_t clusterTaskIdx = 0; tasksToSchedule > 0; clusterTaskIdx++) {
        for (auto& clusterTasks : tasksByCluster) {
            if (clusterTaskIdx < clusterTasks.size()) {
                dmaPortTasks[dmaPort].push_back(clusterTasks[clusterTaskIdx]);
                dmaPort = (dmaPort + 1) % numDmaPorts;
                tasksToSchedule--;
            }
        }
    }

    for (size_t portIdx = 0; portIdx < static_cast<size_t>(numDmaPorts); portIdx++) {
        if (dmaPortTasks[portIdx].empty()) {
            continue;
        }
        auto firstDmaOp = mlir::cast<VPUIP::DMATypeOpInterface>(dmaPortTasks[portIdx][0].getInnerTaskOp());
        firstDmaOp.setPortAttribute(getIntAttr(firstDmaOp.getContext(), portIdx));
        for (size_t dmaIdx = 1; dmaIdx < dmaPortTasks[portIdx].size(); dmaIdx++) {
            auto dmaOp = mlir::cast<VPUIP::DMATypeOpInterface>(dmaPortTasks[portIdx][dmaIdx].getInnerTaskOp());
            dmaOp.setPortAttribute(getIntAttr(dmaOp.getContext(), portIdx));
            dmaPortTasks[portIdx][dmaIdx]->moveAfter(dmaPortTasks[portIdx][dmaIdx - 1]);
        }
    }
}

void handleGroupSplit(std::map<uint64_t, SmallVector<VPURT::TaskOp>>& tasksByUnrollIdxMap, int64_t numDmaPorts,
                      size_t numClusters, Logger log) {
    for (auto& [unrollIdx, dmaTaskGroup] : tasksByUnrollIdxMap) {
        removeUnrollIDxAttr(dmaTaskGroup);

        auto isSegmentedReadGroup = isDmaSegmentedRead(dmaTaskGroup);

        const auto dmaGroupSize = dmaTaskGroup.size();
        const auto unmatchedDMACount = dmaGroupSize % numDmaPorts;
        if (unmatchedDMACount == 0) {
            log.trace(
                    "Unroll idx {0} has {1} DMAs, which is divisible by the number of DMA ports {2}. No need to split.",
                    unrollIdx, dmaGroupSize, numDmaPorts);
            continue;
        }

        VPUX_THROW_WHEN(unmatchedDMACount > dmaGroupSize,
                        "Invalid state: unmatched DMAs count {0} is greater than total DMAs number {1}",
                        unmatchedDMACount, dmaGroupSize);
        const auto firstDMAIdxToSplit = dmaGroupSize - unmatchedDMACount;

        // Calculate number of splits for each unmatched DMA to equally balance the load across DMA ports.
        // Below formula favors doing as few splits as possible on a single DMA task.
        // For instance on a 6 cluster 4 port system below formula will split
        // each of the 2 unmatched DMAs into 2 equal parts for total of 4 DMAs instead of splitting it
        // into 4 parts and creating 8 DMAs.
        auto noOfParts =
                static_cast<size_t>(numDmaPorts) / std::gcd(unmatchedDMACount, static_cast<size_t>(numDmaPorts));
        SmallVector<VPUIP::DMATypeOpInterface> splitDMAs;
        splitDMAs.reserve((dmaGroupSize - firstDMAIdxToSplit) * noOfParts);
        SmallVector<VPURT::TaskOp> unsplitTasks(dmaTaskGroup.begin(), dmaTaskGroup.begin() + firstDMAIdxToSplit);
        for (size_t i = firstDMAIdxToSplit; i < dmaGroupSize; ++i) {
            auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(dmaTaskGroup[i].getInnerTaskOp());
            if (!dmaOp) {
                continue;
            }

            if (!canSplitDma(dmaOp, noOfParts, log)) {
                unsplitTasks.push_back(dmaTaskGroup[i]);
                continue;
            }

            mlir::OpBuilder builder(dmaTaskGroup[i].getOperation());

            auto tileDim = getDMATilingDim(dmaOp);
            if (!tileDim.has_value()) {
                log.trace("No valid tiling dimension found for unroll idx {0}, skip splitting for this group",
                          unrollIdx);
                continue;
            }

            const auto partSizes = getSplitPartSizes(dmaOp.getOutputBuff().getType(), tileDim.value(), noOfParts);
            if (partSizes.size() <= 1) {
                log.trace("Split is not beneficial for unroll idx {0}, only one part", unrollIdx);
                continue;
            }

            auto newDmas = handleDMATaskSplit(dmaTaskGroup[i], dmaOp, tileDim.value(), partSizes, builder, log);
            splitDMAs.append(newDmas.begin(), newDmas.end());
        }

        if (splitDMAs.empty()) {
            continue;
        }

        if (isSegmentedReadGroup) {
            llvm::SmallVector<llvm::SmallVector<VPURT::TaskOp>> clusterTasks(numClusters);
            for (auto dmaTask : unsplitTasks) {
                auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(dmaTask.getInnerTaskOp());
                auto sourceIdx = mlir::cast<NDTypeInterface>(dmaOp.getInput().getType()).getMemSpace().getIndex();
                VPUX_THROW_UNLESS(sourceIdx.has_value(),
                                  "Expected segmented read DMA source memspace index for unsplit task");
                clusterTasks[sourceIdx.value()].push_back(dmaTask);
            }
            for (auto splitDma : splitDMAs) {
                auto sourceIdx = mlir::cast<NDTypeInterface>(splitDma.getInput().getType()).getMemSpace().getIndex();
                VPUX_THROW_UNLESS(sourceIdx.has_value(),
                                  "Expected segmented read DMA source memspace index for split DMA task");
                clusterTasks[sourceIdx.value()].push_back(splitDma->getParentOfType<VPURT::TaskOp>());
            }
            scheduleCMXReadTasks(clusterTasks, numDmaPorts);
        } else {
            // Distribute split DMA tasks across available DMA ports using round-robin assignment
            // with a group-local counter that tracks all split DMA parts within the same group.
            // The counter increments for each part, ensuring continuous port assignment across
            // all split operations and balancing the load for parallel execution.
            // Example: with 4 ports, splitting 2 DMAs (each into 2 parts):
            //    DMA 1: part_1 -> port_0
            //           part_2 -> port_1
            //    DMA 2: part_1 -> port_2
            //           part_2 -> port_3
            int64_t dmaPort = 0;
            for (auto dma : splitDMAs) {
                dma.setPortAttribute(getIntAttr(dma.getContext(), dmaPort));
                dmaPort = (dmaPort + 1) % numDmaPorts;
            }
        }
    }
}

//
// SplitDMAToBalanceLoad
//

class SplitDMAToBalanceLoad final : public VPUIP::impl::SplitDMAToBalanceLoadBase<SplitDMAToBalanceLoad> {
public:
    explicit SplitDMAToBalanceLoad(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitDMAToBalanceLoad::safeRunOnFunc() {
    auto func = getOperation();
    const auto dmaPortCount = config::getNumOfDMAPorts(func);
    const auto numClusters = config::getTileExecutor(func).getCount();
    if (dmaPortCount < 2) {
        return;
    }

    std::map<uint64_t, SmallVector<VPURT::TaskOp>> nnDMAByUnrollIdxMap;
    std::map<uint64_t, SmallVector<VPURT::TaskOp>> gatherDMAByUnrollIdxMap;

    func->walk([&](VPURT::TaskOp taskOp) {
        mlir::OpBuilder builder(taskOp.getOperation());

        // Do group split first. This balances the tasks on architectures that have tile
        // count indivisible by DMA port count.
        if (auto nnDMAOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(taskOp.getInnerTaskOp())) {
            // Current DMA split algorithm requires a flat DMA which would make it incompatible
            // with dynamic strides which requiers DMA to have as many dimensions as the IO tensor. After DMA split
            // algorithm is updated to handle split for non-flat DMAs this check can be removed #E194757
            if (nnDMAOp->getAttr(stridedInputAttrName) != nullptr ||
                nnDMAOp->getAttr(stridedOutputAttrName) != nullptr) {
                removeUnrollIDxAttr(nnDMAOp);
                return;
            }

            const bool hasUnrollIdx = nnDMAOp->hasAttr(VPUIP::UNROLL_IDX);
            if (!hasUnrollIdx) {
                return;
            }
            _log.trace("Found group split candidate at '{0}'", nnDMAOp->getLoc());

            const auto unrollIdx = nnDMAOp->getAttrOfType<mlir::IntegerAttr>(VPUIP::UNROLL_IDX).getInt();
            if (mlir::isa<VPUIP::NNDMAOp>(nnDMAOp.getOperation())) {
                nnDMAByUnrollIdxMap[unrollIdx].push_back(taskOp);
            } else if (mlir::isa<VPUIP::GatherDMAOp>(nnDMAOp.getOperation())) {
                gatherDMAByUnrollIdxMap[unrollIdx].push_back(taskOp);
            }
            return;
        }
    });

    if (!nnDMAByUnrollIdxMap.empty()) {
        handleGroupSplit(nnDMAByUnrollIdxMap, dmaPortCount, numClusters, _log.nest());
    }
    if (!gatherDMAByUnrollIdxMap.empty()) {
        handleGroupSplit(gatherDMAByUnrollIdxMap, dmaPortCount, numClusters, _log.nest());
    }

    // After group split is done additionally split tasks that were detected in DetectDmaSplitCandidate pass.
    // This should be simplified once we move large unbalanced DMA detection to this pass(should be done after group
    // split anyway)
    func->walk([&](VPURT::TaskOp taskOp) {
        mlir::OpBuilder builder(taskOp.getOperation());

        if (auto nnDMAOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp())) {
            // Current DMA split algorithm requires a flat DMA which would make it incompatible
            // with dynamic strides which requiers DMA to have as many dimensions as the IO tensor. After DMA split
            // algorithm is updated to handle split for non-flat DMAs this check can be removed #E194757
            if (nnDMAOp->getAttr(stridedInputAttrName) != nullptr ||
                nnDMAOp->getAttr(stridedOutputAttrName) != nullptr) {
                return;
            }

            const bool hasSplitCandidate =
                    nnDMAOp.getSplitCandidate().has_value() && nnDMAOp.getSplitCandidate().value();
            if (!hasSplitCandidate) {
                return;
            }
            _log.trace("Found split candidate at '{0}'", nnDMAOp->getLoc());

            handleSingleDMASplit(taskOp, nnDMAOp, dmaPortCount, builder, _log.nest());
            return;
        }

        if (auto gatherDMAOp = mlir::dyn_cast<VPUIP::GatherDMAOp>(taskOp.getInnerTaskOp())) {
            VPUX_THROW_UNLESS(gatherDMAOp.getPort().has_value(), "Gather DMA at '{0}' has no portId",
                              gatherDMAOp->getLoc());
            // Current DMA split algorithm requires a flat DMA which would make it incompatible
            // with dynamic strides which requires DMA to have as many dimensions as the IO tensor. After DMA split
            // algorithm is updated to handle split for non-flat DMAs this check can be removed #E194757
            if (gatherDMAOp->getAttr(stridedInputAttrName) != nullptr ||
                gatherDMAOp->getAttr(stridedOutputAttrName) != nullptr) {
                return;
            }
            const bool hasSplitCandidate =
                    gatherDMAOp.getSplitCandidate().has_value() && gatherDMAOp.getSplitCandidate().value();
            if (!hasSplitCandidate) {
                return;
            }
            _log.trace("Found split candidate at '{0}'", gatherDMAOp->getLoc());

            handleSingleDMASplit(taskOp, gatherDMAOp, dmaPortCount, builder, _log.nest());
            return;
        }

        if (auto convertDMAOp = mlir::dyn_cast<VPUIP::ConvertDMAOp>(taskOp.getInnerTaskOp())) {
            if (convertDMAOp->getAttr(stridedInputAttrName) != nullptr ||
                convertDMAOp->getAttr(stridedOutputAttrName) != nullptr) {
                return;
            }
            const bool hasSplitCandidate =
                    convertDMAOp.getSplitCandidate().has_value() && convertDMAOp.getSplitCandidate().value();
            if (!hasSplitCandidate) {
                return;
            }
            _log.trace("Found split candidate at '{0}'", convertDMAOp->getLoc());

            handleSingleDMASplit(taskOp, convertDMAOp, dmaPortCount, builder, _log.nest());
            return;
        }
    });

    _log.trace("Done");
}

}  // namespace

//
// createSplitDMAToBalanceLoadPass
//

std::unique_ptr<mlir::Pass> VPUIP::createSplitDMAToBalanceLoadPass(Logger log) {
    return std::make_unique<SplitDMAToBalanceLoad>(log);
}
