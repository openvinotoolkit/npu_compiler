//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/utils/dma_transaction_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <utility>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_LEGALIZENNDMADIMSIZES
#define GEN_PASS_DEF_LEGALIZENNDMADIMSIZES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

constexpr int64_t BURST_BUFFER_SIZE =
        1024;  // 1 KiB, the maximum number of bytes a single burst can transfer efficiently
constexpr int64_t MAX_TASK_BYTE_BUDGET =
        2 * 1024 * 1024;  // 2 MiB, the maximum number of bytes a single task can transfer which is ~100us
static_assert(MAX_TASK_BYTE_BUDGET > 0, "Byte budget must be greater than 0!");

struct SplitDecision {
    int64_t tileDim = -1;        // Logical dimension index to tile along
    int64_t numParts = -1;       // Number of parts to split into
    int64_t planesPerTile = -1;  // Planes assigned to every non-trailing tile (already byte-aligned)
};

struct DMAContigChunk {
    int64_t contigBytes = 0;     // Size in bytes of one innermost contiguous transfer chunk.
    int64_t outerMemDimIdx = 0;  // Memory-order dimension that repeats the contiguous chunk.
};

std::optional<Dim> getDMATilingDimForRequiredParts(vpux::NDTypeInterface type, int64_t noParts) {
    const auto shape = type.getShape();
    const auto order = type.getDimsOrder();

    for (size_t index = 0; index < shape.size(); ++index) {
        const auto dim = order.toDim(MemDim(index));
        if (shape[dim] >= noParts) {
            return dim;
        }
    }

    return std::nullopt;
}

std::optional<DMAContigChunk> analyzeContigChunk(vpux::NDTypeInterface type, const DMAPattern& reducedPattern) {
    if (reducedPattern.dims.size() <= 1) {
        return std::nullopt;
    }

    const auto contigBytes = reducedPattern.dims.back();
    const auto origMemShape = type.getMemShape();
    const auto rank = static_cast<int64_t>(origMemShape.size());
    // Walk in bits to support sub-byte element types (e.g. i4). The contiguous chunk always spans a whole
    // number of bytes on any strided memref, so the eventual match happens at a byte-aligned boundary.
    const int64_t elemBits = type.getElemTypeSize().count();
    const int64_t contigBits = contigBytes * CHAR_BIT;
    int64_t accBits = elemBits;
    for (int64_t i = rank - 1; i >= 0; --i) {
        if (accBits == contigBits) {
            // Found the boundary between the contiguous chunk and the plane iteration.
            // Skip any degenerate (size == 1) outer memdims: splitting them yields a single
            // non-empty tile of the same size plus (numParts-1) empty tiles and does not
            // reduce per-tile payload. Walk outward until we find a splittable dim.
            int64_t outerIdx = i;
            while (outerIdx > 0 && origMemShape[MemDim(outerIdx)] == 1) {
                --outerIdx;
            }
            return DMAContigChunk{contigBytes, outerIdx};
        }
        accBits *= origMemShape[MemDim(i)];
    }

    return DMAContigChunk{contigBytes, 0};
}

// Compute planesPerTile and align it upward so that per-tile byte offsets are byte-aligned. Relevant for
// sub-byte element types (e.g. i4) whose stride at tileDim is not a whole number of bytes; a no-op when
// the stride is already a byte multiple. May reduce numParts. Assumes input and output share element
// type, so the alignment required on the opposite side is the same.
SplitDecision alignPlanesToByte(SplitDecision result, vpux::NDTypeInterface type) {
    if (result.numParts <= 1 || result.tileDim < 0) {
        return result;
    }
    const auto tileDim = Dim(result.tileDim);
    const auto totalPlanes = type.getShape()[tileDim];
    const auto strideBits = type.getStrides()[tileDim].count();
    const auto planeAlign = CHAR_BIT / std::gcd<int64_t>(strideBits, CHAR_BIT);
    result.planesPerTile = vpux::alignValUp<int64_t>(vpux::divUp(totalPlanes, result.numParts), planeAlign);
    result.numParts = vpux::divUp(totalPlanes, result.planesPerTile);
    return result;
}

// Split contiguous payload bytes: divides the transfer into tiles based on the byte budget.
SplitDecision splitContiguousTransfer(vpux::NDTypeInterface inType) {
    const int64_t payloadBytes = inType.getCompactAllocSize().count();
    if (payloadBytes <= MAX_TASK_BYTE_BUDGET) {
        // fits into the budget
        return {};
    }

    const auto numTiles = vpux::divUp(payloadBytes, MAX_TASK_BYTE_BUDGET);
    if (numTiles <= 1) {
        return {};
    }

    const auto dimOpt = getDMATilingDimForRequiredParts(inType, numTiles);
    if (!dimOpt.has_value()) {
        // no splittable dim exists, cannot tile
        return {};
    }

    return alignPlanesToByte(SplitDecision{dimOpt.value().ind(), numTiles}, inType);
}

bool isStrided(vpux::NDTypeInterface type) {
    return !StrideReqs::compact(type.getRank()).checkStrides(type);
}

SplitDecision getSplitDecision(VPUIP::NNDMAOp nndmaOp, vpux::NDTypeInterface type, bool scaleBudget) {
    if (!isStrided(type)) {
        return splitContiguousTransfer(type);
    }

    auto pattern = reduceDimsForDma(to_small_vector(type.getMemShape()), to_small_vector(type.getMemStrides()),
                                    type.getElemTypeSize().count(), false);
    if (config::getArch(nndmaOp.getOperation()) == config::ArchKind::NPU37XX) {
        patchDimsForNPU37XX(pattern);
    }
    VPUX_THROW_WHEN(pattern.dims.size() != pattern.strides.size(),
                    "Non matching rank between dims {0} and strides {1} for input", pattern.dims.size(),
                    pattern.strides.size());

    auto chunk = analyzeContigChunk(type, pattern);
    if (!chunk.has_value()) {
        // reduceDimsForDma collapsed all dims (e.g. N=1 with a padded stride) into a single contiguous
        // descriptor chunk so treat this as a flat contiguous transfer.
        return splitContiguousTransfer(type);
    }
    VPUX_THROW_WHEN(chunk->contigBytes <= 0, "Strided DMA '{0}' has invalid contiguous chunk size {1} B",
                    nndmaOp->getLoc(), chunk->contigBytes);

    const auto scaledBudget =
            scaleBudget ? std::min<int64_t>(MAX_TASK_BYTE_BUDGET,
                                            (2 * MAX_TASK_BYTE_BUDGET * chunk->contigBytes) / BURST_BUFFER_SIZE)
                        : MAX_TASK_BYTE_BUDGET;
    const auto payloadBytes = type.getCompactAllocSize().count();

    SplitDecision result;
    if (scaledBudget >= chunk->contigBytes) {
        const auto elPerPart = scaledBudget / chunk->contigBytes;
        const auto outerPlaneCount = payloadBytes / chunk->contigBytes;
        VPUX_THROW_WHEN(outerPlaneCount <= 0, "Strided DMA '{0}' has invalid outer plane count {1}", nndmaOp->getLoc(),
                        outerPlaneCount);
        result.numParts = vpux::divUp(outerPlaneCount, elPerPart);
        result.tileDim = type.getDimsOrder().toDim(MemDim(chunk->outerMemDimIdx)).ind();
    } else {
        // The contiguous chunk is larger than the scaled budget, so we need to split the contiguous
        // chunk itself.
        result.numParts = vpux::divUp(payloadBytes, scaledBudget);

        const auto rank = static_cast<int64_t>(type.getMemShape().size());
        const auto order = type.getDimsOrder();
        const auto& shape = type.getShape();
        for (int64_t memIdx = chunk->outerMemDimIdx + 1; memIdx < rank; ++memIdx) {
            const auto candidate = order.toDim(MemDim(memIdx));
            if (shape[candidate] >= result.numParts) {
                result.tileDim = candidate.ind();
                break;
            }
        }
    }
    result = alignPlanesToByte(result, type);

    return result;
}

// Determine preemption split: ensures each DMA task stays within the byte budget so firmware can preempt at
// predictable boundaries. For strided DMAs the split respects the contiguous inner chunk boundary.
SplitDecision getDmaSplitForPreemption(VPUIP::NNDMAOp nndmaOp, vpux::NDTypeInterface inType,
                                       vpux::NDTypeInterface outType) {
    const auto inTransType = inType.getMemoryKind();
    const auto outTransType = outType.getMemoryKind();

    using namespace vpux::VPU;
    const bool inIsDDR = (inTransType == MemoryKind::DDR);
    const bool inIsCMX = (inTransType == MemoryKind::CMX_NN);
    const bool outIsDDR = (outTransType == MemoryKind::DDR);
    const bool outIsCMX = (outTransType == MemoryKind::CMX_NN);

    if ((inIsDDR && outIsDDR) || (inIsDDR && outIsCMX)) {  // DDR -> DDR or DDR -> CMX
        // tiling is calculated for input and applied for output
        return getSplitDecision(nndmaOp, inType, /*scaleBudget=*/true);
    } else if (inIsCMX && outIsCMX) {  // CMX -> CMX
        // tiling is calculated for input and applied for output
        return getSplitDecision(nndmaOp, inType, /*scaleBudget=*/false);
    } else if (inIsCMX && outIsDDR) {  // CMX -> DDR
        // tiling is calculated for DDR and applied for CMX transfer
        return getSplitDecision(nndmaOp, outType, /*scaleBudget=*/true);
    }

    VPUX_THROW("Unsupported transfer type: {0} -> {1}", inTransType, outTransType);
}

//
// LegalizeNNDMADimSizesRewriter
//

class LegalizeNNDMADimSizesRewriter final : public mlir::OpRewritePattern<VPUIP::NNDMAOp> {
public:
    LegalizeNNDMADimSizesRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::NNDMAOp>(ctx), _log(std::move(log)) {
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp nndmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    void legalizeDMA(VPUIP::NNDMAOp nndmaOp, const SplitDecision& tiling, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

void LegalizeNNDMADimSizesRewriter::legalizeDMA(VPUIP::NNDMAOp nndmaOp, const SplitDecision& tiling,
                                                mlir::PatternRewriter& rewriter) const {
    auto inputDeclBuff = nndmaOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    auto outputDeclBuff = nndmaOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
    auto taskOp = nndmaOp->getParentOfType<VPURT::TaskOp>();

    VPUX_THROW_WHEN(taskOp == nullptr, "NNDMAOp '{0}' is not wrapped into a VPURT.Task", nndmaOp->getLoc());
    rewriter.setInsertionPointAfter(taskOp);

    const auto tileDim = Dim(tiling.tileDim);
    const auto totalPlanes = getShape(nndmaOp.getInput())[tileDim];
    const auto planesPerTile = tiling.planesPerTile;
    const auto numParts = tiling.numParts;
    VPUX_THROW_WHEN(planesPerTile <= 0 || numParts <= 0,
                    "NNDMAOp '{0}': invalid SplitDecision (planesPerTile={1}, numParts={2})", nndmaOp->getLoc(),
                    planesPerTile, numParts);

    Byte inputOffset{inputDeclBuff.getByteOffset()};
    Byte outputOffset{outputDeclBuff.getByteOffset()};
    auto inputInsertionPoint = nndmaOp.getInput().getDefiningOp();
    auto outputInsertionPoint = nndmaOp.getOutputBuff().getDefiningOp();

    const auto getTiledBuf = [&](VPURT::DeclareBufferOp origBuf, int64_t dimOffset, int64_t newDimSize,
                                 vpux::Byte newByteOffset, mlir::Operation* insertionPoint) -> VPURT::DeclareBufferOp {
        auto origType = mlir::cast<vpux::NDTypeInterface>(origBuf.getType());
        const auto origStrides = origType.getStrides();
        auto newType = VPUIP::getNewBufferType(origType, tileDim, dimOffset, newDimSize)
                               .changeStrides(vpux::StridesRef(origStrides));
        return VPUIP::createNewDeclareBuffer(rewriter, insertionPoint, origBuf, newType, newByteOffset.count());
    };

    const auto port = nndmaOp.getPort().value_or(0);

    const auto isOutOfOrder = nndmaOp.getIsOutOfOrder();
    const auto isCritical = nndmaOp.getIsCritical();
    const auto spillIdAttr = nndmaOp.getSpillIdAttr();
    const auto compressCandidate = nndmaOp.getCompressCandidate();
    const auto profilingBufferMgmt = nndmaOp.getProfilingBufferMgmt();

    auto planesLeft = totalPlanes;
    for (int64_t partIdx = 0; partIdx < numParts; ++partIdx) {
        const auto curPlanes = std::min(planesLeft, planesPerTile);
        const auto dimOffset = totalPlanes - planesLeft;

        auto newInBuff = getTiledBuf(inputDeclBuff, dimOffset, curPlanes, inputOffset, inputInsertionPoint);
        inputInsertionPoint = newInBuff.getResult().getDefiningOp();

        auto newOutBuff = getTiledBuf(outputDeclBuff, dimOffset, curPlanes, outputOffset, outputInsertionPoint);
        outputInsertionPoint = newOutBuff.getResult().getDefiningOp();

        // Advance byte offsets for the next iteration. Skip on the last iteration because the values are not
        // consumed and, for sub-byte types, curPlanes on the last (short) tile may not be byte-aligned.
        if (partIdx + 1 < numParts) {
            const auto inStrides = mlir::cast<vpux::NDTypeInterface>(newInBuff.getType()).getStrides();
            inputOffset += Byte(planesPerTile * inStrides[tileDim]);
            const auto outStrides = mlir::cast<vpux::NDTypeInterface>(newOutBuff.getType()).getStrides();
            outputOffset += Byte(planesPerTile * outStrides[tileDim]);
        }

        // The tiles run sequentially on the same DMA port (FIFO order), so only the first tile waits on the
        // original input barriers and only the last tile updates the original output barriers. The intermediate
        // tiles need no synchronization.
        const auto waitBarriers = (partIdx == 0) ? mlir::ValueRange(taskOp.getWaitBarriers()) : mlir::ValueRange();
        const auto updateBarriers =
                (partIdx == numParts - 1) ? mlir::ValueRange(taskOp.getUpdateBarriers()) : mlir::ValueRange();

        const auto tileLoc = appendLoc(nndmaOp->getLoc(), "preempt_split_{0}", partIdx + 1);
        auto newDmaOp = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(rewriter, waitBarriers, updateBarriers, tileLoc,
                                                              newInBuff, newOutBuff, port, isOutOfOrder, isCritical,
                                                              spillIdAttr, compressCandidate);

        if (nndmaOp->hasAttr(vpux::stridedInputAttrName)) {
            newDmaOp->setAttr(vpux::stridedInputAttrName, nndmaOp->getAttr(vpux::stridedInputAttrName));
        }
        if (nndmaOp->hasAttr(vpux::stridedOutputAttrName)) {
            newDmaOp->setAttr(vpux::stridedOutputAttrName, nndmaOp->getAttr(vpux::stridedOutputAttrName));
        }
        if (profilingBufferMgmt) {
            newDmaOp.setProfilingBufferMgmt(true);
        }

        planesLeft -= curPlanes;
    }

    VPUX_THROW_UNLESS(planesLeft == 0, "[{0}]: part of the original shape was not covered by tiles", nndmaOp->getLoc());
    rewriter.eraseOp(taskOp);
}

mlir::LogicalResult LegalizeNNDMADimSizesRewriter::matchAndRewrite(VPUIP::NNDMAOp nndmaOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    // TODO: This pass currently tiles NNDMA along at most one dimension. As a result, some DMAs that exceed the
    // byte budget still remain unsplit when no single dimension can be partitioned to satisfy the budget.
    // These corner cases, as well as constants and PermuteDMAs will be handled under E#228871.
    if (nndmaOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>() == nullptr ||
        nndmaOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>() == nullptr) {
        _log.debug("[{0}] Skip split: DMA endpoints are not direct VPURT.DeclareBuffer values", nndmaOp->getLoc());
        return mlir::failure();
    }
    if (nndmaOp->getParentOfType<VPURT::TaskOp>() == nullptr) {
        _log.debug("[{0}] Skip split: NNDMA is not wrapped in VPURT.Task", nndmaOp->getLoc());
        return mlir::failure();
    }

    const auto stridedInAttr = nndmaOp->getAttr(vpux::stridedInputAttrName);
    const auto stridedOutAttr = nndmaOp->getAttr(vpux::stridedOutputAttrName);

    if (stridedInAttr || stridedOutAttr) {
        _log.debug("[{0}] Skip split: DMA is strided or has dynamic strides", nndmaOp->getLoc());
        return mlir::failure();
    }

    const auto inShape = getShape(nndmaOp.getInput());
    const auto outShape = getShape(nndmaOp.getOutput());
    if (inShape != outShape) {
        _log.debug("[{0}] Skip split: input/output shapes differ ({1} vs {2})", nndmaOp->getLoc(), inShape, outShape);
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(nndmaOp.getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(nndmaOp.getOutput().getType());

    const auto tiling = getDmaSplitForPreemption(nndmaOp, inType, outType);

    if (tiling.numParts <= 1 || tiling.tileDim < 0) {
        _log.debug("[{0}] No split decision -> leave DMA as-is", nndmaOp->getLoc());
        return mlir::failure();
    }

    _log.trace("[{0}] Applying split: tileDim={1}, numParts={2}", nndmaOp->getLoc(), tiling.tileDim, tiling.numParts);
    legalizeDMA(nndmaOp, tiling, rewriter);
    return mlir::success();
}

//
// LegalizeNNDMADimSizesPass
//

class LegalizeNNDMADimSizesPass final : public VPUIP::impl::LegalizeNNDMADimSizesBase<LegalizeNNDMADimSizesPass> {
public:
    LegalizeNNDMADimSizesPass() = default;
    explicit LegalizeNNDMADimSizesPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final {
        if (mlir::failed(Base::initialize(ctx))) {
            return mlir::failure();
        }
        return mlir::success();
    };

private:
    void safeRunOnFunc() final;
};

void LegalizeNNDMADimSizesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    _log.trace("Running LegalizeNNDMADimSizesPass on function '{0}'", func.getName());

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LegalizeNNDMADimSizesRewriter>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createLegalizeNNDMADimSizesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createLegalizeNNDMADimSizesPass(Logger log) {
    return std::make_unique<LegalizeNNDMADimSizesPass>(std::move(log));
}
