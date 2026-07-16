//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <map>
#include <optional>
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_FLASHSDPATILING
#define GEN_PASS_DEF_FLASHSDPATILING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// Transient op-attached attributes used to communicate the tiling strategy from
// the annotate walk to the two rewriters. The Q rewriter consumes
// head_and_q_num_tiles (a [headTiles, querySeqTiles] array, heads-first) and
// preserves kv_num_tiles on cloned Q tiles for the KV rewriter.
constexpr llvm::StringLiteral HEAD_AND_Q_NUM_TILES_ATTR_NAME = "head_and_q_num_tiles";
constexpr llvm::StringLiteral KV_NUM_TILES_ATTR_NAME = "kv_num_tiles";

// Explicit tiling strategy with semantic dimension names
struct FlashSDPATilingStrategy {
    int64_t headTiles{1};      // Number of tiles on the Heads (C) dimension
    int64_t querySeqTiles{1};  // Number of tiles on Query sequence length (H)
    int64_t kvNumBlocks{1};    // Number of KV sequence unrolls
};

struct FlashSDPAHeadTiling {
    int64_t alignedGroupSize{1};
    int64_t numTiles{1};
};

class FlashSDPAQTilingRewrite final : public mlir::OpRewritePattern<VPU::FlashSDPAOp> {
public:
    FlashSDPAQTilingRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::FlashSDPAOp>(ctx), _log(log) {
        setDebugName("FlashSDPAQTilingRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::FlashSDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class FlashSDPAKVUnrollRewrite final : public mlir::OpRewritePattern<VPU::FlashSDPAOp> {
public:
    FlashSDPAKVUnrollRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::FlashSDPAOp>(ctx), _log(log) {
        setDebugName("FlashSDPAKVUnrollRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::FlashSDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Estimate if we apply isolated tiling with the given Query number of slices
// how many times we would have to tile on Key/Value to fit everything in CMX
std::optional<int64_t> estimateRequiredKvNumBlocks(VPU::FlashSDPAOp origOp, ArrayRef<NDTypeInterface> tiledTensors,
                                                   Byte reservedMem) {
    if (origOp.fitIntoCMXAfterKeyValueTiling(tiledTensors, reservedMem, /*kvNumBlocks*/ 1)) {
        return 1;
    }

    const auto keyShape = getShape(origOp.getKey());
    const auto sourceSeqLen = keyShape[Dims4D::Act::H];

    const auto keyType = mlir::cast<NDTypeInterface>(origOp.getKey().getType());
    const auto elemType = keyType.getElementType();
    const auto alignment = vpux::VPU::NCEInvariant::getAlignment(elemType);

    auto kvNumBlocks = int64_t{1};
    auto dimSize = sourceSeqLen;
    while (dimSize > alignment) {
        kvNumBlocks = divUp(sourceSeqLen, dimSize - alignment);
        dimSize = alignValUp(divUp(sourceSeqLen, kvNumBlocks), alignment);
        const auto effectiveKvNumBlocks = divUp(sourceSeqLen, dimSize);

        if (origOp.fitIntoCMXAfterKeyValueTiling(tiledTensors, reservedMem, effectiveKvNumBlocks)) {
            return effectiveKvNumBlocks;
        }
    }

    // No split on Key/Value was found that would fit CMX
    return std::nullopt;
}

// Operand indices that are pipelined (duplicated) across tiled operations.
// These need reserved CMX memory for double-buffering during tiling estimation.
SmallVector<int64_t> getPipelinedBufferIndices(bool hasAttentionMask) {
    auto indices = SmallVector<int64_t>{
            0,   // query
            7,   // input_running_output
            8,   // input_running_max
            9,   // input_running_sum
            10,  // attention_mask (optional)
            11,  // result_running_output
            12,  // result_running_max
            13,  // result_running_sum
    };

    if (!hasAttentionMask) {
        indices.pop_back();
    }

    return indices;
}

// Number of head tiles is determined purely from input shapes: the kernel requires
// exactly 1 KV head per cluster invocation. getAlignment() returns alignment[C] =
// numClusters for SOK and 1 for Clustering, so per-cluster KV head count after
// distribution is always 1.
FlashSDPAHeadTiling computeHeadTiling(VPU::FlashSDPAOp origOp) {
    const auto keyShape = getShape(origOp.getKey());
    const auto kvHeads = keyShape[Dims4D::Act::C];

    const auto resultShape = getShape(origOp.getResultRunningOutput());
    const auto qHeads = resultShape[Dims4D::Act::C];

    const auto alignment = getAlignment(origOp.getOperation(), {}, {});

    VPUX_THROW_UNLESS(qHeads % kvHeads == 0,
                      "Incorrect '{0}' configurations. Query heads dimension = '{1}' must be divisible by Key/Value "
                      "heads dimension '{2}'",
                      origOp->getName(), qHeads, kvHeads);

    const auto groupSize = qHeads / kvHeads;
    const auto alignedGroupSize = groupSize * alignment[Dims4D::Act::C.ind()];
    return FlashSDPAHeadTiling{alignedGroupSize, divUp(qHeads, alignedGroupSize)};
}

// Estimate the query sequence tiling and KV blocks needed for a given head tile size,
// always reserving memory for query pipelining.
//
// CMX layout for 1 operation with pipelining:
// [ ======== Total CMX Size ======== ]        v-- do not duplicate Shared buffers for pipelined ops
// [ Shared | Pipelined0 | Pipelined1 ] + [ Shared ]
// [     CMX for 1 Op    ]     ^-- reserved CMX
std::optional<FlashSDPATilingStrategy> estimateTiling(VPU::FlashSDPAOp origOp, bool enablePipelining, Logger log) {
    const auto resultShape = getShape(origOp.getResultRunningOutput());
    const auto alignment = getAlignment(origOp.getOperation(), {}, {});
    const auto headTiling = computeHeadTiling(origOp);

    // Unroll on the Heads dimension (see computeHeadTiling for the 1-KV-head-per-cluster rationale).
    auto tiledResultShape = Shape(resultShape);
    tiledResultShape[Dims4D::Act::C] = headTiling.alignedGroupSize;

    // Now try to find query sequence tiling that fits CMX
    const auto seqLenDimSize = resultShape[Dims4D::Act::H];
    const auto seqAlignment = alignment[Dims4D::Act::H.ind()];

    auto querySeqTiles = int64_t{1};
    auto curSeqSize = seqLenDimSize;
    auto kvNumBlocks = std::optional<int64_t>{};
    auto swOp = mlir::cast<VPU::SWOpInterface>(origOp.getOperation());

    while (true) {
        auto tiledTensorTypes = getAllOperandsSwInterface(swOp, TileInfo{tiledResultShape}, log);

        const auto hasAttentionMask = (static_cast<int64_t>(tiledTensorTypes.size()) == 14);
        auto pipelinedBuffersIndices = getPipelinedBufferIndices(hasAttentionMask);

        auto reservedMemory = Byte{0};
        if (enablePipelining) {
            for (auto index : pipelinedBuffersIndices) {
                reservedMemory += tiledTensorTypes[index].getTotalAllocSize();
            }
        }

        kvNumBlocks = estimateRequiredKvNumBlocks(origOp, tiledTensorTypes, reservedMemory);

        // Optimal strategy is to not tile on Key/Value tensors if possible
        if (kvNumBlocks.has_value() && kvNumBlocks.value() == 1) {
            break;
        }

        if (curSeqSize <= seqAlignment) {
            break;
        }

        querySeqTiles = divUp(seqLenDimSize, curSeqSize - seqAlignment);
        curSeqSize = alignValUp(divUp(seqLenDimSize, querySeqTiles), seqAlignment);

        tiledResultShape[Dims4D::Act::H] = curSeqSize;
    }

    if (!kvNumBlocks.has_value()) {
        return std::nullopt;
    }

    return FlashSDPATilingStrategy{headTiling.numTiles, querySeqTiles, kvNumBlocks.value()};
}

// Mirrors the generic VPU::applyTileStrategy (generate_tiling.cpp) but adds an input-tile
// cache so identical operand slices are emitted only once: e.g. all query-sequence tiles of
// a single head share the same key/value slice. The LIT test relies on this reuse (cloned Q
// tiles point at the same Slice SSA value), so this specialized copy must not silently
// diverge from the generic version's tiling/concat semantics.
mlir::LogicalResult applyTileStrategyFlashSDPA(VPU::TilingBuilderOpInterface origOp, const OutputTiling& tiles,
                                               mlir::RewriterBase& rewriter, Logger log) {
    const auto results = origOp->getResults();

    auto resultTileValues = SmallVector<SmallVector<mlir::Value>>(results.size());
    auto resultTileOffsets = SmallVector<SmallVector<Shape>>(results.size());

    // Cache tiled inputs by (operandIndex, tile shape, tile offsets).
    // If two output tiles produce the same input slice for an operand, reuse it.
    using TileCacheKey = SmallVector<int64_t>;
    auto tiledInputCache = std::map<TileCacheKey, mlir::Value>();

    for (const auto& outputTile : tiles) {
        auto inputTiling = origOp.backInferTileInfo(outputTile, log);
        auto& inTiles = inputTiling.tiles;

        VPUX_THROW_UNLESS(!inTiles.empty(), "Got empty tile information");

        mlir::IRMapping mapper;
        for (auto [inputIdx, origInput] : origOp->getOperands() | indexed) {
            const auto& inTile = inTiles[inputIdx];

            auto cacheKey = SmallVector<int64_t>{static_cast<int64_t>(inputIdx)};
            cacheKey.append(inTile.shape.begin(), inTile.shape.end());
            cacheKey.append(inTile.offsets.begin(), inTile.offsets.end());

            auto it = tiledInputCache.find(cacheKey);
            mlir::Value tiledInput;
            if (it != tiledInputCache.end()) {
                tiledInput = it->second;
            } else {
                const auto valName = printToString("input {0}", inputIdx);
                tiledInput = vpux::VPU::makeTile(rewriter, origOp->getLoc(), origInput, inTile, valName);
                tiledInputCache[cacheKey] = tiledInput;
            }

            mapper.map(origInput, tiledInput);
        }

        const auto tileLoc = appendLoc(origOp->getLoc(), "output tile {0}", outputTile.offsets);

        auto* tiledOp = rewriter.clone(*origOp, mapper);
        tiledOp->setLoc(tileLoc);

        auto tiledBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(tiledOp);
        VPUX_THROW_WHEN(tiledBuilderOp == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                        tiledOp->getName());

        tiledBuilderOp.adjustAttrs(inputTiling, outputTile);

        vpux::inferReturnTypes(tiledOp, vpux::InferShapedTypeMode::ALL);

        auto tiledResults = tiledOp->getResults();

        const auto outputTiling = origOp.getOutputTiling(outputTile, log);
        VPUX_THROW_UNLESS(results.size() == outputTiling.size(),
                          "Number of results '{0}' doesn't match with number of output tiles '{1}' at '{2}'",
                          results.size(), outputTiling.size(), origOp->getLoc());

        for (const auto i : irange(results.size())) {
            const auto& outputTile = outputTiling[i];
            auto tiledResult = tiledResults[i];

            const auto tiledShape = getShape(tiledResult);
            VPUX_THROW_UNLESS(tiledShape == outputTile.shape,
                              "Inferred output shape '{0}' doesn't match tiled shape '{1}' at '{2}'", tiledShape,
                              outputTile.shape, tiledResult.getDefiningOp()->getLoc());

            const auto resultType = mlir::cast<vpux::NDTypeInterface>(results[i].getType());
            const auto resultDenseTile = resultType.extractDenseTile(outputTile.offsets, outputTile.shape);

            tiledResult.setType(resultDenseTile);

            copyLoopAttributes(origOp, tiledResult.getDefiningOp());

            resultTileValues[i].push_back(tiledResult);
            resultTileOffsets[i].push_back(outputTiling[i].offsets);
        }
    }

    SmallVector<mlir::Value> concatOps;
    for (const auto i : irange(results.size())) {
        auto resultType = origOp->getResult(i).getType();
        auto tileValues = mlir::ValueRange(resultTileValues[i]);
        auto tileOffsets = ArrayRef(resultTileOffsets[i]);

        auto concatOp = rewriter.create<VPU::ConcatOp>(origOp->getLoc(), resultType, tileValues, tileOffsets);

        concatOps.push_back(concatOp.getOutput());
    }

    rewriter.replaceOp(origOp, concatOps);

    return mlir::success();
}

// Estimate the tiling strategy once and stamp it onto the op as two transient
// attributes. The Q rewriter and the KV rewriter each consume their respective
// attribute and clear it. The Q rewriter leaves kv_num_tiles on cloned Q tiles
// so KV unrolling can consume the same strategy per Q tile.
mlir::LogicalResult annotateTilingStrategy(VPU::FlashSDPAOp op, bool enablePipelining, Logger log) {
    auto resultShape = getShape(op.getResult(0));
    if (resultShape.size() < 2) {
        return errorAt(op, "Output shape must at least have a rank 2, got {0}", resultShape.size());
    }

    auto strategy = estimateTiling(op, enablePipelining, log.nest());
    if (!strategy.has_value()) {
        return errorAt(op, "Failed to estimate tiling for FlashSDPA operation. Tensors will not fit CMX.");
    }

    log.trace("Annotated tiling for '{0}': heads={1}, querySeq={2}, kv={3}", op->getLoc(), strategy->headTiles,
              strategy->querySeqTiles, strategy->kvNumBlocks);

    auto* ctx = op->getContext();
    op->setAttr(HEAD_AND_Q_NUM_TILES_ATTR_NAME,
                getIntArrayAttr(ctx, ArrayRef<int64_t>{strategy->headTiles, strategy->querySeqTiles}));
    op->setAttr(KV_NUM_TILES_ATTR_NAME, getIntAttr(ctx, strategy->kvNumBlocks));
    return mlir::success();
}

mlir::LogicalResult FlashSDPAQTilingRewrite::matchAndRewrite(VPU::FlashSDPAOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto headAndQNumTilesAttr = origOp->getAttrOfType<mlir::ArrayAttr>(HEAD_AND_Q_NUM_TILES_ATTR_NAME);
    if (headAndQNumTilesAttr == nullptr) {
        return mlir::failure();
    }

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto log = _log.nest();

    const auto headAndQNumTiles = parseIntArrayAttr<int64_t>(headAndQNumTilesAttr);
    VPUX_THROW_UNLESS(headAndQNumTiles.size() == 2, "Expected '{0}' to hold [headTiles, querySeqTiles], got {1} values",
                      HEAD_AND_Q_NUM_TILES_ATTR_NAME, headAndQNumTiles.size());
    const auto headTiles = headAndQNumTiles[0];
    const auto querySeqTiles = headAndQNumTiles[1];

    // Clear the marker before cloning so the Q-tile clones don't re-trigger this rewriter.
    // The KV_NUM_TILES_ATTR_NAME attribute is left in place so the cloned tiles carry the
    // KV-unrolling decision forward to the KV unrolling stage.
    rewriter.modifyOpInPlace(origOp, [&] {
        origOp->removeAttr(HEAD_AND_Q_NUM_TILES_ATTR_NAME);
    });

    // Build the tiling divisor shape: [N=Batch, C=headTiles, H=querySeqTiles, W=1].
    // Heads (C) precede Q-sequence (H) so, with unrollSpatialFirst=false (NCHW order),
    // tiles are enumerated heads-first then query-sequence.
    const auto firstOutputShape = getShape(origOp.getResultRunningOutput());
    auto tilingStrategy = Shape(firstOutputShape.size(), 1);
    tilingStrategy[Dims4D::Act::N] = firstOutputShape[Dims4D::Act::N];
    tilingStrategy[Dims4D::Act::C] = headTiles;
    tilingStrategy[Dims4D::Act::H] = querySeqTiles;

    const auto alignment = getAlignment(origOp.getOperation(), {}, {});
    const auto unrollSpatialFirst = false;
    const auto firstOutputTiles = fillDividedTiles(tilingStrategy, firstOutputShape, alignment, unrollSpatialFirst);

    if (mlir::failed(firstOutputTiles)) {
        return errorAt(origOp,
                       "Failed to compute tiling for output shape: '{0}', tiling strategy: '{1}', alignment: '{2}'",
                       firstOutputShape, tilingStrategy, alignment);
    }

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    origOp->getName());

    auto result = applyTileStrategyFlashSDPA(tilingBuilder, firstOutputTiles.value(), rewriter, log);
    if (mlir::failed(result)) {
        return errorAt(origOp, "Failed to rewrite original operation with the tiled one");
    }

    return mlir::success();
}

//
// FlashSDPAKVUnrollRewrite
//

mlir::Value createSlice(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value value, Dim dimension,
                        int64_t beginOffset, int64_t endOffset, const Logger& log) {
    auto shape = getShape(value);

    auto sliceOffset = Shape(shape.size(), 0);
    sliceOffset[dimension] = checked_cast<int64_t>(beginOffset);
    auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), sliceOffset);

    auto sliceSize = Shape(shape);
    sliceSize[dimension] = endOffset - beginOffset;
    auto sizesAttr = getIntArrayAttr(rewriter.getContext(), sliceSize);

    log.trace("Created SliceOp with offset {0} and size {1} at {2}", sliceOffset, sliceSize, loc);
    return rewriter.create<VPU::SliceOp>(loc, value, offsetsAttr, sizesAttr);
}

mlir::LogicalResult FlashSDPAKVUnrollRewrite::matchAndRewrite(VPU::FlashSDPAOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    auto kvNumTilesAttr = origOp->getAttrOfType<mlir::IntegerAttr>(KV_NUM_TILES_ATTR_NAME);
    if (kvNumTilesAttr == nullptr) {
        return mlir::failure();
    }

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto log = _log.nest();
    auto ctx = rewriter.getContext();

    const auto kvNumBlocks = parseIntAttr<int64_t>(kvNumTilesAttr);

    rewriter.modifyOpInPlace(origOp, [&] {
        origOp->removeAttr(KV_NUM_TILES_ATTR_NAME);
    });

    if (kvNumBlocks < 2) {
        log.trace("No need to tile the operation");
        return mlir::success();
    }

    // Tiling parameters
    const auto keyShape = getShape(origOp.getKey());
    const auto sourceSeqLen = keyShape[Dims4D::Act::H];

    // MatMul computed as DPU DWConv from SHAVE that requires channel alignment
    // Because we use NCHW layout for the input tensors, the channel dimension is actually the width
    // Second MatMul has Attention scores and Values tensors as an input, with "channels" == sourceSeqLen
    // So we must align SourceSeqLen dimension to have a correct WeightsTable
    const auto keyType = mlir::cast<NDTypeInterface>(origOp.getKey().getType());
    const auto elemType = keyType.getElementType();
    const auto alignment = vpux::VPU::NCEInvariant::getAlignment(elemType);

    const auto tileSize = alignValUp(divUp(sourceSeqLen, kvNumBlocks), alignment);
    const auto effectiveKvNumBlocks = divUp(sourceSeqLen, tileSize);

    // Partial values that are chained through FlashSDPAOp
    auto out = origOp.getInputRunningOutput();
    auto max = origOp.getInputRunningMax();
    auto sum = origOp.getInputRunningSum();

    // Padding on SequenceLength is 0 for all operations except the last one
    auto zeroPadAttr = getIntAttr(rewriter, 0);

    auto query = origOp.getQuery();

    const auto initIsHead = origOp.getIsHead();
    const auto initIsTail = origOp.getIsTail();

    VPU::FlashSDPAOp tiledOp;
    for (auto i = int64_t{0}; i < effectiveKvNumBlocks; ++i) {
        log.trace("Unrolling {0} - {1} / {2} times", origOp->getName(), i + 1, effectiveKvNumBlocks);
        auto beginOffset = i * tileSize;
        auto endOffset = std::min(beginOffset + tileSize, sourceSeqLen);

        auto keySlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "key_slice_{0}", i), origOp.getKey(),
                                    Dims4D::Act::H, beginOffset, endOffset, log);
        auto valueSlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "value_slice_{0}", i), origOp.getValue(),
                                      Dims4D::Act::H, beginOffset, endOffset, log);

        auto attentionMaskSlice = mlir::Value{nullptr};
        if (origOp.getAttentionMask() != nullptr) {
            attentionMaskSlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "attention_mask_slice_{0}", i),
                                             origOp.getAttentionMask(), Dims4D::Act::W, beginOffset, endOffset, log);
        }

        auto isHeadAttr = mlir::BoolAttr::get(ctx, initIsHead && (i == 0));
        auto isTailAttr = mlir::BoolAttr::get(ctx, initIsTail && (i + 1 == effectiveKvNumBlocks));

        auto sourceSeqLenPadSize = (i + 1 == effectiveKvNumBlocks) ? origOp.getSourceSeqLenPadSizeAttr() : zeroPadAttr;

        auto tileLoc = appendLoc(origOp->getLoc(), "flash_sdpa_kv_tile_{0}", i);
        tiledOp = rewriter.create<VPU::FlashSDPAOp>(tileLoc, query, keySlice, valueSlice, out, max, sum,
                                                    attentionMaskSlice, sourceSeqLenPadSize, isHeadAttr, isTailAttr,
                                                    origOp.getMultiClusterStrategyAttr());

        copyLoopAttributes(origOp, tiledOp.getOperation());
        log.trace("Unrolled {0} - {1}", tiledOp->getName(), tiledOp->getResult(0));

        // Propagate intermediate values to the next FlashSDPAOp (unused after the final tile).
        out = tiledOp.getResultRunningOutput();
        max = tiledOp.getResultRunningMax();
        sum = tiledOp.getResultRunningSum();
    }

    rewriter.replaceOp(origOp, tiledOp);
    return mlir::success();
}

//
// FlashSDPATiling
//

class FlashSDPATiling final : public VPU::impl::FlashSDPATilingBase<FlashSDPATiling> {
public:
    explicit FlashSDPATiling(bool enablePipelining, Logger log): _enablePipelining(enablePipelining) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    void applyQTilingPatterns(mlir::Operation* func);
    void applyKVUnrollPatterns(mlir::Operation* func);

    bool _enablePipelining = true;
};

mlir::LogicalResult FlashSDPATiling::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enablePipelining.hasValue()) {
        _log.trace("Overloading FlashSDPATiling enablePipelining argument by MLIR variable");
        _enablePipelining = enablePipelining.getValue();
    }
    return mlir::success();
}

void FlashSDPATiling::applyQTilingPatterns(mlir::Operation* func) {
    auto* ctx = func->getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<FlashSDPAQTilingRewrite>(ctx, _log);
    mlir::walkAndApplyPatterns(func, std::move(patterns));
}

void FlashSDPATiling::applyKVUnrollPatterns(mlir::Operation* func) {
    auto* ctx = func->getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<FlashSDPAKVUnrollRewrite>(ctx, _log);
    mlir::walkAndApplyPatterns(func, std::move(patterns));
}

void FlashSDPATiling::safeRunOnFunc() {
    auto func = getOperation();

    // 1) Annotate every FlashSDPA with head_and_q_num_tiles + kv_num_tiles. The Q rewriter
    // consumes head_and_q_num_tiles and propagates kv_num_tiles to the cloned Q tiles.
    auto annotateStatus = mlir::success();
    func->walk([&](VPU::FlashSDPAOp op) {
        if (mlir::failed(annotateTilingStrategy(op, _enablePipelining, _log))) {
            annotateStatus = mlir::failure();
        }
    });
    if (mlir::failed(annotateStatus)) {
        signalPassFailure();
        return;
    }

    // 2) Mark each operation with a tiling loop index for loop-allocation scheduling.
    func->walk([tilingIndex = 0ll](VPU::FlashSDPAOp flashSdpa) mutable {
        flashSdpa->setAttr(TILING_LOOP_INDEX_ATTR_NAME, TilingLoopIndexAttr::get(flashSdpa->getContext(), tilingIndex));
        ++tilingIndex;
    });

    // 3) Tile Heads/Query first, then unroll each resulting Q tile on Key/Value.
    applyQTilingPatterns(func);
    applyKVUnrollPatterns(func);
}

}  // namespace

//
// createFlashSDPATilingPass
//

std::unique_ptr<mlir::Pass> VPU::createFlashSDPATilingPass(bool enablePipelining, Logger log) {
    return std::make_unique<FlashSDPATiling>(enablePipelining, log);
}
