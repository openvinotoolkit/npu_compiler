//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <limits>
#include <map>
#include <optional>
#include <vpux/utils/core/error.hpp>
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
// the annotate walk to the three rewriters. Each rewriter consumes its own
// attribute and propagates the remaining ones to newly created ops.
constexpr llvm::StringLiteral HEAD_TILE_SIZE_ATTR_NAME = "head_tile_size";
constexpr llvm::StringLiteral Q_NUM_TILES_ATTR_NAME = "q_num_tiles";
constexpr llvm::StringLiteral KV_NUM_TILES_ATTR_NAME = "kv_num_tiles";

// Explicit tiling strategy with semantic dimension names
struct FlashSDPATilingStrategy {
    int64_t headTileSize{1};   // Number of Q heads (C) per tile
    int64_t querySeqTiles{1};  // Number of tiles on Query sequence length (H)
    int64_t kvNumBlocks{1};    // Number of KV sequence unrolls (consumed by FlashSDPAKVUnrollRewrite)
};

class FlashSDPAHeadTilingRewrite final : public mlir::OpRewritePattern<VPU::FlashSDPAOp> {
public:
    FlashSDPAHeadTilingRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::FlashSDPAOp>(ctx), _log(log) {
        setDebugName("FlashSDPAHeadTilingRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::FlashSDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class FlashSDPAQuerySeqTilingRewrite final : public mlir::OpRewritePattern<VPU::FlashSDPAOp> {
public:
    FlashSDPAQuerySeqTilingRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::FlashSDPAOp>(ctx), _log(log) {
        setDebugName("FlashSDPAQuerySeqTilingRewrite");
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

// Buffer count of a FlashSDPA with an attention mask: 11 operands + 3 results.
constexpr int64_t NUM_BUFFERS_WITH_ATTENTION_MASK = 14;
// Operand index of the optional attention mask, the last operand when present.
constexpr int64_t ATTENTION_MASK_BUFFER_INDEX = 10;

// Operand indices that are pipelined (double-buffered) across Q-tile operations.
// These buffers change between consecutive Q tiles within a KV block and need
// an extra CMX copy for prefetching the next Q tile while the current executes.
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
        // Without a mask the three results shift down to indices 10-12, so dropping the
        // last entry leaves exactly {query, input running state, result running state}.
        indices.pop_back();
    }

    return indices;
}

// Memory that has to stay free so the next Q tile's buffers can be prefetched while the
// current tile executes. The attention mask is sliced on the KV dimension as well, so only
// a single KV block of it is reserved - matching the footprint computed by
// FlashSDPAOp::fitIntoCMXAfterKeyValueTiling for the same kvBlocks.
Byte computeQPipeliningReservation(VPU::FlashSDPAOp origOp, ArrayRef<vpux::NDTypeInterface> tiledTensorTypes,
                                   int64_t kvBlocks) {
    const auto hasAttentionMask = (static_cast<int64_t>(tiledTensorTypes.size()) == NUM_BUFFERS_WITH_ATTENTION_MASK);
    const auto sourceSeqLen = getShape(origOp.getKey())[Dims4D::Act::H];
    const auto kvBlockSize = origOp.getKeyValueBlockSize(kvBlocks);

    auto reservedMemory = Byte{0};
    for (auto index : getPipelinedBufferIndices(hasAttentionMask)) {
        auto bufferType = tiledTensorTypes[index];

        const auto isMask = hasAttentionMask && (index == ATTENTION_MASK_BUFFER_INDEX);
        if (isMask && bufferType.getShape()[Dims4D::Act::W] == sourceSeqLen) {
            auto maskShape = Shape(bufferType.getShape());
            maskShape[Dims4D::Act::W] = kvBlockSize;
            bufferType = bufferType.changeShape(maskShape);
        }

        reservedMemory += bufferType.getTotalAllocSize();
    }

    return reservedMemory;
}

// Head (C) tile sizes to try, largest first.
//
// The kernel requires exactly 1 KV head per cluster invocation, so the natural tile size follows from
// how many KV heads a tile has to supply. getAlignment() returns alignment[C] = numClusters for SOK
// and 1 for SOH/Clustering:
//   * SOK with several KV heads - Key/Value are SEGMENTED on C, so the tile carries one KV head per
//     cluster it lands on, 'min(numClusters, kvHeads)' of them. When there are fewer KV heads than
//     clusters, getOptimalNumClusters() drops the spare ones; it does the same for the tail tile, so
//     qHeads needs no relation to the cluster count.
//   * SOK with a single KV head - Key/Value are DUPLICATED and only Q is SEGMENTED, so one head
//     already covers every cluster.
//   * SOH/Clustering - C is not distributed, one invocation per tile, one KV head.
// Scaled by 'groupSize' that is always a whole number of GQA groups and never exceeds qHeads, so
// every tile boundary lands on a group boundary and no candidate describes a tile larger than the
// tensor. That is the only candidate needed as long as the operation fits CMX.
//
// When it does not, a tile holding a single KV head can be split further: the kernel never advances
// its Key/Value pointers inside the head loop, so a Q head tile that is a strict subset of one GQA
// group still reads the same single KV head and is equally correct. Each extra head tile re-reads the
// whole Key/Value sequence, so the sub-group candidates are ordered largest first and only reached
// when CMX forces it.
SmallVector<int64_t> computeHeadTileCandidates(VPU::FlashSDPAOp origOp) {
    const auto keyShape = getShape(origOp.getKey());
    const auto kvHeads = keyShape[Dims4D::Act::C];

    const auto resultShape = getShape(origOp.getResultRunningOutput());
    const auto qHeads = resultShape[Dims4D::Act::C];
    const auto targetSeqLen = resultShape[Dims4D::Act::H];

    const auto alignment = getAlignment(origOp.getOperation(), {}, {});
    const auto alignmentC = alignment[Dims4D::Act::C.ind()];

    VPUX_THROW_UNLESS(qHeads % kvHeads == 0,
                      "Incorrect '{0}' configurations. Query heads dimension = '{1}' must be divisible by Key/Value "
                      "heads dimension '{2}'",
                      origOp->getName(), qHeads, kvHeads);

    const auto groupSize = qHeads / kvHeads;

    // KV heads one tile has to carry. 'alignmentC' is the cluster count for SOK and 1 otherwise, and
    // a tile can never hold more KV heads than the operation has, so the product with 'groupSize'
    // stays within qHeads by construction.
    const auto kvHeadsPerTile = std::min(alignmentC, kvHeads);
    const auto baseTileSize = kvHeadsPerTile * groupSize;

    auto candidates = SmallVector<int64_t>{baseTileSize};

    // Splitting a GQA group across clusters is only safe when every cluster of a tile still maps to
    // the KV head it reads. With alignment[C] == 1 (SOH/Clustering) the C dimension is not
    // distributed at all. With SOK and kvHeads == 1 the Key/Value tensors are DUPLICATED and Q heads
    // are SEGMENTED instead, so all clusters correctly share the one KV head. With SOK and multiple
    // KV heads, Key/Value are SEGMENTED on C: two clusters landing inside the same group would read
    // different KV heads, so one whole group per cluster is already the floor.
    const auto canSplitGroupAcrossClusters = (alignmentC == 1) || (kvHeads == 1);

    // Folded GQA (targetSeqLen == 1): the kernel folds every Q head of the tile into the row
    // dimension and runs one batched pair of DPU matmuls against the shared K/V head. Sub-group tiles
    // would still be correct, but they do not pay: backInferTileInfo pins the auxiliary buffer to the
    // full 'groupSize' rows for any tile with C > 1, so the dominant buffer does not shrink, while
    // every extra tile re-reads the whole Key/Value sequence. The remaining Q-side buffers are
    // single-row here and already negligible.
    const auto isFoldedGQA = (targetSeqLen == 1) && (kvHeads < qHeads);

    if (canSplitGroupAcrossClusters && !isFoldedGQA) {
        // Only divisors of the base tile: tiles are strided from offset 0, so a size that divides it
        // guarantees no tile ever straddles a group boundary. The branch is reachable only when the
        // base tile holds a single KV head, so every sub-tile keeps reading that same head.
        for (auto tileSize = baseTileSize / 2; tileSize >= 1; --tileSize) {
            if (baseTileSize % tileSize == 0) {
                candidates.push_back(tileSize);
            }
        }
    }

    return candidates;
}

// Estimate the head tile size, query sequence tiling and KV blocks the operation needs.
//
// CMX layout for 1 operation with Q-tile pipelining:
// [ ============ Total CMX Size ============ ]
// [ KV_tile | Q_buffers_curr | Q_buffers_next ]
// [     Required memory      | Reserved memory ]
//
// The reservation doubles the cost of every per-Q-tile buffer, which for large head counts or
// embedding sizes can exceed CMX no matter how far the query and Key/Value sequences are split: the
// query tile is clamped to alignment[H], and Key/Value blocking does not shrink the Q side at all.
// The head tile is the only lever that does, so the search walks the head tile candidates from
// largest to smallest and keeps the reservation mandatory - Q tiles executing back to back cost far
// more than the extra Key/Value re-reads an additional head tile brings.

std::optional<FlashSDPATilingStrategy> estimateTiling(VPU::FlashSDPAOp origOp, Logger log) {
    const auto resultShape = getShape(origOp.getResultRunningOutput());
    const auto alignment = getAlignment(origOp.getOperation(), {}, {});
    const auto qHeads = resultShape[Dims4D::Act::C];

    // Unroll on the Heads dimension. The size is picked by the search below; every candidate leaves
    // exactly 1 KV head per cluster invocation as the kernel requires.
    auto tiledResultShape = Shape(resultShape);

    // Query sequence (target SL) and KV sequence (source SL) dimensions
    const auto seqLenDimSize = resultShape[Dims4D::Act::H];
    const auto seqAlignment = alignment[Dims4D::Act::H.ind()];

    const auto keyShape = getShape(origOp.getKey());
    const auto sourceSeqLen = keyShape[Dims4D::Act::H];
    const auto keyType = mlir::cast<NDTypeInterface>(origOp.getKey().getType());
    const auto elemType = keyType.getElementType();
    const auto kvAlignment = vpux::VPU::NCEInvariant::getAlignment(elemType);

    auto swOp = mlir::cast<VPU::SWOpInterface>(origOp.getOperation());

    // Generate the progression of kvNumBlocks values to iterate (outer loop).
    // Starts with 1 (full KV in one block), then 2, 3, ... following alignment.
    auto kvNumBlocksValues = SmallVector<int64_t>{1};
    {
        auto dimSize = sourceSeqLen;
        while (dimSize > kvAlignment) {
            auto kv = divUp(sourceSeqLen, dimSize - kvAlignment);
            dimSize = alignValUp(divUp(sourceSeqLen, kv), kvAlignment);
            kvNumBlocksValues.push_back(kv);
        }
    }

    // For a given head tile size and kvBlocks, find the smallest querySeqTiles such that the tiled
    // configuration fits in CMX including the Q-pipelining reservation.
    // Returns nullopt if no valid querySeqTiles exists for this combination.
    auto findQuerySeqTiles = [&](int64_t headTileSize, int64_t kvBlocks) -> std::optional<int64_t> {
        auto querySeqTiles = int64_t{1};
        auto curSeqSize = seqLenDimSize;
        tiledResultShape[Dims4D::Act::C] = headTileSize;
        tiledResultShape[Dims4D::Act::H] = curSeqSize;

        while (true) {
            auto tiledTensorTypes = getAllOperandsSwInterface(swOp, TileInfo{tiledResultShape}, log);

            const auto reservedMemory = computeQPipeliningReservation(origOp, tiledTensorTypes, kvBlocks);

            const bool fits = origOp.fitIntoCMXAfterKeyValueTiling(tiledTensorTypes, reservedMemory, kvBlocks);

            if (fits) {
                return querySeqTiles;
            }

            // Increase querySeqTiles: smaller query tiles free more CMX for KV.
            if (curSeqSize <= seqAlignment) {
                return std::nullopt;
            }

            querySeqTiles = divUp(seqLenDimSize, curSeqSize - seqAlignment);
            curSeqSize = alignValUp(divUp(seqLenDimSize, querySeqTiles), seqAlignment);
            tiledResultShape[Dims4D::Act::H] = curSeqSize;
        }
    };

    auto searchStrategy = [&](int64_t headTileSize) -> std::optional<FlashSDPATilingStrategy> {
        const auto headTiles = divUp(qHeads, headTileSize);

        // kvBlocks=1 means the full KV sequence fits in a single block — no KV unrolling.
        // Check it first and return immediately; it is preferred over any KV-split strategy.
        if (auto querySeqTiles = findQuerySeqTiles(headTileSize, /*kvBlocks=*/1)) {
            return FlashSDPATilingStrategy{headTileSize, *querySeqTiles, /*kvNumBlocks=*/1};
        }

        // The full KV sequence doesn't fit. Search over KV splits and keep the candidate
        // with the minimum total op count.
        auto bestQuerySeqTiles = int64_t{0};
        auto bestKvNumBlocks = int64_t{0};
        auto bestTotalOps = std::numeric_limits<int64_t>::max();

        for (auto kvBlocks : llvm::ArrayRef(kvNumBlocksValues).drop_front()) {
            if (auto querySeqTiles = findQuerySeqTiles(headTileSize, kvBlocks)) {
                const auto totalOps = headTiles * *querySeqTiles * kvBlocks;
                if (totalOps < bestTotalOps) {
                    bestTotalOps = totalOps;
                    bestQuerySeqTiles = *querySeqTiles;
                    bestKvNumBlocks = kvBlocks;
                }
            }
        }

        if (bestTotalOps < std::numeric_limits<int64_t>::max()) {
            return FlashSDPATilingStrategy{headTileSize, bestQuerySeqTiles, bestKvNumBlocks};
        }

        return std::nullopt;
    };

    // One whole GQA group per tile comes first; smaller head tiles are only reached when nothing
    // fits with the Q-pipelining reservation at the larger ones.
    const auto headTileCandidates = computeHeadTileCandidates(origOp);
    for (auto [index, headTileSize] : headTileCandidates | indexed) {
        if (auto strategy = searchStrategy(headTileSize)) {
            return strategy;
        }
        const auto isLastCandidate = (index + 1 == headTileCandidates.size());
        log.trace("No tiling fits CMX with {0} Q heads per tile{1}", headTileSize,
                  isLastCandidate ? "" : ", trying a smaller head tile");
    }

    return std::nullopt;
}

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

// Estimate the tiling strategy once and stamp it onto the op as three transient
// attributes (head_tile_size, kv_num_tiles, q_num_tiles). The head rewriter tiles
// heads first, then the KV rewriter unrolls, then the query-seq rewriter tiles last.
mlir::LogicalResult annotateTilingStrategy(VPU::FlashSDPAOp op, Logger log) {
    auto resultShape = getShape(op.getResult(0));
    if (resultShape.size() < 2) {
        return errorAt(op, "Output shape must at least have a rank 2, got {0}", resultShape.size());
    }

    auto strategy = estimateTiling(op, log.nest());
    if (!strategy.has_value()) {
        const auto alignment = getAlignment(op.getOperation(), {}, {});
        return errorAt(
                op,
                "Failed to estimate tiling for FlashSDPA operation. Tensors will not fit {0} bytes of CMX with "
                "memory reserved for Q-tile pipelining, even with the smallest supported tile: {1} heads and {2} "
                "TargetSequenceLength rows against a single aligned Key/Value block. Query {3}, Key {4}, "
                "AttentionMask {5}.",
                VPU::getTotalCMXSize(op.getOperation()).count(), computeHeadTileCandidates(op).back(),
                alignment[Dims4D::Act::H.ind()], getShape(op.getQuery()), getShape(op.getKey()),
                op.getAttentionMask() != nullptr ? getShape(op.getAttentionMask()) : ShapeRef());
    }

    log.trace("Annotated tiling for '{0}': headsPerTile={1}, querySeq={2}, kv={3}", op->getLoc(),
              strategy->headTileSize, strategy->querySeqTiles, strategy->kvNumBlocks);

    auto* ctx = op->getContext();
    op->setAttr(HEAD_TILE_SIZE_ATTR_NAME, getIntAttr(ctx, strategy->headTileSize));
    op->setAttr(Q_NUM_TILES_ATTR_NAME, getIntAttr(ctx, strategy->querySeqTiles));
    op->setAttr(KV_NUM_TILES_ATTR_NAME, getIntAttr(ctx, strategy->kvNumBlocks));
    return mlir::success();
}

mlir::LogicalResult FlashSDPAHeadTilingRewrite::matchAndRewrite(VPU::FlashSDPAOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    auto headTileSizeAttr = origOp->getAttrOfType<mlir::IntegerAttr>(HEAD_TILE_SIZE_ATTR_NAME);
    if (headTileSizeAttr == nullptr) {
        return mlir::failure();
    }

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto log = _log.nest();

    const auto headTileSize = parseIntAttr<int64_t>(headTileSizeAttr);
    if (headTileSize <= 0) {
        return errorAt(origOp, "Head tiling size attribute must be a positive number, got {0}", headTileSize);
    }

    // Clear the marker before cloning so head-tile clones don't re-trigger this rewriter.
    // kv_num_tiles and q_num_tiles remain on origOp and will be inherited by clones.
    rewriter.modifyOpInPlace(origOp, [&] {
        origOp->removeAttr(HEAD_TILE_SIZE_ATTR_NAME);
    });

    const auto firstOutputShape = getShape(origOp.getResultRunningOutput());
    const auto qHeads = firstOutputShape[Dims4D::Act::C];
    const auto headTiles = divUp(qHeads, headTileSize);

    if (headTiles <= 1) {
        log.trace("No head tiling needed");
        return mlir::success();
    }

    // Build output tiles by striding C in steps of the estimated head tile size.
    // fillDividedTiles with alignment[C]=numClusters rounds each tile boundary up to the
    // nearest multiple of numClusters (e.g. alignUp(32/2, 3) = 18 for 32 Q-heads, 2 tiles,
    // 3 clusters), which produces tile C sizes that violate FlashSDPAOpInputTiling's GQA
    // precondition. Striding directly by the estimated size keeps every tile either a whole
    // number of GQA groups or a group-aligned subset of a single group (see
    // computeHeadTileCandidates); the last tile receives the remainder.
    OutputTiling firstOutputTilesList;
    for (int64_t offset = 0; offset < qHeads; offset += headTileSize) {
        TileInfo tile(firstOutputShape);
        tile.shape[Dims4D::Act::C] = std::min(headTileSize, qHeads - offset);
        tile.offsets[Dims4D::Act::C] = offset;
        tile.axis[Dims4D::Act::C] = headTiles;
        firstOutputTilesList.push_back(tile);
    }
    VPUX_THROW_UNLESS(static_cast<int64_t>(firstOutputTilesList.size()) == headTiles,
                      "Head tile count mismatch: expected {0}, got {1}", headTiles, firstOutputTilesList.size());

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    origOp->getName());

    auto result = applyTileStrategyFlashSDPA(tilingBuilder, firstOutputTilesList, rewriter, log);
    if (mlir::failed(result)) {
        return errorAt(origOp, "Failed to rewrite original operation with the head-tiled one");
    }

    return mlir::success();
}

mlir::LogicalResult FlashSDPAQuerySeqTilingRewrite::matchAndRewrite(VPU::FlashSDPAOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    auto qNumTilesAttr = origOp->getAttrOfType<mlir::IntegerAttr>(Q_NUM_TILES_ATTR_NAME);
    if (qNumTilesAttr == nullptr) {
        return mlir::failure();
    }

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto log = _log.nest();

    const auto querySeqTiles = parseIntAttr<int64_t>(qNumTilesAttr);

    // Clear the marker before cloning so the Q-tile clones don't re-trigger this rewriter.
    rewriter.modifyOpInPlace(origOp, [&] {
        origOp->removeAttr(Q_NUM_TILES_ATTR_NAME);
    });

    if (querySeqTiles <= 1) {
        log.trace("No query sequence tiling needed");
        return mlir::success();
    }

    // Build the tiling divisor shape: [N=Batch, C=1, H=querySeqTiles, W=1]
    const auto firstOutputShape = getShape(origOp.getResultRunningOutput());
    auto tilingStrategy = Shape(firstOutputShape.size(), 1);
    tilingStrategy[Dims4D::Act::N] = firstOutputShape[Dims4D::Act::N];
    tilingStrategy[Dims4D::Act::H] = querySeqTiles;

    const auto alignment = getAlignment(origOp.getOperation(), {}, {});
    const auto unrollSpatialFirst = false;
    const auto firstOutputTiles = fillDividedTiles(tilingStrategy, firstOutputShape, alignment, unrollSpatialFirst);

    if (mlir::failed(firstOutputTiles)) {
        return errorAt(origOp,
                       "Failed to compute query-seq tiling for output shape: '{0}', tiling strategy: '{1}', "
                       "alignment: '{2}'",
                       firstOutputShape, tilingStrategy, alignment);
    }

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    origOp->getName());

    auto result = applyTileStrategyFlashSDPA(tilingBuilder, firstOutputTiles.value(), rewriter, log);
    if (mlir::failed(result)) {
        return errorAt(origOp,
                       "Failed to rewrite FlashSDPA with query-sequence tiling (H-dimension split into {0} tiles)",
                       querySeqTiles);
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
    const auto tileSize = origOp.getKeyValueBlockSize(kvNumBlocks);

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
    for (auto i = int64_t{0}; i < kvNumBlocks; ++i) {
        log.trace("Unrolling {0} - {1} / {2} times", origOp->getName(), i + 1, kvNumBlocks);
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
        auto isTailAttr = mlir::BoolAttr::get(ctx, initIsTail && (i + 1 == kvNumBlocks));

        auto sourceSeqLenPadSize = (i + 1 == kvNumBlocks) ? origOp.getSourceSeqLenPadSizeAttr() : zeroPadAttr;

        auto tileLoc = appendLoc(origOp->getLoc(), "flash_sdpa_kv_tile_{0}", i);
        tiledOp = rewriter.create<VPU::FlashSDPAOp>(tileLoc, query, keySlice, valueSlice, out, max, sum,
                                                    attentionMaskSlice, sourceSeqLenPadSize, isHeadAttr, isTailAttr,
                                                    origOp.getMultiClusterStrategyAttr());

        // Propagate q_num_tiles so the Q rewriter can tile each KV-unrolled op
        if (auto qAttr = origOp->getAttr(Q_NUM_TILES_ATTR_NAME)) {
            tiledOp->setAttr(Q_NUM_TILES_ATTR_NAME, qAttr);
        }

        copyLoopAttributes(origOp, tiledOp.getOperation());
        log.trace("Unrolled {0} - {1}", tiledOp->getName(), tiledOp->getResult(0));

        // Propagate intermediate values to the next FlashSDPAOp
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
    explicit FlashSDPATiling(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    void applyHeadTilingPatterns(mlir::Operation* func);
    void applyKVUnrollPatterns(mlir::Operation* func);
    void applyQuerySeqTilingPatterns(mlir::Operation* func);
};

void FlashSDPATiling::applyHeadTilingPatterns(mlir::Operation* func) {
    auto* ctx = func->getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<FlashSDPAHeadTilingRewrite>(ctx, _log);
    mlir::walkAndApplyPatterns(func, std::move(patterns));
}

void FlashSDPATiling::applyKVUnrollPatterns(mlir::Operation* func) {
    auto* ctx = func->getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<FlashSDPAKVUnrollRewrite>(ctx, _log);
    mlir::walkAndApplyPatterns(func, std::move(patterns));
}

void FlashSDPATiling::applyQuerySeqTilingPatterns(mlir::Operation* func) {
    auto* ctx = func->getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<FlashSDPAQuerySeqTilingRewrite>(ctx, _log);
    mlir::walkAndApplyPatterns(func, std::move(patterns));
}

void FlashSDPATiling::safeRunOnFunc() {
    auto func = getOperation();

    // 1) Annotate every FlashSDPA with head_tile_size + kv_num_tiles + q_num_tiles.
    //    Each rewriter consumes its own attribute and propagates the rest.
    auto annotateStatus = mlir::success();
    func->walk([&](VPU::FlashSDPAOp op) {
        if (mlir::failed(annotateTilingStrategy(op, _log))) {
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

    // 3) Three-phase tiling producing {Heads, sSL, tSL} execution order:
    //    - Heads: outermost, independent tiles (Concat)
    //    - sSL (KV blocks): middle, sequential chain via running state
    //    - tSL (query seq): innermost, pipelined within each KV block
    applyHeadTilingPatterns(func);
    applyKVUnrollPatterns(func);

    // Re-assign loop indices after KV unrolling so that each KV-block op receives a
    // unique index. Q-tiles created in the next phase inherit this index via
    // copyLoopAttributes, which lets the scheduler identify which Q-tiles belong to the
    // same pipeline loop (those within one KV block) versus sequential KV-block chains.
    // Without this, all Q-tiles from all KV blocks share the pre-tiling index, so the
    // scheduler cannot isolate the correct pipeline group for double-buffering.
    func->walk([tilingIndex = 0ll](VPU::FlashSDPAOp flashSdpa) mutable {
        flashSdpa->setAttr(TILING_LOOP_INDEX_ATTR_NAME, TilingLoopIndexAttr::get(flashSdpa->getContext(), tilingIndex));
        ++tilingIndex;
    });

    applyQuerySeqTilingPatterns(func);
}

}  // namespace

//
// createFlashSDPATilingPass
//

std::unique_ptr<mlir::Pass> VPU::createFlashSDPATilingPass(Logger log) {
    return std::make_unique<FlashSDPATiling>(log);
}
