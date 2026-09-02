//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/gather_dma_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

#include <limits>
#include <numeric>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTOPTODMAFORPERFORMANTEXECUTION
#define GEN_PASS_DEF_CONVERTOPTODMAFORPERFORMANTEXECUTION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// ExpandIndices
//
// Keeps a big-element Gather as a single GatherDMA instead of letting TileGatherElement split it into
// many small GatherDMA ops joined by a Concat. Each original index `i` selecting a row of
// K = rowElems / chunkElems chunks is expanded into K consecutive sub-indices (i*K + [0 .. K-1])
// gathering from the row-major reshaped input [N*K, chunkElems]. The selected chunks stay contiguous,
// so the result matches the original Gather. The resulting Gather is then lowered by MoveToDMAGather.
//
// dimsBeforeAxis == 1: K = rowElems / chunkElems consecutive sub-indices per row.
// (example: N rows, K=2 chunks per row, picking rows [2, 0]):
//
//        input [N, R]            indices [2,0]
//             |                       |
//             | Reshape               | newIdx = idx*K + [0..K-1]
//             v                       v
//   reshaped [N*K, chunkElems]   flatIndices [4,5, 0,1]
//             |                       |
//             +-----------+-----------+
//                         |
//                         v
//                  VPU::GatherOp  (chunk-granular, GatherDMA-legal)
//                         |
//                         | Reshape
//                         v
//                  output [2, R]   == original Gather result
//
// dimsBeforeAxis > 1 and batchDimsProduct == 1 (all batch dims trivially 1):
// The K pre-axis dims fold into the row axis. Each index p maps to K non-contiguous rows
// with stride Na (= numRows): newIdx = p + [0, Na, ..., (K-1)*Na].
// (example: input [1, K=2, Na=35, R], batch_dims=1, axis=2, picking axis-index p):
//
//   input [1, K, Na, R]      indices (value: p)
//        |                         |
//        | Reshape                 | newIdx = p + [0, Na, ..., (K-1)*Na]
//        v                         v
//   reshaped [Na*K, R]       flatIndices [p, p+Na]
//        |                         |
//        +---------+---------------+
//                  |
//                  v
//           VPU::GatherOp  (axis=0, batch_dims=0, GatherDMA-legal)
//                  |
//                  | Reshape
//                  v
//           output [1, K, 1, R]   == original Gather result
//

class ExpandIndices final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    ExpandIndices(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("ExpandIndices");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandIndices::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp.getLoc());

    // The index arithmetic below replaces a constant index list with runtime eltwise ops, defeating the
    // negative-index normalization in MoveToDMAGather. Bail so element-tiling handles negative indices.
    if (auto indicesCst = origOp.getIndices().getDefiningOp<Const::DeclareOp>()) {
        const auto indicesContent = indicesCst.getContent();
        if (llvm::any_of(indicesContent.getValues<int64_t>(), [](int64_t val) {
                return val < 0;
            })) {
            return mlir::failure();
        }
    }

    auto ctx = rewriter.getContext();
    const auto inType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto outType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    const auto indicesType = mlir::cast<NDTypeInterface>(origOp.getIndices().getType());
    const auto inShape = inType.getShape();
    const auto axis = origOp.getAxisValue();
    const auto arch = config::getArch(origOp);
    const auto outShape = outType.getShape();
    const auto indicesShape = indicesType.getShape();
    if (inShape.isDynamic() || outShape.isDynamic() || indicesShape.isDynamic()) {
        return mlir::failure();
    }

    int64_t dimsBeforeAxis = 1;
    for (size_t idx = 0; idx < (static_cast<size_t>(axis)); ++idx) {
        dimsBeforeAxis *= inShape[Dim(idx)];
    }

    // Before-axis dims are all 1 for a GatherDMA-legal op, so output = numIndices rows of rowElems.
    const int64_t numIndices = indicesShape.totalSize();
    const int64_t rowElems = outShape.totalSize() / (numIndices * dimsBeforeAxis);
    const int64_t numRows = inShape[Dim(axis)];

    // Largest chunk (in elements) fitting the per-index size limit; it must evenly divide the row.
    const int64_t elemBits = vpux::getElemTypeSize(outType).count();
    int64_t chunkElems = checked_cast<int64_t>(VPU::getGatherDMAMaxElementSize(arch)) * CHAR_BIT / elemBits;
    int64_t chunksPerRow;
    if (dimsBeforeAxis == 1) {
        if (!VPU::isLegalConvertToGatherDMA(origOp, /*isElementTile=*/true, /*isIndicesTile=*/false, _log)) {
            return mlir::failure();
        }
        if (origOp.getBatchDims() != 0) {
            return mlir::failure();
        }
        if (chunkElems == 0 || rowElems % chunkElems != 0) {
            return mlir::failure();
        }
        chunksPerRow = rowElems / chunkElems;
    } else {
        if (!VPU::isLegalConvertToGatherDMA(origOp, /*isElementTile=*/false, /*isIndicesTile=*/false, _log, false)) {
            return mlir::failure();
        }
        for (int64_t i = 0; i < origOp.getBatchDims(); ++i) {
            if (inShape[Dim(i)] != 1) {
                return mlir::failure();
            }
        }

        chunksPerRow = dimsBeforeAxis;
        chunkElems = rowElems;
    }

    // The expanded index list must still fit the DMA list-length limit.
    if (checked_cast<size_t>(numIndices * chunksPerRow) > VPU::getGatherDMAMaxIndicesListLength(arch)) {
        return mlir::failure();
    }

    // Index arithmetic runs in SI32. The largest sub-index is numRows * chunksPerRow - 1
    if (numRows * chunksPerRow > std::numeric_limits<int32_t>::max()) {
        return mlir::failure();
    }
    const auto si32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);

    const auto broadcast = vpux::IE::AutoBroadcastTypeAttr::get(ctx, vpux::IE::AutoBroadcastType::NUMPY);

    auto reshape = [&](mlir::Value operand, ArrayRef<int64_t> shape, StringRef name) -> mlir::Value {
        return rewriter.createOrFold<VPU::ReshapeOp>(takeOpLoc(origOp, name), operand, getIntArrayAttr(ctx, shape));
    };
    auto idxConst = [&](ArrayRef<int64_t> shape, ArrayRef<int32_t> values) -> mlir::Value {
        return Const::createConst(rewriter, takeOpLoc(origOp, "idx_cst"), mlir::RankedTensorType::get(shape, si32Type),
                                  values);
    };

    // Reshape input rows into chunk-sized rows: [.., N, R] -> [N * chunksPerRow, chunkElems].
    auto reshapedInput = reshape(origOp.getInput(), {numRows * chunksPerRow, chunkElems}, "reshape_in");

    // Expand each index: newIdx[m, k] = idx[m] * chunksPerRow + k. VPU eltwise kernels need 4D SI32
    // operands, so reshape to [1, 1, numIndices, 1] and normalize the index type to SI32.
    //
    // dimsBeforeAxis == 1: each original index p maps to p*K, p*K+1, ..., p*K+(K-1).
    //   Broadcast layout: idx[1,1,numIndices,1] + iota[1,1,1,K] -> [1,1,numIndices,K]
    //   Flattened (row-major): position i*K+k -> interleaved order, matches reshape to [numIndices,K,chunkElems].
    //
    // dimsBeforeAxis > 1: each original index p maps to p, p+Na, ..., p+(K-1)*Na.
    //   Grouped order needed: all numIndices values for k=0, then all for k=1, ...
    //   Broadcast layout: iota[1,1,K,1] + idx[1,1,1,numIndices] -> [1,1,K,numIndices]
    //   Flattened (row-major): position k*numIndices+i -> grouped order, matches reshape to [K,numIndices,R].
    const auto idxReshape = (dimsBeforeAxis == 1) ? SmallVector<int64_t>{1, 1, numIndices, 1}
                                                  : SmallVector<int64_t>{1, 1, 1, numIndices};
    mlir::Value idx4D = reshape(origOp.getIndices(), idxReshape, "reshape_idx");
    if (indicesType.getElementType() != si32Type) {
        idx4D = rewriter.createOrFold<VPU::ConvertOp>(takeOpLoc(origOp, "idx_to_si32"), idx4D,
                                                      mlir::TypeAttr::get(si32Type));
    }

    const int32_t idxScale = (dimsBeforeAxis == 1) ? checked_cast<int32_t>(chunksPerRow) : int32_t(1);
    auto scaled = rewriter.create<VPU::MultiplyOp>(takeOpLoc(origOp, "idx_scale"), idx4D,
                                                   idxConst({1, 1, 1, 1}, {idxScale}), broadcast, /*post_op=*/nullptr)
                          .getOutput();
    SmallVector<int32_t> iotaValues(chunksPerRow);
    const auto iotaShape = dimsBeforeAxis == 1 ? SmallVector<int64_t>{1, 1, 1, chunksPerRow}
                                               : SmallVector<int64_t>{1, 1, chunksPerRow, 1};
    if (dimsBeforeAxis == 1) {
        std::iota(iotaValues.begin(), iotaValues.end(), int32_t(0));
    } else {
        for (int64_t n = 0; n < chunksPerRow; ++n) {
            iotaValues[n] = checked_cast<int32_t>(n * numRows);
        }
    }
    auto expanded = rewriter.create<VPU::AddOp>(takeOpLoc(origOp, "idx_offset"), scaled,
                                                idxConst(iotaShape, iotaValues), broadcast,
                                                /*post_op=*/nullptr)
                            .getOutput();
    auto flatIndices = reshape(expanded, {numIndices * chunksPerRow}, "reshape_idx_flat");

    auto newGather = rewriter.create<VPU::GatherOp>(takeOpLoc(origOp, "chunked"), reshapedInput, flatIndices,
                                                    getIntAttr(ctx, /*axis_value=*/int64_t(0)),
                                                    getIntAttr(ctx, /*batch_dims=*/int64_t(0)),
                                                    /*indices_rank=*/nullptr);

    rewriter.replaceOp(origOp, reshape(newGather.getOutput(), to_small_vector(outType.getShape()), "reshape_out"));

    _log.trace("[{0}] Split big-element Gather: chunksPerRow={1}, chunkElems={2}, numIndices={3}", getDebugName(),
               chunksPerRow, chunkElems, numIndices);
    return mlir::success();
}

//
// TileGatherElement
//

class TileGatherElement final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    TileGatherElement(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("TileGatherElement");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TileGatherElement::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!VPU::isLegalConvertToGatherDMA(origOp, /*isElementTile*/ true, /*isIndicesTile*/ false, _log)) {
        return mlir::failure();
    }

    const auto axis = static_cast<size_t>(origOp.getAxisValue());
    const auto inputShape = getShape(origOp.getInput());
    const auto outputShape = getShape(origOp.getOutput());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto arch = config::getArch(origOp);

    Shape nTilesOnDim(outputShape.size(), 1);
    DimArr tileDimOrder;
    // Tiling the dim after axis. Gather Output shape size is different from input size, but the dim after axis will
    // keep.
    auto shapeSizeDiff = outputShape.size() - inputShape.size();
    for (size_t idx = axis + 1; idx < inputShape.size(); ++idx) {
        tileDimOrder.push_back(vpux::Dim(idx + shapeSizeDiff));
    }

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(origOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        const size_t GATHER_DMA_MAX_ELEMENT_SIZE_ARCH_BASED = VPU::getGatherDMAMaxElementSize(arch);

        for (const auto& tile : tiles.value()) {
            size_t elementSizeInBit = vpux::getElemTypeSize(outputType).count();
            auto inputTiling = origOp.backInferTileInfo(tile, _log);
            auto& inTiles = inputTiling.tiles;
            for (size_t idx = axis + 1; idx < inputShape.size(); ++idx) {
                elementSizeInBit *= inTiles.begin()->shape.raw()[idx];
            }
            if (elementSizeInBit > GATHER_DMA_MAX_ELEMENT_SIZE_ARCH_BASED * CHAR_BIT) {
                return false;
            }
        }
        return true;
    };

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;
    while (tileDimIter < tileDimOrder.end() && !isSupportedTileSize(nTilesOnDim)) {
        if (nTilesOnDim[Dim(dimToTile)] >= outputShape[Dim(dimToTile)]) {
            dimToTile = *(++tileDimIter);
        } else {
            ++nTilesOnDim[Dim(dimToTile)];
        }
    }

    const auto tilesNew = fillDividedTiles(origOp, nTilesOnDim, outputShape);
    if (mlir::failed(tilesNew)) {
        return mlir::failure();
    }

    return VPU::applyTileStrategy(origOp, tilesNew.value(), rewriter, _log.nest());
}

//
// TileGatherIndices
//

class TileGatherIndices final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    TileGatherIndices(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("TileGatherIndices");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TileGatherIndices::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!VPU::isLegalConvertToGatherDMA(origOp, /*isElementTile*/ false, /*isIndicesTile*/ true, _log)) {
        return mlir::failure();
    }

    const auto outputShape = getShape(origOp.getOutput());
    const auto indicesType = mlir::cast<vpux::NDTypeInterface>(origOp.getIndices().getType());
    const auto indicesShape = indicesType.getShape();
    const auto indicesRank = origOp.getIndicesRank().value_or(indicesShape.size());
    const auto arch = config::getArch(origOp);

    Shape nTilesOnDim(outputShape.size(), 1);

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim) -> bool {
        const auto tiles = fillDividedTiles(origOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        const size_t DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED = VPU::getGatherDMAMaxIndicesListLength(arch);

        for (auto tile : tiles.value()) {
            const auto inputTiling = origOp.backInferTileInfo(tile, _log);
            const auto indicesTiling = inputTiling.tiles[1];
            const auto newIndicesType = indicesType.extractDenseTile(indicesTiling.offsets, indicesTiling.shape);
            const size_t numberOfIndices = newIndicesType.getNumElements();
            if (numberOfIndices <= DMA_MAX_INDICES_LIST_LENGTH_ARCH_BASED) {
                return true;
            }
        }
        return false;
    };

    const int64_t axisValue = origOp.getAxisValue();
    const int64_t batchDims = origOp.getBatchDims();

    const auto dimToTile = axisValue + indicesRank - batchDims - 1;
    while (!isSupportedTileSize(nTilesOnDim)) {
        if (nTilesOnDim[Dim(dimToTile)] >= outputShape[Dim(dimToTile)]) {
            return mlir::failure();
        }
        ++nTilesOnDim[Dim(dimToTile)];
    }

    const auto tilesNew = fillDividedTiles(origOp, nTilesOnDim, outputShape);
    if (mlir::failed(tilesNew)) {
        return mlir::failure();
    }

    return VPU::applyTileStrategy(origOp, tilesNew.value(), rewriter, _log.nest());
}

//
// MoveToDMAGather
//

class MoveToDMAGather final : public mlir::OpRewritePattern<VPU::GatherOp> {
public:
    MoveToDMAGather(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::GatherOp>(ctx), _log(log) {
        setDebugName("MoveToDMAGather");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// GatherDMA indices only support positive values
// - If the indices is constant, iterate through the values and convert any negatives to positives
// - If the indices is dynamic, TODO: E#149660
mlir::Value handleNegativeIndices(mlir::Value indices, ShapeRef dataShape, const Dim axis,
                                  mlir::PatternRewriter& rewriter) {
    if (auto indicesCst = mlir::dyn_cast_or_null<Const::DeclareOp>(indices.getDefiningOp())) {
        const auto indicesContent = indicesCst.getContent();
        auto indicesVals = to_small_vector(indicesContent.getValues<int64_t>());
        auto firstNegativeIt = std::find_if(indicesVals.begin(), indicesVals.end(), [](int64_t val) {
            return val < 0;
        });

        if (firstNegativeIt != indicesVals.end()) {
            for (auto it = firstNegativeIt; it != indicesVals.end(); ++it) {
                if (*it < 0) {
                    *it += dataShape[axis];
                }
            }

            auto indicesType = mlir::cast<NDTypeInterface>(indicesCst.getOutput().getType());
            auto indicesStorageType = mlir::cast<mlir::RankedTensorType>(
                    indicesType.changeElemType(mlir::IntegerType::get(indicesCst.getContext(), 64)));
            auto indicesStorageAttr = Const::createConstContent(indicesStorageType, ArrayRef(indicesVals));

            return rewriter
                    .create<Const::DeclareOp>(indicesCst.getLoc(), indicesStorageType,
                                              Const::ContentAttr::get(indicesStorageAttr))
                    .getOutput();
        }
    }
    return indices;
}

// Verify if adaption passes tiled Gather op on the dim after gather axis
bool hasTilingDoneToBenefitAbsAddressing(VPU::GatherOp origOp) {
    const auto gatherAxis = origOp.getAxisValue();
    if (auto concatOp = mlir::dyn_cast_or_null<VPU::ConcatOp>(*origOp->getUsers().begin())) {
        auto concatAxes = VPU::getConcatAxes(concatOp);

        // Check tiling over 1 axis
        if (concatAxes.size() != 1) {
            return false;
        }
        if (*concatAxes.begin() != gatherAxis + 1) {
            return false;
        }
    }
    return true;
}

// Verify if multiclustering is needed and can be done on dim after gather axis
bool canHaveMulticlusteringToBenefitAbsAddressing(VPU::GatherDMAOp origOp) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    const auto gatherAxis = origOp.getAxisValue();

    size_t numTile = config::getNumOfTiles(origOp);
    if (gatherAxis == Dims4D::Act::C.ind() &&
        clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight, numTile)) {
        return true;
    }
    if (gatherAxis == Dims4D::Act::H.ind() &&
        clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverWidth, numTile)) {
        return true;
    }

    return false;
}

mlir::LogicalResult MoveToDMAGather::matchAndRewrite(VPU::GatherOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp.getLoc());

    if (!VPU::isLegalConvertToGatherDMA(origOp, /*isElementTile*/ false, /*isIndicesTile*/ false, _log)) {
        return mlir::failure();
    }

    auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    auto axis = Dim(origOp.getAxisValue());

    auto indices = handleNegativeIndices(origOp.getIndices(), inputType.getShape(), axis, rewriter);

    auto reshapeOperand = [&](mlir::Value operand, ShapeRef newShape, const mlir::Location& location) {
        auto newShapeAttr = getIntArrayAttr(operand.getContext(), newShape);
        return rewriter.createOrFold<VPU::ReshapeOp>(location, operand, newShapeAttr);
    };

    // Ensure Indices tensor has the same rank as the Input tensor for GatherDMA
    //  - Fuse Indices into one dimension and align it with the axis dimension of Input
    //  - Fill other dimensions with 1
    // Example:                          Reshape To:
    //   Input:   [1, 16, 32, 32]          Input:   [1, 16, 32, 32]
    //   Indices: [2, 5]                   Indices: [1, 10, 1, 1]
    //   Axis:    1                        Axis:    1
    //   Output:  [1, 2, 5, 32, 32]        Output:  [1, 10, 32, 32]

    auto indicesType = mlir::cast<NDTypeInterface>(indices.getType());
    auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    Shape newIndicesShape(inputType.getRank(), 1);
    newIndicesShape[axis] = indicesType.getShape().totalSize();
    auto reshapeIndicesOp = reshapeOperand(indices, newIndicesShape, takeOpLoc(origOp, "reshape_indices"));

    // HW requirement: each list entry must be 64 bits
    auto requiredType64 = mlir::IntegerType::get(origOp.getContext(), 64);

    // Convert non-4D shape to 4D for better hardware utilization
    // Only do this if the operation will benefit from multi-shave execution
    const auto reshapeIndicesType = mlir::cast<NDTypeInterface>(reshapeIndicesOp.getType());
    const auto reshapeIndicesShape = reshapeIndicesType.getShape();

    const bool shouldConvertTo4D =
            (reshapeIndicesShape.size() != 4) && VPU::shouldConvertUseMultiShaves(reshapeIndicesType);

    mlir::Value convertIndicesOp = [&]() -> mlir::Value {
        if (shouldConvertTo4D) {
            // Reshape to 4D: [1, totalSize, 1, 1] for non-4D inputs
            const auto totalSize = reshapeIndicesShape.totalSize();
            const Shape shape4D = {1, totalSize, 1, 1};
            const auto reshapeTo4DOp = reshapeOperand(reshapeIndicesOp, shape4D, takeOpLoc(origOp, "reshape_to_4d"));

            const auto convertOp = rewriter.createOrFold<VPU::ConvertOp>(origOp->getLoc(), reshapeTo4DOp,
                                                                         mlir::TypeAttr::get(requiredType64));

            // Reshape back to original shape
            return reshapeOperand(convertOp, reshapeIndicesShape, takeOpLoc(origOp, "reshape_from_4d"));
        } else {
            return rewriter.createOrFold<VPU::ConvertOp>(origOp->getLoc(), reshapeIndicesOp,
                                                         mlir::TypeAttr::get(requiredType64));
        }
    }();

    auto gatherDMAOp = rewriter.create<VPU::GatherDMAOp>(origOp.getLoc(), origOp.getInput(), convertIndicesOp,
                                                         origOp.getAxisValue(), origOp.getBatchDims(),
                                                         /*multiClusterStrategy*/ nullptr, /*addressingMode*/ nullptr);

    // TODO (E#175972) Set ABSOLUTE Addressing mode when feature is enabled.
    // Until then will set default value as INDEXED addressing mode.
    // In order to set ABSOLUTE addressing mode we need to have tiling in order to satisfy HW requirements,
    // multiclustering and also tiling to fit in CMX done on the dim exactly after gather axis
    // (eg. for NCHW tensor : gather axis - C  tiling and multiclustering done on H)
    if (hasTilingDoneToBenefitAbsAddressing(origOp) && canHaveMulticlusteringToBenefitAbsAddressing(gatherDMAOp)) {
        gatherDMAOp.setAddressingMode(VPU::GatherAddressingMode::INDEXED);
    }
    auto reshapeOutOp =
            reshapeOperand(gatherDMAOp.getOutput(), outputType.getShape(), takeOpLoc(origOp, "reshape_output"));

    origOp.getOutput().replaceAllUsesWith(reshapeOutOp);
    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// MoveToDMAPass
//

class ConvertOpToDMAForPerformantExecutionPass final :
        public VPU::impl::ConvertOpToDMAForPerformantExecutionBase<ConvertOpToDMAForPerformantExecutionPass> {
public:
    explicit ConvertOpToDMAForPerformantExecutionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertOpToDMAForPerformantExecutionPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet splitPatterns(&ctx);
    splitPatterns.add<ExpandIndices>(&ctx, _log);
    walkAndApplyPatterns(func, std::move(splitPatterns));

    mlir::RewritePatternSet adaptionPatterns(&ctx);
    adaptionPatterns.add<TileGatherElement>(&ctx, _log);
    adaptionPatterns.add<TileGatherIndices>(&ctx, _log);
    walkAndApplyPatterns(func, std::move(adaptionPatterns));

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<MoveToDMAGather>(&ctx, _log);
    walkAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertOpToDMAForPerformantExecutionPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertOpToDMAForPerformantExecutionPass(Logger log) {
    return std::make_unique<ConvertOpToDMAForPerformantExecutionPass>(log);
}
