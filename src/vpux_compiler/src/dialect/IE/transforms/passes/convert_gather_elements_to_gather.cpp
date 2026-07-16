//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGATHERELEMENTSTOGATHER
#define GEN_PASS_DEF_CONVERTGATHERELEMENTSTOGATHER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Maximum number of indices that is safe for GatherDMA with non-constant indices.
// Hardware single-task limit is 65535 indices, but empirical testing shows that exceeding
// 32768 with non-constant indices causes device hang or performance regression,
// depending on platform. Using half the hardware limit as a safe upper bound.
// See E#149660 for details.
constexpr int64_t MAX_SAFE_GATHER_DMA_INDICES = 32768;

//
// GatherElementsFlatConverter
//
// Converts GatherElements to a sequence of chunked flat Gather ops suitable for GatherDMA.
// Each chunk has at most MAX_SAFE_GATHER_DMA_INDICES indices to avoid hardware hang.
//
// Applicable when:
//   - data and indices have the same shape
//   - all dimensions after the gather axis are 1 (or axis is last)
//   - total elements exceed the safe DMA limit (chunking needed)
//
// Algorithm:
//   1. Reshape data and indices to 2D: [batch, gather_size]
//   2. For each chunk of rows:
//      a. Slice data and indices
//      b. Add row offsets to indices (row_idx * gather_size)
//      c. Flatten both to 1D
//      d. Gather(flat_data, flat_indices_with_offsets, axis=0)
//      e. Reshape result back to 2D chunk
//   3. Concat all chunks and reshape to original shape
//

class GatherElementsFlatConverter final : public mlir::OpRewritePattern<IE::GatherElementsOp> {
public:
    GatherElementsFlatConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GatherElementsOp>(ctx, /*benefit=*/2), _log(std::move(log)) {
        setDebugName("GatherElementsFlatConverter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherElementsOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GatherElementsFlatConverter::matchAndRewrite(IE::GatherElementsOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto* ctx = origOp.getContext();
    const mlir::Location loc = origOp.getLoc();

    const auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(origOp.getInput().getType());
    const auto indicesType = mlir::dyn_cast<mlir::RankedTensorType>(origOp.getIndices().getType());

    // Reject dynamic shapes: reshape/slice need static dimension values
    if (!inputType || !indicesType || !inputType.hasStaticShape() || !indicesType.hasStaticShape()) {
        return matchFailed(rewriter, origOp, "Dynamic shapes are not supported");
    }

    const auto dataShape = getShape(origOp.getInput());
    const auto indicesShape = getShape(origOp.getIndices());

    // Require data and indices to have the same shape.
    // When shapes differ, the GatherElements semantics involve implicit broadcasting which
    // requires a different decomposition strategy. The existing GatherElementsOpConverter
    // handles the Tile-based pattern; other cases fall back to SHAVE execution.
    if (dataShape != indicesShape) {
        return matchFailed(rewriter, origOp, "Data and indices shapes differ");
    }

    const int64_t rank = static_cast<int64_t>(dataShape.size());
    int64_t axis = origOp.getAxis();
    if (axis < 0) {
        axis += rank;
    }

    // Validate axis is within valid range after normalization
    if (axis < 0 || axis >= rank) {
        return matchFailed(rewriter, origOp, "Axis out of range");
    }

    // All dimensions after axis must be 1
    for (int64_t d = axis + 1; d < rank; ++d) {
        if (dataShape[Dim(d)] != 1) {
            return matchFailed(rewriter, origOp, "Non-unit dimension after gather axis");
        }
    }

    const int64_t gatherSize = dataShape[Dim(axis)];
    int64_t batchSize = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (d != axis) {
            batchSize *= dataShape[Dim(d)];
        }
    }

    const int64_t totalElements = batchSize * gatherSize;
    if (totalElements <= MAX_SAFE_GATHER_DMA_INDICES) {
        return matchFailed(rewriter, origOp, "Total elements within safe limit, no chunking needed");
    }

    if (gatherSize > MAX_SAFE_GATHER_DMA_INDICES) {
        return matchFailed(rewriter, origOp, "Gather dimension exceeds safe DMA limit");
    }

    const int64_t rowsPerChunk = MAX_SAFE_GATHER_DMA_INDICES / gatherSize;
    if (rowsPerChunk == 0) {
        return matchFailed(rewriter, origOp, "Cannot form valid chunks");
    }

    // Determine indices element type and validate it is a supported integer width
    const auto indicesElemType = indicesType.getElementType();
    const bool isSI32 = indicesElemType == getSInt32Type(ctx);
    const bool isSI64 = indicesElemType == getSInt64Type(ctx);
    if (!isSI32 && !isSI64) {
        return matchFailed(rewriter, origOp, "Unsupported indices element type (only si32/si64 supported)");
    }

    _log.trace("Converting GatherElements: batch={0}, gatherSize={1}, rowsPerChunk={2}, numChunks={3}", batchSize,
               gatherSize, rowsPerChunk, (batchSize + rowsPerChunk - 1) / rowsPerChunk);

    // Reshape data and indices to 2D: [batchSize, gatherSize]
    const auto shape2D = SmallVector<int64_t>{batchSize, gatherSize};
    auto dataReshaped =
            rewriter.create<IE::ReshapeOp>(appendLoc(loc, "data_2d"), origOp.getInput(), getIntArrayAttr(ctx, shape2D))
                    .getOutput();
    auto indicesReshaped = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "indices_2d"), origOp.getIndices(),
                                                          getIntArrayAttr(ctx, shape2D))
                                   .getOutput();

    // Build row offsets constant matching the indices element type
    auto buildRowOffsets = [&](int64_t rows, llvm::StringRef suffix) -> mlir::Value {
        const int64_t count = rows * gatherSize;
        auto type = mlir::RankedTensorType::get({count}, indicesElemType);
        if (isSI32) {
            std::vector<int32_t> values(count);
            for (int64_t r = 0; r < rows; ++r) {
                const auto offset = static_cast<int32_t>(r * gatherSize);
                std::fill_n(values.begin() + r * gatherSize, gatherSize, offset);
            }
            return Const::createConst(rewriter, appendLoc(loc, suffix), type, ArrayRef(values));
        }
        std::vector<int64_t> values(count);
        for (int64_t r = 0; r < rows; ++r) {
            const auto offset = r * gatherSize;
            std::fill_n(values.begin() + r * gatherSize, gatherSize, offset);
        }
        return Const::createConst(rewriter, appendLoc(loc, suffix), type, ArrayRef(values));
    };

    auto offsetConst = buildRowOffsets(rowsPerChunk, "row_offsets");

    // Process chunks
    SmallVector<mlir::Value> chunkResults;
    const int64_t numChunks = (batchSize + rowsPerChunk - 1) / rowsPerChunk;

    for (int64_t chunk = 0; chunk < numChunks; ++chunk) {
        const int64_t startRow = chunk * rowsPerChunk;
        const int64_t endRow = std::min(startRow + rowsPerChunk, batchSize);
        const int64_t chunkRows = endRow - startRow;
        const int64_t chunkElements = chunkRows * gatherSize;

        // Slice data chunk: [chunkRows, gatherSize]
        auto dataSlice =
                rewriter.createOrFold<IE::SliceOp>(appendLoc(loc, "data_chunk_{0}", chunk), dataReshaped,
                                                   getIntArrayAttr(ctx, SmallVector<int64_t>{startRow, 0}),
                                                   getIntArrayAttr(ctx, SmallVector<int64_t>{chunkRows, gatherSize}));

        // Slice indices chunk: [chunkRows, gatherSize]
        auto indicesSlice =
                rewriter.createOrFold<IE::SliceOp>(appendLoc(loc, "indices_chunk_{0}", chunk), indicesReshaped,
                                                   getIntArrayAttr(ctx, SmallVector<int64_t>{startRow, 0}),
                                                   getIntArrayAttr(ctx, SmallVector<int64_t>{chunkRows, gatherSize}));

        // Flatten indices to 1D: [chunkElements]
        auto indicesFlat = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "indices_flat_{0}", chunk), indicesSlice,
                                                          getIntArrayAttr(ctx, SmallVector<int64_t>{chunkElements}))
                                   .getOutput();

        // Add row offsets to indices
        mlir::Value offsetVal =
                (chunkRows == rowsPerChunk) ? offsetConst : buildRowOffsets(chunkRows, "row_offsets_last");

        auto indicesAbsolute =
                rewriter.create<IE::AddOp>(appendLoc(loc, "indices_abs_{0}", chunk), indicesFlat, offsetVal,
                                           IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT),
                                           nullptr, nullptr, nullptr, nullptr);

        // Flatten data to 1D: [chunkElements]
        auto dataFlat = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "data_flat_{0}", chunk), dataSlice,
                                                       getIntArrayAttr(ctx, SmallVector<int64_t>{chunkElements}))
                                .getOutput();

        // Gather: output[i] = data_flat[indices_absolute[i]]
        auto gatherResult = rewriter.create<IE::GatherOp>(appendLoc(loc, "gather_{0}", chunk), dataFlat,
                                                          indicesAbsolute.getOutput(), nullptr, getIntAttr(ctx, 0),
                                                          /*batchDims=*/0, nullptr);

        // Reshape to 2D chunk: [chunkRows, gatherSize]
        auto chunkReshaped =
                rewriter.create<IE::ReshapeOp>(appendLoc(loc, "chunk_2d_{0}", chunk), gatherResult.getOutput(),
                                               getIntArrayAttr(ctx, SmallVector<int64_t>{chunkRows, gatherSize}))
                        .getOutput();

        chunkResults.push_back(chunkReshaped);
    }

    // Concat all chunks along dim 0: [batchSize, gatherSize]
    auto combined = rewriter.create<IE::ConcatOp>(appendLoc(loc, "concat_chunks"), mlir::ValueRange(chunkResults),
                                                  getIntAttr(ctx, 0))
                            .getOutput();

    // Reshape back to original shape
    const auto origShapeVec = to_small_vector(dataShape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, combined, getIntArrayAttr(ctx, origShapeVec));

    return mlir::success();
}

//
//   Convert GatherElementsOp to GatherOp
//
//   input0: 1x1x5376x80    input1: 1x1x300x1
//        |                     |
//        |               ┌────────────┐
//        │               │    Tile    │ repeats_values: [1, 1, 1, 80]
//        │               └─────┬──────┘
//         \                   / 1x1x300x80
//           ┌────────────────┐
//           │ GatherElements │ axis = 2
//           └────────┬───────┘
//                    │
//           output: 1x1x300x80
//
//                 ======►
//
//   input0: 1x1x5376x80    input1: 1x1x300x1
//        |                     |
//        |               ┌────────────┐
//        │               │   Squeeze  │ axes_values: [0, 1, 3]
//        │               └─────┬──────┘
//         \                   / 300
//           ┌────────────────┐
//           │     Gather     │ axis_value = 2
//           └────────┬───────┘
//                    │
//           output: 1x1x300x80
//

//
// GatherElementsOpConverter
//

class GatherElementsOpConverter final : public mlir::OpRewritePattern<IE::GatherElementsOp> {
public:
    GatherElementsOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GatherElementsOp>(ctx), _log(std::move(log)) {
        setDebugName("GatherElementsOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherElementsOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GatherElementsOpConverter::matchAndRewrite(IE::GatherElementsOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto& ctx = origOp.getContext();
    const mlir::Location location = origOp.getLoc();

    auto maybeTileOp = mlir::dyn_cast_or_null<IE::TileOp>(origOp.getOperand(1).getDefiningOp());
    if (maybeTileOp == nullptr) {
        return matchFailed(rewriter, origOp, "No parent TileOp found");
    }

    auto tileOpInShape = getShape(maybeTileOp.getInput());
    SmallVector<Dim> nonOneDims = getNonOneDim(tileOpInShape);
    if (nonOneDims.size() > 1) {
        return matchFailed(rewriter, maybeTileOp, "Not supported TileOp with input shape of non-one dims size > 1");
    }

    // Get repeatsValues from TileOp
    auto repeatsValues = maybeTileOp.getRepeatsValues();
    if (!repeatsValues.has_value()) {
        return matchFailed(rewriter, maybeTileOp, "No repeats values found");
    }

    auto getNonOneRepeatAxesValue = [](mlir::ArrayAttr repeatsValues) -> SmallVector<std::pair<size_t, size_t>> {
        SmallVector<std::pair<size_t, size_t>> repeatAxesValue;
        auto repeatsVector = parseIntArrayAttr<int64_t>(repeatsValues);
        for (const auto& repeatValue : repeatsVector | indexed) {
            if (repeatValue.value() != 1) {
                repeatAxesValue.emplace_back(repeatValue.index(), repeatValue.value());
            }
        }
        return repeatAxesValue;
    };

    auto repeatAxesValue = getNonOneRepeatAxesValue(repeatsValues.value());
    if (repeatAxesValue.size() != 1) {
        return matchFailed(rewriter, maybeTileOp, "Not one repeat axis");
    }

    int64_t repeatAxis = repeatAxesValue[0].first;
    if (nonOneDims[0] == Dim(repeatAxis)) {
        return matchFailed(rewriter, maybeTileOp, "Not supported repeat axis");
    }

    // Get axis from GatherElementsOp
    int64_t axis = origOp.getAxis();
    if (Dim(axis) != nonOneDims[0]) {
        return matchFailed(rewriter, origOp, "Not supported GatherElementsOp axis");
    }

    auto origOpInShape = getShape(origOp.getInput());
    if (origOpInShape[Dim(repeatAxis)] != (int64_t)repeatAxesValue[0].second) {
        return matchFailed(rewriter, origOp, "Not supported GatherElementsOp shape");
    }

    auto generateAxesValue = [](size_t shapeSize, int64_t nonOneAxis) -> SmallVector<size_t> {
        SmallVector<size_t> axisOneArray;
        for (auto index : irange(shapeSize)) {
            if (index != (size_t)nonOneAxis) {
                axisOneArray.push_back(index);
            }
        }
        return axisOneArray;
    };

    // Create SqueezeOp
    auto axisOneArray = generateAxesValue(tileOpInShape.size(), axis);
    const auto axisOneArrayAttr = getIntArrayAttr(ctx, axisOneArray);
    auto squeezeOpResult = rewriter.createOrFold<IE::SqueezeOp>(appendLoc(location, "squeeze"), maybeTileOp.getInput(),
                                                                nullptr, axisOneArrayAttr);

    // Create GatherOp
    int64_t batchDims = 0;
    rewriter.replaceOpWithNewOp<IE::GatherOp>(origOp, origOp.getOperand(0), squeezeOpResult, nullptr,
                                              getIntAttr(ctx, axis), batchDims, nullptr);

    return mlir::success();
}

//
// ConvertGatherElementsToGatherPass
//

class ConvertGatherElementsToGatherPass final :
        public IE::impl::ConvertGatherElementsToGatherBase<ConvertGatherElementsToGatherPass> {
public:
    explicit ConvertGatherElementsToGatherPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertGatherElementsToGatherPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GatherElementsFlatConverter>(&ctx, _log);
    patterns.add<GatherElementsOpConverter>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertGatherElementsToGatherPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGatherElementsToGatherPass(Logger log) {
    return std::make_unique<ConvertGatherElementsToGatherPass>(log);
}
