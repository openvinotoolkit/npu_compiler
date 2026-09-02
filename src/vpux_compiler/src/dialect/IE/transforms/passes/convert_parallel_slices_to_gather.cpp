//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/gather_dma_constants.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTPARALLELSLICESTOGATHER
#define GEN_PASS_DEF_CONVERTPARALLELSLICESTOGATHER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MergeParallelGatherSameIndices
//
// Fuses N parallel embedding Gathers sharing the same token index into one Gather.
// Requires all Gather outputs to be direct inputs of a single downstream ConcatOp.
//
// Before:
//   table_0 ──┐         table_1 ──┐         table_N-1 ──┐
//   Gather(·,idx)    Gather(·,idx)    ...   Gather(·,idx)
//       │                │                       │
//       └────────────────┴───── Concat ───────────┘
//
// After:
//   Concat(table_0,..,table_N-1) ──┐
//   Add(idx, [0,S,2S,..(N-1)S]) ───┤  S = table_size
//                           Gather ─┤
//                          Reshape ──► (same shape as original Concat output)
//

class MergeParallelGatherSameIndices final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    MergeParallelGatherSameIndices(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("MergeParallelGatherSameIndices");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MergeParallelGatherSameIndices::matchAndRewrite(IE::ConcatOp concatOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got ConcatOp '{0}'", concatOp->getLoc());

    const int64_t numGathers = checked_cast<int64_t>(concatOp.getInputs().size());
    if (numGathers < 2) {
        return mlir::failure();
    }
    if (!IE::getConcatAxis(concatOp).has_value()) {
        return mlir::failure();
    }

    // All inputs must be single-use GatherOps sharing the same indices value.
    SmallVector<IE::GatherOp> gatherOps;
    gatherOps.reserve(numGathers);
    for (auto input : concatOp.getInputs()) {
        auto g = mlir::dyn_cast_if_present<IE::GatherOp>(input.getDefiningOp());
        if (g == nullptr || !g->hasOneUse()) {
            return mlir::failure();
        }
        gatherOps.push_back(g);
    }

    // Indices must be a 1-D single-token tensor [1] of si32 or si64.
    const auto indices = gatherOps.front().getIndices();
    const auto indicesType = mlir::cast<NDTypeInterface>(indices.getType());
    const auto indicesElemType = indicesType.getElementType();
    if (!indicesElemType.isInteger(32) && !indicesElemType.isInteger(64)) {
        return mlir::failure();
    }
    const auto indicesShape = indicesType.getShape();
    if (indicesShape.size() != 1 || indicesShape[Dim(0)] != 1) {
        return mlir::failure();
    }

    // All GatherOps must share axis, batch_dims == 0, indicesRank, and constant table of equal shape.
    auto anchor = gatherOps.front();
    if (anchor.getBatchDims() != 0) {
        return mlir::failure();
    }
    const auto anchorAxis = anchor.getAxisValue();
    const auto anchorTableShape = getShape(anchor.getInput());

    SmallVector<mlir::Value> tableInputs;
    tableInputs.reserve(numGathers);

    for (auto g : gatherOps) {
        if (g.getAxisValue() != anchorAxis) {
            return mlir::failure();
        }
        if (g.getBatchDims() != 0) {
            return mlir::failure();
        }
        if (g.getIndicesRankAttr() != anchor.getIndicesRankAttr()) {
            return mlir::failure();
        }
        if (g.getInput().getDefiningOp<Const::DeclareOp>() == nullptr) {
            return mlir::failure();
        }
        if (getShape(g.getInput()) != anchorTableShape) {
            return mlir::failure();
        }
        if (g.getIndices() != indices) {
            return mlir::failure();
        }
        tableInputs.push_back(g.getInput());
    }

    // Guard for si32: the largest offset (numGathers-1)*tableSize must fit in int32.
    const int64_t tableSize = anchorTableShape[Dim(anchorAxis)];
    if (!indicesElemType.isInteger(64) && (numGathers - 1) * tableSize > std::numeric_limits<int32_t>::max()) {
        return mlir::failure();
    }

    _log.trace("Merging {0} parallel GatherOps into single Gather+Reshape", numGathers);

    auto concatTables = rewriter.create<IE::ConcatOp>(concatOp->getLoc(), tableInputs, Dim(anchorAxis));
    const auto offsetType = mlir::RankedTensorType::get({numGathers}, indicesElemType);
    auto makeOffsets = [&](auto zeroTag) -> mlir::Value {
        using T = decltype(zeroTag);
        SmallVector<T> vals(numGathers);
        for (int64_t i = 0; i < numGathers; ++i) {
            vals[i] = checked_cast<T>(i * tableSize);
        }
        return Const::createConst(rewriter, concatOp->getLoc(), offsetType, ArrayRef<T>(vals));
    };
    mlir::Value offsetConst = indicesElemType.isInteger(64) ? makeOffsets(int64_t{0}) : makeOffsets(int32_t{0});

    auto numpyAttr = IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NUMPY);
    auto mergedIdx = rewriter.create<IE::AddOp>(concatOp->getLoc(), indices, offsetConst, numpyAttr,
                                                /*post_op=*/nullptr, /*clamp=*/nullptr,
                                                /*output_padding=*/nullptr, /*input_padding=*/nullptr);

    auto mergedGather =
            rewriter.create<IE::GatherOp>(concatOp->getLoc(), concatTables.getOutput(), mergedIdx.getOutput(),
                                          anchor.getAxisValue(), anchor.getBatchDims(), anchor.getIndicesRankAttr());

    const auto concatShape = getShape(concatOp.getOutput());
    auto reshaped = rewriter.create<IE::ReshapeOp>(concatOp->getLoc(), mergedGather.getOutput(),
                                                   getIntArrayAttr(rewriter.getContext(), concatShape));

    rewriter.replaceOp(concatOp, reshaped.getOutput());
    return mlir::success();
}

class ConvertToGather final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConvertToGather(mlir::MLIRContext* ctx, size_t maxGatherDMAIndicesListLength, size_t maxGatherDMAElementSize,
                    Logger log)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx),
              _maxGatherDMAIndicesListLength(maxGatherDMAIndicesListLength),
              _maxGatherDMAElementSize(maxGatherDMAElementSize),
              _log(log) {
        setDebugName("ConvertToGather");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t SUPPORTED_RANK = 2;
    bool isLegalConcat(IE::ConcatOp origOp) const;
    SmallVector<IE::SliceOp> getValidInputs(IE::ConcatOp origOp) const;
    SmallVector<SmallVector<IE::SliceOp>> groupSliceOperations(SmallVector<IE::SliceOp> sliceOps) const;
    mlir::Value convertSliceGroupToGather(SmallVector<IE::SliceOp> groupedSliceOps,
                                          mlir::PatternRewriter& rewriter) const;
    mlir::Value reshape3DGatherTo2D(mlir::Value gather, mlir::PatternRewriter& rewriter) const;

    size_t _maxGatherDMAIndicesListLength;
    size_t _maxGatherDMAElementSize;
    Logger _log;
};

bool ConvertToGather::isLegalConcat(IE::ConcatOp origOp) const {
    auto concatType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    auto concatShape = concatType.getShape();
    if (checked_cast<int64_t>(concatShape.size()) != SUPPORTED_RANK) {
        return false;
    }

    const auto elementByteSize = concatType.getElemTypeSize().to<Byte>().count();

    auto canBeSupportedByGatherDMA =
            concatShape[Dim(0)] <= checked_cast<int64_t>(_maxGatherDMAIndicesListLength) &&
            concatShape[Dim(1)] * elementByteSize <= checked_cast<int64_t>(_maxGatherDMAElementSize);
    if (!canBeSupportedByGatherDMA) {
        _log.trace(
                "Not convert {0} to prevent tiled GatherDMA becasue tiled GatherDMA is not going to be more efficient",
                origOp);
        return false;
    }

    return true;
}

// Try to match valid inputs pattern:
// 1. All inputs of ConcatOp should be SliceOps
// 2. All SliceOps should have the same shape [1, elementSize]
// 3. All SliceOps should be concatenated on d0
// 4. All SliceOps offset[1] should can be divided by elementSize
SmallVector<IE::SliceOp> ConvertToGather::getValidInputs(IE::ConcatOp origOp) const {
    SmallVector<IE::SliceOp> sliceOps;
    for (auto input : origOp.getInputs()) {
        auto sliceOp = input.getDefiningOp<IE::SliceOp>();
        if (sliceOp == nullptr || !sliceOp->hasOneUse()) {
            return {};
        }
        sliceOps.push_back(sliceOp);
    }

    auto concatShape = getShape(origOp.getOutput());
    const auto elementSize = concatShape[Dim(1)];
    auto haveSameShapeAndValidOffsets = [&](IE::SliceOp slice) {
        auto shape = getShape(slice.getResult());
        auto offsets = parseIntArrayAttr<int64_t>(slice.getStaticOffsets());

        return checked_cast<int64_t>(shape.size()) == SUPPORTED_RANK && shape[Dim(0)] == 1 &&
               shape[Dim(1)] == elementSize && checked_cast<int64_t>(offsets.size()) == SUPPORTED_RANK &&
               offsets[1] % elementSize == 0;
    };

    if (!llvm::all_of(sliceOps, haveSameShapeAndValidOffsets)) {
        return {};
    }

    return sliceOps;
}

SmallVector<SmallVector<IE::SliceOp>> ConvertToGather::groupSliceOperations(SmallVector<IE::SliceOp> sliceOps) const {
    SmallVector<SmallVector<IE::SliceOp>> groupedSliceOps;
    if (sliceOps.empty()) {
        return groupedSliceOps;
    }

    // Initialize the first group
    SmallVector<IE::SliceOp> currentGroup;
    mlir::Value currentInput = sliceOps[0].getSource();

    for (auto sliceOp : sliceOps) {
        auto input = sliceOp.getSource();
        if (input == currentInput) {
            // Push SliceOp to current group if it has the same source
            currentGroup.push_back(sliceOp);
        } else {
            // If SliceOp has a different source, store current group and push the SliceOp into a new group
            groupedSliceOps.push_back(std::move(currentGroup));
            currentGroup = {sliceOp};
            currentInput = input;
        }
    }

    // Store the last group
    if (!currentGroup.empty()) {
        groupedSliceOps.push_back(currentGroup);
    }

    return groupedSliceOps;
}

/*
    When SliceOp branches share the same source and are finally concatenated,
    these Slice operations can be converted into single Gather operation.

    For NPU4000+, Gather operation will be mapped to GatherDMA.
    The conversion in below can reduce DMA workloads.

    Convert subgraph:

            source(8x12288)                             source(6x9216)
    /               |               \                   /           \
Slice(1x1536)   Slice(1x1536)   ... Slice(1x1536)   Slice(1x1536)   ... Slice(1x1536)
    \               |                  |                |                   /
     -----------------------------------------------------------------------------
                                            |
                                    Concat(14x1536)
                                            |

    to:

        source(8x12288)                         source(6x9216)
            |                                       |
    Reshape(64x1536)    Indices(1x8)        Reshape(36x1536)    Indices(1x6)
            \               /                       \               /
            Gather(1x8x1536)                        Gather(1x6x1536)
                    |                                       |
                Reshape(8x1536)                 Reshape(6x1536)
                            \                   /
                                Concat(14x1536)
                                        |

*/

mlir::Value ConvertToGather::convertSliceGroupToGather(SmallVector<IE::SliceOp> groupedSliceOps,
                                                       mlir::PatternRewriter& rewriter) const {
    auto source = groupedSliceOps.front().getSource();
    auto sourceShape = getShape(source);

    auto sliceShape = getShape(groupedSliceOps.front().getResult());
    // Source is with 2D shape: [Height, (chunkNum x elementSize)]
    // Reshape source shape to [(Height x chunkNum), elementSize]
    // Build indices constant for Gather operation to select corresponding elements
    auto elementSize = sliceShape.back();
    auto chunkNum = sourceShape.back() / elementSize;
    VPUX_THROW_UNLESS(sourceShape.back() % elementSize == 0, "Source shape {0} does not match with elementSize {1}",
                      sourceShape, elementSize);

    auto newInputShapeVec = SmallVector<int64_t>(SUPPORTED_RANK, 1);
    newInputShapeVec[0] = sourceShape.totalSize() / elementSize;
    newInputShapeVec[1] = elementSize;
    const auto newType = mlir::cast<NDTypeInterface>(source.getType()).changeShape(ShapeRef(newInputShapeVec));
    auto newInput = rewriter.create<IE::ReshapeOp>(source.getLoc(), newType, source,
                                                   getIntArrayAttr(rewriter.getContext(), ShapeRef(newInputShapeVec)));

    const auto numOfSliceOps = groupedSliceOps.size();
    _log.debug("{0} Slice operations in current group:", numOfSliceOps);
    SmallVector<int32_t> indicesValues(numOfSliceOps);
    for (size_t i = 0; i < indicesValues.size(); ++i) {
        auto sliceOp = groupedSliceOps[i];
        auto offsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
        indicesValues[i] = checked_cast<int32_t>(offsets[0] * chunkNum + offsets[0]);
        _log.nest().debug("Set indices to {0} for sliceOp {1} ", indicesValues[i], sliceOp);
    }
    auto indicesShapeVec = SmallVector<int64_t>(SUPPORTED_RANK, 1);
    indicesShapeVec[1] = checked_cast<int64_t>(numOfSliceOps);

    const auto indicesType =
            mlir::RankedTensorType::get(ShapeRef(indicesShapeVec), getSInt32Type(rewriter.getContext()));

    const auto indices = Const::createConst(rewriter, source.getLoc(), indicesType, ArrayRef(indicesValues));

    int64_t batchDims = 0;
    int64_t axis = 0;
    auto newGather = rewriter.create<IE::GatherOp>(source.getLoc(), newInput, indices, axis, batchDims, nullptr);

    _log.trace("Created gather {0}", newGather);
    return newGather.getOutput();
}

mlir::Value ConvertToGather::reshape3DGatherTo2D(mlir::Value gather, mlir::PatternRewriter& rewriter) const {
    auto origGatherShape = getShape(gather);
    VPUX_THROW_UNLESS(origGatherShape.size() == 3, "Gather should be 3D shape but got {0}D", origGatherShape.size());

    auto newShapeVec = SmallVector<int64_t>(SUPPORTED_RANK, 1);
    newShapeVec[0] = origGatherShape[Dim(0)] * origGatherShape[Dim(1)];
    newShapeVec[1] = origGatherShape[Dim(2)];
    auto newGatherShape = ShapeRef(newShapeVec);
    _log.debug("Reshape Gather shape from {0} to {1}", origGatherShape, newGatherShape);
    auto newType = mlir::cast<NDTypeInterface>(gather.getType()).changeShape(newGatherShape);
    auto reshape = rewriter.create<IE::ReshapeOp>(gather.getLoc(), newType, gather,
                                                  getIntArrayAttr(rewriter.getContext(), newGatherShape));

    _log.trace("Created Reshape {0}", reshape);
    return reshape.getOutput();
}

mlir::LogicalResult ConvertToGather::matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!isLegalConcat(origOp)) {
        return mlir::failure();
    }

    auto sliceOps = getValidInputs(origOp);
    if (sliceOps.empty()) {
        return mlir::failure();
    }

    SmallVector<SmallVector<IE::SliceOp>> groupedSliceOps = groupSliceOperations(std::move(sliceOps));
    if (groupedSliceOps.empty()) {
        return mlir::failure();
    }

    auto containOneSliceInGroup = [](ArrayRef<IE::SliceOp> opsInGroup) {
        return opsInGroup.size() == 1;
    };
    if (llvm::all_of(groupedSliceOps, containOneSliceInGroup)) {
        _log.trace("Skip conversion: all groups contain only one SliceOp");
        return mlir::failure();
    }

    // All Slice operations have 2D shape [1, elementSize]
    // Source should be with 2D shape: [Height, (chunkNum x elementSize)]
    auto sliceShape = getShape(groupedSliceOps.front().front().getResult());
    auto elementSize = sliceShape.back();
    auto isSourceShapeCompatibleWithElementSize = [elementSize](SmallVector<IE::SliceOp> group) {
        auto source = group.front().getSource();
        auto sourceShape = getShape(source);
        return sourceShape.back() % elementSize == 0;
    };

    if (!llvm::all_of(groupedSliceOps, isSourceShapeCompatibleWithElementSize)) {
        _log.trace("Source shape does not match with Slice shape");
        return mlir::failure();
    }

    _log.trace("Start to convert {0}", origOp);

    SmallVector<mlir::Value> newConcatInput;
    for (auto& group : groupedSliceOps) {
        // No need to convert single SliceOp in group to Gather
        if (group.size() == 1) {
            newConcatInput.push_back(group.front().getResult());
            continue;
        }

        auto gather = convertSliceGroupToGather(group, rewriter);
        auto reshape = reshape3DGatherTo2D(gather, rewriter);
        newConcatInput.push_back(reshape);
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(origOp.getLoc(), newConcatInput, Dim(0));

    rewriter.replaceOp(origOp, newConcat);

    return mlir::success();
}

//
// ConvertParallelSlicesToGatherPass
//

class ConvertParallelSlicesToGatherPass final :
        public IE::impl::ConvertParallelSlicesToGatherBase<ConvertParallelSlicesToGatherPass> {
public:
    explicit ConvertParallelSlicesToGatherPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    size_t _maxGatherDMAIndicesListLength{};
    size_t _maxGatherDMAElementSize{};
};

void ConvertParallelSlicesToGatherPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    const auto arch = config::getArch(func);
    _maxGatherDMAIndicesListLength = VPU::getGatherDMAMaxIndicesListLength(arch);
    _maxGatherDMAElementSize = VPU::getGatherDMAMaxElementSize(arch);
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertToGather>(&ctx, _maxGatherDMAIndicesListLength, _maxGatherDMAElementSize, _log);
    patterns.add<MergeParallelGatherSameIndices>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertParallelSlicesToGatherPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertParallelSlicesToGatherPass(Logger log) {
    return std::make_unique<ConvertParallelSlicesToGatherPass>(log);
}
