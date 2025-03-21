//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

mlir::SmallVector<int64_t> calcTileOutputShape(mlir::Value input, ArrayRef<int64_t> repeatsVals) {
    // If number of elements in *"repeats"* is more than shape of *"data"*, then *"data"* will be promoted to
    // "*repeats*" by prepending new axes, e.g. let's shape of *"data"* is equal to (2, 3) and *"repeats"* is equal to
    // [2, 2, 2], then shape of *"data"* will be promoted to (1, 2, 3) and result shape will be (2, 4, 6).
    //
    // If number of elements in *"repeats"* is less than shape of *"data"*, then *"repeats"* will be promoted to
    // "*data*" by prepending 1's to it, e.g. let's shape of *"data"* is equal to (4, 2, 3) and *"repeats"* is equal to
    // [2, 2], then *"repeats"* will be promoted to [1, 2, 2] and result shape will be (4, 4, 6)

    const auto inType = input.getType().cast<mlir::ShapedType>();
    auto outShape = to_small_vector(inType.getShape());

    while (repeatsVals.size() > outShape.size()) {
        outShape.insert(outShape.begin(), 1);
    }

    auto out_shape_iter = std::prev(outShape.end());
    auto repeats_iter = std::prev(repeatsVals.end());
    for (; out_shape_iter != std::prev(outShape.begin()) && repeats_iter != std::prev(repeatsVals.begin());
         --out_shape_iter, --repeats_iter) {
        *out_shape_iter *= *repeats_iter;
    }
    return outShape;
}
}  // namespace

mlir::LogicalResult vpux::IE::TileOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::TileOpAdaptor tile(operands, attrs, prop);
    if (mlir::failed(tile.verify(loc))) {
        return mlir::failure();
    }

    if (tile.getRepeats() != nullptr && tile.getRepeatsValues().has_value()) {
        return errorAt(loc, "Ambiguous repeats representation");
    }

    llvm::SmallVector<int64_t> repeatsVector;
    const auto inType = tile.getInput().getType().cast<mlir::ShapedType>();
    if (tile.getRepeats() != nullptr) {
        auto repeatsConst = tile.getRepeats().getDefiningOp<Const::DeclareOp>().getContent();
        repeatsVector = to_small_vector(repeatsConst.getValues<int64_t>());
    } else if (tile.getRepeatsValues().has_value()) {
        repeatsVector = parseIntArrayAttr<int64_t>(tile.getRepeatsValues().value());
    } else {
        return errorAt(loc, "Repeats was not provided properly");
    }
    auto outShape = calcTileOutputShape(tile.getInput(), repeatsVector);

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// ConvertRepeatsToAttr
//

namespace {
class ConvertRepeatsToAttr final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    using mlir::OpRewritePattern<IE::TileOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp tileOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertRepeatsToAttr::matchAndRewrite(IE::TileOp tileOp, mlir::PatternRewriter& rewriter) const {
    // check if input was already converted to Attr
    if (tileOp.getRepeatsValues().has_value()) {
        return mlir::failure();
    }
    // convert repeats into attribute
    const auto repeatsContent = tileOp.getRepeats().getDefiningOp<Const::DeclareOp>().getContent();
    auto repeats_values = to_small_vector(repeatsContent.getValues<int64_t>());
    const auto repeatsAttr = getIntArrayAttr(tileOp.getContext(), repeats_values);

    // rewrite layer pattern
    rewriter.replaceOpWithNewOp<IE::TileOp>(tileOp, tileOp.getInput(), nullptr, repeatsAttr);

    return mlir::success();
}

class AddUnsqueeze final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    using mlir::OpRewritePattern<IE::TileOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const final;
};

}  // namespace

//
// AddUnsqueeze
//

namespace {
mlir::LogicalResult AddUnsqueeze::matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const {
    // repeats attribute has no value
    if (!origOp.getRepeatsValues().has_value()) {
        return mlir::failure();
    }

    auto newRepeats = parseIntArrayAttr<int64_t>(origOp.getRepeatsValues().value());
    auto inputRank = origOp.getInput().getType().cast<mlir::ShapedType>().getRank();
    auto numRepeats = static_cast<int64_t>(newRepeats.size());
    int64_t nDimsToAdd = 0;

    if (numRepeats == inputRank) {
        // don't need to increase rank of input tensor or repeats size
        return mlir::failure();
    }
    if (numRepeats < inputRank) {
        // need to increase repeats size
        newRepeats.insert(newRepeats.begin(), inputRank - numRepeats, 1);

        auto newRepeatsAttr = getIntArrayAttr(origOp.getContext(), newRepeats);
        rewriter.replaceOpWithNewOp<IE::TileOp>(origOp, origOp.getType(), origOp.getInput(), nullptr, newRepeatsAttr);
    } else {
        // need to increase input rank
        nDimsToAdd = numRepeats - inputRank;

        SmallVector<int64_t> unsqueezeParam;
        for (int i = 0; i < nDimsToAdd; ++i) {
            unsqueezeParam.push_back(i);
        }

        const auto unsqueezeParamsAttr = getIntArrayAttr(getContext(), unsqueezeParam);
        auto unsqueezeOp =
                rewriter.create<IE::UnsqueezeOp>(origOp->getLoc(), origOp.getInput(), nullptr, unsqueezeParamsAttr);
        rewriter.replaceOpWithNewOp<IE::TileOp>(origOp, origOp.getType(), unsqueezeOp->getResult(0), nullptr,
                                                origOp.getRepeatsValuesAttr());
    }

    return mlir::success();
}

}  // namespace

//
// FuseTileOp
//

namespace {
class FuseTileOp final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    using mlir::OpRewritePattern<IE::TileOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp tileOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseTileOp::matchAndRewrite(IE::TileOp tileOp, mlir::PatternRewriter& rewriter) const {
    if (!tileOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto nextTileOp = mlir::dyn_cast_or_null<IE::TileOp>(*tileOp.getOutput().getUsers().begin());
    if (nextTileOp == nullptr) {
        return mlir::failure();
    }

    if (!tileOp.getRepeatsValues().has_value() || !nextTileOp.getRepeatsValues().has_value()) {
        return mlir::failure();
    }

    auto firstTileRepeatsVal = parseIntArrayAttr<int64_t>(tileOp.getRepeatsValues().value());
    auto secondTileRepeatsVal = parseIntArrayAttr<int64_t>(nextTileOp.getRepeatsValues().value());

    const auto firstRepeatsSize = firstTileRepeatsVal.size();
    const auto secondRepeatsSize = secondTileRepeatsVal.size();
    if (firstRepeatsSize != secondRepeatsSize) {
        if (firstRepeatsSize > secondRepeatsSize) {
            secondTileRepeatsVal.insert(secondTileRepeatsVal.begin(), firstRepeatsSize - secondRepeatsSize, 1);
        } else {
            firstTileRepeatsVal.insert(firstTileRepeatsVal.begin(), secondRepeatsSize - firstRepeatsSize, 1);
        }
    }

    SmallVector<int64_t> fusedRepeatsVal(firstTileRepeatsVal.size());
    std::transform(firstTileRepeatsVal.begin(), firstTileRepeatsVal.end(), secondTileRepeatsVal.begin(),
                   fusedRepeatsVal.begin(), std::multiplies<int>());

    rewriter.replaceOpWithNewOp<IE::TileOp>(nextTileOp, tileOp.getInput(), nullptr,
                                            getIntArrayAttr(tileOp.getContext(), fusedRepeatsVal));

    return mlir::success();
}

}  // namespace

void vpux::IE::TileOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<ConvertRepeatsToAttr>(ctx);
    patterns.add<AddUnsqueeze>(ctx);
    patterns.add<FuseTileOp>(ctx);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::TileOp::fold(FoldAdaptor adaptor) {
    // Tile with current param do nothing and should be optimized
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    auto operands = adaptor.getOperands();
    // move tile to const attribute as broadcast.
    if (auto contentAttr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto inputShape = to_small_vector(getShape(getInput()));
        const auto outputShape = to_small_vector(getShape(getOutput()));
        if (inputShape.size() != outputShape.size()) {
            // If input and output have different ranks, UnsqueezeOp will be inserted in canonicalization pattern to
            // guarantee same shape rank first.
            return nullptr;
        }

        auto contentAttrSetup = static_cast<Const::ContentAttr>(contentAttr).transform();

        for (const auto& dim : enumerate(outputShape)) {
            if (dim.value() > 1 && dim.value() != inputShape[dim.index()]) {
                contentAttrSetup = contentAttrSetup.broadcast(Dim(dim.index()), outputShape[dim.index()]);
            }
        }

        return contentAttrSetup.get();
    }
    return nullptr;
}

mlir::LogicalResult vpux::IE::PerAxisTileOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PerAxisTileOpAdaptor perAxisTile(operands, attrs, prop);
    if (mlir::failed(perAxisTile.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = perAxisTile.getInput().getType().cast<mlir::ShapedType>();

    const auto axis = checked_cast<unsigned int>(perAxisTile.getAxis());
    const auto tiles = checked_cast<unsigned int>(perAxisTile.getTiles());

    auto outShape = to_small_vector(inType.getShape());

    if (axis > outShape.size()) {
        return errorAt(loc, "Axis is out of range. Available range [0, {0}), but got axis = {1}", outShape.size(),
                       axis);
    }

    outShape[axis] *= tiles;

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}
