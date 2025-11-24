//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

namespace {

template <typename TransposeType, std::enable_if_t<or_<std::is_same<TransposeType, IE::TransposeOp>,
                                                       std::is_same<TransposeType, IE::TransposeOpAdaptor>>::value,
                                                   bool> = true>
mlir::LogicalResult getOrder(TransposeType transpose, SmallVector<uint64_t>& order, mlir::Location loc) {
    const auto getDefaultOrder = [](mlir::ShapedType inType) {
        SmallVector<uint64_t> orderIndices{};
        for (const auto& idx : irange(inType.getRank()) | reversed) {
            orderIndices.push_back(idx);
        }

        return orderIndices;
    };

    if (transpose.getOrder() != nullptr && transpose.getOrderValue().has_value()) {
        return errorAt(loc, "Ambiguous order representation");
    }
    if (transpose.getOrder() == nullptr && !transpose.getOrderValue().has_value()) {
        return errorAt(loc, "Missed order representation");
    }

    const auto inDataType = mlir::cast<mlir::ShapedType>(transpose.getInput().getType());

    if (transpose.getOrder() != nullptr) {
        auto orderOp = transpose.getOrder().template getDefiningOp<Const::DeclareOp>();
        if (orderOp == nullptr) {
            return errorAt(loc, "Only constant input is supported");
        }

        const auto orderContent = orderOp.getContent();
        const auto orderVals = orderContent.template getValues<uint64_t>();

        order = orderVals.empty() ? getDefaultOrder(inDataType) : to_small_vector(orderVals);

        return mlir::success();
    }

    const auto perm = DimsOrder::fromAffineMap(transpose.getOrderValue().value());
    order = to_small_vector(irange(perm.numDims()) | transformed([&](uint64_t idx) {
                                return checked_cast<uint64_t>(perm.dimAt(idx).ind());
                            }));

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::TransposeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::TransposeOpAdaptor transpose(operands, attrs, prop);
    if (mlir::failed(transpose.verify(loc))) {
        return mlir::failure();
    }

    const auto inDataType = mlir::cast<mlir::RankedTensorType>(transpose.getInput().getType());
    SmallVector<uint64_t> order{};
    if (::getOrder(transpose, order, loc).failed()) {
        return mlir::failure();
    }

    if (inDataType.getShape().size() != order.size()) {
        return errorAt(loc, "Order vector size doesn't match input rank");
    }

    const auto [outShape, outDesc] = callOnShapeOf(inDataType, [&](const auto& inShape) {
        const auto outRank = static_cast<uint64_t>(inShape.size());
        auto outAnyShape = makeShape(inShape, outRank, 1);

        for (size_t i = 0; i < order.size(); ++i) {
            VPUX_THROW_WHEN(order[i] >= outRank, "Order index: {0} for dim: {1} is higher than the shape rank: {2}.",
                            order[i], i, outRank);
            outAnyShape[Dim(i)] = inShape[Dim(order[i])];
        }

        auto outShape = extractShape(outAnyShape);
        const auto outDesc = getTensorAttr(inDataType, vpux::getOrder(inDataType), /*memSpace=*/nullptr, outAnyShape);
        return std::make_pair(std::move(outShape), outDesc);
    });

    SmallVector<uint32_t> uorder;
    std::transform(order.begin(), order.end(), std::back_inserter(uorder), [](uint64_t dim) -> uint32_t {
        return static_cast<uint32_t>(dim);
    });

    const auto permutationMap = mlir::AffineMap::getPermutationMap(ArrayRef(uorder), ctx);
    const auto outputElemType = inferElemTypeTranspose(permutationMap, inDataType.getElementType());

    inferredReturnShapes.emplace_back(outShape.raw(), outputElemType, outDesc);
    return mlir::success();
}

namespace {

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    using mlir::OpRewritePattern<IE::TransposeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp transposeOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::TransposeOp transposeOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (transposeOp.getOrderValue().has_value()) {
        return mlir::failure();
    }

    SmallVector<uint64_t> order{};
    if (getOrder(transposeOp, order, transposeOp->getLoc()).failed()) {
        return mlir::failure();
    }

    const auto perm = to_small_vector(order | transformed([](uint64_t val) {
                                          return checked_cast<unsigned>(val);
                                      }));

    const auto orderAttr =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, transposeOp->getContext()));

    auto newOp = rewriter.replaceOpWithNewOp<IE::TransposeOp>(transposeOp, transposeOp.getType(),
                                                              transposeOp.getInput(), nullptr, orderAttr);
    extendOpLoc(newOp, "const_to_attr");

    return mlir::success();
}

//
// FuseTransposes
//

class FuseTransposes final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    using mlir::OpRewritePattern<IE::TransposeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp transposeOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseTransposes::matchAndRewrite(IE::TransposeOp transposeOp,
                                                    mlir::PatternRewriter& rewriter) const {
    if (!transposeOp.getInput().hasOneUse()) {
        return mlir::failure();
    }

    auto prevTransposeOp = mlir::dyn_cast_or_null<IE::TransposeOp>(transposeOp.getInput().getDefiningOp());
    if (prevTransposeOp == nullptr) {
        return mlir::failure();
    }

    SmallVector<uint64_t> prevOrder{};
    VPUX_THROW_UNLESS(getOrder(prevTransposeOp, prevOrder, prevTransposeOp->getLoc()).succeeded(),
                      "Failed to get order for Transpose operation '{0}'", prevTransposeOp->getName());

    SmallVector<uint64_t> order{};
    VPUX_THROW_UNLESS(getOrder(transposeOp, order, transposeOp->getLoc()).succeeded(),
                      "Failed to get order for Transpose operation '{0}'", transposeOp->getName());

    const auto prevPerm = to_small_vector(prevOrder | transformed([](uint64_t val) {
                                              return checked_cast<unsigned>(val);
                                          }));

    const auto perm = to_small_vector(order | transformed([](uint64_t val) {
                                          return checked_cast<unsigned>(val);
                                      }));

    auto prevPermMap = mlir::AffineMap::getPermutationMap(prevPerm, transposeOp->getContext());
    auto permMap = mlir::AffineMap::getPermutationMap(perm, transposeOp->getContext());

    const auto permAttr = mlir::AffineMapAttr::get(permMap.compose(prevPermMap));
    auto newOp = rewriter.replaceOpWithNewOp<IE::TransposeOp>(transposeOp, transposeOp.getType(),
                                                              prevTransposeOp.getInput(), nullptr, permAttr);
    extendOpLoc(newOp, "fused");

    return mlir::success();
}

//
// CollapseMutualTransposes
//

class CollapseMutualTransposes final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    using mlir::OpRewritePattern<IE::TransposeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp transposeOp, mlir::PatternRewriter& rewriter) const final;
};

bool isTrivialTransformation(const ShapeRef inShape, const ShapeRef outShape) {
    // Remove all trivial dimensions.
    const auto isTrivial = [](const int64_t dimVal) -> bool {
        return dimVal > 1;
    };
    Shape inShapeDiminished;
    std::copy_if(inShape.begin(), inShape.end(), std::back_inserter(inShapeDiminished), isTrivial);

    Shape outShapeDiminished;
    std::copy_if(outShape.begin(), outShape.end(), std::back_inserter(outShapeDiminished), isTrivial);

    // Check that resulting shapes preserve the order of dimensions.
    return inShapeDiminished == outShapeDiminished;
}

bool canBeCollapsed(IE::TransposeOp op) {
    // First, check whether the input of that transpose operation is a reshape operation.
    const auto lastTransposeIn = op.getInput();
    auto maybeReshapeOp = lastTransposeIn.getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::ReshapeOp, IE::AffineReshapeOp, IE::SqueezeOp, IE::UnsqueezeOp>(maybeReshapeOp)) {
        return false;
    }

    // Now, find out whether this reshape operation has another transpose as an input.
    const auto reshapeIn = maybeReshapeOp->getOperand(0);
    auto firstTranspose = reshapeIn.getDefiningOp<IE::TransposeOp>();
    if (firstTranspose == nullptr) {
        return false;
    }

    // Only trivial reshapes can be collapsed.
    // Trivial means that all dimensions larger than 1 preserve order.
    // Examples:
    // 1x1x28x70 -> Reshape -> 1x28x70 -- trivial
    // 1x28x70x1 -> Reshape -> 1x28x70 -- trivial
    // 1x28x1x70 -> Reshape -> 1x28x70 -- trivial
    // 1x28x1x70 -> Reshape -> 1x70x28 -- non-trivial, since the order is not preserved.
    if (!isTrivialTransformation(getShape(maybeReshapeOp->getOperand(0)), getShape(maybeReshapeOp->getResult(0)))) {
        return false;
    }

    // Check that the second transpose is the inverse of the first one.
    if (!isTrivialTransformation(getShape(firstTranspose.getInput()), getShape(op.getOutput()))) {
        return false;
    }

    return true;
}

// Replaces chain of Transpose -> Reshape -> Transpose with single Reshape.
// Checks whether `Reshape` operation between two `Transpose` operations gathers any dimensions.
// If so, it checks whether the second transposition cancels the effect of the first one.
// In such cases, the whole subgraph can be replaced with a single reshape.
mlir::LogicalResult CollapseMutualTransposes::matchAndRewrite(IE::TransposeOp transposeOp,
                                                              mlir::PatternRewriter& rewriter) const {
    if (!canBeCollapsed(transposeOp)) {
        return mlir::failure();
    }

    const auto lastTransposeIn = transposeOp.getInput();
    auto maybeReshapeOp = lastTransposeIn.getDefiningOp();
    const auto reshapeIn = maybeReshapeOp->getOperand(0);
    auto firstTranspose = reshapeIn.getDefiningOp<IE::TransposeOp>();
    const auto firstTransposeIn = firstTranspose.getInput();

    const auto shape = mlir::cast<NDTypeInterface>(transposeOp.getOutput().getType()).getShape();
    const auto newShape = to_small_vector(shape);
    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    auto newOp =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(transposeOp, firstTransposeIn, nullptr, false, newShapeAttr);
    extendOpLoc(newOp, "as_reshape");

    return mlir::success();
}

//
// ConvertTrivialTransposeToReshape
//

class ConvertTrivialTransposeToReshape final : public mlir::OpRewritePattern<IE::TransposeOp> {
public:
    using mlir::OpRewritePattern<IE::TransposeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposeOp transposeOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertTrivialTransposeToReshape::matchAndRewrite(IE::TransposeOp transposeOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    if (!transposeOp.getOrderValue().has_value()) {
        return mlir::failure();
    }

    const auto inOrder = DimsOrder::fromValue(transposeOp.getInput());
    const auto inShape = getShape(transposeOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto perm = transposeOp.getOrderValue().value();

    if (!isTrivialPermute(inMemShape, perm)) {
        return mlir::failure();
    }

    const auto outputShape = mlir::cast<mlir::ShapedType>(transposeOp.getOutput().getType()).getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);
    auto newOp = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(transposeOp, transposeOp.getInput(), nullptr, false,
                                                            outputShapeAttr);
    extendOpLoc(newOp, "as_reshape");

    return mlir::success();
}

}  // namespace

void vpux::IE::TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
    patterns.add<FuseTransposes>(context);
    patterns.add<ConvertTrivialTransposeToReshape>(context);
    patterns.add<CollapseMutualTransposes>(context);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::TransposeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();

    if (getInput().getType() == getOutput().getType() && getOrderValue().has_value()) {
        const auto inputRank = static_cast<uint32_t>(mlir::cast<mlir::ShapedType>(getInput().getType()).getRank());
        const auto idMap = mlir::AffineMap::getMultiDimIdentityMap(inputRank, getContext());
        const auto orderMap = getOrderValue().value();
        if (idMap == orderMap) {
            return getInput();
        }
    }

    if (const auto cst = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        if (getOrderValue().has_value()) {
            const auto orderAttr = DimsOrder::fromAffineMap(getOrderValue().value());
            return static_cast<Const::ContentAttr>(cst).transform().transpose(orderAttr).get();
        }
    }

    return nullptr;
}

mlir::LogicalResult vpux::IE::TransposeOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                             mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    SmallVector<mlir::OpFoldResult> shapes;
    const auto loc = getLoc();
    const auto inputShapedType = mlir::cast<mlir::ShapedType>(getInput().getType());
    const auto outputShapedType = mlir::cast<mlir::ShapedType>(getOutput().getType());
    SmallVector<uint64_t> order{};
    if (::getOrder(*this, order, loc).failed()) {
        return mlir::failure();
    }
    for (const auto& dimIdx : irange(outputShapedType.getRank())) {
        if (outputShapedType.isDynamicDim(dimIdx)) {
            // Dynamic dimension: return mlir::Value according to permutation.
            auto dimLoc = appendLoc(loc, llvm::StringLiteral("dim_{0}"), dimIdx);
            auto index = builder.create<mlir::arith::ConstantIndexOp>(appendLoc(dimLoc, "const_index"), order[dimIdx]);
            mlir::OpFoldResult dimOp = builder.createOrFold<mlir::tensor::DimOp>(dimLoc, getInput(), index);
            shapes.push_back(mlir::getValueOrCreateConstantIndexOp(builder, appendLoc(loc, "const_index"), dimOp));
        } else {
            // Static dimension: return mlir::IntegerAttr.
            shapes.push_back(builder.getIndexAttr(inputShapedType.getDimSize(dimIdx)));
        }
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
    return mlir::success();
}

bool vpux::IE::TransposeOp::requiresStaticShape() {
    return config::hasEnableExtraStaticShapeOps(getModuleOp(this->getOperation()));
}
