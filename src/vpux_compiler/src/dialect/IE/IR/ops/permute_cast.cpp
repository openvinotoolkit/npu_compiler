///
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::PermuteCastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PermuteCastOpAdaptor permuteCast(operands, attrs, prop);
    if (mlir::failed(permuteCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inOrder = DimsOrder::fromValue(permuteCast.getInput());
    const auto inShape = getShape(permuteCast.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    if (!isTrivialPermute(inMemShape, permuteCast.getMemPerm())) {
        return errorAt(loc, "Operation represents non trivial permutation");
    }

    inferPermuteReturnTypeComponents(permuteCast.getInput(), permuteCast.getMemPerm(), permuteCast.getDstOrder(),
                                     inferredReturnShapes, true);

    return mlir::success();
}

namespace {

//
// FusePermuteCasts
//

class FusePermuteCasts final : public mlir::OpRewritePattern<IE::PermuteCastOp> {
public:
    using mlir::OpRewritePattern<IE::PermuteCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp permuteCastOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FusePermuteCasts::matchAndRewrite(IE::PermuteCastOp permuteCastOp,
                                                      mlir::PatternRewriter& rewriter) const {
    return fusePermutations<IE::PermuteCastOp, IE::PermuteCastOp>(permuteCastOp, rewriter);
}

//
// FuseMemPermAndPermCast
//

// MemPermute -> PermuteCast ===> MemPermute

class FuseMemPermAndPermCast final : public mlir::OpRewritePattern<IE::PermuteCastOp> {
public:
    using mlir::OpRewritePattern<IE::PermuteCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteCastOp permuteCastOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseMemPermAndPermCast::matchAndRewrite(IE::PermuteCastOp permuteCastOp,
                                                            mlir::PatternRewriter& rewriter) const {
    return fusePermutations<IE::MemPermuteOp, IE::PermuteCastOp>(permuteCastOp, rewriter);
}

}  // namespace

void vpux::IE::PermuteCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<FusePermuteCasts>(context);
    patterns.add<FuseMemPermAndPermCast>(context);
}

mlir::OpFoldResult vpux::IE::PermuteCastOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType() && getMemPerm().isIdentity()) {
        return getInput();
    }

    if (auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getInput())) {
        // PermuteCastOp ensures that it is always a trivial permutation. That's why we can just add MemPermuteAttr
        // which will not perform any data movements.
        auto result =
                attr.transform()
                        .memPermute(DimsOrder::fromAffineMap(getDstOrder()), DimsOrder::fromAffineMap(getMemPerm()))
                        .get();
        return result;
    }

    return nullptr;
}
