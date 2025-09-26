//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::DynamicReshapeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicReshapeOpAdaptor reshape(operands, attrs, prop);
    if (mlir::failed(reshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(reshape.getOutputShape());
    const auto outBounds = parseIntArrayAttr<int64_t>(reshape.getOutputBoundsAttr());
    const auto inType = mlir::cast<mlir::RankedTensorType>(reshape.getInput().getType());

    const auto outDesc = vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), vpux::getMemorySpace(inType),
                                             BoundsRef(outBounds));

    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);
    return mlir::success();
}

mlir::LogicalResult vpux::IE::DynamicReshapeOp::verify() {
    if (!IE::hasDynamicTensors(getOperation())) {
        return errorAt(getLoc(), "Operation must have dynamic tensors");
    }

    return mlir::success();
}

//
// FuseDynamicReshapes
//

namespace {

// The FuseDynamicReshapes pass optimizes MLIR by fusing consecutive DynamicReshapeOp operations into a single
// operation. It identifies sequences of reshapes, extracts dynamic dimensions, constructs new shape inputs, and
// replaces the original sequence with a fused operation. This reduces the number of operations, simplifies the
// computation graph, and improves performance by minimizing overhead and memory usage.
class FuseDynamicReshapes final : public mlir::OpRewritePattern<IE::DynamicReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::DynamicReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::DynamicReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

SmallVector<mlir::Value> constructNewShapeInputs(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                                 mlir::Value prevShape, mlir::Value origShape,
                                                 ArrayRef<int64_t> origOutputShapeParsed,
                                                 ArrayRef<int64_t> prevOutputShapeParsed) {
    SmallVector<mlir::Value> dynamicDims;
    SmallVector<mlir::Value> newShapeInputs;

    auto elemType = mlir::cast<vpux::NDTypeInterface>(prevShape.getType()).getElementType();
    auto shapeType = mlir::RankedTensorType::get({1}, elemType);

    // Extract dynamic dimensions from the previous shape
    for (int64_t i = 0; i < static_cast<int64_t>(prevOutputShapeParsed.size()); i++) {
        if (prevOutputShapeParsed[i] == mlir::ShapedType::kDynamic) {
            auto sliceOp = rewriter.create<IE::SliceOp>(appendLoc(loc, "_slice_dyn_prev_{0}", i), prevShape,
                                                        rewriter.getI64ArrayAttr({i}), rewriter.getI64ArrayAttr({1}));
            dynamicDims.push_back(sliceOp.getResult());
        }
    }

    // Construct new shape inputs by integrating static and dynamic dimensions.
    // The process involves iterating over the original output shape and evaluating each dimension:
    // - For static dimensions, a constant value is created.
    // - Dynamic dimensions are sourced from the previous shape's dynamic values,
    //   or from the original shape if the count of dynamic dimensions differs.
    // This approach enables the creation of a new shape that merges static dimensions from the original
    // shape with dynamic dimensions from the previous shape, effectively fusing the reshapes.
    for (int64_t i = 0, j = 0; i < static_cast<int64_t>(origOutputShapeParsed.size()); i++) {
        if (origOutputShapeParsed[i] != mlir::ShapedType::kDynamic) {
            mlir::Value constOp;
            if (elemType.isInteger(32)) {
                int32_t constValue = static_cast<int32_t>(origOutputShapeParsed[i]);
                constOp = Const::createConst(rewriter, appendLoc(loc, "_dim_{0}", i), shapeType,
                                             ArrayRef<int32_t>(constValue));
            } else if (elemType.isInteger(64)) {
                int64_t constValue = origOutputShapeParsed[i];
                constOp = Const::createConst(rewriter, appendLoc(loc, "_dim_{0}", i), shapeType,
                                             ArrayRef<int64_t>(constValue));
            } else {
                VPUX_THROW("Invalid element type {0}", elemType);
            }
            newShapeInputs.push_back(constOp);
        } else {
            if (j < static_cast<int64_t>(dynamicDims.size())) {
                newShapeInputs.push_back(dynamicDims[j]);
                j++;
            } else {
                auto sliceOp =
                        rewriter.create<IE::SliceOp>(appendLoc(loc, "_slice_dyn_orig_{0}", i), origShape,
                                                     rewriter.getI64ArrayAttr({i}), rewriter.getI64ArrayAttr({1}));
                newShapeInputs.push_back(sliceOp.getResult());
            }
        }
    }
    return newShapeInputs;
}

size_t checkUses(IE::DynamicReshapeOp op) {
    auto uses = op->getResult(0).getUses();
    return static_cast<size_t>(std::distance(uses.begin(), uses.end()));
}

bool hasSingleUser(IE::DynamicReshapeOp op) {
    return checkUses(op) == 1;
}

bool hasMoreUsesAndValid(IE::DynamicReshapeOp op) {
    if (checkUses(op) < 2) {
        return false;
    }

    auto isShapeConcat = mlir::isa<IE::ConcatOp>(op.getShape().getDefiningOp());
    auto usedInShapeOf =
            std::all_of(op->getResult(0).getUses().begin(), op->getResult(0).getUses().end(), [&](const auto& use) {
                return mlir::isa<IE::ShapeOfOp>(use.getOwner()) || mlir::isa<IE::DynamicReshapeOp>(use.getOwner());
            });

    return isShapeConcat && usedInShapeOf;
}

bool isConstantShape(mlir::Value shape) {
    return mlir::isa<Const::DeclareOp>(shape.getDefiningOp());
}

bool shapesAreEqual(mlir::Value shape1, mlir::Value shape2) {
    auto extractShape = [](mlir::Value shape) -> std::optional<llvm::SmallVector<int64_t>> {
        if (auto shapeDecl = shape.getDefiningOp<Const::DeclareOp>()) {
            auto content = shapeDecl.getContent();
            return to_small_vector(content.getValues<int64_t>());
        }
        return std::nullopt;
    };

    auto shapeVec1 = extractShape(shape1);
    auto shapeVec2 = extractShape(shape2);

    // Check if both shapes were successfully extracted and compare them
    return shapeVec1 && shapeVec2 && (*shapeVec1 == *shapeVec2);
}

void replaceWithNewDynamicReshape(mlir::PatternRewriter& rewriter, IE::DynamicReshapeOp origOp, mlir::Value input,
                                  mlir::Value shape, bool onlySetShape) {
    rewriter.replaceOpWithNewOp<IE::DynamicReshapeOp>(origOp, input, shape, origOp.getOutputShapeAttr(),
                                                      origOp.getOutputBoundsAttr(), onlySetShape);
}

mlir::LogicalResult FuseDynamicReshapes::matchAndRewrite(IE::DynamicReshapeOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp<IE::DynamicReshapeOp>();
    if (prevOp == nullptr) {
        return mlir::failure();
    }

    // Verify that the result of the preceding DynamicReshapeOp has a single user. This condition must be met to allow
    // the fusion of consecutive DynamicReshapeOp operations into a single operation.
    if (!hasSingleUser(prevOp) && !hasMoreUsesAndValid(prevOp)) {
        return mlir::failure();
    }

    if (hasMoreUsesAndValid(prevOp)) {
        // Optimization: Replace the ShapeOf operation with the operation used as the destination shape.
        // This transformation simplifies the computation graph by directly connecting the dstShape operation
        // to the subsequent DynamicReshape, eliminating the intermediate ShapeOf operation.
        //
        // Original Structure:
        //                   dstShape
        //                      |
        //                DynamicReshape
        //             |                  |
        //         ShapeOf -> .. ->  DynamicReshape
        //
        // Optimized Result:
        //         dstShape -> .. ->  DynamicReshape
        //
        // This refactoring reduces the number of operations and streamlines the data flow, enhancing performance (in
        // theory).
        auto uses = prevOp->getResult(0).getUses();
        for (auto& use : uses) {
            auto userOpShapeOf = mlir::dyn_cast<IE::ShapeOfOp>(use.getOwner());
            auto dstShape = prevOp.getShape().getDefiningOp();
            if (userOpShapeOf != nullptr) {
                rewriter.replaceOp(userOpShapeOf, dstShape->getResults());
            }
        }
    }

    // Determine the value of the onlySetShape attribute for the new reshape operation
    bool onlySetShape = origOp.getOnlySetShape() && prevOp.getOnlySetShape();

    // Handle constant shape case, when origOp.getShape() and prevOp.getShape() are equal.
    if (isConstantShape(origOp.getShape()) && isConstantShape(prevOp.getShape())) {
        if (shapesAreEqual(origOp.getShape(), prevOp.getShape())) {
            replaceWithNewDynamicReshape(rewriter, origOp, prevOp->getOperand(0), origOp.getShape(), onlySetShape);
            return mlir::success();
        }
    }

    auto origOutputShapeParsed = parseIntArrayAttr<int64_t>(origOp.getOutputShape());
    auto prevOutputShapeParsed = parseIntArrayAttr<int64_t>(prevOp.getOutputShape());

    auto newShapeInputs = constructNewShapeInputs(rewriter, origOp.getLoc(), prevOp.getShape(), origOp.getShape(),
                                                  origOutputShapeParsed, prevOutputShapeParsed);

    auto newShape = rewriter.create<IE::ConcatOp>(appendLoc(origOp.getLoc(), "_fused_shape"), newShapeInputs,
                                                  getIntAttr(getContext(), 0));

    replaceWithNewDynamicReshape(rewriter, origOp, prevOp->getOperand(0), newShape.getResult(), onlySetShape);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::DynamicReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                             mlir::MLIRContext* ctx) {
    patterns.add<FuseDynamicReshapes>(ctx);
}
