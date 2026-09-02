//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/dense_map.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEMEMPERMUTEBEFOREOP
#define GEN_PASS_DEF_PROPAGATEMEMPERMUTEBEFOREOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

SmallVector<int64_t> deduceInAxis(SmallVector<SmallVector<int64_t>> dimMapping, int64_t outAxis) {
    SmallVector<int64_t> inAxis;
    for (size_t inIdx = 0; inIdx < dimMapping.size(); inIdx++) {
        auto mappedDim = dimMapping[inIdx];

        for (size_t mapId = 0; mapId < mappedDim.size(); mapId++) {
            auto outIdx = mappedDim[mapId];
            if (outIdx == outAxis) {
                inAxis.push_back(checked_cast<int64_t>(inIdx));
            }
        }
    }
    return inAxis;
}

bool isSplitOutAxis(SmallVector<SmallVector<int64_t>> dimMapping, int64_t outAxis) {
    for (size_t inIdx = 0; inIdx < dimMapping.size(); inIdx++) {
        auto mappedDim = dimMapping[inIdx];
        // mappedDim.size() > 1 indicates a split of input axis
        if (mappedDim.size() > 1) {
            for (size_t mapId = 0; mapId < mappedDim.size(); mapId++) {
                auto outIdx = mappedDim[mapId];
                if (outIdx == outAxis) {
                    return true;
                }
            }
        }
    }
    return false;
}

mlir::AffineMap calculateNewPermutation(SmallVector<SmallVector<int64_t>>& dimMapping, ArrayRef<int64_t> origPermVec,
                                        const DimsOrder& affineReshapeInOrder, const DimsOrder& affineReshapeOutOrder,
                                        vpux::MemShapeRef inMemShapeRef, vpux::MemShapeRef outMemShapeRef, Logger log,
                                        mlir::MLIRContext* ctx) {
    SmallVector<int64_t> inAxesVec;
    const auto inShape = affineReshapeInOrder.toLogicalOrder(inMemShapeRef).raw();

    for (size_t i = 0; i < origPermVec.size(); i++) {
        // Step 1.1: Map original permutation axes to out dimsMapping axes.
        auto outIdx = affineReshapeOutOrder.toDim(MemDim(origPermVec[i])).ind();
        // Step 1.2: Deduce input dimsMapping axes from out dimsMapping axes.
        auto inIdx = deduceInAxis(dimMapping, checked_cast<int64_t>(outIdx));
        VPUX_THROW_WHEN(inIdx.empty(), "Unexpected dimMapping {0} and input dim index {1}", dimMapping, inIdx);
        // Step 1.3: Save input dimsMapping axes.
        // Ignore the dim with shape 1 which is from split axes, and the split axes is non-trivial.
        if (isSplitOutAxis(dimMapping, outIdx) && outMemShapeRef[MemDim(origPermVec[i])] == 1 &&
            inShape[inIdx[0]] != 1) {
            continue;
        }

        for (auto idx : inIdx) {
            auto isAxisAlreadySaved = llvm::find(inAxesVec, idx) != inAxesVec.end();
            if (!isAxisAlreadySaved) {
                inAxesVec.push_back(idx);
            }
        }
    }

    SmallVector<unsigned> newPermVec;
    for (size_t idx = 0; idx < inAxesVec.size(); idx++) {
        // Step 2.1: Map saved input dimsMapping axes to permutation axes.
        auto memDim = affineReshapeInOrder.toMemDim(Dim(inAxesVec[idx]));
        // Step 2.2: Save the permutation axes as new permutation.
        newPermVec.push_back(checked_cast<unsigned>(memDim.ind()));
    }

    log.trace("Got newPermVec {0} converted from inAxesVec {1} with order {2}", newPermVec, inAxesVec,
              affineReshapeInOrder);
    VPUX_THROW_UNLESS(newPermVec.size() == affineReshapeInOrder.numDims(),
                      "New permutation and output dimensions do not match.");

    return mlir::AffineMap::getPermutationMap(ArrayRef(newPermVec), ctx);
}

// Create a new sub graph in below:
//
//      PermuteCastOp
//           |
//      MemPermuteOp
//           |
//       ReshapeOp
//           |
//      PermuteCastOp
//
// to replace the original pattern:
//
//      AffineReshapeOp
//           |
//      MemPermuteOp

mlir::LogicalResult replaceWithNewSubGraph(mlir::Value affineReshape, mlir::Value memPermute, mlir::AffineMap newPerm,
                                           mlir::PatternRewriter& rewriter, Logger log) {
    const auto ctx = rewriter.getContext();
    auto affineReshapeOp = affineReshape.getDefiningOp<IE::AffineReshapeOp>();
    VPUX_THROW_WHEN(affineReshapeOp == nullptr, "Not an AffineReshape operation");
    auto permuteOp = memPermute.getDefiningOp();
    if (!mlir::isa<IE::MemPermuteOp, IE::PermuteQuantizeOp>(permuteOp)) {
        VPUX_THROW("Not a MemPermute or PermuteQuantize operation");
    }

    const auto affineInShape = getShape(affineReshapeOp.getInput());

    // Cast to canonical order for convenience
    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<unsigned>(affineInShape.size()), ctx);
    auto inputCast = rewriter.create<IE::PermuteCastOp>(permuteOp->getLoc(), affineReshapeOp.getInput(), identityMap,
                                                        identityMap);

    // Create new permute
    const auto newPermAttr = mlir::AffineMapAttr::get(newPerm);
    const auto identityOrderAttr = mlir::AffineMapAttr::get(identityMap);

    auto newPermute =
            rewriter.create<IE::MemPermuteOp>(takeOpLoc(permuteOp, "{0}", DimsOrder::fromAffineMap(identityMap)),
                                              inputCast.getOutput(), identityOrderAttr, newPermAttr);

    // Reshape to original output shape
    auto outputType = mlir::cast<vpux::NDTypeInterface>(permuteOp->getResult(0).getType());
    auto outputShape = outputType.getMemShape();
    auto outputShapeAttr = getIntArrayAttr(ctx, outputShape);
    const auto reassociationMap =
            vpux::IE::getReassociationMap(getShape(newPermute.getOutput()).raw(), outputShape.raw());
    if (mlir::failed(reassociationMap)) {
        log.nest().trace("getReassociationMap failed for op {0}", affineReshapeOp.getLoc());
        newPermute->dropAllReferences();
        rewriter.eraseOp(newPermute);
        inputCast->dropAllReferences();
        rewriter.eraseOp(inputCast);
        return mlir::failure();
    }

    const auto reassociationMapAttr = getIntArrayOfArray(ctx, reassociationMap.value());
    auto outputReshape = rewriter.create<IE::AffineReshapeOp>(affineReshapeOp.getLoc(), newPermute.getOutput(),
                                                              reassociationMapAttr, outputShapeAttr);
    inferReturnTypes(outputReshape, InferShapedTypeMode::ELEM_TYPE);

    // Set destination order
    mlir::AffineMap dstOrder;
    if (mlir::isa<IE::MemPermuteOp>(permuteOp)) {
        auto memPermuteOp = mlir::dyn_cast<IE::MemPermuteOp>(permuteOp);
        dstOrder = memPermuteOp.getDstOrder();
    } else if (mlir::isa<IE::PermuteQuantizeOp>(permuteOp)) {
        auto permuteQuantizeOp = mlir::dyn_cast<IE::PermuteQuantizeOp>(permuteOp);
        dstOrder = permuteQuantizeOp.getDstOrder();
    } else {
        VPUX_THROW("Not a MemPermute or PermuteQuantize operation");
    }

    auto newPermuteCast = rewriter.createOrFold<IE::PermuteCastOp>(
            permuteOp->getLoc(), outputReshape->getResult(0), dstOrder,
            mlir::AffineMap::getMultiDimIdentityMap(checked_cast<unsigned>(outputShape.size()), ctx));

    // Replace with new sub graph
    memPermute.replaceAllUsesWith(newPermuteCast);
    rewriter.eraseOp(permuteOp);
    rewriter.eraseOp(affineReshapeOp);
    return mlir::success();
}

//
// OptimizeMemPermute
//

class OptimizeMemPermute final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    OptimizeMemPermute(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _log(log) {
        this->setDebugName("OptimizeMemPermute");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Move MemPermuteOp to the front of AffineReshapeOp.
//
// This conversion can be performed in case the permutation is not breaking split dims.
//
// e.g.
//
// Original pattern: Reshape is before permutation.
//      A x B x C x D   (input mem shape)
//          |  /   /|         |
//          | /   / |   affinReshape  (dim_mapping [0], [1], [1], [2, 3]:
//          |/   /  |         |        means B & C are merged into B', D is split to C' & D')
//      A'x B'x C'x D'  (temp mem shape)
//            |               |
//            |          MemPermute    (perm [0, 2, 3, 1])
//            |               |
//      A'x C'x D'x B'  (output mem shape)
//
// After the pass: Permutation is before reshape.
//
//      A x B x C x D   (input mem shape)
//            |               |
//            |          new MemPermute    (new perm [0, 3, 1, 2])
//            |               |
//      A x D x B x C   (temp mem shape)
//          |\   \  |         |
//          | \   \ |     Reshape
//          |  \   \|         |
//      A'x C'x D'x B'  (output mem shape)
//

mlir::LogicalResult OptimizeMemPermute::matchAndRewrite(IE::MemPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    auto affineReshape = origOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (affineReshape == nullptr || !affineReshape->hasOneUse()) {
        return mlir::failure();
    }

    // Check that tensor rank is 4, otherwise compilation fails in later passes
    auto inType = mlir::cast<vpux::NDTypeInterface>(affineReshape.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(affineReshape.getOutput().getType());
    auto inRank = inType.getRank();
    auto outRank = outType.getRank();
    if (inRank != 4 || outRank != 4) {
        return mlir::failure();
    }

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshape.getDimMapping());
    const auto originPerm = DimsOrder::fromAffineMap(origOp.getMemPerm());
    const auto originPermVec = to_small_vector(originPerm.toPermutation() | transformed([](Dim dim) {
                                                   return checked_cast<int64_t>(dim.ind());
                                               }));
    const auto origPermRef = ArrayRef(originPermVec);
    const auto inMemShape = inType.getMemShape();
    const auto outMemShape = outType.getMemShape();

    if (!areReshapedAxesPermutedIntegratedly(dimMapping, origPermRef, outType.getDimsOrder(), outMemShape)) {
        const auto extendReassociationMap =
                vpux::IE::getReassociationMapExtension(inType.getShape().raw(), outType.getShape().raw());
        if (mlir::failed(extendReassociationMap)) {
            return matchFailed(rewriter, origOp, "Failed to get extension map");
        }

        if (!areReshapedAxesPermutedIntegratedly(extendReassociationMap.value(), origPermRef, outType.getDimsOrder(),
                                                 outMemShape)) {
            return matchFailed(rewriter, origOp, "[{0}]: Swap the split axes", getDebugName());
        }

        dimMapping = extendReassociationMap.value();
        _log.trace("The extended ReassociationMap {0} is used", dimMapping);
    }

    _log.trace("[{0}]: Rewriting {1}", getDebugName(), origOp->getLoc());

    auto newPerm = calculateNewPermutation(dimMapping, origPermRef, inType.getDimsOrder(), outType.getDimsOrder(),
                                           inMemShape, outMemShape, _log, ctx);

    const mlir::OperationName affineReshapeName = affineReshape->getName();
    const mlir::Location affineReshapeLoc = affineReshape->getLoc();
    auto result = replaceWithNewSubGraph(affineReshape.getOutput(), origOp.getOutput(), newPerm, rewriter, _log);
    if (result.succeeded()) {
        _log.nest().trace("[{0}]: Replaced {1} at {2} with new sub graph: newPerm '{3}'", getDebugName(),
                          affineReshapeName, affineReshapeLoc, newPerm);
        return mlir::success();
    } else {
        _log.nest().trace("[{0}]: Failed to replace {1} at {2}", getDebugName(), affineReshapeName, affineReshapeLoc);
        return mlir::failure();
    }
}

//
// PropagatePermuteQuantize
//
// Catch the pattern in below:
//
//      MemPermuteOp
//            |
//      AffineReshape
//            |
//     PermuteQuantizeOp
//            |
//
// If PermuteQuantizeOp only performs permutation, propagate permuteQuantize through AffineReshape
// so that the permutes can be folded or converted into PermuteCast.
//

class PropagatePermuteQuantize final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    PropagatePermuteQuantize(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx, benefit), _log(log) {
        this->setDebugName("PropagatePermuteQuantize");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult movePermuteQuantize(mlir::Value affineReshape, mlir::Value permuteQuantize,
                                            mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::LogicalResult PropagatePermuteQuantize::movePermuteQuantize(mlir::Value affineReshape,
                                                                  mlir::Value permuteQuantize,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();
    auto affineReshapeOp = affineReshape.getDefiningOp<IE::AffineReshapeOp>();
    VPUX_THROW_WHEN(affineReshapeOp == nullptr, "Not an AffineReshape operation");
    auto permuteQuantizeOp = mlir::dyn_cast_or_null<IE::PermuteQuantizeOp>(permuteQuantize.getDefiningOp());
    VPUX_THROW_WHEN(permuteQuantizeOp == nullptr, "Not a PermuteQuantize operation");

    auto isSupportedNCEPermuteAfterPropagation = [](IE::AffineReshapeOp affineReshapeOp,
                                                    IE::PermuteQuantizeOp permuteQuantizeOp) {
        const auto reshapeInType = mlir::cast<NDTypeInterface>(affineReshapeOp.getInput().getType());
        const auto reshapeOutType = mlir::cast<NDTypeInterface>(affineReshapeOp.getOutput().getType());
        const auto reshapeInOrder = reshapeInType.getDimsOrder();
        const auto reshapeOutOrder = reshapeOutType.getDimsOrder();
        if (reshapeInOrder != reshapeOutOrder) {
            return false;
        }

        const auto newPermuteQuantizeInShape = reshapeInType.getShape();
        mlir::SmallVector<mlir::ShapedTypeComponents> inferredReturnShapes;
        inferPermuteReturnTypeComponents(affineReshapeOp->getOperand(0), permuteQuantizeOp.getMemPerm(),
                                         permuteQuantizeOp.getDstOrder(), inferredReturnShapes, false);
        VPUX_THROW_WHEN(inferredReturnShapes.size() != 1, "Should be 1 but got {0}", inferredReturnShapes.size());
        const auto newPermuteQuantizeOutShape = ShapeRef(inferredReturnShapes.front().getDims());

        const auto alignment = VPU::NCEInvariant::getAlignment(reshapeInType.getElementType());
        return IE::checkNCEPermuteShapeCompatibility(newPermuteQuantizeInShape, newPermuteQuantizeOutShape, alignment);
    };
    if (!isSupportedNCEPermuteAfterPropagation(affineReshapeOp, permuteQuantizeOp)) {
        _log.nest().warning("Not supported by NCEPermute after propagation");
        return mlir::failure();
    }

    // Create new PermuteQuantizeOp
    mlir::IRMapping mapper;
    mapper.map(permuteQuantizeOp.getInput(), affineReshapeOp.getInput());
    auto newOp = rewriter.clone(*permuteQuantizeOp, mapper);
    inferReturnTypes(newOp, InferShapedTypeMode::ALL);

    // Reshape to original output shape
    auto outputType = mlir::cast<vpux::NDTypeInterface>(permuteQuantizeOp.getOutput().getType());
    auto outputShape = outputType.getShape();
    auto outputShapeAttr = getIntArrayAttr(ctx, outputShape);
    auto outputReshape = rewriter.create<IE::ShapeCastOp>(
            affineReshapeOp.getLoc(),
            mlir::cast<vpux::NDTypeInterface>(newOp->getResult(0).getType()).changeShape(outputShape),
            newOp->getResult(0), outputShapeAttr);
    permuteQuantize.replaceAllUsesWith(outputReshape.getResult());
    rewriter.eraseOp(permuteQuantizeOp);
    rewriter.eraseOp(affineReshapeOp);
    return mlir::success();
}

mlir::LogicalResult PropagatePermuteQuantize::matchAndRewrite(IE::PermuteQuantizeOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();

    const auto permuteQuantizeInElemType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
    const auto permuteQuantizeOutElemType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
    if (permuteQuantizeInElemType != permuteQuantizeOutElemType) {
        return mlir::failure();
    }

    // Check PermuteQuantize pads attributes.
    const auto padStart = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());

    const auto nonZeroPadStart = llvm::any_of(padStart, [](auto pad) {
        return pad > 0;
    });
    const auto nonZeroPadEnd = llvm::any_of(padEnd, [](auto pad) {
        return pad > 0;
    });
    if (nonZeroPadStart || nonZeroPadEnd) {
        return mlir::failure();
    }

    auto affineReshape = origOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (affineReshape == nullptr || !affineReshape->hasOneUse()) {
        return mlir::failure();
    }

    // Check that tensor rank is 4, otherwise compilation fails in later passes
    const int64_t SUPPORTED_RANK = 4;
    auto inType = mlir::cast<vpux::NDTypeInterface>(affineReshape.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(affineReshape.getOutput().getType());
    auto inRank = inType.getRank();
    auto outRank = outType.getRank();
    if (inRank != SUPPORTED_RANK || outRank != SUPPORTED_RANK) {
        return mlir::failure();
    }

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshape.getDimMapping());
    const auto originPerm = DimsOrder::fromAffineMap(origOp.getMemPerm());
    const auto originPermVec = to_small_vector(originPerm.toPermutation() | transformed([](Dim dim) {
                                                   return checked_cast<int64_t>(dim.ind());
                                               }));
    const auto origPermRef = ArrayRef(originPermVec);
    const auto inMemShape = inType.getMemShape();
    const auto outMemShape = outType.getMemShape();
    if (!areReshapedAxesPermutedIntegratedly(dimMapping, origPermRef, outType.getDimsOrder(), outMemShape)) {
        return matchFailed(rewriter, origOp, "[{0}]: Split axes are broken", getDebugName());
    }

    auto newPerm = calculateNewPermutation(dimMapping, origPermRef, inType.getDimsOrder(), outType.getDimsOrder(),
                                           inMemShape, outMemShape, _log, ctx);

    _log.trace("[{0}]: Rewriting {1}", getDebugName(), origOp->getLoc());

    auto memPermute = affineReshape.getInput().getDefiningOp<IE::MemPermuteOp>();
    if (memPermute != nullptr && memPermute->hasOneUse()) {
        // Create subgraph with MemPermute to fuse or even eliminate permutation
        return replaceWithNewSubGraph(affineReshape.getOutput(), origOp.getOutput(), newPerm, rewriter, _log);
    }

    // We can move PermuteQuantize through AffineReshape when newPerm and origPerm have the same merged
    // permutation.
    // Otherwise there would a failure when lowering IE.PermuteQuantize to VPU.NCEPermute with newPerm, because newPerm
    // might be not a valid permutation for VPU.NCEPermute
    auto newMergedPermAndShape = vpux::getMergedPermutationAndShape(inType, newPerm, SUPPORTED_RANK);
    auto origMergedPermAndShape = vpux::getMergedPermutationAndShape(inType, origOp.getMemPerm(), SUPPORTED_RANK);
    if (newMergedPermAndShape.first != origMergedPermAndShape.first) {
        _log.nest().trace("Can't move PermuteQuantize because merged permutation needs to be changed");
        return mlir::failure();
    }

    return movePermuteQuantize(affineReshape.getOutput(), origOp.getOutput(), rewriter);
}

//
// MoveThroughOpBase
//
// Catch the pattern in below:
//              Op
//              |
//  MemPermute / PermuteQuantize
//
// Move the MemPermute / PermuteQuantize before Op

template <class ConcreteOp>
class MoveThroughOpBase : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MoveThroughOpBase(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefit), _log(log) {
    }

    bool genericCheck(mlir::Operation* permuteOp) const;

private:
    mlir::LogicalResult matchAndRewrite(ConcreteOp concreteOp, mlir::PatternRewriter& rewriter) const final;

    virtual bool checkMemPermutePattern(mlir::Operation* permuteOp, mlir::PatternRewriter& rewriter) const = 0;

    virtual mlir::AffineMap getPermutationMap(mlir::Operation* permuteOp) const = 0;

    virtual mlir::Operation* createNewPermuteOp(mlir::Operation* permuteOp, mlir::Value newInput,
                                                mlir::AffineMap dstOrder, mlir::PatternRewriter& rewriter) const = 0;

private:
    Logger _log;
};

template <class ConcreteOp>
bool MoveThroughOpBase<ConcreteOp>::genericCheck(mlir::Operation* permuteOp) const {
    // Check pattern Op -> MemPermuteOp / PermuteQuantizeOp.
    auto op = permuteOp->getOperand(0).getDefiningOp();
    if (!mlir::isa_and_nonnull<ConcreteOp>(op)) {
        return false;
    }

    if (!op->hasOneUse()) {
        return false;
    }

    // ConcreteOp should not receive input from a BlockArgument
    const auto inOperands = op->getOperands();
    const auto hasBlockArgumentInput = std::any_of(inOperands.begin(), inOperands.end(), [](const auto input) {
        return mlir::dyn_cast_or_null<mlir::BlockArgument>(input);
    });
    if (hasBlockArgumentInput) {
        return false;
    }

    // The ConcreteOp must not have input or output quantized per axis
    auto inElemType = mlir::dyn_cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    auto outElemType = mlir::dyn_cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType();
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inElemType) ||
        mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outElemType)) {
        return false;
    }

    // E#127631: If MemPermute input is QuantizedType the storage type should not be sub byte quantization because this
    // would result in compilation error later
    const auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(inElemType);
    if (quantType != nullptr) {
        return !vpux::isSubByteType(quantType.getStorageType());
    }

    return true;
}

template <class ConcreteOp>
mlir::LogicalResult MoveThroughOpBase<ConcreteOp>::matchAndRewrite(ConcreteOp concreteOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    auto permuteOp = *concreteOp->getUsers().begin();
    if (!checkMemPermutePattern(permuteOp, rewriter)) {
        return mlir::failure();
    }

    auto memPerm = getPermutationMap(permuteOp);
    const auto originPerm = DimsOrder::fromAffineMap(memPerm);
    const auto originPermVec = to_small_vector(originPerm.toPermutation() | transformed([](Dim dim) {
                                                   return checked_cast<int64_t>(dim.ind());
                                               }));

    auto ctx = permuteOp->getContext();
    auto inOrder = DimsOrder::fromValue(permuteOp->getOperand(0)).toAffineMap(ctx);
    auto perm = memPerm.compose(inOrder);
    auto outOrder = DimsOrder::fromAffineMap(perm);

    auto operation = concreteOp->getResult(0).getDefiningOp();
    if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(operation)) {
        auto orderInfo = iface.getLayoutInfo();
        orderInfo.setInput(0, outOrder);
        iface.inferLayoutInfo(orderInfo);
        if (orderInfo.getInput(0) != outOrder || orderInfo.getOutput(0) != outOrder) {
            return mlir::failure();
        }
    }

    _log.trace("Got '{0}' at '{1}'", concreteOp->getName(), concreteOp->getLoc());

    // create new permute operation which keeps the shape unchanged and adjust the dst order only.
    SmallVector<mlir::Value> newInputs;
    DenseMap<mlir::Value, mlir::Operation*> operandsMap;
    for (auto& input : concreteOp->getOpOperands()) {
        mlir::Operation* newPermuteOp = nullptr;
        auto it = operandsMap.find(input.get());
        if (it == operandsMap.end()) {
            newPermuteOp = createNewPermuteOp(permuteOp, input.get(), perm, rewriter);
            auto newLoc = takeOpLoc(concreteOp, "input_{0}", input.getOperandNumber());
            newPermuteOp->setLoc(newLoc);
            operandsMap.insert({input.get(), newPermuteOp});
        } else {
            // Some operands may have the same input, re-use the Permute operation that has been created already
            newPermuteOp = it->second;
        }
        newInputs.push_back(newPermuteOp->getResult(0));
    }

    mlir::IRMapping mapper;
    mapper.map(concreteOp->getOperands(), newInputs);
    mlir::Operation* newOp = rewriter.clone(*concreteOp, mapper);
    auto newOutput = newOp->getResult(0);
    newOutput.setType(mlir::cast<vpux::NDTypeInterface>(concreteOp->getResult(0).getType()).changeDimsOrder(outOrder));

    auto origOrder = mlir::cast<vpux::NDTypeInterface>(permuteOp->getResult(0).getType()).getDimsOrder();
    auto newPermuteCast = rewriter.createOrFold<IE::PermuteCastOp>(
            permuteOp->getLoc(), newOp->getResult(0), origOrder.toAffineMap(ctx),
            mlir::AffineMap::getMultiDimIdentityMap(outOrder.numDims(), ctx));

    _log.nest().trace("Propagate Permute operation {0} before {1} at {2}", permuteOp->getLoc(), concreteOp->getName(),
                      concreteOp->getLoc());

    rewriter.replaceOp(permuteOp, newPermuteCast);
    // Prevent concreteOp from being added to the worklist again
    concreteOp->dropAllReferences();
    rewriter.eraseOp(concreteOp);
    return mlir::success();
}

//
// MoveMemPermuteThroughOp
//

template <class ConcreteOp>
class MoveMemPermuteThroughOp final : public MoveThroughOpBase<ConcreteOp> {
public:
    MoveMemPermuteThroughOp(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : MoveThroughOpBase<ConcreteOp>(ctx, benefit, log), _log(log) {
    }

    bool checkMemPermutePattern(mlir::Operation* permuteOp, mlir::PatternRewriter& rewriter) const override;

    mlir::AffineMap getPermutationMap(mlir::Operation* permuteOp) const override;

    mlir::Operation* createNewPermuteOp(mlir::Operation* permuteOp, mlir::Value newInput, mlir::AffineMap dstOrder,
                                        mlir::PatternRewriter& rewriter) const override;

    bool isPropagationBeneficialForConcatAndSlice(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

template <class ConcreteOp>
bool MoveMemPermuteThroughOp<ConcreteOp>::isPropagationBeneficialForConcatAndSlice(
        IE::MemPermuteOp permuteOp, mlir::PatternRewriter& rewriter) const {
    auto parentOp = permuteOp.getInput().getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::ConcatOp, IE::SliceOp, IE::StridedSliceOp>(parentOp)) {
        return false;
    }

    if (mlir::isa<IE::StridedSliceOp>(parentOp) && getShape(parentOp->getResult(0)).isDynamic()) {
        // Skip propagation through StridedSliceOp with dynamic output shapes to avoid functional test failures.
        _log.trace("StridedSliceOp has dynamic output shape");
        return false;
    }

    auto permuteInType = mlir::cast<vpux::NDTypeInterface>(permuteOp.getInput().getType());
    const auto permuteInMemShape = permuteInType.getMemShape();
    auto memPerm = permuteOp.getMemPerm();
    if (isTrivialPermute(permuteInMemShape, memPerm)) {
        return false;
    }

    // Benefit from stride DMA due to axis transition from lower to higher dimension
    // Example: Concat operation (NHWC, connected at C) -> Mempermute (NHWC to NCHW)
    // Propagating Permute does not change the total permutation data size but eliminates the need for stride DMAs
    const auto ctx = permuteOp.getContext();
    const auto inOrder = DimsOrder::fromValue(permuteOp.getInput());
    auto dstPerm = memPerm.compose(inOrder.toAffineMap(ctx));
    auto dstOrder = DimsOrder::fromAffineMap(dstPerm);
    auto srcOrder = permuteInType.getDimsOrder();
    auto isBeneficialStrideDMA = [&](ShapeRef inShape, ShapeRef outShape) {
        const int64_t CONTIGUOUS_BUFFER_SIZE_LIMITATION = 8;
        for (auto ioShape : zip(inShape, outShape) | indexed) {
            const auto inSize = std::get<0>(ioShape.value());
            const auto outSize = std::get<1>(ioShape.value());
            const auto dim = Dim(ioShape.index());
            // When the axis transitions from a higher to a lower dimension, the stride DMA becomes inefficient
            // However, if the contiguous buffer size is larger than 8 (an experimental value)
            // The efficiency of stride DMA improves and approaches the performance of non-stride DMA for larger sizes
            if (inSize != outSize && dstOrder.dimPos(dim) > srcOrder.dimPos(dim)) {
                const auto inMemShape = dstOrder.toMemoryOrder(inShape);
                auto contiguousBufferSize = std::min(inSize, outSize);
                contiguousBufferSize = std::accumulate(inMemShape.begin() + dstOrder.dimPos(dim) + 1, inMemShape.end(),
                                                       contiguousBufferSize, std::multiplies<int64_t>());
                if (contiguousBufferSize < CONTIGUOUS_BUFFER_SIZE_LIMITATION) {
                    return false;
                }
            }
        }
        return true;
    };

    const auto parentInShape = getShape(parentOp->getOperand(0));
    const auto parentOutShape = getShape(parentOp->getResult(0));
    const auto beneficialStrideDMA = isBeneficialStrideDMA(parentInShape, parentOutShape);

    auto isInputWithDuplicateSlice = [&](mlir::Value input) {
        IE::SliceOp firstSliceOp = nullptr;
        for (auto user : input.getUsers()) {
            firstSliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(user);
            if (firstSliceOp != nullptr) {
                break;
            }
        }

        if (firstSliceOp == nullptr) {
            return false;
        }

        auto sliceAxes = getSliceAxes(firstSliceOp);
        if (sliceAxes.size() != 1) {
            return false;
        }

        uint64_t firstSliceAxe = sliceAxes.front();
        int64_t sliceSize = 0;
        SmallVector<IE::MemPermuteOp> memPermuteOps;
        SmallVector<IE::MemPermuteOp> inputUserMemPermuteOps;

        for (auto user : input.getUsers()) {
            if (!mlir::isa_and_nonnull<IE::MemPermuteOp, IE::SliceOp>(user)) {
                return false;
            }

            auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(user);
            if (sliceOp == nullptr) {
                auto memPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(user);
                if (memPermuteOp != nullptr) {
                    memPermuteOps.push_back(memPermuteOp);
                    inputUserMemPermuteOps.push_back(memPermuteOp);
                }
                continue;
            }

            sliceAxes = getSliceAxes(sliceOp);
            if (sliceAxes.size() != 1 || firstSliceAxe != sliceAxes.front()) {
                return false;
            }

            const auto sliceShape = getShape(sliceOp.getResult()).raw();
            sliceSize += sliceShape[firstSliceAxe];

            if (!sliceOp.getResult().hasOneUse()) {
                continue;
            }

            if (auto memPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(*sliceOp.getResult().getUsers().begin())) {
                memPermuteOps.push_back(memPermuteOp);
            }
        }

        // If input has MemPermute user, and the Slice's MemPermute user has the same permute, it's beneficial to
        // propagate the MemPermute before Slice, otherwise it's not beneficial.
        bool hasDuplicateMemPermute = false;

        // For SliceOp, the propagation is beneficial once parent op has multi slice users, and total slice ops tensor
        // size is bigger than the original size.
        bool isSliceSizeGreaterThanInput = sliceSize > getShape(input).raw()[firstSliceAxe];

        bool isSameMemPermute = false;
        if (!memPermuteOps.empty()) {
            // Check all MemPermute with same permutation.
            const auto firstMemPermLayout = DimsOrder::fromAffineMap(memPermuteOps[0].getMemPermAttr().getValue());
            isSameMemPermute = llvm::all_of(memPermuteOps, [firstMemPermLayout](IE::MemPermuteOp& memPermuteOp) {
                return firstMemPermLayout == DimsOrder::fromAffineMap(memPermuteOp.getMemPermAttr().getValue());
            });
        }

        if (inputUserMemPermuteOps.size() > 0 && isSameMemPermute) {
            hasDuplicateMemPermute = true;
        }

        int64_t slicesWithMemPermute = memPermuteOps.size() - inputUserMemPermuteOps.size();
        return hasDuplicateMemPermute || (isSliceSizeGreaterThanInput && slicesWithMemPermute > 1);
    };

    if (auto sliceOp = mlir::dyn_cast_or_null<IE::SliceOp>(parentOp)) {
        auto isDupSlice = isInputWithDuplicateSlice(sliceOp.getSource());
        if (isDupSlice && beneficialStrideDMA) {
            return true;
        }
    }

    // For SliceOp, the propagation is beneficial with the additional condition
    // that the new MemPermute can be fused or removed, because moving memPermute through
    // SliceOp will increase the data movement from sliced tensor to the full tensor.
    // Current supported beneficial permutation conditions:
    // 1. Input has MemPermute, allowing subsequent MemPermute to be fused
    // 2. New MemPermute after propagation is a trivial permutation
    // 3. New MemPermute after propagation can be fused into an NCE task
    auto isBeneficialPermutation = [&](mlir::Value input) {
        if (mlir::isa_and_nonnull<IE::MemPermuteOp>(input.getDefiningOp())) {
            return true;
        }

        auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
        auto inputMemShape = inputType.getMemShape();
        if (isTrivialPermute(inputMemShape, memPerm)) {
            return true;
        }

        auto newPermuteOp = rewriter.create<IE::MemPermuteOp>(
                takeOpLoc(permuteOp, "mempermute_{0}", DimsOrder::fromAffineMap(permuteOp.getMemPerm())), input,
                permuteOp.getDstOrderAttr(), permuteOp.getMemPermAttr());
        auto doesFusedIntoNCE = false;
        if (auto layerWithPermute = IE::getFusableLayerWithPermuteInterface(newPermuteOp.getOperation())) {
            doesFusedIntoNCE = layerWithPermute.isSupportedPermutation(newPermuteOp);
        }
        rewriter.eraseOp(newPermuteOp);

        const auto isNceHasOneUse = input.getDefiningOp()->hasOneUse();
        return isNceHasOneUse && doesFusedIntoNCE;
    };

    if (auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(parentOp)) {
        // For ConcatOp, the propagation is beneficial once it benefits the stride DMA,
        // because moving memPermute through ConcatOp brings no extra data movement
        if (beneficialStrideDMA) {
            return true;
        }

        // In some cases, the propagation can introduce stride DMA on some inputs, but it can eliminate the MemPermute
        // on other inputs.
        // Experiments show that this is still beneficial overall for these cases.
        int64_t beneficialLevel = 0;
        for (auto input : concatOp.getInputs()) {
            auto inShape = getShape(input);
            if (!isBeneficialStrideDMA(inShape, parentOutShape)) {
                beneficialLevel--;
            } else {
                if (isBeneficialPermutation(input)) {
                    beneficialLevel++;
                }
            }
        }

        return beneficialLevel >= 0;
    }

    const auto beneficialPermutation = llvm::any_of(parentOp->getOperands(), isBeneficialPermutation);
    return beneficialStrideDMA && beneficialPermutation;
}

bool isPropagationBeneficialForMultiply(IE::MemPermuteOp memPermuteOp) {
    auto multiplyOp = memPermuteOp.getInput().getDefiningOp<IE::MultiplyOp>();
    if (multiplyOp == nullptr) {
        return false;
    }

    auto outputType = mlir::cast<NDTypeInterface>(memPermuteOp.getOutput().getType());
    auto order = outputType.getDimsOrder();
    auto outputShape = outputType.getShape();
    auto innerDim = getInnermostNonTrivialDim(outputShape, order);
    if (!innerDim.has_value()) {
        return false;
    }

    auto input1Shape = getShape(multiplyOp.getInput1());
    auto input2Shape = getShape(multiplyOp.getInput2());
    constexpr int64_t ACT_SHAVE_VAU_LENGTH = 512;
    // When it's going to be a SW Multiply，need to ensure the inner most dim size will be aligned to SHAVE VAU length
    // after propagation
    if (input1Shape != input2Shape &&
        outputShape[innerDim.value()] % Bit(ACT_SHAVE_VAU_LENGTH).to<Byte>().count() != 0) {
        return false;
    }

    auto isMemPermuteOrTrivial = [&](mlir::Value input) {
        if (mlir::isa_and_nonnull<IE::MemPermuteOp>(input.getDefiningOp())) {
            return true;
        }

        auto inputMemShape = getMemShape(input);
        auto perm = memPermuteOp.getMemPerm();
        return isTrivialPermute(inputMemShape, perm);
    };

    return static_cast<bool>(llvm::any_of(multiplyOp.getOperands(), isMemPermuteOrTrivial));
}

template <class ConcreteOp>
bool MoveMemPermuteThroughOp<ConcreteOp>::checkMemPermutePattern(mlir::Operation* permuteOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto memPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(permuteOp);
    if (memPermuteOp == nullptr) {
        return false;
    }
    if (!MoveThroughOpBase<ConcreteOp>::genericCheck(permuteOp)) {
        return false;
    }

    auto concreteOp = permuteOp->getOperand(0).getDefiningOp();
    if (mlir::isa<IE::ConcatOp, IE::SliceOp, IE::StridedSliceOp>(concreteOp) &&
        !isPropagationBeneficialForConcatAndSlice(memPermuteOp, rewriter)) {
        return false;
    }

    if (mlir::isa<IE::MultiplyOp>(concreteOp) && !isPropagationBeneficialForMultiply(memPermuteOp)) {
        return false;
    }

    return true;
}

template <class ConcreteOp>
mlir::AffineMap MoveMemPermuteThroughOp<ConcreteOp>::getPermutationMap(mlir::Operation* permuteOp) const {
    auto memPermuteOp = mlir::dyn_cast<IE::MemPermuteOp>(permuteOp);
    VPUX_THROW_WHEN(memPermuteOp == nullptr, "Not a MemPermuteOp");

    return memPermuteOp.getMemPerm();
}

template <class ConcreteOp>
mlir::Operation* MoveMemPermuteThroughOp<ConcreteOp>::createNewPermuteOp(mlir::Operation* permuteOp,
                                                                         mlir::Value newInput, mlir::AffineMap dstOrder,
                                                                         mlir::PatternRewriter& rewriter) const {
    auto memPermuteOp = mlir::dyn_cast<IE::MemPermuteOp>(permuteOp);
    VPUX_THROW_WHEN(memPermuteOp == nullptr, "Not a MemPermuteOp");

    return rewriter.create<IE::MemPermuteOp>(
            appendLoc(newInput.getLoc(),
                      llvm::formatv("_mempermute_{0}", DimsOrder::fromAffineMap(memPermuteOp.getMemPerm()))),
            newInput, dstOrder, memPermuteOp.getMemPerm());
}

//
// MoveMemPermuteThroughUpsampling
//
// Catch the pattern:
//
//      IE.Upsampling
//            |
//      IE.MemPermute
//
// And rewrite it to:
//
//      IE.MemPermute
//            |
//      IE.Upsampling
//            |
//     [IE.PermuteCast]
//
// The rewrite fires only when the swap turns the lowered Upsampling DMA from strided to
// non-strided. UpsamplingDMA writes its growth (factor>1) along the source axes; if a
// growth axis sits at or inside the innermost non-trivial memory dimension, the DMA must
// stride between rows/planes. Permuting the growth axis to an outer memory position
// removes that stride.

class MoveMemPermuteThroughUpsampling final : public MoveThroughOpBase<IE::UpsamplingOp> {
public:
    MoveMemPermuteThroughUpsampling(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : MoveThroughOpBase<IE::UpsamplingOp>(ctx, benefit, log), _log(log) {
    }

    bool checkMemPermutePattern(mlir::Operation* permuteOp, mlir::PatternRewriter& rewriter) const override;

    mlir::AffineMap getPermutationMap(mlir::Operation* permuteOp) const override;

    mlir::Operation* createNewPermuteOp(mlir::Operation* permuteOp, mlir::Value newInput, mlir::AffineMap dstOrder,
                                        mlir::PatternRewriter& rewriter) const override;

private:
    Logger _log;
};

namespace {
// Returns true iff any upsampling growth axis (factor>1) sits at the innermost
// non-trivial memory dimension under `order`. That is the condition for the lowered
// UpsamplingDMA to require strided writes.
bool isUpsamplingGrowthOnInnerMemDim(ShapeRef logicalShape, DimsOrder order, ArrayRef<int64_t> factor) {
    // factor is laid out as [W, H, C] (matches UpsamplingOp::upsampling_factor).
    // Map each factor entry back to its logical Dim.
    const SmallVector<Dim> factorAxes = {Dims4D::Act::W, Dims4D::Act::H, Dims4D::Act::C};
    VPUX_THROW_UNLESS(factor.size() == factorAxes.size(), "Upsampling factor must have 3 entries, got {0}",
                      factor.size());

    const auto memShape = order.toMemoryOrder(logicalShape);
    int64_t innermostNonTrivial = -1;
    for (int64_t pos = checked_cast<int64_t>(memShape.size()) - 1; pos >= 0; --pos) {
        if (memShape[MemDim(pos)] != 1) {
            innermostNonTrivial = pos;
            break;
        }
    }
    if (innermostNonTrivial < 0) {
        return false;
    }

    for (auto idx : irange(factor.size())) {
        const auto axis = factorAxes[idx];
        if (factor[idx] <= 1 || logicalShape[axis] == 1) {
            continue;
        }
        const auto memPos = order.toMemDim(axis).ind();
        if (memPos == innermostNonTrivial) {
            return true;
        }
    }
    return false;
}

mlir::AffineMap getMemPermFromPermuteOp(mlir::Operation* op) {
    if (auto memPermuteOp = mlir::dyn_cast_if_present<IE::MemPermuteOp>(op); memPermuteOp != nullptr) {
        return memPermuteOp.getMemPerm();
    }
    if (auto permuteQuantizeOp = mlir::dyn_cast_if_present<IE::PermuteQuantizeOp>(op); permuteQuantizeOp != nullptr) {
        return permuteQuantizeOp.getMemPerm();
    }
    VPUX_THROW("Not a permute operation");
}

bool isPlainPermuteQuantizeStrict(IE::PermuteQuantizeOp pq) {
    if (pq == nullptr) {
        return false;
    }
    const auto padStart = parseIntArrayAttr<int64_t>(pq.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(pq.getPadsEnd());
    const auto nonZero = [](auto p) {
        return p > 0;
    };
    if (llvm::any_of(padStart, nonZero) || llvm::any_of(padEnd, nonZero)) {
        return false;
    }
    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(pq.getInput().getType()).getElementType();
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(pq.getOutput().getType()).getElementType();
    if (mlir::isa<mlir::quant::QuantizedType>(outElemType)) {
        return false;
    }
    return inElemType == outElemType;
}

}  // namespace

bool MoveMemPermuteThroughUpsampling::checkMemPermutePattern(mlir::Operation* permuteOp, mlir::PatternRewriter&) const {
    if (!mlir::isa_and_present<IE::MemPermuteOp, IE::PermuteQuantizeOp>(permuteOp)) {
        return false;
    }

    if (!MoveThroughOpBase<IE::UpsamplingOp>::genericCheck(permuteOp)) {
        return false;
    }

    if (auto pq = mlir::dyn_cast_if_present<IE::PermuteQuantizeOp>(permuteOp); pq != nullptr) {
        if (!isPlainPermuteQuantizeStrict(pq)) {
            return false;
        }
    }

    auto upsamplingOp = mlir::dyn_cast<IE::UpsamplingOp>(permuteOp->getOperand(0).getDefiningOp());
    VPUX_THROW_WHEN(upsamplingOp == nullptr, "Not an UpsamplingOp");
    const auto factor = parseIntArrayAttr<int64_t>(upsamplingOp.getUpsamplingFactor());
    const auto inLogicalShape = getShape(upsamplingOp.getInput());
    const auto srcOrder = DimsOrder::fromValue(upsamplingOp.getInput());

    // Compute the layout the Upsampling input would have after the swap.
    auto* ctx = permuteOp->getContext();
    const auto memPerm = getMemPermFromPermuteOp(permuteOp);
    const auto newOrder = DimsOrder::fromAffineMap(memPerm.compose(srcOrder.toAffineMap(ctx)));

    const bool stridedBefore = isUpsamplingGrowthOnInnerMemDim(inLogicalShape, srcOrder, factor);
    const bool stridedAfter = isUpsamplingGrowthOnInnerMemDim(inLogicalShape, newOrder, factor);
    if (!stridedBefore || stridedAfter) {
        return false;
    }

    _log.trace("[MoveMemPermuteThroughUpsampling] gate passed: srcOrder {0} -> newOrder {1}, factor {2}", srcOrder,
               newOrder, factor);
    return true;
}

mlir::AffineMap MoveMemPermuteThroughUpsampling::getPermutationMap(mlir::Operation* permuteOp) const {
    return getMemPermFromPermuteOp(permuteOp);
}

mlir::Operation* MoveMemPermuteThroughUpsampling::createNewPermuteOp(mlir::Operation* permuteOp, mlir::Value newInput,
                                                                     mlir::AffineMap dstOrder,
                                                                     mlir::PatternRewriter& rewriter) const {
    if (auto memPermuteOp = mlir::dyn_cast_if_present<IE::MemPermuteOp>(permuteOp); memPermuteOp != nullptr) {
        return rewriter.create<IE::MemPermuteOp>(appendLoc(memPermuteOp.getLoc(), "swap"), newInput, dstOrder,
                                                 memPermuteOp.getMemPerm());
    }
    if (auto pqOp = mlir::dyn_cast_if_present<IE::PermuteQuantizeOp>(permuteOp); pqOp != nullptr) {
        const auto dstOrderAttr = mlir::AffineMapAttr::get(dstOrder);
        return rewriter.create<IE::PermuteQuantizeOp>(appendLoc(pqOp.getLoc(), "swap"), newInput, dstOrderAttr,
                                                      pqOp.getMemPermAttr(), pqOp.getDstElemTypeAttr(),
                                                      pqOp.getPadsBeginAttr(), pqOp.getPadsEndAttr());
    }
    VPUX_THROW("Not a permute operation to create");
}

//
// MovePermuteQuantizeThroughOp
//

template <class ConcreteOp>
class MovePermuteQuantizeThroughOp final : public MoveThroughOpBase<ConcreteOp> {
public:
    MovePermuteQuantizeThroughOp(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : MoveThroughOpBase<ConcreteOp>(ctx, benefit, log) {
    }

    bool checkMemPermutePattern(mlir::Operation* permuteOp, mlir::PatternRewriter& rewriter) const override;

    mlir::AffineMap getPermutationMap(mlir::Operation* permuteOp) const override;

    mlir::Operation* createNewPermuteOp(mlir::Operation* permuteOp, mlir::Value newInput, mlir::AffineMap dstOrder,
                                        mlir::PatternRewriter& rewriter) const override;
};

template <class ConcreteOp>
bool MovePermuteQuantizeThroughOp<ConcreteOp>::checkMemPermutePattern(mlir::Operation* permuteOp,
                                                                      mlir::PatternRewriter&) const {
    auto permuteQuantizeOp = mlir::dyn_cast_or_null<IE::PermuteQuantizeOp>(permuteOp);
    if (permuteQuantizeOp == nullptr) {
        return false;
    }

    // Check PermuteQuantize pads attributes.
    const auto padStart = parseIntArrayAttr<int64_t>(permuteQuantizeOp.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(permuteQuantizeOp.getPadsEnd());

    const auto nonZeroPadStart = llvm::any_of(padStart, [](auto pad) {
        return pad > 0;
    });
    const auto nonZeroPadEnd = llvm::any_of(padEnd, [](auto pad) {
        return pad > 0;
    });
    if (nonZeroPadStart || nonZeroPadEnd) {
        return false;
    }

    // Check PermuteQuantize output element type.
    const auto permuteQuantizeOutElemType =
            mlir::cast<vpux::NDTypeInterface>(permuteQuantizeOp.getOutput().getType()).getElementType();
    if (mlir::isa<mlir::quant::QuantizedType>(permuteQuantizeOutElemType)) {
        return false;
    }

    // ConcreteOp should have single input or all inputs are from the same parent
    auto concreteOp = permuteOp->getOperand(0).getDefiningOp();
    auto operands = concreteOp->getOperands();
    auto hasTheSameOperands = llvm::all_of(operands, [&](const mlir::Value operand) {
        return operand == operands.front();
    });
    if (!hasTheSameOperands) {
        return false;
    }

    return MoveThroughOpBase<ConcreteOp>::genericCheck(permuteOp);
}

template <class ConcreteOp>
mlir::AffineMap MovePermuteQuantizeThroughOp<ConcreteOp>::getPermutationMap(mlir::Operation* permuteOp) const {
    auto permuteQuantizeOp = mlir::dyn_cast<IE::PermuteQuantizeOp>(permuteOp);
    VPUX_THROW_WHEN(permuteQuantizeOp == nullptr, "Not a PermuteQuantizeOp");

    return permuteQuantizeOp.getMemPerm();
}

template <class ConcreteOp>
mlir::Operation* MovePermuteQuantizeThroughOp<ConcreteOp>::createNewPermuteOp(mlir::Operation* permuteOp,
                                                                              mlir::Value newInput, mlir::AffineMap,
                                                                              mlir::PatternRewriter& rewriter) const {
    auto permuteQuantizeOp = mlir::dyn_cast<IE::PermuteQuantizeOp>(permuteOp);
    VPUX_THROW_WHEN(permuteQuantizeOp == nullptr, "Not a PermuteQuantizeOp");

    return rewriter.create<IE::PermuteQuantizeOp>(
            permuteQuantizeOp->getLoc(), newInput, permuteQuantizeOp.getDstOrderAttr(),
            permuteQuantizeOp.getMemPermAttr(), permuteQuantizeOp.getDstElemTypeAttr(),
            permuteQuantizeOp.getPadsBeginAttr(), permuteQuantizeOp.getPadsEndAttr());
}

//
// PropagatePermuteThroughConcatSlice
//
// Closed-form rewriter for the pattern:
//
//     Concat
//      ├── Slice ── MemPermute / PermuteQuantize
//      ├── Slice ── MemPermute / PermuteQuantize
//      └── ...
//
// To:
//
//     each Concat input ── MemPermute
//                            |
//                          new Concat ── Slice ── PermuteCast (per original permute user)
//
// Match conditions (strict, single-shot rewrite — every gate must pass):
//   1. The ConcatOp is not per-axis quantized.
//   2. The ConcatOp has exactly one unique non-const operand (others must be Const::DeclareOp).
//   3. All ConcatOp users are SliceOps; each Slice has exactly one user, which is either
//      a MemPermute or a "plain" PermuteQuantize (pad==0, non-quantized output,
//      dstElemType == inputElemType — fully equivalent to a MemPermute).
//   4. At least one permute user must be a MemPermute.
//   5. All Slices slice along the same single axis.
//   6. Sum of slice sizes along that axis > Concat output size on that axis (otherwise
//      lifting the permute would touch more data than the original sliced reads).
//   7. All permute users share the same memPerm and dstOrder.
//   8. Every permute user is a pure reorder (input and output logical shapes are equal),
//      so the original Slice's logical-dim offsets/sizes remain valid after the lift.
//   9. The permute is non-trivial.
//  10. The Slice DMA must transition from strided to non-strided after the lift
//      (otherwise the rewrite would pessimize the slice DMA).
//  11. No Concat-input DMA regresses from non-strided to strided after the lift.
//

class PropagatePermuteThroughConcatSlice final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    PropagatePermuteThroughConcatSlice(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::SliceOp>(ctx, benefit), _log(log) {
        this->setDebugName("PropagatePermuteThroughConcatSlice");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

    Logger _log;
};

mlir::LogicalResult PropagatePermuteThroughConcatSlice::matchAndRewrite(IE::SliceOp rootSliceOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    auto concatOp = rootSliceOp.getInput().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr) {
        return mlir::failure();
    }

    // Per-axis quantized element types are excluded (mirroring MoveThroughOpBase::genericCheck).
    const auto concatInElemType = mlir::cast<vpux::NDTypeInterface>(concatOp->getOperand(0).getType()).getElementType();
    const auto concatOutElemType = mlir::cast<vpux::NDTypeInterface>(concatOp.getOutput().getType()).getElementType();
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(concatInElemType) ||
        mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(concatOutElemType)) {
        return mlir::failure();
    }

    llvm::SmallPtrSet<mlir::Value, 4> uniqueNonConst;
    for (auto operand : concatOp->getOperands()) {
        if (!mlir::isa_and_nonnull<Const::DeclareOp>(operand.getDefiningOp())) {
            uniqueNonConst.insert(operand);
        }
    }
    if (uniqueNonConst.size() != 1) {
        return mlir::failure();
    }

    auto concatUsers = llvm::to_vector(concatOp.getOutput().getUsers());
    if (concatUsers.empty()) {
        return mlir::failure();
    }

    SmallVector<IE::SliceOp> slices;
    SmallVector<mlir::Operation*> permuteUsers;
    slices.reserve(concatUsers.size());
    permuteUsers.reserve(concatUsers.size());

    // All slices must slice along the same single axis.
    int64_t commonSliceAxis = -1;
    int64_t totalSliceSize = 0;

    for (auto* user : concatUsers) {
        auto slice = mlir::dyn_cast<IE::SliceOp>(user);
        if (slice == nullptr) {
            return mlir::failure();
        }
        if (!slice->hasOneUse()) {
            return mlir::failure();
        }
        auto sliceAxes = getSliceAxes(slice);
        if (sliceAxes.size() != 1) {
            return mlir::failure();
        }
        if (commonSliceAxis == -1) {
            commonSliceAxis = sliceAxes.front();
        } else if (commonSliceAxis != static_cast<int64_t>(sliceAxes.front())) {
            return mlir::failure();
        }
        const auto sliceShape = getShape(slice.getResult()).raw();
        totalSliceSize += sliceShape[commonSliceAxis];

        auto* sliceUser = *slice.getResult().getUsers().begin();
        if (auto pq = mlir::dyn_cast<IE::PermuteQuantizeOp>(sliceUser)) {
            if (!isPlainPermuteQuantizeStrict(pq)) {
                return mlir::failure();
            }
        } else if (!mlir::isa<IE::MemPermuteOp>(sliceUser)) {
            return mlir::failure();
        }

        slices.push_back(slice);
        permuteUsers.push_back(sliceUser);
    }

    if (commonSliceAxis < 0) {
        return mlir::failure();
    }

    // Require at least one MemPermute user; skip patterns where every user is PermuteQuantize.
    const bool hasMemPermuteUser = llvm::any_of(permuteUsers, [](mlir::Operation* op) {
        return mlir::isa<IE::MemPermuteOp>(op);
    });
    if (!hasMemPermuteUser) {
        return mlir::failure();
    }

    // Propagation is only beneficial when the total slice size along the slice axis is
    // greater than the Concat's size on that axis (slices have overlapping/repeated reads).
    // Otherwise the Permute would touch more data after propagation than before — a net loss.
    const auto concatAxisSize = getShape(concatOp.getOutput()).raw()[commonSliceAxis];
    if (totalSliceSize <= concatAxisSize) {
        return mlir::failure();
    }

    // All permute users must share the same memPerm and dstOrder.
    auto getMemPerm = [](mlir::Operation* op) {
        if (auto mp = mlir::dyn_cast<IE::MemPermuteOp>(op)) {
            return mp.getMemPerm();
        }
        return mlir::cast<IE::PermuteQuantizeOp>(op).getMemPerm();
    };
    auto getDstOrder = [](mlir::Operation* op) {
        if (auto mp = mlir::dyn_cast<IE::MemPermuteOp>(op)) {
            return mp.getDstOrder();
        }
        return mlir::cast<IE::PermuteQuantizeOp>(op).getDstOrder();
    };

    const auto memPerm = getMemPerm(permuteUsers.front());
    const auto dstOrder = getDstOrder(permuteUsers.front());
    for (auto* user : permuteUsers) {
        if (getMemPerm(user) != memPerm || getDstOrder(user) != dstOrder) {
            return mlir::failure();
        }
        if (getShape(user->getOperand(0)) != getShape(user->getResult(0))) {
            return mlir::failure();
        }
    }

    // Skip a trivial permute (would be a no-op rewrite that may loop the pattern).
    const auto concatType = mlir::cast<vpux::NDTypeInterface>(concatOp.getOutput().getType());
    const auto concatMemShape = concatType.getMemShape();
    if (isTrivialPermute(concatMemShape, memPerm)) {
        return mlir::failure();
    }

    auto* ctx = rewriter.getContext();
    const auto inOrder = DimsOrder::fromValue(concatOp->getOperand(0));
    const auto perm = memPerm.compose(inOrder.toAffineMap(ctx));
    const auto outOrder = DimsOrder::fromAffineMap(perm);

    // DMA-friendliness gate: only fire when the rewrite turns a strided Slice DMA into a
    // non-strided one. A Slice (or Concat input write) along memory position p is non-
    // strided iff every memory dim before p has size 1.
    auto isNonStridedAt = [](MemShapeRef memShape, size_t memPos) {
        for (size_t i = 0; i < memPos; ++i) {
            if (memShape[MemDim(i)] != 1) {
                return false;
            }
        }
        return true;
    };
    const auto inOrderConcat = concatType.getDimsOrder();
    const auto outConcatMemShape = concatType.changeDimsOrder(outOrder).getMemShape();
    const auto sliceMemPosBefore = inOrderConcat.dimPos(Dim(commonSliceAxis));
    const auto sliceMemPosAfter = outOrder.dimPos(Dim(commonSliceAxis));
    const bool stridedBefore = !isNonStridedAt(concatMemShape, sliceMemPosBefore);
    const bool nonStridedAfter = isNonStridedAt(outConcatMemShape, sliceMemPosAfter);
    if (!stridedBefore || !nonStridedAfter) {
        return mlir::failure();
    }

    // Concat-input DMA gate: every concat-input write that was non-strided before the
    // rewrite must remain non-strided after. The rewrite changes the Concat output layout
    // from inOrderConcat to outOrder, which can shift concat-axis memory positions and
    // pessimize the per-input writes. Reject the rewrite if any concat axis would regress.
    for (const auto axis : IE::getConcatAxes(concatOp)) {
        const auto memPosBefore = inOrderConcat.dimPos(Dim(axis));
        const auto memPosAfter = outOrder.dimPos(Dim(axis));
        if (isNonStridedAt(concatMemShape, memPosBefore) && !isNonStridedAt(outConcatMemShape, memPosAfter)) {
            return mlir::failure();
        }
    }

    _log.trace("PropagatePermuteThroughConcatSlice: matched Concat at {0} with {1} slices", concatOp->getLoc(),
               slices.size());

    rewriter.setInsertionPoint(concatOp);
    // Rewrite: per-input MemPermute -> new Concat -> per-slice Slice -> PermuteCast.
    SmallVector<mlir::Value> newInputs;
    newInputs.reserve(concatOp->getNumOperands());
    DenseMap<mlir::Value, mlir::Value> operandsMap;
    for (auto& opOperand : concatOp->getOpOperands()) {
        auto input = opOperand.get();
        auto it = operandsMap.find(input);
        if (it != operandsMap.end()) {
            newInputs.push_back(it->second);
            continue;
        }

        auto newPermute = rewriter.create<IE::MemPermuteOp>(
                takeOpLoc(concatOp, "input_permute_{0}", opOperand.getOperandNumber()), input,
                outOrder.toAffineMap(ctx), memPerm);
        operandsMap.insert({input, newPermute.getResult()});
        newInputs.push_back(newPermute.getResult());
    }

    mlir::IRMapping mapper;
    mapper.map(concatOp->getOperands(), newInputs);
    auto* newConcatOp = rewriter.clone(*concatOp.getOperation(), mapper);
    auto newConcatOut = newConcatOp->getResult(0);
    rewriter.modifyOpInPlace(newConcatOp, [&] {
        newConcatOut.setType(concatType.changeDimsOrder(outOrder));
    });

    for (auto&& [origSlice, permuteUser] : llvm::zip(slices, permuteUsers)) {
        const auto origSliceType = mlir::cast<vpux::NDTypeInterface>(origSlice.getResult().getType());
        const auto newSliceType = origSliceType.changeDimsOrder(outOrder);
        rewriter.setInsertionPoint(origSlice);
        auto newSlice = rewriter.create<IE::SliceOp>(origSlice->getLoc(), newSliceType, newConcatOut,
                                                     origSlice.getStaticOffsetsAttr(), origSlice.getStaticSizesAttr());

        const auto origPermResultType = permuteUser->getResult(0).getType();
        const auto origPermOrder = mlir::cast<vpux::NDTypeInterface>(origPermResultType).getDimsOrder();
        auto permuteCast = rewriter.createOrFold<IE::PermuteCastOp>(
                permuteUser->getLoc(), newSlice.getResult(), origPermOrder.toAffineMap(ctx),
                mlir::AffineMap::getMultiDimIdentityMap(outOrder.numDims(), ctx));

        rewriter.replaceOp(permuteUser, permuteCast);
        rewriter.eraseOp(origSlice);
    }

    rewriter.eraseOp(concatOp);
    return mlir::success();
}

//
// MoveThroughSlice
//
// Replace the pattern:
//
//     ShapeCast
//         |
//     MemPermute
//
// With below subgraph:
//
//     PermuteCast
//          |
//     MemPermute
//          |
//      ShapeCast
//          |
//     PermuteCast

class MoveThroughShapeCast final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    MoveThroughShapeCast(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ShapeCastOp>(ctx, benefit), _log(log) {
        this->setDebugName("MoveThroughSlice");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp shapeCastOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// TODO: E-121944 Try to convert ShapeCast to AffineReshape in Canonicalization
mlir::LogicalResult MoveThroughShapeCast::matchAndRewrite(IE::ShapeCastOp shapeCastOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    _log.trace("MoveThroughShapeCast: Got {0}", shapeCastOp->getLoc());
    if (!shapeCastOp->hasOneUse()) {
        return mlir::failure();
    }

    auto memPermuteOp = mlir::dyn_cast<IE::MemPermuteOp>(*shapeCastOp->getUsers().begin());
    if (memPermuteOp == nullptr) {
        return mlir::failure();
    }

    const auto origReshapeInType = mlir::cast<vpux::NDTypeInterface>(shapeCastOp->getOperand(0).getType());
    const auto origReshapeOutType = mlir::cast<vpux::NDTypeInterface>(shapeCastOp->getResult(0).getType());
    const auto origReshapeInShape = origReshapeInType.getShape();
    const auto origReshapeOutShape = origReshapeOutType.getShape();
    const auto origReshapeInMemShape = origReshapeInType.getMemShape();
    const auto origReshapeOutMemShape = origReshapeOutType.getMemShape();
    const auto originPerm = DimsOrder::fromAffineMap(memPermuteOp.getMemPerm());
    const auto originPermVec = to_small_vector(originPerm.toPermutation() | transformed([](Dim dim) {
                                                   return checked_cast<int64_t>(dim.ind());
                                               }));
    const auto origPermRef = ArrayRef(originPermVec);

    // Check that tensor rank is 4, otherwise compilation fails in later passes
    auto inRank = origReshapeInType.getRank();
    auto outRank = origReshapeOutType.getRank();
    if (inRank != 4 || outRank != 4) {
        return mlir::failure();
    }

    // Since ShapeCast's reshaped axes might not be continous in logical shape,
    // but must be continous in memory shape.
    // So we should use the dim mapping in memory shape for compatibility check and
    // new permutation deduction
    auto memDimMapping = vpux::IE::getReassociationMap(origReshapeInMemShape.raw(), origReshapeOutMemShape.raw());
    if (mlir::failed(memDimMapping)) {
        _log.trace("Cannot get correct memDimMapping");
        return mlir::failure();
    }
    if (!areReshapedAxesPermutedIntegratedly(memDimMapping.value(), origPermRef, DimsOrder::NCHW,
                                             origReshapeOutMemShape)) {
        _log.trace("Split axes are permuted");
        return mlir::failure();
    }

    // Cast to canonical order for convenience
    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<unsigned>(origReshapeInShape.size()), ctx);
    auto inputCast = rewriter.create<IE::PermuteCastOp>(shapeCastOp->getLoc(), shapeCastOp->getOperand(0), identityMap,
                                                        identityMap);

    auto newPerm = calculateNewPermutation(memDimMapping.value(), origPermRef, DimsOrder::NCHW, DimsOrder::NCHW,
                                           origReshapeInMemShape, origReshapeOutMemShape, _log, ctx);
    const auto identityOrderAttr = mlir::AffineMapAttr::get(identityMap);
    auto newMemPermute = rewriter.create<IE::MemPermuteOp>(takeOpLoc(shapeCastOp, "mempermute"), inputCast.getOutput(),
                                                           identityOrderAttr, mlir::AffineMapAttr::get(newPerm));

    auto outputShapeAttr =
            getIntArrayAttr(ctx, mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getOutput().getType()).getMemShape());
    auto newShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(shapeCastOp->getLoc(), newMemPermute.getOutput(), outputShapeAttr);

    auto newPermuteCast = rewriter.createOrFold<IE::PermuteCastOp>(
            shapeCastOp->getLoc(), newShapeCastOp->getResult(0), memPermuteOp.getDstOrder(),
            mlir::AffineMap::getMultiDimIdentityMap(checked_cast<unsigned>(origReshapeOutShape.size()), ctx));

    rewriter.replaceOp(memPermuteOp, newPermuteCast);
    return mlir::success();
}

//
// MoveMemPermuteThroughReshape
//
// Replace the pattern:
//      Reshape
//         |
//     MemPermute
//         |
// With below subgraph:
//     MemPermute
//         |
//      Reshape
//         |

class MoveMemPermuteThroughReshape final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    MoveMemPermuteThroughReshape(mlir::MLIRContext* ctx, Logger log, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::ReshapeOp>(ctx, benefit), _log(log) {
        this->setDebugName("MoveMemPermuteThroughReshape");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp reshapeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveMemPermuteThroughReshape::matchAndRewrite(IE::ReshapeOp reshapeOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    _log.trace("[{0}]: Got {1}", getDebugName(), reshapeOp->getLoc());
    if (!reshapeOp->hasOneUse()) {
        return mlir::failure();
    }

    auto memPermuteOp = mlir::dyn_cast<IE::MemPermuteOp>(*reshapeOp->getUsers().begin());
    if (memPermuteOp == nullptr) {
        _log.trace("There is no MemPermute user op");
        return mlir::failure();
    }

    const auto origReshapeInType = mlir::cast<vpux::NDTypeInterface>(reshapeOp->getOperand(0).getType());
    const auto origReshapeOutType = mlir::cast<vpux::NDTypeInterface>(reshapeOp->getResult(0).getType());
    const auto origReshapeInShape = origReshapeInType.getShape();
    const auto origReshapeOutShape = origReshapeOutType.getShape();
    const auto originPerm = DimsOrder::fromAffineMap(memPermuteOp.getMemPerm());

    auto inRank = origReshapeInType.getRank();
    auto outRank = origReshapeOutType.getRank();
    if (inRank != 4 || outRank != 4) {
        _log.trace("Only support 4D rank");
        return mlir::failure();
    }

    // Check MemPermute output layout due to Reshape only support default order
    const auto memPermuteOutputOrder =
            mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getOutput().getType()).getDimsOrder();
    const auto expectedOutOrder = DimsOrder::NCHW;
    if (memPermuteOutputOrder != expectedOutOrder) {
        _log.trace("Unsupported output layout. Expected: '{0}', got: '{1}'", expectedOutOrder, memPermuteOutputOrder);
        return mlir::failure();
    }

    SmallVector<int32_t> reshapedDims{};
    for (auto dim : origReshapeInShape | indexed) {
        auto dimIndex = dim.index();
        auto dimValue = dim.value();

        if (origReshapeOutShape[Dim(dimIndex)] != dimValue) {
            reshapedDims.push_back(dimIndex);
        }
    }
    auto permuteOrder = to_small_vector(irange(originPerm.numDims()) | transformed([&](uint64_t idx) {
                                            return checked_cast<uint64_t>(originPerm.dimAt(idx).ind());
                                        }));
    const auto memPermuteInputOrder =
            mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getInput().getType()).getDimsOrder();
    const auto inputOrder = to_small_vector(irange(memPermuteInputOrder.numDims()) | transformed([&](uint64_t idx) {
                                                return checked_cast<uint64_t>(memPermuteInputOrder.dimAt(idx).ind());
                                            }));

    // Check if the permute dims have been reshaped
    for (size_t ind = 0; ind < inputOrder.size(); ind++) {
        if (permuteOrder[ind] != inputOrder[ind]) {
            if (llvm::find(reshapedDims, ind) != reshapedDims.end()) {
                _log.trace("Permute dims have been reshaped");
                return mlir::failure();
            }
        }
    }

    // Create new MemPermute
    auto newMemPermuteOp =
            rewriter.create<IE::MemPermuteOp>(takeOpLoc(reshapeOp, "mempermute"), reshapeOp.getInput(),
                                              memPermuteOp.getDstOrderAttr(), memPermuteOp.getMemPermAttr());

    // Create new Reshape
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(memPermuteOp, newMemPermuteOp.getOutput(),
                                               getIntArrayAttr(ctx, getShape(memPermuteOp.getOutput())));

    return mlir::success();
}

//
// PropagateMemPermuteBeforeOpPass
//

class PropagateMemPermuteBeforeOpPass final :
        public IE::impl::PropagateMemPermuteBeforeOpBase<PropagateMemPermuteBeforeOpPass> {
public:
    explicit PropagateMemPermuteBeforeOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void PropagateMemPermuteBeforeOpPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptimizeMemPermute>(&ctx, _log);
    patterns.add<PropagatePermuteQuantize>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughOp<IE::MVNOp>>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughOp<IE::GeluOp>>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughOp<IE::SqrtOp>>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughOp<IE::QuantizeCastOp>>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughOp<IE::ConcatOp>>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughOp<IE::SliceOp>>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughOp<IE::StridedSliceOp>>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughOp<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<MovePermuteQuantizeThroughOp<IE::MultiplyOp>>(&ctx, _log);
    // Higher benefit so the closed Concat→Slice→Permute pattern is rewritten atomically before
    // per-Slice MoveMemPermuteThroughOp<Slice> can split it up.
    patterns.add<PropagatePermuteThroughConcatSlice>(&ctx, _log, vpux::benefitMid);
    patterns.add<MoveThroughShapeCast>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughReshape>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughUpsampling>(&ctx, _log);
    IE::ReshapeOp::getCanonicalizationPatterns(patterns, &ctx);
    IE::MemPermuteOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateMemPermuteBeforeOpPass(Logger log) {
    return std::make_unique<PropagateMemPermuteBeforeOpPass>(log);
}

void vpux::IE::registerPropagateMemPermuteBeforeOpRewriters(RewriterRegistry& registry,
                                                            ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                                            Logger log) {
    registry.registerRewriterSet("propagate-mem-permute-before-op-set", [&registry, benefitLevels, index, log]() {
        registry.registerRewriter<OptimizeMemPermute>("optimize-mem-permute", log, benefitLevels[index]);
        registry.registerRewriter<PropagatePermuteQuantize>("propagate-permute-quantize", log, benefitLevels[index]);
        registry.registerRewriter<MoveMemPermuteThroughOp<IE::MVNOp>>("move-mem-permute-through-op-mvn", log,
                                                                      benefitLevels[index]);
        registry.registerRewriter<MoveMemPermuteThroughOp<IE::GeluOp>>("move-mem-permute-through-op-gelu", log,
                                                                       benefitLevels[index]);
        registry.registerRewriter<MoveMemPermuteThroughOp<IE::SqrtOp>>("move-mem-permute-through-op-sqrt", log,
                                                                       benefitLevels[index]);
        registry.registerRewriter<MoveMemPermuteThroughOp<IE::QuantizeCastOp>>(
                "move-mem-permute-through-op-quantize-cast", log, benefitLevels[index]);
        registry.registerRewriter<MoveMemPermuteThroughOp<IE::ConcatOp>>("move-mem-permute-through-op-concat", log,
                                                                         benefitLevels[index]);
        registry.registerRewriter<MoveMemPermuteThroughOp<IE::SliceOp>>("move-mem-permute-through-op-slice", log,
                                                                        benefitLevels[index]);
        registry.registerRewriter<MoveMemPermuteThroughOp<IE::StridedSliceOp>>(
                "move-mem-permute-through-op-strided-slice", log, benefitLevels[index]);

        registry.registerRewriter<MoveMemPermuteThroughOp<IE::MultiplyOp>>("move-mem-permute-through-op-multiply", log,
                                                                           benefitLevels[index]);
        registry.registerRewriter<MovePermuteQuantizeThroughOp<IE::MultiplyOp>>(
                "move-permute-quantize-through-op-multiply", log, benefitLevels[index]);
        // Higher benefit so the closed Concat→Slice→Permute pattern is rewritten atomically before
        // per-Slice MoveMemPermuteThroughOp can split it up.
        registry.registerRewriter<PropagatePermuteThroughConcatSlice>(
                "propagate-permute-through-concat-slice", log,
                mlir::PatternBenefit(benefitLevels[index].getBenefit() + 1));
        registry.registerRewriter<MoveThroughShapeCast>("move-through-shape-cast", log, benefitLevels[index]);
        registry.registerRewriter<MoveMemPermuteThroughReshape>("move-mem-permute-through-reshape", log,
                                                                benefitLevels[index]);
        registry.registerRewriter<MoveMemPermuteThroughUpsampling>("move-mem-permute-through-op-upsampling", log,
                                                                   benefitLevels[index]);
        // Manually invoking these rewriter despite canonicalizer handling it. This is required for dynamic rewriter
        // implementation
        IE::registerReshapeOpRewriters(registry, benefitLevels, index);
        IE::registerMemPermuteOpRewriters(registry, benefitLevels, index);
    });
}
