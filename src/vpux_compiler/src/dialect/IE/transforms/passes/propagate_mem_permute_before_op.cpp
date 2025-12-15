//
// Copyright (C) 2023-2025 Intel Corporation.
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
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
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

bool doesVectorContainSubVector(ArrayRef<int64_t> vec, ArrayRef<int64_t> subVec) {
    return std::search(vec.begin(), vec.end(), subVec.begin(), subVec.end()) != vec.end();
}

bool areReshapedAxesPermutedIntegratedly(const SmallVector<SmallVector<int64_t>>& dimMapping,
                                         ArrayRef<int64_t> permAxis, DimsOrder permInOrder,
                                         vpux::MemShapeRef memShapeRef) {
    SmallVector<SmallVector<int64_t>> splitAxesVec;

    for (size_t inIdx = 0; inIdx < dimMapping.size(); inIdx++) {
        auto mappedDim = dimMapping[inIdx];
        SmallVector<int64_t> splitAxes;
        // mappedDim.size() > 1 indicates a split of input axis
        if (mappedDim.size() > 1) {
            for (size_t mapId = 0; mapId < mappedDim.size(); mapId++) {
                size_t outIdx = mappedDim[mapId];
                auto memDim = permInOrder.toMemDim(Dim(outIdx));
                splitAxes.push_back(memDim.ind());
            }
            splitAxesVec.push_back(splitAxes);
        }
    }

    if (splitAxesVec.empty()) {
        return true;
    }

    const auto nonTrivialMemDimPredicate = [&](const int64_t memDim) -> bool {
        return memShapeRef[MemDim(memDim)] > 1;
    };

    for (auto& splitAxes : splitAxesVec) {
        auto splitAxesRef = ArrayRef(splitAxes);
        const auto nonTrivialMemDims =
                std::count_if(splitAxesRef.begin(), splitAxesRef.end(), nonTrivialMemDimPredicate);
        // Allow the propagation if the data on split axes are moved as a whole.
        // Below cases can meet this requirement.
        // 1.Permutation does not break split axes.
        // 2.Split axes have no more than one non-trivial memDim, it's allowed to break split axes.
        if (!doesVectorContainSubVector(permAxis, splitAxesRef) && (nonTrivialMemDims > 1)) {
            return false;
        }
    }

    return true;
}

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
                                        DimsOrder affineReshapeInOrder, DimsOrder affineReshapeOutOrder,
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

    auto newPermute = rewriter.create<IE::MemPermuteOp>(
            takeOpLoc(permuteOp, llvm::formatv("_{0}", DimsOrder::fromAffineMap(identityMap))), inputCast.getOutput(),
            identityOrderAttr, newPermAttr);

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
    OptimizeMemPermute(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
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
    PropagatePermuteQuantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx), _log(log) {
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
    MoveThroughOpBase(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
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
        iface.inferLayoutInfo(orderInfo, /*seOpsEnabled=*/false, /*seExperimentalOpsEnabled=*/false);
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
            auto newLoc = takeOpLoc(concreteOp, llvm::formatv("_input_{0}", input.getOperandNumber()));
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
    MoveMemPermuteThroughOp(mlir::MLIRContext* ctx, Logger log): MoveThroughOpBase<ConcreteOp>(ctx, log) {
    }

    bool checkMemPermutePattern(mlir::Operation* permuteOp, mlir::PatternRewriter& rewriter) const override;

    mlir::AffineMap getPermutationMap(mlir::Operation* permuteOp) const override;

    mlir::Operation* createNewPermuteOp(mlir::Operation* permuteOp, mlir::Value newInput, mlir::AffineMap dstOrder,
                                        mlir::PatternRewriter& rewriter) const override;

    bool isPropagationBeneficialForConcatAndSlice(IE::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const;
};

template <class ConcreteOp>
bool MoveMemPermuteThroughOp<ConcreteOp>::isPropagationBeneficialForConcatAndSlice(
        IE::MemPermuteOp permuteOp, mlir::PatternRewriter& rewriter) const {
    auto parentOp = permuteOp.getInput().getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::ConcatOp, IE::SliceOp>(parentOp)) {
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

        return hasDuplicateMemPermute || isSliceSizeGreaterThanInput;
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
                takeOpLoc(permuteOp,
                          llvm::formatv("_mempermute_{0}", DimsOrder::fromAffineMap(permuteOp.getMemPerm()))),
                input, permuteOp.getDstOrderAttr(), permuteOp.getMemPermAttr());
        auto doesFusedIntoNCE = false;
        if (auto layerWithPermute = getFusableLayerWithPermuteInterface(newPermuteOp.getOperation())) {
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
    if (mlir::isa<IE::ConcatOp, IE::SliceOp>(concreteOp) &&
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
// MovePermuteQuantizeThroughOp
//

template <class ConcreteOp>
class MovePermuteQuantizeThroughOp final : public MoveThroughOpBase<ConcreteOp> {
public:
    MovePermuteQuantizeThroughOp(mlir::MLIRContext* ctx, Logger log): MoveThroughOpBase<ConcreteOp>(ctx, log) {
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
    MoveThroughShapeCast(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ShapeCastOp>(ctx), _log(log) {
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
    auto newMemPermute = rewriter.create<IE::MemPermuteOp>(takeOpLoc(shapeCastOp, "_mempermute"), inputCast.getOutput(),
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
    MoveMemPermuteThroughReshape(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReshapeOp>(ctx), _log(log) {
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
            rewriter.create<IE::MemPermuteOp>(takeOpLoc(reshapeOp, "_mempermute"), reshapeOp.getInput(),
                                              memPermuteOp.getDstOrderAttr(), memPermuteOp.getMemPermAttr());

    // Create new Reshape
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(memPermuteOp, newMemPermuteOp.getOutput(), nullptr, false,
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
    patterns.add<MoveMemPermuteThroughOp<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<MovePermuteQuantizeThroughOp<IE::MultiplyOp>>(&ctx, _log);
    patterns.add<MoveThroughShapeCast>(&ctx, _log);
    patterns.add<MoveMemPermuteThroughReshape>(&ctx, _log);
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
