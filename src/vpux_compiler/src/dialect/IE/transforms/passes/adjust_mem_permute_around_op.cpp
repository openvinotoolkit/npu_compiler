//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTMEMPERMUTEAROUNDOP
#define GEN_PASS_DEF_ADJUSTMEMPERMUTEAROUNDOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// Common Utils
//

mlir::Operation* getSupportedInputPermuteLikeOp(mlir::Value input) {
    auto inputOp = input.getDefiningOp();
    if (mlir::isa_and_nonnull<IE::MemPermuteOp, IE::PermuteQuantizeOp>(inputOp) && inputOp->hasOneUse()) {
        return inputOp;
    }
    return nullptr;
}

mlir::AffineMap getMemPermFromPermuteLikeOp(mlir::Operation* op) {
    if (auto memPermuteOp = mlir::dyn_cast<IE::MemPermuteOp>(op)) {
        return memPermuteOp.getMemPerm();

    } else if (auto permuteQuantizeOp = mlir::dyn_cast<IE::PermuteQuantizeOp>(op)) {
        return permuteQuantizeOp.getMemPerm();
    }
    VPUX_THROW("Unexpected op type at '{0}'", op->getLoc());
}

IE::MemPermuteOp getSupportedOutputMemPermute(mlir::Value output) {
    if (!output.hasOneUse()) {
        return nullptr;
    }
    auto outputMemPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(*output.getUsers().begin());
    if (outputMemPermuteOp == nullptr) {
        return nullptr;
    }
    return outputMemPermuteOp;
}

// Calculate the number of non-trivial permutes around the eltwise if inserting
// mempermutes with given permutation for all inputs of the layerOp
int64_t calcNumNonTrivialPermutesAroundEltwiseWithMemPerm(mlir::Operation* layerOp, mlir::AffineMap newMemPerm) {
    auto ctx = layerOp->getContext();
    auto idMap = mlir::AffineMap::getMultiDimIdentityMap(newMemPerm.getNumDims(), ctx);
    int64_t totalNumNonTrivialPermutes = 0;
    // calculate number of mempermutes on input side
    for (auto input : layerOp->getOperands()) {
        auto inputMemPermuteOp = getSupportedInputPermuteLikeOp(input);
        mlir::AffineMap inMemPerm = idMap;
        mlir::Value permuteInput = input;
        if (inputMemPermuteOp != nullptr) {
            inMemPerm = getMemPermFromPermuteLikeOp(inputMemPermuteOp);
            permuteInput = inputMemPermuteOp->getOperand(0);
        }
        const auto inMemShape = getMemShape(permuteInput);
        const auto newInMemPerm = newMemPerm.compose(inMemPerm);
        if (!isTrivialPermute(inMemShape, newInMemPerm)) {
            totalNumNonTrivialPermutes++;
        }
    }
    // calculate number of mempermutes on output side
    auto outMemPerm = idMap;
    if (layerOp->hasOneUse()) {
        auto outputMemPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(*layerOp->getUsers().begin());
        if (outputMemPermuteOp != nullptr) {
            outMemPerm = outputMemPermuteOp.getMemPerm();
        }
    }
    const auto newMemShape = applyPerm(getMemShape(layerOp->getResult(0)), newMemPerm);
    const auto newOutMemPerm = outMemPerm.compose(mlir::inversePermutation(newMemPerm));
    if (!isTrivialPermute(newMemShape, newOutMemPerm)) {
        totalNumNonTrivialPermutes++;
    }
    return totalNumNonTrivialPermutes;
}

mlir::AffineMap getBestMemPerm(mlir::Operation* origOp, vpux::NDTypeInterface outType, mlir::AffineMap idMap,
                               Logger log) {
    const auto outOrder = outType.getDimsOrder();
    auto bestMemPerm = idMap;
    auto bestNumNonTrivialPermutes = calcNumNonTrivialPermutesAroundEltwiseWithMemPerm(origOp, bestMemPerm);
    const auto checkBetterMemPerm = [&](mlir::AffineMap newMemPerm,
                                        int64_t origNumMemPermutes) -> std::optional<int64_t> {
        const auto numNonTrivialPermutes = calcNumNonTrivialPermutesAroundEltwiseWithMemPerm(origOp, newMemPerm);
        if (numNonTrivialPermutes >= origNumMemPermutes) {
            return std::nullopt;
        }
        return numNonTrivialPermutes;
    };

    // try with input permutes
    for (auto input : origOp->getOperands()) {
        // input order should be same as output order
        auto inOrder = DimsOrder::fromValue(input);
        if (inOrder != outOrder) {
            return idMap;
        }
        // get input mempermute op
        auto inputMemPermuteOp = getSupportedInputPermuteLikeOp(input);
        if (inputMemPermuteOp == nullptr) {
            continue;
        }

        // if there are unfused MemPermute ops on the input chain, return to give IE::MemPermuteOp Canonicalization a
        // chance to be executed
        auto inputMemPermuteParentOp = inputMemPermuteOp->getOperand(0).getDefiningOp();
        if (mlir::isa_and_nonnull<IE::MemPermuteOp>(inputMemPermuteParentOp)) {
            log.trace("Unfused MemPermute ops on input chain");
            return idMap;
        }
        // need to permute back to input of the parent mempermute
        auto memPerm = getMemPermFromPermuteLikeOp(inputMemPermuteOp);
        auto inversedMemPerm = mlir::inversePermutation(memPerm);
        auto betterNumMemPermutes = checkBetterMemPerm(inversedMemPerm, bestNumNonTrivialPermutes);
        if (betterNumMemPermutes.has_value()) {
            bestMemPerm = inversedMemPerm;
            bestNumNonTrivialPermutes = betterNumMemPermutes.value();
        }
    }

    // try with output permute
    auto outputMemPermuteOp = getSupportedOutputMemPermute(origOp->getResult(0));
    if (outputMemPermuteOp != nullptr) {
        // if there are unfused MemPermute ops on the output chain, return to give IE::MemPermuteOp Canonicalization a
        // chance to be executed
        auto outputMemPermuteChildOp = *outputMemPermuteOp.getOutput().getUsers().begin();
        if (mlir::isa_and_nonnull<IE::MemPermuteOp>(outputMemPermuteChildOp)) {
            log.trace("Unfused MemPermute ops on output chain");
            return idMap;
        }

        auto memPerm = outputMemPermuteOp.getMemPerm();
        auto betterNumMemPermutes = checkBetterMemPerm(memPerm, bestNumNonTrivialPermutes);
        if (betterNumMemPermutes.has_value()) {
            bestMemPerm = memPerm;
            bestNumNonTrivialPermutes = betterNumMemPermutes.value();
        }
    }

    return bestMemPerm;
}

//
// AdjustForEltwise
//
// This pattern tries to adjust the mempermutes around an eltwise to find the solution
// with least number of nontrivial permutes
class AdjustForEltwise final : public mlir::OpInterfaceRewritePattern<IE::LayerOpInterface> {
public:
    AdjustForEltwise(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayerOpInterface>(ctx), _log(log) {
        setDebugName("AdjustForEltwise");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AdjustForEltwise::matchAndRewrite(IE::LayerOpInterface origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto ctx = getContext();

    if (!origOp->hasTrait<IE::EltwiseOp>()) {
        return matchFailed(rewriter, origOp, "LayerOp is not Eltwise");
    }

    const auto outType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto rank = outType.getRank();
    const auto idMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<unsigned>(rank), getContext());

    auto bestMemPerm = getBestMemPerm(origOp.getOperation(), outType, idMap, _log);
    if (bestMemPerm == idMap) {
        return matchFailed(rewriter, origOp, "Already the best solution");
    }

    rewriter.startOpModification(origOp);
    rewriter.setInsertionPoint(origOp);

    // add permutes to inputs
    const auto origOrder = DimsOrder::fromValue(origOp->getResult(0));
    const auto newOrder = applyPermutation(origOrder, DimsOrder::fromAffineMap(bestMemPerm));

    if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(origOp.getOperation())) {
        auto orderInfo = iface.getLayoutInfo();
        orderInfo.setInput(0, newOrder);
        iface.inferLayoutInfo(orderInfo, /*seOpsEnabled=*/false, /*seExperimentalOpsEnabled=*/false);
        if (orderInfo.getInput(0) != newOrder || orderInfo.getOutput(0) != newOrder) {
            return matchFailed(rewriter, origOp, "New order could not be supported");
        }
    }

    for (auto& inputOperand : origOp->getOpOperands()) {
        auto inMemPermuteOp = rewriter.createOrFold<IE::MemPermuteOp>(
                takeOpLoc(origOp, llvm::formatv("_input_{0}", inputOperand.getOperandNumber())), inputOperand.get(),
                newOrder.toAffineMap(ctx), bestMemPerm);
        inputOperand.set(inMemPermuteOp);
    }

    // change output type of layerOp
    auto output = origOp->getOpResult(0);
    const auto origType = mlir::cast<vpux::NDTypeInterface>(output.getType());
    const auto newType = inferNewTypeWithMemPerm(origType, bestMemPerm, newOrder);
    output.setType(newType);

    // add permutes to output
    rewriter.setInsertionPointAfter(origOp);
    auto outMemPermuteOp = rewriter.create<IE::MemPermuteOp>(
            takeOpLoc(origOp, "_output"), output, origOrder.toAffineMap(ctx), mlir::inversePermutation(bestMemPerm));
    output.replaceAllUsesExcept(outMemPermuteOp.getOutput(), outMemPermuteOp);

    rewriter.finalizeOpModification(origOp);

    return mlir::success();
}

//
// AdjustForTile
//
// This pattern tries to move the permutes after tileOp up if it will become
// a trivial permute after the adjustment
class AdjustForTile final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    AdjustForTile(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
        setDebugName("AdjustForTile");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AdjustForTile::matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto ctx = getContext();

    if (!origOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "TileOp has multiple uses");
    }

    auto outputPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(*origOp->getUsers().begin());
    if (outputPermuteOp == nullptr) {
        return matchFailed(rewriter, origOp, "No MemPermuteOp found");
    }

    auto tileInType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto tileInMemShape = tileInType.getMemShape();

    auto memPerm = outputPermuteOp.getMemPerm();
    if (!isTrivialPermute(tileInMemShape, memPerm)) {
        return matchFailed(rewriter, origOp, "Not beneficial moving MemPermute up");
    }

    auto repeatsValues = origOp.getRepeatsValues();
    if (!repeatsValues.has_value()) {
        return matchFailed(rewriter, origOp, "No repeats values found, please run canonicalizer before this pass");
    }

    auto dstOrder = DimsOrder::fromAffineMap(outputPermuteOp.getDstOrder());
    auto newPermuteOutType = inferNewTypeWithMemPerm(tileInType, memPerm, dstOrder);
    auto newPermuteOp = rewriter.create<IE::PermuteCastOp>(outputPermuteOp->getLoc(), newPermuteOutType,
                                                           origOp.getInput(), dstOrder.toAffineMap(ctx), memPerm);

    auto origOrder = tileInType.getDimsOrder();
    auto repeatsOnOrigShape = Shape(parseIntArrayAttr<int64_t>(repeatsValues.value()));
    auto repeatsOnOrigMemShape = origOrder.toMemoryOrder(repeatsOnOrigShape);
    auto repeatsOnNewMemShape = applyPerm(repeatsOnOrigMemShape, memPerm);
    auto repeatsOnNewShape = dstOrder.toLogicalOrder(repeatsOnNewMemShape);
    auto newTileOutType = outputPermuteOp.getOutput().getType();
    auto newTileOp = rewriter.create<IE::TileOp>(origOp->getLoc(), newTileOutType, newPermuteOp.getOutput(), nullptr,
                                                 getIntArrayAttr(ctx, repeatsOnNewShape));

    outputPermuteOp.replaceAllUsesWith(newTileOp.getOutput());

    return mlir::success();
}

//
// AdjustForConvert
//
// This pattern tries to move the permuteCast after convertOp if the convertOp is the last Operation before return.
// It will give the chance to enable vertical fusion between convertOp and the previous layer.
// Note that this pattern can be removed after we can support permuteCast for vertical fusion E#106960
class AdjustForConvert final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    AdjustForConvert(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("AdjustForConvert");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/*
Convert subgraph:
       |                  |
   PermuteCast       ConvertOp
       |                  |
    ConvertOp    =>  PermuteCast
       |                  |
    ReturnOp           ReturnOp
*/
mlir::LogicalResult AdjustForConvert::matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto usedByReturnOp = llvm::all_of(origOp->getUsers(), [](const auto& user) {
        return mlir::isa<mlir::func::ReturnOp>(user);
    });
    if (!usedByReturnOp) {
        return matchFailed(rewriter, origOp, "Used by non-return ops");
    }

    auto inputPermuteCastOp = mlir::dyn_cast_or_null<IE::PermuteCastOp>(origOp.getInput().getDefiningOp());
    if (inputPermuteCastOp == nullptr) {
        return matchFailed(rewriter, origOp, "No PermuteCastOp found");
    }
    if (!inputPermuteCastOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "PermuteCastOp has other uses");
    }
    _log.trace("Move PermuteCast after convert op at '{0}'", origOp->getLoc());

    auto newConvertOp =
            rewriter.create<IE::ConvertOp>(origOp->getLoc(), inputPermuteCastOp.getInput(), origOp.getDstElemType());

    auto origPermuteCastOutType = mlir::cast<vpux::NDTypeInterface>(inputPermuteCastOp.getOutput().getType());
    auto newPermuteCastOutType = origPermuteCastOutType.changeElemType(origOp.getDstElemType());
    auto newPermuteCastOp =
            rewriter.create<IE::PermuteCastOp>(origOp->getLoc(), newPermuteCastOutType, newConvertOp.getOutput(),
                                               inputPermuteCastOp.getDstOrder(), inputPermuteCastOp.getMemPerm());

    rewriter.replaceOp(origOp, newPermuteCastOp);
    rewriter.eraseOp(inputPermuteCastOp);
    return mlir::success();
}

//
// AdjustForSoftmax
//
// This pattern tries to move the PermuteCast after Softmax when the previous op is Eltwise
// It will give the chance to enable vertical fusion between Eltwise and SoftMax.
// Note that this pattern can be removed after we can support PermuteCast for vertical fusion E#106960
class AdjustForSoftmax final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    AdjustForSoftmax(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        setDebugName("AdjustForSoftmax");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/*
Convert subgraph:

   src1   src2            src1           src2
     \    /                 \             /
    Eltwise                (ShapeCast) (ShapeCast)
       |                       \       /
   (ShapeCast)                  Eltwise
       |                           |
   PermuteCast     =>           SoftmaxOp
       |                           |
    SoftmaxOp                  PermuteCast
       |                           |
   AffineReshape              AffineReshape
*/
mlir::LogicalResult AdjustForSoftmax::matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());
    auto permuteCastOp = origOp->getOperand(0).getDefiningOp<IE::PermuteCastOp>();
    if (permuteCastOp == nullptr || !permuteCastOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "no compatible permute op found before {0} at {1}", origOp->getName(),
                           origOp->getLoc());
    }
    // check whether memshape is consistent after the permute
    // e.g. for memshape [1,12,512,1] and [1,12,1,512], though there's no actual
    // mempermute, the softmax axis is not consistent at the last memory dimension
    if (getMemShape(permuteCastOp.getInput()) != getMemShape(permuteCastOp.getResult())) {
        return matchFailed(rewriter, origOp, "the memshape should be consistent after permute for {0} at {1}",
                           origOp->getName(), origOp->getLoc());
    }

    auto eltwiseOp = permuteCastOp->getOperand(0).getDefiningOp();
    if (eltwiseOp == nullptr) {
        return matchFailed(rewriter, origOp, "no compatible eltwise op found before {0} at {1}", origOp->getName(),
                           origOp->getLoc());
    }
    bool hasShapeCastBetween = false;
    if (mlir::isa<IE::ShapeCastOp>(eltwiseOp) && eltwiseOp->hasOneUse()) {
        eltwiseOp = eltwiseOp->getOperand(0).getDefiningOp();
        hasShapeCastBetween = true;
    }
    if (!eltwiseOp->hasTrait<IE::EltwiseOp>() || !eltwiseOp->hasOneUse()) {
        return matchFailed(rewriter, origOp, "no compatible eltwise op found before {0} at {1}", origOp->getName(),
                           origOp->getLoc());
    }

    _log.trace("[{0}] start to move PermuteCast for {1} at {2}", this->getDebugName(), origOp->getName(),
               origOp->getLoc());
    auto ctx = rewriter.getContext();
    // check whether the original softmax axis is optimal (the inner most dimension in memory)
    // if so, move the premute after and change the softmax to the correct axis under new order
    const auto axis = origOp.getAxisInd();
    const auto origMemOrder = DimsOrder::fromValue(origOp->getResult(0));
    const auto origMemOrderVec = to_small_vector(origMemOrder.toPermutation() | transformed([](Dim dim) {
                                                     return checked_cast<int64_t>(dim.ind());
                                                 }));
    if (axis != origMemOrderVec.back()) {
        return matchFailed(rewriter, origOp, "the axis is not optimal for {0} at {1}", origOp->getName(),
                           origOp->getLoc());
    }

    // get optimal order for softmax with new mem order
    const auto newMemOrder = DimsOrder::fromValue(eltwiseOp->getResult(0));
    const auto newMemOrderVec = to_small_vector(newMemOrder.toPermutation() | transformed([](Dim dim) {
                                                    return checked_cast<int64_t>(dim.ind());
                                                }));
    const auto optimalAxisAttr = getIntAttr(ctx, newMemOrderVec.back());

    // if there's shapecast between eltwise and softmax, we need to propagate the shapecast before the eltwise by
    // adding shapecast for all inputs of the eltwise, and update the eltwise for the new inputs
    rewriter.startOpModification(eltwiseOp);
    if (hasShapeCastBetween) {
        auto shapeCastOp = mlir::dyn_cast<IE::ShapeCastOp>(*eltwiseOp->getResult(0).getUsers().begin());
        if (shapeCastOp == nullptr) {
            return matchFailed(rewriter, shapeCastOp, "unexpected op (should be Shapecast) found: {0} at {1}",
                               shapeCastOp->getName(), shapeCastOp->getLoc());
        }
        rewriter.setInsertionPoint(eltwiseOp);
        for (auto& eltwiseOpOperand : eltwiseOp->getOpOperands()) {
            auto newShapeCastOp = rewriter.create<IE::ShapeCastOp>(shapeCastOp->getLoc(), eltwiseOpOperand.get(),
                                                                   shapeCastOp.getShapeAttr());
            eltwiseOpOperand.set(newShapeCastOp.getResult());
        }
        vpux::inferReturnTypes(eltwiseOp, vpux::InferShapedTypeMode::ALL);
    }
    rewriter.setInsertionPointAfter(eltwiseOp);
    auto newSoftMaxOp =
            rewriter.create<IE::SoftMaxOp>(origOp->getLoc(), eltwiseOp->getResult(0), optimalAxisAttr, nullptr);
    auto newPermuteCastOp =
            rewriter.create<IE::PermuteCastOp>(permuteCastOp->getLoc(), newSoftMaxOp.getOutput(),
                                               permuteCastOp.getDstOrderAttr(), permuteCastOp.getMemPermAttr());

    rewriter.replaceOp(origOp, newPermuteCastOp);
    rewriter.finalizeOpModification(eltwiseOp);

    return mlir::success();
}

//
// AdjustForConcat
//

class AdjustForConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    AdjustForConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("AdjustForConcatSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
//    input1      input2       BlockArgument
//      |            |             |
//  MemPermute1  MemPermute2       |
//      |            |             |
//   -----------------------------------
//  |              Concat               |
//   -----------------------------------
//                   |
//              MemPermute3
//                   |
//                 Output
//
//
//  Propagate MemPermute3 through Concat:
//
//
//    input1      input2       BlockArgument
//      |            |             |
// PermuteCast1 PermuteCast2   MemPermute
//      |            |             |
//   -----------------------------------
//  |              Concat               |
//   -----------------------------------
//                   |
//                 Output
//

mlir::LogicalResult AdjustForConcat::matchAndRewrite(IE::ConcatOp concatOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] start to move MemPermute for {1} at {2}", this->getDebugName(), concatOp->getName(),
               concatOp->getLoc());

    if (!concatOp->hasOneUse()) {
        return mlir::failure();
    }

    auto outMemPermuteOp = mlir::dyn_cast<IE::MemPermuteOp>(*concatOp->getUsers().begin());
    if (outMemPermuteOp == nullptr) {
        return mlir::failure();
    }
    const auto outPermuteMemPerm = outMemPermuteOp.getMemPerm();

    auto isFusedPermuteTrivial = [&](IE::MemPermuteOp currentOp) {
        // Check if the fused MemPermute is trivial.
        const auto inMemShape = getMemShape(currentOp.getInput());
        const auto newMemPerm = outPermuteMemPerm.compose(currentOp.getMemPerm());
        return isTrivialPermute(inMemShape, newMemPerm);
    };

    _log.trace("start checking inputs");

    const auto concatOputShape = getShape(concatOp.getOutput());
    std::optional<Dim> concatAxis;
    int inputMemPermuteOpNum = 0;
    for (const auto& input : concatOp.getInputs()) {
        // only support concatenating on single dimension
        auto axis = vpux::IE::getSingleDiffAxis(getShape(input), concatOputShape);
        if (!axis.has_value()) {
            return mlir::failure();
        }
        if (!concatAxis.has_value()) {
            concatAxis = axis.value();
        }
        if (concatAxis.value() != axis.value()) {
            return mlir::failure();
        }

        if (auto inPermuteOp = input.getDefiningOp<IE::MemPermuteOp>()) {
            if (!isFusedPermuteTrivial(inPermuteOp)) {
                _log.trace("Can not convert to PermuteCast or fold them");
                return mlir::failure();
            }
            ++inputMemPermuteOpNum;
            continue;
        }

        if (mlir::isa<mlir::BlockArgument>(input)) {
            continue;
        }

        _log.trace("There are ops other than MemPermuteOp and BlockArgument");
        return mlir::failure();
    }

    if (inputMemPermuteOpNum == 0) {
        _log.trace("Inputs of Concat are all BlockArgument");
        return mlir::failure();
    }

    const auto outPermuteInOrder = DimsOrder::fromValue(outMemPermuteOp.getInput());
    const auto outPermuteOutOrder = DimsOrder::fromValue(outMemPermuteOp.getOutput());
    const auto concatInferDim =
            inferDimAfterPermutation(concatAxis.value(), outPermuteInOrder, outPermuteOutOrder, outPermuteMemPerm);

    SmallVector<mlir::Value> newConcatInputs;
    for (const auto& item : concatOp.getInputs() | indexed) {
        const auto& input = item.value();
        const auto& idx = item.index();
        auto newInPermuteOp =
                rewriter.createOrFold<IE::MemPermuteOp>(takeOpLoc(concatOp, llvm::formatv("_input_{0}", idx)), input,
                                                        outMemPermuteOp.getDstOrder(), outPermuteMemPerm);
        newConcatInputs.push_back(newInPermuteOp);
    }

    auto newConcat = rewriter.create<IE::ConcatOp>(concatOp->getLoc(), newConcatInputs, concatInferDim);
    rewriter.replaceOp(outMemPermuteOp, newConcat);

    return mlir::success();
}

//
// AdjustForNCEEltwise
//

template <typename ConcreteOp>
class AdjustForNCEEltwise final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    AdjustForNCEEltwise(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

private:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult AdjustForNCEEltwise<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto ctx = origOp.getContext();

    auto output = origOp->getResult(0);
    const auto outType = mlir::cast<vpux::NDTypeInterface>(output.getType());
    const auto rank = outType.getRank();
    const auto idMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<unsigned>(rank), ctx);

    auto bestMemPerm = getBestMemPerm(origOp.getOperation(), outType, idMap, _log);
    if (bestMemPerm == idMap) {
        return matchFailed(rewriter, origOp, "Already the best solution");
    }

    rewriter.startOpModification(origOp);
    rewriter.setInsertionPoint(origOp);

    auto shape = getShape(origOp->getResult(0));
    const auto origOrder = DimsOrder::fromValue(origOp->getResult(0));
    const auto newOrder = applyPermutation(origOrder, DimsOrder::fromAffineMap(bestMemPerm));

    const auto canConvertToNCE = [&]() {
        auto input1 = origOp->getOperand(0);
        auto input2 = origOp->getOperand(1);
        auto input1Type = mlir::cast<vpux::NDTypeInterface>(input1.getType());
        auto input2Type = mlir::cast<vpux::NDTypeInterface>(input2.getType());

        if (input1Type.getShape() != input2Type.getShape()) {
            return false;
        }

        auto isQuantizedInput = [](vpux::NDTypeInterface value) {
            return mlir::isa<mlir::quant::QuantizedType>(value.getElementType());
        };

        const auto mixedInputs = (isQuantizedInput(input1Type) && !isQuantizedInput(input2Type)) ||
                                 (!isQuantizedInput(input1Type) && isQuantizedInput(input2Type));
        if (mixedInputs) {
            return false;
        }

        if ((shape.size() != 4) || (shape[Dims4D::Act::N] != 1)) {
            return false;
        }

        if (auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp.getOperation())) {
            auto alignment = iface.getOutputChannelAlignment();
            auto memShape = newOrder.toMemoryOrder(shape);
            auto innerMostDimSize = memShape.back();
            return innerMostDimSize % alignment == 0;
        }

        return true;
    };

    if (!canConvertToNCE()) {
        return matchFailed(rewriter, origOp, "Cannot convert to NCE");
    }

    // Add permutes to inputs
    const auto expectedLayout = DimsOrder::NHWC;
    const auto dstOrder = expectedLayout.toAffineMap(ctx);
    int index = 0;  // Initialize a counter for unique identifiers
    for (auto& inputOperand : origOp->getOpOperands()) {
        auto inMemPermuteOp =
                rewriter.create<IE::MemPermuteOp>(appendLoc(origOp->getLoc(), "_input_permute" + std::to_string(index)),
                                                  inputOperand.get(), newOrder.toAffineMap(ctx), bestMemPerm);
        // Create permute cast to satisfy type requirement of NCE Eltwise
        auto inputPermuteCastOp = rewriter.create<IE::PermuteCastOp>(
                appendLoc(origOp->getLoc(), "_input_permute_cast" + std::to_string(index)),
                inMemPermuteOp->getResult(0), dstOrder, idMap);
        inputOperand.set(inputPermuteCastOp->getResult(0));
        index++;
    }

    // Change output type of layerOp
    auto originalElementType = outType.getElementType();
    auto newInputType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    auto newOutputType = newInputType.changeElemType(originalElementType);
    output.setType(newOutputType);

    rewriter.setInsertionPointAfter(origOp);
    auto outputPermuteCastOp = rewriter.create<IE::PermuteCastOp>(appendLoc(origOp->getLoc(), "_output_permute_cast"),
                                                                  output, newOrder.toAffineMap(ctx), idMap);
    // Add permutes to output
    auto outMemPermuteOp = rewriter.create<IE::MemPermuteOp>(
            appendLoc(origOp->getLoc(), "_output_permute"), outputPermuteCastOp->getResult(0),
            origOrder.toAffineMap(ctx), mlir::inversePermutation(bestMemPerm));
    output.replaceAllUsesExcept(outMemPermuteOp.getOutput(), outputPermuteCastOp);

    rewriter.finalizeOpModification(origOp);

    return mlir::success();
}

//
// AdjustMemPermuteAroundOpPass
//

class AdjustMemPermuteAroundOpPass final : public IE::impl::AdjustMemPermuteAroundOpBase<AdjustMemPermuteAroundOpPass> {
public:
    explicit AdjustMemPermuteAroundOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustMemPermuteAroundOpPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AdjustForEltwise>(&ctx, _log);
    patterns.add<AdjustForTile>(&ctx, _log);
    patterns.add<AdjustForConvert>(&ctx, _log);
    patterns.add<AdjustForSoftmax>(&ctx, _log);
    patterns.add<AdjustForConcat>(&ctx, _log);
    patterns.add<AdjustForNCEEltwise<IE::AddOp>>(&ctx, _log);
    IE::MemPermuteOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustMemPermuteAroundOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustMemPermuteAroundOpPass(Logger log) {
    return std::make_unique<AdjustMemPermuteAroundOpPass>(log);
}
