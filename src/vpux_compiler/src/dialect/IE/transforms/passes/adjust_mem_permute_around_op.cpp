//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <climits>

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
    if (mlir::isa_and_nonnull<IE::MemPermuteOp, IE::PermuteQuantizeOp>(inputOp)) {
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

// Returns a sibling MemPermute op that can be reused by consumerOp.
// A sibling is another user of the same input with mem_perm equal to expectedMemPerm.
// Reuse is constrained to the same block so the sibling can be moved before consumerOp if needed.
std::optional<IE::MemPermuteOp> getReusableSiblingPermuteOp(mlir::Value input, mlir::Operation* consumerOp,
                                                            mlir::AffineMap expectedMemPerm) {
    for (auto* userOp : input.getUsers()) {
        if (userOp == consumerOp) {
            continue;
        }
        if (!mlir::isa<IE::MemPermuteOp>(userOp)) {
            continue;
        }
        if (userOp->getBlock() != consumerOp->getBlock()) {
            continue;
        }

        const auto siblingMemPerm = getMemPermFromPermuteLikeOp(userOp);
        if (siblingMemPerm != expectedMemPerm) {
            continue;
        }
        return mlir::cast<IE::MemPermuteOp>(userOp);
    }
    return std::nullopt;
}

// Returns the result value of a reusable sibling MemPermute without changing IR order.
// Use this helper in analysis/cost-model code paths where mutation is not allowed.
std::optional<mlir::Value> getReusableSiblingPermuteValue(mlir::Value input, mlir::Operation* consumerOp,
                                                          mlir::AffineMap expectedMemPerm) {
    auto reusableSibling = getReusableSiblingPermuteOp(input, consumerOp, expectedMemPerm);
    if (!reusableSibling.has_value()) {
        return std::nullopt;
    }
    return reusableSibling.value()->getResult(0);
}

// Returns the result value of a reusable sibling MemPermute and ensures SSA-safe ordering.
// If the sibling is located after consumerOp in the same block, it is moved before consumerOp.
std::optional<mlir::Value> getReusableSiblingPermuteValueWithMove(mlir::Value input, mlir::Operation* consumerOp,
                                                                  mlir::AffineMap expectedMemPerm,
                                                                  mlir::PatternRewriter& rewriter) {
    auto reusableSibling = getReusableSiblingPermuteOp(input, consumerOp, expectedMemPerm);
    if (!reusableSibling.has_value()) {
        return std::nullopt;
    }

    auto siblingPermuteOp = reusableSibling.value();
    if (!siblingPermuteOp->isBeforeInBlock(consumerOp)) {
        rewriter.moveOpBefore(siblingPermuteOp, consumerOp);
    }

    return siblingPermuteOp->getResult(0);
}

// Check whether an inserted MemPermute on `input` can be propagated upstream through
// a single-use chain of PermuteCast ops and absorbed at an eltwise whose inputs are
// all effectively 1D (at most one non-unit memory dim), making any permute trivial.
bool canAbsorbPermuteInUpstreamChain(mlir::Value input) {
    auto* currentOp = input.getDefiningOp();
    while (currentOp != nullptr && currentOp->hasOneUse() && mlir::isa<IE::PermuteCastOp>(currentOp)) {
        currentOp = currentOp->getOperand(0).getDefiningOp();
    }

    if (currentOp == nullptr || !currentOp->hasOneUse() || !currentOp->hasTrait<IE::EltwiseOp>() ||
        mlir::isa<IE::DynamicDequantizeOp>(currentOp)) {
        return false;
    }

    return llvm::all_of(currentOp->getOperands(), [](mlir::Value operand) {
        const auto memShape = getMemShape(operand);
        return llvm::count_if(memShape, [](int64_t d) {
                   return d != 1;
               }) <= 1;
    });
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
        } else {
            // No input-side permute exists on this branch. If a sibling already has an exact MemPermute (same
            // mem_perm), we can reuse that value and avoid adding a new permute for this operand, so it contributes no
            // extra cost.
            if (getReusableSiblingPermuteValue(input, layerOp, newMemPerm).has_value()) {
                continue;
            }

            // If input comes from a chain of view-like ops rooted at an eltwise with 1D
            // inputs, the inserted MemPermute can be propagated upstream and absorbed.
            if (canAbsorbPermuteInUpstreamChain(input)) {
                continue;
            }
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
        return matchFailed(_log, rewriter, origOp, "LayerOp is not Eltwise");
    }
    if (mlir::isa_and_present<IE::DynamicDequantizeOp>(origOp)) {
        return matchFailed(_log, rewriter, origOp, "DynamicDequantizeOp should not adjust MemPermute");
    }

    const auto outType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto rank = outType.getRank();
    const auto idMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<unsigned>(rank), getContext());

    auto bestMemPerm = getBestMemPerm(origOp.getOperation(), outType, idMap, _log);
    if (bestMemPerm == idMap) {
        return matchFailed(_log, rewriter, origOp, "Already the best solution");
    }

    // add permutes to inputs
    const auto origOrder = DimsOrder::fromValue(origOp->getResult(0));
    const auto newOrder = applyPermutation(origOrder, DimsOrder::fromAffineMap(bestMemPerm));

    if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(origOp.getOperation())) {
        auto orderInfo = iface.getLayoutInfo();
        orderInfo.setInput(0, newOrder);
        iface.inferLayoutInfo(orderInfo);
        if (orderInfo.getInput(0) != newOrder || orderInfo.getOutput(0) != newOrder) {
            return matchFailed(_log, rewriter, origOp, "New order could not be supported");
        }
    }

    rewriter.startOpModification(origOp);
    rewriter.setInsertionPoint(origOp);

    for (auto& inputOperand : origOp->getOpOperands()) {
        auto input = inputOperand.get();
        if (getSupportedInputPermuteLikeOp(input) == nullptr) {
            auto reusableSibling =
                    getReusableSiblingPermuteValueWithMove(input, origOp.getOperation(), bestMemPerm, rewriter);
            if (reusableSibling.has_value()) {
                auto reusableValue = reusableSibling.value();
                // If the reusable sibling is not in the new order, we need to insert a PermuteCast to convert it to the
                // new order
                if (DimsOrder::fromValue(reusableValue) != newOrder) {
                    auto siblingMemPermute = reusableValue.getDefiningOp<IE::MemPermuteOp>();
                    VPUX_THROW_WHEN(siblingMemPermute == nullptr,
                                    "Expected reusable sibling to be produced by IE.MemPermute");

                    const auto reusableType = mlir::cast<vpux::NDTypeInterface>(reusableValue.getType());
                    const auto idMemPerm = mlir::AffineMap::getMultiDimIdentityMap(
                            checked_cast<unsigned>(reusableType.getRank()), ctx);
                    const auto castOutType = inferNewTypeWithMemPerm(reusableType, idMemPerm, newOrder);
                    auto reusedSiblingPermCastOp = IE::PermuteCastOp::create(
                            rewriter,
                            takeOpLoc(origOp, "reused_sibling_perm_cast_{0}", inputOperand.getOperandNumber()),
                            castOutType, reusableValue, newOrder.toAffineMap(ctx), idMemPerm);
                    reusableValue = reusedSiblingPermCastOp.getOutput();
                }

                inputOperand.set(reusableValue);
                continue;
            }
        }

        auto inMemPermuteOp =
                IE::MemPermuteOp::create(rewriter, takeOpLoc(origOp, "input_{0}", inputOperand.getOperandNumber()),
                                         input, newOrder.toAffineMap(ctx), bestMemPerm);
        inputOperand.set(inMemPermuteOp.getOutput());
    }

    // change output type of layerOp
    auto output = origOp->getOpResult(0);
    const auto origType = mlir::cast<vpux::NDTypeInterface>(output.getType());
    const auto newType = inferNewTypeWithMemPerm(origType, bestMemPerm, newOrder);
    output.setType(newType);

    // add permutes to output
    rewriter.setInsertionPointAfter(origOp);
    auto outMemPermuteOp = IE::MemPermuteOp::create(rewriter, takeOpLoc(origOp, "output"), output,
                                                    origOrder.toAffineMap(ctx), mlir::inversePermutation(bestMemPerm));
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
        return matchFailed(_log, rewriter, origOp, "TileOp has multiple uses");
    }

    auto outputPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(*origOp->getUsers().begin());
    if (outputPermuteOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "No MemPermuteOp found");
    }

    auto tileInType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto tileInMemShape = tileInType.getMemShape();

    auto memPerm = outputPermuteOp.getMemPerm();
    if (!isTrivialPermute(tileInMemShape, memPerm)) {
        return matchFailed(_log, rewriter, origOp, "Not beneficial moving MemPermute up");
    }

    auto repeatsValues = origOp.getRepeatsValues();
    auto dstOrder = DimsOrder::fromAffineMap(outputPermuteOp.getDstOrder());
    auto newPermuteOutType = inferNewTypeWithMemPerm(tileInType, memPerm, dstOrder);
    auto newPermuteOp = IE::PermuteCastOp::create(rewriter, outputPermuteOp->getLoc(), newPermuteOutType,
                                                  origOp.getInput(), dstOrder.toAffineMap(ctx), memPerm);

    auto origOrder = tileInType.getDimsOrder();
    auto repeatsOnOrigShape = Shape(parseIntArrayAttr<int64_t>(repeatsValues));
    auto repeatsOnOrigMemShape = origOrder.toMemoryOrder(repeatsOnOrigShape);
    auto repeatsOnNewMemShape = applyPerm(repeatsOnOrigMemShape, memPerm);
    auto repeatsOnNewShape = dstOrder.toLogicalOrder(repeatsOnNewMemShape);
    auto newTileOutType = outputPermuteOp.getOutput().getType();
    auto newTileOp = IE::TileOp::create(rewriter, origOp->getLoc(), newTileOutType, newPermuteOp.getOutput(),
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
        return matchFailed(_log, rewriter, origOp, "Used by non-return ops");
    }

    auto inputPermuteCastOp = mlir::dyn_cast_or_null<IE::PermuteCastOp>(origOp.getInput().getDefiningOp());
    if (inputPermuteCastOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "No PermuteCastOp found");
    }
    if (!inputPermuteCastOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "PermuteCastOp has other uses");
    }
    _log.trace("Move PermuteCast after convert op at '{0}'", origOp->getLoc());

    auto newConvertOp =
            IE::ConvertOp::create(rewriter, origOp->getLoc(), inputPermuteCastOp.getInput(), origOp.getDstElemType());

    auto origPermuteCastOutType = mlir::cast<vpux::NDTypeInterface>(inputPermuteCastOp.getOutput().getType());
    auto newPermuteCastOutType = origPermuteCastOutType.changeElemType(origOp.getDstElemType());
    auto newPermuteCastOp =
            IE::PermuteCastOp::create(rewriter, origOp->getLoc(), newPermuteCastOutType, newConvertOp.getOutput(),
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
        return matchFailed(_log, rewriter, origOp, "no compatible permute op found before {0} at {1}",
                           origOp->getName(), origOp->getLoc());
    }
    // check whether memshape is consistent after the permute
    // e.g. for memshape [1,12,512,1] and [1,12,1,512], though there's no actual
    // mempermute, the softmax axis is not consistent at the last memory dimension
    if (getMemShape(permuteCastOp.getInput()) != getMemShape(permuteCastOp.getResult())) {
        return matchFailed(_log, rewriter, origOp, "the memshape should be consistent after permute for {0} at {1}",
                           origOp->getName(), origOp->getLoc());
    }

    auto eltwiseOp = permuteCastOp->getOperand(0).getDefiningOp();
    if (eltwiseOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "no compatible eltwise op found before {0} at {1}",
                           origOp->getName(), origOp->getLoc());
    }
    bool hasShapeCastBetween = false;
    if (mlir::isa<IE::ShapeCastOp>(eltwiseOp) && eltwiseOp->hasOneUse()) {
        eltwiseOp = eltwiseOp->getOperand(0).getDefiningOp();
        hasShapeCastBetween = true;
    }
    if (!eltwiseOp->hasTrait<IE::EltwiseOp>() || !eltwiseOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "no compatible eltwise op found before {0} at {1}",
                           origOp->getName(), origOp->getLoc());
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
        return matchFailed(_log, rewriter, origOp, "the axis is not optimal for {0} at {1}", origOp->getName(),
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
    IE::ShapeCastOp shapeCastOp = nullptr;
    if (hasShapeCastBetween) {
        shapeCastOp = mlir::dyn_cast<IE::ShapeCastOp>(*eltwiseOp->getResult(0).getUsers().begin());
        if (shapeCastOp == nullptr) {
            return matchFailed(_log, rewriter, origOp,
                               "unexpected op (should be Shapecast) found between eltwise and {0} at {1}",
                               origOp->getName(), origOp->getLoc());
        }
    }

    rewriter.startOpModification(eltwiseOp);
    if (hasShapeCastBetween) {
        rewriter.setInsertionPoint(eltwiseOp);
        for (auto& eltwiseOpOperand : eltwiseOp->getOpOperands()) {
            auto newShapeCastOp = IE::ShapeCastOp::create(rewriter, shapeCastOp->getLoc(), eltwiseOpOperand.get(),
                                                          shapeCastOp.getShapeAttr());
            eltwiseOpOperand.set(newShapeCastOp.getResult());
        }
        vpux::inferReturnTypes(eltwiseOp, vpux::InferShapedTypeMode::ALL);
    }
    rewriter.setInsertionPointAfter(eltwiseOp);
    auto newSoftMaxOp = IE::SoftMaxOp::create(rewriter, origOp->getLoc(), eltwiseOp->getResult(0), optimalAxisAttr,
                                              nullptr, origOp.getDstElemTypeAttr(), origOp.getMaskAwareAttr());
    auto newPermuteCastOp = IE::PermuteCastOp::create(rewriter, permuteCastOp->getLoc(), newSoftMaxOp.getOutput(),
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
        auto newInPermuteOp = IE::MemPermuteOp::create(rewriter, takeOpLoc(concatOp, "input_{0}", idx), input,
                                                       outMemPermuteOp.getDstOrder(), outPermuteMemPerm);
        newConcatInputs.push_back(newInPermuteOp.getOutput());
    }

    auto newConcat = IE::ConcatOp::create(rewriter, concatOp->getLoc(), newConcatInputs, concatInferDim);
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
        return matchFailed(_log, rewriter, origOp, "Already the best solution");
    }

    auto firstInputOp = origOp->getOperand(0).getDefiningOp();
    if (firstInputOp != nullptr && llvm::all_of(origOp->getOperands(), [&](mlir::Value operand) {
            return operand.getDefiningOp() == firstInputOp;
        })) {
        return matchFailed(_log, rewriter, origOp, "All inputs come from the same permute");
    }

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
        return matchFailed(_log, rewriter, origOp, "Cannot convert to NCE");
    }

    rewriter.startOpModification(origOp);
    rewriter.setInsertionPoint(origOp);

    // Add permutes to inputs
    const auto expectedLayout = DimsOrder::NHWC;
    const auto dstOrder = expectedLayout.toAffineMap(ctx);
    int index = 0;  // Initialize a counter for unique identifiers
    for (auto& inputOperand : origOp->getOpOperands()) {
        auto input = inputOperand.get();
        mlir::Value permutedInput;
        bool reusedSiblingPermute = false;

        if (getSupportedInputPermuteLikeOp(input) == nullptr) {
            auto reusableSibling =
                    getReusableSiblingPermuteValueWithMove(input, origOp.getOperation(), bestMemPerm, rewriter);
            if (reusableSibling.has_value()) {
                permutedInput = reusableSibling.value();
                reusedSiblingPermute = true;
            }
        }

        auto bestInputMemPerm = bestMemPerm;
        if (!reusedSiblingPermute) {
            if (auto inputMemPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(input.getDefiningOp())) {
                const auto inMemShape = getMemShape(input);
                const auto newInMemPerm = bestMemPerm.compose(inputMemPermuteOp.getMemPerm());
                if (isTrivialPermute(inMemShape, newInMemPerm)) {
                    input = inputMemPermuteOp.getInput();
                    bestInputMemPerm = newInMemPerm;
                }
            }

            auto inMemPermuteOp = IE::MemPermuteOp::create(
                    rewriter, appendLoc(origOp->getLoc(), "input_permute" + std::to_string(index)), input,
                    newOrder.toAffineMap(ctx), bestInputMemPerm);
            permutedInput = inMemPermuteOp->getResult(0);
        }

        // Create permute cast to satisfy type requirement of NCE Eltwise
        auto inputPermuteCastOp = IE::PermuteCastOp::create(
                rewriter, appendLoc(origOp->getLoc(), "input_permute_cast" + std::to_string(index)), permutedInput,
                dstOrder, idMap);
        inputOperand.set(inputPermuteCastOp->getResult(0));
        index++;
    }

    // Change output type of layerOp
    auto originalElementType = outType.getElementType();
    auto newInputType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    auto newOutputType = newInputType.changeElemType(originalElementType);
    output.setType(newOutputType);

    rewriter.setInsertionPointAfter(origOp);
    auto outputPermuteCastOp = IE::PermuteCastOp::create(rewriter, appendLoc(origOp->getLoc(), "output_permute_cast"),
                                                         output, newOrder.toAffineMap(ctx), idMap);
    // Add permutes to output
    auto outMemPermuteOp = IE::MemPermuteOp::create(rewriter, appendLoc(origOp->getLoc(), "output_permute"),
                                                    outputPermuteCastOp->getResult(0), origOrder.toAffineMap(ctx),
                                                    mlir::inversePermutation(bestMemPerm));
    output.replaceAllUsesExcept(outMemPermuteOp.getOutput(), outputPermuteCastOp);

    rewriter.finalizeOpModification(origOp);

    return mlir::success();
}

//
// FusePermuteCastAroundEltwise
//
// Removes a PermuteCast+AffineReshape sandwich wrapped around a binary Eltwise op (IE.Multiply/IE.Add)
// when it is provably a no-op: each branch's pre-sandwich value has the same shape+order as the final
// PermuteCast's output, and the eltwise's actual (post-reshape) operands have identical shape+order
// (no real broadcast, and no order mismatch). AffineReshapeOp/PermuteCastOp cannot change element type,
// so element type is structurally preserved through the whole chain and needs no separate check.
template <typename ConcreteOp>
class FusePermuteCastAroundEltwise final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    FusePermuteCastAroundEltwise(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("FusePermuteCastAroundEltwise");
    }

private:
    // One matched input branch: operand <- AffineReshape <- PermuteCast <- newInput.
    struct InputChain {
        IE::AffineReshapeOp reshape;
        IE::PermuteCastOp permuteCast;
        mlir::Value newInput;
    };

    // The whole matched sandwich: two input branches plus the output branch origOp -> AffineReshape ->
    // PermuteCast -> newOutput.
    struct MatchedPattern {
        InputChain input1Chain;
        InputChain input2Chain;
        IE::AffineReshapeOp reshapeOut;
        IE::PermuteCastOp permuteCastOut;
        mlir::Value newOutput;
    };

    // Returns std::nullopt if the expected PermuteCast+AffineReshape sandwich pattern is not found around
    // origOp.
    std::optional<MatchedPattern> findPattern(ConcreteOp origOp) const;

    // Whether an Eltwise op with the given input/output values is a supported NCE.Eltwise candidate.
    bool canConvertEltwiseToNCE(mlir::Operation* op, mlir::Value input1, mlir::Value input2, mlir::Value output) const;

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
std::optional<typename FusePermuteCastAroundEltwise<ConcreteOp>::MatchedPattern>
FusePermuteCastAroundEltwise<ConcreteOp>::findPattern(ConcreteOp origOp) const {
    const auto matchInputChain = [](mlir::Value operand) -> std::optional<InputChain> {
        auto reshapeIn = operand.getDefiningOp<IE::AffineReshapeOp>();
        if (reshapeIn == nullptr || !reshapeIn->hasOneUse()) {
            return std::nullopt;
        }
        auto permuteCastIn = reshapeIn.getInput().getDefiningOp<IE::PermuteCastOp>();
        if (permuteCastIn == nullptr || !permuteCastIn->hasOneUse()) {
            return std::nullopt;
        }
        return InputChain{reshapeIn, permuteCastIn, permuteCastIn.getInput()};
    };

    auto input1Chain = matchInputChain(origOp.getInput1());
    if (!input1Chain.has_value()) {
        return std::nullopt;
    }
    auto input2Chain = matchInputChain(origOp.getInput2());
    if (!input2Chain.has_value()) {
        return std::nullopt;
    }

    // Match the output branch: origOp -> AffineReshape -> PermuteCast -> newOutput.
    if (!origOp->hasOneUse()) {
        return std::nullopt;
    }
    auto reshapeOut = mlir::dyn_cast<IE::AffineReshapeOp>(*origOp->getUsers().begin());
    if (reshapeOut == nullptr || !reshapeOut->hasOneUse()) {
        return std::nullopt;
    }
    auto permuteCastOut = mlir::dyn_cast<IE::PermuteCastOp>(*reshapeOut->getUsers().begin());
    if (permuteCastOut == nullptr) {
        return std::nullopt;
    }

    const auto newOutput = permuteCastOut.getOutput();

    // Core legality check: each branch's pre-sandwich value must match newOutput's type exactly.
    if (input1Chain->newInput.getType() != newOutput.getType() ||
        input2Chain->newInput.getType() != newOutput.getType()) {
        return std::nullopt;
    }

    return MatchedPattern{std::move(input1Chain.value()), std::move(input2Chain.value()), reshapeOut, permuteCastOut,
                          newOutput};
}

template <typename ConcreteOp>
bool FusePermuteCastAroundEltwise<ConcreteOp>::canConvertEltwiseToNCE(mlir::Operation* op, mlir::Value input1,
                                                                      mlir::Value input2, mlir::Value output) const {
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.nest().trace("{0}", msg.str());
    };
    return VPU::isNCEEltwiseSupported(op, mlir::cast<vpux::NDTypeInterface>(input1.getType()),
                                      mlir::cast<vpux::NDTypeInterface>(input2.getType()),
                                      mlir::cast<vpux::NDTypeInterface>(output.getType()),
                                      /*allowDifferentScales=*/true, /*allowDifferentZp=*/true,
                                      /*checkLayout=*/true, /*checkChannelAlignment=*/true, logCb);
}

template <typename ConcreteOp>
mlir::LogicalResult FusePermuteCastAroundEltwise<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    // Only strictly-binary eltwise ops are covered by the derivation above.
    if (origOp->getNumOperands() != 2) {
        return matchFailed(_log, rewriter, origOp, "Eltwise op is not strictly binary");
    }

    // No real broadcast, and no order mismatch, between the eltwise's actual operands.
    if (origOp.getInput1().getType() != origOp.getInput2().getType()) {
        return matchFailed(_log, rewriter, origOp,
                           "Eltwise operands have different types (real broadcast or order mismatch)");
    }

    // If origOp is already convertible to NCE as-is, fusing away the sandwich brings no benefit.
    if (canConvertEltwiseToNCE(origOp.getOperation(), origOp.getInput1(), origOp.getInput2(), origOp->getResult(0))) {
        return matchFailed(_log, rewriter, origOp, "Eltwise op is already convertible to NCE");
    }

    auto matched = findPattern(origOp);
    if (!matched.has_value()) {
        return matchFailed(_log, rewriter, origOp, "No matching PermuteCast+AffineReshape sandwich pattern found");
    }

    const auto v1 = matched->input1Chain.newInput;
    const auto v2 = matched->input2Chain.newInput;
    const auto newOutput = matched->newOutput;

    // Only fuse if doing so would actually make the eltwise op convertible to NCE.
    if (!canConvertEltwiseToNCE(origOp.getOperation(), v1, v2, newOutput)) {
        return matchFailed(_log, rewriter, origOp, "Fused eltwise op would still not be convertible to NCE");
    }

    _log.trace("[{0}] Fusing sandwich around '{1}' at '{2}'", this->getDebugName(), origOp->getName(),
               origOp->getLoc());

    // Rewrite: rewire operands directly onto V1/V2 and retype to newOutput's (already-proven-equal) type.
    rewriter.modifyOpInPlace(origOp, [&] {
        origOp->setOperand(0, v1);
        origOp->setOperand(1, v2);
        origOp->getResult(0).setType(newOutput.getType());
    });

    rewriter.replaceAllUsesWith(newOutput, origOp->getResult(0));

    return mlir::success();
}

//
// AdjustForDynamicDequantize
//
// This pattern tries to move the permutes before DynamicDequantizeOp.
class AdjustForDynamicDequantize final : public mlir::OpRewritePattern<IE::DynamicDequantizeOp> {
public:
    AdjustForDynamicDequantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DynamicDequantizeOp>(ctx), _log(log) {
        setDebugName("AdjustForDynamicDequantize");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AdjustForDynamicDequantize::matchAndRewrite(IE::DynamicDequantizeOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!origOp->hasOneUse()) {
        _log.trace("DynamicDequantizeOp has multiple uses");
        return mlir::failure();
    }

    auto outputPermuteOp = mlir::dyn_cast_if_present<IE::MemPermuteOp>(*origOp->getUsers().begin());
    if (outputPermuteOp == nullptr) {
        _log.trace("No MemPermuteOp found");
        return mlir::failure();
    }

    mlir::Value dynamicDequantizeInput = origOp.getInput();
    auto dynamicDequantizeInType = mlir::cast<vpux::NDTypeInterface>(dynamicDequantizeInput.getType());

    while (mlir::isa_and_present<IE::ViewLikeOpInterface>(dynamicDequantizeInput.getDefiningOp())) {
        dynamicDequantizeInput = dynamicDequantizeInput.getDefiningOp()->getOperand(0);
    }

    if (!mlir::isa_and_present<mlir::BlockArgument>(dynamicDequantizeInput)) {
        _log.trace("DynamicDequantizeOp input is not a BlockArgument");
        return mlir::failure();
    }

    auto quantizedType =
            mlir::dyn_cast_if_present<mlir::quant::QuantizedType>(dynamicDequantizeInType.getElementType());
    const auto bitWidth = quantizedType.getStorageTypeIntegralWidth();
    if (!vpux::Const::isSubByte(bitWidth)) {
        _log.trace("SubByte quantization is required");
        return mlir::failure();
    }

    const auto inputDynDequantShape = getShape(origOp.getInput());

    if (inputDynDequantShape.size() != 4) {
        _log.trace("Only support 4D input shape for DynamicDequantizeOp");
        return mlir::failure();
    }

    const auto elementsInOneByte = CHAR_BIT / bitWidth;

    const auto isNAlign =
            inputDynDequantShape[Dims4D::Act::N] == 1 || inputDynDequantShape[Dims4D::Act::N] % elementsInOneByte == 0;
    const auto isCAlign =
            inputDynDequantShape[Dims4D::Act::C] == 1 || inputDynDequantShape[Dims4D::Act::C] % elementsInOneByte == 0;
    const auto isHAlign =
            inputDynDequantShape[Dims4D::Act::H] == 1 || inputDynDequantShape[Dims4D::Act::H] % elementsInOneByte == 0;
    const auto isWAlign =
            inputDynDequantShape[Dims4D::Act::W] == 1 || inputDynDequantShape[Dims4D::Act::W] % elementsInOneByte == 0;
    if (!isNAlign || !isCAlign || !isHAlign || !isWAlign) {
        _log.trace("Not beneficial moving MemPermute up due to non-aligned shape");
        return mlir::failure();
    }

    // For sub-byte types, only HWC->WHC permutations (mergedMemPerm == {1,0,2}) are supported
    // by the DMA hardware. Other permutations on sub-byte PermuteDMAOps cannot be unrolled by
    // UnrollPermuteDMAPass and will cause a crash in ConvertVPUIP2VPUMI40XX.
    const auto isSubBytePermuteSupported = [](vpux::NDTypeInterface inType, mlir::AffineMap memPerm) -> bool {
        const auto inputMemShape = inType.getMemShape();
        // Source-dim indices for each non-unit output position, in output order.
        SmallVector<int64_t> nonUnitPerm;
        for (unsigned outIdx = 0; outIdx < memPerm.getNumResults(); ++outIdx) {
            const auto srcDim = static_cast<int64_t>(memPerm.getDimPosition(outIdx));
            if (inputMemShape[MemDim(srcDim)] != 1) {
                nonUnitPerm.push_back(srcDim);
            }
        }
        if (nonUnitPerm.size() <= 1) {
            return true;
        }
        // Argsort nonUnitPerm: output positions listed in source-dim order.
        const auto n = static_cast<int64_t>(nonUnitPerm.size());
        SmallVector<int64_t> outPosBySrcDim(n);
        std::iota(outPosBySrcDim.begin(), outPosBySrcDim.end(), 0);
        llvm::sort(outPosBySrcDim, [&nonUnitPerm](int64_t a, int64_t b) {
            return nonUnitPerm[a] < nonUnitPerm[b];
        });
        // Group runs of consecutive output-positions (in source-dim order).
        // Each run is a "super-dim"; record the starting output-position of each group.
        SmallVector<int64_t> groupStartOutPos;
        for (int64_t i = 0; i < n; ++i) {
            if (i == 0 || outPosBySrcDim[i - 1] + 1 != outPosBySrcDim[i]) {
                groupStartOutPos.push_back(outPosBySrcDim[i]);
            }
        }
        // Only the HxWxC (3-group) shape maps to the supported {1,0,2} permutation.
        if (groupStartOutPos.size() != 3) {
            return false;
        }
        // {1,0,2} means group-1 < group-0 < group-2 in output-position order.
        return groupStartOutPos[1] < groupStartOutPos[0] && groupStartOutPos[0] < groupStartOutPos[2];
    };
    if (!isSubBytePermuteSupported(dynamicDequantizeInType, outputPermuteOp.getMemPerm())) {
        _log.trace("Sub-byte permutation would produce an unsupported DMA pattern, skipping transformation");
        return mlir::failure();
    }

    // Add permutes to inputs
    auto index{0};  // Initialize a counter for unique identifiers
    for (auto& inputOperand : origOp->getOpOperands()) {
        auto inMemPermuteOp = IE::MemPermuteOp::create(
                rewriter, appendLoc(origOp->getLoc(), "input_permute" + std::to_string(index)), inputOperand.get(),
                outputPermuteOp.getDstOrder(), outputPermuteOp.getMemPerm());
        rewriter.replaceUsesWithIf(inputOperand.get(), inMemPermuteOp->getResult(0), [&](mlir::OpOperand& operand) {
            return operand.getOwner() == origOp;
        });
        index++;
    }

    rewriter.modifyOpInPlace(origOp, [&] {
        vpux::inferReturnTypes(origOp, vpux::InferShapedTypeMode::ALL);
    });

    rewriter.replaceAllUsesWith(outputPermuteOp.getOutput(), origOp.getOutput());
    rewriter.eraseOp(outputPermuteOp);

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
    auto func = getOperation();
    auto arch = config::getArch(func);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AdjustForEltwise>(&ctx, _log);
    patterns.add<AdjustForTile>(&ctx, _log);
    patterns.add<AdjustForConvert>(&ctx, _log);
    patterns.add<AdjustForSoftmax>(&ctx, _log);
    patterns.add<AdjustForConcat>(&ctx, _log);
    patterns.add<FusePermuteCastAroundEltwise<IE::AddOp>>(&ctx, _log);
    // NCE.Eltwise only supports Multiply starting from NPU50XX.
    if (arch >= config::ArchKind::NPU50XX) {
        patterns.add<FusePermuteCastAroundEltwise<IE::MultiplyOp>>(&ctx, _log);
    }

    // E#208387 NPU37XX doesn't expect sub-byte storage type
    if (arch >= config::ArchKind::NPU40XX) {
        patterns.add<AdjustForDynamicDequantize>(&ctx, _log);
    }
    patterns.add<AdjustForNCEEltwise<IE::AddOp>>(&ctx, _log);
    IE::MemPermuteOp::getCanonicalizationPatterns(patterns, &ctx);

    // Raise the greedy iteration cap above the shared default. Large models can require more rewrite sweeps to
    // reach a fixpoint than the default allows, causing the pass to fail even though the sequence terminates.
    // The greedy driver stops as soon as it converges, so a larger cap is free for models that converge early
    // and only lets slow-but-terminating cases finish instead of failing prematurely.
    auto config = getDefaultGreedyRewriteConfig();
    config.setMaxIterations(config.getMaxIterations() * 10);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), config))) {
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
