//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <map>

namespace vpux::IE {
#define GEN_PASS_DECL_LEGALIZENDMEMPERMUTE
#define GEN_PASS_DEF_LEGALIZENDMEMPERMUTE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

const int64_t LEGAL_RANK = 4;

/*
Decompose complex permutation (more than one dimension is moved) into several simple permutations.

Take the complex permutation [0, 1, 2, 3, 4, 5, 6, 7, 8] -> [0, 3, 2, 6, 4, 1, 5, 7, 8] as an example, it will be
decomposed as below:

Step 1 - move dim [1] after dim [4]
    : [0, 1, 2, 3, 4, 5, 6, 7, 8] -> [0, 2, 3, 4, 1, 5, 6, 7, 8]
Step 2 - move dim [2] after dim [3]
    : [0, 2, 3, 4, 1, 5, 6, 7, 8] -> [0, 3, 2, 4, 1, 5, 6, 7, 8]
Step 3 - move dim [4, 1, 5] after dim [6]
    : [0, 3, 2, 4, 1, 5, 6, 7, 8] -> [0, 3, 2, 6, 4, 1, 5, 7, 8]

With the 3 steps, we can finally get the original [0, 1, 2, 3, 4, 5, 6, 7, 8] -> [0, 3, 2, 6, 4, 1, 5, 7, 8]
permutation.
*/
SmallVector<mlir::AffineMap> decomposeNDPermutation(ArrayRef<uint32_t> targetPerm, mlir::MLIRContext* ctx, Logger log) {
    const auto initialMap = mlir::AffineMap::getMultiDimIdentityMap(targetPerm.size(), ctx);
    auto currentOrder = to_small_vector(DimsOrder::fromAffineMap(initialMap).toPermutation() | transformed([](Dim dim) {
                                            return checked_cast<uint32_t>(dim.ind());
                                        }));

    SmallVector<SmallVector<uint32_t>> decomposedOrders;

    while (currentOrder != targetPerm) {
        SmallVector<uint32_t> tempOrder = currentOrder;
        SmallVector<uint32_t> dimsTobeMoved;

        size_t moveFromPos = 0;
        uint32_t prevDim = 0;

        // 1.find the dim(s) to be moved
        for (size_t i = 0; i < currentOrder.size(); i++) {
            if (currentOrder[i] == targetPerm[i]) {
                continue;
            }

            // find the first dim to be moved
            moveFromPos = i;

            auto targetIt = std::find(targetPerm.begin(), targetPerm.end(), currentOrder[i]);
            VPUX_THROW_WHEN(targetIt == targetPerm.end(), "'{0}' is not found in targetPerm '{1}'", currentOrder[i],
                            targetPerm);
            auto pos = targetIt - targetPerm.begin();
            prevDim = *(targetIt - 1);

            // try to find more dims that can be moved together
            for (size_t j = moveFromPos; j < currentOrder.size(); j++) {
                if (pos == checked_cast<int32_t>(targetPerm.size()) || currentOrder[j] != targetPerm[pos]) {
                    break;
                }

                dimsTobeMoved.push_back(currentOrder[j]);
                pos++;
            }
            break;
        }

        // 2.erase dims index from src position
        if (dimsTobeMoved.size() > 1) {
            tempOrder.erase(tempOrder.begin() + moveFromPos, tempOrder.begin() + moveFromPos + dimsTobeMoved.size());
        } else {
            tempOrder.erase(tempOrder.begin() + moveFromPos);
        }

        // 3.find the dst position and append or insert dims index
        auto moveAfterIter = std::find(tempOrder.begin(), tempOrder.end(), prevDim);
        VPUX_THROW_WHEN(moveAfterIter == tempOrder.end(), "'{0}' is not found in tempOrder '{1}'", prevDim, tempOrder);
        if (moveAfterIter == tempOrder.end() - 1) {
            tempOrder.append(dimsTobeMoved.begin(), dimsTobeMoved.end());
        } else {
            tempOrder.insert(moveAfterIter + 1, dimsTobeMoved.begin(), dimsTobeMoved.end());
        }

        decomposedOrders.push_back(tempOrder);
        currentOrder.assign(tempOrder);
    }

    log.trace("Decomposed permutation {0} to orders {1}", targetPerm, decomposedOrders);

    SmallVector<mlir::AffineMap> decomposedPermMaps;
    for (size_t i = 0; i < decomposedOrders.size(); i++) {
        DimsOrder inOrder =
                (i == 0) ? DimsOrder::fromAffineMap(initialMap)
                         : DimsOrder::fromAffineMap(mlir::AffineMap::getPermutationMap(decomposedOrders[i - 1], ctx));
        DimsOrder outOrder = DimsOrder::fromAffineMap(mlir::AffineMap::getPermutationMap(decomposedOrders[i], ctx));
        auto perm = getPermutationFromOrders(inOrder, outOrder, ctx);
        decomposedPermMaps.push_back(perm);
    }

    log.trace("Decomposed permutation {0} to {1}", targetPerm, decomposedPermMaps);

    return decomposedPermMaps;
}

class LegalizeNDMemPermute final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    LegalizeNDMemPermute(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("LegalizeNDMemPermute");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LegalizeNDMemPermute::matchAndRewrite(IE::MemPermuteOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    // Only enabled for NPU37XX, NPU40XX
    // where Tiling for SW kernels is limited to 4D ops.
    // ToDo: Remove pass after limitation.

    auto inputType = origOp.getInput().getType().cast<NDTypeInterface>();

    // do not apply pass for dynamic shapes
    if (inputType.getShape().isDynamic()) {
        return mlir::failure();
    }

    if (inputType.getRank() == LEGAL_RANK) {
        return mlir::failure();
    }

    auto outputType = origOp.getOutput().getType().cast<NDTypeInterface>();
    auto permutation = origOp.getMemPerm();

    _log.trace("Got MemPermute op at '{0}' with shape '{1}' and permutation map '{2}'", origOp->getLoc(),
               inputType.getShape(), permutation);

    auto ctx = rewriter.getContext();

    auto mergedPermAndShape = vpux::getMergedPermutationAndShape(inputType, permutation, LEGAL_RANK);
    auto mergedPermutation = mergedPermAndShape.first;
    auto mergedShape = mergedPermAndShape.second;

    SmallVector<mlir::AffineMap> decomposedPermMaps;
    if (mergedPermutation.size() > LEGAL_RANK) {
        _log.trace("Decompose rank '{0}' MemPermute: perm {1}", mergedPermutation.size(), mergedPermutation);
        decomposedPermMaps = decomposeNDPermutation(mergedPermutation, ctx, _log);
    }

    extendPermutationAndShape(mergedPermutation, mergedShape, LEGAL_RANK);
    auto reducedPermutation = mlir::AffineMap::getPermutationMap(ArrayRef(mergedPermutation), ctx);

    // Cast to canonical order for convenience
    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<uint32_t>(inputType.getRank()), ctx);
    auto inputCast = rewriter.create<IE::PermuteCastOp>(origOp.getLoc(), origOp.getInput(), identityMap, identityMap);

    // Build input reshape operation
    auto reducedShapeAttr = getIntArrayAttr(ctx, mergedShape);
    auto inputReshape = rewriter.create<IE::ReshapeOp>(origOp.getLoc(), inputCast.getOutput(), /*shape=*/nullptr,
                                                       /*special_zero=*/nullptr, reducedShapeAttr);

    // Build reduced permutation operation
    mlir::Value permuteInput = inputReshape.getOutput();
    mlir::Value permuteOutput;
    if (decomposedPermMaps.empty()) {
        permuteOutput = rewriter.create<IE::MemPermuteOp>(origOp.getLoc(), permuteInput,
                                                          mlir::AffineMap::getMultiDimIdentityMap(
                                                                  checked_cast<uint32_t>(mergedShape.size()), ctx),
                                                          reducedPermutation)
                                .getOutput();
    } else {
        // decompose permutation
        for (auto perm : decomposedPermMaps) {
            permuteOutput = rewriter.create<IE::MemPermuteOp>(origOp.getLoc(), permuteInput,
                                                              mlir::AffineMap::getMultiDimIdentityMap(
                                                                      checked_cast<uint32_t>(mergedShape.size()), ctx),
                                                              perm)
                                    .getOutput();

            permuteInput = permuteOutput;
        }
    }

    // Reshape to original output shape
    auto outputShape = outputType.getMemShape();
    auto outputShapeAttr = getIntArrayAttr(ctx, outputShape);
    auto outputReshape = rewriter.create<IE::ReshapeOp>(origOp.getLoc(), permuteOutput, /*shape=*/nullptr,
                                                        /*special_zero=*/nullptr, outputShapeAttr);

    // Set destination order
    auto dstOrder = origOp.getDstOrder();
    auto newSequence = rewriter.create<IE::PermuteCastOp>(
            origOp.getLoc(), outputReshape.getOutput(), dstOrder,
            mlir::AffineMap::getMultiDimIdentityMap(checked_cast<uint32_t>(outputShape.size()), ctx));

    rewriter.replaceOp(origOp, newSequence.getOutput());

    _log.nest().trace("Replaced with shape '{0}' and permutation map '{1}'", mergedShape, reducedPermutation);
    return mlir::success();
}

//
// LegalizeNDMemPermutePass
//

class LegalizeNDMemPermutePass final : public IE::impl::LegalizeNDMemPermuteBase<LegalizeNDMemPermutePass> {
public:
    explicit LegalizeNDMemPermutePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void LegalizeNDMemPermutePass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LegalizeNDMemPermute>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToMemPermutePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLegalizeNDMemPermutePass(Logger log) {
    return std::make_unique<LegalizeNDMemPermutePass>(log);
}
