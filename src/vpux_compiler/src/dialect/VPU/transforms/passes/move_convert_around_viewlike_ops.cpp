//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_MOVECONVERTAROUNDVIEWLIKEOPS
#define GEN_PASS_DEF_MOVECONVERTAROUNDVIEWLIKEOPS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

template <class ViewLikeOp>
class MoveConvertAfterOperation : public mlir::OpRewritePattern<ViewLikeOp> {
public:
    MoveConvertAfterOperation(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ViewLikeOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ViewLikeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ViewLikeOp>
mlir::LogicalResult MoveConvertAfterOperation<ViewLikeOp>::matchAndRewrite(ViewLikeOp originOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", originOp->getName(), originOp->getLoc());
    auto convertOp = originOp->getOperand(0).template getDefiningOp<VPU::ConvertOp>();
    if (convertOp == nullptr) {
        return matchFailed(_log, rewriter, originOp, "Did not find input to be ConvertOp at '{0}'", originOp->getLoc());
    }

    if (!convertOp->hasOneUse()) {
        return matchFailed(_log, rewriter, originOp, "ConvertOp has more than 1 users at '{0}'", convertOp->getLoc());
    }

    if (!isConvertSupportedOnDMA<VPU::ConvertOp>(convertOp)) {
        return matchFailed(_log, rewriter, originOp,
                           "ConvertOp not supported on DMA only FP32->BF16/F16 is supported at '{0}'",
                           originOp->getLoc());
    }

    // Move ConvertOp after ViewLikeOp so we can later fuse Copy and convertDMAOp
    auto newViewLikeOp = rewriter.clone(*originOp);
    auto result = newViewLikeOp->getResult(0);
    newViewLikeOp->setOperand(0, convertOp.getInput());

    auto newOpResultType = mlir::cast<vpux::NDTypeInterface>(result.getType());
    auto inputType = mlir::cast<vpux::NDTypeInterface>(convertOp.getInput().getType());
    result.setType(newOpResultType.changeElemType(inputType.getElementType()));

    auto originOpType = mlir::cast<vpux::NDTypeInterface>(originOp->getResult(0).getType());
    auto newConvert = rewriter.replaceOpWithNewOp<VPU::ConvertOp>(originOp, result, convertOp.getDstElemTypeAttr());
    newConvert->getResult(0).setType(originOpType);

    return mlir::success();
}

//
// MoveConvertBeforeAffineReshape
//
// Move the ConvertOp before AffineReshape
// AffineReshape              ConvertOp
//      |                         |
//   ConvertOp       ->      Affine Reshape
class MoveConvertBeforeAffineReshape final : public mlir::OpRewritePattern<VPU::ConvertOp> {
public:
    MoveConvertBeforeAffineReshape(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::ConvertOp>(ctx), _log(log) {
        this->setDebugName("MoveConvertAroundViewLikeOpsPass::MoveConvertBeforeAffineReshape");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::ConvertOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveConvertBeforeAffineReshape::matchAndRewrite(VPU::ConvertOp originOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", originOp->getName(), originOp->getLoc());
    auto nestedLogger = _log.nest();
    auto affineReshapeOp = originOp.getInput().getDefiningOp<VPU::AffineReshapeOp>();
    if (affineReshapeOp == nullptr) {
        nestedLogger.trace("ConvertOp does not have AffineReshape input {0}", originOp->getName());
        return mlir::failure();
    }

    auto affineReshapeOutputType = mlir::cast<vpux::NDTypeInterface>(affineReshapeOp.getOutput().getType());
    if (affineReshapeOutputType.getShape().size() == 4) {
        nestedLogger.trace("AffineReshape output is already 4D {0}", affineReshapeOp->getName());
        return mlir::failure();
    }

    // If the AffineReshape input is not 4D then this movement is useless
    auto affineReshapeInputType = mlir::cast<vpux::NDTypeInterface>(affineReshapeOp.getInput().getType());
    if (affineReshapeInputType.getShape().size() != 4) {
        nestedLogger.trace("AffineReshape input is not 4D {0}", affineReshapeOp->getName());
        return mlir::failure();
    }
    auto originOpType = mlir::cast<vpux::NDTypeInterface>(originOp->getResult(0).getType());
    auto newConvertOp = VPU::ConvertOp::create(rewriter, originOp.getLoc(), affineReshapeOp.getInput(),
                                               originOp.getDstElemTypeAttr());
    auto newAffineReshape = rewriter.replaceOpWithNewOp<VPU::AffineReshapeOp>(originOp, newConvertOp->getResult(0),
                                                                              affineReshapeOp.getDimMappingAttr(),
                                                                              affineReshapeOp.getShapeValueAttr());

    newAffineReshape->getResult(0).setType(originOpType);

    return mlir::success();
}

//
// MergeConvertBeforeSliceBranches
//
// Several branches slice a 1D tensor (produced by an AffineReshape from a 4D tensor), reshape each
// slice back to 4D and convert it to a different element type, e.g.:
//
//                4D root                     4D root
//                   |                            |
//             AffineReshape                  Convert
//              /    |    \.       ->             |
//          Slice  Slice  Slice              AffineReshape
//            |      |      |                 /    |    \.
//         Reshape Reshape Reshape         Slice  Slice  Slice
//            |      |      |                 |      |      |
//         Convert Convert Convert         Reshape Reshape Reshape
//
// The Reshape between Slice and Convert is optional on a per-chain basis. A single Slice may also
// feed more than one such Reshape(-optional)->Convert chain (e.g. the same slice converted to the
// same element type for multiple consumers); every chain hanging off a given Slice is rebuilt, and
// the Slice contributes its output size once per chain towards the coverage check below, since the
// original IR converted that slice's data once per chain.
//
// Each individual Convert may be too small to benefit from multi-shave tiling. Moving a single
// merged Convert before the AffineReshape lets it run on the full 4D root tensor instead, which:
//   1. Keeps the 4D shape of the root so the merged Convert can still be multi-clustered.
//   2. Does not increase the total amount of converted data: the merged Convert always processes
//      exactly the root/AffineReshape output size, so this only pays off when the original
//      per-branch Converts already summed to at least that many elements (slices may freely
//      overlap or leave gaps, since each branch still reads the same elementwise-converted values
//      it would have from its own Convert).
//   3. Is only applied when it is actually beneficial: the merged Convert must become eligible for
//      multi-shave execution, regardless of whether the individual per-branch Converts already
//      were.
class MergeConvertBeforeSliceBranches final : public mlir::OpRewritePattern<VPU::AffineReshapeOp> {
public:
    MergeConvertBeforeSliceBranches(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::AffineReshapeOp>(ctx), _log(log) {
        this->setDebugName("MoveConvertAroundViewLikeOpsPass::MergeConvertBeforeSliceBranches");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::AffineReshapeOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MergeConvertBeforeSliceBranches::matchAndRewrite(VPU::AffineReshapeOp originOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", originOp->getName(), originOp->getLoc());
    auto nestedLogger = _log.nest();

    if (originOp->use_empty()) {
        return mlir::failure();
    }

    // The new Convert is placed before the AffineReshape, so its input must stay 4D to keep the
    // ability to multi-cluster the Convert.
    auto rootType = mlir::cast<vpux::NDTypeInterface>(originOp.getInput().getType());
    if (rootType.getShape().size() != 4) {
        return matchFailed(nestedLogger, rewriter, originOp, "AffineReshape input is not 4D at '{0}'",
                           originOp->getLoc());
    }

    auto affineReshapeOutType = mlir::cast<vpux::NDTypeInterface>(originOp.getOutput().getType());

    // The coverage/size checks below rely on totalSize(), which throws on dynamic shapes. Bail out
    // early when the root or the AffineReshape output cannot be reduced to a static bound.
    if (getBoundedShape(originOp.getInput()).isDynamic() || getBoundedShape(originOp.getOutput()).isDynamic()) {
        return matchFailed(nestedLogger, rewriter, originOp, "AffineReshape has a dynamic (unbounded) shape at '{0}'",
                           originOp->getLoc());
    }

    struct Chain {
        // ReshapeOp is optional: some chains may feed the Slice output directly into Convert
        // without an intermediate reshape to 4D.
        VPU::ReshapeOp reshapeOp;
        VPU::ConvertOp convertOp;
    };
    struct Branch {
        VPU::SliceOp sliceOp;
        // A single Slice may feed more than one Reshape(-optional)->Convert chain.
        SmallVector<Chain> chains;
    };
    SmallVector<Branch> branches;

    mlir::Type dstElemType = nullptr;
    for (auto* user : originOp.getOutput().getUsers()) {
        auto sliceOp = mlir::dyn_cast<VPU::SliceOp>(user);
        if (sliceOp == nullptr) {
            return matchFailed(nestedLogger, rewriter, originOp, "Not all users of AffineReshape are SliceOp at '{0}'",
                               originOp->getLoc());
        }

        if (getBoundedShape(sliceOp.getOutput()).isDynamic()) {
            return matchFailed(nestedLogger, rewriter, originOp, "Slice has a dynamic (unbounded) shape at '{0}'",
                               sliceOp->getLoc());
        }

        Branch branch{sliceOp, {}};
        for (auto* sliceUser : sliceOp->getUsers()) {
            // The Reshape between Slice and Convert is optional: it only reshapes the 1D slice to
            // the shape expected by Convert/downstream ops and does not affect how many elements
            // are converted, so its absence does not change the legality of the merge.
            VPU::ReshapeOp reshapeOp = mlir::dyn_cast<VPU::ReshapeOp>(sliceUser);
            mlir::Operation* preConvertOp = sliceUser;
            if (reshapeOp != nullptr) {
                if (!reshapeOp->hasOneUse()) {
                    return matchFailed(nestedLogger, rewriter, originOp, "Reshape has more than 1 users at '{0}'",
                                       reshapeOp->getLoc());
                }
                preConvertOp = *reshapeOp->getUsers().begin();
            }

            auto convertOp = mlir::dyn_cast<VPU::ConvertOp>(preConvertOp);
            if (convertOp == nullptr || !convertOp->hasOneUse()) {
                return matchFailed(nestedLogger, rewriter, originOp,
                                   "Slice/Reshape user is not a single-use ConvertOp at '{0}'", preConvertOp->getLoc());
            }

            if (dstElemType == nullptr) {
                dstElemType = convertOp.getDstElemType();
            } else if (dstElemType != convertOp.getDstElemType()) {
                return matchFailed(nestedLogger, rewriter, originOp,
                                   "ConvertOp branches have different destination types at '{0}'", convertOp->getLoc());
            }

            branch.chains.push_back(Chain{reshapeOp, convertOp});
        }

        // Reject Slice branches that do not contribute any eligible Reshape?->Convert chain.
        // Keep matching rules simple and require every Slice branch to be meaningful.
        if (branch.chains.empty()) {
            return matchFailed(nestedLogger, rewriter, originOp, "Slice branch has no eligible convert chains at '{0}'",
                               sliceOp->getLoc());
        }

        branches.push_back(std::move(branch));
    }

    if (branches.size() < 8) {
        return matchFailed(nestedLogger, rewriter, originOp, "Need at least 8 branches to merge at '{0}'",
                           originOp->getLoc());
    }

    // See point 2 above: each branch's contribution is scaled by its number of chains, since the
    // original IR converted that slice's data once per chain.
    int64_t totalSlicedSize = 0;
    for (auto& branch : branches) {
        totalSlicedSize +=
                getBoundedShape(branch.sliceOp.getOutput()).totalSize() * static_cast<int64_t>(branch.chains.size());
    }

    if (totalSlicedSize < getBoundedShape(originOp.getOutput()).totalSize()) {
        return matchFailed(nestedLogger, rewriter, originOp,
                           "Merged ConvertOp would process more data than the original branches at '{0}'",
                           originOp->getLoc());
    }

    // The merged Convert must actually unlock multi-shave execution, otherwise there is nothing to
    // gain from the transformation.
    if (!VPU::shouldConvertUseMultiShaves(rootType, rootType.changeElemType(dstElemType))) {
        return matchFailed(nestedLogger, rewriter, originOp,
                           "Merged ConvertOp would not benefit from multi-shave execution at '{0}'",
                           originOp->getLoc());
    }

    // Move the Convert before the AffineReshape so it operates on the full 4D root tensor.
    auto dstElemTypeAttr = mlir::TypeAttr::get(dstElemType);
    auto newConvertOp = VPU::ConvertOp::create(rewriter, originOp.getLoc(), originOp.getInput(), dstElemTypeAttr);
    newConvertOp.getOutput().setType(rootType.changeElemType(dstElemType));

    auto newAffineReshapeOp = VPU::AffineReshapeOp::create(
            rewriter, originOp.getLoc(), affineReshapeOutType.changeElemType(dstElemType), newConvertOp.getOutput(),
            originOp.getDimMappingAttr(), originOp.getShapeValueAttr());

    for (auto& branch : branches) {
        auto sliceOutType = mlir::cast<vpux::NDTypeInterface>(branch.sliceOp.getOutput().getType());
        auto newSliceOp =
                VPU::SliceOp::create(rewriter, branch.sliceOp.getLoc(), sliceOutType.changeElemType(dstElemType),
                                     newAffineReshapeOp.getOutput(), branch.sliceOp.getStaticOffsetsAttr(),
                                     branch.sliceOp.getStaticSizesAttr());

        // Rebuild every chain that used to hang off this Slice: each one gets its own (optional)
        // Reshape recreated from the new, already-converted Slice output.
        for (auto& chain : branch.chains) {
            mlir::Value newConvertInput = newSliceOp.getOutput();
            if (chain.reshapeOp != nullptr) {
                auto reshapeOutType = mlir::cast<vpux::NDTypeInterface>(chain.reshapeOp.getOutput().getType());
                auto newReshapeOp = VPU::ReshapeOp::create(rewriter, chain.reshapeOp.getLoc(),
                                                           reshapeOutType.changeElemType(dstElemType),
                                                           newSliceOp.getOutput(), chain.reshapeOp.getShapeValueAttr());
                newConvertInput = newReshapeOp.getOutput();
            }

            rewriter.replaceOp(chain.convertOp, newConvertInput);
            if (chain.reshapeOp != nullptr) {
                rewriter.eraseOp(chain.reshapeOp);
            }
        }

        rewriter.eraseOp(branch.sliceOp);
    }

    rewriter.eraseOp(originOp);

    return mlir::success();
}

//
// MoveConvertAroundViewLikeOpsPass
//
class MoveConvertAroundViewLikeOpsPass final :
        public VPU::impl::MoveConvertAroundViewLikeOpsBase<MoveConvertAroundViewLikeOpsPass> {
public:
    explicit MoveConvertAroundViewLikeOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MoveConvertAroundViewLikeOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<MoveConvertAfterOperation<VPU::PermuteCastOp>>(&ctx, _log.nest());
        patterns.add<MoveConvertAfterOperation<VPU::ShapeCastOp>>(&ctx, _log.nest());
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<MoveConvertBeforeAffineReshape>(&ctx, _log.nest());
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<MergeConvertBeforeSliceBranches>(&ctx, _log.nest());
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }
}

}  // namespace

//
// createMoveConvertAroundViewLikeOpsPass
//
std::unique_ptr<mlir::Pass> vpux::VPU::createMoveConvertAroundViewLikeOpsPass(Logger log) {
    return std::make_unique<MoveConvertAroundViewLikeOpsPass>(log);
}
